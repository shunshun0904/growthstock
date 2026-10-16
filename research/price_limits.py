#!/usr/bin/env python3
"""
値幅制限（東証の制限値幅）から、日足の高値・終値・安値が「ストップ高・ストップ安」に達したかを判定する。

なぜ自前で判定するか
--------------------
J-Quants V2 の株価四本値（/equities/bars/daily）には、その日にストップ高・ストップ安に達したかの
フラグ UL / LL がある（実測: research/probe_endpoints.json の keys。V1 の UpperLimit / LowerLimit）。
だが取り込み（research/jq_bulk.py の BAR_COLS）はこの列を保存していないので、Release の bars_*.parquet
にはフラグが無い。取り直すと全期間で約2.5時間かかる（docs/MODEL_DESIGN.md「生データの保存と差分取得」）。
まずは保存済みの四本値と制限値幅の表から判定し、API のフラグとの一致は research/probe_limit_flags.py で測る。

判定の約束
----------
  基準値段  前営業日の終値。売買の無い日（四本値が欠測）が挟まれば、その前の最後の終値。
            分割・併合をまたぐ日は、分割調整の倍率（AdjC ÷ C）の前日比で直す（1:2 の分割なら半分）。
            実際の取引所の基準値段は「前日の終値、終値が無ければ最終気配」で、気配が基準になった日は
            ここで出す基準値段より数円ずれる（上の probe で実測する）
  制限値幅  基準値段の帯ごとの固定額（下の TABLE）。基準値段 1,000円以上 1,500円未満なら 300円、など
  ストップ高  高値 ≥ 基準値段 + 制限値幅（ul）。ちょうどでなく「以上」にしてあるのは、最終気配が基準値段に
            なった日（計算より数円上が上限）と、制限値幅が4倍に拡大された日（2営業日続けてストップ高で
            売買が成立しない等）も拾うため。実測（2023〜2026 の日足 387万行）では、高値が計算した上限と
            ちょうど一致する日 9,303 に対し、上限を超える日は 854（全行の 0.02%）で、その大半は
            上限の +2% 以内（気配が基準値段になった日）、残りが 1.5〜4倍（拡大の日）
  張り付き  終値 ≥ 基準値段 + 制限値幅（ul_close。引けまでストップ高のまま）
  ストップ安  安値 ≤ 基準値段 − 制限値幅（ll）
  ちょうど  高値 = 基準値段 + 制限値幅（ul_exact。確かめ用）
  known     判定できた行（高値と基準値段がある）。売買の無い日・上場初日は判定できない（False のまま）
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 東証の制限値幅（基準値段の上限（この値未満）, 制限値幅）。円。
#: 100円未満 30円 … 1,000円以上 1,500円未満 300円 … 7,000円以上 10,000円未満 1,500円 … 5,000万円以上 1,000万円
TABLE: Tuple[Tuple[float, float], ...] = (
    (100, 30), (200, 50), (500, 80), (700, 100), (1000, 150),
    (1500, 300), (2000, 400), (3000, 500), (5000, 700), (7000, 1000), (10000, 1500),
    (15000, 3000), (20000, 4000), (30000, 5000), (50000, 7000), (70000, 10000), (100000, 15000),
    (150000, 30000), (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000), (1000000, 150000),
    (1500000, 300000), (2000000, 400000), (3000000, 500000), (5000000, 700000), (7000000, 1000000),
    (10000000, 1500000), (15000000, 3000000), (20000000, 4000000), (30000000, 5000000), (50000000, 7000000),
    (float("inf"), 10000000),
)
_UPPER = np.array([u for u, _ in TABLE], dtype=float)
_WIDTH = np.array([w for _, w in TABLE], dtype=float)

#: 値段の比べの許容（円）。呼値は最小 0.1円なので、浮動小数点の誤差だけを吸収する
TOL = 1e-6
#: 分割調整の倍率の前日比がこれ未満しか 1 とずれなければ「分割なし」とみなす（AdjC ÷ C の丸めの揺れを吸収）
SPLIT_TOL = 1e-3


def width(base) -> np.ndarray:
    """基準値段ごとの制限値幅（円）。基準値段が欠測なら NaN。"""
    b = np.asarray(base, dtype=float)
    idx = np.searchsorted(_UPPER, b, side="right")           # base < _UPPER[idx]
    w = _WIDTH[np.minimum(idx, len(_WIDTH) - 1)]
    return np.where(np.isfinite(b) & (b >= 0), w, np.nan)


def flag_limits(bars: pd.DataFrame, tol: float = TOL) -> pd.DataFrame:
    """
    日足（Date, Code, H, L, C, AdjC。未調整の四本値と、分割調整後の終値）に、基準値段・制限値幅・判定の列を付けて返す。

    列: Date, Code, base, width, known, ul, ul_close, ll, ul_exact（判定の約束はモジュールの docstring）。
    判定できない行（known=False）は ul などが False のまま。
    """
    need = ["Date", "Code", "H", "L", "C", "AdjC"]
    missing = [c for c in need if c not in bars.columns]
    if missing:
        raise KeyError(f"日足に列が無い: {missing}")
    b = bars[need].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    for c in ("H", "L", "C", "AdjC"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    code = b["Code"]
    adj = (b["AdjC"] / b["C"]).where(b["C"] > 0)
    prev_c = b["C"].groupby(code).ffill().groupby(code).shift(1)
    prev_adj = adj.groupby(code).ffill().groupby(code).shift(1)
    ratio = prev_adj / adj
    ratio = ratio.where((ratio - 1.0).abs() >= SPLIT_TOL, 1.0)   # 欠測・丸めの揺れは「分割なし」
    base = (prev_c * ratio).round(1)
    w = pd.Series(width(base.to_numpy()), index=b.index)
    up = base + w
    dn = base - w
    known = b["H"].notna() & base.notna()
    out = b[["Date", "Code"]].copy()
    out["base"] = base
    out["width"] = w
    out["known"] = known
    out["ul"] = known & (b["H"] >= up - tol)
    out["ul_close"] = known & b["C"].notna() & (b["C"] >= up - tol)
    out["ll"] = known & b["L"].notna() & (b["L"] <= dn + tol)
    out["ul_exact"] = known & ((b["H"] - up).abs() <= tol)
    return out


def flags_summary(flags: pd.DataFrame) -> Dict:
    """判定の集計（件数と割合だけ）。"""
    n = int(len(flags))
    known = flags["known"].to_numpy(dtype=bool)
    ul = flags["ul"].to_numpy(dtype=bool)
    nk = int(known.sum())
    out = {"rows": n, "known": nk, "known_share": nk / n * 100 if n else float("nan"),
           "ul": int(ul.sum()), "ul_close": int(flags["ul_close"].sum()), "ll": int(flags["ll"].sum()),
           "ul_exact": int(flags["ul_exact"].sum())}
    out["ul_share"] = out["ul"] / nk * 100 if nk else float("nan")
    out["exact_among_ul"] = out["ul_exact"] / out["ul"] * 100 if out["ul"] else float("nan")
    return out


def code_arrays(flags: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """銘柄ごとの (日付, known, ul, ul_close, ll) の配列。limit_events と day_flag が使う。"""
    f = flags.sort_values(["Code", "Date"], kind="mergesort")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c, g in f.groupby("Code", sort=False):
        out[str(c)] = {"dates": g["Date"].to_numpy(dtype="datetime64[ns]"),
                       "known": g["known"].to_numpy(dtype=bool),
                       "ul": g["ul"].to_numpy(dtype=bool),
                       "ul_close": g["ul_close"].to_numpy(dtype=bool),
                       "ll": g["ll"].to_numpy(dtype=bool)}
    return out


def _locate(arrays: Dict, code: str, day) -> Tuple[Optional[Dict], int]:
    a = arrays.get(str(code))
    if a is None:
        return None, -1
    d = np.datetime64(pd.Timestamp(day), "ns")
    i = int(np.searchsorted(a["dates"], d))
    if i >= len(a["dates"]) or a["dates"][i] != d:
        return a, -1
    return a, i


def limit_events(arrays: Dict, keys: pd.DataFrame, hold: int = 20, past: int = 250) -> pd.DataFrame:
    """
    (Code, Date) ごとに、基準日 Date のあとの hold 営業日（翌営業日 = 1日目）と、前の past 営業日のストップ高を数える。

    列（NaN = 判定できない）:
      ul_t        基準日そのものがストップ高か（1/0）
      fwd_ok      あとの hold 営業日の日足がそろっているか（そろわなければ fwd の列は NaN）
      ul_fwd      翌営業日から hold 営業日目までにストップ高の日があるか（1/0）、ul_fwd_n その日数、ul_fwd_first 最初の日（1 = 翌営業日）
      ulc_fwd     同じ期間に張り付き（終値がストップ高）の日があるか、ulc_fwd_n その日数
      ll_fwd      同じ期間にストップ安の日があるか
      fwd_known   同じ期間で判定できた日の数
      ul_past_n   基準日の前 past 営業日のストップ高の日数、past_bars その期間にあった日足の数、ul_past20_n 前20営業日の日数
    """
    need = keys[["Code", "Date"]].drop_duplicates()
    rows = []
    nan = float("nan")
    for code, day in need.itertuples(index=False):
        rec = {"Code": str(code), "Date": pd.Timestamp(day), "ul_t": nan, "fwd_ok": 0.0, "ul_fwd": nan, "ul_fwd_n": nan,
               "ul_fwd_first": nan, "ulc_fwd": nan, "ulc_fwd_n": nan, "ll_fwd": nan, "fwd_known": nan,
               "ul_past_n": nan, "past_bars": nan, "ul_past20_n": nan}
        a, i = _locate(arrays, code, day)
        if a is None or i < 0:
            rows.append(rec)
            continue
        n = len(a["dates"])
        rec["ul_t"] = float(a["ul"][i]) if a["known"][i] else nan
        if i + hold < n:
            sl = slice(i + 1, i + hold + 1)
            ul = a["ul"][sl]
            rec["fwd_ok"] = 1.0
            rec["fwd_known"] = float(a["known"][sl].sum())
            rec["ul_fwd"] = float(ul.any())
            rec["ul_fwd_n"] = float(ul.sum())
            first = np.flatnonzero(ul)
            rec["ul_fwd_first"] = float(first[0] + 1) if len(first) else nan
            ulc = a["ul_close"][sl]
            rec["ulc_fwd"] = float(ulc.any())
            rec["ulc_fwd_n"] = float(ulc.sum())
            rec["ll_fwd"] = float(a["ll"][sl].any())
        lo = max(0, i - past)
        rec["past_bars"] = float(i - lo)
        rec["ul_past_n"] = float(a["ul"][lo:i].sum())
        rec["ul_past20_n"] = float(a["ul"][max(0, i - 20):i].sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    k = keys[["Code", "Date"]].copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"])
    return k.merge(out, on=["Code", "Date"], how="left")


def day_flag(arrays: Dict, codes: Sequence, days: Sequence, offsets: Iterable, col: str = "ul") -> np.ndarray:
    """(Code, Date) から offset 営業日あとの日（0 = その日、1 = 翌営業日）の判定（1/0。判定できなければ NaN）。"""
    out = np.full(len(codes), np.nan)
    for k, (code, day, off) in enumerate(zip(codes, days, offsets)):
        if off is None or not np.isfinite(off):
            continue
        a, i = _locate(arrays, code, day)
        if a is None or i < 0:
            continue
        j = i + int(off)
        if 0 <= j < len(a["dates"]) and a["known"][j]:
            out[k] = float(a[col][j])
    return out
