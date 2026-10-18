#!/usr/bin/env python3
"""
相対位置の特徴量（実験76）: 主要な財務指標を「市場全体の中の百分位」と「同じ33業種の中の百分位」に直す。

運用者の問い（2026-10-11）「いまの特徴量に、同業種内での相対的な差（位置）がわかる特徴量は入っているか」
→ 株価・出来高の業種相対（rel_sector_20・vol_rel_sector）だけで、財務指標を業種の中で順位付けた列は無い。
続く指示「サンプル外の市場全体の中での相対位置も入れたい」→ 参照の母集団を学習データ（ブレイクした行。
日に数件〜数十件）ではなく **上場全銘柄** にする。学習データの行どうしで同じ日・同じ業種の順位を付けると
母集団が 1〜2件になって順位が定義できない（そのため既存の _r 列（同日順位）には財務の業種内順位が無い）。

作り
  参照（reference_panel）
    毎月末 M に、上場全銘柄（普通株だけ。ETF・REIT 等は build_dataset と同じ規則で除く）について
    「M までに開示済みの直近決算（STALE_DAYS 以内）」から METRICS の値を作り、M の終値（M から
    PRICE_TOLERANCE_DAYS 以内の最後のバー）で PER / PBR を作る（API の値を優先し、上限 PER_MAX / PBR_MAX。
    build_dataset と同じ）。33業種は master_hist の M 時点の S33。決算の展開は build_dataset.quarterize_panel
    そのもの（ROE の API 置き換えも含む）なので、行の側（データセットの列）と同じ定義の値どうしを比べる
  付ける（attach）
    行の日付 t より **前** の月末 M（M < t）の参照に対して、行の値の百分位（0〜1。参照の中で行の値より小さい
    ものの割合 + 同じ値の半分）を、市場全体（rel_mkt_<指標>）と同じ業種（rel_ind_<指標>）の2本で付ける。
    業種の母集団が MIN_INDUSTRY 未満の月末は業種内を欠測に、市場全体が MIN_MARKET 未満なら両方を欠測に。
    行の値が欠測なら欠測（0.5 で埋めない。「中位だった」という観測していない情報を与えない）

時点整合
  参照は M の終値と M までの開示だけで作り、M より後の日付の行にしか付けない。行の側の値は build_dataset が
  t の時点で作ったもの（t 以前の開示）。参照の分布が最大 1か月古いのは意図したもの（「毎月末に更新される
  一覧表と比べる」のと同じ）。

公開ログには件数・割合・日付だけを出す（銘柄ごとの値は出さない）。
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402

#: 相対位置に直す指標（データセットの列名。すべて本番の列（239列。2026-10-18 からは249列）に入っている）。
#: 成長率・利益率・ROE/ROA・財務の安全性・会社予想・バリュエーションの 10本
METRICS = ["sales_growth_sym_q0", "eps_growth_sym_q0", "op_margin_q0", "net_margin",
           "ROE_q0", "ROA_q0", "equity_ratio_q0", "guidance_op_growth", "per", "pbr"]
METRIC_JA = {"sales_growth_sym_q0": "売上成長率（対称）", "eps_growth_sym_q0": "EPS成長率（対称）",
             "op_margin_q0": "営業利益率", "net_margin": "純利益率", "ROE_q0": "ROE", "ROA_q0": "ROA",
             "equity_ratio_q0": "自己資本比率", "guidance_op_growth": "会社予想の営業利益成長率",
             "per": "PER", "pbr": "PBR"}
KINDS = ("ind", "mkt")
KIND_JA = {"ind": "同じ33業種の中", "mkt": "市場全体の中"}
#: 業種の母集団がこれより少ない月末は、業種内の百分位を欠測にする
MIN_INDUSTRY = 20
#: 市場全体の母集団がこれより少ない月末は、両方を欠測にする（データの端・取得漏れの月）
MIN_MARKET = 200
#: 決算がこれより古ければ参照に使わない（build_dataset の「1年以上古い決算は欠測扱い」と同じ）
STALE_DAYS = 365
#: 月末の終値は、月末からこの日数以内の最後のバー
PRICE_TOLERANCE_DAYS = 10


def columns(kind: str = "all", metrics: Iterable[str] = METRICS) -> list:
    """付ける列の名前。kind は ind / mkt / all（ind → mkt の順）。"""
    kinds = KINDS if kind == "all" else (kind,)
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise KeyError(f"未知の種類: {bad}. 利用可能: {KINDS + ('all',)}")
    return [f"rel_{k}_{m}" for k in kinds for m in metrics]


def month_ends(start, end) -> pd.DatetimeIndex:
    """start 以降 end 以前の暦の月末。"""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=pd.offsets.MonthEnd())


# --------------------------------------------------------------------------- #
# 参照（毎月末・上場全銘柄）
# --------------------------------------------------------------------------- #
def month_end_prices(bars: pd.DataFrame, ends: pd.DatetimeIndex,
                     tolerance_days: int = PRICE_TOLERANCE_DAYS) -> pd.DataFrame:
    """各月末 M の終値（未調整 C。M から tolerance_days 以内の最後のバー）。列: Code / Date(=M) / px / px_date。"""
    b = bars[["Code", "Date", "C"]].dropna(subset=["C"]).copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b["M"] = b["Date"] + pd.offsets.MonthEnd(0)
    last = (b.sort_values(["Code", "Date"], kind="mergesort")
             .groupby(["Code", "M"], sort=False).last().reset_index())
    last = last[last["M"].isin(ends) & ((last["M"] - last["Date"]).dt.days <= tolerance_days)]
    return (last.rename(columns={"M": "Date", "Date": "px_date", "C": "px"})
                [["Code", "Date", "px", "px_date"]].reset_index(drop=True))


def universe(prices: pd.DataFrame, master: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    参照の母集団。月末の終値のある銘柄に、その時点の33業種（S33）と市場区分を付け、普通株以外と
    ETF・REIT 等（B.EXCLUDE_MKT_CODES）を外す（build_dataset と同じ規則）。master が無ければ業種は欠測。
    """
    p = prices.copy()
    p["Code"] = p["Code"].astype(str)
    p["Date"] = pd.to_datetime(p["Date"])
    if master is not None and len(master):
        m = master.copy()
        m["Code"] = m["Code"].astype(str)
        m["Date"] = pd.to_datetime(m["Date"])
        keep = ["Date", "Code"] + [c for c in ("S33", "Mkt") if c in m.columns]
        m = (m[keep].dropna(subset=["Date", "Code"]).sort_values("Date", kind="mergesort")
              .drop_duplicates(["Date", "Code"], keep="last"))
        p = pd.merge_asof(p.sort_values("Date", kind="mergesort"), m, on="Date", by="Code",
                          direction="backward")
    p["s33"] = pd.to_numeric(p["S33"], errors="coerce") if "S33" in p.columns else np.nan
    p["mkt_code"] = pd.to_numeric(p["Mkt"], errors="coerce") if "Mkt" in p.columns else np.nan
    code = p["Code"].str.strip()
    non_common = (code.str.len() == 5) & ~code.str.endswith("0")
    if B.EXCLUDE_NON_COMMON:
        p = p[~non_common]
    if B.EXCLUDE_MKT_CODES:
        p = p[~p["mkt_code"].isin(B.EXCLUDE_MKT_CODES)]
    return p[["Code", "Date", "px", "px_date", "s33", "mkt_code"]].reset_index(drop=True)


def panel_from_quarters(keys: pd.DataFrame, q: pd.DataFrame, data_dir: str,
                        metrics: Iterable[str] = METRICS, stale_days: int = STALE_DAYS) -> pd.DataFrame:
    """
    母集団（Code / Date(=M) / px / s33）に、M までに開示済みの直近決算（quarterize_panel の出力 q）を付け、
    PER / PBR を作る。決算が stale_days より古い行は決算の値を欠測に（API の PER / PBR はそのまま。
    build_dataset と同じ）。戻り値の列: Code / Date / px / s33 + metrics（q にあるものだけ）。
    """
    k = keys.copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"])
    p = B.attach_fins(k, q)
    fin_cols = [c for c in q.columns if c not in ("Code", "DiscDate", "CurFYSt")]
    stale = (p["Date"] - p["DiscDate"]).dt.days > stale_days
    p.loc[stale, fin_cols] = np.nan
    px = p["px"].to_numpy(dtype=float)
    eps = pd.to_numeric(p["eps_ttm"], errors="coerce").to_numpy(dtype=float) if "eps_ttm" in p.columns \
        else np.full(len(p), np.nan)
    bps = pd.to_numeric(p["BPS"], errors="coerce").to_numpy(dtype=float) if "BPS" in p.columns \
        else np.full(len(p), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        own = {"per": np.where(eps > 0, px / eps, np.nan), "pbr": np.where(bps > 0, px / bps, np.nan)}
    av = B.api_valuation(p, "Date", data_dir=data_dir, asof=True)
    for name, cap in (("per", B.PER_MAX), ("pbr", B.PBR_MAX)):
        a = (av[f"api_{name}"].to_numpy(dtype=float) if B.USE_API_VALUATION.get(name, True)
             else np.full(len(p), np.nan))
        merged = np.where(np.isfinite(a) & (a > 0), a, own[name])
        p[name] = np.where(np.isfinite(merged) & (merged <= cap), merged, np.nan)
    cols = ["Code", "Date", "px", "s33"] + [m for m in metrics if m in p.columns]
    return p[cols].reset_index(drop=True)


def load_master(data_dir: str) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(x) for x in paths], ignore_index=True)


def load_bars_close(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit(f"bars_*.parquet が {data_dir} にありません")
    return pd.concat([pd.read_parquet(x, columns=["Code", "Date", "C"]) for x in paths], ignore_index=True)


def reference_panel(data_dir: str = B.DATA_DIR, ends: Optional[pd.DatetimeIndex] = None,
                    metrics: Iterable[str] = METRICS) -> pd.DataFrame:
    """毎月末 × 上場全銘柄の参照の表。ends を省くとバーのある期間の全月末。"""
    bars = load_bars_close(data_dir)
    bars["Date"] = pd.to_datetime(bars["Date"])
    if ends is None:
        ends = month_ends(bars["Date"].min(), bars["Date"].max())
    prices = month_end_prices(bars, ends)
    del bars
    keys = universe(prices, load_master(data_dir))
    fins = B.load_parts("fins", data_dir)
    q = B.quarterize_panel(fins, data_dir)
    del fins
    panel = panel_from_quarters(keys, q, data_dir, metrics)
    print(f"[rel] 参照: {panel['Date'].nunique()}月末 × 延べ {len(panel):,}銘柄月"
          f"（{panel['Date'].min().date()} 〜 {panel['Date'].max().date()}）")
    return panel


# --------------------------------------------------------------------------- #
# 百分位
# --------------------------------------------------------------------------- #
def percentile(values, ref_sorted) -> np.ndarray:
    """
    参照（昇順）の中での values の百分位（0〜1）: 参照の中で値より小さいものの割合 + 同じ値の割合の半分。
    値が欠測なら欠測。参照が空なら全部欠測。
    """
    v = np.asarray(values, dtype=float)
    a = np.asarray(ref_sorted, dtype=float)
    out = np.full(len(v), np.nan)
    n = len(a)
    if n == 0:
        return out
    ok = np.isfinite(v)
    lo = np.searchsorted(a, v[ok], side="left")
    hi = np.searchsorted(a, v[ok], side="right")
    out[ok] = (lo + hi) / (2.0 * n)
    return out


def reference_dates(dates, ends: np.ndarray) -> np.ndarray:
    """各行の日付より前（<）の最後の月末。無ければ NaT。"""
    d = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    e = np.sort(np.asarray(ends, dtype="datetime64[ns]"))
    pos = np.searchsorted(e, d, side="left") - 1
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    has = pos >= 0
    out[has] = e[pos[has]]
    return out


def attach(frame: pd.DataFrame, panel: pd.DataFrame, metrics: Iterable[str] = METRICS,
           min_industry: int = MIN_INDUSTRY, min_market: int = MIN_MARKET,
           industry_col: str = "s33_code") -> pd.DataFrame:
    """
    frame（Code / Date / 指標の列 / s33_code）に rel_mkt_<指標> と rel_ind_<指標> を付けた写し。
    行の日付より前の最後の月末の参照を使う。参照が無い行（最初の月末より前）は欠測。
    """
    metrics = list(metrics)
    out = frame.copy()
    n = len(out)
    ends = panel["Date"].unique()
    ref = reference_dates(out["Date"], ends)
    ind = (pd.to_numeric(out[industry_col], errors="coerce").to_numpy(dtype=float)
           if industry_col in out.columns else np.full(n, np.nan))
    res = {c: np.full(n, np.nan) for c in columns("all", metrics)}
    vals = {m: pd.to_numeric(out[m], errors="coerce").to_numpy(dtype=float) for m in metrics if m in out.columns}
    by_end = {pd.Timestamp(d): g for d, g in panel.groupby("Date", sort=False)}
    groups = pd.Series(ref).groupby(pd.Series(ref)).indices    # NaT は群にならない
    for d, rows in groups.items():
        g = by_end.get(pd.Timestamp(d))
        if g is None:
            continue
        rows = np.asarray(rows)
        n_mkt = len(g)
        if n_mkt < min_market:
            continue
        ind_rows = ind[rows]
        ind_groups = g.groupby("s33", sort=False).indices if "s33" in g.columns else {}
        for m in metrics:
            if m not in vals or m not in g.columns:
                continue
            v = vals[m][rows]
            col = g[m].to_numpy(dtype=float)
            a = np.sort(col[np.isfinite(col)])
            res[f"rel_mkt_{m}"][rows] = percentile(v, a)
            for s in np.unique(ind_rows[np.isfinite(ind_rows)]):
                gi = ind_groups.get(s)
                if gi is None or len(gi) < min_industry:
                    continue
                ai = col[gi]
                ai = np.sort(ai[np.isfinite(ai)])
                sel = ind_rows == s
                res[f"rel_ind_{m}"][rows[sel]] = percentile(v[sel], ai)
    for c, v in res.items():
        out[c] = v
    return out


def coverage(panel: pd.DataFrame, metrics: Iterable[str] = METRICS,
             min_industry: int = MIN_INDUSTRY) -> dict:
    """参照の大きさ（月末ごとの銘柄数・業種の数・指標の充足）。件数と割合だけ。"""
    metrics = [m for m in metrics if m in panel.columns]
    names = panel.groupby("Date").size()
    ind_n = panel.dropna(subset=["s33"]).groupby(["Date", "s33"]).size()
    ind_ok = ind_n[ind_n >= min_industry].groupby(level=0).size().reindex(names.index, fill_value=0)
    fill = panel.groupby("Date")[metrics].apply(lambda g: g.notna().mean())
    return {"month_ends": int(len(names)),
            "first": str(names.index.min().date()), "last": str(names.index.max().date()),
            "names": {"min": int(names.min()), "median": float(names.median()), "max": int(names.max())},
            "industries": int(panel["s33"].nunique()),
            "industries_ok": {"min": int(ind_ok.min()), "median": float(ind_ok.median()), "max": int(ind_ok.max())},
            "with_industry": float(panel["s33"].notna().mean()),
            "fill": {m: float(fill[m].median()) for m in metrics}}
