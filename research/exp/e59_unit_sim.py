#!/usr/bin/env python3
"""
実験59: 選び方4通りで、条件を満たす銘柄を **1日1銘柄・1単元（100株）** ずつ買ったときの損益と
資産回転率。枠3が埋まっているときに良い候補が出たら「どれを売るか（そもそも乗り換えるか）」も測る。

運用者の依頼（2026-10-03）
  「銘柄の選び方を gbdt3モデルで97.5%以上 / 95%以上、全5モデルで95%以上 / 90%以上 の4パターンで、
   条件を満たす銘柄を1単元（100株）を購入した場合の損益シミュレーションをしてほしいです。
   同時保有数は３銘柄なので、資産回転率も指標に入れたいです。」
  「1日での購入数はベストなもの1銘柄にしてほしいです。なぜならスコアがいつ予測したのかという日付に
   依存するので。購入日付も分散させたいので。ですので、問題はすでに3つ持ってる際に、よりよい4つ目の
   候補がでた際にどれを売るかという話になるかと思います。そもそも銘柄を乗り換えるかどうかも含めて。」

何を測るか
  本番モデル5つ（LightGBM / XGBoost / CatBoost / ロジスティック回帰 / MLP）の out-of-fold
  （Release の oof.parquet / <algo>_oof.parquet。その行より前のデータだけで学習した採点）に選び方を
  当て、**その日のベスト1件**（合議の最小の百分位 p_min が最大。同点は LightGBM のスコア順）を翌営業日の
  寄りで 100株 買う。+20% の指値、届かなければ 20営業日目の終値で売る。同時に持つのは 3銘柄まで。

選び方（百分位 = そのモデルの OOF の分布のうち、そのスコアより低い割合 × 100。画面と同じ式）
  GBDT3 97.5以上     ブースティング3モデルすべてが 97.5 以上
  GBDT3 95以上       同 95 以上
  全5モデル 95以上   5モデルすべてが 95 以上
  全5モデル 90以上   同 90 以上
  （参考）現行の規則  GBDT3 90以上・発火8件以上の日だけ・日内上位2件・乗り換えなし（画面の規則そのまま）

枠が埋まっているときの乗り換え（腕）。判断はその日の終値で、売りも買いも翌営業日の寄り
  A なし              埋まっていたら見送る（先着順。実験33〜35・実運用の追跡と同じ）
  B 一番古い玉        保有日数が最長の玉を売る
  C 含みが一番悪い玉  終値での含み損益が最小の玉を売る
  D 格が一番低い玉    入ったときの p_min が最小の玉を、新しい候補の p_min がそれより高いときだけ売る
  E 伸びていない玉    買ってから 5営業日以上たって含みが 0% 以下の玉があるときだけ、その中で一番悪い玉を売る
  F 含み益の薄い玉    含み益が 0% 以上の玉があるときだけ、その中で一番薄い玉を売る（実験34 の運用者案の形）
  売った玉は「持ち切っていたら幾らになったか」（規則どおりの出口）も数え、乗り換えの損得を出す。

百分位の取り方は2通り出す
  過去分布（本命）   その日より前の OOF の分布で百分位を出す（過去の行が 500 未満の日は判定しない）。
                     画面が「その時点の本番モデルの OOF」で百分位を出すのと同じ立場で、先の値を見ない
  全期間分布（参考） OOF 全体の分布で百分位を出す（実運用の追跡 research/live_track.py の見込みと
                     同じ約束）。後半ほどスコアが高いので、選定が 2024〜25年に偏る。腕 A だけ出す

数え方
  買値        翌営業日の始値。買付額は **調整前の始値 × 100株**（分割調整後の値だと、後で分割した
              銘柄の当時の金額を小さく見積もる）。円の損益 = 買付額 × 収益率（値動きは分割調整後）
  出口        +20% は 1営業日目（買った日）から高値で見る。届けば +20% ちょうど（指値）。
              届かなければ 20営業日目の終値。乗り換えの売りは翌営業日の始値
  手数料・税・配当・スリッページは含まない。終わっていない玉（保有中）は成績に数えない。
  持っている銘柄は買い増さない
  投下資本    その営業日に持っていた玉の買付額の合計。平均（空き枠は 0 と数える）と
              最大（= 必要資金の目安）
  回転率      年間の買付額 ÷ 必要資金（最大同時投下資本）。用意した資金が年に何回まわるか。
              ÷ 平均投下資本 も出す（こちらは保有期間の短さだけを映す）
  稼働率      枠が埋まっていた営業日の割合（Σ保有日数 ÷ 枠数 ÷ 営業日数）

出力  research/_data/oof/e59_summary.csv（選び方 × 腕 × 百分位の取り方）、e59_by_year.csv、
      e59_trades.csv（取引の一覧。銘柄コードと買値を含むので、公開の場には出さない）。
      本番の設定は何も変えない。ログに出すのは件数・割合・金額の合計だけ。

使い方
    python3 research/exp/e59_unit_sim.py
    python3 research/exp/e59_unit_sim.py --oof-dir research/_data --hows past --arms A,B,C
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_track as L  # noqa: E402

OOF_DIR = os.path.join(L.DATA_DIR, "oof")

UNIT = 100            # 1単元の株数
MIN_HIST = 500        # 過去分布で百分位を出すのに要る過去の行数（約 40営業日ぶん）
YEAR_DAYS = 365.25    # 「年間」に直すときの日数（暦の日数で割る）

#: (名前, 合議に使うモデル, 百分位の下限)。発火数の足切りなし、1日に最大 枠数 件
PATTERNS = (
    ("GBDT3 97.5以上", L.BOOST, 97.5),
    ("GBDT3 95以上", L.BOOST, 95.0),
    ("全5モデル 95以上", L.ALL5, 95.0),
    ("全5モデル 90以上", L.ALL5, 90.0),
)
#: 参考: 画面の規則そのまま（GBDT3 90以上・発火 8件以上・上位 2件・乗り換えなし）
REFERENCE = ("参考: 現行の規則（GBDT3 90・発火8件以上・上位2件）", L.BOOST, L.AGREE_PCT)
HOWS = {"past": "過去分布", "whole": "全期間分布"}
SWAP = "乗り換え売却"
DONE = ("+20%到達", "満了", SWAP)
#: 乗り換えの腕。枠が埋まっているときに、どの玉を売るか
ARMS = {
    "A": "A なし（埋まっていたら見送る）",
    "B": "B 一番古い玉を売る",
    "C": "C 含みが一番悪い玉を売る",
    "D": "D 格（入ったときの p_min）が一番低い玉を、候補がそれより上なら売る",
    "E": "E 伸びていない玉（5日以上で含み0%以下）があれば一番悪い玉を売る",
    "F": "F 含み益の玉があれば一番薄い玉を売る（実験34 の運用者案）",
}
STALL_DAYS = 5        # 腕 E: 買ってから何営業日たったら「伸びていない」を見るか（買った日 = 1）
STALL_GAIN = 0.0      # 腕 E: 含みがこの % 以下なら「伸びていない」


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- #
# 百分位
# ---------------------------------------------------------------- #

def pct_expanding(dates, scores, min_hist: int = MIN_HIST) -> np.ndarray:
    """
    その日より前（同じ日は含まない）の分布で百分位。丸めは live_track.pct_of と同じ
    （小数1桁、Python の round）。過去の行が min_hist 未満の日と、スコアが NaN の行は NaN。
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    s = np.asarray(scores, dtype=float)
    out = np.full(len(s), np.nan)
    order = np.argsort(d, kind="stable")
    hist = np.array([], dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and d[order[j]] == d[order[i]]:
            j += 1
        idx = order[i:j]
        if len(hist) >= min_hist:
            k = np.searchsorted(hist, s[idx], side="left")
            vals = np.array([round(float(x / len(hist)) * 100, 1) for x in k])
            vals[~np.isfinite(s[idx])] = np.nan
            out[idx] = vals
        new = s[idx]
        hist = np.sort(np.concatenate([hist, new[np.isfinite(new)]]))
        i = j
    return out


def pct_whole(scores) -> np.ndarray:
    """OOF 全体の分布で百分位（live_track.pct_of）。スコアが NaN の行は NaN。"""
    s = np.asarray(scores, dtype=float)
    out = L.pct_of(s, s[np.isfinite(s)]).astype(float)
    out[~np.isfinite(s)] = np.nan
    return out


def load_oofs(oof_dir: str, model_dir: str, algos: Sequence[str] = L.ALL5) -> Dict[str, pd.DataFrame]:
    """本番モデルの OOF を algo ごとに読む（置き場所の探し方は live_track.find_oof）。"""
    oofs = {}
    for a in algos:
        path = L.find_oof(a, oof_dir, model_dir)
        if path is None:
            raise SystemExit(f"{a} の OOF が見つかりません（{oof_dir} / {model_dir}）")
        o = pd.read_parquet(path)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        oofs[a] = o
    return oofs


def rows_with_pct(oofs: Dict[str, pd.DataFrame], how: str, min_hist: int = MIN_HIST) -> pd.DataFrame:
    """
    5モデルの OOF を LightGBM の行に揃え（左結合。欠けたモデルの百分位は NaN のまま →
    そのモデルを使う選び方では「判定できない」）、s_<algo> / p_<algo> を付ける。
    """
    base = None
    for a, o in oofs.items():
        keep = ["Code", "Date", "score"] + ([c for c in ("label", "ret_o1_20") if c in o]
                                            if a == "lgbm" else [])
        o = o[keep].rename(columns={"score": f"s_{a}"}).copy()
        s = o[f"s_{a}"].to_numpy(dtype=float)
        o[f"p_{a}"] = pct_whole(s) if how == "whole" else pct_expanding(o["Date"], s, min_hist)
        if base is None:
            base = o
        else:
            base = base.merge(o, on=["Code", "Date"], how="left")
    base["score"] = base["s_lgbm"]
    return base.sort_values(["Date", "Code"]).reset_index(drop=True)


# ---------------------------------------------------------------- #
# 値動きの表: 買う候補の銘柄だけ、営業日 × 銘柄の行列にする
# ---------------------------------------------------------------- #

class PriceGrid:
    """
    営業日（bar_days の順）× 銘柄 の行列。AdjO / AdjH / AdjC（分割調整後）と O（調整前の始値）。
    その日に日足が無い升は NaN。
    """

    def __init__(self, bars: pd.DataFrame, bar_days: Sequence, codes: Sequence[str]):
        codes = sorted(set(str(c) for c in codes))
        self.days = pd.DatetimeIndex(bar_days)
        self.col = {c: i for i, c in enumerate(codes)}
        b = bars[bars["Code"].isin(codes)]

        def grid(name: str) -> np.ndarray:
            if not codes:
                return np.zeros((len(self.days), 0))
            m = b.pivot_table(index="Date", columns="Code", values=name, aggfunc="last")
            return m.reindex(index=self.days, columns=codes).to_numpy(dtype=float)

        self.adjo, self.adjh, self.adjc, self.o = grid("AdjO"), grid("AdjH"), grid("AdjC"), grid("O")


def rule_exit(px: PriceGrid, col: int, buy: int, entry: float, *, hold: int = L.HOLD,
              target: float = L.TAKE_PROFIT):
    """
    買った日 buy（index）から規則どおりに持ったときの出口。(出口の index, 収益率 %, status)。
    +20% は買った日から hold 日目までの高値で見る（live_track.forward と同じ）。届かなければ
    hold 日目の終値。その日の終値が無ければ、その後の最初の終値。先の足が無ければ (None, nan, "保有中")。
    """
    level = entry * (1.0 + target / 100.0)
    T = len(px.days)
    for k in range(buy, min(buy + hold, T)):
        h = px.adjh[k, col]
        if np.isfinite(h) and h >= level:
            return k, float(target), "+20%到達"
    for k in range(buy + hold - 1, T):
        c = px.adjc[k, col]
        if np.isfinite(c):
            return k, float((c / entry - 1.0) * 100.0), "満了"
    return None, float("nan"), "保有中"


# ---------------------------------------------------------------- #
# 枠3の模擬: 1日1件、埋まっていたら腕の規則で乗り換える
# ---------------------------------------------------------------- #

def unrealized(px: PriceGrid, p: dict, t: int) -> float:
    """玉 p の、営業日 t の終値での含み損益（%）。終値が無ければ NaN。"""
    c = px.adjc[t, p["col"]]
    return float((c / p["entry"] - 1.0) * 100.0) if np.isfinite(c) else float("nan")


def choose_sell(arm: str, pos: Dict[int, dict], cand, t: int, px: PriceGrid, *,
                stall_days: int = STALL_DAYS, stall_gain: float = STALL_GAIN):
    """
    枠が埋まっているとき、腕 arm の規則で売る枠を選ぶ。売らないなら None。
    判断は営業日 t の終値（翌営業日の寄りで売って買う）。同点は枠の番号が小さいほう。
    """
    items = sorted(pos.items())
    if arm == "A" or not items:
        return None
    if arm == "B":
        return min(items, key=lambda kv: (kv[1]["buy"], kv[0]))[0]
    if arm == "D":
        s, p = min(items, key=lambda kv: (kv[1]["p_min"], kv[0]))
        return s if float(cand.p_min) > p["p_min"] else None
    vals = [(unrealized(px, p, t), s, p) for s, p in items]
    vals = [(u, s, p) for u, s, p in vals if np.isfinite(u)]
    if arm == "C":
        pass
    elif arm == "E":
        vals = [(u, s, p) for u, s, p in vals if (t - p["buy"] + 1) >= stall_days and u <= stall_gain]
    elif arm == "F":
        vals = [(u, s, p) for u, s, p in vals if u >= 0.0]
    else:
        raise ValueError(f"知らない腕: {arm}")
    return min(vals, key=lambda v: (v[0], v[1]))[1] if vals else None


def _closed(p: dict, px: PriceGrid, exit_idx: int, last_idx: int, ret: float, status: str,
            unit: int) -> dict:
    kept_idx, kept_ret, kept_status = p["rule"]
    return {"Code": p["code"], "Date": p["sel"], "buy_date": px.days[p["buy"]],
            "exit_date": px.days[exit_idx], "status": status, "days": float(last_idx - p["buy"] + 1),
            "rank": p["rank"], "slot": p["slot"], "p_min": p["p_min"], "via_swap": p["via_swap"],
            "entry": p["entry"], "raw_open": p["capital"] / unit, "capital": p["capital"],
            "ret": float(ret), "pnl": p["capital"] * float(ret) / 100.0,
            "kept_ret": float(kept_ret) if status == SWAP else float("nan"),
            "kept_days": float(kept_idx - p["buy"] + 1) if status == SWAP and kept_idx is not None else float("nan"),
            "kept_status": kept_status if status == SWAP else "",
            "kept_pnl": (p["capital"] * float(kept_ret) / 100.0
                         if status == SWAP and np.isfinite(kept_ret) else float("nan"))}


def simulate_arm(picks: pd.DataFrame, px: PriceGrid, *, arm: str = "A", slots: int = L.SLOTS,
                 hold: int = L.HOLD, target: float = L.TAKE_PROFIT, unit: int = UNIT,
                 stall_days: int = STALL_DAYS, stall_gain: float = STALL_GAIN):
    """
    日ごとに進める模擬。picks は decide の買う銘柄（Date / Code / rank / p_min）。
      寄り      前日の引け後に決めた「乗り換えの売り」→「買い」を始値で。始値が無ければ何もしない
      場中〜引け 規則どおりの出口（+20% は高値、hold 日目は終値）
      引け後    その日の候補（rank 順）。空き枠があれば翌寄りで買う。埋まっていれば腕の規則で
                売る枠を選ぶ（候補が複数ある日は腕 A だけ）。持っている銘柄は買い増さない
    戻り値: (終わった取引の表, 数え上げ dict, まだ持っている玉のリスト)
    """
    T = len(px.days)
    idx = {d: i for i, d in enumerate(px.days)}
    by_day: Dict[int, list] = {}
    for r in picks.sort_values(["Date", "rank"]).itertuples(index=False):
        i = idx.get(pd.Timestamp(r.Date))
        if i is not None:
            by_day.setdefault(i, []).append(r)
    pos: Dict[int, dict] = {}
    trades = []
    pending: list = []
    n_full = n_swaps = n_noopen = n_dup = 0
    full_days: list = []          # 枠が無くて見送った候補の選定日
    for t in range(T):
        # 1) 寄り: 乗り換えの売り → 買い
        for cand, sell_slot in pending:
            col = px.col[str(cand.Code)]
            o = px.adjo[t, col]
            if not (np.isfinite(o) and o > 0):
                n_noopen += 1
                continue
            if sell_slot is not None:
                p = pos[sell_slot]
                po = px.adjo[t, p["col"]]              # 売る玉は **その玉の** 始値で売る
                if not (np.isfinite(po) and po > 0):
                    n_noopen += 1                      # 売る側が寄り付かなければ乗り換えない
                    continue
                pos.pop(sell_slot)
                trades.append(_closed(p, px, t, t - 1, (po / p["entry"] - 1.0) * 100.0, SWAP, unit))
                n_swaps += 1
                slot = sell_slot
            else:
                free = [s for s in range(slots) if s not in pos]
                if not free:
                    n_full += 1
                    full_days.append(pd.Timestamp(cand.Date))
                    continue
                slot = free[0]
            pos[slot] = {"code": str(cand.Code), "col": col, "buy": t, "entry": float(o),
                         "capital": float(px.o[t, col]) * unit, "p_min": float(cand.p_min),
                         "sel": pd.Timestamp(cand.Date), "rank": int(cand.rank), "slot": slot,
                         "via_swap": sell_slot is not None,
                         "rule": rule_exit(px, col, t, float(o), hold=hold, target=target)}
        pending = []
        # 2) 場中〜引け: 規則どおりの出口
        for s, p in list(pos.items()):
            if p["rule"][0] == t:
                trades.append(_closed(p, px, t, t, p["rule"][1], p["rule"][2], unit))
                del pos[s]
        # 3) 引け後: その日の候補
        cands = by_day.get(t, [])
        if not cands or t + 1 >= T:
            continue
        held = {p["code"] for p in pos.values()}
        n_free = slots - len(pos)
        for cand in cands:
            if str(cand.Code) not in px.col or str(cand.Code) in held:
                n_dup += 1
                continue
            if n_free > 0:
                pending.append((cand, None))
                n_free -= 1
                held.add(str(cand.Code))
            elif not pending and len(cands) == 1:
                s = choose_sell(arm, pos, cand, t, px, stall_days=stall_days, stall_gain=stall_gain)
                if s is None:
                    n_full += 1
                    full_days.append(pd.Timestamp(cand.Date))
                else:
                    pending.append((cand, s))
            else:
                n_full += 1
                full_days.append(pd.Timestamp(cand.Date))
    counts = {"n_full": n_full, "n_swaps": n_swaps, "n_noopen": n_noopen, "n_dup": n_dup,
              "n_open": len(pos), "full_days": full_days}
    cols = ["Code", "Date", "buy_date", "exit_date", "status", "days", "rank", "slot", "p_min", "via_swap",
            "entry", "raw_open", "capital", "ret", "pnl", "kept_ret", "kept_days", "kept_status", "kept_pnl"]
    return pd.DataFrame(trades, columns=cols), counts, list(pos.values())


def empty_trades() -> pd.DataFrame:
    """取引が1件も無いときの空の表（stats が読む列だけ持つ）。"""
    cols = {c: pd.Series(dtype=float) for c in ("ret", "capital", "pnl", "days", "kept_ret", "kept_pnl")}
    cols["status"] = pd.Series(dtype=object)
    cols["via_swap"] = pd.Series(dtype=bool)
    cols["buy_date"] = pd.Series(dtype="datetime64[ns]")
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- #
# 投下資本と成績
# ---------------------------------------------------------------- #

def capital_curve(done: pd.DataFrame, bar_days: Sequence) -> pd.Series:
    """
    営業日ごとの投下資本（その日に持っていた玉の買付額の合計）。
    買った日から、最後に持ち越した日まで（days 日ぶん。乗り換えの売りは翌寄りなので前日まで）。
    """
    idx = {pd.Timestamp(d): i for i, d in enumerate(bar_days)}
    diff = np.zeros(len(bar_days) + 1)
    for r in done.itertuples(index=False):
        if not (np.isfinite(r.capital) and np.isfinite(r.days)) or r.days < 1:
            continue
        b = idx[pd.Timestamp(r.buy_date)]
        e = min(b + int(r.days) - 1, len(bar_days) - 1)
        diff[b] += r.capital
        diff[e + 1] -= r.capital
    return pd.Series(np.cumsum(diff)[:len(bar_days)], index=pd.DatetimeIndex(bar_days))


def stats(done: pd.DataFrame, curve: pd.Series, lo, hi, *, slots: int, counts: Optional[dict] = None,
          n_pass: int = 0) -> dict:
    """
    期間 [lo, hi]（日付）の成績。done は終わった取引（status が DONE）だけ。
    curve は capital_curve（期間で切る）。年間は暦の日数で割る。
    """
    counts = counts or {}
    if len(curve):
        c = curve[(curve.index >= pd.Timestamp(lo)) & (curve.index <= pd.Timestamp(hi))]
    else:
        c = curve
    n_days = int(len(c))
    years = max((pd.Timestamp(hi) - pd.Timestamp(lo)).days, 1) / YEAR_DAYS
    r = pd.to_numeric(done["ret"], errors="coerce")
    ok = r.notna()
    r = r[ok]
    d = done.loc[ok]
    n = int(len(r))
    cap = d["capital"]
    pnl = d["pnl"]
    has_cap = cap.notna()
    buy_total = float(cap[has_cap].sum())
    cap_max = float(c.max()) if n_days else float("nan")
    cap_avg = float(c.mean()) if n_days else float("nan")
    held = pd.to_numeric(d["days"], errors="coerce")
    sw = d[d["status"] == SWAP]
    kept = pd.to_numeric(sw["kept_ret"], errors="coerce")
    both = kept.notna()
    via = d[d["via_swap"].astype(bool)] if "via_swap" in d else d.iloc[:0]
    counts = {k: v for k, v in counts.items() if k != "full_days"}
    out = {
        "n": n, "per_year": n / years, "years": years, "n_days": n_days, "n_pass": int(n_pass),
        "n_full": int(counts.get("n_full", 0)), "n_swaps": int(counts.get("n_swaps", 0)),
        "n_noopen": int(counts.get("n_noopen", 0)), "n_dup": int(counts.get("n_dup", 0)),
        "n_open": int(counts.get("n_open", 0)),
        "win": float((r > 0).mean() * 100) if n else float("nan"),
        "hit": float((d["status"] == "+20%到達").mean() * 100) if n else float("nan"),
        "mean": float(r.mean()) if n else float("nan"),
        "median": float(r.median()) if n else float("nan"),
        "se": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "worst": float(r.min()) if n else float("nan"),
        "n_no_capital": int((~has_cap).sum()),
        "pnl_total": float(pnl[has_cap].sum()),
        "pnl_per_trade": float(pnl[has_cap].mean()) if has_cap.any() else float("nan"),
        "pnl_per_year": float(pnl[has_cap].sum()) / years,
        "pnl_worst": float(pnl[has_cap].min()) if has_cap.any() else float("nan"),
        "buy_total": buy_total,
        "buy_median": float(cap[has_cap].median()) if has_cap.any() else float("nan"),
        "buy_max": float(cap[has_cap].max()) if has_cap.any() else float("nan"),
        "cap_max": cap_max, "cap_avg": cap_avg,
        "turnover": (buy_total / years / cap_max) if cap_max and cap_max > 0 else float("nan"),
        "turnover_avg": (buy_total / years / cap_avg) if cap_avg and cap_avg > 0 else float("nan"),
        "roi_max": (float(pnl[has_cap].sum()) / years / cap_max * 100) if cap_max and cap_max > 0 else float("nan"),
        "roi_avg": (float(pnl[has_cap].sum()) / years / cap_avg * 100) if cap_avg and cap_avg > 0 else float("nan"),
        "util": float(held.sum() / (slots * n_days) * 100) if n_days else float("nan"),
        "hold_mean": float(held.mean()) if n else float("nan"),
        # 乗り換えの損得（売った玉: 実現 vs 持ち切っていたら）
        "swap_n": int(len(sw)),
        "swap_realized": float(sw["ret"].mean()) if len(sw) else float("nan"),
        "swap_kept": float(kept[both].mean()) if both.any() else float("nan"),
        "swap_realized_b": float(sw.loc[both, "ret"].mean()) if both.any() else float("nan"),
        "swap_lost_share": float((kept[both] > sw.loc[both, "ret"]).mean() * 100) if both.any() else float("nan"),
        "swap_lost_yen": float((sw.loc[both, "kept_pnl"] - sw.loc[both, "pnl"]).sum()) if both.any() else 0.0,
        "swap_unknown": int((~both).sum()),
        "via_n": int(len(via)),
        "via_mean": float(via["ret"].mean()) if len(via) else float("nan"),
    }
    return out


def complete_cutoff(bar_days: Sequence, hold: int = L.HOLD):
    """選定日がこの日以前なら 20営業日の出口まで日足が揃っている（live_track と同じ）。"""
    return bar_days[-1 - hold] if len(bar_days) > hold else None


# ---------------------------------------------------------------- #
# 表示
# ---------------------------------------------------------------- #

def _man(v) -> str:
    """円を万円で（小数なし、カンマ区切り）。−5千円〜0円は「-0」ではなく「0」。"""
    if v is None or not np.isfinite(v):
        return "-"
    out = f"{v / 1e4:,.0f}"
    return "0" if out == "-0" else out


def _pct(v, fmt="{:+.2f}") -> str:
    return "-" if v is None or not np.isfinite(v) else fmt.format(v)


NAME_W = 46
COLS = (("取引", 5), ("年間", 6), ("乗換", 5), ("勝率", 6), ("到達", 6), ("1取引%", 8), ("SE", 6),
        ("総損益万円", 11), ("年万円", 8), ("必要資金", 9), ("平均投下", 9), ("回転", 6), ("稼働", 6), ("保有日", 7))
HEAD = "  " + L._l("腕 / 選び方", NAME_W) + "".join(L._r(h, w) for h, w in COLS)


def line(name: str, s: dict) -> str:
    if not s["n"]:
        return f"  {L._l(name, NAME_W)}（取引なし。基準を満たした {s['n_pass']}件）"
    vals = (s["n"], f"{s['per_year']:.1f}", s["n_swaps"], _pct(s["win"], "{:.0f}%"), _pct(s["hit"], "{:.0f}%"),
            _pct(s["mean"]), _pct(s["se"], "{:.2f}"), _man(s["pnl_total"]), _man(s["pnl_per_year"]),
            _man(s["cap_max"]), _man(s["cap_avg"]), _pct(s["turnover"], "{:.1f}"),
            _pct(s["util"], "{:.0f}%"), _pct(s["hold_mean"], "{:.1f}"))
    return "  " + L._l(name, NAME_W) + "".join(L._r(v, w) for v, (_, w) in zip(vals, COLS))


SWAP_COLS = (("乗換", 5), ("実現%", 8), ("持ち切り%", 10), ("差pt", 7), ("切って損", 9), ("取り逃し万円", 13),
             ("入替後%", 8), ("不明", 5))
SWAP_HEAD = "  " + L._l("腕 / 選び方", NAME_W) + "".join(L._r(h, w) for h, w in SWAP_COLS)


def swap_line(name: str, s: dict) -> str:
    diff = s["swap_realized_b"] - s["swap_kept"] if np.isfinite(s["swap_kept"]) else float("nan")
    vals = (s["swap_n"], _pct(s["swap_realized_b"]), _pct(s["swap_kept"]), _pct(diff, "{:+.2f}"),
            _pct(s["swap_lost_share"], "{:.0f}%"), _man(s["swap_lost_yen"]), _pct(s["via_mean"]), s["swap_unknown"])
    return "  " + L._l(name, NAME_W) + "".join(L._r(v, w) for v, (_, w) in zip(vals, SWAP_COLS))


YCOLS = (("候補", 6), ("取引", 5), ("満杯", 5), ("勝率", 6), ("1取引%", 8), ("損益万円", 9), ("買付万円", 10),
         ("必要資金", 9), ("回転", 6), ("稼働", 6))
YEAR_HEAD = "    " + L._l("年", 6) + "".join(L._r(h, w) for h, w in YCOLS)


def year_line(y, s: dict) -> str:
    if not s["n"]:
        vals = (s["n_pass"], 0, s["n_full"]) + ("-",) * 7
    else:
        vals = (s["n_pass"], s["n"], s["n_full"], _pct(s["win"], "{:.0f}%"), _pct(s["mean"]),
                _man(s["pnl_total"]), _man(s["buy_total"]), _man(s["cap_max"]),
                _pct(s["turnover"], "{:.1f}"), _pct(s["util"], "{:.0f}%"))
    return "    " + L._l(str(y), 6) + "".join(L._r(v, w) for v, (_, w) in zip(vals, YCOLS))


# ---------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------- #

def selections(base: pd.DataFrame, slots: int):
    """選び方ごとの (名前, 基準を満たした行, 買う銘柄) を返す。4つの選び方は 1日1件、参考は画面の規則。"""
    out = []
    for name, models, agree in PATTERNS:
        r, _, picks = L.decide(base, agree=agree, min_break=0, top_k=1, models=models)
        out.append((name, r[r["passed"] & r["buyable_day"]], picks))
    r, _, picks = L.decide(base, agree=REFERENCE[2], min_break=L.SKIP_BREAKS, top_k=L.TOP_K, models=REFERENCE[1])
    out.append((REFERENCE[0], r[r["passed"] & r["buyable_day"]], picks))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験59: 選び方4通り × 1日1件 × 1単元 × 枠3（乗り換えの腕つき）")
    ap.add_argument("--data-dir", default=L.DATA_DIR)
    ap.add_argument("--oof-dir", default=L.DATA_DIR,
                    help="Release から落とした OOF（oof.parquet / <algo>_oof.parquet）の場所")
    ap.add_argument("--model-dir", default=L.MODEL_DIR)
    ap.add_argument("--slots", type=int, default=L.SLOTS)
    ap.add_argument("--unit", type=int, default=UNIT, help="1単元の株数")
    ap.add_argument("--min-hist", type=int, default=MIN_HIST, help="過去分布に要る過去の行数")
    ap.add_argument("--hows", default="past,whole", help="百分位の取り方（past / whole）")
    ap.add_argument("--arms", default=",".join(ARMS), help="乗り換えの腕（A〜F）。全期間分布は A だけ")
    ap.add_argument("--stall-days", type=int, default=STALL_DAYS)
    ap.add_argument("--stall-gain", type=float, default=STALL_GAIN)
    ap.add_argument("--prefix", default="e59", help="出力ファイル名の頭")
    args = ap.parse_args(argv)
    hows = [h for h in args.hows.split(",") if h]
    bad = [h for h in hows if h not in HOWS]
    if bad:
        raise SystemExit(f"百分位の取り方は {list(HOWS)} から: {bad}")
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad or "A" not in arms:
        raise SystemExit(f"腕は {list(ARMS)} から、A を含めて: {arms}")
    os.makedirs(OOF_DIR, exist_ok=True)

    oofs = load_oofs(args.oof_dir, args.model_dir)
    lo_oof = min(o["Date"].min() for o in oofs.values())
    hi_oof = max(o["Date"].max() for o in oofs.values())
    meta_path = os.path.join(args.model_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    bars = L.load_bars(args.data_dir, start=lo_oof - pd.Timedelta(days=10), extra=("O",))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    cutoff = complete_cutoff(bar_days)
    if cutoff is None:
        raise SystemExit("日足が足りません")
    lo = lo_oof
    hi = min(hi_oof, cutoff)

    print("=" * 110)
    print(f"実験59 選び方4通り × 1日1件 × 1単元（{args.unit}株）× 枠{args.slots}: 損益・資産回転率・乗り換えの腕")
    print("=" * 110)
    print(f"  OOF {lo_oof.date()}〜{hi_oof.date()}（{len(oofs['lgbm']):,}行 / "
          f"{oofs['lgbm']['Date'].nunique():,}営業日）、モデル {len(oofs)}つ"
          + (f"、学習 {str(meta.get('trainedAt', ''))[:10]} / {meta.get('preset', '')}" if meta else ""))
    print(f"  日足 〜{bar_days[-1].date()}。成績に数える選定日 {lo.date()}〜{hi.date()}"
          f"（20営業日の出口まで日足が揃う範囲）")
    print(f"  出口: +{L.TAKE_PROFIT:.0f}% の指値 / {L.HOLD}営業日目の終値 / 乗り換えは翌寄り。手数料・税・配当は含まない。"
          f"年間は暦の日数（{YEAR_DAYS}日）で割る")
    print(f"  腕 E の「伸びていない」= 買ってから {args.stall_days}営業日以上で含み {args.stall_gain:+.1f}% 以下")

    summary, by_year, trades = [], [], []
    for how in hows:
        log(f"百分位 {HOWS[how]} を付ける")
        base = rows_with_pct(oofs, how, args.min_hist)
        base = base[(base["Date"] >= lo) & (base["Date"] <= hi)].reset_index(drop=True)
        sels = selections(base, args.slots)
        codes = set()
        for _, _, picks in sels:
            codes |= set(picks["Code"].astype(str))
        px = PriceGrid(bars, bar_days, codes)
        use_arms = arms if how == "past" else ["A"]
        print(f"\n■ 百分位の取り方: {HOWS[how]}"
              + ("（その日より前の分布。先の値を見ない。本命）" if how == "past"
                 else "（OOF 全体の分布。実運用の追跡と同じ約束。参考。腕 A だけ）"))
        print(HEAD)
        for name, passed, picks in sels:
            is_ref = name.startswith("参考")
            print(f"  ▼ {name}（基準を満たした {len(passed)}件 / 年 {len(passed) / max((hi - lo).days, 1) * YEAR_DAYS:.0f}件）")
            for arm in (["A"] if is_ref else use_arms):
                t, counts, _ = simulate_arm(picks, px, arm=arm, slots=args.slots, unit=args.unit,
                                            stall_days=args.stall_days, stall_gain=args.stall_gain)
                done = t[t["status"].isin(DONE)].copy() if len(t) else empty_trades()
                curve = capital_curve(done, bar_days) if len(done) else pd.Series(dtype=float)
                s = stats(done, curve, lo, hi, slots=args.slots, counts=counts, n_pass=len(passed))
                label = ARMS[arm].split("（")[0] if not is_ref else "A なし"
                print(line(f"  {label}", s))
                summary.append({"how": HOWS[how], "pattern": name, "arm": arm, **s})
                for y in range(lo.year, hi.year + 1):
                    ylo = max(pd.Timestamp(f"{y}-01-01"), lo)
                    yhi = min(pd.Timestamp(f"{y}-12-31"), hi)
                    dy = done[done["buy_date"].dt.year == y] if len(done) else done
                    st = stats(dy, curve, ylo, yhi, slots=args.slots,
                               counts={"n_full": sum(1 for d in counts["full_days"] if d.year == y)},
                               n_pass=int((passed["Date"].dt.year == y).sum()))
                    by_year.append({"how": HOWS[how], "pattern": name, "arm": arm, "year": y, **st})
                if len(done):
                    tt = done.copy()
                    tt.insert(0, "arm", arm)
                    tt.insert(0, "pattern", name)
                    tt.insert(0, "how", HOWS[how])
                    trades.append(tt)
        print(f"  （取引 = 枠に入って終わった取引（乗り換えで売った玉も1件）。年間 = 取引 ÷ 年数。乗換 = 乗り換えの回数。"
              f"到達 = +{L.TAKE_PROFIT:.0f}% に届いた割合。")
        print(f"    必要資金 = 最大同時投下資本。回転 = 年間買付額 ÷ 必要資金。稼働 = 枠が埋まっていた営業日の割合。万円）")

        if how == "past" and len(use_arms) > 1:
            print(f"\n■ 乗り換えの損得（{HOWS[how]}）: 売った玉を「持ち切っていたら」規則どおりの出口で幾らだったか")
            print(SWAP_HEAD)
            for name, _, _ in sels:
                if name.startswith("参考"):
                    continue
                print(f"  ▼ {name}")
                for r in summary:
                    if r["how"] == HOWS[how] and r["pattern"] == name and r["arm"] != "A" and r["swap_n"]:
                        print(swap_line(f"  {ARMS[r['arm']].split('（')[0]}", r))
            print("  （実現 = 乗り換えで売った玉の収益率、持ち切り = その玉を規則どおり持っていたら、差 = 実現 − 持ち切り、")
            print("    切って損 = 持ち切りのほうが良かった割合、取り逃し = Σ(持ち切りの損益 − 実現の損益)、入替後 = 乗り換えで入れた玉の1取引、")
            print("    不明 = 持ち切りの出口まで日足が無い件数（差の計算から除く））")

    main_how = HOWS[hows[0]]
    print(f"\n■ 年ごと（{main_how}・腕 A、買った日の年。候補 = 基準を満たした件数）")
    for name, *_ in list(PATTERNS) + [REFERENCE]:
        rows = [r for r in by_year if r["how"] == main_how and r["pattern"] == name and r["arm"] == "A"]
        if not rows:
            continue
        print(f"\n  {name}")
        print(YEAR_HEAD)
        for r in rows:
            print(year_line(r["year"], r))

    print("\n■ 補足")
    for r in summary:
        if r["n"] and r["arm"] == "A":
            print(f"  {r['how']} / {r['pattern']}: 最悪の1取引 {_pct(r['worst'])}% / {_man(r['pnl_worst'])}万円、"
                  f"1取引の買付額 中央値 {_man(r['buy_median'])}万円・最大 {_man(r['buy_max'])}万円、"
                  f"枠が満杯で見送り {r['n_full']}件、寄り付かず {r['n_noopen']}件、"
                  f"途中の玉 {r['n_open']}件、回転（÷平均投下資本）{_pct(r['turnover_avg'], '{:.1f}')}、"
                  f"年あたり損益 ÷ 必要資金 {_pct(r['roi_max'], '{:+.1f}')}%")

    pd.DataFrame(summary).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_summary.csv"), index=False)
    pd.DataFrame(by_year).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_by_year.csv"), index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_trades.csv"), index=False)
    log(f"書いた: {OOF_DIR}/{args.prefix}_summary.csv / _by_year.csv / _trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
