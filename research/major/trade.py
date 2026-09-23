"""
本ブレイク予測モデルの売買の模擬（ブランチ claude/major-breakout-model）。

運用者の方針（2026-09-23）
  「こちらの本ブレイクモデルは「数打てば当たる作戦」ではないので、1ヶ月〜2カ月に一度、
   本当に「絶対これだ」と言えるものしか買いません。その代わり、半年後には資産が2倍に
   なっているので。ですので、3つのgbdtモデルで95%（上位5%）の&条件を満たすものという
   銘柄で購入戦略で行きたいです。売るのは、購入時価格の2倍に達したらその時点で売り、
   達さない場合は120営業日保有し、120営業日終値で売却」

約定の仮定
  買う  ブレイク日の翌営業日の寄り（AdjO[t+1]）。買った日を1日目と数える
  2倍   場中の高値が買値の2倍に届いたら、ちょうど2倍で売る（指値。窓を開けて飛び越えても
        2倍ちょうどで数える = 控えめ）
  期限  届かなければ120営業日目の終値。途中で上場廃止したら最後の終値
  手数料・税は入れない
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = 2.0      # 買値の何倍で売るか
DAYS = 120        # 届かなければ何営業日目の終値で売るか（買った日が1日目）
WHY = {1: "2倍", 2: "期限", 3: "上場廃止"}


def forward_hc(b: pd.DataFrame, keys: pd.DataFrame, days: int = DAYS, delist_gap_days: int = 30) -> dict:
    """
    keys の各行（基準日 t）について、1〜days 日目の高値・終値と買値、使える足の数、
    売買を数えられるか（tradable）を返す。
    tradable = days 日目まで上場している、または途中で上場廃止した（その銘柄の最後の足が
    データの最終日より delist_gap_days 日以上前）。まだ days 日たっていない直近の行は False。
    """
    pos = b[["Code", "Date"]].assign(_i=np.arange(len(b)))
    i0 = keys[["Code", "Date"]].merge(pos, on=["Code", "Date"], how="left")["_i"]
    if i0.isna().any():
        raise SystemExit(f"bars に無い基準日が {int(i0.isna().sum())}件")
    i0 = i0.to_numpy(dtype=np.int64)
    left = b.groupby("Code", sort=False).cumcount(ascending=False).to_numpy()[i0]
    step = np.arange(1, days + 1)
    ok = step[None, :] <= left[:, None]
    idx = np.where(ok, i0[:, None] + step[None, :], 0)
    out = {}
    for name, col in (("H", "AdjH"), ("C", "AdjC")):
        a = b[col].to_numpy(dtype=float)[idx]
        a[~ok] = np.nan
        out[name] = a
    o = b["AdjO"].to_numpy(dtype=float)
    out["entry"] = np.where(left >= 1, o[np.minimum(i0 + 1, len(o) - 1)], np.nan)
    out["n"] = np.minimum(left, days)
    last = pd.to_datetime(b.groupby("Code")["Date"].max())
    end = pd.to_datetime(b["Date"]).max()
    code_last = pd.to_datetime(keys["Code"].map(last)).to_numpy()
    delisted = code_last < np.datetime64(end - pd.Timedelta(days=delist_gap_days))
    out["tradable"] = np.isfinite(out["entry"]) & ((left >= days) | delisted)
    return out


def exit_target(P: dict, target: float = TARGET, days: int = DAYS, on: str = "H") -> tuple:
    """
    戻り値: 収益, 持った日数, 理由（1 = target倍, 2 = 期限の終値, 3 = 上場廃止の最後の終値）。
    on="H" は場中の高値で判定してちょうど target 倍で売る。on="C" は終値で判定してその終値で売る。
    """
    e = P["entry"]
    X = P[on][:, :days]
    with np.errstate(invalid="ignore"):
        hit = X >= (e * target)[:, None]
    any_ = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    C = P["C"][:, :days]
    # 期限（または上場廃止）の日の終値。売買の無い日は、その前の終値
    valid = np.isfinite(C)
    last_i = np.where(valid.any(axis=1), days - 1 - np.argmax(valid[:, ::-1], axis=1), -1)
    lastc = np.where(last_i >= 0, C[np.arange(len(e)), np.maximum(last_i, 0)], np.nan)
    if on == "H":
        hit_ret = np.full(len(e), target - 1.0)
    else:
        hit_ret = C[np.arange(len(e)), first - 1] / e - 1.0
    ret = np.where(any_, hit_ret, lastc / e - 1.0)
    held = np.where(any_, first, np.where(P["n"] >= days, days, P["n"]))
    why = np.where(any_, 1, np.where(P["n"] >= days, 2, 3)).astype(np.int8)
    return ret, held.astype(float), why


def select_prior(frame: pd.DataFrame, cols: list, pct: float, min_prev: int = 500) -> np.ndarray:
    """
    cols のすべてで、点数が「それより前の窓の点数」の pct% 点を超えた行を True にする。
    前の窓の行が min_prev 未満の窓（最初の窓など）は選ばない。frame は fold 列を持つ。
    """
    ok_all = np.zeros(len(frame), dtype=bool)
    fold = frame["fold"].to_numpy()
    for f in np.unique(fold):
        prev = frame[fold < f]
        if len(prev) < min_prev:
            continue
        cur = fold == f
        ok = cur.copy()
        for c in cols:
            thr = np.nanpercentile(prev[c].to_numpy(dtype=float), pct)
            with np.errstate(invalid="ignore"):
                ok &= frame[c].to_numpy(dtype=float) > thr
        ok_all |= ok
    return ok_all


def one_at_a_time(buy_idx: np.ndarray, held: np.ndarray, ret: np.ndarray, priority: np.ndarray,
                  per_year: float = 245.0) -> dict:
    """
    1銘柄ずつ買う（持っている間は次を買わない）。同じ日に複数出たら priority の高いものを1つ。
    buy_idx は買う日の営業日の番号。売った日の翌営業日から次を買える。資金は全額で、損益は複利。
    """
    if len(buy_idx) == 0:
        return {"trades": 0}
    order = np.lexsort((-priority, buy_idx))
    eq, peak, mdd, free = 1.0, 1.0, 0.0, -1
    taken, first_buy, last_exit = [], None, None
    for i in order:
        if buy_idx[i] <= free:
            continue
        eq *= 1.0 + ret[i]
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        free = buy_idx[i] + int(held[i]) - 1                 # 売った日
        taken.append(i)
        first_buy = buy_idx[i] if first_buy is None else first_buy
        last_exit = free
    years = max((last_exit - first_buy + 1) / per_year, 1e-9)
    r = ret[taken]
    return {"trades": len(taken), "multiple": eq, "cagr": eq ** (1.0 / years) - 1.0, "years": years,
            "worst": float(r.min()), "hit": float((r >= TARGET - 1.0 - 1e-12).mean()), "mdd": mdd,
            "taken": np.asarray(taken)}


def summarize(ret: np.ndarray, held: np.ndarray, why: np.ndarray) -> dict:
    if len(ret) == 0:
        return {"n": 0}
    return {"n": int(len(ret)), "hit": float((why == 1).mean()), "mean": float(np.mean(ret)),
            "median": float(np.median(ret)), "win": float((ret > 0).mean()),
            "p10": float(np.percentile(ret, 10)), "worst": float(np.min(ret)),
            "days": float(np.mean(held)), "sum": float(np.sum(ret)), "delist": int((why == 3).sum())}
