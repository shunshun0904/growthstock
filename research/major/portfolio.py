"""
本ブレイク予測モデル: 運用に近い形の資金の増え方（ブランチ claude/major-breakout-model）。

運用者の依頼（2026-09-23）「aでお願いします」
  a = 月の買い付け数と同時に持つ数に上限を置いて、資金の増え方を窓ごと・区切り方3通りで測る
  （1銘柄ずつ全額だと、数回の当たり外れで結果が決まってしまうため）
  以前の方針「1ヶ月〜2カ月に一度、本当に「絶対これだ」と言えるものしか買いません」

資金の分け方（枠）
  資金を slots 等分した枠を用意し、1つの枠で1銘柄だけ持つ。空いている枠があり、その期間（暦の月、
  または2か月）の買い付け数が上限に達していなければ、条件を満たした銘柄をその枠の資金全部で買う。
  売ったら枠の資金がその分だけ増減し、次の買いに使う（枠ごとの複利）。同じ日に複数あれば priority の
  高い順。売った日の翌営業日から、その枠で次を買える
日々の資産
  空いている枠の資金 + 持っている銘柄の時価（枠の資金 × その日の終値 ÷ 買値。売った日は売値）。
  銘柄の n 日目は、買った日の営業日番号 + n − 1 の日に置く（売買が止まった日がある銘柄は少しずれる）
"""

from __future__ import annotations

import numpy as np


def simulate(buy_idx, period, held, ret, priority, slots: int, per_period: int) -> dict:
    """
    候補ごとの配列から、枠と買い付け数の上限の下で実際に買うものを決める。
      buy_idx   買う日の営業日番号
      period    買い付け数を数える期間の番号（同じ番号の中で per_period 件まで）
      held      持つ日数（買った日が1日目。売った日 = buy_idx + held − 1）
      ret       売ったときの収益
      priority  同じ日に複数あるときの順番（大きいほど先）
    戻り値: trades = [(枠, 買った日, 持った日数, 収益, 候補の番号, 買う前の枠の資金), ...]（買った順）、
            skipped_slot = 枠が空いていなくて見送った数、skipped_quota = 期間の上限で見送った数
    """
    buy_idx = np.asarray(buy_idx, dtype=np.int64)
    order = np.lexsort((-np.asarray(priority, dtype=float), buy_idx))
    free_after = np.full(slots, -1, dtype=np.int64)          # この日より後なら、その枠で買える（売った日）
    cap = np.full(slots, 1.0 / slots)
    used: dict = {}
    trades = []
    skipped_slot = skipped_quota = 0
    for i in order:
        d = int(buy_idx[i])
        p = period[i]
        if used.get(p, 0) >= per_period:
            skipped_quota += 1
            continue
        free = np.flatnonzero(free_after < d)
        if not len(free):
            skipped_slot += 1
            continue
        s = int(free[0])
        h = int(held[i])
        trades.append((s, d, h, float(ret[i]), int(i), float(cap[s])))
        cap[s] *= 1.0 + float(ret[i])
        free_after[s] = d + h - 1
        used[p] = used.get(p, 0) + 1
    return {"trades": trades, "skipped_slot": skipped_slot, "skipped_quota": skipped_quota, "slots": slots}


def _ffill(x: np.ndarray, first: float) -> np.ndarray:
    x = np.array(x, dtype=float)
    if not np.isfinite(x[0]):
        x[0] = first
    for j in range(1, len(x)):
        if not np.isfinite(x[j]):
            x[j] = x[j - 1]
    return x


def equity(res: dict, path: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """
    lo〜hi 営業日の毎日の資産（最初の資金 = 1）。path は候補ごとの 1〜N 日目の終値 ÷ 買値（欠けた日は NaN）。
    lo より前に買ったもの・hi より後の日は数えない（hi の後に売るものは hi の時価で止める）。
    """
    n = hi - lo + 1
    slots = res["slots"]
    eq = np.zeros(n)
    for s in range(slots):
        val = np.full(n, 1.0 / slots)
        for (s_, d, h, r, i, c) in sorted(t for t in res["trades"] if t[0] == s):
            seg = _ffill(path[i, :h], 1.0)
            seg[h - 1] = 1.0 + r
            a = d - lo
            if a >= n:
                continue
            b = min(a + h, n)
            if a < 0:
                raise ValueError("lo より前に買った取引がある")
            val[a:b] = c * seg[:b - a]
            val[b:] = c * (1.0 + r)
        eq += val
    return eq


def metrics(eq: np.ndarray, per_year: float = 245.0) -> dict:
    """
    資産の倍率・年率・最大の下げ（日々の時価で測る）。eq は最初の資金を 1 とした毎日の資産で、
    その前の日（まだ何も買っていない日）の 1 を起点にする。
    """
    eq = np.concatenate([[1.0], np.asarray(eq, dtype=float)])
    mult = float(eq[-1])
    years = max((len(eq) - 1) / per_year, 1e-9)
    peak = np.maximum.accumulate(eq)
    return {"mult": mult, "cagr": mult ** (1.0 / years) - 1.0, "mdd": float((eq / peak - 1.0).min()),
            "years": years}


def window_returns(eq: np.ndarray, lo: int, bounds: list) -> list:
    """窓ごとの資産の増減。bounds は (最初の日, 最後の日) の営業日番号。前の日の終わりから最後の日まで。"""
    out = []
    for a, b in bounds:
        ia, ib = a - lo, b - lo
        if ia < 0 or ib >= len(eq):
            out.append(float("nan"))
            continue
        base = eq[ia - 1] if ia > 0 else 1.0
        out.append(float(eq[ib] / base - 1.0))
    return out
