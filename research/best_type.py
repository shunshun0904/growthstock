#!/usr/bin/env python3
"""
最も当たる「型」に当てはまるかの判定（画面の候補に付ける）。

実験60（docs/MODEL_TENDENCIES.md §5）で、5モデルの上位10% を5つの型に分けたとき、最も当たった型は
「小型・78週高値の近く・終値が高値側・PBR 低め」（OOF の正例率 33%、最も外れる型は 20%）だった。
運用者の依頼（2026-10-03）「発火した銘柄が、この条件を満たしているかどうかも、ダッシュボードに表示してほしい」。

型は「同じ日の候補の中での百分位」で決めたものなので、判定も同じ日の候補の中で行う。
4条件それぞれについて、その日の候補の中央値より望ましい側にあれば ○。中央値ちょうどは ×。
候補が MIN_IN_DAY 件未満の日は、中央値が意味を持たないので判定しない（None）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: (キー, 列, 望ましい側, 画面の説明)。順は画面の並び
CONDITIONS = (
    ("small", "log_trading_value", "low", "小型（20日平均売買代金が、その日の候補の中央値より少ない）"),
    ("nearHigh", "r_high", "high", "78週高値の近く（高値に対する終値の位置が中央値より上）"),
    ("closeHigh", "close_position", "high", "終値が高値側（その日の値幅の中の終値の位置が中央値より上）"),
    ("lowPbr", "book_yield", "high", "PBR 低め（PBR の逆数が中央値より上）"),
)
KEYS = tuple(k for k, _, _, _ in CONDITIONS)
MIN_IN_DAY = 3
N_CONDITIONS = len(CONDITIONS)


def within_day_pct(df: pd.DataFrame, cols) -> pd.DataFrame:
    """同じ日の候補の中での百分位。(順位 − 0.5) ÷ その日の件数。偏りが無ければ平均 0.5。NaN はそのまま。"""
    g = df.groupby("Date")[list(cols)]
    return (g.rank(method="average") - 0.5) / g.transform("count")


def profiles(cand: pd.DataFrame) -> List[Dict]:
    """
    候補の表（Date と CONDITIONS の列を持つ）から、行の順に判定の dict を返す。
      {"small": True/False/None, ..., "n": 当てはまった条件の数（判定しない日は None）,
       "nInDay": その日の候補数, "pct": {キー: その日の中での百分位（0〜1、小数2桁）}}
    列が無い・値が無い条件は None（数えない）。
    """
    cols = [c for _, c, _, _ in CONDITIONS]
    missing = [c for c in cols if c not in cand.columns]
    df = cand[["Date"] + [c for c in cols if c not in missing]].copy()
    for c in missing:
        df[c] = np.nan
    df["Date"] = pd.to_datetime(df["Date"])
    pct = within_day_pct(df, cols)
    n_day = df.groupby("Date")["Date"].transform("size")
    out: List[Dict] = []
    for i in range(len(df)):
        n_in_day = int(n_day.iloc[i])
        rec: Dict = {"nInDay": n_in_day, "pct": {}}
        matched: Optional[int] = 0
        for key, col, side, _ in CONDITIONS:
            p = pct[col].iloc[i]
            rec["pct"][key] = None if not np.isfinite(p) else round(float(p), 2)
            if n_in_day < MIN_IN_DAY or not np.isfinite(p):
                rec[key] = None
                continue
            ok = bool(p > 0.5) if side == "high" else bool(p < 0.5)
            rec[key] = ok
        if n_in_day < MIN_IN_DAY:
            matched = None
        else:
            flags = [rec[k] for k in KEYS if rec[k] is not None]
            matched = int(sum(flags)) if flags else None
        rec["n"] = matched
        rec["total"] = N_CONDITIONS
        out.append(rec)
    return out
