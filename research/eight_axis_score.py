"""
現行ダッシュボードの8軸総合スコア（src/lib/scoring.js）を Python で再現する。

目的は「既存スコアにブレイクアウトの予測力があるか」をベースラインとして
検証すること。数式は src/lib/scoring.js と1対1で対応させる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOLUME_DECAY = {"mega": 1.0, "high": 1.0, "moderate": 0.85,
                "low": 0.65, "cap_low": 0.7, "none": 0.35}


def _norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) * 10.0, 0, 10)


def _technical(r):
    return np.select(
        [r >= 98, r >= 90, r >= 80],
        [10.0, 8.0 + (r - 90) / 8 * 1.5, 6.0 + (r - 80) / 10 * 2.0],
        default=np.clip(r / 80 * 6.0, 0, None),
    )


def _credit(c):
    return np.select(
        [c <= 1.0, c <= 3.0, c <= 10.0],
        [10.0, 10.0 - (c - 1.0) / 2.0 * 3.0, 7.0 - (c - 3.0) / 7.0 * 5.0],
        default=np.clip(2.0 - (c - 10.0) / 10.0 * 2.0, 0, None),
    )


def _institutional_decay(trading_value, market_cap):
    tier = np.select(
        [trading_value >= 30, trading_value >= 10, trading_value >= 5, trading_value >= 1],
        ["mega", "high", "moderate", "low"], default="none",
    )
    tier = np.where(np.isfinite(market_cap) & (market_cap < 100), "cap_low", tier)
    return np.vectorize(lambda t: VOLUME_DECAY.get(t, 1.0))(tier)


def eight_axis_total(df: pd.DataFrame) -> np.ndarray:
    """
    8軸スコアの平均。欠測軸は平均から除外する
    （scoring.js の totalScore と同じ扱い）。
    """
    tv = df["tv_ma20"].to_numpy(dtype=float)
    cap = df["market_cap"].to_numpy(dtype=float)
    decay = _institutional_decay(tv, cap)
    vol_base = np.clip(5.0 + (df["volume_trend"].to_numpy(dtype=float) - 100) / 100 * 5.0, 0, 10)

    axes = np.vstack([
        _norm(df["eps_growth_q0"].to_numpy(dtype=float), 0, 50),
        _norm(df["sales_growth_q0"].to_numpy(dtype=float), 0, 40),
        _norm(df["ROE_q0"].to_numpy(dtype=float), 5, 25),
        _norm(df["op_margin_q0"].to_numpy(dtype=float), 0, 20),
        _technical(df["r_high"].to_numpy(dtype=float)),
        np.clip(vol_base * decay, 0, 10),
        _credit(df["credit_ratio"].to_numpy(dtype=float)),
        np.clip(5.0 + df["progress_vs_base"].to_numpy(dtype=float) / 2.0, 0, 10),
    ])
    with np.errstate(invalid="ignore"):
        return np.nanmean(axes, axis=0)
