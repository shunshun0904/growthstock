"""Causal within-security features and independent research-only presets."""
from __future__ import annotations
import numpy as np
import pandas as pd
import features as F

CONTROLS = ["roic_conservative_tax30", "roic_operating_tax30", "wacc", "actual_tax", "annual_age",
            "cash_identified", "revision_unverified", "debt_reviewed", "equity_reviewed", "valuation_data_quality"]
GAPS = ["valuation_gap", "valuation_log_gap", "ev_valuation_gap", "roic_wacc_spread"]
TIME = ["valuation_gap_zscore_252", "valuation_gap_pct_252", "valuation_gap_change_5d",
        "valuation_gap_change_20d", "valuation_gap_change_60d", "valuation_gap_ma_20", "valuation_gap_ma_60",
        "valuation_gap_vol_60", "theoretical_price_change_20d", "wacc_change_20d", "roic_change_latest",
        "days_since_roic_update", "valuation_price_log_change_20d", "valuation_value_log_change_20d",
        "valuation_gap_log_change_20d"]
MARKET_CONTROLS = ["control_price_zscore_252", "control_price_pct_252", "control_price_return_5d",
                   "control_price_return_20d", "control_price_return_60d", "control_price_vol_60",
                   "control_forecast_per_zscore_252", "control_pbr_zscore_252"]


def compare_with_market(fundamental_value_per_share, market_price):
    """Terminal comparison only; this module is NOT imported by EVA valuation."""
    value, price = np.broadcast_arrays(np.asarray(fundamental_value_per_share, float), np.asarray(market_price, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = price / value
    return np.where(np.isfinite(price + value) & (price > 0) & (value > 0), gap, np.nan)


def column_sets():
    """Current production baseline; do not register or change F.DEFAULT_PRESET."""
    base = F.columns(F.DEFAULT_PRESET)
    unique = lambda x: list(dict.fromkeys(x))
    b = unique(base + CONTROLS)
    c = unique(b + GAPS)
    d = unique(c + TIME)
    return {"A": base, "B": b, "C": c, "D": d, "P": d, "M": unique(b + MARKET_CONTROLS)}


def market_control_features(frame):
    """Same-window price/PER/PBR controls to test a momentum-proxy explanation."""
    out = pd.DataFrame(np.nan, index=frame.index, columns=MARKET_CONTROLS)
    for _, group in frame.groupby("Code", sort=False):
        g = group.sort_values("Date")
        p = pd.to_numeric(g.price_adjusted, errors="coerce").where(lambda x: x.gt(0))
        window = p.rolling(252, min_periods=126)
        out.loc[g.index, "control_price_zscore_252"] = (p - window.mean()) / window.std(ddof=1).where(lambda x: x.gt(1e-12))
        out.loc[g.index, "control_price_pct_252"] = window.apply(
            lambda a: (np.count_nonzero(a[np.isfinite(a)] < a[-1]) + (np.count_nonzero(a[np.isfinite(a)] == a[-1]) + 1) / 2)
            / np.count_nonzero(np.isfinite(a)) if np.isfinite(a[-1]) else np.nan, raw=True)
        for n in (5, 20, 60):
            out.loc[g.index, f"control_price_return_{n}d"] = p / p.shift(n) - 1
        out.loc[g.index, "control_price_vol_60"] = p.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=1)
        per = pd.to_numeric(g.get("market_per_forecast", pd.Series(np.nan, index=g.index)), errors="coerce").where(lambda x: x.gt(0))
        bps = pd.to_numeric(g.get("bps_reported", pd.Series(np.nan, index=g.index)), errors="coerce").where(lambda x: x.gt(0))
        pbr = g.price / bps
        for name, s in [("forecast_per", per), ("pbr", pbr)]:
            r = s.rolling(252, min_periods=126)
            out.loc[g.index, f"control_{name}_zscore_252"] = (s - r.mean()) / r.std(ddof=1).where(lambda x: x.gt(1e-12))
    return out


def time_features(frame, z_window=252, min_z_observations=126):
    """Market-session rows, current observation included; never backfill.

    Z scores use sample SD (ddof=1), requiring >=126 valid observations among
    the trailing 252 session rows. Zero SD => missing. Percentile is average
    rank/n in the same window (ties retained). MAs/60-day SD require complete
    20/60 observations. Differences are session lags, not last-valid lags.
    Price/value log changes use a COMMON split-adjusted share basis.
    """
    if min_z_observations < 2 or z_window < min_z_observations:
        raise ValueError("Invalid historical window")
    if frame.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate security/session observations")
    d = frame.copy()
    d["Date"] = pd.to_datetime(d.Date)
    d["_vg_order"] = np.arange(len(d))
    pieces = []
    for _, group in d.groupby("Code", sort=False):
        g = group.sort_values("Date").copy()
        gap = pd.to_numeric(g.valuation_gap, errors="coerce").where(lambda s: s.gt(0))
        g["valuation_log_gap"] = np.log(gap)
        roll = gap.rolling(z_window, min_periods=min_z_observations)
        sd = roll.std(ddof=1)
        g["valuation_gap_zscore_252"] = (gap - roll.mean()) / sd.where(sd.gt(1e-12))
        def percentile(a):
            if not np.isfinite(a[-1]):
                return np.nan
            observed = a[np.isfinite(a)]
            return (np.count_nonzero(observed < a[-1]) + (np.count_nonzero(observed == a[-1]) + 1) / 2) / len(observed)
        g["valuation_gap_pct_252"] = roll.apply(percentile, raw=True)
        for lag in (5, 20, 60):
            g[f"valuation_gap_change_{lag}d"] = gap.diff(lag)
        for n in (20, 60):
            g[f"valuation_gap_ma_{n}"] = gap.rolling(n, min_periods=n).mean()
        g["valuation_gap_vol_60"] = gap.rolling(60, min_periods=60).std(ddof=1)
        p = pd.to_numeric(g.price_adjusted, errors="coerce").where(lambda s: s.gt(0))
        value = pd.to_numeric(g.theoretical_price_adjusted, errors="coerce").where(lambda s: s.gt(0))
        g["theoretical_price_change_20d"] = value / value.shift(20) - 1
        g["wacc_change_20d"] = g.wacc.diff(20)
        g["valuation_price_log_change_20d"] = np.log(p / p.shift(20))
        g["valuation_value_log_change_20d"] = np.log(value / value.shift(20))
        g["valuation_gap_log_change_20d"] = np.log(gap / gap.shift(20))
        if "action_uncertainty" in g:
            unverified_span = g.action_uncertainty.ne(g.action_uncertainty.shift(20))
            g.loc[unverified_span, ["theoretical_price_change_20d", "valuation_price_log_change_20d",
                                    "valuation_value_log_change_20d", "valuation_gap_log_change_20d"]] = np.nan
        if "avail_date" in g:
            event = g.avail_date.notna() & g.avail_date.ne(g.avail_date.shift())
            changed = g.loc[event, "roic"].diff()
            update_change = pd.Series(np.nan, index=g.index)
            update_change.loc[event] = changed
            # Carry the CHANGE OF THE CURRENT EVENT, including its missingness.
            groups = event.cumsum()
            g["roic_change_latest"] = update_change.groupby(groups).transform(lambda s: s.iloc[0])
            g["days_since_roic_update"] = (g.Date - pd.to_datetime(g.avail_date)).dt.days
        else:
            g["roic_change_latest"] = np.nan
            g["days_since_roic_update"] = np.nan
        pieces.append(g)
    if not pieces:
        for key in set(GAPS + TIME):
            if key not in d:
                d[key] = np.nan
        return d.drop(columns="_vg_order")
    return pd.concat(pieces).sort_values("_vg_order").drop(columns="_vg_order").replace([np.inf, -np.inf], np.nan)


def shuffle_within_date(frame, columns, seed=20261115):
    """Joint permutation by date AND availability mask, never permute targets."""
    out = frame.copy()
    columns = list(columns)
    if not columns:
        return out
    masks = frame[columns].notna().apply(lambda row: tuple(row), axis=1)
    rng = np.random.default_rng(seed)
    for ix in frame.assign(_mask=masks).groupby(["Date", "_mask"], sort=True).indices.values():
        out.iloc[ix, out.columns.get_indexer(columns)] = frame.iloc[rng.permutation(ix)][columns].to_numpy()
    np.testing.assert_array_equal(out[columns].isna(), frame[columns].isna())
    return out
