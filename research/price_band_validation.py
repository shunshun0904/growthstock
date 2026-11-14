"""Mean-entry-price supplementary validation, independent of production profiles.

Contract amounts/targets are never accepted by the feature/estimation functions.
The caller supplies training labels only to select_coefficient; there is no test
grid search. Financial pairs are never assembled from different disclosures.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def estimate_means(prices, volumes, turnovers, coefficients):
    p, v, r = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (prices, volumes, turnovers))
    if p.shape != v.shape or p.shape != r.shape:
        raise ValueError('window shapes differ')
    if not all(np.isfinite(x).all() for x in (p, v, r)) or (v < 0).any() or (r < 0).any():
        raise ValueError('invalid estimation inputs')
    exposure = np.cumsum(r[:, ::-1], axis=1)[:, ::-1] - r
    result = []
    for coefficient in coefficients:
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError('invalid coefficient')
        weights = v * np.exp(-coefficient * exposure)
        totals = weights.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError('zero total weight')
        result.append((weights * p).sum(axis=1) / totals)
    return np.column_stack(result)


def rmspe(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all() or (a <= 0).any():
        raise ValueError('RMSPE requires matched finite records and positive actual means')
    return float(100 * np.sqrt(np.mean(np.square((p - a) / a))))


def split_dates(dates, fraction=.7):
    ordered = sorted(set(dates))
    cut = math.floor(fraction * len(ordered))
    if cut < 1 or cut >= len(ordered):
        raise ValueError('insufficient dates for fixed chronological split')
    return ordered[:cut], ordered[cut:]


def select_coefficient(training_actual, training_predictions, coefficients):
    candidates = sorted(zip(coefficients, np.asarray(training_predictions).T), key=lambda x: x[0])
    curve = [{'coefficient': float(k), 'training_rmspe': rmspe(training_actual, pred)}
             for k, pred in candidates]
    best = min(curve, key=lambda row: (row['training_rmspe'], row['coefficient']))
    ties = [row['coefficient'] for row in curve if row['training_rmspe'] == best['training_rmspe']]
    return best['coefficient'], pd.DataFrame(curve), ties


def point_in_time_shares(dates, adjustment_factors, filings):
    """Return native-day net shares using disclosures available at 23:59 JST.

    dates must be the exchange calendar, not a stock's sparse observations.
    adjustment_factors are daily factors on that calendar. A missing factor on
    a required measurement-to-observation chain makes that share count unknown.
    """
    days = pd.DatetimeIndex(dates).normalize()
    factors = np.asarray(adjustment_factors, float)
    valid_factor = np.isfinite(factors) & (factors > 0)
    cumulative = np.cumprod(np.where(valid_factor, factors, 1.0))
    missing = np.cumsum(~valid_factor)
    out = pd.DataFrame({'shares': np.nan, 'disclosure_date': pd.NaT,
                        'measurement_date': pd.NaT, 'share_reason': 'no_prior_valid_share_pair'}, index=days)
    f = filings.copy()
    if f.empty:
        return out
    for col in ('DiscDate', 'CurPerEn'):
        f[col] = pd.to_datetime(f[col], errors='coerce').dt.normalize()
    for col in ('ShOutFY', 'TrShFY'):
        f[col] = pd.to_numeric(f[col], errors='coerce')
    valid = (np.isfinite(f.ShOutFY) & np.isfinite(f.TrShFY) &
             (f.TrShFY >= 0) & (f.ShOutFY > f.TrShFY) &
             f.CurPerEn.notna() & (f.CurPerEn <= f.DiscDate))
    f = f.loc[valid].copy()
    sort_fields = ['DiscDate'] + (['DiscTime'] if 'DiscTime' in f else []) + (['DiscNo'] if 'DiscNo' in f else [])
    f = f.sort_values(sort_fields, kind='stable').drop_duplicates('DiscDate', keep='last')
    if f.empty:
        return out
    selected = np.searchsorted(f.DiscDate.to_numpy(), days.to_numpy(), side='right') - 1
    has_pair = selected >= 0
    indices = np.flatnonzero(has_pair)
    rows = f.iloc[selected[has_pair]]
    measured = np.searchsorted(days.to_numpy(), rows.CurPerEn.to_numpy(), side='right') - 1
    covered = measured >= 0
    safe = np.maximum(measured, 0)
    covered &= missing[indices] == missing[safe]
    covered &= valid_factor[indices]
    out.iloc[indices, out.columns.get_loc('disclosure_date')] = rows.DiscDate.to_numpy()
    out.iloc[indices, out.columns.get_loc('measurement_date')] = rows.CurPerEn.to_numpy()
    out.iloc[indices, out.columns.get_loc('share_reason')] = 'adjustment_chain_unavailable'
    good = indices[covered]
    values = (rows.ShOutFY.to_numpy() - rows.TrShFY.to_numpy()) * cumulative[safe] / cumulative[indices]
    out.iloc[good, out.columns.get_loc('shares')] = values[covered]
    out.iloc[good, out.columns.get_loc('share_reason')] = 'available'
    return out


def make_window(bars, lookback=60, require_shares=True):
    if len(bars) != lookback or ('present' in bars and not bars.present.all()):
        return None, 'incomplete_60_day_history'
    v = pd.to_numeric(bars.Vo, errors='coerce').to_numpy(float)
    a = pd.to_numeric(bars.Va, errors='coerce').to_numpy(float)
    f = pd.to_numeric(bars.AdjFactor, errors='coerce').to_numpy(float)
    s = pd.to_numeric(bars.shares, errors='coerce').to_numpy(float) if require_shares else None
    if not np.isfinite(v).all() or (v < 0).any():
        return None, 'missing_or_invalid_volume'
    if not (v > 0).any():
        return None, 'zero_total_volume'
    active = v > 0
    if not np.isfinite(a[active]).all() or (a[active] <= 0).any():
        return None, 'missing_or_invalid_vwap'
    if not np.isfinite(f).all() or (f <= 0).any():
        return None, 'missing_split_adjustment'
    if require_shares and (not np.isfinite(s).all() or (s <= 0).any()):
        return None, 'missing_point_in_time_net_shares'
    cumulative = np.cumprod(f)
    to_reference = cumulative[-1] / cumulative
    prices = np.divide(a, v, out=np.zeros_like(a), where=active) * to_reference
    volumes = v / to_reference
    turnover = v / s if require_shares else np.zeros_like(v)
    return {'price': prices, 'volume': volumes, 'turnover': turnover,
            'price_min': float(prices[active].min()), 'price_max': float(prices[active].max()),
            'split_in_window': bool(np.any(f[1:] != 1)),
            'zero_volume_days': int((~active).sum())}, None


def paired_block_interval(daily, block_length=5, replicates=1000, seed=20261114, coverage=.95):
    d = daily.sort_values('Date')
    count = len(d)
    if count < block_length:
        return {'lower': None, 'upper': None, 'status': 'fewer_dates_than_block_length'}
    arrays = d[['n', 'baseline_sse', 'candidate_sse']].to_numpy(float)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        starts = rng.integers(0, count - block_length + 1, size=math.ceil(count / block_length))
        indices = np.concatenate([np.arange(x, x + block_length) for x in starts])[:count]
        n, baseline, candidate = arrays[indices].sum(axis=0)
        differences.append(float(100 * (np.sqrt(candidate / n) - np.sqrt(baseline / n))))
    alpha = (1 - coverage) / 2
    lower, upper = np.quantile(differences, [alpha, 1 - alpha])
    return {'lower': float(lower), 'upper': float(upper), 'status': 'reference_only',
            'replicates': replicates, 'block_length': block_length, 'seed': seed,
            'difference': 'candidate_minus_baseline_RMSPE_percentage_points',
            'resampled_differences': differences}
