"""Display-only margin-position estimates; no inputs to prediction/training.

Volume-at-daily-VWAP is a prior, not observed outstanding positions at price.
When contract amounts exist, exponential tilting matches their implied mean.
All prices/quantities are on the share basis at the requested price date.
"""
from __future__ import annotations

import datetime as dt
import math

WINDOWS = (60, 120, 250)
BIN_COUNT = 20
DAILY_MARGIN_START = '2026-09-25'


def _num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _positive(value):
    n = _num(value)
    return n if n is not None and n > 0 else None


def _weekly_publication(day, trading_days):
    d = dt.date.fromisoformat(day)
    monday = d + dt.timedelta(days=7 - d.weekday())
    days = sorted(dt.date.fromisoformat(str(x)[:10]) for x in (trading_days or []))
    after = [x for x in days if x >= monday]
    if len(after) >= 2:
        return after[1].isoformat()
    count = 0
    while True:
        if monday.weekday() < 5:
            count += 1
            if count == 2:
                return monday.isoformat()
        monday += dt.timedelta(days=1)


def _select_balance(rows, as_of, trading_days, observed_on):
    eligible = []
    for row in rows:
        day = str(row.get('Date') or '')[:10]
        if not day or day > as_of:
            continue
        pub = str(row.get('PubDate') or '')[:10] or None
        if pub:
            available, basis = pub, 'published'
        elif observed_on and day <= observed_on <= as_of:
            available, basis = observed_on, 'public_snapshot'
        elif day < DAILY_MARGIN_START:
            available, basis = _weekly_publication(day, trading_days), 'weekly_calendar_estimate'
        else:
            continue  # Daily publication cannot be recovered from the reference date.
        if day <= available <= as_of:
            eligible.append((day, available, row, pub, basis))
    return max(eligible, key=lambda x: x[:2]) if eligible else None


def _price_rows(quotes, as_of):
    rows = sorted((r for r in quotes if r.get('Date') and r['Date'] <= as_of),
                  key=lambda r: r['Date'])
    valid = [r for r in rows if _positive(r.get('C')) or _positive(r.get('AdjC'))]
    if not valid:
        return [], None, None
    latest = valid[-1]
    rows = [r for r in rows if r['Date'] <= latest['Date']]
    raw = _positive(latest.get('C'))
    adj = _positive(latest.get('AdjC'))
    anchor = adj / raw if raw and adj else 1.0
    result = []
    for r in rows:
        empty = dict(date=r['Date'], close=None, factor=None, volume=None,
                     vwap=None, source='unavailable')
        close, adjusted = _positive(r.get('C')), _positive(r.get('AdjC'))
        # Adjusted/raw ratios remove splits AFTER the price date as well.
        factor = adjusted / close / anchor if adjusted and close else None
        volume = _positive(r.get('Vo'))
        if factor is None and close:
            # Raw-only data are usable only if no known split lies ahead.
            later = [x for x in rows if x['Date'] > r['Date']]
            factors = [_positive(x.get('AdjFactor')) for x in later]
            if any(x is not None and x != 1 for x in factors):
                result.append(empty)
                continue
            factor = 1.0
        if factor is None:
            result.append(empty)
            continue
        price = close * factor
        turnover = _positive(r.get('Va'))
        if turnover and volume:
            vwap, source = turnover / volume * factor, 'daily_vwap'
        else:
            vwap, source = price, 'close_fallback'
        result.append(dict(date=r['Date'], close=price, factor=factor,
                           volume=volume / factor if volume else None,
                           vwap=vwap, source=source))
    return result, raw or adj / anchor, latest['Date']


def _tilt(prices, weights, target):
    lo, hi = min(prices), max(prices)
    if target < lo - 1e-8 or target > hi + 1e-8:
        return None
    if hi - lo < 1e-10:
        return [w / sum(weights) for w in weights]
    if abs(target - lo) < 1e-8 or abs(target - hi) < 1e-8:
        edge = lo if abs(target - lo) < 1e-8 else hi
        w = [v if abs(p - edge) < 1e-8 else 0 for p, v in zip(prices, weights)]
        return [v / sum(w) for v in w]
    xs = [(p - lo) / (hi - lo) for p in prices]
    logs = [math.log(w) for w in weights]

    def probabilities(lam):
        z = [w + lam * x for x, w in zip(xs, logs)]
        zmax = max(z)
        exps = [math.exp(v - zmax) for v in z]
        return [v / sum(exps) for v in exps]

    low, high = -1.0, 1.0
    while sum(p*w for p, w in zip(prices, probabilities(low))) > target:
        low *= 2
    while sum(p*w for p, w in zip(prices, probabilities(high))) < target:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if sum(p*w for p, w in zip(prices, probabilities(middle))) < target:
            low = middle
        else:
            high = middle
    return probabilities((low + high) / 2)


def _side(row, prefix, factor, current):
    reported = _num(row.get(prefix + 'Vol'))
    reported = reported if reported is not None and reported >= 0 else None
    amount = _num(row.get(prefix + 'Val'))
    amount = amount if amount is not None and amount >= 0 else None
    shares = reported / factor if reported is not None and factor else None
    average = amount / shares if amount and shares else None
    sign = 1 if prefix == 'Long' else -1
    return dict(reportedShares=reported, shares=shares, contractAmount=amount,
                averagePrice=average, averagePriceKind='contract_amount' if average else None,
                referencePnlRate=sign * (current / average - 1) if average else None)


def _profile_side(side, rows):
    qty, average = side['shares'], side['averagePrice']
    out = dict(status='ok', method=None, bins=[], estimatedAveragePrice=None,
               estimatedPnlRate=None)
    if qty is None:
        out['status'] = 'missing_quantity'
        return out
    if qty == 0:
        out['status'] = 'no_positions'
        return out
    if not rows:
        out['status'] = 'missing_volume'
        return out
    prices, weights = [r['vwap'] for r in rows], [r['volume'] for r in rows]
    if average is not None:
        probabilities = _tilt(prices, weights, average)
        if probabilities is None:
            out['status'] = 'mean_outside_window'
            return out
        out['method'] = 'mean_calibrated'
    else:
        probabilities = [w / sum(weights) for w in weights]
        out['method'] = 'volume_prior'
    lo, hi = min(prices), max(prices)
    width = (hi - lo) / BIN_COUNT if hi > lo else max(lo * .01, 1)
    start = lo if hi > lo else lo - width / 2
    groups = {}
    for price, probability in zip(prices, probabilities):
        idx = min(BIN_COUNT - 1, max(0, int((price - start) / width)))
        q, value = groups.get(idx, (0, 0))
        groups[idx] = q + qty * probability, value + qty * probability * price
    for idx, (q, value) in sorted(groups.items()):
        if q > 0:
            out['bins'].append(dict(low=start+idx*width, high=start+(idx+1)*width,
                                    price=value/q, shares=q))
    out['estimatedAveragePrice'] = sum(p*w for p, w in zip(prices, probabilities))
    return out


def build_supply_demand(quotes, margin_rows, as_of=None, trading_days=None, observed_on=None):
    """Build small 60/120/250-session profiles without fetching or changing model inputs.

    observed_on is ONLY for backfilling a published dashboard snapshot: evidence
    of availability by that day, not an invented actual publication date.
    """
    as_of = as_of or max((r.get('Date', '') for r in quotes), default='')
    prices, current, price_date = _price_rows(quotes, as_of)
    out = dict(schemaVersion=1, status='missing_price', asOf=price_date or as_of,
               currentPrice=current, balance=None, long=None, short=None, ratio=None,
               profiles={})
    if current is None:
        return out
    selected = _select_balance(margin_rows, price_date, trading_days, observed_on)
    if not selected:
        out['status'] = 'missing_balance'
        return out
    day, available, row, published, basis = selected
    reference = next((r for r in prices if r['date'] == day), None)
    factor = reference['factor'] if reference else None
    out['balance'] = dict(date=day, publishedOn=published, availableOn=available,
                          availabilityBasis=basis, shareAdjustment=factor)
    out['long'], out['short'] = _side(row, 'Long', factor, current), _side(row, 'Shrt', factor, current)
    if out['short']['shares'] and out['long']['shares'] is not None:
        out['ratio'] = out['long']['shares'] / out['short']['shares']
    # Do not allocate existing positions to prices traded AFTER the balance date.
    history = [r for r in prices if r['date'] <= day]
    for window in WINDOWS:
        selected_rows = history[-window:]
        traded = [r for r in selected_rows if r['volume']]
        profile = dict(sessions=len(selected_rows), tradedSessions=len(traded),
                       **{'from': selected_rows[0]['date'] if selected_rows else None,
                          'to': selected_rows[-1]['date'] if selected_rows else None},
                       closeFallbackSessions=sum(r['source'] == 'close_fallback' for r in traded))
        for side in ('long', 'short'):
            profile[side] = _profile_side(out[side], traded)
            avg = profile[side]['estimatedAveragePrice']
            profile[side]['estimatedPnlRate'] = ((current / avg - 1) * (1 if side == 'long' else -1)
                                                   if avg else None)
        out['profiles'][str(window)] = profile
    out['status'] = 'ok' if factor else 'missing_adjustment'
    return out
