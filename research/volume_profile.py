"""Price-band volume from verified intraday sessions."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

JST = timezone(timedelta(hours=9))
WINDOWS = (1, 5, 20)
SOURCES = ("tick", "minute")


def normalize_code(value):
    code = str(value).strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4,5}", code):
        raise ValueError("4桁または5桁の銘柄コードが必要です。")
    return code + "0" if len(code) == 4 else code


@dataclass(frozen=True)
class Quote:
    low: float | None
    high: float | None
    close: float | None
    volume: float | None
    turnover: float | None
    factor: float | None
    action: str | None
    observed_at: datetime

    @property
    def no_trades(self):
        return self.volume in (None, 0) and self.turnover in (None, 0) and all(
            x is None for x in (self.low, self.high, self.close)
        )

    @property
    def valid(self):
        values = (self.low, self.high, self.close, self.volume, self.turnover)
        return (all(x is not None and math.isfinite(x) and x > 0 for x in values)
                and self.low <= self.close <= self.high)


@dataclass(frozen=True)
class Intraday:
    # Columns are representative price, shares, and actual turnover. Tick
    # prices are observed. Each minute's volume is placed at Va / Vo.
    points: np.ndarray
    observed_at: datetime


def lookback(days, end, count):
    """Last exchange sessions including end, when end itself is a session."""
    stop = bisect_right(days, end)
    return days[max(0, stop - count):stop] if stop and days[stop - 1] == end else []


def reconciles(points, quote):
    """Compare like-for-like, unadjusted TSE regular-session totals."""
    if quote.no_trades:
        return len(points) == 0
    if not quote.valid or not len(points):
        return False
    price, qty, value = points.T
    return (np.isfinite(points).all() and (points > 0).all()
            and abs(qty.sum() - quote.volume) < .01
            and abs(value.sum() - quote.turnover) <= 1
            and price.min() >= quote.low - 1e-6 and price.max() <= quote.high + 1e-6)


def share_factors(dates, quotes):
    """Price/share multipliers between a past session and the anchor session.

    A missing factor or unknown non-unit action inside the window makes the
    basis unknown. An action on the first date affects no included session.
    """
    combined = 1.
    factors = {}
    for day in reversed(dates):
        factors[day] = combined
        if day == dates[0]:
            break
        quote = quotes[day]
        factor = quote.factor
        if factor is None or not math.isfinite(factor) or factor <= 0:
            return None
        if factor != 1 and quote.action not in ("1", "2"):
            return None
        combined *= factor
        if not math.isfinite(combined) or combined <= 0:
            return None
    return factors


def bin_grid(low, high):
    """A 1/2/5 × 10^n yen grid, aligned across the two data sources."""
    target = max((high - low) / 36, high * .0002, .1)
    exponent = 10 ** math.floor(math.log10(target))
    width = next(s * exponent for s in (1, 2, 5, 10) if s * exponent >= target)
    first = price_bin(low, width)
    last = price_bin(high, width)
    return float(width), int(first), int(last)


def price_bin(price, width):
    # Remove binary floating point noise at decimal yen boundaries only.
    return np.floor(np.round(np.asarray(price) / width, 10)).astype(np.int64)


def make_profile(points, close, grid):
    width, first, last = grid
    price, shares, turnover = points.T
    index = np.clip(price_bin(price, width) - first, 0, last - first)
    counts = np.bincount(index, weights=shares, minlength=last - first + 1)
    amounts = np.bincount(index, weights=turnover, minlength=len(counts))
    total = float(counts.sum())
    vwap = float(amounts.sum()) / total
    peak = int(np.argmax(counts))
    lo = hi = peak
    included = counts[peak]
    while included < .7 * total:
        left = counts[lo - 1] if lo > 0 else -1
        right = counts[hi + 1] if hi + 1 < len(counts) else -1
        if left >= right:
            lo -= 1
            included += left
        else:
            hi += 1
            included += right
    bounds = lambda i: round((first + i) * width, 8)
    above = float(shares[price > close + 1e-8].sum()) / total
    nearby = float(shares[(price >= close * .99 - 1e-8) & (price <= close * 1.01 + 1e-8)].sum()) / total
    return {
        "vwap": vwap, "poc": [bounds(peak), bounds(peak + 1)],
        "valueArea": [bounds(lo), bounds(hi + 1)], "shares": total,
        "turnover": float(amounts.sum()), "pocShare": float(counts[peak]) / total,
        "aboveCloseShare": above, "nearCloseShare": nearby,
        "bands": [[bounds(i), bounds(i + 1), float(qty), float(cash)]
                  for i, (qty, cash) in enumerate(zip(counts, amounts))],
    }


def window_profiles(dates, count, quotes, intraday, cutoff):
    """Return a comparable profile for each source and explicit gaps."""
    def missing(reason):
        return {source: {"status": reason, "coveredSessions": 0} for source in SOURCES}

    if len(dates) != count:
        return missing("calendar_history_missing")
    if any(day not in quotes for day in dates):
        return missing("daily_missing")
    if any(quotes[day].observed_at > cutoff for day in dates):
        return missing("not_available_at_cutoff")
    if any(not (quotes[day].valid or quotes[day].no_trades) for day in dates):
        return missing("invalid_daily")
    close = quotes[dates[-1]].close
    if close is None:
        return missing("close_missing")
    factors = share_factors(dates, quotes)
    if factors is None:
        return missing("share_basis_unknown")
    traded = [d for d in dates if quotes[d].valid]
    grid = bin_grid(min(quotes[d].low * factors[d] for d in traded),
                    max(quotes[d].high * factors[d] for d in traded))
    results = {}
    for source in SOURCES:
        parts = []
        reasons = []
        for day in dates:
            session = intraday[source].get(day)
            quote = quotes[day]
            if session is None and quote.no_trades:
                continue
            if session is None:
                reasons.append("intraday_missing")
            elif session.observed_at > cutoff:
                reasons.append("not_available_at_cutoff")
            elif not reconciles(session.points, quote):
                reasons.append("daily_totals_mismatch")
            elif len(session.points):
                factor = factors[day]
                parts.append(session.points * np.array([factor, 1 / factor, 1]))
        info = {"coveredSessions": count - len(reasons), "from": dates[0], "to": dates[-1]}
        if reasons:
            results[source] = {**info, "status": reasons[0], "reasons": sorted(set(reasons))}
            continue
        results[source] = {**info, "status": "ok", "adjustedShares": any(f != 1 for f in factors.values()),
                           **make_profile(np.concatenate(parts), close, grid)}
    if all(result["status"] == "ok" for result in results.values()):
        tick = results["tick"]
        minute = results["minute"]
        tv = .5 * sum(abs(a[2] / tick["shares"] - b[2] / minute["shares"])
                      for a, b in zip(tick["bands"], minute["bands"]))
        for result in results.values():
            result["tickMinuteDistance"] = tv
    return results
