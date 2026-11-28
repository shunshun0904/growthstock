"""価格帯別出来高の計算。

蓄積した分足・ティック（docs/MINUTE_BARS.md・docs/TICKS.md）を、日ごとに日足（調整前の安値・高値・
出来高・売買代金）と照合してから、1・5・20営業日の価格帯別の分布と、次の指標にまとめる
（docs/VOLUME_PROFILE.md「計算」）:

- 価格帯別の株数・売買代金、最多出来高の帯（POC）、出来高の70%範囲、VWAP
- VWAP と終値の差（終値 ÷ VWAP − 1）
- 時間帯別の出来高（寄り・前場・後場・引け。株数の割合）
- 約定代金の大きさ別の割合（ティックだけ。ザラバの約定を 1件の約定代金で 小口・中口・大口 に分ける）

分足は1分の出来高をその分の VWAP（売買代金 ÷ 出来高）に置く近似、ティックは約定価格そのもの。
"""

from __future__ import annotations

import datetime as dt
import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

WINDOWS = (1, 5, 20)
SOURCES = ("tick", "minute")
#: 時間帯（株数の割合）
#:   open      寄り。その日の最初の約定（板寄せ）。分足では最初の1分（板寄せのあとの約定も入る）
#:   morning   前場の残り（前引けの板寄せを含む）
#:   afternoon 後場（後場寄りの板寄せを含む）から引けを除いたもの
#:   close     引け（大引けの板寄せ）
BUCKETS = ("open", "morning", "afternoon", "close")
#: 1件の約定の約定代金（価格 × 株数）の区切り（円）。小口 < 100万円 ≤ 中口 < 1,000万円 ≤ 大口
SIZE_EDGES = (1_000_000.0, 10_000_000.0)
SIZE_LABELS = ("small", "medium", "large")
#: 2024-11-05 から東証の取引は 15:30 まで。15:25〜15:30 はクロージング・オークション（注文を受けるだけで約定
#: しない）で、15:30 の板寄せで引ける。それより前は 15:00 の板寄せで引けた
CLOSING_AUCTION_FROM = dt.date(2024, 11, 5)
_US = 1_000_000
MORNING_CLOSE_US = (11 * 3600 + 30 * 60) * _US
NOON_US = 12 * 3600 * _US
NOON_HHMM = "12:00"

STATUSES = (
    "ok", "calendar_history_missing", "daily_missing", "invalid_daily", "close_missing",
    "share_basis_unknown", "intraday_missing", "intraday_invalid", "daily_totals_mismatch",
)


def normalize_code(value) -> str:
    code = str(value).strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4,5}", code):
        raise ValueError("4桁または5桁の銘柄コードが必要です。")
    return code + "0" if len(code) == 4 else code


def close_start_us(day: dt.date) -> int:
    """引けの板寄せが始まる時刻（その日の0時からのマイクロ秒）。これ以降の後場の約定が引け。"""
    return ((15 * 3600 + 25 * 60) if day >= CLOSING_AUCTION_FROM else 15 * 3600) * _US


def close_start_hhmm(day: dt.date) -> str:
    return "15:25" if day >= CLOSING_AUCTION_FROM else "15:00"


@dataclass(frozen=True)
class Quote:
    """調整前の日足（東証の立会内）。factor は J-Quants の AdjFactor（権利落ち日の行に入る）。"""
    low: Optional[float]
    high: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    turnover: Optional[float]
    factor: Optional[float]

    @property
    def no_trades(self) -> bool:
        return self.volume in (None, 0) and self.turnover in (None, 0) and all(
            x is None for x in (self.low, self.high, self.close))

    @property
    def valid(self) -> bool:
        values = (self.low, self.high, self.close, self.volume, self.turnover)
        return (all(x is not None and math.isfinite(x) and x > 0 for x in values)
                and self.low <= self.close <= self.high)


@dataclass
class Session:
    """1銘柄・1日の分足またはティックをまとめたもの（生の行は持たない）。"""
    #: 価格ごとの (価格, 株数, 売買代金)。価格の昇順
    points: np.ndarray
    #: BUCKETS の順の株数
    buckets: np.ndarray
    #: ティックだけ: ザラバ（板寄せを除く）の約定代金と件数。SIZE_LABELS の順
    sizes: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    #: 行に不正な値があったときの理由（STATUSES の intraday_invalid）
    problem: Optional[str] = None


def _invalid() -> Session:
    return Session(np.empty((0, 3)), np.zeros(len(BUCKETS)), problem="intraday_invalid")


def _by_price(price: np.ndarray, shares: np.ndarray, turnover: np.ndarray) -> np.ndarray:
    levels, index = np.unique(price, return_inverse=True)
    return np.column_stack((levels, np.bincount(index, weights=shares, minlength=len(levels)),
                            np.bincount(index, weights=turnover, minlength=len(levels))))


def _first_group(mask: np.ndarray, time: np.ndarray, price: np.ndarray) -> np.ndarray:
    """その場の最初の約定（板寄せ）: 最初の約定と同じ時刻・同じ価格の約定。"""
    out = np.zeros(len(mask), dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size:
        first = idx[0]
        out[idx[(time[idx] == time[first]) & (price[idx] == price[first])]] = True
    return out


def tick_session(day: dt.date, time_us: np.ndarray, session: np.ndarray,
                 price: np.ndarray, shares: np.ndarray) -> Session:
    """
    約定の順（TransactionId の昇順）に並んだ1銘柄・1日のティックをまとめる。

    session は SessionDistinction（"01" 前場 / "02" 後場）。それ以外の値の行は時刻で分ける（12時より前は前場）。
    板寄せ: 前場・後場それぞれの最初の約定（同じ時刻・同じ価格の約定をまとめて）、前場の 11:30 以降の約定
    （前引け）、後場の引けの時刻以降の約定（大引け）。大きさ別の割合は板寄せを除いたザラバの約定で数える。
    """
    time_us = np.asarray(time_us, dtype=np.int64)
    price = np.asarray(price, dtype=float)
    shares = np.asarray(shares, dtype=float)
    session = np.asarray(session).astype(str)
    if not (len(time_us) == len(price) == len(shares) == len(session)):
        return _invalid()
    if len(price) and not (np.isfinite(price).all() and np.isfinite(shares).all()
                           and (price > 0).all() and (shares > 0).all() and (shares % 1 == 0).all()):
        return _invalid()
    value = price * shares
    morning = (session == "01") | ((session != "02") & (time_us < NOON_US))
    afternoon = ~morning
    open_m = _first_group(morning, time_us, price)
    open_a = _first_group(afternoon, time_us, price)
    close_m = morning & (time_us >= MORNING_CLOSE_US)
    close_a = afternoon & (time_us >= close_start_us(day))
    day_open = open_m if morning.any() else open_a
    close_auction = close_a & ~day_open
    buckets = np.array([shares[day_open].sum(), shares[morning & ~day_open].sum(),
                        shares[afternoon & ~close_auction & ~day_open].sum(), shares[close_auction].sum()])
    continuous = ~(open_m | open_a | close_m | close_a)
    small = continuous & (value < SIZE_EDGES[0])
    large = continuous & (value >= SIZE_EDGES[1])
    medium = continuous & ~small & ~large
    masks = (small, medium, large)
    return Session(_by_price(price, shares, value), buckets,
                   sizes=np.array([value[m].sum() for m in masks]),
                   counts=np.array([m.sum() for m in masks], dtype=np.int64))


def minute_session(day: dt.date, times: np.ndarray, volume: np.ndarray, turnover: np.ndarray,
                   low: np.ndarray, high: np.ndarray) -> Session:
    """
    1銘柄・1日の分足をまとめる。times は "HH:MM"。各分の株数と売買代金をその分の VWAP に置く。
    出来高も売買代金も 0 の分は約定が無いので飛ばす。
    """
    times = np.asarray(times).astype(str)
    volume = np.asarray(volume, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    traded = ~((volume == 0) & (turnover == 0))
    times, volume, turnover, low, high = (a[traded] for a in (times, volume, turnover, low, high))
    if not (np.isfinite(volume).all() and np.isfinite(turnover).all() and (volume > 0).all()
            and (turnover > 0).all() and (volume % 1 == 0).all()
            and all(re.fullmatch(r"\d{2}:\d{2}", t) for t in times)):
        return _invalid()
    with np.errstate(invalid="ignore", divide="ignore"):
        price = turnover / volume
    if len(price) and not ((low > 0).all() and (price >= low - 1e-6).all() and (price <= high + 1e-6).all()):
        return _invalid()
    if not len(price):
        return Session(np.empty((0, 3)), np.zeros(len(BUCKETS)))
    day_open = times == min(times)
    morning = times < NOON_HHMM
    close_auction = (times >= close_start_hhmm(day)) & ~day_open
    buckets = np.array([volume[day_open].sum(), volume[morning & ~day_open].sum(),
                        volume[~morning & ~close_auction & ~day_open].sum(), volume[close_auction].sum()])
    return Session(_by_price(price, volume, turnover), buckets)


def lookback(days: List[str], end: str, count: int) -> List[str]:
    """end を含む直近 count 営業日（end が営業日のときだけ）。"""
    stop = bisect_right(days, end)
    return days[max(0, stop - count):stop] if stop and days[stop - 1] == end else []


def reconcile(points: np.ndarray, quote: Quote) -> str:
    """
    日足との照合。"ok" か、合わなかった理由（"volume" 株数 / "turnover" 売買代金 / "range" 安値〜高値の外 /
    "invalid" 日足か集計が不正 / "traded" 日足は取引なしなのに約定がある）。
    株数は 0.01株未満、売買代金は 1円以内の差まで同じとみなす。
    """
    if quote.no_trades:
        return "ok" if len(points) == 0 else "traded"
    if not quote.valid or not len(points):
        return "invalid"
    price, qty, value = points.T
    if not (np.isfinite(points).all() and (points > 0).all()):
        return "invalid"
    if abs(qty.sum() - quote.volume) >= .01:
        return "volume"
    if abs(value.sum() - quote.turnover) > 1:
        return "turnover"
    if price.min() < quote.low - 1e-6 or price.max() > quote.high + 1e-6:
        return "range"
    return "ok"


def share_factors(dates: List[str], quotes: Dict[str, Quote]) -> Optional[Dict[str, float]]:
    """
    各営業日の価格に掛ける係数（株数は割る）。基準日（最後の日）の株数の単位に揃える。
    AdjFactor は権利落ち日の行に入り、その日より前の日に効く（1:2 の分割なら 0.5）。
    係数が無い・不正な日があれば基準が分からないので None。最初の日の係数は窓の中の日に効かない。
    """
    combined = 1.
    factors = {}
    for day in reversed(dates):
        factors[day] = combined
        if day == dates[0]:
            break
        factor = quotes[day].factor
        if factor is None or not math.isfinite(factor) or factor <= 0:
            return None
        combined *= factor
        if not math.isfinite(combined) or combined <= 0:
            return None
    return factors


def bin_grid(low: float, high: float):
    """1・2・5 × 10の整数乗 円の幅の価格帯。分足とティックで同じ帯を使う。"""
    target = max((high - low) / 36, high * .0002, .1)
    exponent = 10 ** math.floor(math.log10(target))
    width = next(s * exponent for s in (1, 2, 5, 10) if s * exponent >= target)
    return float(width), int(price_bin(low, width)), int(price_bin(high, width))


def price_bin(price, width):
    # 10進の円の境目で出る2進の浮動小数点の誤差だけを除く
    return np.floor(np.round(np.asarray(price) / width, 10)).astype(np.int64)


def make_profile(points: np.ndarray, close: float, grid) -> dict:
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

    def bounds(i):
        return round((first + i) * width, 8)

    above = float(shares[price > close + 1e-8].sum()) / total
    nearby = float(shares[(price >= close * .99 - 1e-8) & (price <= close * 1.01 + 1e-8)].sum()) / total
    return {
        "vwap": vwap, "closeVsVwap": close / vwap - 1,
        "poc": [bounds(peak), bounds(peak + 1)], "valueArea": [bounds(lo), bounds(hi + 1)],
        "shares": total, "turnover": float(amounts.sum()), "pocShare": float(counts[peak]) / total,
        "aboveCloseShare": above, "nearCloseShare": nearby,
        "bands": [[bounds(i), bounds(i + 1), float(qty), float(cash)]
                  for i, (qty, cash) in enumerate(zip(counts, amounts))],
    }


def window_profiles(dates: List[str], count: int, quotes: Dict[str, Quote],
                    intraday: Dict[str, Dict[str, Session]]) -> dict:
    """1つの期間の、データ種別ごとの集計。揃わない期間・種別は理由（status）だけを返す。"""
    def missing(reason):
        return {source: {"status": reason, "coveredSessions": 0} for source in SOURCES}

    if len(dates) != count:
        return missing("calendar_history_missing")
    if any(day not in quotes for day in dates):
        return missing("daily_missing")
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
        parts, reasons = [], []
        buckets = np.zeros(len(BUCKETS))
        sizes = np.zeros(len(SIZE_LABELS))
        counts = np.zeros(len(SIZE_LABELS), dtype=np.int64)
        for day in dates:
            session = intraday.get(source, {}).get(day)
            quote = quotes[day]
            if session is None and quote.no_trades:
                continue
            if session is None:
                reasons.append("intraday_missing")
            elif session.problem:
                reasons.append(session.problem)
            elif reconcile(session.points, quote) != "ok":
                reasons.append("daily_totals_mismatch")
            elif len(session.points):
                factor = factors[day]
                parts.append(session.points * np.array([factor, 1 / factor, 1]))
                buckets += session.buckets / factor
                if session.sizes is not None:
                    sizes += session.sizes
                    counts += session.counts
        info = {"coveredSessions": count - len(reasons), "from": dates[0], "to": dates[-1]}
        if reasons:
            results[source] = {**info, "status": reasons[0], "reasons": sorted(set(reasons))}
            continue
        profile = make_profile(np.concatenate(parts), close, grid)
        profile["sessions"] = dict(zip(BUCKETS, (buckets / buckets.sum()).tolist()))
        if source == "tick":
            continuous = float(sizes.sum())
            profile["tradeSizes"] = {
                "turnover": continuous,
                "shares": (sizes / continuous).tolist() if continuous > 0 else [0.] * len(SIZE_LABELS),
                "counts": counts.tolist(),
            }
        results[source] = {**info, "status": "ok", "adjustedShares": any(f != 1 for f in factors.values()),
                           **profile}
    if all(result["status"] == "ok" for result in results.values()):
        tick, minute = results["tick"], results["minute"]
        distance = .5 * sum(abs(a[2] / tick["shares"] - b[2] / minute["shares"])
                            for a, b in zip(tick["bands"], minute["bands"]))
        for result in results.values():
            result["tickMinuteDistance"] = distance
    return results
