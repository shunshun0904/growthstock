"""Research-only next-open / at-most-20-session exit analysis.

All returns are ratios. A high-water mark is an oracle diagnostic, never a fill.
This module does not change production models, signals, or trading rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


CLASSIFIERS = ("lgbm", "xgb", "cat", "lgbm_k15", "xgb_k15", "cat_k15")
SCORE_FEATURES = tuple(f"pct_{a}" for a in CLASSIFIERS) + ("pct_ltr", "pct_ltr5")


def historical_percentile(scores, reference):
    """Strictly-less empirical CDF, frozen by the caller before evaluation."""
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        raise ValueError("Empty calibration reference")
    values = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(values), 100 * np.searchsorted(ref, values, side="left") / len(ref), np.nan)


def within_day_percentile(frame):
    """Same convention as ltr.day_percentile; singleton is neutral, 50%."""
    g = frame.groupby("Date")["score"]
    n = g.transform("size")
    return ((g.rank(method="average") - 1) / (n - 1)).where(n > 1, .5) * 100


def combine_scores(frames, cutoff):
    """Use the entire rolling pool before selecting candidate-day rows."""
    cutoff = pd.Timestamp(cutoff)
    out = None
    for algo in CLASSIFIERS:
        d = frames[algo][["Code", "Date", "score"]].copy()
        if d.duplicated(["Code", "Date"]).any():
            raise ValueError(f"Duplicate keys: {algo}")
        d[f"pct_{algo}"] = historical_percentile(d.score, d.loc[d.Date <= cutoff, "score"])
        d = d.rename(columns={"score": f"score_{algo}"})
        out = d if out is None else out.merge(d, on=["Code", "Date"], how="outer", validate="one_to_one")
    for algo in ("ltr", "ltr5"):
        d = frames[algo].copy()
        d[f"pct_{algo}"] = within_day_percentile(d)
        if algo == "ltr5":
            d = d[d.cand_date.eq(d.Date)]
        if d.duplicated(["Code", "Date"]).any():
            raise ValueError(f"Duplicate ranking keys: {algo}")
        out = out.merge(d[["Code", "Date", "score", f"pct_{algo}", "fold"]].rename(
            columns={"score": f"score_{algo}", "fold": f"fold_{algo}"}),
            on=["Code", "Date"], how="left", validate="one_to_one")
    out["base_min"] = out[[f"pct_{a}" for a in CLASSIFIERS[:3]]].min(axis=1, skipna=False)
    out["six_min"] = out[[f"pct_{a}" for a in CLASSIFIERS]].min(axis=1, skipna=False)
    return out.sort_values(["Date", "Code"]).reset_index(drop=True)


@dataclass
class Paths:
    rows: pd.DataFrame
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    dates: np.ndarray


def next_open_paths(keys, bars, calendar, hold=20):
    """Require the next *market* session's open; do not skip a suspended day.

    Require a full valid OHLC path even when a profit target hit early. This
    prevents preferential inclusion of quick winners in incomplete cohorts.
    Incomplete / invalid paths remain in the audit with an explicit status.
    """
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    k = keys[["Code", "Date"]].copy().reset_index(drop=True)
    k["Date"] = pd.to_datetime(k.Date)
    k["Code"] = k.Code.astype(str)
    if k.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate signal keys")
    size = len(k)
    arrays = [np.full((size, hold), np.nan) for _ in range(4)]
    when = np.full((size, hold), np.datetime64("NaT"), dtype="datetime64[ns]")
    statuses = np.full(size, "no_price_history", dtype=object)
    entry = np.full(size, np.nan)
    px = {str(code): g.set_index("Date") for code, g in bars.groupby("Code", sort=False)}
    last = pd.Timestamp(bars.Date.max())
    for code, idx in k.groupby("Code", sort=False).groups.items():
        g = px.get(code)
        if g is None:
            continue
        for i in idx:
            pos = cal.searchsorted(k.at[i, "Date"], side="right")
            days = cal[pos:pos + hold]
            when[i, :len(days)] = days.to_numpy()
            if len(days) < hold or days[-1] > last:
                statuses[i] = "not_mature"
                continue
            z = g.reindex(days)[["AdjO", "AdjH", "AdjL", "AdjC"]].to_numpy(float)
            entry[i] = z[0, 0]
            if not np.isfinite(entry[i]) or entry[i] <= 0:
                statuses[i] = "no_next_open"
                continue
            if not np.isfinite(z).all() or (z <= 0).any():
                statuses[i] = "incomplete_ohlc"
                continue
            tol = 1e-8 * z[:, 0]
            if ((z[:, 1] + tol < z[:, [0, 2, 3]].max(axis=1)) |
                    (z[:, 2] - tol > z[:, [0, 1, 3]].min(axis=1))).any():
                statuses[i] = "invalid_ohlc"
                continue
            for j in range(4):
                arrays[j][i] = z[:, j] / entry[i] - 1
            statuses[i] = "complete"
    k["status"] = statuses
    k["entry_adjusted"] = entry
    k["entry_date"] = when[:, 0]
    k["mature_date"] = when[:, -1]
    good = statuses == "complete"
    for name, values in [("mfe", arrays[1]), ("mae", arrays[2])]:
        k[name] = np.nan
        k.loc[good, name] = (np.max if name == "mfe" else np.min)(values[good], axis=1)
    k["ret_close20"] = arrays[3][:, -1]
    k["peak_day"] = np.nan
    k.loc[good, "peak_day"] = np.argmax(arrays[1][good], axis=1) + 1
    for target in (.03, .05, .08, .10, .15):
        name = f"hit{round(target * 100)}"
        touch = arrays[1] >= target - 1e-12
        k[name] = np.where(good, touch.any(axis=1).astype(float), np.nan)
        k[name + "_day"] = np.where(good & touch.any(axis=1), touch.argmax(axis=1) + 1, np.nan)
    return Paths(k, *arrays, when)


def label_maturity(keys, bars, hold=20):
    """Original labels use company-bar shifts; audit those separately from trades."""
    out = np.full(len(keys), np.datetime64("NaT"), dtype="datetime64[ns]")
    groups = {str(c): np.sort(g.Date.to_numpy(dtype="datetime64[ns]"))
              for c, g in bars.groupby("Code", sort=False)}
    k = keys.reset_index(drop=True)
    for code, ids in k.groupby("Code", sort=False).groups.items():
        dates = groups.get(str(code))
        if dates is None:
            continue
        ix = np.asarray(ids)
        pos = np.searchsorted(dates, k.loc[ix, "Date"].to_numpy(dtype="datetime64[ns]"), side="right") + hold - 1
        ok = pos < len(dates)
        out[ix[ok]] = dates[pos[ok]]
    return pd.to_datetime(out)


def entry_mask(d, spec):
    mask = d.base_min.ge(spec["base"])
    if "six" in spec:
        mask &= d.six_min.ge(spec["six"])
    if "strict" in spec:
        mask &= d[[f"pct_{a}" for a in CLASSIFIERS[3:]]].min(axis=1, skipna=False).ge(spec["strict"])
    if "daily" in spec:
        mask &= d.pct_ltr.ge(spec["daily"])
    if "rolling" in spec:
        mask &= d.pct_ltr5.ge(spec["rolling"])
    return mask.fillna(False)


def require_full_ranking_pools(scores, coverage):
    """A rank of future-label-eligible rows is not a full decision-time rank."""
    out = scores.copy()
    for algo, field in [('ltr', 'daily_complete'), ('ltr5', 'rolling_complete')]:
        safe = out.Date.map(coverage[field]).fillna(False)
        out[f'{algo}_pool_complete'] = safe
        out.loc[~safe, f'pct_{algo}'] = np.nan
    return out


def selected_ids(d, spec, top_k=2):
    """Signal selection precedes checking future path completeness."""
    return d.loc[entry_mask(d, spec)].sort_values(
        ["Date", "base_min", "score_lgbm", "Code"], ascending=[True, False, False, True],
        kind="stable").groupby("Date", sort=False).head(top_k).index.to_numpy()


def simulate(paths, ids, *, target=None, stop=None, timeout=20,
             trail=None, activate=.05, ambiguity="stop_first"):
    """Daily-bar executable approximation, gross of round-trip costs.

    Profit gaps fill at the target (conservative limit price). Stop gaps fill
    at the open (including losses worse than the stop). Trailing levels use
    previous closes, and can first become active on the following session.
    The buy-day O is the entry itself. Same-bar order is explicitly varied.
    """
    if not 1 <= timeout <= paths.h.shape[1]:
        raise ValueError("Invalid timeout")
    if ambiguity not in ("stop_first", "target_first"):
        raise ValueError("Invalid intraday-order assumption")
    results = []
    for i in ids:
        if paths.rows.at[i, "status"] != "complete":
            continue
        peak_close = 0.0
        active = False
        ambiguous = False
        for day in range(timeout):
            o, h, l, c = (a[i, day] for a in (paths.o, paths.h, paths.l, paths.c))
            floor = -stop if stop is not None else -np.inf
            if trail is not None and active:
                floor = max(floor, (1 + peak_close) * (1 - trail) - 1)
            tp = target is not None and h >= target - 1e-12
            sl = np.isfinite(floor) and l <= floor + 1e-12
            # The open precedes intraday high/low; open gaps are unambiguous.
            if sl and o <= floor + 1e-12:
                value, reason = o, "stop_gap"
            elif tp and o >= target - 1e-12:
                value, reason = target, "target_gap"
            elif tp and sl:
                ambiguous = True
                value, reason = (floor, "stop") if ambiguity == "stop_first" else (target, "target")
            elif sl:
                value, reason = floor, "stop"
            elif tp:
                value, reason = target, "target"
            elif day == timeout - 1:
                value, reason = c, "timeout"
            else:
                peak_close = max(peak_close, c)
                active = active or c >= activate
                continue
            results.append({"id": int(i), "gross": value, "exit_day": day + 1,
                            "exit_date": paths.dates[i, day], "reason": reason,
                            "ambiguous": ambiguous})
            break
    return pd.DataFrame(results, columns=["id", "gross", "exit_day", "exit_date", "reason", "ambiguous"])


def trade_summary(trades, cost=.002):
    if not len(trades):
        return {"n": 0}
    r = trades.gross.to_numpy(float) - cost
    return {"n": len(r), "mean_net_pct": 100 * r.mean(), "median_net_pct": 100 * np.median(r),
            "win_pct": 100 * (r > 0).mean(), "p05_net_pct": 100 * np.quantile(r, .05),
            "worst_net_pct": 100 * r.min(), "avg_days": trades.exit_day.mean(),
            "ambiguous_n": int(trades.ambiguous.sum()) if "ambiguous" in trades else 0}


def block_interval(values, dates, calendar, block=21, draws=2000, seed=123):
    """Circular moving blocks of market dates; includes zero-signal days."""
    dates = pd.to_datetime(dates)
    cal = pd.DatetimeIndex(calendar)
    if not len(dates):
        return [None, None]
    cal = cal[(cal >= dates.min()) & (cal <= dates.max())]
    z = pd.DataFrame({"Date": dates, "v": np.asarray(values, float)})
    agg = z.groupby("Date").v.agg(["sum", "count"]).reindex(cal, fill_value=0)
    n = len(cal)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        ix = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        num = agg["sum"].to_numpy()[ix].sum()
        den = agg["count"].to_numpy()[ix].sum()
        if den:
            samples.append(num / den)
    return np.quantile(samples, [.025, .975]).tolist() if samples else [None, None]
