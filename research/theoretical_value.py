"""Opt-in scenario valuations. No production features or models are modified.

All rates are fractions. Yen values use candidate-date share units. Ranges are
unweighted scenario sensitivity, never confidence intervals. CAPM supplies the
equity discount rate; it does not determine cash flows or a stock price.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

ERP = (.03, .04, .05, .06, .07)
HORIZONS = (5, 10)
PAYOUTS = (.25, .5, .75)
GROWTH = (0., .01, .02)
STATS = ("min", "max", "mean", "median", "std")
VALUES = [f"tv_{m}_{s}_rel" for m in ("ri", "ddm") for s in STATS]
CONTROLS = ["tv_beta", "tv_beta_n", "tv_rf", "tv_rf_age", "tv_fin_age",
            "tv_earnings_yield", "tv_book_yield", "tv_div_yield", "tv_payout",
            "tv_ri_valid_fraction", "tv_ddm_valid_fraction"]


def effective_yield(quoted_percent):
    """MOF semiannual compound yield -> effective annual fraction; keep negatives."""
    return (1 + np.asarray(quoted_percent, dtype=float) / 200.) ** 2 - 1


def riskfree(dates, yields):
    """MOF reference date t is released next business day at 09:30 (MOF QA).

    For an after-close trading decision, require reference date strictly < t.
    Refuse gaps over seven calendar days; do not extrapolate a missing history.
    """
    left = pd.DataFrame({"Date": pd.to_datetime(dates), "_i": np.arange(len(dates))})
    right = yields[["Date", "10Y"]].copy().rename(columns={"Date": "tv_rf_date"})
    right["tv_rf_date"] = pd.to_datetime(right.tv_rf_date)
    right["tv_rf"] = effective_yield(pd.to_numeric(right["10Y"], errors="coerce"))
    right = right.dropna(subset=["tv_rf"]).sort_values("tv_rf_date").drop_duplicates("tv_rf_date")
    out = pd.merge_asof(left.sort_values("Date"), right, left_on="Date", right_on="tv_rf_date",
                        direction="backward", allow_exact_matches=False).sort_values("_i")
    out["tv_rf_age"] = (out.Date - out.tv_rf_date).dt.days
    out.loc[out.tv_rf_age.gt(7), "tv_rf"] = np.nan
    return out[["tv_rf", "tv_rf_age", "tv_rf_date"]].reset_index(drop=True)


def beta(keys, prices, market, yields, window=504, min_obs=252):
    """Trailing pairwise OLS excess-price-return beta, then 2/3 beta + 1/3.

    A missing stock close breaks BOTH adjacent daily returns: a multi-day stock
    return must never be paired with one daily market return. Window counts
    market sessions; observations count only valid pairs. No future bar is used.
    TOPIX and stock histories exclude cash dividends (explicit approximation).
    """
    market = market.sort_values("Date").drop_duplicates("Date").set_index("Date")
    dates = pd.DatetimeIndex(market.index)
    rf = riskfree(dates, yields)
    daily_rf = pd.Series(np.power(1 + rf.tv_rf.to_numpy(), 1/252) - 1, index=dates)
    m = market.topix.pct_change(fill_method=None) - daily_rf
    keys = keys[["Code", "Date"]].reset_index(drop=True)
    out = pd.DataFrame(index=keys.index, columns=["tv_beta_raw", "tv_beta", "tv_beta_n"], dtype=float)
    wanted = {c: g.index.to_numpy() for c, g in keys.groupby("Code", sort=False)}
    for code, bars in prices.groupby("Code", sort=False):
        if code not in wanted:
            continue
        stock = bars.drop_duplicates("Date").set_index("Date").close.reindex(dates)
        s = stock.pct_change(fill_method=None) - daily_rf
        valid = s.notna() & m.notna()
        x, y = m.where(valid), s.where(valid)
        n = valid.astype(float).rolling(window, min_periods=1).sum()
        sx = x.rolling(window, min_periods=1).sum()
        sy = y.rolling(window, min_periods=1).sum()
        cov = (x*y).rolling(window, min_periods=1).sum() - sx*sy/n
        var = (x*x).rolling(window, min_periods=1).sum() - sx*sx/n
        b = (cov / var).where((n >= min_obs) & (var > 1e-12))
        ix = wanted[code]
        frame = pd.DataFrame({"tv_beta_raw": b, "tv_beta": 2*b/3 + 1/3, "tv_beta_n": n})
        out.loc[ix] = frame.reindex(pd.DatetimeIndex(keys.loc[ix, "Date"])).to_numpy()
    return out


def financial_inputs(keys, fins, prices):
    """Build historical financial state from versions disclosed BEFORE candidate day.

    TTM income = current cumulative + previous FY - previous same quarter;
    FY income can be used directly. Require approximately 12-month FYs and
    aligned year-on-year periods. Use common stockholders' equity ShEq only.
    Shares at period end are adjusted to candidate units using split-factor
    RATIOS; future common adjustment factors cancel. No NC/consolidated mixing.
    DDM uses latest actual FY total cash dividend / current shares (D0), never
    company forecast DPS. Fiscal-end dividend data older than 550 days is absent.
    """
    keys = keys[["Code", "Date", "close_raw"]].reset_index(drop=True)
    f = fins.copy()
    for c in ("DiscDate", "CurFYSt", "CurPerEn"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    f = f[f.CurPerType.isin(["1Q", "2Q", "3Q", "FY"]) &
          f.DocType.str.contains("FinancialStatements", na=False)].copy()
    for c in ["NP", "ShEq", "ShOutFY", "TrShFY", "DivTotalAnn"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["_scope"] = f.DocType.str.extract(r"_(Consolidated|NonConsolidated)_", expand=False)
    sources = {c: g for c, g in f.groupby("Code", sort=False)}
    price_groups = {c: g.sort_values("Date") for c, g in prices.groupby("Code", sort=False)}
    result = []
    for code, group in keys.groupby("Code", sort=False):
        if code not in sources or code not in price_groups:
            continue
        fc, pc = sources[code], price_groups[code]
        pdates = pd.DatetimeIndex(pc.Date)
        factors = (pc.close_raw / pc.close).to_numpy()
        def factor(date):
            j = pdates.searchsorted(date, side="right") - 1
            if j < 0 or (date - pdates[j]).days > 10:
                return np.nan
            return factors[j]
        for i, row in group.iterrows():
            snap = fc[(fc.DiscDate < row.Date) & (fc.CurPerEn <= fc.DiscDate)]
            snap = snap.sort_values(["DiscDate", "DiscTime"]).drop_duplicates(["CurFYSt", "CurPerType", "_scope"], keep="last")
            if snap.empty:
                continue
            latest = snap.sort_values(["CurPerEn", "DiscDate", "DiscTime"]).iloc[-1]
            if (row.Date - latest.DiscDate).days > 365:
                continue
            snap = snap[snap._scope.eq(latest._scope)]
            # An unknown treasury-share count is not a known zero.
            shares = latest.ShOutFY - latest.TrShFY
            shares *= factor(latest.CurPerEn) / factor(row.Date)
            if not np.isfinite(shares) or shares <= 0 or row.close_raw <= 0:
                continue
            fy = snap[snap.CurPerType.eq("FY") & (snap.CurPerEn - snap.CurFYSt).dt.days.between(350, 380)]
            ttm = np.nan
            used_date = latest.DiscDate
            if latest.CurPerType == "FY":
                if 350 <= (latest.CurPerEn - latest.CurFYSt).days <= 380:
                    ttm = latest.NP
            else:
                prevfy = fy[(latest.CurFYSt - fy.CurPerEn).dt.days.between(1, 15)]
                prevq = snap[snap.CurPerType.eq(latest.CurPerType) &
                             (latest.CurFYSt - snap.CurFYSt).dt.days.between(350, 380) &
                             (latest.CurPerEn - snap.CurPerEn).dt.days.between(350, 380)]
                if len(prevfy) and len(prevq):
                    a, b = prevfy.iloc[-1], prevq.iloc[-1]
                    ttm = latest.NP + a.NP - b.NP
                    used_date = max(used_date, a.DiscDate, b.DiscDate)
            eps, bps = ttm / shares, latest.ShEq / shares
            div, payout, divdate = np.nan, np.nan, pd.NaT
            if len(fy):
                annual = fy.sort_values("CurPerEn").iloc[-1]
                if (row.Date - annual.CurPerEn).days <= 550:
                    div = annual.DivTotalAnn / shares
                    payout = annual.DivTotalAnn / annual.NP if annual.NP > 0 else np.nan
                    divdate = annual.DiscDate
            result.append({"_i": i, "tv_eps": eps, "tv_bps": bps, "tv_div0": div,
                           "tv_payout": payout, "tv_fin_source_date": used_date,
                           "tv_div_source_date": divdate, "tv_period_end": latest.CurPerEn,
                           "tv_fin_age": (row.Date - latest.DiscDate).days,
                           "tv_earnings_yield": eps/row.close_raw,
                           "tv_book_yield": bps/row.close_raw, "tv_div_yield": div/row.close_raw})
    columns = ["tv_eps", "tv_bps", "tv_div0", "tv_payout", "tv_fin_source_date", "tv_div_source_date",
               "tv_period_end", "tv_fin_age", "tv_earnings_yield", "tv_book_yield", "tv_div_yield"]
    if not result:
        return pd.DataFrame(index=keys.index, columns=columns)
    return pd.DataFrame(result).set_index("_i").reindex(keys.index)[columns]


def residual_value(book, earnings, ke, horizon, payout):
    """Clean-surplus RI with ROE fading to ke; terminal RI = 0, terminal V=B."""
    book, earnings, ke = np.broadcast_arrays(np.asarray(book, float), np.asarray(earnings, float), np.asarray(ke, float))
    b = book.copy()
    roe = earnings / book
    value = book.copy()
    for t in range(1, horizon + 1):
        rt = ke + (roe-ke)*(horizon-t)/horizon
        e = rt*b
        value += (e-ke*b) / (1+ke)**t
        b = b + e*(1-payout)
    return value


def summarize(values, minimum_fraction=.8):
    a = np.asarray(values, dtype=float)
    valid = np.isfinite(a) & (a > 0)
    a = np.where(valid, a, np.nan)
    fraction = valid.mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.column_stack([np.nanmin(a, axis=1), np.nanmax(a, axis=1),
                               np.nanmean(a, axis=1), np.nanmedian(a, axis=1), np.nanstd(a, axis=1, ddof=0)])
    out[fraction < minimum_fraction] = np.nan
    return out, fraction


def valuations(inputs):
    """Append 5 yen statistics + 5 price-relative statistics for EACH model."""
    d = inputs.copy()
    book, earn, div = (d[c].to_numpy(dtype=float) for c in ["tv_bps", "tv_eps", "tv_div0"])
    beta_, rf, price = (d[c].to_numpy(dtype=float) for c in ["tv_beta", "tv_rf", "close_raw"])
    ri, ddm = [], []
    for premium in ERP:
        ke = rf + beta_*premium
        for horizon in HORIZONS:
            for payout in PAYOUTS:
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    v = residual_value(book, earn, ke, horizon, payout)
                ri.append(np.where((book > 0) & (earn > 0) & (ke > 0), v, np.nan))
        for g in GROWTH:
            with np.errstate(invalid="ignore", divide="ignore"):
                v = div*(1+g)/(ke-g)
            ok = (div > 0) & (earn > 0) & d.tv_payout.between(0, 1, inclusive="right").to_numpy() & (ke-g >= .01)
            ddm.append(np.where(ok, v, np.nan))
    for name, scenarios in [("ri", ri), ("ddm", ddm)]:
        stats, fraction = summarize(np.column_stack(scenarios))
        d[f"tv_{name}_valid_fraction"] = fraction
        for j, s in enumerate(STATS):
            d[f"tv_{name}_{s}"] = stats[:, j]
            d[f"tv_{name}_{s}_rel"] = np.where(price > 0, stats[:, j]/price - (0 if s == "std" else 1), np.nan)
    return d
