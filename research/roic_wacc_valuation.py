"""Opt-in ROIC/WACC research. Never registered with production training.

Amounts are yen; rates are fractions. Missing is not zero. The no-cash-
deduction ROIC is a separate diagnostic, NOT permission to set unknown
non-operating cash to zero in an enterprise-to-equity bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd

import capital_efficiency as CE
import theoretical_value as TV

ERP = (.03, .05, .07)
GROWTH = (0., .01, .02, .03)
TAXES = ("tax30", "actual")
MODELS = ("steady", "fade5")
STATS = ("min", "max", "mean", "median", "std")


@dataclass(frozen=True)
class Config:
    erp: tuple = ERP
    growth: tuple = GROWTH
    credit_spread: float = .015
    min_discount_spread: float = .01
    beta_window: int = 504
    beta_min_observations: int = 252
    annual_stale_days: int = 400

    def __post_init__(self):
        if (not self.erp or not self.growth or any(not np.isfinite(x) or x < 0 for x in (*self.erp, *self.growth))
                or not np.isfinite(self.credit_spread) or self.credit_spread < 0
                or self.min_discount_spread < .01 or self.beta_min_observations < 2
                or self.beta_window < self.beta_min_observations or self.annual_stale_days < 1):
            raise ValueError("Invalid valuation scenario configuration")


def next_session(disclosure_dates, sessions):
    """Strictly the NEXT market session, even for morning disclosures."""
    cal = pd.DatetimeIndex(pd.to_datetime(sessions)).normalize().unique().sort_values()
    d = pd.DatetimeIndex(pd.to_datetime(disclosure_dates, errors="coerce")).normalize()
    pos = cal.searchsorted(d, side="right")
    result = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = ~d.isna() & (pos < len(cal))
    result[valid] = cal.to_numpy()[pos[valid]]
    return pd.Series(result)


def _numeric(frame, name):
    return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)


def normalize_annual(panel):
    """Reuse CE debt/lease/scope guards, without CE's all-cash deduction.

    CE normalized equity is TOTAL net assets including NCI: do not add NCI
    again. An observed debt subtotal remains explicitly unreviewed.
    """
    p = panel.copy()
    cur, prev = CE._normalize(p), CE._normalize(p, "prev_")
    for label, norm in [("", cur), ("previous_", prev)]:
        for key in ("debt", "equity", "partial", "scope_risk", "debt_reviewed", "equity_reviewed"):
            p[label + key] = norm[key]
    p["nci"] = _numeric(p, "non_controlling_interests")
    p.loc[p.basis.eq("standalone"), "nci"] = 0.  # Structural fact, not imputation.
    p["non_operating_cash"] = np.nan
    p["previous_non_operating_cash"] = np.nan
    p["cash_identified"] = False
    p["balance_source"] = p.doc_id
    p["review_source"] = ""
    p["revision_unverified"] = True
    return p


def reviewed_states(panel, sessions, adjustments=None):
    """Dated annual events plus OPTIONAL private source-backed balance reviews.

    Review columns: Code, fiscal_year, basis, accounting_standard,
    source_doc_id, source_url, source_submit_date; at least one normalized
    balance: debt, equity, nci, non_operating_cash. Older comparative-year
    reviews update the current report only from their own availability date.
    They never displace a newer fiscal year or backdate a later review.
    """
    p = panel.copy()
    if p.empty:
        return p
    p["Code"] = p.Code.astype(str)
    p["avail_date"] = next_session(p.submit_date, sessions).to_numpy()
    p["filing_avail_date"] = p.avail_date
    p = normalize_annual(p).dropna(subset=["avail_date"])
    reviews = pd.DataFrame() if adjustments is None else adjustments.copy()
    fields = ("debt", "equity", "nci", "non_operating_cash")
    if len(reviews):
        required = {"Code", "fiscal_year", "basis", "accounting_standard", "source_doc_id", "source_url", "source_submit_date"}
        if not required <= set(reviews):
            raise ValueError("Balance reviews require source, date, fiscal year, and scope")
        reviews["Code"] = reviews.Code.astype(str)
        for key in ["source_doc_id", "source_url", "basis", "accounting_standard"]:
            if reviews[key].isna().any() or reviews[key].astype(str).str.strip().eq("").any():
                raise ValueError("Missing review provenance: " + key)
        if not reviews.source_url.str.startswith("https://").all():
            raise ValueError("Review URLs must use HTTPS")
        reviews["known_at"] = next_session(reviews.source_submit_date, sessions).to_numpy()
        if reviews.known_at.isna().any() or reviews.duplicated(["Code", "fiscal_year", "known_at"]).any():
            raise ValueError("Invalid or ambiguous balance review")
        for key in fields:
            reviews[key] = _numeric(reviews, key)
            if np.isinf(reviews[key]).any() or (key != "equity" and reviews[key].lt(0).any()):
                raise ValueError("Invalid reviewed balance: " + key)
        if reviews[list(fields)].notna().sum(axis=1).eq(0).any():
            raise ValueError("Balance review contains no observed balance")
        check = reviews.merge(p[["Code", "fiscal_year", "basis", "accounting_standard"]].drop_duplicates(),
                              on=["Code", "fiscal_year"], how="left", suffixes=("", "_raw"), validate="many_to_one")
        if any(check[k].ne(check[k + "_raw"]).any() for k in ["basis", "accounting_standard"]):
            raise ValueError("Reviewed and original accounting scope differ")
        extra = []
        for (code, date), _ in reviews.groupby(["Code", "known_at"], sort=True):
            known = p[p.Code.eq(code) & p.avail_date.le(date)]
            if len(known):
                event = known.sort_values(["fiscal_year", "avail_date"]).tail(1).copy()
                event["avail_date"] = date
                extra.append(event)
        if extra:
            p = pd.concat([p, *extra], ignore_index=True)
        for r in reviews.sort_values("known_at").itertuples():
            for offset in (0, 1):
                mask = p.Code.eq(r.Code) & p.fiscal_year.eq(r.fiscal_year + offset) & p.avail_date.ge(r.known_at)
                for key in fields:
                    value = getattr(r, key)
                    if pd.notna(value) and (offset == 0 or key != "nci"):
                        col = ("previous_" if offset else "") + key
                        p.loc[mask, col] = value
                        if key in ("debt", "equity"):
                            p.loc[mask, ("previous_" if offset else "") + key + "_reviewed"] = True
                            p.loc[mask, ("previous_" if offset else "") + ("partial" if key == "debt" else "scope_risk")] = False
                p.loc[mask, "review_source"] += str(r.source_doc_id) + ";"
    p["cash_identified"] = p.non_operating_cash.notna() & p.previous_non_operating_cash.notna()
    p["average_valid"] = p.ce_average_invalid.eq(0)
    p["financial_or_unknown"] = p.ce_financial_or_unknown.ne(0)
    p["actual_tax"] = (p.income_taxes / p.profit_before_tax.where(p.profit_before_tax.gt(0))).where(lambda x: x.between(0, 1))
    p["conservative_capital"] = p.equity + p.debt
    p["previous_conservative_capital"] = p.previous_equity + p.previous_debt
    p["operating_capital"] = p.conservative_capital - p.non_operating_cash
    p["previous_operating_capital"] = p.previous_conservative_capital - p.previous_non_operating_cash
    quality = (~p.partial & ~p.previous_partial & ~p.scope_risk & ~p.previous_scope_risk
               & p.average_valid & ~p.financial_or_unknown)
    for kind in ("conservative", "operating"):
        a, b = p[kind + "_capital"], p["previous_" + kind + "_capital"]
        avg = ((a + b) / 2).where(quality & a.gt(0) & b.gt(0))
        p["average_" + kind + "_capital"] = avg
        for tax_name, rate in [("tax30", .30), ("actual", p.actual_tax)]:
            nopat = p.operating_income * (1 - rate)
            p["nopat_" + tax_name] = nopat
            p["roic_" + kind + "_" + tax_name] = (nopat / avg).replace([np.inf, -np.inf], np.nan)
    p = p.sort_values(["avail_date", "Code", "fiscal_year"], kind="mergesort")
    p = p[p.fiscal_year.ge(p.groupby("Code").fiscal_year.cummax())]
    return p.drop_duplicates(["Code", "avail_date"], keep="last").reset_index(drop=True)


def attach_annual(keys, states, stale_days=400):
    """No fallback to an older valid report when the latest report is invalid."""
    left = keys.copy().reset_index(drop=True).assign(_rw_order=np.arange(len(keys)))
    left["Date"] = pd.to_datetime(left.Date)
    left["Code"] = left.Code.astype(str)
    if states.empty:
        return left.drop(columns="_rw_order")
    overlap = (set(left) & set(states)) - {"Code"}
    if overlap:
        raise ValueError("Ambiguous annual input columns: " + str(sorted(overlap)))
    out = pd.merge_asof(left.sort_values("Date"), states.sort_values("avail_date"),
                        left_on="Date", right_on="avail_date", by="Code", direction="backward")
    out["annual_age"] = (out.Date - out.filing_avail_date).dt.days
    out["annual_fresh"] = out.annual_age.between(0, stale_days)
    value_columns = [c for c in out if c.startswith(("roic_", "nopat_")) or c == "actual_tax"]
    out.loc[~out.annual_fresh, value_columns] = np.nan
    return out.sort_values("_rw_order").drop(columns="_rw_order").reset_index(drop=True)


def wacc(market_equity, debt, risk_free, beta, erp, tax, credit_spread=.015, debt_cost=None):
    """Specified CAPM/common-equity weights, book debt proxy; no size premium.

    Kd defaults to an EXPLICIT Rf + credit-spread scenario, not an observed
    borrowing rate. A supplied zero debt is allowed, unknown debt is not.
    """
    e, d, rf, b, premium, t = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
        [market_equity, debt, risk_free, beta, erp, tax]))
    ke = rf + b * premium
    kd = rf + credit_spread if debt_cost is None else np.broadcast_to(np.asarray(debt_cost, float), e.shape)
    valid = (np.isfinite(e) & (e > 0) & np.isfinite(d) & (d >= 0) & np.isfinite(ke)
             & (ke > 0) & np.isfinite(kd) & (kd >= 0) & np.isfinite(t) & (t >= 0) & (t <= 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = ke * e / (e + d) + kd * (1 - t) * d / (e + d)
    return np.where(valid & (value > 0), value, np.nan)


def enterprise_value(nopat, roic, discount, growth, model="steady", min_spread=.01):
    """Scenario NOPAT_next = observed annual NOPAT * (1+g).

    fade5: five explicit FCFF years; the reinvestment-return assumption fades
    linearly from observed ROIC to WACC. Terminal ROIC = WACC. This is a
    value-driver forecast, not an accounting capital roll-forward identity.
    """
    if model not in MODELS:
        raise ValueError("Unknown ROIC/WACC value model")
    n, r, w, g = np.broadcast_arrays(*map(lambda x: np.asarray(x, float), [nopat, roic, discount, growth]))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        valid = (np.isfinite(n) & np.isfinite(r) & np.isfinite(w) & np.isfinite(g)
                 & (n > 0) & (r > 0) & (g >= 0) & (w - g >= min_spread - 1e-12)
                 & (g / r >= 0) & (g / r <= 1))
        if model == "steady":
            ev = n * (1 + g) * (1 - g / r) / (w - g)
        else:
            ev = np.zeros_like(n)
            for year in range(1, 6):
                rt = r + (w - r) * year / 5
                reinvest = g / rt
                valid &= (rt > 0) & (reinvest >= 0) & (reinvest <= 1)
                ev += n * (1 + g) ** year * (1 - reinvest) / (1 + w) ** year
            ev += n * (1 + g) ** 6 * (1 - g / w) / (w - g) / (1 + w) ** 5
    return np.where(valid & np.isfinite(ev) & (ev > 0), ev, np.nan)


def equity_bridge(ev, debt, nci, non_operating_cash, shares):
    """Strict EV -> parent common equity -> per-share price, without clipping."""
    ev, d, minority, cash, q = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
        [ev, debt, nci, non_operating_cash, shares]))
    equity = ev - d - minority + cash
    valid = (np.isfinite(ev) & (ev > 0) & np.isfinite(d) & (d >= 0)
             & np.isfinite(minority) & (minority >= 0) & np.isfinite(cash) & (cash >= 0)
             & np.isfinite(q) & (q > 0) & np.isfinite(equity) & (equity > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        price = equity / q
    return np.where(valid, equity, np.nan), np.where(valid, price, np.nan)


def simplified_seven_factor(margin, turnover, leverage, bps, roic, discount, growth):
    """Illustration only: NI=NOPAT, no debt/cash/NCI, coherent share basis.

    This implements the requested expression EXACTLY. M*T*L*BPS must mean
    NEXT-period EPS to compare with the EV model's NOPAT_next. With current
    EPS and NOPAT_next=NOPAT_current*(1+g), the EV model has an additional
    (1+g) factor. This is NOT a general stock-price identity.
    """
    eps = np.asarray(margin) * np.asarray(turnover) * np.asarray(leverage) * np.asarray(bps)
    return enterprise_value(eps / (1 + np.asarray(growth)), roic, discount, growth)


def dupont(price, net_income_parent, sales, average_assets, average_parent_equity, shares):
    p, n, s, a, e, q = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
        [price, net_income_parent, sales, average_assets, average_parent_equity, shares]))
    valid = np.isfinite(p + n + s + a + e + q) & (p > 0) & (n > 0) & (s > 0) & (a > 0) & (e > 0) & (q > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        factors = {"net_margin": n / s, "asset_turnover": s / a, "leverage": a / e,
                   "bps_average_equity": e / q, "per_coherent": p / (n / q), "roe": n / e}
    return {k: np.where(valid, v, np.nan) for k, v in factors.items()}


def implied_growth_fixed_nopat(market_ev, nopat_next_fixed, roic, discount, min_spread=.01, tolerance=1e-10):
    ev, n, r, w = np.broadcast_arrays(*map(lambda x: np.asarray(x, float),
        [market_ev, nopat_next_fixed, roic, discount]))
    with np.errstate(invalid="ignore", divide="ignore"):
        multiple = ev / n
        denom = multiple - 1 / r
        g = (multiple * w - 1) / denom
        valid = (np.isfinite(ev + n + r + w) & (ev > 0) & (n > 0) & (r > 0)
                 & (np.abs(denom) > tolerance * np.maximum(1, np.maximum(np.abs(multiple), np.abs(1 / r))))
                 & (g >= 0) & (g / r <= 1) & (w - g >= min_spread - 1e-12))
    return np.where(valid & np.isfinite(g), g, np.nan)


def scenario_id(model, tax, erp, growth):
    return f"{model}_{tax}_erp{round(100 * erp):02d}_g{round(100 * growth):02d}"


def scenarios(inputs, config=Config()):
    """Wide, per-date sensitivity grid; never fill missing bridge inputs.

    Required inputs: price, shares, debt, nci, non_operating_cash,
    roic_operating_{tax30,actual}, nopat_{tax30,actual}, actual_tax, tv_rf,
    tv_beta. Optional annual_fresh/financial_or_unknown gate all scenarios.
    """
    d = inputs.copy()
    price, q, debt, nci, cash = [_numeric(d, x).to_numpy(float) for x in
                                ("price", "shares", "debt", "nci", "non_operating_cash")]
    market_equity = price * q
    market_ev = market_equity + debt + nci - cash
    d["market_equity"] = market_equity
    d["market_ev"] = np.where(np.isfinite(market_ev) & (market_ev > 0), market_ev, np.nan)
    rf, beta = [_numeric(d, x).to_numpy(float) for x in ("tv_rf", "tv_beta")]
    fresh = d.annual_fresh.fillna(False).to_numpy(bool) if "annual_fresh" in d else np.ones(len(d), bool)
    nonfin = ~d.financial_or_unknown.astype("boolean").fillna(True).to_numpy(bool) if "financial_or_unknown" in d else np.ones(len(d), bool)
    profitable = (_numeric(d, "net_income_parent").gt(0).to_numpy() if "net_income_parent" in d
                  else np.ones(len(d), bool))
    meta = []
    for tax_name in TAXES:
        tax = .30 if tax_name == "tax30" else _numeric(d, "actual_tax").to_numpy(float)
        roic, nopat = [_numeric(d, x + tax_name).to_numpy(float) for x in ("roic_operating_", "nopat_")]
        for erp in config.erp:
            w = wacc(market_equity, debt, rf, beta, erp, tax, config.credit_spread)
            w = np.where(fresh & nonfin, w, np.nan)
            d[f"wacc_{tax_name}_{round(erp * 100):02d}"] = w
            for model in MODELS:
                for growth in config.growth:
                    sid = scenario_id(model, tax_name, erp, growth)
                    ev = enterprise_value(nopat, roic, w, growth, model, config.min_discount_spread)
                    equity, v = equity_bridge(ev, debt, nci, cash, q)
                    valid = fresh & nonfin & profitable & np.isfinite(v)
                    d["price_" + sid] = np.where(valid, v, np.nan)
                    d["ev_" + sid] = np.where(valid, ev, np.nan)
                    invalid = np.zeros(len(d), dtype=np.uint16)
                    for bit, bad in [
                        (1, ~np.isfinite(nopat) | (nopat <= 0) | ~profitable),
                        (2, ~np.isfinite(roic) | (roic <= 0)),
                        (4, ~np.isfinite(w) | (w - growth < config.min_discount_spread - 1e-12)),
                        (8, ~np.isfinite(roic) | (growth > roic)),
                        (16, ~np.isfinite(debt) | (debt < 0) | ~np.isfinite(nci) | (nci < 0)
                             | ~np.isfinite(cash) | (cash < 0) | ~np.isfinite(q) | (q <= 0)),
                        (32, np.isfinite(ev) & (ev - debt - nci + cash <= 0)),
                        (64, ~fresh), (128, ~nonfin)]:
                        invalid[bad] |= bit
                    d["invalid_" + sid] = invalid
                    d.loc[~np.isfinite(v) & (invalid == 0), "invalid_" + sid] = 256
                    meta.append({"id": sid, "model": model, "tax": tax_name, "erp": erp, "growth": growth})
    for model in MODELS:
        for tax_name in TAXES:
            names = [x["id"] for x in meta if x["model"] == model and x["tax"] == tax_name]
            for measure in ("price", "ev"):
                a = d[[measure + "_" + name for name in names]].to_numpy(float)
                count = np.isfinite(a).sum(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    values = [np.nanmin(a, axis=1), np.nanmax(a, axis=1), np.nanmean(a, axis=1),
                              np.nanmedian(a, axis=1), np.nanstd(a, axis=1, ddof=0)]
                for stat, v in zip(STATS, values):
                    d[f"{model}_{tax_name}_{measure}_{stat}"] = np.where(count > 0, v, np.nan)
                d[f"{model}_{tax_name}_{measure}_valid_count"] = count
    d["theoretical_price"] = d.steady_tax30_price_median
    d["theoretical_ev"] = d.steady_tax30_ev_median
    d["roic"] = _numeric(d, "roic_operating_tax30")
    d["wacc"] = d.get("wacc_tax30_05", np.nan)
    d["cost_of_debt"] = rf + config.credit_spread
    d["cost_of_debt_basis"] = f"scenario_rf_plus_{config.credit_spread:.6f}"
    for kind in ("forecast", "ttm"):
        eps = _numeric(d, "eps_" + kind).where(lambda x: x.gt(0))
        d["market_per_" + kind] = d.price / eps
        d["theoretical_per_" + kind] = d.theoretical_price / eps
    d["valuation_gap"] = d.price / d.theoretical_price.where(d.theoretical_price.gt(0))
    d["ev_valuation_gap"] = d.market_ev / d.theoretical_ev.where(d.theoretical_ev.gt(0))
    d["roic_wacc_spread"] = d.roic - d.wacc
    d["implied_growth_fixed_nopat"] = implied_growth_fixed_nopat(d.market_ev, _numeric(d, "nopat_tax30"), d.roic, d.wacc)
    # No independent company growth estimate exists unless explicitly supplied.
    d["growth_gap"] = _numeric(d, "independent_growth") - d.implied_growth_fixed_nopat
    reason = pd.Series("", index=d.index)
    conditions = {
        "financial_or_unknown": ~nonfin, "annual_stale_or_missing": ~fresh,
        "company_loss_or_profit_unverified": ~profitable,
        "debt_missing": ~np.isfinite(debt), "nci_missing": ~np.isfinite(nci),
        "non_operating_cash_unidentified": ~np.isfinite(cash),
        "shares_missing_or_nonpositive": ~np.isfinite(q) | (q <= 0),
        "risk_free_missing": ~np.isfinite(rf), "beta_unavailable": ~np.isfinite(beta),
        "operating_roic_unavailable_or_nonpositive": ~np.isfinite(d.roic) | (d.roic <= 0),
        "nopat_nonpositive_or_missing": ~np.isfinite(_numeric(d, "nopat_tax30")) | (_numeric(d, "nopat_tax30") <= 0),
    }
    for name, mask in conditions.items():
        reason.loc[mask] += name + ";"
    no_value = d.theoretical_price.isna()
    reason.loc[no_value & reason.eq("")] = "no_valid_positive_equity_scenario;"
    d["unavailable_reason"] = reason.str.rstrip(";")
    revision = d.get("revision_unverified", pd.Series(True, index=d.index)).astype("boolean").fillna(True)
    d["valuation_data_quality"] = (~no_value & ~revision).astype(float)
    d["scenario_valid_count"] = d.steady_tax30_price_valid_count
    d["scenario_total_count"] = len(config.erp) * len(config.growth)
    d.attrs["scenarios"] = meta
    return d.replace([np.inf, -np.inf], np.nan)
