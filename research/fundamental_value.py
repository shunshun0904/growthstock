"""Price-independent economic-profit valuation kernel.

This module imports NO market-data, CAPM, price, PER, PBR, or reverse-DCF
module. Its closed input type cannot accept a price/beta/market-cap field.
Book values, corporate share counts, and an external government yield are
inputs, not a claim of complete economic independence or causality.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Fundamentals:
    operating_capital: object
    historical_roic: object
    historical_nopat: object
    total_book_equity: object
    debt_book: object
    non_controlling_interests: object
    non_operating_cash: object
    shares: object
    risk_free: object
    tax_rate: object


@dataclass(frozen=True, slots=True)
class TotalCapitalFundamentals:
    """Explicit cash-unsegregated approximation; no unknown-cash argument."""
    total_book_equity: object
    debt_book: object
    historical_capital_return: object
    historical_nopat: object
    non_controlling_interests: object
    shares: object
    risk_free: object
    tax_rate: object


@dataclass(frozen=True, slots=True)
class RimFundamentals:
    parent_equity: object
    annual_parent_profit: object
    shares: object
    risk_free: object
    payout: object


@dataclass(frozen=True, slots=True)
class DiscountPolicy:
    equity_risk_premium: float = .05
    credit_spread: float = .015
    target_debt_weight: float | None = None
    minimum_wacc_minus_growth: float = .01

    def __post_init__(self):
        if (not np.isfinite(self.equity_risk_premium) or self.equity_risk_premium <= 0
                or not np.isfinite(self.credit_spread) or self.credit_spread < 0
                or not np.isfinite(self.minimum_wacc_minus_growth) or self.minimum_wacc_minus_growth < .01
                or self.target_debt_weight is not None and not 0 <= self.target_debt_weight <= 1):
            raise ValueError("Discount policy must be fixed independently of target prices")


def book_wacc(book_equity, debt_book, risk_free, tax_rate, policy=DiscountPolicy()):
    e, d, rf, tax = np.broadcast_arrays(*[np.asarray(x, float) for x in (book_equity, debt_book, risk_free, tax_rate)])
    with np.errstate(invalid="ignore", divide="ignore"):
        wd = d / (e + d) if policy.target_debt_weight is None else np.full(e.shape, policy.target_debt_weight)
        ke = rf + policy.equity_risk_premium
        kd = rf + policy.credit_spread
        value = ke * (1 - wd) + kd * (1 - tax) * wd
    valid = (np.isfinite(e + d + rf + tax) & (e > 0) & (d >= 0) & (tax >= 0) & (tax <= 1)
             & (ke > 0) & (kd >= 0) & (value > 0))
    return np.where(valid, value, np.nan)


def economic_profit(capital, roic, discount, growth, *, fade=True, years=5, min_spread=.01):
    """Reconciled EVA and FCFF on EXACTLY THE SAME capital/profit path.

    NOPAT_t = ROIC_t * IC_(t-1); reinvestment_t = g * IC_(t-1);
    IC_t = IC_(t-1) + reinvestment_t. The specified g is long-run growth;
    during ROIC fade, NOPAT growth also reflects the change in ROIC.
    In fade mode year 1 retains observed ROIC, year 5 reaches WACC, and
    terminal ROIC=WACC. Terminal EVA=0 and terminal enterprise value=IC_T.
    In steady mode terminal ROIC stays at the observed ROIC.

    EVA EV = IC_0 + PV(EVA_1..T) + PV(terminal EVA value).
    FCFF EV = PV(FCFF_1..T) + PV(terminal FCFF value).
    No subtraction/clipping of negative total parent equity is hidden here.
    """
    if years < 2 or min_spread < .01:
        raise ValueError("At least two forecast years and a 1pp safety spread required")
    ic0, r0, w, g = np.broadcast_arrays(*[np.asarray(x, float) for x in (capital, roic, discount, growth)])
    valid = (np.isfinite(ic0 + r0 + w + g) & (ic0 > 0) & (r0 > 0) & (g >= 0)
             & (w - g >= min_spread - 1e-12))
    ic = ic0.copy()
    eva_value = ic0.copy()
    dcf_value = np.zeros_like(ic0)
    first_nopat = np.full_like(ic0, np.nan)
    first_reinvestment_rate = np.full_like(ic0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for year in range(1, years + 1):
            r = r0 + (w - r0) * (year - 1) / (years - 1) if fade else r0
            nopat = r * ic
            investment = g * ic
            reinvestment_rate = investment / nopat
            valid &= (r > 0) & (nopat > 0) & (reinvestment_rate >= 0) & (reinvestment_rate <= 1)
            eva_value += (nopat - w * ic) / (1 + w) ** year
            dcf_value += (nopat - investment) / (1 + w) ** year
            if year == 1:
                first_nopat = nopat
                first_reinvestment_rate = reinvestment_rate
            ic = ic + investment
        terminal_roic = w if fade else r0
        terminal_nopat = terminal_roic * ic
        terminal_investment = g * ic
        terminal_eva_value = (terminal_nopat - w * ic) / (w - g)
        terminal_dcf_value = (terminal_nopat - terminal_investment) / (w - g)
        valid &= (g / terminal_roic >= 0) & (g / terminal_roic <= 1)
        eva_value += terminal_eva_value / (1 + w) ** years
        dcf_value += terminal_dcf_value / (1 + w) ** years
    # At reinvestment=100%, an exactly zero operating EV can still support
    # positive parent equity through IDENTIFIED non-operating assets/cash.
    scale = np.maximum(1, np.abs(ic0))
    eva_value = np.where(np.abs(eva_value) < 1e-12 * scale, 0., eva_value)
    dcf_value = np.where(np.abs(dcf_value) < 1e-12 * scale, 0., dcf_value)
    valid &= np.isfinite(eva_value) & np.isfinite(dcf_value) & (eva_value >= 0) & (dcf_value >= 0)
    err = np.abs(eva_value - dcf_value) / np.maximum(1, np.abs(eva_value))
    if np.any(valid & (err > 1e-10)):
        raise ArithmeticError("EVA and DCF disagree on a common forecast path")
    return {"enterprise_value": np.where(valid, eva_value, np.nan),
            "dcf_enterprise_value": np.where(valid, dcf_value, np.nan),
            "eva_dcf_relative_error": np.where(valid, err, np.nan),
            "next_nopat": np.where(valid, first_nopat, np.nan),
            "reinvestment_rate": np.where(valid, first_reinvestment_rate, np.nan)}


def evaluate(inputs: Fundamentals, growth, policy=DiscountPolicy(), *, fade=True):
    """Pure valuation BEFORE any market-price comparison.

    A Mapping/DataFrame is refused instead of silently accepting extra price
    fields. The caller must bind the finite, audited allowlist explicitly.
    """
    if type(inputs) is not Fundamentals:
        raise TypeError("A closed Fundamentals input, without market-derived fields, is required")
    wacc = book_wacc(inputs.total_book_equity, inputs.debt_book, inputs.risk_free, inputs.tax_rate, policy)
    result = economic_profit(inputs.operating_capital, inputs.historical_roic, wacc, growth,
                             fade=fade, min_spread=policy.minimum_wacc_minus_growth)
    ev = result["enterprise_value"]
    debt, nci, cash, q, nopat = np.broadcast_arrays(*[np.asarray(x, float) for x in
        (inputs.debt_book, inputs.non_controlling_interests, inputs.non_operating_cash, inputs.shares, inputs.historical_nopat)])
    equity = ev - debt - nci + cash
    valid = (np.isfinite(ev) & np.isfinite(debt + nci + cash + q + nopat)
             & (debt >= 0) & (nci >= 0) & (cash >= 0) & (q > 0) & (nopat > 0) & (equity > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        value = equity / q
    result.update(fundamental_value_per_share=np.where(valid, value, np.nan),
                  equity_value=np.where(valid, equity, np.nan), discount_rate=wacc)
    return result


def evaluate_total(inputs: TotalCapitalFundamentals, growth, policy=DiscountPolicy(), *, fade=True):
    """EVA of ALL book capital, with no separate cash deduction/add-back.

    This is an operating-profit-based approximation, NOT a measurement that
    excess cash is zero, nor a guaranteed lower bound on equity value. Its
    forecast treats all capital as invested in the common profit/reinvestment
    path; financial income and separately realizable assets are not valued.
    """
    if type(inputs) is not TotalCapitalFundamentals:
        raise TypeError('Closed TotalCapitalFundamentals inputs required')
    e, d, nci, q, n = np.broadcast_arrays(*[np.asarray(v, float) for v in (
        inputs.total_book_equity, inputs.debt_book, inputs.non_controlling_interests,
        inputs.shares, inputs.historical_nopat)])
    w = book_wacc(e, d, inputs.risk_free, inputs.tax_rate, policy)
    result = economic_profit(e + d, inputs.historical_capital_return, w, growth,
                             fade=fade, min_spread=policy.minimum_wacc_minus_growth)
    equity = result['enterprise_value'] - d - nci
    valid = (np.isfinite(e + d + nci + q + n + equity) & (e > 0) & (d >= 0)
             & (nci >= 0) & (q > 0) & (n > 0) & (equity > 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        value = equity / q
    result.update(fundamental_value_per_share=np.where(valid, value, np.nan),
                  equity_value=np.where(valid, equity, np.nan), discount_rate=w)
    return result


def evaluate_rim(inputs: RimFundamentals, equity_risk_premium=.05, *, horizon=5):
    """Exogenous Ke and the existing RIM's linear ROE fade / clean surplus.

    RI = parent book + PV((ROE_t-Ke)*beginning book). Reconcile independently
    with PV(dividends)+PV(terminal book), using EXACTLY the same equity path.
    Payout is an observed annual ratio held fixed as a forecast assumption;
    missing dividends are never a zero payout. No debt/NCI/cash bridge.
    """
    if type(inputs) is not RimFundamentals:
        raise TypeError('Closed RimFundamentals inputs required')
    if horizon not in (5, 10) or not np.isfinite(equity_risk_premium) or equity_risk_premium <= 0:
        raise ValueError('Predeclared positive equity premium and 5/10-year horizon required')
    book, earn, q, rf, payout = np.broadcast_arrays(*[np.asarray(v, float) for v in (
        inputs.parent_equity, inputs.annual_parent_profit, inputs.shares, inputs.risk_free, inputs.payout)])
    ke = rf + equity_risk_premium
    valid = (np.isfinite(book + earn + q + rf + payout) & (book > 0) & (earn > 0)
             & (q > 0) & (ke > 0) & (payout >= 0) & (payout <= 1))
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        roe = earn / book
        current = book.copy()
        value = book.copy()
        ddm = np.zeros_like(book)
        for year in range(1, horizon + 1):
            rt = ke + (roe - ke) * (horizon - year) / horizon
            profit = rt * current
            dividend = profit * payout
            value += (profit - ke * current) / (1 + ke) ** year
            ddm += dividend / (1 + ke) ** year
            current += profit - dividend
        ddm += current / (1 + ke) ** horizon
        error = np.abs(value - ddm) / np.maximum(1, np.abs(value))
        valid &= np.isfinite(value + ddm) & (value > 0) & (current > 0)
        per_share = value / q
    if np.any(valid & (error > 1e-10)):
        raise ArithmeticError('RIM and dividend DCF disagree on a common equity path')
    return {'fundamental_value_per_share':np.where(valid, per_share, np.nan),
            'equity_value':np.where(valid, value, np.nan),
            'dividend_dcf_value':np.where(valid, ddm, np.nan),
            'rim_dcf_relative_error':np.where(valid, error, np.nan),
            'cost_of_equity':np.where(np.isfinite(ke) & (ke > 0), ke, np.nan),
            'roe':np.where(valid, roe, np.nan)}


def business_growth(sales, prior_sales, nopat, prior_nopat, capital_change=None, marginal_roic=None, cap=.03):
    """Predeclared naive operating-trend forecast; NEVER reverse-solve a price.

    Median of observed sales/NOPAT growth and, if observed, reinvestment-rate
    times marginal ROIC. Both primary growth observations are required and
    must be nonnegative. Bound to [0,3%] BEFORE any return-label inspection.
    This is a forecast assumption, not independently validated predictive skill.
    """
    sales, prev_s, n, prev_n = np.broadcast_arrays(*[np.asarray(x, float) for x in (sales, prior_sales, nopat, prior_nopat)])
    with np.errstate(invalid="ignore", divide="ignore"):
        sg, ng = sales / prev_s - 1, n / prev_n - 1
        valid = (np.isfinite(sales + prev_s + n + prev_n) & (sales > 0) & (prev_s > 0)
                 & (n > 0) & (prev_n > 0) & (sg >= 0) & (ng >= 0))
        candidates = [sg, ng]
        if capital_change is not None and marginal_roic is not None:
            reinvest = np.asarray(capital_change, float) / n
            marginal = np.asarray(marginal_roic, float)
            candidates.append(np.where((reinvest >= 0) & (reinvest <= 1) & (marginal > 0), reinvest * marginal, np.nan))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            trend = np.nanmedian(np.stack(candidates), axis=0)
    return np.where(valid, np.clip(trend, 0, cap), np.nan)


def accounting_foundations(net_income, sales, average_assets, average_equity, shares):
    """Four accounting foundations, not a seven-factor price identity."""
    n, s, a, e, q = np.broadcast_arrays(*[np.asarray(x, float) for x in (net_income, sales, average_assets, average_equity, shares)])
    valid = np.isfinite(n + s + a + e + q) & (s > 0) & (a > 0) & (e > 0) & (q > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = {"net_margin": n / s, "asset_turnover": s / a, "leverage": a / e, "bps_average_equity": e / q}
    return {key: np.where(valid, value, np.nan) for key, value in values.items()}


def corporate_action_units(adjustment_factors, event_types):
    """Only VERIFIED split/consolidation coefficients can change share units.

    J-Quants AdjFactor ALSO contains price-based rights adjustments. Never
    treat a non-unit coefficient as a split without ExRT=1/2. A rights event
    or unclassified non-unit coefficient increments an uncertainty counter;
    a financial share snapshot can be used across an interval only when that
    counter does not change. A later post-event snapshot can restore coverage.
    No market quote, return, or volume ratio is inspected to guess the type.
    """
    factors = np.asarray(adjustment_factors, float)
    kinds = np.asarray(event_types).astype(str)
    if factors.shape != kinds.shape or factors.ndim != 1:
        raise ValueError("One dated corporate-action stream is required")
    known_split = np.isin(kinds, ["1", "1.0", "2", "2.0"])
    no_event = np.isin(kinds, ["", "None", "nan", "<NA>"])
    finite = np.isfinite(factors) & (factors > 0)
    safe = finite & (known_split | ((factors == 1) & no_event))
    applicable = np.where(finite & known_split, factors, 1.)
    return np.cumprod(applicable), np.cumsum(~safe)
