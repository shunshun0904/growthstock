"""Strict allowlist binding of non-price financial inputs to the EVA kernel."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import fundamental_value as FV

MODELS = ("eva_steady", "eva_fade5")
INPUT_FIELDS = ("operating_capital", "roic_operating_tax30", "roic_operating_actual", "nopat_tax30", "nopat_actual",
                "equity", "debt", "nci", "non_operating_cash", "independent_shares", "tv_rf", "actual_tax",
                "annual_fresh", "financial_or_unknown", "net_income_parent", "revision_unverified",
                "revenue", "prev_revenue", "previous_nopat_tax30", "previous_operating_capital")


def meta(erp=(.03, .05, .07), growth=(0., .01, .02, .03)):
    return [{"id": f"{model}_{tax}_erp{round(100 * premium):02d}_g{round(100 * g):02d}" if g != "business"
                      else f"{model}_{tax}_erp{round(100 * premium):02d}_business",
             "model": model, "tax": tax, "erp": premium, "growth": g, "priceIndependent": True}
            for model in MODELS for tax in ("tax30", "actual") for premium in erp for g in (*growth, "business")]


def calculate(financial, *, erp=(.03, .05, .07), growth=(0., .01, .02, .03), credit_spread=.015):
    extra = set(financial) - set(INPUT_FIELDS)
    if extra:
        raise ValueError("Non-financial or unknown inputs refused by EVA binding: " + str(sorted(extra)))
    d = financial.reindex(columns=INPUT_FIELDS).copy()
    numeric = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy(float)
    fresh = d.annual_fresh.astype("boolean").fillna(False).to_numpy(bool)
    nonfin = ~d.financial_or_unknown.astype("boolean").fillna(True).to_numpy(bool)
    positive_profit = numeric("net_income_parent") > 0
    allowed = fresh & nonfin & positive_profit
    values = {}
    change = numeric("operating_capital") - numeric("previous_operating_capital")
    with np.errstate(invalid="ignore", divide="ignore"):
        marginal_roic = (numeric("nopat_tax30") - numeric("previous_nopat_tax30")) / change
    business = FV.business_growth(numeric("revenue"), numeric("prev_revenue"), numeric("nopat_tax30"),
                                 numeric("previous_nopat_tax30"), change, marginal_roic)
    values["business_growth_estimate"] = business
    for s in meta(erp, growth):
        tax = .30 if s["tax"] == "tax30" else numeric("actual_tax")
        policy = FV.DiscountPolicy(equity_risk_premium=s["erp"], credit_spread=credit_spread)
        inputs = FV.Fundamentals(
            operating_capital=np.where(allowed, numeric("operating_capital"), np.nan),
            historical_roic=numeric("roic_operating_" + s["tax"]), historical_nopat=numeric("nopat_" + s["tax"]),
            total_book_equity=numeric("equity"), debt_book=numeric("debt"), non_controlling_interests=numeric("nci"),
            non_operating_cash=numeric("non_operating_cash"), shares=numeric("independent_shares"),
            risk_free=numeric("tv_rf"), tax_rate=tax)
        g = business if s["growth"] == "business" else s["growth"]
        result = FV.evaluate(inputs, g, policy, fade=s["model"] == "eva_fade5")
        sid = s["id"]
        values["price_" + sid] = result["fundamental_value_per_share"]
        values["ev_" + sid] = np.where(np.isfinite(result["fundamental_value_per_share"]), result["enterprise_value"], np.nan)
        values["eva_dcf_error_" + sid] = result["eva_dcf_relative_error"]
        values["reinvestment_" + sid] = result["reinvestment_rate"]
        values[f"fundamental_wacc_{s['tax']}_{round(100 * s['erp']):02d}"] = np.where(fresh & nonfin, result["discount_rate"], np.nan)
    out = pd.DataFrame(values, index=financial.index)
    for model in MODELS:
        for tax in ("tax30", "actual"):
            names = [s["id"] for s in meta(erp, growth) if s["model"] == model and s["tax"] == tax and s["growth"] != "business"]
            for measure in ("price", "ev"):
                a = out[[measure + "_" + name for name in names]].to_numpy(float)
                count = np.isfinite(a).sum(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stats = [np.nanmin(a, axis=1), np.nanmax(a, axis=1), np.nanmean(a, axis=1), np.nanmedian(a, axis=1), np.nanstd(a, axis=1)]
                for name, stat in zip(("min", "max", "mean", "median", "std"), stats):
                    out[f"{model}_{tax}_{measure}_{name}"] = np.where(count > 0, stat, np.nan)
                out[f"{model}_{tax}_{measure}_valid_count"] = count
    out["fundamental_value_per_share"] = out.eva_fade5_tax30_price_median
    out["fundamental_enterprise_value"] = out.eva_fade5_tax30_ev_median
    return out
