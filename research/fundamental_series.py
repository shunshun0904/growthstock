"""Strict allowlist binding of non-price financial inputs to the EVA kernel."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import fundamental_value as FV

MODELS = ("eva_steady", "eva_fade5", "eva_total_steady", "eva_total_fade5")
RIM_MODELS = {"rim_book5":5, "rim_book10":10}
PRIMARY_MODEL = "eva_total_fade5"
INPUT_FIELDS = ("operating_capital", "roic_operating_tax30", "roic_operating_actual", "nopat_tax30", "nopat_actual",
                "equity", "debt", "nci", "non_operating_cash", "independent_shares", "tv_rf", "actual_tax",
                "annual_fresh", "financial_or_unknown", "net_income_parent", "revision_unverified",
                "revenue", "prev_revenue", "previous_nopat_tax30", "previous_operating_capital",
                "conservative_capital", "previous_conservative_capital", "roic_conservative_tax30", "roic_conservative_actual",
                "rim_parent_equity", "rim_parent_profit", "rim_payout", "rim_financial_fresh")


def meta(erp=(.03, .05, .07), growth=(0., .01, .02, .03)):
    eva = [{"id": f"{model}_{tax}_erp{round(100 * premium):02d}_g{round(100 * g):02d}" if g != "business"
                      else f"{model}_{tax}_erp{round(100 * premium):02d}_business",
             "model": model, "tax": tax, "erp": premium, "growth": g, "priceIndependent": True}
            for model in MODELS for tax in ("tax30", "actual") for premium in erp for g in (*growth, "business")]
    return eva + [{"id":f"{model}_erp{round(100 * premium):02d}", "model":model, "tax":"reported",
                   "erp":premium, "growth":"not_applicable", "horizon":horizon, "priceIndependent":True}
                  for model,horizon in RIM_MODELS.items() for premium in erp]


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
    total_change = numeric("conservative_capital") - numeric("previous_conservative_capital")
    with np.errstate(invalid="ignore", divide="ignore"):
        total_marginal = (numeric("nopat_tax30") - numeric("previous_nopat_tax30")) / total_change
    total_business = FV.business_growth(numeric("revenue"), numeric("prev_revenue"), numeric("nopat_tax30"),
                        numeric("previous_nopat_tax30"), total_change, total_marginal)
    values["business_growth_total_estimate"] = total_business
    for s in meta(erp, growth):
        if s['model'] in RIM_MODELS:
            rim_fresh = d.rim_financial_fresh.astype('boolean').fillna(False).to_numpy(bool)
            inp = FV.RimFundamentals(np.where(rim_fresh, numeric('rim_parent_equity'), np.nan),
                    numeric('rim_parent_profit'), numeric('independent_shares'), numeric('tv_rf'), numeric('rim_payout'))
            result = FV.evaluate_rim(inp, s['erp'], horizon=s['horizon'])
            values['price_'+s['id']] = result['fundamental_value_per_share']
            values['rim_dcf_error_'+s['id']] = result['rim_dcf_relative_error']
            values[f"rim_ke_{round(100*s['erp']):02d}"] = np.where(rim_fresh, result['cost_of_equity'], np.nan)
            values['rim_roe'] = result['roe']
            continue
        tax = .30 if s["tax"] == "tax30" else numeric("actual_tax")
        policy = FV.DiscountPolicy(equity_risk_premium=s["erp"], credit_spread=credit_spread)
        total = s['model'].startswith('eva_total_')
        g = (total_business if total else business) if s["growth"] == "business" else s["growth"]
        if total:
            inputs = FV.TotalCapitalFundamentals(numeric('equity'), numeric('debt'),
                np.where(allowed, numeric('roic_conservative_'+s['tax']), np.nan), numeric('nopat_'+s['tax']),
                numeric('nci'), numeric('independent_shares'), numeric('tv_rf'), tax)
            result = FV.evaluate_total(inputs, g, policy, fade=s['model'].endswith('fade5'))
        else:
            inputs = FV.Fundamentals(
                operating_capital=np.where(allowed, numeric("operating_capital"), np.nan),
                historical_roic=numeric("roic_operating_" + s["tax"]), historical_nopat=numeric("nopat_" + s["tax"]),
                total_book_equity=numeric("equity"), debt_book=numeric("debt"), non_controlling_interests=numeric("nci"),
                non_operating_cash=numeric("non_operating_cash"), shares=numeric("independent_shares"),
                risk_free=numeric("tv_rf"), tax_rate=tax)
            result = FV.evaluate(inputs, g, policy, fade=s["model"] == "eva_fade5")
        sid = s["id"]
        values["price_" + sid] = result["fundamental_value_per_share"]
        values["ev_" + sid] = np.where(np.isfinite(result["fundamental_value_per_share"]), result["enterprise_value"], np.nan)
        values["eva_dcf_error_" + sid] = result["eva_dcf_relative_error"]
        values["reinvestment_" + sid] = result["reinvestment_rate"]
        values[f"fundamental_wacc_{s['tax']}_{round(100 * s['erp']):02d}"] = np.where(fresh & nonfin, result["discount_rate"], np.nan)
    out = pd.DataFrame(values, index=financial.index)
    statistics = {}
    for model in (*MODELS, *RIM_MODELS):
        for tax in (('reported',) if model in RIM_MODELS else ("tax30", "actual")):
            names = [s["id"] for s in meta(erp, growth) if s["model"] == model and s["tax"] == tax and s["growth"] != "business"]
            for measure in (('price',) if model in RIM_MODELS else ("price", "ev")):
                a = out[[measure + "_" + name for name in names]].to_numpy(float)
                count = np.isfinite(a).sum(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stats = [np.nanmin(a, axis=1), np.nanmax(a, axis=1), np.nanmean(a, axis=1), np.nanmedian(a, axis=1), np.nanstd(a, axis=1)]
                for name, stat in zip(("min", "max", "mean", "median", "std"), stats):
                    statistics[f"{model}_{tax}_{measure}_{name}"] = np.where(count > 0, stat, np.nan)
                statistics[f"{model}_{tax}_{measure}_valid_count"] = count
    out = pd.concat([out,pd.DataFrame(statistics,index=financial.index)],axis=1)
    out['strict_fundamental_value_per_share'] = out.eva_fade5_tax30_price_median
    out['strict_fundamental_enterprise_value'] = out.eva_fade5_tax30_ev_median
    out["fundamental_value_per_share"] = out.eva_total_fade5_tax30_price_median
    out["fundamental_enterprise_value"] = out.eva_total_fade5_tax30_ev_median
    out['rim_value_per_share'] = out.rim_book5_reported_price_median
    return out
