#!/usr/bin/env python3
"""Build private daily research tables and optional derived per-stock JSON.

No network, model registration, training, or production file writes. Input
directory contains existing bars_*.parquet, fins_*.parquet, topix_*.parquet,
edinet_fin.parquet, edinet_companies.parquet and jgb_yields.csv caches.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capital_efficiency as CE
import theoretical_value as TV
import roic_wacc_valuation as R
import valuation_gap as G
import fundamental_value as FV
import fundamental_series as FS


def code5(value):
    text = str(value).strip().upper()
    if not re.fullmatch(r"[0-9][0-9A-Z]{3,4}", text):
        raise ValueError("Invalid security code")
    return text + "0" if len(text) == 4 else text


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    tmp.write_bytes(gzip.compress(body, compresslevel=6, mtime=0) if path.suffix == ".gz" else body)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parent_income_snapshot(statements, latest):
    """Price-free version of TV's period/scope-aware TTM accounting identity.

    Raw PARENT profit and equity amounts precede ALL per-share conversion.
    No conversion from legacy price-adjusted EPS/BPS is permitted here.
    """
    snap = statements[statements._scope.eq(latest._scope)].sort_values(
        ['DiscDate','DiscTime','DiscNo'], kind='mergesort').drop_duplicates(['CurFYSt','CurPerType'], keep='last')
    annual = snap[snap.CurPerType.eq('FY') & (snap.CurPerEn-snap.CurFYSt).dt.days.between(350,380)]
    profit, used = np.nan, latest.DiscDate
    if latest.CurPerType == 'FY':
        if 350 <= (latest.CurPerEn-latest.CurFYSt).days <= 380:
            profit = latest.NP
    elif latest.CurPerType in ('1Q','2Q','3Q'):
        fy = annual[(latest.CurFYSt-annual.CurPerEn).dt.days.between(1,15)]
        q = snap[snap.CurPerType.eq(latest.CurPerType)
                 & (latest.CurFYSt-snap.CurFYSt).dt.days.between(350,380)
                 & (latest.CurPerEn-snap.CurPerEn).dt.days.between(350,380)]
        if len(fy) and len(q):
            a, b = fy.sort_values('CurPerEn').iloc[-1], q.sort_values('CurPerEn').iloc[-1]
            profit, used = latest.NP+a.NP-b.NP, max(used,a.DiscDate,b.DiscDate)
    payout, payout_date, payout_period = np.nan, pd.NaT, pd.NaT
    if len(annual):
        a = annual.sort_values('CurPerEn').iloc[-1]
        if 0 <= (latest.CurPerEn-a.CurPerEn).days <= 550 and a.NP > 0:
            observed = a.DivTotalAnn/a.NP
            if np.isfinite(observed) and 0 <= observed <= 1:
                payout, payout_date, payout_period = observed, a.DiscDate, a.CurPerEn
    return {'rim_parent_equity':latest.ShEq, 'rim_parent_profit':profit,
            'rim_payout':payout, 'rim_source_date':used, 'rim_payout_source_date':payout_date,
            'rim_payout_period_end':payout_period}


def financial_events(fins, prices, sessions):
    """Reuse TV's period/scope-aware TTM engine AT DISCLOSURE EVENTS only.

    Shares are outstanding minus OBSERVED treasury shares, adjusted using
    ratios of split factors. Forecast EPS is independently reconstructed from
    published company forecast profit/current shares, avoiding an undocumented
    future-split convention of provider forecast EPS. Forecast and TTM never
    substitute for one another. Missing/withdrawn forecasts remain missing.
    """
    f = fins.copy()
    f["Code"] = f.Code.map(code5)
    for c in ["DiscDate", "CurFYSt", "CurFYEn", "CurPerEn"]:
        f[c] = pd.to_datetime(f[c], errors="coerce")
    if "DiscTime" not in f:
        f["DiscTime"] = ""
    if "DiscNo" not in f:
        f["DiscNo"] = ""
    f["available"] = R.next_session(f.DiscDate, sessions).to_numpy()
    f = f[f.available.notna() & f.CurPerEn.le(f.DiscDate)].copy()
    f = f.sort_values(["DiscDate", "DiscTime", "DiscNo"], kind="mergesort")
    f["_scope"] = f.DocType.str.extract(r"_(Consolidated|NonConsolidated)_", expand=False)
    for key in ["ShOutFY", "TrShFY", "FNP", "NxFNp", "NP", "ShEq", "DivTotalAnn"]:
        f[key] = pd.to_numeric(f[key], errors="coerce") if key in f else np.nan
    rows = []
    for code, group in f.groupby("Code", sort=False):
        pc = prices[prices.Code.eq(code)].sort_values("Date")
        if pc.empty:
            continue
        dates = pd.DatetimeIndex(pc.Date)
        factors = (pc.close_raw / pc.close.where(pc.close.gt(0))).to_numpy(float)
        kinds = pc.get("action_type", pd.Series(None, index=pc.index, dtype=object))
        actions, uncertain = FV.corporate_action_units(pc.action_factor, kinds)
        for date in sorted(group.available.unique()):
            date = pd.Timestamp(date)
            if date > pc.Date.max():
                continue
            known = group[group.available.le(date)]
            statements = known[known.DocType.str.contains("FinancialStatements", na=False)]
            if statements.empty:
                continue
            statements = statements.assign(_conso=statements._scope.eq("Consolidated").astype(int))
            latest = statements.sort_values(["CurPerEn", "_conso", "DiscDate", "DiscTime", "DiscNo"], kind="mergesort").iloc[-1]
            def factor(day):
                if pd.isna(day):
                    return np.nan
                j = dates.searchsorted(day, side="right") - 1
                return factors[j] if j >= 0 and (day - dates[j]).days <= 10 else np.nan
            event_factor = factor(date)
            period_factor = factor(latest.CurPerEn)
            shares = (latest.ShOutFY - latest.TrShFY) * period_factor / event_factor
            if not np.isfinite(shares) or shares <= 0:
                shares = np.nan
            def action_index(day):
                if pd.isna(day) or day < dates[0]:
                    return np.nan
                j = dates.searchsorted(day, side="right") - 1
                return actions[j] if j >= 0 and (day - dates[j]).days <= 10 else np.nan
            def uncertainty(day):
                if pd.isna(day) or day < dates[0]:
                    return np.nan
                j = dates.searchsorted(day, side="right") - 1
                return uncertain[j] if j >= 0 and (day - dates[j]).days <= 10 else np.nan
            event_action = action_index(date)
            independent_shares = (latest.ShOutFY - latest.TrShFY) * action_index(latest.CurPerEn) / event_action
            action_unverified = uncertainty(date) != uncertainty(latest.CurPerEn)
            if action_unverified:
                independent_shares = np.nan
            # A later correction to an OLD report must not select an old forecast.
            guidance = known[known.CurFYEn.eq(latest.CurFYEn) &
                             known.DocType.str.contains("FinancialStatements|EarnForecastRevision", na=False)]
            forecast = np.nan
            forecast_date = pd.NaT
            if len(guidance):
                last = guidance.iloc[-1]
                forecast = last.NxFNp if latest.CurPerType == "FY" else last.FNP
                forecast_date = last.DiscDate
            j = dates.searchsorted(date, side="right") - 1
            close = pc.close_raw.iloc[j] if j >= 0 else np.nan
            rows.append({"Code": code, "Date": date, "close_raw": close,
                "financial_source_date": latest.DiscDate, "financial_source_id": str(latest.DiscNo),
                "financial_period_end": latest.CurPerEn, "financial_available_date": date,
                "financial_event_factor": event_factor, "shares_at_event": shares,
                "shares_independent_at_event": independent_shares, "event_action_index": event_action,
                "event_action_uncertainty": uncertainty(date), "period_action_unverified": action_unverified,
                "forecast_profit": forecast, "forecast_source_date": forecast_date,
                **parent_income_snapshot(statements, latest)})
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    events['eps_ttm_at_event'] = events.rim_parent_profit/events.shares_at_event.where(events.shares_at_event.gt(0))
    events['bps_at_event'] = events.rim_parent_equity/events.shares_at_event.where(events.shares_at_event.gt(0))
    return events.drop(columns="close_raw").rename(columns={"Date": "financial_event_date"})


def attach_financial(keys, events):
    d = keys.copy().reset_index(drop=True).assign(_financial_order=np.arange(len(keys)))
    if events.empty:
        for c in ["shares", "independent_shares", "eps_forecast", "eps_ttm", "forecast_source_date", "financial_source_date",
                  "financial_source_id", "financial_available_date", "financial_event_date",
                  "rim_parent_equity", "rim_parent_profit", "rim_payout", "rim_source_date", "rim_payout_source_date", "rim_payout_period_end"]:
            d[c] = np.nan
        d['rim_financial_fresh'] = False
        return d.drop(columns="_financial_order")
    d = pd.merge_asof(d.sort_values("Date"), events.sort_values("financial_event_date"),
                      left_on="Date", right_on="financial_event_date", by="Code", direction="backward")
    current_factor = d.close_raw / d.close.where(d.close.gt(0))
    d["shares"] = d.shares_at_event * d.financial_event_factor / current_factor
    d["independent_shares"] = d.shares_independent_at_event * d.event_action_index / d.action_index
    d["corporate_action_unverified"] = d.period_action_unverified.astype("boolean").fillna(True) | d.event_action_uncertainty.ne(d.action_uncertainty)
    d.loc[d.corporate_action_unverified, "independent_shares"] = np.nan
    d["eps_ttm"] = d.eps_ttm_at_event * current_factor / d.financial_event_factor
    d["bps_reported"] = d.bps_at_event * current_factor / d.financial_event_factor
    d["eps_forecast"] = d.forecast_profit / d.shares.where(d.shares.gt(0))
    # Shares and earnings cannot stay perpetually fresh after reporting stops.
    stale = (d.Date - pd.to_datetime(d.financial_source_date)).dt.days.gt(365)
    d['rim_financial_fresh'] = (d.Date-pd.to_datetime(d.financial_source_date)).dt.days.between(0,365)
    payout_fresh = (d.Date-pd.to_datetime(d.rim_payout_period_end)).dt.days.between(0,550)
    d.loc[~payout_fresh, 'rim_payout'] = np.nan
    d.loc[stale, ['rim_parent_equity','rim_parent_profit','rim_payout']] = np.nan
    d.loc[stale, ["shares", "independent_shares", "eps_ttm", "eps_forecast", "bps_reported"]] = np.nan
    return d.sort_values("_financial_order").drop(columns="_financial_order").reset_index(drop=True)


def annual_dupont_inputs(annual, fins):
    """Coherent annual J-Quants parent income/equity; no forecast/TTM mixing."""
    out = annual.copy()
    fields = ["dupont_net_income", "dupont_sales", "dupont_average_assets", "dupont_average_parent_equity"]
    for c in fields:
        out[c] = np.nan
    out["dupont_source_date"] = pd.NaT
    if out.empty:
        return out
    f = fins[fins.CurPerType.eq("FY") & fins.DocType.str.contains("FinancialStatements", na=False)].copy()
    for c in ["CurPerEn", "DiscDate"]:
        f[c] = pd.to_datetime(f[c], errors="coerce")
    f["Code"] = f.Code.map(code5)
    f["basis"] = np.where(f.DocType.str.contains("NonConsolidated"), "standalone", "consolidated")
    f["accounting_standard"] = f.DocType.str.extract(r"_(JP|IFRS|US)$", expand=False).replace({"US": "USGAAP"})
    for idx, r in out.iterrows():
        snap = f[f.Code.eq(r.Code) & f.DiscDate.lt(r.avail_date) & f.basis.eq(r.basis)
                 & f.accounting_standard.eq(r.accounting_standard)]
        cur = snap[snap.CurPerEn.eq(r.period_end)].sort_values(["DiscDate", "DiscTime"])
        prev = snap[snap.CurPerEn.eq(r.prev_period_end)].sort_values(["DiscDate", "DiscTime"])
        if cur.empty:
            continue
        a = cur.iloc[-1]
        out.loc[idx, fields[:2]] = [a.NP, a.Sales]
        out.loc[idx, "dupont_source_date"] = a.DiscDate
        if len(prev) and r.average_valid:
            b = prev.iloc[-1]
            out.loc[idx, fields[2:]] = [(a.TA + b.TA) / 2, (a.ShEq + b.ShEq) / 2]
            out.loc[idx, "dupont_source_date"] = max(a.DiscDate, b.DiscDate)
    return out


def make_series(code, prices, market, rates, annual, events, config=R.Config(), *, include_market_comparison=True):
    p = prices[prices.Code.eq(code)].sort_values("Date").drop_duplicates("Date", keep="last")
    p = p.copy()
    p["action_index"], p["action_uncertainty"] = FV.corporate_action_units(
        p.action_factor, p.get("action_type", pd.Series(None, index=p.index, dtype=object)))
    dates = pd.DatetimeIndex(market.Date)
    dates = dates[(dates >= p.Date.min()) & (dates <= p.Date.max())]
    keys = pd.DataFrame({"Code": code, "Date": dates})
    keys = keys.merge(p, on=["Code", "Date"], how="left", validate="one_to_one")
    fin = attach_financial(keys, events[events.Code.eq(code)] if len(events) else events)
    d = R.attach_annual(fin, annual[annual.Code.eq(code)], config.annual_stale_days)
    if "annual_fresh" not in d:
        # No annual report is not a zero-valued financial statement.
        d["annual_fresh"] = False
        d["financial_or_unknown"] = True
        d["revision_unverified"] = True
        for name in G.CONTROLS + ["debt", "nci", "non_operating_cash", "nopat_tax30", "nopat_actual",
                                  "roic_conservative_actual", "roic_operating_actual"]:
            if name not in d:
                d[name] = np.nan
    d["price"] = d.close_raw
    beta = (TV.beta(keys, p, market, rates, window=config.beta_window, min_obs=config.beta_min_observations)
            if include_market_comparison else pd.DataFrame(index=keys.index))
    rf = TV.riskfree(keys.Date, rates)
    d = pd.concat([d.reset_index(drop=True), beta, rf], axis=1)
    # No claim that annual EDINET net_income's parent/NCI scope is audited.
    # Where available this is a conservative LOSS exclusion, separately flagged.
    d["net_income_parent"] = pd.to_numeric(d.get("dupont_net_income", pd.Series(np.nan, index=d.index)), errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        if include_market_comparison:
            d = R.scenarios(d, config)
    # MARKET-DEPENDENT COMPARISON ONLY. Preserve it under explicit names.
    for col in ["theoretical_price", "theoretical_ev", "valuation_gap", "ev_valuation_gap", "wacc", "unavailable_reason"]:
        d["comparison_" + col] = d[col] if include_market_comparison else np.nan
    d["comparison_ev_bridge_per_share"] = (d.debt + d.nci - d.non_operating_cash) / d.shares.where(d.shares.gt(0))
    d["previous_nopat_tax30"] = pd.to_numeric(d.get("prev_operating_income", pd.Series(np.nan, index=d.index)), errors="coerce") * .7
    # Closed allowlist: no price, beta, market cap, PER/PBR, or implied growth.
    independent = FS.calculate(d.reindex(columns=FS.INPUT_FIELDS), erp=config.erp,
                               growth=config.growth, credit_spread=config.credit_spread)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        d = pd.concat([d, independent], axis=1)
        d["theoretical_price"] = d.fundamental_value_per_share
        d["theoretical_ev"] = d.fundamental_enterprise_value
        d['roic'] = d.roic_conservative_tax30
        d["wacc"] = d.fundamental_wacc_tax30_05
        d["scenario_valid_count"] = d.eva_total_fade5_tax30_price_valid_count
        d['scenario_total_count'] = len(config.erp)*len(config.growth)
        # Only AFTER FV.evaluate has returned can market prices enter comparisons.
        d["valuation_gap"] = G.compare_with_market(d.fundamental_value_per_share, d.price)
        bridge = d.debt + d.nci
        market_ev = d.price * d.independent_shares + bridge
        d["ev_valuation_gap"] = (market_ev / d.theoretical_ev).where(market_ev.gt(0) & d.theoretical_ev.gt(0))
        # Market-implied growth is terminal DIAGNOSTICS ONLY, with next
        # NOPAT fixed to the independent steady model's first forecast.
        fixed_next_nopat = d.roic * pd.to_numeric(d.get("conservative_capital", pd.Series(np.nan, index=d.index)), errors="coerce")
        d["implied_growth_fixed_nopat"] = R.implied_growth_fixed_nopat(market_ev, fixed_next_nopat, d.roic, d.wacc)
        d["growth_gap"] = d.business_growth_total_estimate - d.implied_growth_fixed_nopat
        d["roic_wacc_spread"] = d.roic - d.wacc
        d["eps_forecast"] = d.forecast_profit / d.independent_shares.where(d.independent_shares.gt(0)) if "forecast_profit" in d else np.nan
        d['eps_ttm'] = d.rim_parent_profit / d.independent_shares.where(d.independent_shares.gt(0))
        d['rim_valuation_gap'] = G.compare_with_market(d.rim_value_per_share, d.price)
        d['rim_valuation_log_gap'] = np.log(d.rim_valuation_gap)
        d['rim_financial_age'] = (d.Date-pd.to_datetime(d.financial_source_date)).dt.days
        for kind in ("forecast", "ttm"):
            eps = d["eps_" + kind].where(d["eps_" + kind].gt(0))
            d["market_per_" + kind] = d.price / eps
            d["theoretical_per_" + kind] = d.theoretical_price / eps
        reason = pd.Series("", index=d.index)
        for label, mask in {
            "annual_stale_or_missing": ~d.annual_fresh.fillna(False),
            "financial_or_unknown": d.financial_or_unknown.astype("boolean").fillna(True),
            "debt_missing": d.debt.isna(), "nci_missing": d.nci.isna(),
            "shares_missing_or_nonpositive": ~d.independent_shares.gt(0),
            "total_capital_return_unavailable_or_nonpositive": ~d.roic.gt(0),
            "risk_free_missing": d.tv_rf.isna(),
            "company_loss_or_profit_unverified": ~d.net_income_parent.gt(0),
            "corporate_action_type_unverified": d.get("corporate_action_unverified", pd.Series(True, index=d.index)).astype("boolean").fillna(True),
        }.items():
            reason.loc[mask] += label + ";"
        reason.loc[d.theoretical_price.isna() & reason.eq("")] = "no_valid_positive_equity_scenario;"
        d["unavailable_reason"] = reason.str.rstrip(";")
        rim_reason = pd.Series('', index=d.index)
        for label,mask in {
            'parent_equity_missing_or_nonpositive':~d.rim_parent_equity.gt(0),
            'parent_ttm_profit_missing_or_nonpositive':~d.rim_parent_profit.gt(0),
            'payout_missing_or_invalid':~d.rim_payout.between(0,1),
            'shares_missing_or_nonpositive':~d.independent_shares.gt(0),
            'financial_stale_or_missing':~d.rim_financial_fresh.fillna(False),
            'risk_free_missing':d.tv_rf.isna(),
        }.items():
            rim_reason.loc[mask] += label+';'
        rim_reason.loc[d.rim_value_per_share.isna() & rim_reason.eq('')] = 'no_valid_positive_equity_scenario;'
        d['rim_unavailable_reason'] = rim_reason.str.rstrip(';')
        d["valuation_data_quality"] = (d.theoretical_price.notna() & ~d.revision_unverified.astype("boolean").fillna(True)).astype(float)
    # Even the adjusted INTRINSIC VALUE series must not use a stock-price
    # ratio. Normalize both final series to independently tracked share units.
    d["share_unit_index"] = d.action_index
    d["price_adjusted"] = d.close_raw / d.share_unit_index
    d["theoretical_price_adjusted"] = d.theoretical_price / d.share_unit_index
    d["ev_bridge_per_share"] = (d.debt + d.nci) / d.independent_shares.where(d.independent_shares.gt(0))
    d['strict_ev_bridge_per_share'] = (d.debt+d.nci-d.non_operating_cash) / d.independent_shares.where(d.independent_shares.gt(0))
    numeric = lambda name: pd.to_numeric(d.get(name, pd.Series(np.nan, index=d.index)), errors="coerce")
    factors = FV.accounting_foundations(numeric("dupont_net_income"), numeric("dupont_sales"),
                      numeric("dupont_average_assets"), numeric("dupont_average_parent_equity"), d.independent_shares)
    for key, value in factors.items():
        d[key] = value
    d["reinvestment_rate"] = d["reinvestment_eva_total_fade5_tax30_erp05_g01"]
    if "avail_date" not in d:
        d["avail_date"] = pd.NaT
    d = G.time_features(d)
    rim = d[['Code','Date','price_adjusted','action_uncertainty']].copy()
    rim['valuation_gap'], rim['roic'], rim['wacc'] = d.rim_valuation_gap, d.rim_roe, d.rim_ke_05
    rim['theoretical_price_adjusted'] = d.rim_value_per_share/d.share_unit_index
    rim['avail_date'] = d.financial_event_date
    rim = G.time_features(rim)
    d = pd.concat([d, rim[list(G.RIM_TIME_RENAME)].rename(columns=G.RIM_TIME_RENAME)], axis=1)
    for source in ["avail_date", "financial_event_date", "forecast_source_date", "tv_rf_date", "rim_source_date", "rim_payout_source_date"]:
        if source in d:
            values = pd.to_datetime(d[source], errors="coerce")
            valid = values.isna() | (values.le(d.Date) if source.endswith("event_date") or source == "avail_date" else values.lt(d.Date))
            if not valid.all():
                raise ValueError("Unreleased source used: " + source)
    return d


PUBLIC_FIELDS = ["Date", "price", "price_adjusted", "theoretical_price", "valuation_gap", "ev_valuation_gap",
    "eps_forecast", "eps_ttm", "market_per_forecast", "market_per_ttm", "theoretical_per_forecast", "theoretical_per_ttm",
    "roic", "roic_conservative_tax30", "roic_conservative_actual", "roic_operating_tax30", "roic_operating_actual",
    "wacc", "actual_tax", "roic_wacc_spread",
    "tv_beta", "tv_beta_raw", "tv_beta_n", "tv_rf", "tv_rf_date", "annual_age", "avail_date",
    "financial_source_date", "financial_available_date", "forecast_source_date", "unavailable_reason",
    "revision_unverified", "cash_identified", "debt_reviewed", "equity_reviewed", "valuation_data_quality",
    "scenario_valid_count", "scenario_total_count", "implied_growth_fixed_nopat", "growth_gap", "ev_bridge_per_share",
    "net_margin", "asset_turnover", "leverage", "bps_average_equity", "reinvestment_rate", "business_growth_estimate",
    "comparison_theoretical_price", "comparison_wacc", "comparison_valuation_gap", "comparison_ev_bridge_per_share",
    "corporate_action_unverified", "share_unit_index", "action_uncertainty", *G.TIME,
    'strict_fundamental_value_per_share','strict_ev_bridge_per_share','rim_value_per_share','rim_unavailable_reason',
    'business_growth_total_estimate','rim_source_date','rim_payout_source_date', *G.RIM_FIELDS]


def public_payload(code, name, frame, config=R.Config()):
    meta = [{"id": R.scenario_id(m, t, e, g), "model": m, "tax": t, "erp": e, "growth": g}
            for m in R.MODELS for t in R.TAXES for e in config.erp for g in config.growth]
    meta = FS.meta(config.erp, config.growth) + meta
    cols = [c for c in PUBLIC_FIELDS if c in frame]
    data = frame[cols].copy()
    for col in data:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime("%Y-%m-%d")
    # Pandas handles nullable floats/dates without serializing NaN/Infinity.
    rows = json.loads(data.to_json(orient="records", date_format="iso"))
    for i, row in enumerate(rows):
        row["scenarios"] = {s["id"]: float(frame["price_" + s["id"]].iloc[i]) for s in meta
                            if np.isfinite(frame["price_" + s["id"]].iloc[i])}
        row["reinvestmentScenarios"] = {s["id"]: float(frame["reinvestment_" + s["id"]].iloc[i]) for s in meta
                                       if "reinvestment_" + s["id"] in frame and np.isfinite(frame["reinvestment_" + s["id"]].iloc[i])}
        if row["scenarios"]:
            row["scenarioInvalid"] = {s["id"]: int(frame["invalid_" + s["id"]].iloc[i]) for s in meta
                                      if "invalid_" + s["id"] in frame and frame["invalid_" + s["id"]].iloc[i]}
        for key, prefix in [("waccScenarios", "fundamental_wacc_"), ("comparisonWaccScenarios", "wacc_")]:
            row[key] = {f"{t}_{round(100 * e):02d}": (float(frame[f"{prefix}{t}_{round(100 * e):02d}"].iloc[i])
                if np.isfinite(frame[f"{prefix}{t}_{round(100 * e):02d}"].iloc[i]) else None) for t in R.TAXES for e in config.erp}
        row['rimKeScenarios'] = {f'{round(100*e):02d}':float(frame[f'rim_ke_{round(100*e):02d}'].iloc[i])
                                for e in config.erp if np.isfinite(frame[f'rim_ke_{round(100*e):02d}'].iloc[i])}
    # Columnar JSON repeats field names once, keeping full histories modest.
    columns = list(dict.fromkeys([key for row in rows for key in row]))
    values = [[row.get(key) for key in columns] for row in rows]
    return {"schemaVersion": 1, "researchOnly": True, "code": code[:-1], "name": name, "columns": columns, "data": values,
            "scenarios": meta, "creditSpread": config.credit_spread, "scenarioRangeIsConfidenceInterval": False,
            "primaryModel": FS.PRIMARY_MODEL, "primaryPriceIndependent": True,
            'cashTreatment':'cash included in total book capital; no separate deduction or add-back; approximate',
            'evGapBasis':'total capital value including cash, not cash-adjusted operating EV',
            "limitations": ["主EVAは現金区分なしの近似。現金内訳が0と確認できたという意味ではない", "過去APIデータの訂正履歴は未照合",
                            "主モデルの資本構成は簿価。負債コストは国債金利＋信用スプレッドの仮定",
                            "EPSの会社予想系列とTTM実績系列は別系列", "ROICは利用可能な年次財務に基づく"]}


def load_inputs(raw, codes):
    codes = set(codes)
    paths = {kind: sorted(raw.glob(kind + "_*.parquet")) for kind in ("bars", "fins", "topix")}
    if any(not v for v in paths.values()):
        raise ValueError("Missing cached bars, financial statements, or TOPIX")
    prices = []
    for p in paths["bars"]:
        cols = ["Code", "Date", "C", "AdjC", "AdjFactor"]
        if "ExRT" in pq.ParquetFile(p).schema.names:
            cols.append("ExRT")
        b = pd.read_parquet(p, columns=cols, filters=[("Code", "in", sorted(codes))])
        if "ExRT" not in b:
            b["ExRT"] = None
        prices.append(b.rename(columns={"C": "close_raw", "AdjC": "close", "AdjFactor": "action_factor", "ExRT": "action_type"}))
    prices = pd.concat(prices, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices.Date)
    prices["Code"] = prices.Code.map(code5)
    prices = prices.drop_duplicates(["Code", "Date"], keep="last").sort_values(["Code", "Date"])
    fins = pd.concat([pd.read_parquet(p, filters=[("Code", "in", sorted(codes))]) for p in paths["fins"]], ignore_index=True)
    market = pd.concat([pd.read_parquet(p) for p in paths["topix"]], ignore_index=True)
    market["Date"] = pd.to_datetime(market.Date)
    market = market.sort_values("Date").drop_duplicates("Date", keep="last")
    rates = pd.read_csv(raw / "jgb_yields.csv", skiprows=1, na_values=["-"])
    rates["Date"] = pd.to_datetime(rates.Date)
    fin = pd.read_parquet(raw / "edinet_fin.parquet")
    valid_codes = fin.jq_code.astype(str).str.upper().str.fullmatch(r"[0-9][0-9A-Z]{3,4}")
    fin = fin[valid_codes.fillna(False)].copy()
    fin = fin[fin.jq_code.astype(str).map(code5).isin(codes)].copy()
    fin["jq_code"] = fin.jq_code.map(code5)
    companies = pd.read_parquet(raw / "edinet_companies.parquet")
    return prices, fins, market, rates, fin, companies, paths


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.data_dir
    predictions = json.loads(args.predictions.read_text())
    names = {code5(s["code"]): s.get("name", s["code"]) for s in predictions.get("candidates", [])}
    selected = [code5(c) for c in args.codes.split(",")] if args.codes else list(names)
    if args.universe_dataset:
        selected = sorted(set(pd.read_parquet(args.universe_dataset, columns=["Code"]).Code.map(code5)) | set(names))
    if not selected:
        raise ValueError("No current inference securities selected")
    public_codes = set(names) if args.universe_dataset else set(selected)
    if args.public_dir and len(public_codes) > args.public_limit:
        raise ValueError("Explicitly increase --public-limit or export a smaller UI universe")
    config = R.Config(credit_spread=args.credit_spread)
    prices, fins, market, rates, fin, companies, paths = load_inputs(raw, selected)
    panel = CE.annual_panel(fin, CE.periods_from_fins(fins), companies)
    adjustments = pd.read_csv(args.adjustments, dtype={"Code": str}) if args.adjustments else None
    annual = R.reviewed_states(panel, market.Date, adjustments)
    annual = annual_dupont_inputs(annual, fins)
    events = financial_events(fins, prices, market.Date)
    annual.to_parquet(args.output / "annual_states.parquet", index=False)
    events.to_parquet(args.output / "financial_events.parquet", index=False)
    catalog, coverage, reasons = [], [], []
    series_dir = args.output / "series"
    series_dir.mkdir(exist_ok=True)
    source_paths = [*paths["bars"], *paths["fins"], *paths["topix"], raw / "edinet_fin.parquet",
                    raw / "edinet_companies.parquet", raw / "jgb_yields.csv", args.predictions]
    if args.adjustments:
        source_paths.append(args.adjustments)
    if args.universe_dataset:
        source_paths.append(args.universe_dataset)
    source = {"inputs": {str(p): digest(p) for p in source_paths}, "codes": selected,
              "selection": "historical training universe" if args.universe_dataset else "current inference targets" if not args.codes else "explicit codes",
              "config": config.__dict__, "historical_versions_verified": False,
              "primary_model": "price-independent cash-unsegregated approximate EVA; external-Ke RIM comparison",
              "primary_input_fields": FS.INPUT_FIELDS,
              "share_units": "disclosed outstanding minus observed treasury shares; only verified ExRT=1/2 split/consolidation factors",
              "modules": {p.name: digest(p) for p in [Path(__file__), HERE / "fundamental_value.py", HERE / "fundamental_series.py",
                          HERE / "roic_wacc_valuation.py", HERE / "valuation_gap.py", HERE / "theoretical_value.py", HERE / "capital_efficiency.py"]},
              "latest_price": str(prices.Date.max().date()), "latest_market": str(market.Date.max().date()),
              "latest_risk_free_reference": str(rates.Date.max().date())}
    json_write(args.output / "inputs.json", source)
    input_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    for code in selected:
        if not prices.Code.eq(code).any():
            catalog.append({"code": code[:-1], "name": names.get(code, code[:-1]), "rows": 0, "reason": "株価履歴なし"})
            continue
        path = series_dir / f"{code}.parquet"
        tag = series_dir / f"{code}.meta.json"
        reusable = args.resume and path.exists() and tag.exists() and json.loads(tag.read_text()).get("input_key") == input_key
        d = pd.read_parquet(path) if reusable else make_series(code, prices, market, rates, annual, events, config)
        if not reusable:
            temp = path.with_suffix(".parquet.tmp")
            d.to_parquet(temp, index=False)
            temp.replace(path)
            json_write(tag, {"input_key": input_key, "file_sha256": digest(path)})
        fields = ["roic_conservative_tax30", "roic_conservative_actual", "roic_operating_tax30", "wacc",
                  "theoretical_price", "valuation_gap", "ev_valuation_gap", "valuation_gap_zscore_252",
                  'strict_fundamental_value_per_share','rim_value_per_share','rim_valuation_gap','rim_gap_zscore_252']
        for field in fields:
            count = int(d[field].notna().sum())
            coverage.append({"Code": code, "feature": field, "available": count, "rows": len(d), "fraction": count / len(d)})
        for reason, count in d.unavailable_reason.value_counts().items():
            reasons.append({"Code": code, "reason": reason or "available", "rows": int(count)})
        entry = {"code": code[:-1], "name": names.get(code, code[:-1]), "rows": len(d),
                 "from": str(d.Date.min().date()), "to": str(d.Date.max().date()),
                 "valuedRows": int(d.theoretical_price.notna().sum()), "latestReason": d.unavailable_reason.iloc[-1]}
        entry["file"] = f"{code}.json" + ("" if getattr(args, "plain_json", False) else ".gz")
        catalog.append(entry)
        if args.public_dir and code in public_codes:
            json_write(args.public_dir / entry["file"], public_payload(code, entry["name"], d, config))
        print(f"{code}: {len(d)} sessions; strict valuations {entry['valuedRows']}", flush=True)
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    pd.DataFrame(reasons).to_csv(args.output / "unavailable_reasons.csv", index=False)
    index = {"schemaVersion": 1, "researchOnly": True, "asOf": str(prices.Date.max().date()), "stocks": catalog,
             "selection": source["selection"], "notForProductionModels": True}
    json_write(args.output / "catalog.json", index)
    if args.public_dir:
        json_write(args.public_dir / "index.json", {**index, "stocks": [s for s in catalog if code5(s["code"]) in public_codes]})
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, default=HERE.parent / "public/data/predictions.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--public-dir", type=Path)
    p.add_argument("--plain-json", action="store_true", help="Write uncompressed per-security JSON (index is always plain)")
    p.add_argument("--codes", default="", help="Default: current inference securities, not hand-picked examples")
    p.add_argument("--universe-dataset", type=Path, help="Full historical training universe; UI export stays limited to current targets")
    p.add_argument("--public-limit", type=int, default=200)
    p.add_argument("--resume", action="store_true", help="Reuse only atomically completed securities with matching data/code fingerprints")
    p.add_argument("--adjustments", type=Path, help="Private source-backed balance reviews")
    p.add_argument("--credit-spread", type=float, default=.015)
    args = p.parse_args()
    protected = {HERE / "model", HERE / "_data", HERE.parent / "public/data"}
    destination = args.output.resolve()
    if any(destination == x.resolve() for x in protected) or destination.is_relative_to((HERE / "model").resolve()) or destination.is_relative_to((HERE.parent / "public").resolve()):
        p.error("Use an independent research output directory")
    if args.public_dir and destination.is_relative_to(args.public_dir.resolve()):
        p.error("Public exports must not contain the private research directory")
    run(args)


if __name__ == "__main__":
    main()
