#!/usr/bin/env python3
"""Opt-in annual CCC and capital-efficiency candidates; not production features.

An observed debt subtotal is not proof of completeness. Reject detectable
partial IFRS debt and ambiguous parent-equity scope. Missing items are never
evidence of zero. Corrections require a source document and become usable only
the day after its submission, including corrections to comparative balances.

Raw annual snapshots may contain revisions whose original versions are absent.
Submission-date alignment alone does not certify historical point-in-time data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STALE_DAYS = 400
VALUES = ["dso", "dio", "dpo_cogs_proxy", "ccc_cogs_proxy", "nwc_to_sales",
          "gross_margin", "sga_ratio", "op_margin", "tax_rate",
          "nopat_margin_etr_proxy", "ic_turnover_candidate",
          "roic_etr_candidate", "roic_tax30_candidate"]
QUALITY = ["has_annual", "days_since_filing", "period_unknown", "debt_missing",
           "debt_partial", "equity_scope_risk", "debt_reviewed", "equity_reviewed",
           "average_invalid", "financial_or_unknown", "capital_nonpositive",
           "revision_unverified"]
FINANCIAL = {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}
DEBT_JP = ["short_term_loans", "current_portion_lt_loans", "long_term_loans",
           "bonds_payable", "current_portion_bonds_payable", "commercial_papers",
           "short_term_bonds_payable", "other_interest_bearing_debt_jp",
           "lease_obligations_cl", "lease_obligations_ncl"]
DEBT_IFRS = ["ibd_current", "ibd_noncurrent", "ibd_total_unsplit_ifrs",
             "lease_liabilities_cl", "lease_liabilities_ncl",
             "lease_liabilities_total_unsplit_ifrs"]
NUMERIC = list(dict.fromkeys(DEBT_JP + DEBT_IFRS + [
    "revenue", "operating_income", "profit_before_tax", "income_taxes",
    "cost_of_sales", "gross_profit", "sga", "trade_receivables", "trade_payables",
    "inventories", "net_assets", "shareholders_equity", "non_controlling_interests",
    "cash", "total_assets", "total_liabilities"]))
GUARDS = ["basis", "accounting_standard"]
NORMAL_DAYS = {364, 365, 366, 371}


def columns(kind="all"):
    groups = {"values": VALUES, "quality": QUALITY, "all": VALUES + QUALITY}
    if kind not in groups:
        raise KeyError(kind)
    return ["ce_" + c for c in groups[kind]]


def periods_from_fins(fins):
    """Convert J-Quants FY summary records to dated accounting-period evidence."""
    f = fins[fins.CurPerType.eq("FY") & fins.DocType.astype(str).str.contains("FinancialStatements")].copy()
    for c in ["DiscDate", "CurFYSt", "CurFYEn", "CurPerSt", "CurPerEn"]:
        f[c] = pd.to_datetime(f[c], errors="coerce")
    return pd.DataFrame({
        "Code": f.Code.astype(str), "fiscal_year": f.CurFYEn.fillna(f.CurPerEn).dt.year,
        "period_start": f.CurFYSt.fillna(f.CurPerSt),
        "period_end": f.CurFYEn.fillna(f.CurPerEn), "available_date": f.DiscDate,
        "basis": np.where(f.DocType.str.contains("NonConsolidated"), "standalone", "consolidated"),
        "accounting_standard": f.DocType.str.extract(r"_(JP|IFRS|US)$", expand=False).replace({"US": "USGAAP"}),
    }).drop_duplicates()


def _periods(base, periods):
    p = periods.copy()
    for c in ["period_start", "period_end", "available_date"]:
        p[c] = pd.to_datetime(p[c], errors="coerce")
    keys = ["Code", "fiscal_year"] + GUARDS
    x = base[keys + ["filing_avail_date"]].assign(_rid=base.index).merge(p, on=keys, how="left")
    x = x[x.available_date.le(x.filing_avail_date) & x.period_start.notna() & x.period_end.notna()]
    x = x.drop_duplicates(["_rid", "period_start", "period_end"])
    count = x.groupby("_rid").size()
    x = x[x._rid.map(count).eq(1)].set_index("_rid")
    for c in ["period_start", "period_end"]:
        base[c] = pd.to_datetime(x[c].reindex(base.index))
    return base


def _corrections(corrections, base):
    if corrections is None or not len(corrections):
        return pd.DataFrame()
    c = corrections.copy()
    required = ["Code", "fiscal_year", *GUARDS, "source_doc_id", "source_url", "source_submit_date"]
    if any(k not in c for k in required):
        raise ValueError("Corrections require code, fiscal year, scope and source provenance")
    c["Code"] = c.Code.astype(str)
    for key in ["source_doc_id", "source_url", *GUARDS]:
        if c[key].isna().any() or c[key].astype(str).str.strip().eq("").any():
            raise ValueError("Missing correction provenance: " + key)
    if not c.source_url.str.startswith("https://").all():
        raise ValueError("Source URL must use HTTPS")
    c["known_at"] = pd.to_datetime(c.source_submit_date, errors="coerce").dt.normalize() + pd.Timedelta(days=1)
    if c.known_at.isna().any() or c.duplicated(["Code", "fiscal_year", "known_at"]).any():
        raise ValueError("Invalid or ambiguous correction dates")
    scope = c.merge(base[["Code", "fiscal_year", *GUARDS]], on=["Code", "fiscal_year"],
                    how="left", suffixes=("", "_raw"), validate="many_to_one")
    if any((scope[k].ne(scope[k + "_raw"]) | scope[k + "_raw"].isna()).any() for k in GUARDS):
        raise ValueError("Correction scope differs from the original annual record")
    present = []
    for field in ["normalized_debt", "normalized_equity", "normalized_cash"]:
        if field in c:
            c[field] = pd.to_numeric(c[field], errors="raise")
            if c[field].notna().any():
                present.append(field)
            if np.isinf(c[field]).any() or (field != "normalized_equity" and c[field].lt(0).any()):
                raise ValueError("Invalid correction amount: " + field)
    if not present or c[present].notna().sum(axis=1).eq(0).any():
        raise ValueError("Correction has no normalized balance")
    return c.sort_values("known_at", kind="mergesort")


def _normalize(df, prefix=""):
    get = lambda key: df[prefix + key]
    jp = pd.concat([get(c) for c in DEBT_JP], axis=1).sum(axis=1, min_count=1)
    borrow = pd.concat([get("ibd_current") + get("ibd_noncurrent"), get("ibd_total_unsplit_ifrs")], axis=1).max(axis=1)
    lease = pd.concat([pd.concat([get("lease_liabilities_cl"), get("lease_liabilities_ncl")], axis=1).sum(axis=1, min_count=1),
                       get("lease_liabilities_total_unsplit_ifrs")], axis=1).max(axis=1)
    partial = df.accounting_standard.eq("IFRS") & get("ibd_total_unsplit_ifrs").isna() & (
        get("ibd_current").isna() | get("ibd_noncurrent").isna())
    ifrs = pd.concat([borrow, lease], axis=1).sum(axis=1, min_count=1).where(~partial)
    debt = jp.where(df.accounting_standard.eq("JP"), ifrs.where(df.accounting_standard.eq("IFRS")))
    scope = df.accounting_standard.eq("IFRS") & get("non_controlling_interests").gt(0) & np.isclose(
        get("net_assets"), get("shareholders_equity"), rtol=1e-6, atol=1)
    return pd.DataFrame({"debt": debt, "equity": get("net_assets").where(~scope), "cash": get("cash"),
                         "partial": partial, "scope_risk": scope,
                         "debt_reviewed": False, "equity_reviewed": False}, index=df.index)


def annual_panel(fin, periods, companies=None, corrections=None):
    """Build dated candidate events, preserving both raw input and source metadata.

    Corrections may target a preceding year's comparative balance. They affect
    only events at or after source submission + 1 day. No raw file is modified.
    """
    f = fin.copy()
    f["Code"] = f.jq_code.astype(str)
    f["fiscal_year"] = pd.to_numeric(f.fiscal_year, errors="coerce")
    f["submit_date"] = pd.to_datetime(f.submit_date, errors="coerce")
    f = f.dropna(subset=["fiscal_year"]).sort_values(["Code", "fiscal_year", "submit_date"], kind="mergesort", na_position="last")
    f = f.drop_duplicates(["Code", "fiscal_year"], keep="first").reset_index(drop=True)
    if not len(f):
        return pd.DataFrame(columns=["Code", "fiscal_year", "avail_date", "filing_avail_date", "doc_id", *columns()])
    for key in NUMERIC:
        f[key] = pd.to_numeric(f[key], errors="coerce").astype(float) if key in f else np.nan
    f["filing_avail_date"] = f.submit_date.dt.normalize() + pd.Timedelta(days=1)
    f.loc[f.doc_id.isna(), "filing_avail_date"] = pd.NaT
    f = _periods(f, periods)
    if "industry" not in f:
        f["industry"] = f.edinet_code.map(companies.drop_duplicates("edinet_code").set_index("edinet_code").industry) if companies is not None else None
    corr = _corrections(corrections, f)
    f["avail_date"] = f.filing_avail_date
    events = [f]
    if len(corr):
        # A correction to an old year must not displace a newer fiscal report.
        for code, date in corr[["Code", "known_at"]].drop_duplicates().itertuples(index=False, name=None):
            known = f[f.Code.eq(code) & f.filing_avail_date.le(date)]
            if len(known):
                row = known.sort_values(["fiscal_year", "filing_avail_date"], kind="mergesort").tail(1).copy()
                row["avail_date"] = date
                events.append(row)
    out = pd.concat(events, ignore_index=True).drop_duplicates(["Code", "fiscal_year", "avail_date"])
    prevcols = NUMERIC + ["submit_date", "period_end", "doc_id"]
    prev = f[["Code", "fiscal_year", *GUARDS, *prevcols]].copy()
    prev["fiscal_year"] += 1
    prev = prev.rename(columns={key: "prev_" + key for key in prevcols})
    out = out.merge(prev, on=["Code", "fiscal_year", *GUARDS], how="left", validate="many_to_one")
    cur, prior = _normalize(out), _normalize(out, "prev_")
    out["correction_sources"] = ""
    for _, c in corr.iterrows():
        for offset, norm in [(0, cur), (1, prior)]:
            mask = out.Code.eq(c.Code) & out.fiscal_year.eq(c.fiscal_year + offset) & out.avail_date.ge(c.known_at)
            for name in ["debt", "equity", "cash"]:
                key = "normalized_" + name
                if key in c and pd.notna(c[key]):
                    norm.loc[mask, name] = c[key]
                    if name in ["debt", "equity"]:
                        norm.loc[mask, name + "_reviewed"] = True
                        norm.loc[mask, "partial" if name == "debt" else "scope_risk"] = False
            out.loc[mask, "correction_sources"] += c.source_doc_id + ";"
    days = (out.period_end - out.period_start).dt.days + 1
    comparable = (out.prev_submit_date.notna() & out.prev_submit_date.le(out.avail_date) &
                  out.prev_period_end.add(pd.Timedelta(days=1)).eq(out.period_start) & days.isin(NORMAL_DAYS) &
                  out.basis.isin(["consolidated", "standalone"]) & out.accounting_standard.isin(["JP", "IFRS", "USGAAP"]))
    rev = out.revenue.where(out.revenue.gt(0))
    cogs = out.cost_of_sales.where(out.cost_of_sales.gt(0))
    averages = {key: ((out[key] + out["prev_" + key]) / 2).where(comparable & out[key].ge(0) & out["prev_" + key].ge(0))
                for key in ["trade_receivables", "inventories", "trade_payables"]}
    vals = {}
    vals["dso"] = averages["trade_receivables"] / rev * days
    vals["dio"] = averages["inventories"] / cogs * days
    vals["dpo_cogs_proxy"] = averages["trade_payables"] / cogs * days
    vals["ccc_cogs_proxy"] = vals["dso"] + vals["dio"] - vals["dpo_cogs_proxy"]
    vals["nwc_to_sales"] = (averages["trade_receivables"] + averages["inventories"] - averages["trade_payables"]) / rev
    for metric, raw in [("gross_margin", "gross_profit"), ("sga_ratio", "sga"), ("op_margin", "operating_income")]:
        vals[metric] = out[raw] / rev
    tax = (out.income_taxes / out.profit_before_tax.where(out.profit_before_tax.gt(0))).where(lambda x: x.between(0, 1))
    vals["tax_rate"] = tax
    vals["nopat_margin_etr_proxy"] = vals["op_margin"] * (1 - tax)
    ic = (cur.equity + cur.debt - cur.cash).where(cur.equity.gt(0))
    old_ic = (prior.equity + prior.debt - prior.cash).where(prior.equity.gt(0))
    avg_ic = ((ic + old_ic) / 2).where(comparable & ic.gt(0) & old_ic.gt(0))
    vals["ic_turnover_candidate"] = rev / avg_ic
    vals["roic_etr_candidate"] = out.operating_income * (1 - tax) / avg_ic
    vals["roic_tax30_candidate"] = out.operating_income * .7 / avg_ic
    excluded = out.industry.isna() | out.industry.isin(FINANCIAL)
    for key, value in vals.items():
        out["ce_" + key] = value.where(~excluded).replace([np.inf, -np.inf], np.nan)
    flags = {"has_annual": True, "period_unknown": days.isna(), "debt_missing": cur.debt.isna() | prior.debt.isna(),
             "debt_partial": cur.partial | prior.partial, "equity_scope_risk": cur.scope_risk | prior.scope_risk,
             "debt_reviewed": cur.debt_reviewed & prior.debt_reviewed,
             "equity_reviewed": cur.equity_reviewed & prior.equity_reviewed,
             "average_invalid": ~comparable, "financial_or_unknown": excluded,
             "capital_nonpositive": ic.le(0) | old_ic.le(0), "revision_unverified": True}
    for key, value in flags.items():
        out["ce_" + key] = float(value) if isinstance(value, bool) else value.astype(float)
    out["ce_days_since_filing"] = np.nan  # Calculated against the sample date in attach.
    return out.sort_values(["avail_date", "Code", "fiscal_year"], kind="mergesort").reset_index(drop=True)


def attach(frame, panel, stale_days=STALE_DAYS):
    """Attach the newest disclosed fiscal year, retaining input order and index."""
    clashes = set(columns() + ["ce_fiscal_year", "ce_avail_date", "ce_source_doc_id"]) & set(frame.columns)
    if clashes:
        raise ValueError("Capital-efficiency columns already attached: " + str(sorted(clashes)))
    d = frame.copy()
    original_index = d.index.copy()
    d["Date"] = pd.to_datetime(d.Date)
    d["Code"] = d.Code.astype(str)
    d["_ce_order"] = np.arange(len(d))
    p = panel[panel.avail_date.notna()].copy()
    if not len(p):
        d = d.drop(columns="_ce_order")
        for key in columns() + ["ce_fiscal_year", "ce_avail_date", "ce_source_doc_id"]:
            d[key] = np.nan
        return d
    p = p.sort_values(["avail_date", "Code", "fiscal_year"], kind="mergesort")
    p = p[p.fiscal_year.ge(p.groupby("Code").fiscal_year.cummax())].drop_duplicates(["Code", "avail_date"], keep="last")
    p = p[["Code", "fiscal_year", "avail_date", "filing_avail_date", "doc_id", *columns()]].rename(columns={
        "fiscal_year": "ce_fiscal_year", "avail_date": "ce_avail_date", "doc_id": "ce_source_doc_id"})
    d = pd.merge_asof(d.sort_values("Date", kind="mergesort"), p.sort_values("ce_avail_date", kind="mergesort"),
                      left_on="Date", right_on="ce_avail_date", by="Code", direction="backward")
    age = (d.Date - d.filing_avail_date).dt.days
    stale = age.gt(stale_days)
    d["ce_days_since_filing"] = age
    d.loc[stale, columns()] = np.nan
    d = d.sort_values("_ce_order", kind="mergesort").drop(columns=["_ce_order", "filing_avail_date"])
    d.index = original_index
    return d
