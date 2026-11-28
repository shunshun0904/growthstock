"""Capital-efficiency contracts: missing debt, scope, and disclosure timing."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research"))
import capital_efficiency as CE


def inputs(standard="JP", **changes):
    rows=[]
    for year in [2024, 2025]:
        row=dict(jq_code="10000",fiscal_year=year,doc_id=f"S{year}",
                 submit_date=f"{year}-06-25 15:00",basis="consolidated",
                 accounting_standard=standard,industry="サービス業",revenue=1000.,
                 operating_income=100.,profit_before_tax=100.,income_taxes=30.,
                 net_assets=500.,cash=100.,total_assets=1000.,total_liabilities=500.,
                 short_term_loans=100.,long_term_loans=100.,trade_receivables=100.,
                 trade_payables=40.,inventories=50.,cost_of_sales=500.,gross_profit=500.,sga=400.)
        for key,value in changes.items():row[key]=value[year-2024] if isinstance(value,list) else value
        rows.append(row)
    fin=pd.DataFrame(rows)
    periods=pd.DataFrame([dict(Code="10000",fiscal_year=y,basis="consolidated",
        accounting_standard=standard,period_start=f"{y-1}-04-01",period_end=f"{y}-03-31",
        available_date=f"{y}-05-01") for y in [2024,2025]])
    return fin,periods


def correction(year=2024,**values):
    row=dict(Code="10000",fiscal_year=year,basis="consolidated",accounting_standard="JP",
        source_doc_id="S2025",source_url="https://example.com/annual.pdf",
        source_submit_date="2025-06-25 15:00")
    row.update(values)
    return row


class CapitalEfficiency(unittest.TestCase):
    def panel(self,standard="JP",**changes):
        fin,periods=inputs(standard,**changes)
        return CE.annual_panel(fin,periods)

    def test_decomposition_and_ccc_independent_example(self):
        p=self.panel().set_index("fiscal_year").loc[2025]
        self.assertAlmostEqual(p.ce_ic_turnover_candidate,1000/600)
        self.assertAlmostEqual(p.ce_roic_etr_candidate,70/600)
        self.assertAlmostEqual(p.ce_roic_etr_candidate,p.ce_nopat_margin_etr_proxy*p.ce_ic_turnover_candidate)
        self.assertAlmostEqual(p.ce_ccc_cogs_proxy,36.5+36.5-29.2)

    def test_missing_debt_is_not_zero(self):
        p=self.panel(short_term_loans=None,long_term_loans=None)
        self.assertTrue(p.ce_ic_turnover_candidate.isna().all())
        self.assertTrue(p.ce_debt_missing.eq(1).all())
        self.assertTrue(p.ce_dso.notna().any())

    def test_ifrs_one_side_is_rejected(self):
        p=self.panel("IFRS",ibd_current=100.,ibd_noncurrent=None,lease_liabilities_cl=10.)
        self.assertTrue(p.ce_roic_etr_candidate.isna().all())
        self.assertTrue(p.ce_debt_partial.eq(1).all())

    def test_ifrs_lease_only_is_rejected(self):
        p=self.panel("IFRS",lease_liabilities_cl=10.,lease_liabilities_ncl=20.)
        self.assertTrue(p.ce_roic_etr_candidate.isna().all())

    def test_ifrs_total_and_parts_are_not_added_twice(self):
        p=self.panel("IFRS",ibd_current=100.,ibd_noncurrent=100.,ibd_total_unsplit_ifrs=200.,
                     lease_liabilities_cl=10.,lease_liabilities_ncl=20.,lease_liabilities_total_unsplit_ifrs=30.)
        self.assertAlmostEqual(p.iloc[-1].ce_ic_turnover_candidate,1000/630)

    def test_parent_equity_scope_is_rejected(self):
        p=self.panel("IFRS",ibd_current=100.,ibd_noncurrent=100.,
                     shareholders_equity=500.,non_controlling_interests=50.)
        self.assertTrue(p.ce_ic_turnover_candidate.isna().all())
        self.assertTrue(p.ce_equity_scope_risk.eq(1).all())

    def test_correction_does_not_backfill_earlier_samples(self):
        fin,periods=inputs(short_term_loans=None,long_term_loans=None)
        corrections=pd.DataFrame([correction(y,normalized_debt=200.,normalized_equity=500.) for y in [2024,2025]])
        p=CE.annual_panel(fin,periods,corrections=corrections)
        d=pd.DataFrame({'Code':['10000']*3,'Date':['2025-06-25','2025-06-26','2024-06-26']})
        a=CE.attach(d,p)
        self.assertTrue(pd.isna(a.ce_roic_etr_candidate.iloc[0]))
        self.assertAlmostEqual(a.ce_roic_etr_candidate.iloc[1],70/600)
        self.assertTrue(pd.isna(a.ce_roic_etr_candidate.iloc[2]))

    def test_later_correction_creates_new_event_without_rejuvenating_filing(self):
        fin,periods=inputs()
        c=pd.DataFrame([correction(2025,normalized_debt=400.,source_submit_date="2025-07-01")])
        p=CE.annual_panel(fin,periods,corrections=c)
        a=CE.attach(pd.DataFrame({'Code':['10000']*2,'Date':['2025-07-01','2025-07-02']}),p)
        self.assertAlmostEqual(a.ce_ic_turnover_candidate.iloc[0],1000/600)
        self.assertAlmostEqual(a.ce_ic_turnover_candidate.iloc[1],1000/700)
        self.assertEqual(a.ce_days_since_filing.iloc[1],6)

    def test_correction_requires_provenance_and_consistent_scope(self):
        fin,periods=inputs()
        for field,value in [('source_url',''),('source_doc_id',''),('source_submit_date',None)]:
            c=correction(2025,normalized_debt=200.);c[field]=value
            with self.assertRaises(ValueError):CE.annual_panel(fin,periods,corrections=pd.DataFrame([c]))
        c=correction(2025,normalized_debt=400.,basis="standalone")
        with self.assertRaises(ValueError):CE.annual_panel(fin,periods,corrections=pd.DataFrame([c]))
        c=pd.DataFrame([correction(2024,normalized_debt=200.),correction(2025)])
        with self.assertRaises(ValueError):CE.annual_panel(fin,periods,corrections=c)

    def test_nonpositive_capital_and_invalid_tax_do_not_form_roic(self):
        p=self.panel(cash=1000.)
        self.assertTrue(p.ce_roic_etr_candidate.isna().all())
        self.assertTrue(p.ce_capital_nonpositive.eq(1).any())
        p=self.panel(profit_before_tax=-100.)
        self.assertTrue(p.ce_roic_etr_candidate.isna().all())
        self.assertTrue(p.ce_roic_tax30_candidate.notna().any())

    def test_average_requires_same_basis_normal_contiguous_period_and_known_prior(self):
        fin,periods=inputs()
        variants=[]
        z=fin.copy();z.loc[0,'basis']='standalone';variants.append((z,periods))
        z=fin.copy();z.loc[0,'submit_date']=None;variants.append((z,periods))
        z=periods.copy();z.loc[1,'period_start']='2024-05-01';variants.append((fin,z))
        z=periods.copy();z.loc[1,'period_start']='2023-04-01';variants.append((fin,z))
        for f,p in variants:
            self.assertTrue(CE.annual_panel(f,p).ce_ic_turnover_candidate.isna().all())

    def test_ambiguous_or_future_period_is_not_used(self):
        fin,p=inputs()
        z=p.copy();z.loc[1,'available_date']='2026-01-01'
        self.assertTrue(CE.annual_panel(fin,z).ce_ic_turnover_candidate.isna().all())
        z=pd.concat([p,p.iloc[[1]].assign(period_end='2025-12-31')])
        self.assertTrue(CE.annual_panel(fin,z).ce_ic_turnover_candidate.isna().all())

    def test_financials_and_unknown_industry_are_excluded(self):
        for industry in ['銀行業','保険業','その他金融業',None]:
            p=self.panel(industry=industry)
            self.assertTrue(p[CE.columns('values')].isna().all().all())

    def test_latest_report_selected_even_if_less_complete(self):
        fin,p=inputs(operating_income=[100.,None])
        a=CE.attach(pd.DataFrame({'Code':['10000'],'Date':['2025-06-26']}),CE.annual_panel(fin,p))
        self.assertEqual(a.ce_fiscal_year.iloc[0],2025)
        self.assertTrue(pd.isna(a.ce_op_margin.iloc[0]))

    def test_staleness_order_and_duplicate_input_index(self):
        p=self.panel()
        d=pd.DataFrame({'Code':['10000','99990','10000'],'Date':['2026-09-01','2025-06-26','2025-06-26'],'x':[3,2,1]},index=[7,7,3])
        a=CE.attach(d,p)
        self.assertEqual(a.index.tolist(),[7,7,3]);self.assertEqual(a.x.tolist(),[3,2,1])
        self.assertTrue(a.ce_op_margin.iloc[:2].isna().all())
        self.assertAlmostEqual(a.ce_op_margin.iloc[2],.1)

    def test_current_source_duplicate_uses_first_original_and_inputs_are_unmodified(self):
        fin,p=inputs();z=fin.iloc[[1]].assign(doc_id='CORRECTED',submit_date='2025-10-01',revenue=99999.)
        fin=pd.concat([fin,z],ignore_index=True);before=fin.copy(deep=True)
        q=CE.annual_panel(fin,p)
        self.assertEqual(q.loc[q.fiscal_year.eq(2025),'revenue'].iloc[0],1000)
        pd.testing.assert_frame_equal(fin,before)

    def test_empty_input_and_feature_names(self):
        fin,p=inputs();empty=CE.annual_panel(fin.iloc[:0],p)
        a=CE.attach(pd.DataFrame({'Code':['10000'],'Date':['2025-07-01']}),empty)
        self.assertTrue(a[CE.columns('values')].isna().all().all())
        self.assertEqual(len(CE.columns()),len(set(CE.columns())))
        with self.assertRaises(KeyError):CE.columns('bad')

    def test_correction_can_explicitly_confirm_zero_debt(self):
        fin,p=inputs(short_term_loans=None,long_term_loans=None)
        c=pd.DataFrame([correction(y,normalized_debt=0.) for y in [2024,2025]])
        a=CE.annual_panel(fin,p,corrections=c)
        self.assertAlmostEqual(a.iloc[-1].ce_ic_turnover_candidate,2.5)

    def test_missing_document_does_not_create_a_dated_report(self):
        fin,p=inputs();fin['doc_id']=None
        a=CE.attach(pd.DataFrame({'Code':['10000'],'Date':['2025-07-01']}),CE.annual_panel(fin,p))
        self.assertTrue(a[CE.columns('values')].isna().all().all())

    def test_late_older_report_does_not_displace_newer_fiscal_year(self):
        fin,p=inputs(submit_date=['2025-08-01','2025-06-25'])
        a=CE.attach(pd.DataFrame({'Code':['10000'],'Date':['2025-08-02']}),CE.annual_panel(fin,p))
        self.assertEqual(a.ce_fiscal_year.iloc[0],2025)

    def test_fiscal_periods_use_fy_period_end_when_fy_end_is_empty(self):
        f=pd.DataFrame([dict(Code='10000',DiscDate='2025-05-01',DocType='FYFinancialStatements_Consolidated_JP',
            CurPerType='FY',CurFYSt='2024-04-01',CurFYEn=None,CurPerSt='2024-04-01',CurPerEn='2025-03-31')])
        p=CE.periods_from_fins(f)
        self.assertEqual(p.fiscal_year.iloc[0],2025)
        self.assertEqual(p.period_end.iloc[0],pd.Timestamp('2025-03-31'))


if __name__=='__main__':unittest.main()
