"""Describe realised history versus the declared (uncalibrated) assumptions.
Usage: python e122_assumptions.py OUTPUT_DIRECTORY ANNUAL_PANEL_PARQUET
These are record-weighted historical diagnostics, not forecast-error calibration.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import sys
P=Path(sys.argv[1])
d=pd.read_parquet(P/'dataset.parquet')
p=pd.read_parquet(sys.argv[2])
keys=['Code','fiscal_year','avail_date']
fields=['prev_revenue','prev_operating_income','prev_trade_receivables','prev_trade_payables','prev_inventories',
        'income_taxes','profit_before_tax','interest_expenses','ce_average_invalid']
p=p.dropna(subset=['fiscal_year','avail_date']).drop_duplicates(keys)
x=d.merge(p[keys+fields],on=keys,how='left',validate='many_to_one')
for c in fields:x[c]=pd.to_numeric(x[c],errors='coerce')
base=x.mv_dcf_flex_input_ready.eq(True)
comparable=base&x.ce_average_invalid.eq(0)&x.prev_revenue.gt(0)
rows=[]
def describe(name,series,valid,inside=None,note=''):
    s=pd.to_numeric(series.loc[valid],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    rows.append({'check':name,'rows':len(s),'median':s.median() if len(s) else None,
       'p05':s.quantile(.05) if len(s) else None,'p95':s.quantile(.95) if len(s) else None,
       'within_assumed_range_pct':100*inside.loc[s.index].mean() if inside is not None and len(s) else None,
       'note':note})
growth=x.revenue/x.prev_revenue-1
describe('historical_sales_growth',growth,comparable,growth.between(-.05,.10),
         'One realised YoY change per sample; not a forecast-error distribution; repeated companies/years included')
margin=x.operating_income/x.revenue
old_margin=x.prev_operating_income/x.prev_revenue
change=margin/old_margin-1
describe('historical_margin_relative_change',change,comparable&old_margin.gt(0),change.between(-.2,.2),
         'Compared with assumed +/-20% margin prior; actual YoY changes are not prediction errors')
tax=x.income_taxes/x.profit_before_tax
describe('historical_effective_tax_rate',tax,base&x.profit_before_tax.gt(0)&tax.between(0,1),tax.between(.25,.35),
         'Company effective rate differs from marginal tax rate required by DCF')
cost=x.interest_expenses/x.v_debt
describe('historical_interest_over_ending_debt',cost,base&x.v_debt.gt(0)&x.interest_expenses.ge(0),
         (cost>=x.tv_rf+.005)&(cost<=x.tv_rf+.03),
         'Ending debt subtotal, not average debt; historical coupon cost differs from future marginal funding cost')
ratio=x.capex/x.v_cash_capex
describe('disclosed_capex_over_cash_capex',ratio,base&x.capex.gt(0)&x.v_cash_capex.gt(0),ratio.between(.8,1.2),
         'Limited overlapping rows cannot establish that the two investment definitions are interchangeable')
wc=x.v_nwc/x.revenue
oldwc=(x.prev_trade_receivables+x.prev_inventories-x.prev_trade_payables)/x.prev_revenue
describe('annual_NWC_ratio_change_pp',(wc-oldwc)*100,comparable,None,
         'Forecast freezes current working-capital/sales ratio; only AR+inventory-AP are included')
for kind in ['cash','flex']:
    col='mv_dcf_'+kind+'_terminal_pv_ratio'
    v=x[col];valid=x['mv_dcf_'+kind+'_median'].gt(0)&v.notna()
    describe('terminal_PV_ratio_'+kind,v,valid,v.le(.8),'Ratio>80% indicates strong terminal-assumption dependence')
    width=(x['mv_reverse_'+kind+'_growth_high_premium']-x['mv_reverse_'+kind+'_growth_low_premium'])*100
    describe('reverse_growth_ERP3_to7_width_pp_'+kind,width,x['mv_reverse_'+kind+'_growth'].notna(),None,
             'Other assumptions held fixed; difference in annual growth percentage points')
pd.DataFrame(rows).to_csv(P/'assumption_evidence.csv',index=False)
peer=[]
for method in ['per','pbr','ev_ebitda']:
    for n in [5,10,20]:
        available=x['mv_'+method+'_peers'].ge(n)
        peer.append({'method':method,'min_other_peers':n,'input_candidate_rows':int(available.sum()),
                     'pct_all_rows':100*available.mean()})
pd.DataFrame(peer).to_csv(P/'peer_count_sensitivity.csv',index=False)
reverse=[]
for kind in ['cash','flex']:
    ready=x['mv_reverse_'+kind+'_input_ready'].eq(True);roots=x['mv_reverse_'+kind+'_roots']
    reverse.append({'method':'reverse_'+kind,'input_rows':int(ready.sum()),
                    'unique_growth_rows':int(x['mv_reverse_'+kind+'_growth'].notna().sum()),
                    'no_root_in_bounds':int((ready&roots.eq(0)).sum()),
                    'multiple_roots':int((ready&roots.gt(1)).sum()),
                    'invalid_rate_or_nonpositive_market_EV':int((ready&roots.isna()).sum()),
                    'max_relative_pricing_residual':x['mv_reverse_'+kind+'_relative_residual'].max()})
pd.DataFrame(reverse).to_csv(P/'reverse_diagnostics.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
