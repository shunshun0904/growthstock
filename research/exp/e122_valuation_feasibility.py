#!/usr/bin/env python3
"""Evaluate valuation arithmetic availability across ALL frozen mart rows.

No training, tuning, subscription upgrade, or production changes. Peers come
from the full historical observed market rather than selected trading samples.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import valuation_methods as M


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))


def sectors(frame,master):
    frame=frame.copy();frame['Code']=frame.Code.astype(str)
    right=master.rename(columns={'Date':'sector_date'})
    right=right.copy();right['Code']=right.Code.astype(str)
    out=pd.merge_asof(frame.sort_values('Date'),right.sort_values('sector_date'),
         left_on='Date',right_on='sector_date',by='Code',direction='backward')
    out.loc[(out.Date-out.sector_date).dt.days.gt(62),'S33']=None
    return out


def ev_fields(x):
    x=x.copy()
    ok=(x.v_fresh.fillna(False)&x.v_period_ok.fillna(False)&x.v_jp_basis.fillna(False)&
        ~x.v_scope_risk.fillna(True)&~x.v_partial.fillna(True)&x.v_debt.ge(0)&
        x.v_debt.le(x.total_liabilities+1)&x.v_cash.ge(0)&x.v_nci.ge(0)&x.MktCap.gt(0)&
        x.v_equity.gt(0)&x.v_balance_ok.fillna(False)&x.depreciation.ge(0)&
        x.operating_income.notna()&x.S33.notna()&~x.S33.isin(M.FINANCIAL))
    x['EBITDA']=(x.operating_income+x.depreciation).where(ok).where(lambda v:v>0)
    x['EV_EBITDA']=(x.MktCap+x.v_debt+x.v_nci-x.v_cash)/x.EBITDA
    return x


def run(args):
    p=args.output;p.mkdir(parents=True,exist_ok=True)
    raw=args.data_dir
    original=pd.read_parquet(args.dataset)
    base=original[['Code','Date','close_raw','tv_beta','tv_rf']].copy()
    base['rid']=np.arange(len(base))
    panel=pd.read_parquet(args.annual_panel)
    corr=pd.read_csv(args.corrections,dtype={'Code':str})
    states=M.annual_states(panel,corr)
    master_files=sorted(raw.glob('master_hist_*.parquet'))+[raw/'master.parquet']
    master=pd.concat([pd.read_parquet(f,columns=['Code','Date','S33']) for f in master_files],ignore_index=True)
    master.Date=pd.to_datetime(master.Date)
    master=master.sort_values(['Date','Code']).drop_duplicates(['Code','Date'],keep='last')
    annual_cols=['Code','avail_date','filing_avail_date','fiscal_year','period_end','basis','accounting_standard',
       'operating_income','depreciation','total_liabilities','v_equity','v_nci','v_cash','v_debt','v_scope_risk',
       'v_partial','v_period_ok','v_jp_basis','v_balance_ok','v_debt_reviewed','v_equity_reviewed']
    chunks=[];peer_results={k:[] for k in ['per','pbr','ev_ebitda']};year_checks=[]
    val_files=[]
    for year,group in base.groupby(base.Date.dt.year):
        print('Historical peers and annual inputs:',year,flush=True)
        f=raw/f'valuation_{year}.parquet';val_files.append(f)
        v=pd.read_parquet(f,columns=['Code','Date','PER','PBR','MktCap'])
        v.Date=pd.to_datetime(v.Date)
        v=v[v.Date.isin(group.Date.unique())].copy()
        v.MktCap=pd.to_numeric(v.MktCap,errors='coerce')*1e6
        v=sectors(v,master)
        targets=group.merge(v,on=['Code','Date'],how='left',validate='one_to_one')
        target_fund=ev_fields(M.attach(targets,states))
        chunks.append(target_fund)
        # Positive actual PER/PBR (TTM/latest-book) match provider definitions.
        for metric,key in [('PER','per'),('PBR','pbr')]:
            q=M.peer_prices(v,target_fund,metric)
            q['rid']=target_fund.rid.to_numpy();peer_results[key].append(q)
        peers=ev_fields(M.attach(v,states[annual_cols]))
        q=M.peer_prices(peers,target_fund,'EV_EBITDA',enterprise=True)
        q['rid']=target_fund.rid.to_numpy();peer_results['ev_ebitda'].append(q)
        year_checks.append({'year':int(year),'market_rows':len(v),'market_codes':v.Code.nunique(),
                            'sample_rows':len(group),'ev_peer_rows':int(peers.EV_EBITDA.gt(0).sum())})
        del peers,v
    d=pd.concat(chunks,ignore_index=True).sort_values('rid').reset_index(drop=True)
    assert len(d)==len(original)
    np.testing.assert_array_equal(d[['Code','Date']].to_numpy(),original[['Code','Date']].to_numpy())
    for key in peer_results:
        q=pd.concat(peer_results[key]).set_index('rid').reindex(d.rid).reset_index(drop=True)
        for col in q:d[f'mv_{key}_{col}']=q[col].to_numpy()
    plan={'denominator_rows':len(d),'denominator_codes':d.Code.nunique(),'date_min':d.Date.min(),'date_max':d.Date.max(),
      'peer_population':'all observed valuation rows, historical TSE33, self excluded, minimum 10 other positive peers',
      'peer_sector_max_age_days':62,'annual_max_age_days':400,'financial_future_values_used':False,
      'DCF_basis':'JP nonfinancial only; actual cash CapEx vs cash-or-disclosed-investment proxy',
      'DCF_scenarios':324,'high_growth':[-.05,0.,.05,.10],'ERP':[.03,.05,.07],
      'debt_spread':[.005,.015,.03],'terminal_growth':[0.,.01,.02],'tax':[.25,.30,.35],
      'stable_ROIC':'WACC; terminal reinvestment = g/WACC*NOPAT','explicit_horizon':5,
      'NWC_proxy':'receivables + inventories - trade payables; other operating balances excluded',
      'debt_warning':'observed component sum is provisional, not proof that missing components equal zero',
      'reverse_growth_bounds':[-.5,.5],'reverse_fixed':{'ERP':.05,'spread':.015,'g_terminal':.01,'tax':.30},
      'MC_draws':args.draws,'MC_seed':20261104,'MC_priors_calibrated':False,
      'MC_priors':{'growth':'triangular(-5%,0%,10%)','ERP':'triangular(3%,5%,7%)',
        'spread':'triangular(.5%,1.5%,3%)','tax':'uniform(25%,35%)','margin':'triangular(.8*m,m,min(1,1.2*m))',
        'terminal_g':'uniform(0,min(2%,WACC-1%)), conditional on WACC>1%'},
      'MC_dependence':'parameters independent except terminal_g/WACC; each path persistent for 5 years',
      'equity_floor':'max(signed equity,0); zero paths retained; not a distress/option valuation',
      'valid_fraction_min':.8,'asset_stress':'book NAV minus 0/20/40% PPE, 0/20/40% inventories, 0/50/100% intangibles',
      'fair_value_adjusted_NAV':'unavailable, no asset-specific market values or comprehensive off-balance claims',
      'historical_provider_original_versions_independently_verified':False,'production_changed':False}
    save(p/'calculation_plan.json',plan)
    methods=[]
    def collect(name,calc,main='median',family='',status='conditional'):
        for c in calc:d['mv_'+name+'_'+c]=calc[c].to_numpy()
        available=calc[main].notna()
        ready=calc.input_ready.fillna(False) if 'input_ready' in calc else available
        positive=available&(calc[main]>0) if main!='growth' else None
        methods.append({'method':name,'family':family,'status':status,'input_rows':int(ready.sum()),
                        'calculated_rows':int(available.sum()),'calculated_pct':100*available.mean(),
                        'positive_value_rows':int(positive.sum()) if positive is not None else None,
                        'positive_value_pct':100*positive.mean() if positive is not None else None,
                        'codes':int(d.loc[available,'Code'].nunique()),
                        'reviewed_debt_rows':int((available&d.v_debt_reviewed.fillna(False)).sum()) if name not in ['per','pbr','asset_stress'] else None,
                        'capex_note_proxy_rows':int((available&d.v_note_used.fillna(False)).sum()) if name.endswith('flex') else None})
    for name in ['per','pbr','ev_ebitda']:
        available=d['mv_'+name+'_median'].notna()
        q=pd.DataFrame({s:d['mv_'+name+'_'+s] for s in M.STATS})
        q['input_ready']=(d.EBITDA.gt(0) if name=='ev_ebitda' else d[name.upper()].gt(0))&d.S33.notna()&d.S33.ne('9999')
        collect(name,q,family='relative',status='peer multiple' if name!='ev_ebitda' else 'OP+depreciation proxy')
    for kind in ['cash','flex']:
        print('DCF scenario grid:',kind,flush=True)
        collect('dcf_'+kind,M.value_grid(d,kind),family='DCF')
        print('Reverse DCF roots:',kind,flush=True)
        collect('reverse_'+kind,M.reverse(d,kind),main='growth',family='reverse DCF')
        print('Assumed-prior Monte Carlo:',kind,flush=True)
        collect('mc_'+kind,M.monte_carlo(d,kind,args.draws),family='stochastic DCF',status='uncalibrated assumed priors')
    collect('asset_stress',M.asset_stress(d),family='NAV',status='book haircut proxy; NOT market-adjusted NAV')
    methods.append({'method':'adjusted_nav_market','family':'NAV','status':'missing fair values/off-balance completeness',
       'input_rows':0,'calculated_rows':0,'calculated_pct':0.,'positive_value_rows':0,'positive_value_pct':0.,'codes':0,'reviewed_debt_rows':0})
    pd.DataFrame(methods).to_csv(p/'coverage.csv',index=False)
    quality=[]
    for label,mask in [('no_market_cap',d.MktCap.isna()),('no_sector',d.S33.isna()),
        ('no_fresh_annual',~d.v_fresh.fillna(False)),('unknown_period',~d.v_period_ok.fillna(False)),
        ('not_JP_basis',~d.v_jp_basis.fillna(False)),('financial_or_other',d.S33.isin(M.FINANCIAL)),
        ('no_cash_capex',d.v_cash_capex.isna()),('no_flex_capex',d.v_flex_capex.isna()),
        ('no_DA',d.depreciation.isna()),('no_debt',d.v_debt.isna()),('unreviewed_debt_subtotal',d.v_debt.notna()&~d.v_debt_reviewed.fillna(False)),
        ('no_NCI',d.v_nci.isna()),('equity_scope_risk',d.v_scope_risk.fillna(False)),
        ('balance_not_reconciled',~d.v_balance_ok.fillna(False)),('negative_or_missing_operating_profit',~d.operating_income.gt(0)),
        ('missing_WC_components',d[['trade_receivables','inventories','trade_payables']].isna().any(axis=1)),
        ('capex_definition_conflict',d.v_capex_conflict.fillna(False))]:
        quality.append({'check':label,'rows':int(mask.sum()),'pct':100*mask.mean()})
    pd.DataFrame(quality).to_csv(p/'input_quality.csv',index=False)
    years=[]
    for method in methods:
        name=method['method'];col='mv_'+name+('_growth' if name.startswith('reverse') else '_median')
        if col not in d:continue
        for year,g in d.groupby(d.Date.dt.year):years.append({'method':name,'year':int(year),'total':len(g),'calculated':int(g[col].notna().sum()),'pct':100*g[col].notna().mean()})
    pd.DataFrame(years).to_csv(p/'coverage_by_year.csv',index=False)
    pd.DataFrame(year_checks).to_csv(p/'peer_universe_checks.csv',index=False)
    valuecols=[c for c in d if c.startswith('mv_') and c.endswith((*M.STATS,'_growth','_p05','_p95'))]
    d[valuecols].describe(percentiles=[.05,.5,.95]).T.reset_index(names='feature').to_csv(p/'feature_summary.csv',index=False)
    d.to_parquet(p/'dataset.parquet',index=False)
    d[['Code','Date','close_raw']+valuecols].to_csv(p/'valuation_candidates.csv.gz',index=False,compression='gzip')
    files=[args.dataset,args.annual_panel,args.corrections,Path(__file__),HERE/'valuation_methods.py',HERE/'capital_efficiency.py']+master_files+val_files
    save(p/'inputs.json',{'files':{str(f):sha(f) for f in files},'rows':len(d),'codes':d.Code.nunique(),'numpy':np.__version__,'pandas':pd.__version__})
    save(p/'completed.json',{'methods':len(methods),'rows':len(d),'production_changed':False,'backtest_or_tuning_done_for_new_methods':False})
    print(pd.DataFrame(methods)[['method','input_rows','calculated_rows','calculated_pct']].to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','annual-panel','corrections','data-dir','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--draws',type=int,default=1024)
    run(ap.parse_args())


if __name__=='__main__':main()
