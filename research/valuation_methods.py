"""Conditional valuation candidates, never production features.

Coverage is arithmetic availability under declared assumptions, not audit or
forecast accuracy. Missing debt components are NOT certified zero: an observed
subtotal is explicitly flagged and separately counted from reviewed balances.
"""
from __future__ import annotations
import hashlib
import warnings
import numpy as np
import pandas as pd
import capital_efficiency as CE

STATS=('min','max','mean','median','std')
FINANCIAL={'7050','7100','7150','7200','9999'}
BASE_FIELDS=['revenue','operating_income','depreciation','inventories','trade_receivables','trade_payables',
             'capex','purchase_of_ppe','purchase_of_intangible_assets','purchase_of_ppe_and_intangible_assets',
             'total_assets','total_liabilities','ppe','intangible_assets','period_start','period_end','basis',
             'accounting_standard','filing_avail_date','fiscal_year','doc_id']


def annual_states(panel, corrections=None):
    p=panel.copy()
    for col in BASE_FIELDS:
        if col not in ['period_start','period_end','basis','accounting_standard','filing_avail_date','doc_id']:
            p[col]=pd.to_numeric(p[col],errors='coerce').astype(float)
    norm=CE._normalize(p)
    if corrections is not None:
        for _,c in corrections.iterrows():
            known=pd.Timestamp(c.source_submit_date).normalize()+pd.Timedelta(days=1)
            mask=p.Code.eq(str(c.Code))&p.fiscal_year.eq(c.fiscal_year)&p.avail_date.ge(known)
            for key in ['debt','equity','cash']:
                name='normalized_'+key
                if name in c and pd.notna(c[name]):
                    norm.loc[mask,key]=float(c[name])
                    if key in ['debt','equity']:norm.loc[mask,key+'_reviewed']=True
                    if key=='debt':norm.loc[mask,'partial']=False
                    if key=='equity':norm.loc[mask,'scope_risk']=False
    for key in norm:p['v_'+key]=norm[key].to_numpy()
    # A missing NCI is zero only for standalone statements by definition.
    p['v_nci']=pd.to_numeric(p.non_controlling_interests,errors='coerce')
    p.loc[p.basis.eq('standalone'),'v_nci']=0.
    combined=pd.to_numeric(p.purchase_of_ppe_and_intangible_assets,errors='coerce')
    parts=p.purchase_of_ppe+p.purchase_of_intangible_assets
    have_parts=p.purchase_of_ppe.le(0)&p.purchase_of_intangible_assets.le(0)
    have_combined=combined.le(0)
    conflict=have_parts&have_combined&~np.isclose(combined,parts,rtol=.005,atol=1)
    p['v_cash_capex']=(-combined).where(have_combined,(-parts).where(have_parts)).where(~conflict)
    p['v_capex_conflict']=conflict
    p['v_flex_capex']=p.v_cash_capex.where(p.v_cash_capex.notna(),p.capex.where(p.capex.ge(0)))
    p['v_note_used']=p.v_cash_capex.isna()&p.capex.ge(0)
    p['v_nwc']=p.trade_receivables+p.inventories-p.trade_payables
    p['v_period_ok']=((p.period_end-p.period_start).dt.days+1).isin(CE.NORMAL_DAYS)
    # IFRS lease reinvestment/DA reconciliation is not silently assumed.
    p['v_jp_basis']=p.accounting_standard.eq('JP')&p.basis.isin(['consolidated','standalone'])
    p['v_balance_ok']=np.isclose(p.total_assets-p.total_liabilities,p.v_equity,rtol=.0001,atol=1)
    for col in [c for c in p if c.startswith('v_') and p[c].dtype==bool]:
        p[col]=p[col].astype('boolean')
    cols=['Code','avail_date']+BASE_FIELDS+[x for x in p if x.startswith('v_')]
    p=p[p.avail_date.notna()][cols].sort_values(['avail_date','Code','fiscal_year'],kind='mergesort')
    p=p[p.fiscal_year.ge(p.groupby('Code').fiscal_year.cummax())]
    return p.drop_duplicates(['Code','avail_date'],keep='last')


def attach(frame, states, stale_days=400):
    left=frame.copy().assign(_i=np.arange(len(frame)))
    left['Code']=left.Code.astype(str)
    states=states.copy()
    states['Code']=states.Code.astype(str)
    out=pd.merge_asof(left.sort_values('Date'),states.sort_values('avail_date'),
                      left_on='Date',right_on='avail_date',by='Code',direction='backward')
    out['v_annual_age']=(out.Date-out.filing_avail_date).dt.days
    out['v_fresh']=out.v_annual_age.between(0,stale_days)
    return out.sort_values('_i').drop(columns='_i').reset_index(drop=True)


def ready(d, capex='cash'):
    nonfin=d.S33.notna()&~d.S33.isin(FINANCIAL)
    op=d.operating_income;rev=d.revenue
    return (d.v_fresh.fillna(False)&d.v_period_ok.fillna(False)&d.v_jp_basis.fillna(False)&nonfin&
            d.v_debt.ge(0)&d.v_debt.le(d.total_liabilities+1)&d.v_cash.ge(0)&d.v_nci.ge(0)&~d.v_partial.fillna(True)&
            ~d.v_scope_risk.fillna(True)&d.v_equity.gt(0)&d.v_balance_ok.fillna(False)&
            rev.gt(0)&op.gt(0)&op.le(rev)&d.depreciation.ge(0)&d['v_'+capex+'_capex'].ge(0)&
            d.trade_receivables.ge(0)&d.inventories.ge(0)&d.trade_payables.ge(0)&
            d.MktCap.gt(0)&d.close_raw.gt(0)&d.tv_beta.notna()&d.tv_rf.notna())


def wacc(mcap,debt,nci,rf,beta,premium=.05,spread=.015,tax=.30):
    # Book NCI and book debt approximate market values; NCI shares parent ke.
    equity=mcap+nci
    return ((rf+beta*premium)*equity+(rf+spread)*(1-tax)*debt)/(equity+debt)


def dcf_value(sales,op,da,capex,nwc,mcap,debt,cash,nci,rf,beta,
              growth=0.,premium=.05,spread=.015,terminal_growth=.01,tax=.30,margin=None):
    """Five-year driver forecast; stable ROIC=WACC links growth to reinvestment.

    DA/CapEx/NWC ratios remain constant for five explicit years. Terminal
    reinvestment = g/WACC * NOPAT; this is a different steady-state assumption.
    Return SIGNED parent equity, EV, terminal PV / absolute EV, and WACC.
    """
    w=wacc(mcap,debt,nci,rf,beta,premium,spread,tax)
    margin=op/sales if margin is None else margin
    previous=np.asarray(sales,float)
    ev=previous*0.
    for t in range(1,6):
        current=previous*(1+growth)
        fcff=current*(margin*(1-tax)+da/sales-capex/sales)-(current-previous)*nwc/sales
        ev=ev+fcff/(1+w)**t
        previous=current
    terminal_nopat=previous*(1+terminal_growth)*margin*(1-tax)
    tv=terminal_nopat*(1-terminal_growth/w)/(w-terminal_growth)
    tvpv=tv/(1+w)**5
    ev=ev+tvpv
    return ev-debt+cash-nci,ev,tvpv/np.maximum(np.abs(ev),1.),w


def stats(values, valid=None, min_fraction=.8):
    a=np.asarray(values,float)
    mask=np.isfinite(a) if valid is None else np.asarray(valid,bool)&np.isfinite(a)
    fraction=mask.mean(axis=1)
    a=np.where(mask,a,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        out=np.column_stack([np.nanmin(a,axis=1),np.nanmax(a,axis=1),np.nanmean(a,axis=1),
                             np.nanmedian(a,axis=1),np.nanstd(a,axis=1,ddof=0)])
    out[fraction<min_fraction]=np.nan
    return out,fraction


def value_grid(d, capex='cash'):
    ok=ready(d,capex)
    z=d.loc[ok]
    values=[];valid=[];terminal=[];neg=[]
    a=[z[c].to_numpy(float) for c in ['revenue','operating_income','depreciation','v_'+capex+'_capex',
        'v_nwc','MktCap','v_debt','v_cash','v_nci','tv_rf','tv_beta']]
    # MktCap is converted to yen in the runner before this module.
    with np.errstate(all='ignore'):
        for growth in [-.05,0.,.05,.10]:
            for premium in [.03,.05,.07]:
                for spread in [.005,.015,.03]:
                    for gterm in [0.,.01,.02]:
                        for tax in [.25,.30,.35]:
                            e,ev,tvp,w=dcf_value(*a,growth,premium,spread,gterm,tax)
                            admissible=(w-gterm>=.01)&(w>0)&np.isfinite(e)
                            values.append(np.maximum(e,0)/a[5]*z.close_raw.to_numpy())
                            valid.append(admissible);terminal.append(tvp);neg.append(e<0)
    out=pd.DataFrame(index=d.index)
    matrix=np.column_stack(values)
    summary,fraction=stats(matrix,np.column_stack(valid))
    for j,s in enumerate(STATS):out.loc[ok,s]=summary[:,j]
    out.loc[ok,'valid_fraction']=fraction
    vm=np.column_stack(valid)
    count=vm.sum(axis=1)
    out.loc[ok,'negative_equity_fraction']=np.divide((np.column_stack(neg)&vm).sum(axis=1),count,
                                                   out=np.full(len(count),np.nan),where=count>0)
    tvstats,_=stats(np.column_stack(terminal),np.column_stack(valid))
    out.loc[ok,'terminal_pv_ratio']=tvstats[:,3]
    out['input_ready']=ok
    return out


def reverse_coefficients(sales,op,da,capex,nwc,w,tax=.30,gterm=.01):
    """Ascending powers of x=1+annual sales growth; exact degree-five DCF."""
    coeff=np.zeros(6)
    a=op*(1-tax)+da-capex-nwc
    for t in range(1,6):
        coeff[t]+=a/(1+w)**t
        coeff[t-1]+=nwc/(1+w)**t
    coeff[5]+=op*(1-tax)*(1+gterm)/w/(1+w)**5
    return coeff


def real_growth_roots(coeff, enterprise_value, low=-.5, high=.5):
    c=np.asarray(coeff,float).copy();c[0]-=enterprise_value
    scale=np.max(np.abs(c))
    if not np.isfinite(scale) or scale==0:return []
    roots=np.polynomial.polynomial.polyroots(c/scale)
    out=[]
    for root in roots:
        if abs(root.imag)>1e-6:continue
        g=float(root.real-1)
        if low-1e-10<=g<=high+1e-10 and abs(np.polynomial.polynomial.polyval(root.real,c))/max(abs(enterprise_value),1)<1e-7:
            if not any(abs(g-x)<1e-6 for x in out):out.append(g)
    return sorted(out)


def reverse(d,capex='cash'):
    ok=ready(d,capex)
    out=pd.DataFrame(index=d.index,columns=['growth','roots','relative_residual','growth_low_premium','growth_high_premium'],dtype=float)
    out['input_ready']=ok
    for i,r in d.loc[ok].iterrows():
        target=r.MktCap+r.v_debt+r.v_nci-r.v_cash
        if target<=0:continue
        for premium,col in [(.05,'growth'),(.03,'growth_low_premium'),(.07,'growth_high_premium')]:
            w=wacc(r.MktCap,r.v_debt,r.v_nci,r.tv_rf,r.tv_beta,premium)
            if w-.01<.01:continue
            coeff=reverse_coefficients(r.revenue,r.operating_income,r.depreciation,r['v_'+capex+'_capex'],r.v_nwc,w)
            roots=real_growth_roots(coeff,target)
            if col=='growth':out.loc[i,'roots']=len(roots)
            if len(roots)==1:
                out.loc[i,col]=roots[0]
                if col=='growth':out.loc[i,'relative_residual']=abs(np.polynomial.polynomial.polyval(1+roots[0],coeff)-target)/target
    return out


def monte_carlo(d,capex='cash',draws=1024,seed=20261104):
    """ASSUMED, uncalibrated priors. Zero-equity paths are retained.

    All uncertain parameters are constant for each five-year path. Independence
    is a declared modeling assumption, except terminal g is conditional on WACC.
    Per-key seeds make future appended rows/order irrelevant to earlier values.
    """
    ok=ready(d,capex)
    out=pd.DataFrame(index=d.index,columns=[*STATS,'p05','p95','valid_fraction','negative_equity_fraction'],dtype=float)
    out['input_ready']=ok
    for i,r in d.loc[ok].iterrows():
        key=f'{r.Code}/{pd.Timestamp(r.Date).date()}/{seed}'
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big'))
        growth=rng.triangular(-.05,0.,.10,draws)
        premium=rng.triangular(.03,.05,.07,draws)
        spread=rng.triangular(.005,.015,.03,draws)
        tax=rng.uniform(.25,.35,draws)
        m=r.operating_income/r.revenue
        margin=rng.triangular(.8*m,m,min(1.,1.2*m),draws)
        w=wacc(r.MktCap,r.v_debt,r.v_nci,r.tv_rf,r.tv_beta,premium,spread,tax)
        gterm=rng.uniform(0,1,draws)*np.maximum(0,np.minimum(.02,w-.01))
        with np.errstate(all='ignore'):
            e,_,_,_=dcf_value(r.revenue,r.operating_income,r.depreciation,r['v_'+capex+'_capex'],r.v_nwc,
                r.MktCap,r.v_debt,r.v_cash,r.v_nci,r.tv_rf,r.tv_beta,growth,premium,spread,gterm,tax,margin)
        valid=(w>.01)&np.isfinite(e)
        out.loc[i,'valid_fraction']=valid.mean()
        if valid.mean()<.8:continue
        v=np.maximum(e[valid],0)/r.MktCap*r.close_raw
        out.loc[i,list(STATS)]=[v.min(),v.max(),v.mean(),np.median(v),v.std(ddof=0)]
        out.loc[i,['p05','p95']]=np.quantile(v,[.05,.95])
        out.loc[i,'negative_equity_fraction']=(e[valid]<0).mean()
    return out


def asset_stress(d):
    # This is NOT market-value adjusted NAV or a liquidation-value floor.
    ok=(d.v_fresh.fillna(False)&d.v_period_ok.fillna(False)&d.v_balance_ok.fillna(False)&
        d.v_nci.ge(0)&d.total_assets.ge(0)&d.total_liabilities.ge(0)&d.ppe.ge(0)&
        d.intangible_assets.ge(0)&d.inventories.ge(0)&d.MktCap.gt(0)&d.close_raw.gt(0)&
        d.S33.notna()&~d.S33.isin(FINANCIAL)&~d.v_scope_risk.fillna(True))
    z=d.loc[ok];values=[]
    for hp in [0,.2,.4]:
        for hi in [0,.2,.4]:
            for hn in [0,.5,1.]:
                e=z.total_assets-z.total_liabilities-z.v_nci-hp*z.ppe-hi*z.inventories-hn*z.intangible_assets
                values.append(np.maximum(e,0)/z.MktCap*z.close_raw)
    out=pd.DataFrame(index=d.index)
    summary,_=stats(np.column_stack(values))
    for j,s in enumerate(STATS):out.loc[ok,s]=summary[:,j]
    out['input_ready']=ok
    return out


def peer_prices(frame,targets,metric,min_peers=10,enterprise=False):
    """Full observed market, same date/sector, self excluded; no sample-only peers."""
    groups={k:g for k,g in frame[frame[metric].gt(0)&np.isfinite(frame[metric])].groupby(['Date','S33'],observed=True)}
    out=pd.DataFrame(index=targets.index,columns=[*STATS,'p10','p90','peers'],dtype=float)
    for i,r in targets.iterrows():
        if pd.isna(r.S33) or r.S33=='9999':continue
        g=groups.get((r.Date,r.S33))
        if g is None:continue
        if enterprise:
            if not np.isfinite(r['EBITDA']) or r['EBITDA']<=0 or r.MktCap<=0:continue
        elif not np.isfinite(r[metric]) or r[metric]<=0:continue
        peers=g[g.Code.ne(r.Code)][metric].to_numpy(float)
        out.loc[i,'peers']=len(peers)
        if len(peers)<min_peers:continue
        # PER/PBR: P * peer multiple / own multiple. EV case transformed outside.
        values=(np.maximum(peers*r.EBITDA-r.v_debt+r.v_cash-r.v_nci,0)/r.MktCap*r.close_raw
                if enterprise else peers/r[metric]*r.close_raw)
        out.loc[i,list(STATS)]=[values.min(),values.max(),values.mean(),np.median(values),values.std(ddof=0)]
        out.loc[i,['p10','p90']]=np.quantile(values,[.1,.9])
    return out
