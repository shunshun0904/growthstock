"""Research-only daily +10%/+15% arrival from a fixed next-open purchase.

Features only observe the prefix. Future high/close/exit labels are separate.
The last column is the close of session19 with session20 still ahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGETS=(10,15)
PREFIX_FEATURES=('elapsed','remaining','spot_return','known_peak','known_low',
                 'drawdown_from_peak','sessions_since_peak','prefix_vol','last_return')
TECH_FEATURES=('ret1','ret3','ret5','ret10','ma5_gap','ma20_gap','atr14',
               'vol20','volume_ratio20','volume_change','intraday_range','body','gap')


def landmark_arrays(o,h,l,c,stop=.05):
    o,h,l,c=(np.asarray(a,float) for a in (o,h,l,c))
    if not(o.shape==h.shape==l.shape==c.shape) or o.ndim!=2:
        raise ValueError('OHLC must be aligned trade-by-session matrices')
    n,hold=c.shape
    age=np.broadcast_to(np.arange(hold),(n,hold)).copy()
    spot=np.column_stack([np.zeros(n),c[:,:-1]])
    peak=np.column_stack([np.zeros(n),np.maximum.accumulate(h,axis=1)[:,:-1]])
    low=np.column_stack([np.zeros(n),np.minimum.accumulate(l,axis=1)[:,:-1]])
    prev=np.column_stack([np.zeros(n),spot[:,:-1]])
    last=(1+spot)/(1+prev)-1
    # First return is entry-open -> first close; later returns are close-close.
    rr=(1+c)/(1+np.column_stack([np.zeros(n),c[:,:-1]]))-1
    sums=np.column_stack([np.zeros(n),np.cumsum(rr,axis=1)[:,:-1]])
    sq=np.column_stack([np.zeros(n),np.cumsum(rr*rr,axis=1)[:,:-1]])
    denom=np.maximum(age,1)
    variance=np.maximum(sq/denom-(sums/denom)**2,0)
    std=np.sqrt(variance)
    std[age<2]=np.nan
    since=np.zeros_like(age)
    latest=np.zeros(n,int)
    for k in range(1,hold):
        latest=np.where(h[:,k-1]>=peak[:,k-1]-1e-12,k,latest)
        since[:,k]=k-latest
    future_h=np.maximum.accumulate(h[:,::-1],axis=1)[:,::-1]
    future_c=np.maximum.accumulate(c[:,::-1],axis=1)[:,::-1]
    s={'elapsed':age,'remaining':hold-age,'spot_return':spot,'known_peak':peak,'known_low':low,
       'drawdown_from_peak':(1+spot)/(1+peak)-1,'sessions_since_peak':since,'prefix_vol':std,
       'last_return':last,'future_max_from_entry':future_h,
       'future_max_from_spot':(1+future_h)/(1+spot)-1}
    days=np.broadcast_to(np.arange(hold),(n,hold))
    first_stop=np.minimum.accumulate(np.where(l<=-stop+1e-12,days,hold)[:,::-1],axis=1)[:,::-1]
    for target in TARGETS:
        a=target/100
        arrived=peak>=a-1e-12
        first_tp=np.minimum.accumulate(np.where(h>=a-1e-12,days,hold)[:,::-1],axis=1)[:,::-1]
        tp=first_tp<hold
        sl=first_stop<hold
        idx=np.minimum(first_tp,hold-1)
        open_tp=np.take_along_axis(o,idx,axis=1)>=a-1e-12
        target_first=tp&((first_tp<first_stop)|((first_tp==first_stop)&open_tp))
        outcome=np.where(target_first,1,np.where(sl,2,0))
        held=np.where(tp,a,c[:,-1,None])
        # Match the target-limit benchmark's conservative profit-gap fill.
        immediate=np.minimum(o,a)
        s.update({f'arrived{target}':arrived,f'risk{target}':~arrived,
                  f'y{target}':future_h>=a-1e-12,f'y_close{target}':future_c>=a-1e-12,
                  f'comp{target}':outcome,
                  f'comp_risk{target}':~arrived&(low>-stop+1e-12),
                  f'gain{target}':held-immediate})
    return s


def matured_mask(rows,cutoff):
    """Withhold an entire trade until its original twenty-session label matures."""
    return pd.to_datetime(rows.mature_date).lt(pd.Timestamp(cutoff))


def coherent_probabilities(p10,p15,arrived10=None,arrived15=None):
    a=np.clip(np.asarray(p10,float),0,1).copy()
    b=np.clip(np.asarray(p15,float),0,1).copy()
    if arrived10 is not None:a=np.where(arrived10,1,a)
    if arrived15 is not None:b=np.where(arrived15,1,b)
    crossing=b>a
    mean=(a+b)/2
    a=np.where(crossing,mean,a);b=np.where(crossing,mean,b)
    return a,b


def daily_technicals(frame,volume=None):
    """One company, date order; relative values cancel future uniform split scales."""
    c=frame.AdjC.astype(float);o=frame.AdjO;h=frame.AdjH;l=frame.AdjL
    prev=c.shift(1)
    ret=c/prev-1
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    v=pd.Series(np.nan,index=frame.index) if volume is None else pd.Series(volume,index=frame.index,dtype=float)
    z=pd.DataFrame(index=frame.index)
    for lag in (1,3,5,10):z[f'ret{lag}']=c/c.shift(lag)-1
    for win in (5,20):z[f'ma{win}_gap']=c/c.rolling(win,min_periods=win).mean()-1
    z['atr14']=tr.rolling(14,min_periods=14).mean()/c
    z['vol20']=ret.rolling(20,min_periods=20).std()
    z['volume_ratio20']=v/v.rolling(20,min_periods=20).mean()
    z['volume_change']=v/v.shift(1)-1
    z['intraday_range']=(h-l)/prev
    z['body']=c/o-1
    z['gap']=o/prev-1
    return z[list(TECH_FEATURES)].replace([np.inf,-np.inf],np.nan)


def trade_mean(values,rows):
    return float(pd.DataFrame({'trade':rows.trade.to_numpy(),'v':np.asarray(values,float)}).groupby('trade').v.mean().mean())


def simulate_daily_exit(o,h,l,c,gains,*,target,cutoff=0,profit_gap='limit'):
    """A close-k decision sells at the open of k+1, never at known close-k."""
    hold=len(c)
    for day in range(hold):
        # An exit decided at yesterday's close, before any new intraday data.
        if day>=1 and np.isfinite(gains[day]) and gains[day]<cutoff:
            value=min(o[day],target) if profit_gap=='limit' else o[day]
            return {'gross':float(value),'exit_day':day+1,'reason':'daily_exit'}
        if h[day]>=target-1e-12:
            value=max(o[day],target) if profit_gap=='open' else target
            return {'gross':float(value),'exit_day':day+1,'reason':'target'}
        if day==hold-1:
            return {'gross':float(c[day]),'exit_day':hold,'reason':'deadline'}
    raise AssertionError('Unreachable')
