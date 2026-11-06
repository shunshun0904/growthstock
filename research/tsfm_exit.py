"""Zero-shot forecasts of a future maximum, not maxima of marginal forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_max(high,window):
    return pd.Series(np.asarray(high,float)).rolling(int(window),min_periods=int(window)).max().to_numpy()


def context_at(high,asof_index,remaining,context,current_close):
    if remaining<1 or current_close<=0:raise ValueError('Invalid landmark')
    # Only the prefix is supplied even to the rolling transform.
    known=np.asarray(high,float)[:asof_index+1]
    mx=rolling_max(known,remaining)
    with np.errstate(divide='ignore',invalid='ignore'):
        values=np.log(mx[-context:]/current_close)
    values[~np.isfinite(values)]=np.nan
    return values.astype(np.float32)


def complete_quantiles(values,levels,tail='constant'):
    v=np.sort(np.asarray(values,float));q=np.asarray(levels,float)
    if len(v)<2 or len(v)!=len(q) or not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid quantiles')
    if (np.diff(q)<=0).any() or q[0]<0 or q[-1]>1:raise ValueError('Invalid levels')
    if tail not in ('constant','linear'):raise ValueError('Invalid tail assumption')
    lo=v[0] if tail=='constant' else max(0,v[0]-(v[1]-v[0])/(q[1]-q[0])*q[0])
    hi=v[-1] if tail=='constant' else v[-1]+(v[-1]-v[-2])/(q[-1]-q[-2])*(1-q[-1])
    if q[0]>0:q=np.r_[0,q];v=np.r_[lo,v]
    if q[-1]<1:q=np.r_[q,1];v=np.r_[v,hi]
    return v,q


def distribution_summary(values,levels,threshold,tail='constant'):
    """Piecewise-price-linear quantile function including declared endpoint tails.

    P(X>=threshold) uses the left edge of a flat quantile atom; equality is a hit.
    Mean and second moment integrate this approximate distribution analytically.
    """
    v,q=complete_quantiles(values,levels,tail)
    if threshold<=v[0]:p=1.0
    elif threshold>v[-1]:p=0.0
    else:
        i=int(np.searchsorted(v,threshold,side='left'))
        u=q[i-1]+(threshold-v[i-1])/(v[i]-v[i-1])*(q[i]-q[i-1])
        p=float(1-u)
    du=np.diff(q)
    mean=float(np.sum(du*(v[:-1]+v[1:])/2))
    second=float(np.sum(du*(v[:-1]**2+v[:-1]*v[1:]+v[1:]**2)/3))
    return p,mean,float(np.sqrt(max(second-mean*mean,0)))


def volatility_paths(remaining,sigma_gap,sigma_intraday,paths,seed):
    """Gaussian overnight jumps plus conditional intraday Brownian-bridge highs.

    Excludes the previous close from future trading highs: all future highs may
    be below today's close after an overnight drop. Zero log drift is assumed.
    """
    if min(sigma_gap,sigma_intraday)<0 or remaining<1:raise ValueError('Invalid volatility')
    rng=np.random.default_rng(seed)
    last=np.zeros(paths);maximum=np.full(paths,-np.inf)
    for _ in range(remaining):
        opening=last+sigma_gap*rng.standard_normal(paths)
        closing=opening+sigma_intraday*rng.standard_normal(paths)
        u=np.maximum(rng.random(paths),np.finfo(float).tiny)
        high=(opening+closing+np.sqrt((opening-closing)**2-2*sigma_intraday**2*np.log(u)))/2
        maximum=np.maximum(maximum,high);last=closing
    return np.exp(maximum)


def volatility_estimates(frame,window=60,minimum=32):
    c=frame.AdjC.to_numpy(float);o=frame.AdjO.to_numpy(float)
    h=frame.AdjH.to_numpy(float);l=frame.AdjL.to_numpy(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        gap=np.log(o/np.r_[np.nan,c[:-1]])
        park=np.log(h/l)**2/(4*np.log(2))
    gap=pd.Series(gap).replace([np.inf,-np.inf],np.nan)
    park=pd.Series(park).replace([np.inf,-np.inf],np.nan)
    return np.sqrt(gap.rolling(window,min_periods=minimum).var(ddof=1)).to_numpy(),np.sqrt(park.rolling(window,min_periods=minimum).mean()).to_numpy()


def policy_exit(o,h,c,probabilities,target,rule='absolute',threshold=.20,profit_gap='limit'):
    hold=len(c);p=np.asarray(probabilities,float)
    for day in range(hold):
        sell=False
        if day>=1 and np.isfinite(p[day-1]):
            if rule=='absolute':sell=p[day-1]<threshold
            elif rule=='half':sell=day>=2 and p[day-1]<.5*p[0]
            elif rule!='fixed':raise ValueError('Unknown exit rule')
        if sell:
            value=min(o[day],target) if profit_gap=='limit' else o[day]
            return {'gross':float(value),'exit_day':day+1,'reason':'probability_exit'}
        if h[day]>=target-1e-12:
            value=max(o[day],target) if profit_gap=='open' else target
            return {'gross':float(value),'exit_day':day+1,'reason':'target'}
        if day==hold-1:return {'gross':float(c[day]),'exit_day':hold,'reason':'deadline'}
    raise AssertionError('Unreachable')
