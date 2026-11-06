"""Finite-horizon maximum-relative stopping, with observed next-open execution.

The mathematical boundary assumes constant GBM coefficients and immediate sale.
Applying it after each close and executing at the next open is a separately
evaluated approximation, not an optimality theorem for the trading implementation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.sparse import csr_matrix


def maximum_cdf(level, delta, sigma, horizon):
    """CDF of sup_{0<=s<=horizon}(delta*s+sigma*W_s), level in log units."""
    x,d,s,t=np.broadcast_arrays(np.asarray(level,float),delta,sigma,horizon)
    if np.any(s<=0) or np.any(t<=0): raise ValueError('Positive volatility and horizon required')
    scale=s*np.sqrt(t)
    a=(x-d*t)/scale; b=(-x-d*t)/scale
    out=ndtr(a)-np.exp(2*d*x/(s*s)+log_ndtr(b))
    return np.where(x<=0,0,np.clip(out,0,1))


def stopping_payoff(horizon, normalized_drawdown, sigma, alpha):
    """E[M_T/P_now] given past maximum, alpha=price drift/sigma^2.

Analytic conditional gain G from du Toit/Peskir (2009), including its mu=0 limit.
"""
    y,s,a,t=np.broadcast_arrays(np.asarray(normalized_drawdown,float),sigma,alpha,horizon)
    if np.any(s<=0) or np.any(t<0) or np.any(y<0): raise ValueError('Invalid GBM state')
    x=s*y; safe_t=np.maximum(t,np.finfo(float).tiny)
    root=np.sqrt(safe_t); mu=a*s*s; delta=mu-.5*s*s
    z1=(-x+(mu+.5*s*s)*safe_t)/(s*root)
    z2=(x-delta*safe_t)/(s*root)
    z3=(-x-delta*safe_t)/(s*root)
    near=np.abs(a)<1e-7
    denom=np.where(near,1,a)
    out=(1+1/(2*denom))*np.exp(mu*safe_t+log_ndtr(z1))+np.exp(x+log_ndtr(z2))-np.exp(2*a*x+log_ndtr(z3))/(2*denom)
    z=(x+.5*s*s*safe_t)/(s*root)
    z0=(-x+.5*s*s*safe_t)/(s*root)
    zero=s*root*np.exp(-z0*z0/2)/np.sqrt(2*np.pi)+np.exp(x)*ndtr(z)+(1-x+.5*s*s*safe_t)*ndtr(z0)
    out=np.where(near,zero,out)
    return np.where(t==0,np.exp(x),np.maximum(out,np.exp(x)))


def maximum_quantile(q,delta,sigma,horizon):
    d,s,t=np.broadcast_arrays(np.asarray(delta,float),sigma,horizon)
    if not 0<q<1:raise ValueError('Interior quantile required')
    lo=np.zeros_like(d);hi=np.maximum(d*t,0)+12*s*np.sqrt(t)+.1
    for _ in range(48):
        mid=(lo+hi)/2
        below=maximum_cdf(mid,d,s,t)<q
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    return np.exp((lo+hi)/2)


def reflected_cdf(destination, start, lam):
    """One-session transition CDF for normalized drawdown, drift=-lam.

This exact CDF integrates over the within-session maximum as well as the close.
It is not a close-only approximation to a running high.
"""
    z,x=np.broadcast_arrays(np.asarray(destination,float),start)
    out=ndtr(z-x+lam)-np.exp(-2*lam*z+log_ndtr(-z-x+lam))
    return np.where(z<=0,0,np.clip(out,0,1))


def transition_matrix(grid,lam):
    g=np.asarray(grid,float)
    edges=(g[1:]+g[:-1])/2
    cumulative=reflected_cdf(edges[None,:],g[:,None],lam)
    probabilities=np.diff(np.column_stack([np.zeros(len(g)),cumulative,np.ones(len(g))]),axis=1)
    probabilities=np.maximum(probabilities,0)
    probabilities[probabilities<1e-14]=0
    probabilities/=probabilities.sum(axis=1)[:,None]
    return csr_matrix(probabilities)


def reflected_exponential_moment(destination,start,lam,sigma):
    """E[exp(sigma*Y_next) 1(Y_next<=destination)] for reflected Brownian Y.

    Analytic integral of exp(sigma*y) times the derivative of reflected_cdf.
    This also supplies the exact positive tail used at a finite grid edge.
    """
    z,x=np.broadcast_arrays(np.asarray(destination,float),start)
    finite=np.isfinite(z);b=np.where(finite,z,100+np.abs(x)+abs(lam))
    m=lam-x;k=sigma-2*lam
    first=np.exp(sigma*(x-lam)+sigma*sigma/2)*(ndtr(b+m-sigma)-ndtr(m-sigma))
    if abs(k)<1e-9:
        second=ndtr(b-m)-ndtr(-m)
        phi=lambda t:np.exp(-t*t/2)/np.sqrt(2*np.pi)
        integral=m*ndtr(m)+phi(m)-(m-b)*ndtr(m-b)-phi(m-b)
        out=first+second+2*lam*integral
        total=np.exp(sigma*(x-lam)+sigma*sigma/2)*ndtr(-m+sigma)+ndtr(m)+2*lam*(m*ndtr(m)+phi(m))
    else:
        coefficient=np.exp(k*m+k*k/2)
        second=coefficient*(ndtr(b-m-k)-ndtr(-m-k))
        out=first+sigma/k*second+2*lam/k*(np.exp(k*b+log_ndtr(m-b))-ndtr(m))
        total=np.exp(sigma*(x-lam)+sigma*sigma/2)*ndtr(-m+sigma)+sigma/k*coefficient*ndtr(m+k)-2*lam/k*ndtr(m)
    return np.where(finite,np.maximum(out,0),total)


def value_transition_matrix(grid,lam,sigma):
    """Cell-integrated operator for V, interpolating V(y)/exp(sigma*y).

    The exponential tail and its expectation are exact. This prevents a small
    grid bias from overturning a near-zero hold-vs-sell difference at alpha~1.
    """
    g=np.asarray(grid,float);edges=(g[1:]+g[:-1])/2
    cumulative=reflected_exponential_moment(edges[None,:],g[:,None],lam,sigma)
    total=reflected_exponential_moment(np.inf,g,lam,sigma)
    moments=np.maximum(np.diff(np.column_stack([np.zeros(len(g)),cumulative,total]),axis=1),0)
    moments*=total[:,None]/moments.sum(axis=1)[:,None]
    weights=moments/np.exp(sigma*g)[None,:]
    weights[weights<1e-14]=0
    return csr_matrix(weights)


def solve_boundary(sigma,alpha,hold=19,points=801,ymax=40):
    """Backward DP, returning log-drawdown thresholds for each remaining horizon.

The upper stopping-set shape is imposed only when locating a boundary; the
value function itself always uses the pointwise minimum of stop/continue.
"""
    grid=np.linspace(0,ymax,points)
    lam=(alpha-.5)*sigma
    transition=value_transition_matrix(grid,lam,sigma)
    value=np.exp(sigma*grid)
    boundary=np.full(hold+1,np.inf);boundary[0]=0
    values_at_peak=[1.0];violations=0
    for r in range(1,hold+1):
        stop=stopping_payoff(r,grid,sigma,alpha)
        continuation=transition@value
        difference=stop-continuation
        raw=difference<=1e-12
        upper=np.logical_and.accumulate(raw[::-1])[::-1]
        violations+=int(np.sum(raw & ~upper))
        hits=np.flatnonzero(upper)
        if len(hits):
            j=int(hits[0])
            root=0.0 if j==0 else grid[j-1]+(grid[j]-grid[j-1])*difference[j-1]/(difference[j-1]-difference[j])
            boundary[r]=sigma*max(root,0)
        value=np.minimum(stop,continuation)
        values_at_peak.append(float(value[0]))
    return boundary,dict(non_upper_stop_nodes=violations,value_at_peak=values_at_peak,
                         min_transition_row_sum=float(np.min(transition.sum(axis=1))),
                         max_transition_row_sum=float(np.max(transition.sum(axis=1))))


def posterior_drift(prior,log_gain,elapsed,strength):
    prior,gain,k=np.broadcast_arrays(np.asarray(prior,float),log_gain,elapsed)
    if strength==np.inf:return prior.copy()
    if strength<=0 or np.any(k<0):raise ValueError('Invalid prior strength or age')
    return (strength*prior+gain)/(strength+k)


def execute_exit(opening,high,close,decisions=None,target=None,gap_fill='open'):
    """All prices are returns relative to the original purchase open.

decisions[k-1] is the close-of-session-k signal. Its earliest fill is open k+1.
"""
    o,h,c=(np.asarray(v,float) for v in (opening,high,close))
    if len(o)!=len(h) or len(o)!=len(c):raise ValueError('Misaligned OHLC')
    signal=np.zeros(len(c)-1,bool) if decisions is None else np.asarray(decisions,bool)
    if len(signal)!=len(c)-1:raise ValueError('Need one decision per pre-deadline close')
    for day in range(len(c)):
        if day and signal[day-1]:return dict(gross=float(o[day]),exit_day=day+1,reason='next_open')
        if target is not None and h[day]>=target-1e-12:
            price=max(o[day],target) if gap_fill=='open' else target
            return dict(gross=float(price),exit_day=day+1,reason='target')
    return dict(gross=float(c[-1]),exit_day=len(c),reason='deadline')
