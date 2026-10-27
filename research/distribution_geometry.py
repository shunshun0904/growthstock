"""Per-stock Gaussian distribution changes, observable after the signal-day close.

Opt-in research adapter; the production feature preset and dataset are unchanged.
Prices must be split-adjusted AdjC. No raw-price fallback, interpolation, winsorizing,
or missing-session compression. Windows/lags count supplied exchange sessions.
The Gaussian fit uses W complete log returns and MLE sigma (ddof=0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

ARMS = ("M0", "M1", "M2", "M3")


@dataclass(frozen=True)
class GeometryConfig:
    window: int = 30
    lags: tuple[int, ...] = (5, 20, 60)
    reference_lag: int = 20
    ma_window: int = 10
    z_window: int = 60
    sigma_floor: float = 1e-8

    def __post_init__(self):
        for name in ("window", "ma_window", "z_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if (not self.lags or len(set(self.lags)) != len(self.lags)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.lags)):
            raise ValueError("lags must be unique positive integers")
        if self.reference_lag not in self.lags:
            raise ValueError("reference_lag must be included in lags")
        if not np.isfinite(self.sigma_floor) or self.sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive")

    @property
    def prefix(self):
        return f"geom_w{self.window}_"


def gaussian_fisher_rao_distance(mu1, sigma1, mu2, sigma2):
    """Exact geodesic distance on the *full* univariate Gaussian family.

    d = 2 sqrt(2) asinh(sqrt((mu1-mu2)^2 + 2(s1-s2)^2) / sqrt(8 s1 s2)).
    This is algebraically the supplied sqrt(2)*log((b+a)/(b-a)), evaluated
    in log space to avoid cancellation, overflow and atanh saturation.
    Broadcastable inputs; invalid/nonpositive sigma or nonfinite means -> NaN.
    No implicit floor here: rolling estimates are regularized by the caller.
    See docs/STOCK_DISTRIBUTION_GEOMETRY.md for derivation and sources.
    """
    m1, s1, m2, s2 = np.broadcast_arrays(*[np.asarray(x, dtype=float)
                                         for x in (mu1, sigma1, mu2, sigma2)])
    valid = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(s1) & np.isfinite(s2) & (s1 > 0) & (s2 > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        delta = np.abs(m1 - m2)
        scale = np.maximum(np.abs(m1), np.abs(m2))
        log_dm = np.where(np.isinf(delta), np.log(scale) + np.log(np.abs(m1/scale-m2/scale)), np.log(delta))
        log_ds = np.log(np.abs(s1 - s2))
        log_a = 0.5 * np.logaddexp(2*log_dm, np.log(2.0) + 2*log_ds)
        log_x = log_a - 1.5*np.log(2.0) - 0.5*np.log(s1) - 0.5*np.log(s2)
        asinh_x = np.logaddexp(log_x, 0.5*np.logaddexp(0.0, 2*log_x))
        result = np.where(valid, 2*np.sqrt(2.0)*asinh_x, np.nan)
    return float(result) if result.ndim == 0 else result


def session_index(sessions: Sequence) -> pd.DatetimeIndex:
    """Require an explicit, ordered exchange calendar; never infer business days."""
    if sessions is None:
        raise ValueError("An explicit exchange-session calendar is required")
    days = pd.DatetimeIndex(pd.to_datetime(sessions))
    if (not len(days) or days.hasnans or days.has_duplicates or not days.is_monotonic_increasing
            or days.tz is not None or not days.equals(days.normalize())):
        raise ValueError("sessions must be unique, sorted, timezone-naive dates at midnight")
    return days


def keyed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize keys, rejecting duplicate observations instead of choosing a vintage."""
    if not {"Code", "Date"}.issubset(frame):
        raise ValueError("Code and Date are required")
    out = frame.copy()
    if out[["Code", "Date"]].isna().any().any():
        raise ValueError("Code/Date cannot be missing")
    out["Code"] = out["Code"].astype(str)
    out["Date"] = pd.to_datetime(out["Date"])
    if out.Date.dt.tz is not None or not out.Date.equals(out.Date.dt.normalize()):
        raise ValueError("Date must be timezone-naive and normalized to midnight")
    if out.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate (Code, Date); resolve data vintages explicitly")
    return out


def feature_columns(base: Sequence[str], arm: str = "M0", config: GeometryConfig = GeometryConfig()) -> list[str]:
    """M0 unchanged; M1 signed changes; M2 distances/dynamics; M3 their union."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    out = list(base)
    if len(set(out)) != len(out) or any(c.startswith("geom_") for c in out):
        raise ValueError("base columns must be unique and contain no geometry features")
    p = config.prefix
    if arm in ("M1", "M3"):
        out += [p+f"{kind}_{lag}" for lag in config.lags for kind in ("delta_mu", "delta_log_sigma")]
    if arm in ("M2", "M3"):
        out += [p+f"fid_{lag}" for lag in config.lags]
        out += [p+"fid_velocity", p+f"fid_ma_{config.ma_window}", p+f"fid_zscore_{config.z_window}"]
    return out


def compute_features(bars: pd.DataFrame, sessions: Sequence,
                     config: GeometryConfig = GeometryConfig()) -> pd.DataFrame:
    """Return features at observed bar keys, ordered (Code, Date).

    A missing/invalid close or nonpositive AdjVo (if supplied) breaks the daily
    return both on that session and on the next one. A full W-return window is
    required. Zero/tiny fitted sigma is floored only for distances/log changes.
    FID moving mean includes t. Its z-score reference is strictly [t-Z, t-1];
    a reference standard deviation <= 1e-12 produces NaN, including flat prices.
    """
    days = session_index(sessions)
    b = keyed_frame(bars)
    if "AdjC" not in b:
        raise ValueError("Split-adjusted AdjC is required; raw C is not a substitute")
    if not b.Date.isin(days).all():
        raise ValueError("Calendar does not cover every bar date")
    cols = feature_columns([], "M3", config)
    parts = []
    for code, stock in b.groupby("Code", sort=True):
        stock = stock.sort_values("Date")
        calendar = days[(days >= stock.Date.iloc[0]) & (days <= stock.Date.iloc[-1])]
        daily = stock.set_index("Date").reindex(calendar)
        close = pd.to_numeric(daily.AdjC, errors="coerce")
        valid = np.isfinite(close) & (close > 0)
        if "AdjVo" in daily:
            volume = pd.to_numeric(daily.AdjVo, errors="coerce")
            valid &= np.isfinite(volume) & (volume > 0)
        r = np.log(close.where(valid)).diff()
        mu = r.rolling(config.window, min_periods=config.window).mean()
        sigma = r.rolling(config.window, min_periods=config.window).std(ddof=0).clip(lower=config.sigma_floor)
        out = pd.DataFrame(index=calendar)
        p = config.prefix
        for lag in config.lags:
            old_mu, old_sigma = mu.shift(lag), sigma.shift(lag)
            out[p+f"delta_mu_{lag}"] = mu - old_mu
            out[p+f"delta_log_sigma_{lag}"] = np.log(sigma) - np.log(old_sigma)
            out[p+f"fid_{lag}"] = gaussian_fisher_rao_distance(mu, sigma, old_mu, old_sigma)
        fid = out[p+f"fid_{config.reference_lag}"]
        out[p+"fid_velocity"] = fid.diff()
        out[p+f"fid_ma_{config.ma_window}"] = fid.rolling(config.ma_window, min_periods=config.ma_window).mean()
        history = fid.shift(1).rolling(config.z_window, min_periods=config.z_window)
        sd = history.std(ddof=0)
        out[p+f"fid_zscore_{config.z_window}"] = (fid-history.mean()) / sd.where(sd > 1e-12)
        out = out.loc[pd.DatetimeIndex(stock.Date), cols]
        out.insert(0, "Date", out.index)
        out.insert(0, "Code", code)
        parts.append(out.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Code", "Date"]+cols)


def attach_features(samples: pd.DataFrame, bars: pd.DataFrame | None, sessions: Sequence | None,
                    config: GeometryConfig = GeometryConfig(), *, arm: str = "M0") -> pd.DataFrame:
    """Research-only on/off adapter. M0 does not access bars or change existing data."""
    cols = feature_columns([], arm, config)
    if arm == "M0":
        return samples.copy()
    if set(cols).intersection(samples.columns):
        raise ValueError("Geometry columns already exist; refusing an ambiguous join")
    keys = keyed_frame(samples[["Code", "Date"]])
    feats = compute_features(bars, sessions, config)
    joined = keys.merge(feats[["Code", "Date"]+cols], on=["Code", "Date"], how="left", validate="one_to_one", sort=False)
    out = samples.copy()
    for col in cols:
        out[col] = joined[col].to_numpy()
    return out
