"""Versioned production label contracts. Legacy model IDs remain stable for schedules."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd

VERSION = "fixed_return_v1"
NUMERIC_TOLERANCE = 1e-12
BASE_MODELS = ("lgbm", "xgb", "cat")
# These IDs are retained for existing weekly jobs and historical prediction keys.
STRICT_MODELS = ("lgbm_k15", "xgb_k15", "cat_k15")

def for_model(algo: str = "lgbm") -> dict | None:
    if algo not in BASE_MODELS + STRICT_MODELS:
        return None
    target = .15 if algo in STRICT_MODELS else .10
    return {"version": VERSION, "kind": "fixed_return", "threshold": target,
            "end_ratio": target / 2, "horizon": 20, "end_window": 5,
            "entry": "next_open", "require_uptrend": True,
            "trend_short": 5, "trend_long": 20, "vol_norm_k": None,
            "numeric_tolerance": NUMERIC_TOLERANCE}

def signature(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]

def at_least(values, threshold):
    values = pd.to_numeric(values, errors="coerce")
    return values.ge(threshold) | pd.Series(
        np.isclose(values.to_numpy(dtype=float), threshold, rtol=0,
                   atol=NUMERIC_TOLERANCE), index=values.index)

def labels(frame: pd.DataFrame, algo: str = "lgbm") -> pd.Series:
    spec = for_model(algo)
    if spec is None:
        raise ValueError(f"Production label is undefined for {algo}")
    required = ["future_rise", "end_level", "uptrend_end", "entry_price"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing label materials: {missing}")
    known = frame[required].notna().all(axis=1)
    ok = (at_least(frame.future_rise, spec["threshold"])
          & at_least(frame.end_level, spec["end_ratio"])
          & frame.uptrend_end.eq(1))
    return ok.astype(float).where(known)

def name(algo: str = "lgbm") -> str:
    spec = for_model(algo)
    return (f"1ヶ月内+{spec['threshold'] * 100:g}% / 終盤+{spec['end_ratio'] * 100:g}% / "
            "MA5>=MA20 / 翌営業日寄り基準")

def require(meta: dict, algo: str = "lgbm", field: str = "labelSpec") -> None:
    expected = for_model(algo)
    if expected is not None and meta.get(field) != expected:
        raise ValueError(f"{algo}: label definition mismatch; expected {name(algo)}")

def check_training_frame(frame: pd.DataFrame, algo: str = "lgbm") -> None:
    want = labels(frame, algo)
    actual = pd.to_numeric(frame["label"], errors="coerce")
    same = (want.isna() & actual.isna()) | (want.notna() & actual.eq(want))
    if not same.all():
        raise ValueError(f"{algo}: {int((~same).sum())} labels disagree with the production definition")
