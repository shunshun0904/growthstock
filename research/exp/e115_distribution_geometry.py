#!/usr/bin/env python3
"""E115: stock-specific distribution geometry vs the existing 360-feature LTR.

    python research/exp/e115_distribution_geometry.py --windows 30
    python research/exp/e115_distribution_geometry.py --windows 20,30,60 --seed 7

Requires an existing dataset.parquet, split-adjusted bars_*.parquet and the
exchange calendar. Outputs only under a new directory; no model/live writes.
Public stdout contains aggregates only; row-level files stay in ignored _data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import distribution_geometry as G
import geometry_evaluation as E
import features as F
import trading_calendar as TC

METRICS = ["rank_ic", "ndcg", "top_return", "bottom_return", "top_bottom"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    # Strict JSON; unavailable estimates are null, never nonstandard NaN literals.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def summarize(predictions, bars, days, out, config):
    daily, portfolio, portfolio_rows, strata = [], [], [], []
    for (window, arm), part in predictions.groupby(["window", "arm"], sort=True):
        dm = E.daily_metrics(part, top_k=config.top_k, min_day=config.min_day)
        dm["window"], dm["arm"] = window, arm
        daily.append(dm)
        for cost in sorted(set((0., 5., config.cost_bps, 25.))):
            rec = {"window": int(window), "arm": arm, "cost_bps_per_side": cost}
            try:
                curve = E.portfolio_curve(part, bars, days, hold=config.hold, top_k=config.top_k, cost_bps=cost)
                rec.update(E.portfolio_stats(curve), status="ok")
                curve["window"], curve["arm"], curve["cost_bps_per_side"] = window, arm, cost
                portfolio_rows.append(curve)
            except ValueError as ex:
                rec.update(status="unverified", reason=str(ex))
            portfolio.append(rec)
        # All grouping variables are observed on the signal date. Descriptive only.
        p = part.copy()
        if "ret_20d" in p:
            p["momentum_sign"] = np.where(p.ret_20d.isna(), "unknown", np.where(p.ret_20d >= 0, "nonnegative", "negative"))
        if "topix_ma200_gap" in p:
            p["market_regime"] = np.where(p.topix_ma200_gap.isna(), "unknown", np.where(p.topix_ma200_gap >= 0, "above_ma200", "below_ma200"))
        for col in ("cap_band", "momentum_sign", "market_regime", "momentum_fid"):
            if col not in p:
                continue
            for value, sub in p.groupby(col, dropna=False):
                sm = E.daily_metrics(sub, top_k=config.top_k, min_day=config.min_day)
                strata.append({"window": int(window), "arm": arm, "stratum": col, "value": str(value),
                               "rows": len(sub), "ic_days": int(sm.rank_ic.notna().sum()),
                               **{m: float(sm[m].mean()) for m in METRICS}})
    daily = pd.concat(daily, ignore_index=True)
    daily.to_csv(out / "daily_metrics.csv", index=False)
    for name, period in (("monthly", "M"), ("yearly", "Y")):
        grouped = daily.assign(period=daily.Date.dt.to_period(period).astype(str)).groupby(["window", "arm", "period"])
        table = grouped[METRICS].mean().join(grouped.rank_ic.count().rename("ic_days"))
        table.to_csv(out / f"{name}_metrics.csv")
    summary = daily.groupby(["window", "arm"])[METRICS].mean().reset_index()
    summary.to_csv(out / "ranking_summary.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "strata.csv", index=False)
    if portfolio_rows:
        pd.concat(portfolio_rows, ignore_index=True).to_csv(out / "portfolio_daily.csv", index=False)
    save_json(out / "portfolio_summary.json", portfolio)
    comparisons = []
    for window, group in daily.groupby("window"):
        calendar = days[(days >= group.Date.min()) & (days <= group.Date.max())]
        for metric in ("rank_ic", "ndcg", "top_bottom"):
            matrix = group.pivot(index="Date", columns="arm", values=metric).reindex(calendar)
            for baseline, treatment in (("M1", "M3"), ("M0", "M1"), ("M0", "M2"), ("M0", "M3"), ("M2", "M3")):
                ci = E.paired_block_interval(matrix[baseline], matrix[treatment], block=config.bootstrap_block,
                                             repetitions=config.bootstrap_repetitions, seed=config.seed)
                comparisons.append({"window": int(window), "metric": metric, "baseline": baseline,
                                    "treatment": treatment, "primary": window == 30 and baseline == "M1" and treatment == "M3" and metric == "rank_ic", **ci})
    save_json(out / "paired_comparisons.json", comparisons)
    print(summary.to_string(index=False), flush=True)
    print("[geometry] Intervals are conditional moving-block estimates; no multiple-search correction.", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "research/_data")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--windows", default="30", help="Predeclared sensitivity windows; never select the winner on OOS")
    ap.add_argument("--lags", default="5,20,60")
    ap.add_argument("--reference-lag", type=int, default=20)
    ap.add_argument("--preset", default=F.DEFAULT_PRESET)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=10.)
    ap.add_argument("--audit-only", action="store_true", help="Build features/targets/splits without fitting models")
    ap.add_argument("--resume", action="store_true", help="Resume matching checkpoints in an existing --out directory")
    args = ap.parse_args(argv)
    if args.resume and args.out is None:
        ap.error('--resume requires --out')
    windows = tuple(int(x) for x in args.windows.split(","))
    if not windows or len(set(windows)) != len(windows):
        ap.error("windows must be nonempty and unique")
    lags = tuple(int(x) for x in args.lags.split(","))
    geometries = [G.GeometryConfig(window=w, lags=lags, reference_lag=args.reference_lag) for w in windows]
    config = E.EvaluationConfig(seed=args.seed, candidates=args.candidates, n_jobs=args.n_jobs, cost_bps=args.cost_bps)
    base = F.columns(args.preset)
    dataset_path, calendar_path = args.data_dir / "dataset.parquet", args.data_dir / TC.FILENAME
    bar_paths = sorted(args.data_dir.glob("bars_*.parquet"))
    if not dataset_path.is_file() or not calendar_path.is_file() or not bar_paths:
        ap.error("Need dataset.parquet, calendar.parquet, and bars_*.parquet; build the existing dataset first")
    cal = TC.load(str(args.data_dir))
    if not cal:
        ap.error("Cannot read exchange calendar; no inferred weekday fallback")
    days = G.session_index(sorted(cal.days))
    samples = G.keyed_frame(pd.read_parquet(dataset_path))
    missing = set(base+['label'])-set(samples)
    if missing:
        ap.error(f"Missing baseline columns: {sorted(missing)}")
    bars = pd.concat([pd.read_parquet(p, columns=["Code", "Date", "AdjO", "AdjC", "AdjVo"]) for p in bar_paths], ignore_index=True)
    bars = G.keyed_frame(bars)
    bars = bars[bars.Code.isin(samples.Code.unique())]
    samples = E.attach_targets(samples, bars, days, config.hold).sort_values(["Date", "Code"]).reset_index(drop=True)
    out = args.out or args.data_dir / "oof" / ("e115_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if args.resume:
        if not (out / 'manifest.json').is_file():
            ap.error('--resume requires an existing manifest')
    else:
        out.mkdir(parents=True, exist_ok=False)
    source_paths = [ROOT / p for p in ("research/distribution_geometry.py", "research/geometry_evaluation.py",
                    "research/exp/e115_distribution_geometry.py", "research/ltr.py", "research/features.py",
                    "research/walkforward.py", "research/build_dataset.py")]
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                "inputs_sha256": {p.name: sha256(p) for p in [dataset_path, calendar_path]+bar_paths},
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow")},
                "evaluation": asdict(config), "geometries": [asdict(g) for g in geometries], "arms": G.ARMS,
                "preset": args.preset, "baseline_features": base, "candidate_grid": E.candidate_params(config),
                "rows": len(samples), "codes": samples.Code.nunique(), "start": str(samples.Date.min()), "end": str(samples.Date.max()),
                "missing_forward_returns": int(samples.ret_c20.isna().sum()),
                "primary": "M3 minus M1, equal-date mean rank IC, W30; other windows/metrics are sensitivity analyses",
                "as_of": "signal-day close; next exchange-session open entry; H-th session close exit",
                "universe": "existing breakout dataset; no geometry-dependent row exclusions",
                "limitations": ["Retrospective current-vintage data, not untouched prospective OOS",
                                "Baseline features were historically selected; nested tuning cannot undo that selection",
                                "AdjC is split-adjusted; dividends/delisting proceeds are not reconstructed",
                                "Dataset's existing outcome-dependent eligibility/survivorship limitations remain",
                                "Portfolio is a 20-session top-K cohort experiment, not the live TP10/three-slot rule"],
                "audit_only": args.audit_only}
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        for key in ('base_commit', 'branch', 'source_sha256', 'inputs_sha256', 'versions', 'evaluation', 'geometries',
                    'preset', 'baseline_features', 'candidate_grid', 'audit_only'):
            if json.dumps(previous[key], sort_keys=True) != json.dumps(manifest[key], sort_keys=True):
                ap.error(f'Cannot resume: {key} differs; use a new output directory')
        manifest['created_utc'] = previous['created_utc']
        manifest['resumed_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(out / "manifest.json", manifest)
    try:
        predictions, fits, coverage, redundancy = [], [], [], []
        for geo in geometries:
            frame = G.attach_features(samples, bars, days, geo, arm="M3")
            cols = G.feature_columns([], "M3", geo)
            for year, sub in frame.groupby(frame.Date.dt.year):
                for col in cols:
                    coverage.append({"window": geo.window, "year": int(year), "feature": col,
                                     "rows": len(sub), "missing_fraction": float(sub[col].isna().mean())})
            splits = E.walk_forward_splits(frame, days, config)
            save_json(out / f"splits_w{geo.window}.json", splits)
            if args.audit_only:
                continue
            pred, audit = E.run_walk_forward(frame, base, days, geo, config,
                                            checkpoint_dir=out / 'checkpoints' / f'w{geo.window}')
            fid = geo.prefix+f"fid_{geo.reference_lag}"
            diagnostics = frame[["Code", "Date", fid]].copy()
            diagnostics["fid_high"] = diagnostics[fid] >= diagnostics.groupby("Date")[fid].transform("median")
            pred = pred.merge(diagnostics, on=["Code", "Date"], how="left", validate="many_to_one")
            if "ret_20d" in pred:
                known = pred[fid].notna() & pred.ret_20d.notna()
                pred["momentum_fid"] = np.where(known, np.char.add(np.where(pred.ret_20d >= 0,"up_","down_"),
                                                                  np.where(pred.fid_high,"high_fid","low_fid")), "unknown")
            predictions.append(pred)
            fits.extend(audit)
            oos = frame[frame.Date.isin(pred.Date)]
            for col in cols:
                for reference in ("vol_20d", "ret_20d"):
                    if reference in oos:
                        corr = oos[[col, reference]].corr(method="spearman").iloc[0,1]
                        redundancy.append({"window": geo.window, "feature": col, "reference": reference, "oos_spearman": corr})
        pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
        if not args.audit_only:
            pred = pd.concat(predictions, ignore_index=True)
            pred.to_parquet(out / "predictions.parquet", index=False)
            save_json(out / "fits.json", fits)
            pd.DataFrame(redundancy).to_csv(out / "redundancy.csv", index=False)
            summarize(pred, bars, days, out, config)
        manifest["status"] = "audit_complete" if args.audit_only else "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(out / "manifest.json", manifest)
    print(f"[geometry] Saved {manifest['status']} outputs to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
