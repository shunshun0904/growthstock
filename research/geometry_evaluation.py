"""Purged, nested temporal comparisons of M0--M3 using the existing LambdaRank.

Research only. Never reads production tuned parameters, writes model assets, or
changes the live strategy. See docs/STOCK_DISTRIBUTION_GEOMETRY.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import distribution_geometry as G
import ltr


@dataclass(frozen=True)
class EvaluationConfig:
    hold: int = 20
    min_train_months: int = 36
    test_months: int = 6
    validation_months: int = 6
    embargo_sessions: int = 1
    min_train_rows: int = 1000
    min_test_rows: int = 200
    min_day: int = 5
    top_k: int = 2
    candidates: int = 4
    n_estimators: int = 200
    n_jobs: int = 2
    seed: int = 42
    cost_bps: float = 10.0
    bootstrap_block: int = 40
    bootstrap_repetitions: int = 2000

    def __post_init__(self):
        for name in ("hold", "min_train_months", "test_months", "validation_months", "min_train_rows",
                     "min_test_rows", "min_day", "top_k", "candidates", "n_estimators", "n_jobs",
                     "bootstrap_block", "bootstrap_repetitions"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_day < 2 or not 1 <= self.candidates <= 4 or self.bootstrap_block < self.hold:
            raise ValueError("min_day >= 2, 1 <= candidates <= 4, bootstrap_block >= hold required")
        if (not isinstance(self.embargo_sessions, int) or self.embargo_sessions < 0
                or not np.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000):
            raise ValueError("Invalid embargo_sessions or cost_bps")


def attach_targets(samples, bars, sessions, hold=20):
    """AdjO[t+1] -> AdjC[t+H] on actual exchange sessions, with maturity dates.

    Matches ltr.attach_r20 on complete histories. An absent bar never shifts the
    buy/exit to the next observed row. Missing exits remain unknown, not zero.
    """
    if not isinstance(hold, int) or hold < 1:
        raise ValueError("hold must be a positive integer")
    days = G.session_index(sessions)
    out, b = G.keyed_frame(samples), G.keyed_frame(bars)
    if not {"AdjO", "AdjC"}.issubset(b) or not out.Date.isin(days).all() or not b.Date.isin(days).all():
        raise ValueError("Adjusted open/close and calendar coverage are required")
    for col in (ltr.R20, "entry_date", "label_end"):
        if col in out:
            raise ValueError(f"Refusing to overwrite existing target column: {col}")
    i = days.get_indexer(out.Date)
    out["entry_date"] = pd.to_datetime([days[j+1] if j+1 < len(days) else pd.NaT for j in i])
    out["label_end"] = pd.to_datetime([days[j+hold] if j+hold < len(days) else pd.NaT for j in i])
    prices = b.set_index(["Code", "Date"])
    entry = pd.to_numeric(prices.AdjO.reindex(pd.MultiIndex.from_arrays([out.Code, out.entry_date])), errors="coerce").to_numpy()
    close = pd.to_numeric(prices.AdjC.reindex(pd.MultiIndex.from_arrays([out.Code, out.label_end])), errors="coerce").to_numpy()
    ok = np.isfinite(entry) & np.isfinite(close) & (entry > 0) & (close > 0)
    out[ltr.R20] = np.divide(close, entry, out=np.full(len(out), np.nan), where=ok)-1
    return out


def _mature_before(frame, boundary, days, embargo):
    idx = days.searchsorted(pd.Timestamp(boundary))-embargo
    if idx <= 0:
        return pd.Series(False, index=frame.index)
    cutoff = days[min(idx, len(days)-1)]
    return frame.label_end.notna() & (frame.label_end < cutoff)


def walk_forward_splits(frame, sessions, config=EvaluationConfig()):
    """Existing 36/6/6-month expanding boundaries plus exact label purging.

    Inner validation is the last six months of each outer training sample period.
    Every arm gets exactly the same inner/validation/train/test row indices.
    """
    import walkforward as WF

    if not frame.index.is_unique:
        raise ValueError("Frame index must be unique")
    days = G.session_index(sessions)
    folds = WF.make_folds(frame.Date, min_train_months=config.min_train_months,
                          test_months=config.test_months, step_months=config.test_months,
                          embargo_days=config.hold)
    labelled = np.isfinite(frame[ltr.R20]) & frame["label"].notna()
    splits = []
    for f in folds:
        te = frame[(frame.Date >= f.test_start) & (frame.Date <= f.test_end)]
        if len(te) < config.min_test_rows:
            continue
        tr = frame[(frame.Date <= f.train_end) & labelled
                   & _mature_before(frame, te.Date.min(), days, config.embargo_sessions)]
        if len(tr) < config.min_train_rows:
            continue
        validation_start = tr.Date.max() - pd.DateOffset(months=config.validation_months)
        va = tr[tr.Date >= validation_start]
        inner = tr[(tr.Date < validation_start)
                   & _mature_before(tr, validation_start, days, config.embargo_sessions)]
        if len(inner) < config.min_train_rows or len(va) < config.min_test_rows:
            continue
        splits.append({"fold": f.index, "train": tr.index.tolist(), "validation": va.index.tolist(),
                       "inner_train": inner.index.tolist(), "test": te.index.tolist()})
    if not splits:
        raise ValueError("No usable temporal folds; need more history/rows for nested validation")
    return splits


def candidate_params(config=EvaluationConfig()):
    """Small preregistered grid, same budget for all arms; no OOS-driven selection."""
    shapes = ((15, 30, .03), (31, 30, .03), (15, 60, .05), (31, 60, .05))
    return [dict(boosting_type="gbdt", num_leaves=leaves, min_child_samples=child,
                 learning_rate=rate, n_estimators=config.n_estimators, n_jobs=config.n_jobs,
                 subsample=1., colsample_bytree=1., reg_lambda=1., verbose=-1,
                 deterministic=True, force_col_wise=True)
            for leaves, child, rate in shapes[:config.candidates]]


def run_walk_forward(frame, base_columns, sessions, geometry=G.GeometryConfig(), config=EvaluationConfig(), *, checkpoint_dir=None):
    """Tune each arm on its purged past validation, refit on past, then predict OOS."""
    d = G.keyed_frame(frame).sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    cols_all = G.feature_columns(base_columns, "M3", geometry)
    if not set(cols_all).issubset(d):
        raise ValueError("Required baseline/geometry columns are missing")
    if np.isinf(d[cols_all].to_numpy(dtype=float)).any():
        raise ValueError("Features contain infinity; fix input instead of silently clipping")
    grid = candidate_params(config)
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(pd.util.hash_pandas_object(d, index=True).values.tobytes())
        digest.update(repr((list(d.columns), list(base_columns), list(sessions), geometry, config)).encode())
        for source in (Path(__file__), Path(G.__file__), Path(ltr.__file__)):
            digest.update(source.read_bytes())
        signature = digest.hexdigest()
    outputs, audit = [], []
    for split in walk_forward_splits(d, sessions, config):
        tr, va, inner, te = (d.loc[split[k]] for k in ("train", "validation", "inner_train", "test"))
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
            d.loc[split["train"]+split["test"], ["Date", "Code"]], index=False).values.tobytes()).hexdigest()
        for arm in G.ARMS:
            cols = G.feature_columns(base_columns, arm, geometry)
            if checkpoint is not None:
                stem = checkpoint / f"fold{split['fold']}_{arm}"
                meta_path, pred_path = stem.with_suffix('.json'), stem.with_suffix('.parquet')
                if meta_path.exists() and pred_path.exists():
                    cached = json.loads(meta_path.read_text())
                    if cached['signature'] != signature:
                        raise ValueError('Checkpoint inputs/configuration/source changed; use a new output directory')
                    if hashlib.sha256(pred_path.read_bytes()).hexdigest() != cached['prediction_sha256']:
                        raise ValueError('Checkpoint predictions are corrupted')
                    outputs.append(pd.read_parquet(pred_path))
                    audit.append(cached['audit'])
                    print(f"[geometry] W={geometry.window} fold={split['fold']} {arm}: resumed", flush=True)
                    continue
            scores = []
            for params in grid:
                model = ltr.fit(inner, cols, params, config.seed)
                pred = ltr.predict(model, va[cols].to_numpy(dtype=float))
                scores.append(ltr.ndcg_at_k(va.Date, pred, ltr.grades(va.Date, va[ltr.R20]),
                                             k=5, min_n=config.min_day))
            if not np.isfinite(scores).all():
                raise ValueError("Validation lacks usable date groups for NDCG; no fallback to test data")
            selected = int(np.argmax(scores))  # deterministic first-candidate tie break
            model = ltr.fit(tr, cols, grid[selected], config.seed)
            keep = ["Code", "Date", "entry_date", "label_end", ltr.R20]
            keep += [c for c in ("cap_band", "ret_20d", "vol_20d", "topix_ma200_gap") if c in te]
            part = te[keep].copy()
            part["score"] = ltr.predict(model, te[cols].to_numpy(dtype=float))
            if not np.isfinite(part.score).all():
                raise ValueError("Nonfinite predictions")
            part["fold"], part["arm"], part["window"] = split["fold"], arm, geometry.window
            outputs.append(part)
            audit.append({"fold": split["fold"], "arm": arm, "window": geometry.window,
                          "n_train": len(tr), "n_inner": len(inner), "n_validation": len(va), "n_test": len(te),
                          "train_last_signal": str(tr.Date.max()), "train_last_label": str(tr.label_end.max()),
                          "inner_last_label": str(inner.label_end.max()), "validation_start": str(va.Date.min()),
                          "test_start": str(te.Date.min()), "test_end": str(te.Date.max()),
                          "rows_fingerprint": fingerprint, "features": cols, "selected_candidate": selected,
                          "validation_scores": scores, "params": grid[selected]})
            if checkpoint is not None:
                tmp = stem.with_suffix('.parquet.tmp')
                part.to_parquet(tmp, index=False)
                tmp.replace(pred_path)
                payload = {'signature': signature, 'audit': audit[-1],
                           'prediction_sha256': hashlib.sha256(pred_path.read_bytes()).hexdigest()}
                tmp = stem.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
                tmp.replace(meta_path)
            print(f"[geometry] W={geometry.window} fold={split['fold']} {arm}: train={len(tr)}, test={len(te)}", flush=True)
    return pd.concat(outputs, ignore_index=True), audit


def daily_metrics(predictions, top_k=2, min_day=5):
    """Equal-weight dates. Rank IC/NDCG on known labels; top selection before filtering labels."""
    rows = []
    for date, g in predictions.groupby("Date", sort=True):
        g = g.sort_values(["score", "Code"], ascending=[False, True], kind="mergesort")
        known = g[np.isfinite(g[ltr.R20]) & np.isfinite(g.score)]
        rec = {"Date": date, "n_scored": len(g), "n_labelled": len(known), "rank_ic": np.nan,
               "ndcg": np.nan, "top_return": np.nan, "bottom_return": np.nan, "top_bottom": np.nan}
        if len(known) >= min_day:
            if known.score.nunique() > 1 and known[ltr.R20].nunique() > 1:
                rec["rank_ic"] = known.score.rank().corr(known[ltr.R20].rank())
            rec["ndcg"] = ltr.ndcg_at_k(known.Date, known.score, ltr.grades(known.Date, known[ltr.R20]), k=5, min_n=min_day)
        if len(g) >= 2*top_k:
            top, bottom = g[ltr.R20].iloc[:top_k], g[ltr.R20].iloc[-top_k:]
            rec["top_return"] = float(top.mean()) if np.isfinite(top).all() else np.nan
            rec["bottom_return"] = float(bottom.mean()) if np.isfinite(bottom).all() else np.nan
            rec["top_bottom"] = rec["top_return"]-rec["bottom_return"]
        rows.append(rec)
    return pd.DataFrame(rows)


def portfolio_curve(predictions, bars, sessions, hold=20, top_k=2, cost_bps=10.):
    """Nonoverlapping H-session cohorts, picked after close, bought next open.

    Fixed calendar schedule starts at first OOS signal. Each slot gets 1/K NAV;
    absent candidates/unfilled opens leave cash. Exit at session t+H close,
    including both sides' notional costs. Daily mark-to-market is mandatory:
    unknown close of an owned stock raises (never drops the trade or forward-fills).
    Only complete scheduled cohorts enter; the truncated tail is not invested.
    """
    if hold < 1 or top_k < 1 or not 0 <= cost_bps < 10000:
        raise ValueError("Invalid portfolio configuration")
    days = G.session_index(sessions)
    pred = G.keyed_frame(predictions)
    if pred.empty or not pred.Date.isin(days).all():
        raise ValueError("Predictions/calendar are empty or inconsistent")
    if not np.isfinite(pred.score).all():
        raise ValueError("Invalid scores")
    start, last = days.searchsorted(pred.Date.min()), days.searchsorted(pred.Date.max())
    signals = list(range(start, min(last+1, len(days)-hold), hold))
    if not signals:
        raise ValueError("No complete portfolio cohorts")
    selected = {i: pred[pred.Date == days[i]].sort_values(["score", "Code"], ascending=[False, True]).head(top_k)
                for i in signals}
    codes = {code for picks in selected.values() for code in picks.Code}
    # Only the selected positions need price lookup; avoid indexing ten million bars
    # again for every arm/cost scenario. Selection uses scores, never future prices.
    b = G.keyed_frame(bars[bars.Code.astype(str).isin(codes)])
    prices = b.set_index(["Code", "Date"])
    nav, fee, records = 1., cost_bps/10000., []
    for i in signals:
        picks = selected[i]
        budget, cash, positions, buy_notional = nav/top_k, nav, {}, 0.
        for code in picks.Code:
            opening = prices.AdjO.get((code, days[i+1]), np.nan)
            if np.isfinite(opening) and opening > 0:
                notional = budget/(1+fee)
                positions[code] = notional/opening
                cash -= budget
                buy_notional += notional
        for j in range(i+1, i+hold+1):
            value = 0.
            for code, shares in positions.items():
                closing = prices.AdjC.get((code, days[j]), np.nan)
                if not np.isfinite(closing) or closing <= 0:
                    raise ValueError("Held position has an unknown daily close; portfolio valuation is unverified")
                value += shares*closing
            traded = buy_notional if j == i+1 else 0.
            costs = buy_notional*fee if j == i+1 else 0.
            if j == i+hold:
                traded += value
                costs += value*fee
                cash += value*(1-fee)
                value = 0.
            new_nav = cash+value
            records.append({"Date": days[j], "nav": new_nav, "return": new_nav/nav-1,
                            "turnover": traded/nav, "cost": costs, "positions": len(positions) if j < i+hold else 0,
                            "unfilled": len(picks)-len(positions) if j == i+1 else 0})
            nav = new_nav
    return pd.DataFrame(records)


def portfolio_stats(curve):
    r = curve["return"].to_numpy(dtype=float)
    nav = np.r_[1., curve.nav.to_numpy(dtype=float)]
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return {"sessions": len(r), "total_return": float(nav[-1]-1),
            "annualized_return": float(nav[-1]**(252/len(r))-1),
            "sharpe": float(np.sqrt(252)*r.mean()/sd) if sd > 0 else np.nan,
            "max_drawdown": float((nav/np.maximum.accumulate(nav)-1).min()),
            "annualized_turnover": float(curve.turnover.mean()*252),
            "unfilled_entries": int(curve.unfilled.sum())}


def paired_block_interval(baseline, treatment, block=40, repetitions=2000, seed=42):
    """Moving-block percentile CI for treatment-baseline, preserving missing dates.

    Call with arrays reindexed to all exchange sessions, not compressed event days.
    Inference is conditional on fitted models; it does not correct multiple searches.
    """
    a, b = np.asarray(baseline, float), np.asarray(treatment, float)
    if a.shape != b.shape or a.ndim != 1 or block < 1 or repetitions < 1:
        raise ValueError("Invalid paired bootstrap inputs")
    delta = b-a
    finite = np.isfinite(delta)
    result = {"n_pairs": int(finite.sum()), "block_sessions": block,
              "difference": float(delta[finite].mean()) if finite.any() else np.nan,
              "ci_low": np.nan, "ci_high": np.nan, "status": "insufficient_history"}
    if len(delta) < 2*block or finite.sum() < 2:
        return result
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repetitions):
        starts = rng.integers(0, len(delta)-block+1, size=int(np.ceil(len(delta)/block)))
        ix = (starts[:, None]+np.arange(block)).ravel()[:len(delta)]
        sample = delta[ix]
        if np.isfinite(sample).any():
            estimates.append(float(sample[np.isfinite(sample)].mean()))
    if estimates:
        result.update(ci_low=float(np.quantile(estimates,.025)), ci_high=float(np.quantile(estimates,.975)), status="ok")
    return result
