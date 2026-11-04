#!/usr/bin/env python3
"""Exploratory holdout ablation of annual CCC/ROIC components.

No subscription, production feature list, parameters, or models are changed.
Input corrections are private, source-backed CSVs; do not check financial data
into the public repository. Console output contains aggregates only.

Arms fixed before evaluating outcomes:
 A: existing features; Q: A + quality/availability indicators;
 B: Q + financial values; P: Q + values shuffled within decision date, with
    each feature's missing mask preserved.
Classifier +10/+15 and the existing daily/rolling-five-day LambdaRank tasks
use the same inputs and frozen parameters in every arm, seeds 42/43/44.
Historical EDINET revisions remain unverified: this is not adoption evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import capital_efficiency as CE
import features as F
import live_track as LT
import ltr
import ltr5
import label_specs as LS
import train_model as T
import tuning

SEEDS = (42, 43, 44)
ARMS = ("A", "Q", "B", "P")
PERM_SEED = 20261103


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def forward_maturity(keys, bars, hold=20):
    """Exact per-code bar at decision+hold, matching live_track.forward.

    A date without a decision-day bar is unavailable, even if bars exist later.
    This also handles company suspensions, which a calendar-day embargo misses.
    """
    keys = keys[["Code", "Date"]].reset_index(drop=True).copy()
    keys["Date"] = pd.to_datetime(keys.Date)
    result = np.full(len(keys), np.datetime64("NaT"), dtype="datetime64[ns]")
    want = {code: g.index.to_numpy() for code, g in keys.groupby("Code", sort=False)}
    for code, g in bars.groupby("Code", sort=False):
        if code not in want:
            continue
        ix = want[code]
        dates = np.sort(pd.to_datetime(g.Date).to_numpy(dtype="datetime64[ns]"))
        selected = keys.loc[ix, "Date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(dates, selected)
        valid = (pos + hold < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == selected)
        result[ix[valid]] = dates[pos[valid] + hold]
    return pd.Series(result, index=keys.index)


def add_controls(df):
    out = df.copy()
    for c in CE.columns("values"):
        out["missing_" + c] = out[c].isna().astype(float)
    return out


def shuffled_values(df, seed=PERM_SEED):
    """Keep the original date and missing mask, destroying numerical ordering."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    groups = out.groupby("Date", sort=True).indices
    for c in CE.columns("values"):
        values = out[c].to_numpy(copy=True)
        for ix in groups.values():
            observed = ix[np.isfinite(values[ix])]
            values[observed] = rng.permutation(values[observed])
        out[c] = values
        assert np.array_equal(out[c].isna(), df[c].isna())
    return out


def column_sets(rolling=False):
    base = F.columns("all_plus_prog_listing_vol_book_shape_dd_recent_tx") + (list(ltr5.MOVE_COLS) if rolling else [])
    quality = CE.columns("quality") + ["missing_" + c for c in CE.columns("values")]
    return {"A": base, "Q": base + quality, "B": base + quality + CE.columns("values"),
            "P": base + quality + CE.columns("values")}


def parameter_outcome_end(daily, pool, train_end):
    """Include rolling decision dates as well as original candidate dates."""
    values = [d.loc[d.Date.le(train_end), "maturity_date"].max() for d in [daily, pool]]
    return max(x for x in values if pd.notna(x)) if any(pd.notna(x) for x in values) else pd.NaT


def prepare(args):
    p = args.output
    fin = pd.read_parquet(args.data_dir / "edinet_fin.parquet")
    companies = pd.read_parquet(args.data_dir / "edinet_companies.parquet")
    cols = ["Code", "DiscDate", "DocType", "CurPerType", "CurFYSt", "CurFYEn", "CurPerSt", "CurPerEn"]
    paths = sorted(args.data_dir.glob("fins_*.parquet"))
    fins = pd.concat([pd.read_parquet(x, columns=cols) for x in paths], ignore_index=True)
    corrections = pd.read_csv(args.corrections, dtype={"Code": str}) if args.corrections else None
    panel = CE.annual_panel(fin, CE.periods_from_fins(fins), companies, corrections)
    original = pd.read_parquet(args.dataset)
    original["Date"] = pd.to_datetime(original.Date)
    daily = CE.attach(original, panel)
    for c in original:
        pd.testing.assert_series_equal(daily[c], original[c])
    daily = add_controls(daily)
    coverage = pd.DataFrame([{"feature": c, "rows": int(daily[c].notna().sum()),
                             "pct": 100 * daily[c].notna().mean()} for c in CE.columns()])
    coverage.to_csv(p / "coverage.csv", index=False)
    panel.to_parquet(p / "annual_panel.parquet", index=False)
    daily.to_parquet(p / "dataset.parquet", index=False)
    source = {"dataset_sha256": digest(args.dataset), "prices_sha256": digest(args.prices),
              "annual_sha256": digest(args.data_dir / "edinet_fin.parquet"),
              "companies_sha256": digest(args.data_dir / "edinet_companies.parquet"),
              "fins_sha256": {x.name: digest(x) for x in paths},
              "corrections_sha256": digest(args.corrections) if args.corrections else None,
              "module_sha256": digest(HERE / "capital_efficiency.py"),
              "experiment_sha256": digest(__file__), "existing_columns_identical": True,
              "parameter_files_sha256": {name: digest(HERE / name) for name in
                  ["lgbm_params.json", "multi_params.json", "ltr_params.json", "ltr5_params.json"]},
              "rows": len(daily), "codes": int(daily.Code.nunique()),
              "historical_revision_lineage_verified": False}
    write_json(p / "inputs.json", source)
    del panel, fins, fin, original
    return daily


def prepare_ranking(daily, prices):
    bars = pd.read_parquet(prices, columns=["Code", "Date", "open", "high", "low", "close"])
    bars = bars[bars.Code.isin(set(daily.Code)) & bars.Date.ge(daily.Date.min() - pd.Timedelta(days=10))]
    bars = bars.rename(columns={"open": "AdjO", "high": "AdjH", "low": "AdjL", "close": "AdjC"})
    daily = ltr.attach_r20(daily, bars)
    daily["maturity_date"] = forward_maturity(daily, bars).to_numpy()
    cols = column_sets()["B"]
    cal = ltr5.calendar(bars.Date.unique())
    pool = ltr5.pool_frame(daily, bars, cal, cols,
                           decision_days=cal[(cal >= daily.Date.min()) & (cal <= daily.Date.max())],
                           keep=("label", "ce_avail_date", "ce_fiscal_year"))
    pool = ltr5.attach_exit(pool, bars)
    pool["maturity_date"] = forward_maturity(pool, bars).to_numpy()
    move = ltr5.check_move_cols(pool, bars, cal)
    same = ltr5.check_same_day(pool, daily)
    if move["same"] != move["rows"] or same["mismatch"]:
        raise ValueError("Rolling pool differs from production outcome/feature definitions")
    if (pool.ce_avail_date > pool.cand_date).any():
        raise ValueError("Financial information was not available on the candidate date")
    return daily, pool, {"rolling_rows": len(pool), "move_checks": move, "same_day_checks": same}


def parameters():
    standard = json.loads((HERE / "lgbm_params.json").read_text())[F.DEFAULT_PRESET]
    extra = json.loads((HERE / "multi_params.json").read_text())["lgbm_k15"]
    return {"class10": {k: v for k, v in standard.items() if not str(k).startswith("_")},
            "class15": dict(extra["params"], objective="binary", boosting_type="gbdt", n_estimators=200,
                            subsample_freq=1, verbose=-1),
            "ltr": ltr.load_params()["params"], "ltr5": ltr5.load_params()["params"]}


def classification_labels(daily):
    """Use the production contract, including tolerance at an exact threshold."""
    return {"y_class10": LS.labels(daily, "lgbm"),
            "y_class15": LS.labels(daily, "lgbm_k15")}


def classifier_params(params, y, seed, threads):
    """Match production's imbalance correction, using training labels only."""
    return dict(params, n_jobs=threads, random_state=seed, n_estimators=200,
                verbose=-1, scale_pos_weight=tuning.scale_pos_weight(y))


def evaluate(kind, d, score):
    if not len(d):
        return {}
    if kind.startswith("class"):
        y = d["y_" + kind].to_numpy(dtype=int)
        order = np.argsort(-score, kind="mergesort")[:max(1, int(np.ceil(.1 * len(d))))]
        return {"rows": len(d), "days": int(d.Date.nunique()), "positive_rate": float(y.mean()),
                "primary": float(average_precision_score(y, score)),
                "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                "top_decile_precision": float(y[order].mean()),
                "top_decile_close_return_pct": 100 * float(d[ltr.R20].iloc[order].mean())}
    gain = ltr.grades(d.Date, d[ltr.R20])
    f = d[["Code", "Date", ltr.R20]].assign(score=score)
    eligible_days = f.groupby("Date").size().loc[lambda x: x >= 5].index
    f = f[f.Date.isin(eligible_days)].sort_values(["Date", "score", "Code"], ascending=[True, False, True], kind="mergesort")
    return {"rows": len(d), "days": int(d.Date.nunique()), "eligible_days": len(eligible_days),
            "primary": float(ltr.ndcg_at_k(d.Date, score, gain)),
            "top1_close_return_pct": 100 * float(f.groupby("Date", sort=False).head(1)[ltr.R20].mean()),
            "top5_close_return_pct": 100 * float(f.groupby("Date", sort=False).head(5).groupby("Date")[ltr.R20].mean().mean())}


def run(args):
    import lightgbm as lgb

    daily = prepare(args)
    daily, pool, checks = prepare_ranking(daily, args.prices)
    train_end, test_start, test_end = T.holdout_bounds(daily.Date)
    # The frozen classifier params used data through train_end; avoid evaluating
    # while any of those training outcomes are still being determined.
    latest_param_outcome = parameter_outcome_end(daily, pool, train_end)
    if pd.isna(latest_param_outcome):
        raise ValueError("Cannot establish the latest parameter-training outcome")
    test_start = max(test_start, latest_param_outcome + pd.Timedelta(days=1))
    if test_start >= test_end:
        raise ValueError("No holdout remains after exact outcome separation")
    for name, values in classification_labels(daily).items():
        daily[name] = values
    np.testing.assert_array_equal(daily.y_class10.to_numpy(dtype=int), daily.label.to_numpy(dtype=int))
    params = parameters()
    plan = {"arms": {"A": "existing features", "Q": "A + quality and per-value missing indicators",
                     "B": "Q + 13 financial components", "P": "Q + date-shuffled observed values; missing masks identical"},
            "seeds": SEEDS, "permutation_seed": PERM_SEED, "trees": 200,
            "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "latest_parameter_training_outcome": latest_param_outcome,
            "parameters": params, "ranking_checks": checks,
            "rolling_feature_date": "candidate date; never refreshed at the pool decision date",
            "re_tuned": False, "primary_metrics": {"class10": "average precision", "class15": "average precision", "ltr": "daily NDCG@5", "ltr5": "pool NDCG@5"},
            "classifier_imbalance": "production tuning.scale_pos_weight(y_train); equal across arms",
            "adoption_test": False, "historical_revision_lineage_verified": False,
            "known_limitations": ["single previously observed holdout, not a fresh future period", "classifier family is LightGBM only", "ROIC/CCC include disclosed approximations and unreviewed source rows"]}
    write_json(args.output / "experiment_plan.json", plan)
    cache = args.output / "score_cache"
    cache.mkdir(exist_ok=True)
    fingerprint = digest(args.output / "inputs.json") + digest(args.output / "experiment_plan.json")
    all_results, windows, seed_results, splits = [], [], [], []
    for kind in ["class10", "class15", "ltr", "ltr5"]:
        d = (pool if kind == "ltr5" else daily).copy()
        valid = d[ltr.R20].notna() & d.maturity_date.notna()
        if kind.startswith("class"):
            valid &= d["y_" + kind].notna()
        elif kind == "ltr5":
            valid &= ltr5.trainable(d)
        tr = d[valid & d.Date.le(train_end) & d.maturity_date.lt(test_start)].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
        te = d[valid & d.Date.ge(test_start) & d.Date.le(test_end)].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
        assert tr.maturity_date.max() < te.Date.min()
        split = {"task": kind, "train_rows": len(tr), "test_rows": len(te),
                 "train_max_date": tr.Date.max(), "train_max_outcome": tr.maturity_date.max(),
                 "test_min_date": te.Date.min(), "test_max_date": te.Date.max()}
        splits.append(split)
        write_json(args.output / "split_checks.json", splits)
        shuffled_tr, shuffled_te = shuffled_values(tr), shuffled_values(te)
        cols_of = column_sets(kind == "ltr5")
        outputs = te[["Code", "Date", ltr.R20, "ce_roic_etr_candidate", "ce_ccc_cogs_proxy"]].copy()
        for arm in ARMS:
            cols = cols_of[arm]
            xtr, xte = (shuffled_tr, shuffled_te) if arm == "P" else (tr, te)
            scores = []
            for seed in SEEDS:
                start = time.monotonic()
                p = dict(params[kind], n_jobs=args.threads, random_state=seed, n_estimators=200, verbose=-1)
                cache_key = hashlib.sha256((fingerprint + kind + arm + str(seed)).encode()).hexdigest()
                cached_score = cache / (cache_key + ".npy")
                if cached_score.exists():
                    score = np.load(cached_score, allow_pickle=False)
                    if score.shape != (len(te),) or not np.isfinite(score).all():
                        raise ValueError("Invalid cached predictions")
                elif kind.startswith("class"):
                    y = tr["y_" + kind].to_numpy(dtype=int)
                    model = lgb.LGBMClassifier(**classifier_params(p, y, seed, args.threads))
                    model.fit(xtr[cols].to_numpy(dtype=float), y)
                    score = model.predict_proba(xte[cols].to_numpy(dtype=float))[:, 1]
                else:
                    model = ltr.fit(xtr, cols, p, seed=seed)
                    score = ltr.predict(model, xte[cols].to_numpy(dtype=float))
                if not cached_score.exists():
                    tmp = cached_score.with_suffix(".tmp")
                    with tmp.open("wb") as out:
                        np.save(out, score, allow_pickle=False)
                    tmp.replace(cached_score)
                scores.append(score)
                seed_results.append({"task": kind, "arm": arm, "seed": seed, **evaluate(kind, te, score)})
                print(json.dumps({"task": kind, "arm": arm, "seed": seed, "seconds": round(time.monotonic() - start, 1)}, ensure_ascii=False), flush=True)
            score = np.mean(scores, axis=0)
            outputs[arm] = score
            for subset, mask in [("all", np.ones(len(te), dtype=bool)), ("roic_available", te.ce_roic_etr_candidate.notna().to_numpy())]:
                if mask.any():
                    all_results.append({"task": kind, "arm": arm, "subset": subset, "features": len(cols), **evaluate(kind, te.loc[mask], score[mask])})
            for period in sorted(te.Date.dt.to_period("Q").unique()):
                mask = te.Date.dt.to_period("Q").eq(period).to_numpy()
                if mask.sum() >= 100 and te.loc[mask, "Date"].nunique() >= 20:
                    windows.append({"task": kind, "arm": arm, "quarter": str(period), **evaluate(kind, te.loc[mask], score[mask])})
            pd.DataFrame(all_results).to_csv(args.output / "results.csv", index=False)
            pd.DataFrame(seed_results).to_csv(args.output / "seed_results.csv", index=False)
            pd.DataFrame(windows).to_csv(args.output / "quarter_results.csv", index=False)
        outputs.to_parquet(args.output / (kind + "_predictions.parquet"), index=False)
    write_json(args.output / "completed.json", {"tasks": 4, "arms": 4, "seeds": len(SEEDS), "fits": len(seed_results),
                                               "production_changed": False, "premium_changed": False})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True, help="Saved adjusted panel: Code/Date/open/high/low/close")
    ap.add_argument("--corrections", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    if args.dataset.resolve() == (args.output / "dataset.parquet").resolve():
        raise ValueError("Output must not replace the input dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
