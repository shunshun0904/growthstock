#!/usr/bin/env python3
"""Research-only A/B/C/D/P walk-forward of price-independent EVA gaps.

Prepare-only by default. A complete historical series universe AND observed
fundamental values are required before fitting. Current inference targets are
for UI checks, never a future-selected historical backtest universe.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "exp"))
import features as F
import valuation_gap as G
import ltr
import e119_capital_efficiency as E
import e121_theoretical_value_retune as RETUNE
from build_roic_wacc_series import code5, json_write, digest

ARMS = ("A", "B", "C", "D", "P", "M")
SEEDS = (42, 43, 44)
# Neutral, predeclared settings, NOT production settings tuned on later dates.
FIXED = dict(learning_rate=.05, num_leaves=31, min_child_samples=20,
             subsample=.8, colsample_bytree=.8, reg_alpha=0., reg_lambda=1.)


def walk_forward(frame, train_months=36, test_months=6):
    dates = pd.to_datetime(frame.Date)
    first = dates.min()
    boundary = first + pd.DateOffset(months=train_months)
    folds = []
    while boundary <= dates.max():
        end = boundary + pd.DateOffset(months=test_months)
        test = np.flatnonzero((dates.ge(boundary) & dates.lt(end)).to_numpy())
        if len(test):
            start = dates.iloc[test].min()
            train = np.flatnonzero((dates.ge(boundary - pd.DateOffset(months=train_months))
                                    & dates.lt(start) & frame.maturity_date.lt(start)).to_numpy())
            if len(train):
                assert frame.iloc[train].maturity_date.max() < start
                assert not set(dates.iloc[train]) & set(dates.iloc[test])
                folds.append((train, test))
        boundary = end
    return folds


def ranking_metrics(frame, score):
    d = frame[["Date", "Code", ltr.R20]].copy().assign(score=np.asarray(score))
    d = d[d.groupby("Date").Date.transform("size").ge(5)]
    if d.empty:
        return {"eligible_days": 0}
    d = d.sort_values(["Date", "score", "Code"], ascending=[True, False, True], kind="mergesort")
    top5, top10, ic = [], [], []
    for _, group in d.groupby("Date", sort=True):
        top5.append(group.head(5)[ltr.R20].mean())
        top10.append(group.head(max(1, int(np.ceil(len(group) * .1))))[ltr.R20].mean() - group[ltr.R20].mean())
        if group.score.nunique() > 1 and group[ltr.R20].nunique() > 1:
            ic.append(group.score.corr(group[ltr.R20], method="spearman"))
    return {"eligible_days": int(d.Date.nunique()),
            "ndcg_at_5": float(ltr.ndcg_at_k(d.Date, d.score, ltr.grades(d.Date, d[ltr.R20]))),
            "top5_return_20d": float(np.mean(top5)), "top10pct_excess_20d": float(np.mean(top10)),
            "spearman_ic": float(np.mean(ic)) if ic else None}


def prepare(dataset, series_dir, prices, output):
    original = pd.read_parquet(dataset)
    missing = set(F.columns(F.DEFAULT_PRESET)) - set(original)
    if missing:
        raise ValueError("Current baseline features missing: " + str(sorted(missing)))
    parts, codes = [], set()
    fields = list(dict.fromkeys(G.CONTROLS + G.GAPS + G.TIME + G.MARKET_CONTROLS))
    for p in sorted(series_dir.glob("*.parquet")):
        f = pd.read_parquet(p)
        # Verify these are NEW independent-EVA series, not legacy CAPM files.
        if "fundamental_value_per_share" not in f or "independent_shares" not in f:
            raise ValueError("Series is not from the independent EVA pipeline: " + p.name)
        controls = G.market_control_features(f)
        f = pd.concat([f, controls], axis=1)
        codes.update(f.Code.map(code5))
        parts.append(f.reindex(columns=["Code", "Date", *fields]))
    features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Code", "Date", *fields])
    if features.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate feature security/date")
    features["Code"] = features.Code.map(code5)
    features["Date"] = pd.to_datetime(features.Date)
    original["Date"] = pd.to_datetime(original.Date)
    key = original[["Code", "Date"]].assign(_order=np.arange(len(original)))
    key["Code"] = key.Code.map(code5)
    attached = key.merge(features, on=["Code", "Date"], how="left", validate="many_to_one").sort_values("_order")
    d = original.copy()
    for c in fields:
        d[c] = attached[c].to_numpy()
    for c in original:
        pd.testing.assert_series_equal(d[c], original[c])
    bars = pd.read_parquet(prices, columns=["Code", "Date", "open", "high", "low", "close"])
    bars = bars.rename(columns={"open": "AdjO", "high": "AdjH", "low": "AdjL", "close": "AdjC"})
    d = ltr.attach_r20(d, bars)
    d["maturity_date"] = E.forward_maturity(d, bars).to_numpy()
    for c, values in E.classification_labels(d).items():
        d[c] = values
    coverage = pd.DataFrame([{"feature": c, "available": int(d[c].notna().sum()), "rows": len(d),
                             "fraction": float(d[c].notna().mean())} for c in fields])
    coverage.to_csv(output / "coverage.csv", index=False)
    absent = sorted(set(original.Code.map(code5)) - codes)
    status = "ready"
    reasons = []
    if absent:
        status = "blocked"
        reasons.append("incomplete_historical_universe")
    if not d.valuation_gap.notna().any():
        status = "blocked"
        reasons.append("no_observed_fundamental_valuation_gap")
    report = {"status": status, "reasons": reasons, "rows": len(d), "codes": int(original.Code.nunique()),
              "date_min": str(original.Date.min().date()), "date_max": str(original.Date.max().date()),
              "covered_codes": len(set(original.Code.map(code5)) & codes), "missing_code_count": len(absent),
              "gap_available_rows": int(d.valuation_gap.notna().sum()), "original_columns_identical": True,
              "production_preset": F.DEFAULT_PRESET, "production_feature_count": len(F.columns(F.DEFAULT_PRESET)),
              "adoption_status": "not_evaluated", "backtest_completed": False,
              "limitations": ["E120/E121 periods already observed; no independent holdout claim",
                              "Historical API revisions not fully archived", "Current inference selection cannot define historical universe"]}
    json_write(output / "preflight.json", report)
    return d, report


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = {"dataset": digest(args.dataset), "prices": digest(args.prices),
              "series": {p.name: digest(p) for p in sorted(args.series_dir.glob("*.parquet"))},
              "source": {p.name: digest(p) for p in [Path(__file__), HERE / "fundamental_value.py", HERE / "fundamental_series.py",
                           HERE / "valuation_gap.py", HERE / "features.py"]}}
    json_write(args.output / "inputs.json", source)
    d, preflight = prepare(args.dataset, args.series_dir, args.prices, args.output)
    cols = G.column_sets()
    plan = {"arms": {k: v for k, v in cols.items()}, "fixed_parameters": FIXED,
            "retune": args.retune, "trials_per_arm_fold": args.trials, "inner_folds": args.inner_folds,
            "outer_train_months": 36, "outer_test_months": 6, "trees": args.trees, "seeds": SEEDS,
            "shuffle": "joint permutation within date and feature-availability pattern",
            "selection": "all arm/fold selections frozen before evaluating outer predictions",
            "automatic_adoption": False, "independent_holdout": False}
    json_write(args.output / "plan.json", plan)
    if preflight["status"] != "ready" or not args.run:
        return preflight
    d = d[d[ltr.R20].notna() & d.maturity_date.notna() & d.label.notna()].sort_values(["Date", "Code"]).reset_index(drop=True)
    folds = walk_forward(d)
    if not folds:
        raise ValueError("No eligible walk-forward windows")
    shuffled = G.shuffle_within_date(d, list(dict.fromkeys(
        ["roic_conservative_tax30", "roic_operating_tax30", "wacc", "actual_tax"] + G.GAPS + G.TIME)))
    tasks = ["ltr", "class10", "class15"] if args.classification else ["ltr"]
    bank = RETUNE.candidate_bank(FIXED, args.trials) if args.retune else [FIXED]
    selected = []
    # Stage 1: only INNER validation outcomes, for EVERY arm/task/window.
    for task in tasks:
        for outer, (ti, vi) in enumerate(folds):
            for arm in ARMS:
                frame = shuffled if arm == "P" else d
                train = frame.iloc[ti].reset_index(drop=True)
                winner = FIXED
                if args.retune:
                    inner, _ = RETUNE.forward_folds(train, task, n_splits=args.inner_folds, min_train_months=12)
                    records = []
                    for candidate_id, candidate in enumerate(bank):
                        scores = []
                        for it, iv in inner:
                            trainer = RETUNE.Trainer(train.iloc[it], cols[arm], task, args.threads)
                            model = trainer.fit(candidate, trees=args.trees)
                            prediction = model.predict(train.iloc[iv][cols[arm]].to_numpy(float))
                            scores.append(RETUNE.primary(task, train.iloc[iv], prediction))
                        records.append({"candidate": candidate_id, "cv_mean": float(np.mean(scores))})
                    winner = bank[RETUNE.select_candidate(records)["candidate"]]
                    json_write(args.output / f"cv-{task}-{outer}-{arm}.json", records)
                selected.append({"task": task, "outer": outer, "arm": arm, "parameters": winner})
    json_write(args.output / "selection_frozen.json", selected)
    # Stage 2: outer evaluation can start only after the complete freeze file.
    results, importance = [], []
    for spec in selected:
        task, outer, arm = spec["task"], spec["outer"], spec["arm"]
        ti, vi = folds[outer]
        frame = shuffled if arm == "P" else d
        train, test = frame.iloc[ti], frame.iloc[vi]
        predictions = []
        for seed in SEEDS:
            trainer = RETUNE.Trainer(train, cols[arm], task, args.threads)
            model = trainer.fit(spec["parameters"], seed, args.trees)
            predictions.append(model.predict(test[cols[arm]].to_numpy(float)))
            for name, gain in zip(cols[arm], model.feature_importance(importance_type="gain")):
                importance.append({"task": task, "outer": outer, "arm": arm, "seed": seed, "feature": name, "gain": float(gain)})
        score = np.mean(predictions, axis=0)
        metric = ranking_metrics(test, score) if task == "ltr" else E.evaluate(task, test, score)
        results.append({**spec, "parameters": json.dumps(spec["parameters"], sort_keys=True), "cohort": "identical_all_rows", **metric})
        complete = test.valuation_gap.notna().to_numpy()
        if complete.any():
            metric = ranking_metrics(test.loc[complete], score[complete]) if task == "ltr" else E.evaluate(task, test.loc[complete], score[complete])
            results.append({**spec, "parameters": json.dumps(spec["parameters"], sort_keys=True), "cohort": "identical_valued_rows", **metric})
        pd.DataFrame({"Code": test.Code, "Date": test.Date, "score": score, "return_20d": test[ltr.R20]}).to_parquet(
            args.output / f"predictions-{task}-{outer}-{arm}.parquet", index=False)
    pd.DataFrame(results).to_csv(args.output / "results.csv", index=False)
    pd.DataFrame(importance).to_csv(args.output / "feature_importance.csv", index=False)
    # Diagnostic overlap with pre-existing price/PER/PBR/momentum inputs.
    existing = [c for c in cols["A"] if any(x in c.lower() for x in ("per", "pbr", "roe", "momentum", "ret_20"))]
    correlations = []
    for new in G.CONTROLS + G.GAPS + G.TIME:
        for old in existing:
            z = d[[new, old]].dropna()
            if len(z) >= 30 and z[new].nunique() > 1 and z[old].nunique() > 1:
                correlations.append({"new_feature": new, "existing_feature": old, "rows": len(z), "spearman": z[new].corr(z[old], method="spearman")})
    pd.DataFrame(correlations).to_csv(args.output / "feature_overlap.csv", index=False)
    result = {"status": "exploratory_completed", "outer_windows": len(folds), "adoption_status": "not_approved",
              "independent_holdout": False, "selection_sha256": digest(args.output / "selection_frozen.json")}
    json_write(args.output / "completed.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "prices", "series-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--retune", action="store_true")
    p.add_argument("--classification", action="store_true")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.trials < 2 or args.inner_folds < 1 or args.trees < 1 or args.threads < 1:
        p.error("Invalid experiment budget")
    if args.output.resolve().is_relative_to((HERE / "model").resolve()) or args.output.resolve().is_relative_to((HERE.parent / "public").resolve()):
        p.error("Experiment outputs must remain outside production model and public asset directories")
    print(json.dumps(evaluate(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
