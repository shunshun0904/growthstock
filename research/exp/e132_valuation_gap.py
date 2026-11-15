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
import time
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from importlib.metadata import version
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

ARMS = ("A", "B", "C", "D", "R", "P", "M")
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


def prepare(dataset, series_dir, prices, output, feature_mart=None):
    original = pd.read_parquet(dataset)
    missing = set(F.columns(F.DEFAULT_PRESET)) - set(original)
    if missing:
        raise ValueError("Current baseline features missing: " + str(sorted(missing)))
    parts, codes = [], set()
    fields = list(dict.fromkeys(G.CONTROLS + G.GAPS + G.TIME + G.MARKET_CONTROLS + G.RIM_FIELDS))
    mart_complete = False
    if feature_mart:
        mart_status = json.loads((feature_mart/'completed.json').read_text())
        inputs = json.loads((feature_mart/'inputs.json').read_text())
        if mart_status.get('status')!='completed' or not mart_status.get('all_original_keys_attempted') or inputs['data'].get(dataset.name)!=digest(dataset):
            raise ValueError('Historical feature mart is incomplete or belongs to another training dataset')
        for p in sorted(feature_mart.glob('features-*.parquet')):
            tag=json.loads(p.with_suffix('.meta.json').read_text())
            if tag.get('fingerprint')!=mart_status['fingerprint'] or tag.get('sha256')!=digest(p):
                raise ValueError('Historical feature chunk provenance mismatch')
            f=pd.read_parquet(p)
            parts.append(f.reindex(columns=['Code','Date',*fields]))
            codes.update(f.Code.map(code5))
        mart_complete=True
    for p in ([] if feature_mart else sorted(series_dir.glob("*.parquet"))):
        f = pd.read_parquet(p)
        # Verify these are NEW independent-EVA series, not legacy CAPM files.
        if "strict_fundamental_value_per_share" not in f or "rim_value_per_share" not in f or "independent_shares" not in f:
            raise ValueError("Series is not from the cash-unsegregated EVA / external-Ke RIM pipeline: " + p.name)
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
    if mart_complete:
        expected=key[['Code','Date']].drop_duplicates().sort_values(['Code','Date']).reset_index(drop=True)
        actual=features[['Code','Date']].sort_values(['Code','Date']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected)
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
    if not d.rim_valuation_gap.notna().any():
        status='blocked'
        reasons.append('no_observed_rim_valuation_gap')
    report = {"status": status, "reasons": reasons, "rows": len(d), "codes": int(original.Code.nunique()),
              "date_min": str(original.Date.min().date()), "date_max": str(original.Date.max().date()),
              "covered_codes": len(set(original.Code.map(code5)) & codes), "missing_code_count": len(absent),
              "gap_available_rows": int(d.valuation_gap.notna().sum()), "original_columns_identical": True,
              'rim_gap_available_rows':int(d.rim_valuation_gap.notna().sum()),
              'common_gap_available_rows':int((d.valuation_gap.notna()&d.rim_valuation_gap.notna()).sum()),
              'all_historical_keys_attempted':mart_complete,
              'missing_price_history_codes':mart_status.get('missing_price_history_codes') if feature_mart else None,
              "production_preset": F.DEFAULT_PRESET, "production_feature_count": len(F.columns(F.DEFAULT_PRESET)),
              "adoption_status": "not_evaluated", "backtest_completed": False,
              "limitations": ["E120/E121 periods already observed; no independent holdout claim",
                              "Historical API revisions not fully archived", "Current inference selection cannot define historical universe"]}
    json_write(output / "preflight.json", report)
    return d, report


def retune_study(task,outer,arm,train,columns,bank,output,source_fingerprint,inner_folds,threads,trees,outer_start):
    """One isolated CV study; never receives outer evaluation indices/scores."""
    if not train.maturity_date.lt(outer_start).all() or not train.Date.lt(outer_start).all():
        raise ValueError('Immature or outer evaluation rows entered a tuning study')
    inner,_=RETUNE.forward_folds(train,task,n_splits=inner_folds,min_train_months=12)
    records=[]
    path=output/f'cv-{task}-{outer}-{arm}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('fingerprint')!=source_fingerprint:
            raise ValueError('CV cache belongs to different inputs')
        records=saved['records']
        if [r['candidate'] for r in records]!=list(range(len(records))) or len(records)>len(bank):
            raise ValueError('Incomplete or invalid CV candidate sequence')
    trainers=[(RETUNE.Trainer(train.iloc[it],columns,task,threads),train.iloc[iv],
               train.iloc[iv][columns].to_numpy(float)) for it,iv in inner] if len(records)<len(bank) else []
    for candidate_id in range(len(records),len(bank)):
        start=time.monotonic()
        scores=[]
        for trainer,valid,xvalid in trainers:
            model=trainer.fit(bank[candidate_id],trees=trees)
            prediction=model.predict(xvalid,num_threads=threads)
            scores.append(RETUNE.primary(task,valid,prediction))
        records.append({'candidate':candidate_id,'cv_mean':float(np.mean(scores)),
                        'inner_scores':scores,'seconds':round(time.monotonic()-start,3)})
        json_write(path,{'fingerprint':source_fingerprint,'records':records})
        print(json.dumps({'phase':'retune','task':task,'outer':outer,'arm':arm,
                          'candidate':candidate_id+1,'budget':len(bank)}),flush=True)
    return {'task':task,'outer':outer,'arm':arm,'parameters':bank[RETUNE.select_candidate(records)['candidate']]}


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run and (args.output/'completed.json').exists():
        raise ValueError('Completed evaluation exists: choose a fresh output directory for a new run')
    source = {"dataset": digest(args.dataset), "prices": digest(args.prices),
              'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                         'lightgbm':version('lightgbm'),'optuna':version('optuna'),'scikit-learn':version('scikit-learn')},
              "series": {p.name: digest(p) for p in sorted((args.feature_mart or args.series_dir).glob("*.parquet"))},
              "source": {p.name: digest(p) for p in [Path(__file__), HERE / "fundamental_value.py", HERE / "fundamental_series.py",
                           HERE / "valuation_gap.py", HERE / "features.py",HERE/'ltr.py',HERE/'tuning.py',
                           HERE/'exp/e119_capital_efficiency.py',HERE/'exp/e121_theoretical_value_retune.py']}}
    old_inputs=args.output/'inputs.json'
    if old_inputs.exists() and json.loads(old_inputs.read_text())!=source:
        raise ValueError('Experiment inputs changed: choose a fresh output directory')
    json_write(args.output / "inputs.json", source)
    source_fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    d, preflight = prepare(args.dataset, args.series_dir, args.prices, args.output,args.feature_mart)
    cols = G.column_sets()
    plan = {"arms": {k: v for k, v in cols.items()}, "fixed_parameters": FIXED,
            "retune": args.retune, "trials_per_arm_fold": args.trials, "inner_folds": args.inner_folds,
            "outer_train_months": 36, "outer_test_months": 6, "trees": args.trees, "seeds": SEEDS,
            'last_outer_folds':args.last_folds,
            'threads_per_fit':args.threads,
            "shuffle": "joint permutation within date and feature-availability pattern",
            "selection": "all arm/fold selections frozen before evaluating outer predictions",
            "automatic_adoption": False, "independent_holdout": False}
    old_plan=args.output/'plan.json'
    if old_plan.exists() and json.loads(old_plan.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('Experiment plan changed: choose a fresh output directory')
    json_write(old_plan, plan)
    if preflight["status"] != "ready" or not args.run:
        return preflight
    d = d[d[ltr.R20].notna() & d.maturity_date.notna() & d.label.notna()].sort_values(["Date", "Code"]).reset_index(drop=True)
    folds = walk_forward(d)
    if args.last_folds:
        folds=folds[-args.last_folds:]
    if not folds:
        raise ValueError("No eligible walk-forward windows")
    shuffled = G.shuffle_within_date(d, list(dict.fromkeys(
        ["roic_conservative_tax30", "roic_operating_tax30", "wacc", "actual_tax"] + G.GAPS + G.TIME)))
    tasks = ["ltr", "class10", "class15"] if args.classification else ["ltr"]
    bank = RETUNE.candidate_bank(FIXED, args.trials) if args.retune else [FIXED]
    selected = []
    # Stage 1: only INNER validation outcomes, for EVERY arm/task/window.
    def study_inputs():
        for task in tasks:
            for outer,(ti,vi) in enumerate(folds):
                for arm in ARMS:
                    frame=shuffled if arm=='P' else d
                    needed=list(dict.fromkeys(cols[arm]+['Date',ltr.R20,'maturity_date']+
                        (['y_'+task] if task.startswith('class') else [])))
                    yield dict(task=task,outer=outer,arm=arm,
                        train=frame.iloc[ti][needed].reset_index(drop=True) if args.retune else None,
                        columns=cols[arm],bank=bank,output=args.output,source_fingerprint=source_fingerprint,
                        inner_folds=args.inner_folds,threads=args.threads,trees=args.trees,outer_start=d.iloc[vi].Date.min())
    studies=iter(study_inputs())
    if not args.retune:
        selected=[{'task':s['task'],'outer':s['outer'],'arm':s['arm'],'parameters':FIXED} for s in studies]
    elif args.jobs==1:
        selected=[retune_study(**study) for study in studies]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            # Bound pending training frames: three tasks × all outer windows
            # must not retain hundreds of dense DataFrame copies in memory.
            pending=set()
            for _ in range(args.jobs):
                study=next(studies,None)
                if study is not None: pending.add(pool.submit(retune_study,**study))
            try:
                while pending:
                    result=next(as_completed(pending))
                    pending.remove(result)
                    selected.append(result.result())
                    study=next(studies,None)
                    if study is not None: pending.add(pool.submit(retune_study,**study))
            except BaseException:
                for result in pending: result.cancel()
                raise
        selected.sort(key=lambda s:(tasks.index(s['task']),s['outer'],ARMS.index(s['arm'])))
    json_write(args.output / "selection_frozen.json", selected)
    print(json.dumps({'phase':'all_selections_frozen','studies':len(selected)}),flush=True)
    del studies
    # Stage 2: outer evaluation can start only after the complete freeze file.
    results, importance = [], []
    for spec in selected:
        task, outer, arm = spec["task"], spec["outer"], spec["arm"]
        ti, vi = folds[outer]
        frame = shuffled if arm == "P" else d
        train, test = frame.iloc[ti], frame.iloc[vi]
        predictions = []
        trainer = RETUNE.Trainer(train, cols[arm], task, args.threads)
        for seed in SEEDS:
            model = trainer.fit(spec["parameters"], seed, args.trees)
            predictions.append(model.predict(test[cols[arm]].to_numpy(float)))
            for name, gain in zip(cols[arm], model.feature_importance(importance_type="gain")):
                importance.append({"task": task, "outer": outer, "arm": arm, "seed": seed, "feature": name, "gain": float(gain)})
        score = np.mean(predictions, axis=0)
        metric = ranking_metrics(test, score) if task == "ltr" else E.evaluate(task, test, score)
        results.append({**spec, "parameters": json.dumps(spec["parameters"], sort_keys=True), "cohort": "identical_all_rows", **metric})
        complete = (test.valuation_gap.notna()&test.rim_valuation_gap.notna()).to_numpy()
        if complete.any():
            metric = ranking_metrics(test.loc[complete], score[complete]) if task == "ltr" else E.evaluate(task, test.loc[complete], score[complete])
            results.append({**spec, "parameters": json.dumps(spec["parameters"], sort_keys=True), "cohort": "identical_valued_rows", **metric})
        pd.DataFrame({"Code": test.Code, "Date": test.Date, "score": score, "return_20d": test[ltr.R20]}).to_parquet(
            args.output / f"predictions-{task}-{outer}-{arm}.parquet", index=False)
        print(json.dumps({'phase':'outer_evaluation','task':task,'outer':outer,'arm':arm,'rows':len(test)}),flush=True)
    pd.DataFrame(results).to_csv(args.output / "results.csv", index=False)
    pd.DataFrame(importance).to_csv(args.output / "feature_importance.csv", index=False)
    # Diagnostic overlap with pre-existing price/PER/PBR/momentum inputs.
    existing = [c for c in cols["A"] if any(x in c.lower() for x in ("per", "pbr", "roe", "momentum", "ret_20"))]
    correlations = []
    for new in G.CONTROLS + G.GAPS + G.TIME + G.RIM_FIELDS:
        for old in existing:
            z = d[[new, old]].dropna()
            if len(z) >= 30 and z[new].nunique() > 1 and z[old].nunique() > 1:
                correlations.append({"new_feature": new, "existing_feature": old, "rows": len(z), "spearman": z[new].corr(z[old], method="spearman")})
    pd.DataFrame(correlations).to_csv(args.output / "feature_overlap.csv", index=False)
    direct=[]
    for outer,(_,vi) in enumerate(folds):
        test=d.iloc[vi]
        common=test.valuation_gap.notna()&test.rim_valuation_gap.notna()
        test=test.loc[common]
        for method,gap in [('cash_unsegregated_EVA','valuation_gap'),('external_Ke_RIM','rim_valuation_gap')]:
            direct.append({'outer':outer,'method':method,'rows':len(test),**ranking_metrics(test,-test[gap].to_numpy())})
    pd.DataFrame(direct).to_csv(args.output/'direct_gap_comparison.csv',index=False)
    result = {"status": "exploratory_completed", "outer_windows": len(folds), "adoption_status": "not_approved",
              'backtest_completed':True,'arms':list(ARMS),'trials_per_study':len(bank),
              'logical_cv_fits':len(selected)*len(bank)*args.inner_folds if args.retune else 0,
              'outer_fits':len(selected)*len(SEEDS),
              'evaluation_windows':[{'from':str(d.iloc[vi].Date.min().date()),'to':str(d.iloc[vi].Date.max().date())} for _,vi in folds],
              "independent_holdout": False, "selection_sha256": digest(args.output / "selection_frozen.json")}
    json_write(args.output / "completed.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "prices", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument('--series-dir',type=Path)
    p.add_argument('--feature-mart',type=Path,help='Complete original-key feature mart; no future-selected security subset')
    p.add_argument("--run", action="store_true")
    p.add_argument("--retune", action="store_true")
    p.add_argument("--classification", action="store_true")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument('--last-folds',type=int,default=0,help='Use a predeclared number of latest outer folds; 0 means all')
    p.add_argument('--jobs',type=int,default=1,help='1/2 independent CV studies; outer evaluation starts after ALL complete')
    args = p.parse_args()
    if args.trials < 2 or args.inner_folds < 1 or args.trees < 1 or args.threads < 1 or args.last_folds<0 or args.jobs not in (1,2):
        p.error("Invalid experiment budget")
    if bool(args.series_dir)==bool(args.feature_mart):
        p.error('Supply exactly one of --series-dir / --feature-mart')
    if args.output.resolve().is_relative_to((HERE / "model").resolve()) or args.output.resolve().is_relative_to((HERE.parent / "public").resolve()):
        p.error("Experiment outputs must remain outside production model and public asset directories")
    print(json.dumps(evaluate(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
