#!/usr/bin/env python3
"""E123: current eight-model signals -> next-open upside and exit research.

Usage: python research/exp/e123_exit_upside.py --input INPUT --mart FROZEN_MART --output OUTPUT
INPUT contains oof/ and prices/ downloads from the data-raw release.
No production files are written. Holdout calculations follow selection_frozen.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_store
import exit_upside as E
import walkforward

REPO = ROOT.parent
PROTOCOL = REPO / "docs/experiments/exit-upside-2026-11-05/protocol.json"
OOF_FILES = {"lgbm": "fixed_return_v1_oof.parquet", "xgb": "fixed_return_v1_xgb_oof.parquet",
             "cat": "fixed_return_v1_cat_oof.parquet", "lgbm_k15": "fixed_return_v1_lgbm_k15_oof.parquet",
             "xgb_k15": "fixed_return_v1_xgb_k15_oof.parquet", "cat_k15": "fixed_return_v1_cat_k15_oof.parquet",
             "ltr": "ltr_oof.parquet", "ltr5": "ltr5_oof.parquet"}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    def default(v):
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return str(pd.Timestamp(v).date())
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default) + "\n")


def price_panel(inputs, codes, output):
    cache = output / "adjusted_prices.parquet"
    paths = sorted((inputs / "prices").glob("bars_*.parquet"))
    hashes = {p.name: digest(p) for p in paths}
    if cache.exists() and (output / "price_inputs.json").exists():
        check = json.loads((output / "price_inputs.json").read_text())
        if check["hashes"] == hashes and check["codes"] == sorted(codes):
            return pd.read_parquet(cache), check
        raise ValueError("Existing price cache has different inputs; use a new output directory")
    parts = []
    fields = ["Code", "Date", "O", "H", "L", "C", "AdjC", "AdjFactor"]
    for p in paths:
        if int(p.stem.split("_")[-1]) < 2018:
            continue
        d = pd.read_parquet(p, columns=fields)
        parts.append(d[d.Code.isin(codes)])
    d = pd.concat(parts, ignore_index=True).sort_values(["Code", "Date"], kind="stable").reset_index(drop=True)
    if d.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate price keys")
    d, adjustment = data_store.readjust_bars(d)
    d = d[["Code", "Date", "AdjO", "AdjH", "AdjL", "AdjC"]]
    d.to_parquet(cache, index=False)
    info = {"hashes": hashes, "codes": sorted(codes), "rows": len(d),
            "through": d.Date.max(), "adjustment": adjustment}
    dump(output / "price_inputs.json", info)
    return d, info


def prepare(args, plan):
    p = args.output
    frames, identities = {}, []
    for algo, file in OOF_FILES.items():
        meta_path = ROOT / "model" / ("meta.json" if algo == "lgbm" else f"models/{algo}/meta.json")
        m = json.loads(meta_path.read_text())
        source = args.input / "oof" / file
        actual = digest(source)
        expected = m["artifactHashes"]["oof.parquet"]
        if actual != expected:
            raise ValueError(f"OOF mismatch for {algo}")
        d = pd.read_parquet(source)
        d["Date"] = pd.to_datetime(d.Date)
        d["Code"] = d.Code.astype(str)
        if "cand_date" in d:
            d["cand_date"] = pd.to_datetime(d.cand_date)
        frames[algo] = d
        identities.append({"model": algo, "sha256": actual, "matches_production_meta": True,
                           "rows": len(d), "features": len(m["features"]), "trained_at": m["trainedAt"],
                           "train_to": m["trainTo"], "target": m.get("target", m.get("label"))})
    pd.DataFrame(identities).to_csv(p / "model_identity.csv", index=False)
    scores = E.combine_scores(frames, plan["calibration_cutoff"])
    base = pd.read_parquet(args.mart, columns=["Code", "Date"])
    base["Date"] = pd.to_datetime(base.Date)
    base["Code"] = base.Code.astype(str)
    keys = pd.concat([base, scores[["Code", "Date"]]], ignore_index=True).drop_duplicates().sort_values(["Date", "Code"]).reset_index(drop=True)
    calraw = pd.read_parquet(args.input / "prices/calendar.parquet")
    cal = pd.DatetimeIndex(pd.to_datetime(calraw.loc[calraw.HolDiv.astype(str).isin(["1", "2"]), "Date"])).sort_values().unique()
    bars, price_info = price_panel(args.input, set(keys.Code), p)
    print(f"prices {len(bars):,}, through {bars.Date.max().date()}, building paths", flush=True)
    paths = E.next_open_paths(scores, bars, cal)
    scores = pd.concat([scores, paths.rows.drop(columns=["Code", "Date"])], axis=1)
    # Conservative key superset: last frozen mart plus all exact current OOF keys.
    keys["label_mature"] = E.label_maturity(keys, bars)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    parts = []
    for lag in range(5):
        z = keys[["Code", "Date"]].copy()
        ix = z.Date.map(pos).to_numpy(int) + lag
        z = z[ix < len(cal)].copy()
        z["cand_date"] = z.Date
        z["Date"] = cal[ix[ix < len(cal)]]
        parts.append(z)
    pool = pd.concat(parts, ignore_index=True).sort_values(["Date", "Code", "cand_date"], kind="stable").drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    pool["label_mature"] = E.label_maturity(pool, bars)
    # These rows cannot have finite R20 in production (not even 20 later bars).
    pool.loc[pool.label_mature.isna(), ["Code", "Date", "cand_date"]].to_csv(p / "rolling_unresolved_keys.csv", index=False)
    folds = walkforward.make_folds(keys.Date, min_train_months=36, test_months=6, step_months=6, embargo_days=20)
    pool.loc[pool.label_mature.isna() & pool.Date.le(max(f.train_end for f in folds)),
             ["Code", "Date", "cand_date"]].to_csv(p / "rolling_untrainable_keys.csv", index=False)
    coverage = pd.DataFrame(index=pd.DatetimeIndex(scores.Date.unique()).sort_values())
    coverage.index.name = "Date"
    coverage['expected_daily'] = keys.groupby('Date').size().reindex(coverage.index, fill_value=0)
    coverage['scored_daily'] = frames['ltr'].groupby('Date').size().reindex(coverage.index, fill_value=0)
    coverage['expected_rolling'] = pool.groupby('Date').size().reindex(coverage.index, fill_value=0)
    coverage['scored_rolling'] = frames['ltr5'].groupby('Date').size().reindex(coverage.index, fill_value=0)
    coverage['daily_complete'] = coverage.expected_daily.eq(coverage.scored_daily)
    coverage['rolling_complete'] = coverage.expected_rolling.eq(coverage.scored_rolling)
    for algo, universe in [('ltr',keys),('ltr5',pool)]:
        extra = frames[algo][['Code','Date']].merge(universe[['Code','Date']], on=['Code','Date'], how='left', indicator=True)
        if extra._merge.eq('left_only').any():
            raise ValueError(f'OOF rank keys outside reconstructed universe: {algo}')
    coverage.to_csv(p / 'ranking_pool_coverage.csv')
    # Ranking OOF excludes unavailable future R20 rows. Do not interpret ranks
    # among that subset as ranks of the decision-time full candidate universe.
    scores = E.require_full_ranking_pools(scores, coverage)
    scores["fold"] = 0
    scores["fold_safe"] = False
    checks = []
    for f in folds:
        tr = keys[keys.Date.le(f.train_end)]
        rt = pool[pool.Date.le(f.train_end)]
        latest = max(tr.label_mature.max(), rt.label_mature.max())
        mask = scores.Date.between(f.test_start, f.test_end)
        scores.loc[mask, "fold"] = f.index
        scores.loc[mask, "fold_safe"] = scores.loc[mask, "Date"].gt(latest)
        checks.append({"fold": f.index, "train_end": f.train_end, "test_start": f.test_start, "test_end": f.test_end,
                       "daily_last_label": tr.label_mature.max(), "rolling_last_label": rt.label_mature.max(),
                       "daily_unknown_maturity": int(tr.label_mature.isna().sum()),
                       "rolling_unknown_maturity": int(rt.label_mature.isna().sum()),
                       "removed_boundary_rows": int((mask & ~scores.fold_safe).sum()),
                       "last_train_label": latest})
    for algo in ("ltr", "ltr5"):
        comparable = scores[f"fold_{algo}"].notna()
        if not scores.loc[comparable, "fold"].eq(scores.loc[comparable, f"fold_{algo}"]).all():
            raise ValueError(f"Reconstructed fold differs: {algo}")
    # Rolling HPO sees decision days through cutoff; include those maturity dates.
    cutoff = plan["calibration_cutoff"]
    tune_last = max(keys.loc[keys.Date.le(cutoff), "label_mature"].max(), pool.loc[pool.Date.le(cutoff), "label_mature"].max())
    scores["post_tuning"] = scores.Date.gt(tune_last)
    scores["all_eight"] = scores[list(E.SCORE_FEATURES)].notna().all(axis=1)
    # Classifier-only gates do not depend on a ranking pool's completeness.
    all_six = scores[[f'pct_{a}' for a in E.CLASSIFIERS]].notna().all(axis=1)
    scores["eligible"] = scores.fold_safe & scores.post_tuning & all_six
    # Ineligible historical rows may be meta-model fitting inputs, never test observations.
    scores["fit_reference"] = scores.fold_safe & scores.all_eight
    scores["period"] = "reference"
    scores.loc[scores.Date.between(*plan["development"]) & scores.eligible, "period"] = "development"
    scores.loc[scores.Date.between(*plan["sell_rule_holdout"]) & scores.eligible, "period"] = "holdout"
    scores.to_parquet(p / "signal_paths.parquet", index=False)
    np.savez_compressed(p / "paths.npz", o=paths.o, h=paths.h, l=paths.l, c=paths.c, dates=paths.dates)
    pd.DataFrame(checks).to_csv(p / "fold_maturity_audit.csv", index=False)
    pd.DataFrame({"Date": cal}).to_csv(p / "calendar.csv", index=False)
    audit = {"protocol_sha256": digest(PROTOCOL), "frozen_mart_sha256": digest(args.mart),
             "calendar_sha256": digest(args.input / "prices/calendar.parquet"),
             "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
             "price_through": price_info["through"], "tuning_latest_label": tune_last,
             "key_superset_rows": len(keys), "old_mart_rows": len(base),
             "new_oof_keys_not_in_old_mart": len(keys)-len(base),
             "historical_oof_rows": len(scores), "complete_paths": int(scores.status.eq("complete").sum()),
             "path_status": scores.status.value_counts().to_dict(),
             "period_rows": scores.period.value_counts().to_dict(),
             "all8_missing": int((~scores.all_eight).sum()),
             "ranking_incomplete_study_dates": int((~coverage.daily_complete | ~coverage.rolling_complete).loc[
                 coverage.index.isin(scores.loc[scores.period.isin(['development','holdout']),'Date'])].sum()),
             "identity_matches": len(identities),
             "key_audit_limitation": plan["known_limitations"][2]}
    dump(p / "data_audit.json", audit)
    print(json.dumps(audit, default=str, ensure_ascii=False), flush=True)
    return scores, paths, cal


def exit_candidates(plan):
    rules = []
    for timeout in plan["timeout_sessions"]:
        for target in plan["take_profit"]:
            for stop in plan["stop_loss"]:
                rule = {"target": target, "stop": stop, "timeout": timeout}
                rule["name"] = f"tp{round(100*target) if target else 0}_sl{round(100*stop) if stop else 0}_d{timeout}"
                rules.append(rule)
        for trail in plan["trailing_candidates"]:
            rules.append({"name": f"trail{round(100*trail['trail'])}_d{timeout}", "target": None, "stop": .08,
                          "timeout": timeout, **trail})
    return rules


def simulate_rule(paths, ids, rule, **kwargs):
    return E.simulate(paths, ids, **{k: v for k, v in rule.items() if k != "name"}, **kwargs)


def objective(trades, d, cost):
    daily = pd.DataFrame({"Date": d.loc[trades.id, "Date"].to_numpy(), "r": trades.gross.to_numpy() - cost}).groupby("Date").r.mean()
    se = daily.std(ddof=1) / np.sqrt(len(daily)) if len(daily) > 1 else np.inf
    return float(trades.gross.mean() - cost - se), float(se)


def estimate_rows(d, paths, query_ids, k):
    """Monthly expanding neighbors; all 20-day outcomes known before month start."""
    out = []
    for month, ids in d.loc[query_ids].groupby(d.loc[query_ids, "Date"].dt.to_period("M")).groups.items():
        boundary = month.start_time
        fit = d[d.fit_reference & d.status.eq("complete") & d.base_min.ge(95) & d.mature_date.lt(boundary)]
        if len(fit) < 30:
            continue
        fit_ids = fit.index.to_numpy()
        x = fit[list(E.SCORE_FEATURES)].to_numpy(float) / 100
        for i in ids:
            q = d.loc[i, list(E.SCORE_FEATURES)].to_numpy(float) / 100
            distances = ((x - q) ** 2).sum(axis=1)
            order = np.argsort(distances, kind="stable")[:min(k, len(fit))]
            neighbors = fit_ids[order]
            z = d.loc[neighbors]
            curve = paths.c[neighbors].mean(axis=0)
            bcurve = paths.c[fit_ids].mean(axis=0)
            day = int(np.argmax(curve)) + 1
            baseline_day = int(np.argmax(bcurve)) + 1
            rec = {"id": i, "Date": d.at[i, "Date"], "Code": d.at[i, "Code"], "k": k,
                   "n_fit": len(fit), "neighbors": len(neighbors), "fit_latest_maturity": fit.mature_date.max(),
                   "expected_mfe": z.mfe.mean(), "median_mfe": z.mfe.median(), "q25_mfe": z.mfe.quantile(.25),
                   "q75_mfe": z.mfe.quantile(.75), "baseline_mfe": fit.mfe.mean(), "actual_mfe": d.at[i, "mfe"],
                   "suggested_close_day": day, "baseline_close_day": baseline_day,
                   "predicted_close_return": curve[day-1], "actual_close_return": paths.c[i, day-1],
                   "baseline_close_return": paths.c[i, baseline_day-1], "actual_close20": paths.c[i, -1]}
            for target in (5, 8):
                rec[f"p{target}"] = z[f"hit{target}"].mean()
                rec[f"baseline_p{target}"] = fit[f"hit{target}"].mean()
                hitdays = z[f"hit{target}_day"].dropna()
                rec[f"median_hit{target}_day_if_hit"] = hitdays.median() if len(hitdays) else np.nan
                for horizon in (5, 10, 20):
                    rec[f"p{target}_by{horizon}"] = z[f"hit{target}_day"].le(horizon).mean()
            out.append(rec)
    return pd.DataFrame(out)


def develop(d, paths, cal, plan, p):
    cost = plan["cost_ratio_primary"]
    dev = d[d.period.eq("development")]
    rules = exit_candidates(plan)
    rows = []
    for spec in plan["entry_candidates"]:
        ids = E.selected_ids(dev, spec)
        complete_ids = [i for i in ids if d.at[i, "status"] == "complete"]
        dates = d.loc[complete_ids, "Date"].nunique()
        for rule in rules:
            trades = simulate_rule(paths, ids, rule)
            if len(trades):
                obj, se = objective(trades, d, cost)
                rows.append({"entry": spec["name"], "exit": rule["name"], "n_dates": dates,
                             "eligible_selection": len(trades) >= plan["min_development_trades"] and dates >= plan["min_development_signal_dates"],
                             "selection_objective_pct": 100*obj, "date_se_pct": 100*se, **E.trade_summary(trades, cost)})
    results = pd.DataFrame(rows)
    results.to_csv(p / "development_search.csv", index=False)
    choices = results[results.eligible_selection].sort_values(["selection_objective_pct", "entry", "exit"], ascending=[False, True, True])
    choice = choices.iloc[0] if len(choices) else None
    chosen_entry = next(x for x in plan["entry_candidates"] if x["name"] == (choice.entry if choice is not None else "base95"))
    chosen_exit = next(x for x in rules if x["name"] == choice.exit) if choice is not None else {"name": "reference", **plan["reference_exit"]}
    # Tune estimator only in development, using all high-base95 daily top2 queries.
    query_ids = E.selected_ids(dev, {"base": 95})
    query_ids = np.array([i for i in query_ids if d.at[i, "status"] == "complete" and d.at[i,'all_eight']], dtype=int)
    forecasts = []
    for k in (50, 100, 200):
        f = estimate_rows(d, paths, query_ids, k)
        if len(f):
            forecasts.append(f)
    fc = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    if len(fc):
        fc.to_csv(p / "development_forecasts.csv", index=False)
        perf = fc.groupby("k").apply(lambda g: pd.Series({"n": len(g), "mfe_rmse_pct": 100*np.sqrt(np.mean((g.expected_mfe-g.actual_mfe)**2)),
                                                          "baseline_rmse_pct": 100*np.sqrt(np.mean((g.baseline_mfe-g.actual_mfe)**2))}), include_groups=False).reset_index()
        perf.to_csv(p / "estimator_tuning.csv", index=False)
        k_best = int(perf.sort_values(["mfe_rmse_pct", "k"]).iloc[0].k)
    else:
        k_best = 100
    selection = {"entry": chosen_entry, "exit": chosen_exit, "neighbor_k": k_best,
                 "development_candidates_evaluated": len(results), "distinct_exit_rules": len(rules),
                 "selection_reason": "development objective" if choice is not None else "insufficient development sample; reference fallback",
                 "protocol_sha256": digest(PROTOCOL), "holdout_metrics_computed": False}
    dump(p / "selection_frozen.json", selection)
    print("FROZEN " + json.dumps(selection, ensure_ascii=False), flush=True)
    return selection


def opportunity_stats(d, ids):
    z = d.loc[ids]
    v = z[z.status.eq("complete")]
    row = {"signals": len(z), "n": len(v), "n_dates": v.Date.nunique(), "n_codes": v.Code.nunique(),
           "excluded_paths": len(z)-len(v)}
    if not len(v):
        return row
    row.update({"mfe_mean_pct": 100*v.mfe.mean(), "mfe_median_pct": 100*v.mfe.median(),
                "mfe_q25_pct": 100*v.mfe.quantile(.25), "mfe_q75_pct": 100*v.mfe.quantile(.75),
                "mae_median_pct": 100*v.mae.median(), "close20_mean_pct": 100*v.ret_close20.mean(),
                "peak_day_median": v.peak_day.median()})
    for t in (3, 5, 8, 10, 15):
        row[f"hit{t}_pct"] = 100*v[f"hit{t}"].mean()
        hitdays = v[f"hit{t}_day"].dropna()
        row[f"hit{t}_day_median"] = hitdays.median() if len(hitdays) else np.nan
        row[f"hit{t}_before5_pct"] = 100*v[f"hit{t}_day"].le(5).mean()
        row[f"hit{t}_before10_pct"] = 100*v[f"hit{t}_day"].le(10).mean()
    return row


def three_slot_subset(trades, d, slots=3):
    """Diagnostic only: same-code skip, no reuse of an intraday exit's slot at that day's open."""
    active = []
    chosen = []
    for row in trades.itertuples(index=False):
        entry_date = d.at[row.id, "entry_date"]
        active = [(code, end) for code, end in active if end >= entry_date]
        code = d.at[row.id, "Code"]
        if len(active) >= slots or any(c == code for c, _ in active):
            continue
        chosen.append(row.id)
        active.append((code, pd.Timestamp(row.exit_date)))
    return trades[trades.id.isin(chosen)]


def evaluate(d, paths, cal, plan, selected, p):
    cost = plan["cost_ratio_primary"]
    # No entry or rule re-selection below this point.
    opportunities, exitrows, thresholds = [], [], []
    exploratory = []
    for period in ("development", "holdout"):
        sub = d[d.period.eq(period)]
        for spec in plan["entry_candidates"]:
            ids = E.selected_ids(sub, spec)
            opp = {"period": period, "entry": spec["name"], **opportunity_stats(d, ids)}
            if period == "holdout":
                good = [i for i in ids if d.at[i, "status"] == "complete"]
                for t in (5, 8):
                    lo, hi = E.block_interval(d.loc[good, f"hit{t}"], d.loc[good, "Date"], cal) if len(good) >= 30 and d.loc[good,"Date"].nunique() >= 10 else (None,None)
                    opp[f"hit{t}_ci_low_pct"], opp[f"hit{t}_ci_high_pct"] = (100*lo, 100*hi) if lo is not None else (None, None)
            opportunities.append(opp)
            trades = E.simulate(paths, ids, **plan["reference_exit"])
            thresholds.append({"period": period, "entry": spec["name"], **E.trade_summary(trades, cost)})
        ids = E.selected_ids(sub, selected["entry"])
        fixed_rules = [{"name": "current_tp10_day20", **plan["reference_exit"]},
                       {"name": "tp5_day20", "target": .05, "timeout": 20},
                       {"name": "tp8_day20", "target": .08, "timeout": 20},
                       {"name": "close_day20", "timeout": 20},
                       {**selected["exit"], "name": "development_selected"}]
        for rule in fixed_rules:
            for order in ("stop_first", "target_first"):
                trades = simulate_rule(paths, ids, rule, ambiguity=order)
                for charge in plan["cost_sensitivity"]:
                    row = {"period": period, "entry": selected["entry"]["name"], "exit": rule["name"],
                           "cost_pct": 100*charge, "order": order, **E.trade_summary(trades, charge)}
                    if period == "holdout" and charge == cost and order == "stop_first" and len(trades):
                        lo, hi = E.block_interval(trades.gross.to_numpy()-charge, d.loc[trades.id,"Date"], cal)
                        row.update(ci_low_pct=100*lo, ci_high_pct=100*hi)
                        trades.merge(d[["Code", "Date", "entry_date", "mature_date", "base_min", "mfe", "mae", "hit5", "hit8"]], left_on="id", right_index=True).to_csv(p / f"holdout_trades_{rule['name']}.csv", index=False)
                    exitrows.append(row)
                if period == "holdout" and order == "stop_first":
                    limited = three_slot_subset(trades, d)
                    exitrows.append({"period": "holdout_3slots", "entry": selected["entry"]["name"], "exit": rule["name"],
                                     "cost_pct": 100*cost, "order": order, **E.trade_summary(limited, cost)})
        if period == "holdout":
            # The grid was declared before reading outcomes. Its holdout values
            # are descriptive only; no winner is selected from this table.
            for rule in exit_candidates(plan):
                lo = simulate_rule(paths, ids, rule)
                hi = simulate_rule(paths, ids, rule, ambiguity="target_first")
                exploratory.append({"exit": rule["name"], **{k:v for k,v in rule.items() if k != "name"},
                                    **E.trade_summary(lo, cost),
                                    "target_first_net_pct": E.trade_summary(hi, cost).get("mean_net_pct")})
        # Locked-estimator forecasts for both base95 and the selected entry.
        q = E.selected_ids(sub, {"base": 95})
        q = np.array([i for i in q if d.at[i, "status"] == "complete" and d.at[i,'all_eight']], dtype=int)
        f = estimate_rows(d, paths, q, selected["neighbor_k"])
        if len(f):
            f["selected_entry"] = f.id.isin(ids)
            f.to_csv(p / f"{period}_forecasts.csv", index=False)
    pd.DataFrame(opportunities).to_csv(p / "opportunity_by_threshold.csv", index=False)
    pd.DataFrame(thresholds).to_csv(p / "reference_exit_by_threshold.csv", index=False)
    pd.DataFrame(exitrows).to_csv(p / "exit_comparison.csv", index=False)
    pd.DataFrame(exploratory).to_csv(p / "holdout_exit_grid_exploratory.csv", index=False)
    current = pd.read_csv(p / "holdout_trades_current_tp10_day20.csv")
    comparisons = []
    for name in ("tp5_day20", "tp8_day20", "close_day20"):
        alt = pd.read_csv(p / f"holdout_trades_{name}.csv")
        pair = alt[["id","gross"]].merge(current[["id","gross","Date"]], on="id", suffixes=("_alt","_current"), validate="one_to_one")
        diff = pair.gross_alt-pair.gross_current
        lo, hi = E.block_interval(diff, pair.Date, cal)
        comparisons.append({"exit":name,"n":len(pair),"vs_tp10_mean_diff_pct":100*diff.mean(),
                            "ci_low_pct":100*lo,"ci_high_pct":100*hi})
    pd.DataFrame(comparisons).to_csv(p / "paired_exit_comparison.csv", index=False)
    # Quarter slices and expected-vs-observed diagnostics, never used to reselect.
    hp = p / "holdout_forecasts.csv"
    if hp.exists():
        f = pd.read_csv(hp)
        modelrows = []
        for label, z in [("base95", f), ("selected_entry", f[f.selected_entry])]:
            if not len(z):
                continue
            n = {"cohort": label, "n": len(z), "k": selected["neighbor_k"],
                 "mfe_rmse_pct": 100*np.sqrt(np.mean((z.expected_mfe-z.actual_mfe)**2)),
                 "baseline_rmse_pct": 100*np.sqrt(np.mean((z.baseline_mfe-z.actual_mfe)**2)),
                 "forecast_mean_mfe_pct": 100*z.expected_mfe.mean(), "realized_mean_mfe_pct": 100*z.actual_mfe.mean(),
                 "mean_suggested_day": z.suggested_close_day.mean(),
                 "forecast_day_net_pct": 100*(z.actual_close_return.mean()-cost),
                 "baseline_day_net_pct": 100*(z.baseline_close_return.mean()-cost),
                 "fixed20_net_pct": 100*(z.actual_close20.mean()-cost)}
            for t in (5, 8):
                actual = d.loc[z.id, f"hit{t}"].to_numpy()
                n[f"brier{t}"] = np.mean((z[f"p{t}"].to_numpy()-actual)**2)
                n[f"baseline_brier{t}"] = np.mean((z[f"baseline_p{t}"].to_numpy()-actual)**2)
            modelrows.append(n)
        pd.DataFrame(modelrows).to_csv(p / "upside_estimator_evaluation.csv", index=False)
        calibration = []
        for t in (5, 8):
            actual = d.loc[f.id, f"hit{t}"].to_numpy()
            for low, high in [(0,.5),(.5,.7),(.7,1.01)]:
                mask = f[f"p{t}"].ge(low)&f[f"p{t}"].lt(high)
                if mask.any():
                    calibration.append({"target":t,"probability_band":f"{low:.0%}–{min(high,1):.0%}",
                                        "n":int(mask.sum()),"predicted_pct":100*f.loc[mask,f"p{t}"].mean(),
                                        "actual_pct":100*actual[mask.to_numpy()].mean()})
        pd.DataFrame(calibration).to_csv(p / "hit_probability_calibration.csv", index=False)
    all_trades = pd.read_csv(p / "holdout_trades_development_selected.csv", parse_dates=["Date"])
    all_trades["quarter"] = all_trades.Date.dt.to_period("Q").astype(str)
    quarters = []
    for quarter, group in all_trades.groupby("quarter"):
        quarters.append({"quarter": quarter, **E.trade_summary(group, cost)})
    pd.DataFrame(quarters).to_csv(p / "holdout_quarters.csv", index=False)
    print("Evaluation complete", flush=True)


def live_snapshot(d, paths, selected, plan, output):
    """Illustrative estimates for today's actual signal, never trade instructions."""
    source = REPO / "public/data/predictions.json"
    predictions = json.loads(source.read_text())
    asof = pd.Timestamp(predictions["asOf"])
    # Preserve the exact input snapshot privately with the result bundle.
    (output / "prediction_snapshot.json").write_text(source.read_text())
    rows = []
    for c in predictions["candidates"]:
        if pd.Timestamp(c["date"]) != asof:
            continue
        rec = {"Code":c["jqCode"], "Date":asof, "name":c["name"], "signal_close":c["close"],
               "model_ltr": c.get("ltr",{}), "model_ltr5": c.get("ltr5",{})}
        for a in E.CLASSIFIERS:
            rec[f"score_{a}"] = c.get("byModel",{}).get(a,{}).get("score", np.nan)
            ref = d.loc[d.Date.le(plan["calibration_cutoff"]),f"score_{a}"]
            rec[f"pct_{a}"] = E.historical_percentile([rec[f"score_{a}"]],ref)[0]
        rec["pct_ltr"] = rec["model_ltr"].get("dayPct", np.nan)
        rec["pct_ltr5"] = rec["model_ltr5"].get("pct", np.nan)
        rec.pop("model_ltr");rec.pop("model_ltr5")
        rows.append(rec)
    today = pd.DataFrame(rows)
    if not len(today):
        return
    today["base_min"] = today[[f"pct_{a}" for a in E.CLASSIFIERS[:3]]].min(axis=1,skipna=False)
    today["six_min"] = today[[f"pct_{a}" for a in E.CLASSIFIERS]].min(axis=1,skipna=False)
    today["high_signal"] = E.entry_mask(today,selected["entry"])
    today.to_csv(output / "live_signal_screen.csv",index=False)
    # Same monthly update convention as retrospective evaluation.
    boundary = asof.to_period("M").start_time
    fit = d[d.fit_reference & d.status.eq("complete") & d.base_min.ge(95) & d.mature_date.lt(boundary)]
    recs = []
    x = fit[list(E.SCORE_FEATURES)].to_numpy(float)/100
    for i in E.selected_ids(today,selected["entry"]):
        r = today.loc[i]
        if not np.isfinite(r[list(E.SCORE_FEATURES)].to_numpy(float)).all() or len(fit) < 30:
            continue
        dist = ((x-r[list(E.SCORE_FEATURES)].to_numpy(float)/100)**2).sum(axis=1)
        ni = fit.index.to_numpy()[np.argsort(dist,kind="stable")[:selected["neighbor_k"]]]
        z = d.loc[ni]
        curve = paths.c[ni].mean(axis=0)
        recs.append({"asof":asof,"code":r.Code,"name":r['name'],"base_min_pct":r.base_min,
                     "n_neighbors":len(z),"fit_latest_maturity":fit.mature_date.max(),
                     "expected_mfe_pct":100*z.mfe.mean(),"median_mfe_pct":100*z.mfe.median(),
                     "mfe_q25_pct":100*z.mfe.quantile(.25),"mfe_q75_pct":100*z.mfe.quantile(.75),
                     "p5_pct":100*z.hit5.mean(),"p8_pct":100*z.hit8.mean(),
                     "suggested_close_day":int(np.argmax(curve))+1,
                     "predicted_close_return_pct":100*curve.max()})
    pd.DataFrame(recs,columns=["asof","code","name","base_min_pct","n_neighbors","fit_latest_maturity",
                              "expected_mfe_pct","median_mfe_pct","mfe_q25_pct","mfe_q75_pct","p5_pct","p8_pct",
                              "suggested_close_day","predicted_close_return_pct"]).to_csv(output / "live_forecasts.csv",index=False)
    dump(output / "live_input_identity.json",{"asof":asof,"prediction_sha256":digest(source),
                                              "high_signal_n":int(today.high_signal.sum()),"shown_top2_n":len(recs),
                                              "note":"Provisional research estimator; raw scores of full fitted models may drift from historical OOF distributions."})


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--mart", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PROTOCOL.read_text())
    (args.output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    d, paths, cal = prepare(args, plan)
    selected = develop(d, paths, cal, plan, args.output)
    evaluate(d, paths, cal, plan, selected, args.output)
    live_snapshot(d, paths, selected, plan, args.output)


if __name__ == "__main__":
    main()
