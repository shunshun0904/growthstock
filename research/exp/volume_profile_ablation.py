#!/usr/bin/env python3
"""Compare LightGBM with and without the private intraday feature mart."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as F
import label_specs as LS
import ltr
from e119_capital_efficiency import evaluate, forward_maturity
from e121_theoretical_value_retune import Trainer, candidate_bank, forward_folds, primary, select_candidate
from volume_profile import feature_columns, normalize_code
from volume_profile_data import file_sha256, private_dir

SEEDS = (42, 43, 44)
FIXED = dict(learning_rate=.05, num_leaves=31, min_child_samples=20,
             subsample=.8, colsample_bytree=.8, reg_alpha=0., reg_lambda=1.)


def key_index(frame):
    return pd.MultiIndex.from_arrays([frame.Code.map(normalize_code), pd.to_datetime(frame.Date)])


def attach(base, derived):
    """An exact left join keeps the original samples, labels, and row order."""
    cols = feature_columns("tick") + feature_columns("minute")
    if set(cols) & set(base.columns):
        raise ValueError("元マートに同名の出来高特徴量があります。")
    keys = key_index(derived)
    if keys.has_duplicates or key_index(base).has_duplicates:
        raise ValueError("銘柄と日付のキーが重複しています。")
    values = derived[cols].set_axis(keys).reindex(key_index(base)).astype(float)
    return pd.concat([base, values.set_axis(base.index)], axis=1)


def load_bars(path):
    adjusted = {"open": "AdjO", "high": "AdjH", "close": "AdjC"}
    names = set(pq.ParquetFile(path).schema_arrow.names)
    mapping = {"Code": "Code", "Date": "Date"}
    mapping.update({source if target not in names else target: target for source, target in adjusted.items()})
    bars = pd.read_parquet(path, columns=list(mapping)).rename(columns=mapping)
    bars = bars.assign(Code=bars.Code.map(normalize_code), Date=pd.to_datetime(bars.Date))
    if bars.duplicated(["Code", "Date"]).any():
        raise ValueError("収益計算用日足の銘柄と日付が重複しています。")
    return bars


def checked_folds(frame, task):
    """Full three-month evaluation windows after a one-year warm-up."""
    first, last = frame.Date.min(), frame.Date.max()
    start = first + pd.DateOffset(months=12)
    folds = []
    while start + pd.DateOffset(months=3) - pd.Timedelta(days=7) <= last:
        end = start + pd.DateOffset(months=3)
        test = frame.loc[frame.Date.ge(start) & frame.Date.lt(end)]
        if len(test):
            cutoff = test.Date.min()
            train = frame.loc[frame.Date.lt(cutoff) & frame.maturity_date.lt(cutoff)]
            try:
                inner, checks = forward_folds(train, task, n_splits=3, months=2, min_train_months=3)
            except ValueError as exc:
                raise ValueError(f"{start.date()}の内側検証が成立しません: {exc}") from exc
            if task == "class10" and test.y_class10.nunique() != 2:
                raise ValueError(f"{start.date()}の評価期間に両クラスがありません。")
            if task == "ltr" and not test.groupby("Date").size().ge(5).any():
                raise ValueError(f"{start.date()}の評価期間に日付内5銘柄以上の日がありません。")
            folds.append((train, test, inner, checks))
        start = end
    if not folds:
        raise ValueError("両ソースの20営業日集計が揃う期間で、訓練12か月・評価3か月を確保できません。")
    return folds


def run(args):
    summary = json.loads(args.features.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if summary.get("demo"):
        raise ValueError("合成データは実データの比較には使えません。")
    if summary.get("schemaVersion") != 1 or summary.get("sha256") != file_sha256(args.features):
        raise ValueError("特徴量と集計記録の組み合わせが一致しません。")
    base = pd.read_parquet(args.dataset)
    joined = attach(base, pd.read_parquet(args.features))
    base_cols = F.columns(F.DEFAULT_PRESET)
    tick = feature_columns("tick")
    minute = feature_columns("minute")
    arms = {"baseline": base_cols, "minute": base_cols + minute,
            "tick": base_cols + tick, "both": base_cols + minute + tick}
    output = private_dir(args.output)
    if args.run and (output / "walk-forward.csv").exists():
        raise ValueError("比較済みの出力があります。別の --output を指定してください。")
    joined.to_parquet(output / "dataset_with_volume_profile.parquet", index=False)
    common = np.isfinite(joined[tick + minute]).all(axis=1)
    coverage = {"rows": len(base), "common_rows": int(common.sum()),
                "tick_full_rows": int(np.isfinite(joined[tick]).all(axis=1).sum()),
                "minute_full_rows": int(np.isfinite(joined[minute]).all(axis=1).sum()),
                "feature_data": summary, "feature_columns": arms, "dataset_sha256": file_sha256(args.dataset)}
    (output / "feature-coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f"共通行 {int(common.sum())}/{len(base)}。既存特徴量に価格帯別出来高を結合しました。", flush=True)
    if not args.run:
        return
    if not common.any():
        raise ValueError("両方の20営業日集計が揃う評価対象がありません。")
    frame = joined.loc[common].copy()
    frame = frame.assign(Code=frame.Code.map(normalize_code), Date=pd.to_datetime(frame.Date))
    frame = frame.sort_values(["Date", "Code"]).reset_index(drop=True)
    bars = load_bars(args.prices)
    frame = ltr.attach_r20(frame, bars)
    frame["maturity_date"] = forward_maturity(frame, bars).to_numpy()
    required = [ltr.R20, "maturity_date"]
    if args.task == "class10":
        frame["y_class10"] = LS.labels(frame, "lgbm")
        required.append("y_class10")
    frame = frame.dropna(subset=required).reset_index(drop=True)
    if frame.empty:
        raise ValueError("共通行のうち20営業日のラベルが確定した行がありません。")
    folds = checked_folds(frame, args.task)
    bank = candidate_bank(FIXED, args.trials)
    frozen = []
    trials = []
    for fold, (train, _, inner, _) in enumerate(folds):
        for arm, columns in arms.items():
            inner_data = [(Trainer(train.iloc[ti], columns, args.task, args.threads), train.iloc[vi]) for ti, vi in inner]
            records = []
            for index, params in enumerate(bank):
                scores = []
                for trainer, valid in inner_data:
                    model = trainer.fit(params, trees=200)
                    score = model.predict(valid[columns].to_numpy(float), num_threads=args.threads)
                    scores.append(primary(args.task, valid, score))
                records.append({"candidate": index, "inner_scores": scores, "cv_mean": float(np.mean(scores))})
            best = select_candidate(records)
            frozen.append({"fold": fold, "arm": arm, "candidate": best["candidate"],
                           "parameters": bank[best["candidate"]]})
            trials.extend({"fold": fold, "arm": arm, **record} for record in records)
            print(f"評価窓 {fold + 1}/{len(folds)}、{arm} の探索完了", flush=True)
    (output / "tuning.json").write_text(json.dumps({"task": args.task, "seeds": SEEDS, "trees": 200,
        "price_panel_sha256": file_sha256(args.prices), "inner_windows": [checks for _, _, _, checks in folds],
        "bank": bank, "selections": frozen, "trials": trials}, indent=2, default=str))

    results = []
    for record in frozen:
        fold, arm = record["fold"], record["arm"]
        train, test, _, _ = folds[fold]
        columns = arms[arm]
        trainer = Trainer(train, columns, args.task, args.threads)
        for mode, params in [("fixed", FIXED), ("retuned", record["parameters"])]:
            score = np.mean([trainer.fit(params, seed, 200).predict(test[columns].to_numpy(float), num_threads=args.threads)
                             for seed in SEEDS], axis=0)
            results.append({"task": args.task, "mode": mode, "fold": fold, "arm": arm,
                            "start": test.Date.min().date(), "end": test.Date.max().date(),
                            "train_rows": len(train), "test_rows": len(test), **evaluate(args.task, test, score)})
    pd.DataFrame(results).to_csv(output / "walk-forward.csv", index=False)
    print(f"{args.task}: {len(folds)}窓の評価完了。出力 {output / 'walk-forward.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prices", type=Path, help="20日先までの調整済み日足Parquet（AdjO/AdjH/AdjCまたはopen/high/close）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="特徴量を結合した後にウォークフォワード評価を実施")
    parser.add_argument("--task", choices=["class10", "ltr"], default="class10")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.trials < 2 or args.threads < 1 or (args.run and not args.prices):
        parser.error("探索候補は2以上、スレッドは1以上が必要です。--run には --prices が必要です。")
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
