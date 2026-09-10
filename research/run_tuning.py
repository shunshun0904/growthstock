#!/usr/bin/env python3
"""
LightGBM のハイパーパラメータを探索して保存する。

探索に使う期間を「ホールドアウトより前」に限る。
ここを間違えるとリークし、以降の評価がすべて無効になるので、
境界は train_model.holdout_bounds から機械的に導く（手で書かない）。

    |--- 探索に使ってよい ---|--エンバーゴ--|--- ホールドアウト(直近1年) ---|
                            ↑ ここで打ち切る

以前はウォークフォワードの最初のフォールドの訓練終了日で打ち切っていた。
それだと min_train_months（根拠のない36という数字）が探索に使える
データ量を決めてしまい、実測で 2018-07〜2021-07 の約2割しか使えていなかった。
ホールドアウト方式なら直前までの全期間が使える。

結果は research/_data/lgbm_params.json に保存し、
単一分割（train_model.py）とウォークフォワード（walkforward.py）が共有する。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import tuning  # noqa: E402
from train_model import (  # noqa: E402
    DATA_DIR, EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds,
)
from walkforward import DEFAULT_PRESETS  # noqa: E402


def tuning_cutoff(dates: pd.Series, args) -> pd.Timestamp:
    """
    探索に使ってよい最終日。ホールドアウト開始日からエンバーゴぶん戻した日。

    学習側（train_model.holdout_split）と同じ関数から取るので、
    ホールドアウトの長さを変えても片方だけずれることがない。
    """
    train_end, _, _ = holdout_bounds(dates, args.holdout_months, EMBARGO_DAYS)
    return train_end


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LightGBM のパラメータ探索")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--features", nargs="*", default=None,
                    help=f"探索するプリセット。既定: {' '.join(DEFAULT_PRESETS)}")
    ap.add_argument("--holdout-months", type=int, default=HOLDOUT_MONTHS,
                    help="評価用に取り分ける直近の月数。ここより後は探索に使わない")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="探索の評価に使う分割の数")
    ap.add_argument("--cv",
                    choices=["year", "year_cap", "year_cap_date", "cap",
                             "timeseries"],
                    default="year",
                    help="分割方式。year=年で層別（既定） / "
                         "year_cap=年×時価総額帯 / "
                         "year_cap_date=それに加えて日付単位で分割 / "
                         "cap=時価総額帯のみ / "
                         "timeseries=時系列")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset).sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"])

    cutoff = tuning_cutoff(dates, args)
    tune_df = df[dates <= cutoff]
    print(f"[load] 全体 {len(df):,}件 / {dates.min().date()} 〜 {dates.max().date()}")
    print(f"[tune] 探索に使う期間: 〜{cutoff.date()} ({len(tune_df):,}件 "
          f"/ 正例率 {tune_df['label'].mean()*100:.2f}%)")
    print(f"[tune] これ以降は一切見ない（ホールドアウトより前で打ち切り）")
    if len(tune_df) == 0 or tune_df["label"].nunique() < 2:
        raise SystemExit("探索期間に両クラスが揃いません")

    presets = args.features or DEFAULT_PRESETS
    store = tuning.load_params()
    t0 = time.time()
    for i, preset in enumerate(presets, 1):
        cols = [c for c in F.columns(preset) if c in df.columns]
        missing = [c for c in F.columns(preset) if c not in df.columns]
        if missing:
            print(f"[skip] {preset}: 列がありません {missing[:3]}...")
            continue
        el = time.time() - t0
        print(f"\n[{i}/{len(presets)}] {preset} ({len(cols)}列) "
              f"— 経過 {el/60:.1f}分")
        params = tuning.tune(tune_df, cols, n_trials=args.n_trials,
                             n_splits=args.n_splits, embargo_days=EMBARGO_DAYS,
                             scheme=args.cv)
        # 探索の記録を一緒に保存する。params_for が読むときに落とすので
        # LightGBM には渡らない
        store[preset] = {**params, "_cv": dict(tuning.LAST_CV),
                         "_n_features": len(cols)}
        tuning.save_params(store)      # 途中で落ちても結果を失わない

    print(f"\n[done] {len(store)}件を {tuning.PARAMS_PATH} に保存 "
          f"({(time.time()-t0)/60:.1f}分)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
