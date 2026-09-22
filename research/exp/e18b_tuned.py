#!/usr/bin/env python3
"""
実験18b: 各条件に自分のハイパーパラメータを与えても、結論は変わらないか。

なぜ測り直すか
------------
実験18 は4条件すべてに同じパラメータを使った。そのパラメータは
`research/lgbm_params.json` の探索済みのもので、**旧ラベル
（h=60 / MA20>=MA60）に対して、しかも ETF を含む母集団で**探索された値。

つまり旧条件だけが自陣で戦っていた。新条件が負けたように見えたのが
「ラベルが悪いから」なのか「借り物のパラメータだから」なのかを分けられない。

ここでは **条件ごとに自分のラベルで探索し直してから**同じ比較をする。
これで全条件が自陣になる。

実験18 との違いはパラメータだけ。行・窓・シード・物差しはすべて同じに
してあるので、2つの表を並べれば探索の効果そのものが読める。

コスト
-----
1条件あたり 50試行 × 5分割。4条件で1時間前後。
探索結果は research/_data/oof/h_tuned_params.json に貯めるので、
途中で止めても次回は続きから走る。

**本番の research/lgbm_params.json には書かない。** 実験が本番の設定を
書き換えると、日曜の週次実行が何で走ったのか分からなくなる。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
from e18_horizon import ARMS, OUTCOMES, SEEDS, edge, label_columns  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "h_tuned_params.json")
N_TRIALS = 50
N_SPLITS = 5


def load_store() -> dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)


def tune_arm(i: int, name: str, cfg, df: pd.DataFrame, cols: list) -> dict:
    """条件 i のラベルで探索する。探索期間はホールドアウトより前で打ち切る。"""
    store = load_store()
    key = f"h{i}"
    if key in store:
        print(f"  {name}: 探索済みを読む "
              f"(PR-AUC {store[key]['_cv'].get('mean_pr_auc')})")
        return store[key]

    sub = df[df[f"y{i}"].notna()].copy()
    sub["label"] = sub[f"y{i}"]
    # 探索に使ってよいのはホールドアウトより前だけ。エンバーゴは
    # その条件自身のホライズンから取る（借りるとリークする）
    d = pd.to_datetime(sub["Date"])
    train_end, _, _ = T.holdout_bounds(d, embargo_days=cfg.horizon)
    tune_df = sub[d <= train_end]
    print(f"  {name}: 探索 {len(tune_df):,}件 (〜{train_end.date()} / "
          f"正例率 {tune_df['label'].mean()*100:.2f}% / "
          f"エンバーゴ {cfg.horizon}営業日)")
    t0 = time.time()
    params = tuning.tune(tune_df, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                         embargo_days=cfg.horizon, scheme="year",
                         verbose=False)
    rec = {**params, "_cv": dict(tuning.LAST_CV)}
    store[key] = rec
    save_store(store)
    print(f"    {(time.time()-t0)/60:.1f}分 / "
          f"木{params['n_estimators']}本 lr {params['learning_rate']:.4f} "
          f"葉{params['num_leaves']} / CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


def main() -> int:
    # 実験18 と同じ窓にする（比較できなくなるので変えない）
    B.RISE_HORIZON = 60

    df = lab.frame()
    cols = F.columns("all")
    os.makedirs(OOF_DIR, exist_ok=True)

    print(f"母集団 {len(df):,}件")
    print(f"ラベルを{len(ARMS)}条件ぶん作り直す")
    df = label_columns(df)

    ycols = [f"y{i}" for i in range(len(ARMS))]
    df = df[df[ycols].notna().all(axis=1)].reset_index(drop=True)
    print(f"\n全条件でラベルが確定している行 {len(df):,}件（実験18 と同じ）")

    print("\n=== 条件ごとに自分のラベルで探索する ===")
    params = {}
    for i, (name, cfg) in enumerate(ARMS):
        params[name] = tune_arm(i, name, cfg, df, cols)

    print("\n=== 探索したパラメータで学習し直す ===")
    runs = {}
    for i, (name, cfg) in enumerate(ARMS):
        p = os.path.join(OOF_DIR, f"ht_{i}.parquet")
        if os.path.exists(p):
            runs[name] = lab.attach_outcomes(pd.read_parquet(p), df)
            print(f"  {name}: 保存済みを読む")
            continue
        sub = df.copy()
        sub["label"] = sub[f"y{i}"]
        pp = {k: v for k, v in params[name].items() if not k.startswith("_")}
        r = lab.run_multi(sub, lambda s: lab.lgbm(pp, seed=s), seeds=SEEDS,
                          cols=cols, name=f"ht{i}")
        r.oof.to_parquet(p, index=False)
        runs[name] = r.oof
        print(f"  {name}: out-of-fold {len(r.oof):,}件")

    from sklearn.metrics import average_precision_score, roc_auc_score
    print("\n=== 分離力（out-of-fold）===")
    print(f"  {'条件':<24}{'正例率':>9}{'PR-AUC':>9}{'PR/正例率':>11}"
          f"{'ROC-AUC':>10}{'日内AUC':>10}")
    for name, oof in runs.items():
        y = oof["label"].to_numpy(dtype=int)
        s = oof["score"].to_numpy(dtype=float)
        br = y.mean()
        pr = average_precision_score(y, s)
        print(f"  {name:<24}{br*100:>8.2f}%{pr:>9.4f}{pr/br:>10.2f}x"
              f"{roc_auc_score(y, s):>10.4f}{lab.auc_in_day(oof):>10.4f}")

    for outcome in OUTCOMES:
        print(f"\n=== 物差し {outcome} ===")
        print(f"  {'条件':<24}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
        ms = {}
        for name, oof in runs.items():
            m = edge(oof, outcome)
            ms[name] = m
            print(f"  {name:<24}{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt{m['thr_n']:>9,}")

        print("  --- 差の検定（足切り z>2）---")
        names = [n for n, _ in ARMS]
        pairs = [(0, 2, "旧 -> 新（どちらも自分のパラメータ）"),
                 (3, 2, "トレンド条件なし -> MA5>=MA20 を足す"),
                 (1, 3, "「中間」はトレンド条件なしと同じか")]
        for a, b, note in pairs:
            ma, mb = ms[names[a]], ms[names[b]]
            d = mb["thr_fold_mean"] - ma["thr_fold_mean"]
            se = np.sqrt(ma["se"] ** 2 + mb["se"] ** 2)
            z = d / se if se else float("nan")
            v = "採用可" if z > 2 else ("要確認" if z > 1 else
                                     ("差なし" if z > -1 else "悪化"))
            print(f"    {note}")
            print(f"      差 {d:+.2f}pt / 合成SE {se:.2f} / z {z:+.2f} → {v}")

    print("\n実験18（全条件で同じパラメータ）の表と並べて読むこと。")
    print("行・窓・シード・物差しは同じにしてあるので、差はパラメータだけ。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
