#!/usr/bin/env python3
"""
実験19c: 旧データ側も探索の種を振る（19b の対称版）。

なぜ要るか
--------
実験19b で、新データを4通り探索したところ

  種0 +0.50pt / 種1 +1.10pt / 種2 +0.89pt / 種3 +0.72pt   (ret_o1_20)

と 0.60pt のレンジで散った。ばらつきの大きさは分かったが、
**4本すべてが A（旧データ・旧パラメータ +1.20pt）を下回った**。
純粋なばらつきなら半々になるはずで、4/4 は説明がつかない。

ただし A 側は「旧データで1回引いた」n=1 でしかない。
つまり今の時点では

  (a) 旧パラメータがたまたま良い引きだった
  (b) 新データで探索すると系統的に悪い引きになる

を区別できない。ここで旧データ側も同じ4通りに振れば、2つの分布を
並べて読める。(a) なら旧側も同じように散り、A はその上端に位置する。

データ・行・窓・種（モデル側）はすべて 19b と同じ。違うのは
探索に使うデータが旧か新かだけ。

**本番の research/lgbm_params.json には書かない。**
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e19_freshdata import (  # noqa: E402
    CV_SCHEME, N_SPLITS, N_TRIALS, OOF_DIR, OUTCOMES, PRESET, frames, oof_for)
from e19b_tuneseeds import TUNE_SEEDS  # noqa: E402

PARAMS_PATH = os.path.join(OOF_DIR, "e19c_params.json")


def load_store() -> dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)


def tune_seed(seed: int, df: pd.DataFrame, cols: list) -> dict:
    store = load_store()
    key = f"s{seed}"
    if key in store:
        print(f"  種{seed}: 探索済みを読む "
              f"(CV PR-AUC {store[key]['_cv'].get('mean_pr_auc')})")
        return store[key]
    # 種0 は実験19 の "old"（＝A のパラメータ）と同じ条件なので読む
    if seed == 0:
        p19 = os.path.join(OOF_DIR, "e19_params.json")
        if os.path.exists(p19):
            with open(p19, encoding="utf-8") as fh:
                s19 = json.load(fh)
            if "old" in s19:
                store[key] = s19["old"]
                save_store(store)
                print(f"  種{seed}: 実験19 の探索結果（A のパラメータ）を使う "
                      f"(CV PR-AUC {s19['old']['_cv'].get('mean_pr_auc')})")
                return store[key]

    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = T.holdout_bounds(d)
    t0 = time.time()
    params = tuning.tune(df[d <= cutoff], cols, n_trials=N_TRIALS,
                         n_splits=N_SPLITS, embargo_days=T.EMBARGO_DAYS,
                         scheme=CV_SCHEME, seed=seed, verbose=False)
    rec = {**params, "_cv": dict(tuning.LAST_CV)}
    store[key] = rec
    save_store(store)
    print(f"  種{seed}: {(time.time()-t0)/60:.1f}分 / "
          f"lr {params['learning_rate']:.4f} 葉{params['num_leaves']} / "
          f"CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    old, new = frames()
    cols = [c for c in F.columns(PRESET)
            if c in old.columns and c in new.columns]
    print(f"\n旧データ {len(old):,}件（共通行）/ 特徴量 {len(cols)}列")
    print(f"探索の種 {TUNE_SEEDS} —— 19b と同じ振り方、データだけ旧\n")

    print("=== 旧データで種を変えて探索する ===")
    params = {s: tune_seed(s, old, cols) for s in TUNE_SEEDS}
    strip = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}

    print("\n=== それぞれで out-of-fold を作る（旧データで学習・採点）===")
    runs = {}
    for s in TUNE_SEEDS:
        p = os.path.join(OOF_DIR, f"e19c_{s}.parquet")
        alt = os.path.join(OOF_DIR, "e19_0.parquet")   # 種0 = A
        if os.path.exists(p):
            runs[s] = pd.read_parquet(p)
            print(f"  種{s}: 保存済みを読む")
            continue
        if s == 0 and os.path.exists(alt):
            runs[s] = pd.read_parquet(alt)
            print(f"  種{s}: 実験19 の A を使う")
            continue
        t0 = time.time()
        o = oof_for(old, cols, strip(params[s]))
        o.to_parquet(p, index=False)
        runs[s] = o
        print(f"  種{s}: {len(o):,}件 / {time.time()-t0:.0f}秒")

    # 19b（新データ側）を読み込んで並べる
    nb = {0: "e19_2.parquet", 1: "e19b_1.parquet",
          2: "e19b_2.parquet", 3: "e19b_3.parquet"}
    newruns = {s: pd.read_parquet(os.path.join(OOF_DIR, f))
               for s, f in nb.items()
               if os.path.exists(os.path.join(OOF_DIR, f))}

    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome} —— 探索の種ごと ===")
        print(f"  {'探索の種':<10}{'旧データ':>12}{'新データ':>12}")
        ov, nv = [], []
        for s in TUNE_SEEDS:
            a = edge(runs[s], outcome)["thr_fold_mean"]
            ov.append(a)
            if s in newruns:
                b = edge(newruns[s], outcome)["thr_fold_mean"]
                nv.append(b)
                print(f"  {s:<10}{a:>+11.2f}pt{b:>+11.2f}pt")
            else:
                print(f"  {s:<10}{a:>+11.2f}pt{'—':>12}")
        ov = np.array(ov)
        print(f"  --- 分布 ---")
        print(f"    旧データ 平均 {ov.mean():+.2f}pt / レンジ {ov.max()-ov.min():.2f}pt "
              f"(最小 {ov.min():+.2f} / 最大 {ov.max():+.2f}) / SD {ov.std(ddof=1):.2f}")
        if nv:
            nv = np.array(nv)
            print(f"    新データ 平均 {nv.mean():+.2f}pt / レンジ {nv.max()-nv.min():.2f}pt "
                  f"(最小 {nv.min():+.2f} / 最大 {nv.max():+.2f}) / SD {nv.std(ddof=1):.2f}")
            d = nv.mean() - ov.mean()
            se = np.sqrt(ov.std(ddof=1)**2/len(ov) + nv.std(ddof=1)**2/len(nv))
            print(f"    平均の差（新 - 旧） {d:+.2f}pt / SE {se:.2f} / "
                  f"z {d/se if se else float('nan'):+.2f}")
            print(f"    ※ このzは『探索の引きのばらつき』だけを見たもの。"
                  f"窓のばらつきは含まない")
            print(f"    旧側で A(+{ov[0]:.2f}pt) 以上だった引き: "
                  f"{int((ov >= ov[0]).sum())}/{len(ov)}")

    print("\n読み方: 旧側も同じくらい散り、A がその上端なら『A が良い引きだっただけ』。")
    print("        旧側だけ高い位置に固まるなら、新データ側に何かある。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
