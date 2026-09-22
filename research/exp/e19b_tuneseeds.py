#!/usr/bin/env python3
"""
実験19b: 探索を引き直すと、答えはどれだけ動くか。

なぜ測るか
--------
実験19 で、同じ新データに対して

  B 旧パラメータのまま  ret_o1_20 窓平均 +1.28pt
  C 探索を引き直した    ret_o1_20 窓平均 +0.50pt

と出た。データの差（+0.08pt）より、**探索を引き直した差（-0.78pt）のほうが
10倍近く大きい**。しかも探索側のCV PR-AUC は C のほうが良い（0.3141 >
0.3095）のに、ウォークフォワード OOF は悪い。

ここで分からないのは、C が

  (a) 本当に悪いパラメータ
  (b) 探索がばらつくだけで、たまたま悪い引きだった

のどちらかということ。1回引いただけでは区別できない。日曜の本番学習は
毎週かならず引き直すので、**引き直しそのもののばらつき**を知らないと、
来週スコアが動いたときに「相場が変わった」のか「探索の引きが変わった」
のかを切り分けられない。

測り方
-----
新データ・同じ行・同じ探索条件（50試行 / 5分割 / year_cap_date）で、
**Optuna の種だけ**を変えて4回引く。tuning.tune の seed は探索器の種と
分割の種の両方を決めるので、これが「もう一度探索し直す」に相当する。

それぞれの結果で out-of-fold を作り、窓平均のばらつきを見る。
比較の基準は実験19 の A（旧データ・旧パラメータ、+1.20pt）。

読み方
-----
4回の窓平均のレンジが 0.78pt 前後まで開くなら (b)。
4回とも A より低く固まるなら (a)。

実験11 のノイズ床（モデルの種だけを振ったときの窓平均レンジ 0.143pt）と
並べれば、「探索の引き直し」がモデルの種より何倍うるさいかも出る。

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

PARAMS_PATH = os.path.join(OOF_DIR, "e19b_params.json")
#: 探索の種。0 は実験19 の C と同じ引き（結果を使い回す）
TUNE_SEEDS = (0, 1, 2, 3)


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
        cv = store[key].get("_cv", {})
        print(f"  種{seed}: 探索済みを読む (CV PR-AUC {cv.get('mean_pr_auc')})")
        return store[key]
    # 種0 は実験19 の "new" と同じ条件なので、あれば読む
    if seed == 0:
        p19 = os.path.join(OOF_DIR, "e19_params.json")
        if os.path.exists(p19):
            with open(p19, encoding="utf-8") as fh:
                s19 = json.load(fh)
            if "new" in s19:
                store[key] = s19["new"]
                save_store(store)
                cv = s19["new"].get("_cv", {})
                print(f"  種{seed}: 実験19 の探索結果を使う "
                      f"(CV PR-AUC {cv.get('mean_pr_auc')})")
                return store[key]

    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = T.holdout_bounds(d)
    tune_df = df[d <= cutoff]
    t0 = time.time()
    params = tuning.tune(tune_df, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME,
                         seed=seed, verbose=False)
    rec = {**params, "_cv": dict(tuning.LAST_CV)}
    store[key] = rec
    save_store(store)
    cv = rec["_cv"]
    print(f"  種{seed}: {(time.time()-t0)/60:.1f}分 / "
          f"lr {params['learning_rate']:.4f} 葉{params['num_leaves']} "
          f"min_child {params['min_child_samples']} / "
          f"CV PR-AUC {cv.get('mean_pr_auc')} ROC {cv.get('mean_roc_auc')}")
    return rec


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score

    os.makedirs(OOF_DIR, exist_ok=True)
    old, new = frames()
    cols = [c for c in F.columns(PRESET)
            if c in old.columns and c in new.columns]
    print(f"\n新データ {len(new):,}件（共通行）/ 特徴量 {len(cols)}列")
    print(f"探索 {N_TRIALS}試行 × {N_SPLITS}分割 / {CV_SCHEME}")
    print(f"探索の種 {TUNE_SEEDS} —— 変えるのはこれだけ\n")

    print("=== 種を変えて探索し直す ===")
    params = {s: tune_seed(s, new, cols) for s in TUNE_SEEDS}
    strip = lambda p: {k: v for k, v in p.items() if not k.startswith("_")}

    print("\n=== それぞれで out-of-fold を作る ===")
    runs = {}
    for s in TUNE_SEEDS:
        p = os.path.join(OOF_DIR, f"e19b_{s}.parquet")
        # 種0 は実験19 の C と同一なので、あれば使い回す
        alt = os.path.join(OOF_DIR, "e19_2.parquet")
        if os.path.exists(p):
            runs[s] = pd.read_parquet(p)
            print(f"  種{s}: 保存済みを読む")
            continue
        if s == 0 and os.path.exists(alt):
            runs[s] = pd.read_parquet(alt)
            print(f"  種{s}: 実験19 の C を使う")
            continue
        t0 = time.time()
        o = oof_for(new, cols, strip(params[s]))
        o.to_parquet(p, index=False)
        runs[s] = o
        print(f"  種{s}: {len(o):,}件 / {time.time()-t0:.0f}秒")

    # 比較の基準（実験19 の A: 旧データ・旧パラメータ）
    base_path = os.path.join(OOF_DIR, "e19_0.parquet")
    base = pd.read_parquet(base_path) if os.path.exists(base_path) else None

    print("\n=== 分離力（out-of-fold）===")
    print(f"  {'探索の種':<10}{'PR-AUC':>9}{'ROC-AUC':>10}{'日内AUC':>10}"
          f"{'探索のCV PR-AUC':>18}")
    for s in TUNE_SEEDS:
        o = runs[s]
        y = o["label"].to_numpy(dtype=int)
        sc = o["score"].to_numpy(dtype=float)
        cv = params[s].get("_cv", {}).get("mean_pr_auc")
        print(f"  {s:<10}{average_precision_score(y, sc):>9.4f}"
              f"{roc_auc_score(y, sc):>10.4f}{lab.auc_in_day(o):>10.4f}"
              f"{cv:>18}")
    if base is not None:
        y = base["label"].to_numpy(dtype=int)
        sc = base["score"].to_numpy(dtype=float)
        print(f"  {'(A 旧)':<10}{average_precision_score(y, sc):>9.4f}"
              f"{roc_auc_score(y, sc):>10.4f}{lab.auc_in_day(base):>10.4f}"
              f"{'—':>18}")

    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome} ===")
        print(f"  {'探索の種':<10}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}")
        vals = []
        for s in TUNE_SEEDS:
            m = edge(runs[s], outcome)
            vals.append(m["thr_fold_mean"])
            print(f"  {s:<10}{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt")
        if base is not None:
            mb = edge(base, outcome)
            print(f"  {'(A 旧)':<10}{mb['thr_fold_mean']:>+9.2f}pt"
                  f"{mb['se']:>10.2f}{mb['thr_folds_won']:>6}/"
                  f"{mb['thr_folds']:<2}{mb['thr_worst']:>+10.2f}pt")
        v = np.array(vals)
        print(f"  --- 探索を引き直しただけのばらつき ---")
        print(f"    レンジ {v.max()-v.min():.2f}pt "
              f"(最小 {v.min():+.2f} / 最大 {v.max():+.2f}) / "
              f"標準偏差 {v.std(ddof=1):.2f}pt")
        print(f"    実験11 のノイズ床（モデルの種だけ）: 窓平均レンジ 0.143pt")
        print(f"    -> 探索の引き直しは {(v.max()-v.min())/0.143:.0f}倍うるさい")
        if base is not None:
            n_below = int((v < mb["thr_fold_mean"]).sum())
            print(f"    A（旧データ・旧パラメータ {mb['thr_fold_mean']:+.2f}pt）を"
                  f"下回った引き: {n_below}/{len(v)}")

    print("\n読み方: レンジが実験19 の B->C 差（-0.78pt）と同じ桁なら、")
    print("        あの差は『悪いパラメータ』ではなく『探索のばらつき』。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
