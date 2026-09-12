#!/usr/bin/env python3
"""
実験15: 5モデルすべてを同じ条件で探索する。

なぜ全部探索するか
-----------------
これまで lgbm だけが探索済みで、他は手で置いた既定値だった。
「lgbm が一貫している」という以前の読みは、探索労力の差を見ていただけ。
ダッシュボードに5モデルのスコアを並べるなら、全部に同じ手間をかける。

条件（lgbm と完全に同一）
  分割      年 × 時価総額帯で層別・日付単位で分割・5分割
  目的関数  分割平均の PR-AUC
  試行数    50
  木の本数  200 固定
  期間      ホールドアウト（直近12ヶ月）より手前のみ。実測 17,449件・
            2018-04-03 〜 2025-03-21

探索が終わったら、探索済みパラメータで out-of-fold を作り直して
運用指標（しきい値運用・翌営業日の寄り買い・40営業日後の5日平均終値売り）
で並べる。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning_multi as TM  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402

N_TRIALS = 50


def main() -> int:
    algos = sys.argv[1:] or list(TM.ALGOS)
    df = lab.frame()
    cols = F.columns("all")
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    print(f"探索期間 {d.min().date()} 〜 {train_end.date()} / {len(sub):,}件 / "
          f"正例率 {sub['label'].mean()*100:.1f}%")
    print(f"対象: {', '.join(algos)} / 各{N_TRIALS}試行")
    print()

    store = TM.load()
    for algo in algos:
        if algo in store:
            print(f"  [{algo}] 探索済みを読む "
                  f"(PR-AUC {store[algo]['_cv']['mean_pr_auc']:.4f})")
            continue
        t0 = time.time()
        store[algo] = TM.tune(algo, sub, cols, n_trials=N_TRIALS)
        # 途中で落ちても結果を失わないよう、1つ終わるたびに保存する
        TM.save(store)
        print(f"  [{algo}] {time.time()-t0:.0f}秒")
        print()

    print("=" * 96)
    print("探索結果（内側検証の PR-AUC。運用指標ではない）")
    print("=" * 96)
    print(f"  {'モデル':<8}{'PR-AUC':>10}{'±':>9}{'ROC-AUC':>10}{'正例率':>9}  パラメータ")
    for algo in algos:
        cv = store[algo]["_cv"]
        p = store[algo]["params"]
        short = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(p.items())[:4])
        print(f"  {algo:<8}{cv['mean_pr_auc']:>10.4f}{cv['std']:>9.4f}"
              f"{cv['mean_roc_auc']:>10.4f}{cv['base_rate']*100:>8.1f}%  {short}")
    print()
    print(f"  保存先: {TM.PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
