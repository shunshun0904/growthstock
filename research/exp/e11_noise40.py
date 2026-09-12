#!/usr/bin/env python3
"""
実験11: 出口を40営業日にした物差しで、ノイズ床を測り直す。

なぜ измер直すか
---------------
足切り線 0.51pt は出口60営業日の物差しで測ったもの。40営業日にすると
窓ごとのばらつき（窓SD）が 1.53 -> 0.90 に下がっており、指標の解像度が
変わっている。古い足切り線で判定すると、採用/却下を両方向に誤る。

同時に t値 のノイズ床も出す。40営業日では19設定のうち多くが t>=2 に
なったが、t値そのものが種で何段動くのかを知らないと「有意」を読めない。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

KEYS = (("thr_lift", "しきい値優位", 1), ("thr_lift_same_day", "対同日候補", 1),
        ("thr_fold_mean", "窓平均", 1), ("thr_fold_sd", "窓SD", 1),
        ("thr_t", "t値", 1), ("thr_folds_won", "勝った窓", 1),
        ("thr_worst", "最悪の窓", 1), ("thr_win", "勝率", 100),
        ("pr_auc", "PR-AUC", 1))


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    print(f"物差し: {lab.OUTCOME}（翌営業日の寄り買い / 40営業日後の5日平均終値売り）")
    print(f"種 {len(lab.SEEDS)}個で同じ設定を回す")
    print()
    rs = [lab.run(df, lab.lgbm(seed=s), cols=cols, name=f"s{s}") for s in lab.SEEDS]
    print(f"  {'指標':<16}{'平均':>10}{'標準偏差':>10}{'最小':>10}{'最大':>10}{'レンジ':>10}")
    for k, title, sc in KEYS:
        v = np.array([r.metrics[k] for r in rs], dtype=float) * sc
        print(f"  {title:<16}{v.mean():>10.3f}{v.std(ddof=1):>10.3f}"
              f"{v.min():>10.3f}{v.max():>10.3f}{v.max()-v.min():>10.3f}")
    print()
    print("  種ごとの t値: " + " / ".join(f"{r.metrics['thr_t']:+.2f}" for r in rs))
    print()
    print("  読み方: レンジより小さい差は種を変えただけでも起きる。採用しない。")
    print("          t値のレンジが大きければ、t>=2 という判定自体が種に依存する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
