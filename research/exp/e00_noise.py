#!/usr/bin/env python3
"""
実験00: ノイズ床を測る。

これを最初にやる理由
------------------
LightGBM は特徴量サブサンプル・バギングに乱数を使うので、同じ設定でも
種を変えると結果が動く。その揺れ幅を知らないまま実験を並べると、
乱数の揺れを「改善した」と読んでしまう。

ここで各指標の種間ばらつき（標準偏差とレンジ）を出し、
「この幅を超えない差は無かったことにする」という足切り線を決める。

実験01（欠損を特徴量にする）の差は上位5%収益で -0.47 〜 +0.04pt だった。
この幅がノイズ床の中なら、あの実験は「効かない」ではなく「測れていない」。
どちらなのかをはっきりさせる。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402

SEEDS = (42, 7, 123, 2024, 31337)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    base = {k: v for k, v in tuning.params_for("all").items()
            if not k.startswith("_")}

    rows = []
    for s in SEEDS:
        p = dict(base, random_state=s, verbose=-1)
        r = lab.run(df, lab.lgbm(p), cols=cols, name=f"seed{s}")
        rows.append(r)
        print(f"  seed {s:<6} 完了")

    print()
    print(lab.table({f"seed={s}": r for s, r in zip(SEEDS, rows)}))
    print()
    print("=== 種を変えただけのばらつき（＝ノイズ床）===")
    print(f"{'指標':<14}{'平均':>10}{'標準偏差':>10}{'最小':>10}{'最大':>10}{'レンジ':>10}")
    for key, title, scale in (("pr_auc", "PR-AUC", 1), ("roc_auc", "ROC-AUC", 1),
                              ("auc_in_day", "日付内AUC", 1), ("p_at_5", "P@5%", 100),
                              ("end_5", "上位5%収益", 1), ("end_1", "上位1%収益", 1),
                              ("end_10", "上位10%収益", 1)):
        v = np.array([r.metrics[key] for r in rows]) * scale
        print(f"{title:<14}{v.mean():>10.4f}{v.std(ddof=1):>10.4f}"
              f"{v.min():>10.4f}{v.max():>10.4f}{v.max()-v.min():>10.4f}")
    print()
    print("読み方: レンジより小さい差は、種を変えただけでも起きる。")
    print("        実験の差がこの幅に収まるなら『効果なし』ではなく『測れていない』。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
