#!/usr/bin/env python3
"""
実験07: 新しい指標のノイズ床を測り、そのうえでアルゴリズムを比べる。

的を変えた
---------
運用は「スコアが過去分布の上位10%なら買う」という日をまたいだ絶対しきい値。
日付内の順位は使わない。よって指標は

  しきい値優位   選んだ行の実収益 − 同じ窓の全候補の実収益
  対同日候補     選んだ行の実収益 − 買った日の候補平均（銘柄選定の分だけ）
  勝った窓       窓ごとに全候補を上回った数。再現性
  最悪の窓       いちばん負けた窓の差。運用で耐えられるか

現行 base は +0.93pt / 6of9窓 / 最悪 -1.89pt。これを超えるものを探す。

順番を守る
---------
まずノイズ床。前回これを飛ばして「実験01は負け」と誤判定したので、
新しい指標でも先に測る。同じ設定で種を5つ振り、しきい値優位と
勝った窓がどれだけ揺れるかを見る。この幅を超えない差は採用しない。

比べるもの
---------
  lgbm       現行（探索済みパラメータ）
  xgb        欠損はネイティブ。木の作り方が leaf-wise でなく depth-wise
  cat        欠損と正則化の扱いが違う。誤りの出方が変わることを期待
  rf         バギング。欠損は中央値補完（Pipeline なので fold 内で決まる）
  logit_q    線形。分位変換 + 中央値補完 + 欠損指示子
  logit_s    線形。標準化 + 中央値補完 + 欠損指示子

線形を入れる理由は2つ。ご質問の「スケール変換をどうするか」を実地で測れる
（木は単調変換に不変なので線形でしか差が出ない）のと、木と誤りの出方が
大きく違うのでアンサンブルの相方として価値があるため。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
SCREEN_SEEDS = (42, 7, 123)
KEYS = (("thr_lift", "しきい値優位", 1), ("thr_lift_same_day", "対同日候補", 1),
        ("thr_folds_won", "勝った窓", 1), ("thr_worst", "最悪の窓", 1),
        ("thr_win", "勝率", 100), ("pr_auc", "PR-AUC", 1))


def noise_floor(df, cols) -> None:
    print("=" * 84)
    print("1. 新しい指標のノイズ床（同じ設定で種を5つ振る）")
    print("=" * 84)
    rs = [lab.run(df, lab.lgbm(seed=s), cols=cols, name=f"s{s}") for s in lab.SEEDS]
    print(f"  {'指標':<16}{'平均':>10}{'標準偏差':>10}{'最小':>10}{'最大':>10}{'レンジ':>10}")
    for k, title, sc in KEYS:
        v = np.array([r.metrics[k] for r in rs], dtype=float) * sc
        print(f"  {title:<16}{v.mean():>10.3f}{v.std(ddof=1):>10.3f}"
              f"{v.min():>10.3f}{v.max():>10.3f}{v.max()-v.min():>10.3f}")
    print()
    print("  読み方: レンジより小さい差は種を変えただけでも起きる。採用しない。")


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")

    if "--skip-noise" not in sys.argv:
        noise_floor(df, cols)

    print()
    print("=" * 84)
    print("2. アルゴリズム比較（各3種平均）")
    print("=" * 84)
    algos = {
        "lgbm": lambda s: lab.lgbm(seed=s),
        "xgb": lambda s: lab.xgb(seed=s),
        "cat": lambda s: lab.cat(seed=s),
        "rf": lambda s: lab.rf(seed=s),
        "logit_q": lambda s: lab.logit(seed=s, scaler="quantile"),
        "logit_s": lambda s: lab.logit(seed=s, scaler="standard"),
    }
    os.makedirs(OOF_DIR, exist_ok=True)
    results = {}
    for name, mk in algos.items():
        p = os.path.join(OOF_DIR, f"algo_{name}.parquet")
        if os.path.exists(p):
            o = pd.read_parquet(p)
            results[name] = lab.Result(name=name, oof=o, metrics=lab.metrics(o))
            print(f"  {name:<10} 保存済みを読む")
            continue
        r = lab.run_multi(df, mk, seeds=SCREEN_SEEDS, cols=cols, name=name)
        r.oof.to_parquet(p, index=False)
        results[name] = r
        m = r.metrics
        print(f"  {name:<10} 完了  しきい値優位 {m['thr_lift']:+.2f}pt / "
              f"勝った窓 {m['thr_folds_won']}/{m['thr_folds']}")

    print()
    print(lab.table(results))

    print()
    print("=" * 84)
    print("3. out-of-fold スコアの相関（アンサンブルの効きを見積もる）")
    print("=" * 84)
    key = ["Code", "Date"]
    base = results["lgbm"].oof[key].copy()
    for name, r in results.items():
        base = base.merge(r.oof[key + ["score"]].rename(columns={"score": name}),
                          on=key, how="left")
    names = list(results)
    cm = base[names].corr(method="spearman")
    print(f"  {'':<10}" + "".join(f"{n:>10}" for n in names))
    for n in names:
        print(f"  {n:<10}" + "".join(f"{cm.loc[n, m]:>10.3f}" for m in names))
    print()
    print("  相関が低い組ほどアンサンブルで効く見込みが大きい。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
