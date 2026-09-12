#!/usr/bin/env python3
"""
実験08: アンサンブルと、探索そのものの効果。

なぜこの2つを一緒に測るか
------------------------
実験07 で2つ分かった。

1. 無調整のロジスティック回帰(logit_s)が、探索済み LightGBM と運用指標で並んだ
   (しきい値優位 +0.93pt vs +0.93pt、最悪の窓 -1.43pt vs -1.89pt)。
   しかも PR-AUC は最下位(0.2803 vs 0.2996)。
2. 線形モデルと木のスコア相関は 0.58〜0.67 で、木どうし(0.77〜0.88)より
   ずっと低い。アンサンブルの相方として最有力。

1 は「lgbm の有利さは探索労力の差にすぎない」という指摘を裏側から補強する。
探索が運用指標に効いていないなら、比較の不公平さ以前に、週36分かけている
探索自体を見直す必要がある。そこで探索の有無も同じ台で測る。

測るもの
--------
  A. 探索の効果      lgbm_tuned(現行) vs lgbm_default(無調整)
  B. アンサンブル    保存済み out-of-fold を組み合わせる。学習不要

アンサンブルの作り方
------------------
  rank  各モデルのスコアを窓ごとにパーセンタイル化して平均
        スケールが違うモデルを混ぜるので、確率の平均では
        分布の広いモデルに引っ張られる。順位平均なら影響を受けない
  prob  確率の単純平均（比較用）

窓ごとに順位化するのは、しきい値運用が窓をまたいだスコアの水準を使うため。
全期間で順位化すると未来を見ることになる。
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
SCREEN_SEEDS = (42, 7, 123)
KEY = ["Code", "Date", "label", "ref_end", "ref_rise", "fold"]

#: 無調整の LightGBM。探索の効果を測る相手
LGBM_DEFAULT = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1)


def load(name: str) -> pd.DataFrame:
    p = os.path.join(OOF_DIR, f"{name}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"{p} がありません")
    return pd.read_parquet(p)


def combine(oofs: Dict[str, pd.DataFrame], names: List[str],
            how: str = "rank") -> pd.DataFrame:
    """複数の out-of-fold を1つのスコアにまとめる。"""
    base = oofs[names[0]][KEY].copy()
    cols = []
    for n in names:
        o = oofs[n][["Code", "Date", "score", "fold"]].copy()
        if how == "rank":
            # 窓ごとに順位化。しきい値運用は窓をまたいだ水準を使うので、
            # 全期間で順位化すると未来を見てしまう
            o["_v"] = o.groupby("fold")["score"].rank(pct=True)
        else:
            o["_v"] = o["score"]
        base = base.merge(o[["Code", "Date", "_v"]].rename(columns={"_v": n}),
                          on=["Code", "Date"], how="left")
        cols.append(n)
    base["score"] = base[cols].mean(axis=1)
    return base.drop(columns=cols)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")

    print("=" * 96)
    print("A. 探索そのものに効果があるか（lgbm 探索済み vs 無調整）")
    print("=" * 96)
    res = {}
    p = os.path.join(OOF_DIR, "algo_lgbm_default.parquet")
    if os.path.exists(p):
        o = pd.read_parquet(p)
        res["lgbm_default"] = lab.Result("lgbm_default", o, lab.metrics(o))
        print("  保存済みを読む")
    else:
        r = lab.run_multi(df, lambda s: lab.lgbm(dict(LGBM_DEFAULT), seed=s),
                          seeds=SCREEN_SEEDS, cols=cols, name="lgbm_default")
        r.oof.to_parquet(p, index=False)
        res["lgbm_default"] = r
        print("  学習して保存")
    o = load("algo_lgbm")
    res["lgbm_tuned"] = lab.Result("lgbm_tuned", o, lab.metrics(o))
    print()
    print(lab.table(res))

    print()
    print("=" * 96)
    print("B. アンサンブル（保存済み out-of-fold の組み合わせ。学習不要）")
    print("=" * 96)
    singles = ["lgbm", "xgb", "cat", "rf", "logit_q", "logit_s"]
    oofs = {n: load(f"algo_{n}") for n in singles}
    oofs["lgbm_default"] = res["lgbm_default"].oof

    combos = {
        # 相関のいちばん低い組（木 + 線形）
        "lgbm+logit_q": ["lgbm", "logit_q"],
        "lgbm+logit_s": ["lgbm", "logit_s"],
        "xgb+logit_q": ["xgb", "logit_q"],
        # 木だけ（相関0.77〜0.88なので効きは小さいはず）
        "lgbm+xgb": ["lgbm", "xgb"],
        "lgbm+xgb+cat": ["lgbm", "xgb", "cat"],
        # 木3種 + 線形2種
        "trees+linear": ["lgbm", "xgb", "cat", "logit_q", "logit_s"],
        # 全部
        "all6": singles,
    }
    ens = {}
    for name, members in combos.items():
        o = combine(oofs, members, how="rank")
        ens[name] = lab.Result(name, o, lab.metrics(o))
    # 単体も並べて比べる
    show = {f"[単体] {n}": lab.Result(n, oofs[n], lab.metrics(oofs[n]))
            for n in ("lgbm", "xgb", "logit_q", "logit_s")}
    show.update(ens)
    print(lab.table(show))

    print()
    print("  --- 確率平均と順位平均の比較（混ぜ方の影響）---")
    cmp_ = {}
    for name in ("lgbm+logit_q", "trees+linear"):
        for how in ("rank", "prob"):
            o = combine(oofs, combos[name], how=how)
            cmp_[f"{name}/{how}"] = lab.Result(name, o, lab.metrics(o))
    print(lab.table(cmp_))

    print()
    print("=" * 96)
    print("C. 判定（ノイズ床 0.51pt / 窓平均のノイズは t値で見る）")
    print("=" * 96)
    ref = lab.metrics(oofs["lgbm"])
    print(f"  基準: lgbm 単体  しきい値優位 {ref['thr_lift']:+.2f}pt / "
          f"窓平均 {ref['thr_fold_mean']:+.2f}pt / t値 {ref['thr_t']:+.2f}")
    print()
    rows = []
    for name, r in ens.items():
        m = r.metrics
        rows.append((name, m["thr_lift"] - ref["thr_lift"],
                     m["thr_fold_mean"] - ref["thr_fold_mean"],
                     m["thr_t"], m["thr_worst"] - ref["thr_worst"]))
    rows.sort(key=lambda x: -x[1])
    print(f"  {'組み合わせ':<18}{'優位の差':>10}{'窓平均の差':>12}{'t値':>8}{'最悪の窓の差':>14}{'判定':>8}")
    for name, d, dm, t, dw in rows:
        verdict = "採用可" if d > 0.51 else ("要確認" if d > 0.25 else "ノイズ内")
        print(f"  {name:<18}{d:>+9.2f}pt{dm:>+11.2f}pt{t:>+8.2f}{dw:>+13.2f}pt{verdict:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
