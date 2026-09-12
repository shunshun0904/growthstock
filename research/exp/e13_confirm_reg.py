#!/usr/bin/env python3
"""
実験13: 回帰の優位を5種で確認する。

実験12 で reg_l2（実収益を二乗誤差で当てる回帰）が、窓平均 +2.80pt と
現行の分類器 +0.91pt を大きく上回った（差/合成SE = 2.44）。
ただし3種平均のみで、5設定からの選択でもある。

ここでやること
-------------
1. clf_lgbm と reg_l2 を5種で回し直す
2. 種ごとの窓平均を並べて、差が種に依存しないかを見る
3. 実収益の分位ごとに「回帰と分類でどちらが上位に置いたか」を見て、
   勝率が下がって窓平均が上がる仕組みを確かめる

勝率が 62.3% -> 53.8% に下がるのに窓平均が3倍になるのは、
回帰が裾（大きく上がる銘柄）を狙いに行っているためと見ている。
それが本当なら、選ばれた銘柄の実収益の分布が右に伸びているはず。
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


def picks(oof: pd.DataFrame, pct: float = 90) -> pd.DataFrame:
    out = []
    for f in sorted(oof["fold"].unique()):
        ref = oof.loc[oof["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        t = float(np.percentile(ref, pct))
        cur = oof[oof["fold"] == f]
        out.append(cur[cur["score"] > t])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    tgt = lab.OUTCOME

    cfgs = {
        "clf_lgbm": (lambda s: lab.lgbm(seed=s), "label"),
        "reg_l2": (lambda s: lab.lgbm_reg(seed=s, objective="l2"), tgt),
    }
    res = {}
    for name, (mk, t) in cfgs.items():
        p = os.path.join(OOF_DIR, f"c5_{name}.parquet")
        if os.path.exists(p):
            o = lab.attach_outcomes(pd.read_parquet(p), df)
            res[name] = lab.Result(name, o, lab.metrics(o))
            print(f"  {name:<10} 保存済みを読む")
            continue
        r = lab.run_multi(df, mk, seeds=lab.SEEDS, cols=cols, name=name, target=t)
        r.oof.to_parquet(p, index=False)
        res[name] = r
        print(f"  {name:<10} 完了（5種）")

    print()
    print(lab.table(res))

    print()
    print("=== 種ごとの窓平均（差が種に依存しないか）===")
    print(f"  {'種':<10}{'clf':>10}{'reg_l2':>10}{'差':>10}")
    diffs = []
    for i, s in enumerate(lab.SEEDS):
        a = res["clf_lgbm"].per_seed[i].metrics["thr_fold_mean"]
        b = res["reg_l2"].per_seed[i].metrics["thr_fold_mean"]
        diffs.append(b - a)
        print(f"  {s:<10}{a:>+9.2f}pt{b:>+9.2f}pt{b-a:>+9.2f}pt")
    d = np.array(diffs)
    print(f"  {'平均':<10}{'':>10}{'':>10}{d.mean():>+9.2f}pt  "
          f"(SD {d.std(ddof=1):.2f} / 全種で正: {(d > 0).all()})")

    print()
    print("=== 選ばれた銘柄の実収益の分布（仕組みの確認）===")
    print(f"  {'モデル':<10}{'件数':>7}{'平均':>9}{'中央':>9}{'勝率':>8}"
          f"{'25%':>9}{'75%':>9}{'95%':>9}{'歪度':>8}")
    for name, r in res.items():
        p = picks(r.oof)
        e = pd.to_numeric(p[tgt], errors="coerce").dropna()
        print(f"  {name:<10}{len(p):>7,}{e.mean()*100:>+8.2f}%{e.median()*100:>+8.2f}%"
              f"{(e > 0).mean()*100:>7.1f}%{e.quantile(.25)*100:>+8.1f}%"
              f"{e.quantile(.75)*100:>+8.1f}%{e.quantile(.95)*100:>+8.1f}%{e.skew():>8.2f}")
    uni = res["clf_lgbm"].oof
    uni = uni[uni["fold"] > uni["fold"].min()]
    e = pd.to_numeric(uni[tgt], errors="coerce").dropna()
    print(f"  {'(全候補)':<10}{len(uni):>7,}{e.mean()*100:>+8.2f}%{e.median()*100:>+8.2f}%"
          f"{(e > 0).mean()*100:>7.1f}%{e.quantile(.25)*100:>+8.1f}%"
          f"{e.quantile(.75)*100:>+8.1f}%{e.quantile(.95)*100:>+8.1f}%{e.skew():>8.2f}")

    print()
    print("=== 両モデルのスコアの相関（ダッシュボードに並べる価値があるか）===")
    key = ["Code", "Date"]
    m = res["clf_lgbm"].oof[key + ["score"]].merge(
        res["reg_l2"].oof[key + ["score"]], on=key, suffixes=("_clf", "_reg"))
    print(f"  Spearman {m['score_clf'].corr(m['score_reg'], method='spearman'):.3f}")
    pc = set(map(tuple, picks(res["clf_lgbm"].oof)[key].to_numpy()))
    pr = set(map(tuple, picks(res["reg_l2"].oof)[key].to_numpy()))
    print(f"  上位10%に選ばれた銘柄: clf {len(pc):,} / reg {len(pr):,} / "
          f"重複 {len(pc & pr):,} ({len(pc & pr)/max(1, len(pc | pr))*100:.0f}%)")
    print("  -> 重複が小さいほど、2つ並べて見せる価値がある")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
