#!/usr/bin/env python3
"""
実験12: ラベルを迂回して、実収益を直接当てる。

なぜこれが本題か
---------------
現行のラベルは「40〜60営業日のうちに 1.2σ 上昇したか」の二値。
しきい値が vol_20d に比例するので、ボラが高い局面ではハードルが上がる。
実測（実験10）でその副作用がはっきり出た。

  決算からの経過  日次ボラ  必要上昇率  正例率  実収益
  0-5日          2.59%    24.1%     16.0%  +3.91%
  21-45日        1.83%    17.0%     19.9%  +0.71%
  71-95日        1.68%    15.6%     24.2%  +2.81%

決算直後は 24.1% の上昇を要求されるのに、決算周期の終盤なら 15.6% で足りる。
**同じ値動きなら決算直後は1.5倍のハードル**。そして実収益がいちばん良いのは
その決算直後（+3.91%）。ラベルが、最も儲かる局面を選んで罰している。

回帰なら、この歪みを通さずに実収益そのものを狙える。
これで「ラベルがボトルネックか、そもそも予測が難しいのか」が切り分かる。

比べるもの
---------
  clf_lgbm      現行。ラベル（1.2σ二値）を当てる分類器
  reg_huber     実収益を当てる LightGBM（huber）
  reg_l2        同じく二乗誤差。裾の重さがどう出るかの対照
  reg_rank      実収益の**窓内パーセンタイル**を当てる。水準ではなく順位を
                狙うので、年ごとの相場水準の違いに引っ張られない
  reg_ridge     実収益を当てる線形回帰（標準化 + 中央値補完 + 欠損指示子）

指標は共通のしきい値運用（翌営業日の寄り買い / 40営業日後の5日平均終値売り）。
分類か回帰かでスコアの意味は変わるが、しきい値は窓ごとに過去分布から
決めるので、スケールの違いは吸収される。

判定の足切り
-----------
実験11 のノイズ床（同一設定・種5個）
  しきい値優位  レンジ 0.484pt
  窓平均        レンジ 0.143pt  <- 最も解像度が高い
  t値           レンジ 3.294    <- 9窓では使えない（種だけで2.50〜5.79）

窓平均で判定する。ただし窓平均自身の標準誤差はモデルの窓SD/√9 なので、
窓SD の大きいモデルでは足切りも広く取る必要がある。両方併記する。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TARGET = lab.OUTCOME          # ret_o1_40
RANK_TARGET = "ret_rank_40"


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")

    # 窓内パーセンタイルの目的変数。fold ではなく年で割る
    # （fold は評価用の区切りで、訓練時には使えない情報）
    yr = pd.to_datetime(df["Date"]).dt.year
    df[RANK_TARGET] = df.groupby(yr)[TARGET].rank(pct=True)

    e = pd.to_numeric(df[TARGET], errors="coerce")
    print(f"目的変数 {TARGET}: 欠損 {e.isna().mean()*100:.1f}% / "
          f"平均 {e.mean()*100:+.2f}% / 中央 {e.median()*100:+.2f}%")
    print(f"  裾: 5%分位 {e.quantile(.05)*100:+.1f}% / "
          f"95%分位 {e.quantile(.95)*100:+.1f}% / 歪度 {e.skew():.2f}")
    print()

    runs = {
        "clf_lgbm": (lambda s: lab.lgbm(seed=s), "label"),
        "reg_huber": (lambda s: lab.lgbm_reg(seed=s, objective="huber"), TARGET),
        "reg_l2": (lambda s: lab.lgbm_reg(seed=s, objective="l2"), TARGET),
        "reg_rank": (lambda s: lab.lgbm_reg(seed=s, objective="l2"), RANK_TARGET),
        "reg_ridge": (lambda s: lab.ridge_reg(seed=s), TARGET),
    }
    results = {}
    for name, (mk, tgt) in runs.items():
        p = os.path.join(OOF_DIR, f"reg_{name}.parquet")
        if os.path.exists(p):
            o = lab.attach_outcomes(pd.read_parquet(p), df)
            results[name] = lab.Result(name, o, lab.metrics(o))
            print(f"  {name:<12} 保存済みを読む")
            continue
        r = lab.run_multi(df, mk, seeds=SCREEN_SEEDS, cols=cols, name=name,
                          target=tgt)
        r.oof.to_parquet(p, index=False)
        results[name] = r
        m = r.metrics
        print(f"  {name:<12} 完了  優位 {m['thr_lift']:+.2f}pt / "
              f"窓平均 {m['thr_fold_mean']:+.2f}pt (SD {m['thr_fold_sd']:.2f})")

    print()
    print(lab.table(results))

    print()
    print("=== 判定（窓平均で見る。足切りは窓SD/√9 から出す）===")
    ref = results["clf_lgbm"].metrics
    n_f = max(1, ref["thr_folds"])
    print(f"  基準 clf_lgbm: 窓平均 {ref['thr_fold_mean']:+.2f}pt "
          f"± {ref['thr_fold_sd']/np.sqrt(n_f):.2f} (窓SD {ref['thr_fold_sd']:.2f})")
    print()
    print(f"  {'モデル':<12}{'窓平均':>10}{'標準誤差':>10}{'基準との差':>12}"
          f"{'差/合成SE':>11}{'判定':>10}")
    for name, r in results.items():
        if name == "clf_lgbm":
            continue
        m = r.metrics
        se = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
        se_ref = ref["thr_fold_sd"] / np.sqrt(n_f)
        d = m["thr_fold_mean"] - ref["thr_fold_mean"]
        z = d / np.sqrt(se ** 2 + se_ref ** 2)
        v = "採用可" if z > 2 else ("要確認" if z > 1 else "差なし")
        print(f"  {name:<12}{m['thr_fold_mean']:>+9.2f}pt{se:>10.2f}"
              f"{d:>+11.2f}pt{z:>11.2f}{v:>11}")
    print()
    print("  ※ 合成SE = sqrt(SE_model^2 + SE_base^2)。窓が9しかないので")
    print("     窓SD の大きいモデルは差が大きくても判定できない")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
