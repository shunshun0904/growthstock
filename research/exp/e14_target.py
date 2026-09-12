#!/usr/bin/env python3
"""
実験14: 目的変数を運用の目的に合わせる。

背景
----
現行ラベルは「40〜60営業日のうちに 1.2σ(17%前後)上昇したか」で、
**大きく上がったか**を問う問題。正例率 20.9%。

一方、運用の目的は「安定的な少額の利益」で、手仕舞いは
「20営業日で+なら決済、でなければ40営業日まで持つ」という柔軟売却。
実測でこのルールは固定売却より優れていた（clf 上位10%）。

  売却ルール       平均      中央    勝率   25%分位
  20日固定       +1.75%   +1.06%  59.0%   -2.3%
  40日固定       +3.21%   +2.54%  62.3%   -3.1%
  柔軟(20->40)   +2.18%   +2.30%  71.7%   -1.3%

回帰（実収益を直接当てる）は平均では勝つが、中央値は半分・勝率は8pt低く・
下振れは2倍で、目的に合わないため採用しない。ここでやるのは分類器のまま
**ラベルの定義だけを目的に合わせる**こと。

試す目的変数
-----------
  label       現行。1.2σ以上上昇（正例率 20.9%）
  up_flex     柔軟売却で+（61.0%）          <- 運用の目的そのもの
  up_40       40営業日後に+（49.9%）
  up5_40      40営業日後に+5%以上（33.1%）  <- 「少額の利益」を明示した版

評価
----
指標は柔軟売却の実収益（ret_flex）で測る。運用がそれなので。
勝率・中央値・下振れ(25%分位)も一緒に出す。平均だけ見ると回帰のような
宝くじ型が有利に見えてしまう。

判定は実験11 のノイズ床に従い、窓平均 ± 窓SD/√9 の合成SEで見る。
t値は9窓では種だけで2.50〜5.79 動くので使わない。
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
OUTCOME = "ret_flex"


def picks(o: pd.DataFrame, pct: float = 90) -> pd.DataFrame:
    out = []
    for f in sorted(o["fold"].unique()):
        ref = o.loc[o["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        t = float(np.percentile(ref, pct))
        cur = o[o["fold"] == f]
        out.append(cur[cur["score"] > t])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    r40 = pd.to_numeric(df["ret_o1_40"], errors="coerce")
    fx = pd.to_numeric(df[OUTCOME], errors="coerce")
    df["up_flex"] = np.where(fx.notna(), (fx > 0).astype(float), np.nan)
    df["up_40"] = np.where(r40.notna(), (r40 > 0).astype(float), np.nan)
    df["up5_40"] = np.where(r40.notna(), (r40 > 0.05).astype(float), np.nan)

    targets = ["label", "up_flex", "up_40", "up5_40"]
    for t in targets:
        print(f"  {t:<10} 正例率 {df[t].mean()*100:>5.1f}% / 欠損 {df[t].isna().mean()*100:.1f}%")
    print()

    results = {}
    for t in targets:
        p = os.path.join(OOF_DIR, f"tgt_{t}.parquet")
        if os.path.exists(p):
            o = lab.attach_outcomes(pd.read_parquet(p), df)
            results[t] = lab.Result(t, o, lab.metrics(o))
            print(f"  {t:<10} 保存済みを読む")
            continue
        r = lab.run_multi(df, lambda s: lab.lgbm(seed=s), seeds=SCREEN_SEEDS,
                          cols=cols, name=t, target=t)
        r.oof.to_parquet(p, index=False)
        results[t] = r
        m = lab.threshold_edge(r.oof, outcome=OUTCOME)
        print(f"  {t:<10} 完了  柔軟売却での優位 {m['thr_lift']:+.2f}pt / "
              f"窓平均 {m['thr_fold_mean']:+.2f}pt (SD {m['thr_fold_sd']:.2f})")

    print()
    print(f"=== 柔軟売却（{OUTCOME}）で測った、しきい値運用(>90)の成績 ===")
    print(f"  {'目的変数':<10}{'優位':>9}{'窓平均':>10}{'窓SD':>8}{'取引数':>8}"
          f"{'平均':>9}{'中央':>9}{'勝率':>8}{'25%分位':>10}{'最悪の窓':>11}")
    stats = {}
    for t, r in results.items():
        m = lab.threshold_edge(r.oof, outcome=OUTCOME)
        p_ = picks(r.oof)
        e = pd.to_numeric(p_[OUTCOME], errors="coerce").dropna()
        stats[t] = m
        print(f"  {t:<10}{m['thr_lift']:>+8.2f}pt{m['thr_fold_mean']:>+9.2f}pt"
              f"{m['thr_fold_sd']:>8.2f}{m['thr_n']:>8,}{e.mean()*100:>+8.2f}%"
              f"{e.median()*100:>+8.2f}%{(e>0).mean()*100:>7.1f}%"
              f"{e.quantile(.25)*100:>+9.1f}%{m['thr_worst']:>+10.2f}pt")
    uni = results["label"].oof
    uni = uni[uni["fold"] > uni["fold"].min()]
    e = pd.to_numeric(uni[OUTCOME], errors="coerce").dropna()
    print(f"  {'(全候補)':<10}{'':>10}{'':>11}{'':>8}{len(uni):>8,}{e.mean()*100:>+8.2f}%"
          f"{e.median()*100:>+8.2f}%{(e>0).mean()*100:>7.1f}%{e.quantile(.25)*100:>+9.1f}%")

    print()
    print("=== 判定（窓平均。足切りは合成SE。現行 label を基準に）===")
    ref = stats["label"]
    se_r = ref["thr_fold_sd"] / np.sqrt(max(1, ref["thr_folds"]))
    print(f"  基準 label: 窓平均 {ref['thr_fold_mean']:+.2f}pt ± {se_r:.2f}")
    print()
    print(f"  {'目的変数':<10}{'窓平均':>10}{'標準誤差':>10}{'差':>10}{'差/合成SE':>11}{'判定':>10}")
    for t, m in stats.items():
        if t == "label":
            continue
        se = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
        d = m["thr_fold_mean"] - ref["thr_fold_mean"]
        z = d / np.sqrt(se ** 2 + se_r ** 2)
        v = "採用可" if z > 2 else ("要確認" if z > 1 else "差なし")
        print(f"  {t:<10}{m['thr_fold_mean']:>+9.2f}pt{se:>10.2f}{d:>+9.2f}pt{z:>11.2f}{v:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
