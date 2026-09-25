#!/usr/bin/env python3
"""
実験46: 進捗期待の新しい定義（画面の8軸と同じ。前年同期の進捗で割った比率）を
モデルの特徴量にすると、分離力と運用の規則での取引がどう変わるか。

運用者の指示（2026-09-25）で、画面の「進捗期待」を
「進捗率 ÷ 前年同期の進捗」の、四半期ごとの過去の順位に変えた（9ff76ee）。
同じ直しを特徴量にも入れる候補。本番の progress_vs_base（進捗率 − Q×25）は、
下期に利益が偏る会社をいつも「遅れ」と見て、四半期ごとの幅の違い（第1四半期は
第3四半期より何倍も広い）もそろえていない。

腕（データは同じ。列だけが違う）
  A  本番の205列（all_plus）
  B  置き換え: progress_vs_base の位置に progress_pct（all_plus_prog_swap、205列）
  C  追加: A + progress_ratio + progress_pct（all_plus_prog、207列）
  P  対照: C の2列を**同じ日の銘柄どうしで入れ替えた**もの。列を足しただけで動く幅
     （実験44 では中身の無い4列で LightGBM の PR-AUC が +0.0022 動いた）を測る
共通: 本番のパラメータ（読むだけ。e27_timing_multi.prod_params。記録した run では
      LightGBM は 205列の探索結果がまだ無く、2026-09-20 に all の151列で探索した結果）、
      ブースティング3モデル、種3つの平均、窓の切り方3通り（research/exp/ab_oof.py）

採否（docs/MODEL_ADOPTION_RULES.md §7）: LightGBM の PR-AUC の改善が 0.0048 を超え、
XGBoost / CatBoost も同じ向き。超えたら探索し直し（B2）でも同じ向きかを確かめてから本番へ。

時点整合（build_dataset.seasonal_progress / progress_percentile）
  ・進捗率・比率は決算の開示日から（ほかの決算の列と同じ結合）
  ・前年同期・前年通期の値は、その開示日までに出ていた版（後日の訂正を先に見ない）
  ・順位は、その開示日より前の開示だけで付ける（全期間の分布で順位を付けない）

  --shifts 0,2,4  --seeds 3  --algos lgbm,xgb,cat
  結果は research/_data/oof/e46_*。本番の設定には書かない。

2026-09-25 に運用者の決定で B（置き換え）と上場からの年数を本番に入れた
（features.DEFAULT_PRESET = all_plus_prog_listing）。記録した結果は 86c7b4d までの版で
回したもの。いまは lab.frame() の母集団も直してある（build_dataset.GENERAL_MARKET_START）
ので、回し直すと行がわずかに変わる（実験47 の1回目では 21,867 -> 21,864行）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

PRESETS = {"A": "all_plus", "B": "all_plus_prog_swap", "C": "all_plus_prog"}
LABELS = {"A": "A 本番205列", "B": "B 置き換え（順位）", "C": "C 追加（比率・順位）",
          "P": "P 対照（Cの2列を日付内で入れ替え）"}
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20260925


def coverage(fb: pd.DataFrame) -> None:
    """新しい列の充足と、単独での見分け（参考。採否は窓ごとの比較で決める）。"""
    from sklearn.metrics import roc_auc_score

    y = fb["Date"].dt.year
    print("\n■ 1. 列の充足（年ごと）")
    print(f"  {'年':<6}{'行':>7}{'旧 vs_base':>11}{'比率':>8}{'順位':>8}"
          f"{'前年同期':>9}{'下限':>6}{'Q×25%':>7}")
    for yr, g in fb.groupby(y):
        b = g["progress_basis"]
        print(f"  {yr:<6}{len(g):>7}{g['progress_vs_base'].notna().mean()*100:>10.1f}%"
              f"{g['progress_ratio'].notna().mean()*100:>7.1f}%"
              f"{g['progress_pct'].notna().mean()*100:>7.1f}%"
              f"{(b == 'seasonal').mean()*100:>8.1f}%{(b == 'floor').mean()*100:>5.1f}%"
              f"{(b == 'linear').mean()*100:>6.1f}%")
    old = fb["progress_vs_base"].notna()
    new = fb["progress_ratio"].notna()
    # 本決算（4Q）には旧も新も無い。旧があって新が無い行が多ければ作りを疑う
    lost = int((old & ~new).sum())
    print(f"  旧の列はあるが比率が無い行: {lost:,}（全 {len(fb):,}）")
    if lost > 0.01 * max(1, int(old.sum())):
        raise SystemExit("比率の欠けが旧の列より1%超多い。build_dataset.seasonal_progress を確かめる")

    print("\n■ 2. 単独での見分け（その年の行の中での AUC。参考）")
    print(f"  {'年':<6}{'旧 vs_base':>11}{'比率':>8}{'順位':>8}")
    for yr, g in fb.groupby(y):
        row = []
        for c in ("progress_vs_base", "progress_ratio", "progress_pct"):
            m = g[c].notna()
            yy = g.loc[m, "label"].astype(int)
            row.append(roc_auc_score(yy, g.loc[m, c]) if yy.nunique() == 2 else np.nan)
        print(f"  {yr:<6}" + "".join(f"{v:>9.3f}" for v in row))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験46: 進捗期待の新しい定義を特徴量に")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]

    fb = lab.frame()
    fb["Date"] = pd.to_datetime(fb["Date"])
    fb["Code"] = fb["Code"].astype(str)
    fb = fb.dropna(subset=["label"]).reset_index(drop=True)
    cols = {arm: F.columns(p) for arm, p in PRESETS.items()}
    miss = sorted({c for cs in cols.values() for c in cs if c not in fb.columns}
                  | ({"progress_basis"} - set(fb.columns)))
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:8]}")
    print("=" * 78)
    print(f"実験46 進捗期待の新しい定義（{len(fb):,}行・種{len(seeds)}つ・ずらし {shifts}か月）")
    for arm, p in PRESETS.items():
        print(f"  {LABELS[arm]:<24}{p:<20}{len(cols[arm])}列  指紋 {F.signature(cols[arm])}")
    print("=" * 78)
    coverage(fb)
    fp = permuted(fb, F.GROUPS["progress_seasonal"], seed=PERM_SEED)
    arms = {arm: (fb, cols[arm]) for arm in PRESETS}
    arms["P"] = (fp, cols["C"])
    AB.compare("e46", arms, "A", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
