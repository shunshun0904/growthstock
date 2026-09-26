#!/usr/bin/env python3
"""
実験51: vol_20d を「どんな種類のボラか」に分解した6列（features.GROUPS["vol_factors"]）を
本番の206列に足すと、分離力と運用の規則での取引がどう変わるか。

運用者の依頼（2026-09-26）「vol_20d の特徴量をもっと2因子、3因子と分解したほうが良さそう
ですね」→「①〜⑤で実験51を進めてください」。
背景（実験50）: vol_20d は特徴量であると同時にラベルの到達しきい値（1.2σ×√20）の σ でも
あるので、3モデルとも SHAP の筆頭は vol_20d で、大外れ（正例なのに下位10% / 負例なのに
上位5%）はボラの帯にほぼ決まっていた。

列（build_dataset.add_vol_factors / attach_vol_vs_market。すべて当日までの値）
  ① vol_rel_long    vol_20d ÷ vol_120d          いま異常に荒いか、もともと荒い銘柄か
  ② vol_rel_mkt     vol_20d ÷ topix_vol_20      相場全体が荒いか、この銘柄だけか
     vol_rel_sector vol_20d ÷ 業種指数の20日ボラ
  ③ vol_updown      上方半分散の平方根 ÷ 下方   上に跳ねて荒いか、投げられて荒いか
  ④ vol_rel_short   vol_5d ÷ vol_20d            直前に膨らんだか、落ち着いてきたか
  ⑤ vol_gap_ratio   ギャップの σ ÷ 日中レンジの σ  材料で飛んだか、日中の売買で動いたか

腕
  T  本番の206列（all_plus_prog_listing）          ← 比べる基準
  V  T + 6列（all_plus_prog_listing_vol、212列）
  P  対照: V の6列を同じ日の銘柄どうしで入れ替えたもの（列を足しただけの幅）
共通: 本番のパラメータ（読むだけ）、ブースティング3モデル、種3つの平均、窓の切り方3通り

採否（docs/MODEL_ADOPTION_RULES.md §7）: V − T の LightGBM の PR-AUC が 0.0048 を超え、
XGBoost / CatBoost も同じ向きで、対照 P との差でも上回ること。

  --shifts 0,2,4  --seeds 3  --algos lgbm,xgb,cat
  結果は research/_data/oof/e51_*。本番の設定には書かない。
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

BASE_PRESET = "all_plus_prog_listing"
VOL_PRESET = "all_plus_prog_listing_vol"
NEW_COLS = list(F.GROUPS["vol_factors"])
LABELS = {"T": "T 本番の206列", "V": "V T + ボラの分解6列", "P": "P 対照（6列を日付内で入れ替え）"}
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20260927


def coverage(df: pd.DataFrame) -> None:
    y = pd.to_datetime(df["Date"]).dt.year
    print("\n■ 1. 6列の充足（年ごと、値ありの割合）")
    print("  " + f"{'年':<6}{'行':>7}" + "".join(f"{c:>16}" for c in NEW_COLS))
    for yr, g in df.groupby(y):
        print("  " + f"{yr:<6}{len(g):>7}" + "".join(f"{g[c].notna().mean()*100:>15.1f}%" for c in NEW_COLS))

    print("\n■ 2. vol_20d との相関（Spearman）と、5等分ごとの正例率・正例の実収益の中央値")
    print("  水準（vol_20d）と相関が強い列は、分解になっていない")
    r = df[OUT_COL].to_numpy(dtype=float)
    for c in NEW_COLS:
        v = df[c]
        rho = v.corr(df["vol_20d"], method="spearman")
        rho_r = v.corr(df["ret_20d"], method="spearman")
        bins = pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop")
        parts = []
        for b, g in df.groupby(bins, observed=True):
            pos = g["label"] == 1
            parts.append(f"{g['label'].mean()*100:.1f}%/{np.nanmedian(r[g.index][pos.to_numpy()])*100:+.0f}%")
        print(f"  {c:<15} ρ(vol_20d) {rho:+.2f}  ρ(ret_20d) {rho_r:+.2f}  中央値 {v.median():.2f}  "
              f"5等分（低→高）: " + " | ".join(parts))

    print("\n■ 3. 6列どうしの相関（Spearman、|ρ| ≥ 0.5 だけ）")
    m = df[NEW_COLS].corr(method="spearman")
    pairs = [(a, b, m.loc[a, b]) for i, a in enumerate(NEW_COLS) for b in NEW_COLS[i + 1:]
             if abs(m.loc[a, b]) >= 0.5]
    print("  " + (" / ".join(f"{a}×{b} {v:+.2f}" for a, b, v in pairs) or "0.5 以上の組なし"))


OUT_COL = lab.OUTCOME


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験51: ボラの分解")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]

    base = F.columns(BASE_PRESET)
    vol = F.columns(VOL_PRESET)
    assert vol == base + NEW_COLS, "V は T の後ろに6列を足しただけの並びにする"
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    miss = [c for c in vol if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss}。research/build_dataset.py を回し直してください")

    print("=" * 78)
    print(f"実験51 ボラの分解（種{len(seeds)}つ・ずらし {shifts}か月）: {len(df):,}件")
    for arm, cols in (("T", base), ("V", vol), ("P", vol)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    coverage(df)
    fp = permuted(df, NEW_COLS, seed=PERM_SEED)
    arms = {"T": (df, base), "V": (df, vol), "P": (fp, vol)}
    AB.compare("e51", arms, "T", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
