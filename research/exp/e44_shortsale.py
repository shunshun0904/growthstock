#!/usr/bin/env python3
"""
実験44: 空売り残高報告と市場全体の空売り比率を特徴量に足すと、分離力と運用の規則での
取引がどう変わるか。

腕（データは同じ。列だけが違う）
  A  本番の205列（all_plus）
  B  A + 空売り残高報告4列（all_plus_ss）
       ss_ratio   報告者ごとの最新の持ち高（0.5% 以上）の合計（発行済みに対する%）
       ss_n       その報告者の数
       ss_chg_20  ss_ratio の28暦日前との差
       ss_days    最後の報告が使えるようになってからの暦日数
  C  B + 市場全体の空売り比率2列（all_plus_ss_mkt: short_ratio_mkt / _20）。
     本番の short_ratio は S33=9999（その他）の1行で、市場全体ではなかった
  P  対照。B と同じ4列を、**同じ日の銘柄どうしで入れ替えた**もの（分布と日ごとの
     充足はそのまま、銘柄との対応だけを壊す）。列を足しただけで動く幅（偶然と、
     木が増えた列を使うことによる揺れ）を測る。B の差は P の差と比べて読む。
     手元の試運転では、乱数の4列を足しただけで LightGBM の PR-AUC が
     +0.0035 ± 0.0035（11窓中8窓で上）動いた
共通: 本番のパラメータ（205列で探索したもの。読むだけ）、ブースティング3モデル、
      種3つの平均、窓の切り方3通り（research/exp/ab_oof.py）

時点整合: 報告は公表日（DiscDate）の翌営業日から使う（availability.SHORTSALE_SAME_DAY）。
市場全体の空売り比率は当日（Date）から（availability.RULES の shortratio。当日の夜に
間に合うかは probe で確かめる）。

先に充足を出し、取り込みが届いていなければ止める（全欠測・全ゼロの列で比べない）。

  --shifts 0,2,4  --seeds 3  --algos lgbm,xgb,cat
  結果は research/_data/oof/e44_*。本番の設定には書かない。
  2026-09-24 の結果（B・C とも採用しない）は docs/MODEL_ADOPTION_RULES.md §9。
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402

PRESETS = {"A": "all_plus", "B": "all_plus_ss", "C": "all_plus_ss_mkt"}
LABELS = {"A": "A 本番205列", "B": "B +空売り残高報告", "C": "C +市場全体の空売り比率",
          "P": "P 対照（Bの4列を日付内で入れ替え）"}
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20260924


def permuted(fb, cols, seed: int = PERM_SEED):
    """cols の値を、同じ日付の行どうしで入れ替えた写し（行の並びと他の列はそのまま）。"""
    import numpy as np

    out = fb.copy()
    rng = np.random.default_rng(seed)
    idx = out.groupby("Date").indices
    for c in cols:
        v = out[c].to_numpy().copy()
        for rows in idx.values():
            v[rows] = v[rng.permutation(rows)]
        out[c] = v
    return out
#: これより少ない行にしか報告が無ければ、取り込みが届いていないとみなして止める
MIN_SS_SHARE = 0.02


def coverage(fb: pd.DataFrame) -> None:
    y = fb["Date"].dt.year
    print("\n■ 1. 候補の列の充足（年ごと。ss_ratio>0 = 0.5% 以上の報告がある行）")
    print(f"  {'年':<6}{'行':>7}{'ss_ratio>0':>12}{'ss_n 平均':>10}{'ss_days あり':>13}"
          f"{'市場全体あり':>12}")
    for yr, g in fb.groupby(y):
        print(f"  {yr:<6}{len(g):>7}{(g['ss_ratio'] > 0).mean()*100:>11.1f}%"
              f"{g['ss_n'].mean():>10.2f}{g['ss_days'].notna().mean()*100:>12.1f}%"
              f"{g['short_ratio_mkt'].notna().mean()*100:>11.1f}%")
    share = float((fb["ss_ratio"] > 0).mean())
    if share < MIN_SS_SHARE:
        raise SystemExit(f"ss_ratio>0 の行が {share*100:.2f}% しかない。空売り残高報告の"
                         f"取り込みが届いていない（Update Data Store の shortsale を確かめる）")
    mk = float(fb["short_ratio_mkt"].notna().mean())
    if mk < 0.9:
        raise SystemExit(f"short_ratio_mkt が {mk*100:.1f}% の行にしか無い。業種別の空売り比率の"
                         f"取り直しが届いていない（shortratio を forget して取り直す）")
    # 分離の手がかり: 正例と負例で値がどう違うか（参考。採否はこの後の窓ごとの比較で決める）
    print("\n  正例と負例の平均（参考）")
    for c in ("ss_ratio", "ss_n", "ss_chg_20", "ss_days", "short_ratio_mkt", "short_ratio_mkt_20"):
        pos, neg = fb.loc[fb["label"] == 1, c], fb.loc[fb["label"] == 0, c]
        print(f"  {c:<20}正例 {pos.mean():>8.3f}  負例 {neg.mean():>8.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験44: 空売り残高報告・市場全体の空売り比率")
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
    miss = sorted({c for cs in cols.values() for c in cs if c not in fb.columns})
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:8]}")
    print("=" * 78)
    print(f"実験44 空売り残高報告・市場全体の空売り比率（{len(fb):,}行・種{len(seeds)}つ・"
          f"ずらし {shifts}か月）")
    for arm, p in PRESETS.items():
        print(f"  {LABELS[arm]:<24}{p:<18}{len(cols[arm])}列  指紋 {F.signature(cols[arm])}")
    print("=" * 78)
    coverage(fb)
    ss_cols = F.GROUPS["short_pos"]
    fp = permuted(fb, ss_cols)
    arms = {arm: (fb, cols[arm]) for arm in PRESETS}
    arms["P"] = (fp, cols["B"])
    AB.compare("e44", arms, "A", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
