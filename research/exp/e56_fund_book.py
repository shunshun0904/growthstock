#!/usr/bin/env python3
"""
実験56: 本（ファンダメンタルズ分析の目次）の第2・3章で、取得済みの J-Quants から作れるのに
特徴量にしていなかった12指標（features.GROUPS["fund_book"]、27列）を本番の206列に足して比べる。

運用者の指示（2026-09-26）「4章5章の優先は低くてよいです。2章と3章がほぼ全部網羅したいです」。
対応表は docs/BOOK_INDICATOR_COVERAGE.md（○ の項目）。列の定義は build_dataset.add_book_ratios。

  cash_mcap / cfi_mcap / cff_mcap   現金同等物・投資CF・財務CF ÷ 時価総額
  cfo_yoy_sym / fcf_yoy_sym         営業CF・フリーCF の前年同期比（対称）
  sustainable_growth                ROE × (1 − 配当性向)
  equity_turnover                   売上TTM ÷ 自己資本
  div_growth_sym / div_up           増配率・増配フラグ
  bps_yoy / shares_yoy              BPS・株数の前年同期比（増資・自社株買い）
  acct_ifrs                         会計基準（IFRS/US なら1）
運用者の追加指示（2026-09-26）で各指標に「水準・直近の変化・その前の変化」の3つを持たせた
（_chg1 / _chg2 は前回開示との差、_p1 は1年前の前年同期比、cash_chg*_sym は開示された現金の
対称変化率）。合わせて27列。

腕（実験51 と同じ形。ab_oof.compare）
  T  本番の206列
  B  T + 27列（233列）
  P  対照: 同じ27列を日付内で入れ替えたもの（列の情報を壊して列数だけそろえる）
3モデル（lgbm / xgb / cat）× 種3つ × ずらし 0/2/4か月。判定は §7（LightGBM の PR-AUC が
0.0048 超、xgb / cat も同じ向き）。実収益は運用の規則で記録。

  python3 research/exp/e56_fund_book.py [--shifts 0,2,4] [--seeds 3] [--algos lgbm,xgb,cat]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

BASE_PRESET = F.DEFAULT_PRESET
BOOK_PRESET = "all_plus_prog_listing_book"
NEW_COLS = list(F.GROUPS["fund_book"])
LABELS = {"T": "T 本番の206列", "B": "B T + 本の27列", "P": "P 対照（27列を日付内で入れ替え）"}
PERM_SEED = 20260928
OUT_COL = lab.OUTCOME


def coverage(df: pd.DataFrame) -> None:
    """27列の充足（年ごと）、既存の列との相関、正例率の5等分。"""
    print("\n■ 1. 27列の充足（年ごと、値ありの割合）")
    yr = pd.to_datetime(df["Date"]).dt.year
    tab = df[NEW_COLS].notna().groupby(yr).mean() * 100
    print("  " + f"{'年':<6}" + "".join(f"{c:>18}" for c in NEW_COLS))
    for y, r in tab.iterrows():
        print(f"  {y:<6}" + "".join(f"{v:>17.1f}%" for v in r))
    print("\n■ 2. 既存の列との相関（Spearman、|ρ| ≥ 0.5 だけ）と、5等分ごとの正例率（低→高）")
    base = F.columns(BASE_PRESET)
    for c in NEW_COLS:
        s = df[c]
        rel = []
        if s.notna().sum() > 100 and s.nunique() > 2:
            for b in base:
                if df[b].dtype.kind in "fi" and df[b].notna().sum() > 100:
                    r = s.corr(df[b], method="spearman")
                    if abs(r) >= 0.5:
                        rel.append(f"{b} {r:+.2f}")
            try:
                q = pd.qcut(s.rank(method="first"), 5, labels=False)
                pos = df.groupby(q)["label"].mean() * 100
                bands = " / ".join(f"{v:.1f}%" for v in pos)
            except ValueError:
                bands = "（分けられない）"
        else:
            pos = df.groupby(s)["label"].mean() * 100 if s.notna().any() else pd.Series(dtype=float)
            bands = " / ".join(f"{k:g}: {v:.1f}%" for k, v in pos.items())
        print(f"  {c:<20} 充足 {s.notna().mean()*100:5.1f}%  正例率 {bands}"
              + (f"  相関: {', '.join(rel[:4])}" if rel else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験56: 本の第2・3章の抜け27列")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]

    base = F.columns(BASE_PRESET)
    book = F.columns(BOOK_PRESET)
    assert book == base + NEW_COLS, "B は T の後ろに27列を足しただけの並びにする"
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    miss = [c for c in book if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss}。research/build_dataset.py を回し直してください")

    print("=" * 78)
    print(f"実験56 本の第2・3章の抜け27列（種{len(seeds)}つ・ずらし {shifts}か月）: {len(df):,}件")
    for arm, cols in (("T", base), ("B", book), ("P", book)):
        print(f"  {LABELS[arm]:<32}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    coverage(df)
    fp = permuted(df, NEW_COLS, seed=PERM_SEED)
    arms = {"T": (df, base), "B": (df, book), "P": (fp, book)}
    AB.compare("e56", arms, "T", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
