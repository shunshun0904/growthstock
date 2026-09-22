#!/usr/bin/env python3
"""
実験22: EDINET の年次特徴量（research/edinet_features.py）の検出力を
実験20 と同じ方法で測る。

EDINET の取得は 1社1リクエスト・日 85社 なので、揃った範囲で回す。
行の充足率（母集団のうち EDINET の y0 が付いた割合）を必ず先に出す。
充足が低いうちは「件数不足」が並ぶ。それは結果ではなく待ちである。

  $ python3 research/exp/e22_edinet_screen.py [--group core|detail|ratio|ratio_chg|all]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import edinet_features as EF  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from e20_annual_trajectory import OUTCOME, screen  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OUT = os.path.join(lab.DATA_DIR, "oof", "e22_screen.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    ap.add_argument("--fin", default=EF.FIN)
    args = ap.parse_args(argv)

    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    if not os.path.exists(args.fin):
        raise SystemExit(f"{args.fin} がありません（fetch-edinetdb.yml の成果物）")
    fin = EF.load_fin(args.fin)
    panel = EF.annual_panel(fin)
    df = EF.attach(frame, EF.feature_frame(panel))
    have = df["ed_fiscal_year"].notna()
    n_codes = fin[EF.KEY].nunique()
    print(f"母集団 {len(frame):,}件 / EDINET 取得済み {n_codes:,}社 / "
          f"y0 が付いた行 {have.sum():,}件（{have.mean()*100:.1f}%）"
          f"/ 銘柄 {df.loc[have, 'Code'].nunique():,}")
    if have.sum() < 2000:
        print("  ※ 充足が低い。窓ごとの検定に足りる目安は 2,000行（実験20 の1/10）")

    cols = EF.columns(args.group)
    folds = WF.make_folds(df["Date"], min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    windows = [(np.datetime64(f.test_start), np.datetime64(f.test_end)) for f in folds]
    print(f"特徴量 {args.group}（{len(cols)}列）/ 評価窓 {len(windows)}本 / 物差し {OUTCOME}\n")

    res = screen(df, cols, windows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== 検出力（|z| の順。z は上位10%の実収益の超過 / 窓SE）===")
    print(f"  {'特徴量':<30}{'充足':>6}{'AUC併合':>9}{'AUC窓':>8}{'>0.5':>6}"
          f"{'超過pt':>8}{'SE':>6}{'z':>7}{'正の窓':>7}")
    short = 0
    for _, r in res.iterrows():
        if np.isnan(r.get("edge_z", np.nan)):
            short += 1
            continue
        print(f"  {r['feature']:<30}{r['coverage']*100:>5.0f}%{r['auc_pooled']:>9.3f}"
              f"{r['auc_win']:>8.3f}{r['auc_win_gt05']:>3}/{r['n_win']:<2}"
              f"{r['edge_pt']:>+8.2f}{r['edge_se']:>6.2f}{r['edge_z']:>+7.2f}"
              f"{r['edge_win_pos']:>4}/{r['n_win']:<2}")
    if short:
        print(f"  （件数不足で測れない列 {short}本）")
    hits = res[res["abs_z"] > 2]
    print(f"\n|z|>2: {len(hits)}本 / {len(res)}本（偶然でも 5% ≈ {len(res)*0.05:.1f}本は超える）")
    print(f"記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
