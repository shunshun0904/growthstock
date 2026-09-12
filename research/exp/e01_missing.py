#!/usr/bin/env python3
"""
実験01: 欠損そのものを特徴量にする。

動機
----
財務系118列の欠損率は平均15.7%、guidance は57.9%、cashflow は52.2%。
LightGBM は NaN を「どちらの枝に送るか」を列ごとに学習するが、
**いくつ欠けているか**という集計量は見ていない。

株式だけに絞ると、欠損数と成績は単調に効いていた（ETF は財務が
118列全欠損・正例率32%なので、全行で見ると U 字に見えて逆方向に誤読する）。

    欠損 1列  正例率 21.7%  実収益 +3.19%
    欠損41列  正例率 17.7%  実収益 +0.31%

「開示が欠けている企業ほど成績が悪い」なら、集計量を直接渡せば効くはず。

試す形
------
  base        現行
  +count      財務系の欠損数（1列）
  +groups     グループ別の欠損率（8列）
  +both       両方
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

#: 欠損を数える対象のグループ（財務・開示まわり）
FUND_GROUPS = ("fund_level", "fund_growth", "fund_lag", "fund_trend", "fund_streak",
               "valuation", "cashflow", "guidance", "dividend", "efficiency",
               "progress", "turnaround")


def add_missing_features(df: pd.DataFrame, cols) -> tuple:
    """欠損の集計量を列として足す。行ごとの計算なのでリークしない。"""
    fund = [c for c in cols if F.group_of(c) in FUND_GROUPS]
    out = df.copy()
    out["miss_n_fund"] = out[fund].isna().sum(axis=1)
    added = ["miss_n_fund"]
    for g in FUND_GROUPS:
        gc = [c for c in cols if F.group_of(c) == g]
        if not gc:
            continue
        out[f"miss_r_{g}"] = out[gc].isna().mean(axis=1)
        added.append(f"miss_r_{g}")
    return out, added


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    df, added = add_missing_features(df, cols)
    count_col = ["miss_n_fund"]
    group_cols = [c for c in added if c.startswith("miss_r_")]

    runs = {
        "base": cols,
        "+count": cols + count_col,
        "+groups": cols + group_cols,
        "+both": cols + added,
    }
    results = {}
    for name, use in runs.items():
        results[name] = lab.run(df, lab.lgbm(), cols=use, name=name)
        print(f"  {name:<10} 完了（{len(use)}列）")

    print()
    print(lab.table(results))
    print()
    base = results["base"]
    for name, r in results.items():
        if name == "base":
            continue
        c = lab.compare(base, r)
        lo, hi = c["end_ci"]
        print(f"{name:<10} 上位5%収益 {c['end_a']:+.2f}% -> {c['end_b']:+.2f}% "
              f"({c['end_diff']:+.2f}pt [{lo:+.2f}, {hi:+.2f}] 改善確率 {c['p_better']*100:.0f}%)  "
              f"PR-AUC {c['pr_auc_diff']:+.4f}  日付内AUC {c['auc_in_day_diff']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
