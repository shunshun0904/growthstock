#!/usr/bin/env python3
"""
決算特徴量がどこで失われているかを調べる。

きっかけ:
  ウォークフォワードで決算特徴量が全局面で基準に負けた（0勝9敗）。
  しかしデータセットの欠測率を見ると ROE_chg は 94.4%、
  eps_growth_chg1 は 92.9% が欠測だった。
  つまり「決算に予測力が無い」ではなく
  「ほとんど値が入っていない列で測っていた」可能性がある。
  結論を出す前に、どの段階で値が落ちているかを特定する。

見るところ:
  1. 生の決算開示（/fins/summary）に各項目がどれだけ入っているか
  2. 四半期差分と前年同期比を作った後にどれだけ残るか
  3. ラグ列（q1/q2）と変化量（chg/slope）を作った後にどれだけ残るか
  4. 最終サンプルに結合した後にどれだけ残るか

3 で大きく落ちるなら原因は「値が疎な列を開示単位でずらしている」こと。
たとえば ROE が通期開示にしか入っていない場合、
直前の開示（四半期）を shift(1) で参照しても NaN にしかならない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import DATA_DIR, load_parts, quarterize_panel  # noqa: E402

AXES = ["eps_growth", "sales_growth", "ROE", "op_margin"]
RAW_FIELDS = ["Sales", "OP", "NP", "EPS", "Eq", "ROE", "FOP"]


def _rate(s: pd.Series) -> float:
    return float(s.notna().mean() * 100.0)


def raw_coverage(fins: pd.DataFrame) -> List[Dict]:
    """生の開示に各項目がどれだけ入っているか。開示種別ごとにも見る。"""
    out = []
    for f in RAW_FIELDS:
        if f not in fins.columns:
            out.append({"field": f, "present_pct": 0.0, "note": "列が存在しない"})
            continue
        row = {"field": f, "present_pct": _rate(fins[f]), "note": ""}
        # 開示種別ごとの充足率。通期にしか入らない項目を見つけるため
        if "CurPerType" in fins.columns:
            by = fins.groupby("CurPerType")[f].apply(_rate).round(1)
            row["by_period"] = {str(k): float(v) for k, v in by.items()}
        out.append(row)
    return out


def derived_coverage(q: pd.DataFrame) -> List[Dict]:
    """四半期化・ラグ・変化量の各段階でどれだけ残るか。"""
    out = []
    for a in AXES:
        stage = {"axis": a}
        for suffix in ("q0", "q1", "q2", "chg1", "chg", "slope"):
            col = f"{a}_{suffix}"
            stage[suffix] = round(_rate(q[col]), 1) if col in q.columns else None
        out.append(stage)
    return out


def lag_loss(q: pd.DataFrame) -> List[Dict]:
    """
    ラグでどれだけ落ちるかを分解する。

    「q0 が入っている行のうち、q1 も入っている割合」を見れば、
    値が疎なせいでラグが取れないのかが分かる。
    開示単位で shift しているので、間に値の無い開示が挟まると切れる。
    """
    out = []
    for a in AXES:
        q0, q1, q2 = f"{a}_q0", f"{a}_q1", f"{a}_q2"
        if not all(c in q.columns for c in (q0, q1, q2)):
            continue
        have0 = q[q0].notna()
        n0 = int(have0.sum())
        out.append({
            "axis": a,
            "n_with_q0": n0,
            "q1_given_q0_pct": round(float(q.loc[have0, q1].notna().mean() * 100), 1)
            if n0 else None,
            "q2_given_q0_pct": round(float(q.loc[have0, q2].notna().mean() * 100), 1)
            if n0 else None,
        })
    return out


def build_report(raw: List[Dict], derived: List[Dict], lag: List[Dict],
                 sample: Optional[List[Dict]], n_fins: int, n_q: int) -> str:
    lines = [
        "# 決算特徴量のカバレッジ",
        "",
        "`research/fundamental_coverage.py` の出力。**実測値のみ**を記載する。",
        "",
        "ウォークフォワードで決算特徴量は0勝9敗だったが、",
        "データセットの欠測率は `ROE_chg` 94.4% / `eps_growth_chg1` 92.9% だった。",
        "「予測力が無い」のか「値が入っていない」のかを切り分ける。",
        "",
        f"- 生の決算開示: {n_fins:,}行",
        f"- 四半期化・実績値のみに絞った後: {n_q:,}行",
        "",
        "## 1. 生の開示での充足率",
        "",
        "| 項目 | 全体 | 開示種別ごと |",
        "| --- | ---: | --- |",
    ]
    for r in raw:
        by = r.get("by_period")
        detail = " / ".join(f"{k}: {v}%" for k, v in sorted(by.items())) if by else r["note"]
        lines.append(f"| `{r['field']}` | {r['present_pct']:.1f}% | {detail} |")

    lines += [
        "",
        "**ある項目が特定の開示種別にしか入っていない場合、そこが原因**。",
        "四半期開示に値が無い項目を開示単位でラグさせても NaN にしかならない。",
        "",
        "## 2. 派生列の充足率（四半期パネル上）",
        "",
        "| 軸 | q0 | q1 | q2 | chg1 | chg | slope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in derived:
        cells = " | ".join("—" if r[k] is None else f"{r[k]}%"
                           for k in ("q0", "q1", "q2", "chg1", "chg", "slope"))
        lines.append(f"| `{r['axis']}` | {cells} |")

    lines += [
        "",
        "## 3. ラグで失われる割合",
        "",
        "`q0` が入っている行のうち、ラグ列も入っている割合。",
        "ここが低ければ、値が疎なため開示単位のラグが取れていない。",
        "",
        "| 軸 | q0がある行 | うちq1もある | うちq2もある |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in lag:
        lines.append(f"| `{r['axis']}` | {r['n_with_q0']:,} | "
                     f"{r['q1_given_q0_pct']}% | {r['q2_given_q0_pct']}% |")

    if sample:
        lines += [
            "",
            "## 4. 最終サンプルでの充足率",
            "",
            "フィルタと開示日ベースの結合を通した後。実際に学習で使われる値。",
            "",
            "| 特徴量 | 充足率 |",
            "| --- | ---: |",
        ]
        for r in sample:
            lines.append(f"| `{r['feature']}` | {r['present_pct']:.1f}% |")

    lines += [
        "",
        "## 読み方",
        "",
        "- 1 で特定の開示種別にしか入っていない項目があれば、そこが根本原因",
        "- 3 が低い軸は、ラグの取り方（開示単位の shift）が合っていない。",
        "  値のある開示だけを詰めてラグを取れば回復する",
        "- 4 が低いまま学習していた場合、「決算に予測力が無い」という結論は保留になる。",
        "  ほとんど欠測の列で測っていたことになるため",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="決算特徴量のカバレッジ調査")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_FUNDAMENTAL_COVERAGE.md"))
    args = ap.parse_args(argv)

    fins = load_parts("fins", args.data_dir)
    q = quarterize_panel(fins)

    print("\n[1] 生の開示での充足率")
    raw = raw_coverage(fins)
    for r in raw:
        by = r.get("by_period")
        print(f"  {r['field']:<8} {r['present_pct']:5.1f}%"
              + (f"  {by}" if by else f"  {r['note']}"))

    print("\n[2] 派生列の充足率")
    derived = derived_coverage(q)
    for r in derived:
        print(f"  {r['axis']:<14} q0 {r['q0']}% / q1 {r['q1']}% / q2 {r['q2']}% "
              f"/ chg {r['chg']}%")

    print("\n[3] ラグで失われる割合")
    lag = lag_loss(q)
    for r in lag:
        print(f"  {r['axis']:<14} q0={r['n_with_q0']:,} "
              f"-> q1 {r['q1_given_q0_pct']}% / q2 {r['q2_given_q0_pct']}%")

    sample = None
    if os.path.exists(args.dataset):
        df = pd.read_parquet(args.dataset)
        cols = [c for a in AXES for c in
                (f"{a}_q0", f"{a}_q1", f"{a}_q2", f"{a}_chg1", f"{a}_chg", f"{a}_slope")
                if c in df.columns]
        sample = [{"feature": c, "present_pct": _rate(df[c])} for c in cols]
        print("\n[4] 最終サンプルでの充足率（低い順に5件）")
        for r in sorted(sample, key=lambda r: r["present_pct"])[:5]:
            print(f"  {r['feature']:<22} {r['present_pct']:5.1f}%")

    body = build_report(raw, derived, lag, sample, len(fins), len(q))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    with open(os.path.join(DATA_DIR, "fundamental_coverage.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"raw": raw, "derived": derived, "lag": lag, "sample": sample},
                  fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
