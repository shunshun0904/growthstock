#!/usr/bin/env python3
"""
日付内での分離力を直接測る。

なぜこれを先に測るか:
  ウォークフォワードでは、どの特徴量セットも基準(R_high)を上回らなかった。
  次の候補は learning-to-rank（同一日内の順位を直接最適化する）だが、
  LTR が利用できるのは「同じ日の銘柄どうしを分ける情報」だけである。
  その情報が存在しないなら、LTR を作っても取り出せない。
  モデルを組む前に、情報があるかどうかを測る。

測るもの:
  1. 単独の分離力
     各日付ごとに「特徴量 vs ラベル」の AUC を計算し、全日付で平均する。
     AUC 0.5 は分離力ゼロ。日付内で完結するので正例率の局面差に影響されない。

  2. R_high を与えた上での増分
     各日付を R_high の5分位に切り、その中で同じ AUC を計算する。
     「高値からの距離が同程度の銘柄どうしを、その特徴量が更に分けられるか」。
     LTR に見込みがあるかを決めるのはこちら。
     1 で高くても R_high と相関しているだけなら、2 で 0.5 に落ちる。

なお順位版 (*_r) は日付内での単調変換なので、日付内 AUC は元の列と一致する。
測る意味が無いので対象から外す（一致することはテストで確認している）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from train_model import DATA_DIR  # noqa: E402
from walkforward import sign_test  # noqa: E402

MIN_ROWS = 30       # この数を下回る日付・セルは推定が不安定なので捨てる
MIN_POS = 3         # 正例がこれ未満だと AUC がほぼ無意味
N_BUCKETS = 5       # R_high の分位数


def _auc(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """欠測を除いて AUC。両クラス揃わない・数が足りない場合は None。"""
    ok = np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < MIN_ROWS:
        return None
    pos = int(y.sum())
    if pos < MIN_POS or pos > len(y) - MIN_POS:
        return None
    return float(roc_auc_score(y, x))


def _summarise(name: str, aucs: List[float]) -> Optional[Dict]:
    if not aucs:
        return None
    a = np.array(aucs)
    wins = int((a > 0.5).sum())
    losses = int((a < 0.5).sum())
    return {
        "feature": name,
        "n_cells": len(a),
        "mean_auc": float(a.mean()),
        "std_auc": float(a.std()),
        # 0.5 からどれだけ離れているか。向きは問わない
        "abs_edge": float(abs(a.mean() - 0.5)),
        "wins": wins,
        "losses": losses,
        "p_sign": sign_test(wins, losses),
    }


def marginal(df: pd.DataFrame, cols: List[str]) -> List[Dict]:
    """日付ごとの AUC。"""
    out = []
    grouped = list(df.groupby("Date", sort=True))
    for c in cols:
        aucs = []
        for _, g in grouped:
            v = _auc(g["label"].to_numpy(dtype=int), g[c].to_numpy(dtype=float))
            if v is not None:
                aucs.append(v)
        s = _summarise(c, aucs)
        if s:
            out.append(s)
    out.sort(key=lambda r: -r["abs_edge"])
    return out


def conditional(df: pd.DataFrame, cols: List[str], on: str = "r_high") -> List[Dict]:
    """日付 × on の分位 で切ったセル内の AUC。"""
    work = df.copy()
    # 分位は日付ごとに切る（局面で R_high の分布が動くため）
    work["_b"] = (work.groupby("Date", sort=False)[on]
                  .transform(lambda s: pd.qcut(s, N_BUCKETS, labels=False,
                                               duplicates="drop")))
    cells = list(work.groupby(["Date", "_b"], sort=True))
    out = []
    for c in cols:
        if c == on:
            continue
        aucs = []
        for _, g in cells:
            v = _auc(g["label"].to_numpy(dtype=int), g[c].to_numpy(dtype=float))
            if v is not None:
                aucs.append(v)
        s = _summarise(c, aucs)
        if s:
            out.append(s)
    out.sort(key=lambda r: -r["abs_edge"])
    return out


def _table(rows: List[Dict], limit: int) -> List[str]:
    lines = ["| 特徴量 | 平均AUC | 標準偏差 | 0.5からの差 | 上回った回数 | p |",
             "| --- | ---: | ---: | ---: | :---: | ---: |"]
    for r in rows[:limit]:
        lines.append(
            f"| `{r['feature']}` | {r['mean_auc']:.4f} | {r['std_auc']:.4f} | "
            f"{r['abs_edge']:.4f} | {r['wins']}/{r['wins']+r['losses']} | "
            f"{r['p_sign']:.4f} |")
    return lines


def build_report(marg: List[Dict], cond: List[Dict], n_dates: int, n_cells: int) -> str:
    ref_m = next((r for r in marg if r["feature"] == "r_high"), None)
    lines = [
        "# 日付内での分離力",
        "",
        "`research/within_date_signal.py` の出力。**実測値のみ**を記載する。",
        "",
        "ウォークフォワード（[MODEL_WALKFORWARD.md](MODEL_WALKFORWARD.md)）では",
        "どの特徴量セットも基準を上回らなかった。次の候補である learning-to-rank が",
        "利用できるのは「同じ日の銘柄どうしを分ける情報」だけなので、",
        "モデルを組む前にその情報が存在するかを測る。",
        "",
        "- AUC 0.5 = 分離力ゼロ。日付内で完結するため正例率の局面差に影響されない",
        f"- 対象日付: {n_dates}",
        f"- 条件付きのセル数: {n_cells}（日付 × R_high の{N_BUCKETS}分位）",
        f"- 1セルあたり最低 {MIN_ROWS}行・正例{MIN_POS}件を要求。満たさないセルは除外",
        "",
        "## 1. 単独の分離力（日付内）",
        "",
    ]
    if ref_m:
        lines += [f"基準 `r_high` の平均AUC は **{ref_m['mean_auc']:.4f}** "
                  f"（{ref_m['wins']}/{ref_m['wins']+ref_m['losses']}日で0.5超）。", ""]
    lines += _table(marg, 20)
    lines += [
        "",
        "## 2. R_high を与えた上での増分",
        "",
        "各日付を `r_high` の分位で切り、その中だけで測る。",
        "「高値からの距離が同程度の銘柄どうしを、その特徴量が更に分けられるか」。",
        "**LTR に見込みがあるかを決めるのはこちら。**",
        "1 で高くても `r_high` と相関しているだけの特徴量は、ここで 0.5 に落ちる。",
        "",
    ]
    lines += _table(cond, 25)
    lines += [
        "",
        "## 読み方",
        "",
        "- 2 の平均AUC が軒並み 0.5 付近なら、**日付内に取り出せる情報が無い**。",
        "  learning-to-rank を作っても結果は変わらない（存在しない情報は最適化できない）",
        "- 2 で 0.5 から離れ、かつ符号が一貫している特徴量があれば、",
        "  それが LTR で拾える候補になる",
        "- 平均AUC が 0.5 未満（負の向き）でも情報はある。符号を反転すればよい",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="日付内での分離力を測る")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_WITHIN_DATE.md"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    # 順位版は日付内の単調変換なので AUC が元の列と一致する。測る意味が無い
    cols = [c for c in F.columns("all") if c in df.columns]
    n_dates = df["Date"].nunique()
    print(f"[load] {len(df):,}件 / {n_dates}日付 / 特徴量 {len(cols)}個")

    print("\n[1] 単独の分離力（日付内 AUC）")
    marg = marginal(df, cols)
    for r in marg[:8]:
        print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
              f"({r['wins']}/{r['wins']+r['losses']}日)")

    print(f"\n[2] R_high の{N_BUCKETS}分位で条件付け")
    cond = conditional(df, cols)
    for r in cond[:8]:
        print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
              f"(差 {r['abs_edge']:.4f} / p={r['p_sign']:.4f})")

    n_cells = cond[0]["n_cells"] if cond else 0
    body = build_report(marg, cond, n_dates, n_cells)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    with open(os.path.join(DATA_DIR, "within_date.json"), "w", encoding="utf-8") as fh:
        json.dump({"marginal": marg, "conditional": cond,
                   "n_dates": int(n_dates)}, fh, ensure_ascii=False, indent=2)

    strong = [r for r in cond if r["abs_edge"] >= 0.02 and r["p_sign"] < 0.05]
    print("\n[判定] R_high を与えた上でも分離力が残る特徴量:")
    if strong:
        for r in strong:
            print(f"  {r['feature']:<24} AUC {r['mean_auc']:.4f} "
                  f"(差 {r['abs_edge']:+.4f})")
        print("  -> learning-to-rank に見込みがある")
    else:
        print("  なし（|AUC-0.5| >= 0.02 かつ p<0.05 を満たすものが無い）")
        print("  -> learning-to-rank を作っても結果は変わらない見込み")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
