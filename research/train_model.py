#!/usr/bin/env python3
"""
新高値ブレイクアウト予測モデルの学習と評価。

設計は docs/MODEL_DESIGN.md を参照。方法論上の要点:

  * 分割は時系列。ランダム分割は使わない
    （同一日の銘柄間が強く相関し、ラベル窓が重なるため、
      ランダム分割は必ず楽観的な数字を出す）
  * 訓練と検証の間に 140営業日のエンバーゴを置く
    （訓練最終日のラベルは最大140営業日先の情報を含むため）
  * Accuracy は使わない。正例が少数なので「全部起きない」でも高く出る
  * ベースラインに勝てなければ「特徴量に追加の予測力なし」と結論する
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
META_COLS = ["Code", "Date", "close", "high52w", "tv_ma20", "market_cap", "label"]

EMBARGO_DAYS = 140   # ラベル窓の長さ。訓練と検証の間に必ず空ける


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #

def time_split(df: pd.DataFrame, val_start: str, test_start: str) -> Dict[str, pd.DataFrame]:
    """
    時系列で 訓練 / 検証 / テスト に分ける。境界にエンバーゴを入れる。

        |--- 訓練 ---|--エンバーゴ--|--- 検証 ---|--エンバーゴ--|--- テスト ---|
    """
    d = pd.to_datetime(df["Date"])
    val_start_ts = pd.Timestamp(val_start)
    test_start_ts = pd.Timestamp(test_start)
    # 営業日ベースのエンバーゴを暦日に換算（1営業日 ≒ 1.45暦日）
    embargo = pd.Timedelta(days=int(EMBARGO_DAYS * 1.45))

    parts = {
        "train": df[d < val_start_ts - embargo],
        "val": df[(d >= val_start_ts) & (d < test_start_ts - embargo)],
        "test": df[d >= test_start_ts],
    }
    for name, p in parts.items():
        if len(p) == 0:
            raise SystemExit(f"{name} が空です。val_start/test_start を見直してください")
        dd = pd.to_datetime(p["Date"])
        print(f"  {name:<6} {len(p):>8,}件  {dd.min().date()} 〜 {dd.max().date()}  "
              f"正例率 {p['label'].mean()*100:5.2f}%")
    return parts


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def precision_at_k(y_true: np.ndarray, score: np.ndarray, k_pct: float) -> float:
    """スコア上位 k% の的中率。実運用（上位n銘柄だけ見る）に最も近い指標。"""
    n = max(1, int(len(score) * k_pct / 100))
    idx = np.argsort(-score)[:n]
    return float(y_true[idx].mean())


def evaluate(name: str, y: np.ndarray, score: np.ndarray) -> Dict:
    """欠測スコアは最下位として扱う（予測できないものを高評価にしない）。"""
    score = np.where(np.isfinite(score), score, -np.inf)
    base_rate = float(y.mean())
    res = {
        "name": name,
        "n": int(len(y)),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y, np.where(np.isfinite(score), score, np.nanmin(score[np.isfinite(score)]) - 1))),
        "roc_auc": float(roc_auc_score(y, np.where(np.isfinite(score), score, np.nanmin(score[np.isfinite(score)]) - 1))),
    }
    for k in (1, 5, 10):
        p = precision_at_k(y, score, k)
        res[f"precision@{k}%"] = p
        res[f"lift@{k}%"] = p / base_rate if base_rate > 0 else float("nan")
    return res


def report(rows: List[Dict], title: str) -> str:
    lines = [f"\n### {title}", "",
             "| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | P@10% | Lift@5% |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['precision@1%']*100:.1f}% | {r['precision@5%']*100:.1f}% | "
            f"{r['precision@10%']*100:.1f}% | {r['lift@5%']:.2f}x |"
        )
    lines.append(f"\n（正例率 = {rows[0]['base_rate']*100:.2f}% / n = {rows[0]['n']:,}）")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ベースライン
# --------------------------------------------------------------------------- #

def baseline_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    これらに勝てなければ34特徴量のモデルを作る意味がない。
    """
    out = {}
    # 1. 52週高値接近率のみ（高値に近いほどブレイクしやすい、という素朴な仮説）
    out["ベースライン: R_high のみ"] = df["r_high"].to_numpy(dtype=float)
    # 2. 出来高モメンタムのみ
    out["ベースライン: 出来高モメンタムのみ"] = df["volume_trend"].to_numpy(dtype=float)
    # 3. 既存ダッシュボードの8軸総合スコア相当（現行スコアに予測力があるかの検証）
    from eight_axis_score import eight_axis_total  # noqa: E402
    out["ベースライン: 既存の8軸総合スコア"] = eight_axis_total(df)
    return out


# --------------------------------------------------------------------------- #
# 学習
# --------------------------------------------------------------------------- #

def fit_models(train: pd.DataFrame, features: List[str]) -> Dict:
    Xtr = train[features].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=int)
    # 不均衡対策: 正例に負例/正例 の重みを与える
    pos = max(1, int(ytr.sum()))
    w = np.where(ytr == 1, (len(ytr) - pos) / pos, 1.0)

    print("\n[fit] HistGradientBoosting")
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )
    hgb.fit(Xtr, ytr, sample_weight=w)

    print("[fit] ロジスティック回帰（解釈用）")
    lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    )
    lr.fit(Xtr, ytr)

    return {"勾配ブースティング": hgb, "ロジスティック回帰": lr}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ブレイクアウト予測モデルの学習と評価")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--val-start", default="2023-01-01")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "MODEL_RESULTS.md"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.dataset):
        raise SystemExit(f"{args.dataset} がありません。先に build_dataset.py を実行してください")

    df = pd.read_parquet(args.dataset)
    features = [c for c in df.columns if c not in META_COLS]
    print(f"[data] {len(df):,}サンプル / 特徴量{len(features)}個 / 正例率 {df['label'].mean()*100:.2f}%")

    print("\n[split] 時系列分割（エンバーゴ140営業日）")
    parts = time_split(df, args.val_start, args.test_start)

    models = fit_models(parts["train"], features)

    sections = []
    all_results = {}
    for split_name in ("val", "test"):
        part = parts[split_name]
        y = part["label"].to_numpy(dtype=int)
        rows = []
        for bname, bscore in baseline_scores(part).items():
            rows.append(evaluate(bname, y, bscore))
        for mname, model in models.items():
            X = part[features].to_numpy(dtype=float)
            score = model.predict_proba(X)[:, 1]
            rows.append(evaluate(mname, y, score))
        rows.sort(key=lambda r: -r["pr_auc"])
        title = "検証データ (val)" if split_name == "val" else "テストデータ (test) — 最終評価"
        sections.append(report(rows, title))
        all_results[split_name] = rows
        print(sections[-1])

    # --- 特徴量重要度（ロジスティック回帰の標準化係数） --- #
    lr = models["ロジスティック回帰"]
    coefs = lr.named_steps["logisticregression"].coef_[0]
    imp = sorted(zip(features, coefs), key=lambda x: -abs(x[1]))[:15]
    imp_lines = ["\n### 特徴量の寄与（ロジスティック回帰の標準化係数・上位15）", "",
                 "| 特徴量 | 係数 | 向き |", "| --- | ---: | --- |"]
    for name, c in imp:
        imp_lines.append(f"| `{name}` | {c:+.3f} | {'ブレイクしやすい' if c > 0 else 'しにくい'} |")

    body = "\n".join([
        "# 新高値ブレイクアウト予測モデル 結果",
        "",
        "本文書は `research/train_model.py` の出力を貼り付けたもので、**実測値のみ**を記載する。",
        "設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。",
        "",
        f"- データセット: {len(df):,}サンプル / 特徴量 {len(features)}個",
        f"- 全体の正例率: {df['label'].mean()*100:.2f}%",
        f"- 期間: {pd.to_datetime(df['Date']).min().date()} 〜 {pd.to_datetime(df['Date']).max().date()}",
        f"- 銘柄数: {df['Code'].nunique():,}",
        f"- 分割: 訓練 〜{args.val_start} / 検証 {args.val_start}〜{args.test_start} / テスト {args.test_start}〜"
        f"（境界に {EMBARGO_DAYS}営業日のエンバーゴ）",
        *sections,
        *imp_lines,
        "",
        "## 読み方",
        "",
        "- **PR-AUC** が主指標。正例率（上表の脚注）が下限で、それを大きく上回るほど良い",
        "- **Lift@5%** は「スコア上位5%の正例率 ÷ 全体の正例率」。1.0 なら無意味、2.0 なら2倍当たる",
        "- ベースライン（`R_high` 単独など）を上回らなければ、"
        "**追加の特徴量に予測力が無い**という結論になる",
        "- 数字が極端に良い場合はまずリークを疑う（`tests/test_dataset.py` の先読み検出テストを参照）",
    ])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    with open(os.path.join(DATA_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
