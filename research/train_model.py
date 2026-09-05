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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from build_dataset import DEFAULT_LABEL  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
META_COLS = ["Code", "Date", "close", "high52w", "tv_ma20", "market_cap", "label"]

# エンバーゴはラベル定義から導く。ハードコードするとラベルを変えたときにリークする。
# 採用定義 E は sustain_days=60 を含むため 120+60 = 180営業日必要。
EMBARGO_DAYS = DEFAULT_LABEL.forward_needed


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
    ap.add_argument("--features", nargs="*", default=None,
                    help=f"評価する特徴量セット。既定は全プリセット。"
                         f"利用可能: {sorted(F.PRESETS)}")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "MODEL_RESULTS.md"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.dataset):
        raise SystemExit(f"{args.dataset} がありません。先に build_dataset.py を実行してください")

    df = pd.read_parquet(args.dataset)
    print(f"[data] {len(df):,}サンプル / 正例率 {df['label'].mean()*100:.2f}%")
    print(f"[label] 定義: {DEFAULT_LABEL.name}")
    print(f"[split] エンバーゴ {EMBARGO_DAYS}営業日（ラベル定義から自動導出）\n")

    parts = time_split(df, args.val_start, args.test_start)

    presets = args.features or list(F.PRESETS.keys())
    unknown = [p for p in presets if p not in F.PRESETS]
    if unknown:
        raise SystemExit(f"未知の特徴量セット: {unknown}. 利用可能: {sorted(F.PRESETS)}")

    # --- ベースラインは特徴量セットに依らないので先に1度だけ算出する ---
    baselines: Dict[str, List[Dict]] = {}
    for split in ("val", "test"):
        part = parts[split]
        y = part["label"].to_numpy(dtype=int)
        baselines[split] = [evaluate(n, y, sc) for n, sc in baseline_scores(part).items()]

    # --- 特徴量セットごとに学習・評価 ---
    experiments: List[Dict] = []
    for preset in presets:
        cols = F.columns(preset)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[skip] {preset}: 列がありません {missing}")
            continue
        print(f"\n{'='*70}\n[experiment] {F.describe(preset)}\n{'='*70}")
        models = fit_models(parts["train"], cols)
        rec = {"preset": preset, "groups": F.PRESETS[preset], "n_features": len(cols),
               "results": {}}
        for split in ("val", "test"):
            part = parts[split]
            y = part["label"].to_numpy(dtype=int)
            X = part[cols].to_numpy(dtype=float)
            rows = []
            for mname, model in models.items():
                rows.append(evaluate(f"{mname}", y, model.predict_proba(X)[:, 1]))
            rec["results"][split] = rows
            best = max(rows, key=lambda r: r["pr_auc"])
            print(f"  {split:<5} 最良 {best['name']}: "
                  f"PR-AUC {best['pr_auc']:.4f} / Lift@5% {best['lift@5%']:.2f}x")
        rec["_models"] = models
        rec["_cols"] = cols
        experiments.append(rec)

    if not experiments:
        raise SystemExit("実行できた実験がありません")

    # --- レポート ---
    body = _report(df, parts, baselines, experiments, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    payload = {
        "labelConfig": DEFAULT_LABEL.name,
        "embargoDays": EMBARGO_DAYS,
        "split": {"val_start": args.val_start, "test_start": args.test_start},
        "baselines": baselines,
        "experiments": [{k: v for k, v in e.items() if not k.startswith("_")}
                        for e in experiments],
    }
    with open(os.path.join(DATA_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return 0


def _report(df, parts, baselines, experiments, args) -> str:
    """実測値だけを並べたレポートを組み立てる。"""
    lines = [
        "# 新高値ブレイクアウト予測モデル 結果",
        "",
        "`research/train_model.py` の出力。**実測値のみ**を記載する。",
        "設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。",
        "",
        "## 条件",
        "",
        f"- **ラベル定義**: {DEFAULT_LABEL.name}",
        f"- データセット: {len(df):,}サンプル / 全体の正例率 **{df['label'].mean()*100:.2f}%**",
        f"- 期間: {pd.to_datetime(df['Date']).min().date()} 〜 {pd.to_datetime(df['Date']).max().date()}"
        f" / 銘柄数 {df['Code'].nunique():,}",
        f"- 分割: 訓練 〜{args.val_start} / 検証 {args.val_start}〜{args.test_start} / テスト {args.test_start}〜",
        f"- **エンバーゴ {EMBARGO_DAYS}営業日**（ラベル確定に必要な将来日数から自動導出）",
        "",
    ]
    for name, part in parts.items():
        d = pd.to_datetime(part["Date"])
        lines.append(f"  - {name}: {len(part):,}件 / {d.min().date()} 〜 {d.max().date()}"
                     f" / 正例率 {part['label'].mean()*100:.2f}%")
    lines.append("")

    for split, title in (("val", "検証データ (val)"), ("test", "テストデータ (test) — 最終評価")):
        rows = list(baselines[split])
        for e in experiments:
            for r in e["results"][split]:
                rows.append({**r, "name": f"{r['name']} [{e['preset']}]"})
        rows.sort(key=lambda r: -r["pr_auc"])
        lines += ["", f"## {title}", "",
                  "| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            lines.append(f"| {r['name']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
                         f"{r['precision@1%']*100:.1f}% | {r['precision@5%']*100:.1f}% | "
                         f"{r['lift@5%']:.2f}x |")
        lines.append(f"\n（正例率 = {rows[0]['base_rate']*100:.2f}% / n = {rows[0]['n']:,}）")

    # --- 特徴量セット別の要約 ---
    lines += ["", "## 特徴量セット別の比較（テストデータ・勾配ブースティング）", "",
              "| セット | 列数 | 構成 | PR-AUC | Lift@5% |",
              "| --- | ---: | --- | ---: | ---: |"]
    for e in sorted(experiments,
                    key=lambda e: -max(r["pr_auc"] for r in e["results"]["test"])):
        best = max(e["results"]["test"], key=lambda r: r["pr_auc"])
        lines.append(f"| `{e['preset']}` | {e['n_features']} | {' + '.join(e['groups'])} | "
                     f"{best['pr_auc']:.4f} | {best['lift@5%']:.2f}x |")

    # --- 特徴量の寄与 ---
    best_exp = max(experiments, key=lambda e: max(r["pr_auc"] for r in e["results"]["test"]))
    lr = best_exp["_models"].get("ロジスティック回帰")
    if lr is not None:
        coefs = lr.named_steps["logisticregression"].coef_[0]
        imp = sorted(zip(best_exp["_cols"], coefs), key=lambda x: -abs(x[1]))[:15]
        lines += ["", f"## 特徴量の寄与（`{best_exp['preset']}` のロジスティック回帰・標準化係数 上位15）",
                  "", "| 特徴量 | 係数 | 向き |", "| --- | ---: | --- |"]
        for name, c in imp:
            lines.append(f"| `{name}` | {c:+.3f} | {'ブレイクしやすい' if c > 0 else 'しにくい'} |")

    lines += ["", "## 読み方", "",
              "- **PR-AUC** が主指標。下限は正例率で、それを大きく上回るほど良い",
              "- **Lift@5%** は「スコア上位5%の正例率 ÷ 全体の正例率」。1.0 なら無意味",
              "- ベースライン（`R_high` 単独など）を上回らなければ、"
              "**追加の特徴量に予測力が無い**という結論になる",
              "- `all` と `all_no_market` の差が、"
              "**モデルが相場局面をどれだけ暗記していたか**の目安になる",
              "- 数字が極端に良い場合はまずリークを疑う"
              "（`tests/test_dataset.py` の先読み検出テストを参照）",
              "",
              "## 特徴量を足して試すには",
              "",
              "1. `research/features.py` の `GROUPS` に列を足す",
              "2. `research/build_dataset.py` でその列を作る",
              "3. データセットを再構築（Release から読むので数分・API取得なし）",
              "4. `PRESETS` にセットを1行足して再学習",
              ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
