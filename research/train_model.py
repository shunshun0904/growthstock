#!/usr/bin/env python3
"""
新高値ブレイクアウト予測モデルの学習と評価。

設計は docs/MODEL_DESIGN.md を参照。方法論上の要点:

  * 分割は時系列。ランダム分割は使わない
    （同一日の銘柄間が強く相関し、ラベル窓が重なるため、
      ランダム分割は必ず楽観的な数字を出す）
  * 訓練と検証の間にエンバーゴを置く。日数はラベル定義から導出する
    （訓練最終日のラベルは forward_needed 営業日先の情報を含むため。
      ここをハードコードするとラベル変更時に静かにリークする）
  * Accuracy は使わない。正例が少数なので「全部起きない」でも高く出る
  * ベースラインに勝てなければ「特徴量に追加の予測力なし」と結論する
  * ベースラインとの差は対応のあるブートストラップで有意性を確認する
    （テスト期間が約1年しかなく、PR-AUC の 0.004 差は目視で判断できない）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import tuning  # noqa: E402
import build_dataset as B  # noqa: E402
from build_dataset import DEFAULT_LABEL  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

#: 比較の基準。決算を使わないモデルを基準にして、
#: 「決算を足すと良くなるか」を直接問う。
#: 単変量のベースライン（R_high など）は母集団の変更で定数化したため廃止した。
REFERENCE_MODEL = "LightGBM [technical]"

#: レポートに出すラベル定義の名前。母集団で中身が変わる。
LABEL_NAME = (B.DEFAULT_RISE.name if B.POPULATION == "breakout"
              else DEFAULT_LABEL.name)

# エンバーゴはラベル定義から導く。ハードコードするとラベルを変えたときにリークする。
# 採用定義 E は sustain_days=60 を含むため 120+60 = 180営業日必要。
# 母集団が breakout のときのラベルは「先 RISE_HORIZON 営業日以内の上昇」なので、
# 確定に必要な将来日数は RISE_HORIZON。従来の定義E（180営業日）より大幅に短い。
# ここをラベルに追随させないとリークする。
EMBARGO_DAYS = (B.RISE_HORIZON if B.POPULATION == "breakout"
                else DEFAULT_LABEL.forward_needed)


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


def clean_score(score: np.ndarray) -> np.ndarray:
    """欠測スコアは最下位として扱う（予測できないものを高評価にしない）。"""
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if not ok.any():
        return np.zeros_like(score)
    return np.where(ok, score, score[ok].min() - 1.0)


def evaluate(name: str, y: np.ndarray, score: np.ndarray) -> Dict:
    score = clean_score(score)
    base_rate = float(y.mean())
    res = {
        "name": name,
        "n": int(len(y)),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }
    for k in (1, 5, 10):
        p = precision_at_k(y, score, k)
        res[f"precision@{k}%"] = p
        res[f"lift@{k}%"] = p / base_rate if base_rate > 0 else float("nan")
    return res


def paired_bootstrap(y: np.ndarray, scores: Dict[str, np.ndarray], reference: str,
                     n_boot: int = 1000, seed: int = 0) -> List[Dict]:
    """
    PR-AUC の差が誤差の範囲かを、対応のあるブートストラップで測る。

    テスト期間は1年ほどしかなく、n が同じでも PR-AUC の差 0.004 が
    意味のある差なのかは目視では判断できない。特徴量を足すたびに
    「わずかに上回った」が出るので、毎回ここで判定する。

    同一のリサンプルで全モデルを評価する（対応のある比較）。
    そうしないとモデル間の差にリサンプル自体のばらつきが混ざる。
    """
    y = np.asarray(y, dtype=int)
    clean = {k: clean_score(v) for k, v in scores.items()}
    names = [k for k in clean if k != reference]
    diffs = {k: np.empty(n_boot, dtype=float) for k in names}
    rng = np.random.default_rng(seed)
    n = len(y)

    done = 0
    while done < n_boot:
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # 正例が無いリサンプルでは PR-AUC が定義できない。引き直す
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ref_ap = average_precision_score(yb, clean[reference][idx])
        for k in names:
            diffs[k][done] = average_precision_score(yb, clean[k][idx]) - ref_ap
        done += 1

    out = []
    for k in names:
        d = diffs[k]
        out.append({
            "name": k,
            "pr_auc": float(average_precision_score(y, clean[k])),
            "diff": float(average_precision_score(y, clean[k])
                          - average_precision_score(y, clean[reference])),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean()),
        })
    out.sort(key=lambda r: -r["diff"])
    return out


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
    単変量のベースラインは使わない。

    かつては「R_high のみ」「出来高モメンタムのみ」「既存の8軸スコア」を
    基準にしていたが、これは月末を母集団にしてラベルが
    「先1〜6ヶ月で52週高値を抜くか」だった時代の設計。
    いまの母集団は高値更新日そのもので、R_high は全件ほぼ100の定数になる。
    定数に勝っても何も言えないので、比較の基準にはできない。

    代わりの基準は REFERENCE_MODEL（決算を使わないテクニカルのモデル）。
    問うべきは「決算を足すと、テクニカルだけより良くなるか」なので、
    基準もモデルにする。
    """
    return {}


# --------------------------------------------------------------------------- #
# 学習
# --------------------------------------------------------------------------- #

def lgbm_importance(gbm, cols: List[str]) -> List[Dict]:
    """
    LightGBM の特徴量重要度。gain（分割で減った損失の合計）を主に見る。

    split（使われた回数）だけだと、値の種類が多い列がただ多く選ばれる。
    gain は「その分割がどれだけ効いたか」なので、寄与の大きさに近い。
    どちらも出して、片方だけ大きい列は解釈のときに疑えるようにする。
    """
    booster = gbm.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total = float(gain.sum()) or 1.0
    rows = [{"col": c, "gain": float(g), "share": float(g) / total,
             "split": int(sp)}
            for c, g, sp in zip(cols, gain, split)]
    rows.sort(key=lambda r: -r["gain"])
    return rows


def fit_models(train: pd.DataFrame, features: List[str],
               verbose: bool = True, preset: str = "",
               params_store: Dict | None = None) -> Dict:
    """
    LightGBM と ロジスティック回帰 を学習する。

    LightGBM のハイパーパラメータは research/tuning.py が Optuna で
    探索した結果を使う（research/_data/lgbm_params.json）。
    探索結果が無ければ既定値。

    木の本数は探索時に early stopping で決めた本数を固定して使う。
    ここで再び early stopping を掛けると検証用の切り出しが必要になり、
    フォールドごとに訓練量が変わって比較の条件が揃わない。
    """
    import lightgbm as lgb

    Xtr = train[features].to_numpy(dtype=float)
    ytr = train["label"].to_numpy(dtype=int)

    params = tuning.params_for(preset, params_store)
    if verbose:
        print(f"\n[fit] LightGBM (木{params['n_estimators']}本 "
              f"/ lr {params['learning_rate']:.3f} "
              f"/ 葉 {params['num_leaves']})")
    gbm = lgb.LGBMClassifier(**params,
                             scale_pos_weight=tuning.scale_pos_weight(ytr))
    gbm.fit(Xtr, ytr)

    if verbose:
        print("[fit] ロジスティック回帰（解釈用）")
    lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    )
    lr.fit(Xtr, ytr)

    return {"LightGBM": gbm, "ロジスティック回帰": lr}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ブレイクアウト予測モデルの学習と評価")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--val-start", default="2023-01-01")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="ブートストラップ反復回数。0で無効")
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
    print(f"[label] 定義: {LABEL_NAME}")
    print(f"[split] エンバーゴ {EMBARGO_DAYS}営業日（ラベル定義から自動導出）\n")

    parts = time_split(df, args.val_start, args.test_start)

    presets = args.features or list(F.PRESETS.keys())
    unknown = [p for p in presets if p not in F.PRESETS]
    if unknown:
        raise SystemExit(f"未知の特徴量セット: {unknown}. 利用可能: {sorted(F.PRESETS)}")

    # --- ベースラインは特徴量セットに依らないので先に1度だけ算出する ---
    baselines: Dict[str, List[Dict]] = {}
    test_scores: Dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        part = parts[split]
        y = part["label"].to_numpy(dtype=int)
        bs = baseline_scores(part)
        baselines[split] = [evaluate(n, y, sc) for n, sc in bs.items()]
        if split == "test":
            test_scores.update(bs)

    # --- 特徴量セットごとに学習・評価 ---
    params_store = tuning.load_params()
    if params_store:
        print(f"[tune] 探索済みパラメータ {len(params_store)}件を使用")
    else:
        print("[tune] 探索結果が無いため既定値を使用")
    experiments: List[Dict] = []
    for preset in presets:
        cols = F.columns(preset)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[skip] {preset}: 列がありません {missing}")
            continue
        print(f"\n{'='*70}\n[experiment] {F.describe(preset)}\n{'='*70}")
        models = fit_models(parts["train"], cols, preset=preset,
                            params_store=params_store)
        rec = {"preset": preset, "groups": F.PRESETS[preset], "n_features": len(cols),
               "results": {}}
        for split in ("val", "test"):
            part = parts[split]
            y = part["label"].to_numpy(dtype=int)
            X = part[cols].to_numpy(dtype=float)
            rows = []
            for mname, model in models.items():
                sc = model.predict_proba(X)[:, 1]
                rows.append(evaluate(f"{mname}", y, sc))
                if split == "test":
                    test_scores[f"{mname} [{preset}]"] = sc
            rec["results"][split] = rows
            best = max(rows, key=lambda r: r["pr_auc"])
            print(f"  {split:<5} 最良 {best['name']}: "
                  f"PR-AUC {best['pr_auc']:.4f} / Lift@5% {best['lift@5%']:.2f}x")
        rec["_models"] = models
        rec["_cols"] = cols
        rec["importance"] = lgbm_importance(models["LightGBM"], cols)
        experiments.append(rec)

    if not experiments:
        raise SystemExit("実行できた実験がありません")

    # --- 基準との差が誤差かを判定 ---
    # 全モデルを対象にすると遅いので、上位と「順位版 vs 絶対値版」の対に絞る。
    y_test = parts["test"]["label"].to_numpy(dtype=int)
    REF = (REFERENCE_MODEL if REFERENCE_MODEL in test_scores
           else max(test_scores, key=lambda k: average_precision_score(
               y_test, clean_score(test_scores[k]))))
    ranked = sorted((k for k in test_scores if k != REF),
                    key=lambda k: -average_precision_score(y_test, clean_score(test_scores[k])))
    pairs = [f"{m} [{p_}]"
             for base in ("price_only", "technical", "all", "fundamental")
             for p_ in (base, f"rank_{base}")
             for m in ("ロジスティック回帰", "LightGBM")]
    keep = list(dict.fromkeys(ranked[:6] + [k for k in pairs if k in test_scores]))
    print(f"\n[bootstrap] {len(keep)}モデル × {args.n_boot}回 の対応のあるブートストラップ")
    boot = paired_bootstrap(y_test, {REF: test_scores[REF],
                                     **{k: test_scores[k] for k in keep}},
                            reference=REF, n_boot=args.n_boot)
    for r in boot[:5]:
        print(f"  {r['name']:<40} 差 {r['diff']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] P(差>0)={r['p_better']:.3f}")

    # --- レポート ---
    body = _report(df, parts, baselines, experiments, args, boot, REF)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    payload = {
        "labelConfig": LABEL_NAME,
        "embargoDays": EMBARGO_DAYS,
        "split": {"val_start": args.val_start, "test_start": args.test_start},
        "baselines": baselines,
        "bootstrap": {"reference": REF, "n_boot": args.n_boot, "test": boot},
        "experiments": [{k: v for k, v in e.items() if not k.startswith("_")}
                        for e in experiments],
    }
    with open(os.path.join(DATA_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return 0


def _report(df, parts, baselines, experiments, args, boot=None, ref="") -> str:
    """実測値だけを並べたレポートを組み立てる。"""
    lines = [
        "# 新高値ブレイクアウト予測モデル 結果",
        "",
        "`research/train_model.py` の出力。**実測値のみ**を記載する。",
        "設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。",
        "",
        "## 条件",
        "",
        f"- **ラベル定義**: {LABEL_NAME}",
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

    # --- ベースラインとの差の有意性 ---
    if boot:
        lines += ["", f"## 差は誤差か（対応のあるブートストラップ B={args.n_boot}）", "",
                  f"基準は **{ref}**（テスト PR-AUC "
                  f"{[b for b in baselines['test'] if b['name'] == ref][0]['pr_auc']:.4f}）。",
                  "95%CI が 0 をまたぐ場合、その差は誤差と区別できない。", "",
                  "| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |",
                  "| --- | ---: | ---: | :---: | ---: | --- |"]
        for r in boot:
            sig = "有意" if r["ci_low"] > 0 else ("有意に劣る" if r["ci_high"] < 0 else "誤差")
            lines.append(f"| {r['name']} | {r['pr_auc']:.4f} | {r['diff']:+.4f} | "
                         f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                         f"{r['p_better']:.3f} | {sig} |")

    # --- 特徴量セット別の要約 ---
    lines += ["", "## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）", "",
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
