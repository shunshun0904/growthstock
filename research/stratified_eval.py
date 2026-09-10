#!/usr/bin/env python3
"""
時価総額（対数）の分位で層別して評価する。

## なぜ層で切るか

母集団を「78週高値の更新日」に固定したので、どの銘柄も高値からの距離は
同じところにいる。残る最大の交絡は規模で、「先3ヶ月で+20%」の起きやすさが
小型と大型で2倍近く違う。層を切らずに評価すると、モデルの優位が
「小型を上に持ってきただけ」なのか「同じ規模帯の中で選べている」のかを
区別できない。

## ここでやること

日付ごとに log_market_cap の分位で層に分け、**層の中だけで**評価する。
「同じくらいの規模の銘柄どうしで、どれが実際に伸びるか」を測る。

各層で PR-AUC を出し、層ごとの正例率で割った Lift を見る。
比較の相手は層内での log_market_cap 単独。層は日付ごとの分位なので
層内にもレンジは残り、Lift が1.0まで落ちる必要はない。
モデルがそれを上回れば、規模以外の情報で選べていることになる。

## 過去の経緯

母集団が月末×全銘柄だったころは r_high（52週高値への近さ）で層別していた。
ラベルが「52週高値を上抜けるか」で r_high が「その何%の位置か」なので、
r_high は「ゴールまでの距離」を測っているだけの同語反復だった
（全9局面で 1.69倍±0.15 という異常に安定したリフトはこのため）。
母集団を更新日にしたことでこの問題自体が消えている。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import tuning  # noqa: E402
from train_model import (  # noqa: E402
    DATA_DIR, EMBARGO_DAYS, clean_score, fit_models, precision_at_k,
)
from walkforward import TRADING_TO_CALENDAR, make_folds  # noqa: E402

N_STRATA = 5

# 層別の軸。母集団によって意味のある軸が変わる。
#   month_end … r_high（ゴールまでの距離）を揃える
#   breakout  … 高値からの距離は全銘柄で同じなので軸にならない。
#               代わりに時価総額で揃える。値動きの大きさが規模に強く依存し、
#               「+20%上昇」の起きやすさが規模で違うため
import build_dataset as _B  # noqa: E402
STRATIFY_BY = "log_market_cap" if _B.POPULATION == "breakout" else "r_high"
REFERENCE = STRATIFY_BY
#: レポートの見出しに使う日本語名。軸を変えたらここも一緒に動く。
AXIS_JA = {"log_market_cap": "時価総額（対数）", "r_high": "R_high"}.get(
    STRATIFY_BY, STRATIFY_BY)

# 層別で比べる特徴量セット。層内では株価位置がほぼ効かないので、
# 決算・バリュエーション側を中心に見る
DEFAULT_PRESETS = ["breakout_only", "technical", "fundamental", "fundamental_v3",
                   "extras_only", "valuation_only", "all"]


def assign_strata(df: pd.DataFrame, n: int = N_STRATA) -> pd.Series:
    """日付ごとに STRATIFY_BY の分位で層を振る。局面で分布が動くため日付内で切る。"""
    return df.groupby("Date", sort=False)[STRATIFY_BY].transform(
        lambda s: pd.qcut(s, n, labels=False, duplicates="drop"))


def evaluate_within(y: np.ndarray, score: np.ndarray) -> Optional[Dict]:
    if len(y) < 50 or y.sum() < 5 or y.sum() == len(y):
        return None
    s = clean_score(score)
    base = float(y.mean())
    ap = float(average_precision_score(y, s))
    # 上位k%は train_model と同じ実装を使う（同点を期待値で扱う版）。
    # ここだけ argsort のままだと、同点時に層内の並び順を読んでしまう。
    p10 = precision_at_k(y, s, 10)
    return {"n": int(len(y)), "base_rate": base, "pr_auc": ap,
            "lift": ap / base if base > 0 else float("nan"),
            "precision@10%": p10,
            "lift@10%": p10 / base if base > 0 else float("nan")}


def run(df: pd.DataFrame, presets: List[str], folds) -> Dict:
    dates = pd.to_datetime(df["Date"])
    df = df.assign(_stratum=assign_strata(df))
    params_store = tuning.load_params()
    out: List[Dict] = []

    for fold in folds:
        tr = df[(dates >= fold.train_start) & (dates <= fold.train_end)]
        te = df[(dates >= fold.test_start) & (dates <= fold.test_end)]
        if len(tr) == 0 or len(te) == 0 or tr["label"].nunique() < 2:
            continue
        print(f"\n[fold {fold.index}] テスト {fold.test_start}〜{fold.test_end}")

        scores: Dict[str, np.ndarray] = {
            REFERENCE: te[STRATIFY_BY].to_numpy(float)}
        for preset in presets:
            cols = [c for c in F.columns(preset) if c in df.columns]
            if len(cols) < len(F.columns(preset)):
                continue
            models = fit_models(tr, cols, verbose=False, preset=preset,
                                params_store=params_store)
            X = te[cols].to_numpy(dtype=float)
            for mname, model in models.items():
                scores[f"{mname} [{preset}]"] = model.predict_proba(X)[:, 1]

        for st in sorted(te["_stratum"].dropna().unique()):
            m = (te["_stratum"] == st).to_numpy()
            y = te.loc[m, "label"].to_numpy(dtype=int)
            rh = te.loc[m, STRATIFY_BY]
            for name, sc in scores.items():
                res = evaluate_within(y, sc[m])
                if res is None:
                    continue
                out.append({"fold": fold.index, "stratum": int(st), "model": name,
                            "axis_lo": float(rh.min()),
                            "axis_hi": float(rh.max()), **res})
    return {"rows": out, "summary": summarize(out)}


def summarize(rows: List[Dict]) -> List[Dict]:
    """モデル × 層 で、全フォールドを通した平均リフトと勝敗を出す。"""
    if not rows:
        return []
    d = pd.DataFrame(rows)
    ref = d[d["model"] == REFERENCE][["fold", "stratum", "pr_auc"]] \
        .rename(columns={"pr_auc": "ref_pr_auc"})
    d = d.merge(ref, on=["fold", "stratum"], how="left")
    d["diff"] = d["pr_auc"] - d["ref_pr_auc"]

    out = []
    for (model, st), g in d[d["model"] != REFERENCE].groupby(["model", "stratum"]):
        wins = int((g["diff"] > 0).sum())
        losses = int((g["diff"] < 0).sum())
        out.append({
            "model": model, "stratum": int(st), "n_folds": int(len(g)),
            "mean_lift": float(g["lift"].mean()),
            "mean_lift10": float(g["lift@10%"].mean()),
            "mean_diff": float(g["diff"].mean()),
            "wins": wins, "losses": losses,
            "mean_base_rate": float(g["base_rate"].mean()),
        })
    out.sort(key=lambda r: (r["stratum"], -r["mean_diff"]))
    return out


def build_report(res: Dict, args) -> str:
    rows, summary = res["rows"], res["summary"]
    d = pd.DataFrame(rows) if rows else pd.DataFrame()
    lines = [
        f"# {AXIS_JA}で層別した評価",
        "",
        "`research/stratified_eval.py` の出力。**実測値のみ**を記載する。",
        "",
        "## なぜ層別にするか",
        "",
        "母集団を高値更新日に変える前は `r_high`（52週高値への近さ）で層別していた。",
        "ラベルが「52週高値を上抜けるか」で `r_high` が「その何%の位置か」なので、",
        "`r_high` は**ゴールまでの距離**を測っているにすぎず、比較にならなかった。",
        "",
        "いまの母集団は高値更新日なので、高値からの距離は全銘柄で同じになり、",
        "層別の軸にならない。",
        f"代わりに **{AXIS_JA}** で層に分け、**層の中だけで**評価する。",
        "「+20%上昇」の起きやすさが規模で2倍近く違うため",
        "（時価総額帯ごとの正例率は `research/eda_report.html` の目的変数の節を参照。",
        "ここに数字を焼き込むとデータを組み直したときに古い値が残る）。",
        "「同じくらいの規模の銘柄どうしで、どれが伸びるか」を測る。",
        "",
        f"- 層数: {N_STRATA}（日付ごとに `{STRATIFY_BY}` の分位で切る）",
        f"- フォールド: ウォークフォワードと同じ切り方",
        "- `Lift` = PR-AUC ÷ その層の正例率。1.0 なら無意味",
        "",
    ]
    if not d.empty:
        lines += ["## 層の中身", "",
                  f"| 層 | `{STRATIFY_BY}` の範囲 | 平均サンプル数 | 平均正例率 |",
                  "| ---: | --- | ---: | ---: |"]
        for st, g in d[d["model"] == REFERENCE].groupby("stratum"):
            lines.append(f"| {int(st)} | {g['axis_lo'].min():.1f} 〜 "
                         f"{g['axis_hi'].max():.1f} | {g['n'].mean():,.0f} | "
                         f"{g['base_rate'].mean()*100:.1f}% |")
        lines += ["", f"正例率が層によって大きく違えば、`{STRATIFY_BY}` を",
                  "揃えずに評価した数字は規模構成に引きずられていたことになる。", ""]

        lines += [f"## 層内での `{STRATIFY_BY}` 自体の効き", "",
                  f"層は日付ごとに `{STRATIFY_BY}` の分位で切るので、層内でも",
                  "レンジは残る（上の表の範囲を参照）。したがって Lift が完全に",
                  "1.0 まで落ちる必要はない。ただし層を切らずに測ったときより",
                  "はっきり下がっていなければ、規模の効果を取り除けていない。", "",
                  "| 層 | 平均PR-AUC | 平均正例率 | Lift |",
                  "| ---: | ---: | ---: | ---: |"]
        for st, g in d[d["model"] == REFERENCE].groupby("stratum"):
            lines.append(f"| {int(st)} | {g['pr_auc'].mean():.4f} | "
                         f"{g['base_rate'].mean()*100:.1f}% | "
                         f"{g['lift'].mean():.2f}x |")

    lines += ["", "## 層別のモデル成績", "",
              f"`勝敗` は層内で `{STRATIFY_BY}` を上回ったフォールド数。", "",
              f"| 層 | モデル | 平均Lift | Lift@10% | 対`{STRATIFY_BY}` | 勝敗 |",
              "| ---: | --- | ---: | ---: | ---: | :---: |"]
    for s in summary:
        lines.append(f"| {s['stratum']} | {s['model']} | {s['mean_lift']:.2f}x | "
                     f"{s['mean_lift10']:.2f}x | {s['mean_diff']:+.4f} | "
                     f"{s['wins']}勝{s['losses']}敗 |")

    lines += [
        "",
        "## 読み方",
        "",
        "- **Lift が 1.0 付近なら、その層では何も予測できていない**",
        "- 実運用で使えるのは、狙う規模帯の層で Lift が出ている場合だけ",
        "  （全層平均が良くても、買う層で 1.0 なら意味がない）",
        f"- 層内で `{STRATIFY_BY}` の Lift が下がっていれば、",
        "  「規模が大きいほど+20%が起きにくい」だけの効果を取り除けている",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{STRATIFY_BY} で層別した評価")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--min-train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_STRATIFIED.md"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset).sort_values("Date").reset_index(drop=True)
    folds = make_folds(pd.to_datetime(df["Date"]),
                       min_train_months=args.min_train_months,
                       test_months=args.test_months, step_months=args.step_months,
                       embargo_days=EMBARGO_DAYS)
    presets = args.features or DEFAULT_PRESETS
    print(f"[load] {len(df):,}件 / {len(folds)}フォールド / "
          f"{len(presets)}プリセット / {N_STRATA}層")

    res = run(df, presets, folds)
    body = build_report(res, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")
    with open(os.path.join(DATA_DIR, "stratified.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print(f"\n[層内で {STRATIFY_BY} を上回ったモデル]")
    hit = [s for s in res["summary"] if s["wins"] > s["losses"]]
    for s in sorted(hit, key=lambda r: -r["mean_diff"])[:10]:
        print(f"  層{s['stratum']} {s['model']:<34} "
              f"Lift {s['mean_lift']:.2f}x  {s['wins']}勝{s['losses']}敗")
    if not hit:
        print("  なし")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
