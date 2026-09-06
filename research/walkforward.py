#!/usr/bin/env python3
"""
ウォークフォワード評価。

単一分割の問題:
  train_model.py はテスト期間が1つ（約1年）しかない。
  ブートストラップはその期間内のばらつきしか測れないので、
  「特徴量に予測力が無い」のか「その1年がたまたま不利だった」のかを
  区別できない。実際、正例率は訓練 6.19% / テスト 21.66% と3.5倍ずれており、
  単一分割の結果を一般化してよい理由が無い。

ここでやること:
  訓練期間を伸ばしながらテスト窓を前に進め、複数の局面で同じ比較を繰り返す。
  各フォールドで「ベースライン(R_high)との PR-AUC 差」を計算し、
  その符号が何回正になったかを見る（符号検定）。
  差の大きさではなく向きの一貫性を見るので、
  局面ごとに PR-AUC の水準が違っても比較が成立する。

  |--- 訓練 ---|--エンバーゴ--|- テスト -|
  |------ 訓練 ------|--エンバーゴ--|- テスト -|
  |--------- 訓練 ---------|--エンバーゴ--|- テスト -|

エンバーゴはラベル確定に必要な将来日数（既定180営業日）。
訓練最終日のサンプルのラベルが、テスト期間の値を含まないようにする。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import tuning  # noqa: E402
from train_model import (  # noqa: E402
    DATA_DIR, EMBARGO_DAYS, baseline_scores, clean_score, evaluate, fit_models,
)

# 営業日→暦日の換算。年間約250営業日 / 365日 なので 1営業日 ≒ 1.45暦日
TRADING_TO_CALENDAR = 1.45

# ウォークフォワードは学習回数がフォールド数だけ増える。
# 既定は「絶対値 vs 順位版」の対になっているセットに絞る。
# 全部見たいときは --features で明示する。
DEFAULT_PRESETS = [
    "price_only", "rank_price_only",
    # 高値更新日を母集団にしたので「どう抜けたか」が素朴なベースラインになる
    "breakout_only",
    "technical", "rank_technical",
    "all", "rank_all",
    "fundamental", "rank_fundamental",
    # 決算 + バリュエーション + 黒字転換。
    # PER/PBR/ROA を足した効果を見るのが目的なので、既定に入れておく
    "fundamental_v2", "rank_fundamental_v2",
    # 算出できるファンダメンタルズを可能な限り入れたセットと、
    # 今回足したものだけのセット。前者は業種も含む
    "fundamental_v3", "extras_only",
    "valuation_only",
]


def uncovered_presets(presets: List[str]) -> List[str]:
    """
    既定から漏れているプリセットを返す。

    ここは手で維持する一覧なので、プリセットを足しても
    ウォークフォワードで測られないまま気づかない、ということが実際に起きた
    （valuation_only / fundamental_v2 を追加したのに既定に入れ忘れ、
      単一分割でしか評価していなかった）。
    漏れを黙って通さず、実行時に出す。
    """
    return [p for p in F.PRESETS if p not in set(presets)]

REFERENCE = "ベースライン: R_high のみ"

# これを下回るテスト窓は評価しない。
# 少数のサンプルで出した PR-AUC は、勝敗の符号がほぼ運で決まる
MIN_TEST_ROWS = 200
MIN_TEST_POSITIVES = 20


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def make_folds(dates: pd.Series, *, min_train_months: int, test_months: int,
               step_months: int, embargo_days: int) -> List[Fold]:
    """
    訓練窓を伸ばしながら（expanding window）テスト窓を前に進める。

    金融データは履歴が限られるので、古い期間を捨てる rolling ではなく
    全履歴を使う expanding にする。テスト窓は重ねない
    （重ねるとフォールド間が相関し、一致して見えるだけになる）。
    """
    d0 = pd.Timestamp(dates.min())
    dmax = pd.Timestamp(dates.max())
    embargo = pd.Timedelta(days=int(round(embargo_days * TRADING_TO_CALENDAR)))

    # まず (訓練終了, テスト開始) の列を作る。
    starts: List[tuple] = []
    i = 0
    while True:
        train_end = d0 + pd.DateOffset(months=min_train_months + i * step_months)
        test_start = train_end + embargo
        if test_start > dmax:
            break
        starts.append((train_end, test_start))
        i += 1

    # テスト窓の終端は「次のテスト窓の開始の前日」にする。
    # test_start + test_months で計算すると月の長さの違いで隣と数日重なる。
    # 重なると同じサンプルが2つのフォールドに入り、独立でなくなる。
    folds: List[Fold] = []
    for j, (train_end, test_start) in enumerate(starts):
        if j + 1 < len(starts):
            test_end = starts[j + 1][1] - pd.Timedelta(days=1)
        else:
            test_end = min(test_start + pd.DateOffset(months=test_months), dmax)
        test_end = min(test_end, dmax)
        if test_end < test_start:
            continue
        folds.append(Fold(len(folds) + 1, str(d0.date()), str(train_end.date()),
                          str(test_start.date()), str(test_end.date())))
    return folds


def sign_test(wins: int, losses: int) -> float:
    """
    帰無仮説「勝率0.5」の両側二項検定。

    PR-AUC の差の大きさは局面によって水準が違うので平均しにくい。
    符号だけを見れば局面差に影響されない。
    """
    n = wins + losses
    if n == 0:
        return float("nan")
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run(df: pd.DataFrame, presets: List[str], folds: List[Fold]) -> Dict:
    # Optuna の探索結果は全フォールドで共有する。
    # フォールドごとに探索すると計算量が現実的でないうえ、
    # 条件が揃わなくなる。探索は最初のテスト窓より前のデータだけで行っている
    # （research/tuning.py）。
    params_store = tuning.load_params()
    print(f"[tune] 探索済みパラメータ {len(params_store)}件"
          if params_store else "[tune] 探索結果が無いため既定値を使用")
    dates = pd.to_datetime(df["Date"])
    per_fold: List[Dict] = []

    for fold in folds:
        tr = df[(dates >= fold.train_start) & (dates <= fold.train_end)]
        te = df[(dates >= fold.test_start) & (dates <= fold.test_end)]
        ytr, yte = tr["label"].to_numpy(dtype=int), te["label"].to_numpy(dtype=int)

        print(f"\n{'='*70}")
        print(f"[fold {fold.index}] 訓練 〜{fold.train_end} ({len(tr):,}件 正例率 "
              f"{ytr.mean()*100:.2f}%) / テスト {fold.test_start}〜{fold.test_end} "
              f"({len(te):,}件 正例率 {yte.mean()*100:.2f}%)")
        print('='*70)

        # 件数が少なすぎるフォールドは評価に値しない。
        # 母集団を高値更新日に変えてサンプルが減り、
        # 実際に「テスト0件」のフォールドが出た。
        if len(te) < MIN_TEST_ROWS or yte.sum() < MIN_TEST_POSITIVES:
            print(f"  [skip] テストが小さすぎる（{len(te)}件 / 正例 "
                  f"{int(yte.sum())}件。最低 {MIN_TEST_ROWS}件・"
                  f"正例{MIN_TEST_POSITIVES}件）")
            continue
        if len(tr) == 0 or ytr.sum() == 0:
            print("  [skip] 訓練に正例が無い")
            continue

        scores: Dict[str, np.ndarray] = dict(baseline_scores(te))
        for preset in presets:
            cols = F.columns(preset)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"  [skip] {preset}: 列がありません {missing}")
                continue
            models = fit_models(tr, cols, verbose=False, preset=preset,
                                params_store=params_store)
            Xte = te[cols].to_numpy(dtype=float)
            for mname, model in models.items():
                scores[f"{mname} [{preset}]"] = model.predict_proba(Xte)[:, 1]

        rows = [evaluate(name, yte, sc) for name, sc in scores.items()]
        ref = next(r for r in rows if r["name"] == REFERENCE)
        for r in rows:
            r["diff_vs_ref"] = r["pr_auc"] - ref["pr_auc"]
        rows.sort(key=lambda r: -r["pr_auc"])

        print(f"  基準 {REFERENCE}: PR-AUC {ref['pr_auc']:.4f}")
        for r in rows[:4]:
            if r["name"] == REFERENCE:
                continue
            print(f"    {r['name']:<40} {r['pr_auc']:.4f} ({r['diff_vs_ref']:+.4f})")

        per_fold.append({"fold": asdict(fold),
                         "n_train": int(len(tr)), "n_test": int(len(te)),
                         "train_pos_rate": float(ytr.mean()),
                         "test_pos_rate": float(yte.mean()),
                         "results": rows})

    return {"folds": per_fold, "summary": summarize(per_fold)}


def summarize(per_fold: List[Dict]) -> List[Dict]:
    """モデルごとに、全フォールドを通した成績をまとめる。"""
    names: List[str] = []
    for f in per_fold:
        for r in f["results"]:
            if r["name"] not in names:
                names.append(r["name"])

    out = []
    for name in names:
        if name == REFERENCE:
            continue
        diffs, aucs, lifts = [], [], []
        for f in per_fold:
            r = next((x for x in f["results"] if x["name"] == name), None)
            if r is None:
                continue
            diffs.append(r["diff_vs_ref"])
            aucs.append(r["pr_auc"])
            lifts.append(r["lift@5%"])
        if not diffs:
            continue
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        out.append({
            "name": name,
            "n_folds": len(diffs),
            "mean_pr_auc": float(np.mean(aucs)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "worst_diff": float(np.min(diffs)),
            "best_diff": float(np.max(diffs)),
            "mean_lift5": float(np.mean(lifts)),
            "wins": wins,
            "losses": losses,
            "p_sign": sign_test(wins, losses),
        })
    out.sort(key=lambda r: -r["mean_diff"])
    return out


def _wins_needed(n_folds: int) -> int:
    """符号検定で p<0.05 に達するのに必要な勝数。レポートの注記に使う。"""
    for w in range(n_folds, -1, -1):
        if sign_test(w, n_folds - w) >= 0.05:
            return w + 1
    return n_folds


def _significance_note(n_folds: int) -> str:
    need = _wins_needed(n_folds)
    if need > n_folds:
        return (f"**注意: フォールドが{n_folds}個しかないため、符号検定では "
                f"全勝しても p<0.05 に届かない**（{n_folds}戦全勝で "
                f"p={sign_test(n_folds, 0):.3f}）。テスト窓を短くして"
                "フォールド数を増やすか、この検定を判断材料にしないこと。")
    return (f"フォールドが{n_folds}個なので、この検定で p<0.05 に達するには"
            f"**{need}勝以上**が必要（{n_folds}戦）。厳しい基準だが、"
            "少数のフォールドで偶然勝ち越すことは珍しくないため、"
            "この水準を満たさない差は「一貫しない」と扱う。")


def build_report(df: pd.DataFrame, res: Dict, args) -> str:
    per_fold, summary = res["folds"], res["summary"]
    lines = [
        "# ウォークフォワード評価",
        "",
        "`research/walkforward.py` の出力。**実測値のみ**を記載する。",
        "",
        "単一分割（[MODEL_RESULTS.md](MODEL_RESULTS.md)）はテスト期間が1つしかなく、",
        "「特徴量に予測力が無い」のか「その1年が不利だっただけ」かを区別できない。",
        "ここでは訓練窓を伸ばしながらテスト窓を前に進め、複数の局面で同じ比較を繰り返す。",
        "",
        "## 条件",
        "",
        f"- 訓練窓: expanding（最初の {args.min_train_months}ヶ月から開始し、"
        f"{args.step_months}ヶ月ずつ延長）",
        f"- テスト窓: {args.test_months}ヶ月・重複なし",
        f"- エンバーゴ: {EMBARGO_DAYS}営業日"
        f"（≒{int(round(EMBARGO_DAYS * TRADING_TO_CALENDAR))}暦日。ラベル確定に必要な将来日数）",
        f"- 基準: **{REFERENCE}**",
        f"- フォールド数: **{len(per_fold)}**",
        "",
        "## フォールドごとの局面",
        "",
        "| # | 訓練 〜 | 訓練件数 | 訓練正例率 | テスト期間 | テスト件数 | テスト正例率 | 基準PR-AUC |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for f in per_fold:
        ref = next(r for r in f["results"] if r["name"] == REFERENCE)
        fo = f["fold"]
        lines.append(
            f"| {fo['index']} | {fo['train_end']} | {f['n_train']:,} | "
            f"{f['train_pos_rate']*100:.2f}% | {fo['test_start']}〜{fo['test_end']} | "
            f"{f['n_test']:,} | {f['test_pos_rate']*100:.2f}% | {ref['pr_auc']:.4f} |"
        )

    lines += [
        "",
        "正例率がフォールド間で大きく動いていれば、単一分割の結果を一般化できない",
        "という当初の見立てが裏づけられる。基準 PR-AUC の変動も同じことを示す。",
        "",
        "## 総合（全フォールド）",
        "",
        f"`勝敗` は各フォールドで基準を上回った回数。`p` は勝率0.5の両側符号検定。",
        "PR-AUC の水準は局面ごとに違うので、差の平均より**符号の一貫性**を重視する。",
        "",
        _significance_note(len(per_fold)),
        "",
        "| モデル | 平均PR-AUC | 平均差 | 中央値差 | 最悪差 | 勝敗 | p | 判定 |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | --- |",
    ]
    for s in summary:
        if s["p_sign"] < 0.05 and s["wins"] > s["losses"]:
            verdict = "**一貫して上回る**"
        elif s["p_sign"] < 0.05:
            verdict = "一貫して下回る"
        else:
            verdict = "一貫しない"
        lines.append(
            f"| {s['name']} | {s['mean_pr_auc']:.4f} | {s['mean_diff']:+.4f} | "
            f"{s['median_diff']:+.4f} | {s['worst_diff']:+.4f} | "
            f"{s['wins']}勝{s['losses']}敗 | {s['p_sign']:.3f} | {verdict} |"
        )

    lines += [
        "",
        "## フォールド別 PR-AUC（基準との差）",
        "",
    ]
    names = [s["name"] for s in summary]
    header = "| モデル | " + " | ".join(f"F{f['fold']['index']}" for f in per_fold) + " |"
    lines += [header, "| --- |" + " ---: |" * len(per_fold)]
    for name in names:
        cells = []
        for f in per_fold:
            r = next((x for x in f["results"] if x["name"] == name), None)
            cells.append(f"{r['diff_vs_ref']:+.3f}" if r else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 読み方",
        "",
        "- 各フォールドの差が**正負に散らばる**なら、その特徴量セットに安定した優位は無い",
        "- 一貫して正でも差が小さければ、実用上の意味は別途 Lift@5% で見る",
        "- 訓練正例率とテスト正例率の乖離が大きいフォールドほど、",
        "  絶対確率を当てる問題としては難しい（順位付けの問題に変えると緩和する）",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ウォークフォワード評価")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--min-train-months", type=int, default=36)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--step-months", type=int, default=6)
    ap.add_argument("--features", nargs="*", default=None,
                    help=f"評価する特徴量セット。既定: {' '.join(DEFAULT_PRESETS)}")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "MODEL_WALKFORWARD.md"))
    args = ap.parse_args(argv)

    presets = args.features or DEFAULT_PRESETS
    unknown = [p for p in presets if p not in F.PRESETS]
    if unknown:
        raise SystemExit(f"未知の特徴量セット: {unknown}. 利用可能: {sorted(F.PRESETS)}")

    missing = uncovered_presets(presets)
    if missing and not args.features:
        print(f"[warn] ウォークフォワードで評価しないプリセット: {missing}")
        print("       単一分割でしか測られない。意図的でなければ "
              "DEFAULT_PRESETS に足すこと")

    df = pd.read_parquet(args.dataset)
    df = df.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"])
    print(f"[load] {len(df):,}件 / {dates.min().date()} 〜 {dates.max().date()}")

    folds = make_folds(dates, min_train_months=args.min_train_months,
                       test_months=args.test_months, step_months=args.step_months,
                       embargo_days=EMBARGO_DAYS)
    if not folds:
        raise SystemExit("フォールドを作れません。期間が短すぎます")
    print(f"[folds] {len(folds)}フォールド / 特徴量セット {len(presets)}種")

    res = run(df, presets, folds)
    if not res["folds"]:
        raise SystemExit("評価できたフォールドがありません")

    body = build_report(df, res, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")

    with open(os.path.join(DATA_DIR, "walkforward.json"), "w", encoding="utf-8") as fh:
        json.dump({"reference": REFERENCE, "embargoDays": EMBARGO_DAYS,
                   "config": {"min_train_months": args.min_train_months,
                              "test_months": args.test_months,
                              "step_months": args.step_months},
                   **res}, fh, ensure_ascii=False, indent=2)

    print("\n[総合] 基準を一貫して上回ったモデル:")
    hit = [s for s in res["summary"] if s["p_sign"] < 0.05 and s["wins"] > s["losses"]]
    if hit:
        for s in hit:
            print(f"  {s['name']:<40} 平均差 {s['mean_diff']:+.4f} "
                  f"{s['wins']}勝{s['losses']}敗 p={s['p_sign']:.3f}")
    else:
        print("  なし")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
