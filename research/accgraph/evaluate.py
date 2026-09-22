#!/usr/bin/env python3
"""
評価パイプライン。ベースライン比較の入口。

  python3 research/accgraph/evaluate.py
  python3 research/accgraph/evaluate.py --benchmark sector --horizon 10

やること:
  1. データセットを読む
  2. リーク検査を通す（失敗したらここで止まる）
  3. Purged / Embargo つきウォークフォワードで分割する
  4. 特徴量セット × モデルの全組み合わせを学習・評価する
  5. 分類指標と、取引コスト控除後のバックテストを並べる
  6. docs/ACCGRAPH_BASELINE.md に書き出す

GNN はここには入っていない。まずこの表を埋め、
GNN がこの表を上回るかどうかで有効性を判断する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, RESEARCH)

from accgraph import backtest, baselines, build, labels as L, leakage, splits  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "ACCGRAPH_BASELINE.md")
OUT_JSON = os.path.join(RESEARCH, "_data", "accgraph", "baseline_results.json")

#: 比較する特徴量セット。`_rank` は同じ発表日の中で順位に直したもので、
#: 規模と地合いを抜いても情報が残るかを見るために並べる
FEATURE_SETS = ("latest", "seq", "latest_rank", "seq_rank")
#: 既定から外してあるが --feature-sets で指定できるもの。
#: nodes（エッジ特徴量を外したセット）は一度測って寄与が小さかった
EXTRA_FEATURE_SETS = ("nodes", "nodes_rank")
MODEL_ORDER = ("majority", "logit", "lgbm", "mlp")
CLASSES = np.array([0, 1, 2])


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def classification_metrics(y: np.ndarray, proba: np.ndarray) -> Dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 roc_auc_score)

    pred = CLASSES[proba.argmax(axis=1)]
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=CLASSES,
                                   zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "class_share": [float((y == c).mean()) for c in CLASSES],
    }
    # 全クラスが揃っていないと ovr の AUC は定義できない
    if len(np.unique(y)) == len(CLASSES):
        out["roc_auc_macro"] = float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro",
                          labels=CLASSES))
    else:
        out["roc_auc_macro"] = float("nan")
    return out


def _mean(values: List[float]) -> float:
    v = [x for x in values if not (x is None or np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def evaluate(meta: pd.DataFrame, node_feat: np.ndarray, edge_feat: np.ndarray,
             period_mask: np.ndarray, *, benchmark: str = "topix",
             horizon: int = 20, max_horizon: int = 20,
             feature_sets=FEATURE_SETS, models=MODEL_ORDER,
             cost_bps: float = backtest.DEFAULT_COST_BPS,
             min_train_months: int = 48, test_months: int = 12,
             step_months: int = 12, min_test_rows: int = 200,
             seed: int = 0) -> Dict:
    y_col = f"y_{benchmark}_{horizon}d"
    excess_col = f"excess_{benchmark}_{horizon}d"
    if y_col not in meta.columns:
        raise SystemExit(f"{y_col} がデータセットにありません")

    usable = meta[y_col].to_numpy() >= 0
    meta = meta[usable].reset_index(drop=True)
    node_feat, edge_feat, period_mask = (node_feat[usable], edge_feat[usable],
                                         period_mask[usable])
    y = meta[y_col].to_numpy().astype(int)
    print(f"[eval] {y_col}: {len(meta):,}件 / クラス比 "
          + " ".join(f"{L.CLASS_NAMES[c]} {float((y == c).mean()) * 100:.1f}%"
                     for c in CLASSES))

    folds, train_masks, test_masks = splits.walk_forward(
        meta, min_train_months=min_train_months, test_months=test_months,
        step_months=step_months, embargo_trading_days=max_horizon,
        min_test_rows=min_test_rows)
    print(splits.describe(folds))

    results: Dict[str, Dict] = {}
    for fs in feature_sets:
        X, names = baselines.flatten(node_feat, edge_feat, period_mask, meta,
                                     kind=fs)
        print(f"[eval] 特徴量セット {fs}: {X.shape[1]:,}次元")
        for mname in models:
            per_fold = []
            oos_rows, oos_proba = [], []
            for f, tr, te in zip(folds, train_masks, test_masks):
                fit = baselines.MODELS[mname]
                model = fit(X[tr], y[tr], CLASSES, seed=seed,
                            feature_names=names)
                proba = baselines.align_proba(model, X[te], CLASSES)
                m = classification_metrics(y[te], proba)
                per_fold.append({"fold": f.to_dict(), "metrics": m})
                oos_rows.append(meta[te])
                oos_proba.append(proba)
                print(f"  {fs:<7} {mname:<9} fold{f.index} "
                      f"acc={m['accuracy']:.4f} auc={m['roc_auc_macro']:.4f}")

            # フォールドのテスト窓は重ならないので、つなげば連続した
            # アウトオブサンプルの系列になる。20営業日保有だと1フォールドに
            # 十数コホートしか取れず、フォールド単位のシャープレシオは
            # ほぼ運で決まる。損益はつないだ系列で測る
            oos_meta = pd.concat(oos_rows, ignore_index=True)
            oos_p = np.concatenate(oos_proba, axis=0)
            oos_y = oos_meta[y_col].to_numpy().astype(int)
            pooled_metrics = classification_metrics(oos_y, oos_p)
            pooled_bt = {
                rule: backtest.run(
                    oos_meta, oos_p, CLASSES, excess_col=excess_col,
                    horizon=horizon, max_horizon=max_horizon, rule=rule,
                    cost_bps=cost_bps).to_dict()
                for rule in ("predicted_up", "topk", "all")
            }
            accs = [p["metrics"]["accuracy"] for p in per_fold]
            print(f"  {fs:<7} {mname:<9} pooled "
                  f"acc={pooled_metrics['accuracy']:.4f} "
                  f"auc={pooled_metrics['roc_auc_macro']:.4f} "
                  f"sharpe={pooled_bt['predicted_up']['sharpe']:.2f} "
                  f"({pooled_bt['predicted_up']['n_cohorts']}コホート)")
            results[f"{fs}/{mname}"] = {
                "feature_set": fs, "model": mname,
                "folds": per_fold,
                "fold_accuracy_mean": _mean(accs),
                "fold_accuracy_std": float(np.std(accs)) if accs else float("nan"),
                "pooled": {"metrics": pooled_metrics, "backtest": pooled_bt},
            }

    return {
        "benchmark": benchmark, "horizon": horizon, "cost_bps": cost_bps,
        "n_samples": int(len(meta)),
        "period": [str(pd.to_datetime(meta["entry_date"]).min().date()),
                   str(pd.to_datetime(meta["entry_date"]).max().date())],
        "class_share": [float((y == c).mean()) for c in CLASSES],
        "folds": [f.to_dict() for f in folds],
        "results": results,
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def to_markdown(res: Dict) -> str:
    b = res["benchmark"]
    lines = [
        "# 会計フローグラフ — ベースライン比較",
        "",
        "`research/accgraph/evaluate.py` の出力。**実行した結果のみ**を載せる。",
        "",
        f"- ベンチマーク: {'TOPIX' if b == 'topix' else '業種指数'}控除",
        f"- 保有期間: {res['horizon']}営業日（発表翌営業日の始値でエントリー）",
        f"- 取引コスト: 往復 {res['cost_bps']:.0f}bp",
        f"- サンプル: {res['n_samples']:,}件 "
        f"({res['period'][0]} 〜 {res['period'][1]})",
        "- クラス比: "
        + " / ".join(f"{L.CLASS_NAMES[i]} {s * 100:.1f}%"
                     for i, s in enumerate(res["class_share"])),
        "",
        "## 分割",
        "",
        "訓練窓を伸ばしながらテスト窓を前に進める。テスト窓は重ねない。",
        "purge はラベルがテスト期間の価格で決まるため訓練から外した件数、",
        "embargo はテスト直前の緩衝期間で外した件数。",
        "",
        "| # | 訓練 | 訓練件数 | テスト | テスト件数 | purge | embargo |",
        "| :-: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for f in res["folds"]:
        lines.append(
            f"| {f['index']} | {f['train_start']}〜{f['train_end']} "
            f"| {f['n_train']:,} | {f['test_start']}〜{f['test_end']} "
            f"| {f['n_test']:,} | {f['n_purged']:,} | {f['n_embargoed']:,} |")

    lines += [
        "",
        "## アウトオブサンプル（全フォールドを連結）",
        "",
        "テスト窓は重ならないので、連結すると連続した検証系列になる。",
        "フォールド単位の損益は20営業日保有だと十数コホートしか取れず、",
        "運で決まってしまうため、損益はこの連結系列で測る。",
        "",
        "| 特徴量 | モデル | Accuracy | フォールド間の標準偏差 | ROC-AUC(macro) "
        "| F1(macro) | Sharpe(上昇判定) | 累積 | 最大DD | 1トレード平均 | t値 "
        "| トレード数 | コホート数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: |",
    ]
    order = sorted(res["results"].items(),
                   key=lambda kv: -(kv[1]["pooled"]["metrics"]["accuracy"] or 0))
    for _, r in order:
        m = r["pooled"]["metrics"]
        bt = r["pooled"]["backtest"]["predicted_up"]
        lines.append(
            f"| {r['feature_set']} | {r['model']} | {m['accuracy']:.4f} "
            f"| {r['fold_accuracy_std']:.4f} | {m['roc_auc_macro']:.4f} "
            f"| {m['f1_macro']:.4f} | {bt['sharpe']:.2f} "
            f"| {bt['cum_return'] * 100:.1f}% | {bt['max_drawdown'] * 100:.1f}% "
            f"| {bt['mean_net_per_trade'] * 100:.3f}% "
            f"| {bt['t_stat_clustered']:.2f} "
            f"| {bt['n_trades']:,} | {bt['n_cohorts']:,} |")

    # 「全件買い」は判定を使わないので、どのモデルの行でも同じ値になる。
    # 念のため全行から取って、ばらついていたら気づけるようにしておく
    all_sharpes = [r["pooled"]["backtest"]["all"]["sharpe"]
                   for r in res["results"].values()]
    all_sharpe = _mean(all_sharpes)
    finite = [x for x in all_sharpes if not np.isnan(x)]
    spread = (max(finite) - min(finite)) if finite else 0.0
    lines += [
        "",
        f"参考: 全件買い（判定を使わない）の Sharpe は {all_sharpe:.2f}。"
        "モデルがこれを超えていなければ、判定に意味は無い。"
        + (f"（モデル間で {spread:.2f} ばらついている。"
           "同じ値になるはずなので実装を疑うこと）" if spread > 1e-6 else ""),
        "",
        "## 読み方",
        "",
        "- `majority` は訓練期間のクラス比率をそのまま返すだけのモデル。"
        "これを上回らない行は、特徴量に情報が無いということ。",
        "- Accuracy はクラス比に引きずられる。中立クラスを当てているだけでも上がるので、"
        "ROC-AUC とバックテストを併せて見る。",
        "- `latest` は当該四半期のグラフだけ、`seq` は過去8四半期ぶん、"
        "`nodes` はエッジ特徴量を外したもの。"
        "`seq` が `latest` を上回らなければ、系列を持つ意味が無い。",
        "- `_rank` 付きは、同じ発表日の中で各特徴量を順位に直したもの。"
        "その日の地合いと企業規模の絶対水準が消えるので、"
        "会計構造そのものに情報があるかを分離して測れる。"
        "順位版が素の版とほぼ同じなら、拾っていたのは規模ではない。",
        "- t値は同じエントリー日のトレードをクラスタした値。"
        "日をまたいで独立とみなすと過大評価になる。",
        "- 効率的市場仮説の下では、この種の予測が安定して当たるとは想定しない。"
        "Accuracy 55% を大きく超える行が出たら、まずリークを疑うこと。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="会計フローグラフのベースライン評価")
    ap.add_argument("--data-dir", default=build.OUT_DIR)
    ap.add_argument("--benchmark", choices=["topix", "sector"], default="topix")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--max-horizon", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=backtest.DEFAULT_COST_BPS)
    ap.add_argument("--min-train-months", type=int, default=48)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=12)
    ap.add_argument("--min-test-rows", type=int, default=200,
                    help="これを下回るテスト窓は評価しない"
                         "（少数だと指標が運で決まる）")
    ap.add_argument("--models", nargs="*", default=list(MODEL_ORDER))
    ap.add_argument("--feature-sets", nargs="*", default=list(FEATURE_SETS))
    ap.add_argument("--all-stocks", action="store_true",
                    help="流動性フィルタを外す")
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    meta, nf, ef, pm = build.load(args.data_dir, liquid_only=not args.all_stocks)
    print(f"[eval] 読み込み {len(meta):,}件")

    leakage.run_all(meta, node_feat=nf, period_mask=pm)

    res = evaluate(meta, nf, ef, pm, benchmark=args.benchmark,
                   horizon=args.horizon, max_horizon=args.max_horizon,
                   feature_sets=tuple(args.feature_sets),
                   models=tuple(args.models), cost_bps=args.cost_bps,
                   min_train_months=args.min_train_months,
                   test_months=args.test_months, step_months=args.step_months,
                   min_test_rows=args.min_test_rows)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(res))
    print(f"[eval] 書き出し {args.out_md} / {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
