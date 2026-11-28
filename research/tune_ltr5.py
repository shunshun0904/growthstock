#!/usr/bin/env python3
"""
直近5営業日の組のランキング学習（research/ltr5.py）の木の形を探索し直して research/ltr5_params.json に書く。
**買いの規則には使わない。**

運用者の選択（2026-10-30）「画面に並べて確かめる」（docs/MODEL_ADOPTION_RULES.md §82-6）: 日曜の週次（Retrain Weekly の
「Tune 5-day ranking model」。tune=yes の回）で毎週探索する。探索は実験117 の W5 と同じ（ltr5.tune: 探索の期間は run_tuning.py
と同じ打ち切り・判断日を暦月の塊にした5分割（前後4営業日を訓練から外す）・種0・50試行・木200本固定・探索空間は ltr.search_space・
目的は検証分割の組の中の NDCG@5）。実験117 の実測で 31.6分。

    python3 research/tune_ltr5.py

書くのは探索が最後まで終わったときだけ（落ちたり時間切れになったりしたら、前の木の形のまま。学習は train_ltr5.py が
その木の形で行う）。書く形は ltr_params.json と同じ（params・列の指紋 _features_sig・列数 _n_features・探索の打ち切り
_cutoff・探索の記録 _cv・出どころ _source）。列は基準モデルの列＋値動き4列。
公開ログには集計だけ（件数・日数・NDCG・木の形）。銘柄ごとの値は出さない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import ltr  # noqa: E402
import ltr5  # noqa: E402
import tune_ltr as TL  # noqa: E402
from train_production import DATA_DIR  # noqa: E402


def source_note(cv: dict, cutoff: str, env=None) -> str:
    """ltr5_params.json の _source（どの回の探索か。Actions なら workflow の名前・run・コミット）。"""
    env = os.environ if env is None else env
    run = env.get("GITHUB_RUN_ID")
    where = (f"{env.get('GITHUB_WORKFLOW', 'Actions')} run {run}・{env.get('GITHUB_SHA', '')[:7]}" if run else "手元")
    return (f"週次の探索（{where}。{cutoff} までの組の行 {cv['n_rows']:,}件・{cv['n_trials']}試行・{cv['n_splits']}分割・"
            f"目的は検証分割の組の中の NDCG@{ltr5.NDCG_K}）")


def record(rec: dict, cols_w, cutoff: str, env=None) -> dict:
    """ltr5_params.json に書く形（試行ごとの値 _trials は入れない）。"""
    return {"_source": source_note(rec["_cv"], cutoff, env), "_features_sig": F.signature(list(cols_w)),
            "_n_features": len(cols_w), "_cutoff": cutoff, "_cv": rec["_cv"], "params": rec["params"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="直近5営業日の組のランキング学習の木の形を探索し直す（画面と記録に並べるだけ。"
                                             "買いの規則には使わない）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", default=F.DEFAULT_PRESET)
    ap.add_argument("--out", default=ltr5.PARAMS_PATH)
    ap.add_argument("--trials", type=int, default=ltr5.TUNE_TRIALS)
    args = ap.parse_args(argv)

    t0 = time.time()
    cols = list(F.columns(args.features))
    cols_w = cols + list(ltr5.MOVE_COLS)
    pool, daily, cal, info = ltr5.load_pool(args.dataset, args.data_dir, cols)
    cutoff = ltr.search_cutoff(daily["Date"])
    del daily
    sub, no_label = ltr5.search_rows(pool, cutoff)
    del pool
    folds = ltr5.search_folds(sub, cal)
    print(f"[search] {args.features}（{len(cols)}列＋値動き{len(ltr5.MOVE_COLS)}列・指紋 {F.signature(cols_w)}）/ 組の行 "
          f"{info['rows']:,}件・判断日 {info['days']:,}日・1日の組 中央 {info['per_day_median']:.0f}件 ｜ 確かめ: 値動きの列 "
          f"{info['check_cols']['same']} / {info['check_cols']['rows']} 行が同じ・判断日に入った候補の R20 の合わない行 "
          f"{info['check_same_day']['mismatch']}", flush=True)
    print(f"[search] 探索の期間 〜{cutoff.date()}（run_tuning.py と同じ打ち切り）: 組の行 {len(sub):,}件・"
          f"{sub['Date'].nunique():,}日（ラベルが無くて外した {no_label}件）/ {args.trials}試行 × {len(folds)}分割"
          f"（暦月の塊・前後{ltr5.PURGE}営業日を訓練から外す）・種 {ltr5.TUNE_SEED}: "
          + " / ".join(f"訓練 {len(tr):,}・検証 {len(va):,}" for tr, va in folds), flush=True)
    rec = ltr5.tune(sub, cols_w, folds, n_trials=args.trials)
    cv = rec["_cv"]
    prev = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh).get("params")
    out = record(rec, cols_w, str(cutoff.date()))
    TL.write_json(args.out, out)
    print(f"[search] 最良は試行{cv['best_trial']}・分割平均 NDCG@{ltr5.NDCG_K} {cv['ndcg']:.4f}"
          f"（分割ごと {' / '.join(f'{x:.4f}' for x in cv['ndcg_folds'])}）/ {cv['minutes']}分")
    print(f"[search] 木の形: {TL.tree_line(rec['params'])}"
          + ("（前と同じ）" if prev == rec["params"] else ("（前から変わった）" if prev else "")))
    print(f"[done] {os.path.relpath(args.out)} / {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
