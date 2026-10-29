#!/usr/bin/env python3
"""
ランキング学習（research/ltr.py）の木の形を探索し直して research/ltr_params.json に書く。**買いの規則には使わない。**

運用者の選択「毎週探索し直す」（2026-10-24。docs/MODEL_ADOPTION_RULES.md §78-10）: 基準モデルと同じく、週次の再学習で
毎回探索する。探索は実験114 の L と同じ（ltr.tune: ホールドアウトより前・年×時価総額帯で層別・日付単位の5分割・種0・
50試行・木200本固定・探索空間は tuning.tune と同じ・目的は検証分割の日付内 NDCG@5）。Actions で約27分（実験114 の実測）。
Retrain Weekly に入れるのは実験114c の確かめの後（いきなり切り替えない）。

    python3 research/tune_ltr.py

書くのは探索が最後まで終わったときだけ（落ちたり時間切れになったりしたら、前の木の形のまま。学習は train_ltr.py が
その木の形で行う）。書く形は今の ltr_params.json と同じ（params・列の指紋 _features_sig・列数 _n_features・探索の打ち切り
_cutoff・探索の記録 _cv・出どころ _source）。
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
from train_production import DATA_DIR  # noqa: E402

TREE_KEYS = ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha",
             "reg_lambda")


def tree_line(params: dict) -> str:
    return " / ".join(f"{k} {params[k]:.4g}" for k in TREE_KEYS if k in params)


def source_note(cv: dict, cutoff: str, env=None) -> str:
    """ltr_params.json の _source（どの回の探索か。Actions なら workflow の名前・run・コミット）。"""
    env = os.environ if env is None else env
    run = env.get("GITHUB_RUN_ID")
    where = (f"{env.get('GITHUB_WORKFLOW', 'Actions')} run {run}・{env.get('GITHUB_SHA', '')[:7]}" if run else "手元")
    return (f"週次の探索（{where}。{cutoff} までの {cv['n_rows']:,}件・{cv['n_trials']}試行・{cv['n_splits']}分割・"
            f"目的は検証分割の日付内 NDCG@{ltr.NDCG_K}）")


def record(rec: dict, cols, cutoff: str, env=None) -> dict:
    """ltr_params.json に書く形（試行ごとの値 _trials は入れない）。"""
    return {"_source": source_note(rec["_cv"], cutoff, env), "_features_sig": F.signature(list(cols)),
            "_n_features": len(cols), "_cutoff": cutoff, "_cv": rec["_cv"], "params": rec["params"]}


def write_json(path: str, obj: dict) -> None:
    """書きかけのファイルを残さない（一時ファイルに書いて置き換える）。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ランキング学習の木の形を探索し直す（画面と追跡に並べるだけ。買いの規則には使わない）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", default=F.DEFAULT_PRESET)
    ap.add_argument("--out", default=ltr.PARAMS_PATH)
    ap.add_argument("--trials", type=int, default=ltr.TUNE_TRIALS)
    args = ap.parse_args(argv)

    t0 = time.time()
    cols = F.columns(args.features)
    ds = ltr.load_frame(args.dataset, args.data_dir, cols)
    cutoff = ltr.search_cutoff(ds["Date"])
    sub = ltr.search_rows(ds, cutoff)
    del ds
    print(f"[search] {args.features}（{len(cols)}列・指紋 {F.signature(cols)}）/ 探索の期間 〜{cutoff.date()}"
          f"（run_tuning.py と同じ打ち切り）: ラベルと20営業日目の終値がある {len(sub):,}件・{sub['Date'].nunique():,}日 / "
          f"{args.trials}試行 × {ltr.TUNE_SPLITS}分割・種 {ltr.TUNE_SEED}", flush=True)
    rec = ltr.tune(sub, cols, n_trials=args.trials)
    cv = rec["_cv"]
    prev = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh).get("params")
    out = record(rec, cols, str(cutoff.date()))
    write_json(args.out, out)
    print(f"[search] 最良は試行{cv['best_trial']}・分割平均 NDCG@{ltr.NDCG_K} {cv['ndcg']:.4f}"
          f"（分割ごと {' / '.join(f'{x:.4f}' for x in cv['ndcg_folds'])}）/ {cv['minutes']}分")
    print(f"[search] 木の形: {tree_line(rec['params'])}"
          + ("（前と同じ）" if prev == rec["params"] else ("（前から変わった）" if prev else "")))
    print(f"[done] {os.path.relpath(args.out)} / {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
