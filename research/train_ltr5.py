#!/usr/bin/env python3
"""
直近5営業日の組のランキング学習（research/ltr5.py）を学習して research/model/models/ltr5/ に置く（週次。Retrain Weekly の
「Train 5-day ranking model」）。**買いの規則には使わない。** 予測タブとスプレッドシートに「直近5営業日の中の順位」と3条件・
4条件の印を並べ、これからの成績を記録して確かめるためのもの（運用者の選択「画面に並べて確かめる」2026-10-30。
docs/MODEL_ADOPTION_RULES.md §82-6）。

    python3 research/train_ltr5.py

木の形は research/ltr5_params.json（最初は実験117 の W5 の探索の結果。日曜の再学習の「Tune 5-day ranking model」が
research/tune_ltr5.py で毎週探索し直す）。学習する列（基準モデルの列＋値動き4列）で探索したものでなければ学習しない
（前回のモデルのまま。ほかの追加モデルと同じ）。

作るもの（ほかの追加モデルと同じ置き場・同じ名前）
  model.joblib   全期間（収益と20営業日目の終値がある組の行）で学習したモデル。毎日の予測の組の採点に使う
  oof.parquet    out-of-fold のスコア（Code・Date（判断日）・cand_date・score・fold）。窓は基準モデルの out-of-fold と同じ
                 （判断日で分ける。実験117 のずらし0と同じ）
  meta.json      木の形・学習日・学習の期間・確かめの数（git に残る）

確かめ（合わなければ止まる。ltr5.load_pool）: 値動き4列を無作為な300行で日足から直接計算し直すと同じ・判断日に入った候補の
組の R20 が日次の R20 と同じ。
公開ログには集計だけ（件数・日数・NDCG）。銘柄ごとの値は出さない。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import ltr  # noqa: E402
import ltr5  # noqa: E402
import models as M  # noqa: E402
from train_production import DATA_DIR, MODEL_DIR  # noqa: E402


def save(model, meta: dict, oof: pd.DataFrame, out_root: str) -> str:
    """models.save と同じ置き場・同じ名前で保存する（Release への上げ方・日次予測の降ろし方をほかのモデルと揃える）。"""
    return M.save(ltr5.NAME, model, meta, oof[["Code", "Date", "cand_date", "score", "fold"]], root=out_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="直近5営業日の組のランキング学習を学習する（画面と記録に並べるだけ。買いの規則には"
                                             "使わない）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out-dir", default=MODEL_DIR)
    ap.add_argument("--features", default=F.DEFAULT_PRESET)
    ap.add_argument("--params", default=ltr5.PARAMS_PATH)
    args = ap.parse_args(argv)

    t0 = time.time()
    cols = list(F.columns(args.features))
    cols_w = cols + list(ltr5.MOVE_COLS)
    rec = ltr5.load_params(args.params)
    why = ltr.params_mismatch(rec, cols_w)
    if why:
        print(f"[stop] 木の形が学習する列（{len(cols_w)}列・指紋 {F.signature(cols_w)}）で探索したものではない（{why}）。"
              "学習しない（前回のモデルのまま）")
        return 1
    params = rec["params"]

    pool, daily, cal, info = ltr5.load_pool(args.dataset, args.data_dir, cols)
    del daily
    ok = ltr5.trainable(pool)
    print(f"[load] 組の行 {info['rows']:,}件・判断日 {info['days']:,}日（日次の候補 {info['daily_rows']:,}件から）・1日の組 中央 "
          f"{info['per_day_median']:.0f}件・候補の日からの営業日数 "
          + " / ".join(f"{j}日 {x * 100:.1f}%" for j, x in enumerate(info["age_share"])))
    print(f"[check] 値動き4列を日足から直接計算し直す {info['check_cols']['same']} / {info['check_cols']['rows']} 行が同じ・"
          f"判断日に入った候補 {info['check_same_day']['rows']:,}件のうち R20 が日次と合わない行 "
          f"{info['check_same_day']['mismatch']}")
    print(f"[load] 学習に使える組の行（収益と20営業日目の終値がある）{int(ok.sum()):,}件・"
          f"{pool.loc[ok, 'Date'].min().date()} 〜 {pool.loc[ok, 'Date'].max().date()}")
    print(f"[setup] {args.features}（{len(cols)}列＋値動き{len(ltr5.MOVE_COLS)}列・指紋 {F.signature(cols_w)}）/ 種 "
          f"{ltr5.SEED} / 木の形: " + " / ".join(
              f"{k} {params[k]:.4g}" for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample",
                                                "colsample_bytree", "reg_alpha", "reg_lambda") if k in params))
    print(f"[setup] 木の形の出どころ: {rec.get('_source', '?')}")

    oof = ltr5.oof_scores(pool, cols_w, params)
    g = ltr.grades(oof["Date"], oof[ltr5.R20].to_numpy(dtype=float))
    ndcg = ltr.ndcg_at_k(oof["Date"], oof["score"], g, ltr5.NDCG_K, ltr5.MIN_POOL)
    rnd = ltr.ndcg_at_k(oof["Date"], np.zeros(len(oof)), g, ltr5.NDCG_K, ltr5.MIN_POOL)
    print(f"[oof] {oof['fold'].nunique()}窓・組の行 {len(oof):,}件・{oof['Date'].min().date()} 〜 "
          f"{oof['Date'].max().date()} / 組の中の NDCG@{ltr5.NDCG_K}（候補{ltr5.MIN_POOL}件以上の組）{ndcg:.4f}"
          f"（銘柄コードの順に並べたとき {rnd:.4f}）")

    tr = pool[ok]
    model = ltr.fit(tr, cols_w, params, ltr5.SEED)
    meta = {
        "algo": ltr5.NAME, "name": ltr5.JA, "short": ltr5.SHORT, "note": ltr5.NOTE,
        "trainedAt": pd.Timestamp.utcnow().isoformat(),
        "kind": "lambdarank",
        "target": ("判断日までの直近5営業日の候補の組の中の、判断日の翌営業日の寄りで買い20営業日目の終値で売った収益の5分位"
                   "（0〜4）"),
        "pool": {"days": ltr5.K, "moveCols": list(ltr5.MOVE_COLS)},
        "preset": args.features, "features": cols_w, "params": params, "seed": ltr5.SEED,
        "nTrain": int(len(tr)), "nDays": int(tr["Date"].nunique()),
        "trainFrom": str(tr["Date"].min().date()), "trainTo": str(tr["Date"].max().date()),
        "nOof": int(len(oof)), "oofFolds": int(oof["fold"].nunique()),
        "oofMetrics": {"ndcg5": round(ndcg, 4), "ndcg5CodeOrder": round(rnd, 4)},
        "checks": {"moveCols": info["check_cols"], "sameDay": info["check_same_day"]},
        "tuning": rec.get("_cv", {}), "tuningSource": rec.get("_source"),
    }
    d = save(model, meta, oof, args.out_dir)
    size = os.path.getsize(os.path.join(d, "model.joblib")) / 1e6
    print(f"[done] {os.path.relpath(d)} / {size:.2f}MB / {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
