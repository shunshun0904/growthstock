#!/usr/bin/env python3
"""
ランキング学習（research/ltr.py）を学習して research/model/models/ltr/ に置く（週次。Retrain Weekly の
「Train ranking model」）。**買いの規則には使わない。** 画面と実運用の追跡に「両方使う形」を並べるためのもの
（運用者の選択「まず並べて見る」2026-10-24。docs/MODEL_ADOPTION_RULES.md §78-9）。

    python3 research/train_ltr.py

木の形は research/ltr_params.json（最初は実験114 の探索の結果。探索し直すのは research/tune_ltr.py。§78-10）。学習する列で
探索したものでなければ学習しない（前回のモデルのまま。ほかの追加モデルと同じ）。

作るもの（ほかの追加モデルと同じ置き場・同じ名前）
  model.joblib   全期間（ラベルと20営業日目の終値がある行）で学習したモデル。毎日の候補の採点に使う
  oof.parquet    out-of-fold のスコア（Code・Date・score・fold）。窓は基準モデルの out-of-fold と同じ。画面の百分位の基準と、
                 実験114 の L（種42・ずらし0）との突き合わせ（exp=e114_ltr_retest.py args="--check-ltr"）に使う
  meta.json      木の形・学習日・学習の期間・確かめの数（git に残る）

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
import models as M  # noqa: E402
from train_production import DATA_DIR, MODEL_DIR  # noqa: E402


def save(model, meta: dict, oof: pd.DataFrame, out_root: str) -> str:
    """models.save と同じ置き場・同じ名前で保存する（Release への上げ方・日次予測の降ろし方をほかのモデルと揃える）。"""
    return M.save(ltr.NAME, model, meta, oof[["Code", "Date", "score", "fold"]], root=out_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ランキング学習を学習する（画面と追跡に並べるだけ。買いの規則には使わない）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out-dir", default=MODEL_DIR)
    ap.add_argument("--features", default=F.DEFAULT_PRESET)
    ap.add_argument("--params", default=ltr.PARAMS_PATH)
    args = ap.parse_args(argv)

    t0 = time.time()
    cols = F.columns(args.features)
    rec = ltr.load_params(args.params)
    why = ltr.params_mismatch(rec, cols)
    if why:
        print(f"[stop] 木の形が学習する列（{len(cols)}列・指紋 {F.signature(cols)}）で探索したものではない（{why}）。"
              "学習しない（前回のモデルのまま）")
        return 1
    params = rec["params"]

    ds = ltr.load_frame(args.dataset, args.data_dir, cols)
    ok = ltr.trainable(ds)
    sz = ds[ok].groupby("Date").size()
    print(f"[load] {len(ds):,}件（ラベルあり）のうち20営業日目の終値がある {int(ok.sum()):,}件・{len(sz):,}日・"
          f"{ds.loc[ok, 'Date'].min().date()} 〜 {ds.loc[ok, 'Date'].max().date()}")
    print(f"[load] 1日の候補 中央 {int(sz.median())}件・5件以上の日 {np.mean(sz >= 5) * 100:.1f}%・1件だけの日 "
          f"{np.mean(sz == 1) * 100:.1f}%")
    print(f"[setup] {args.features}（{len(cols)}列・指紋 {F.signature(cols)}）/ 種 {ltr.SEED} / 木の形: "
          + " / ".join(f"{k} {params[k]:.4g}" for k in ("learning_rate", "num_leaves", "min_child_samples", "subsample",
                                                        "colsample_bytree", "reg_alpha", "reg_lambda") if k in params))
    print(f"[setup] 木の形の出どころ: {rec.get('_source', '?')}")

    oof = ltr.oof_scores(ds, cols, params)
    g = ltr.grades(oof["Date"], oof[ltr.R20].to_numpy(dtype=float))
    ndcg = ltr.ndcg_at_k(oof["Date"], oof["score"], g)
    rnd = ltr.ndcg_at_k(oof["Date"], np.zeros(len(oof)), g)
    print(f"[oof] {oof['fold'].nunique()}窓・{len(oof):,}件・{oof['Date'].min().date()} 〜 {oof['Date'].max().date()} / "
          f"日付内 NDCG@5（候補5件以上の日）{ndcg:.4f}（銘柄コードの順に並べたとき {rnd:.4f}）")

    model = ltr.fit(ds[ok], cols, params)
    tr = ds[ok]
    meta = {
        "algo": ltr.NAME, "name": ltr.JA, "short": ltr.SHORT, "note": ltr.NOTE,
        "trainedAt": pd.Timestamp.utcnow().isoformat(),
        "kind": "lambdarank",
        "target": "同じ日の候補の中の、翌営業日の寄りで買い20営業日目の終値で売った収益の5分位（0〜4）",
        "preset": args.features, "features": cols, "params": params, "seed": ltr.SEED,
        "dayUpper": ltr.DAY_UPPER,
        "nTrain": int(len(tr)), "nDays": int(tr["Date"].nunique()),
        "trainFrom": str(tr["Date"].min().date()), "trainTo": str(tr["Date"].max().date()),
        "nOof": int(len(oof)), "oofFolds": int(oof["fold"].nunique()),
        "oofMetrics": {"ndcg5": round(ndcg, 4), "ndcg5CodeOrder": round(rnd, 4)},
        "tuning": rec.get("_cv", {}), "tuningSource": rec.get("_source"),
    }
    d = save(model, meta, oof, args.out_dir)
    size = os.path.getsize(os.path.join(d, "model.joblib")) / 1e6
    print(f"[done] {os.path.relpath(d)} / {size:.2f}MB / {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
