#!/usr/bin/env python3
"""
画面に並べる複数モデルを学習する（週次）。

    python3 research/train_multi.py
    python3 research/train_multi.py --models lgbm,logit      # 一部だけ

なぜ train_production.py と別にするか
-----------------------------------
train_production.py は本番の学習パスとして動いていて、日次予測が読む
research/model/{model.txt, meta.json, oof.parquet} を作っている。そこに
5モデルぶんの分岐を入れると、失敗したときに日次予測まで止まる。

ここは research/model/models/<algo>/ 以下だけを作る。このスクリプトが
丸ごと失敗しても、基準モデル（lgbm）による日次予測は動き続ける。

基準モデルはここで作り直さない
---------------------------
既定の対象は models.EXTRA（xgb / cat / logit / mlp）で、lgbm は入らない。
同じ LightGBM を別のパラメータで当てはめると、内側検証の PR-AUC はほぼ
同じ（0.4016 対 0.4029）のに上位10%の重複が 52.8% しかなく、画面に
「LightGBM」が2本並んで最大62pt ずれる。順位・帯・較正・SHAP の基準は
本番モデルなので、画面の LightGBM もそれに一本化する。
--models lgbm を明示すれば比較用に作れる（画面には使わない）。

何を作るか
---------
モデルごとに3つ。

  model.joblib   全期間で学習した最終モデル
  oof.parquet    ウォークフォワードの out-of-fold スコア
                 （画面の pctHistorical の基準。モデルごとに別の分布）
  meta.json      較正表・スコア帯の実績・パラメータ・学習日

out-of-fold を各行「その行より前のデータだけで学習したモデル」で採点する
のは train_production.py と同じ。全期間モデルの自己採点を使うと、
訓練データを当てているだけの楽観的な表になる。

アンサンブルはしない
-------------------
5つのスコアを混ぜて1つにはしない。運用は手動の指値なので、画面に並べて
人間が統合判断する。実測で lgbm と logit のスコア相関は 0.412、
上位10%の重複は12%しかなく、ほぼ別の銘柄を選んでいる。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import sweep_design as S  # noqa: E402
import tuning_multi as TM  # noqa: E402
from train_production import (  # noqa: E402
    DATA_DIR, MODEL_DIR, OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS,
    OOF_TEST_MONTHS, calibration, score_bands,
)


def oof_scores(algo: str, ds: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    各行を「その行より前のデータだけで学習したモデル」で採点する。

    分割は train_production.oof_scores と同じ（36ヶ月 / 6ヶ月 / 6ヶ月、
    エンバーゴ = ラベル確定に要る営業日数）。ここをモデルごとに変えると、
    画面に並べたときの pctHistorical が互いに比較できなくなる。
    """
    import walkforward as WF

    folds = WF.make_folds(pd.to_datetime(ds["Date"]),
                          min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(ds["Date"])
    parts = []
    for f in folds:
        tr = ds[(d <= pd.Timestamp(f.train_end)) & ds["label"].notna()]
        te = ds[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                & ds["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        m = M.fit(algo, tr[cols].to_numpy(dtype=float),
                  tr["label"].to_numpy(dtype=int), cols)
        part = te[["Code", "Date", "label", "ref_end", "ref_rise"]].copy()
        part["score"] = M.predict(m, te[cols].to_numpy(dtype=float))
        parts.append(part)
    if not parts:
        raise SystemExit("out-of-fold を作れません。期間が短すぎます")
    return pd.concat(parts, ignore_index=True)


def train_one(algo: str, ds: pd.DataFrame, cols: List[str],
              out_root: str) -> Dict:
    """1モデルぶんを学習して保存し、要約を返す。"""
    t0 = time.time()
    oof = oof_scores(algo, ds, cols)
    bands = score_bands(oof)
    calib = calibration(oof)

    y = ds["label"].to_numpy(dtype=int)
    model = M.fit(algo, ds[cols].to_numpy(dtype=float), y, cols)

    params = TM.params_for(algo)
    store = TM.load().get(algo, {})
    meta = {
        "algo": algo,
        "name": M.JA.get(algo, algo),
        "note": M.NOTE.get(algo, ""),
        "trainedAt": pd.Timestamp.utcnow().isoformat(),
        "label": B.DEFAULT_RISE.name,
        "preset": "all",
        "features": cols,
        "params": params,
        "nTrain": int(len(ds)),
        "trainFrom": str(ds["Date"].min().date()),
        "trainTo": str(ds["Date"].max().date()),
        "positiveRate": round(float(ds["label"].mean()), 4),
        "nOof": int(len(oof)),
        "calibration": calib,
        "scoreBands": bands,
        # 探索の記録。どの条件で選ばれたパラメータかを後から辿れるように
        "tuning": store.get("_cv", {}),
    }
    d = M.save(algo, model, meta, oof, root=out_root)
    size = os.path.getsize(os.path.join(d, "model.joblib")) / 1e6
    return {"algo": algo, "dir": d, "size_mb": round(size, 2),
            "n_oof": len(oof), "secs": round(time.time() - t0),
            "band9": bands["bands"][-1] if bands.get("bands") else {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="画面に並べる複数モデルを学習する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out-dir", default=MODEL_DIR)
    ap.add_argument("--models", default=",".join(M.EXTRA),
                    help=f"学習するモデル（カンマ区切り）。既定: {','.join(M.EXTRA)}"
                         f"。基準モデル({M.BASELINE})は train_production.py が作る")
    ap.add_argument("--features", default="all")
    args = ap.parse_args(argv)

    algos = [a.strip() for a in args.models.split(",") if a.strip()]
    unknown = [a for a in algos if a not in M.ALGOS]
    if unknown:
        raise SystemExit(f"知らないモデル: {unknown}（{','.join(M.ALGOS)} のいずれか）")

    cols = F.columns(args.features)
    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    # 予測用データセット（ラベル未確定を含む）で学習させない。
    # メタファイルの有無に頼らず、中身で判定する
    n_unlabeled = int(ds["label"].isna().sum())
    if n_unlabeled:
        raise SystemExit(
            f"ラベル未確定が {n_unlabeled:,}件あります。予測用データセットでは"
            "学習できません（--keep-unlabeled を付けずに build_dataset.py を回す）")

    # 実収益（ラベル非依存の物差し）。スコア帯統計に使う
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ds = ds.merge(S.reference_outcome(S.Panels(bars).get(B.HIGH_WINDOW)),
                  on=["Code", "Date"], how="left")
    del bars

    tuned = set(TM.load())
    missing = [a for a in algos if a not in tuned]
    print(f"[load] {len(ds):,}件 / 正例率 {ds['label'].mean()*100:.2f}% "
          f"/ {ds['Date'].min().date()} 〜 {ds['Date'].max().date()}")
    print(f"[setup] 特徴量 {len(cols)}列 / 学習するモデル {algos}")
    if missing:
        print(f"[warn] 探索済みパラメータが無いモデル: {missing}")
        print("       既定値で学習します（research/exp/e15_tune_all.py で探索できます）")
    print()

    rows = []
    for algo in algos:
        print(f"[{algo}] out-of-fold を作って学習")
        try:
            r = train_one(algo, ds, cols, args.out_dir)
        except Exception as exc:          # noqa: BLE001
            # 1モデルの失敗で他を落とさない。画面はあるモデルだけ並べる
            print(f"[{algo}] 失敗: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        b = r["band9"]
        print(f"[{algo}] {r['secs']}秒 / {r['size_mb']}MB / "
              f"out-of-fold {r['n_oof']:,}件 / 最上位帯の正例率 "
              f"{(b.get('positive_rate') or 0)*100:.1f}% / "
              f"実収益 {b.get('end_median')}%")
        print()

    print("=" * 72)
    print(f"{'モデル':<10}{'秒':>6}{'MB':>7}{'OOF':>8}  保存先")
    for r in rows:
        print(f"{r['algo']:<10}{r['secs']:>6}{r['size_mb']:>7}{r['n_oof']:>8,}  "
              f"{os.path.relpath(r['dir'])}")
    ok = [r["algo"] for r in rows]
    print()
    print(f"[done] {len(ok)}/{len(algos)} モデル: {ok}")
    if len(ok) < len(algos):
        print(f"[warn] 失敗: {[a for a in algos if a not in ok]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
