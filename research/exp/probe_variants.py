#!/usr/bin/env python3
"""
ラベル違いのモデル（lgbm_k15 / lgbm_k18。models.VARIANTS）が日次予測と同じ経路で読めて採点できるかの通し確認。
本番の出力（public/data・スプレッドシート）には触らない。

run-experiment.yml は Release（data-raw）の資産をまるごと research/_data に落とすので、そこにある
<name>_model.joblib / <name>_oof.parquet を predict.yml と同じ形（research/model/models/<name>/）に並べ直し、
meta.json は git 側のものを使う。データセット（学習用。build_dataset が作る）の直近5営業日の行を
predict_daily.score_others で採点し、byModel の一致（agreement）まで数える。出すのは件数・割合だけ。

    exp=probe_variants.py
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export_sheets as ES  # noqa: E402
import features as F  # noqa: E402
import models as M  # noqa: E402
import predict_daily as PD  # noqa: E402

DATA_DIR = PD.DATA_DIR
MODEL_DIR = PD.MODEL_DIR


def arrange_models() -> list:
    """Release から落ちた <name>_model.joblib / <name>_oof.parquet を models/<name>/ に並べ直す（predict.yml と同じ）。"""
    placed = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*_model.joblib"))):
        a = os.path.basename(f)[: -len("_model.joblib")]
        d = M.sub_dir(a, MODEL_DIR)
        os.makedirs(d, exist_ok=True)
        shutil.copy(f, os.path.join(d, "model.joblib"))
        o = os.path.join(DATA_DIR, f"{a}_oof.parquet")
        if os.path.exists(o):
            shutil.copy(o, os.path.join(d, "oof.parquet"))
        placed.append(a)
    return placed


def main() -> int:
    placed = arrange_models()
    print(f"[models] Release から並べ直した: {placed}")
    avail = M.available(MODEL_DIR)
    print(f"[models] 読めるモデル（LINEUP の順）: {avail}")
    for a in M.VARIANTS:
        if a not in avail:
            print(f"[fail] {a} が読めない（model.joblib か meta.json が無い）")
            return 1
    ds = pd.read_parquet(os.path.join(DATA_DIR, "dataset.parquet"))
    ds["Date"] = pd.to_datetime(ds["Date"])
    cols = F.columns(F.DEFAULT_PRESET)
    days = sorted(ds["Date"].unique())[-5:]
    cand = ds[ds["Date"].isin(days)].copy()
    print(f"[cand] 直近{len(days)}営業日 {len(cand):,}行（{pd.Timestamp(days[0]).date()} 〜 {pd.Timestamp(days[-1]).date()}）")
    scores, info = PD.score_others(cand, cols, MODEL_DIR)
    print(f"[score] 採点できたモデル: {sorted(scores)}")
    for m in info:
        if m.get("variant"):
            print(f"  {m['algo']}: label k={m.get('labelK')} / 正例率 {m.get('positiveRate')} / OOF {m.get('nOof')} / "
                  f"探索 PR-AUC {(m.get('cv') or {}).get('mean_pr_auc')}")
    hist = {a: M.hist_scores(a, MODEL_DIR) for a in scores}
    per_rows = []
    for i in range(len(cand)):
        per = {a: {"score": float(sc[i]), "pctHistorical": M.pct_historical(float(sc[i]), hist[a])}
               for a, sc in scores.items()}
        per_rows.append(per)
    for a in M.VARIANTS:
        if a not in scores:
            print(f"[fail] {a} の採点が無い")
            return 1
        p = np.array([r[a]["pctHistorical"] for r in per_rows], dtype=float)
        print(f"  {a}: 百分位 平均 {np.nanmean(p):.1f} / 90以上 {(p >= 90).mean()*100:.1f}% / 95以上 {(p >= 95).mean()*100:.1f}% "
              f"/ 欠測 {np.isnan(p).sum()}")
    ag = [PD.agreement(r, M.VARIANTS) for r in per_rows]
    ks = pd.Series([f"{a}/{n}" for a, n in ag]).value_counts()
    print("  ラベル違いの一致（agreeK90/nK）の内訳: " + "  ".join(f"{k} {v}件" for k, v in ks.items()))
    # 本番の4モデルとラベル違いの相関（同じ行で）
    if "lgbm_k15" in scores and "xgb" in scores:
        from scipy.stats import spearmanr
        for a in M.VARIANTS:
            for b in ("xgb", "cat", "logit"):
                if b in scores:
                    print(f"  順位相関 {a} 対 {b}: {spearmanr(scores[a], scores[b]).correlation:.3f}")
    # スプレッドシートの列（通信しない）
    pred = {"models": [{"algo": m["algo"], "short": m.get("short")} for m in info],
            "candidates": [{"date": "2026-10-09", "code": "0000", "jqCode": "00000", "name": "", "rankInDay": 1,
                            "nInDay": 1, "score": 0.0, "band": 0, "byModel": per_rows[0],
                            "agree90": 0, "nModels": 0, "agreeK90": ag[0][0], "nK": ag[0][1]}]}
    row = ES.rows_from_predictions(pred)[0]
    have = [c for c in ES.MODEL_COLS + ES.SCORE_COLS + ES.AGREE_COLS if c in row]
    print(f"[sheet] 行に入る列: {have}")
    missing = [c for c in ("L15%", "L18%", "L15スコア", "L18スコア", ES.AGREE_K_COL) if c not in row]
    if missing:
        print(f"[fail] シートの列が足りない: {missing}")
        return 1
    print("[ok] ラベル違いのモデルは日次予測と同じ経路で読めて採点でき、シートの列も作れる")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
