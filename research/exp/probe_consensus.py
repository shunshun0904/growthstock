#!/usr/bin/env python3
"""
12本の合意の百分位（research/predict_daily.py の consensus_reference / consensus_fields）を、本番のモデルと
当日の候補で日次予測と同じ経路に通して確かめる。本番の出力（public/data・スプレッドシート）には触らない。

運用者の指示（2026-10-12）「ダッシュボードに12本の合意の百分位を表示してください」。

run-experiment.yml は Release（data-raw）の資産を research/_data に落とす。基準モデル（model.txt / oof.parquet）と
追加モデル・ラベル違い（<名前>_model.joblib / <名前>_oof.parquet）を predict.yml と同じ形に並べ、予測用のデータ
セット（ラベル未確定を含む）を作って predict_daily.main を一時フォルダへの出力で走らせる。出てきた predictions.json
の合意の値を検査し、件数・割合・分位だけを出す（銘柄は出さない）。

検査
  - 物差し: 12本・過去の行 16,320件。上位10% の線が実験79 の「合意 86.3 以上」と同じ（同じ定義なので一致するはず）
  - 候補ごと: nCons = 12、cons12 / consPct が 0〜100、cons12 が byModel の12本の百分位の平均と一致（丸めの差 0.1 以内）、
    cons12 の大きい順に consPct も大きい（単調）
  - 分布: consPct が 90 以上・97.5 以上の割合、日ごとの最大

    exp=probe_consensus.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa: E402
import predict_daily as PD  # noqa: E402
import probe_variants as PV  # noqa: E402

DATA_DIR = PD.DATA_DIR
MODEL_DIR = PD.MODEL_DIR
E79_LINE90 = 86.3        # 実験79（run 38183373898）の「合意の上位10%（合意 86.3 以上）」


def main() -> int:
    for f in ("model.txt", "oof.parquet"):
        src = os.path.join(DATA_DIR, f)
        if not os.path.exists(src):
            print(f"[fail] Release に {f} がありません")
            return 1
        shutil.copy(src, os.path.join(MODEL_DIR, f))
    placed = PV.arrange_models()
    print(f"[models] 基準モデル + Release から並べ直した {len(placed)}本: {placed}")
    ds_path = os.path.join(DATA_DIR, "dataset_predict.parquet")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(DATA_DIR), "build_dataset.py"),
                        "--keep-unlabeled", "--out", ds_path], capture_output=True, text=True)
    print("[dataset] " + (r.stdout.strip().splitlines() or ["(出力なし)"])[-1])
    if r.returncode != 0:
        print("[fail] 予測用データセットを作れない")
        print(r.stderr[-2000:])
        return 1
    out = tempfile.mkdtemp()
    rc = PD.main(["--dataset", ds_path, "--model-dir", MODEL_DIR, "--out-dir", out, "--days", "5"])
    if rc:
        print(f"[fail] predict_daily が {rc} で終わった")
        return 1
    with open(os.path.join(out, "predictions.json"), encoding="utf-8") as fh:
        pred = json.load(fh)
    cm = pred.get("consensus") or {}
    cands = pred.get("candidates", [])
    fails = []
    print(f"\n[consensus] モデル {len(cm.get('models', []))}本 / 過去の行 {cm.get('nHist', 0):,}件 / "
          f"上位10% の線 {cm.get('line90')}（実験79 {E79_LINE90}）/ 上位2.5% の線 {cm.get('line975')}")
    if list(cm.get("models", [])) != list(M.LINEUP):
        fails.append(f"合意のモデルが LINEUP と違う: {cm.get('models')}")
    if cm.get("nHist") != 16320:
        fails.append(f"過去の行が 16,320件でない: {cm.get('nHist')}")
    if cm.get("line90") is None or abs(cm["line90"] - E79_LINE90) > 0.15:
        fails.append(f"上位10% の線が実験79 と合わない: {cm.get('line90')}")
    ok = [c for c in cands if c.get("consPct") is not None]
    print(f"[cand] 候補 {len(cands):,}件 / 合意が付いた {len(ok):,}件 / 日数 {len(pred.get('dates', []))}")
    if len(ok) != len(cands):
        fails.append(f"合意の付かない候補 {len(cands) - len(ok)}件")
    bad_n = sum(1 for c in ok if c.get("nCons") != 12)
    bad_rng = sum(1 for c in ok if not (0 <= c["consPct"] <= 100 and 0 <= c["cons12"] <= 100))
    diffs = []
    for c in ok:
        p = [c["byModel"][a]["pctHistorical"] for a in M.LINEUP if a in c.get("byModel", {})]
        if len(p) == 12 and all(v is not None for v in p):
            diffs.append(abs(np.mean(p) - c["cons12"]))
    order = sorted(ok, key=lambda c: c["cons12"])
    mono = all(order[i]["consPct"] <= order[i + 1]["consPct"] for i in range(len(order) - 1))
    print(f"[check] nCons ≠ 12: {bad_n}件 / 範囲外: {bad_rng}件 / byModel の平均との差 最大 "
          f"{max(diffs) if diffs else float('nan'):.3f}（{len(diffs)}件で比べた）/ 単調: {mono}")
    if bad_n:
        fails.append(f"nCons が 12 でない {bad_n}件")
    if bad_rng:
        fails.append(f"範囲外 {bad_rng}件")
    if not diffs or max(diffs) > 0.1:
        fails.append("cons12 が byModel の平均と合わない")
    if not mono:
        fails.append("cons12 と consPct の並びが合わない")
    v = np.array([c["consPct"] for c in ok], dtype=float)
    if len(v):
        print(f"[dist] consPct 中央 {np.median(v):.1f} / 90以上 {(v >= 90).mean() * 100:.1f}%（{int((v >= 90).sum())}件）"
              f" / 97.5以上 {(v >= 97.5).mean() * 100:.1f}%（{int((v >= 97.5).sum())}件）")
        for d in pred.get("dates", []):
            dv = np.array([c["consPct"] for c in ok if c["date"] == d], dtype=float)
            if len(dv):
                print(f"  {d}: 候補 {len(dv)}件 / 合意の百分位の最大 {dv.max():.1f} / 90以上 {int((dv >= 90).sum())}件"
                      f" / 97.5以上 {int((dv >= 97.5).sum())}件")
    if fails:
        for f in fails:
            print(f"[fail] {f}")
        return 1
    print("[ok] 12本の合意の百分位は日次予測と同じ経路で付き、実験79 の定義と一致する")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
