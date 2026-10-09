#!/usr/bin/env python3
"""
実験70: σ_dfa のラベル（実験69）を、実験55 と同じ作りで確かめる。

運用者の依頼（2026-10-09）「σ_dfa のラベルを実験55 と同じ作りで確かめる。そのラベルで LightGBM を探索し直し、
XGBoost・CatBoost と運用規則の取引まで」。

作りは実験55（research/exp/e55_label_sigma_full.py）そのもので、腕だけを差し替える:
  L0  1.2 × σ20（現行）
  L5  1.2 × σ_dfa（σ_dfa = σ20^w × σ120^(1−w)、w = clip((α − 0.5) / 0.3, 0, 1)、α = nl_dfa_abs120。実験69 と同じ式）
学習（本番の239列）: lgbm B1（本番の木の形）/ lgbm B2（そのラベルで探索し直し: 50試行 × 5分割 year_cap_date、
ホールドアウトより前）/ xgb・cat（本番の木の形）。評価は実験55 と同じ（閾値ルール・発火数そろえ・窓の中の上位10%・
日付内・ボラの帯・運用規則 ops_rule と live_track の取引、3モデル合議の百分位 90 / 95）。3切り方 × 種3つ。

実装は実験55 の関数を流用する。差し替えるのは3点だけ:
  1. E54.sigmas_from_bars に σ_dfa を足す（α は research/_data/nonlinear_features.parquet から）
  2. 腕の表（ARMS / LABELS）を L0 / L5 にする
  3. 保存名を e70_*（out-of-fold と探索結果）にする

  python3 research/exp/e70_label_sigma_dfa_full.py [--trials 50] [--shifts 0,2,4] [--seeds 3] [--consensus 90,95]
  結果は research/_data/oof/e70_*。本番の設定には書かない。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e54_label_sigma as E54  # noqa: E402
import e55_label_sigma_full as E55  # noqa: E402
import e69_label_sigma_dfa as E69  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PREFIX = "e70"
ARMS = {"L0": "sigma20", "L5": "sigma_dfa"}
LABELS = {"L0": "L0 1.2σ20（現行）", "L5": "L5 1.2σ_dfa"}

_orig_sigmas = E54.sigmas_from_bars


def sigmas_with_dfa(bars: pd.DataFrame) -> pd.DataFrame:
    """実験54 の σ の表に σ_dfa を足す（α は nonlinear_features.parquet の nl_dfa_abs120）。"""
    sg = _orig_sigmas(bars)
    if not os.path.exists(NL.OUT_PATH):
        raise SystemExit(f"{NL.OUT_PATH} がありません。python3 research/nonlinear_features.py で作ってください")
    nl = pd.read_parquet(NL.OUT_PATH, columns=["Code", "Date", E69.ALPHA_COL])
    nl["Code"] = nl["Code"].astype(str)
    nl["Date"] = pd.to_datetime(nl["Date"])
    sg = sg.merge(nl, on=["Code", "Date"], how="left")
    sg["sigma_dfa"] = E69.sigma_dfa(sg["sigma20"], sg["sigma120"], sg[E69.ALPHA_COL]).to_numpy()
    return sg.drop(columns=[E69.ALPHA_COL])


def oof70(df, cols, arm, fit, algo, shift, seeds, params):
    """E55.oof と同じ。保存名だけ e70_。"""
    folds = E41.folds_for(df["Date"], shift)
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"{PREFIX}_{arm}_{fit}_{algo}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds(algo, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        E55.log(f"  腕{arm} {fit} {algo} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def main(argv=None) -> int:
    E54.sigmas_from_bars = sigmas_with_dfa
    E55.ARMS.clear()
    E55.ARMS.update(ARMS)
    E55.LABELS.clear()
    E55.LABELS.update(LABELS)
    E55.PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
    E55.oof = oof70
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a.startswith("--arms") for a in argv):
        argv += ["--arms", "L0,L5"]
    if not any(a.startswith("--consensus") for a in argv):
        argv += ["--consensus", "90,95"]
    print("=" * 78)
    print(f"実験70 σ_dfa のラベルを実験55 と同じ作りで確かめる（腕 {list(ARMS)}、保存名 {PREFIX}_*）")
    print(f"  σ_dfa = σ20^w × σ120^(1−w)、w = clip((α − {E69.A_LO}) / {E69.A_SPAN}, 0, 1)、α = {E69.ALPHA_COL}")
    print("=" * 78)
    return E55.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
