#!/usr/bin/env python3
"""
実験21: 年単位の業績の軌道を特徴量に足すと、モデルの実収益は上がるか（A/B）。

実験20 は特徴量1本ずつの検出力（上位10%の実収益の超過）を見た。
ここでは本番の特徴量（`all`、151列）に足して、同じ行・同じパラメータで
out-of-fold の実収益がどれだけ動くかを測る。判定は実験19と同じ
z = 差 / √(SE_a² + SE_b²) > 2（ノイズ床 0.143pt、実験11）。

腕
--
  A  本番の特徴量（`all`）
  B  A + 変化率だけ（前年比 yoy1/yoy2/yoy3、2年・3年の変化）        20列
  C  A + 変化率 + 軌道（加速・過去3期の山からの回復・落ち込み・V字・
     4年ぶり高値）+ 営業利益率の変化                                42列

B は依頼の文言どおり「直近だけでなく2年前・3年前と前年比」。
C はそれに実験20 で見た軌道の形を足したもの。

公平にするための細工
------------------
- パラメータは本番の `research/lgbm_params.json`（`all`）を全腕で共用する。
  探索を引き直すと、その引きの差（0.6〜1.5pt）が特徴量の差を覆い隠す
  （docs/MODEL_TUNING_NOISE.md）
- 行は同じ（特徴量を足すだけ。無い行は NaN で LightGBM に任せる）
- 種3つ（42 / 7 / 123）の確率平均。窓は本番の out-of-fold と同じ
- 年次の値は実験20 と同じ作り方（通期の最初の開示、開示日で時点整合、
  年度末の間隔が揃うときだけ前期とみなす）

本番の設定には書かない。結果は research/_data/oof/e21_*.parquet。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e19_freshdata import oof_for, SEEDS  # noqa: E402
import e20_annual_trajectory as E20  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

OUTCOMES = ("ret_o1_20", "ret_o1_40")
PRESET = "all"
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "lgbm_params.json")

CHANGE = [f"{c}_{k}" for c in E20.ITEMS for k in ("yoy1", "yoy2", "yoy3", "2y", "3y")]
SHAPE = [f"{c}_{k}" for c in E20.ITEMS
         for k in ("accel", "recovery3", "dip3", "vshape3", "4y_high")] + \
        ["opm_chg1", "opm_chg2"]


def with_annual(frame: pd.DataFrame) -> pd.DataFrame:
    panel = E20.annual_panel(lab.DATA_DIR)
    frame = frame.sort_values("Date").reset_index(drop=True)
    panel = panel.sort_values("DiscDate")
    m = pd.merge_asof(frame, panel, left_on="Date", right_on="DiscDate", by="Code",
                      direction="backward", allow_exact_matches=True)
    stale = (m["Date"] - m["DiscDate"]).dt.days > E20.STALE_DAYS
    lagcols = [c for c in m.columns
               if c in E20.ITEMS or c.endswith(tuple(f"_y{k}" for k in E20.LAGS))]
    m.loc[stale | m["DiscDate"].isna(), lagcols] = np.nan
    for k, v in E20.features(m).items():
        m[k] = v
    return m


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    base = [c for c in F.columns(PRESET) if c in frame.columns]
    with open(PARAMS, encoding="utf-8") as fh:
        params = {k: v for k, v in json.load(fh)[PRESET].items() if not k.startswith("_")}
    print(f"母集団 {len(frame):,}件 / 正例率 {frame['label'].mean()*100:.2f}%")
    print(f"本番の特徴量 {PRESET}（{len(base)}列）/ 本番のパラメータ（木{params['n_estimators']}本 "
          f"lr {params['learning_rate']:.4f} 葉{params['num_leaves']}）を全腕で共用")
    print(f"種 {SEEDS} の確率平均 / 物差し {OUTCOMES}\n")

    df = with_annual(frame)
    for c in CHANGE + SHAPE:
        if c not in df.columns:
            raise SystemExit(f"特徴量が無い: {c}")
    print("足す列の充足率:")
    for name, cols in (("変化率", CHANGE), ("軌道", SHAPE)):
        cov = df[cols].notna().mean()
        print(f"  {name} {len(cols)}列: 平均 {cov.mean()*100:.1f}% "
              f"（最小 {cov.min()*100:.1f}% {cov.idxmin()}）")

    arms = [
        ("A 本番の特徴量", base),
        ("B A + 変化率(20)", base + CHANGE),
        ("C A + 変化率 + 軌道(42)", base + CHANGE + SHAPE),
    ]
    runs = {}
    print("\n=== out-of-fold を作る ===")
    for i, (name, cols) in enumerate(arms):
        p = os.path.join(OOF_DIR, f"e21_{i}.parquet")
        if os.path.exists(p):
            runs[name] = pd.read_parquet(p)
            print(f"  {name}: 保存済みを読む ({len(runs[name]):,}件)")
            continue
        t0 = time.time()
        o = oof_for(df, cols, params)
        o.to_parquet(p, index=False)
        runs[name] = o
        print(f"  {name}: {len(o):,}件 / {len(cols)}列 / {time.time()-t0:.0f}秒")

    print("\n=== 分離力（out-of-fold）===")
    print(f"  {'条件':<26}{'件数':>8}{'正例率':>9}{'PR-AUC':>9}"
          f"{'PR/正例率':>11}{'ROC-AUC':>10}{'日内AUC':>10}")
    for name, o in runs.items():
        y = o["label"].to_numpy(dtype=int)
        s = o["score"].to_numpy(dtype=float)
        br = y.mean()
        pr = average_precision_score(y, s)
        print(f"  {name:<26}{len(o):>8,}{br*100:>8.2f}%{pr:>9.4f}"
              f"{pr/br:>10.2f}x{roc_auc_score(y, s):>10.4f}"
              f"{lab.auc_in_day(o):>10.4f}")

    names = [n for n, _ in arms]
    summary = {}
    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome}（上位10%）===")
        print(f"  {'条件':<26}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
        ms = {}
        for name, o in runs.items():
            m = edge(o, outcome)
            ms[name] = m
            print(f"  {name:<26}{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt{m['thr_n']:>9,}")
        print("  --- 差の検定（足切り z>2）---")
        for a, b in ((0, 1), (0, 2), (1, 2)):
            ma, mb = ms[names[a]], ms[names[b]]
            diff = mb["thr_fold_mean"] - ma["thr_fold_mean"]
            se = float(np.sqrt(ma["se"] ** 2 + mb["se"] ** 2))
            z = diff / se if se > 0 else float("nan")
            print(f"  {names[a][:1]} -> {names[b][:1]}: {diff:+.2f}pt / z {z:+.2f}"
                  f"{'  ← 足切りを越えた' if abs(z) > 2 else ''}")
            summary[f"{outcome}:{names[a][:1]}->{names[b][:1]}"] = {
                "diff_pt": round(diff, 3), "z": round(z, 3)}
    with open(os.path.join(OOF_DIR, "e21_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\nノイズ床（実験11）0.143pt / 記録 {OOF_DIR}/e21_*.parquet, e21_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
