#!/usr/bin/env python3
"""
実験24: 「直近の決算開示からの日数」を本番の特徴量に足すと実収益は動くか（A/B）。

実験23 で、J-Quants から作れる代理特徴量のうち `jq_days_since_disc`
（直近の決算開示から Date までの日数）が、下位10%（開示直後のブレイク）で
+1.45pt（z +3.35、10/11窓）、上位10%（開示から遠い = 次の開示の直前）で
+0.64pt（z +2.54）だった。窓 AUC 0.553 は単体の特徴量として一番高い。
本番の特徴量に「開示からの日数」は入っていない。

腕
--
  A  本番の `all`（151列）
  B  A + 開示からの日数 2本（直近の開示 / 直近の通期開示）
  C  B + 年次の変化率 20列（実験21 の腕 B）      —— 足し合わせで積み上がるか

細工は実験21 と同じ（本番のパラメータを共用、同じ行、種3つ、判定 z>2）。
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e19_freshdata import oof_for, SEEDS  # noqa: E402
from e21_annual_ab import CHANGE, PARAMS, PRESET, with_annual  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

OUTCOMES = ("ret_o1_20", "ret_o1_40")
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TIMING = ["jq_days_since_disc", "jq_days_since_fy"]


def with_timing(frame: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "fins_*.parquet")))
    q = pd.concat([pd.read_parquet(p, columns=["Code", "DiscDate", "CurPerType", "Sales", "NP"])
                   for p in paths], ignore_index=True)
    q = q[q[["Sales", "NP"]].notna().any(axis=1)].copy()
    q["DiscDate"] = pd.to_datetime(q["DiscDate"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    any_ = q[["Code", "DiscDate"]].sort_values("DiscDate").rename(columns={"DiscDate": "AnyDisc"})
    fy = q[q["CurPerType"] == "FY"][["Code", "DiscDate"]].sort_values("DiscDate") \
        .rename(columns={"DiscDate": "FyDisc"})
    m = pd.merge_asof(frame, any_, left_on="Date", right_on="AnyDisc", by="Code",
                      direction="backward", allow_exact_matches=True)
    m = pd.merge_asof(m, fy, left_on="Date", right_on="FyDisc", by="Code",
                      direction="backward", allow_exact_matches=True)
    m["jq_days_since_disc"] = (m["Date"] - m["AnyDisc"]).dt.days.astype(float).clip(upper=400)
    m["jq_days_since_fy"] = (m["Date"] - m["FyDisc"]).dt.days.astype(float).clip(upper=800)
    return m.drop(columns=["AnyDisc", "FyDisc"])


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    base = [c for c in F.columns(PRESET) if c in frame.columns]
    with open(PARAMS, encoding="utf-8") as fh:
        params = {k: v for k, v in json.load(fh)[PRESET].items() if not k.startswith("_")}
    df = with_annual(with_timing(frame))
    print(f"母集団 {len(df):,}件 / 本番の特徴量 {len(base)}列 / 種 {SEEDS} / 物差し {OUTCOMES}")
    print("開示からの日数の充足:", {c: f"{df[c].notna().mean()*100:.1f}%" for c in TIMING})

    arms = [
        ("A 本番の特徴量", base),
        ("B A + 開示からの日数(2)", base + TIMING),
        ("C B + 年次の変化率(20)", base + TIMING + CHANGE),
    ]
    runs = {}
    print("\n=== out-of-fold を作る ===")
    for i, (name, cols) in enumerate(arms):
        p = os.path.join(OOF_DIR, f"e24_{i}.parquet")
        if i == 0 and os.path.exists(os.path.join(OOF_DIR, "e21_0.parquet")):
            p = os.path.join(OOF_DIR, "e21_0.parquet")       # 腕 A は実験21 と同じもの
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
    for name, o in runs.items():
        y = o["label"].to_numpy(dtype=int)
        s = o["score"].to_numpy(dtype=float)
        pr = average_precision_score(y, s)
        print(f"  {name:<28} PR-AUC {pr:.4f} ({pr/y.mean():.2f}x)  ROC {roc_auc_score(y, s):.4f}"
              f"  日内AUC {lab.auc_in_day(o):.4f}")

    names = [n for n, _ in arms]
    summary = {}
    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome}（上位10%）===")
        ms = {}
        for name, o in runs.items():
            m = edge(o, outcome)
            ms[name] = m
            print(f"  {name:<28}{m['thr_fold_mean']:>+8.2f}pt  SE {m['se']:.2f}  "
                  f"勝ち {m['thr_folds_won']}/{m['thr_folds']}  最悪 {m['thr_worst']:+.2f}pt")
        for a, b in ((0, 1), (1, 2), (0, 2)):
            ma, mb = ms[names[a]], ms[names[b]]
            diff = mb["thr_fold_mean"] - ma["thr_fold_mean"]
            se = float(np.sqrt(ma["se"] ** 2 + mb["se"] ** 2))
            z = diff / se if se > 0 else float("nan")
            print(f"  {names[a][:1]} -> {names[b][:1]}: {diff:+.2f}pt / z {z:+.2f}"
                  f"{'  ← 足切りを越えた' if abs(z) > 2 else ''}")
            summary[f"{outcome}:{names[a][:1]}->{names[b][:1]}"] = {
                "diff_pt": round(diff, 3), "z": round(z, 3)}
    with open(os.path.join(OOF_DIR, "e24_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\nノイズ床（実験11）0.143pt / 記録 {OOF_DIR}/e24_*.parquet, e24_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
