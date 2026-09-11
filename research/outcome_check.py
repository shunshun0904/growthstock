#!/usr/bin/env python3
"""
本番モデルが「実際にいくら取れたか」を、ラベルに依存しない物差しで測る。

PR-AUC も ROC-AUC もラベルの関数なので、ラベル定義を変えた前後を
これらの数字だけで比べても「良くなった」ことにはならない。
固定の参照ホライズン（60営業日後の5日平均終値の上昇率）で、
上位k%が実際に何%取れたかを出す。この定義はラベルを変えても動かない。

同時に、モデルを使わず1列で並べただけの規則（大型順・低ボラ順など）と
**同じ行集合の上で対で**比べる。別々の区間の重なりでは判定できないため。

  python3 research/outcome_check.py --features all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import sweep_design as S  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402

DATA_DIR = S.DATA_DIR


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, cols: List[str],
                  preset: str) -> Dict[str, np.ndarray]:
    """train_model と同じ作り方で、同じモデルを当てる。"""
    models = T.fit_models(train, cols, verbose=True, preset=preset)
    X = test[cols].to_numpy(dtype=float)
    return {name: m.predict_proba(X)[:, 1] for name, m in models.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本番モデルの実収益を測る")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--features", default="all")
    ap.add_argument("--k-pct", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "outcome_check.json"))
    args = ap.parse_args(argv)

    cols = F.columns(args.features)
    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    print(f"[load] {len(ds):,}件 / 正例率 {ds['label'].mean()*100:.2f}% "
          f"/ 特徴量 {args.features}（{len(cols)}列）")

    # 参照ホライズンの実リターンをパネルから作る。
    # データセットの列（future_rise / end_level）はラベル定義と一緒に動くので使わない
    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel = S.Panels(bars).get(B.HIGH_WINDOW)
    print(f"[ref] 参照ホライズン {S.REF_HORIZON}営業日（ラベル定義に依存しない物差し）")
    ref = S.reference_outcome(panel)
    ds = ds.merge(ref, on=["Code", "Date"], how="left")

    parts = T.holdout_split(ds)
    train, test = parts["train"], parts["test"]
    scores = fit_and_score(train, test, cols, args.features)

    k = max(1, int(len(test) * args.k_pct / 100))
    base_all = S.outcome_stats(test)
    rows = []
    print(f"\n[実収益] 上位{args.k_pct:.0f}%（{k:,}件）が参照ホライズンで実際にどうなったか")
    print(f"    {'モデル':<20}{'上位の中央値':>12}{'勝率':>8}{'差':>9}{'95%区間':>18}"
          f"{'上位の日次ボラ':>14}")
    print(f"    {'（全件）':<20}{base_all['end_median']:>+11.2f}%"
          f"{base_all['win_rate']*100:>7.1f}%{'—':>9}{'—':>18}"
          f"{test['vol_20d'].median():>13.2f}%")
    for name, sc in scores.items():
        t = test.copy()
        t["score"] = sc
        top = t.nlargest(k, "score")
        b = S.outcome_stats(top)
        lo, hi = S.edge_ci(t, k_pct=args.k_pct)
        row = {
            "model": name, "preset": args.features, "k_pct": args.k_pct, "n_top": k,
            "top_end_median": b["end_median"], "all_end_median": base_all["end_median"],
            "top_win_rate": b["win_rate"], "all_win_rate": base_all["win_rate"],
            "top_rise_median": b["rise_median"],
            "edge": round(b["end_median"] - base_all["end_median"], 2),
            "edge_ci": [lo, hi], "significant": bool(lo > 0),
            "top_vol_20d": round(float(top["vol_20d"].median()), 3),
            "all_vol_20d": round(float(t["vol_20d"].median()), 3),
            "top_log_market_cap": round(float(top["log_market_cap"].median()), 3),
            "all_log_market_cap": round(float(t["log_market_cap"].median()), 3),
        }
        # 1列で並べただけの規則との対比較
        row["vs_rules"] = {}
        for nm, col, desc in S.NAIVE_RULES:
            if col not in t.columns:
                continue
            med, lo2, hi2, pp = S.paired_vs_rule(t, col, desc, k_pct=args.k_pct)
            row["vs_rules"][nm] = {"diff": med, "ci": [lo2, hi2], "p_positive": pp,
                                   "significant": bool(lo2 > 0)}
        rows.append(row)
        print(f"    {name:<20}{b['end_median']:>+11.2f}%{b['win_rate']*100:>7.1f}%"
              f"{row['edge']:>+9.2f}{f'[{lo:+.2f},{hi:+.2f}]':>18}"
              f"{row['top_vol_20d']:>13.2f}%")

    print("\n[対] 1列で並べただけの規則との差（同じ行集合の上で対で比較）")
    for row in rows:
        print(f"  {row['model']}")
        for nm, v in row["vs_rules"].items():
            mark = ("有意に上" if v["significant"]
                    else "有意に下" if v["ci"][1] < 0 else "")
            print(f"    対 {nm:<26}{v['diff']:+7.2f}pt "
                  f"[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] "
                  f"上回った割合 {v['p_positive']*100:5.1f}% {mark}")

    payload = {"label": B.DEFAULT_RISE.name, "preset": args.features,
               "k_pct": args.k_pct, "reference_horizon": S.REF_HORIZON,
               "test_from": str(test["Date"].min().date()),
               "test_to": str(test["Date"].max().date()),
               "n_test": int(len(test)), "results": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
