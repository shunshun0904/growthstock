#!/usr/bin/env python3
"""
実験27b: LightGBM の「探索し直し」の引きは、151列でも153列でも同じくらい散るか。

実験27 で、153列で探索し直した B2 は OOF の PR-AUC こそ B1 と同じ（+0.017）
だったが、上位10% の窓平均が +1.12 → +0.85pt、最悪の窓が −2.41 → −5.43pt と
悪化した。特徴量のせいか、探索の引き（docs/MODEL_TUNING_NOISE.md: 引きだけで
0.6〜1.5pt 動く）のせいかを分けるため、**151列と153列の両方**で Optuna の
種を変えて引き直し、引きごとの分布を並べる。

  A: 151列 / 探索の種 0（本番）, 1, 2
  B: 153列 / 探索の種 0（実験27 の B2）, 1, 2

out-of-fold は種3つの確率平均（実験27 と同じ）。
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
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
from e24_timing_ab import with_timing  # noqa: E402
from e25_auc_noise import metrics  # noqa: E402
from e27_timing_multi import (  # noqa: E402
    N_SPLITS, N_TRIALS, OOF_DIR, TIMING, log, oof_arm, prod_params)

DRAW_SEEDS = (1, 2)


def tune_draw(arm: str, seed: int, sub: pd.DataFrame, cols: list) -> dict:
    path = os.path.join(OOF_DIR, f"e27b_params_{arm}_t{seed}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = time.time()
    params = tuning.tune(sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme="year_cap_date",
                         seed=seed, verbose=False)
    rec = {"params": dict(params), "_cv": dict(tuning.LAST_CV)}
    rec["_cv"]["seconds"] = round(time.time() - t0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{arm} 種{seed}] 探索 {(time.time()-t0)/60:.1f}分 / CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


def main() -> int:
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    df = with_timing(frame).rename(columns={"jq_days_since_disc": "days_since_disc",
                                            "jq_days_since_fy": "days_since_fy"})
    base = [c for c in F.columns("all") if c in df.columns and c not in TIMING]
    full = base + TIMING
    d = df["Date"]
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    log(f"LightGBM の引き直し: 151列 / 153列 × 探索の種 {(0,) + DRAW_SEEDS}")

    rows = []
    draws = {"A": {}, "B": {}}
    # 種0 = 本番（A）と実験27 の B2（B）
    draws["A"][0] = prod_params("lgbm")
    with open(os.path.join(OOF_DIR, "e27_params_lgbm.json"), encoding="utf-8") as fh:
        draws["B"][0] = json.load(fh)
    for s in DRAW_SEEDS:
        draws["A"][s] = tune_draw("A", s, sub, base)
        draws["B"][s] = tune_draw("B", s, sub, full)
    for arm, cols in (("A", base), ("B", full)):
        for s, rec in draws[arm].items():
            tag = {("A", 0): "A", ("B", 0): "B2"}.get((arm, s), f"{arm}_t{s}")
            prefix = "lgbm"
            o = oof_arm(prefix, tag if s == 0 else f"draw{arm}{s}", df, cols, rec["params"])
            m = metrics(o)
            rows.append({"arm": arm, "draw": s, "cv": rec["_cv"].get("mean_pr_auc"), **m})
    res = pd.DataFrame(rows)
    print(f"\n  {'腕':<4}{'引き':>4}{'CV PR-AUC':>10}{'OOF PR-AUC':>11}{'ROC':>8}{'窓平均20':>10}{'勝ち':>6}{'最悪':>9}{'窓平均40':>10}{'最悪':>9}")
    for _, r in res.iterrows():
        print(f"  {r['arm']:<4}{int(r['draw']):>4}{r['cv']:>10.4f}{r['pr_auc']:>11.4f}{r['roc_auc']:>8.4f}"
              f"{r['ret_o1_20_mean']:>+9.2f}pt{int(r['ret_o1_20_won']):>3}/{int(r['ret_o1_20_n']):<2}"
              f"{r['ret_o1_20_worst']:>+8.2f}pt{r['ret_o1_40_mean']:>+9.2f}pt{r['ret_o1_40_worst']:>+8.2f}pt")
    print("\n  引きの分布（3引き）:")
    for arm in ("A", "B"):
        g = res[res["arm"] == arm]
        print(f"  {arm}: OOF PR-AUC {g['pr_auc'].mean():.4f} ({g['pr_auc'].min():.4f}〜{g['pr_auc'].max():.4f}) / "
              f"窓平均20 {g['ret_o1_20_mean'].mean():+.2f}pt ({g['ret_o1_20_mean'].min():+.2f}〜{g['ret_o1_20_mean'].max():+.2f}) / "
              f"最悪 {g['ret_o1_20_worst'].mean():+.2f}pt ({g['ret_o1_20_worst'].min():+.2f}〜{g['ret_o1_20_worst'].max():+.2f})")
    res.to_csv(os.path.join(OOF_DIR, "e27b_draws.csv"), index=False)
    log(f"記録: {OOF_DIR}/e27b_draws.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
