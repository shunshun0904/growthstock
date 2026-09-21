#!/usr/bin/env python3
"""
実験26: 実験24 の腕 B（開示からの日数）を、規則に依存しない形で確認する。

docs/MODEL_ADOPTION_RULES.md §4 で先に決めた手順:
  - 種を 5 つ（42 / 7 / 123 / 2024 / 31337）
  - 元の窓と、学習開始を 3か月遅らせて切り直した窓の両方
  - 両方で §3 の 1〜4（全窓で正・最悪 ≥ 0・z > −1・PR-AUC の改善が
    種のレンジ超え）を満たしたら採用

腕 A の元の窓・5種は実験25 の保存済みを使う。
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
import tuning  # noqa: E402
import walkforward as WF  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e21_annual_ab import PARAMS, PRESET  # noqa: E402
from e24_timing_ab import TIMING, with_timing  # noqa: E402
from e25_auc_noise import SEEDS5, OUTCOMES, average, metrics  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
SHIFT_MONTHS = 3


def folds_for(dates: pd.Series, shift_months: int):
    d = pd.to_datetime(dates)
    if shift_months:
        d = d[d >= d.min() + pd.DateOffset(months=shift_months)]
    return WF.make_folds(d, min_train_months=OOF_MIN_TRAIN_MONTHS,
                         test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                         embargo_days=B.RISE_HORIZON)


def oof_single(df: pd.DataFrame, cols: list, params: dict, seed: int, folds) -> pd.DataFrame:
    import lightgbm as lgb

    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "label", "ref_end"] + list(OUTCOMES)
    parts = []
    for f in folds:
        tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
        te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                & df["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        ytr = tr["label"].to_numpy(dtype=int)
        gbm = lgb.LGBMClassifier(**{**params, "random_state": seed},
                                 scale_pos_weight=tuning.scale_pos_weight(ytr))
        gbm.fit(tr[cols].to_numpy(dtype=float), ytr)
        part = te[[c for c in keep if c in te.columns]].copy()
        part["score"] = gbm.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
        part["fold"] = f.index
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def arm(df, cols, params, tag, folds, reuse_prefix=None):
    oofs = []
    for s in SEEDS5:
        p = os.path.join(OOF_DIR, f"e26_{tag}_s{s}.parquet")
        if reuse_prefix and os.path.exists(os.path.join(OOF_DIR, f"{reuse_prefix}_s{s}.parquet")):
            p = os.path.join(OOF_DIR, f"{reuse_prefix}_s{s}.parquet")
        if os.path.exists(p):
            oofs.append(pd.read_parquet(p))
            continue
        t0 = time.time()
        o = oof_single(df, cols, params, s, folds)
        o.to_parquet(p, index=False)
        oofs.append(o)
        print(f"    {tag} 種 {s}: {len(o):,}件 / {time.time()-t0:.0f}秒", flush=True)
    return average(oofs)


def judge(ma: dict, mb: dict, ea: dict, eb: dict, auc_range: float) -> dict:
    """§3 の 1〜4。ea/eb は edge() の結果（物差しごと）。"""
    res = {}
    for oc in OUTCOMES:
        a, b = ea[oc], eb[oc]
        diff = b["thr_fold_mean"] - a["thr_fold_mean"]
        se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
        res[oc] = {"all_won": b["thr_folds_won"] == b["thr_folds"],
                   "worst_ge0": b["thr_worst"] >= 0,
                   "z": diff / se if se > 0 else float("nan"),
                   "mean_ok": (diff / se if se > 0 else 0) > -1,
                   "diff_pt": diff, "won": f"{b['thr_folds_won']}/{b['thr_folds']}",
                   "worst": b["thr_worst"]}
    res["pr_auc_gain"] = mb["pr_auc"] - ma["pr_auc"]
    res["pr_auc_ok"] = res["pr_auc_gain"] > auc_range
    res["pass"] = all(res[oc]["all_won"] and res[oc]["worst_ge0"] and res[oc]["mean_ok"]
                      for oc in OUTCOMES) and res["pr_auc_ok"]
    return res


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    base = [c for c in F.columns(PRESET) if c in frame.columns]
    with open(PARAMS, encoding="utf-8") as fh:
        params = {k: v for k, v in json.load(fh)[PRESET].items() if not k.startswith("_")}
    df = with_timing(frame)
    with open(os.path.join(OOF_DIR, "e25_summary.json"), encoding="utf-8") as fh:
        noise = json.load(fh)
    auc_range = float(noise["triple"]["pr_auc"]["range"])
    print(f"母集団 {len(df):,}件 / 種 {SEEDS5} / PR-AUC の種レンジ（3種平均）{auc_range:.4f}")

    verdicts = {}
    for name, shift in (("元の窓", 0), (f"{SHIFT_MONTHS}か月ずらした窓", SHIFT_MONTHS)):
        folds = folds_for(df["Date"], shift)
        print(f"\n=== {name}: {len(folds)}窓 "
              f"{folds[0].test_start}〜{folds[-1].test_end} ===")
        tag = "orig" if shift == 0 else f"shift{shift}"
        oa = arm(df, base, params, f"A_{tag}", folds, reuse_prefix="e25_A" if shift == 0 else None)
        ob = arm(df, base + TIMING, params, f"B_{tag}", folds)
        ma, mb = metrics(oa), metrics(ob)
        ea = {oc: edge(oa, oc) for oc in OUTCOMES}
        eb = {oc: edge(ob, oc) for oc in OUTCOMES}
        print(f"  {'腕':<24}{'PR-AUC':>8}{'ROC':>8}{'日内':>8}"
              f"{'窓平均20':>10}{'勝ち':>7}{'最悪':>9}{'窓平均40':>10}{'勝ち':>7}{'最悪':>9}")
        for label, m, e in (("A 本番", ma, ea), ("B +開示からの日数", mb, eb)):
            print(f"  {label:<24}{m['pr_auc']:>8.4f}{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{e['ret_o1_20']['thr_fold_mean']:>+9.2f}pt"
                  f"{e['ret_o1_20']['thr_folds_won']:>4}/{e['ret_o1_20']['thr_folds']:<2}"
                  f"{e['ret_o1_20']['thr_worst']:>+8.2f}pt"
                  f"{e['ret_o1_40']['thr_fold_mean']:>+9.2f}pt"
                  f"{e['ret_o1_40']['thr_folds_won']:>4}/{e['ret_o1_40']['thr_folds']:<2}"
                  f"{e['ret_o1_40']['thr_worst']:>+8.2f}pt")
        v = judge(ma, mb, ea, eb, auc_range)
        verdicts[name] = v
        print("  --- 第2の規則の判定 ---")
        for oc in OUTCOMES:
            r = v[oc]
            print(f"  {oc}: 全窓で正 {'○' if r['all_won'] else '×'}({r['won']}) / "
                  f"最悪≥0 {'○' if r['worst_ge0'] else '×'}({r['worst']:+.2f}pt) / "
                  f"z {r['z']:+.2f} {'○' if r['mean_ok'] else '×'} / 差 {r['diff_pt']:+.2f}pt")
        print(f"  PR-AUC の改善 {v['pr_auc_gain']:+.4f} vs 種レンジ {auc_range:.4f} "
              f"{'○' if v['pr_auc_ok'] else '×'}")
        print(f"  → {'満たす' if v['pass'] else '満たさない'}")

    ok = all(v["pass"] for v in verdicts.values())
    print(f"\n=== 総合: {'採用（両方の窓で満たした）' if ok else '不採用（どちらかで満たさない）'} ===")
    with open(os.path.join(OOF_DIR, "e26_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"verdicts": verdicts, "adopt": ok, "auc_range": auc_range},
                  fh, ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
