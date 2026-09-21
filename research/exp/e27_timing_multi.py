#!/usr/bin/env python3
"""
実験27: 「開示からの日数」2列を足したとき、探索込みで4モデルのバックテストは
どう動くか。

実験24〜26 は本番のパラメータを固定して比べた（探索を引き直すと、その引きの
差 0.6〜1.5pt が特徴量の差を覆い隠すため）。ここでは本番が実際にやること
（週次で5分割の探索をやり直す）まで含めて、4モデル（LightGBM / XGBoost /
CatBoost / MLP。ロジスティック回帰は除く）で測る。

腕（モデルごとに3つ）
  A   151列 / 本番のパラメータ（lgbm_params.json の all、multi_params.json）。
      いずれも 151列で 5分割・50試行で探索済みのもの
  B1  153列 / A と同じパラメータ            —— 特徴量だけの効果
  B2  153列 / 153列で探索し直したパラメータ —— 本番が実際にやること
      （year_cap_date・5分割・50試行・木200本・ホールドアウトより手前、
        探索の種 0。本番の run_tuning.py / e15_tune_all.py と同じ）

out-of-fold は本番と同じ窓（36ヶ月 / 6ヶ月 / 6ヶ月・エンバーゴ20営業日）、
種3つ（42 / 7 / 123）の確率平均。判定は docs/MODEL_ADOPTION_RULES.md の
改訂版（§3）。PR-AUC の種レンジは LightGBM でしか測っていない（実験25:
3種平均 0.0024）ので、他のモデルは同じ値を仮に使う。

探索の DB は research/_data/oof/e27_optuna.db（本番の DB と混ぜない）。
結果は research/_data/oof/e27_*。本番の設定には書かない。
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
import models as M  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import walkforward as WF  # noqa: E402
import e19_freshdata as E19  # noqa: E402
from e18_horizon import edge  # noqa: E402
from e24_timing_ab import with_timing  # noqa: E402
from e25_auc_noise import OUTCOMES, average, metrics  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS)

MODELS = ("lgbm", "xgb", "cat", "mlp")
SEEDS3 = (42, 7, 123)
N_TRIALS = 50
N_SPLITS = 5
TIMING = ["days_since_disc", "days_since_fy"]
AUC_RANGE = 0.0024          # 実験25（LightGBM・3種平均）
WORST_TOL = 1.2             # 実験25（最悪の窓の種レンジ 1.18pt）
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LGBM_PARAMS = os.path.join(HERE, "lgbm_params.json")
MULTI_PARAMS = os.path.join(HERE, "multi_params.json")
STUDY_DB = os.path.join(OOF_DIR, "e27_optuna.db")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- #
# パラメータ
# ---------------------------------------------------------------------- #

def prod_params(algo: str) -> dict:
    if algo == "lgbm":
        with open(LGBM_PARAMS, encoding="utf-8") as fh:
            rec = json.load(fh)["all"]
        return {"params": {k: v for k, v in rec.items() if not k.startswith("_")},
                "_cv": rec.get("_cv", {})}
    with open(MULTI_PARAMS, encoding="utf-8") as fh:
        rec = json.load(fh)[algo]
    return {"params": dict(rec["params"]), "_cv": rec.get("_cv", {})}


def tune_b2(algo: str, sub: pd.DataFrame, cols: list) -> dict:
    path = os.path.join(OOF_DIR, f"e27_params_{algo}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        log(f"  [{algo}] 探索済みを読む (CV PR-AUC {rec['_cv'].get('mean_pr_auc')})")
        return rec
    t0 = time.time()
    if algo == "lgbm":
        params = tuning.tune(sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme="year_cap_date",
                             verbose=False)
        rec = {"params": dict(params), "_cv": dict(tuning.LAST_CV)}
    else:
        TM.STUDY_DB = STUDY_DB
        rec = TM.tune(algo, sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                      verbose=False)
    rec["_cv"]["seconds"] = round(time.time() - t0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{algo}] 探索 {(time.time()-t0)/60:.1f}分 / CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


# ---------------------------------------------------------------------- #
# out-of-fold
# ---------------------------------------------------------------------- #

def oof_multi(algo: str, df: pd.DataFrame, cols: list, params: dict, seed: int) -> pd.DataFrame:
    folds = WF.make_folds(pd.to_datetime(df["Date"]),
                          min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "label", "ref_end"] + list(OUTCOMES)
    parts = []
    TM.SEED = seed                       # build() が random_state に使う
    for f in folds:
        tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
        te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                & df["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        m = M.fit(algo, tr[cols].to_numpy(dtype=float), tr["label"].to_numpy(dtype=int),
                  cols, params=params)
        part = te[[c for c in keep if c in te.columns]].copy()
        part["score"] = M.predict(m, te[cols].to_numpy(dtype=float))
        part["fold"] = f.index
        parts.append(part)
    TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


def oof_arm(algo: str, tag: str, df: pd.DataFrame, cols: list, params: dict) -> pd.DataFrame:
    oofs = []
    for s in SEEDS3:
        p = os.path.join(OOF_DIR, f"e27_{algo}_{tag}_s{s}.parquet")
        if os.path.exists(p):
            oofs.append(pd.read_parquet(p))
            continue
        t0 = time.time()
        if algo == "lgbm":
            E19.SEEDS = (s,)
            o = E19.oof_for(df, cols, params)
        else:
            o = oof_multi(algo, df, cols, params, s)
        o.to_parquet(p, index=False)
        oofs.append(o)
        log(f"    {algo} {tag} 種 {s}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    return average(oofs)


# ---------------------------------------------------------------------- #
# 判定
# ---------------------------------------------------------------------- #

def fold_edges(o: pd.DataFrame, outcome: str, pct: int = 90, min_rows: int = 100) -> dict:
    out = {}
    for f in sorted(o["fold"].unique()):
        ref = o.loc[o["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        cur = o[o["fold"] == f]
        r = pd.to_numeric(cur[outcome], errors="coerce")
        if r.notna().sum() < min_rows:
            continue
        sel = cur[cur["score"] > np.percentile(ref, pct)]
        if not len(sel):
            continue
        out[int(f)] = float((pd.to_numeric(sel[outcome], errors="coerce").mean() - r.mean()) * 100)
    return out


def judge(ma: dict, mb: dict, oa: pd.DataFrame, ob: pd.DataFrame) -> dict:
    res = {"pr_auc_gain": mb["pr_auc"] - ma["pr_auc"]}
    res["auc_ok"] = res["pr_auc_gain"] > 2 * AUC_RANGE
    for oc in OUTCOMES:
        ea, eb = fold_edges(oa, oc), fold_edges(ob, oc)
        ks = sorted(set(ea) & set(eb))
        diffs = np.array([eb[k] - ea[k] for k in ks])
        recent = diffs[-4:]
        res[oc] = {"diff_all": float(diffs.mean()), "diff_recent4": float(recent.mean()),
                   "worst_a": float(min(ea[k] for k in ks)), "worst_b": float(min(eb[k] for k in ks)),
                   "won_b": f"{sum(eb[k] > 0 for k in ks)}/{len(ks)}",
                   "b_gt_a": f"{int((diffs > 0).sum())}/{len(ks)}"}
    r20, r40 = res["ret_o1_20"], res["ret_o1_40"]
    res["mean_ok"] = r20["diff_all"] >= 0 and r20["diff_recent4"] >= 0
    res["r40_ok"] = r40["diff_all"] >= 0
    res["worst_ok"] = all(res[oc]["worst_b"] >= res[oc]["worst_a"] - WORST_TOL for oc in OUTCOMES)
    res["pass"] = bool(res["auc_ok"] and res["mean_ok"] and res["r40_ok"] and res["worst_ok"])
    return res


def main(argv=None) -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    algos = [a for a in (argv or sys.argv[1:]) if a in MODELS] or list(MODELS)
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    df = with_timing(frame).rename(columns={"jq_days_since_disc": "days_since_disc",
                                            "jq_days_since_fy": "days_since_fy"})
    base = [c for c in F.columns("all") if c in df.columns and c not in TIMING]
    full = base + TIMING
    d = df["Date"]
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    log(f"母集団 {len(df):,}件 / 151列 vs 153列 / 探索 〜{train_end.date()} {len(sub):,}件 "
        f"/ {N_TRIALS}試行 × {N_SPLITS}分割 / 種 {SEEDS3} / モデル {algos}")

    summary = {}
    for algo in algos:
        log(f"=== {algo} ===")
        pa = prod_params(algo)
        pb2 = tune_b2(algo, sub, full)
        runs = {
            "A  151列/本番のパラメータ": oof_arm(algo, "A", df, base, pa["params"]),
            "B1 153列/同じパラメータ": oof_arm(algo, "B1", df, full, pa["params"]),
            "B2 153列/探索し直し": oof_arm(algo, "B2", df, full, pb2["params"]),
        }
        ms = {k: metrics(o) for k, o in runs.items()}
        cvs = {"A  151列/本番のパラメータ": pa["_cv"].get("mean_pr_auc"),
               "B1 153列/同じパラメータ": None,
               "B2 153列/探索し直し": pb2["_cv"].get("mean_pr_auc")}
        print(f"  {'腕':<26}{'CV PR-AUC':>10}{'OOF PR-AUC':>11}{'リフト':>7}{'ROC':>8}{'日内':>8}"
              f"{'窓平均20':>10}{'勝ち':>7}{'最悪':>9}{'窓平均40':>10}{'勝ち':>7}{'最悪':>9}")
        for k, m in ms.items():
            cv = cvs[k]
            print(f"  {k:<26}{(f'{cv:.4f}' if cv else '-'):>10}{m['pr_auc']:>11.4f}{m['lift']:>6.2f}x"
                  f"{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{m['ret_o1_20_mean']:>+9.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt"
                  f"{m['ret_o1_40_mean']:>+9.2f}pt{int(m['ret_o1_40_won']):>4}/{int(m['ret_o1_40_n']):<2}"
                  f"{m['ret_o1_40_worst']:>+8.2f}pt")
        names = list(runs)
        summary[algo] = {"metrics": {k: {kk: float(vv) for kk, vv in m.items()} for k, m in ms.items()},
                         "cv": cvs, "b2_params": pb2["params"], "judge": {}}
        for b in (1, 2):
            v = judge(ms[names[0]], ms[names[b]], runs[names[0]], runs[names[b]])
            summary[algo]["judge"][names[b][:2].strip()] = v
            r20, r40 = v["ret_o1_20"], v["ret_o1_40"]
            print(f"  --- A → {names[b][:2].strip()} の判定（改訂版の規則）---")
            print(f"    PR-AUC {v['pr_auc_gain']:+.4f} (> {2*AUC_RANGE:.4f}) {'○' if v['auc_ok'] else '×'} / "
                  f"窓平均20 全期間 {r20['diff_all']:+.2f}pt 直近4窓 {r20['diff_recent4']:+.2f}pt {'○' if v['mean_ok'] else '×'} / "
                  f"窓平均40 {r40['diff_all']:+.2f}pt {'○' if v['r40_ok'] else '×'} / "
                  f"最悪 {r20['worst_a']:+.2f}→{r20['worst_b']:+.2f}pt {'○' if v['worst_ok'] else '×'} "
                  f"→ {'満たす' if v['pass'] else '満たさない'}  (B>A の窓 {r20['b_gt_a']})")
        with open(os.path.join(OOF_DIR, "e27_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OOF_DIR}/e27_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
