#!/usr/bin/env python3
"""
実験28: EDINET の年次特徴量を、探索込み・5モデルで A/B する。

実験27（開示からの日数）と同じ手順を EDINET の候補（research/edinet_features.py）に
当てる。取得は 1社1リクエスト・日 85社 で貯まっていくので、

  --dry     行の充足だけ出して終わる（取得の進み具合の確認）
  --rows    covered: EDINET の y0 が付いた行だけで A/B（既定。充足が低い間は
                     全行で比べても差が薄まるだけ）
            all:     全行（充足が 8割を超えたらこちらも見る）
  --set     足す列の集合。core / core+mcap / all（既定 core+mcap）。
            検定（実験22）の結果で列を選ばない（同じ窓で選ぶと楽観になる）

腕（モデルごと。ロジスティック回帰は A と B1）
  A   本番の特徴量（153列: all）/ 本番のパラメータ
  B1  A + EDINET / 同じパラメータ            —— 特徴量だけの効果
  B2  A + EDINET / 探索し直し（5分割 50試行） —— 本番が実際にやること
      LightGBM は探索の種を 3つ（0 / 1 / 2）引き、引きの分布で見る

判定は docs/MODEL_ADOPTION_RULES.md §3（B1 で判定）と §5（B2 の扱い）。
結果は research/_data/oof/e28_*。本番の設定には書かない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edinet_features as EF  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
from e24_timing_ab import with_timing  # noqa: E402
from e25_auc_noise import metrics  # noqa: E402
from e27_timing_multi import (  # noqa: E402
    MODELS, N_SPLITS, N_TRIALS, NO_TUNE, OOF_DIR, judge, log, oof_arm, prod_params)

TIMING = ["days_since_disc", "days_since_fy"]
SETS = {
    "core": lambda: EF.columns("core"),
    "core+mcap": lambda: EF.columns("core") + EF.columns("mcap"),
    "all": lambda: EF.columns("all"),
}
LGBM_DRAWS = (0, 1, 2)
STUDY_DB = os.path.join(OOF_DIR, "e28_optuna.db")


def tune_set(algo: str, sub: pd.DataFrame, cols: list, tag: str, seed: int = 0) -> dict:
    path = os.path.join(OOF_DIR, f"e28_params_{tag}_{algo}_t{seed}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = time.time()
    if algo == "lgbm":
        params = tuning.tune(sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme="year_cap_date",
                             seed=seed, verbose=False)
        rec = {"params": dict(params), "_cv": dict(tuning.LAST_CV)}
    else:
        TM.STUDY_DB = STUDY_DB
        rec = TM.tune(algo, sub, cols, n_trials=N_TRIALS, n_splits=N_SPLITS,
                      seed=seed, verbose=False)
    rec["_cv"]["seconds"] = round(time.time() - t0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{algo} 種{seed}] 探索 {(time.time()-t0)/60:.1f}分 / CV PR-AUC {rec['_cv'].get('mean_pr_auc')}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rows", choices=["covered", "all"], default="covered")
    ap.add_argument("--set", choices=list(SETS), default="core+mcap")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--fin", default=EF.FIN)
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)
    tag = f"{args.set.replace('+', '_')}_{args.rows}"

    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    df = with_timing(frame).rename(columns={"jq_days_since_disc": "days_since_disc",
                                            "jq_days_since_fy": "days_since_fy"})
    base = [c for c in F.columns("all") if c in df.columns]
    if not os.path.exists(args.fin):
        raise SystemExit(f"{args.fin} がありません（fetch-edinetdb.yml の成果物）")
    fin = EF.load_fin(args.fin)
    df = EF.attach(df, EF.feature_frame(EF.annual_panel(fin)))
    have = df["ed_fiscal_year"].notna()
    add = [c for c in SETS[args.set]() if c in df.columns]
    n_codes = fin[EF.KEY].nunique()
    log(f"母集団 {len(df):,}件 / EDINET 取得済み {n_codes:,}社 / y0 が付いた行 "
        f"{have.sum():,}件（{have.mean()*100:.1f}%）/ 足す列 {args.set}（{len(add)}列）"
        f"/ 充足（付いた行の中で）中央値 {df.loc[have, add].notna().mean().median()*100:.0f}%")
    if args.dry:
        by_year = df.loc[have, "Date"].dt.year.value_counts().sort_index()
        print("  y0 が付いた行の年別:", by_year.to_dict())
        return 0
    if args.rows == "covered":
        df = df[have].reset_index(drop=True)
        log(f"  covered: {len(df):,}件で A/B（両腕とも同じ行）")
    if len(df) < 2000:
        log("  ※ 2,000行未満。窓ごとの判定には足りない（--dry で待つ）")

    d = df["Date"]
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    full = base + add
    summary = {}
    for algo in [a for a in args.models.split(",") if a in MODELS]:
        log(f"=== {algo} ===")
        pa = prod_params(algo)
        runs = {"A  本番/本番のパラメータ": oof_arm(algo, f"e28{tag}_A", df, base, pa["params"]),
                "B1 +EDINET/同じパラメータ": oof_arm(algo, f"e28{tag}_B1", df, full, pa["params"])}
        cvs = {"A  本番/本番のパラメータ": pa["_cv"].get("mean_pr_auc"), "B1 +EDINET/同じパラメータ": None}
        if algo not in NO_TUNE:
            draws = LGBM_DRAWS if algo == "lgbm" else (0,)
            for s in draws:
                rec = tune_set(algo, sub, full, tag, s)
                name = f"B2 +EDINET/探索し直し 種{s}"
                runs[name] = oof_arm(algo, f"e28{tag}_B2t{s}", df, full, rec["params"])
                cvs[name] = rec["_cv"].get("mean_pr_auc")
        ms = {k: metrics(o) for k, o in runs.items()}
        print(f"  {'腕':<30}{'CV PR-AUC':>10}{'OOF PR-AUC':>11}{'リフト':>7}{'ROC':>8}"
              f"{'窓平均20':>10}{'勝ち':>7}{'最悪':>9}{'窓平均40':>10}{'最悪':>9}")
        for k, m in ms.items():
            cv = cvs[k]
            print(f"  {k:<30}{(f'{cv:.4f}' if cv else '-'):>10}{m['pr_auc']:>11.4f}{m['lift']:>6.2f}x"
                  f"{m['roc_auc']:>8.4f}{m['ret_o1_20_mean']:>+9.2f}pt"
                  f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                  f"{m['ret_o1_40_mean']:>+9.2f}pt{m['ret_o1_40_worst']:>+8.2f}pt")
        names = list(runs)
        summary[algo] = {"metrics": {k: {kk: float(vv) for kk, vv in m.items()} for k, m in ms.items()},
                         "cv": cvs, "judge": {}}
        for b in range(1, len(names)):
            v = judge(ms[names[0]], ms[names[b]], runs[names[0]], runs[names[b]])
            summary[algo]["judge"][names[b]] = v
            r20 = v["ret_o1_20"]
            print(f"  A → {names[b]}: PR-AUC {v['pr_auc_gain']:+.4f} {'○' if v['auc_ok'] else '×'} / "
                  f"窓平均20 {r20['diff_all']:+.2f}pt（直近4窓 {r20['diff_recent4']:+.2f}）{'○' if v['mean_ok'] else '×'} / "
                  f"窓平均40 {v['ret_o1_40']['diff_all']:+.2f}pt {'○' if v['r40_ok'] else '×'} / "
                  f"最悪 {r20['worst_a']:+.2f}→{r20['worst_b']:+.2f} {'○' if v['worst_ok'] else '×'} "
                  f"→ {'満たす' if v['pass'] else '満たさない'}")
        with open(os.path.join(OOF_DIR, f"e28_summary_{tag}_{algo}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary[algo], fh, ensure_ascii=False, indent=1, default=float)
    # 運用の規則（3モデル揃って上位）で実収益を測る。3モデル分が揃っているときだけ。
    # out-of-fold の保存名は oof_arm の規則（e27_{algo}_{tag}_s{seed}.parquet）
    try:
        import ops_rule as OR
        from e25_auc_noise import average
        if all(a in summary for a in OR.BOOST):
            print("\n=== 運用の規則: ブースティング3モデルすべてが過去窓の 85 パーセンタイル以上 ===")
            print(OR.HEADER)

            def load(algo, arm_tag):
                files = [os.path.join(OOF_DIR, f"e27_{algo}_e28{tag}_{arm_tag}_s{s}.parquet")
                         for s in (42, 7, 123)]
                return average([pd.read_parquet(f) for f in files]) if all(map(os.path.exists, files)) else None

            arms = [("A", {a: "A" for a in OR.BOOST}), ("B1", {a: "B1" for a in OR.BOOST})]
            for s_ in LGBM_DRAWS:
                arms.append((f"B2（lgbm 種{s_} / 他は種0）",
                             {"lgbm": f"B2t{s_}", "xgb": "B2t0", "cat": "B2t0"}))
            for name, tags in arms:
                oofs = {a: load(a, t) for a, t in tags.items()}
                if any(v is None for v in oofs.values()):
                    continue
                print(OR.fmt(name, OR.consensus(oofs, 85)))
    except Exception as exc:  # noqa: BLE001
        print(f"  運用の規則の集計に失敗: {type(exc).__name__}: {exc}")
    log(f"記録: {OOF_DIR}/e28_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
