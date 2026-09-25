#!/usr/bin/env python3
"""
実験49: LightGBM の木の本数を 200 -> 500 にする前に、同じデータで3通りを探索して比べる。

運用者の指示（2026-09-25）「どうせならn_estimatorの数を500に変更して下さい。特徴量も
増えてきたので。」本番の探索（tuning.SEARCH_N_ESTIMATORS）は 500 に変えた。日曜（9/27）の
週次実行を待たずに、同じ作りで次を確かめる。

1. 500本の探索が本番と同じ関数で最後まで動くこと
2. 学習率が探索範囲の端に張り付かないか。200本では 0.016〜0.024 が選ばれてきた
   （学習率 × 本数 = 歩幅の合計 3.2〜4.8）。500本で同じ合計になる学習率は 0.006〜0.010 で、
   以前の範囲 0.01〜0.2 の下限より下。本番は範囲を本数に合わせて 0.004〜0.08 にした
   （tuning.LR_TOTAL）。範囲を動かさない腕（C）も並べて、動かした効果を見る
3. 200本（これまで）と比べて、out-of-fold と窓ごとでどうか

腕（探索の関数・条件はどれも本番の週次実行と同じ。run_tuning.py --n-trials 50
--cv year_cap_date。違うのは木の本数と学習率の範囲だけ）
  A  木200本・学習率 0.01〜0.2    これまでの週次実行（実験48 の N と同じ作り）
  B  木500本・学習率 0.004〜0.08  日曜からの週次実行（本数に合わせて範囲を動かす）
  C  木500本・学習率 0.01〜0.2    本数だけ変えた場合

評価（実験48 と同じ）
1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）と所要時間
2. 本番と同じ作りの out-of-fold（train_production.oof_scores。種1つ・ずらし0か月）
3. 窓ごと: 窓の境界を 0/2/4か月ずらした3通り（計32窓）・種3つの平均。B−A / C−A / B−C。
   上位10%の ret_o1_20 の超過（しきい値は前の窓だけから決める。lab.threshold_edge）

本番の設定（research/lgbm_params.json）には書かない（research/_data/oof/e49_*）。

使い方
    python3 research/exp/e49_trees.py                  # 探索して評価
    python3 research/exp/e49_trees.py --shifts 0       # 窓ごとをずらし0か月だけ
    python3 research/exp/e49_trees.py --n-trials 2     # 試運転
"""
from __future__ import annotations

import argparse
import hashlib
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
import train_model as T  # noqa: E402
import train_production as TP  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = "year_cap_date"          # 本番の retrain-weekly.yml と同じ
SEEDS = (42, 7, 123)
OUTCOME = TP.OUTCOME_COL             # ret_o1_20（翌営業日の寄り買い・20営業日）

#: 腕 -> (木の本数, 学習率の範囲)
ARMS = {
    "A": (200, tuning.lr_range(200)),
    "B": (500, tuning.lr_range(500)),
    "C": (500, tuning.lr_range(200)),
}
LABELS = {"A": "A 木200本・学習率 0.01〜0.2（これまで）",
          "B": "B 木500本・学習率 0.004〜0.08（日曜から）",
          "C": "C 木500本・学習率 0.01〜0.2（本数だけ）"}
PAIRS = (("A", "B"), ("A", "C"), ("C", "B"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]


def tune_arm(arm: str, tune_df: pd.DataFrame, cols: list, cutoff, n_trials: int) -> dict:
    """run_tuning.py と同じ入力・同じ関数で探索する。本数と学習率の範囲だけを変える。"""
    trees, lr = ARMS[arm]
    path = os.path.join(OOF_DIR, f"e49_params_{arm}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_features_sig") == F.signature(cols) and rec.get("_n_trials") == n_trials
                and rec.get("n_estimators") == trees and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_cv", {}).get("lr_range") == list(lr)):
            log(f"[{arm}] 探索済みを読む（{path}）")
            return rec
    log(f"[{arm}] 探索: 木{trees}本・学習率 {lr[0]:g}〜{lr[1]:g} / {n_trials}試行 × {N_SPLITS}分割")
    t0 = time.time()
    params = tuning.tune(tune_df, cols, n_trials=n_trials, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, model="classifier",
                         n_estimators=trees, lr_bounds=lr)
    rec = {**params, "_cv": dict(tuning.LAST_CV), "_n_features": len(cols),
           "_features_sig": F.signature(cols), "_preset": F.DEFAULT_PRESET,
           "_n_trials": n_trials, "_minutes": round((time.time() - t0) / 60, 1),
           "_cutoff": str(cutoff.date())}
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def show_tuning(recs: dict) -> None:
    print("\n■ 1. 探索（本番と同じ条件。層別5分割の CV は楽観側に出る。パラメータ選び用）")
    keys = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    for arm, rec in recs.items():
        cv = rec.get("_cv", {})
        p = {k: v for k, v in rec.items() if not k.startswith("_")}
        lo, hi = cv.get("lr_range", [np.nan, np.nan])
        lr = p["learning_rate"]
        # 範囲の端からどれだけ離れているか（対数の目盛りで 0=下限・1=上限）
        pos = (np.log(lr) - np.log(lo)) / (np.log(hi) - np.log(lo))
        print(f"\n  {LABELS[arm]}")
        print(f"    所要 {rec.get('_minutes')}分 / 探索に使った期間 〜{rec.get('_cutoff')} / "
              f"{rec.get('_n_trials')}試行")
        print(f"    CV PR-AUC {cv.get('mean_pr_auc')} ± {cv.get('std')}（正例率 {cv.get('base_rate')}）"
              f" / ROC-AUC {cv.get('mean_roc_auc')} ± {cv.get('roc_std')}")
        if cv.get("fold_scores"):
            print("    分割ごとの PR-AUC: " + " / ".join(f"{v:.4f}" for v in cv["fold_scores"]))
        print("    パラメータ: " + " / ".join(f"{k} {p[k]:.4g}" if isinstance(p.get(k), float)
                                       else f"{k} {p.get(k)}" for k in keys if k in p))
        print(f"    学習率 {lr:.5f}（範囲 {lo:g}〜{hi:g} の中の位置 {pos:.2f}。0 が下限）/ "
              f"歩幅の合計（学習率 × 本数）{lr * p['n_estimators']:.2f}")


def production_oof(df: pd.DataFrame, cols: list, params: dict, arm: str) -> pd.DataFrame:
    """train_production.oof_scores そのもの（種1つ・ずらし0か月）。日曜に画面へ出る数字と同じ作り。"""
    path = os.path.join(OOF_DIR, f"e49_prod_{arm}_{AB.fingerprint(df, cols)}_"
                                 f"{params_hash(params)}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    oof = TP.oof_scores(df, cols, params)
    oof.to_parquet(path, index=False)
    log(f"  本番と同じ作りの out-of-fold（{arm}）{time.time()-t0:.0f}秒")
    return oof


def show_production(oofs: dict) -> None:
    print("\n■ 2. 本番と同じ作りの out-of-fold（train_production.oof_scores。種1つ・ずらし0か月）")
    print(f"  {'':<40}{'件数':>7}{'正例率':>8}{'PR-AUC':>8}{'÷正例率':>8}{'ROC-AUC':>9}{'日内AUC':>9}")
    for arm, oof in oofs.items():
        m = TP.oof_metrics(oof)
        print(f"  {LABELS[arm]:<40}{m['n']:>7,}{m['positiveRate']*100:>7.2f}%{m['prAuc']:>8.4f}"
              f"{m['prAucOverBase']:>8.3f}{m['rocAuc']:>9.4f}{m['aucInDay']:>9.4f}")
    print(f"\n  上位3つのスコア帯（帯10が最上位。実収益は {OUTCOME}）")
    print(f"  {'':<40}{'帯':>4}{'件数':>7}{'正例率':>8}{'実収益の中央':>12}{'平均':>8}{'勝率':>7}")
    for arm, oof in oofs.items():
        bands = TP.score_bands(oof)
        for r in bands["bands"][-3:]:
            med = r["outcome_median"] if r["outcome_median"] is not None else float("nan")
            mean = r["outcome_mean"] if r["outcome_mean"] is not None else float("nan")
            print(f"  {LABELS[arm]:<40}{r['band'] + 1:>4}{r['n']:>7,}"
                  f"{r['positive_rate']*100:>7.2f}%{med:>+11.2f}%{mean:>+7.2f}%"
                  f"{(r['win_rate'] or 0) * 100:>6.1f}%")


def window_oof(df: pd.DataFrame, cols: list, params: dict, arm: str, shift: int):
    """窓の境界を shift か月ずらした out-of-fold（種3つの平均）。保存済みなら読む。"""
    folds = E41.folds_for(df["Date"], shift)
    fp = AB.fingerprint(df, cols)
    ph = params_hash(params)
    parts = []
    for sd in SEEDS:
        path = os.path.join(OOF_DIR, f"e49_win_{arm}_{fp}_{ph}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds("lgbm", df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    out = average(parts)
    out["Date"] = pd.to_datetime(out["Date"])
    return out, folds


def show_windows(df: pd.DataFrame, cols: list, params: dict, shifts: list) -> pd.DataFrame:
    rows, edges = [], []
    for sh in shifts:
        res = {}
        for arm in ARMS:
            res[arm], folds = window_oof(df, cols, params[arm], arm, sh)
        spans = {f.index: (pd.Timestamp(f.test_start).date(), pd.Timestamp(f.test_end).date())
                 for f in folds}
        w = None
        for arm in ARMS:
            a = AB.auc_by_window(res[arm]).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            w = a if w is None else w.merge(a, on="fold")
        print(f"\n■ 3. 窓ごと（ずらし{sh}か月・種3つの平均）: 腕の違いは木の本数と学習率の範囲だけ")
        print(f"  {'窓':>3} {'検証の期間':<23}{'件数':>6}{'正例率':>8}"
              f"{'PR A':>8}{'PR B':>8}{'PR C':>8}{'B−A':>9}{'C−A':>9}"
              f"{'ROC A':>8}{'ROC B':>8}{'ROC C':>8}")
        for _, r in w.iterrows():
            g = res["A"][res["A"]["fold"] == r["fold"]]
            a, b = spans.get(int(r["fold"]), ("?", "?"))
            print(f"  {int(r['fold']):>3} {str(a)}〜{str(b)}{len(g):>6,}{g['label'].mean()*100:>7.2f}%"
                  f"{r['pr_A']:>8.4f}{r['pr_B']:>8.4f}{r['pr_C']:>8.4f}"
                  f"{r['pr_B'] - r['pr_A']:>+9.4f}{r['pr_C'] - r['pr_A']:>+9.4f}"
                  f"{r['roc_A']:>8.4f}{r['roc_B']:>8.4f}{r['roc_C']:>8.4f}")
            rows.append({"shift": sh, "fold": int(r["fold"]), "test_start": str(a),
                         "test_end": str(b), "n": int(len(g)), "pos": float(g["label"].mean()),
                         **{f"{m}_{arm}": r[f"{m}_{arm}"] for m in ("pr", "roc") for arm in ARMS}})
        print(f"  {'':<18}{'前':>9}{'後':>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
        for x, y in PAIRS:
            print(AB.pair_line(f"PR-AUC {y}−{x}", w[f"pr_{x}"].to_numpy(), w[f"pr_{y}"].to_numpy()))
        for x, y in PAIRS:
            print(AB.pair_line(f"ROC-AUC {y}−{x}", w[f"roc_{x}"].to_numpy(), w[f"roc_{y}"].to_numpy()))
        # 上位10%の実収益の超過（しきい値は前の窓のスコア分布から。窓1は対象外）
        for arm in ARMS:
            e = lab.threshold_edge(res[arm], outcome=OUTCOME)
            se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
            edges.append({"shift": sh, "arm": arm, **e, "se": se})
            print(f"  上位10%の {OUTCOME}（{arm}）: 取引 {e['thr_n']:,}件 / 平均 {e['thr_end']:+.2f}% / "
                  f"全体との差 {e['thr_lift']:+.2f}pt / 窓ごとの差の平均 {e['thr_fold_mean']:+.2f} ± "
                  f"{se:.2f}pt（上の窓 {e['thr_folds_won']}/{e['thr_folds']}、最悪 {e['thr_worst']:+.2f}pt）")
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OOF_DIR, "e49_auc_by_window.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(OOF_DIR, "e49_edges.csv"), index=False)
    if len(shifts) > 1 and len(s):
        print(f"\n■ 4. 切り方{len(shifts)}通りをまとめて（{len(s)}窓）")
        print(f"  {'':<18}{'前':>9}{'後':>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
        for x, y in PAIRS:
            print(AB.pair_line(f"PR-AUC {y}−{x}", s[f"pr_{x}"].to_numpy(), s[f"pr_{y}"].to_numpy()))
        for x, y in PAIRS:
            print(AB.pair_line(f"ROC-AUC {y}−{x}", s[f"roc_{x}"].to_numpy(), s[f"roc_{y}"].to_numpy()))
        ed = pd.DataFrame(edges)
        for arm in ARMS:
            g = ed[ed["arm"] == arm]
            print(f"  上位10%の {OUTCOME} の全体との差（{arm}）: 切り方ごと "
                  + " / ".join(f"{v:+.2f}" for v in g["thr_lift"]) + " pt"
                  f"（平均 {g['thr_lift'].mean():+.2f}pt）")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験49: 木の本数 200 と 500 を比べる")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print(f"実験49 木の本数 200 と 500 を同じデータで探索して比べる（{F.DEFAULT_PRESET}、"
          f"{len(cols)}列、指紋 {F.signature(cols)}）")
    print(f"  目的変数 {B.DEFAULT_RISE.name}")
    print(f"  本番の探索の本数 tuning.SEARCH_N_ESTIMATORS = {tuning.SEARCH_N_ESTIMATORS}"
          f"（学習率 {tuning.lr_range(tuning.SEARCH_N_ESTIMATORS)[0]:g}〜"
          f"{tuning.lr_range(tuning.SEARCH_N_ESTIMATORS)[1]:g}）")
    print("=" * 78)

    # run_tuning.py と同じ: データセットを (Date, Code) の順で読み、ホールドアウトより前で打ち切る
    raw = (pd.read_parquet(os.path.join(lab.DATA_DIR, "dataset.parquet"))
           .sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True))
    dates = pd.to_datetime(raw["Date"])
    cutoff, _, _ = T.holdout_bounds(dates, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    tune_df = raw[dates <= cutoff]
    log(f"[tune] 全体 {len(raw):,}件 / 探索に使う期間 〜{cutoff.date()} "
        f"（{len(tune_df):,}件 / 正例率 {tune_df['label'].mean()*100:.2f}%）/ "
        f"{args.n_trials}試行 × {N_SPLITS}分割 / {CV_SCHEME} / エンバーゴ {T.EMBARGO_DAYS}営業日")
    recs = {arm: tune_arm(arm, tune_df, cols, cutoff, args.n_trials) for arm in ARMS}
    del raw, tune_df
    show_tuning(recs)
    params = {arm: tuning.params_for(arm, store={arm: rec}) for arm, rec in recs.items()}

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:8]}")
    key = list(zip(df["Date"], df["Code"]))
    log(f"データ {len(df):,}件 / 正例率 {df['label'].mean()*100:.2f}% / "
        f"{df['Date'].min().date()} 〜 {df['Date'].max().date()} / "
        f"(Date, Code) の順: {key == sorted(key)}")

    oofs = {arm: production_oof(df, cols, params[arm], arm) for arm in ARMS}
    show_production(oofs)
    show_windows(df, cols, params, shifts)
    log(f"記録: {OOF_DIR}/e49_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
