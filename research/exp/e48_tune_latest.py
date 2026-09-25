#!/usr/bin/env python3
"""
実験48: 2026-09-25 の直しをすべて入れた最新の作りで、LightGBM を5分割探索し、
out-of-fold と窓ごとで評価する。

運用者の依頼（2026-09-25）「いまlgbmだけで良いので、諸々修正を加えた最新版で、5cvの
パラメータチューニングとoof評価をしてほしいです。」日曜（9/27）の定時の週次実行を待たずに、
同じ作りの結果を先に見る。

最新の作り（この実験を回すコミットの build_dataset・features の既定）
- 列: 本番の206列（all_plus_prog_listing。進捗期待の新しい定義・上場からの年数）
- 母集団: TOKYO PRO MARKET の時期の行を78週の履歴に数えない（GENERAL_MARKET_START）
- 目的変数: 翌営業日の寄りを基準にする（LABEL_ENTRY = next_open）
- 学習データの行の並び: (Date, Code)
- 取り込みの直し（重複除去・同じ日に2つの期の決算の結合 など）の後の生データ

探索は本番の週次実行（run_tuning.py --n-trials 50 --cv year_cap_date）と同じ関数・同じ条件
（50試行・5分割・year_cap_date・ホールドアウトより前で打ち切り・種0）。保存先だけが違い、
本番の設定（research/lgbm_params.json）には書かない（research/_data/oof/e48_params.json）。

評価
1. 探索の CV（層別5分割）。訓練側に将来のデータが入るので楽観側に出る。パラメータを
   選ぶためだけの値で、実力の推定には使わない
2. 本番と同じ作りの out-of-fold（train_production.oof_scores。種1つ・ずらし0か月）の
   PR-AUC / ROC-AUC / 日内AUC とスコア帯（正例率・ret_o1_20）。日曜に画面へ出る数字と同じ作り
3. 窓ごと（運用者「oof だけでなく、各cv（異なる窓）でもみたい」）: 窓の境界を 0/2/4か月
   ずらした3通り（計32窓）・種3つの平均で、探索したパラメータ（N）と、これまで実験で
   使っていたパラメータ（O: 2026-09-20 に all の151列で探索）を同じデータで比べる。
   差はパラメータだけ。あわせて上位10%の ret_o1_20 の超過（しきい値は前の窓だけから決める。
   lab.threshold_edge）

使い方
    python3 research/exp/e48_tune_latest.py                  # 探索して評価
    python3 research/exp/e48_tune_latest.py --shifts 0       # 窓ごとをずらし0か月だけ
    python3 research/exp/e48_tune_latest.py --n-trials 2     # 試運転
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
PARAMS_PATH = os.path.join(OOF_DIR, "e48_params.json")
N_SPLITS = 5
CV_SCHEME = "year_cap_date"          # 本番の retrain-weekly.yml と同じ
SEEDS = (42, 7, 123)
OUTCOME = TP.OUTCOME_COL             # ret_o1_20（翌営業日の寄り買い・20営業日）
LABELS = {"O": "O 9/20 のパラメータ（all 151列で探索）",
          "N": "N 今回探索したパラメータ（206列）"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]


def tune_latest(cols: list, n_trials: int) -> dict:
    """run_tuning.py と同じ入力・同じ関数で探索する。保存先だけが違う。"""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("_features_sig") == F.signature(cols) and rec.get("_n_trials") == n_trials:
            log(f"探索済みを読む（{PARAMS_PATH}）")
            return rec
    # run_tuning.py と同じ: データセットを (Date, Code) の順で読み、ホールドアウトより前で打ち切る
    df = (pd.read_parquet(os.path.join(lab.DATA_DIR, "dataset.parquet"))
          .sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True))
    dates = pd.to_datetime(df["Date"])
    cutoff, _, _ = T.holdout_bounds(dates, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    tune_df = df[dates <= cutoff]
    log(f"[tune] 全体 {len(df):,}件 / 探索に使う期間 〜{cutoff.date()} "
        f"（{len(tune_df):,}件 / 正例率 {tune_df['label'].mean()*100:.2f}%）/ "
        f"{n_trials}試行 × {N_SPLITS}分割 / {CV_SCHEME} / エンバーゴ {T.EMBARGO_DAYS}営業日")
    t0 = time.time()
    params = tuning.tune(tune_df, cols, n_trials=n_trials, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, model="classifier")
    rec = {**params, "_cv": dict(tuning.LAST_CV), "_n_features": len(cols),
           "_features_sig": F.signature(cols), "_preset": F.DEFAULT_PRESET,
           "_n_trials": n_trials, "_minutes": round((time.time() - t0) / 60, 1),
           "_cutoff": str(cutoff.date())}
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def show_tuning(rec: dict) -> None:
    cv = rec.get("_cv", {})
    p = {k: v for k, v in rec.items() if not k.startswith("_")}
    print("\n■ 1. 探索（本番と同じ条件。層別5分割の CV は楽観側に出る。パラメータ選び用）")
    print(f"  所要 {rec.get('_minutes')}分 / 探索に使った期間 〜{rec.get('_cutoff')} / "
          f"{rec.get('_n_trials')}試行")
    print(f"  CV PR-AUC {cv.get('mean_pr_auc')} ± {cv.get('std')}（正例率 {cv.get('base_rate')}）"
          f" / ROC-AUC {cv.get('mean_roc_auc')} ± {cv.get('roc_std')}")
    if cv.get("fold_scores"):
        print("  分割ごとの PR-AUC: " + " / ".join(f"{v:.4f}" for v in cv["fold_scores"]))
    if cv.get("fold_roc"):
        print("  分割ごとの ROC-AUC: " + " / ".join(f"{v:.4f}" for v in cv["fold_roc"]))
    keys = ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
            "subsample", "subsample_freq", "colsample_bytree", "reg_alpha", "reg_lambda"]
    print("  パラメータ: " + " / ".join(f"{k} {p[k]:.4g}" if isinstance(p.get(k), float)
                                     else f"{k} {p.get(k)}" for k in keys if k in p))


def production_oof(df: pd.DataFrame, cols: list, params: dict, name: str) -> pd.DataFrame:
    """train_production.oof_scores そのもの（種1つ・ずらし0か月）。日曜に画面へ出る数字と同じ作り。"""
    path = os.path.join(OOF_DIR, f"e48_prod_{name}_{AB.fingerprint(df, cols)}_"
                                 f"{params_hash(params)}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    log(f"  本番と同じ作りの out-of-fold（{name}）")
    oof = TP.oof_scores(df, cols, params)
    oof.to_parquet(path, index=False)
    return oof


def show_production(oofs: dict) -> None:
    print("\n■ 2. 本番と同じ作りの out-of-fold（train_production.oof_scores。種1つ・ずらし0か月）")
    print(f"  {'':<40}{'件数':>7}{'正例率':>8}{'PR-AUC':>8}{'÷正例率':>8}{'ROC-AUC':>9}{'日内AUC':>9}")
    for name, oof in oofs.items():
        m = TP.oof_metrics(oof)
        print(f"  {LABELS[name]:<40}{m['n']:>7,}{m['positiveRate']*100:>7.2f}%{m['prAuc']:>8.4f}"
              f"{m['prAucOverBase']:>8.3f}{m['rocAuc']:>9.4f}{m['aucInDay']:>9.4f}")
    for name, oof in oofs.items():
        bands = TP.score_bands(oof)
        print(f"\n  スコア帯（{LABELS[name]}。帯10が最上位。実収益は {OUTCOME}）")
        print(f"  {'帯':>4}{'件数':>7}{'正例率':>8}{'実収益の中央':>12}{'平均':>8}{'勝率':>7}")
        for r in bands["bands"]:
            print(f"  {r['band'] + 1:>4}{r['n']:>7,}{r['positive_rate']*100:>7.2f}%"
                  f"{(r['outcome_median'] if r['outcome_median'] is not None else float('nan')):>+11.2f}%"
                  f"{(r['outcome_mean'] if r['outcome_mean'] is not None else float('nan')):>+7.2f}%"
                  f"{(r['win_rate'] or 0) * 100:>6.1f}%")
        print(f"  全体: 正例率 {bands['base_positive_rate']*100:.2f}% / 実収益の中央 "
              f"{bands['base_outcome_median']:+.2f}% / 勝率 {bands['base_win_rate']*100:.1f}%")


def window_oof(df: pd.DataFrame, cols: list, params: dict, name: str, shift: int):
    """窓の境界を shift か月ずらした out-of-fold（種3つの平均）。保存済みなら読む。"""
    folds = E41.folds_for(df["Date"], shift)
    fp = AB.fingerprint(df, cols)
    ph = params_hash(params)
    parts = []
    for sd in SEEDS:
        path = os.path.join(OOF_DIR, f"e48_win_{name}_{fp}_{ph}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds("lgbm", df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  {name} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    out = average(parts)
    out["Date"] = pd.to_datetime(out["Date"])
    return out, folds


def show_windows(df: pd.DataFrame, cols: list, params: dict, shifts: list) -> pd.DataFrame:
    rows = []
    edges = []
    for sh in shifts:
        res = {}
        for name in ("O", "N"):
            res[name], folds = window_oof(df, cols, params[name], name, sh)
        spans = {f.index: (pd.Timestamp(f.test_start).date(), pd.Timestamp(f.test_end).date())
                 for f in folds}
        wo, wn = AB.auc_by_window(res["O"]), AB.auc_by_window(res["N"])
        m = wo.merge(wn, on="fold", suffixes=("_o", "_n"))
        print(f"\n■ 3. 窓ごと（ずらし{sh}か月・種3つの平均）: N − O は差がパラメータだけ")
        print(f"  {'窓':>3} {'検証の期間':<23}{'件数':>6}{'正例率':>8}"
              f"{'PR O':>8}{'PR N':>8}{'差':>9}{'ROC O':>8}{'ROC N':>8}")
        for _, r in m.iterrows():
            g = res["N"][res["N"]["fold"] == r["fold"]]
            a, b = spans.get(int(r["fold"]), ("?", "?"))
            print(f"  {int(r['fold']):>3} {str(a)}〜{str(b)}{len(g):>6,}{g['label'].mean()*100:>7.2f}%"
                  f"{r['pr_o']:>8.4f}{r['pr_n']:>8.4f}{r['pr_n'] - r['pr_o']:>+9.4f}"
                  f"{r['roc_o']:>8.4f}{r['roc_n']:>8.4f}")
            rows.append({"shift": sh, "fold": int(r["fold"]), "test_start": str(a),
                         "test_end": str(b), "n": int(len(g)), "pos": float(g["label"].mean()),
                         "pr_o": r["pr_o"], "pr_n": r["pr_n"],
                         "roc_o": r["roc_o"], "roc_n": r["roc_n"]})
        print(f"  {'':<40}{'O':>9}{'N':>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
        print(AB.pair_line("PR-AUC", m["pr_o"].to_numpy(), m["pr_n"].to_numpy()))
        print(AB.pair_line("ROC-AUC", m["roc_o"].to_numpy(), m["roc_n"].to_numpy()))
        # 上位10%の実収益の超過（しきい値は前の窓のスコア分布から。窓1は対象外）
        for name in ("O", "N"):
            e = lab.threshold_edge(res[name], outcome=OUTCOME)
            se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
            edges.append({"shift": sh, "arm": name, **e, "se": se})
            print(f"  上位10%の {OUTCOME}（{name}）: 取引 {e['thr_n']:,}件 / 平均 {e['thr_end']:+.2f}% / "
                  f"全体との差 {e['thr_lift']:+.2f}pt / 窓ごとの差の平均 {e['thr_fold_mean']:+.2f} ± "
                  f"{se:.2f}pt（上の窓 {e['thr_folds_won']}/{e['thr_folds']}、最悪 {e['thr_worst']:+.2f}pt）")
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OOF_DIR, "e48_auc_by_window.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(OOF_DIR, "e48_edges.csv"), index=False)
    if len(shifts) > 1 and len(s):
        print(f"\n■ 4. 切り方{len(shifts)}通りをまとめて（{len(s)}窓）")
        print(f"  {'':<18}{'O':>9}{'N':>9}{'差':>10}{'SE':>9}{'上の窓':>9}")
        print(AB.pair_line("PR-AUC", s["pr_o"].to_numpy(), s["pr_n"].to_numpy()))
        print(AB.pair_line("ROC-AUC", s["roc_o"].to_numpy(), s["roc_n"].to_numpy()))
        ed = pd.DataFrame(edges)
        for name in ("O", "N"):
            g = ed[ed["arm"] == name]
            print(f"  上位10%の {OUTCOME} の全体との差（{name}）: 切り方ごと "
                  + " / ".join(f"{v:+.2f}" for v in g["thr_lift"]) + " pt"
                  f"（平均 {g['thr_lift'].mean():+.2f}pt）")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験48: 最新の作りで LightGBM を探索・評価する")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print(f"実験48 最新の作りで LightGBM を探索・評価（{F.DEFAULT_PRESET}、{len(cols)}列、"
          f"指紋 {F.signature(cols)}）")
    print(f"  目的変数 {B.DEFAULT_RISE.name}")
    print(f"  母集団の直し GENERAL_MARKET_START={B.GENERAL_MARKET_START}"
          f" / 目的変数の基準 {B.LABEL_ENTRY}")
    print("=" * 78)

    rec = tune_latest(cols, args.n_trials)
    show_tuning(rec)
    params = {"N": tuning.params_for("e48", store={"e48": rec}),
              "O": tuning.params_for("all")}
    old = tuning.load_params().get("all", {})
    print(f"\n  比べる O: research/lgbm_params.json の all（{old.get('_n_features')}列で探索、"
          f"CV PR-AUC {old.get('_cv', {}).get('mean_pr_auc')}）。本番の列の探索結果"
          f"（{F.DEFAULT_PRESET}）はまだ無い: {F.DEFAULT_PRESET in tuning.load_params()}")

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

    oofs = {name: production_oof(df, cols, params[name], name) for name in ("O", "N")}
    show_production(oofs)
    show_windows(df, cols, params, shifts)
    log(f"記録: {OOF_DIR}/e48_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
