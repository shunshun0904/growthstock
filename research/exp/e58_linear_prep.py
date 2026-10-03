#!/usr/bin/env python3
"""
実験58: ロジスティック回帰と MLP の前処理 v2 を、いまの前処理 v1 と同じ手順で比べる。

運用者の指示（2026-10-03）「特徴量の前処理が GBDT と比べると慎重にならないといけないので、
全特徴量に関して精査して」→ 精査（docs/MODEL_LINEAR_PREPROCESSING.md）→「1と2を進めてください」。

v1  tuning_multi.preprocess: 中央値補完 + 欠損指示子 + 標準化 + one-hot（本番）
v2  linear_preprocess.preprocess_v2: 列の型ごとに 分位点で切る → asinh / log1p → 標準化、
    欠損は意味で 0 / 上限 / 最頻値 / 中央値（指示子は残す）、one-hot は同じ

腕（探索は本番の週次実行と同じ関数・同じ条件: tuning_multi.tune、Optuna 50試行、5分割
year_cap_date、ホールドアウトより前で打ち切り、種0。列は本番の239列）
  P  v1 で探索    本番の週次実行がやること
  V  v2 で探索

評価（実験43・57 と同じ作り。木は触らない）
1. 探索の CV（PR-AUC 平均±SD、ROC-AUC）と所要時間
2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）:
   PR-AUC / リフト / ROC-AUC / 日内 AUC / 上位10% の ret_o1_20 の超過（e25_auc_noise.metrics）
3. 窓ごと: 窓の境界を 0/2/4か月ずらした3通り（計32窓）。logit は種1つ（決定的）、mlp は種3つの平均。
   V−P の平均・SE・V が上の窓の数。採否は §7 に準じ「OOF と窓の過半で上回る」

本番の設定（research/multi_params.json、tuning_multi.PREPROCESS / PREPROCESS_BY_ALGO）には書かない。
記録は research/_data/oof/e58_*（探索の study は e58_optuna_{P,V}.db。途中で止まっても引き継ぐ）。

使い方
    python3 research/exp/e58_linear_prep.py                     # 探索して評価
    python3 research/exp/e58_linear_prep.py --n-trials 2 --shifts 0 --algos logit   # 試運転
"""
from __future__ import annotations

import argparse
import contextlib
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
import linear_preprocess as LP  # noqa: E402
import train_model as T  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
ARMS = ("P", "V")
PREP = {"P": "v1", "V": "v2"}
LABELS = {"P": "P v1（いまの前処理）", "V": "V v2（型ごとの変換）"}
ALGOS = ("logit", "mlp")
#: 窓ごとの種。logit は決定的（lbfgs / liblinear）なので1つ、mlp は初期値で変わるので3つ
SEEDS = {"logit": (42,), "mlp": E27.SEEDS3}
PROD_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextlib.contextmanager
def prep(version: str):
    """
    tuning_multi.build が組む前処理の版を一時的に切り替える（戻し忘れを防ぐ）。
    モデルごとの設定（PREPROCESS_BY_ALGO。本番は logit だけ v2）も外して、腕の版を両モデルに強制する。
    """
    old = TM.PREPROCESS
    old_by = TM.PREPROCESS_BY_ALGO
    TM.PREPROCESS = version
    TM.PREPROCESS_BY_ALGO = {}
    try:
        yield
    finally:
        TM.PREPROCESS = old
        TM.PREPROCESS_BY_ALGO = old_by


def width(cols: list, version: str, X: np.ndarray) -> int:
    """前処理後の入力の列数（one-hot と指示子を含む）。"""
    with prep(version):
        ct = TM._preprocess(cols)
    return int(ct.fit_transform(X).shape[1])


def tune_arm(algo: str, arm: str, tune_df: pd.DataFrame, cols: list, n_trials: int) -> dict:
    """探索する。結果は残して、2回目以降は読むだけ。study は腕ごとの DB に残す（途中再開）。"""
    path = os.path.join(OOF_DIR, f"e58_params_{algo}_{arm}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("_cv", {}).get("n_trials") == n_trials:
            log(f"  [{algo} {arm}] 探索済みを読む（CV PR-AUC {rec['_cv'].get('mean_pr_auc')}）")
            return rec
    t0 = time.time()
    TM.STUDY_DB = os.path.join(OOF_DIR, f"e58_optuna_{arm}.db")
    with prep(PREP[arm]):
        rec = TM.tune(algo, tune_df, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False)
    rec["_cv"]["seconds"] = round(time.time() - t0)
    rec["_cv"]["preprocess"] = PREP[arm]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{algo} {arm}] 探索 {time.time()-t0:.0f}秒 / CV PR-AUC {rec['_cv']['mean_pr_auc']:.4f} "
        f"(±{rec['_cv']['std']:.4f}) / ROC {rec['_cv']['mean_roc_auc']:.4f} / {rec['params']}")
    return rec


def oof_arm(algo: str, arm: str, df: pd.DataFrame, cols: list, params: dict,
            shift: int, seeds) -> pd.DataFrame:
    """腕・ずらし・種ごとの out-of-fold（種の平均）。保存済みなら読む。"""
    folds = E41.folds_for(df["Date"], shift)
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"e58_{algo}_{arm}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        with prep(PREP[arm]):
            o = E41.oof_folds(algo, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"    {algo} {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def pair_line(name: str, a: np.ndarray, b: np.ndarray, fmt: str = "{:.4f}") -> str:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return (f"  {name:<18}{fmt.format(a.mean()):>9}{fmt.format(b.mean()):>9}"
            f"{('+' if d.mean() >= 0 else '') + fmt.format(d.mean()):>10}{fmt.format(se):>9}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験58: 線形・MLP の前処理 v2 を v1 と比べる")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--algos", default=",".join(ALGOS))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    algos = [a for a in args.algos.split(",") if a]
    os.makedirs(OOF_DIR, exist_ok=True)

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print("実験58 ロジスティック回帰・MLP の前処理 v2 を、いまの前処理 v1 と同じ手順で比べる")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 指紋 {F.signature(cols)} / 型の表に無い列: "
          f"{LP.unclassified(cols) or 'なし'}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（year_cap_date）/ 窓のずらし {shifts}か月 / "
          f"種 {SEEDS}")
    print("=" * 78)

    raw = (pd.read_parquet(os.path.join(lab.DATA_DIR, "dataset.parquet"))
           .sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True))
    miss = [c for c in cols if c not in raw.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:8]}")
    dates = pd.to_datetime(raw["Date"])
    cutoff, _, _ = T.holdout_bounds(dates, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    tune_df = raw[dates <= cutoff]
    log(f"[tune] 全体 {len(raw):,}件 / 探索に使う期間 〜{cutoff.date()}（{len(tune_df):,}件 / "
        f"正例率 {tune_df['label'].mean()*100:.2f}%）")
    X = tune_df[cols].to_numpy(dtype=float)
    for arm in ARMS:
        log(f"  前処理後の入力の列数 {LABELS[arm]}: {width(cols, PREP[arm], X)}")
    del X

    recs = {}
    for algo in algos:
        for arm in ARMS:
            recs[(algo, arm)] = tune_arm(algo, arm, tune_df, cols, args.n_trials)
    del raw, tune_df

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<10}{'腕':<4}{'PR-AUC':>9}{'±SD':>8}{'ROC':>9}{'秒':>7}  パラメータ")
    for (algo, arm), rec in recs.items():
        cv = rec["_cv"]
        print(f"  {algo:<10}{arm:<4}{cv['mean_pr_auc']:>9.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>9.4f}"
              f"{cv.get('seconds', 0):>7}  {rec['params']}")

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log(f"データ {len(df):,}件 / 正例率 {df['label'].mean()*100:.2f}% / "
        f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}")

    summary = {"tuning": {f"{a}_{arm}": r["_cv"] | {"params": r["params"]} for (a, arm), r in recs.items()},
               "production_oof": {}}
    print("\n■ 2. 本番と同じ作りの out-of-fold（ずらし0・種42）")
    print(f"  {'':<10}{'腕':<4}{'PR-AUC':>9}{'リフト':>7}{'ROC':>9}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}")
    prod = {}
    for algo in algos:
        for arm in ARMS:
            o = oof_arm(algo, arm, df, cols, recs[(algo, arm)]["params"], 0, (PROD_SEED,))
            prod[(algo, arm)] = o
            m = metrics(o)
            summary["production_oof"][f"{algo}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {algo:<10}{arm:<4}{m['pr_auc']:>9.4f}{m['lift']:>6.2f}x{m['roc_auc']:>9.4f}"
                  f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                  f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt")
        a, b = metrics(prod[(algo, "P")]), metrics(prod[(algo, "V")])
        print(f"  {algo:<10}V−P {b['pr_auc']-a['pr_auc']:>+9.4f}{'':>7}{b['roc_auc']-a['roc_auc']:>+9.4f}"
              f"{b['day_auc']-a['day_auc']:>+8.4f}")

    rows = []
    for sh in shifts:
        print(f"\n■ 3. 分離力（窓ごと。ずらし{sh}か月）")
        print(f"  {'':<18}{'P v1':>9}{'V v2':>9}{'V−P':>10}{'SE':>9}{'Vが上':>9}")
        for algo in algos:
            res = {arm: oof_arm(algo, arm, df, cols, recs[(algo, arm)]["params"], sh, SEEDS[algo])
                   for arm in ARMS}
            wa, wb = AB.auc_by_window(res["P"]), AB.auc_by_window(res["V"])
            m = wa.merge(wb, on="fold", suffixes=("_p", "_v"))
            print(pair_line(f"{algo} PR-AUC", m["pr_p"].to_numpy(), m["pr_v"].to_numpy()))
            print(pair_line(f"{algo} ROC-AUC", m["roc_p"].to_numpy(), m["roc_v"].to_numpy()))
            for _, r in m.iterrows():
                rows.append({"shift": sh, "algo": algo, "fold": int(r["fold"]),
                             "pr_p": r["pr_p"], "pr_v": r["pr_v"], "roc_p": r["roc_p"], "roc_v": r["roc_v"]})
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e58_auc_by_window.csv"), index=False)

    if rows:
        s = pd.DataFrame(rows)
        print("\n■ 4. 切り方3通りをまとめた窓ごとの V−P")
        print(f"  {'':<16}{'窓の数':>7}{'V−P の平均':>12}{'SE':>9}{'Vが上':>9}")
        summary["windows"] = {}
        for algo, g in s.groupby("algo"):
            for met in ("pr", "roc"):
                d = (g[f"{met}_v"] - g[f"{met}_p"]).to_numpy()
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                print(f"  {algo + ' ' + met.upper():<16}{len(d):>7}{d.mean():>+12.4f}{se:>9.4f}"
                      f"{int((d > 0).sum()):>5}/{len(d)}")
                summary["windows"][f"{algo}_{met}"] = {"n": int(len(d)), "mean_diff": float(d.mean()),
                                                      "se": float(se), "wins": int((d > 0).sum())}
    with open(os.path.join(OOF_DIR, "e58_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OOF_DIR}/e58_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
