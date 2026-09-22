#!/usr/bin/env python3
"""
実験18c: 5モデル全部を新ラベルで探索し直したら、どう並ぶか。

なぜ測るか
--------
実験18 / 18b は lgbm だけだった。画面に並べている5モデル
（lgbm / xgb / cat / logit / mlp）のパラメータは `multi_params.json` の
`train_to=2025-03-21` のもので、**旧ラベル（h=60 / MA20>=MA60）に対して、
しかも ETF を含む母集団で**探索されている。

日曜の週次実行では5モデルすべてを新ラベルで探索し直す。そこで初めて
壊れるのを見るより、先に同じことをやって確かめておく。過去に mlp が
探索済みパラメータをそのまま渡せず落ちたことがある（`h1` が
`hidden_layer_sizes` に畳まれていなかった）。ああいう類を先に出す。

測り方
-----
母集団と目的変数は**本番そのまま**（`research/_data/dataset.parquet`）。
条件を並べる実験ではないので、ラベルは1つだけ。

  探索  tuning_multi.tune（本番の5モデル探索と同じ。年×時価総額帯で層別・
        日付単位で分割）。ホールドアウトより手前だけを渡す
  採点  train_production と同じ窓（36ヶ月 / 6ヶ月 / 6ヶ月・エンバーゴは
        ラベル確定に要る営業日数）。窓の定数は train_production から
        取り込むので、本番とずれない
  指標  PR-AUC / ROC-AUC / 日内AUC と、実収益（ret_o1_20 / ret_o1_40）

種平均はしない。本番の5モデル（train_multi）も単一種なので、そこに揃える。

**本番の multi_params.json には書かない。** 実験が本番の設定を書き換えると、
日曜の週次実行が何で走ったのか分からなくなる。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import train_model as T  # noqa: E402
import tuning_multi as TM  # noqa: E402
from e18_horizon import OUTCOMES, edge  # noqa: E402
from train_production import (  # noqa: E402
    OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS, OUTCOME_COL)

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "mt_params.json")
N_TRIALS = 50


def tune_all(ds: pd.DataFrame, cols: list) -> dict:
    """5モデルを新ラベルで探索する。結果は実験用のファイルに貯める。"""
    store = TM.load(PARAMS_PATH)
    d = pd.to_datetime(ds["Date"])
    train_end, _, _ = T.holdout_bounds(d)
    tune_df = ds[d <= train_end]
    print(f"[tune] 探索に使う期間: 〜{train_end.date()} ({len(tune_df):,}件 / "
          f"正例率 {tune_df['label'].mean()*100:.2f}%)")
    for algo in M.ALGOS:
        if algo in store:
            cv = store[algo].get("_cv", {})
            print(f"  [{algo}] 探索済みを読む "
                  f"(CV PR-AUC {cv.get('mean_pr_auc')})")
            continue
        t0 = time.time()
        # TM.tune は {"params": {...}, "_cv": {...}} を返す。本番の
        # multi_params.json と同じ形なので、そのまま store に入れれば
        # TM.params_for(algo, store) で取り出せる
        store[algo] = TM.tune(algo, tune_df, cols, n_trials=N_TRIALS,
                              verbose=False)
        TM.save(store, PARAMS_PATH)   # 途中で落ちても失わない
        cv = store[algo]["_cv"]
        print(f"  [{algo}] {(time.time()-t0)/60:.1f}分 / "
              f"CV PR-AUC {cv.get('mean_pr_auc')} / "
              f"ROC {cv.get('mean_roc_auc')} / "
              f"正例率 {cv.get('base_rate')}")
    return store


def oof_for(algo: str, ds: pd.DataFrame, cols: list, params: dict
            ) -> pd.DataFrame:
    """
    本番と同じ窓で out-of-fold を作る。違いは params を明示で渡すことだけ。

    窓の定数は train_production から取り込んでいる。ここに数字を書くと
    本番とずれたときに気づけない。
    """
    import walkforward as WF

    folds = WF.make_folds(pd.to_datetime(ds["Date"]),
                          min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(ds["Date"])
    parts = []
    for f in folds:
        tr = ds[(d <= pd.Timestamp(f.train_end)) & ds["label"].notna()]
        te = ds[(d >= pd.Timestamp(f.test_start))
                & (d <= pd.Timestamp(f.test_end)) & ds["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        m = M.fit(algo, tr[cols].to_numpy(dtype=float),
                  tr["label"].to_numpy(dtype=int), cols, params=params)
        keep = ["Code", "Date", "label"] + [
            c for c in ("ref_end",) + tuple(OUTCOMES) if c in te.columns]
        part = te[keep].copy()
        part["score"] = M.predict(m, te[cols].to_numpy(dtype=float))
        part["fold"] = f.index
        parts.append(part)
    if not parts:
        raise SystemExit(f"{algo}: out-of-fold を作れません")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score

    ds = lab.frame()
    cols = [c for c in F.columns("all") if c in ds.columns]
    os.makedirs(OOF_DIR, exist_ok=True)

    print(f"母集団 {len(ds):,}件 / 正例率 {ds['label'].mean()*100:.2f}%")
    print(f"目的変数 {B.DEFAULT_RISE.name}")
    print(f"エンバーゴ {B.RISE_HORIZON}営業日 / 特徴量 {len(cols)}列")
    print(f"物差し {OUTCOMES}（{OUTCOME_COL} が本番の既定）\n")

    store = tune_all(ds, cols)

    print("\n=== 探索したパラメータで out-of-fold を作る ===")
    runs, failed = {}, {}
    for algo in M.ALGOS:
        p = os.path.join(OOF_DIR, f"mt_{algo}.parquet")
        if os.path.exists(p):
            runs[algo] = pd.read_parquet(p)
            print(f"  [{algo}] 保存済みを読む ({len(runs[algo]):,}件)")
            continue
        pp = TM.params_for(algo, store)
        t0 = time.time()
        try:
            o = oof_for(algo, ds, cols, pp)
        except Exception as exc:          # noqa: BLE001
            # 1モデル落ちても残りは測る。どれがなぜ落ちたかを最後に出す。
            # 黙って飛ばすと「5モデル並んでいる」つもりで4モデルを見てしまう
            failed[algo] = f"{type(exc).__name__}: {exc}"
            print(f"  [{algo}] 失敗: {failed[algo]}")
            continue
        o.to_parquet(p, index=False)
        runs[algo] = o
        print(f"  [{algo}] {len(o):,}件 / {time.time()-t0:.0f}秒")

    print("\n=== 分離力（out-of-fold）===")
    print(f"  {'モデル':<8}{'件数':>8}{'正例率':>9}{'PR-AUC':>9}"
          f"{'PR/正例率':>11}{'ROC-AUC':>10}{'日内AUC':>10}")
    for algo, o in runs.items():
        y = o["label"].to_numpy(dtype=int)
        s = o["score"].to_numpy(dtype=float)
        br = y.mean()
        pr = average_precision_score(y, s)
        print(f"  {M.SHORT.get(algo, algo):<8}{len(o):>8,}{br*100:>8.2f}%"
              f"{pr:>9.4f}{pr/br:>10.2f}x{roc_auc_score(y, s):>10.4f}"
              f"{lab.auc_in_day(o):>10.4f}")

    for outcome in OUTCOMES:
        print(f"\n=== 実収益 {outcome} ===")
        print(f"  {'モデル':<8}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
        for algo, o in runs.items():
            m = edge(o, outcome)
            print(f"  {M.SHORT.get(algo, algo):<8}"
                  f"{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt{m['thr_n']:>9,}")

    # 画面は5モデルを並べて出すので、似すぎていないかも見る。
    # 相関が高いモデルを並べても、同じことを2回言っているだけになる。
    base = M.BASELINE
    if base in runs:
        print(f"\n=== 基準モデル({M.SHORT.get(base)})との重なり ===")
        b = runs[base][["Code", "Date", "score"]].rename(
            columns={"score": "s_base"})
        print(f"  {'モデル':<8}{'順位相関':>10}{'上位10%の一致':>16}")
        for algo, o in runs.items():
            if algo == base:
                continue
            j = o[["Code", "Date", "score"]].merge(b, on=["Code", "Date"])
            rho = j["score"].corr(j["s_base"], method="spearman")
            t1 = set(zip(j.loc[j["score"] >= j["score"].quantile(.9), "Code"],
                         j.loc[j["score"] >= j["score"].quantile(.9), "Date"]))
            t2 = set(zip(j.loc[j["s_base"] >= j["s_base"].quantile(.9), "Code"],
                         j.loc[j["s_base"] >= j["s_base"].quantile(.9), "Date"]))
            ov = len(t1 & t2) / max(1, len(t2)) * 100
            print(f"  {M.SHORT.get(algo, algo):<8}{rho:>10.3f}{ov:>15.1f}%")

    if failed:
        print("\n=== 落ちたモデル（日曜の本番でも落ちる）===")
        for algo, why in failed.items():
            print(f"  {algo}: {why}")
        return 1
    print("\n5モデルすべて通った。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
