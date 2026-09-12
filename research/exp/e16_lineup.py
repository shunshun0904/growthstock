#!/usr/bin/env python3
"""
実験16: 探索済みの5モデルを運用指標で並べ、画面に並べる価値を測る。

なぜ内側検証の PR-AUC では足りないか
----------------------------------
探索（実験15）が出したのは 5分割の内側検証 PR-AUC。

  lgbm 0.4029 / xgb 0.3979 / cat 0.3831 / mlp 0.3044 / logit 0.2897

これは「探索がその学習器の中で選べた最良」であって、運用の成績ではない。
運用は上位だけを買う。全体の順位品質より、上位10%の実収益が効く。
ここでは10窓のウォークフォワードで out-of-fold を作り直し、
運用指標（しきい値運用・翌営業日の寄り買い・40営業日後の5日平均終値売り）
で並べる。

測るもの
-------
1. 運用指標。しきい値は窓ごとに**過去の窓だけ**から決める（全期間の分布で
   決めると約0.9pt 楽観的になる）。
2. モデル間のスコア相関と上位10%の重複。**並べる価値はここで決まる。**
   相関が高いモデルは画面で2本ぶんの場所を取るだけで、情報が増えない。

判定の足切り（実験11 のノイズ床・同一設定で種5個）
  しきい値優位  レンジ 0.484pt
  窓平均        レンジ 0.143pt  <- 最も解像度が高い
  t値           レンジ 3.294    <- 9窓では使えない
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import tuning_multi as TM  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")


def seeded(algo: str, cols, seed: int):
    """
    本番と同じ定義（tuning_multi.build）で学習器を組み、種だけ差し替える。

    ここで lab.lgbm などの実験用ファクトリを使わないのが肝。画面に出るのは
    models.fit が組むモデルで、それは TM.build を呼ぶ。測る対象と出す対象を
    同じコードにしておかないと、測った成績が画面のモデルの成績にならない。
    """
    def fit(X, y):
        m = TM.build(algo, TM.params_for(algo), y, list(cols))
        # Pipeline の中の推定器まで届くよう、deep な名前を総当たりで見る
        over = {k: seed for k in m.get_params(deep=True)
                if k.endswith("random_state") or k.endswith("random_seed")}
        if over:
            m.set_params(**over)
        m.fit(X, y)
        return lambda Z: np.asarray(m.predict_proba(Z), dtype=float)[:, 1]
    return fit


def overlap(a: pd.DataFrame, b: pd.DataFrame, pct: float = 90) -> tuple:
    """
    2モデルのスコア相関と、上位10%に選ぶ銘柄の重複率。

    同じ (Code, Date) の行で突き合わせる。out-of-fold は同じ分割で
    作っているので行は一致するが、念のため結合して確かめる。
    """
    j = a[["Code", "Date", "score"]].merge(
        b[["Code", "Date", "score"]], on=["Code", "Date"], suffixes=("_a", "_b"))
    if j.empty:
        return float("nan"), float("nan"), 0
    rho = j["score_a"].corr(j["score_b"], method="spearman")
    ta = j["score_a"] >= np.percentile(j["score_a"], pct)
    tb = j["score_b"] >= np.percentile(j["score_b"], pct)
    inter = int((ta & tb).sum())
    union = int((ta | tb).sum())
    return float(rho), (inter / union if union else float("nan")), len(j)


def main() -> int:
    algos = sys.argv[1:] or list(M.ALGOS)
    df = lab.frame()
    cols = F.columns("all")
    os.makedirs(OOF_DIR, exist_ok=True)

    store = TM.load()
    miss = [a for a in algos if a not in store]
    if miss:
        print(f"[warn] 探索済みパラメータが無い: {miss}（既定値で走る）")
    print(f"対象 {algos} / 特徴量 {len(cols)}列 / 種 {SCREEN_SEEDS}")
    print(f"目的変数 {lab.OUTCOME}（翌営業日の寄り買い・40営業日後の5日平均終値売り）")
    print()

    results = {}
    for algo in algos:
        p = os.path.join(OOF_DIR, f"lineup_{algo}.parquet")
        if os.path.exists(p):
            o = lab.attach_outcomes(pd.read_parquet(p), df)
            results[algo] = lab.Result(algo, o, lab.metrics(o))
            print(f"  {algo:<7} 保存済みを読む")
            continue
        t0 = time.time()
        r = lab.run_multi(df, lambda s, a=algo: seeded(a, cols, s),
                          seeds=SCREEN_SEEDS, cols=cols, name=algo)
        r.oof.to_parquet(p, index=False)
        results[algo] = r
        m = r.metrics
        print(f"  {algo:<7} {time.time()-t0:>5.0f}秒  優位 {m['thr_lift']:+.2f}pt / "
              f"窓平均 {m['thr_fold_mean']:+.2f}pt (SD {m['thr_fold_sd']:.2f}) / "
              f"勝ち窓 {m['thr_folds_won']}/{m['thr_folds']}")

    print()
    print(lab.table(results))

    print()
    print("=== 運用指標（窓平均で見る。足切りは窓SD/√窓数）===")
    print(f"  {'モデル':<8}{'窓平均':>10}{'標準誤差':>10}{'勝ち窓':>8}"
          f"{'最悪の窓':>10}{'探索PR-AUC':>12}")
    for algo, r in results.items():
        m = r.metrics
        se = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
        cv = store.get(algo, {}).get("_cv", {})
        pr = cv.get("mean_pr_auc")
        print(f"  {M.JA.get(algo, algo)[:7]:<8}{m['thr_fold_mean']:>+9.2f}pt"
              f"{se:>10.2f}{m['thr_folds_won']:>5}/{m['thr_folds']:<2}"
              f"{m['thr_worst']:>+9.2f}pt"
              f"{(f'{pr:.4f}' if pr else '-'):>12}")

    print()
    print("=== 並べる価値（スコア相関と上位10%の重複）===")
    print("  相関が高い組は画面で2本ぶんの場所を取るだけで、情報が増えない")
    print()
    names = list(results)
    print(f"  {'':<8}" + "".join(f"{M.JA.get(a, a)[:6]:>9}" for a in names))
    for i, a in enumerate(names):
        row = f"  {M.JA.get(a, a)[:7]:<8}"
        for j, b in enumerate(names):
            if i == j:
                row += f"{'—':>9}"
            elif j < i:
                row += f"{'':>9}"
            else:
                rho, _, _ = overlap(results[a].oof, results[b].oof)
                row += f"{rho:>9.3f}"
        print(row)
    print()
    print("  上位10%の重複（Jaccard）")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho, jac, n = overlap(results[a].oof, results[b].oof)
            print(f"    {M.JA.get(a,a)[:7]:<9} × {M.JA.get(b,b)[:7]:<9} "
                  f"相関 {rho:>6.3f} / 重複 {jac*100:>5.1f}% / {n:,}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
