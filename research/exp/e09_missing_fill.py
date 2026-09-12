#!/usr/bin/env python3
"""
実験09: 欠損を何で埋めるか。

前提
----
特徴量151列の欠損率は平均15.7%。guidance 57.9%、cashflow 52.2%。
LightGBM は NaN を「どちらの枝に送るか」を列ごとに学習するので、
埋めないことにも意味がある。ただし木が使えるのは「欠損かどうか」の1ビットで、
欠損した行に何を代入すれば他の行と比較可能になるか、は木が決められない。

株式だけに絞ると、財務の欠損数と成績は単調に効いていた
（欠損1列で正例率21.7%/実収益+3.19%、41列で17.7%/+0.31%）。
開示が欠けている企業ほど成績が悪い。この情報を残したまま、
欠損行を他の行と比較可能にできるかを測る。

実験01 で欠損の集計量（欠損数・グループ別欠損率）を足したが効かなかった。
あれは「欠損の量」を渡す試み。今回は「欠損した場所に何を入れるか」。

試す埋め方
---------
  none          現行。LightGBM のネイティブ NaN 処理
  median        訓練側の中央値。素直な基準
  median_ind    中央値 + 欠損指示子。値は比較可能にしつつ、欠損の事実も残す
  sector_median 業種別の中央値。PER や ROE は業種で水準が違うので、
                全体中央値より近い値になるはず
  date_median   その日の他の候補の中央値。相場水準の影響を受ける特徴量
                （バリュエーションなど）では、同じ日の仲間が最も近い基準
  sentinel      範囲外の値(-999)。全ての欠損を片端のビンに寄せる。
                ネイティブ処理に近いが方向を固定する
  zero          0埋め。比較用の悪い例として置く

リークを防ぐ
-----------
median / sector_median は訓練側だけで決めて検証側に当てる（prep フックは
fold の中で呼ばれる）。date_median は各行が属する日の中で計算するので
訓練・検証をまたがない。予測時にはその日の候補が全部揃っているので、
運用でも同じものが作れる。
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
SENTINEL = -999.0
#: 業種別中央値の基準にする列。時点別の業種コード
SECTOR_COL = "s33_code"


def _np(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=float)


def make_prep(how: str) -> Callable:
    """(train, test, cols) -> (X_train, X_test) を返す。"""

    def prep(tr: pd.DataFrame, te: pd.DataFrame,
             cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if how == "none":
            return _np(tr, cols), _np(te, cols)

        if how == "zero":
            return np.nan_to_num(_np(tr, cols)), np.nan_to_num(_np(te, cols))

        if how == "sentinel":
            a, b = _np(tr, cols), _np(te, cols)
            return np.where(np.isnan(a), SENTINEL, a), \
                np.where(np.isnan(b), SENTINEL, b)

        if how in ("median", "median_ind"):
            med = tr[cols].median()
            A = tr[cols].fillna(med)
            B = te[cols].fillna(med)
            if how == "median_ind":
                # 欠損率の高い列だけ指示子を付ける。151列すべてに付けると
                # 列数が倍になり、ほぼ定数の列が大量に混ざる
                hi = [c for c in cols if tr[c].isna().mean() > 0.05]
                A = pd.concat([A, tr[hi].isna().astype(float)
                               .add_suffix("_na")], axis=1)
                B = pd.concat([B, te[hi].isna().astype(float)
                               .add_suffix("_na")], axis=1)
            return A.to_numpy(dtype=float), B.to_numpy(dtype=float)

        if how == "sector_median":
            # 業種別の中央値を訓練側で作る。業種が欠けている行や、
            # その業種に訓練側で値が無い列は全体中央値で埋める
            gmed = tr[cols].median()
            smed = tr.groupby(SECTOR_COL)[cols].median()

            def fill(d: pd.DataFrame) -> pd.DataFrame:
                key = d[SECTOR_COL]
                per_row = smed.reindex(key).set_axis(d.index)
                return d[cols].fillna(per_row).fillna(gmed)

            return fill(tr).to_numpy(dtype=float), fill(te).to_numpy(dtype=float)

        if how == "date_median":
            # その日の他の候補の中央値。日をまたがないので訓練・検証を汚さない
            def fill(d: pd.DataFrame) -> pd.DataFrame:
                g = d.groupby("Date")[cols].transform("median")
                return d[cols].fillna(g).fillna(tr[cols].median())

            return fill(tr).to_numpy(dtype=float), fill(te).to_numpy(dtype=float)

        raise ValueError(how)

    return prep


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    miss = df[cols].isna().mean()
    print(f"特徴量 {len(cols)}列 / 平均欠損率 {miss.mean()*100:.1f}% "
          f"/ 欠損5%超の列 {(miss > 0.05).sum()}列")
    print(f"業種列 {SECTOR_COL}: 欠損 {df[SECTOR_COL].isna().mean()*100:.1f}% "
          f"/ 種類 {df[SECTOR_COL].nunique()}")
    print()

    hows = ("none", "median", "median_ind", "sector_median", "date_median",
            "sentinel", "zero")
    results = {}
    for how in hows:
        p = os.path.join(OOF_DIR, f"fill_{how}.parquet")
        if os.path.exists(p):
            o = pd.read_parquet(p)
            results[how] = lab.Result(how, o, lab.metrics(o))
            print(f"  {how:<14} 保存済みを読む")
            continue
        r = lab.run_multi(df, lambda s: lab.lgbm(seed=s), seeds=SCREEN_SEEDS,
                          cols=cols, name=how, prep=make_prep(how))
        r.oof.to_parquet(p, index=False)
        results[how] = r
        m = r.metrics
        print(f"  {how:<14} 完了  しきい値優位 {m['thr_lift']:+.2f}pt / "
              f"窓平均 {m['thr_fold_mean']:+.2f}pt / t値 {m['thr_t']:+.2f}")

    print()
    print(lab.table(results))

    print()
    print("=== 判定（ノイズ床: しきい値優位 0.51pt）===")
    ref = results["none"].metrics
    print(f"  基準 none: しきい値優位 {ref['thr_lift']:+.2f}pt / "
          f"窓平均 {ref['thr_fold_mean']:+.2f}pt / t値 {ref['thr_t']:+.2f} / "
          f"最悪 {ref['thr_worst']:+.2f}pt")
    print()
    print(f"  {'埋め方':<15}{'優位の差':>10}{'窓平均の差':>12}{'t値':>8}"
          f"{'最悪の窓の差':>14}{'判定':>9}")
    rows = []
    for how, r in results.items():
        if how == "none":
            continue
        m = r.metrics
        rows.append((how, m["thr_lift"] - ref["thr_lift"],
                     m["thr_fold_mean"] - ref["thr_fold_mean"],
                     m["thr_t"], m["thr_worst"] - ref["thr_worst"]))
    for how, d, dm, t, dw in sorted(rows, key=lambda x: -x[1]):
        v = "採用可" if d > 0.51 else ("要確認" if d > 0.25 else "ノイズ内")
        print(f"  {how:<15}{d:>+9.2f}pt{dm:>+11.2f}pt{t:>+8.2f}{dw:>+13.2f}pt{v:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
