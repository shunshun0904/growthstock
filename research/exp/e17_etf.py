#!/usr/bin/env python3
"""
実験17: ETF・ETN・REIT を母集団から外すと、運用成績は良くなるか。

なぜ測り直すか
------------
2026-09-11 に一度測って docs/OPERATIONS.md に「外すコストは -0.15pt で
ほぼゼロ」と記録した。しかしその測り方は、その後に確立した手法より甘い。

  当時                          いま
  全期間プールの上位5%          窓ごとに評価（プールは44%が1窓に偏る歪みが出る）
  しきい値は全期間分布から       過去の窓だけから決める（全期間だと約0.9pt楽観）
  ref_end（終値で買う・60日）   ret_o1_40（翌営業日の寄りで買う・40日）
  単発                          種3個の確率平均

同じ結論になるとは限らないので、いまの物差しで測る。

何と何を比べるか
--------------
評価する母集団が違うと「上位10% − 同日候補全体」の基準線が変わるので、
ETF込みと株式のみを直接並べても比較にならない。切り分けて3つ測る。

  A  学習=全部  評価=全部    現行
  A' 学習=全部  評価=株式のみ 画面から ETF を落とすだけ（再学習しない）
  B  学習=株式  評価=株式のみ 母集団から外して学習し直す

A と A' の差 = 「ETF を候補から外す」ことの効果。
A' と B の差 = 「ETF を学習データから外す」ことの効果。**同じ評価母集団なので
この2つは直接比べられる。**

A' は A の out-of-fold を株式だけに絞り直して測る。同じモデルの採点なので
学習し直す必要がない（しきい値も絞った後の過去の窓から決め直す）。

判定の足切り
-----------
実験11 のノイズ床（同一設定・種5個）
  しきい値優位  レンジ 0.484pt
  窓平均        レンジ 0.143pt  <- 最も解像度が高い
  t値           レンジ 3.294    <- 9窓では使えない
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SEEDS = (42, 7, 123)
ETF_MKT_CODE = 109          # 上場投信・ETN・REIT 等（J-Quants の市場区分）
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")


def with_mkt(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """out-of-fold に市場区分を戻す（絞り込みに要る）。"""
    key = df[["Code", "Date", "mkt_code"]].drop_duplicates(["Code", "Date"])
    return oof.merge(key, on=["Code", "Date"], how="left")


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    etf = df["mkt_code"] == ETF_MKT_CODE
    os.makedirs(OOF_DIR, exist_ok=True)

    print(f"母集団 {len(df):,}件 / ETF等 {etf.sum():,}件 ({etf.mean()*100:.1f}%)")
    print(f"目的変数 {lab.OUTCOME}（翌営業日の寄り買い・40営業日後の5日平均終値売り）")
    print(f"種 {SEEDS} の確率平均")
    print()

    runs = {}
    for name, sub in (("all", df), ("stock", df[~etf])):
        p = os.path.join(OOF_DIR, f"etf_{name}.parquet")
        if os.path.exists(p):
            runs[name] = lab.attach_outcomes(pd.read_parquet(p), df)
            print(f"  学習={name:<6} 保存済みを読む")
            continue
        r = lab.run_multi(df if name == "all" else sub,
                          lambda s: lab.lgbm(seed=s), seeds=SEEDS, cols=cols,
                          name=name)
        r.oof.to_parquet(p, index=False)
        runs[name] = r.oof
        print(f"  学習={name:<6} 完了 out-of-fold {len(r.oof):,}件")

    oof_all = with_mkt(runs["all"], df)
    oof_stock = with_mkt(runs["stock"], df)

    results = {
        "A  学習=全部/評価=全部": lab.Result("A", oof_all, lab.metrics(oof_all)),
        "A' 学習=全部/評価=株式": None,
        "B  学習=株式/評価=株式": lab.Result("B", oof_stock, lab.metrics(oof_stock)),
    }
    a2 = oof_all[oof_all["mkt_code"] != ETF_MKT_CODE].reset_index(drop=True)
    results["A' 学習=全部/評価=株式"] = lab.Result("A2", a2, lab.metrics(a2))

    print()
    print(lab.table(results))

    print()
    print("=== 判定（窓平均。足切りは窓SD/√窓数）===")
    print(f"  {'条件':<24}{'窓平均':>10}{'標準誤差':>10}{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
    for name, r in results.items():
        m = r.metrics
        se = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
        print(f"  {name:<24}{m['thr_fold_mean']:>+9.2f}pt{se:>10.2f}"
              f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}{m['thr_worst']:>+10.2f}pt"
              f"{m['thr_n']:>9,}")

    print()
    print("=== 差の検定（同じ評価母集団どうしだけ意味がある）===")
    pairs = [("A' 学習=全部/評価=株式", "B  学習=株式/評価=株式",
              "ETF を学習から外す効果（評価母集団は同じ）"),
             ("A  学習=全部/評価=全部", "A' 学習=全部/評価=株式",
              "ETF を候補から外す効果（※評価母集団が違うので参考値）")]
    for a, b, note in pairs:
        ma, mb = results[a].metrics, results[b].metrics
        sa = ma["thr_fold_sd"] / np.sqrt(max(1, ma["thr_folds"]))
        sb = mb["thr_fold_sd"] / np.sqrt(max(1, mb["thr_folds"]))
        d = mb["thr_fold_mean"] - ma["thr_fold_mean"]
        z = d / np.sqrt(sa ** 2 + sb ** 2)
        v = "採用可" if z > 2 else ("要確認" if z > 1 else "差なし")
        print(f"  {note}")
        print(f"    差 {d:+.2f}pt / 合成SE {np.sqrt(sa**2+sb**2):.2f} / z {z:+.2f} → {v}")

    # 上位10%に ETF がどれだけ入るか（現行運用で実際に何を買わされるか）
    print()
    print("=== 現行（A）の上位10%に ETF がどれだけ混ざるか ===")
    e = lab.threshold_edge(oof_all)
    sel = e.get("selected")
    if sel is not None and len(sel):
        s = with_mkt(sel[["Code", "Date"]], df) if "mkt_code" not in sel else sel
        share = (s["mkt_code"] == ETF_MKT_CODE).mean() * 100
        print(f"  上位10% {len(s):,}件のうち ETF等 {share:.1f}%（母集団では {etf.mean()*100:.1f}%）")
    else:
        top = oof_all[oof_all["score"] >= oof_all["score"].quantile(0.90)]
        share = (top["mkt_code"] == ETF_MKT_CODE).mean() * 100
        print(f"  スコア上位10% {len(top):,}件のうち ETF等 {share:.1f}%"
              f"（母集団では {etf.mean()*100:.1f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
