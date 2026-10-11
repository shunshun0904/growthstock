#!/usr/bin/env python3
"""
実験77（集計だけ。学習しない）: 目的変数の定義どおりに、価格の上昇幅の分布を母集団全体 / 正例 / 負例で出す。

運用者の問い（2026-10-11）「いまの目的変数の定義を教えてください。価格としては、購入時より平均的に何%上がっていれば
正例になるか。母集団全体の分布と正例のみ、負例のみの分布、中央値も」。

定義（research/build_dataset.py の DEFAULT_RISE。docs/MODEL_DESIGN.md §2.1）
  買値   翌営業日の寄り O[t+1]（分割調整後）
  期間   t+1 〜 t+20 営業日（約1か月）
  到達   max(終値 t+1..t+20) ÷ 買値 − 1 ≥ 1.2σ、σ = vol_20d/100 × √20（銘柄自身の20日ボラ。必要な上昇率は銘柄ごとに違う）
  終盤   t+16..t+20 の平均終値 ÷ 買値 − 1 ≥ 0.6σ（到達の半分。往って来いを外す）
  トレンド t+20 で MA5 ≥ MA20
  3つすべて満たせば正例。rise_need / end_need はデータセットに残らないので vol_20d から同じ式で引き直す
  （B.rise_thresholds）。引き直したラベルが本番のラベルと全行一致することを最初に確かめる。

出すもの（すべて行の集計。銘柄ごとの値は出さない）
  1. 必要な上昇率（1.2σ）・終盤に必要な水準（0.6σ）の分布
  2. 実際の上昇（判定期間の最高終値 ÷ 買値 − 1）・終盤の水準・余裕（実際 − 必要）の分布
  3. 実収益（翌営業日寄り買い・20 / 40営業日後の5日平均終値で売り）の分布
  4. ファネル（到達 → 終盤 → トレンド で何件落ちるか）と年ごとの正例率
  群: 母集団全体 / 正例 / 負例 / 到達したが負例（終盤かトレンドで落ちた）

    exp=e77_label_distribution.py
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
import build_dataset as B  # noqa: E402
import lab  # noqa: E402

QUANTS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
#: (列, 説明, 倍率。比率は ×100 で %)
MEASURES = (
    ("rise_need", "到達に必要な上昇率（1.2σ）", 100.0),
    ("end_need", "終盤に必要な水準（0.6σ）", 100.0),
    ("future_rise", "判定期間の最高終値 ÷ 買値 − 1", 100.0),
    ("end_level", "終盤（t+16〜20 の平均終値）÷ 買値 − 1", 100.0),
    ("margin", "余裕 = 実際の最高の上昇率 − 必要な上昇率", 100.0),
    ("ret_o1_20", "実収益 20営業日（翌営業日寄り買い・5日平均終値で売り）", 100.0),
    ("ret_o1_40", "実収益 40営業日（同上）", 100.0),
    ("entry_gap", "買値（翌営業日の寄り）÷ 基準日の終値 − 1", 100.0),
    ("vol_20d", "日次ボラ（直近20日の日次リターンの標準偏差。%）", 1.0),
)
GROUPS = (("all", "母集団全体"), ("pos", "正例"), ("neg", "負例"), ("hit_neg", "到達したが負例"))
OUT = os.path.join(lab.DATA_DIR, "oof", "e77_label_distribution.json")


def rederive(df: pd.DataFrame) -> pd.DataFrame:
    """vol_20d から必要な上昇率・終盤の水準を引き直し、ラベルを合成し直す（attach_rise_label と同じ式）。"""
    out = df.copy()
    need, end_need = B.rise_thresholds(out["vol_20d"], B.DEFAULT_RISE)
    out["rise_need"] = need.to_numpy()
    out["end_need"] = end_need.to_numpy()
    out["margin"] = out["future_rise"] - out["rise_need"]
    out["hit"] = out["future_rise"] >= out["rise_need"]
    out["end_ok"] = out["end_level"] >= out["end_need"]
    out["trend_ok"] = out["uptrend_end"] == 1.0
    out["label_re"] = (out["hit"] & out["end_ok"] & out["trend_ok"]).astype(int)
    return out


def masks(df: pd.DataFrame) -> dict:
    y = df["label"].astype(int)
    return {"all": pd.Series(True, index=df.index), "pos": y == 1, "neg": y == 0,
            "hit_neg": df["hit"] & (y == 0)}


def stats(v: pd.Series, scale: float) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna() * scale
    if not len(v):
        return {"n": 0}
    q = v.quantile(QUANTS)
    return {"n": int(len(v)), "mean": float(v.mean()), **{f"p{int(p*100)}": float(q[p]) for p in QUANTS}}


def table(df: pd.DataFrame, col: str, scale: float) -> dict:
    m = masks(df)
    return {g: stats(df.loc[m[g], col], scale) for g, _ in GROUPS}


def funnel(df: pd.DataFrame) -> dict:
    y = df["label"].astype(int)
    hit = df["hit"]
    return {"n": int(len(df)), "hit": int(hit.sum()), "hit_end": int((hit & df["end_ok"]).sum()),
            "hit_end_trend": int((hit & df["end_ok"] & df["trend_ok"]).sum()), "pos": int(y.sum()),
            "hit_fail_end": int((hit & ~df["end_ok"]).sum()),
            "hit_end_fail_trend": int((hit & df["end_ok"] & ~df["trend_ok"]).sum()),
            "mismatch": int((df["label_re"] != y).sum())}


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    y = df["label"].astype(int)
    g = df.assign(_y=df["Date"].dt.year, _lab=y).groupby("_y")
    return pd.DataFrame({"n": g.size(), "pos_rate": g["_lab"].mean(),
                         "need_median": g["rise_need"].median() * 100.0,
                         "rise_median": g["future_rise"].median() * 100.0,
                         "ret20_median": g["ret_o1_20"].median() * 100.0})


def print_table(name: str, t: dict, unit: str = "%") -> None:
    print(f"\n  {name}（{unit}）")
    print(f"  {'':<16}{'件数':>8}{'平均':>8}" + "".join(f"{'p' + str(int(p*100)):>8}" for p in QUANTS))
    for g, ja in GROUPS:
        s = t[g]
        if not s.get("n"):
            print(f"  {ja:<16}{0:>8}")
            continue
        print(f"  {ja:<16}{s['n']:>8,}{s['mean']:>8.1f}" + "".join(f"{s[f'p{int(p*100)}']:>8.1f}" for p in QUANTS))


def report(df: pd.DataFrame) -> dict:
    f = funnel(df)
    print("=" * 78)
    print(f"目的変数: {B.DEFAULT_RISE.name}")
    print(f"  行 {f['n']:,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}・{df['Code'].nunique():,}銘柄）/ "
          f"正例 {f['pos']:,}（{f['pos']/f['n']*100:.2f}%）/ 引き直したラベルと本番のラベルの不一致 {f['mismatch']}件")
    print(f"  ファネル: 到達 {f['hit']:,}（{f['hit']/f['n']*100:.1f}%）→ 終盤も満たす {f['hit_end']:,} → トレンドも満たす "
          f"{f['hit_end_trend']:,}（= 正例）。到達したが終盤で落ちた {f['hit_fail_end']:,}・トレンドで落ちた {f['hit_end_fail_trend']:,}")
    tabs = {}
    for col, ja, scale in MEASURES:
        if col not in df.columns:
            continue
        tabs[col] = table(df, col, scale)
        print_table(ja, tabs[col], "%" if scale == 100.0 else "%/日")
    yr = by_year(df)
    print("\n  年ごと（必要な上昇率・実際の最高の上昇率・実収益20 は中央値。%）")
    print(f"  {'年':<6}{'件数':>8}{'正例率':>8}{'必要':>8}{'最高':>8}{'実収益20':>10}")
    for y, r in yr.iterrows():
        print(f"  {y:<6}{int(r['n']):>8,}{r['pos_rate']*100:>7.1f}%{r['need_median']:>8.1f}{r['rise_median']:>8.1f}"
              f"{r['ret20_median']:>10.1f}")
    summary = {"definition": B.DEFAULT_RISE.name, "funnel": f, "tables": tabs,
               "by_year": {str(y): {k: float(v) for k, v in r.items()} for y, r in yr.iterrows()}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験77: 目的変数の定義と上昇幅の分布（集計だけ）")
    ap.add_argument("--rebuild", action="store_true", help="lab.frame を作り直す")
    args = ap.parse_args(argv)
    t0 = time.time()
    df = lab.frame(rebuild=args.rebuild)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["label"].notna()].reset_index(drop=True)
    need = ["vol_20d", "future_rise", "end_level", "uptrend_end", "label"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss}")
    print(f"[data] {len(df):,}行 / {time.time()-t0:.0f}秒")
    df = rederive(df)
    report(df)
    print(f"\n記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
