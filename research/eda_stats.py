#!/usr/bin/env python3
"""
特徴量の EDA 用サマリを出す。

学習の前に、欠損・異常値・分布・時系列の安定性を確認する。
描画はここではせず、集計結果を JSON にする。
可視化は手元で行うため（データセットは CI 側にしか無い）。

見るもの:
  1. 欠損率。全体と年別。年別に見るのは、途中から取れるようになった項目や
     開示制度の変更で欠測が偏るのを見つけるため
  2. 分位点と外れ値。1%/99% の外側がどれだけあるか
  3. ヒストグラム。ビンごとの件数（描画側で棒グラフにする）
  4. 正例率。全体・年別・時価総額帯別
  5. 相関。冗長な列（slope は chg の定数倍など）を洗い出す
  6. 定数列・単一値列。学習に寄与しないので検出する
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from train_model import DATA_DIR  # noqa: E402

N_BINS = 40
# 相関がこれを超える組を「冗長かもしれない」として報告する
CORR_WARN = 0.95


def histogram(v: pd.Series, bins: int = N_BINS) -> Optional[Dict]:
    """
    ビンごとの件数。外れ値で潰れないよう 1%〜99% で範囲を切る。
    範囲外の件数は別に持ち、描画側で「範囲外 N件」と出せるようにする。
    """
    x = pd.to_numeric(v, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    lo, hi = float(x.quantile(0.01)), float(x.quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return {"lo": lo, "hi": hi, "counts": [int(len(x))], "edges": [lo, hi],
                    "below": 0, "above": 0, "constant": True}
    counts, edges = np.histogram(x.clip(lo, hi), bins=bins, range=(lo, hi))
    return {
        "lo": lo, "hi": hi,
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "below": int((x < lo).sum()), "above": int((x > hi).sum()),
        "constant": False,
    }


def describe_column(v: pd.Series) -> Dict:
    x = pd.to_numeric(v, errors="coerce")
    finite = x[np.isfinite(x)]
    out = {
        "n": int(len(x)),
        "missing_pct": round(float(x.isna().mean() * 100), 2),
        "n_inf": int(np.isinf(x.to_numpy(dtype=float)).sum()),
        "n_unique": int(finite.nunique()),
        "n_zero": int((finite == 0).sum()),
    }
    if len(finite):
        q = finite.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        out.update({
            "min": float(q.iloc[0]), "p1": float(q.iloc[1]), "p5": float(q.iloc[2]),
            "p25": float(q.iloc[3]), "median": float(q.iloc[4]),
            "p75": float(q.iloc[5]), "p95": float(q.iloc[6]),
            "p99": float(q.iloc[7]), "max": float(q.iloc[8]),
            "mean": float(finite.mean()), "std": float(finite.std()),
        })
        # 外れ値: 四分位範囲の3倍を超えるもの（箱ひげの外側より広め）
        iqr = out["p75"] - out["p25"]
        if iqr > 0:
            lo, hi = out["p25"] - 3 * iqr, out["p75"] + 3 * iqr
            out["n_outlier"] = int(((finite < lo) | (finite > hi)).sum())
            out["outlier_pct"] = round(out["n_outlier"] / len(finite) * 100, 2)
        else:
            out["n_outlier"] = 0
            out["outlier_pct"] = 0.0
    return out


def missing_by_year(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """年ごとの欠損率。途中から取れるようになった項目を見つける。"""
    year = pd.to_datetime(df["Date"]).dt.year
    out: Dict[str, Dict[str, float]] = {}
    for c in cols:
        g = df[c].isna().groupby(year).mean() * 100
        out[c] = {str(int(k)): round(float(v), 1) for k, v in g.items()}
    return out


def label_stats(df: pd.DataFrame) -> Dict:
    y = df["label"].astype(float)
    date = pd.to_datetime(df["Date"])
    by_year = (y.groupby(date.dt.year).agg(["mean", "size"]))
    out = {
        "overall_rate": round(float(y.mean() * 100), 2),
        "n_positive": int(y.sum()), "n": int(len(y)),
        "by_year": {str(int(k)): {"rate": round(float(r["mean"] * 100), 2),
                                  "n": int(r["size"])}
                    for k, r in by_year.iterrows()},
    }
    if "market_cap" in df.columns:
        mc = pd.to_numeric(df["market_cap"], errors="coerce")
        bands = pd.cut(mc, [0, 100, 300, 1000, 3000, np.inf],
                       labels=["〜100億", "100〜300億", "300〜1000億",
                               "1000〜3000億", "3000億〜"])
        g = y.groupby(bands, observed=True).agg(["mean", "size"])
        out["by_market_cap"] = {str(k): {"rate": round(float(r["mean"] * 100), 2),
                                         "n": int(r["size"])}
                                for k, r in g.iterrows()}
    if "future_rise" in df.columns:
        fr = pd.to_numeric(df["future_rise"], errors="coerce") * 100
        out["future_rise_hist"] = histogram(fr)
        out["future_rise_pct"] = {
            k: round(float(fr.quantile(q)), 2)
            for k, q in (("p10", .1), ("p25", .25), ("median", .5),
                         ("p75", .75), ("p90", .9))}
    return out


def redundant_pairs(df: pd.DataFrame, cols: List[str],
                    thresh: float = CORR_WARN) -> List[Dict]:
    """相関が極端に高い組。冗長な列を洗い出す。"""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    # 全欠損・定数列は相関が定義できないので外す
    keep = [c for c in cols if sub[c].notna().sum() > 100 and sub[c].nunique() > 1]
    if len(keep) < 2:
        return []
    corr = sub[keep].corr()
    out = []
    seen = set()
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= thresh and (a, b) not in seen:
                seen.add((a, b))
                out.append({"a": a, "b": b, "corr": round(float(r), 4)})
    out.sort(key=lambda d: -abs(d["corr"]))
    return out[:80]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="特徴量の EDA サマリ")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "eda.json"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    cols = [c for c in F.all_columns() if c in df.columns]
    print(f"[load] {len(df):,}行 / 特徴量 {len(cols)}列")

    print("[eda] 列ごとの要約")
    stats = {c: describe_column(df[c]) for c in cols}

    print("[eda] ヒストグラム")
    hists = {c: histogram(df[c]) for c in cols}

    print("[eda] 年別の欠損率")
    miss_year = missing_by_year(df, cols)

    print("[eda] ラベル")
    labels = label_stats(df)

    print("[eda] 冗長な列の検出")
    raw = [c for c in cols if not c.endswith("_r")]
    dup = redundant_pairs(df, raw)

    # 問題のある列を洗い出す
    problems = []
    for c, s in stats.items():
        if s["missing_pct"] >= 90:
            problems.append({"col": c, "kind": "欠損が9割超",
                             "detail": f"{s['missing_pct']}%"})
        elif s["n_unique"] <= 1:
            problems.append({"col": c, "kind": "値が1種類しかない",
                             "detail": f"{s['n_unique']}種"})
        if s["n_inf"]:
            problems.append({"col": c, "kind": "無限大が含まれる",
                             "detail": f"{s['n_inf']}件"})
        if s.get("outlier_pct", 0) >= 10:
            problems.append({"col": c, "kind": "外れ値が1割超",
                             "detail": f"{s['outlier_pct']}%"})

    payload = {
        "n_rows": int(len(df)),
        "n_features": len(cols),
        "date_min": str(pd.to_datetime(df["Date"]).min().date()),
        "date_max": str(pd.to_datetime(df["Date"]).max().date()),
        "n_codes": int(df["Code"].nunique()),
        "groups": {g: [c for c in F.GROUPS[g] if c in df.columns]
                   for g in F.GROUPS if not g.endswith("_rank")},
        "stats": stats, "hist": hists, "missing_by_year": miss_year,
        "label": labels, "redundant": dup, "problems": problems,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"[done] {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")

    print(f"\n[問題のある列] {len(problems)}件")
    for p in problems[:20]:
        print(f"  {p['col']:<26} {p['kind']:<16} {p['detail']}")
    print(f"\n[冗長な組] 相関 >= {CORR_WARN}: {len(dup)}組")
    for d in dup[:10]:
        print(f"  {d['a']:<24} {d['b']:<24} r={d['corr']:+.4f}")
    print(f"\n[ラベル] 正例率 {labels['overall_rate']}% "
          f"({labels['n_positive']:,} / {labels['n']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
