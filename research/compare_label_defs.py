#!/usr/bin/env python3
"""
ラベル定義を複数比較して、正例率とホライズン内の値動き分布を実測する。

「52週高値 → 78週高値」「1〜6ヶ月 → 1〜3ヶ月」のように定義を厳しくすると
何がどう変わるかを、推測ではなく同一データ上の実測で並べる。

生データ (research/_data/bars_*.parquet) を一度だけ読み、
定義ごとに指標とラベルを計算し直す。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import (  # noqa: E402
    LabelConfig, attach_labels, breakout_flags, price_panel,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

# 比較する定義。名前は LabelConfig.name が自動生成する。
CONFIGS: List[LabelConfig] = [
    LabelConfig(high_window=245, horizon_start=20, horizon_end=120),  # 現行
    LabelConfig(high_window=245, horizon_start=20, horizon_end=60),   # 期間だけ短縮
    LabelConfig(high_window=368, horizon_start=20, horizon_end=120),  # 高値窓だけ拡大
    LabelConfig(high_window=368, horizon_start=20, horizon_end=60),   # 両方（ご提案）
]


def evaluate(bars: pd.DataFrame, cfg: LabelConfig) -> Dict:
    """1つの定義について、サンプル数・正例率・値動き分布を算出する。"""
    df = price_panel(bars, cfg)
    df = breakout_flags(df, cfg)
    df = attach_labels(df, cfg)

    # 月末営業日をサンプルとする
    df["ym"] = df["Date"].dt.to_period("M")
    is_me = df.groupby(["Code", "ym"], sort=False)["Date"].transform("max") == df["Date"]
    s = df[is_me].copy()

    steps = {"月末サンプル": len(s)}
    s = s[s["label"].notna()];            steps["ラベル確定"] = len(s)
    s = s[s["high52w"].notna()];          steps["高値窓あり"] = len(s)
    s = s[s["r_high"] < cfg.max_rhigh_at_t];   steps["高値圏を除外"] = len(s)
    s = s[s["tv_ma20"] >= cfg.min_trading_value]; steps["流動性フィルタ"] = len(s)

    if len(s) == 0:
        return {"config": cfg.name, "n": 0, "note": "サンプルが残らない"}

    # ホライズン内の値動き（ラベルの妥当性を見るため、正例/負例それぞれで）
    g = df.groupby("Code", sort=False)
    fwd_max = g["close"].transform(
        lambda x: x[::-1].rolling(cfg.horizon_end - cfg.horizon_start + 1, min_periods=1)
                   .max()[::-1].shift(-cfg.horizon_start)
    )
    fwd_end = g["close"].shift(-cfg.horizon_end)
    df["_gain_max"] = (fwd_max / df["close"] - 1) * 100
    df["_gain_end"] = (fwd_end / df["close"] - 1) * 100
    s = s.join(df[["_gain_max", "_gain_end"]], rsuffix="_r")

    out = {
        "config": cfg.name,
        "high_window": cfg.high_window,
        "horizon": [cfg.horizon_start, cfg.horizon_end],
        "n": int(len(s)),
        "positive_rate": round(float(s["label"].mean()), 4),
        "n_positive": int(s["label"].sum()),
        "n_stocks": int(s["Code"].nunique()),
        "date_from": str(s["Date"].min().date()),
        "date_to": str(s["Date"].max().date()),
        "funnel": steps,
    }
    for lab, tag in [(1, "positive"), (0, "negative")]:
        part = s[s["label"] == lab]
        if len(part) == 0:
            continue
        gm = part["_gain_max"].dropna()
        ge = part["_gain_end"].dropna()
        out[tag] = {
            "n": int(len(part)),
            "maxGain_median": round(float(gm.median()), 2) if len(gm) else None,
            "maxGain_p25": round(float(gm.quantile(.25)), 2) if len(gm) else None,
            "maxGain_p75": round(float(gm.quantile(.75)), 2) if len(gm) else None,
            "endGain_median": round(float(ge.median()), 2) if len(ge) else None,
            "endGain_positive_rate": round(float((ge > 0).mean()), 3) if len(ge) else None,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ラベル定義を比較する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "label_comparison.json"))
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"[load] 日次バー {len(bars):,}行 / {bars['Code'].nunique():,}銘柄")

    results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg.name} (高値窓 {cfg.high_window}日 / "
              f"ホライズン t+{cfg.horizon_start}〜t+{cfg.horizon_end}) ===")
        r = evaluate(bars, cfg)
        results.append(r)
        if r["n"] == 0:
            print("    サンプルが残りません")
            continue
        print(f"    サンプル {r['n']:,} / 正例 {r['n_positive']:,} "
              f"({r['positive_rate']*100:.2f}%) / 銘柄 {r['n_stocks']:,}")
        print(f"    期間 {r['date_from']} 〜 {r['date_to']}")
        for tag, ja in [("positive", "正例"), ("negative", "負例")]:
            d = r.get(tag)
            if not d:
                continue
            print(f"    {ja}: 最大上昇 中央値 {d['maxGain_median']}% "
                  f"(四分位 {d['maxGain_p25']}〜{d['maxGain_p75']}%) / "
                  f"終端 中央値 {d['endGain_median']}% "
                  f"/ 終端プラス率 {d['endGain_positive_rate']}")

    print("\n" + "=" * 78)
    print(f"{'定義':<24}{'サンプル':>10}{'正例率':>9}{'正例の最大上昇':>16}{'負例の最大上昇':>16}")
    print("-" * 78)
    for r in results:
        if r["n"] == 0:
            print(f"{r['config']:<24}{'—':>10}")
            continue
        p = r.get("positive", {}); n = r.get("negative", {})
        print(f"{r['config']:<24}{r['n']:>10,}{r['positive_rate']*100:>8.2f}%"
              f"{str(p.get('maxGain_median'))+'%':>16}{str(n.get('maxGain_median'))+'%':>16}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
