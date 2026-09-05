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
#
# 変更軸は3つ:
#   (a) 高値窓     52週(245日) → 78週(368日)   … 超えるべき水準を上げる
#   (b) ホライズン 1〜6ヶ月 → 1〜3ヶ月          … 予測する期間を絞る
#   (c) 定着条件   期間延長 / 水準維持の追加     … 失速したケースを正例から外す
#
# (c) は2案:
#   c1. hold_days を 20→40 に延長 … ブレイク後40営業日、-8%を割らない
#   c2. sustain を追加            … ブレイク60営業日後もブレイク時終値以上
CONFIGS: List[LabelConfig] = [
    # --- 基準 ---
    LabelConfig(high_window=245, horizon_end=120),                       # A 現行
    # --- 単一軸の変更 ---
    LabelConfig(high_window=245, horizon_end=60),                        # B 期間のみ短縮
    LabelConfig(high_window=368, horizon_end=120),                       # C 高値窓のみ拡大
    LabelConfig(high_window=245, horizon_end=120, hold_days=40),         # D 定着期間のみ延長
    LabelConfig(high_window=245, horizon_end=120, sustain_days=60),      # E 水準維持のみ追加
    # --- ご提案（高値窓 + 期間）---
    LabelConfig(high_window=368, horizon_end=60),                        # F
    # --- ご提案 + 定着条件 ---
    LabelConfig(high_window=368, horizon_end=60, hold_days=40),          # G F + 定着40日
    LabelConfig(high_window=368, horizon_end=60, sustain_days=60),       # H F + 水準維持
    LabelConfig(high_window=368, horizon_end=60, hold_days=40,
                sustain_days=60),                                        # I 全部
    # --- 中間案: 期間を絞りつつ水準維持を課す（高値窓は52週のまま）---
    LabelConfig(high_window=245, horizon_end=60, sustain_days=60),        # J
]

#: ラベルの重なりを調べる組み合わせ（インデックス）。
#: 「2モデルを作って両方1なら確実」が成り立つかを検証するため。
#: 独立でない（片方が他方の部分集合など）なら、併用しても情報は増えない。
OVERLAP_PAIRS = [(4, 9)]   # E(52週/1〜6ヶ月/+維持) と J(52週/1〜3ヶ月/+維持)


def label_frame(bars: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """(Code, Date, label) だけを返す。定義間でラベルを突き合わせるため。"""
    df = price_panel(bars, cfg)
    df = breakout_flags(df, cfg)
    df = attach_labels(df, cfg)
    df["ym"] = df["Date"].dt.to_period("M")
    is_me = df.groupby(["Code", "ym"], sort=False)["Date"].transform("max") == df["Date"]
    s = df[is_me]
    s = s[s["label"].notna() & s["high52w"].notna()]
    s = s[(s["r_high"] < cfg.max_rhigh_at_t) & (s["tv_ma20"] >= cfg.min_trading_value)]
    return s[["Code", "Date", "label"]].copy()


def overlap(bars: pd.DataFrame, a: LabelConfig, b: LabelConfig) -> Dict:
    """
    2つの定義のラベルがどれだけ重なるかを調べる。

    「2つのモデルを作って両方が1なら確実」という考えは、
    2つのラベルが独立な情報を持っている場合にのみ成り立つ。
    片方が他方の部分集合なら、併用しても情報は増えず、
    単に片方のモデルの閾値を上げたのと同じことになる。
    """
    fa = label_frame(bars, a).rename(columns={"label": "a"})
    fb = label_frame(bars, b).rename(columns={"label": "b"})
    m = fa.merge(fb, on=["Code", "Date"], how="inner")
    if len(m) == 0:
        return {"n": 0}
    a1, b1 = m["a"] == 1, m["b"] == 1
    both = int((a1 & b1).sum())
    only_a = int((a1 & ~b1).sum())
    only_b = int((~a1 & b1).sum())
    neither = int((~a1 & ~b1).sum())
    return {
        "a": a.name, "b": b.name, "n": int(len(m)),
        "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
        "a_rate": round(float(a1.mean()), 4),
        "b_rate": round(float(b1.mean()), 4),
        # b が a の部分集合か = b=1 なのに a=0 のケースがゼロか
        "b_subset_of_a": only_b == 0,
        "a_subset_of_b": only_a == 0,
        # 両方1 が b=1 とどれだけ一致するか
        "both_equals_b": round(both / max(1, int(b1.sum())), 4),
        # 相関（phi係数）
        "phi": round(float(m["a"].corr(m["b"])), 4),
    }


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

    print("\n" + "=" * 118)
    print("定義ごとの比較（正例率と、ホライズン内の値動き中央値）")
    print("-" * 118)
    hdr = (f"{'#':<3}{'定義':<44}{'サンプル':>10}{'正例率':>9}"
           f"{'正例:最大上昇':>14}{'正例:終端':>12}{'負例:終端':>12}{'分離度':>9}")
    print(hdr)
    print("-" * 118)
    for i, r in enumerate(results):
        tag = chr(ord("A") + i)
        if r["n"] == 0:
            print(f"{tag:<3}{r['config']:<44}{'サンプルなし':>10}")
            continue
        p = r.get("positive", {}); n = r.get("negative", {})
        # 分離度 = 正例の最大上昇 p25 − 負例の最大上昇 p75。正なら分布が重ならない
        sep = None
        if p.get("maxGain_p25") is not None and n.get("maxGain_p75") is not None:
            sep = p["maxGain_p25"] - n["maxGain_p75"]
        print(f"{tag:<3}{r['config']:<44}{r['n']:>10,}{r['positive_rate']*100:>8.2f}%"
              f"{str(p.get('maxGain_median'))+'%':>14}"
              f"{str(p.get('endGain_median'))+'%':>12}"
              f"{str(n.get('endGain_median'))+'%':>12}"
              f"{('+' if sep and sep > 0 else '') + str(round(sep,1)) + 'pt' if sep is not None else '—':>9}")
    print("-" * 118)
    print("分離度 = 正例の最大上昇率p25 − 負例の最大上昇率p75。正なら四分位範囲が重ならない")

    # --- ラベルの重なり分析 ---
    overlaps = []
    for ia, ib in OVERLAP_PAIRS:
        if ia >= len(CONFIGS) or ib >= len(CONFIGS):
            continue
        print("\n" + "=" * 78)
        print(f"ラベルの重なり: {chr(65+ia)} と {chr(65+ib)}")
        print("-" * 78)
        ov = overlap(bars, CONFIGS[ia], CONFIGS[ib])
        overlaps.append(ov)
        if ov["n"] == 0:
            print("  共通サンプルがありません")
            continue
        print(f"  共通サンプル {ov['n']:,}")
        print(f"  {chr(65+ia)}=1 の率 {ov['a_rate']*100:.2f}%  /  "
              f"{chr(65+ib)}=1 の率 {ov['b_rate']*100:.2f}%")
        print(f"  両方1        {ov['both']:>8,}")
        print(f"  {chr(65+ia)}のみ1      {ov['only_a']:>8,}")
        print(f"  {chr(65+ib)}のみ1      {ov['only_b']:>8,}")
        print(f"  両方0        {ov['neither']:>8,}")
        print(f"  相関(phi)     {ov['phi']}")
        if ov["b_subset_of_a"]:
            print(f"  → {chr(65+ib)}=1 は必ず {chr(65+ia)}=1（部分集合）。"
                  f"「両方1」は {chr(65+ib)}=1 と完全に同じ")
        elif ov["a_subset_of_b"]:
            print(f"  → {chr(65+ia)}=1 は必ず {chr(65+ib)}=1（部分集合）")
        else:
            print(f"  → 部分集合ではない。「両方1」は {chr(65+ib)}=1 の "
                  f"{ov['both_equals_b']*100:.1f}% をカバー")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "overlaps": overlaps}, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
