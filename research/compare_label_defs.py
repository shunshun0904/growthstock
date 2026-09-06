#!/usr/bin/env python3
"""
ラベル定義を複数比較して、正例率とホライズン内の値動き分布を実測する。

母集団は「高値を更新した日 × 銘柄」。比較の軸は2つ:

  (a) 高値窓     52週(245日) / 78週(368日)  … 母集団そのものが変わる
  (b) 継続の条件 維持日数 / 終盤の水準 / トレンド
                                            … 母集団は同じでラベルだけ変わる

「到達（先60営業日以内に +20%）」だけを条件にすると、一瞬吹き上げて
すぐ下落トレンドに入った銘柄も正例になる。それはモメンタムではない。
継続の条件を足すと正例は減るが、残った正例が何なのかが変わる。
どれだけ減って何が残るのかを、推測ではなく同一データ上の実測で並べる。

生データ (research/_data/bars_*.parquet) を一度だけ読み、
高値窓ごとにパネルを作り直し、その上で継続条件だけを差し替える。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import (  # noqa: E402
    BREAKOUT_COOLDOWN, LabelConfig, MIN_TRADING_VALUE, RiseConfig,
    attach_rise_label, mark_new_highs, price_panel,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

W52, W78 = 245, 368

#: 比較する (高値窓, 目的変数) の組み合わせ。
#:
#: 到達のみ = 従来の定義。継続の条件を1つずつ足して効果を分離し、
#: 最後に全部入り（＝現在の既定）と、さらに厳しくした案を置く。
_REACH = dict(horizon=60, threshold=0.20, keep_days=0, end_ratio=None,
              require_uptrend=False)
CONFIGS: List[Tuple[int, RiseConfig, str]] = [
    (W52, RiseConfig(**_REACH),                                  "旧定義（52週・到達のみ）"),
    (W78, RiseConfig(**_REACH),                                  "78週・到達のみ"),
    (W78, RiseConfig(**{**_REACH, "keep_days": 10}),             "78週・+維持10日"),
    (W78, RiseConfig(**{**_REACH, "end_ratio": 0.10}),           "78週・+終盤10%"),
    (W78, RiseConfig(**{**_REACH, "require_uptrend": True}),     "78週・+トレンド"),
    (W78, RiseConfig(keep_days=10, end_ratio=0.10,
                     require_uptrend=True),                      "78週・継続すべて（緩い案）"),
    (W78, RiseConfig(keep_days=20, end_ratio=0.15,
                     require_uptrend=True),                      "78週・継続すべて（採用）"),
    (W52, RiseConfig(keep_days=10, end_ratio=0.10,
                     require_uptrend=True),                      "52週・継続すべて"),
]


def panel_for(bars: pd.DataFrame, high_window: int) -> pd.DataFrame:
    """高値窓を変えるとパネルごと作り直しになる（母集団が変わるため）。"""
    df = price_panel(bars, LabelConfig(high_window=high_window))
    return mark_new_highs(df, cooldown=BREAKOUT_COOLDOWN, on_high=True)


#: ラベル計算と絞り込みに要る列だけ。
#: パネルは1,000万行規模あり、定義ごとに丸ごと copy すると
#: ランナーのメモリが持たない。
NEEDED = ["Code", "Date", "close", "is_fresh_break", "high52w", "tv_ma20"]


def evaluate(panel: pd.DataFrame, cfg: RiseConfig, high_window: int) -> Dict:
    """1つの定義について、サンプル数・正例率・値動き分布を算出する。"""
    df = attach_rise_label(panel[NEEDED].copy(), cfg)
    s = df[df["is_fresh_break"] == True].copy()   # noqa: E712

    steps = {"新規ブレイク": len(s)}
    s = s[s["label"].notna()]
    steps["ラベル確定"] = len(s)
    s = s[s["high52w"].notna()]
    steps["高値窓あり"] = len(s)
    s = s[s["tv_ma20"] >= MIN_TRADING_VALUE]
    steps["流動性フィルタ"] = len(s)

    if len(s) == 0:
        return {"config": cfg.name, "high_window": high_window, "n": 0,
                "note": "サンプルが残らない"}

    # 継続条件を1つずつ重ねたときの正例数。どの条件がどれだけ削ったかを見る
    conds = [("到達", s["future_rise"] >= cfg.threshold)]
    if cfg.keep_days:
        conds.append((f"維持{cfg.keep_days}日", s["keep_days_cnt"] >= cfg.keep_days))
    if cfg.end_ratio is not None:
        conds.append((f"終盤+{cfg.end_ratio*100:.0f}%", s["end_level"] >= cfg.end_ratio))
    if cfg.require_uptrend:
        conds.append(("トレンド", s["uptrend_end"] == 1.0))
    cascade, mask = {}, pd.Series(True, index=s.index)
    for name, c in conds:
        mask = mask & c.fillna(False)
        cascade[name] = int(mask.sum())

    out = {
        "config": cfg.name,
        "high_window": high_window,
        "high_weeks": round(high_window / 245 * 52),
        "n": int(len(s)),
        "positive_rate": round(float(s["label"].mean()), 4),
        "n_positive": int(s["label"].sum()),
        "n_stocks": int(s["Code"].nunique()),
        "date_from": str(s["Date"].min().date()),
        "date_to": str(s["Date"].max().date()),
        "funnel": steps,
        "cascade": cascade,
    }
    for lab, tag in [(1, "positive"), (0, "negative")]:
        part = s[s["label"] == lab]
        if len(part) == 0:
            continue
        gm = (part["future_rise"] * 100).dropna()
        ge = (part["end_level"] * 100).dropna()
        kd = part["keep_days_cnt"].dropna()
        up = part["uptrend_end"].dropna()
        out[tag] = {
            "n": int(len(part)),
            "maxGain_median": round(float(gm.median()), 2) if len(gm) else None,
            "maxGain_p25": round(float(gm.quantile(.25)), 2) if len(gm) else None,
            "maxGain_p75": round(float(gm.quantile(.75)), 2) if len(gm) else None,
            "endGain_median": round(float(ge.median()), 2) if len(ge) else None,
            "endGain_positive_rate": round(float((ge > 0).mean()), 3) if len(ge) else None,
            "keepDays_median": round(float(kd.median()), 1) if len(kd) else None,
            "uptrend_rate": round(float(up.mean()), 3) if len(up) else None,
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

    panels: Dict[int, pd.DataFrame] = {}
    results = []
    for hw, cfg, note in CONFIGS:
        if hw not in panels:
            print(f"\n[panel] 高値窓 {hw}営業日（{round(hw/245*52)}週）でパネルを作成")
            panels[hw] = panel_for(bars, hw)
        print(f"\n=== {note} | {cfg.name} ===")
        r = evaluate(panels[hw], cfg, hw)
        r["note"] = note
        results.append(r)
        if r["n"] == 0:
            print("    サンプルが残りません")
            continue
        print(f"    サンプル {r['n']:,} / 正例 {r['n_positive']:,} "
              f"({r['positive_rate']*100:.2f}%) / 銘柄 {r['n_stocks']:,}")
        print("    条件を重ねたときの正例数: " +
              " → ".join(f"{k} {v:,}" for k, v in r["cascade"].items()))
        for tag, ja in [("positive", "正例"), ("negative", "負例")]:
            d = r.get(tag)
            if not d:
                continue
            print(f"    {ja}: 最大上昇 中央値 {d['maxGain_median']}% "
                  f"(四分位 {d['maxGain_p25']}〜{d['maxGain_p75']}%) / "
                  f"終盤 中央値 {d['endGain_median']}% "
                  f"/ 維持 {d['keepDays_median']}日 / 上昇トレンド率 {d['uptrend_rate']}")

    print("\n" + "=" * 122)
    print("定義ごとの比較")
    print("-" * 122)
    print(f"{'#':<3}{'定義':<26}{'母集団':>9}{'正例率':>9}{'正例数':>9}"
          f"{'正例:最大上昇':>14}{'正例:終盤':>11}{'負例:終盤':>11}"
          f"{'正例:上昇ﾄﾚﾝﾄﾞ率':>17}")
    print("-" * 122)
    for i, r in enumerate(results):
        tag = chr(ord("A") + i)
        if r["n"] == 0:
            print(f"{tag:<3}{r['note']:<26}{'サンプルなし':>9}")
            continue
        p, n = r.get("positive", {}), r.get("negative", {})
        print(f"{tag:<3}{r['note']:<26}{r['n']:>9,}{r['positive_rate']*100:>8.2f}%"
              f"{r['n_positive']:>9,}"
              f"{str(p.get('maxGain_median')) + '%':>14}"
              f"{str(p.get('endGain_median')) + '%':>11}"
              f"{str(n.get('endGain_median')) + '%':>11}"
              f"{str(round((p.get('uptrend_rate') or 0) * 100)) + '%':>17}")
    print("-" * 122)
    print("終盤 = t+60 時点の5日平均が基準日終値から何%か。"
          "「一瞬の上昇」を外せているなら、正例の終盤が大きく改善するはず")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
