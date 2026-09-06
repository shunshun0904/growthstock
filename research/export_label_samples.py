#!/usr/bin/env python3
"""
ラベル定義を目視検証するためのチャートデータを書き出す。

数字だけでは定義の良し悪しは判断できないので、実データのチャートで確認する。

現在の定義:
  母集団 = 52週高値を更新した日（高値ベース、連続更新は初回のみ）
  正例   = 更新日の終値から、先60営業日以内に +20% 以上（終値ベース）

各ケースについて、更新日 t の前後を含む値動きと、
  * 判定期間 [t+1, t+60]
  * +20% のライン（到達したかが一目で分かる）
  * 52週高値のライン（t で更新しているはず）
  * 最大上昇率 / 最大下落率
を出力する。

出力: research/_data/label_samples.json
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
    POPULATION, RISE_HORIZON, RISE_THRESHOLD, add_breakout_context,
    attach_labels, attach_rise_label, breakout_flags, mark_new_highs,
    price_panel,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

BEFORE = 120              # 基準日より前に何営業日ぶん見せるか
AFTER = RISE_HORIZON + 20  # 判定期間より少し先まで見せる


def load_bars(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_case(panel: pd.DataFrame, code: str, t_date: pd.Timestamp,
               name: str, label: int) -> Dict | None:
    """1ケース分のチャートデータを組み立てる。"""
    g = panel[panel["Code"] == code].reset_index(drop=True)
    idx = g.index[g["Date"] == t_date]
    if len(idx) == 0:
        return None
    i = int(idx[0])

    lo = max(0, i - BEFORE)
    hi = min(len(g), i + AFTER + 1)
    win = g.iloc[lo:hi]
    t_pos = i - lo   # ウィンドウ内での基準日（＝52週高値の更新日）の位置

    close_t = float(g.iloc[i]["close"])
    # 判定期間は t+1 〜 t+RISE_HORIZON。当日は含まない
    h_lo, h_hi = i + 1, min(len(g) - 1, i + RISE_HORIZON)
    fwd = g.iloc[h_lo:h_hi + 1]["close"] if h_lo <= h_hi else pd.Series(dtype=float)
    fwd_max = float(fwd.max()) if len(fwd) else float("nan")
    fwd_min = float(fwd.min()) if len(fwd) else float("nan")
    end_close = float(g.iloc[h_hi]["close"]) if h_lo <= h_hi else float("nan")

    # +20% に最初に到達した日（正例ならあるはず）
    hit_pos = None
    if len(fwd):
        hits = np.where(fwd.to_numpy() >= close_t * (1 + RISE_THRESHOLD))[0]
        if len(hits):
            hit_pos = int(h_lo + hits[0] - lo)

    def r(v, d=2):
        return None if (v is None or not np.isfinite(v)) else round(float(v), d)

    def col(name_, d=1):
        return [r(v, d) for v in win[name_]] if name_ in win.columns else None

    return {
        "code": code,
        "name": name,
        "label": int(label),
        "t": t_date.date().isoformat(),
        "tPos": t_pos,
        # 判定期間（ウィンドウ内の位置）
        "horizon": [t_pos + 1, t_pos + RISE_HORIZON],
        "closeAtT": r(close_t, 1),
        # 目標ライン。チャートに水平線として引く
        "target": r(close_t * (1 + RISE_THRESHOLD), 1),
        "thresholdPct": round(RISE_THRESHOLD * 100, 1),
        "hitPos": hit_pos,
        "maxGain": r((fwd_max / close_t - 1) * 100),
        "maxDraw": r((fwd_min / close_t - 1) * 100),
        "endGain": r((end_close / close_t - 1) * 100),
        # ブレイクの文脈（新しい特徴量が妥当かの目視確認にも使う）
        "baseLength": r(float(g.iloc[i].get("base_length", np.nan)), 0),
        "breakMargin": r(float(g.iloc[i].get("break_margin", np.nan))),
        "closePosition": r(float(g.iloc[i].get("close_position", np.nan))),
        "volRatio": r(float(g.iloc[i]["vol"] / g.iloc[i]["vol_ma20"]) * 100
                      if np.isfinite(g.iloc[i].get("vol_ma20", np.nan))
                      and g.iloc[i].get("vol_ma20", 0) > 0 else np.nan),
        "dates": [d.date().isoformat() for d in win["Date"]],
        "close": col("close"),
        "high52w": col("high52w"),
        "volume": [None if not np.isfinite(v) else int(v) for v in win["vol"]],
        "volMa20": [None if not np.isfinite(v) else int(v) for v in win["vol_ma20"]],
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ラベル目視検証用のチャートデータを書き出す")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "label_samples.json"))
    ap.add_argument("--n-pos", type=int, default=40, help="正例のサンプル数")
    ap.add_argument("--n-neg", type=int, default=20, help="負例のサンプル数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ds = pd.read_parquet(args.dataset)
    ds["Date"] = pd.to_datetime(ds["Date"])
    print(f"[data] {len(ds):,}サンプル / 正例率 {ds['label'].mean()*100:.2f}%")

    bars = load_bars(args.data_dir)
    print("[panel] 指標とラベルを再計算")
    panel = price_panel(bars)
    if POPULATION == "breakout":
        panel = add_breakout_context(attach_rise_label(mark_new_highs(panel)))
    else:
        panel = attach_labels(breakout_flags(panel))

    # 銘柄名（あれば）
    names: Dict[str, str] = {}
    master = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "public", "data", "stocks.json")
    if os.path.exists(master):
        try:
            for s in json.load(open(master, encoding="utf-8")).get("stocks", []):
                names[s.get("jqCode", "")] = s.get("name", "")
        except Exception:
            pass

    rng = np.random.default_rng(args.seed)
    cases = []
    for label, n in [(1, args.n_pos), (0, args.n_neg)]:
        pool = ds[ds["label"] == label]
        if len(pool) == 0:
            print(f"[warn] label={label} のサンプルがありません")
            continue
        take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1 << 30)))
        print(f"[sample] label={label}: {len(take)}件を抽出")
        for _, row in take.iterrows():
            c = build_case(panel, row["Code"], row["Date"],
                           names.get(row["Code"], ""), label)
            if c:
                cases.append(c)

    # 正例の中身を要約して、定義が意図どおりか数字でも確認できるようにする
    pos = [c for c in cases if c["label"] == 1]
    neg = [c for c in cases if c["label"] == 0]
    summary = {}
    for tag, group in [("positive", pos), ("negative", neg)]:
        if not group:
            continue
        gains = [c["maxGain"] for c in group if c["maxGain"] is not None]
        ends = [c["endGain"] for c in group if c["endGain"] is not None]
        summary[tag] = {
            "n": len(group),
            "maxGain_median": round(float(np.median(gains)), 2) if gains else None,
            "maxGain_p25": round(float(np.percentile(gains, 25)), 2) if gains else None,
            "maxGain_p75": round(float(np.percentile(gains, 75)), 2) if gains else None,
            "endGain_median": round(float(np.median(ends)), 2) if ends else None,
            "endGain_positive_rate": round(float(np.mean([e > 0 for e in ends])), 3) if ends else None,
        }

    print("\n[要約] ホライズン内の値動き")
    for tag, s in summary.items():
        lbl = "正例" if tag == "positive" else "負例"
        print(f"  {lbl}: 最大上昇率 中央値 {s['maxGain_median']}% "
              f"(四分位 {s['maxGain_p25']}〜{s['maxGain_p75']}%) / "
              f"終端リターン 中央値 {s['endGain_median']}% "
              f"/ 終端がプラスの割合 {s['endGain_positive_rate']}")

    out = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "definition": {
            "riseHorizon": RISE_HORIZON,
            "riseThreshold": round(RISE_THRESHOLD * 100, 1),
            "population": POPULATION,
            "note": ("母集団 = 52週高値を更新した日（高値ベース、連続更新は初回のみ）。"
                     f"正例 = 更新日の終値から先{RISE_HORIZON}営業日以内に "
                     f"+{RISE_THRESHOLD*100:.0f}%以上（終値ベース）"),
        },
        "datasetStats": {
            "n": int(len(ds)),
            "positiveRate": round(float(ds["label"].mean()), 4),
            "from": ds["Date"].min().date().isoformat(),
            "to": ds["Date"].max().date().isoformat(),
        },
        "summary": summary,
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"\n[done] {args.out} ({os.path.getsize(args.out)/1e6:.2f}MB / {len(cases)}ケース)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
