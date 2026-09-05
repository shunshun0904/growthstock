#!/usr/bin/env python3
"""
ラベル定義を目視検証するためのチャートデータを書き出す。

「正例としたサンプルが、本当に先1〜6ヶ月で明確なモメンタムを形成しているか」を
実データのチャートで確認するのが目的。定義の良し悪しは数字だけでは判断できない。

各ケースについて、基準日 t の前後を含む値動きと、
  * 予測ホライズン [t+20, t+120]
  * ブレイク判定日 b
  * 52週高値のライン
  * ブレイク後の最大上昇率 / 最大下落率
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
    HORIZON_END, HORIZON_START, attach_labels, breakout_flags, price_panel,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

BEFORE = 120   # 基準日より前に何営業日ぶん見せるか
AFTER = 150    # 基準日より後に何営業日ぶん見せるか


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
    t_pos = i - lo   # ウィンドウ内での基準日の位置

    # ホライズン内のブレイク日
    h_lo, h_hi = i + HORIZON_START, min(len(g) - 1, i + HORIZON_END)
    bo_positions = []
    if h_lo <= h_hi:
        seg = g.iloc[h_lo:h_hi + 1]
        bo_positions = [int(p - lo) for p in seg.index[seg["is_breakout"] == True]]  # noqa: E712

    close_t = float(g.iloc[i]["close"])
    fwd = g.iloc[i + HORIZON_START: h_hi + 1]["close"] if h_lo <= h_hi else pd.Series(dtype=float)
    fwd_max = float(fwd.max()) if len(fwd) else float("nan")
    fwd_min = float(fwd.min()) if len(fwd) else float("nan")

    # 最終的な到達点（ホライズン終端の終値）
    end_close = float(g.iloc[h_hi]["close"]) if h_lo <= h_hi else float("nan")

    def r(v, d=2):
        return None if (v is None or not np.isfinite(v)) else round(float(v), d)

    return {
        "code": code,
        "name": name,
        "label": int(label),
        "t": t_date.date().isoformat(),
        "tPos": t_pos,
        "horizon": [t_pos + HORIZON_START, t_pos + HORIZON_END],
        "breakouts": bo_positions,
        "closeAtT": r(close_t, 1),
        "maxGain": r((fwd_max / close_t - 1) * 100),
        "maxDraw": r((fwd_min / close_t - 1) * 100),
        "endGain": r((end_close / close_t - 1) * 100),
        "rHighAtT": r(float(g.iloc[i]["r_high"])),
        "dates": [d.date().isoformat() for d in win["Date"]],
        "close": [r(v, 1) for v in win["close"]],
        "high52w": [r(v, 1) for v in win["high52w"]],
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
    print("[panel] 指標とブレイク判定を再計算")
    panel = attach_labels(breakout_flags(price_panel(bars)))

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
            "horizonStart": HORIZON_START, "horizonEnd": HORIZON_END,
            "note": "正例 = [t+20, t+120]営業日に「52週高値更新 + 出来高1.5倍 + 20営業日-8%以内で定着」が起きた",
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
