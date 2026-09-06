#!/usr/bin/env python3
"""
ラベル定義を目視検証するためのチャートデータを書き出す。

数字だけでは定義の良し悪しは判断できないので、実データのチャートで確認する。

現在の定義:
  母集団 = 78週高値を更新した日（高値ベース、連続更新は初回のみ）
  正例   = 到達（先60営業日以内に +20%）かつ継続（維持日数・終盤水準・トレンド）

各ケースについて、更新日 t の前後を含む値動きと、
  * 判定期間 [t+1, t+60]
  * +20% のライン（到達したかが一目で分かる）
  * 終盤水準のライン（継続条件）
  * 78週高値のライン（t で更新しているはず）
  * 移動平均（トレンド条件が見えるように）
  * 最大上昇率 / 最大下落率 / 維持日数 / 終盤水準
を出力する。

負例は2種類に分けて抽出する。
  未到達     … そもそも +20% に届かなかった
  継続せず   … +20% に届いたが続かなかった（今回の定義変更で外れた側）
「一瞬の増加ですぐ下落トレンド」が本当に外れているかは後者を見れば分かる。

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
    DEFAULT_RISE, HIGH_WINDOW, POPULATION, RISE_HORIZON, RISE_THRESHOLD,
    RiseConfig, add_breakout_context, attach_labels, attach_rise_label,
    breakout_flags, mark_new_highs, price_panel,
)

#: 採用した定義（G）と、比較のために残す緩い案（F）。
#: 採用案は緩い案の条件をすべて含み、しきい値だけを上げているので、
#: 採用案の正例は必ず緩い案の正例に含まれる（真部分集合）。
#: したがって食い違うのは「緩い案なら正例だが採用案では負例」の1組だけ。
#: 何を切り捨てる定義にしたのかを、あとから見て分かるように残す。
CFG_ADOPTED = DEFAULT_RISE
CFG_LOOSE = RiseConfig(keep_days=10, end_ratio=0.10, require_uptrend=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

BEFORE = 120              # 基準日より前に何営業日ぶん見せるか
AFTER = RISE_HORIZON + 20  # 判定期間より少し先まで見せる


def load_bars(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def verdict(df: pd.DataFrame, cfg: RiseConfig) -> pd.Series:
    """
    データセットに入っている材料からラベルを引き直す。

    2案は同じ材料（future_rise / keep_days_cnt / end_level / uptrend_end）に
    しきい値を変えて当てるだけなので、パネルを作り直す必要はない。
    """
    ok = pd.to_numeric(df["future_rise"], errors="coerce") >= cfg.threshold
    if cfg.keep_days:
        ok &= pd.to_numeric(df["keep_days_cnt"], errors="coerce") >= cfg.keep_days
    if cfg.end_ratio is not None:
        ok &= pd.to_numeric(df["end_level"], errors="coerce") >= cfg.end_ratio
    if cfg.require_uptrend:
        ok &= pd.to_numeric(df["uptrend_end"], errors="coerce") == 1.0
    return ok.fillna(False)


def build_case(panel: pd.DataFrame, code: str, t_date: pd.Timestamp,
               name: str, label: int, bucket: str | None = None) -> Dict | None:
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

    # トレンド条件（MA短期 >= MA長期）が目で見えるように移動平均も出す。
    # 銘柄の全系列で計算してから窓を切る（窓の中だけで計算すると頭が欠ける）
    cfg = DEFAULT_RISE
    ma_s = g["close"].rolling(cfg.trend_short, min_periods=cfg.trend_short).mean()
    ma_l = g["close"].rolling(cfg.trend_long, min_periods=cfg.trend_long).mean()

    close_t = float(g.iloc[i]["close"])
    # 判定期間は t+1 〜 t+RISE_HORIZON。当日は含まない
    h_lo, h_hi = i + 1, min(len(g) - 1, i + RISE_HORIZON)
    fwd = g.iloc[h_lo:h_hi + 1]["close"] if h_lo <= h_hi else pd.Series(dtype=float)
    fwd_max = float(fwd.max()) if len(fwd) else float("nan")
    fwd_min = float(fwd.min()) if len(fwd) else float("nan")
    end_close = float(g.iloc[h_hi]["close"]) if h_lo <= h_hi else float("nan")

    # +20% に最初に到達した日（到達していればあるはず）
    hit_pos = None
    keep_days = None
    if len(fwd):
        above = fwd.to_numpy() >= close_t * (1 + cfg.threshold)
        keep_days = int(above.sum())
        hits = np.where(above)[0]
        if len(hits):
            hit_pos = int(h_lo + hits[0] - lo)

    # 終盤の水準（判定期間の最後 end_window 日の平均）
    end_lo = max(h_lo, h_hi - cfg.end_window + 1)
    end_win = g.iloc[end_lo:h_hi + 1]["close"] if h_lo <= h_hi else pd.Series(dtype=float)
    end_level = (float(end_win.mean()) / close_t - 1) * 100 if len(end_win) else float("nan")
    uptrend = (float(ma_s.iloc[h_hi]) >= float(ma_l.iloc[h_hi])
               if h_lo <= h_hi and np.isfinite(ma_s.iloc[h_hi])
               and np.isfinite(ma_l.iloc[h_hi]) else None)

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
        # 継続の材料。採用案と緩い案のどちらで落ちたかを1件ずつ確認できるようにする
        "bucket": bucket,
        "keepDays": keep_days,
        "keepDaysF": CFG_LOOSE.keep_days,
        "keepDaysG": CFG_ADOPTED.keep_days,
        "endLevel": r(end_level),
        "endLineF": r(close_t * (1 + CFG_LOOSE.end_ratio), 1),
        "endLineG": r(close_t * (1 + CFG_ADOPTED.end_ratio), 1),
        "endRatioF": round(CFG_LOOSE.end_ratio * 100, 1),
        "endRatioG": round(CFG_ADOPTED.end_ratio * 100, 1),
        "uptrendEnd": uptrend,
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
        "maShort": [r(v, 1) for v in ma_s.iloc[lo:hi]],
        "maLong": [r(v, 1) for v in ma_l.iloc[lo:hi]],
        "volume": [None if not np.isfinite(v) else int(v) for v in win["vol"]],
        "volMa20": [None if not np.isfinite(v) else int(v) for v in win["vol_ma20"]],
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ラベル目視検証用のチャートデータを書き出す")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "label_samples.json"))
    ap.add_argument("--n-pos", type=int, default=48, help="正例のサンプル数")
    ap.add_argument("--n-neg", type=int, default=24, help="負例のサンプル数")
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

    # 採用案と緩い案のラベルを引き直して4つに分ける。
    # 採用案は緩い案の真部分集合なので、食い違うのは f_only の1組だけ。
    loose = verdict(ds, CFG_LOOSE)
    adopted = verdict(ds, CFG_ADOPTED)
    reached = pd.to_numeric(ds["future_rise"], errors="coerce") >= RISE_THRESHOLD
    reached = reached.fillna(False)

    assert int((adopted & ~loose).sum()) == 0, \
        "採用案が緩い案の部分集合になっていない（比較の前提が崩れている）"
    # 採用案の判定は dataset の label と一致していなければならない。
    # ずれていれば、ここで引き直した条件か build 側のどちらかが古い。
    mismatch = int((adopted != (ds["label"] == 1)).sum())
    assert mismatch == 0, f"引き直したラベルが dataset と {mismatch}件食い違う"

    buckets = [
        ("both_pos", "正例（採用）", 1, adopted),
        ("f_only", "緩い案なら正例 / 採用案では負例", 0, loose & ~adopted),
        ("reached_only", "到達したが継続せず", 0, ~loose & reached),
        ("not_reached", "未到達（+20%に届かず）", 0, ~loose & ~reached),
    ]
    print("[bucket] " + " / ".join(f"{ja} {int(m.sum()):,}件" for _, ja, _, m in buckets))

    # 正例と、採用案が切り捨てた側（f_only）を厚めに採る。
    # 定義が妥当かは「何を正例にしたか」と「何を外したか」の両方を見ないと分からない。
    quota = {"both_pos": args.n_pos, "f_only": args.n_neg,
             "reached_only": args.n_neg // 2, "not_reached": args.n_neg // 2}

    rng = np.random.default_rng(args.seed)
    cases = []
    for key, ja, label, mask in buckets:
        pool = ds[mask]
        n = quota[key]
        if len(pool) == 0 or n <= 0:
            print(f"[warn] {ja} のサンプルがありません")
            continue
        take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1 << 30)))
        print(f"[sample] {ja}: {len(take)}件を抽出")
        for _, row in take.iterrows():
            c = build_case(panel, row["Code"], row["Date"],
                           names.get(row["Code"], ""), label, key)
            if c:
                cases.append(c)

    # バケットごとに要約する。母集団全体の数字（抽出前）も併せて出す。
    # 抽出したケースだけを見ると、偶然の偏りを定義の性質と読み違える。
    summary = {}
    bucket_stats = {}
    for key, ja, label, mask in buckets:
        part = ds[mask]
        if len(part) == 0:
            continue
        fr = pd.to_numeric(part["future_rise"], errors="coerce") * 100
        el = pd.to_numeric(part["end_level"], errors="coerce") * 100
        kd = pd.to_numeric(part["keep_days_cnt"], errors="coerce")
        bucket_stats[key] = {
            "label": ja, "y": label, "n": int(len(part)),
            "share": round(float(len(part) / len(ds) * 100), 2),
            "maxGain_median": round(float(fr.median()), 2),
            "endGain_median": round(float(el.median()), 2),
            "keepDays_median": round(float(kd.median()), 1),
        }
    print("\n[要約] バケットごと（母集団全体）")
    for key, d in bucket_stats.items():
        print(f"  {d['label']:<34} {d['n']:>6,}件 ({d['share']:>5.2f}%) "
              f"最大上昇 {d['maxGain_median']:>6.2f}% / 終盤 {d['endGain_median']:>6.2f}% "
              f"/ 維持 {d['keepDays_median']:>4.1f}日")

    pos = [c for c in cases if c["label"] == 1]
    neg = [c for c in cases if c["label"] == 0]
    for tag, group in [("positive", pos), ("negative", neg)]:
        if not group:
            continue
        gains = [c["maxGain"] for c in group if c["maxGain"] is not None]
        ends = [c["endGain"] for c in group if c["endGain"] is not None]
        keeps = [c["keepDays"] for c in group if c["keepDays"] is not None]
        ups = [c["uptrendEnd"] for c in group if c["uptrendEnd"] is not None]
        summary[tag] = {
            "n": len(group),
            "maxGain_median": round(float(np.median(gains)), 2) if gains else None,
            "maxGain_p25": round(float(np.percentile(gains, 25)), 2) if gains else None,
            "maxGain_p75": round(float(np.percentile(gains, 75)), 2) if gains else None,
            "endGain_median": round(float(np.median(ends)), 2) if ends else None,
            "endGain_positive_rate": round(float(np.mean([e > 0 for e in ends])), 3) if ends else None,
            # 継続の軸
            "keepDays_median": round(float(np.median(keeps)), 1) if keeps else None,
            "uptrend_rate": round(float(np.mean(ups)), 3) if ups else None,
        }

    print("\n[要約] ホライズン内の値動き")
    for tag, s in summary.items():
        lbl = "正例" if tag == "positive" else "負例"
        print(f"  {lbl}: 最大上昇率 中央値 {s['maxGain_median']}% "
              f"(四分位 {s['maxGain_p25']}〜{s['maxGain_p75']}%) / "
              f"終端リターン 中央値 {s['endGain_median']}% "
              f"/ 終端がプラスの割合 {s['endGain_positive_rate']}")
        print(f"        維持日数 中央値 {s['keepDays_median']}日 / "
              f"終了時に上昇トレンドの割合 {s['uptrend_rate']}")

    out = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "definition": {
            "riseHorizon": RISE_HORIZON,
            "riseThreshold": round(RISE_THRESHOLD * 100, 1),
            "population": POPULATION,
            "highWeeks": round(HIGH_WINDOW / 245 * 52),
            "keepDays": DEFAULT_RISE.keep_days,
            "endRatio": (round(DEFAULT_RISE.end_ratio * 100, 1)
                         if DEFAULT_RISE.end_ratio is not None else None),
            "endWindow": DEFAULT_RISE.end_window,
            "trend": (f"MA{DEFAULT_RISE.trend_short}>=MA{DEFAULT_RISE.trend_long}"
                      if DEFAULT_RISE.require_uptrend else None),
            "note": (f"母集団 = {round(HIGH_WINDOW / 245 * 52)}週高値を更新した日"
                     "（高値ベース、連続更新は初回のみ）。"
                     f"正例 = 到達（先{RISE_HORIZON}営業日以内に終値で"
                     f"+{RISE_THRESHOLD*100:.0f}%以上）かつ継続（{DEFAULT_RISE.name}）"),
        },
        "datasetStats": {
            "n": int(len(ds)),
            "positiveRate": round(float(ds["label"].mean()), 4),
            "from": ds["Date"].min().date().isoformat(),
            "to": ds["Date"].max().date().isoformat(),
        },
        "summary": summary,
        "buckets": bucket_stats,
        "compare": {
            "F": {"name": CFG_LOOSE.name, "keepDays": CFG_LOOSE.keep_days,
                  "endRatio": round(CFG_LOOSE.end_ratio * 100, 1)},
            "G": {"name": CFG_ADOPTED.name, "keepDays": CFG_ADOPTED.keep_days,
                  "endRatio": round(CFG_ADOPTED.end_ratio * 100, 1)},
        },
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"\n[done] {args.out} ({os.path.getsize(args.out)/1e6:.2f}MB / {len(cases)}ケース)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
