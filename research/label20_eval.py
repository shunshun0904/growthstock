#!/usr/bin/env python3
"""
ラベルを20営業日ベースに作り直したときの集計と、入れ替わったチャートを出す。

  python3 research/label20_eval.py

**学習はしない。** 定義を変えると何がどう入れ替わるのかを、
数字とチャートで先に確かめるためのもの。

出力
  docs/LABEL20_EDA.md             集計
  research/samples/label20_samples.json   チャートデータ
  research/viewer/label20_review.html     ブラウザで見る形
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import label20 as L  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
OUT_MD = os.path.join(ROOT, "docs", "LABEL20_EDA.md")
OUT_JSON = os.path.join(HERE, "samples", "label20_samples.json")
OUT_HTML = os.path.join(HERE, "viewer", "label20_review.html")
TEMPLATE = os.path.join(HERE, "viewer", "label20_template.html")


def build(data_dir: str, min_trading_value: Optional[float]):
    bars = B.load_parts("bars", data_dir)
    print("[panel] 株価系の指標を算出")
    panel = B.price_panel(bars)
    print("[panel] 78週高値の更新日を判定")
    panel = B.mark_new_highs(panel)

    # to_numpy() が読み取り専用のビューを返すことがあるので、必ず複製する
    m = (panel["is_fresh_break"] == True).to_numpy().copy()   # noqa: E712
    print(f"[sample] 新規ブレイク: {int(m.sum()):,}件")
    m &= panel["high52w"].notna().to_numpy()
    if min_trading_value is not None:
        m &= (panel["tv_ma20"] >= min_trading_value).fillna(False).to_numpy()
    print(f"[sample] 流動性・高値の条件を満たす: {int(m.sum()):,}件")

    print(f"[label] 現行: {L.CFG_60.name}")
    d60 = L.labels_for(panel, L.CFG_60, m)
    print(f"[label] 新案: {L.CFG_20.name}")
    d20 = L.labels_for(panel, L.CFG_20, m)

    keep = [c for c in ("Code", "Date", "close", "vol_20d", "tv_ma20",
                        "r_high", "volume_trend") if c in panel.columns]
    ev = panel.loc[m, keep].reset_index(drop=True)
    d60 = d60.reset_index(drop=True)
    d20 = d20.reset_index(drop=True)
    return panel, ev, d60, d20


def aggregate(ev: pd.DataFrame, d60: pd.DataFrame, d20: pd.DataFrame) -> Dict:
    l60, l20 = d60["label"], d20["label"]
    rep: Dict = {
        "config": {
            "cfg60": {"name": L.CFG_60.name, "horizon": L.CFG_60.horizon,
                      "k": L.CFG_60.vol_norm_k, "endWindow": L.CFG_60.end_window,
                      "trend": [L.CFG_60.trend_short, L.CFG_60.trend_long]},
            "cfg20": {"name": L.CFG_20.name, "horizon": L.CFG_20.horizon,
                      "k": L.CFG_20.vol_norm_k, "endWindow": L.CFG_20.end_window,
                      "trend": [L.CFG_20.trend_short, L.CFG_20.trend_long]},
        },
        "n_events": len(ev),
        "funnel": [L.funnel(d60, "現行（60営業日）"), L.funnel(d20, "新案（20営業日）")],
        "transition": L.transition(l60, l20),
    }
    ms = L.masks(l60, l20)
    rep["profiles"] = {k: L.profile(ev, d60, d20, v) for k, v in ms.items()}
    year = pd.to_datetime(ev["Date"]).dt.year
    rep["by_year"] = L.by_group(ev, l60, l20, year, "年")
    rep["by_vol"] = L.by_group(ev, l60, l20, L.vol_bands(ev["vol_20d"]), "ボラ帯")
    return rep, ms


def samples(panel: pd.DataFrame, ev: pd.DataFrame, ms: Dict[str, np.ndarray],
            n_per: int, seed: int) -> List[Dict]:
    names: Dict[str, str] = {}
    p = os.path.join(ROOT, "public", "data", "stocks.json")
    if os.path.exists(p):
        try:
            for s in json.load(open(p, encoding="utf-8")).get("stocks", []):
                names[s.get("jqCode", "")] = s.get("name", "")
        except (ValueError, OSError):
            pass

    cases: List[Dict] = []
    for bucket in ("pos_neg", "neg_pos"):
        idx = L.pick(ev, ms[bucket], n_per, seed)
        print(f"[chart] {L.BUCKET_JA[bucket]}: {int(ms[bucket].sum()):,}件から "
              f"{len(idx)}件を抽出")
        for i in idx:
            row = ev.iloc[i]
            c = L.build_case(panel, str(row["Code"]),
                             pd.Timestamp(row["Date"]),
                             names.get(str(row["Code"]), ""), bucket)
            if c:
                cases.append(c)
    return cases


# --------------------------------------------------------------------------- #

def _t(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v, fmt="{:+.2f}"):
    try:
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def write_md(rep: Dict, path: str) -> None:
    L_: List[str] = []
    A = L_.append
    c = rep["config"]
    A("# ラベルを20営業日ベースに変える — 変更前後の集計")
    A("")
    A("`research/label20_eval.py` の出力。**学習はまだしていない。**"
      "定義を変えると何が入れ替わるのかを先に確かめるためのもの。")
    A("")
    A(f"- 実行日時: {rep['ranAt']}")
    A(f"- 母集団: 78週高値の更新日（直前20営業日に更新なし・流動性条件つき）"
      f" {rep['n_events']:,}件")
    A("")
    A("## 何を変えたか")
    A("")
    A(_t(["", "現行", "新案"],
         [["地平", f"{c['cfg60']['horizon']}営業日（約3ヶ月）",
           f"**{c['cfg20']['horizon']}営業日（約1ヶ月）**"],
          ["到達しきい値", "1.2σ√60", "**1.2σ√20**"],
          ["同・σ=3%/日のとき", "+27.9%", "**+16.1%**"],
          ["終盤の水準", "t+56〜t+60 の5日平均 ≧ 到達の半分",
           "t+16〜t+20 の5日平均 ≧ 到達の半分"],
          ["同・σ=3%/日のとき", "+13.9%", "**+8.0%**"],
          ["トレンド", "t+60 で MA20 ≧ MA60", "**t+20 で MA5 ≧ MA20**"],
          ["定義名", f"`{c['cfg60']['name']}`", f"`{c['cfg20']['name']}`"]]))
    A("")
    A("持続の条件（終盤・トレンド）は落としていない。"
      "突発的な高騰を正例にしないための条件なので、地平を縮めてもそこは変えない。"
      "移動平均だけは、20営業日で判定するのに MA20≧MA60 では動かなすぎるため短い組に替えた。")
    A("")

    A("## 正例の作られ方（ファネル）")
    A("")
    A(_t(["定義", "判定できた件数", "到達", "＋終盤", "＋トレンド＝正例",
          "終盤で落ちた", "トレンドで落ちた"],
         [[f["name"], f"{f['n_determined']:,}",
           f"{f['reached']:,}（{f['reached_pct']:.1f}%）",
           f"{f['reached_end']:,}（{f['reached_end_pct']:.1f}%）",
           f"**{f['positive']:,}（{f['positive_rate']:.1f}%）**",
           f"{f['dropped_by_end']:,}", f"{f['dropped_by_trend']:,}"]
          for f in rep["funnel"]]))
    A("")

    tr = rep["transition"]
    A("## 入れ替わり（両方とも判定できた行だけ）")
    A("")
    A(_t(["", "20日で正例", "20日で負例", "計"],
         [["**60日で正例**",
           f"{tr['pos_pos']:,}（{tr['pos_pos_pct']:.1f}%）",
           f"**{tr['pos_neg']:,}（{tr['pos_neg_pct']:.1f}%）**",
           f"{tr['pos_pos'] + tr['pos_neg']:,}"],
          ["**60日で負例**",
           f"**{tr['neg_pos']:,}（{tr['neg_pos_pct']:.1f}%）**",
           f"{tr['neg_neg']:,}（{tr['neg_neg_pct']:.1f}%）",
           f"{tr['neg_pos'] + tr['neg_neg']:,}"],
          ["計", f"{tr['pos_pos'] + tr['neg_pos']:,}",
           f"{tr['pos_neg'] + tr['neg_neg']:,}", f"{tr['n']:,}"]]))
    A("")
    A(f"入れ替わったのは **{tr['flipped']:,}件（{tr['flipped_pct']:.1f}%）**。"
      "太字の2つがチャートで見る対象。")
    A("")

    A("## 群ごとの素性")
    A("")
    A(_t(["群", "件数", "σ(%)", "到達しきい値60", "到達しきい値20",
          "最大上昇60(中央値)", "最大上昇20(中央値)",
          "終盤水準60(中央値)", "終盤水準20(中央値)"],
         [[L.BUCKET_JA[k], f"{p['n']:,}", _n(p["vol_20d"], "{:.2f}"),
           _n(p["need_60"], "{:.1f}"), _n(p["need_20"], "{:.1f}"),
           _n(p["max_gain_60"], "{:+.1f}"), _n(p["max_gain_20"], "{:+.1f}"),
           _n(p["end_level_60"], "{:+.1f}"), _n(p["end_level_20"], "{:+.1f}")]
          for k, p in rep["profiles"].items()]))
    A("")

    for key, title in (("by_year", "年別の正例率"), ("by_vol", "ボラ帯別の正例率")):
        rows = rep.get(key) or []
        if not rows:
            continue
        A(f"## {title}")
        A("")
        A(_t(["区分", "件数", "現行60日", "新案20日", "差"],
             [[r["group"], f"{r['n']:,}", f"{r['rate_60']:.1f}%",
               f"{r['rate_20']:.1f}%", f"{r['diff']:+.1f}pt"] for r in rows]))
        A("")

    A("## 入れ替わったチャート")
    A("")
    A(f"`research/viewer/label20_review.html` をブラウザで開く。"
      f"太字の2群から年をまたいで抽出してある（{rep.get('n_cases', 0)}件）。")
    A("")
    A("各チャートに、両方の定義の線を引いてある。")
    A("")
    A("- 縦の帯: 判定期間（20日＝濃い / 60日＝薄い）")
    A("- 横線: 到達しきい値と終盤の水準（20日用・60日用の2組）")
    A("- 移動平均: MA5 / MA20 / MA60")
    A("- 右上のバッジ: 到達・終盤・トレンドの○×を定義ごとに表示")
    A("")

    A("## この集計の限界")
    A("")
    A("- **学習はしていない。** 正例率や入れ替わりの件数が分かっただけで、"
      "新しい定義のほうが当てやすいかは別に確かめる必要がある")
    A("- 判定期間に決算開示が何件入るかは数えていない"
      "（今回の範囲から外した）。ご指摘の確認をするなら別途")
    A("- 移動平均の組を MA20/MA60 から MA5/MA20 に替えているので、"
      "トレンド条件は地平の変更と同時に中身も変わっている。"
      "この2つの効果は分離していない")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")


def write_html(cases: List[Dict], rep: Dict, template: str, out: str) -> None:
    if not os.path.exists(template):
        print(f"[warn] {template} が無いので HTML は作らない")
        return
    with open(template, encoding="utf-8") as fh:
        tpl = fh.read()
    payload = json.dumps({"cases": cases, "summary": rep},
                         ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tpl.replace("__DATA__", payload))
    print(f"[write] {out} ({os.path.getsize(out)/1e6:.1f}MB)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--min-trading-value", type=float, default=0.1)
    ap.add_argument("--n-per-bucket", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-html", default=OUT_HTML)
    args = ap.parse_args(argv)

    panel, ev, d60, d20 = build(args.data_dir, args.min_trading_value)
    rep, ms = aggregate(ev, d60, d20)
    rep["ranAt"] = dt.datetime.now(dt.timezone.utc).isoformat()

    print("\n[chart] 入れ替わった群からチャートを抽出")
    cases = samples(panel, ev, ms, args.n_per_bucket, args.seed)
    rep["n_cases"] = len(cases)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"cases": cases, "summary": rep}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    write_md(rep, args.out_md)
    write_html(cases, rep, TEMPLATE, args.out_html)
    print(f"\n[write] {args.out_json}\n[write] {args.out_md}")
    t = rep["transition"]
    print(f"\n正例率: 60日 {rep['funnel'][0]['positive_rate']:.2f}% -> "
          f"20日 {rep['funnel'][1]['positive_rate']:.2f}%")
    print(f"入れ替わり: {t['flipped']:,}件 ({t['flipped_pct']:.1f}%) "
          f"[正→負 {t['pos_neg']:,} / 負→正 {t['neg_pos']:,}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
