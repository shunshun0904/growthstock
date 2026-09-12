#!/usr/bin/env python3
"""
「いつ売るか」と「どう候補を選ぶか」を実測して報告を書く。

  python3 research/exit_timing_eval.py

入口は Phase 1 の結論どおり翌営業日の寄りに固定し、出口だけを動かす。
設計は docs/ENTRY_TIMING_DESIGN.md、出力は docs/EXIT_TIMING.md。

答えを出す問い
--------------
  1. 既存ラベルは「3ヶ月のどこかで到達」を見ている。実際いつ売るのが一番取れるか
  2. 「その日の最上位1件」が「スコア上位10%」より弱いのはなぜか
  3. 資金回転まで入れると答えは変わるか
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
import entry_policy as EP  # noqa: E402
import entry_policy_eval as EPE  # noqa: E402
import exit_timing as X  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
MODEL_DIR = os.path.join(HERE, "model")
OUT_JSON = os.path.join(HERE, "exit_timing.json")
OUT_MD = os.path.join(ROOT, "docs", "EXIT_TIMING.md")

#: 「その日の上位N件」が全体のどのあたりにいるかを測る基準
OVERLAP_Q = 0.90


def load_topix(data_dir: str) -> Optional[pd.DataFrame]:
    """
    TOPIX の日次終値。無ければ超過リターンを出さずに続ける。

    長く持つほど成績が上がるのは、地合いが上がっているからかもしれない。
    指数を引かないとそこが分けられないので、取れるなら必ず使う。
    """
    try:
        tp = B.load_parts("topix", data_dir)
    except SystemExit as exc:
        print(f"[topix] 読めないので超過リターンは出さない: {exc}")
        return None
    tp = tp.copy()
    tp["Date"] = pd.to_datetime(tp["Date"])
    tp = tp.dropna().drop_duplicates("Date", keep="last").sort_values("Date")
    print(f"[topix] {len(tp):,}日 {tp['Date'].min().date()}〜{tp['Date'].max().date()}")
    return tp.reset_index(drop=True)


def load(data_dir: str, model_dir: str, min_trading_value: Optional[float]):
    bars = B.load_parts("bars", data_dir)
    n_bars = len(bars)
    print("[panel] 株価系の指標を算出")
    panel = B.price_panel(bars)
    panel["open"] = (panel["AdjO"].fillna(panel["O"])
                     if "AdjO" in panel.columns else panel["O"])
    print("[panel] 78週高値の更新日を判定")
    panel = B.mark_new_highs(panel)
    # forward_matrix が位置で引けるように、整列してから通し番号を振る。
    # build_events は同じキーで並べ直すだけなので、番号はそのまま使える。
    panel = panel.sort_values(["Code", "Date"]).reset_index(drop=True)
    panel["_pos"] = np.arange(len(panel))

    print("[events] 高値更新日を取り出す")
    ev, funnel = EP.build_events(panel, min_trading_value=min_trading_value)
    for k, v in funnel.items():
        print(f"  {k}: {v:,}件")
    ev, score_info = EPE.attach_scores(ev, model_dir)
    return panel, ev, funnel, score_info, n_bars


def restrict_to_full_window(panel: pd.DataFrame, ev: pd.DataFrame,
                            days: int) -> tuple:
    """
    先を days 営業日ぶん持っているイベントだけに絞る。

    保有日数ごとの成績を比べるので、母集団を揃えないと
    「長く持つほうが良い」が「最近のイベントが抜けただけ」と区別できない。
    """
    remaining = panel.groupby("Code", sort=False).cumcount(
        ascending=False).to_numpy()
    have = remaining[ev["_pos"].to_numpy(dtype=int)] >= days
    kept = ev[have].reset_index(drop=True)
    print(f"[window] 先{days}営業日を持つイベント: "
          f"{len(ev):,} -> {len(kept):,}件")
    return kept, int(len(ev) - len(kept))


def evaluate_subsets(panel: pd.DataFrame, ev: pd.DataFrame, fee_pct: float,
                     test_start: str,
                     topix: Optional[pd.DataFrame] = None) -> List[Dict]:
    F = X.forward_matrix(panel, ev, days=X.FORWARD_DAYS)
    BM = (X.benchmark_matrix(panel, ev, topix, days=X.FORWARD_DAYS)
          if topix is not None else None)
    entry = ev["open1"].to_numpy(dtype=float)
    need, _ = B.rise_thresholds(ev["vol_20d"])
    need = need.to_numpy(dtype=float)
    d = pd.to_datetime(ev["Date"])
    is_test = (d >= pd.Timestamp(test_start)).to_numpy()

    thr = (float(ev["score"].quantile(OVERLAP_Q))
           if "score" in ev.columns and ev["score"].notna().any() else None)

    out = []
    for spec in X.subset_masks(ev):
        m = spec["mask"]
        if m.sum() < 100:
            continue
        sec: Dict = {"name": spec["name"], "kind": spec["kind"]}
        sec["profile"] = X.subset_profile(ev, m)
        if thr is not None and spec["kind"] == "top_n":
            sec["profile"]["in_top10pct"] = round(
                100.0 * float((ev["score"][m] >= thr).mean()), 1)
        sec["exits"] = X.compare_exits(F[m], entry[m], need[m], fee_pct,
                                       bm=BM[m] if BM is not None else None)
        sec["peak60"] = X.peak_profile(F[m], entry[m], 60, fee_pct)
        sec["peak120"] = X.peak_profile(F[m], entry[m], X.FORWARD_DAYS, fee_pct)
        # 形が期間で変わらないかを見る。変わるなら「一番良い日数」は選べない
        sec["by_period"] = []
        for tag, sel in (("訓練期間", m & ~is_test), ("テスト期間", m & is_test)):
            if sel.sum() < 100:
                continue
            rows = X.fixed_horizon(F[sel], entry[sel], fee_pct,
                                   bm=BM[sel] if BM is not None else None)
            sec["by_period"].append({"period": tag, "n": int(sel.sum()),
                                     "rows": rows})
        out.append(sec)
        print(f"  {spec['name']}: {int(m.sum()):,}件")
    return out


def headline(sections: List[Dict]) -> Dict:
    """要点を数字から機械的に当てる。文章で足したり引いたりしない。"""
    def best(rows, key):
        c = [r for r in rows if r.get("n") and np.isfinite(r.get(key, np.nan))]
        return max(c, key=lambda r: r[key]) if c else None

    out: Dict = {}
    for sec in sections:
        b1 = best(sec["exits"], "mean")
        b2 = best(sec["exits"], "per_month")
        b3 = best(sec["exits"], "excess_mean")
        out[sec["name"]] = {
            "best_excess": {"rule": b3["rule"], "excess_mean": b3["excess_mean"],
                            "hold_days": b3["hold_days"]} if b3 else None,
            "best_per_trade": {"rule": b1["rule"], "mean": b1["mean"],
                               "hold_days": b1["hold_days"]} if b1 else None,
            "best_per_month": {"rule": b2["rule"], "per_month": b2["per_month"],
                               "mean": b2["mean"],
                               "hold_days": b2["hold_days"]} if b2 else None,
            "peak60_mean": sec["peak60"].get("peak_mean"),
            "peak60_day_median": sec["peak60"].get("day_median"),
        }
    return out


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def _t(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _num(v, fmt="{:+.2f}"):
    """数字が無い欄は「—」にする。nan と書くと値があるように見える。"""
    try:
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def write_md(rep: Dict, path: str) -> None:
    L: List[str] = []
    A = L.append
    A("# いつ売るか / どう候補を選ぶか（日足）")
    A("")
    A("`research/exit_timing_eval.py` の出力。"
      "入口は [Phase 1](ENTRY_POLICY_PHASE1.md) の結論どおり"
      "**翌営業日の寄り成行に固定**し、出口だけを動かしている。")
    A("")
    A(f"- 実行日時: {rep['ranAt']}")
    A(f"- 日次バー: {rep['bars_rows']:,}行")
    A(f"- 手数料: 往復 {rep['fee_pct']}%")
    A(f"- 先読み: {X.FORWARD_DAYS}営業日（約1年）")
    A(f"- 母集団: 先{X.FORWARD_DAYS}営業日ぶんの値動きを持つイベントに揃えた"
      f"（{rep['dropped_recent']:,}件が期間不足で外れた）")
    A("")
    A("**保有日数ごとに母集団を変えていない。** 変えると「長く持つほうが良い」と"
      "「最近のイベントが抜けただけ」を区別できなくなる。")
    A("")

    A("## 要点")
    A("")
    body = []
    for name, h in rep["headline"].items():
        b1, b2, b3 = h["best_per_trade"], h["best_per_month"], h.get("best_excess")
        body.append([
            name,
            f"{b3['rule']}（{_num(b3['excess_mean'])}pt）" if b3 else "—",
            f"{b1['rule']}（{_num(b1['mean'])}%）" if b1 else "—",
            f"{b2['rule']}（{_num(b2['per_month'])}%/月）" if b2 else "—",
            f"{_num(h['peak60_mean'])}%",
            f"{h['peak60_day_median']:.0f}日目" if h["peak60_day_median"] else "—",
        ])
    A(_t(["候補の選び方", "**TOPIX超過が最大の出口**", "1件あたりが最大の出口",
          "時間あたりが最大の出口", "60日以内の最大値（後知恵）",
          "最大値が付く日の中央値"], body))
    A("")
    A("**最初の列を見ること。** 素のリターンは「長く持つほど良い」と必ず出る。"
      "その間の市場全体の上昇が乗るためで、銘柄選定の効き目ではない。"
      "同じ期間の TOPIX を引いた超過リターンが、この選定で取れている分である。")
    A("")
    A("「後知恵」の列は、最大値で売れたらいくらだったかである。"
      "**実現できない上限**であり、実際の出口がそこからどれだけ離れているかを測る物差しとして出す。")
    A("")

    A("## 候補の選び方（部分集合の素性）")
    A("")
    A("成績を比べる前に、そもそも中身が違わないかを見る。"
      "**スコアの水準が違えば成績が違うのは当たり前**なので、先に並べる。")
    A("")
    body = []
    for sec in rep["sections"]:
        p = sec["profile"]
        body.append([sec["name"], f"{p['n']:,}",
                     _num(p.get("per_year"), "{:.0f}"),
                     _num(p.get("score_mean"), "{:.4f}"),
                     _num(p.get("label_rate"), "{:.1f}") + ("%" if
                         np.isfinite(p.get("label_rate", np.nan)) else ""),
                     f"{p['in_top10pct']:.0f}%" if "in_top10pct" in p else "—"])
    A(_t(["選び方", "件数", "年あたり", "平均スコア", "ラベル正例率",
          "全体の上位10%に入っていた割合"], body))
    A("")
    A("最後の列が「その日の上位N件」の正体である。"
      "候補が弱い日でも上位N件は必ず選ばれるので、日をまたぐと低いスコアが混ざる。")
    A("")

    for sec in rep["sections"]:
        A(f"## 出口の比較 — {sec['name']}（{sec['profile']['n']:,}件）")
        A("")
        body = []
        for r in sec["exits"]:
            body.append([
                r["rule"], f"{r['n']:,}",
                _num(r.get("excess_mean")),
                _num(r.get("excess_win"), "{:.1f}") + (
                    "%" if np.isfinite(r.get("excess_win", np.nan)) else ""),
                _num(r.get("mean")), _num(r.get("median")),
                f"{r.get('win_rate', float('nan')):.1f}%",
                _num(r.get("p05")),
                f"{r.get('hold_days', float('nan')):.0f}",
                _num(r.get("per_month")),
                f"{r['hit_rate']:.0f}%" if r.get("hit_rate") is not None else "—",
            ])
        A(_t(["出口", "件数", "**TOPIX超過**", "超過勝率", "素の平均", "中央値",
              "勝率", "下側5%", "平均保有日数", "1ヶ月換算", "到達率"], body))
        A("")
        pk = sec["peak60"]
        A(f"60日以内の最大値で売れた場合（後知恵）: 平均 {_num(pk.get('peak_mean'))}% / "
          f"最大値が付く日は中央値 {pk.get('day_median', float('nan')):.0f}日目 "
          f"（四分位 {pk.get('day_q1', float('nan')):.0f}〜"
          f"{pk.get('day_q3', float('nan')):.0f}日目）")
        A("")
        if sec.get("by_period"):
            A("### 期間で形が変わらないか")
            A("")
            ks = [r["k"] for r in sec["exits"] if r["kind"] == "fixed"]
            head = ["期間", "件数", "指標"] + [f"{k}日" for k in ks]
            rows = []
            for pr in sec["by_period"]:
                by = {r["k"]: r for r in pr["rows"]}
                rows.append([pr["period"], f"{pr['n']:,}", "TOPIX超過"]
                            + [_num(by[k].get("excess_mean")) if k in by else "—"
                               for k in ks])
                rows.append(["", "", "素の平均"]
                            + [_num(by[k]["mean"]) if k in by else "—" for k in ks])
            A(_t(head, rows))
            A("")
            A("**一番良い日数を、この表を見てから選んではいけない。**"
              "選ぶなら訓練期間だけで決めて、テスト期間で確かめること。"
              "ここで見るのは順位ではなく、山の位置がだいたい同じかどうか。")
            A("")

    A("## 最大値は何日目に付くか")
    A("")
    for sec in rep["sections"]:
        pk = sec.get("peak120") or {}
        hist = pk.get("day_hist") or []
        if not hist:
            continue
        A(f"### {sec['name']}（{X.FORWARD_DAYS}営業日以内）")
        A("")
        A(_t(["期間（営業日）", "件数", "割合"],
             [[f"{b['from']}〜{b['to']}", f"{b['n']:,}", f"{b['share']:.1f}%"]
              for b in hist]))
        A("")

    A("## この報告の限界")
    A("")
    A("- 「1ヶ月換算」は、売った資金をすぐ次の1件に回せる前提。"
      "候補は1日数件しか出ないので、常に回せるとは限らない")
    A("- 損切りを入れていない。下側5%はそのままの数字である")
    A("- 到達しきい値は基準日終値から測る（ラベルの定義に合わせている）。"
      "翌日に窓を開けて寄った場合、エントリー価格から見たしきい値はその分低くなる")
    A("- 分割調整後の終値で計算している。配当は含まない")
    A("- 一番良い保有日数をこのデータから選ぶこと自体が選択である。"
      "採用するなら訓練期間だけで決めて、テスト期間で確かめること")
    A("- 超過リターンは「銘柄（手数料込み）− TOPIX（手数料なし）」。"
      "指数側にコストを掛けていないぶん、超過はわずかに銘柄へ不利な側に出る")
    A("- TOPIX は基準日終値を起点にしている。銘柄の入口は翌営業日の寄りなので、"
      "翌日の窓の分だけ起点がずれる。その差は銘柄側のリターンに含まれる")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--test-start", default="2024-10-01")
    ap.add_argument("--fee", type=float, default=0.05)
    ap.add_argument("--min-trading-value", type=float, default=0.1)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    panel, ev, funnel, score_info, n_bars = load(
        args.data_dir, args.model_dir, args.min_trading_value)
    topix = load_topix(args.data_dir)
    ev, dropped = restrict_to_full_window(panel, ev, X.FORWARD_DAYS)
    if ev.empty:
        raise SystemExit("イベントが0件。生データを確認してください")

    print("\n[eval] 部分集合ごとに出口を比べる")
    sections = evaluate_subsets(panel, ev, args.fee, args.test_start, topix)

    rep = {
        "ranAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bars_rows": n_bars,
        "funnel": funnel,
        "dropped_recent": dropped,
        "fee_pct": args.fee,
        "test_start": args.test_start,
        "score_info": score_info,
        "benchmark": "TOPIX" if topix is not None else None,
        "sections": sections,
    }
    rep["headline"] = headline(sections)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
    write_md(rep, args.out_md)
    print(f"\n[write] {args.out_json}\n[write] {args.out_md}")
    for name, h in rep["headline"].items():
        b3 = h.get("best_excess")
        print(f"  {name}: TOPIX超過が最大 = {b3['rule'] if b3 else '—'}"
              f" ({_num(b3['excess_mean']) if b3 else '—'}pt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
