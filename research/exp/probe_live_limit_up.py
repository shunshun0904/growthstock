#!/usr/bin/env python3
"""
見るだけの集計: 実運用の候補（画面の予測の履歴）が、買ってからストップ高になったかを、実験99 と同じ判定で数える。判定はしない。

運用者の依頼（2026-10-16）「直近の実運用の候補についても同じ判定で数えてください」。実験99 は過去の out-of-fold（2021〜2026-08）で、
今の規則の行の 3.9% が買ってから20営業日にストップ高（同じ窓・同じ帯の候補は 5.7%）だった。こちらは運用開始（2026-09-11）からの
画面の候補を git の履歴（public/data/predictions.json の版。live_track と同じ復元）から取り、保存済みの日足で同じ判定をする。

数えるもの（選定日 = 高値更新日、買うのは翌営業日の寄り。ストップ高 = 高値 ≥ 前営業日の終値 + 制限値幅。price_limits）
  - 候補全体（その日に画面に出た候補 = 発火）
  - 以前の規則（1.2σ の3本 90以上。運用開始から）と今の規則（1.5σ の3本 95以上。1.5σ の百分位が予測に入った 2026-10-13 から）を
    満たした候補、その上位2件（買う銘柄）
  - 買ってから20営業日目まで（途中の候補は日足のある日まで）にストップ高の日があったか、最初の日、張り付き、+10% に届いたか
  - 時価総額の帯（画面の marketCap。億円）ごとの内訳
実運用の追跡（docs/LIVE_TRACKING.md）と同じく、銘柄コード・名前・選定日は出す。買値（日足の生の値）は出さない。

    exp=probe_live_limit_up.py
    args="--since 2026-09-11"     （既定）
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import price_limits as PL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SINCE = "2026-09-11"
#: (記号, 表示, モデル, 線)
RULES = (("PREV", "以前の規則: 1.2σ の3本 90以上", LT.PREV_MODELS, LT.PREV_AGREE),
         ("K15", "今の規則: 1.5σ の3本 95以上", LT.RULE_MODELS, LT.AGREE_PCT))
CAP_EDGES = (100.0, 300.0, 1000.0, 3000.0)
CAP_LABELS = ("〜100億", "100〜300億", "300〜1,000億", "1,000〜3,000億", "3,000億〜")


def cap_band(mc) -> pd.Series:
    x = pd.to_numeric(pd.Series(mc), errors="coerce")
    return pd.Series(np.digitize(x.fillna(-1.0), CAP_EDGES), index=x.index, dtype=float).where(x.notna())


def market_caps(versions) -> pd.DataFrame:
    """候補ごとの画面の時価総額（億円）。同じ (Date, Code) は最後の版の値。"""
    rows = {}
    for v in versions:
        for c in v["data"].get("candidates", []):
            if c.get("date"):
                rows[(pd.Timestamp(c["date"]), str(c.get("jqCode") or c.get("code") or ""))] = c.get("marketCap")
    return pd.DataFrame([{"Date": d, "Code": c, "marketCap": pd.to_numeric(m, errors="coerce")} for (d, c), m in rows.items()])


def collect(repo: str, data_dir: str, since: str) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """候補1行ごとに、規則の印・順位・ストップ高・出口の列を付けた表と、日足の最終日。"""
    LT.ensure_history(repo)
    versions = LT.load_versions(repo)
    days = LT.trading_days(data_dir)
    rows = LT.live_rows(versions, days)
    rows = rows[rows["Date"] >= pd.Timestamp(since)].copy()
    if not len(rows):
        return rows, None
    out = rows.copy()
    for key, _, models, line in RULES:
        r, _, picks = LT.decide(rows, agree=line, min_break=1, top_k=LT.TOP_K, models=models)
        flag = r.set_index(["Date", "Code"])["passed"]
        rank = picks.set_index(["Date", "Code"])["rank"]
        idx = pd.MultiIndex.from_frame(out[["Date", "Code"]])
        out[f"pass_{key}"] = flag.reindex(idx).fillna(False).to_numpy(dtype=bool)
        out[f"rank_{key}"] = rank.reindex(idx).to_numpy()
        out[f"pmin_{key}"] = r.set_index(["Date", "Code"])["p_min"].reindex(idx).to_numpy()
    out["n_break"] = out.groupby("Date")["Code"].transform("size")
    out = out.merge(market_caps(versions), on=["Date", "Code"], how="left")
    out["cap_band"] = cap_band(out["marketCap"]).to_numpy()
    start = pd.Timestamp(since) - pd.Timedelta(days=400)
    bars = LT.load_bars(data_dir, start=start, extra=("H", "L", "C"))
    flags = PL.flag_limits(bars)
    arrays = PL.code_arrays(flags)
    ev = PL.limit_events(arrays, out[["Code", "Date"]], hold=LT.HOLD, past=250, partial=True)
    out = out.merge(ev, on=["Code", "Date"], how="left")
    fw = LT.forward(bars, out[["Code", "Date"]], target=LT.TAKE_PROFIT, hold=LT.HOLD)
    out = out.merge(fw[["Code", "Date", "status", "hit_day", "ret", "ret_now", "elapsed"]], on=["Code", "Date"], how="left")
    return out, pd.Timestamp(bars["Date"].max())


def line_of(r) -> str:
    ul = r["ul_fwd"]
    if not np.isfinite(ul):
        mark = "日足なし"
    elif ul == 1:
        mark = f"ストップ高 {int(r['ul_fwd_first'])}日目" + ("（張り付き）" if r.get("ulc_fwd") == 1 else "")
    else:
        mark = "なし"
    days = int(r["fwd_days"]) if np.isfinite(r["fwd_days"]) else 0
    st = r["status"] if isinstance(r["status"], str) else ""
    if st == LT.HIT:
        exitv = f"+10% 到達 {int(r['hit_day'])}日目"
    elif st == "満了":
        exitv = f"満了 {r['ret']:+.1f}%"
    elif st == "保有中":
        exitv = f"途中 {r['ret_now']:+.1f}%"
    else:
        exitv = st
    cap = f"{r['marketCap']:,.0f}億" if np.isfinite(r["marketCap"]) else "—"
    return (f"  {pd.Timestamp(r['Date']).date()}  {r['code']:<6}{LT._l(r['name'], 18)}{cap:>9}  3本の最小 {r['pmin']:5.1f}  "
            f"{days:>2}営業日  {LT._l(mark, 22)}{exitv}")


def summary(g: pd.DataFrame, name: str) -> str:
    n = len(g)
    ok = g["ul_fwd"].notna()
    ul = int((g["ul_fwd"] == 1).sum())
    full = g["fwd_ok"] == 1
    ulc = int((g["ulc_fwd"] == 1).sum())
    cap = []
    for k, lab_ in enumerate(CAP_LABELS):
        m = g["cap_band"] == k
        if m.any():
            cap.append(f"{lab_} {int((g.loc[m, 'ul_fwd'] == 1).sum())}/{int(m.sum())}")
    hit = int((g["status"] == LT.HIT).sum())
    return (f"  {LT._l(name, 34)}{n:>3}件  判定できた {int(ok.sum())}件（20営業日そろった {int(full.sum())}件）  "
            f"買ってからストップ高 {ul}件（{ul / ok.sum() * 100 if ok.sum() else float('nan'):.1f}%。張り付き {ulc}）  +10% 到達 {hit}件"
            + ("  帯: " + " / ".join(cap) if cap else ""))


def report(out: pd.DataFrame, since: str) -> None:
    if not len(out):
        print(f"{since} 以降の候補が無い")
        return
    last = pd.Timestamp(out["Date"].max()).date()
    n_days = out["Date"].nunique()
    print(f"\n■ 実運用の候補 {since} 〜 {last}（{n_days}日・{len(out)}件）。買ってから20営業日目まで（途中の候補は日足のある日まで）の"
          f"ストップ高を、実験99 と同じ判定（高値 ≥ 前営業日の終値 + 制限値幅）で数える")
    print(summary(out, "候補全体（発火）"))
    for key, name, _, _ in RULES:
        g = out[out[f"pass_{key}"]]
        print(summary(g, name + "・満たした候補"))
        print(summary(g[g[f"rank_{key}"].notna()], name + "・上位2件"))
    print("\n■ 規則を満たした候補（いずれかの規則。選定日順）")
    both = out[out["pass_PREV"] | out["pass_K15"]].sort_values(["Date", "pmin_K15", "pmin_PREV"], ascending=[True, False, False])
    for _, r in both.iterrows():
        pm = r["pmin_K15"] if np.isfinite(r["pmin_K15"]) else r["pmin_PREV"]
        which = ("今" if r["pass_K15"] else "") + ("旧" if r["pass_PREV"] else "")
        rk = r["rank_K15"] if np.isfinite(r["rank_K15"]) else r["rank_PREV"]
        tag = f"{which}{'・上位' + str(int(rk)) if np.isfinite(rk) else ''}"
        print(line_of({**r, "pmin": pm}) + f"  [{tag}]")
    ul_all = out[out["ul_fwd"] == 1]
    print(f"\n■ 候補全体のうち、買ってからストップ高になった {len(ul_all)}件（規則を満たした候補は [印]）")
    for _, r in ul_all.sort_values("Date").iterrows():
        pm = r["pmin_K15"] if np.isfinite(r["pmin_K15"]) else r["pmin_PREV"]
        which = ("今" if r["pass_K15"] else "") + ("旧" if r["pass_PREV"] else "")
        print(line_of({**r, "pmin": pm}) + (f"  [{which}]" if which else ""))
    print("\n  注意: 途中の候補は日足のある日までで数えている。買値は出していない。判定は実験99 と同じ自前の判定（API のフラグではない）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実運用の候補が買ってからストップ高になったかを数える（見るだけ）")
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    args = ap.parse_args(argv)
    out, last_bar = collect(args.repo, args.data_dir, args.since)
    if last_bar is not None:
        print(f"日足の最終日: {last_bar.date()}")
    report(out, args.since)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
