#!/usr/bin/env python3
"""
ストップ高・ストップ安の自前の判定（research/price_limits.py）が、J-Quants のフラグ（/equities/bars/daily の UL / LL）と
どれだけ一致するかを実測する下調べ。

保存済みの日足にはフラグが無い（取り込みの列 jq_bulk.BAR_COLS に UL / LL が無い）ので、実験99 は四本値と制限値幅の表から
判定している。その判定が API のフラグとどれだけ合うか・合わないのはどんな日かを、何日かぶんを API から取って数える。
鍵が要るので Actions（Probe Limit Flags）で回す。

やること
  1. 標本の日（既定: 2017〜2026 の各年 3月・9月の15日前後の営業日と、直近の5営業日）とその前営業日の四本値を API から取る
  2. 前営業日の終値を基準値段にして price_limits.flag_limits で判定し、API の UL / LL と突き合わせる
  3. 出すのは件数だけ（一致・API だけ・自前だけ、食い違いの内訳）。銘柄コードは出さない

  $ JQUANTS_API=... python3 research/probe_limit_flags.py [--days 2024-05-15 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import price_limits as PL  # noqa: E402
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402

BARS = "/equities/bars/daily"
CALENDAR = "/markets/calendar"
#: 標本の日（暦日。営業日でなければ、その日以前でいちばん近い営業日にする）
SAMPLE_DAYS = [f"{y}-{m}-15" for y in range(2017, 2027) for m in ("03", "09")]
RECENT = 5
NUM_COLS = ("O", "H", "L", "C", "Vo", "Va", "AdjO", "AdjH", "AdjL", "AdjC", "AdjFactor")


def trading_days(client: JQuantsClient, start: dt.date, end: dt.date) -> List[dt.date]:
    rows = client.get_paginated(CALENDAR, {"from": start.isoformat(), "to": end.isoformat()})
    out = []
    for r in rows:
        div = str(r.get("HolDiv", r.get("HolidayDivision", "")))
        if div in ("1", "2") and r.get("Date"):
            out.append(dt.date.fromisoformat(str(r["Date"])[:10]))
    return sorted(out)


def pick_days(days: Sequence[dt.date], wanted: Sequence[str], recent: int) -> List[Tuple[dt.date, dt.date]]:
    """(標本の日, その前営業日) の組。標本の日は wanted 以前でいちばん近い営業日。直近 recent 営業日も足す。"""
    arr = list(days)
    pairs = []
    for w in wanted:
        d = dt.date.fromisoformat(w)
        prior = [x for x in arr if x <= d]
        if len(prior) >= 2:
            pairs.append((prior[-1], prior[-2]))
    for k in range(len(arr) - recent, len(arr)):
        if k >= 1:
            pairs.append((arr[k], arr[k - 1]))
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def fetch_day(client: JQuantsClient, day: dt.date) -> pd.DataFrame:
    rows = client.get_paginated(BARS, {"date": day.isoformat()})
    df = pd.DataFrame.from_records(rows)
    if not len(df):
        return df
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    return df


def flag_of(df: pd.DataFrame, col: str) -> pd.Series:
    """API のフラグ列（"0" / "1"。無ければ NaN）。"""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col].astype(str).str.strip()
    return s.map({"1": 1.0, "1.0": 1.0, "True": 1.0, "0": 0.0, "0.0": 0.0, "False": 0.0})


def compare_day(today: pd.DataFrame, prev: pd.DataFrame) -> Dict:
    """1日ぶんの突き合わせ（件数だけ）。"""
    both = pd.concat([prev, today], ignore_index=True)
    f = PL.flag_limits(both[["Date", "Code", "H", "L", "C", "AdjC"]])
    f = f[f["Date"] == today["Date"].iloc[0]].set_index("Code")
    t = today.set_index("Code")
    api_ul = flag_of(t, "UL")
    api_ll = flag_of(t, "LL")
    ours_ul = f["ul"].reindex(t.index).fillna(False).astype(bool)
    ours_ll = f["ll"].reindex(t.index).fillna(False).astype(bool)
    known = f["known"].reindex(t.index).fillna(False).astype(bool)
    base = f["base"].reindex(t.index)
    width = f["width"].reindex(t.index)
    out = {"rows": int(len(t)), "known": int(known.sum()), "api_ul_na": int(api_ul.isna().sum())}
    for name, api, ours in (("ul", api_ul, ours_ul), ("ll", api_ll, ours_ll)):
        a = api == 1
        o = ours.to_numpy()
        out[f"{name}_api"] = int(a.sum())
        out[f"{name}_ours"] = int(o.sum())
        out[f"{name}_both"] = int((a & o).sum())
        out[f"{name}_api_only"] = int((a & ~o).sum())
        out[f"{name}_ours_only"] = int((~a & o & api.notna()).sum())
    # 食い違いの内訳（ストップ高）
    a = (api_ul == 1).to_numpy()
    o = ours_ul.to_numpy()
    h = t["H"].to_numpy(dtype=float)
    up = (base + width).to_numpy(dtype=float)
    gap = (up - h) / width.to_numpy(dtype=float)          # 自前の上限にどれだけ届いていないか（幅に対する比）
    api_only = a & ~o
    out["ul_api_only_no_trade"] = int((api_only & ~known.to_numpy()).sum())
    g = gap[api_only & known.to_numpy()]
    out["ul_api_only_within_2pct"] = int((g <= 0.02).sum())
    out["ul_api_only_within_10pct"] = int(((g > 0.02) & (g <= 0.10)).sum())
    out["ul_api_only_farther"] = int((g > 0.10).sum())
    ours_only = ~a & o & api_ul.notna().to_numpy()
    r = (h - (base.to_numpy(dtype=float) - 0)) / width.to_numpy(dtype=float)   # 高値の上げ幅 ÷ 幅
    rr = r[ours_only]
    out["ul_ours_only_le_1_05"] = int((rr <= 1.05).sum())
    out["ul_ours_only_1_05_to_1_5"] = int(((rr > 1.05) & (rr <= 1.5)).sum())
    out["ul_ours_only_over_1_5"] = int((rr > 1.5).sum())
    return out


def report(results: List[Tuple[dt.date, Dict]]) -> List[str]:
    lines = [f"{'日':<12}{'行':>6}{'判定可':>7}{'API':>6}{'自前':>6}{'一致':>6}{'APIのみ':>8}{'自前のみ':>8}   ストップ安 API/自前/一致"]
    tot: Dict[str, int] = {}
    for d, r in results:
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + v
        lines.append(f"{d.isoformat():<12}{r['rows']:>6}{r['known']:>7}{r['ul_api']:>6}{r['ul_ours']:>6}{r['ul_both']:>6}"
                     f"{r['ul_api_only']:>8}{r['ul_ours_only']:>8}   {r['ll_api']}/{r['ll_ours']}/{r['ll_both']}")
    if tot:
        lines.append(f"{'合計':<12}{tot['rows']:>6}{tot['known']:>7}{tot['ul_api']:>6}{tot['ul_ours']:>6}{tot['ul_both']:>6}"
                     f"{tot['ul_api_only']:>8}{tot['ul_ours_only']:>8}   {tot['ll_api']}/{tot['ll_ours']}/{tot['ll_both']}")
        api, ours, both = tot["ul_api"], tot["ul_ours"], tot["ul_both"]
        lines.append(f"ストップ高: API のフラグ {api}件のうち自前も1 {both}件（{both / api * 100 if api else float('nan'):.1f}%）。"
                     f"自前の {ours}件のうち API も1 {both}件（{both / ours * 100 if ours else float('nan'):.1f}%）")
        lines.append(f"  API だけ {tot['ul_api_only']}件の内訳: 売買の無い日（判定できない）{tot['ul_api_only_no_trade']}・"
                     f"自前の上限に幅の2%以内で届かず {tot['ul_api_only_within_2pct']}・2〜10% {tot['ul_api_only_within_10pct']}・"
                     f"それ以上 {tot['ul_api_only_farther']}")
        lines.append(f"  自前だけ {tot['ul_ours_only']}件の内訳（高値の上げ幅 ÷ 幅）: 1.05以下 {tot['ul_ours_only_le_1_05']}・"
                     f"1.05〜1.5 {tot['ul_ours_only_1_05_to_1_5']}・1.5超 {tot['ul_ours_only_over_1_5']}")
        lla, llo, llb = tot["ll_api"], tot["ll_ours"], tot["ll_both"]
        lines.append(f"ストップ安: API {lla}件のうち自前も1 {llb}件（{llb / lla * 100 if lla else float('nan'):.1f}%）。"
                     f"自前 {llo}件のうち API も1 {llb}件（{llb / llo * 100 if llo else float('nan'):.1f}%）")
        if tot["api_ul_na"]:
            lines.append(f"  注意: UL 列が無い・読めない行 {tot['api_ul_na']}件")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ストップ高の自前の判定と API のフラグの突き合わせ")
    ap.add_argument("--days", nargs="*", default=None, help="標本の日（YYYY-MM-DD）。省略すれば既定の標本")
    ap.add_argument("--recent", type=int, default=RECENT)
    args = ap.parse_args(argv)
    key = resolve_api_key()
    if not key:
        print("[stop] JQUANTS_API が無い")
        return 1
    client = JQuantsClient(key, pause=0.3)
    wanted = args.days if args.days else SAMPLE_DAYS
    today = dt.date.today()
    start = min(dt.date.fromisoformat(w) for w in wanted) - dt.timedelta(days=14)
    try:
        days = trading_days(client, start, today)
    except JQuantsError as exc:
        print(f"[stop] カレンダーが取れない: {str(exc)[:120]}")
        return 1
    pairs = pick_days(days, wanted, args.recent)
    print(f"標本 {len(pairs)}日（{pairs[0][0]} 〜 {pairs[-1][0]}）。API の列を確かめてから数える")
    cache: Dict[dt.date, pd.DataFrame] = {}

    def get(d: dt.date) -> pd.DataFrame:
        if d not in cache:
            cache[d] = fetch_day(client, d)
        return cache[d]

    results = []
    cols_seen = None
    for d, p in pairs:
        try:
            t, pr = get(d), get(p)
        except JQuantsError as exc:
            print(f"  {d}: 取れない: {str(exc)[:100]}")
            continue
        if not len(t) or not len(pr):
            print(f"  {d}: 行が無い（{len(t)} / 前営業日 {len(pr)}）")
            continue
        if cols_seen is None:
            cols_seen = sorted(t.columns)
            print(f"列: {cols_seen}")
        results.append((d, compare_day(t, pr)))
    for line in report(results):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
