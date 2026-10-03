#!/usr/bin/env python3
"""
J-Quants プレミアムの `/fins/details`（BS/PL/CF の明細）を叩いて、取り込みに要る事実を測る。

なぜ要るか
--------
運用者の決定（2026-10-03）: 財務諸表の明細は EDINET DB（年次）ではなく J-Quants プレミアムの
`/fins/details`（四半期・開示日つき・全銘柄）で取る。スタンダードの鍵では 403
「This API is not available on your subscription plan」（docs/DATA_FIELDS.md、2026-09-22 実測）。
契約を変えた後、取り込み（research/jq_bulk.py の種別）を書くには次を知る必要がある。

  1. 鍵で開いたか（契約の反映）
  2. 応答の形: 列の名前、入れ子（V1 の fs_details は FinancialStatement の中に約400項目）か平らか、
     行の鍵になる列（開示番号 DiscNo など）と日付列（DiscDate など）
  3. 量: 1日あたりの行数・バイト数・ページ数。全期間（2016-10〜、約2,440営業日）の見積もり
  4. 遡れる範囲: 2016 年より前の日に行が返るか（プレミアムの範囲）

秘密と公開ログ
------------
**リポジトリも Actions のログも公開。** 鍵は印字しない（x-api-key ヘッダにしか載らない）。
応答の**値は印字しない**。印字するのは列の名前・件数・バイト数・日付だけ。エラーも分類と
HTTP の番号だけで、本文は出さない。

  python3 research/probe_fins_details.py            # 実際の鍵で叩く（Actions で）
  python3 research/probe_fins_details.py --fake     # 偽の応答で印字の形だけ確かめる（手元）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402

PATH = "/fins/details"
OUT_JSON = os.path.join(HERE, "probe_fins_details.json")

#: 叩く日。決算が集中する日・普通の日・非営業日・2016 年より前（遡れる範囲の確認）
SAMPLE_DATES = [
    ("2008-05-15", "2016年より前（遡れる範囲）"),
    ("2012-05-15", "2016年より前（遡れる範囲）"),
    ("2017-05-12", "決算集中日"),
    ("2019-05-15", "決算集中日"),
    ("2021-08-06", "決算集中日"),
    ("2023-11-13", "決算集中日"),
    ("2024-03-05", "普通の日"),
    ("2024-05-12", "日曜（0件のはず）"),
    ("2024-05-15", "決算集中日"),
    ("2025-02-14", "決算集中日"),
    ("2025-06-10", "普通の日"),
    ("2026-05-15", "決算集中日"),
    ("2026-08-07", "決算集中日"),
    ("2026-09-26", "普通の日（直近）"),
]
#: 全期間の見積もりに使う営業日数（2016-10-03〜2026-10-01 の取得記録と同じ）
TRADING_DAYS_TOTAL = 2436
#: 1年の営業日のうち決算集中日（1日 300件超）のおおよその数
PEAK_DAYS_PER_YEAR = 24


def classify(exc: Exception) -> Tuple[str, str]:
    """エラーを (分類, HTTP番号) にする。本文は返さない。"""
    msg = str(exc)
    m = re.search(r"HTTP (\d{3})", msg)
    code = m.group(1) if m else "???"
    if "not available on your subscription" in msg:
        return "PLAN", code                     # 在るが契約が足りない
    if "does not exist" in msg:
        return "NOT_FOUND", code
    if "subscription covers" in msg:
        return "OUT_OF_RANGE", code             # 契約の範囲より前の日
    return "OTHER", code


def fetch_pages(client: JQuantsClient, day: str) -> Tuple[List[dict], int]:
    """pagination_key を辿りながらページ数も数える（get_paginated はページ数を返さない）。"""
    rows: List[dict] = []
    cursor: Optional[str] = None
    pages = 0
    for _ in range(200):
        params: Dict[str, Any] = {"date": day}
        if cursor:
            params["pagination_key"] = cursor
        data = client.get(PATH, params)
        pages += 1
        batch = data.get("data")
        if isinstance(batch, list):
            rows.extend(batch)
        cursor = data.get("pagination_key")
        if not cursor:
            break
    return rows, pages


def schema_of(rows: List[dict]) -> Dict[str, Any]:
    """列の名前と形だけ（値は見ない）。入れ子の列はその中の名前も集める。"""
    top: Dict[str, set] = {}
    nested: Dict[str, set] = {}
    for r in rows:
        for k, v in r.items():
            kind = ("dict" if isinstance(v, dict) else "list" if isinstance(v, list)
                    else "null" if v is None else type(v).__name__)
            top.setdefault(k, set()).add(kind)
            if isinstance(v, dict):
                nested.setdefault(k, set()).update(v.keys())
            elif isinstance(v, list):
                for e in v:
                    if isinstance(e, dict):
                        nested.setdefault(k, set()).update(e.keys())
    return {"columns": {k: sorted(t) for k, t in sorted(top.items())},
            "nested": {k: sorted(s) for k, s in sorted(nested.items())}}


def present(rows: List[dict], names: List[str]) -> Dict[str, float]:
    """候補の列名ごとに、値が入っている行の割合（0〜1）。列が無ければ -1。"""
    out = {}
    for n in names:
        if not rows or n not in rows[0] and not any(n in r for r in rows[:50]):
            out[n] = -1.0
            continue
        have = sum(1 for r in rows if r.get(n) not in (None, "", "-"))
        out[n] = have / len(rows)
    return out


class FakeClient:
    """--fake: 印字の形を手元で確かめるための偽の応答。値は適当で、量も本物ではない。"""

    def get(self, path: str, params: dict) -> dict:
        day = params.get("date", "")
        if day < "2016-01-01":
            raise JQuantsError(f"HTTP 400 {path}?date={day} : covers the following dates: 2016-10-03 ~")
        if day == "2024-05-12":
            return {"data": []}
        if params.get("pagination_key"):
            rows = [self._row(day, i) for i in range(3)]
            return {"data": rows}
        rows = [self._row(day, i) for i in range(5)]
        return {"data": rows, "pagination_key": "next"}

    @staticmethod
    def _row(day: str, i: int) -> dict:
        return {"Code": f"{10000 + i}", "DiscDate": day, "DiscTime": "15:00",
                "DiscNo": f"{day.replace('-', '')}{i:05d}", "DocType": "1QFinancialStatements_Consolidated_JP",
                "CurFYSt": "2024-04-01", "CurPerType": "1Q",
                "BS": {"CashAndDeposits": 1.0, "Inventories": 2.0, "Goodwill": None},
                "PL": {"NetSales": 3.0, "GrossProfit": 4.0},
                "CF": {"CapitalExpenditure": -5.0}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="/fins/details の疎通・形・量を測る")
    ap.add_argument("--fake", action="store_true", help="偽の応答で印字の形だけ確かめる")
    ap.add_argument("--dates", default="", help="叩く日をカンマ区切りで（省略で既定の一覧）")
    args = ap.parse_args(argv)

    if args.fake:
        client: Any = FakeClient()
        print("[fake] 偽の応答。量も形も本物ではない")
    else:
        client = JQuantsClient(resolve_api_key(), pause=0.3)
    dates = ([(d.strip(), "指定") for d in args.dates.split(",") if d.strip()]
             if args.dates else SAMPLE_DATES)

    print(f"[probe] {PATH} を {len(dates)}日ぶん叩く（値は印字しない。列名・件数・バイト数だけ）")
    results: List[Dict[str, Any]] = []
    all_rows: List[dict] = []
    plan_blocked = False
    for day, why in dates:
        t0 = time.time()
        rec: Dict[str, Any] = {"date": day, "why": why}
        try:
            rows, pages = fetch_pages(client, day)
            body = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            rec.update({"ok": True, "rows": len(rows), "pages": pages, "bytes": len(body),
                        "seconds": round(time.time() - t0, 1)})
            print(f"  {day}  OK   {len(rows):>6,}行  {pages:>3}ページ  {len(body)/1e6:>7.2f} MB  "
                  f"{rec['seconds']:>5.1f}秒  {why}")
            all_rows.extend(rows)
        except JQuantsError as exc:
            kind, code = classify(exc)
            rec.update({"ok": False, "error": kind, "http": code,
                        "seconds": round(time.time() - t0, 1)})
            print(f"  {day}  {kind:<12} HTTP {code}  {why}")
            if kind == "PLAN":
                plan_blocked = True
                break
        results.append(rec)

    summary: Dict[str, Any] = {"path": PATH, "probed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "fake": bool(args.fake), "results": results}
    if plan_blocked:
        print("\n[plan] 契約が足りない（This API is not available on your subscription plan）。"
              "プレミアムに変えた後にもう一度回す。鍵は同じでよいかもここで分かる")
    elif all_rows:
        sc = schema_of(all_rows)
        summary["schema"] = sc
        ids = ["Code", "DiscDate", "DiscTime", "DiscNo", "DocType", "CurFYSt", "CurPerType", "CurPerEn"]
        summary["id_columns"] = present(all_rows, ids)
        print(f"\n[schema] 列 {len(sc['columns'])}本（入れ子 {len(sc['nested'])}本）:")
        for k, kinds in sc["columns"].items():
            extra = f"  中の名前 {len(sc['nested'][k])}本" if k in sc["nested"] else ""
            print(f"  {k:<28} {'/'.join(kinds):<14}{extra}")
        for k, names in sc["nested"].items():
            print(f"\n[schema] {k} の中の名前（{len(names)}本）:")
            for i in range(0, len(names), 6):
                print("    " + ", ".join(names[i:i + 6]))
        print("\n[id] 鍵・日付の候補列（値が入っている行の割合。-1 は列が無い）:")
        for n, share in summary["id_columns"].items():
            print(f"  {n:<12} {share:>6.1%}" if share >= 0 else f"  {n:<12}   無い")
        ok = [r for r in results if r.get("ok")]
        peak = [r for r in ok if r["rows"] >= 300]
        normal = [r for r in ok if 0 < r["rows"] < 300]
        if peak and normal:
            years = TRADING_DAYS_TOTAL / 245
            est_rows = (sum(r["rows"] for r in peak) / len(peak) * PEAK_DAYS_PER_YEAR
                        + sum(r["rows"] for r in normal) / len(normal) * (245 - PEAK_DAYS_PER_YEAR)) * years
            per_row = sum(r["bytes"] for r in ok) / max(1, sum(r["rows"] for r in ok))
            summary["estimate"] = {"rows": int(est_rows), "json_mb": round(est_rows * per_row / 1e6),
                                   "seconds_per_day_mean": round(sum(r["seconds"] for r in ok) / len(ok), 1)}
            print(f"\n[estimate] 全期間（{TRADING_DAYS_TOTAL}営業日）: 約 {int(est_rows):,}行、"
                  f"JSON で約 {est_rows * per_row / 1e6:,.0f} MB、1日あたり平均 "
                  f"{summary['estimate']['seconds_per_day_mean']}秒 → 取り直し約 "
                  f"{summary['estimate']['seconds_per_day_mean'] * TRADING_DAYS_TOTAL / 3600:.1f}時間"
                  "（parquet は圧縮でこの数分の一）")
        early = [r for r in results if r["date"] < "2016-01-01"]
        if early:
            got = [r for r in early if r.get("ok") and r.get("rows", 0) > 0]
            print(f"\n[range] 2016年より前の {len(early)}日のうち行が返った日: {len(got)}日"
                  + ("（プレミアムの範囲はスタンダードの10年より広い）" if got else
                     "（遡れる範囲は今と同じか、別の理由。分類を見る）"))
    else:
        print("\n[probe] 行が1件も取れなかった。分類を見る")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\n[done] {OUT_JSON}（列名・件数だけ。値は入っていない）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
