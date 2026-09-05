#!/usr/bin/env python3
"""
モデル構築の前提となるデータ量を実測する調査スクリプト。

確認したいこと:
  1. 上場銘柄は何社取得できるか (=母集団の上限)
  2. 日付指定の一括取得が使えるか (銘柄ごとのループを避けられるか)
  3. 日次株価はどこまで遡れるか (=学習期間の上限)
  4. 財務情報 (/fins/summary) は日付指定で取れるか
  5. 信用残 (/markets/margin-interest) の遡及範囲

結果は research/probe_result.json に出力する。
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import JQuantsClient, JQuantsError, resolve_api_key  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_result.json")


def try_call(client, label, path, params):
    """1エンドポイントを叩いて件数と列名を記録する。"""
    try:
        rows = client.get_paginated(path, params)
        return {
            "ok": True, "rows": len(rows),
            "columns": sorted(rows[0].keys()) if rows else [],
            "sample": rows[0] if rows else None,
        }
    except JQuantsError as exc:
        return {"ok": False, "error": str(exc)[:300]}


def main():
    client = JQuantsClient(resolve_api_key(), pause=0.3)
    result = {"probedAt": dt.datetime.now(dt.timezone.utc).isoformat()}

    # --- 1. 上場銘柄数 (code 指定なし) ---
    print("[1] 上場銘柄一覧 (全件)")
    r = try_call(client, "master_all", "/equities/master", {})
    result["listed_universe"] = r
    print(f"    -> {r}" if not r["ok"] else f"    -> {r['rows']}社  列: {len(r['columns'])}個")

    # --- 2. 日付指定で日次株価を一括取得できるか ---
    # 直近の平日を探す
    d = dt.date.today() - dt.timedelta(days=1)
    for _ in range(7):
        if d.weekday() < 5:
            break
        d -= dt.timedelta(days=1)
    print(f"[2] 日次株価の日付一括取得 (date={d})")
    r = try_call(client, "bars_by_date", "/equities/bars/daily", {"date": d.isoformat()})
    result["bars_by_date"] = {**r, "date": d.isoformat()}
    print(f"    -> {r['rows']}行" if r["ok"] else f"    -> {r['error'][:120]}")

    # --- 3. 日次株価の遡及範囲 (単一銘柄で年を遡って試す) ---
    print("[3] 日次株価の遡及範囲 (トヨタ 72030 で年別に確認)")
    history = {}
    for year in [2026, 2024, 2022, 2020, 2018, 2016, 2014, 2010]:
        probe_from = f"{year}-04-01"
        probe_to = f"{year}-04-30"
        try:
            rows = client.get_paginated(
                "/equities/bars/daily", {"code": "72030", "from": probe_from, "to": probe_to}
            )
            history[year] = len(rows)
            print(f"    {year}: {len(rows)}営業日")
        except JQuantsError as exc:
            history[year] = f"ERROR: {str(exc)[:100]}"
            print(f"    {year}: 取得不可")
    result["price_history_by_year"] = history

    # --- 4. 財務情報を日付指定で取れるか ---
    print("[4] /fins/summary の日付指定")
    r = try_call(client, "fins_by_date", "/fins/summary", {"date": d.isoformat()})
    result["fins_by_date"] = {**r, "date": d.isoformat()}
    print(f"    -> {r['rows']}件" if r["ok"] else f"    -> {r['error'][:120]}")

    # --- 5. 信用残の遡及範囲 ---
    print("[5] /markets/margin-interest の遡及範囲 (トヨタ)")
    margin = {}
    for year in [2026, 2024, 2022, 2020, 2018]:
        try:
            rows = client.get_paginated(
                "/markets/margin-interest",
                {"code": "72030", "from": f"{year}-01-01", "to": f"{year}-12-31"},
            )
            margin[year] = len(rows)
            print(f"    {year}: {len(rows)}週")
        except JQuantsError as exc:
            margin[year] = f"ERROR: {str(exc)[:100]}"
            print(f"    {year}: 取得不可")
    result["margin_history_by_year"] = margin

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
