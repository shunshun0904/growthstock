#!/usr/bin/env python3
"""Fetch/import private intraday data for the volume dashboard."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from volume_profile import JST, SOURCES, WINDOWS, lookback, normalize_code, window_profiles
from volume_profile_data import (
    AuthError, AddonClient, COLUMNS, ENDPOINTS, ProfileCache, bulk_plan, csv_chunks, iso_date, private_dir,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "research/_data/volume_profile"
PREDICTIONS = ROOT / "public/data/predictions.json"


def frames(path):
    if path.suffix == ".parquet":
        yield from (batch.to_pandas() for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000))
    else:
        with csv_chunks(path) as reader:
            yield from reader


def prediction_codes(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    latest = iso_date(data["asOf"])
    names = {}
    for item in data["candidates"]:
        if item.get("date") != latest:
            continue
        code = normalize_code(item.get("jqCode") or item["code"])
        names.setdefault(code, str(item.get("name") or code))
    return latest, names


def expected_window(calendar, trading, day, count):
    sessions = lookback(trading, day, count)
    if len(sessions) != count:
        return []
    first = date.fromisoformat(sessions[0])
    last = date.fromisoformat(day)
    if any((first + timedelta(days=i)).isoformat() not in calendar for i in range((last - first).days + 1)):
        return []
    return sessions


def fetch_quotes(cache, client, wanted, refresh):
    missing = wanted if refresh else cache.missing_quotes(wanted)
    by_date = defaultdict(set)
    for day, code in missing:
        by_date[day].add((day, code))
    for day, keys in sorted(by_date.items()):
        params = {"date": day}
        if len(keys) == 1:
            params["code"] = next(iter(keys))[1]
        rows = client.rows("/equities/bars/daily", params)
        if rows:
            cache.add_quotes(pd.DataFrame(rows), keys)


def _report_import(source, stats):
    print(f"{source}: 日足と照合できた銘柄・日付 {stats['verified']}、不一致 {stats['notReconciled']}")


def fetch_intraday(cache, client, source, wanted, refresh):
    missing = wanted if refresh else cache.missing_intraday(source, wanted)
    if not missing:
        return
    by_code = defaultdict(list)
    for day, code in missing:
        by_code[code].append(day)
    endpoint = ENDPOINTS[source]
    # Bounded symbol-specific requests are cheaper for a few charts. For
    # historical universes, use the vendor's monthly/daily gzip streams.
    if source == "minute" and len(by_code) <= 5 and len(missing) <= 100:
        for code, days in sorted(by_code.items()):
            params = {"code": code, "from": min(days), "to": max(days)}
            pages = (pd.DataFrame(page, columns=COLUMNS[source]) for page in client.pages(endpoint, params))
            _report_import(source, cache.add_intraday(pages, source, {(day, code) for day in days}))
        return
    days = [day for day, _ in missing]
    # A month key can cover dates later than its start date.
    files = client.rows("/bulk/list", {"endpoint": endpoint, "from": min(days)[:7], "to": max(days)})
    plan = bulk_plan(files, missing, endpoint)
    covered = sum(len(keys) for keys in plan.values())
    if covered < len(missing):
        print(f"{source}: Bulk一覧に対応ファイルがない銘柄・日付 {len(missing) - covered}。")
    for key, keys in sorted(plan.items()):
        with closing(client.read_bulk(key, source)) as batches:
            _report_import(source, cache.add_intraday(batches, source, keys))


def _stock_row(code, day, name, histories, calendar, trading, cutoff):
    quotes, intraday = histories
    profiles = {}
    for count in WINDOWS:
        sessions = expected_window(calendar, trading, day, count)
        profiles[str(count)] = window_profiles(sessions, count, quotes, intraday, cutoff)
    quote = quotes.get(day)
    close = quote.close if quote and quote.valid and quote.observed_at <= cutoff else None
    return {"code": code, "name": name, "date": day, "close": close, "windows": profiles}


def _write_json(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def build_dashboard(cache, output, latest, names, calendar, trading, demo=False):
    generated = datetime.now(timezone.utc)
    output = private_dir(output)
    stocks = []
    live_start = (date.fromisoformat(latest) - timedelta(days=60)).isoformat()
    for code, name in sorted(names.items()):
        histories = cache.load_code(code, live_start, latest)
        stocks.append(_stock_row(code, latest, name, histories, calendar, trading, generated))
    profile = output / "volume-profile.json"
    _write_json(profile, {"kind": "growthstock.volume-profile", "schemaVersion": 1, "asOf": latest,
                         "generatedAt": generated.isoformat(), "demo": demo, "stocks": stocks})
    return {"asOf": latest, "dashboardStocks": len(stocks), "output": str(profile),
            "dashboardWith1d": sum(any(r["status"] == "ok" for r in stock["windows"]["1"].values())
                                   for stock in stocks)}


def run(args):
    now = datetime.now(JST)
    predicted, names = prediction_codes(args.predictions or PREDICTIONS)
    latest = args.as_of or predicted
    if args.codes:
        names = {code: names.get(code, "") for code in map(normalize_code, args.codes)}
    if latest > now.date().isoformat():
        raise ValueError("将来の日付は利用できません。")
    if not names:
        print(json.dumps(build_dashboard(None, args.output, latest, names, {}, [], args.demo), ensure_ascii=False))
        return 0
    calendar_start = (date.fromisoformat(latest) - timedelta(days=60)).isoformat()
    with closing(ProfileCache(args.cache_dir, demo=args.demo)) as cache:
        client = AddonClient() if args.fetch else None
        if args.calendar:
            for frame in frames(args.calendar):
                cache.add_calendar(frame.to_dict("records"))
        if client:
            rows = client.rows("/markets/calendar", {"from": calendar_start, "to": latest})
            cache.add_calendar(rows)
        calendar = cache.calendar_dates(calendar_start, latest)
        if not calendar:
            raise ValueError("取引カレンダーがありません。--fetch または --calendar を指定してください。")
        trading = [day for day, is_session in calendar.items() if is_session]
        wanted = {(day, code) for day in lookback(trading, latest, args.lookback) for code in names}
        for path in args.daily:
            for frame in frames(path):
                cache.add_quotes(frame, wanted)
        floor = (pd.Timestamp(now.date()) - pd.DateOffset(years=2)).date().isoformat()
        api_keys = {(day, code) for day, code in wanted if day >= floor}
        if client:
            fetch_quotes(cache, client, api_keys, args.refresh)
        for source, paths in [("minute", args.minute), ("tick", args.ticks)]:
            for path in paths:
                with csv_chunks(path, source) as batches:
                    _report_import(source, cache.add_intraday(batches, source, wanted))
        if client:
            for source in SOURCES if args.source == "both" else (args.source,):
                fetch_intraday(cache, client, source, api_keys, args.refresh)
        report = build_dashboard(cache, args.output, latest, names, calendar, trading, args.demo)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="取得にはJ-Quants V2 APIキーとアドオンが必要")
    parser.add_argument("--as-of", type=iso_date, help="集計の最終営業日。既定は予測データの基準日")
    parser.add_argument("--codes", nargs="+", help="画面用銘柄。既定は直近の予測対象")
    parser.add_argument("--predictions", type=Path, help="銘柄一覧を読む予測JSON")
    parser.add_argument("--source", choices=("both", *SOURCES), default="both", help="オンラインで取得するデータ種別")
    parser.add_argument("--lookback", type=int, choices=WINDOWS, default=20, help="取得に使う最大営業日数")
    parser.add_argument("--refresh", action="store_true", help="既存分も取得し直す")
    parser.add_argument("--calendar", type=Path, help="J-Quants取引カレンダー（CSV/GZ/Parquet）")
    parser.add_argument("--daily", type=Path, action="append", default=[], help="調整前の日足（CSV/GZ/Parquet）")
    parser.add_argument("--minute", type=Path, action="append", default=[], help="J-Quants分足CSV/GZ")
    parser.add_argument("--ticks", type=Path, action="append", default=[], help="J-QuantsティックCSV/GZ")
    parser.add_argument("--cache-dir", type=Path, default=PRIVATE / "cache")
    parser.add_argument("--output", type=Path, default=PRIVATE / "output")
    parser.add_argument("--demo", action="store_true", help="合成試料を明示する（ローカル入力専用）")
    args = parser.parse_args(argv)
    if args.demo and args.fetch:
        parser.error("--demo はローカル入力にだけ使えます。")
    try:
        return run(args)
    except AuthError:
        print("J-Quants V2のAPIキーが未設定です。JQUANTS_API_KEY を端末で設定してください。", file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(f"価格帯別出来高: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
