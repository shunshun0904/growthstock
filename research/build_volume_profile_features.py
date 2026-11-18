#!/usr/bin/env python3
"""Build point-in-time volume profile features for private research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from build_volume_profile import PRIVATE, _report_import, _stock_row, _write_json, fetch_intraday, fetch_quotes, frames
from volume_profile import JST, SOURCES, WINDOWS, lookback, normalize_code
from volume_profile_data import AddonClient, AuthError, ProfileCache, csv_chunks, iso_date, private_dir

HISTORICAL_AVAILABILITY = time(17)
FEATURES = ("poc_gap", "vwap_gap", "value_low_gap", "value_high_gap", "poc_share", "above_close_share", "near_close_share")


def feature_columns(source):
    return [f"vp_{source}_{metric}_{window}d" for window in WINDOWS for metric in FEATURES]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_targets(path, start, end):
    cols = ["Code", "Date"]
    keys = (pd.read_parquet(path, columns=cols) if path.suffix == ".parquet"
            else pd.read_csv(path, usecols=cols, dtype=str))
    keys = keys[cols].assign(Code=keys.Code.map(normalize_code), Date=keys.Date.map(iso_date))
    keys = keys.loc[keys.Date.between(start, end)].sort_values(cols)
    if keys.empty or keys.duplicated(cols).any():
        raise ValueError("期間内の学習対象は、重複のない銘柄コード・日付の組で指定してください。")
    return keys


def assumed_timestamps(observations):
    # Apply the vendor timetable assumption to a research copy of each record.
    return {day: replace(row, observed_at=datetime.combine(date.fromisoformat(day), HISTORICAL_AVAILABILITY, JST))
            for day, row in observations.items()}


def feature_values(profile, close):
    return ((sum(profile["poc"]) / 2) / close - 1, profile["vwap"] / close - 1,
            profile["valueArea"][0] / close - 1, profile["valueArea"][1] / close - 1,
            profile["pocShare"], profile["aboveCloseShare"], profile["nearCloseShare"])


def materialize(cache, keys, calendar, trading, historical, decision_time):
    records = []
    counts = Counter()
    for code, group in keys.groupby("Code", sort=False):
        history_start = (date.fromisoformat(group.Date.min()) - timedelta(days=60)).isoformat()
        quotes, intraday = cache.load_code(code, history_start, group.Date.max())
        if historical:
            quotes = assumed_timestamps(quotes)
            intraday = {source: assumed_timestamps(days) for source, days in intraday.items()}
        for day in group.Date:
            cutoff = datetime.combine(date.fromisoformat(day), decision_time, JST)
            stock = _stock_row(code, day, "", (quotes, intraday), calendar, trading, cutoff)
            record = {"Code": code, "Date": pd.Timestamp(day)}
            for count, sources in stock["windows"].items():
                for source, profile in sources.items():
                    status = profile["status"]
                    counts[f"{source}/{count}d/{status}"] += 1
                    if status == "ok":
                        record.update({f"vp_{source}_{metric}_{count}d": value
                                       for metric, value in zip(FEATURES, feature_values(profile, stock["close"]))})
            records.append(record)
    columns = [column for source in SOURCES for column in feature_columns(source)]
    frame = pd.DataFrame(records, columns=["Code", "Date", *columns])
    frame[columns] = frame[columns].astype(float)
    return frame, counts


def run(args):
    today = datetime.now(JST).date()
    end = args.to_date or today.isoformat()
    if end > today.isoformat():
        raise ValueError("将来の日付は利用できません。")
    keys = load_targets(args.targets, args.from_date, end)
    start, end = keys.Date.min(), keys.Date.max()
    calendar_start = (date.fromisoformat(start) - timedelta(days=60)).isoformat()
    with closing(ProfileCache(args.cache_dir, demo=args.demo)) as cache:
        client = AddonClient() if args.fetch else None
        if args.calendar:
            for frame in frames(args.calendar):
                cache.add_calendar(frame.to_dict("records"))
        if client:
            cache.add_calendar(client.rows("/markets/calendar", {"from": calendar_start, "to": end}))
        calendar = cache.calendar_dates(calendar_start, end)
        if not calendar:
            raise ValueError("取引カレンダーがありません。--fetch または --calendar を指定してください。")
        trading = [day for day, is_session in calendar.items() if is_session]
        windows = {day: lookback(trading, day, max(WINDOWS)) for day in keys.Date.unique()}
        wanted = {(session, code) for code, day in keys.itertuples(index=False, name=None) for session in windows[day]}
        for path in args.daily:
            for frame in frames(path):
                cache.add_quotes(frame, wanted)
        floor = (pd.Timestamp(today) - pd.DateOffset(years=2)).date().isoformat()
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
        features, counts = materialize(cache, keys, calendar, trading, args.historical, args.decision_time)

    output = private_dir(args.output)
    path = output / "volume-features.parquet"
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
        temp = Path(handle.name)
    try:
        features.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    report = {"schemaVersion": 1, "asOf": end, "from": start, "generatedAt": datetime.now(timezone.utc).isoformat(),
              "featureRows": len(features), "trainingDecisionJst": args.decision_time.isoformat(),
              "trainingAvailability": "historical_schedule_estimate" if args.historical else "observed_at",
              "historicalVersionsVerified": False, "featureWindowStatusCounts": dict(counts), "demo": args.demo,
              "sha256": file_sha256(path)}
    _write_json(path.with_suffix(".meta.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True, help="学習対象Code/DateのParquetまたはCSV")
    parser.add_argument("--from-date", type=iso_date, default="1900-01-01", help="学習対象の開始日")
    parser.add_argument("--to-date", type=iso_date, help="学習対象の終了日")
    parser.add_argument("--decision-time", type=time.fromisoformat, default=time(19), help="評価時刻（JST）")
    parser.add_argument("--historical", action="store_true", help="公表時刻を推定して過去分を再構築")
    parser.add_argument("--fetch", action="store_true", help="J-Quants V2 APIから取得")
    parser.add_argument("--source", choices=("both", *SOURCES), default="both")
    parser.add_argument("--refresh", action="store_true", help="キャッシュ済みのセッションも取得し直す")
    parser.add_argument("--calendar", type=Path, help="J-Quants取引カレンダー（CSV/GZ/Parquet）")
    parser.add_argument("--daily", type=Path, action="append", default=[], help="調整前の日足（CSV/GZ/Parquet）")
    parser.add_argument("--minute", type=Path, action="append", default=[], help="分足CSV/GZ")
    parser.add_argument("--ticks", type=Path, action="append", default=[], help="ティックCSV/GZ")
    parser.add_argument("--cache-dir", type=Path, default=PRIVATE / "cache")
    parser.add_argument("--output", type=Path, default=PRIVATE / "research")
    parser.add_argument("--demo", action="store_true", help="合成試料を明示する（ローカル入力専用）")
    args = parser.parse_args(argv)
    if args.demo and args.fetch:
        parser.error("--demo はローカル入力にだけ使えます。")
    if args.decision_time.tzinfo is not None:
        parser.error("--decision-time はタイムゾーンなしの日本時刻を指定してください。")
    try:
        return run(args)
    except AuthError:
        print("J-Quants V2のAPIキーが未設定です。JQUANTS_API_KEY を端末で設定してください。", file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print(f"価格帯別出来高の検証: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
