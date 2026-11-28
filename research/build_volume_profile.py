#!/usr/bin/env python3
"""
直近5営業日の予測対象の価格帯別出来高を、蓄積した分足・ティックから集計して書き出す。

  python3 research/build_volume_profile.py --download-cmd "bash scripts/gh_release_download.sh" \\
      --output out/volume-profile.json --summary-json out/summary.json

- 対象: public/data/predictions.json の直近5つの予測日（dates）の候補すべて。銘柄の重複は除く
- 基準日: 予測の基準日（asOf）。どの銘柄も、基準日までの 1・5・20 営業日で集計する
- 入力: 置き場の分足・ティックと、data-raw の日足（照合用）。取引カレンダーは J-Quants（--calendar で手元のファイル）
- 出力: --output に集計の JSON。画面には Release data-volume-profile 経由でデプロイが載せる
  （運用者の選択 2026-11-19「暗号化をやめて公開する」）。毎日の集計を Git に積まないよう、public/・dist/ の下には書かない

ログと --summary-json に出すのは、件数・日付・ファイル名だけ（価格・数量の値は出さない。ワークフローのログは公開）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "research", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import volume_profile_data as data  # noqa: E402
from volume_profile import (  # noqa: E402
    BUCKETS, SIZE_EDGES, SOURCES, WINDOWS, lookback, normalize_code, reconcile, window_profiles,
)

PREDICTIONS = ROOT / "public/data/predictions.json"
WORK = ROOT / "research/_data/volume_profile"
TARGET_DAYS = 5
KIND = "growthstock.volume-profile"
SCHEMA_VERSION = 2


def iso(value) -> str:
    return dt.date.fromisoformat(str(value)[:10]).isoformat()


def prediction_targets(predictions: dict, days: int = TARGET_DAYS):
    """直近 days 個の予測日の候補すべて。戻り値は (基準日, 予測日の一覧, 銘柄 -> {code, name, candidates})。"""
    as_of = iso(predictions["asOf"])
    listed = {iso(d) for d in predictions.get("dates") or []}
    listed |= {iso(item["date"]) for item in predictions.get("candidates", []) if item.get("date")}
    dates = sorted(d for d in listed if d <= as_of)[-days:]
    stocks: Dict[str, dict] = {}
    for item in predictions.get("candidates", []):
        if not item.get("date") or iso(item["date"]) not in dates:
            continue
        code = normalize_code(item.get("jqCode") or item["code"])
        entry = stocks.setdefault(code, {"code": code, "name": str(item.get("name") or code)[:200],
                                         "candidates": []})
        rank = item.get("rankInDay")
        entry["candidates"].append({"date": iso(item["date"]),
                                    "rank": rank if isinstance(rank, int) and rank > 0 else None})
    for entry in stocks.values():
        entry["candidates"].sort(key=lambda c: c["date"], reverse=True)
    return as_of, dates, stocks


def calendar_from_file(path: Path) -> List[str]:
    """J-Quants の取引カレンダー（Date, HolDiv）の CSV か JSON。HolDiv 1 = 営業日、2 = 半日立会。"""
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = rows.get("data", rows) if isinstance(rows, dict) else rows
    else:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    return sorted({iso(r["Date"]) for r in rows if str(r.get("HolDiv")) in ("1", "2")})


def calendar_from_api(start: dt.date, end: dt.date) -> List[str]:
    import minute_bars
    from jquants_data_fetcher import resolve_api_key

    api = minute_bars.Api(resolve_api_key())
    return [d.isoformat() for d in minute_bars.trading_days(api, start, end)]


def outside_public(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if any(p.name in ("public", "dist") for p in (path, *path.parents)):
        raise ValueError("集計は public/・dist/ の外に書いてください（Git に積まず、Release 経由でデプロイが載せる）。")
    return path


def write_json(path: Path, payload, mode: int = 0o644) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        fh.write(text)
        temp = Path(fh.name)
    os.chmod(temp, mode)
    os.replace(temp, path)
    return len(text.encode("utf-8"))


class Build:
    def __init__(self, args):
        self.args = args
        self.fetch = data.Fetcher(args.download_cmd)
        self.work = Path(args.work_dir)
        self.warnings: List[str] = []

    def manifest(self, tag: str, name: str, folder: str) -> Optional[dict]:
        path = self.fetch.get(tag, str(self.work / folder), name, required=False)
        return data.load_manifest(path) if path else None

    def run(self):
        args = self.args
        started = time.time()
        predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        as_of, target_dates, stocks = prediction_targets(predictions)
        if args.as_of:
            as_of = args.as_of
        if args.codes:
            stocks = {c: stocks.get(c, {"code": c, "name": c, "candidates": []})
                      for c in map(normalize_code, args.codes)}
        if as_of > dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat():
            raise ValueError("将来の日付は集計できません。")
        codes = sorted(stocks)
        start = dt.date.fromisoformat(as_of) - dt.timedelta(days=70)
        summary = {"asOf": as_of, "targets": len(codes), "targetDates": target_dates,
                   "files": {}, "stored": {}, "reconcile": {}, "status": {}, "warnings": self.warnings}
        rows = []
        if codes:
            if args.calendar:
                trading = [d for d in calendar_from_file(Path(args.calendar)) if start.isoformat() <= d <= as_of]
            else:
                trading = calendar_from_api(start, dt.date.fromisoformat(as_of))
            sessions = lookback(trading, as_of, max(WINDOWS))
            if len(sessions) != max(WINDOWS):
                raise ValueError(f"{as_of} が営業日でないか、取引カレンダーが足りません。")
            summary["window"] = {"from": sessions[0], "to": sessions[-1], "sessions": len(sessions)}
            rows = self.profiles(as_of, sessions, codes, stocks, summary)
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        payload = {"kind": KIND, "schemaVersion": SCHEMA_VERSION, "asOf": as_of, "generatedAt": generated,
                   "demo": bool(args.demo), "targetDates": target_dates,
                   "definitions": {"windows": list(WINDOWS), "buckets": list(BUCKETS), "sizeEdges": list(SIZE_EDGES)},
                   "stocks": rows}
        summary["generatedAt"] = generated
        plain = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        summary["bytes"] = len(plain)
        if args.output:
            write_json(outside_public(Path(args.output)), payload)
        summary["seconds"] = round(time.time() - started, 1)
        summary["fetchedFiles"] = self.fetch.fetched
        return payload, summary

    def profiles(self, as_of: str, sessions: List[str], codes: List[str], stocks: dict, summary: dict):
        tick_manifest = self.manifest(*data.TICK_MANIFEST, "tick")
        minute_manifest = self.manifest(*data.MINUTE_MANIFEST, "minute")
        stored = {"tick": data.stored_days(tick_manifest), "minute": data.stored_days(minute_manifest)}
        for source in SOURCES:
            have = [d for d in sessions if d in stored[source]]
            summary["stored"][source] = {"days": len(have), "missing": [d for d in sessions if d not in stored[source]]}
        plan = data.plan_assets(sessions, tick_manifest, minute_manifest)
        bar_paths = [self.fetch.get(tag, str(self.work / "raw"), name) for tag, name in plan["bars"]]
        conflicts = set()
        quotes = data.load_quotes(bar_paths, codes, sessions, conflicts)
        if conflicts:
            self.warnings.append(f"日足に同じ銘柄・日付の違う行が {len(conflicts)}件ある（その日を含む期間は日足の欠損として扱う）")
        intraday: Dict[str, Dict[str, dict]] = {s: defaultdict(dict) for s in SOURCES}
        present = 0
        for tag, name in plan["tick"]:
            day = f"{name[5:9]}-{name[9:11]}-{name[11:13]}"
            path = self.fetch.get(tag, str(self.work / "tick"), name, required=False)
            if not path:
                self.warnings.append(f"ティックの取得記録にある {name} が {tag} に無い（その日は欠損として扱う）")
                continue
            present += 1
            for code, session in data.load_tick_day(path, day, codes).items():
                intraday["tick"][code][day] = session
            if self.args.remove_after_read:
                os.remove(path)
        summary["files"]["tick"] = {"planned": len(plan["tick"]), "present": present}
        minute_days = [d for d in sessions if d in stored["minute"]]
        present = 0
        for tag, name in plan["minute"]:
            path = self.fetch.get(tag, str(self.work / "minute"), name, required=False)
            if not path:
                self.warnings.append(f"分足の取得記録にある {name} が {tag} に無い（その月は欠損として扱う）")
                continue
            present += 1
            for code, by_day in data.load_minute_month(path, minute_days, codes).items():
                intraday["minute"][code].update(by_day)
        summary["files"]["minute"] = {"planned": len(plan["minute"]), "present": present}
        summary["files"]["bars"] = [name for _, name in plan["bars"]]

        checks = {s: Counter() for s in SOURCES}
        status = {str(n): {s: Counter() for s in SOURCES} for n in WINDOWS}
        rows = []
        for code in codes:
            daily = quotes.get(code, {})
            series = {s: dict(intraday[s].get(code, {})) for s in SOURCES}
            for source in SOURCES:
                for day in sessions:
                    quote, session = daily.get(day), series[source].get(day)
                    if quote is None:
                        checks[source]["no_daily"] += 1
                    elif session is None:
                        checks[source]["no_trades" if quote.no_trades else "missing"] += 1
                    elif session.problem:
                        checks[source][session.problem] += 1
                    else:
                        checks[source][reconcile(session.points, quote)] += 1
            windows = {str(n): window_profiles(lookback(sessions, as_of, n), n, daily, series) for n in WINDOWS}
            for n, by_source in windows.items():
                for source, result in by_source.items():
                    status[n][source][result["status"]] += 1
            last = daily.get(as_of)
            rows.append({**stocks[code], "close": last.close if last and last.valid else None, "windows": windows})
        summary["reconcile"] = {s: dict(sorted(c.items())) for s, c in checks.items()}
        summary["status"] = {n: {s: dict(sorted(c.items())) for s, c in by.items()} for n, by in status.items()}
        return rows


def print_summary(summary: dict) -> None:
    print(f"基準日 {summary['asOf']} / 対象 {summary['targets']}銘柄（予測日 {', '.join(summary['targetDates'])}）")
    if "window" in summary:
        w = summary["window"]
        print(f"集計に使う営業日: {w['from']} 〜 {w['to']}（{w['sessions']}日）")
    for source in SOURCES:
        st = summary["stored"].get(source)
        if st:
            print(f"{source}: 置き場にある日 {st['days']} / 無い日 {st['missing'] or 'なし'} / "
                  f"ファイル {summary['files'].get(source)}")
            print(f"{source}: 日足との照合（銘柄×日） {summary['reconcile'].get(source)}")
    for n, by in summary.get("status", {}).items():
        print(f"{n}営業日: " + " / ".join(f"{s} {dict(c)}" for s, c in by.items()))
    for message in summary["warnings"]:
        print(f"::warning title=価格帯別出来高::{message}")
    print(f"大きさ: {summary.get('bytes', 0):,} バイト / {summary.get('seconds')}秒")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS, help="対象を読む予測の JSON")
    parser.add_argument("--as-of", type=iso, help="集計の基準日（既定は予測の基準日）")
    parser.add_argument("--codes", nargs="+", help="対象の銘柄を指定する（手元の確認用）")
    parser.add_argument("--work-dir", type=Path, default=WORK, help="置き場から取り出したファイルを置く場所")
    parser.add_argument("--download-cmd", help="置き場から取り出すコマンド（TAG DIR NAME を後ろに付けて呼ぶ）")
    parser.add_argument("--calendar", type=Path, help="取引カレンダーの CSV/JSON（Date, HolDiv）。無ければ J-Quants から")
    parser.add_argument("--output", type=Path, help="集計の JSON の書き出し先（public/・dist/ の外）")
    parser.add_argument("--summary-json", type=Path, help="件数だけのまとめ")
    parser.add_argument("--remove-after-read", action="store_true", help="読み終えたティックのファイルを消す（容量の節約）")
    parser.add_argument("--demo", action="store_true", help="合成データで作ったことを集計に記す")
    args = parser.parse_args(argv)
    try:
        _, summary = Build(args).run()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"価格帯別出来高: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
