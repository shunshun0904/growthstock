"""Private cache and J-Quants V2 input for volume profiles."""

from __future__ import annotations

import gzip
import http.client
import io
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from volume_profile import Intraday, Quote, normalize_code, reconciles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from jquants_data_fetcher import AuthError, JQuantsError, JQuantsClient, resolve_api_key

ENDPOINTS = {"tick": "/equities/trades", "minute": "/equities/bars/minute"}
COLUMNS = {
    "tick": ["Date", "Code", "TransactionId", "Price", "TradingVolume"],
    "minute": ["Date", "Code", "Time", "L", "H", "Vo", "Va"],
}
DAILY_FIELDS = ("L", "H", "C", "Vo", "Va", "AdjFactor")
CHUNK_SIZE = 200_000


def iso_date(value):
    return date.fromisoformat(str(value)[:10]).isoformat()


def private_dir(path):
    path = Path(path).expanduser().resolve()
    if any(p.name in ("public", "dist") for p in (path, *path.parents)):
        raise ValueError("個人用データの保存先は public/・dist/ の外に指定してください。")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _optional_number(value):
    if value is None or pd.isna(value) or value == "":
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("日足に無効な数値があります。")
    return number


def _quote(row):
    return Quote(*row[:6], row[6], datetime.fromisoformat(row[7]))


def _unpack(payload):
    return np.frombuffer(zlib.decompress(payload), dtype="<f8").reshape(-1, 3)


class ProfileCache:
    """Keep per-session price totals and observation times, without raw tapes."""

    def __init__(self, folder, demo=False):
        self.db = sqlite3.connect(private_dir(folder) / "intraday.sqlite3")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calendar (day TEXT PRIMARY KEY, trading INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS quote (
                code TEXT, day TEXT, low REAL, high REAL, close REAL, volume REAL,
                turnover REAL, factor REAL, action TEXT, observed TEXT NOT NULL,
                PRIMARY KEY (code, day)
            );
            CREATE TABLE IF NOT EXISTS intraday (
                code TEXT, day TEXT, source TEXT, observed TEXT NOT NULL, points BLOB NOT NULL,
                PRIMARY KEY (code, day, source)
            );
            CREATE TEMP TABLE stage (
                code TEXT, day TEXT, price REAL, qty REAL, value REAL, PRIMARY KEY (code, day, price)
            ) WITHOUT ROWID;
        """)
        mode = "demo" if demo else "personal"
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES ('mode', ?)", (mode,))
            stored = self.db.execute("SELECT value FROM settings WHERE name='mode'").fetchone()[0]
        if mode != stored:
            self.close()
            raise ValueError("合成データと個人データは別の --cache-dir を使ってください。")

    def close(self):
        self.db.close()

    def add_calendar(self, rows):
        calendar = {}
        for row in rows:
            day = iso_date(row["Date"])
            division = str(row["HolDiv"])
            if division not in {"0", "1", "2", "3"}:
                raise ValueError("取引カレンダーのHolDivが無効です。")
            trading = int(division in {"1", "2"})
            if day in calendar and calendar[day] != trading:
                raise ValueError("取引カレンダーに同じ日付の異なる情報があります。")
            calendar[day] = trading
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO calendar VALUES (?, ?)", calendar.items())

    def calendar_dates(self, start, end):
        return dict(self.db.execute("SELECT day,trading FROM calendar WHERE day BETWEEN ? AND ? ORDER BY day",
                                    (start, end)))

    def add_quotes(self, frame, wanted):
        required = {"Date", "Code", *DAILY_FIELDS}
        if not required <= set(frame):
            raise ValueError("調整前の日足の列が不足しています: " + ", ".join(sorted(required - set(frame))))
        chosen = _select(frame, wanted)
        if chosen.duplicated(["Code", "Date"]).any():
            raise ValueError("日足の銘柄・日付が重複しています。")
        observed = datetime.now(timezone.utc).isoformat()
        records = []
        for row in chosen.to_dict("records"):
            values = tuple(_optional_number(row[k]) for k in DAILY_FIELDS)
            action = _optional_number(row.get("ExRT"))
            records.append((row["Code"], row["Date"], *values,
                            str(int(action)) if action in (0, 1, 2, 3) else None, observed))
        # Retain the earlier observation timestamp when re-reading the same version.
        with self.db:
            self.db.executemany("""
                INSERT INTO quote VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (code, day) DO UPDATE SET low=excluded.low, high=excluded.high,
                    close=excluded.close, volume=excluded.volume, turnover=excluded.turnover,
                    factor=excluded.factor, action=excluded.action, observed=excluded.observed
                WHERE (low,high,close,volume,turnover,factor,action) IS NOT
                      (excluded.low,excluded.high,excluded.close,excluded.volume,excluded.turnover,
                       excluded.factor,excluded.action)
            """, records)
        return len(records)

    def quote(self, code, day):
        raw = self.db.execute("SELECT low,high,close,volume,turnover,factor,action,observed FROM quote "
                              "WHERE code=? AND day=?", (code, day)).fetchone()
        return _quote(raw) if raw else None

    def load_code(self, code, start, end):
        quote_rows = self.db.execute("SELECT day,low,high,close,volume,turnover,factor,action,observed FROM quote "
                                     "WHERE code=? AND day BETWEEN ? AND ?", (code, start, end))
        quotes = {day: _quote(row) for day, *row in quote_rows}
        intraday = {source: {} for source in ENDPOINTS}
        for day, source, seen, payload in self.db.execute(
                "SELECT day,source,observed,points FROM intraday WHERE code=? AND day BETWEEN ? AND ?",
                (code, start, end)):
            intraday[source][day] = Intraday(_unpack(payload), datetime.fromisoformat(seen))
        return quotes, intraday

    def missing_quotes(self, wanted):
        known = set(self.db.execute("SELECT day,code FROM quote"))
        return wanted - known

    def missing_intraday(self, source, wanted):
        # A fully verified no-trade session does not need a tick/bar record.
        ready = set(self.db.execute("SELECT day,code FROM intraday WHERE source=?", (source,)))
        no_trades = set(self.db.execute("SELECT day,code FROM quote WHERE low IS NULL AND high IS NULL "
                                       "AND close IS NULL AND coalesce(volume,0)=0 AND coalesce(turnover,0)=0"))
        return wanted - ready - no_trades

    def add_intraday(self, frames, source, wanted):
        """Merge chunks within one complete input, replace prior snapshots atomically."""
        with self.db:
            self.db.execute("DELETE FROM stage")
            for frame in frames:
                points = _points(_select(frame, wanted), source)
                self.db.executemany("""
                    INSERT INTO stage VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (code,day,price) DO UPDATE SET qty=qty+excluded.qty, value=value+excluded.value
                """, points.itertuples(index=False, name=None))
            observed = datetime.now(timezone.utc).isoformat()
            staged = self.db.execute("SELECT DISTINCT code,day FROM stage").fetchall()
            checked = 0
            for code, day in staged:
                points = np.asarray(self.db.execute("SELECT price,qty,value FROM stage WHERE code=? AND day=? "
                                                    "ORDER BY price", (code, day)).fetchall(), dtype="<f8")
                quote = self.quote(code, day)
                # Keep only complete sessions. A truncated or incomplete import
                # cannot displace a prior checked snapshot.
                if quote is None or not reconciles(points, quote):
                    continue
                payload = zlib.compress(points.tobytes(), level=3)
                self.db.execute("""
                    INSERT INTO intraday VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (code,day,source) DO UPDATE SET observed=excluded.observed, points=excluded.points
                    WHERE points != excluded.points
                """, (code, day, source, observed, payload))
                checked += 1
            self.db.execute("DELETE FROM stage")
        return {"verified": checked, "notReconciled": len(staged) - checked}


def _select(frame, wanted):
    if not {"Date", "Code"} <= set(frame):
        raise ValueError("CSVにDateとCodeが必要です。")
    codes = frame.Code.astype(str).str.strip().str.upper()
    codes = codes.where(codes.str.len().ne(4), codes + "0")
    days = frame.Date.astype(str).str.slice(0, 10)
    keep = pd.MultiIndex.from_arrays([days, codes]).isin(wanted)
    return frame.loc[keep].assign(Code=codes[keep], Date=days[keep])


def _points(frame, source):
    required = COLUMNS[source]
    if not set(required) <= set(frame):
        raise ValueError("分足・ティックの列が不足しています: " + ", ".join(sorted(set(required) - set(frame))))
    id_col = "TransactionId" if source == "tick" else "Time"
    if frame[id_col].isna().any() or frame.duplicated(["Code", "Date", id_col]).any():
        raise ValueError("ティックIDまたは分足の時刻に欠損・重複があります。")
    with np.errstate(invalid="ignore", divide="ignore"):
        if source == "tick":
            price = pd.to_numeric(frame.Price, errors="coerce")
            qty = pd.to_numeric(frame.TradingVolume, errors="coerce")
            value = price * qty
        else:
            qty = pd.to_numeric(frame.Vo, errors="coerce")
            value = pd.to_numeric(frame.Va, errors="coerce")
            price = value / qty
            low = pd.to_numeric(frame.L, errors="coerce")
            high = pd.to_numeric(frame.H, errors="coerce")
            if not ((low > 0) & (price >= low - 1e-6) & (price <= high + 1e-6)).all():
                raise ValueError("分足のVWAPが高値・安値の範囲外にあります。")
    values = np.column_stack((price, qty, value))
    if not np.isfinite(values).all() or not (values > 0).all() or not (qty % 1 == 0).all():
        raise ValueError("分足・ティックに無効な価格・株数・売買代金があります。")
    used = frame[["Code", "Date"]].assign(price=price, qty=qty, value=value)
    return used.groupby(["Code", "Date", "price"], as_index=False, sort=False)[["qty", "value"]].sum()


def csv_chunks(file, source=None):
    options = {"dtype": {"Date": str, "Code": str, "TransactionId": str, "Time": str},
               "encoding": "utf-8-sig", "chunksize": CHUNK_SIZE}
    if source:
        options["usecols"] = COLUMNS[source]
    return pd.read_csv(file, **options)


class AddonClient:
    def __init__(self):
        # The add-on is documented at 60 requests/minute. Bulk payloads use
        # a separate request without the J-Quants API key.
        self.client = JQuantsClient(resolve_api_key(), pause=1.05)

    def get(self, endpoint, params):
        try:
            return self.client.get(endpoint, params)
        except AuthError:
            raise ValueError(f"{endpoint} にアクセスできません。APIキーとアドオン契約を確認してください。") from None
        except JQuantsError:
            raise ValueError(f"J-Quantsの {endpoint} を取得できませんでした。") from None

    def pages(self, endpoint, params):
        # The existing get_paginated helper silently returns a partial result
        # at its page cap. Reconciliation requires the end of the page chain.
        seen = set()
        while True:
            result = self.get(endpoint, params)
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise ValueError(f"{endpoint} のデータ形式が想定と異なります。")
            yield result["data"]
            cursor = result.get("pagination_key")
            if not cursor:
                return
            if not isinstance(cursor, str) or cursor in seen or len(seen) >= 1000:
                raise ValueError(f"{endpoint} のページングを完了できませんでした。")
            seen.add(cursor)
            params = {**params, "pagination_key": cursor}

    def rows(self, endpoint, params):
        return [row for page in self.pages(endpoint, params) for row in page]

    def read_bulk(self, key, source):
        result = self.get("/bulk/get", {"key": key})
        signed = result.get("url") if isinstance(result, dict) else None
        try:
            url = urllib.parse.urlsplit(signed if isinstance(signed, str) else "")
        except ValueError:
            raise ValueError("J-Quants BulkのダウンロードURLが無効です。") from None
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("J-Quants BulkのダウンロードURLが無効です。")
        try:
            with urllib.request.urlopen(signed, timeout=90) as response:
                with gzip.GzipFile(fileobj=response) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8-sig") as csv:
                        with csv_chunks(csv, source) as reader:
                            yield from reader
        except (urllib.error.URLError, OSError, EOFError, http.client.HTTPException, pd.errors.ParserError):
            # Do not print the temporary signed query or downloaded records.
            raise ValueError("J-Quants BulkのCSVを読み切れませんでした。再実行してください。") from None


def bulk_plan(files, wanted, endpoint):
    """One feed file per expected date, preferring a daily file to a month."""
    usable = {}
    for row in files:
        key = row.get("Key", "")
        if not key.startswith(endpoint.lstrip("/") + "/"):
            continue
        match = re.search(r"_(\d{4}-?\d{2}(?:-?\d{2})?)\.csv\.gz$", key)
        if not match:
            continue
        stamp = match[1].replace("-", "")
        choice = (str(row.get("LastModified", "")), key)
        if stamp not in usable or usable[stamp] < choice:
            usable[stamp] = choice
    grouped = defaultdict(set)
    for day, code in wanted:
        stamp = day.replace("-", "")
        choice = usable.get(stamp) or usable.get(stamp[:6])
        if choice:
            grouped[choice[1]].add((day, code))
    return grouped
