#!/usr/bin/env python3
"""
J-Quants の分足（/equities/bars/minute。アドオン「株価 分足・ティック」）を取得して蓄積する。

  python3 research/minute_bars.py --data-dir research/_minute \\
      --fetch-cmd  "bash scripts/gh_release_download.sh data-minute research/_minute" \\
      --upload-cmd "bash scripts/gh_release_upload.sh data-minute"

流れ
----
1. 契約の確認: 直近の営業日の JPX（86970。仕様のサンプルと同じ銘柄）を1回引く。
   アドオンが無ければ HTTP 403「not available on your subscription」なので、ここで止める
2. 一括ダウンロード: /bulk/list?endpoint=/equities/bars/minute の一覧（月ごとの CSV.gz）の
   うち、まだ取り込んでいないもの・更新されたもの（LastModified が変わった）を、古い月から
   /bulk/get の署名付きURL（有効5分・使い回し不可）で落として月のファイルに入れる。
   J-Quants は多数銘柄・長期間の取得に API の連続取得ではなく CSV の一括取得を勧めている
3. API: 営業日のうち、まだ無い日を date= の指定（その日の全銘柄。ページ送り）で取る。
   0行の日は公表前かもしれないので取得済みにしない（次の実行で取り直す）
4. 取得記録を書き、変わったファイルを保存先へ上げる。**データ → 取得記録の順**
   （記録だけが先に上がると、上がっていない日を取得済みと思い込む）

J-Quants の分足は直近2年分だけで、毎日1日ずつ古い日が取れなくなる。取った日は消さない
（2年の範囲の外に出た月も保存先に残る。それが蓄積の意味）。

レート制限は Standard で 1分120回（/bulk/list・/bulk/get も数える）。429 には Retry-After が
付かず、公式は「1分以上（確実には2分程度）待って再開」としているので、既存の取得
（scripts/jquants_data_fetcher.py の _request。待ちは計15秒）とは別に待ち方を持つ。

ログは公開（public リポジトリの Actions）。件数・日付・ファイル名・列名だけを出し、
株価・出来高の値や、一括ファイルの署名付きURLは出さない。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import http.client
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.csv as pcsv  # noqa: E402

import minute_store as S  # noqa: E402
from jquants_data_fetcher import (  # noqa: E402
    API_BASE, USER_AGENT, AuthError, JQuantsError, resolve_api_key)

DATA_DIR = os.path.join(ROOT, "research", "_minute")
JST = dt.timezone(dt.timedelta(hours=9))
ENDPOINT = S.ENDPOINT
BULK_LIST = "/bulk/list"
BULK_GET = "/bulk/get"

#: 分足を遡れる年数（アドオンのデータ格納期間。公式のデータ仕様の表で2年）
RETENTION_YEARS = 2
#: 契約の確認に引く銘柄（仕様のサンプルと同じ JPX）
PROBE_CODE = "86970"
#: リクエストの間隔（秒）。1分120回（Standard）より少し遅く
PAUSE = 0.55
#: 1日ぶんのページ送りの上限（無限ループよけ。超えたら止める）。
#: 既存の get_paginated は50ページで黙って打ち切るので、分足には使わない
MAX_PAGES = 5000
#: 429 のときの待ち（秒）。Retry-After が無く、公式は1分以上・確実には2分程度
RATE_LIMIT_WAITS = (65, 125, 185)
#: 5xx・通信の失敗のときの待ち（秒）
SERVER_ERROR_WAITS = (5, 15, 45, 90)
#: その日の行数が、取得済みの日の中央値のこの割合に満たなければ、公表の途中かもしれないので
#: 取得済みにしない（次の実行で取り直す）。半日立会でも半分ほどはあるので、それより低くする
PARTIAL_RATIO = 0.25
PARTIAL_MIN_DAYS = 5
#: 一括ファイルの取り出しの試行回数（そのたびに署名付きURLを取り直す）
DOWNLOAD_TRIES = 3
DOWNLOAD_CHUNK = 1 << 20

_COVERS = re.compile(r"covers the following dates", re.I)
_PERIOD = re.compile(r"(?<!\d)(\d{8}|\d{6})(?!\d)")


class PlanError(JQuantsError):
    """アドオン（株価 分足・ティック）の契約が無い（HTTP 403 not available on your subscription）。"""


def today_jst() -> dt.date:
    return dt.datetime.now(JST).date()


def retention_start(today: Optional[dt.date] = None, margin_days: int = 1) -> dt.date:
    """
    J-Quants が分足を返す最も古い日の目安（JST の今日の2年前 + margin_days）。

    Standard の10年の範囲は「今日の10年前から」で毎日ずれた（jq_bulk.subscription_start）。
    分足の2年も同じ数え方とみなし、日付の切り替わりの数分のずれに当たらないよう1日余裕を持つ。
    範囲の外を叩いたときは HTTP 400（covers the following dates）なので、その日は飛ばす。
    """
    t = today or today_jst()
    try:
        d = t.replace(year=t.year - RETENTION_YEARS)
    except ValueError:                       # 2/29 の2年前は無い
        d = dt.date(t.year - RETENTION_YEARS, 3, 1)
    return d + dt.timedelta(days=margin_days)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Api:
    """
    分足のための J-Quants API の呼び出し。

    jquants_data_fetcher.JQuantsClient を使わないのは、429 の待ち（あちらは計15秒。公式は
    1分以上）と、ページ送りの上限（あちらは50ページで黙って打ち切る）が分足に合わないため。
    """

    def __init__(self, api_key: str, pause: float = PAUSE,
                 sleep: Callable[[float], None] = time.sleep,
                 urlopen: Callable = urllib.request.urlopen) -> None:
        self._key = api_key
        self.pause = pause
        self.sleep = sleep
        self.urlopen = urlopen
        self.requests = 0

    def redact(self, text: str) -> str:
        if self._key and len(self._key) >= 8:
            text = text.replace(self._key, "[REDACTED]")
        return text

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{API_BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        shown = path + "".join(f" {k}={v}" for k, v in params.items() if k != "pagination_key")
        rate_tries = server_tries = 0
        while True:
            self.sleep(self.pause)
            self.requests += 1
            req = urllib.request.Request(url, headers={
                "x-api-key": self._key, "User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with self.urlopen(req, timeout=120) as res:
                    raw = res.read().decode("utf-8")
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:300]
                except Exception:                            # noqa: BLE001
                    pass
                body = self.redact(body)
                if exc.code == 429 and rate_tries < len(RATE_LIMIT_WAITS):
                    w = RATE_LIMIT_WAITS[rate_tries]
                    rate_tries += 1
                    print(f"  [429] レート制限。{w}秒待って取り直す（{rate_tries}/{len(RATE_LIMIT_WAITS)}）: "
                          f"{shown}", flush=True)
                    self.sleep(w)
                    continue
                if 500 <= exc.code < 600 and server_tries < len(SERVER_ERROR_WAITS):
                    w = SERVER_ERROR_WAITS[server_tries]
                    server_tries += 1
                    print(f"  [{exc.code}] {w}秒待って取り直す: {shown}", flush=True)
                    self.sleep(w)
                    continue
                if exc.code == 403 and "not available on your subscription" in body.lower():
                    raise PlanError(f"HTTP 403 {shown}: {body}") from exc
                if exc.code in (401, 403):
                    raise AuthError(f"HTTP {exc.code} {shown}: {body}") from exc
                raise JQuantsError(f"HTTP {exc.code} {shown}: {body}") from exc
            except (urllib.error.URLError, OSError, http.client.HTTPException,
                    json.JSONDecodeError) as exc:
                if server_tries < len(SERVER_ERROR_WAITS):
                    w = SERVER_ERROR_WAITS[server_tries]
                    server_tries += 1
                    print(f"  [{type(exc).__name__}] {w}秒待って取り直す: {shown}", flush=True)
                    self.sleep(w)
                    continue
                raise JQuantsError(f"{type(exc).__name__} {shown}: "
                                   f"{self.redact(str(exc))[:200]}") from exc

    def pages(self, path: str, params: dict) -> Iterator[dict]:
        """pagination_key を最後まで辿り、1ページずつ返す。キーが繰り返したら止める。"""
        key: Optional[str] = None
        seen = set()
        n = 0
        while True:
            p = dict(params)
            if key:
                p["pagination_key"] = key
            data = self.get(path, p)
            n += 1
            yield data
            key = data.get("pagination_key")
            if not key:
                return
            if key in seen:
                raise JQuantsError(f"{path}: pagination_key が繰り返した（{n}ページ目）")
            seen.add(key)
            if n >= MAX_PAGES:
                raise JQuantsError(f"{path}: {MAX_PAGES}ページを超えた。止める")


# --------------------------------------------------------------------------- #
# 営業日・契約の確認
# --------------------------------------------------------------------------- #

def trading_days(api: Api, start: dt.date, end: dt.date) -> List[dt.date]:
    """/markets/calendar の営業日（HolDiv 1 = 営業日、2 = 半日立会）。"""
    days = set()
    for page in api.pages("/markets/calendar", {"from": start.isoformat(), "to": end.isoformat()}):
        for r in page.get("data") or []:
            div = next((str(r[k]) for k in ("HolDiv", "HolidayDivision")
                        if r.get(k) is not None), "")
            if div in ("1", "2") and r.get("Date"):
                d = dt.date.fromisoformat(str(r["Date"])[:10])
                if start <= d <= end:
                    days.add(d)
    if not days:
        raise JQuantsError(f"/markets/calendar が営業日を返さなかった（{start}〜{end}）")
    return sorted(days)


def probe(api: Api, day: dt.date) -> dict:
    """アドオンの契約を確かめる。1銘柄だけ引く（値は出さず、行数と列名だけ返す）。"""
    data = api.get(ENDPOINT, {"code": PROBE_CODE, "date": day.isoformat()})
    rows = data.get("data") or []
    return {"day": day.isoformat(), "code": PROBE_CODE, "rows": len(rows),
            "columns": sorted(rows[0].keys()) if rows else []}


# --------------------------------------------------------------------------- #
# 一括ダウンロード（CSV.gz）
# --------------------------------------------------------------------------- #

def bulk_period(key: str) -> Optional[Tuple[dt.date, dt.date]]:
    """
    Key のファイル名から、そのファイルが覆う期間（最初の日・最後の日）を読む。
    例: equities/bars/minute/historical/2026/equities_bars_minute_202606.csv.gz -> 6/1〜6/30。
    8桁なら1日。読めなければ None（その Key は取り込まない。ログに出す）。
    """
    base = str(key).rsplit("/", 1)[-1]
    found = _PERIOD.findall(base)
    if not found:
        return None
    s = found[-1]
    try:
        if len(s) == 8:
            d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
            return d, d
        first = dt.date(int(s[:4]), int(s[4:]), 1)
    except ValueError:
        return None
    nxt = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return first, nxt - dt.timedelta(days=1)


def list_bulk(api: Api) -> List[dict]:
    """分足の一括ファイルの一覧。期間の古い順。"""
    entries = []
    last_exc: Optional[Exception] = None
    # endpoint の書き方は公式の一覧では "/equities/bars/minute"。通らなければ先頭の / を外して試す
    for ep in (ENDPOINT, ENDPOINT.lstrip("/")):
        try:
            for page in api.pages(BULK_LIST, {"endpoint": ep}):
                for r in page.get("data") or []:
                    key = r.get("Key") or r.get("key")
                    if not key:
                        continue
                    size = r.get("Size", r.get("size"))
                    entries.append({
                        "key": str(key),
                        "lastModified": r.get("LastModified", r.get("lastModified")),
                        "size": int(size) if isinstance(size, (int, float)) or
                        (isinstance(size, str) and size.isdigit()) else None,
                        "period": bulk_period(str(key)),
                    })
            break
        except PlanError:
            raise
        except JQuantsError as exc:
            last_exc = exc
            if "HTTP 400" not in str(exc) and "HTTP 404" not in str(exc):
                raise
            entries = []
    else:
        raise JQuantsError(f"{BULK_LIST} を引けなかった: {last_exc}")
    entries.sort(key=lambda e: (e["period"][0] if e["period"] else dt.date.max, e["key"]))
    return entries


def plan_bulk(entries: Sequence[dict], manifest: dict,
              start: Optional[dt.date] = None) -> List[dict]:
    """
    取り込む一括ファイル: まだ取り込んでいない Key と、更新日時・大きさが変わった Key。

    並びは「月の古い順、同じ月の中は更新日時の古い順」。一括ファイルには月ごと
    （historical/..._YYYYMM）と、今月の日ごと（live/..._YYYYMMDD）がある（2026-11-18 の実測）。
    同じ日を両方が覆うときは、後から当てた側で日ごと置き換わるので、新しく作られた側を後にする
    （月ごとのファイルは翌月5日ごろに作られ、その月の日ごとのファイルより新しい）。
    """
    todo = []
    for e in entries:
        if e["period"] is None:
            continue
        if start is not None and e["period"][1] < start:
            continue
        got = manifest.get("bulk", {}).get(e["key"])
        if got is not None:
            same = (got.get("lastModified"), got.get("size")) == (e["lastModified"], e["size"])
            if same and (e["lastModified"] is not None or e["size"] is not None):
                continue
        todo.append(e)
    todo.sort(key=lambda e: (S.month_of(e["period"][0]), str(e["lastModified"] or ""), e["key"]))
    return todo


def download_bulk(api: Api, key: str, dest: str, expect_size: Optional[int]) -> int:
    """
    /bulk/get で署名付きURLを取り、すぐに落とす（有効5分・使い回し不可）。
    URL はログに出さない（例外の文字列にも入りうるので、型だけを出す）。
    x-api-key は付けない（J-Quants 以外のホストに鍵を送らない）。
    """
    part = dest + ".part"
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        data = api.get(BULK_GET, {"key": key})
        url = data.get("url") or data.get("URL") or data.get("Url")
        if not url:
            raise JQuantsError(f"{BULK_GET} が url を返さなかった（項目: {sorted(data)[:10]}）")
        n = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with api.urlopen(req, timeout=600) as res, open(part, "wb") as fh:
                while True:
                    chunk = res.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    n += len(chunk)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            code = f" {exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
            print(f"  [bulk] 取り出しに失敗（{attempt}/{DOWNLOAD_TRIES}）: {type(exc).__name__}{code}。"
                  f"URL を取り直す", flush=True)
            api.sleep(10 * attempt)
            continue
        if expect_size is not None and n != int(expect_size):
            print(f"  [bulk] 大きさが一覧と違う（{n:,} / {int(expect_size):,} バイト。"
                  f"{attempt}/{DOWNLOAD_TRIES}）", flush=True)
            if attempt < DOWNLOAD_TRIES:
                api.sleep(10 * attempt)
                continue
            # 一覧の Size の意味が違うだけかもしれない。gzip の終わりの検査（CRC・長さ）で
            # 欠けは読み込みのときに分かるので、ここでは止めない
        os.replace(part, dest)
        return n
    raise JQuantsError(f"{key} を {DOWNLOAD_TRIES}回試しても取り出せなかった")


def _is_gzip(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _open_text(path: str):
    if _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return open(path, encoding="utf-8-sig", newline="")


def _arrow_input(path: str):
    if _is_gzip(path):
        return pa.CompressedInputStream(pa.OSFile(path), "gzip")
    return pa.OSFile(path)


def check_gzip(path: str) -> None:
    """gzip の終わり（CRC・長さ）まで読んで欠けが無いかを確かめる。欠けていれば例外。"""
    if not _is_gzip(path):
        return
    with gzip.open(path, "rb") as fh:
        while fh.read(DOWNLOAD_CHUNK):
            pass


def read_bulk_csv(path: str) -> Tuple[pa.Table, List[str]]:
    """一括の CSV（gzip でもそうでなくても）を読み、保存の形にそろえる。(表, 仕様外の列)。"""
    check_gzip(path)
    with _open_text(path) as fh:
        header = next(csv.reader([fh.readline()]), [])
    if not header:
        return S.SCHEMA.empty_table(), []
    typed = {n: (pa.float64() if S.canonical_name(n) in S.NUM_COLS else pa.string())
             for n in header}
    try:
        tab = pcsv.read_csv(_arrow_input(path), convert_options=pcsv.ConvertOptions(
            column_types=typed, strings_can_be_null=True))
    except pa.ArrowInvalid:
        # 数値の列に欠測の記号（"-" など）が混ざっていた。文字列で読み、normalize で直す
        tab = pcsv.read_csv(_arrow_input(path), convert_options=pcsv.ConvertOptions(
            column_types={n: pa.string() for n in header}, strings_can_be_null=True))
    return S.normalize(tab)


def probe_bulk_file(api: Api, entry: dict, data_dir: str) -> dict:
    """
    確認だけの実行（--probe-only）で、一括ファイルを1つ落として形を見る（保存はしない）。
    出すのは見出しの名前・行数・日数・銘柄数・時刻の範囲だけ（値は出さない）。
    """
    os.makedirs(data_dir, exist_ok=True)
    tmp = os.path.join(data_dir, ".probe.csv.gz")
    out: dict = {"key": entry["key"]}
    try:
        n = download_bulk(api, entry["key"], tmp, entry["size"])
        out["bytes"] = n
        with _open_text(tmp) as fh:
            out["header"] = next(csv.reader([fh.readline()]), [])
        out["gzip"] = _is_gzip(tmp)
        table, extra = read_bulk_csv(tmp)
        st = S.day_stats(table)
        times = sorted(set(table.column("Time").unique().to_pylist()))
        out.update({"rows": table.num_rows, "days": len(st),
                    "from": min(st) if st else None, "to": max(st) if st else None,
                    "codesPerDayMax": max((s["codes"] for s in st.values()), default=0),
                    "rowsPerDayMedian": (sorted(s["rows"] for s in st.values())[len(st) // 2]
                                         if st else 0),
                    "timeFirst": times[0] if times else None, "timeLast": times[-1] if times else None,
                    "distinctTimes": len(times), "extraColumns": extra})
    except (JQuantsError, ValueError, OSError, EOFError, pa.ArrowException) as exc:
        out["error"] = f"{type(exc).__name__}: {api.redact(str(exc))[:300]}"
    finally:
        for p in (tmp, tmp + ".part"):
            if os.path.exists(p):
                os.remove(p)
    print("[bulk] 形の確認（保存しない）: " + json.dumps(out, ensure_ascii=False))
    return out


# --------------------------------------------------------------------------- #
# 1日ぶん（API）
# --------------------------------------------------------------------------- #

def fetch_day(api: Api, day: dt.date) -> Tuple[pa.Table, List[str], int]:
    """その日の全銘柄の分足。(表, 仕様外の列, ページ数)。"""
    tables = []
    extra: set = set()
    pages = 0
    for page in api.pages(ENDPOINT, {"date": day.isoformat()}):
        pages += 1
        recs = page.get("data") or []
        if recs:
            t, ex = S.normalize(S.records_to_table(recs))
            tables.append(t)
            extra.update(ex)
    table = pa.concat_tables(tables) if tables else S.SCHEMA.empty_table()
    return table, sorted(extra), pages


def typical_rows(manifest: dict) -> Optional[float]:
    """取得済みの日の行数の中央値（PARTIAL_MIN_DAYS 日に満たなければ None）。"""
    rows = [int(v.get("rows", 0)) for v in manifest.get("days", {}).values() if v.get("rows")]
    if len(rows) < PARTIAL_MIN_DAYS:
        return None
    return float(statistics.median(rows))


# --------------------------------------------------------------------------- #
# 保存先とのやり取り
# --------------------------------------------------------------------------- #

def make_fetch(cmd: Optional[str], data_dir: str) -> Optional[Callable[[str], None]]:
    """保存先から月のファイルを取り寄せる関数（cmd にファイル名を足して呼ぶ）。"""
    if not cmd:
        return None
    base = shlex.split(cmd)

    def fetch(name: str) -> None:
        print(f"[store] 保存先から {name} を取り寄せる", flush=True)
        subprocess.run(base + [name], check=True)
    return fetch


class Flusher:
    """変わった月のファイル → 取得記録の順に保存先へ上げる。上げ終えたものは pending から外す。"""

    def __init__(self, store: S.MinuteStore, cmd: Optional[str]) -> None:
        self.store = store
        self.base = shlex.split(cmd) if cmd else None
        self.uploads = 0

    def __call__(self) -> None:
        if self.base is None:
            return
        names = sorted(self.store.dirty)
        if names:
            paths = [os.path.join(self.store.data_dir, n) for n in names]
            subprocess.run(self.base + paths, check=True)
            self.store.dirty.difference_update(names)
            self.uploads += len(names)
        if not (names or self.store.unflushed):
            return
        if not os.path.exists(S.manifest_path(self.store.data_dir)):
            self.store.save()
        subprocess.run(self.base + [S.manifest_path(self.store.data_dir)], check=True)
        self.store.unflushed = False
        self.uploads += 1
        print(f"[store] 保存先へ上げた: データ {len(names)}ファイル → 取得記録", flush=True)


def write_pending(store: S.MinuteStore) -> str:
    """
    上げ残し（.pending）を書く。ワークフローの受け皿のステップが読み、上から順に上げる
    （月のファイルが先、取得記録が最後）。
    """
    p = os.path.join(store.data_dir, ".pending")
    with open(p, "w", encoding="utf-8") as fh:
        for n in sorted(store.dirty):
            fh.write(n + "\n")
        if store.dirty or store.unflushed:
            fh.write(S.MANIFEST_NAME + "\n")
    return p


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, api: Api, store: S.MinuteStore, flush: Callable[[], None],
                 deadline: float, today: dt.date, clock: Callable[[], float] = time.monotonic) -> None:
        self.api = api
        self.store = store
        self.flush = flush
        self.deadline = deadline
        self.today = today
        self.clock = clock
        self.report: Dict[str, object] = {"errors": [], "warnings": []}

    def out_of_time(self) -> bool:
        return self.clock() > self.deadline

    def _extra(self, names: Sequence[str]) -> None:
        new = self.store.record_extra_columns(names)
        if new:
            msg = f"仕様に無い列 {new} が来た（保存していない。docs/MINUTE_BARS.md）"
            print(f"::warning title=分足に仕様外の列::{msg}")
            self.report["warnings"].append(msg)

    # -- 一括 --------------------------------------------------------------- #

    def bulk(self, start: Optional[dt.date]) -> None:
        # 一括ダウンロードだけが契約に無い（PlanError）ときも止めない。分足そのものの契約は
        # 最初の確認（probe）で見ているので、API の日ごとの取得で続ける
        try:
            entries = list_bulk(self.api)
        except JQuantsError as exc:
            msg = f"一括ファイルの一覧を引けなかった（API の日ごとの取得で続ける）: {str(exc)[:200]}"
            print(f"::warning title=分足の一括ダウンロード::{msg}")
            self.report["warnings"].append(msg)
            self.report["bulk"] = {"listed": 0}
            return
        unknown = [e["key"] for e in entries if e["period"] is None]
        dated = [e for e in entries if e["period"] is not None]
        size = sum(e["size"] or 0 for e in dated)
        rng = f"{dated[0]['period'][0]} 〜 {dated[-1]['period'][1]}" if dated else "なし"
        print(f"[bulk] 一覧 {len(entries)}件（期間 {rng}、計 {size/1e9:.2f}GB）")
        shown = dated if len(dated) <= 6 else dated[:3] + [None] + dated[-3:]
        for e in shown:
            if e is None:
                print("    …")
                continue
            print(f"    {e['key']}  {(e['size'] or 0)/1e6:.1f}MB  {e['lastModified']}")
        if unknown:
            print(f"[bulk] 期間を読めない Key {len(unknown)}件は取り込まない: {unknown[:3]}")
        todo = plan_bulk(entries, self.store.manifest, start)
        print(f"[bulk] 取り込む {len(todo)}件（まだ取り込んでいない・更新されたもの）")
        done = failed = 0
        tmp = os.path.join(self.store.data_dir, ".download.csv.gz")
        # 同じ月のファイル（今月の日ごとのファイル）はまとめて1回で月のファイルに書き、1回で上げる。
        # 1件ごとに書くと、日ごとのファイルの数だけ月のファイルを書き直して上げ直すことになる
        buf: List[Tuple[dict, pa.Table, Dict[str, Dict[str, int]]]] = []
        buf_month: Optional[str] = None
        buf_days: set = set()

        def commit() -> None:
            nonlocal buf, buf_days
            if not buf:
                return
            t0 = time.time()
            table = buf[0][1] if len(buf) == 1 else pa.concat_tables([t for _, t, _ in buf])
            self.store.write_days(table, "bulk")
            for e, _, st in buf:
                for d in st:
                    self.store.manifest["days"][d]["key"] = e["key"]
                self.store.record_bulk(e["key"], e["lastModified"], e["size"], st)
            self.store.save()
            self.flush()
            days = sorted(buf_days)
            span = f"{days[0]} 〜 {days[-1]}" if days else "なし"
            print(f"[bulk] 月 {buf_month}: {len(buf)}ファイル → {table.num_rows:,}行 / {len(days)}日"
                  f"（{span}）を保存して上げた / {time.time()-t0:.0f}秒", flush=True)
            buf, buf_days = [], set()

        for e in todo:
            if self.out_of_time():
                print("[bulk] 時間切れ。残りは次の実行で取る")
                break
            name = e["key"].rsplit("/", 1)[-1]
            t0 = time.time()
            try:
                n = download_bulk(self.api, e["key"], tmp, e["size"])
                table, extra = read_bulk_csv(tmp)
            except PlanError as exc:
                msg = f"一括ファイルの取り出しが契約で使えない（API の日ごとの取得で続ける）: {str(exc)[:200]}"
                print(f"::warning title=分足の一括ダウンロード::{msg}")
                self.report["warnings"].append(msg)
                break
            except (JQuantsError, ValueError, OSError, EOFError, pa.ArrowException) as exc:
                failed += 1
                msg = f"{name}: {type(exc).__name__}: {self.api.redact(str(exc))[:200]}"
                print(f"::error title=分足の一括ファイルを取り込めない::{msg}")
                self.report["errors"].append(msg)
                continue
            finally:
                for p in (tmp, tmp + ".part"):
                    if os.path.exists(p):
                        os.remove(p)
            self._extra(extra)
            st = S.day_stats(table)
            outside = [d for d in st
                       if not (e["period"][0].isoformat() <= d <= e["period"][1].isoformat())]
            if outside:
                print(f"[bulk] {name}: ファイル名の期間の外の日 {len(outside)}日が入っていた"
                      f"（そのまま保存する）: {outside[:3]}")
            month = S.month_of(e["period"][0])
            # 月が変わる・同じ日を覆うファイルが来たら、先にたまった分を書く（同じ日は後の側で置き換わる）
            if buf and (month != buf_month or buf_days & set(st)):
                commit()
            buf.append((e, table, st))
            buf_month = month
            buf_days |= set(st)
            done += 1
            days = sorted(st)
            span = f"{days[0]} 〜 {days[-1]}" if days else "なし"
            print(f"[bulk] {name}: {n/1e6:.1f}MB → {table.num_rows:,}行 / {len(st)}日（{span}）"
                  f" / {time.time()-t0:.0f}秒", flush=True)
        commit()
        self.report["bulk"] = {"listed": len(entries), "planned": len(todo),
                               "ingested": done, "failed": failed,
                               "listedFrom": dated[0]["period"][0].isoformat() if dated else None,
                               "listedTo": dated[-1]["period"][1].isoformat() if dated else None,
                               "listedBytes": size}

    # -- API ---------------------------------------------------------------- #

    def api_days(self, days: Sequence[dt.date]) -> None:
        have = S.stored_days(self.store.manifest)
        todo = [d for d in days if d.isoformat() not in have]
        print(f"\n[api] 営業日 {len(days)}日のうち未取得 {len(todo)}日"
              + (f"（{todo[0]} 〜 {todo[-1]}）" if todo else ""))
        typical = typical_rows(self.store.manifest)
        buf: List[pa.Table] = []
        buf_month: Optional[str] = None
        added = held = failed = 0

        def commit() -> None:
            nonlocal buf
            if not buf:
                return
            self.store.write_days(pa.concat_tables(buf), "api")
            self.store.save()
            self.flush()
            buf = []

        for d in todo:
            if self.out_of_time():
                print("[api] 時間切れ。残りは次の実行で取る")
                break
            if buf and S.month_of(d) != buf_month:
                commit()
            try:
                table, extra, pages = fetch_day(self.api, d)
            except PlanError:
                raise
            except JQuantsError as exc:
                if _COVERS.search(str(exc)):
                    print(f"[api] {d}: 契約の範囲の外（取得済みにしない）")
                    held += 1
                    continue
                failed += 1
                msg = f"{d}: {self.api.redact(str(exc))[:200]}"
                print(f"::warning title=分足の取得に失敗（次回また取る）::{msg}")
                self.report["errors"].append(msg)
                continue
            self._extra(extra)
            if table.num_rows == 0:
                print(f"[api] {d}: 0行（まだ公表前かもしれない。取得済みにしない）")
                held += 1
                continue
            if typical and table.num_rows < typical * PARTIAL_RATIO:
                msg = (f"{d}: {table.num_rows:,}行は取得済みの日の中央値 {typical:,.0f}行の"
                       f"{PARTIAL_RATIO:.0%}未満。公表の途中かもしれないので取得済みにしない")
                print(f"::warning title=分足が少ない::{msg}")
                self.report["warnings"].append(msg)
                held += 1
                continue
            codes = len(set(table.column("Code").unique().to_pylist()))
            print(f"[api] {d}: {table.num_rows:,}行 / {codes:,}銘柄 / {pages}ページ", flush=True)
            buf.append(table)
            buf_month = S.month_of(d)
            added += 1
        commit()
        self.report["api"] = {"missing": len(todo), "added": added, "held": held, "failed": failed}


def gaps(manifest: dict, days: Sequence[dt.date]) -> List[str]:
    have = S.stored_days(manifest)
    return [d.isoformat() for d in days if d.isoformat() not in have]


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="J-Quants の分足を取得して蓄積する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--from", dest="date_from", default=None,
                    help="YYYY-MM-DD。既定は取れる最も古い日（今日の2年前 + 1日）")
    ap.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD。既定は今日（JST）")
    ap.add_argument("--no-bulk", action="store_true", help="一括ダウンロードを使わない")
    ap.add_argument("--no-api", action="store_true", help="API の日ごとの取得をしない")
    ap.add_argument("--probe-only", action="store_true",
                    help="契約の確認・一括ファイルの一覧と、いちばん小さい1ファイルの形を見るだけ（何も保存しない）")
    ap.add_argument("--forget-from", default=None, metavar="YYYY-MM-DD",
                    help="この日以降の取得記録を外して取り直す（月のファイルは消さない）")
    ap.add_argument("--reingest-bulk", action="store_true",
                    help="一括ファイルを全部取り込み直す（取り込みの記録を外す）")
    ap.add_argument("--max-minutes", type=float, default=300.0,
                    help="この分数で新しい取得を打ち切る（ジョブの上限より短く）")
    ap.add_argument("--pause", type=float, default=PAUSE)
    ap.add_argument("--fetch-cmd", default=None,
                    help="保存先から月のファイルを取り寄せるコマンド（末尾にファイル名を足して呼ぶ）")
    ap.add_argument("--upload-cmd", default=None,
                    help="保存先へ上げるコマンド（末尾にファイルのパスを足して呼ぶ）")
    ap.add_argument("--summary-json", default=None, help="実行の集計を書く先")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None, api: Optional[Api] = None,
         today: Optional[dt.date] = None) -> int:
    args = parse_args(argv)
    t_start = time.monotonic()
    deadline = t_start + args.max_minutes * 60
    today = today or today_jst()
    started = dt.datetime.now(JST)

    store = S.MinuteStore(args.data_dir, fetch_file=make_fetch(args.fetch_cmd, args.data_dir))
    print("[manifest] 取得済み: " + json.dumps(S.summarize(store.manifest), ensure_ascii=False))
    fixed = store.check_local()
    for ym, ds in fixed.items():
        print(f"[manifest] {S.file_name(ym)} と取得記録が食い違っていた。ファイルに合わせた: "
              f"{len(ds)}日（{ds[:3]}）")
    if args.forget_from:
        dropped = S.forget_days(store.manifest, dt.date.fromisoformat(args.forget_from))
        print(f"[forget] 取得記録 {len(dropped)}日を外す"
              + (f"（{dropped[0]} 〜 {dropped[-1]}）" if dropped else "") + "。月のファイルは消さない")
    if args.reingest_bulk:
        print(f"[reingest] 一括ファイルの取り込みの記録 {len(store.manifest['bulk'])}件を外す")
        store.manifest["bulk"] = {}

    floor = retention_start(today)
    start = dt.date.fromisoformat(args.date_from) if args.date_from else floor
    if start < floor:
        print(f"[window] {start} は分足の範囲（今日の{RETENTION_YEARS}年前から）より前なので {floor} から")
        start = floor
    end = dt.date.fromisoformat(args.date_to) if args.date_to else today

    api = api or Api(resolve_api_key(), pause=args.pause)
    flush = Flusher(store, args.upload_cmd)
    run = Run(api, store, flush, deadline, today)
    report = run.report
    report.update({"mode": "probe" if args.probe_only else "update",
                   "startedJst": started.strftime("%Y-%m-%d %H:%M"), "windowFrom": start.isoformat(),
                   "windowTo": end.isoformat()})
    code = 0
    try:
        days = trading_days(api, start, end)
        report["tradingDays"] = len(days)
        print(f"[calendar] {start} 〜 {end} の営業日 {len(days)}日")

        recent = [d for d in days if d < today] or days
        try:
            info = probe(api, recent[-1])
        except PlanError as exc:
            print("::error title=分足のアドオンが有効でない::"
                  + " ".join(str(exc).split()).replace("::", ":")[:300])
            report["errors"].append("アドオンが有効でない（HTTP 403 not available on your subscription）")
            return 2
        report["probe"] = info
        print(f"[probe] {info['day']} {info['code']}: {info['rows']}行 / 列 {info['columns']}")

        if args.probe_only:
            try:
                entries = list_bulk(api)
            except JQuantsError as exc:
                msg = f"一括ファイルの一覧を引けなかった: {str(exc)[:200]}"
                print(f"::warning title=分足の一括ダウンロード::{msg}")
                report["warnings"].append(msg)
                entries = []
            dated = [e for e in entries if e["period"] is not None]
            print(f"[bulk] 一覧 {len(entries)}件")
            for e in entries[:60]:
                print(f"    {e['key']}  {(e['size'] or 0)/1e6:.1f}MB  {e['lastModified']}")
            report["bulk"] = {"listed": len(entries),
                              "listedFrom": dated[0]["period"][0].isoformat() if dated else None,
                              "listedTo": dated[-1]["period"][1].isoformat() if dated else None,
                              "listedBytes": sum(e["size"] or 0 for e in dated)}
            if dated:
                report["bulkSample"] = probe_bulk_file(api, min(dated, key=lambda e: e["size"] or 0),
                                                       args.data_dir)
            return 0

        if not args.no_bulk:
            run.bulk(start if args.date_from else None)
        if not args.no_api:
            run.api_days(days)

        store.save()
        flush()

        missing = gaps(store.manifest, days)
        # 当日ぶんは公表前でありうる（公表時刻は公式に明記が見つからない）。前の日は出ているはず
        waiting = [d for d in missing if d == today.isoformat()]
        others = [d for d in missing if d != today.isoformat()]
        report["missingInWindow"] = missing
        print(f"\n[check] 範囲 {start} 〜 {end} の営業日 {len(days)}日のうち保存済み "
              f"{len(days) - len(missing)}日")
        if waiting:
            print(f"[check] 当日 {today} はまだ無い（{started:%H:%M} JST 時点。公表前なら次の実行で入る）")
        if others:
            print(f"::warning title=分足の未取得の日::{len(others)}日（{others[:5]}）。次の実行で取り直す")
        before = [d for d in store.manifest["days"] if d < floor.isoformat()]
        report["beyondWindowDays"] = len(before)
        if before:
            print(f"[check] J-Quants の範囲より前（{floor} より前）の保存済み {len(before)}日（ここにしか無い）")
        if report["errors"]:
            code = 1
    finally:
        if store.dirty:
            # 途中で落ちた。書いたデータに合わせて取得記録を保存しておき、上げ残しは
            # ワークフローの受け皿（.pending を読む）がデータ → 取得記録の順に上げる
            try:
                store.save()
            except Exception as exc:                         # noqa: BLE001
                print(f"[warn] 取得記録を保存できなかった: {type(exc).__name__}: {str(exc)[:160]}",
                      file=sys.stderr)
        report["requests"] = api.requests
        report["minutes"] = round((time.monotonic() - t_start) / 60, 1)
        report["store"] = S.summarize(store.manifest)
        report["extraColumns"] = store.manifest.get("extraColumns", {})
        write_pending(store)
        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        print("\n[manifest] 更新後: " + json.dumps(S.summarize(store.manifest), ensure_ascii=False))
        print(f"[done] リクエスト {api.requests}回 / {report['minutes']}分")
    return code


if __name__ == "__main__":
    sys.exit(main())
