#!/usr/bin/env python3
"""
分足の取得と蓄積（research/minute_bars.py・research/minute_store.py）のテスト。ネットワークには出ない。

偽の J-Quants（営業日・分足のページ送り・一括ファイルの一覧と署名付きURL・429・契約外）と、
偽の保存先（取り寄せとアップロードの順番を記録する）で確かめる。

- 保存の形: 見出しの別名・日付と時刻の書き方・欠測の記号をそろえる。仕様の列が欠けたら止める
- 置き換えは日ごと。同じキーで中身の違う行は止める
- 保存先にあるはずの月のファイルが手元に無いまま書かない（新しい日だけで1か月を上書きしない）
- 取得記録はファイルに合わせて直す
- 一括ファイル: 期間の読み取り、取り込み済み・更新の見分け、署名付きURLに鍵を送らずログにも出さない
- API: 429 は1分以上待つ、契約外（403）で止める、ページ送りを最後まで辿る、0行の日は取得済みにしない
- 上げる順番はデータ → 取得記録

  python3 -m unittest tests/test_minute_bars.py
"""
import contextlib
import datetime as dt
import gzip
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import minute_bars as M  # noqa: E402
import minute_store as S  # noqa: E402
from jquants_data_fetcher import AuthError, JQuantsError  # noqa: E402

KEY = "test-api-key-0123456789"
SIGNATURE = "sig=VERY-SECRET-SIGNATURE"
CODES = ["86970", "72030", "130A0"]
TIMES = ["09:00", "09:01", "09:02", "11:30", "12:30", "15:30"]


def is_trading(d: dt.date) -> bool:
    return d.weekday() < 5 and d != dt.date(2026, 11, 3)        # 文化の日


def bars_for(day: dt.date, codes=CODES, times=TIMES):
    """その日の分足（dict の並び）。値は日と銘柄と時刻から決まる。"""
    out = []
    for ci, code in enumerate(codes):
        for ti, t in enumerate(times):
            base = 1000.0 + ci * 100 + day.day + ti
            out.append({"Date": day.isoformat(), "Time": t, "Code": code,
                        "O": base, "H": base + 2, "L": base - 1, "C": base + 1,
                        "Vo": 100.0 * (ti + 1), "Va": 100.0 * (ti + 1) * base})
    return out


def csv_gz(rows, header=("Date", "Time", "Code", "O", "H", "L", "C", "Vo", "Va")) -> bytes:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in ("Date", "Time", "Code", "O", "H", "L", "C", "Vo", "Va")))
    return gzip.compress(("\ufeff" + "\n".join(lines) + "\n").encode("utf-8"))


class Resp:
    def __init__(self, body: bytes):
        self._b = io.BytesIO(body)

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(url, code, body):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(json.dumps(body).encode()))


class FakeJQuants:
    """J-Quants の偽物。urlopen の代わりに渡す。"""

    def __init__(self, today: dt.date, bulk_months=(), page_size=7, unpublished=(),
                 plan=True, fail_429_once=False, partial=None):
        self.today = today
        self.bulk = {}                       # key -> (bytes, lastModified)
        for ym in bulk_months:
            self.set_bulk(ym, "2026-11-01T00:00:00Z")
        self.page_size = page_size
        self.unpublished = set(unpublished)
        self.plan = plan
        self.fail_429_once = fail_429_once
        self.partial = partial or {}         # 日 -> 返す銘柄
        self.calls = []
        self.downloads = []

    def bulk_key(self, ym: str) -> str:
        return f"equities/bars/minute/historical/{ym[:4]}/equities_bars_minute_{ym}.csv.gz"

    def set_bulk(self, ym: str, last_modified: str, codes=CODES):
        y, m = int(ym[:4]), int(ym[4:])
        d = dt.date(y, m, 1)
        rows = []
        while d.month == m:
            if is_trading(d):
                rows += bars_for(d, codes)
            d += dt.timedelta(days=1)
        self.bulk[self.bulk_key(ym)] = (csv_gz(rows), last_modified)

    def __call__(self, req, timeout=None):
        url = req.full_url
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.netloc == "files.example":
            assert req.get_header("X-api-key") is None, "鍵を J-Quants 以外に送った"
            key = urllib.parse.unquote(u.path.lstrip("/"))
            self.downloads.append(key)
            return Resp(self.bulk[key][0])
        assert req.get_header("X-api-key") == KEY
        path = u.path.replace("/v2", "", 1)
        self.calls.append((path, q))
        if self.fail_429_once and path == S.ENDPOINT and "code" not in q:
            self.fail_429_once = False
            raise http_error(url, 429, {"message": "Too Many Requests"})
        if path == "/markets/calendar":
            d, end = dt.date.fromisoformat(q["from"]), dt.date.fromisoformat(q["to"])
            data = []
            while d <= end:
                data.append({"Date": d.isoformat(), "HolDiv": "1" if is_trading(d) else "0"})
                d += dt.timedelta(days=1)
            return Resp(json.dumps({"data": data}).encode())
        if path == S.ENDPOINT:
            if not self.plan:
                raise http_error(url, 403, {"message": "This API is not available on your subscription."})
            day = dt.date.fromisoformat(q["date"])
            if day in self.unpublished:
                return Resp(json.dumps({"data": []}).encode())
            if "code" in q:
                return Resp(json.dumps({"data": [r for r in bars_for(day) if r["Code"] == q["code"]]}).encode())
            rows = bars_for(day, self.partial.get(day, CODES))
            start = int(q.get("pagination_key", "0") or 0)
            page = rows[start:start + self.page_size]
            body = {"data": page}
            if start + self.page_size < len(rows):
                body["pagination_key"] = str(start + self.page_size)
            return Resp(json.dumps(body).encode())
        if path == "/bulk/list":
            assert q["endpoint"] == S.ENDPOINT
            data = [{"Key": k, "LastModified": lm, "Size": len(b)} for k, (b, lm) in sorted(self.bulk.items())]
            return Resp(json.dumps({"data": data}).encode())
        if path == "/bulk/get":
            key = q["key"]
            return Resp(json.dumps({"url": f"https://files.example/{urllib.parse.quote(key)}?{SIGNATURE}"}).encode())
        raise AssertionError(f"知らないパス {path}")


def api_for(fake, sleeps=None):
    rec = sleeps if sleeps is not None else []
    return M.Api(KEY, pause=0, sleep=rec.append, urlopen=fake)


class RemoteStore:
    """偽の保存先（Release）。アップロードと取り寄せのスクリプトを temp に書く。"""

    def __init__(self, root: str):
        self.dir = os.path.join(root, "remote")
        os.makedirs(self.dir)
        self.log = os.path.join(root, "upload.log")
        up = os.path.join(root, "up.py")
        with open(up, "w") as fh:
            fh.write("import shutil, sys, os\n"
                     "remote, log = sys.argv[1], sys.argv[2]\n"
                     "for p in sys.argv[3:]:\n"
                     "    shutil.copy(p, os.path.join(remote, os.path.basename(p)))\n"
                     "    open(log, 'a').write(os.path.basename(p) + '\\n')\n")
        down = os.path.join(root, "down.py")
        with open(down, "w") as fh:
            fh.write("import shutil, sys, os\n"
                     "remote, local = sys.argv[1], sys.argv[2]\n"
                     "for n in sys.argv[3:]:\n"
                     "    src = os.path.join(remote, n)\n"
                     "    if os.path.exists(src): shutil.copy(src, os.path.join(local, n))\n")
        py = shlex.quote(sys.executable)
        self.upload_cmd = f"{py} {shlex.quote(up)} {shlex.quote(self.dir)} {shlex.quote(self.log)}"
        self.down_script = down

    def fetch_cmd(self, local: str) -> str:
        return f"{shlex.quote(sys.executable)} {shlex.quote(self.down_script)} {shlex.quote(self.dir)} {shlex.quote(local)}"

    def uploaded(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as fh:
            return fh.read().split()


def norm(rows):
    t, extra = S.normalize(S.records_to_table(rows))
    return t


# --------------------------------------------------------------------------- #
# 保存の形
# --------------------------------------------------------------------------- #

class TestNormalize(unittest.TestCase):
    def test_aliases_formats_and_null_markers(self):
        t = pa.table({
            "\ufeffdate": ["20260601", "2026/06/01", "2026-06-01T00:00:00"],
            "TIME": ["9:00", "0901", "09:02:00"],
            "Code": [86970, 72030, 86970],
            "Open": ["1", "", "-"], "High": [2.0, 3.0, 4.0], "Low": [1, 2, 3],
            "Close": ["1.5", "-2.5", "3"], "Volume": [100, 200, 300], "TurnoverValue": [1.0, 2.0, 3.0],
            "Extra": ["x", "y", "z"],
        })
        out, extra = S.normalize(t)
        self.assertEqual(out.schema, S.SCHEMA)
        self.assertEqual(extra, ["Extra"])
        self.assertEqual(out["Date"].to_pylist(), [dt.date(2026, 6, 1)] * 3)
        self.assertEqual(out["Time"].to_pylist(), ["09:00", "09:01", "09:02"])
        self.assertEqual(out["Code"].to_pylist(), ["86970", "72030", "86970"])
        self.assertEqual(out["O"].to_pylist(), [1.0, None, None])
        self.assertEqual(out["C"].to_pylist(), [1.5, -2.5, 3.0])          # 負の数は欠測にしない

    def test_missing_column_stops(self):
        t = pa.table({"Date": ["2026-06-01"], "Time": ["09:00"], "Code": ["86970"],
                      "O": [1.0], "H": [1.0], "L": [1.0], "C": [1.0], "Vo": [1.0]})
        with self.assertRaises(ValueError):
            S.normalize(t)

    def test_bad_time_stops(self):
        rows = bars_for(dt.date(2026, 6, 1))[:1]
        rows[0]["Time"] = "9時"
        with self.assertRaises(ValueError):
            norm(rows)

    def test_empty(self):
        out, extra = S.normalize(S.records_to_table([]))
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(out.schema, S.SCHEMA)

    def test_dedupe(self):
        rows = bars_for(dt.date(2026, 6, 1))
        same = norm(rows + rows[:2])
        self.assertEqual(S.dedupe(same).num_rows, len(rows))
        bad = [dict(r) for r in rows[:1]]
        bad[0]["C"] = 9999.0
        with self.assertRaises(ValueError):
            S.dedupe(norm(rows + bad))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_replace_and_append(self):
        st = S.MinuteStore(self.dir)
        d1, d2, d3 = dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)
        st.write_days(norm(bars_for(d1) + bars_for(d2)), "bulk", key="k")
        # d2 を2銘柄だけで置き換え（日ごと丸ごと）、d3 を足す
        st.write_days(norm(bars_for(d2, CODES[:2]) + bars_for(d3)), "api")
        t = pq.read_table(S.file_path(self.dir, "202606"))
        self.assertEqual(t.schema, S.SCHEMA)
        st2 = S.day_stats(t)
        self.assertEqual(st2["2026-06-01"]["codes"], 3)
        self.assertEqual(st2["2026-06-02"]["codes"], 2)
        self.assertEqual(st2["2026-06-03"]["rows"], len(bars_for(d3)))
        # 並びは (Date, Code, Time)
        keys = list(zip(t["Date"].to_pylist(), t["Code"].to_pylist(), t["Time"].to_pylist()))
        self.assertEqual(keys, sorted(keys))
        m = st.manifest
        self.assertEqual(m["days"]["2026-06-01"]["source"], "bulk")
        self.assertEqual(m["days"]["2026-06-02"]["source"], "api")
        self.assertEqual(m["files"]["minute_202606.parquet"]["rows"], t.num_rows)
        self.assertEqual(m["files"]["minute_202606.parquet"]["days"], 3)
        self.assertEqual(st.dirty, {"minute_202606.parquet"})

    def test_days_split_into_months(self):
        st = S.MinuteStore(self.dir)
        st.write_days(norm(bars_for(dt.date(2026, 5, 29)) + bars_for(dt.date(2026, 6, 1))), "api")
        self.assertEqual(S.local_months(self.dir), ["202605", "202606"])

    def test_refuses_to_overwrite_month_missing_locally(self):
        st = S.MinuteStore(self.dir)
        st.manifest["files"]["minute_202606.parquet"] = {"rows": 10}
        called = []
        st.fetch_file = called.append
        with self.assertRaises(RuntimeError):
            st.write_days(norm(bars_for(dt.date(2026, 6, 5))), "api")
        self.assertEqual(called, ["minute_202606.parquet"])
        self.assertFalse(os.path.exists(S.file_path(self.dir, "202606")))

    def test_fetched_month_is_merged_not_replaced(self):
        remote = tempfile.TemporaryDirectory()
        try:
            st = S.MinuteStore(remote.name)
            st.write_days(norm(bars_for(dt.date(2026, 6, 1))), "bulk")
            st.save()
            # 手元は記録だけを持つ（ワークフローの最初の状態）
            import shutil
            shutil.copy(S.manifest_path(remote.name), S.manifest_path(self.dir))

            def fetch(name):
                shutil.copy(os.path.join(remote.name, name), os.path.join(self.dir, name))
            local = S.MinuteStore(self.dir, fetch_file=fetch)
            local.write_days(norm(bars_for(dt.date(2026, 6, 2))), "api")
            days = S.day_stats(pq.read_table(S.file_path(self.dir, "202606")))
            self.assertEqual(sorted(days), ["2026-06-01", "2026-06-02"])
        finally:
            remote.cleanup()

    def test_manifest_follows_the_file(self):
        st = S.MinuteStore(self.dir)
        st.write_days(norm(bars_for(dt.date(2026, 6, 1)) + bars_for(dt.date(2026, 6, 2))), "bulk")
        st.manifest["days"]["2026-06-03"] = {"rows": 5, "source": "api"}   # ファイルに無い
        st.manifest["days"].pop("2026-06-01")                             # ファイルにある
        fixed = st.check_against_manifest("202606")
        self.assertEqual(fixed, ["2026-06-01", "2026-06-03"])
        self.assertNotIn("2026-06-03", st.manifest["days"])
        self.assertEqual(st.manifest["days"]["2026-06-01"]["source"], "file")

    def test_check_local_marks_unknown_files_for_upload(self):
        st = S.MinuteStore(self.dir)
        st.write_days(norm(bars_for(dt.date(2026, 6, 1))), "bulk")
        fresh = S.MinuteStore(self.dir, manifest=S.empty_manifest())
        fixed = fresh.check_local()
        self.assertEqual(list(fixed), ["202606"])
        self.assertIn("minute_202606.parquet", fresh.dirty)
        self.assertIn("minute_202606.parquet", fresh.manifest["files"])

    def test_forget_and_save_roundtrip(self):
        st = S.MinuteStore(self.dir)
        st.write_days(norm(bars_for(dt.date(2026, 6, 1)) + bars_for(dt.date(2026, 6, 2))), "bulk")
        dropped = S.forget_days(st.manifest, dt.date(2026, 6, 2))
        self.assertEqual(dropped, ["2026-06-02"])
        st.save()
        m = S.load_manifest(self.dir)
        self.assertEqual(sorted(m["days"]), ["2026-06-01"])
        self.assertEqual(m["summary"]["days"], 1)
        self.assertTrue(st.unflushed)

    def test_read_minutes(self):
        st = S.MinuteStore(self.dir)
        st.write_days(norm(bars_for(dt.date(2026, 5, 29)) + bars_for(dt.date(2026, 6, 1))), "bulk")
        t = S.read_minutes(self.dir, dt.date(2026, 6, 1), dt.date(2026, 6, 30), ["8697"])
        self.assertEqual(set(t["Code"].to_pylist()), {"86970"})
        self.assertEqual(set(t["Date"].to_pylist()), {dt.date(2026, 6, 1)})


# --------------------------------------------------------------------------- #
# 取得
# --------------------------------------------------------------------------- #

class TestHelpers(unittest.TestCase):
    def test_bulk_period(self):
        self.assertEqual(M.bulk_period("a/b/equities_bars_minute_202602.csv.gz"),
                         (dt.date(2026, 2, 1), dt.date(2026, 2, 28)))
        self.assertEqual(M.bulk_period("x/2025/equities_bars_minute_202512.csv.gz"),
                         (dt.date(2025, 12, 1), dt.date(2025, 12, 31)))
        self.assertEqual(M.bulk_period("x/equities_bars_minute_20250107.csv.gz"),
                         (dt.date(2025, 1, 7), dt.date(2025, 1, 7)))
        self.assertIsNone(M.bulk_period("x/equities_bars_minute_202513.csv.gz"))
        self.assertIsNone(M.bulk_period("x/readme.txt"))

    def test_plan_bulk(self):
        e = [{"key": "k1", "lastModified": "t1", "size": 10, "period": (dt.date(2026, 1, 1), dt.date(2026, 1, 31))},
             {"key": "k2", "lastModified": "t2", "size": 20, "period": (dt.date(2026, 2, 1), dt.date(2026, 2, 28))},
             {"key": "k3", "lastModified": "t3", "size": 30, "period": None}]
        m = {"bulk": {"k1": {"lastModified": "t1", "size": 10}, "k2": {"lastModified": "old", "size": 20}}}
        self.assertEqual([x["key"] for x in M.plan_bulk(e, m)], ["k2"])
        self.assertEqual([x["key"] for x in M.plan_bulk(e, {"bulk": {}})], ["k1", "k2"])
        self.assertEqual([x["key"] for x in M.plan_bulk(e, {"bulk": {}}, dt.date(2026, 2, 1))], ["k2"])

    def test_retention_start(self):
        self.assertEqual(M.retention_start(dt.date(2026, 11, 18)), dt.date(2024, 11, 19))
        self.assertEqual(M.retention_start(dt.date(2028, 2, 29)), dt.date(2026, 3, 2))


class TestApi(unittest.TestCase):
    def test_429_waits_a_minute(self):
        fake = FakeJQuants(dt.date(2026, 11, 18), fail_429_once=True)
        sleeps = []
        api = api_for(fake, sleeps)
        table, extra, pages = M.fetch_day(api, dt.date(2026, 11, 17))
        self.assertEqual(table.num_rows, len(bars_for(dt.date(2026, 11, 17))))
        self.assertIn(M.RATE_LIMIT_WAITS[0], sleeps)
        self.assertGreaterEqual(M.RATE_LIMIT_WAITS[0], 60)
        self.assertEqual(pages, -(-len(bars_for(dt.date(2026, 11, 17))) // fake.page_size))

    def test_plan_error_and_auth_error(self):
        api = api_for(FakeJQuants(dt.date(2026, 11, 18), plan=False))
        with self.assertRaises(M.PlanError):
            M.probe(api, dt.date(2026, 11, 17))

        def unauthorized(req, timeout=None):
            raise http_error(req.full_url, 401, {"message": f"bad key {KEY}"})
        api = M.Api(KEY, pause=0, sleep=lambda s: None, urlopen=unauthorized)
        with self.assertRaises(AuthError) as cm:
            api.get("/markets/calendar", {"from": "2026-11-01"})
        self.assertNotIn(KEY, str(cm.exception))

    def test_server_errors_are_retried_then_raised(self):
        n = {"calls": 0}

        def broken(req, timeout=None):
            n["calls"] += 1
            raise http_error(req.full_url, 503, {"message": "busy"})
        api = M.Api(KEY, pause=0, sleep=lambda s: None, urlopen=broken)
        with self.assertRaises(JQuantsError):
            api.get("/markets/calendar")
        self.assertEqual(n["calls"], len(M.SERVER_ERROR_WAITS) + 1)

    def test_repeated_pagination_key_stops(self):
        def loop(req, timeout=None):
            return Resp(json.dumps({"data": [], "pagination_key": "same"}).encode())
        api = M.Api(KEY, pause=0, sleep=lambda s: None, urlopen=loop)
        with self.assertRaises(JQuantsError):
            list(api.pages(S.ENDPOINT, {"date": "2026-11-17"}))


class TestRun(unittest.TestCase):
    TODAY = dt.date(2026, 11, 18)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.data = os.path.join(self.root, "data")
        self.remote = RemoteStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, fake, *extra):
        out = io.StringIO()
        summary = os.path.join(self.root, "summary.json")
        argv = ["--data-dir", self.data, "--from", "2026-09-01", "--upload-cmd", self.remote.upload_cmd,
                "--fetch-cmd", self.remote.fetch_cmd(self.data), "--summary-json", summary, *extra]
        with contextlib.redirect_stdout(out):
            code = M.main(argv, api=api_for(fake), today=self.TODAY)
        with open(summary) as fh:
            return code, out.getvalue(), json.load(fh)

    def test_backfill_then_daily(self):
        fake = FakeJQuants(self.TODAY, bulk_months=["202609", "202610"], unpublished={self.TODAY})
        code, log, rep = self.run_main(fake)
        self.assertEqual(code, 0, log)
        # 一括の2か月 + 11月の営業日（3日は祝日、18日は公表前）
        nov = [dt.date(2026, 11, d) for d in range(1, 18) if is_trading(dt.date(2026, 11, d))]
        m = S.load_manifest(self.data)
        self.assertEqual(rep["bulk"]["ingested"], 2)
        self.assertEqual(rep["api"]["added"], len(nov))
        self.assertEqual(rep["api"]["held"], 1)                          # 当日（公表前）
        self.assertNotIn(self.TODAY.isoformat(), m["days"])
        self.assertEqual(rep["missingInWindow"], [self.TODAY.isoformat()])
        self.assertEqual({m["days"][d.isoformat()]["source"] for d in nov}, {"api"})
        self.assertEqual(m["days"]["2026-09-01"]["source"], "bulk")
        self.assertEqual(m["days"]["2026-11-17"]["rows"], len(bars_for(dt.date(2026, 11, 17))))
        # 保存先: 月のファイル3つと記録。どの回もデータが先、記録が最後
        self.assertEqual(sorted(os.listdir(self.remote.dir)),
                         ["minute_202609.parquet", "minute_202610.parquet", "minute_202611.parquet",
                          "minute_manifest.json"])
        ups = self.remote.uploaded()
        self.assertEqual(ups[-1], "minute_manifest.json")
        self.assertNotEqual(ups[0], "minute_manifest.json")
        for i, n in enumerate(ups):
            if n != "minute_manifest.json":                  # データを上げたら、後で必ず記録も上げる
                self.assertIn("minute_manifest.json", ups[i + 1:])
        # 署名付きURLと鍵はログに出さない
        self.assertNotIn(SIGNATURE, log)
        self.assertNotIn(KEY, log)
        with open(os.path.join(self.data, ".pending")) as fh:
            self.assertEqual(fh.read(), "")

        # 翌日: 手元を空にして記録だけを持って始める（ワークフローと同じ）。18日が出た
        for n in os.listdir(self.data):
            if n != S.MANIFEST_NAME:
                os.remove(os.path.join(self.data, n))
        fake2 = FakeJQuants(dt.date(2026, 11, 19), bulk_months=["202609", "202610"])
        self.TODAY = dt.date(2026, 11, 19)
        fake2.unpublished = {dt.date(2026, 11, 19)}
        code, log, rep = self.run_main(fake2)
        self.assertEqual(code, 0, log)
        self.assertEqual(fake2.downloads, [])                              # 一括は取り込み済み
        self.assertEqual(rep["api"]["added"], 1)
        m = S.load_manifest(self.data)
        self.assertIn("2026-11-18", m["days"])
        self.assertIn("2026-11-02", m["days"])                             # 前の日は消えない
        days = S.day_stats(pq.read_table(os.path.join(self.remote.dir, "minute_202611.parquet")))
        self.assertEqual(len(days), len(nov) + 1)
        self.assertIn("取り寄せる", log)

    def test_bulk_update_is_reingested(self):
        fake = FakeJQuants(self.TODAY, bulk_months=["202609"], unpublished={self.TODAY})
        self.run_main(fake, "--no-api")
        fake.set_bulk("202609", "2026-11-18T00:00:00Z", codes=CODES[:2])  # 訂正で作り直された
        code, log, rep = self.run_main(fake, "--no-api")
        self.assertEqual(rep["bulk"]["ingested"], 1)
        t = pq.read_table(os.path.join(self.remote.dir, "minute_202609.parquet"))
        self.assertEqual(set(t["Code"].to_pylist()), set(CODES[:2]))

    def test_partial_day_is_not_recorded(self):
        fake = FakeJQuants(self.TODAY, bulk_months=["202609", "202610"], unpublished={self.TODAY},
                           partial={dt.date(2026, 11, 17): ["86970"]})
        with unittest.mock.patch.object(M, "PARTIAL_RATIO", 0.5):
            code, log, rep = self.run_main(fake)
        m = S.load_manifest(self.data)
        self.assertNotIn("2026-11-17", m["days"])
        self.assertIn("2026-11-16", m["days"])
        self.assertTrue(any("2026-11-17" in w for w in rep["warnings"]))

    def test_no_addon_stops_early(self):
        fake = FakeJQuants(self.TODAY, plan=False)
        code, log, rep = self.run_main(fake)
        self.assertEqual(code, 2)
        self.assertIn("アドオンが有効でない", log)
        self.assertEqual(self.remote.uploaded(), [])

    def test_probe_only_saves_nothing(self):
        fake = FakeJQuants(self.TODAY, bulk_months=["202609", "202610"])
        code, log, rep = self.run_main(fake, "--probe-only")
        self.assertEqual(code, 0)
        self.assertEqual(rep["bulk"]["listed"], 2)
        self.assertEqual(rep["probe"]["rows"], len(TIMES))
        # 一括ファイルを1つだけ落として形を見る（見出し・行数・日数。値は出さない）
        self.assertEqual(len(fake.downloads), 1)
        sample = rep["bulkSample"]
        self.assertEqual(sample["header"], ["Date", "Time", "Code", "O", "H", "L", "C", "Vo", "Va"])
        self.assertTrue(sample["gzip"])
        self.assertEqual(sample["rowsPerDayMedian"], len(CODES) * len(TIMES))
        self.assertEqual((sample["timeFirst"], sample["timeLast"]), ("09:00", "15:30"))
        self.assertNotIn("error", sample)
        self.assertEqual(self.remote.uploaded(), [])
        self.assertEqual([n for n in os.listdir(self.data) if not n.startswith(".pending")], [])
        self.assertNotIn(SIGNATURE, log)

    def test_japanese_header_is_understood(self):
        rows = bars_for(dt.date(2026, 6, 1))
        path = os.path.join(self.root, "ja.csv.gz")
        with open(path, "wb") as fh:
            fh.write(csv_gz(rows, header=("日付", "時刻", "銘柄コード", "始値", "高値", "安値", "終値", "出来高", "売買代金")))
        t, extra = M.read_bulk_csv(path)
        self.assertEqual(t.num_rows, len(rows))
        self.assertEqual(extra, [])

    def test_truncated_gzip_is_rejected(self):
        path = os.path.join(self.root, "cut.csv.gz")
        whole = csv_gz(bars_for(dt.date(2026, 6, 1)))
        with open(path, "wb") as fh:
            fh.write(whole[: len(whole) // 2])
        with self.assertRaises((EOFError, OSError)):
            M.read_bulk_csv(path)

    def test_bulk_unavailable_falls_back_to_api(self):
        fake = FakeJQuants(self.TODAY, unpublished={self.TODAY})
        real = fake.__call__

        def no_bulk(req, timeout=None):
            if "/bulk/list" in req.full_url:
                raise http_error(req.full_url, 400, {"message": "bad endpoint"})
            return real(req, timeout)
        api = M.Api(KEY, pause=0, sleep=lambda s: None, urlopen=no_bulk)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = M.main(["--data-dir", self.data, "--from", "2026-11-01"], api=api, today=self.TODAY)
        self.assertEqual(code, 0, out.getvalue())
        m = S.load_manifest(self.data)
        self.assertIn("2026-11-17", m["days"])
        self.assertIn("一括ファイルの一覧を引けなかった", out.getvalue())


    def test_bulk_not_in_plan_falls_back_to_api(self):
        fake = FakeJQuants(self.TODAY, bulk_months=["202610"], unpublished={self.TODAY})
        real = fake.__call__

        def plan_bulk_get(req, timeout=None):
            if "/bulk/get" in req.full_url:
                raise http_error(req.full_url, 403, {"message": "This API is not available on your subscription."})
            return real(req, timeout)
        api = M.Api(KEY, pause=0, sleep=lambda s: None, urlopen=plan_bulk_get)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = M.main(["--data-dir", self.data, "--from", "2026-10-26"], api=api, today=self.TODAY)
        self.assertEqual(code, 0, out.getvalue())
        m = S.load_manifest(self.data)
        self.assertEqual(m["days"]["2026-10-30"]["source"], "api")
        self.assertEqual(m["bulk"], {})
        self.assertIn("契約で使えない", out.getvalue())


class TestWorkflow(unittest.TestCase):
    """.github/workflows/update-minute-bars.yml の安全策（文字列で確かめる。CI に YAML の部品が無い）。"""

    def setUp(self):
        with open(os.path.join(ROOT, ".github", "workflows", "update-minute-bars.yml"), encoding="utf-8") as fh:
            self.text = fh.read()

    def test_runs_after_the_daily_ingest(self):
        self.assertRegex(self.text, r"workflow_run:\s*\n\s*workflows: \['Update Data Store'\]")

    def test_refuses_public_storage(self):
        self.assertIn("--jq '.private'", self.text)
        self.assertIn('if [ "$priv" != "true" ]', self.text)

    def test_does_not_write_this_repository(self):
        self.assertRegex(self.text, r"\npermissions:\s*\n\s*contents: read")
        self.assertNotIn("git push", self.text)
        self.assertIn("secrets.MINUTE_DATA_TOKEN", self.text)

    def test_leftovers_go_data_first(self):
        step = self.text[self.text.index("name: Upload leftovers"):self.text.index("name: Summary")]
        self.assertLess(step.index('"${data[@]}"'), step.index('"$man"'))

    def test_probe_does_not_touch_storage(self):
        for name in ("name: Check storage", "name: Download manifest"):
            i = self.text.index(name)
            self.assertIn("if: steps.mode.outputs.mode != 'probe'", self.text[i:i + 200], name)
        step = self.text[self.text.index("name: Fetch minute bars"):self.text.index("name: Upload leftovers")]
        probe_args = step[step.index('if [ "$MODE" = "probe" ]'):]
        self.assertNotIn("--upload-cmd", probe_args.split("fi", 1)[0])

    def test_downloads_only_the_manifest_up_front(self):
        step = self.text[self.text.index("name: Download manifest"):self.text.index("name: Fetch minute bars")]
        self.assertIn("'minute_manifest.json'", step)
        self.assertIn("--fetch-cmd", self.text)


import unittest.mock  # noqa: E402


if __name__ == "__main__":
    unittest.main()
