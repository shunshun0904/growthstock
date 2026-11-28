#!/usr/bin/env python3
"""
ティックの取得と蓄積（research/ticks.py）のテスト。ネットワークには出ない。

偽の J-Quants（一括ファイルの一覧・署名付きURL・営業日）と、偽の保存先（年ごとの Release を
ディレクトリで表し、作った Release と上げた順番を記録する）で確かめる。

- 形をそろえる: 時刻 HH:MM:SS.ffffff をマイクロ秒の時刻に、価格・数量を数に。仕様の列が欠けたら止める
- 1日ぶんを (Code, TransactionId) で並べる（桁がそろわない日も数の順に）。中身の違う重なりは止める
- 月ごとのファイルを少しずつ読んでも、日付が混ざっていても日ごとに分けられる
- 日ごとのファイルは年ごとの Release（data-tick-YYYY）に、取得記録は data-tick に。上げる順はデータ → 記録
- 取り込み済みのファイルは落とし直さない。新しく作られた月ごとのファイルは、同じ日の日ごとのファイルより後に当てる
- 上げられなかったら記録を上げない（次の実行で取り直す）
- 鍵・署名付きURL・価格の値はログに出さない

  python3 -m unittest tests/test_ticks.py
"""
import contextlib
import datetime as dt
import gzip
import io
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
import urllib.parse

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import minute_bars as MB  # noqa: E402
import ticks as T  # noqa: E402

KEY = "tick-test-key-0123456789"
SIG = "sig=TICK-SECRET"
CODES = ["13010", "130A0", "72030"]
PRICE_MARK = "87654.3"          # ログに出てはいけない値


def is_trading(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in (dt.date(2026, 11, 3), dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def trades_rows(day: dt.date, codes=CODES, per_code=6):
    rows = []
    for ci, code in enumerate(codes):
        for i in range(per_code):
            sess = "01" if i < per_code // 2 else "02"
            hh = 9 if sess == "01" else 13
            price = PRICE_MARK if (ci == 0 and i == 0) else f"{1000 + ci * 10 + i}"
            rows.append([day.isoformat(), code, f"{hh:02d}:{i:02d}:{(i * 7) % 60:02d}.{i * 1111:06d}", sess,
                         price, str(100 * (i + 1)), f"{i * 3 + 21:012d}"])
    return rows


def csv_gz(rows) -> bytes:
    lines = ["Date,Code,Time,SessionDistinction,Price,TradingVolume,TransactionId"]
    lines += [",".join(r) for r in rows]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def month_days(ym: str):
    y, m = int(ym[:4]), int(ym[4:])
    d = dt.date(y, m, 1)
    while d.month == m:
        if is_trading(d):
            yield d
        d += dt.timedelta(days=1)


class Resp:
    def __init__(self, body):
        self._b = io.BytesIO(body)

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeJQuants:
    def __init__(self):
        self.files = {}          # key -> (bytes, lastModified)
        self.downloads = []

    def hist_key(self, ym):
        return f"equities/trades/historical/{ym[:4]}/equities_trades_{ym}.csv.gz"

    def live_key(self, day):
        return f"equities/trades/live/equities_trades_{day:%Y%m%d}.csv.gz"

    def set_month(self, ym, last_modified, codes=CODES, interleave=False):
        days = list(month_days(ym))
        if interleave:
            # 日付が混ざった並び（銘柄ごとに全日 → 次の銘柄）
            rows = [r for c in codes for d in days for r in trades_rows(d, [c])]
        else:
            rows = [r for d in days for r in trades_rows(d, codes)]
        self.files[self.hist_key(ym)] = (csv_gz(rows), last_modified)

    def set_live(self, day, last_modified, codes=CODES):
        self.files[self.live_key(day)] = (csv_gz(trades_rows(day, codes)), last_modified)

    def __call__(self, req, timeout=None):
        u = urllib.parse.urlparse(req.full_url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.netloc == "files.example":
            assert req.get_header("X-api-key") is None, "鍵を J-Quants 以外に送った"
            key = urllib.parse.unquote(u.path.lstrip("/"))
            self.downloads.append(key)
            return Resp(self.files[key][0])
        assert req.get_header("X-api-key") == KEY
        path = u.path.replace("/v2", "", 1)
        if path == "/bulk/list":
            assert q["endpoint"] == T.ENDPOINT
            data = [{"Key": k, "LastModified": lm, "Size": len(b)} for k, (b, lm) in sorted(self.files.items())]
            return Resp(json.dumps({"data": data}).encode())
        if path == "/bulk/get":
            return Resp(json.dumps({"url": f"https://files.example/{urllib.parse.quote(q['key'])}?{SIG}"}).encode())
        if path == "/markets/calendar":
            d, end = dt.date.fromisoformat(q["from"]), dt.date.fromisoformat(q["to"])
            data = []
            while d <= end:
                data.append({"Date": d.isoformat(), "HolDiv": "1" if is_trading(d) else "0"})
                d += dt.timedelta(days=1)
            return Resp(json.dumps({"data": data}).encode())
        raise AssertionError(path)


class Remote:
    """偽の保存先。remote/<tag>/ にファイルを置き、上げた順番と作った Release を記録する。"""

    def __init__(self, root, fail_tag=None):
        self.dir = os.path.join(root, "remote")
        os.makedirs(self.dir)
        self.log = os.path.join(root, "remote.log")
        up = os.path.join(root, "up.py")
        with open(up, "w") as fh:
            fh.write("import shutil, sys, os\n"
                     "remote, log, fail = sys.argv[1], sys.argv[2], sys.argv[3]\n"
                     "tag = sys.argv[4]\n"
                     "if tag == fail: sys.exit(1)\n"
                     "assert os.path.isdir(os.path.join(remote, tag)), 'Release が無い: ' + tag\n"
                     "for p in sys.argv[5:]:\n"
                     "    shutil.copy(p, os.path.join(remote, tag, os.path.basename(p)))\n"
                     "    open(log, 'a').write('up ' + tag + ' ' + os.path.basename(p) + '\\n')\n")
        ens = os.path.join(root, "ensure.py")
        with open(ens, "w") as fh:
            fh.write("import sys, os\n"
                     "remote, log = sys.argv[1], sys.argv[2]\n"
                     "tag = sys.argv[3]\n"
                     "os.makedirs(os.path.join(remote, tag), exist_ok=True)\n"
                     "open(log, 'a').write('ensure ' + tag + '\\n')\n")
        py = shlex.quote(sys.executable)
        q = shlex.quote
        self.upload_cmd = f"{py} {q(up)} {q(self.dir)} {q(self.log)} {q(fail_tag or '-')}"
        self.ensure_cmd = f"{py} {q(ens)} {q(self.dir)} {q(self.log)}"

    def lines(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as fh:
            return [ln.split() for ln in fh.read().splitlines()]

    def files(self, tag):
        p = os.path.join(self.dir, tag)
        return sorted(os.listdir(p)) if os.path.isdir(p) else []


def norm(rows):
    header = ["Date", "Code", "Time", "SessionDistinction", "Price", "TradingVolume", "TransactionId"]
    t = pa.table({h: [r[i] for r in rows] for i, h in enumerate(header)})
    return T.normalize(t)


class TestShape(unittest.TestCase):
    def test_normalize_types(self):
        t, extra = norm(trades_rows(dt.date(2025, 12, 1)))
        self.assertEqual(t.schema, T.SCHEMA)
        self.assertEqual(extra, [])
        self.assertEqual(t["Time"][0].as_py(), dt.time(9, 0, 0, 0))
        self.assertEqual(t["TransactionId"][0].as_py(), "000000000021")
        self.assertEqual(t["SessionDistinction"].unique().to_pylist(), ["01", "02"])

    def test_missing_column_and_bad_time_stop(self):
        with self.assertRaises(ValueError):
            T.normalize(pa.table({"Date": ["2025-12-01"], "Code": ["13010"]}))
        rows = trades_rows(dt.date(2025, 12, 1))[:1]
        rows[0][2] = "9:00:00"
        with self.assertRaises(ValueError):
            norm(rows)

    def test_sort_day(self):
        rows = trades_rows(dt.date(2025, 12, 1))
        t, _ = norm(list(reversed(rows)))
        s = T.sort_day(t)
        keys = list(zip(s["Code"].to_pylist(), s["TransactionId"].to_pylist()))
        self.assertEqual(keys, sorted(keys))
        # 桁のそろわない TransactionId は数の順に
        r2 = [list(r) for r in rows if r[1] == "13010"][:3]
        r2[0][6], r2[1][6], r2[2][6] = "10", "9", "100"
        s2 = T.sort_day(norm(r2)[0])
        self.assertEqual(s2["TransactionId"].to_pylist(), ["9", "10", "100"])

    def test_duplicates(self):
        rows = trades_rows(dt.date(2025, 12, 1))
        same, _ = norm(rows + rows[:2])
        self.assertEqual(T.sort_day(same).num_rows, len(rows))
        bad = [list(rows[0])]
        bad[0][5] = "999999"
        with self.assertRaises(ValueError):
            T.sort_day(norm(rows + bad)[0])

    def test_split_interleaved_month_in_small_blocks(self):
        fake = FakeJQuants()
        fake.set_month("202510", "2025-11-05T00:00:00Z", interleave=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.csv.gz")
            with open(path, "wb") as fh:
                fh.write(fake.files[fake.hist_key("202510")][0])
            work = os.path.join(tmp, "work")
            os.makedirs(work)
            with unittest.mock.patch.object(T, "BLOCK_SIZE", 2048):
                parts, extra, rows = T.split_file(path, work)
            days = list(month_days("202510"))
            self.assertEqual(sorted(parts), days)
            self.assertEqual(rows, len(days) * len(CODES) * 6)
            out = os.path.join(tmp, "d.parquet")
            st = T.finalize_day(parts[days[0]], out)
            self.assertEqual((st["rows"], st["codes"]), (len(CODES) * 6, len(CODES)))
            t = pq.read_table(out)
            self.assertEqual(t.schema, T.SCHEMA)
            self.assertEqual(set(t["Date"].to_pylist()), {days[0]})


class TestRun(unittest.TestCase):
    TODAY = dt.date(2026, 11, 18)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.data = os.path.join(self.root, "data")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, fake, remote, *extra, today=None):
        out = io.StringIO()
        summary = os.path.join(self.root, "summary.json")
        argv = ["--data-dir", self.data, "--upload-cmd", remote.upload_cmd, "--ensure-cmd", remote.ensure_cmd,
                "--summary-json", summary, *extra]
        api = MB.Api(KEY, pause=0, sleep=lambda s: None, urlopen=fake)
        with contextlib.redirect_stdout(out):
            code = T.main(argv, api=api, today=today or self.TODAY)
        with open(summary) as fh:
            return code, out.getvalue(), json.load(fh)

    def test_backfill_years_order_and_no_leaks(self):
        fake = FakeJQuants()
        fake.set_month("202412", "2026-01-14T02:00:00Z")
        fake.set_month("202501", "2026-01-14T02:10:00Z")
        nov = [d for d in month_days("202611") if d < self.TODAY]
        for d in nov:
            fake.set_live(d, f"{d.isoformat()}T07:44:00Z")
        remote = Remote(self.root)
        code, log, rep = self.run_main(fake, remote, "--from", "2024-12-01")
        self.assertEqual(code, 0, log)
        self.assertEqual(rep["bulk"]["ingested"], 2 + len(nov))
        # 年ごとの Release に日ごとのファイル
        self.assertEqual(remote.files("data-tick-2024"), [T.day_file(d) for d in month_days("202412")])
        self.assertEqual(remote.files("data-tick-2025"), [T.day_file(d) for d in month_days("202501")])
        self.assertEqual(remote.files("data-tick-2026"), [T.day_file(d) for d in nov])
        self.assertEqual(remote.files("data-tick"), [T.MANIFEST_NAME])
        # Release は1回ずつ作る。どの回もデータが先、記録が後
        lines = remote.lines()
        ensured = [ln[1] for ln in lines if ln[0] == "ensure"]
        self.assertEqual(sorted(ensured), sorted(set(ensured)))
        ups = [ln for ln in lines if ln[0] == "up"]
        self.assertEqual(ups[-1][1:], ["data-tick", T.MANIFEST_NAME])
        for i, ln in enumerate(ups):
            if ln[1] != "data-tick":
                self.assertIn(["up", "data-tick", T.MANIFEST_NAME], ups[i + 1:])
        m = T.load_manifest(self.data)
        d0 = next(month_days("202412"))
        self.assertEqual(m["days"][d0.isoformat()]["release"], "data-tick-2024")
        self.assertEqual(m["days"][d0.isoformat()]["rows"], len(CODES) * 6)
        self.assertEqual(m["days"][nov[-1].isoformat()]["key"], fake.live_key(nov[-1]))
        # 上げた日のファイルは手元に残さない（2年分でランナーのディスクに収まらない）
        self.assertEqual([n for n in os.listdir(self.data) if n.endswith(".parquet")], [])
        # 当日は日ごとのファイルがまだ無い
        self.assertEqual(rep["missingInWindow"][-1], self.TODAY.isoformat())
        for secret in (KEY, SIG, PRICE_MARK):
            self.assertNotIn(secret, log)
        # 2回目: 取り込み済みは落とし直さない
        fake.downloads.clear()
        code, log, rep = self.run_main(fake, remote, "--from", "2024-12-01")
        self.assertEqual(code, 0, log)
        self.assertEqual(fake.downloads, [])

    def test_newer_monthly_file_wins(self):
        fake = FakeJQuants()
        octd = list(month_days("202610"))
        for d in octd[:2]:
            fake.set_live(d, f"{d.isoformat()}T07:44:00Z", codes=CODES)
        fake.set_month("202610", "2026-11-05T00:02:00Z", codes=CODES[:1])
        remote = Remote(self.root)
        code, log, rep = self.run_main(fake, remote, "--from", "2026-10-01")
        self.assertEqual(code, 0, log)
        t = pq.read_table(os.path.join(remote.dir, "data-tick-2026", T.day_file(octd[0])))
        self.assertEqual(set(t["Code"].to_pylist()), set(CODES[:1]))
        m = T.load_manifest(self.data)
        self.assertEqual(m["days"][octd[0].isoformat()]["key"], fake.hist_key("202610"))

    def test_upload_failure_keeps_manifest_back(self):
        fake = FakeJQuants()
        fake.set_month("202501", "2026-01-14T02:10:00Z")
        remote = Remote(self.root, fail_tag="data-tick-2025")
        with self.assertRaises(Exception):
            self.run_main(fake, remote, "--from", "2025-01-01")
        self.assertEqual(remote.files("data-tick"), [])                 # 記録は上げていない
        self.assertNotIn(fake.hist_key("202501"), T.load_manifest(self.data)["bulk"])
        # 次の実行（上げられる）で取り直す
        fake.downloads.clear()
        again = os.path.join(self.root, "again")
        os.makedirs(again)
        ok = Remote(again)
        code, log, rep = self.run_main(fake, ok, "--from", "2025-01-01")
        self.assertEqual(code, 0, log)
        self.assertEqual(fake.downloads, [fake.hist_key("202501")])
        self.assertEqual(ok.files("data-tick-2025"), [T.day_file(d) for d in month_days("202501")])

    def test_forget_from_reingests_the_source_file(self):
        fake = FakeJQuants()
        fake.set_month("202501", "2026-01-14T02:10:00Z")
        remote = Remote(self.root)
        self.run_main(fake, remote, "--from", "2025-01-01")
        fake.downloads.clear()
        code, log, rep = self.run_main(fake, remote, "--from", "2025-01-01", "--forget-from", "2025-01-20")
        self.assertEqual(code, 0, log)
        self.assertEqual(fake.downloads, [fake.hist_key("202501")])

    def test_probe_only_downloads_nothing(self):
        fake = FakeJQuants()
        fake.set_month("202501", "2026-01-14T02:10:00Z")
        remote = Remote(self.root)
        code, log, rep = self.run_main(fake, remote, "--probe-only")
        self.assertEqual(code, 0, log)
        self.assertEqual(rep["bulk"]["planned"], 1)
        self.assertEqual(fake.downloads, [])
        self.assertEqual(remote.lines(), [])

    def test_no_addon_stops(self):
        def plan(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 403, "x", {}, io.BytesIO(
                b'{"message": "This API is not available on your subscription."}'))
        remote = Remote(self.root)
        code, log, rep = self.run_main(plan, remote)
        self.assertEqual(code, 2)
        self.assertIn("アドオンが有効でない", log)


class TestWorkflow(unittest.TestCase):
    """.github/workflows/update-ticks.yml の安全策（文字列で確かめる）。"""

    def setUp(self):
        with open(os.path.join(ROOT, ".github", "workflows", "update-ticks.yml"), encoding="utf-8") as fh:
            self.text = fh.read()

    def test_safety(self):
        self.assertRegex(self.text, r"workflow_run:\s*\n\s*workflows: \['Update Data Store'\]")
        # 置き場は下書きの Release（取得記録の data-tick も、年ごとの data-tick-YYYY も下書きで作り、公開なら止める）
        self.assertIn("GH_ENSURE_DRAFT: '1'", self.text)
        self.assertIn("scripts/gh_release_upload.sh", self.text)
        self.assertIn("scripts/gh_release_ensure.sh", self.text)
        self.assertNotIn("gh release upload ", self.text.replace("# ", ""))
        # 新しい鍵は要らない（Actions の鍵）。git にはコミットしない
        self.assertRegex(self.text, r"\npermissions:\s*\n\s*contents: write")
        self.assertIn("GH_TOKEN: ${{ github.token }}", self.text)
        self.assertNotIn("MINUTE_DATA_TOKEN", self.text)
        self.assertNotIn("git push", self.text)
        self.assertEqual(sorted(set(re.findall(r"secrets\.([A-Z_]+)", self.text))), ["JQUANTS_API"])


if __name__ == "__main__":
    unittest.main()
