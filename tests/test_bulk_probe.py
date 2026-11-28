#!/usr/bin/env python3
"""
research/bulk_probe.py（一括ファイルを取り込む前に測る）のテスト。ネットワークには出ない。

- 一覧を種類（historical / live）ごとに数え、作られた時刻を JST で出す
- 1ファイルの形: 見出し・行数・日数・銘柄数・時刻の範囲・区分の値・キーの一意性・列の形
- 時刻 HH:MM:SS.ffffff を time64[us] にする（形の違う行があれば止める）
- 保存の形の大きさの見積もりを出す
- 株価・出来高の値そのものと、署名付きURLはログに出さない

  python3 -m unittest tests/test_bulk_probe.py
"""
import contextlib
import datetime as dt
import gzip
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.parse

import pyarrow as pa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import bulk_probe as P  # noqa: E402
import minute_bars as MB  # noqa: E402

KEY = "probe-key-0123456789"
SIG = "sig=TOP-SECRET"
PRICE = "98765.4"           # ログに出てはいけない値（ほかの数字と紛れない値にする）


def trades_csv(day="2025-12-01", codes=("13010", "130A0"), per_code=50) -> bytes:
    lines = ["Date,Code,Time,SessionDistinction,Price,TradingVolume,TransactionId"]
    for code in codes:
        for i in range(per_code):
            sess = "01" if i < per_code // 2 else "02"
            hh = 9 if sess == "01" else 13
            t = f"{hh:02d}:{i % 60:02d}:{(i * 7) % 60:02d}.{(i * 12345) % 1000000:06d}"
            price = PRICE if i == 0 else f"{4810 + i % 5}"
            lines.append(f"{day},{code},{t},{sess},{price},{100 * (i + 1)},{i * 3 + 21:012d}")
    return gzip.compress(("\n".join(lines) + "\n").encode())


class Resp:
    def __init__(self, body):
        self._b = io.BytesIO(body)

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeBulk:
    def __init__(self):
        self.files = {
            "equities/trades/historical/2025/equities_trades_20251201.csv.gz":
                (trades_csv(), "2026-01-14T02:00:00+00:00"),
            "equities/trades/live/equities_trades_20261118.csv.gz":
                (trades_csv("2026-11-18", per_code=80), "2026-11-18T08:10:00+00:00"),
        }

    def __call__(self, req, timeout=None):
        u = urllib.parse.urlparse(req.full_url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.netloc == "files.example":
            assert req.get_header("X-api-key") is None
            return Resp(self.files[urllib.parse.unquote(u.path.lstrip("/"))][0])
        path = u.path.replace("/v2", "", 1)
        if path == "/bulk/list":
            assert q["endpoint"] == "/equities/trades"
            data = [{"Key": k, "LastModified": lm, "Size": len(b)} for k, (b, lm) in self.files.items()]
            return Resp(json.dumps({"data": data}).encode())
        if path == "/bulk/get":
            return Resp(json.dumps({"url": f"https://files.example/{urllib.parse.quote(q['key'])}?{SIG}"}).encode())
        raise AssertionError(path)


class TestTime(unittest.TestCase):
    def test_time_to_us(self):
        t = P.time_to_us(pa.chunked_array([["09:00:00.067558", "15:30:00", "12:30:01.5"]]))
        self.assertEqual(t.to_pylist(), [dt.time(9, 0, 0, 67558), dt.time(15, 30), dt.time(12, 30, 1, 500000)])
        with self.assertRaises(ValueError):
            P.time_to_us(pa.chunked_array([["9:00:00"]]))


class TestProfile(unittest.TestCase):
    def test_profile_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv.gz")
            with open(path, "wb") as fh:
                fh.write(trades_csv())
            prof = P.profile_file(path)
        self.assertEqual(prof["header"], ["Date", "Code", "Time", "SessionDistinction", "Price",
                                          "TradingVolume", "TransactionId"])
        self.assertEqual(prof["rows"], 100)
        self.assertEqual((prof["days"], prof["codes"]), (1, 2))
        self.assertEqual(prof["SessionDistinctionValues"], ["01", "02"])
        self.assertEqual(prof["keyUnique"]["duplicateGroups"], 0)
        cols = prof["columns"]
        self.assertTrue(cols["TradingVolume"]["integerValued"])
        self.assertEqual(cols["Price"]["maxDecimals"], 1)
        self.assertTrue(cols["TransactionId"]["digitsOnly"])
        self.assertEqual((cols["TransactionId"]["minLen"], cols["TransactionId"]["maxLen"]), (12, 12))
        est = prof["parquetEstimate"]
        self.assertNotIn("typedError", est)
        self.assertGreater(est["typed"], 0)
        self.assertGreater(est["typedTransactionIdInt"], 0)

    def test_listing_by_kind(self):
        entries = [
            {"key": "equities/trades/historical/2025/equities_trades_20251201.csv.gz",
             "lastModified": "2026-01-14T02:00:00+00:00", "size": 10,
             "period": MB.bulk_period("equities_trades_20251201.csv.gz")},
            {"key": "equities/trades/live/equities_trades_20261118.csv.gz",
             "lastModified": "2026-11-18T08:10:00Z", "size": 30,
             "period": MB.bulk_period("equities_trades_20261118.csv.gz")},
        ]
        s = P.summarize_listing(entries)
        self.assertEqual(sorted(s), ["historical", "live"])
        self.assertEqual(s["live"]["madeAtJst"]["median"], "17:10")
        self.assertEqual(s["historical"]["periodKinds"], ["day"])


class TestMain(unittest.TestCase):
    def test_measures_without_leaking_values(self):
        fake = FakeBulk()
        api = MB.Api(KEY, pause=0, sleep=lambda s: None, urlopen=fake)
        with tempfile.TemporaryDirectory() as tmp:
            summary = os.path.join(tmp, "s.json")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = P.main(["--endpoint", "/equities/trades", "--summary-json", summary], api=api)
            with open(summary) as fh:
                rep = json.load(fh)
        log = out.getvalue()
        self.assertEqual(code, 0, log)
        rec = rep["endpoints"]["/equities/trades"]
        self.assertEqual(rec["listed"], 2)
        self.assertEqual(rec["sample"]["rows"], 100)                    # いちばん小さいファイル
        for secret in (SIG, KEY, PRICE):
            self.assertNotIn(secret, log)
            self.assertNotIn(secret, json.dumps(rep))


if __name__ == "__main__":
    unittest.main()
