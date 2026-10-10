#!/usr/bin/env python3
"""
research/probe_edinet_api.py の単体テスト。ネットワークには触れない。

この probe は公開リポジトリの Actions で走り、ログも公開になる。**鍵が出力に出ないこと**
（URL のクエリに付けるので、印字する URL から鍵を外す）が最優先の要件。
あとは CSV（UTF-16・タブ区切り）の読み方、最新の書類の選び方、恒等式の判定、遡りの日付。

  python3 tests/test_probe_edinet_api.py
"""
import contextlib
import datetime as dt
import io
import os
import sys
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import probe_edinet_api as P  # noqa: E402

KEY = "TESTONLY-not-a-real-key-0123456789abcdef0123456789"
TODAY = dt.date(2026, 10, 10)


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = fn()
    return ret, buf.getvalue()


class TestKeyNeverPrinted(unittest.TestCase):
    def test_url_carries_key_but_output_does_not(self):
        seen = []

        def transport(url):
            seen.append(url)
            return 200, b'{"metadata": {"resultset": {"count": 0}}, "results": []}', {"Content-Type": "application/json"}

        p = P.Probe(KEY, budget=5, transport=transport)
        (st, body), out = capture(lambda: p.get("/documents.json", {"date": "2026-06-26", "type": "2"}))
        self.assertEqual(st, 200)
        self.assertEqual(body["results"], [])
        q = urllib.parse.parse_qs(urllib.parse.urlparse(seen[0]).query)
        self.assertEqual(q["Subscription-Key"], [KEY])              # 鍵はクエリで送る
        self.assertNotIn(KEY, out)                                   # 出力には出ない
        self.assertIn("GET /documents.json?date=2026-06-26&type=2 -> HTTP 200", out)

    def test_say_redacts_even_if_key_leaks_into_text(self):
        p = P.Probe(KEY, budget=1, transport=lambda u: (200, b"{}", {}))
        _, out = capture(lambda: p.say(f"body={KEY}"))
        self.assertNotIn(KEY, out)
        self.assertIn("[REDACTED]", out)

    def test_budget_and_no_key(self):
        p = P.Probe(KEY, budget=1, transport=lambda u: (200, b"{}", {}))
        _, _ = capture(lambda: p.get("/documents.json"))
        (st, why), _ = capture(lambda: p.get("/documents.json"))
        self.assertEqual((st, why), (None, "budget"))
        self.assertEqual(P.Probe(None, budget=5).get("/x"), (None, "no-key"))


class TestCsv(unittest.TestCase):
    def test_decode_utf16_and_utf8(self):
        enc, text = P.decode_csv(b"\xff\xfe" + "要素ID\t値\n".encode("utf-16-le"))
        self.assertEqual(enc, "utf-16")
        self.assertEqual(text, "要素ID\t値\n")
        enc, text = P.decode_csv("要素ID,値\n".encode("utf-8-sig"))
        self.assertEqual(enc, "utf-8-sig")
        self.assertEqual(P.decode_csv("a\tb".encode("utf-8"))[0], "utf-8")

    def test_parse_rows_tab_and_comma(self):
        h, rows, d = P.parse_rows("要素ID\tコンテキストID\t値\nx:A\tC\t1\n\n")
        self.assertEqual((h, d, len(rows)), (["要素ID", "コンテキストID", "値"], "tab", 1))
        self.assertEqual(rows[0]["値"], "1")
        h, rows, d = P.parse_rows("a,b\n1,2\n")
        self.assertEqual((d, rows[0]["b"]), ("comma", "2"))

    def test_numbers(self):
        self.assertEqual(P.to_number("1,234"), 1234.0)
        self.assertEqual(P.to_number("△12"), -12.0)
        self.assertEqual(P.to_number("-5"), -5.0)
        self.assertIsNone(P.to_number(""))
        self.assertIsNone(P.to_number("－"))
        self.assertIsNone(P.to_number("abc"))

    def test_identities(self):
        vals = {"NetSales": 1000.0, "CostOfSales": 600.0, "GrossProfit": 400.0,
                "SellingGeneralAndAdministrativeExpenses": 250.0, "OperatingIncome": 150.0}
        res = dict(P.check_identities(vals, P.IDENTITIES_JGAAP))
        self.assertTrue(res["GrossProfit"])
        self.assertTrue(res["OperatingIncome"])
        self.assertIsNone(res["OrdinaryIncome"])                     # 項目が足りない
        vals["GrossProfit"] = 401.0                                   # 円単位の値で 1 のずれは成立（許容 2）
        self.assertTrue(dict(P.check_identities(vals, P.IDENTITIES_JGAAP))["GrossProfit"])
        vals["GrossProfit"] = 403.0                                   # それ以上は不成立
        self.assertFalse(dict(P.check_identities(vals, P.IDENTITIES_JGAAP))["GrossProfit"])
        # 百万円に丸めた開示: 3項目の足し合わせは 1.5百万円までずれうる
        m = {"OperatingIncome": 150e6, "NonOperatingIncome": 20e6, "NonOperatingExpenses": 10e6, "OrdinaryIncome": 161e6}
        self.assertTrue(dict(P.check_identities(m, P.IDENTITIES_JGAAP))["OrdinaryIncome"])       # ずれ 1百万
        m["OrdinaryIncome"] = 163e6
        self.assertFalse(dict(P.check_identities(m, P.IDENTITIES_JGAAP))["OrdinaryIncome"])      # ずれ 3百万
        self.assertEqual(P.rounding_unit([150e6, 20e6]), 1e6)
        self.assertEqual(P.rounding_unit([150e3, 21e3]), 1e3)
        self.assertEqual(P.rounding_unit([150e6, 21e3 + 5]), 1.0)

    def test_fake_zip_parses_and_identities_hold(self):
        for ifrs in (False, True):
            p = P.Probe(KEY, budget=0, transport=lambda u: (200, b"", {}))
            info, out = capture(lambda: P.describe_zip(p, P.fake_zip(ifrs)))
            self.assertTrue(info["zip"])
            self.assertEqual(len(info["csv"]), 1)                     # 本体（jpcrp*）だけ読む
            c = next(iter(info["csv"].values()))
            self.assertEqual(c["encoding"], "utf-16")
            self.assertEqual(c["delim"], "tab")
            self.assertEqual(c["dei"]["AccountingStandardsDEI"], "IFRS" if ifrs else "Japan GAAP")
            self.assertIn("CurrentYearDuration", c["contexts"])
            ok = c["identities"]["CurrentYearDuration"]
            self.assertTrue(all(v for v in ok.values() if v is not None))
            self.assertTrue(any(v for v in ok.values()))
            self.assertNotIn("1000", out)                             # 値は印字しない
            self.assertIn("成立", out)

    def test_bad_zip(self):
        p = P.Probe(KEY, budget=0, transport=lambda u: (200, b"", {}))
        info, out = capture(lambda: P.describe_zip(p, b"not a zip"))
        self.assertFalse(info["zip"])
        self.assertIn("ZIP として読めない", out)


class TestListing(unittest.TestCase):
    def row(self, **kw):
        base = {"docID": "S100AAAA", "secCode": "79740", "docTypeCode": "120", "submitDateTime": "2026-06-26 15:00",
                "withdrawalStatus": "0", "csvFlag": "1"}
        base.update(kw)
        return base

    def test_pick_latest_rules(self):
        rows = [self.row(docID="A", submitDateTime="2026-06-26 10:00"),
                self.row(docID="B", submitDateTime="2026-06-26 16:00"),
                self.row(docID="C", submitDateTime="2026-06-27 09:00", withdrawalStatus="1"),   # 取り下げ
                self.row(docID="D", submitDateTime="2026-06-27 09:00", csvFlag="0"),            # CSV 無し
                self.row(docID="E", submitDateTime="2026-06-27 09:00", docTypeCode="130"),      # 訂正は対象外
                self.row(docID="F", submitDateTime="2026-06-27 09:00", secCode="72030")]        # 別の会社
        self.assertEqual(P.pick_latest(rows, "79740")["docID"], "B")
        self.assertEqual(P.pick_latest(rows, "72030")["docID"], "F")
        self.assertIsNone(P.pick_latest(rows, "99990"))
        self.assertEqual(P.pick_latest(rows, "79740", types=("130",))["docID"], "E")

    def test_weekdays_back(self):
        days = list(P.weekdays_back(TODAY, 6))                        # 10/10 は土曜
        self.assertEqual([d.isoformat() for d in days],
                         ["2026-10-09", "2026-10-08", "2026-10-07", "2026-10-06", "2026-10-05", "2026-10-02"])

    def test_type_counts_and_results_of(self):
        self.assertEqual(P.type_counts([{"docTypeCode": "120"}, {"docTypeCode": "120"}, {}]), {"120": 2, "?": 1})
        self.assertEqual(P.results_of({"results": [{"a": 1}, 2]}), [{"a": 1}])
        self.assertEqual(P.results_of(b"zip"), [])


class TestRunFake(unittest.TestCase):
    def test_end_to_end_without_leaking(self):
        rc, out = capture(lambda: P.main(["--fake", "--today", "2026-10-10", "--max-requests", "40"]))
        self.assertEqual(rc, 0)
        self.assertNotIn("TESTONLY-fake-key", out)
        self.assertIn("docTypeCode の内訳", out)
        self.assertIn("79740: 2026-10-07 に docTypeCode 120", out)
        self.assertIn("72030: 2026-09-30 に docTypeCode 160", out)
        self.assertIn("恒等式（CurrentYearDuration", out)
        self.assertIn("テスト株式会社", out) if False else self.assertNotIn("テスト株式会社", out)   # 会社名は出さない
        self.assertIn("[done] リクエスト", out)

    def test_rejects_bad_sec_code(self):
        rc, out = capture(lambda: P.main(["--fake", "--sec-codes", "7974"]))
        self.assertEqual(rc, 2)

    def test_stops_when_budget_is_tiny(self):
        rc, out = capture(lambda: P.main(["--fake", "--today", "2026-10-10", "--max-requests", "3", "--busy-date", ""]))
        self.assertIn("予算が残り少ない", out)
        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
