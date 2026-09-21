#!/usr/bin/env python3
"""
research/probe_edinetdb.py の単体テスト。

この probe は公開リポジトリの Actions で走り、ログも公開になる。
**鍵とメールアドレスが出力に出ないこと**が最優先の要件で、ここで固定する。
レスポンスの形を読む小道具（項目名を決め打ちしない）も合わせて見る。
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import probe_edinetdb as P  # noqa: E402

KEY = "TESTONLY-not-a-real-key-0123456789abcdef0123456789"


class TestRedactor(unittest.TestCase):
    def test_key_never_survives(self):
        r = P.Redactor(KEY)
        self.assertNotIn(KEY, r(f"Bearer {KEY} なにか"))
        self.assertIn("[REDACTED]", r(f"x={KEY}"))
        # 鍵が複数回出ても全部消える
        self.assertEqual(r(f"{KEY} {KEY}").count("[REDACTED]"), 2)

    def test_email_is_masked(self):
        r = P.Redactor(KEY)
        out = r('{"email": "nakamura.shun@example.co.jp"}')
        self.assertNotIn("nakamura.shun@example.co.jp", out)
        self.assertIn("…@…", out)

    def test_non_string_input(self):
        r = P.Redactor(KEY)
        self.assertEqual(r(123), "123")
        self.assertEqual(r(None), "None")

    def test_no_key_does_not_crash(self):
        self.assertEqual(P.Redactor(None)("abc"), "abc")


class TestDescribeKey(unittest.TestCase):
    def test_never_contains_the_value(self):
        for k in (KEY, "0123abcd" * 4, "a1b2c3d4-e5f6-7890-abcd-ef1234567890"):
            s = P.describe_key(k)
            self.assertNotIn(k, s)
            self.assertIn(str(len(k)), s)
        self.assertEqual(P.describe_key(None), "未設定")
        self.assertEqual(P.describe_key("  "), "未設定")


class TestShapeHelpers(unittest.TestCase):
    def test_as_rows_accepts_the_common_shapes(self):
        rows = [{"a": 1}, {"a": 2}]
        self.assertEqual(P.as_rows({"data": rows}), rows)
        self.assertEqual(P.as_rows({"items": rows}), rows)
        self.assertEqual(P.as_rows(rows), rows)
        self.assertEqual(P.as_rows({"data": {"a": 1}}), [{"a": 1}])
        self.assertEqual(P.as_rows({"a": 1}), [{"a": 1}])
        self.assertEqual(P.as_rows("not json"), [])
        self.assertEqual(P.as_rows([1, "x", {"a": 1}]), [{"a": 1}])

    def test_find_codes_by_value_shape_not_by_name(self):
        """
        項目名を決め打ちしない。E+5桁 なら EDINET コード。
        4〜5桁の数字は、項目名に code/ticker/sec を含むときだけ証券コードと見る
        （fiscal_year=2024 を証券コードと誤認しない）。
        """
        row = {"edinet_code": "E02367", "fiscal_year": "2024",
               "sec_code": "79740", "name": "任天堂"}
        edinet, sec, where = P.find_codes(row)
        self.assertEqual(edinet, "E02367")
        self.assertEqual(sec, "79740")
        self.assertTrue(any(w.startswith("edinet_code=") for w in where))

    def test_find_codes_when_absent(self):
        edinet, sec, where = P.find_codes({"fiscal_year": "2024", "name": "x"})
        self.assertIsNone(edinet)
        self.assertIsNone(sec)
        self.assertEqual(where, [])


class TestDescribeRows(unittest.TestCase):
    def _capture(self, fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = fn()
        return ret, buf.getvalue()

    def test_flags_date_like_and_redacts(self):
        p = P.Probe(KEY, budget=0)
        rows = [
            {"fiscal_year": 2024, "revenue": 100.5, "filed_date": "2024-06-25",
             "contact": "someone@example.com", "token": KEY, "nested": {"a": 1}},
            {"fiscal_year": 2023, "revenue": None, "filed_date": "2023-06-26",
             "contact": None, "token": KEY, "nested": [1, 2]},
        ]
        date_like, out = self._capture(lambda: p.describe_rows(rows, "財務"))
        self.assertIn("fiscal_year", date_like)
        self.assertIn("filed_date", date_like)
        self.assertNotIn("revenue", date_like)
        self.assertIn("2行 / 6項目", out)
        self.assertIn("1/2", out)                 # revenue の充足
        self.assertIn("<dict len=1>", out)
        self.assertNotIn(KEY, out)                # 鍵は絶対に出ない
        self.assertNotIn("someone@example.com", out)

    def test_budget_is_enforced_before_any_request(self):
        """予算 0 なら1本も送らない（ネットワークにも触れない）。"""
        p = P.Probe(KEY, budget=0)
        (st, why), out = self._capture(lambda: p.get("/status", auth=False))
        self.assertIsNone(st)
        self.assertEqual(why, "budget")
        self.assertEqual(p.used, 0)
        self.assertIn("予算", out)

    def test_auth_without_key_does_not_send(self):
        p = P.Probe(None, budget=5)
        st, why = p.get("/companies")
        self.assertIsNone(st)
        self.assertEqual(why, "no-key")
        self.assertEqual(p.used, 0)


if __name__ == "__main__":
    unittest.main()
