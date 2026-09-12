#!/usr/bin/env python3
"""
research/probe_minute_bars.py の、API を叩かない部分の単体テスト。
ネットワークアクセスは行わない（合成データのみ）。

  python3 tests/test_probe_minute.py

プローブ本体は「叩いてみないと分からないこと」を測る道具なので、
テストできるのは解釈と出力の部分だけである。だがそこが壊れていると、
実測が取れても読めないものが出てくる。
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "research"))

os.environ.setdefault("JQUANTS_API", "dummy-for-import")  # import 時に認証はしない
import probe_minute_bars as P  # noqa: E402


def bar(stamp, **kw):
    row = {"Code": "72030", "DateTime": f"2026-09-11T{stamp}:00+09:00",
           "O": 3000, "H": 3010, "L": 2995, "C": 3005, "Vo": 1200}
    row.update(kw)
    return row


class TestColumnDescription(unittest.TestCase):
    """列名を仮定せずに読めているか。時刻の列名すら分かっていない前提。"""

    def test_detects_time_column_by_value(self):
        rows = [bar("09:00"), bar("09:01")]
        self.assertEqual(P.time_column(rows), "DateTime")

    def test_detects_time_column_under_any_name(self):
        rows = [{"whatever": "09:00:00", "x": 1}]
        self.assertEqual(P.time_column(rows), "whatever")

    def test_reports_kind_and_null_share(self):
        rows = [bar("09:00", Vo=None), bar("09:01")]
        by = {c["column"]: c for c in P.describe_columns(rows)}
        self.assertEqual(by["O"]["kind"], "数値")
        self.assertEqual(by["Code"]["kind"], "文字列")
        self.assertEqual(by["DateTime"]["kind"], "時刻らしい")
        self.assertEqual(by["Vo"]["nullShare"], 50.0)

    def test_keeps_columns_absent_from_first_row(self):
        # 途中の行にしか出てこない列を落とすと、実測の意味が無くなる
        rows = [{"a": 1}, {"a": 2, "b": "x"}]
        self.assertEqual([c["column"] for c in P.describe_columns(rows)], ["a", "b"])

    def test_empty_rows(self):
        self.assertEqual(P.describe_columns([]), [])
        self.assertIsNone(P.time_column([]))


class TestSessionShape(unittest.TestCase):
    """
    立会の形を数え違えないか。押し目指値の約定判定は
    「どの時間帯に行があるか」を前提にするので、ここがずれると設計が崩れる。
    """

    def test_counts_time_bands(self):
        rows = ([bar("08:55")]                      # 09:00 より前
                + [bar("09:00"), bar("10:30")]
                + [bar("11:35"), bar("12:00")]      # 11:30〜12:30
                + [bar("13:00"), bar("15:29")]
                + [bar("15:45")])                   # 15:30 より後
        s = P.session_shape(rows)
        self.assertEqual(s["rows"], 8)
        self.assertEqual(s["rowsBefore0900"], 1)
        self.assertEqual(s["rowsInLunch"], 2)
        self.assertEqual(s["rowsAfter1530"], 1)
        self.assertEqual(s["distinctMinutes"], 8)

    def test_boundaries_are_not_counted_as_outside(self):
        # 11:25 は昼休みではない。15:30 ちょうどは「15:30 より後」ではない
        s = P.session_shape([bar("11:25"), bar("15:30"), bar("09:00")])
        self.assertEqual(s["rowsInLunch"], 0)
        self.assertEqual(s["rowsAfter1530"], 0)
        self.assertEqual(s["rowsBefore0900"], 0)

    def test_first_and_last(self):
        s = P.session_shape([bar("12:00"), bar("09:00"), bar("15:00")])
        self.assertTrue(s["first"].endswith("09:00:00+09:00"))
        self.assertTrue(s["last"].endswith("15:00:00+09:00"))

    def test_no_time_column(self):
        s = P.session_shape([{"O": 1}])
        self.assertIsNone(s["timeColumn"])
        self.assertEqual(s["rows"], 1)


class TestRowsOf(unittest.TestCase):
    def test_v2_data_key(self):
        self.assertEqual(P.rows_of({"data": [{"a": 1}]}), [{"a": 1}])

    def test_missing_or_malformed(self):
        self.assertEqual(P.rows_of({}), [])
        self.assertEqual(P.rows_of({"data": None}), [])
        self.assertEqual(P.rows_of("エラー文"), [])


class TestReport(unittest.TestCase):
    """未契約でも読める報告が出るか。未契約は失敗ではなく測定結果である。"""

    def setUp(self):
        self._orig = P.OUT_MD
        self._tmp = tempfile.mkdtemp()
        P.OUT_MD = os.path.join(self._tmp, "MINUTE_DATA.md")

    def tearDown(self):
        P.OUT_MD = self._orig

    def read(self):
        with open(P.OUT_MD, encoding="utf-8") as fh:
            return fh.read()

    def test_unavailable_reports_error_and_stops(self):
        P.write_md({
            "probedAt": "2026-09-12T00:00:00+00:00",
            "keyWorks": {"ok": True, "rows": 1, "seconds": 0.4},
            "availability": {"ok": False, "date": "2026-09-11",
                             "detail": 'HTTP 403 ... : {"message": "Forbidden"}'},
        })
        md = self.read()
        self.assertIn("引けない", md)
        self.assertIn("HTTP 403", md)
        # 日足が通っているなら、キー失効と取り違えない文言が出る
        self.assertIn("キーの失効ではない", md)
        # 後段の見出しは出さない（測っていないものを見出しだけ出さない）
        self.assertNotIn("## 5.", md)

    def test_available_renders_all_sections(self):
        rows = [bar(f"{h:02d}:{m:02d}") for h in (9, 10) for m in (0, 30)]
        P.write_md({
            "probedAt": "2026-09-12T00:00:00+00:00",
            "keyWorks": {"ok": True, "rows": 1, "seconds": 0.4},
            "availability": {"ok": True, "date": "2026-09-11", "rows": 300,
                             "seconds": 1.2, "sampleRow": rows[0]},
            "paramShapes": [{"params": {}, "ok": False, "detail": "HTTP 400",
                             "seconds": 0.3}],
            "session": {"date": "2026-09-11", "byCode": {
                "72030": dict(P.session_shape(rows), note="板が厚い", ok=True,
                              pages=1, seconds=1.1,
                              columns=P.describe_columns(rows)),
                "90820": {"note": "板が薄い", "ok": False, "detail": "HTTP 500"},
            }},
            "boundary": {"scan": {"2024-09": {"ok": True, "rows": 10}},
                         "earliestMonth": "2024-09"},
            "bulkByDate": {"date": "2026-09-11", "maxPagesProbed": 3,
                           "pages": [{"page": 1, "ok": True, "rows": 5,
                                      "seconds": 8.3, "distinctCodes": 2,
                                      "hasNextPage": True}]},
            "latency": {"stat": {"n": 8, "min": 0.9, "max": 2.1,
                                 "median": 1.2, "mean": 1.3}},
        })
        md = self.read()
        for heading in ("## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
            self.assertIn(heading, md)
        self.assertIn("2024-09", md)
        self.assertIn("1.2秒", md)
        # 取れなかった銘柄も行として残す（黙って消さない）
        self.assertIn("90820", md)

    def test_report_is_written_even_without_optional_stages(self):
        P.write_md({
            "probedAt": "2026-09-12T00:00:00+00:00",
            "keyWorks": {"ok": False, "detail": "HTTP 401"},
        })
        self.assertIn("NG", self.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
