#!/usr/bin/env python3
"""
research/edinet_filings.py の単体テスト。ネットワークには触れない（probe_edinet_api の偽の応答を使う）。

- 索引: まだ無い平日だけを新しい順に取り、最後の日は取り直す。予算が切れたら途中でやめる
- 最新の書類: 証券コードの 120 / 160 のうち提出日時が最新。取り下げ・CSV 無し・訂正は除く
- 書類の読み方: 当期のコンテキスト（連結 / 個別）、DEI、損益の項目、前期、恒等式、売上の項目名での拾い方
- 出力: 画面の銘柄だけ、控えにあるものは取りに行かない、鍵が出力に出ない

  python3 tests/test_edinet_filings.py
"""
import contextlib
import datetime as dt
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import edinet_filings as F  # noqa: E402
import probe_edinet_api as API  # noqa: E402

TODAY = dt.date(2026, 10, 10)
KEY = "TESTONLY-fake-key"


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = fn()
    return ret, buf.getvalue()


class TestIndex(unittest.TestCase):
    def test_dates_to_fetch(self):
        idx = {"dates": {}}
        days = F.dates_to_fetch(idx, TODAY, backfill_days=10)
        self.assertEqual([d.isoformat() for d in days],
                         ["2026-10-09", "2026-10-08", "2026-10-07", "2026-10-06", "2026-10-05",
                          "2026-10-02", "2026-10-01", "2026-09-30"])              # 平日だけ・新しい順
        idx = {"dates": {"2026-10-09": [], "2026-10-07": []}}
        days = F.dates_to_fetch(idx, TODAY, backfill_days=10)
        self.assertEqual([d.isoformat() for d in days][:3], ["2026-10-09", "2026-10-08", "2026-10-06"])  # 最後の日は取り直す

    def test_update_index_and_budget(self):
        p = API.Probe(KEY, budget=4, transport=API.fake_transport(TODAY))
        idx = {"dates": {}}
        _, out = capture(lambda: F.update_index(p, idx, TODAY, backfill_days=20, reserve=1))
        self.assertEqual(p.used, 3)                                          # 予算 4 − 予備 1
        self.assertEqual(sorted(idx["dates"]), ["2026-10-07", "2026-10-08", "2026-10-09"])
        self.assertEqual(len(idx["dates"]["2026-10-07"]), 2)                 # 120 と 160（secCode あり）
        self.assertIn("予算が残り", out)
        self.assertNotIn(KEY, out)

    def test_latest_doc(self):
        idx = {"dates": {
            "2026-06-25": [{"docID": "A", "secCode": "79740", "docTypeCode": "120", "submitDateTime": "2026-06-25 15:00",
                            "csvFlag": "1", "withdrawalStatus": "0"}],
            "2026-09-30": [{"docID": "B", "secCode": "79740", "docTypeCode": "130", "submitDateTime": "2026-09-30 15:00",
                            "csvFlag": "1", "withdrawalStatus": "0"},                                  # 訂正は使わない
                           {"docID": "C", "secCode": "79740", "docTypeCode": "160", "submitDateTime": "2026-09-30 16:00",
                            "csvFlag": "0", "withdrawalStatus": "0"},                                  # CSV 無し
                           {"docID": "D", "secCode": "72030", "docTypeCode": "160", "submitDateTime": "2026-09-30 16:00",
                            "csvFlag": "1", "withdrawalStatus": "0"}],
        }}
        self.assertEqual(F.latest_doc(idx, "79740")["docID"], "A")
        self.assertEqual(F.latest_doc(idx, "72030")["docID"], "D")
        self.assertIsNone(F.latest_doc(idx, "99990"))


class TestParse(unittest.TestCase):
    def test_jgaap(self):
        meta = {"docID": "S100C001", "docTypeCode": "120", "secCode": "79740", "edinetCode": "E00000",
                "submitDateTime": "2026-06-25 15:00", "periodStart": "2025-04-01", "periodEnd": "2026-03-31"}
        d = F.parse_statement(API.fake_zip(ifrs=False), meta)
        self.assertEqual(d["standard"], "Japan GAAP")
        self.assertEqual(d["context"], "CurrentYearDuration")
        self.assertTrue(d["consolidated"])
        self.assertEqual(d["items"]["NetSales"], 1000)
        self.assertEqual(d["items"]["CostOfSales"], 600)
        self.assertEqual(d["priorItems"]["NetSales"], 999)
        self.assertEqual(d["labels"]["NetSales"], "売上高")
        self.assertEqual(d["submitDate"], "2026-06-25")
        self.assertEqual(d["periodType"], "FY")
        self.assertTrue(all(d["checks"].values()))
        self.assertEqual(d["docType"], "120")
        self.assertIsInstance(d["items"]["NetSales"], int)

    def test_ifrs(self):
        d = F.parse_statement(API.fake_zip(ifrs=True), {"docID": "X", "docTypeCode": "160"})
        self.assertEqual(d["standard"], "IFRS")
        self.assertEqual(d["items"]["RevenueIFRS"], 1000)
        self.assertEqual(d["items"]["IncomeTaxExpenseIFRS"], 40)
        self.assertTrue(d["checks"]["ProfitLossIFRS"])

    def _zip_with(self, text: str) -> bytes:
        buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("XBRL_TO_CSV/jpcrp030000-asr-001_E1-000_2026-03-31_01_2026-06-25.csv",
                        b"\xff\xfe" + text.encode("utf-16-le"))
        return buf.getvalue()

    def test_non_consolidated_and_revenue_by_label(self):
        h = "\t".join(["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"])
        rows = [
            ["jpdei_cor:WhetherConsolidatedFinancialStatementsArePreparedDEI", "連結の有無", "FilingDateInstant", "", "その他", "時点", "", "", "false"],
            ["jpdei_cor:AccountingStandardsDEI", "会計基準", "FilingDateInstant", "", "その他", "時点", "", "", "IFRS"],
            ["jpcrp030000-asr_E1-000:OperatingRevenuesIFRS", "営業収益", "CurrentYearDuration_NonConsolidatedMember", "当期", "個別", "期間", "JPY", "円", "5,000"],
            ["jpigp_cor:CostOfSalesIFRS", "売上原価", "CurrentYearDuration_NonConsolidatedMember", "当期", "個別", "期間", "JPY", "円", "3000"],
            ["jpigp_cor:ProfitLossIFRS", "当期利益", "CurrentYearDuration_NonConsolidatedMember", "当期", "個別", "期間", "JPY", "円", "△100"],
            ["jpigp_cor:ProfitLossIFRS", "当期利益", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "9"],
        ]
        text = h + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
        d = F.parse_statement(self._zip_with(text), {"docID": "Y", "docTypeCode": "120"})
        self.assertFalse(d["consolidated"])
        self.assertEqual(d["context"], "CurrentYearDuration_NonConsolidatedMember")
        self.assertEqual(d["items"]["Revenue"], 5000)                       # 項目名「営業収益」で拾う
        self.assertEqual(d["labels"]["Revenue"], "営業収益")
        self.assertEqual(d["items"]["ProfitLossIFRS"], -100)                # △ は負
        self.assertNotIn("NetSales", d["items"])

    def test_unsupported(self):
        d = F.parse_statement(b"not zip", {"docID": "Z", "docTypeCode": "120"})
        self.assertEqual((d["unsupported"], d["docID"], d["items"]), ("no-csv", "Z", {}))
        h = "要素ID\t項目名\tコンテキストID\t値\njpdei_cor:AccountingStandardsDEI\t会計基準\tFilingDateInstant\tUS GAAP\n"
        d = F.parse_statement(self._zip_with(h), {"docID": "Z"})
        self.assertEqual((d["unsupported"], d["standard"]), ("no-context", "US GAAP"))   # 当期のコンテキストが無い
        h2 = h + "jpcrp030000-asr_E1-000:SalesUSGAAP\t売上高\tCurrentYearDuration\t100\n"
        d = F.parse_statement(self._zip_with(h2), {"docID": "Z"})
        self.assertEqual(d["unsupported"], "no-items")                     # 標準の要素が無い（売上の項目名でも拾わない: 段が作れない）


class TestRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pred = os.path.join(self.dir, "predictions.json")
        self.stocks = os.path.join(self.dir, "stocks.json")
        self.out = os.path.join(self.dir, "filings.json")
        with open(self.pred, "w", encoding="utf-8") as fh:
            json.dump({"candidates": [{"jqCode": "79740", "name": "A"}, {"jqCode": "79740", "name": "A"},
                                      {"jqCode": "99990", "name": "Z"}]}, fh)
        with open(self.stocks, "w", encoding="utf-8") as fh:
            json.dump({"stocks": [{"jqCode": "72030", "name": "B"}]}, fh)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _args(self, extra=()):
        return ["--data-dir", self.dir, "--predictions", self.pred, "--stocks", self.stocks, "--out", self.out,
                "--today", "2026-10-10", "--fake", "--backfill-days", "20", *extra]

    def test_targets(self):
        t = F.targets_from_public(self.pred, self.stocks, ["11110"])
        self.assertEqual(t, {"79740": "A", "99990": "Z", "72030": "B", "11110": ""})

    def test_end_to_end_then_offline(self):
        rc, out = capture(lambda: F.main(self._args()))
        self.assertEqual(rc, 0)
        self.assertNotIn(KEY, out)
        self.assertNotIn("テスト株式会社", out)                                 # 会社名は出さない
        with open(self.out, encoding="utf-8") as fh:
            pub = json.load(fh)
        self.assertEqual(sorted(pub["docs"]), ["72030", "79740"])             # 99990 は索引に無い
        self.assertEqual(pub["docs"]["79740"]["name"], "A")
        self.assertEqual(pub["docs"]["79740"]["docType"], "120")
        self.assertEqual(pub["docs"]["72030"]["docType"], "160")
        self.assertEqual(pub["docs"]["72030"]["standard"], "IFRS")
        self.assertEqual(pub["source"]["license"], "公共データ利用規約（PDL1.0）")
        idx = F.load_json(os.path.join(self.dir, F.INDEX), {})
        self.assertGreaterEqual(len(idx["dates"]), 14)
        cache = F.load_json(os.path.join(self.dir, F.CACHE), {})
        self.assertEqual(sorted(cache), ["S100C001", "S100C003"])
        # 2回目: 索引は最後の日だけ取り直し、書類は控えから。リクエストは 1
        rc, out = capture(lambda: F.main(self._args()))
        self.assertIn("控えにあった 2 / 新しく取った 0 / 索引に無い 1 / 対象外 0", out)
        self.assertIn("リクエスト 1 /", out)
        # オフライン: 取りに行かずに出力だけ作り直す
        os.remove(self.out)
        rc, out = capture(lambda: F.main(self._args(["--offline"])))
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(self.out))
        self.assertNotIn("[fetch]", out)

    def test_budget_leaves_room_for_documents(self):
        rc, out = capture(lambda: F.main(self._args(["--max-requests", "5", "--reserve", "2"])))
        self.assertEqual(rc, 0)
        self.assertIn("索引の更新をここでやめる", out)
        pub = F.load_json(self.out, {})
        self.assertIn("79740", pub["docs"])                                   # 新しい日から取るので 10/07 の書類は入る


if __name__ == "__main__":
    unittest.main()
