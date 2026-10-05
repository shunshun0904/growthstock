#!/usr/bin/env python3
"""research/exp/probe_rev_mismatch.py のテスト（おもちゃの fins で、欠測の道 (b) を再現）。"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import probe_rev_mismatch as P  # noqa: E402


def fins(with_prev=True):
    rows = [
        # 決算短信（期初の予想）→ 修正。同じ事業年度
        {"Code": "1001", "DiscDate": "2026-05-10", "DocType": "FYFinancialStatements_Consolidated_JP",
         "CurFYSt": "2026-04-01", "FOP": 100.0, "DiscNo": "a"},
        {"Code": "1001", "DiscDate": "2026-08-20", "DocType": "EarnForecastRevision",
         "CurFYSt": "2026-04-01", "FOP": 120.0, "DiscNo": "b"},
        {"Code": "2002", "DiscDate": "2026-08-01", "DocType": "EarnForecastRevision",
         "CurFYSt": "2026-04-01", "FOP": 50.0, "DiscNo": "c"},
    ]
    if not with_prev:
        rows = [r for r in rows if r["DiscNo"] != "a"]
    return pd.DataFrame(rows)


class Probe(unittest.TestCase):
    def test_prev_row_missing_makes_rev_pct_nan(self):
        live = pd.DataFrame({"Code": ["1001", "2002"], "Date": pd.to_datetime(["2026-09-30", "2026-09-30"]),
                             "rev_pct": [20.0, np.nan], "days_since_rev": [41, 60],
                             "_saved_at": ["2026-09-30T12:00:00+00:00"] * 2})
        buf = io.StringIO()
        with redirect_stdout(buf):
            n_ok = P.diagnose(live, fins(True), fins(True), {"2026-08-20"})
            n_bad = P.diagnose(live, fins(False), fins(True), {"2026-08-20"})
        out = buf.getvalue()
        self.assertEqual(n_ok, 0)
        self.assertEqual(n_bad, 1)                      # 前の予想の行が消えると欠測になる
        self.assertIn("予測時は値あり・今は欠測 1", out)
        self.assertIn("写しだけ 1件", out)              # 消えた行が写しにはある
        self.assertIn("写しの fins で作り直すと rev_pct は あり", out)
        self.assertIn("2026-08-20:済", out)
        self.assertNotIn("1001", out.split("[compare]")[-1])   # 銘柄コードは出さない
        self.assertNotIn("120", out)                             # 予想の値は出さない

    def test_listing_marks_latest_revision(self):
        f = P.rev_pct_by_row(fins(True))
        lst = P.listing(f, "1001", pd.Timestamp("2026-09-30"))
        self.assertEqual(len(lst), 2)
        mark = P.latest_rev_before(lst, pd.Timestamp("2026-09-30"))
        self.assertEqual(str(mark.DiscDate.date()), "2026-08-20")
        self.assertTrue(bool(lst.loc[1, "rev_ok"]))
        self.assertFalse(bool(lst.loc[0, "rev_ok"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
