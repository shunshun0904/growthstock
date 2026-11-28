#!/usr/bin/env python3
"""
research/jsf_hist_compare.py（日証金の過去ファイルを取り直したあと、控えと今を比べる）の単体テスト。

固定すること
  - 重なる行（申込日×銘柄）で値が違う列を数える。欠測どうしは同じ、数の列は誤差 1e-9 まで同じ、
    文字の列は前後の空白と「－」を空とみなす
  - 控えの各銘柄の最終日（取った日の速報のことがある）の違いは分けて数える
  - 控えにしか無い行は「今の窓より前」「今に無い銘柄」「窓の中」に、今にしか無い行は「控えの最終日より後」
    「控えに無い銘柄」「控えの期間の中」に分ける
  - 抜けた5日の銘柄数、文字の列が銘柄の中で日によって変わるか
  - 出力に値を出さない（件数・日付だけ）

  python3 -m unittest tests.test_jsf_hist_compare
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jsf_hist_compare as C  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows, columns=["申込日", "code", "融資残高（株）", "品貸料率（年率換算/％）", "貸借区分", "_fetched_at"])
    df["申込日"] = pd.to_datetime(df["申込日"])
    return df


class TestSameValues(unittest.TestCase):
    def test_numeric_text_and_missing(self):
        a = pd.Series([1.0, np.nan, 3.0, 4.0])
        b = pd.Series([1.0 + 1e-12, np.nan, 3.5, np.nan])
        self.assertEqual(C.same_values(a, b).tolist(), [True, True, False, False])
        t = pd.Series(["貸借", " 貸借 ", "－", None])
        u = pd.Series(["貸借", "貸借", "", ""])
        self.assertEqual(C.same_values(t, u).tolist(), [True, True, True, True])


class TestCompare(unittest.TestCase):
    def setUp(self):
        # 控え: 銘柄A は 2023-09-26〜2026-10-01（最終日は速報）、B は 2023-09-27〜2026-09-30
        self.old = frame([
            ("2023-09-26", "A", 100, np.nan, "貸借融資", "x"),      # 今の窓より前（今には無い）
            ("2024-01-04", "A", 200, 0.5, "貸借融資", "x"),
            ("2026-10-01", "A", 300, np.nan, "貸借融資", "x"),      # 最終日（速報）。今は確報で違う
            ("2023-09-27", "B", 10, np.nan, "貸借", "x"),
            ("2024-01-04", "B", 20, np.nan, "貸借", "x"),           # 今で値が違う（日証金が直した）
            ("2026-09-30", "B", 30, np.nan, "貸借", "x"),
            ("2024-01-04", "Z", 1, np.nan, "貸借", "x"),            # 今に無い銘柄
        ])
        self.new = frame([
            ("2023-11-14", "A", 150, np.nan, "貸借融資", "y"),
            ("2024-01-04", "A", 200, 0.5, "貸借融資", "y"),
            ("2026-10-01", "A", 333, np.nan, "貸借", "y"),          # 確報（最終日の違い）・区分も変わった
            ("2026-10-23", "A", 340, np.nan, "貸借", "y"),          # 抜けた日が埋まった
            ("2026-11-12", "A", 350, np.nan, "貸借", "y"),
            ("2023-11-14", "B", 15, np.nan, "貸借", "y"),
            ("2024-01-04", "B", 21, np.nan, "貸借", "y"),
            ("2026-09-30", "B", 30, np.nan, "貸借", "y"),
            ("2026-10-23", "B", 31, np.nan, "貸借", "y"),
            ("2026-11-12", "N", 5, np.nan, "貸借", "y"),            # 控えに無い銘柄
        ])

    def test_counts(self):
        r = C.compare(self.old, self.new)
        self.assertEqual((r["rows_old"], r["rows_new"]), (7, 10))
        self.assertEqual(r["overlap"], 4)
        self.assertEqual(r["overlap_last_rows"], 2)                  # A の 10/1・B の 9/30
        self.assertEqual(r["only_old"], 3)
        self.assertEqual(r["only_old_before_new_window"], 2)         # A・B の 2023-09
        self.assertEqual(r["only_old_no_new_code"], 1)               # Z
        self.assertEqual(r["only_old_inside"], 0)
        self.assertEqual(r["only_new"], 6)
        self.assertEqual(r["only_new_after_old_end"], 3)             # A の 10/23・11/12、B の 10/23
        self.assertEqual(r["only_new_new_code"], 1)                  # N
        self.assertEqual(r["only_new_inside"], 2)                    # A・B の 2023-11-14（控えの期間の中で、控えに無かった日）
        cols = {c["col"]: c for c in r["columns"]}
        self.assertNotIn("_fetched_at", cols)
        self.assertEqual(cols["融資残高（株）"]["diff"], 1)           # B の 2024-01-04
        self.assertEqual(cols["融資残高（株）"]["diff_last"], 1)      # A の最終日
        self.assertEqual(cols["融資残高（株）"]["codes"], 1)
        self.assertEqual(cols["品貸料率（年率換算/％）"]["diff"], 0)
        self.assertEqual(cols["貸借区分"]["diff"], 0)
        self.assertEqual(cols["貸借区分"]["diff_last"], 1)
        gaps = {g["day"]: g for g in r["gaps"]}
        self.assertEqual((gaps["2026-10-23"]["old"], gaps["2026-10-23"]["new"]), (0, 2))
        nat = {n["col"]: n for n in r["nature"]}
        self.assertEqual(nat["貸借区分"]["old"]["varies"], 0)
        self.assertEqual(nat["貸借区分"]["new"]["varies"], 1)        # A は今では日によって変わる
        self.assertIsNone(nat["市場区分"]["old"])

    def test_report_prints_no_values(self):
        r = C.compare(self.old, self.new)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C.report(r)
        out = buf.getvalue()
        self.assertIn("[compare]", out)
        for v in ("333", "340", "350", "150"):
            self.assertNotIn(v, out.replace("2026", "").replace("2023", ""), v)

    def test_main_reads_files(self):
        d = tempfile.mkdtemp()
        try:
            po, pn = os.path.join(d, "o.parquet"), os.path.join(d, "n.parquet")
            self.old.to_parquet(po, index=False)
            self.new.to_parquet(pn, index=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = C.main(["--old", po, "--new", pn])
            self.assertEqual(rc, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main(["--old", os.path.join(d, "none.parquet"), "--new", pn]), 1)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
