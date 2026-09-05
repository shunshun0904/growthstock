#!/usr/bin/env python3
"""
research/data_store.py の単体テスト。

差分取得の肝は「その日を取得しに行ったか」を記録することで、
行数で判定してはいけない（財務はその日の開示が0件でも取得済み）。
ここを間違えると毎回全期間を叩き直すことになるので、固定しておく。

  python3 tests/test_data_store.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import data_store  # noqa: E402


def days(*iso):
    return [dt.date.fromisoformat(d) for d in iso]


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_manifest_when_missing(self):
        m = data_store.load_manifest(self.dir)
        for k in data_store.KINDS:
            self.assertEqual(m[k]["fetched_days"], [])

    def test_roundtrip(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-01-05"))
        data_store.save_manifest(self.dir, m)

        m2 = data_store.load_manifest(self.dir)
        self.assertEqual(m2["bars"]["fetched_days"], ["2024-01-04", "2024-01-05"])

    def test_missing_days_excludes_fetched(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04"))
        todo = data_store.missing_days(m, "bars", days("2024-01-04", "2024-01-05", "2024-01-09"))
        self.assertEqual([d.isoformat() for d in todo], ["2024-01-05", "2024-01-09"])

    def test_zero_row_day_is_still_marked_fetched(self):
        """
        財務のようにその日の開示が0件でも「取得済み」。
        行数で判定すると毎回叩き直すことになる。
        """
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "fins", days("2024-01-04"))   # 0件だった日
        self.assertEqual(data_store.missing_days(m, "fins", days("2024-01-04")), [])

    def test_mark_is_idempotent(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04"))
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-01-05"))
        self.assertEqual(m["bars"]["fetched_days"], ["2024-01-04", "2024-01-05"])

    def test_summary_reports_range(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-03-01"))
        data_store.save_manifest(self.dir, m)
        out = data_store.summarize(data_store.load_manifest(self.dir))
        self.assertIn("2024-01-04", out)
        self.assertIn("2024-03-01", out)
        self.assertIn("(未取得)", out)   # 他の種別は空


class TestMergeIntoYears(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["Date", "Code", "C"])

    def test_splits_by_year(self):
        df = self._frame([("2023-12-29", "00010", 100), ("2024-01-04", "00010", 110)])
        written = data_store.merge_into_years(self.dir, "bars", df)
        self.assertEqual(len(written), 2)
        self.assertTrue(os.path.exists(data_store.year_path(self.dir, "bars", 2023)))
        self.assertTrue(os.path.exists(data_store.year_path(self.dir, "bars", 2024)))

    def test_appends_without_losing_existing(self):
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 100)]))
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-05", "00010", 110)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 2)
        self.assertEqual(sorted(got["C"].tolist()), [100, 110])

    def test_duplicate_same_day_and_code_keeps_latest(self):
        """訂正が来たら後勝ちで上書きする。"""
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 100)]))
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 999)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 1)
        self.assertEqual(got["C"].iloc[0], 999)

    def test_different_codes_same_day_both_kept(self):
        data_store.merge_into_years(self.dir, "bars", self._frame([
            ("2024-01-04", "00010", 100), ("2024-01-04", "00020", 200)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 2)

    def test_empty_frame_is_noop(self):
        self.assertEqual(data_store.merge_into_years(self.dir, "bars", self._frame([])), [])
        self.assertEqual(data_store.merge_into_years(self.dir, "bars", None), [])

    def test_custom_date_column(self):
        """財務は DiscDate（開示日）で年を分ける。"""
        df = pd.DataFrame([("2024-05-10", "00010", 1)], columns=["DiscDate", "Code", "Sales"])
        written = data_store.merge_into_years(self.dir, "fins", df, date_col="DiscDate")
        self.assertEqual(len(written), 1)
        self.assertIn("fins_2024", written[0])


class TestResetKind(unittest.TestCase):
    """
    取得する列を増やしたとき、既存 parquet には新しい列が入っていないのに
    manifest 上は「取得済み」なので incremental では永久に取り直されない。
    reset_kind はその状態を解消する。
    株価を巻き込むと数時間かかるので、指定した種別だけを消すこと。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for n in ("fins_2020.parquet", "fins_2021.parquet",
                  "bars_2020.parquet", "margin_2020.parquet"):
            open(os.path.join(self.dir, n), "w").write("x")
        # 実際の manifest の形に合わせる（キーは fetched_days）
        self.manifest = {"fins": {"fetched_days": ["2020-01-06"]},
                         "bars": {"fetched_days": ["2020-01-06"]},
                         "margin": {"fetched_days": ["2020-01-10"]}}

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_removes_only_the_named_kind(self):
        removed = data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertEqual(removed, ["fins_2020.parquet", "fins_2021.parquet"])
        left = sorted(os.listdir(self.dir))
        self.assertEqual(left, ["bars_2020.parquet", "margin_2020.parquet"])

    def test_clears_only_that_kind_from_the_manifest(self):
        data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertNotIn("fins", self.manifest)
        self.assertIn("bars", self.manifest)
        self.assertIn("margin", self.manifest)

    def test_reset_makes_every_day_missing_again(self):
        """取り直しの目的はここ。消した後は全営業日が未取得になる。"""
        days = [dt.date(2020, 1, 6), dt.date(2020, 1, 7)]
        data_store.mark_fetched(self.manifest, "fins", days)
        self.assertEqual(data_store.missing_days(self.manifest, "fins", days), [])
        data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertEqual(data_store.missing_days(self.manifest, "fins", days), days)

    def test_unknown_kind_is_harmless(self):
        removed = data_store.reset_kind(self.dir, self.manifest, "nosuch")
        self.assertEqual(removed, [])
        self.assertEqual(len(os.listdir(self.dir)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
