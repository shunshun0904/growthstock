#!/usr/bin/env python3
"""
決算の出し直しを、その日に分かっていた版で付け直す（build_dataset.PIT_FINS・repair_reissued。既定はオフ）。合成の決算で:

- 当たる行（ある期の最初の版の開示日 ≤ 行の日付 < 最後の版の開示日）だけ、決算の列が「その日までに開示された決算だけで
  作った値」（probe_fins_versions.as_of_values と同じ。実データで確かめたもの）になる。当たらない行は1つも変わらない
- 進捗の順位（全銘柄の中の順位）は、付いた版の開示日の順位（progress_pct_by_version）を引く。今の作りは期ごとの最後の版
- progress_pct_point_in_time は、版ごとの順位の表から期ごとの最後の版を引いたものと同じ（作り直しの前と同じ値）
- 窓が無ければ何も変えない。--pit-fins で PIT_FINS が立ち、--no-pit-fins で下りる（両方は付けられない）。
  quarterize_panel を分けて作っても今と同じ表

  python3 tests/test_pit_fins.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import build_dataset as BD  # noqa: E402
import probe_fins_versions as P  # noqa: E402
from test_probe_fins_versions import ROWS, fins, statement  # noqa: E402


def attach_current(rows: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """build と同じ: attach_fins（行の順に戻す）。1年以上古い決算の欠測は build がこのあとに付けるので、ここではしない。"""
    r = rows.assign(_i=np.arange(len(rows)), other=np.arange(len(rows), dtype=float) * 10)
    return BD.attach_fins(r, q).sort_values("_i").reset_index(drop=True).drop(columns=["_i"])


def fake_pct(dates, quarter, basis, ratio, min_history=None, keys=None):
    """版の並び（progress_pct_by_version の並べ替えの後）の番号をそのまま順位にする（引く版を見分けるため）。"""
    return np.arange(len(dates), dtype=float)


class TestRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()               # バリュエーションの無い置き場
        cls.f = fins()
        cls.q = P.panel(cls.f, None, cls.tmp.name)
        cls.cur = attach_current(ROWS, cls.q)
        cls.out, cls.info = BD.repair_reissued(cls.cur, cls.q, cls.f, cls.tmp.name)
        cls.cols = [c for c in cls.q.columns if c not in ("Code", "CurFYSt")]
        cls.mask = BD.in_reissue_windows(ROWS, BD.reissue_windows(BD.fin_versions(cls.f)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_hit(self):
        np.testing.assert_array_equal(self.mask, [True, False, False, False, False, True, False])
        self.assertEqual(self.info["windows"], 2)
        self.assertEqual(self.info["rows"], 2)
        self.assertEqual(self.info["changed"], 2)

    def test_same_as_probe_as_of(self):
        # 当たる行は、probe の「その日の版」（実データで今の作りの再現を確かめた作り）と同じ値。1年以上古い決算の欠測は
        # build が付いた開示日で決めるので、probe の値（欠測を付けた後）とは、古くない行で比べる
        pit = P.as_of_values(ROWS, self.f, None, self.tmp.name)
        for i in np.flatnonzero(self.mask):
            stale = (ROWS["Date"].iloc[i] - pd.Timestamp(self.out["DiscDate"].iloc[i])).days > P.STALE_DAYS
            self.assertFalse(stale)
            for c in self.cols:
                if c in P.CROSS_SECTIONAL:
                    continue
                self.assertTrue(P.same_values(self.out[c].iloc[[i]], pit[c].iloc[[i]]).all(), (i, c))
        d = lambda x: str(pd.Timestamp(x).date())  # noqa: E731
        self.assertEqual(d(self.out["DiscDate"].iloc[0]), "2024-08-05")   # 元の1Q（今の作りは 2024-05-10 の通期）
        self.assertEqual(d(self.cur["DiscDate"].iloc[0]), "2024-05-10")
        self.assertEqual(d(self.out["DiscDate"].iloc[5]), "2024-08-05")   # 30030: 今の作りは1年以上古い開示

    def test_other_rows_unchanged(self):
        keep = ~self.mask
        a, b = self.cur[keep].reset_index(drop=True), self.out[keep].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(list(self.out.columns), list(self.cur.columns))
        np.testing.assert_array_equal(self.out["other"], self.cur["other"])   # 決算でない列は触らない
        for c in self.cols:                                                 # 型は変えない
            self.assertEqual(self.out[c].dtype, self.cur[c].dtype, c)

    def test_no_windows(self):
        f = self.f[self.f["Code"] == "20020"]
        q = P.panel(f, None, self.tmp.name)
        cur = attach_current(ROWS[ROWS["Code"] == "20020"], q)
        out, info = BD.repair_reissued(cur, q, f, self.tmp.name)
        self.assertEqual((info["windows"], info["rows"], info["changed"]), (0, 0, 0))
        self.assertIs(out, cur)


class TestProgressByVersion(unittest.TestCase):
    def test_point_in_time_is_last_version(self):
        f = fins()
        raw_op, raw_ver = BD.raw_versions(BD.actual_disclosures(f))
        by = BD.progress_pct_by_version(raw_ver, raw_op)
        self.assertEqual(len(by), len(raw_ver))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(BD, "progress_percentile", side_effect=fake_pct):
            q = P.panel(f, None, tmp)
            by = BD.progress_pct_by_version(raw_ver, raw_op)
        last = by.drop_duplicates(["Code", "CurFYSt", "quarter"], keep="last")
        want = dict(zip(zip(last["Code"], last["CurFYSt"], last["quarter"]), last["_pct"]))
        got = dict(zip(zip(q["Code"], q["CurFYSt"], q["quarter"]), q["progress_pct"]))
        self.assertEqual(got, {k: want[k] for k in got})

    def test_repair_takes_attached_version(self):
        f = fins()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(BD, "progress_percentile", side_effect=fake_pct):
            q = P.panel(f, None, tmp)
            cur = attach_current(ROWS, q)
            out, _ = BD.repair_reissued(cur, q, f, tmp)
            raw_op, raw_ver = BD.raw_versions(BD.actual_disclosures(f))
            by = BD.progress_pct_by_version(raw_ver, raw_op)

        def pct_of(code, fy, quarter, disc):
            m = by[(by["Code"] == code) & (by["CurFYSt"] == fy) & (by["quarter"] == quarter)
                   & (by["DiscDate"] == pd.Timestamp(disc))]
            return float(m["_pct"].iloc[-1])

        # 10010 の 2024-08-20: その日に分かっていたのは 2024年度1Q の元の版（2024-08-05）
        self.assertEqual(out["progress_pct"].iloc[0], pct_of("10010", "2024-04-01", 1, "2024-08-05"))
        # 出し直しの後（2024-10-01）は今の作りのまま（最後の版 2024-09-20）
        self.assertEqual(out["progress_pct"].iloc[1], pct_of("10010", "2024-04-01", 1, "2024-09-20"))
        self.assertEqual(out["progress_pct"].iloc[1], cur["progress_pct"].iloc[1])
        self.assertEqual(out["progress_pct"].iloc[5], pct_of("30030", "2024-04-01", 1, "2024-08-05"))


class TestSplitAndFlag(unittest.TestCase):
    def test_split_valuation_same_values(self):
        f = fins()
        days = pd.bdate_range("2023-07-01", "2025-10-01")
        val = pd.DataFrame([{"Code": c, "Date": d, "PER": 10.0 + i % 7, "PBR": 1.0 + (i % 5) / 10,
                             "ROE": 0.05 + 0.0001 * i + 0.01 * k}
                            for k, c in enumerate(("10010", "20020", "30030")) for i, d in enumerate(days)])
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as root:
            val.to_parquet(os.path.join(data_dir, "valuation_2024.parquet"), index=False)
            dirs = BD.split_valuation(ROWS["Code"], data_dir, root)
            self.assertEqual(os.listdir(dirs["40040"]), [])
            a = BD.as_of_panel_rows(ROWS, f, None, data_dir=data_dir)
            b = BD.as_of_panel_rows(ROWS, f, None, data_dir=data_dir, dirs=dirs)
        self.assertTrue((a["roe_basis"] == "api").any())
        for c in a.columns:
            self.assertTrue(P.same_values(a[c], b[c]).all(), c)

    def test_flag(self):
        self.assertFalse(BD.PIT_FINS)                                       # 既定は今の作り
        with mock.patch.object(BD, "build") as b, mock.patch.object(BD, "PIT_FINS", False):
            BD.main(["--pit-fins", "--data-dir", "x", "--out", "y.parquet"])
            self.assertTrue(BD.PIT_FINS)
            b.assert_called_once_with("x", "y.parquet")
        self.assertFalse(BD.PIT_FINS)
        with mock.patch.object(BD, "build"), mock.patch.object(BD, "PIT_FINS", False):
            BD.main(["--data-dir", "x", "--out", "y.parquet"])
            self.assertFalse(BD.PIT_FINS)
        with mock.patch.object(BD, "build") as b, mock.patch.object(BD, "PIT_FINS", True):
            BD.main(["--no-pit-fins", "--data-dir", "x", "--out", "y.parquet"])        # 比べ直すときに今の作りへ
            self.assertFalse(BD.PIT_FINS)
            b.assert_called_once_with("x", "y.parquet")
        with mock.patch.object(BD, "build") as b, mock.patch.object(BD, "PIT_FINS", True):
            BD.main(["--data-dir", "x", "--out", "y.parquet"])                        # 何も付けなければ既定のまま
            self.assertTrue(BD.PIT_FINS)
        with mock.patch.object(BD, "build") as b, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            BD.main(["--pit-fins", "--no-pit-fins", "--data-dir", "x", "--out", "y.parquet"])   # 両方は付けられない
        b.assert_not_called()

    def test_panel_unchanged_by_split_helpers(self):
        # quarterize_panel の入り口を actual_disclosures / raw_versions に分けても、同じ行・同じ列
        f = pd.concat([fins(), pd.DataFrame([statement("50050", "2024-04-01", "2Q", "2024-11-06", 300)])],
                      ignore_index=True)
        d = BD.actual_disclosures(f)
        self.assertTrue(set(d["quarter"].unique()) <= {1, 2, 3, 4})
        raw_op, raw_ver = BD.raw_versions(d)
        self.assertEqual(len(raw_op), len(d))
        self.assertIn("FOP", raw_ver.columns)
        self.assertNotIn("FOP", raw_op.columns)


if __name__ == "__main__":
    unittest.main()
