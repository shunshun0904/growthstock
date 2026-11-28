#!/usr/bin/env python3
"""
probe: 決算の出し直しと学習データ（research/exp/probe_fins_versions.py）。合成の決算で:

- 出し直しの窓: 同じ期の版が別の日に出た期の [最初の版, 最後の版)。同じ日の重複は窓にしない
- 当たる行: 窓に入る行だけ（窓の前・後、出し直しの無い銘柄は当たらない）
- 今の作り（全部の決算で quarterize_panel → attach_fins → 1年以上古い決算は欠測）と、その日の版（その日までの決算だけで
  作る）の違い: 窓の中の行は今の作りでより古い開示が付く（older）か、1年以上古くなって決算が付かない（lost）。窓の外の行は
  全部の列が同じ
- 列ごとの数え方（変わった行・欠測になった行・値が戻った行）
- 銘柄ごとに分けたバリュエーションの置き場で作り直しても、全部の置き場と同じ値

  python3 tests/test_probe_fins_versions.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import probe_fins_versions as P  # noqa: E402


def statement(code, fy, period, disc, sales, eq=1000.0):
    """quarterize_panel に渡せる決算の行（tests/test_progress.py の statement と同じ形に、純資産と株数を足した）。"""
    return {"Code": code, "DiscDate": disc, "DiscTime": "15:00", "CurPerType": period,
            "CurFYSt": fy, "CurPerEn": disc, "DocType": "FinancialStatements",
            "Sales": float(sales), "OP": float(sales) / 10, "NP": float(sales) / 20, "EPS": float(sales) / 200,
            "Eq": eq, "ShEq": eq, "TA": 2 * eq, "CashEq": eq / 4, "ROE": None, "FSales": None, "FNP": None,
            "FEPS": None, "FOP": float(sales) * 2, "ShOutFY": 1_000_000, "TrShFY": 0}


def fins():
    rows = []
    for code in ("10010", "20020"):
        rows += [statement(code, "2023-04-01", "1Q", "2023-08-05", 100),
                 statement(code, "2023-04-01", "2Q", "2023-11-05", 210),
                 statement(code, "2023-04-01", "3Q", "2024-02-05", 320),
                 statement(code, "2023-04-01", "FY", "2024-05-10", 440),
                 statement(code, "2024-04-01", "1Q", "2024-08-05", 120),
                 statement(code, "2024-04-01", "2Q", "2024-11-05", 250)]
    # 10010 は 2024年度1Q を 2024-09-20 に出し直す（売上と純資産が変わる）
    rows.append(statement("10010", "2024-04-01", "1Q", "2024-09-20", 130, eq=1100.0))
    # 30030 は 2024年度1Q の出し直しが1年以上あと。1つ前の開示は1年以上古い
    rows += [statement("30030", "2023-04-01", "FY", "2024-05-10", 400),
             statement("30030", "2024-04-01", "1Q", "2024-08-05", 110),
             statement("30030", "2024-04-01", "1Q", "2025-09-01", 115)]
    # 40040 は同じ日に2つの版（窓にしない）
    rows += [statement("40040", "2024-04-01", "1Q", "2024-08-06", 90),
             statement("40040", "2024-04-01", "1Q", "2024-08-06", 91)]
    return pd.DataFrame(rows)


ROWS = pd.DataFrame({"Code": ["10010", "10010", "10010", "10010", "20020", "30030", "40040"],
                     "Date": pd.to_datetime(["2024-08-20", "2024-10-01", "2024-12-01", "2024-08-01",
                                             "2024-08-20", "2025-06-02", "2024-08-20"])})


class TestWindows(unittest.TestCase):
    def test_windows_and_rows(self):
        ver = P.statement_versions(fins())
        win = P.reissue_windows(ver)
        got = {(r.Code, str(r.start.date()), str(r.end.date())) for r in win.itertuples()}
        self.assertEqual(got, {("10010", "2024-08-05", "2024-09-20"), ("30030", "2024-08-05", "2025-09-01")})
        np.testing.assert_array_equal(P.in_windows(ROWS, win), [True, False, False, False, False, True, False])

    def test_versions_skip_revision_only(self):
        f = fins()
        rev = statement("10010", "2024-04-01", "1Q", "2024-08-10", 0)
        rev.update({"Sales": None, "OP": None, "NP": None, "EPS": None, "DocType": "EarnForecastRevision"})
        ver = P.statement_versions(pd.concat([f, pd.DataFrame([rev])], ignore_index=True))
        self.assertEqual(len(ver), len(f))                     # 実績の無い開示（予想の修正だけ）は版に数えない


class TestAsOf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()               # バリュエーションの無い置き場
        f = fins()
        cls.q = P.panel(f, None, cls.tmp.name)
        cls.cur = P.attach(ROWS, cls.q)
        cls.pit = P.as_of_values(ROWS, f, None, cls.tmp.name).reindex(columns=cls.cur.columns)
        cls.cols = [c for c in cls.q.columns if c not in P.KEYS and c not in P.CROSS_SECTIONAL]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_kinds(self):
        kd = P.kinds(self.cur, self.pit, self.cols).tolist()
        self.assertEqual(kd, ["older", "none", "none", "none", "none", "lost", "none"])

    def test_attached_disclosures(self):
        d = lambda x: [None if pd.isna(v) else str(pd.Timestamp(v).date()) for v in x]  # noqa: E731
        # 窓の中（2024-08-20）: 今の作りは前の期（2024-05-10 の通期）、その日の版は元の1Q（2024-08-05）
        self.assertEqual(d(self.cur["DiscDate"])[0], "2024-05-10")
        self.assertEqual(d(self.pit["DiscDate"])[0], "2024-08-05")
        # 出し直しの後（2024-10-01）は、どちらも出し直した版
        self.assertEqual(d(self.cur["DiscDate"])[1], "2024-09-20")
        self.assertEqual(d(self.pit["DiscDate"])[1], "2024-09-20")
        # 30030（2025-06-02）: 今の作りの開示は1年以上古いので財務の列は欠測、その日の版は元の1Q
        self.assertTrue(self.cur.loc[5, self.cols].isna().all())
        self.assertEqual(d(self.pit["DiscDate"])[5], "2024-08-05")

    def test_outside_windows_same(self):
        c = P.compare(self.cur.iloc[[1, 2, 3, 4, 6]], self.pit.iloc[[1, 2, 3, 4, 6]], self.cols)
        self.assertEqual(int(c["changed"].sum()), 0)

    def test_current_equals_full_panel_attach(self):
        # 今の作りは build と同じ（全部の決算の quarterize_panel を attach_fins で付け、1年以上古い決算を欠測に）
        import build_dataset as BD

        r = ROWS.assign(_i=np.arange(len(ROWS)))
        ref = BD.attach_fins(r, self.q).sort_values("_i").reset_index(drop=True)
        stale = (ref["Date"] - ref["DiscDate"]).dt.days > P.STALE_DAYS
        ref.loc[stale, [c for c in self.q.columns if c not in P.KEYS]] = np.nan
        for col in self.cols:
            self.assertTrue(P.same_values(self.cur[col], ref[col]).all(), col)

    def test_compare_counts(self):
        a = pd.DataFrame({"x": [1.0, np.nan, 2.0, 3.0, np.nan], "s": ["a", "b", None, "c", None]})
        b = pd.DataFrame({"x": [1.0, 5.0, np.nan, 3.3, np.nan], "s": ["a", "c", "d", "c", None]})
        c = P.compare(a, b, ["x", "s"]).set_index("col")
        self.assertEqual(c.loc["x", "changed"], 3)
        self.assertEqual(c.loc["x", "to_nan"], 1)              # 今の作りで欠測（その日の版では値がある）
        self.assertEqual(c.loc["x", "from_nan"], 1)
        self.assertEqual(c.loc["x", "both"], 1)
        self.assertAlmostEqual(c.loc["x", "rel_med"], 0.3 / 3.3)
        self.assertEqual(c.loc["s", "changed"], 2)


class TestSplitValuation(unittest.TestCase):
    """銘柄ごとに分けたバリュエーション（API の ROE）の置き場で作り直しても、全部の置き場と同じ値になる。"""

    def test_same_as_full_dir(self):
        f = fins()
        days = pd.bdate_range("2023-07-01", "2025-10-01")
        val = pd.DataFrame([{"Code": c, "Date": d, "PER": 10.0 + i % 7, "PBR": 1.0 + (i % 5) / 10,
                             "ROE": 0.05 + 0.0001 * i + 0.01 * k}
                            for k, c in enumerate(("10010", "20020", "30030", "99990"))
                            for i, d in enumerate(days)])
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as root:
            val[val["Date"] < "2024-07-01"].to_parquet(os.path.join(data_dir, "valuation_2023.parquet"), index=False)
            val[val["Date"] >= "2024-07-01"].to_parquet(os.path.join(data_dir, "valuation_2024.parquet"), index=False)
            dirs = P.split_valuation(ROWS["Code"], data_dir, root)
            self.assertEqual(sorted(dirs), ["10010", "20020", "30030", "40040"])
            one = pd.read_parquet(os.path.join(dirs["10010"], "valuation_0.parquet"))
            self.assertEqual(set(one["Code"]), {"10010"})
            self.assertEqual(len(one), int((val["Code"] == "10010").sum()))
            self.assertEqual(os.listdir(dirs["40040"]), [])       # バリュエーションの無い銘柄は空の置き場
            full = P.as_of_values(ROWS, f, None, data_dir)
            split = P.as_of_values(ROWS, f, None, data_dir, dirs=dirs).reindex(columns=full.columns)
        self.assertTrue((full["roe_basis"] == "api").any())      # API の ROE を使っている
        self.assertTrue(full["ROE_q0"].notna().any())
        for col in full.columns:
            self.assertTrue(P.same_values(full[col], split[col]).all(), col)

    def test_no_valuation(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as root:
            self.assertEqual(P.split_valuation(["10010"], data_dir, root), {"10010": data_dir})


if __name__ == "__main__":
    unittest.main()
