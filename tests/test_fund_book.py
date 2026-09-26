#!/usr/bin/env python3
"""
本の第2・3章の抜け12列（build_dataset.add_book_ratios / BOOK_COLS、features.GROUPS["fund_book"]）の
テスト。合成の四半期パネルで式を確かめる。

  python3 tests/test_fund_book.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import feature_dict as FD  # noqa: E402


def panel():
    """1銘柄 × 2事業年度 × 4四半期（開示順に並ぶ）。CashEq は 2Q と通期だけ。"""
    rows = []
    for fy, base in ((2022, 100.0), (2023, 120.0)):
        for q in (1, 2, 3, 4):
            rows.append({
                "Code": "10000", "CurFYSt": pd.Timestamp(f"{fy}-04-01"), "quarter": q,
                "DiscDate": pd.Timestamp(f"{fy}-{3*q+4 if q < 3 else 3*q+4-12:02d}-10")
                if q < 3 else pd.Timestamp(f"{fy + 1}-{(3*q+4) % 12:02d}-10"),
                "DocType": f"{'FY' if q == 4 else str(q)+'Q'}FinancialStatements_Consolidated_"
                           + ("IFRS" if fy == 2023 else "JP"),
                "ROE": 10.0, "payout_ratio": 30.0, "sales_ttm": 1000.0, "Eq": 500.0,
                "dps": base / 10.0, "BPS": base, "shares_out": 1000.0 * (1.1 if fy == 2023 else 1.0),
                "CashEq": (base * 3 if q in (2, 4) else np.nan),
                "CFO": base * q, "CFI": -base * q / 2, "CFF": -10.0 * q,
            })
    df = pd.DataFrame(rows)
    # 開示日順（quarterize_panel の add_book_ratios 呼び出し時点の並び）
    return df.sort_values(["Code", "DiscDate"]).reset_index(drop=True)


class TestBookRatios(unittest.TestCase):
    def setUp(self):
        self.df = B.add_book_ratios(panel())

    def test_columns_present(self):
        for c in B.BOOK_COLS:
            self.assertIn(c, self.df.columns, c)
        self.assertEqual(set(F.GROUPS["fund_book"]),
                         {"cash_mcap", "cfi_mcap", "cff_mcap"} | (set(B.BOOK_COLS) - {"cash_eq_last", "cfi_cum", "cff_cum"}))
        self.assertEqual(len(F.columns("all_plus_prog_listing_book")), 206 + 12)
        for c in F.GROUPS["fund_book"]:
            self.assertIn(c, FD.COL_JA, c)

    def test_cash_ffill_and_cumulative(self):
        d = self.df
        fy22 = d[d["CurFYSt"] == pd.Timestamp("2022-04-01")].set_index("quarter")
        self.assertTrue(np.isnan(fy22.loc[1, "cash_eq_last"]))      # まだ一度も開示が無い
        self.assertEqual(fy22.loc[2, "cash_eq_last"], 300.0)
        self.assertEqual(fy22.loc[3, "cash_eq_last"], 300.0)         # 3Q は 2Q の値を引き継ぐ
        self.assertEqual(fy22.loc[4, "cash_eq_last"], 300.0)
        self.assertEqual(fy22.loc[3, "cfi_cum"], -150.0)
        self.assertEqual(fy22.loc[3, "cff_cum"], -30.0)

    def test_ratios(self):
        d = self.df
        self.assertTrue(np.allclose(d["sustainable_growth"], 10.0 * 0.7))
        self.assertTrue(np.allclose(d["equity_turnover"], 2.0))
        fy22 = d[d["CurFYSt"] == pd.Timestamp("2022-04-01")]
        fy23 = d[d["CurFYSt"] == pd.Timestamp("2023-04-01")]
        self.assertTrue((fy22["acct_ifrs"] == 0.0).all())
        self.assertTrue((fy23["acct_ifrs"] == 1.0).all())

    def test_year_over_year_same_quarter(self):
        d = self.df
        fy22 = d[d["CurFYSt"] == pd.Timestamp("2022-04-01")].set_index("quarter")
        fy23 = d[d["CurFYSt"] == pd.Timestamp("2023-04-01")].set_index("quarter")
        # 前の年度が無い行は欠測
        for c in ("cfo_yoy_sym", "fcf_yoy_sym", "div_growth_sym", "div_up", "bps_yoy", "shares_yoy"):
            self.assertTrue(fy22[c].isna().all(), c)
        # 2Q どうし: CFO 240 vs 200 → (40)/(440) = 9.09
        self.assertAlmostEqual(fy23.loc[2, "cfo_yoy_sym"], 40 / 440 * 100, places=6)
        # FCF 2Q: 120 vs 100 → 20/220
        self.assertAlmostEqual(fy23.loc[2, "fcf_yoy_sym"], 20 / 220 * 100, places=6)
        self.assertAlmostEqual(fy23.loc[1, "div_growth_sym"], 2 / 22 * 100, places=6)
        self.assertTrue((fy23["div_up"] == 1.0).all())
        self.assertTrue(np.allclose(fy23["bps_yoy"], 20.0))
        self.assertTrue(np.allclose(fy23["shares_yoy"], 10.0, atol=1e-9))

    def test_sym_change_edges(self):
        cur = pd.Series([1.0, -1.0, 0.0, np.nan])
        prev = pd.Series([-1.0, 1.0, 0.0, 1.0])
        out = B._sym_change(cur, prev)
        self.assertEqual(out.iloc[0], 100.0)
        self.assertEqual(out.iloc[1], -100.0)
        self.assertTrue(np.isnan(out.iloc[2]))     # 両方 0 は定義できない
        self.assertTrue(np.isnan(out.iloc[3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSplitAdjust(unittest.TestCase):
    def test_split_does_not_look_like_issuance(self):
        """1:2 の分割（株数 2倍・BPS 半分）を、adj を渡すと前年同期比 0 に戻す。"""
        df = panel()
        df.loc[df["CurFYSt"] == pd.Timestamp("2023-04-01"), "shares_out"] = 2000.0
        df.loc[df["CurFYSt"] == pd.Timestamp("2023-04-01"), "BPS"] = 50.0
        df.loc[df["CurFYSt"] == pd.Timestamp("2022-04-01"), "BPS"] = 100.0
        # 分割日 2023-06-01。それより前の close_raw ÷ close = 2、後は 1
        dates = pd.bdate_range("2022-01-01", "2024-12-31")
        adj = pd.DataFrame({"Code": "10000", "Date": dates, "close": 100.0,
                            "close_raw": np.where(dates < pd.Timestamp("2023-06-01"), 200.0, 100.0)})
        raw = B.add_book_ratios(panel().assign(
            shares_out=lambda d: np.where(d["CurFYSt"] == pd.Timestamp("2023-04-01"), 2000.0, 1000.0),
            BPS=lambda d: np.where(d["CurFYSt"] == pd.Timestamp("2023-04-01"), 50.0, 100.0)))
        fy23_raw = raw[raw["CurFYSt"] == pd.Timestamp("2023-04-01")]
        self.assertTrue(np.allclose(fy23_raw["shares_yoy"], 100.0))     # 未調整だと増資に見える
        self.assertTrue(np.allclose(fy23_raw["bps_yoy"], -50.0))
        out = B.add_book_ratios(df, adj=adj)
        fy23 = out[out["CurFYSt"] == pd.Timestamp("2023-04-01")]
        self.assertTrue(np.allclose(fy23["shares_yoy"], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(fy23["bps_yoy"], 0.0, atol=1e-9))
