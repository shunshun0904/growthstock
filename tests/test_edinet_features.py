#!/usr/bin/env python3
"""
research/edinet_features.py の単体テスト。

肝は「先読みしない」「比べられない年度を比べない」「分割で壊れない」。
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import edinet_features as EF  # noqa: E402


def fin_rows(code, years, rev, op=None, ni=None, submit=None, std="JP", basis="consolidated",
             **extra):
    rows = []
    for i, fy in enumerate(years):
        r = {"jq_code": code, "edinet_code": "E" + code, "fiscal_year": fy,
             "doc_id": f"S{code}{fy}", "revenue": rev[i],
             "operating_income": (op or rev)[i], "net_income": (ni or rev)[i],
             "adjusted_eps": rev[i] / 10, "total_assets": 1000.0, "total_liabilities": 400.0,
             "net_assets": 600.0, "cash": 100.0, "num_employees": 50,
             "accounting_standard": std[i] if isinstance(std, list) else std,
             "basis": basis[i] if isinstance(basis, list) else basis,
             "submit_date": (submit[i] if submit else f"{fy}-06-25 15:16")}
        for k, v in extra.items():
            r[k] = v[i] if isinstance(v, (list, tuple)) else v
        rows.append(r)
    return rows


class TestPanel(unittest.TestCase):
    def test_lags_align_by_fiscal_year(self):
        fin = pd.DataFrame(fin_rows("10000", [2020, 2021, 2022, 2023], [100, 80, 90, 120]))
        p = EF.annual_panel(fin).set_index("fiscal_year")
        self.assertEqual(p.loc[2023, "revenue"], 120)
        self.assertEqual(p.loc[2023, "revenue_y1"], 90)
        self.assertEqual(p.loc[2023, "revenue_y2"], 80)
        self.assertEqual(p.loc[2023, "revenue_y3"], 100)
        self.assertTrue(np.isnan(p.loc[2020, "revenue_y1"]))

    def test_missing_year_breaks_lag(self):
        fin = pd.DataFrame(fin_rows("10000", [2020, 2022, 2023], [100, 90, 120]))
        p = EF.annual_panel(fin).set_index("fiscal_year")
        self.assertTrue(np.isnan(p.loc[2023, "revenue_y2"]))     # 2021 が無い
        self.assertEqual(p.loc[2023, "revenue_y3"], 100)
        self.assertEqual(p.loc[2023, "revenue_y1"], 90)

    def test_first_filing_wins_for_same_year(self):
        rows = fin_rows("10000", [2022, 2022], [100, 999],
                        submit=["2022-06-25 10:00", "2022-09-01 10:00"])
        rows[1]["doc_id"] = "S-amend"
        p = EF.annual_panel(pd.DataFrame(rows))
        self.assertEqual(len(p), 1)
        self.assertEqual(p["revenue"].iloc[0], 100)

    def test_standard_change_blanks_lag(self):
        fin = pd.DataFrame(fin_rows("10000", [2021, 2022, 2023], [100, 110, 120],
                                    std=["JP", "IFRS", "IFRS"]))
        p = EF.annual_panel(fin).set_index("fiscal_year")
        self.assertEqual(p.loc[2023, "revenue_y1"], 110)         # IFRS 同士
        self.assertTrue(np.isnan(p.loc[2023, "revenue_y2"]))     # JP とは比べない
        self.assertTrue(np.isnan(p.loc[2022, "revenue_y1"]))

    def test_basis_change_blanks_lag(self):
        fin = pd.DataFrame(fin_rows("10000", [2022, 2023], [100, 120],
                                    basis=["standalone", "consolidated"]))
        p = EF.annual_panel(fin).set_index("fiscal_year")
        self.assertTrue(np.isnan(p.loc[2023, "revenue_y1"]))

    def test_debt_and_goodwill_default_to_zero_when_bs_exists(self):
        rows = fin_rows("10000", [2022, 2023], [100, 120])
        rows[1]["short_term_loans"] = 30.0
        rows[1]["long_term_loans"] = 70.0
        p = EF.annual_panel(pd.DataFrame(rows)).set_index("fiscal_year")
        self.assertEqual(p.loc[2022, "debt"], 0.0)               # 借入なし = 0
        self.assertEqual(p.loc[2023, "debt"], 100.0)             # ibd_* が無ければ借入金の合計
        self.assertEqual(p.loc[2022, "goodwill"], 0.0)
        self.assertAlmostEqual(p.loc[2023, "debt_r"], 0.1)

    def test_ibd_preferred_over_loans(self):
        rows = fin_rows("10000", [2023], [100])
        rows[0].update({"ibd_current": 10.0, "ibd_noncurrent": 20.0, "short_term_loans": 999.0})
        p = EF.annual_panel(pd.DataFrame(rows))
        self.assertEqual(p["debt"].iloc[0], 30.0)

    def test_no_balance_sheet_keeps_nan(self):
        rows = fin_rows("10000", [2023], [100])
        rows[0]["total_assets"] = None
        rows[0]["total_liabilities"] = None
        p = EF.annual_panel(pd.DataFrame(rows))
        self.assertTrue(np.isnan(p["debt"].iloc[0]))
        self.assertTrue(np.isnan(p["goodwill"].iloc[0]))

    def test_share_ratios_are_split_invariant(self):
        rows = fin_rows("10000", [2022, 2023], [100, 100],
                        shares_issued=[1000, 10000], float_shares=[400, 4000],
                        treasury_shares_count=[50, 500])
        p = EF.annual_panel(pd.DataFrame(rows)).set_index("fiscal_year")
        self.assertAlmostEqual(p.loc[2023, "tsy_r"], 0.05)
        self.assertAlmostEqual(p.loc[2023, "tsy_r_y1"], 0.05)

    def test_avail_date_is_next_day(self):
        fin = pd.DataFrame(fin_rows("10000", [2023], [100], submit=["2023-06-25 15:16"]))
        p = EF.annual_panel(fin)
        self.assertEqual(p["avail_date"].iloc[0], pd.Timestamp("2023-06-26"))

    def test_missing_submit_date_still_serves_as_lag(self):
        fin = pd.DataFrame(fin_rows("10000", [2022, 2023], [100, 120],
                                    submit=[None, "2023-06-25 12:00"]))
        p = EF.annual_panel(fin).set_index("fiscal_year")
        self.assertEqual(p.loc[2023, "revenue_y1"], 100)
        self.assertTrue(pd.isna(p.loc[2022, "avail_date"]))


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # 成長 → 落ち込み → 立て直して新高値（V字）
        fin = pd.DataFrame(fin_rows("10000", [2020, 2021, 2022, 2023], [100, 120, 90, 130]))
        self.panel = EF.annual_panel(fin)
        self.feats = EF.feature_frame(self.panel).set_index("fiscal_year")

    def test_change_rates(self):
        f = self.feats.loc[2023]
        self.assertAlmostEqual(f["ed_revenue_yoy1"], EF.sym(pd.Series([130.0]), pd.Series([90.0]))[0])
        self.assertAlmostEqual(f["ed_revenue_yoy2"], EF.sym(pd.Series([90.0]), pd.Series([120.0]))[0])
        self.assertAlmostEqual(f["ed_revenue_yoy3"], EF.sym(pd.Series([120.0]), pd.Series([100.0]))[0])
        self.assertAlmostEqual(f["ed_revenue_chg3y"], EF.sym(pd.Series([130.0]), pd.Series([100.0]))[0])
        self.assertGreater(f["ed_revenue_accel"], 0)

    def test_trajectory_shape(self):
        f = self.feats.loc[2023]
        self.assertEqual(f["ed_revenue_vshape3"], 1.0)           # 90 < max(120,100) かつ 130 > 90
        self.assertEqual(f["ed_revenue_high4"], 1.0)             # 130 > max(90,120,100)
        self.assertLess(f["ed_revenue_dip3"], 0)                 # 前期は山から落ちていた
        self.assertGreater(f["ed_revenue_recovery3"], 0)         # 過去3期の山を超えた
        # 4期揃わない行では NaN
        self.assertTrue(np.isnan(self.feats.loc[2022, "ed_revenue_vshape3"]))

    def test_ratio_levels_and_changes(self):
        f = self.feats.loc[2023]
        self.assertAlmostEqual(f["ed_opm"], 1.0)                 # op = rev の作り
        self.assertAlmostEqual(f["ed_opm_chg1"], 0.0)
        self.assertAlmostEqual(f["ed_roe"], 130 / 600)
        self.assertAlmostEqual(f["ed_roe_chg1"], (130 - 90) / 600)

    def test_column_groups(self):
        cols = EF.columns("all")
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(len(EF.columns("core")), 40)
        self.assertEqual(len(EF.columns("ratio")), len(EF.RATIOS))
        self.assertEqual(len(EF.columns("mcap")), len(EF.MCAP))
        for c in cols:
            self.assertTrue(c.startswith("ed_"))
            if c not in EF.columns("mcap"):
                self.assertIn(c, self.feats.columns)      # mcap は attach のときに作る
        with self.assertRaises(KeyError):
            EF.columns("nope")

    def test_split_adjusted_share_change(self):
        # 1:2 分割（factor 2 → 1）で株数が倍になっても、調整後の変化は 0
        rows = fin_rows("10000", [2022, 2023], [100, 100],
                        shares_issued=[1000, 2000], split_adjustment_factor=[2.0, 1.0])
        p = EF.annual_panel(pd.DataFrame(rows)).set_index("fiscal_year")
        self.assertAlmostEqual(p.loc[2023, "shares_adj"], p.loc[2023, "shares_adj_y1"])
        f = EF.feature_frame(EF.annual_panel(pd.DataFrame(rows))).set_index("fiscal_year")
        self.assertAlmostEqual(f.loc[2023, "ed_shares_adj_yoy1"], 0.0)

    def test_extraordinary_defaults_to_zero_only_for_jp(self):
        rows = fin_rows("10000", [2023], [100], std="JP") + fin_rows("20000", [2023], [100], std="IFRS")
        rows[0]["extraordinary_loss"] = 30.0
        p = EF.annual_panel(pd.DataFrame(rows)).set_index("jq_code")
        self.assertEqual(p.loc["10000", "extraordinary_net"], -30.0)   # 特別利益は無い = 0
        self.assertTrue(np.isnan(p.loc["20000", "extraordinary_net"]))  # IFRS には概念が無い


class TestAttach(unittest.TestCase):
    def setUp(self):
        fin = pd.DataFrame(
            fin_rows("10000", [2021, 2022, 2023], [100, 110, 120],
                     submit=["2021-06-25 12:00", "2022-06-25 12:00", "2023-06-25 15:16"])
            + fin_rows("20000", [2022], [50], submit=["2022-06-20 12:00"]))
        self.feats = EF.feature_frame(EF.annual_panel(fin))

    def test_point_in_time(self):
        frame = pd.DataFrame({
            "Code": ["10000", "10000", "10000", "20000", "30000"],
            "Date": ["2023-06-25", "2023-06-26", "2023-01-10", "2022-06-21", "2023-01-01"],
            "x": [1, 2, 3, 4, 5],
        })
        out = EF.attach(frame, self.feats)
        self.assertEqual(out["ed_fiscal_year"].tolist()[:3], [2022, 2023, 2022])  # 提出日当日はまだ前期
        self.assertEqual(out["ed_fiscal_year"].iloc[3], 2022)
        self.assertTrue(np.isnan(out["ed_fiscal_year"].iloc[4]))          # データの無い銘柄
        self.assertEqual(out["x"].tolist(), [1, 2, 3, 4, 5])               # 行の順序は元のまま
        self.assertAlmostEqual(out["ed_revenue_yoy1"].iloc[1],
                               EF.sym(pd.Series([120.0]), pd.Series([110.0]))[0])

    def test_market_cap_features(self):
        frame = pd.DataFrame({"Code": ["10000"], "Date": ["2023-09-01"], "market_cap": [10.0]})  # 10億円
        fin = pd.DataFrame(fin_rows("10000", [2023], [100], submit=["2023-06-25 12:00"],
                                    cf_operating=[300.0], capex=[100.0], cash=[500.0],
                                    cash_dividends_paid=[50.0], retained_earnings=[400.0]))
        out = EF.attach(frame, EF.feature_frame(EF.annual_panel(fin)))
        self.assertAlmostEqual(out["ed_fcf_yield"].iloc[0], 200.0 / 1e9)
        self.assertAlmostEqual(out["ed_netcash_mcap"].iloc[0], 500.0 / 1e9)    # 借入なし
        self.assertAlmostEqual(out["ed_div_yield"].iloc[0], 50.0 / 1e9)
        self.assertTrue(np.isnan(out["ed_buyback_yield"].iloc[0]))            # 前期が無い
        frame2 = pd.DataFrame({"Code": ["10000"], "Date": ["2023-09-01"]})
        out2 = EF.attach(frame2, EF.feature_frame(EF.annual_panel(fin)))
        self.assertTrue(np.isnan(out2["ed_fcf_yield"].iloc[0]))               # 時価総額が無ければ NaN

    def test_stale_filing_is_blanked(self):
        frame = pd.DataFrame({"Code": ["20000", "20000"], "Date": ["2023-06-01", "2023-09-01"]})
        out = EF.attach(frame, self.feats)
        self.assertEqual(out["ed_fiscal_year"].iloc[0], 2022)              # 346日: まだ有効
        self.assertTrue(np.isnan(out["ed_fiscal_year"].iloc[1]))          # 438日: 開示が止まっている
        self.assertTrue(np.isnan(out["ed_opm"].iloc[1]))

    def test_build_without_file_adds_nan_columns(self):
        frame = pd.DataFrame({"Code": ["10000"], "Date": ["2023-01-01"]})
        out = EF.build(frame, path="/nonexistent/edinet_fin.parquet")
        for c in EF.columns("all"):
            self.assertTrue(np.isnan(out[c].iloc[0]))


if __name__ == "__main__":
    unittest.main()


class 固有項目の比率(unittest.TestCase):
    """2026-09-22 に足した EDINET DB 固有の7つ。

    運用者の判断「edinet DB固有のデータがあるはずなので、代替のとか
    以前に追加はします」。J-Quants には株主総利回りも役員報酬も
    平均年齢も無いので、比較の対象ではなく単純な追加になる。
    """

    def fin(self):
        """1社1年ぶんの最小限の有報。金額の単位は円。"""
        return pd.DataFrame({
            "jq_code": ["13010"], "edinet_code": ["E00001"],
            "fiscal_year": [2025], "submit_date": [pd.Timestamp("2025-06-20")],
            "accounting_standard": ["JP"], "basis": ["consolidated"],
            "revenue": [1_000_000_000.0], "total_assets": [2_000_000_000.0],
            "net_assets": [1_000_000_000.0], "net_income": [50_000_000.0],
            "trade_receivables": [200_000_000.0],
            "split_adjustment_factor": [1.0], "shares_issued": [1_000_000.0],
            "num_employees": [100.0],
            # ここから今回足したもの
            "total_shareholder_return": [1.25],
            "avg_age": [42.0], "avg_tenure_years": [15.0],
            "net_defined_benefit_liability": [30_000_000.0],
            "deferred_tax_assets": [16_000_000.0],
            # 控除項目なので**負**で来る（実測 99.2%）
            "allowance_for_doubtful_accounts": [-4_000_000.0],
            "director_remuneration_total": [120_000_000.0],
            "director_remuneration_headcount": [6.0],
            "female_director_ratio": [0.25],
        })

    def panel(self):
        return EF.annual_panel(self.fin())

    def test_7列すべて作られる(self):
        p = self.panel()
        for c in EF.RATIOS_EXTRA:
            self.assertIn(c, p.columns, c)
            self.assertTrue(pd.notna(p[c].iloc[0]), f"{c} が欠測")

    def test_割り算の相手が正しい(self):
        p = self.panel().iloc[0]
        self.assertAlmostEqual(p["dbo_r"], 30 / 2000)        # ÷ 総資産
        self.assertAlmostEqual(p["dta_r"], 16 / 2000)        # ÷ 総資産
        self.assertAlmostEqual(p["dir_pay_r"], 120 / 1000)   # ÷ 売上
        self.assertAlmostEqual(p["dir_pay_head"], 20_000_000.0)  # 1人あたり

    def test_貸倒引当金は絶対値で正になる(self):
        """控除項目なので負で来る。そのまま割ると符号が逆になり、
        「引当が厚いほど小さい値」という読み違いを生む。"""
        p = self.panel().iloc[0]
        self.assertGreater(p["bad_debt_r"], 0)
        self.assertAlmostEqual(p["bad_debt_r"], 4 / 200)     # ÷ 売上債権

    def test_比率はそのまま通る(self):
        p = self.panel().iloc[0]
        self.assertAlmostEqual(p["tsr"], 1.25)
        self.assertAlmostEqual(p["fem_dir_r"], 0.25)

    def test_平均年齢と勤続年数は水準として持つ(self):
        p = self.panel().iloc[0]
        for c in ("avg_age", "avg_tenure_years"):
            self.assertIn(c, EF.PEOPLE, c)
            self.assertAlmostEqual(p[c], 42.0 if c == "avg_age" else 15.0)

    def test_充足率の低い列は入れていない(self):
        """ghg_*（1.5〜4.9%）と gender_pay_gap_*（17.1%）。
        いま入れても全欠測に近く、判定できない。"""
        cols = set(EF.columns("all"))
        for bad in ("ghg_scope1", "ghg_scope2", "ghg_scope3",
                    "gender_pay_gap_regular", "gender_pay_gap_all"):
            self.assertFalse(any(bad in c for c in cols), bad)
