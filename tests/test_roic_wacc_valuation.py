"""Independent arithmetic, missingness, source timing, and causal features."""
import sys
import unittest
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import roic_wacc_valuation as R
import valuation_gap as G
import capital_efficiency as CE
import features as F
from test_capital_efficiency import inputs


class Arithmetic(unittest.TestCase):
    def test_wacc_independent_hand_calculation(self):
        # Ke=.06, Kd=.025; E=1000, D=200, tax=.3.
        self.assertAlmostEqual(float(R.wacc(1000, 200, .01, 1, .05, .3)), 63.5 / 1200)

    def test_unknown_debt_is_not_debt_free(self):
        self.assertTrue(np.isnan(R.wacc(1000, np.nan, .01, 1, .05, .3)))
        self.assertAlmostEqual(float(R.wacc(1000, 0, .01, 1, .05, .3)), .06)

    def test_steady_ev_next_nopat_and_equity_bridge(self):
        ev = R.enterprise_value(100, .1, .08, .02)
        self.assertAlmostEqual(float(ev), 1360.)
        equity, price = R.equity_bridge(ev, 200, 10, 50, 10)
        self.assertAlmostEqual(float(equity), 1200.)
        self.assertAlmostEqual(float(price), 120.)

    def test_missing_cash_or_nci_never_zero(self):
        for args in [(1360, 200, 10, np.nan, 10), (1360, 200, np.nan, 50, 10)]:
            self.assertTrue(np.isnan(R.equity_bridge(*args)[1]))

    def test_nonpositive_equity_not_clipped(self):
        self.assertTrue(np.isnan(R.equity_bridge(100, 200, 0, 0, 10)[1]))

    def test_financial_domain_guards(self):
        for n, r, w, g in [(0, .1, .08, .02), (-1, .1, .08, .02), (100, 0, .08, .02),
                           (100, .1, .02, .02), (100, .1, .029, .02), (100, .01, .08, .02),
                           (100, .1, .08, -.01)]:
            with self.subTest(n=n, r=r, w=w, g=g):
                self.assertTrue(np.isnan(R.enterprise_value(n, r, w, g)))
        self.assertTrue(np.isfinite(R.enterprise_value(100, .1, .03, .02)))

    def test_fade_equals_steady_when_roic_equals_wacc(self):
        self.assertAlmostEqual(float(R.enterprise_value(100, .08, .08, .02, "fade5")),
                               float(R.enterprise_value(100, .08, .08, .02)))
        self.assertLess(float(R.enterprise_value(100, .2, .08, .02, "fade5")),
                        float(R.enterprise_value(100, .2, .08, .02)))

    def test_dupont_coherent_average_equity_identity(self):
        a = R.dupont(50, 20, 100, 200, 100, 10)
        self.assertAlmostEqual(float(a["net_margin"] * a["asset_turnover"] * a["leverage"]), .2)
        product = np.prod([a[x] for x in ["net_margin", "asset_turnover", "leverage", "bps_average_equity", "per_coherent"]])
        self.assertAlmostEqual(float(product), 50.)

    def test_simplified_formula_is_distinct_from_current_nopat_EV(self):
        v = R.simplified_seven_factor(.2, .5, 2, 10, .1, .08, .02)
        self.assertAlmostEqual(float(v), 2 * .8 / .06)
        self.assertAlmostEqual(float(R.enterprise_value(2, .1, .08, .02)), float(v) * 1.02)

    def test_fixed_next_nopat_reverse_is_consistent(self):
        self.assertAlmostEqual(float(R.implied_growth_fixed_nopat(1360, 102, .1, .08)), .02)
        self.assertTrue(np.isnan(R.implied_growth_fixed_nopat(1000, 100, .1, .08)))

    def test_split_aligned_price_and_share_count(self):
        before = R.equity_bridge(1360, 200, 10, 50, 10)[1]
        after = R.equity_bridge(1360, 200, 10, 50, 20)[1]
        self.assertAlmostEqual(float(after), float(before) / 2)
        self.assertAlmostEqual(100 / float(before), 50 / float(after))

    def test_scenarios_preserve_missing_and_forecast_ttm_separation(self):
        f = pd.DataFrame(dict(price=[100.], shares=[10.], debt=[200.], nci=[10.],
            non_operating_cash=[50.], roic_operating_tax30=[.1], roic_operating_actual=[.1],
            nopat_tax30=[100.], nopat_actual=[100.], actual_tax=[.3], tv_rf=[.03], tv_beta=[1.],
            eps_forecast=[np.nan], eps_ttm=[10.]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            d = R.scenarios(f)
        self.assertTrue(np.isfinite(d.theoretical_price.iloc[0]))
        self.assertTrue(np.isnan(d.market_per_forecast.iloc[0]))
        self.assertEqual(d.market_per_ttm.iloc[0], 10.)
        self.assertAlmostEqual(d.valuation_gap.iloc[0], d.market_per_ttm.iloc[0] / d.theoretical_per_ttm.iloc[0])
        f["non_operating_cash"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            d = R.scenarios(f)
        self.assertTrue(d.theoretical_price.isna().all())
        self.assertIn("non_operating_cash_unidentified", d.unavailable_reason.iloc[0])


class DatedAnnual(unittest.TestCase):
    def panel(self, **changes):
        f, periods = inputs(**changes)
        return CE.annual_panel(f, periods)

    def test_friday_weekend_and_morning_all_next_session(self):
        cal = pd.bdate_range("2025-06-26", "2025-07-02")
        got = R.next_session(["2025-06-27 09:00", "2025-06-28 15:00", "2025-06-30 08:00"], cal)
        self.assertEqual(got.dt.strftime("%Y-%m-%d").tolist(), ["2025-06-30", "2025-06-30", "2025-07-01"])

    def test_no_cash_deduction_and_no_operating_cash_fabrication(self):
        cal = pd.bdate_range("2024-01-01", "2026-01-01")
        p = R.reviewed_states(self.panel(), cal).iloc[-1]
        self.assertAlmostEqual(p.roic_conservative_tax30, 70 / 700)
        self.assertTrue(np.isnan(p.roic_operating_tax30))
        self.assertTrue(np.isnan(p.non_operating_cash))

    def test_review_known_date_and_comparative_balance(self):
        cal = pd.bdate_range("2024-01-01", "2026-01-01")
        reviews = pd.DataFrame([dict(Code="10000", fiscal_year=y, basis="consolidated", accounting_standard="JP",
            source_doc_id="REVIEW", source_url="https://example.com/annual.pdf", source_submit_date="2025-07-04",
            non_operating_cash=50.) for y in (2024, 2025)])
        s = R.reviewed_states(self.panel(), cal, reviews)
        d = R.attach_annual(pd.DataFrame({"Code": ["10000"] * 2, "Date": ["2025-07-04", "2025-07-07"]}), s)
        self.assertTrue(np.isnan(d.roic_operating_tax30.iloc[0]))
        self.assertAlmostEqual(d.roic_operating_tax30.iloc[1], 70 / 650)
        self.assertEqual(d.filing_avail_date.iloc[1], pd.Timestamp("2025-06-26"))

    def test_financials_stale_and_latest_invalid_do_not_fall_back(self):
        cal = pd.bdate_range("2024-01-01", "2027-01-01")
        states = R.reviewed_states(self.panel(short_term_loans=[100., None], long_term_loans=[100., None]), cal)
        d = R.attach_annual(pd.DataFrame({"Code": ["10000"] * 2, "Date": ["2025-07-01", "2026-12-01"]}), states)
        self.assertTrue(np.isnan(d.roic_conservative_tax30.iloc[0]))
        self.assertFalse(d.annual_fresh.iloc[1])


class CausalFeatures(unittest.TestCase):
    def frame(self, n=300):
        dates = pd.bdate_range("2024-01-01", periods=n)
        price = np.linspace(50, 80, n)
        value = np.linspace(100, 110, n)
        return pd.DataFrame({"Code": "10000", "Date": dates, "valuation_gap": price / value,
                             "price_adjusted": price, "theoretical_price_adjusted": value, "wacc": .08,
                             "roic": .1, "avail_date": dates[0]})

    def test_prefix_invariance_to_future_shock(self):
        f = self.frame()
        prefix = G.time_features(f.iloc[:200])
        f.loc[200:, ["valuation_gap", "price_adjusted", "theoretical_price_adjusted"]] = 1e9
        full = G.time_features(f)
        pd.testing.assert_frame_equal(prefix, full.iloc[:200])

    def test_minimum_observations_and_zero_variance(self):
        f = self.frame()
        f["valuation_gap"] = 1.
        g = G.time_features(f)
        self.assertTrue(g.valuation_gap_zscore_252.isna().all())
        self.assertTrue(g.valuation_gap_pct_252.iloc[:125].isna().all())
        self.assertAlmostEqual(g.valuation_gap_pct_252.iloc[125], 63.5 / 126)

    def test_log_identity_and_missing_not_bridged(self):
        f = self.frame()
        g = G.time_features(f)
        np.testing.assert_allclose(g.valuation_gap_log_change_20d.iloc[20:],
            (g.valuation_price_log_change_20d - g.valuation_value_log_change_20d).iloc[20:], atol=1e-14)
        f.loc[160, "valuation_gap"] = np.nan
        g = G.time_features(f)
        self.assertTrue(np.isnan(g.valuation_gap_change_20d.iloc[180]))

    def test_research_presets_do_not_modify_production(self):
        before = (F.DEFAULT_PRESET, F.columns(F.DEFAULT_PRESET))
        arms = G.column_sets()
        self.assertEqual(arms["A"], before[1])
        self.assertTrue(set(arms["A"]) < set(arms["B"]) < set(arms["C"]) < set(arms["D"]))
        self.assertEqual(before, (F.DEFAULT_PRESET, F.columns(F.DEFAULT_PRESET)))

    def test_shuffle_preserves_dates_targets_and_missingness(self):
        f = self.frame(30).assign(target=np.arange(30), Date=pd.Timestamp("2025-01-01"))
        f.loc[0, "valuation_gap"] = np.nan
        g = G.shuffle_within_date(f, ["valuation_gap"])
        pd.testing.assert_series_equal(f.Date, g.Date)
        pd.testing.assert_series_equal(f.target, g.target)
        np.testing.assert_array_equal(f.valuation_gap.isna(), g.valuation_gap.isna())


if __name__ == "__main__":
    unittest.main()
