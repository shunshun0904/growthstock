"""Price exclusion and common-path EVA/DCF identities, including real binding."""
from dataclasses import fields, replace
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import fundamental_value as F
import fundamental_series as S
import valuation_gap as G
import build_roic_wacc_series as B
import roic_wacc_valuation as R
import capital_efficiency as CE
import valuation_methods as VM
from test_capital_efficiency import inputs as annual_inputs


class FundamentalValue(unittest.TestCase):
    def inputs(self):
        return F.Fundamentals(1000., .1, 100., 1000., 0., 0., 0., 10., .03, .30)

    def test_closed_input_refuses_market_fields_and_mappings(self):
        self.assertFalse({"price", "beta", "market_cap", "per", "pbr", "implied_growth"} & {f.name for f in fields(F.Fundamentals)})
        with self.assertRaises(TypeError):
            F.Fundamentals(price=100.)
        with self.assertRaises(TypeError):
            F.evaluate({"market_price": 100}, .01)
        with self.assertRaises(ValueError):
            S.calculate(pd.DataFrame({"tv_beta": [1.]}))

    def test_book_discount_is_independent_of_target_market_weights(self):
        self.assertAlmostEqual(float(F.book_wacc(1000, 200, .01, .3)), 63.5 / 1200)
        self.assertTrue(np.isnan(F.book_wacc(1000, np.nan, .01, .3)))

    def test_steady_EVA_equals_independent_infinite_DCF(self):
        value = F.evaluate(self.inputs(), .02, fade=False)
        self.assertAlmostEqual(float(value["enterprise_value"]), 1000 * (.1 - .02) / (.08 - .02))
        self.assertAlmostEqual(float(value["fundamental_value_per_share"]), 133.3333333333333)
        self.assertAlmostEqual(float(value["eva_dcf_relative_error"]), 0.)

    def test_fade_EVA_equals_DCF_over_sensitivity_grid(self):
        for r in (.04, .08, .15, .30):
            for w in (.04, .08, .12):
                for g in (0., .01, .02, .03):
                    value = F.economic_profit(1000, r, w, g)
                    if np.isfinite(value["enterprise_value"]):
                        self.assertAlmostEqual(float(value["enterprise_value"]), float(value["dcf_enterprise_value"]), places=9)
        equal = F.economic_profit(1000, .08, .08, .02)
        self.assertAlmostEqual(float(equal["enterprise_value"]), 1000.)

    def test_existing_DCF_matches_when_its_assumptions_are_aligned(self):
        # Legacy DCF's market-dependent discount rate is FIXED to .08 for
        # this compatibility test only; it is never an EVA valuation input.
        _, old_ev, _, old_wacc = VM.dcf_value(1000., 100 / .7, 0., 0., 0., 1000., 0., 0., 0., .03, 1.,
                                             growth=0., premium=.05, terminal_growth=0., tax=.30)
        new = F.evaluate(self.inputs(), 0., fade=False)
        self.assertAlmostEqual(float(old_wacc), .08)
        self.assertAlmostEqual(float(old_ev), float(new["enterprise_value"]), places=9)

    def test_missing_balances_and_nonpositive_equity(self):
        for key in ("debt_book", "non_controlling_interests", "non_operating_cash", "shares"):
            self.assertTrue(np.isnan(F.evaluate(replace(self.inputs(), **{key: np.nan}), .01)["fundamental_value_per_share"]))
        self.assertTrue(np.isnan(F.evaluate(replace(self.inputs(), non_controlling_interests=1e9), .01)["fundamental_value_per_share"]))

    def test_full_reinvestment_can_have_zero_EV_and_positive_cash_backed_equity(self):
        inp = replace(self.inputs(), historical_roic=.02, historical_nopat=20., total_book_equity=2000., non_operating_cash=1000.)
        value = F.evaluate(inp, .02, fade=False)
        self.assertEqual(float(value["enterprise_value"]), 0.)
        self.assertAlmostEqual(float(value["fundamental_value_per_share"]), 100.)

    def test_target_price_only_changes_terminal_gap(self):
        value = F.evaluate(self.inputs(), .01)["fundamental_value_per_share"]
        a, b = G.compare_with_market(value, 100.), G.compare_with_market(value, 200.)
        self.assertAlmostEqual(float(b), 2 * float(a))
        self.assertAlmostEqual(float(F.evaluate(self.inputs(), .01)["fundamental_value_per_share"]), float(value))

    def test_business_growth_is_observed_only_bounded_and_not_implied(self):
        self.assertAlmostEqual(float(F.business_growth(102, 100, 104, 100)), .03)
        self.assertTrue(np.isnan(F.business_growth(90, 100, 104, 100)))
        self.assertTrue(np.isnan(F.business_growth(102, np.nan, 104, 100)))

    def test_rights_and_unclassified_price_factors_never_change_share_units(self):
        units, unsafe = F.corporate_action_units([1., .5, .75, .2, 1.], [None, '1', '3', None, None])
        np.testing.assert_allclose(units, [1., .5, .5, .5, .5])
        np.testing.assert_array_equal(unsafe, [0, 0, 1, 2, 2])


class PricePerturbationIntegration(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2024-03-01", "2025-07-15")
        prices = pd.DataFrame({"Code": "10000", "Date": dates, "close_raw": 100., "close": 100., "action_factor": 1.})
        market = pd.DataFrame({"Date": dates, "topix": 100 * np.cumprod(1 + .001 * np.sin(np.arange(len(dates))))})
        rates = pd.DataFrame({"Date": dates - pd.Timedelta(days=1), "10Y": 2.})
        f, periods = annual_inputs()
        panel = CE.annual_panel(f, periods)
        reviews = pd.DataFrame([dict(Code="10000", fiscal_year=y, basis="consolidated", accounting_standard="JP",
            source_doc_id="REVIEW", source_url="https://example.com/review.pdf", source_submit_date="2025-06-25",
            non_operating_cash=50., nci=0.) for y in (2024, 2025)])
        annual = R.reviewed_states(panel, dates, reviews)
        fins = pd.DataFrame([dict(Code="10000", DocType="FYFinancialStatements_Consolidated_JP", CurPerType="FY",
            CurFYSt=f"{y - 1}-04-01", CurFYEn=f"{y}-03-31", CurPerEn=f"{y}-03-31", DiscDate=f"{y}-05-02",
            DiscTime="16:00", DiscNo=str(y), NP=70., Sales=1000., TA=1000., ShEq=500., ShOutFY=100., TrShFY=0.,
            DivTotalAnn=0., FNP=np.nan, NxFNp=75.) for y in (2024, 2025)])
        annual = B.annual_dupont_inputs(annual, fins)
        return prices, market, rates, annual, fins

    def test_nonvacuous_price_beta_and_market_cap_perturbation(self):
        prices, market, rates, annual, fins = self.fixture()
        events = B.financial_events(fins, prices, market.Date)
        a = B.make_series("10000", prices, market, rates, annual, events)
        changed = prices.copy()
        changed["close_raw"] = 10000 + np.arange(len(changed)) * 123.
        changed["close"] = 50000 + np.sin(np.arange(len(changed))) * 10000
        events2 = B.financial_events(fins, changed, market.Date)
        b = B.make_series("10000", changed, market, rates, annual, events2)
        # This MUST contain positive computed values, not all-NaN equality.
        self.assertGreater(a.fundamental_value_per_share.notna().sum(), 0)
        np.testing.assert_allclose(a.fundamental_value_per_share, b.fundamental_value_per_share, equal_nan=True)
        np.testing.assert_allclose(a.theoretical_price_adjusted, b.theoretical_price_adjusted, equal_nan=True)
        np.testing.assert_allclose(a.independent_shares, b.independent_shares, equal_nan=True)
        np.testing.assert_allclose(a.wacc, b.wacc, equal_nan=True)
        known = a.tv_beta.notna() & b.tv_beta.notna()
        self.assertFalse(np.allclose(a.loc[known, "tv_beta"], b.loc[known, "tv_beta"]))
        self.assertFalse(np.allclose(a.valuation_gap.dropna(), b.valuation_gap.dropna()))

    def test_corporate_action_not_price_ratio_adjusts_primary_shares(self):
        prices, market, rates, annual, fins = self.fixture()
        split = pd.Timestamp("2025-07-01")
        prices.loc[prices.Date.eq(split), "action_factor"] = .5
        prices["action_type"] = None
        prices.loc[prices.Date.eq(split), "action_type"] = '1'
        # Deliberately leave market prices unchanged: share adjustment must
        # still follow the independent corporate-action event.
        events = B.financial_events(fins, prices, market.Date)
        a = B.make_series("10000", prices, market, rates, annual, events)
        before = a[a.Date.eq("2025-06-30")].iloc[0]
        after = a[a.Date.eq(split)].iloc[0]
        self.assertAlmostEqual(after.independent_shares, before.independent_shares * 2)
        self.assertAlmostEqual(after.fundamental_value_per_share, before.fundamental_value_per_share / 2)

    def test_unclassified_or_rights_event_invalidates_shares_until_new_snapshot(self):
        for kind in (None, '3'):
            prices, market, rates, annual, fins = self.fixture()
            event = pd.Timestamp('2025-07-01')
            prices['action_type'] = None
            prices.loc[prices.Date.eq(event), 'action_factor'] = .75
            prices.loc[prices.Date.eq(event), 'action_type'] = kind
            events = B.financial_events(fins, prices, market.Date)
            d = B.make_series('10000', prices, market, rates, annual, events)
            self.assertGreater(d.fundamental_value_per_share.notna().sum(), 0)
            self.assertTrue(d.loc[d.Date.ge(event), 'independent_shares'].isna().all())
            self.assertTrue(d.loc[d.Date.ge(event), 'fundamental_value_per_share'].isna().all())

    def test_post_rights_share_snapshot_restores_only_after_disclosure(self):
        prices, market, rates, annual, fins = self.fixture()
        prices['action_type'] = None
        prices.loc[prices.Date.eq('2025-07-01'), ['action_factor', 'action_type']] = [.75, '3']
        correction = fins.iloc[-1].copy()
        correction['DocType'] = '1QFinancialStatements_Consolidated_JP'
        correction['CurPerType'] = '1Q'
        correction['CurFYSt'], correction['CurFYEn'] = '2025-04-01', '2026-03-31'
        correction['CurPerEn'], correction['DiscDate'], correction['DiscNo'] = '2025-07-04', '2025-07-07', 'NEW'
        correction['ShOutFY'], correction['NP'], correction['FNP'] = 150., 17.5, 75.
        fins = pd.concat([fins, correction.to_frame().T], ignore_index=True)
        events = B.financial_events(fins, prices, market.Date)
        d = B.make_series('10000', prices, market, rates, annual, events)
        self.assertTrue(d.loc[d.Date.eq('2025-07-07'), 'independent_shares'].isna().all())
        self.assertEqual(d.loc[d.Date.eq('2025-07-08'), 'independent_shares'].iloc[0], 150.)
        self.assertGreater(d.loc[d.Date.eq('2025-07-08'), 'fundamental_value_per_share'].iloc[0], 0.)


if __name__ == "__main__":
    unittest.main()
