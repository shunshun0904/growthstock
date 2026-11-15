"""Cash classification is optional only in explicitly different models."""
from dataclasses import fields, replace
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import fundamental_value as F
import fundamental_series as S
import theoretical_value as TV
import test_fundamental_value as T
import build_roic_wacc_series as B


class UnsegregatedValue(unittest.TestCase):
    def inputs(self):
        return F.TotalCapitalFundamentals(1000., 200., .1, 120., 0., 10., .03, .3)

    def test_total_capital_model_has_no_cash_or_price_argument(self):
        forbidden = {'non_operating_cash', 'price', 'beta', 'market_cap', 'per', 'pbr'}
        self.assertFalse(forbidden & {f.name for f in fields(F.TotalCapitalFundamentals)})
        with self.assertRaises(TypeError):
            F.evaluate_total({'price':100}, .01)

    def test_total_capital_EVA_and_DCF_bridge_are_consistent(self):
        inp = self.inputs()
        result = F.evaluate_total(inp, 0., fade=False)
        w = (.08*1000 + .045*.7*200) / 1200
        expected_firm = 120. / w
        self.assertAlmostEqual(float(result['enterprise_value']), expected_firm)
        self.assertAlmostEqual(float(result['fundamental_value_per_share']), (expected_firm-200)/10)
        self.assertLess(float(result['eva_dcf_relative_error']), 1e-10)
        for name in ('debt_book', 'non_controlling_interests', 'shares'):
            self.assertTrue(np.isnan(F.evaluate_total(replace(inp, **{name:np.nan}), 0.)['fundamental_value_per_share']))

    def test_RIM_equals_dividend_DCF_and_existing_RIM_on_same_path(self):
        for horizon in (5,10):
            for payout in (0., .3, 1.):
                inp = F.RimFundamentals(1000., 120., 10., .03, payout)
                result = F.evaluate_rim(inp, .05, horizon=horizon)
                expected = TV.residual_value(1000., 120., .08, horizon, payout)
                self.assertAlmostEqual(float(result['equity_value']), float(expected))
                self.assertAlmostEqual(float(result['equity_value']), float(result['dividend_dcf_value']), places=9)
                self.assertLess(float(result['rim_dcf_relative_error']), 1e-10)

    def test_RIM_does_not_require_cash_debt_or_NCI(self):
        inp = F.RimFundamentals(1000., 80., 10., .03, .3)
        self.assertAlmostEqual(float(F.evaluate_rim(inp, .05)['fundamental_value_per_share']), 100.)
        for name in ('parent_equity', 'annual_parent_profit', 'shares', 'payout'):
            self.assertTrue(np.isnan(F.evaluate_rim(replace(inp, **{name:np.nan}), .05)['fundamental_value_per_share']))
        self.assertTrue(np.isnan(F.evaluate_rim(replace(inp, payout=1.1), .05)['fundamental_value_per_share']))


class UnsegregatedIntegration(unittest.TestCase):
    def test_both_models_compute_without_cash_and_ignore_price_perturbations(self):
        prices, market, rates, annual, fins = T.PricePerturbationIntegration().fixture()
        annual[['non_operating_cash','previous_non_operating_cash','operating_capital',
                'previous_operating_capital','roic_operating_tax30','roic_operating_actual']] = np.nan
        a = B.make_series('10000', prices, market, rates, annual, B.financial_events(fins, prices, market.Date))
        changed = prices.copy()
        changed['close_raw'] = 10000 + np.arange(len(changed))*123.
        changed['close'] = 50000 + np.sin(np.arange(len(changed)))*10000
        b = B.make_series('10000', changed, market, rates, annual, B.financial_events(fins, changed, market.Date))
        for name in ('fundamental_value_per_share','rim_value_per_share'):
            self.assertGreater(a[name].notna().sum(), 0)
            np.testing.assert_allclose(a[name], b[name], equal_nan=True)
        self.assertTrue(a.strict_fundamental_value_per_share.isna().all())
        self.assertTrue(a.non_operating_cash.isna().all())
        self.assertTrue(a.valuation_gap.notna().any())
        self.assertTrue(a.rim_valuation_gap.notna().any())

    def test_missing_payout_is_not_zero_and_income_is_financial_only(self):
        prices, market, rates, annual, fins = T.PricePerturbationIntegration().fixture()
        fins['DivTotalAnn'] = np.nan
        events = B.financial_events(fins, prices, market.Date)
        frame = B.make_series('10000', prices, market, rates, annual, events)
        self.assertTrue(frame.rim_value_per_share.isna().all())
        self.assertTrue(events.rim_parent_profit.eq(70.).all())

    def test_TTM_parent_profit_uses_same_scope_and_next_session_only(self):
        prices, market, rates, annual, fins = T.PricePerturbationIntegration().fixture()
        extra = []
        for year,profit,equity in ((2024,10.,510.),(2025,20.,550.)):
            row = fins.iloc[-1].copy()
            row[['CurFYSt','CurFYEn','CurPerEn','DiscDate']] = [f'{year}-04-01',f'{year+1}-03-31',f'{year}-06-30',f'{year}-07-08']
            row[['CurPerType','DocType','DiscNo','NP','ShEq']] = ['1Q','1QFinancialStatements_Consolidated_JP',f'Q{year}',profit,equity]
            extra.append(row)
        data = pd.concat([fins,pd.DataFrame(extra)],ignore_index=True)
        events = B.financial_events(data,prices,market.Date)
        self.assertEqual(events.loc[events.financial_event_date.eq('2025-07-09'),'rim_parent_profit'].iloc[0],80.)
        frame = B.make_series('10000',prices,market,rates,annual,events)
        self.assertEqual(frame.loc[frame.Date.eq('2025-07-08'),'rim_parent_profit'].iloc[0],70.)
        self.assertEqual(frame.loc[frame.Date.eq('2025-07-09'),'rim_parent_equity'].iloc[0],550.)


if __name__ == '__main__':
    unittest.main()
