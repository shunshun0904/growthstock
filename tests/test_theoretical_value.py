"""Financial identities and historical information boundaries, independent examples."""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research"))
import theoretical_value as V


class TheoreticalValue(unittest.TestCase):
    def test_negative_mof_yield_is_preserved_and_not_same_day(self):
        y = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "10Y": [-.2, 4.]})
        r = V.riskfree(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-01"]), y)
        self.assertAlmostEqual(r.tv_rf.iloc[0], -.001999)
        self.assertAlmostEqual(r.tv_rf.iloc[1], .0404)
        self.assertTrue(pd.isna(r.tv_rf.iloc[2]))

    def test_zero_residual_profit_equals_book_value(self):
        for h in V.HORIZONS:
            for p in V.PAYOUTS:
                self.assertAlmostEqual(float(V.residual_value(100, 8, .08, h, p)), 100.)

    def test_ri_equals_independently_discounted_dividends_and_terminal_book(self):
        # h=2: year 1 ROE=15%, year 2 ROE=10%; payout=50%.
        # B1=107.5, D1=7.5, D2=5.375, B2=112.875.
        expected = 7.5/1.1 + (5.375+112.875)/1.1**2
        self.assertAlmostEqual(float(V.residual_value(100, 20, .10, 2, .5)), expected)

    def test_scenario_statistics_and_insufficient_grid(self):
        s, n = V.summarize([[1., 2., 3., 4., 5.], [1., 2., np.nan, np.nan, 5.]])
        np.testing.assert_allclose(s[0], [1., 5., 3., 3., np.sqrt(2)])
        self.assertTrue(np.isnan(s[1]).all())
        self.assertEqual(n.tolist(), [1., .6])

    def test_capm_ddm_units_and_split_invariance(self):
        # CAPM .02 + beta1 * 5% = .07; g=0: 10/.07.
        d = pd.DataFrame({"tv_bps": [100.], "tv_eps": [10.], "tv_div0": [10.],
                          "tv_payout": [1.], "tv_beta": [1.], "tv_rf": [.02], "close_raw": [200.]})
        out = V.valuations(d)
        self.assertAlmostEqual(out.tv_ddm_min.iloc[0], 10/.09)
        self.assertAlmostEqual(out.tv_ddm_max.iloc[0], 10*1.02/(.05-.02))
        split = d.copy()
        split[["tv_bps", "tv_eps", "tv_div0", "close_raw"]] /= 2
        adjusted = V.valuations(split)
        np.testing.assert_allclose(out[V.VALUES], adjusted[V.VALUES])
        for s in V.STATS:
            self.assertAlmostEqual(out['tv_ri_'+s].iloc[0]/2, adjusted['tv_ri_'+s].iloc[0])

    def test_no_dividend_and_negative_book_are_unavailable(self):
        d = pd.DataFrame({"tv_bps": [-1.], "tv_eps": [10.], "tv_div0": [0.],
                          "tv_payout": [0.], "tv_beta": [1.], "tv_rf": [.02], "close_raw": [200.]})
        self.assertTrue(V.valuations(d)[V.VALUES].isna().all().all())

    def test_beta_identity_missing_sessions_and_future_invariance(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        m = pd.DataFrame({"Date": dates, "topix": 100*np.cumprod(1+np.sin(np.arange(30))*.01)})
        p = m.rename(columns={"topix": "close"}).assign(Code="10000")
        y = pd.DataFrame({"Date": dates-pd.Timedelta(days=1), "10Y": 0.})
        keys = pd.DataFrame({"Code": ["10000"], "Date": [dates[20]]})
        a = V.beta(keys, p, m, y, window=20, min_obs=10)
        self.assertAlmostEqual(a.tv_beta_raw.iloc[0], 1.)
        future = p.copy(); future.loc[future.Date > dates[20], "close"] *= 100
        pd.testing.assert_frame_equal(a, V.beta(keys, future, m, y, window=20, min_obs=10))
        missing = p[p.Date.ne(dates[15])]
        b = V.beta(keys, missing, m, y, window=20, min_obs=10)
        self.assertEqual(b.tv_beta_n.iloc[0], a.tv_beta_n.iloc[0]-2)
        self.assertAlmostEqual(b.tv_beta_raw.iloc[0], 1.)

    def test_financial_versions_and_split_adjusted_cash_amounts(self):
        fins = pd.DataFrame({"Code": ["10000"]*3, "DocType": ["FYFinancialStatements_Consolidated_JP"]*3,
            "CurPerType": ["FY"]*3, "CurFYSt": ["2019-01-01"]*3,
            "CurPerEn": ["2019-12-31"]*3, "DiscDate": ["2020-01-02", "2020-01-04", "2020-01-10"],
            "DiscTime": ["16:00"]*3, "NP": [200., 400., 99999.], "ShEq": [1000.]*3,
            "ShOutFY": [100.]*3, "TrShFY": [0.]*3, "DivTotalAnn": [100.]*3})
        dates = pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
        prices = pd.DataFrame({"Code": ["10000"]*5, "Date": dates, "close_raw": [100.,100.,50.,50.,50.], "close": [50.]*5})
        keys = prices.iloc[2:].copy()
        a = V.financial_inputs(keys, fins, prices)
        self.assertEqual(a.tv_eps.tolist(), [1., 1., 2.])  # correction unavailable on its disclosure day
        self.assertEqual(a.tv_bps.tolist(), [5., 5., 5.])
        self.assertEqual(a.tv_div0.tolist(), [.5, .5, .5])
        b = V.financial_inputs(keys, fins.iloc[:2], prices)
        pd.testing.assert_frame_equal(a, b)  # appending future correction has no effect
        future_split = prices.copy(); future_split['close'] /= 3
        pd.testing.assert_frame_equal(a, V.financial_inputs(keys, fins, future_split))


if __name__ == "__main__":
    unittest.main()
