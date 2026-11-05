"""Independent financial-path cases, no market data or model fitting required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exit_upside as E


class ExitPaths(unittest.TestCase):
    def build(self, future, hold=None):
        n = len(future)
        cal = pd.bdate_range("2026-01-01", periods=n + 2)
        bars = pd.DataFrame([[100, 500, 1, 400]] + future,
                            columns=["AdjO", "AdjH", "AdjL", "AdjC"])
        bars["Date"] = cal[:n+1]
        bars["Code"] = "12340"
        keys = pd.DataFrame({"Code": ["12340"], "Date": [cal[0]]})
        return E.next_open_paths(keys, bars, cal, hold or n), bars, cal

    def test_next_open_and_buy_day_count(self):
        p, _, cal = self.build([[200, 210, 195, 206], [206, 216, 202, 212]])
        self.assertEqual(p.rows.at[0, "entry_date"], cal[1])
        self.assertAlmostEqual(p.rows.at[0, "mfe"], .08)
        self.assertEqual(p.rows.at[0, "hit5_day"], 1)
        self.assertEqual(p.rows.at[0, "hit8_day"], 2)
        self.assertAlmostEqual(p.rows.at[0, "ret_close20"], .06)

    def test_missing_next_open_is_not_later_entry(self):
        p, _, _ = self.build([[np.nan, np.nan, np.nan, np.nan], [100, 110, 95, 105]])
        self.assertEqual(p.rows.at[0, "status"], "no_next_open")
        self.assertTrue(np.isnan(p.rows.at[0, "hit5"]))

    def test_early_winner_with_short_followup_is_censored(self):
        p, _, _ = self.build([[100, 120, 95, 115]], hold=2)
        self.assertEqual(p.rows.at[0, "status"], "not_mature")
        self.assertEqual(len(E.simulate(p, [0], target=.05, timeout=2)), 0)

    def test_same_day_ambiguity_and_open_precedence(self):
        p, _, _ = self.build([[100, 110, 90, 102]])
        lo = E.simulate(p, [0], target=.08, stop=.05, timeout=1)
        hi = E.simulate(p, [0], target=.08, stop=.05, timeout=1, ambiguity="target_first")
        self.assertAlmostEqual(lo.gross.iloc[0], -.05)
        self.assertAlmostEqual(hi.gross.iloc[0], .08)
        self.assertTrue(lo.ambiguous.iloc[0])
        p, _, _ = self.build([[100, 102, 99, 101], [90, 112, 88, 110]])
        r = E.simulate(p, [0], target=.08, stop=.05, timeout=2)
        self.assertAlmostEqual(r.gross.iloc[0], -.10)
        self.assertEqual(r.reason.iloc[0], "stop_gap")
        self.assertFalse(r.ambiguous.iloc[0])

    def test_trail_cannot_use_same_day_high_or_close(self):
        p, _, _ = self.build([[100, 110, 99, 108], [108, 109, 106, 107]])
        r = E.simulate(p, [0], trail=.05, timeout=2)
        self.assertEqual(r.exit_day.iloc[0], 2)
        self.assertEqual(r.reason.iloc[0], "timeout")
        p, _, _ = self.build([[100, 110, 99, 108], [100, 109, 99, 105]])
        r = E.simulate(p, [0], trail=.05, timeout=2)
        self.assertEqual(r.reason.iloc[0], "stop_gap")
        self.assertAlmostEqual(r.gross.iloc[0], 0)

    def test_original_label_maturity_can_differ_from_market_horizon(self):
        p, b, cal = self.build([[100, 104, 98, 102], [100, 108, 99, 106]])
        k = p.rows[["Code", "Date"]]
        self.assertEqual(E.label_maturity(k, b, hold=2)[0], cal[2])
        b = b[b.Date.ne(cal[1])]
        self.assertTrue(pd.isna(E.label_maturity(k, b, hold=2)[0]))

    def test_percentiles_do_not_refit_on_future_scores(self):
        np.testing.assert_allclose(E.historical_percentile([.1, .2, .9], [.1, .2, .3, .4]), [0,25,100])

    def test_pool_percentile_includes_older_candidates(self):
        date = pd.Timestamp("2026-01-05")
        f = pd.DataFrame({"Date": [date]*3, "score": [1., 2., 3.],
                          "cand_date": [date - pd.Timedelta(days=1), date, date]})
        v = E.within_day_percentile(f)
        self.assertEqual(v[f.cand_date.eq(date)].iloc[0], 50)

    def test_invalid_ohlc_does_not_generate_profit(self):
        p, _, _ = self.build([[100, 90, 95, 101]])
        self.assertEqual(p.rows.at[0, "status"], "invalid_ohlc")

    def test_profit_open_precedes_later_stop_and_uses_limit_fill(self):
        p, _, _ = self.build([[100, 103, 99, 102], [110, 112, 90, 95]])
        r = E.simulate(p, [0], target=.05, stop=.05, timeout=2)
        self.assertEqual(r.reason.iloc[0], "target_gap")
        self.assertAlmostEqual(r.gross.iloc[0], .05)
        self.assertFalse(r.ambiguous.iloc[0])

    def test_top_two_selected_before_future_completion_check(self):
        d = pd.DataFrame({"Date": [pd.Timestamp("2026-01-05")]*3, "Code": ["1","2","3"],
                          "base_min":[99,98,97], "score_lgbm":[.9,.8,.7],
                          "status":["incomplete_ohlc","complete","complete"]})
        np.testing.assert_array_equal(E.selected_ids(d,{"base":95}),[0,1])

    def test_missing_classifier_cannot_pass_all_model_gate(self):
        date=pd.Timestamp("2026-01-05")
        frames={a:pd.DataFrame({"Date":[date],"Code":["1"],"score":[.5]}) for a in E.CLASSIFIERS}
        frames['xgb']=frames['xgb'].iloc[:0]
        frames['ltr']=pd.DataFrame({"Date":[date],"Code":["1"],"score":[0.],"fold":[1]})
        frames['ltr5']=frames['ltr'].assign(cand_date=date)
        with self.assertRaises(ValueError):
            E.combine_scores(frames,date)

    def test_omitted_future_label_invalidates_rank_but_not_classifier_gate(self):
        date=pd.Timestamp('2026-01-05')
        d=pd.DataFrame({'Date':[date], 'Code':['1'],'base_min':[98.], 'score_lgbm':[.9],
                        'pct_ltr':[90.], 'pct_ltr5':[95.]})
        cov=pd.DataFrame({'daily_complete':[True], 'rolling_complete':[False]},index=[date])
        out=E.require_full_ranking_pools(d,cov)
        self.assertTrue(E.entry_mask(out,{'base':95}).iloc[0])
        self.assertFalse(E.entry_mask(out,{'base':95,'rolling':80}).iloc[0])
        self.assertTrue(np.isnan(out.pct_ltr5.iloc[0]))
        self.assertEqual(out.pct_ltr.iloc[0],90.)


if __name__ == "__main__":
    unittest.main()
