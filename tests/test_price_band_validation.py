"""Independent numerical and leakage checks for the approved experiment."""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'research'))
from price_band_validation import (
    estimate_means, point_in_time_shares, rmspe, select_coefficient,
    split_dates, paired_block_interval, make_window,
)
from run_price_band_validation import ordinary_universe, make_records, build_features


class MeanEstimationTests(unittest.TestCase):
    def test_survival_uses_only_turnover_after_each_entry_day(self):
        # Last-day turnover log(2) halves only the old position at k=1.
        result = estimate_means([[100, 200]], [[100, 100]], [[99, np.log(2)]], [0, 1])
        np.testing.assert_allclose(result, [[150, 500 / 3]])

    def test_rmspe_is_equal_per_record(self):
        self.assertAlmostEqual(rmspe([100, 200], [110, 160]), np.sqrt(.025) * 100)

    def test_missing_target_is_not_implicitly_dropped(self):
        with self.assertRaises(ValueError):
            rmspe([100, np.nan], [110, 160])

    def test_split_keeps_whole_dates_together(self):
        train, test = split_dates(['2026-10-%02d' % i for i in range(1, 11)])
        self.assertEqual(len(train), 7)
        self.assertEqual(test, ['2026-10-08', '2026-10-09', '2026-10-10'])

    def test_coefficient_is_selected_from_training_labels_only_and_ties_choose_smallest(self):
        # Inputs to this function contain training records only.
        self.assertEqual(select_coefficient([100, 200], [[101, 100], [199, 200]], [0, 1])[0], 1)
        self.assertEqual(select_coefficient([100], [[100, 100]], [0, 1])[0], 0)

    def test_identical_methods_have_zero_bootstrap_difference(self):
        daily = pd.DataFrame({'Date': pd.date_range('2026-10-01', periods=10),
                              'n': np.arange(10) + 1, 'baseline_sse': np.ones(10),
                              'candidate_sse': np.ones(10)})
        result = paired_block_interval(daily, block_length=5, replicates=1000, seed=20261114)
        self.assertEqual(result['lower'], 0)
        self.assertEqual(result['upper'], 0)


class PointInTimeAndSplitTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.date_range('2026-09-01', periods=4)
        self.filings = pd.DataFrame([
            {'DiscDate': '2026-09-01', 'CurPerEn': '2026-09-01', 'DiscNo': '1',
             'ShOutFY': 1100, 'TrShFY': 100},
            {'DiscDate': '2026-09-03', 'CurPerEn': '2026-09-02', 'DiscNo': '2',
             'ShOutFY': 2200, 'TrShFY': np.nan},
            {'DiscDate': '2026-09-04', 'CurPerEn': '2026-09-04', 'DiscNo': '3',
             'ShOutFY': 4400, 'TrShFY': 400},
        ])

    def test_pair_is_carried_without_future_disclosure_or_zero_treasury_fill(self):
        result = point_in_time_shares(self.days, [1, 1, 1, 1], self.filings)
        np.testing.assert_allclose(result.shares, [1000, 1000, 1000, 4000])
        self.assertTrue((result.disclosure_date <= result.index).all())

    def test_split_adjusts_share_denominator_from_measurement_date(self):
        result = point_in_time_shares(self.days, [1, .5, 1, 1], self.filings.iloc[:1])
        np.testing.assert_allclose(result.shares, [1000, 2000, 2000, 2000])

    def test_same_day_after_close_disclosure_is_available(self):
        f = self.filings.iloc[:1].copy()
        f['DiscTime'] = '23:58:00'
        result = point_in_time_shares(self.days, [1, 1, 1, 1], f)
        self.assertEqual(result.shares.iloc[0], 1000)

    def test_window_adjusts_prices_volumes_and_turnover_consistently(self):
        b = pd.DataFrame({'Date': self.days[:2], 'Vo': [100, 200], 'Va': [10000, 10000],
                          'AdjFactor': [1, .5], 'shares': [1000, 2000]})
        window, reason = make_window(b, lookback=2)
        self.assertIsNone(reason)
        np.testing.assert_allclose(window['price'], [50, 50])
        np.testing.assert_allclose(window['volume'], [200, 200])
        np.testing.assert_allclose(window['turnover'], [.1, .1])
        np.testing.assert_allclose(estimate_means([window['price']], [window['volume']],
                                                 [window['turnover']], [0, 3]), [[50, 50]])

    def test_positive_volume_without_value_has_no_close_price_fallback(self):
        b = pd.DataFrame({'Vo': [100], 'Va': [np.nan], 'C': [123], 'AdjFactor': [1], 'shares': [1000]})
        self.assertEqual(make_window(b, lookback=1)[1], 'missing_or_invalid_vwap')

    def test_known_zero_volume_differs_from_missing_bar(self):
        b = pd.DataFrame({'Vo': [0, 100], 'Va': [np.nan, 20000], 'AdjFactor': [1, 1], 'shares': [1000, 1000]})
        w, reason = make_window(b, lookback=2)
        self.assertIsNone(reason)
        self.assertEqual(estimate_means([w['price']], [w['volume']], [w['turnover']], [1])[0, 0], 200)
        b.loc[0, 'Vo'] = np.nan
        self.assertEqual(make_window(b, lookback=2)[1], 'missing_or_invalid_volume')


class DatasetContractTests(unittest.TestCase):
    def test_historical_product_type_and_ordinary_issue_suffix_both_matter(self):
        date = pd.Timestamp('2026-09-25')
        codes = ['13010', '13050', '25935', '66990', '99990']
        margin = pd.DataFrame({'Code': codes, 'Date': [date] * 5})
        master = pd.DataFrame({'Code': codes[:4], 'Date': [date] * 4,
                               'ProdCat': ['011', '014', '011', '021'],
                               'CoName': ['ordinary', 'ETF', 'preferred', 'foreign ordinary'],
                               'Mkt': ['0111'] * 4})
        selected, counts = ordinary_universe(margin, master)
        self.assertEqual(selected.Code.tolist(), ['13010', '66990'])
        self.assertEqual(counts.set_index('universe_status').loc['unknown_historical_product_type', 'issue_dates'], 1)

    def test_target_outside_window_is_kept_and_zero_position_mean_is_undefined(self):
        d = pd.Timestamp('2026-09-25')
        universe = pd.DataFrame([{'Code': '13010', 'Date': d, 'PubDate': '2026-09-28', 'CoName': 'a',
                                  'LongVol': 10, 'LongVal': 10000, 'ShrtVol': 0, 'ShrtVal': 0}])
        features = pd.DataFrame([{'Code': '13010', 'Date': d, 'feature_status': 'available',
                                  'feature_id': 0, 'price_min': 100, 'price_max': 200}])
        records = make_records(universe, features)
        self.assertTrue(records.loc[records.side == 'long', 'included'].item())
        self.assertEqual(records.loc[records.side == 'long', 'actual_mean'].item(), 1000)
        self.assertFalse(records.loc[records.side == 'short', 'included'].item())
        self.assertTrue(np.isnan(records.loc[records.side == 'short', 'actual_mean'].item()))

    def test_feature_builder_rejects_target_columns(self):
        with self.assertRaisesRegex(ValueError, 'code/date only'):
            build_features(pd.DataFrame(columns=['Code', 'Date', 'LongVal']), None, None, None, 60)

    def test_missing_exchange_day_is_not_replaced_by_an_older_price_bar(self):
        dates = pd.date_range('2026-09-01', periods=3)
        requests = pd.DataFrame({'Code': ['13010'], 'Date': [dates[2]]})
        bars = pd.DataFrame({'Code': ['13010'] * 2, 'Date': [dates[0], dates[2]],
                             'Vo': [100, 100], 'Va': [10000, 10000], 'AdjFactor': [1, 1]})
        f = pd.DataFrame([{'Code': '13010', 'DiscDate': dates[0], 'CurPerEn': dates[0],
                           'ShOutFY': 1100, 'TrShFY': 100}])
        with self.assertRaisesRegex(ValueError, 'no common computable'):
            build_features(requests, bars, f, dates, 2)


if __name__ == '__main__':
    unittest.main()
