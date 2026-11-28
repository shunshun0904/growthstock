"""Independent checks of the experimental time boundary and placebo controls."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
import e119_capital_efficiency as E


class ExperimentContract(unittest.TestCase):
    def test_outcome_maturity_uses_actual_company_bars(self):
        bars = pd.DataFrame({"Code": ["10000"] * 4 + ["20000"] * 3,
            "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-15", "2025-02-16",
                                    "2025-01-01", "2025-01-02", "2025-01-03"])})
        keys = pd.DataFrame({"Code": ["10000", "20000", "10000", "10000", "99990"],
            "Date": ["2025-01-01", "2025-01-01", "2025-01-03", "2025-02-15", "2025-01-01"]})
        m = E.forward_maturity(keys, bars, hold=2)
        self.assertEqual(m.iloc[0], pd.Timestamp("2025-02-15"))
        self.assertEqual(m.iloc[1], pd.Timestamp("2025-01-03"))
        self.assertTrue(m.iloc[2:].isna().all())

    def test_placebo_preserves_date_distribution_and_each_missing_mask(self):
        d = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01"] * 5 + ["2025-01-02"] * 5)})
        for c in E.CE.columns("values"):
            d[c] = [1., 2., np.nan, 3., 4., 10., np.nan, 30., 40., 50.]
        before = d.copy(deep=True)
        p = E.shuffled_values(d)
        for c in E.CE.columns("values"):
            np.testing.assert_array_equal(p[c].isna(), d[c].isna())
            for date in d.Date.unique():
                self.assertEqual(sorted(p.loc[p.Date.eq(date), c].dropna()),
                                 sorted(d.loc[d.Date.eq(date), c].dropna()))
        self.assertFalse(p[E.CE.columns("values")].equals(d[E.CE.columns("values")]))
        pd.testing.assert_frame_equal(d, before)

    def test_control_contains_missingness_without_financial_values_or_targets(self):
        sets = E.column_sets()
        self.assertEqual(len(sets["A"]), 360)
        self.assertFalse(set(E.CE.columns("values")) & set(sets["Q"]))
        self.assertTrue(set(sets["Q"]) < set(sets["B"]))
        self.assertEqual(sets["B"], sets["P"])
        self.assertFalse({"label", "future_rise", "end_level", "ret_c20", "maturity_date"} & set(sets["B"]))
        self.assertEqual(len(E.column_sets(True)["A"]), 364)

    def test_classifier_labels_keep_production_threshold_tolerance(self):
        d=pd.DataFrame({'future_rise':[.1-1e-16,.15-1e-16,.1-1e-8],
            'end_level':[.05,.075,.05],'uptrend_end':[True,True,True],
            'entry_price':[100.,100.,100.]})
        labels=E.classification_labels(d)
        self.assertEqual(labels['y_class10'].tolist(),[1.,1.,0.])
        self.assertEqual(labels['y_class15'].tolist(),[0.,1.,0.])

    def test_parameter_boundary_includes_rolling_pool_outcomes(self):
        daily=pd.DataFrame({'Date':pd.to_datetime(['2025-09-03']),
                            'maturity_date':pd.to_datetime(['2025-10-03'])})
        pool=pd.DataFrame({'Date':pd.to_datetime(['2025-09-02','2025-09-04']),
                           'maturity_date':pd.to_datetime(['2025-10-06','2026-01-01'])})
        self.assertEqual(E.parameter_outcome_end(daily,pool,pd.Timestamp('2025-09-03')),
                         pd.Timestamp('2025-10-06'))

    def test_classifier_imbalance_matches_training_prevalence(self):
        p=E.classifier_params({'num_leaves':11},np.array([0,0,0,1]),seed=42,threads=2)
        self.assertEqual(p['scale_pos_weight'],3.)
        self.assertEqual(p['n_estimators'],200)
        self.assertEqual(p['random_state'],42)
        self.assertEqual(p['num_leaves'],11)


if __name__ == "__main__":
    unittest.main()
