"""Fair search opportunity, exact forward purging, native production contracts."""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'research','exp'))
import e121_theoretical_value_retune as R


class RetuningContract(unittest.TestCase):
    def test_matched_bank_contains_previous_and_independently_reproducible_search(self):
        p=R.E.parameters()['class10']
        a=R.candidate_bank(p,30,42);b=R.candidate_bank(p,30,42)
        self.assertEqual(a,b)
        self.assertEqual(len(a),30)
        self.assertEqual(a[0],{k:p[k] for k in R.SEARCH_KEYS})
        self.assertEqual(len({tuple(x.values()) for x in a}),30)
        self.assertNotEqual(a[1:],R.candidate_bank(p,30,43)[1:])
        self.assertEqual(R.ARMS,('A','C','B'))

    def test_purge_suspended_company_outcome_and_keep_whole_date_groups(self):
        dates=pd.date_range('2020-01-01','2021-01-01',freq='7D').repeat(2)
        d=pd.DataFrame({'Date':dates,'maturity_date':dates+pd.Timedelta(days=5),
                        'y_class10':np.tile([0,1],len(dates)//2)})
        # Old signal with a very delayed result must be excluded from CV training.
        d.loc[0,'maturity_date']=pd.Timestamp('2021-02-01')
        folds,checks=R.forward_folds(d,'class10',n_splits=2,months=3,min_train_months=0)
        self.assertEqual(len(folds),2)
        for (ti,vi),check in zip(folds,checks):
            self.assertNotIn(0,ti)
            self.assertLess(d.iloc[ti].maturity_date.max(),d.iloc[vi].Date.min())
            self.assertFalse(set(d.iloc[ti].Date)&set(d.iloc[vi].Date))
            self.assertTrue(d.iloc[vi].groupby('Date').size().eq(2).all())
            self.assertLess(check['train_max_outcome'],check['valid_from'])

    def test_selects_only_cv_with_deterministic_tie_break(self):
        # Intentionally opposite outer-test ordering: selector must ignore it.
        records=[{'candidate':2,'cv_mean':.3,'holdout':.9},
                 {'candidate':1,'cv_mean':.4,'holdout':.1},
                 {'candidate':0,'cv_mean':.4,'holdout':0.}]
        self.assertEqual(R.select_candidate(records)['candidate'],0)
        with self.assertRaises(ValueError):R.select_candidate([{'candidate':0,'cv_mean':np.nan}])

    @staticmethod
    def fixture():
        rng=np.random.default_rng(345)
        d=pd.DataFrame({'Date':pd.bdate_range('2020-01-01',periods=20).repeat(8),
                        'Code':list(map(str,range(8)))*20,'x':rng.normal(size=160),'z':rng.normal(size=160)})
        d['y_class10']=(d.x+d.z*.2>1).astype(int)
        d[R.ltr.R20]=d.x*.1+d.z*.03
        params=dict(learning_rate=.05,num_leaves=7,min_child_samples=5,subsample=.8,
                    colsample_bytree=.8,reg_alpha=.01,reg_lambda=.1)
        return d,params

    def test_native_classifier_matches_production_class_weight_contract(self):
        import lightgbm as lgb
        d,p=self.fixture();cols=['x','z']
        native=R.Trainer(d,cols,'class10',threads=1).fit(p,seed=42,trees=20)
        ref=lgb.LGBMClassifier(**dict(p,n_estimators=20,subsample_freq=1,feature_pre_filter=False,
            data_random_seed=42,n_jobs=1,random_state=42,verbose=-1,
            scale_pos_weight=R.tuning.scale_pos_weight(d.y_class10)))
        ref.fit(d[cols].to_numpy(),d.y_class10)
        np.testing.assert_allclose(native.predict(d[cols].to_numpy(),num_threads=1),
                                   ref.predict_proba(d[cols].to_numpy())[:,1],rtol=1e-12,atol=1e-12)

    def test_native_ranker_matches_production_grades_groups_and_linear_gain(self):
        d,p=self.fixture();cols=['x','z']
        native=R.Trainer(d,cols,'ltr',threads=1).fit(p,seed=42,trees=20)
        ref=R.ltr.fit(d,cols,dict(p,n_estimators=20,n_jobs=1,subsample_freq=1,
                      feature_pre_filter=False,data_random_seed=42,verbose=-1),seed=42)
        np.testing.assert_allclose(native.predict(d[cols].to_numpy(),num_threads=1),
                                   ref.predict(d[cols].to_numpy()),rtol=1e-12,atol=1e-12)

    def test_reusing_bins_allows_smaller_leaf_size_and_seed_changes(self):
        d,p=self.fixture();t=R.Trainer(d,['x','z'],'class10',threads=1)
        t.fit(dict(p,min_child_samples=50),seed=42,trees=5)
        m=t.fit(dict(p,min_child_samples=2),seed=43,trees=5)
        self.assertTrue(np.isfinite(m.predict(d[['x','z']].to_numpy(),num_threads=1)).all())


if __name__=='__main__':unittest.main()
