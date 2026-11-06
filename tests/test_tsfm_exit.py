"""TSFM maximum-functional target, quantile distributions, and timed exits."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import tsfm_exit as T


class TSFMExit(unittest.TestCase):
    def test_rolling_max_final_forecast_target_is_future_max(self):
        h=np.array([100,101,99,102,101,105,104,103,108,106],float)
        for t in range(3,7):
            r=3
            rolling=T.rolling_max(h,r)
            self.assertEqual(rolling[t+r],max(h[t+1:t+r+1]))

    def test_context_never_observes_future(self):
        h=np.linspace(100,140,40);r=4
        a=T.context_at(h,25,r,16,130)
        h[26:]=10000
        np.testing.assert_allclose(a,T.context_at(h,25,r,16,130),equal_nan=True)

    def test_forecast_max_quantiles_are_not_daily_max_quantiles(self):
        # Counterexample: anti-correlated daily highs both marginally50% above.
        paths=np.array([[120,80],[80,120]])
        self.assertEqual((paths.max(axis=1)>=110).mean(),1)
        self.assertEqual(np.max((paths>=110).mean(axis=0)),.5)

    def test_linear_quantile_distribution_probability_and_moments(self):
        # Uniform[0,2] with complete quantile endpoints has mean1,var1/3.
        p,mean,sd=T.distribution_summary([0,1,2],[0,.5,1],threshold=1.5,tail='constant')
        self.assertAlmostEqual(p,.25)
        self.assertAlmostEqual(mean,1)
        self.assertAlmostEqual(sd,np.sqrt(1/3))

    def test_flat_quantile_atom_counts_as_arrival_at_equal_price(self):
        p,mean,sd=T.distribution_summary([1,1,1],[.1,.5,.9],threshold=1,tail='constant')
        self.assertEqual(p,1);self.assertEqual(mean,1);self.assertAlmostEqual(sd,0)
        self.assertEqual(T.distribution_summary([1,1,1],[.1,.5,.9],1.01)[0],0)

    def test_nested_purchase_price_targets(self):
        q=[.9,1.05,1.3];levels=[.1,.5,.9]
        p10=T.distribution_summary(q,levels,1.10)[0]
        p15=T.distribution_summary(q,levels,1.15)[0]
        self.assertGreaterEqual(p10,p15)

    def test_crossed_quantiles_are_sorted_before_use(self):
        a=T.distribution_summary([1.2,1.0,1.1],[.1,.5,.9],1.08)
        b=T.distribution_summary([1.0,1.1,1.2],[.1,.5,.9],1.08)
        np.testing.assert_allclose(a,b)

    def test_future_trading_high_can_be_below_current_close(self):
        maxima=T.volatility_paths(1,.10,0,20000,seed=7)
        self.assertTrue((maxima<1).any())
        self.assertGreater((maxima<1).mean(),.45)

    def test_zero_volatility_and_reproducibility(self):
        a=T.volatility_paths(5,0,0,100,seed=10)
        np.testing.assert_allclose(a,1)
        np.testing.assert_array_equal(T.volatility_paths(3,.02,.03,100,seed=9),T.volatility_paths(3,.02,.03,100,seed=9))

    def test_decision_after_close_exits_next_open(self):
        o=np.array([0,.03,-.02]);h=np.array([.04,.08,.04]);c=np.array([.02,.04,.01]);p=np.array([.7,.1,.5])
        r=T.policy_exit(o,h,c,p,.10,'absolute',.2)
        self.assertEqual(r['exit_day'],3);self.assertAlmostEqual(r['gross'],-.02)

    def test_half_probability_rule_uses_first_close(self):
        o=np.array([0,.03,.01]);h=np.array([.04,.08,.04]);c=np.array([.02,.04,.01]);p=np.array([.6,.25,.5])
        r=T.policy_exit(o,h,c,p,.15,'half',.2)
        self.assertEqual(r['exit_day'],3)

    def test_purchase_day_target_winner_is_never_dropped(self):
        r=T.policy_exit(np.array([0,.03]),np.array([.16,.06]),np.array([.12,.04]),np.array([.01,.01]),.15,'absolute',.2)
        self.assertEqual(r['exit_day'],1);self.assertAlmostEqual(r['gross'],.15)

    def test_deadline_does_not_extend(self):
        r=T.policy_exit(np.array([0,.03]),np.array([.04,.06]),np.array([.02,-.01]),np.array([.8,.8]),.15,'absolute',.2)
        self.assertEqual(r['exit_day'],2);self.assertAlmostEqual(r['gross'],-.01)


if __name__=='__main__':unittest.main()
