"""Independent identities, sampled processes and execution timing for E126."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import optimal_exit as O


class OptimalExit(unittest.TestCase):
    def test_driftless_log_maximum_cdf_reflection_principle(self):
        x=np.array([0,.02,.05,.1]);sigma=.03;t=7
        np.testing.assert_allclose(O.maximum_cdf(x,0,sigma,t),2*ndtr(x/(sigma*np.sqrt(t)))-1)

    def test_driftless_log_maximum_mean_closed_form(self):
        s=.04;t=12
        expected=2*np.exp(s*s*t/2)*ndtr(s*np.sqrt(t))
        self.assertAlmostEqual(float(O.stopping_payoff(t,0,s,.5)),expected,12)

    def test_zero_price_drift_limit_is_continuous(self):
        for y in [0,1,4,10]:
            zero=O.stopping_payoff(9,y,.025,0)
            self.assertAlmostEqual(float(zero),float(O.stopping_payoff(9,y,.025,1e-5)),places=6)

    def test_conditional_payoff_dominates_known_maximum(self):
        y=np.arange(0,20,.25)
        for a in [-.5,0,.5,1,2]:
            self.assertTrue((O.stopping_payoff(19,y,.03,a)>=np.exp(.03*y)-1e-12).all())

    def test_quantile_is_inverse_of_maximum_cdf(self):
        for d in [-.002,0,.003]:
            for q in [.1,.5,.9]:
                value=O.maximum_quantile(q,d,.03,11)
                self.assertAlmostEqual(float(O.maximum_cdf(np.log(value),d,.03,11)),q,11)

    def test_reflected_transition_matches_sampled_bridge(self):
        rng=np.random.default_rng(821);n=200000
        for x,lam in [(0,0),(1,.15),(3,-.1)]:
            closing=lam+rng.standard_normal(n)
            high=(closing+np.sqrt(closing**2-2*np.log(rng.random(n))))/2
            nxt=np.maximum(x,high)-closing
            for z in [.5,1.5,3,5]:
                self.assertAlmostEqual(float(O.reflected_cdf(z,x,lam)),float((nxt<=z).mean()),delta=.004)

    def test_transition_conserves_probability(self):
        grid=np.linspace(0,30,301)
        p=O.transition_matrix(grid,.02)
        np.testing.assert_allclose(np.asarray(p.sum(axis=1)).ravel(),1,atol=1e-12)
        self.assertTrue((p.data>=0).all())

    def test_exponential_moments_against_survival_integration(self):
        for x,lam,sigma in [(0,0,.03),(2,.04,.08),(12,.01,.025),(1,.02,.03)]:
            for b in [1,5,20]:
                # Integration by parts provides an independent expression.
                f=lambda y:float(O.reflected_cdf(y,x,lam))
                expected=np.exp(sigma*b)*f(b)-quad(lambda y:sigma*np.exp(sigma*y)*f(y),0,b,epsabs=1e-11)[0]
                self.assertAlmostEqual(float(O.reflected_exponential_moment(b,x,lam,sigma)),expected,9)

    def test_near_one_alpha_boundary_is_finite(self):
        for s in [.0025,.03]:
            b,_=O.solve_boundary(s,.999,hold=19)
            self.assertTrue(np.isfinite(b).all())
            self.assertTrue((np.diff(b)>=-1e-7).all())

    def test_negative_price_drift_sells_and_strong_drift_holds(self):
        bad,_=O.solve_boundary(.05,-.5,hold=5,points=601,ymax=30)
        good,_=O.solve_boundary(.05,1.5,hold=5,points=601,ymax=30)
        np.testing.assert_allclose(bad,0,atol=.003)
        self.assertTrue(np.isinf(good[1:]).all())

    def test_boundary_increases_with_remaining_time(self):
        b,_=O.solve_boundary(.03,.5,hold=19,points=801,ymax=40)
        self.assertTrue(np.isfinite(b).all())
        self.assertTrue((np.diff(b)>=-1e-8).all())

    def test_posterior_uses_only_observed_gain(self):
        self.assertAlmostEqual(float(O.posterior_drift(.001,.04,5,20)),.06/25)
        self.assertEqual(float(O.posterior_drift(.001,.04,5,np.inf)),.001)

    def test_signal_executes_next_open_with_gap(self):
        r=O.execute_exit([0,-.08,.03],[.04,.02,.08],[.02,-.01,.05],[True,False])
        self.assertEqual(r['exit_day'],2);self.assertAlmostEqual(r['gross'],-.08)

    def test_no_intraday_hindsight_stop(self):
        r=O.execute_exit([0,.01,.02],[.2,.1,.08],[.02,.04,.06],[False,True])
        self.assertEqual(r['exit_day'],3);self.assertAlmostEqual(r['gross'],.02)

    def test_target_gap_at_observed_open_and_purchase_day_winner(self):
        r=O.execute_exit([0,.14],[.05,.17],[.03,.16],target=.1)
        self.assertAlmostEqual(r['gross'],.14)
        r=O.execute_exit([0,.14],[.11,.17],[.03,.16],target=.1)
        self.assertEqual(r['exit_day'],1);self.assertAlmostEqual(r['gross'],.1)

    def test_original_deadline_is_forced(self):
        r=O.execute_exit([0,.02,.01],[.02,.04,.03],[.01,.01,-.02])
        self.assertEqual(r['exit_day'],3);self.assertAlmostEqual(r['gross'],-.02)


if __name__=='__main__':unittest.main()
