"""Daily +10/+15 forecasts: fixed-entry targets, observable prefixes and execution."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import daily_arrival as D


class DailyArrival(unittest.TestCase):
    def path(self):
        # Returns relative to an original purchase at100, not to today's close.
        return [np.array([x],float) for x in [
            [0,.04,.09,.12],[.06,.12,.14,.17],[-.02,.01,.06,.09],[.04,.10,.12,.16]]]

    def test_purchase_target_and_deadline_are_fixed(self):
        o,h,l,c=self.path();s=D.landmark_arrays(o,h,l,c)
        self.assertEqual(s['remaining'][0,2],2)
        self.assertEqual(s['y15'][0,2],1) # Future117 reaches115 despite current112.
        self.assertTrue(s['arrived10'][0,2])
        self.assertFalse(s['risk10'][0,2])
        self.assertTrue(s['risk15'][0,2])

    def test_purchase_open_has_no_buy_day_close_or_high(self):
        s=D.landmark_arrays(*self.path())
        self.assertEqual(s['spot_return'][0,0],0)
        self.assertEqual(s['known_peak'][0,0],0)
        self.assertEqual(s['known_low'][0,0],0)

    def test_future_mutation_changes_labels_not_current_features(self):
        o,h,l,c=self.path();a=D.landmark_arrays(o,h,l,c)
        h2=h.copy();h2[:,2:]=.08;c2=c.copy();c2[:,2:]=.07
        b=D.landmark_arrays(o,h2,l,c2)
        for name in D.PREFIX_FEATURES:
            np.testing.assert_allclose(a[name][:,:3],b[name][:,:3],equal_nan=True,err_msg=name)
        self.assertEqual(a['y15'][0,2],1)
        self.assertEqual(b['y15'][0,2],0)

    def test_future_max_excludes_known_high(self):
        o,h,l,c=self.path();h[:,0]=.5;s=D.landmark_arrays(o,h,l,c)
        self.assertAlmostEqual(s['future_max_from_entry'][0,1],.17)
        self.assertAlmostEqual(s['future_max_from_spot'][0,1],1.17/1.04-1)

    def test_competing_stop_same_bar_and_open_precedence(self):
        o=np.array([[0,.02]]);h=np.array([[.03,.16]]);l=np.array([[-.02,-.08]]);c=np.array([[.01,.04]])
        s=D.landmark_arrays(o,h,l,c)
        self.assertEqual(s['comp10'][0,1],2)
        o[0,1]=.12;s=D.landmark_arrays(o,h,l,c)
        self.assertEqual(s['comp10'][0,1],1)
        self.assertEqual(s['comp15'][0,1],2)

    def test_continuation_compares_next_open_not_known_close(self):
        o,h,l,c=self.path();s=D.landmark_arrays(o,h,l,c)
        self.assertAlmostEqual(s['gain10'][0,1],.10-.04)
        self.assertAlmostEqual(s['gain15'][0,2],.15-.09)

    def test_past_stop_removes_competing_risk_rows(self):
        o,h,l,c=self.path();l[0,0]=-.06;s=D.landmark_arrays(o,h,l,c)
        self.assertFalse(s['comp_risk10'][0,1])
        self.assertTrue(s['risk10'][0,1]) # Unconditional arrival can still occur.

    def test_maturity_blocks_entire_purchase_episode(self):
        d=pd.DataFrame({'trade':[1,1,2,2],'mature_date':pd.to_datetime(['2026-01-20']*2+['2026-01-21']*2)})
        self.assertEqual(D.matured_mask(d,'2026-01-21').tolist(),[True,True,False,False])

    def test_probability_order_and_arrived_status(self):
        p10,p15=D.coherent_probabilities(np.array([.2,.4,.3]),np.array([.4,.2,.9]),np.array([False,False,True]),np.array([False,False,True]))
        np.testing.assert_allclose(p10,[.3,.4,1]);np.testing.assert_allclose(p15,[.3,.2,1])

    def test_daily_decision_executes_at_next_open(self):
        o,h,l,c=self.path();g=np.ones(4);g[1]=-1
        r=D.simulate_daily_exit(o[0],h[0],l[0],c[0],g,target=.15,cutoff=0)
        self.assertEqual(r['exit_day'],2)
        self.assertAlmostEqual(r['gross'],.04)

    def test_target_hit_today_precedes_evening_exit_decision(self):
        o,h,l,c=self.path();g=np.ones(4);g[2]=-1
        r=D.simulate_daily_exit(o[0],h[0],l[0],c[0],g,target=.10,cutoff=0)
        self.assertEqual(r['exit_day'],2)
        self.assertAlmostEqual(r['gross'],.10)
        self.assertEqual(r['reason'],'target')

    def test_day20_exit_does_not_wait_for_day21_open(self):
        o,h,l,c=self.path();g=np.ones(4)
        r=D.simulate_daily_exit(o[0],h[0],l[0],c[0],g,target=.50,cutoff=0)
        self.assertEqual(r['exit_day'],4)
        self.assertAlmostEqual(r['gross'],.16)

    def test_technical_features_cannot_see_later_prices_or_volume(self):
        close=np.arange(100,140,dtype=float)
        frame=pd.DataFrame({'AdjC':close,'AdjO':close-1,'AdjH':close+2,'AdjL':close-2})
        volume=pd.Series(np.arange(1000,1040,dtype=float))
        before=D.daily_technicals(frame,volume)
        altered=frame.copy();altered.loc[30:,'AdjC']*=5
        altered.loc[30:,'AdjH']*=7;volume2=volume.copy();volume2.loc[30:]*=100
        after=D.daily_technicals(altered,volume2)
        pd.testing.assert_frame_equal(before.loc[:29],after.loc[:29])

    def test_profit_gap_assumptions_match_for_target_and_early_exit(self):
        o,h,l,c=self.path();o[0,1]=.20;h[0,1]=.22;g=np.ones(4);g[1]=-1
        early=D.simulate_daily_exit(o[0],h[0],l[0],c[0],g,target=.15,cutoff=0)
        fixed=D.simulate_daily_exit(o[0],h[0],l[0],c[0],np.ones(4),target=.15,cutoff=0)
        self.assertEqual(early['gross'],fixed['gross'])
        for gain in [g,np.ones(4)]:
            r=D.simulate_daily_exit(o[0],h[0],l[0],c[0],gain,target=.15,cutoff=0,profit_gap='open')
            self.assertAlmostEqual(r['gross'],.20)


if __name__=='__main__':unittest.main()
