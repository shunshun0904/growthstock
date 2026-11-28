"""Signed RIM obeys clean surplus without negative dividends or hidden rescue cash."""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import theoretical_value as V
from valuation_display import loss_aware_residual_value, rim_with_loss_recovery

class LossRim(unittest.TestCase):
    def test_negative_years_use_full_losses_and_zero_dividends(self):
        # B0=100, E0=-30, ke=10%, H=3. ROE1=-1/6, ROE2=-1/30, ROE3=10%.
        # D1=D2=0; B1=250/3, B2=725/9. D3+B3=1595/18.
        expected = (1595/18) / 1.1**3
        actual = float(loss_aware_residual_value(100,-30,.1,3,.5))
        self.assertAlmostEqual(actual, expected, places=11)
        self.assertNotAlmostEqual(actual, float(V.residual_value(100,-30,.1,3,.5)),places=5)

    def test_profitable_case_matches_existing_model_and_scales(self):
        for h in [5,10]:
            for payout in [.25,.5,.75]:
                x=float(loss_aware_residual_value(100,20,.1,h,payout))
                self.assertAlmostEqual(x,float(V.residual_value(100,20,.1,h,payout)),places=11)
                self.assertAlmostEqual(float(loss_aware_residual_value(50,-15,.1,h,payout)),
                                       float(loss_aware_residual_value(100,-30,.1,h,payout))/2,places=11)

    def test_exhausted_book_is_rejected_without_resurrection_or_imputed_capital(self):
        self.assertTrue(np.isnan(loss_aware_residual_value(100,-200,.1,5,.5)))
        self.assertTrue(np.isnan(loss_aware_residual_value(-100,-20,.1,5,.5)))
        self.assertTrue(np.isnan(loss_aware_residual_value(100,-20,0,5,.5)))

    def test_missing_object_inputs_are_preserved_and_zero_earnings_are_explicit(self):
        missing=pd.DataFrame({'tv_eps':pd.Series([None],dtype=object),'tv_bps':pd.Series([None],dtype=object)})
        self.assertFalse(rim_with_loss_recovery(missing).ri_recovery_assumed.iloc[0])
        zero=V.valuations(pd.DataFrame({'tv_bps':[100.], 'tv_eps':[0.], 'tv_beta':[1.],
             'tv_rf':[.02], 'close_raw':[100.], 'tv_div0':[0.], 'tv_payout':[np.nan]}))
        result=rim_with_loss_recovery(zero)
        self.assertTrue(result.ri_recovery_assumed.iloc[0])
        self.assertTrue(pd.notna(result.tv_ri_median.iloc[0]))

    def test_insufficient_solvent_scenarios_do_not_produce_a_misleading_range(self):
        # EPS/BPS=-115%: first-year book remains positive in H=5 but not H=10.
        d=V.valuations(pd.DataFrame({'tv_bps':[100.], 'tv_eps':[-115.], 'tv_beta':[1.],
             'tv_rf':[.02], 'close_raw':[100.], 'tv_div0':[0.], 'tv_payout':[np.nan]}))
        r=rim_with_loss_recovery(d)
        self.assertEqual(r.tv_ri_valid_fraction.iloc[0],.5)
        self.assertTrue(pd.isna(r.tv_ri_median.iloc[0]))

    def test_display_extends_only_loss_cases_and_leaves_ddm_and_controls(self):
        d=pd.DataFrame({'tv_bps':[236.,100.], 'tv_eps':[-6.,10.], 'tv_beta':[.43,1.],
                        'tv_rf':[.028,.02], 'close_raw':[385.,100.],
                        'tv_div0':[1.,2.], 'tv_payout':[np.nan,.2],
                        'missing_tv_ri':[1.,0.],'tv_earnings_yield':[-6/385,.1]})
        d=V.valuations(d);out=rim_with_loss_recovery(d)
        self.assertTrue(pd.notna(out.tv_ri_median.iloc[0]))
        self.assertTrue(out.ri_recovery_assumed.iloc[0])
        self.assertFalse(out.ri_recovery_assumed.iloc[1])
        for col in [*V.CONTROLS[5:9], 'missing_tv_ri', *[c for c in d if c.startswith('tv_ddm_')]]:
            if col in d:pd.testing.assert_series_equal(d[col],out[col])
        self.assertEqual(d.tv_ri_median.iloc[1],out.tv_ri_median.iloc[1])
        self.assertTrue(pd.isna(d.tv_ri_median.iloc[0]))
