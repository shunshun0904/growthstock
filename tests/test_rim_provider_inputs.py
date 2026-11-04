import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import theoretical_value as V
from valuation_display import rim_with_provider_inputs as fill

class ProviderInputs(unittest.TestCase):
    def inputs(self):
        d=pd.DataFrame({'Code':['A'],'Date':pd.to_datetime(['2026-11-02']),
          'close_raw':[50.],'tv_eps':[np.nan],'tv_bps':[np.nan],'tv_fin_source_date':[pd.NaT],
          'tv_beta':[1.],'tv_rf':[.02],'tv_div0':[np.nan],'tv_payout':[np.nan],
          'tv_earnings_yield':[np.nan],'missing_tv_ri':[1.]})
        p=pd.DataFrame({'Code':['A','A'],'Date':pd.to_datetime(['2026-10-30','2026-11-02']),
          'close_raw':[100.,50.],'close':[50.,50.]})
        q=pd.DataFrame({'Code':['A'],'Date':pd.to_datetime(['2026-10-30']),'EPS':[10.],'BPS':[100.]})
        return V.valuations(d),p,q,p[['Date']]

    def test_provider_pair_recovers_rim_and_converts_split_units(self):
        d,p,q,m=self.inputs();r=fill(d,p,q,m)
        self.assertEqual(r.tv_eps.iloc[0],5.)
        self.assertEqual(r.tv_bps.iloc[0],50.)
        self.assertTrue(pd.notna(r.tv_ri_median.iloc[0]))
        self.assertEqual(r.ri_input_source.iloc[0],'provider_eps_bps')
        self.assertEqual(r.ri_input_date.iloc[0],pd.Timestamp('2026-10-30'))
        self.assertTrue(pd.isna(r.tv_earnings_yield.iloc[0]))
        self.assertEqual(r.missing_tv_ri.iloc[0],1.)
        self.assertTrue(pd.isna(r.tv_ddm_median.iloc[0]))
        self.assertTrue(pd.isna(d.tv_eps.iloc[0]))

    def test_future_stale_and_missing_pairs_do_not_fill(self):
        for date in ['2026-11-04','2026-10-29']:
            d,p,q,m=self.inputs();q['Date']=pd.to_datetime([date]);m=pd.concat([m,pd.DataFrame({'Date':pd.to_datetime(['2026-10-29'])})])
            extra=p.iloc[[0]].copy();extra['Date']=pd.to_datetime([date]);p=pd.concat([p,extra])
            self.assertTrue(pd.isna(fill(d,p,q,m).tv_ri_median.iloc[0]))
        d,p,q,m=self.inputs();q['BPS']=np.nan
        self.assertTrue(pd.isna(fill(d,p,q,m).tv_ri_median.iloc[0]))

    def test_observed_losses_or_equity_are_never_overridden(self):
        for col,value in [('tv_eps',-1.),('tv_bps',-1.)]:
            d,p,q,m=self.inputs();d[col]=value
            r=fill(d,p,q,m)
            self.assertEqual(r[col].iloc[0],value)
            self.assertTrue(pd.isna(r.tv_ri_median.iloc[0]))

    def test_original_complete_inputs_are_preserved(self):
        d,p,q,m=self.inputs();d['tv_eps']=7.;d['tv_bps']=60.;d=V.valuations(d)
        r=fill(d,p,q,m)
        self.assertEqual(r.tv_eps.iloc[0],7.)
        self.assertEqual(r.tv_bps.iloc[0],60.)
        self.assertEqual(r.tv_ri_median.iloc[0],d.tv_ri_median.iloc[0])
