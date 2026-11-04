"""The production financial-input seam preserves known-time and missing data rules."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import financial_inputs as FI
import features as F
from export_valuations import clean

class FinancialInputs(unittest.TestCase):
    def test_production_schema_is_the_c_arm_without_price_statistics(self):
        self.assertEqual(len(FI.COLUMNS),13)
        self.assertEqual(F.columns(F.DEFAULT_PRESET),F.columns(F.LEGACY_PRESET)+FI.COLUMNS)
        self.assertFalse(any(x.endswith('_rel') for x in FI.COLUMNS))

    def test_live_seam_uses_prior_disclosures_and_preserves_unavailable_beta(self):
        dates=pd.date_range('2020-04-01',periods=4)
        prices=pd.DataFrame({'Code':['A']*4,'Date':dates,'close_raw':[100.]*4,'close':[100.]*4})
        market=pd.DataFrame({'Date':dates,'topix':[100.,101.,102.,103.]})
        fins=pd.DataFrame([{'Code':'A','DocType':'FYFinancialStatements_Consolidated_JP',
          'DiscDate':'2020-04-02','DiscTime':'15:00','CurPerType':'FY','CurFYSt':'2019-04-01',
          'CurPerEn':'2020-03-31','NP':100.,'ShEq':500.,'ShOutFY':100.,'TrShFY':0.,'DivTotalAnn':30.}])
        rates=pd.DataFrame({'Date':dates,'10Y':[1.,2.,3.,4.]})
        history=pd.concat([pd.DataFrame({'Code':['A'],'Date':[pd.Timestamp('2020-03-31')],'close_raw':[100.],'close':[100.]}),prices],ignore_index=True)
        out=FI.calculate(prices[['Code','Date','close_raw']],history,fins,market,rates)
        self.assertTrue(pd.isna(out.tv_earnings_yield.iloc[1]))
        self.assertAlmostEqual(out.tv_earnings_yield.iloc[2],.01)
        self.assertTrue(out.tv_beta.isna().all())
        self.assertTrue(out.missing_tv_ri.eq(1).all())
        self.assertTrue(out.missing_tv_ddm.eq(1).all())
        self.assertAlmostEqual(out.tv_rf.iloc[1],(1+.01/2)**2-1)

    def test_export_does_not_turn_missing_prices_into_zero(self):
        import json
        payload=clean({'zero':0.,'missing':np.nan,'infinite':np.inf,'date':pd.Timestamp('2020-01-01'),'x':[np.float64(1)]})
        self.assertEqual(payload,{'zero':0.,'missing':None,'infinite':None,'date':'2020-01-01','x':[1.]})
        json.dumps(payload,allow_nan=False)
