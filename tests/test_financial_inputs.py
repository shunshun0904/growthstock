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

    def test_yields_join_the_current_month_file(self):
        # 過去分（前月末まで）だけだと月の8日ごろから金利が7日超古くなり、理論株価が欠ける（2026-11-10）。今月分もつなぐ
        import tempfile
        from unittest import mock
        hist=b'Interest Rate (historical)\nDate,1Y,10Y\n2026/10/29,0.9,1.70\n2026/10/30,0.9,1.72\n'
        cur=b'Interest Rate (current)\nDate,1Y,10Y\n2026/10/30,0.9,1.73\n2026/11/4,0.9,-\n2026/11/9,0.9,1.75\n'
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(FI,'_fetch',side_effect=lambda u: hist if u==FI.YIELD_URL else cur):
                out=FI.yields(d)
            self.assertEqual([x.strftime('%Y-%m-%d') for x in out.Date],['2026-10-29','2026-10-30','2026-11-04','2026-11-09'])
            self.assertAlmostEqual(out.set_index('Date')['10Y'][pd.Timestamp('2026-10-30')],1.73)   # 同じ日は今月分
            self.assertTrue(pd.isna(out.set_index('Date')['10Y'][pd.Timestamp('2026-11-04')]))
            again=FI.yields(d,refresh=False)                                                        # 保存した形を読み直せる
            pd.testing.assert_frame_equal(again,out)
            rf=__import__('theoretical_value').riskfree(pd.Series([pd.Timestamp('2026-11-10')]),out)
            self.assertEqual(rf.tv_rf_date.iloc[0],pd.Timestamp('2026-11-09'))
            self.assertTrue(rf.tv_rf.notna().all())
            with mock.patch.object(FI,'_fetch',side_effect=lambda u: hist if u==FI.YIELD_URL else (_ for _ in ()).throw(OSError('x'))):
                only=FI.yields(d)                                                                  # 今月分が取れなくても続ける
            self.assertEqual(only.Date.max(),pd.Timestamp('2026-10-30'))
        with self.assertRaises(ValueError):
            FI._parse(b'title\nDay,X\n1,2\n')

    def test_export_does_not_turn_missing_prices_into_zero(self):
        import json
        payload=clean({'zero':0.,'missing':np.nan,'infinite':np.inf,'date':pd.Timestamp('2020-01-01'),'x':[np.float64(1)]})
        self.assertEqual(payload,{'zero':0.,'missing':None,'infinite':None,'date':'2020-01-01','x':[1.]})
        json.dumps(payload,allow_nan=False)
