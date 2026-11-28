import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from valuation_display import price_keys, unavailable_reason, latest_financials

class DisplayValuation(unittest.TestCase):
    def test_previous_common_close_when_topix_lags_and_no_forward_fill(self):
        p=pd.DataFrame({'Code':['A']*3,'Date':pd.to_datetime(['2026-10-29','2026-10-30','2026-11-02']),'close_raw':[90,100,110]})
        keys, latest=price_keys(p,pd.DataFrame({'Date':['2026-10-29','2026-10-30']}))
        self.assertEqual(keys.Date.iloc[0],pd.Timestamp('2026-10-30'))
        self.assertEqual(keys.close_raw.iloc[0],100)
        self.assertEqual(latest.loc['A','close_raw'],110)
        keys,_=price_keys(p,pd.DataFrame({'Date':['2026-10-29']}))
        self.assertTrue(keys.empty)  # More than one stock bar back is not silently accepted.

    def test_current_close_remains_when_both_are_current(self):
        p=pd.DataFrame({'Code':['A'],'Date':pd.to_datetime(['2026-11-02']),'close_raw':[110]})
        keys,_=price_keys(p,pd.DataFrame({'Date':['2026-11-02']}))
        self.assertEqual(keys.close_raw.iloc[0],110)

    def test_failed_recovery_is_a_scenario_problem_not_a_price_date_problem(self):
        r=pd.Series({'Code':'A','Date':pd.Timestamp('2026-11-02'),'tv_beta':1.,'tv_rf':.02,'tv_eps':-200.,'tv_bps':100.,'ri_recovery_assumed':True})
        reason=unavailable_reason('ri',r,{})
        self.assertIn('純資産が0以下',reason)
        self.assertNotIn('黒字企業のみ',reason)
        self.assertNotIn('TOPIX',reason)

    def test_unknown_treasury_is_not_zero_or_a_claim_of_missing_close(self):
        f=pd.DataFrame({'Code':['A'],'DiscDate':['2026-08-13'],'CurPerEn':['2026-06-30'],'DocType':['FYFinancialStatements_Consolidated_JP'],'CurPerType':['FY'],'ShOutFY':[100.],'TrShFY':[None]})
        r=pd.Series({'Code':'A','Date':pd.Timestamp('2026-11-02'),'tv_beta':1.,'tv_rf':.02,'tv_eps':float('nan'),'tv_bps':float('nan')})
        reason=unavailable_reason('ri',r,latest_financials(f))
        self.assertIn('自己株式数が欠損',reason)
        self.assertIn('ゼロとは仮定しない',reason)
