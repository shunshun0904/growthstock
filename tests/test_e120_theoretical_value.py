"""Experiment isolation, coherent placebo and exact forward outcome boundaries."""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research', 'exp'))
import e120_theoretical_value as E


class ExperimentContract(unittest.TestCase):
    def test_controls_include_model_inputs_but_no_future_outcomes(self):
        s=E.column_sets()
        self.assertEqual(len(s['A']),360)
        self.assertTrue(set(E.V.CONTROLS).issubset(s['C']))
        self.assertFalse(set(E.V.VALUES)&set(s['C']))
        self.assertEqual(s['B'],s['P'])
        self.assertEqual(len(s['R'])-len(s['C']),5)
        self.assertEqual(len(s['D'])-len(s['C']),5)
        self.assertFalse({'label','ret_c20','maturity_date','future_rise'}&set(s['B']))
        self.assertEqual(len(E.column_sets(True)['A']),364)

    def test_joint_shuffle_preserves_ranges_distribution_and_missing_positions(self):
        d=pd.DataFrame({'Date':pd.to_datetime(['2020-01-01']*10+['2020-01-02']*10)})
        for j,c in enumerate(E.V.VALUES):
            d[c]=np.arange(20,dtype=float)+j
        for c in E.V.VALUES[:5]:
            d.loc[[2,3,12],c]=np.nan
        for c in E.V.VALUES[5:]:
            d.loc[[7,17],c]=np.nan
        d['missing_tv_ri']=d.tv_ri_mean_rel.isna().astype(float)
        d['missing_tv_ddm']=d.tv_ddm_mean_rel.isna().astype(float)
        before=d.copy(deep=True)
        p=E.shuffled_values(d)
        np.testing.assert_array_equal(p[E.V.VALUES].isna(),d[E.V.VALUES].isna())
        for _,g in d.groupby(['Date']+E.QUALITY):
            a=d.loc[g.index,E.V.VALUES].fillna(-999).to_numpy()
            b=p.loc[g.index,E.V.VALUES].fillna(-999).to_numpy()
            self.assertEqual(sorted(map(tuple,a)),sorted(map(tuple,b)))
        pd.testing.assert_frame_equal(d,before)
        self.assertFalse(p[E.V.VALUES].equals(d[E.V.VALUES]))

    def test_forward_return_uses_next_open_and_company_bars_not_calendar(self):
        bars=pd.DataFrame({'Code':['10000']*4,'Date':pd.to_datetime(['2020-01-01','2020-01-02','2020-03-01','2020-03-02']),
                           'AdjO':[90.,100.,105.,110.],'AdjC':[95.,103.,120.,130.]})
        keys=pd.DataFrame({'Code':['10000']*3,'Date':pd.to_datetime(['2020-01-01','2020-01-03','2020-03-01'])})
        ret,dates=E.forward_returns(keys,bars,2)
        self.assertAlmostEqual(ret[0],.2)
        self.assertEqual(pd.Timestamp(dates[0]),pd.Timestamp('2020-03-01'))
        self.assertTrue(np.isnan(ret[1:]).all())

    def test_ttm_uses_previous_fy_and_same_interim_without_per_share_mixing(self):
        f=pd.DataFrame({'Code':['10000']*3,'DocType':['1QFinancialStatements_Consolidated_JP',
             'FYFinancialStatements_Consolidated_JP','1QFinancialStatements_Consolidated_JP'],
             'CurPerType':['1Q','FY','1Q'],'CurFYSt':['2020-01-01','2020-01-01','2021-01-01'],
             'CurPerEn':['2020-03-31','2020-12-31','2021-03-31'],
             'DiscDate':['2020-04-30','2021-02-10','2021-04-30'],'DiscTime':['16:00']*3,
             'NP':[20.,100.,30.],'ShEq':[400.,450.,500.],'ShOutFY':[10.]*3,
             'TrShFY':[0.]*3,'DivTotalAnn':[np.nan,25.,np.nan]})
        p=pd.DataFrame({'Code':['10000']*2,'Date':pd.to_datetime(['2021-03-31','2021-05-01']),
                        'close_raw':[50.,50.],'close':[50.,50.]})
        a=E.V.financial_inputs(p.iloc[1:],f,p)
        self.assertEqual(a.tv_eps.iloc[0],11.)
        self.assertEqual(a.tv_bps.iloc[0],50.)
        self.assertEqual(a.tv_div0.iloc[0],2.5)
        # A newer incomplete statement must not silently resurrect the old FY.
        incomplete=f.copy(); incomplete.loc[2,'NP']=np.nan
        self.assertTrue(E.V.financial_inputs(p.iloc[1:],incomplete,p).tv_eps.isna().all())
        incomplete=f.copy(); incomplete.loc[2,'TrShFY']=np.nan
        self.assertTrue(E.V.financial_inputs(p.iloc[1:],incomplete,p).tv_eps.isna().all())


if __name__=='__main__':
    unittest.main()
