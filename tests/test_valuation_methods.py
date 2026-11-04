"""Independent finance identities, ambiguity handling, and information dates."""
import os,sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'research'))
import valuation_methods as V


class ValuationMethods(unittest.TestCase):
    def test_flat_unlevered_no_net_investment_equals_perpetuity(self):
        # EBIT 20, tax 30% -> FCFF=14, WACC=8%, no growth: EV=175.
        equity,ev,_,w=V.dcf_value(100.,20.,5.,5.,15.,100.,0.,0.,0.,.03,1.,terminal_growth=0.)
        self.assertAlmostEqual(w,.08)
        self.assertAlmostEqual(ev,175.)
        self.assertAlmostEqual(equity,175.)

    def test_reverse_polynomial_equals_independent_cashflow_calculation(self):
        args=[100.,20.,5.,10.,15.,100.,20.,10.,3.,.02,1.]
        for g in [-.2,0.,.2]:
            _,ev,_,w=V.dcf_value(*args,growth=g)
            c=V.reverse_coefficients(*args[:5],w)
            self.assertAlmostEqual(np.polynomial.polynomial.polyval(1+g,c),ev)
            roots=V.real_growth_roots(c,float(ev))
            self.assertTrue(any(abs(r-g)<1e-7 for r in roots))

    def test_reverse_retains_multiple_roots_and_rejects_out_of_range(self):
        # (x-.8)*(x-1.2) = x^2-2x+.96.
        np.testing.assert_allclose(V.real_growth_roots([.96,-2,1],0),[-.2,.2],atol=1e-7)
        self.assertEqual(V.real_growth_roots([-3.,1.],0),[])

    def test_zero_equity_paths_are_not_discarded(self):
        s,f=V.stats([[0.,10.,20.]])
        np.testing.assert_allclose(s[0],[0.,20.,10.,10.,np.std([0,10,20])])
        self.assertEqual(f[0],1.)

    def test_asof_ignores_future_and_does_not_fallback_from_incomplete_latest(self):
        s=pd.DataFrame({'Code':['10000']*3,'avail_date':pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01']),
             'filing_avail_date':pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01']),
             'depreciation':[10.,np.nan,99999.]})
        d=pd.DataFrame({'Code':['10000']*2,'Date':pd.to_datetime(['2020-01-15','2020-02-15'])})
        out=V.attach(d,s)
        self.assertEqual(out.depreciation.iloc[0],10.)
        self.assertTrue(np.isnan(out.depreciation.iloc[1]))
        pd.testing.assert_frame_equal(out,V.attach(d,s.iloc[:2]))
        d['Code']=d.Code.astype('string')
        pd.testing.assert_frame_equal(out,V.attach(d,s))

    def test_relative_prices_exclude_self(self):
        f=pd.DataFrame({'Date':pd.to_datetime(['2020-01-01']*3),'Code':['1','2','3'],
                       'S33':['0050']*3,'PER':[100.,10.,20.]})
        target=f.iloc[:1].assign(close_raw=1000.)
        q=V.peer_prices(f,target,'PER',min_peers=2)
        self.assertEqual(q.peers.iloc[0],2)
        self.assertEqual(q['mean'].iloc[0],150.)
        self.assertTrue(V.peer_prices(f,target,'PER',min_peers=3)['mean'].isna().all())

    def test_EV_peer_conversion_subtracts_debt_not_proportional_equity_multiple(self):
        f=pd.DataFrame({'Date':pd.to_datetime(['2020-01-01']*3),'Code':['1','2','3'],
             'S33':['0050']*3,'EV_EBITDA':[11.1,10.,12.]})
        target=f.iloc[:1].assign(close_raw=10.,MktCap=1000.,EBITDA=100.,v_debt=200.,v_cash=100.,v_nci=10.)
        q=V.peer_prices(f,target,'EV_EBITDA',min_peers=2,enterprise=True)
        self.assertAlmostEqual(q['mean'].iloc[0],9.9)

    @staticmethod
    def fixture():
        return pd.DataFrame({'Code':['10000'],'Date':pd.to_datetime(['2020-01-15']),'close_raw':[10.],
          'S33':['0050'],'MktCap':[100.],'tv_beta':[1.],'tv_rf':[.02],
          'revenue':[100.],'operating_income':[20.],'depreciation':[5.],'v_cash_capex':[10.],'v_flex_capex':[10.],
          'v_nwc':[15.],'inventories':[10.],'trade_receivables':[15.],'trade_payables':[10.],
          'v_debt':[20.],'v_cash':[10.],'v_nci':[0.],'v_equity':[50.],'total_assets':[200.],
          'total_liabilities':[150.],'v_fresh':[True],'v_period_ok':[True],'v_jp_basis':[True],
          'v_partial':[False],'v_scope_risk':[False],'v_balance_ok':[True]})

    def test_MC_keyed_seed_is_reproducible_and_order_independent(self):
        d=self.fixture();a=V.monte_carlo(d,draws=128)
        other=d.copy();other.Code='20000'
        both=pd.concat([other,d],ignore_index=True)
        b=V.monte_carlo(both,draws=128).iloc[[1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True),b)
        self.assertGreater(a['std'].iloc[0],0)

    def test_no_missing_WC_imputation_or_debt_larger_than_liabilities(self):
        d=self.fixture();d.loc[0,'trade_payables']=np.nan
        self.assertFalse(V.ready(d).iloc[0])

    def test_cash_capex_requires_both_parts_and_does_not_fill_unknown_with_zero(self):
        p=pd.DataFrame(index=range(3))
        for c in set(V.CE.NUMERIC)|set(V.BASE_FIELDS):p[c]=np.nan
        p['Code']=['1','2','3'];p['fiscal_year']=2020.;p['basis']='consolidated';p['accounting_standard']='JP'
        for c,date in [('period_start','2020-01-01'),('period_end','2020-12-31'),
                       ('filing_avail_date','2021-03-01'),('avail_date','2021-03-01')]:p[c]=pd.Timestamp(date)
        p['doc_id']='sample';p['non_controlling_interests']=np.nan
        p['purchase_of_ppe']=[-40.,-40.,-40.];p['purchase_of_intangible_assets']=[-10.,np.nan,-10.]
        p['purchase_of_ppe_and_intangible_assets']=[np.nan,np.nan,-70.];p['capex']=60.
        s=V.annual_states(p).set_index('Code')
        self.assertEqual(s.loc['1','v_cash_capex'],50.)
        self.assertTrue(np.isnan(s.loc['2','v_cash_capex']))
        self.assertTrue(np.isnan(s.loc['3','v_cash_capex']))
        self.assertEqual(s.loc['2','v_flex_capex'],60.)
        self.assertTrue(s.v_nci.isna().all())
        d=self.fixture();d.loc[0,'v_debt']=200.
        self.assertFalse(V.ready(d).iloc[0])


if __name__=='__main__':unittest.main()
