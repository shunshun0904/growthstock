"""Research gating, exact maturity purging, equal cohorts and proxy controls."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "exp"))
import e132_valuation_gap as E
import valuation_gap as G
import features as F
import build_valuation_feature_mart as B
from build_roic_wacc_series import digest,json_write


class ExperimentContracts(unittest.TestCase):
    def test_class10_only_cli_runs_matched_studies_and_exports_fixed_labels(self):
        dates = pd.date_range('2018-01-01', periods=60, freq='BMS')
        codes = ['10000', '20000', '30000', '40000', '50000']
        expected = dict(zip(codes, [1, 0, 0, 0, 1]))
        records = []
        for date in dates:
            for i, code in enumerate(codes):
                record = {c: 0. for c in F.columns(F.DEFAULT_PRESET)}
                record.update(Code=code, Date=date, label=float(expected[code]),
                              future_rise=[.10, .04, .13, .11, .20][i],
                              end_level=[.05, .03, .04, .06, .09][i],
                              uptrend_end=[1., 1., 1., 0., 1.][i], entry_price=100.)
                record[F.columns(F.DEFAULT_PRESET)[0]] = float(i)
                records.append(record)
        dataset = pd.DataFrame(records)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mart = root/'mart'; mart.mkdir()
            dataset.to_parquet(root/'dataset.parquet', index=False)
            bars = pd.DataFrame({'Date': pd.bdate_range('2018-01-01', '2023-02-28'),
                                 'open': 100., 'high': 112., 'low': 99., 'close': 106.})
            pd.concat([bars.assign(Code=code) for code in codes]).to_parquet(root/'prices.parquet', index=False)
            features = dataset[['Code', 'Date']].copy()
            for field in B.FIELDS:
                features[field] = np.nan
            features['valuation_gap'] = .5
            features['rim_valuation_gap'] = .6
            file = mart/'features-0000.parquet'; features.to_parquet(file, index=False)
            json_write(mart/'inputs.json', {'data': {'dataset.parquet': digest(root/'dataset.parquet')}})
            json_write(mart/'completed.json', {'status': 'completed', 'all_original_keys_attempted': True,
                       'fingerprint': 'synthetic-class10', 'missing_price_history_codes': 0})
            json_write(file.with_suffix('.meta.json'), {'fingerprint': 'synthetic-class10', 'sha256': digest(file)})
            result = subprocess.run([sys.executable, str(HERE/'exp/e132_valuation_gap.py'),
                '--dataset', str(root/'dataset.parquet'), '--feature-mart', str(mart),
                '--prices', str(root/'prices.parquet'), '--output', str(root/'output'),
                '--run', '--tasks', 'class10', '--retune', '--trials', '2', '--inner-folds', '1',
                '--last-folds', '2', '--trees', '3', '--jobs', '2', '--threads', '1'],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            out = root/'output'
            self.assertEqual(json.loads((out/'plan.json').read_text())['tasks'], ['class10'])
            completed = json.loads((out/'completed.json').read_text())
            self.assertEqual(completed['tasks'], ['class10'])
            self.assertEqual(completed['logical_cv_fits'], 28)
            self.assertEqual(completed['outer_fits'], 42)
            predictions = list(out.glob('predictions-*.parquet'))
            self.assertEqual(len(predictions), 14)
            self.assertTrue(all(p.name.startswith('predictions-class10-') for p in predictions))
            for file in predictions:
                prediction = pd.read_parquet(file)
                np.testing.assert_array_equal(prediction.y_true, prediction.Code.map(expected))
                self.assertTrue(prediction.score.between(0, 1).all())
            metrics = pd.read_csv(out/'results.csv')
            self.assertEqual(set(metrics.task), {'class10'})
            self.assertTrue(metrics.primary.between(0, 1).all())
            direct = pd.read_csv(out/'direct_gap_comparison.csv')
            self.assertEqual(set(direct.task), {'class10'})
            self.assertIn('top_decile_precision', direct)
            self.assertNotIn('ndcg_at_5', direct)

    def test_task_selection_preserves_defaults_and_rejects_ambiguous_requests(self):
        from argparse import Namespace
        self.assertEqual(E.selected_tasks(Namespace(classification=False)), ['ltr'])
        self.assertEqual(E.selected_tasks(Namespace(classification=True)), ['ltr', 'class10', 'class15'])
        self.assertEqual(E.selected_tasks(Namespace(tasks=['class10'], classification=False)), ['class10'])
        for tasks, legacy in [(['class10'], True), (['class10', 'class10'], False), (['unknown'], False)]:
            with self.assertRaises(ValueError):
                E.selected_tasks(Namespace(tasks=tasks, classification=legacy))

    def test_CV_parallelism_preserves_selection_and_rejects_immature_rows(self):
        dates=pd.date_range('2018-01-01',periods=42,freq='MS').repeat(5)
        x=np.tile(np.arange(5),42).astype(float)
        frame=pd.DataFrame({'Date':dates,'maturity_date':dates+pd.offsets.BDay(20),'x':x,'ret_c20':x/100})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'serial').mkdir(); (root/'parallel').mkdir()
            args=dict(task='ltr',outer=0,arm='synthetic',train=frame,columns=['x'],
                      bank=[E.FIXED,{**E.FIXED,'num_leaves':3}],output=root/'serial',
                      source_fingerprint='synthetic-test',inner_folds=1,threads=1,trees=5,
                      outer_start=frame.maturity_date.max()+pd.Timedelta(days=1))
            sequential=E.retune_study(**args)
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                parallel=pool.submit(E.retune_study,**{**args,'output':root/'parallel'}).result(timeout=90)
            self.assertEqual(sequential,parallel)
            self.assertEqual(len(json.loads((root/'serial/cv-ltr-0-synthetic.json').read_text())['records']),2)
            self.assertEqual(sequential,E.retune_study(**args))
            with self.assertRaisesRegex(ValueError,'Immature'):
                E.retune_study(**{**args,'outer_start':frame.Date.max()})
            with self.assertRaisesRegex(ValueError,'different inputs'):
                E.retune_study(**{**args,'source_fingerprint':'wrong'})

    def test_sampling_preserves_missing_keys_instead_of_selecting_current_securities(self):
        keys=pd.DataFrame({'Code':['10000','20000'],'Date':pd.to_datetime(['2025-01-02','2025-01-03'])})
        history=pd.DataFrame({'Code':['10000'],'Date':pd.to_datetime(['2025-01-02']),
                              'price':[100.],'price_adjusted':[100.],'valuation_gap':[.5]})
        result=B.sample_history(keys,history)
        pd.testing.assert_frame_equal(result[['Code','Date']],keys)
        self.assertEqual(result.valuation_gap.iloc[0],.5)
        self.assertTrue(result.iloc[1][B.FIELDS].isna().all())

    def test_complete_feature_mart_requires_all_original_keys_and_valid_chunk_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); mart=root/'mart';mart.mkdir()
            dates=pd.bdate_range('2025-01-01',periods=30)
            record={c:0. for c in F.columns(F.DEFAULT_PRESET)}
            record.update(Code='10000',Date=dates[0],label=1.,future_rise=.12,end_level=.06,uptrend_end=1.,entry_price=100.)
            pd.DataFrame([record]).to_parquet(root/'dataset.parquet',index=False)
            pd.DataFrame({'Code':'10000','Date':dates,'open':100.,'high':112.,'low':99.,'close':106.}).to_parquet(root/'prices.parquet',index=False)
            feature={c:np.nan for c in B.FIELDS}
            feature.update(Code='10000',Date=dates[0],valuation_gap=.5,rim_valuation_gap=.6)
            file=mart/'features-0000.parquet';pd.DataFrame([feature]).to_parquet(file,index=False)
            json_write(mart/'inputs.json',{'data':{'dataset.parquet':digest(root/'dataset.parquet')}})
            json_write(mart/'completed.json',{'status':'completed','all_original_keys_attempted':True,'fingerprint':'synthetic-test','missing_price_history_codes':0})
            json_write(file.with_suffix('.meta.json'),{'fingerprint':'synthetic-test','sha256':digest(file)})
            result,status=E.prepare(root/'dataset.parquet',None,root/'prices.parquet',root,mart)
            self.assertEqual(status['status'],'ready')
            self.assertEqual(status['common_gap_available_rows'],1)
            self.assertEqual(result.label.iloc[0],1.)
            file.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'provenance'):
                E.prepare(root/'dataset.parquet',None,root/'prices.parquet',root,mart)

    def test_training_labels_are_mature_before_each_walk_forward_test(self):
        dates = pd.date_range("2018-01-01", "2025-01-01", freq="7D").repeat(5)
        d = pd.DataFrame({"Date": dates, "maturity_date": dates + pd.offsets.BDay(20)})
        folds = E.walk_forward(d)
        self.assertGreater(len(folds), 0)
        for train, test in folds:
            self.assertLess(d.iloc[train].maturity_date.max(), d.iloc[test].Date.min())
            self.assertFalse(set(d.iloc[train].Date) & set(d.iloc[test].Date))

    def test_ranking_metrics_independent_perfect_order(self):
        frame = pd.DataFrame({"Date": np.repeat(pd.to_datetime(["2025-01-02", "2025-01-03"]), 5),
                              "Code": [str(i) for i in range(5)] * 2, "ret_c20": list(np.arange(5) / 100) * 2})
        m = E.ranking_metrics(frame, frame.ret_c20.to_numpy())
        self.assertAlmostEqual(m["ndcg_at_5"], 1.)
        self.assertAlmostEqual(m["top5_return_20d"], .02)
        self.assertAlmostEqual(m["top10pct_excess_20d"], .02)
        self.assertAlmostEqual(m["spearman_ic"], 1.)

    def test_no_valuation_and_incomplete_universe_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); series = root / "series"; series.mkdir()
            dates = pd.bdate_range("2025-01-01", periods=30)
            record = {c: 0. for c in F.columns(F.DEFAULT_PRESET)}
            record.update(Code="10000", Date=dates[0], label=1., future_rise=.12, end_level=.06, uptrend_end=1., entry_price=100.)
            pd.DataFrame([record]).to_parquet(root / "dataset.parquet", index=False)
            pd.DataFrame({"Code": "10000", "Date": dates, "open": 100., "high": 112., "low": 99., "close": 106.}).to_parquet(root / "prices.parquet", index=False)
            d, status = E.prepare(root / "dataset.parquet", series, root / "prices.parquet", root)
            self.assertEqual(status["status"], "blocked")
            self.assertIn("no_observed_fundamental_valuation_gap", status["reasons"])
            self.assertIn("incomplete_historical_universe", status["reasons"])
            self.assertFalse(status["backtest_completed"])
            self.assertEqual(d.label.iloc[0], 1.)

    def test_market_control_history_has_no_future_information(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        price = np.linspace(100, 200, 300)
        f = pd.DataFrame({"Code": "10000", "Date": dates, "price": price, "price_adjusted": price,
                          "market_per_forecast": price / 10, "bps_reported": 50.})
        a = G.market_control_features(f.iloc[:200])
        f.loc[200:, ["price", "price_adjusted", "market_per_forecast"]] = 1e9
        b = G.market_control_features(f)
        pd.testing.assert_frame_equal(a, b.iloc[:200])
        self.assertEqual(G.column_sets()["A"], F.columns(F.DEFAULT_PRESET))
        self.assertIn("M", G.column_sets())


if __name__ == "__main__":
    unittest.main()
