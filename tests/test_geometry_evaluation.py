"""Leakage boundaries, executable portfolio accounting, and real LTR integration."""
import os
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research"))
import distribution_geometry as G
import geometry_evaluation as E
import ltr


def fixture(n=450, stocks=6):
    days = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(22)
    parts = []
    for i in range(stocks):
        c = 100*np.exp(np.cumsum(rng.normal(.0004, .01, n)))
        parts.append(pd.DataFrame({"Code": str(1000+i), "Date": days, "AdjC": c,
                                   "AdjO": c*.998, "AdjH": c*1.01, "AdjVo": 1000.}))
    bars = pd.concat(parts, ignore_index=True)
    samples = bars[["Code", "Date"]].copy()
    samples["label"] = 1.
    samples["x"] = bars.groupby("Code").AdjC.pct_change(fill_method=None)
    return samples, bars, days


class TestTargetsAndPurging(unittest.TestCase):
    def test_target_matches_existing_ltr_on_complete_bars(self):
        samples, bars, days = fixture(n=80, stocks=2)
        got = E.attach_targets(samples, bars, days, hold=20)
        old = ltr.attach_r20(samples, bars)
        np.testing.assert_allclose(got.ret_c20, old.ret_c20, equal_nan=True)
        self.assertEqual(got.label_end.iloc[0], days[20])
        self.assertEqual(got.entry_date.iloc[0], days[1])

    def test_missing_open_and_missing_session_do_not_shift_execution(self):
        samples, bars, days = fixture(n=45, stocks=1)
        bars = bars[bars.Date != days[1]]
        got = E.attach_targets(samples.iloc[:1], bars, days)
        self.assertTrue(got.ret_c20.isna().all())
        self.assertEqual(got.entry_date.iloc[0], days[1])
        self.assertEqual(got.label_end.iloc[0], days[20])

    def test_actual_label_intervals_purged_on_exchange_sessions(self):
        samples, bars, days = fixture()
        f = E.attach_targets(samples, bars, days)
        cfg = E.EvaluationConfig(min_train_months=8, test_months=3, validation_months=2,
                                 min_train_rows=20, min_test_rows=10)
        splits = E.walk_forward_splits(f, days, cfg)
        self.assertGreater(len(splits), 1)
        all_test = []
        for s in splits:
            tr, va, inner, te = (f.loc[s[k]] for k in ("train", "validation", "inner_train", "test"))
            self.assertLess(tr.label_end.max(), days[days.searchsorted(te.Date.min())-cfg.embargo_sessions])
            self.assertLess(inner.label_end.max(), days[days.searchsorted(va.Date.min())-cfg.embargo_sessions])
            all_test.extend(s["test"])
        self.assertEqual(len(all_test), len(set(all_test)))


class TestMetricsAndAccounting(unittest.TestCase):
    def test_rank_metrics_and_no_future_return_filter_in_top_selection(self):
        f = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]*6), "Code": list("abcdef"),
                          "score": [6., 5., 4., 3., 2., 1.], "ret_c20": [.6,.5,.4,.3,.2,.1]})
        r = E.daily_metrics(f, top_k=2).iloc[0]
        self.assertAlmostEqual(r.rank_ic, 1.)
        self.assertAlmostEqual(r.ndcg, 1.)
        self.assertAlmostEqual(r.top_bottom, .4)
        f.loc[0, "ret_c20"] = np.nan
        r = E.daily_metrics(f, top_k=2).iloc[0]
        self.assertTrue(np.isnan(r.top_return))
        self.assertEqual(r.n_scored, 6)
        f["score"] = 1.
        self.assertTrue(np.isnan(E.daily_metrics(f).rank_ic.iloc[0]))

    def test_nonoverlapping_portfolio_cost_cash_and_initial_drawdown(self):
        days = pd.bdate_range("2024-01-01", periods=5)
        bars = pd.DataFrame({"Code": "a", "Date": days, "AdjO": 100., "AdjC": [100.,100.,110.,110.,110.]})
        scores = pd.DataFrame({"Code": ["a"], "Date": [days[0]], "score": [1.]})
        curve = E.portfolio_curve(scores, bars, days, hold=2, top_k=1, cost_bps=100)
        expected = 1.1*.99/1.01
        self.assertAlmostEqual(curve.nav.iloc[-1], expected)
        self.assertAlmostEqual(curve.nav.iloc[0], 1/1.01)
        stats = E.portfolio_stats(curve)
        self.assertAlmostEqual(stats["total_return"], expected-1)
        self.assertAlmostEqual(stats["max_drawdown"], 1/1.01-1)
        # Two slots, one available stock -> half cash, not leveraged concentration.
        cash = E.portfolio_curve(scores, bars, days, hold=2, top_k=2, cost_bps=0)
        self.assertAlmostEqual(cash.nav.iloc[-1], 1.05)

    def test_missing_entry_keeps_cash_missing_valuation_stops(self):
        days = pd.bdate_range("2024-01-01", periods=5)
        bars = pd.DataFrame({"Code": "a", "Date": days, "AdjO": [100,np.nan,100,100,100], "AdjC": 100.})
        scores = pd.DataFrame({"Code": ["a"], "Date": [days[0]], "score": [1.]})
        curve = E.portfolio_curve(scores, bars, days, hold=2, top_k=1)
        self.assertEqual(curve.nav.iloc[-1], 1.)
        bars.loc[1,"AdjO"] = 100.
        bars.loc[2,"AdjC"] = np.nan
        with self.assertRaises(ValueError):
            E.portfolio_curve(scores, bars, days, hold=2, top_k=1)

    def test_block_bootstrap_paired_difference_and_reproducibility(self):
        a = np.tile([.1,.2,np.nan,.3], 40)
        b = a + .025
        one = E.paired_block_interval(a, b, block=20, repetitions=200, seed=42)
        two = E.paired_block_interval(a, b, block=20, repetitions=200, seed=42)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one["difference"], .025)
        self.assertAlmostEqual(one["ci_low"], .025)
        self.assertAlmostEqual(one["ci_high"], .025)


class TestRealRanker(unittest.TestCase):
    def test_checkpoint_resume_and_changed_input_rejection(self):
        from unittest.mock import patch
        samples, bars, days = fixture(n=200)
        gc = G.GeometryConfig(window=5, lags=(1,2,5), reference_lag=2, ma_window=3, z_window=6)
        frame = E.attach_targets(G.attach_features(samples, bars, days, gc, arm='M3'), bars, days, hold=5)
        cfg = E.EvaluationConfig(hold=5, min_train_months=4, test_months=3, validation_months=1,
                                 min_train_rows=20, min_test_rows=10, candidates=1, n_estimators=5)
        with tempfile.TemporaryDirectory() as tmp:
            first, audit = E.run_walk_forward(frame, ['x'], days, gc, cfg, checkpoint_dir=tmp)
            with patch.object(ltr, 'fit', side_effect=AssertionError('Resume must not refit')):
                resumed, resumed_audit = E.run_walk_forward(frame, ['x'], days, gc, cfg, checkpoint_dir=tmp)
            assert_frame_equal(first, resumed, check_exact=True)
            self.assertEqual(audit, resumed_audit)
            changed = frame.copy()
            changed.loc[0, 'x'] = 123.
            with self.assertRaisesRegex(ValueError, 'changed'):
                E.run_walk_forward(changed, ['x'], days, gc, cfg, checkpoint_dir=tmp)
            from pathlib import Path
            p = next(Path(tmp).glob('*.parquet'))
            p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                E.run_walk_forward(frame, ['x'], days, gc, cfg, checkpoint_dir=tmp)

    def test_all_arms_same_queries_and_purged_validation(self):
        samples, bars, days = fixture(n=200)
        gc = G.GeometryConfig(window=5, lags=(1,2,5), reference_lag=2, ma_window=3, z_window=6)
        frame = G.attach_features(samples, bars, days, gc, arm="M3")
        frame = E.attach_targets(frame, bars, days, hold=5)
        cfg = E.EvaluationConfig(hold=5, min_train_months=4, test_months=3, validation_months=1,
                                 min_train_rows=20, min_test_rows=10, candidates=2, n_estimators=5,
                                 min_day=3, top_k=2, bootstrap_repetitions=20)
        out, audit = E.run_walk_forward(frame, ["x"], days, gc, cfg)
        self.assertEqual(set(out.arm), set(G.ARMS))
        ref = out[out.arm == "M0"][["Code", "Date", "fold"]].reset_index(drop=True)
        for arm in G.ARMS:
            assert_frame_equal(ref, out[out.arm == arm][ref.columns].reset_index(drop=True))
        self.assertTrue(np.isfinite(out.score).all())
        self.assertTrue(all(len(a["validation_scores"]) == 2 for a in audit))
        self.assertEqual(len({a["rows_fingerprint"] for a in audit if a["fold"] == 1}), 1)
        # Corrupt future outcomes: the first fold's tuning and predictions cannot change.
        corrupt = frame.copy()
        corrupt.loc[corrupt.Date >= out.Date.min(), ltr.R20] *= -10
        changed, changed_audit = E.run_walk_forward(corrupt, ["x"], days, gc, cfg)
        first = out.fold.min()
        np.testing.assert_array_equal(out.loc[out.fold == first, "score"], changed.loc[changed.fold == first, "score"])
        old_tuning = [a["validation_scores"] for a in audit if a["fold"] == first]
        new_tuning = [a["validation_scores"] for a in changed_audit if a["fold"] == first]
        self.assertEqual(old_tuning, new_tuning)
        # Exercise persisted metrics/portfolio/paired intervals, not just fit/predict.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research", "exp"))
        import e115_distribution_geometry as CLI
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory() as tmp:
            summary = CLI.summarize(out, bars, days, Path(tmp), cfg)
            self.assertEqual(set(summary.arm), set(G.ARMS))
            with open(Path(tmp)/"paired_comparisons.json") as fh:
                pairs = json.load(fh)
            self.assertTrue(any(p["baseline"] == "M1" and p["treatment"] == "M3" for p in pairs))
            self.assertFalse(any(p["primary"] for p in pairs))  # W=5 is only a test fixture.
            self.assertTrue((Path(tmp)/"portfolio_daily.csv").is_file())
            with open(Path(tmp)/"portfolio_summary.json") as fh:
                portfolios = json.load(fh)
            self.assertTrue(all(p["status"] == "ok" for p in portfolios))


if __name__ == "__main__":
    unittest.main()
