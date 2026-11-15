"""Research gating, exact maturity purging, equal cohorts and proxy controls."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "exp"))
import e132_valuation_gap as E
import valuation_gap as G
import features as F


class ExperimentContracts(unittest.TestCase):
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
