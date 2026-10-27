"""Independent identities, session semantics, and as-of invariance for stock FID."""
import os
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "research"))
import distribution_geometry as G


def bars_fixture(n=240):
    days = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    parts = []
    for code in ("1001", "1002"):
        c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
        parts.append(pd.DataFrame({"Code": code, "Date": days, "AdjC": c,
                                   "AdjO": c * 0.999, "AdjVo": 1000.0}))
    return pd.concat(parts, ignore_index=True), days


class TestDistance(unittest.TestCase):
    def test_identities_and_special_cases(self):
        d = G.gaussian_fisher_rao_distance
        self.assertEqual(d(3, 2, 3, 2), 0)
        self.assertAlmostEqual(d(3, 2, 3, 7), np.sqrt(2) * np.log(7 / 2))
        # Hyperbolic-plane geodesic; not the length of the constant-sigma path.
        self.assertAlmostEqual(d(0, 1, 2, 1), 2 * np.sqrt(2) * np.arcsinh(1 / np.sqrt(2)))
        self.assertLess(d(0, 1, 2, 1), 2)

    def test_against_independent_arcosh_formula(self):
        rng = np.random.default_rng(8)
        m1, m2 = rng.normal(size=(2, 500))
        s1, s2 = np.exp(rng.uniform(-3, 3, (2, 500)))
        expected = np.sqrt(2) * np.arccosh(1 + ((m1-m2)**2 + 2*(s1-s2)**2)/(4*s1*s2))
        got = G.gaussian_fisher_rao_distance(m1, s1, m2, s2)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(got, G.gaussian_fisher_rao_distance(m2, s2, m1, s1))
        np.testing.assert_allclose(got, G.gaussian_fisher_rao_distance(7*m1+3, 7*s1, 7*m2+3, 7*s2))
        self.assertTrue((got >= 0).all())

    def test_extreme_scales_no_saturation_and_invalid_values(self):
        d = G.gaussian_fisher_rao_distance
        self.assertAlmostEqual(d(0, 1e-300, 0, 1e300), np.sqrt(2)*600*np.log(10))
        self.assertTrue(np.isfinite(d(-1e308, 1e-300, 1e308, 1e-300)))
        self.assertGreater(d(0, 1, 1e-12, 1), 0)
        self.assertTrue(np.isnan(d(0, 0, 0, 1)))
        self.assertTrue(np.isnan(d(0, -1, 0, 1)))
        self.assertTrue(np.isnan(d(np.inf, 1, 0, 1)))


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.bars, self.days = bars_fixture()
        self.cfg = G.GeometryConfig(window=20)

    def test_hand_calculated_window_and_signed_changes(self):
        f = G.compute_features(self.bars, self.days, self.cfg).set_index(["Code", "Date"])
        c = self.bars[self.bars.Code == "1001"].AdjC.to_numpy()
        r = np.diff(np.log(c))
        t, lag, w = 110, 20, self.cfg.window
        cur, old = r[t-w:t], r[t-lag-w:t-lag]
        row = f.loc[("1001", self.days[t])]
        self.assertAlmostEqual(row["geom_w20_delta_mu_20"], cur.mean()-old.mean())
        self.assertAlmostEqual(row["geom_w20_delta_log_sigma_20"], np.log(cur.std()/old.std()))
        expected = np.sqrt(2)*np.arccosh(1+((cur.mean()-old.mean())**2+2*(cur.std()-old.std())**2)/(4*cur.std()*old.std()))
        self.assertAlmostEqual(row["geom_w20_fid_20"], expected)
        self.assertTrue(f.loc[("1001", self.days[24])].isna().all())

    def test_past_only_zscore(self):
        f = G.compute_features(self.bars, self.days, self.cfg)
        one = f[f.Code == "1001"].reset_index(drop=True)
        t = 170
        past = one.geom_w20_fid_20.iloc[t-60:t]
        expected = (one.geom_w20_fid_20.iloc[t]-past.mean())/past.std(ddof=0)
        self.assertAlmostEqual(one.geom_w20_fid_zscore_60.iloc[t], expected)
        self.assertAlmostEqual(one.geom_w20_fid_velocity.iloc[t], one.geom_w20_fid_20.iloc[t]-one.geom_w20_fid_20.iloc[t-1])
        self.assertAlmostEqual(one.geom_w20_fid_ma_10.iloc[t], one.geom_w20_fid_20.iloc[t-9:t+1].mean())

    def test_future_mutation_truncation_and_ticker_isolation(self):
        baseline = G.compute_features(self.bars, self.days, self.cfg)
        cutoff = self.days[175]
        changed = self.bars.copy()
        changed.loc[changed.Date > cutoff, "AdjC"] *= 100
        actual = G.compute_features(changed, self.days, self.cfg)
        assert_frame_equal(baseline[baseline.Date <= cutoff], actual[actual.Date <= cutoff])
        truncated = G.compute_features(self.bars[self.bars.Date <= cutoff], self.days[self.days <= cutoff], self.cfg)
        assert_frame_equal(baseline[baseline.Date <= cutoff].reset_index(drop=True), truncated)
        one = G.compute_features(self.bars[self.bars.Code == "1001"], self.days, self.cfg)
        assert_frame_equal(baseline[baseline.Code == "1001"].reset_index(drop=True), one)

    def test_missing_session_is_not_compressed_or_filled(self):
        bad = (self.bars.Code == "1001") & (self.bars.Date == self.days[100])
        absent = G.compute_features(self.bars[~bad], self.days, self.cfg)
        missing = self.bars.copy()
        missing.loc[bad, "AdjC"] = np.nan
        explicit = G.compute_features(missing, self.days, self.cfg)
        # Dropping a missing bar must not turn a two-session move into a daily return.
        joined = absent.merge(explicit, on=["Code", "Date"], suffixes=("_a", "_b"))
        for c in G.feature_columns([], "M3", self.cfg):
            np.testing.assert_allclose(joined[c+"_a"], joined[c+"_b"], equal_nan=True)
        row = absent[(absent.Code == "1001") & (absent.Date == self.days[101])]
        self.assertTrue(row.geom_w20_fid_5.isna().all())

    def test_flat_price_zero_volume_and_adjustment(self):
        flat = self.bars.copy()
        flat["AdjC"] = 100.0
        flat["C"] = np.where(flat.Date < self.days[100], 100.0, 50.0)
        f = G.compute_features(flat, self.days, self.cfg)
        self.assertEqual(f.geom_w20_fid_20.dropna().abs().max(), 0)
        self.assertTrue(f.geom_w20_fid_zscore_60.isna().all())
        flat.loc[flat.Date == self.days[100], "AdjVo"] = 0
        f = G.compute_features(flat, self.days, self.cfg)
        self.assertTrue(f.loc[f.Date == self.days[101], "geom_w20_fid_5"].isna().all())

    def test_short_history_large_jump_and_shuffle(self):
        b = self.bars.copy()
        b.loc[b.Date >= self.days[150], "AdjC"] *= 1000
        b = b[(b.Code != "1002") | (b.Date >= self.days[-10])]
        actual = G.compute_features(b.sample(frac=1, random_state=8), self.days, self.cfg)
        self.assertTrue(actual[actual.Code == "1002"].filter(like="geom_").isna().all().all())
        self.assertFalse(np.isinf(actual.filter(like="geom_").to_numpy()).any())
        assert_frame_equal(actual, G.compute_features(b, self.days, self.cfg))

    def test_switch_preserves_baseline_and_row_order(self):
        samples = self.bars.iloc[[190, 3, 360]][["Code", "Date"]].copy()
        samples["baseline"] = [1., 2., 3.]
        assert_frame_equal(samples, G.attach_features(samples, None, None, arm="M0"))
        self.assertEqual(G.feature_columns(["baseline"], "M0"), ["baseline"])
        arms = {a: G.feature_columns(["baseline"], a, self.cfg) for a in G.ARMS}
        self.assertEqual(set(arms["M3"]), set(arms["M1"]) | set(arms["M2"]))
        self.assertEqual(set(arms["M1"]) & set(arms["M2"]), {"baseline"})
        joined = G.attach_features(samples, self.bars, self.days, self.cfg, arm="M3")
        assert_frame_equal(joined[samples.columns], samples)

    def test_rejects_ambiguous_input(self):
        with self.assertRaises(ValueError):
            G.compute_features(pd.concat([self.bars, self.bars.iloc[:1]]), self.days)
        with self.assertRaises(ValueError):
            G.compute_features(self.bars, self.days[1:])
        with self.assertRaises(ValueError):
            G.GeometryConfig(window=1)
        with self.assertRaises(ValueError):
            G.GeometryConfig(lags=(5, 60))


if __name__ == "__main__":
    unittest.main()
