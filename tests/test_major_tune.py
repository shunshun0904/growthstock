"""本ブレイク予測モデルの探索（research/exp/m02_major_tune.py）の部品。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import m02_major_tune as M2  # noqa: E402


class 時期で切って前後を外す(unittest.TestCase):

    def setUp(self):
        self.cal = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=1000))
        rng = np.random.default_rng(0)
        self.d = pd.DataFrame({"Date": self.cal[rng.integers(0, 1000, 3000)]})

    def test_検証の前後の営業日は学習に入らない(self):
        folds = M2.purged_blocks(self.d, self.cal, k=5, purge=121)
        pos = self.cal.searchsorted(self.d["Date"].to_numpy())
        for tr, va in folds:
            lo, hi = pos[va].min(), pos[va].max()
            gap = np.minimum(np.abs(pos[tr] - lo), np.abs(pos[tr] - hi))
            inside = (pos[tr] >= lo) & (pos[tr] <= hi)
            self.assertFalse(inside.any())
            self.assertTrue((gap > 121).all())

    def test_どの行もどこかの検証にちょうど1回入る(self):
        folds = M2.purged_blocks(self.d, self.cal, k=5, purge=121)
        seen = np.concatenate([va for _, va in folds])
        self.assertEqual(sorted(seen.tolist()), list(range(len(self.d))))


class ボラ系の列を外す(unittest.TestCase):

    def test_グループと相関で外す(self):
        rng = np.random.default_rng(1)
        n = 500
        vol = rng.random(n)
        rows = pd.DataFrame({
            "vol_20d": vol,
            "ret_20d": vol + rng.normal(0, 0.01, n),           # breakout グループ → グループで外す
            "ROE_q0": rng.random(n),                            # 無関係 → 残す
            "div_yield": -vol + rng.normal(0, 0.05, n),         # 相関が強い → 相関で外す
        })
        cols = ["vol_20d", "ret_20d", "ROE_q0", "div_yield"]
        keep, by_group, by_rho = M2.exvol_columns(cols, rows)
        self.assertEqual(keep, ["ROE_q0"])
        self.assertEqual(set(by_group), {"vol_20d", "ret_20d"})
        self.assertEqual(list(by_rho.index), ["div_yield"])


if __name__ == "__main__":
    unittest.main()
