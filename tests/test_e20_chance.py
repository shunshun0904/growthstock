#!/usr/bin/env python3
"""
列ごとの検定（research/exp/e20_annual_trajectory.py の screen）で、偶然 |z| を超える本数の見込み。

z は窓ごとの超過の平均 / 窓の SE なので、窓が n 本なら自由度 n−1 の t 分布。正規分布の 5% で数えると
窓が少ないとき過小になる（2026-10-10、EDINET の 523列・窓11本で「5% ≈ 26本」と出したが、t 分布では 38本）。
"""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

from e20_annual_trajectory import chance_hits  # noqa: E402


class TestChanceHits(unittest.TestCase):
    def test_eleven_windows_is_wider_than_normal(self):
        res = pd.DataFrame({"n_win": [11] * 523, "abs_z": [0.0] * 523})
        n, p, n_win = chance_hits(res)
        self.assertEqual(n_win, 11)
        self.assertAlmostEqual(p, 0.0734, places=3)        # 正規分布なら 0.0455
        self.assertAlmostEqual(n, 38.4, places=1)
        n3, p3, _ = chance_hits(res, 3.0)
        self.assertAlmostEqual(n3, 7.0, places=1)

    def test_many_windows_approaches_normal(self):
        res = pd.DataFrame({"n_win": [200] * 100, "abs_z": [0.0] * 100})
        _, p, _ = chance_hits(res)
        self.assertAlmostEqual(p, 0.0468, places=3)

    def test_unmeasured_rows_and_no_windows(self):
        res = pd.DataFrame({"n_win": [6, 6, None], "abs_z": [0.0, 0.0, None]})
        n, p, n_win = chance_hits(res)
        self.assertEqual(n_win, 6)
        self.assertAlmostEqual(p, 0.1019, places=3)
        self.assertEqual(chance_hits(pd.DataFrame({"abs_z": [0.0]}))[1], 0.05)   # 窓の列が無ければ従来の 5%


if __name__ == "__main__":
    unittest.main()
