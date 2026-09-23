"""本ブレイク: 目的変数「+30%に −20%より先に届くか」のラベル（research/exp/m04_major_direction.py）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import numpy as np  # noqa: E402

import m04_major_direction as M4  # noqa: E402


class 上が先のラベル(unittest.TestCase):

    def test_Aは届かなければ0_参考のBは届かない行を除く(self):
        cat = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        y_a, y_b = M4.labels_from(cat, np.ones(5, dtype=bool))
        self.assertEqual(y_a.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(y_b[:4].tolist(), [1.0, 0.0, 0.0, 0.0])      # 同じ日に両方（4）は下が先
        self.assertTrue(np.isnan(y_b[4]))

    def test_売買を数えられない行は使わない(self):
        y_a, y_b = M4.labels_from(np.array([1, 3], dtype=np.int8), np.array([False, True]))
        self.assertTrue(np.isnan(y_a[0]) and np.isnan(y_b[0]))
        self.assertEqual((y_a[1], y_b[1]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
