"""本ブレイク予測モデルのラベル（research/major/label_eda.py）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import label_eda as M  # noqa: E402


class 本ブレイクのラベル(unittest.TestCase):

    def test_両方届いたときだけ正例(self):
        r60 = pd.Series([0.50, 0.49, 0.80, 0.60])
        r120 = pd.Series([1.00, 1.50, 0.99, 1.20])
        self.assertEqual(M.major_label(r60, r120).tolist(), [1.0, 0.0, 0.0, 1.0])

    def test_どちらかが欠測なら判定しない(self):
        y = M.major_label(pd.Series([0.6, np.nan]), pd.Series([np.nan, 1.2]))
        self.assertTrue(y.isna().all())

    def test_チャートの区間は約2年半で78週高値の判定区間を含む(self):
        n = M.BEFORE + M.AFTER + 1
        self.assertTrue(600 <= n <= 625)                   # 245営業日 × 2.5 ≒ 612
        self.assertGreater(M.BEFORE, M.B.HIGH_WINDOW)      # 判定区間（368営業日）がまるごと入る
        self.assertGreaterEqual(M.AFTER, 120)              # 120営業日後の判定日が入る


if __name__ == "__main__":
    unittest.main()
