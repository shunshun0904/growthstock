"""本ブレイク予測モデルのラベル（research/major/label_eda.py）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402

import label_eda as M  # noqa: E402

NAN = float("nan")


def path(points, n=M.DAYS_2):
    """{日: 値} から、1日目〜n日目の値（買値 = 1.0）の並びを作る。書いていない日は 1.0。"""
    v = np.ones(n)
    for d, x in points.items():
        v[d - 1] = x
    return v


class 本ブレイクのラベル(unittest.TestCase):

    def label(self, *rows):
        return M.major_label(np.vstack(rows))

    def test_期限までに一度でも届けば正例(self):
        # 40日目に +50%、100日目に2倍。期限の日（60・120日目）には下がっていても正例
        L = self.label(path({40: 1.55, 60: 1.2, 100: 2.05, 120: 1.4}))
        self.assertEqual(L["y"][0], 1.0)
        self.assertEqual((L["first1"][0], L["first2"][0]), (40.0, 100.0))

    def test_50パーセントが60日を過ぎてからなら負例(self):
        L = self.label(path({61: 1.6, 90: 2.1}))
        self.assertEqual(L["y"][0], 0.0)
        self.assertTrue(np.isnan(L["first1"][0]))
        self.assertEqual(L["first2"][0], 90.0)

    def test_2倍が120日を過ぎてからなら負例(self):
        L = M.major_label(path({30: 1.6, 121: 2.2}, n=130)[None, :])
        self.assertEqual(L["y"][0], 0.0)

    def test_買った当日の終値で50パーセントなら急騰で負例(self):
        # 運用で予測する銘柄を前もって絞らないので、急騰は外さずに負例にする。2日目以降はふつうに数える
        L = self.label(path({1: 1.5, 50: 2.0}), path({2: 1.5, 50: 2.0}))
        self.assertEqual(L["spike"].tolist(), [True, False])
        self.assertEqual(L["reach"].tolist(), [True, True])
        self.assertEqual(L["y"].tolist(), [0.0, 1.0])

    def test_売買の無い日は届かなかった扱い(self):
        L = self.label(path({20: NAN, 40: 1.49, 80: 2.0}))
        self.assertEqual(L["y"][0], 0.0)

    def test_チャートの区間は約2年半で78週高値の判定区間を含む(self):
        n = M.BEFORE + M.AFTER + 1
        self.assertTrue(600 <= n <= 625)                   # 245営業日 × 2.5 ≒ 612
        self.assertGreater(M.BEFORE, M.B.HIGH_WINDOW)      # 判定区間（368営業日）がまるごと入る
        self.assertGreaterEqual(M.AFTER, M.DAYS_2)         # 120営業日目が入る


if __name__ == "__main__":
    unittest.main()
