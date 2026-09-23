"""本ブレイク: 運用に近い形の資金の増え方（research/major/portfolio.py）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402

import portfolio as PF  # noqa: E402

NAN = float("nan")


def sim(buy, held, ret, prio=None, period=None, slots=1, per=1):
    n = len(buy)
    return PF.simulate(np.array(buy), np.array(period if period is not None else [0] * n), np.array(held),
                       np.array(ret, dtype=float), np.array(prio if prio is not None else [0.0] * n), slots, per)


class 買うものの決め方(unittest.TestCase):

    def test_同じ期間は上限まで_同じ日は点数の高い方(self):
        r = sim([5, 5, 6], [3, 3, 3], [0.1, 0.2, 0.3], prio=[0.1, 0.9, 0.5], period=[1, 1, 1], slots=3, per=1)
        self.assertEqual([t[4] for t in r["trades"]], [1])
        self.assertEqual(r["skipped_quota"], 2)

    def test_枠が埋まっていたら見送る_売った翌日から買える(self):
        # 枠1つ。0番は5日目に買い7日目に売る。1番（7日目）は同じ日なので買えず、2番（8日目）は買える
        r = sim([5, 7, 8], [3, 2, 2], [0.1, 0.1, 0.1], period=[1, 2, 3], slots=1, per=1)
        self.assertEqual([t[4] for t in r["trades"]], [0, 2])
        self.assertEqual(r["skipped_slot"], 1)

    def test_枠ごとに複利(self):
        r = sim([0, 1, 10], [5, 5, 5], [0.5, -0.5, 0.1], period=[0, 1, 2], slots=2, per=1)
        caps = [(t[0], t[5]) for t in r["trades"]]
        self.assertEqual(caps, [(0, 0.5), (1, 0.5), (0, 0.75)])     # 枠0は +50% のあとの 0.75 で次を買う


class 日々の資産(unittest.TestCase):

    def test_持っている間は時価_売った日は売値_その後は現金(self):
        path = np.array([[1.1, 0.9, 1.5, 2.0]])
        r = sim([2], [3], [0.2])
        eq = PF.equity(r, path, lo=0, hi=6)
        np.testing.assert_allclose(eq, [1.0, 1.0, 1.1, 0.9, 1.2, 1.2, 1.2])

    def test_枠が2つなら半分ずつ(self):
        path = np.array([[1.0, 1.0], [0.5, 0.5]])
        r = sim([0, 0], [2, 2], [0.0, -0.5], period=[0, 1], slots=2, per=1)
        eq = PF.equity(r, path, lo=0, hi=2)
        np.testing.assert_allclose(eq, [0.75, 0.75, 0.75])

    def test_欠けた日は前の日の値(self):
        path = np.array([[NAN, 1.2, NAN, 1.0]])
        r = sim([0], [4], [0.0])
        np.testing.assert_allclose(PF.equity(r, path, lo=0, hi=3), [1.0, 1.2, 1.2, 1.0])

    def test_最大の下げと窓ごとの増減(self):
        eq = np.array([1.0, 1.2, 0.9, 1.08])
        m = PF.metrics(eq)
        self.assertAlmostEqual(m["mdd"], 0.9 / 1.2 - 1.0)
        self.assertAlmostEqual(m["mult"], 1.08)
        w = PF.window_returns(eq, lo=0, bounds=[(0, 1), (2, 3)])
        np.testing.assert_allclose(w, [0.2, 1.08 / 1.2 - 1.0])


if __name__ == "__main__":
    unittest.main()
