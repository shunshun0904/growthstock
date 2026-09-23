"""本ブレイク予測モデルの売買の模擬（research/major/trade.py）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trade as TR  # noqa: E402

NAN = float("nan")


def paths(H, C, entry=100.0, n=None):
    H = np.asarray(H, dtype=float)[None, :]
    C = np.asarray(C, dtype=float)[None, :]
    return {"H": H, "C": C, "entry": np.array([entry]), "n": np.array([H.shape[1] if n is None else n])}


class 売り方(unittest.TestCase):

    def test_場中に2倍に届いたらちょうど2倍で売る(self):
        H = [101] * 9 + [205] + [150] * 10
        C = [100] * 9 + [180] + [150] * 10
        r, d, w = TR.exit_target(paths(H, C), days=20)
        self.assertAlmostEqual(r[0], 1.0)
        self.assertEqual((d[0], w[0]), (10, 1))

    def test_届かなければ期限の日の終値(self):
        H = [120] * 20
        C = [110] * 19 + [130]
        r, d, w = TR.exit_target(paths(H, C), days=20)
        self.assertAlmostEqual(r[0], 0.30)
        self.assertEqual((d[0], w[0]), (20, 2))

    def test_期限の日に売買が無ければその前の終値(self):
        H = [120] * 19 + [NAN]
        C = [110] * 18 + [125, NAN]
        r, _, _ = TR.exit_target(paths(H, C), days=20)
        self.assertAlmostEqual(r[0], 0.25)

    def test_途中で上場廃止したら最後の終値(self):
        H = [120] * 5 + [NAN] * 15
        C = [110] * 4 + [90] + [NAN] * 15
        r, d, w = TR.exit_target(paths(H, C, n=5), days=20)
        self.assertAlmostEqual(r[0], -0.10)
        self.assertEqual((d[0], w[0]), (5, 3))

    def test_終値で判定するとその日の終値で売る(self):
        H = [205] + [150] * 4 + [230] + [150] * 14
        C = [150] * 5 + [220] + [150] * 14
        r, d, w = TR.exit_target(paths(H, C), days=20, on="C")
        self.assertAlmostEqual(r[0], 1.2)
        self.assertEqual(d[0], 6)


class 前の窓の点数で選ぶ(unittest.TestCase):

    def test_すべての列でそれより前の窓の95パーセント点を超えたものだけ(self):
        rng = np.random.default_rng(0)
        n = 2000
        fr = pd.DataFrame({"fold": np.repeat([1, 2], n // 2),
                           "a": rng.random(n), "b": rng.random(n)})
        fr.loc[1500, ["a", "b"]] = [2.0, 2.0]       # 窓2で両方とも飛び抜けて高い
        fr.loc[1501, ["a", "b"]] = [2.0, 0.0]       # 片方だけ
        m = TR.select_prior(fr, ["a", "b"], 95.0)
        self.assertTrue(m[1500])
        self.assertFalse(m[1501])
        self.assertFalse(m[:1000].any())            # 窓1は比べる前の窓が無い


class 一銘柄ずつ(unittest.TestCase):

    def test_持っている間は次を買わず複利で増える(self):
        buy = np.array([0, 5, 30, 30])
        held = np.array([20.0, 10.0, 20.0, 20.0])
        ret = np.array([1.0, 0.5, -0.2, 0.3])
        pr = np.array([1.0, 1.0, 0.1, 0.9])          # 同じ日（30）は priority の高い方
        r = TR.one_at_a_time(buy, held, ret, pr)
        self.assertEqual(r["trades"], 2)             # 0番（0〜19日）と3番（30日）。1番は保有中
        self.assertAlmostEqual(r["multiple"], 2.0 * 1.3)


if __name__ == "__main__":
    unittest.main()
