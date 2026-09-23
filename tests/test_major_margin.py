"""本ブレイク: 信用残から作る特徴量（research/major/margin_feats.py）と、上下どちらが先か（trade.touch_order）。"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "major"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import margin_feats as MF  # noqa: E402
import trade as TR  # noqa: E402

NAN = float("nan")


def bars_of(code, dates, vo, c=100.0, adjc=100.0):
    return pd.DataFrame({"Code": code, "Date": dates, "C": c, "AdjC": adjc, "AdjVo": vo})


class 信用残の特徴量(unittest.TestCase):

    def setUp(self):
        self.days = pd.bdate_range("2024-01-01", periods=60)
        fridays = [d for d in self.days if d.dayofweek == 4]
        self.f0 = fridays[6]                          # 使う記録
        self.f4 = fridays[2]                          # その4週前
        self.bars = bars_of("A", self.days, 1000.0)
        self.margin = pd.DataFrame({
            "Code": "A", "Date": fridays,
            "LongVol": [2500.0 if f == self.f4 else 5000.0 if f == self.f0 else 3000.0 for f in fridays],
            "ShrtVol": 1000.0})

    def build(self, t):
        return MF.build(pd.DataFrame({"Code": ["A"], "Date": [t]}), self.bars, self.margin).iloc[0]

    def test_記録日から6日たてば使う(self):
        r = self.build(self.f0 + pd.Timedelta(days=6))        # 翌週の木曜
        self.assertAlmostEqual(r["m_long_days"], 5.0)
        self.assertAlmostEqual(r["m_short_days"], 1.0)
        self.assertAlmostEqual(r["m_long_chg4"], 5000.0 / 2500.0 - 1.0)

    def test_公表前の記録は使わない(self):
        r = self.build(self.f0 + pd.Timedelta(days=5))        # 翌週の水曜は1つ前の記録
        self.assertAlmostEqual(r["m_long_days"], 3.0)

    def test_分割をまたいでも株数をそろえる(self):
        # 記録日の時点では分割前（生の株価200・調整後100 = 1株が後の2株）。出来高は調整後で1,000株/日
        bars = bars_of("A", self.days, 1000.0, c=np.where(self.days <= self.f0, 200.0, 100.0), adjc=100.0)
        r = MF.build(pd.DataFrame({"Code": ["A"], "Date": [self.f0 + pd.Timedelta(days=6)]}), bars, self.margin).iloc[0]
        self.assertAlmostEqual(r["m_long_days"], 10.0)        # 生の5,000株 = 調整後の10,000株

    def test_古すぎる記録は使わない(self):
        t = self.f0 + pd.Timedelta(days=6 + MF.STALE_DAYS + 7)
        margin = self.margin[self.margin["Date"] <= self.f0]
        r = MF.build(pd.DataFrame({"Code": ["A"], "Date": [t]}), self.bars, margin).iloc[0]
        self.assertTrue(np.isnan(r["m_long_days"]))

    def test_行の順番を保つ(self):
        ts = [self.f0 + pd.Timedelta(days=6), self.f0 + pd.Timedelta(days=5)]
        r = MF.build(pd.DataFrame({"Code": ["A", "A"], "Date": ts}), self.bars, self.margin)
        self.assertEqual(r["m_long_days"].round(6).tolist(), [5.0, 3.0])


def hl(H, L, entry=100.0):
    return {"H": np.asarray(H, dtype=float)[None, :], "L": np.asarray(L, dtype=float)[None, :],
            "entry": np.array([entry])}


class 上下どちらが先か(unittest.TestCase):

    def order(self, H, L):
        c, du, dd = TR.touch_order(hl(H, L), up=1.3, down=0.8, days=len(H))
        return int(c[0]), int(du[0]), int(dd[0])

    def test_上が先(self):
        self.assertEqual(self.order([110, 131, 100, 100], [100, 100, 79, 70]), (1, 2, 3))

    def test_下が先で後から上に届く(self):
        self.assertEqual(self.order([100, 100, 135, 100], [95, 79, 90, 90]), (2, 3, 2))

    def test_下が先で上に届かない(self):
        self.assertEqual(self.order([100, 120, 110, 100], [79, 90, 90, 90]), (3, 0, 1))

    def test_同じ日に両方(self):
        self.assertEqual(self.order([100, 131, 100], [95, 75, 90]), (4, 2, 2))

    def test_どちらも無し_欠けた日は数えない(self):
        self.assertEqual(self.order([100, 129, NAN], [95, 81, NAN]), (5, 0, 0))

    def test_方向のラベル(self):
        y = TR.direction_label(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(y[:3].tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(y[3:]).all())


def ohlc(O, H, L, C, entry=100.0):
    a = {k: np.asarray(v, dtype=float)[None, :] for k, v in (("O", O), ("H", H), ("L", L), ("C", C))}
    return {**a, "entry": np.array([entry]), "n": np.array([len(C)])}


class 利確と損切りの両方(unittest.TestCase):

    def exit(self, O, H, L, C):
        r, d, w = TR.exit_bracket(ohlc(O, H, L, C), up=1.3, down=0.8, days=len(C))
        return round(float(r[0]), 6), int(d[0]), int(w[0])

    def test_利確が先(self):
        self.assertEqual(self.exit([100, 110, 120], [110, 131, 125], [95, 100, 70], [105, 125, 75]), (0.3, 2, 1))

    def test_損切りが先(self):
        self.assertEqual(self.exit([100, 95, 90], [105, 96, 140], [96, 79, 88], [98, 85, 130]), (-0.2, 2, 4))

    def test_窓を開けて下げたら寄りで売る(self):
        self.assertEqual(self.exit([100, 70, 75], [105, 72, 80], [96, 65, 70], [98, 70, 78]), (-0.3, 2, 4))

    def test_同じ日に両方なら損切り(self):
        self.assertEqual(self.exit([100, 100, 100], [105, 135, 100], [96, 75, 100], [98, 100, 100]), (-0.2, 2, 4))

    def test_どちらにも触れなければ期限の終値(self):
        self.assertEqual(self.exit([100, 100, 100], [105, 125, 110], [96, 85, 90], [98, 110, 108]), (0.08, 3, 2))


if __name__ == "__main__":
    unittest.main()
