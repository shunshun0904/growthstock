#!/usr/bin/env python3
"""
実験41（損切りの設計）の値動きの表と、損切り・利確の約定の単体テスト。

結論は「どの値段で・どの日に手仕舞ったか」の数え方にそのまま乗るので、
約定の仮定（docs と e41 の冒頭に書いたもの）をここで固定しておく。
  - 窓を開けて損切り線を割ったら、損切り線ではなく寄値で約定する
  - 場中に触れたら損切り線ちょうど
  - 利確は +20% ちょうど（飛び越えても）
  - 同じ日に両方触れたら損切りが先
  - 終値で判定する損切りは、次に寄りが付いた日の寄値
  - 日数で切るのは、k日目の終値で判定して次の寄値
  - どれにも掛からなければ N日保有（ret_o1_N と同じ出口）

  python3 tests/test_stop_loss.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lab  # noqa: E402
import ops_rule as OR  # noqa: E402
import e41_stop_loss as E41  # noqa: E402

NAN = float("nan")


def paths(rows, x):
    """
    rows: 行ごとに [(O, H, L, C), ...]（1日目から）。買値は1日目の O。
    x:    期間満了（k日保有。k = 日数）の収益。simulate(P, k) の出口になる
    """
    n = len(rows)
    k = max(len(r) for r in rows)
    P = {c: np.full((n, k), np.nan) for c in "OHLC"}
    for i, r in enumerate(rows):
        for j, (o, h, lo, c) in enumerate(r):
            P["O"][i, j], P["H"][i, j], P["L"][i, j], P["C"][i, j] = o, h, lo, c
    P["entry"] = P["O"][:, 0].copy()
    P[f"x{k}"] = np.asarray(x, dtype=float)
    return P


QUIET = (100.0, 101.0, 99.0, 100.0)      # 損切り −3% にも利確 +20% にも触れない日


class 約定の仮定(unittest.TestCase):

    def test_窓を開けて割ったら寄値で約定する(self):
        P = paths([[QUIET, (95.0, 96.0, 94.0, 95.5), QUIET]], [0.01])
        ret, day, why, _ = E41.simulate(P, 3, stop=0.03)
        self.assertAlmostEqual(ret[0], -0.05)
        self.assertEqual((day[0], why[0]), (2, 1))

    def test_場中に触れたら損切り線ちょうど(self):
        P = paths([[(100.0, 100.5, 96.5, 99.0), QUIET]], [0.01])
        ret, day, why, _ = E41.simulate(P, 2, stop=0.03)
        self.assertAlmostEqual(ret[0], -0.03)
        self.assertEqual((day[0], why[0]), (1, 1))      # 買った日のうちに触れても数える

    def test_利確は飛び越えても20パーセントちょうど(self):
        P = paths([[QUIET, (125.0, 130.0, 124.0, 128.0), QUIET]], [0.01])
        ret, day, why, _ = E41.simulate(P, 3, tp=0.20)
        self.assertAlmostEqual(ret[0], 0.20)
        self.assertEqual((day[0], why[0]), (2, 2))

    def test_同じ日に両方触れたら損切りが先(self):
        P = paths([[QUIET, (100.0, 121.0, 96.0, 110.0), QUIET]], [0.01])
        ret, day, why, both = E41.simulate(P, 3, stop=0.03, tp=0.20)
        self.assertAlmostEqual(ret[0], -0.03)
        self.assertEqual((why[0], both), (1, 1))

    def test_終値で割ったら次の寄値(self):
        # 2日目に終値で割る -> 3日目は売買不成立（寄りが無い） -> 4日目の寄りで売る
        P = paths([[QUIET, (99.0, 99.5, 95.0, 96.0), (NAN, NAN, NAN, NAN),
                    (98.0, 99.0, 97.5, 98.5), QUIET]], [0.01])
        ret, day, why, _ = E41.simulate(P, 5, stop=0.03, mode="close")
        self.assertAlmostEqual(ret[0], -0.02)
        self.assertEqual((day[0], why[0]), (4, 1))

    def test_終値で判定なら場中のヒゲでは切らない(self):
        P = paths([[QUIET, (99.0, 99.5, 90.0, 99.0), QUIET]], [0.01])
        ret, _, why, _ = E41.simulate(P, 3, stop=0.03, mode="close")
        self.assertEqual(why[0], 0)
        self.assertAlmostEqual(ret[0], 0.01)

    def test_日数で切るのはk日目の終値で判定して次の寄値(self):
        P = paths([[QUIET, (100.0, 100.5, 98.5, 99.0), (99.5, 100.0, 99.0, 99.8), QUIET],
                   [QUIET, (100.0, 102.0, 99.5, 101.0), QUIET, QUIET]], [0.01, 0.02])
        ret, day, why, _ = E41.simulate(P, 4, tstop=(2, 0.0))
        self.assertAlmostEqual(ret[0], -0.005)          # 2日目 99.0 < 100 -> 3日目の寄り 99.5
        self.assertEqual((day[0], why[0]), (3, 3))
        self.assertAlmostEqual(ret[1], 0.02)            # 2日目 101 >= 100 -> 持ち切り
        self.assertEqual(why[1], 0)

    def test_どれにも掛からなければN日保有(self):
        P = paths([[QUIET] * 5], [0.037])
        ret, day, why, _ = E41.simulate(P, 5, stop=0.03, tp=0.20)
        self.assertAlmostEqual(ret[0], 0.037)
        self.assertEqual((day[0], why[0]), (5, 0))

    def test_損切り幅は行ごとに変えられる(self):
        # σ 基準: 1行目は −3%、2行目は −6%
        day2 = (100.0, 100.0, 95.0, 96.0)
        P = paths([[QUIET, day2], [QUIET, day2]], [0.0, 0.0])
        ret, _, why, _ = E41.simulate(P, 2, stop=np.array([0.03, 0.06]))
        self.assertAlmostEqual(ret[0], -0.03)
        self.assertEqual(why[1], 0)


class 値動きの表(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        rows = []
        for code, n in (("11110", 90), ("22220", 70)):
            dates = pd.bdate_range("2024-01-01", periods=n)
            c = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
            o = c * np.exp(rng.normal(0, 0.005, n))
            h = np.maximum(o, c) * 1.01
            lo = np.minimum(o, c) * 0.99
            b = pd.DataFrame({"Date": dates, "Code": code, "AdjO": o, "AdjH": h,
                              "AdjL": lo, "AdjC": c})
            b.loc[10, ["AdjO", "AdjH", "AdjL", "AdjC"]] = np.nan      # 売買不成立の日
            rows.append(b)
        self.bars = pd.concat(rows, ignore_index=True).sort_values(
            ["Code", "Date"]).reset_index(drop=True)

    def test_20日保有が物差しret_o1_20と一致する(self):
        keys = self.bars[["Code", "Date"]]
        P = E41.forward(self.bars, keys)
        ref = lab.realized_returns(self.bars).set_index(["Code", "Date"])["ret_o1_20"]
        want = ref.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        both = np.isfinite(want) & np.isfinite(P["x20"])
        self.assertGreater(both.sum(), 50)
        np.testing.assert_allclose(P["x20"][both], want[both], rtol=0, atol=1e-12)
        # 片方だけ欠測、は無い
        self.assertFalse((np.isfinite(want) ^ np.isfinite(P["x20"])).any())

    def test_銘柄の切れ目をまたがない(self):
        last = self.bars[self.bars["Code"] == "11110"].tail(3)[["Code", "Date"]]
        P = E41.forward(self.bars, last)
        # 最後から3本目: 先は2本だけ。3日目以降は隣の銘柄の足ではなく欠測
        self.assertTrue(np.isfinite(P["C"][0, :2]).all())
        self.assertTrue(np.isnan(P["C"][0, 2:]).all())
        self.assertTrue(np.isnan(P["x20"]).all())

    def test_最大上昇と最大下落は買値から(self):
        P = paths([[(100.0, 102.0, 99.0, 101.0), (101.0, 110.0, 100.0, 108.0),
                    (108.0, 109.0, 92.0, 95.0)]], [0.0])
        ex = E41.excursions(P, 3)
        self.assertAlmostEqual(ex["mfe3"][0], 0.10)
        self.assertAlmostEqual(ex["mae3"][0], -0.08)
        self.assertEqual((ex["day_mfe3"][0], ex["day_mae3"][0]), (2, 3))
        self.assertAlmostEqual(ex["pre_peak_dd3"][0], -0.01)       # 高値の日までの最安値は 99
        self.assertEqual(E41.touch_day(P, 0.05, 3)[0], 3)
        self.assertTrue(np.isnan(E41.touch_day(P, 0.10, 3)[0]))


class 選んだ行を返す(unittest.TestCase):
    """keep=True は行を足すだけで、集計の数字は変えない。"""

    def test_数字は同じ(self):
        rng = np.random.default_rng(1)
        n = 3000
        base = pd.DataFrame({"Code": [f"{i:05d}" for i in range(n)],
                             "Date": pd.Timestamp("2024-01-01"),
                             "fold": np.repeat([1, 2, 3], n // 3),
                             "label": rng.integers(0, 2, n),
                             "ret_o1_20": rng.normal(0, 0.1, n),
                             "ret_o1_40": rng.normal(0, 0.1, n)})
        oofs = {a: base.assign(score=rng.random(n)) for a in OR.BOOST}
        a = OR.consensus(oofs, 80.0)
        b = OR.consensus(oofs, 80.0, keep=True)
        rows = b.pop("rows")
        self.assertEqual(a, b)
        self.assertEqual(len(rows), a["n"])
        self.assertTrue((rows["fold"] > 1).all())


if __name__ == "__main__":
    unittest.main()
