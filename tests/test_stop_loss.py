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


class 大負けの共通点(unittest.TestCase):

    def test_日内一定の列だけを拾う(self):
        d = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
        fr = pd.DataFrame({"Date": d,
                           "mkt": [1.0] * 3 + [2.0] * 3 + [3.0] * 3,      # 日ごとに同じ
                           "stock": np.arange(9, dtype=float),            # 銘柄ごとに違う
                           "rare": [0.0] * 7 + [1.0, 0.0]})               # まれに立つ旗
        self.assertEqual(E41.date_constant(fr, ["mkt", "stock", "rare"]), {"mkt"})

    def test_除外のしきい値はそれより前の窓だけで決める(self):
        ref = pd.DataFrame({"fold": [1] * 600 + [2] * 600,
                            "x": np.r_[np.arange(600, dtype=float), np.arange(600, dtype=float) + 1000]})
        s = pd.DataFrame({"fold": [2, 2, 2], "x": [100.0, 560.0, np.nan]})
        m = E41.exclusion(s, ref, "x", "top")
        # 窓2 のしきい値は窓1（0〜599）の90パーセンタイル ≒ 539。窓2 自身の値（1000〜）は使わない
        self.assertEqual(m.tolist(), [False, True, False])     # 値が無い行は外さない

    def test_大負けが多い側を拾う(self):
        rng = np.random.default_rng(3)
        n = 6000
        x = rng.normal(size=n)
        # x の上位で大負け（-10% 以下）が多くなるように作る
        r = np.where((x > 1.3) & (rng.random(n) < 0.6), -0.2, rng.normal(0.01, 0.03, n))
        u = pd.DataFrame({"fold": np.repeat(np.arange(2, 8), n // 6), "ret_o1_20": r,
                          "x": x, "noise": rng.normal(size=n)})
        scr = E41.loser_screen(u, ["x", "noise"]).set_index("feature")
        self.assertGreater(scr.loc["x", "top_z"], 3)
        self.assertLess(abs(scr.loc["noise", "top_z"]), 3)


class 運用者の出口案(unittest.TestCase):
    """+20% 指値 / 20日目にプラスなら売る / マイナスなら持ち続けてプラ転で売る / 上限で売る。"""

    def _p(self, rows):
        return paths(rows, [0.0] * len(rows))

    def test_利確は20日目までに届けばちょうど20パーセント(self):
        P = self._p([[QUIET] * 5 + [(110.0, 121.0, 109.0, 118.0)] + [QUIET] * 30])
        r, d, w = E41.plan_exit(P, cap=30, decide=20)
        self.assertAlmostEqual(r[0], 0.20)
        self.assertEqual((d[0], w[0]), (6, 1))

    def test_20日目にプラスならその終値で売る(self):
        P = self._p([[QUIET] * 19 + [(101.0, 104.0, 100.5, 103.0)] + [QUIET] * 10])
        r, d, w = E41.plan_exit(P, cap=30, decide=20)
        self.assertAlmostEqual(r[0], 0.03)
        self.assertEqual((d[0], w[0]), (20, 2))

    def test_マイナスなら持ち続けて買値に戻したら売る(self):
        low = (95.0, 96.0, 94.0, 95.0)
        P = self._p([[QUIET] * 19 + [low] * 5 + [(97.0, 100.5, 96.5, 99.0)] + [QUIET] * 5])
        r, d, w = E41.plan_exit(P, cap=30, decide=20)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertEqual((d[0], w[0]), (25, 3))

    def test_戻らなければ上限の日の終値で売る(self):
        low = (95.0, 96.0, 94.0, 95.0)
        P = self._p([[QUIET] * 19 + [low] * 11])
        r, d, w = E41.plan_exit(P, cap=30, decide=20)
        self.assertAlmostEqual(r[0], -0.05)
        self.assertEqual((d[0], w[0]), (30, 4))

    def test_上限が20日なら20日目の終値で必ず売る(self):
        low = (95.0, 96.0, 94.0, 95.0)
        P = self._p([[QUIET] * 19 + [low] * 11])
        r, d, w = E41.plan_exit(P, cap=20, tp=None)
        self.assertAlmostEqual(r[0], -0.05)
        self.assertEqual(d[0], 20)

    def test_逆指値を重ねられる(self):
        # 20日目にマイナスで持ち越し、23日目に場中で −15% に触れる
        low = (95.0, 96.0, 94.0, 95.0)
        P = self._p([[QUIET] * 19 + [low] * 3 + [(90.0, 91.0, 80.0, 82.0)] + [QUIET] * 7])
        r, d, w = E41.plan_exit(P, cap=30, decide=20, stop=0.15)
        self.assertAlmostEqual(r[0], -0.15)
        self.assertEqual((d[0], w[0]), (23, 5))

    def test_枠3で売った翌日から枠が空く(self):
        cal = pd.Series(pd.bdate_range("2024-01-01", periods=60))
        sig = pd.DataFrame({"Date": [cal[0], cal[0], cal[0], cal[0], cal[5]],
                            "i": [0, 1, 2, 3, 4]})
        ret = np.array([0.10, 0.0, -0.05, 0.2, 0.2])
        day = np.array([5.0, 20.0, 20.0, 20.0, 3.0])      # 0番は 1〜5日目（cal[1]〜cal[5]）
        why = np.array([2, 3, 4, 1, 1], dtype=np.int8)
        r = E41.slot_sim(sig, cal, ret, day, why, slots=3)
        # 4件目（同じ日）は枠が無くて見送り。cal[5] の信号は cal[6] に買う: 0番が空くのは cal[6]
        self.assertEqual((r["taken"], r["skipped"]), (4, 1))


class 窓をずらした_out_of_fold(unittest.TestCase):
    """§10 の切り方違い。Actions で20分学習してから落ちないよう、形だけ先に確かめる。"""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2018-01-01", "2023-12-29")
        n = len(dates) * 4
        d = np.repeat(dates, 4)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        y = (x1 + rng.normal(scale=1.5, size=n) > 1.0).astype(float)
        cls.df = pd.DataFrame({"Code": np.tile(["1", "2", "3", "4"], len(dates)), "Date": d,
                               "f1": x1, "f2": x2, "label": y,
                               "ret_o1_20": rng.normal(0, 0.05, n), "ret_o1_40": rng.normal(0, 0.05, n)})

    def test_ずらすと境界が動く(self):
        a = E41.folds_for(self.df["Date"], 0)
        b = E41.folds_for(self.df["Date"], 2)
        self.assertGreater(len(a), 2)
        self.assertGreater(pd.Timestamp(b[0].test_start), pd.Timestamp(a[0].test_start))
        # テスト窓は重ならない
        for f, g in zip(b, b[1:]):
            self.assertLess(pd.Timestamp(f.test_end), pd.Timestamp(g.test_start))

    def test_3モデルとも回る(self):
        folds = E41.folds_for(self.df["Date"], 2)
        lg = {"objective": "binary", "n_estimators": 20, "learning_rate": 0.1,
              "num_leaves": 7, "verbose": -1, "n_jobs": 1, "random_state": 0}
        for algo, par in (("lgbm", {"params": lg}), ("xgb", {"max_depth": 2}),
                          ("cat", {"depth": 2})):
            o = E41.oof_folds(algo, self.df, ["f1", "f2"], par, 7, folds)
            self.assertEqual(set(o["fold"]) <= {f.index for f in folds}, True, algo)
            self.assertTrue(np.isfinite(o["score"]).all(), algo)
            self.assertIn("ret_o1_20", o.columns)


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
