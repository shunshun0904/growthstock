#!/usr/bin/env python3
"""
実験90（research/exp/e90_wasserstein_knn_pick.py）。

- 分位関数: 値が60個なら並べ替えた値そのもの。W1 は「分布関数の差の面積」で別に計算した値と一致する
- 分布: リターンの形は平均0・標準偏差1。出来高が一定なら比率の log はすべて0。履歴が足りなければ NaN。結果が出そろう日は
  その銘柄の日足で i+20 本目。ブレイクの日より後の日足を変えても分布は変わらない
- 近傍の母集団: 判断日と同じ日に出そろう行は入れない（先読みしない）。記録した母集団の大きさが、別に数えた件数と一致し、
  候補の行は自分の母集団に入らない
- 仕込んだ合成データ（形の違う銘柄だけ収益を高くする）で、K が今の並べ方と無作為を上回り「効く」。仕込まなければ「効く」に
  ならない
- 報告が全部の段を出し、銘柄コードを出さない

  python3 tests/test_e90_wasserstein_knn_pick.py
"""
import contextlib
import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e79_model_clusters as E79  # noqa: E402
import e90_wasserstein_knn_pick as E  # noqa: E402
import models as M  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402

#: 合成データは1日の行が少ないので、線を下げて判断日を増やす（本番の線 95 / 85 は台本の既定のまま）
LINE_T, POOL_T = 80.0, 70.0


def w1_cdf(x, y) -> float:
    """1次元の W1 を、分布関数の差の面積で（台本とは別の計算）。"""
    x, y = np.sort(np.asarray(x, dtype=float)), np.sort(np.asarray(y, dtype=float))
    allv = np.sort(np.concatenate([x, y]))
    fx = np.searchsorted(x, allv[:-1], side="right") / len(x)
    fy = np.searchsorted(y, allv[:-1], side="right") / len(y)
    return float(np.sum(np.abs(fx - fy) * np.diff(allv)))


def bars_for(codes, days, kinds, seed=0) -> pd.DataFrame:
    """kinds[c] が True の銘柄は右に歪んだリターンと荒い出来高、False は正規のリターンと穏やかな出来高。"""
    rng = np.random.default_rng(seed)
    n = len(days)
    parts = []
    for c in codes:
        if kinds[c]:
            r = rng.exponential(0.02, size=n) - 0.02
            v = np.exp(rng.normal(10.0, 1.0, size=n))
        else:
            r = rng.normal(0.0, 0.02, size=n)
            v = np.exp(rng.normal(10.0, 0.2, size=n))
        close = 1000.0 * np.cumprod(1.0 + r)
        parts.append(pd.DataFrame({"Date": days, "Code": c, "AdjO": close, "AdjH": close, "AdjC": close, "C": close,
                                   "AdjVo": v, "Vo": v}))
    return pd.concat(parts, ignore_index=True)


def case(planted: float, seed: int = 0):
    """実験83 の合成データに日足を付ける。形の違わない銘柄の収益だけ planted 下げる（+10% の上限に当たらない向き）。"""
    df, folds = T83.synth_with_exits(seed=seed)
    codes = sorted(df["Code"].unique())
    rng = np.random.default_rng(seed + 11)
    kinds = {c: bool(rng.random() < 0.5) for c in codes}
    bars = bars_for(codes, pd.bdate_range("2020-09-01", "2024-12-31"), kinds, seed=seed + 3)
    t = df["Code"].map(kinds).to_numpy(dtype=bool)
    df[E.TP] = df[E.TP] - planted * (~t)
    df[E.HIT] = df[E.TP] >= E.TARGET
    df["score"] = df[f"score_{M.BASELINE}"]
    df["vol_20d"] = np.where(t, 3.0, 1.5)
    dist = E.event_distributions(bars, df[["Code", "Date"]])
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    return df, dist, split, starts, kinds


class TestQuantileAndW1(unittest.TestCase):
    def test_sorted_when_60(self):
        x = np.random.default_rng(0).normal(size=60)
        np.testing.assert_array_equal(E.quantile_vector(x), np.sort(x))

    def test_w1_matches_cdf_area(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            a, b = rng.standard_t(3, 60), rng.normal(0.3, 2.0, 60)
            got = float(E.w1(E.quantile_vector(a), E.quantile_vector(b)[None, :])[0])
            self.assertAlmostEqual(got, w1_cdf(a, b), places=10)

    def test_w1_unequal_sizes_close(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=120), rng.normal(1.0, 1.0, size=55)
        got = float(E.w1(E.quantile_vector(a), E.quantile_vector(b)[None, :])[0])
        self.assertAlmostEqual(got, w1_cdf(a, b), delta=0.1)


class TestDistributions(unittest.TestCase):
    @staticmethod
    def one_code(n=200, seed=4):
        days = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(seed)
        close = 500.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, size=n))
        return days, pd.DataFrame({"Date": days, "Code": "10010", "AdjO": close, "AdjH": close, "AdjC": close,
                                   "C": close, "AdjVo": 1000.0, "Vo": 1000.0})

    def test_shape_volume_done(self):
        days, bars = self.one_code()
        keys = pd.DataFrame({"Code": ["10010", "10010", "10010", "99990"],
                             "Date": [days[150], days[30], days[190], days[150]]})
        d = E.event_distributions(bars, keys)
        q = d["QR"][0]
        self.assertAlmostEqual(float(q.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(q.std()), 1.0, places=10)
        np.testing.assert_allclose(d["QV"][0], 0.0, atol=1e-12)          # 出来高が一定なら比率は1
        self.assertTrue(np.isnan(d["QR"][1]).all() and np.isnan(d["QV"][1]).all())   # 履歴が足りない
        self.assertEqual(pd.Timestamp(d["done"][0]), days[150 + E.HOLD])
        self.assertTrue(np.isnat(d["done"][2]))                         # 20本先の日足が無い
        self.assertTrue(np.isfinite(d["QR"][2]).all())
        self.assertTrue(np.isnan(d["QR"][3]).all() and np.isnat(d["done"][3]))      # 日足の無い銘柄

    def test_uses_only_past_bars(self):
        days, bars = self.one_code(seed=5)
        keys = pd.DataFrame({"Code": ["10010"], "Date": [days[150]]})
        a = E.event_distributions(bars, keys)
        b2 = bars.copy()
        later = b2["Date"] > days[150]
        b2.loc[later, ["AdjC", "C"]] *= 3.0
        b2.loc[later, ["AdjVo", "Vo"]] *= 10.0
        b = E.event_distributions(b2, keys)
        np.testing.assert_array_equal(a["QR"], b["QR"])
        np.testing.assert_array_equal(a["QV"], b["QV"])

    def test_volume_ratio_uses_previous_days(self):
        days, bars = self.one_code(seed=6)
        bars.loc[bars.index >= 120, ["AdjVo", "Vo"]] = 2000.0             # 120本目から出来高が2倍
        keys = pd.DataFrame({"Code": ["10010"], "Date": [days[119]]})
        q = E.event_distributions(bars, keys)["QV"][0]
        np.testing.assert_allclose(q, 0.0, atol=1e-12)                    # その日までは一定（比率1）
        keys2 = pd.DataFrame({"Code": ["10010"], "Date": [days[179]]})          # 窓 120〜179 本目はすべて2倍の日
        q2 = E.event_distributions(bars, keys2)["QV"][0]
        self.assertGreater(float(q2.min()), 0.0)                           # 2倍になった日は前の平均より大きい


class TestPoolHelpers(unittest.TestCase):
    def test_done_before_excludes_same_day(self):
        d = np.array(["2023-01-05", "2023-01-06", "2023-01-06", "2023-01-10"], dtype="datetime64[ns]")
        self.assertEqual(E.done_before(d, "2023-01-06"), 1)
        self.assertEqual(E.done_before(d, "2023-01-07"), 3)
        self.assertEqual(E.done_before(d, "2023-01-01"), 0)

    def test_pair_median(self):
        Q = np.array([[0.0] * 60, [1.0] * 60, [3.0] * 60])
        self.assertAlmostEqual(E.pair_median(Q), 2.0)                      # 組の W1 は 1・3・2
        self.assertTrue(np.isnan(E.pair_median(Q[:1])))

    def test_knn(self):
        est, nn = E.knn(np.array([0.5, 0.1, 0.9, 0.2]), np.array([10.0, 1.0, 100.0, 3.0]), 2)
        self.assertAlmostEqual(est, 2.0)
        self.assertEqual(sorted(nn.tolist()), [1, 3])


class TestPlanted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.dist, cls.split, cls.starts, cls.kinds = case(planted=6.0)
        cls.c, cls.info = E.collect(cls.df, cls.dist, cls.split, cls.starts, line=LINE_T, pool_line=POOL_T)

    def test_counts(self):
        self.assertGreater(self.info["days"], 40)
        self.assertEqual(self.info["days"], self.c["Date"].nunique())
        self.assertEqual(sum(self.info["per_window"].values()), self.info["days"])
        self.assertEqual(self.info["days2"], self.info["days"] + self.info["days_small_pool"])

    def test_pool_is_past_only(self):
        """記録した母集団の大きさ = 窓の物差しで線以上・結果が判断日より前に出そろった行の数。候補は自分の母集団に入らない。"""
        QR, QV, done = self.dist["QR"], self.dist["QV"], self.dist["done"]
        tp = self.df[E.TP].to_numpy(dtype=float)
        ok = np.isfinite(tp) & np.isfinite(QR).all(axis=1) & np.isfinite(QV).all(axis=1)
        mins = E.window_min(self.df, self.split)
        for (d, w), g in list(self.c.groupby(["Date", "window"]))[:20]:
            U, mn = mins[w]
            pool = U[(mn >= POOL_T) & ok[U] & ~np.isnat(done[U])]
            day = np.datetime64(pd.Timestamp(d), "ns")
            self.assertEqual(int(g["pool_n"].iat[0]), int((done[pool] < day).sum()))
            rows = g["row"].to_numpy()
            self.assertTrue((done[rows][~np.isnat(done[rows])] > day).all())

    def test_knn_beats_order_and_random(self):
        ev = E.evaluate(self.c, f"est_RV_{E.K_MAIN}", draws=500)
        self.assertTrue(ev["effective"], ev["conds"])
        self.assertGreater(ev["K"]["mean"], ev["M"]["mean"])
        self.assertLess(ev["p_K"], 0.05)

    def test_estimates_follow_planted_kind(self):
        kind = self.df["Code"].iloc[self.c["row"].to_numpy()].map(self.kinds).to_numpy(dtype=bool)
        est = self.c[f"est_RV_{E.K_MAIN}"].to_numpy(dtype=float)
        self.assertGreater(est[kind].mean() - est[~kind].mean(), 3.0)

    def test_report_sections_and_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.c, self.info, draws=200, perm=30)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ まとめ", "判定", "後工程として効く"):
            self.assertIn(tag, out)
        for k in ("main", "variants", "rho"):
            self.assertIn(k, s)
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


class TestNull(unittest.TestCase):
    def test_no_plant_is_not_effective(self):
        df, dist, split, starts, _ = case(planted=0.0, seed=1)
        c, info = E.collect(df, dist, split, starts, line=LINE_T, pool_line=POOL_T)
        ev = E.evaluate(c, f"est_RV_{E.K_MAIN}", draws=500)
        self.assertGreater(ev["days"], 40)
        self.assertGreater(ev["p_K"], 0.05)
        self.assertFalse(ev["effective"])

    def test_empty(self):
        c = pd.DataFrame(columns=["Date", "window", "row", "min15", "score", E.TP, E.HIT, "est_RV_50"])
        self.assertEqual(E.evaluate(c, "est_RV_50")["days"], 0)
        self.assertEqual(E.within_day_rho(c, "est_RV_50", E.TP)["n"], 0)


if __name__ == "__main__":
    unittest.main()
