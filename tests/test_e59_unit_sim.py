#!/usr/bin/env python3
"""
実験59（research/exp/e59_unit_sim.py）の数え方のテスト。重い計算は回さない。

- 過去分布の百分位は「その日より前」だけで出し、過去の行が足りない日は NaN
- 全期間分布の百分位は live_track.pct_of と同じ
- 規則どおりの出口: +20% は指値ちょうど、満了は 20営業日目の終値、途中の玉は NaN
- 1単元の買付額は調整前の始値 × 株数、円の損益 = 買付額 × 収益率
- 投下資本の曲線は買った日から売った日まで（枠が空く日の前日まで）
- 回転率 = 年間買付額 ÷ 最大同時投下資本、稼働率 = Σ保有日 ÷ 枠 ÷ 営業日
- 5モデルの合議（live_track.decide(models=ALL5)）は、1つでも百分位が欠ければ通らない

  python3 tests/test_e59_unit_sim.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import live_track as L  # noqa: E402
import e59_unit_sim as E  # noqa: E402

NAN = float("nan")


class Percentile(unittest.TestCase):
    def test_expanding_uses_only_earlier_days(self):
        dates = ["2021-01-04"] * 3 + ["2021-01-05"] * 2 + ["2021-01-06"]
        scores = [0.1, 0.2, 0.3, 0.25, 0.9, 0.2]
        p = E.pct_expanding(dates, scores, min_hist=3)
        self.assertTrue(np.isnan(p[:3]).all())                # 最初の日は過去が無い
        # 1/5: 過去 {0.1, 0.2, 0.3}。0.25 は 2/3 → 66.7、0.9 は 3/3 → 100.0
        self.assertAlmostEqual(p[3], 66.7)
        self.assertAlmostEqual(p[4], 100.0)
        # 1/6: 過去 {0.1, 0.2, 0.3, 0.25, 0.9}。0.2 より低いのは 0.1 だけ → 20.0（同じ値は数えない）
        self.assertAlmostEqual(p[5], 20.0)

    def test_expanding_min_hist_and_nan(self):
        dates = ["2021-01-04"] * 2 + ["2021-01-05"] * 2 + ["2021-01-06"]
        scores = [0.1, 0.2, NAN, 0.3, 0.15]
        p = E.pct_expanding(dates, scores, min_hist=3)
        self.assertTrue(np.isnan(p[2]))                       # 過去が 2 行 → 判定しない
        self.assertTrue(np.isnan(p[3]))
        self.assertAlmostEqual(p[4], 1 / 3 * 100, places=1)    # 過去 {0.1, 0.2, 0.3}（NaN は入らない）
        p2 = E.pct_expanding(["2021-01-04", "2021-01-05"], [0.5, NAN], min_hist=1)
        self.assertTrue(np.isnan(p2[1]))                      # スコアが NaN の行は NaN

    def test_expanding_matches_brute_force(self):
        rng = np.random.default_rng(0)
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2021-01-04", periods=40), 300))
        scores = rng.random(300)
        p = E.pct_expanding(dates, scores, min_hist=20)
        for i in range(300):
            hist = scores[dates < dates[i]]
            want = round(float((hist < scores[i]).mean()) * 100, 1) if len(hist) >= 20 else NAN
            if np.isnan(want):
                self.assertTrue(np.isnan(p[i]))
            else:
                self.assertAlmostEqual(p[i], want)

    def test_whole_matches_live_track(self):
        s = np.array([0.3, 0.1, NAN, 0.2])
        p = E.pct_whole(s)
        self.assertTrue(np.isnan(p[2]))
        np.testing.assert_allclose(p[[0, 1, 3]], L.pct_of(s[[0, 1, 3]], s[[0, 1, 3]]))


class Exit(unittest.TestCase):
    def test_exec_ret_and_days(self):
        st = pd.Series(["+20%到達", "満了", "保有中", "未約定"])
        rc = pd.Series([35.0, -3.5, 1.0, NAN])
        r = E.exec_ret(st, rc)
        self.assertEqual(r.iloc[0], 20.0)                     # 指値ちょうど（35% まで伸びていても）
        self.assertEqual(r.iloc[1], -3.5)
        self.assertTrue(np.isnan(r.iloc[2]) and np.isnan(r.iloc[3]))
        d = E.held_days(st, pd.Series([4.0, NAN, NAN, NAN]))
        self.assertEqual(d.iloc[0], 4.0)
        self.assertEqual(d.iloc[1], float(L.HOLD))
        self.assertTrue(np.isnan(d.iloc[2]))

    def test_capital_uses_raw_open_times_unit(self):
        taken = pd.DataFrame({"Code": ["10000", "20000"], "buy_date": pd.to_datetime(["2024-07-02", "2024-07-02"]),
                              "status": ["満了", "+20%到達"], "ret_close": [-10.0, 50.0],
                              "hit_day": [NAN, 2.0], "ret": [-9.0, 20.0]})
        # 10000 は後で 1:5 の分割があり、調整後の始値は当時の 1/5。買付額は当時の値段で数える
        raw = pd.DataFrame({"Code": ["10000", "20000", "10000"],
                            "Date": pd.to_datetime(["2024-07-02", "2024-07-02", "2024-07-03"]),
                            "O": [5000.0, 1200.0, 5100.0]})
        t = E.attach_capital(taken, raw, unit=100)
        self.assertEqual(t["capital"].tolist(), [500000.0, 120000.0])
        self.assertAlmostEqual(t["pnl"].iloc[0], -50000.0)    # 50万 × −10%
        self.assertAlmostEqual(t["pnl"].iloc[1], 24000.0)     # 12万 × +20%（指値）
        self.assertAlmostEqual(t["pnl_ma5"].iloc[0], -45000.0)
        self.assertEqual(t["days"].tolist(), [20.0, 2.0])


class Capital(unittest.TestCase):
    def setUp(self):
        self.days = list(pd.bdate_range("2024-01-01", periods=60))

    def test_curve_covers_buy_to_exit(self):
        done = pd.DataFrame({"buy_date": [self.days[1], self.days[2]], "capital": [100.0, 50.0],
                             "days": [3.0, 20.0]})
        c = E.capital_curve(done, self.days)
        self.assertEqual(c.iloc[0], 0.0)
        self.assertEqual(c.iloc[1], 100.0)                    # 買った日
        self.assertEqual(c.iloc[2], 150.0)                    # 2本目が重なる
        self.assertEqual(c.iloc[3], 150.0)                    # 1本目の最終日（3日目）
        self.assertEqual(c.iloc[4], 50.0)                     # 枠が空く日（buy+3）には外れている
        self.assertEqual(c.iloc[21], 50.0)                    # 2本目の20日目（index 2+19）
        self.assertEqual(c.iloc[22], 0.0)

    def test_stats_turnover_and_utilization(self):
        done = pd.DataFrame({"buy_date": [self.days[1], self.days[2]], "capital": [100.0, 50.0],
                             "days": [3.0, 20.0], "ret_exec": [20.0, -4.0], "pnl": [20.0, -2.0],
                             "ret": [20.0, -3.0], "status": ["+20%到達", "満了"]})
        c = E.capital_curve(done, self.days)
        lo, hi = self.days[0], self.days[-1]
        s = E.stats(done, c, lo, hi, slots=3, n_full=4, n_pass=9, n_open=1)
        years = (hi - lo).days / E.YEAR_DAYS
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["win"], 50.0)
        self.assertAlmostEqual(s["hit"], 50.0)
        self.assertAlmostEqual(s["pnl_total"], 18.0)
        self.assertAlmostEqual(s["buy_total"], 150.0)
        self.assertAlmostEqual(s["cap_max"], 150.0)
        self.assertAlmostEqual(s["turnover"], 150.0 / years / 150.0)
        self.assertAlmostEqual(s["util"], 23.0 / (3 * 60) * 100)
        self.assertAlmostEqual(s["hold_mean"], 11.5)
        self.assertEqual((s["n_full"], s["n_pass"], s["n_open"]), (4, 9, 1))
        self.assertAlmostEqual(s["mean_ma5"], 8.5)

    def test_stats_empty(self):
        s = E.stats(E.empty_trades(), pd.Series(dtype=float), self.days[0], self.days[-1], slots=3, n_pass=2)
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["n_pass"], 2)
        self.assertTrue(np.isnan(s["mean"]))
        self.assertEqual(s["pnl_total"], 0.0)


class FiveModels(unittest.TestCase):
    def rows(self, pcts):
        return pd.DataFrame([{"Date": pd.Timestamp("2026-09-16"), "Code": f"{1000 + i}0", "score": 0.5 - i * 0.01,
                              "p_lgbm": p[0], "p_xgb": p[1], "p_cat": p[2], "p_logit": p[3], "p_mlp": p[4]}
                             for i, p in enumerate(pcts)])

    def test_all_five_must_pass(self):
        rows = self.rows([(99, 99, 99, 99, 95),      # 通る（最小 95）
                          (99, 99, 99, 94.9, 99),    # 通らない（logit 94.9）
                          (99, 99, 99, NAN, 99),     # 通らない（欠け）
                          (99, 99, 99, 99, 99)])     # 通る（最小 99 → 1番）
        r, days, picks = L.decide(rows, agree=95.0, min_break=0, top_k=3, models=L.ALL5)
        self.assertEqual(picks["Code"].tolist(), ["10030", "10000"])
        self.assertEqual(int(days.iloc[0]["n_missing"]), 1)
        self.assertEqual(int(days.iloc[0]["n_pass"]), 2)
        # 3モデルなら logit / mlp を見ないので 4件とも通る
        _, _, p3 = L.decide(rows, agree=95.0, min_break=0, top_k=4, models=L.BOOST)
        self.assertEqual(len(p3), 4)

    def test_patterns(self):
        self.assertEqual([p[0] for p in E.PATTERNS],
                         ["GBDT3 97.5以上", "GBDT3 95以上", "全5モデル 95以上", "全5モデル 90以上"])
        self.assertEqual(E.PATTERNS[0][1], L.BOOST)
        self.assertEqual(E.PATTERNS[2][1], L.ALL5)
        self.assertEqual(E.UNIT, 100)
        self.assertEqual(L.ALL5, ("lgbm", "xgb", "cat", "logit", "mlp"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
