#!/usr/bin/env python3
"""
実験91（research/exp/e91_expectation_monitor.py）。

- 分布の比べ方: W1 = 下側 + 上側、平均(過去) − 平均(直近) = 下側 − 上側、内訳の合計 = 下側。手で数えた例と一致する
- 無作為の下側: 過去と同じ分布から取った直近は合図が出にくく、下にずらした直近は合図が出る
- 結果が出そろう日: その銘柄の日足でブレイクの日から20本目。日足が足りなければ NaT
- 前向きの百分位の3本の最小は、実験87・88 の作り（E87.build の min_K15）と一致する
- 見張り: 仕込んだ悪い時期の後に合図が多く出る。月末 t の数字は、t より後に出そろう結果を書き換えても変わらない（先読みしない）
- 下限の当たり・合図の比べ・入力のずれ・報告（全部の段を出し、銘柄コードを出さない）

  python3 tests/test_e91_expectation_monitor.py
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
import e87_monthly_return_search as E87  # noqa: E402
import e91_expectation_monitor as E  # noqa: E402
import test_e83_tp10_rule as T83  # noqa: E402


def pop_case(seed=0, bad=0.0, months=48, per_month=12, bad_from=24, bad_len=8):
    """月に per_month 件の規則の行。bad_from か月目から bad_len か月だけ収益を bad 下げ、vol_20d を上げる。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=months * 21)
    parts = []
    for m in range(months):
        d = np.sort(rng.choice(days[m * 21:(m + 1) * 21], size=per_month))
        hit = bad_from <= m < bad_from + bad_len
        r = np.minimum(rng.normal(3.0, 7.0, size=per_month) - (bad if hit else 0.0), 10.0)
        parts.append(pd.DataFrame({
            "Code": [f"{7000 + (m * per_month + i) % 300}0" for i in range(per_month)], "Date": d, E.TP: r,
            "vol_20d": rng.normal(2.0 + (1.5 if hit else 0.0), 0.3, size=per_month),
            "ret_20d": rng.normal(5.0, 2.0, size=per_month), "log_market_cap": rng.normal(10.0, 1.0, size=per_month),
            "log_trading_value": rng.normal(0.0, 1.0, size=per_month), "pct15": rng.uniform(95, 100, size=per_month)}))
    pop = pd.concat(parts, ignore_index=True)
    pop["done"] = pop["Date"] + pd.Timedelta(days=28)
    return pop


class TestAreas(unittest.TestCase):
    def test_hand_example(self):
        a = E.cdf_areas([0.0, 1.0], [2.0, 3.0], ranges=((-np.inf, 1.0, "a"), (1.0, np.inf, "b")))
        self.assertAlmostEqual(a["w1"], 2.0)
        self.assertAlmostEqual(a["down"], 2.0)
        self.assertAlmostEqual(a["up"], 0.0)
        np.testing.assert_allclose(a["down_parts"], [0.5, 1.5])

    def test_identities(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            q = np.minimum(rng.normal(rng.normal(), 7, size=rng.integers(5, 60)), 10)
            r = np.minimum(rng.normal(3, 7, size=rng.integers(20, 300)), 10)
            a = E.cdf_areas(q, r)
            self.assertAlmostEqual(a["w1"], a["down"] + a["up"], places=9)
            self.assertAlmostEqual(r.mean() - q.mean(), a["down"] - a["up"], places=9)
            self.assertAlmostEqual(sum(a["down_parts"]), a["down"], places=9)
            self.assertGreaterEqual(a["down"] + 1e-12, r.mean() - q.mean())          # 平均の悪化は下側以下

    def test_parts_by_return_range(self):
        """負けの2割が −3% から −20% へ動くと、面積は分布関数が食い違う範囲（−20〜−10% と −10〜−3%）に分かれる。"""
        r = np.r_[np.full(80, 5.0), np.full(20, -3.0)]
        q = np.r_[np.full(80, 5.0), np.full(20, -20.0)]
        a = E.cdf_areas(q, r)
        self.assertAlmostEqual(a["down"], 0.2 * 17)
        np.testing.assert_allclose(a["down_parts"], [0.2 * 10, 0.2 * 7, 0.0])
        self.assertAlmostEqual(a["up"], 0.0)

    def test_empty(self):
        self.assertTrue(np.isnan(E.cdf_areas([], [1.0, 2.0])["w1"]))


class TestNull(unittest.TestCase):
    def test_same_vs_shifted(self):
        rng = np.random.default_rng(2)
        r = np.minimum(rng.normal(3, 7, size=400), 10)
        null = E.null_down(r, 30, draws=400, seed=3)
        same = E.cdf_areas(np.minimum(rng.normal(3, 7, size=30), 10), r)["down"]
        low = E.cdf_areas(np.minimum(rng.normal(-5, 7, size=30), 10), r)["down"]
        self.assertLess(same, np.quantile(null, 0.99))
        self.assertGreater(low, np.quantile(null, E.ALARM_Q))


class TestFinalize(unittest.TestCase):
    def test_done_dates(self):
        days = pd.bdate_range("2023-01-02", periods=60)
        bars = pd.DataFrame({"Date": days, "Code": "10010", "AdjO": 1.0, "AdjH": 1.0, "AdjC": 1.0})
        keys = pd.DataFrame({"Code": ["10010", "10010", "99990", "10010"],
                             "Date": [days[10], days[45], days[10], pd.Timestamp("2023-01-01")]})
        d = E.finalize_dates(bars, keys)
        self.assertEqual(pd.Timestamp(d[0]), days[10 + E.HOLD])
        self.assertTrue(np.isnat(d[1]))                                             # 20本先が無い
        self.assertTrue(np.isnat(d[2]))                                             # 日足の無い銘柄
        self.assertTrue(np.isnat(d[3]))                                             # 日足の無い日


class TestForward(unittest.TestCase):
    def test_min15_matches_e87_build(self):
        df, folds = T83.synth_with_exits()
        split = E79.window_split(df, folds)
        ft = E87.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        out = E.attach_forward(df, split)
        m = out.dropna(subset=["min15"])[["Code", "Date", "min15"]].merge(
            ft[["Code", "Date", "min_K15"]], on=["Code", "Date"], how="inner")
        self.assertEqual(len(m), len(ft))
        np.testing.assert_allclose(m["min15"].to_numpy(), m["min_K15"].to_numpy())
        self.assertTrue(out.loc[out["window"] <= 2, "min15"].isna().all())         # テスト窓でない行は NaN


class TestMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = pop_case(bad=10.0)
        cls.end = cls.pop["Date"].max()
        cls.tab = E.monitor(cls.pop, draws=200, data_end=cls.end)

    def test_alarm_after_bad_period(self):
        t = self.tab.dropna(subset=["alarm"])
        months = (t["t"].dt.year - 2022) * 12 + t["t"].dt.month - 1                 # 0 = 2022-01
        in_bad = (months >= 24) & (months <= 24 + 8 + 2)
        self.assertGreater(t.loc[in_bad, "alarm"].mean(), t.loc[~in_bad, "alarm"].mean() + 0.4)

    def test_no_lookahead(self):
        k = 20
        t = self.tab["t"].iloc[k]
        pop2 = self.pop.copy()
        later = pop2["done"] > t
        pop2.loc[later, E.TP] = -50.0
        tab2 = E.monitor(pop2, draws=200, data_end=self.end)
        for c in ("n_B", "n_Q", "n_R", "mean_B", "w1", "down", "up", "null_pct", "alarm", "eps", "floor_w"):
            a, b = self.tab[c].iloc[k], tab2[c].iloc[k]
            if isinstance(a, float) and np.isnan(a):
                self.assertTrue(np.isnan(b), c)
            else:
                self.assertEqual(a, b, c)
        self.assertLess(tab2["mean_next"].iloc[k], self.tab["mean_next"].iloc[k])   # 次の塊は書き換えた側

    def test_floor_needs_history(self):
        t = self.tab
        first = t["eps"].first_valid_index()
        self.assertIsNotNone(first)
        self.assertGreaterEqual(int(t.loc[:first, "alarm"].notna().sum()), E.MIN_EPS)

    def test_no_next_after_data_end(self):
        late = self.tab["t"] + pd.DateOffset(months=E.BLOCK) > self.end
        self.assertTrue(self.tab.loc[late, "n_next"].isna().all())


class TestCoverageAndAlarm(unittest.TestCase):
    def test_coverage_stationary(self):
        pop = pop_case(seed=4, bad=0.0)
        tab = E.monitor(pop, draws=200, data_end=pop["Date"].max())
        c = E.coverage(tab)
        self.assertGreater(c["n"], 10)
        self.assertGreaterEqual(c["cover_w"], 0.6)
        self.assertTrue(0.0 <= c["cover_se"] <= 1.0)

    def test_alarm_precedes_bad(self):
        pop = pop_case(seed=5, bad=12.0, bad_len=10)
        tab = E.monitor(pop, draws=200, data_end=pop["Date"].max())
        at = E.alarm_test(tab)
        self.assertTrue(at["judgeable"])
        self.assertLess(at["diff"], 0)

    def test_alarm_not_judgeable(self):
        tab = pd.DataFrame({"alarm": [0.0, 0.0, 1.0, 0.0], "n_next": [10, 10, 10, 10], "mean_next": [1.0, 2.0, 3.0, 4.0]})
        self.assertFalse(E.alarm_test(tab)["judgeable"])
        self.assertEqual(E.coverage(pd.DataFrame())["n"], 0)

    def test_coverage_needs_enough_months(self):
        tab = pd.DataFrame({"floor_w": [0.0] * 5, "floor_se": [0.0] * 5, "n_next": [20] * 5, "n_B": [100] * 5,
                            "mean_next": [1.0] * 5, "eps": [1.0] * 5})
        c = E.coverage(tab)
        self.assertEqual(c["n"], 5)
        self.assertEqual(c["cover_w"], 1.0)
        self.assertFalse(c["judgeable"])                                            # 5か月では判定しない
        self.assertFalse(c["usable"])
        c6 = E.coverage(pd.concat([tab, tab.iloc[:1]], ignore_index=True))
        self.assertTrue(c6["judgeable"] and c6["usable"])


class TestInputDrift(unittest.TestCase):
    def test_shifted_feature(self):
        pop = pop_case(seed=6, bad=0.0)
        recent = pop["Date"] > pop["Date"].max() - pd.DateOffset(months=3)
        pop.loc[recent, "vol_20d"] += 3.0
        ends = E.month_ends(pop["Date"].min(), pop["Date"].max())
        dr = E.input_drift(pop, ends)
        last = dr.dropna(subset=["vol_20d"]).iloc[-1]
        self.assertGreater(last["vol_20d"], 3 * last["ret_20d"])


class TestReport(unittest.TestCase):
    def test_sections_and_no_codes(self):
        pop = pop_case(seed=7, bad=8.0)
        band = pd.concat([pop, pop_case(seed=8, bad=0.0)], ignore_index=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(pop, band, data_end=pop["Date"].max(), draws=100)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ まとめ", "判定"):
            self.assertIn(tag, out)
        for k in ("main", "alarm", "ref", "input", "verdict", "table"):
            self.assertIn(k, s)
        for code in pop["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
