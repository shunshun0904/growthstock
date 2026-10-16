#!/usr/bin/env python3
"""
実験99（research/exp/e99_limit_up.py）。

- 並べ替え検定: D は層の中の差を規則の行の数で重み付けした平均。差を仕込めば片側 p が小さく、仕込まなければ小さくならない。
  帯ごとの割合が違うだけ（規則の行が小型株に偏っているだけ）なら、全部を足した差は正でも D ≈ 0 で p は小さくならない
- 銘柄単位: 同じ層・同じ銘柄・同じ側の行を平均して1つにする
- 判定: D > 0・行の p < 0.05・銘柄単位の p < 0.05 の3つがそろったときだけ「支持する」。判定2 は小型株の行だけ
- 列を付ける: 時価総額の帯（既存の帯と細かい帯）、ストップ高の列、到達の日がストップ高か
- 合成データで報告が全部の段を出し、帯の件数の合計が合い、銘柄コードを出さない

  python3 tests/test_e99_limit_up.py
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

import price_limits as PL  # noqa: E402
import e99_limit_up as E  # noqa: E402


def frame(n_per=120, windows=(3, 4, 5), effect=0.0, small_bias=False, seed=0):
    """
    窓ごとに n_per 行の前向きの行。帯ごとに素の割合が違う（小型ほど高い）。規則の行（min_K15 ≥ 95）には effect（pt/100）を足す。
    small_bias なら規則の行を小型株に偏らせる（帯の違いだけで全部を足した差が出る形）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_rate = {0: 0.45, 1: 0.35, 2: 0.25, 3: 0.15, 4: 0.08}
    caps = {0: 60.0, 1: 180.0, 2: 600.0, 3: 2000.0, 4: 8000.0}
    for w in windows:
        band = rng.integers(0, 5, size=n_per)
        pct = rng.uniform(0, 100, size=n_per)
        if small_bias:
            pct = np.where(band <= 1, rng.uniform(60, 100, size=n_per), rng.uniform(0, 90, size=n_per))
        rule = pct >= 95
        p = np.array([base_rate[int(b)] for b in band]) + np.where(rule, effect, 0.0)
        ul = (rng.uniform(size=n_per) < np.clip(p, 0, 1)).astype(float)
        tp = rng.normal(2.0, 6.0, size=n_per) + ul * 3.0
        hit = tp >= 10
        d = pd.bdate_range(f"{2020 + w}-01-05", periods=n_per)
        rows.append(pd.DataFrame({
            "Code": [f"{1000 + (i % 30)}0" for i in range(n_per)], "Date": d, "window": w, "n_break": rng.integers(2, 30, size=n_per),
            "min_K15": np.round(pct, 1), "min_T3": np.round(rng.uniform(0, 100, size=n_per), 1),
            "tp10c": np.where(hit, 10.0, tp), "hit10": hit, "hitday10": np.where(hit, rng.integers(1, 21, size=n_per), np.nan),
            "market_cap": [caps[int(b)] * rng.uniform(0.6, 1.4) for b in band], "cap_band": band.astype(float),
            "fine_band": E.fine_band([caps[int(b)] for b in band]).to_numpy(),
            "ul_fwd": ul, "ul_fwd_n": ul * rng.integers(1, 3, size=n_per), "ul_fwd_first": np.where(ul == 1, rng.integers(1, 21, size=n_per), np.nan),
            "ulc_fwd": ul * (rng.uniform(size=n_per) < 0.6), "ll_fwd": (rng.uniform(size=n_per) < 0.05).astype(float),
            "ul_t": (rng.uniform(size=n_per) < 0.1).astype(float), "ul_past_n": rng.poisson(0.5, size=n_per).astype(float),
            "hit_on_ul": np.where(hit, (rng.uniform(size=n_per) < 0.3).astype(float), np.nan)}))
    return pd.concat(rows, ignore_index=True)


class TestStratPerm(unittest.TestCase):
    def test_weighted_difference(self):
        v = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        f = np.array([True, False, True, False, True, True, False, False])
        s = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        t = E.strat_perm(v, f, s, n_perm=30)
        self.assertAlmostEqual(t["D"], 100.0)                 # どちらの層も 印 100% − 残り 0%
        self.assertAlmostEqual(t["D_pooled"], 100.0)
        self.assertEqual((t["n_yes"], t["n_no"], t["strata_used"]), (4, 4, 2))
        self.assertAlmostEqual(t["rate_yes"], 100.0)

    def test_planted_and_null(self):
        g = frame(effect=0.3)
        s = E.strata_of(g)
        t = E.strat_perm(g["ul_fwd"].to_numpy(), g["min_K15"].to_numpy() >= 95, s, n_perm=400)
        self.assertGreater(t["D"], 0)
        self.assertLess(t["p_more"], 0.05)
        g0 = frame(effect=0.0, seed=5)
        t0 = E.strat_perm(g0["ul_fwd"].to_numpy(), g0["min_K15"].to_numpy() >= 95, E.strata_of(g0), n_perm=400)
        self.assertGreater(t0["p_more"], 0.05)

    def test_band_composition_alone_is_not_an_effect(self):
        # 規則の行は割合の高い帯（0・1）にしか無く、帯の中では規則の行とそれ以外の割合が同じ（ul を同じ比で並べる）
        rows = []
        for w in (3, 4):
            for band, rate, n_rule, n_rest in ((0, 0.5, 20, 20), (1, 0.4, 20, 40), (3, 0.1, 0, 60), (4, 0.0, 0, 60)):
                rows += [(w, float(band), True, 1.0 if k < round(rate * n_rule) else 0.0) for k in range(n_rule)]
                rows += [(w, float(band), False, 1.0 if k < round(rate * n_rest) else 0.0) for k in range(n_rest)]
        g = pd.DataFrame(rows, columns=["window", "cap_band", "f", "v"])
        t = E.strat_perm(g["v"].to_numpy(), g["f"].to_numpy(), E.strata_of(g), n_perm=200)
        self.assertAlmostEqual(t["D"], 0.0)                    # 層の中では差が無い
        self.assertGreater(t["D_pooled"], 10.0)                # 全部を足した差は帯の偏りで大きく出る
        self.assertGreater(t["p_more"], 0.05)
        self.assertEqual(t["strata_used"], 4)                  # 両側のある層 = 2窓 × 帯 0・1

    def test_one_side_empty(self):
        t = E.strat_perm(np.array([1.0, 0.0]), np.array([False, False]), np.zeros(2), n_perm=10)
        self.assertTrue(np.isnan(t["D"]))
        self.assertIn("0件", E.fmt_t(t))

    def test_nan_rows_are_dropped(self):
        t = E.strat_perm(np.array([1.0, np.nan, 0.0, 0.0]), np.array([True, True, False, False]), np.zeros(4), n_perm=10)
        self.assertEqual((t["n_yes"], t["n_no"]), (1, 2))


class TestCodeLevel(unittest.TestCase):
    def test_collapse(self):
        g = pd.DataFrame({"ul_fwd": [1.0, 0.0, 1.0, np.nan, 1.0], "Code": ["10010"] * 5})
        v, f, s = E.code_level(g, np.array([True, True, False, True, True]), np.array([30, 30, 30, 30, 41]))
        got = sorted(zip(s.tolist(), f.tolist(), v.tolist()))
        self.assertEqual(got, [(30, False, 1.0), (30, True, 0.5), (41, True, 1.0)])


class TestRuleTest(unittest.TestCase):
    def test_support_needs_all_three(self):
        t = E.rule_test(frame(n_per=200, effect=0.35), n_perm=300)
        self.assertTrue(t["support"], t["row"])
        self.assertEqual(set(t["bands"]), set(E.CAP_LABELS))
        n = sum(b["n_yes"] + b["n_no"] for b in t["bands"].values())
        self.assertEqual(n, t["n_rows"])
        t0 = E.rule_test(frame(n_per=200, effect=-0.2, seed=3), n_perm=300)
        self.assertFalse(t0["support"])
        self.assertLess(t0["row"]["D"], 0)

    def test_small_cap_subset(self):
        g = frame(n_per=200, effect=0.35, seed=7)
        t = E.rule_test(g, sub=g["market_cap"].to_numpy() < E.SMALL_CAP_MAX, n_perm=200)
        self.assertEqual(t["n_rows"], int((g["market_cap"] < E.SMALL_CAP_MAX).sum()))
        for name in E.CAP_LABELS[2:]:
            self.assertEqual(t["bands"][name]["n_yes"] + t["bands"][name]["n_no"], 0)

    def test_rows_without_cap_are_excluded(self):
        g = frame(n_per=60, seed=2)
        g.loc[:9, "cap_band"] = np.nan
        t = E.rule_test(g, n_perm=20)
        self.assertEqual(t["n_rows"], len(g) - 10)


class TestAttach(unittest.TestCase):
    def test_fine_band_and_caps(self):
        self.assertEqual(E.fine_band([50, 100, 299, 300, 499, 500, 999, 1000, 2999, 3000, np.nan]).tolist()[:10],
                         [0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        self.assertTrue(np.isnan(E.fine_band([np.nan]).iloc[0]))
        df = pd.DataFrame({"Code": ["10010", "20020"], "Date": pd.to_datetime(["2024-01-05", "2024-01-05"])})
        fr = pd.DataFrame({"Code": ["10010", "20020", "20020"], "Date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-05"]),
                           "market_cap": [250.0, 4500.0, 4500.0]})
        out = E.attach_caps(df, fr)
        self.assertEqual(out["cap_band"].tolist(), [1.0, 4.0])
        self.assertEqual(out["fine_band"].tolist(), [1.0, 5.0])
        self.assertEqual(len(out), 2)

    def test_attach_limits_and_hit_on_ul(self):
        n = 60
        closes = np.full(n, 1000.0)
        highs = closes.copy()
        highs[33] = 1300                                       # t=30 の3日目（買った日 = 1日目）にストップ高
        bars = pd.DataFrame({"Date": pd.bdate_range("2024-01-04", periods=n), "Code": "10010", "H": highs, "L": closes, "C": closes,
                             "AdjC": closes})
        arrays = PL.code_arrays(PL.flag_limits(bars))
        t = bars["Date"].iloc[30]
        df = pd.DataFrame({"Code": ["10010"], "Date": [t], "hitday10": [3.0]})
        out = E.attach_limits(df, arrays)
        self.assertEqual(out.loc[0, "ul_fwd"], 1.0)
        self.assertEqual(out.loc[0, "ul_fwd_first"], 3.0)
        self.assertEqual(out.loc[0, "hit_on_ul"], 1.0)
        out2 = E.attach_limits(df.assign(hitday10=[4.0]), arrays)
        self.assertEqual(out2.loc[0, "hit_on_ul"], 0.0)
        out3 = E.attach_limits(df.assign(hitday10=[np.nan]), arrays)
        self.assertTrue(np.isnan(out3.loc[0, "hit_on_ul"]))


class TestReport(unittest.TestCase):
    def test_sections_sums_and_no_codes(self):
        ft = frame(n_per=100, effect=0.2, seed=9)
        ff = frame(n_per=100, effect=0.2, seed=10)
        info = {"rows": 1000, "known_share": 97.0, "ul": 30, "ul_share": 3.0, "exact_among_ul": 90.0, "ul_close": 20, "ll": 10}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(ft, ff=ff, flags_info=info, n_perm=40)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ 7.", "■ 8.", "■ まとめ", "判定1", "判定2"):
            self.assertIn(tag, out)
        for k in ("rows", "main", "small", "fine", "prev", "pct_bands", "extras", "past", "returns", "years", "full"):
            self.assertIn(k, s)
        self.assertEqual(s["rows"]["rule"], int((ft["min_K15"] >= 95).sum()))
        fine_n = sum(b["n_all"] for name, b in s["fine"].items() if name != "全体")
        self.assertEqual(fine_n, s["fine"]["全体"]["n_all"])
        for code in ft["Code"].unique():
            self.assertNotIn(code, out)
        self.assertEqual(len(s["years"]), 3)


if __name__ == "__main__":
    unittest.main()
