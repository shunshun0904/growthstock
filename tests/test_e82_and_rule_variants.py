#!/usr/bin/env python3
"""
実験82（research/exp/e82_and_rule_variants.py）。

- 丸めは画面（live_track.pct_of）と同じ
- 規則の印: A15 / A18 / A9 は R0 の中、4本版は R0_4 の中
- R0 の中の 印 − それ以外: 仕込んだ効果は拾い（銘柄単位でも）、効果が無ければ p は小さくならない
- 日ごとの選び方（発火8件以上・上位2件・1.2σ の3本の最小の高い順、同点は lgbm のスコア順）と、運用の通りの売買・
  月に2銘柄
- 報告に銘柄コードが出ない

  python3 tests/test_e82_and_rule_variants.py
"""
import contextlib
import io
import os
import sys
import types
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E  # noqa: E402


def synth(n_windows=6, per=1500, seed=0, planted=0.08):
    """
    合意 c と「1.5σ の4本だけ高い」ずれ q。c が高く（> 1.5）q > 0 の行の実収益を planted だけ上げる。
    R0（1.2σ の3本が95以上）はほぼ c が高い行なので、その中で 1.5σ の3本も95以上（q が高い）の行が良くなる。
    """
    rng = np.random.default_rng(seed)
    folds, rows = [], []
    start = pd.Timestamp("2021-05-03")
    for w in range(1, n_windows + 1):
        ts = start + pd.DateOffset(months=6 * (w - 1))
        te = start + pd.DateOffset(months=6 * w) - pd.Timedelta(days=1)
        folds.append(types.SimpleNamespace(index=w, test_start=str(ts.date()), test_end=str(te.date())))
        d = rng.choice(pd.bdate_range(ts, te), size=per)
        c = rng.normal(size=per)
        q = rng.normal(size=per)
        ret = 0.01 * c + planted * ((q > 0) & (c > 1.5)) + rng.normal(0, 0.05, size=per)
        rec = {"Code": [f"{8000 + i % 900}0" for i in range(per)], "Date": d, "window": w, E79.OUTCOME: ret}
        for a in E79.MODELS:
            tilt = 0.8 if a in E79.K15 else 0.0
            rec[f"score_{a}"] = c + tilt * q + rng.normal(0, 0.3, size=per)
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True).drop_duplicates(["Code", "Date"])
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    r = df[E79.OUTCOME].to_numpy() * 100
    hit = r >= 20
    df["ret_tp"] = np.where(hit, 20.0, r)
    df["status"] = np.where(hit, "+20%到達", "満了")
    df["hit_day"] = np.where(hit, 5.0, np.nan)
    return df, folds


def forward(df, folds):
    split = E79.window_split(df, folds)
    return E.build_frame(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])


class TestRounding(unittest.TestCase):
    def test_same_as_screen(self):
        rng = np.random.default_rng(3)
        hist = rng.normal(size=3001)
        x = np.r_[rng.normal(size=500), np.sort(hist)[[2849, 2850, 2851, 2852]]]   # 95 の線のあたりも入れる
        want = LT.pct_of(x, hist)
        got = E.rounded(E79.pct_within(np.sort(hist), x))
        np.testing.assert_array_equal(got, want)
        self.assertTrue(np.isnan(E.rounded(np.array([np.nan]))[0]))


class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.folds = synth()
        cls.ft = forward(cls.df, cls.folds)
        cls.df0, folds0 = synth(planted=0.0, seed=11)
        cls.ft0 = forward(cls.df0, folds0)
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_flags_are_nested(self):
        ft = self.ft
        for sub, sup in (("A15", "R0"), ("A18", "R0"), ("A9", "A15"), ("A9", "A18"), ("A15_4", "R0_4"),
                         ("A18_4", "R0_4"), ("R0_4", "R0")):
            self.assertFalse((ft[sub] & ~ft[sup]).any(), (sub, sup))
        np.testing.assert_array_equal(ft["A9"], ft["A15"] & ft["A18"])
        share = ft["R0"].mean()
        self.assertTrue(0.01 < share < 0.10, share)
        self.assertTrue(0.2 < ft.loc[ft["R0"], "A15"].mean() < 0.9)
        self.assertEqual(sorted(ft["window"].unique()), [3, 4, 5, 6])

    def test_planted_effect_is_found(self):
        res = E.split_test(self.ft, "A15", "R0", n_perm=300)
        self.assertGreater(res["summary"]["mean"], 1.0)
        self.assertLess(res["p"], 0.05)
        self.assertTrue(E.judge(res)["pass"])
        rc = E.split_test(self.ft, "A15", "R0", n_perm=300, unit="code")
        self.assertLess(rc["p"], 0.05)
        rt = E.split_test(self.ft, "A15", "R0", col="ret_tp", n_perm=300)
        self.assertGreater(rt["summary"]["mean"], 1.0)

    def test_no_effect_is_not_significant(self):
        res = E.split_test(self.ft0, "A15", "R0", n_perm=300)
        self.assertGreater(res["p"], 0.01)
        self.assertFalse(E.judge(res)["pass"])

    def test_daily_picks(self):
        d1, d2 = pd.Timestamp("2022-06-01"), pd.Timestamp("2022-06-02")
        rows = []
        for i in range(10):                                   # 発火10件。R0 は3件（99 / 97 / 97）
            rows.append({"Code": f"1{i:03d}0", "Date": d1, "R0": i < 3, "pmin3": [99.0, 97.0, 97.0][i] if i < 3 else 50.0,
                         "score": [0.1, 0.2, 0.3][i] if i < 3 else 0.0})
        for i in range(7):                                    # 発火7件は見送り
            rows.append({"Code": f"2{i:03d}0", "Date": d2, "R0": True, "pmin3": 99.0, "score": 0.5})
        p = E.daily_picks(pd.DataFrame(rows), "R0")
        self.assertEqual(list(p["Code"]), ["10000", "10020"])  # 99 が先、97 の同点はスコアの高い方
        self.assertEqual(list(p["rank"]), [1, 2])

    def test_trade_sim_and_monthly(self):
        n_days = LT.n_bar_days(self.bar_days, self.ft["Date"].min(), self.ft["Date"].max())
        st = E.trade_sim(self.ft, "R0", self.bar_days, n_days)
        self.assertGreater(st["n"], 0)
        self.assertLessEqual(st["n"], st["picks"])
        self.assertTrue(set(st["by_window"]) <= {3, 4, 5, 6})
        mt = E.monthly_top(self.ft, "A15")
        self.assertTrue(mt["A15"].all())
        self.assertLessEqual(mt.groupby(pd.to_datetime(mt["Date"]).dt.to_period("M")).size().max(), 2)

    def test_report_prints_no_codes(self):
        ff = E.build_frame(self.df, [(np.arange(len(self.df)), E79.pct_matrix(self.df, E79.MODELS))])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, bar_days=self.bar_days, ff=ff, n_perm=50)
        out = buf.getvalue()
        for tag in ("■ 2.", "■ 4.", "■ 5.", "■ 6.", "■ まとめ"):
            self.assertIn(tag, out)
        self.assertEqual(set(s["main"]), {"A15", "A18"})
        self.assertIn("vs_R0", s["trade"]["A15"])
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
