#!/usr/bin/env python3
"""
実験70（research/exp/e70_borderline.py）: 際どい候補の形と、日証金での分け方。重い計算を回さない部分。

- 形: 運用者の例（xgb 96・cat 98・lgbm 85）が「2つ95以上・残り 85〜90」、最下位が lgbm になること。全部の形
- 百分位: その行より前の窓の分布だけで付ける（先の窓・同じ窓を使わない）。窓の当て方
- 日証金の線: その月より前の行だけで引く。逆日歩は1日以上ないと悪い側にしない
- SE: 1銘柄1行なら普通の SE と同じ。同じ銘柄の行が続くと大きくなる

  python3 tests/test_e70_borderline.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e70_borderline as E  # noqa: E402


def pcts(rows):
    return pd.DataFrame(rows, columns=["hp_lgbm", "hp_xgb", "hp_cat"], dtype=float)


class TestShape(unittest.TestCase):
    def test_users_example(self):
        p = pcts([[85.0, 96.0, 98.0]])
        self.assertEqual(E.shape_of(p).iloc[0], "two95_mid")
        self.assertEqual(E.laggard_of(p).iloc[0], "lgbm")
        self.assertIn("two95_mid", E.BORDER)

    def test_all_shapes(self):
        p = pcts([[96, 97, 98],      # all95
                  [91, 96, 97],      # two95_hi
                  [96, 89.9, 97],    # two95_mid
                  [70, 99, 99],      # two95_lo
                  [92, 93, 96],      # min90（95以上は1つ）
                  [90, 91, 92],      # min90
                  [86, 88, 99],      # min85
                  [84.9, 99, 94],    # other（95以上は1つ・最小 85未満）
                  [95, 95, 95],      # all95（ちょうど95は上）
                  [np.nan, 99, 99]])
        self.assertEqual(list(E.shape_of(p)),
                         ["all95", "two95_hi", "two95_mid", "two95_lo", "min90", "min90", "min85",
                          "other", "all95", None])
        self.assertEqual(list(E.laggard_of(p))[:4], ["lgbm", "lgbm", "xgb", "lgbm"])
        self.assertIsNone(E.laggard_of(p).iloc[-1])
        self.assertEqual(E.laggard_of(pcts([[90, 90, 99]])).iloc[0], "lgbm")     # 同点は先のモデル
        self.assertEqual(set(E.BORDER) | {"all95", "other"}, {k for k, _ in E.SHAPES})


class TestHistPct(unittest.TestCase):
    def test_uses_only_earlier_folds(self):
        d = pd.DataFrame({"fold": [0] * 3 + [1] * 4 + [2] * 4 + [3] * 2,
                          "s_lgbm": [9, 9, 9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.25, 10.0]})
        out = E.hist_pct(d, ["lgbm"], min_ref=4)
        self.assertTrue(out.loc[out["fold"] <= 1, "hp_lgbm"].isna().all())  # 窓1は参照が無い・0 は窓の外
        # 窓2は窓1（0.1〜0.4）だけが参照: 0.5 は 4/4 = 100、窓0 の 9 は使わない
        self.assertEqual(list(out.loc[out["fold"] == 2, "hp_lgbm"]), [100.0, 100.0, 100.0, 100.0])
        # 窓3は窓1・2（8行）: 0.25 より低いのは 0.1・0.2 の2行 → 25.0、10.0 は 100
        self.assertEqual(list(out.loc[out["fold"] == 3, "hp_lgbm"]), [25.0, 100.0])

    def test_assign_folds(self):
        ref = pd.Series(pd.bdate_range("2018-01-01", "2023-12-29"))
        f = E.assign_folds(pd.Series(pd.to_datetime(["2018-06-01", "2023-12-29"])), ref)
        self.assertEqual(f[0], 0)                    # 訓練だけの期間は窓の外
        self.assertGreater(f[1], 1)


class TestJsf(unittest.TestCase):
    def test_threshold_uses_only_earlier_months(self):
        dates = pd.Series(pd.to_datetime(["2024-01-05"] * 4 + ["2024-02-05"] * 2 + ["2024-03-05"]))
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 0.0])
        t = E.past_threshold(dates, x, q=50, min_rows=4)
        self.assertTrue(t.iloc[:4].isna().all())            # 1月は前が無い
        self.assertEqual(list(t.iloc[4:6]), [2.5, 2.5])     # 2月は1月の4行だけ（同じ月の 100・200 は使わない）
        self.assertEqual(t.iloc[6], 3.5)                    # 3月は1〜2月の6行

    def test_flags(self):
        n = 400
        rng = np.random.default_rng(0)
        d = pd.DataFrame({"Date": pd.to_datetime(["2024-01-10"] * n + ["2024-02-10"] * 4),
                          "jsf_ratio": 1.0,
                          "jsf_loan_chg20_v": np.r_[rng.normal(0, 1, n), [5.0, -5.0, 5.0, 0.0]],
                          "jsf_fee_days20": np.r_[np.zeros(n), [0.0, 0.0, 3.0, 20.0]]})
        d.loc[3, "jsf_ratio"] = np.nan
        f = E.jsf_flags(d)
        feb = f.iloc[n:]
        self.assertTrue(feb["jsf_ok"].all())
        self.assertFalse(f["jsf_ok"].iloc[:n].any())        # 1月は前の月が無い
        self.assertEqual(list(feb["jsf_good"]), [True, False, True, False])
        # 1月の逆日歩は全部 0 → 線は 0。それでも 1日以上ないと悪い側にしない
        self.assertEqual(list(feb["jsf_bad"]), [False, False, True, True])


class TestStats(unittest.TestCase):
    def test_cluster_se(self):
        x = np.array([1.0, 3.0, 2.0, 6.0, 4.0])
        self.assertAlmostEqual(E.cluster_se(x, np.arange(5)), x.std(ddof=1) / np.sqrt(5))
        same = E.cluster_se(np.array([1.0, 1.0, 5.0, 5.0]), np.array(["a", "a", "b", "b"]))
        indep = E.cluster_se(np.array([1.0, 1.0, 5.0, 5.0]), np.array(["a", "b", "c", "d"]))
        self.assertGreater(same, indep)                     # 同じ銘柄が続くと SE は大きい
        self.assertTrue(np.isnan(E.cluster_se([1.0], ["a"])))

    def test_summarize_and_diff(self):
        g = pd.DataFrame({"Code": ["10000", "20000", "30000", "40000"], "tp10": [10.0, -5.0, 10.0, 2.0],
                          "r": [12.0, -11.0, 3.0, 2.0], "hit10": [True, False, True, False],
                          "label": [1, 0, 1, 0]})
        s = E.summarize(g)
        self.assertEqual((s["n"], s["codes"]), (4, 4))
        self.assertAlmostEqual(s["tp10"], 4.25)
        self.assertEqual((s["win"], s["hit10"], s["lose10"], s["label"]), (75.0, 50.0, 25.0, 50.0))
        dd = E.diff(s, {**s, "tp10": 1.25})
        self.assertAlmostEqual(dd["d"], 3.0)
        self.assertAlmostEqual(dd["se"], np.sqrt(2) * s["tp10_se"])
        self.assertTrue(np.isnan(E.diff(s, {"n": 0})["d"]))
        self.assertEqual(E.summarize(g.iloc[:0]), {"n": 0})


class TestEndToEnd(unittest.TestCase):
    """作った表で main を最後まで回す（全部の節が落ちずに出て、記録が書けること）。"""

    def test_main_runs(self):
        rng = np.random.default_rng(1)
        n = 6000
        dates = pd.bdate_range("2023-01-04", "2026-08-31")
        d = pd.DataFrame({"Code": [f"{i:04d}0" for i in rng.integers(0, 800, n)],
                          "Date": rng.choice(dates, n)})
        hp = rng.uniform(60, 100, (n, 4))
        for i, a in enumerate(("lgbm", "xgb", "cat", "logit")):
            d[f"hp_{a}"] = np.round(hp[:, i], 1)
        d["r"] = rng.normal(1, 10, n)
        d["hit10"] = rng.random(n) < 0.3
        d["tp10"] = np.where(d["hit10"], 10.0, d["r"])
        d["label"] = (d["r"] > 5).astype(int)
        d["n_break"] = rng.integers(1, 40, n)
        d["entry"] = 100.0
        d["jsf_ratio"] = np.where(d["Date"] >= pd.Timestamp("2023-10-01"), 1.0, np.nan)
        d["jsf_loan_chg20_v"] = rng.normal(0, 0.2, n)
        d["jsf_fee_days20"] = np.where(rng.random(n) < 0.25, rng.integers(1, 21, n), 0).astype(float)
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "e70.json")
            with mock.patch.object(E, "load", return_value=(d.sort_values("Date").reset_index(drop=True), True)), \
                    mock.patch.object(E, "OUT", out), mock.patch("builtins.print"):
                self.assertEqual(E.main([]), 0)
            with open(out, encoding="utf-8") as fh:
                res = json.load(fh)
            self.assertEqual(res["rows"], n)
            self.assertEqual(set(res["shape_all"]), {k for k, _ in E.SHAPES} | {"border"})
            self.assertIn("two95_mid_lgbm", res["two95"])
            self.assertIn("min85_lgbm", res["border_laggard"])
            self.assertIn("際どい候補_diff", res["border_laggard"])
            self.assertIn("good", res["jsf"])
            total = sum(v["n"] for k, v in res["shape_all"].items() if k != "border")
            self.assertEqual(total, n)                                    # 形は漏れなく重ならない
            lag = sum(res["border_laggard"][f"{k}_{a}"]["n"] for k in E.BORDER for a in E.TREES)
            self.assertEqual(lag, res["shape_all"]["border"]["n"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
