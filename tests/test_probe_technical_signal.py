#!/usr/bin/env python3
"""
probe_technical_signal（research/exp/probe_technical_signal.py）の部品。データには触らない。

- 日付内 AUC: lab._auc_in_day（同じ日の正例・負例の両方がある日だけ、日の件数で重み付け）と同じ値。欠測の行は除く
- 5分位ごとの正例率: 件数の合計が欠測でない行の数、境目は6つ
- 予想（HYPOTHESIS）は111列すべてにあり、向きは + / − / 0、強さは 強・中・弱・無。比べる27列は今の列の組にある
- 列ごとの表と予想との突き合わせが通る

  python3 tests/test_probe_technical_signal.py
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import lab  # noqa: E402
import probe_technical_signal as P  # noqa: E402
import technicals as TX  # noqa: E402


def frame(n_days=60, per_day=7, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2021-06-01", periods=n_days), per_day)
    n = len(dates)
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 1.5, n) > 0.8).astype(float)
    return pd.DataFrame({"Date": dates, "x": x, "y": y})


class TestAuc(unittest.TestCase):
    def test_same_as_lab(self):
        d = frame()
        want = lab._auc_in_day(d.rename(columns={"x": "score", "y": "label"}))
        got = P.auc_in_day(d["x"].to_numpy(), d["y"].to_numpy(), d["Date"].to_numpy())
        self.assertAlmostEqual(got, want, places=12)
        # 同じ値が多い列（順位が同じ）でも同じ
        xr = np.round(d["x"].to_numpy(), 0)
        want2 = lab._auc_in_day(d.assign(score=xr).rename(columns={"y": "label"}))
        self.assertAlmostEqual(P.auc_in_day(xr, d["y"].to_numpy(), d["Date"].to_numpy()), want2, places=12)

    def test_missing_rows_are_dropped(self):
        d = frame(seed=1)
        x = d["x"].to_numpy().copy()
        x[::5] = np.nan
        kept = d[~np.isnan(x)]
        want = lab._auc_in_day(kept.rename(columns={"x": "score", "y": "label"}))
        self.assertAlmostEqual(P.auc_in_day(x, d["y"].to_numpy(), d["Date"].to_numpy()), want, places=12)
        self.assertTrue(np.isnan(P.auc_in_day(np.full(len(d), np.nan), d["y"].to_numpy(), d["Date"].to_numpy())))

    def test_direction(self):
        d = frame(seed=2)
        a = P.auc_in_day(d["x"].to_numpy(), d["y"].to_numpy(), d["Date"].to_numpy())
        b = P.auc_in_day(-d["x"].to_numpy(), d["y"].to_numpy(), d["Date"].to_numpy())
        self.assertGreater(a, 0.5)
        self.assertAlmostEqual(a + b, 1.0, places=12)


class TestQuintiles(unittest.TestCase):
    def test_rates(self):
        d = frame(seed=3)
        x = d["x"].to_numpy().copy()
        x[:10] = np.nan
        q = P.quintile_rates(x, d["y"].to_numpy(), d["y"].to_numpy())
        self.assertEqual(len(q["q"]), 5)
        self.assertEqual(len(q["edges"]), 6)
        self.assertEqual(sum(q["n"]), int(np.isfinite(x).sum()))
        self.assertEqual(q["q"], q["h"])
        self.assertGreater(q["q"][-1], q["q"][0])                              # 高い分位ほど正例が多い作り


class TestHypothesis(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(set(P.HYPOTHESIS), set(TX.COLUMNS))
        self.assertTrue(all(s in ("+", "-", "0") and k in P.STRENGTH for s, k in P.HYPOTHESIS.values()))
        self.assertTrue(all(k == "無" for s, k in P.HYPOTHESIS.values() if s == "0"))
        self.assertEqual(len(P.PRICE_COLS), 27)
        self.assertFalse(set(P.PRICE_COLS) - set(F.columns(P.E105.BASE_PRESET)))

    def test_table(self):
        rng = np.random.default_rng(4)
        d = frame(n_days=80, per_day=8, seed=4)
        n = len(d)
        cols = {c: rng.normal(size=n) for c in TX.COLUMNS[:6]}
        cols.update({"vol_20d": rng.normal(size=n), "r_high": d["x"].to_numpy()})
        df = pd.concat([d, pd.DataFrame(cols)], axis=1)
        df["y15"], df["y12"], df["hit10"] = d["y"], d["y"], d["y"]
        kinds = {**{c: TX.GROUP_OF[c] for c in TX.COLUMNS[:6]}, "r_high": "既存"}
        t = P.signal_table(df, TX.COLUMNS[:6] + ["r_high"], kinds)
        self.assertEqual(len(t), 7)
        self.assertGreater(t.set_index("col").loc["r_high", "dev"], 0.1)       # x そのものなので強く分ける
        with redirect_stdout(io.StringIO()):
            hc = P.hypothesis_check(t)
        self.assertEqual(hc["all"], 6)


if __name__ == "__main__":
    unittest.main()
