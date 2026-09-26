#!/usr/bin/env python3
"""
実験54（research/exp/e54_label_sigma.py）のテスト。

- bars から作る σ20 は price_panel と同じ式（rolling(20, min_periods=15).std × 100）
- 日付ごとの中央値と縮約 √(σ20 × 中央値) が正しい
- 引き直したラベルは attach_rise_label の合成（到達 & 終盤 0.5倍 & MA5>=MA20）と一致し、
  判定できない行は NaN のまま
- 正例率をそろえる二分探索は目標の正例率に届き、k に単調
- 固定%（σ なし）でも同じ

  python3 tests/test_e54_label_sigma.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_dataset as B  # noqa: E402
import e54_label_sigma as E  # noqa: E402


def bars(n_codes=5, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    for c in range(n_codes):
        p = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01 * (c + 1), n_days)))
        for d, v in zip(dates, p):
            rows.append({"Code": f"{1000 + c}", "Date": d, "C": v, "AdjC": v})
    return pd.DataFrame(rows)


class TestSigmas(unittest.TestCase):
    def test_matches_price_panel_formula(self):
        b = bars()
        s = E.sigmas_from_bars(b)
        b2 = b.copy()
        b2["Date"] = pd.to_datetime(b2["Date"])
        b2["O"] = b2["H"] = b2["L"] = b2["C"]
        b2["AdjO"] = b2["AdjH"] = b2["AdjL"] = b2["C"]
        b2["Vo"] = b2["AdjVo"] = 1000
        panel = B.price_panel(b2)
        m = s.merge(panel[["Code", "Date", "vol_20d"]], on=["Code", "Date"])
        both = m["sigma20"].notna() & m["vol_20d"].notna()
        self.assertGreater(both.sum(), 400)
        np.testing.assert_allclose(m.loc[both, "sigma20"], m.loc[both, "vol_20d"], rtol=1e-9)
        self.assertEqual(int(m["sigma20"].isna().sum()), int(m["vol_20d"].isna().sum()))

    def test_median_and_shrink(self):
        s = E.sigmas_from_bars(bars())
        d = s["Date"].iloc[-1]
        day = s[s["Date"] == d]
        self.assertAlmostEqual(day["sigma20_med"].iloc[0], day["sigma20"].median())
        np.testing.assert_allclose(day["sigma_shrink"], np.sqrt(day["sigma20"] * day["sigma20_med"]))
        # σ60 は 45 本たまるまで欠測
        first = s[s["Code"] == "1000"].reset_index(drop=True)
        self.assertTrue(first["sigma60"].iloc[:44].isna().all())
        self.assertTrue(first["sigma60"].iloc[45:].notna().all())


def frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "future_rise": rng.normal(0.05, 0.15, n),
        "end_level": rng.normal(0.02, 0.15, n),
        "uptrend_end": rng.integers(0, 2, n).astype(float),
        "sigma": rng.uniform(0.5, 5.0, n),
    })
    need0 = E.need_from_sigma(df["sigma"], 1.2)
    ok = (df["future_rise"] >= need0) & (df["end_level"] >= 0.5 * need0) & (df["uptrend_end"] == 1)
    df["label"] = ok.astype(float)
    df.loc[:9, "label"] = np.nan          # 判定できない行
    return df


class TestRelabel(unittest.TestCase):
    def test_reproduces_and_keeps_nan(self):
        df = frame()
        y = E.relabel(df, E.need_from_sigma(df["sigma"], 1.2))
        self.assertTrue(y.iloc[:10].isna().all())
        self.assertTrue((y.iloc[10:] == df["label"].iloc[10:]).all())
        # need が欠測の行も NaN
        need = E.need_from_sigma(df["sigma"], 1.2)
        need.iloc[20] = np.nan
        self.assertTrue(np.isnan(E.relabel(df, need).iloc[20]))

    def test_end_mult_is_production_ratio(self):
        self.assertAlmostEqual(E.END_MULT, B.END_RATIO / B.RISE_THRESHOLD)
        self.assertEqual(E.K0, B.VOL_NORM_K)

    def test_match_rate_hits_target(self):
        df = frame()
        target = 0.10               # 合成データの正例率の上限（k→0）は 17% ほど
        k = E.match_rate(df, df["sigma"], target)
        y = E.relabel(df, E.need_from_sigma(df["sigma"], k))
        self.assertLess(abs(y.mean() - target), 0.01)
        # k が大きいほど正例率は下がる
        y2 = E.relabel(df, E.need_from_sigma(df["sigma"], k * 1.5))
        self.assertLess(y2.mean(), y.mean())
        x = E.match_rate(df, None, target)
        yx = E.relabel(df, pd.Series(x, index=df.index))
        self.assertLess(abs(yx.mean() - target), 0.01)

    def test_build_labels(self):
        df = frame()
        df["sigma20"] = df["sigma"]
        df["sigma60"] = df["sigma"] * 1.1
        df["sigma_shrink"] = np.sqrt(df["sigma"] * df["sigma"].median())
        df, tab = E.build_labels(df, ["L0", "L1m", "L3m"])
        self.assertEqual(tab["arm"].tolist(), ["L0", "L1m", "L3m"])
        self.assertAlmostEqual(tab.loc[0, "k_or_x"], 1.2)
        t = tab.loc[0, "pos_rate"]
        self.assertLess(abs(tab.loc[1, "pos_rate"] - t), 0.01)
        self.assertLess(abs(tab.loc[2, "pos_rate"] - t), 0.01)
        self.assertTrue((df["y_L0"].dropna() == df["label"].dropna()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
