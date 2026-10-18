#!/usr/bin/env python3
"""
上位2% を5つの条件で絞る probe（research/exp/probe_top2_filters.py）。データには触らない（小さな合成データで確かめる）。

- 決算発表からの営業日数: (開示日, 基準日] に入る営業日の数。基準日の開示は 0、休日の開示はその後の最初の営業日が 1。
  同じ銘柄の、基準日以前で一番新しい開示を使う（ほかの銘柄の開示は見ない）。開示が無ければ欠測
- 条件の境目（500億円・20営業日・配当 1・信用倍率 10倍・実績PER 30倍はちょうどなら満たす）。欠測・赤字（PER なし）は満たさない
- 百分位は画面と同じ（同点は数えない）。Wilson の幅
- 合成データで最後まで走り、報告に銘柄コードが出ない

  python3 tests/test_probe_top2_filters.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e79_model_clusters as E79  # noqa: E402
import models as M  # noqa: E402
import probe_top2_filters as E  # noqa: E402

D = pd.Timestamp


class TestTradingDays(unittest.TestCase):
    # 1/16（火）は休場にした営業日の表
    DAYS = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                           "2024-01-15", "2024-01-17"]).to_numpy()

    def test_counts(self):
        disc = pd.DataFrame({"Code": ["10010", "10010", "10020", "10030", "10030"],
                             "DiscDate": pd.to_datetime(["2024-01-09", "2024-01-12", "2024-01-13", "2024-01-16",
                                                         "2024-01-17"])})
        keys = pd.DataFrame({"Code": ["10010", "10010", "10020", "10030", "10030", "10040", "10010"],
                             "Date": pd.to_datetime(["2024-01-15", "2024-01-12", "2024-01-15", "2024-01-17",
                                                     "2024-01-15", "2024-01-15", "2024-01-11"])})
        td, cal = E.trading_days_since(keys, disc, self.DAYS)
        # 金曜の開示 → 月曜は1 / 当日の開示は0 / 土曜の開示 → 月曜は1 / 1/17 の開示は 1/17 に0（1/16 の開示より新しい）
        # / 1/15 の時点では 1/16・1/17 の開示はまだ無い（先を見ない）→ 欠測 / 開示の無い銘柄は欠測
        # / 1/11 の時点では 1/9 の開示から2営業日（1/12 の開示はまだ）
        np.testing.assert_array_equal(td, [1, 0, 1, 0, np.nan, np.nan, 2])
        np.testing.assert_array_equal(cal, [3, 0, 2, 0, np.nan, np.nan, 2])

    def test_holiday_disclosure(self):
        disc = pd.DataFrame({"Code": ["10030"], "DiscDate": pd.to_datetime(["2024-01-16"])})
        keys = pd.DataFrame({"Code": ["10030"], "Date": pd.to_datetime(["2024-01-17"])})
        td, cal = E.trading_days_since(keys, disc, self.DAYS)
        self.assertEqual(td.tolist(), [1.0])                 # 休場日の開示は、その後の最初の営業日が 1
        self.assertEqual(cal.tolist(), [1.0])


class TestFilters(unittest.TestCase):
    def test_boundaries_and_missing(self):
        f = pd.DataFrame({"market_cap": [500.0, 500.1, np.nan, 10.0, 10.0],
                          "earn_td": [20.0, 21.0, np.nan, 0.0, 0.0],
                          "has_dividend": [1.0, 0.0, np.nan, 1.0, 1.0],
                          "credit_ratio": [10.0, 10.01, np.nan, 0.5, 0.5],
                          "per": [30.0, 30.1, np.nan, 0.0, -5.0]})
        m = E.filter_masks(f)
        self.assertEqual(set(m), set(E.FILTERS))
        self.assertEqual(m["cap"].tolist(), [True, False, False, True, True])
        self.assertEqual(m["earn"].tolist(), [True, False, False, True, True])
        self.assertEqual(m["div"].tolist(), [True, False, False, True, True])
        self.assertEqual(m["credit"].tolist(), [True, False, False, True, True])
        self.assertEqual(m["per"].tolist(), [True, False, False, False, False])     # 0 以下・欠測は満たさない

    def test_constants(self):
        self.assertEqual((E.LINE, E.CAP_MAX, E.EARN_TD, E.CREDIT_MAX, E.PER_MAX), (98.0, 500.0, 20, 10.0, 30.0))
        self.assertEqual(E.RULE_MODELS, ("lgbm_k15", "xgb_k15", "cat_k15"))
        self.assertEqual(len(E79.MODELS), 12)


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = E.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = E.wilson(5, 10)
        self.assertAlmostEqual(lo + hi, 1.0, places=12)       # 半分なら 0.5 を挟んで対称
        self.assertTrue(np.isnan(E.wilson(0, 0)[0]))

    def test_pct_like_screen(self):
        self.assertEqual(E.pct_all(np.array([1.0, 1.0, 2.0, 3.0])).tolist(), [0.0, 0.0, 50.0, 75.0])


def synthetic(dir_: str, seed: int = 3) -> None:
    """
    12本の OOF・学習データ・決算・日足を、窓（2021-05〜2026-09）に収まる日付で作る。日足は銘柄ごとの乱歩で、
    OOF の実収益（ret_o1_20）は lab と同じ作り（翌営業日の寄り → 20営業日目の5日平均終値）で日足から計算する。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", "2026-09-30")
    codes = [f"{10010 + 10 * i}" for i in range(20)]
    bars, keys = [], []
    for c in codes:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.025, len(days))))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.005, len(days)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.02, len(days))))
        bars.append(pd.DataFrame({"Date": days, "Code": c, "AdjO": open_, "AdjH": high, "AdjC": close}))
        ma5 = pd.Series(close).rolling(5, min_periods=1).mean().to_numpy()
        ok = np.flatnonzero((days >= D("2021-05-06")) & (days <= D("2026-08-31")))
        for i in rng.choice(ok, 110, replace=False):
            keys.append((c, days[i], ma5[i + 20] / open_[i + 1] - 1.0))
    pd.concat(bars, ignore_index=True).to_parquet(os.path.join(dir_, "bars_2021.parquet"))
    keys = pd.DataFrame(keys, columns=["Code", "Date", "ret_o1_20"]).sort_values(["Date", "Code"]).reset_index(drop=True)
    n = len(keys)
    base = rng.normal(size=n)
    for a in E79.MODELS:
        o = keys.copy()
        o["score"] = base + rng.normal(scale=0.5, size=n)
        o["label"] = (rng.random(n) < 0.15 + 0.1 * (o["score"] > 1.5)).astype(float)
        o.to_parquet(E79.oof_path(a, dir_))
    disc = pd.DataFrame([(c, d) for c in codes for d in pd.date_range("2020-11-10", "2026-09-30", freq="91D")],
                        columns=["Code", "DiscDate"])
    disc["Sales"] = 1.0
    disc["NP"] = 1.0
    disc.loc[disc.index % 7 == 0, ["Sales", "NP"]] = np.nan          # 実績の無い開示（予想の修正だけ）は数えない
    disc.to_parquet(os.path.join(dir_, "fins_2021.parquet"))
    real = disc.dropna(subset=["Sales", "NP"], how="all").sort_values("DiscDate")
    m = pd.merge_asof(keys[["Code", "Date"]].sort_values("Date"), real[["Code", "DiscDate"]], left_on="Date",
                      right_on="DiscDate", by="Code", direction="backward", allow_exact_matches=True)
    ds = m[["Code", "Date"]].copy()
    ds["days_since_disc"] = ((m["Date"] - m["DiscDate"]).dt.days.astype(float)).clip(upper=400)
    ds["log_market_cap"] = np.log1p(rng.uniform(50, 2000, len(ds)))
    ds["has_dividend"] = rng.choice([0.0, 1.0, np.nan], len(ds), p=[0.3, 0.65, 0.05])
    ds["credit_ratio"] = np.where(rng.random(len(ds)) < 0.2, np.nan, rng.uniform(0.2, 30, len(ds)))
    ds["per"] = np.where(rng.random(len(ds)) < 0.15, np.nan, rng.uniform(3, 80, len(ds)))
    ds.to_parquet(os.path.join(dir_, "dataset.parquet"))


class TestEndToEnd(unittest.TestCase):
    def test_runs_and_reports_no_codes(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic(d)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = E.main(["--data-dir", d])
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("■ 2.", out)
        self.assertIn("days_since_disc と一致", out)
        self.assertIn("ret_o1_20 と一致", out)
        self.assertIn("に届いた割合", out)
        self.assertIn(f"[{M.BASELINE}]", out)
        for c in [f"{10010 + 10 * i}" for i in range(20)]:
            self.assertNotIn(c, out)                                       # 銘柄コードを出さない


if __name__ == "__main__":
    unittest.main()
