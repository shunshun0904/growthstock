"""
日証金の特徴量（research/jsf_features.py）。値は架空、列の見出しは実測（tests/test_jsf_fetch.py と同じ）。
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))
import jsf_features as JF  # noqa: E402


def hist_rows(code, dates, loan, stock, fee=None, restrict=None):
    n = len(dates)
    fee = fee or [0.0] * n
    restrict = restrict or [""] * n
    loan, stock = list(loan), list(stock)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"銘柄コード": code[:4], "code": code, "申込日": pd.Timestamp(d),
                     "融資新規（株）": 10.0, "融資返済（株）": (10.0 - (loan[i] - loan[i - 1])) if i else np.nan,
                     "融資残高（株）": float(loan[i]),
                     "貸株新規（株）": 5.0, "貸株返済（株）": (5.0 - (stock[i] - stock[i - 1])) if i else np.nan,
                     "貸株残高（株）": float(stock[i]), "差引残高（株）": float(loan[i] - stock[i]),
                     "貸借値段（円）": 1000.0, "品貸料率（品貸日数分/円）": np.nan, "品貸日数": np.nan,
                     "品貸料率（年率換算/％）": (fee[i] if fee[i] else np.nan), "制限措置": restrict[i],
                     "貸借区分": "貸借"})
    return pd.DataFrame(rows)


class TestRestrictLevel(unittest.TestCase):
    def test_words(self):
        s = pd.Series(["", "－", "注意喚起", "申込制限", "申込停止", "貸株申込制限", "なにか別の"])
        out = JF.restrict_level(s).tolist()
        self.assertEqual(out[:6], [0.0, 0.0, 1.0, 2.0, 3.0, 2.0])
        self.assertTrue(np.isnan(out[6]))

    def test_lendable(self):
        out = JF.lendable_flag(pd.Series(["貸借", "貸借融資", "非貸借", "", "？"])).tolist()
        self.assertEqual(out[:3], [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(out[3]) and np.isnan(out[4]))


class TestPanel(unittest.TestCase):
    def test_daily_wins_and_hist_fills_fee_and_restrict(self):
        dates = pd.bdate_range("2026-09-21", periods=4)
        h = hist_rows("72030", dates, [100, 110, 120, 130], [50, 50, 60, 60],
                      fee=[0, 0, 0.73, 0.73], restrict=["", "", "注意喚起", "注意喚起"])
        # load_hist と同じ共通の列にそろえる
        hh = pd.DataFrame({"app_date": h["申込日"], "Code": h["code"]})
        for k, c in JF.HIST_COLS.items():
            hh[k] = (JF.restrict_level(h[c]) if k == "restrict" else JF.lendable_flag(h[c]) if k == "lendable"
                     else pd.to_numeric(h[c], errors="coerce"))
        hh["source"] = "hist"
        d = pd.DataFrame({"app_date": [dates[-1]], "Code": ["72030"], "loan_new": [1.0], "loan_ret": [0.0],
                          "loan_bal": [999.0], "stock_new": [0.0], "stock_ret": [0.0], "stock_bal": [60.0],
                          "net_bal": [939.0], "price": [np.nan], "fee_yen": [np.nan], "fee_days": [np.nan],
                          "fee_ann": [np.nan], "restrict": [np.nan], "lendable": [np.nan], "source": ["daily"]})
        p = JF.panel(hh, d)
        self.assertEqual(len(p), 4)
        last = p[p["app_date"] == dates[-1]].iloc[0]
        self.assertEqual(last["loan_bal"], 999.0)            # 同じ申込日は daily（確報）
        self.assertEqual(last["source"], "daily")
        self.assertEqual(last["restrict"], 1.0)              # 制限措置は hist から補う
        self.assertAlmostEqual(last["fee_ann"], 0.73)
        self.assertEqual(p[p["app_date"] == dates[0]].iloc[0]["fee_ann"], 0.0)   # 逆日歩なし = 0
        self.assertEqual(last["lendable"], 1.0)              # 貸借区分も hist から

    def test_lendable_is_carried_to_daily_only_days(self):
        dates = pd.bdate_range("2026-09-21", periods=3)
        hh = pd.DataFrame({"app_date": dates[:2], "Code": "72030", "loan_bal": [1.0, 2.0], "stock_bal": [0.0, 0.0],
                           "loan_new": 0.0, "loan_ret": 0.0, "stock_new": 0.0, "stock_ret": 0.0, "net_bal": 1.0,
                           "price": 100.0, "fee_yen": np.nan, "fee_days": np.nan, "fee_ann": np.nan,
                           "restrict": 0.0, "lendable": [0.0, 0.0], "source": "hist"})
        d = hh.iloc[[1]].copy()
        d["app_date"] = dates[2]
        d[["lendable", "restrict", "source"]] = [np.nan, np.nan, "daily"]
        p = JF.panel(hh, d)
        self.assertEqual(p["lendable"].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(np.isnan(p["restrict"].iloc[-1]))       # 制限措置は引き継がない


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-07-01", periods=30)
        loan = [1000 + 10 * i for i in range(30)]
        stock = [500] * 30
        fee = [0.0] * 20 + [1.0] * 10
        h = hist_rows("72030", self.dates, loan, stock, fee=fee)
        self.p = pd.DataFrame({"app_date": h["申込日"], "Code": h["code"]})
        for k, c in JF.HIST_COLS.items():
            self.p[k] = (JF.restrict_level(h[c]) if k == "restrict" else JF.lendable_flag(h[c]) if k == "lendable"
                         else pd.to_numeric(h[c], errors="coerce"))
        self.p["source"] = "hist"
        self.p["fee_ann"] = self.p["fee_ann"].fillna(0.0)
        self.vol = pd.DataFrame({"Date": self.dates, "Code": "72030", "avg_vol": 100.0})

    def test_values(self):
        f = JF.features_on_panel(self.p, self.vol).set_index("app_date")
        last = f.iloc[-1]
        self.assertAlmostEqual(last["jsf_ratio"], np.log((1290 + 1) / (500 + 1)))
        self.assertAlmostEqual(last["jsf_ratio_chg5"], np.log(1291 / 501) - np.log(1241 / 501))
        self.assertAlmostEqual(last["jsf_loan_v"], np.arcsinh(12.9))     # 裾が重いので asinh
        self.assertAlmostEqual(last["jsf_stock_v"], np.arcsinh(5.0))
        self.assertAlmostEqual(last["jsf_net_v"], np.arcsinh(7.9))
        self.assertAlmostEqual(last["jsf_loan_chg5_v"], np.arcsinh(0.5))
        self.assertAlmostEqual(last["jsf_loan_chg20_v"], np.arcsinh(2.0))
        self.assertAlmostEqual(last["jsf_stock_chg20_v"], 0.0)
        self.assertAlmostEqual(last["jsf_long_new5_v"], np.arcsinh(0.5))      # 新規 10 × 5日 ÷ 100
        self.assertAlmostEqual(last["jsf_fee"], np.log1p(1.0))             # 逆日歩は log1p
        self.assertAlmostEqual(last["jsf_fee_days20"], 10.0)
        self.assertAlmostEqual(last["jsf_fee_max20"], np.log1p(1.0))
        self.assertEqual(last["jsf_restrict"], 0.0)
        self.assertEqual(last["jsf_lendable"], 1.0)
        self.assertTrue(np.isnan(f.iloc[3]["jsf_ratio_chg5"]))     # 5日前が無い

    def test_no_volume_gives_nan_for_scaled_columns(self):
        f = JF.features_on_panel(self.p, None)
        self.assertTrue(f["jsf_loan_v"].isna().all())
        self.assertFalse(f["jsf_ratio"].isna().any())


class TestAttach(unittest.TestCase):
    def test_uses_previous_application_day_only(self):
        dates = pd.bdate_range("2026-09-21", periods=5)       # 月〜金
        feats = pd.DataFrame({"Code": "72030", "app_date": dates, "jsf_ratio": np.arange(5, dtype=float)})
        for c in JF.columns("all"):
            if c not in feats.columns:
                feats[c] = 1.0
        frame = pd.DataFrame({"Code": ["72030"] * 4 + ["99840"],
                              "Date": [dates[2], dates[4], pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-20"),
                                       dates[2]],
                              "label": [1, 0, 1, 0, 1]})
        out = JF.attach(frame, feats)
        self.assertEqual(out["jsf_ratio"].tolist()[0], 1.0)      # 水曜の行 → 火曜の申込日（当日は使わない）
        self.assertEqual(out["jsf_ratio"].tolist()[1], 3.0)      # 金曜 → 木曜
        self.assertEqual(out["jsf_ratio"].tolist()[2], 4.0)      # 翌週月曜 → 金曜（3暦日前）
        self.assertEqual(out["jsf_lag"].tolist()[2], 3.0)
        self.assertTrue(np.isnan(out["jsf_ratio"].tolist()[3]))  # 1か月後は古すぎる
        self.assertTrue(np.isnan(out["jsf_ratio"].tolist()[4]))  # 日証金に無い銘柄
        self.assertEqual(list(out.index), list(frame.index))
        self.assertIn("label", out.columns)


class TestFee(unittest.TestCase):
    def test_annualize(self):
        out = JF.annualize_fee(pd.Series([0.05, 0.10, np.nan]), pd.Series([1, 2, 1]), pd.Series([2500, 1000, 100]))
        self.assertAlmostEqual(out.iloc[0], 0.05 / 2500 * 365 * 100)
        self.assertAlmostEqual(out.iloc[1], 0.10 / 2 / 1000 * 365 * 100)
        self.assertTrue(np.isnan(out.iloc[2]))


class TestLoadFromParquet(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip(self):
        dates = pd.bdate_range("2026-09-21", periods=3)
        h = hist_rows("72030", dates, [100, 110, 120], [50, 50, 60], restrict=["", "注意喚起", ""])
        h.to_parquet(os.path.join(self.dir, "jsf_hist.parquet"), index=False)
        b = pd.DataFrame({"申込日": [dates[-1]], "code": ["72030"], "取引所区分名": ["東証"],
                          "融資新規株数": [1.0], "融資返済株数": [0.0], "融資残高株数": [121.0],
                          "貸株新規株数": [0.0], "貸株返済株数": [0.0], "貸株残高株数": [60.0], "差引残高株数": [61.0]})
        b.to_parquet(os.path.join(self.dir, "jsf_balance.parquet"), index=False)
        ln = pd.DataFrame({"貸借申込日": [dates[-1]], "code": ["72030"], "取引所区分": ["東証"],
                           "貸借値段（円）": [1000.0], "当日品貸料率（円）": [0.05], "当日品貸日数": [1]})
        ln.to_parquet(os.path.join(self.dir, "jsf_lending.parquet"), index=False)
        hist = JF.load_hist(os.path.join(self.dir, "jsf_hist.parquet"))
        daily = JF.load_daily(os.path.join(self.dir, "jsf_balance.parquet"), os.path.join(self.dir, "jsf_lending.parquet"))
        self.assertEqual(hist["restrict"].tolist(), [0.0, 1.0, 0.0])
        self.assertAlmostEqual(daily["fee_ann"].iloc[0], 0.05 / 1000 * 365 * 100)
        p = JF.panel(hist, daily)
        self.assertEqual(p.loc[p["app_date"] == dates[-1], "loan_bal"].iloc[0], 121.0)
        self.assertEqual(JF.columns("both"), JF.columns("all") + ["jsf_lag"])


if __name__ == "__main__":
    unittest.main()
