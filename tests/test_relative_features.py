#!/usr/bin/env python3
"""
相対位置の特徴量（research/relative_features.py。実験76）。

- 列の名前: 業種内 10 + 市場全体 10 の 20列。本番の239列と重ならず、元の指標は全部 239列に入っている
- 参照の作り: 月末の終値は月末から10日以内の最後のバー。母集団は普通株だけ（ETF・REIT 等は除く）で、
  業種は master_hist のその時点（月末以前の最後の版）の S33
- 決算の付け方: 月末までに開示済みの直近。1年より古ければ指標は欠測（API の PER / PBR は残る）。
  PER / PBR は API を優先し、上限を超えたら欠測
- 百分位: 参照の中で値より小さい割合 + 同じ値の半分（0〜1）。値が欠測なら欠測
- 付け方（時点整合）: 行の日付より **前** の最後の月末の参照を使う。最初の月末以前の行・月末ちょうどの行には
  その月末の参照は付かない。業種の母集団が 20未満なら業種内は欠測、市場全体が 200未満なら両方欠測

  python3 tests/test_relative_features.py
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import relative_features as RF  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_names_and_no_overlap(self):
        cols = RF.columns("all")
        self.assertEqual(len(cols), 20)
        self.assertEqual(cols[:10], [f"rel_ind_{m}" for m in RF.METRICS])
        self.assertEqual(cols[10:], [f"rel_mkt_{m}" for m in RF.METRICS])
        self.assertEqual(RF.columns("ind"), cols[:10])
        self.assertEqual(RF.columns("mkt"), cols[10:])
        base = F.columns(F.DEFAULT_PRESET)
        self.assertFalse(set(base) & set(cols))
        self.assertTrue(set(RF.METRICS) <= set(base))              # 元の指標は本番の列
        self.assertEqual(len(RF.METRICS), 10)
        with self.assertRaises(KeyError):
            RF.columns("sector")

    def test_month_ends(self):
        e = RF.month_ends("2024-01-15", "2024-04-10")
        self.assertEqual([str(x.date()) for x in e], ["2024-01-31", "2024-02-29", "2024-03-31"])
        self.assertEqual(len(RF.month_ends("2024-01-31", "2024-01-31")), 1)


class TestPrices(unittest.TestCase):
    def test_month_end_prices(self):
        days = pd.bdate_range("2024-01-04", "2024-03-29")
        rows = []
        for i, d in enumerate(days):
            rows.append({"Code": "10000", "Date": d, "C": 100.0 + i})
            if d < pd.Timestamp("2024-02-10"):                     # 2月は10日までしかバーが無い
                rows.append({"Code": "20000", "Date": d, "C": 50.0})
        bars = pd.DataFrame(rows)
        ends = RF.month_ends("2024-01-01", "2024-03-31")
        px = RF.month_end_prices(bars, ends)
        a = px[px["Code"] == "10000"].set_index("Date")
        self.assertEqual([str(x.date()) for x in a.index], ["2024-01-31", "2024-02-29", "2024-03-31"])
        self.assertEqual(a.loc["2024-01-31", "px"], 100.0 + list(days).index(pd.Timestamp("2024-01-31")))
        self.assertEqual(str(a.loc["2024-03-31", "px_date"].date()), "2024-03-29")   # 月末が休みなら最後のバー
        b = px[px["Code"] == "20000"]
        self.assertEqual([str(x.date()) for x in b["Date"]], ["2024-01-31"])       # 2月は 19日前 → 付かない

    def test_universe_filters_and_sector_as_of(self):
        ends = RF.month_ends("2024-01-01", "2024-02-29")
        prices = pd.DataFrame({"Code": ["10000", "10000", "12345", "30000", "40000"],
                               "Date": [ends[0], ends[1], ends[0], ends[0], ends[0]],
                               "px": [1.0, 1.0, 1.0, 1.0, 1.0], "px_date": [ends[0], ends[1], ends[0], ends[0], ends[0]]})
        master = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01"]),
                               "Code": ["10000", "10000", "30000", "40000"],
                               "S33": ["0050", "3050", "0050", "9999"], "Mkt": ["0111", "0111", "0109", "0113"]})
        u = RF.universe(prices, master)
        self.assertEqual(sorted(u["Code"].unique()), ["10000", "40000"])      # 12345 は普通株でない。30000 は 109
        a = u[u["Code"] == "10000"].set_index("Date")["s33"]
        self.assertEqual(float(a.loc[ends[0]]), 50.0)                            # 1月末は 1/1 の版
        self.assertEqual(float(a.loc[ends[1]]), 3050.0)                          # 2月末は 2/1 の版
        self.assertEqual(float(u.loc[u["Code"] == "40000", "s33"].iloc[0]), 9999.0)
        u0 = RF.universe(prices, None)
        self.assertTrue(u0["s33"].isna().all())
        self.assertEqual(sorted(u0["Code"].unique()), ["10000", "30000", "40000"])  # master が無ければ市場で除けない


class TestPanel(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_panel_from_quarters(self):
        m1, m2 = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
        keys = pd.DataFrame({"Code": ["10000", "10000", "20000", "30000", "40000"],
                             "Date": [m1, m2, m1, m1, m1],
                             "px": [1000.0, 1200.0, 500.0, 100.0, 300.0], "s33": [50.0, 50.0, 50.0, 60.0, 60.0]})
        q = pd.DataFrame({"Code": ["10000", "10000", "20000", "30000", "40000"],
                          "DiscDate": pd.to_datetime(["2023-11-10", "2024-02-14", "2022-11-10", "2023-12-01", "2023-12-01"]),
                          "CurFYSt": pd.to_datetime(["2023-04-01", "2023-04-01", "2022-04-01", "2023-04-01", "2023-04-01"]),
                          "quarter": [2, 3, 2, 2, 2],
                          "eps_ttm": [50.0, 60.0, 10.0, 0.01, 30.0], "BPS": [500.0, 520.0, 100.0, -5.0, 200.0],
                          "net_margin": [5.0, 6.0, 1.0, 2.0, 3.0], "ROE_q0": [10.0, 11.0, 2.0, 3.0, 4.0]})
        # API の PER（20000 の 1月末に 3日前の値。10000 には無い）
        pd.DataFrame({"Code": ["20000"], "Date": [pd.Timestamp("2024-01-28")], "PER": [12.0], "PBR": [np.nan],
                      "ROE": [np.nan]}).to_parquet(os.path.join(self.dir, "valuation_2024.parquet"), index=False)
        p = RF.panel_from_quarters(keys, q, self.dir, metrics=["net_margin", "ROE_q0", "per", "pbr"])
        self.assertEqual(list(p.columns), ["Code", "Date", "px", "s33", "net_margin", "ROE_q0", "per", "pbr"])
        r = p.set_index(["Code", "Date"])
        self.assertAlmostEqual(r.loc[("10000", m1), "per"], 1000.0 / 50.0)          # 1月末は 11/10 の決算
        self.assertAlmostEqual(r.loc[("10000", m1), "net_margin"], 5.0)
        self.assertAlmostEqual(r.loc[("10000", m2), "per"], 1200.0 / 60.0)          # 2月末は 2/14 の決算
        self.assertAlmostEqual(r.loc[("10000", m2), "pbr"], 1200.0 / 520.0)
        self.assertTrue(np.isnan(r.loc[("20000", m1), "net_margin"]))                # 1年より古い → 欠測
        self.assertAlmostEqual(r.loc[("20000", m1), "per"], 12.0)                    # API の PER は残る
        self.assertTrue(np.isnan(r.loc[("20000", m1), "pbr"]))                       # 自前も古い → 欠測
        self.assertTrue(np.isnan(r.loc[("30000", m1), "per"]))                       # 100/0.01 = 10,000 > 上限
        self.assertTrue(np.isnan(r.loc[("30000", m1), "pbr"]))                       # BPS ≤ 0
        self.assertAlmostEqual(r.loc[("40000", m1), "per"], 10.0)
        self.assertAlmostEqual(r.loc[("40000", m1), "pbr"], 1.5)
        self.assertEqual(B.PER_MAX, 1000.0)


class TestPercentile(unittest.TestCase):
    def test_values(self):
        a = [1.0, 2.0, 3.0, 4.0]
        out = RF.percentile([0.0, 5.0, 2.5, 2.0, np.nan], a)
        np.testing.assert_allclose(out[:4], [0.0, 1.0, 0.5, 0.375])
        self.assertTrue(np.isnan(out[4]))
        self.assertTrue(np.isnan(RF.percentile([1.0], [])).all())
        self.assertAlmostEqual(RF.percentile([7.0], [1.0, 7.0, 7.0, 7.0])[0], (1 + 4) / 8)   # 同じ値は半分（参照は昇順）

    def test_reference_dates(self):
        ends = pd.to_datetime(["2024-01-31", "2024-02-29"]).to_numpy()
        out = RF.reference_dates(pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-01", "2024-03-10"]), ends)
        self.assertTrue(np.isnat(out[0]))
        self.assertTrue(np.isnat(out[1]))                                              # 月末ちょうどは前の月末（無い）
        self.assertEqual(pd.Timestamp(out[2]), pd.Timestamp("2024-01-31"))
        self.assertEqual(pd.Timestamp(out[3]), pd.Timestamp("2024-02-29"))


class TestAttach(unittest.TestCase):
    def panel(self):
        """3月末: 1月末・2月末は 260銘柄（業種 50 が 25・60 が 10・残り 70）、3月末は 100銘柄だけ（市場が小さい）。"""
        rows = []
        for m in ("2024-01-31", "2024-02-29"):
            for i in range(260):
                s33 = 50.0 if i < 25 else (60.0 if i < 35 else 70.0)
                v = float(i + 1) if m == "2024-01-31" else float(260 - i)
                rows.append({"Code": f"{i:04d}0", "Date": pd.Timestamp(m), "px": 1.0, "s33": s33, "per": v,
                             "net_margin": np.nan if i % 2 else v})
        for i in range(100):
            rows.append({"Code": f"{i:04d}0", "Date": pd.Timestamp("2024-03-31"), "px": 1.0, "s33": 50.0,
                         "per": float(i), "net_margin": float(i)})
        return pd.DataFrame(rows)

    def test_point_in_time_and_groups(self):
        panel = self.panel()
        frame = pd.DataFrame({
            "Code": ["00000"] * 7,
            "Date": pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-05", "2024-02-05", "2024-02-05", "2024-03-05",
                                    "2024-04-05"]),
            "s33_code": [50.0, 50.0, 50.0, 60.0, np.nan, 50.0, 50.0],
            "per": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "net_margin": [np.nan] * 7})
        out = RF.attach(frame, panel, metrics=["per", "net_margin"])
        self.assertEqual(len(out), 7)
        self.assertTrue({"rel_mkt_per", "rel_ind_per", "rel_mkt_net_margin", "rel_ind_net_margin"} <= set(out.columns))
        self.assertTrue(np.isnan(out.loc[0, "rel_mkt_per"]))                     # 最初の月末より前
        self.assertTrue(np.isnan(out.loc[1, "rel_mkt_per"]))                     # 月末ちょうど（その月末の参照は使わない）
        # 2/5 は 1月末の参照: 市場 260 の中で 10 は「1..9 が下、10 が同じ」→ (9+10)/520
        self.assertAlmostEqual(out.loc[2, "rel_mkt_per"], 19 / 520)
        # 業種 50（25銘柄・値 1..25）の中で 10 → (9+10)/50
        self.assertAlmostEqual(out.loc[2, "rel_ind_per"], 19 / 50)
        self.assertTrue(np.isnan(out.loc[3, "rel_ind_per"]))                      # 業種 60 は 10銘柄 < 20
        self.assertAlmostEqual(out.loc[3, "rel_mkt_per"], 19 / 520)               # 市場全体は付く
        self.assertTrue(np.isnan(out.loc[4, "rel_ind_per"]))                      # 業種が欠測
        self.assertAlmostEqual(out.loc[4, "rel_mkt_per"], 19 / 520)
        # 3/5 は 2月末の参照（値は逆順 260..1。10 より下は 1..9 の 9つ）→ 同じ百分位
        self.assertAlmostEqual(out.loc[5, "rel_mkt_per"], 19 / 520)
        self.assertAlmostEqual(out.loc[5, "rel_ind_per"], 0.0)                     # 業種 50 は値 260..236 → 全部上 → 0
        self.assertTrue(np.isnan(out.loc[6, "rel_mkt_per"]))                      # 3月末は 100銘柄 < 200 → 両方欠測
        self.assertTrue(np.isnan(out.loc[6, "rel_ind_per"]))
        self.assertTrue(out["rel_mkt_net_margin"].isna().all())                   # 行の値が欠測なら欠測
        # 元の列はそのまま
        pd.testing.assert_frame_equal(out[frame.columns], frame)

    def test_industry_percentile_uses_only_the_industry(self):
        panel = self.panel()
        frame = pd.DataFrame({"Code": ["00000"], "Date": pd.to_datetime(["2024-03-05"]), "s33_code": [50.0],
                              "per": [250.0]})
        out = RF.attach(frame, panel, metrics=["per"])
        # 2月末: 業種 50 の値は 260..236（25個）。250 より下は 236..249 の 14個、同じ値 1個 → (14+15)/50
        self.assertAlmostEqual(out.loc[0, "rel_ind_per"], 29 / 50)
        self.assertAlmostEqual(out.loc[0, "rel_mkt_per"], (249 + 250) / 520)

    def test_without_industry_column(self):
        panel = self.panel()
        frame = pd.DataFrame({"Code": ["00000"], "Date": pd.to_datetime(["2024-02-05"]), "per": [10.0]})
        out = RF.attach(frame, panel, metrics=["per"])
        self.assertAlmostEqual(out.loc[0, "rel_mkt_per"], 19 / 520)
        self.assertTrue(np.isnan(out.loc[0, "rel_ind_per"]))

    def test_coverage_counts_only(self):
        c = RF.coverage(self.panel(), metrics=["per", "net_margin"])
        self.assertEqual(c["month_ends"], 3)
        self.assertEqual((c["first"], c["last"]), ("2024-01-31", "2024-03-31"))
        self.assertEqual(c["names"], {"min": 100, "median": 260.0, "max": 260})
        self.assertEqual(c["industries"], 3)
        self.assertEqual(c["industries_ok"], {"min": 1, "median": 2.0, "max": 2})   # 1・2月は 50 と 70、3月は 50 だけ
        self.assertAlmostEqual(c["with_industry"], 1.0)
        self.assertAlmostEqual(c["fill"]["per"], 1.0)
        self.assertAlmostEqual(c["fill"]["net_margin"], 0.5)                        # 1・2月は半分、3月は全部 → 中央値 0.5


if __name__ == "__main__":
    unittest.main()
