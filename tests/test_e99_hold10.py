#!/usr/bin/env python3
"""
実験99（research/exp/e99_hold10.py）。

- 出口: 保有 H 日ごとに、+10% に届いた日・収益・H 日目の終値が live_track.forward の約束どおり（12 日目に届く行は H=10 では
  10 日目の終値、H=20 では +10%）。時価総額の規則（+5%）は行ごとの指値で測る
- 待つ vs 切り替える: A は 10 日目までに届かなかった行の「10 日目の終値から出口まで」、B は全行の最初の 10 日
- 到達の時間: k 日目までの割合と、10 日までに届かなかった行のうち後半に届く割合
- 3 枠の形: 保有を短くすると買える取引は同じか増える
- 判定: (a)(b) がそろったときだけ「出す」
- 合成データで報告が全部の段を出し、銘柄コードを出さない

  python3 tests/test_e99_hold10.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e79_model_clusters as E79  # noqa: E402
import e99_hold10 as E  # noqa: E402
import live_track as LT  # noqa: E402
import test_e82_and_rule_variants as T82  # noqa: E402


def bars_case():
    """3銘柄 × 45営業日。選定日は days[0]、買うのは翌日の寄り 100。"""
    days = pd.bdate_range("2024-01-01", periods=45)
    rows = []
    for code, h_fn, c in (
        ("10000", lambda k: {4: 106.0, 12: 110.5}.get(k, 102.0), 103.0),   # +5% は4日目、+10% は12日目。終値 103
        ("20000", lambda k: 111.0 if k == 3 else 102.0, 101.0),             # +10% に3日目
        ("30000", lambda k: 103.0, 99.0),                                   # どこにも届かない。終値 99
    ):
        for k, d in enumerate(days):
            rows.append({"Date": d, "Code": code, "AdjO": 100.0, "AdjH": h_fn(k), "AdjC": c})
    return pd.DataFrame(rows), days


class TestAttachHolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bars, days = bars_case()
        df = pd.DataFrame({"Code": ["10000", "20000", "30000"], "Date": [days[0]] * 3})
        out = E.attach_holds(df, bars)
        out = E.attach_holds(out, bars, prefix="cap_", mcap=pd.Series([1500.0, 1500.0, 500.0]))
        cls.o = out.set_index("Code")

    def test_plus10_by_hold(self):
        o = self.o
        self.assertTrue(np.isnan(o.loc["10000", "hitday_h10"]))
        self.assertAlmostEqual(o.loc["10000", "ret_h10"], 3.0)            # 10 日目の終値 103
        self.assertAlmostEqual(o.loc["10000", "close_h10"], 3.0)
        self.assertEqual(o.loc["10000", "hitday_h20"], 12)
        self.assertEqual(o.loc["10000", "ret_h20"], 10.0)
        self.assertAlmostEqual(o.loc["10000", "close_h20"], 3.0)
        self.assertEqual((o.loc["20000", "hitday_h10"], o.loc["20000", "ret_h10"], o.loc["20000", "ret_h20"]), (3, 10.0, 10.0))
        self.assertEqual((o.loc["20000", "hitday_h5"], o.loc["20000", "ret_h5"]), (3, 10.0))
        self.assertAlmostEqual(o.loc["30000", "ret_h10"], -1.0)
        self.assertAlmostEqual(o.loc["30000", "ret_h20"], -1.0)
        self.assertTrue(np.isnan(o.loc["30000", "hitday_h20"]))
        self.assertEqual(o.loc["10000", "tgt"], 10.0)

    def test_cap_rule_uses_the_rows_target(self):
        o = self.o
        self.assertEqual((o.loc["10000", "cap_tgt"], o.loc["30000", "cap_tgt"]), (5.0, 10.0))
        self.assertEqual((o.loc["10000", "cap_hitday_h10"], o.loc["10000", "cap_ret_h10"]), (4, 5.0))
        self.assertAlmostEqual(o.loc["30000", "cap_ret_h10"], -1.0)


class TestPieces(unittest.TestCase):
    def test_wait_vs_switch(self):
        g = pd.DataFrame({"window": [3, 3, 3, 4, 4, 4, 4],
                          "ret_h10": [3.0, -2.0, 10.0, 1.0, 4.0, 10.0, -3.0],
                          "ret_h20": [10.0, -4.0, 10.0, 10.0, 2.0, 10.0, -1.0],
                          "hitday_h20": [12.0, np.nan, 5.0, 15.0, np.nan, 2.0, np.nan]})
        ws = E.wait_vs_switch(g)
        self.assertEqual((ws["A"]["n"], ws["B"]["n"], ws["n_full"]), (5, 7, 7))
        self.assertAlmostEqual(ws["A"]["hit"], 40.0)                        # 11〜20 日目に届いたのは 5 件中 2 件
        seg1 = (1.10 / 1.03 - 1.0) * 100.0
        self.assertAlmostEqual(ws["buckets"]["0〜+5%"]["n"], 3)
        self.assertAlmostEqual(ws["buckets"]["10日目に含み損"]["mean"], (((0.96 / 0.98) - 1) * 100 + ((0.99 / 0.97) - 1) * 100) / 2)
        self.assertAlmostEqual(ws["A"]["mean"], np.mean([seg1, (0.96 / 0.98 - 1) * 100, (1.10 / 1.01 - 1) * 100,
                                                         (1.02 / 1.04 - 1) * 100, (0.99 / 0.97 - 1) * 100]))
        self.assertAlmostEqual(ws["B"]["mean"], np.mean([3, -2, 10, 1, 4, 10, -3]))
        self.assertEqual(ws["diff"]["n"], 0)                                 # 窓あたり 5 行に足りない

    def test_hazard(self):
        g = pd.DataFrame({"hitday_h20": [3.0, 12.0, np.nan, 5.0, 18.0], "close_h20": [1.0, 2.0, 3.0, 4.0, np.nan]})
        hz = E.hazard(g)
        self.assertEqual(hz["n"], 4)                                         # 20 日ぶんの日足がある行だけ
        self.assertAlmostEqual(hz["cum"][10], 50.0)
        self.assertAlmostEqual(hz["cum"][20], 75.0)
        self.assertEqual((hz["n_not10"], hz["n_late"]), (2, 1))
        self.assertAlmostEqual(hz["cond_late"], 50.0)

    def test_gaps(self):
        days = list(pd.bdate_range("2024-01-01", periods=30))
        g = pd.DataFrame({"Date": [days[0], days[1], days[1], days[6], days[20]]})
        gp = E.gaps(g, days)
        self.assertEqual((gp["n"], gp["days_with_rule"]), (3, 4))
        self.assertAlmostEqual(gp["median"], 5.0)
        self.assertAlmostEqual(gp["le1"], 100 / 3)
        self.assertAlmostEqual(gp["le10"], 200 / 3)

    def test_judge(self):
        good = {"n": 9, "mean": 0.3, "se": 0.1, "wins": 6}
        bad = {"n": 9, "mean": 0.3, "se": 0.1, "wins": 4}
        self.assertEqual(E.judge(good, good)["verdict"], "10日の案を運用者に出す")
        self.assertEqual(E.judge(good, bad)["verdict"], "結果を見せて運用者が決める")
        self.assertFalse(E.judge({"n": 0, "mean": float("nan"), "se": float("nan"), "wins": 0}, good)["a"])


def synth_ft():
    """実験82 の合成データに、保有 H 日の列を作りもので足して、前向きの行にする。"""
    df, folds = T82.synth()
    r = df[E79.OUTCOME].to_numpy() * 100
    hit = r >= 10
    for h in E.HOLDS:
        df[f"close_h{h}"] = r * h / 20.0
        df[f"hitday_h{h}"] = np.where(hit, 5.0, np.nan)
        df[f"ret_h{h}"] = np.where(hit, 10.0, r * h / 20.0)
    df["tgt"] = 10.0
    split = E79.window_split(df, folds)
    ft = E.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    return df, ft


class TestTradeAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.ft = synth_ft()
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_frame(self):
        for c in ("min_K15", "score", "n_break", "ret_h10", "hitday_h20", "close_h5", "tgt"):
            self.assertIn(c, self.ft.columns)
        self.assertGreater(len(E.rule_rows(self.ft)), 0)

    def test_shorter_hold_trades_at_least_as_often(self):
        months = int(self.ft["Date"].dt.to_period("M").nunique())
        s10, t10 = E.trade(self.ft, self.bar_days, 10, months)
        s20, t20 = E.trade(self.ft, self.bar_days, 20, months)
        self.assertGreaterEqual(s10["n"], s20["n"])
        self.assertEqual(s10["picks"], s20["picks"])
        self.assertTrue(np.isfinite(s10["capital_pm"]))
        self.assertEqual(sum(v["n"] for v in s10["per_window"].values()), s10["n"])
        d = E.trade_diff(s10, s20, "capital_pm")
        self.assertEqual(d["n"], len(set(s10["per_window"]) & set(s20["per_window"])))

    def test_report_sections_and_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, self.bar_days)
        out = buf.getvalue()
        for tag in ("■ 0.", "■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 6.", "判定"):
            self.assertIn(tag, out)
        self.assertNotIn("■ 5.", out)                                        # 時価総額の列が無ければ出さない
        for k in ("rows", "hazard", "holds", "hold_diffs", "switch", "gaps", "trade", "trade_diffs", "judge"):
            self.assertIn(k, s)
        self.assertEqual(set(s["holds"]), set(E.HOLDS))
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
