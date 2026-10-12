#!/usr/bin/env python3
"""
実験83（research/exp/e83_tp10_rule.py）。

- 出口: 指値に届いた行は +k% ちょうど（到達日は 1 = 買った日）、届かなければ 20営業日目の終値。+20% の 5日平均の出口も。
  高い指値に届いた行は低い指値にも同じ日以前に届いている（食い違い 0件）。5日平均は ret_o1_20 と一致する
- 運用の通りの売買は、どの水準の指値でも到達日の翌営業日から枠が空く
- 報告に銘柄コードが出ない

  python3 tests/test_e83_tp10_rule.py
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

import lab  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e83_tp10_rule as E  # noqa: E402
import test_e82_and_rule_variants as T82  # noqa: E402


def bars_case():
    """3銘柄 × 45営業日。選定日は1日目（買うのは2日目の寄り = 100）。"""
    days = pd.bdate_range("2024-01-01", periods=45)
    rows = []
    for code, h_fn, c in (
        ("10000", lambda k: 110.5 if k == 3 else 102.0, 101.0),     # +10% に3日目で届く。+15% には届かない
        ("20000", lambda k: 103.0, 99.0),                            # どこにも届かない
        ("30000", lambda k: {2: 106.0, 5: 121.0}.get(k, 104.0), 100.0),  # +5% は2日目、+10〜20% は5日目
    ):
        for k, d in enumerate(days):
            rows.append({"Date": d, "Code": code, "AdjO": 100.0, "AdjH": h_fn(k), "AdjC": c})
    return pd.DataFrame(rows), days


class TestExits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bars, days = bars_case()
        rr = lab.realized_returns(bars)
        df = pd.DataFrame({"Code": ["10000", "20000", "30000"], "Date": [days[0]] * 3})
        df = df.merge(rr[["Code", "Date", E79.OUTCOME]], on=["Code", "Date"], how="left")
        cls.out, cls.chk = E.attach_exits(df, bars)
        cls.o = cls.out.set_index("Code")

    def test_take_profit_levels(self):
        o = self.o
        self.assertEqual(o.loc["10000", "tp10c"], 10.0)
        self.assertEqual(o.loc["10000", "hitday10"], 3)
        self.assertAlmostEqual(o.loc["10000", "tp15c"], 1.0)                  # 届かない: 20日目の終値 101 / 100
        self.assertAlmostEqual(o.loc["10000", "tp20a"], 1.0)                  # 5日平均も 101
        self.assertAlmostEqual(o.loc["20000", "tp10c"], -1.0)
        self.assertFalse(o.loc["20000", "hit5"])
        self.assertEqual(o.loc["30000", "tp5c"], 5.0)
        self.assertEqual(o.loc["30000", "hitday5"], 2)
        self.assertEqual(o.loc["30000", "tp10c"], 10.0)
        self.assertEqual(o.loc["30000", "hitday10"], 5)
        self.assertEqual(o.loc["30000", "tp20c"], 20.0)
        self.assertEqual(o.loc["30000", "tp20a"], 20.0)
        self.assertAlmostEqual(o.loc["30000", "c20"], 0.0)

    def test_checks(self):
        self.assertEqual(self.chk["order_violations"], 0)
        self.assertEqual(self.chk["hold5_match"], self.chk["hold5_both"])
        self.assertEqual(self.chk["hold5_both"], 3)
        self.assertAlmostEqual(self.chk["hit_share"]["10"], 200 / 3)


def synth_with_exits(**kw):
    """実験82 の合成データに、出口の列を作りもので足す（実収益から: +k% 以上なら到達）。"""
    df, folds = T82.synth(**kw)
    r = df[E79.OUTCOME].to_numpy() * 100
    df["c20"] = r
    df["hold5"] = r
    for t in E.TARGETS:
        k = E.tp_key(t)
        hit = r >= t
        df[f"hit{k}"] = hit
        df[f"hitday{k}"] = np.where(hit, 5.0, np.nan)
        df[f"tp{k}c"] = np.where(hit, t, r)
    df["tp20a"] = df["tp20c"]
    return df, folds


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, folds = synth_with_exits()
        split = E79.window_split(cls.df, folds)
        cls.ft = E.build(cls.df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
        cls.ff = E.build(cls.df, [(np.arange(len(cls.df)), E79.pct_matrix(cls.df, E79.MODELS))])
        cls.bar_days = list(pd.bdate_range("2021-04-01", "2024-08-30"))

    def test_frame_has_exits(self):
        for c in ("tp10c", "tp20c", "tp20a", "c20", "hit10", "hitday10", "R0", "A15", "pmin3"):
            self.assertIn(c, self.ft.columns)
        self.assertFalse((self.ft["A15"] & ~self.ft["R0"]).any())
        np.testing.assert_allclose(np.minimum(self.ft["tp10c"], 10.0), self.ft["tp10c"])   # +10% より上は無い

    def test_trade_sim_frees_slots_on_any_target(self):
        n_days = len(self.bar_days)
        st = E.trade_sim(self.ft, "R0", "tp10c", "hit10", "hitday10", self.bar_days, n_days)
        self.assertGreater(st["n"], 0)
        self.assertLessEqual(st["n"], st["picks"])
        st0 = E.trade_sim(self.ft, "R0", "c20", None, None, self.bar_days, n_days)
        self.assertLessEqual(st0["n"], st["n"])          # 早く降りる出口のほうが、買える数は同じか多い

    def test_report_prints_no_codes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(self.ft, bar_days=self.bar_days, ff=self.ff, n_perm=50)
        out = buf.getvalue()
        for tag in ("■ 1.", "■ 2.", "■ 3.", "■ 4.", "■ 5.", "■ 6.", "■ 7.", "■ まとめ"):
            self.assertIn(tag, out)
        self.assertIn("A15_tp10c", s["trade"])
        self.assertIn("judge", s["main"])
        for code in self.df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
