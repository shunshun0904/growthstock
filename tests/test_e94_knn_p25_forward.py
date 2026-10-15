#!/usr/bin/env python3
"""
実験94（research/exp/e94_knn_p25_forward.py）: 実験93 の後工程を、決めたあとの判断日だけで確かめ直す。

- 決めごと（決めた日・使う判断日の始まり・判定の日・延長の期限・判断日の最小・選び方・3条件）が決めたとおりであること
  （あとから動かしたら落ちる）
- 決めた時点の値（ORIGIN）が実験93 の記録と同じ
- 判定に使う判断日: 始まりより前の日は入らない。規則の行に欠け（収益か分布が出せない候補）のある日は数えない
- 札: 判定の日より前は途中経過、足りなければ延ばす、期限を過ぎれば打ち切り、足りれば再現 / 再現しない
- 合成データで最後まで回り、銘柄コードを出さない

  python3 tests/test_e94_knn_p25_forward.py
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

import e93_knn_quantile_pick as E93  # noqa: E402
import e94_knn_p25_forward as X  # noqa: E402
import test_e91_wasserstein_knn_pick as T91  # noqa: E402

LINE_T, POOL_T = T91.LINE_T, T91.POOL_T


class TestDecisions(unittest.TestCase):
    def test_fixed_as_decided(self):
        self.assertEqual(X.DECIDED, "2026-10-15")
        self.assertEqual(X.FIRST_DAY, "2026-10-16")
        self.assertEqual(X.INTERIM, "2027-04-15")
        self.assertEqual(X.JUDGE_FROM, "2027-10-15")
        self.assertEqual(X.EXTEND_TO, "2028-04-15")
        self.assertEqual(X.MIN_DAYS, 20)
        self.assertEqual(X.RULE, {"key": "p25_50", "ascending": False})
        self.assertEqual(X.CONDS, ("not_below_M", "p_random", "no_worse_elsewhere"))
        self.assertEqual((E93.MAIN_P, E93.K_MAIN), (25, 50))                 # 実験93 の主と同じ物差し

    def test_origin_matches_e93_record(self):
        self.assertEqual(X.ORIGIN, {"days": 96, "Q_p25": +0.21, "M_p25": -0.90, "R_p25": -1.04, "p_Q": 0.021,
                                    "down_M": 0.06, "up_M": 0.69})

    def test_conditions_are_e93_conditions(self):
        df, dist, split, starts, _ = T91.case(planted=8.0)
        c, _ = E93.collect(df, dist, split, starts, line=LINE_T, pool_line=POOL_T)
        ev = E93.evaluate(c, X.RULE["key"], X.RULE["ascending"], draws=100)
        self.assertEqual(set(ev["conds"]), set(X.CONDS))


class TestForwardDays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.dist, cls.split, cls.starts, _ = T91.case(planted=8.0)
        cls.c, _ = E93.collect(cls.df, cls.dist, cls.split, cls.starts, line=LINE_T, pool_line=POOL_T)
        cls.counts = X.rule_counts(cls.df, cls.split, line=LINE_T)

    def test_counts_match_collect_when_complete(self):
        per = self.c.groupby("Date").size()
        for d, n in list(per.items())[:50]:
            self.assertEqual(int(self.counts.get(pd.Timestamp(d), 0)), int(n))

    def test_filters_by_first_day(self):
        first = str(pd.Timestamp(self.c["Date"].median()).date())
        fwd, info = X.forward_days(self.c, self.counts, first_day=first)
        self.assertGreater(info["days"], 10)
        self.assertTrue((fwd["Date"] >= pd.Timestamp(first)).all())
        self.assertEqual(info["days"], fwd["Date"].nunique())
        self.assertEqual(info["days_after"], info["days"] + info["days_incomplete"])

    def test_incomplete_day_is_dropped(self):
        d = self.c["Date"].iloc[0]
        g = self.c[self.c["Date"] == d]
        self.assertGreaterEqual(len(g), 2)
        c2 = self.c.drop(index=g.index[:1])                                    # その日の候補を1件欠けさせる
        fwd, info = X.forward_days(c2, self.counts, first_day=str(pd.Timestamp(d).date()))
        self.assertFalse((fwd["Date"] == d).any())
        self.assertGreaterEqual(info["days_incomplete"], 1)

    def test_empty(self):
        fwd, info = X.forward_days(self.c.iloc[0:0], self.counts)
        self.assertEqual(info["days"], 0)
        self.assertEqual(len(fwd), 0)


class TestJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, dist, split, starts, _ = T91.case(planted=8.0)
        cls.c, _ = E93.collect(df, dist, split, starts, line=LINE_T, pool_line=POOL_T)

    def test_status(self):
        self.assertEqual(X.status("2027-04-15", 50), "interim")
        self.assertEqual(X.status("2027-10-14", 50), "interim")
        self.assertEqual(X.status("2027-10-15", 19), "short")
        self.assertEqual(X.status("2027-10-15", 20), "judge")
        self.assertEqual(X.status("2028-04-15", 19), "closed")
        self.assertEqual(X.status("2028-04-15", 20), "judge")

    def test_pass_and_interim(self):
        res = X.judge(self.c, "2027-10-15", draws=200)
        self.assertEqual(res["status"], "pass")                                 # 仕込んだデータでは再現
        self.assertEqual(res["label"], "再現")
        res_i = X.judge(self.c, "2026-12-01", draws=200)
        self.assertEqual(res_i["status"], "interim")
        self.assertEqual(res_i["eval"]["days"], res["eval"]["days"])            # 数字は出すが判定しない

    def test_short_and_empty(self):
        few = self.c[self.c["Date"].isin(sorted(self.c["Date"].unique())[:5])]
        self.assertEqual(X.judge(few, "2027-10-15", draws=50)["status"], "short")
        self.assertEqual(X.judge(few, "2028-04-15", draws=50)["status"], "closed")
        empty = X.judge(self.c.iloc[0:0], "2027-10-15")
        self.assertEqual((empty["n_days"], empty["status"]), (0, "short"))

    def test_report_no_codes(self):
        res = X.judge(self.c, "2027-10-15", draws=100)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            X.report(res, {"days_after": 60, "days_incomplete": 1, "days": 59}, draws=100)
        out = buf.getvalue()
        for tag in ("実験94", "条件", "結果: 再現", "扱い"):
            self.assertIn(tag, out)
        for code in self.c["row"].astype(str).unique()[:10]:
            pass
        df, *_ = T91.case(planted=8.0)
        for code in df["Code"].unique()[:80]:
            self.assertNotIn(code, out)


if __name__ == "__main__":
    unittest.main()
