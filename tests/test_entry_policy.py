#!/usr/bin/env python3
"""
research/entry_policy.py の単体テスト（合成データのみ・ネットワークなし）。

  python3 tests/test_entry_policy.py

約定判定は方策比較の結論を丸ごと左右する。ここが1つずれると
「押し目待ちが勝つ」という誤った結論がそのまま出てくるので、
境界（指値ちょうど・指値より下で寄る・期限の当日）を個別に固定する。
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import entry_policy as EP  # noqa: E402

N_TAIL = EP.EXIT_HORIZON + EP.EXIT_WINDOW + 2   # 手仕舞い価格が作れる長さ


def panel(rows, code="10000", start="2020-01-06"):
    """
    rows: (open, high, low, close, is_fresh_break) の並び。
    手仕舞い価格を作れるように、末尾へ終値一定の行を足す。
    """
    d = dt.date.fromisoformat(start)
    recs = []
    for i, (o, h, lo, c, br) in enumerate(rows):
        recs.append({"Code": code, "Date": pd.Timestamp(d + dt.timedelta(days=i)),
                     "open": o, "high": h, "low": lo, "close": c,
                     "is_fresh_break": br, "high52w": 100.0, "tv_ma20": 5.0})
    last = rows[-1][3]
    for j in range(N_TAIL):
        i = len(rows) + j
        recs.append({"Code": code, "Date": pd.Timestamp(d + dt.timedelta(days=i)),
                     "open": last, "high": last, "low": last, "close": last,
                     "is_fresh_break": False, "high52w": 100.0, "tv_ma20": 5.0})
    return pd.DataFrame(recs)


def flat(open_=100.0, low=100.0, close=100.0, high=100.0):
    return (open_, high, low, close, False)


class TestBuildEvents(unittest.TestCase):
    def test_forward_columns_and_exit(self):
        rows = [flat(), (100, 105, 95, 100, True), (110, 115, 90, 112, False),
                (112, 120, 108, 118, False)]
        ev, funnel = EP.build_events(panel(rows))
        self.assertEqual(len(ev), 1)
        r = ev.iloc[0]
        self.assertEqual(r["open1"], 110)
        self.assertEqual(r["low1"], 90)
        self.assertEqual(r["close1"], 112)
        self.assertEqual(r["open2"], 112)
        # 末尾は終値一定なので、t+56〜t+60 の平均はその値になる
        self.assertAlmostEqual(r["exit_price"], 118.0)
        self.assertAlmostEqual(r["gap_pct"], 10.0)
        self.assertEqual(funnel["新規ブレイク"], 1)
        self.assertEqual(funnel["翌営業日に値が付いた"], 1)

    def test_exit_is_five_day_mean(self):
        # 最後の5日だけ値を変えて、平均になっていることを確かめる
        rows = [flat(), (100, 105, 95, 100, True)]
        p = panel(rows)
        # 基準日は index 1。t+56〜t+60 は index 57〜61
        p.loc[57:61, "close"] = [10.0, 20.0, 30.0, 40.0, 50.0]
        ev, _ = EP.build_events(p)
        self.assertAlmostEqual(ev.iloc[0]["exit_price"], 30.0)

    def test_does_not_cross_codes(self):
        a = panel([flat(), (100, 105, 95, 100, True)], code="10000")
        b = panel([flat(), flat()], code="20000")
        b["close"] = 999.0
        ev, _ = EP.build_events(pd.concat([a, b], ignore_index=True))
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev.iloc[0]["exit_price"], 100.0)

    def test_liquidity_filter(self):
        p = panel([flat(), (100, 105, 95, 100, True)])
        p["tv_ma20"] = 0.05
        ev, funnel = EP.build_events(p, min_trading_value=0.1)
        self.assertEqual(len(ev), 0)
        self.assertEqual(funnel["20日平均売買代金>=0.1億円"], 0)

    def test_missing_next_open_is_dropped(self):
        rows = [flat(), (100, 105, 95, 100, True)]
        p = panel(rows)
        p.loc[2, "open"] = np.nan
        ev, funnel = EP.build_events(p)
        self.assertEqual(len(ev), 0)
        self.assertEqual(funnel["手仕舞い価格が確定"], 1)
        self.assertEqual(funnel["翌営業日に値が付いた"], 0)


class TestOpenPolicy(unittest.TestCase):
    def setUp(self):
        rows = [flat(), (100, 105, 95, 100, True), (110, 115, 105, 112, False)]
        self.ev, _ = EP.build_events(panel(rows))
        self.costs = EP.Costs(fee_pct=0.0, slack_bp=0.0)

    def test_buys_at_open(self):
        r = EP.execute(self.ev, EP.Policy("A", "open"), self.costs, True)
        self.assertTrue(r["filled"][0])
        self.assertEqual(r["entry"][0], 110)

    def test_gap_cap_skips(self):
        # ギャップは +10%。上限5%なら見送り、上限10%なら買う（境界は含む）
        skip = EP.Policy("cap5", "open", gap_cap=5.0)
        take = EP.Policy("cap10", "open", gap_cap=10.0)
        self.assertFalse(EP.execute(self.ev, skip, self.costs, True)["filled"][0])
        self.assertTrue(EP.execute(self.ev, take, self.costs, True)["filled"][0])

    def test_skipped_return_is_zero(self):
        ret = EP.returns_of(self.ev, EP.Policy("cap5", "open", gap_cap=5.0),
                            self.costs, True)
        self.assertEqual(ret[0], 0.0)


class TestDipPolicy(unittest.TestCase):
    def costs(self, slack=0.0, fee=0.0):
        return EP.Costs(fee_pct=fee, slack_bp=slack)

    def ev_for(self, day1, day2=None):
        """day*: (open, high, low, close)"""
        rows = [flat(), (100, 105, 95, 100, True), (*day1, False)]
        if day2:
            rows.append((*day2, False))
        ev, _ = EP.build_events(panel(rows))
        return ev

    def test_fills_at_limit_anchored_on_open(self):
        # 始値100 → 指値 99。安値98 なので約定、価格は指値の 99
        ev = self.ev_for((100, 101, 98, 100))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        r = EP.execute(ev, pol, self.costs(), True)
        self.assertTrue(r["filled"][0])
        self.assertAlmostEqual(r["entry"][0], 99.0)

    def test_no_fill_when_low_above_limit(self):
        ev = self.ev_for((100, 105, 99.5, 104))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        self.assertFalse(EP.execute(ev, pol, self.costs(), True)["filled"][0])

    def test_exact_touch_optimistic_vs_pessimistic(self):
        # 安値がちょうど指値。楽観は約定、悲観（余裕10bp）は約定しない
        ev = self.ev_for((100, 105, 99.0, 104))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        self.assertTrue(EP.execute(ev, pol, self.costs(slack=10), True)["filled"][0])
        self.assertFalse(EP.execute(ev, pol, self.costs(slack=10), False)["filled"][0])

    def test_gap_down_below_limit_fills_at_open(self):
        # 前夜に基準日終値100から -1% = 99 の指値。翌日は 95 で寄る。
        # 99 では買えない。約定価格は寄り値の 95
        ev = self.ev_for((95, 96, 94, 95))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="close")
        r = EP.execute(ev, pol, self.costs(), True)
        self.assertTrue(r["filled"][0])
        self.assertAlmostEqual(r["entry"][0], 95.0)

    def test_anchor_open_never_fills_at_the_open_itself(self):
        # 始値を見てから置く指値は、定義上いつも始値より下にある
        ev = self.ev_for((100, 100, 100, 100))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        self.assertFalse(EP.execute(ev, pol, self.costs(), True)["filled"][0])

    def test_second_day_fill(self):
        ev = self.ev_for((100, 105, 99.5, 104), (104, 106, 98.0, 100))
        one = EP.Policy("h1", "dip", depth=1.0, hold=1, anchor="open")
        two = EP.Policy("h2", "dip", depth=1.0, hold=2, anchor="open")
        self.assertFalse(EP.execute(ev, one, self.costs(), True)["filled"][0])
        r = EP.execute(ev, two, self.costs(), True)
        self.assertTrue(r["filled"][0])
        self.assertAlmostEqual(r["entry"][0], 99.0)

    def test_chase_uses_close_of_deadline_day(self):
        ev = self.ev_for((100, 105, 99.5, 104), (104, 106, 103.0, 107))
        pol = EP.Policy("chase", "dip", depth=1.0, hold=2,
                        anchor="open", chase=True)
        r = EP.execute(ev, pol, self.costs(), True)
        self.assertTrue(r["filled"][0])
        self.assertAlmostEqual(r["entry"][0], 107.0)   # 2日目の終値

    def test_unfilled_return_is_zero_not_negative(self):
        ev = self.ev_for((100, 105, 99.5, 104))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        self.assertEqual(EP.returns_of(ev, pol, self.costs(), True)[0], 0.0)

    def test_fee_is_subtracted_only_when_filled(self):
        ev = self.ev_for((100, 101, 98, 100))
        pol = EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")
        free = EP.returns_of(ev, pol, self.costs(fee=0.0), True)[0]
        paid = EP.returns_of(ev, pol, self.costs(fee=0.2), True)[0]
        self.assertAlmostEqual(free - paid, 0.2)


class TestSummaryAndBootstrap(unittest.TestCase):
    def make(self, n_days=40, seed=3):
        """複数日・複数銘柄の合成イベント。"""
        rng = np.random.default_rng(seed)
        frames = []
        for c in range(6):
            base = 100.0
            rows = [flat(base)]
            for _ in range(2):
                rows.append((base, base * 1.05, base * 0.95, base, True))
                rows.append((base * (1 + rng.normal(0, 0.02)),
                             base * 1.06, base * 0.94,
                             base * (1 + rng.normal(0, 0.02)), False))
            frames.append(panel(rows, code=f"{10000 + c}"))
        p = pd.concat(frames, ignore_index=True)
        ev, _ = EP.build_events(p)
        return ev

    def test_summary_fields(self):
        ev = self.make()
        s = EP.summarize(ev, EP.Policy("A", "open"), EP.Costs(0.0, 0.0), True)
        self.assertEqual(s["n"], len(ev))
        self.assertAlmostEqual(s["fill_rate"], 100.0)
        self.assertAlmostEqual(s["entry_improve"], 0.0)      # 寄りで買うので改善0
        self.assertEqual(s["missed_rate"], 0.0)              # 未約定が無い

    def test_bootstrap_zero_for_identical(self):
        ev = self.make()
        a = EP.returns_of(ev, EP.Policy("A", "open"), EP.Costs(0.0, 0.0), True)
        mat = np.column_stack([a, a])
        d = EP.bootstrap_means(ev, mat, baseline=0, n_boot=200, seed=1)[1]
        self.assertAlmostEqual(d["diff"], 0.0)
        self.assertAlmostEqual(d["lo"], 0.0)
        self.assertAlmostEqual(d["hi"], 0.0)

    def test_bootstrap_resamples_dates_not_rows(self):
        # 日をまるごと入れ替えるので、同じ日の行はいつも一緒に動く。
        # 日が1日しか無ければ、どの復元抽出でも平均は変わらない
        ev = self.make()
        one_day = ev[ev["Date"] == ev["Date"].iloc[0]].copy()
        self.assertGreater(len(one_day), 1)
        a = EP.returns_of(one_day, EP.Policy("A", "open"), EP.Costs(0.0, 0.0), True)
        mat = np.column_stack([np.zeros(len(one_day)), a])
        d = EP.bootstrap_means(one_day, mat, baseline=0, n_boot=200, seed=1)[1]
        self.assertTrue(np.isnan(d["lo"]) or abs(d["lo"] - d["hi"]) < 1e-9)

    def test_bootstrap_widens_with_more_variation(self):
        ev = self.make()
        rng = np.random.default_rng(0)
        base = np.zeros(len(ev))
        narrow = rng.normal(0, 0.1, len(ev))
        wide = rng.normal(0, 5.0, len(ev))
        a = EP.bootstrap_means(ev, np.column_stack([base, narrow]), 0,
                               n_boot=400, seed=2)[1]
        b = EP.bootstrap_means(ev, np.column_stack([base, wide]), 0,
                               n_boot=400, seed=2)[1]
        self.assertGreater(b["hi"] - b["lo"], a["hi"] - a["lo"])

    def test_compare_includes_baseline_row(self):
        ev = self.make()
        pols = [EP.Policy("A", "open"),
                EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")]
        rows = EP.compare(ev, pols, EP.Costs(0.0, 0.0), True, n_boot=100)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["vs_base"]["diff"], 0.0)

    def test_flip_check_detects_sign_change(self):
        ev = self.make()
        pols = [EP.Policy("A", "open"),
                EP.Policy("dip", "dip", depth=1.0, hold=1, anchor="open")]
        rows = EP.flip_check(ev, pols, EP.Costs(0.0, 25.0))
        by = {r["policy"]: r for r in rows}
        self.assertIn("dip", by)
        self.assertIn("flips", by["dip"])
        # 楽観のほうが約定しやすいので、約定率は楽観 >= 悲観
        fo = EP.execute(ev, pols[1], EP.Costs(0.0, 25.0), True)["filled"].sum()
        fp = EP.execute(ev, pols[1], EP.Costs(0.0, 25.0), False)["filled"].sum()
        self.assertGreaterEqual(fo, fp)


class TestPolicyGrid(unittest.TestCase):
    def test_grid_is_fixed_and_first_is_baseline(self):
        pols = EP.default_policies()
        self.assertEqual(pols[0].name, "寄り成行")
        self.assertEqual(pols[0].kind, "open")
        self.assertEqual(len(pols), len(set(p.name for p in pols)))

    def test_decision_time_split(self):
        self.assertEqual(
            EP.Policy("x", "dip", depth=1, anchor="close").decision_time, "T0 寄り前")
        self.assertEqual(
            EP.Policy("x", "dip", depth=1, anchor="open").decision_time, "T1 寄り後")
        self.assertEqual(EP.Policy("x", "open").decision_time, "T1 寄り後")


if __name__ == "__main__":
    unittest.main(verbosity=2)
