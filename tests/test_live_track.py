#!/usr/bin/env python3
"""
実運用の追跡（research/live_track.py）の単体テスト。

実運用と OOF の見込みは同じ関数で数えるので、ここで固定する約束が
そのまま比べ方の約束になる。
  - 規則の値は画面（src/lib/strategy.js の STRATEGY）と同じ
  - 3モデルのどれかの百分位が欠けた候補は基準を満たさない（画面と同じ）
  - 買うのは発火8件以上の日の、3モデルの最小の百分位が高い順に上位2件
  - 買値は翌営業日の始値。+20% は1営業日目（買った日）の高値から見る
  - 届かなければ20営業日目までの5日平均終値（ret_o1_20 と同じ）
  - 枠は3つ。+20% に k営業日目で届けば buy+k から空く。途中の玉は枠をふさぐ
  - 判断に使う版は、翌営業日の寄り（0:00 UTC）より前の最後の版
  - 選定日から20営業日たっていない取引は成績に数えない（勝ち側に偏るため）

  python3 tests/test_live_track.py
"""
import datetime as dt
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import live_track as L  # noqa: E402

NAN = float("nan")


def cands(date, pcts, scores=None):
    """1日ぶんの候補。pcts は [(lgbm, xgb, cat), ...]。"""
    scores = scores or [0.5] * len(pcts)
    return pd.DataFrame([{"Date": pd.Timestamp(date), "Code": f"{1000 + i}0",
                          "code": f"{1000 + i}", "score": s,
                          "p_lgbm": p[0], "p_xgb": p[1], "p_cat": p[2]}
                         for i, (p, s) in enumerate(zip(pcts, scores))])


def make_bars(code, start, o, h, c):
    """営業日（平日）に1本ずつ。o/h/c は同じ長さのリスト。"""
    days = pd.bdate_range(start, periods=len(o))
    return pd.DataFrame({"Date": days, "Code": code, "AdjO": o, "AdjH": h, "AdjC": c})


class RulesMatchScreen(unittest.TestCase):
    def test_constants_match_strategy_js(self):
        with open(os.path.join(ROOT, "src", "lib", "strategy.js"), encoding="utf-8") as fh:
            js = fh.read()
        block = js[js.index("export const STRATEGY"):]
        block = block[:block.index("};")]

        def val(key):
            m = re.search(rf"\b{key}:\s*([0-9.]+)", block)
            self.assertIsNotNone(m, key)
            return float(m.group(1))

        self.assertEqual(val("agreePct"), L.AGREE_PCT)
        self.assertEqual(val("strongBreaks"), L.STRONG_BREAKS)
        self.assertEqual(val("skipBreaks"), L.SKIP_BREAKS)
        self.assertEqual(val("topK"), L.TOP_K)
        self.assertEqual(val("takeProfit"), L.TAKE_PROFIT)
        self.assertEqual(val("holdDays"), L.HOLD)
        self.assertEqual(val("maxSlots"), L.SLOTS)
        boost = re.search(r"export const BOOST = \[([^\]]*)\]", js).group(1)
        self.assertEqual(tuple(re.findall(r"'(\w+)'", boost)), L.BOOST)

    def test_pct_matches_production_formula(self):
        rng = np.random.default_rng(0)
        hist = rng.random(997)
        got = L.pct_of(hist[:200], hist)
        want = [round(float((hist < s).mean()) * 100, 1) for s in hist[:200]]
        self.assertEqual(list(got), want)


class Decide(unittest.TestCase):
    def test_top2_by_min_pct_then_score(self):
        rows = cands("2026-09-16", [(96, 95, 99), (99, 92, 93), (93, 92, 97),
                                    (93, 97, 92), (95, 89, 99)] + [(50, 50, 50)] * 3,
                     scores=[0.1, 0.2, 0.3, 0.9, 0.5, 0.1, 0.1, 0.1])
        _, days, picks = L.decide(rows)
        # 最小: 95, 92, 92, 92, 89 → 95 が1番、92 の3件はスコア順で 0.9 が2番
        self.assertEqual(picks["code"].tolist(), ["1000", "1003"])
        self.assertEqual(picks["rank"].tolist(), [1, 2])
        self.assertEqual(int(days.iloc[0]["n_pass"]), 4)
        self.assertTrue(bool(days.iloc[0]["buy"]))

    def test_seven_breaks_is_skip(self):
        rows = cands("2026-09-14", [(99, 99, 99)] * 7)
        _, days, picks = L.decide(rows)
        self.assertTrue(picks.empty)
        self.assertFalse(bool(days.iloc[0]["buy"]))
        self.assertIn("見送り", days.iloc[0]["verdict"])

    def test_missing_model_never_passes(self):
        rows = cands("2026-09-17", [(99, NAN, 99)] + [(50, 50, 50)] * 8)
        r, days, picks = L.decide(rows)
        self.assertTrue(picks.empty)
        self.assertFalse(bool(r["passed"].any()))
        self.assertIn("判定できない", days.iloc[0]["verdict"])

    def test_exactly_90_passes(self):
        rows = cands("2026-09-17", [(90.0, 90.0, 90.0)] + [(50, 50, 50)] * 7)
        _, _, picks = L.decide(rows)
        self.assertEqual(len(picks), 1)


class Forward(unittest.TestCase):
    def keys(self, bars, i=0):
        return pd.DataFrame({"Code": [bars["Code"].iloc[0]], "Date": [bars["Date"].iloc[i]]})

    def test_hit_on_day_k(self):
        n = 30
        h = [110.0] * n
        h[5] = 121.0                       # 選定日の5営業日後 = 買って5日目
        b = make_bars("11110", "2026-01-05", [100.0] * n, h, [100.0] * n)
        f = L.forward(b, self.keys(b)).iloc[0]
        self.assertEqual(f["status"], "+20%到達")
        self.assertEqual(f["hit_day"], 5)
        self.assertEqual(f["ret"], 20.0)
        self.assertEqual(f["exit_date"], b["Date"].iloc[5])
        self.assertEqual(f["entry"], 100.0)

    def test_hit_on_buy_day(self):
        n = 30
        h = [110.0] * n
        h[1] = 125.0
        b = make_bars("11110", "2026-01-05", [100.0] * n, h, [100.0] * n)
        f = L.forward(b, self.keys(b)).iloc[0]
        self.assertEqual(f["hit_day"], 1)

    def test_no_hit_uses_5day_average_close(self):
        n = 30
        c = [100.0 + k for k in range(n)]
        o = [100.0] * n
        o[1] = 110.0                       # 買値
        b = make_bars("11110", "2026-01-05", o, [115.0] * n, c)
        f = L.forward(b, self.keys(b)).iloc[0]
        self.assertEqual(f["status"], "満了")
        ma5 = np.mean(c[16:21])            # 20営業日目までの5日平均
        self.assertAlmostEqual(f["ret"], (ma5 / 110.0 - 1) * 100)
        self.assertAlmostEqual(f["ret_close"], (c[20] / 110.0 - 1) * 100)
        self.assertEqual(f["exit_date"], b["Date"].iloc[20])

    def test_open_position(self):
        n = 11
        c = [100.0] * n
        c[-1] = 90.0
        b = make_bars("11110", "2026-01-05", [100.0] * n, [105.0] * n, c)
        f = L.forward(b, self.keys(b)).iloc[0]
        self.assertEqual(f["status"], "保有中")
        self.assertTrue(np.isnan(f["ret"]))
        self.assertEqual(f["elapsed"], 10)
        self.assertAlmostEqual(f["ret_now"], -10.0)

    def test_not_bought_yet_and_missing(self):
        b = make_bars("11110", "2026-01-05", [100.0] * 3, [100.0] * 3, [100.0] * 3)
        f = L.forward(b, self.keys(b, i=2)).iloc[0]
        self.assertEqual(f["status"], "未約定")
        k = pd.DataFrame({"Code": ["99990"], "Date": [b["Date"].iloc[0]]})
        self.assertEqual(L.forward(b, k).iloc[0]["status"], "日足なし")

    def test_no_open_price(self):
        o = [100.0, NAN, 100.0, 100.0]
        b = make_bars("11110", "2026-01-05", o, [100.0] * 4, [100.0] * 4)
        self.assertEqual(L.forward(b, self.keys(b)).iloc[0]["status"], "寄り付かず")


class Slots(unittest.TestCase):
    def pick(self, day, rank, status, hit=NAN, ret=0.0, code=None):
        return {"Date": self.days[day], "Code": code or f"{day}{rank}", "rank": rank,
                "status": status, "hit_day": hit, "ret": ret}

    def setUp(self):
        self.days = list(pd.bdate_range("2026-01-05", periods=60))

    def test_first_come_and_release(self):
        p = pd.DataFrame([
            self.pick(0, 1, "+20%到達", hit=3, ret=20.0),   # 枠0: buy=1, 空くのは 4
            self.pick(0, 2, "満了", ret=-1.0),              # 枠1: buy=1, 空くのは 21
            self.pick(1, 1, "満了", ret=2.0),               # 枠2: buy=2, 空くのは 22
            self.pick(1, 2, "満了", ret=5.0),               # 満杯
            self.pick(2, 1, "満了", ret=3.0),               # buy=3 < 4 → 満杯
            self.pick(3, 1, "満了", ret=4.0),               # buy=4 → 枠0 が空く
        ])
        s = L.simulate(p, self.days)
        self.assertEqual(s["taken"].tolist(),
                         [L.TAKEN, L.TAKEN, L.TAKEN, L.FULL, L.FULL, L.TAKEN])
        self.assertEqual(s["slot"].tolist()[5], 0)

    def test_open_position_blocks(self):
        p = pd.DataFrame([self.pick(0, 1, "保有中"), self.pick(0, 2, "保有中"),
                          self.pick(1, 1, "未約定"), self.pick(40, 1, "満了")])
        s = L.simulate(p, self.days)
        self.assertEqual(s["taken"].tolist(), [L.TAKEN, L.TAKEN, L.TAKEN, L.FULL])

    def test_monthly_is_sum_over_slots(self):
        t = pd.DataFrame({"ret": [10.0, -5.0, 20.0], "status": ["満了", "満了", "+20%到達"]})
        st = L.trade_stats(t, n_days=21, slots=3)
        self.assertAlmostEqual(st["monthly"], 25.0 / 3)
        self.assertAlmostEqual(st["hit"], 100 / 3)


class DecisionVersion(unittest.TestCase):
    def version(self, sha, when, rows):
        return {"sha": sha, "committed": pd.Timestamp(when, tz="UTC"),
                "data": {"model": {"trainedAt": "2026-09-13T00:00:00"},
                         "candidates": rows}}

    def cand(self, date, code, p):
        return {"date": date, "jqCode": f"{code}0", "code": code, "name": code, "score": 0.5,
                "byModel": {a: {"pctHistorical": p} for a in L.BOOST}}

    def test_latest_before_next_open(self):
        days = [dt.date(2026, 9, d) for d in (15, 16, 17, 18)]
        vers = [
            self.version("a" * 40, "2026-09-16 13:00", [self.cand("2026-09-16", "1111", 91)]),
            self.version("b" * 40, "2026-09-16 18:00", [self.cand("2026-09-16", "1111", 95)]),
            # 翌営業日の寄り（9/17 0:00 UTC）より後。判断には使えない
            self.version("c" * 40, "2026-09-17 01:00", [self.cand("2026-09-16", "1111", 50),
                                                         self.cand("2026-09-15", "2222", 99)]),
        ]
        r = L.live_rows(vers, days)
        d16 = r[r["Date"] == pd.Timestamp("2026-09-16")].iloc[0]
        self.assertEqual(d16["sha"], "bbbbbbb")
        self.assertEqual(d16["p_lgbm"], 95)
        self.assertFalse(bool(d16["late"]))
        # 9/15 は寄り（9/16 0:00 UTC）の後に初めて出た → 遅れ。最初の版を使う
        d15 = r[r["Date"] == pd.Timestamp("2026-09-15")].iloc[0]
        self.assertTrue(bool(d15["late"]))
        self.assertEqual(d15["sha"], "ccccccc")

    def test_next_open_skips_holidays(self):
        days = [dt.date(2026, 9, 18), dt.date(2026, 9, 24)]     # 連休
        self.assertEqual(L.next_open_utc(dt.date(2026, 9, 18), days),
                         pd.Timestamp("2026-09-24", tz="UTC"))
        # カレンダーの外は次の平日
        self.assertEqual(L.next_trading_day(dt.date(2026, 12, 4), days), dt.date(2026, 12, 7))


class CompleteOnly(unittest.TestCase):
    def test_open_and_recent_trades_excluded(self):
        """+20% に早く届いた最近の取引を入れると勝ち側に偏るので数えない。"""
        bar_days = list(pd.bdate_range("2026-01-05", periods=60))
        rows = pd.concat([cands(bar_days[0], [(99, 99, 99)] + [(50, 50, 50)] * 7),
                          cands(bar_days[30], [(99, 99, 99)] + [(50, 50, 50)] * 7)])
        r, days, picks = L.decide(rows)
        fwd = picks.assign(status=["満了", "+20%到達"], hit_day=[NAN, 2.0],
                           ret=[-2.0, 20.0], ret_now=NAN)
        allf = r.merge(fwd[["Code", "Date", "status", "hit_day", "ret"]],
                       on=["Code", "Date"], how="left")
        st = L.period_stats(r, days, fwd, allf, bar_days, bar_days[0], bar_days[40], 41,
                            complete_to=bar_days[10])
        self.assertEqual(st["trade"]["n"], 1)
        self.assertEqual(st["trade"]["mean"], -2.0)
        self.assertEqual(st["taken"], 2)            # 枠の上では両方買っている


if __name__ == "__main__":
    unittest.main(verbosity=2)
