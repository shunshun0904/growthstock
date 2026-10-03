#!/usr/bin/env python3
"""
実験59（research/exp/e59_unit_sim.py）の数え方のテスト。重い計算は回さない。

- 過去分布の百分位は「その日より前」だけで出し、過去の行が足りない日は NaN
- 全期間分布の百分位は live_track.pct_of と同じ
- 規則どおりの出口（rule_exit）: +20% は買った日から高値で、届かなければ 20営業日目の終値
- 枠の模擬（simulate_arm）: 1日1件、翌寄りで買う、腕 A は live_track.forward + simulate と同じ取引
- 乗り換え: 判断は終値、売りは **売る玉の** 翌寄り、買いは候補の翌寄り。持ち切りの対照（kept）を記録
- 腕ごとの「売る枠」の選び方（choose_sell）
- 1単元の買付額 = 調整前の始値 × 株数、円の損益 = 買付額 × 収益率
- 投下資本の曲線・回転率・稼働率
- 5モデルの合議（live_track.decide(models=ALL5)）は、1つでも百分位が欠ければ通らない

  python3 tests/test_e59_unit_sim.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import live_track as L  # noqa: E402
import e59_unit_sim as E  # noqa: E402

NAN = float("nan")
DAYS = list(pd.bdate_range("2024-01-01", periods=60))


def bars_of(code, o, h=None, c=None, raw=None, start=0):
    """code の日足。start 日目から o/h/c を並べる（無い日は行なし = NaN の升）。raw は調整前の始値。"""
    n = len(o)
    h = h if h is not None else [x * 1.01 for x in o]
    c = c if c is not None else list(o)
    raw = raw if raw is not None else list(o)
    return pd.DataFrame({"Date": DAYS[start:start + n], "Code": code, "O": raw,
                         "AdjO": o, "AdjH": h, "AdjC": c})


def pick(day, code, p_min=99.0, rank=1):
    return {"Date": DAYS[day], "Code": code, "rank": rank, "p_min": p_min, "score": 0.5}


class Percentile(unittest.TestCase):
    def test_expanding_uses_only_earlier_days(self):
        dates = ["2021-01-04"] * 3 + ["2021-01-05"] * 2 + ["2021-01-06"]
        scores = [0.1, 0.2, 0.3, 0.25, 0.9, 0.2]
        p = E.pct_expanding(dates, scores, min_hist=3)
        self.assertTrue(np.isnan(p[:3]).all())                # 最初の日は過去が無い
        self.assertAlmostEqual(p[3], 66.7)                    # 過去 {0.1, 0.2, 0.3} のうち 0.25 より低いのは 2つ
        self.assertAlmostEqual(p[4], 100.0)
        self.assertAlmostEqual(p[5], 20.0)                    # 同じ値（0.2）は「低い」に数えない

    def test_expanding_min_hist_and_nan(self):
        dates = ["2021-01-04"] * 2 + ["2021-01-05"] * 2 + ["2021-01-06"]
        scores = [0.1, 0.2, NAN, 0.3, 0.15]
        p = E.pct_expanding(dates, scores, min_hist=3)
        self.assertTrue(np.isnan(p[2]) and np.isnan(p[3]))    # 過去が 2 行 → 判定しない
        self.assertAlmostEqual(p[4], 1 / 3 * 100, places=1)    # 過去 {0.1, 0.2, 0.3}（NaN は入らない）
        p2 = E.pct_expanding(["2021-01-04", "2021-01-05"], [0.5, NAN], min_hist=1)
        self.assertTrue(np.isnan(p2[1]))                      # スコアが NaN の行は NaN

    def test_expanding_matches_brute_force(self):
        rng = np.random.default_rng(0)
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2021-01-04", periods=40), 300))
        scores = rng.random(300)
        p = E.pct_expanding(dates, scores, min_hist=20)
        for i in range(300):
            hist = scores[dates < dates[i]]
            want = round(float((hist < scores[i]).mean()) * 100, 1) if len(hist) >= 20 else NAN
            if np.isnan(want):
                self.assertTrue(np.isnan(p[i]))
            else:
                self.assertAlmostEqual(p[i], want)

    def test_whole_matches_live_track(self):
        s = np.array([0.3, 0.1, NAN, 0.2])
        p = E.pct_whole(s)
        self.assertTrue(np.isnan(p[2]))
        np.testing.assert_allclose(p[[0, 1, 3]], L.pct_of(s[[0, 1, 3]], s[[0, 1, 3]]))


class RuleExit(unittest.TestCase):
    def test_hit_within_hold_then_close(self):
        o = [100.0] * 30
        h = [105.0] * 30
        h[7] = 121.0                                           # 買いが index 1 なら 7 日目に +20%
        c = [100.0 + k for k in range(30)]
        px = E.PriceGrid(bars_of("10000", o, h, c), DAYS, ["10000"])
        self.assertEqual(E.rule_exit(px, 0, 1, 100.0), (7, 20.0, "+20%到達"))
        h[7] = 105.0
        px = E.PriceGrid(bars_of("10000", o, h, c), DAYS, ["10000"])
        k, r, st = E.rule_exit(px, 0, 1, 100.0)
        self.assertEqual((k, st), (20, "満了"))                # 買い index 1 の 20 日目 = index 20
        self.assertAlmostEqual(r, (c[20] / 100.0 - 1) * 100)
        # 20 日目の高値がちょうど +20% なら到達が優先
        h[20] = 120.0
        px = E.PriceGrid(bars_of("10000", o, h, c), DAYS, ["10000"])
        self.assertEqual(E.rule_exit(px, 0, 1, 100.0)[2], "+20%到達")

    def test_missing_close_and_no_future(self):
        o = [100.0] * 25
        px = E.PriceGrid(bars_of("10000", o), DAYS, ["10000"])
        self.assertEqual(E.rule_exit(px, 0, 10, 100.0)[2], "保有中")   # 20日目（index 29）の足が無い
        # 20日目の終値だけ無い → その後の最初の終値
        b = bars_of("10000", [100.0] * 30)
        b = b[b["Date"] != DAYS[20]]
        px = E.PriceGrid(b, DAYS, ["10000"])
        self.assertEqual(E.rule_exit(px, 0, 1, 100.0)[0], 21)


class Simulate(unittest.TestCase):
    """3銘柄の日足。10000 は買ってすぐ +20%、20000 は満了で −10%、30000 は満了で +5%。"""

    def setUp(self):
        o1 = [100.0] * 60
        h1 = [100.0] * 60
        h1[2] = 125.0                                          # 選定 0 → 買い 1 → 2 日目（index 2）に到達
        o2 = [200.0] * 60
        c2 = [200.0] * 60
        for k in range(21, 60):
            c2[k] = 180.0
        o3 = [50.0] * 60
        c3 = [50.0] * 60
        for k in range(23, 60):
            c3[k] = 52.5
        self.bars = pd.concat([bars_of("10000", o1, h1, o1, raw=[500.0] * 60),   # 後の分割で調整後は 1/5
                               bars_of("20000", o2, None, c2),
                               bars_of("30000", o3, None, c3)], ignore_index=True)
        self.px = E.PriceGrid(self.bars, DAYS, ["10000", "20000", "30000"])

    def test_arm_a_matches_live_track(self):
        picks = pd.DataFrame([pick(0, "10000"), pick(1, "20000"), pick(3, "30000")])
        t, counts, pos = E.simulate_arm(picks, self.px, arm="A", unit=100)
        self.assertEqual(t["status"].tolist(), ["+20%到達", "満了", "満了"])
        self.assertEqual(t["days"].tolist(), [2.0, 20.0, 20.0])
        self.assertEqual(t["capital"].tolist(), [50000.0, 20000.0, 5000.0])      # 調整前 500円 × 100株
        self.assertAlmostEqual(t["pnl"].iloc[0], 10000.0)                         # 5万 × +20%
        self.assertAlmostEqual(t["pnl"].iloc[1], -2000.0)                         # 2万 × −10%
        self.assertAlmostEqual(t["pnl"].iloc[2], 250.0)                           # 5千 × +5%
        self.assertEqual(counts["n_swaps"], 0)
        self.assertFalse(pos)
        # live_track の forward + simulate と同じ取引・同じ出口
        fwd = L.forward(self.bars, picks)
        sim = L.simulate(fwd, DAYS)
        self.assertEqual(sim["taken"].tolist(), [L.TAKEN] * 3)
        self.assertEqual(sim["status"].tolist(), t["status"].tolist())
        lt = np.where(sim["status"] == "+20%到達", 20.0, sim["ret_close"])
        np.testing.assert_allclose(t["ret"].to_numpy(), lt)

    def test_full_slots_skip_and_no_double_buy(self):
        picks = pd.DataFrame([pick(1, "20000"), pick(2, "30000"), pick(3, "10000"),
                              pick(4, "20000"),       # 持っている銘柄 → 買い増さない
                              pick(5, "40000")])      # 日足なし → 飛ばす
        picks2 = pd.concat([picks, pd.DataFrame([pick(6, "30000")])], ignore_index=True)
        t, counts, _ = E.simulate_arm(picks2, self.px, arm="A", slots=2)
        # 枠2: 20000（買い 2）と 30000（買い 3）で満杯 → 10000（選定 3）は見送り
        self.assertEqual(counts["n_full"], 1)
        self.assertEqual(counts["n_dup"], 3)                   # 20000 の買い増し、40000、30000 の買い増し
        self.assertEqual(sorted(t["Code"].tolist()), ["20000", "30000"])

    def test_swap_sells_at_own_open_and_records_kept(self):
        # 枠1。20000 を持っている最中（含み 0%）に 10000 が候補 → 腕 B（一番古い）で乗り換え
        o2 = [200.0] * 60
        o2[4] = 190.0                                          # 売る日（選定 3 の翌日 = index 4）の 20000 の始値
        bars = pd.concat([self.bars[self.bars["Code"] != "20000"], bars_of("20000", o2, None, [200.0] * 60)],
                         ignore_index=True)
        px = E.PriceGrid(bars, DAYS, ["10000", "20000", "30000"])
        picks = pd.DataFrame([pick(1, "20000", p_min=97.0), pick(3, "10000", p_min=99.0)])
        t, counts, _ = E.simulate_arm(picks, px, arm="B", slots=1, unit=100)
        self.assertEqual(counts["n_swaps"], 1)
        sold = t[t["status"] == E.SWAP].iloc[0]
        self.assertEqual(sold["Code"], "20000")
        self.assertAlmostEqual(sold["ret"], -5.0)              # 190 / 200 − 1。候補（10000）の始値ではない
        self.assertEqual(sold["exit_date"], DAYS[4])
        self.assertEqual(sold["days"], 2.0)                    # index 2, 3 を持ち越した
        self.assertEqual(sold["kept_status"], "満了")           # 持ち切っていたら 20 日目の終値 200 → 0%
        self.assertAlmostEqual(sold["kept_ret"], 0.0)
        self.assertAlmostEqual(sold["kept_pnl"], 0.0)
        new = t[t["Code"] == "10000"].iloc[0]
        self.assertTrue(bool(new["via_swap"]))
        self.assertEqual(new["buy_date"], DAYS[4])
        # 腕 A なら乗り換えない
        t2, counts2, _ = E.simulate_arm(picks, px, arm="A", slots=1)
        self.assertEqual(counts2["n_swaps"], 0)
        self.assertEqual(counts2["n_full"], 1)
        self.assertEqual(t2["Code"].tolist(), ["20000"])

    def test_swap_without_open_price_is_skipped(self):
        b = self.bars[~((self.bars["Code"] == "20000") & (self.bars["Date"] == DAYS[4]))]
        px = E.PriceGrid(b, DAYS, ["10000", "20000", "30000"])
        picks = pd.DataFrame([pick(1, "20000"), pick(3, "10000")])
        t, counts, _ = E.simulate_arm(picks, px, arm="B", slots=1)
        self.assertEqual(counts["n_swaps"], 0)
        self.assertEqual(counts["n_noopen"], 1)
        self.assertEqual(t["Code"].tolist(), ["20000"])


class ChooseSell(unittest.TestCase):
    def setUp(self):
        # 3枠: 0 = 古い・含み +3%、1 = 新しい・含み −4%、2 = 中間・含み +1%・格が一番低い
        o = [100.0] * 10
        bars = pd.concat([bars_of("10000", o, None, [103.0] * 10),
                          bars_of("20000", o, None, [96.0] * 10),
                          bars_of("30000", o, None, [101.0] * 10)], ignore_index=True)
        self.px = E.PriceGrid(bars, DAYS[:10], ["10000", "20000", "30000"])
        mk = lambda code, buy, pm: {"code": code, "col": self.px.col[code], "buy": buy, "entry": 100.0,
                                    "p_min": pm, "capital": 1e4, "sel": DAYS[buy - 1], "rank": 1,
                                    "slot": 0, "via_swap": False, "rule": (None, NAN, "保有中")}
        self.pos = {0: mk("10000", 1, 98.0), 1: mk("20000", 6, 99.0), 2: mk("30000", 3, 97.0)}
        self.cand = pd.Series({"Code": "40000", "p_min": 97.5, "Date": DAYS[8], "rank": 1})

    def test_each_arm(self):
        t = 8
        self.assertIsNone(E.choose_sell("A", self.pos, self.cand, t, self.px))
        self.assertEqual(E.choose_sell("B", self.pos, self.cand, t, self.px), 0)      # 一番古い
        self.assertEqual(E.choose_sell("C", self.pos, self.cand, t, self.px), 1)      # 含み −4%
        self.assertEqual(E.choose_sell("D", self.pos, self.cand, t, self.px), 2)      # 97.5 は最小 97.0 より上 → 売る
        self.cand["p_min"] = 96.0
        self.assertIsNone(E.choose_sell("D", self.pos, self.cand, t, self.px))        # 96 は 97 より下 → 売らない
        self.cand["p_min"] = 97.5
        self.assertEqual(E.choose_sell("D", self.pos, self.cand, t, self.px), 2)
        # E: 5日以上持って含み 0% 以下 → 20000 は買い 6 で 3 日目なのでまだ対象外 → None
        self.assertIsNone(E.choose_sell("E", self.pos, self.cand, t, self.px, stall_days=5))
        self.assertEqual(E.choose_sell("E", self.pos, self.cand, t, self.px, stall_days=3), 1)
        # F: 含み益 0% 以上の中で一番薄い → 30000（+1%）
        self.assertEqual(E.choose_sell("F", self.pos, self.cand, t, self.px), 2)
        with self.assertRaises(ValueError):
            E.choose_sell("Z", self.pos, self.cand, t, self.px)


class Capital(unittest.TestCase):
    def test_curve_covers_buy_to_last_held_day(self):
        done = pd.DataFrame({"buy_date": [DAYS[1], DAYS[2]], "capital": [100.0, 50.0], "days": [3.0, 20.0]})
        c = E.capital_curve(done, DAYS)
        self.assertEqual(c.iloc[0], 0.0)
        self.assertEqual(c.iloc[1], 100.0)                    # 買った日
        self.assertEqual(c.iloc[2], 150.0)                    # 2本目が重なる
        self.assertEqual(c.iloc[3], 150.0)                    # 1本目の最終日（3日目）
        self.assertEqual(c.iloc[4], 50.0)                     # 枠が空く日には外れている
        self.assertEqual(c.iloc[21], 50.0)                    # 2本目の20日目（index 2+19）
        self.assertEqual(c.iloc[22], 0.0)

    def test_stats_turnover_utilization_and_swap_summary(self):
        done = pd.DataFrame({"buy_date": [DAYS[1], DAYS[2], DAYS[5]], "capital": [100.0, 50.0, 80.0],
                             "days": [3.0, 20.0, 2.0], "ret": [20.0, -4.0, -5.0], "pnl": [20.0, -2.0, -4.0],
                             "status": ["+20%到達", "満了", E.SWAP], "via_swap": [False, False, False],
                             "kept_ret": [NAN, NAN, 3.0], "kept_pnl": [NAN, NAN, 2.4]})
        c = E.capital_curve(done, DAYS)
        lo, hi = DAYS[0], DAYS[-1]
        s = E.stats(done, c, lo, hi, slots=3, counts={"n_full": 4, "n_swaps": 1, "full_days": [DAYS[9]]}, n_pass=9)
        years = (hi - lo).days / E.YEAR_DAYS
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["win"], 100 / 3)
        self.assertAlmostEqual(s["hit"], 100 / 3)
        self.assertAlmostEqual(s["pnl_total"], 14.0)
        self.assertAlmostEqual(s["buy_total"], 230.0)
        self.assertAlmostEqual(s["cap_max"], 150.0)           # index 2,3 の 100 + 50（index 5,6 は 50 + 80 = 130）
        self.assertAlmostEqual(s["turnover"], 230.0 / years / s["cap_max"])
        self.assertAlmostEqual(s["util"], 25.0 / (3 * 60) * 100)
        self.assertEqual((s["n_full"], s["n_swaps"], s["n_pass"]), (4, 1, 9))
        self.assertEqual(s["swap_n"], 1)
        self.assertAlmostEqual(s["swap_realized"], -5.0)
        self.assertAlmostEqual(s["swap_kept"], 3.0)
        self.assertAlmostEqual(s["swap_lost_share"], 100.0)
        self.assertAlmostEqual(s["swap_lost_yen"], 2.4 - (-4.0))
        self.assertNotIn("full_days", s)

    def test_stats_empty(self):
        s = E.stats(E.empty_trades(), pd.Series(dtype=float), DAYS[0], DAYS[-1], slots=3, n_pass=2)
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["n_pass"], 2)
        self.assertTrue(np.isnan(s["mean"]))
        self.assertEqual(s["pnl_total"], 0.0)


class FiveModels(unittest.TestCase):
    def rows(self, pcts):
        return pd.DataFrame([{"Date": pd.Timestamp("2026-09-16"), "Code": f"{1000 + i}0", "score": 0.5 - i * 0.01,
                              "p_lgbm": p[0], "p_xgb": p[1], "p_cat": p[2], "p_logit": p[3], "p_mlp": p[4]}
                             for i, p in enumerate(pcts)])

    def test_all_five_must_pass_and_one_per_day(self):
        rows = self.rows([(99, 99, 99, 99, 95),      # 通る（最小 95）
                          (99, 99, 99, 94.9, 99),    # 通らない（logit 94.9）
                          (99, 99, 99, NAN, 99),     # 通らない（欠け）
                          (99, 99, 99, 99, 99)])     # 通る（最小 99 → 1番）
        r, days, picks = L.decide(rows, agree=95.0, min_break=0, top_k=1, models=L.ALL5)
        self.assertEqual(picks["Code"].tolist(), ["10030"])  # 1日1件: p_min が最大の 1件だけ
        self.assertEqual(int(days.iloc[0]["n_missing"]), 1)
        self.assertEqual(int(days.iloc[0]["n_pass"]), 2)
        _, _, p3 = L.decide(rows, agree=95.0, min_break=0, top_k=4, models=L.BOOST)
        self.assertEqual(len(p3), 4)

    def test_patterns_and_arms(self):
        self.assertEqual([p[0] for p in E.PATTERNS],
                         ["GBDT3 97.5以上", "GBDT3 95以上", "全5モデル 95以上", "全5モデル 90以上"])
        self.assertEqual(E.PATTERNS[0][1], L.BOOST)
        self.assertEqual(E.PATTERNS[2][1], L.ALL5)
        self.assertEqual(E.UNIT, 100)
        self.assertEqual(list(E.ARMS), ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(L.ALL5, ("lgbm", "xgb", "cat", "logit", "mlp"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
