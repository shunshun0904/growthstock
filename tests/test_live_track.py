#!/usr/bin/env python3
"""
実運用の追跡（research/live_track.py）の単体テスト。

実運用と OOF の見込みは同じ関数で数えるので、ここで固定する約束が
そのまま比べ方の約束になる。
  - 規則の値は画面（src/lib/strategy.js の STRATEGY・RULE_MODELS）と同じ。2026-10-30 から 1.2σ の3本がすべて95以上
    （2026-10-23〜10-29 も同じ。2026-10-12〜10-23 と 2026-10-29〜10-30 は 1.5σ の3本がすべて95以上。PREV_RULES に参考として並べる）
  - 3モデルのどれかの百分位が欠けた候補は基準を満たさない（画面と同じ）
  - 買うのは発火8件以上の日の、3モデルの最小の百分位が高い順に上位2件
  - 以前の規則（1.2σ の3本がすべて90以上）と月2件の先着は参考として数える
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


def cands(date, pcts, scores=None, models=L.RULE_MODELS):
    """1日ぶんの候補。pcts は [(1本目, 2本目, 3本目), ...]（既定は今の規則のモデル lgbm / xgb / cat の順）。"""
    scores = scores or [0.5] * len(pcts)
    return pd.DataFrame([{"Date": pd.Timestamp(date), "Code": f"{1000 + i}0",
                          "code": f"{1000 + i}", "score": s,
                          **{f"p_{a}": v for a, v in zip(models, p)}}
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
        # 届かなければ 20営業日目の終値で売る（運用者の実際の売り。2026-10-12 から。画面の手順の文言と同じ）
        self.assertEqual(L.FALLBACK, "close")
        self.assertEqual(val("holdDays"), L.HOLD)
        self.assertEqual(val("maxSlots"), L.SLOTS)
        boost = re.search(r"export const BOOST = \[([^\]]*)\]", js).group(1)
        self.assertEqual(tuple(re.findall(r"'(\w+)'", boost)), L.BOOST)
        rule = re.search(r"export const RULE_MODELS = \[([^\]]*)\]", js).group(1)
        self.assertEqual(tuple(re.findall(r"'(\w+)'", rule)), L.RULE_MODELS)
        self.assertEqual(L.RULE_MODELS, ("lgbm", "xgb", "cat"))
        self.assertEqual(L.RULE_MODELS, L.BOOST)
        self.assertEqual(L.AGREE_PCT, 95.0)
        rule_ja = re.search(r"export const RULE_JA = '([^']*)'", js).group(1)
        self.assertEqual(rule_ja, L.RULE_JA)
        # 以前の規則（参考）: 1.5σ の3本 95以上（2026-10-12〜10-23・2026-10-29〜10-30）と、1.2σ の3本 90以上・発火8件以上の日
        # （2026-10-12 まで）。1.2σ の3本 95以上の 2026-10-23〜10-29 は今の規則と同じなので、今の規則の記録に入る
        self.assertEqual([(m, a, b) for _, m, a, b, _ in L.PREV_RULES],
                         [(("lgbm_k15", "xgb_k15", "cat_k15"), 95.0, 1), (L.BOOST, 90.0, 8)])
        self.assertEqual(L.K15, ("lgbm_k15", "xgb_k15", "cat_k15"))

    def test_pct_matches_production_formula(self):
        rng = np.random.default_rng(0)
        hist = rng.random(997)
        got = L.pct_of(hist[:200], hist)
        want = [round(float((hist < s).mean()) * 100, 1) for s in hist[:200]]
        self.assertEqual(list(got), want)


class Decide(unittest.TestCase):
    def test_top2_by_min_pct_then_score(self):
        rows = cands("2026-09-16", [(98, 97, 99), (99, 96, 97), (97, 96, 99),
                                    (97, 99, 96), (99, 94, 99)] + [(50, 50, 50)] * 3,
                     scores=[0.1, 0.2, 0.3, 0.9, 0.5, 0.1, 0.1, 0.1])
        _, days, picks = L.decide(rows)
        # 最小: 97, 96, 96, 96, 94 → 97 が1番、96 の3件はスコア順で 0.9 が2番（94 は線 95 の下）
        self.assertEqual(picks["code"].tolist(), ["1000", "1003"])
        self.assertEqual(picks["rank"].tolist(), [1, 2])
        self.assertEqual(int(days.iloc[0]["n_pass"]), 4)
        self.assertTrue(bool(days.iloc[0]["buy"]))

    def test_seven_breaks_buys_from_2026_10_12(self):
        # 2026-10-12 から発火の数では見送らない（運用者の決定・実験88）。7件の日も上位2件を買う
        self.assertEqual(L.SKIP_BREAKS, 1)
        rows = cands("2026-09-14", [(99, 99, 99)] * 7)
        _, days, picks = L.decide(rows)
        self.assertEqual(len(picks), 2)
        self.assertTrue(bool(days.iloc[0]["buy"]))
        self.assertIn("買う", days.iloc[0]["verdict"])

    def test_seven_breaks_is_skip_with_previous_line(self):
        # 以前の線（PREV_SKIP_BREAKS = 8。■5 の以前の規則と過去の実験の再現）では見送り
        self.assertEqual(L.PREV_SKIP_BREAKS, 8)
        rows = cands("2026-09-14", [(99, 99, 99)] * 7)
        _, days, picks = L.decide(rows, min_break=L.PREV_SKIP_BREAKS)
        self.assertTrue(picks.empty)
        self.assertFalse(bool(days.iloc[0]["buy"]))
        self.assertIn("見送り", days.iloc[0]["verdict"])
        self.assertEqual(L.breaks_txt(L.SKIP_BREAKS), "発火の数は問わず")
        self.assertEqual(L.breaks_txt(L.PREV_SKIP_BREAKS), "発火8件以上の日に")

    def test_missing_model_never_passes(self):
        rows = cands("2026-09-17", [(99, NAN, 99)] + [(50, 50, 50)] * 8)
        r, days, picks = L.decide(rows)
        self.assertTrue(picks.empty)
        self.assertFalse(bool(r["passed"].any()))
        self.assertIn("判定できない", days.iloc[0]["verdict"])

    def test_exactly_95_passes(self):
        rows = cands("2026-09-17", [(95.0, 95.0, 95.0), (94.9, 99.0, 99.0)] + [(50, 50, 50)] * 6)
        _, _, picks = L.decide(rows)
        self.assertEqual(picks["code"].tolist(), ["1000"])

    def test_default_models_are_the_rule_models(self):
        # 既定の合議は RULE_MODELS（2026-10-30 から 1.2σ の3本）。判定できない日の文言は「3モデル」（画面と同じ）
        rows = cands("2026-09-17", [(99, NAN, 99)] + [(50, 50, 50)] * 8)
        rows["p_logit"] = 10.0                                 # 余計な列があっても見ない
        _, days, picks = L.decide(rows)
        self.assertTrue(picks.empty)
        self.assertIn("3モデルの百分位が無い", days.iloc[0]["verdict"])
        # 1.5σ の3本（2026-10-29〜10-30 の規則）の列だけでは、今の規則では誰も通らない（列が無ければ欠けと同じ。判定できない日）
        old = cands("2026-09-17", [(99, 99, 99)] * 8, models=L.K15)
        _, od, op = L.decide(old)
        self.assertTrue(op.empty)
        self.assertIn("判定できない", od.iloc[0]["verdict"])
        _, _, kp = L.decide(old, agree=95.0, models=L.K15)
        self.assertEqual(len(kp), 2)
        # 2026-10-12 までの規則（1.2σ の3本が90以上）なら 92 でも通る。今の線 95 では通らない
        prev = cands("2026-09-17", [(92, 92, 92)] * 8, models=L.BOOST)
        _, _, pp = L.decide(prev, agree=L.PREV_AGREE, models=L.PREV_MODELS)
        self.assertEqual(len(pp), 2)
        self.assertTrue(L.decide(prev)[2].empty)

    def test_all5_models(self):
        # models を5つにすると、logit の 10 で全件が落ちる（実験59 の呼び方）
        rows = cands("2026-09-17", [(99, NAN, 99)] + [(50, 50, 50)] * 8, models=L.BOOST)
        rows["p_logit"] = 10.0
        rows["p_mlp"] = 99.0
        rows.loc[0, "p_xgb"] = 99.0
        _, _, p5 = L.decide(rows, models=L.ALL5)
        self.assertTrue(p5.empty)
        rows["p_logit"] = 99.0
        _, _, p5 = L.decide(rows, models=L.ALL5)
        self.assertEqual(len(p5), 1)


class Forward(unittest.TestCase):
    def keys(self, bars, i=0):
        return pd.DataFrame({"Code": [bars["Code"].iloc[0]], "Date": [bars["Date"].iloc[i]]})

    def test_hit_on_day_k(self):
        n = 30
        h = [105.0] * n
        h[5] = 111.0                       # 選定日の5営業日後 = 買って5日目に +10% に届く
        b = make_bars("11110", "2026-01-05", [100.0] * n, h, [100.0] * n)
        f = L.forward(b, self.keys(b), target=10.0, fallback="close").iloc[0]
        self.assertEqual(f["status"], L.HIT)
        self.assertEqual(f["hit_day"], 5)
        self.assertEqual(f["ret"], 10.0)
        self.assertEqual(f["exit_date"], b["Date"].iloc[5])
        self.assertEqual(f["entry"], 100.0)

    def test_hit_with_other_target(self):
        n = 30
        h = [110.0] * n
        h[5] = 121.0                       # +20% には5日目で届く（+10% なら1日目）
        b = make_bars("11110", "2026-01-05", [100.0] * n, h, [100.0] * n)
        f20 = L.forward(b, self.keys(b), target=20.0).iloc[0]
        self.assertEqual((f20["hit_day"], f20["ret"]), (5, 20.0))
        f10 = L.forward(b, self.keys(b), target=10.0).iloc[0]
        self.assertEqual((f10["hit_day"], f10["ret"]), (1, 10.0))   # 高値 110 = 買値 100 の +10% ちょうど（浮動小数点の誤差で取りこぼさない）

    def test_hit_on_buy_day(self):
        n = 30
        h = [110.0] * n
        h[1] = 125.0
        b = make_bars("11110", "2026-01-05", [100.0] * n, h, [100.0] * n)
        f = L.forward(b, self.keys(b)).iloc[0]
        self.assertEqual(f["hit_day"], 1)

    def test_no_hit_sells_at_day20_close(self):
        n = 30
        c = [100.0 + k for k in range(n)]
        o = [100.0] * n
        o[1] = 110.0                       # 買値。高値 115 は +10% に届かない
        b = make_bars("11110", "2026-01-05", o, [115.0] * n, c)
        f = L.forward(b, self.keys(b), target=10.0, fallback="close").iloc[0]
        self.assertEqual(f["status"], "満了")
        ma5 = np.mean(c[16:21])            # 20営業日目までの5日平均
        self.assertAlmostEqual(f["ret"], (c[20] / 110.0 - 1) * 100)          # 届かなければ 20営業日目の終値
        self.assertAlmostEqual(f["ret_close"], (c[20] / 110.0 - 1) * 100)
        self.assertAlmostEqual(f["ret_hold"], (ma5 / 110.0 - 1) * 100)
        self.assertEqual(f["exit_date"], b["Date"].iloc[20])
        # 5日平均で数える出口（実験32〜35 と記録した実験の数え方）
        f_old = L.forward(b, self.keys(b), target=20.0, fallback="ma5").iloc[0]
        self.assertAlmostEqual(f_old["ret"], (ma5 / 110.0 - 1) * 100)
        with self.assertRaises(ValueError):
            L.forward(b, self.keys(b), fallback="open")

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


class Bars(unittest.TestCase):
    def test_load_bars_extra_columns(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            b = make_bars("11110", "2026-01-05", [100.0, 101.0], [102.0, 103.0], [100.0, 102.0])
            b["O"] = [500.0, 505.0]                            # 調整前の始値（後の分割で 1/5 になった想定）
            b["AdjL"] = [99.0, 100.0]
            b.to_parquet(os.path.join(d, "bars_2026.parquet"), index=False)
            got = L.load_bars(d)
            self.assertEqual(list(got.columns), ["Date", "Code", "AdjO", "AdjH", "AdjC"])
            got2 = L.load_bars(d, extra=("O", "AdjO"))         # 重複は増やさない
            self.assertEqual(list(got2.columns), ["Date", "Code", "AdjO", "AdjH", "AdjC", "O"])
            self.assertEqual(got2["O"].tolist(), [500.0, 505.0])
            self.assertEqual(got2["Code"].dtype, object)


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

    def cand(self, date, code, p, k15=None):
        per = {a: {"pctHistorical": p} for a in L.BOOST}
        if k15 is not None:
            per.update({a: {"pctHistorical": k15} for a in L.K15})
        return {"date": date, "jqCode": f"{code}0", "code": code, "name": code, "score": 0.5, "byModel": per}

    def test_latest_before_next_open(self):
        days = [dt.date(2026, 9, d) for d in (15, 16, 17, 18)]
        vers = [
            self.version("a" * 40, "2026-09-16 13:00", [self.cand("2026-09-16", "1111", 91)]),
            self.version("b" * 40, "2026-09-16 18:00", [self.cand("2026-09-16", "1111", 95, k15=97)]),
            # 翌営業日の寄り（9/17 0:00 UTC）より後。判断には使えない
            self.version("c" * 40, "2026-09-17 01:00", [self.cand("2026-09-16", "1111", 50),
                                                         self.cand("2026-09-15", "2222", 99)]),
        ]
        r = L.live_rows(vers, days)
        d16 = r[r["Date"] == pd.Timestamp("2026-09-16")].iloc[0]
        self.assertEqual(d16["sha"], "bbbbbbb")
        self.assertEqual(d16["p_lgbm"], 95)
        self.assertEqual(d16["p_lgbm_k15"], 97)                # 以前の規則（参考）の 1.5σ の3本も読む
        self.assertFalse(bool(d16["late"]))
        # 9/15 の版には 1.5σ の百分位が無い → 欠け（NaN）。今の規則（1.2σ の3本）の記録は 9/15 から、1.5σ の参考は 9/16 から
        self.assertTrue(np.isnan(r[r["Date"] == pd.Timestamp("2026-09-15")].iloc[0]["p_cat_k15"]))
        self.assertEqual(L.rule_start(r), pd.Timestamp("2026-09-15"))
        self.assertEqual(L.rule_start(r, L.K15), pd.Timestamp("2026-09-16"))
        self.assertIsNone(L.rule_start(r[r["Date"] == pd.Timestamp("2026-09-15")], L.K15))
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


class References(unittest.TestCase):
    def test_monthly_first_two(self):
        rows = pd.concat([cands("2026-10-13", [(96, 96, 96), (99, 98, 97), (50, 50, 50)]),      # 発火3件でも数える
                          cands("2026-10-14", [(99, 99, 99)] + [(50, 50, 50)] * 7),
                          cands("2026-11-02", [(95, 95, 95)])], ignore_index=True)
        r, _, _ = L.decide(rows)
        mf = L.monthly_first(r)
        self.assertEqual(mf["Date"].dt.strftime("%m-%d").tolist(), ["10-13", "10-13", "11-02"])
        self.assertEqual(mf["p_min"].tolist(), [97.0, 96.0, 95.0])            # 同じ日は最小の百分位の順
        self.assertEqual(mf["rank"].tolist(), [1, 2, 1])
        self.assertTrue(L.monthly_first(r.iloc[0:0].assign(passed=pd.Series(dtype=bool))).empty)

    def test_ref_summary_counts_only_complete(self):
        bar_days = list(pd.bdate_range("2026-01-05", periods=60))
        picks = pd.DataFrame({"Date": [bar_days[0], bar_days[1], bar_days[30]], "Code": ["10000", "20000", "30000"],
                              "rank": [1, 1, 1], "status": ["満了", "指値到達", "保有中"], "hit_day": [NAN, 3.0, NAN],
                              "ret": [-2.0, 10.0, NAN]})
        st = L.ref_summary(picks, bar_days, complete_to=bar_days[10], slots=None)
        self.assertEqual((st["picks"], st["taken"], st["done"]), (3, 3, 2))
        self.assertAlmostEqual(st["mean"], 4.0)
        self.assertAlmostEqual(st["hit"], 50.0)
        st1 = L.ref_summary(picks, bar_days, complete_to=bar_days[10], slots=1)   # 1枠: 2件目は枠が満杯
        self.assertEqual((st1["taken"], st1["done"]), (2, 1))


class BothUse(unittest.TestCase):
    """両方使う形（参考。今の規則 かつ その日の中でランキング学習が上半分。■7）。画面の passesBoth と同じ判定。"""

    def test_live_rows_read_ltr(self):
        def cand(code, upper=None, day=None):
            c = {"date": "2026-10-26", "jqCode": f"{code}0", "code": code, "name": code, "score": 0.5,
                 "byModel": {a: {"pctHistorical": 96} for a in L.BOOST}}
            if upper is not None:
                c["ltr"] = {"upper": upper, "dayPct": day, "dayRank": 1, "nDay": 3}
            return c
        vers = [{"sha": "a" * 40, "committed": pd.Timestamp("2026-10-26 10:00", tz="UTC"),
                 "data": {"model": {}, "candidates": [cand("1111", True, 100.0), cand("2222", False, 0.0),
                                                      cand("3333")]}}]
        r = L.live_rows(vers, [dt.date(2026, 10, 26), dt.date(2026, 10, 27)]).set_index("code")
        self.assertEqual(r.loc["1111", "ltr_upper"], 1.0)
        self.assertEqual(r.loc["2222", "ltr_upper"], 0.0)
        self.assertEqual(r.loc["1111", "ltr_day"], 100.0)
        self.assertTrue(np.isnan(r.loc["3333", "ltr_upper"]))
        self.assertEqual(L.col_start(r.reset_index(), "ltr_upper"), pd.Timestamp("2026-10-26"))
        self.assertIsNone(L.col_start(r.reset_index().iloc[0:0], "ltr_upper"))

    def test_decide_with_day_col(self):
        d1 = cands("2026-10-26", [(99, 99, 99), (98, 98, 98), (97, 97, 97), (50, 50, 50)],
                   scores=[0.9, 0.8, 0.7, 0.6]).assign(ltr_upper=[0.0, 1.0, 1.0, 1.0])
        r, days, picks = L.decide(d1, day_col="ltr_upper")
        self.assertEqual(sorted(r.loc[r["passed"], "Code"]), ["10010", "10020"])
        self.assertEqual(picks["Code"].tolist(), ["10010", "10020"])       # 両方使う形の中で最小の百分位の順
        self.assertEqual(picks["rank"].tolist(), [1, 2])
        self.assertTrue(bool(days["buy"].iloc[0]))
        self.assertNotIn("_missing", r.columns)
        # 今の規則だけなら上位2件は 99 と 98（ランキング学習は見ない）
        _, _, rp = L.decide(d1)
        self.assertEqual(rp["Code"].tolist(), ["10000", "10010"])
        # ランキング学習の順位が欠けた候補が1件でもあれば判定できない（買わない）
        d2 = d1.assign(ltr_upper=[1.0, 1.0, np.nan, 0.0])
        _, days2, picks2 = L.decide(d2, day_col="ltr_upper")
        self.assertFalse(bool(days2["buy"].iloc[0]))
        self.assertIn("ランキング学習の順位", days2["verdict"].iloc[0])
        self.assertEqual(int(days2["n_missing"].iloc[0]), 1)
        # 列ごと無ければ全部欠け
        _, days3, _ = L.decide(d1.drop(columns="ltr_upper"), day_col="ltr_upper")
        self.assertEqual(int(days3["n_missing"].iloc[0]), 4)

    def test_print_both(self):
        import contextlib
        import io
        start = pd.Timestamp("2026-01-05")
        bars = pd.concat([make_bars(f"{1000 + i}0", start, [100.0] * 40, [101.0] * 40,
                                    [100.0 + i] * 40) for i in range(3)], ignore_index=True)
        bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
        rows = cands(bar_days[0], [(99, 99, 99), (98, 98, 98), (97, 97, 97)],
                     scores=[0.9, 0.8, 0.7]).assign(ltr_upper=[0.0, 1.0, 1.0], ltr_day=[0.0, 50.0, 100.0])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = L.print_both(rows, bars, bar_days, complete_to=bar_days[-1])
            none = L.print_both(rows.drop(columns=["ltr_upper"]), bars, bar_days, complete_to=bar_days[-1])
        self.assertIsNone(none)
        self.assertEqual(out["start"], str(bar_days[0].date()))
        self.assertEqual((out["both"]["picks"], out["rule"]["picks"]), (2, 2))
        # 両方使う形は 98・97 を買い（+1%・+2%）、今の規則は 99・98（+0%・+1%）
        self.assertAlmostEqual(out["both"]["mean"], 1.5)
        self.assertAlmostEqual(out["rule"]["mean"], 0.5)
        self.assertIn("両方使う形", buf.getvalue())


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
