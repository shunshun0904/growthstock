#!/usr/bin/env python3
"""
実験117（research/exp/e117_rolling5_ltr.py）。合成の日足と候補で:

- 毎日の組: 判断日ごとに、候補の日が判断日の4営業日前〜判断日の行（祝日を飛ばして取引日で数える）。同じ銘柄は最後の候補の日の
  行だけ。Date は判断日、cand_date に候補の日、r5_days は取引日で数えた営業日数。値動き4列（候補の日の終値 → 判断日の終値・
  その間の高値の最大・安値の最小）。判断日に入った候補は r5_ret 0・高値と安値は欠測。取引日でない日付・足す列と同じ名前の列が
  あれば止める
- 確かめ: 組の中身をデータセットから直接選び直したもの・値動きの列の直接の計算し直し・判断日に入った候補の収益（合わなければ数える）
- 探索の分割: 同じ月の判断日は同じ側・どの月もどれか1つの検証側・訓練側に検証側の判断日の前後4営業日が入らない
- 毎日の買い（上位2件のうち R15 を満たす未購入。参考に条件なし）: 手で数える例（同じ候補は1回・持っている銘柄は買い増さない・
  上位2件より下に降りない・上位2件が R15 を満たさなければ買わない）。3枠（live_track.simulate）
- 今の買い（R15）: 候補の日の R15 の行を翌営業日に買う。同じ日は3本の最小の順・同点は基準モデルのスコアの順・持っている銘柄は
  買い増さない（手で数える例）
- R15 の印の付け方（銘柄・候補の日で突き合わせる）と、印が組のすべてに付いた判断日だけを数えること
- 新しさ・読み方（回す前に決めたもの。W5 の買いが R15・D・B のどれよりも平均収益が高く −10% 未満が低いか）
- out-of-fold: W5・W5X は判断日が訓練の最終日以前の組の行、D・B は候補の日が訓練の最終日以前の日次の行で学び、採点はテスト窓の
  組の行。W5 だけ値動きの列を使う。保存した out-of-fold は次の回に読む
- R15（1.5σ の木3本）の印と3本の最小: 本番の out-of-fold から直接計算し直したものと同じ
- 通しで回す（探索は差し替え、当てはめは小さな木）: 集計と読み方を記録に書き、公開ログに銘柄コードを出さない。本番の
  out-of-fold が無ければ止まる

  python3 tests/test_e117_rolling5_ltr.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import live_track as LT  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e115_weekly_ltr as W  # noqa: E402
import e117_rolling5_ltr as R  # noqa: E402
import test_e115_weekly_ltr as T115  # noqa: E402

HOLIDAYS = (pd.Timestamp("2024-01-12"), pd.Timestamp("2024-02-12"))     # 金曜・月曜の祝日
COLS = ["f1"]


def small():
    """祝日を含む短い暦と、組の作りを決める候補。"""
    days = pd.bdate_range("2024-01-01", "2024-04-30").drop(list(HOLIDAYS))
    bars = T115.make_bars(["10010", "20020", "30030", "40040"], days, skip={("40040", pd.Timestamp("2024-01-19"))})
    cands = pd.DataFrame({
        "Code": ["10010", "10010", "20020", "30030", "40040", "10010"],
        "Date": pd.to_datetime(["2024-01-08", "2024-01-10", "2024-01-11", "2024-01-15", "2024-01-16", "2024-04-30"]),
        "f1": np.arange(6, dtype=float), "label": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})
    full = T115.with_daily_exit(cands, bars)
    return days, bars, full


class TestRollingFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days, cls.bars, cls.full = small()
        cls.rl, cls.info = R.rolling_frame(cls.full, cls.bars, cls.days, COLS)

    def row(self, code, t):
        r = self.rl[(self.rl["Code"] == code) & (self.rl["Date"] == pd.Timestamp(t))]
        self.assertEqual(len(r), 1, (code, t))
        return r.iloc[0]

    def px(self, code, d, col):
        b = self.bars
        return float(b.loc[(b["Code"] == code) & (b["Date"] == pd.Timestamp(d)), col].iloc[0])

    def test_keys_and_columns(self):
        rl = self.rl
        self.assertFalse(rl.duplicated(["Code", "Date"]).any())
        self.assertTrue(rl[["Date", "Code"]].equals(rl.sort_values(["Date", "Code"])[["Date", "Code"]]))
        for c in (R.D_RET, R.D_HIT, R.D_R20, "cand_date", "label", "f1") + R.MOVE_COLS:
            self.assertIn(c, rl.columns)
        for c in (E.RET, E.HIT, E.HITDAY, E.R20):
            self.assertNotIn(c, rl.columns)                        # 組の収益はあとで付ける
        self.assertTrue(((rl["r5_days"] >= 0) & (rl["r5_days"] <= 4)).all())

    def test_pool_span_with_holiday(self):
        # 10010 の 01-10 の候補は、01-10・01-11・01-15・01-16・01-17 の5判断日（01-12 は祝日）
        got = sorted(self.rl.loc[(self.rl["Code"] == "10010") & (self.rl["cand_date"] == "2024-01-10"), "Date"])
        self.assertEqual(got, list(pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-15", "2024-01-16", "2024-01-17"])))
        self.assertEqual(self.row("10010", "2024-01-17")["r5_days"], 4)
        self.assertEqual(len(self.rl[(self.rl["Code"] == "10010") & (self.rl["Date"] == "2024-01-18")]), 0)

    def test_repeat_keeps_last(self):
        r = self.row("10010", "2024-01-09")                         # 01-08 の候補だけ
        self.assertEqual(r["cand_date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(r["f1"], 0.0)
        r = self.row("10010", "2024-01-11")                         # 01-08 と 01-10 の両方が入る日は 01-10 の行
        self.assertEqual(r["cand_date"], pd.Timestamp("2024-01-10"))
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["r5_days"], 1)
        c0 = self.px("10010", "2024-01-10", "AdjC")
        self.assertAlmostEqual(r["r5_ret"], self.px("10010", "2024-01-11", "AdjC") / c0 - 1, places=12)
        self.assertAlmostEqual(r["r5_high"], self.px("10010", "2024-01-11", "AdjH") / c0 - 1, places=12)
        self.assertAlmostEqual(r["r5_low"], self.px("10010", "2024-01-11", "AdjL") / c0 - 1, places=12)
        d = self.full[(self.full["Code"] == "10010") & (self.full["Date"] == pd.Timestamp("2024-01-10"))].iloc[0]
        self.assertEqual(r[R.D_RET], d[E.RET])                      # 日次の収益は d_ret に残る
        # 01-08 の候補は 01-08・01-09 の2日だけ（01-10 からは 01-10 の行に負ける）→ 外した行は3件
        self.assertEqual(self.info["dropped_repeats"], 3)

    def test_same_day_and_span(self):
        r = self.row("20020", "2024-01-11")
        self.assertEqual(r["r5_days"], 0)
        self.assertEqual(r["r5_ret"], 0.0)
        self.assertTrue(np.isnan(r["r5_high"]) and np.isnan(r["r5_low"]))
        r = self.row("30030", "2024-01-19")
        self.assertEqual(r["r5_days"], 4)
        c0 = self.px("30030", "2024-01-15", "AdjC")
        b = self.bars[(self.bars["Code"] == "30030") & (self.bars["Date"] > "2024-01-15") & (self.bars["Date"] <= "2024-01-19")]
        self.assertEqual(len(b), 4)
        self.assertAlmostEqual(r["r5_high"], b["AdjH"].max() / c0 - 1, places=12)
        self.assertAlmostEqual(r["r5_low"], b["AdjL"].min() / c0 - 1, places=12)

    def test_no_bar_and_last_day(self):
        r = self.row("40040", "2024-01-19")                         # 判断日の終値が無い
        self.assertEqual(r["r5_days"], 3)
        self.assertTrue(np.isnan(r["r5_ret"]))
        self.assertTrue(np.isfinite(r["r5_high"]))
        last = self.rl[self.rl["cand_date"] == pd.Timestamp("2024-04-30")]
        self.assertEqual(list(last["Date"]), [pd.Timestamp("2024-04-30")])   # 暦の最後の日の候補は1判断日だけ

    def test_rejects(self):
        bad = self.full.copy()
        bad.loc[0, "Date"] = pd.Timestamp("2024-01-13")            # 土曜
        with self.assertRaises(SystemExit):
            R.rolling_frame(bad, self.bars, self.days, COLS)
        with self.assertRaises(SystemExit):
            R.rolling_frame(self.full.assign(r5_ret=0.0), self.bars, self.days, COLS)

    def test_info(self):
        self.assertEqual(self.info["rows"], len(self.rl))
        self.assertAlmostEqual(sum(self.info["age_share"]), 1.0)
        self.assertEqual(self.info["first"], "2024-01-08")


class TestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days, cls.bars, cls.full = small()
        rl, _ = R.rolling_frame(cls.full, cls.bars, cls.days, COLS)
        cls.rl = rl
        cls.rx, cls.xinfo = W.attach_week_exit(rl, cls.bars, cls.days)

    def test_pools(self):
        cp = R.check_pools(self.rl, self.full, self.days, n=100)
        self.assertEqual(cp["same"], cp["days"])
        self.assertEqual(cp["days"], self.rl["Date"].nunique())
        bad = self.rl[~((self.rl["Code"] == "20020") & (self.rl["Date"] == "2024-01-15"))]
        self.assertEqual(R.check_pools(bad, self.full, self.days, n=100)["same"], cp["days"] - 1)

    def test_move_cols(self):
        cm = R.check_move_cols(self.rl, self.bars, self.days, n=500)
        self.assertEqual(cm["same"], cm["rows"])
        self.assertEqual(cm["rows"], len(self.rl))
        bad = self.rl.copy()
        i = bad.index[(bad["Code"] == "30030") & (bad["r5_days"] == 2)][0]
        bad.loc[i, "r5_low"] += 1e-6
        self.assertEqual(R.check_move_cols(bad, self.bars, self.days, n=500)["same"], cm["rows"] - 1)

    def test_exit_and_same_day(self):
        ref = T115.with_daily_exit(self.rl[["Code", "Date"]], self.bars)   # 判断日を選定日にした日次の出口と同じ
        for c in (E.RET, E.R20, E.HITDAY):
            a = self.rx[c].to_numpy(dtype=float)
            b = ref[c].to_numpy(dtype=float)
            self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)), c)
            np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-15, err_msg=c)
        sd = R.check_same_day(self.rx)
        self.assertEqual(sd["rows"], int((self.rl["r5_days"] == 0).sum()))
        self.assertEqual(sd["mismatch"], 0)
        bad = self.rx.copy()
        i = bad.index[(bad["r5_days"] == 0) & bad[E.RET].notna()][0]
        bad.loc[i, E.RET] += 0.01
        self.assertEqual(R.check_same_day(bad)["ret"], 1)


class TestSearchFolds(unittest.TestCase):
    def test_months_and_purge(self):
        days = pd.bdate_range("2016-01-04", "2019-12-31")
        sub = pd.DataFrame({"Date": np.repeat(days, 2), "Code": np.tile(["a", "b"], len(days))})
        folds = R.search_folds(sub, days, n_splits=5, seed=0)
        self.assertEqual(len(folds), 5)
        pos = pd.Series(np.arange(len(days)), index=days)
        seen = []
        for tr, va in folds:
            vm = set(pd.to_datetime(va["Date"]).dt.to_period("M"))
            tm = set(pd.to_datetime(tr["Date"]).dt.to_period("M"))
            seen.extend(vm)
            # 同じ月の判断日は検証側にそろう（その月の日はすべて検証側にある）
            for m in vm:
                self.assertEqual(int((pd.to_datetime(va["Date"]).dt.to_period("M") == m).sum()),
                                 int((pd.Series(days).dt.to_period("M") == m).sum()) * 2)
            vp = pos[pd.to_datetime(va["Date"]).unique()].to_numpy()
            tp = pos[pd.to_datetime(tr["Date"]).unique()].to_numpy()
            gap = np.min(np.abs(tp[:, None] - vp[None, :]))
            self.assertGreaterEqual(gap, 5)                         # 前後4営業日は訓練に入らない
            self.assertFalse(vm & set(m for m in tm if (pd.to_datetime(tr["Date"]).dt.to_period("M") == m).sum()
                                      == (pd.Series(days).dt.to_period("M") == m).sum() * 2))
        self.assertEqual(sorted(seen), sorted(set(pd.Series(days).dt.to_period("M"))))   # どの月も1回だけ検証側
        # 年で層別: どの検証側にも各年の月が入る
        for _, va in folds:
            self.assertEqual(set(pd.to_datetime(va["Date"]).dt.year), {2016, 2017, 2018, 2019})

    def test_year_groups(self):
        years = np.array([2018] * 3 + [2019] * 12 + [2020] * 2)
        g = R._year_groups(years, 5)
        self.assertEqual(len(set(g[:3])), 1)
        self.assertEqual(g[0], g[3])                                 # 3か月の年は次の年とまとめる
        self.assertEqual(g[-1], g[-3])                               # 最後の短い年は前の組へ
        self.assertTrue(len(R.search_folds(pd.DataFrame({"Date": pd.bdate_range("2020-01-01", "2020-03-31")}),
                                           pd.bdate_range("2020-01-01", "2020-03-31"))) == 0)


def hand_pools():
    """手で数える毎日の組（判断日4日・銘柄5つ）。"""
    days = pd.bdate_range("2021-01-04", "2021-03-31")
    rows = [  # 判断日, 銘柄, 候補の日, スコア, RET, HIT, HITDAY
        ("2021-01-04", "1001", "2021-01-04", 0.9, 0.10, True, 3.0),
        ("2021-01-04", "1002", "2021-01-04", 0.8, -0.05, False, np.nan),
        ("2021-01-04", "1003", "2021-01-04", 0.1, 0.01, False, np.nan),
        ("2021-01-05", "1001", "2021-01-04", 0.9, 0.10, True, 2.0),   # 同じ候補 → 買わない
        ("2021-01-05", "1002", "2021-01-04", 0.7, -0.04, False, np.nan),  # 同じ候補 → 買わない
        ("2021-01-05", "1003", "2021-01-04", 0.6, 0.02, False, np.nan),   # 3位 → 降りない
        ("2021-01-06", "1001", "2021-01-06", 0.95, 0.05, False, np.nan),  # 新しい候補だが持っている（01-05 買い・3日目に届いて 01-08 まで）
        ("2021-01-06", "1004", "2021-01-06", 0.5, 0.03, False, np.nan),
        ("2021-01-06", "1003", "2021-01-04", 0.4, 0.02, False, np.nan),
        ("2021-01-11", "1001", "2021-01-06", 0.99, 0.08, False, np.nan),  # 01-08 に手放したので、同じ候補でなければ買える
        ("2021-01-11", "1005", "2021-01-11", 0.98, 0.01, False, np.nan),
    ]
    o = pd.DataFrame(rows, columns=["Date", "Code", "cand_date", "score_W5", E.RET, E.HIT, E.HITDAY])
    o["Date"] = pd.to_datetime(o["Date"])
    o["cand_date"] = pd.to_datetime(o["cand_date"])
    o[E.R20] = o[E.RET]
    o["fold"] = 1
    o["r5_days"] = [(pd.bdate_range(c, d).size - 1) for c, d in zip(o["cand_date"], o["Date"])]
    return o, days


class TestDailyBuys(unittest.TestCase):
    def test_by_hand(self):
        o, days = hand_pools()
        b = R.daily_buys(o, "score_W5", days, top_n=2)
        got = list(zip(b["Date"].dt.strftime("%m-%d"), b["Code"]))
        # 01-04: 1001・1002 を買う。01-05: 上位2件はどちらも買った候補（3位の 1003 には降りない）。
        # 01-06: 1001 は持っている（01-05 に買い、3営業日目に届くので 01-08 まで）→ 1004 だけ。
        # 01-11: 1001 の 01-06 の候補は前に買っていないので買える・1005 も買う
        self.assertEqual(got, [("01-04", "1001"), ("01-04", "1002"), ("01-06", "1004"), ("01-11", "1001"), ("01-11", "1005")])
        self.assertEqual(list(b["rank"]), [1, 2, 2, 1, 2])

    def test_hold_until(self):
        o, days = hand_pools()
        # 1001 の最初の買いが届かなければ（20営業日持つ）、01-11 の 1001 も持っている間なので買わない
        o2 = o.copy()
        o2.loc[0, E.HIT] = False
        o2.loc[0, E.HITDAY] = np.nan
        b = R.daily_buys(o2, "score_W5", days, top_n=2)
        self.assertNotIn(("2021-01-11", "1001"), list(zip(b["Date"].dt.strftime("%Y-%m-%d"), b["Code"])))

    def test_with_rule(self):
        o, days = hand_pools()
        # R15 の印は候補（銘柄・候補の日）ごと: 1001 の 01-04・1004 は満たさない、1003 の 01-04 は満たすが上位2件に入らない
        flags = {("1001", "2021-01-04"): 0.0, ("1002", "2021-01-04"): 1.0, ("1003", "2021-01-04"): 1.0,
                 ("1001", "2021-01-06"): 1.0, ("1004", "2021-01-06"): 0.0, ("1005", "2021-01-11"): 1.0}
        o["r15"] = [flags[(c, d.strftime("%Y-%m-%d"))] for c, d in zip(o["Code"], o["cand_date"])]
        b = R.daily_buys(o, "score_W5", days, top_n=2, need="r15")
        got = list(zip(b["Date"].dt.strftime("%m-%d"), b["Code"]))
        # 01-04: 1位の 1001 は満たさない → 2位の 1002 だけ。01-05: 1001 は満たさず 1002 は買った候補（3位の 1003 には降りない）。
        # 01-06: 1001 の 01-06 の候補は満たし、1001 は持っていない（01-04 の候補は買っていない）→ 買う。1004 は満たさない。
        # 01-11: 1001 の 01-06 の候補は買った候補 → 1005 だけ
        self.assertEqual(got, [("01-04", "1002"), ("01-06", "1001"), ("01-11", "1005")])
        self.assertEqual(list(b["rank"]), [2, 1, 2])
        self.assertTrue((b["r15"] == 1.0).all())
        o.loc[o["Code"] == "1005", "r15"] = np.nan                    # 印の無い候補は満たさない扱い（数える日は flagged で絞る）
        self.assertNotIn("1005", list(R.daily_buys(o, "score_W5", days, top_n=2, need="r15")["Code"]))

    def test_slots(self):
        o, days = hand_pools()
        b = R.daily_buys(o, "score_W5", days, top_n=2)
        s = R.slot_trades(b, days, n_months=1)
        self.assertEqual(s["offered"], 5)
        # 3枠: 01-04 の2件（1001 は01-07に空く… 1002 は20営業日）・01-06 の 1004 で3枠、01-11 は 1001 の枠（01-05 買い +3 = 01-08 に空く）に1件
        self.assertEqual(s["n"], 4)
        sim_ret = (0.10 - 0.05 + 0.03 + 0.08) / 4
        self.assertAlmostEqual(s["ret"], sim_ret)
        self.assertAlmostEqual(s["capital_pm"], (0.10 - 0.05 + 0.03 + 0.08) / 3)
        self.assertEqual(R.slot_trades(b.iloc[:0], days, 1)["n"], 0)


def scored(seed=0):
    """3つの窓 × 各30判断日 × 1日8件の、4本のスコアと収益（W5 のスコアだけ収益と強く結びつく）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", "2021-12-31")
    rows = []
    for f in (1, 2, 3):
        for k in range(30):
            d = days[(f - 1) * 70 + k]
            for j in range(8):
                rows.append({"Code": f"{1000 + j}", "Date": d, "cand_date": days[max(0, (f - 1) * 70 + k - (j % 5))],
                             "r5_days": int(j % 5), "fold": f})
    o = pd.DataFrame(rows)
    z = rng.normal(size=len(o))
    o[E.R20] = 0.03 * z + rng.normal(0, 0.03, len(o))
    o[E.RET] = np.minimum(o[E.R20], 0.10)
    o[E.HIT] = o[E.RET] >= 0.10
    o[E.HITDAY] = np.where(o[E.HIT], 5.0, np.nan)
    o[R.D_RET] = o[E.RET]
    o[R.D_R20] = o[E.R20]
    o["score_W5"] = z
    for a in ("W5X", "D", "B"):
        o[f"score_{a}"] = rng.normal(size=len(o))
    o["r15"] = (rng.random(len(o)) < 0.5).astype(float)
    o["pmin"] = np.round(rng.uniform(90.0, 100.0, len(o)), 1)
    return o, days


class TestEvaluate(unittest.TestCase):
    def test_evaluate_and_age(self):
        o, days = scored()
        gap_day = o["Date"].unique()[40]
        o.loc[(o["Date"] == gap_day) & (o["Code"] == "1003"), "r15"] = np.nan   # 印の無い候補がある日は数えない
        res, buys = R.evaluate(o, days, 2)
        self.assertEqual(res["days_all"], o["Date"].nunique())
        self.assertEqual(res["days"], o["Date"].nunique() - 1)
        self.assertEqual(res["rows"], len(o) - 8)
        self.assertGreater(res["buys"]["W5"]["ret_w"], res["all"]["ret_w"])
        self.assertGreater(res["rank"]["W5"]["ndcg"], res["rank"]["B"]["ndcg"])
        self.assertEqual(res["diff"]["B"]["n"], 3)
        self.assertEqual(res["diff"]["R15"]["n"], 3)
        for a in R.ARMS:
            b = buys[a]
            self.assertLessEqual(len(b), 2 * res["days"])
            self.assertTrue((b["r15"] == 1.0).all())                 # 買うのは R15 を満たす候補だけ
            self.assertNotIn(pd.Timestamp(gap_day), set(b["Date"]))
            self.assertLessEqual(b.groupby("Date").size().max(), 2)
            self.assertEqual(res["buys"][a]["n"], len(b))
            self.assertLessEqual(res["buys"][a]["n"], res["rule_cands"])
            self.assertLessEqual(res["slots"][a]["n"], len(b))
            self.assertGreaterEqual(res["free"][a]["n"], 1)
        # 今の買い: 候補の日（r5_days = 0）の R15 の行だけ
        z = o[(o["r5_days"] == 0) & (o["r15"] == 1.0) & (o["Date"] != gap_day)]
        self.assertGreater(res["rule"]["n"], 0)
        self.assertLessEqual(res["rule"]["n"], len(z))
        self.assertLessEqual(res["slots"]["R15"]["n"], res["rule"]["n"])
        ag = res["age"]
        self.assertAlmostEqual(sum(ag["share"]), 1.0)
        self.assertAlmostEqual(sum(ag["buy_share"]["W5"]), 1.0)
        for a in R.ARMS:                                             # 組の中の百分位の平均は全体で 0.5
            sh = np.array(ag["share"])
            self.assertAlmostEqual(float(np.nansum(np.array(ag["pct"][a]) * sh)), 0.5, places=6)

    def test_reading(self):
        def row(sh, w, r, b, d, gap):
            st = lambda x: {"ret_w": x[0], "bad_w": x[1]}  # noqa: E731
            return {"shift": sh, "buys": {"W5": st(w), "B": st(b), "D": st(d)}, "rule": st(r),
                    "age": {"fresh_gap": {"W5": gap}}}

        good = [row(s, (0.02, 0.05), (0.018, 0.06), (0.01, 0.08), (0.015, 0.06), 0.051) for s in (0, 2, 4)]
        r = R.reading(good)
        self.assertTrue(r["better"])
        self.assertEqual(r["lean"], "fresh")
        edge = [row(s, (0.02, 0.05), (0.018, 0.06), (0.01, 0.08), (0.015, 0.06), 0.0449) for s in (0, 2, 4)]
        self.assertEqual(R.reading(edge)["lean"], "none")            # 小数2桁に丸めると 0.04
        aged = [row(s, (0.02, 0.07), (0.018, 0.08), (0.01, 0.08), (0.015, 0.06), -0.06) for s in (0, 2, 4)]
        r = R.reading(aged)
        self.assertFalse(r["better"])                                 # D より大負けが多い
        self.assertEqual(r["lean"], "aged")
        # B・D には勝つが、今の買い R15 より平均収益が低い（1つのずらしでも）→ 良いとは読まない
        mixed = good[:2] + [row(4, (0.02, 0.05), (0.021, 0.06), (0.01, 0.08), (0.015, 0.06), 0.051)]
        r = R.reading(mixed)
        self.assertFalse(r["better"])
        self.assertEqual([x["better"] for x in r["rows"]], [True, True, False])
        # 平均収益は上でも、R15 のほうが −10% 未満が少ない → 良いとは読まない
        self.assertFalse(R.reading([row(0, (0.02, 0.05), (0.018, 0.04), (0.01, 0.08), (0.015, 0.06), 0.0)])["better"])
        nan = [row(0, (np.nan, 0.05), (0.018, 0.06), (0.01, 0.08), (0.015, 0.06), np.nan)]
        self.assertFalse(R.reading(nan)["better"])
        self.assertEqual(R.reading(nan)["lean"], "none")


def long_frames(seed=0, n_codes=8):
    """2016〜2021年の組の行（1日 n_codes 件）と日次の候補。out-of-fold の窓が作れる長さ。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2021-12-31")
    rows = []
    for i, d in enumerate(days):
        for j in range(n_codes):
            back = int(rng.integers(0, 5))
            rows.append({"Code": f"{3000 + j}", "Date": d, "cand_date": days[max(0, i - back)], "r5_days": back})
    rl = pd.DataFrame(rows)
    n = len(rl)
    z = rng.normal(size=n)
    for c in ("f1", "f2", "f3"):
        rl[c] = z + rng.normal(0, 1.0, n)
    rl["r5_ret"] = rng.normal(0, 0.02, n)
    rl["r5_high"] = np.abs(rl["r5_ret"]) + 0.01
    rl["r5_low"] = -np.abs(rl["r5_ret"]) - 0.01
    rl[E.R20] = 0.03 * z + rng.normal(0, 0.03, n)
    rl[E.RET] = np.minimum(rl[E.R20], 0.10)
    rl[E.HIT] = rl[E.RET] >= 0.10
    rl[E.HITDAY] = np.where(rl[E.HIT], 4.0, np.nan)
    rl[R.D_RET] = rl[E.RET]
    rl[R.D_R20] = rl[E.R20]
    rl["label"] = (z > 0.8).astype(float)
    wk, daily = T115.long_frames(seed)
    return rl, daily, days


class TestOof(unittest.TestCase):
    def test_windows_columns_and_cache(self):
        rl, daily, _ = long_frames()
        cols = ["f1", "f2", "f3"]
        cols_w = cols + list(R.MOVE_COLS)
        params = {"W5": dict(T115.SMALL_RANK, num_leaves=5), "D": dict(T115.SMALL_RANK, num_leaves=9),
                  "B": T115.SMALL_CLF}
        calls = []

        def fake_rank(p, tr, te, c):
            calls.append(("rank", p["num_leaves"], list(c), pd.Timestamp(tr["Date"].max()),
                          pd.Timestamp(te["Date"].min()), pd.Timestamp(te["Date"].max()),
                          pd.Timestamp(te["cand_date"].min()), "r5_days" in tr.columns))
            return np.zeros(len(te))

        def fake_base(p, tr, tes, c):
            calls.append(("base", None, list(c), pd.Timestamp(tr["Date"].max()), None, None,
                          pd.Timestamp(tes[0]["cand_date"].min()), "r5_days" in tr.columns))
            return [np.zeros(len(t)) for t in tes]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(R.W, "fit_rank", side_effect=fake_rank), \
                mock.patch.object(R.W, "fit_base", side_effect=fake_base):
            o = R.oof_shift(rl, daily, cols, cols_w, params, 0, out_dir=tmp)
            folds = {f.index: f for f in E.E41.folds_for(rl["Date"], 0)}
            self.assertGreater(o["fold"].nunique(), 1)
            for c in ("score_W5", "score_W5X", "score_D", "score_B", "fold", "cand_date", "r5_days", E.HITDAY):
                self.assertIn(c, o.columns)
            n_folds = o["fold"].nunique()
            self.assertEqual(len(calls), 4 * n_folds)
            for k, f in enumerate(sorted(o["fold"].unique())):
                fd = folds[f]
                end, lo, hi = pd.Timestamp(fd.train_end), pd.Timestamp(fd.test_start), pd.Timestamp(fd.test_end)
                wcall, wxcall, dcall, bcall = calls[4 * k: 4 * k + 4]
                self.assertEqual(wcall[2], cols_w)                     # W5 だけ値動きの列を使う
                self.assertEqual(wxcall[2], cols)
                self.assertEqual(dcall[2], cols)
                self.assertEqual(bcall[2], cols)
                self.assertTrue(wcall[7] and wxcall[7])                # W5・W5X は組の行で学ぶ
                self.assertFalse(dcall[7] or bcall[7])                 # D・B は日次の行で学ぶ
                self.assertEqual(dcall[1], 9)
                for c in (wcall, wxcall, dcall, bcall):
                    self.assertLessEqual(c[3], end)                    # 訓練は訓練の最終日まで
                    self.assertGreater(c[6], end)                      # 採点する候補の日は訓練の最終日より後
                self.assertGreaterEqual(wcall[4], lo)
                self.assertLessEqual(wcall[5], hi)
                got = o[o["fold"] == f]
                self.assertTrue(((got["Date"] >= lo) & (got["Date"] <= hi)).all())
            n_calls = len(calls)
            o2 = R.oof_shift(rl, daily, cols, cols_w, params, 0, out_dir=tmp)   # 保存したものを読む
            self.assertEqual(len(calls), n_calls)
            pd.testing.assert_frame_equal(o.reset_index(drop=True), o2)
            params2 = dict(params, W5=dict(T115.SMALL_RANK, num_leaves=6))
            R.oof_shift(rl, daily, cols, cols_w, params2, 0, out_dir=tmp)       # 木の形が変われば作り直す
            self.assertGreater(len(calls), n_calls)


def expected_rule(oof: pd.DataFrame, folds, models) -> tuple:
    """R15 の印と3本の最小を numpy だけで直接計算し直す（test_e115_weekly_ltr.expected_r0 と同じ式で、3本を引数に）。"""
    d = oof["Date"].to_numpy()
    win = np.zeros(len(oof), dtype=int)
    for f in folds:
        win[(d >= np.datetime64(pd.Timestamp(f.test_start))) & (d <= np.datetime64(pd.Timestamp(f.test_end)))] = f.index
    start = {f.index: pd.Timestamp(f.test_start) for f in folds}
    present = sorted(set(int(x) for x in win[win > 0]))
    flag = np.full(len(oof), np.nan)
    pmin = np.full(len(oof), np.nan)
    for ww in present:
        prior = [x for x in present if x < ww]
        if len(prior) < 2:
            continue
        ref = oof[np.isin(win, prior) & (d <= np.datetime64(start[ww] - T115.RULE_GAP))]
        te = np.flatnonzero(win == ww)
        pcts = []
        for a in models:
            r = ref[f"score_{a}"].to_numpy(dtype=float)
            x = oof[f"score_{a}"].to_numpy(dtype=float)[te]
            pcts.append(np.array([round(float((r < v).sum()) / len(r) * 100.0, 1) for v in x]))
        pc = np.vstack(pcts)
        flag[te] = (pc >= 95.0).all(axis=0)
        pmin[te] = pc.min(axis=0)
    return flag, pmin


class TestDailyRule(unittest.TestCase):
    """今の毎日の買い（R15。1.5σ の木3本）の印と3本の最小。合成の本番の out-of-fold を置き場に書いて読む。"""

    def test_same_as_direct_and_missing(self):
        rng = np.random.default_rng(5)
        days = pd.bdate_range("2016-01-04", "2021-12-31")
        codes = [f"{6000 + j}" for j in range(6)]
        daily = pd.DataFrame({"Code": np.tile(codes, len(days)), "Date": np.repeat(days, len(codes))})
        z = rng.normal(size=len(daily))
        sc = {a: z + rng.normal(0, 0.3, len(daily)) for a in R.RULE_TREES}
        folds = E.E41.folds_for(daily["Date"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            for a in R.RULE_TREES:
                daily.assign(score=sc[a], label=(z > 1.0).astype(float), outcome=0.0).to_parquet(
                    os.path.join(tmp, f"{a}_oof.parquet"), index=False)
            with mock.patch.object(R, "RULE_TREES", R.RULE_TREES):
                out, info = R.daily_rule(daily, data_dir=tmp)
            empty, info2 = R.daily_rule(daily, data_dir=os.path.join(tmp, "none"))
        oof = daily.copy()
        for a in R.RULE_TREES:
            oof[f"score_{a}"] = sc[a]
        flag, pmin = expected_rule(oof, folds, R.RULE_TREES)
        np.testing.assert_array_equal(np.isnan(out["r15"].to_numpy()), np.isnan(flag))
        ok = ~np.isnan(flag)
        np.testing.assert_array_equal(out["r15"].to_numpy()[ok], flag[ok])
        np.testing.assert_allclose(out["pmin"].to_numpy()[ok], pmin[ok], rtol=0, atol=1e-9)
        self.assertGreater(int(np.nansum(flag)), 10)
        self.assertEqual(info["r15"], int(np.nansum(flag)))
        self.assertFalse(info2["found"])
        self.assertEqual(len(empty), 0)
        self.assertEqual(tuple(R.RULE_TREES), tuple(LT.K15))


def rule_pools():
    """今の買い（R15）を手で数える組の行。"""
    days = pd.bdate_range("2021-01-04", "2021-03-31")
    rows = [  # 判断日, 銘柄, 候補の日, r5_days, r15, pmin, score_B, RET, HIT, HITDAY
        ("2021-01-04", "1001", "2021-01-04", 0, 1.0, 97.0, 0.2, 0.10, True, 2.0),
        ("2021-01-04", "1002", "2021-01-04", 0, 1.0, 97.0, 0.5, -0.05, False, np.nan),  # 1001 と同点 → 基準モデルの順で上
        ("2021-01-04", "1003", "2021-01-04", 0, 1.0, 99.0, 0.1, 0.02, False, np.nan),
        ("2021-01-04", "1004", "2021-01-04", 0, 0.0, 80.0, 0.9, 0.03, False, np.nan),   # 満たさない
        ("2021-01-05", "1001", "2021-01-04", 1, 1.0, 97.0, 0.2, 0.09, True, 1.0),      # 候補の日でない → 今の買いでは見ない
        ("2021-01-07", "1003", "2021-01-07", 0, 1.0, 99.5, 0.3, 0.04, False, np.nan),   # 01-05 に買って20営業日持つ → 買わない
        ("2021-01-07", "1001", "2021-01-07", 0, 1.0, 96.0, 0.3, 0.06, False, np.nan),   # 01-05 に買い2営業日目に届いた → 買える
    ]
    o = pd.DataFrame(rows, columns=["Date", "Code", "cand_date", "r5_days", "r15", "pmin", "score_B",
                                    E.RET, E.HIT, E.HITDAY])
    o["Date"] = pd.to_datetime(o["Date"])
    o["cand_date"] = pd.to_datetime(o["cand_date"])
    o[E.R20] = o[E.RET]
    o["fold"] = 1
    return o, days


class TestRuleBuys(unittest.TestCase):
    def test_by_hand(self):
        o, days = rule_pools()
        b = R.rule_buys(o, days)
        got = list(zip(b["Date"].dt.strftime("%m-%d"), b["Code"], b["rank"]))
        # 01-04: 3本の最小の順（1003 99.0）→ 同点の 1001・1002 は基準モデルのスコアの順（1002 が先）。1004 は満たさない。
        # 01-07: 1003 は持っている間なので買わない（順位は2位のまま）、1001 は 01-07 に枠が空くので買う
        self.assertEqual(got, [("01-04", "1003", 1), ("01-04", "1002", 2), ("01-04", "1001", 3), ("01-07", "1001", 2)])
        self.assertTrue((b["r5_days"] == 0).all())
        s = R.slot_trades(b[b["rank"] <= LT.TOP_K], days, n_months=1)   # 3枠はその日の上位2件
        self.assertEqual(s["offered"], 3)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["ret"], (0.02 - 0.05 + 0.06) / 3)


def triple_pools():
    """3つとも満たす候補（日次の1位・組の1位・R15）を手で数える組の行。"""
    days = pd.bdate_range("2020-12-28", "2021-03-31")
    rows = [  # 判断日, 銘柄, 候補の日, r15, D, W5, RET
        ("2021-01-04", "1001", "2021-01-04", 1.0, 0.90, 0.80, 0.05),    # 日次1位・組2位
        ("2021-01-04", "1002", "2021-01-04", 1.0, 0.50, 0.95, -0.02),   # 日次2位・組1位
        ("2021-01-04", "1003", "2020-12-29", 1.0, 0.99, 0.70, 0.01),    # 候補の日でない（D が高くても日次の順位は付かない）
        ("2021-01-05", "1004", "2021-01-05", 1.0, 0.80, 0.90, 0.09),    # 3つとも満たす
        ("2021-01-05", "1005", "2021-01-05", 1.0, 0.70, 0.85, 0.03),
        ("2021-01-05", "1002", "2021-01-04", 1.0, 0.50, 0.88, -0.02),   # 1日前の候補
        ("2021-01-06", "1006", "2021-01-06", 0.0, 0.90, 0.99, 0.08),    # 日次1位・組1位だが R15 を満たさない
        ("2021-01-06", "1007", "2021-01-06", 1.0, 0.10, 0.20, -0.12),
        ("2021-01-07", "1004", "2021-01-07", 1.0, 0.90, 0.95, 0.04),    # 3つとも満たすが、01-05 に買って持っている
        ("2021-01-07", "1008", "2021-01-07", 1.0, 0.90, 0.30, 0.02),    # 日次で 1004 と同点 → 銘柄コードの順で2位
    ]
    o = pd.DataFrame(rows, columns=["Date", "Code", "cand_date", "r15", "score_D", "score_W5", E.RET])
    o["Date"] = pd.to_datetime(o["Date"])
    o["cand_date"] = pd.to_datetime(o["cand_date"])
    o["r5_days"] = [(pd.bdate_range(c, d).size - 1) for c, d in zip(o["cand_date"], o["Date"])]
    o[E.HIT] = False
    o[E.HITDAY] = np.nan
    o[E.R20] = o[E.RET]
    o["fold"] = 1
    return o, days


class TestTriple(unittest.TestCase):
    def test_by_hand(self):
        o, days = triple_pools()
        got = {k: list(zip(b["Date"].dt.strftime("%m-%d"), b["Code"])) for k, b in R.triple_buys(o, days).items()}
        self.assertEqual(got["D1W1"], [("01-05", "1004")])
        self.assertEqual(got["D1"], [("01-04", "1001"), ("01-05", "1004")])
        self.assertEqual(got["W1"], [("01-04", "1002"), ("01-05", "1004")])
        self.assertEqual(got["R15"], [("01-04", "1001"), ("01-04", "1002"), ("01-05", "1004"), ("01-05", "1005"),
                                      ("01-06", "1007"), ("01-07", "1008")])
        self.assertEqual(got["REST"], [("01-04", "1001"), ("01-04", "1002"), ("01-05", "1005"), ("01-06", "1007"),
                                       ("01-07", "1008")])

    def test_stats(self):
        o, days = triple_pools()
        t = R.triple_stats(o, days)
        self.assertEqual(t["days"], 4)
        self.assertEqual(t["months"], 1)
        s = t["sets"]
        self.assertEqual([s[k]["n"] for k in R.TRIPLE_SETS], [1, 2, 2, 6, 5])
        self.assertAlmostEqual(s["D1W1"]["ret"], 0.09)
        self.assertTrue(np.isnan(s["D1W1"]["se"]))                  # 1件では標準誤差を出さない
        r = np.array([0.05, -0.02, 0.09, 0.03, -0.12, 0.02])
        self.assertAlmostEqual(s["R15"]["ret"], r.mean())
        self.assertAlmostEqual(s["R15"]["se"], r.std(ddof=1) / np.sqrt(6))
        self.assertAlmostEqual(s["R15"]["win"], 4 / 6)
        self.assertAlmostEqual(s["R15"]["bad"], 1 / 6)
        self.assertAlmostEqual(s["R15"]["pm"], 6.0)
        o2 = o.copy()
        o2.loc[o2["Code"] == "1008", "r15"] = np.nan                  # 01-07 の組に印の無い行がある → その日は数えない
        self.assertEqual(R.triple_stats(o2, days)["days"], 3)


class TestAttachAndFlagged(unittest.TestCase):
    def test_attach_rule(self):
        o, _ = hand_pools()
        rule = pd.DataFrame({"Code": ["1001", "1002", "1003", "1001", "1004"],
                             "Date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-04", "2021-01-06", "2021-01-06"]),
                             "r15": [1.0, 0.0, 1.0, 0.0, 1.0], "pmin": [96.0, 50.0, 95.0, 94.9, 99.0]})
        got = R.attach_rule(o.assign(r15=9.0), rule)                  # 前の印は捨てて付け直す
        self.assertEqual(len(got), len(o))
        self.assertEqual(list(got["Code"]), list(o["Code"]))
        a = got[(got["Code"] == "1001") & (got["cand_date"] == "2021-01-04")]
        self.assertEqual(list(a["r15"]), [1.0, 1.0])                  # 同じ候補はどの判断日でも同じ印
        self.assertEqual(float(got.loc[(got["Code"] == "1001") & (got["cand_date"] == "2021-01-06"), "r15"].iloc[0]), 0.0)
        self.assertTrue(got.loc[got["Code"] == "1005", "r15"].isna().all())   # 印の無い候補は欠測
        with self.assertRaises(pd.errors.MergeError):                 # 同じ候補に印が2つあれば止まる
            R.attach_rule(o, pd.concat([rule, rule.iloc[:1]]))

    def test_flagged(self):
        o, _ = hand_pools()
        o["r15"] = 1.0
        o.loc[(o["Code"] == "1004"), "r15"] = np.nan                 # 01-06 の組に印の無い行がある
        f = R.flagged(o)
        self.assertEqual(sorted(f["Date"].dt.strftime("%m-%d").unique()), ["01-04", "01-05", "01-11"])
        self.assertEqual(len(f), len(o) - 3)


def main_data():
    """通しで回す合成データ（2016〜2021年・8銘柄・候補は日足の約4分の1）。"""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2016-01-04", "2021-12-31")
    codes = [f"{5000 + j}" for j in range(8)]
    bars = T115.make_bars(codes, days, seed=4)
    rows = [{"Code": c, "Date": d} for d in days for c in codes if rng.random() < 0.25]
    cands = pd.DataFrame(rows).sort_values(["Date", "Code"]).reset_index(drop=True)
    n = len(cands)
    z = rng.normal(size=n)
    for c in ("f1", "f2", "f3"):
        cands[c] = z + rng.normal(0, 1.0, n)
    cands["label"] = (z > 0.8).astype(float)
    cands["cap_band"] = rng.integers(0, 5, n).astype(float)
    full = T115.with_daily_exit(cands, bars)
    return full, bars, codes


class TestMain(unittest.TestCase):
    """通しで回す（データの読み込み・探索・木の形の読み込み・R15 の印は差し替え。当てはめは小さな木で本物）。"""

    def patches(self, tmp, full, bars, cutoff, fake_rule, fake_tune):
        chk = {"hold5_match": 0, "hold5_both": 0}
        return [mock.patch.object(R, "OOF_DIR", tmp),
                mock.patch.object(R, "PARAMS_PATH", os.path.join(tmp, "e117_params.json")),
                mock.patch.object(R, "SUMMARY_PATH", os.path.join(tmp, "e117_summary.json")),
                mock.patch.object(R.F, "columns", return_value=["f1", "f2", "f3"]),
                mock.patch.object(R.W, "load_all", return_value=(full, bars, cutoff, chk)),
                mock.patch.object(R.W, "base_params", return_value=dict(T115.SMALL_CLF)),
                mock.patch.object(R.W, "daily_ltr_params", return_value={"params": dict(T115.SMALL_RANK), "_cutoff": "x"}),
                mock.patch.object(R, "daily_rule", side_effect=fake_rule),
                mock.patch.object(R.E, "tune_l", side_effect=fake_tune)]

    def test_main(self):
        full, bars, codes = main_data()
        cutoff = pd.Timestamp("2019-12-31")
        no_pct_until = pd.Timestamp("2019-03-01")                  # 本番の out-of-fold の最初の窓のように、百分位が付かない期間

        def fake_tune(sub, cols, n_trials, folds):
            self.assertEqual(list(cols), ["f1", "f2", "f3"] + list(R.MOVE_COLS))
            self.assertLessEqual(sub["Date"].max(), cutoff)
            self.assertTrue(sub["label"].notna().all())
            self.assertEqual(len(folds), 5)
            return {"params": dict(T115.SMALL_RANK), "_cv": {"objective": "x", "ndcg": 0.5, "best_trial": 0, "trials": []},
                    "_minutes": 0.0}

        def fake_rule(daily):
            self.assertTrue(daily["label"].notna().all())
            r = daily[["Code", "Date"]].assign(r15=(daily["f1"] > 0.5).astype(float), pmin=np.round(daily["f1"] * 10, 1))
            r.loc[r["Date"] < no_pct_until, ["r15", "pmin"]] = np.nan
            return r, {"found": True, "joined": len(r), "rows": len(r), "with_pct": int(r["r15"].notna().sum()),
                       "r15": int(r["r15"].sum()), "matched": int(r["r15"].notna().sum())}

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            mocks = [st.enter_context(p) for p in self.patches(tmp, full, bars, cutoff, fake_rule, fake_tune)]
            dr, tl = mocks[-2], mocks[-1]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(R.main(["--trials", "2", "--shifts", "0,2"]), 0)
            out = buf.getvalue()
            with open(os.path.join(tmp, "e117_summary.json"), encoding="utf-8") as fh:
                sm = json.load(fh)
            self.assertEqual(tl.call_count, 1)
            self.assertEqual(dr.call_count, 1)                                 # 印は1回だけ作り、どのずらし方でも使う
            self.assertEqual(sorted(sm["shifts"]), ["0", "2"])
            self.assertEqual(sm["check_same_day"]["mismatch"], 0)
            self.assertEqual(sm["check_cols"]["same"], sm["check_cols"]["rows"])
            self.assertEqual(sm["check_pools"]["same"], sm["check_pools"]["days"])
            self.assertTrue(sm["rule_info"]["found"])
            self.assertIn(sm["reading"]["lean"], ("fresh", "aged", "none"))
            self.assertIsInstance(sm["reading"]["better"], bool)
            s0 = sm["shifts"]["0"]
            self.assertLess(s0["days"], s0["days_all"])                        # 百分位が付かない期間の判断日は数えない
            self.assertGreater(s0["days"], 0)
            b0 = s0["buys"]["W5"]
            self.assertGreater(b0["n"], 0)
            self.assertLessEqual(b0["n"], 2 * s0["days"])
            self.assertLessEqual(b0["n"], s0["rule_cands"])
            self.assertGreater(s0["rule"]["n"], 0)
            self.assertIn("R15", s0["slots"])
            self.assertIn("R15", s0["diff"])
            self.assertIn("読み方", out)
            self.assertIn("毎日の買い", out)
            self.assertIn("今の買い R15", out)
            for code in codes:                                                 # 公開ログに銘柄コードを出さない
                self.assertNotIn(code, out)
            self.assertNotIn("triple", sm)
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):                             # 2回目は探索と out-of-fold を読む（参考の表も出す）
                self.assertEqual(R.main(["--trials", "2", "--shifts", "0,2", "--triple"]), 0)
            self.assertEqual(tl.call_count, 1)
            out2 = buf2.getvalue()
            with open(os.path.join(tmp, "e117_summary.json"), encoding="utf-8") as fh:
                sm2 = json.load(fh)
            self.assertIn("■ 参考（運用者の確認 2026-10-30", out2)
            self.assertEqual(sorted(sm2["triple"]), ["0", "2"])
            for sh in ("0", "2"):
                t = sm2["triple"][sh]["sets"]
                self.assertEqual(sm2["triple"][sh]["days"], sm2["shifts"][sh]["days"])
                self.assertEqual(t["R15"]["n"], sm2["shifts"][sh]["rule"]["n"])     # 今の買いは本体の R15 と同じ
                self.assertLessEqual(t["D1W1"]["n"], min(t["D1"]["n"], t["W1"]["n"]))
                self.assertLessEqual(max(t["D1"]["n"], t["W1"]["n"]), t["R15"]["n"])
                self.assertLessEqual(t["REST"]["n"], t["R15"]["n"])
            for code in codes:
                self.assertNotIn(code, out2)

    def test_stops_without_production_oof(self):
        full, bars, _ = main_data()

        def no_rule(daily):
            return daily[["Code", "Date"]].iloc[:0], {"found": False}

        def never(*a, **k):
            raise AssertionError("探索まで進まない")

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            for p in self.patches(tmp, full, bars, pd.Timestamp("2019-12-31"), no_rule, never):
                st.enter_context(p)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                R.main(["--trials", "2", "--shifts", "0"])


if __name__ == "__main__":
    unittest.main()
