#!/usr/bin/env python3
"""
実験115（research/exp/e115_weekly_ltr.py）。合成の日足と候補で:

- 週末: その週（月〜金）の最後の取引日（金曜が祝日なら木曜）。最後の週（終わっていないかもしれない）は外す
- 週の候補: 同じ (銘柄, 週) は最後の候補の日の行だけ。Date は週末、cand_date に候補の日。wk_days は取引日の暦で数える。
  値動き4列（候補の日の終値 → 週末の終値・その間の高値の最大・安値の最小）。週末に入った候補は wk_ret 0・高値と安値は欠測。
  週末に日足が無い銘柄は wk_ret が欠測。取引日でない日付・足す列と同じ名前の列があれば止める
- 週の収益: 実験114 の attach_exit を Date = 週末で回したものと同じ。買った日は週末の翌取引日。週末に入った候補は日次の収益と
  同じ（合わなければ数える）。値動きの列の直接の計算し直し（合わなければ数える）
- 上位 n 件・窓ごとの集計・曜日・読み方（回す前に決めたもの）
- out-of-fold: W・WX は週末が訓練の最終日以前の週の行、D・B は候補の日が訓練の最終日以前の日次の行で学び、採点はテスト窓の週の
  行（訓練に使った行より後）。W は値動きの列を使い、WX・D・B は使わない。保存した out-of-fold は次の回に読む
- 木の形: 学習する列で探索したものでなければ止める
- 通しで回す（探索は差し替え、当てはめは小さな木）: 集計と読み方を記録に書く

  python3 tests/test_e115_weekly_ltr.py
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

import features as F  # noqa: E402
import lab  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e115_weekly_ltr as W  # noqa: E402

HOLIDAYS = (pd.Timestamp("2024-01-12"), pd.Timestamp("2024-02-12"))     # 金曜・月曜の祝日
SMALL_RANK = {"boosting_type": "gbdt", "n_estimators": 5, "learning_rate": 0.1, "num_leaves": 7,
              "min_child_samples": 5, "subsample": 1.0, "subsample_freq": 1, "colsample_bytree": 1.0,
              "n_jobs": 1, "verbose": -1}
SMALL_CLF = {"objective": "binary", "boosting_type": "gbdt", "n_estimators": 5, "learning_rate": 0.1, "num_leaves": 7,
             "min_child_samples": 5, "subsample": 1.0, "subsample_freq": 1, "colsample_bytree": 1.0,
             "random_state": 0, "n_jobs": 1, "verbose": -1}


def make_bars(codes, days, seed=0, skip=()):
    """銘柄ごとのでたらめな歩みの日足（調整後の始値・高値・安値・終値）。skip の (銘柄, 日) は日足を作らない。"""
    rng = np.random.default_rng(seed)
    rows = []
    for c in codes:
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, len(days)))
        for i, d in enumerate(days):
            if (c, d) in skip:
                continue
            o = px[i] * (1.0 + rng.normal(0, 0.005))
            rows.append({"Date": d, "Code": c, "AdjO": o, "AdjH": max(o, px[i]) * (1.0 + abs(rng.normal(0, 0.01))),
                         "AdjL": min(o, px[i]) * (1.0 - abs(rng.normal(0, 0.01))), "AdjC": px[i]})
    return pd.DataFrame(rows)


def with_daily_exit(cands: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """候補に日次の今の売り方の収益（実験114 の attach_exit）を足す（load_all と同じ）。"""
    c = cands.copy()
    c[lab.OUTCOME] = np.nan
    out, _ = E.attach_exit(c, bars)
    return out


class SmallWeek(unittest.TestCase):
    """祝日を含む短い暦と、週の作りを決める候補。"""

    @classmethod
    def setUpClass(cls):
        days = pd.bdate_range("2024-01-01", "2024-04-30")
        cls.days = days.drop(list(HOLIDAYS))
        # 40040 は 2024-01-19（週末）に日足が無い（売買停止）
        cls.bars = make_bars(["10010", "20020", "30030", "40040"], cls.days,
                             skip={("40040", pd.Timestamp("2024-01-19"))})
        cands = pd.DataFrame({
            "Code": ["10010", "10010", "20020", "30030", "10010", "20020", "40040"],
            "Date": pd.to_datetime(["2024-01-08", "2024-01-10", "2024-01-11", "2024-01-15", "2024-02-13",
                                    "2024-04-29", "2024-01-16"]),
            "f1": np.arange(7, dtype=float), "label": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})
        cls.full = with_daily_exit(cands, cls.bars)
        cls.wk, cls.info = W.weekly_frame(cls.full, cls.bars, cls.days)

    def row(self, code, wend):
        r = self.wk[(self.wk["Code"] == code) & (self.wk["Date"] == pd.Timestamp(wend))]
        self.assertEqual(len(r), 1)
        return r.iloc[0]

    def px(self, code, d, col):
        b = self.bars
        return float(b.loc[(b["Code"] == code) & (b["Date"] == pd.Timestamp(d)), col].iloc[0])


class TestWeekEnds(unittest.TestCase):
    def test_holidays_and_last_week(self):
        days = pd.bdate_range("2024-01-01", "2024-02-29").drop(list(HOLIDAYS))
        m = W.week_ends(days)
        self.assertEqual(m[pd.Timestamp("2024-01-08")], pd.Timestamp("2024-01-11"))     # 金曜が祝日 → 木曜
        self.assertEqual(m[pd.Timestamp("2024-01-11")], pd.Timestamp("2024-01-11"))
        self.assertEqual(m[pd.Timestamp("2024-02-13")], pd.Timestamp("2024-02-16"))     # 月曜が祝日
        self.assertEqual(m[pd.Timestamp("2024-02-09")], pd.Timestamp("2024-02-09"))
        self.assertNotIn(pd.Timestamp("2024-02-26"), m.index)                             # 最後の週は外す
        self.assertEqual(m.index.max(), pd.Timestamp("2024-02-23"))
        self.assertTrue((m.index <= m.to_numpy()).all())


class TestWeeklyFrame(SmallWeek):
    def test_rows_and_keys(self):
        wk = self.wk
        self.assertEqual(len(wk), 5)
        self.assertEqual(self.info["open_week_rows"], 1)          # 2024-04-29 の週は最後の週
        self.assertEqual(self.info["dropped_repeats"], 1)         # 10010 の 01-08 は同じ週の 01-10 に負ける
        self.assertFalse(wk.duplicated(["Code", "Date"]).any())
        self.assertTrue(wk[["Date", "Code"]].equals(wk.sort_values(["Date", "Code"])[["Date", "Code"]]))
        for c in (W.D_RET, W.D_HIT, W.D_R20, "cand_date", "wk_hits") + W.WEEK_COLS:
            self.assertIn(c, wk.columns)
        for c in (E.RET, E.HIT, E.HITDAY, E.R20):
            self.assertNotIn(c, wk.columns)                       # 週の収益はあとで付ける

    def test_repeat_keeps_last(self):
        r = self.row("10010", "2024-01-11")
        self.assertEqual(r["cand_date"], pd.Timestamp("2024-01-10"))
        self.assertEqual(r["wk_hits"], 2)
        self.assertEqual(r["f1"], 1.0)                            # 01-10 の行の値
        self.assertEqual(r["wk_days"], 1)
        c0 = self.px("10010", "2024-01-10", "AdjC")
        self.assertAlmostEqual(r["wk_ret"], self.px("10010", "2024-01-11", "AdjC") / c0 - 1, places=12)
        self.assertAlmostEqual(r["wk_high"], self.px("10010", "2024-01-11", "AdjH") / c0 - 1, places=12)
        self.assertAlmostEqual(r["wk_low"], self.px("10010", "2024-01-11", "AdjL") / c0 - 1, places=12)
        # 日次の収益は d_ret に残る（01-10 の候補の翌営業日に買った場合）
        d = self.full[(self.full["Code"] == "10010") & (self.full["Date"] == pd.Timestamp("2024-01-10"))].iloc[0]
        self.assertEqual(r[W.D_RET], d[E.RET])

    def test_last_day_of_week(self):
        r = self.row("20020", "2024-01-11")                       # 祝日の週の最後の取引日に入った候補
        self.assertEqual(r["wk_days"], 0)
        self.assertEqual(r["wk_ret"], 0.0)
        self.assertTrue(np.isnan(r["wk_high"]) and np.isnan(r["wk_low"]))

    def test_full_week_and_holiday_week(self):
        r = self.row("30030", "2024-01-19")
        self.assertEqual(r["wk_days"], 4)
        c0 = self.px("30030", "2024-01-15", "AdjC")
        b = self.bars[(self.bars["Code"] == "30030") & (self.bars["Date"] > "2024-01-15") & (self.bars["Date"] <= "2024-01-19")]
        self.assertEqual(len(b), 4)
        self.assertAlmostEqual(r["wk_high"], b["AdjH"].max() / c0 - 1, places=12)
        self.assertAlmostEqual(r["wk_low"], b["AdjL"].min() / c0 - 1, places=12)
        r2 = self.row("10010", "2024-02-16")                      # 月曜が祝日の週の火曜の候補
        self.assertEqual(r2["wk_days"], 3)

    def test_no_bar_on_week_end(self):
        r = self.row("40040", "2024-01-19")
        self.assertEqual(r["wk_days"], 3)
        self.assertTrue(np.isnan(r["wk_ret"]))                    # 週末の終値が無い
        self.assertTrue(np.isfinite(r["wk_high"]))                # 間の日の高値はある

    def test_rejects(self):
        bad = self.full.copy()
        bad.loc[0, "Date"] = pd.Timestamp("2024-01-13")           # 土曜
        with self.assertRaises(SystemExit):
            W.weekly_frame(bad, self.bars, self.days)
        clash = self.full.assign(wk_ret=0.0)
        with self.assertRaises(SystemExit):
            W.weekly_frame(clash, self.bars, self.days)

    def test_info(self):
        self.assertEqual(self.info["rows"], 5)
        self.assertAlmostEqual(sum(self.info["weekday_share"]), 1.0)
        self.assertAlmostEqual(self.info["multi_share"], 1 / 5)


class TestWeeklyExit(SmallWeek):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.wx, cls.xinfo = W.attach_week_exit(cls.wk, cls.bars, cls.days)

    def test_same_as_attach_exit_on_week_end(self):
        ref = with_daily_exit(self.wk[["Code", "Date"]], self.bars)
        for c in (E.RET, E.R20, E.HITDAY):
            a = self.wx[c].to_numpy(dtype=float)
            b = ref[c].to_numpy(dtype=float)
            self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)), c)
            np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-15, err_msg=c)
        np.testing.assert_array_equal(self.wx[E.HIT].to_numpy(dtype=bool), ref[E.HIT].to_numpy(dtype=bool))

    def test_buy_next_trading_day(self):
        r = self.wx[(self.wx["Code"] == "10010") & (self.wx["Date"] == pd.Timestamp("2024-01-11"))].iloc[0]
        self.assertEqual(pd.Timestamp(r["buy_date"]), pd.Timestamp("2024-01-15"))   # 01-12 は祝日
        self.assertTrue(np.isnan(self.wx.loc[self.wx["Code"] == "40040", E.R20].iloc[0]))  # 週末に日足なし
        self.assertEqual(self.xinfo["next_day_share"], 1.0)
        self.assertEqual(self.xinfo["late"], 0)

    def test_same_day_check(self):
        sd = W.check_same_day(self.wx)
        self.assertEqual(sd["rows"], 1)
        self.assertEqual(sd["mismatch"], 0)
        bad = self.wx.copy()
        i = bad.index[bad["wk_days"] == 0][0]
        bad.loc[i, E.RET] = bad.loc[i, E.RET] + 0.01
        self.assertEqual(W.check_same_day(bad)["ret"], 1)
        bad.loc[i, E.R20] = np.nan
        self.assertEqual(W.check_same_day(bad)["r20"], 1)

    def test_check_week_cols(self):
        cw = W.check_week_cols(self.wk, self.bars, self.days, n=50)
        self.assertEqual((cw["rows"], cw["same"]), (5, 5))
        bad = self.wk.copy()
        bad.loc[bad["Code"] == "30030", "wk_high"] += 1e-6
        cw2 = W.check_week_cols(bad, self.bars, self.days, n=50)
        self.assertEqual(cw2["same"], 4)
        bad2 = self.wk.copy()
        bad2.loc[bad2["Code"] == "30030", "wk_days"] = 3
        self.assertEqual(W.check_week_cols(bad2, self.bars, self.days, n=50)["same"], 4)


def scored(seed=0):
    """3つの窓 × 各10週 × 1週6件の、4本のスコアと収益（W のスコアだけ収益と強く結びつく）。"""
    rng = np.random.default_rng(seed)
    rows = []
    fri = pd.Timestamp("2021-01-08")
    for f in (1, 2, 3):
        for k in range(10):
            d = fri + pd.Timedelta(weeks=(f - 1) * 20 + k)
            for j in range(6):
                rows.append({"Code": f"{1000 + j}", "Date": d, "cand_date": d - pd.Timedelta(days=int(j % 5)),
                             "fold": f})
    o = pd.DataFrame(rows)
    z = rng.normal(size=len(o))
    o[E.R20] = 0.03 * z + rng.normal(0, 0.03, len(o))
    o[E.RET] = np.minimum(o[E.R20], 0.10)
    o[E.HIT] = o[E.RET] >= 0.10
    o[W.D_RET] = o[E.RET] - 0.005
    o[W.D_R20] = o[E.R20]
    o["score_W"] = z
    for a in ("WX", "D", "B"):
        o[f"score_{a}"] = rng.normal(size=len(o))
    return o


class TestEvaluate(unittest.TestCase):
    def test_top_per_week(self):
        o = pd.DataFrame({"Code": list("abcde") + ["f"], "Date": [pd.Timestamp("2021-01-08")] * 5 + [pd.Timestamp("2021-01-15")],
                          "s": [0.1, 0.9, 0.5, 0.9, 0.3, 0.2]})
        t = W.top_per_week(o, "s", 2)
        self.assertEqual(list(t["Code"]), ["b", "d", "f"])           # 同じスコアは銘柄コードの順・1件の週はその1件
        self.assertEqual(list(W.top_per_week(o, "s", 1)["Code"]), ["b", "f"])

    def test_stats(self):
        sel = pd.DataFrame({"fold": [1, 1, 2], "Date": pd.to_datetime(["2021-01-08", "2021-01-08", "2021-07-02"]),
                            E.RET: [0.10, -0.20, 0.02], E.HIT: [True, False, False], E.R20: [0.12, -0.20, 0.02]})
        s = W.pick_stats(sel)
        self.assertEqual((s["n"], s["groups"], s["windows"]), (3, 2, 2))
        self.assertAlmostEqual(s["ret"], (0.10 - 0.20 + 0.02) / 3)
        self.assertAlmostEqual(s["ret_w"], ((0.10 - 0.20) / 2 + 0.02) / 2)
        self.assertAlmostEqual(s["bad"], 1 / 3)
        self.assertAlmostEqual(s["bad_w"], (0.5 + 0.0) / 2)
        self.assertAlmostEqual(s["hit"], 1 / 3)
        e = W.pick_stats(sel.iloc[:0])
        self.assertEqual(e["n"], 0)
        self.assertTrue(np.isnan(e["ret_w"]))

    def test_weekday(self):
        o = scored()
        picks = {a: W.top_per_week(o, f"score_{a}", 2) for a in W.ARMS}
        wd = W.weekday_stats(o, picks)
        self.assertAlmostEqual(sum(wd["share"]), 1.0)
        self.assertAlmostEqual(sum(wd["pick_share"]["W"]), 1.0)
        cw = pd.to_datetime(o["cand_date"]).dt.weekday
        self.assertAlmostEqual(wd["tf_cand"], float(cw.isin([3, 4]).mean()))
        pw = pd.to_datetime(picks["W"]["cand_date"]).dt.weekday
        self.assertAlmostEqual(wd["tf_pick"]["W"], float(pw.isin([3, 4]).mean()))
        for a in W.ARMS:                                              # 週の中の百分位の平均は全体で 0.5
            n = np.array(wd["n"], dtype=float)
            self.assertAlmostEqual(float(np.nansum(np.array(wd["pct"][a]) * n) / n.sum()), 0.5)

    def test_evaluate(self):
        o = scored()
        od = pd.DataFrame({"Code": [f"{2000 + i}" for i in range(1800)],
                           "Date": np.repeat(pd.bdate_range("2021-01-04", periods=600), 3),
                           "fold": np.repeat([1, 2, 3], 600)})
        rng = np.random.default_rng(1)
        od[E.RET] = rng.normal(0, 0.05, len(od))
        od[E.HIT] = od[E.RET] > 0.08
        od[E.R20] = od[E.RET]
        od["score_B"] = od[E.RET] + rng.normal(0, 0.05, len(od))
        res = W.evaluate(o, od, 2)
        self.assertEqual(res["top"]["W"]["n"], 60)                    # 30週 × 2件
        self.assertGreater(res["top"]["W"]["ret_w"], res["all"]["ret_w"])
        self.assertGreater(res["rank"]["W"]["ndcg"], res["rank"]["B"]["ndcg"])
        self.assertEqual(res["diff"]["B"]["n"], 3)
        self.assertEqual(res["daily_ref_windows"], [2, 3])            # 最初の窓は前の分布が無い
        self.assertGreater(res["daily_ref"]["n"], 0)
        self.assertLessEqual(res["top_ref_windows"]["W"]["n"], 40)

    def test_reading(self):
        def row(sh, w, b, d, tf):
            st = lambda r, bad: {"ret_w": r, "bad_w": bad}  # noqa: E731
            return {"shift": sh, "top": {"W": st(*w), "B": st(*b), "D": st(*d)},
                    "weekday": {"tf_pick": {"W": 0.4 + tf / 100}, "tf_cand": 0.4}}

        good = [row(s, (0.02, 0.05), (0.01, 0.08), (0.015, 0.06), 6.0) for s in (0, 2, 4)]
        r = W.reading(good)
        self.assertTrue(r["better"])
        self.assertEqual(r["lean"], "thu_fri")
        mixed = good[:2] + [row(4, (0.02, 0.07), (0.01, 0.08), (0.015, 0.06), -6.0)]   # D より大負けが多い
        r = W.reading(mixed)
        self.assertFalse(r["better"])
        self.assertEqual(r["lean"], "none")
        avoid = [row(s, (0.02, 0.05), (0.03, 0.08), (0.015, 0.06), -5.0) for s in (0, 2, 4)]
        r = W.reading(avoid)
        self.assertFalse(r["better"])                                  # B より平均収益が低い
        self.assertEqual(r["lean"], "avoid")
        nan = [row(0, (np.nan, 0.05), (0.01, 0.08), (0.015, 0.06), np.nan)]
        self.assertFalse(W.reading(nan)["better"])
        self.assertEqual(W.reading(nan)["lean"], "none")


def long_frames(seed=0, n_codes=12):
    """2016〜2021年の週の候補（1週 n_codes 件）と日次の候補。out-of-fold の窓が作れる長さ。"""
    rng = np.random.default_rng(seed)
    fridays = pd.date_range("2016-01-08", "2021-12-31", freq="W-FRI")
    rows = []
    for d in fridays:
        for j in range(n_codes):
            back = int(rng.integers(0, 5))
            rows.append({"Code": f"{3000 + j}", "Date": d, "cand_date": d - pd.offsets.BDay(back), "wk_days": back,
                         "wk_hits": 1})
    wk = pd.DataFrame(rows)
    n = len(wk)
    z = rng.normal(size=n)
    for c in ("f1", "f2", "f3"):
        wk[c] = z + rng.normal(0, 1.0, n)
    wk["wk_ret"] = rng.normal(0, 0.02, n)
    wk["wk_high"] = np.abs(wk["wk_ret"]) + 0.01
    wk["wk_low"] = -np.abs(wk["wk_ret"]) - 0.01
    wk[E.R20] = 0.03 * z + rng.normal(0, 0.03, n)
    wk[E.RET] = np.minimum(wk[E.R20], 0.10)
    wk[E.HIT] = wk[E.RET] >= 0.10
    wk[W.D_RET] = wk[E.RET]
    wk[W.D_R20] = wk[E.R20]
    wk["label"] = (z > 0.8).astype(float)
    days = pd.bdate_range("2016-01-04", "2021-12-31")
    daily = pd.DataFrame({"Code": np.tile([f"{3000 + j}" for j in range(3)], len(days)),
                          "Date": np.repeat(days, 3)})
    m = len(daily)
    zd = rng.normal(size=m)
    for c in ("f1", "f2", "f3"):
        daily[c] = zd + rng.normal(0, 1.0, m)
    daily[E.R20] = 0.03 * zd + rng.normal(0, 0.03, m)
    daily[E.RET] = np.minimum(daily[E.R20], 0.10)
    daily[E.HIT] = daily[E.RET] >= 0.10
    daily["label"] = (zd > 0.8).astype(float)
    return wk, daily


class TestOof(unittest.TestCase):
    def test_windows_columns_and_cache(self):
        wk, daily = long_frames()
        cols = ["f1", "f2", "f3"]
        cols_w = cols + list(W.WEEK_COLS)
        params = {"W": dict(SMALL_RANK, num_leaves=5), "D": dict(SMALL_RANK, num_leaves=9), "B": SMALL_CLF}
        calls = []

        def fake_rank(p, tr, te, c):
            calls.append(("rank", p["num_leaves"], list(c), pd.Timestamp(tr["Date"].max()),
                          pd.Timestamp(te["Date"].min()), pd.Timestamp(te["Date"].max()),
                          pd.Timestamp(te["cand_date"].min()), "wk_days" in tr.columns))
            return np.zeros(len(te))

        def fake_base(p, tr, tes, c):
            calls.append(("base", None, list(c), pd.Timestamp(tr["Date"].max()), None, None,
                          pd.Timestamp(tes[0]["cand_date"].min()), "wk_days" in tr.columns))
            return [np.zeros(len(t)) for t in tes]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(W, "fit_rank", side_effect=fake_rank), \
                mock.patch.object(W, "fit_base", side_effect=fake_base):
            o, od = W.oof_shift(wk, daily, cols, cols_w, params, 0, out_dir=tmp)
            folds = {f.index: f for f in E.E41.folds_for(wk["Date"], 0)}
            self.assertGreater(o["fold"].nunique(), 1)
            for c in ("score_W", "score_WX", "score_D", "score_B", "fold", "cand_date", W.D_RET):
                self.assertIn(c, o.columns)
            self.assertEqual(list(od.columns), list(W.KEEP_D) + ["score_B", "fold"])
            n_folds = o["fold"].nunique()
            self.assertEqual(len(calls), 4 * n_folds)
            for k, f in enumerate(sorted(o["fold"].unique())):
                fd = folds[f]
                end, lo, hi = pd.Timestamp(fd.train_end), pd.Timestamp(fd.test_start), pd.Timestamp(fd.test_end)
                wcall, wxcall, dcall, bcall = calls[4 * k: 4 * k + 4]
                self.assertEqual(wcall[2], cols_w)                     # W だけ値動きの列を使う
                self.assertEqual(wxcall[2], cols)
                self.assertEqual(dcall[2], cols)
                self.assertEqual(bcall[2], cols)
                self.assertTrue(wcall[7] and wxcall[7])                # W・WX は週の行で学ぶ
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
            o2, od2 = W.oof_shift(wk, daily, cols, cols_w, params, 0, out_dir=tmp)   # 保存したものを読む
            self.assertEqual(len(calls), n_calls)
            pd.testing.assert_frame_equal(o.reset_index(drop=True), o2)
            params2 = dict(params, W=dict(SMALL_RANK, num_leaves=6))
            W.oof_shift(wk, daily, cols, cols_w, params2, 0, out_dir=tmp)            # 木の形が変われば作り直す
            self.assertGreater(len(calls), n_calls)

    def test_real_fits(self):
        wk, daily = long_frames()
        tr, te = wk[wk["Date"] < "2019-01-01"], wk[wk["Date"] >= "2019-01-01"]
        s = W.fit_rank(SMALL_RANK, tr, te, ["f1", "f2", "f3", "wk_days", "wk_ret", "wk_high", "wk_low"])
        self.assertEqual(len(s), len(te))
        self.assertTrue(np.isfinite(s).all())
        a, b = W.fit_base(SMALL_CLF, daily, [te, te.iloc[:0]], ["f1", "f2", "f3"])
        self.assertEqual(len(a), len(te))
        self.assertTrue(((a > 0) & (a < 1)).all())
        self.assertEqual(len(b), 0)


class TestParams(unittest.TestCase):
    def test_base_params(self):
        cols = ["a", "b"]
        good = {F.DEFAULT_PRESET: {"_features_sig": F.signature(cols), "_n_features": 2, "learning_rate": 0.05}}
        p = W.base_params(cols, store=good)
        self.assertEqual(p["learning_rate"], 0.05)
        self.assertNotIn("_features_sig", p)
        with self.assertRaises(SystemExit):
            W.base_params(cols, store={F.DEFAULT_PRESET: {"_features_sig": "other", "_n_features": 2}})
        with self.assertRaises(SystemExit):
            W.base_params(cols, store={})

    def test_daily_ltr_params(self):
        cols = ["a", "b"]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "ltr_params.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"params": {"num_leaves": 9}, "_features_sig": F.signature(cols), "_n_features": 2}, fh)
            self.assertEqual(W.daily_ltr_params(cols, p)["params"], {"num_leaves": 9})
            with self.assertRaises(SystemExit):
                W.daily_ltr_params(["a", "b", "c"], p)


class TestMain(unittest.TestCase):
    """通しで回す（データの読み込み・探索・木の形の読み込みは差し替え。当てはめは小さな木で本物）。"""

    def test_main(self):
        rng = np.random.default_rng(3)
        days = pd.bdate_range("2016-01-04", "2021-12-31")
        codes = [f"{5000 + j}" for j in range(12)]
        bars = make_bars(codes, days, seed=4)
        weeks = pd.Series(days).groupby(pd.Series(days).dt.to_period("W-SUN")).apply(list)
        rows = []
        for ws in weeks:
            for c in codes:
                k = int(rng.integers(0, len(ws)))
                rows.append({"Code": c, "Date": ws[k]})
                if rng.random() < 0.2 and k > 0:                       # 同じ週に2回入る銘柄
                    rows.append({"Code": c, "Date": ws[0]})
        cands = pd.DataFrame(rows).drop_duplicates(["Code", "Date"]).sort_values(["Date", "Code"]).reset_index(drop=True)
        n = len(cands)
        z = rng.normal(size=n)
        for c in ("f1", "f2", "f3"):
            cands[c] = z + rng.normal(0, 1.0, n)
        cands["label"] = (z > 0.8).astype(float)
        cands["cap_band"] = rng.integers(0, 5, n).astype(float)
        full = with_daily_exit(cands, bars)
        cutoff = pd.Timestamp("2019-12-31")
        chk = {"hold5_match": 0, "hold5_both": 0}

        def fake_tune(sub, cols, n_trials, folds):
            self.assertEqual(list(cols), ["f1", "f2", "f3"] + list(W.WEEK_COLS))
            self.assertLessEqual(sub["Date"].max(), cutoff)
            self.assertTrue(sub["label"].notna().all())
            return {"params": dict(SMALL_RANK), "_cv": {"objective": "x", "ndcg": 0.5, "best_trial": 0, "trials": []},
                    "_minutes": 0.0}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(W, "OOF_DIR", tmp), \
                mock.patch.object(W, "PARAMS_PATH", os.path.join(tmp, "e115_params.json")), \
                mock.patch.object(W, "SUMMARY_PATH", os.path.join(tmp, "e115_summary.json")), \
                mock.patch.object(W.F, "columns", return_value=["f1", "f2", "f3"]), \
                mock.patch.object(W, "load_all", return_value=(full, bars, cutoff, chk)), \
                mock.patch.object(W, "base_params", return_value=dict(SMALL_CLF)), \
                mock.patch.object(W, "daily_ltr_params", return_value={"params": dict(SMALL_RANK), "_cutoff": "x"}), \
                mock.patch.object(W.E, "tune_l", side_effect=fake_tune) as tl:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(W.main(["--trials", "2", "--shifts", "0,2"]), 0)
            out = buf.getvalue()
            with open(os.path.join(tmp, "e115_summary.json"), encoding="utf-8") as fh:
                sm = json.load(fh)
            self.assertEqual(tl.call_count, 1)
            self.assertEqual(sorted(sm["shifts"]), ["0", "2"])
            self.assertEqual(sm["check_same_day"]["mismatch"], 0)
            self.assertEqual(sm["check_cols"]["same"], sm["check_cols"]["rows"])
            self.assertIn(sm["reading"]["lean"], ("thu_fri", "avoid", "none"))
            self.assertEqual(sm["shifts"]["0"]["top"]["W"]["n"], 2 * sm["shifts"]["0"]["weeks"])
            self.assertIn("読み方", out)
            self.assertIn("週ごとの上位2件", out)
            for code in codes:                                         # 公開ログに銘柄コードを出さない
                self.assertNotIn(code, out)
            with contextlib.redirect_stdout(io.StringIO()):            # 2回目は探索と out-of-fold を読む
                self.assertEqual(W.main(["--trials", "2", "--shifts", "0,2"]), 0)
            self.assertEqual(tl.call_count, 1)


if __name__ == "__main__":
    unittest.main()
