#!/usr/bin/env python3
"""
実験118（research/exp/e118_ltr5_grades.py）。合成の日足と候補で:

- 段階: 5段階は本番（ltr.grades）と同じ。10段階は組の中の10分位（10件の組は 0〜9 が1件ずつ、20件の組は2件ずつ、1件の組は
  真ん中の 5）
- 当てはめ: 5段階は本番の当てはめ（ltr.fit・ltr.predict）とスコアが同じ。LGBMRanker に渡す label_gain と段階は 0〜n−1
- 探索: 5段階は本番の探索（ltr5.tune）と同じ木の形・同じ値になる。10段階は利得も10段階
- 確かめ: ずらし0の最初の窓の訓練で、5段階の当てはめと本番の当てはめが同じ
- out-of-fold: 窓ごとに G5（5段階）・G10・G10T（10段階）を値動きの列ありで組の行から、D を値動きの列なしで日次の行から学び、
  テスト窓の組の行を採点する。保存した out-of-fold は次の回に読み、木の形が変われば作り直す
- 集計: 組の上位（同じスコアは銘柄コードの順）・組の1位と上位2件が G5 と違う割合・組の大きさの3分位・印がそろう判断日・3条件
  （日次の1位・組の1位・R0）と4条件（＋R15）（手で数える例。持っている銘柄は買い増さない）
- 読み方（回す前に決めたもの。丸めた値で、すべてのずらしで組の1位の平均収益と順位相関がどちらも上 / どちらも下）
- 通しで回す（データの読み込み・印・探索は差し替え、当てはめは小さな木で本物）: 集計と読み方を記録に書き、公開ログに銘柄
  コードを出さない。探索は2回目に読む。本番の out-of-fold が無ければ止まる

  python3 tests/test_e118_ltr5_grades.py
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

import ltr as LTR  # noqa: E402
import ltr5 as L5  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e118_ltr5_grades as X  # noqa: E402
import test_e115_weekly_ltr as T115  # noqa: E402
import test_e117_rolling5_ltr as T117  # noqa: E402

COLS = ["f1", "f2", "f3"]
COLS_W = COLS + list(X.MOVE_COLS)


def pool_rows(seed=0, days=40, per_day=12):
    """判断日ごとに per_day 件の組の行（値動きの列・収益つき）。"""
    rng = np.random.default_rng(seed)
    ds = pd.bdate_range("2020-01-06", periods=days)
    rows = [{"Code": f"{4000 + j}", "Date": d} for d in ds for j in range(per_day)]
    f = pd.DataFrame(rows)
    n = len(f)
    z = rng.normal(size=n)
    for c in COLS:
        f[c] = z + rng.normal(0, 1.0, n)
    for c in X.MOVE_COLS:
        f[c] = rng.normal(0, 0.02, n)
    f[X.R20] = 0.03 * z + rng.normal(0, 0.03, n)
    return f


class TestGrades(unittest.TestCase):
    def test_five_same_as_production(self):
        rng = np.random.default_rng(1)
        dates = np.repeat(pd.bdate_range("2021-01-04", periods=30).to_numpy(), rng.integers(1, 25, 30))
        r = np.round(rng.normal(0, 0.05, len(dates)), 2)                  # 同じ値も入れる
        np.testing.assert_array_equal(X.grades(dates, r, 5), LTR.grades(dates, r))

    def test_ten(self):
        d10 = np.array(["2021-01-04"] * 10)
        self.assertEqual(sorted(X.grades(d10, np.arange(10.0), 10).tolist()), list(range(10)))
        d20 = np.array(["2021-01-04"] * 20)
        g = X.grades(d20, np.arange(20.0)[::-1], 10)
        self.assertEqual(sorted(g.tolist()), sorted(list(range(10)) * 2))
        self.assertEqual(g[0], 9)                                         # 収益の一番大きい行が一番上の段階
        self.assertEqual(X.grades(np.array(["2021-01-04"]), np.array([0.1]), 10).tolist(), [5])
        self.assertEqual(X.grades(np.array(["2021-01-04"]), np.array([0.1]), 5).tolist(), [2])


class TestFit(unittest.TestCase):
    def test_five_same_as_production(self):
        f = pool_rows()
        tr, te = f[f["Date"] < f["Date"].unique()[30]], f[f["Date"] >= f["Date"].unique()[30]]
        params = dict(T115.SMALL_RANK, num_leaves=5)
        mine = X.fit_rank(params, tr.sample(frac=1.0, random_state=0), te, COLS_W, 5)   # 並びが違っても同じ
        prod = LTR.predict(LTR.fit(tr, COLS_W, params, X.SEED), te[COLS_W].to_numpy(dtype=float))
        np.testing.assert_array_equal(mine, prod)

    def test_gain_and_grades(self):
        f = pool_rows(seed=2)
        seen = {}

        class FakeRanker:
            def __init__(self, **kw):
                seen["kw"] = kw

            def fit(self, Xa, y, group):
                seen["y"] = np.asarray(y)
                seen["group"] = np.asarray(group)
                return self

            def predict(self, Xa):
                return np.zeros(len(Xa))

        with mock.patch("lightgbm.LGBMRanker", FakeRanker):
            out = X.fit_rank(dict(T115.SMALL_RANK), f, f.head(5), COLS_W, 10)
        self.assertEqual(len(out), 5)
        self.assertEqual(seen["kw"]["objective"], "lambdarank")
        self.assertEqual(seen["kw"]["label_gain"], list(range(10)))
        self.assertEqual(seen["kw"]["random_state"], X.SEED)
        self.assertEqual(int(seen["y"].min()), 0)
        self.assertEqual(int(seen["y"].max()), 9)
        self.assertEqual(seen["group"].tolist(), [12] * 40)

    def test_check_fit(self):
        rl, _, _ = T117.long_frames()
        chk = X.check_fit(rl, COLS_W, dict(T115.SMALL_RANK, num_leaves=5))
        self.assertTrue(chk["same"])
        self.assertEqual(chk["max_diff"], 0.0)
        self.assertGreater(chk["train"], 0)
        self.assertGreater(chk["rows"], 0)


class TestTune(unittest.TestCase):
    def folds(self, f):
        ds = f["Date"].unique()
        return [(f[~f["Date"].isin(ds[k::3])], f[f["Date"].isin(ds[k::3])]) for k in range(3)]

    def test_five_same_as_production(self):
        f = pool_rows(seed=3, days=45, per_day=10)
        folds = self.folds(f)
        mine = X.tune_g(f, COLS_W, folds, 5, n_trials=2)
        prod = L5.tune(f, COLS_W, folds, n_trials=2)
        self.assertEqual(mine["params"], prod["params"])
        self.assertEqual(mine["_cv"]["ndcg"], prod["_cv"]["ndcg"])
        self.assertEqual(mine["_cv"]["ndcg_folds"], prod["_cv"]["ndcg_folds"])
        self.assertEqual(mine["_cv"]["best_trial"], prod["_cv"]["best_trial"])

    def test_ten_uses_ten_gains(self):
        f = pool_rows(seed=4, days=30, per_day=10)
        gains = []
        real = LTR.ndcg_at_k

        def spy(dates, score, gain, k=5, min_n=5):
            gains.append(np.asarray(gain))
            return real(dates, score, gain, k, min_n)

        with mock.patch.object(X, "fit_rank", side_effect=lambda p, tr, va, c, n, s: va[X.R20].to_numpy()), \
                mock.patch.object(X.LTR, "ndcg_at_k", side_effect=spy):
            rec = X.tune_g(f, COLS_W, self.folds(f), 10, n_trials=1)
        self.assertEqual(max(int(g.max()) for g in gains), 9)
        self.assertEqual(rec["_cv"]["grades"], 10)
        self.assertIn("0〜9", rec["_cv"]["objective"])
        self.assertEqual(rec["_cv"]["ndcg"], 1.0)                          # 収益の順に並べれば満点


class TestOOF(unittest.TestCase):
    def test_windows_columns_and_cache(self):
        rl, daily, _ = T117.long_frames()
        params = {"G5": dict(T115.SMALL_RANK, num_leaves=5), "G10": dict(T115.SMALL_RANK, num_leaves=5),
                  "G10T": dict(T115.SMALL_RANK, num_leaves=6), "D": dict(T115.SMALL_RANK, num_leaves=9)}
        calls = []

        def fake_g(p, tr, te, c, n, seed=X.SEED):
            calls.append(("g", n, p["num_leaves"], list(c), pd.Timestamp(tr["Date"].max()),
                          pd.Timestamp(te["Date"].min()), "r5_days" in tr.columns))
            return np.zeros(len(te))

        def fake_d(p, tr, te, c):
            calls.append(("d", None, p["num_leaves"], list(c), pd.Timestamp(tr["Date"].max()),
                          pd.Timestamp(te["Date"].min()), "r5_days" in tr.columns))
            return np.zeros(len(te))

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(X, "fit_rank", side_effect=fake_g), \
                mock.patch.object(X.W, "fit_rank", side_effect=fake_d):
            o = X.oof_shift(rl, daily, COLS, COLS_W, params, 0, out_dir=tmp)
            folds = {f.index: f for f in E.E41.folds_for(rl["Date"], 0)}
            n_folds = o["fold"].nunique()
            self.assertGreater(n_folds, 1)
            for c in ("score_G5", "score_G10", "score_G10T", "score_D", "fold", "cand_date", "r5_days", E.HITDAY):
                self.assertIn(c, o.columns)
            self.assertEqual(len(calls), 4 * n_folds)
            for k, f in enumerate(sorted(o["fold"].unique())):
                fd = folds[f]
                end = pd.Timestamp(fd.train_end)
                g5, g10, g10t, d = calls[4 * k: 4 * k + 4]
                self.assertEqual([g5[1], g10[1], g10t[1]], [5, 10, 10])          # 段階の数
                self.assertEqual([g5[2], g10[2], g10t[2], d[2]], [5, 5, 6, 9])   # 木の形
                for c in (g5, g10, g10t):
                    self.assertEqual(c[3], COLS_W)                               # 値動きの列ありで組の行から学ぶ
                    self.assertTrue(c[6])
                self.assertEqual(d[3], COLS)
                self.assertFalse(d[6])                                           # D は日次の行から学ぶ
                for c in (g5, g10, g10t, d):
                    self.assertLessEqual(c[4], end)
                    self.assertGreaterEqual(c[5], pd.Timestamp(fd.test_start))
            n_calls = len(calls)
            o2 = X.oof_shift(rl, daily, COLS, COLS_W, params, 0, out_dir=tmp)   # 保存したものを読む
            self.assertEqual(len(calls), n_calls)
            pd.testing.assert_frame_equal(o.reset_index(drop=True), o2)
            X.oof_shift(rl, daily, COLS, COLS_W, dict(params, G10T=dict(T115.SMALL_RANK, num_leaves=7)), 0, out_dir=tmp)
            self.assertGreater(len(calls), n_calls)                              # 木の形が変われば作り直す


def scored(o: pd.DataFrame, seed=0) -> pd.DataFrame:
    """組の行に腕と D のスコア・印を足したもの（集計のテスト用）。"""
    rng = np.random.default_rng(seed)
    o = o.copy()
    z = o[X.RET].to_numpy(dtype=float)
    for a, s in (("G5", 0.5), ("G10", 0.6), ("G10T", 0.4), ("D", 0.5)):
        o[f"score_{a}"] = s * z + rng.normal(0, 0.02, len(o))
    o["fold"] = pd.to_datetime(o["Date"]).dt.year
    o["r0"] = (rng.random(len(o)) < 0.3).astype(float)
    o["r15"] = (rng.random(len(o)) < 0.3).astype(float)
    early = pd.to_datetime(o["Date"]) < pd.Timestamp("2020-01-01")
    o.loc[early, "r0"] = np.nan                                            # 本番の百分位が付かない期間のように
    o.loc[pd.to_datetime(o["Date"]) < pd.Timestamp("2020-07-01"), "r15"] = np.nan
    return o


class TestEvaluateParts(unittest.TestCase):
    def frame(self):
        d1, d2 = pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")
        return pd.DataFrame({
            "Date": [d1, d1, d1, d2, d2, d2],
            "Code": ["1001", "1002", "1003", "1001", "1002", "1003"],
            "score_G5": [0.9, 0.5, 0.1, 0.2, 0.8, 0.8],
            "score_G10": [0.9, 0.5, 0.1, 0.9, 0.1, 0.1],
        })

    def test_top_rows_ties_by_code(self):
        t = X.top_rows(self.frame(), "score_G5", 1)
        self.assertEqual(t["Code"].tolist(), ["1001", "1002"])               # 2日目は同点で小さいコード
        t2 = X.top_rows(self.frame(), "score_G5", 2)
        self.assertEqual(t2["Code"].tolist(), ["1001", "1002", "1002", "1003"])

    def test_change_stats(self):
        f = self.frame()
        with mock.patch.object(X, "MIN_POOL", 3):
            c = X.change_stats(f, "G10")
        self.assertEqual(c["days"], 2)
        self.assertEqual(c["top1_changed"], 0.5)                              # 2日目だけ1位が違う
        self.assertEqual(c["top2_changed"], 0.5)                              # 2日目は {1002,1003} → {1001,1002}

    def test_flagged(self):
        f = self.frame().assign(r0=[1.0, 0.0, np.nan, 1.0, 0.0, 0.0], r15=[1.0, 1.0, 1.0, 1.0, np.nan, 0.0])
        self.assertEqual(X.flagged(f, ("r0",))["Date"].nunique(), 1)
        self.assertEqual(X.flagged(f, ("r0",))["Date"].iloc[0], pd.Timestamp("2024-01-05"))
        self.assertEqual(len(X.flagged(f, ("r0", "r15"))), 0)

    def test_size_stats(self):
        rows = []
        for i, n in enumerate([3, 3, 6, 6, 9, 9]):
            d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            for j in range(n):
                rows.append({"Date": d, "Code": f"{1000 + j}", X.RET: 0.01 * j,
                             "score_G5": float(j == 0), "score_G10": float(j == n - 1), "score_G10T": float(j == 1)})
        z = X.size_stats(pd.DataFrame(rows))
        self.assertEqual([b["days"] for b in z["bins"]], [2, 2, 2])
        self.assertEqual([(b["n_min"], b["n_max"]) for b in z["bins"]], [(3, 3), (6, 6), (9, 9)])
        big = z["bins"][2]
        self.assertAlmostEqual(big["ret"]["G5"], 0.0)
        self.assertAlmostEqual(big["ret"]["G10"], 0.08)
        self.assertAlmostEqual(big["diff"]["G10"], 0.08)
        self.assertAlmostEqual(big["diff"]["G10T"], 0.01)
        self.assertEqual(z["ge10"], 0.0)

    def test_mark_buys_by_hand(self):
        days = pd.bdate_range("2024-01-01", "2024-03-29")
        d = days[5]
        base = {X.RET: 0.05, X.HIT: False, X.HITDAY: np.nan, X.R20: 0.05}
        rows = [
            # 判断日 d の候補（r5_days = 0）: A は日次の1位・組の1位、B は日次の2位
            dict(base, Code="A", Date=d, cand_date=d, r5_days=0, score_D=0.9, score_G5=0.9, r0=1.0, r15=1.0),
            dict(base, Code="B", Date=d, cand_date=d, r5_days=0, score_D=0.5, score_G5=0.1, r0=1.0, r15=0.0),
            # 前の日からの候補（組の1位にはならない）
            dict(base, Code="C", Date=d, cand_date=days[3], r5_days=2, score_D=0.0, score_G5=0.5, r0=1.0, r15=1.0),
            # 次の判断日: C が組の1位だが候補の日ではないので3条件ではない。B は日次の1位だが組の1位ではない
            dict(base, Code="B", Date=days[6], cand_date=days[6], r5_days=0, score_D=0.9, score_G5=0.2, r0=1.0, r15=1.0),
            dict(base, Code="C", Date=days[6], cand_date=days[3], r5_days=3, score_D=0.0, score_G5=0.8, r0=1.0, r15=1.0),
            # さらに次の判断日: A がまた日次の1位・組の1位だが、まだ持っている（20営業日）ので買い増さない
            dict(base, Code="A", Date=days[7], cand_date=days[7], r5_days=0, score_D=0.9, score_G5=0.9, r0=1.0, r15=1.0),
        ]
        b = X.mark_buys(pd.DataFrame(rows), days, "G5")
        self.assertEqual(b["C3"][["Code", "Date"]].values.tolist(), [["A", d]])
        self.assertEqual(b["C4"][["Code", "Date"]].values.tolist(), [["A", d]])
        self.assertEqual(b["W1R0"][["Code", "Date"]].values.tolist(), [["A", d]])
        # d の A に R15 が無ければ、d の A は3条件だけ。4条件は d では買わず、days[7] の A を買う（3条件では d に買って
        # 持っているので買い増さない）
        rows[0]["r15"] = 0.0
        b = X.mark_buys(pd.DataFrame(rows), days, "G5")
        self.assertEqual(b["C3"][["Code", "Date"]].values.tolist(), [["A", d]])
        self.assertEqual(b["C4"][["Code", "Date"]].values.tolist(), [["A", days[7]]])
        rows[0]["r0"] = 0.0                                                    # d の A に R0 も無ければ3条件でもない
        b = X.mark_buys(pd.DataFrame(rows), days, "G5")
        self.assertEqual(b["C3"][["Code", "Date"]].values.tolist(), [["A", days[7]]])
        self.assertEqual(b["W1R0"][["Code", "Date"]].values.tolist(), [["A", days[7]]])

    def test_evaluate(self):
        rl, _, days = T117.long_frames()
        o = scored(rl)
        res = X.evaluate(o, days)
        self.assertEqual(res["top1"]["G5"]["n"], res["days"])                 # 毎日1件
        self.assertEqual(res["top2"]["G5"]["n"], 2 * res["days"])
        for a in X.ARMS:
            self.assertTrue(np.isfinite(res["rank"][a]["rho"]))
            self.assertTrue(0.0 < res["rank"][a]["ndcg5"] <= 1.0)
            self.assertTrue(0.0 < res["rank"][a]["ndcg10"] <= 1.0)
            self.assertLessEqual(res["rule"]["buys"][a]["n"], 2 * res["rule"]["days"])
            for k in X.MARK_SETS:
                self.assertIn(k, res["marks"]["sets"][a])
            self.assertLessEqual(res["marks"]["sets"][a]["C4"]["n"], res["marks"]["sets"][a]["C3"]["n"])
        for a in ("G10", "G10T"):
            self.assertIn(a, res["diff_top1"])
            self.assertTrue(0.0 <= res["change"][a]["top1_changed"] <= 1.0)
        self.assertEqual(sum(b["days"] for b in res["size"]["bins"]), res["days"])
        self.assertLess(res["rule"]["days"], res["days"])                     # 印がそろわない日は数えない
        self.assertLess(res["marks"]["days"], res["rule"]["days"])            # 3条件・4条件は R0 と R15 の両方がそろう日
        self.assertGreater(res["marks"]["days"], 0)


class TestReading(unittest.TestCase):
    def row(self, sh, ret, rho):
        """ret・rho は (G5, G10, G10T)。"""
        return {"shift": sh, "top1": {a: {"ret_w": r} for a, r in zip(X.ARMS, ret)},
                "rank": {a: {"rho": p} for a, p in zip(X.ARMS, rho)}}

    def test_better_worse_none(self):
        rows = [self.row(0, (0.010, 0.012, 0.008), (0.050, 0.060, 0.040)),
                self.row(2, (0.011, 0.013, 0.009), (0.051, 0.061, 0.041)),
                self.row(4, (0.012, 0.014, 0.013), (0.052, 0.062, 0.042))]
        rd = X.reading(rows)
        self.assertEqual(rd["G10"]["read"], "better")
        self.assertEqual(rd["G10T"]["read"], "none")                          # ずらし4か月だけ収益が上
        rows[2] = self.row(4, (0.012, 0.014, 0.010), (0.052, 0.062, 0.042))
        self.assertEqual(X.reading(rows)["G10T"]["read"], "worse")

    def test_rounding_ties(self):
        rows = [self.row(0, (0.01000, 0.01004, 0.01), (0.0500, 0.0504, 0.05))]   # 丸めると同じ → 上でも下でもない
        rd = X.reading(rows)
        self.assertEqual(rd["G10"]["read"], "none")
        self.assertFalse(rd["G10"]["rows"][0]["up"])
        self.assertEqual(X.reading([])["G10"]["read"], "none")


class TestMain(unittest.TestCase):
    """通しで回す（データの読み込み・印・探索・本番の木の形は差し替え。当てはめは小さな木で本物）。"""

    def patches(self, tmp, full, bars, cutoff, fake_rule, fake_tune):
        chk = {"hold5_match": 0, "hold5_both": 0}
        rec5 = {"params": dict(T115.SMALL_RANK, num_leaves=5), "_cv": {"n_rows": 1, "train_to": "2019-01-01"},
                "_source": "テスト"}
        return [mock.patch.object(X, "OOF_DIR", tmp),
                mock.patch.object(X, "PARAMS_PATH", os.path.join(tmp, "e118_params.json")),
                mock.patch.object(X, "SUMMARY_PATH", os.path.join(tmp, "e118_summary.json")),
                mock.patch.object(X.F, "columns", return_value=list(COLS)),
                mock.patch.object(X.W, "load_all", return_value=(full, bars, cutoff, chk)),
                mock.patch.object(X.W, "daily_ltr_params",
                                  return_value={"params": dict(T115.SMALL_RANK, num_leaves=9), "_cutoff": "x"}),
                mock.patch.object(X.L5, "load_params", return_value=rec5),
                mock.patch.object(X.LTR, "params_mismatch", return_value=None),
                mock.patch.object(X.R, "daily_rule", side_effect=fake_rule),
                mock.patch.object(X, "tune_g", side_effect=fake_tune)]

    def test_main(self):
        full, bars, codes = T117.main_data()
        cutoff = pd.Timestamp("2019-12-31")
        no_pct_until = pd.Timestamp("2019-03-01")

        def fake_tune(sub, cols_w, folds, n, n_trials):
            self.assertEqual(list(cols_w), COLS_W)
            self.assertEqual(n, 10)
            self.assertEqual(n_trials, 2)
            self.assertLessEqual(sub["Date"].max(), cutoff)
            self.assertEqual(len(folds), 5)
            return {"params": dict(T115.SMALL_RANK, num_leaves=6),
                    "_cv": {"objective": "x", "grades": 10, "ndcg": 0.5, "best_trial": 0, "minutes": 0.0}}

        def fake_rule(daily, models=None):
            line = 0.9 if tuple(models) == X.R0_TREES else 0.5
            r = daily[["Code", "Date"]].assign(r15=(daily["f1"] > line).astype(float), pmin=np.round(daily["f1"] * 10, 1))
            r.loc[r["Date"] < no_pct_until, ["r15", "pmin"]] = np.nan
            return r, {"found": True, "joined": len(r), "rows": len(r), "with_pct": int(r["r15"].notna().sum()),
                       "r15": int(r["r15"].sum()), "matched": int(r["r15"].notna().sum())}

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            mocks = [st.enter_context(p) for p in self.patches(tmp, full, bars, cutoff, fake_rule, fake_tune)]
            dr, tg = mocks[-2], mocks[-1]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(X.main(["--trials", "2", "--shifts", "0,2"]), 0)
            out = buf.getvalue()
            with open(os.path.join(tmp, "e118_summary.json"), encoding="utf-8") as fh:
                sm = json.load(fh)
            self.assertEqual(tg.call_count, 1)
            self.assertEqual(dr.call_count, 2)                                 # R0 と R15 を1回ずつ
            self.assertEqual(sorted(sm["shifts"]), ["0", "2"])
            self.assertTrue(sm["check_fit"]["same"])
            self.assertFalse(sm["prod_params_same_rows"])
            self.assertAlmostEqual(sum(sm["grade_share10"]), 1.0)
            self.assertEqual(sm["params"]["G10T"]["num_leaves"], 6)
            self.assertEqual(sm["params"]["G5"], sm["params"]["G10"])
            for a in ("G10", "G10T"):
                self.assertIn(sm["reading"][a]["read"], ("better", "worse", "none"))
                self.assertEqual(len(sm["reading"][a]["rows"]), 2)
            s0 = sm["shifts"]["0"]
            self.assertEqual(s0["top1"]["G5"]["n"], s0["days"])
            self.assertLess(s0["rule"]["days"], s0["days"])                   # 印が付かない期間の判断日は数えない
            self.assertGreater(s0["rule"]["days"], 0)
            self.assertIn("読み方", out)
            self.assertIn("組の1位を毎日1件", out)
            self.assertIn("3条件", out)
            for code in codes:                                                 # 公開ログに銘柄コードを出さない
                self.assertNotIn(code, out)
            with contextlib.redirect_stdout(io.StringIO()):                    # 2回目は探索と out-of-fold を読む
                self.assertEqual(X.main(["--trials", "2", "--shifts", "0,2"]), 0)
            self.assertEqual(tg.call_count, 1)

    def test_stops_without_production_oof(self):
        full, bars, _ = T117.main_data()

        def no_rule(daily, models=None):
            return daily[["Code", "Date"]].iloc[:0], {"found": False}

        def never(*a, **k):
            raise AssertionError("探索まで進まない")

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            for p in self.patches(tmp, full, bars, pd.Timestamp("2019-12-31"), no_rule, never):
                st.enter_context(p)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                X.main(["--trials", "2", "--shifts", "0"])

    def test_stops_on_params_mismatch(self):
        full, bars, _ = T117.main_data()
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as st:
            for p in self.patches(tmp, full, bars, pd.Timestamp("2019-12-31"), None, None):
                st.enter_context(p)
            st.enter_context(mock.patch.object(X.LTR, "params_mismatch", return_value="列が違う"))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                X.main(["--trials", "2", "--shifts", "0"])


if __name__ == "__main__":
    unittest.main()
