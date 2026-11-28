#!/usr/bin/env python3
"""
実験131（research/exp/e131_exit_rl.py。売りのタイミングの強化学習）の部品。データには触らない（作りものの値動きで確かめる）。

- 値動きの切り出し: 買値は選定日の翌営業日の寄り。20日目まで日足が無い・値の付かない日がある取引は外す
- 「いつも持ち続ける」方針の損益と買値が live_track.forward と一致する（指値あり・なし）
- 状態は k日目までの値だけから作る（k日目より後の値を変えても変わらない）。手で計算した値と一致する
- 売買の約束: 売ると決めたら翌営業日の寄り、指値を残す版は +10% で頭打ち、指値はその日の判断より先に効く、20日目は終値
- 上限（先が見えた場合）はどの方針よりも低くならない
- 学習: 下げ続ける取引を売ることを学び、作りものの値動きで「いつも持ち続ける」を上回る。学習に使うのは渡した取引だけ
- 判定: 3条件（平均が正・選定日ごとの符号の入れ替えで p < 0.05・上の窓が過半）。窓に5行未満なら数えない
- 窓ごとの評価: 学習の取引は20日目がテスト窓の始まりより前のものだけ。出力に銘柄コードが出ない

  python3 -m unittest tests.test_e131_exit_rl
"""
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e131_exit_rl as E  # noqa: E402

W = E.PRE + 1 + E.HOLD


def make_bars(codes, n=160, seed=0, vol=0.03, start="2022-01-04"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    out = []
    for c in codes:
        close = 1000 * np.exp(np.cumsum(rng.normal(0.001, vol, n)))
        o = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 2, n))
        hi = np.maximum(o, close) * (1 + rng.uniform(0, vol, n))
        lo = np.minimum(o, close) * (1 - rng.uniform(0, vol, n))
        out.append(pd.DataFrame({"Code": c, "Date": d, "AdjO": o, "AdjH": hi, "AdjL": lo, "AdjC": close,
                                 "AdjVo": rng.integers(1000, 90000, n).astype(float)}))
    return pd.concat(out, ignore_index=True)


def make_topix(n=160, seed=9, start="2022-01-04"):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Date": pd.bdate_range(start, periods=n), "topix": 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def synth_paths(n=3000, seed=0, drift_sd=0.012, noise=0.012, start="2019-01-01", days=900):
    """取引ごとに決まった向き（drift）があり、日々の雑音が乗る値動き。下げる取引は下げ続ける。"""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, drift_sd, n)
    r = mu[:, None] + rng.normal(0, noise, (n, W))
    r[:, :E.PRE + 1] = rng.normal(0, noise, (n, E.PRE + 1))             # 買う前は向きなし
    C = 1000 * np.exp(np.cumsum(r, axis=1))
    O = np.column_stack([C[:, 0], C[:, :-1]]) * np.exp(rng.normal(0, noise / 4, (n, W)))
    H = np.maximum(O, C) * (1 + rng.uniform(0, noise / 2, (n, W)))
    L = np.minimum(O, C) * (1 - rng.uniform(0, noise / 2, (n, W)))
    V = rng.integers(1000, 90000, (n, W)).astype(float)
    T = np.exp(np.cumsum(rng.normal(0, 0.005, (n, E.HOLD + 1)), axis=1))
    dates = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, days, n)), unit="D")
    keys = pd.DataFrame({"Code": [f"{10000 + i}" for i in range(n)], "Date": dates})
    scores = pd.DataFrame({c: rng.random(n) for c in E.SCORE_COLS})
    return E.Paths(keys, O, H, L, C, V, T, np.full(n, 30000.0), rng.uniform(1, 5, n), dates + pd.Timedelta(days=30), scores), mu


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.bars = make_bars(["11110", "22220"], seed=1)
        self.days = sorted(self.bars["Date"].unique())
        self.topix = make_topix()

    def test_build_and_drop_reasons(self):
        b = self.bars.copy()
        hole = (b["Code"] == "22220") & (b["Date"] == self.days[60])
        b.loc[hole, ["AdjO", "AdjH", "AdjL", "AdjC"]] = np.nan              # 値の付かない日
        keys = pd.DataFrame({"Code": ["11110", "11110", "22220", "22220", "33330", "11110"],
                             "Date": [self.days[30], self.days[150], self.days[50], self.days[70],
                                      self.days[30], pd.Timestamp("2021-01-01")]})
        P, info = E.build_paths(keys, b, self.topix)
        self.assertEqual(info, {"rows": 6, "ok": 2, "no_bar": 2, "short": 1, "gap": 1})
        self.assertEqual(P.keys["_row"].tolist(), [0, 3])
        g = self.bars[self.bars["Code"] == "11110"].reset_index(drop=True)
        self.assertEqual(P.entry[0], g.loc[31, "AdjO"])                     # 翌営業日の寄り
        self.assertEqual(P.C[0, E.PRE], g.loc[30, "AdjC"])                  # 0日目 = 選定日
        self.assertEqual(P.C[0, E.PRE + E.HOLD], g.loc[50, "AdjC"])
        self.assertEqual(pd.Timestamp(P.end_date[0]), self.days[50])
        self.assertAlmostEqual(P.pre_vol[0], g.loc[11:30, "AdjVo"].mean())
        self.assertAlmostEqual(P.T[0, 3], float(self.topix.set_index("Date").loc[self.days[33], "topix"]))
        self.assertAlmostEqual(P.r_close(E.HOLD)[0], (g.loc[50, "AdjC"] / g.loc[31, "AdjO"] - 1) * 100)

    def test_always_hold_matches_live_track(self):
        bars = make_bars([f"{1000 + i}0" for i in range(12)], n=200, seed=3, vol=0.04)   # 指値に届く取引が混ざる
        days = sorted(bars["Date"].unique())
        rng = np.random.default_rng(0)
        keys = pd.DataFrame({"Code": rng.choice(bars["Code"].unique(), 150), "Date": rng.choice(days[10:170], 150)})
        keys = keys.drop_duplicates().reset_index(drop=True)
        P, info = E.build_paths(keys, bars, make_topix(200))
        self.assertEqual(info["ok"], len(keys))
        with redirect_stdout(io.StringIO()):
            chk = E.check_against_live_track(P, bars[["Date", "Code", "AdjO", "AdjH", "AdjC"]], np.arange(len(P)))
        self.assertEqual(chk["limit"], chk["rows"])
        hits = (E.simulate(P, E.always_hold, "L")["how"] == "limit").mean()
        self.assertTrue(0.1 < hits < 0.9)                                    # 指値あり・なしの両方を通っている
        bad = E.Paths(P.keys, P.O, P.H * 1.2, P.L, P.C, P.V, P.T, P.pre_vol, P.vol_20d, P.end_date, P.scores)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_against_live_track(bad, bars[["Date", "Code", "AdjO", "AdjH", "AdjC"]], np.arange(len(P)))


class TestState(unittest.TestCase):
    def setUp(self):
        self.P, _ = synth_paths(200, seed=2)

    def test_columns(self):
        s = E.state_at(self.P, 7)
        self.assertEqual(list(s.columns), E.STATE_COLS)
        self.assertEqual(len(E.STATE_COLS), 29)
        self.assertEqual(len(set(E.STATE_COLS)), 29)
        for k in (0, E.HOLD):
            with self.assertRaises(ValueError):
                E.state_at(self.P, k)

    def test_no_lookahead(self):
        """k日目より後の値をすべて変えても、k日目の状態は変わらない。"""
        P = self.P
        rng = np.random.default_rng(5)
        for k in (1, 2, 5, 12, 19):
            want = E.state_at(P, k)
            arrs = {}
            for name in ("O", "H", "L", "C", "V"):
                a = getattr(P, name).copy()
                a[:, E.PRE + k + 1:] = a[:, E.PRE + k + 1:] * rng.uniform(0.2, 3.0, a[:, E.PRE + k + 1:].shape)
                arrs[name] = a
            T = P.T.copy()
            T[:, k + 1:] = rng.uniform(0.2, 3.0, T[:, k + 1:].shape)
            Q = E.Paths(P.keys, arrs["O"], arrs["H"], arrs["L"], arrs["C"], arrs["V"], T, P.pre_vol, P.vol_20d, P.end_date, P.scores)
            pd.testing.assert_frame_equal(E.state_at(Q, k), want, check_exact=True, obj=f"k={k}")
        # 逆に、k日目の値を変えると状態が変わる（確かめが効いていること）
        C = P.C.copy()
        C[:, E.PRE + 5] *= 1.01
        Q = E.Paths(P.keys, P.O, P.H, P.L, C, P.V, P.T, P.pre_vol, P.vol_20d, P.end_date, P.scores)
        self.assertFalse(E.state_at(Q, 5)["ret_now"].equals(E.state_at(P, 5)["ret_now"]))

    def test_values_by_hand(self):
        P, k, i = self.P, 6, 17
        s = E.state_at(P, k).iloc[i]
        j = E.PRE + k
        entry = P.O[i, E.PRE + 1]
        hi = P.H[i, E.PRE + 1:j + 1].max()
        self.assertAlmostEqual(s["ret_now"], P.C[i, j] / entry - 1)
        self.assertAlmostEqual(s["max_high"], hi / entry - 1)
        self.assertAlmostEqual(s["min_low"], P.L[i, E.PRE + 1:j + 1].min() / entry - 1)
        self.assertAlmostEqual(s["dd_peak"], P.C[i, j] / hi - 1)
        self.assertAlmostEqual(s["ret_1d"], P.C[i, j] / P.C[i, j - 1] - 1)
        self.assertAlmostEqual(s["ret_5d"], P.C[i, j] / P.C[i, j - 5] - 1)
        self.assertAlmostEqual(s["gap_today"], P.O[i, j] / P.C[i, j - 1] - 1)
        self.assertAlmostEqual(s["vol_ratio"], P.V[i, j] / P.pre_vol[i])
        self.assertAlmostEqual(s["vol_ratio3"], P.V[i, j - 2:j + 1].mean() / P.pre_vol[i])
        self.assertAlmostEqual(s["entry_gap"], entry / P.C[i, E.PRE] - 1)
        self.assertAlmostEqual(s["to_limit"], entry * 1.1 / P.C[i, j] - 1)
        dc = P.C[i, E.PRE + 1:j + 1] / P.C[i, E.PRE:j] - 1
        self.assertAlmostEqual(s["up_share"], (dc > 0).mean())
        self.assertAlmostEqual(s["rvol"], dc.std(ddof=1))
        self.assertEqual(s["days_since_peak"], k - 1 - P.H[i, E.PRE + 1:j + 1].argmax())
        self.assertAlmostEqual(s["topix_since"], P.T[i, k] / P.T[i, 0] - 1)
        self.assertAlmostEqual(s["excess"], s["ret_now"] - s["topix_since"])
        self.assertEqual(s["score_cat"], P.scores["score_cat"].iloc[i])
        self.assertTrue(np.isnan(E.state_at(P, 1)["rvol"]).all())             # 1日目は日々の変化が1つしかない
        self.assertEqual(E.state_at(P, 1)["new_high_today"].min(), 1.0)


def flat_paths(n=4, level=1000.0):
    """すべて同じ値（買値 1000）の値動き。テストで必要な日だけ書き換える。"""
    a = np.full((n, W), level)
    keys = pd.DataFrame({"Code": [f"{i}" for i in range(n)], "Date": pd.Timestamp("2024-01-04")})
    return E.Paths(keys, a.copy(), a.copy(), a.copy(), a.copy(), a.copy(), np.ones((n, E.HOLD + 1)), np.full(n, 1000.0),
                   np.full(n, 2.0), pd.Series([pd.Timestamp("2024-02-05")] * n))


class TestSimulate(unittest.TestCase):
    def test_conventions(self):
        P = flat_paths(4)
        d = lambda k: E.PRE + k  # noqa: E731
        P.H[0, d(3)] = 1100.0                           # 0: 3日目の高値がちょうど +10%
        P.C[1, d(E.HOLD)] = 1050.0                      # 1: 20日目の終値 +5%
        P.O[2, d(5)] = 1150.0                           # 2: 5日目の寄りが +15%（高値も）
        P.H[2, d(5)] = 1150.0
        P.O[3, d(2)] = 970.0                            # 3: 2日目の寄りが −3%
        hold = E.simulate(P, E.always_hold, "L")
        self.assertEqual(hold["ret"].round(9).tolist(), [10.0, 5.0, 10.0, 0.0])
        self.assertEqual(hold["how"].tolist(), ["limit", "end", "limit", "end"])
        self.assertEqual(hold["exit_day"].tolist(), [3, 20, 5, 20])
        self.assertEqual(E.simulate(P, E.always_hold, "N")["ret"].round(9).tolist(), [0.0, 5.0, 0.0, 0.0])
        sell1 = lambda k, rows: np.zeros(len(rows), dtype=bool)      # noqa: E731  1日目の引けで売ると決める
        s = E.simulate(P, sell1, "N")
        self.assertEqual(s["ret"].round(9).tolist(), [0.0, 0.0, 0.0, -3.0])
        self.assertEqual(s["exit_day"].tolist(), [2, 2, 2, 2])
        self.assertEqual(set(s["how"]), {"agent"})
        sell4 = lambda k, rows: np.full(len(rows), k < 4)            # noqa: E731  4日目の引けで売ると決める
        self.assertEqual(E.simulate(P, sell4, "N")["ret"].round(9).tolist(), [0.0, 0.0, 15.0, 0.0])
        s = E.simulate(P, sell4, "L")
        self.assertEqual(s["ret"].round(9).tolist(), [10.0, 0.0, 10.0, 0.0])     # 寄りが +15% でも +10% で数える
        self.assertEqual(s["how"].tolist(), ["limit", "agent", "agent", "agent"])  # 0 は 3日目の指値が先
        self.assertEqual(s["exit_day"].tolist(), [3, 5, 5, 5])
        with self.assertRaises(KeyError):
            E.simulate(P, E.always_hold, "X")
        with self.assertRaises(ValueError):
            E.simulate(P, lambda k, rows: np.ones(1, dtype=bool), "N")

    def test_oracle_bounds_every_policy(self):
        P, _ = synth_paths(800, seed=4, drift_sd=0.02, noise=0.03)
        rng = np.random.default_rng(1)
        self.assertGreater((E.simulate(P, E.always_hold, "L")["how"] == "limit").mean(), 0.1)
        for v in E.VARIANTS:
            top = E.oracle(P, v)
            for p in (0.0, 0.5, 0.9, 1.0):
                r = E.simulate(P, lambda k, rows: rng.random(len(rows)) < p, v)["ret"].to_numpy()
                self.assertTrue((r <= top + 1e-9).all(), (v, p))
        self.assertTrue((E.oracle(P, "L") <= E.LIMIT + 1e-9).all())
        ev = pd.DataFrame({"rule": [1.0, 2.0], "c20": [1.0, 5.0], "oracle_L": [1.0, 2.0], "oracle_N": [3.0, 4.0]})
        self.assertEqual(E.check_oracle(ev), 1)


class TestFit(unittest.TestCase):
    def test_learns_to_sell_falling_trades(self):
        train, _ = synth_paths(4000, seed=10)
        test, mu = synth_paths(2000, seed=11)
        for v in E.VARIANTS:
            ag = E.fit_agent(train, v, E.STATE_COLS, seed=42)
            self.assertEqual(sorted(ag.models), list(range(1, E.HOLD)))
            res = E.simulate(test, ag.hold_fn(test), v)
            base = E.simulate(test, E.always_hold, v)
            gain = res["ret"].mean() - base["ret"].mean()
            self.assertGreater(gain, 1.0, v)                                   # 下げ続ける取引を売って上回る
            self.assertTrue((res["ret"].to_numpy() <= E.oracle(test, v) + 1e-9).all())
            sold = (res["how"] == "agent").to_numpy()
            self.assertLess(mu[sold].mean(), mu[~sold].mean())                 # 売ったのは下げる向きの取引
            self.assertTrue(0.05 < sold.mean() < 0.95)
        # 学習は渡した取引だけ・決まった種なら同じ結果
        a = E.fit_agent(train.take(np.arange(1500)), "N", E.PRICE_COLS, seed=42)
        b = E.fit_agent(train.take(np.arange(1500)), "N", E.PRICE_COLS, seed=42)
        ra = E.simulate(test, a.hold_fn(test), "N")["ret"]
        self.assertTrue(ra.equals(E.simulate(test, b.hold_fn(test), "N")["ret"]))
        self.assertEqual(a.cols, E.PRICE_COLS)

    def test_too_few_rows_means_always_hold(self):
        P, _ = synth_paths(120, seed=3)
        for v in E.VARIANTS:
            ag = E.fit_agent(P, v)
            self.assertEqual(ag.models, {})
            base = E.simulate(P, E.always_hold, v)["ret"]
            self.assertTrue(E.simulate(P, ag.hold_fn(P), v)["ret"].equals(base))
            # 学習の中の成績: 売らない方針なら「いつも持ち続ける」と同じ値になる（数え方の確かめ）
            self.assertAlmostEqual(ag.train["value"], float(base.mean()), places=9)
            self.assertAlmostEqual(ag.train["base"], float(base.mean()), places=9)
            im = E.fit_improver(P, v)
            self.assertEqual(im.models, {})
            self.assertEqual(im.delta, float("inf"))
            self.assertTrue(E.simulate(P, im.hold_fn(P), v)["ret"].equals(base))

    def test_train_value_is_the_cross_fitted_policy_value(self):
        P, _ = synth_paths(2500, seed=12)
        ag = E.fit_agent(P, "N", E.PRICE_COLS)
        self.assertEqual(ag.train["n"], 2500)
        self.assertAlmostEqual(ag.train["base"], float(E.simulate(P, E.always_hold, "N")["ret"].mean()), places=9)
        self.assertGreater(ag.train["value"], ag.train["base"] + 0.5)        # 下げ続ける取引を売れるので、学習の中でも上回る
        self.assertTrue(0.0 < ag.train["hold_share"] < 1.0)


class TestImprover(unittest.TestCase):
    """追加の作り B（案）: 今のルールを土台にした方針の改善。売ると決める線を学習の取引の中で選ぶ。"""

    def test_learns_and_picks_a_line(self):
        train, _ = synth_paths(4000, seed=10)
        test, mu = synth_paths(2000, seed=11)
        for v in E.VARIANTS:
            im = E.fit_improver(train, v)
            self.assertEqual(sorted(im.models), list(range(1, E.HOLD)))
            t = im.train
            self.assertEqual([x["delta"] for x in t["table"]], [float(d) for d in E.DELTAS])
            base = float(E.simulate(train, E.always_hold, v)["ret"].mean())
            self.assertAlmostEqual(t["table"][0]["value"], base, places=9)    # 線 = 売らない → 今のルールと同じ
            self.assertEqual(t["table"][0]["sold"], 0.0)
            self.assertAlmostEqual(t["base"], base, places=9)
            self.assertEqual(t["value"], max(x["value"] for x in t["table"]))
            self.assertTrue(np.isfinite(im.delta))                            # 下げ続ける取引があるので、売る線が選ばれる
            self.assertGreater(t["value"], t["base"] + 0.5)
            res = E.simulate(test, im.hold_fn(test), v)
            gain = res["ret"].mean() - E.simulate(test, E.always_hold, v)["ret"].mean()
            self.assertGreater(gain, 1.0, v)
            sold = (res["how"] == "agent").to_numpy()
            self.assertLess(mu[sold].mean(), mu[~sold].mean())
            self.assertTrue((res["ret"].to_numpy() <= E.oracle(test, v) + 1e-9).all())
        again = E.fit_improver(train, "N")
        self.assertEqual(again.delta, im.delta)
        self.assertEqual(again.train["table"], im.train["table"])

    def test_keeps_the_current_rule_when_selling_only_hurts(self):
        """どの取引も上げ続けるなら、売らない（今のルールのまま）を選ぶ。学習の中の成績は今のルールを下回らない。"""
        rng = np.random.default_rng(7)
        P, _ = synth_paths(3000, seed=13, drift_sd=0.0, noise=0.01)
        up = np.exp(0.004 * np.arange(W))[None, :]                             # 毎日 +0.4% の上げを全取引に乗せる
        Q = E.Paths(P.keys, P.O * up, P.H * up, P.L * up, P.C * up, P.V, P.T, P.pre_vol, P.vol_20d, P.end_date, P.scores)
        test, _ = synth_paths(1500, seed=14, drift_sd=0.0, noise=0.01)
        T = E.Paths(test.keys, test.O * up, test.H * up, test.L * up, test.C * up, test.V, test.T, test.pre_vol, test.vol_20d,
                    test.end_date, test.scores)
        del rng
        im = E.fit_improver(Q, "N")
        self.assertGreaterEqual(im.train["value"], im.train["base"])
        self.assertLess(im.train["sold"], 0.05)
        res = E.simulate(T, im.hold_fn(T), "N")
        base = E.simulate(T, E.always_hold, "N")
        self.assertLess((res["how"] == "agent").mean(), 0.05)
        self.assertGreater(res["ret"].mean(), base["ret"].mean() - 0.1)
        # 線 = 売らない なら、予測によらずいつも持ち続ける
        im.delta = float("inf")
        self.assertTrue(E.simulate(T, im.hold_fn(T), "N")["ret"].equals(base["ret"]))

    def test_ties_keep_the_more_conservative_line(self):
        P, _ = synth_paths(120, seed=3)                                        # モデルを作れない → どの線でも同じ値
        im = E.fit_improver(P, "L")
        self.assertEqual(len({round(x["value"], 12) for x in im.train["table"]}), 1)
        self.assertEqual(im.delta, float("inf"))

    def test_fold_groups_keep_a_quarter_together(self):
        d = pd.to_datetime(pd.Series(["2024-01-05", "2024-03-29", "2024-04-01", "2025-04-01", "2025-07-01"]))
        g = E.fold_groups(d)
        self.assertEqual(g[0], g[1])
        self.assertNotEqual(g[1], g[2])
        self.assertEqual(int(g[3]), (int(g[2]) + 4) % E.N_FOLDS)
        self.assertTrue(((g >= 0) & (g < E.N_FOLDS)).all())


def eval_frame(shift, n_windows=6, per=40, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_windows + 1):
        dates = pd.Timestamp("2022-01-03") + pd.to_timedelta(w * 180 + rng.integers(0, 120, per), unit="D")
        base = rng.normal(2, 8, per)
        rows.append(pd.DataFrame({"window": w, "Date": dates, "rule": base, "agent": base + shift + rng.normal(0, noise, per)}))
    return pd.concat(rows, ignore_index=True)


class TestJudge(unittest.TestCase):
    def test_pass_and_fail(self):
        j = E.judge(eval_frame(1.0), "agent", "rule", n_perm=500)
        self.assertTrue(j["pass"])
        self.assertEqual((j["windows"], j["wins"], j["need"]), (6, 6, 4))
        self.assertLess(j["p"], 0.01)
        self.assertIn("通る", E.judge_text(j))
        j0 = E.judge(eval_frame(0.0, seed=1), "agent", "rule", n_perm=500)
        self.assertFalse(j0["pass"])
        self.assertGreater(j0["p"], 0.05)
        jn = E.judge(eval_frame(-1.0), "agent", "rule", n_perm=200)
        self.assertFalse(jn["positive"])
        self.assertFalse(jn["pass"])
        self.assertGreater(jn["p"], 0.9)

    def test_windows_need_five_rows_and_dates_flip_together(self):
        ev = eval_frame(1.0, n_windows=3, per=4)
        self.assertEqual(E.judge(ev, "agent", "rule", n_perm=50)["windows"], 0)
        self.assertIn("測れない", E.judge_text(E.judge(ev, "agent", "rule", n_perm=50)))
        self.assertEqual(len(E.window_diffs(ev, "agent", "rule", 1)), 3)
        # 同じ選定日に全部の行があるなら、符号の入れ替えは2通りしかない → p は 0.5 前後より下がらない
        one = eval_frame(1.0, n_windows=6, per=40)
        one["Date"] = pd.Timestamp("2024-01-04")
        self.assertGreater(E.judge(one, "agent", "rule", n_perm=400)["p"], 0.3)


class TestRunWindows(unittest.TestCase):
    def test_end_to_end(self):
        P, _ = synth_paths(2600, seed=20, days=1200)
        d = P.keys["Date"]
        folds = [types.SimpleNamespace(index=1, test_start="2021-01-01", test_end="2021-06-30"),
                 types.SimpleNamespace(index=2, test_start="2021-07-01", test_end="2021-12-31")]
        rng = np.random.default_rng(0)
        fl = P.keys[["Code", "Date"]].copy()
        fl["window"] = np.where(d < "2021-01-01", 0, np.where(d < "2021-07-01", 1, 2))
        fl = fl[fl["window"] > 0].reset_index(drop=True)
        fl["r0"] = rng.random(len(fl)) < 0.3
        fl["top10"] = fl["r0"] | (rng.random(len(fl)) < 0.2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ev, record = E.run_windows(P, fl, folds, with_ref=True)
            self.assertEqual(E.check_oracle(ev), 0)
            summary = E.report(ev, record, n_perm=100)
        out = buf.getvalue()
        self.assertEqual(len(ev), len(fl))
        self.assertEqual(sorted(record), [1, 2])
        for w, rec in record.items():
            self.assertLess(rec["train_to"], rec["test_from"])               # 学習は窓の始まりより前に終わった取引だけ
            self.assertEqual(rec["agent_N"]["models"], E.HOLD - 1)
        self.assertGreater(record[2]["train"], record[1]["train"])            # 学習の取引は窓が進むと増える
        for col, _ in E.POLICIES:
            if col.startswith("agentb"):
                continue                                                      # 追加の作り B は頼んだときだけ（下で確かめる）
            self.assertIn(col, ev.columns)
            self.assertFalse(ev[col].isna().any(), col)
        pos = fl.merge(P.keys[["Code", "Date"]].assign(_p=np.arange(len(P))), on=["Code", "Date"])
        want = E.simulate(P.take(pos["_p"].to_numpy()), E.always_hold, "L")["ret"]
        got = pos[["Code", "Date"]].merge(ev[["Code", "Date", "rule"]], on=["Code", "Date"])["rule"]
        self.assertTrue(np.allclose(got.to_numpy(), want.to_numpy()))         # 今のルール = いつも持ち続ける・指値あり
        self.assertEqual(set(summary["judge"]), {"r0", "top10", "all", "ref"})
        self.assertEqual(set(summary["judge"]["r0"]), {"L", "N"})
        self.assertGreater(summary["groups"]["all"]["agent_N"]["mean"], summary["groups"]["all"]["c20"]["mean"])
        self.assertIn("判定", out)
        for code in P.keys["Code"].head(300):
            self.assertNotIn(code, out)                                       # 銘柄コードを出さない
        self.assertNotIn("agentb_L", ev.columns)                              # 追加の作り B は頼んだときだけ
        self.assertIn("学習の中の確かめ", out)
        self.assertAlmostEqual(record[1]["agent_N"]["train"]["base"],
                               float(E.simulate(P.take(np.flatnonzero(P.end_date < np.datetime64("2021-01-01"))),
                                                E.always_hold, "N")["ret"].mean()), places=9)
        # 最後の n 窓だけ
        with redirect_stdout(io.StringIO()):
            ev1, rec1 = E.run_windows(P, fl, folds, with_ref=False, max_windows=1)
        self.assertEqual(sorted(rec1), [2])
        self.assertNotIn("agentp_L", ev1.columns)

    def test_with_b_and_train_only(self):
        P, _ = synth_paths(2200, seed=21, days=1200)
        d = P.keys["Date"]
        folds = [types.SimpleNamespace(index=1, test_start="2021-01-01", test_end="2021-06-30"),
                 types.SimpleNamespace(index=2, test_start="2021-07-01", test_end="2021-12-31")]
        rng = np.random.default_rng(1)
        fl = P.keys[["Code", "Date"]].copy()
        fl["window"] = np.where(d < "2021-01-01", 0, np.where(d < "2021-07-01", 1, 2))
        fl = fl[fl["window"] > 0].reset_index(drop=True)
        fl["r0"] = rng.random(len(fl)) < 0.3
        fl["top10"] = fl["r0"] | (rng.random(len(fl)) < 0.2)
        # 学習だけ: テスト窓の取引には当てない（評価の列が1つも無い）
        calls = []
        real = E.simulate

        def spy(Pp, fn, v):
            calls.append(len(Pp))
            return real(Pp, fn, v)
        E.simulate = spy
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ev0, rec0 = E.run_windows(P, fl, folds, with_ref=False, with_b=True, train_only=True)
                E.train_report(rec0)
        finally:
            E.simulate = real
        self.assertEqual(len(ev0), 0)
        self.assertEqual(sorted(rec0), [1, 2])
        self.assertTrue(set(calls) <= {rec0[1]["train"], rec0[2]["train"]})    # 売買したのは学習の取引だけ
        for w in (1, 2):
            self.assertIn("delta", rec0[w]["agentb_L"]["train"])
            self.assertGreaterEqual(rec0[w]["agentb_N"]["train"]["value"], rec0[w]["agentb_N"]["train"]["base"])
            self.assertIn("value", rec0[w]["agent_L"]["train"])
        out0 = buf.getvalue()
        self.assertIn("追加の作り B", out0)
        self.assertIn("選んだ線", out0)
        self.assertIn("テスト窓には当てない", out0)
        for code in P.keys["Code"].head(200):
            self.assertNotIn(code, out0)
        # B も評価する回: 列・判定・試運転で見た窓を除いた形
        old = E.TRIAL_WINDOWS
        E.TRIAL_WINDOWS = (2,)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                ev, record = E.run_windows(P, fl, folds, with_ref=False, with_b=True)
                self.assertEqual(E.check_oracle(ev), 0)
                summary = E.report(ev, record, n_perm=50)
        finally:
            E.TRIAL_WINDOWS = old
        out = buf.getvalue()
        for col in ("agentb_L", "agentb_N", "agentb_L_day", "agentb_N_how"):
            self.assertIn(col, ev.columns)
        self.assertEqual(set(summary["judge"]["r0"]), {"L", "N", "B_L", "B_N"})
        self.assertEqual(set(summary["judge"]["r0_untried"]), {"agent_L", "agent_N", "agentb_L", "agentb_N"})
        self.assertEqual(summary["judge"]["r0_untried"]["agentb_N"]["windows"], 1)   # 窓2を除いた1窓
        self.assertIn("試運転で見た窓", out)
        self.assertGreater(summary["groups"]["all"]["agentb_N"]["mean"], summary["groups"]["all"]["c20"]["mean"])
        self.assertEqual(rec0[1]["agentb_N"]["train"]["delta"], record[1]["agentb_N"]["train"]["delta"])


if __name__ == "__main__":
    unittest.main()
