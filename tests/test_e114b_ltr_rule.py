#!/usr/bin/env python3
"""
実験114b（research/exp/e114b_ltr_rule.py）。合成データで:

- 本番の3本の木の形: 学習する列で探索したものでなければ止める（lgbm / xgb・cat それぞれ）
- 4本の out-of-fold の行がそろわなければ止める。前向きの百分位は最初の2窓を外し、0〜100 に入る
- 規則の印: R0 = 3本すべて95以上（百分位が無い行は満たさない）、L95、R0L、R0D（その日の中でランキング学習が上半分）、
  LN（窓ごとに R0 と同じ件数をランキング学習の上から）
- 日付をまたいだ力: 日ごとの平均百分位と日ごとの平均収益の順位相関が、日の水準を持つモデルで高く出る
- 3枠の形: 今の並び（3本の最小・同点は基準モデルのスコア）とランキング学習の並び、上位2件まで、枠が埋まれば見送る
- 本番の out-of-fold との突き合わせ・判定（すべてのずらし方で R0D（既定）が上かつ大負けが少ない。LN でも同じ判定ができる）
- 本番の3本の out-of-fold は、本番と同じ行（ラベルがある行）で学習し、実験114 と同じ行を採点する

  python3 tests/test_e114b_ltr_rule.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import walkforward as WF  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e114b_ltr_rule as RB  # noqa: E402

SMALL_LGBM = {"objective": "binary", "boosting_type": "gbdt", "n_estimators": 30, "learning_rate": 0.1,
              "num_leaves": 7, "min_child_samples": 10, "subsample": 1.0, "subsample_freq": 1,
              "colsample_bytree": 1.0, "random_state": 0, "n_jobs": 1, "verbose": -1}
FEATS = ["x0", "x1", "x2", "x3"]


class TestProdParams(unittest.TestCase):
    def test_signature_required(self):
        cols = ["a", "b", "c"]
        sig = F.signature(cols)
        good_l = {F.DEFAULT_PRESET: {"_features_sig": sig, "_n_features": 3, "learning_rate": 0.05}}
        bad_l = {F.DEFAULT_PRESET: {"_features_sig": "other", "_n_features": 3}}
        good_m = {a: {"params": {"learning_rate": 0.1}, "_cv": {"features_sig": sig, "n_features": 3}} for a in ("xgb", "cat")}
        bad_m = {**good_m, "cat": {"params": {}, "_cv": {"features_sig": "other", "n_features": 239}}}
        with self.assertRaises(SystemExit):
            RB.prod_params(cols, lgbm_store=bad_l, multi_store=good_m)
        with self.assertRaises(SystemExit):
            RB.prod_params(cols, lgbm_store={}, multi_store=good_m)
        with self.assertRaises(SystemExit):
            RB.prod_params(cols, lgbm_store=good_l, multi_store=bad_m)
        p = RB.prod_params(cols, lgbm_store=good_l, multi_store=good_m)
        self.assertEqual(p["lgbm"]["learning_rate"], 0.05)
        self.assertNotIn("_features_sig", p["lgbm"])
        self.assertEqual(p["xgb"], {"learning_rate": 0.1})


def fold(i, start, end):
    return WF.Fold(index=i, train_start="2015-01-01", train_end=str(pd.Timestamp(start) - pd.Timedelta(days=40)),
                   test_start=start, test_end=end)


def synth_wide(seed=0, per_day=6):
    """5つのテスト窓（各窓 60営業日）の4本のスコアと収益。窓の間は十分に離す。"""
    rng = np.random.default_rng(seed)
    folds, rows = [], []
    start = pd.Timestamp("2021-01-04")
    for i in range(1, 6):
        days = pd.bdate_range(start, periods=60)
        folds.append(fold(i, str(days[0].date()), str(days[-1].date())))
        for d in days:
            for j in range(per_day):
                rows.append({"Code": f"{1000 + j}", "Date": d, "window": i})
        start = days[-1] + pd.Timedelta(days=60)
    w = pd.DataFrame(rows)
    n = len(w)
    z = rng.normal(size=n)
    for a in RB.MODELS4:
        w[f"score_{a}"] = z + rng.normal(0, 0.5, n)
    w[E.RET] = np.clip(0.03 * z + rng.normal(0, 0.05, n), -0.3, 0.10)
    w[E.HIT] = w[E.RET] >= 0.10
    w[E.HITDAY] = np.where(w[E.HIT], 3.0, np.nan)
    w[E.R20] = 0.03 * z + rng.normal(0, 0.05, n)
    w["label"] = (w[E.R20] > 0.03).astype(float)
    w["vol_20d"] = 1.5
    return w, folds


class TestForwardAndRules(unittest.TestCase):
    def setUp(self):
        self.w, self.folds = synth_wide()
        self.ft = RB.forward_frame(self.w, self.folds)

    def test_forward_excludes_first_windows(self):
        self.assertEqual(sorted(self.ft["window"].unique()), [3, 4, 5])
        for a in RB.MODELS4:
            v = self.ft[f"pct_{a}"].to_numpy()
            self.assertTrue(((v >= 0) & (v <= 100)).all())
            # 画面と同じ小数1桁
            self.assertTrue(np.allclose(v, np.round(v, 1)))
        self.assertTrue((self.ft["n_break"] == 6).all())

    def test_masks(self):
        m = RB.rule_masks(self.ft)
        t3 = (self.ft[[f"pct_{a}" for a in RB.TREES]] >= 95).all(axis=1).to_numpy()
        np.testing.assert_array_equal(m["R0"], t3)
        np.testing.assert_array_equal(m["L95"], self.ft["pct_ltr"].to_numpy() >= 95)
        np.testing.assert_array_equal(m["R0L"], t3 & (self.ft["pct_ltr"].to_numpy() >= 50))
        for ww, g in self.ft.groupby("window"):
            n = int(m["R0"][g.index].sum())
            self.assertEqual(int(m["LN"][g.index].sum()), n)
            if n:
                top = g.sort_values("score_ltr", ascending=False).index[:n]
                self.assertTrue(m["LN"][top].all())

    def test_r0d_uses_rank_within_day(self):
        m = RB.rule_masks(self.ft)
        dp = self.ft["day_pct_ltr"].to_numpy()
        np.testing.assert_array_equal(m["R0D"], m["R0"] & (dp >= 0.5))
        for _, g in self.ft.groupby("Date"):
            best = g["score_ltr"].idxmax()
            worst = g["score_ltr"].idxmin()
            self.assertAlmostEqual(self.ft.loc[best, "day_pct_ltr"], 1.0)
            self.assertAlmostEqual(self.ft.loc[worst, "day_pct_ltr"], 0.0)

    def test_nan_percentile_does_not_pass(self):
        ft = self.ft.copy()
        ft.loc[0, "pct_xgb"] = np.nan
        ft.loc[0, ["pct_lgbm", "pct_cat"]] = 99.0
        ft["min_t3"] = ft[[f"pct_{a}" for a in RB.TREES]].min(axis=1, skipna=False)
        self.assertFalse(RB.rule_masks(ft)["R0"][0])

    def test_stats_and_diff(self):
        m = RB.rule_masks(self.ft)
        s = RB.rule_stats(self.ft, m["R0"])
        sel = self.ft[m["R0"]]
        self.assertEqual(s["n"], len(sel))
        self.assertAlmostEqual(s["ret"], float(sel[E.RET].mean()))
        self.assertAlmostEqual(s["ret_w"], float(sel.groupby("window")[E.RET].mean().mean()))
        d = RB.window_diff(self.ft, m["R0"], m["R0"])
        self.assertAlmostEqual(d["mean"], 0.0)
        self.assertEqual(d["won"], 0)


class TestSlots(unittest.TestCase):
    def frame(self):
        days = pd.bdate_range("2024-01-08", periods=4)
        rows = []
        for i, d in enumerate(days):
            for j in range(3):
                rows.append({"Code": f"{2000 + j}", "Date": d, "window": 3, "min_t3": [96.0, 99.0, 97.0][j],
                             "score_lgbm": 0.1 * j, "score_ltr": [3.0, 1.0, 2.0][j], E.RET: 0.01 * (j + 1),
                             E.HIT: False, E.HITDAY: np.nan, "n_break": 3})
        return pd.DataFrame(rows), days

    def test_orders(self):
        ft, _ = self.frame()
        mask = np.ones(len(ft), dtype=bool)
        cm = RB.ordered_candidates(ft, mask, "min")
        cl = RB.ordered_candidates(ft, mask, "ltr")
        first = ft["Date"].min()
        self.assertEqual(list(cm[cm["Date"] == first]["Code"]), ["2001", "2002"])
        self.assertEqual(list(cl[cl["Date"] == first]["Code"]), ["2000", "2002"])
        self.assertEqual(len(RB.ordered_candidates(ft, mask, "min", min_break=8)), 0)

    def test_slots_fill_and_skip(self):
        ft, days = self.frame()
        bar_days = list(pd.bdate_range(days[0], periods=40))
        c = RB.ordered_candidates(ft, np.ones(len(ft), dtype=bool), "min")
        t = RB.slot_trades(c, bar_days, slots=3)
        # 届かなければ20営業日ふさぐので、最初の日の2件と次の日の1件で3枠が埋まり、あとは見送る
        self.assertEqual(len(t), 3)
        s = RB.slot_stats(t, months=1, slots=3)
        self.assertAlmostEqual(s["capital_pm"], float(t[E.RET].sum() * 100 / 3))


class TestChecks(unittest.TestCase):
    def test_prod_check(self):
        o = pd.DataFrame({"Code": ["1", "2", "3"], "Date": pd.to_datetime(["2024-01-04"] * 3), "score": [0.1, 0.2, 0.3]})
        with tempfile.TemporaryDirectory() as tmp:
            o.to_parquet(os.path.join(tmp, RB.PROD_OOF["xgb"]), index=False)
            c = RB.prod_check(o, "xgb", data_dir=tmp)
            self.assertEqual((c["matched"], c["max_diff"], c["same_share"]), (3, 0.0, 1.0))
            o2 = o.assign(score=[0.1, 0.25, 0.3])
            c2 = RB.prod_check(o2, "xgb", data_dir=tmp)
            self.assertAlmostEqual(c2["same_share"], 2 / 3)
            self.assertFalse(RB.prod_check(o, "cat", data_dir=tmp)["found"])

    def test_wide_requires_same_rows(self):
        w, _ = synth_wide()
        base = w[["Code", "Date", "window"]].rename(columns={"window": "fold"})
        parts = {a: base.assign(score=w[f"score_{a}"]) for a in RB.MODELS4}
        df = w.drop(columns=[f"score_{a}" for a in RB.MODELS4] + ["window"])
        out = RB.wide(parts, df)
        self.assertEqual(len(out), len(w))
        parts["cat"] = parts["cat"].iloc[1:]
        with self.assertRaises(SystemExit):
            RB.wide(parts, df)

    def test_verdict(self):
        good = {"R0": {"ret_w": 0.01, "bad_w": 0.10}, "LN": {"ret_w": 0.02, "bad_w": 0.08},
                "R0D": {"ret_w": 0.015, "bad_w": 0.09}}
        bad = {"R0": {"ret_w": 0.02, "bad_w": 0.10}, "LN": {"ret_w": 0.01, "bad_w": 0.08},
               "R0D": {"ret_w": 0.03, "bad_w": 0.10}}
        self.assertEqual(RB.PRIMARY, "R0D")
        self.assertTrue(RB.verdict([{"shift": s, **good} for s in (0, 2, 4)])["pass"])
        self.assertTrue(RB.verdict([{"shift": s, **good} for s in (0, 2, 4)], "LN")["pass"])
        # R0D は大負けの割合が同じ（0.10）なので満たさない。LN は収益が下なので満たさない
        self.assertFalse(RB.verdict([{"shift": 0, **good}, {"shift": 2, **bad}])["pass"])
        self.assertFalse(RB.verdict([{"shift": 0, **good}, {"shift": 2, **bad}], "LN")["pass"])
        self.assertFalse(RB.verdict([])["pass"])

    def test_cross_date(self):
        rng = np.random.default_rng(1)
        days = pd.bdate_range("2023-01-02", periods=200)
        rows = []
        for d in days:
            lvl = rng.normal()                       # その日の地合い（日の水準）
            for j in range(6):
                rows.append({"Date": d, "Code": f"{3000 + j}", "pct_lgbm": 50 + 15 * lvl + rng.normal(0, 5),
                             "pct_xgb": rng.uniform(0, 100), "pct_cat": rng.uniform(0, 100),
                             "pct_ltr": rng.uniform(0, 100), E.RET: 0.02 * lvl + rng.normal(0, 0.03),
                             E.R20: 0.02 * lvl + rng.normal(0, 0.03)})
        ft = pd.DataFrame(rows)
        cd = RB.cross_date(ft)
        self.assertEqual(cd["lgbm"]["days"], 200)
        self.assertGreater(cd["lgbm"]["rho"], 0.5)
        self.assertLess(abs(cd["ltr"]["rho"]), 0.3)
        self.assertGreater(cd["lgbm"]["top_days"], cd["lgbm"]["bottom_days"])
        # 候補が少ない日は数えない
        self.assertEqual(RB.cross_date(ft[ft["Code"].isin(["3000", "3001"])]), {})


class TestOofTree(unittest.TestCase):
    def test_rows_match_experiment_114(self):
        rng = np.random.default_rng(0)
        days = pd.bdate_range("2018-04-02", "2023-06-30")
        rows = [{"Code": f"{1000 + j}", "Date": d} for d in days for j in range(3)]
        df = pd.DataFrame(rows)
        n = len(df)
        for c in FEATS:
            df[c] = rng.normal(size=n)
        df[E.R20] = 0.04 * df["x0"] + rng.normal(0, 0.03, n)
        df[E.RET] = np.minimum(df[E.R20], 0.10)
        df.loc[df.index[::97], E.R20] = np.nan        # 20営業日目の終値が無い行は採点しない
        df[E.HIT] = df[E.RET] >= 0.10
        df["label"] = (df["x0"] + rng.normal(0, 1, n) > 0.8).astype(float)
        df[E.OUTCOME] = df[E.RET]
        df["vol_20d"] = 1.5
        with tempfile.TemporaryDirectory() as tmp:
            o = RB.oof_tree(df, FEATS, "lgbm", SMALL_LGBM, 0, out_dir=tmp)
            ol = E.oof_arm(df, FEATS, "L", {k: v for k, v in SMALL_LGBM.items() if k != "objective"}, 0, (42,),
                           out_dir=tmp)
            a = o.sort_values(["Date", "Code"])[["Code", "Date", "fold"]].astype(str).to_numpy()
            b = ol.sort_values(["Date", "Code"])[["Code", "Date", "fold"]].astype(str).to_numpy()
            np.testing.assert_array_equal(a, b)
            self.assertTrue(((o["score"] >= 0) & (o["score"] <= 1)).all())
            again = RB.oof_tree(df, FEATS, "lgbm", SMALL_LGBM, 0, out_dir=tmp)
            np.testing.assert_array_equal(again["score"].to_numpy(), o["score"].to_numpy())


if __name__ == "__main__":
    unittest.main()
