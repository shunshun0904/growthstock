#!/usr/bin/env python3
"""
実験133（research/exp/e133_train_years.py）の部品。データには触らない（作りものの表で確かめる）。

- 決めたこと: 年数 2〜6年・ずらし 0/2/4か月・種3つ・判定は L6−L4・土台は回した時点の既定 373列（名前で固定）
- 窓: 本番と同じ作りのうち、train_end − 6年 がデータの最初の日以降の窓だけ
- 学習の行: (train_end − L 年, train_end]。ALL は最初から train_end まで。短い年数 ⊂ 長い年数 ⊂ ALL
- 判定（厳しめ）: ① 平均が正 ② 上の窓が過半（同じ値は上に数えない）③ 平均が 2SE を超える。3つとも満たすときだけ「見込みあり」
- 参考の傾き: log2(年数) に当てた傾きを窓ごとに出して平均する
- 一通り: 作りものの表で、腕ごとの out-of-fold・テストの行が腕どうしで同じ・報告が最後まで通る

  python3 -m unittest tests.test_e133_train_years
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e133_train_years as E  # noqa: E402
import features as F  # noqa: E402


class TestDecided(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(E.YEARS, (2, 3, 4, 5, 6))
        self.assertEqual(E.SHIFTS, (0, 2, 4))
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.JUDGE, ("L4", "L6"))
        self.assertEqual(E.BASE_PRESET, F.DEFAULT_PRESET)               # 回す時点の既定（変わったら見直す）
        self.assertEqual(len(F.columns(E.BASE_PRESET)), 373)
        self.assertEqual(E.ALGO, "lgbm")

    def test_arm_names(self):
        self.assertEqual(E.arm_name(3), "L3")
        self.assertEqual(E.arm_name(None), "ALL")
        self.assertEqual(E.arm_years("L6"), 6)
        self.assertIsNone(E.arm_years("ALL"))


class TestWindows(unittest.TestCase):
    def setUp(self):
        self.dates = pd.Series(pd.bdate_range("2018-04-03", "2026-10-26"))

    def test_only_windows_with_enough_history(self):
        first = self.dates.min()
        n_all = 0
        for sh in (0, 2, 4):
            all_f = E.E41.folds_for(self.dates, sh)
            el = E.eligible_folds(self.dates, sh, 6)
            self.assertTrue(el)
            n_all += len(el)
            for f in all_f:
                ok = pd.Timestamp(f.train_end) - pd.DateOffset(years=6) >= first
                self.assertEqual(ok, f in el)
            # テストは 2024年5月以降
            self.assertGreaterEqual(min(pd.Timestamp(f.test_start) for f in el), pd.Timestamp("2024-04-30"))
        self.assertEqual(n_all, 15)                                     # 運用者の選択「2〜6年・15窓」

    def test_train_mask(self):
        d = self.dates
        end = "2024-04-03"
        m2 = E.train_mask(d, end, 2)
        self.assertEqual(d[m2].min(), pd.Timestamp("2022-04-04"))       # 2022-04-03（日曜）の翌営業日
        self.assertEqual(d[m2].max(), pd.Timestamp(end))
        self.assertFalse(m2[d == pd.Timestamp("2022-04-01")].any())
        m_all = E.train_mask(d, end, None)
        self.assertEqual(d[m_all].min(), d.min())
        self.assertEqual(d[m_all].max(), pd.Timestamp(end))
        # 入れ子: 2年 ⊂ 4年 ⊂ 6年 ⊂ 全部
        m4, m6 = E.train_mask(d, end, 4), E.train_mask(d, end, 6)
        self.assertTrue((~m2 | m4).all() and (~m4 | m6).all() and (~m6 | m_all).all())


class TestJudge(unittest.TestCase):
    def test_pass_needs_all_three(self):
        j = E.judge_strict(np.array([0.010, 0.012, 0.008, 0.011, 0.009]))
        self.assertTrue(j["positive"] and j["majority"] and j["beyond_2se"] and j["pass"])
        # 平均は正・過半も上だが、ばらつきが大きく 2SE を超えない
        j = E.judge_strict(np.array([0.030, 0.001, 0.002, -0.020, 0.001]))
        self.assertTrue(j["positive"] and j["majority"])
        self.assertFalse(j["beyond_2se"])
        self.assertFalse(j["pass"])
        # 平均は 2SE を超えるが、上の窓が過半でない（同じ値は上に数えない）
        j = E.judge_strict(np.array([0.05, 0.05, 0.0, 0.0, 0.0]))
        self.assertEqual(j["wins"], 2)
        self.assertEqual(j["need"], 3)
        self.assertFalse(j["majority"])
        self.assertFalse(j["pass"])
        j = E.judge_strict(np.array([-0.01, -0.02, -0.01]))
        self.assertFalse(j["positive"] or j["pass"])

    def test_slope(self):
        arms = ["L2", "L3", "L4", "L5", "L6", "ALL"]
        rows = []
        for b in (0.01, 0.02):
            r = {f"pr_L{y}": 0.2 + b * np.log2(y) for y in (2, 3, 4, 5, 6)}
            r["pr_ALL"] = 0.5                                            # ALL は傾きに入れない
            rows.append(r)
        sl = E.slope_per_doubling(pd.DataFrame(rows), arms)
        self.assertEqual(sl["n"], 2)
        self.assertAlmostEqual(sl["mean"], 0.015, places=10)
        self.assertEqual(sl["positive_windows"], 2)


def make_frame(seed=0, per_day=3):
    """2015〜2023年。新しい年ほど関係が強まるように作る（どの腕も最後まで通ることだけを見る）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", "2023-12-29")
    n = len(dates) * per_day
    df = pd.DataFrame({"Code": [f"{1000 + i % per_day}" for i in range(n)], "Date": np.repeat(dates, per_day)})
    for j in range(4):
        df[f"b{j}"] = rng.normal(size=n)
    df["label"] = (df["b1"] + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["ret_o1_20"] = rng.normal(0.01, 0.1, n)
    df["ret_o1_40"] = rng.normal(0.01, 0.15, n)
    df["hit10"] = (rng.random(n) < 0.3).astype(float)
    df["tp10c"] = np.where(df["hit10"] > 0, 10.0, rng.normal(-1, 8, n))
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


SMALL = {"objective": "binary", "n_estimators": 8, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20,
         "verbose": -1, "n_jobs": 1, "random_state": 0}


class TestPlan(unittest.TestCase):
    def test_nested_and_minimums(self):
        df = make_frame(0)
        arms = ["L2", "L4", "ALL"]
        p = E.plan(df, arms, [0, 2], 4)
        self.assertTrue(len(p))
        with redirect_stdout(io.StringIO()):
            out = E.check_plan(p, arms)
        self.assertEqual(out["not_nested"], 0)
        self.assertEqual(out["too_small_train"], 0)
        # 最後の窓はテストが200行に満たず使わない（本番の out-of-fold と同じ下限）
        self.assertEqual(out["dropped_small_test"], int((p["n_test"] < E.MIN_TEST_ROWS).sum()))
        self.assertEqual(out["windows"], int(p["use"].sum()))
        self.assertTrue((p["n_L2"] <= p["n_L4"]).all() and (p["n_L4"] <= p["n_ALL"]).all())
        bad = p.copy()
        bad.loc[bad.index[0], "n_L2"] = 10
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            E.check_plan(bad, arms)


class TestRunAndReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_end_to_end(self):
        df = make_frame(1)
        cols = ["b0", "b1", "b2", "b3"]
        arms = ["L2", "L3", "L4", "ALL"]
        shifts, seeds = [0, 2], [42]
        stamp = {"rows": len(df), "key": E.data_key(df)}
        with mock.patch.object(E, "OOF_DIR", self.tmp), mock.patch.object(E, "TAG", "t133"), \
                mock.patch.object(E, "JUDGE", ("L2", "L4")):
            p = E.plan(df, arms, shifts, 4)
            buf = io.StringIO()
            with redirect_stdout(buf):
                E.check_plan(p, arms)
                res = E.run(df, cols, SMALL, arms, shifts, seeds, 4, stamp["key"], p)
                s = E.report(res, arms, shifts, seeds, stamp, p, SMALL)
            out = buf.getvalue()
            self.assertIn("テストの行が腕どうしで同じ → 同じ", out)
            self.assertIn("■ 3. 判定", out)
            self.assertIn("■ 6. 窓ごとの値", out)
            self.assertEqual(set(s["windows"]) >= {"L2", "L3", "L4", "ALL"}, True)
            self.assertIn("pass", s["judge"])
            self.assertEqual(len(s["per_window"]), int(p["use"].sum()))
            with open(os.path.join(self.tmp, "t133_summary.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["arms"], arms)
            # 2回目は保存した out-of-fold を読む（学習しない）
            with redirect_stdout(io.StringIO()), mock.patch.object(E.E41, "oof_folds",
                                                                   side_effect=RuntimeError("trained again")):
                res2 = E.run(df, cols, SMALL, arms, shifts, seeds, 4, stamp["key"], p)
            for k in res["oofs"]:
                np.testing.assert_allclose(res["oofs"][k]["score"].to_numpy(), res2["oofs"][k]["score"].to_numpy())

    def test_training_rows_follow_the_arm(self):
        df = make_frame(2)
        cols = ["b0", "b1", "b2", "b3"]
        f = E.eligible_folds(df["Date"], 0, 4)[0]
        seen = {}

        def spy(algo, d, c, params, seed, folds, train_mask=None):
            m = pd.Series(train_mask, index=d.index) & d["label"].notna()
            seen[len(seen)] = (d.loc[m & (d["Date"] <= pd.Timestamp(f.train_end)), "Date"].min(),
                               d.loc[m, "Date"].max())
            return pd.DataFrame({"Code": ["x"], "Date": [pd.Timestamp(f.test_start)], "label": [1.0], "score": [0.5],
                                 "fold": [f.index]})

        with mock.patch.object(E.E41, "oof_folds", side_effect=spy):
            E.oof_arm(df, cols, SMALL, [f], "L2", 42)
            E.oof_arm(df, cols, SMALL, [f], "ALL", 42)
        lo2, hi2 = seen[0]
        lo_all, hi_all = seen[1]
        self.assertEqual(hi2, df.loc[df["Date"] <= pd.Timestamp(f.train_end), "Date"].max())  # train_end 以前の最後の日
        self.assertGreater(lo2, pd.Timestamp(f.train_end) - pd.DateOffset(years=2))
        self.assertEqual(lo_all, df["Date"].min())


if __name__ == "__main__":
    unittest.main()
