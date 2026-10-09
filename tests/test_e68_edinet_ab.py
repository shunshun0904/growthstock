#!/usr/bin/env python3
"""
実験68（research/exp/e68_edinet_ab.py）の、重い計算を回さない部分のテスト。

- 足す列の組（all / core / core+mcap / unique）の大きさと、本番の239列と重ならないこと
- §7（2026-09-26 改訂）の判定: 平均 C−A が正・対照 P−A を上回る・上の窓が過半
- 行の選び方（covered は y0 の付いた行だけ）と評価の行の組
- 時間の上限を過ぎたら、探索も out-of-fold も新しい計算を始めない（ファイルも作らない）

  python3 tests/test_e68_edinet_ab.py
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import edinet_features as EF  # noqa: E402
import features as F  # noqa: E402
import e68_edinet_ab as E  # noqa: E402


class TestColumnSets(unittest.TestCase):
    def test_sizes_and_prefix(self):
        sizes = {name: len(E.column_set(name)) for name in E.SET_NAMES}
        self.assertEqual(sizes["all"], len(EF.columns("all")))
        self.assertEqual(sizes["core"], 40)
        self.assertEqual(sizes["core+mcap"], 47)
        self.assertEqual(sizes["unique"], sizes["all"] - sizes["core"])
        for name in E.SET_NAMES:
            cols = E.column_set(name)
            self.assertTrue(all(c.startswith("ed_") for c in cols))
            self.assertEqual(len(cols), len(set(cols)))                     # 重複なし

    def test_unique_excludes_core_and_keeps_the_rest(self):
        core, uniq = set(E.column_set("core")), set(E.column_set("unique"))
        self.assertFalse(core & uniq)
        self.assertEqual(core | uniq, set(E.column_set("all")))

    def test_no_overlap_with_production_columns(self):
        base = set(F.columns(F.DEFAULT_PRESET))
        self.assertEqual(len(base), 239)
        self.assertFalse(base & set(E.column_set("all")))

    def test_unknown_name(self):
        with self.assertRaises(KeyError):
            E.column_set("screened")


class TestLineup(unittest.TestCase):
    def test_logit_is_compared_but_not_a_vote(self):
        self.assertEqual(E.ALGOS, ("lgbm", "xgb", "cat", "logit"))       # 2026-10-09 夜: 実験にもロジスティックを含める
        self.assertEqual(E.JUDGE_ALGOS, ("lgbm", "xgb", "cat"))          # §7 の票は木3つ
        self.assertEqual(set(E.SEEDS), set(E.ALGOS))
        self.assertEqual(E.SEEDS["logit"], (42,))                        # 決定的なので種1つ
        self.assertTrue(all(len(E.SEEDS[a]) == 3 for a in E.JUDGE_ALGOS))


class TestJudge7(unittest.TestCase):
    def test_pass_when_positive_beats_placebo_and_majority(self):
        d_ca = np.array([0.01] * 20 + [-0.005] * 12)          # 20/32 が上
        d_pa = np.array([0.001] * 32)
        j = E.judge7(d_ca, d_pa)
        self.assertEqual((j["n"], j["wins"], j["need"]), (32, 20, 17))
        self.assertTrue(j["positive"] and j["beats_placebo"] and j["majority"] and j["pass"])

    def test_fail_when_placebo_is_as_good(self):
        d_ca = np.array([0.01] * 20 + [-0.005] * 12)
        d_pa = np.array([0.02] * 32)                            # 対照のほうが上
        j = E.judge7(d_ca, d_pa)
        self.assertTrue(j["positive"] and j["majority"])
        self.assertFalse(j["beats_placebo"] or j["pass"])

    def test_fail_without_majority(self):
        d_ca = np.array([0.10] * 16 + [-0.001] * 16)          # 平均は正だが 16/32 は過半でない
        j = E.judge7(d_ca, np.zeros(32))
        self.assertTrue(j["positive"] and j["beats_placebo"])
        self.assertFalse(j["majority"] or j["pass"])
        self.assertEqual(E.judge7(np.array([0.1] * 6 + [-0.1] * 4), np.zeros(10))["need"], 6)

    def test_empty(self):
        j = E.judge7(np.array([]), np.array([]))
        self.assertEqual(j["n"], 0)
        self.assertFalse(j["pass"])


class TestRows(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-04", periods=6)
        self.df = pd.DataFrame({"Code": ["10000", "20000", "30000"] * 2, "Date": list(dates[:3]) * 2,
                                "ed_fiscal_year": [2023.0, np.nan, 2023.0, np.nan, 2023.0, 2023.0],
                                "label": [1, 0, 1, 0, 0, 1]})

    def test_covered_keeps_only_rows_with_y0(self):
        sub = E.select_rows(self.df, "covered")
        self.assertEqual(len(sub), 4)
        self.assertTrue(sub["ed_fiscal_year"].notna().all())
        self.assertEqual(list(sub.index), [0, 1, 2, 3])
        self.assertEqual(len(E.select_rows(self.df, "all")), 6)
        with self.assertRaises(KeyError):
            E.select_rows(self.df, "half")

    def test_eval_keysets(self):
        ks = E.eval_keysets(self.df, "all")
        self.assertEqual(list(ks), ["all", "covered"])
        self.assertIsNone(ks["all"])
        self.assertEqual(len(ks["covered"]), 4)
        ks = E.eval_keysets(E.select_rows(self.df, "covered"), "covered")
        self.assertEqual(list(ks), ["covered"])
        self.assertIsNone(ks["covered"])


class TestTimeBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT)
        E.OOF_DIR, E.RUN = self.dir, "e68test_covered_all"

    def tearDown(self):
        E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_oof_and_tune_do_nothing_after_the_deadline(self):
        frame = pd.DataFrame({"Code": ["10000"] * 3, "Date": pd.bdate_range("2024-01-04", periods=3),
                              "label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        frame.attrs["built_utc"] = "2026-10-09 00:00 UTC"
        E.DEADLINE, E.TIMED_OUT = time.time() - 1, False
        self.assertIsNone(E.oof("lgbm", "ed", frame, ["x"], {"n_estimators": 1}, 0, (42,)))
        self.assertIsNone(E.tune("lgbm", "A", frame, ["x"], pd.Timestamp("2024-01-10"), 2,
                                 {"built_utc": "2026-10-09 00:00 UTC"}))
        self.assertTrue(E.TIMED_OUT)
        self.assertEqual(os.listdir(self.dir), [])                           # 何も保存しない
        E.DEADLINE = None
        self.assertTrue(E.time_left())                                       # 無制限

    def test_paths_use_run_and_tag(self):
        self.assertTrue(E.path("summary.json").endswith("e68test_covered_all_summary.json"))
        self.assertTrue(E.fpath("frame.parquet").endswith(f"{E.TAG}_frame.parquet"))


if __name__ == "__main__":
    unittest.main()
