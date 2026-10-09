#!/usr/bin/env python3
"""
実験66（research/exp/e66_nonlinear_screen.py）の部品のテスト。

  python3 tests/test_e66_nonlinear_screen.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402


class TestArms(unittest.TestCase):
    def test_v_is_t_plus_32_new_columns(self):
        base = F.columns(F.DEFAULT_PRESET)
        v = base + E66.NEW_COLS
        self.assertEqual(len(v), len(set(v)), "重複する列がある")
        self.assertEqual(len(E66.NEW_COLS), 32)
        self.assertFalse(set(E66.NEW_COLS) & set(F.all_columns()),
                         "候補の列は本番の設定に入れていない（採否が決まるまで）")

    def test_reference_columns_are_production_columns(self):
        prod = set(F.columns(F.DEFAULT_PRESET))
        for c in E66.REF_COLS:
            self.assertIn(c, prod, c)

    def test_labels_cover_arms(self):
        self.assertEqual(set(E66.LABELS), {"T", "V", "P"})


class TestWithinWindowPct(unittest.TestCase):
    def test_rank_is_within_window(self):
        d = pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-20", "2024-07-05", "2024-07-10"])
        df = pd.DataFrame({"Date": d, "x": [3.0, 1.0, 2.0, 10.0, np.nan]})
        windows = [(np.datetime64("2024-01-01"), np.datetime64("2024-06-30")),
                   (np.datetime64("2024-07-01"), np.datetime64("2024-12-31"))]
        p = E66.within_window_pct(df, "x", windows)
        # 1窓目: 1.0 < 2.0 < 3.0 → 1/3, 2/3, 3/3（並びは元の順）
        np.testing.assert_allclose(p[:3], [1.0, 1 / 3, 2 / 3])
        # 2窓目: 値のある行は1つなので 1.0、欠測は欠測のまま
        self.assertEqual(p[3], 1.0)
        self.assertTrue(np.isnan(p[4]))

    def test_outside_any_window_is_missing(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]), "x": [1.0]})
        p = E66.within_window_pct(df, "x", [(np.datetime64("2024-01-01"), np.datetime64("2024-12-31"))])
        self.assertTrue(np.isnan(p[0]))


class TestTrees(unittest.TestCase):
    def test_tag_separates_tree_counts(self):
        self.assertEqual(E66.tag_for(200), "e66")
        self.assertEqual(E66.tag_for(1000), "e66t1000")
        self.assertNotEqual(E66.tag_for(500), E66.tag_for(200))

    def test_with_trees_changes_all_three_models_and_restores(self):
        import e27_timing_multi as E27
        import tuning_multi as TM
        before_lgbm = E27.prod_params("lgbm")["params"]["n_estimators"]
        before_n = TM.N_ESTIMATORS
        restore = E66.with_trees(1000)
        try:
            self.assertEqual(E27.prod_params("lgbm")["params"]["n_estimators"], 1000)
            self.assertEqual(TM.N_ESTIMATORS, 1000)
            # 学習率などは触らない
            lr = E27.prod_params("lgbm")["params"]["learning_rate"]
            self.assertGreater(lr, 0)
            # xgb / cat の学習器が実際に 1000 本で組まれる
            y = np.array([0, 1] * 20)
            self.assertEqual(TM.build("xgb", E27.prod_params("xgb")["params"], y).get_params()["n_estimators"], 1000)
            self.assertEqual(TM.build("cat", E27.prod_params("cat")["params"], y).get_params()["iterations"], 1000)
        finally:
            restore()
        self.assertEqual(E27.prod_params("lgbm")["params"]["n_estimators"], before_lgbm)
        self.assertEqual(TM.N_ESTIMATORS, before_n)
        self.assertEqual(before_lgbm, E66.PROD_TREES)


class TestTuned(unittest.TestCase):
    def test_tags_do_not_collide(self):
        self.assertEqual(E66.tuned_tag(1000), "e66u1000")
        self.assertNotEqual(E66.tuned_tag(1000), E66.tag_for(1000))
        self.assertNotEqual(E66.tuned_tag(200), E66.tag_for(200))

    def test_params_hash_is_stable_and_order_free(self):
        a = E66.params_hash({"a": 1, "b": 0.5})
        b = E66.params_hash({"b": 0.5, "a": 1})
        self.assertEqual(a, b)
        self.assertNotEqual(a, E66.params_hash({"a": 2, "b": 0.5}))

    def test_cache_matches_requires_same_conditions(self):
        want = {"_features_sig": "x", "_n_trials": 50, "_n_estimators": 1000, "_cutoff": "2025-03-21",
                "_lr_range": [0.002, 0.04], "_n_splits": 5}
        rec = {"params": {"learning_rate": 0.01}, **want}
        self.assertTrue(E66.cache_matches(rec, want))
        self.assertFalse(E66.cache_matches({**rec, "_n_estimators": 500}, want))
        self.assertFalse(E66.cache_matches({**rec, "_lr_range": [0.01, 0.2]}, want))
        self.assertFalse(E66.cache_matches({k: v for k, v in rec.items() if k != "params"}, want))
        self.assertFalse(E66.cache_matches(None, want))

    def test_lr_range_scales_with_trees(self):
        import tuning
        lo, hi = tuning.lr_range(1000)
        self.assertAlmostEqual(lo * 1000, tuning.LR_TOTAL[0])
        self.assertAlmostEqual(hi * 1000, tuning.LR_TOTAL[1])
        self.assertEqual(tuning.lr_range(200), (0.01, 0.2))

    def test_tune_arm_refuses_other_models(self):
        with self.assertRaises(SystemExit):
            E66.tune_arm("xgb", "T", pd.DataFrame(), ["a"], 1000, (0.002, 0.04), pd.Timestamp("2025-01-01"))

    def test_compare_tags_reads_both_csvs(self):
        import tempfile
        rows_a, rows_b = [], []
        for sh in (0, 2):
            for fold in range(3):
                for arm in ("V", "P"):
                    rows_a.append({"shift": sh, "arm": arm, "algo": "lgbm", "fold": fold,
                                   "pr_base": 0.25, "pr_arm": 0.26 if arm == "V" else 0.25,
                                   "roc_base": 0.64, "roc_arm": 0.65})
                    rows_b.append({"shift": sh, "arm": arm, "algo": "lgbm", "fold": fold,
                                   "pr_base": 0.27, "pr_arm": 0.27, "roc_base": 0.66, "roc_arm": 0.66})
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(rows_a).to_csv(os.path.join(d, "ta_auc_by_window.csv"), index=False)
            pd.DataFrame(rows_b).to_csv(os.path.join(d, "tb_auc_by_window.csv"), index=False)
            saved = E66.OOF_DIR
            E66.OOF_DIR = d
            try:
                res = E66.compare_tags("ta", "tb", "a", "b")
            finally:
                E66.OOF_DIR = saved
            v_pr = res[(res["arm"] == "V") & (res["metric"] == "pr")].iloc[0]
            self.assertEqual(int(v_pr["n_win"]), 6)
            self.assertAlmostEqual(v_pr["T_b-T_a"], 0.02)       # T: 0.25 → 0.27
            self.assertAlmostEqual(v_pr["arm_b-arm_a"], 0.01)   # V: 0.26 → 0.27
            self.assertAlmostEqual(v_pr["inc_b"], 0.0)          # V−T @b
            self.assertAlmostEqual(v_pr["inc_a"], 0.01)         # V−T @a
            self.assertTrue(os.path.exists(os.path.join(d, "tb_vs_ta.csv")))
            # 片方しか無ければ空
            E66.OOF_DIR = d
            try:
                self.assertTrue(E66.compare_tags("ta", "none", "a", "b").empty)
            finally:
                E66.OOF_DIR = saved


class TestColumnsAgree(unittest.TestCase):
    def test_new_cols_match_module(self):
        self.assertEqual(E66.NEW_COLS, list(NL.NL_COLS))


if __name__ == "__main__":
    unittest.main()
