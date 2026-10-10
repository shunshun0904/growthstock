#!/usr/bin/env python3
"""
実験73（research/exp/e73_cv_scheme.py）: 探索の分割（層別 / 前進分割）と目的関数で lgbm の精度はどう変わるか。
重い計算を回さない部分。

- 腕 S は year_cap_date・PR-AUC、W は walkforward・PR-AUC、WL は walkforward・リフト。E68.tune に渡る分割と目的関数が
  腕ごとに切り替わり、呼んだあと元に戻る
- 窓ごとの表を「打ち切り日より後に始まる窓」で分けられる
- 作った結果で report が最後まで出て、記録が書ける（3腕・3組の差）

  python3 tests/test_e73_cv_scheme.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e68_edinet_ab as E68  # noqa: E402
import e73_cv_scheme as E  # noqa: E402


class TestArms(unittest.TestCase):
    def test_specs(self):
        import tuning
        self.assertEqual(E.ARMS, ("S", "W", "WL"))
        self.assertEqual(E.ARM_SPEC["S"], (tuning.PRODUCTION_CV, tuning.PRODUCTION_OBJECTIVE))   # S が今の本番
        self.assertEqual(E.ARM_SPEC["W"], ("walkforward", "pr_auc"))
        self.assertEqual(E.ARM_SPEC["WL"], ("walkforward", "lift"))
        for x, y in E.PAIRS:
            self.assertIn(x, E.ARMS)
            self.assertIn(y, E.ARMS)

    def test_tune_arm_switches_scheme_and_objective_and_restores(self):
        seen = []

        def fake_tune(algo, arm, df, cols, cutoff, n_trials, stamp, compute):
            seen.append((arm, E68.CV_SCHEME, E68.CV_OBJECTIVE))
            return {"params": {"n_estimators": 1}, "_cv": {}}

        before = (E68.CV_SCHEME, E68.CV_OBJECTIVE)
        with mock.patch.object(E68, "tune", side_effect=fake_tune):
            for arm in E.ARMS:
                E.tune_arm(arm, None, [], None, 2, {}, True)
        self.assertEqual(seen, [("S", "year_cap_date", "pr_auc"), ("W", "walkforward", "pr_auc"),
                                ("WL", "walkforward", "lift")])
        self.assertEqual((E68.CV_SCHEME, E68.CV_OBJECTIVE), before)


class TestWindows(unittest.TestCase):
    def test_split_by_cutoff(self):
        dates = pd.Series(pd.bdate_range("2018-04-02", "2026-09-08"))
        starts = E.window_starts(dates, [0, 2])
        self.assertTrue(all(isinstance(v, pd.Timestamp) for v in starts.values()))
        self.assertGreater(len([k for k in starts if k[0] == 0]), 5)
        s = pd.DataFrame({"shift": [0] * 3, "fold": [1, 2, max(f for sh, f in starts if sh == 0)],
                          "pr_S": [0.2, 0.3, 0.25], "pr_W": [0.21, 0.29, 0.27]})
        cutoff = pd.Timestamp("2025-08-10")
        out = E.split_windows(s, starts, cutoff)
        self.assertEqual(list(out["holdout"]), [False, False, True])         # 最後の窓は打ち切り日より後に始まる
        self.assertTrue(out["start"].notna().all())


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E68.OOF_DIR, E68.RUN, E68.TAG)
        E68.OOF_DIR, E68.RUN, E68.TAG = self.dir, "e73test_lgbm", "e73test"

    def tearDown(self):
        E68.OOF_DIR, E68.RUN, E68.TAG = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_runs(self):
        rng = np.random.default_rng(2)
        n = 3000
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2023-01-04", "2026-06-30"), n))
        base = pd.DataFrame({"Code": [f"{i % 200:04d}0" for i in range(n)], "Date": dates,
                             "fold": np.searchsorted(pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]), dates) + 1,
                             "label": (rng.random(n) < 0.2).astype(int)})
        base["ret_o1_20"] = rng.normal(0, 5, n) + 2 * base["label"]
        base["ret_o1_40"] = rng.normal(0, 8, n)
        prod = {}
        for arm, st in (("S", 1.0), ("W", 1.1), ("WL", 1.05)):
            o = base.copy()
            o["score"] = st * o["label"] + rng.normal(0, 1, n)
            prod[arm] = o
        wins = {(0, arm): E.windows_on(prod[arm], None) for arm in E.ARMS}
        cv = {"scheme": "year_cap_date", "objective": "pr_auc", "objective_value": 0.3, "mean_pr_auc": 0.3, "std": 0.01,
              "mean_roc_auc": 0.6, "fold_scores": [0.3, 0.3], "fold_pos_rate": [0.2, 0.2]}
        wf = {"scheme": "walkforward", "fold_windows": [{"valid_from": "2025-01-02", "valid_to": "2025-06-30", "pos_rate": 0.2},
                                                        {"valid_from": "2025-07-01", "valid_to": "2025-12-30", "pos_rate": 0.2}]}
        recs = {"S": {"params": {"a": 1}, "_cv": cv, "_seconds": 1},
                "W": {"params": {"a": 2}, "_cv": {**cv, **wf}, "_seconds": 1},
                "WL": {"params": {"a": 3}, "_cv": {**cv, **wf, "objective": "lift", "objective_value": 1.5}, "_seconds": 1}}
        res = {"recs": recs, "params": {arm: recs[arm]["params"] for arm in E.ARMS}, "prod": prod, "wins": wins}
        starts = {(0, 1): pd.Timestamp("2023-01-04"), (0, 2): pd.Timestamp("2024-01-01"),
                  (0, 3): pd.Timestamp("2025-01-01"), (0, 4): pd.Timestamp("2026-01-01")}
        with mock.patch("builtins.print"):
            s = E.report(res, [0], pd.Timestamp("2025-08-10"), starts)
        self.assertIn("全窓", s["windows"])
        self.assertEqual(s["windows"]["打ち切り日より後に始まる窓"]["n"], 1)
        self.assertEqual(s["windows"]["全窓"]["n"], 4)
        self.assertIn("pr_WL-W", s["windows"]["全窓"])
        self.assertIn("pr_WL-S", s["windows"]["全窓"])
        self.assertGreater(s["production_oof"]["WL"]["pr_auc"], 0)
        self.assertEqual(set(s["cv_vs_oof"]), {"W", "WL"})                    # 前進分割の腕だけ
        self.assertEqual(len(s["cv_vs_oof"]["W"]), 2)
        with open(os.path.join(self.dir, "e73test_lgbm_summary.json"), encoding="utf-8") as fh:
            self.assertIn("windows", json.load(fh))


if __name__ == "__main__":
    unittest.main()
