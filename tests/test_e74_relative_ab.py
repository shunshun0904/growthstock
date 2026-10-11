#!/usr/bin/env python3
"""
実験74（research/exp/e74_relative_ab.py）の、重い計算を回さない部分のテスト。

- 腕: A / C1（+業種内 10列）/ C2（+業種内 + 市場全体 20列）/ P1・P2（対照）/ B（20列で探索し直し）。
  列の組は 239 / 249 / 259 で、足す列は本番の列と重ならない。探索は A と B だけ
- 対照の入れ替えは足した列だけを同じ日付の中で入れ替え、本番の列は変えない
- 評価の行の組: 全行と、参照（業種内・市場全体とも）の付いた行
- 報告: 作った結果で最後まで出て、§7 の判定が C1（対 P1）と C2（対 P2）の両方で出る

  python3 tests/test_e74_relative_ab.py
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

import features as F  # noqa: E402
import relative_features as RF  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
import e74_relative_ab as E  # noqa: E402
from e44_shortsale import permuted  # noqa: E402


class TestArms(unittest.TestCase):
    def test_column_sets(self):
        s = E.column_sets()
        self.assertEqual(s["base"], F.columns(F.DEFAULT_PRESET))
        self.assertEqual((len(s["base"]), len(s["ind"]), len(s["both"])), (239, 249, 259))
        self.assertEqual(s["ind"], s["base"] + RF.columns("ind"))
        self.assertEqual(s["both"], s["base"] + RF.columns("all"))
        self.assertEqual(s["perm_ind"], s["ind"])
        self.assertEqual(s["perm_both"], s["both"])
        self.assertEqual(s["add_ind"], RF.columns("ind"))
        self.assertEqual(s["add_both"], RF.columns("all"))
        self.assertFalse(set(s["base"]) & set(s["add_both"]))
        self.assertEqual(len(s["both"]), len(set(s["both"])))

    def test_arms(self):
        self.assertEqual(E.ARMS, ("A", "C1", "C2", "P1", "P2", "B"))
        self.assertEqual(set(E.COLS_OF), set(E.ARMS))
        self.assertEqual(set(E.PARAMS_OF), set(E.ARMS))
        self.assertEqual(E.TUNE_ARMS, ("A", "B"))
        self.assertEqual({E.PARAMS_OF[a] for a in E.ARMS}, {"A", "B"})
        self.assertEqual(E.PARAMS_OF["B"], "B")
        self.assertTrue(all(E.PARAMS_OF[a] == "A" for a in ("C1", "C2", "P1", "P2")))
        self.assertEqual(E.COLS_OF["C2"], E.COLS_OF["B"])
        self.assertEqual(E.JUDGE_PAIRS, (("C1", "P1"), ("C2", "P2")))
        for x, y in E.PAIRS:
            self.assertIn(x, E.ARMS)
            self.assertIn(y, E.ARMS)
        self.assertEqual(E.DEFAULT_ALGOS, "lgbm")                              # 運用者の「lgbm のみで良い」
        self.assertEqual(E.ALGOS, E68.ALGOS)
        self.assertNotEqual(E.PERM_SEED, E68.PERM_SEED)


class TestRows(unittest.TestCase):
    def frame(self):
        n = 6
        df = pd.DataFrame({"Code": [f"{i:04d}0" for i in range(n)], "Date": pd.to_datetime(["2024-01-05"] * 3 + ["2024-01-09"] * 3)})
        for c in RF.columns("all"):
            df[c] = np.nan
        df.loc[0, "rel_ind_per"] = 0.3
        df.loc[0, "rel_mkt_per"] = 0.4
        df.loc[1, "rel_mkt_per"] = 0.5                       # 市場全体だけ
        df.loc[2, "rel_ind_ROE_q0"] = 0.1                    # 業種内だけ
        df.loc[3, "rel_ind_ROE_q0"] = 0.2
        df.loc[3, "rel_mkt_ROE_q0"] = 0.9
        return df

    def test_covered_needs_both(self):
        df = self.frame()
        self.assertEqual(list(E.covered_mask(df)), [True, False, False, True, False, False])
        ks = E.eval_keysets(df)
        self.assertEqual(list(ks), ["all", "covered"])
        self.assertIsNone(ks["all"])
        self.assertEqual(len(ks["covered"]), 2)

    def test_permuted_keeps_base_columns(self):
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({"Code": [f"{i:04d}0" for i in range(n)],
                           "Date": pd.to_datetime(["2024-01-05"] * 20 + ["2024-01-09"] * 20),
                           "per": rng.normal(size=n), "label": (rng.random(n) < 0.3).astype(int)})
        add = RF.columns("ind")
        for c in add:
            df[c] = rng.random(n)
        p = permuted(df, add, seed=E.PERM_SEED)
        pd.testing.assert_series_equal(p["per"], df["per"])
        pd.testing.assert_series_equal(p["label"], df["label"])
        self.assertFalse(p[add].equals(df[add]))
        for d, g in df.groupby("Date"):
            for c in add:
                self.assertEqual(sorted(g[c]), sorted(p.loc[g.index, c]))   # 同じ日付の中で入れ替わるだけ


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E68.OOF_DIR, E68.RUN, E68.TAG)
        E68.OOF_DIR, E68.RUN, E68.TAG = self.dir, "e74test_rel", "e74test"

    def tearDown(self):
        E68.OOF_DIR, E68.RUN, E68.TAG = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_runs_and_judges_both_candidates(self):
        rng = np.random.default_rng(3)
        n = 3000
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2023-01-04", "2025-12-30"), n))
        y = (rng.random(n) < 0.2).astype(int)
        base = pd.DataFrame({"Code": [f"{i % 200:04d}0" for i in range(n)], "Date": dates,
                             "fold": np.where(dates < pd.Timestamp("2024-01-01"), 1,
                                              np.where(dates < pd.Timestamp("2025-01-01"), 2, 3)),
                             "label": y, "ret_o1_20": rng.normal(0, 5, n) + 2.0 * y, "ret_o1_40": rng.normal(0, 8, n)})
        strength = {"A": 1.0, "C1": 1.1, "C2": 1.2, "P1": 1.0, "P2": 0.95, "B": 1.15}
        prod_all = {}
        for arm in E.ARMS:
            o = base.copy()
            o["score"] = strength[arm] * o["label"] + rng.normal(0, 1, n)
            prod_all[arm] = o
        keys_cov = E68.keys_of(base.iloc[::2])
        keysets = {"all": None, "covered": keys_cov}
        prod = {rs: {arm: (o if k is None else E68.on_rows(o, k)) for arm, o in prod_all.items()}
                for rs, k in keysets.items()}
        wins = {rs: {(sh, arm): E68.windows_on(prod_all[arm], k) for sh in (0,) for arm in E.ARMS}
                for rs, k in keysets.items()}
        cv = {"mean_pr_auc": 0.3, "std": 0.02, "mean_roc_auc": 0.6, "fold_scores": [0.3, 0.3, 0.3]}
        recs = {"A": {"params": {"n_estimators": 10}, "_cv": cv, "_seconds": 1},
                "B": {"params": {"n_estimators": 12}, "_cv": {**cv, "mean_pr_auc": 0.31}, "_seconds": 1}}
        results = {"lgbm": {"recs": recs, "params": {arm: recs[E.PARAMS_OF[arm]]["params"] for arm in E.ARMS},
                            "prod": prod, "wins": wins}}
        stamp = {"built_utc": "2026-10-11 10:00 UTC", "rows": n, "date_max": "2025-12-30",
                 "reference": {"month_ends": 36, "first": "2023-01-31", "last": "2025-12-31",
                               "names": {"min": 3700, "median": 3800.0, "max": 3900}, "industries": 33,
                               "industries_ok": {"min": 30, "median": 31.0, "max": 32}, "with_industry": 0.99,
                               "fill": {m: 0.9 for m in RF.METRICS}},
                 "metrics": list(RF.METRICS), "min_industry": 20, "min_market": 200, "covered": n // 2,
                 "fill": {c: 0.9 for c in RF.columns("all")},
                 "by_year": {"2023": {"rows": 1000, "covered": 500}}}
        with mock.patch("builtins.print"):
            s = E.report(results, [0], stamp, keysets, {"covered": n // 2, "covered_share": 0.5, "fill": {}})
        self.assertIn("lgbm_C2_all", s["production_oof"])
        self.assertIn("lgbm_pr_C2-A_all", s["windows"])
        self.assertIn("lgbm_pr_C1-P1_covered", s["windows"])
        self.assertIn("lgbm_pr_C2-C1_all", s["windows"])
        for c in ("C1", "C2"):
            for rs in keysets:
                self.assertIn(f"lgbm_{c}_{rs}", s["judge7"])
                v = s["judge7"][f"verdict_{c}_{rs}"]
                self.assertFalse(v["complete"])                               # lgbm だけでは判定しない
                self.assertEqual(v["of"], 1)
        self.assertGreater(s["production_oof"]["lgbm_C2_all"]["pr_auc"], s["production_oof"]["lgbm_A_all"]["pr_auc"])
        self.assertNotIn("fill", s["data"])                                   # 要約は列ごとの充足を持たない（stamp に）
        with open(os.path.join(self.dir, "e74test_rel_summary.json"), encoding="utf-8") as fh:
            self.assertIn("judge7", json.load(fh))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e74test_rel_auc_by_window_all.csv")))


class TestMain(unittest.TestCase):
    def test_unknown_algo(self):
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit):
                E.main(["--algos", "mlp"])


if __name__ == "__main__":
    unittest.main()
