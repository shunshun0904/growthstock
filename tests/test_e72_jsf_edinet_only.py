#!/usr/bin/env python3
"""
実験72（research/exp/e72_jsf_edinet_only.py）の、重い計算を回さない部分のテスト。

- 列の組: J は日証金の16列だけ、E は EDINET の列から時価総額の組み合わせを除いたもの、JE はその和。
  本番の239列とは一切重ならない（「現行の特徴量は一切使わない」）
- 評価の行の組: all / EDINET の付いた行 / 両方の付いた行
- 窓ごとの表: 正例率が付く。スコアが一定（情報が無い）なら PR-AUC は正例率に等しい。小さい窓は落とす
- 報告: 作った結果で最後まで出て、要約に J / E / JE のリフトが入る

  python3 tests/test_e72_jsf_edinet_only.py
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

import edinet_features as EF  # noqa: E402
import features as F  # noqa: E402
import jsf_features as JF  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
import e72_jsf_edinet_only as E  # noqa: E402


class TestColumns(unittest.TestCase):
    def test_sets(self):
        s = E.column_sets()
        self.assertEqual(s["base"], F.columns(F.DEFAULT_PRESET))
        self.assertEqual(len(s["base"]), 239)
        self.assertEqual(s["jsf"], JF.columns("all"))
        self.assertEqual(len(s["jsf"]), 17)
        self.assertEqual(len(s["ed"]), len(EF.columns("all")) - len(EF.columns("mcap")))
        self.assertFalse(set(s["ed"]) & set(EF.columns("mcap")))             # 時価総額の組み合わせは入れない
        self.assertEqual(s["je"], s["jsf"] + s["ed"])
        self.assertTrue(all(c.startswith("jsf_") for c in s["jsf"]))
        self.assertTrue(all(c.startswith("ed_") for c in s["ed"]))
        self.assertFalse(set(s["base"]) & set(s["je"]))                      # 本番の列は一切使わない
        self.assertEqual(len(s["je"]), len(set(s["je"])))

    def test_arms(self):
        self.assertEqual(E.ARMS, ("A", "J", "E", "JE"))
        self.assertEqual({E.COLS_OF[a] for a in E.ARMS}, {"base", "jsf", "ed", "je"})
        self.assertEqual(E.ALGOS, ("lgbm", "xgb", "cat", "logit"))


class TestRows(unittest.TestCase):
    def test_eval_keysets(self):
        df = pd.DataFrame({"Code": ["10000", "20000", "30000", "40000"],
                           "Date": pd.to_datetime(["2024-01-04"] * 4),
                           "ed_fiscal_year": [2023.0, np.nan, 2023.0, 2023.0],
                           "jsf_ratio": [0.1, 0.2, np.nan, 0.3]})
        ks = E.eval_keysets(df)
        self.assertEqual(list(ks), ["all", "ed", "both"])
        self.assertIsNone(ks["all"])
        self.assertEqual(len(ks["ed"]), 3)
        self.assertEqual(len(ks["both"]), 2)
        self.assertTrue(set(ks["both"]) <= set(ks["ed"]))


class TestWindows(unittest.TestCase):
    def oof(self, n_per_fold=150, folds=(1, 2, 3)):
        rng = np.random.default_rng(0)
        parts = []
        for f in folds:
            y = (rng.random(n_per_fold) < 0.2).astype(int)
            parts.append(pd.DataFrame({"Code": [f"{i:04d}0" for i in range(n_per_fold)],
                                       "Date": pd.Timestamp("2024-01-04") + pd.Timedelta(days=30 * f),
                                       "fold": f, "label": y, "score": y + rng.normal(0, 1, n_per_fold)}))
        return pd.concat(parts, ignore_index=True)

    def test_rate_and_constant_score(self):
        o = self.oof()
        w = E.windows_on(o, None)
        self.assertEqual(list(w["fold"]), [1, 2, 3])
        self.assertTrue((w["n"] == 150).all())
        self.assertTrue(((w["rate"] > 0.05) & (w["rate"] < 0.5)).all())
        self.assertTrue((w["pr"] > w["rate"]).all())                         # 情報があれば正例率より上
        flat = o.copy()
        flat["score"] = 0.5
        wf = E.windows_on(flat, None)
        for _, r in wf.iterrows():
            self.assertAlmostEqual(r["pr"], r["rate"])                        # 一定のスコアは当てずっぽう
            self.assertAlmostEqual(r["roc"], 0.5)

    def test_small_windows_and_keys_are_dropped(self):
        o = self.oof()
        keys = E68.keys_of(o[o["fold"] != 2])
        w = E.windows_on(o, keys)
        self.assertEqual(list(w["fold"]), [1, 3])
        small = o[(o["fold"] != 1) | (o.index < 50)]                           # 窓1 を 50件に
        self.assertEqual(list(E.windows_on(small, None)["fold"]), [2, 3])
        self.assertEqual(len(E.windows_on(o.iloc[:0], None)), 0)


class TestReport(unittest.TestCase):
    """作った結果で report が最後まで出て、記録が書けること（探索・OOF は回さない）。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E68.OOF_DIR, E68.RUN, E68.TAG)
        E68.OOF_DIR, E68.RUN, E68.TAG = self.dir, "e72test_only", "e72test"

    def tearDown(self):
        E68.OOF_DIR, E68.RUN, E68.TAG = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def fake_base(self, rng, n=3000):
        """行（Code・Date・窓・ラベル・結果）は腕で共通。スコアだけ腕ごとに作る。"""
        y = (rng.random(n) < 0.2).astype(int)
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2023-01-04", "2025-12-30"), n))
        return pd.DataFrame({"Code": [f"{i % 200:04d}0" for i in range(n)], "Date": dates,
                             "fold": np.where(dates < pd.Timestamp("2024-01-01"), 1,
                                              np.where(dates < pd.Timestamp("2025-01-01"), 2, 3)),
                             "label": y, "ret_o1_20": rng.normal(0, 5, n) + 2.0 * y, "ret_o1_40": rng.normal(0, 8, n)})

    def fake_oof(self, rng, base, strength=1.0):
        o = base.copy()
        o["score"] = strength * o["label"] + rng.normal(0, 1, len(o))
        return o

    def test_report_writes_summary(self):
        rng = np.random.default_rng(1)
        rec = {"params": {"n_estimators": 10}, "_cv": {"mean_pr_auc": 0.3, "std": 0.02, "mean_roc_auc": 0.6,
                                                        "fold_scores": [0.3, 0.3]}, "_seconds": 1}
        strength = {"A": 1.0, "J": 0.3, "E": 0.5, "JE": 0.6}
        base = self.fake_base(rng)
        prod_all = {arm: self.fake_oof(rng, base, strength=strength[arm]) for arm in E.ARMS}
        keys_ed = E68.keys_of(prod_all["A"].iloc[::2])
        keys_both = E68.keys_of(prod_all["A"].iloc[::3])
        keysets = {"all": None, "ed": keys_ed, "both": keys_both}
        prod = {rs: {arm: (o if k is None else E68.on_rows(o, k)) for arm, o in prod_all.items()}
                for rs, k in keysets.items()}
        wins = {rs: {(sh, arm): E.windows_on(prod_all[arm], k) for sh in (0,) for arm in E.ARMS}
                for rs, k in keysets.items()}
        results = {"lgbm": {"recs": {arm: rec for arm in E.ARMS}, "params": {arm: rec["params"] for arm in E.ARMS},
                            "prod": prod, "wins": wins}}
        stamp = {"built_utc": "2026-10-10 14:00 UTC", "n_codes": 1670, "rows": 600, "date_max": "2025-12-30",
                 "by_year": {}}
        with mock.patch("builtins.print"):
            s = E.report(results, [0], stamp, keysets, {"fill": {}})
        self.assertIn("lgbm_JE_both", s["production_oof"])
        self.assertIn("lgbm_pr_JE-A_all", s["windows"])
        self.assertIn("lgbm_pr_J-rate_both", s["windows"])
        self.assertEqual(set(s["headline"]["lgbm_all"]), set(E.ARMS))
        self.assertGreater(s["production_oof"]["lgbm_A_all"]["pr_auc"], s["production_oof"]["lgbm_J_all"]["pr_auc"])
        self.assertAlmostEqual(s["production_oof"]["lgbm_A_all"]["rate"], prod_all["A"]["label"].mean())
        self.assertGreater(s["production_oof"]["lgbm_A_all"]["ret_o1_20_n"], 0)          # 全行ならしきい値が引ける
        both_n = s["production_oof"]["lgbm_A_both"]["ret_o1_20_n"]
        self.assertGreaterEqual(both_n, 0)                                                 # 少ない行でも落ちない
        with open(os.path.join(self.dir, "e72test_only_summary.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["headline"].keys() >= {"lgbm_all", "lgbm_ed", "lgbm_both"}, True)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e72test_only_auc_by_window_all.csv")))


class TestMain(unittest.TestCase):
    def test_unknown_algo(self):
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit):
                E.main(["--algos", "mlp"])


if __name__ == "__main__":
    unittest.main()
