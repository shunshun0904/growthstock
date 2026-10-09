#!/usr/bin/env python3
"""実験70（research/exp/e70_label_sigma_dfa_full.py）の部品のテスト。学習はしない。"""
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

import nonlinear_features as NL  # noqa: E402
import ops_rule as OR  # noqa: E402
import e54_label_sigma as E54  # noqa: E402
import e55_label_sigma_full as E55  # noqa: E402
import e69_label_sigma_dfa as E69  # noqa: E402
import e70_label_sigma_dfa_full as E70  # noqa: E402


class TestConfigure(unittest.TestCase):
    def setUp(self):
        self.saved = (dict(E55.ARMS), dict(E55.LABELS), E55.PARAMS_PATH, E55.CSV_PREFIX, E55.oof,
                      E54.sigmas_from_bars)

    def tearDown(self):
        arms, labels, params, csv, oof, sig = self.saved
        E55.ARMS.clear()
        E55.ARMS.update(arms)
        E55.LABELS.clear()
        E55.LABELS.update(labels)
        E55.PARAMS_PATH, E55.CSV_PREFIX, E55.oof = params, csv, oof
        E54.sigmas_from_bars = sig

    def test_swaps_arms_names_and_sigma(self):
        self.assertEqual(E55.CSV_PREFIX, "e55")
        E70.configure()
        self.assertEqual(E55.ARMS, {"L0": "sigma20", "L5": "sigma_dfa"})
        self.assertEqual(set(E55.LABELS), {"L0", "L5"})
        self.assertTrue(E55.PARAMS_PATH.endswith("e70_params.json"))
        self.assertEqual(E55.CSV_PREFIX, "e70")
        self.assertIs(E55.oof, E70.oof70)
        self.assertIs(E54.sigmas_from_bars, E70.sigmas_with_dfa)

    def test_oof_path_uses_e70_prefix(self):
        p = E70.oof_path("L5", "B2", "lgbm", 2, 7)
        self.assertTrue(p.endswith("e70_L5_B2_lgbm_sh2_s7.parquet"))


class TestSigmasWithDfa(unittest.TestCase):
    def test_matches_e69_formula_and_drops_alpha(self):
        dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-04"])
        sg = pd.DataFrame({"Code": ["1", "1", "2"], "Date": dates,
                           "sigma20": [2.0, 2.0, 2.0], "sigma120": [8.0, 8.0, 8.0]})
        nl = pd.DataFrame({"Code": ["1", "1"], "Date": dates[:2], E69.ALPHA_COL: [0.9, 0.4]})  # 銘柄 2 は α 無し
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nl.parquet")
            nl.to_parquet(path, index=False)
            with mock.patch.object(E70, "_orig_sigmas", lambda bars: sg.copy()), \
                    mock.patch.object(NL, "OUT_PATH", path):
                out = E70.sigmas_with_dfa(pd.DataFrame())
        self.assertEqual(len(out), 3)
        self.assertNotIn(E69.ALPHA_COL, out.columns)
        want = E69.sigma_dfa(sg["sigma20"], sg["sigma120"], pd.Series([0.9, 0.4, np.nan]))
        np.testing.assert_allclose(out["sigma_dfa"].to_numpy(), want.to_numpy())
        self.assertAlmostEqual(out["sigma_dfa"].iloc[0], 2.0)  # α 高い → σ20
        self.assertAlmostEqual(out["sigma_dfa"].iloc[1], 8.0)  # α 低い → σ120

    def test_missing_parquet_is_an_error(self):
        with mock.patch.object(E70, "_orig_sigmas", lambda bars: pd.DataFrame()), \
                mock.patch.object(NL, "OUT_PATH", "/nonexistent/nl.parquet"):
            with self.assertRaises(SystemExit):
                E70.sigmas_with_dfa(pd.DataFrame())


def _oofs(seed: int, n_fold: int = 3, per: int = 600, flip: bool = False):
    """3モデル分の合成 out-of-fold。3モデルのスコアは相関させる（合議で 5 件以上選ばれるように）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fold):
        dates = pd.date_range("2024-01-01", periods=per, freq="D") + pd.Timedelta(days=f * per)
        s = rng.random(per)
        lab = (rng.random(per) < 0.2 + 0.4 * s).astype(float)
        rows.append(pd.DataFrame({"Code": [str(1000 + i) for i in range(per)], "Date": dates, "fold": f,
                                  "label": 1 - lab if flip else lab, "score": s,
                                  "ret_o1_20": rng.normal(0.01, 0.1, per), "ret_o1_40": rng.normal(0.01, 0.15, per)}))
    base = pd.concat(rows, ignore_index=True)
    out = {}
    for a, noise in (("lgbm", 0.0), ("xgb", 0.02), ("cat", 0.02)):
        o = base.copy()
        o["score"] = o["score"] + rng.normal(0, noise, len(o)) if noise else o["score"]
        out[a] = o
    return out


class TestPrecision(unittest.TestCase):
    def test_band_stats_fractions(self):
        rows = pd.DataFrame({"label": [1, 0, 1, 0], "y_L0": [1, 1, 0, 0],
                             "ret_o1_20": [0.15, -0.12, 0.02, np.nan]})
        st = E70.band_stats(rows)
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["hit_own"], 0.5)
        self.assertAlmostEqual(st["hit_l0"], 0.5)
        self.assertAlmostEqual(st["win"], 2 / 3)       # 欠測は分母に入れない
        self.assertAlmostEqual(st["big_up"], 1 / 3)
        self.assertAlmostEqual(st["big_down"], 1 / 3)
        self.assertAlmostEqual(st["ret20"], (0.15 - 0.12 + 0.02) / 3 * 100)

    def test_band_stats_without_base_label(self):
        st = E70.band_stats(pd.DataFrame({"label": [1.0], "ret_o1_20": [0.05]}))
        self.assertTrue(np.isnan(st["hit_l0"]))
        self.assertEqual(st["n"], 1)

    def test_precision_table_matches_consensus_and_base_label(self):
        by_arm = {"L0": _oofs(1), "L5": _oofs(1, flip=True)}  # 同じ行・同じスコアで、ラベルだけ反転
        t = E70.precision_table(by_arm)
        self.assertEqual(sorted(t["rule"].unique()), sorted(s[0] for s in E70.SELECTIONS))
        self.assertEqual(len(t), 6)
        l0 = t[t["arm"] == "L0"].set_index("rule")
        for name, models, pct in E70.SELECTIONS:
            c = OR.consensus(by_arm["L0"], pct, models=models)
            self.assertEqual(int(l0.loc[name, "n"]), c["n"])
            self.assertAlmostEqual(float(l0.loc[name, "hit_own"]), c["label_rate"])
            self.assertAlmostEqual(float(l0.loc[name, "hit_own"]), float(l0.loc[name, "hit_l0"]))  # 現行の腕は同じ
            self.assertAlmostEqual(float(l0.loc[name, "ret20"]), c["ret20"])
        self.assertGreater(int(l0.loc["3モデル 95以上", "n"]), 0)
        l5 = t[t["arm"] == "L5"].set_index("rule")
        # L5 の「現行ラベルでの正例率」は L0 のラベルを同じ行に付けたもの（反転ラベルなので自分の正例率と足して 1）
        for name in l5.index:
            self.assertAlmostEqual(float(l5.loc[name, "hit_own"]) + float(l5.loc[name, "hit_l0"]), 1.0)

    def test_summarize_weights_by_count(self):
        t = pd.DataFrame({"fit": ["B1", "B1"], "arm": ["L0", "L0"], "rule": ["r", "r"],
                          "n": [100, 300], "hit_own": [0.5, 0.3], "hit_l0": [0.5, 0.3], "ret20": [1.0, 2.0],
                          "win": [0.6, 0.4], "big_up": [0.1, 0.2], "big_down": [0.1, 0.1],
                          "won": [3, 4], "folds": [5, 6]})
        s = E70.summarize_precision(t)
        self.assertEqual(len(s), 1)
        self.assertEqual(int(s.loc[0, "n"]), 400)
        self.assertAlmostEqual(float(s.loc[0, "hit_own"]), 0.35)
        self.assertAlmostEqual(float(s.loc[0, "ret20"]), 1.75)
        self.assertEqual(int(s.loc[0, "won"]), 7)
        self.assertEqual(int(s.loc[0, "folds"]), 11)

    def test_load_oof_averages_scores_and_requires_files(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E70, "OOF_DIR", d):
            with self.assertRaises(FileNotFoundError):
                E70.load_oof("L0", "B1", "lgbm", 0, seeds=(1,))
            a = _oofs(3)["lgbm"]
            b = a.copy()
            b["score"] = b["score"] + 0.5
            a.to_parquet(E70.oof_path("L0", "B1", "lgbm", 0, 1), index=False)
            b.to_parquet(E70.oof_path("L0", "B1", "lgbm", 0, 2), index=False)
            o = E70.load_oof("L0", "B1", "lgbm", 0, seeds=(1, 2))
        np.testing.assert_allclose(o["score"].to_numpy(), a["score"].to_numpy() + 0.25)
        self.assertEqual(str(o["Code"].dtype), "object")

    def test_precision_main_end_to_end_on_saved_files(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E70, "OOF_DIR", d):
            for arm, seed in (("L0", 11), ("L5", 12)):
                for fit in ("B1", "B2"):
                    oofs = _oofs(seed + (0 if fit == "B1" else 100))
                    for algo in ("lgbm", "xgb", "cat"):
                        if algo != "lgbm" and fit == "B2":
                            continue
                        for sd in (1, 2):
                            oofs[algo].to_parquet(E70.oof_path(arm, fit, algo, 0, sd), index=False)
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = E70.precision_main([0], seeds=(1, 2))
            self.assertTrue(os.path.exists(os.path.join(d, "e70_precision.csv")))
        self.assertEqual(set(out["fit"]), {"B1", "B2"})
        self.assertEqual(len(out), 12)  # 2 腕 × 3 選定 × 2 木の形
        self.assertIn("切り方1通りをまとめて", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
