#!/usr/bin/env python3
"""
実験78（research/exp/e78_label_k.py）: 到達しきい値 k を変えたラベルで lgbm を探索し直す。重い計算を回さない部分。

- 腕の名前（K12 / K15 / K18）と、k=1.2 の引き直しが本番のラベルと同じ合成（到達 & 終盤 0.5倍 & MA5>=MA20）
- k を上げると正例は減る（単調）。必要な上昇率は k に比例
- 月に n 銘柄（各月のスコア上位 n 件）と窓ごとの差
- 相手のラベルで測る表、評価列の結合
- 作った結果で report が最後まで出る

  python3 tests/test_e78_label_k.py
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

import build_dataset as B  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
import e78_label_k as E  # noqa: E402


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.8, 5.0, n)
    sig = vol / 100 * np.sqrt(20)
    rise = rng.normal(0.05, 0.15, n)
    end = rise - rng.uniform(0.0, 0.12, n)
    up = (rng.random(n) < 0.7).astype(float)
    df = pd.DataFrame({"Code": [f"{i % 80:04d}0" for i in range(n)],
                       "Date": pd.to_datetime(rng.choice(pd.bdate_range("2023-01-04", "2025-12-30"), n)),
                       "vol_20d": vol, "future_rise": rise, "end_level": end, "uptrend_end": up,
                       "ret_o1_20": end, "ret_o1_40": end * 1.2})
    need, end_need = B.rise_thresholds(df["vol_20d"], B.DEFAULT_RISE)
    df["label"] = ((df["future_rise"] >= need.to_numpy()) & (df["end_level"] >= end_need.to_numpy())
                   & (df["uptrend_end"] == 1.0)).astype(float)
    del sig
    return df


class TestLabels(unittest.TestCase):
    def test_names(self):
        self.assertEqual([E.arm_name(k) for k in (1.2, 1.5, 1.8)], ["K12", "K15", "K18"])
        self.assertEqual(E.BASE, "K12")
        self.assertEqual(E.K0, 1.2)
        self.assertIn("現行", E.arm_label(1.2))
        self.assertNotIn("現行", E.arm_label(1.5))

    def test_relabel_matches_production_and_is_monotone(self):
        df = frame()
        y0, need0 = E.relabel(df, 1.2)
        self.assertTrue((y0 == df["label"]).all())
        y15, need15 = E.relabel(df, 1.5)
        np.testing.assert_allclose(need15.to_numpy(), need0.to_numpy() * 1.5 / 1.2)
        self.assertLess(y15.sum(), y0.sum())
        self.assertTrue(((y15 == 1) <= (y0 == 1)).all())                      # 1.5σ の正例は 1.2σ の正例の部分集合
        # 終盤は到達の 0.5倍（1.5σ → 0.75σ）
        cfg = B.RiseConfig(vol_norm_k=1.5)
        need, end_need = B.rise_thresholds(df["vol_20d"].iloc[:3], cfg)
        np.testing.assert_allclose(end_need.to_numpy(), 0.5 * need.to_numpy())
        np.testing.assert_allclose(need.to_numpy(), 1.5 * df["vol_20d"].iloc[:3].to_numpy() / 100 * np.sqrt(20))
        d, tab = E.build_labels(df.copy(), [1.2, 1.5])
        self.assertEqual(list(tab["arm"]), ["K12", "K15"])
        self.assertGreater(tab.loc[0, "pos_rate"], tab.loc[1, "pos_rate"])
        self.assertIn("y_K15", d.columns)
        self.assertIn("need_K15", d.columns)


class TestEval(unittest.TestCase):
    def oof(self, seed=1):
        rng = np.random.default_rng(seed)
        n = 600
        dates = pd.to_datetime(rng.choice(pd.bdate_range("2024-01-04", "2025-12-30")[::12], n))   # 約43日 × 約14件（日付内の順位相関が引ける）
        y = (rng.random(n) < 0.2).astype(int)
        return pd.DataFrame({"Code": [f"{i % 100:04d}0" for i in range(n)], "Date": dates,
                             "fold": np.where(dates < pd.Timestamp("2025-01-01"), 1, 2),
                             "label": y, "score": y + rng.normal(0, 1, n),
                             "ret_o1_20": rng.normal(0.01, 0.1, n) + 0.1 * y, "ret_o1_40": rng.normal(0.01, 0.15, n),
                             "y_K12": y, "y_K15": (y & (rng.random(n) < 0.6)).astype(int)}
                            ).drop_duplicates(["Code", "Date"]).reset_index(drop=True)      # 行は (Code, Date) で一意

    def test_monthly_top(self):
        o = self.oof()
        picks = E.monthly_top(o, 2)
        months = pd.to_datetime(o["Date"]).dt.to_period("M").nunique()
        self.assertEqual(len(picks), 2 * months)
        for m, g in picks.groupby("_m"):
            top = o[pd.to_datetime(o["Date"]).dt.to_period("M") == m].nlargest(2, "score")["score"]
            self.assertEqual(sorted(g["score"]), sorted(top))
        st = E.monthly_stats(picks, "label")
        self.assertEqual(st["n"], len(picks))
        self.assertEqual(st["months"], months)
        self.assertTrue(0 <= st["win"] <= 1 and 0 <= st["hit10"] <= 1 and 0 <= st["bad10"] <= 1)
        bf = E.monthly_by_fold(picks)
        self.assertEqual(set(bf["fold"]), {1, 2})
        d = E.diff_by_fold(bf, bf)
        self.assertEqual((d["n"], d["mean"], d["wins"]), (2, 0.0, 0))
        self.assertEqual(E.fmt_diff({}), "")
        self.assertIn("±", E.fmt_diff(d))

    def test_windows_cross(self):
        o = self.oof()
        w = E.windows_cross(o, "y_K12")
        self.assertEqual(list(w["fold"]), [1, 2])
        self.assertTrue((w["pr"] > w["rate"]).all())
        self.assertTrue(((w["roc"] > 0.5) & (w["roc"] <= 1.0)).all())
        small = o[(o["fold"] != 1) | (o.index < 50)]                     # 窓1 を 50件に → 落ちる
        self.assertEqual(list(E.windows_cross(small, "y_K12")["fold"]), [2])

    def test_rank_agreement(self):
        o = self.oof()
        same = E.rank_agreement(o, o)
        self.assertEqual(same["n"], len(o))
        self.assertAlmostEqual(same["rho_all"], 1.0)
        self.assertAlmostEqual(same["rho_day"], 1.0)
        self.assertAlmostEqual(same["top5_in_b"], 1.0)
        self.assertAlmostEqual(same["top10_in_b"], 1.0)
        self.assertGreater(same["top5_pct_b"], 0.95)
        self.assertAlmostEqual(same["month_overlap"], 1.0)
        rng = np.random.default_rng(9)
        other = o.copy()
        other["score"] = rng.normal(size=len(o))                      # 無関係なスコア
        r = E.rank_agreement(o, other)
        self.assertLess(abs(r["rho_all"]), 0.2)
        self.assertLess(r["top10_in_b"], 0.4)
        self.assertLess(r["month_overlap"], 0.5)
        self.assertEqual(E.rank_agreement(o.iloc[:5], o.iloc[:5])["n"], 5)        # 少なすぎれば件数だけ

    def test_cross_label(self):
        o = self.oof()
        cl = E.cross_label(o, ["y_K12", "y_K15"])
        self.assertEqual(set(cl), {"y_K12", "y_K15"})
        self.assertGreater(cl["y_K12"]["pr"], cl["y_K12"]["rate"])
        self.assertAlmostEqual(cl["y_K12"]["lift"], cl["y_K12"]["pr"] / cl["y_K12"]["rate"])
        self.assertLess(cl["y_K15"]["rate"], cl["y_K12"]["rate"])
        flat = o.copy()
        flat["y_K15"] = 1
        self.assertTrue(np.isnan(E.cross_label(flat, ["y_K15"])["y_K15"]["pr"]))        # 1クラスなら NaN


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E68.OOF_DIR, E68.RUN, E68.TAG)
        E68.OOF_DIR, E68.RUN, E68.TAG = self.dir, "e78test_k", "e78test"

    def tearDown(self):
        E68.OOF_DIR, E68.RUN, E68.TAG = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_report_runs(self):
        rng = np.random.default_rng(4)
        n = 3000
        df = frame(n, seed=4)
        df["Date"] = pd.to_datetime(rng.choice(pd.bdate_range("2023-01-04", "2025-12-30"), n))
        df["Code"] = [f"{i % 300:04d}0" for i in range(n)]
        df = df.drop_duplicates(["Code", "Date"]).reset_index(drop=True)
        df, _ = E.build_labels(df, [1.2, 1.5])
        base = df[["Code", "Date", "ret_o1_20", "ret_o1_40"]].copy()
        base["fold"] = np.searchsorted(pd.to_datetime(["2024-01-01", "2025-01-01"]), base["Date"]) + 1
        prod, shifted = {}, {}
        for a, st in (("K12", 1.0), ("K15", 1.2)):
            o = base.copy()
            o["label"] = df[f"y_{a}"]
            o["score"] = st * o["label"] + rng.normal(0, 1, len(o))
            o = E.attach_eval_cols(o, df, ["K12", "K15"])
            prod[a] = o
            shifted[(0, a)] = o
        cv = {"mean_pr_auc": 0.3, "std": 0.02, "mean_roc_auc": 0.6, "fold_scores": [0.3, 0.3]}
        res = {"recs": {a: {"params": {"n_estimators": 10}, "_cv": cv, "_seconds": 1} for a in ("K12", "K15")},
               "params": {a: {"n_estimators": 10} for a in ("K12", "K15")}, "prod": prod, "shifted": shifted}
        stamp = {"built_utc": "2026-10-11 12:00 UTC", "rows": len(df), "date_max": "2025-12-30", "ks": [1.2, 1.5], "labels": []}
        with mock.patch("builtins.print"):
            s = E.report(res, df, [1.2, 1.5], [0], stamp, "lgbm")
        self.assertEqual(s["algo"], "lgbm")
        self.assertEqual(set(s["labels"]), {"K12", "K15"})
        self.assertLess(s["labels"]["K15"]["pos_rate"], s["labels"]["K12"]["pos_rate"])
        self.assertIn("y_K15", s["production_oof"]["K12"])
        self.assertIn("y_K12_pr_K15-K12", s["windows"])
        self.assertIn("y_K15_roc_K15-K12", s["windows"])
        self.assertEqual(s["windows"]["y_K12_pr_K15-K12"]["n"], 3)              # 3窓（ずらし0だけ）
        self.assertIn("K15->K12", s["rank"])
        self.assertIn("K12->K15", s["rank"])
        self.assertTrue(0 <= s["rank"]["K15->K12"]["top10_in_b"] <= 1)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e78test_k_lgbm_rank_agreement.csv")))
        kinds = {r["kind"] for r in s["shifts"]}
        self.assertTrue({"thr95", "thr90", "match95", "match90", "top10_within", "within_date"} <= kinds)
        self.assertEqual({r["arm"] for r in s["monthly"]}, {"K12", "K15"})
        with open(os.path.join(self.dir, "e78test_k_lgbm_summary.json"), encoding="utf-8") as fh:
            self.assertIn("monthly", json.load(fh))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e78test_k_lgbm_monthly.csv")))


class TestMain(unittest.TestCase):
    def test_requires_base_k(self):
        with mock.patch("builtins.print"):
            with self.assertRaises(SystemExit):
                E.main(["--ks", "1.5,1.8"])
            with self.assertRaises(SystemExit):
                E.main(["--ks", "1.2,1.21"])
            with self.assertRaises(SystemExit):
                E.main(["--algos", "mlp"])


if __name__ == "__main__":
    unittest.main()
