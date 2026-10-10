#!/usr/bin/env python3
"""
探索の分割（research/tuning.py）の「本番の窓と同じ前進分割」（walkforward。2026-10-10 から本番）。

- 検証窓は探索データの最終日から遡って 6か月 × 5本。古い窓が先。窓は重ならず、最後の窓は最終日で終わる
- 訓練はその窓より前の全部で、訓練の最終日と検証の初日の間にエンバーゴ（20営業日 ≒ 29暦日）がある
- 同じ日の行は必ず同じ側（日付で切る）
- 訓練が 36か月に満たない窓は落とす（データが短ければ分割の数が減る）
- 窓の長さ・最初の訓練の長さは本番の out-of-fold（train_production）と同じ値
- cv_folds は名前で方式を選ぶ。本番の既定（PRODUCTION_CV）は層別 year_cap_date のまま（walkforward は候補。実験73 の
  結果を見て運用者が決める）で、tuning_multi の既定も同じ
- tuning_multi の study 名は分割方式が year_cap_date 以外なら名前に入る（前の分割の試行を引き継がない）
- e15_tune_all.why_retune は分割方式が違えば探索し直す

  python3 tests/test_tuning_walkforward.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import train_model as T  # noqa: E402
import train_production as TP  # noqa: E402


def frame(start="2018-04-02", end="2025-08-08", per_day=3, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days) * per_day
    return pd.DataFrame({"Date": np.repeat(days, per_day),
                         "Code": [f"{i % 50:04d}0" for i in range(n)],
                         "label": (rng.random(n) < 0.18).astype(int),
                         "market_cap": rng.lognormal(5, 1, n)})


class TestWalkforward(unittest.TestCase):
    def test_windows_anchor_at_the_end_and_do_not_overlap(self):
        df = frame()
        folds = tuning.walkforward_folds(df, n_splits=5, embargo_days=20)
        self.assertEqual(len(folds), 5)
        dmax = pd.Timestamp(df["Date"].max())
        prev_hi = None
        for k, (tr, va) in enumerate(folds):
            vd = pd.to_datetime(va["Date"])
            td = pd.to_datetime(tr["Date"])
            lo = dmax - pd.DateOffset(months=6 * (5 - k))
            hi = dmax - pd.DateOffset(months=6 * (4 - k))
            self.assertGreater(vd.min(), lo)                                   # 窓は (lo, hi]
            self.assertLessEqual(vd.max(), hi)
            self.assertLessEqual(td.max(), lo - pd.Timedelta(days=29))         # エンバーゴ 20営業日 ≒ 29暦日
            self.assertGreaterEqual(td.max(), lo - pd.Timedelta(days=29) - pd.Timedelta(days=4))
            self.assertEqual(td.min(), pd.Timestamp("2018-04-02"))             # 訓練は最初から（expanding）
            if prev_hi is not None:
                self.assertGreater(vd.min(), prev_hi)                           # 重ならない・古い窓が先
            prev_hi = vd.max()
            # 同じ日の行は同じ側
            self.assertFalse(set(vd.unique()) & set(td.unique()))
        self.assertEqual(pd.to_datetime(folds[-1][1]["Date"]).max(), dmax)     # 最後の窓は最終日で終わる
        self.assertEqual(sum(len(v) for _, v in folds), int((pd.to_datetime(df["Date"])
                                                             > dmax - pd.DateOffset(months=30)).sum()))

    def test_short_history_drops_early_windows(self):
        df = frame(start="2021-06-01", end="2025-08-08")                        # 約4年2か月
        folds = tuning.walkforward_folds(df, n_splits=5, embargo_days=20)
        # 最初の窓（2023-02〜2023-08）の訓練は 2021-06〜2023-01 = 20か月 < 36 → 落ちる
        self.assertLess(len(folds), 5)
        self.assertGreaterEqual(len(folds), 2)
        for tr, _ in folds:
            self.assertGreaterEqual(pd.to_datetime(tr["Date"]).max(),
                                    pd.Timestamp("2021-06-01") + pd.DateOffset(months=36))
        self.assertEqual(tuning.walkforward_folds(frame(start="2024-01-04", end="2025-08-08")), [])

    def test_single_class_window_is_dropped(self):
        df = frame()
        d = pd.to_datetime(df["Date"])
        df.loc[d > pd.Timestamp("2025-02-08"), "label"] = 0                      # 最後の窓に正例が無い
        folds = tuning.walkforward_folds(df, n_splits=5)
        self.assertEqual(len(folds), 4)

    def test_matches_production_oof_shape(self):
        self.assertEqual(tuning.WALKFORWARD_TEST_MONTHS, TP.OOF_TEST_MONTHS)
        self.assertEqual(tuning.WALKFORWARD_MIN_TRAIN_MONTHS, TP.OOF_MIN_TRAIN_MONTHS)
        self.assertEqual(T.EMBARGO_DAYS, 20)

    def test_fold_windows_record(self):
        folds = tuning.walkforward_folds(frame(), n_splits=5)
        w = tuning.fold_windows(folds)
        self.assertEqual(len(w), 5)
        self.assertEqual(set(w[0]), {"valid_from", "valid_to", "train_to", "n_train", "n_valid", "pos_rate"})
        self.assertLess(w[0]["train_to"], w[0]["valid_from"])
        self.assertLess(w[0]["valid_to"], w[1]["valid_from"])


class TestSchemes(unittest.TestCase):
    def test_production_default(self):
        self.assertEqual(tuning.PRODUCTION_CV, "year_cap_date")                 # 本番は層別のまま（§32。検証してから）
        self.assertIn("walkforward", tuning.SCHEMES)
        self.assertEqual(set(tuning.SCHEME_JA), set(tuning.SCHEMES))
        import inspect
        self.assertEqual(inspect.signature(TM.tune).parameters["scheme"].default, tuning.PRODUCTION_CV)

    def test_cv_folds_dispatch(self):
        df = frame()
        wf = tuning.cv_folds(df, "walkforward", n_splits=5, embargo_days=20)
        self.assertEqual(len(wf), 5)
        st = tuning.cv_folds(df, "year_cap_date", n_splits=5, seed=0)
        self.assertEqual(len(st), 5)
        # 層別は訓練側に検証より後の日が入る。前進分割は入らない
        tr, va = st[0]
        self.assertGreater(pd.to_datetime(tr["Date"]).max(), pd.to_datetime(va["Date"]).min())
        for tr, va in wf:
            self.assertLess(pd.to_datetime(tr["Date"]).max(), pd.to_datetime(va["Date"]).min())
        with self.assertRaises(SystemExit):
            tuning.cv_folds(df, "random")

    def test_run_tuning_default_and_choices(self):
        import run_tuning
        import argparse
        # main の中で parser を組むので、--help の文字列から確かめる
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            run_tuning.main(["--help"])
        text = buf.getvalue()
        self.assertIn("walkforward", text)
        self.assertIn("year_cap_date", text)

    def test_study_name_carries_scheme(self):
        cols = ["a", "b"]
        base = TM.study_name("xgb", 5, "2025-08-08", cols)
        self.assertEqual(TM.study_name("xgb", 5, "2025-08-08", cols, "year_cap_date"), base)
        wf = TM.study_name("xgb", 5, "2025-08-08", cols, "walkforward")
        self.assertNotEqual(wf, base)
        self.assertIn("_walkforward", wf)                                        # 木の本数（_t200）が後ろに付く

    def test_why_retune_on_scheme_change(self):
        from e15_tune_all import why_retune
        prev = {"features_sig": "sig", "n_estimators": 200, "train_to": "2025-08-08", "scheme": "year_cap_date"}
        self.assertIsNone(why_retune(prev, "2025-08-08", "sig", 200))
        self.assertIn("分割", why_retune(prev, "2025-08-08", "sig", 200, scheme="walkforward"))
        self.assertIsNone(why_retune({**prev, "scheme": "walkforward"}, "2025-08-08", "sig", 200, scheme="walkforward"))


class TestObjective(unittest.TestCase):
    """探索の目的関数（5分割 CV の中だけ）。pr_auc はそのまま、lift は正例率で割る、excess は正例率を引く。"""

    def test_values(self):
        prs, rates = [0.20, 0.50], [0.10, 0.25]
        self.assertAlmostEqual(tuning.cv_objective(prs, rates, "pr_auc"), 0.35)
        self.assertAlmostEqual(tuning.cv_objective(prs, rates, "lift"), 2.0)            # 2.0 と 2.0 の平均
        self.assertAlmostEqual(tuning.cv_objective(prs, rates, "excess"), 0.175)        # 0.10 と 0.25 の平均
        with self.assertRaises(SystemExit):
            tuning.cv_objective(prs, rates, "f1")
        self.assertEqual(tuning.PRODUCTION_OBJECTIVE, "pr_auc")                           # 本番は PR-AUC のまま
        self.assertEqual(set(tuning.OBJECTIVE_JA), set(tuning.OBJECTIVES))

    def test_lift_equalises_windows(self):
        # 正例率の高い窓が PR-AUC の平均を引っ張る。リフトなら両方の窓が同じ重みになる
        hi = [0.51, 0.20]     # 正例率 23.9% の窓と 9.8% の窓（実験73 の実測に近い）
        rates = [0.239, 0.098]
        lo = [0.45, 0.25]     # 高い窓を少し落とし、低い窓を少し上げた設定（PR-AUC の平均は 0.350 対 0.355）
        self.assertGreater(tuning.cv_objective(hi, rates, "pr_auc"), tuning.cv_objective(lo, rates, "pr_auc"))
        self.assertLess(tuning.cv_objective(hi, rates, "lift"), tuning.cv_objective(lo, rates, "lift"))

    def test_tune_rejects_unknown_objective(self):
        with self.assertRaises(SystemExit):
            tuning.tune(frame(), ["market_cap"], n_trials=1, scheme="walkforward", embargo_days=20,
                        objective="f1", verbose=False)

    def test_study_name_carries_objective(self):
        cols = ["a", "b"]
        base = TM.study_name("xgb", 5, "2025-08-08", cols, "walkforward")
        self.assertEqual(TM.study_name("xgb", 5, "2025-08-08", cols, "walkforward", "pr_auc"), base)
        self.assertIn("_lift", TM.study_name("xgb", 5, "2025-08-08", cols, "walkforward", "lift"))

    def test_why_retune_on_objective_change(self):
        from e15_tune_all import why_retune
        prev = {"features_sig": "sig", "n_estimators": 200, "train_to": "2025-08-08", "scheme": "year_cap_date"}
        self.assertIsNone(why_retune(prev, "2025-08-08", "sig", 200, scheme="year_cap_date", objective="pr_auc"))
        self.assertIn("目的関数", why_retune(prev, "2025-08-08", "sig", 200, scheme="year_cap_date", objective="lift"))


if __name__ == "__main__":
    unittest.main()
