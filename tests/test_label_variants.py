#!/usr/bin/env python3
"""
ラベル違いのモデル（到達しきい値 1.5σ / 1.8σ で学習した lgbm / xgb / cat / logit の8本。tuning_multi.VARIANTS。
運用者の指示 2026-10-11。同日夜に xgb / cat / logit の版を足した）。

- 名前と元の学習器の対応（base_algo / label_k / is_tree）。探索・学習は元の学習器（lgbm）で組み、study 名は別
- 画面・スプレッドシートの並び（models.LINEUP。1.2σ の木3本 → 1.5σ の木3本。1.8σ の4本は 2026-10-24 に、ロジスティック
  回帰（logit・logit_k15）は 2026-11-02 の予測から外した）。train_multi の既定は EXTRA（日曜）で、ラベル違いは名指ししたとき
  だけ（土曜）。本番の合議（ALGOS）には入れない
- ラベルの引き直し（build_dataset.rise_label_for_k）: k = 1.2 なら本番のラベルと一致、k を上げると正例は部分集合
- train_multi.untuned は木の本数をラベル違いでも見る
- predict_daily.agreement: 本番の一致（ALGOS）とラベル違いの一致（VARIANTS）を別に数える
- export_sheets: L15% / L18% / L15スコア / L18スコア と「ラベル違い一致」の列

  python3 tests/test_label_variants.py
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import export_sheets as ES  # noqa: E402
import features as F  # noqa: E402
import model_lineup as ML  # noqa: E402
import models as M  # noqa: E402
import predict_daily as PD  # noqa: E402
import train_multi as TR  # noqa: E402
import tuning_multi as TM  # noqa: E402


class TestNames(unittest.TestCase):
    def test_variants(self):
        k15 = ("lgbm_k15", "xgb_k15", "cat_k15", "logit_k15")
        k18 = ("lgbm_k18", "xgb_k18", "cat_k18", "logit_k18")
        self.assertEqual(tuple(TM.VARIANTS), k15 + k18)                            # 1.5σ の4本 → 1.8σ の4本
        self.assertEqual(TM.variants_of(1.5), k15)
        self.assertEqual(TM.variants_of(1.8), k18)
        for a in k15 + k18:
            self.assertEqual(TM.base_algo(a), a.split("_")[0])
            self.assertEqual(TM.label_k(a), 1.5 if a in k15 else 1.8)
            self.assertEqual(TM.is_tree(a), a.split("_")[0] in TM.TREE_ALGOS)
        self.assertEqual(TM.base_algo("xgb"), "xgb")
        self.assertIsNone(TM.label_k("lgbm"))
        self.assertFalse(TM.is_tree("logit"))
        self.assertEqual(TM.preprocess_version("lgbm_k15"), "")                  # 木なので前処理なし
        self.assertEqual(TM.preprocess_version("logit_k15"), TM.preprocess_version("logit"))  # 線形は元の前処理の版
        self.assertEqual(M.VARIANTS, TM.VARIANTS)                                  # 過去の実験の記録を読むために 1.8σ も残す
        # 1.8σ の4本は 2026-10-24 に外した（運用者の依頼）。ロジスティック回帰（logit・logit_k15）は運用者の指示（2026-10-29）で
        # 2026-11-02 の予測から外した: 学習・毎日の採点・画面・シートは 1.2σ・1.5σ の木の各3本だけ
        k15_trees = ("lgbm_k15", "xgb_k15", "cat_k15")
        self.assertEqual(M.ALGOS, ("lgbm", "xgb", "cat"))
        self.assertEqual(M.RETIRED, ("mlp", "logit"))
        self.assertEqual(M.EXTRA, ("xgb", "cat"))
        self.assertEqual(M.RETIRED_VARIANTS, k18 + ("logit_k15",))
        self.assertEqual(M.ACTIVE_VARIANTS, k15_trees)
        self.assertEqual(M.LINEUP, M.ALGOS + k15_trees)
        self.assertEqual(M.MULTI, M.EXTRA + k15_trees)
        for a in k18 + ("logit", "logit_k15", "mlp"):
            self.assertNotIn(a, M.LINEUP)
            self.assertNotIn(a, M.MULTI)
        for a in M.VARIANTS:
            self.assertNotIn(a, M.ALGOS)                                           # 本番の合議には入れない
            self.assertIn(a, M.JA)
            self.assertIn(a, M.SHORT)
            self.assertIn(a, M.NOTE)
        self.assertEqual([M.SHORT[a] for a in k15], ["L15", "X15", "C15", "LR15"])
        self.assertEqual([M.SHORT[a] for a in k18], ["L18", "X18", "C18", "LR18"])
        self.assertEqual(len(set(M.SHORT.values())), len(M.SHORT))                 # 記号は重ならない（シートの列名）

    def test_build_and_study_name(self):
        import lightgbm as lgb
        y = np.array([0, 1, 0, 1])
        m = TM.build("lgbm_k15", {"num_leaves": 7}, y, ["a", "b"])
        self.assertIsInstance(m, lgb.LGBMClassifier)
        self.assertEqual(m.get_params()["n_estimators"], TM.N_ESTIMATORS)
        self.assertEqual(m.get_params()["num_leaves"], 7)
        cols = F.columns(F.DEFAULT_PRESET)
        base = TM.study_name("lgbm", 5, "2025-08-08", cols)
        v15 = TM.study_name("lgbm_k15", 5, "2025-08-08", cols)
        v18 = TM.study_name("lgbm_k18", 5, "2025-08-08", cols)
        self.assertNotEqual(base, v15)
        self.assertNotEqual(v15, v18)
        self.assertTrue(v15.startswith("lgbm_k15_") and v15.endswith(f"_t{TM.N_ESTIMATORS}"))   # 木の本数も入る

    def test_logit_variant_uses_the_base_preprocessing(self):
        """ロジスティック回帰の 1.5σ / 1.8σ 版は、logit の前処理の版（v2）で組む。"""
        for a in ("logit_k15", "logit_k18"):
            self.assertEqual(TM.base_algo(a), "logit")
            self.assertFalse(TM.is_tree(a))
            self.assertEqual(TM.preprocess_version(a), TM.preprocess_version("logit"))
            self.assertEqual(TM.preprocess_version(a), "v2")
            cols = F.columns(F.DEFAULT_PRESET)
            name = TM.study_name(a, 5, "2025-08-08", cols)
            self.assertTrue(name.startswith(a + "_") and name.endswith("_v2"))          # 木の本数は入らず、前処理の版が入る
            self.assertNotEqual(name, TM.study_name("logit", 5, "2025-08-08", cols))
        y = np.array([0, 1, 0, 1, 1, 0])
        m = TM.build("logit_k15", {"C": 0.5, "penalty": "l2"}, y, ["a", "b"])
        self.assertTrue(hasattr(m, "predict_proba"))
        self.assertEqual(type(m).__name__, "Pipeline")
        # xgb / cat の版は元の学習器で組む
        import xgboost as xgb
        from catboost import CatBoostClassifier
        self.assertIsInstance(TM.build("xgb_k18", {"max_depth": 3}, np.array([0, 1, 0, 1]), ["a", "b"]), xgb.XGBClassifier)
        self.assertIsInstance(TM.build("cat_k15", {"depth": 3}, np.array([0, 1, 0, 1]), ["a", "b"]), CatBoostClassifier)

    def test_available_lists_variants_in_lineup_order(self):
        d = tempfile.mkdtemp()
        try:
            for a in ("lgbm_k15", "xgb"):
                os.makedirs(M.sub_dir(a, d))
                open(os.path.join(M.sub_dir(a, d), "model.joblib"), "w").close()
            self.assertEqual(M.available(d), ["xgb", "lgbm_k15"])
            os.makedirs(M.sub_dir("cat_k15", d))
            open(os.path.join(M.sub_dir("cat_k15", d), "model.joblib"), "w").close()
            self.assertEqual(M.available(d), ["xgb", "lgbm_k15", "cat_k15"])      # LINEUP の順
            # 外したモデル（1.8σ は 2026-10-24、ロジスティック回帰の 1.2σ・1.5σ は 2026-11-02 の予測から）は、Release から
            # 落ちてきて置き場にあっても採点しない
            for a in ("logit_k18", "logit_k15", "logit"):
                os.makedirs(M.sub_dir(a, d))
                open(os.path.join(M.sub_dir(a, d), "model.joblib"), "w").close()
            self.assertEqual(M.available(d), ["xgb", "lgbm_k15", "cat_k15"])
            r = ML.row("cat_k15", d)
            self.assertEqual((r["state"], r["name"]), ("未学習", M.JA["cat_k15"]))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"vol_20d": rng.uniform(0.8, 5.0, n), "future_rise": rng.normal(0.05, 0.15, n),
                       "uptrend_end": (rng.random(n) < 0.7).astype(float)})
    df["end_level"] = df["future_rise"] - rng.uniform(0.0, 0.12, n)
    need, end_need = B.rise_thresholds(df["vol_20d"], B.DEFAULT_RISE)
    df["label"] = ((df["future_rise"] >= need.to_numpy()) & (df["end_level"] >= end_need.to_numpy())
                   & (df["uptrend_end"] == 1.0)).astype(float)
    return df


class TestRelabel(unittest.TestCase):
    def test_k12_matches_and_higher_k_is_subset(self):
        df = frame()
        y12 = B.rise_label_for_k(df, B.VOL_NORM_K)
        self.assertTrue((y12 == df["label"]).all())
        y15 = B.rise_label_for_k(df, 1.5)
        y18 = B.rise_label_for_k(df, 1.8)
        self.assertLess(y15.sum(), y12.sum())
        self.assertLess(y18.sum(), y15.sum())
        self.assertTrue(((y15 == 1) <= (y12 == 1)).all())
        self.assertTrue(((y18 == 1) <= (y15 == 1)).all())
        # 判定できない行は NaN（元のラベルが未確定 / vol_20d 欠測）
        d2 = df.copy()
        d2.loc[0, "label"] = np.nan
        d2.loc[1, "vol_20d"] = np.nan
        y = B.rise_label_for_k(d2, 1.5)
        self.assertTrue(np.isnan(y.iloc[0]) and np.isnan(y.iloc[1]))
        self.assertEqual(y.iloc[2:].isna().sum(), 0)

    def test_guards(self):
        df = frame()
        with self.assertRaises(SystemExit):
            B.rise_label_for_k(df.drop(columns=["end_level"]), 1.5)
        with self.assertRaises(SystemExit):
            B.rise_label_for_k(df, 1.5, B.RiseConfig(keep_days=10))


class TestTrainMulti(unittest.TestCase):
    def test_untuned_checks_trees_for_variants(self):
        cols = ["a", "b", "c"]
        sig = F.signature(cols)
        ok = {"lgbm_k15": {"_cv": {"features_sig": sig, "n_features": 3, "n_estimators": TM.N_ESTIMATORS}}}
        self.assertEqual(TR.untuned(["lgbm_k15"], ok, cols), {})
        bad = {"lgbm_k15": {"_cv": {"features_sig": sig, "n_features": 3, "n_estimators": 500}}}
        self.assertIn("本数", TR.untuned(["lgbm_k15"], bad, cols)["lgbm_k15"])
        self.assertIn("lgbm_k18", TR.untuned(["lgbm_k18"], ok, cols))                # 探索が無い


class TestPredict(unittest.TestCase):
    def test_agreement_counts_production_and_variants_separately(self):
        per = {"lgbm": {"pctHistorical": 96.0}, "xgb": {"pctHistorical": 91.0}, "cat": {"pctHistorical": 80.0},
               "logit": {"pctHistorical": None},
               "lgbm_k15": {"pctHistorical": 95.0}, "lgbm_k18": {"pctHistorical": 70.0}}
        self.assertEqual(PD.agreement(per, M.ALGOS), (2, 3))
        self.assertEqual(PD.agreement(per, M.VARIANTS), (1, 2))                   # 無いモデルは数えない
        per.update({"xgb_k15": {"pctHistorical": 92.0}, "logit_k18": {"pctHistorical": 50.0}})
        self.assertEqual(PD.agreement(per, M.VARIANTS), (2, 4))
        self.assertEqual(PD.agreement({}, M.VARIANTS), (0, 0))


class TestSheets(unittest.TestCase):
    def test_columns_and_rows(self):
        self.assertIn("L15%", ES.MODEL_COLS)
        self.assertIn("L15スコア", ES.SCORE_COLS)
        for c in ("X15%", "C15%"):
            self.assertIn(c, ES.MODEL_COLS)
        # 1.8σ の4本は 2026-10-24 に、ロジスティック回帰（LR・LR15）は 2026-11-02 の予測から外した: 列を書かない（シートの
        # 見出しは残り、前の行の値もそのまま）
        for c in ("L18%", "X18%", "C18%", "LR18%", "LR%", "LR15%"):
            self.assertNotIn(c, ES.MODEL_COLS)
        for c in ("LR18スコア", "LRスコア", "LR15スコア"):
            self.assertNotIn(c, ES.SCORE_COLS)
        self.assertEqual(len(ES.MODEL_COLS), len(M.LINEUP))
        self.assertEqual(len(M.LINEUP), 6)
        self.assertEqual(ES.AGREE_COLS, [ES.AGREE_COL, ES.AGREE_K_COL])
        pred = {"models": [{"algo": "lgbm", "short": "LGB"}, {"algo": "lgbm_k15", "short": "L15"}],
                "candidates": [{"date": "2026-10-13", "code": "1234", "jqCode": "12340", "name": "x", "rankInDay": 1,
                                "nInDay": 5, "score": 0.5, "band": 9,
                                "byModel": {"lgbm": {"score": 0.5, "pctHistorical": 96.0},
                                            "lgbm_k15": {"score": 0.4, "pctHistorical": 93.0}},
                                "agree90": 1, "nModels": 1, "agreeK90": 1, "nK": 1}]}
        row = ES.rows_from_predictions(pred)[0]
        self.assertEqual(row["L15%"], 93.0)
        self.assertEqual(row["L15スコア"], 0.4)
        self.assertEqual(row[ES.AGREE_COL], "1/1")
        self.assertEqual(row[ES.AGREE_K_COL], "1/1")
        old = {"models": [{"algo": "lgbm", "short": "LGB"}],
               "candidates": [{**pred["candidates"][0], "byModel": {"lgbm": {"score": 0.5, "pctHistorical": 96.0}},
                               "nK": 0}]}
        row = ES.rows_from_predictions(old)[0]
        self.assertNotIn("L15%", row)                                              # 無いモデルの列は作らない
        self.assertIsNone(row[ES.AGREE_K_COL])
        # 外す前に出した予測ファイル（models に 1.8σ がある）でも、1.8σ の列は書かない
        k18 = {"models": pred["models"] + [{"algo": "lgbm_k18", "short": "L18"}],
               "candidates": [{**pred["candidates"][0],
                               "byModel": {**pred["candidates"][0]["byModel"],
                                           "lgbm_k18": {"score": 0.3, "pctHistorical": 90.0}}}]}
        row = ES.rows_from_predictions(k18)[0]
        self.assertEqual(row["L15%"], 93.0)
        self.assertNotIn("L18%", row)
        self.assertNotIn("L18スコア", row)
        # ロジスティック回帰（2026-11-02 の予測から外した）も、外す前の予測ファイルにあっても書かない
        lr = {"models": pred["models"] + [{"algo": "logit", "short": "LR"}, {"algo": "logit_k15", "short": "LR15"}],
              "candidates": [{**pred["candidates"][0],
                              "byModel": {**pred["candidates"][0]["byModel"],
                                          "logit": {"score": 0.2, "pctHistorical": 80.0},
                                          "logit_k15": {"score": 0.2, "pctHistorical": 81.0}}}]}
        row = ES.rows_from_predictions(lr)[0]
        self.assertEqual(row["L15%"], 93.0)
        for c in ("LR%", "LRスコア", "LR15%", "LR15スコア"):
            self.assertNotIn(c, row)


if __name__ == "__main__":
    unittest.main()
