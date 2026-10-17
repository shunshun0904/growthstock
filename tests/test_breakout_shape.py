#!/usr/bin/env python3
"""
抜けた土台の形の2列（build_dataset._breakout_shape。実験101 で検証し、2026-10-17 に運用者の決定で本番へ。
docs/MODEL_ADOPTION_RULES.md §62）。データには触らない（小さな日足の表で確かめる）。

固定すること
  - 手で数えた値: 抜いた高値の古さ（同じ値なら一番新しい行）・前回の更新日の高値に対する今回の高値・長い下げの後の戻り
  - add_breakout_context の2列は、実験101 の shape_columns と更新日の全行で同じ値（欠測・窓の端・銘柄の切れ目・同じ値の高値）
  - 更新日でない行は両方欠測。銘柄の最初の更新日は prev_nh_ratio だけ欠測。t より先の行を変えても値は変わらない
  - 241列の組 = 239列の組の後ろに2列（実験101 の腕 C と同じ並び）。順位版は作らない
  - 線形の前処理の型（日数・正の比率）、日本語名、重要度の大区分
  - 日次予測の採点は、モデルごとに学習した列で行う（列の組が違うモデルも飛ばさない。学習した列が無いモデルだけ飛ばす）

  python3 tests/test_breakout_shape.py
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e101_base_shape_ab as E  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402
import importance_report as IR  # noqa: E402
import linear_preprocess as LP  # noqa: E402
import models as M  # noqa: E402
import predict_daily as PD  # noqa: E402

NEW = ["broken_high_age", "prev_nh_ratio"]


def panel(highs_by_code, w):
    """銘柄ごとの日中の高値から、add_breakout_context に渡せる表を作る（それまでの w 行の最大を超えた日が更新日）。"""
    rows = []
    for code, highs in highs_by_code.items():
        for i, h in enumerate(highs):
            rows.append({"Code": code, "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                         "high": float(h), "close": float(h), "low": float(h) * 0.98, "vol_20d": 2.0})
    df = pd.DataFrame(rows)
    g = df.groupby("Code", sort=False)
    prior = g["high"].transform(lambda s: s.rolling(w, min_periods=1).max().shift(1))
    df["high52w_prior"] = prior.where(g.cumcount() >= w)          # w 行の履歴がたまってから
    df["is_new_high"] = (df["high"] > df["high52w_prior"]) & df["high52w_prior"].notna()
    return df


def context(df, w):
    with mock.patch.object(B, "HIGH_WINDOW", w):
        return B.add_breakout_context(df.copy())


class TestHandCounted(unittest.TestCase):
    """窓 3 行で手で数えた値。"""

    def setUp(self):
        self.df = context(panel({"10010": [5, 7, 6, 8, 4, 9, 9, 10],
                                 "10020": [5, 7, 6, 8, 4, 3, 2, 3, 3.5, 3.6]}, 3), 3)

    def row(self, code, i):
        return self.df[self.df["Code"] == code].iloc[i]

    def test_first_new_high(self):
        r = self.row("10010", 3)                # 窓 [5, 7, 6] の最大 7（2行前）を 8 で抜いた。前回の更新は無い
        self.assertTrue(r["is_new_high"])
        self.assertEqual(r["broken_high_age"], 2.0)
        self.assertTrue(np.isnan(r["base_length"]))
        self.assertTrue(np.isnan(r["prev_nh_ratio"]))

    def test_previous_high_and_ties(self):
        r = self.row("10010", 5)                # 窓 [6, 8, 4] の最大 8（前回の更新日・2行前）を 9 で抜いた
        self.assertEqual((r["broken_high_age"], r["base_length"]), (2.0, 2.0))
        self.assertAlmostEqual(r["prev_nh_ratio"], 9 / 8 * 100)
        self.assertFalse(self.row("10010", 6)["is_new_high"])        # 9 は 9 を超えない
        r = self.row("10010", 7)                # 窓 [4, 9, 9] の最大 9 は2つ。新しい方（1行前）から数える
        self.assertEqual((r["broken_high_age"], r["base_length"]), (1.0, 2.0))
        self.assertAlmostEqual(r["prev_nh_ratio"], 10 / 9 * 100)

    def test_long_base_recovery(self):
        r = self.row("10020", 8)                # 8 で更新した後に 2 まで下げ、窓 [3, 2, 3] の最大 3 を 3.5 で抜いた
        self.assertEqual((r["broken_high_age"], r["base_length"]), (1.0, 5.0))
        self.assertAlmostEqual(r["prev_nh_ratio"], 3.5 / 8 * 100)      # 前回の更新の高値に届かない戻り（100 未満）
        r = self.row("10020", 9)
        self.assertEqual((r["broken_high_age"], r["base_length"]), (1.0, 1.0))
        self.assertAlmostEqual(r["prev_nh_ratio"], 3.6 / 3.5 * 100)

    def test_only_new_high_rows(self):
        off = self.df[~self.df["is_new_high"]]
        self.assertTrue(off[NEW].isna().all().all())
        self.assertFalse(self.df.loc[self.df["is_new_high"], "broken_high_age"].isna().any())


class TestSameAsExperiment101(unittest.TestCase):
    """乱数の日足（欠測・同じ値・銘柄の切れ目あり）で、実験101 の shape_columns と更新日の全行で一致する。"""

    def test_random_panels(self):
        rng = np.random.default_rng(20261017)
        for w in (3, 5, 12):
            highs = {}
            for k in range(6):
                h = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.04, 90))))   # 丸めて同じ値の高値を作る
                h[rng.random(90) < 0.05] = np.nan                                  # 売買の無い日
                highs[f"{10010 + k * 10}"] = h
            df = context(panel(highs, w), w)
            nh = df["is_new_high"].to_numpy()
            self.assertGreater(int(nh.sum()), 20)
            want = E.shape_columns(df[["Code", "Date"]], df, w=w)
            for c in NEW:
                a = df[c].to_numpy()[nh]
                b = want[c].to_numpy()[nh]
                same = (a == b) | (np.isnan(a) & np.isnan(b))
                self.assertTrue(same.all(), (w, c, a[~same][:5], b[~same][:5]))
            self.assertTrue(df.loc[~nh, NEW].isna().all().all())
            # 銘柄の最初の更新日: 抜いた高値の古さはあり、前回の更新日の高値に対する比は欠測
            first = df[df["is_new_high"]].groupby("Code").head(1)
            self.assertTrue(first["prev_nh_ratio"].isna().all())
            self.assertFalse(first["broken_high_age"].isna().any())

    def test_future_rows_do_not_change_values(self):
        w = 5
        base = {"10010": [5, 7, 6, 8, 4, 9, 9, 10, 3, 11, 12, 2, 13],
                "10020": [3, 4, 5, 6, 7, 8, 9, 10]}
        a = context(panel(base, w), w)
        later = {k: list(v) for k, v in base.items()}
        later["10010"][-2:] = [50, 60]                        # 後ろの2行だけ変える
        b = context(panel(later, w), w)
        keep = a.index[:len(base["10010"]) - 2]
        pd.testing.assert_frame_equal(a.loc[keep, NEW], b.loc[keep, NEW])


class TestColumnSet(unittest.TestCase):
    def test_preset_is_239_plus_two(self):
        base = F.columns("all_plus_prog_listing_vol_book")
        new = F.columns("all_plus_prog_listing_vol_book_shape")
        self.assertEqual(len(base), 239)
        self.assertEqual(new, base + NEW)                        # 実験101 の腕 C と同じ並び
        self.assertEqual(F.GROUPS["breakout_shape"], NEW)

    def test_dataset_builds_them_without_rank_versions(self):
        self.assertTrue(set(NEW) <= set(F.all_columns()))
        self.assertFalse(set(NEW) & set(F.RAW_FOR_RANK))
        self.assertFalse({f"{c}_r" for c in NEW} & set(F.all_columns()))

    def test_linear_kinds(self):
        self.assertEqual(LP.kind_of("broken_high_age"), "count")
        self.assertEqual(LP.kind_of("prev_nh_ratio"), "pos")
        self.assertEqual(LP.unclassified(F.columns("all_plus_prog_listing_vol_book_shape")), [])

    def test_names_and_groups(self):
        for c in NEW:
            self.assertIn(c, FD.NAME_JA)
            self.assertIn(c, FD.COL_JA)
            self.assertEqual(IR.macro_of(c), "price")
        self.assertIn("breakout_shape", [g for _, gs, _ in FD.MACRO for g in gs])


class FakeModel:
    """渡された列の和をロジスティックにしたものを確率として返す（どの列で採点したかが値で分かる）。"""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.asarray(X, dtype=float).sum(axis=1)))
        return np.c_[1.0 - p, p]


class TestScoreOthersUsesEachModelsColumns(unittest.TestCase):
    def test_each_model_scores_with_its_own_columns(self):
        cand = pd.DataFrame({"Code": ["10010", "10020"], "Date": pd.to_datetime(["2026-10-16"] * 2),
                             "a": [0.1, 0.2], "b": [0.3, -0.4], "c": [1.0, 2.0]})
        metas = {"lgbm_k15": {"features": ["a", "b", "c"], "name": "LightGBM（1.5σ）"},   # 列の組が基準モデルと違う
                 "xgb": {"features": ["a", "b"], "name": "XGBoost"},                    # 基準モデルと同じ
                 "cat": {"features": ["a", "b", "zz"], "name": "CatBoost"}}             # 学習した列がデータに無い

        def load(algo, root):
            return FakeModel(), dict(metas[algo])

        with mock.patch.object(M, "available", return_value=["lgbm", "xgb", "cat", "lgbm_k15"]), \
                mock.patch.object(M, "load", side_effect=load):
            scores, info = PD.score_others(cand, ["a", "b"], "unused")
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))                 # noqa: E731
        self.assertEqual(sorted(scores), ["lgbm_k15", "xgb"])    # 基準モデルはここで採点しない・列が無い cat は飛ばす
        np.testing.assert_allclose(scores["lgbm_k15"], sig(cand[["a", "b", "c"]].sum(axis=1).to_numpy()))
        np.testing.assert_allclose(scores["xgb"], sig(cand[["a", "b"]].sum(axis=1).to_numpy()))
        self.assertEqual([x["algo"] for x in info], ["xgb", "lgbm_k15"])


if __name__ == "__main__":
    unittest.main()
