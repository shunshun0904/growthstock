#!/usr/bin/env python3
"""
実験133（research/exp/e133_minute_dynamics.py）の部品。データには触らない（作りものの行で確かめる）。

- 時間順の窓: 2025-05-01 から3か月ずつ。学習はテストの始まりの 28暦日前まで
- ロジスティックの前処理は学習側の値だけで決める（テスト側の値を変えても学習側の変換は変わらない）
- 一通り: A が使っていない情報（推移の列）でラベルが決まるなら、単独で当たり・A に足して上積みがあり・A の上位を絞り込める。
  ラベルが推移の列と無関係なら、推移だけのモデルはどれにも当てはまらない
- 公開ログに銘柄コードを出さない

  python3 -m unittest tests.test_e133_minute_dynamics
"""
import io
import re
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e133_minute_dynamics as E  # noqa: E402
import minute_dynamics as MD  # noqa: E402

MINUTE_COLS = [f"vp_minute_x{j}" for j in range(4)]
FAST = {**E.LGBM, "n_estimators": 30}
FAST_ALT = {**E.LGBM_ALT, "n_estimators": 30}


def frame(seed=0, effect=1.2, per_day=8, info_col="vp_dyn_mean_shift"):
    """2024-12〜2026-10 の候補の作りもの。A の予測は signal だけ、ラベルは signal と推移の列（mean_shift）で決まる。"""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2024-12-02", "2026-10-21"), per_day)
    n = len(dates)
    signal = rng.normal(size=n)
    info = rng.normal(size=n)
    logit = -1.5 + 0.8 * signal + effect * info
    rows = {"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": dates,
            "label": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float),
            "score": 1 / (1 + np.exp(-(-1.5 + 0.8 * signal)))}
    for c in MD.COLUMNS + MD.SHAPE_DYN_COLUMNS:
        rows[c] = rng.normal(size=n)
    rows[info_col] = info
    for c in MINUTE_COLS + list(E.TECH_COLUMNS):
        rows[c] = rng.normal(size=n)
    return pd.DataFrame(rows)


class TestParts(unittest.TestCase):
    def test_windows(self):
        wins = E.windows(pd.Series(pd.to_datetime(["2024-12-02", "2026-10-21"])))
        self.assertEqual([w[1] for w in wins], list(pd.to_datetime(["2025-05-01", "2025-08-01", "2025-11-01", "2026-02-01",
                                                                     "2026-05-01", "2026-08-01"])))
        self.assertEqual(wins[0][0], pd.Timestamp("2025-04-03"))                 # 28暦日前
        self.assertEqual(wins[0][2], pd.Timestamp("2025-07-31"))
        self.assertEqual(wins[-1][2], pd.Timestamp("2026-10-21"))                # 最後は表の終わりまで

    def test_prep_uses_train_only(self):
        tr = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 100.0], "b": [5.0, 5.0, 5.0, 5.0, 5.0]})
        te1 = pd.DataFrame({"a": [3.0, 1e6], "b": [5.0, 5.0]})
        te2 = pd.DataFrame({"a": [3.0, -1e6], "b": [5.0, 5.0]})
        x1, y1, used = E._prep(tr, te1, ["a", "b"])
        x2, y2, _ = E._prep(tr, te2, ["a", "b"])
        np.testing.assert_allclose(x1, x2)                                     # 学習側はテスト側に左右されない
        self.assertAlmostEqual(y1[0, 0], y2[0, 0])
        self.assertEqual(used, ["a", "b"])
        self.assertTrue(np.isfinite(x1).all())

    def test_rank_within(self):
        r = E.rank_within(np.array([3.0, 1.0, 2.0, 10.0, 20.0]), np.array([0, 0, 0, 1, 1]))
        np.testing.assert_allclose(r, [1.0, 1 / 3, 2 / 3, 0.5, 1.0])


def inverse_frame(seed=0, effect=1.5, per_day=12):
    """全体では x が高いほど正例が多いが、A の上位（signal が高い行）の中では x が高いほど正例が少ない作りもの。"""
    rows = frame(seed=seed, effect=0.0, per_day=per_day)
    rng = np.random.default_rng(seed + 100)
    n = len(rows)
    signal = np.log(rows["score"] / (1 - rows["score"])).to_numpy() + 1.5          # frame の A の予測の logit に戻す
    x = rng.normal(size=n)
    high = signal > np.quantile(signal, 0.8)
    logit = -1.5 + 0.8 * signal + 0.5 * x - effect * x * high
    rows["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    for c in ("vp_dyn_mean_shift", "vp_shp_skew_change"):
        rows[c] = x + 0.3 * rng.normal(size=n)
    return rows


class TestInverse(unittest.TestCase):
    def run_check(self, rows):
        with mock.patch.object(E, "LGBM", FAST), redirect_stdout(io.StringIO()) as buf:
            out = E.inverse_check(rows, MINUTE_COLS, n_perm=99)
        return out, buf.getvalue()

    def test_finds_the_inverse_relation_inside_the_top(self):
        out, log = self.run_check(inverse_frame())
        c = out["confirm"]
        self.assertGreater(c["kept_pos"], c["cut_pos"])
        self.assertLess(c["p"], 0.05)
        self.assertGreaterEqual(c["windows_same_direction"], 2)
        self.assertTrue(out["reading"]["inverse_is_real"])
        self.assertIn(out["chosen"], {f"{a}/{b}" for a, b in E.INVERSE_CANDIDATES})
        self.assertIn("後半の窓で確かめる", log)

    def test_no_inverse_when_absent(self):
        out, _ = self.run_check(inverse_frame(seed=5, effect=0.0))
        self.assertFalse(out["reading"]["inverse_is_real"])


def copy_frame(seed=4, per_day=8):
    """ラベルは A の点（本当の確率）だけで決まり、推移の列の1つは A の情報の薄い写し（A の signal ＋ 雑音）。"""
    rows = frame(seed=seed, effect=0.0, per_day=per_day)
    rng = np.random.default_rng(seed + 200)
    signal = np.log(rows["score"] / (1 - rows["score"])).to_numpy() + 1.5
    rows["vp_dyn_mean_shift"] = signal + 0.5 * rng.normal(size=len(rows))
    return rows


def window_folds(rows):
    wins = E.windows(rows["Date"])
    fold = np.full(len(rows), -1)
    for k, (_, s, e) in enumerate(wins):
        fold[((rows["Date"] >= s) & (rows["Date"] <= e)).to_numpy()] = k
    return wins, fold


class TestIncremental(unittest.TestCase):
    def run_check(self, rows, n_perm=49):
        with mock.patch.object(E, "LGBM", FAST), redirect_stdout(io.StringIO()) as buf:
            out = E.incremental_check(rows, MINUTE_COLS, n_perm=n_perm)
        return out, buf.getvalue()

    def test_date_normal(self):
        codes = np.array([0, 0, 0, 1, 2, 2])
        v = E.date_normal(np.array([3.0, 1.0, 2.0, 9.0, 5.0, 4.0]), codes)
        self.assertEqual(v[3], 0.0)                                       # 1行だけの日は 0
        self.assertTrue(v[1] < v[2] < v[0])                               # 日付の中の順位を保つ
        self.assertAlmostEqual(v[0], -v[1])
        self.assertAlmostEqual(v[4], -v[5])
        w = E.date_normal(np.array([3.0, 1.0, 2.0, -100.0, 50.0, 4.0]), codes)
        np.testing.assert_allclose(v, w)                                  # ほかの日付の値に左右されない

    def test_fast_day_auc_matches_day_auc(self):
        rng = np.random.default_rng(1)
        codes = np.repeat(np.arange(40), 7)
        y = (rng.random(len(codes)) < 0.3).astype(int)
        s = np.round(rng.normal(size=len(codes)), 1)                      # 同じ値を含む
        self.assertAlmostEqual(E.fast_day_auc(y, s, codes), E.day_auc(y, s, codes), places=12)

    def test_walk_forward_uses_only_earlier_windows(self):
        rows = frame(seed=1)
        wins, fold = window_folds(rows)
        dates = rows["Date"].to_numpy()
        rng = np.random.default_rng(0)
        y = rows["label"].to_numpy(dtype=float)
        la, v = rng.normal(size=len(y)), rng.normal(size=len(y))
        ev = list(range(1, len(wins)))
        st1, coefs = E.walk_forward(y, la, v, fold, dates, wins, ev)
        self.assertEqual(sorted(coefs), ev)
        self.assertTrue(np.isnan(st1[fold <= 0]).all())                    # 最初の窓は学習する行が無い
        y2 = y.copy()
        y2[fold == 3] = 1 - y2[fold == 3]                                  # 窓 3 のラベルを変える
        st2, _ = E.walk_forward(y2, la, v, fold, dates, wins, ev)
        early = (fold >= 1) & (fold <= 3)
        np.testing.assert_allclose(st1[early], st2[early])                 # 窓 3 までの点は窓 3 のラベルを使わない
        self.assertFalse(np.allclose(st1[fold == 4], st2[fold == 4]))      # 窓 4 からは使う
        y3 = y.copy()
        y3[fold == 5] = 1 - y3[fold == 5]                                  # 最後の窓のラベルはどの点にも使わない
        st3, _ = E.walk_forward(y3, la, v, fold, dates, wins, ev)
        np.testing.assert_allclose(st1[fold >= 1], st3[fold >= 1])

    def test_finds_information_that_a_lacks(self):
        out, log = self.run_check(frame())
        r = out["candidates"]["dyn/logit"]
        self.assertGreater(r["coef"], 0)
        self.assertLess(r["p_dll_adj"], 0.05)
        self.assertGreater(r["stack_gain"], 0)
        self.assertLess(r["p_stack_adj"], 0.05)
        self.assertGreaterEqual(r["stack_up"], 3)
        self.assertGreater(r["gain_upper"], r["gain_point"])
        self.assertTrue(out["reading"]["information"])
        self.assertTrue(out["reading"]["usable"])
        self.assertIn("dyn/logit", out["reading"]["information_sets"])
        self.assertEqual(len(out["candidates"]), 14)
        for title in ("1. 残差", "2. 重みを過去の窓で決めて A に足す", "読み"):
            self.assertIn(title, log)
        self.assertIsNone(re.search(r"(?<![\d.,])10[0-4]\d(?!\d)", log))     # 銘柄コードを出さない

    def test_copy_of_a_predicts_but_is_not_information(self):
        out, log = self.run_check(copy_frame())
        r = out["candidates"]["dyn/logit"]
        self.assertGreater(r["day_auc"], 0.55)                             # 単独では当たる
        self.assertGreater(r["within_corr_a"], 0.5)                        # A と重なる
        self.assertFalse(out["reading"]["information"])                    # でも A にない情報ではない
        self.assertFalse(out["reading"]["usable"])
        self.assertIn("見つからない", log)


class TestReport(unittest.TestCase):
    def run_report(self, rows):
        with mock.patch.object(E, "LGBM", FAST), mock.patch.object(E, "LGBM_ALT", FAST_ALT), \
                redirect_stdout(io.StringIO()) as buf:
            out = E.report(rows, MINUTE_COLS, n_perm=49)
        return out, buf.getvalue()

    def test_finds_information_that_a_does_not_use(self):
        out, log = self.run_report(frame())
        m = out["models"]["dyn/logit"]
        self.assertGreater(m["roc"], 0.6)
        self.assertLess(m["p_roc"], 0.05)
        self.assertGreater(m["blend_gain"], 0)
        self.assertLess(m["p_blend"], 0.05)
        self.assertGreater(m["top_a_auc"], 0.55)
        self.assertGreater(m["top_a_upper_pos"], m["top_a_lower_pos"])
        read = out["reading"]["dyn/logit"]
        self.assertTrue(read["単独で当たる"])
        self.assertTrue(read["今のモデルにない情報"])
        self.assertTrue(read["A の上位を絞り込める"])
        self.assertEqual(len(out["windows"]), 6)
        coef = out["coefficients"]["dyn"]
        self.assertEqual(max(coef, key=lambda c: abs(coef[c]["coef"])), "vp_dyn_mean_shift")
        self.assertGreater(len(out["rule"]["leaves"]), 1)
        for title in ("1. 単独の当たり", "2. A との重なり", "3. A の上位", "4. 統計モデルの係数", "5. ルール", "読み"):
            self.assertIn(title, log)
        self.assertNotIn("1000", log.replace("1,000", ""))                      # 銘柄コードを出さない

    def test_shape_only_sets(self):
        rows = frame(seed=3, info_col="vp_shp_skew_change")
        with mock.patch.object(E, "LGBM", FAST), mock.patch.object(E, "LGBM_ALT", FAST_ALT), \
                redirect_stdout(io.StringIO()) as buf:
            out = E.report(rows, MINUTE_COLS, n_perm=49, sets=E.shape_sets(MINUTE_COLS), primary=E.PRIMARY_SHAPE,
                           coef_sets=("shp", "shp_tech"), rule_set="shp_tech", need=MD.SHAPE_DYN_COLUMNS,
                           title="形だけの推移")
        self.assertEqual(set(out["reading"]), {"shp/logit", "shp_tech/lgbm"})
        self.assertTrue(out["reading"]["shp/logit"]["単独で当たる"])
        self.assertTrue(out["reading"]["shp/logit"]["今のモデルにない情報"])
        coef = out["coefficients"]["shp"]
        self.assertEqual(max(coef, key=lambda c: abs(coef[c]["coef"])), "vp_shp_skew_change")
        self.assertEqual(set(E.shape_sets(MINUTE_COLS)["shp"]), set(MD.SHAPE_DYN_COLUMNS))
        self.assertIn("形だけの推移", buf.getvalue())

    def test_nothing_when_dynamics_carry_no_information(self):
        out, _ = self.run_report(frame(seed=2, effect=0.0))
        read = out["reading"]["dyn/logit"]
        self.assertFalse(read["単独で当たる"])
        self.assertFalse(read["今のモデルにない情報"])


if __name__ == "__main__":
    unittest.main()
