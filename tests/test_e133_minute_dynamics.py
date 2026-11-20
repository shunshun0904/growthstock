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


def frame(seed=0, effect=1.2, per_day=8):
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
    for c in MD.COLUMNS:
        rows[c] = rng.normal(size=n)
    rows["vp_dyn_mean_shift"] = info
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

    def test_nothing_when_dynamics_carry_no_information(self):
        out, _ = self.run_report(frame(seed=2, effect=0.0))
        read = out["reading"]["dyn/logit"]
        self.assertFalse(read["単独で当たる"])
        self.assertFalse(read["今のモデルにない情報"])


if __name__ == "__main__":
    unittest.main()
