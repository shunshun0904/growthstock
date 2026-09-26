#!/usr/bin/env python3
"""
ボラの分解（build_dataset.add_vol_factors / attach_vol_vs_market、実験51）のテスト。

- 6列とも当日までの値だけで作る（先読みなし）: 将来の足を変えても値が動かない
- 定義どおりの値になる（手計算と一致）
- 分母が 0・欠測なら欠測（無限大を残さない）
- 本番のプリセット（206列）には入っていない。候補のプリセットは T の後ろに6列
- 説明（feature_dict）と大区分（MACRO）に登録されている

  python3 tests/test_vol_factors.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_dataset as B  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402

NEW = ["vol_rel_long", "vol_rel_mkt", "vol_rel_sector", "vol_updown", "vol_rel_short",
       "vol_gap_ratio"]


def bars(n=200, seed=0, code="1301"):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.02, n)
    close = 1000 * np.cumprod(1 + r)
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    d = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Date": d, "Code": code, "AdjO": open_, "AdjH": high, "AdjL": low,
                         "AdjC": close, "AdjVo": 1000.0, "O": open_, "H": high, "L": low,
                         "C": close, "Vo": 1000.0, "Va": close * 1000})


def panel(df):
    return B.price_panel(df.copy())


class TestDefinitions(unittest.TestCase):
    def setUp(self):
        self.p = panel(bars())

    def test_columns_exist_and_are_finite_or_nan(self):
        for c in ["vol_rel_long", "vol_updown", "vol_rel_short", "vol_gap_ratio"]:
            self.assertIn(c, self.p.columns)
            v = self.p[c].to_numpy(dtype=float)
            self.assertFalse(np.isinf(v).any(), c)

    def test_matches_hand_calculation(self):
        p = self.p.reset_index(drop=True)
        i = 150
        r = (p["close"] / p["close"].shift(1) - 1).to_numpy()
        w20, w120, w5 = r[i - 19:i + 1], r[i - 119:i + 1], r[i - 4:i + 1]
        sd = lambda x: np.std(x, ddof=1)  # noqa: E731
        self.assertAlmostEqual(p.loc[i, "vol_rel_long"], sd(w20) / sd(w120), places=9)
        self.assertAlmostEqual(p.loc[i, "vol_rel_short"], sd(w5) / sd(w20), places=9)
        up = np.sqrt(np.mean(np.clip(w20, 0, None) ** 2))
        dn = np.sqrt(np.mean(np.clip(w20, None, 0) ** 2))
        self.assertAlmostEqual(p.loc[i, "vol_updown"], up / dn, places=9)
        gap = (p["open"] / p["close"].shift(1) - 1).to_numpy()[i - 19:i + 1]
        rng = np.log(p["high"] / p["low"]).to_numpy()[i - 19:i + 1]
        self.assertAlmostEqual(p.loc[i, "vol_gap_ratio"], sd(gap) / sd(rng), places=9)

    def test_no_lookahead(self):
        """i より後の足を書き換えても i の値は動かない。"""
        b = bars()
        p1 = panel(b).reset_index(drop=True)
        b2 = b.copy()
        for c in ("AdjO", "AdjH", "AdjL", "AdjC", "O", "H", "L", "C"):
            b2.loc[160:, c] *= 1.5
        p2 = panel(b2).reset_index(drop=True)
        for c in ["vol_rel_long", "vol_updown", "vol_rel_short", "vol_gap_ratio"]:
            np.testing.assert_allclose(p1.loc[:159, c].to_numpy(dtype=float),
                                       p2.loc[:159, c].to_numpy(dtype=float), equal_nan=True)

    def test_short_history_is_missing_not_zero(self):
        p = self.p.reset_index(drop=True)
        self.assertTrue(p.loc[:10, "vol_rel_long"].isna().all())
        self.assertTrue(p.loc[:3, "vol_rel_short"].isna().all())

    def test_only_up_days_gives_missing_updown(self):
        b = bars()
        b["AdjC"] = b["C"] = 1000 * np.cumprod(np.full(len(b), 1.01))   # 毎日上昇
        p = panel(b)
        self.assertTrue(p["vol_updown"].isna().all())


class TestSafeRatio(unittest.TestCase):
    def test_zero_or_missing_denominator(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([0.0, np.nan, -1.0, 2.0])
        out = B._safe_ratio(a, b)
        self.assertTrue(out.iloc[:3].isna().all())
        self.assertEqual(out.iloc[3], 2.0)


class TestVsMarket(unittest.TestCase):
    def test_ratio_to_market_and_sector(self):
        s = pd.DataFrame({"vol_20d": [2.0, 3.0, 1.0], "topix_vol_20": [1.0, 0.0, np.nan],
                          "sector_vol_20": [4.0, 1.5, 2.0]})
        out = B.attach_vol_vs_market(s)
        self.assertEqual(out.loc[0, "vol_rel_mkt"], 2.0)
        self.assertTrue(out.loc[1:, "vol_rel_mkt"].isna().all())
        np.testing.assert_allclose(out["vol_rel_sector"], [0.5, 2.0, 0.5])

    def test_without_sector_column(self):
        s = pd.DataFrame({"vol_20d": [2.0], "topix_vol_20": [1.0]})
        out = B.attach_vol_vs_market(s)
        self.assertTrue(np.isnan(out.loc[0, "vol_rel_sector"]))


class TestRegistration(unittest.TestCase):
    def test_group_preset_and_dictionary(self):
        self.assertEqual(F.GROUPS["vol_factors"], NEW)
        base = F.columns("all_plus_prog_listing")
        for c in NEW:
            self.assertNotIn(c, base, "206列のプリセットには入れない")
            # 2026-09-26 に運用者の決定で本番（239列）に入れた（docs/MODEL_ADOPTION_RULES.md §20）
            self.assertIn(c, F.columns(F.DEFAULT_PRESET))
        self.assertEqual(F.columns("all_plus_prog_listing_vol"), base + NEW)
        self.assertEqual(len(F.columns("all_plus_prog_listing_vol")), 212)
        for c in NEW:
            self.assertNotEqual(FD.describe(c), "（説明未登録）", c)
        macro = {g for _, gs, _ in FD.MACRO for g in gs}
        self.assertIn("vol_factors", macro)

    def test_not_ranked(self):
        """比は全期間で同じ物差しなので、日付内の順位版は作らない。"""
        self.assertNotIn("vol_factors", F.RANKED_GROUPS)
        self.assertNotIn("vol_factors_rank", F.GROUPS)


class TestExperiment(unittest.TestCase):
    def test_arms(self):
        import e51_vol_factors as E
        self.assertEqual(E.NEW_COLS, NEW)
        self.assertEqual(F.columns(E.VOL_PRESET), F.columns(E.BASE_PRESET) + NEW)
        self.assertNotIn("e51_params", str(E.LABELS))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
    unittest.main(verbosity=2)
