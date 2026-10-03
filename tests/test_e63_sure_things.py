#!/usr/bin/env python3
"""
実験63（research/exp/e63_sure_things.py）のテスト。重い計算は回さない。

- 発火は5モデルの過去分布の百分位がすべて 95 以上の行だけ
- 並びは p_min → LightGBM のスコア
- 上位 k 件の正例率は件数どおり
- 日内百分位は 0〜1 で日の平均が 0.5、型（profile）は同じ向きにそろった列が先頭に来る

  python3 tests/test_e63_sure_things.py
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e63_sure_things as E  # noqa: E402

warnings.filterwarnings("ignore")


def toy_frame(n_days=80, per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2024-01-01", periods=n_days)):
        for j in range(per_day):
            z = rng.standard_normal()
            rows.append({"Code": f"{1000 + j}", "Date": d, "fold": i // 20 + 1, "label": int(z > 1.0),
                         "ret_o1_20": z / 10, **{f"s_{a}": 1 / (1 + np.exp(-(z + 0.3 * rng.standard_normal())))
                                               for a in E.ALGOS}})
    return pd.DataFrame(rows)


class Rule(unittest.TestCase):
    def test_add_rule_and_ranking(self):
        fr = E.add_rule(toy_frame(), pct=95.0, min_hist=100)
        self.assertTrue((fr.loc[fr["passed"], [f"p_{a}" for a in E.ALGOS]].min(axis=1) >= 95).all())
        self.assertTrue(fr["p_min"].isna().iloc[0])          # 最初の日は過去分布が無い → 判定できない
        self.assertFalse(fr["passed"].iloc[0])
        fired = E.fired_ranking(fr)
        self.assertTrue((fired["p_min"].diff().dropna() <= 0).all())
        self.assertEqual(fired["rank"].tolist(), list(range(1, len(fired) + 1)))
        self.assertGreater(len(fired), 0)

    def test_precision_at(self):
        fired = pd.DataFrame({"label": [1, 1, 0, 1, 0, 0, 1, 1, 1, 1]})
        pk = E.precision_at(fired, ks=(2, 5, 8))
        got = {int(r.k): (int(r.positives), round(float(r.precision), 1)) for r in pk.itertuples()}
        self.assertEqual(got[2], (2, 100.0))
        self.assertEqual(got[5], (3, 60.0))
        self.assertEqual(got[8], (5, 62.5))
        self.assertEqual(got[10], (7, 70.0))


class Profile(unittest.TestCase):
    def test_within_date_pct_and_profile(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"Code": [f"{i}" for i in range(30)],
                           "Date": pd.to_datetime(["2024-01-04"] * 10 + ["2024-01-05"] * 10 + ["2024-01-09"] * 10),
                           "a": rng.standard_normal(30), "b": rng.standard_normal(30)})
        df.loc[df.index[:5], "b"] = np.nan                   # 欠損は順位から外れる
        pct = E.within_date_pct(df, ["a", "b"])
        self.assertTrue(((pct["a"] > 0) & (pct["a"] < 1)).all())
        np.testing.assert_allclose(pct.groupby(df["Date"])["a"].mean(), 0.5)
        self.assertTrue(pct["b"].iloc[:5].isna().all())
        self.assertAlmostEqual(float(pct["b"].iloc[5:10].mean()), 0.5)
        # 型: 上位5件が a で高い側、b で低い側、c はばらつき
        rows = pd.DataFrame({"a": [0.95, 0.9, 0.85, 0.99, 0.7], "b": [0.1, 0.05, 0.15, 0.3, 0.1],
                             "c": [0.1, 0.9, 0.5, 0.2, 0.8]})
        prof = E.profile(rows, ["c", "b", "a"])
        self.assertEqual(prof["col"].tolist()[:2], ["a", "b"]) if prof.iloc[0]["agree"] > prof.iloc[1]["agree"] else \
            self.assertEqual(set(prof["col"].tolist()[:2]), {"a", "b"})
        self.assertEqual(prof.set_index("col").loc["a", "side"], "高")
        self.assertEqual(prof.set_index("col").loc["b", "side"], "低")
        self.assertEqual(int(prof.set_index("col").loc["c", "agree"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
