#!/usr/bin/env python3
"""
学習データの行の並びを (Date, Code) に固定したことのテスト（2026-09-25、運用者の了承）。

学習は行を並べ替えずに使い、LightGBM・XGBoost は行を間引く（subsample）ので、並びが
変わると同じ種でも別の行を引く。これまで並びは結合の順しだいで、コードを直すたびに
変わっていた。実測では、25行（0.11%）の値と同じ日の中の並びが違うだけで、運用の規則の
月あたりが +0.90% → +1.36% 動いた（docs/MODEL_ADOPTION_RULES.md §10 実験46 の読み方 4）。
"""
import inspect
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402


def sample(seed: int = 0) -> pd.DataFrame:
    """4日 × 4銘柄（英字入りのコードも含む）を、並びを崩して返す。"""
    days = pd.to_datetime(["2024-01-05", "2024-01-04", "2024-01-09", "2024-01-08"])
    codes = ["72030", "154A0", "99840", "13010"]
    rows = [(d, c) for d in days for c in codes]
    df = pd.DataFrame(rows, columns=["Date", "Code"])
    df["x"] = np.arange(len(df), dtype=float)
    df["label"] = (np.arange(len(df)) % 3 == 0).astype(int)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TestCanonicalOrder(unittest.TestCase):
    def test_sorted_by_date_then_code(self):
        out = B.canonical_order(sample())
        key = list(zip(out["Date"], out["Code"]))
        self.assertEqual(key, sorted(key))
        self.assertEqual(list(out.index), list(range(len(out))))

    def test_same_result_whatever_the_input_order(self):
        """結合の順が変わっても（入力の並びが違っても）同じ並びになる。"""
        a = B.canonical_order(sample(seed=1))
        b = B.canonical_order(sample(seed=2))
        pd.testing.assert_frame_equal(a, b)

    def test_values_travel_with_their_rows(self):
        src = sample(seed=3)
        out = B.canonical_order(src)
        merged = out.merge(src, on=["Date", "Code"], suffixes=("", "_src"))
        self.assertEqual(len(merged), len(src))
        self.assertTrue((merged["x"] == merged["x_src"]).all())
        self.assertTrue((merged["label"] == merged["label_src"]).all())

    def test_duplicate_keys_keep_their_relative_order(self):
        """同じキーの行が残っても並びが決まる（安定な並べ替え）。"""
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05"] * 3 + ["2024-01-04"]),
                           "Code": ["10000", "10000", "10000", "20000"],
                           "x": [3.0, 1.0, 2.0, 9.0]})
        out = B.canonical_order(df)
        self.assertEqual(out["x"].tolist(), [9.0, 3.0, 1.0, 2.0])

    def test_code_column_is_not_converted(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05", "2024-01-04"]),
                           "Code": [72030, 13010]})
        out = B.canonical_order(df)
        self.assertEqual(out["Code"].tolist(), [13010, 72030])
        self.assertTrue(pd.api.types.is_integer_dtype(out["Code"]))

    def test_build_writes_the_canonical_order(self):
        """build() は出力の行を並べてから書く（呼び出しを消すとここで落ちる）。"""
        src = inspect.getsource(B.build)
        i_out = src.index("out = samples[meta_cols + feature_cols]")
        i_sort = src.index("out = canonical_order(out)")
        i_save = src.index("out.to_parquet(out_path")
        self.assertLess(i_out, i_sort)
        self.assertLess(i_sort, i_save)


class TestSameOrderInTuningAndWalkforward(unittest.TestCase):
    """探索とウォークフォワードは日付だけで並べ直していた（安定でない並べ替え）。

    日付だけの並べ替えは同じ日の中の並びを入力しだいで入れ替えうるので、学習データと
    同じ (Date, Code) の安定な並べ替えにそろえる。
    """

    def test_run_tuning(self):
        import run_tuning
        src = inspect.getsource(run_tuning)
        self.assertIn('sort_values(["Date", "Code"], kind="mergesort")', src)
        self.assertNotIn('sort_values("Date").reset_index', src)

    def test_walkforward(self):
        import walkforward
        src = inspect.getsource(walkforward)
        self.assertIn('sort_values(["Date", "Code"], kind="mergesort")', src)
        self.assertNotIn('sort_values("Date").reset_index', src)


class TestFingerprintSeesRowOrder(unittest.TestCase):
    """実験の out-of-fold を読み直す鍵（中身の指紋）が、行の並びも見ること。

    前は行ごとの値の和で、並びが違っても同じ指紋になり、並びの違う out-of-fold を
    読みえた。
    """

    def check(self, fingerprint):
        a = B.canonical_order(sample())
        self.assertEqual(fingerprint(a, ["x"]), fingerprint(a.copy(), ["x"]))
        shuffled = a.sample(frac=1.0, random_state=7).reset_index(drop=True)
        self.assertNotEqual(fingerprint(a, ["x"]), fingerprint(shuffled, ["x"]))
        changed = a.copy()
        changed.loc[0, "x"] += 1.0
        self.assertNotEqual(fingerprint(a, ["x"]), fingerprint(changed, ["x"]))

    def test_ab_oof(self):
        import ab_oof
        self.check(ab_oof.fingerprint)

    def test_e43(self):
        import e43_pit_fix
        self.check(e43_pit_fix.fingerprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
