#!/usr/bin/env python3
"""
実験スクリプトが学習器に渡すパラメータの形を守っているかの単体テスト。

実害があった（2026-09-22、実験39 の空回し）:
  1. 探索の記録は {"params": {...}, "_cv": {...}} という入れ子で持つ。
     これを**ほどかずに** LightGBM へ渡すと
       TypeError: Unknown type of parameter:params, got:dict
     で落ちる。実験27 は呼び出し側でほどいていたが、実験39 は忘れていた。
  2. 表示側が metrics() の返す形を取り違えていた。実際は平らな辞書で
     `auc` も `label_rate` も入れ子も無いのに、m['auc'] を読んで
       KeyError: 'auc'
     で落ちた。腕Aの1本目が終わった**直後**に落ちるので、学習を
     やり直す羽目になる。

  どちらも3モデル×3腕の長い実行の途中で落ちる類なので、
  事前に押さえる価値がある。

  python3 tests/test_exp_params.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import pandas as pd  # noqa: E402

import e19_freshdata as E19  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e25_auc_noise as E25  # noqa: E402
import e39_extra_multi as E39  # noqa: E402


class 探索の記録をほどく(unittest.TestCase):
    """oof_arm は入れ子でも素でも受け、学習器には素だけを渡す。"""

    def setUp(self):
        self.seen = []
        self._oof_for = E19.oof_for
        self._oof_multi = E27.oof_multi
        self._seeds = E39.SEEDS3
        self._dir = E39.OOF_DIR
        # 学習はせず、渡されたパラメータだけ控える
        E19.oof_for = lambda df, cols, params: (
            self.seen.append(params) or pd.DataFrame({"p": [0.5]}))
        E27.oof_multi = lambda algo, df, cols, params, seed: (
            self.seen.append(params) or pd.DataFrame({"p": [0.5]}))
        E39.SEEDS3 = (42,)
        # 保存先を触らせない
        E39.OOF_DIR = os.path.join(ROOT, "tests", "fixtures", "_nowhere")

    def tearDown(self):
        E19.oof_for = self._oof_for
        E27.oof_multi = self._oof_multi
        E39.SEEDS3 = self._seeds
        E39.OOF_DIR = self._dir

    def run_arm(self, algo, params):
        self.seen.clear()
        try:
            E39.oof_arm(algo, "t", pd.DataFrame({"x": [1]}), ["x"], params)
        except OSError:
            pass          # 保存先が無いのは想定内。渡された値だけ見る
        self.assertTrue(self.seen, "学習器が呼ばれていない")
        return self.seen[0]

    def test_入れ子はほどかれる(self):
        for algo in ("lgbm", "xgb", "cat"):
            got = self.run_arm(algo, {"params": {"num_leaves": 31}, "_cv": {}})
            self.assertEqual(got, {"num_leaves": 31}, algo)

    def test_素はそのまま通る(self):
        for algo in ("lgbm", "xgb", "cat"):
            got = self.run_arm(algo, {"num_leaves": 31})
            self.assertEqual(got, {"num_leaves": 31}, algo)

    def test_学習器に渡す辞書に入れ子は残らない(self):
        for algo in ("lgbm", "xgb", "cat"):
            got = self.run_arm(algo, {"params": {"a": 1}, "_cv": {"b": 2}})
            self.assertNotIn("params", got, algo)
            self.assertNotIn("_cv", got, algo)


class 表示が読む鍵(unittest.TestCase):
    """metrics() が実際に返す鍵だけで、表の1行を組み立てられること。

    metrics() は平らな辞書を返す
      pr_auc / roc_auc / day_auc / lift
      ret_o1_20_mean / _won / _n / _worst（ret_o1_40 も同じ形）
    """

    def oof(self, n=2000, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "Code": [f"{1000 + i % 50}0" for i in range(n)],
            "Date": (pd.to_datetime("2024-01-01")
                     + pd.to_timedelta(rng.integers(0, 400, n), "D")),
            "label": rng.integers(0, 2, n),
            "ret_o1_20": rng.normal(0, 5, n),
            "ret_o1_40": rng.normal(0, 7, n),
            "score": rng.random(n),
            "fold": rng.integers(0, 5, n),
        })

    def test_1行を組み立てられる(self):
        m = E25.metrics(self.oof())
        got = E39.line("腕", 205, m)                 # KeyError が出たら失敗
        self.assertIn("205", got)
        self.assertTrue(got.strip().startswith("腕"))

    def test_metrics_は平らな辞書(self):
        m = E25.metrics(self.oof())
        for k, v in m.items():
            self.assertNotIsInstance(v, dict, f"{k} が入れ子になっている")

    def test_見出しと本体の幅がそろう(self):
        m = E25.metrics(self.oof())
        self.assertEqual(len(E39.HEAD), len(E39.line("x" * 26, 205, m)))

    def test_CSV_に入れる列が_metrics_と同じ鍵(self):
        """rows へ **m で流し込むので、入れ子が混ざると壊れる。"""
        m = E25.metrics(self.oof())
        row = {"algo": "lgbm", "arm": "A", "ncol": 205, "label_rate": 0.17, **m}
        pd.DataFrame([row])                          # 例外が出たら失敗
        self.assertIn("ret_o1_20_mean", row)
        self.assertNotIn("ret_o1_20", row)


class 本番パラメータの形(unittest.TestCase):
    """prod_params は入れ子で返す。実験側はそれを前提にしてよい。"""

    def test_3モデルとも入れ子で返る(self):
        for algo in ("lgbm", "xgb", "cat"):
            rec = E27.prod_params(algo)
            self.assertIsInstance(rec, dict, algo)
            self.assertIsInstance(rec.get("params"), dict, algo)
            self.assertTrue(rec["params"], f"{algo}: パラメータが空")

    def test_下線で始まる鍵は学習器へ渡らない(self):
        for algo in ("lgbm", "xgb", "cat"):
            for k in E27.prod_params(algo)["params"]:
                self.assertFalse(k.startswith("_"), f"{algo}: {k}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
