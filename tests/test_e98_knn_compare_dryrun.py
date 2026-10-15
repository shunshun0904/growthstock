#!/usr/bin/env python3
"""
実験98（research/exp/e98_knn_compare_dryrun.py）: 候補の比較の集計を実データで試しに作る台本。

- 合成の out-of-fold（Release の <名前>_oof.parquet の形）と日足、候補の predictions.json から、集計を作って要約を出す
- 出力は集計だけ（候補ごとの値・過去の事例の銘柄コードを出さない）
- out-of-fold が無ければ「出せない」理由を出し、落ちない

  python3 tests/test_e98_knn_compare_dryrun.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import e98_knn_compare_dryrun as E  # noqa: E402
import knn_compare as KC  # noqa: E402
import test_knn_compare as TK  # noqa: E402


class TestDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir, cls.model_dir, cls.bars, cls.oof, cls.codes = TK.make_case()
        # Release の形（<名前>_oof.parquet）に置き直す
        for a in KC.RULE_MODELS:
            src = os.path.join(cls.model_dir, "models", a, "oof.parquet")
            pd.read_parquet(src).to_parquet(os.path.join(cls.data_dir, f"{a}_oof.parquet"), index=False)
        cands = [TK.cand(cls.codes[1], TK.DAY, [97.0, 98.5, 96.2], 0.9), TK.cand(cls.codes[2], TK.DAY, [99.0, 95.0, 95.5], 0.8),
                 TK.cand(cls.codes[3], "2025-06-30", [96.0, 96.0, 96.0], 0.7)]
        cls.pred = os.path.join(cls.data_dir, "predictions.json")
        with open(cls.pred, "w", encoding="utf-8") as fh:
            json.dump({"dates": ["2025-06-30", TK.DAY], "candidates": cands}, fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def run_main(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = E.main(["--pred", self.pred, "--data-dir", self.data_dir, *extra])
        return rc, buf.getvalue()

    def test_summary_is_aggregate_only(self):
        rc, out = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("候補 3件・日 2日", out)
        self.assertIn(f"{TK.DAY}: ok 集計の付いた候補 2件", out)
        self.assertIn("2025-06-30: ok 集計の付いた候補 1件", out)
        self.assertIn("文脈: 過去の候補", out)
        self.assertRegex(out, r"JSON [\d,]+文字")
        for code in set(self.codes) - {self.codes[1], self.codes[2], self.codes[3]}:
            self.assertNotIn(code, out)                         # 過去の事例の銘柄コードは出ない
        for code in (self.codes[1], self.codes[2], self.codes[3]):
            self.assertNotIn(code, out)                         # 候補ごとの値も出ない
        self.assertNotIn("p25", out)

    def test_without_oof_reports_reason_and_does_not_crash(self):
        empty = tempfile.mkdtemp()
        try:
            rc, out = self.run_main("--oof-dir", empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
        self.assertEqual(rc, 0)
        self.assertIn("unavailable 理由: FileNotFoundError", out)
        self.assertIn("文脈: 無し", out)


if __name__ == "__main__":
    unittest.main()
