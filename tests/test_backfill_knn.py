#!/usr/bin/env python3
"""
候補の比較の集計の埋め戻し（research/backfill_knn.py）。

- predictions.json に knn を足す。候補・日付・モデルの素性などほかの項目は一切変えない
- --only-missing は、前の knn で ok の日を引き継ぎ、無い日だけ作る
- out-of-fold が無ければ書かずに 1 で終わる。--dry-run は書かない
- 出力は集計だけ（候補ごとの値・過去の事例の銘柄を出さない）

  python3 tests/test_backfill_knn.py
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
sys.path.insert(0, os.path.join(ROOT, "tests"))

import backfill_knn as BF  # noqa: E402
import knn_compare as KC  # noqa: E402
import test_knn_compare as TK  # noqa: E402


class TestBackfill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir, cls.model_dir, cls.bars, cls.oof, cls.codes = TK.make_case()
        for a in KC.RULE_MODELS:                       # Release の形（<名前>_oof.parquet）
            src = KC.LT.find_oof(a, cls.model_dir, cls.model_dir)   # 基準モデル（lgbm）は oof.parquet、ほかは <名前>_oof.parquet
            dst = "oof.parquet" if a == "lgbm" else f"{a}_oof.parquet"
            pd.read_parquet(src).to_parquet(os.path.join(cls.data_dir, dst), index=False)
        cls.payload = {
            "generatedAt": "2025-07-01T09:00:00+00:00", "asOf": TK.DAY, "dates": ["2025-06-30", TK.DAY],
            "dayInfo": {"2025-06-30": {"trainedAt": "x"}, TK.DAY: {"trainedAt": "x"}},
            "candidates": [TK.cand(cls.codes[1], TK.DAY, [97.0, 98.5, 96.2], 0.9),
                           TK.cand(cls.codes[2], TK.DAY, [99.0, 95.0, 95.5], 0.8),
                           TK.cand(cls.codes[3], "2025-06-30", [96.0, 96.0, 96.0], 0.7)],
            "model": {"trainedAt": "x"}, "notes": ["a"],
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        shutil.rmtree(cls.model_dir, ignore_errors=True)

    def write(self, payload):
        path = os.path.join(tempfile.mkdtemp(), "predictions.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        return path

    def run_main(self, path, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = BF.main(["--pred", path, "--data-dir", self.data_dir, "--model-dir", self.model_dir, *extra])
        return rc, buf.getvalue()

    def test_adds_knn_and_changes_nothing_else(self):
        path = self.write(self.payload)
        rc, out = self.run_main(path)
        self.assertEqual(rc, 0)
        with open(path, encoding="utf-8") as fh:
            got = json.load(fh)
        self.assertEqual({k: v for k, v in got.items() if k != "knn"}, self.payload)
        self.assertEqual(got["knn"]["byDay"][TK.DAY]["status"], "ok")
        self.assertEqual(got["knn"]["byDay"][TK.DAY]["n"], 2)
        self.assertEqual(got["knn"]["byDay"]["2025-06-30"]["n"], 1)
        self.assertIn("knn を書いた", out)
        for code in set(self.codes) - {self.codes[1], self.codes[2], self.codes[3]}:
            self.assertNotIn(code, out)

    def test_only_missing_keeps_previous_days(self):
        prev = dict(self.payload)
        prev["knn"] = {"byDay": {"2025-06-30": {"status": "ok", "poolN": 1, "n": 9, "candidates": {"x": {}}}}}
        path = self.write(prev)
        rc, _ = self.run_main(path, "--only-missing")
        self.assertEqual(rc, 0)
        with open(path, encoding="utf-8") as fh:
            got = json.load(fh)
        self.assertEqual(got["knn"]["byDay"]["2025-06-30"]["n"], 9)          # 前の値のまま
        self.assertEqual(got["knn"]["byDay"][TK.DAY]["n"], 2)                # 無い日は作る
        rc2, _ = self.run_main(path)                                          # 既定は全部作り直す
        with open(path, encoding="utf-8") as fh:
            got2 = json.load(fh)
        self.assertEqual(rc2, 0)
        self.assertEqual(got2["knn"]["byDay"]["2025-06-30"]["n"], 1)

    def test_dry_run_and_missing_oof(self):
        path = self.write(self.payload)
        before = open(path, encoding="utf-8").read()
        rc, out = self.run_main(path, "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("--dry-run", out)
        self.assertEqual(open(path, encoding="utf-8").read(), before)
        empty = tempfile.mkdtemp()
        try:
            rc, out = self.run_main(path, "--oof-dir", empty, "--model-dir", empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
        self.assertEqual(rc, 1)
        self.assertIn("作れなかった", out)
        self.assertEqual(open(path, encoding="utf-8").read(), before)


if __name__ == "__main__":
    unittest.main()
