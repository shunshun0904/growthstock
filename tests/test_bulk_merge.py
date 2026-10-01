#!/usr/bin/env python3
"""
毎回まるごと取り直す種別（jq_bulk.BULK_KINDS。いまは投資部門別売買）の保存のテスト。

J-Quants は契約の範囲（今日の10年前から。毎日1日ずつ後ろへずれる）の行しか返さない。
応答で保存済みの parquet を上書きすると、範囲から外れた週が毎週消えていく
（2026-09-26 の写し 2,377行 → 10-01 の取り込み後 2,375行。2016-09-26・09-29 公表の
10行が消え、範囲より前なので二度と取り直せない。docs/OPERATIONS.md）。

- 取り直した行の最古の日付より前の保存済みの行は残す
- それ以降は応答で置き換える（訂正・取り下げを反映。範囲の中は応答が正）
- 保存済みが無い・読めない・日付列が無いなら、応答だけで作る
- 応答に新しい列があっても古い行は残る
- 取り込み本体（_run_incremental --what investor）がこの経路を通り、取得記録に残した数を書く

  python3 tests/test_bulk_merge.py
"""
import datetime as dt
import json
import os
import sys
import tempfile
import types
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import jq_bulk as J  # noqa: E402


def rows(*items):
    """(公表日, 区分, 値) の並びから、応答と同じ形の表を作る。"""
    return pd.DataFrame([{"PubDate": d, "StDate": d, "EnDate": d, "Section": s, "PropBal": v}
                         for d, s, v in items])


class TestMergeBulk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "investor.parquet")

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_before_the_window_are_kept(self):
        rows(("2016-09-26", "TSE1st", 1.0), ("2016-09-29", "TSE1st", 2.0),
             ("2016-10-06", "TSE1st", 3.0)).to_parquet(self.path, index=False)
        fresh = rows(("2016-10-06", "TSE1st", 30.0), ("2016-10-13", "TSE1st", 4.0))
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 2)
        self.assertEqual(out["PubDate"].tolist(),
                         ["2016-09-26", "2016-09-29", "2016-10-06", "2016-10-13"])
        # 範囲の中は応答が正（訂正を反映する）
        self.assertEqual(float(out.loc[out["PubDate"] == "2016-10-06", "PropBal"].iloc[0]), 30.0)

    def test_inside_the_window_the_response_wins(self):
        """保存済みにあって応答に無い行でも、範囲の中なら残さない（取り下げを反映）。"""
        rows(("2016-10-06", "TSE1st", 1.0), ("2016-10-13", "TSE1st", 2.0)) \
            .to_parquet(self.path, index=False)
        fresh = rows(("2016-10-06", "TSE1st", 1.0), ("2016-10-20", "TSE1st", 3.0))
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 0)
        self.assertEqual(out["PubDate"].tolist(), ["2016-10-06", "2016-10-20"])

    def test_first_run_has_nothing_to_keep(self):
        fresh = rows(("2016-10-06", "TSE1st", 1.0))
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 0)
        self.assertIs(out, fresh)

    def test_unreadable_file_is_rebuilt_from_the_response(self):
        with open(self.path, "wb") as fh:
            fh.write(b"not a parquet")
        fresh = rows(("2016-10-06", "TSE1st", 1.0))
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 0)
        self.assertEqual(len(out), 1)

    def test_missing_date_column_falls_back_to_the_response(self):
        rows(("2016-09-26", "TSE1st", 1.0)).to_parquet(self.path, index=False)
        fresh = pd.DataFrame([{"Section": "TSE1st", "PropBal": 1.0}])
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 0)
        self.assertIs(out, fresh)

    def test_new_column_in_the_response_keeps_old_rows(self):
        rows(("2016-09-26", "TSE1st", 1.0)).to_parquet(self.path, index=False)
        fresh = rows(("2016-10-06", "TSE1st", 2.0)).assign(NewCol=5.0)
        out, kept = J.merge_bulk(self.path, fresh, "PubDate")
        self.assertEqual(kept, 1)
        self.assertEqual(out["PubDate"].tolist(), ["2016-09-26", "2016-10-06"])
        self.assertTrue(pd.isna(out.loc[0, "NewCol"]))
        self.assertEqual(float(out.loc[1, "NewCol"]), 5.0)


class FakeClient:
    """パスごとに決まった行を返す。"""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def get_paginated(self, path, params):
        self.calls.append((path, dict(params)))
        return list(self.table.get(path, []))


class TestIngestionKeepsOldRows(unittest.TestCase):
    """取り込み本体（--incremental --what investor）がこの経路を通ること。"""

    def test_run_incremental_keeps_rows_before_the_window(self):
        with tempfile.TemporaryDirectory() as d:
            rows(("2016-09-26", "TSE1st", 1.0), ("2016-10-06", "TSE1st", 2.0)) \
                .to_parquet(os.path.join(d, "investor.parquet"), index=False)
            resp = [{"PubDate": "2016-10-06", "StDate": "2016-09-26", "EnDate": "2016-09-30",
                     "Section": "TSE1st", "PropBal": "20"},
                    {"PubDate": "2016-10-13", "StDate": "2016-10-03", "EnDate": "2016-10-07",
                     "Section": "TSE1st", "PropBal": "3"}]
            client = FakeClient({"/equities/investor-types": resp})
            args = types.SimpleNamespace(out_dir=d, what=["investor"], reset=[], forget=[],
                                         forget_from=None)
            rc = J._run_incremental(client, [], dt.date(2016, 10, 1), dt.date(2016, 10, 31), args)
            self.assertEqual(rc, 0)
            self.assertEqual(client.calls, [("/equities/investor-types", {})])
            saved = pd.read_parquet(os.path.join(d, "investor.parquet"))
            self.assertEqual(saved["PubDate"].tolist(), ["2016-09-26", "2016-10-06", "2016-10-13"])
            self.assertEqual(float(saved.loc[saved["PubDate"] == "2016-10-06", "PropBal"].iloc[0]),
                             20.0)
            with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            self.assertEqual(m["investor"]["rows"], 3)
            self.assertEqual(m["investor"]["kept_before_window"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
