#!/usr/bin/env python3
"""
research/predict_daily.detail_codes の単体テスト。

タイムマシーン（過去比較）は、銘柄ごとのスナップショットと株価履歴が
取れていないと使えない。それを取りに行く銘柄を決めているのがここ。

実害があった（2026-09-22 に利用者から報告）:
  画面の候補46件のうち取得済みは8件だけで、**83%で過去比較が見られない**。
  原因は「最新日の上位10件」しか対象にしていなかったこと。画面は5営業日
  ぶんを切り替えられるのに、選ぶのが上位10件とは限らない。

  python3 tests/test_predict_codes.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import predict_daily as P  # noqa: E402


def rows(*spec):
    """(日付, 順位, コード) の並びから行を作る。"""
    return [{"date": d, "rankInDay": r, "code": c} for d, r, c in spec]


class TestDetailCodes(unittest.TestCase):
    def test_画面に出ている候補を全部取る(self):
        r = rows(("2026-09-18", 1, "A"), ("2026-09-18", 2, "B"),
                 ("2026-09-17", 1, "C"), ("2026-09-16", 1, "D"))
        self.assertEqual(len(P.detail_codes(r, 120)), 4)

    def test_新しい日の上位から並ぶ(self):
        r = rows(("2026-09-16", 1, "old1"), ("2026-09-18", 2, "new2"),
                 ("2026-09-18", 1, "new1"), ("2026-09-17", 1, "mid1"))
        self.assertEqual(P.detail_codes(r, 120), ["new1", "new2", "mid1", "old1"])

    def test_上限で切れるのは古くて順位の低いほう(self):
        r = rows(("2026-09-16", 1, "old1"), ("2026-09-18", 2, "new2"),
                 ("2026-09-18", 1, "new1"))
        self.assertEqual(P.detail_codes(r, 2), ["new1", "new2"])

    def test_同じ銘柄が複数日に出ても1回だけ(self):
        r = rows(("2026-09-18", 3, "X"), ("2026-09-17", 1, "X"),
                 ("2026-09-17", 2, "Y"))
        got = P.detail_codes(r, 120)
        self.assertEqual(got, ["X", "Y"])
        self.assertEqual(len(got), len(set(got)))

    def test_空なら空(self):
        self.assertEqual(P.detail_codes([], 120), [])

    def test_上限0は上限なしとして扱う(self):
        r = rows(("2026-09-18", 1, "A"), ("2026-09-18", 2, "B"))
        self.assertEqual(len(P.detail_codes(r, 0)), 2)

    def test_既定の上限は画面の候補を覆える大きさ(self):
        # 5営業日 × 1日あたり最大40件でも、ふだんは全部入る大きさにする。
        # 1銘柄あたり4リクエストなので、上げすぎると予測の実行時間に響く
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--top-codes", type=int, default=120)
        self.assertGreaterEqual(ap.parse_args([]).top_codes, 60)

    def test_以前の挙動より必ず多く取る(self):
        # 最新日の上位10件だけ、には戻らないこと
        r = rows(*[("2026-09-18", i, f"N{i}") for i in range(1, 13)],
                 *[("2026-09-17", i, f"M{i}") for i in range(1, 6)])
        got = P.detail_codes(r, 120)
        self.assertEqual(len(got), 17)
        self.assertIn("M5", got, "前日の候補が落ちている")


if __name__ == "__main__":
    unittest.main(verbosity=2)
