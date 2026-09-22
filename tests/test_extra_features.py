#!/usr/bin/env python3
"""
research/extra_features.py の単体テスト。

いちばん大事なのは **時点整合**。EDINET は提出日、信用規制は公表日、
決算予定は公表日で結合する。決算予定の SchDate は実測で中央値64日先
なので、これを直接キーにすると未来を見る。

次に大事なのは **取り込みが届いていなくても落ちないこと**。8本のうち
一部しか無い状態で学習が止まると、段階的に入れられない。

  python3 tests/test_extra_features.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import extra_features as EF  # noqa: E402
import features as F  # noqa: E402

D = pd.Timestamp


def samples(dates, code="13010"):
    return pd.DataFrame({"Code": [code] * len(dates),
                         "Date": [D(d) for d in dates],
                         "close_raw": [1000.0] * len(dates),
                         "market_cap": [100.0] * len(dates),
                         "per": [20.0] * len(dates),
                         "ROE_q0": [8.0] * len(dates)})


class TestPointInTime(unittest.TestCase):
    """その日までに出ているものしか見ない。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, kind, df):
        df.to_parquet(os.path.join(self.dir, f"{kind}.parquet"), index=False)

    def test_大量保有は提出日より前には出てこない(self):
        self._write("lvshld", pd.DataFrame({
            "Code": ["13010"], "SubDate": [D("2024-03-10")],
            "TotalShsRatio": [0.0612], "TotalShsRatioLast": [0.0503]}))
        s = samples(["2024-03-08", "2024-03-10", "2024-03-15"])
        out = EF.large_volume(s, self.dir)
        self.assertTrue(np.isnan(out["lvs_ratio"].iloc[0]), "提出前に見えている")
        self.assertAlmostEqual(out["lvs_ratio"].iloc[1], 6.12, places=2)
        self.assertAlmostEqual(out["lvs_ratio_chg"].iloc[2], 1.09, places=2)
        self.assertEqual(out["lvs_days"].iloc[2], 5)

    def test_決算予定は公表済みのものだけ使う(self):
        # 3/20 に「5/10 に発表」と公表された。3/15 時点では知らない
        self._write("earndate", pd.DataFrame({
            "Code": ["13010"], "PubDate": [D("2024-03-20")],
            "SchDate": [D("2024-05-10")]}))
        s = samples(["2024-03-15", "2024-03-25", "2024-05-10", "2024-05-20"])
        out = EF.earnings_ahead(s, self.dir)
        self.assertTrue(np.isnan(out["days_to_earn"].iloc[0]),
                        "公表前の予定が見えている（未来を見ている）")
        self.assertEqual(out["days_to_earn"].iloc[1], 46)
        self.assertEqual(out["days_to_earn"].iloc[2], 0)
        self.assertTrue(np.isnan(out["days_to_earn"].iloc[3]), "過ぎた予定が残っている")

    def test_信用規制は公表日で結合する(self):
        self._write("marginalert", pd.DataFrame({
            "Code": ["13010"], "PubDate": [D("2024-06-03")],
            "LongOutRatio": [3.5], "ShrtOutRatio": [1.2], "SLRatio": [2.9]}))
        s = samples(["2024-06-01", "2024-06-05"])
        out = EF.margin_alert(s, self.dir)
        self.assertTrue(np.isnan(out["alert_longoutratio"].iloc[0]))
        self.assertAlmostEqual(out["alert_longoutratio"].iloc[1], 3.5)
        self.assertEqual(out["alert_days"].iloc[1], 2)

    def test_予想指標は当日の値だけ使う(self):
        self._write("valuation", pd.DataFrame({
            "Code": ["13010", "13010"], "Date": [D("2024-04-01"), D("2024-04-02")],
            "FwdEPS": [50.0, 60.0], "FwdPER": [20.0, 16.7], "FwdROE": [12.0, 13.0]}))
        s = samples(["2024-04-01", "2024-04-02", "2024-04-03"])
        out = EF.valuation(s, self.dir)
        self.assertAlmostEqual(out["jq_fwdeps"].iloc[0], 50.0)
        self.assertAlmostEqual(out["jq_fwdeps"].iloc[1], 60.0)
        self.assertTrue(np.isnan(out["jq_fwdeps"].iloc[2]), "翌日に持ち越している")
        # 予想と実績の差
        self.assertAlmostEqual(out["jq_roe_gap"].iloc[0], 4.0)       # 12 - 8
        self.assertAlmostEqual(out["jq_fwd_earnings_yield"].iloc[0], 5.0)


class TestHolders(unittest.TestCase):
    """大株主の集計。信託と個人を名前で分ける。"""

    def test_集中度と主体の内訳(self):
        hs = [{"Rank": "1", "HldrName": "株式会社タカオカ興産", "ShsRatio": "0.10"},
              {"Rank": "2", "HldrName": "日本マスタートラスト信託銀行株式会社（信託口）",
               "ShsRatio": "0.08"},
              {"Rank": "3", "HldrName": "高岡伸夫", "ShsRatio": "0.06"}]
        st = EF._holder_stats(hs)
        self.assertAlmostEqual(st["mjr_top1"], 10.0)
        self.assertAlmostEqual(st["mjr_top10"], 24.0)
        self.assertAlmostEqual(st["mjr_trust"], 8.0)      # 信託口
        self.assertAlmostEqual(st["mjr_indiv"], 6.0)      # 法人語が無い＝個人
        self.assertEqual(st["mjr_n"], 3)
        self.assertAlmostEqual(st["mjr_conc"], 10.0 / 24.0)

    def test_文字列で来ても読める(self):
        st = EF._holder_stats(str([{"Rank": "1", "HldrName": "株式会社A",
                                    "ShsRatio": "0.2"}]))
        self.assertAlmostEqual(st["mjr_top1"], 20.0)

    def test_空なら何も返さない(self):
        self.assertEqual(EF._holder_stats(None), {})
        self.assertEqual(EF._holder_stats([]), {})
        self.assertEqual(EF._holder_stats("こわれた"), {})


class TestResilience(unittest.TestCase):
    """取り込みが届いていなくても落ちない。0 で埋めない。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_何も無くても落ちない(self):
        s = samples(["2024-04-01", "2024-04-02"])
        out = EF.attach(s, self.dir, verbose=False)
        self.assertEqual(len(out), len(s))

    def test_一部だけあっても他は空で返る(self):
        pd.DataFrame({"Code": ["13010"], "Date": [D("2024-04-01")],
                      "FwdEPS": [50.0]}).to_parquet(
            os.path.join(self.dir, "valuation.parquet"), index=False)
        s = samples(["2024-04-01"])
        out = EF.attach(s, self.dir, verbose=False)
        self.assertIn("jq_fwdeps", out.columns)
        self.assertNotIn("lvs_ratio", out.columns)

    def test_行の順序と数を変えない(self):
        pd.DataFrame({"Code": ["13010"], "SubDate": [D("2024-03-10")],
                      "TotalShsRatio": [0.06]}).to_parquet(
            os.path.join(self.dir, "lvshld.parquet"), index=False)
        s = samples(["2024-03-15", "2024-03-01", "2024-03-20"])   # 日付順でない
        out = EF.attach(s, self.dir, verbose=False)
        self.assertEqual(len(out), 3)
        # 2行目（3/01）は提出前なので欠測のまま
        self.assertTrue(np.isnan(out["lvs_ratio"].iloc[1]))
        self.assertFalse(np.isnan(out["lvs_ratio"].iloc[0]))

    def test_欠測を0で埋めない(self):
        s = samples(["2024-04-01"])
        out = EF.valuation(s, self.dir)
        self.assertEqual(out.shape[1], 0)     # 列ごと無い（0 の列を作らない）


class TestGroups(unittest.TestCase):
    """特徴量グループの登録。all には入れず、B 側のプリセットだけで使う。"""

    def test_all_には入れない(self):
        for g in F.EXTRA_GROUPS:
            self.assertNotIn(g, F.ALL_GROUPS, f"{g} が all に入っている")

    def test_AB_のプリセットがある(self):
        a, b = F.columns("all"), F.columns("all_plus")
        self.assertEqual(len(a), 153)
        self.assertEqual(len(b), 197)
        self.assertTrue(set(a) < set(b), "B は A を含んでいない")

    def test_グループの列名が重複しない(self):
        seen = {}
        for g in F.EXTRA_GROUPS:
            for c in F.GROUPS[g]:
                self.assertNotIn(c, seen, f"{c} が {g} と {seen.get(c)} に重複")
                seen[c] = g


if __name__ == "__main__":
    unittest.main(verbosity=2)
