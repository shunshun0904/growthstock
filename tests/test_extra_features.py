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

    def test_一部だけあっても他は全欠測で返る(self):
        pd.DataFrame({"Code": ["13010"], "Date": [D("2024-04-01")],
                      "FwdEPS": [50.0]}).to_parquet(
            os.path.join(self.dir, "valuation.parquet"), index=False)
        s = samples(["2024-04-01"])
        out = EF.attach(s, self.dir, verbose=False)
        self.assertFalse(out["jq_fwdeps"].isna().all())
        self.assertTrue(out["lvs_ratio"].isna().all())

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
        out = EF.attach(s, self.dir, verbose=False)
        self.assertTrue(out["jq_fwdeps"].isna().all(), "0 で埋めている")

    def test_取り込みが無くても列構成は変わらない(self):
        # ここが崩れると build_dataset が SystemExit で落ち、日次予測が
        # 止まる（features.all_columns() の全列を要求するため）。
        # 2026-09-22 に実際に止まった
        s = samples(["2024-04-01", "2024-04-02"])
        out = EF.attach(s, self.dir, verbose=False)
        self.assertEqual(list(out.columns), EF.expected_columns())
        self.assertEqual(len(out), len(s))

    def test_一部だけ届いていても列構成は変わらない(self):
        pd.DataFrame({"Code": ["13010"], "Date": [D("2024-04-01")],
                      "FwdEPS": [50.0]}).to_parquet(
            os.path.join(self.dir, "valuation.parquet"), index=False)
        out = EF.attach(samples(["2024-04-01"]), self.dir, verbose=False)
        self.assertEqual(list(out.columns), EF.expected_columns())
        self.assertFalse(out["jq_fwdeps"].isna().all())
        self.assertTrue(out["lvs_ratio"].isna().all())

    def test_期待する列は_features_pyが正本(self):
        import features as _F
        mine = ("fwd", "holders_lvs", "holders_major", "holders_cross",
                "margin_alert", "earn_ahead", "flow")
        want = [c for g in mine for c in _F.GROUPS[g]]
        self.assertEqual(EF.expected_columns(), want)


class TestGroups(unittest.TestCase):
    """特徴量グループの登録。all には入れず、B 側のプリセットだけで使う。"""

    def test_all_には入れない(self):
        for g in F.EXTRA_GROUPS:
            self.assertNotIn(g, F.ALL_GROUPS, f"{g} が all に入っている")

    def test_AB_のプリセットがある(self):
        # 列数は直書きしない。グループを足すたびにテストを直すことになり、
        # そのとき「なぜこの数か」を確かめずに数字だけ合わせてしまう。
        # A を真に含むこと と 足した列がちょうど乗っていることを見る
        a, b = F.columns("all"), F.columns("all_plus")
        self.assertEqual(len(a), 153, "本番の列数が変わっている")
        self.assertTrue(set(a) < set(b), "B は A を真に含んでいない")
        extra = {c for g in F.EXTRA_GROUPS for c in F.GROUPS[g]}
        self.assertEqual(set(b) - set(a), extra,
                         "B と A の差が EXTRA_GROUPS と一致しない（重複か取りこぼし）")
        self.assertEqual(len(b), len(a) + len(extra))

    def test_グループの列名が重複しない(self):
        seen = {}
        for g in F.EXTRA_GROUPS:
            for c in F.GROUPS[g]:
                self.assertNotIn(c, seen, f"{c} が {g} と {seen.get(c)} に重複")
                seen[c] = g


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestForecastRevisions(unittest.TestCase):
    """
    予想修正イベント（DocType）。追加の取得はいらない塊。

    既存の guidance_revision が修正の「幅」なのに対し、こちらは
    「発生とタイミング」。時点整合（開示日より前に見えない）を固定する。
    """

    def setUp(self):
        import build_dataset as B
        self.B = B

    def fins(self):
        return pd.DataFrame({
            "Code": ["13010"] * 4,
            "DiscDate": [D("2024-05-10"), D("2024-08-05"),
                         D("2024-09-20"), D("2024-11-01")],
            "DocType": ["FYFinancialStatements_Consolidated_JP",
                        "1QFinancialStatements_Consolidated_JP",
                        "EarnForecastRevision",
                        "DividendForecastRevision"],
            "CurFYSt": [D("2023-04-01")] + [D("2024-04-01")] * 3,
            "FOP": [np.nan, 1000.0, 1200.0, np.nan],
            "Sales": [5000.0, 1200.0, np.nan, np.nan],
            "NP": [400.0, 90.0, np.nan, np.nan]})

    def test_修正の前には見えない(self):
        s = samples(["2024-09-19", "2024-09-20", "2024-09-30"])
        out = self.B.forecast_revisions(s, self.fins())
        self.assertTrue(np.isnan(out["days_since_rev"].iloc[0]), "修正前に見えている")
        self.assertEqual(out["days_since_rev"].iloc[1], 0, "当日は0日")
        self.assertEqual(out["days_since_rev"].iloc[2], 10)

    def test_上方修正の向きが取れる(self):
        s = samples(["2024-09-25"])
        out = self.B.forecast_revisions(s, self.fins())
        # 1Q の 1000 -> 修正の 1200 で +20%
        self.assertAlmostEqual(out["rev_pct"].iloc[0], 20.0, places=6)
        self.assertEqual(out["rev_up"].iloc[0], 1.0)

    def test_配当の修正は別に数える(self):
        s = samples(["2024-11-05"])
        out = self.B.forecast_revisions(s, self.fins())
        self.assertEqual(out["days_since_divrev"].iloc[0], 4)
        # 業績の修正は 9/20 なので 46日前のまま（配当で上書きしない）
        self.assertEqual(out["days_since_rev"].iloc[0], 46)

    def test_件数を窓で数える(self):
        s = samples(["2024-12-01"])
        out = self.B.forecast_revisions(s, self.fins())
        self.assertEqual(out["rev_n_250"].iloc[0], 1)
        self.assertEqual(out["rev_up_n_250"].iloc[0], 1)

    def test_前の予想が無ければ向きは欠測(self):
        f = self.fins()
        f.loc[1, "FOP"] = np.nan          # 1Q の予想を消す
        out = self.B.forecast_revisions(samples(["2024-09-25"]), f)
        self.assertTrue(np.isnan(out["rev_pct"].iloc[0]), "0 で埋めている")
        self.assertTrue(np.isnan(out["rev_up"].iloc[0]))

    def test_DocType_が無くても列構成は変わらない(self):
        # 列ごと落とすと features.all_columns() の要求を満たせず
        # build_dataset が SystemExit で落ちる。値は全欠測にする
        f = self.fins().drop(columns=["DocType"])
        out = self.B.forecast_revisions(samples(["2024-09-25"]), f)
        self.assertEqual(list(out.columns), F.GROUPS["revision"])
        self.assertTrue(out.isna().all().all(), "0 で埋めている")

    def test_配当修正が無くても列は出る(self):
        f = self.fins()
        f = f[f["DocType"] != "DividendForecastRevision"]
        out = self.B.forecast_revisions(samples(["2024-09-25"]), f)
        self.assertIn("days_since_divrev", out.columns)
        self.assertTrue(out["days_since_divrev"].isna().all())


class TestGuidanceGap(unittest.TestCase):
    """通期決算で会社予想が NxFOP に入る件（実測 FY 87.8% / 1Q 0.0%）。"""

    def test_グループに入っている(self):
        self.assertIn("revision", F.EXTRA_GROUPS)
        self.assertIn("guidance_op_growth", F.GROUPS["guidance"])


class 入れ子の列がparquetから戻る形(unittest.TestCase):
    """Hldrs / Report は経路によって来る形が変わる。全部受けること。

    実害があった（2026-09-22）: mjrshld を 77,730行 取り込んだのに
    mjr_* 7列が**全欠測**になった。pyarrow の list 型は読み戻すと
    numpy.ndarray になるが、_holders が list/tuple しか見ていなかった。
    データは1行も欠けていないのに 0% という、いちばん気づきにくい壊れ方。
    """

    def holders(self):
        return [{"HldrName": "日本マスタートラスト信託銀行株式会社（信託口）",
                 "ShsRatio": 0.10},
                {"HldrName": "株式会社サンプル", "ShsRatio": 0.05},
                {"HldrName": "山田太郎", "ShsRatio": 0.03}]

    def test_ndarray_でも_list_でも同じ(self):
        want = EF._holder_stats(self.holders())
        got = EF._holder_stats(np.array(self.holders(), dtype=object))
        self.assertTrue(want, "list で stats が空になっている")
        self.assertEqual(want, got)

    def test_文字列で来ても読める(self):
        want = EF._holder_stats(self.holders())
        got = EF._holder_stats(str(self.holders()))
        self.assertEqual(want, got)

    def test_中身が無ければ空(self):
        for cell in (None, [], np.array([], dtype=object), "", "なにか"):
            self.assertEqual(EF._holder_stats(cell), {}, repr(cell))

    def test_信託と個人を分ける(self):
        s = EF._holder_stats(np.array(self.holders(), dtype=object))
        self.assertAlmostEqual(s["mjr_top1"], 10.0)
        self.assertAlmostEqual(s["mjr_top10"], 18.0)
        self.assertEqual(s["mjr_n"], 3.0)
        self.assertAlmostEqual(s["mjr_trust"], 10.0)   # 信託銀行
        self.assertAlmostEqual(s["mjr_indiv"], 3.0)    # 法人名に見えない先

    def test_政策保有の_Report_も同じ(self):
        rep = {"ListedIss": 12.0, "ListedBookVal": 1.0e9}
        self.assertEqual(EF._xh(rep, "ListedIss"), 12.0)
        self.assertEqual(EF._xh(str(rep), "ListedIss"), 12.0)
        self.assertEqual(EF._xh(np.array([rep], dtype=object), "ListedIss"), 12.0)
