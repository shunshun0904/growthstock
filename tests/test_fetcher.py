#!/usr/bin/env python3
"""
scripts/jquants_data_fetcher.py の純粋計算部分の単体テスト。
ネットワークアクセスは行わない (合成データのみ)。

  python3 tests/test_fetcher.py
"""
import datetime as dt
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "research"))

from jquants_data_fetcher import (  # noqa: E402
    build_milestones, credit_metrics, describe_secret, display_code,
    fundamental_metrics, normalize_code, pct_change, price_metrics, quarterize,
)


def make_quotes(closes, volumes=None, start="2025-01-06"):
    """営業日を1日ずつ進める簡易な日次バー列を作る (High=Close, Open=Close-1)。"""
    d = dt.date.fromisoformat(start)
    rows = []
    for i, c in enumerate(closes):
        v = volumes[i] if volumes else 100000
        # V2 /equities/bars/daily の列名 (O/H/L/C/Vo/Va, 調整後は Adj*)
        rows.append({
            "Date": (d + dt.timedelta(days=i)).isoformat(),
            "O": c - 1, "H": c, "L": c - 2, "C": c, "Vo": v, "Va": c * v,
            "AdjO": c - 1, "AdjH": c, "AdjL": c - 2, "AdjC": c, "AdjVo": v,
        })
    return rows


class TestCodeNormalization(unittest.TestCase):
    def test_four_digit_gets_trailing_zero(self):
        # 仕様書 §3.1 : 6928 -> 69280
        self.assertEqual(normalize_code("6928"), "69280")
        self.assertEqual(normalize_code(" 7203 "), "72030")

    def test_five_digit_untouched(self):
        self.assertEqual(normalize_code("13010"), "13010")
        self.assertEqual(normalize_code("130A0"), "130A0")

    def test_display_code_roundtrip(self):
        self.assertEqual(display_code("69280"), "6928")
        self.assertEqual(display_code("130A0"), "130A")


class TestPctChange(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(pct_change(150, 100), 50.0)

    def test_undefined_when_base_non_positive(self):
        # 赤字 -> 黒字を「+○○%」と表現しない (値を捏造しない)
        self.assertIsNone(pct_change(50, 0))
        self.assertIsNone(pct_change(50, -20))
        self.assertIsNone(pct_change(None, 100))


class TestPriceMetrics(unittest.TestCase):
    def test_high_ratio_and_trading_value(self):
        closes = list(range(1000, 1100))          # 単調上昇 -> 最終日が52週高値
        q = make_quotes(closes, volumes=[200000] * 100)
        m = price_metrics(q)
        self.assertEqual(m["price"], 1099)
        self.assertEqual(m["high52w"], 1099)
        self.assertAlmostEqual(m["highRatio"], 100.0)
        # 売買代金 = 1099 * 200000 / 1e8 = 2.198 億円
        self.assertAlmostEqual(m["tradingValue"], 1099 * 200000 / 1e8)

    def test_volume_trend_excludes_latest_bar(self):
        # 直前20日が 100000、最終日が 300000 -> 300%
        vols = [100000] * 40 + [300000]
        q = make_quotes([1000] * 41, volumes=vols)
        m = price_metrics(q)
        self.assertAlmostEqual(m["ma20Volume"], 100000)
        self.assertAlmostEqual(m["volumeTrend"], 300.0)

    def test_point_in_time_ignores_future_bars(self):
        closes = [1000] * 50 + [2000] * 10        # 後半で急騰
        q = make_quotes(closes)
        as_of = q[49]["Date"]
        m = price_metrics(q, as_of=as_of)
        self.assertEqual(m["date"], as_of)
        self.assertEqual(m["price"], 1000)
        self.assertEqual(m["high52w"], 1000)      # 未来の 2000 を見ていない

    def test_empty_series(self):
        m = price_metrics([])
        self.assertIsNone(m["price"])
        self.assertIsNone(m["highRatio"])


def statement(fy_start, period, disclosed, sales, op, profit, eps, **extra):
    """V2 /fins/summary の列名で決算開示行を作る。"""
    row = {
        "DocType": "3QFinancialStatements_Consolidated_JP",
        "CurPerType": period,
        "CurFYSt": fy_start,
        "CurPerEn": disclosed,
        "DiscDate": disclosed,
        "DiscTime": "15:00",
        "Sales": sales, "OP": op, "NP": profit, "EPS": eps,
    }
    row.update(extra)
    return row


class TestQuarterize(unittest.TestCase):
    def setUp(self):
        # 累計ベースの開示データ (1Q=100, 2Q累計=220, 3Q累計=360, 4Q累計=520)
        self.rows = [
            statement("2024-04-01", "1Q", "2024-08-05", 100, 10, 6, 6.0),
            statement("2024-04-01", "2Q", "2024-11-05", 220, 24, 15, 15.0),
            statement("2024-04-01", "3Q", "2025-02-05", 360, 42, 26, 26.0),
            statement("2024-04-01", "4Q", "2025-05-12", 520, 64, 40, 40.0,
                      Eq=1000, FOP=80, ShOutFY=1_000_000, TrShFY=50_000),
            statement("2025-04-01", "1Q", "2025-08-05", 140, 16, 10, 10.0,
                      Eq=1050, FOP=90),
        ]

    def test_cumulative_values_are_differenced(self):
        qs = quarterize(self.rows)
        by = {(r["fiscalYearStart"], r["quarter"]): r for r in qs}
        self.assertEqual(by[("2024-04-01", 1)]["qNetSales"], 100)   # 1Q は累計=単期
        self.assertEqual(by[("2024-04-01", 2)]["qNetSales"], 120)   # 220 - 100
        self.assertEqual(by[("2024-04-01", 3)]["qNetSales"], 140)   # 360 - 220
        self.assertEqual(by[("2024-04-01", 4)]["qNetSales"], 160)   # 520 - 360
        self.assertEqual(by[("2024-04-01", 4)]["qOperatingProfit"], 22)  # 64 - 42

    def test_shares_outstanding_excludes_treasury(self):
        fm = fundamental_metrics(quarterize(self.rows), as_of="2025-05-31")
        self.assertEqual(fm["sharesOutstanding"], 950_000)

    def test_yoy_growth_compares_same_quarter(self):
        fm = fundamental_metrics(quarterize(self.rows))
        # 2025年1Q 売上 140 vs 2024年1Q 売上 100 -> +40%
        self.assertAlmostEqual(fm["salesGrowth"], 40.0)
        self.assertAlmostEqual(fm["epsGrowth"], (10.0 - 6.0) / 6.0 * 100)
        self.assertEqual(fm["quarter"], 1)

    def test_roe_uses_trailing_four_quarters_when_not_reported(self):
        fm = fundamental_metrics(quarterize(self.rows), as_of="2025-05-31")
        # 直近4四半期純利益 = 6 + 9 + 11 + 14 = 40、自己資本 1000 -> 4.0%
        self.assertAlmostEqual(fm["roe"], 4.0)
        self.assertIn("TTM", fm["roeBasis"])

    def test_roe_prefers_api_reported_value(self):
        """V2 の /fins/summary は ROE を直接返すため、提供値があればそれを使う。"""
        rows = list(self.rows)
        rows[3] = statement("2024-04-01", "4Q", "2025-05-12", 520, 64, 40, 40.0,
                            Eq=1000, FOP=80, ShOutFY=1_000_000, TrShFY=50_000,
                            ROE=12.5)
        fm = fundamental_metrics(quarterize(rows), as_of="2025-05-31")
        self.assertAlmostEqual(fm["roe"], 12.5)
        self.assertIn("提供値", fm["roeBasis"])

    def test_progress_rate(self):
        fm = fundamental_metrics(quarterize(self.rows))
        # 2025年1Q 累計営業利益 16 / 通期予想 90 -> 17.8%
        self.assertAlmostEqual(fm["progressRate"], 16 / 90 * 100)

    def test_point_in_time_excludes_undisclosed(self):
        # 2025-08-05 の開示前に見れば、最新は 2024年度4Q
        fm = fundamental_metrics(quarterize(self.rows), as_of="2025-07-01")
        self.assertEqual(fm["fiscalPeriod"], "4Q")
        self.assertEqual(fm["disclosedDate"], "2025-05-12")

    def test_forecast_revision_documents_are_ignored(self):
        """実績値を持たない開示 (業績予想の修正のみ) は四半期系列に混ぜない。"""
        noise = self.rows + [{
            "DocType": "ForecastRevision", "CurPerType": "1Q",
            "CurFYSt": "2025-04-01", "DiscDate": "2025-09-01",
            "Sales": None, "OP": None, "NP": None, "EPS": None,
            "FOP": 120,   # 予想だけが入っている
        }]
        qs = quarterize(noise)
        self.assertTrue(all(r["disclosedDate"] != "2025-09-01" for r in qs))

    def test_rows_without_valid_period_are_ignored(self):
        noise = self.rows + [{
            "DocType": "Other", "CurPerType": "", "CurFYSt": "2025-04-01",
            "DiscDate": "2025-09-02", "Sales": 999,
        }]
        qs = quarterize(noise)
        self.assertTrue(all(r["disclosedDate"] != "2025-09-02" for r in qs))


class TestCreditMetrics(unittest.TestCase):
    def test_ratio(self):
        rows = [
            {"Date": "2025-08-01", "LongVol": 1000, "ShrtVol": 500},
            {"Date": "2025-08-08", "LongVol": 900, "ShrtVol": 600},
        ]
        self.assertAlmostEqual(credit_metrics(rows)["creditRatio"], 1.5)
        self.assertAlmostEqual(credit_metrics(rows, as_of="2025-08-05")["creditRatio"], 2.0)

    def test_absent_data_is_none_not_zero(self):
        self.assertIsNone(credit_metrics([])["creditRatio"])
        self.assertIsNone(
            credit_metrics([{"Date": "2025-08-01", "LongVol": 10, "ShrtVol": 0}])["creditRatio"]
        )


class TestDescribeSecret(unittest.TestCase):
    """認証エラー時の切り分け情報。secret の値そのものは絶対に出力しない。"""

    def test_never_leaks_the_value(self):
        secret = "SUPERSECRETVALUE1234567890abcdefghijklmno"
        out = describe_secret(secret)
        self.assertNotIn(secret, out)
        self.assertNotIn("SUPERSECRET", out)
        self.assertIn(str(len(secret)), out)

    def test_plain_api_key_is_accepted_without_warning(self):
        out = describe_secret("x" * 43)
        self.assertIn("JWTではない", out)
        self.assertNotIn("→", out)

    def test_flags_leftover_v1_token(self):
        """V1 のトークン (JWT) が残っていたら移行漏れとして警告する。"""
        out = describe_secret(".".join(["x" * 300] * 3))
        self.assertIn("V1 のリフレッシュトークン", out)
        self.assertIn("APIキー", out)


class TestMilestones(unittest.TestCase):
    def test_detects_breakout_and_volume_spike(self):
        base = dt.date.today() - dt.timedelta(days=80)
        closes = [1000] * 60 + [1500]
        vols = [100000] * 60 + [500000]
        q = make_quotes(closes, vols, start=base.isoformat())
        events = build_milestones(q, [])
        kinds = {e["type"] for e in events}
        self.assertIn("breakout", kinds)
        self.assertIn("volume_spike", kinds)

    def test_flat_series_produces_no_events(self):
        base = dt.date.today() - dt.timedelta(days=80)
        q = make_quotes([1000] * 60, [100000] * 60, start=base.isoformat())
        self.assertEqual(build_milestones(q, []), [])


class TestFieldMatching(unittest.TestCase):
    """
    データ探索で「どの項目が何に当たるか」を判定する部分。

    最初の実装は部分一致だったため、CurPerType が PER に、
    CashEq が 自己資本 に当たってしまい、
    存在しない指標を「取得できる」と報告しかけた。
    誤検出を出さないことを固定する。
    """

    # 実際に /fins/summary が返した項目名（probe の実測結果より）
    SEEN = ["EPS", "NCEPS", "DEPS", "FEPS", "FEPS2Q", "NxFEPS", "BPS", "NCBPS",
            "CurPerEn", "CurPerSt", "CurPerType", "ROE", "NCROE", "TA", "NCTA",
            "Eq", "EqAR", "NCEq", "CashEq", "ShEq", "NCShEq", "ShOutFY",
            "DivTotalAnn", "FDivTotalAnn"]

    def _m(self):
        from probe_fins_fields import match_wanted
        return match_wanted(self.SEEN)

    def test_per_and_pbr_and_roa_are_not_provided(self):
        """API が返していない指標を「ある」と言わないこと。"""
        m = self._m()
        self.assertEqual(m["PER"], [])
        self.assertEqual(m["PBR"], [])
        self.assertEqual(m["ROA"], [])

    def test_curpertype_does_not_match_per(self):
        m = self._m()
        self.assertNotIn("CurPerType", m["PER"])
        self.assertNotIn("CurPerEn", m["PER"])

    def test_casheq_is_not_equity(self):
        """現金及び現金同等物は自己資本ではない。"""
        self.assertNotIn("CashEq", self._m()["自己資本"])

    def test_dividend_total_is_not_total_assets(self):
        self.assertNotIn("DivTotalAnn", self._m()["総資産"])

    def test_finds_what_is_actually_there(self):
        m = self._m()
        self.assertIn("BPS", m["BPS(1株純資産)"])
        self.assertIn("NCBPS", m["BPS(1株純資産)"])
        self.assertIn("TA", m["総資産"])
        self.assertIn("ROE", m["ROE"])
        self.assertIn("EPS", m["EPS(1株利益)"])

    def test_reads_fin_cols_without_pandas(self):
        """probe は標準ライブラリだけで動く必要がある。
        jq_bulk を import すると pandas が要り、probe.yml で落ちた。"""
        from probe_fins_fields import read_fin_cols
        cols = read_fin_cols()
        self.assertIn("Sales", cols)
        self.assertIn("TA", cols)
        self.assertNotIn("pandas", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
