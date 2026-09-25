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
    fundamental_metrics, fundamentals_as_of, margin_published_on, normalize_code,
    pct_change, price_metrics, quarterize,
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
    """
    高値は78週（368営業日）。予測モデル（build_dataset.HIGH_WINDOW）と同じ窓・同じ条件:
    その日を含む直近368本の日足の最高値、履歴が368本に満たない・窓の半分以上で高値が
    無いときは出さない（2026-09-25 に 52週=365暦日 から変えた）。
    make_quotes は1日1本なので、本数がそのまま営業日の数になる。
    """

    def test_high_ratio_and_trading_value(self):
        closes = list(range(1000, 1400))          # 単調上昇 -> 最終日が78週高値
        q = make_quotes(closes, volumes=[200000] * 400)
        m = price_metrics(q)
        self.assertEqual(m["price"], 1399)
        self.assertEqual(m["high52w"], 1399)      # 列名は互換のため high52w（中身は78週）
        self.assertAlmostEqual(m["highRatio"], 100.0)
        # 売買代金 = 1399 * 200000 / 1e8 億円
        self.assertAlmostEqual(m["tradingValue"], 1399 * 200000 / 1e8)

    def test_window_is_368_bars_not_52_weeks(self):
        """
        367本前の高値は窓に入り、368本前は入らない。52週（365暦日）で切ると
        367本前（=367日前）の高値を落とすので、この2本で窓の長さを見分けられる。
        """
        n = 400
        inside = [1000] * n
        inside[n - 1 - 367] = 2000                # 367本前: 窓の中（最古の1本）
        m = price_metrics(make_quotes(inside))
        self.assertEqual(m["high52w"], 2000)
        self.assertAlmostEqual(m["highRatio"], 50.0)
        outside = [1000] * n
        outside[n - 1 - 368] = 2000               # 368本前: 窓の外
        m = price_metrics(make_quotes(outside))
        self.assertEqual(m["high52w"], 1000)
        self.assertAlmostEqual(m["highRatio"], 100.0)

    def test_no_value_without_78_weeks_of_history(self):
        """上場から368営業日に満たない銘柄は出さない（モデルもブレイクの母集団に入れない）。"""
        m = price_metrics(make_quotes([1000] * 367))
        self.assertIsNone(m["highRatio"])
        self.assertIsNone(m["high52w"])
        self.assertEqual(m["price"], 1000)        # 株価そのものは出す
        self.assertIsNotNone(price_metrics(make_quotes([1000] * 368))["highRatio"])

    def test_no_value_when_most_highs_are_missing(self):
        """窓の半分以上で高値が無い（売買停止が長い）ときは出さない。"""
        q = make_quotes([1000] * 400)
        for r in q[-368:-183]:                    # 窓368本のうち185本の高値を消す
            r["H"] = r["AdjH"] = None
        self.assertIsNone(price_metrics(q)["highRatio"])
        q = make_quotes([1000] * 400)
        for r in q[-368:-184]:                    # 184本を消す（残り184本 = ちょうど半分）
            r["H"] = r["AdjH"] = None
        self.assertIsNotNone(price_metrics(q)["highRatio"])

    def test_volume_trend_excludes_latest_bar(self):
        # 直前20日が 100000、最終日が 300000 -> 300%
        vols = [100000] * 40 + [300000]
        q = make_quotes([1000] * 41, volumes=vols)
        m = price_metrics(q)
        self.assertAlmostEqual(m["ma20Volume"], 100000)
        self.assertAlmostEqual(m["volumeTrend"], 300.0)

    def test_point_in_time_ignores_future_bars(self):
        closes = [1000] * 400 + [2000] * 10       # 後半で急騰
        q = make_quotes(closes)
        as_of = q[399]["Date"]
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

    def test_roe_uses_ttm_even_when_api_value_exists(self):
        """
        提供値（本決算の行にだけある）より、直近4四半期の計算を優先する。
        提供値を先にすると、本決算の直後の時点だけ ROE の定義が変わり、
        タイムマシーンの時点どうし・銘柄どうしで別の物差しを比べることになる。
        """
        rows = list(self.rows)
        rows[3] = statement("2024-04-01", "4Q", "2025-05-12", 520, 64, 40, 40.0,
                            Eq=1000, FOP=80, ShOutFY=1_000_000, TrShFY=50_000,
                            ROE=0.125)
        fm = fundamental_metrics(quarterize(rows), as_of="2025-05-31")
        self.assertAlmostEqual(fm["roe"], 4.0)              # 40 / 1000
        self.assertIn("TTM", fm["roeBasis"])

    def test_api_roe_is_a_ratio_and_only_a_fallback(self):
        """
        J-Quants の ROE は**小数**（0.125 = 12.5%）。実データ（9081 の本決算で 0.06、
        research/build_dataset.py も100倍している）で確かめた。4四半期が揃わない
        ときだけ使い、% に直す。以前はこのテスト自体が「提供値は %」という誤った
        前提（ROE=12.5）で書かれていて、100分の1の値を通していた。
        """
        rows = [statement("2024-04-01", "FY", "2025-05-12", 520, 64, 40, 40.0,
                          Eq=1000, ROE=0.125)]
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



class TestRefiledPeriods(unittest.TestCase):
    """
    過去の決算を後から出し直した（訂正・再提出）銘柄で、直近の四半期がずれないこと。

    2026-09-25 に 9081 神奈川中央交通で実際に起きた形を小さく再現する:
    2024年度の本決算（2025-05-12）を 2025-12-01 に同じ数字で再提出。開示日の順に
    最後の4行を取ると、2024年度4Q が紛れ込み 2025年度2Q が抜ける。
    """

    def setUp(self):
        self.rows = [
            statement("2024-04-01", "1Q", "2024-08-05", 100, 10, 6, 6.0),
            statement("2024-04-01", "2Q", "2024-11-05", 220, 24, 15, 15.0),
            statement("2024-04-01", "3Q", "2025-02-05", 360, 42, 26, 26.0),
            # 4Q は赤字（単期 NP = 20 - 26 = -6）
            statement("2024-04-01", "FY", "2025-05-12", 520, 50, 20, 20.0, Eq=1000),
            statement("2025-04-01", "1Q", "2025-08-05", 140, 16, 10, 10.0, Eq=1010),
            statement("2025-04-01", "2Q", "2025-11-05", 300, 36, 22, 22.0, Eq=1020),
            # 2024年度の本決算を同じ数字で再提出
            statement("2024-04-01", "FY", "2025-12-01", 520, 50, 20, 20.0, Eq=1000),
            statement("2025-04-01", "3Q", "2026-02-05", 470, 57, 35, 35.0, Eq=1030,
                      FOP=80),
            # 4Q は赤字（単期 NP = 30 - 35 = -5）
            statement("2025-04-01", "FY", "2026-05-10", 600, 70, 30, 30.0, Eq=1040),
            statement("2026-04-01", "1Q", "2026-08-05", 150, 18, 11, 11.0, Eq=1050,
                      FOP=90),
        ]

    def test_ttm_is_the_last_four_fiscal_quarters(self):
        """
        9081 と同じ形: 再提出（2025-12-01）の後に本決算と1Q が出た時点。開示日の順の
        最後の4行は「2024年度4Q・2025年度3Q・2025年度4Q・2026年度1Q」で、2025年度2Q が
        抜ける（修正前のコードは純利益 13 / ROE 1.24% を返していた）。
        """
        fm = fundamentals_as_of(self.rows, as_of="2026-09-01")
        # 直近4四半期 = 2025年度2Q/3Q/4Q, 2026年度1Q
        # 純利益 12 + 13 + (-5) + 11 = 31 / 自己資本 1050
        self.assertAlmostEqual(fm["roe"], 31 / 1050 * 100)
        # 営業利益 20 + 21 + 13 + 18 = 72 / 売上 160 + 170 + 130 + 150 = 610
        self.assertAlmostEqual(fm["opMargin"], 72 / 610 * 100)
        self.assertEqual(fm["opMarginBasis"], "TTM (直近4四半期)")
        self.assertAlmostEqual(fm["salesGrowth"], (150 - 140) / 140 * 100)

    def test_latest_is_the_latest_fiscal_period_not_the_latest_filing(self):
        """再提出の直後でも「最新の決算」は 2025年度2Q（2024年度の本決算ではない）。"""
        fm = fundamentals_as_of(self.rows, as_of="2025-12-15")
        self.assertEqual(fm["fiscalPeriod"], "2Q")
        self.assertEqual(fm["quarter"], 2)
        # 前年同期比は 2024年度2Q（単期 売上 120）と比べる: 160 / 120 - 1
        self.assertAlmostEqual(fm["salesGrowth"], (160 - 120) / 120 * 100)

    def test_ttm_right_after_the_refiling(self):
        """再提出の直後の時点でも、4四半期は 2024年度3Q〜2025年度2Q。"""
        fm = fundamentals_as_of(self.rows, as_of="2025-12-15")
        # 純利益 11 + (-6) + 10 + 12 = 27 / 自己資本 1020
        self.assertAlmostEqual(fm["roe"], 27 / 1020 * 100)

    def test_past_snapshot_keeps_the_original_before_the_refiling(self):
        """
        再提出の前の時点では、元の開示（2025-05-12）が見えていた。後日の再提出で
        置き換わったせいで、その時点から 2024年度の本決算が消えてはいけない。
        """
        fm = fundamentals_as_of(self.rows, as_of="2025-06-30")
        self.assertEqual(fm["fiscalPeriod"], "FY")
        self.assertEqual(fm["disclosedDate"], "2025-05-12")
        # 2024年度 1Q〜4Q の純利益 6 + 9 + 11 + (-6) = 20 / 1000
        self.assertAlmostEqual(fm["roe"], 20 / 1000 * 100)

    def test_yoy_needs_the_immediately_previous_year(self):
        """前年の同じ四半期が無いとき、2年前と比べない。"""
        rows = [
            statement("2023-04-01", "1Q", "2023-08-05", 80, 8, 5, 5.0),
            statement("2025-04-01", "1Q", "2025-08-05", 140, 16, 10, 10.0),
            statement("2024-04-01", "2Q", "2024-11-05", 220, 24, 15, 15.0),
        ]
        fm = fundamentals_as_of(rows, as_of="2025-09-01")
        self.assertEqual(fm["fiscalPeriod"], "1Q")
        self.assertIsNone(fm["salesGrowth"])


class TestMarginPublication(unittest.TestCase):
    """週次の信用残は、基準日の翌週の第2営業日に公表される（予測モデルと同じ規則）。"""

    def test_friday_is_published_on_the_next_tuesday(self):
        self.assertEqual(margin_published_on("2025-08-01"), "2025-08-05")

    def test_holiday_monday_moves_it_to_wednesday(self):
        # 2025-09-15（月）は敬老の日。カレンダーがあれば 9/17（水）
        days = [dt.date(2025, 9, d) for d in (11, 12, 16, 17, 18, 19)]
        self.assertEqual(margin_published_on("2025-09-12", days), "2025-09-17")

    def test_uses_weekdays_outside_the_calendar(self):
        days = [dt.date(2026, 1, 5)]           # 覆っている範囲より後の基準日
        self.assertEqual(margin_published_on("2026-03-06", days), "2026-03-10")

    def test_week_is_not_used_before_it_is_published(self):
        rows = [
            {"Date": "2025-08-01", "LongVol": 1000, "ShrtVol": 500},
            {"Date": "2025-08-08", "LongVol": 900, "ShrtVol": 600},
        ]
        # 8/8（金）の週は 8/12（火）に公表。8/11（月）の時点ではまだ 8/1 の週
        self.assertAlmostEqual(credit_metrics(rows, as_of="2025-08-11")["creditRatio"], 2.0)
        self.assertAlmostEqual(credit_metrics(rows, as_of="2025-08-12")["creditRatio"], 1.5)
        # 基準日の当日（8/8）の時点でも、その週の値は使わない
        self.assertEqual(credit_metrics(rows, as_of="2025-08-08")["marginDate"], "2025-08-01")

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
        base = dt.date.today() - dt.timedelta(days=450)
        closes = [1000] * 400 + [1500]
        vols = [100000] * 400 + [500000]
        q = make_quotes(closes, vols, start=base.isoformat())
        events = build_milestones(q, [])
        kinds = {e["type"] for e in events}
        self.assertIn("breakout", kinds)
        self.assertIn("volume_spike", kinds)
        title = next(e["title"] for e in events if e["type"] == "breakout")
        self.assertEqual(title, "78週高値を更新")

    def test_breakout_needs_78_weeks_before_it(self):
        """前日までに368営業日の履歴が無い日の高値更新は、78週高値の更新とは言えない。"""
        base = dt.date.today() - dt.timedelta(days=350)
        closes = [1000] * 300 + [1500]
        q = make_quotes(closes, [100000] * 301, start=base.isoformat())
        self.assertNotIn("breakout", {e["type"] for e in build_milestones(q, [])})

    def test_old_high_outside_the_window_does_not_block(self):
        """369本前の高値（窓の外）は、上抜けの判定に使わない。"""
        base = dt.date.today() - dt.timedelta(days=450)
        closes = [3000] + [1000] * 399 + [1500]     # 先頭の 3000 は 400本前
        q = make_quotes(closes, [100000] * 401, start=base.isoformat())
        self.assertIn("breakout", {e["type"] for e in build_milestones(q, [])})

    def test_flat_series_produces_no_events(self):
        base = dt.date.today() - dt.timedelta(days=450)
        q = make_quotes([1000] * 400, [100000] * 400, start=base.isoformat())
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
        jq_bulk を import すると pandas が要り、probe.yml で落ちた。

        sys.modules は差し替えの効かないプロセス共有の状態なので、
        同じプロセス内で判定すると他のテストが入れた pandas を拾う
        （単体では通るのに一括実行だけ落ちていた）。別プロセスで確かめる。
        """
        import subprocess
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import probe_fins_fields as m;"
            "c = m.read_fin_cols();"
            "assert c is None or isinstance(c, list), type(c);"
            "assert 'pandas' not in sys.modules, 'pandas を import している';"
            "print('ok')" % os.path.join(ROOT, "research")
        )
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_fins_keeps_every_field(self):
        """
        決算をホワイトリストで絞ると、書き漏らした項目が黙って捨てられる。
        実際 BPS がそれで失われ、PBR を作れなかった。
        絞らない設定であることを固定する。
        """
        from probe_fins_fields import read_fin_cols
        self.assertIsNone(read_fin_cols(),
                          "FIN_COLS は None（全項目保持）であるべき")


class TestWaitForData(unittest.TestCase):
    """
    research/wait_for_data.py。取り込みの前に当日データを待つ。

    起動時刻を 21:30 JST から 16:05 JST へ前倒ししたので、定刻どおりに
    起動した日は「まだ当日データが無い」状態で走る。待てば済む話を
    失敗にしないことが、この部品の要件そのもの。

    **当日データの有無に関わる経路は、どれも 0 で返ること**をここで固定する。
    ここが非ゼロを返すと、祝日や鍵の不備でパイプライン全体がその日だけ
    止まる。古いデータで予測する事故は後段の check_freshness.py が見ている。
    （引数の書き間違いだけは別扱いで、非ゼロで落とす。）
    """

    def setUp(self):
        import wait_for_data as W
        self.W = W
        self._client = W.JQuantsClient
        self._resolve = W.resolve_api_key
        self._sleep = W.time.sleep
        self.slept = []
        W.resolve_api_key = lambda: "dummy-key"

        def fake_sleep(sec):
            # 本当には眠らない。ただし無限ループを「遅いテスト」ではなく
            # 「失敗」として出す。締切の判定を壊すと実測で数時間回り続ける
            self.slept.append(sec)
            if len(self.slept) > 100:
                raise AssertionError("待ちが終わらない（締切の判定が壊れている）")

        W.time.sleep = fake_sleep

    def tearDown(self):
        self.W.JQuantsClient = self._client
        self.W.resolve_api_key = self._resolve
        self.W.time.sleep = self._sleep

    def _client_returning(self, pages):
        """呼ばれるたびに pages を順に返す偽クライアント。"""
        seq = list(pages)
        calls = []

        class Fake:
            def __init__(self, *a, **k):
                pass

            def get_paginated(self, path, params):
                calls.append((path, params))
                return seq.pop(0) if seq else []

        self.W.JQuantsClient = Fake
        return calls

    @staticmethod
    def _future_weekday():
        """
        実時刻より必ず未来の平日を返す。

        締切は「その日の HH:MM」で作られるので、固定日を書くと日が経った
        ときに締切が過去になり、ポーリングせず1巡で諦めるようになる
        （実際に 2026-09-18 固定で書いていて、日付が変わって落ちた）。
        """
        import datetime as _dt
        d = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).date() \
            + _dt.timedelta(days=7)
        while d.weekday() >= 5:
            d += _dt.timedelta(days=1)
        return d.isoformat()

    def _client_by_path(self, table):
        """パスごとに返す行を決める偽クライアント。table: パス -> 行の列。"""
        seqs = {k: list(v) for k, v in table.items()}
        calls = []

        class Fake:
            def __init__(self, *a, **k):
                pass

            def get_paginated(self, path, params):
                calls.append((path, params))
                q = seqs.get(path, [])
                return q.pop(0) if q else []

        self.W.JQuantsClient = Fake
        return calls

    def test_weekend_does_not_wait(self):
        # 2026-09-19 は土曜。API を1度も叩かないこと
        calls = self._client_returning([])
        self.assertEqual(self.W.main(["--date", "2026-09-19"]), 0)
        self.assertEqual(calls, [])

    def test_missing_key_does_not_wait(self):
        def boom():
            from jquants_data_fetcher import AuthError
            raise AuthError("鍵がありません")
        self.W.resolve_api_key = boom
        calls = self._client_returning([])
        self.assertEqual(self.W.main(["--date", "2026-09-18"]), 0)
        self.assertEqual(calls, [])

    def test_returns_as_soon_as_rows_appear(self):
        day = self._future_weekday()
        bar = {"Code": "13010", "C": 1000}
        calls = self._client_returning([[], [], [bar]])
        rc = self.W.main(["--date", day, "--feeds", "bars",
                          "--deadline", "23:59", "--interval", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "/equities/bars/daily")
        self.assertEqual(calls[0][1], {"date": day})

    def test_waits_for_every_feed_not_just_bars(self):
        """
        四本値だけ待って走ると、指数・TOPIX が0件のまま取り込むことになる。
        build_dataset は merge_asof(backward) で結合するので、
        欠測ではなく**前日の値**が黙って入る。だから全部揃うまで待つ。
        """
        day = self._future_weekday()
        bar = {"Code": "13010", "C": 1000}
        calls = self._client_by_path({
            "/equities/bars/daily": [[bar]],              # 1回目で揃う
            "/indices/bars/daily": [[], [{"Code": "0040"}]],
            "/indices/bars/daily/topix": [[], [], [{"C": 2800}]],
        })
        rc = self.W.main(["--date", day, "--deadline", "23:59",
                          "--interval", "1"])
        self.assertEqual(rc, 0)
        # 揃った対象は二度と叩かない
        paths = [c[0] for c in calls]
        self.assertEqual(paths.count("/equities/bars/daily"), 1)
        self.assertEqual(paths.count("/indices/bars/daily"), 2)
        self.assertEqual(paths.count("/indices/bars/daily/topix"), 3)
        # TOPIX だけ期間指定で引く（実測に使った叩き方と同じ）
        tp = [c for c in calls if c[0] == "/indices/bars/daily/topix"][0]
        self.assertEqual(tp[1], {"from": day, "to": day})

    def test_bars_needs_a_close_not_just_a_row(self):
        """
        前場ぶんだけ入って終値が空、という出方をされたら「まだ」と見る。
        行数だけ見ていると、欠測を揃ったと誤読して取り込む。
        """
        calls = self._client_returning([
            [{"Code": "13010", "C": None}],   # 行はあるが終値が無い
            [{"Code": "13010", "C": 1000}],
        ])
        rc = self.W.main(["--date", self._future_weekday(), "--feeds", "bars",
                          "--deadline", "23:59", "--interval", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)

    def test_unknown_feed_is_an_error(self):
        """
        対象名の書き間違いは当日データの有無と関係がない。
        ここを 0 で素通りさせると、待っているつもりで待たなくなる。
        """
        calls = self._client_returning([])
        self.assertEqual(self.W.main(["--date", "2026-09-18",
                                      "--feeds", "bars,barz"]), 2)
        self.assertEqual(calls, [])

    def test_past_deadline_gives_up_without_failing(self):
        """
        祝日は当日データが永遠に出ない。締切を過ぎたら 0 で抜けること。
        ここで非ゼロを返すと、祝日のたびに取り込みが落ちる。
        """
        calls = self._client_returning([[], [], []])
        # 過去の営業日の 00:00 を締切にする = 既に締切を過ぎているので
        # 1回だけ見て諦める。「今日の00:00」にすると実行時刻によっては
        # まだ締切前で、実測で何時間も回り続ける
        rc = self.W.main(["--date", "2020-01-06", "--feeds", "bars",
                          "--deadline", "00:00", "--interval", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.slept, [])

    def test_query_error_is_not_fatal(self):
        """問い合わせが失敗しても落ちない（次の回に再試行する）。"""
        from jquants_data_fetcher import JQuantsError

        class Flaky:
            seen: list = []

            def __init__(self, *a, **k):
                self.n = 0

            def get_paginated(self, path, params):
                self.n += 1
                Flaky.seen.append(path)
                if self.n == 1:
                    raise JQuantsError("503")
                return [{"Code": "13010", "C": 1000}]

        seen = []
        Flaky.seen = seen
        self.W.JQuantsClient = Flaky
        rc = self.W.main(["--date", self._future_weekday(), "--feeds", "bars",
                          "--deadline", "23:59", "--interval", "1"])
        self.assertEqual(rc, 0)
        # 1回目は失敗、2回目で揃う。1巡で諦めていないこと
        self.assertEqual(len(seen), 2)

    def test_never_sleeps_past_the_deadline(self):
        """
        締切をまたいで眠ると締切の意味が無くなる。間隔60分・締切まで5分の
        ときに60分眠ると、締切から55分過ぎて目を覚ます。
        """
        import datetime as _dt
        base = _dt.datetime(2026, 9, 18, 16, 5, tzinfo=self.W.JST)
        # 締切まで5分しか無い -> 間隔(3600秒)ではなく残り(300秒)で眠る
        self.assertEqual(
            self.W.nap_seconds(base, base + _dt.timedelta(minutes=5), 3600),
            300.0)
        # 締切まで十分ある -> 間隔どおり
        self.assertEqual(
            self.W.nap_seconds(base, base + _dt.timedelta(hours=3), 300), 300)
        # 締切を過ぎている -> 0秒にはせず最低1秒（忙しいループにしない）
        self.assertEqual(
            self.W.nap_seconds(base, base - _dt.timedelta(minutes=1), 300), 1.0)


class TestWatchlistAndExtraCodes(unittest.TestCase):
    """
    ウォッチリストは空にしてあり、日次の予測が「その日の上位」を足す。

    空で落ちると取得ワークフローが毎回失敗して他の更新まで止まるので、
    空を許すこと自体が要件。
    """

    def test_shipped_watchlist_is_empty(self):
        """
        既定のウォッチリストは空。特定銘柄を毎日見たいときだけ足す。
        サンプル銘柄が残っていると、予測とは無関係な銘柄が画面を占める。
        """
        import json
        import os
        path = os.path.join(ROOT, "scripts", "watchlist.json")
        d = json.load(open(path, encoding="utf-8"))
        self.assertEqual(d["stocks"], [])

    def test_load_watchlist_accepts_an_empty_list(self):
        import json
        import tempfile
        from jquants_data_fetcher import load_watchlist
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump({"stocks": []}, fh)
            path = fh.name
        try:
            self.assertEqual(load_watchlist(path), [])
        finally:
            os.unlink(path)

    def test_extra_codes_are_deduped_against_the_watchlist(self):
        """
        4桁と5桁が混ざっても同じ銘柄は1回だけ。
        normalize_code を通さずに比べると、同じ銘柄を2回取得して
        画面にも2つ並ぶ。
        """
        from jquants_data_fetcher import normalize_code
        targets = [{"code": "7203", "note": ""}]
        known = {normalize_code(t["code"]) for t in targets}
        for c in ["72030", "3845", "3845"]:
            if normalize_code(c) in known:
                continue
            known.add(normalize_code(c))
            targets.append({"code": c, "note": "x"})
        self.assertEqual([t["code"] for t in targets], ["7203", "3845"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
