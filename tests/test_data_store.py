#!/usr/bin/env python3
"""
research/data_store.py の単体テスト。

差分取得の肝は「その日を取得しに行ったか」を記録することで、
行数で判定してはいけない（財務はその日の開示が0件でも取得済み）。
ここを間違えると毎回全期間を叩き直すことになるので、固定しておく。

  python3 tests/test_data_store.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import data_store  # noqa: E402


def days(*iso):
    return [dt.date.fromisoformat(d) for d in iso]


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_empty_manifest_when_missing(self):
        m = data_store.load_manifest(self.dir)
        for k in data_store.KINDS:
            self.assertEqual(m[k]["fetched_days"], [])

    def test_roundtrip(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-01-05"))
        data_store.save_manifest(self.dir, m)

        m2 = data_store.load_manifest(self.dir)
        self.assertEqual(m2["bars"]["fetched_days"], ["2024-01-04", "2024-01-05"])

    def test_missing_days_excludes_fetched(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04"))
        todo = data_store.missing_days(m, "bars", days("2024-01-04", "2024-01-05", "2024-01-09"))
        self.assertEqual([d.isoformat() for d in todo], ["2024-01-05", "2024-01-09"])

    def test_zero_row_day_is_still_marked_fetched(self):
        """
        財務のようにその日の開示が0件でも「取得済み」。
        行数で判定すると毎回叩き直すことになる。
        """
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "fins", days("2024-01-04"))   # 0件だった日
        self.assertEqual(data_store.missing_days(m, "fins", days("2024-01-04")), [])

    def test_mark_is_idempotent(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04"))
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-01-05"))
        self.assertEqual(m["bars"]["fetched_days"], ["2024-01-04", "2024-01-05"])

    def test_summary_reports_range(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "bars", days("2024-01-04", "2024-03-01"))
        data_store.save_manifest(self.dir, m)
        out = data_store.summarize(data_store.load_manifest(self.dir))
        self.assertIn("2024-01-04", out)
        self.assertIn("2024-03-01", out)
        self.assertIn("(未取得)", out)   # 他の種別は空


class TestMergeIntoYears(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["Date", "Code", "C"])

    def test_splits_by_year(self):
        df = self._frame([("2023-12-29", "00010", 100), ("2024-01-04", "00010", 110)])
        written = data_store.merge_into_years(self.dir, "bars", df)
        self.assertEqual(len(written), 2)
        self.assertTrue(os.path.exists(data_store.year_path(self.dir, "bars", 2023)))
        self.assertTrue(os.path.exists(data_store.year_path(self.dir, "bars", 2024)))

    def test_appends_without_losing_existing(self):
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 100)]))
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-05", "00010", 110)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 2)
        self.assertEqual(sorted(got["C"].tolist()), [100, 110])

    def test_duplicate_same_day_and_code_keeps_latest(self):
        """訂正が来たら後勝ちで上書きする。"""
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 100)]))
        data_store.merge_into_years(self.dir, "bars",
                                    self._frame([("2024-01-04", "00010", 999)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 1)
        self.assertEqual(got["C"].iloc[0], 999)

    def test_different_codes_same_day_both_kept(self):
        data_store.merge_into_years(self.dir, "bars", self._frame([
            ("2024-01-04", "00010", 100), ("2024-01-04", "00020", 200)]))
        got = pd.read_parquet(data_store.year_path(self.dir, "bars", 2024))
        self.assertEqual(len(got), 2)

    def test_empty_frame_is_noop(self):
        self.assertEqual(data_store.merge_into_years(self.dir, "bars", self._frame([])), [])
        self.assertEqual(data_store.merge_into_years(self.dir, "bars", None), [])

    def test_custom_date_column(self):
        """財務は DiscDate（開示日）で年を分ける。"""
        df = pd.DataFrame([("2024-05-10", "00010", 1, 1)],
                          columns=["DiscDate", "Code", "DiscNo", "Sales"])
        written = data_store.merge_into_years(self.dir, "fins", df, date_col="DiscDate")
        self.assertEqual(len(written), 1)
        self.assertIn("fins_2024", written[0])


class TestRowKeys(unittest.TestCase):
    """
    同じ日に同じ銘柄の行が複数あるデータを1行に潰さない（2026-09-24 に直した）。

    直す前は全種別を (日付列, Code) で重複除去していて、実測で業種別の空売り比率の
    97%（1日34行 -> 1行）、大量保有報告書の15%、大株主の8%、決算発表予定の6%、
    決算の3% を捨てていた（research/probe_dedupe_keys.py）。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def read(self, kind, year):
        return pd.read_parquet(data_store.year_path(self.dir, kind, year))

    def test_sector_rows_of_one_day_are_all_kept(self):
        """業種別の空売り比率は銘柄の列が無い。日付だけで潰すと1日1行になる。"""
        df = pd.DataFrame({"Date": ["2026-09-18"] * 3, "S33": ["0050", "3050", "9999"],
                           "SellExShortVa": [1.0, 2.0, 3.0]})
        data_store.merge_into_years(self.dir, "shortratio", df, "Date")
        self.assertEqual(len(self.read("shortratio", 2026)), 3)

    def test_two_filings_same_day_same_code_are_both_kept(self):
        """大量保有報告書: 同じ日・同じ銘柄に提出者が2人。"""
        df = pd.DataFrame({"DocId": ["S1", "S2"], "Code": ["72030", "72030"],
                           "SubDate": ["2026-02-13", "2026-02-13"],
                           "TotalShsRatio": [0.05, 0.07]})
        data_store.merge_into_years(self.dir, "lvshld", df, "SubDate")
        self.assertEqual(len(self.read("lvshld", 2026)), 2)

    def test_same_document_on_two_submission_dates_is_kept_twice(self):
        """
        大株主: 同じ DocId が別の提出日でも出る（実測で612文書）。DocId だけをキーに
        すると、全期間を取り直したときに重なって保存を止める（2026-09-24 に実際に止まった）。
        """
        df = pd.DataFrame({"DocId": ["S1", "S1"], "Code": ["72030", "72030"],
                           "SubDate": ["2025-06-27", "2025-07-10"]})
        data_store.merge_into_years(self.dir, "mjrshld", df, "SubDate")
        self.assertEqual(len(self.read("mjrshld", 2025)), 2)

    def test_same_day_disclosures_are_both_kept(self):
        """決算: 同じ日・同じ銘柄に決算短信と業績予想の修正。"""
        df = pd.DataFrame({"DiscNo": [1, 2], "Code": ["72030", "72030"],
                           "DiscDate": ["2024-05-14", "2024-05-14"],
                           "DocType": ["FY", "EarnForecastRevision"]})
        data_store.merge_into_years(self.dir, "fins", df, "DiscDate")
        self.assertEqual(len(self.read("fins", 2024)), 2)

    def test_short_sale_reports_keep_every_distinct_row(self):
        """空売り残高報告は ID が無い。同じ報告者・同じ計算日でも比率が違えば別の行。"""
        df = pd.DataFrame({"DiscDate": ["2019-06-27"] * 3, "CalcDate": ["2019-06-25"] * 3,
                           "Code": ["72030"] * 3, "SSName": ["A", "A", "B"],
                           "ShrtPosToSO": [0.0052, 0.0061, 0.0052]})
        data_store.merge_into_years(self.dir, "shortsale", df, "DiscDate")
        self.assertEqual(len(self.read("shortsale", 2019)), 3)
        # 同じ日を取り直しても増えない（まったく同じ行は1行）
        data_store.merge_into_years(self.dir, "shortsale", df, "DiscDate")
        self.assertEqual(len(self.read("shortsale", 2019)), 3)

    def test_refetch_replaces_by_key(self):
        """取り直すと同じキーの行は後勝ち（訂正を反映）で、行は増えない。"""
        a = pd.DataFrame({"DocId": ["S1", "S2"], "Code": ["72030"] * 2,
                          "SubDate": ["2026-02-13"] * 2, "TotalShsRatio": [0.05, 0.07]})
        b = a.assign(TotalShsRatio=[0.05, 0.08])
        data_store.merge_into_years(self.dir, "lvshld", a, "SubDate")
        data_store.merge_into_years(self.dir, "lvshld", b, "SubDate")
        got = self.read("lvshld", 2026).set_index("DocId")["TotalShsRatio"]
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(float(got["S2"]), 0.08)

    def test_nested_columns_are_compared_by_content(self):
        """大株主の一覧（入れ子の列）があっても重複の判定ができる。"""
        df = pd.DataFrame({"DocId": ["S1"], "Code": ["72030"], "SubDate": ["2025-06-27"],
                           "Hldrs": [[{"HldrName": "x", "ShsRatio": 0.1}]]})
        data_store.merge_into_years(self.dir, "mjrshld", df, "SubDate")
        data_store.merge_into_years(self.dir, "mjrshld", df, "SubDate")
        self.assertEqual(len(self.read("mjrshld", 2025)), 1)

    def test_a_key_that_would_collapse_rows_refuses_to_save(self):
        """今回取った行の中でキーが重なり、中身が違う = キーの決め方が誤り。保存しない。"""
        df = pd.DataFrame({"Date": ["2024-01-04", "2024-01-04"], "Code": ["00010"] * 2,
                           "C": [100, 999]})
        with self.assertRaises(ValueError):
            data_store.merge_into_years(self.dir, "bars", df, "Date")
        self.assertFalse(os.path.exists(data_store.year_path(self.dir, "bars", 2024)))

    def test_identical_rows_in_one_batch_are_merged(self):
        df = pd.DataFrame({"Date": ["2024-01-04"] * 2, "Code": ["00010"] * 2, "C": [100, 100]})
        data_store.merge_into_years(self.dir, "bars", df, "Date")
        self.assertEqual(len(self.read("bars", 2024)), 1)

    def test_unregistered_kind_is_not_saved(self):
        df = pd.DataFrame({"Date": ["2024-01-04"], "Code": ["00010"]})
        with self.assertRaises(KeyError):
            data_store.merge_into_years(self.dir, "新しい種別", df, "Date")

    def test_every_kind_the_ingestion_writes_has_a_key(self):
        import jq_bulk as J
        kinds = {"bars", "fins", "margin", "topix", "indices", "master_hist"} | set(J.DAILY_KINDS)
        self.assertEqual(sorted(kinds - set(data_store.ROW_KEYS)), [])


class TestResetKind(unittest.TestCase):
    """
    取得する列を増やしたとき、既存 parquet には新しい列が入っていないのに
    manifest 上は「取得済み」なので incremental では永久に取り直されない。
    reset_kind はその状態を解消する。
    株価を巻き込むと数時間かかるので、指定した種別だけを消すこと。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for n in ("fins_2020.parquet", "fins_2021.parquet",
                  "bars_2020.parquet", "margin_2020.parquet"):
            open(os.path.join(self.dir, n), "w").write("x")
        # 実際の manifest の形に合わせる（キーは fetched_days）
        self.manifest = {"fins": {"fetched_days": ["2020-01-06"]},
                         "bars": {"fetched_days": ["2020-01-06"]},
                         "margin": {"fetched_days": ["2020-01-10"]}}

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_removes_only_the_named_kind(self):
        removed = data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertEqual(removed, ["fins_2020.parquet", "fins_2021.parquet"])
        left = sorted(os.listdir(self.dir))
        self.assertEqual(left, ["bars_2020.parquet", "margin_2020.parquet"])

    def test_clears_only_that_kind_from_the_manifest(self):
        data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertNotIn("fins", self.manifest)
        self.assertIn("bars", self.manifest)
        self.assertIn("margin", self.manifest)

    def test_reset_makes_every_day_missing_again(self):
        """取り直しの目的はここ。消した後は全営業日が未取得になる。"""
        days = [dt.date(2020, 1, 6), dt.date(2020, 1, 7)]
        data_store.mark_fetched(self.manifest, "fins", days)
        self.assertEqual(data_store.missing_days(self.manifest, "fins", days), [])
        data_store.reset_kind(self.dir, self.manifest, "fins")
        self.assertEqual(data_store.missing_days(self.manifest, "fins", days), days)

    def test_unknown_kind_is_harmless(self):
        removed = data_store.reset_kind(self.dir, self.manifest, "nosuch")
        self.assertEqual(removed, [])
        self.assertEqual(len(os.listdir(self.dir)), 4)


class TestKindDispatch(unittest.TestCase):
    """
    jq_bulk の日付ループが種別をどう振り分けるか。

    以前は catch-all の else で /markets/margin-interest を叩いており、
    master がそこに落ちて信用残のデータを master_YYYY.parquet に
    書き込んでいた（2,425日ぶん・約38分を毎回浪費）。
    種別を増やしたときに同じ事故が起きないようにする。
    """

    def _source(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "research", "jq_bulk.py")
        return open(path, encoding="utf-8").read()

    def test_no_catch_all_else_for_endpoints(self):
        """未知の種別は黙って別のエンドポイントを叩かず、落ちること。"""
        src = self._source()
        self.assertIn("日付ループで扱えない種別です", src)

    def test_master_is_excluded_from_the_day_loop(self):
        """master は日付ごとの蓄積ではないので、日付ループに入れない。"""
        src = self._source()
        self.assertIn('if name == "master":', src)

    def test_master_is_not_a_tracked_kind(self):
        """master は manifest の追跡対象ではない。
        追跡対象にすると missing_days が毎回「全日未取得」を返す。"""
        self.assertNotIn("master", data_store.KINDS)


class TestConfirmedDays(unittest.TestCase):
    """
    公表前に叩いた日を「取得済み」にしないこと。

    missing_days は「候補 − 記録」なので、一度記録した日は二度と候補に
    入らない。取り込みを 16:05 JST に前倒ししたことで、指数(16:30)や
    財務(18:00)を公表前に叩く日ができた。そこで0件を取得済みにすると、
    **その日のデータは永久に失われる**。実際に信用残がそうなっていた
    （記録は 2026-09-07 まであるのに実データは 2026-08-28 が最後）。

    かといって「0件なら記録しない」にすると、開示が本当に0件の日を
    毎回叩き続けることになる。だから公表ラグで線を引く。
    """

    def test_rows_present_is_always_recorded(self):
        today = dt.date(2026, 9, 18)
        got = days("2026-09-18")
        out = data_store.confirmed_days("bars", days("2026-09-18"), got, today)
        self.assertEqual(out, days("2026-09-18"))

    def test_today_with_no_rows_is_held_back(self):
        today = dt.date(2026, 9, 18)
        out = data_store.confirmed_days("indices", days("2026-09-18"), [], today)
        self.assertEqual(out, [])

    def test_past_day_with_no_rows_is_recorded(self):
        """
        公表ラグを過ぎても0件なら、本当に0件（祝日・開示なし）。
        ここで記録しないと、その日を永久に叩き続けることになる。
        """
        today = dt.date(2026, 9, 18)
        out = data_store.confirmed_days("fins", days("2026-09-17"), [], today)
        self.assertEqual(out, days("2026-09-17"))

    def test_margin_keeps_retrying_for_a_week(self):
        """信用残は週次で、金曜ぶんが翌週に出る。当日だけの猶予では足りない。"""
        today = dt.date(2026, 9, 18)          # 金
        # 同じ週の金曜と、10日以上前の金曜
        out = data_store.confirmed_days(
            "margin", days("2026-09-18", "2026-09-04"), [], today)
        self.assertEqual(out, days("2026-09-04"))

    def test_unknown_kind_falls_back_to_today_only(self):
        today = dt.date(2026, 9, 18)
        self.assertEqual(
            data_store.confirmed_days("nazo", days("2026-09-18"), [], today), [])
        self.assertEqual(
            data_store.confirmed_days("nazo", days("2026-09-17"), [], today),
            days("2026-09-17"))

    def test_order_is_preserved(self):
        today = dt.date(2026, 9, 18)
        req = days("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-18")
        out = data_store.confirmed_days("bars", req, days("2026-09-15"), today)
        self.assertEqual(out, days("2026-09-14", "2026-09-15", "2026-09-16"))


class TestMarginCandidates(unittest.TestCase):
    """
    信用残は週次公表で、残高は**その週の最終営業日**時点のもの。

    以前は「金曜を全部足し、金曜が無い週はその週の適当な日」を候補に
    していた。この「適当な日」が週の先頭（月曜）になるため、金曜が祝日の
    週は月曜を叩いて0件のまま取得済みになり、木曜ぶんが永久に失われた。
    保存データを数えると10年ぶんで23週がこれで空いていた。
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import jq_bulk
        self.jq = jq_bulk

    def test_normal_week_picks_friday(self):
        wk = days("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17",
                  "2026-09-18")
        self.assertEqual(self.jq.margin_candidates(wk), days("2026-09-18"))

    def test_holiday_friday_picks_thursday_not_monday(self):
        """2026-03-20 は春分の日。週の最終営業日は木曜 03-19。"""
        wk = days("2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19")
        self.assertEqual(self.jq.margin_candidates(wk), days("2026-03-19"))

    def test_one_day_per_week(self):
        wk = days("2026-09-07", "2026-09-08", "2026-09-11",
                  "2026-09-14", "2026-09-18")
        self.assertEqual(self.jq.margin_candidates(wk),
                         days("2026-09-11", "2026-09-18"))

    def test_year_boundary_uses_iso_week(self):
        """2025-12-29〜2026-01-02 は ISO では同じ週（2026年第1週）。"""
        wk = days("2025-12-29", "2025-12-30", "2026-01-05", "2026-01-06")
        self.assertEqual(self.jq.margin_candidates(wk),
                         days("2025-12-30", "2026-01-06"))

    def test_empty(self):
        self.assertEqual(self.jq.margin_candidates([]), [])


class TestFailedDaysAreNotRecorded(unittest.TestCase):
    """
    問い合わせ自体が失敗した日を「取得済み」にしないこと。

    _fetch_by_day は失敗した日を読み飛ばすが、記録は todo 全部に付いて
    いた。一過性の 503 で1日ぶんが永久に失われる（記録した日は
    missing_days の候補から消える）。
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import jq_bulk
        self.jq = jq_bulk

    def test_failed_day_is_held_back(self):
        m = {"bars": {"fetched_days": []}}
        df = pd.DataFrame({"Date": ["2026-09-16"], "Code": ["13010"]})
        self.jq._record_fetched(
            m, "bars", days("2026-09-16", "2026-09-17"), df, "Date",
            today=dt.date(2026, 9, 18), failed=days("2026-09-17"))
        self.assertEqual(m["bars"]["fetched_days"], ["2026-09-16"])

    def test_fetch_by_day_reports_which_days_failed(self):
        from jquants_data_fetcher import JQuantsError

        class Flaky:
            def get_paginated(self, path, params):
                if params["date"] == "2026-09-17":
                    raise JQuantsError("503")
                return [{"Date": params["date"], "Code": "13010"}]

        df = self.jq._fetch_by_day(Flaky(), "/x",
                                   days("2026-09-16", "2026-09-17"),
                                   ["Date", "Code"], "bars")
        self.assertEqual(len(df), 1)
        self.assertEqual(self.jq.LAST_FAILED_DAYS, days("2026-09-17"))

    def test_failures_are_cleared_between_calls(self):
        """前回の失敗が残ると、無関係な日まで記録されなくなる。"""
        from jquants_data_fetcher import JQuantsError

        class Once:
            def __init__(self):
                self.first = True

            def get_paginated(self, path, params):
                if self.first:
                    self.first = False
                    raise JQuantsError("503")
                return [{"Date": params["date"], "Code": "13010"}]

        c = Once()
        self.jq._fetch_by_day(c, "/x", days("2026-09-16"), ["Date", "Code"],
                              "bars")
        self.assertEqual(self.jq.LAST_FAILED_DAYS, days("2026-09-16"))
        self.jq._fetch_by_day(c, "/x", days("2026-09-17"), ["Date", "Code"],
                              "bars")
        self.assertEqual(self.jq.LAST_FAILED_DAYS, [])


class TestForgetDays(unittest.TestCase):
    """
    取得記録だけを外して取り直せること（parquet は消さない）。

    公表前に叩いて0件のまま取得済みになった日を戻すための操作。
    reset_kind は保存データごと消すので、取り直しが途中で落ちると
    穴が開く。こちらは取れなければ現状維持で済む。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _manifest(self):
        m = data_store.load_manifest(self.dir)
        data_store.mark_fetched(m, "margin", days(
            "2026-08-28", "2026-09-04", "2026-09-11", "2026-09-14"))
        return m

    def test_forgets_only_from_the_given_date(self):
        m = self._manifest()
        dropped = data_store.forget_days(m, "margin",
                                         since=dt.date(2026, 9, 1))
        self.assertEqual(dropped,
                         ["2026-09-04", "2026-09-11", "2026-09-14"])
        self.assertEqual(m["margin"]["fetched_days"], ["2026-08-28"])

    def test_forgets_everything_without_a_date(self):
        m = self._manifest()
        dropped = data_store.forget_days(m, "margin")
        self.assertEqual(len(dropped), 4)
        self.assertEqual(m["margin"]["fetched_days"], [])

    def test_forgotten_days_come_back_as_candidates(self):
        """外した日が missing_days に戻ること。これが目的そのもの。"""
        m = self._manifest()
        cand = days("2026-08-28", "2026-09-04", "2026-09-11")
        self.assertEqual(data_store.missing_days(m, "margin", cand), [])
        data_store.forget_days(m, "margin", since=dt.date(2026, 9, 1))
        self.assertEqual(data_store.missing_days(m, "margin", cand),
                         days("2026-09-04", "2026-09-11"))

    def test_does_not_touch_saved_files(self):
        df = pd.DataFrame({"Date": ["2026-08-28"], "Code": ["13010"],
                           "LongVol": [1.0], "ShrtVol": [2.0]})
        data_store.merge_into_years(self.dir, "margin", df, "Date")
        before = sorted(os.listdir(self.dir))
        m = self._manifest()
        data_store.forget_days(m, "margin")
        self.assertEqual(sorted(os.listdir(self.dir)), before)


class TestRecordFetched(unittest.TestCase):
    """
    jq_bulk 側の配線。取得した DataFrame から「行が取れた日」を拾って
    data_store.confirmed_days に渡すところ。
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import jq_bulk
        self.jq = jq_bulk
        # 失敗した日はモジュール変数で持ち回している。前のテストの残りが
        # 効いてしまわないように消す（本番では _fetch_by_day が毎回消す）
        jq_bulk.LAST_FAILED_DAYS = []

    def test_holds_back_todays_empty_fetch(self):
        m = {"indices": {"fetched_days": []}}
        empty = pd.DataFrame(columns=["Date", "Code", "C"])
        self.jq._record_fetched(m, "indices", days("2026-09-17", "2026-09-18"),
                                empty, "Date", today=dt.date(2026, 9, 18))
        self.assertEqual(m["indices"]["fetched_days"], ["2026-09-17"])

    def test_records_today_when_rows_came_back(self):
        m = {"indices": {"fetched_days": []}}
        df = pd.DataFrame({"Date": ["2026-09-18"], "Code": ["0040"], "C": [1.0]})
        self.jq._record_fetched(m, "indices", days("2026-09-18"), df, "Date",
                                today=dt.date(2026, 9, 18))
        self.assertEqual(m["indices"]["fetched_days"], ["2026-09-18"])

    def test_empty_frame_without_the_date_column_does_not_crash(self):
        """
        FIN_COLS / MASTER_COLS は None なので、0件のときの DataFrame には
        列が1つも無い。ここで落ちると取り込み全体が止まる。
        """
        m = {"fins": {"fetched_days": []}}
        self.jq._record_fetched(m, "fins", days("2026-09-18"),
                                pd.DataFrame(), "DiscDate",
                                today=dt.date(2026, 9, 18))
        self.assertEqual(m["fins"]["fetched_days"], [])

    def test_fins_uses_the_disclosure_date_column(self):
        m = {"fins": {"fetched_days": []}}
        df = pd.DataFrame({"DiscDate": ["2026-09-18"], "Code": ["13010"]})
        self.jq._record_fetched(m, "fins", days("2026-09-18"), df, "DiscDate",
                                today=dt.date(2026, 9, 18))
        self.assertEqual(m["fins"]["fetched_days"], ["2026-09-18"])


class TestCalendarNote(unittest.TestCase):
    """
    取り込みが「今日は営業日か」を manifest に書き残すところ。

    営業日カレンダーを引いているのは取り込みだけなので、ここが唯一の
    情報源になる。予測側が自分で祝日を判断しないための信号。
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import jq_bulk
        self.jq = jq_bulk
        self.days = days("2026-09-16", "2026-09-17", "2026-09-18")

    def test_trading_day(self):
        c = self.jq.calendar_note(self.days, dt.date(2026, 9, 18),
                                  today=dt.date(2026, 9, 18))
        self.assertIs(c["isTradingDay"], True)
        self.assertEqual(c["lastTradingDay"], "2026-09-18")

    def test_holiday(self):
        """2026-09-21 は敬老の日。カレンダーに入らないので非営業日。"""
        c = self.jq.calendar_note(self.days, dt.date(2026, 9, 21),
                                  today=dt.date(2026, 9, 21))
        self.assertIs(c["isTradingDay"], False)
        self.assertEqual(c["lastTradingDay"], "2026-09-18")

    def test_past_range_cannot_judge_today(self):
        """
        過去日を指定して取り直した場合、days に今日は入らない。
        「入っていない＝非営業日」と読むと誤判定になるので None にする。
        """
        c = self.jq.calendar_note(self.days, dt.date(2026, 9, 18),
                                  today=dt.date(2026, 9, 21))
        self.assertIsNone(c["isTradingDay"])

    def test_no_days(self):
        c = self.jq.calendar_note([], dt.date(2026, 9, 21),
                                  today=dt.date(2026, 9, 21))
        self.assertIsNone(c["isTradingDay"])
        self.assertIsNone(c["lastTradingDay"])


class TestTradingDayGate(unittest.TestCase):
    """
    非営業日に日次予測を飛ばす判定。

    **飛ばす側に倒してはいけない。** 取り込みが壊れている日を「休場だから」
    と飛ばすと、気づかないまま画面が止まる。judge は3条件すべてが
    揃ったときだけ False（飛ばす）を返し、それ以外は必ず True を返す。
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "research"))
        import trading_day_gate
        self.g = trading_day_gate
        self.now = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.timezone.utc)

    def _manifest(self, **over):
        cal = {"asOfJst": "2026-09-21", "requestedTo": "2026-09-21",
               "isTradingDay": False, "lastTradingDay": "2026-09-18"}
        cal.update(over.pop("calendar", {}))
        m = {"calendar": cal,
             "updatedAt": "2026-09-21T11:30:00+00:00"}
        m.update(over)
        return m

    def test_holiday_with_complete_data_is_skipped(self):
        run, why = self.g.decide(self._manifest(), dt.date(2026, 9, 18),
                                 self.now)
        self.assertFalse(run, why)

    def test_trading_day_runs(self):
        m = self._manifest(calendar={"isTradingDay": True})
        run, _ = self.g.decide(m, dt.date(2026, 9, 21), self.now)
        self.assertTrue(run)

    def test_unknown_runs(self):
        m = self._manifest(calendar={"isTradingDay": None})
        run, _ = self.g.decide(m, dt.date(2026, 9, 18), self.now)
        self.assertTrue(run)

    def test_old_manifest_without_calendar_runs(self):
        run, _ = self.g.decide({"updatedAt": "2026-09-21T11:30:00+00:00"},
                               dt.date(2026, 9, 18), self.now)
        self.assertTrue(run)

    def test_stale_manifest_runs(self):
        """
        取り込みが今日走っていないなら、その calendar は昨日以前の判断。
        休場日でも飛ばさず、鮮度チェックに任せる。
        """
        m = self._manifest(updatedAt="2026-09-19T11:30:00+00:00")
        run, why = self.g.decide(m, dt.date(2026, 9, 18), self.now)
        self.assertTrue(run)
        self.assertIn("古い", why)

    def test_data_behind_last_trading_day_runs(self):
        """
        ここが肝。休場日でも、保存データが直近の営業日に届いていなければ
        取り込みが壊れている。飛ばさずに鮮度チェックで落とす。
        """
        run, why = self.g.decide(self._manifest(), dt.date(2026, 9, 11),
                                 self.now)
        self.assertTrue(run)
        self.assertIn("取り込みを疑う", why)

    def test_missing_last_trading_day_runs(self):
        m = self._manifest(calendar={"lastTradingDay": None})
        run, _ = self.g.decide(m, dt.date(2026, 9, 18), self.now)
        self.assertTrue(run)

    def test_no_bars_runs(self):
        run, _ = self.g.decide(self._manifest(), None, self.now)
        self.assertTrue(run)

    def test_broken_updated_at_runs(self):
        m = self._manifest(updatedAt="いつか")
        run, _ = self.g.decide(m, dt.date(2026, 9, 18), self.now)
        self.assertTrue(run)

    def test_main_always_returns_zero(self):
        """
        判定できない事情が何であれ exit 0。ここが新しい失敗の入口に
        なってはいけない（止めるのは鮮度チェックの仕事）。
        """
        empty = tempfile.mkdtemp()
        try:
            self.assertEqual(self.g.main(["--data-dir", empty]), 0)
            with open(os.path.join(empty, "manifest.json"), "w") as fh:
                fh.write("これは JSON ではない")
            self.assertEqual(self.g.main(["--data-dir", empty]), 0)
        finally:
            shutil.rmtree(empty, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
