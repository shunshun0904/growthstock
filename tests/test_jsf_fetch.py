#!/usr/bin/env python3
"""
research/jsf_fetch.py の単体テスト（ネットワークに出ない。偽のサイトで叩く）。

固定すること
  - DATA ページの CSV（題・注記の行のあとに見出し）を表にし、コードは J-Quants の5桁、
    日付と数値は型に直す（「－」は欠測）
  - 積むときはキーが重なれば新しい方を残す（確報で速報を上書きする）
  - 銘柄ごとの過去は「銘柄のページ → 期間を入れて検索 → CSV」の順に叩き、済んだ銘柄は飛ばす。
    続けて失敗したら止める。銘柄のページが無い（404）銘柄はデータ無しとして二度と叩かない
  - 出力にデータの値を出さない（ログは公開）
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import pandas as pd  # noqa: E402

import jsf_fetch as J  # noqa: E402
import probe_jsf as PJ  # noqa: E402

#: 値は架空。見出しは実測（Probe JSF、2026-09-25）のもの
ZANDAKA = "\n".join([
    "銘柄別融資・貸株残高一覧表",
    "申込日,決済日,銘柄コード,銘柄名,取引所区分名,上場区分,速報／確報,融資新規株数,融資返済株数,"
    "融資残高株数,貸株新規株数,貸株返済株数,貸株残高株数,差引残高株数",
    "2026/09/24,2026/09/28,7203,テスト銘柄A,東証,,確報,\"1,200\",300,987654,10,20,111222,876432",
    "2026/09/24,2026/09/28,154A,テスト銘柄B,東証,,確報,5,6,333444,－,1,55,333389",
])
SHINA = "\n".join([
    "品貸料率一覧", "（注）テスト", "抽出条件,全体",
    "貸借申込日,決済日,コード,銘柄名,取引所区分,決算事由,決算等,貸借値段（円）,貸株超過株数,"
    "最高料率（円）,当日品貸料率（円）,当日品貸日数,前日品貸料率（円）,備考,制限,応札倍率ランク",
    "2026/09/24,2026/09/28,7203,テスト銘柄A,東証,,,2500,44556,24.00,0.05,1,0.10,,,A",
])
SEIGEN = "\n".join([
    "貸借取引銘柄別制限措置等一覧", "（注１）テスト",
    "直近発表,銘柄コード,銘柄名,実施措置,実施内容,通知日・実施日,後場停止",
    ",9984,テスト銘柄C,注意喚起,,2026/09/10,",
])
HIST_CSV = "\n".join([
    "銘柄コード,銘柄名,直後基準日,直近制限措置,直近臨時措置,直近特別措置,申込日,市場区分,貸借区分,"
    "融資新規（株）,融資返済（株）,融資残高（株）,貸株新規（株）,貸株返済（株）,貸株残高（株）,"
    "差引残高（株）,貸借値段（円）,品貸料率（品貸日数分/円）,品貸日数,品貸料率（年率換算/％）,"
    "最高料率（品貸日数分/円）,最低料率（品貸日数分/円）,応札ランク,制限措置,臨時措置,特別措置,"
    "新株引受・権利入札",
    "7203,テスト銘柄A,2026/09/30,,,,2023/09/26,東証プライム,貸借,1,2,424242,3,4,515151,-90909,2500,－,1,－,20,0,,,,,",
    "7203,テスト銘柄A,2026/09/30,,,,2026/09/24,東証プライム,貸借,5,6,434343,7,8,525252,-90909,2500,0.05,1,0.73,20,0,A,,,,",
])
SECRETS = ["987654", "876432", "333444", "44556", "424242", "515151", "434343", "525252", "テスト銘柄"]
DETAIL_HTML = """<html><body>
<form action="/app/stock/detail/{c}/search" method="post">
<input type="hidden" name="csrf_test_name" value="tok123">
<input type="text" name="mkYmdFrom" value=""><input type="text" name="mkYmdTo" value="">
<input type="radio" name="kjnYmdDays" value="7"><input type="radio" name="kjnYmdDays" value="" checked>
<input type="radio" name="trjoKbn" value="01" checked>
</form></body></html>"""


def printed(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


class Resp:
    def __init__(self, status, body=b"", headers=None, url="https://www.taisyaku.jp/"):
        self.status, self.body, self.url = status, body, url
        self.headers = headers or {"Last-Modified": "Fri, 25 Sep 2026 03:21:40 GMT"}

    def read(self):
        return self.body

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSite:
    """URL で応答を返す。404 は HTTPError で返す（本物の urllib と同じ）。"""

    def __init__(self, routes):
        self.routes = routes           # {url: (status, bytes)} or {url: callable(req)}
        self.calls = []

    def open(self, req, timeout=30):
        url = req.full_url
        self.calls.append((req.get_method(), url))
        r = self.routes.get(url)
        if callable(r):
            r = r(req)
        if r is None:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        status, body = r
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "err", {}, None)
        return Resp(status, body, url=url)


def cp932(s):
    return s.encode("cp932")


class TestRead(unittest.TestCase):
    def test_header_after_preamble_and_types(self):
        df = J.normalize(J.read_table(ZANDAKA, J.SPECS["balance"]["keys"]), "銘柄コード",
                         ("申込日", "決済日"))
        self.assertEqual(list(df["code"]), ["72030", "154A0"])
        self.assertEqual(df["申込日"].iloc[0], pd.Timestamp("2026-09-24"))
        self.assertEqual(df["融資新規株数"].iloc[0], 1200)          # 桁区切りを外す
        self.assertTrue(pd.isna(df["貸株新規株数"].iloc[1]))        # 「－」は欠測
        self.assertEqual(df["速報／確報"].iloc[0], "確報")          # 文字の列は文字のまま

    def test_blank_and_duplicate_headers_get_names(self):
        t = J.read_table("題\nA,B,－,B,\n1,2,3,4,5\n", ("A", "B"))
        self.assertEqual(list(t.columns), ["A", "B", "－", "B_3", "_col4"])

    def test_missing_header_raises(self):
        with self.assertRaises(ValueError):
            J.read_table("題\nX,Y\n1,2\n", ("申込日",))

    def test_dates(self):
        s = J.to_date(pd.Series(["2026/09/24", "2026-9-4", "20260924", "", "2026/13/40"]))
        self.assertEqual(list(s[:3]), [pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-04"),
                                       pd.Timestamp("2026-09-24")])
        self.assertTrue(s[3:].isna().all())

    def test_rank_stays_text(self):
        df = J.normalize(J.read_table(SHINA, J.SPECS["lending"]["keys"]), "コード",
                         ("貸借申込日", "決済日"))
        self.assertEqual(df["応札倍率ランク"].iloc[0], "A")
        self.assertAlmostEqual(df["当日品貸料率（円）"].iloc[0], 0.05)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_later_value_wins(self):
        p = os.path.join(self.dir, "x.parquet")
        a = pd.DataFrame({"d": pd.to_datetime(["2026-09-24"]), "code": ["72030"], "v": [1.0],
                          "k": ["速報"]})
        b = a.assign(v=[2.0], k=["確報"])
        J.merge_store(p, a, ["d", "code"])
        allf, added = J.merge_store(p, b, ["d", "code"])
        self.assertEqual(len(allf), 1)
        self.assertEqual(added, 0)
        self.assertEqual(float(allf["v"].iloc[0]), 2.0)
        self.assertEqual(str(allf["k"].iloc[0]), "確報")


class TestDaily(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_three_files_are_stored_without_printing_values(self):
        site = FakeSite({J.SPECS["balance"]["url"]: (200, cp932(ZANDAKA)),
                         J.SPECS["lending"]["url"]: (200, cp932(SHINA)),
                         J.SPECS["restrict"]["url"]: (200, cp932(SEIGEN))})
        f = PJ.Fetcher(10, 0, opener=site)
        m = J.load_manifest(self.dir)
        failed, out = printed(J.fetch_daily, f, self.dir, m)
        self.assertEqual(failed, 0)
        for sp in J.SPECS.values():
            self.assertTrue(os.path.exists(os.path.join(self.dir, sp["file"])), sp["file"])
        self.assertEqual(m["daily"]["balance"]["last"], "2026-09-24")
        r = pd.read_parquet(os.path.join(self.dir, "jsf_restrict.parquet"))
        self.assertIn("snap_date", r.columns)
        for v in SECRETS:
            self.assertNotIn(v, out, v)

    def test_changed_shape_is_reported_not_saved(self):
        site = FakeSite({J.SPECS["balance"]["url"]: (200, cp932("題\nX,Y\n1,2\n"))})
        f = PJ.Fetcher(10, 0, opener=site)
        failed, out = printed(J.fetch_daily, f, self.dir, J.load_manifest(self.dir))
        self.assertEqual(failed, 3)                   # 1本は形が違い、2本は 404
        self.assertIn("形が変わった", out)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "jsf_balance.parquet")))


def history_site(codes_ok, codes_404=(), csv_html=()):
    routes = {}
    for c4 in codes_ok:
        routes[f"{J.BASE}/app/stock/detail/{c4}-01"] = (200, DETAIL_HTML.format(c=c4).encode())
        routes[f"{J.BASE}/app/stock/detail/{c4}/search"] = (200, b"<html>ok</html>")
        body = b"<html>search</html>" if c4 in csv_html else cp932(HIST_CSV.replace("7203", c4))
        routes[f"{J.BASE}/app/stock/detail/{c4}/csv"] = (200, body)
    return FakeSite(routes)


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_session_flow_and_types(self):
        site = history_site(["7203"])
        f = PJ.Fetcher(10, 0, opener=site)
        st, df = J.fetch_one_history(f, "72030", pd.Timestamp("2023-09-26").date(),
                                     pd.Timestamp("2026-09-25").date())
        self.assertEqual(st, "ok")
        self.assertEqual(len(df), 2)
        self.assertEqual([m for m, _ in site.calls], ["GET", "POST", "GET"])
        self.assertTrue(pd.isna(df["品貸料率（品貸日数分/円）"].iloc[0]))
        self.assertAlmostEqual(df["品貸料率（品貸日数分/円）"].iloc[1], 0.05)
        self.assertEqual(df["応札ランク"].iloc[1], "A")

    def test_period_and_csrf_are_sent(self):
        seen = {}

        def search(req):
            seen["body"] = req.data.decode()
            return (200, b"<html>ok</html>")
        site = history_site(["7203"])
        site.routes[f"{J.BASE}/app/stock/detail/7203/search"] = search
        f = PJ.Fetcher(10, 0, opener=site)
        J.fetch_one_history(f, "72030", pd.Timestamp("2023-09-26").date(),
                            pd.Timestamp("2026-09-25").date())
        self.assertIn("csrf_test_name=tok123", seen["body"])
        self.assertIn("mkYmdFrom=2023%2F09%2F26", seen["body"])
        self.assertIn("mkYmdTo=2026%2F09%2F25", seen["body"])
        self.assertIn("kjnYmdDays=&", seen["body"] + "&")   # 既定（全日）のまま

    def test_missing_page_is_no_data_and_html_csv_is_error(self):
        f = PJ.Fetcher(10, 0, opener=history_site([], csv_html=()))
        self.assertEqual(J.fetch_one_history(f, "99990", pd.Timestamp("2023-09-26").date(),
                                             pd.Timestamp("2026-09-25").date())[0], "no_data")
        f = PJ.Fetcher(10, 0, opener=history_site(["7203"], csv_html=("7203",)))
        self.assertEqual(J.fetch_one_history(f, "72030", pd.Timestamp("2023-09-26").date(),
                                             pd.Timestamp("2026-09-25").date())[0],
                         "error:csv html")

    def test_resume_and_stop_after_consecutive_failures(self):
        codes = ["72030", "13010", "13050", "13060", "13080", "13090", "13100", "13110"]
        site = history_site(["7203"], csv_html=("1301", "1305", "1306", "1308", "1309", "1310",
                                                "1311"))
        for c4 in ("1301", "1305", "1306", "1308", "1309", "1310", "1311"):
            site.routes[f"{J.BASE}/app/stock/detail/{c4}-01"] = (200, DETAIL_HTML.format(c=c4).encode())
            site.routes[f"{J.BASE}/app/stock/detail/{c4}/search"] = (200, b"<html>ok</html>")
        f = PJ.Fetcher(100, 0, opener=site)
        m = J.load_manifest(self.dir)
        (done, added), out = printed(J.fetch_history, f, self.dir, m, codes, 20, 3.0)
        self.assertEqual(m["hist"]["72030"]["status"], "ok")
        self.assertEqual(done, 1 + J.MAX_CONSECUTIVE_FAIL)          # 5つ続けて失敗で止まる
        self.assertEqual(m["hist"]["13010"]["status"], "error")
        self.assertNotIn("13110", m["hist"])                        # 止めた後は叩いていない
        for v in SECRETS:
            self.assertNotIn(v, out, v)
        # 2回目は済んだ銘柄を飛ばし、失敗した銘柄をもう一度試す
        site2 = history_site(["1301"])
        f2 = PJ.Fetcher(100, 0, opener=site2)
        printed(J.fetch_history, f2, self.dir, m, codes[:2], 20, 3.0)
        self.assertEqual(m["hist"]["13010"]["status"], "ok")
        self.assertNotIn(("GET", f"{J.BASE}/app/stock/detail/7203-01"), site2.calls)
        h = pd.read_parquet(os.path.join(self.dir, "jsf_hist.parquet"))
        self.assertEqual(sorted(h["code"].unique()), ["13010", "72030"])


class TestTargets(unittest.TestCase):
    def test_population_order_first(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "t.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("# 見出し\n99990\n13010\n72030\n")
            self.assertEqual(J.read_targets(p, ["72030", "13010", "55550"]),
                             ["13010", "72030", "55550"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
