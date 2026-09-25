#!/usr/bin/env python3
"""
research/probe_jsf.py の単体テスト（ネットワークには出ない）。

この probe は公開リポジトリの Actions で走り、ログも公開になる。日証金のデータは
利用条件で公開・第三者への提供が禁止されているので、**データの値が出力に出ないこと**が
最優先の要件。あわせて robots.txt と叩く数の上限を守ることを固定する。
"""
import contextlib
import io
import os
import sys
import unittest
import urllib.robotparser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import probe_jsf as P  # noqa: E402

#: 銘柄一覧の形（題・見出し・データ）。値は架空
LISTING = "\n".join([
    "銘柄別融資・貸株残高一覧表,2026/09/24",
    "申込日,コード,銘柄名,融資残高,貸株残高,差引残高",
    "2026/09/24,7203,テスト銘柄A,123456,654321,-530865",
    "2026/09/24,154A,テスト銘柄B,98765,4321,94444",
    "2026/09/24,9984,テスト銘柄C,,777,-777",
])
#: 銘柄ごとの履歴の形（先頭の列が日付）。値は架空
HISTORY = "\n".join([
    "日付,融資残高,貸株残高,品貸料率",
    "2023/09/25,111111,222222,0",
    "2024/01/05,333333,444444,0.05",
    "2026/09/24,555555,666666,0",
])
SECRET_VALUES = ["123456", "654321", "530865", "98765", "94444",
                 "111111", "222222", "333333", "444444", "555555", "666666"]


def printed(fn, *a) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()


class TestSanitize(unittest.TestCase):
    def test_keeps_dates_and_masks_other_numbers(self):
        out = P.sanitize("基準日 2026/09/24 融資残高 123456 株")
        self.assertIn("2026/09/24", out)
        self.assertNotIn("123456", out)
        self.assertIn("#", out)

    def test_dates_in_formats(self):
        self.assertEqual(P.dates_in("2026/09/24"), ["2026-09-24"])
        self.assertEqual(P.dates_in("2026-9-4"), ["2026-09-04"])
        self.assertEqual(P.dates_in("2026年9月24日"), ["2026-09-24"])
        self.assertEqual(P.dates_in("20260924"), ["2026-09-24"])
        self.assertEqual(P.dates_in("2026/13/40"), [])


class TestTable(unittest.TestCase):
    def test_listing_shape(self):
        d = P.describe_table(LISTING)
        self.assertEqual(d["data_rows"], 3)
        self.assertEqual(d["codes"], 3)            # 英字入りのコードも数える
        self.assertEqual(len(d["preamble"]), 2)    # 題と見出しの2行
        self.assertEqual(d["dates"][0], "2026-09-24")
        self.assertEqual(d["fill"][3], 2)          # 融資残高が空の行が1つ

    def test_history_date_range(self):
        d = P.describe_table(HISTORY)
        self.assertEqual(d["data_rows"], 3)
        self.assertEqual(d["first_col_dates"], ("2023-09-25", "2026-09-24", 3))

    def test_values_never_printed(self):
        for text in (LISTING, HISTORY):
            out = printed(P.show_table, P.describe_table(text))
            for v in SECRET_VALUES:
                self.assertNotIn(v, out, v)
            # 銘柄名も出さない（見出しの行だけを出す）
            self.assertNotIn("テスト銘柄", out)

    def test_values_masked_even_if_the_header_guess_fails(self):
        """データの行を見出しと取り違えても、数値は # に伏せて出る。"""
        text = "題,2026/09/24\nA,B,C\nx,123456,654321\n"
        out = printed(P.show_table, P.describe_table(text))
        self.assertNotIn("123456", out)
        self.assertNotIn("654321", out)


class TestPage(unittest.TestCase):
    HTML = """<html><head><title>DATA | テスト</title>
      <script src="/js/app.js"></script>
      <script>fetch("/api/stock/balance.csv?code=7203");</script></head>
      <body><a href="/download/">DATA</a>
      <a href="/sys-list/data/shina.csv">品貸料率一覧（CSV）</a>
      <a href="seigenichiran.csv">制限措置等一覧</a>
      <a href="/about/">説明</a>
      <form action="/search/result" method="post">
        <input type="text" name="code"><input type="date" name="date_from">
        <input type="hidden" name="_token" value="secret-token-value">
        <button type="submit" name="csv">CSV</button>
      </form></body></html>"""

    def test_links_forms_and_api_paths(self):
        p = P.parse_html(self.HTML.encode("utf-8"))
        self.assertEqual(p.title, "DATA | テスト")
        self.assertIn(("/sys-list/data/shina.csv", "品貸料率一覧（CSV）"), p.links)
        self.assertEqual(p.forms[0]["method"], "post")
        self.assertIn(("input", "date", "date_from"), p.forms[0]["fields"])
        self.assertIn("/api/stock/balance.csv?code=7203", p.api_paths)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            files = P.show_page(p, "https://www.taisyaku.jp/download/", only_files=True)
        self.assertIn("https://www.taisyaku.jp/sys-list/data/shina.csv", files)
        self.assertIn("https://www.taisyaku.jp/download/seigenichiran.csv", files)
        # 隠し項目の値（トークン）は出さない。項目名だけ
        self.assertNotIn("secret-token-value", out.getvalue())
        self.assertIn("date_from", out.getvalue())


class FakeResp:
    status = 200

    def __init__(self, body=b"ok"):
        self.body = body
        self.headers = {"Content-Type": "text/plain"}

    def read(self):
        return self.body

    def geturl(self):
        return "https://www.taisyaku.jp/x"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    def __init__(self):
        self.calls = []

    def open(self, req, timeout=30):
        self.calls.append(req.full_url)
        return FakeResp()


class TestFetcher(unittest.TestCase):
    def test_robots_disallow_is_respected(self):
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /search/"])
        op = FakeOpener()
        f = P.Fetcher(10, 0, robots=rp, opener=op)
        with contextlib.redirect_stdout(io.StringIO()):
            st, _, _, _ = f.get("https://www.taisyaku.jp/search/detail/pcsl/7203")
        self.assertIsNone(st)
        self.assertEqual(op.calls, [])
        with contextlib.redirect_stdout(io.StringIO()):
            st, _, _, _ = f.get("https://www.taisyaku.jp/download/")
        self.assertEqual(st, 200)

    def test_request_cap(self):
        op = FakeOpener()
        f = P.Fetcher(2, 0, opener=op)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(5):
                f.get("https://www.taisyaku.jp/download/")
        self.assertEqual(len(op.calls), 2)
        self.assertEqual(f.used, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
