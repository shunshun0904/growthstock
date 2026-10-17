#!/usr/bin/env python3
"""
リンクの形と規約の下調べの部品（research/exp/probe_news_links.py）。ネットには出ない。

- URL の組み立てで銘柄名をエンコードし、出力では銘柄コード・銘柄名を {code} / {name} に戻す
- 文字コードはヘッダ → meta → UTF-8 → cp932 の順
- 規約らしいリンクを同じホストから先に拾い、自動取得・転載に触れる文だけを抜く
- robots.txt は User-agent: * の塊だけを見て、使う道に当たる行を返す
- RSS の記事数と出どころを数える

  python3 tests/test_probe_news_links.py
"""
import datetime as dt
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import probe_news_links as P  # noqa: E402


class TestUrl(unittest.TestCase):
    def test_build_and_mask(self):
        u = P.build_url("https://news.google.com/search?q={name}&x={code}", "7203", "トヨタ自動車")
        self.assertIn("%E3%83%88", u)                          # エンコードされている
        self.assertEqual(P.mask(u, "7203", "トヨタ自動車"), "https://news.google.com/search?q={name}&x={code}")
        self.assertEqual(P.mask("https://a.jp/quote/7203.T/news", "7203", "x"), "https://a.jp/quote/{code}.T/news")
        # 長い数字の一部は置き換えない
        self.assertEqual(P.mask("https://a.jp/?id=172030", "7203", "x"), "https://a.jp/?id=172030")


class TestDecode(unittest.TestCase):
    def test_order(self):
        s = "株探ニュース"
        self.assertEqual(P.decode(s.encode("cp932"), "text/html; charset=Shift_JIS"), s)
        self.assertEqual(P.decode(b'<meta charset="euc-jp">' + s.encode("euc_jp")), '<meta charset="euc-jp">' + s)
        self.assertEqual(P.decode(s.encode("utf-8")), s)
        self.assertEqual(P.decode(s.encode("cp932")), s)               # 指定なしでも cp932 まで試す

    def test_title(self):
        self.assertEqual(P.page_title("<html><title>\n A &amp; B\n</title>"), "A & B")
        self.assertEqual(P.page_title("<html>"), "")


class TestTerms(unittest.TestCase):
    def test_links_same_host_first(self):
        text = ('<a href="https://other.jp/kiyaku">利用規約</a>'
                '<a href="/policy/terms.html"><span>ご利用にあたって</span></a>'
                '<a href="/news">ニュース</a>')
        got = P.terms_links(text, "https://kabutan.jp/stock/news?code=7203")
        self.assertEqual(got, ["https://kabutan.jp/policy/terms.html", "https://other.jp/kiyaku"])

    def test_rule_sentences(self):
        text = ("<p>当サイトの情報を無断で転載・複製することを禁止します。</p><script>var a='転載';</script>"
                "<p>天気の話。</p><p>プログラムによる自動的な取得を禁止します。</p>")
        got = P.rule_sentences(text)
        self.assertEqual(got, ["当サイトの情報を無断で転載・複製することを禁止します",
                               "プログラムによる自動的な取得を禁止します"])


class TestRobots(unittest.TestCase):
    def test_star_block_only(self):
        robots = ("User-agent: Googlebot\nDisallow: /stock/\n\n"
                  "User-agent: *\nDisallow: /stock/news\nAllow: /stock/\nDisallow: /search*\n# Disallow: /rss\n")
        # 当たる行は全部返す（どちらが勝つかは長い方。読む側が判断できるように両方出す）
        self.assertEqual(P.robots_lines(robots, ["/stock/news", "/rss/search"]),
                         ["Disallow: /stock/news", "Allow: /stock/"])
        self.assertEqual(P.robots_lines(robots, ["/search"]), ["Disallow: /search*"])
        self.assertEqual(P.robots_lines(robots, ["/other"]), [])


class TestRss(unittest.TestCase):
    def test_count(self):
        rss = ("<rss><channel><item><title>a</title><source url='x'>日本経済新聞</source></item>"
               "<item><source>日本経済新聞</source></item><item><source>株探</source></item>"
               "<item><title>no source</title></item></channel></rss>")
        n, top = P.count_items(rss)
        self.assertEqual(n, 4)
        self.assertEqual(top, ["日本経済新聞 2", "株探 1"])

    def test_weekdays(self):
        self.assertEqual(P.last_weekdays(dt.date(2026, 10, 19), 3),          # 月曜 → 金・木・水
                         [dt.date(2026, 10, 16), dt.date(2026, 10, 15), dt.date(2026, 10, 14)])


class TestAlnum(unittest.TestCase):
    def test_pick_alnum_code(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"candidates": [{"code": "7203", "name": "A"}, {"code": "999X", "name": "B"}]}, fh)
            self.assertEqual(P.alnum_sample(p), ("999X", "B"))
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"candidates": [{"code": "7203", "name": "A"}]}, fh)
            self.assertIsNone(P.alnum_sample(p))
        self.assertIsNone(P.alnum_sample(os.path.join(ROOT, "no_such_file.json")))


if __name__ == "__main__":
    unittest.main()
