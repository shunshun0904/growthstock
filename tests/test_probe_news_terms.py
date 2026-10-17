#!/usr/bin/env python3
"""
規約の下調べの部品（research/exp/probe_news_terms.py）。ネットには出ない。

- 規約・ポリシーらしい文字のリンクだけを拾い、相対の URL を絶対にする（同じものは1回）
- robots.txt は User-agent: * の塊の行だけを返す

  python3 tests/test_probe_news_terms.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import probe_news_terms as T  # noqa: E402


class TestPolicyLinks(unittest.TestCase):
    def test_pick(self):
        text = ('<a href="/info/terms.html">利用規約</a><a href="/news">ニュース</a>'
                '<a href="https://x.jp/p.pdf"><b>サイトポリシー</b></a><a href="/info/terms.html">規約</a>')
        self.assertEqual(T.policy_links(text, "https://www.nikkei.com/nkd/company/"),
                         ["https://www.nikkei.com/info/terms.html", "https://x.jp/p.pdf"])


class TestStarBlock(unittest.TestCase):
    def test_only_star(self):
        robots = "User-agent: Googlebot\nDisallow: /a\n\nUser-agent: *\nDisallow: /b # c\nCrawl-delay: 5\n"
        self.assertEqual(T.star_block(robots), ["disallow: /b", "crawl-delay: 5"])


if __name__ == "__main__":
    unittest.main()
