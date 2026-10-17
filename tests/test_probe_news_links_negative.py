#!/usr/bin/env python3
"""
本物のページで弾けるかの probe（research/exp/probe_news_links_negative.py）の組み立て。ネットには出ない（偽の取得）。

- 取り違えは、URL はそのままで見出し（題名に要る言葉）だけを差し替えたもの
- 開くのは自動で開いてよいホスト（Yahoo!ファイナンス・JPX）だけ。同じ URL は1回だけ開く
- 偽のサイトが本物どおりなら、期待との食い違いは 0件。ログに候補の銘柄のコード・名前を出さない

  python3 tests/test_probe_news_links_negative.py
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import news_links as N  # noqa: E402
import probe_news_links_negative as P  # noqa: E402

NAMES = {"7203": "トヨタ自動車", "9999": "テスト商事", "285A": "テスト英字工業"}
KINDS = {"news": "ニュース・最新情報", "disclosure": "適時開示", "performance": "業績・財務", "forum": "掲示板"}


def fake(url):
    """本物の題名の書き方をまねた偽のサイト。無いコード・無いページは 404。"""
    u = urllib.parse.urlsplit(url)
    html = "text/html; charset=utf-8"
    if u.netloc == "finance.yahoo.co.jp" and u.path == "/news":
        return 200, url, html, "<title>経済ニュース - Yahoo!ファイナンス</title>".encode()
    if u.netloc == "www.jpx.co.jp":
        word = "空売り" if "short-selling" in u.path else "投資部門別"
        return 200, url, html, f"<title>{word}の集計 | 日本取引所グループ</title>".encode()
    m = re.match(r"^/quote/([0-9A-Z]{4})\.T/([a-z-]+)$", u.path)
    if u.netloc != "finance.yahoo.co.jp" or not m or m.group(1) not in NAMES or m.group(2) not in KINDS:
        return 404, url, html, b"<title>404</title>"
    code, kind = m.groups()
    return 200, url, html, f"<title>{NAMES[code]}(株)【{code}】：{KINDS[kind]} - Yahoo!ファイナンス</title>".encode()


def pred():
    hi = {m: {"pctHistorical": 99.0} for m in N.LT.RULE_MODELS}
    lo = {m: {"pctHistorical": 50.0} for m in N.LT.RULE_MODELS}
    return {"dates": ["2026-10-16"], "candidates": [
        {"date": "2026-10-16", "code": "9999", "name": NAMES["9999"], "byModel": hi},
        {"date": "2026-10-16", "code": "285a", "name": NAMES["285A"], "byModel": lo},
        {"date": "2026-10-16", "code": "12345", "name": "五桁", "byModel": lo},
    ]}


class TestShapes(unittest.TestCase):
    def test_as_label_and_pinned(self):
        lie = P.as_label(P.src("yahoo_news"), P.src("yahoo_forum"))
        self.assertEqual(lie.url, P.src("yahoo_news").url)
        self.assertEqual(lie.words, ("掲示板",))
        pin = P.pinned(P.src("yahoo_news"), "7203")
        self.assertEqual(pin.url.format(code="9999"), "https://finance.yahoo.co.jp/quote/7203.T/news")

    def test_others(self):
        other, alnum = P.others(pred())
        self.assertEqual(other, ("9999", NAMES["9999"]))     # 規則を通った銘柄
        self.assertEqual(alnum, ("285A", NAMES["285A"]))     # 英字入りの4桁（5桁は使わない）
        self.assertEqual(P.others({}), (None, None))

    def test_cases_only_auto_hosts(self):
        items = P.cases(*P.others(pred()))
        for _, s, code, _, _ in items:
            url = s.url if isinstance(s, N.Market) else s.url.format(code=code)
            self.assertTrue(N.auto_ok(url), url)
            self.assertIn(N.host_of(url), ("finance.yahoo.co.jp", "www.jpx.co.jp"))
        # 対照4・取り違え12・無効1 4・無効2 1・作った名前4・別の銘柄2 4・別の銘柄3 8・英字2・市場 9
        self.assertEqual(len(items), 48)
        self.assertEqual(sum(1 for x in items if x[4] == N.OK), 4 + 4 + 2 + 3)


class TestMemo(unittest.TestCase):
    def test_memo_keeps_success_only(self):
        calls = []

        def f(url):
            calls.append(url)
            return (503 if "busy" in url else 200), url, "", b""

        m = P.MemoFetch(f)
        m("https://a/x"), m("https://a/x"), m("https://a/busy"), m("https://a/busy")
        self.assertEqual(calls, ["https://a/x", "https://a/busy", "https://a/busy"])
        self.assertEqual(m.network, 3)


class TestRun(unittest.TestCase):
    def test_fake_site_no_mismatch_and_no_candidate_in_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "predictions.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(pred(), fh, ensure_ascii=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = P.main(fetcher=fake, pause=0, pred_path=path)
        log = out.getvalue()
        self.assertEqual(rc, 0, log)
        self.assertIn("食い違い 0件", log)
        self.assertNotIn("←", log)
        # 7203 の4ページ + 0000 の4ページ + 無いページ1 + 候補の4ページ + 英字2 + 市場3 = 18
        self.assertIn("ネットで開いたページ 18件", log)
        for s in ("9999", "285A", "285a", NAMES["9999"], NAMES["285A"]):
            self.assertNotIn(s, log)

    def test_lie_detected_when_site_titles_overlap(self):
        """題名の言葉が見分けられないサイトなら、取り違えを食い違いとして数える（probe が見逃さない）。"""
        def loose(url):
            st, final, ct, body = fake(url)
            return st, final, ct, body.replace("：".encode(), "：ニュース・適時開示・業績・掲示板 ".encode())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = P.main(fetcher=loose, pause=0, pred_path=os.devnull)
        self.assertEqual(rc, 1)
        self.assertIn("取り違え yahoo_news を yahoo_forum として: ok（期待 ng） ← 食い違い", out.getvalue())


if __name__ == "__main__":
    unittest.main()
