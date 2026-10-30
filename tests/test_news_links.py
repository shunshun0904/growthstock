#!/usr/bin/env python3
"""
候補銘柄のニュースのリンクの確かめ（research/news_links.py）。ネットには出ない（偽の取得で確かめる）。

- 候補は規則（1.5σ の3本すべて95以上）を通った銘柄だけ。1本でも百分位が無ければ通さない。日は予測ファイルの dates の中だけ
- 1本のページの判定: HTTP 200・同じページ（転送でトップや検索に飛ばされていない）・題名にそのサイトの書き方のコード・
  題名にリンクの中身を表す言葉・題名に候補の銘柄名（全角・半角と「(株)」はそろえる）。理由の文
- 自動では開かないホスト（日経・株予報Pro・株探・みんかぶ）へは1件も接続しない（unchecked として URL だけ置く）。
  安全網として、取得の一番下の層でも断る。市場全体のリンクは置かない（2026-10-17 に画面から外した）
- 同じホストは間を空けて開く。一時的な失敗（接続できない・5xx・429）だけ1回やり直す
- 確かめ方そのものの確かめ: 通ったページを、わざと間違えた見出し・ほかの銘柄のコードで判定し直し、通ってしまったら出さない
  （題名がどの見出しの言葉も含むようなサイトの変化に気づく）。追加の接続はしない
- ログの行に銘柄コードを出さない。書き出すファイルの形

  python3 tests/test_news_links.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import news_links as N  # noqa: E402

RULE = ("lgbm", "xgb", "cat")                # 今日の戦略の規則の3本（2026-10-30 から 1.2σ。2026-10-23〜10-29 も同じ。2026-10-29〜10-30 は 1.5σ の3本）


def cand(code, date, pcts, name="テスト"):
    return {"code": code, "date": date, "name": name,
            "byModel": {a: {"pctHistorical": p} for a, p in zip(RULE, pcts) if p is not None}}


def page(title, body=""):
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>".encode("utf-8")


class FakeWeb:
    """URL → (状態, 最後の URL, 本文)。呼ばれた URL を控える。"""

    def __init__(self, pages=None, default=None):
        self.pages = pages or {}
        self.default = default
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url in self.pages:
            st, final, body = self.pages[url]
            return st, final or url, "text/html; charset=utf-8", body
        if self.default:
            return self.default(url)
        return 404, url, "", b""


def yahoo_ok(url):
    """Yahoo!ファイナンスの銘柄のページを、それらしい題名で返す（コードは URL から）。"""
    p = urllib.parse.urlsplit(url)
    if p.netloc == "finance.yahoo.co.jp" and p.path.startswith("/quote/"):
        code = p.path.split("/")[2].split(".")[0]
        kind = {"news": "ニュース・最新情報", "disclosure": "適時開示情報", "performance": "業績・財務・アナリスト予想",
                "forum": "掲示板"}[p.path.rsplit("/", 1)[1]]
        return 200, url, "text/html", page(f"テスト(株)【{code}】：{kind} - Yahoo!ファイナンス")
    return 404, url, "", b""


def opener(web):
    clock = {"t": 0.0}

    def sleep(sec):
        clock["t"] += sec

    return N.Opener(web, pause=1.5, retry_wait=5.0, sleep=sleep, clock=lambda: clock["t"]), clock


class TestRuleCodes(unittest.TestCase):
    def test_only_passing(self):
        pred = {"dates": ["2026-10-15", "2026-10-16"], "candidates": [
            cand("1111", "2026-10-16", [95.0, 99.0, 96.0]),          # 通る（ちょうど95）
            cand("2222", "2026-10-16", [94.9, 99.0, 99.0]),          # 1本だけ下
            cand("3333", "2026-10-16", [99.0, 99.0, None]),          # 1本欠ける
            cand("130a", "2026-10-15", [97.0, 97.0, 97.0]),          # 英字入り（大文字にそろえる）
            cand("1111", "2026-10-15", [96.0, 96.0, 96.0]),          # 同じ銘柄が別の日にも
            cand("4444", "2026-10-01", [99.0, 99.0, 99.0]),          # dates に無い日
            cand("55555", "2026-10-16", [99.0, 99.0, 99.0]),         # 5桁は使わない
        ]}
        self.assertEqual(N.rule_codes(pred), {"1111": ["2026-10-15", "2026-10-16"], "130A": ["2026-10-15"]})

    def test_rule_is_the_live_rule(self):
        self.assertEqual(tuple(N.LT.RULE_MODELS), RULE)
        self.assertEqual(N.LT.AGREE_PCT, 95.0)


class TestJudge(unittest.TestCase):
    URL = "https://finance.yahoo.co.jp/quote/7203.T/news"

    def j(self, st=200, final=None, title="トヨタ(株)【7203】：ニュース", **kw):
        kw.setdefault("mark", "【7203】")
        kw.setdefault("title_words", ("ニュース",))
        return N.judge(self.URL, st, final or self.URL, title, **kw)

    def test_ok(self):
        self.assertEqual(self.j(), (True, ""))
        self.assertEqual(self.j(final=self.URL + "/"), (True, ""))                 # 末尾の / は問わない

    def test_reasons(self):
        self.assertEqual(self.j(st=0), (False, "接続できない"))
        self.assertEqual(self.j(st=404), (False, "HTTP 404"))
        self.assertEqual(self.j(final="https://finance.yahoo.co.jp/"), (False, "別のページに移った"))
        self.assertEqual(self.j(final="https://finance.yahoo.co.jp/search?q=7203"), (False, "別のページに移った"))
        self.assertEqual(self.j(title="ページが見つかりません - Yahoo!ファイナンス"), (False, "題名に銘柄コードが無い"))
        # 題名にコードはあるが、見出しと違うページ（ウソのリンク）
        self.assertEqual(self.j(title="トヨタ(株)【7203】：株価・株式情報"), (False, "題名に「ニュース」が無い"))
        # コードが別の書き方（検索の結果の一覧など）では通さない
        self.assertEqual(self.j(title="「7203」の検索結果 ニュース"), (False, "題名に銘柄コードが無い"))

    def test_name(self):
        self.assertEqual(self.j(name="トヨタ"), (True, ""))
        self.assertEqual(self.j(name="ＴＯＹＯＴＡ自動車", title="TOYOTA自動車(株)【7203】：ニュース"), (True, ""))   # 全角・半角
        self.assertEqual(self.j(name="別の会社"), (False, "題名に銘柄名が無い"))
        self.assertTrue(N.name_in_title("", "何でも"))                                    # 名前が無ければ比べない
        # 長い名前はサイトが縮めることがあるので頭の8文字で比べる
        self.assertTrue(N.name_in_title("とても長い会社名ホールディングス", "とても長い会社名ＨＤ(株)【2222】"))
        self.assertFalse(N.name_in_title("短い名前", "短い名(株)【3333】"))

    def test_same_page_query(self):
        self.assertTrue(N.same_page("https://a.jp/p?scode=7203", "https://a.jp/p/?scode=7203&x=1"))
        self.assertFalse(N.same_page("https://a.jp/p?scode=7203", "https://a.jp/p?scode=7204"))
        self.assertFalse(N.same_page("https://a.jp/p?scode=7203", "https://b.jp/p?scode=7203"))


class TestNoAuto(unittest.TestCase):
    def test_hosts(self):
        for h in ("www.nikkei.com", "indexes.nikkei.co.jp", "kabuyoho.jp", "kabutan.jp", "minkabu.jp"):
            self.assertIn(h, N.NO_AUTO)
        self.assertTrue(N.auto_ok("https://finance.yahoo.co.jp/quote/7203.T/news"))
        self.assertFalse(N.auto_ok("https://www.nikkei.com/markets/"))

    def test_sources_are_consistent(self):
        ids = [s.id for s in N.SOURCES]
        self.assertEqual(len(ids), len(set(ids)))
        groups = {g for g, _ in N.GROUPS}
        for s in N.SOURCES:
            self.assertIn(s.group, groups, s.id)
            url = s.url.format(code="7203")
            self.assertTrue(url.startswith("https://"), s.id)
            if N.auto_ok(url):
                # 自動で確かめるリンクは、題名のコードの書き方と中身の言葉を必ず持つ
                self.assertTrue(s.mark and s.words, s.id)
        # 置かないサイト
        hosts = {N.host_of(s.url.format(code="7203")) for s in N.SOURCES}
        self.assertFalse(hosts & {"news.google.com", "x.com", "twitter.com", "shikiho.toyokeizai.net", "irbank.net"})
        # 自動で開くのは Yahoo!ファイナンスだけ（市場全体の JPX などは置かない）
        self.assertEqual({h for h in hosts if N.auto_ok(f"https://{h}/")}, {"finance.yahoo.co.jp"})

    def test_opener_refuses(self):
        op, _ = opener(FakeWeb(default=yahoo_ok))
        with self.assertRaises(ValueError):
            op.open("https://www.nikkei.com/nkd/company/news/?scode=7203")


class TestOpener(unittest.TestCase):
    def test_pause_per_host_and_retry(self):
        seq = {"n": 0}

        def flaky(url):
            seq["n"] += 1
            if "retry" in url and seq["n"] == 1:
                return 0, url, "", b""               # 1回目だけ接続できない
            return 200, url, "text/html", page("x")

        op, clock = opener(flaky)
        op.open("https://a.jp/retry")               # 0 → 5秒待って やり直し（同じホストなので 1.5秒の間も空く）
        self.assertEqual(op.requests, 2)
        t0 = clock["t"]
        op.open("https://b.jp/1")                    # 別のホストは待たない
        self.assertEqual(clock["t"], t0)
        op.open("https://b.jp/2")                    # 同じホストは 1.5秒空ける
        self.assertAlmostEqual(clock["t"] - t0, 1.5)

    def test_no_retry_on_404(self):
        web = FakeWeb()
        op, _ = opener(web)
        self.assertEqual(op.open("https://a.jp/x")[0], 404)
        self.assertEqual(len(web.calls), 1)


class TestBuild(unittest.TestCase):
    PRED = {"asOf": "2026-10-16", "dates": ["2026-10-16"], "candidates": [
        cand("7203", "2026-10-16", [99.0, 98.0, 97.0]),
        cand("285A", "2026-10-16", [96.0, 96.0, 96.0]),
        cand("9999", "2026-10-16", [50.0, 50.0, 50.0]),        # 規則を通らない → 開かない
    ]}

    def run_build(self, web):
        op, _ = opener(web)
        return N.build(self.PRED, op), web

    def test_states_and_no_access_to_no_auto(self):
        res, web = self.run_build(FakeWeb(default=yahoo_ok))
        self.assertEqual(sorted(res["stocks"]), ["285A", "7203"])
        # 自動では開かないホストには1件も接続していない。規則を通らない銘柄も開いていない
        hosts = {N.host_of(u) for u in web.calls}
        self.assertEqual(hosts, {"finance.yahoo.co.jp"})
        self.assertFalse(any("9999" in u for u in web.calls))
        links = res["stocks"]["7203"]["links"]
        st = {l["id"]: l["state"] for l in links}
        self.assertEqual({k for k, v in st.items() if v == N.OK},
                         {"yahoo_news", "yahoo_disclosure", "yahoo_perf", "yahoo_forum"})
        self.assertEqual({k for k, v in st.items() if v == N.UNCHECKED},
                         {"kabutan_news", "nikkei_news", "kabuyoho_analyst", "minkabu_analyst", "nikkei_kessan",
                          "minkabu_pick"})
        unchecked = [l for l in links if l["state"] == N.UNCHECKED]
        self.assertTrue(all(l["reason"] and l["href"].startswith("https://") for l in unchecked))
        self.assertEqual(res["stocks"]["285A"]["links"][0]["href"], "https://finance.yahoo.co.jp/quote/285A.T/news")
        s = res["summary"]
        self.assertEqual((s["stocks"], s["links"], s["ok"], s["ng"], s["unchecked"]), (2, 20, 8, 0, 12))
        self.assertEqual(s["requests"], 8)                  # 2銘柄 × 4本
        self.assertNotIn("market", res)

    def test_name_mismatch_is_ng(self):
        pred = {"asOf": "2026-10-16", "dates": ["2026-10-16"],
                "candidates": [cand("7203", "2026-10-16", [99.0, 98.0, 97.0], name="別の会社")]}
        op, _ = opener(FakeWeb(default=yahoo_ok))
        res = N.build(pred, op)
        st = {l["id"]: (l["state"], l["reason"]) for l in res["stocks"]["7203"]["links"]}
        self.assertEqual(st["yahoo_news"], (N.NG, "題名に銘柄名が無い"))
        self.assertEqual(res["summary"]["ok"], 0)

    def test_ng_is_recorded_with_reason(self):
        def broken(url):
            if url.endswith("/forum"):
                return 200, "https://finance.yahoo.co.jp/", "text/html", page("Yahoo!ファイナンス")
            if url.endswith("/disclosure"):
                return 200, url, "text/html", page("ページが見つかりません")
            return yahoo_ok(url)

        res, _ = self.run_build(FakeWeb(default=broken))
        st = {l["id"]: (l["state"], l["reason"]) for l in res["stocks"]["7203"]["links"]}
        self.assertEqual(st["yahoo_forum"], (N.NG, "別のページに移った"))
        self.assertEqual(st["yahoo_disclosure"], (N.NG, "題名に銘柄コードが無い"))
        self.assertEqual(res["summary"]["ng"], 4)

    def test_report_has_no_codes(self):
        res, _ = self.run_build(FakeWeb(default=yahoo_ok))
        out = "\n".join(N.report(res))
        self.assertNotIn("7203", out)
        self.assertNotIn("285A", out)
        self.assertIn("確かめて開けた 8", out)
        self.assertIn("32通り判定し直し、通ってしまったのは 0通り", out)

    def test_decoys(self):
        # 同じ銘柄のほかの3本の見出しと、ほかの銘柄のコード
        self.assertEqual(N.source_decoys(N.SOURCES[0], "7203", "285A"),
                         [("【7203】", ("適時開示",)), ("【7203】", ("業績",)), ("【7203】", ("掲示板",)),
                          ("【285A】", ("ニュース",))])

    def test_decoys_rejected_without_extra_requests(self):
        res, web = self.run_build(FakeWeb(default=yahoo_ok))
        s = res["summary"]
        # 2銘柄 × 4ページ × 4通り。どれも弾き、リンクの状態は変わらない
        self.assertEqual((s["decoyTried"], s["decoyPassed"]), (32, 0))
        self.assertEqual(s["ok"], 8)
        self.assertEqual(len(web.calls), 8)                  # 判定し直すだけで、開き直さない

    def test_lenient_titles_fail_closed(self):
        """どの見出しの言葉も題名にあるページ（見分けられない）は、1〜5 を通っても出さない。"""
        def lenient(url):
            st, final, ct, body = yahoo_ok(url)
            body = body.replace("</title>".encode(), " ニュース 適時開示 業績 掲示板</title>".encode())
            return st, final, ct, body

        res, _ = self.run_build(FakeWeb(default=lenient))
        links = [l for v in res["stocks"].values() for l in v["links"] if l["state"] != N.UNCHECKED]
        self.assertEqual({(l["state"], l["reason"]) for l in links}, {(N.NG, N.LIE_REASON)})
        self.assertEqual((res["summary"]["ok"], res["summary"]["ng"]), (0, 8))
        self.assertGreater(res["summary"]["decoyPassed"], 0)

    def test_code_decoy(self):
        """題名にほかの銘柄のコードも出るページ（どの銘柄か見分けられない）は出さない。候補が1つなら存在しないコードで試す。"""
        def many_codes(url):
            st, final, ct, body = yahoo_ok(url)
            return st, final, ct, body.replace("】".encode(), "】【285A】【7203】【0000】".encode(), 1)

        res, _ = self.run_build(FakeWeb(default=many_codes))
        self.assertEqual({l["reason"] for l in res["stocks"]["7203"]["links"] if l["state"] != N.UNCHECKED},
                         {N.LIE_REASON})
        pred = {"asOf": "2026-10-16", "dates": ["2026-10-16"],
                "candidates": [cand("7203", "2026-10-16", [99.0, 98.0, 97.0])]}
        op, _ = opener(FakeWeb(default=many_codes))
        one = N.build(pred, op)
        self.assertEqual(one["summary"]["ok"], 0)
        op, _ = opener(FakeWeb(default=yahoo_ok))
        self.assertEqual(N.build(pred, op)["summary"]["ok"], 4)

    def test_main_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            pred_p, out_p = os.path.join(d, "p.json"), os.path.join(d, "o.json")
            with open(pred_p, "w", encoding="utf-8") as fh:
                json.dump(self.PRED, fh)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(N.main(["--pred", pred_p, "--out", out_p], fetcher=yahoo_ok, pause=0.0), 0)
            self.assertNotIn("7203", buf.getvalue())
            with open(out_p, encoding="utf-8") as fh:
                d2 = json.load(fh)
            self.assertEqual(sorted(d2), sorted(["checkedAt", "asOf", "rule", "groups", "noAuto", "excluded", "stocks",
                                                 "summary"]))
            self.assertEqual(d2["rule"], {"models": list(RULE), "agreePct": 95.0})
            self.assertEqual([g["id"] for g in d2["groups"]], ["news", "disclosure", "forecast", "voice"])


if __name__ == "__main__":
    unittest.main()
