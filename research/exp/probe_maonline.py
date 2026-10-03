#!/usr/bin/env python3
"""
M&A Online（maonline.jp）の企業データベースを、M&A の特徴量の元にできるかの下見。

運用者の依頼（2026-10-03）「M&A の特徴量を入れてもよい。何もなければ https://maonline.jp/db/companies/ から
クローリングして取得してほしい。リクエスト頻度は落として」。作業環境からは maonline.jp に出られない（egress）ので、
Actions で動かす。取得するのは下の 5 ページまでで、10 秒に 1 回。

見るもの（HTML そのものはログに出さない。構造と、規約の該当条項だけ）
  1. robots.txt: /db/ が許可されているか
  2. 利用規約: 複製・転載・自動取得・データベース化に触れる条項
  3. /db/companies/: 企業ページへのリンクの形、ページ送り、総件数の表記
  4. 企業ページ 1 つ: 見出し・表の項目名・案件へのリンクの形（値は出さない）
  5. 案件ページ 1 つ（あれば）: 項目名だけ

使い方（Actions の run-experiment で）
    python3 research/exp/probe_maonline.py
"""
from __future__ import annotations

import re
import sys
import time
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests

BASE = "https://maonline.jp"
INDEX = BASE + "/db/companies/"
UA = "Mozilla/5.0 (compatible; growthstock-research/0.1; polite probe, 1 request per 10 s)"
INTERVAL = 10.0
MAX_REQUESTS = 5
TIMEOUT = 30
KEYWORDS = ("複製", "転載", "自動", "クロール", "クローラ", "スクレイピング", "データベース", "商用", "営利", "禁止", "再利用", "二次利用")

_last = 0.0
_count = 0


def fetch(url: str) -> Tuple[Optional[int], str, float]:
    """1 回取る。前の取得から INTERVAL 秒あける。MAX_REQUESTS を超えたら取らない。"""
    global _last, _count
    if _count >= MAX_REQUESTS:
        print(f"  [skip] 取得回数の上限 {MAX_REQUESTS} に達したので取らない: {url}")
        return None, "", 0.0
    wait = INTERVAL - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    t0 = time.time()
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "ja"}, timeout=TIMEOUT)
        status, body = r.status_code, r.text
    except requests.RequestException as exc:
        print(f"  [error] {url}: {type(exc).__name__}: {str(exc)[:120]}")
        status, body = None, ""
    _last = time.time()
    _count += 1
    print(f"  GET {url} -> HTTP {status} ({len(body):,} 文字 / {time.time() - t0:.1f}秒)")
    return status, body, time.time() - t0


class Page(HTMLParser):
    """リンク・見出し・表の見出し・本文の文字数だけを拾う。"""

    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self.heads: List[Tuple[str, str]] = []
        self.ths: List[str] = []
        self.title = ""
        self._stack: List[str] = []
        self._buf: List[str] = []
        self._link_text: List[str] = []
        self._href: Optional[str] = None
        self.text_len = 0
        self.n_tr = 0
        self.n_table = 0

    def handle_starttag(self, tag, attrs):
        self._stack.append(tag)
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self._href = a["href"]
            self._link_text = []
        if tag in ("h1", "h2", "h3", "th", "title"):
            self._buf = []
        if tag == "tr":
            self.n_tr += 1
        if tag == "table":
            self.n_table += 1

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._link_text).strip()[:40]))
            self._href = None
        if tag in ("h1", "h2", "h3"):
            self.heads.append((tag, "".join(self._buf).strip()[:60]))
        if tag == "th":
            self.ths.append("".join(self._buf).strip()[:30])
        if tag == "title":
            self.title = "".join(self._buf).strip()[:100]
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

    def handle_data(self, data):
        if self._stack and self._stack[-1] in ("script", "style"):
            return
        self.text_len += len(data.strip())
        self._buf.append(data)
        if self._href is not None:
            self._link_text.append(data)


def parse(body: str) -> Page:
    p = Page()
    p.feed(body)
    return p


def pattern_of(href: str) -> str:
    """リンクの形（数字や長い英数の部分を <id> に潰す）。"""
    path = href.split("?")[0]
    path = re.sub(r"/\d+", "/<n>", path)
    path = re.sub(r"/[A-Za-z0-9_-]{8,}", "/<id>", path)
    return path


def summarize_links(p: Page, prefix: str):
    from collections import Counter
    pats = Counter(pattern_of(h) for h, _ in p.links if h.startswith(prefix) or h.startswith(BASE + prefix))
    return pats.most_common(12)


def main() -> int:
    print("=" * 78)
    print("M&A Online の企業データベースの下見（構造と規約だけ。中身は出さない。10秒に1回・最大5回）")
    print("=" * 78)

    # 1. robots.txt
    print("\n■ 1. robots.txt")
    status, body, _ = fetch(BASE + "/robots.txt")
    if status == 200:
        lines = [ln for ln in body.splitlines() if ln.strip()][:40]
        for ln in lines:
            print("   ", ln[:120])
        db_rules = [ln for ln in body.splitlines() if "/db" in ln]
        print(f"  /db に触れる行: {db_rules if db_rules else 'なし'}")
    else:
        print("  robots.txt が取れない")

    # 2. 一覧
    print("\n■ 2. /db/companies/ の構造")
    status, body, _ = fetch(INDEX)
    terms_url = None
    if status == 200 and body:
        p = parse(body)
        print(f"  title: {p.title}")
        print(f"  本文の文字数 {p.text_len:,} / リンク {len(p.links)} / 表 {p.n_table}（行 {p.n_tr}）")
        print(f"  見出し: {[t for _, t in p.heads][:12]}")
        print(f"  /db/ 配下のリンクの形: {summarize_links(p, '/db')}")
        pages = sorted(set(h for h, _ in p.links if re.search(r"[?&]page=\d+|/page/\d+", h)))
        print(f"  ページ送りのリンク: {len(pages)}本（例: {[pattern_of(x) for x in pages[:3]]}）")
        m = re.findall(r"([\d,]{1,9})\s*件", body)
        print(f"  「件」の表記: {m[:6]}")
        print(f"  ログイン / 会員 の語: {body.count('ログイン')} / {body.count('会員')}")
        for h, t in p.links:
            if "利用規約" in t or "terms" in h.lower() or "kiyaku" in h.lower():
                terms_url = urljoin(INDEX, h)
                break
        comp = [h for h, _ in p.links if re.match(r"(https?://maonline\.jp)?/db/companies/[^/?#]+/?$", h)]
        comp = [urljoin(INDEX, h) for h in comp]
        print(f"  企業ページらしいリンク: {len(comp)}本")
    else:
        comp = []
        print("  一覧が取れない（ここで止める）")

    # 3. 利用規約
    print("\n■ 3. 利用規約（複製・転載・自動取得・データベース化に触れる文）")
    if terms_url is None:
        terms_url = BASE + "/terms/"
        print(f"  一覧にリンクが無いので {terms_url} を試す")
    status, body, _ = fetch(terms_url)
    if status == 200 and body:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
        text = re.sub(r"<[^>]+>", "\n", text)
        sents = [s.strip() for s in re.split(r"[\n。]", text) if s.strip()]
        hit = [s for s in sents if any(k in s for k in KEYWORDS)]
        print(f"  文の数 {len(sents)} / 該当 {len(hit)}")
        for s in hit[:15]:
            print("   -", s[:160])
    else:
        print("  規約が取れない")

    # 4. 企業ページ 1 つ
    print("\n■ 4. 企業ページ 1 つの構造（項目名だけ）")
    if comp:
        status, body, _ = fetch(comp[0])
        if status == 200 and body:
            p = parse(body)
            print(f"  本文の文字数 {p.text_len:,} / リンク {len(p.links)} / 表 {p.n_table}（行 {p.n_tr}）")
            print(f"  見出しのタグ: {[h for h, _ in p.heads][:20]}")
            print(f"  表の見出し: {p.ths[:30]}")
            print(f"  /db/ 配下のリンクの形: {summarize_links(p, '/db')}")
            n_dates = len(re.findall(r"20\d\d[年/.-]\d{1,2}[月/.-]\d{1,2}", body))
            print(f"  日付らしい表記の数: {n_dates}")
            deals = [h for h, _ in p.links if re.search(r"/db/(deals?|ma|cases?|news)/", h)]
            print(f"  案件らしいリンク: {len(deals)}本（形: {sorted(set(pattern_of(h) for h in deals))[:5]}）")
            # 5. 案件ページ 1 つ
            if deals:
                print("\n■ 5. 案件ページ 1 つの構造（項目名だけ）")
                status, body, _ = fetch(urljoin(comp[0], deals[0]))
                if status == 200 and body:
                    q = parse(body)
                    print(f"  本文の文字数 {q.text_len:,} / 表 {q.n_table}（行 {q.n_tr}）/ 表の見出し: {q.ths[:30]}")
                    print(f"  見出しのタグ: {[h for h, _ in q.heads][:20]}")
    else:
        print("  企業ページのリンクが無い")
    print(f"\n取得回数 {_count} / 上限 {MAX_REQUESTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
