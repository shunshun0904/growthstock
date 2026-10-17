#!/usr/bin/env python3
"""
probe: 候補銘柄のニュースのリンクを毎日自動で確かめてよいか（各サイトの利用規約・robots.txt の、自動アクセスに触れる文を全部）。

運用者の指示（2026-10-17）「取得対象は、候補銘柄のみで良いです…ウソのリンクや無効なリンクがないことのバリデーションに
時間をかけてください」。確かめる処理（候補ごとにページを開いて銘柄のページか見る）もサイトへの自動アクセスになるので、
先に規約を読む。probe_news_links（run 39017769927）は規約の文を1ページ8文までしか出していなかった。

§1 規約のページ … 各サイトのページから「規約」「ポリシー」「ご利用」「著作権」などのリンクをたどり（1サイト6本まで）、
   自動アクセス・負荷・転載・リンクに触れる文を全部出す（1ページ40文まで）。既に分かっている規約の URL も読む
§2 robots.txt … User-agent: * の塊を全部（Allow / Disallow / Crawl-delay）
§3 名乗った User-Agent … ブラウザを装わず、リンク確認のプログラムだと名乗って（HONEST_UA）同じページを開けるか

出すのは題名・URL・規約の文だけ。銘柄の値は使わない（見本の銘柄 7203 のページから規約のリンクを探すだけ）。

  $ python3 research/exp/probe_news_terms.py
"""
from __future__ import annotations

import os
import re
import sys
import time
import urllib.parse
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_news_links as L  # noqa: E402

#: 規約のリンクを探すページ（見本の銘柄のページと市場のページ）
START_PAGES = (
    "https://finance.yahoo.co.jp/quote/7203.T/news",
    "https://www.nikkei.com/nkd/company/news/?scode=7203",
    "https://kabuyoho.jp/reportAnalyst?bcode=7203",
    "https://www.jpx.co.jp/markets/statistics-equities/short-selling/index.html",
    "https://indexes.nikkei.co.jp/nkave/index/profile?idx=nk225vi",
)
#: 前回の probe で分かっている規約の URL（必ず読む）
KNOWN_TERMS = (
    "https://www.lycorp.co.jp/ja/company/terms/",
    "https://www.nikkei.com/info/policy.html",
    "https://kabuyoho.jp/sitepolicy",
    "https://www.jpx.co.jp/term-of-use/index.html",
)
LINK_WORDS = ("利用規約", "規約", "ポリシー", "ご利用", "著作権", "免責", "サイトについて", "ガイドライン")
AUTO_WORDS = re.compile(r"(スクレイピング|クローリング|クローラ|ロボット|自動的|自動化|自動で|機械的|プログラム|"
                        r"過度な|過大な|負荷|負担|大量|ボット|bot|ツール|転載|複製|再配信|二次利用|リンク)", re.I)
HOSTS = ("finance.yahoo.co.jp", "www.nikkei.com", "kabuyoho.jp", "www.jpx.co.jp", "indexes.nikkei.co.jp")
#: 確かめる処理で使う名乗り（ブラウザを装わない）
HONEST_UA = "Mozilla/5.0 (compatible; growthstock-linkcheck/1.0; +https://github.com/shunshun0904/growthstock)"


def policy_links(text: str, base: str, limit: int = 6) -> List[str]:
    """本文のリンクのうち、文字が規約・ポリシーらしいもの（LINK_WORDS）。PDF も含める。"""
    out: List[str] = []
    for href, label in re.findall(r"<a[^>]+href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", text, re.I | re.S):
        lab = re.sub(r"<[^>]+>", "", label).strip()
        if any(w in lab for w in LINK_WORDS):
            u = urllib.parse.urljoin(base, L.html.unescape(href))
            if u.startswith("http") and u not in out:
                out.append(u)
    return out[:limit]


def star_block(robots: str) -> List[str]:
    """User-agent: * の塊の行（コメントを除く）。"""
    lines, active, out = robots.splitlines(), False, []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        if key == "user-agent":
            active = (val.strip() == "*")
            continue
        if active:
            out.append(f"{key}: {val.strip()}")
    return out


def main() -> int:
    seen: Dict[str, bool] = {}
    queue: List[str] = list(KNOWN_TERMS)
    print("■ §1 規約のページ")
    for start in START_PAGES:
        st, final, ctype, body = L.fetch(start)
        if st != 200:
            print(f"  {start} → {st}（規約のリンクを探せない）")
            continue
        links = policy_links(L.decode(body, ctype), final)
        print(f"  {urllib.parse.urlparse(start).netloc}: 規約らしいリンク {len(links)}本: " + " / ".join(links))
        queue += [u for u in links if u not in queue]
        time.sleep(L.PAUSE)
    for u in queue:
        if u in seen:
            continue
        seen[u] = True
        st, final, ctype, body = L.fetch(u)
        if st != 200:
            print(f"\n  [{st}] {u}")
            continue
        if "pdf" in (ctype or "").lower() or u.lower().endswith(".pdf"):
            print(f"\n  [PDF] {u}（{len(body):,} バイト。PDF の本文は読まない）")
            continue
        text = L.decode(body, ctype)
        body_text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        body_text = L.html.unescape(re.sub(r"<[^>]+>", "\n", body_text))
        sents: List[str] = []
        for part in re.split(r"[。\n]", body_text):
            s = re.sub(r"\s+", " ", part).strip()
            if len(s) >= 6 and AUTO_WORDS.search(s) and s not in sents:
                sents.append(s[:220])
        print(f"\n  「{L.page_title(text)[:60]}」 {final}（当たる文 {len(sents)}）")
        for s in sents[:40]:
            print(f"     - {s}")
        time.sleep(L.PAUSE)

    print("\n■ §2 robots.txt（User-agent: * の塊の全部）")
    for host in HOSTS:
        st, _, ctype, body = L.fetch(f"https://{host}/robots.txt")
        if st != 200:
            print(f"  {host}: {st}")
            continue
        lines = star_block(L.decode(body, ctype))
        print(f"  {host}: {len(lines)}行" + ("" if not lines else "（" + " / ".join(lines[:30]) + (" …" if len(lines) > 30 else "") + "）"))
        time.sleep(L.PAUSE)

    print("\n■ §3 名乗った User-Agent で開けるか（ブラウザを装わない）")
    for u in START_PAGES:
        req = L.urllib.request.Request(u, headers={"User-Agent": HONEST_UA, "Accept-Language": "ja"})
        try:
            with L.urllib.request.urlopen(req, timeout=L.TIMEOUT) as r:
                st, final, ctype, body = r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read(2_000_000)
        except L.urllib.error.HTTPError as e:
            st, final, ctype, body = e.code, u, "", b""
        except Exception as e:  # noqa: BLE001
            st, final, ctype, body = 0, u, "", str(e)[:80].encode()
        title = L.page_title(L.decode(body, ctype)) if st == 200 else ""
        print(f"  {urllib.parse.urlparse(u).netloc}: {st} 題名にコード{'○' if '7203' in title else '×'}「{title[:50]}」")
        time.sleep(L.PAUSE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
