#!/usr/bin/env python3
"""
probe: 「候補銘柄のニュース」タブに置くリンクの形が実在するか、見出しを自動で集められそうな出どころの規則は何か。

運用者の依頼（2026-10-17）「候補銘柄に関してのニュースや市場のセンチメント情報を収集して、ダッシュボードに表示する
ことは可能ですか？リンクでも良いです」。手元の作業環境からは株探・Yahoo!ファイナンス・株予報・みんかぶへの接続が
通信の制限で拒否されるので、外に出られる Actions で確かめる。

§1 リンクの形 … 見本の銘柄（7203。テストで既に使っている見本）で各 URL を開き、HTTP の状態・最後の URL
   （銘柄コードは {code} に戻す）・ページの題名・本文に銘柄コードと銘柄名があるかを出す
§2 robots.txt … 各ホストの「User-agent: *」の Disallow / Allow のうち、使う道に当たる行
§3 利用規約 … 開いたページから「利用規約」「ご利用にあたって」などのリンクをたどり、自動での取得・転載・複製に
   触れる文を抜き出す（見つからなければ「見つからない」と出す。推測で補わない）
§4 自動収集の候補 … Google ニュースの RSS（見本の銘柄名の検索）と、TDnet の適時開示の日ごとの一覧の件数

出すのは状態・題名・件数・規約の文だけ。手元のデータ（J-Quants・日証金・EDINET DB）の値は使わない。

  $ python3 research/exp/probe_news_links.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

SAMPLE_CODE = "7203"
SAMPLE_NAME = "トヨタ自動車"
SAMPLE_HINT = "トヨタ"            # 本文にこの語があるか（社名の表記ゆれを避けて短く）
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
TIMEOUT = 20
PRED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "public", "data", "predictions.json")
PAUSE = 1.0                       # 同じホストを続けて叩かない

#: (区分, 名前, URL の形)。{code} は4桁のコード、{name} は銘柄名（URL エンコードする）
LINKS: Sequence[Tuple[str, str, str]] = (
    ("ニュース", "株探 ニュース", "https://kabutan.jp/stock/news?code={code}"),
    ("ニュース", "株探 基本情報", "https://kabutan.jp/stock/?code={code}"),
    ("ニュース", "Yahoo!ファイナンス ニュース", "https://finance.yahoo.co.jp/quote/{code}.T/news"),
    ("ニュース", "日経 会社情報", "https://www.nikkei.com/nkd/company/?scode={code}"),
    ("ニュース", "日経 会社情報 ニュース", "https://www.nikkei.com/nkd/company/news/?scode={code}"),
    ("ニュース", "みんかぶ ニュース", "https://minkabu.jp/stock/{code}/news"),
    ("ニュース", "Google ニュース 検索", "https://news.google.com/search?q={name}&hl=ja&gl=JP&ceid=JP:ja"),
    ("開示", "株探 開示（nmode=3）", "https://kabutan.jp/stock/news?code={code}&nmode=3"),
    ("開示", "IR BANK", "https://irbank.net/{code}"),
    ("開示", "IR BANK 適時開示", "https://irbank.net/{code}/ir"),
    ("開示", "Yahoo!ファイナンス 適時開示", "https://finance.yahoo.co.jp/quote/{code}.T/disclosure"),
    ("予想", "株予報 銘柄", "https://kabuyoho.jp/reportTop?bcode={code}"),
    ("予想", "株予報 アナリスト", "https://kabuyoho.jp/reportAnalyst?bcode={code}"),
    ("予想", "株予報 アナリスト（スマホ）", "https://kabuyoho.jp/sp/reportAnalyst?bcode={code}"),
    ("予想", "みんかぶ 銘柄", "https://minkabu.jp/stock/{code}"),
    ("予想", "みんかぶ アナリスト予想", "https://minkabu.jp/stock/{code}/analyst_consensus"),
    ("予想", "四季報オンライン", "https://shikiho.toyokeizai.net/stocks/{code}"),
    ("予想", "Yahoo!ファイナンス 業績", "https://finance.yahoo.co.jp/quote/{code}.T/performance"),
    ("予想", "日経 業績", "https://www.nikkei.com/nkd/company/kessan/?scode={code}"),
    ("声", "Yahoo!ファイナンス 掲示板", "https://finance.yahoo.co.jp/quote/{code}.T/forum"),
    ("声", "みんかぶ 売買予想", "https://minkabu.jp/stock/{code}/pick"),
    ("声", "X 検索", "https://x.com/search?q={name}&f=live"),
    # 市場全体（銘柄によらない）
    ("市場", "株探 市況ニュース", "https://kabutan.jp/news/marketnews/"),
    ("市場", "日経 マーケット", "https://www.nikkei.com/markets/"),
    ("市場", "Yahoo!ファイナンス ニュース", "https://finance.yahoo.co.jp/news"),
    ("市場", "日経平均VI", "https://indexes.nikkei.co.jp/nkave/index/profile?idx=nk225vi"),
    ("市場", "JPX 空売り集計", "https://www.jpx.co.jp/markets/statistics-equities/short-selling/index.html"),
    ("市場", "JPX 投資部門別売買状況", "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"),
)

#: 自動で集める候補（§4）。robots.txt も見る
RSS_URL = "https://news.google.com/rss/search?q={name}&hl=ja&gl=JP&ceid=JP:ja"
TDNET_MAIN = "https://www.release.tdnet.info/inbs/I_main_00.html"
TDNET_LIST = "https://www.release.tdnet.info/inbs/I_list_001_{ymd}.html"

TERMS_WORDS = ("利用規約", "ご利用にあたって", "ご利用上の注意", "サイトポリシー", "免責", "規約")
RULE_WORDS = re.compile(r"(スクレイピング|クローリング|クローラ|ロボット|自動的|機械的|プログラム|転載|複製|二次利用|"
                        r"再配布|再配信|リンク|私的使用|私的利用|営利)")


# --------------------------------------------------------------------------- #
# 部品（ネットに出ないので単体テストできる）
# --------------------------------------------------------------------------- #

def build_url(template: str, code: str, name: str) -> str:
    return template.format(code=code, name=urllib.parse.quote(name))


def mask(url: str, code: str, name: str) -> str:
    """URL の中の銘柄コード・銘柄名を {code} / {name} に戻す（形だけ出す）。"""
    out = url.replace(urllib.parse.quote(name), "{name}").replace(name, "{name}")
    return re.sub(rf"(?<![0-9]){re.escape(code)}(?![0-9])", "{code}", out)


def decode(body: bytes, content_type: str = "") -> str:
    """文字コードはヘッダ → meta → UTF-8 → cp932 → EUC-JP の順に試す。"""
    cands: List[str] = []
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if m:
        cands.append(m.group(1))
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w\-]+)", body[:4096], re.I)
    if m:
        cands.append(m.group(1).decode("ascii", "ignore"))
    cands += ["utf-8", "cp932", "euc_jp"]
    for enc in cands:
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def page_title(text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else ""


def terms_links(text: str, base: str, limit: int = 2) -> List[str]:
    """本文のリンクのうち、文字が規約らしいもの（TERMS_WORDS）の URL。同じホストのものを先に。"""
    out: List[str] = []
    for href, label in re.findall(r"<a[^>]+href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", text, re.I | re.S):
        lab = re.sub(r"<[^>]+>", "", label).strip()
        if any(w in lab for w in TERMS_WORDS):
            u = urllib.parse.urljoin(base, html.unescape(href))
            if u.startswith("http") and u not in out:
                out.append(u)
    host = urllib.parse.urlparse(base).netloc
    out.sort(key=lambda u: urllib.parse.urlparse(u).netloc != host)
    return out[:limit]


def rule_sentences(text: str, limit: int = 8, width: int = 170) -> List[str]:
    """本文（タグを除く）から、自動取得・転載・複製・リンクに触れる文を抜く。"""
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    body = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    sents: List[str] = []
    for part in re.split(r"[。\n]", body):
        s = re.sub(r"\s+", " ", part).strip()
        if len(s) >= 8 and RULE_WORDS.search(s) and s not in sents:
            sents.append(s[:width])
        if len(sents) >= limit:
            break
    return sents


def robots_lines(robots: str, paths: Sequence[str]) -> List[str]:
    """User-agent: * の塊の Allow / Disallow のうち、paths のどれかの頭に当たるもの（無ければ空）。"""
    lines, active, out = robots.splitlines(), False, []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key == "user-agent":
            active = (val == "*")
            continue
        if active and key in ("allow", "disallow") and val:
            pat = re.escape(val).replace(r"\*", ".*").replace(r"\$", "$")
            if any(re.match(pat, p) for p in paths):
                out.append(f"{key.capitalize()}: {val}")
    return out


def count_items(text: str) -> Tuple[int, List[str]]:
    """RSS の <item> の数と、<source> の出どころ（多い順 5つ）。"""
    items = re.findall(r"<item>(.*?)</item>", text, re.S)
    srcs: Dict[str, int] = {}
    for it in items:
        m = re.search(r"<source[^>]*>(.*?)</source>", it, re.S)
        if m:
            k = html.unescape(m.group(1)).strip()
            srcs[k] = srcs.get(k, 0) + 1
    top = [f"{k} {v}" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1])[:5]]
    return len(items), top


def alnum_sample(path: str = PRED_PATH) -> Optional[Tuple[str, str]]:
    """予測ファイルの候補から、英字入りの4桁コード（例: 4桁目が英字）を1つ（コード, 銘柄名）。無ければ None。"""
    try:
        with open(path, encoding="utf-8") as fh:
            cands = json.load(fh).get("candidates") or []
    except (OSError, ValueError):
        return None
    for c in cands:
        code = str(c.get("code") or "")
        if len(code) == 4 and not code.isdigit() and c.get("name"):
            return code, str(c["name"])
    return None


def last_weekdays(today: dt.date, n: int = 3) -> List[dt.date]:
    out, d = [], today
    while len(out) < n:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


# --------------------------------------------------------------------------- #
# ネット
# --------------------------------------------------------------------------- #

def fetch(url: str) -> Tuple[int, str, str, bytes]:
    """(状態, 最後の URL, Content-Type, 本文)。失敗は状態 0 で、本文に理由。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja,en;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read(3_000_000)
    except urllib.error.HTTPError as e:
        return e.code, e.geturl() or url, e.headers.get("Content-Type", "") if e.headers else "", b""
    except Exception as e:  # noqa: BLE001  タイムアウト・名前解決など
        return 0, url, "", str(e)[:120].encode()


def main() -> int:
    code, name = SAMPLE_CODE, SAMPLE_NAME
    print(f"見本の銘柄 {code}（{name}）。{dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    print("\n■ §1 リンクの形（状態 / 最後の URL / 本文にコード・社名があるか / 題名）")
    terms_queue: Dict[str, List[str]] = {}
    paths_by_host: Dict[str, List[str]] = {}
    for group, label, tpl in LINKS:
        url = build_url(tpl, code, name)
        st, final, ctype, body = fetch(url)
        text = decode(body, ctype) if st else body.decode("utf-8", "replace")
        has_code = code in text
        has_name = SAMPLE_HINT in text
        title = page_title(text) if st else text
        print(f"  [{group}] {label}: {st} {mask(final, code, name)}  コード{'○' if has_code else '×'} "
              f"社名{'○' if has_name else '×'}  「{title[:70]}」")
        host = urllib.parse.urlparse(url).netloc
        paths_by_host.setdefault(host, []).append(urllib.parse.urlparse(url).path or "/")
        if st == 200 and host not in terms_queue:
            terms_queue[host] = terms_links(text, final)
        time.sleep(PAUSE)

    # 英字入りのコード（2024年から割り当て）でも開くか。候補の銘柄なので、コード・社名・題名は出さず状態だけ
    alt = alnum_sample()
    if alt is None:
        print("\n■ §1b 英字入りのコード: 予測ファイルの候補に無いので確かめない")
    else:
        acode, aname = alt
        print("\n■ §1b 英字入りのコード（候補から1つ。コード・社名・題名は出さない）")
        for group, label, tpl in LINKS:
            if "{code}" not in tpl:
                continue
            st, final, ctype, body = fetch(build_url(tpl, acode, aname))
            text = decode(body, ctype) if st else ""
            moved = mask(final, acode, aname) != mask(build_url(tpl, acode, aname), acode, aname)
            print(f"  [{group}] {label}: {st}  コード{'○' if acode in text else '×'}"
                  f"{'  （別の URL に移った）' if moved else ''}")
            time.sleep(PAUSE)

    # §4 の分も robots を見る
    rss = build_url(RSS_URL, code, name)
    paths_by_host.setdefault("news.google.com", []).append(urllib.parse.urlparse(rss).path)
    paths_by_host.setdefault("www.release.tdnet.info", []).append("/inbs/I_main_00.html")

    print("\n■ §2 robots.txt（User-agent: * の行のうち、使う道に当たるもの）")
    for host, paths in paths_by_host.items():
        st, _, ctype, body = fetch(f"https://{host}/robots.txt")
        if st != 200:
            print(f"  {host}: robots.txt を読めない（{st}）")
            continue
        hit = robots_lines(decode(body, ctype), sorted(set(paths)))
        print(f"  {host}: 当たる行 {len(hit)}" + ("（" + " / ".join(hit[:6]) + "）" if hit else "（禁止の行は無い）"))
        time.sleep(PAUSE)

    print("\n■ §3 利用規約（たどれたページの、自動取得・転載・複製・リンクに触れる文）")
    for host, links in terms_queue.items():
        if not links:
            print(f"  {host}: 規約らしいリンクがページに見つからない")
            continue
        for u in links:
            st, final, ctype, body = fetch(u)
            if st != 200:
                print(f"  {host}: {u} → {st}")
                continue
            text = decode(body, ctype)
            sents = rule_sentences(text)
            print(f"  {host}: 「{page_title(text)[:50]}」 {final}")
            for s in sents or ["（該当する文が見つからない）"]:
                print(f"     - {s}")
            time.sleep(PAUSE)

    print("\n■ §4 自動収集の候補")
    st, final, ctype, body = fetch(rss)
    if st == 200:
        n, top = count_items(decode(body, ctype))
        print(f"  Google ニュース RSS（{mask(final, code, name)}）: {st}、記事 {n}件、出どころ " + ("、".join(top) or "-"))
    else:
        print(f"  Google ニュース RSS: {st}")
    st, final, ctype, body = fetch(TDNET_MAIN)
    print(f"  TDnet 適時開示 トップ: {st}「{page_title(decode(body, ctype))[:40] if st == 200 else ''}」")
    for d in last_weekdays(dt.date.today()):
        u = TDNET_LIST.format(ymd=d.strftime("%Y%m%d"))
        st, final, ctype, body = fetch(u)
        text = decode(body, ctype) if st == 200 else ""
        rows = len(re.findall(r"<tr", text, re.I))
        pdfs = len(re.findall(r"\.pdf", text, re.I))
        print(f"  TDnet の日ごとの一覧 {d}: {st}、行 {rows}、PDF へのリンク {pdfs}")
        time.sleep(PAUSE)
        if st == 200:
            if code in text:
                print("     見本の銘柄の開示がこの日の一覧（1ページ目）にある")
            terms = terms_links(text, final)
            for t in terms:
                st2, f2, ct2, b2 = fetch(t)
                if st2 == 200:
                    tx = decode(b2, ct2)
                    print(f"  TDnet の規約らしいページ「{page_title(tx)[:50]}」 {f2}")
                    for s in rule_sentences(tx) or ["（該当する文が見つからない）"]:
                        print(f"     - {s}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
