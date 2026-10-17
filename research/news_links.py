#!/usr/bin/env python3
"""
候補銘柄のニュースのリンクを、候補ごとに実際に開いて確かめる（運用者の指示 2026-10-17「取得対象は、候補銘柄のみで良いです。
その日の予測対象全銘柄ですと、時間もかかると思うので。それよりは、ウソのリンクや無効なリンクがないことのバリデーションに
時間をかけてください」）。

毎晩の予測（Predict Breakouts）の後に Check News Links（.github/workflows/news-links.yml）が続けて走り、
public/data/predictions.json に載っている日の**候補**（今日の戦略の規則 = 1.5σ の3本すべて95以上。live_track.RULE_MODELS /
AGREE_PCT。画面の strategy.js と同じ）について、リンクを1本ずつ開き、次をすべて満たしたものだけを「確かめた」とする。

  1. HTTP 200（転送の後）
  2. 最後の URL が、開いた URL と同じホスト・同じ道・同じクエリ（トップや検索のページに飛ばされていない）
  3. ページの題名に、そのサイトの書き方で銘柄コードがある（Yahoo!ファイナンス【7203】、日経 [7203]、株予報Pro (7203)）。
     無い銘柄や見つからないページの題名にはコードが出ないので、中身の無いページ（無効なリンク）を弾ける
  4. ページの題名に、リンクの見出しと同じ中身を表す言葉がある（ニュース・適時開示・業績・掲示板）。
     見出しと違うページへのリンク（ウソのリンク）を出さない
  5. ページの題名に、その候補の銘柄名（予測ファイルの名前。全角・半角と「(株)」「株式会社」の違いはそろえて比べる）がある。
     コードが合っていても別の会社のページへのリンクを出さない
  6. 確かめ方そのものの確かめ: 1〜5 を通ったページを、わざと間違えた見出し（同じ銘柄のほかのリンクの言葉）と、ほかの銘柄の
     コードで判定し直し、どれも通らない（追加の接続はしない）。通ってしまったら、題名で見分けられていないので、そのリンクも
     出さない。サイトが題名の書き方を変えて確かめが甘くなったときに、気づかないまま出し続けないため

本物のサイトに、わざと間違えた組み合わせ 48通り（見出しの取り違え・存在しないコード・存在しないページ・違う銘柄名・別の
銘柄のページ）を渡し、すべて期待どおりに弾き、正しい組み合わせはすべて通ることを確かめた（2026-10-17。
research/exp/probe_news_links_negative.py、run 39024430457）。

結果は public/data/news_links.json（銘柄コードごとに、リンクと結果・理由）。**URL の正本はこのファイル**（SOURCES）で、
画面（src/views/NewsView.jsx）は書き写さない。画面は、確かめて弾いたリンク（ng）を出さず、ほかは印を付けずに見出しの
塊ごとに並べる（運用者の指示 2026-10-17「確かめた、未確認などの表示はいらないです。バックでやってくれれば良いので」）。

自動では開かないサイト（NO_AUTO。各サイトの規約・robots.txt を全文で読んだ結果。research/exp/probe_news_terms.py、
run 39022372591）は、URL の形だけ確かめたリンク（unchecked）として置き、画面ではほかのリンクと同じ塊に印なしで並べる
（運用者の決定 2026-10-17「全部を印なしで出す」。それまでは「未確認」の印を付けて分けていた）:
  - 日経（www.nikkei.com・indexes.nikkei.co.jp）… 著作権のページが、スクレイピング・クローリング・ボットなど自動化された
    手段での取得を禁じている
  - 株予報Pro（kabuyoho.jp）… robots.txt が自動アクセスの間隔を 72000秒（20時間）と求めている（Crawl-delay）
  - 株探・みんかぶ … 自動の接続を断る（HTTP 405）
Google ニュースは robots.txt が全体を禁止し、URL の形も確かめられていないので置かない。

自動で開くのは Yahoo!ファイナンスだけ（robots.txt は対象のページを禁止していない。規約が禁じるのは BOT による不正な
操作と転載など）。名乗った User-Agent（ブラウザを装わない）で、同じホストは間を空けて1件ずつ。1晩に開くのは候補の数 ×
4本だけ。市場全体のリンク（Yahoo!ファイナンスの経済ニュース・JPX の空売り集計と投資部門別売買状況など）は、画面から
外した（運用者の指示 2026-10-17「市場全体のカードも不要です」）ので置かず、開かない。ログは公開なので、件数と理由の
内訳だけを出す（銘柄コード・銘柄名は出さない）。

  $ python3 research/news_links.py [--pred public/data/predictions.json] [--out public/data/news_links.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import live_track as LT  # noqa: E402

PRED_PATH = os.path.join(ROOT, "public", "data", "predictions.json")
OUT_PATH = os.path.join(ROOT, "public", "data", "news_links.json")

#: 確かめる側の名乗り（ブラウザを装わない）
UA = "Mozilla/5.0 (compatible; growthstock-linkcheck/1.0; +https://github.com/shunshun0904/growthstock)"
TIMEOUT = 20
PAUSE = 1.5            # 同じホストを続けて開くときの間（秒）
RETRY_WAIT = 5.0       # 一時的な失敗（接続できない・5xx・429）を1回だけやり直すまでの間（秒）
MAX_BYTES = 3_000_000
CODE_RE = re.compile(r"^[0-9A-Z]{4}$")

#: リンクの塊（画面の並び）
GROUPS: Sequence[Tuple[str, str]] = (
    ("news", "ニュース"),
    ("disclosure", "適時開示"),
    ("forecast", "アナリスト予想・業績"),
    ("voice", "個人投資家の声"),
)


#: 自動では開かないホストと理由（news_links.json にも残す）。ここにあるホストへは、このファイルも probe も開かない
NO_AUTO: Dict[str, str] = {
    "www.nikkei.com": "日経は、自動化された手段での取得を規約（著作権について）で禁じている",
    "indexes.nikkei.co.jp": "日経は、自動化された手段での取得を規約（著作権について）で禁じている",
    "kabuyoho.jp": "株予報Pro は、robots.txt で自動アクセスの間隔を 20時間（Crawl-delay 72000）と求めている",
    "kabutan.jp": "株探は、自動の接続を断る（HTTP 405）",
    "minkabu.jp": "みんかぶは、自動の接続を断る（HTTP 405）",
}


def host_of(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def auto_ok(url: str) -> bool:
    """自動で開いてよい URL か（NO_AUTO のホストでない）。"""
    return host_of(url) not in NO_AUTO


@dataclass(frozen=True)
class Source:
    """
    銘柄ごとのリンク。url の {code} は4桁のコード。mark は題名の中のコードの書き方、words は題名に要る言葉。
    ホストが NO_AUTO なら開かず unchecked（URL の形は 2026-10-17 に見本の銘柄で確かめたもの）。
    """
    id: str
    group: str
    site: str
    label: str
    url: str
    mark: str
    words: Tuple[str, ...]
    note: str


#: 題名の書き方は 2026-10-17 に見本の銘柄で実際に開いて確かめたもの（probe_news_links、run 39017769927）:
#:   Yahoo!ファイナンス「トヨタ自動車(株)【7203】：ニュース・最新情報 - Yahoo!ファイナンス」
#: 自動では開かないホスト（NO_AUTO）のリンクの URL の形は、同じ run で見本の銘柄のページを開いて（日経・株予報Pro。規約を
#: 読む前）、株探・みんかぶは検索結果に実在するページ（株探 stock/news?code=5185・6594・285A、みんかぶ
#: stock/285A/analyst_consensus・stock/285A/pick）で確かめた。候補の銘柄ごとには開いていない
SOURCES: Sequence[Source] = (
    Source("yahoo_news", "news", "Yahoo!ファイナンス", "ニュース",
           "https://finance.yahoo.co.jp/quote/{code}.T/news", "【{code}】", ("ニュース",),
           "複数の配信元のニュースの一覧"),
    Source("kabutan_news", "news", "株探", "ニュース",
           "https://kabutan.jp/stock/news?code={code}", "", (),
           "材料ニュース・決算速報・適時開示の一覧。一部の記事は有料（株探プレミアム）"),
    Source("nikkei_news", "news", "日経", "ニュース",
           "https://www.nikkei.com/nkd/company/news/?scode={code}", "", (),
           "日本経済新聞の記事の一覧。有料会員限定の記事がある"),
    Source("yahoo_disclosure", "disclosure", "Yahoo!ファイナンス", "適時開示",
           "https://finance.yahoo.co.jp/quote/{code}.T/disclosure", "【{code}】", ("適時開示",),
           "会社が出した適時開示（決算短信・業績予想の修正など）の一覧"),
    Source("yahoo_perf", "forecast", "Yahoo!ファイナンス", "業績・予想",
           "https://finance.yahoo.co.jp/quote/{code}.T/performance", "【{code}】", ("業績",),
           "業績の推移と会社予想・アナリスト予想"),
    Source("kabuyoho_analyst", "forecast", "株予報Pro", "コンセンサス",
           "https://kabuyoho.jp/reportAnalyst?bcode={code}", "", (),
           "アナリスト予想の平均（コンセンサス）と、その変化"),
    Source("minkabu_analyst", "forecast", "みんかぶ", "アナリスト予想",
           "https://minkabu.jp/stock/{code}/analyst_consensus", "", (),
           "アナリストの目標株価とレーティングの内訳"),
    Source("nikkei_kessan", "forecast", "日経", "決算・財務",
           "https://www.nikkei.com/nkd/company/kessan/?scode={code}", "", (),
           "決算発表と業績・財務の情報"),
    Source("yahoo_forum", "voice", "Yahoo!ファイナンス", "掲示板",
           "https://finance.yahoo.co.jp/quote/{code}.T/forum", "【{code}】", ("掲示板",),
           "個人投資家の書き込み。真偽の確かめられない書き込みも多い"),
    Source("minkabu_pick", "voice", "みんかぶ", "個人の売買予想",
           "https://minkabu.jp/stock/{code}/pick", "", (),
           "個人投資家の買い・売りの予想と予想株価"),
)

#: 置いていないサイト（news_links.json に理由を残す）
EXCLUDED: Sequence[Dict[str, str]] = (
    {"site": "Google ニュース", "reason": "robots.txt が自動のアクセスを全体で禁止し、URL の形も確かめられていない"},
    {"site": "X", "reason": "検索はログインが要り、robots.txt も自動のアクセスを全体で禁止している"},
)


# --------------------------------------------------------------------------- #
# 候補（今日の戦略の規則を通った銘柄）
# --------------------------------------------------------------------------- #

def rule_codes(pred: dict, models: Sequence[str] = LT.RULE_MODELS, line: float = LT.AGREE_PCT) -> Dict[str, List[str]]:
    """
    予測ファイルの日ごとの候補のうち、規則（models の百分位がすべて line 以上）を通った銘柄 → 通った日の一覧。
    画面の strategy.js（passesAgree / minPct）と同じ: どれか1本でも百分位が無ければ通さない。
    """
    days = set(pred.get("dates") or [])
    out: Dict[str, List[str]] = {}
    for c in pred.get("candidates") or []:
        if days and c.get("date") not in days:
            continue
        per = c.get("byModel") or {}
        pcts = [(per.get(a) or {}).get("pctHistorical") for a in models]
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in pcts):
            continue
        if min(pcts) < line:
            continue
        code = str(c.get("code") or "").strip().upper()
        if not CODE_RE.match(code):
            continue
        out.setdefault(code, [])
        if c.get("date") and c["date"] not in out[code]:
            out[code].append(c["date"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def names_of(pred: dict) -> Dict[str, str]:
    """銘柄コード（4桁・大文字）→ 予測ファイルの銘柄名（最初に出てきたもの）。"""
    out: Dict[str, str] = {}
    for c in pred.get("candidates") or []:
        code = str(c.get("code") or "").strip().upper()
        if code and c.get("name") and code not in out:
            out[code] = str(c["name"])
    return out


# --------------------------------------------------------------------------- #
# 1本のページの判定（ネットに出ないので単体テストできる）
# --------------------------------------------------------------------------- #

def decode(body: bytes, content_type: str = "") -> str:
    cands: List[str] = []
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if m:
        cands.append(m.group(1))
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w\-]+)", body[:4096], re.I)
    if m:
        cands.append(m.group(1).decode("ascii", "ignore"))
    for enc in cands + ["utf-8", "cp932", "euc_jp"]:
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def page_title(text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip() if m else ""


def same_page(asked: str, final: str) -> bool:
    """最後の URL が開いた URL と同じページか（ホスト・道・クエリの値。末尾の / と http/https は問わない）。"""
    a, f = urllib.parse.urlsplit(asked), urllib.parse.urlsplit(final)
    if a.netloc.lower() != f.netloc.lower():
        return False
    if a.path.rstrip("/") != f.path.rstrip("/"):
        return False
    qa, qf = urllib.parse.parse_qs(a.query), urllib.parse.parse_qs(f.query)
    return all(qf.get(k) == v for k, v in qa.items())


#: 銘柄名を比べる前に外す、会社の形の書き方（NFKC でそろえた後の形）
LEGAL_FORMS = ("株式会社", "(株)", "㈱")


def norm_name(s: str) -> str:
    """銘柄名・題名をそろえる: NFKC（全角英数・全角括弧を半角に）、会社の形の書き方と空白を外す。"""
    t = unicodedata.normalize("NFKC", s or "")
    for f in LEGAL_FORMS:
        t = t.replace(f, "")
    return re.sub(r"\s+", "", t)


def name_in_title(name: str, title: str) -> bool:
    """
    題名に銘柄名があるか。名前が長い（8文字以上）ときは頭の8文字で比べる（サイトが長い名前を縮めて出すことがある）。
    名前が無ければ比べない（True）。
    """
    n, t = norm_name(name), norm_name(title)
    if not n:
        return True
    return n in t or (len(n) >= 8 and n[:8] in t)


def judge(asked: str, status: int, final: str, title: str, *, mark: str = "", title_words: Sequence[str] = (),
          name: str = "") -> Tuple[bool, str]:
    """(開けたか, 理由)。理由は画面とログに出す短い文。"""
    if status == 0:
        return False, "接続できない"
    if status != 200:
        return False, f"HTTP {status}"
    if not same_page(asked, final):
        return False, "別のページに移った"
    if mark and mark not in title:
        return False, "題名に銘柄コードが無い"
    for w in title_words:
        if w not in title:
            return False, f"題名に「{w}」が無い"
    if name and not name_in_title(name, title):
        return False, "題名に銘柄名が無い"
    return True, ""


# --------------------------------------------------------------------------- #
# ネット
# --------------------------------------------------------------------------- #

Fetcher = Callable[[str], Tuple[int, str, str, bytes]]


def fetch(url: str) -> Tuple[int, str, str, bytes]:
    """(状態, 最後の URL, Content-Type, 本文)。接続できなければ状態 0。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read(MAX_BYTES)
    except urllib.error.HTTPError as e:
        return e.code, e.geturl() or url, "", b""
    except Exception:  # noqa: BLE001  タイムアウト・名前解決・TLS など
        return 0, url, "", b""


class Opener:
    """同じホストを続けて開くときに間を空け、一時的な失敗は1回だけやり直す。"""

    def __init__(self, fetcher: Fetcher = fetch, pause: float = PAUSE, retry_wait: float = RETRY_WAIT,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.fetcher, self.pause, self.retry_wait = fetcher, pause, retry_wait
        self.sleep, self.clock = sleep, clock
        self.last: Dict[str, float] = {}
        self.requests = 0

    def _one(self, url: str) -> Tuple[int, str, str, bytes]:
        # 安全網: 自動では開かないホスト（NO_AUTO）へは、呼び出し側の誤りでも接続しない
        if not auto_ok(url):
            raise ValueError(f"自動では開かないホスト: {host_of(url)}")
        host = urllib.parse.urlsplit(url).netloc
        wait = self.pause - (self.clock() - self.last.get(host, -1e9))
        if wait > 0:
            self.sleep(wait)
        res = self.fetcher(url)
        self.last[host] = self.clock()
        self.requests += 1
        return res

    def open(self, url: str) -> Tuple[int, str, str, bytes]:
        res = self._one(url)
        if res[0] == 0 or res[0] == 429 or res[0] >= 500:
            self.sleep(self.retry_wait)
            res = self._one(url)
        return res


#: リンクの状態。ok = 開いて確かめた / ng = 開いたが条件を満たさない（画面に出さない） / unchecked = 自動では開かない
#: （URL の形だけ確かめた。画面には ok と同じく印なしで出す）
OK, NG, UNCHECKED = "ok", "ng", "unchecked"

#: 開いたページ: (状態, 最後の URL, 題名)
Page = Tuple[int, str, str]


def open_page(op: Opener, url: str) -> Page:
    st, final, ctype, raw = op.open(url)
    text = decode(raw, ctype) if st == 200 else ""
    return st, final, page_title(text)


def check_source(op: Opener, src: Source, code: str, name: str = "", page: Optional[Page] = None) -> dict:
    """page は開いた結果（無ければここで開く）。"""
    url = src.url.format(code=urllib.parse.quote(code))
    rec = {"id": src.id, "group": src.group, "site": src.site, "label": src.label, "href": url, "note": src.note}
    if not auto_ok(url):
        return {**rec, "state": UNCHECKED, "reason": NO_AUTO[host_of(url)]}
    st, final, title = page if page is not None else open_page(op, url)
    ok, reason = judge(url, st, final, title, mark=src.mark.format(code=code), title_words=src.words, name=name)
    return {**rec, "state": OK if ok else NG, "reason": reason}


# --------------------------------------------------------------------------- #
# 確かめ方そのものの確かめ（毎晩。開いたページを判定し直すだけで、追加の接続はしない）
# --------------------------------------------------------------------------- #

#: ほかに候補が無いときの、わざと間違えた銘柄のコード（存在しないコード）
DECOY_CODE = "0000"
#: 判定し直して通ってしまったリンクの理由（画面とログに出す）
LIE_REASON = "確かめ方で見分けられない（ほかの見出し・ほかの銘柄でも通った）"


def source_decoys(src: Source, code: str, decoy: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    src のページを判定し直す、わざと間違えた見出し: (題名に要るコードの書き方, 題名に要る言葉) の組。
    同じ銘柄のほかの自動で確かめるリンクの見出しと、ほかの銘柄（decoy）のコード。どれも通ってはいけない。
    """
    q = urllib.parse.quote(code)
    out = [(o.mark.format(code=code), o.words) for o in SOURCES
           if o.id != src.id and o.words and auto_ok(o.url.format(code=q))]
    if src.mark:
        out.append((src.mark.format(code=decoy), src.words))
    return out


def build(pred: dict, op: Opener, now: Optional[dt.datetime] = None) -> dict:
    """予測ファイルから候補を選び、全部のリンクを確かめた結果（news_links.json の中身）を返す。"""
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    codes = rule_codes(pred)
    names = names_of(pred)
    tried = passed = 0

    def recheck(rec: dict, hits: List[bool]) -> dict:
        # 1〜5 を通ったページを、わざと間違えた見出しで判定し直した結果。1つでも通ったら見分けられていないので出さない
        nonlocal tried, passed
        tried, passed = tried + len(hits), passed + sum(hits)
        return {**rec, "state": NG, "reason": LIE_REASON} if any(hits) else rec

    stocks = {}
    for code, dates in codes.items():
        name = names.get(code, "")
        decoy = next((c for c in codes if c != code), DECOY_CODE)
        links = []
        for s in SOURCES:
            url = s.url.format(code=urllib.parse.quote(code))
            page = open_page(op, url) if auto_ok(url) else None
            rec = check_source(op, s, code, name, page=page)
            if rec["state"] == OK:
                st, final, title = page
                rec = recheck(rec, [judge(url, st, final, title, mark=mk, title_words=w, name=name)[0]
                                    for mk, w in source_decoys(s, code, decoy)])
            links.append(rec)
        stocks[code] = {"dates": dates, "links": links}
    links = [l for v in stocks.values() for l in v["links"]]

    def count(rows: List[dict], state: str) -> int:
        return sum(r["state"] == state for r in rows)

    return {
        "checkedAt": stamp,
        "asOf": pred.get("asOf"),
        "rule": {"models": list(LT.RULE_MODELS), "agreePct": LT.AGREE_PCT},
        "groups": [{"id": g, "label": lab} for g, lab in GROUPS],
        "noAuto": [{"host": h, "reason": r} for h, r in NO_AUTO.items()],
        "excluded": list(EXCLUDED),
        "stocks": stocks,
        "summary": {"stocks": len(stocks), "links": len(links), "ok": count(links, OK), "ng": count(links, NG),
                    "unchecked": count(links, UNCHECKED), "requests": op.requests,
                    "decoyTried": tried, "decoyPassed": passed},
    }


def report(res: dict) -> List[str]:
    """ログの行（件数と理由の内訳だけ。銘柄コード・銘柄名は出さない）。"""
    s = res["summary"]
    links = [l for v in res["stocks"].values() for l in v["links"]]
    lines = [f"[news-links] 候補 {s['stocks']}銘柄（予測ファイルの {res.get('asOf')} までの日）、リンク {s['links']}本: "
             f"確かめて開けた {s['ok']} / 開けなかった {s['ng']} / 自動では開かないサイト {s['unchecked']}。"
             f"開いたページ {s['requests']}件",
             f"[news-links] 確かめ方の確かめ: 開けたページを、わざと間違えた見出し・ほかの銘柄のコードで "
             f"{s['decoyTried']}通り判定し直し、通ってしまったのは {s['decoyPassed']}通り（通ったページのリンクは出さない）"]
    bad = Counter((l["id"], l["reason"]) for l in links if l["state"] == NG)
    for (sid, reason), n in sorted(bad.items()):
        lines.append(f"[news-links]   弾いた: {sid} {reason} ×{n}")
    return lines


def main(argv: Optional[Sequence[str]] = None, fetcher: Optional[Fetcher] = None, pause: float = PAUSE) -> int:
    """fetcher / pause はテスト用（既定は fetch と PAUSE）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=PRED_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)
    try:
        with open(args.pred, encoding="utf-8") as fh:
            pred = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"[news-links] 予測ファイルを読めない: {type(e).__name__}")
        return 1
    res = build(pred, Opener(fetcher or fetch, pause=pause))
    for line in report(res):
        print(line)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
