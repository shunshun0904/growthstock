#!/usr/bin/env python3
"""
日本証券金融（日証金、taisyaku.jp）の公開データが、実際にどういう形で取れるかを実測する。

なぜ実測するか
------------
運用者の希望（2026-09-25）で、日証金の需給データ（銘柄別の融資・貸株残高、
品貸料率＝逆日歩、制限措置）を特徴量の候補にする。作業環境（Claude Code）からは
taisyaku.jp につなげない（egress の設定）ので、GitHub Actions から叩いて形を確かめる。

公式ページの記述（検索結果で確認、2026-09-25。ページそのものは作業環境から開けない）
  - 一括の CSV は直近の日だけ（銘柄一覧の形）。過去は銘柄ごとの検索から CSV（3年まで）
  - 速報は当日18時半過ぎ、確報は翌営業日の11時ごろ
  - 掲載内容は私的利用の範囲を超えて使えない（改変・複製・公開・第三者への提供の禁止）

何を見るか
---------
  1. robots.txt（自動の取得が許されている範囲。許されない道は叩かない）
  2. DATA ページ（/download/）に並ぶファイルの一覧（リンクと見出し）
  3. そのファイルの形: 文字コード・見出しの行・列の数・データの行数・日付の範囲
  4. 銘柄ごとのページ（--codes）で、CSV の取り方（リンク・フォームの項目名・
     スクリプトの中の API らしいパス）
  5. 銘柄ごとの CSV が取れれば、その行数と日付の範囲（どこまで遡れるか）

安全（リポジトリも Actions のログも公開）
--------------------------------------
- **データの値は出さない。** 出すのは URL・リンクの見出し・列の見出し・件数・
  日付の範囲だけ（利用条件で公開・第三者への提供が禁止されているため）。
  見出しの行として出す文字列も、日付以外の2桁以上の数字は # に伏せる
  （見出しの判定を誤ってデータの行が混ざっても値が出ないように）
- 取ったファイルは保存しない
- 1回の実行で叩く数を --max-requests で固定し、リクエストの間を空ける

  $ python3 research/probe_jsf.py --codes 7203 9984
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zipfile
from collections import Counter
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

BASE = "https://www.taisyaku.jp"
UA = "growthstock-research-probe/1.0 (+https://github.com/shunshun0904/growthstock)"
JST = dt.timezone(dt.timedelta(hours=9))

#: ファイルらしいリンク
FILE_EXT = re.compile(r"\.(csv|xlsx?|zip|txt|pdf)(\?|#|$)", re.I)
#: CSV の取り方に関わりそうなリンク・項目名
CSVISH = re.compile(r"csv|download|export|dl=|output", re.I)
#: スクリプトの中の API らしいパス
API_LIKE = re.compile(r"""["'](/[A-Za-z0-9_\-./]*(?:api|csv|json|download|export)[A-Za-z0-9_\-./?=&]*)["']""",
                      re.I)
#: 証券コード（4文字。英字入りも可: 154A）
CODE_FIELD = re.compile(r"^[0-9][0-9A-Z]{3}$")
#: 日付（2026/09/24・2026-09-24・2026年9月24日・20260924）
DATE_TOKEN = re.compile(r"(20\d{2})(?:[/\-.年](\d{1,2})[/\-.月](\d{1,2})|(\d{2})(\d{2}))(?!\d)")
#: 見出しの行を出すときに伏せる数字（2桁以上）
DIGITS = re.compile(r"\d{2,}")
#: 期間の指定に使いそうなフォームの項目名
PERIOD_LIKE = re.compile(r"date|from|to|start|end|term|period|ymd|year|month|day", re.I)


def now_jst() -> str:
    return dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def dates_in(s: str) -> List[str]:
    """文字列の中の日付を YYYY-MM-DD にして返す。"""
    out = []
    for y, m, d, m2, d2 in DATE_TOKEN.findall(s):
        mm, dd = (m, d) if m else (m2, d2)
        try:
            out.append(dt.date(int(y), int(mm), int(dd)).isoformat())
        except ValueError:
            continue
    return out


def sanitize(s: str, limit: int = 100) -> str:
    """見出しとして出す文字列。日付は残し、ほかの2桁以上の数字は # に伏せる。"""
    s = " ".join(str(s).split())
    keep = {}

    def hold(m: re.Match) -> str:
        k = f"\x00{len(keep)}\x00"
        keep[k] = m.group(0)
        return k

    t = DATE_TOKEN.sub(hold, s)
    t = DIGITS.sub("#", t)
    for k, v in keep.items():
        t = t.replace(k, v)
    return t[:limit]


def decode(b: bytes) -> Tuple[str, str]:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1"), "不明（latin-1 で読んだ）"


# ---------------------------------------------------------------------- #
# 取得
# ---------------------------------------------------------------------- #

class Fetcher:
    """叩く数に上限を置き、robots.txt で許されない道は叩かない。"""

    def __init__(self, max_requests: int, pause: float,
                 robots: Optional[urllib.robotparser.RobotFileParser] = None,
                 opener=None):
        self.max = max_requests
        self.pause = pause
        self.robots = robots
        self.used = 0
        self.opener = opener or urllib.request.build_opener()

    def allowed(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch(UA, url)

    def get(self, url: str) -> Tuple[Optional[int], Dict[str, str], bytes, str]:
        """(HTTP の状態, 応答ヘッダ, 本文, 最後の URL)。叩かなかったときは状態が None。"""
        if not self.allowed(url):
            print(f"  [robots] 許可されていないので叩かない: {url}")
            return None, {}, b"", url
        if self.used >= self.max:
            print(f"  [skip] この実行の上限 {self.max} に達した: {url}")
            return None, {}, b"", url
        if self.used:
            time.sleep(self.pause)
        self.used += 1
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ja"})
        try:
            with self.opener.open(req, timeout=30) as r:
                return r.status, dict(r.headers), r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), b"", url
        except Exception as e:  # noqa: BLE001 — 調査なので止めずに記録する
            print(f"  [error] {type(e).__name__}: {str(e)[:120]}")
            return None, {}, b"", url


def show_status(label: str, status, headers: Dict[str, str], body: bytes, final: str) -> None:
    keep = {k: v for k, v in headers.items()
            if k.lower() in ("content-type", "last-modified", "date", "content-disposition",
                             "cache-control", "etag")}
    print(f"  {label}: HTTP {status} / {len(body):,} bytes / 最後の URL {final}")
    for k, v in keep.items():
        print(f"      {k}: {sanitize(v, 120) if k.lower() == 'content-disposition' else v}")


# ---------------------------------------------------------------------- #
# HTML
# ---------------------------------------------------------------------- #

class PageParser(HTMLParser):
    """リンク・フォーム・スクリプトの中の API らしいパス・title を拾う。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self.forms: List[Dict] = []
        self.api_paths: List[str] = []
        self.scripts: List[str] = []
        self.title = ""
        self._a: Optional[str] = None
        self._text: List[str] = []
        self._form: Optional[Dict] = None
        self._in_script = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self._a = a["href"]
            self._text = []
        elif tag == "form":
            self._form = {"action": a.get("action") or "", "method": (a.get("method") or "get").lower(),
                          "fields": []}
            self.forms.append(self._form)
        elif tag in ("input", "select", "textarea", "button") and self._form is not None:
            self._form["fields"].append((tag, (a.get("type") or "").lower(), a.get("name") or ""))
        elif tag == "script":
            self._in_script = True
            if a.get("src"):
                self.scripts.append(a["src"])
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "a" and self._a is not None:
            self.links.append((self._a, " ".join("".join(self._text).split())[:60]))
            self._a = None
        elif tag == "form":
            self._form = None
        elif tag == "script":
            self._in_script = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._a is not None:
            self._text.append(data)
        if self._in_script:
            self.api_paths.extend(m.group(1) for m in API_LIKE.finditer(data))
        if self._in_title:
            self.title += data.strip()


def parse_html(body: bytes) -> PageParser:
    text, _ = decode(body)
    p = PageParser()
    p.feed(text)
    return p


def show_page(p: PageParser, base_url: str, only_files: bool = False, cap: int = 40) -> List[str]:
    """ページの中身の形を出し、ファイルらしいリンク（絶対 URL）を返す。"""
    print(f"      title: {sanitize(p.title, 80)}")
    files, other = [], []
    for href, text in p.links:
        url = urllib.parse.urljoin(base_url, href)
        if FILE_EXT.search(href) or CSVISH.search(href) or CSVISH.search(text):
            files.append((url, text))
        elif not only_files:
            other.append((url, text))
    print(f"      リンク {len(p.links)}本（ファイル・CSV らしいもの {len(files)}本）")
    for url, text in files[:cap]:
        print(f"        [file] {url}  「{sanitize(text, 50)}」")
    for f in p.forms[:6]:
        names = [f"{t}:{ty or '-'}:{n}" for t, ty, n in f["fields"] if n][:25]
        period = [n for _, _, n in f["fields"] if n and PERIOD_LIKE.search(n)]
        print(f"        [form] {f['method'].upper()} {urllib.parse.urljoin(base_url, f['action'])} "
              f"項目 {names}{' / 期間らしい項目 ' + str(period) if period else ''}")
    for s in sorted(set(p.scripts))[:10]:
        print(f"        [script] {urllib.parse.urljoin(base_url, s)}")
    for a in sorted(set(p.api_paths))[:15]:
        print(f"        [api?] {a}")
    return list(dict.fromkeys(u for u, _ in files))


# ---------------------------------------------------------------------- #
# 表（CSV・Excel）
# ---------------------------------------------------------------------- #

def _is_data_row(r: List[str]) -> bool:
    cells = [x.strip() for x in r]
    if any(CODE_FIELD.match(x) for x in cells[:3]):
        return True
    return bool(cells) and bool(dates_in(cells[0])) and len(cells) >= 3 and \
        sum(1 for x in cells[1:] if re.fullmatch(r"-?[\d,.]+", x or "")) >= 1


def describe_table(text: str) -> Dict:
    """CSV の形（値は含めない）。"""
    rows = [r for r in csv.reader(io.StringIO(text)) if any(x.strip() for x in r)]
    first = next((i for i, r in enumerate(rows) if _is_data_row(r)), None)
    pre = rows[:first] if first is not None else rows[:6]
    data = [r for r in rows[first:] if _is_data_row(r)] if first is not None else []
    code_rows = [r for r in data if any(CODE_FIELD.match(x.strip()) for x in r[:3])]
    codes = {next(x.strip() for x in r[:3] if CODE_FIELD.match(x.strip())) for r in code_rows}
    first_col_dates = sorted({d for r in data for d in dates_in(r[0])}) if data else []
    all_dates = sorted({d for r in rows for x in r for d in dates_in(x)})
    ncols = Counter(len(r) for r in data)
    width = max(ncols, default=0)
    fill = [sum(1 for r in data if len(r) > j and r[j].strip()) for j in range(width)]
    numeric = [sum(1 for r in data if len(r) > j and re.fullmatch(r"-?[\d,.]+", r[j].strip() or "x"))
               for j in range(width)]
    return {"rows": len(rows), "preamble": [[sanitize(x, 40) for x in r] for r in pre[:6]],
            "data_rows": len(data), "codes": len(codes), "ncols": dict(ncols),
            "first_col_dates": (first_col_dates[0], first_col_dates[-1], len(first_col_dates))
            if first_col_dates else None,
            "dates": (all_dates[0], all_dates[-1], len(all_dates)) if all_dates else None,
            "fill": fill, "numeric": numeric}


def show_table(d: Dict) -> None:
    print(f"      行 {d['rows']:,}（データらしい行 {d['data_rows']:,} / 銘柄コードの数 {d['codes']:,}）"
          f" / 列の数 {d['ncols']}")
    for r in d["preamble"]:
        print(f"        [見出し] {r}")
    if d["first_col_dates"]:
        a, b, n = d["first_col_dates"]
        print(f"      先頭の列の日付: {a} 〜 {b}（{n}日）")
    if d["dates"]:
        a, b, n = d["dates"]
        print(f"      出てくる日付の範囲: {a} 〜 {b}（{n}種類）")
    if d["data_rows"]:
        print(f"      列ごとの値あり {d['fill']}")
        print(f"      列ごとの数値     {d['numeric']}")


def describe_xlsx(b: bytes) -> Dict:
    """Excel の形（シート名と行の数）。値は読まない。"""
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
        names = re.findall(r'<sheet[^>]*name="([^"]+)"', wb)
        rows = {}
        for n in z.namelist():
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"):
                rows[n.rsplit("/", 1)[-1]] = z.read(n).count(b"<row ")
    return {"sheets": names, "rows": rows}


def show_file(f: Fetcher, url: str) -> None:
    status, headers, body, final = f.get(url)
    show_status("ファイル", status, headers, body, final)
    if status != 200 or not body:
        return
    low = final.lower()
    if low.endswith((".xlsx", ".xls")) or body[:2] == b"PK":
        try:
            d = describe_xlsx(body)
            print(f"      Excel: シート {[sanitize(s, 30) for s in d['sheets']]} / 行の数 {d['rows']}")
        except (zipfile.BadZipFile, KeyError) as e:
            print(f"      Excel として読めない: {type(e).__name__}")
        return
    if low.endswith(".pdf") or body[:4] == b"%PDF":
        print("      PDF（中身は読まない）")
        return
    text, enc = decode(body)
    if text.lstrip().startswith("<"):
        print(f"      HTML が返った（文字コード {enc}）")
        show_page(parse_html(body), final, only_files=True, cap=10)
        return
    print(f"      文字コード {enc}")
    show_table(describe_table(text))


# ---------------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------------- #

def load_robots(f: Fetcher) -> Optional[urllib.robotparser.RobotFileParser]:
    status, headers, body, final = f.get(f"{BASE}/robots.txt")
    show_status("robots.txt", status, headers, body, final)
    rp = urllib.robotparser.RobotFileParser()
    if status == 200 and body:
        text, _ = decode(body)
        for line in text.splitlines()[:40]:
            print(f"      | {line}")
        rp.parse(text.splitlines())
        return rp
    if status in (401, 403):
        print("      robots.txt が拒否された。すべて叩かない扱いにする")
        rp.parse(["User-agent: *", "Disallow: /"])
        return rp
    print("      robots.txt が無い（制限なしとして扱う）")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="日証金（taisyaku.jp）の公開データの形を実測する")
    ap.add_argument("--codes", nargs="*", default=["7203", "9984"],
                    help="銘柄ごとのページを見る証券コード（4桁）")
    ap.add_argument("--max-requests", type=int, default=25)
    ap.add_argument("--max-files", type=int, default=6, help="DATA ページから取るファイルの数")
    ap.add_argument("--pause", type=float, default=2.0, help="リクエストの間の秒数")
    ap.add_argument("--extra-paths", nargs="*", default=[],
                    help="追加で形を見るパス（/ から。例: /search/detail/pcsl/7203）")
    args = ap.parse_args(argv)

    print(f"[jsf] 実行時刻 {now_jst()} / 上限 {args.max_requests}回 / 間隔 {args.pause}秒")
    f = Fetcher(args.max_requests, args.pause)

    print("\n=== 1. robots.txt ===")
    f.robots = load_robots(f)
    if f.robots is not None and not f.allowed(f"{BASE}/download/"):
        print("[stop] DATA ページが robots.txt で許されていない。ここで止める")
        return 0

    print("\n=== 2. DATA ページ ===")
    status, headers, body, final = f.get(f"{BASE}/download/")
    show_status("DATA", status, headers, body, final)
    files: List[str] = []
    if status == 200 and body:
        files = show_page(parse_html(body), final, only_files=True)

    print(f"\n=== 3. ファイルの形（先頭 {args.max_files}本） ===")
    for url in [u for u in files if FILE_EXT.search(u)][:args.max_files]:
        print(f"  --- {url}")
        show_file(f, url)

    print("\n=== 4. 銘柄ごとのページ ===")
    paths = []
    for c in args.codes:
        paths += [f"/search/detail/pcsl/{c}", f"/app/stock/detail/{c}-01"]
    paths += args.extra_paths
    per_stock_files: List[str] = []
    for path in paths:
        url = urllib.parse.urljoin(BASE, path)
        print(f"  --- {url}")
        status, headers, body, final = f.get(url)
        show_status("ページ", status, headers, body, final)
        if status == 200 and body:
            got = show_page(parse_html(body), final)
            per_stock_files += [u for u in got if u not in per_stock_files]

    print("\n=== 5. 銘柄ごとの CSV（見つかったもの） ===")
    # ナビゲーションの「DATA」などは除き、CSV かファイルらしいものだけ
    csv_like = [u for u in per_stock_files
                if (re.search(r"csv", u, re.I) or FILE_EXT.search(u))
                and u.rstrip("/") != f"{BASE}/download"]
    if not csv_like:
        print("  CSV らしいリンクは無かった（スクリプトで作る形なら [api?] を見る）")
    for url in csv_like[:4]:
        print(f"  --- {url}")
        show_file(f, url)

    print(f"\n[done] 使ったリクエスト {f.used} / {args.max_requests}（{now_jst()}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
