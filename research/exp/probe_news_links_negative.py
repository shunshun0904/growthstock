#!/usr/bin/env python3
"""
probe: リンクの確かめ（research/news_links.py）が、本物のサイトで無効なリンク・ウソのリンクを本当に弾くか。

単体テスト（tests/test_news_links.py）は偽の取得でしか確かめていない。ここでは本物のページで、毎晩と同じ判定
（news_links.check_source / check_market）に、わざと間違えた組み合わせを渡す:

  対照       見本の銘柄（7203。テストで使っている見本）の Yahoo!ファイナンスの4ページを、正しい見出しで → ok
  取り違え   その4ページを、ほかの3つの見出しで確かめる（例: ニュースのページを「掲示板」として）→ ng
  無効1      存在しないコード（0000）の4ページ → ng（HTTP 404、か題名にコードが出ない）
  無効2      存在しないページ（Yahoo!ファイナンスの銘柄の下の、無い道）→ ng
  別の銘柄1  見本の銘柄のページに、違う銘柄名（作った名前）→ ng
  別の銘柄2  見本の銘柄のページを、候補の別の銘柄のリンクとして（コード・名前ともその銘柄）→ ng
  別の銘柄3  候補の別の銘柄のページに、見本の銘柄の名前 → ng。正しい名前なら → ok
  形         英字入りのコード（候補にあれば）→ ok
  市場       市場全体の3ページを正しい見出しで → ok。互いの見出しで → ng

同じ URL は1回だけ開く（取り違えは、開いた結果を使い回して判定だけ変える）。開くのは Yahoo!ファイナンスと JPX の
20ページほど（news_links の Opener。名乗った User-Agent・間を空ける・自動では開かないホストは開かない）。
公開のログに出すのは、組み合わせの種類・状態・理由だけ（候補の銘柄のコード・名前は出さない）。

  $ python3 research/exp/probe_news_links_negative.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import news_links as N  # noqa: E402

SAMPLE_CODE, SAMPLE_NAME = "7203", "トヨタ自動車"
FAKE_NAME = "存在しない会社名"
YAHOO = ("yahoo_news", "yahoo_disclosure", "yahoo_perf", "yahoo_forum")
MISSING_URL = "https://finance.yahoo.co.jp/quote/{code}.T/no-such-page"


def src(sid: str) -> N.Source:
    return next(s for s in N.SOURCES if s.id == sid)


def mkt(mid: str) -> N.Market:
    return next(m for m in N.MARKET if m.id == mid)


def as_label(page: N.Source, label: N.Source) -> N.Source:
    """page の URL を、label の見出し（題名に要るコードの書き方と言葉）で確かめる。"""
    return replace(page, id=f"{page.id}->{label.id}", mark=label.mark, words=label.words)


def pinned(s: N.Source, code: str) -> N.Source:
    """URL を code の銘柄のページに固定する（確かめるときに別のコードを渡しても URL は変わらない）。"""
    return replace(s, url=s.url.format(code=code))


class MemoFetch:
    """同じ URL は1回だけ開く。一時的な失敗（接続できない・429・5xx）は覚えない（Opener のやり直しを効かせる）。"""

    def __init__(self, fetcher: Callable[[str], Tuple[int, str, str, bytes]] = N.fetch):
        self.fetcher = fetcher
        self.memo: Dict[str, Tuple[int, str, str, bytes]] = {}
        self.network = 0

    def __call__(self, url: str) -> Tuple[int, str, str, bytes]:
        if url in self.memo:
            return self.memo[url]
        res = self.fetcher(url)
        self.network += 1
        if not (res[0] == 0 or res[0] == 429 or res[0] >= 500):
            self.memo[url] = res
        return res


def others(pred: dict) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    """
    (候補の別の銘柄, 英字入りのコードの候補)。どちらも (コード, 名前)、無ければ None。
    別の銘柄は、毎晩の確かめの対象（規則を通った銘柄）のうち名前の分かる最初の1つ（見本の銘柄は除く）。
    """
    names = N.names_of(pred)
    rule = [c for c in sorted(N.rule_codes(pred)) if c != SAMPLE_CODE and names.get(c)]
    other = (rule[0], names[rule[0]]) if rule else None
    alnum = next(((c, n) for c, n in sorted(names.items()) if N.CODE_RE.match(c) and not c.isdigit() and n), None)
    return other, alnum


def cases(other: Optional[Tuple[str, str]], alnum: Optional[Tuple[str, str]]) -> List[Tuple[str, object, str, str, str]]:
    """(種類, 確かめるもの（Source か Market）, コード, 名前, 期待)。種類はログに出す（コード・名前は出さない）。"""
    out: List[Tuple[str, object, str, str, str]] = []
    for p in YAHOO:
        out.append((f"対照 {p}", src(p), SAMPLE_CODE, SAMPLE_NAME, N.OK))
    for p in YAHOO:
        for lab in YAHOO:
            if lab != p:
                out.append((f"取り違え {p} を {lab} として", as_label(src(p), src(lab)), SAMPLE_CODE, SAMPLE_NAME, N.NG))
    for p in YAHOO:
        out.append((f"無効1 存在しないコード {p}", src(p), "0000", "", N.NG))
    out.append(("無効2 存在しないページ", replace(src("yahoo_news"), id="missing", url=MISSING_URL), SAMPLE_CODE,
                SAMPLE_NAME, N.NG))
    for p in YAHOO:
        out.append((f"別の銘柄1 作った名前 {p}", src(p), SAMPLE_CODE, FAKE_NAME, N.NG))
    if other:
        for p in YAHOO:
            out.append((f"別の銘柄2 見本のページを候補のリンクとして {p}", pinned(src(p), SAMPLE_CODE), other[0], other[1],
                        N.NG))
        for p in YAHOO:
            out.append((f"別の銘柄3 候補のページに見本の名前 {p}", src(p), other[0], SAMPLE_NAME, N.NG))
            out.append((f"別の銘柄3 候補のページに正しい名前 {p}", src(p), other[0], other[1], N.OK))
    if alnum:
        for p in ("yahoo_news", "yahoo_forum"):
            out.append((f"形 英字入りのコード {p}", src(p), alnum[0], alnum[1], N.OK))
    ids = ("yahoo_newslist", "jpx_short", "jpx_investor")
    for m in ids:
        out.append((f"市場 対照 {m}", mkt(m), "", "", N.OK))
        for lab in ids:
            if lab != m:
                out.append((f"市場 取り違え {m} を {lab} として",
                            replace(mkt(m), id=f"{m}->{lab}", title_words=mkt(lab).title_words), "", "", N.NG))
    return out


def run(op: N.Opener, items) -> Tuple[List[str], int]:
    lines, bad = [], 0
    for label, s, code, name, want in items:
        rec = N.check_market(op, s) if isinstance(s, N.Market) else N.check_source(op, s, code, name)
        hit = rec["state"] == want
        bad += not hit
        lines.append(f"  {label}: {rec['state']}（期待 {want}）{'' if hit else ' ← 食い違い'}"
                     + (f"  理由: {rec['reason']}" if rec["reason"] else ""))
    return lines, bad


def main(fetcher: Callable[[str], Tuple[int, str, str, bytes]] = N.fetch, pause: float = N.PAUSE,
         pred_path: str = N.PRED_PATH) -> int:
    try:
        with open(pred_path, encoding="utf-8") as fh:
            pred = json.load(fh)
    except (OSError, ValueError):
        pred = {}
    other, alnum = others(pred)
    memo = MemoFetch(fetcher)
    op = N.Opener(fetcher=memo, pause=pause)
    items = cases(other, alnum)
    lines, bad = run(op, items)
    print("■ 本物のページで、確かめの判定が期待どおりか（わざと間違えた組み合わせを渡す）")
    print("\n".join(lines))
    if not other:
        print("  別の銘柄2・3: 予測ファイルに規則を通った別の銘柄が無いので確かめない")
    if not alnum:
        print("  形（英字入りのコード）: 予測ファイルの候補に無いので確かめない")
    want_ok = sum(1 for x in items if x[4] == N.OK)
    print(f"■ {len(items)}通り（ok のはず {want_ok}・ng のはず {len(items) - want_ok}）。食い違い {bad}件。"
          f"ネットで開いたページ {memo.network}件")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
