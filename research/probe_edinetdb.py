#!/usr/bin/env python3
"""
EDINET DB (edinetdb.jp) の API が実際に何を返すかを実測する。

なぜ実測するか
------------
J-Quants スタンダードでは BS/PL/CF の明細（販管費・研究開発費・
有利子負債・のれん・棚卸資産・設備投資）が取れない（`/fins/details` は
プレミアム限定。docs/DATA_FIELDS.md）。EDINET DB がそこを埋められる
可能性がある。ただし特徴量に使えるかどうかは**時点整合性**で決まる。

  1. 各値に「提出日（開示日）」が付いているか
  2. 訂正報告書で値が上書きされるか、原本のまま残るか
  3. 過去の期がどこまで遡れるか（学習データは 2018 年から）

開発者ページはこの作業環境から到達できない（egress ブロック）。
そこで、公開情報（公式 Qiita 記事 / developers ページからの引用）で
形が確定しているエンドポイントだけを GitHub Actions 上から叩き、
**返ってきた項目名と型を並べる**。項目名を推測して書かない。

叩くもの（すべて公開情報に出ている形）
  GET /v1/status                                     鍵不要
  GET /v1/companies?per_page=N                        X-API-Key
  GET /v1/companies/{EDINETコード}/financials?years=N
  GET /v1/rankings/roe?limit=N
  GET /v1/screener                                    引数なし

安全（リポジトリも Actions のログも公開である前提）
----------------------------------------------
- 鍵は X-API-Key ヘッダにだけ載せ、出力には絶対に出さない。出力に鍵の
  文字列が混ざったら [REDACTED] に置換してから印字する。
- メールアドレスらしき文字列は伏せる（登録に使ったものが返りうる）。
- フリープランは 100 リクエスト/日。1回の実行で使う上限を --max-requests
  で固定し（既定 30）、超えたら残りを飛ばす。バグで日の枠を使い切らない。

  $ EDINET_API_KEY=... python3 research/probe_edinetdb.py --years 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://edinetdb.jp/v1"
#: 公式サンプルに出ている EDINET コード（任天堂）。実在が確認できている唯一の値
SAMPLE_EDINET_CODE = "E02367"
#: 出力に出してよいレスポンスヘッダ（利用枠の確認用）。それ以外は印字しない
HEADER_ALLOW = re.compile(r"ratelimit|remaining|quota|limit|retry|plan", re.I)
#: 日付・期間らしい項目名。時点整合性の判断材料になるので目立たせる
DATE_LIKE = re.compile(r"date|filed|submit|period|fiscal|year|updated|created|"
                       r"doc|report", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EDINET_CODE = re.compile(r"^E\d{5}$")
#: 証券コード。4桁（英字入りも可: 154A）と、末尾に 0 を足した J-Quants 形式（77770 / 154A0）
SEC_CODE = re.compile(r"^[0-9][0-9A-Z]{3}0?$")
#: 有利子負債・のれんの有無を見るための項目名。任天堂には無いので別の会社で確かめる
DEBT_LIKE = re.compile(r"debt|borrow|bond|loan|goodwill|lease", re.I)


class Redactor:
    """鍵とメールアドレスを出力から消す。印字はすべてここを通す。"""

    def __init__(self, key: Optional[str]):
        self.key = key or ""

    def __call__(self, s: Any) -> str:
        t = str(s)
        if self.key and self.key in t:
            t = t.replace(self.key, "[REDACTED]")
        t = EMAIL.sub(lambda m: f"{m.group(0)[:3]}…@…", t)
        return t


class Probe:
    def __init__(self, key: Optional[str], budget: int, timeout: int = 30):
        self.key = key
        self.budget = budget
        self.used = 0
        self.timeout = timeout
        self.red = Redactor(key)
        self.last_headers: Dict[str, str] = {}

    def say(self, *parts: Any) -> None:
        print(" ".join(self.red(p) for p in parts), flush=True)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            auth: bool = True) -> Tuple[Optional[int], Any]:
        """(HTTPステータス, 本文)。予算切れ・接続失敗は (None, 理由)。"""
        if self.used >= self.budget:
            self.say(f"  [skip] リクエスト予算 {self.budget} を使い切ったので飛ばす: {path}")
            return None, "budget"
        url = BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        hdrs = {"Accept": "application/json", "User-Agent": "growthstock-probe/1"}
        if auth:
            if not self.key:
                return None, "no-key"
            hdrs["X-API-Key"] = self.key
        req = urllib.request.Request(url, headers=hdrs, method="GET")
        self.used += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                self.last_headers = {k: v for k, v in resp.headers.items()}
                status = resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            self.last_headers = {k: v for k, v in exc.headers.items()}
            status = exc.code
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.say(f"  [error] 接続できない: {type(exc).__name__}: {str(exc)[:160]}")
            return None, "unreachable"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        self.say(f"  GET {path}{'?' + urllib.parse.urlencode(params) if params else ''}"
                 f" -> HTTP {status} ({len(raw):,} bytes)")
        return status, body

    # ------------------------------------------------------------------ #
    # 表示
    # ------------------------------------------------------------------ #
    def show_headers(self) -> None:
        picked = {k: v for k, v in self.last_headers.items() if HEADER_ALLOW.search(k)}
        if picked:
            self.say("  利用枠に関するヘッダ:")
            for k, v in sorted(picked.items()):
                self.say(f"    {k}: {v}")
        else:
            self.say("  利用枠に関するヘッダは無し")

    def show_json(self, body: Any, limit: int = 1500) -> None:
        t = json.dumps(body, ensure_ascii=False, indent=2) if not isinstance(body, str) else body
        if len(t) > limit:
            t = t[:limit] + f"\n  … ({len(t):,} 文字中 {limit} 文字まで)"
        for line in t.splitlines():
            self.say("    " + line)

    def describe_rows(self, rows: List[Dict[str, Any]], title: str) -> List[str]:
        """
        行の集まりについて、項目ごとに 型 / 充足 / 例 を並べる。
        日付・期間らしい項目には印を付ける。返り値は日付らしい項目名。
        """
        if not rows:
            self.say(f"  {title}: 行なし")
            return []
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        self.say(f"  {title}: {len(rows)}行 / {len(keys)}項目")
        self.say(f"    {'項目':<34}{'型':<10}{'充足':>7}  例")
        date_like = []
        for k in keys:
            vals = [r.get(k) for r in rows]
            nonnull = [v for v in vals if v is not None and v != ""]
            types = sorted({type(v).__name__ for v in nonnull}) or ["null"]
            ex = ""
            if nonnull:
                v = nonnull[0]
                if isinstance(v, (dict, list)):
                    ex = f"<{type(v).__name__} len={len(v)}>"
                else:
                    ex = str(v)
                    if len(ex) > 48:
                        ex = ex[:48] + "…"
            mark = " ◀ 日付/期間?" if DATE_LIKE.search(k) else ""
            if mark:
                date_like.append(k)
            self.say(f"    {k:<34}{'/'.join(types):<10}{len(nonnull):>3}/{len(rows):<3}  {ex}{mark}")
        return date_like


# ---------------------------------------------------------------------- #
# レスポンスの形を読む小道具（項目名を推測しないための最低限）
# ---------------------------------------------------------------------- #

def as_rows(body: Any) -> List[Dict[str, Any]]:
    """`{"data": [...]}` / `[...]` / `{"data": {...}}` / `{...}` を行の列に揃える。"""
    if isinstance(body, dict):
        inner = body.get("data", body.get("items", body.get("results")))
        if isinstance(inner, list):
            return [r for r in inner if isinstance(r, dict)]
        if isinstance(inner, dict):
            return [inner]
        return [body]
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    return []


def find_codes(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    1行から (EDINETコード, 証券コードらしき値, 見つけた項目名) を拾う。
    項目名は決め打ちしない。値の形（E+5桁 / 4〜5桁の数字）で見る。
    """
    edinet, sec, where = None, None, []
    for k, v in row.items():
        s = str(v) if v is not None else ""
        if edinet is None and EDINET_CODE.match(s):
            edinet, where = s, where + [f"{k}={s}"]
        elif sec is None and SEC_CODE.match(s) and re.search(r"code|ticker|sec", k, re.I):
            sec, where = s, where + [f"{k}={s}"]
    return edinet, sec, where


def describe_key(key: Optional[str]) -> str:
    """鍵の形だけ（値は出さない）。空白だけの値も未設定と読む。"""
    v = (key or "").strip()
    if not v:
        return "未設定"
    shape = "16進のみ" if re.fullmatch(r"[0-9a-fA-F]+", v) else (
        "ハイフン区切り" if "-" in v else "英数混在")
    return f"設定あり（長さ {len(v)} 文字 / {shape}）"


def basic(p: "Probe", args) -> Optional[int]:
    """初回の発見用。返り値が None 以外なら main はそれで終了する。"""
    # ---- 0. 鍵不要の疎通 ----
    p.say("\n=== 0. /status（鍵不要）===")
    st, body = p.get("/status", auth=False)
    if st is None:
        p.say("[stop] edinetdb.jp に到達できない。ネットワークの問題")
        return 1
    p.show_json(body)
    p.show_headers()

    # ---- 1. 会社一覧: 項目名と、証券コードへの対応が取れるか ----
    p.say("\n=== 1. /companies?per_page=3 ===")
    st, body = p.get("/companies", {"per_page": 3})
    if st in (401, 403):
        p.say(f"[stop] 鍵が通らない（HTTP {st}）。登録した値を確認する")
        p.show_json(body, 600)
        return 2
    first_code: Optional[str] = None
    if st == 200:
        rows = as_rows(body)
        if isinstance(body, dict):
            meta = {k: v for k, v in body.items() if not isinstance(v, (list, dict))}
            if meta:
                p.say(f"  トップレベルの付随情報: {json.dumps(meta, ensure_ascii=False)}")
        p.describe_rows(rows, "会社")
        if rows:
            edinet, sec, where = find_codes(rows[0])
            p.say(f"  1行目から拾えたコード: EDINET={edinet} 証券={sec}  ({', '.join(where) or '該当なし'})")
            first_code = edinet
    else:
        p.show_json(body, 600)
    p.show_headers()

    # ---- 2. 財務: 項目名・型・充足。時点整合性の手がかりはここ ----
    p.say(f"\n=== 2. /companies/{args.edinet_code}/financials?years={args.years} ===")
    st, body = p.get(f"/companies/{args.edinet_code}/financials", {"years": args.years})
    if st == 200:
        rows = as_rows(body)
        date_like = p.describe_rows(rows, "財務（1行=1期）")
        if date_like:
            p.say("  日付/期間らしい項目の値（行ごと）:")
            for i, r in enumerate(rows):
                p.say(f"    行{i}: " + ", ".join(f"{k}={r.get(k)}" for k in date_like))
    else:
        p.show_json(body, 800)
    p.show_headers()

    # ---- 3. 遡及の上限 ----
    p.say(f"\n=== 3. 同じ会社で years={args.deep_years}（どこまで遡れるか）===")
    st, body = p.get(f"/companies/{args.edinet_code}/financials", {"years": args.deep_years})
    if st == 200:
        rows = as_rows(body)
        p.say(f"  返った期数: {len(rows)}")
        if rows:
            k0 = next((k for k in rows[0] if DATE_LIKE.search(k)), None)
            if k0:
                vals = [str(r.get(k0)) for r in rows]
                p.say(f"  {k0}: 最古 {min(vals)} / 最新 {max(vals)}")
    else:
        p.show_json(body, 400)

    # ---- 4. 一覧の1社目でも同じ形で引けるか ----
    if first_code and first_code != args.edinet_code:
        p.say(f"\n=== 4. 一覧の1社目 {first_code} の financials?years=2 ===")
        st, body = p.get(f"/companies/{first_code}/financials", {"years": 2})
        if st == 200:
            rows = as_rows(body)
            p.say(f"  返った期数: {len(rows)} / 項目数: {len(rows[0]) if rows else 0}")
        else:
            p.show_json(body, 400)

    # ---- 5. ランキング / スクリーナー ----
    p.say("\n=== 5. /rankings/roe?limit=3 ===")
    st, body = p.get("/rankings/roe", {"limit": 3})
    if st == 200:
        p.describe_rows(as_rows(body), "ランキング")
    else:
        p.show_json(body, 400)

    p.say("\n=== 6. /screener（引数なし。何を要求されるかを見る）===")
    st, body = p.get("/screener")
    if st == 200:
        rows = as_rows(body)
        p.say(f"  返った行数: {len(rows)}")
        if rows:
            p.describe_rows(rows[:3], "スクリーナー（先頭3行）")
    else:
        p.show_json(body, 600)
    p.show_headers()
    return None


def extra(p: "Probe", args) -> None:
    """
    2回目以降の追加確認（初回の結果を受けて）。

    - 有利子負債・のれんの項目があるか。初回の任天堂はどちらもほぼ無く、
      「項目が無い」のか「値が無いので省かれた」のか区別できなかった。
      借入とのれんのある大企業で見る。会社はランキングから拾い、
      EDINET コードを推測しない
    - 一括取得の可否。1社1リクエストでは全社の履歴に4か月かかる（月900）。
      screener の短縮引数（roe_gte=10 の形は 400 の文言で判明）と
      companies のページサイズを見る
    """
    codes: List[str] = [c for c in (args.codes or "").split(",") if c.strip()]
    if args.from_ranking:
        p.say(f"\n=== A. /rankings/{args.from_ranking}?limit={args.top}（会社を拾う）===")
        st, body = p.get(f"/rankings/{args.from_ranking}", {"limit": args.top})
        if st == 200:
            rows = as_rows(body)
            for r in rows:
                edinet, sec, _ = find_codes(r)
                p.say(f"  {edinet} 証券={sec} {r.get('name') or r.get('name_ja')}")
                if edinet:
                    codes.append(edinet)
        else:
            p.show_json(body, 400)

    union: Dict[str, int] = {}
    per_company: Dict[str, set] = {}
    for c in codes:
        p.say(f"\n=== B. /companies/{c}/financials?years=1 ===")
        st, body = p.get(f"/companies/{c}/financials", {"years": 1})
        if st != 200:
            p.show_json(body, 300)
            continue
        rows = as_rows(body)
        keys = set()
        for r in rows:
            keys |= set(r.keys())
        per_company[c] = keys
        for k in keys:
            union[k] = union.get(k, 0) + 1
        hits = sorted(k for k in keys if DEBT_LIKE.search(k))
        p.say(f"  項目数 {len(keys)} / 有利子負債・のれんらしい項目: "
              f"{', '.join(hits) if hits else 'なし'}")
        if hits and rows:
            for k in hits:
                p.say(f"    {k} = {rows[0].get(k)}")
    if per_company:
        base = set.intersection(*per_company.values())
        p.say(f"\n  全社に共通する項目 {len(base)} / 和集合 {len(union)}")
        only_some = sorted(k for k, n in union.items() if n < len(per_company))
        if only_some:
            p.say("  一部の会社にしか無い項目（値が無いと省かれる可能性）:")
            for k in only_some:
                p.say(f"    {k}  ({union[k]}/{len(per_company)}社)")

    if args.bulk_check:
        p.say("\n=== C. /companies?per_page=200（ページサイズの上限と総数）===")
        st, body = p.get("/companies", {"per_page": 200})
        if st == 200:
            rows = as_rows(body)
            p.say(f"  返った行数: {len(rows)}")
            if isinstance(body, dict):
                meta = {k: v for k, v in body.items() if not isinstance(v, (list, dict))}
                p.say(f"  付随情報: {json.dumps(meta, ensure_ascii=False)}")
                for k, v in body.items():
                    if isinstance(v, dict) and k != "data":
                        p.say(f"  {k}: {json.dumps(v, ensure_ascii=False)[:300]}")
        else:
            p.show_json(body, 400)
        p.show_headers()

        p.say("\n=== D. /screener?revenue_gte=0（一括で何が返るか）===")
        st, body = p.get("/screener", {"revenue_gte": 0})
        if st == 200:
            rows = as_rows(body)
            p.say(f"  返った行数: {len(rows)}")
            if isinstance(body, dict):
                meta = {k: v for k, v in body.items() if not isinstance(v, (list, dict))}
                p.say(f"  付随情報: {json.dumps(meta, ensure_ascii=False)}")
                for k, v in body.items():
                    if isinstance(v, dict) and k != "data":
                        p.say(f"  {k}: {json.dumps(v, ensure_ascii=False)[:300]}")
            if rows:
                p.describe_rows(rows[:3], "スクリーナー（先頭3行）")
        else:
            p.show_json(body, 600)
        p.show_headers()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EDINET DB API の形を実測する")
    ap.add_argument("--max-requests", type=int, default=30,
                    help="この実行で使ってよいリクエスト数の上限（フリープランは 100/日・900/月）")
    ap.add_argument("--edinet-code", default=SAMPLE_EDINET_CODE,
                    help=f"財務を引く EDINET コード。既定 {SAMPLE_EDINET_CODE}（公式サンプル）")
    ap.add_argument("--years", type=int, default=5, help="financials の years")
    ap.add_argument("--deep-years", type=int, default=30,
                    help="遡及の上限を見るために大きく指定する years")
    ap.add_argument("--skip-basic", action="store_true",
                    help="初回の発見用ステップ（7リクエスト）を飛ばす")
    ap.add_argument("--codes", default="",
                    help="追加で financials の項目を見る EDINET コード（カンマ区切り）")
    ap.add_argument("--from-ranking", default="",
                    help="このランキング（例 market-cap）の上位から会社を拾って項目を見る")
    ap.add_argument("--top", type=int, default=3, help="--from-ranking で拾う件数")
    ap.add_argument("--bulk-check", action="store_true",
                    help="一括取得の可否（companies のページサイズ / screener の短縮引数）を見る")
    args = ap.parse_args(argv)

    key = os.environ.get("EDINET_API_KEY", "").strip() or None
    p = Probe(key, args.max_requests)
    p.say(f"鍵: {describe_key(key)} / リクエスト上限 {args.max_requests}")
    if not key:
        p.say("[stop] EDINET_API_KEY が無い。Secrets に登録してから実行する")
        return 1

    if not args.skip_basic:
        rc = basic(p, args)
        if rc is not None:
            return rc
    if args.codes or args.from_ranking or args.bulk_check:
        extra(p, args)

    p.say(f"\n[done] この実行で使ったリクエスト: {p.used} / 上限 {args.max_requests}"
          f"（フリープランは 100/日・900/月）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
