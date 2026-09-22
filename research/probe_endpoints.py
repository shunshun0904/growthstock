#!/usr/bin/env python3
"""
J-Quants API のエンドポイント一覧を**発見してから**、全部叩いて可否を測る。

なぜ要るか
--------
`research/probe_fins_fields.py` の疎通表は、こちらが思いついた名前を
並べて叩いているだけだった。「無い」と出ても、名前が違うのか機能が無いのか
区別がつかない。実際、運用者の問い（適時開示・アナリスト予想・株主構成）に
対して10本叩いて全部 `The requested endpoint does not exist` になったが、
それは何の証明にもならなかった。

Claude の作業環境からは公式ドキュメント（jpx-jquants.com / jpx.gitbook.io）に
到達できない（egress ブロック）。**GitHub Actions のランナーからは到達できる**
ので、一覧の取得もここでやる。

やること
------
1. 公式ドキュメントと仕様ファイルの候補を取りに行き、`/v1/...` `/v2/...`
   らしき文字列を拾って**エンドポイント名の候補**にする
2. 1 で見つけたもの ＋ 手書きの候補 を、実際の鍵で叩く
3. 応答を3つに分ける
     OK           使える
     PLAN         在るが契約が足りない（プレミアムで開く）
     NOT_FOUND    そのパスには何も無い
4. docs/DATA_FIELDS.md に書き戻す

秘密の扱い
--------
**リポジトリは公開で、Actions のログも公開される。** 鍵は印字しない。
出力はすべて Redactor を通す。ヘッダもエラー本文も例外ではない。
取ってきた HTML は**中身を印字しない**（パスだけ抜く）。

  python3 research/probe_endpoints.py [--no-doc] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

API_BASE = "https://api.jquants.com/v2"
UA = "growthstock-probe/1.0"
TIMEOUT = 30
PAUSE = 0.25

#: 一覧が載っていそうな場所。到達できなければ黙って飛ばす
DOC_SOURCES = [
    "https://api.jquants.com/v2/openapi.json",
    "https://api.jquants.com/v2/swagger.json",
    "https://api.jquants.com/openapi.json",
    "https://jpx-jquants.com/",
    "https://jpx.gitbook.io/j-quants-ja",
    "https://jpx.gitbook.io/j-quants-ja/api-reference",
    "https://jpx.gitbook.io/j-quants-en/api-reference",
    "https://jpx.gitbook.io/sitemap.xml",
    "https://jpx.gitbook.io/j-quants-ja/sitemap.xml",
]

#: 手で思いついたぶん。発見できた一覧と合わせて叩く
HAND = [
    "/fins/summary", "/fins/details", "/fins/dividend", "/fins/announcement",
    "/fins/forecast", "/fins/consensus", "/fins/disclosure",
    "/equities/master", "/equities/bars/daily", "/equities/shareholders",
    "/equities/ownership",
    "/markets/margin-interest", "/markets/short-selling", "/markets/breakdown",
    "/markets/trades-spec", "/markets/ownership",
    "/indices/topix", "/indices/prices",
]

#: パスらしき文字列。/v1/ /v2/ で始まるものと、api-reference 配下の見出し
PATH_RE = re.compile(r"/v[12]/[a-z0-9][a-z0-9_\-/]{2,60}")
REF_RE = re.compile(r"api-reference/([a-z0-9][a-z0-9_\-/]{2,60})")

#: 叩くときに順に試すパラメータの形。400 が返ったら次を試す
PARAM_SHAPES: List[Tuple[str, dict]] = [
    ("なし", {}),
    ("date", {"date": "2024-05-15"}),
    ("code", {"code": "72030"}),
    ("from/to", {"from": "2024-05-01", "to": "2024-05-15"}),
]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


class Redactor:
    """鍵とメールアドレスを出力から消す。印字はすべてここを通す。"""

    def __init__(self, key: Optional[str]):
        self.key = key or ""

    def __call__(self, s: Any) -> str:
        t = str(s)
        if self.key and len(self.key) >= 8 and self.key in t:
            t = t.replace(self.key, "[REDACTED]")
        return EMAIL.sub(lambda m: f"{m.group(0)[:3]}…@…", t)


def fetch(url: str, headers: Optional[dict] = None,
          timeout: int = TIMEOUT) -> Tuple[int, str]:
    """(HTTPステータス, 本文) を返す。落ちたら (0, 理由)。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:                                    # noqa: BLE001
            body = ""
        return exc.code, body
    except Exception as exc:                                 # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def discover(red: Redactor) -> Tuple[List[str], List[dict]]:
    """
    公式ドキュメントからパスらしき文字列を拾う。

    **中身は印字しない。** 拾えたパスと、取得できたかどうかだけ出す。
    """
    found: set[str] = set()
    log: List[dict] = []
    for url in DOC_SOURCES:
        status, body = fetch(url)
        hits: set[str] = set()
        if status == 200 and body:
            hits |= {m.rstrip("/.") for m in PATH_RE.findall(body)}
            # OpenAPI なら paths キーが正解
            if body.lstrip().startswith("{"):
                try:
                    obj = json.loads(body)
                    if isinstance(obj.get("paths"), dict):
                        hits |= {k for k in obj["paths"] if k.startswith("/")}
                except json.JSONDecodeError:
                    pass
            hits |= {"/" + m.rstrip("/.") for m in REF_RE.findall(body)}
        found |= hits
        log.append({"url": url, "status": status, "found": len(hits),
                    "bytes": len(body) if status == 200 else 0})
        print(f"  {status:>3} {len(hits):>4}本  {red(url)}")
    # /v1/ /v2/ の前置きを外して、叩く形（/fins/summary）に揃える
    norm = set()
    for p in found:
        p = re.sub(r"^/v[12]", "", p)
        if p.startswith("/") and 3 <= len(p) <= 64 and not p.endswith(("}", ".json")):
            norm.add(p)
    return sorted(norm), log


def probe(paths: List[str], key: str, red: Redactor) -> List[dict]:
    """各パスを叩いて OK / PLAN（契約不足）/ NOT_FOUND に分ける。"""
    headers = {"x-api-key": key}
    out = []
    for path in paths:
        rec: Dict[str, Any] = {"path": path}
        for shape, params in PARAM_SHAPES:
            url = API_BASE + path
            if params:
                url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
            time.sleep(PAUSE)
            status, body = fetch(url, headers)
            msg = ""
            try:
                msg = str(json.loads(body).get("message", ""))[:200]
            except Exception:                                # noqa: BLE001
                msg = body[:200]
            if status == 200:
                try:
                    data = json.loads(body)
                    batch = data.get("data")
                    rec.update(kind="OK", shape=shape, status=status,
                               rows=len(batch) if isinstance(batch, list) else 0,
                               keys=sorted(batch[0].keys())[:40] if batch else [])
                except Exception:                            # noqa: BLE001
                    rec.update(kind="OK", shape=shape, status=status, rows=0, keys=[])
                break
            low = msg.lower()
            if "not available on your subscription" in low:
                rec.update(kind="PLAN", shape=shape, status=status, msg=msg)
                break
            if "does not exist" in low:
                rec.update(kind="NOT_FOUND", shape=shape, status=status, msg=msg)
                break
            # 400 系は「パラメータが足りない」= 在る。次の形を試す
            rec.update(kind="NEEDS_PARAMS", shape=shape, status=status, msg=msg)
        out.append(rec)
        print(f"  {rec.get('kind','?'):<13}{rec.get('status',''):>4} "
              f"{path:<40}{red(rec.get('msg',''))[:90]}")
    return out


def render(res: List[dict], log: List[dict], red: Redactor) -> str:
    def rows(kind: str) -> List[dict]:
        return [r for r in res if r.get("kind") == kind]

    L = ["", "### エンドポイント一覧を取りに行ってから叩いた結果（2026-09-22）", "",
         "Claude の作業環境からは公式ドキュメントに到達できない（egress ブロック）ので、",
         "**GitHub Actions のランナーで一覧の取得ごとやる**"
         "（`research/probe_endpoints.py` / `Probe Endpoints`）。",
         "思いついた名前を並べるのではなく、取ってきた一覧を全部叩いている。", "",
         "#### 一覧の取得元", "",
         "| 取得元 | HTTP | 拾えたパス |", "|---|---:|---:|"]
    for e in log:
        L.append(f"| `{red(e['url'])}` | {e['status'] or '到達不可'} | {e['found']} |")
    L += ["", f"叩いたパス **{len(res)}本**"
          f"（OK {len(rows('OK'))} / 契約不足 {len(rows('PLAN'))} / "
          f"存在しない {len(rows('NOT_FOUND'))} / "
          f"引数不足 {len(rows('NEEDS_PARAMS'))}）", ""]

    L += ["#### 使える（OK）", "", "| パス | 引数 | 件数 | 主な項目 |", "|---|---|---:|---|"]
    for r in sorted(rows("OK"), key=lambda x: x["path"]):
        k = ", ".join(f"`{c}`" for c in r.get("keys", [])[:8])
        L.append(f"| `{r['path']}` | {r.get('shape')} | {r.get('rows',0)} | {k} |")

    L += ["", "#### 在るが契約が足りない（プレミアムで開く）", "",
          "| パス | メッセージ |", "|---|---|"]
    for r in sorted(rows("PLAN"), key=lambda x: x["path"]):
        L.append(f"| `{r['path']}` | {red(r.get('msg',''))[:110]} |")
    if not rows("PLAN"):
        L.append("| （該当なし） | |")

    if rows("NEEDS_PARAMS"):
        L += ["", "#### 引数が足りず判定できなかった", "", "| パス | HTTP | メッセージ |",
              "|---|---:|---|"]
        for r in sorted(rows("NEEDS_PARAMS"), key=lambda x: x["path"]):
            L.append(f"| `{r['path']}` | {r.get('status')} | {red(r.get('msg',''))[:110]} |")

    L += ["", "#### そのパスには何も無い", "",
          "、".join(f"`{r['path']}`" for r in sorted(rows("NOT_FOUND"),
                                                    key=lambda x: x["path"])) or "（該当なし）",
          ""]
    return "\n".join(L)


def write_back(section: str, path: str = None) -> None:
    path = path or os.path.join(os.path.dirname(HERE), "docs", "DATA_FIELDS.md")
    mark = "### エンドポイント一覧を取りに行ってから叩いた結果"
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    if mark in s:
        i = s.index(mark)
        j = s.find("\n## ", i)
        s = s[:i].rstrip("\n") + "\n" + section + (s[j:] if j > 0 else "\n")
    else:
        s = s.rstrip("\n") + "\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s)
    print(f"[write] {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="J-Quants のエンドポイントを発見して叩く")
    ap.add_argument("--no-doc", action="store_true", help="一覧の取得を飛ばす")
    ap.add_argument("--limit", type=int, default=120, help="叩く本数の上限")
    args = ap.parse_args(argv)

    key = os.environ.get("JQUANTS_API") or os.environ.get("JQUANTS_API_KEY") or ""
    red = Redactor(key)
    if not key:
        print("[fatal] JQUANTS_API が未設定です", file=sys.stderr)
        return 1
    print(f"[auth] 鍵を読み込みました（長さ {len(key)}、先頭 {key[:2]}…）")

    print("\n=== 1. 一覧の取得（ランナーの外向き通信）===")
    found, log = ([], []) if args.no_doc else discover(red)
    print(f"  発見 {len(found)}本")

    paths = sorted(set(found) | set(HAND))[: args.limit]
    print(f"\n=== 2. 叩く（{len(paths)}本）===")
    res = probe(paths, key, red)

    out = os.path.join(HERE, "probe_endpoints.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"discovered": found, "sources": log, "results": res},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n[write] {out}")
    write_back(render(res, log, red))
    return 0


if __name__ == "__main__":
    sys.exit(main())
