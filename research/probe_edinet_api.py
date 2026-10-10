#!/usr/bin/env python3
"""
金融庁 EDINET の公式 API（Version 2）が実際に何を返すかを実測する。

なぜ実測するか
------------
運用者の指示（2026-10-10）: 候補銘柄の直近の決算・有価証券報告書をサイトから直接取り、
原価・販管費まで分けた損益のサンキー図を画面に出す。四半期報告書は 2024年4月以降の四半期から
廃止されたので、明細が取れる法定の書類は 有価証券報告書（docTypeCode 120）と 半期報告書（160）。
EDINET の公式 API は無料で、鍵（Subscription-Key）の発行だけが要る。利用規約は公共データ利用規約
（PDL1.0。出典を書けば加工・公開してよい。サイトのスクレイピングは禁止で、API が正しい経路）。

取り込みを書く前に、次を実測で確かめる（項目名を推測して書かない）。
  1. 鍵の渡し方（仕様書では Subscription-Key をクエリに付ける）で開くか
  2. 書類一覧 API（documents.json?date=…&type=2）の応答の形: 項目名・docTypeCode の内訳・csvFlag
  3. 日付を遡って、ある証券コードの最新の 120 / 160 を見つけるのに何リクエスト要るか
  4. 書類取得 API（documents/{docID}?type=5）の CSV（ZIP）の中身: ファイル名・文字コード・列名・
     コンテキスト ID・損益の要素 ID（日本基準 jppfs_cor / IFRS jpigp_cor）・DEI（会計基準・期間の種類）
  5. 損益の恒等式（粗利 = 売上 − 原価、営業利益 = 粗利 − 販管費 …）が成り立つか（真偽だけ印字）

安全（リポジトリも Actions のログも公開である前提）
----------------------------------------------
- 鍵は出力に絶対に出さない。URL は鍵を除いた形でしか印字せず、出力に鍵の文字列が混ざったら
  [REDACTED] に置換する（probe_edinetdb.Redactor）。
- 会社名の一覧は印字しない（件数と docTypeCode の内訳だけ）。財務の値も印字しない（恒等式の真偽と
  件数だけ）。EDINET の情報は PDL1.0 で公開してよいが、ここでは必要が無い。
- リクエスト数は --max-requests で上限を固定する（既定 150）。1リクエストごとに 0.3 秒空ける。

  $ EDINET_FSA_API_KEY=... python3 research/probe_edinet_api.py --sec-codes 79740 72030
  $ python3 research/probe_edinet_api.py --fake        # 偽の応答で印字の形だけ確かめる（手元）
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_edinetdb import Redactor  # noqa: E402

BASE = "https://api.edinet-fsa.go.jp/api/v2"
KEY_ENV = "EDINET_FSA_API_KEY"
#: 書類の種類（EDINET の docTypeCode）。有報・半期報告書とその訂正
DOC_TYPES = {"120": "有価証券報告書", "130": "訂正有価証券報告書", "140": "四半期報告書",
             "150": "訂正四半期報告書", "160": "半期報告書", "170": "訂正半期報告書"}
WANT = ("120", "160")
#: CSV の列（EDINET 閲覧操作ガイドの説明に基づく候補。実際の見出しを印字して確かめる）
COLS = {"element": ("要素ID",), "label": ("項目名",), "context": ("コンテキストID",),
        "rel_year": ("相対年度",), "cons": ("連結・個別",), "period": ("期間・時点",),
        "unit": ("ユニットID",), "unit_label": ("単位",), "value": ("値",)}
#: 損益の要素 ID の候補（接頭辞は無し）。日本基準と IFRS。無いものは無いと印字する
PL_JGAAP = ["NetSales", "OperatingRevenue1", "CostOfSales", "GrossProfit",
            "SellingGeneralAndAdministrativeExpenses", "OperatingIncome",
            "NonOperatingIncome", "NonOperatingExpenses", "OrdinaryIncome",
            "ExtraordinaryIncome", "ExtraordinaryLoss", "IncomeBeforeIncomeTaxes",
            "IncomeTaxes", "IncomeTaxesCurrent", "IncomeTaxesDeferred", "ProfitLoss",
            "ProfitLossAttributableToOwnersOfParent",
            "ProfitLossAttributableToNonControllingInterests"]
PL_IFRS = ["RevenueIFRS", "CostOfSalesIFRS", "GrossProfitIFRS",
           "SellingGeneralAndAdministrativeExpensesIFRS", "OtherIncomeIFRS", "OtherExpensesIFRS",
           "OperatingProfitLossIFRS", "FinanceIncomeIFRS", "FinanceCostsIFRS",
           "ProfitLossBeforeTaxIFRS", "IncomeTaxExpenseIFRS", "ProfitLossIFRS",
           "ProfitLossAttributableToOwnersOfParentIFRS",
           "ProfitLossAttributableToNonControllingInterestsIFRS"]
#: 書類の情報（DEI）。値は会計基準・期間の種類などで、財務の値ではないので印字する
DEI = ["EDINETCodeDEI", "SecurityCodeDEI", "DocumentTypeDEI", "AccountingStandardsDEI",
       "WhetherConsolidatedFinancialStatementsArePreparedDEI", "TypeOfCurrentPeriodDEI",
       "CurrentFiscalYearStartDateDEI", "CurrentFiscalYearEndDateDEI", "CurrentPeriodEndDateDEI",
       "AmendmentFlagDEI"]
#: 恒等式（左辺 = 右辺の和。符号は +1 / −1）。成り立つかの真偽だけ印字する
IDENTITIES_JGAAP = [
    ("GrossProfit", [("NetSales", 1), ("CostOfSales", -1)]),
    ("OperatingIncome", [("GrossProfit", 1), ("SellingGeneralAndAdministrativeExpenses", -1)]),
    ("OrdinaryIncome", [("OperatingIncome", 1), ("NonOperatingIncome", 1), ("NonOperatingExpenses", -1)]),
    ("IncomeBeforeIncomeTaxes", [("OrdinaryIncome", 1), ("ExtraordinaryIncome", 1), ("ExtraordinaryLoss", -1)]),
    ("ProfitLoss", [("IncomeBeforeIncomeTaxes", 1), ("IncomeTaxes", -1)]),
]
IDENTITIES_IFRS = [
    ("GrossProfitIFRS", [("RevenueIFRS", 1), ("CostOfSalesIFRS", -1)]),
    ("ProfitLossIFRS", [("ProfitLossBeforeTaxIFRS", 1), ("IncomeTaxExpenseIFRS", -1)]),
]
HEADER_ALLOW = re.compile(r"ratelimit|remaining|quota|limit|retry|content-type|content-length", re.I)
SEC_CODE = re.compile(r"^[0-9][0-9A-Z]{3}0$")


class Probe:
    def __init__(self, key: Optional[str], budget: int, timeout: int = 60, pause: float = 0.3,
                 transport=None):
        self.key = key
        self.budget = budget
        self.used = 0
        self.timeout = timeout
        self.pause = pause
        self.red = Redactor(key)
        self.last_headers: Dict[str, str] = {}
        self.timings: List[float] = []
        self.transport = transport          # テスト用: (url) -> (status, bytes, headers)

    def say(self, *parts: Any) -> None:
        print(" ".join(self.red(p) for p in parts), flush=True)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], Any]:
        """
        (HTTP ステータス, 本文)。本文は JSON なら dict / list、それ以外は bytes。
        予算切れ・鍵なし・接続失敗は (None, 理由)。印字する URL に鍵は含めない。
        """
        if self.used >= self.budget:
            self.say(f"  [skip] リクエスト予算 {self.budget} を使い切ったので飛ばす: {path}")
            return None, "budget"
        if not self.key:
            return None, "no-key"
        shown = path + ("?" + urllib.parse.urlencode(params) if params else "")
        q = dict(params or {})
        q["Subscription-Key"] = self.key
        url = BASE + path + "?" + urllib.parse.urlencode(q)
        self.used += 1
        t0 = time.time()
        try:
            if self.transport is not None:
                status, raw, headers = self.transport(url)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "growthstock-probe/1"})
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        raw, status = resp.read(), resp.status
                        headers = {k: v for k, v in resp.headers.items()}
                except urllib.error.HTTPError as exc:
                    raw, status = exc.read(), exc.code
                    headers = {k: v for k, v in exc.headers.items()}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.say(f"  [error] 接続できない: {type(exc).__name__}: {str(exc)[:160]}")
            return None, "unreachable"
        self.timings.append(time.time() - t0)
        self.last_headers = headers
        ctype = next((v for k, v in headers.items() if k.lower() == "content-type"), "")
        body: Any = raw
        if "json" in ctype or (raw[:1] in (b"{", b"[")):
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = raw
        self.say(f"  GET {shown} -> HTTP {status} ({len(raw):,} bytes, {self.timings[-1]:.2f}s)")
        if self.pause and self.transport is None:
            time.sleep(self.pause)
        return status, body

    def show_headers(self) -> None:
        picked = {k: v for k, v in self.last_headers.items() if HEADER_ALLOW.search(k)}
        if picked:
            for k, v in sorted(picked.items()):
                self.say(f"    {k}: {v}")
        else:
            self.say("    （利用枠らしいヘッダは無し）")


# ---------------------------------------------------------------------- #
# 書類一覧
# ---------------------------------------------------------------------- #

def weekdays_back(today: dt.date, n: int) -> Iterable[dt.date]:
    """today から過去へ、平日だけを n 日ぶん。"""
    d = today
    k = 0
    while k < n:
        if d.weekday() < 5:
            yield d
            k += 1
        d -= dt.timedelta(days=1)


def results_of(body: Any) -> List[Dict[str, Any]]:
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        return [r for r in body["results"] if isinstance(r, dict)]
    return []


def describe_list(p: Probe, rows: List[Dict[str, Any]]) -> None:
    """一覧の形: 項目名・型・充足。会社名などの文字列の例は出さない（件数の情報だけ）。"""
    if not rows:
        p.say("  行なし")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p.say(f"  {len(rows)}行 / {len(keys)}項目")
    p.say(f"    {'項目':<26}{'型':<10}{'充足':>8}  値の種類（文字列は長さ・数の分布だけ）")
    for k in keys:
        vals = [r.get(k) for r in rows]
        nonnull = [v for v in vals if v not in (None, "")]
        types = sorted({type(v).__name__ for v in nonnull}) or ["null"]
        note = ""
        if nonnull:
            if k in ("docTypeCode", "ordinanceCode", "formCode", "xbrlFlag", "pdfFlag", "csvFlag",
                     "attachDocFlag", "englishDocFlag", "withdrawalStatus", "docInfoEditStatus",
                     "disclosureStatus", "legalStatus"):
                vc: Dict[str, int] = {}
                for v in nonnull:
                    vc[str(v)] = vc.get(str(v), 0) + 1
                note = ", ".join(f"{a}×{b}" for a, b in sorted(vc.items())[:12])
            elif k in ("docID", "edinetCode", "secCode", "JCN", "fundCode", "parentDocID",
                       "issuerEdinetCode", "subjectEdinetCode", "subsidiaryEdinetCode"):
                lens = sorted({len(str(v)) for v in nonnull})
                note = f"長さ {lens}"
            elif k in ("submitDateTime", "periodStart", "periodEnd", "opeDateTime"):
                note = f"例 {nonnull[0]}"
            elif k == "seqNumber":
                note = f"{min(nonnull)}〜{max(nonnull)}"
            else:
                note = "（文字列。印字しない）"
        p.say(f"    {k:<26}{'/'.join(types):<10}{len(nonnull):>4}/{len(rows):<3}  {note}")


def type_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("docTypeCode") or "?")
        out[k] = out.get(k, 0) + 1
    return out


def pick_latest(rows: Iterable[Dict[str, Any]], sec_code: str, types: Tuple[str, ...] = WANT) -> Optional[Dict[str, Any]]:
    """
    一覧の行から、証券コードが sec_code で docTypeCode が types のうち、提出日時が最新のもの。
    取り下げ（withdrawalStatus が "0" でない）と CSV の無いものは除く。
    """
    best = None
    for r in rows:
        if str(r.get("secCode") or "") != sec_code or str(r.get("docTypeCode") or "") not in types:
            continue
        if str(r.get("withdrawalStatus") or "0") != "0":
            continue
        if str(r.get("csvFlag") or "0") != "1":
            continue
        if best is None or str(r.get("submitDateTime") or "") > str(best.get("submitDateTime") or ""):
            best = r
    return best


# ---------------------------------------------------------------------- #
# CSV（type=5 の ZIP）
# ---------------------------------------------------------------------- #

def decode_csv(raw: bytes) -> Tuple[str, str]:
    """(文字コードの名前, 文字列)。UTF-16（BOM）→ UTF-8（BOM）→ UTF-8 → CP932 の順に試す。"""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16", raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig", raw.decode("utf-8-sig")
    try:
        return "utf-8", raw.decode("utf-8")
    except UnicodeDecodeError:
        return "cp932", raw.decode("cp932", errors="replace")


def parse_rows(text: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    """
    タブ区切り（タブが無ければカンマ）の表を読む。見出しは1行目。
    戻り値: (見出し, 行の dict の列, 区切り文字の名前)。
    """
    first = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first else ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header = [h.strip() for h in next(reader, [])]
    rows = [dict(zip(header, r)) for r in reader if any(c.strip() for c in r)]
    return header, rows, ("tab" if delim == "\t" else "comma")


def col(header: List[str], key: str) -> Optional[str]:
    """COLS の候補のうち見出しにあるもの。無ければ None。"""
    for c in COLS[key]:
        if c in header:
            return c
    return None


def local_name(element: str) -> str:
    return element.split(":", 1)[1] if ":" in element else element


def to_number(v: str) -> Optional[float]:
    s = (v or "").strip().replace(",", "")
    if s in ("", "-", "－", "―"):
        return None
    neg = s.startswith("△") or s.startswith("▲")
    s = s.lstrip("△▲")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def pl_values(rows: List[Dict[str, str]], header: List[str], context: str) -> Dict[str, float]:
    """あるコンテキストの、損益候補の要素 ID（接頭辞なし）→ 値。"""
    ce, cc, cv = col(header, "element"), col(header, "context"), col(header, "value")
    out: Dict[str, float] = {}
    if not (ce and cc and cv):
        return out
    want = set(PL_JGAAP) | set(PL_IFRS)
    for r in rows:
        name = local_name(r.get(ce, ""))
        if name in want and r.get(cc) == context:
            x = to_number(r.get(cv, ""))
            if x is not None and name not in out:
                out[name] = x
    return out


def rounding_unit(values: Iterable[float]) -> float:
    """値がすべて割り切れる最大の単位（百万・千・1）。開示が百万円に丸めてあれば 1e6。"""
    vs = [abs(float(v)) for v in values if v is not None]
    for unit in (1e6, 1e3):
        if vs and all(v % unit == 0 for v in vs):
            return unit
    return 1.0


def check_identities(vals: Dict[str, float], identities) -> List[Tuple[str, Optional[bool]]]:
    """
    各恒等式が成り立つか（None は項目が足りない）。値は返さない。
    開示が百万円（千円）に丸めてあると、丸めた項目の足し合わせは最大で 項目数 × 0.5 単位ずれる
    （実測 2026-10-10: 小さい会社の有報で経常・税引前・純利益の3本が「不成立」になった）。その分は許す。
    """
    out = []
    for lhs, terms in identities:
        if lhs not in vals or any(t not in vals for t, _ in terms):
            out.append((lhs, None))
            continue
        rhs = sum(s * vals[t] for t, s in terms)
        unit = rounding_unit([vals[lhs]] + [vals[t] for t, _ in terms])
        tol = max(2.0, 0.5 * unit * (len(terms) + 1))
        out.append((lhs, abs(vals[lhs] - rhs) <= tol))
    return out


def describe_csv(p: Probe, name: str, raw: bytes) -> Dict[str, Any]:
    """1つの CSV の形を印字し、要点を dict で返す（テスト用）。"""
    enc, text = decode_csv(raw)
    header, rows, delim = parse_rows(text)
    p.say(f"  {name}: {enc} / {delim} 区切り / {len(rows):,}行")
    p.say(f"    見出し: {header}")
    missing = [k for k in COLS if col(header, k) is None]
    if missing:
        p.say(f"    [注意] 想定の列が無い: {missing}（上の見出しで取り込みを書く）")
    ce, cc, cv = col(header, "element"), col(header, "context"), col(header, "value")
    info: Dict[str, Any] = {"encoding": enc, "delim": delim, "header": header, "rows": len(rows)}
    if not (ce and cc):
        return info
    # 分類の列の語彙
    for key in ("cons", "period", "rel_year", "unit"):
        c = col(header, key)
        if c:
            vc: Dict[str, int] = {}
            for r in rows:
                vc[r.get(c, "")] = vc.get(r.get(c, ""), 0) + 1
            top = sorted(vc.items(), key=lambda kv: -kv[1])[:8]
            p.say(f"    {c}: " + ", ".join(f"{a or '(空)'}×{b}" for a, b in top))
    # コンテキスト ID（件数の多い順）
    vc = {}
    for r in rows:
        vc[r.get(cc, "")] = vc.get(r.get(cc, ""), 0) + 1
    ctxs = sorted(vc.items(), key=lambda kv: -kv[1])
    p.say(f"    コンテキスト ID {len(ctxs)}種（多い順に 20）:")
    for a, b in ctxs[:20]:
        p.say(f"      {a or '(空)'}  ×{b}")
    info["contexts"] = [a for a, _ in ctxs]
    # DEI
    dei = {}
    for r in rows:
        nm = local_name(r.get(ce, ""))
        if nm in DEI and nm not in dei:
            dei[nm] = (r.get(cv, "") if cv else "")
    p.say("    DEI（書類の情報）:")
    for k in DEI:
        p.say(f"      {k:<56} {dei.get(k, '（無し）')}")
    info["dei"] = dei
    # 損益の要素 ID の有無と、どのコンテキストに出るか
    present: Dict[str, List[str]] = {}
    for r in rows:
        nm = local_name(r.get(ce, ""))
        if nm in PL_JGAAP or nm in PL_IFRS:
            present.setdefault(nm, [])
            if r.get(cc, "") not in present[nm]:
                present[nm].append(r.get(cc, ""))
    for title, names in (("日本基準（jppfs_cor）", PL_JGAAP), ("IFRS（jpigp_cor）", PL_IFRS)):
        hit = [n for n in names if n in present]
        p.say(f"    損益の要素 {title}: {len(hit)}/{len(names)} 本ある")
        for n in names:
            if n in present:
                p.say(f"      {n:<52} {', '.join(present[n][:6])}")
    info["present"] = sorted(present)
    # 当期（連結）らしいコンテキストでの恒等式
    cur = [c for c, _ in ctxs if c.startswith("Current") and c.endswith("Duration")]
    cur = sorted(cur, key=lambda c: (("NonConsolidated" in c), len(c)))
    info["identities"] = {}
    for ctx in cur[:2]:
        vals = pl_values(rows, header, ctx)
        ids = IDENTITIES_IFRS if any(n.endswith("IFRS") for n in vals) else IDENTITIES_JGAAP
        res = check_identities(vals, ids)
        p.say(f"    恒等式（{ctx}、項目 {len(vals)}本）: "
              + ", ".join(f"{lhs}={'成立' if ok else '不成立' if ok is False else '項目不足'}" for lhs, ok in res))
        info["identities"][ctx] = {lhs: ok for lhs, ok in res}
    # 当期の Duration に出る要素のうち、損益の候補に無いもの（名前だけ。拡張の手がかり）
    if cur:
        extra = []
        for r in rows:
            if r.get(cc, "") == cur[0]:
                el = r.get(ce, "")
                if (el.startswith("jppfs_cor:") or el.startswith("jpigp_cor:")) and local_name(el) not in present:
                    if el not in extra:
                        extra.append(el)
        p.say(f"    {cur[0]} に出る jppfs_cor / jpigp_cor のうち候補に無い要素 {len(extra)}本（先頭 40）:")
        for el in extra[:40]:
            p.say(f"      {el}")
    return info


def describe_zip(p: Probe, raw: bytes) -> Dict[str, Any]:
    """ZIP の中身を並べ、本体の CSV（jpcrp*）を読む。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        p.say(f"  [注意] ZIP として読めない（先頭 {raw[:16]!r}）")
        return {"zip": False}
    names = zf.namelist()
    p.say(f"  ZIP: {len(names)} ファイル")
    for n in names[:30]:
        p.say(f"    {n}  ({zf.getinfo(n).file_size:,} bytes)")
    csvs = [n for n in names if n.lower().endswith(".csv")]
    main_ = [n for n in csvs if os.path.basename(n).startswith("jpcrp")] or csvs
    out: Dict[str, Any] = {"zip": True, "files": names, "csv": {}}
    for n in main_[:2]:
        out["csv"][n] = describe_csv(p, os.path.basename(n), zf.read(n))
    return out


# ---------------------------------------------------------------------- #

def run(p: Probe, sec_codes: List[str], scan_days: int, busy_date: Optional[str], today: dt.date) -> int:
    p.say("=== 1. 書類一覧 API（決算の集中する日の内訳）===")
    if busy_date:
        st, body = p.get("/documents.json", {"date": busy_date, "type": "2"})
        p.show_headers()
        if st == 200 and isinstance(body, dict):
            meta = body.get("metadata", {})
            p.say(f"  metadata.status={meta.get('status')} message={meta.get('message')} "
                  f"resultset={meta.get('resultset')}")
            rows = results_of(body)
            tc = type_counts(rows)
            p.say("  docTypeCode の内訳: " + ", ".join(
                f"{k}({DOC_TYPES.get(k, '?')})×{v}" for k, v in sorted(tc.items(), key=lambda kv: -kv[1])[:12]))
            want = [r for r in rows if str(r.get("docTypeCode")) in WANT]
            csv_ok = sum(1 for r in want if str(r.get("csvFlag")) == "1")
            p.say(f"  120/160 は {len(want)}件、うち csvFlag=1 は {csv_ok}件")
            describe_list(p, rows)
        elif st is not None:
            p.say(f"  [stop] 一覧が取れない（HTTP {st}）。本文の先頭: "
                  f"{(body if isinstance(body, (dict, list)) else body[:200])!r}"[:300])
            return 1
        else:
            return 1

    p.say(f"\n=== 2. 日付を遡って、証券コード {sec_codes} の最新の 120 / 160 を探す（平日 {scan_days} 日まで）===")
    found: Dict[str, Dict[str, Any]] = {}
    first_160: Optional[Dict[str, Any]] = None
    dates_hit: Dict[str, str] = {}
    for d in weekdays_back(today, scan_days):
        if all(c in found for c in sec_codes) and first_160 is not None:
            break
        if p.used >= p.budget - 4:              # 書類取得のぶんを残す
            p.say("  [stop] 予算が残り少ないので遡るのをやめる")
            break
        st, body = p.get("/documents.json", {"date": d.isoformat(), "type": "2"})
        if st != 200 or not isinstance(body, dict):
            p.say(f"  [注意] {d} は HTTP {st}")
            continue
        rows = results_of(body)
        for c in sec_codes:
            if c not in found:
                hit = pick_latest(rows, c)
                if hit:
                    found[c] = hit
                    dates_hit[c] = d.isoformat()
                    p.say(f"  {c}: {d} に docTypeCode {hit.get('docTypeCode')}"
                          f"（{DOC_TYPES.get(str(hit.get('docTypeCode')), '?')}）"
                          f" periodEnd={hit.get('periodEnd')} docID の長さ {len(str(hit.get('docID')))}")
        if first_160 is None:
            cand = [r for r in rows if str(r.get("docTypeCode")) == "160" and str(r.get("csvFlag")) == "1"
                    and str(r.get("withdrawalStatus") or "0") == "0"]
            if cand:
                first_160 = cand[0]
                p.say(f"  半期報告書（160）を {d} に見つけた（{len(cand)}件のうち1件を使う）")
    p.say(f"  遡った結果: 見つかった {sorted(found)} / 見つからない {[c for c in sec_codes if c not in found]}"
          f" / ここまでのリクエスト {p.used}")
    if p.timings:
        ts = sorted(p.timings)
        p.say(f"  応答時間: 中央値 {ts[len(ts)//2]:.2f}s / 最大 {ts[-1]:.2f}s")

    p.say("\n=== 3. 書類取得 API（type=5 の CSV）===")
    docs = [(f"{c}（{DOC_TYPES.get(str(r.get('docTypeCode')), '?')}）", r) for c, r in found.items()]
    if first_160 is not None and all(r.get("docID") != first_160.get("docID") for _, r in docs):
        docs.append(("半期報告書の例", first_160))
    results: Dict[str, Any] = {}
    for title, r in docs:
        doc_id = str(r.get("docID") or "")
        p.say(f"\n--- {title} ---")
        st, body = p.get(f"/documents/{doc_id}", {"type": "5"})
        p.show_headers()
        if st != 200 or not isinstance(body, (bytes, bytearray)):
            p.say(f"  [注意] 取れない（HTTP {st}）: {str(body)[:200]!r}")
            continue
        results[title] = describe_zip(p, bytes(body))

    p.say(f"\n[done] リクエスト {p.used} / 予算 {p.budget}")
    return 0 if results else 1


# ---------------------------------------------------------------------- #
# 偽の応答（印字の形の確認とテスト）
# ---------------------------------------------------------------------- #

def fake_csv_text(ifrs: bool = False) -> str:
    """恒等式が成り立つ偽の損益（値は作りもの）。"""
    ctx = "CurrentYearDuration"
    if ifrs:
        rows = [("jpigp_cor:RevenueIFRS", "売上収益", 1000), ("jpigp_cor:CostOfSalesIFRS", "売上原価", 600),
                ("jpigp_cor:GrossProfitIFRS", "売上総利益", 400),
                ("jpigp_cor:SellingGeneralAndAdministrativeExpensesIFRS", "販売費及び一般管理費", 250),
                ("jpigp_cor:OperatingProfitLossIFRS", "営業利益", 150),
                ("jpigp_cor:ProfitLossBeforeTaxIFRS", "税引前利益", 140),
                ("jpigp_cor:IncomeTaxExpenseIFRS", "法人所得税費用", 40),
                ("jpigp_cor:ProfitLossIFRS", "当期利益", 100)]
        std = "IFRS"
    else:
        rows = [("jppfs_cor:NetSales", "売上高", 1000), ("jppfs_cor:CostOfSales", "売上原価", 600),
                ("jppfs_cor:GrossProfit", "売上総利益", 400),
                ("jppfs_cor:SellingGeneralAndAdministrativeExpenses", "販売費及び一般管理費", 250),
                ("jppfs_cor:OperatingIncome", "営業利益", 150),
                ("jppfs_cor:NonOperatingIncome", "営業外収益", 20), ("jppfs_cor:NonOperatingExpenses", "営業外費用", 10),
                ("jppfs_cor:OrdinaryIncome", "経常利益", 160),
                ("jppfs_cor:ExtraordinaryIncome", "特別利益", 5), ("jppfs_cor:ExtraordinaryLoss", "特別損失", 15),
                ("jppfs_cor:IncomeBeforeIncomeTaxes", "税金等調整前当期純利益", 150),
                ("jppfs_cor:IncomeTaxes", "法人税等", 45), ("jppfs_cor:ProfitLoss", "当期純利益", 105),
                ("jppfs_cor:ProfitLossAttributableToOwnersOfParent", "親会社株主に帰属する当期純利益", 100)]
        std = "Japan GAAP"
    lines = ["\t".join(["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"])]
    lines.append("\t".join(["jpdei_cor:AccountingStandardsDEI", "会計基準", "FilingDateInstant", "提出日時点", "その他", "時点", "", "", std]))
    lines.append("\t".join(["jpdei_cor:TypeOfCurrentPeriodDEI", "当会計期間の種類", "FilingDateInstant", "提出日時点", "その他", "時点", "", "", "FY"]))
    lines.append("\t".join(["jpdei_cor:SecurityCodeDEI", "証券コード", "FilingDateInstant", "提出日時点", "その他", "時点", "", "", "00000"]))
    for el, lab, v in rows:
        lines.append("\t".join([el, lab, ctx, "当期", "連結", "期間", "JPY", "円", str(v)]))
        lines.append("\t".join([el, lab, "Prior1YearDuration", "前期", "連結", "期間", "JPY", "円", str(v - 1)]))
    return "\n".join(lines) + "\n"


def fake_zip(ifrs: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000_2026-03-31_01_2026-06-25.csv",
                    b"\xff\xfe" + fake_csv_text(ifrs).encode("utf-16-le"))
        zf.writestr("XBRL_TO_CSV/jpaud-aar-cn-001_E00000-000_2026-03-31_01_2026-06-25.csv",
                    b"\xff\xfe" + "要素ID\t値\n".encode("utf-16-le"))
    return buf.getvalue()


def fake_transport(today: dt.date):
    """偽の API。一覧は決まった日に 120 / 160 を返し、取得は偽の ZIP を返す。"""
    def row(doc_id, sec, typ, date):
        return {"seqNumber": 1, "docID": doc_id, "edinetCode": "E00000", "secCode": sec, "JCN": "1" * 13,
                "filerName": "テスト株式会社", "fundCode": None, "ordinanceCode": "010", "formCode": "030000",
                "docTypeCode": typ, "periodStart": "2025-04-01", "periodEnd": "2026-03-31",
                "submitDateTime": f"{date} 15:00", "docDescription": "テスト", "issuerEdinetCode": None,
                "subjectEdinetCode": None, "subsidiaryEdinetCode": None, "currentReportReason": None,
                "parentDocID": None, "opeDateTime": None, "withdrawalStatus": "0", "docInfoEditStatus": "0",
                "disclosureStatus": "0", "xbrlFlag": "1", "pdfFlag": "1", "attachDocFlag": "1",
                "englishDocFlag": "0", "csvFlag": "1", "legalStatus": "1"}
    d1 = (today - dt.timedelta(days=3)).isoformat()
    d2 = (today - dt.timedelta(days=10)).isoformat()
    listing = {"2026-06-26": [row("S100AAAA", "99990", "120", "2026-06-26")] * 3 + [row("S100BBBB", "99980", "160", "2026-06-26")],
               d1: [row("S100C001", "79740", "120", d1), row("S100C002", "11110", "160", d1)],
               d2: [row("S100C003", "72030", "160", d2)]}

    def transport(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        path = urllib.parse.urlparse(url).path
        hdr = {"Content-Type": "application/json; charset=utf-8"}
        if path.endswith("/documents.json"):
            d = q.get("date", [""])[0]
            rows = listing.get(d, [])
            body = {"metadata": {"title": "提出された書類を把握するためのAPI", "parameter": {"date": d, "type": "2"},
                                 "resultset": {"count": len(rows)}, "processDateTime": f"{d} 00:00",
                                 "status": "200", "message": "OK"}, "results": rows}
            return 200, json.dumps(body, ensure_ascii=False).encode("utf-8"), hdr
        if "/documents/" in path:
            doc = path.rsplit("/", 1)[1]
            return 200, fake_zip(ifrs=doc.endswith("3")), {"Content-Type": "application/octet-stream"}
        return 404, b"{}", hdr
    return transport


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EDINET 公式 API（v2）の形を実測する")
    ap.add_argument("--sec-codes", nargs="*", default=["79740", "72030"],
                    help="最新の 120 / 160 を探す証券コード（5桁。末尾 0）")
    ap.add_argument("--max-requests", type=int, default=150)
    ap.add_argument("--scan-days", type=int, default=140, help="遡る平日の数")
    ap.add_argument("--busy-date", default="2026-06-26", help="内訳を見る日（空なら飛ばす）")
    ap.add_argument("--today", default=None, help="遡り始める日（既定は今日）")
    ap.add_argument("--fake", action="store_true", help="偽の応答で印字の形だけ確かめる")
    args = ap.parse_args(argv)
    for c in args.sec_codes:
        if not SEC_CODE.match(c):
            print(f"[stop] 証券コードは5桁（末尾 0）: {c}")
            return 2
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    key = os.environ.get(KEY_ENV, "").strip() or None
    if args.fake:
        p = Probe("TESTONLY-fake-key", args.max_requests, transport=fake_transport(today))
    else:
        if not key:
            print(f"[stop] {KEY_ENV} が無い")
            return 1
        p = Probe(key, args.max_requests)
    p.say(f"EDINET API v2 / 予算 {args.max_requests} / 遡り {args.scan_days} 平日 / 鍵 {'あり' if p.key else 'なし'}"
          f"（長さ {len(p.key or '')}）")
    return run(p, args.sec_codes, args.scan_days, args.busy_date or None, today)


if __name__ == "__main__":
    raise SystemExit(main())
