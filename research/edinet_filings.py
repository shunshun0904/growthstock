#!/usr/bin/env python3
"""
候補銘柄の最新の有価証券報告書・半期報告書を 金融庁 EDINET の公式 API（v2）から取り、
損益の明細を画面用の JSON（public/data/filings.json）にする。

運用者の指示（2026-10-10）: 候補銘柄の直近の決算・有報をサイトから直接取り、原価・販管費まで分けた
損益のサンキー図を画面のタブに出す。四半期報告書は廃止されたので（2024年4月以降に始まる四半期）、
明細が取れる法定の書類は 有価証券報告書（docTypeCode 120）と 半期報告書（160）。第1・第3四半期の明細は
扱わない（決算短信の XBRL を自動で取ってよい公式の経路が見つからない）。

実測（research/probe_edinet_api.py、run 38050003592、2026-10-10）で確かめた形
  - 鍵は Subscription-Key をクエリに付ける。書類一覧は documents.json?date=YYYY-MM-DD&type=2
    （1日ぶんが 0.2〜1.8MB、応答 中央値 1.1秒）。証券コードで引く API は無いので、日付ごとの一覧を
    溜めた索引（edinet_filings_index.json）から探す
  - 書類取得は documents/{docID}?type=5 → ZIP。中に XBRL_TO_CSV/jpcrp*.csv（UTF-16・タブ区切り、
    列は 要素ID / 項目名 / コンテキストID / 相対年度 / 連結・個別 / 期間・時点 / ユニットID / 単位 / 値）
  - 当期の損益はコンテキスト CurrentYearDuration（連結）/ CurrentYearDuration_NonConsolidatedMember（個別）。
    DEI の AccountingStandardsDEI（Japan GAAP / IFRS）と WhetherConsolidatedFinancialStatementsArePreparedDEI で
    会計基準と連結の有無が分かる
  - IFRS の会社は jpigp_cor:RevenueIFRS が無いことがある（売上収益を独自の要素で開示）。売上は要素 ID で
    見つからなければ項目名（売上収益・売上高・営業収益）で拾う

置き場所
  研究用の索引と書類ごとの控えは Release data-raw（edinet_filings_index.json / edinet_filings_cache.json）。
  画面用の filings.json は git（public/data）。EDINET の情報は公共データ利用規約（PDL1.0）で、出典を
  書けば加工して公開してよい。ログには件数だけを出す（会社名の一覧・財務の値は出さない）。

  EDINET_FSA_API_KEY=... python3 research/edinet_filings.py
  python3 research/edinet_filings.py --fake            # 偽の応答で最後まで回す（手元）
  python3 research/edinet_filings.py --offline         # 取りに行かず、控えから filings.json を作り直す
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import time
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_edinet_api as API  # noqa: E402

DATA_DIR = os.path.join(HERE, "_data")
INDEX = "edinet_filings_index.json"
CACHE = "edinet_filings_cache.json"
ROOT = os.path.dirname(HERE)
PREDICTIONS = os.path.join(ROOT, "public", "data", "predictions.json")
STOCKS = os.path.join(ROOT, "public", "data", "stocks.json")
OUT = os.path.join(ROOT, "public", "data", "filings.json")

JST = dt.timezone(dt.timedelta(hours=9))
#: 索引に入れる書類の種類（有報・半期報告書と、その訂正）
INDEX_TYPES = ("120", "130", "160", "170")
#: 画面に使う書類の種類（訂正報告書は全文が入っているとは限らないので使わない）
USE_TYPES = ("120", "160")
#: 索引の初回の遡り（暦日）。どの会社も1年に1回は有報を出すので、400日でほぼ全社の最新が入る
BACKFILL_DAYS = 400
#: 当期のコンテキスト（連結）。先にあるものを使う。半期報告書の名前は実測できていないので候補を広く持つ
CURRENT_CONTEXTS = ("CurrentYearDuration", "CurrentYTDDuration", "InterimDuration",
                    "CurrentInterimDuration", "CurrentHalfYearDuration", "CurrentQuarterDuration")
PRIOR_OF = {"CurrentYearDuration": "Prior1YearDuration", "CurrentYTDDuration": "Prior1YTDDuration",
            "InterimDuration": "Prior1InterimDuration", "CurrentInterimDuration": "Prior1InterimDuration",
            "CurrentHalfYearDuration": "Prior1HalfYearDuration", "CurrentQuarterDuration": "Prior1QuarterDuration"}
NON_CONS = "_NonConsolidatedMember"
#: 画面に持っていく要素（接頭辞なし）。損益の段と、その内訳
ITEMS = list(dict.fromkeys(API.PL_JGAAP + API.PL_IFRS + ["Revenue"]))
#: 要素 ID で売上が見つからないときに項目名で拾う（IFRS の独自要素など）。CSV の並びで先に出たもの
REVENUE_LABELS = ("売上収益", "売上高", "営業収益", "売上収益合計", "営業収益合計")
SOURCE = {"name": "EDINET（金融庁）", "url": "https://disclosure2dl.edinet-fsa.go.jp/",
          "license": "公共データ利用規約（PDL1.0）",
          "credit": "EDINET閲覧（提出）サイトの書類をもとに作成"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def today_jst() -> dt.date:
    return dt.datetime.now(JST).date()


# ---------------------------------------------------------------------- #
# 索引
# ---------------------------------------------------------------------- #

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def load_index(data_dir: str) -> dict:
    idx = load_json(os.path.join(data_dir, INDEX), {})
    idx.setdefault("dates", {})          # "YYYY-MM-DD" -> [row, ...]（120/130/160/170 で secCode のあるもの）
    return idx


def slim(r: dict) -> dict:
    return {k: r.get(k) for k in ("docID", "edinetCode", "secCode", "docTypeCode", "periodStart",
                                  "periodEnd", "submitDateTime", "csvFlag", "withdrawalStatus")}


def dates_to_fetch(idx: dict, today: dt.date, backfill_days: int) -> List[dt.date]:
    """
    まだ索引に無い平日を、新しい順に。索引が空なら today − backfill_days から。
    最後に取った日は取り直す（その日の夕方以降に出た書類を拾う）。
    """
    have = set(idx["dates"])
    start = today - dt.timedelta(days=backfill_days)
    newest = max(have) if have else None
    out = []
    d = today
    while d >= start:
        if d.weekday() < 5 and (d.isoformat() not in have or d.isoformat() == newest):
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def update_index(p: API.Probe, idx: dict, today: dt.date, backfill_days: int, reserve: int) -> int:
    """索引を今日まで埋める。使ったリクエスト数を返す。予算が reserve を切ったら途中でやめる。"""
    used0 = p.used
    todo = dates_to_fetch(idx, today, backfill_days)
    log(f"[index] 索引 {len(idx['dates'])}日ぶん / 取りに行く {len(todo)}日")
    for d in todo:
        if p.used >= p.budget - reserve:
            log(f"[index] 予算が残り {p.budget - p.used} なので索引の更新をここでやめる（続きは次回）")
            break
        st, body = p.get("/documents.json", {"date": d.isoformat(), "type": "2"})
        if st != 200 or not isinstance(body, dict):
            log(f"[index] {d} は HTTP {st}。飛ばす")
            continue
        rows = [slim(r) for r in API.results_of(body)
                if str(r.get("docTypeCode") or "") in INDEX_TYPES and r.get("secCode")]
        idx["dates"][d.isoformat()] = rows
    idx["updatedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return p.used - used0


def latest_doc(idx: dict, sec_code: str, types: Tuple[str, ...] = USE_TYPES) -> Optional[dict]:
    """索引の中で、証券コードの最新の 120 / 160（取り下げ無し・CSV あり）。"""
    best = None
    for rows in idx["dates"].values():
        for r in rows:
            if str(r.get("secCode") or "") != sec_code or str(r.get("docTypeCode") or "") not in types:
                continue
            if str(r.get("withdrawalStatus") or "0") != "0" or str(r.get("csvFlag") or "0") != "1":
                continue
            if best is None or str(r.get("submitDateTime") or "") > str(best.get("submitDateTime") or ""):
                best = r
    return best


# ---------------------------------------------------------------------- #
# 書類 → 損益の明細
# ---------------------------------------------------------------------- #

def main_csv(raw_zip: bytes) -> Optional[Tuple[str, bytes]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_zip))
    except zipfile.BadZipFile:
        return None
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    main_ = [n for n in names if os.path.basename(n).startswith("jpcrp")]
    if not main_:
        return None
    return main_[0], zf.read(main_[0])


def choose_context(rows: List[Dict[str, str]], header: List[str], consolidated: bool) -> Optional[str]:
    cc = API.col(header, "context")
    if not cc:
        return None
    present = {r.get(cc, "") for r in rows}
    for base in CURRENT_CONTEXTS:
        name = base if consolidated else base + NON_CONS
        if name in present:
            return name
    return None


def parse_statement(raw_zip: bytes, meta: dict) -> Optional[dict]:
    """
    ZIP（type=5）から画面用の1件を作る。損益の標準の要素が1本も無い（米国基準など）・当期のコンテキストが無い
    書類は unsupported（理由）を付けて返す（控えに残し、画面は「対象外」と出す）。
    値は円（CSV の値そのまま。開示は百万円などに丸めてあることがある）。
    """
    base = {"docID": meta.get("docID"), "docType": str(meta.get("docTypeCode") or ""),
            "edinetCode": meta.get("edinetCode"), "secCode": meta.get("secCode"),
            "submitDate": str(meta.get("submitDateTime") or "")[:10],
            "periodStart": meta.get("periodStart"), "periodEnd": meta.get("periodEnd"),
            "fetchedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    def unsupported(reason: str, dei: Optional[dict] = None) -> dict:
        # 読めない書類も控えに残す（同じ書類を毎晩取り直さない）。画面は「対象外」と理由を出す
        d = dei or {}
        return {**base, "unsupported": reason, "standard": d.get("AccountingStandardsDEI", ""),
                "periodType": d.get("TypeOfCurrentPeriodDEI", ""), "items": {}, "labels": {},
                "priorItems": {}, "checks": {}}

    got = main_csv(raw_zip)
    if got is None:
        return unsupported("no-csv")
    name, raw = got
    _, text = API.decode_csv(raw)
    header, rows, _ = API.parse_rows(text)
    ce, cl, cc, cv = (API.col(header, k) for k in ("element", "label", "context", "value"))
    if not (ce and cc and cv):
        return unsupported("no-columns")
    dei: Dict[str, str] = {}
    for r in rows:
        nm = API.local_name(r.get(ce, ""))
        if nm in API.DEI and nm not in dei:
            dei[nm] = r.get(cv, "")
    consolidated = dei.get("WhetherConsolidatedFinancialStatementsArePreparedDEI", "true").strip().lower() == "true"
    ctx = choose_context(rows, header, consolidated)
    if ctx is None:
        return unsupported("no-context", dei)
    prior = PRIOR_OF.get(ctx.replace(NON_CONS, ""), "") + (NON_CONS if ctx.endswith(NON_CONS) else "")

    def collect(context: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        items: Dict[str, float] = {}
        labels: Dict[str, str] = {}
        for r in rows:
            if r.get(cc, "") != context:
                continue
            el = r.get(ce, "")
            nm = API.local_name(el)
            x = API.to_number(r.get(cv, ""))
            if x is None:
                continue
            if (el.startswith("jppfs_cor:") or el.startswith("jpigp_cor:")) and nm in ITEMS and nm not in items:
                items[nm] = x
                labels[nm] = (r.get(cl, "") if cl else "")
        if not any(k in items for k in ("NetSales", "OperatingRevenue1", "RevenueIFRS", "Revenue")):
            # 要素 ID で売上が無い（IFRS の独自要素など）。項目名で先に出たものを売上にする
            for r in rows:
                if r.get(cc, "") != context or not cl:
                    continue
                lab = (r.get(cl, "") or "").strip()
                x = API.to_number(r.get(cv, ""))
                if lab in REVENUE_LABELS and x is not None and x > 0:
                    items["Revenue"] = x
                    labels["Revenue"] = lab
                    break
        return items, labels

    items, labels = collect(ctx)
    if not any(k != "Revenue" for k in items):
        # 損益の段の標準の要素が1本も無い（米国基準の会社は独自の要素で開示する）。売上だけ項目名で拾えても段は作れない
        return unsupported("no-items", dei)
    prior_items, _ = collect(prior) if prior else ({}, {})
    ids = API.IDENTITIES_IFRS if any(k.endswith("IFRS") for k in items) else API.IDENTITIES_JGAAP
    checks = {lhs: ok for lhs, ok in API.check_identities(items, ids) if ok is not None}
    submit = str(meta.get("submitDateTime") or "")[:10]
    return {
        **base,
        "edinetCode": meta.get("edinetCode") or dei.get("EDINETCodeDEI"),
        "secCode": meta.get("secCode") or dei.get("SecurityCodeDEI"),
        "periodStart": meta.get("periodStart") or dei.get("CurrentFiscalYearStartDateDEI"),
        "periodEnd": meta.get("periodEnd") or dei.get("CurrentPeriodEndDateDEI"),
        "periodType": dei.get("TypeOfCurrentPeriodDEI", ""), "standard": dei.get("AccountingStandardsDEI", ""),
        "consolidated": consolidated, "context": ctx, "csv": os.path.basename(name),
        "items": {k: int(v) if float(v).is_integer() else v for k, v in items.items()},
        "labels": labels,
        "priorItems": {k: int(v) if float(v).is_integer() else v for k, v in prior_items.items()},
        "checks": checks,
    }


# ---------------------------------------------------------------------- #
# 対象と出力
# ---------------------------------------------------------------------- #

def targets_from_public(predictions: str, stocks: str, extra: Iterable[str] = ()) -> Dict[str, str]:
    """画面で選べる銘柄（予測の候補 + 8軸の銘柄）。J-Quants コード -> 名前。"""
    out: Dict[str, str] = {}
    pred = load_json(predictions, {})
    for c in pred.get("candidates", []) or []:
        jq = str(c.get("jqCode") or "")
        if jq:
            out.setdefault(jq, c.get("name") or "")
    st = load_json(stocks, {})
    for s in st.get("stocks", []) or []:
        jq = str(s.get("jqCode") or "")
        if jq:
            out.setdefault(jq, s.get("name") or "")
    for e in extra:
        out.setdefault(str(e), "")
    return out


def build_public(cache: dict, idx: dict, targets: Dict[str, str]) -> dict:
    docs = {}
    for jq, name in sorted(targets.items()):
        d = latest_doc(idx, jq)
        if d is None:
            continue
        parsed = cache.get(str(d.get("docID")))
        if parsed:
            docs[jq] = {**parsed, "name": name}
    return {"generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": SOURCE, "docTypes": {"120": "有価証券報告書", "160": "半期報告書"},
            "docs": docs}


def run(p: Optional[API.Probe], args, today: dt.date) -> int:
    idx = load_index(args.data_dir)
    cache = load_json(os.path.join(args.data_dir, CACHE), {})
    targets = targets_from_public(args.predictions, args.stocks, args.codes or [])
    log(f"[target] 画面の銘柄 {len(targets)}件 / 控え {len(cache)}書類")
    if p is not None:
        update_index(p, idx, today, args.backfill_days, reserve=args.reserve)
        save_json(os.path.join(args.data_dir, INDEX), idx)
        n_days = len(idx["dates"])
        n_rows = sum(len(v) for v in idx["dates"].values())
        log(f"[index] {n_days}日 / {n_rows:,}件（{min(idx['dates']) if n_days else '-'}〜{max(idx['dates']) if n_days else '-'}）")
        got = new = miss = fail = 0
        for jq in sorted(targets):
            d = latest_doc(idx, jq)
            if d is None:
                miss += 1
                continue
            doc_id = str(d.get("docID"))
            if doc_id in cache:
                got += 1
                continue
            if p.used >= p.budget:
                log("[fetch] 予算を使い切ったので残りは次回")
                break
            st, body = p.get(f"/documents/{doc_id}", {"type": "5"})
            if st != 200 or not isinstance(body, (bytes, bytearray)):
                fail += 1
                continue
            parsed = parse_statement(bytes(body), d)
            if parsed.get("unsupported"):
                fail += 1
                log(f"[fetch] 対象外の書類が1件（docTypeCode {d.get('docTypeCode')} / {parsed['unsupported']} / "
                    f"会計基準 {parsed.get('standard') or '不明'}）")
            cache[doc_id] = parsed
            new += 1
        save_json(os.path.join(args.data_dir, CACHE), cache)
        log(f"[fetch] 控えにあった {got} / 新しく取った {new} / 索引に無い {miss} / 対象外 {fail} "
            f"/ リクエスト {p.used} / 予算 {p.budget}")
    pub = build_public(cache, idx, targets)
    save_json(args.out, pub)
    std = {}
    for d in pub["docs"].values():
        std[d.get("standard") or "?"] = std.get(d.get("standard") or "?", 0) + 1
    ok = sum(1 for d in pub["docs"].values() if d["checks"] and all(d["checks"].values()))
    uns = sum(1 for d in pub["docs"].values() if d.get("unsupported"))
    log(f"[out] {args.out}: {len(pub['docs'])}銘柄（会計基準 {std} / 恒等式が全部成立 {ok} / 対象外 {uns}）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="候補銘柄の有報・半期報告書の損益を EDINET から取る")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--predictions", default=PREDICTIONS)
    ap.add_argument("--stocks", default=STOCKS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--codes", nargs="*", help="追加で取る J-Quants コード（5桁）")
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--reserve", type=int, default=60, help="索引の更新で残しておく、書類取得ぶんのリクエスト")
    ap.add_argument("--backfill-days", type=int, default=BACKFILL_DAYS)
    ap.add_argument("--today", default=None)
    ap.add_argument("--offline", action="store_true", help="取りに行かず、控えから出力だけ作る")
    ap.add_argument("--fake", action="store_true", help="偽の応答で回す（手元の確認）")
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.today) if args.today else today_jst()
    if args.offline:
        p = None
    elif args.fake:
        p = API.Probe("TESTONLY-fake-key", args.max_requests, transport=API.fake_transport(today))
    else:
        key = os.environ.get(API.KEY_ENV, "").strip() or None
        if not key:
            print(f"[stop] {API.KEY_ENV} が無い")
            return 1
        p = API.Probe(key, args.max_requests)
    return run(p, args, today)


if __name__ == "__main__":
    raise SystemExit(main())
