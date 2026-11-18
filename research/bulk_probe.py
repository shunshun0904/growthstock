#!/usr/bin/env python3
"""
J-Quants の一括ダウンロード（/bulk/list・/bulk/get）のファイルを、取り込む前に測る。**何も保存しない。**

  python3 research/bulk_probe.py --endpoint /equities/trades --summary-json bulk_probe.json

1. 一覧: 件数・期間・大きさ・作られた時刻（種類（historical / live など）ごと）
2. 1ファイル（既定はいちばん小さいもの）を落として、見出し・行数・日数・銘柄数・時刻の範囲・
   列ごとの形（数字だけか・小数の桁・整数か）・キーの一意性を数える
3. 保存の形（parquet・zstd 9）にしたときの大きさを、型のそろえ方3通りで見積もる

株価・出来高の値そのものは出さない（public リポジトリの Actions のログは誰でも読める）。

2026-11-18、ティック（/equities/trades。CSV でしか提供されず、API では取れない）の取り込みを
決める前に、ファイルの数・大きさ・1日の行数を実測するために作った（docs/MINUTE_BARS.md の分足の確認と同じ考え）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import io
import json
import os
import statistics
import sys
import tempfile
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.csv as pcsv  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import minute_bars as MB  # noqa: E402
from ticks import time_to_us  # noqa: E402,F401  （時刻の変換はティックの取り込みと同じものを使う）
from jquants_data_fetcher import JQuantsError, resolve_api_key  # noqa: E402

#: 数字だけでも数値にしない列（識別子・区分。先頭の 0 に意味がある）
ID_COLUMNS = {"Code", "TransactionId", "SessionDistinction"}
_HMS = r"^\d{2}:\d{2}:\d{2}(\.\d{1,6})?$"


def kind_of(key: str) -> str:
    """Key の種類（equities/trades/historical/2025/... -> historical）。"""
    parts = str(key).split("/")
    for p in parts:
        if p in ("historical", "live", "daily", "monthly"):
            return p
    return parts[-2] if len(parts) >= 2 else ""


def summarize_listing(entries: List[dict]) -> dict:
    """一覧を種類ごとに数える（件数・期間・大きさ・作られた時刻（JST の時刻））。"""
    out: Dict[str, dict] = {}
    groups: Dict[str, List[dict]] = collections.defaultdict(list)
    for e in entries:
        groups[kind_of(e["key"])].append(e)
    for k, es in sorted(groups.items()):
        sizes = [e["size"] for e in es if e["size"] is not None]
        periods = [e["period"] for e in es if e["period"] is not None]
        hours = []
        for e in es:
            lm = e.get("lastModified")
            try:
                t = dt.datetime.fromisoformat(str(lm).replace("Z", "+00:00")).astimezone(MB.JST)
                hours.append(t.strftime("%H:%M"))
            except ValueError:
                pass
        out[k] = {
            "files": len(es),
            "from": min(p[0] for p in periods).isoformat() if periods else None,
            "to": max(p[1] for p in periods).isoformat() if periods else None,
            "bytes": int(sum(sizes)),
            "sizeMin": min(sizes) if sizes else None,
            "sizeMedian": int(statistics.median(sizes)) if sizes else None,
            "sizeMax": max(sizes) if sizes else None,
            "periodKinds": sorted({("day" if p[0] == p[1] else "month") for p in periods}),
            "madeAtJst": {"earliest": min(hours), "latest": max(hours),
                          "median": sorted(hours)[len(hours) // 2]} if hours else None,
            "examples": [e["key"] for e in es[:2]] + ([es[-1]["key"]] if len(es) > 2 else []),
        }
    return out


def gunzip_size(path: str) -> int:
    """展開した大きさ（gzip の終わりの検査も兼ねる。欠けていれば例外）。"""
    if not MB._is_gzip(path):
        return os.path.getsize(path)
    n = 0
    with gzip.open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                return n
            n += len(b)


def column_profile(col: pa.ChunkedArray) -> dict:
    """文字列の列の形（値は出さない）: 欠測・長さ・数字だけか・数値か・小数の桁・整数か。"""
    n = len(col)
    nulls = col.null_count
    lens = pc.utf8_length(col)
    prof = {"nulls": int(nulls), "minLen": pc.min(lens).as_py(), "maxLen": pc.max(lens).as_py()}
    nonnull = n - nulls
    if nonnull == 0:
        return prof
    digits = pc.sum(pc.cast(pc.fill_null(pc.match_substring_regex(col, r"^\d+$"), False), pa.int64())).as_py()
    numeric = pc.fill_null(pc.match_substring_regex(col, r"^-?\d+(\.\d+)?$"), False)
    n_numeric = pc.sum(pc.cast(numeric, pa.int64())).as_py()
    prof["digitsOnly"] = digits == nonnull
    prof["numeric"] = n_numeric == nonnull
    if prof["numeric"]:
        dec = pc.utf8_length(pc.replace_substring_regex(col, r"^-?\d+\.?", ""))
        prof["maxDecimals"] = pc.max(dec).as_py()
        # 小数部がすべて 0 なら整数
        frac_nonzero = pc.fill_null(pc.match_substring_regex(col, r"\.\d*[1-9]"), False)
        prof["integerValued"] = not pc.any(frac_nonzero).as_py()
    return prof


def typed(table: pa.Table, transaction_as_int: bool) -> pa.Table:
    """保存に使う型にそろえる（見積もり用）。"""
    cols = {}
    for name in table.column_names:
        col = table.column(name)
        if name == "Date":
            cols[name] = pc.cast(pc.strptime(col, format="%Y-%m-%d", unit="s"), pa.date32())
        elif name == "Time" and pc.all(pc.fill_null(pc.match_substring_regex(col, _HMS), False)).as_py():
            cols[name] = time_to_us(col)
        elif name == "TransactionId" and transaction_as_int and \
                pc.all(pc.fill_null(pc.match_substring_regex(col, r"^\d{1,18}$"), False)).as_py():
            cols[name] = pc.cast(col, pa.int64())
        elif name in ID_COLUMNS:
            cols[name] = col
        else:
            prof_numeric = pc.all(pc.fill_null(pc.match_substring_regex(col, r"^-?\d+(\.\d+)?$"), True)).as_py()
            if prof_numeric:
                as_f = pc.cast(col, pa.float64())
                integral = not pc.any(pc.fill_null(pc.match_substring_regex(col, r"\.\d*[1-9]"), False)).as_py()
                cols[name] = pc.cast(as_f, pa.int64()) if integral and name.endswith("Volume") else as_f
            else:
                cols[name] = col
    out = pa.table(cols)
    keys = [k for k in ("Date", "Code", "TransactionId", "Time") if k in out.column_names]
    return out.sort_by([(k, "ascending") for k in keys]) if keys else out


def parquet_bytes(table: pa.Table) -> int:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", compression_level=9, row_group_size=1_000_000)
    return len(buf.getvalue())


def profile_file(path: str) -> dict:
    """1ファイルを測る（値は出さない）。"""
    t0 = time.time()
    out: dict = {"gzip": MB._is_gzip(path), "bytes": os.path.getsize(path)}
    out["uncompressedBytes"] = gunzip_size(path)
    with MB._open_text(path) as fh:
        header = next(csv.reader([fh.readline()]), [])
    out["header"] = header
    table = pcsv.read_csv(MB._arrow_input(path), read_options=pcsv.ReadOptions(block_size=64 << 20),
                          convert_options=pcsv.ConvertOptions(
                              column_types={n: pa.string() for n in header}, strings_can_be_null=True))
    out["rows"] = table.num_rows
    out["readSeconds"] = round(time.time() - t0, 1)
    out["columns"] = {n: column_profile(table.column(n)) for n in header}
    if "Date" in header:
        days = pc.unique(table.column("Date")).to_pylist()
        out["days"] = len(days)
        out["dayFrom"], out["dayTo"] = (min(days), max(days)) if days else (None, None)
        if days:
            g = table.group_by("Date").aggregate([([], "count_all")])
            out["rowsPerDay"] = {"min": pc.min(g["count_all"]).as_py(), "max": pc.max(g["count_all"]).as_py(),
                                 "median": int(statistics.median(g["count_all"].to_pylist()))}
    if "Code" in header:
        out["codes"] = len(pc.unique(table.column("Code")))
    if "Time" in header:
        out["timeFirst"] = pc.min(table.column("Time")).as_py()
        out["timeLast"] = pc.max(table.column("Time")).as_py()
    for cat in ("SessionDistinction",):
        if cat in header:
            vals = pc.unique(table.column(cat)).to_pylist()
            out[f"{cat}Values"] = sorted(str(v) for v in vals)[:20]
    key = [k for k in ("Date", "Code", "TransactionId") if k in header]
    if len(key) == 3:
        g = table.group_by(key).aggregate([([], "count_all")])
        out["keyUnique"] = {"key": key, "duplicateGroups": int(pc.sum(pc.cast(
            pc.greater(g["count_all"], 1), pa.int64())).as_py() or 0)}
    est = {}
    t1 = time.time()
    est["allStrings"] = parquet_bytes(table)
    try:
        est["typed"] = parquet_bytes(typed(table, transaction_as_int=False))
        est["typedTransactionIdInt"] = parquet_bytes(typed(table, transaction_as_int=True))
    except (ValueError, pa.ArrowException) as exc:
        est["typedError"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    est["seconds"] = round(time.time() - t1, 1)
    out["parquetEstimate"] = est
    out["seconds"] = round(time.time() - t0, 1)
    return out


def pick(entries: List[dict], how: str, max_bytes: int) -> Optional[dict]:
    cands = [e for e in entries if e["size"] and e["size"] <= max_bytes]
    if not cands:
        return None
    if how == "largest":
        return max(cands, key=lambda e: e["size"])
    if how == "latest":
        return max(cands, key=lambda e: (e["period"] or (dt.date.min, dt.date.min))[1])
    return min(cands, key=lambda e: e["size"])


def main(argv: Optional[List[str]] = None, api: Optional[MB.Api] = None) -> int:
    ap = argparse.ArgumentParser(description="J-Quants の一括ファイルを取り込む前に測る（何も保存しない）")
    ap.add_argument("--endpoint", action="append", default=None,
                    help="測るエンドポイント（繰り返し可）。既定 /equities/trades")
    ap.add_argument("--sample", choices=["smallest", "latest", "largest"], default="smallest",
                    help="落として測る1ファイルの選び方")
    ap.add_argument("--max-sample-mb", type=float, default=3000.0,
                    help="これより大きいファイルは落とさない")
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    endpoints = args.endpoint or ["/equities/trades"]
    api = api or MB.Api(resolve_api_key())
    report: dict = {"atJst": dt.datetime.now(MB.JST).strftime("%Y-%m-%d %H:%M"), "endpoints": {}}
    code = 0
    for ep in endpoints:
        rec: dict = {}
        report["endpoints"][ep] = rec
        try:
            entries = MB.list_bulk(api, ep)
        except JQuantsError as exc:
            rec["error"] = f"{type(exc).__name__}: {api.redact(str(exc))[:300]}"
            print(f"[{ep}] 一覧を引けなかった: {rec['error']}")
            code = 1
            continue
        rec["listed"] = len(entries)
        rec["byKind"] = summarize_listing(entries)
        rec["totalBytes"] = int(sum(e["size"] or 0 for e in entries))
        print(f"[{ep}] 一覧 {len(entries)}件 / 計 {rec['totalBytes']/1e9:.2f}GB")
        print(json.dumps(rec["byKind"], ensure_ascii=False, indent=1))
        target = pick(entries, args.sample, int(args.max_sample_mb * 1e6))
        if target is None:
            print(f"[{ep}] 落として測れるファイルが無い（{args.max_sample_mb:.0f}MB 以下）")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.csv.gz")
            try:
                MB.download_bulk(api, target["key"], path, target["size"])
                prof = profile_file(path)
            except (JQuantsError, ValueError, OSError, EOFError, pa.ArrowException) as exc:
                prof = {"error": f"{type(exc).__name__}: {api.redact(str(exc))[:300]}"}
                code = 1
        prof["key"] = target["key"]
        rec["sample"] = prof
        print(f"[{ep}] 1ファイルの形: " + json.dumps(prof, ensure_ascii=False, default=str))
    report["requests"] = api.requests
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[done] リクエスト {api.requests}回")
    return code


if __name__ == "__main__":
    sys.exit(main())
