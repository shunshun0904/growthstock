#!/usr/bin/env python3
"""
J-Quants のティック（約定。/equities/trades。アドオン「株価 分足・ティック」）を取得して蓄積する。

  python3 research/ticks.py --data-dir research/_tick \\
      --upload-cmd "bash scripts/gh_release_upload.sh" \\
      --ensure-cmd "bash scripts/gh_release_ensure.sh"

ティックは CSV の一括ダウンロードでしか提供されない（API では取れない）。全銘柄が1ファイルに入っていて、
銘柄だけを選んでは落とせない。J-Quants が持つのは直近2年分で、毎日1日ずつ古い日が取れなくなる。
取った日は消さない（運用者の依頼 2026-11-18「過去分のデータの取得、差分更新、データの蓄積」と、
選択「全銘柄」「分足と同じ置き場」）。

実測（2026-11-18、Probe Bulk Files run 44283243735。docs/TICKS.md）
  一覧 36件・計 21.0GB（CSV.gz）
    historical/YYYY/equities_trades_YYYYMM.csv.gz   月ごと 24件（2024-11〜2026-10）、1件 645MB〜1.16GB
    live/equities_trades_YYYYMMDD.csv.gz            今月の日ごと、1件 51〜103MB、その日の 16:44 JST ごろにできる
  1日 約514万行・4,197銘柄（2026-11-02）。parquet にすると1日 約33MB

置き場（分足と同じく、このリポジトリの**下書き（draft）の Release**。書き込み権限のある人にしか見えない。
公開しない。2026-11-19 に確かめた。docs/MINUTE_BARS.md「置き場」）
  Release data-tick        取得記録 tick_manifest.json だけ
  Release data-tick-YYYY   その年の日ごとのファイル tick_YYYYMMDD.parquet（ensure_cmd が下書きで作る）
1つの Release に置けるアセットは 1,000 個までなので、年ごとに分ける（日ごとで年 約245ファイル）。
月ごとの1ファイルにしないのは、毎日の差分で月のファイル（500MB 超）を落として書き直して上げ直さずに済むため。

流れ
  1. 一覧（/bulk/list?endpoint=/equities/trades）。まだ取り込んでいない・LastModified が変わったファイルを、
     月の古い順・同じ月の中は作られた順に取り込む（同じ日は後から当てた側に置き換わる）
  2. 1ファイルずつ: 署名付きURLで落とす → gzip の終わりまで確かめる → 64MB ずつ読んで型をそろえ、日ごとの
     途中のファイルに書き分ける（月ごとのファイルは1億行を超えるので、まるごとメモリに載せない）
     → 日ごとに (Code, TransactionId) で並べて tick_YYYYMMDD.parquet にする（TransactionId の順が約定の順）
  3. 年ごとの Release に上げてから、取得記録を上げる（**データ → 記録の順**）。上げた日のファイルは手元から消す
     （2年分で 13GB を超え、ランナーのディスクに収まらない）

ログは公開（public リポジトリの Actions）。件数・日付・ファイル名・大きさだけを出し、価格・数量の値や
署名付きURLは出さない。
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.csv as pcsv  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import minute_bars as MB  # noqa: E402
import minute_store as S  # noqa: E402
from jquants_data_fetcher import JQuantsError, resolve_api_key  # noqa: E402

DATA_DIR = os.path.join(ROOT, "research", "_tick")
ENDPOINT = "/equities/trades"
MANIFEST_NAME = "tick_manifest.json"
MANIFEST_TAG = "data-tick"
SCHEMA_VERSION = 1

#: 仕様の項目（すべて Required）。https://jpx-jquants.com/ja/spec/eq-trades の写し（運用者が貼った仕様）
COLUMNS = ["Date", "Code", "Time", "SessionDistinction", "Price", "TradingVolume", "TransactionId"]
#: 保存の型。Time は HH:MM:SS.ffffff をマイクロ秒の時刻に。SessionDistinction（01/02）と TransactionId（12桁）は
#: 先頭の 0 に意味があるので文字列のまま（整数にしても大きさは変わらなかった。Probe Bulk Files の見積もり）
SCHEMA = pa.schema([
    ("Date", pa.date32()), ("Code", pa.string()), ("Time", pa.time64("us")),
    ("SessionDistinction", pa.string()), ("Price", pa.float64()),
    ("TradingVolume", pa.float64()), ("TransactionId", pa.string()),
])
#: 同じ日・同じ銘柄の中で一意で、昇順が約定の順（仕様）
KEY = ["Date", "Code", "TransactionId"]
WRITE_OPTS = {"compression": "zstd", "compression_level": 9}
ROW_GROUP_SIZE = 1_000_000
#: CSV を読む1回ぶん（展開後）。月ごとのファイルは展開すると 4〜7GB あるので少しずつ読む
BLOCK_SIZE = 64 << 20
_HMS = r"^\d{2}:\d{2}:\d{2}(\.\d{1,6})?$"


def year_tag(day: dt.date) -> str:
    return f"{MANIFEST_TAG}-{day.year}"


def day_file(day: dt.date) -> str:
    return f"tick_{day:%Y%m%d}.parquet"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# 取得記録
# --------------------------------------------------------------------------- #

def empty_manifest() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "endpoint": ENDPOINT,
        #: 日 -> {rows, codes, bytes, release, key, at}
        "days": {},
        #: 一括ファイルの Key -> {lastModified, size, days, rows, ingestedAt}
        "bulk": {},
        #: 仕様に無い列（保存はしない。見つけた日を残す）
        "extraColumns": {},
    }


def load_manifest(data_dir: str) -> dict:
    p = os.path.join(data_dir, MANIFEST_NAME)
    m = empty_manifest()
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8") as fh:
        got = json.load(fh)
    if got.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"{MANIFEST_NAME} の形式 {got.get('schema')} は読めない")
    m.update(got)
    for k in ("days", "bulk", "extraColumns"):
        if not isinstance(m.get(k), dict):
            m[k] = {}
    return m


def summarize(manifest: dict) -> dict:
    days = sorted(manifest.get("days", {}))
    vals = manifest.get("days", {}).values()
    return {
        "days": len(days),
        "from": days[0] if days else None,
        "to": days[-1] if days else None,
        "rows": int(sum(int(v.get("rows", 0)) for v in vals)),
        "bytes": int(sum(int(v.get("bytes", 0)) for v in manifest.get("days", {}).values())),
        "releases": sorted({v.get("release") for v in manifest.get("days", {}).values() if v.get("release")}),
    }


def save_manifest(data_dir: str, manifest: dict) -> str:
    os.makedirs(data_dir, exist_ok=True)
    manifest["schema"] = SCHEMA_VERSION
    manifest["endpoint"] = ENDPOINT
    manifest["days"] = dict(sorted(manifest.get("days", {}).items()))
    manifest["bulk"] = dict(sorted(manifest.get("bulk", {}).items()))
    manifest["summary"] = summarize(manifest)
    manifest["updatedAt"] = _utc_now()
    p = os.path.join(data_dir, MANIFEST_NAME)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# 形をそろえる
# --------------------------------------------------------------------------- #

def time_to_us(arr) -> pa.ChunkedArray:
    """"HH:MM:SS(.ffffff)" -> time64[us]。形の違う行があれば ValueError（pyarrow 18 は文字列から直接は変えられない）。"""
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    s = pc.utf8_trim_whitespace(arr)
    ok = pc.match_substring_regex(s, _HMS)
    bad = pc.sum(pc.cast(pc.invert(pc.fill_null(ok, False)), pa.int64())).as_py() or 0
    if bad:
        raise ValueError(f"時刻が HH:MM:SS(.ffffff) でない行が {bad}行")
    h = pc.cast(pc.utf8_slice_codeunits(s, 0, 2), pa.int64())
    m = pc.cast(pc.utf8_slice_codeunits(s, 3, 5), pa.int64())
    sec = pc.cast(pc.utf8_slice_codeunits(s, 6, 8), pa.int64())
    frac = pc.utf8_rpad(pc.utf8_slice_codeunits(s, 9, 15), width=6, padding="0")
    us = pc.add(pc.multiply(pc.add(pc.multiply(pc.add(pc.multiply(h, 60), m), 60), sec), 1_000_000),
                pc.cast(frac, pa.int64()))
    return pc.cast(us, pa.time64("us"))


def normalize(table: pa.Table) -> Tuple[pa.Table, List[str]]:
    """
    CSV の1かたまりを保存の形（SCHEMA）にそろえる。(表, 仕様に無い列)。
    仕様の列が1つでも無ければ止める。キーの列（Date・Code・Time・TransactionId）に欠測があれば止める。
    """
    if table.num_rows == 0 and not table.column_names:
        return SCHEMA.empty_table(), []
    picked: Dict[str, int] = {}
    extra: List[str] = []
    for i, name in enumerate(table.column_names):
        n = str(name).strip().lstrip("﻿")
        if n in COLUMNS:
            picked.setdefault(n, i)
        else:
            extra.append(str(name))
    missing = [c for c in COLUMNS if c not in picked]
    if missing:
        raise ValueError(f"ティックの列が足りない: {missing}（来た列: {table.column_names[:20]}）")

    def col(c: str):
        return table.column(picked[c])

    arrays = [
        S._to_date32(col("Date")),
        S._as_string(col("Code")),
        time_to_us(S._as_string(col("Time"))),
        S._as_string(col("SessionDistinction")),
        S._to_float(col("Price")),
        S._to_float(col("TradingVolume")),
        S._as_string(col("TransactionId")),
    ]
    out = pa.Table.from_arrays(arrays, schema=SCHEMA)
    nulls = {c: out.column(c).null_count for c in ("Date", "Code", "Time", "TransactionId")}
    if any(nulls.values()):
        raise ValueError(f"キーの列に欠測がある: {nulls}")
    return out, extra


def sort_day(table: pa.Table) -> pa.Table:
    """
    1日ぶんを (Code, TransactionId) で並べ、(Date, Code, TransactionId) の重なりを見る。
    まったく同じ行の重なりは1行に、中身が違えば止める（仕様では一意）。
    TransactionId の桁がそろっていない日は、桁をそろえた写しで並べる（文字列の順と数の順を合わせる）。
    """
    if table.num_rows == 0:
        return table
    tid = table["TransactionId"]
    lens = pc.utf8_length(tid)
    width = pc.max(lens).as_py()
    if pc.min(lens).as_py() != width:
        key = pc.utf8_lpad(tid, width=width, padding="0")
        t = table.append_column("_k", key).sort_by([("Code", "ascending"), ("_k", "ascending")]).drop_columns(["_k"])
    else:
        t = table.sort_by([("Code", "ascending"), ("TransactionId", "ascending")])
    n = t.num_rows
    if n < 2:
        return t
    code = t["Code"].combine_chunks()
    tid = t["TransactionId"].combine_chunks()
    same = pc.and_(pc.equal(code.slice(1), code.slice(0, n - 1)),
                   pc.equal(tid.slice(1), tid.slice(0, n - 1)))
    if not pc.any(same).as_py():
        return t
    uniq = t.group_by(COLUMNS).aggregate([])
    counts = uniq.group_by(KEY).aggregate([([], "count_all")])
    clash = pc.sum(pc.cast(pc.greater(counts["count_all"], 1), pa.int64())).as_py() or 0
    if clash:
        raise ValueError(f"(Date, Code, TransactionId) が同じで中身の違う行が {clash}組ある。保存しない")
    return sort_day(uniq.select(COLUMNS).cast(SCHEMA))


class DaySplitter:
    """一括ファイルを少しずつ読みながら、日ごとの途中のファイル（並べ替え前）に書き分ける。"""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self.writers: Dict[dt.date, pq.ParquetWriter] = {}
        self.rows: Dict[dt.date, int] = collections.Counter()

    def path(self, day: dt.date) -> str:
        return os.path.join(self.work_dir, f"part_{day:%Y%m%d}.parquet")

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        days = pc.unique(table["Date"]).to_pylist()
        for d in days:
            part = table if len(days) == 1 else table.filter(pc.equal(table["Date"], pa.scalar(d, pa.date32())))
            w = self.writers.get(d)
            if w is None:
                w = pq.ParquetWriter(self.path(d), SCHEMA, compression="zstd", compression_level=1)
                self.writers[d] = w
            w.write_table(part)
            self.rows[d] += part.num_rows

    def close(self) -> Dict[dt.date, str]:
        for w in self.writers.values():
            w.close()
        return {d: self.path(d) for d in sorted(self.writers)}


def read_header(path: str) -> List[str]:
    with MB._open_text(path) as fh:
        return next(csv.reader([fh.readline()]), [])


def split_file(path: str, work_dir: str) -> Tuple[Dict[dt.date, str], List[str], int]:
    """一括ファイルを日ごとの途中のファイルに分ける。(日 -> 途中のファイル, 仕様外の列, 行数)。"""
    header = read_header(path)
    if not header:
        return {}, [], 0
    reader = pcsv.open_csv(
        MB._arrow_input(path),
        read_options=pcsv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pcsv.ConvertOptions(column_types={n: pa.string() for n in header},
                                            strings_can_be_null=True))
    splitter = DaySplitter(work_dir)
    extra: set = set()
    rows = 0
    try:
        for batch in reader:
            t, ex = normalize(pa.Table.from_batches([batch]))
            extra.update(ex)
            splitter.add(t)
            rows += t.num_rows
    finally:
        parts = splitter.close()
    return parts, sorted(extra), rows


def finalize_day(part_path: str, out_path: str) -> Dict[str, int]:
    """途中のファイルを並べて、日のファイルにする。{rows, codes, bytes}。"""
    t = sort_day(pq.read_table(part_path).cast(SCHEMA))
    tmp = out_path + ".tmp"
    pq.write_table(t, tmp, row_group_size=ROW_GROUP_SIZE, **WRITE_OPTS)
    os.replace(tmp, out_path)
    return {"rows": int(t.num_rows), "codes": int(pc.count_distinct(t["Code"]).as_py()),
            "bytes": int(os.path.getsize(out_path))}


# --------------------------------------------------------------------------- #
# 保存先とのやり取り
# --------------------------------------------------------------------------- #

class Store:
    """
    日ごとのファイルを年ごとの Release に上げ、取得記録を data-tick に上げる。
    upload_cmd は「TAG FILE...」を足して呼ぶ（scripts/gh_release_upload.sh）。ensure_cmd は「TAG TITLE NOTES」を
    足して呼び、Release が無ければ作る（scripts/gh_release_ensure.sh）。upload_cmd が無ければ手元に残すだけ。
    """

    def __init__(self, data_dir: str, upload_cmd: Optional[str] = None,
                 ensure_cmd: Optional[str] = None, keep_local: bool = False) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.manifest = load_manifest(data_dir)
        self.upload = shlex.split(upload_cmd) if upload_cmd else None
        self.ensure = shlex.split(ensure_cmd) if ensure_cmd else None
        self.keep_local = keep_local or self.upload is None
        self.ensured: set = set()
        self.uploaded_files = 0
        #: 取得記録を保存してから、まだ上げていない
        self.unflushed = False

    def save(self) -> str:
        p = save_manifest(self.data_dir, self.manifest)
        self.unflushed = True
        return p

    def _ensure(self, tag: str) -> None:
        if self.ensure is None or tag in self.ensured:
            return
        notes = ("J-Quants API V2 のティック（/equities/trades）の日ごとのファイル（tick_YYYYMMDD.parquet）。"
                 if tag != MANIFEST_TAG else "J-Quants のティックの取得記録（tick_manifest.json）。")
        subprocess.run(self.ensure + [tag, f"J-Quants ティック {tag}（下書きのまま。公開しない）",
                                      notes + " J-Quants のデータは第三者に配信できないので、下書きのまま使う（公開しない）。"
                                      "growthstock の Update Ticks が書く（docs/TICKS.md）。"],
                       check=True)
        self.ensured.add(tag)

    def put_days(self, paths: Dict[dt.date, str]) -> None:
        """日のファイルを年ごとの Release に上げる。上げたら手元から消す（keep_local でなければ）。"""
        if self.upload is None:
            return
        by_tag: Dict[str, List[str]] = collections.defaultdict(list)
        for d, p in sorted(paths.items()):
            by_tag[year_tag(d)].append(p)
        for tag, files in sorted(by_tag.items()):
            self._ensure(tag)
            subprocess.run(self.upload + [tag] + files, check=True)
            self.uploaded_files += len(files)
            print(f"[store] {tag} に {len(files)}ファイルを上げた", flush=True)
        if not self.keep_local:
            for p in paths.values():
                if os.path.exists(p):
                    os.remove(p)

    def flush_manifest(self) -> None:
        if self.upload is None or not self.unflushed:
            return
        self._ensure(MANIFEST_TAG)
        subprocess.run(self.upload + [MANIFEST_TAG, os.path.join(self.data_dir, MANIFEST_NAME)], check=True)
        self.unflushed = False
        print("[store] 取得記録を上げた", flush=True)


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, api: MB.Api, store: Store, deadline: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.api = api
        self.store = store
        self.deadline = deadline
        self.clock = clock
        self.report: Dict[str, object] = {"errors": [], "warnings": []}

    def ingest(self, entry: dict) -> Dict[str, Dict[str, int]]:
        """1ファイルを取り込む（落とす → 日ごとに分ける → 並べる → 上げる → 記録）。日 -> 統計。"""
        d = self.store.data_dir
        tmp = os.path.join(d, ".download.csv.gz")
        work = tempfile.mkdtemp(prefix=".work_", dir=d)
        made: Dict[dt.date, str] = {}
        try:
            MB.download_bulk(self.api, entry["key"], tmp, entry["size"])
            MB.check_gzip(tmp)
            parts, extra, rows = split_file(tmp, work)
            os.remove(tmp)
            new = [c for c in extra if c not in self.store.manifest["extraColumns"]]
            for c in new:
                self.store.manifest["extraColumns"][c] = dt.date.today().isoformat()
            if new:
                msg = f"仕様に無い列 {new} が来た（保存していない。docs/TICKS.md）"
                print(f"::warning title=ティックに仕様外の列::{msg}")
                self.report["warnings"].append(msg)
            stats: Dict[str, Dict[str, int]] = {}
            for day, part in parts.items():
                out = os.path.join(d, day_file(day))
                stats[day.isoformat()] = finalize_day(part, out)
                os.remove(part)
                made[day] = out
            self.store.put_days(made)
            now = _utc_now()
            for day in made:
                st = stats[day.isoformat()]
                self.store.manifest["days"][day.isoformat()] = {
                    **st, "release": year_tag(day), "key": entry["key"], "at": now}
            self.store.manifest["bulk"][entry["key"]] = {
                "lastModified": entry["lastModified"], "size": entry["size"],
                "days": len(stats), "rows": int(sum(s["rows"] for s in stats.values())),
                "ingestedAt": now}
            self.store.save()
            self.store.flush_manifest()
            return stats
        finally:
            for p in (tmp, tmp + ".part"):
                if os.path.exists(p):
                    os.remove(p)
            shutil.rmtree(work, ignore_errors=True)
            if not self.store.keep_local:
                for p in made.values():
                    if os.path.exists(p):
                        os.remove(p)

    def bulk(self, start: Optional[dt.date]) -> None:
        entries = MB.list_bulk(self.api, ENDPOINT)
        dated = [e for e in entries if e["period"] is not None]
        unknown = [e["key"] for e in entries if e["period"] is None]
        size = sum(e["size"] or 0 for e in dated)
        rng = f"{dated[0]['period'][0]} 〜 {dated[-1]['period'][1]}" if dated else "なし"
        print(f"[bulk] 一覧 {len(entries)}件（期間 {rng}、計 {size/1e9:.2f}GB）")
        if unknown:
            print(f"[bulk] 期間を読めない Key {len(unknown)}件は取り込まない: {unknown[:3]}")
        todo = MB.plan_bulk(entries, self.store.manifest, start)
        print(f"[bulk] 取り込む {len(todo)}件（{sum(e['size'] or 0 for e in todo)/1e9:.2f}GB。"
              f"まだ取り込んでいない・更新されたもの）", flush=True)
        done = failed = days = 0
        for e in todo:
            if self.clock() > self.deadline:
                print("[bulk] 時間切れ。残りは次の実行で取る")
                break
            name = e["key"].rsplit("/", 1)[-1]
            t0 = time.time()
            try:
                stats = self.ingest(e)
            except (JQuantsError, ValueError, OSError, EOFError, pa.ArrowException) as exc:
                failed += 1
                msg = f"{name}: {type(exc).__name__}: {self.api.redact(str(exc))[:200]}"
                print(f"::error title=ティックの一括ファイルを取り込めない::{msg}")
                self.report["errors"].append(msg)
                continue
            done += 1
            days += len(stats)
            ds = sorted(stats)
            rows = sum(s["rows"] for s in stats.values())
            nbytes = sum(s["bytes"] for s in stats.values())
            span = f"{ds[0]} 〜 {ds[-1]}" if ds else "なし"
            print(f"[bulk] {name}: {(e['size'] or 0)/1e6:.0f}MB → {len(stats)}日（{span}）/ {rows:,}行 / "
                  f"parquet {nbytes/1e6:.0f}MB / {time.time()-t0:.0f}秒", flush=True)
        self.report["bulk"] = {
            "listed": len(entries), "planned": len(todo), "ingested": done, "failed": failed,
            "days": days, "listedFrom": dated[0]["period"][0].isoformat() if dated else None,
            "listedTo": dated[-1]["period"][1].isoformat() if dated else None, "listedBytes": size,
            "plannedBytes": sum(e["size"] or 0 for e in todo)}


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="J-Quants のティックを取得して蓄積する（一括ダウンロードだけ）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--from", dest="date_from", default=None,
                    help="YYYY-MM-DD。この日より前で終わる一括ファイルは取り込まない（既定は全部）")
    ap.add_argument("--probe-only", action="store_true", help="一覧と取り込む予定だけ（何も落とさない・保存しない）")
    ap.add_argument("--forget-from", default=None, metavar="YYYY-MM-DD",
                    help="この日以降の取得記録を外し、その日を含む一括ファイルを取り込み直す")
    ap.add_argument("--reingest-bulk", action="store_true", help="一括ファイルを全部取り込み直す")
    ap.add_argument("--max-minutes", type=float, default=300.0,
                    help="この分数を過ぎたら新しいファイルに手を付けない（ジョブの上限より短く）")
    ap.add_argument("--upload-cmd", default=None, help="保存先へ上げるコマンド（TAG FILE... を足して呼ぶ）")
    ap.add_argument("--ensure-cmd", default=None, help="Release が無ければ作るコマンド（TAG TITLE NOTES を足して呼ぶ）")
    ap.add_argument("--keep-local", action="store_true", help="上げた日のファイルを手元に残す")
    ap.add_argument("--summary-json", default=None)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None, api: Optional[MB.Api] = None,
         today: Optional[dt.date] = None) -> int:
    args = parse_args(argv)
    t_start = time.monotonic()
    today = today or MB.today_jst()
    started = dt.datetime.now(MB.JST)
    store = Store(args.data_dir, args.upload_cmd, args.ensure_cmd, args.keep_local)
    print("[manifest] 取得済み: " + json.dumps(summarize(store.manifest), ensure_ascii=False))
    if args.forget_from:
        since = dt.date.fromisoformat(args.forget_from)
        dropped = [d for d in sorted(store.manifest["days"]) if dt.date.fromisoformat(d) >= since]
        keys = {store.manifest["days"][d].get("key") for d in dropped}
        for d in dropped:
            store.manifest["days"].pop(d)
        for k in keys:
            store.manifest["bulk"].pop(k, None)          # その日を含むファイルを取り込み直させる
        print(f"[forget] 取得記録 {len(dropped)}日と、その元の一括ファイル {len(keys)}件の記録を外す")
    if args.reingest_bulk:
        print(f"[reingest] 一括ファイルの取り込みの記録 {len(store.manifest['bulk'])}件を外す")
        store.manifest["bulk"] = {}

    api = api or MB.Api(resolve_api_key())
    run = Run(api, store, t_start + args.max_minutes * 60)
    report = run.report
    report.update({"mode": "probe" if args.probe_only else "update",
                   "startedJst": started.strftime("%Y-%m-%d %H:%M")})
    floor = MB.retention_start(today)
    start = dt.date.fromisoformat(args.date_from) if args.date_from else None
    code = 0
    try:
        if args.probe_only:
            try:
                entries = MB.list_bulk(api, ENDPOINT)
            except MB.PlanError as exc:
                print("::error title=ティックのアドオンが有効でない::" + " ".join(str(exc).split())[:300])
                report["errors"].append("アドオンが有効でない")
                return 2
            todo = MB.plan_bulk(entries, store.manifest, start)
            dated = [e for e in entries if e["period"] is not None]
            report["bulk"] = {"listed": len(entries), "planned": len(todo),
                              "listedFrom": dated[0]["period"][0].isoformat() if dated else None,
                              "listedTo": dated[-1]["period"][1].isoformat() if dated else None,
                              "listedBytes": sum(e["size"] or 0 for e in dated),
                              "plannedBytes": sum(e["size"] or 0 for e in todo)}
            print(f"[bulk] 一覧 {len(entries)}件 / 取り込む予定 {len(todo)}件"
                  f"（{report['bulk']['plannedBytes']/1e9:.2f}GB）")
            for e in todo[:40]:
                print(f"    {e['key']}  {(e['size'] or 0)/1e6:.0f}MB  {e['lastModified']}")
            return 0

        try:
            run.bulk(start)
        except MB.PlanError as exc:
            print("::error title=ティックのアドオンが有効でない::" + " ".join(str(exc).split())[:300])
            report["errors"].append("アドオンが有効でない（HTTP 403 not available on your subscription）")
            return 2

        days = MB.trading_days(api, floor, today)
        report["tradingDays"] = len(days)
        have = set(store.manifest["days"])
        missing = [d.isoformat() for d in days if d.isoformat() not in have]
        waiting = [d for d in missing if d == today.isoformat()]
        others = [d for d in missing if d != today.isoformat()]
        report["missingInWindow"] = missing
        print(f"\n[check] 範囲 {floor} 〜 {today} の営業日 {len(days)}日のうち保存済み {len(days) - len(missing)}日")
        if waiting:
            print(f"[check] 当日 {today} はまだ無い（{started:%H:%M} JST 時点。日ごとのファイルは 16:44 JST ごろにできる）")
        if others:
            print(f"::warning title=ティックの未取得の日::{len(others)}日（{others[:5]}）。一括ファイルができ次第取る")
        before = [d for d in store.manifest["days"] if d < floor.isoformat()]
        report["beyondWindowDays"] = len(before)
        if before:
            print(f"[check] J-Quants の範囲より前（{floor} より前）の保存済み {len(before)}日（ここにしか無い）")
        if report["errors"]:
            code = 1
    finally:
        report["requests"] = api.requests
        report["minutes"] = round((time.monotonic() - t_start) / 60, 1)
        report["store"] = summarize(store.manifest)
        report["uploadedFiles"] = store.uploaded_files
        report["extraColumns"] = store.manifest.get("extraColumns", {})
        if args.summary_json:
            with open(args.summary_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        print("\n[manifest] 更新後: " + json.dumps(summarize(store.manifest), ensure_ascii=False))
        print(f"[done] リクエスト {api.requests}回 / {report['minutes']}分")
    return code


if __name__ == "__main__":
    sys.exit(main())
