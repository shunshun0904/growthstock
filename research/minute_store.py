#!/usr/bin/env python3
"""
分足（J-Quants の /equities/bars/minute）の保存。月ごとの parquet と取得記録。

置き場
----
**非公開**リポジトリの Release（既定は <owner>/growthstock-minute のタグ data-minute）。
このリポジトリは公開で、Release に置いたファイルは誰でもダウンロードできる。J-Quants の
個人の利用は私的利用に限られ、データの第三者への配信は禁止とされているので、分足は
公開の側に置かない（運用者の選択 2026-11-18。docs/MINUTE_BARS.md）。

ファイル
------
  minute_YYYYMM.parquet   その月の全銘柄の分足。並びは (Date, Code, Time)
  minute_manifest.json    取得記録。どの日を・どこから（一括の CSV / API）取ったか、
                          ファイルごとの行数・日数・大きさ、一括ファイルの更新日時

月ごとにしたのは、GitHub の Release のアセットが1リリースあたり 1,000 個までのため
（日ごとでは年に約245個で、4年ほどで埋まる）。1ファイル 2GiB 未満の制限にも収まる。

置き換えの単位は「日」。一括の CSV も API（date= の指定）も、その日の全銘柄ぶんが
そろって返るので、同じ日の行は丸ごと新しい側で置き換え、行単位では混ぜない。

取得記録は保存先にある月のファイルの目録を兼ねる（files）。保存先にあるはずの
月のファイルが手元に無いまま書き込むと、その月を新しい日だけで作り直し、保存先の
1か月ぶんを上書きしてしまう。書き込む前に取り寄せ（fetch_file）、それでも無ければ止める。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MANIFEST_NAME = "minute_manifest.json"
SCHEMA_VERSION = 1
ENDPOINT = "/equities/bars/minute"

#: 仕様の項目（すべて Required）。https://jpx-jquants.com/ja/spec/eq-bars-minute
COLUMNS = ["Date", "Time", "Code", "O", "H", "L", "C", "Vo", "Va"]
NUM_COLS = ["O", "H", "L", "C", "Vo", "Va"]
SCHEMA = pa.schema(
    [("Date", pa.date32()), ("Time", pa.string()), ("Code", pa.string())]
    + [(c, pa.float64()) for c in NUM_COLS]
)
ROW_KEY = ["Date", "Code", "Time"]
#: 月のファイルの並び。日ごとに (Code, Time) で並べて日の順に書く（MinuteStore._merge_month）
SORT_KEYS = [("Date", "ascending"), ("Code", "ascending"), ("Time", "ascending")]

#: 見出しの別名。一括の CSV の見出しが API の項目名と違ったときのため（V1 の長い名前と、
#: 仕様の日本語の項目名）。大文字小文字の違いは別に吸収する。実際の見出しは --probe-only で確かめる
ALIASES = {"Open": "O", "High": "H", "Low": "L", "Close": "C",
           "Volume": "Vo", "TurnoverValue": "Va",
           "日付": "Date", "時刻": "Time", "銘柄コード": "Code",
           "始値": "O", "高値": "H", "安値": "L", "終値": "C", "出来高": "Vo", "売買代金": "Va"}

FILE_RE = re.compile(r"^minute_(\d{6})\.parquet$")

#: zstd の 9。合成データ（1日約45万行）で 3 より約13%小さく、1日ぶんの書き出しは約1秒
WRITE_OPTS = {"compression": "zstd", "compression_level": 9}
ROW_GROUP_SIZE = 1_000_000


# --------------------------------------------------------------------------- #
# 名前
# --------------------------------------------------------------------------- #

def month_of(day: dt.date) -> str:
    return f"{day.year:04d}{day.month:02d}"


def file_name(ym: str) -> str:
    return f"minute_{ym}.parquet"


def file_path(data_dir: str, ym: str) -> str:
    return os.path.join(data_dir, file_name(ym))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# 取得記録
# --------------------------------------------------------------------------- #

def empty_manifest() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "endpoint": ENDPOINT,
        #: 日 -> {rows, codes, source(bulk/api/file), at, key?}
        "days": {},
        #: 一括ファイルの Key -> {lastModified, size, rows, days, ingestedAt}
        "bulk": {},
        #: 保存先にある月のファイル -> {rows, days, from, to, bytes}
        "files": {},
        #: 仕様に無い列（保存はしない。見つけた日を残し、気づけるようにする）
        "extraColumns": {},
    }


def manifest_path(data_dir: str) -> str:
    return os.path.join(data_dir, MANIFEST_NAME)


def load_manifest(data_dir: str) -> dict:
    p = manifest_path(data_dir)
    m = empty_manifest()
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8") as fh:
        got = json.load(fh)
    if got.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"{MANIFEST_NAME} の形式 {got.get('schema')} は読めない（{SCHEMA_VERSION} のみ）")
    m.update(got)
    for k in ("days", "bulk", "files", "extraColumns"):
        if not isinstance(m.get(k), dict):
            m[k] = {}
    return m


def summarize(manifest: dict) -> dict:
    days = sorted(manifest.get("days", {}))
    files = manifest.get("files", {})
    return {
        "days": len(days),
        "from": days[0] if days else None,
        "to": days[-1] if days else None,
        "rows": int(sum(int(v.get("rows", 0)) for v in files.values())),
        "bytes": int(sum(int(v.get("bytes", 0)) for v in files.values())),
        "files": len(files),
    }


def save_manifest(data_dir: str, manifest: dict) -> str:
    """書き途中のファイルを残さないよう、別名に書いてから置き換える。"""
    os.makedirs(data_dir, exist_ok=True)
    manifest["schema"] = SCHEMA_VERSION
    manifest["endpoint"] = ENDPOINT
    manifest["days"] = dict(sorted(manifest.get("days", {}).items()))
    manifest["files"] = dict(sorted(manifest.get("files", {}).items()))
    manifest["bulk"] = dict(sorted(manifest.get("bulk", {}).items()))
    manifest["summary"] = summarize(manifest)
    manifest["updatedAt"] = _utc_now()
    p = manifest_path(data_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, p)
    return p


def stored_days(manifest: dict) -> Set[str]:
    return set(manifest.get("days", {}))


def forget_days(manifest: dict, since: Optional[dt.date] = None) -> List[str]:
    """
    取得記録だけを外す（月のファイルは消さない）。外した日は次の取得で取り直し、
    同じ日の行は置き換わる。取り直せなければ（2年の範囲の外など）ファイルはそのまま。
    """
    dropped = []
    for d in sorted(manifest.get("days", {})):
        if since is None or dt.date.fromisoformat(d) >= since:
            dropped.append(d)
    for d in dropped:
        manifest["days"].pop(d, None)
    return dropped


# --------------------------------------------------------------------------- #
# 表の形をそろえる
# --------------------------------------------------------------------------- #

def canonical_name(name: str) -> str:
    """見出しを保存の項目名に寄せる。知らない名前はそのまま返す。"""
    n = str(name).strip().lstrip("﻿")
    if n in COLUMNS:
        return n
    if n in ALIASES:
        return ALIASES[n]
    low = n.lower()
    for c in COLUMNS:
        if c.lower() == low:
            return c
    for a, c in ALIASES.items():
        if a.lower() == low:
            return c
    return n


def _as_string(arr) -> pa.ChunkedArray:
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return pc.utf8_trim_whitespace(pc.cast(arr, pa.string()))
    if pa.types.is_null(arr.type):
        return pc.cast(arr, pa.string())
    if pa.types.is_floating(arr.type):
        # 数値で来た銘柄コード（72030.0）は小数点なしの文字列に
        as_int = pc.cast(arr, pa.int64(), safe=False)
        return pc.cast(as_int, pa.string())
    return pc.cast(arr, pa.string())


def _to_date32(arr) -> pa.ChunkedArray:
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    t = arr.type
    if pa.types.is_date32(t):
        return arr
    if pa.types.is_date64(t) or pa.types.is_timestamp(t):
        return pc.cast(arr, pa.date32())
    s = _as_string(arr)
    s = pc.replace_substring_regex(s, r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3")
    s = pc.replace_substring(s, "/", "-")
    s = pc.replace_substring_regex(s, r"^(\d{4}-\d{2}-\d{2})[T ].*$", r"\1")
    try:
        ts = pc.strptime(s, format="%Y-%m-%d", unit="s", error_is_null=False)
    except pa.ArrowInvalid as exc:
        raise ValueError(f"Date を日付として読めない: {str(exc)[:160]}") from exc
    return pc.cast(ts, pa.date32())


def _to_hhmm(arr) -> pa.ChunkedArray:
    s = _as_string(arr)
    s = pc.replace_substring_regex(s, r"^(\d{1,2}):(\d{2})(:\d{2}(\.\d+)?)?$", r"\1:\2")
    s = pc.replace_substring_regex(s, r"^(\d{2})(\d{2})$", r"\1:\2")
    s = pc.replace_substring_regex(s, r"^(\d):", r"0\1:")
    ok = pc.match_substring_regex(s, r"^\d{2}:\d{2}$")
    bad = pc.sum(pc.cast(pc.invert(pc.fill_null(ok, False)), pa.int64())).as_py() or 0
    if bad:
        raise ValueError(f"Time が HH:MM の形でない行が {bad}行ある")
    return s


def _to_float(arr) -> pa.ChunkedArray:
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    t = arr.type
    if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_null(t) \
            or pa.types.is_decimal(t):
        return pc.cast(arr, pa.float64())
    s = _as_string(arr)
    # 欠測の記号（空・ハイフン）は欠測に。負の数（"-1.5"）は残る
    blank = pc.match_substring_regex(s, r"^[\-－—]?$")
    s = pc.if_else(pc.fill_null(blank, False), pa.scalar(None, pa.string()), s)
    try:
        return pc.cast(s, pa.float64())
    except pa.ArrowInvalid as exc:
        raise ValueError(f"数値に直せない値がある: {str(exc)[:160]}") from exc


def normalize(table: pa.Table) -> Tuple[pa.Table, List[str]]:
    """
    API の応答・一括の CSV のどちらも、保存の形（SCHEMA）にそろえる。

    返り値は (表, 仕様に無い列の名前)。**仕様の列が1つでも無ければ止める**（全項目が
    Required。欠けたまま保存すると気づけない）。仕様に無い列は保存しないが、名前を返し、
    呼び出し側が取得記録と実行の画面に残す（黙って捨てない）。
    """
    if table.num_rows == 0 and not table.column_names:
        return SCHEMA.empty_table(), []
    picked: Dict[str, int] = {}
    extra: List[str] = []
    for i, name in enumerate(table.column_names):
        c = canonical_name(name)
        if c in COLUMNS:
            picked.setdefault(c, i)
        else:
            extra.append(str(name))
    missing = [c for c in COLUMNS if c not in picked]
    if missing:
        raise ValueError(f"分足の列が足りない: {missing}（来た列: {table.column_names[:20]}）")
    arrays = [
        _to_date32(table.column(picked["Date"])),
        _to_hhmm(table.column(picked["Time"])),
        _as_string(table.column(picked["Code"])),
    ] + [_to_float(table.column(picked[c])) for c in NUM_COLS]
    out = pa.Table.from_arrays(arrays, schema=SCHEMA)
    nulls = {c: out.column(c).null_count for c in ("Date", "Time", "Code")}
    if any(nulls.values()):
        raise ValueError(f"キーの列に欠測がある: {nulls}")
    return out, extra


def records_to_table(records: List[dict]) -> pa.Table:
    """API の data（dict の並び）を表にする。値の型は normalize でそろえる。"""
    if not records:
        return pa.table({})
    names: List[str] = []
    seen = set()
    for r in records[:50]:
        for k in r:
            if k not in seen:
                seen.add(k)
                names.append(k)
    cols = {}
    for k in names:
        vals = [r.get(k) for r in records]
        try:
            cols[k] = pa.array(vals)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            cols[k] = pa.array([None if v is None else str(v) for v in vals], pa.string())
    return pa.table(cols)


def dedupe(table: pa.Table) -> pa.Table:
    """
    同じ (Date, Code, Time) の行が重なるとき、まったく同じ行は1行にする。
    **中身が違う重なりがあれば止める**（キーの決め方が誤っていて、別々の行を潰すことになる）。
    """
    if table.num_rows == 0:
        return table
    counts = table.group_by(ROW_KEY).aggregate([([], "count_all")])
    if pc.max(counts["count_all"]).as_py() <= 1:
        return table
    uniq = table.group_by(COLUMNS).aggregate([])
    counts = uniq.group_by(ROW_KEY).aggregate([([], "count_all")])
    clash = pc.sum(pc.cast(pc.greater(counts["count_all"], 1), pa.int64())).as_py() or 0
    if clash:
        raise ValueError(f"(Date, Code, Time) が同じで中身の違う行が {clash}組ある。保存しない")
    return uniq.select(COLUMNS).cast(SCHEMA)


def sort_day(table: pa.Table) -> pa.Table:
    """
    1日ぶんを (Code, Time) で並べ、重なりを見る。重なりが無ければ並べた表をそのまま返し、
    あれば dedupe（まったく同じ行は1行に、中身が違えば止める）。

    並べた後の隣どうしを比べるだけなので、月まるごとの group_by よりメモリが小さい。
    """
    t = table.sort_by([("Code", "ascending"), ("Time", "ascending")])
    n = t.num_rows
    if n < 2:
        return t
    code = t["Code"].combine_chunks()
    tm = t["Time"].combine_chunks()
    same = pc.and_(pc.equal(code.slice(1), code.slice(0, n - 1)),
                   pc.equal(tm.slice(1), tm.slice(0, n - 1)))
    if not pc.any(same).as_py():
        return t
    return dedupe(t).sort_by([("Code", "ascending"), ("Time", "ascending")])


def day_stats(table: pa.Table) -> Dict[str, Dict[str, int]]:
    """日 -> {rows, codes}。"""
    if table.num_rows == 0:
        return {}
    g = table.group_by("Date").aggregate([("Code", "count_distinct"), ([], "count_all")])
    out = {}
    for d, codes, rows in zip(g["Date"].to_pylist(), g["Code_count_distinct"].to_pylist(),
                              g["count_all"].to_pylist()):
        out[d.isoformat()] = {"rows": int(rows), "codes": int(codes)}
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# 月のファイル
# --------------------------------------------------------------------------- #

class MinuteStore:
    """
    月ごとのファイルへの書き込みと取得記録をまとめて持つ。

    fetch_file(name) は保存先から月のファイルを data_dir に取り寄せる関数（無ければ
    何もしない）。取得記録の files にある月を、手元に無いまま書き換えないために使う。
    """

    def __init__(self, data_dir: str, manifest: Optional[dict] = None,
                 fetch_file: Optional[Callable[[str], None]] = None) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.manifest = manifest if manifest is not None else load_manifest(data_dir)
        self.fetch_file = fetch_file
        #: この実行で書いた月のファイルの名前（保存先へ上げる対象）
        self.dirty: Set[str] = set()
        #: 取得記録を保存してから、まだ保存先へ上げていない
        self.unflushed = False

    # -- 手元にそろえる ------------------------------------------------------ #

    def ensure_local(self, ym: str) -> Optional[str]:
        """保存先にあるはずの月のファイルを手元にそろえて path を返す。無いはずなら None。"""
        name = file_name(ym)
        path = file_path(self.data_dir, ym)
        if os.path.exists(path):
            return path
        if name not in self.manifest["files"]:
            return None
        if self.fetch_file is not None:
            self.fetch_file(name)
        if not os.path.exists(path):
            raise RuntimeError(
                f"{name} は保存先にあるはず（取得記録の files）なのに手元に無い。"
                f"新しい日だけで作り直して保存先の1か月ぶんを上書きしないよう止める")
        self.check_against_manifest(ym)
        return path

    def check_against_manifest(self, ym: str) -> List[str]:
        """
        手元の月のファイルと取得記録の日を突き合わせ、記録を**ファイルに合わせる**。

        記録にあってファイルに無い日は記録から外す（2年の範囲の中なら次の取得で取り直す）。
        ファイルにあって記録に無い日は source=file で足す。直した日の一覧を返す。
        データを上げてから記録を上げる順なので、ふつうは食い違わない。
        """
        path = file_path(self.data_dir, ym)
        if not os.path.exists(path):
            return []
        got = day_stats(pq.read_table(path, columns=["Date", "Code"]))
        fixed = []
        days = self.manifest["days"]
        for d in [d for d in days if d[:7].replace("-", "") == ym]:
            if d not in got:
                days.pop(d)
                fixed.append(d)
        for d, st in got.items():
            cur = days.get(d)
            if cur is None:
                days[d] = {**st, "source": "file", "at": _utc_now()}
                fixed.append(d)
            elif int(cur.get("rows", -1)) != st["rows"]:
                cur.update(st)
                fixed.append(d)
        if fixed or file_name(ym) not in self.manifest["files"]:
            self._record_file(ym, path, sum(s["rows"] for s in got.values()),
                              [dt.date.fromisoformat(d) for d in got])
        return sorted(set(fixed))

    def check_local(self) -> Dict[str, List[str]]:
        """
        手元にある月のファイルを全部、取得記録と突き合わせる（check_against_manifest）。
        記録の files に無いファイルは、保存先にも無いかもしれないので上げる対象にする
        （記録だけが「ある」と言い、保存先に無い状態を作らない）。月 -> 直した日。
        """
        out: Dict[str, List[str]] = {}
        for ym in local_months(self.data_dir):
            if file_name(ym) not in self.manifest["files"]:
                self.dirty.add(file_name(ym))
            fixed = self.check_against_manifest(ym)
            if fixed:
                out[ym] = fixed
        return out

    # -- 書き込み ------------------------------------------------------------ #

    def write_days(self, table: pa.Table, source: str,
                   key: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        table（normalize 済み）の日を、月のファイルの中で丸ごと置き換える。
        取得記録の days も書き換える。返り値は 日 -> {rows, codes}。
        """
        if table.num_rows == 0:
            return {}
        table = table.cast(SCHEMA)
        by_month: Dict[str, List[dt.date]] = {}
        for d in sorted(pc.unique(table["Date"]).to_pylist()):
            by_month.setdefault(month_of(d), []).append(d)
        stats: Dict[str, Dict[str, int]] = {}
        for ym, days in by_month.items():
            stats.update(self._merge_month(ym, table, days))
        now = _utc_now()
        for d, st in stats.items():
            entry = {**st, "source": source, "at": now}
            if key:
                entry["key"] = key
            self.manifest["days"][d] = entry
        return dict(sorted(stats.items()))

    def _merge_month(self, ym: str, table: pa.Table, new_days: List[dt.date]) -> Dict[str, Dict[str, int]]:
        """
        月のファイルを日の順に書き直す。new_days の日は table から、それ以外の日は今のファイルから。

        日ごとに並べて書き出すので、月まるごとを並べ替えない（実データ1か月で千万行を超えうる。
        合成データ約950万行の月で、まるごと並べると 4.7GB だった）。日ごとに行グループが分かれ、
        日で絞って読むときに要る行グループだけを読める。
        """
        import pyarrow.dataset as ds

        path = self.ensure_local(ym)
        old = ds.dataset(path, format="parquet") if path else None
        old_days = (pc.unique(old.to_table(columns=["Date"])["Date"]).to_pylist()
                    if old is not None else [])
        new_set = set(new_days)
        out = file_path(self.data_dir, ym)
        tmp = out + ".tmp"
        stats: Dict[str, Dict[str, int]] = {}
        rows = 0
        all_days = sorted(set(old_days) | new_set)
        with pq.ParquetWriter(tmp, SCHEMA, **WRITE_OPTS) as w:
            for d in all_days:
                if d in new_set:
                    part = sort_day(table.filter(pc.equal(table["Date"], pa.scalar(d, pa.date32()))))
                    codes = pc.count_distinct(part["Code"]).as_py()
                    stats[d.isoformat()] = {"rows": int(part.num_rows), "codes": int(codes)}
                else:
                    part = old.to_table(filter=ds.field("Date") == pa.scalar(d, pa.date32()))
                    part = part.select(COLUMNS).cast(SCHEMA)
                w.write_table(part, row_group_size=ROW_GROUP_SIZE)
                rows += part.num_rows
        os.replace(tmp, out)
        self._record_file(ym, out, rows, all_days)
        self.dirty.add(file_name(ym))
        return stats

    def _record_file(self, ym: str, path: str, rows: int, days: List[dt.date]) -> None:
        days = sorted(days)
        self.manifest["files"][file_name(ym)] = {
            "rows": int(rows),
            "days": len(days),
            "from": days[0].isoformat() if days else None,
            "to": days[-1].isoformat() if days else None,
            "bytes": int(os.path.getsize(path)),
        }

    # -- 一括ファイル・知らない列 ---------------------------------------------- #

    def record_bulk(self, key: str, last_modified: Optional[str], size: Optional[int],
                    stats: Dict[str, Dict[str, int]]) -> None:
        self.manifest["bulk"][key] = {
            "lastModified": last_modified,
            "size": size,
            "rows": int(sum(s["rows"] for s in stats.values())),
            "days": len(stats),
            "ingestedAt": _utc_now(),
        }

    def record_extra_columns(self, names: Iterable[str]) -> List[str]:
        """初めて見た仕様外の列だけを返す。"""
        new = []
        for n in names:
            if n not in self.manifest["extraColumns"]:
                self.manifest["extraColumns"][n] = dt.date.today().isoformat()
                new.append(n)
        return new

    def save(self) -> str:
        path = save_manifest(self.data_dir, self.manifest)
        self.unflushed = True
        return path


def local_months(data_dir: str) -> List[str]:
    out = []
    for n in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        m = FILE_RE.match(n)
        if m:
            out.append(m.group(1))
    return out


def read_minutes(data_dir: str, start: Optional[dt.date] = None,
                 end: Optional[dt.date] = None, codes: Optional[Iterable[str]] = None,
                 columns: Optional[List[str]] = None) -> pa.Table:
    """
    手元の月のファイルから読む（分析用）。期間・銘柄で絞れる。

      >>> t = read_minutes("research/_minute", dt.date(2026, 6, 1), dt.date(2026, 6, 30), ["86970"])
      >>> t.to_pandas()
    """
    import pyarrow.dataset as ds

    months = local_months(data_dir)
    if start is not None:
        months = [m for m in months if m >= month_of(start)]
    if end is not None:
        months = [m for m in months if m <= month_of(end)]
    if not months:
        return SCHEMA.empty_table() if columns is None else SCHEMA.empty_table().select(columns)
    dataset = ds.dataset([file_path(data_dir, m) for m in months], format="parquet", schema=SCHEMA)
    flt = None

    def _and(a, b):
        return b if a is None else (a & b)

    if start is not None:
        flt = _and(flt, ds.field("Date") >= pa.scalar(start, pa.date32()))
    if end is not None:
        flt = _and(flt, ds.field("Date") <= pa.scalar(end, pa.date32()))
    if codes is not None:
        cs = [str(c) + "0" if len(str(c)) == 4 else str(c) for c in codes]
        flt = _and(flt, ds.field("Code").isin(cs))
    return dataset.to_table(columns=columns, filter=flt)
