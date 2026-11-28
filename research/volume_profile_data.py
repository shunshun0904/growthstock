"""価格帯別出来高の入力を読む。

- ティック: 下書きの Release data-tick-YYYY の tick_YYYYMMDD.parquet（docs/TICKS.md）。取得記録 tick_manifest.json は data-tick
- 分足: 下書きの Release data-minute の minute_YYYYMM.parquet と minute_manifest.json（docs/MINUTE_BARS.md）
- 日足（照合用の調整前の値と AdjFactor）: Release data-raw の bars_YYYY.parquet（Update Data Store が書く）

置き場からの取り出しは、ワークフローが渡す取り出しのコマンド（bash scripts/gh_release_download.sh）を
TAG DIR NAME の形で呼ぶ。ログに出すのはファイル名と件数だけ（中身の値は出さない）。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from volume_profile import Quote, Session, minute_session, tick_session

TICK_MANIFEST = ("data-tick", "tick_manifest.json")
MINUTE_MANIFEST = ("data-minute", "minute_manifest.json")
MINUTE_TAG = "data-minute"
RAW_TAG = "data-raw"
TICK_COLUMNS = ["Date", "Code", "Time", "SessionDistinction", "Price", "TradingVolume", "TransactionId"]
MINUTE_COLUMNS = ["Date", "Time", "Code", "L", "H", "Vo", "Va"]
BAR_COLUMNS = ["Date", "Code", "L", "H", "C", "Vo", "Va", "AdjFactor"]


def tick_file(day: str) -> str:
    return f"tick_{day.replace('-', '')}.parquet"


def minute_file(day: str) -> str:
    return f"minute_{day[:4]}{day[5:7]}.parquet"


def bars_file(year: int) -> str:
    return f"bars_{year}.parquet"


def load_manifest(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("days"), dict):
        raise ValueError(f"{os.path.basename(path)} の形式が想定と違う")
    return data


def stored_days(manifest: Optional[dict]) -> Set[str]:
    return set((manifest or {}).get("days", {}))


def plan_assets(days: List[str], tick_manifest: Optional[dict], minute_manifest: Optional[dict]
                ) -> Dict[str, List[Tuple[str, str]]]:
    """取り出すファイル（種別 -> [(Release のタグ, ファイル名)]）。置き場に無い日は入れない（その日は欠損になる）。"""
    ticks = []
    for day in days:
        entry = (tick_manifest or {}).get("days", {}).get(day)
        if entry is not None:
            ticks.append((entry.get("release") or f"data-tick-{day[:4]}", tick_file(day)))
    minutes = sorted({minute_file(day) for day in days if day in stored_days(minute_manifest)})
    bars = sorted({bars_file(int(day[:4])) for day in days})
    return {"tick": ticks, "minute": [(MINUTE_TAG, name) for name in minutes],
            "bars": [(RAW_TAG, name) for name in bars]}


class Fetcher:
    """置き場から手元へ取り出す。cmd が無ければ手元にあるものだけを使う（テスト・手元の確認）。"""

    def __init__(self, cmd: Optional[str]):
        self.cmd = shlex.split(cmd) if cmd else None
        self.fetched = 0

    def get(self, tag: str, directory: str, name: str, required: bool = True) -> Optional[str]:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
        if self.cmd:
            os.makedirs(directory, exist_ok=True)
            done = subprocess.run([*self.cmd, tag, directory, name], check=False)
            if done.returncode != 0:
                raise RuntimeError(f"{tag} の {name} を取り出せなかった（exit {done.returncode}）")
            self.fetched += 1
        if os.path.exists(path):
            return path
        if required:
            raise RuntimeError(f"{tag} に {name} が無い")
        return None


def _optional(value) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _read_filtered(path: str, columns: List[str], codes: List[str]) -> pa.Table:
    try:
        return pq.read_table(path, columns=columns, filters=[("Code", "in", codes)])
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
        # Code の型が違う古いファイルでも読めるように、全部読んでから絞る
        table = pq.read_table(path, columns=columns)
        code = pc.cast(table["Code"], pa.string())
        return table.set_column(table.schema.get_field_index("Code"), "Code", code).filter(
            pc.is_in(code, value_set=pa.array(codes)))


def _iso_dates(column: pa.ChunkedArray) -> np.ndarray:
    values = column.to_pandas()
    import pandas as pd

    return pd.to_datetime(values).dt.strftime("%Y-%m-%d").to_numpy()


def load_quotes(paths: Iterable[str], codes: List[str], days: Iterable[str],
                conflicts: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, Dict[str, Quote]]:
    """
    日足（調整前）を 銘柄 -> 日 -> Quote に。AdjFactor が空の行は 1（data_store.py と同じ）。
    同じ銘柄・日に違う行があれば、どちらが正しいか分からないのでその日を外し（その日を含む期間は日足の欠損になる）、
    conflicts に (銘柄, 日) を足す。
    """
    wanted = set(days)
    quotes: Dict[str, Dict[str, Quote]] = defaultdict(dict)
    clashes: Set[Tuple[str, str]] = set()
    for path in paths:
        names = set(pq.read_schema(path).names)
        missing = [c for c in BAR_COLUMNS if c not in names and c != "AdjFactor"]
        if missing:
            raise ValueError(f"{os.path.basename(path)} に日足の列 {missing} が無い")
        table = _read_filtered(path, [c for c in BAR_COLUMNS if c in names], codes)
        if not table.num_rows:
            continue
        frame = table.to_pandas()
        frame["Date"] = _iso_dates(table["Date"])
        frame["Code"] = frame["Code"].astype(str).str.strip()
        frame = frame[frame["Date"].isin(wanted)]
        for row in frame.itertuples(index=False):
            factor = _optional(getattr(row, "AdjFactor", None))
            quote = Quote(_optional(row.L), _optional(row.H), _optional(row.C), _optional(row.Vo),
                          _optional(row.Va), 1.0 if factor is None else factor)
            if row.Date in quotes[row.Code] and quotes[row.Code][row.Date] != quote:
                clashes.add((row.Code, row.Date))
            quotes[row.Code][row.Date] = quote
    for code, day in clashes:
        quotes[code].pop(day, None)
    if conflicts is not None:
        conflicts.update(clashes)
    return dict(quotes)


def _groups(codes: np.ndarray):
    """並べ替え済みの銘柄の列から、銘柄ごとの範囲を返す。"""
    if not len(codes):
        return
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for start, end in zip(starts, ends):
        yield str(codes[start]), start, end


def load_tick_day(path: str, day: str, codes: List[str]) -> Dict[str, Session]:
    """1日のティックのファイルから、対象の銘柄の Session を作る。"""
    table = _read_filtered(path, TICK_COLUMNS, codes)
    if not table.num_rows:
        return {}
    dates = set(pc.unique(table["Date"]).to_pylist())
    if dates != {dt.date.fromisoformat(day)}:
        raise ValueError(f"{os.path.basename(path)} に {day} 以外の日の行がある")
    code = table["Code"].to_numpy(zero_copy_only=False).astype(str)
    txid = table["TransactionId"].to_numpy(zero_copy_only=False).astype(str)
    try:
        key = txid.astype(np.int64)
    except ValueError:
        key = txid
    # 約定の順は TransactionId の昇順（仕様）
    order = np.lexsort((key, code))
    code = code[order]
    time_us = table["Time"].cast(pa.int64()).to_numpy()[order]
    session = table["SessionDistinction"].to_numpy(zero_copy_only=False).astype(str)[order]
    price = table["Price"].to_numpy()[order]
    shares = table["TradingVolume"].to_numpy()[order]
    when = dt.date.fromisoformat(day)
    return {c: tick_session(when, time_us[a:b], session[a:b], price[a:b], shares[a:b])
            for c, a, b in _groups(code)}


def load_minute_month(path: str, days: List[str], codes: List[str]) -> Dict[str, Dict[str, Session]]:
    """月の分足のファイルから、対象の日・銘柄の Session を作る（銘柄 -> 日 -> Session）。"""
    table = _read_filtered(path, MINUTE_COLUMNS, codes)
    wanted = pa.array([dt.date.fromisoformat(d) for d in days], pa.date32())
    table = table.filter(pc.is_in(table["Date"], value_set=wanted))
    out: Dict[str, Dict[str, Session]] = defaultdict(dict)
    if not table.num_rows:
        return dict(out)
    date = np.array([d.isoformat() for d in table["Date"].to_pylist()])
    code = table["Code"].to_numpy(zero_copy_only=False).astype(str)
    times = table["Time"].to_numpy(zero_copy_only=False).astype(str)
    order = np.lexsort((times, code, date))
    date, code, times = date[order], code[order], times[order]
    low, high, volume, turnover = (table[c].to_numpy()[order] for c in ("L", "H", "Vo", "Va"))
    key = np.char.add(np.char.add(date, "|"), code)
    for group, a, b in _groups(key):
        day, c = group.split("|")
        out[c][day] = minute_session(dt.date.fromisoformat(day), times[a:b], volume[a:b],
                                     turnover[a:b], low[a:b], high[a:b])
    return dict(out)
