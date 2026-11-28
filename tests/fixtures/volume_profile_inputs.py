"""価格帯別出来高のテスト用に、置き場と同じ形の合成データを書く（実データではない）。

  root/calendar.csv                取引カレンダー（Date, HolDiv）
  root/raw/bars_2025.parquet       日足（調整前の値と AdjFactor。data-raw と同じ列）
  root/tick/tick_manifest.json     ティックの取得記録と、日ごとの tick_YYYYMMDD.parquet（123A0 だけ）
  root/minute/minute_manifest.json 分足の取得記録と、月ごとの minute_YYYYMM.parquet（2銘柄）

ティックは 寄り（9:00 の板寄せ。同じ時刻・価格の2件）・前場のザラバ・前引け（11:30）・後場寄り（12:30）・
後場のザラバ・引け（15:30）を持つ。分足はそのティックを1分ごとにまとめたもの（99990 は分足だけ）。
"""

from __future__ import annotations

import json
import sys
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import ticks as T  # noqa: E402
import minute_store as S  # noqa: E402

CODES = {"123A0": 1340, "99990": 1810}
HOLIDAYS = {"2025-10-13", "2025-11-03"}


def day_trades(index: int, base: int):
    """1日ぶんの約定（時刻, 前場/後場, 価格, 株数）。株数は 100株単位。"""
    anchor = base + int(21 * np.sin(index / 7)) + index
    rng = np.random.RandomState(index * 31 + base)
    rows = [(time(9, 0, 0), "01", anchor, 3000), (time(9, 0, 0), "01", anchor, 2000)]
    for minute in range(1, 150, 7):
        hh, mm = divmod(9 * 60 + minute, 60)
        offset = int(rng.randint(-18, 19))
        size = int(rng.choice([100, 300, 1000, 2000, 8000]))
        rows.append((time(hh, mm, int(rng.randint(0, 60))), "01", anchor + offset, size))
    rows.append((time(11, 30, 0), "01", anchor + 2, 1500))
    rows.append((time(12, 30, 0), "02", anchor - 3, 2500))
    for minute in range(1, 170, 9):
        hh, mm = divmod(12 * 60 + 30 + minute, 60)
        offset = int(rng.randint(-18, 19))
        size = int(rng.choice([100, 200, 500, 1000, 9000]))
        rows.append((time(hh, mm, int(rng.randint(0, 60))), "02", anchor + offset, size))
    rows.append((time(15, 30, 0), "02", anchor + 7, 6000))
    return rows


def write_inputs(output) -> dict:
    root = Path(output)
    for folder in ("raw", "tick", "minute"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    calendar, bars = [], []
    tick_days, minute_frames = {}, []
    for index, current in enumerate(pd.date_range("2025-10-01", "2025-11-18")):
        day = current.date()
        trading = current.weekday() < 5 and day.isoformat() not in HOLIDAYS
        calendar.append((day.isoformat(), 1 if trading else 0))
        if not trading:
            continue
        ticks = []
        for code, base in CODES.items():
            trades = day_trades(index, base)
            frame = pd.DataFrame(trades, columns=["Time", "SessionDistinction", "Price", "TradingVolume"])
            frame["Value"] = frame.Price * frame.TradingVolume
            bars.append({"Date": day.isoformat(), "Code": code, "O": float(frame.Price.iloc[0]),
                         "H": float(frame.Price.max()), "L": float(frame.Price.min()),
                         "C": float(frame.Price.iloc[-1]), "Vo": float(frame.TradingVolume.sum()),
                         "Va": float(frame.Value.sum()), "AdjFactor": 1.0})
            frame["Minute"] = [f"{t.hour:02d}:{t.minute:02d}" for t in frame.Time]
            grouped = frame.groupby("Minute", sort=True)
            minute_frames.append(pd.DataFrame({
                "Date": day, "Time": list(grouped.groups), "Code": code,
                "O": grouped.Price.first().to_numpy(float), "H": grouped.Price.max().to_numpy(float),
                "L": grouped.Price.min().to_numpy(float), "C": grouped.Price.last().to_numpy(float),
                "Vo": grouped.TradingVolume.sum().to_numpy(float), "Va": grouped.Value.sum().to_numpy(float)}))
            if code == "123A0":
                ticks.append(frame.assign(Date=day, Code=code,
                                          TransactionId=[f"{i + 1:012d}" for i in range(len(frame))]))
        tick = pd.concat(ticks, ignore_index=True)
        table = pa.Table.from_pandas(tick[T.COLUMNS], preserve_index=False).cast(T.SCHEMA)
        path = root / "tick" / T.day_file(day)
        pq.write_table(table, path)
        tick_days[day.isoformat()] = {"rows": len(tick), "codes": 1, "bytes": path.stat().st_size,
                                      "release": T.year_tag(day), "key": "synthetic", "at": "2025-11-18T09:00:00+00:00"}
    pd.DataFrame(calendar, columns=["Date", "HolDiv"]).to_csv(root / "calendar.csv", index=False)
    pd.DataFrame(bars).to_parquet(root / "raw" / "bars_2025.parquet", index=False)
    minutes = pd.concat(minute_frames, ignore_index=True)
    minute_days = {}
    for ym, part in minutes.groupby(minutes.Date.map(lambda d: f"{d.year:04d}{d.month:02d}")):
        table = pa.Table.from_pandas(part[S.COLUMNS], preserve_index=False).cast(S.SCHEMA)
        pq.write_table(table, root / "minute" / S.file_name(ym))
        for day, rows in part.groupby("Date"):
            minute_days[day.isoformat()] = {"rows": len(rows), "codes": int(rows.Code.nunique()),
                                            "source": "file", "at": "2025-11-18T09:00:00+00:00"}
    manifest = T.empty_manifest()
    manifest["days"] = tick_days
    (root / "tick" / T.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    manifest = S.empty_manifest()
    manifest["days"] = minute_days
    (root / "minute" / S.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return {"days": sorted(tick_days), "bars": bars}


def write_predictions(path, as_of="2025-11-18") -> None:
    """直近5つの予測日に候補がある予測の JSON（123A は2日、9999 は1日、8899 は日足も分足も無い銘柄）。"""
    dates = ["2025-11-12", "2025-11-13", "2025-11-14", "2025-11-17", as_of]
    rows = [("2025-11-11", "7777", "対象外の古い日", 1), ("2025-11-14", "123A", "サンプル甲", 2),
            (as_of, "123A", "サンプル甲", 1), ("2025-11-17", "9999", "サンプル乙", 1),
            ("2025-11-12", "8899", "集計できない銘柄", 3)]
    candidates = [{"date": d, "code": c, "jqCode": c + "0" if len(c) == 4 else c, "name": n, "rankInDay": r}
                  for d, c, n, r in rows]
    Path(path).write_text(json.dumps({"asOf": as_of, "dates": dates, "candidates": candidates},
                                     ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    write_inputs(sys.argv[1])
    if len(sys.argv) > 2:
        write_predictions(sys.argv[2])
