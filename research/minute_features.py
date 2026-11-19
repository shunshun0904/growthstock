"""
分足から作る価格帯別出来高の特徴量（実験132。研究用で、本番の特徴量にはまだ入れていない）。

学習データの各行（銘柄・日 D）について、D を含む直近 1・5・20 営業日の分足を、日足（調整前の安値・高値・出来高・
売買代金）と照合してから集計し、次の11指標 × 3期間 = 33列にする。集計の作りは「価格帯別出来高」タブと同じ
（research/volume_profile.py の minute_session・window_profiles。docs/VOLUME_PROFILE.md「計算」）。

  vp_minute_poc_gap_{n}d            最多出来高の帯の中点 ÷ 終値 − 1       （PR #8 の7指標）
  vp_minute_vwap_gap_{n}d           期間の VWAP ÷ 終値 − 1
  vp_minute_value_low_gap_{n}d      出来高の70%範囲の下限 ÷ 終値 − 1
  vp_minute_value_high_gap_{n}d     出来高の70%範囲の上限 ÷ 終値 − 1
  vp_minute_poc_share_{n}d          最多出来高の帯の株数の割合
  vp_minute_above_close_share_{n}d  終値より上の価格の株数の割合
  vp_minute_near_close_share_{n}d   終値 ±1% の株数の割合
  vp_minute_open_share_{n}d         寄り（最初の1分）の株数の割合        （時間帯の4指標）
  vp_minute_morning_share_{n}d      前場の残り
  vp_minute_afternoon_share_{n}d    後場（引けを除く）
  vp_minute_close_share_{n}d        引け（15:25 以降の分。2024-11-05 より前は 15:00 以降）

時点: D の当日の引けまで（日足の特徴量と同じ。運用者の選択 2026-11-19「当日 D まで」）。期間の営業日の分足が置き場に無い・
日足と合わない・分割併合の係数が分からない、などで集計できない期間は欠測（NaN）にする。値を埋めない。
分足は 2024-11-01 からなので、それより前の行はすべて欠測になる。
"""

from __future__ import annotations

import datetime as dt
import glob
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import volume_profile as VP
import volume_profile_data as VD

WINDOWS = VP.WINDOWS
PROFILE_METRICS = ("poc_gap", "vwap_gap", "value_low_gap", "value_high_gap",
                   "poc_share", "above_close_share", "near_close_share")
SESSION_METRICS = tuple(f"{b}_share" for b in VP.BUCKETS)
METRICS = PROFILE_METRICS + SESSION_METRICS
COLUMNS = [f"vp_minute_{metric}_{n}d" for n in WINDOWS for metric in METRICS]


def column(metric: str, n: int) -> str:
    return f"vp_minute_{metric}_{n}d"


def values(result: dict, close: float) -> Dict[str, float]:
    """window_profiles の1期間・分足の結果（status ok）から11指標。"""
    poc_mid = (result["poc"][0] + result["poc"][1]) / 2
    out = {
        "poc_gap": poc_mid / close - 1,
        "vwap_gap": result["vwap"] / close - 1,
        "value_low_gap": result["valueArea"][0] / close - 1,
        "value_high_gap": result["valueArea"][1] / close - 1,
        "poc_share": result["pocShare"],
        "above_close_share": result["aboveCloseShare"],
        "near_close_share": result["nearCloseShare"],
    }
    for bucket in VP.BUCKETS:
        out[f"{bucket}_share"] = result["sessions"][bucket]
    return out


def trading_days(bars_paths: Iterable[str], start: str) -> List[str]:
    """日足のある日（営業日）。取り込み（data_store）は営業日だけを取るので、日足の日付をそのまま営業日とする。"""
    days = set()
    for path in bars_paths:
        frame = pd.read_parquet(path, columns=["Date"])
        iso = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        days.update(iso[iso >= start].unique().tolist())
    return sorted(days)


def build(keys: pd.DataFrame, minute_dir: str, data_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    keys（Code, Date の行）に33列を付けた表（keys と同じ行・同じ並び）と、件数だけの記録を返す。

    minute_dir: minute_manifest.json と minute_YYYYMM.parquet の置き場（Release data-minute を落とした場所）
    data_dir:   bars_YYYY.parquet（Release data-raw）の置き場
    """
    manifest = VD.load_manifest(os.path.join(minute_dir, "minute_manifest.json"))
    stored = sorted(VD.stored_days(manifest))
    out = pd.DataFrame({"Code": keys["Code"].to_numpy(), "Date": pd.to_datetime(keys["Date"]).to_numpy()})
    for col in COLUMNS:
        out[col] = np.nan
    report = {"rows": int(len(out)), "minute_days": len(stored),
              "minute_from": stored[0] if stored else None, "minute_to": stored[-1] if stored else None}
    if not stored:
        report["attached"] = 0
        return out, report
    first = stored[0]
    codes = keys["Code"].map(VP.normalize_code).to_numpy()
    iso = pd.to_datetime(keys["Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    target = np.flatnonzero(iso >= first)
    years = sorted({int(d[:4]) for d in iso[target]} | {int(first[:4])})
    bars_paths = [p for p in (os.path.join(data_dir, VD.bars_file(y)) for y in years) if os.path.exists(p)]
    missing_bars = [VD.bars_file(y) for y in years if not os.path.exists(os.path.join(data_dir, VD.bars_file(y)))]
    if missing_bars:
        raise FileNotFoundError(f"日足のファイルが無い: {missing_bars}")
    sessions = trading_days(bars_paths, first)
    windows = {i: VP.lookback(sessions, iso[i], max(WINDOWS)) for i in target}
    need: Dict[str, set] = defaultdict(set)            # 日 -> 銘柄
    for i in target:
        for day in windows[i]:
            need[day].add(codes[i])
    need_codes = sorted({c for cs in need.values() for c in cs})
    quotes = VD.load_quotes(bars_paths, need_codes, need.keys())
    minute: Dict[str, Dict[str, VP.Session]] = defaultdict(dict)
    by_month: Dict[str, List[str]] = defaultdict(list)
    stored_set = set(stored)
    for day in sorted(need):
        if day in stored_set:
            by_month[VD.minute_file(day)].append(day)
    for name, days in sorted(by_month.items()):
        path = os.path.join(minute_dir, name)
        if not os.path.exists(path):
            continue                                    # その月の日は欠測（置き場の記録にあるのに無い。記録に数える）
        month_codes = sorted({c for d in days for c in need[d]})
        for code, by_day in VD.load_minute_month(path, days, month_codes).items():
            minute[code].update(by_day)
    status = Counter()
    for i in target:
        code, day = codes[i], iso[i]
        daily = quotes.get(code, {})
        series = {"tick": {}, "minute": minute.get(code, {})}
        close = daily.get(day).close if daily.get(day) and daily[day].valid else None
        for n in WINDOWS:
            result = VP.window_profiles(VP.lookback(sessions, day, n), n, daily, series)["minute"]
            status[f"{n}d/{result['status']}"] += 1
            if result["status"] != "ok" or close is None:
                continue
            for metric, value in values(result, close).items():
                out.iat[i, out.columns.get_loc(column(metric, n))] = value
    report["target_rows"] = int(len(target))
    report["attached"] = int(out[COLUMNS].notna().any(axis=1).sum())
    report["attached_by_window"] = {f"{n}d": int(out[column("vwap_gap", n)].notna().sum()) for n in WINDOWS}
    report["status"] = dict(sorted(status.items()))
    report["missing_month_files"] = sorted(n for n in by_month if not os.path.exists(os.path.join(minute_dir, n)))
    return out, report
