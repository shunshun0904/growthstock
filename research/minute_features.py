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


#: 分布の形（実験132の --shape。運用者の問い 2026-11-19「正例になりやすい、価格帯別出来高の分布などないでしょうか？」）:
#: 期間の株数を、D の終値からの距離の帯に分けた割合（合計1）。帯は次の8つ
SHAPE_EDGES = (-np.inf, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, np.inf)
SHAPE_LABELS = ("−20%以下", "−20〜−10%", "−10〜−5%", "−5〜−2%", "−2〜0%", "0〜+2%", "+2〜+5%", "+5%超")


def shape_columns(n: int) -> List[str]:
    return [f"vp_shape_{n}d_b{i}" for i in range(len(SHAPE_EDGES) - 1)]


SHAPE_COLUMNS = [c for n in WINDOWS for c in shape_columns(n)]


def shape_of(bands, close: float, edges=SHAPE_EDGES) -> np.ndarray:
    """
    window_profiles の bands（[下限, 上限, 株数, 売買代金]。D の値の基準）の株数を、終値からの距離（価格 ÷ 終値 − 1）の帯に、
    重なる長さで按分した割合（合計1）。帯の中では株数が一様にあるとみなす。
    """
    b = np.asarray(bands, dtype=float)
    if not len(b) or close is None or close <= 0:
        return np.full(len(edges) - 1, np.nan)
    lo, hi, qty = b[:, 0] / close - 1.0, b[:, 1] / close - 1.0, b[:, 2]
    span = np.where(hi > lo, hi - lo, 1.0)
    out = []
    for a, z in zip(edges[:-1], edges[1:]):
        overlap = np.minimum(hi, z) - np.maximum(lo, a)
        overlap = np.where(overlap > 1e-12, overlap, 0.0)          # 境目の浮動小数点の誤差は重なりにしない
        out.append(float(np.sum(qty * overlap / span)))
    out = np.array(out)
    total = float(qty.sum())
    return out / total if total > 0 else np.full(len(edges) - 1, np.nan)


STATUS_COLUMNS = [f"vp_minute_status_{n}d" for n in WINDOWS]
#: 分足の始まりより前の行の状態（集計しない）
BEFORE_MINUTE = "before_minute"


class Inputs:
    """load_inputs の結果: 行ごとの銘柄・日、分足の始まり以降の行の添字、営業日、日足、分足（銘柄 → 日 → Session）。"""

    def __init__(self, stored, codes, iso, target, sessions, quotes, minute, by_month):
        self.stored, self.codes, self.iso, self.target = stored, codes, iso, target
        self.sessions, self.quotes, self.minute, self.by_month = sessions, quotes, minute, by_month


def load_inputs(keys: pd.DataFrame, minute_dir: str, data_dir: str, lookback_days: int = max(WINDOWS)):
    """
    keys（Code, Date）の行に要る日足と分足を読む（build と minute_dynamics が使う）。分足の置き場に日が無ければ None。
    lookback_days: 行ごとに D を含む直近何営業日を読むか
    """
    manifest = VD.load_manifest(os.path.join(minute_dir, "minute_manifest.json"))
    stored = sorted(VD.stored_days(manifest))
    if not stored:
        return None
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
    need: Dict[str, set] = defaultdict(set)            # 日 -> 銘柄
    for i in target:
        for day in VP.lookback(sessions, iso[i], lookback_days):
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
    return Inputs(stored, codes, iso, target, sessions, quotes, minute, by_month)


def build(keys: pd.DataFrame, minute_dir: str, data_dir: str,
          with_status: bool = False, with_shape: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    keys（Code, Date の行）に33列を付けた表（keys と同じ行・同じ並び）と、件数だけの記録を返す。
    with_status なら、期間ごとの集計の状態（volume_profile.STATUSES か BEFORE_MINUTE）の列も付ける（EDA 用）。
    with_shape なら、分布の形の列（SHAPE_COLUMNS。終値からの距離の帯ごとの株数の割合）も付ける（実験132の --shape 用）。

    minute_dir: minute_manifest.json と minute_YYYYMM.parquet の置き場（Release data-minute を落とした場所）
    data_dir:   bars_YYYY.parquet（Release data-raw）の置き場
    """
    inp = load_inputs(keys, minute_dir, data_dir)
    stored = inp.stored if inp is not None else []
    out = pd.DataFrame({"Code": keys["Code"].to_numpy(), "Date": pd.to_datetime(keys["Date"]).to_numpy()})
    for col in COLUMNS:
        out[col] = np.nan
    if with_status:
        for col in STATUS_COLUMNS:
            out[col] = BEFORE_MINUTE
    if with_shape:
        for col in SHAPE_COLUMNS:
            out[col] = np.nan
    report = {"rows": int(len(out)), "minute_days": len(stored),
              "minute_from": stored[0] if stored else None, "minute_to": stored[-1] if stored else None}
    if inp is None:
        report["attached"] = 0
        return out, report
    codes, iso, target, sessions = inp.codes, inp.iso, inp.target, inp.sessions
    quotes, minute, by_month = inp.quotes, inp.minute, inp.by_month
    status = Counter()
    for i in target:
        code, day = codes[i], iso[i]
        daily = quotes.get(code, {})
        series = {"tick": {}, "minute": minute.get(code, {})}
        close = daily.get(day).close if daily.get(day) and daily[day].valid else None
        for n in WINDOWS:
            result = VP.window_profiles(VP.lookback(sessions, day, n), n, daily, series)["minute"]
            status[f"{n}d/{result['status']}"] += 1
            if with_status:
                out.iat[i, out.columns.get_loc(f"vp_minute_status_{n}d")] = result["status"]
            if result["status"] != "ok" or close is None:
                continue
            for metric, value in values(result, close).items():
                out.iat[i, out.columns.get_loc(column(metric, n))] = value
            if with_shape:
                for col, value in zip(shape_columns(n), shape_of(result["bands"], close)):
                    out.iat[i, out.columns.get_loc(col)] = value
    report["target_rows"] = int(len(target))
    report["attached"] = int(out[COLUMNS].notna().any(axis=1).sum())
    report["attached_by_window"] = {f"{n}d": int(out[column("vwap_gap", n)].notna().sum()) for n in WINDOWS}
    report["status"] = dict(sorted(status.items()))
    report["missing_month_files"] = sorted(n for n in by_month if not os.path.exists(os.path.join(minute_dir, n)))
    return out, report
