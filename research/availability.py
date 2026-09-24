#!/usr/bin/env python3
"""
データの種類ごとに「その行を予測に使ってよい日（知りえた日）」を決める1か所。

特徴量はこの日で結合する（merge_asof の backward、当日を含む）。日次予測は
取り込みの後（平日の 21〜23 時 JST が実績。docs/OPERATIONS.md）に走るので、
その日の夜までに取り込めるものは当日から使ってよい。翌営業日以降にしか
出ないものは、出た日から使う。

**記録の日付（基準日・集計期間の末日）で結合してはいけない。** 学習だけが
公表前の値を見て、予測では同じ新しさの値が手に入らない。2026-09-24 に2つ
見つかった（docs/DATA_TIMING.md）:

  信用残（週次）     基準日（金曜）で結合 → 母集団の 38.5% の行が公表前の値
  投資部門別（週次） 集計期間の末日で結合 → 母集団の 85.7% の行が公表前の値

どの種別がどの日で結合されるかは RULES に並べ、tests/test_availability.py が
結合の結果（公表前の値が入らないこと）を種別ごとに確かめる。

  種別を足すときは、ここに行を足し、テストを足す。
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 週次の信用残（/markets/margin-interest）は、基準日（通常は金曜）の
#: **翌週の第何営業日**に公表されるか。JPX の週次公表は翌週第2営業日。
#: J-Quants がその日の夜の取り込みに間に合うかは research/probe_update_time.py の
#: margin_fri で測る（2026-09-24〜25 に測定。結果は docs/DATA_TIMING.md）
MARGIN_PUBLISH_BD = 2

#: 投資部門別（/equities/investor-types）の「使ってよい日」の列。
#: 集計期間の末日（EnDate）ではない。実測で EnDate の 6暦日後（木曜）が中心
INVESTOR_DATE_COL = "PubDate"

#: 修正の前後を比べる実験（research/exp/e43_pit_fix.py）のためだけのスイッチ。
#: True にすると 2026-09-24 以前の結合（信用残は基準日、投資部門別は集計期間の
#: 末日）に戻る。**本番のコードからは触らない。** 前後で同じコードを通すことで、
#: 比べた差が結合の違いだけから来ることを保証する（旧コードで作ったデータセットと
#: 1列も違わないことを確かめてから使う）
LEGACY = False

#: 種別 -> (使ってよい日, 根拠)。docs/DATA_TIMING.md と同じ中身
RULES: Dict[str, Tuple[str, str]] = {
    "bars":        ("Date（当日）", "16:30 JST までに出る（実測 2026-09-18）"),
    "indices":     ("Date（当日）", "16:30 JST（実測）"),
    "topix":       ("Date（当日）", "16:30 JST（実測）"),
    "master_hist": ("Date（当日）", "16:00 JST より前（実測）"),
    "fins":        ("DiscDate（当日）", "18:00 JST ごろ出る（実測）。取り込みはその後"),
    "margin":      (f"基準日の翌週 第{MARGIN_PUBLISH_BD}営業日", "JPX の週次公表"),
    "investor":    (INVESTOR_DATE_COL, "J-Quants の公表日の列"),
    "marginalert": ("PubDate", "公表日の列"),
    "earndate":    ("PubDate", "公表日の列。SchDate（予定日）では結合しない"),
    "valuation":   ("Date（当日）", "要確認: 2026-09-24 の probe で測る"),
    "shortratio":  ("Date（当日）", "要確認: 2026-09-24 の probe で測る"),
    "lvshld":      ("SubDate（提出日）", "要確認: J-Quants に載る時刻を probe で測る"),
    "mjrshld":     ("SubDate（提出日）", "要確認: 同上"),
    "xhold":       ("SubDate（提出日）", "要確認: 同上"),
}


def _trading_days(days: Optional[Sequence]) -> np.ndarray:
    if days is None or not len(days):
        return np.array([], dtype="datetime64[D]")
    return np.array(sorted(pd.Timestamp(d).date() for d in days), dtype="datetime64[D]")


def next_week_trading_day(asof: pd.Series, days: Optional[Sequence],
                          n: int = MARGIN_PUBLISH_BD) -> pd.Series:
    """
    基準日の**翌週の第 n 営業日**。

    翌週 = 基準日の次の月曜から。営業日は取引所カレンダー（days）で数える。
    カレンダーが覆っていない日は平日で数える（祝日を知らないぶん1日早く
    なりうるが、覆っていない範囲は学習に使っていない 2016 年以前だけ）。
    """
    a = pd.to_datetime(asof).dt.normalize()
    wd = a.dt.weekday
    monday = a + pd.to_timedelta(7 - wd, unit="D")
    cal = _trading_days(days)
    out = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
    m = monday.to_numpy(dtype="datetime64[D]")
    ok = a.notna().to_numpy()
    if len(cal):
        i = np.searchsorted(cal, m, side="left") + (n - 1)
        inside = ok & (m >= cal[0]) & (i < len(cal))
        out[inside] = pd.to_datetime(cal[i[inside]])
    else:
        inside = np.zeros(len(a), dtype=bool)
    rest = ok & ~inside
    if rest.any():
        # 平日で数える: 翌週の月曜 + (n-1) 営業日
        out[rest] = pd.to_datetime(np.busday_offset(m[rest], n - 1, roll="forward"))
    return out


def margin_available(margin: pd.DataFrame, days: Optional[Sequence]) -> pd.Series:
    """週次の信用残の各行を、予測に使ってよい日。"""
    if LEGACY:
        return pd.to_datetime(margin["Date"])
    return next_week_trading_day(margin["Date"], days, MARGIN_PUBLISH_BD)


def investor_available(d: pd.DataFrame) -> pd.Series:
    """
    投資部門別の各行を、予測に使ってよい日。

    PubDate が無い古い保存形式では、実測の中心（EnDate + 6暦日）で代える。
    代えた行が出たら学習に気づけるよう、呼び出し側でログに出す。
    """
    if LEGACY:
        return pd.to_datetime(d["EnDate"], errors="coerce")
    if INVESTOR_DATE_COL in d.columns:
        pub = pd.to_datetime(d[INVESTOR_DATE_COL], errors="coerce")
    else:
        pub = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    if "EnDate" in d.columns:
        fallback = pd.to_datetime(d["EnDate"], errors="coerce") + pd.Timedelta(days=6)
        pub = pub.fillna(fallback)
    return pub
