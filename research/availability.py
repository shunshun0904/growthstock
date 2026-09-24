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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 週次の信用残（/markets/margin-interest）は、基準日（通常は金曜）の
#: **翌週の第何営業日**に公表されるか。JPX の週次公表は翌週第2営業日。
#: J-Quants がその日の夜の取り込みに間に合うかは research/probe_update_time.py の
#: margin_fri で測る。9/24（連休明けの第1営業日）は 20:00 JST までに出なかった
#: （規則どおり）。第2営業日の 9/25 に測る（結果は docs/DATA_TIMING.md）
MARGIN_PUBLISH_BD = 2

#: 投資部門別（/equities/investor-types）の「使ってよい日」の列。
#: 集計期間の末日（EnDate）ではない。実測で EnDate の 6暦日後（木曜）が中心
INVESTOR_DATE_COL = "PubDate"

#: 空売り残高報告（/markets/short-sale-report）を公表日（DiscDate）の**当日**から
#: 使うか。実測（research/probe_update_time.py、2026-09-24）では当日の 18:02 JST に
#: 出た。夜の取り込み（実績 21〜23時）には間に合うが、取り込みが定刻（16:05 JST）
#: どおりに起動した日には間に合わない。間に合わないのに当日から使うと学習だけが
#: 1日早い値を見るので、1日遅れを受け入れて翌営業日から使う
SHORTSALE_SAME_DAY = False

#: 修正の前後を比べる実験（research/exp/e43_pit_fix.py）のためだけのスイッチ。
#: True にすると 2026-09-24 以前の結合（信用残は基準日、投資部門別は集計期間の
#: 末日）に戻る。**本番のコードからは触らない。** 前後で同じコードを通すことで、
#: 比べた差が結合の違いだけから来ることを保証する（旧コードで作ったデータセットと
#: 1列も違わないことを確かめてから使う）
LEGACY = False

#: 種別 -> (使ってよい日, 根拠)。docs/DATA_TIMING.md と同じ中身
RULES: Dict[str, Tuple[str, str]] = {
    "bars":        ("Date（当日）", "16:00 JST（実測 2026-09-18・09-24）"),
    "indices":     ("Date（当日）", "16:30 JST（実測）"),
    "topix":       ("Date（当日）", "16:30 JST（実測）"),
    "master_hist": ("Date（当日）", "16:00 JST より前（実測）"),
    "fins":        ("DiscDate（当日）", "18:00 JST ごろ出る（実測）。取り込みはその後"),
    "margin":      (f"基準日の翌週 第{MARGIN_PUBLISH_BD}営業日", "JPX の週次公表"),
    "investor":    (INVESTOR_DATE_COL, "J-Quants の公表日の列"),
    "marginalert": ("PubDate", "公表日の列。16:30 JST に出る（実測 2026-09-24）"),
    "earndate":    ("PubDate", "公表日の列。SchDate（予定日）では結合しない。16:00 JST（実測）"),
    "valuation":   ("Date（当日）", "16:00 JST（実測 2026-09-24）"),
    "shortratio":  ("Date（当日）", "16:30 JST（実測 2026-09-24）"),
    "lvshld":      ("SubDate（提出日）", "16:00〜17:31 JST に少しずつ出る（実測 2026-09-24）。"
                    "今日の日は取得済みにせず翌日に取り直す（jq_bulk._record_fetched）"),
    "mjrshld":     ("SubDate（提出日）", "同上"),
    "xhold":       ("SubDate（提出日）", "同上"),
    "shortsale":   ("DiscDate（公表日）の翌営業日",
                    "18:02 JST に出る（実測 2026-09-24）。取り込みの時刻に左右されないよう"
                    "翌営業日のまま（SHORTSALE_SAME_DAY）"),
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


def next_trading_day(dates: pd.Series, days: Optional[Sequence]) -> pd.Series:
    """
    その日より**後**の最初の営業日（当日は含まない）。

    営業日は取引所カレンダー（days）で数える。カレンダーが覆っていない日は
    平日で数える。
    """
    a = pd.to_datetime(dates, errors="coerce").dt.normalize()
    cal = _trading_days(days)
    out = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
    ok = a.notna().to_numpy()
    d = a.to_numpy(dtype="datetime64[D]")
    if len(cal):
        i = np.searchsorted(cal, d, side="right")
        inside = ok & (d >= cal[0]) & (i < len(cal))
        out[inside] = pd.to_datetime(cal[i[inside]])
    else:
        inside = np.zeros(len(a), dtype=bool)
    rest = ok & ~inside
    if rest.any():
        out[rest] = pd.to_datetime(np.busday_offset(d[rest], 1, roll="forward"))
    return out


def shortsale_available(disc: pd.Series, days: Optional[Sequence]) -> pd.Series:
    """空売り残高報告の各行を、予測に使ってよい日（SHORTSALE_SAME_DAY を見る）。"""
    if SHORTSALE_SAME_DAY:
        return pd.to_datetime(disc, errors="coerce").dt.normalize()
    return next_trading_day(disc, days)
