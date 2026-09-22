#!/usr/bin/env python3
"""
東証の営業日カレンダー。取り込みが保存した事実を読むだけの部品。

なぜ要るか
--------
2026-09-22（敬老の日〜秋分の日の連休の中日）に日次予測が落ちた。
鮮度チェック（check_freshness.py）は遅れを **平日** で数えており、
3連休だと「遅れ2営業日」になって許容の1を超えるため。

祝日カレンダーを持ち込まなかったのは意図的で、「更新が止まったのに
祝日だからと自分で言い訳する」余地を作らないためだった
（check_freshness.py の元のコメント）。

**その懸念は「自分で推測する」場合の話**で、いまは違う。取り込み
（jq_bulk.trading_days）は元から `/markets/calendar` を叩いて営業日を
決めている。叩いているのに捨てていただけなので、保存して読む。
推測ではなく取引所が公表した事実を使う。

言い訳にしないための決め事
----------------------
- **カレンダーが対象日を含んでいないときは使わない。** 呼び出し側は
  平日で数える従来の方法に倒す（厳しい側）
- カレンダーが「営業日だ」と言っている日にデータが無いなら、それは
  取り込み障害。カレンダーは何も言い訳しない
- ファイルが無くても壊れていても、例外を投げずに「分からない」を返す。
  ここが新しい障害の入口になってはいけない

  >>> cal = load()
  >>> cal.is_trading_day(dt.date(2026, 9, 22))
  False
  >>> cal.count_between(dt.date(2026, 9, 18), dt.date(2026, 9, 22))
  0
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
FILENAME = "calendar.parquet"

#: 営業日とみなす区分。J-Quants の HolDiv（V1 は HolidayDivision）
#:   "0" 非営業日 / "1" 営業日 / "2" 東証半日立会 / "3" 非営業日（祝日取引あり）
#: "3" は東証の立会が無いので営業日に数えない。jq_bulk.trading_days と同じ判定
TRADING_DIV = ("1", "2")
DIV_KEYS = ("HolDiv", "HolidayDivision", "HolidayDiv")


class Calendar:
    """営業日の集合。範囲外は「分からない」として扱う。"""

    def __init__(self, days: Iterable[dt.date], covered: Iterable[dt.date] = ()):
        self.days = frozenset(days)
        cov = sorted(covered) or sorted(self.days)
        self.start: Optional[dt.date] = cov[0] if cov else None
        self.end: Optional[dt.date] = cov[-1] if cov else None

    def __bool__(self) -> bool:
        return bool(self.days)

    def covers(self, *dates: dt.date) -> bool:
        """与えた日をすべて含む範囲を持っているか。1つでも外なら False。"""
        if not self.days or self.start is None:
            return False
        return all(self.start <= d <= self.end for d in dates)

    def is_trading_day(self, day: dt.date) -> Optional[bool]:
        """営業日か。範囲外なら None（分からない）。"""
        if not self.covers(day):
            return None
        return day in self.days

    def count_between(self, start: dt.date, end: dt.date) -> Optional[int]:
        """
        [start, end) に入る営業日の数。範囲外なら None。

        np.busday_count と同じ半開区間にしてある。差し替えても意味が
        変わらないようにするため。start 自身を含むので、

          1  最終バーが前営業日（= 当日ぶんはまだ出ていない。正常）
          2  1営業日ぶん取りこぼしている

        となる。鮮度チェックの既定 --max-age 1 はこの数え方が前提。
        """
        if not self.covers(start, end):
            return None
        return sum(1 for d in self.days if start <= d < end)

    def last_trading_day(self, as_of: dt.date) -> Optional[dt.date]:
        """as_of 以前でいちばん新しい営業日。範囲外なら None。"""
        if not self.covers(as_of):
            return None
        past = [d for d in self.days if d <= as_of]
        return max(past) if past else None


def parse_rows(rows: Iterable[dict]) -> List[dt.date]:
    """API の応答（または保存した行）から営業日だけを取り出す。"""
    out = []
    for r in rows or ():
        raw = r.get("Date")
        if not raw:
            continue
        div = next((str(r[k]) for k in DIV_KEYS if r.get(k) is not None), "")
        if div not in TRADING_DIV:
            continue
        try:
            out.append(dt.date.fromisoformat(str(raw)[:10]))
        except ValueError:
            continue
    return sorted(out)


def load(data_dir: str = DATA_DIR) -> Calendar:
    """
    保存済みのカレンダーを読む。**失敗しても例外を投げない。**

    読めなければ空の Calendar を返し、呼び出し側は従来どおり平日で数える。
    """
    path = os.path.join(data_dir, FILENAME)
    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:                                        # noqa: BLE001
        return Calendar([])
    if not len(df):
        return Calendar([])
    rows = df.to_dict("records")
    days = parse_rows(rows)
    covered = []
    for r in rows:
        try:
            covered.append(dt.date.fromisoformat(str(r.get("Date"))[:10]))
        except (ValueError, TypeError):
            continue
    return Calendar(days, covered)


def save(rows: Iterable[dict], data_dir: str = DATA_DIR) -> Optional[str]:
    """
    取り込みが叩いた応答をそのまま残す。既存の行とは日付で突き合わせて上書き。

    **区分（HolDiv）を落とさずに残す。** 営業日だけに絞って保存すると、
    「非営業日だと分かっている日」と「カレンダーの範囲外」が区別できなくなる。
    """
    import pandas as pd

    rows = [r for r in (rows or ()) if r.get("Date")]
    if not rows:
        return None
    new = pd.DataFrame(rows)
    new["Date"] = new["Date"].astype(str).str.slice(0, 10)
    keep = ["Date"] + [k for k in DIV_KEYS if k in new.columns]
    new = new[keep].drop_duplicates("Date", keep="last")

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, FILENAME)
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            new = pd.concat([old, new], ignore_index=True) \
                    .drop_duplicates("Date", keep="last")
        except Exception:                                    # noqa: BLE001
            pass                                             # 壊れていれば作り直す
    new = new.sort_values("Date").reset_index(drop=True)
    new.to_parquet(path, index=False, compression="zstd")
    return path
