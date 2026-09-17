#!/usr/bin/env python3
"""
その日の株価四本値が J-Quants に出るまで待つ（取り込みの前に挟む）。

なぜ要るか
--------
取り込みの起動時刻を前倒ししたい。実測では GitHub Actions の
スケジュール起動が毎回4〜5.6時間遅れているので、cron を早い時刻に置くほど
実際の公開が早くなる。ただし遅延は一定ではないので、たまたま定刻に
起動した日には「まだ当日データが無い」状態で走ることになる。

そこで、取り込みの前にここで待つ。待てば済む話を、失敗にしない。

止めない
------
締切まで出なかった場合も **exit 0 で先へ進む**。祝日・臨時休場のように
そもそも当日データが存在しない日があり、ここで落とすとパイプライン全体が
その日だけ止まる。古いデータで予測してしまう事故は、後段の
`research/check_freshness.py` が既に見ている。ここは前倒しのための
最適化であって、新しい失敗の入口にはしない。

  $ JQUANTS_API=... python3 research/wait_for_bars.py --deadline 20:00
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key)

JST = dt.timezone(dt.timedelta(hours=9))
PATH = "/equities/bars/daily"


def _hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def nap_seconds(now: dt.datetime, deadline: dt.datetime,
                interval: int) -> float:
    """
    次に見るまで眠る秒数。

    締切をまたいで眠らない。またぐと締切の意味が無くなる（間隔60分・
    締切まで5分のときに60分眠ると、締切から55分過ぎて目を覚ます）。
    """
    return min(interval, max(1.0, (deadline - now).total_seconds()))


def count_rows(client: JQuantsClient, day: dt.date) -> Optional[int]:
    """その日の行数。問い合わせ自体が失敗したら None。"""
    try:
        return len(client.get_paginated(PATH, {"date": day.isoformat()}))
    except JQuantsError as exc:
        print(f"  [warn] 問い合わせに失敗: {str(exc)[:120]}")
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="当日の四本値が出るまで待つ")
    ap.add_argument("--date", default=None, help="対象日 (YYYY-MM-DD, JST)")
    ap.add_argument("--deadline", default="20:00",
                    help="この時刻(JST)まで待って、出なければ諦めて進む")
    ap.add_argument("--interval", type=int, default=300,
                    help="確認の間隔（秒）")
    args = ap.parse_args(argv)

    now = dt.datetime.now(JST)
    day = dt.date.fromisoformat(args.date) if args.date else now.date()
    deadline = dt.datetime.combine(day, _hhmm(args.deadline), tzinfo=JST)

    if day.weekday() >= 5:
        print(f"[skip] {day} は土日なので待たない")
        return 0

    try:
        key = resolve_api_key()
    except JQuantsError as exc:
        # 鍵が無い・壊れているのはここで落とす話ではない。取り込み本体が
        # 同じ鍵で走って、そこで落ちて理由を出す。ここで先に落とすと
        # 「待ち」が新しい失敗の入口になる。
        print(f"[skip] APIキーを解決できないので待たない: {str(exc)[:120]}")
        return 0
    client = JQuantsClient(key, pause=0.2)

    print(f"[wait] {day} の {PATH} が出るまで待つ "
          f"(締切 {deadline:%H:%M} JST / {args.interval}秒ごと)")
    while True:
        t = dt.datetime.now(JST)
        n = count_rows(client, day)
        print(f"  {t:%H:%M:%S} JST  {n if n is not None else '?'}行")
        if n:
            waited = (t - now).total_seconds() / 60
            print(f"[ready] {day} の四本値が {t:%H:%M:%S} JST に揃った"
                  f"（{waited:.0f}分待った）")
            return 0
        if t >= deadline:
            print(f"[giveup] 締切 {deadline:%H:%M} JST までに出なかった。"
                  "祝日・臨時休場の可能性もあるのでこのまま進む。"
                  "古いデータで予測する事故は check_freshness.py が見ている")
            return 0
        time.sleep(nap_seconds(t, deadline, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
