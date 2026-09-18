#!/usr/bin/env python3
"""
その日の入力データが J-Quants に出るまで待つ（取り込みの前に挟む）。

なぜ要るか
--------
取り込みの起動時刻を前倒ししたい。実測では GitHub Actions の
スケジュール起動が毎回4〜5.6時間遅れているので、cron を早い時刻に置くほど
実際の公開が早くなる。ただし遅延は一定ではないので、たまたま定刻に
起動した日には「まだ当日データが無い」状態で走ることになる。

そこで、取り込みの前にここで待つ。待てば済む話を、失敗にしない。

なぜ四本値だけでは足りないか
--------------------------
実測（research/probe_update_time.py, 2026-09-18）では出る時刻が違う。

  四本値・銘柄マスタ  16:00 までに出ている
  業種別指数・TOPIX   16:30
  決算開示            18:00

四本値だけ待って 16:05 に走ると、指数と TOPIX が0件のまま取り込むことに
なる。build_dataset は指数・TOPIX を merge_asof(direction="backward") で
結合するので、**その日の行が無いと黙って前日の値が入る**。欠測として
現れないぶん質が悪い。だから既定では指数・TOPIX まで待つ。

決算開示（18:00）は待たない。1時間半の遅れに見合わないと判断した。
取りこぼしても失われはしない ——`data_store.confirmed_days` が
「0件で、まだ公表前かもしれない日」を取得済みにしないので、翌営業日の
取り込みで入る（1日遅れる）。同じ日に入れたければ `--feeds` に fins を
足して `--deadline` を 18:30 にすればよい。

止めない
------
締切まで出なかった場合も **exit 0 で先へ進む**。祝日・臨時休場のように
そもそも当日データが存在しない日があり、ここで落とすとパイプライン全体が
その日だけ止まる。古いデータで予測してしまう事故は、後段の
`research/check_freshness.py` が既に見ている。ここは前倒しのための
最適化であって、新しい失敗の入口にはしない。

  $ JQUANTS_API=... python3 research/wait_for_data.py --deadline 18:00
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key)

JST = dt.timezone(dt.timedelta(hours=9))

#: 待てる対象。名前 -> (パス, パラメータの形, 何に使うか)
#: パスとパラメータの形は research/probe_update_time.py と同じにしてある
#: （実測したのと違う叩き方で待つと、測った時刻が当てにならない）。
FEEDS: Dict[str, tuple] = {
    "bars": ("/equities/bars/daily", "date", "株価四本値。予測の主データ"),
    "indices": ("/indices/bars/daily", "date", "業種別指数。sector_ret_* の元"),
    "topix": ("/indices/bars/daily/topix", "range", "TOPIX。地合いの元"),
    "master": ("/equities/master", "date", "銘柄マスタ（市場区分・業種）"),
    "fins": ("/fins/summary", "date", "決算開示"),
}

#: 既定で待つ対象。ここが欠けると、その日の予測が前日の値で作られる。
DEFAULT_FEEDS = ("bars", "indices", "topix")


def _hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def _params(shape: str, day: dt.date) -> dict:
    if shape == "range":
        return {"from": day.isoformat(), "to": day.isoformat()}
    return {"date": day.isoformat()}


def nap_seconds(now: dt.datetime, deadline: dt.datetime,
                interval: int) -> float:
    """
    次に見るまで眠る秒数。

    締切をまたいで眠らない。またぐと締切の意味が無くなる（間隔60分・
    締切まで5分のときに60分眠ると、締切から55分過ぎて目を覚ます）。
    """
    return min(interval, max(1.0, (deadline - now).total_seconds()))


def count_rows(client: JQuantsClient, feed: str,
               day: dt.date) -> Optional[int]:
    """その日の行数。問い合わせ自体が失敗したら None。"""
    path, shape, _ = FEEDS[feed]
    try:
        rows = client.get_paginated(path, _params(shape, day))
    except JQuantsError as exc:
        print(f"  [warn] {feed} の問い合わせに失敗: {str(exc)[:120]}")
        return None
    if feed == "bars":
        # 行が返るだけでは足りない。前場ぶんだけ入って終値が空、という
        # 出方をされると、気づかずに欠測を取り込むことになる。
        return sum(1 for r in rows if r.get("C") not in (None, "", "-"))
    return len(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="当日の入力データが出るまで待つ")
    ap.add_argument("--date", default=None, help="対象日 (YYYY-MM-DD, JST)")
    ap.add_argument("--feeds", default=",".join(DEFAULT_FEEDS),
                    help="待つ対象をカンマ区切りで。既定 "
                         + ",".join(DEFAULT_FEEDS))
    ap.add_argument("--deadline", default="18:00",
                    help="この時刻(JST)まで待って、出なければ諦めて進む")
    ap.add_argument("--interval", type=int, default=300,
                    help="確認の間隔（秒）")
    args = ap.parse_args(argv)

    feeds = [f for f in (x.strip() for x in args.feeds.split(",")) if f]
    unknown = [f for f in feeds if f not in FEEDS]
    if unknown:
        # 名前を間違えたときに「待たずに素通り」にはしない。ここは
        # 引数の書き間違いで、当日データの有無とは関係がない
        print(f"[stop] 知らない対象: {', '.join(unknown)} "
              f"(選べるのは {', '.join(FEEDS)})")
        return 2

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

    print(f"[wait] {day} のデータが出るまで待つ "
          f"(締切 {deadline:%H:%M} JST / {args.interval}秒ごと)")
    for f in feeds:
        print(f"  {f:8s} {FEEDS[f][0]:32s} {FEEDS[f][2]}")

    ready: Dict[str, dt.datetime] = {}
    while True:
        t = dt.datetime.now(JST)
        for f in [x for x in feeds if x not in ready]:
            n = count_rows(client, f, day)
            print(f"  {t:%H:%M:%S} JST  {f:8s} "
                  f"{n if n is not None else '?'}行")
            if n:
                ready[f] = t
                print(f"  ** {f} が {t:%H:%M:%S} JST に揃った")
        if len(ready) == len(feeds):
            waited = (t - now).total_seconds() / 60
            print(f"[ready] {day} の {', '.join(feeds)} が揃った"
                  f"（{waited:.0f}分待った）")
            return 0
        if t >= deadline:
            missing = [f for f in feeds if f not in ready]
            print(f"[giveup] 締切 {deadline:%H:%M} JST までに "
                  f"{', '.join(missing)} が出なかった。祝日・臨時休場の"
                  "可能性もあるのでこのまま進む。0件の日を取得済みにして"
                  "しまう事故は data_store.confirmed_days が、"
                  "古いデータで予測する事故は check_freshness.py が見ている")
            return 0
        time.sleep(nap_seconds(t, deadline, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
