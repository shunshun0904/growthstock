#!/usr/bin/env python3
"""
J-Quants API が「その日のデータ」を何時に出すかを実測する。

なぜ実測するか
------------
取り込みを平日21:30 JST に置いているが、その時刻に根拠が無い。
`.github/workflows/update-data.yml` のコメントには「日次データが出そろった
後に」としか書いておらず、いつ出そろうのかを測っていない。

公式ドキュメント（jpx-jquants.com / jpx.gitbook.io）はこの実行環境から
到達できない（egress ブロック）。検索結果は「16:30頃」と「17:30頃」で
食い違い、しかも V1 の /prices/daily_quotes の記述が混ざる。
このリポジトリが使うのは V2 の /equities/bars/daily なので、
どちらにせよそのまま当てにはできない。だから叩いて測る。

測り方
------
当日（JST）の日付を指定して、必要なエンドポイントを一定間隔で叩き続け、
**最初に行が返ってきた時刻**を記録する。1回の実行の中でポーリングするので、
GitHub Actions のスケジュール遅延に影響されない（遅れるのは開始時刻だけで、
観測時刻は実測のまま残る）。

  $ JQUANTS_API=... python3 research/probe_update_time.py --until 21:00

出力は標準出力と、--log を渡せば追記用のCSV。
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

#: 日次の取り込みで必ず要るもの。ここが揃わないと予測が走らない。
#: (表示名, パス, パラメータの作り方, 行数を数えるときの説明)
TARGETS = [
    ("bars", "/equities/bars/daily", "date",
     "株価四本値（全銘柄）。予測の主データ"),
    ("indices", "/indices/bars/daily", "date",
     "業種別指数。sector_ret_* の元"),
    ("topix", "/indices/bars/daily/topix", "range",
     "TOPIX。地合いの特徴量の元"),
    ("master", "/equities/master", "date",
     "銘柄マスタ（市場区分・業種）"),
    ("fins", "/fins/summary", "date",
     "決算開示。当日ぶんが遅れても翌営業日に入る"),
    ("margin", "/markets/margin-interest", "date",
     "信用残。週次なので当日に出ない日が普通"),
]


def _params(kind: str, day: dt.date) -> dict:
    if kind == "range":
        return {"from": day.isoformat(), "to": day.isoformat()}
    return {"date": day.isoformat()}


def probe_once(client: JQuantsClient, day: dt.date) -> Dict[str, object]:
    """各エンドポイントについて、その日の行数を数える。"""
    out: Dict[str, object] = {}
    for name, path, kind, _ in TARGETS:
        try:
            rows = client.get_paginated(path, _params(kind, day))
        except JQuantsError as exc:
            out[name] = f"err:{str(exc)[:60]}"
            continue
        if name == "bars":
            # 行が返るだけでは足りない。前場だけ入って終値が空、という
            # 出方をされると気づかずに欠測を取り込むことになる。
            closed = sum(1 for r in rows
                         if r.get("C") not in (None, "", "-"))
            out[name] = f"{len(rows)}行/終値{closed}件"
        else:
            out[name] = f"{len(rows)}行"
    return out


def _hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="J-Quants が当日データを出す時刻を実測する")
    ap.add_argument("--date", default=None,
                    help="対象日 (YYYY-MM-DD, JST)。既定は今日")
    ap.add_argument("--start", default=None,
                    help="この時刻(JST)まで待ってから見始める。例 15:00")
    ap.add_argument("--until", default="21:00",
                    help="この時刻(JST)まで待つ。既定 21:00")
    ap.add_argument("--interval", type=int, default=300,
                    help="ポーリング間隔（秒）。既定 300")
    ap.add_argument("--log", default=None, help="観測を追記するCSV")
    args = ap.parse_args(argv)

    now = dt.datetime.now(JST)
    day = (dt.date.fromisoformat(args.date) if args.date else now.date())
    until = dt.datetime.combine(day, _hhmm(args.until), tzinfo=JST)

    key = resolve_api_key()
    if not key:
        print("[stop] JQUANTS_API が無い")
        return 1
    client = JQuantsClient(key, pause=0.2)

    print(f"対象日 {day}（JST）/ {until:%H:%M} まで {args.interval}秒ごとに見る")
    print(f"開始 {now:%Y-%m-%d %H:%M:%S} JST")
    for name, path, _, why in TARGETS:
        print(f"  {name:8s} {path:32s} {why}")
    print()

    # 対照: 直前の営業日を1回だけ見る。
    # これが返らないなら「まだ出ていない」ではなく問い合わせ方が間違っている。
    # 対照を置かずに 0行 を「未更新」と読むと、壊れた問い合わせを
    # 「データが遅い」と誤読することになる。
    prev = day - dt.timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= dt.timedelta(days=1)
    print(f"[対照] 直前の平日 {prev} を1回見る（問い合わせが正しいかの確認）")
    for k, v in probe_once(client, prev).items():
        print(f"  {k:8s} {v}")
    print()

    if args.start:
        begin = dt.datetime.combine(day, _hhmm(args.start), tzinfo=JST)
        wait = (begin - dt.datetime.now(JST)).total_seconds()
        if wait > 0:
            print(f"[wait] {begin:%H:%M} JST まで {wait/60:.0f}分待つ\n")
            time.sleep(wait)

    first_seen: Dict[str, dt.datetime] = {}
    log_rows: List[str] = []
    while True:
        t = dt.datetime.now(JST)
        seen = probe_once(client, day)
        parts = []
        for name, *_ in TARGETS:
            v = str(seen.get(name, "?"))
            parts.append(f"{name}={v}")
            # 「0行」は「まだ出ていない」と「その日は元々0件」の両方があり得る。
            # 判定は行が1件でも返ったときだけにする。
            if name not in first_seen and not v.startswith(("0行", "err:")):
                first_seen[name] = t
                print(f"  ** {name} が {t:%H:%M:%S} に出た")
        line = f"{t:%Y-%m-%d %H:%M:%S}," + ",".join(parts)
        print(line)
        log_rows.append(line)

        if len(first_seen) == len(TARGETS) or t >= until:
            break
        time.sleep(args.interval)

    print("\n=== 最初に行が返った時刻（JST） ===")
    for name, path, _, _why in TARGETS:
        when = first_seen.get(name)
        print(f"  {name:8s} {path:32s} "
              + (f"{when:%H:%M:%S}" if when else f"{args.until} までに出ず"))

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as fh:
            for line in log_rows:
                fh.write(line + "\n")
        print(f"\n[log] {args.log} に {len(log_rows)}行を追記")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
