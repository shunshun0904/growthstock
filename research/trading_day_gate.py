#!/usr/bin/env python3
"""
今日が非営業日で、かつデータが揃っているなら、日次予測を飛ばす。

なぜ要るか
--------
鮮度チェック（check_freshness.py）は遅れを**平日**で数える。祝日カレンダーを
持ち込んでいないのは意図的で、「更新が止まったのに祝日だからと自分で
言い訳する」余地を作らないため。

その代わり、連休では区別がつかなくなる。日次予測の許容は1営業日なので、
3連休（例 2026-09-21〜23）だと火・水は「遅れ2〜3営業日」で止まる。
止まること自体は正しいが、**正常な休場と本物の取り込み障害が同じ赤い×
として出る**。毎回の連休で赤を見ることになり、本物を見逃す訓練になる。

そこで、営業日かどうかを**推測せずに受け取る**。営業日カレンダーを引いて
いるのは取り込み（jq_bulk.py）だけなので、そこが manifest に事実として
書き残す（`calendar` ブロック / jq_bulk.calendar_note）。ここはそれを読む
だけで、自分でカレンダーを判断しない。

飛ばす条件（3つすべて）
--------------------
1. `calendar.isTradingDay` が **明示的に False**
   （None＝判断材料なし、キー無し＝古い manifest。どちらも飛ばさない）
2. 保存データの最終バー日 == `calendar.lastTradingDay`
   **ここが肝**。取り込みが壊れていれば保存データは直近の営業日に
   届いていないので、休場日であっても飛ばさず、鮮度チェックで落とす。
3. manifest が十分に新しい（既定12時間以内）
   取り込みが今日走っていないなら、その calendar は昨日以前の判断。

どれか1つでも欠ければ「通常どおり進む」。**飛ばす側に倒さない。**

止めない
------
判断できない事情（manifest が読めない、バーが無い、時刻が壊れている）は
すべて「通常どおり進む」に倒し、**必ず exit 0 で返す**。ここが新しい
失敗の入口になってはいけない。止めるかどうかは鮮度チェックの仕事。

  $ python3 research/trading_day_gate.py
  -> 標準出力に判定、$GITHUB_OUTPUT に run=true|false
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from typing import Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
#: manifest がこれより古ければ「今日の判断ではない」とみなす
STALE_HOURS = 12


def last_bar_date(data_dir: str) -> Optional[dt.date]:
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        return None
    d = pd.read_parquet(paths[-1], columns=["Date"])
    if not len(d):
        return None
    return pd.Timestamp(pd.to_datetime(d["Date"]).max()).date()


def decide(manifest: dict, last_bar: Optional[dt.date],
           now: dt.datetime, stale_hours: int = STALE_HOURS
           ) -> Tuple[bool, str]:
    """
    (予測を走らせるか, 理由) を返す。判断できなければ走らせる側に倒す。
    """
    cal = manifest.get("calendar")
    if not isinstance(cal, dict):
        return True, "manifest に営業日カレンダーが無い（古い取り込み）"

    if cal.get("isTradingDay") is not False:
        v = cal.get("isTradingDay")
        if v is True:
            return True, f"{cal.get('asOfJst')} は営業日"
        return True, "今日が営業日かを取り込み側が判断していない"

    updated = manifest.get("updatedAt")
    if not updated:
        return True, "manifest に updatedAt が無い"
    try:
        ts = dt.datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
    except ValueError:
        return True, f"updatedAt を読めない（{updated}）"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    age_h = (now - ts).total_seconds() / 3600
    if age_h > stale_hours:
        return True, (f"取り込みが {age_h:.1f}時間前で古い"
                      f"（{stale_hours}時間以内でないと非営業日と判断しない）")

    ltd = cal.get("lastTradingDay")
    if not ltd:
        return True, "直近の営業日が記録されていない"
    if last_bar is None:
        return True, "保存データの最終バー日が分からない"
    if last_bar.isoformat() != ltd:
        return True, (f"保存データの最終バー {last_bar} が"
                      f"直近の営業日 {ltd} に届いていない（取り込みを疑う）")

    return False, (f"{cal.get('asOfJst')} は非営業日で、"
                   f"直近の営業日 {ltd} まで揃っている")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="非営業日なら日次予測を飛ばす（判断できなければ進む）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--stale-hours", type=int, default=STALE_HOURS)
    args = ap.parse_args(argv)

    run, why = True, "判定できなかったので通常どおり進む"
    try:
        path = os.path.join(args.data_dir, "manifest.json")
        manifest = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        run, why = decide(manifest, last_bar_date(args.data_dir),
                          dt.datetime.now(dt.timezone.utc), args.stale_hours)
    except Exception as exc:                      # noqa: BLE001
        # 判断できない理由が何であれ、止めずに通常どおり進む。
        # ここを失敗にすると、連休対応のための部品が新しい障害になる
        print(f"[warn] 判定に失敗したので通常どおり進む: "
              f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)

    if run:
        print(f"[gate] 予測を実行する — {why}")
    else:
        print(f"[gate] 予測を飛ばす — {why}")
        print("       休場日なので新しい候補は出ない。画面は前営業日のまま。")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        # 理由はそのまま workflow の shell に埋め込まれる。改行は
        # $GITHUB_OUTPUT の形式を壊し、二重引用符は shell を壊すので落とす
        safe = " ".join(str(why).split()).replace('"', "'")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"run={'true' if run else 'false'}\n")
            fh.write(f"reason={safe}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
