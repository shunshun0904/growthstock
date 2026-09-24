#!/usr/bin/env python3
"""
保存データが API の応答を1行も落としていないかを、日を選んで叩き直して確かめる。

2026-09-24 に、年別ファイルへのマージが「同じ日・同じ銘柄」を1行に潰していて、
業種別の空売り比率の97%・大量保有報告書の15% などを捨てていたことが分かった
（docs/DATA_TIMING.md）。しかも EDA は**保存データを数えて**「API は1系列しか
返さない」と結論していて、捨てていることに気づけなかった。

ここでは保存データではなく API の応答を正として、種別ごとに数日ぶんを叩き直し、
**その日の保存データの行数**と比べる。行のキーの決め方（data_store.ROW_KEYS）の
誤り、取り込みの取りこぼし、書き戻しの失敗のどれでも数が合わなくなる。

  - 日は、過去2年の営業日から種別ごとに無作為に選ぶ（種は週ごとに変える）
  - 公表のラグの中の日（直近）は選ばない（まだ出ていないだけの日を誤検知しない）
  - まったく同じ行の重なりは1行と数える（保存も同じ扱い）
  - 出すのは件数だけ。値そのものは出さない（Actions のログは公開）
  - 足りない種別があれば exit 1

  $ JQUANTS_API=... python3 research/check_store_completeness.py --days 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import data_store  # noqa: E402
import jq_bulk as J  # noqa: E402
import trading_calendar as TC  # noqa: E402

#: 確かめる種別 -> (パス, 日付の引数, 保存の日付列)。株価は1日4,400行あるので日数を絞る
KINDS: Dict[str, Tuple[str, str, str]] = {
    "bars": ("/equities/bars/daily", "date", "Date"),
    "fins": ("/fins/summary", "date", "DiscDate"),
    **{k: (p, J.DAY_PARAM.get(k, "date"), col) for k, (p, col, _) in J.DAILY_KINDS.items()},
}
#: 公表のラグ（暦日）。これより新しい日は選ばない
RECENT_DAYS = 14


def distinct_rows(df: pd.DataFrame) -> int:
    """まったく同じ行を1行と数えた行数（入れ子の列は中身で比べる）。"""
    if not len(df):
        return 0
    return int(len(data_store._comparable(df).drop_duplicates()))


def stored_rows(data_dir: str, kind: str, date_col: str, day: dt.date) -> int:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{kind}_{day.year}.parquet")))
    if not paths:
        return 0
    d = pd.read_parquet(paths[0])
    if date_col not in d.columns:
        return 0
    got = d[pd.to_datetime(d[date_col], errors="coerce").dt.date == day]
    return distinct_rows(got)


def pick_days(days: List[dt.date], n: int, seed: int, today: dt.date) -> List[dt.date]:
    lo = today - dt.timedelta(days=730)
    hi = today - dt.timedelta(days=RECENT_DAYS)
    pool = [d for d in days if lo <= d <= hi]
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="保存データが API の応答を落としていないか")
    ap.add_argument("--data-dir", default=J.DATA_DIR)
    ap.add_argument("--days", type=int, default=5, help="種別ごとに叩き直す日数")
    ap.add_argument("--kinds", default="", help="種別（カンマ区切り。省略で全部）")
    ap.add_argument("--seed", type=int, default=None, help="省略すると ISO 週番号")
    args = ap.parse_args(argv)

    today = dt.datetime.now(J.JST).date()
    seed = args.seed if args.seed is not None else today.isocalendar().week
    cal = TC.load(args.data_dir)
    days = sorted(cal.days) if cal else []
    if not days:
        print("[stop] 営業日カレンダーが無い（research/_data/calendar.parquet）")
        return 1
    kinds = [k for k in (args.kinds.split(",") if args.kinds else KINDS) if k]
    client = J.JQuantsClient(J.resolve_api_key(), pause=0.2)

    bad = []
    print(f"{'種別':<12}{'日':>12}{'API':>8}{'保存':>8}{'足りない':>10}")
    for kind in kinds:
        path, param, col = KINDS[kind]
        n_days = 2 if kind == "bars" else args.days
        for day in pick_days(days, n_days, seed + len(kind), today):
            try:
                rows = client.get_paginated(path, {param: day.isoformat()})
            except J.JQuantsError as exc:
                print(f"{kind:<12}{day.isoformat():>12}  問い合わせ失敗: {str(exc)[:60]}")
                continue
            api = distinct_rows(J._sanitize(pd.DataFrame.from_records(rows))) if rows else 0
            got = stored_rows(args.data_dir, kind, col, day)
            short = api - got
            mark = "  <-" if short > 0 else ""
            print(f"{kind:<12}{day.isoformat():>12}{api:>8}{got:>8}{short:>10}{mark}")
            if short > 0:
                bad.append((kind, day, short))
    if bad:
        kinds_bad = sorted({k for k, _, _ in bad})
        print(f"\n[NG] 保存データが API の応答より少ない種別: {', '.join(kinds_bad)}"
              f"（{len(bad)}日）。取り込みの行のキー（data_store.ROW_KEYS）か取りこぼしを疑う")
        return 1
    print("\n[OK] 選んだ日はすべて、保存データが API の応答と同じ行数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
