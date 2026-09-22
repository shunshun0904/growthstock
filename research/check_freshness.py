#!/usr/bin/env python3
"""
生データが十分に新しいかを確かめる。古ければ終了コード1で止める。

日次の予測も週次の学習も、保存済みの生データ（GitHub Release）を読む。
取り込みが失敗した日に黙って古いデータで走ると、
「昨日と同じ候補が出ているのに気づかない」「先週のデータで学習し直す」
という事故になる。気づけないほうが害が大きいので、止める。

  python3 research/check_freshness.py --max-age 1

--max-age は「最後の営業日から何営業日ぶん遅れていてよいか」。
市場休日で日次バーが増えないのは正常なので、営業日で数える。

数え方は2通りある。

  カレンダー  取り込みが /markets/calendar から保存した営業日で数える。
              取引所が公表した事実なので、連休を正しく飛ばせる
  平日        カレンダーが無い・対象日を覆っていないときの控え。
              祝日を知らないので、連休では遅れを多く見積もる（厳しい側）

**カレンダーが「営業日だ」と言っている日にデータが無ければ、遅れとして
数える。** 祝日を言い訳にできるのは、取引所が非営業日だと言った日だけ。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trading_calendar  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")


def last_bar_date(data_dir: str) -> pd.Timestamp:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit(f"bars_*.parquet がありません（{data_dir}）")
    # 末尾のファイルだけ見れば足りる（年別に分かれている）
    d = pd.read_parquet(paths[-1], columns=["Date"])
    return pd.Timestamp(pd.to_datetime(d["Date"]).max())


def count_age(last: "dt.date", today: "dt.date", data_dir: str):
    """
    (遅れ, 数え方の名前) を返す。

    取り込みが保存したカレンダーがその範囲を覆っていればそれで数える。
    覆っていなければ平日で数える（祝日を知らないぶん厳しく出る側）。
    カレンダーを持ち出せるのは「取引所が非営業日と言った日」だけで、
    データが無い理由を自分で作れるわけではない。
    """
    cal = trading_calendar.load(data_dir)
    n = cal.count_between(last, today)
    if n is not None:
        return n, "カレンダー"
    return int(np.busday_count(last, today)), "平日"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生データの鮮度を確かめる")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--max-age", type=int, default=1,
                    help="許容する遅れ（営業日）。既定1＝前営業日まで")
    ap.add_argument("--as-of", default=None, help="基準日（省略時は今日）")
    args = ap.parse_args(argv)

    last = last_bar_date(args.data_dir)
    today = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.utcnow().normalize()
    age, basis = count_age(last.date(), today.date(), args.data_dir)
    print(f"[freshness] 日次バーの最終日 {last.date()} / 基準日 {today.date()} "
          f"/ 遅れ {age}営業日（{basis}で計算、許容 {args.max_age}）")
    if age > args.max_age:
        print(f"[fatal] 生データが古すぎます。Update Data Store が"
              f"失敗していないか確認してください")
        return 1
    print("[freshness] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
