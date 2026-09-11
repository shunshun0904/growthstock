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
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")


def last_bar_date(data_dir: str) -> pd.Timestamp:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit(f"bars_*.parquet がありません（{data_dir}）")
    # 末尾のファイルだけ見れば足りる（年別に分かれている）
    d = pd.read_parquet(paths[-1], columns=["Date"])
    return pd.Timestamp(pd.to_datetime(d["Date"]).max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生データの鮮度を確かめる")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--max-age", type=int, default=1,
                    help="許容する遅れ（営業日）。既定1＝前営業日まで")
    ap.add_argument("--as-of", default=None, help="基準日（省略時は今日）")
    args = ap.parse_args(argv)

    last = last_bar_date(args.data_dir)
    today = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.utcnow().normalize()
    # 東証の営業日そのものではなく平日で数える。祝日ぶんは --max-age の余裕で吸収する。
    # ここで祝日カレンダーを持ち込むと、更新が止まったときに
    # 「祝日だから」と自分で言い訳する余地を作ってしまう
    age = int(np.busday_count(last.date(), today.date()))
    print(f"[freshness] 日次バーの最終日 {last.date()} / 基準日 {today.date()} "
          f"/ 遅れ {age}営業日（許容 {args.max_age}）")
    if age > args.max_age:
        print(f"[fatal] 生データが古すぎます。Update Data Store が"
              f"失敗していないか確認してください")
        return 1
    print("[freshness] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
