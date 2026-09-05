#!/usr/bin/env python3
"""
シャード別 artifact (bars_2017-01-01_2017-12-31.parquet 等) を
年別の保存形式 (bars_2017.parquet) に変換し、manifest を作る。

差分取得の仕組みを入れる前に走らせた取得の結果を捨てずに済ませるための一度きりの移行。
取得済みの日付は、実データに含まれる日付から復元する。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATE_COL = {"bars": "Date", "fins": "DiscDate", "margin": "Date", "topix": "Date"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="artifact を年別の保存形式へ移行する")
    ap.add_argument("--in-dir", default=os.path.join(HERE, "_incoming"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "_data"))
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = data_store.load_manifest(args.out_dir)

    for kind, date_col in DATE_COL.items():
        paths = sorted(glob.glob(os.path.join(args.in_dir, f"{kind}_*.parquet")))
        if not paths:
            print(f"[{kind}] 入力なし")
            continue

        frames = []
        for p in paths:
            df = pd.read_parquet(p)
            if len(df):
                frames.append(df)
            print(f"[{kind}] {os.path.basename(p)}: {len(df):,}行")
        if not frames:
            continue

        allrows = pd.concat(frames, ignore_index=True)
        written = data_store.merge_into_years(args.out_dir, kind, allrows, date_col)

        # 取得済みの日付を実データから復元する。
        # bars/topix は「その日のデータがある = その日を取得した」で正しい。
        # fins/margin は開示が無い日もあるため、bars の日付範囲で補う。
        got = pd.to_datetime(allrows[date_col]).dt.date
        days = sorted(set(got))
        data_store.mark_fetched(manifest, kind, days)
        print(f"[{kind}] -> {len(written)}ファイル / 日付 {days[0]} 〜 {days[-1]} ({len(days)}日)")

    # fins / margin は「開示が無い日」も取得済みとして扱う。
    # そうしないと、開示ゼロの日を毎回叩き直すことになる。
    bars_days = data_store.fetched_days(manifest, "bars")
    if bars_days:
        lo, hi = min(bars_days), max(bars_days)
        for kind in ("fins", "margin"):
            have = data_store.fetched_days(manifest, kind)
            if not have:
                continue
            # bars の範囲内で、その種別がカバーしているとみなせる期間を埋める
            covered = {d for d in bars_days if min(have) <= d <= max(have)}
            added = covered - have
            if added:
                data_store.mark_fetched(
                    manifest, kind, [dt.date.fromisoformat(d) for d in added])
                print(f"[{kind}] 開示0件の日 {len(added)}日を取得済みとして記録 "
                      f"(範囲 {min(have)} 〜 {max(have)})")
        print(f"[bars] 全体範囲: {lo} 〜 {hi}")

    data_store.save_manifest(args.out_dir, manifest)
    print("\n[manifest]")
    print(data_store.summarize(data_store.load_manifest(args.out_dir)))

    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out_dir, "*.parquet")))
    print(f"\n[done] 合計 {total/1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
