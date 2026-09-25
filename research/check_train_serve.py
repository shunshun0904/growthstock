#!/usr/bin/env python3
"""
学習と予測で特徴量の値が一致するかを確かめる（train/serve の食い違いの検出）。

日次予測は、その日に使った特徴量を控えている（predict_daily.save_live_features、
Release の live_features.parquet）。日次予測は毎回、全期間のデータセットを
作り直すので（build_dataset --keep-unlabeled）、**過去に予測した日の行を、
今日のデータで作り直した値**と比べられる。学習に使うのはこの作り直した側。

値が違う列は「学習だけが、予測の時点では手に入らなかった値を見ている」可能性が
ある。2026-09-24 に見つかった2件（信用残を基準日で、投資部門別を集計期間の
末日で結合していた）は、どちらもここで捕まる種類の誤り。実際、9/10〜9/18 の
画面の信用倍率はすべて 8/28 の週の値だったのに、学習は同じ週の値を見ていた。

違いが出る理由は3つあり、どれも見に行く価値がある
  - 公表前の値で結合している（先読み。直す）
  - 予測の時刻より後に届いたデータ（決算の遅れなど。予測の時刻か結合の日を直す）
  - データの後からの訂正（J-Quants 側の書き換え。記録して様子を見る）

  python3 research/check_train_serve.py                  # 比べて表示
  python3 research/check_train_serve.py --max-rate 0.01  # この割合を超えた列があれば exit 1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
LIVE = os.path.join(DATA_DIR, "live_features.parquet")
REBUILT = os.path.join(DATA_DIR, "dataset_predict.parquet")

#: この件数より少ない列は判定しない（1件の違いで割合が跳ねるため）
MIN_ROWS = 30
#: 既定の許容割合。これを超えた列があれば exit 1（--max-rate で変える）
MAX_RATE = 0.01


def compare(live: pd.DataFrame, rebuilt: pd.DataFrame, cols: List[str],
            before: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    列ごとの食い違いの割合。行は (Code, Date) で突き合わせる。

    before を渡すと、その日より前の行だけを比べる（今日の行は同じ入力から
    作ったばかりなので、違わないのが当たり前）。

    控えに _features（その行で控えた列の組。predict_daily.save_live_features）が
    あれば、各列はその列を控えた行だけで比べる。モデルの列が変わると、前の行には
    新しい列が無く、ファイルの上では欠測に見えるため（比べると「予測時は欠測」の
    食い違いが並ぶ）。_features が無い控えは全部の行で比べる。
    """
    key = ["Code", "Date"]
    lv = live.copy()
    rb = rebuilt.copy()
    for d in (lv, rb):
        d["Code"] = d["Code"].astype(str)
        d["Date"] = pd.to_datetime(d["Date"])
    if before is not None:
        lv = lv[lv["Date"] < before]
    cols = [c for c in cols if c in lv.columns and c in rb.columns]
    sets = "_features" in lv.columns
    left = lv[key + cols + (["_features"] if sets else [])]
    m = left.merge(rb[key + cols], on=key, how="inner", suffixes=("_l", "_r"))
    saved = {}
    if sets:
        saved = {s: set(str(s).split(",")) for s in m["_features"].dropna().unique()}
    rows = []
    for c in cols:
        if sets:
            # _features が欠けた行（無いはずだが）は比べる側に倒す
            use = m["_features"].map(lambda s: c in saved[s] if s in saved else True)
            mc = m[use.to_numpy(dtype=bool)]
        else:
            mc = m
        a = pd.to_numeric(mc[f"{c}_l"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(mc[f"{c}_r"], errors="coerce").to_numpy(dtype=float)
        same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
        n = int(len(mc))
        bad = int((~same).sum())
        rows.append({"column": c, "n": n, "diff": bad,
                     "rate": bad / n if n else np.nan,
                     "live_nan_rebuilt_value": int((np.isnan(a) & ~np.isnan(b)).sum()),
                     "live_value_rebuilt_nan": int((~np.isnan(a) & np.isnan(b)).sum()),
                     "dates": int(mc.loc[~same, "Date"].nunique()) if bad else 0})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["rate", "diff"], ascending=False).reset_index(drop=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="学習と予測で特徴量の値が一致するか")
    ap.add_argument("--live", default=LIVE)
    ap.add_argument("--rebuilt", default=REBUILT)
    ap.add_argument("--max-rate", type=float, default=MAX_RATE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = ap.parse_args(argv)

    if not os.path.exists(args.live):
        print("[train/serve] 予測時の特徴量の控えがまだ無い（初回）。比べるものが無い")
        return 0
    if not os.path.exists(args.rebuilt):
        print(f"[train/serve] 作り直したデータセットが無い: {args.rebuilt}")
        return 0
    live = pd.read_parquet(args.live)
    rebuilt = pd.read_parquet(args.rebuilt)
    latest = pd.to_datetime(rebuilt["Date"]).max()
    cols = [c for c in live.columns if c not in ("Code", "Date") and not c.startswith("_")]
    res = compare(live, rebuilt, cols, before=latest)
    n_rows = int(res["n"].max()) if len(res) else 0
    dates = pd.to_datetime(live["Date"])
    past = dates[dates < latest]
    print(f"[train/serve] 控え {dates.nunique()}日 / 比べた行 {n_rows}件"
          f"（{past.min().date() if len(past) else '-'}〜"
          f"{past.max().date() if len(past) else '-'}。今日 {latest.date()} の行は除く）")
    if "_features" in live.columns and live["_features"].nunique() > 1:
        print(f"  控えた列の組が {live['_features'].nunique()}通り（モデルの列が変わった）。"
              "各列はその列を控えた行だけで比べる")
    if n_rows == 0:
        print("[train/serve] 比べられる過去の行がまだ無い")
        return 0
    bad = res[(res["n"] >= args.min_rows) & (res["rate"] > args.max_rate)]
    shown = res[res["diff"] > 0].head(15)
    if len(shown):
        print(f"  {'列':<28}{'違う行':>8}{'割合':>8}{'予測時は欠測':>12}{'今は欠測':>10}{'日数':>6}")
        for r in shown.itertuples(index=False):
            print(f"  {r.column:<28}{r.diff:>8}{r.rate*100:>7.1f}%"
                  f"{r.live_nan_rebuilt_value:>12}{r.live_value_rebuilt_nan:>10}{r.dates:>6}")
    else:
        print("  全列一致（予測に使った値と、学習用に作り直した値が同じ）")
    if len(bad):
        print(f"[train/serve] 許容（{args.max_rate*100:.1f}%）を超えた列が {len(bad)}本: "
              + ", ".join(bad["column"].head(10)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
