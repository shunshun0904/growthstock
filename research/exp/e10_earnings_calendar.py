#!/usr/bin/env python3
"""
実験10: 決算日程を特徴量にする。

なぜこれか
---------
実験04〜05 で分かったのは、日付定数の特徴量（market 11列）が優位の大半を
担っていて、銘柄側の140列では実収益のエッジがほぼ出ないということだった。
足すべきは「もう1つの局面ダイヤル」ではなく、**銘柄ごとに値が変わるもの**。

決算日程はその条件を満たす。しかも新規のデータ取得が要らない
（fins_*.parquet の DiscDate / CurPerType が既にある）。

なお現状のデータセットには DiscDate が残っていない（財務の数値だけ結合して
日付は落としている）ので、ここでは実験用に別途結合する。効くと分かってから
build_dataset.py に入れる。

作る特徴量
---------
  days_since_disc   直近の開示からの経過日数
  disc_quarter      直近開示の会計期（1Q=1 / 2Q=2 / 3Q=3 / FY=4）
  disc_gap_est      その銘柄の開示間隔の中央値（過去のみから推定）
  days_to_next_est  推定次回開示日までの残日数
  post_earn_5       開示から5営業日以内か
  pre_earn_5        推定次回開示まで5営業日以内か
  n_disc_90d        過去90暦日の開示件数（修正・予想変更が多いか）

先読みを避ける
-------------
・直近開示は merge_asof の backward で引く。既存の財務結合と同じ作法
  （allow_exact_matches=True、by=Code）に揃える。ここを変えると、
  既にデータセットに入っている財務数値と時点がずれる。
・**次回開示日は実測値を使わない**。実際の次回 DiscDate を引くと未来を見る。
  その銘柄の過去の開示間隔の中央値から推定する。企業の決算周期は
  事前に分かる情報なので、運用でも同じものが作れる。
・開示間隔の中央値も「その時点までの開示」だけから累積で計算する。
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
Q_MAP = {"1Q": 1.0, "2Q": 2.0, "3Q": 3.0, "FY": 4.0}
NEW = ["days_since_disc", "disc_quarter", "disc_gap_est", "days_to_next_est",
       "post_earn_5", "pre_earn_5", "n_disc_90d"]


def disclosure_table() -> pd.DataFrame:
    """
    開示の一覧を作る。1銘柄1日に複数開示があれば最後のものを残す。

    過去のみから推定する量（開示間隔の中央値・過去90日の件数）を
    ここで一緒に作る。expanding を使うので、各行はその行までの
    開示しか見ていない。
    """
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "fins_*.parquet")))
    if not paths:
        raise SystemExit("fins_*.parquet がありません")
    cols = ["Code", "DiscDate", "DiscTime", "CurPerType"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths],
                  ignore_index=True)
    d["DiscDate"] = pd.to_datetime(d["DiscDate"])
    d = (d.sort_values(["Code", "DiscDate", "DiscTime"])
          .drop_duplicates(["Code", "DiscDate"], keep="last")
          .reset_index(drop=True))
    d["disc_quarter"] = d["CurPerType"].map(Q_MAP)

    g = d.groupby("Code", sort=False)
    gap = g["DiscDate"].diff().dt.days
    # その行までの間隔の中央値。shift しないと自分の間隔を含むが、
    # 自分の間隔は「直前の開示から今回まで」なので既に観測済み。先読みではない
    d["disc_gap_est"] = (gap.groupby(d["Code"]).expanding().median()
                         .reset_index(level=0, drop=True))
    # 過去90暦日の開示件数。自分を含めて数える
    d["n_disc_90d"] = np.nan
    for code, sub in d.groupby("Code", sort=False):
        s = pd.Series(1.0, index=sub["DiscDate"].to_numpy())
        d.loc[sub.index, "n_disc_90d"] = (
            s.rolling("90D").sum().to_numpy())
    return d[["Code", "DiscDate", "disc_quarter", "disc_gap_est", "n_disc_90d"]]


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """データセットに決算日程の特徴量を付ける。"""
    disc = disclosure_table()
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date")
    disc = disc.sort_values("DiscDate")
    out = pd.merge_asof(out, disc, left_on="Date", right_on="DiscDate",
                        by="Code", direction="backward",
                        allow_exact_matches=True)

    out["days_since_disc"] = (out["Date"] - out["DiscDate"]).dt.days
    # 1年以上前の開示は「直近」と呼べない。既存の財務結合も1年で切っている
    stale = out["days_since_disc"] > 365
    for c in ("days_since_disc", "disc_quarter", "disc_gap_est", "n_disc_90d"):
        out.loc[stale, c] = np.nan

    # 推定次回開示日までの残日数。実測の次回 DiscDate は使わない
    gap = out["disc_gap_est"].clip(lower=30, upper=200)
    out["days_to_next_est"] = gap - out["days_since_disc"]
    out["post_earn_5"] = (out["days_since_disc"] <= 5).astype(float)
    out["pre_earn_5"] = ((out["days_to_next_est"] >= 0)
                         & (out["days_to_next_est"] <= 5)).astype(float)
    # 欠損のときフラグを1にしない
    out.loc[out["days_since_disc"].isna(), ["post_earn_5", "pre_earn_5"]] = np.nan
    return out


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    df = attach(df)

    print("=== 作った特徴量の素性 ===")
    print(f"  {'列':<20}{'欠損率':>8}{'中央値':>10}{'5%':>9}{'95%':>9}")
    for c in NEW:
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"  {c:<20}{s.isna().mean()*100:>7.1f}%{s.median():>10.2f}"
              f"{s.quantile(.05):>9.2f}{s.quantile(.95):>9.2f}")
    print()
    print("=== 経過日数と成績（素の関係）===")
    b = pd.cut(df["days_since_disc"], [-1, 5, 20, 45, 70, 95, 400],
               labels=["0-5日", "6-20日", "21-45日", "46-70日", "71-95日", "96日+"])
    t = df.groupby(b, observed=True).agg(件数=("label", "size"),
                                         正例率=("label", "mean"),
                                         実収益=("ref_end", "mean"))
    for k, r in t.iterrows():
        print(f"  {str(k):<10}{int(r['件数']):>7,}件  正例率 {r['正例率']*100:>5.1f}%"
              f"  実収益 {r['実収益']*100:>+6.2f}%")
    print()

    runs = {
        "base": cols,
        "+calendar": cols + NEW,
        # 経過日数だけ。いちばん素直な1列で足りるのかを見る
        "+days_only": cols + ["days_since_disc"],
    }
    results = {}
    for name, use in runs.items():
        p = os.path.join(OOF_DIR, f"cal_{name.replace('+', 'p')}.parquet")
        if os.path.exists(p):
            o = pd.read_parquet(p)
            results[name] = lab.Result(name, o, lab.metrics(o))
            print(f"  {name:<12} 保存済みを読む")
            continue
        r = lab.run_multi(df, lambda s: lab.lgbm(seed=s), seeds=SCREEN_SEEDS,
                          cols=use, name=name)
        r.oof.to_parquet(p, index=False)
        results[name] = r
        m = r.metrics
        print(f"  {name:<12} 完了（{len(use)}列） しきい値優位 {m['thr_lift']:+.2f}pt"
              f" / 窓平均 {m['thr_fold_mean']:+.2f}pt / t値 {m['thr_t']:+.2f}")

    print()
    print(lab.table(results))
    print()
    print("=== 判定（ノイズ床: しきい値優位 0.51pt）===")
    ref = results["base"].metrics
    for name, r in results.items():
        if name == "base":
            continue
        m = r.metrics
        d = m["thr_lift"] - ref["thr_lift"]
        v = "採用可" if d > 0.51 else ("要確認" if d > 0.25 else "ノイズ内")
        print(f"  {name:<12} 優位の差 {d:+.2f}pt / 窓平均の差 "
              f"{m['thr_fold_mean'] - ref['thr_fold_mean']:+.2f}pt / "
              f"t値 {m['thr_t']:+.2f} / 最悪の窓 {m['thr_worst']:+.2f}pt  -> {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
