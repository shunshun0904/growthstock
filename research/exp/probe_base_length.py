#!/usr/bin/env python3
"""
probe: 特徴量 base_length（前回の高値更新からの営業日数）の定義を実データで確かめる。

運用者の質問（2026-10-17）「前回の高値更新からの日数ですが、1000日を超えるものがあります。78週ぶりの高値更新がモデルの
母集団なので、最高でも約200日くらいなのかなと思ってますが、いまのつくり（この列の定義）はどうなっていますか？」に答える。

定義（build_dataset.add_breakout_context）: 78週高値を更新した日（is_new_high = 当日の高値 > それまでの368営業日の高値の最大）
ごとに、同じ銘柄の1つ前の更新日から何行（その銘柄の営業日）空いたか。母集団（is_fresh_break = 直前20営業日に更新が無い
更新日）の行は 21 以上。1つ前の更新日がデータの中に無ければ欠測。上限は窓（368）では決まらない、というのが読みで、
次を数えて確かめる（学習データの行 = dataset.parquet の行）:

  1. 再計算の一致   更新日の位置を数え直した値と base_length が全行一致するか（データセットの値とも）
  2. 分布           最小・分位・最大、368 / 1000 を超える行の数
  3. 抜いた高値の古さ  今回抜いた「それまでの78週高値」を付けた日から何営業日か（窓の中なので必ず 368 以下）
  4. 1000 を超える行の形  前回の更新日の高値に対する今回の高値・その間の最安値（長い下げのあとの戻りか）
  5. 行の数と営業日数  行の差が市場の営業日の差と一致するか（売買停止で行が抜けた銘柄があるか）

出すのは件数・割合・分位だけ。銘柄コード・銘柄名・値は出さない（Actions のログは公開）。

  $ python3 research/exp/probe_base_length.py [--data-dir research/_data]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR
QS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def recompute(df: pd.DataFrame) -> pd.Series:
    """更新日ごとに、同じ銘柄の1つ前の更新日から何行か（銘柄の中の位置の差。add_breakout_context とは別の書き方）。"""
    pos = df.groupby("Code", sort=False).cumcount()
    nh = df["is_new_high"].fillna(False).astype(bool)
    prev = pos[nh].groupby(df.loc[nh, "Code"], sort=False).shift(1)
    out = pd.Series(np.nan, index=df.index)
    out[nh] = (pos[nh] - prev).to_numpy(dtype=float)
    return out


def peak_age(high: np.ndarray, p: int, w: int) -> float:
    """位置 p の行が抜いた「それまでの w 行の高値の最大」を付けた行から何行か（同じ値が複数なら一番新しいもの）。"""
    win = high[max(0, p - w):p]
    if not len(win) or np.all(np.isnan(win)):
        return np.nan
    r = len(win) - 1 - int(np.argmax(win[::-1] == np.nanmax(win)))   # 最大を付けた一番新しい行（窓の中の位置）
    return float(len(win) - r)


def describe(name: str, v: pd.Series, marks: Sequence[int]) -> List[str]:
    v = pd.to_numeric(v, errors="coerce")
    ok = v.dropna()
    lines = [f"  {name}: {len(v):,}行（欠測 {int(v.isna().sum()):,}）"]
    if len(ok):
        q = ok.quantile(list(QS))
        lines.append("    分位 " + " / ".join(f"{int(k * 100)}%: {q[k]:,.0f}" for k in QS))
        for m in marks:
            n = int((ok > m).sum())
            lines.append(f"    {m:,} を超える行: {n:,}（{n / len(ok) * 100:.1f}%）")
    return lines


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args(argv)
    w = B.HIGH_WINDOW

    bars = B.load_parts("bars", args.data_dir)
    segments = B.load_market_segments(args.data_dir)
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    df = B.add_breakout_context(B.mark_new_highs(B.price_panel(bars, start=start)))
    df["Code"] = df["Code"].astype(str)
    print(f"[panel] {len(df):,}行・{df['Code'].nunique():,}銘柄。78週高値の窓 {w}行・新規のブレイクの間隔 {B.BREAKOUT_COOLDOWN}行"
          f"・更新の判定 {B.BREAKOUT_REF}（{'日中の高値' if B.BREAKOUT_ON_HIGH else '終値'}で超えたか）")

    # 1. 再計算の一致（更新日すべて）
    again = recompute(df)
    nh = df["is_new_high"].fillna(False).astype(bool)
    a, b = df.loc[nh, "base_length"], again[nh]
    same = int(((a == b) | (a.isna() & b.isna())).sum())
    print(f"■ 1. 数え直した値との一致（更新日 {int(nh.sum()):,}行）: {same:,}行が一致・食い違い {int(nh.sum()) - same:,}")

    # 学習データの行
    ds = pd.read_parquet(os.path.join(args.data_dir, "dataset.parquet"), columns=["Code", "Date", "base_length", "label"])
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    df["_pos"] = df.groupby("Code", sort=False).cumcount()
    m = ds.merge(df[["Code", "Date", "base_length", "is_fresh_break", "_pos"]], on=["Code", "Date"], how="left",
                 suffixes=("", "_panel"))
    found = int(m["_pos"].notna().sum())
    eq = int(((m["base_length"] == m["base_length_panel"]) | (m["base_length"].isna() & m["base_length_panel"].isna())).sum())
    print(f"  学習データ {len(ds):,}行: 組み直した表に見つかった {found:,}・新規のブレイクの日 "
          f"{int(m['is_fresh_break'].fillna(False).astype(bool).sum()):,}・base_length がデータセットと一致 {eq:,}")

    # 2. 分布（学習データの行）
    print("■ 2. 学習データの base_length（営業日 = その銘柄の行の数）")
    for line in describe("base_length", m["base_length"], (B.BREAKOUT_COOLDOWN, w, 500, 1000)):
        print(line)
    low = int((m["base_length"] <= B.BREAKOUT_COOLDOWN).sum())
    print(f"    {B.BREAKOUT_COOLDOWN} 以下の行（新規のブレイクなら無いはず）: {low:,}")

    # 3・4. 抜いた高値の古さと、長い行の形
    days = np.sort(pd.to_datetime(bars["Date"]).unique())
    df["_day"] = np.searchsorted(days, df["Date"].to_numpy())             # 市場の営業日の通し番号
    per = {c: (g["high"].to_numpy(dtype=float), g["low"].to_numpy(dtype=float), g["_day"].to_numpy())
           for c, g in df.groupby("Code", sort=False)}
    ages, ratio_prev, dd_prev, ratio_peak, gaps, n5 = [], [], [], [], 0, 0
    for code, p, bl in zip(m["Code"], m["_pos"], m["base_length"]):
        arrs = per.get(code)
        if arrs is None or pd.isna(p):
            ages.append(np.nan); ratio_prev.append(np.nan); dd_prev.append(np.nan); ratio_peak.append(np.nan)
            continue
        h, lo, dayn = arrs
        p = int(p)
        ages.append(peak_age(h, p, w))
        if pd.notna(bl) and 0 < int(bl) <= p and h[p - int(bl)] > 0:
            q = p - int(bl)                                          # 1つ前の更新日の位置
            ratio_prev.append(h[p] / h[q] * 100.0)
            dd_prev.append(np.nanmin(lo[q:p + 1]) / h[q] * 100.0)
            ratio_peak.append(np.nanmax(h[q + 1:p]) / h[q] * 100.0 if p - q > 1 else np.nan)
            n5 += 1
            gaps += int(dayn[p] - dayn[q] != int(bl))
        else:
            ratio_prev.append(np.nan); dd_prev.append(np.nan); ratio_peak.append(np.nan)
    m["peak_age"] = ages
    m["ratio_prev"] = ratio_prev
    m["dd_prev"] = dd_prev
    m["ratio_between"] = ratio_peak
    print("■ 3. 今回抜いた「それまでの78週高値」を付けた日から何営業日か（窓の中なので最大でも窓の長さ）")
    for line in describe("抜いた高値の古さ", m["peak_age"], (w,)):
        print(line)
    both = m.dropna(subset=["base_length", "peak_age"])
    eqp = int((both["base_length"] == both["peak_age"]).sum())
    gtp = int((both["base_length"] > both["peak_age"]).sum())
    print(f"    base_length と同じ（1つ前の更新日に付けた高値をそのまま抜いた）: {eqp:,}（{eqp / max(len(both), 1) * 100:.1f}%）"
          f" / base_length の方が長い（その間に更新の無い高値を付けて、それを抜いた）: {gtp:,}（{gtp / max(len(both), 1) * 100:.1f}%）")

    print("■ 4. base_length の長さ別の形（1つ前の更新日の高値を 100 とした中央値）")
    bins = [(B.BREAKOUT_COOLDOWN, w), (w, 1000), (1000, 10 ** 9)]
    for lo, hi in bins:
        s = m[(m["base_length"] > lo) & (m["base_length"] <= hi)]
        if s.empty:
            print(f"  {lo:,}〜{hi:,}: 0行")
            continue
        lab_hi = f"{hi:,}" if hi < 10 ** 9 else "上限なし"
        print(f"  {lo + 1:,}〜{lab_hi}: {len(s):,}行 / 今回の高値 {s['ratio_prev'].median():.0f}"
              f" / その間の最安値 {s['dd_prev'].median():.0f} / その間の最高値 {s['ratio_between'].median():.0f}"
              f" / 今回の高値が前回より低い行 {int((s['ratio_prev'] < 100).sum()):,}（{(s['ratio_prev'] < 100).mean() * 100:.0f}%）"
              f" / 正例率 {s['label'].mean() * 100:.1f}%")

    # 5. 行の数と営業日数（上の繰り返しで数えた）
    print(f"■ 5. 行の差と市場の営業日の差が違う行（売買停止などで行が抜けた銘柄）: {gaps:,} / {n5:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
