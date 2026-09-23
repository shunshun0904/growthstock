#!/usr/bin/env python3
"""
本ブレイク予測モデルのラベル（案）と、その正例・負例のチャート用データ。

運用者の依頼（2026-09-23）
  「数カ月や年単位保有前提の、本ブレイク予測モデルを別（別ブランチ）で開発しようと
   思います。60営業日後に+50%かつ、120営業日後に+100%（買値の2倍）になっている、
   条件でラベルを作ってみてほしいです。まずは、正例の割合と、正例と不例のチャートを
   みしてほしいです。…チャートの時系列の区間は2.5年あれば良いと思います。」

定義
  母集団  今のモデルと同じ。78週高値の更新日（場中の高値で判定。直前20営業日に
          更新が無いもの）。research/_data/dataset.parquet の行
  買値    翌営業日の寄り（AdjO[t+1]）
  売値    60 / 120 営業日後の5日平均終値（lab.realized_returns の ret_o1_60 /
          ret_o1_120 と同じ物差し。1日だけの値だと、たまたまの上下で結果が変わる）
  正例    ret_o1_60 >= +50% かつ ret_o1_120 >= +100%
  参考    5日平均ではなく、その日の終値1日で測った場合の正例率も出す

チャート
  ブレイク日 t の 486営業日前 〜 125営業日後（612営業日 ≒ 2.5年）の終値。
  78週高値の判定区間（368営業日）と、120営業日後の判定日がどちらも入る。
  値は買値で割る（買値 = 1.0。+50% = 1.5、2倍 = 2.0）。

使い方
  python research/major/label_eda.py --out <チャート用 JSON の出力先>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402

RISE_60 = 0.50            # 60営業日後に +50%
RISE_120 = 1.00           # 120営業日後に +100%（2倍）
BEFORE = 486              # チャートの左端（ブレイク日から何営業日前か）
AFTER = 125               # チャートの右端（ブレイク日から何営業日後か）。120日目の5日平均まで入る
N_RANDOM = 24             # ランダムに選ぶ負例の数
N_NEAR = 12               # 惜しい負例（片方だけ届いた）をそれぞれ何件選ぶか
SEED = 7


def major_label(r60: pd.Series, r120: pd.Series) -> pd.Series:
    """正例 = 60営業日後 +50% 以上 かつ 120営業日後 +100% 以上。どちらかが欠測なら NaN。"""
    ok = r60.notna() & r120.notna()
    y = ((r60 >= RISE_60) & (r120 >= RISE_120)).astype(float)
    return y.where(ok)


def load_bars() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    cols = ["Date", "Code", "AdjO", "AdjH", "AdjC"]
    b = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"]).reset_index(drop=True)
    if b.duplicated(["Code", "Date"]).any():
        raise SystemExit("bars に (Code, Date) の重複がある")
    return b


def windows(b: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """
    keys の各行（基準日 t）について、t-BEFORE 〜 t+AFTER の終値と、
    t+1 の寄り（買値）、t+60 / t+120 の終値、それまでの78週高値を返す。
    銘柄の上場前・データ末尾で足りない所は NaN。
    """
    pos = b[["Code", "Date"]].assign(_i=np.arange(len(b)))
    i0 = keys[["Code", "Date"]].merge(pos, on=["Code", "Date"], how="left")["_i"]
    if i0.isna().any():
        raise SystemExit(f"bars に無い基準日が {int(i0.isna().sum())}件")
    i0 = i0.to_numpy(dtype=np.int64)
    code = b["Code"].to_numpy()
    first = b.groupby("Code", sort=False).cumcount().to_numpy()          # 銘柄内の位置
    left = b.groupby("Code", sort=False).cumcount(ascending=False).to_numpy()
    step = np.arange(-BEFORE, AFTER + 1)
    idx = i0[:, None] + step[None, :]
    ok = (step[None, :] >= -first[i0][:, None]) & (step[None, :] <= left[i0][:, None])
    idx = np.where(ok, idx, 0)
    C = b["AdjC"].to_numpy(dtype=float)[idx]
    C[~ok] = np.nan
    o = b["AdjO"].to_numpy(dtype=float)
    entry = np.where(left[i0] >= 1, o[np.minimum(i0 + 1, len(o) - 1)], np.nan)
    # それまでの78週高値（当日を含まない。build_dataset と同じ窓の長さ）
    h = b["AdjH"].to_numpy(dtype=float)
    w = B.HIGH_WINDOW
    prior = np.full(len(i0), np.nan)
    for k, i in enumerate(i0):
        lo = max(i - w, i - first[i])
        if i - lo >= w and code[lo] == code[i]:
            prior[k] = np.nanmax(h[lo:i])
    return {"C": C, "step": step, "entry": entry, "prior_high": prior}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="", help="チャート用 JSON の出力先（省略なら書かない）")
    args = ap.parse_args(argv)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["y"] = major_label(df["ret_o1_60"], df["ret_o1_120"])
    d = df[df["y"].notna()].reset_index(drop=True)
    pos = d["y"] == 1
    print("=== 1. 正例率（5日平均終値で測る。今のモデルの ret_o1_60 / ret_o1_120 と同じ物差し）===")
    print(f"  母集団（78週高値の更新日）で120営業日先まで判定できる行: {len(d):,}件"
          f"（買った日 {d['Date'].min().date()}〜{d['Date'].max().date()}）")
    print(f"  正例: {int(pos.sum())}件 / {pos.mean()*100:.2f}%  （銘柄の数 {d.loc[pos, 'Code'].nunique()}）")
    print(f"  60営業日後 +50% 以上: {(d['ret_o1_60'] >= RISE_60).mean()*100:.2f}%"
          f" / 120営業日後 +100% 以上: {(d['ret_o1_120'] >= RISE_120).mean()*100:.2f}%")
    a_only = (d["ret_o1_60"] >= RISE_60) & (d["ret_o1_120"] < RISE_120)
    b_only = (d["ret_o1_60"] < RISE_60) & (d["ret_o1_120"] >= RISE_120)
    print(f"  60日だけ届いた（120日で2倍に届かず）: {int(a_only.sum())}件"
          f" / 120日だけ届いた（60日で+50%に届かず）: {int(b_only.sum())}件")
    yr = d.groupby(d["Date"].dt.year).agg(n=("y", "size"), pos=("y", "sum"))
    yr["rate"] = yr["pos"] / yr["n"] * 100
    print("  年ごと（買った日の年）")
    for y_, r in yr.iterrows():
        print(f"    {y_}: {int(r['n']):>6,}件 正例 {int(r['pos']):>3}件 {r['rate']:>5.2f}%")

    bars = load_bars()
    W = windows(bars, d)
    C, step, e = W["C"], W["step"], W["entry"]
    j60, j120 = BEFORE + 60, BEFORE + 120
    c60 = C[:, j60] / e - 1.0
    c120 = C[:, j120] / e - 1.0
    yc = ((c60 >= RISE_60) & (c120 >= RISE_120)).astype(float)
    yc[~(np.isfinite(c60) & np.isfinite(c120))] = np.nan
    print("\n=== 2. 参考: その日の終値1日で測った場合 ===")
    print(f"  正例 {int(np.nansum(yc))}件 / {np.nanmean(yc)*100:.2f}%。5日平均の正例と一致しないもの"
          f" {int(np.nansum(np.abs(yc - d['y'].to_numpy())))}件")
    pm = d["break_margin"].to_numpy(dtype=float)
    bm_prior = d["close"].to_numpy(dtype=float) / (1 + pm / 100.0)
    rel = np.abs(W["prior_high"] / bm_prior - 1)
    print(f"  確認: 78週高値（チャートに引く線）と build_dataset の値の差 中央値 {np.nanmedian(rel)*100:.3f}%"
          f" / 99%点 {np.nanpercentile(rel, 99)*100:.3f}%")

    # 正例の中身（何日目に届いたか、120日間の最大）
    P = C[pos.to_numpy()] / e[pos.to_numpy(), None]
    fwd = P[:, BEFORE + 1:BEFORE + 121]
    first50 = np.argmax(fwd >= 1.5, axis=1) + 1
    first100 = np.where((fwd >= 2.0).any(axis=1), np.argmax(fwd >= 2.0, axis=1) + 1, np.nan)
    print("\n=== 3. 正例の中身（終値ベース）===")
    print(f"  初めて +50% に届いた日 中央値 {np.median(first50):.0f}営業日目 /"
          f" 初めて2倍に届いた日 中央値 {np.nanmedian(first100):.0f}営業日目")
    print(f"  120営業日の最高値（終値）中央値 {np.nanmedian(np.nanmax(fwd, axis=1)):.2f}倍")

    if not args.out:
        return 0

    # --- チャート用データ --- #
    # 名前は過去の銘柄一覧（上場廃止した銘柄も入っている）の最後の名前。今の一覧があればそちらを使う
    hist = [pd.read_parquet(f, columns=["Date", "Code", "CoName"])
            for f in sorted(glob.glob(os.path.join(lab.DATA_DIR, "master_hist_*.parquet")))]
    names = {}
    if hist:
        h = pd.concat(hist, ignore_index=True).sort_values("Date")
        names = dict(zip(h["Code"], h["CoName"]))          # 後の日付で上書き = 最後の名前
    master = pd.read_parquet(os.path.join(lab.DATA_DIR, "master.parquet"), columns=["Code", "CoName"])
    names.update(zip(master["Code"], master["CoName"]))
    rng = np.random.default_rng(SEED)
    groups = {
        "pos": np.flatnonzero(pos.to_numpy()),
        "near60": rng.permutation(np.flatnonzero(a_only.to_numpy()))[:N_NEAR],
        "near120": rng.permutation(np.flatnonzero(b_only.to_numpy()))[:N_NEAR],
        "neg": rng.permutation(np.flatnonzero((d["y"] == 0).to_numpy()))[:N_RANDOM],
    }

    # 日付の表示用に、市場の営業日の並び（どれかの銘柄に行がある日）と、
    # ブレイク日がその何番目かを渡す。銘柄の行は上場中の営業日が抜けなく並ぶ前提
    # （行が抜けている銘柄はチャートに出す分には無いことを確かめてある）
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))

    def series(i: int) -> dict:
        v = C[i] / e[i]
        return {
            "code": str(d.at[i, "Code"])[:4], "name": names.get(d.at[i, "Code"], ""),
            "date": d.at[i, "Date"].strftime("%Y-%m-%d"),
            "ci": int(cal.get_loc(d.at[i, "Date"])),
            "r60": round(float(d.at[i, "ret_o1_60"]), 4), "r120": round(float(d.at[i, "ret_o1_120"]), 4),
            "prior": round(float(W["prior_high"][i] / e[i]), 4) if np.isfinite(W["prior_high"][i]) else None,
            "v": [None if not np.isfinite(x) else int(round(x * 1000)) for x in v],
        }

    out = {"before": BEFORE, "after": AFTER, "window": B.HIGH_WINDOW,
           "cal": [x.strftime("%Y-%m-%d") for x in cal],
           "groups": {k: [series(int(i)) for i in ix] for k, ix in groups.items()}}
    # 全件の形（中央値と四分位）。正例と負例で分ける
    Pn = C / e[:, None]
    band = {}
    for k, m in (("pos", pos.to_numpy()), ("neg", (d["y"] == 0).to_numpy())):
        with np.errstate(invalid="ignore"):
            q = np.nanpercentile(Pn[m], [25, 50, 75], axis=0)
        band[k] = {"q25": np.round(q[0], 4).tolist(), "q50": np.round(q[1], 4).tolist(),
                   "q75": np.round(q[2], 4).tolist(), "n": int(m.sum())}
    out["band"] = band
    out["summary"] = {
        "n": int(len(d)), "pos": int(pos.sum()), "codes": int(d.loc[pos, "Code"].nunique()),
        "rate": float(pos.mean()), "r60": float((d["ret_o1_60"] >= RISE_60).mean()),
        "r120": float((d["ret_o1_120"] >= RISE_120).mean()),
        "a_only": int(a_only.sum()), "b_only": int(b_only.sum()),
        "close_rate": float(np.nanmean(yc)), "close_pos": int(np.nansum(yc)),
        "from": d["Date"].min().strftime("%Y-%m-%d"), "to": d["Date"].max().strftime("%Y-%m-%d"),
        "years": [{"year": int(y_), "n": int(r["n"]), "pos": int(r["pos"])} for y_, r in yr.iterrows()],
        "first50": float(np.median(first50)), "first100": float(np.nanmedian(first100)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n  チャート用データ: {args.out}（{os.path.getsize(args.out)/1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
