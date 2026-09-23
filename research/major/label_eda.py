#!/usr/bin/env python3
"""
本ブレイク予測モデルのラベル（案）と、その正例・負例のチャート用データ。

運用者の依頼（2026-09-23）
  「数カ月や年単位保有前提の、本ブレイク予測モデルを別（別ブランチ）で開発しようと
   思います。60営業日後に+50%かつ、120営業日後に+100%（買値の2倍）になっている、
   条件でラベルを作ってみてほしいです。まずは、正例の割合と、正例と不例のチャートを
   みしてほしいです。…チャートの時系列の区間は2.5年あれば良いと思います。」
  続けて「厳密に、その日付でなくとも、それより以前に一度でも到達していればよしと
   します。買うのは、新高値を更新した次の日の寄り付き始値を想定していますが、
   さすがにその後3営業日以内の到達のような（超短期での急騰のような）、特殊な事例は
   除いてほしいです」
  さらに「終値の判定で良いです。急騰は不例にも入れてください。というより、運用の際に
   予測対象銘柄に事前ガードレールを入れるつもりはないので」
  さらに「やっぱり当日到達以外は入れてください。もちろん終値ベースです」

定義
  母集団  今のモデルと同じ。78週高値の更新日（場中の高値で判定。直前20営業日に
          更新が無いもの）。research/_data/dataset.parquet の行
  買値    翌営業日の寄り（AdjO[t+1]）。買った日を1日目と数える
  正例    60営業日以内に一度でも終値が 買値+50% 以上、かつ
          120営業日以内に一度でも終値が 買値の2倍 以上
  急騰    買った当日（1日目）の終値で 買値+50% に届いたものは負例。2日目以降に届いたものは
          ふつうに数える。運用では予測する銘柄を前もって絞らないので、急騰も学習から外さない
  判定    120営業日先まで上場している行だけ（途中で上場廃止した行は判定できないので外す）
  終値で測るのは、場中に一瞬触れただけを到達にしないため（今のモデルのラベルと同じ考え方）。
  参考に、場中の高値で測った場合の件数も出す

チャート
  ブレイク日 t の 486営業日前 〜 125営業日後（612営業日 ≒ 2.5年）の終値。
  78週高値の判定区間（368営業日）と、120営業日目がどちらも入る。
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
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402

RISE_1, DAYS_1 = 0.50, 60      # 60営業日以内に +50%
RISE_2, DAYS_2 = 1.00, 120     # 120営業日以内に 2倍
SPIKE_DAYS = 1                 # この日数以内（1 = 買った当日）に +50% に届いたものは急騰として負例にする
BEFORE = 486                   # チャートの左端（ブレイク日から何営業日前か）
AFTER = 125                    # チャートの右端（ブレイク日から何営業日後か）
N_RANDOM = 24                  # ランダムに選ぶ負例の数
N_NEAR = 12                    # 惜しい負例（片方だけ届いた）をそれぞれ何件選ぶか
SEED = 7


def first_reach(P: np.ndarray, level: float, days: int) -> np.ndarray:
    """
    P[:, k] = k+1 日目の値を買値で割ったもの（1日目 = 買った日）。
    days 日以内に初めて level 以上になった日（1始まり）。届かなければ NaN。
    欠測の日（売買が無い日）は届かなかった扱い。
    """
    with np.errstate(invalid="ignore"):
        hit = P[:, :days] >= level
    first = hit.argmax(axis=1) + 1.0
    return np.where(hit.any(axis=1), first, np.nan)


def major_label(P: np.ndarray) -> dict:
    """
    P = 買った日からの値（終値など）÷ 買値。列は1日目から少なくとも DAYS_2 日ぶん。
    戻り値
      y      正例 1 / 負例 0。急騰（spike）は両方に届いていても負例
      first1 +50% に初めて届いた日（DAYS_1 日以内。届かなければ NaN）
      first2 2倍 に初めて届いた日（DAYS_2 日以内）
      spike  SPIKE_DAYS 日以内に +50% に届いた（超短期の急騰）
      reach  急騰かどうかを問わず、両方に届いた
    """
    f1 = first_reach(P, 1.0 + RISE_1, DAYS_1)
    f2 = first_reach(P, 1.0 + RISE_2, DAYS_2)
    reach = np.isfinite(f1) & np.isfinite(f2)
    with np.errstate(invalid="ignore"):
        spike = f1 <= SPIKE_DAYS
    y = (reach & ~spike).astype(float)
    return {"y": y, "first1": f1, "first2": f2, "spike": spike, "reach": reach}


def load_bars(extra: tuple = ()) -> pd.DataFrame:
    """日次の足（分割調整済みの寄り・高値・終値）。extra で列を足せる（例: ("AdjL",)）。"""
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    cols = ["Date", "Code", "AdjO", "AdjH", "AdjC", *extra]
    b = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"]).reset_index(drop=True)
    if b.duplicated(["Code", "Date"]).any():
        raise SystemExit("bars に (Code, Date) の重複がある")
    return b


def windows(b: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """
    keys の各行（基準日 t）について、t-BEFORE 〜 t+AFTER の終値・高値と、
    t+1 の寄り（買値）、t+DAYS_2 まで上場しているか、それまでの78週高値を返す。
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
    out = {"step": step}
    for name, col in (("C", "AdjC"), ("H", "AdjH")):
        a = b[col].to_numpy(dtype=float)[idx]
        a[~ok] = np.nan
        out[name] = a
    o = b["AdjO"].to_numpy(dtype=float)
    out["entry"] = np.where(left[i0] >= 1, o[np.minimum(i0 + 1, len(o) - 1)], np.nan)
    out["listed"] = left[i0] >= DAYS_2          # t+DAYS_2 の行がある（判定できる）
    # それまでの78週高値（当日を含まない。build_dataset と同じ窓の長さ）
    h = b["AdjH"].to_numpy(dtype=float)
    w = B.HIGH_WINDOW
    prior = np.full(len(i0), np.nan)
    for k, i in enumerate(i0):
        lo = i - w
        if first[i] >= w and code[lo] == code[i]:
            prior[k] = np.nanmax(h[lo:i])
    out["prior_high"] = prior
    return out


def forward_closes(b: pd.DataFrame, keys: pd.DataFrame, days: int = DAYS_2) -> dict:
    """
    keys の各行（基準日 t）について、1〜days 日目（t+1 〜 t+days）の終値と、
    買値（t+1 の寄り）、t+days まで上場しているか（判定できるか）を返す。
    windows() の前後2.5年ぶんは要らない学習用の軽い版。
    """
    pos = b[["Code", "Date"]].assign(_i=np.arange(len(b)))
    i0 = keys[["Code", "Date"]].merge(pos, on=["Code", "Date"], how="left")["_i"]
    if i0.isna().any():
        raise SystemExit(f"bars に無い基準日が {int(i0.isna().sum())}件")
    i0 = i0.to_numpy(dtype=np.int64)
    left = b.groupby("Code", sort=False).cumcount(ascending=False).to_numpy()[i0]
    step = np.arange(1, days + 1)
    ok = step[None, :] <= left[:, None]
    idx = np.where(ok, i0[:, None] + step[None, :], 0)
    C = b["AdjC"].to_numpy(dtype=float)[idx]
    C[~ok] = np.nan
    o = b["AdjO"].to_numpy(dtype=float)
    entry = np.where(left >= 1, o[np.minimum(i0 + 1, len(o) - 1)], np.nan)
    return {"C": C, "entry": entry, "listed": left >= days}


def label_frame(df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    df の行（Code, Date）に本ブレイクのラベルを付ける。
    y_major は 1 / 0。120営業日先まで上場していない行（判定できない行）は NaN。
    """
    W = forward_closes(bars, df)
    L = major_label(W["C"] / W["entry"][:, None])
    det = W["listed"] & np.isfinite(W["entry"])
    return pd.DataFrame({"Code": df["Code"].to_numpy(), "Date": df["Date"].to_numpy(),
                         "y_major": np.where(det, L["y"], np.nan),
                         "first1": L["first1"], "first2": L["first2"], "spike": L["spike"] & det})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="", help="チャート用 JSON の出力先（省略なら書かない）")
    args = ap.parse_args(argv)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    bars = load_bars()
    W = windows(bars, df)
    C, e = W["C"], W["entry"]
    fwd = slice(BEFORE + 1, BEFORE + 1 + DAYS_2)                          # 1〜120日目
    Lc = major_label(C[:, fwd] / e[:, None])
    Lh = major_label(W["H"][:, fwd] / e[:, None])
    det = W["listed"] & np.isfinite(e)
    keep = det                      # 急騰も負例として残す（運用で予測する銘柄を前もって絞らない）
    y = Lc["y"]

    print("=== 1. 件数（終値で測る。一度でも届けばよい）===")
    print(f"  母集団（78週高値の更新日）: {len(df):,}件 / 120営業日先まで上場していて判定できる: {int(det.sum()):,}件"
          f"（買った日 {df.loc[det, 'Date'].min().date()}〜{df.loc[det, 'Date'].max().date()}）")
    sp = det & Lc["spike"]
    print(f"  急騰（{SPIKE_DAYS}日目までに終値が +50%。1日目 = 買った当日）: {int(sp.sum())}件。負例にする"
          f"（そのうち正例の条件も満たしていたもの {int((sp & Lc['reach']).sum())}件）")
    n, npos = int(keep.sum()), int(y[keep].sum())
    print(f"  {n:,}件のうち 正例 {npos}件 / {npos / n * 100:.2f}%"
          f"（銘柄の数 {df.loc[keep & (y == 1), 'Code'].nunique()}）")
    h1, h2 = np.isfinite(Lc["first1"]), np.isfinite(Lc["first2"])
    print(f"  60営業日以内に +50%: {h1[keep].mean()*100:.2f}% / 120営業日以内に 2倍: {h2[keep].mean()*100:.2f}%")
    near1 = keep & h1 & ~h2 & ~Lc["spike"]
    near2 = keep & h2 & ~h1
    print(f"  +50% は届いたが 2倍 に届かず: {int(near1.sum())}件"
          f" / 2倍 には届いたが 60日以内の +50% に届かず（出足が遅い）: {int(near2.sum())}件")
    d = df.loc[keep, ["Code", "Date"]].assign(y=y[keep])
    yr = d.groupby(d["Date"].dt.year).agg(n=("y", "size"), pos=("y", "sum"))
    yr["rate"] = yr["pos"] / yr["n"] * 100
    print("  年ごと（買った日の年）")
    for y_, r in yr.iterrows():
        print(f"    {y_}: {int(r['n']):>6,}件 正例 {int(r['pos']):>3}件 {r['rate']:>5.2f}%")

    print("\n=== 2. 参考 ===")
    keep_h = det
    print(f"  場中の高値で測ると: 急騰 {int((det & Lh['spike']).sum())}件 / 正例 {int(Lh['y'][keep_h].sum())}件"
          f" / {Lh['y'][keep_h].mean()*100:.2f}%")
    old = ((df["ret_o1_60"] >= RISE_1) & (df["ret_o1_120"] >= RISE_2)).to_numpy()
    print(f"  前の定義（60・120営業日目の5日平均がそれぞれ届いている）: 正例 {int(old[det].sum())}件"
          f" / {old[det].mean()*100:.2f}%")
    pm = df["break_margin"].to_numpy(dtype=float)
    bm_prior = df["close"].to_numpy(dtype=float) / (1 + pm / 100.0)
    rel = np.abs(W["prior_high"] / bm_prior - 1)
    print(f"  確認: 78週高値（チャートに引く線）と build_dataset の値の差 中央値 {np.nanmedian(rel)*100:.3f}%"
          f" / 99%点 {np.nanpercentile(rel, 99)*100:.3f}%")

    pos = keep & (y == 1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 判定できない行は全部 NaN
        mx = np.nanmax(C[:, fwd] / e[:, None], axis=1)
    print("\n=== 3. 正例の中身（終値）===")
    print(f"  初めて +50% に届いた日 中央値 {np.median(Lc['first1'][pos]):.0f}営業日目 /"
          f" 初めて2倍に届いた日 中央値 {np.median(Lc['first2'][pos]):.0f}営業日目")
    print(f"  120営業日のうちの最高値（終値）中央値 {np.nanmedian(mx[pos]):.2f}倍")

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
    # 日付の表示用に、市場の営業日の並び（どれかの銘柄に行がある日）と、ブレイク日がその何番目かを渡す。
    # 銘柄の行は上場中の営業日が抜けなく並ぶ前提（行の抜けている銘柄が6つあるが、出す銘柄には無い）
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))
    rng = np.random.default_rng(SEED)
    groups = {
        "pos": np.flatnonzero(pos),
        "near1": rng.permutation(np.flatnonzero(near1))[:N_NEAR],
        "near2": rng.permutation(np.flatnonzero(near2))[:N_NEAR],
        "neg": rng.permutation(np.flatnonzero(keep & (y == 0)))[:N_RANDOM],
        "spike": np.flatnonzero(sp),
    }

    def num(x):
        return None if not np.isfinite(x) else float(x)

    def series(i: int) -> dict:
        v = C[i] / e[i]
        return {
            "code": str(df.at[i, "Code"])[:4], "name": names.get(df.at[i, "Code"], ""),
            "date": df.at[i, "Date"].strftime("%Y-%m-%d"),
            "ci": int(cal.get_loc(df.at[i, "Date"])),
            "f1": num(Lc["first1"][i]), "f2": num(Lc["first2"][i]), "mx": round(float(mx[i]), 4),
            "prior": round(float(W["prior_high"][i] / e[i]), 4) if np.isfinite(W["prior_high"][i]) else None,
            "v": [None if not np.isfinite(x) else int(round(x * 1000)) for x in v],
        }

    out = {"before": BEFORE, "after": AFTER, "window": B.HIGH_WINDOW,
           "rule": {"rise1": RISE_1, "days1": DAYS_1, "rise2": RISE_2, "days2": DAYS_2, "spike": SPIKE_DAYS},
           "cal": [x.strftime("%Y-%m-%d") for x in cal],
           "groups": {k: [series(int(i)) for i in ix] for k, ix in groups.items()}}
    Pn = C / e[:, None]
    band = {}
    for k, m in (("pos", pos), ("neg", keep & (y == 0))):
        with np.errstate(invalid="ignore"):
            q = np.nanpercentile(Pn[m], [25, 50, 75], axis=0)
        band[k] = {"q25": np.round(q[0], 4).tolist(), "q50": np.round(q[1], 4).tolist(),
                   "q75": np.round(q[2], 4).tolist(), "n": int(m.sum())}
    out["band"] = band
    out["summary"] = {
        "det": int(det.sum()), "n": n, "pos": npos, "codes": int(df.loc[pos, "Code"].nunique()),
        "rate": npos / n, "r1": float(h1[keep].mean()), "r2": float(h2[keep].mean()),
        "near1": int(near1.sum()), "near2": int(near2.sum()), "neg": int((keep & (y == 0)).sum()),
        "spike": int(sp.sum()), "spike_pos": int((sp & Lc["reach"]).sum()),
        "high_pos": int(Lh["y"][keep_h].sum()), "high_rate": float(Lh["y"][keep_h].mean()),
        "old_pos": int(old[det].sum()),
        "from": df.loc[det, "Date"].min().strftime("%Y-%m-%d"), "to": df.loc[det, "Date"].max().strftime("%Y-%m-%d"),
        "years": [{"year": int(y_), "n": int(r["n"]), "pos": int(r["pos"])} for y_, r in yr.iterrows()],
        "first1": float(np.median(Lc["first1"][pos])), "first2": float(np.median(Lc["first2"][pos])),
        "mx": float(np.nanmedian(mx[pos])),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n  チャート用データ: {args.out}（{os.path.getsize(args.out)/1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
