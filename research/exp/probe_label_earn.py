#!/usr/bin/env python3
"""
probe: 判定の窓（翌営業日の寄りから 20 営業日）に決算発表が入る行は、ラベル 1 の方に多いか。

運用者の質問（2026-10-17）「目的変数の定義 20日以内の決算の有無は、ラベルの 1 の方が多いですか？」に答える。
ラベルは t+1〜t+20 営業日の終値で判定する（build_dataset.attach_rise_label、基準は翌営業日の寄り）ので、
「窓に決算が入る」= 決算の開示日が t+1〜t+20 営業日にあること。2つの見方で数える。

  事後（実際）  … fins の DiscDate（実績値 Sales か NP のある開示 = 決算短信。予想修正だけの開示は除く）。
                 当日 t の開示（引け後。予測はその後に走るので特徴量は知っている）は別に数える
  事前（予定）  … 特徴量 days_to_earn（その日までに公表済みの予定日まで何暦日か）が窓の終わりの暦日以内。
                 モデルが知り得るのはこちら

ラベルは 1.2σ（本番の label 列）・1.5σ・1.8σ（build_dataset.rise_label_for_k で引き直し）。
出すのは行数・割合・差だけ。銘柄名・値は出さない（Actions のログは公開）。

  $ python3 research/exp/probe_label_earn.py [--data-dir research/_data] [--horizon 20]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import trading_calendar as TC  # noqa: E402

DATA_DIR = lab.DATA_DIR
LABEL_KS = (1.2, 1.5, 1.8)
NEED = ["Date", "Code", "label", "vol_20d", "future_rise", "end_level", "uptrend_end"]


def window_bounds(dates: np.ndarray, trading_days: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    各 t に対する判定の窓の両端（t+1 営業日と t+horizon 営業日）。datetime64[D] の配列で返す。
    営業日の並びが足りない t は NaT。
    """
    days = np.sort(np.asarray(trading_days, dtype="datetime64[D]"))
    d = np.asarray(dates, dtype="datetime64[D]")
    first = np.searchsorted(days, d, side="right")          # t より後の最初の営業日の位置
    last = first + horizon - 1
    ok = last < len(days)
    start = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[D]")
    end = start.copy()
    start[ok] = days[first[ok]]
    end[ok] = days[last[ok]]
    return start, end


def next_disclosure(samples: pd.DataFrame, disc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    行ごとに、t より後の最初の開示日（next）と、t 当日に開示があったか（same）。
    disc は Code, DiscDate の表。行の順序は samples のまま。
    """
    left = samples[["Code", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    ev = disc[["Code", "DiscDate"]].copy()
    ev["DiscDate"] = pd.to_datetime(ev["DiscDate"], errors="coerce")
    ev = ev.dropna(subset=["DiscDate"]).drop_duplicates().sort_values("DiscDate")
    ev["_d"] = ev["DiscDate"]
    fwd = pd.merge_asof(left.sort_values("Date"), ev, left_on="Date", right_on="DiscDate", by="Code",
                        direction="forward", allow_exact_matches=False).sort_values("_i")
    back = pd.merge_asof(left.sort_values("Date"), ev, left_on="Date", right_on="DiscDate", by="Code",
                         direction="backward", allow_exact_matches=True).sort_values("_i")
    nxt = fwd["_d"].to_numpy(dtype="datetime64[ns]")
    same = (back["_d"].to_numpy(dtype="datetime64[ns]") == left["Date"].to_numpy(dtype="datetime64[ns]"))
    return nxt, same


def in_window(nxt: np.ndarray, end: np.ndarray) -> np.ndarray:
    """次の開示日が窓の終わり（t+horizon 営業日）以内か。どちらかが NaT なら False。"""
    n = np.asarray(nxt, dtype="datetime64[D]")
    e = np.asarray(end, dtype="datetime64[D]")
    ok = ~(np.isnat(n) | np.isnat(e))
    out = np.zeros(len(n), dtype=bool)
    out[ok] = n[ok] <= e[ok]
    return out


def ztest(p1: float, n1: int, p0: float, n0: int) -> float:
    """2つの割合の差の z 値（行を独立とみなした目安。日付で固まるので控えめに読む）。"""
    if min(n1, n0) == 0:
        return float("nan")
    p = (p1 * n1 + p0 * n0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return float((p1 - p0) / se) if se > 0 else float("nan")


def share_table(y: np.ndarray, flags: Dict[str, np.ndarray]) -> None:
    """ラベル 1 / 0 それぞれで、旗が立つ行の割合。"""
    pos, neg = y == 1, y == 0
    print(f"  {'見方':30s}{'ラベル1':>9s}{'ラベル0':>9s}{'差(pt)':>8s}{'z':>7s}")
    for name, f in flags.items():
        p1, p0 = float(f[pos].mean()), float(f[neg].mean())
        print(f"  {name:30s}{p1 * 100:8.1f}%{p0 * 100:8.1f}%{(p1 - p0) * 100:+8.1f}{ztest(p1, int(pos.sum()), p0, int(neg.sum())):7.1f}")


def rate_table(y: np.ndarray, flags: Dict[str, np.ndarray]) -> None:
    """旗の有無ごとの正例率（ラベル 1 の割合）。"""
    print(f"  {'見方':30s}{'あり: 行':>9s}{'正例率':>8s}{'なし: 行':>9s}{'正例率':>8s}{'差(pt)':>8s}{'倍':>6s}")
    for name, f in flags.items():
        a, b = y[f], y[~f]
        ra = float(a.mean()) if len(a) else float("nan")
        rb = float(b.mean()) if len(b) else float("nan")
        ratio = ra / rb if rb > 0 else float("nan")
        print(f"  {name:30s}{len(a):9,d}{ra * 100:7.1f}%{len(b):9,d}{rb * 100:7.1f}%{(ra - rb) * 100:+8.1f}{ratio:6.2f}")


def year_table(years: np.ndarray, y: np.ndarray, f: np.ndarray) -> None:
    print(f"  {'年':6s}{'行':>8s}{'正例率':>8s}{'窓に決算: ラベル1':>16s}{'ラベル0':>9s}{'正例率: 決算あり':>16s}{'なし':>8s}")
    for yr in sorted(set(years.tolist())):
        m = years == yr
        yy, ff = y[m], f[m]
        if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
            continue
        ra = float(yy[ff].mean()) if ff.any() else float("nan")
        rb = float(yy[~ff].mean()) if (~ff).any() else float("nan")
        print(f"  {yr:<6d}{m.sum():8,d}{yy.mean() * 100:7.1f}%{ff[yy == 1].mean() * 100:15.1f}%{ff[yy == 0].mean() * 100:8.1f}%"
              f"{ra * 100:15.1f}%{rb * 100:7.1f}%")


def position_table(y: np.ndarray, nxt: np.ndarray, start: np.ndarray, end: np.ndarray,
                   trading_days: np.ndarray, horizon: int) -> None:
    """次の開示日が窓の何営業日目かで分けた正例率（窓の中だけ）。"""
    days = np.sort(np.asarray(trading_days, dtype="datetime64[D]"))
    n = np.asarray(nxt, dtype="datetime64[D]")
    s = np.asarray(start, dtype="datetime64[D]")
    ok = in_window(nxt, end)
    pos = np.full(len(n), -1)
    pos[ok] = np.searchsorted(days, n[ok], side="left") - np.searchsorted(days, s[ok], side="left") + 1
    bins = [(1, 5), (6, 10), (11, 15), (16, horizon)]
    print(f"  {'開示が窓の何日目':18s}{'行':>8s}{'正例率':>8s}")
    for lo, hi in bins:
        m = (pos >= lo) & (pos <= hi)
        if m.sum():
            print(f"  {f'{lo}〜{hi}日目':18s}{m.sum():8,d}{y[m].mean() * 100:7.1f}%")
    m = ~ok
    if m.sum():
        print(f"  {'窓に無し':18s}{m.sum():8,d}{y[m].mean() * 100:7.1f}%")


def labels_for(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    for k in LABEL_KS:
        if abs(k - float(B.DEFAULT_RISE.vol_norm_k or 0)) < 1e-9:
            out[f"{k:.1f}σ（本番）"] = pd.to_numeric(df["label"], errors="coerce")
        else:
            out[f"{k:.1f}σ"] = B.rise_label_for_k(df, k)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--horizon", type=int, default=B.DEFAULT_RISE.horizon)
    args = ap.parse_args(argv)
    h = int(args.horizon)

    ds_path = os.path.join(args.data_dir, "dataset.parquet")
    df = pd.read_parquet(ds_path)
    missing = [c for c in NEED if c not in df.columns]
    if missing:
        raise SystemExit(f"データセットに要る列がありません: {missing}")
    has_ahead = "days_to_earn" in df.columns
    df = df[NEED + (["days_to_earn"] if has_ahead else [])].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    print(f"データセット {len(df):,}行（{df['Date'].min().date()}〜{df['Date'].max().date()}）、"
          f"days_to_earn の列 {'あり' if has_ahead else 'なし'}、判定の窓 {h}営業日")

    # 営業日（取引所のカレンダー。無ければ平日で代用し、その旨を出す）
    cal = TC.load(args.data_dir)
    if cal and cal.covers(df["Date"].min().date(), (df["Date"].max() + pd.Timedelta(days=60)).date()):
        tdays = np.array(sorted(cal.days), dtype="datetime64[D]")
        print(f"営業日カレンダー {len(tdays):,}日（{cal.start}〜{cal.end}）")
    else:
        lo, hi = df["Date"].min().date(), (df["Date"].max() + pd.Timedelta(days=60)).date()
        tdays = np.array([d for d in pd.bdate_range(lo, hi).date], dtype="datetime64[D]")
        print(f"[warn] 営業日カレンダーが範囲を覆っていないので平日で代用（祝日を営業日に数える）: {len(tdays):,}日")
    start, end = window_bounds(df["Date"].to_numpy(), tdays, h)
    have_end = ~np.isnat(end)
    print(f"窓の終わりが決まる行 {int(have_end.sum()):,} / {len(df):,}、窓の暦日の長さ 中央値 "
          f"{np.nanmedian((end[have_end] - df['Date'].to_numpy(dtype='datetime64[D]')[have_end]).astype(float)):.0f}日")

    # 実際の開示日（fins）。実績値のある開示 = 決算短信
    fins = B.load_parts("fins", args.data_dir)
    fins["Code"] = fins["Code"].astype(str).str.strip()
    if fins["Code"].str.len().mode().iat[0] != df["Code"].str.len().mode().iat[0]:
        n = min(int(fins["Code"].str.len().mode().iat[0]), int(df["Code"].str.len().mode().iat[0]))
        fins["Code"] = fins["Code"].str[:n]
        df["Code"] = df["Code"].str[:n]
        print(f"[note] 銘柄コードの桁が違うので先頭 {n} 桁で合わせた")
    has_result = fins[["Sales", "NP"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    disc_res = fins.loc[has_result, ["Code", "DiscDate"]]
    disc_any = fins[["Code", "DiscDate"]]
    print(f"fins の開示 {len(disc_any):,}行、うち実績値あり {len(disc_res):,}行、"
          f"データセットの銘柄のうち fins に開示がある割合 {df['Code'].isin(set(fins['Code'])).mean() * 100:.1f}%")

    nxt_res, same_res = next_disclosure(df, disc_res)
    nxt_any, _ = next_disclosure(df, disc_any)
    post_res = in_window(nxt_res, end)
    post_any = in_window(nxt_any, end)
    # 事前（予定）: days_to_earn（暦日）が窓の終わりまでの暦日以内
    span = (end - df["Date"].to_numpy(dtype="datetime64[D]")).astype(float)
    if has_ahead:
        dte = pd.to_numeric(df["days_to_earn"], errors="coerce").to_numpy(dtype=float)
        pre = np.isfinite(dte) & have_end & (dte <= span) & (dte > 0)
        pre_known = np.isfinite(dte)
    else:
        pre = np.zeros(len(df), dtype=bool)
        pre_known = pre

    labels = labels_for(df)
    years = df["Date"].dt.year.to_numpy()
    for name, lab_s in labels.items():
        y_all = lab_s.to_numpy(dtype=float)
        m = np.isfinite(y_all) & have_end
        y = y_all[m].astype(int)
        print(f"\n■ ラベル {name}: 判定できる行 {m.sum():,}、正例 {int(y.sum()):,}（{y.mean() * 100:.2f}%）")
        flags = {
            "事後: 決算短信が窓(t+1〜t+h)に": post_res[m],
            "事後: 何らかの開示が窓に": post_any[m],
            "事後: 当日 t に決算短信": same_res[m],
        }
        if has_ahead:
            flags["事前: 公表済みの予定日が窓に"] = pre[m]
            flags["事前: 予定日が公表済み(何日でも)"] = pre_known[m]
        print("  --- 窓に決算が入る行の割合（ラベル別）")
        share_table(y, flags)
        print("  --- 決算の有無ごとの正例率")
        rate_table(y, {k: v for k, v in flags.items() if not k.startswith("事前: 予定日が公表済み")})
        if name.startswith("1.5") or "本番" in name:
            print("  --- 年ごと（事後: 決算短信が窓に）")
            year_table(years[m], y, post_res[m])
            print("  --- 開示が窓の何営業日目か（事後: 決算短信）")
            position_table(y, nxt_res[m], start[m], end[m], tdays, h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
