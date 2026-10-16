#!/usr/bin/env python3
"""
probe: 市場環境の ETF（日経225 = 13210、金 = 15400、グロース250 = 25160）の調整後終値（AdjC）に段差が無いか。

実験100（run 38907694687）で、実運用の行の nk225_ret_120（日経225 ETF の120日リターン）が −98.8% だった（過去の最小は
−17.1%）。日次バーは差分で取り足しているので、分割・併合の後に J-Quants が過去の AdjC を付け直しても、手元の過去の行は
取り直していない。段差があればその日を特定する。あわせて、全銘柄で同じ形の段差（1日で ±50% を超える AdjC の変化）が
いくつあるかを数える（件数だけ。銘柄は出さない）。

公開ログには日付・変化率・件数だけ出す（終値の水準は出さない）。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR
YEARS = (2025, 2026)


def load_bars(years=YEARS) -> pd.DataFrame:
    parts = []
    for y in years:
        p = os.path.join(DATA_DIR, f"bars_{y}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    for c in ("C", "AdjC"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def steps(s: pd.Series, dates: pd.Series, k: int = 3) -> list:
    """日次の変化率の絶対値が大きい順に k 件（日付と変化率 %）。"""
    r = s.pct_change() * 100
    idx = r.abs().sort_values(ascending=False).head(k).index
    return [(pd.Timestamp(dates.loc[i]).date(), float(r.loc[i])) for i in idx]


def main() -> int:
    bars = load_bars()
    print(f"日次バー {len(bars):,}行 {bars['Date'].min().date()}〜{bars['Date'].max().date()} 列: {list(bars.columns)}")
    has_adj = "AdjC" in bars.columns
    for label, code in B.MACRO_ETFS.items():
        sub = bars[bars["Code"] == code].sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
        print(f"\n■ {label}（{code}）: {len(sub):,}日 {sub['Date'].min().date() if len(sub) else '-'}〜{sub['Date'].max().date() if len(sub) else '-'}")
        if sub.empty:
            continue
        for col in (["C", "AdjC"] if has_adj else ["C"]):
            print(f"  {col} の日次変化の大きい順: " + ", ".join(f"{d} {v:+.1f}%" for d, v in steps(sub[col], sub["Date"])))
        if has_adj:
            ratio = (sub["AdjC"] / sub["C"]).round(6)
            chg = ratio.ne(ratio.shift()).to_numpy()
            chg[0] = False
            n_ratio_changes = int(chg.sum())
            print(f"  AdjC ÷ C が変わった日: {n_ratio_changes}日"
                  + ("" if not n_ratio_changes else "（" + ", ".join(str(d.date()) for d in sub.loc[chg, "Date"].head(5)) + "…）"))
        s = sub.set_index("Date")["AdjC" if has_adj else "C"]
        r20, r120 = s.pct_change(20) * 100, s.pct_change(120) * 100
        tail = s.index[s.index >= pd.Timestamp("2026-09-01")]
        print("  ret_20 / ret_120（9月以降、週ごとの最終日）:")
        wk = pd.Series(tail).groupby(pd.Series(tail).dt.to_period("W-SUN")).max()
        for d in wk:
            print(f"    {d.date()}  ret_20 {r20.loc[d]:+7.1f}%  ret_120 {r120.loc[d]:+7.1f}%")

    # 全銘柄の段差（分割・併合の未調整の目安）。件数だけ
    if has_adj:
        b26 = bars[bars["Date"] >= pd.Timestamp("2026-01-01")].sort_values(["Code", "Date"])
        lr = np.log(b26["AdjC"]).groupby(b26["Code"]).diff()
        big = lr.abs() > np.log(1.5)
        huge = lr.abs() > np.log(3.0)
        per_code = big.groupby(b26["Code"]).any()
        per_code_h = huge.groupby(b26["Code"]).any()
        print(f"\n■ 2026年の全銘柄: AdjC が1日で ±50% を超えて動いた銘柄 {int(per_code.sum())}件 / {per_code.size:,}銘柄"
              f"（3倍以上・1/3以下 {int(per_code_h.sum())}件）。日ごとの件数（多い順 5日）:")
        by_day = b26.loc[big.fillna(False).to_numpy(), "Date"].value_counts().head(5)
        for d, n in by_day.items():
            print(f"    {d.date()}: {int(n)}件")
        # 同じ銘柄で C も同じ日に同じ向きに動いていれば「生の値の段差 = 分割・併合」。AdjC だけなら付け直しの食い違い
        lr_c = np.log(b26["C"]).groupby(b26["Code"]).diff()
        both = (big & (lr_c.abs() > np.log(1.5))).groupby(b26["Code"]).any()
        print(f"  そのうち C（生の終値）も同じ日に同じ大きさで動いた銘柄 {int(both.sum())}件（= 分割・併合が AdjC に未調整の疑い）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
