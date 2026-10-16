#!/usr/bin/env python3
"""
probe: 2026年の日次バーで、調整後終値（AdjC）だけが段差を持つ日（生の終値 C は動いていない）の正体を数える。

probe_macro_etf（run 38909054969）: 2026年に AdjC が1日で ±50% 超動いた銘柄 85件のうち、2026-09-28 に 53件が集中し、
その多くは C は動いていない。J-Quants の AdjC は「その日より後の分割・併合で割り戻した値」なので、取得した時点が
違う行どうしで AdjC ÷ C の比が食い違うと、こういう段差になる（手元は差分で取り足していて過去の行を取り直さない）。
ここでは、段差の形（比の値・向き・その後に C が動くか・コードの帯）を件数で出す。銘柄名・値は出さない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR
BIG, SMALL = np.log(1.5), np.log(1.1)


def main() -> int:
    parts = []
    for y in (2025, 2026):
        p = os.path.join(DATA_DIR, f"bars_{y}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p, columns=["Date", "Code", "C", "AdjC"]))
    b = pd.concat(parts, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str).str.strip()
    b["C"] = pd.to_numeric(b["C"], errors="coerce")
    b["AdjC"] = pd.to_numeric(b["AdjC"], errors="coerce")
    b = b[b["Date"] >= pd.Timestamp("2025-07-01")].sort_values(["Code", "Date"]).reset_index(drop=True)
    g = b.groupby("Code")
    b["dA"] = np.log(b["AdjC"]).groupby(b["Code"]).diff()
    b["dC"] = np.log(b["C"]).groupby(b["Code"]).diff()
    b["ratio"] = (b["AdjC"] / b["C"]).round(4)

    adj_only = (b["dA"].abs() > BIG) & (b["dC"].abs() < SMALL)
    both = (b["dA"].abs() > BIG) & (b["dC"].abs() > BIG)
    c_only = (b["dC"].abs() > BIG) & (b["dA"].abs() < SMALL)
    print(f"日次バー {len(b):,}行（2025-07-01〜{b['Date'].max().date()}）、銘柄 {b['Code'].nunique():,}")
    print(f"段差の形（1日の変化が ±50% 超）: AdjC だけ {int(adj_only.sum())}行 / C と AdjC の両方 {int(both.sum())}行 / C だけ {int(c_only.sum())}行")

    print("\n■ AdjC だけが動いた日（日ごとの件数、多い順 8日）")
    for d, n in b.loc[adj_only, "Date"].value_counts().head(8).items():
        print(f"  {d.date()}: {int(n)}件")

    print("\n■ C だけが動いた日（= 権利落ち日。AdjC は連続 = 手元の過去の行も調整済み。多い順 8日）")
    for d, n in b.loc[c_only, "Date"].value_counts().head(8).items():
        print(f"  {d.date()}: {int(n)}件")

    # 9/28 の群の正体
    day = pd.Timestamp("2026-09-28")
    codes = sorted(set(b.loc[adj_only & (b["Date"] == day), "Code"]))
    print(f"\n■ {day.date()} に AdjC だけが動いた {len(codes)}銘柄")
    if codes:
        sub = b[b["Code"].isin(codes)]
        step = sub[sub["Date"] == day]
        print("  変化の比（AdjC 後 ÷ 前、2桁に丸め）の件数: " +
              ", ".join(f"{r:g}: {int(n)}" for r, n in np.exp(step["dA"]).round(2).value_counts().head(8).items()))
        print("  コードの先頭1桁の件数: " + ", ".join(f"{k}: {int(v)}" for k, v in step["Code"].str[0].value_counts().sort_index().items()))
        before = sub[sub["Date"] < day].groupby("Code")["ratio"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
        after = sub[sub["Date"] >= day].groupby("Code")["ratio"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        print("  AdjC ÷ C の比（前日 → 当日）の件数: " +
              ", ".join(f"{bf:g}→{af:g}: {int(n)}" for (bf, af), n in pd.DataFrame({"b": before, "a": after}).value_counts().head(8).items()))
        # その後 C が同じ向きに動く日があるか（= 分割の権利落ち日）
        later = sub[(sub["Date"] > day) & (sub["dC"].abs() > BIG)]
        n_later = later["Code"].nunique()
        print(f"  その後に C が ±50% 超動いた銘柄: {n_later}件" +
              ("（日ごと: " + ", ".join(f"{d.date()} {int(n)}件" for d, n in later.groupby("Code")["Date"].min().value_counts().sort_index().items()) + "）" if n_later else ""))
        # 当日以降、AdjC ÷ C が 1 に戻る日があるか
        back = sub[(sub["Date"] > day) & (sub["ratio"] == 1.0)].groupby("Code")["Date"].min()
        print(f"  その後 AdjC ÷ C が 1 に戻った銘柄: {len(back)}件" +
              ("（日ごと: " + ", ".join(f"{d.date()} {int(n)}件" for d, n in back.value_counts().sort_index().head(6).items()) + "）" if len(back) else ""))

    # 全体: AdjC ÷ C が 1 でない行の割合（時期ごと）
    print("\n■ AdjC ÷ C が 1 でない行の割合（月ごと）")
    m = b.groupby(b["Date"].dt.to_period("M"))
    for per, grp in m:
        r = grp["ratio"]
        print(f"  {per}: {float((r != 1.0).mean() * 100):5.2f}%（行 {len(grp):,}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
