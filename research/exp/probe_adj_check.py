#!/usr/bin/env python3
"""
probe: 日次バーの調整後の値が、素の値と分割・併合の係数（AdjFactor）から作り直した値と一致するか（検算）。

全期間の取り直し（2026-10-16、運用者の決定 §61）の後に回す。J-Quants が付けた AdjC と、手元の作り直し
（data_store.readjust_bars: その日より後の係数の積を素の値に掛ける）が一致すれば、係数の意味（権利落ち日の行に入り、
その日より前の株価に掛ける）が合っていて、これ以後の分割も手元で付け直せる。一致しない行が多ければ意味が違う。

出す: 年ごとの行数・係数の列の有無・係数が 1 でない行数・作り直して AdjC が変わった行数（相対差 1e-6 超）・最大の相対差、
市場環境の ETF 3本の AdjC の日次変化の最大（段差が消えたか）、2026 年に C と AdjC が同じ日に ±50% 超で動いた銘柄数
（未調整の段差。0 になるはず）。銘柄名・値は出さない。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import data_store  # noqa: E402
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR


def main() -> int:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "bars_*.parquet")))
    print(f"日次バーのファイル {len(paths)}件")
    # 係数はその日より後の全期間の分を掛けるので、年のファイルごとではなく全期間を一度に作り直して比べる
    # （初回 run 38934573452 は年ごとに作り直していて、翌年以降の分割の分が「不一致」に数えられた）
    parts = []
    for p in paths:
        d = pd.read_parquet(p, columns=["Date", "Code", "C", "AdjC", "Vo", "AdjVo"] +
                            ([data_store.FACTOR_COL] if data_store.FACTOR_COL in pd.read_parquet(p).columns else []))
        d["_year"] = int(os.path.basename(p)[5:9])
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str).str.strip()
    for c in ("C", "AdjC"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    has = data_store.FACTOR_COL in b.columns
    fixed, info = data_store.readjust_bars(b)
    before = b["AdjC"].to_numpy(dtype=float)
    after = pd.to_numeric(fixed["AdjC"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(before) & np.isfinite(after) & (before != 0)
    rel = np.where(ok, np.abs(after - before) / np.abs(np.where(ok, before, 1.0)), 0.0)
    changed = rel > 1e-6
    f = pd.to_numeric(b[data_store.FACTOR_COL], errors="coerce").fillna(1.0) if has else pd.Series(1.0, index=b.index)
    print(f"  {'年':6s}{'行数':>12s}{'係数≠1 の行':>12s}{'変わった行':>11s}{'最大相対差':>12s}")
    for y, g in b.groupby("_year"):
        idx = g.index.to_numpy()
        print(f"  {y:<6d}{len(g):>12,d}{int((f.loc[idx] != 1.0).sum()):>12,d}{int(changed[idx].sum()):>11,d}{rel[idx].max():>12.2e}")
    n_changed = int(changed.sum())
    print(f"  合計 {len(b):,}行、係数の列 {'あり' if has else 'なし'}、係数のある銘柄 {info['codes_with_factor']:,}、"
          f"全期間で作り直して AdjC が変わる行 {n_changed:,}（{n_changed / max(len(b), 1) * 100:.4f}%）、最大相対差 {rel.max():.2e}")
    print("  判定: " + ("一致（保存した調整後の値は係数から作り直した値と同じ）" if n_changed / max(len(b), 1) < 0.001
                     else "不一致が多い（係数の意味か取り込みを見直す）"))

    recent = b[b["Date"] >= pd.Timestamp("2025-01-01")].copy()
    print("\n■ 市場環境の ETF の AdjC の日次変化の最大（2025〜）")
    for label, code in B.MACRO_ETFS.items():
        s = recent[recent["Code"] == code].sort_values("Date").drop_duplicates("Date", keep="last")
        r = (s["AdjC"].pct_change() * 100).abs()
        print(f"  {label}（{code}）: {len(s):,}日、最大 {r.max():.1f}%（{s['Date'].iloc[int(r.values.argmax())].date() if len(s) else '-'}）")
    b26 = recent[recent["Date"] >= pd.Timestamp("2026-01-01")].sort_values(["Code", "Date"])
    la = np.log(b26["AdjC"]).groupby(b26["Code"]).diff()
    lc = np.log(b26["C"]).groupby(b26["Code"]).diff()
    both = (la.abs() > np.log(1.5)) & (lc.abs() > np.log(1.5)) & (np.sign(la) == np.sign(lc))
    adj_only = (la.abs() > np.log(1.5)) & (lc.abs() < np.log(1.1))
    both_codes = set(b26.loc[both.fillna(False).to_numpy(), "Code"])
    # C と AdjC が同じ日に動いた銘柄に、その前後1日に係数（≠1）があるか（あれば係数の付け方が1日ずれている疑い）
    near = 0
    if has:
        fb = pd.to_numeric(b26[data_store.FACTOR_COL], errors="coerce").fillna(1.0)
        hit_days = b26.loc[both.fillna(False).to_numpy(), ["Code", "Date"]]
        fac_days = b26.loc[(fb != 1.0).to_numpy(), ["Code", "Date"]]
        for code, d in hit_days.itertuples(index=False):
            fd = fac_days[fac_days["Code"] == code]["Date"]
            if len(fd) and (fd - d).abs().min() <= pd.Timedelta(days=4):
                near += 1
    print(f"\n■ 2026年: C と AdjC が同じ日に ±50% 超で動いた銘柄 {len(both_codes)}件（そのうち前後4日に係数のある行 {near}件。"
          f"0 なら係数と無関係の値動き）、AdjC だけが動いた銘柄 {int(adj_only.groupby(b26['Code']).any().sum())}件（0 が正しい）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
