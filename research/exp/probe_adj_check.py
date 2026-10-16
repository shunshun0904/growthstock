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
    total_changed, total_rows = 0, 0
    worst = 0.0
    recent = []
    print(f"  {'年':6s}{'行数':>12s}{'係数の列':>9s}{'係数≠1 の行':>12s}{'変わった行':>11s}{'最大相対差':>12s}")
    for p in paths:
        d = pd.read_parquet(p)
        year = os.path.basename(p)[5:9]
        has = data_store.FACTOR_COL in d.columns
        n_f = int(((pd.to_numeric(d[data_store.FACTOR_COL], errors="coerce").fillna(1.0)) != 1.0).sum()) if has else 0
        fixed, info = data_store.readjust_bars(d)
        total_changed += info["changed"]
        total_rows += info["rows"]
        worst = max(worst, info["max_rel_diff"])
        print(f"  {year:6s}{info['rows']:>12,d}{'あり' if has else 'なし':>9s}{n_f:>12,d}{info['changed']:>11,d}{info['max_rel_diff']:>12.2e}")
        if int(year) >= 2025:
            recent.append(d)
    print(f"  合計 {total_rows:,}行、作り直して AdjC が変わる行 {total_changed:,}（{total_changed / max(total_rows, 1) * 100:.3f}%）、最大相対差 {worst:.2e}")
    verdict = "一致（係数の意味は合っている）" if total_changed / max(total_rows, 1) < 0.001 else "不一致が多い（係数の意味を見直す）"
    print(f"  判定: {verdict}")

    if recent:
        b = pd.concat(recent, ignore_index=True)
        b["Date"] = pd.to_datetime(b["Date"])
        b["Code"] = b["Code"].astype(str).str.strip()
        for c in ("C", "AdjC"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
        print("\n■ 市場環境の ETF の AdjC の日次変化の最大（2025〜）")
        for label, code in B.MACRO_ETFS.items():
            s = b[b["Code"] == code].sort_values("Date").drop_duplicates("Date", keep="last")
            r = (s["AdjC"].pct_change() * 100).abs()
            print(f"  {label}（{code}）: {len(s):,}日、最大 {r.max():.1f}%（{s['Date'].iloc[int(r.values.argmax())].date() if len(s) else '-'}）")
        b26 = b[b["Date"] >= pd.Timestamp("2026-01-01")].sort_values(["Code", "Date"])
        la = np.log(b26["AdjC"]).groupby(b26["Code"]).diff()
        lc = np.log(b26["C"]).groupby(b26["Code"]).diff()
        both = (la.abs() > np.log(1.5)) & (lc.abs() > np.log(1.5)) & (np.sign(la) == np.sign(lc))
        adj_only = (la.abs() > np.log(1.5)) & (lc.abs() < np.log(1.1))
        print(f"\n■ 2026年: C と AdjC が同じ日に ±50% 超で動いた銘柄 {int(both.groupby(b26['Code']).any().sum())}件、"
              f"AdjC だけが動いた銘柄 {int(adj_only.groupby(b26['Code']).any().sum())}件（どちらも 0 が正しい）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
