#!/usr/bin/env python3
"""
progress_pct が時点どおり（その日に見えていた版だけで計算した値）になっているかを測る。

2026-10-01 の学習と予測の一致チェック（check_train_serve）で progress_pct が 14行（2日ぶん）
ずれた。原因は quarterize_panel の「同じ期の開示は最後の版を採用」。あとから出し直し
（訂正）が来ると、
  (1) 元の版が順位の母集団から消え、その期より後に開示された行の順位が少し動く
  (2) その期の開示日が出し直しの日に移り、間の日の行は一つ前の期に落ちる
のどちらも、その日の予測では起きていなかったこと。ここでは版をすべて残して
  - 母集団は「その日までに出ていた最新の版」の値で持ち（出し直しが来たら差し替える）
  - 行が見る開示は「その日までに出ていた最新の期の、その日までの最新の版」
として計算し、学習データ（dataset.parquet）の値と比べる。学習側は build_dataset と同じ
関数（quarterize_panel → attach_fins）で作り直して突き合わせる（近似しない）。

実測（2026-09-26 のデータ、学習 21,855行）:
  (1) 同じ開示を見ているのに母集団が違う 14,022行。差は中央値 0.007・90% 0.026・最大 0.48
      パーセンタイル点（1点以上は0行）。チェックの比較（rtol 1e-6）には引っかかる
  (2) 学習が一つ前の期を見ている 260行（1.19%）。差は中央値 12.7 点、最大 73 点、
      片方だけ欠測 192行。学習が見ている期は、予測が見ていた期より中央値 92日古い

  python3 research/probe_progress_pit.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_dataset as B  # noqa: E402

data_dir = os.path.join(ROOT, "research", "_data")
fins = B.load_parts("fins", data_dir)
key = ["Code", "CurFYSt", "quarter"]

# --- 時点側: 版をすべて残した表 --- #
t0 = time.time()
q_map = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}
v = fins.copy()
v["quarter"] = v["CurPerType"].map(q_map)
has = v[["Sales", "OP", "NP", "EPS"]].notna().any(axis=1)
v = v[v["quarter"].notna() & has].copy()
v["quarter"] = v["quarter"].astype(int)
v["DiscDate"] = pd.to_datetime(v["DiscDate"])
v["Code"] = v["Code"].astype(str)
for c in ("OP", "FOP"):
    v[c] = pd.to_numeric(v[c], errors="coerce")
v = v.sort_values(["DiscDate", "DiscTime", "Code", "CurFYSt", "quarter"], kind="mergesort") \
     .reset_index(drop=True)
v["progress_rate"] = np.where((v["FOP"] > 0) & v["OP"].notna(), v["OP"] / v["FOP"] * 100.0, np.nan)
sp = B.seasonal_progress(v, v[["Code", "CurFYSt", "quarter", "DiscDate", "DiscTime", "OP"]].copy())
for c in sp.columns:
    v[c] = sp[c]
print(f"versions: {len(v):,} rows | periods: {v.groupby(key).ngroups:,} "
      f"| later versions: {len(v) - v.groupby(key).ngroups:,} ({time.time() - t0:.0f}s)")


def pit_percentile(v: pd.DataFrame, min_history: int = B.PROGRESS_PCT_MIN_HISTORY):
    """版をすべて残した表で、その日までの最新の版だけを母集団にして順位を付ける。"""
    ratio = v["progress_ratio"].to_numpy(dtype=float)
    quarter = v["quarter"].to_numpy()
    table = np.where(v["progress_basis"].isin(["seasonal", "floor"]).to_numpy(), "seasonal", "linear")
    days = v["DiscDate"].to_numpy()
    keys = list(zip(v["Code"], v["CurFYSt"], v["quarter"]))
    ok = np.isfinite(ratio) & np.isin(quarter, (1, 2, 3))
    out = np.full(len(v), np.nan)
    tot = np.full(len(v), np.nan)
    for qq in (1, 2, 3):
        for t in ("seasonal", "linear"):
            idx = np.where(ok & (quarter == qq) & (table == t))[0]
            if not len(idx):
                continue
            vals, rank = np.unique(ratio[idx], return_inverse=True)
            n = len(vals)
            tree = np.zeros(n + 1, dtype=np.int64)

            def add(k: int, delta: int) -> None:
                k += 1
                while k <= n:
                    tree[k] += delta
                    k += k & -k

            def below(k: int) -> int:
                s = 0
                while k > 0:
                    s += tree[k]
                    k -= k & -k
                return int(s)

            seq = idx[np.argsort(days[idx], kind="mergesort")]
            pos = {int(i): int(r) for i, r in zip(idx, rank)}
            cur = {}
            total = 0
            i = 0
            while i < len(seq):
                j = i
                while j < len(seq) and days[seq[j]] == days[seq[i]]:
                    j += 1
                if total >= min_history:
                    for s_ in seq[i:j]:
                        k = pos[int(s_)]
                        less = below(k)
                        eq = below(k + 1) - less
                        out[s_] = (less + 0.5 * eq) / total * 100.0
                        tot[s_] = total
                for s_ in seq[i:j]:
                    kk = keys[s_]
                    k = pos[int(s_)]
                    if kk in cur:
                        add(cur[kk], -1)          # 古い版を母集団から外して差し替える
                    else:
                        total += 1
                    cur[kk] = k
                    add(k, +1)
                i = j
    return out, tot


v["pct_pit"], v["tot_pit"] = pit_percentile(v)

# 版ごとに「この版が出た時点での、最新の期の最新の版」の行番号
v["_po"] = list(zip(pd.to_datetime(v["CurFYSt"], errors="coerce").astype("int64"), v["quarter"]))
eff = np.empty(len(v), dtype=np.int64)
for code, g in v.groupby("Code", sort=False):
    best, best_row = None, -1
    for row, po in zip(g.index.to_numpy(), g["_po"]):
        if best is None or po >= best:
            best, best_row = po, row
        eff[row] = best_row
v["_eff"] = eff

# --- 学習側: build_dataset と同じ関数で作り直す --- #
ds = pd.read_parquet(os.path.join(data_dir, "dataset.parquet"), columns=["Code", "Date", "progress_pct"])
ds["Code"] = ds["Code"].astype(str)
ds["Date"] = pd.to_datetime(ds["Date"])
ds["_i"] = np.arange(len(ds))
t0 = time.time()
qA = B.quarterize_panel(fins, data_dir)
tr = B.attach_fins(ds[["Code", "Date", "_i"]].copy(), qA)
tr = tr.sort_values("_i").reset_index(drop=True)
stale = (tr["Date"] - tr["DiscDate"]).dt.days > 365
tr.loc[stale, "progress_pct"] = np.nan
ok_tr = np.isclose(tr["progress_pct"], ds["progress_pct"], rtol=1e-6, atol=1e-9, equal_nan=True)
print(f"training side rebuilt with quarterize_panel/attach_fins: equals dataset.parquet in "
      f"{int(ok_tr.sum()):,} / {len(ds):,} rows ({time.time() - t0:.0f}s)")

# --- 時点側を行に付ける --- #
t = v[["Code", "DiscDate", "_eff"]].sort_values("DiscDate", kind="mergesort")
m = pd.merge_asof(ds.sort_values("Date"), t, left_on="Date", right_on="DiscDate", by="Code",
                  direction="backward", allow_exact_matches=True)
m = m.sort_values("_i").reset_index(drop=True)
seen = m["_eff"].notna()
e = m.loc[seen, "_eff"].astype(int).to_numpy()
m["pct_pit"] = np.nan
m["disc_pit"] = pd.NaT
m["q_pit"] = np.nan
m.loc[seen, "pct_pit"] = v["pct_pit"].to_numpy()[e]
m.loc[seen, "disc_pit"] = v["DiscDate"].to_numpy()[e]
m.loc[seen, "q_pit"] = v["quarter"].to_numpy()[e]
stale_pit = (m["Date"] - m["disc_pit"]).dt.days > 365
m.loc[stale_pit, "pct_pit"] = np.nan
m["pct_tr"] = tr["progress_pct"].to_numpy()
m["disc_tr"] = tr["DiscDate"].to_numpy()

same = np.isclose(m["pct_tr"], m["pct_pit"], rtol=1e-6, atol=1e-9, equal_nan=True)
diff = m[~same].copy()
same_disc = diff["disc_tr"] == diff["disc_pit"]
print(f"training rows: {len(m):,} | training != point-in-time: {len(diff):,} ({len(diff) / len(m) * 100:.2f}%)")
print(f"  (1) same disclosure, pool differs: {int(same_disc.sum()):,}")
print(f"  (2) training sees a different disclosure (older period): {int((~same_disc).sum()):,}")
d1 = diff[same_disc & diff["pct_tr"].notna() & diff["pct_pit"].notna()]
a1 = (d1["pct_tr"] - d1["pct_pit"]).abs()
print(f"  (1) |diff| percentile points: median {a1.median():.3f} p90 {a1.quantile(0.9):.3f} "
      f"max {a1.max():.3f} | >=1pt: {int((a1 >= 1).sum())}")
d2 = diff[~same_disc]
a2 = (d2["pct_tr"] - d2["pct_pit"]).abs()
print(f"  (2) |diff| percentile points (both present): median {a2.median():.2f} max {a2.max():.2f} "
      f"| one side NaN: {int((d2['pct_tr'].isna() != d2['pct_pit'].isna()).sum())}")
gap2 = (d2["disc_pit"] - d2["disc_tr"]).dt.days
print(f"  (2) days between the period training sees and the one serving saw: median {gap2.median():.0f} "
      f"p90 {gap2.quantile(0.9):.0f}")
print("  (1) by year:", {int(k): int(x) for k, x in diff[same_disc].groupby(diff.loc[same_disc, 'Date'].dt.year).size().items()})
print("  (2) by year:", {int(k): int(x) for k, x in d2.groupby(d2['Date'].dt.year).size().items()})
n1 = diff[same_disc & diff["pct_tr"].isna() & diff["pct_pit"].notna()]
n2 = diff[same_disc & diff["pct_tr"].notna() & diff["pct_pit"].isna()]
print(f"  (1) one side missing: training NaN {len(n1):,} / point-in-time NaN {len(n2):,}")
