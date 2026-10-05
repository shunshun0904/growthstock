#!/usr/bin/env python3
"""
学習と予測の一致チェック（research/check_train_serve.py）で出た `rev_pct` / `rev_up` の
食い違いの原因を見る。

2026-10-05 の予測（run 37326163279）で、過去の予測日の行のうち 1 行（2.4%）だけ、予測時には
rev_pct に値があり、今日の保存データで作り直すと欠測になった（docs/MODEL_ADOPTION_RULES.md §22）。

rev_pct は「直近の業績予想の修正（DocType に EarnForecastRevision を含む開示）の FOP を、同じ事業年度
（CurFYSt）の直前の開示の FOP と比べた幅」（build_dataset.forecast_revisions）。欠測になる道は3つ:
  (a) 予測時にあった修正の開示行が、今の保存データに無い
  (b) 修正の開示行はあるが、比較相手（同じ事業年度の直前の開示）の行が無い、または FOP が無い
  (c) 予測日以前に、別の開示行（同じ事業年度）が後から増え、比較相手がそれに変わった

ここでは、控え（live_features.parquet）の該当行について、その銘柄の開示の一覧（日付・種別・事業年度・
FOP の有無だけ）を、今の保存データと、取り直す前の写し（Snapshot Raw Data、タグ pre-dedupe-20260924。
run-experiment の snapshot 入力で research/_data/snapshot/<タグ>/ に置く）で並べて比べる。
取得記録（manifest.json の fins の fetched_days）に、関係する開示日が入っているかも見る。

ログに出すのは日付・種別・件数だけ。銘柄コード・社名・予想の値・修正の向きは出さない。

  Actions: run-experiment.yml  exp=probe_rev_mismatch.py  snapshot=pre-dedupe-20260924
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "research"))
import build_dataset as B  # noqa: E402

DATA = os.path.join(ROOT, "research", "_data")
LIVE = os.path.join(DATA, "live_features.parquet")
SNAP_TAG = os.environ.get("SNAP_TAG", "pre-dedupe-20260924")
WINDOW_BACK = 450   # 開示の一覧を見る期間（暦日）。REV_CLIP（400）より少し広く
WINDOW_FWD = 3


def rev_pct_by_row(fins: pd.DataFrame) -> pd.DataFrame:
    """forecast_revisions と同じ作りで、開示行ごとの修正幅の**有無**を付けた写し。"""
    f = fins.copy()
    f["Code"] = f["Code"].astype(str)
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f.dropna(subset=["DiscDate", "Code"]).sort_values(["Code", "DiscDate"], kind="stable")
    if {"FOP", "CurFYSt"} <= set(f.columns):
        fop = pd.to_numeric(f["FOP"], errors="coerce")
        prev = fop.groupby([f["Code"], f["CurFYSt"]], sort=False).shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["_rev_pct"] = np.where(prev > 0, fop / prev * 100.0 - 100.0, np.nan)
        f["_fop_ok"] = np.isfinite(fop.to_numpy(dtype=float))
        f["_prev_ok"] = np.isfinite(prev.to_numpy(dtype=float)) & (prev.to_numpy(dtype=float) > 0)
    else:
        f["_rev_pct"] = np.nan
        f["_fop_ok"] = False
        f["_prev_ok"] = False
    return f


def listing(f: pd.DataFrame, code: str, date: pd.Timestamp) -> pd.DataFrame:
    """その銘柄の、予測日の前 WINDOW_BACK 日〜後 WINDOW_FWD 日の開示（構造だけ）。"""
    g = f[(f["Code"] == code)
          & (f["DiscDate"] >= date - pd.Timedelta(days=WINDOW_BACK))
          & (f["DiscDate"] <= date + pd.Timedelta(days=WINDOW_FWD))].copy()
    g["kind"] = g["DocType"].astype(str) if "DocType" in g.columns else "?"
    g["fy"] = pd.to_datetime(g["CurFYSt"], errors="coerce").dt.date.astype(str) if "CurFYSt" in g.columns else "?"
    g["is_rev"] = g["kind"].str.contains("EarnForecastRevision", na=False)
    # itertuples は _ で始まる列名を位置名に変えるので、ここで名前を付け直す
    g["fop_ok"] = g["_fop_ok"].astype(bool)
    g["prev_ok"] = g["_prev_ok"].astype(bool)
    g["rev_ok"] = np.isfinite(g["_rev_pct"].to_numpy(dtype=float))
    cols = ["DiscDate", "kind", "fy", "fop_ok", "prev_ok", "is_rev", "rev_ok"]
    return g[cols].sort_values("DiscDate").reset_index(drop=True)


def latest_rev_before(lst: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    r = lst[lst["is_rev"] & (lst["DiscDate"] <= date)]
    return r.iloc[-1] if len(r) else None


def print_listing(title: str, lst: pd.DataFrame, date: pd.Timestamp) -> None:
    print(f"  {title}: {len(lst)}行")
    print(f"    {'開示日':<12}{'種別':<30}{'事業年度':<12}{'FOP':>4}{'前の予想':>9}{'修正幅':>7}")
    mark = latest_rev_before(lst, date)
    for r in lst.itertuples(index=False):
        flag = " ← 予測日時点の直近の修正" if mark is not None and r.DiscDate == mark.DiscDate and r.is_rev else ""
        print(f"    {str(r.DiscDate.date()):<12}{r.kind[:28]:<30}{r.fy:<12}"
              f"{'有' if r.fop_ok else '無':>4}{'有' if r.prev_ok else '無':>9}{'有' if r.rev_ok else '無':>7}{flag}")


def diff_sets(cur: pd.DataFrame, snap: pd.DataFrame) -> Dict[str, List[str]]:
    key = lambda d: {(str(r.DiscDate.date()), r.kind, r.fy) for r in d.itertuples(index=False)}  # noqa: E731
    a, b = key(cur), key(snap)
    return {"今だけ": sorted(f"{d} {k} FY{fy}" for d, k, fy in a - b),
            "写しだけ": sorted(f"{d} {k} FY{fy}" for d, k, fy in b - a)}


def diagnose(sub: pd.DataFrame, fins: pd.DataFrame, fins_snap: Optional[pd.DataFrame],
             fetched: Optional[set]) -> int:
    """控え sub（Code, Date, rev_pct, days_since_rev, _saved_at）と今の fins を比べ、食い違いを解剖する。"""
    samples = sub[["Code", "Date"]].reset_index(drop=True)
    rebuilt = B.forecast_revisions(samples, fins)
    a = pd.to_numeric(sub["rev_pct"], errors="coerce").to_numpy(dtype=float)
    b = rebuilt["rev_pct"].to_numpy(dtype=float)
    same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
    n_bad = int((~same).sum())
    print(f"[compare] 比べた行 {len(sub)}件 / 違う行 {n_bad}件"
          f"（予測時は値あり・今は欠測 {int((np.isfinite(a) & ~np.isfinite(b)).sum())}、"
          f"予測時は欠測・今は値あり {int((~np.isfinite(a) & np.isfinite(b)).sum())}、"
          f"両方値ありで違う {int((np.isfinite(a) & np.isfinite(b) & ~same).sum())}）")
    f = rev_pct_by_row(fins)
    fs = rev_pct_by_row(fins_snap) if fins_snap is not None else None
    for i in np.where(~same)[0]:
        code = str(sub["Code"].iloc[i])
        date = pd.Timestamp(sub["Date"].iloc[i])
        saved = str(sub["_saved_at"].iloc[i])[:10] if "_saved_at" in sub.columns else "?"
        d_live = sub["days_since_rev"].iloc[i] if "days_since_rev" in sub.columns else np.nan
        d_now = rebuilt["days_since_rev"].iloc[i]
        print(f"\n■ 食い違いの行: 予測日 {date.date()}（控えた日 {saved}）")
        imp_live = (date - pd.Timedelta(days=int(d_live))).date() if np.isfinite(d_live) else None
        imp_now = (date - pd.Timedelta(days=int(d_now))).date() if np.isfinite(d_now) else None
        print(f"  予測時: rev_pct {'あり' if np.isfinite(a[i]) else '欠測'}、直近の修正からの日数 "
              f"{int(d_live) if np.isfinite(d_live) else '欠測'} → 修正の開示日 {imp_live}")
        print(f"  今    : rev_pct {'あり' if np.isfinite(b[i]) else '欠測'}、直近の修正からの日数 "
              f"{int(d_now) if np.isfinite(d_now) else '欠測'} → 修正の開示日 {imp_now}")
        cur = listing(f, code, date)
        print_listing("今の保存データの開示（この銘柄）", cur, date)
        if fs is not None:
            snap = listing(fs, code, date)
            print_listing(f"取り直す前の写し（{SNAP_TAG}）の開示", snap, date)
            d = diff_sets(cur, snap)
            print(f"  開示の出入り（開示日・種別・事業年度）: 今だけ {len(d['今だけ'])}件 {d['今だけ']} / "
                  f"写しだけ {len(d['写しだけ'])}件 {d['写しだけ']}")
            r_snap = B.forecast_revisions(samples.iloc[[i]].reset_index(drop=True), fins_snap)
            print(f"  写しの fins で作り直すと rev_pct は "
                  f"{'あり' if np.isfinite(float(r_snap['rev_pct'].iloc[0])) else '欠測'}")
        if fetched is not None:
            days = sorted({str(x.date()) for x in cur["DiscDate"]} | ({str(imp_live)} if imp_live else set()))
            print("  取得記録（fins の fetched_days）: "
                  + ", ".join(f"{d}:{'済' if d in fetched else '未'}" for d in days))
    return n_bad


def main() -> int:
    live = pd.read_parquet(LIVE)
    live["Code"] = live["Code"].astype(str)
    live["Date"] = pd.to_datetime(live["Date"])
    latest = live["Date"].max()
    has = live["_features"].astype(str).str.contains("rev_pct") if "_features" in live.columns \
        else pd.Series(True, index=live.index)
    sub = live[has & (live["Date"] < latest)].copy()
    print(f"[live] 控え {live['Date'].nunique()}日 / rev_pct を控えた過去の行 {len(sub)}件"
          f"（{sub['Date'].min().date() if len(sub) else '-'}〜{sub['Date'].max().date() if len(sub) else '-'}。"
          f"今日 {latest.date()} の行は除く）")
    fins = B.load_parts("fins", DATA)
    snap_dir = os.path.join(DATA, "snapshot", SNAP_TAG)
    fins_snap = B.load_parts("fins", snap_dir) if os.path.isdir(snap_dir) else None
    if fins_snap is None:
        print(f"[snapshot] {snap_dir} が無いので、写しとの比較は飛ばす")
    fetched = None
    mp = os.path.join(DATA, "manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            fetched = set(json.load(fh).get("fins", {}).get("fetched_days", []))
        print(f"[manifest] fins の取得記録 {len(fetched)}日")
    diagnose(sub, fins, fins_snap, fetched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
