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

1回目（2026-10-05、run 37367494496）の結果: 該当の銘柄は 2026-07-22 に業績予想の修正を**同じ日に2件**
出していて（取り直す前の写しでは1行に潰れていた）、同じ事業年度の最初の予想がその2件目なので、
1件目は「前の予想なし」で欠測、2件目は「1件目との比」で値あり。forecast_revisions は修正の行を
`sort_values("DiscDate")`（1列・quicksort = 安定でない）で並べてから merge_asof（backward）で
**同じ日の最後の行**を取るので、同じ日の2行のどちらが最後になるかは並べ替えの偶然で決まり、
日によって値あり・欠測が入れ替わる。2回目はそれを確かめる:
  - 同じ (Code, DiscDate) に修正が2行以上ある日の数と、予想の値が違うか（件数だけ）
  - 学習用データセットで、直近の修正がそういう日に当たる行の数
  - fins の行の順番を変えるだけで、該当行の rev_pct が 有⇄欠測 に入れ替わること

  Actions: run-experiment.yml  exp=probe_rev_mismatch.py  snapshot=pre-dedupe-20260924

2026-10-06 に直した（運用者の決定で案 1: 開示日・開示時刻・開示番号の安定な並べ替え。
build_dataset.forecast_revisions の DISC_ORDER）。いま回すと order_flip は入れ替わらない。
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
        # 同じ日の修正が2行以上なら、その行どうしで違う列の名前だけを出す（値は出さない）
        if imp_now is not None:
            same_day = f[(f["Code"] == code) & (f["DiscDate"] == pd.Timestamp(imp_now))
                         & f["DocType"].astype(str).str.contains("EarnForecastRevision", na=False)]
            if len(same_day) >= 2:
                cols = [c for c in same_day.columns if not c.startswith("_")]
                differ = [c for c in cols if same_day[c].astype(str).nunique() > 1]
                print(f"  {imp_now} の修正は {len(same_day)}行。行どうしで違う列: {differ}")
        order_flip(sub.iloc[[i]][["Code", "Date"]], fins)
    return n_bad


def same_day_duplicates(f: pd.DataFrame) -> pd.DataFrame:
    """(Code, DiscDate) で業績予想の修正（EarnForecastRevision）が2行以上ある日。FOP の値が何通りかも数える。"""
    rev = f[f["DocType"].astype(str).str.contains("EarnForecastRevision", na=False)].copy()
    rev["_fop"] = pd.to_numeric(rev["FOP"], errors="coerce") if "FOP" in rev.columns else np.nan
    g = rev.groupby(["Code", "DiscDate"]).agg(n=("DiscDate", "size"), n_fop=("_fop", "nunique"))
    return g[g["n"] >= 2]


def population(f: pd.DataFrame, dataset_path: str) -> None:
    """同じ日に2件以上の修正がある日の数と、学習用データセットでそれに当たる行の数（件数だけ）。"""
    rev = f[f["DocType"].astype(str).str.contains("EarnForecastRevision", na=False)]
    dup = same_day_duplicates(f)
    print(f"\n[population] 修正の行 {len(rev):,} / (銘柄, 開示日) {rev.groupby(['Code', 'DiscDate']).ngroups:,}"
          f" / 同じ日に2行以上 {len(dup):,}（うち FOP の値が違う {int((dup['n_fop'] >= 2).sum()):,}、"
          f"同じか欠測 {int((dup['n_fop'] < 2).sum()):,}）")
    if len(dup):
        by_year = dup.reset_index()["DiscDate"].dt.year.value_counts().sort_index()
        print("  年別: " + ", ".join(f"{y}:{n}" for y, n in by_year.items()))
    if not os.path.exists(dataset_path):
        print(f"  学習用データセットが無い: {dataset_path}")
        return
    ds = pd.read_parquet(dataset_path, columns=["Code", "Date", "days_since_rev", "rev_pct"])
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    has = ds["days_since_rev"].notna() & (ds["days_since_rev"] < B.REV_CLIP)
    d = ds[has].copy()
    d["_rev_d"] = d["Date"] - pd.to_timedelta(d["days_since_rev"].astype(int), unit="D")
    keys = set(map(tuple, dup.reset_index()[["Code", "DiscDate"]].astype({"Code": str}).itertuples(index=False)))
    hit = np.fromiter(((c, dd) in keys for c, dd in zip(d["Code"], d["_rev_d"])), dtype=bool, count=len(d))
    nan_rate_hit = float(d.loc[hit, "rev_pct"].isna().mean()) if hit.any() else float("nan")
    nan_rate_other = float(d.loc[~hit, "rev_pct"].isna().mean()) if (~hit).any() else float("nan")
    print(f"  学習用データセット {len(ds):,}行 / 直近の修正がある行 {len(d):,} / そのうち修正が同じ日に2行以上の日に当たる行 "
          f"{int(hit.sum()):,}（{hit.mean()*100:.2f}%）。rev_pct の欠測率: 当たる行 {nan_rate_hit*100:.1f}% / それ以外 {nan_rate_other*100:.1f}%")


def order_flip(sample: pd.DataFrame, fins: pd.DataFrame, n_shuffle: int = 5) -> None:
    """fins の行の順番を変えるだけで rev_pct の有無が入れ替わるかを見る（同じ行・同じデータ）。"""
    def ok(fr):
        r = B.forecast_revisions(sample.reset_index(drop=True), fr)
        return bool(np.isfinite(float(r["rev_pct"].iloc[0])))
    base = ok(fins)
    rev_order = ok(fins.iloc[::-1].reset_index(drop=True))
    shuffled = [ok(fins.sample(frac=1.0, random_state=k).reset_index(drop=True)) for k in range(n_shuffle)]
    print(f"  行の順番を変えて作り直す: そのまま {'有' if base else '欠測'} / 逆順 {'有' if rev_order else '欠測'} / "
          f"無作為 {n_shuffle}通り → 有 {sum(shuffled)}回・欠測 {n_shuffle - sum(shuffled)}回")


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
    population(rev_pct_by_row(fins), os.path.join(DATA, "dataset.parquet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
