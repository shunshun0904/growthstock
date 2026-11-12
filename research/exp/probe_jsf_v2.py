#!/usr/bin/env python3
"""
日証金の需給の特徴量・作り直し版（research/jsf_features_v2.py。20列）の値の確認（docs/DATA_JSF.md「特徴量の作り直し（v2）」）。

運用者の依頼（2026-11-12）「列方向、横方向、銘柄間でのバリデーションチェックも欠かさずに、特徴量作成みすだけはないように」。
採否（A/B）の前に、元の数・作った列・付け方を確かめる。結果を見て列の作りを変えることはしない（変えるなら運用者に確かめる）。

出すのは件数・割合・日付・分位だけ（銘柄・値は出さない。日証金は私的利用の範囲で、Actions のログは公開）。

  [0] 入力: 過去ファイル・毎日の確報・品貸料率・制限措置の写し・J-Quants の信用区分・営業日の件数と範囲
  [1] 行方向（元の数）: 申込日×銘柄の重複、差引 = 融資 − 貸株、前日差 = 新規 − 返済（続いた営業日）、値の範囲
  [2] 銘柄間（元の数）: 申込日ごとの銘柄数の落ち込み、格子の穴（確報が欠けた日・日証金に載っていない期間）、
      コードの付け替え（日証金の銘柄名と J-Quants の銘柄名、貸借値段と J-Quants の終値）、同じ日に別の銘柄で同じ値の組、
      過去ファイルと毎日の確報の一致（重なる申込日）、品貸料率の年率換算、制限措置の写しと過去ファイルの一致（写しの日と
      申込日の対応 −1 / 0 / +1 営業日）、過去ファイルの貸借区分と J-Quants の信用区分
  [3] 作った列（候補の行に付けた値）: 付き方・時点（使った申込日 < T、間の暦日）、列方向（欠測・範囲・分位・無限）、
      期間ごとの分位（過去ファイルの時期と毎日の確報の時期でずれないか）、列どうしの関係（日数 0 なら最大 0 など）、
      同じ日の銘柄の中で1つの値に偏った列
  [4] 作り直し: 別の書き方（1行ずつ、元の数とカレンダーと日足から）で抜き出した行を計算し、全列が一致するか
  [5] 先読み: T より前のデータだけで作り直して、T の行の値が全部同じか（K 日）
  [6] 10/9 の作り（v1）との違い: 定義が同じ列は、窓に欠けた日がかからない行で一致するか
  [7] 取り直し前の控えがあれば比べる（research/jsf_hist_compare.py）

    exp=probe_jsf_v2.py                         ← 写しを使わない（RESTRICT_SNAP_SHIFT 未定）
    exp=probe_jsf_v2.py args="--shift 1"        ← 写しの日 + 1営業日 = 申込日 で作る
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import jsf_features as V1  # noqa: E402
import jsf_features_v2 as V2  # noqa: E402

DATA_DIR = V1.DATA_DIR
GAP_DAYS = pd.to_datetime(["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-30", "2026-11-02"])
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def pct(x: float) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.3f}%"


def q(s: pd.Series, qs=(0.01, 0.5, 0.99)) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return "値なし"
    return " / ".join(f"p{k * 100:g} {s.quantile(k):.4g}" for k in qs)


# ---------------------------------------------------------------------- #
# [1] [2] 元の数
# ---------------------------------------------------------------------- #

def split_days(data_dir: str) -> set:
    """J-Quants の日足で株式分割・併合のあった (日, 銘柄)（AdjFactor が 1 でない日）。"""
    out = set()
    for p in sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet"))):
        f = pd.read_parquet(p, columns=["Date", "Code", "AdjFactor"])
        f = f[(pd.to_datetime(f["Date"]) >= pd.Timestamp("2023-09-01")) & (f["AdjFactor"].astype(float) != 1.0)]
        out.update(zip(pd.to_datetime(f["Date"]), f["Code"].astype(str)))
    return out


def raw_checks(hist: pd.DataFrame, daily: Optional[pd.DataFrame], days: pd.DatetimeIndex,
               splits: Optional[set] = None) -> None:
    log("[1] 行方向（元の数）")
    for name, df in (("過去ファイル", hist), ("毎日の確報", daily)):
        if df is None or not len(df):
            continue
        dup = int(df.duplicated(["app_date", "Code"]).sum())
        net = df["net_bal"].astype(float)
        diff = df["loan_bal"].astype(float) - df["stock_bal"].astype(float)
        ok = (net - diff).abs() < 0.5
        have = net.notna() & diff.notna()
        neg = int(((df[["loan_bal", "stock_bal", "loan_new", "loan_ret", "stock_new", "stock_ret"]]
                    .astype(float) < 0).any(axis=1)).sum())
        print(f"  {name}: {len(df):,}行 / 申込日×銘柄の重複 {dup:,} / 差引 = 融資 − 貸株 {pct(ok[have].mean())}"
              f"（{int(have.sum()):,}行）/ 負の残高・新規・返済のある行 {neg:,}")
    p = pd.concat([x for x in (daily, hist) if x is not None], ignore_index=True)
    p = p.drop_duplicates(["app_date", "Code"], keep="first").sort_values(["Code", "app_date"])
    pos = {d: i for i, d in enumerate(days)}
    p["bd_i"] = p["app_date"].map(pos)
    prev = p.groupby("Code")[["bd_i", "loan_bal", "stock_bal"]].shift(1)
    cont = (p["bd_i"] - prev["bd_i"]) == 1
    for side, new, ret in (("融資", "loan_new", "loan_ret"), ("貸株", "stock_new", "stock_ret")):
        bal = "loan_bal" if side == "融資" else "stock_bal"
        d = p[bal].astype(float) - prev[bal].astype(float)
        flow = p[new].astype(float).fillna(0) - p[ret].astype(float).fillna(0)
        have = cont & d.notna()
        ok = (d - flow).abs() < 0.5
        bad = p[have & ~ok]
        near = 0
        if splits:
            pi = {i: dd for dd, i in pos.items()}
            for r in bad.itertuples():
                i = int(r.bd_i)
                if any((pi.get(k), r.Code) in splits for k in (i - 2, i - 1, i, i + 1)):
                    near += 1
        print(f"  {side}: 前日差 = 新規 − 返済（続いた営業日 {int(have.sum()):,}行）{pct(ok[have].mean())}・合わない "
              f"{len(bad):,}行（うち J-Quants で分割・併合のあった日の前後 {near:,}行・{bad['Code'].nunique():,}銘柄・"
              f"{bad['app_date'].nunique():,}日）")


def coverage_checks(g: pd.DataFrame, days: pd.DatetimeIndex) -> None:
    log("[2] 銘柄間（元の数）: 申込日ごとの銘柄数・格子の穴")
    obs = g[g["observed"]]
    per = obs.groupby("app_date")["Code"].nunique()
    med = per.rolling(21, center=True, min_periods=5).median()
    drop = per[(per < med * 0.95)]
    ramp = per.index[:15]                          # 過去ファイルの始まり（銘柄ごとに取った日から3年なので少しずつ増える）
    rest = drop[~drop.index.isin(ramp)]
    print(f"  申込日 {len(per):,}日・銘柄数 最小 {per.min():,} / 中央 {int(per.median()):,} / 最大 {per.max():,}・"
          f"前後21日の中央値より5%以上少ない日 {len(drop)}（始まりの15営業日 {int(drop.index.isin(ramp).sum())}・ほか "
          f"{len(rest)}: {', '.join(d.date().isoformat() for d in rest.index)}）")
    holes = g[~g["observed"]]
    on_gap = holes["app_date"].isin(GAP_DAYS)
    print(f"  格子の穴 {len(holes):,}行: 確報が欠けた5日 {int(on_gap.sum()):,}行（"
          + ", ".join(f"{d.date().isoformat()} {int((holes['app_date'] == d).sum()):,}" for d in GAP_DAYS)
          + f"）/ ほか {int((~on_gap).sum()):,}行・{holes.loc[~on_gap, 'Code'].nunique():,}銘柄")


def mapping_checks(hist_raw: pd.DataFrame, data_dir: str) -> None:
    """コードの付け替え: 日証金の銘柄名と J-Quants の銘柄名、貸借値段と J-Quants の終値（前の営業日・同じ日）。"""
    log("[2] 銘柄間（元の数）: コードの付け替え")
    paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    m = pd.concat([pd.read_parquet(x, columns=["Date", "Code", "CoName"]) for x in paths], ignore_index=True)
    m = m.sort_values("Date").drop_duplicates("Code", keep="last")

    def norm(s):
        s = unicodedata.normalize("NFKC", str(s or ""))
        return "".join(ch for ch in s if not ch.isspace()).replace("ホールディングス", "HD").replace("株式会社", "")
    names = hist_raw.drop_duplicates("code", keep="last")[["code", "銘柄名"]].rename(columns={"code": "Code"})
    j = names.merge(m[["Code", "CoName"]], on="Code", how="left")
    have = j["CoName"].notna()
    a = j.loc[have, "銘柄名"].map(norm)
    b = j.loc[have, "CoName"].map(norm)
    exact = (a == b)
    part = exact.to_numpy() | np.array([x[:4] in y or y[:4] in x for x, y in zip(a, b)], dtype=bool)
    print(f"  銘柄名: 日証金の {len(j):,}銘柄のうち J-Quants にある {int(have.sum()):,}・名前が同じ {pct(exact.mean())}・"
          f"頭4文字が含まれる {pct(np.mean(part))}")
    bars = []
    for p in sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet"))):
        f = pd.read_parquet(p, columns=["Date", "Code", "C"])
        f = f[pd.to_datetime(f["Date"]) >= pd.Timestamp("2023-09-01")]
        bars.append(f)
    if not bars:
        print("  終値: 日足が無い")
        return
    b = pd.concat(bars, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    b = b.sort_values(["Code", "Date"])
    b["C_prev"] = b.groupby("Code")["C"].shift(1)
    h = pd.DataFrame({"Date": pd.to_datetime(hist_raw["申込日"]), "Code": hist_raw["code"].astype(str),
                      "price": pd.to_numeric(hist_raw["貸借値段（円）"], errors="coerce")})
    h = h[h["price"] > 0]
    x = h.merge(b, on=["Date", "Code"], how="inner")
    bad_names = set(j.loc[have, "Code"][~part])
    if bad_names:
        xb = x[x["Code"].isin(bad_names)]
        rr = (xb["price"] / xb["C"]).replace([np.inf, -np.inf], np.nan)
        per = rr.groupby(xb["Code"]).apply(lambda s: ((s - 1).abs() <= 0.005).mean())
        print(f"  名前が合わない {len(bad_names)}銘柄: 貸借値段が同じ日の終値と ±0.5% 以内の行が 99% 以上の銘柄 "
              f"{int((per >= 0.99).sum())} / {len(per)}（日足の無い銘柄 {len(bad_names) - len(per)}）")
    for col, label in (("C_prev", "前の営業日の終値"), ("C", "同じ日の終値")):
        r = (x["price"] / x[col]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  貸借値段 ÷ J-Quants の{label}: {len(r):,}行・±0.5% 以内 {pct(((r - 1).abs() <= 0.005).mean())}・"
              f"±5% 以内 {pct(((r - 1).abs() <= 0.05).mean())}・{q(r)}")


def same_vector_check(g: pd.DataFrame) -> None:
    """
    同じ申込日に、別の銘柄でまったく同じ値の組（融資・貸株の残高・新規）があれば、付け違いの疑い。小さな残高（100株単位）
    どうしは偶然そろうので、残高の大きさで分けて数える。大きい組（融資 10万株以上・貸株 1万株以上）で同じなら付け違いを疑う。
    """
    o = g[g["observed"] & (g["loan_bal"].fillna(0) > 0) & (g["stock_bal"].fillna(0) > 0)]
    keys = ["app_date", "loan_bal", "stock_bal", "loan_new", "stock_new"]
    dup = o[keys].duplicated(keep=False)
    big = (o["loan_bal"] >= 100_000) & (o["stock_bal"] >= 10_000)
    ob = o[big]
    dup_big = ob[keys].duplicated(keep=False)
    small = (o.loc[dup, "loan_bal"] < 10_000).mean() if dup.any() else float("nan")
    print(f"  同じ日に別の銘柄で融資・貸株の残高と新規が全部同じ行: {int(dup.sum()):,} / {len(o):,}（{o.loc[dup, 'app_date'].nunique()}日・"
          f"うち融資1万株未満 {pct(small)}）/ 大きい組（融資10万株・貸株1万株以上）では {int(dup_big.sum()):,} / {len(ob):,}")


def hist_vs_daily(hist: pd.DataFrame, daily: Optional[pd.DataFrame]) -> None:
    log("[2] 過去ファイルと毎日の確報（重なる申込日）")
    if daily is None or not len(daily):
        print("  毎日の確報が無い")
        return
    last = hist.groupby("Code")["app_date"].max()
    m = hist.merge(daily, on=["app_date", "Code"], suffixes=("_h", "_d"))
    m["_last"] = m["app_date"].eq(m["Code"].map(last))
    body = m[~m["_last"]]
    print(f"  重なり {len(m):,}行（各銘柄の過去ファイルの最終日 {int(m['_last'].sum()):,}行は分けて数える）")
    for c in ("loan_bal", "stock_bal", "net_bal", "loan_new", "loan_ret", "stock_new", "stock_ret"):
        same = ((m[f"{c}_h"] - m[f"{c}_d"]).abs() < 0.5) | (m[f"{c}_h"].isna() & m[f"{c}_d"].isna())
        print(f"    {c}: 一致 {pct(same[~m['_last']].mean())}（{len(body):,}行）/ 最終日の行で一致 "
              f"{pct(same[m['_last']].mean())}")
    fh, fd = body["fee_ann_h"].fillna(0), body["fee_ann_d"].fillna(0)
    both = (fh > 0) & (fd > 0)
    r = fd[both] / fh[both]
    print(f"    品貸料率の年率: 両方あり {int(both.sum()):,}行（確報 ÷ 過去 {q(r)}）/ 過去だけ {int(((fh > 0) & ~(fd > 0)).sum()):,}"
          f" / 確報だけ {int((~(fh > 0) & (fd > 0)).sum()):,}（最終日を除く）")
    lm = m[m["_last"]]
    print(f"    最終日の行: 確報だけに料率がある {int((~(lm['fee_ann_h'].fillna(0) > 0) & (lm['fee_ann_d'].fillna(0) > 0)).sum()):,}"
          f" / {len(lm):,}（速報には料率がまだ無い。同じ日は確報を使う）")


def fee_annualization(hist_raw: pd.DataFrame) -> None:
    fa = pd.to_numeric(hist_raw["品貸料率（年率換算/％）"], errors="coerce")
    fy = pd.to_numeric(hist_raw["品貸料率（品貸日数分/円）"], errors="coerce")
    fd = pd.to_numeric(hist_raw["品貸日数"], errors="coerce")
    pr = pd.to_numeric(hist_raw["貸借値段（円）"], errors="coerce")
    calc = V1.annualize_fee(fy, fd, pr)
    ok = fa.notna() & calc.notna() & (fa > 0)
    r = calc[ok] / fa[ok]
    print(f"  品貸料率の年率換算（過去ファイルの列 ÷ 円・日数・値段から計算した値）: {int(ok.sum()):,}行・±5% 以内 "
          f"{pct(((r - 1).abs() <= 0.05).mean())}・{q(r, (0.001, 0.5, 0.999))}")


def restrict_alignment(hist: pd.DataFrame, snaps: Optional[pd.DataFrame], days: pd.DatetimeIndex) -> None:
    log("[2] 制限措置の写しと過去ファイル（写しの日と申込日の対応）")
    if snaps is None or not len(snaps):
        print("  写しが無い")
        return
    last = hist.groupby("Code")["app_date"].max()
    h = hist[hist["app_date"] < hist["Code"].map(last)][["app_date", "Code", "restrict"]]
    for shift in (-1, 0, 1, 2):
        sl = V2.snapshot_levels(snaps, days, shift)
        cov = pd.DatetimeIndex(sl.attrs["covered"])
        base = h[h["app_date"].isin(cov)]
        x = base.merge(sl.rename(columns={"code": "Code", "level": "snap"}), on=["app_date", "Code"], how="left")
        x["snap"] = x["snap"].fillna(0.0)
        ok = x["snap"] == x["restrict"]
        ch = x["restrict"] > 0
        print(f"  写しの日 +{shift}営業日 = 申込日: {len(x):,}行・{x['app_date'].nunique()}日・一致 {pct(ok.mean())}・"
              f"合わない {int((~ok).sum()):,}（過去ファイルで措置あり {int(ch.sum()):,}・写しで措置あり {int((x['snap'] > 0).sum()):,}）")


def lendable_crosstab(hist_raw: pd.DataFrame, margin: pd.DataFrame) -> None:
    log("[2] 過去ファイルの貸借区分と J-Quants の信用区分")
    h = pd.DataFrame({"app_date": pd.to_datetime(hist_raw["申込日"]), "Code": hist_raw["code"].astype(str),
                      "jsf": hist_raw["貸借区分"].astype("string").fillna("").str.strip()})
    mg = margin.rename(columns={"Date": "app_date"}).sort_values("app_date")
    x = pd.merge_asof(h.sort_values("app_date"), mg, on="app_date", by="Code", direction="backward")
    tab = pd.crosstab(x["jsf"], x["lendable"].map({1.0: "貸借(1)", 0.0: "信用・その他(0)"}).fillna("無し"))
    for idx, r in tab.iterrows():
        print(f"  日証金「{idx or '(空)'}」: " + " / ".join(f"{c} {int(v):,}" for c, v in r.items()))
    agree = ((x["jsf"] == "貸借") & (x["lendable"] == 1)) | (x["jsf"].isin(["貸借融資", "非貸借"]) & (x["lendable"] == 0))
    print(f"  一致（貸借↔1、貸借融資・非貸借↔0）{pct(agree[x['lendable'].notna()].mean())}")


# ---------------------------------------------------------------------- #
# [3] 作った列
# ---------------------------------------------------------------------- #

PERIODS = [("2023-10〜2024-03", "2023-10-01", "2024-03-31"), ("2024-04〜09", "2024-04-01", "2024-09-30"),
           ("2024-10〜2025-03", "2024-10-01", "2025-03-31"), ("2025-04〜09", "2025-04-01", "2025-09-30"),
           ("2025-10〜2026-03", "2025-10-01", "2026-03-31"), ("2026-04〜09", "2026-04-01", "2026-09-30"),
           ("2026-10〜", "2026-10-01", "2099-12-31")]
RANGES = {"jsf2_fee": (0, None), "jsf2_fee_days20": (0, 20), "jsf2_fee_max20": (0, None),
          "jsf2_restrict": (0, 3), "jsf2_lendable": (0, 1), "jsf2_loan_pos60": (0, 1),
          "jsf2_stock_pos60": (0, 1), "jsf2_ratio_pos60": (0, 1), "jsf2_lag": (1, 7)}


def column_checks(df: pd.DataFrame) -> None:
    log("[3] 作った列（候補の行に付けた値）")
    have = df["jsf2_lag"].notna()
    print(f"  候補の行 {len(df):,}・付いた行 {int(have.sum()):,}（{pct(have.mean())}）・2023年10月以降で "
          f"{pct(have[df['Date'] >= '2023-10-01'].mean())}")
    by = df[df["Date"] >= "2023-10-01"].groupby(df["Date"].dt.to_period("Q"))["jsf2_lag"].apply(lambda s: s.notna().mean())
    print("  付いた割合（四半期）: " + " / ".join(f"{k} {v * 100:.1f}%" for k, v in by.items()))
    lag = df.loc[have, "jsf2_lag"]
    print("  使った申込日から T までの暦日: " + " / ".join(f"{int(k)}日 {int(v):,}" for k, v in lag.value_counts().sort_index().items()))
    a = df[have]
    for c in V2.columns("all"):
        s = a[c].astype(float)
        inf = int(np.isinf(s).sum())
        lo, hi = RANGES.get(c, (None, None))
        bad = 0
        if lo is not None:
            bad += int((s < lo).sum())
        if hi is not None:
            bad += int((s > hi).sum())
        print(f"  {c}: 欠測 {pct(s.isna().mean())}・無限 {inf}・範囲の外 {bad}・{q(s)}")
    fd, fm, fe = a["jsf2_fee_days20"], a["jsf2_fee_max20"], a["jsf2_fee"]
    both = fd.notna() & fm.notna()
    print(f"  関係: 逆日歩の日数 0 なのに最大 > 0 {int(((fd == 0) & (fm > 0) & both).sum())}・日数 > 0 なのに最大 0 "
          f"{int(((fd > 0) & (fm == 0) & both).sum())}・当日 > 20日の最大 {int(((fe > fm + 1e-12) & fm.notna()).sum())}・"
          f"日数が整数でない {int((fd.notna() & ((fd % 1) != 0)).sum())}・段階が整数でない "
          f"{int((a['jsf2_restrict'].notna() & ((a['jsf2_restrict'] % 1) != 0)).sum())}")
    log("[3] 期間ごとの分位（中央値・p90。過去ファイルの時期と毎日の確報の時期でずれないか）")
    for c in V2.columns("all"):
        parts = []
        for name, lo, hi in PERIODS:
            s = a.loc[(a["Date"] >= lo) & (a["Date"] <= hi), c].dropna()
            parts.append(f"{name} {s.median():.3g}/{s.quantile(0.9):.3g}" if len(s) else f"{name} -")
        print(f"  {c}: " + " | ".join(parts))
    log("[3] 銘柄間: 同じ日の銘柄の中で1つの値に偏った列（その日に付いた銘柄が10以上の日）")
    days_ok = a.groupby("Date")["Code"].transform("size") >= 10
    b = a[days_ok]
    for c in V2.columns("all"):
        share = b.groupby("Date")[c].agg(lambda s: s.value_counts(normalize=True, dropna=True).iloc[0]
                                         if s.notna().any() else np.nan)
        print(f"  {c}: 最も多い値の割合（日ごと）の中央値 {pct(share.median())}・1つの値だけの日 {int((share >= 1).sum())} / {share.notna().sum()}")


# ---------------------------------------------------------------------- #
# [4] 別の書き方での作り直し
# ---------------------------------------------------------------------- #

def reference_values(code: str, t: pd.Timestamp, obs: Dict, days: pd.DatetimeIndex, pos: Dict,
                     vol_by_code: Dict, snap_lv: Optional[Dict], snap_cov: set, hist_r: Dict,
                     margin_by_code: Dict) -> Dict[str, float]:
    """1行ずつ元の数から作る（jsf_features_v2 とは別の書き方）。obs[code] = {申込日: 値の dict}。"""
    out = {c: np.nan for c in V2.columns("both")}
    rows = obs.get(code)
    if not rows:
        return out
    cands = [d for d in rows if d < t and (t - d).days <= V2.MAX_STALE - 1]
    if not cands:
        return out
    a = max(cands)
    ia = pos[a]
    first = pos[min(rows)]

    def val(k, field):
        if k < first:
            return np.nan
        d = days[k]
        r = rows.get(d)
        return np.nan if r is None else r[field]

    def ratio(k):
        lb, sb = val(k, "loan_bal"), val(k, "stock_bal")
        if np.isnan(lb) or np.isnan(sb):
            return np.nan
        return math.log((max(lb, 0) + 1.0) / (max(sb, 0) + 1.0))
    vb = vol_by_code.get(code)
    av = np.nan
    if vb is not None:
        vv = vb[vb["Date"] <= a]
        if len(vv) and (a - vv["Date"].iloc[-1]).days <= 10:
            tail = vv["Vo"].iloc[-V2.VOL_WINDOW:]
            if tail.notna().sum() >= max(5, V2.VOL_WINDOW // 2):
                av = float(tail.mean())
        if not (av > 0):
            av = np.nan
    lb, sb = val(ia, "loan_bal"), val(ia, "stock_bal")
    out["jsf2_ratio"] = ratio(ia)
    for n in (5, 20):
        out[f"jsf2_ratio_chg{n}"] = ratio(ia) - ratio(ia - n)
        out[f"jsf2_loan_chg{n}_v"] = math.asinh((lb - val(ia - n, "loan_bal")) / av)
        out[f"jsf2_stock_chg{n}_v"] = math.asinh((sb - val(ia - n, "stock_bal")) / av)
    out["jsf2_loan_v"] = math.asinh(lb / av)
    out["jsf2_stock_v"] = math.asinh(sb / av)
    out["jsf2_net_v"] = math.asinh((lb - sb) / av)
    for field, name in (("loan_new", "jsf2_long_new5_v"), ("stock_new", "jsf2_short_new5_v")):
        xs = [val(k, field) for k in range(ia - 4, ia + 1)]
        out[name] = math.asinh(sum(xs) / av) if not any(np.isnan(x) for x in xs) else np.nan
    fees = [val(k, "fee_ann") for k in range(max(ia - V2.FEE_WINDOW + 1, first), ia + 1)]
    fees = [max(f, 0.0) for f in fees if not np.isnan(f)]
    out["jsf2_fee"] = math.log1p(max(val(ia, "fee_ann"), 0.0))
    if len(fees) >= V2.FEE_MIN:
        out["jsf2_fee_days20"] = float(sum(1 for f in fees if f > 0))
        out["jsf2_fee_max20"] = math.log1p(max(fees))
    for field, name in (("loan_bal", "jsf2_loan_pos60"), ("stock_bal", "jsf2_stock_pos60"), ("ratio", "jsf2_ratio_pos60")):
        ks = range(max(ia - V2.POS_WINDOW + 1, first), ia + 1)
        xs = [ratio(k) if field == "ratio" else val(k, field) for k in ks]
        xs = [x for x in xs if not np.isnan(x)]
        x = ratio(ia) if field == "ratio" else val(ia, field)
        if len(xs) >= V2.POS_MIN and not np.isnan(x):
            less = sum(1 for y in xs if y < x)
            eq = sum(1 for y in xs if y == x)
            out[name] = (less + (eq - 1) / 2.0) / (len(xs) - 1)
    r = hist_r.get((a, code), np.nan)
    if not np.isnan(r):
        out["jsf2_restrict"] = r
    elif snap_lv is not None and a in snap_cov:
        out["jsf2_restrict"] = snap_lv.get((a, code), 0.0)
    mb = margin_by_code.get(code)
    if mb is not None:
        mm = mb[mb["Date"] <= a]
        if len(mm):
            out["jsf2_lendable"] = mm["lendable"].iloc[-1]
    out["jsf2_lag"] = float((t - a).days)
    return out


def raw_observations(hist: pd.DataFrame, daily: Optional[pd.DataFrame], codes: set) -> Dict[str, Dict]:
    """元の数から、銘柄ごとの {申込日: 値} を作る（panel_v2 を使わない）。同じ日は確報、料率は確報 → 過去 → 0。"""
    fields = ("loan_bal", "stock_bal", "loan_new", "stock_new")
    out: Dict[str, Dict] = {}
    hh = hist[hist["Code"].isin(codes)]
    for r in hh.itertuples():
        out.setdefault(r.Code, {})[r.app_date] = {**{f: float(getattr(r, f)) for f in fields},
                                                   "fee_ann": float(r.fee_ann)}
    if daily is not None:
        for r in daily[daily["Code"].isin(codes)].itertuples():
            prev = out.setdefault(r.Code, {}).get(r.app_date)
            fee = float(r.fee_ann)
            if np.isnan(fee) and prev is not None:
                fee = prev["fee_ann"]
            out[r.Code][r.app_date] = {**{f: float(getattr(r, f)) for f in fields}, "fee_ann": fee}
    for rows in out.values():
        for v in rows.values():
            if np.isnan(v["fee_ann"]):
                v["fee_ann"] = 0.0
    return out


def rebuild_check(df: pd.DataFrame, hist: pd.DataFrame, daily, snaps, shift, margin, days,
                  data_dir: str, n: int, seed: int) -> None:
    log(f"[4] 別の書き方で作り直す（抜き出した {n:,}行）")
    rng = np.random.default_rng(seed)
    have = df[df["jsf2_lag"].notna()]
    idx = rng.choice(len(have), size=min(n, len(have)), replace=False)
    samp = have.iloc[np.sort(idx)]
    codes = set(samp["Code"])
    obs = raw_observations(hist, daily, codes)
    pos = {d: i for i, d in enumerate(days)}
    vb = V1.load_bars(data_dir, codes=codes)
    vb["Date"] = pd.to_datetime(vb["Date"])
    vb["Code"] = vb["Code"].astype(str)
    vb["Vo"] = pd.to_numeric(vb["Vo"], errors="coerce")
    vol_by_code = {c: p.sort_values("Date").reset_index(drop=True) for c, p in vb.groupby("Code")}
    snap_lv, snap_cov = None, set()
    if snaps is not None and shift is not None:
        sl = V2.snapshot_levels(snaps, days, shift)
        snap_cov = set(pd.DatetimeIndex(sl.attrs["covered"]))
        snap_lv = {(r.app_date, r.code): float(r.level) for r in sl.itertuples()}
    hh = hist[hist["Code"].isin(codes)]
    hist_r = {(r.app_date, r.Code): float(r.restrict) for r in hh.itertuples()}
    mg = margin[margin["Code"].isin(codes)]
    margin_by_code = {c: p.sort_values("Date").reset_index(drop=True) for c, p in mg.groupby("Code")}
    cols = V2.columns("both")
    same = {c: 0 for c in cols}
    for r in samp.itertuples():
        ref = reference_values(r.Code, r.Date, obs, days, pos, vol_by_code, snap_lv, snap_cov, hist_r, margin_by_code)
        for c in cols:
            x, y = getattr(r, c), ref[c]
            if (np.isnan(x) and np.isnan(y)) or (not np.isnan(x) and not np.isnan(y) and abs(x - y) <= 1e-9 * max(1, abs(y))):
                same[c] += 1
    bad = {c: len(samp) - v for c, v in same.items() if v != len(samp)}
    print(f"  {len(samp):,}行 × {len(cols)}列: 全部一致の列 {len(cols) - len(bad)} / {len(cols)}"
          + ("" if not bad else "・合わない列 " + ", ".join(f"{c} {v}行" for c, v in bad.items())))


# ---------------------------------------------------------------------- #
# [5] 先読み
# ---------------------------------------------------------------------- #

def lookahead_check(df: pd.DataFrame, hist, daily, snaps, shift, margin, days, data_dir: str, k: int, seed: int) -> None:
    log(f"[5] 先読み: T より前のデータだけで作り直す（{k}日）")
    rng = np.random.default_rng(seed + 1)
    have = df[df["jsf2_lag"].notna()]
    dates = sorted(have["Date"].unique())
    pick = sorted(rng.choice(len(dates), size=min(k, len(dates)), replace=False))
    picked = [pd.Timestamp(dates[i]) for i in pick]
    later = sorted(d for d in dates if pd.Timestamp(d) >= pd.Timestamp("2026-09-24"))
    for d in later[-3:]:
        if pd.Timestamp(d) not in picked:
            picked.append(pd.Timestamp(d))
    bars = V1.load_bars(data_dir, codes=set(have["Code"]))
    bars["Date"] = pd.to_datetime(bars["Date"])
    cols = V2.columns("both")
    worst = 0
    nrows = 0
    for t in picked:
        rows = have[have["Date"] == t]
        cs = set(rows["Code"])
        hh = hist[(hist["app_date"] < t) & hist["Code"].isin(cs)]
        dd = None if daily is None else daily[(daily["app_date"] < t) & daily["Code"].isin(cs)]
        ss = None if snaps is None else snaps[snaps["snap_date"] <= t]     # 写しは T の 13:10 に取る（夜の予測より前）
        mg = margin[margin["Date"] < t]
        if not len(hh) and (dd is None or not len(dd)):
            continue
        g = V2.panel_v2(hh, dd, days, ss, shift, mg)
        vol = V1.avg_volume(bars[(bars["Date"] < t) & bars["Code"].isin(cs)])
        got = V2.attach_v2(rows[["Code", "Date"]], V2.features_v2(g, vol))
        a = got[cols].to_numpy(float)
        b = rows[cols].to_numpy(float)
        diff = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9))
        worst += int(diff.any(axis=1).sum())
        nrows += len(rows)
    print(f"  {len(picked)}日・{nrows:,}行: 値が変わった行 {worst}")


# ---------------------------------------------------------------------- #
# [6] v1 との違い
# ---------------------------------------------------------------------- #

SAME_DEF = [("jsf_ratio", "jsf2_ratio"), ("jsf_loan_v", "jsf2_loan_v"), ("jsf_stock_v", "jsf2_stock_v"),
            ("jsf_net_v", "jsf2_net_v"), ("jsf_fee", "jsf2_fee"), ("jsf_ratio_chg5", "jsf2_ratio_chg5"),
            ("jsf_ratio_chg20", "jsf2_ratio_chg20"), ("jsf_loan_chg5_v", "jsf2_loan_chg5_v"),
            ("jsf_loan_chg20_v", "jsf2_loan_chg20_v"), ("jsf_stock_chg5_v", "jsf2_stock_chg5_v"),
            ("jsf_stock_chg20_v", "jsf2_stock_chg20_v"), ("jsf_long_new5_v", "jsf2_long_new5_v"),
            ("jsf_short_new5_v", "jsf2_short_new5_v"), ("jsf_fee_days20", "jsf2_fee_days20"),
            ("jsf_fee_max20", "jsf2_fee_max20"), ("jsf_restrict", "jsf2_restrict"), ("jsf_lendable", "jsf2_lendable")]


def v1_check(df: pd.DataFrame, data_dir: str) -> None:
    log("[6] 10/9 の作り（v1）との違い（定義が同じ列。窓に欠けた日がかからない 2026-09 までの行）")
    v1 = V1.build(df[["Code", "Date"]].copy(), data_dir)
    x = pd.concat([df, v1[V1.columns("both")]], axis=1)
    x = x[(x["Date"] < "2026-10-01") & x["jsf_lag"].notna() & x["jsf2_lag"].notna()]
    print(f"  比べた行 {len(x):,}（使った申込日が同じ行 {pct((x['jsf_lag'] == x['jsf2_lag']).mean())}）")
    for a, b in SAME_DEF:
        u, v = x[a].astype(float), x[b].astype(float)
        same = (u.isna() & v.isna()) | ((u - v).abs() <= 1e-9)
        print(f"    {a} ↔ {b}: 一致 {pct(same.mean())}・v1 だけ値あり {int((u.notna() & v.isna()).sum()):,}・"
              f"v2 だけ値あり {int((u.isna() & v.notna()).sum()):,}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="日証金の特徴量 v2 の値の確認（件数・割合・分位だけ）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--shift", type=int, default=None, help="写しの日と申込日の対応（営業日）。省略で写しを使わない")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--lookahead-days", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d = a.data_dir

    log("[0] 入力")
    hist_raw = pd.read_parquet(os.path.join(d, "jsf_hist.parquet"))
    hist, daily, snaps, margin, days = V2.inputs(d)
    print(f"  過去ファイル {len(hist):,}行・{hist['Code'].nunique():,}銘柄（{hist['app_date'].min().date()} 〜 "
          f"{hist['app_date'].max().date()}）/ 毎日の確報 {0 if daily is None else len(daily):,}行"
          f"（{'-' if daily is None else daily['app_date'].min().date()} 〜 {'-' if daily is None else daily['app_date'].max().date()}）"
          f" / 写し {0 if snaps is None else snaps['snap_date'].nunique()}日 / 信用区分 {len(margin):,}行 / 営業日 {len(days):,}")
    raw_checks(hist, daily, days, split_days(d))
    g = V2.panel_v2(hist, daily, days, snaps, a.shift, margin)
    coverage_checks(g, days)
    mapping_checks(hist_raw, d)
    same_vector_check(g)
    hist_vs_daily(hist, daily)
    fee_annualization(hist_raw)
    restrict_alignment(hist, snaps, days)
    lendable_crosstab(hist_raw, margin)

    ds = pd.read_parquet(os.path.join(d, "dataset.parquet"), columns=["Code", "Date"])
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    log(f"[3] 候補の行 {len(ds):,}（{ds['Date'].min().date()} 〜 {ds['Date'].max().date()}）に付ける"
        f"（写しの対応 {'使わない' if a.shift is None else f'+{a.shift}営業日'}）")
    vol = V1.avg_volume(V1.load_bars(d, codes=g["Code"].unique()))
    feats = V2.features_v2(g, vol)
    df = V2.attach_v2(ds, feats)
    column_checks(df)
    rebuild_check(df, hist, daily, snaps, a.shift, margin, days, d, a.sample, a.seed)
    lookahead_check(df, hist, daily, snaps, a.shift, margin, days, d, a.lookahead_days, a.seed)
    v1_check(df, d)
    bk = os.path.join(d, "backup_jsf_hist_20261112.parquet")
    if os.path.exists(bk):
        log("[7] 取り直し前の控えと比べる")
        import jsf_hist_compare as HC
        HC.report(HC.compare(HC._load(bk), HC._load(os.path.join(d, "jsf_hist.parquet"))))
    log("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
