#!/usr/bin/env python3
"""
probe: 本番に入れる2列（broken_high_age・prev_nh_ratio。build_dataset._breakout_shape）を全期間のデータで確かめる
（2026-10-17、運用者の決定「本番に入れる」。docs/MODEL_ADOPTION_RULES.md §62）。

Run Experiment が先に作った学習用のデータセット（research/_data/dataset.parquet。週次の再学習と同じ
`python3 research/build_dataset.py`）と、日次の予測と同じ作り（--keep-unlabeled）について:

  1. 一致     2列が、実験101 の作り方（e101_base_shape_ab.shape_columns。組み直した日足から銘柄ごとに別に数える）と
              一致するか。日足の更新日の全行と、データセットの全行で。base_length も日足と一致するか
  2. 付き方   欠測の数（broken_high_age は無いはず。prev_nh_ratio は base_length が欠測の行だけのはず）・分位・範囲
  3. 性質     base_length が窓（368）以下の行: broken_high_age = base_length の割合・prev_nh_ratio が 100 を超える割合
              （前回の更新日の高値が窓の中にあるので、超えたはず）。窓より長い行: prev_nh_ratio が 100 未満の割合
  4. 予測用   --keep-unlabeled で作り直したデータセットにも2列が付くか（ラベル未確定の行・直近の日の欠測の数）
  5. 列の組   239列の組は変わっていないか（指紋）。241列の組 = 239列の後ろに2列で、その列がすべてデータセットにあるか

出すのは件数・割合・分位・日付だけ（Actions のログは公開。銘柄ごとの値は出さない）。食い違いがあれば終了コード 1。

  exp=probe_breakout_shape.py
  exp=probe_breakout_shape.py args="--skip-predict"    （4 を飛ばす。データセットの作り直しが1回減る）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import e101_base_shape_ab as E  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR
NEW = ["broken_high_age", "prev_nh_ratio"]
BASE_PRESET = "all_plus_prog_listing_vol_book"
SHAPE_PRESET = "all_plus_prog_listing_vol_book_shape"
#: 239列の組の指紋（2026-09-26 から 2026-10-17 の決定までの既定。日曜 10/18 の再学習はこの列）
BASE_SIG = "2cf1283a9de5"
QS = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def same(a: pd.Series, b: pd.Series) -> int:
    """値が同じ行の数（両方欠測も同じと数える）。"""
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return int(((a == b) | (np.isnan(a) & np.isnan(b))).sum())


def quantiles(v: pd.Series) -> str:
    ok = pd.to_numeric(v, errors="coerce").dropna()
    if not len(ok):
        return "値なし"
    q = ok.quantile(list(QS))
    return " / ".join(f"{k * 100:g}%: {q[k]:,.1f}" for k in QS)


def properties(df: pd.DataFrame, w: int) -> Dict[str, int]:
    """base_length が窓以内か・窓より長いかで分けた件数（3. 性質）。"""
    bl, age, r = df["base_length"], df["broken_high_age"], df["prev_nh_ratio"]
    near, far = bl.notna() & (bl <= w), bl.notna() & (bl > w)
    return {"near": int(near.sum()), "near_age_eq": int((near & (age == bl)).sum()),
            "near_age_lt": int((near & (age < bl)).sum()), "near_age_gt": int((near & (age > bl)).sum()),
            "near_ratio_gt100": int((near & (r > 100)).sum()),
            "far": int(far.sum()), "far_ratio_lt100": int((far & (r < 100)).sum()),
            "age_out_of_range": int((age.notna() & ((age < 1) | (age > w))).sum()),
            "ratio_nan_with_base": int((bl.notna() & r.isna()).sum()),
            "ratio_with_no_base": int((bl.isna() & r.notna()).sum())}


def compare(name: str, got: pd.DataFrame, want: pd.DataFrame) -> bool:
    ok = True
    for c in NEW:
        n = same(got[c].reset_index(drop=True), want[c].reset_index(drop=True))
        print(f"  {name} {c}: {n:,} / {len(got):,}行が一致・食い違い {len(got) - n:,}")
        ok &= n == len(got)
    return ok


def report(name: str, df: pd.DataFrame, w: int) -> bool:
    print(f"■ 2・3. {name} {len(df):,}行")
    for c in NEW:
        print(f"  {c}: 欠測 {int(df[c].isna().sum()):,}・分位 {quantiles(df[c])}")
    p = properties(df, w)
    near, far = max(p["near"], 1), max(p["far"], 1)
    print(f"  base_length が {w} 以下 {p['near']:,}行: 抜いた高値の古さ = base_length {p['near_age_eq']:,}"
          f"（{p['near_age_eq'] / near * 100:.1f}%）・短い {p['near_age_lt']:,}（同じ値の高値をその後にも付けた）"
          f"・長い {p['near_age_gt']:,} / 今回の高値が前回の更新日の高値を超えた {p['near_ratio_gt100']:,}"
          f"（{p['near_ratio_gt100'] / near * 100:.1f}%）")
    print(f"  base_length が {w} より長い {p['far']:,}行: 今回の高値が前回の更新日の高値に届いていない "
          f"{p['far_ratio_lt100']:,}（{p['far_ratio_lt100'] / far * 100:.1f}%）")
    print(f"  範囲の外（1〜{w} でない）{p['age_out_of_range']:,} / base_length があるのに比が欠測 "
          f"{p['ratio_nan_with_base']:,} / base_length が欠測なのに比がある {p['ratio_with_no_base']:,}")
    return p["age_out_of_range"] == 0 and p["ratio_with_no_base"] == 0 and p["near_age_gt"] == 0


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args(argv)
    w = B.HIGH_WINDOW
    ok = True

    # 5. 列の組
    base = F.columns(BASE_PRESET)
    shape = F.columns(SHAPE_PRESET)
    print(f"■ 5. 既定 {F.DEFAULT_PRESET}（{len(F.columns(F.DEFAULT_PRESET))}列）/ {BASE_PRESET} {len(base)}列・指紋 "
          f"{F.signature(base)}（期待 {BASE_SIG}）/ {SHAPE_PRESET} {len(shape)}列・指紋 {F.signature(shape)}・"
          f"239列の後ろに2列: {shape == base + NEW}")
    ok &= F.signature(base) == BASE_SIG and shape == base + NEW

    # 4. 予測用（日次の予測と同じ作り）。日足を組む前に別のプロセスで作る（2つを同時にメモリに置かない）
    pr = None
    if not args.skip_predict:
        out = os.path.join(args.data_dir, "dataset_probe_predict.parquet")
        print("■ 4. --keep-unlabeled で作り直す（日次の予測と同じ作り）")
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "build_dataset.py"),
                            "--keep-unlabeled", "--out", out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  作れなかった（終了コード {r.returncode}）: {r.stderr.strip().splitlines()[-1:]}")
            return 1
        pr = pd.read_parquet(out, columns=["Code", "Date", "label", "base_length"] + NEW)
        os.remove(out)
        pr["Code"] = pr["Code"].astype(str)
        pr["Date"] = pd.to_datetime(pr["Date"])
        un = pr[pr["label"].isna()]
        last = pr[pr["Date"] == pr["Date"].max()]
        print(f"  {len(pr):,}行・ラベル未確定 {len(un):,}行（{un['Date'].min().date() if len(un) else '-'} 〜 "
              f"{un['Date'].max().date() if len(un) else '-'}）・直近の日 {pr['Date'].max().date()} {len(last):,}行")
        for c in NEW:
            print(f"  {c}: ラベル未確定の行の欠測 {int(un[c].isna().sum()):,} / 直近の日の欠測 {int(last[c].isna().sum()):,}")
        ok &= int(un["broken_high_age"].isna().sum()) == 0
        if len(un):
            ok &= report("予測用（ラベル未確定の行）", un, w)

    # 1. 日足の更新日の全行
    panel = E.build_panel(args.data_dir)
    panel["Code"] = panel["Code"].astype(str)
    print(f"\n[panel] {len(panel):,}行・{panel['Code'].nunique():,}銘柄・窓 {w}行・更新の判定 {B.BREAKOUT_REF}")
    nh = panel["is_new_high"].fillna(False).astype(bool)
    want = E.shape_columns(panel.loc[nh, ["Code", "Date"]], panel, w=w)
    print(f"■ 1. 一致（日足の更新日 {int(nh.sum()):,}行。build_dataset の2列 と 実験101 の数え方）")
    ok &= compare("更新日", panel.loc[nh, NEW], want)
    off = int(panel.loc[~nh, NEW].notna().sum().sum())
    print(f"  更新日でない行に値がある: {off:,}（無いはず）")
    ok &= off == 0

    # 1. データセット（学習用）の全行
    ds = pd.read_parquet(os.path.join(args.data_dir, "dataset.parquet"))
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    miss = [c for c in shape if c not in ds.columns]
    print(f"■ 5. {SHAPE_PRESET} の列でデータセットに無いもの: {len(miss)}（{', '.join(miss[:5])}）")
    if miss:
        return 1
    print(f"■ 1. 一致（学習用のデータセット {len(ds):,}行・{ds['Date'].min().date()} 〜 {ds['Date'].max().date()}）")
    ok &= compare("データセット", ds[NEW], E.shape_columns(ds[["Code", "Date"]], panel, w=w))
    m = ds[["Code", "Date", "base_length"]].merge(panel[["Code", "Date", "base_length"]], on=["Code", "Date"],
                                                  how="left", suffixes=("", "_panel"))
    n = same(m["base_length"], m["base_length_panel"])
    print(f"  base_length が日足と一致: {n:,} / {len(ds):,}")
    ok &= n == len(ds)
    if pr is not None:
        ok &= compare("予測用", pr[NEW], E.shape_columns(pr[["Code", "Date"]], panel, w=w))
    ok &= report("学習用のデータセット", ds, w)
    lab_ = ds["label"].astype(float)
    for c, edges in (("broken_high_age", tuple(e for e in (0, 5, 20, 60, 120, 250) if e < w) + (w,)),
                     ("prev_nh_ratio", (0, 60, 80, 95, 100, 105, 120, float("inf")))):
        cut = pd.cut(pd.to_numeric(ds[c], errors="coerce"), bins=list(edges))
        rows = [f"{iv} {int(g.notna().sum()):,}行 {g.mean() * 100:.1f}%" for iv, g in lab_.groupby(cut, observed=False)
                if g.notna().sum()]
        print(f"  {c} の区切りごとの正例率（1.2σ）: " + "  ".join(rows))

    print(f"\n[結論] {'すべて一致・想定どおり' if ok else '食い違いあり（上の行）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
