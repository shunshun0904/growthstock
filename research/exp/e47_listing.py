#!/usr/bin/env python3
"""
実験47: TOKYO PRO MARKET（TPM）の時期の行を78週の履歴に数えない（運用者の選択 A）と、
上場からの年数（運用者の選択 ②）を入れると、分離力と運用の規則での取引がどう変わるか。

A: J-Quants は TPM の銘柄にも日足の行を持つが値はほぼ空で、一般市場へ移った銘柄は
   その空の行も368本の履歴に数えられていた（5537 は約9か月の高値が「78週高値」に）。
   build_dataset.GENERAL_MARKET_START=True で、最後に TPM だった月末の後の最初の値から数える。
②: listing_years = 一般市場に上場（TPM から移行）してからの年数。5年で打ち止め
   （2026-09-25 に①の3年から変えた）。2016-10 より前から上場している銘柄は 2021-10 までは
   欠測、その後は 5。欠測が 2018〜2021年に偏るので、欠測そのものが「時期」の目印になる。
   対照 P は同じ日の中で入れ替えるので時期の情報は残る。L − P で銘柄ごとの情報だけを読む

腕
  T  新しい母集団（GENERAL_MARKET_START=True）・205列（all_plus）   ← 比べる基準
  A  今の母集団（False）・205列。T − A が母集団の直しの効果（行はごくわずかしか
     変わらないので、差は種のばらつきの中に収まるはず）
  L  T + listing_years（all_plus_listing、206列）。L − T が上場年数の効果
  P  対照: L の listing_years を同じ日の銘柄どうしで入れ替えたもの（列を足しただけの幅）
共通: 本番のパラメータ（読むだけ）、ブースティング3モデル、種3つの平均、窓の切り方3通り

採否（docs/MODEL_ADOPTION_RULES.md §7）
  上場年数: L − T の LightGBM の PR-AUC が 0.0048 を超え、XGBoost / CatBoost も同じ向き
  母集団の直し: 正しさの修正なので、A − T が種のばらつき（0.0048）の中なら入れる

  --shifts 0,2,4  --seeds 3  --algos lgbm,xgb,cat
  結果は research/_data/oof/e47_*。本番の設定には書かない。

2026-09-25 に運用者の決定で、母集団の直し（True）と上場からの年数を本番に入れた
（features.DEFAULT_PRESET = all_plus_prog_listing）。記録した結果は 86c7b4d で回したもの。
回し直しても同じ比較になるよう、列は all_plus を名前で指定し、母集団は2通りとも
フラグを明示して作る（lab.frame() の既定の母集団には頼らない）。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
from e44_shortsale import permuted  # noqa: E402

#: 比べる列。実験を組んだときの本番（205列）。本番の既定が変わっても動かさない
BASE_PRESET = "all_plus"
LABELS = {"T": "T 新しい母集団（TPMの行を数えない）", "A": "A 今の母集団",
          "L": "L T + 上場からの年数", "P": "P 対照（上場年数を日付内で入れ替え）"}
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20260926


def build_with(flag: bool, out_path: str) -> pd.DataFrame:
    """同じ生データから、GENERAL_MARKET_START=flag でデータセットを作る（保存済みなら読む）。"""
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)
    saved = B.GENERAL_MARKET_START
    B.GENERAL_MARKET_START = flag
    B.SWEEP_OVERRIDES["GENERAL_MARKET_START"] = flag
    try:
        B.build(lab.DATA_DIR, out_path)
    finally:
        B.GENERAL_MARKET_START = saved
        B.SWEEP_OVERRIDES.pop("GENERAL_MARKET_START", None)
    return pd.read_parquet(out_path)


def with_outcomes(ds: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """lab.frame() と同じ実収益の列を付ける（同じ関数・同じ結合）。"""
    import sweep_design as S

    ds = ds.copy()
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds["Code"] = ds["Code"].astype(str)
    ref = S.reference_outcome(S.Panels(bars).get(B.HIGH_WINDOW))
    out = ds.merge(ref, on=["Code", "Date"], how="left")
    out = out.merge(lab.realized_returns(bars), on=["Code", "Date"], how="left")
    return out.dropna(subset=["label"]).reset_index(drop=True)


def report_rows(fa: pd.DataFrame, fb: pd.DataFrame, cols) -> None:
    key = ["Code", "Date"]
    both = fb[key].merge(fa[key], on=key, how="inner")
    only_old = fb[key].merge(fa[key], on=key, how="left", indicator=True)
    only_old = only_old[only_old["_merge"] == "left_only"]
    only_new = fa[key].merge(fb[key], on=key, how="left", indicator=True)
    only_new = only_new[only_new["_merge"] == "left_only"]
    print(f"\n■ 1. 行の出入り: 今 {len(fb):,} / 新 {len(fa):,} / 共通 {len(both):,}"
          f"（今だけ {len(only_old):,}・{only_old['Code'].nunique()}銘柄 / "
          f"新だけ {len(only_new):,}）")
    for c, g in only_old.groupby("Code"):
        print(f"    今だけ {c}: {len(g)}行（{g['Date'].min().date()}〜{g['Date'].max().date()}）")
    a = fb.merge(both, on=key).sort_values(key).reset_index(drop=True)
    b = fa.merge(both, on=key).sort_values(key).reset_index(drop=True)
    changed = [c for c in cols + ["listing_years"]
               if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                                  equal_nan=True)]
    print(f"  共通の行で値が変わった列: {len(changed)}本 {changed[:10]}")


def coverage(fa: pd.DataFrame) -> None:
    ly = fa["listing_years"]
    y = fa["Date"].dt.year
    print("\n■ 2. 上場からの年数の充足（年ごと）と、帯ごとの正例率（参考）")
    cap = B.LISTING_CAP_YEARS
    print(f"  打ち止め {cap:g}年")
    print(f"  {'年':<6}{'行':>7}{'値あり':>8}{'打ち止め未満':>12}")
    for yr, g in fa.groupby(y):
        v = g["listing_years"]
        print(f"  {yr:<6}{len(g):>7}{v.notna().mean()*100:>7.1f}%{(v < cap).mean()*100:>11.1f}%")
    edges = sorted({e for e in (0.0, 1.5, 2.0, 3.0, 4.0) if e < cap} | {cap, cap + 1e-4})
    names = [f"{a:g}-{b:g}" for a, b in zip(edges[:-2], edges[1:-1])] + [f"{cap:g}(打ち止め)"]
    band = pd.cut(ly, edges, right=False, labels=names)
    t = fa.groupby(band.cat.add_categories("欠測").fillna("欠測"),
                   observed=False)["label"].agg(["size", "mean"])
    for name, r in t.iterrows():
        print(f"  {name:<12}{int(r['size']):>7}  正例率 {r['mean']*100:5.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験47: TPM の行を数えない・上場からの年数")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default=",".join(L.BOOST))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    algos = [a for a in args.algos.split(",") if a]

    prod = F.columns(BASE_PRESET)
    lst = F.columns("all_plus_listing")
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    fb = with_outcomes(build_with(
        False, os.path.join(lab.DATA_DIR, "dataset_e47_tpm_rows.parquet")), bars)
    fa = with_outcomes(build_with(
        True, os.path.join(lab.DATA_DIR, "dataset_e47_general_market.parquet")), bars)
    del bars
    miss = sorted({c for d in (fa, fb) for c in lst if c not in d.columns})
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:8]}")

    print("=" * 78)
    print(f"実験47 TPM の行を数えない・上場からの年数（種{len(seeds)}つ・ずらし {shifts}か月）")
    for arm, cols in (("T", prod), ("A", prod), ("L", lst)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)
    report_rows(fa, fb, prod)
    coverage(fa)
    fp = permuted(fa, ["listing_years"], seed=PERM_SEED)
    arms = {"T": (fa, prod), "A": (fb, prod), "L": (fa, lst), "P": (fp, lst)}
    AB.compare("e47", arms, "T", LABELS, shifts, seeds, algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
