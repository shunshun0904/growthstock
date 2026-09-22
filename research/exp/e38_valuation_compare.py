#!/usr/bin/env python3
"""
実験38: PER / PBR / ROE / BPS / EPS は、API と自前計算のどちらが正確か。

背景
----
`/equities/valuation` に `BPS` `EPS` `PER` `PBR` `ROE` `MktCap` があり、
`FwdEPS` `FwdPER` `FwdROE` も付く（実測4,359銘柄/日）。このうち
PER/PBR/ROE は自前計算の同名特徴量と重なる。**両方は持たない**ので、
どちらが正確かを測って片方に決める。

自前計算の作り（build_dataset.py）
  per    = 終値 / eps_ttm（単期EPSの4期和、4期そろったときだけ）。
           eps_ttm>0 のときだけ。PER_MAX で上限
  pbr    = 終値 / BPS（BPS は API 提供値を優先し、無ければ 株主資本/株数）
  ROE_q0 = API の ROE を優先し、無ければ TTM純利益/自己資本。±500 に丸め

測ること
------
1. **充足率** 同じ行で、どちらがどれだけ埋まるか
2. **内部整合** API の値どうしが終値と合うか
     PER × EPS ≒ 終値 / PBR × BPS ≒ 終値
   合わないなら、その API 値は別の定義（予想EPS など）を指している
3. **素の値との突き合わせ** 決算（fins）の生の EPS・BPS と比べる
4. **異常値** 発散・負・桁外れがどれだけ出るか。自前の per は実測で
   4.0e17 まで出て上限が要った（build_dataset.py のコメント）
5. **一致度** 両方ある行での相関と相対差

結果は research/_data/oof/e38_*.csv。
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
from e27_timing_multi import OOF_DIR, log  # noqa: E402

#: API の列 -> (自前の列, 説明)
PAIRS = {
    "PER": ("per", "株価収益率"),
    "PBR": ("pbr", "純資産倍率"),
    "ROE": ("ROE_q0", "自己資本利益率(%)"),
}
#: API 値に掛けて自前と単位を揃える係数。
#: **API の ROE / FwdROE は小数**（0.0791 = 7.91%）。実測（2026-09-22、
#: valuation_2025）で FwdEPS/BPS の中央値 0.0791 が API の FwdROE と
#: 一致することを確かめた。自前の ROE_q0 は %。揃えずに比べると
#: 「100倍ずれている」としか出ず、どちらが正確かの判定にならない。
SCALE = {"ROE": 100.0, "FwdROE": 100.0}
#: 自前に対応が無い列（新規情報）
NEW_COLS = ["FwdEPS", "FwdPER", "FwdROE", "MktCap"]


def load_valuation(data_dir: str = None) -> pd.DataFrame:
    data_dir = data_dir or lab.DATA_DIR
    paths = sorted(glob.glob(os.path.join(data_dir, "valuation_*.parquet")))
    if not paths:
        return pd.DataFrame()
    v = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    v["Date"] = pd.to_datetime(v["Date"])
    v["Code"] = v["Code"].astype(str)
    for c, k in SCALE.items():                   # 単位を自前に合わせる
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce") * k
    return v


def pct(x) -> str:
    return "—" if not np.isfinite(x) else f"{x*100:.1f}%"


def describe(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce")
    ok = v[np.isfinite(v)]
    return {"充足": v.notna().mean(), "件数": int(ok.size),
            "最小": ok.min() if ok.size else np.nan,
            "中央値": ok.median() if ok.size else np.nan,
            "最大": ok.max() if ok.size else np.nan,
            "負": (ok < 0).mean() if ok.size else np.nan}


def main() -> int:
    v = load_valuation()
    if not len(v):
        log("valuation_*.parquet がまだありません。取り込みを待つ")
        return 0
    log(f"valuation {len(v):,}行 / {v['Date'].min().date()}〜{v['Date'].max().date()}"
        f" / {v['Code'].nunique():,}銘柄")
    log(f"  列: {', '.join(sorted(v.columns))}")

    fr = lab.frame()
    fr["Date"] = pd.to_datetime(fr["Date"])
    fr["Code"] = fr["Code"].astype(str)
    need = ["Code", "Date", "close_raw", "eps_ttm", "BPS", "market_cap"] + \
           [c for _, (c, _) in zip(PAIRS, PAIRS.values())]
    have = [c for c in dict.fromkeys(need) if c in fr.columns]
    d = fr[have].merge(v, on=["Code", "Date"], how="inner", suffixes=("", "_api"))
    log(f"突き合わせ {len(d):,}行（母集団のうち valuation がある行）")
    if len(d) < 50:
        log("重なりが少なすぎる。取り込みが進んでから回す")
        return 0

    print(f"\n=== 1. 充足率と値の広がり（突き合わせた {len(d):,}行）===")
    print(f"  {'列':<22}{'充足':>8}{'最小':>14}{'中央値':>12}{'最大':>14}{'負':>8}")
    rows = []
    for api, (own, ja) in PAIRS.items():
        for name, col in ((f"API {api}", api), (f"自前 {own}", own)):
            if col not in d.columns:
                print(f"  {name:<22}   （列が無い）")
                continue
            st = describe(d[col])
            rows.append({"col": name, "ja": ja, **st})
            print(f"  {name:<22}{pct(st['充足']):>8}{st['最小']:>14.4g}"
                  f"{st['中央値']:>12.4g}{st['最大']:>14.4g}{pct(st['負']):>8}")
        print()
    for c in NEW_COLS:
        if c in d.columns:
            st = describe(d[c])
            rows.append({"col": f"API {c}", "ja": "（自前に対応なし）", **st})
            print(f"  {f'API {c}':<22}{pct(st['充足']):>8}{st['最小']:>14.4g}"
                  f"{st['中央値']:>12.4g}{st['最大']:>14.4g}{pct(st['負']):>8}")
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e38_coverage.csv"), index=False)

    print(f"\n=== 2. 内部整合: API の値どうしが終値と合うか ===")
    px = pd.to_numeric(d["close_raw"], errors="coerce")
    checks = [("PER × EPS = 終値", "PER", "EPS"), ("PBR × BPS = 終値", "PBR", "BPS_api"
                                                  if "BPS_api" in d.columns else "BPS")]
    print(f"  {'関係':<22}{'件数':>8}{'誤差中央値':>12}{'1%以内':>9}{'5%以内':>9}")
    for ja, a, b in checks:
        if a not in d.columns or b not in d.columns:
            print(f"  {ja:<22}   （列が無い）")
            continue
        est = pd.to_numeric(d[a], errors="coerce") * pd.to_numeric(d[b], errors="coerce")
        rel = ((est - px) / px).abs()
        ok = rel[np.isfinite(rel)]
        if not ok.size:
            print(f"  {ja:<22}   （比べられる行が無い）")
            continue
        print(f"  {ja:<22}{ok.size:>8,}{ok.median()*100:>11.2f}%"
              f"{(ok < 0.01).mean()*100:>8.0f}%{(ok < 0.05).mean()*100:>8.0f}%")
    print("  ※ 合わないなら、その API 値は別の定義（予想EPS など）を指している")

    print(f"\n=== 3. 素の値との突き合わせ（自前が使っている元データ）===")
    print(f"  {'比べるもの':<28}{'件数':>8}{'相関':>8}{'誤差中央値':>12}{'5%以内':>9}")
    for ja, a, b in (("API EPS vs 自前 eps_ttm", "EPS", "eps_ttm"),
                     ("API BPS vs 自前 BPS", "BPS_api" if "BPS_api" in d.columns
                      else "BPS", "BPS"),
                     ("API MktCap vs 自前 market_cap(億)", "MktCap", "market_cap")):
        if a not in d.columns or b not in d.columns or a == b:
            print(f"  {ja:<28}   （比べられない）")
            continue
        x = pd.to_numeric(d[a], errors="coerce")
        y = pd.to_numeric(d[b], errors="coerce")
        if a == "MktCap":
            y = y * 1e8                      # 億円 -> 円
        m = np.isfinite(x) & np.isfinite(y) & (y != 0)
        if m.sum() < 20:
            print(f"  {ja:<28}{int(m.sum()):>8}   （少なすぎる）")
            continue
        rel = ((x[m] - y[m]) / y[m]).abs()
        print(f"  {ja:<28}{int(m.sum()):>8,}{np.corrcoef(x[m], y[m])[0,1]:>8.3f}"
              f"{rel.median()*100:>11.2f}%{(rel < 0.05).mean()*100:>8.0f}%")

    print(f"\n=== 4. 一致度（両方ある行）===")
    print(f"  {'対':<22}{'件数':>8}{'相関':>8}{'誤差中央値':>12}{'5%以内':>9}{'10%超':>8}")
    for api, (own, ja) in PAIRS.items():
        if api not in d.columns or own not in d.columns:
            continue
        x = pd.to_numeric(d[api], errors="coerce")
        y = pd.to_numeric(d[own], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y) & (y != 0)
        if m.sum() < 20:
            print(f"  {f'{api} / {own}':<22}{int(m.sum()):>8}   （少なすぎる）")
            continue
        rel = ((x[m] - y[m]) / y[m]).abs()
        print(f"  {f'{api} / {own}':<22}{int(m.sum()):>8,}"
              f"{np.corrcoef(x[m], y[m])[0,1]:>8.3f}{rel.median()*100:>11.2f}%"
              f"{(rel < 0.05).mean()*100:>8.0f}%{(rel > 0.10).mean()*100:>7.0f}%")

    print(f"\n=== 5. 片方にしか無い行（どちらが拾えているか）===")
    print(f"  {'対':<22}{'API だけ':>10}{'自前だけ':>10}{'両方':>8}{'どちらも無し':>12}")
    for api, (own, ja) in PAIRS.items():
        if api not in d.columns or own not in d.columns:
            continue
        a = pd.to_numeric(d[api], errors="coerce").notna()
        b = pd.to_numeric(d[own], errors="coerce").notna()
        print(f"  {f'{api} / {own}':<22}{int((a & ~b).sum()):>10,}"
              f"{int((~a & b).sum()):>10,}{int((a & b).sum()):>8,}"
              f"{int((~a & ~b).sum()):>12,}")

    # 充足率の差をそのまま「API の勝ち」と読むと間違える。
    #
    # 自前の per は eps_ttm>0 のときだけ作る（赤字の会社の PER は
    # 意味を持たないため NaN にしている）。API が同じ行に負の PER を
    # 返しているなら、それは「拾えている」のではなく
    # **使えない値が埋まっているだけ**で、充足率は見かけ上だけ上がる。
    # 運用者の判定基準は「欠損率や値の正当性」なので、差の中身を出す。
    print(f"\n=== 6. API にだけ値がある行の中身（充足率の差の正体）===")
    print(f"  {'対':<22}{'件数':>8}{'負':>8}{'極端':>8}{'中央値':>12}"
          f"{'まともな増分':>14}")
    LIMITS = {"PER": 200.0, "PBR": 50.0, "ROE": 200.0}   # これを超えたら極端
    for api, (own, ja) in PAIRS.items():
        if api not in d.columns or own not in d.columns:
            continue
        x = pd.to_numeric(d[api], errors="coerce")
        y = pd.to_numeric(d[own], errors="coerce")
        only = x[x.notna() & y.isna()]
        only = only[np.isfinite(only)]
        if not only.size:
            print(f"  {f'{api} / {own}':<22}{0:>8}   （差なし）")
            continue
        lim = LIMITS.get(api, np.inf)
        neg = (only < 0)
        ext = (only.abs() > lim)
        good = only[~neg & ~ext]
        print(f"  {f'{api} / {own}':<22}{only.size:>8,}"
              f"{neg.mean()*100:>7.0f}%{ext.mean()*100:>7.0f}%"
              f"{only.median():>12.4g}{good.size/max(len(d),1)*100:>13.1f}pt")
    print("  ※ 「まともな増分」= 負でも極端でもない行が、母集団全体に対して")
    print("     何ポイント充足率を押し上げるか。ここが薄いなら充足率の差は見かけ")

    d.to_parquet(os.path.join(OOF_DIR, "e38_merged.parquet"), index=False)
    log(f"記録: {OOF_DIR}/e38_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
