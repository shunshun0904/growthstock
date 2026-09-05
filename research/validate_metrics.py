#!/usr/bin/env python3
"""
バリュエーション指標の値そのものを検証する。

なぜ必要か:
  PER / PBR / ROE / ROA は API に無く、株価と決算から自前で計算している。
  割り算で作る値は分母が小さいときに発散する。
  「列は存在し、学習も通るが、値が壊れている」状態はこれまで何度も起きた。
  モデルに入れる前に、値の定義として最低限の保証を確認する。

確認すること:
  1. 発散していないか（±inf、極端な絶対値）
  2. 全部ゼロ・全部欠測になっていないか
  3. 恒等式が成り立つか
       per × earnings_yield == 100
       pbr × book_yield     == 1
     成り立たなければ、どちらかの計算が壊れている
  4. 分布（最小/1%/中央値/99%/最大）。閾値は推測でなく実測から決める
  5. 符号の整合。株価は正なので
       per の符号 == eps_ttm の符号
       pbr の符号 == BPS の符号

ROE / ROA の上限について:
  比率だが 100% を超えることは実際にある。
  ROE = 純利益 / 自己資本 で、自己資本が小さい会社（自社株買いを重ねた、
  あるいは債務超過に近い）では 100% を超える。ROA も資産の軽い業種では起こりうる。
  よって「100%以下」を必須条件にはしない。
  上限を機械的に切ると、実在する高収益企業を落とすことになる。
  ここでは「発散（分母が丸め誤差レベル）」と「実在しうる高い値」を
  分けて報告し、前者だけを異常として扱う。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import DATA_DIR  # noqa: E402

# 検証対象と、その値がどこまでなら実在しうるかの目安。
# 目安を超えたものは「異常」ではなく「要確認」として件数を出す。
METRICS = {
    "per":            {"plausible": (0.0, 1000.0),     "unit": "倍"},
    "pbr":            {"plausible": (0.0, 100.0),      "unit": "倍"},
    "earnings_yield": {"plausible": (-200.0, 100.0),   "unit": "%"},
    "book_yield":     {"plausible": (-10.0, 100.0),    "unit": "倍"},
    "ROE_q0":         {"plausible": (-500.0, 500.0),   "unit": "%"},
    "ROA_q0":         {"plausible": (-200.0, 200.0),   "unit": "%"},
    "equity_ratio_q0": {"plausible": (-100.0, 100.0),  "unit": "%"},
    "BPS":            {"plausible": (0.0, 1e6),        "unit": "円"},
}


def describe(s: pd.Series, plausible) -> Dict:
    v = pd.to_numeric(s, errors="coerce")
    n = len(v)
    finite = v[np.isfinite(v)]
    lo, hi = plausible
    out = {
        "n": int(n),
        "present_pct": round(float(v.notna().mean() * 100), 1),
        "n_inf": int(np.isinf(v.to_numpy(dtype=float)).sum()),
        "n_zero": int((finite == 0).sum()),
        "zero_pct": round(float((finite == 0).mean() * 100), 2) if len(finite) else None,
        "n_negative": int((finite < 0).sum()),
        "outside_plausible": int(((finite < lo) | (finite > hi)).sum()),
    }
    if len(finite):
        q = finite.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        out.update({
            "min": float(q.iloc[0]), "p1": float(q.iloc[1]), "p25": float(q.iloc[2]),
            "median": float(q.iloc[3]), "p75": float(q.iloc[4]),
            "p99": float(q.iloc[5]), "max": float(q.iloc[6]),
        })
    return out


def check_identities(df: pd.DataFrame) -> List[Dict]:
    """恒等式。片方の計算が壊れていれば、ここでずれる。"""
    checks = []

    def rel_err(a, b):
        m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
        if not m.any():
            return None, 0
        return float(np.nanmax(np.abs(a[m] - b[m]) / np.abs(b[m]))), int(m.sum())

    if {"per", "earnings_yield"} <= set(df.columns):
        a = (df["per"] * df["earnings_yield"]).to_numpy(dtype=float)
        err, n = rel_err(a, np.full(len(a), 100.0))
        checks.append({"name": "per × earnings_yield == 100",
                       "max_rel_err": err, "n_checked": n})

    if {"pbr", "book_yield"} <= set(df.columns):
        a = (df["pbr"] * df["book_yield"]).to_numpy(dtype=float)
        err, n = rel_err(a, np.ones(len(a)))
        checks.append({"name": "pbr × book_yield == 1",
                       "max_rel_err": err, "n_checked": n})

    # 符号の整合。株価は正なので、PER の符号は EPS の符号と一致するはず
    for ratio, base in (("per", "eps_ttm"), ("pbr", "BPS")):
        if {ratio, base} <= set(df.columns):
            r, b = df[ratio].to_numpy(dtype=float), df[base].to_numpy(dtype=float)
            m = np.isfinite(r) & np.isfinite(b) & (r != 0) & (b != 0)
            bad = int((np.sign(r[m]) != np.sign(b[m])).sum())
            checks.append({"name": f"sign({ratio}) == sign({base})",
                           "mismatches": bad, "n_checked": int(m.sum())})
    return checks


def build_report(stats: Dict[str, Dict], identities: List[Dict], n_rows: int) -> str:
    lines = [
        "# バリュエーション指標の検証",
        "",
        "`research/validate_metrics.py` の出力。**実測値のみ**を記載する。",
        "",
        "PER / PBR / ROE / ROA は API に無く、株価と決算から自前で計算している。",
        "割り算で作る値は分母が小さいと発散するため、モデルに入れる前に値を確認する。",
        "",
        f"- サンプル数: {n_rows:,}",
        "",
        "## 恒等式",
        "",
        "逆数どうしの積が定数になるはず。ずれていれば計算が壊れている。",
        "",
        "| 検査 | 結果 |",
        "| --- | --- |",
    ]
    for c in identities:
        if "max_rel_err" in c:
            e = c["max_rel_err"]
            v = "検査対象なし" if e is None else f"最大相対誤差 {e:.3e}（{c['n_checked']:,}件）"
        else:
            v = f"符号不一致 {c['mismatches']}件 / {c['n_checked']:,}件"
        lines.append(f"| `{c['name']}` | {v} |")

    lines += [
        "",
        "## 分布",
        "",
        "| 指標 | 充足率 | inf | ゼロ | 負 | 最小 | 1% | 中央値 | 99% | 最大 | 目安外 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, st in stats.items():
        if "median" not in st:
            lines.append(f"| `{name}` | {st['present_pct']}% | — | — | — "
                         f"| — | — | **値なし** | — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {st['present_pct']}% | {st['n_inf']} | {st['n_zero']:,} "
            f"| {st['n_negative']:,} | {st['min']:.2f} | {st['p1']:.2f} "
            f"| {st['median']:.2f} | {st['p99']:.2f} | {st['max']:.2f} "
            f"| {st['outside_plausible']:,} |")

    lines += [
        "",
        "## 読み方",
        "",
        "- **inf が1件でもあれば計算が壊れている**。分母のガードが効いていない",
        "- 充足率が0%、または全件ゼロなら列が作られていない",
        "- 「目安外」は異常ではなく要確認。ROE/ROA は比率だが100%を超えることが",
        "  実際にある（自己資本の小さい会社など）。機械的に切ると実在する",
        "  高収益企業を落とすため、上限で除外はしない",
        "- 最大値が目安を大きく超えている場合、分母が丸め誤差レベルの可能性が高い。",
        "  その場合は逆数側（`earnings_yield` / `book_yield`）を特徴量に使う",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="バリュエーション指標の検証")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "DATA_VALIDATION.md"))
    ap.add_argument("--strict", action="store_true",
                    help="inf や全欠測を見つけたら異常終了する")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    print(f"[load] {len(df):,}行 / {len(df.columns)}列")

    stats, problems = {}, []
    for name, cfg in METRICS.items():
        if name not in df.columns:
            problems.append(f"{name}: 列が存在しない")
            stats[name] = {"n": 0, "present_pct": 0.0, "n_inf": 0,
                           "n_zero": 0, "n_negative": 0, "outside_plausible": 0}
            continue
        st = describe(df[name], cfg["plausible"])
        stats[name] = st
        if st["n_inf"]:
            problems.append(f"{name}: inf が {st['n_inf']}件")
        if st["present_pct"] == 0.0:
            problems.append(f"{name}: 全件欠測")
        elif st.get("zero_pct") == 100.0:
            problems.append(f"{name}: 全件ゼロ")
        print(f"  {name:<16} 充足 {st['present_pct']:5.1f}%  "
              f"inf {st['n_inf']:>3}  ゼロ {st['n_zero']:>6,}  "
              + (f"中央値 {st['median']:>10.2f}  最大 {st['max']:>12.2f}  "
                 f"目安外 {st['outside_plausible']:,}" if "median" in st else "値なし"))

    identities = check_identities(df)
    print("\n[恒等式]")
    for c in identities:
        if "max_rel_err" in c:
            e = c["max_rel_err"]
            print(f"  {c['name']:<34} "
                  + ("検査対象なし" if e is None
                     else f"最大相対誤差 {e:.3e} ({c['n_checked']:,}件)"))
            if e is not None and e > 1e-6:
                problems.append(f"恒等式が成り立たない: {c['name']} (誤差 {e:.3e})")
        else:
            print(f"  {c['name']:<34} 符号不一致 {c['mismatches']}件 "
                  f"/ {c['n_checked']:,}件")
            if c["mismatches"]:
                problems.append(f"符号が不整合: {c['name']} ({c['mismatches']}件)")

    body = build_report(stats, identities, len(df))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(f"\n[done] {args.out}")
    with open(os.path.join(DATA_DIR, "validation.json"), "w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "identities": identities,
                   "problems": problems}, fh, ensure_ascii=False, indent=2)

    if problems:
        print("\n[異常]")
        for p in problems:
            print(f"  - {p}")
        if args.strict:
            return 1
    else:
        print("\n[異常なし] inf 無し / 全欠測・全ゼロ無し / 恒等式が成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
