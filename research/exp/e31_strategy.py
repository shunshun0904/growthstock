#!/usr/bin/env python3
"""
実験31: 「地合いが良く発火の多い日に、上位1〜2銘柄だけ買う」戦略の検証。

運用者の案（2026-09-22）
  その日の発火数が多い日に限り、3モデル 90以上を満たした銘柄の
  スコア上位1〜2件を買う。発火が少ない日は見送る。

実験30 で「発火26件以上の日の最良1件は正例率 39.7〜41.1%・+3.10〜3.28%」と
出たことが発端。ただし **26件という線は5分割の帯の上端を後から選んだもの**
なので、そのまま信じると選び過ぎになる。ここでは

  1. しきい値を 1〜40件まで動かして、線の位置に依存しないかを見る
  2. 上位1件・2件・3件・全件で、順位ごとの落ち方を見る
  3. 年ごと・窓ごとにばらつきを見る（1つの年に偏っていないか）
  4. 地合いの条件（TOPIX の200日線より上）を足したときの変化を見る
  5. 見送ることで何を捨てているか（機会損失）を出す

物差しは ret_o1_20（翌営業日の寄りで買い、20営業日後）。
スコアは実験27 の腕 B1（本番と同じ153列・本番のパラメータ・種3つの平均）。
結果は research/_data/oof/e31_*.csv。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import ops_rule as OR  # noqa: E402
from e27_timing_multi import OOF_DIR, log  # noqa: E402
from e30_day_regime import add_rules, load_scores  # noqa: E402

OUTCOME = "ret_o1_20"
RULE = "3モデル 90以上"
YEARS = 4.8          # out-of-fold の対象期間（2021-11〜2026-08）


def prepare() -> pd.DataFrame:
    df = add_rules(load_scores())
    d = df[df["usable"]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.merge(d.groupby("Date").size().rename("n_break"), on="Date")
    fr = lab.frame()
    fr["Date"] = pd.to_datetime(fr["Date"])
    mkt = fr[["Date", "topix_ma200_gap"]].drop_duplicates("Date")
    d = d.merge(mkt, on="Date", how="left")
    d = d.merge(fr[["Code", "Date", "ref_rise"]], on=["Code", "Date"], how="left")
    d["r"] = pd.to_numeric(d[OUTCOME], errors="coerce") * 100
    d["rise"] = pd.to_numeric(d["ref_rise"], errors="coerce") * 100
    d["year"] = d["Date"].dt.year
    return d


def trades(d: pd.DataFrame, min_break: int, top_k: int, *,
           rule: str = RULE, gap_min: float | None = None) -> pd.DataFrame:
    """条件を満たす日の、基準を満たした銘柄のスコア上位 top_k 件。"""
    sel = d[d[rule] & (d["n_break"] >= min_break)]
    if gap_min is not None:
        sel = sel[sel["topix_ma200_gap"] >= gap_min]
    if not len(sel):
        return sel
    sel = sel.sort_values(["Date", "s_lgbm"], ascending=[True, False])
    sel = sel.assign(rank=sel.groupby("Date").cumcount() + 1)
    return sel[sel["rank"] <= top_k]


def stats(t: pd.DataFrame, d: pd.DataFrame) -> dict:
    if not len(t):
        return {"n": 0}
    r = t["r"].dropna()
    per_year = t.groupby("year")["r"].mean()
    per_fold = t.groupby("fold")["r"].mean()
    days = t["Date"].nunique()
    return {"n": len(t), "days": days, "per_year": len(t) / YEARS,
            "mean": r.mean(), "median": r.median(), "win": (r > 0).mean() * 100,
            "label": t["label"].mean() * 100, "rise": t["rise"].mean(),
            "worst": r.min(), "p10": r.quantile(0.1), "p90": r.quantile(0.9),
            "se": r.std(ddof=1) / np.sqrt(len(r)),
            "year_min": per_year.min(), "year_max": per_year.max(),
            "years_pos": int((per_year > 0).sum()), "n_years": len(per_year),
            "fold_min": per_fold.min(), "folds_pos": int((per_fold > 0).sum()),
            "n_folds": len(per_fold)}


def line(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {name:<28} （該当なし）"
    return (f"  {name:<28}{s['n']:>6}{s['per_year']:>7.0f}{s['mean']:>+8.2f}%{s['se']:>6.2f}"
            f"{s['median']:>+8.2f}%{s['win']:>7.0f}%{s['label']:>7.0f}%{s['rise']:>+8.1f}%"
            f"{s['p10']:>+8.1f}%{s['worst']:>+8.1f}%"
            f"{s['years_pos']:>3}/{s['n_years']:<2}{s['folds_pos']:>4}/{s['n_folds']:<2}")


HEAD = (f"  {'条件':<28}{'取引':>6}{'年間':>7}{'平均':>8}{'SE':>6}{'中央値':>8}{'勝率':>7}"
        f"{'正例率':>7}{'最大上昇':>8}{'下位10%':>8}{'最悪':>8}{'勝ち年':>6}{'勝ち窓':>7}")


def main() -> int:
    d = prepare()
    log(f"対象 {d['Date'].nunique():,}営業日 / {len(d):,}件 / "
        f"{d['Date'].min().date()}〜{d['Date'].max().date()}")

    print("\n=== 1. 運用者の案（3モデル 90以上・発火26件以上・上位1〜2件）===")
    print(HEAD)
    base = trades(d, 0, 99)
    print(line("基準のみ（全日・全件）", stats(base, d)))
    for k in (1, 2, 3):
        print(line(f"全日・上位{k}件", stats(trades(d, 0, k), d)))
    for k in (1, 2, 3, 99):
        nm = f"発火26件以上・上位{k}件" if k < 99 else "発火26件以上・全件"
        print(line(nm, stats(trades(d, 26, k), d)))

    print("\n=== 2. しきい値をずらす（上位2件）。26件という線に依存していないか ===")
    print(HEAD)
    rows = []
    for m in (1, 5, 10, 15, 20, 26, 30, 35, 40):
        s = stats(trades(d, m, 2), d)
        rows.append({"min_break": m, **s})
        print(line(f"発火{m}件以上・上位2件", s))
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e31_threshold.csv"), index=False)

    print("\n=== 3. 順位ごとの落ち方（発火26件以上）===")
    sel = trades(d, 26, 99)
    print(f"  {'順位':<8}{'取引':>6}{'平均':>9}{'中央値':>9}{'勝率':>7}{'正例率':>7}{'最大上昇':>9}")
    for rk in range(1, 7):
        g = sel[sel["rank"] == rk]
        if len(g) < 10:
            continue
        print(f"  {rk:<8}{len(g):>6}{g['r'].mean():>+8.2f}%{g['r'].median():>+8.2f}%"
              f"{(g['r'] > 0).mean()*100:>6.0f}%{g['label'].mean()*100:>6.0f}%{g['rise'].mean():>+8.1f}%")

    print("\n=== 4. 地合いの条件を足す（TOPIX が200日線より上）===")
    print(HEAD)
    for m, k in ((26, 2), (15, 2), (0, 2)):
        print(line(f"発火{m}件以上・上位2件・地合い問わず", stats(trades(d, m, k), d)))
        print(line(f"発火{m}件以上・上位2件・200日線より上", stats(trades(d, m, k, gap_min=0.0), d)))

    print("\n=== 5. 年ごと（発火26件以上・上位2件）===")
    t = trades(d, 26, 2)
    print(f"  {'年':<6}{'取引':>6}{'平均':>9}{'中央値':>9}{'勝率':>7}{'正例率':>7}"
          f"   （参考）全日・上位2件")
    a = trades(d, 0, 2)
    for y, g in t.groupby("year"):
        ga = a[a["year"] == y]
        print(f"  {y:<6}{len(g):>6}{g['r'].mean():>+8.2f}%{g['r'].median():>+8.2f}%"
              f"{(g['r'] > 0).mean()*100:>6.0f}%{g['label'].mean()*100:>6.0f}%"
              f"   {len(ga):>4}件 {ga['r'].mean():>+6.2f}%")

    print("\n=== 6. 見送ることで何を捨てているか（発火26件未満の日の、上位2件）===")
    print(HEAD)
    skip = trades(d, 0, 2)
    skip = skip[skip["n_break"] < 26]
    print(line("見送る側（発火26件未満）", stats(skip, d)))
    for lo, hi in ((0, 8), (8, 15), (15, 26)):
        g = skip[(skip["n_break"] >= lo) & (skip["n_break"] < hi)]
        print(line(f"  うち発火{lo}〜{hi-1}件", stats(g, d)))
    t.to_csv(os.path.join(OOF_DIR, "e31_trades.csv"), index=False)
    log(f"記録: {OOF_DIR}/e31_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
