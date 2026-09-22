#!/usr/bin/env python3
"""
実験36: 合議の線（3モデルすべてが90パーセンタイル以上）はどこに引くべきか。

きっかけ（運用者、2026-09-22）
  クニミネ工業(5388) 2026-09-18 は lgbm 86.5 / xgb 91.7 / cat 91.5。
  最小 86.5 で 90 の線を 3.5pt 下回るため見送りになる。
  「個人的には悪い条件と思わない。保有が1以下なら買うと思う」

実験35 で「日の足切り(発火20件)」が空き枠の機会費用を無視した線だと
分かった。同じ問いをスコアの線にも当てる。

測り方
  しきい値は**本番と同じ基準**で作る。すなわち、その行の属する窓より
  前の窓のスコア分布での百分位（ops_rule.consensus / e30.add_rules と
  同じ。画面の pctHistorical と同じ）。全期間で順位を付けると先の情報が
  混ざるので使わない。

見るもの
  1. 3モデルの最小百分位の帯ごとの実績（枠の制約なし）
  2. 「1モデルだけ下」と「3モデルが揃って低い」で違うか（ばらつきで分ける）
  3. 線を 80/85/87/90/95 に動かして、枠3の運用で何が起きるか
  4. 空き枠の数で線を変える（運用者案）が効くか

結果は research/_data/oof/e36_*.csv。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e27_timing_multi import OOF_DIR, log  # noqa: E402
from e31_strategy import prepare, RULE  # noqa: E402
from e32_takeprofit import attach, forward_paths  # noqa: E402
from e33_portfolio import calendar  # noqa: E402
from e35_slack import simulate  # noqa: E402

BOOST = ("lgbm", "xgb", "cat")
SLOTS = 3
TOP_K = 2
MIN_BREAK = 8          # 実験35 で決めた日の足切り


def hist_pct(d: pd.DataFrame) -> pd.DataFrame:
    """
    本番と同じ百分位を付ける: その行より前の窓のスコア分布での位置。

    窓1は参照する分布が無いので NaN（usable=False と同じ扱い）。
    """
    for a in BOOST:
        out = np.full(len(d), np.nan)
        for f in sorted(d["fold"].unique()):
            prev = d.loc[d["fold"] < f, f"s_{a}"].to_numpy()
            if len(prev) < 500:
                continue
            m = (d["fold"] == f).to_numpy()
            ref = np.sort(prev)
            out[m] = np.searchsorted(ref, d.loc[m, f"s_{a}"].to_numpy(),
                                     side="left") / len(ref) * 100.0
        d[f"hp_{a}"] = out
    cols = [f"hp_{a}" for a in BOOST]
    d["hp_min"] = d[cols].min(axis=1)
    d["hp_max"] = d[cols].max(axis=1)
    d["hp_spread"] = d["hp_max"] - d["hp_min"]
    return d


def signals(d: pd.DataFrame, pct: float, top_k: int = TOP_K) -> pd.DataFrame:
    """3モデルすべてが pct 以上の銘柄を、その日の上位 top_k 件まで。"""
    s = d[d["hp_min"] >= pct].sort_values(["Date", "hp_min"], ascending=[True, False])
    s = s.assign(rank=s.groupby("Date").cumcount() + 1)
    return s[s["rank"] <= top_k].sort_values(["Date", "rank"]).reset_index(drop=True)


def band_row(nm: str, g: pd.DataFrame) -> str:
    if len(g) < 5:
        return f"  {nm:<14}{len(g):>6}   （件数が少なすぎる）"
    r = g["r"]
    return (f"  {nm:<14}{len(g):>6}{r.mean():>+9.2f}%{g['tp20'].mean():>+9.2f}%"
            f"{(r > 0).mean()*100:>6.0f}%{g['label'].mean()*100:>7.0f}%"
            f"{r.std(ddof=1)/np.sqrt(len(r)):>7.2f}{(r < -10).mean()*100:>8.1f}%")


BAND_HEAD = (f"  {'帯':<14}{'件数':>6}{'持ち切り':>10}{'+20%利確':>10}{'勝率':>6}"
             f"{'正例率':>7}{'SE':>7}{'−10%割れ':>9}")


def main() -> int:
    d = prepare()
    d = attach(d, forward_paths())
    d = hist_pct(d)
    d = d[d["entry"].notna() & d["r"].notna() & d["hp_min"].notna()]
    log(f"対象 {len(d):,}件 / {d['Date'].min().date()}〜{d['Date'].max().date()}")

    agree = (d["hp_min"] >= 90)
    log(f"  hp_min>=90 {int(agree.sum()):,}件 / 本番の規則列 {int(d[RULE].sum()):,}件 "
        f"/ 一致率 {(agree == d[RULE].astype(bool)).mean()*100:.1f}%")

    lo = d[d["n_break"] >= MIN_BREAK]
    print(f"\n=== 1. 3モデルの最小百分位の帯ごと（発火{MIN_BREAK}件以上の日・枠の制約なし）===")
    print(BAND_HEAD)
    bins = [0, 70, 80, 85, 90, 95, 100.1]
    names = ["〜70", "70〜80", "80〜85", "85〜90", "90〜95", "95〜"]
    lo = lo.assign(帯=pd.cut(lo["hp_min"], bins, labels=names, right=False))
    rows = []
    for b, g in lo.groupby("帯", observed=True):
        print(band_row(str(b), g))
        rows.append({"band": str(b), "n": len(g), "hold": g["r"].mean(),
                     "tp20": g["tp20"].mean(), "win": (g["r"] > 0).mean()*100,
                     "label": g["label"].mean()*100, "lose10": (g["r"] < -10).mean()*100})
    print(f"\n  まとめ:")
    for nm, m in (("85〜90（クニミネの帯）", (lo["hp_min"] >= 85) & (lo["hp_min"] < 90)),
                  ("90以上（いまの基準）", lo["hp_min"] >= 90),
                  ("85以上（線を下げる）", lo["hp_min"] >= 85),
                  ("母集団（この日の全候補）", lo["hp_min"].notna())):
        print(band_row(nm, lo[m]))

    print(f"\n=== 2. 「1モデルだけ下」か「3モデルとも低い」か（85〜90の帯）===")
    b = lo[(lo["hp_min"] >= 85) & (lo["hp_min"] < 90)].copy()
    print(f"  3モデルのばらつき（最大−最小）で分ける。クニミネは 86.5/91.7/91.5 "
          f"＝ばらつき 5.2")
    print(BAND_HEAD)
    b["幅"] = pd.cut(b["hp_spread"], [0, 2, 5, 10, 200],
                     labels=["〜2（3つとも横並び）", "2〜5", "5〜10（1つだけ下）",
                             "10〜（大きく割れる）"], right=False)
    for k, g in b.groupby("幅", observed=True):
        print(band_row(str(k), g))

    print(f"\n=== 3. 線を動かす（枠{SLOTS}・発火{MIN_BREAK}件以上・上位{TOP_K}件・+20%利確）===")
    cal = calendar()
    print("  {:<16}{:>6}{:>6}{:>9}{:>6}{:>7}{:>9}{:>8}{:>8}".format(
        "線", "取引", "年間", "平均", "勝率", "稼働率", "資産倍率", "年率", "最悪"))
    keep = {}
    for pct in (80.0, 85.0, 87.0, 90.0, 93.0, 95.0):
        sig = signals(d, pct)
        s = simulate(sig, cal, {i: MIN_BREAK for i in range(1, SLOTS + 1)}, slots=SLOTS)
        keep[pct] = s
        if not s.get("taken"):
            print(f"  {pct:.0f}以上          （該当なし）")
            continue
        print(f"  {f'{pct:.0f}以上':<16}{s['taken']:>6}{s['per_year']:>6.1f}"
              f"{s['mean']:>+8.2f}%{s['win']:>5.0f}%{s['util']:>6.0f}%"
              f"{s['equity']:>9.2f}{s['cagr']:>+7.1f}%{s['worst']:>+7.1f}%")

    print(f"\n=== 4. 年ごと（線 85 と 90）===")
    print(f"  {'年':<6}" + f"{'85: 取引':>10}{'平均':>9}" + f"{'90: 取引':>11}{'平均':>9}")
    ta, tb = keep[85.0]["trades"].copy(), keep[90.0]["trades"].copy()
    for t in (ta, tb):
        t["year"] = pd.to_datetime(t["Date"]).dt.year
    for y in range(2021, 2027):
        out = f"  {y:<6}"
        for t, w in ((ta, 10), (tb, 11)):
            g = t[t["year"] == y]
            out += (f"{len(g):>{w}}{g['r' if 'r' in g else 'ret'].mean():>+8.2f}%"
                    if len(g) else f"{0:>{w}}{'—':>9}")
        print(out)

    print(f"\n=== 5. 空き枠の数で線を変える（枠{SLOTS}・発火{MIN_BREAK}件以上）===")
    print("  {:<28}{:>6}{:>6}{:>9}{:>6}{:>7}{:>9}{:>8}".format(
        "腕", "取引", "年間", "平均", "勝率", "稼働率", "資産倍率", "年率"))
    ladders = {
        "線90 固定（いまの基準）": {1: 90.0, 2: 90.0, 3: 90.0},
        "線85 固定": {1: 85.0, 2: 85.0, 3: 85.0},
        "運用者案 1枠:90 2枠:85 3枠:85": {1: 90.0, 2: 85.0, 3: 85.0},
        "1枠:90 2枠:87 3枠:85": {1: 90.0, 2: 87.0, 3: 85.0},
    }
    out_rows = []
    for nm, lad in ladders.items():
        # 線が空き枠で変わるので、全部の行を渡して simulate 側では発火だけ見る
        # → ここでは線ごとに信号を作り、空き枠に応じて採否を決める簡易版を使う
        s = simulate_line(d, cal, lad)
        out_rows.append({"arm": nm, **{k: v for k, v in s.items() if k != "trades"}})
        print(f"  {nm:<28}{s['taken']:>6}{s['per_year']:>6.1f}{s['mean']:>+8.2f}%"
              f"{s['win']:>5.0f}%{s['util']:>6.0f}%{s['equity']:>9.2f}{s['cagr']:>+7.1f}%")

    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e36_bands.csv"), index=False)
    pd.DataFrame(out_rows).to_csv(os.path.join(OOF_DIR, "e36_ladder.csv"), index=False)
    log(f"記録: {OOF_DIR}/e36_*")
    return 0


def simulate_line(d: pd.DataFrame, cal: pd.Series, ladder: dict, *,
                  slots: int = SLOTS, target: float = 20.0,
                  min_break: int = MIN_BREAK, top_k: int = TOP_K) -> dict:
    """空き枠の数でスコアの線が変わる版。入り・出口は実験35 と同じ。"""
    from e32_takeprofit import HOLD
    k = int(target)
    lowest = min(ladder.values())
    sig = signals(d, lowest, top_k)
    sig = sig[sig["n_break"] >= min_break].reset_index(drop=True)
    idx = {x: i for i, x in enumerate(cal)}
    pos: list[dict | None] = [None] * slots
    busy = [0] * slots
    cash = 1.0
    done, skip_full, skip_gate = [], 0, 0
    for _, row in sig.iterrows():
        di = idx.get(pd.Timestamp(row["Date"]))
        if di is None or not np.isfinite(row[f"tp{k}"]):
            continue
        buy = di + 1
        if buy >= len(cal):
            continue
        for s, p in enumerate(pos):
            if p is not None and p["free"] <= buy:
                busy[s] += p["free"] - p["buy"]
                cash += p["cost"] * (1.0 + p["ret"] / 100.0)
                done.append({"Date": p["date"], "Code": p["code"], "ret": p["ret"],
                             "days": p["free"] - p["buy"], "hp_min": p["hp_min"],
                             "n_break": p["n_break"], "slot": s})
                pos[s] = None
        empty = [s for s, p in enumerate(pos) if p is None]
        if not empty:
            skip_full += 1
            continue
        if float(row["hp_min"]) < ladder[len(empty)]:
            skip_gate += 1
            continue
        hit = bool(row[f"hit{k}"])
        days = int(row[f"day{k}"]) if hit and pd.notna(row[f"day{k}"]) else HOLD
        cost = (cash + sum(q["cost"] for q in pos if q)) / slots
        cash -= cost
        pos[empty[0]] = {"code": row["Code"], "date": row["Date"], "buy": buy,
                         "free": buy + days, "ret": float(row[f"tp{k}"]), "cost": cost,
                         "hp_min": float(row["hp_min"]), "n_break": int(row["n_break"])}
    for s, p in enumerate(pos):
        if p is not None:
            busy[s] += p["free"] - p["buy"]
            cash += p["cost"] * (1.0 + p["ret"] / 100.0)
            done.append({"Date": p["date"], "Code": p["code"], "ret": p["ret"],
                         "days": p["free"] - p["buy"], "hp_min": p["hp_min"],
                         "n_break": p["n_break"], "slot": s})
    t = pd.DataFrame(done)
    if not len(t):
        return {"taken": 0}
    years = (pd.Timestamp(sig["Date"].max()) - pd.Timestamp(sig["Date"].min())).days / 365.25
    span = max(1, idx[pd.Timestamp(sig["Date"].max())] - idx[pd.Timestamp(sig["Date"].min())] + 20)
    return {"taken": len(t), "skip_full": skip_full, "skip_gate": skip_gate,
            "per_year": len(t) / years, "mean": t["ret"].mean(),
            "win": (t["ret"] > 0).mean() * 100, "util": float(np.mean(busy)) / span * 100,
            "equity": cash, "cagr": (cash ** (1 / years) - 1) * 100,
            "worst": t["ret"].min(), "trades": t}


if __name__ == "__main__":
    raise SystemExit(main())
