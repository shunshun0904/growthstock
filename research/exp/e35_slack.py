#!/usr/bin/env python3
"""
実験35: 枠が空いているとき、発火の少ない日でも買うべきか。

運用者の指摘（2026-09-22）
  「同時発火数が少なくても、他の条件で満たしていれば、かつその時に、
    保有銘柄数が０や１で、資金に余裕があれば買うと思います」

実験31 の「発火20件以上の日だけ買う」は **1取引あたりの成績**だけで引いた線で、
**空き枠の機会費用を勘定に入れていない**。空き枠の収益は 0% である。
発火8〜19件の日の選定銘柄が +1.8〜2.0% なら、枠が遊んでいる限り
それを買うほうが良いはずで、線を「常に20件」に置くのは損かもしれない。

そこで、しきい値を **空き枠の数の関数**にして測る。

  梯子の書き方   {空き枠: 必要な発火数}
    {1: 20, 2: 8, 3: 1}  = 残り1枠なら20件以上、残り2枠なら8件以上、
                           何も持っていなければ発火があれば買う

比較する腕
  固定20      いまの手順（空き枠によらず20件以上）
  運用者案    {1: 20, 2: 8, 3: 1}
  控えめ      {1: 20, 2: 15, 3: 8}
  もっと緩い  {1: 10, 2: 5, 3: 1}
  開放        発火数を一切見ない

決め手になる数字は「緩めたから取れた取引（発火20件未満の日）」の成績と、
それが埋めた枠が**そのままなら 0% だった**という事実の比較。

選定（3モデル90以上）・入り（翌営業日の寄り）・出口（+20%または20営業日）・
枠3は実験34 と同じ。結果は research/_data/oof/e35_*.csv。
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
from e32_takeprofit import HOLD, attach, forward_paths  # noqa: E402
from e33_portfolio import calendar  # noqa: E402

SLOTS = 3
TOP_K = 2

#: 腕の定義。{空き枠の数: 必要な発火数}。空き枠0 では買えないので書かない
LADDERS = {
    "固定20（いまの手順）": {1: 20, 2: 20, 3: 20},
    "運用者案 1:20 2:8 3:1": {1: 20, 2: 8, 3: 1},
    "控えめ  1:20 2:15 3:8": {1: 20, 2: 15, 3: 8},
    "緩い    1:10 2:5 3:1": {1: 10, 2: 5, 3: 1},
    "開放（発火を見ない）": {1: 1, 2: 1, 3: 1},
    "固定8（空き枠を見ない）": {1: 8, 2: 8, 3: 8},
}


def all_signals(d: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """発火数で絞らず、基準を満たした銘柄をその日の上位 top_k 件まで残す。"""
    s = d[d[RULE]].sort_values(["Date", "p_min"], ascending=[True, False])
    s = s.assign(rank=s.groupby("Date").cumcount() + 1)
    return s[s["rank"] <= top_k].sort_values(["Date", "rank"]).reset_index(drop=True)


def simulate(sig: pd.DataFrame, cal: pd.Series, ladder: dict, *,
             slots: int = SLOTS, target: float = 20.0) -> dict:
    """
    先着順に枠を埋める。買うかどうかは **そのときの空き枠の数**で決まる。
    枠ごとの資金は実資金の複利（買うたびに「現金＋建玉の簿価」の 1/slots）。
    """
    k = int(target)
    idx = {d: i for i, d in enumerate(cal)}
    pos: list[dict | None] = [None] * slots
    busy = [0] * slots
    cash = 1.0
    done: list[dict] = []
    skip_full = skip_gate = 0

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
                done.append({"Date": p["date"], "Code": p["code"], "slot": s,
                             "ret": p["ret"], "days": p["free"] - p["buy"],
                             "n_break": p["n_break"], "free_at_buy": p["free_at_buy"],
                             "hit": p["hit"]})
                pos[s] = None
        empty = [s for s, p in enumerate(pos) if p is None]
        if not empty:
            skip_full += 1
            continue
        # ここが実験の肝。空き枠が多いほど低い発火数でも許す
        if int(row["n_break"]) < ladder[len(empty)]:
            skip_gate += 1
            continue
        hit = bool(row[f"hit{k}"])
        days = int(row[f"day{k}"]) if hit and pd.notna(row[f"day{k}"]) else HOLD
        cost = (cash + sum(q["cost"] for q in pos if q)) / slots
        cash -= cost
        pos[empty[0]] = {"code": row["Code"], "date": row["Date"], "buy": buy,
                         "free": buy + days, "ret": float(row[f"tp{k}"]),
                         "cost": cost, "hit": hit,
                         "n_break": int(row["n_break"]), "free_at_buy": len(empty)}
    for s, p in enumerate(pos):
        if p is not None:
            busy[s] += p["free"] - p["buy"]
            cash += p["cost"] * (1.0 + p["ret"] / 100.0)
            done.append({"Date": p["date"], "Code": p["code"], "slot": s,
                         "ret": p["ret"], "days": p["free"] - p["buy"],
                         "n_break": p["n_break"], "free_at_buy": p["free_at_buy"],
                         "hit": p["hit"]})

    t = pd.DataFrame(done)
    if not len(t):
        return {"taken": 0}
    years = (pd.Timestamp(sig["Date"].max()) - pd.Timestamp(sig["Date"].min())).days / 365.25
    span = max(1, idx[pd.Timestamp(sig["Date"].max())]
               - idx[pd.Timestamp(sig["Date"].min())] + HOLD)
    lo = t[t["n_break"] < 20]
    return {"taken": len(t), "skip_full": skip_full, "skip_gate": skip_gate,
            "per_year": len(t) / years, "mean": t["ret"].mean(),
            "win": (t["ret"] > 0).mean() * 100, "days": t["days"].mean(),
            "util": float(np.mean(busy)) / span * 100,
            "equity": cash, "cagr": (cash ** (1 / years) - 1) * 100,
            "worst": t["ret"].min(), "p10": t["ret"].quantile(0.1),
            "n_low": len(lo), "low_mean": lo["ret"].mean() if len(lo) else np.nan,
            "low_win": (lo["ret"] > 0).mean() * 100 if len(lo) else np.nan,
            "years": years, "trades": t}


HEAD = ("  {:<24}{:>6}{:>6}{:>9}{:>6}{:>7}{:>7}{:>9}{:>8}{:>8}{:>8}".format(
    "腕", "取引", "年間", "平均", "勝率", "保有日", "稼働率", "資産倍率",
    "年率", "下位10%", "最悪"))


def show(nm: str, s: dict) -> None:
    if not s.get("taken"):
        print(f"  {nm:<24} （該当なし）")
        return
    print(f"  {nm:<24}{s['taken']:>6}{s['per_year']:>6.1f}{s['mean']:>+8.2f}%"
          f"{s['win']:>5.0f}%{s['days']:>7.1f}{s['util']:>6.0f}%{s['equity']:>9.2f}"
          f"{s['cagr']:>+7.1f}%{s['p10']:>+7.1f}%{s['worst']:>+7.1f}%")


def main() -> int:
    d = prepare()
    d = attach(d, forward_paths())
    d = d[d["entry"].notna() & d["r"].notna()]
    for c in ("lgbm", "xgb", "cat"):
        d[f"p_{c}"] = d[f"s_{c}"].rank(pct=True)
    d["p_min"] = d[["p_lgbm", "p_xgb", "p_cat"]].min(axis=1)
    cal = calendar()
    sig = all_signals(d)
    log(f"基準を満たした銘柄 {len(sig):,}件（上位{TOP_K}件まで） / "
        f"うち発火20件以上 {(sig['n_break'] >= 20).sum():,}件")

    rows, keep = [], {}
    print(f"\n=== 1. 空き枠でしきい値を変える（枠{SLOTS}・上位{TOP_K}件・+20%利確）===")
    print(HEAD)
    for nm, lad in LADDERS.items():
        s = simulate(sig, cal, lad)
        keep[nm] = s
        rows.append({"arm": nm, **{k: v for k, v in s.items() if k != "trades"}})
        show(nm, s)

    print(f"\n=== 2. 緩めて取れた取引（発火20件未満の日）だけ取り出す ===")
    print(f"  {'腕':<24}{'件数':>6}{'平均':>9}{'勝率':>7}"
          f"{'（空き枠が無ければ 0%）':>24}")
    for nm, s in keep.items():
        if not s.get("n_low"):
            print(f"  {nm:<24}{0:>6}   （緩めていない）")
            continue
        print(f"  {nm:<24}{s['n_low']:>6}{s['low_mean']:>+8.2f}%{s['low_win']:>6.0f}%")

    print(f"\n=== 3. 買えなかった理由の内訳 ===")
    print(f"  {'腕':<24}{'取れた':>7}{'枠が満杯':>9}{'発火不足':>9}{'信号計':>8}")
    for nm, s in keep.items():
        tot = s["taken"] + s["skip_full"] + s["skip_gate"]
        print(f"  {nm:<24}{s['taken']:>7}{s['skip_full']:>9}{s['skip_gate']:>9}{tot:>8}")

    print(f"\n=== 4. 発火数の帯ごと（腕をまたいで、取れた取引の実績）===")
    t = keep["開放（発火を見ない）"]["trades"].copy()
    t["帯"] = pd.cut(t["n_break"], [0, 3, 7, 14, 19, 10_000],
                     labels=["1〜3件", "4〜7件", "8〜14件", "15〜19件", "20件以上"])
    print(f"  {'発火数':<10}{'件数':>6}{'平均':>9}{'勝率':>7}{'下位10%':>9}{'最悪':>9}")
    for b, g in t.groupby("帯", observed=True):
        print(f"  {str(b):<10}{len(g):>6}{g['ret'].mean():>+8.2f}%"
              f"{(g['ret']>0).mean()*100:>6.0f}%{g['ret'].quantile(0.1):>+8.1f}%"
              f"{g['ret'].min():>+8.1f}%")

    print(f"\n=== 5. 年ごとの取引数（0件の年が消えるか）===")
    print(f"  {'腕':<24}" + "".join(f"{y:>7}" for y in range(2021, 2027)))
    for nm, s in keep.items():
        c = pd.to_datetime(s["trades"]["Date"]).dt.year.value_counts()
        print(f"  {nm:<24}" + "".join(f"{int(c.get(y, 0)):>7}" for y in range(2021, 2027)))

    print(f"\n=== 6. しきい値を1件ずつ動かす（空き枠を見ない・枠{SLOTS}）===")
    print(HEAD)
    sweep = []
    for m in (1, 4, 6, 8, 10, 12, 15, 18, 20, 26):
        s = simulate(sig, cal, {1: m, 2: m, 3: m})
        sweep.append({"min_break": m, **{k: v for k, v in s.items() if k != "trades"}})
        show(f"発火{m}件以上", s)
    pd.DataFrame(sweep).to_csv(os.path.join(OOF_DIR, "e35_sweep.csv"), index=False)

    print(f"\n=== 7. 枠の数を変えても「8 > 20」は保たれるか（+20%利確）===")
    print(f"  {'':<10}" + "".join(f"{'枠'+str(n):>20}" for n in (1, 2, 3, 4, 5)))
    for m in (8, 15, 20):
        line = f"  発火{m}件"
        for n in (1, 2, 3, 4, 5):
            s = simulate(sig, cal, {i: m for i in range(1, n + 1)}, slots=n)
            line += f"{s['equity']:>10.2f}倍{s['cagr']:>+8.1f}%"
        print(line)

    print(f"\n=== 8. 出口を変えても保たれるか（枠{SLOTS}）===")
    print(f"  {'':<10}" + "".join(f"{'+'+str(t)+'%':>20}" for t in (10, 15, 20)))
    for m in (8, 15, 20):
        line = f"  発火{m}件"
        for tgt in (10.0, 15.0, 20.0):
            s = simulate(sig, cal, {i: m for i in range(1, SLOTS + 1)}, target=tgt)
            line += f"{s['equity']:>10.2f}倍{s['cagr']:>+8.1f}%"
        print(line)

    print(f"\n=== 9. 年ごとの成績（発火8 と 発火20、枠{SLOTS}・+20%利確）===")
    print(f"  {'年':<6}" + f"{'発火8: 取引':>12}{'平均':>9}{'年の倍率':>10}"
          + f"{'発火20: 取引':>13}{'平均':>9}{'年の倍率':>10}")
    ta = simulate(sig, cal, {1: 8, 2: 8, 3: 8})["trades"]
    tb = simulate(sig, cal, {1: 20, 2: 20, 3: 20})["trades"]
    for t in (ta, tb):
        t["year"] = pd.to_datetime(t["Date"]).dt.year
    for y in range(2021, 2027):
        out = f"  {y:<6}"
        for t in (ta, tb):
            g = t[t["year"] == y]
            if not len(g):
                out += f"{0:>12}{'—':>9}{'—':>10}" if t is ta else f"{0:>13}{'—':>9}{'—':>10}"
                continue
            w = float(np.prod(1 + g["ret"] / 100) ** (1 / max(1, g["slot"].nunique())))
            wd = 12 if t is ta else 13
            out += f"{len(g):>{wd}}{g['ret'].mean():>+8.2f}%{w:>10.3f}"
        print(out)

    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e35_slack.csv"), index=False)
    for nm, s in keep.items():
        if nm.startswith("運用者案"):
            s["trades"].to_csv(os.path.join(OOF_DIR, "e35_trades.csv"), index=False)
    log(f"記録: {OOF_DIR}/e35_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
