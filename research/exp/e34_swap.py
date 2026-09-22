#!/usr/bin/env python3
"""
実験34: 同時保有 3枠 と「乗り換え」規則。

運用者の指定（2026-09-22）
  - 同時に持つのは **最大3銘柄**
  - 保有中により良い候補が発火したら、**そのときマイナスでなければ**
    利確して乗り換える

実験33 は「枠が埋まっていたら見送る（先着順）」だった。先着順だと、
発火が固まる日（実測で 77% は5営業日以内に次が来る）に良い候補を
取りこぼす。乗り換えはその取りこぼしを拾いにいく規則。

乗り換えの約定
  発火した日 d の翌営業日 buy = d+1 の **寄り**で、
    1. 保有中の銘柄の寄り値を見る（寄り値は板が開いた時点で見える）
    2. 買値以上（含み損でない）で、かつ 新規候補より p_min が低い枠があれば
       その枠を寄りで売り、同じ寄りで新規を買う
    3. どれも含み損なら見送る
  条件付き注文ではなく、寄り付きに値を見てから出す成行。先の値は使っていない。

比較する腕
  先着順   枠が埋まっていたら見送る（実験33 と同じ）
  乗り換え 上の規則で入れ替える
  しきい値 含み益が 0% / +3% / +5% 以上のときだけ乗り換える、も出す

出口は実験32 と同じ（+20%/+10% に届けば利確、届かなければ20営業日）。
結果は research/_data/oof/e34_*.csv。
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
from e31_strategy import prepare, RULE  # noqa: E402
from e32_takeprofit import HOLD, attach, forward_paths  # noqa: E402
from e33_portfolio import calendar, signals  # noqa: E402

SLOTS = 3            # 運用者の指定（同時保有の上限）
MIN_BREAK = 20       # 発火数のしきい値（実験31 で 15〜30 のどこでも同じ結論）
TOP_K = 2            # その日の上位何件まで見るか


def open_matrix(cal: pd.Series) -> pd.DataFrame:
    """営業日の番号 × 銘柄コードの、分割調整後の寄り値。乗り換えの約定に使う。"""
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    b = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "AdjO"])
                   for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b[b["Date"].isin(set(cal))]
    o = b.pivot_table(index="Date", columns="Code", values="AdjO", aggfunc="last")
    o = o.reindex(cal).astype("float32")
    o.index = range(len(cal))
    log(f"  寄り値の表 {o.shape[0]}営業日 × {o.shape[1]:,}銘柄")
    return o


def simulate(sig: pd.DataFrame, cal: pd.Series, opens: pd.DataFrame, *,
             slots: int = SLOTS, target: float = 20.0,
             swap: bool = False, need_gain: float = 0.0,
             pick: str = "pmin", require_worse: bool = True) -> dict:
    """
    枠を埋めながら進む。swap=True なら、埋まっているときに入れ替えを試す。

    枠ごとに複利（1枠 = 資金の 1/slots）。清算は枠が空く日にまとめて行う。
    """
    k = int(target)
    idx = {d: i for i, d in enumerate(cal)}
    pos: list[dict | None] = [None] * slots
    wealth = [1.0] * slots
    busy = [0] * slots
    cash = 1.0
    done: list[dict] = []
    skipped = swaps = 0

    def settle(s: int, day: int, ret: float, why: str) -> None:
        p = pos[s]
        wealth[s] *= (1.0 + ret / 100.0)
        busy[s] += day - p["buy"]
        nonlocal cash
        cash += p["cost"] * (1.0 + ret / 100.0)
        done.append({"Date": p["date"], "Code": p["code"], "slot": s,
                     "ret": ret, "days": day - p["buy"], "exit": why,
                     "pmin": p["pmin"], "kept": p["ret"], "kept_days": p["free"] - p["buy"]})
        pos[s] = None

    def enter(s: int, row: pd.Series, buy: int) -> None:
        nonlocal cash
        hit = bool(row[f"hit{k}"])
        days = int(row[f"day{k}"]) if hit and pd.notna(row[f"day{k}"]) else HOLD
        # 実資金の複利: そのときの資産（現金＋建玉の簿価）の 1/slots を入れる
        cost = (cash + sum(q["cost"] for q in pos if q)) / slots
        cash -= cost
        pos[s] = {"code": row["Code"], "date": row["Date"], "buy": buy,
                  "free": buy + days, "ret": float(row[f"tp{k}"]),
                  "entry": float(row["entry"]), "pmin": float(row["p_min"]),
                  "hit": hit, "cost": cost}

    for _, row in sig.iterrows():
        di = idx.get(pd.Timestamp(row["Date"]))
        if di is None or not np.isfinite(row[f"tp{k}"]):
            continue
        buy = di + 1
        if buy >= len(cal):
            continue
        # 1) 期限・利確で空く枠を清算する
        for s, p in enumerate(pos):
            if p is not None and p["free"] <= buy:
                settle(s, p["free"], p["ret"], "利確" if p["hit"] else "満期")
        # 2) 空きがあれば入る
        empty = [s for s, p in enumerate(pos) if p is None]
        if empty:
            enter(empty[0], row, buy)
            continue
        if not swap:
            skipped += 1
            continue
        # 3) 乗り換え。寄り値で含み損でなく、新規より格下の枠を探す
        cand = []
        for s, p in enumerate(pos):
            if p["buy"] >= buy:            # 同じ日に買ったものは入れ替えない
                continue
            px = opens.iat[buy, opens.columns.get_loc(p["code"])] \
                if p["code"] in opens.columns else np.nan
            if not np.isfinite(px):
                continue
            gain = (float(px) / p["entry"] - 1.0) * 100.0
            if gain < need_gain:
                continue
            if require_worse and p["pmin"] >= float(row["p_min"]):
                continue
            key = gain if pick == "gain" else p["pmin"]
            cand.append((key, gain, s))
        if not cand:
            skipped += 1
            continue
        cand.sort()                         # 一番格下の枠から入れ替える
        _, gain, s = cand[0]
        settle(s, buy, gain, "乗り換え")
        swaps += 1
        enter(s, row, buy)

    # 4) 最後まで持っていた枠を清算する
    for s, p in enumerate(pos):
        if p is not None:
            settle(s, p["free"], p["ret"], "利確" if p["hit"] else "満期")

    t = pd.DataFrame(done)
    if not len(t):
        return {"taken": 0}
    years = (pd.Timestamp(sig["Date"].max()) - pd.Timestamp(sig["Date"].min())).days / 365.25
    total = float(np.mean(wealth))
    # 稼働率の分母は「最初に買ってから最後に売るまで」の営業日数
    span = max(1, idx[pd.Timestamp(sig["Date"].max())]
               - idx[pd.Timestamp(sig["Date"].min())] + HOLD)
    sw = t[t["exit"] == "乗り換え"]
    return {"slots": slots, "target": k, "swap": swap, "need_gain": need_gain,
            "util": np.mean(busy) / span * 100, "equity": cash,
            "eq_cagr": (cash ** (1 / years) - 1) * 100,
            "swap_got": sw["ret"].mean() if len(sw) else np.nan,
            "swap_kept": sw["kept"].mean() if len(sw) else np.nan,
            "taken": len(t), "skipped": skipped, "swaps": swaps,
            "rate": len(t) / (len(t) + skipped) * 100,
            "per_year": len(t) / years, "mean": t["ret"].mean(),
            "win": (t["ret"] > 0).mean() * 100,
            "days": t["days"].mean(), "wealth": total,
            "cagr": (total ** (1 / years) - 1) * 100,
            "worst": t["ret"].min(), "years": years, "trades": t}


HEAD = ("  {:<26}{:>6}{:>6}{:>6}{:>9}{:>6}{:>7}{:>7}{:>9}{:>8}{:>8}".format(
    "腕", "取引", "乗換", "年間", "平均", "勝率", "保有日", "稼働率", "資産倍率", "年率", "最悪"))


def show(nm: str, s: dict) -> None:
    """稼働率＝枠が埋まっていた営業日の割合。資産倍率は実資金（1/枠ずつ）。"""
    if not s.get("taken"):
        print(f"  {nm:<26} （該当なし）")
        return
    print(f"  {nm:<26}{s['taken']:>6}{s['swaps']:>6}{s['per_year']:>6.1f}"
          f"{s['mean']:>+8.2f}%{s['win']:>5.0f}%{s['days']:>7.1f}{s['util']:>6.0f}%"
          f"{s['equity']:>9.2f}{s['eq_cagr']:>+7.1f}%{s['worst']:>+8.1f}%")


def main() -> int:
    d = prepare()
    d = attach(d, forward_paths())
    d = d[d["entry"].notna() & d["r"].notna()]
    for c in ("lgbm", "xgb", "cat"):
        d[f"p_{c}"] = d[f"s_{c}"].rank(pct=True)
    d["p_min"] = d[["p_lgbm", "p_xgb", "p_cat"]].min(axis=1)
    cal = calendar()
    opens = open_matrix(cal)
    log(f"対象 {len(d):,}件 / {d['Date'].min().date()}〜{d['Date'].max().date()}")

    rows = []

    print(f"\n=== 1. 枠を変える（発火{MIN_BREAK}件以上・上位{TOP_K}件・+20%利確）===")
    print(HEAD)
    sig = signals(d, MIN_BREAK, TOP_K)
    for slots in (1, 2, 3, 4, 5):
        s = simulate(sig, cal, opens, slots=slots, target=20.0, swap=False)
        rows.append({"arm": "先着順", **{k: v for k, v in s.items() if k != "trades"}})
        show(f"先着順・枠{slots}", s)

    print(f"\n=== 2. 枠{SLOTS}で乗り換えを入れる（+20%利確）===")
    print(HEAD)
    base = simulate(sig, cal, opens, slots=SLOTS, target=20.0, swap=False)
    rows.append({"arm": "先着順", **{k: v for k, v in base.items() if k != "trades"}})
    show(f"先着順・枠{SLOTS}", base)
    variants = [
        ("乗り換え・含み益0%以上", dict(need_gain=0.0)),
        ("乗り換え・含み益3%以上", dict(need_gain=3.0)),
        ("乗り換え・含み益5%以上", dict(need_gain=5.0)),
        ("乗り換え・含み益の薄い枠から", dict(need_gain=0.0, pick="gain")),
        ("乗り換え・格の条件なし", dict(need_gain=0.0, require_worse=False)),
    ]
    for nm, kw in variants:
        v = simulate(sig, cal, opens, slots=SLOTS, target=20.0, swap=True, **kw)
        rows.append({"arm": nm, **{k: q for k, q in v.items() if k != "trades"}})
        show(nm, v)

    print(f"\n  --- 乗り換えで**捨てた**ぶん（持ち切っていたら幾らだったか）---")
    print(f"  {'腕':<26}{'乗換':>6}{'実現':>9}{'持ち切り':>10}{'差':>9}{'損した回':>9}")
    for nm, kw in variants:
        v = simulate(sig, cal, opens, slots=SLOTS, target=20.0, swap=True, **kw)
        sw = v["trades"]
        sw = sw[sw["exit"] == "乗り換え"]
        if not len(sw):
            continue
        gap = sw["ret"] - sw["kept"]
        print(f"  {nm:<26}{len(sw):>6}{sw['ret'].mean():>+8.2f}%{sw['kept'].mean():>+9.2f}%"
              f"{gap.mean():>+8.2f}%{(gap < 0).mean()*100:>8.0f}%")

    print(f"\n=== 3. 上位何件まで見るか（枠{SLOTS}・+20%利確・乗り換え0%）===")
    print(HEAD)
    for tk in (1, 2, 3):
        sg = signals(d, MIN_BREAK, tk)
        for sw in (False, True):
            s = simulate(sg, cal, opens, slots=SLOTS, target=20.0, swap=sw)
            rows.append({"arm": ("乗り換え" if sw else "先着順") + f"/上位{tk}",
                         "top_k": tk, **{k: v for k, v in s.items() if k != "trades"}})
            show(f"上位{tk}件・{'乗り換え' if sw else '先着順'}", s)

    print(f"\n=== 4. 出口を変える（枠{SLOTS}・発火{MIN_BREAK}件以上・上位{TOP_K}件）===")
    print(HEAD)
    for tgt in (10.0, 15.0, 20.0):
        for sw in (False, True):
            s = simulate(sig, cal, opens, slots=SLOTS, target=tgt, swap=sw)
            rows.append({"arm": ("乗り換え" if sw else "先着順") + f"/+{int(tgt)}%",
                         **{k: v for k, v in s.items() if k != "trades"}})
            show(f"+{int(tgt)}%利確・{'乗り換え' if sw else '先着順'}", s)

    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e34_swap.csv"), index=False)

    # 本命の設定の明細
    best = simulate(sig, cal, opens, slots=SLOTS, target=20.0, swap=True)
    t = best["trades"].copy()
    t["year"] = pd.to_datetime(t["Date"]).dt.year
    print(f"\n=== 5. 本命（枠{SLOTS}・乗り換えあり・+20%利確）の内訳 ===")
    print(f"  {'出口':<10}{'件数':>6}{'平均':>9}{'勝率':>7}{'保有日':>8}")
    for why, g in t.groupby("exit"):
        print(f"  {why:<10}{len(g):>6}{g['ret'].mean():>+8.2f}%"
              f"{(g['ret']>0).mean()*100:>6.0f}%{g['days'].mean():>8.1f}")
    print(f"\n  {'年':<6}{'取引':>6}{'平均':>9}{'勝率':>7}{'乗換':>6}")
    for y, g in t.groupby("year"):
        print(f"  {y:<6}{len(g):>6}{g['ret'].mean():>+8.2f}%"
              f"{(g['ret']>0).mean()*100:>6.0f}%{(g['exit']=='乗り換え').sum():>6}")
    sw = t[t["exit"] == "乗り換え"]
    print(f"\n  乗り換えた {len(sw)}件: 実現 {sw['ret'].mean():+.2f}% / "
          f"そのまま持っていたら {sw['kept'].mean():+.2f}% "
          f"（差 {(sw['ret']-sw['kept']).mean():+.2f}pt、"
          f"{(sw['ret']<sw['kept']).mean()*100:.0f}% は切らないほうが良かった）")
    print(f"\n  期間 {best['years']:.1f}年 / 取引 {best['taken']}件"
          f"（見送り {best['skipped']}件 / 乗り換え {best['swaps']}件）"
          f" / 資産 {best['wealth']:.2f}倍 / 年率 {best['cagr']:+.1f}%")
    print("  ※ 手数料・税・スリッページは含まない。1枠あたり資金の1/3を投じる前提")
    t.to_csv(os.path.join(OOF_DIR, "e34_trades.csv"), index=False)
    log(f"記録: {OOF_DIR}/e34_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
