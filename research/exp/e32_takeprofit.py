#!/usr/bin/env python3
"""
実験32: 買値から +10% / +5% に届く銘柄はどれくらいあるか（利確の出口）。

運用者の想定
  - 高値更新の**翌営業日の寄り**で買う（既存の物差し ret_o1_20 と同じ入り口）
  - 20営業日以内に買値から **+10%** 上げたら売る（届かなければ +5% も見る）
  - 届かなければ持ち切り（= ret_o1_20。20営業日後の5日平均終値）

測り方
  entry  = AdjO[t+1]（分割調整後の翌営業日始値）
  到達   = t+1 〜 t+20 の **AdjH の最大**が entry × 1.10（または 1.05）以上
           指値を置いていれば約定する水準。買った当日（t+1）の高値も数える
  利確の収益 = 到達したら +10%（または +5%）、しなければ ret_o1_20
           ギャップで飛び越えた場合は実際にはもっと取れるので、**保守側**の見積もり
  日数   = 最初に到達した営業日（t+1 を1日目とする）

母集団・選定・運用者の戦略（発火の多い日の上位）それぞれで出す。
スコアは実験27 の腕 B1。結果は research/_data/oof/e32_*.csv。
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

HOLD = 20
TARGETS = (5.0, 10.0, 15.0, 20.0)


def forward_paths(hold: int = HOLD) -> pd.DataFrame:
    """
    (Code, Date) ごとに entry と、t+1〜t+hold の値動きを付ける。
      entry     翌営業日の始値
      max_high  その期間の高値の最大
      hit{p}_day  buy × (1+p) に最初に届いた営業日（1 = 買った当日）
    """
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    b = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "AdjO", "AdjH", "AdjC"])
                   for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Date", "Code"])
    o = b.pivot(index="Date", columns="Code", values="AdjO").astype("float32")
    h = b.pivot(index="Date", columns="Code", values="AdjH").astype("float32")
    entry = o.shift(-1)

    # t+1 〜 t+hold の高値の最大と、しきい値に最初に届いた日
    out = {"entry": entry}
    cum = None
    for k in range(1, hold + 1):
        hk = h.shift(-k)
        cum = hk if cum is None else np.maximum(cum, hk)
        out[f"max_to_{k}"] = cum.copy() if k in (1, 3, 5, 10, hold) else None
    out["max_high"] = cum
    log("  値動きの表を作った")
    res = []
    for name, df in (("entry", entry), ("max_high", cum)):
        m = df.stack(future_stack=True).rename(name)
        res.append(m)
    frame = pd.concat(res, axis=1).reset_index()
    frame.columns = ["Date", "Code", "entry", "max_high"]

    # 最初に届いた日（しきい値ごと）
    first = {p: pd.DataFrame(np.nan, index=h.index, columns=h.columns, dtype="float32")
             for p in TARGETS}
    run = None
    for k in range(1, hold + 1):
        hk = h.shift(-k)
        run = hk if run is None else np.maximum(run, hk)
        for p in TARGETS:
            tgt = entry * (1.0 + p / 100.0)
            hit = (run >= tgt) & first[p].isna()
            first[p] = first[p].mask(hit, float(k))
    for p in TARGETS:
        frame[f"day{int(p)}"] = first[p].stack(future_stack=True).to_numpy()
    log("  到達日を作った")
    return frame.dropna(subset=["entry"])


def attach(df: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(paths, on=["Code", "Date"], how="left")
    d["max_gain"] = (d["max_high"] / d["entry"] - 1.0) * 100
    for p in TARGETS:
        k = int(p)
        d[f"hit{k}"] = d[f"day{k}"].notna()
        # 利確したときの収益。届かなければ持ち切り（ret_o1_20）
        d[f"tp{k}"] = np.where(d[f"hit{k}"], p, d["r"])
    return d


def stats(g: pd.DataFrame) -> dict:
    if len(g) < 10:
        return {}
    r = g["r"]
    out = {"n": len(g), "hold": r.mean(), "hold_med": r.median(),
           "hold_win": (r > 0).mean() * 100, "hold_sd": r.std(ddof=1),
           "hold_p10": r.quantile(0.1), "hold_day": HOLD,
           "hold_per_day": r.mean() / HOLD,
           "max_gain": g["max_gain"].mean(), "max_gain_med": g["max_gain"].median()}
    for p in TARGETS:
        k = int(p)
        hit = g[f"hit{k}"]
        tp = g[f"tp{k}"]
        # 保有日数: 到達したらその日、しなければ 20営業日
        days = np.where(hit, g[f"day{k}"].fillna(HOLD), HOLD)
        out[f"hit{k}"] = hit.mean() * 100
        out[f"day{k}"] = g.loc[hit, f"day{k}"].median()
        out[f"tp{k}"] = tp.mean()
        out[f"tp{k}_win"] = (tp > 0).mean() * 100
        out[f"tp{k}_sd"] = tp.std(ddof=1)
        out[f"tp{k}_p10"] = tp.quantile(0.1)
        out[f"tp{k}_day"] = days.mean()
        out[f"tp{k}_per_day"] = tp.mean() / days.mean()
        out[f"miss{k}"] = g.loc[~hit, "r"].mean()
    return out


HEAD = (f"  {'条件':<26}{'件数':>6}{'+5%到達':>8}{'日数':>5}{'+10%到達':>9}{'日数':>5}"
        f"{'+15%':>7}{'+20%':>7}{'最大上昇':>8}{'持ち切り':>9}{'勝率':>6}"
        f"{'+5%利確':>9}{'+10%利確':>9}{'勝率':>6}{'未到達':>8}")


def line(name: str, s: dict) -> str:
    if not s:
        return f"  {name:<26} （件数不足）"
    return (f"  {name:<26}{s['n']:>6}{s['hit5']:>7.1f}%{s['day5']:>5.0f}{s['hit10']:>8.1f}%"
            f"{s['day10']:>5.0f}{s['hit15']:>6.1f}%{s['hit20']:>6.1f}%"
            f"{s['max_gain']:>+7.1f}%{s['hold']:>+8.2f}%{s['hold_win']:>5.0f}%"
            f"{s['tp5']:>+8.2f}%{s['tp10']:>+8.2f}%{s['tp10_win']:>5.0f}%{s['miss10']:>+7.2f}%")


def perday(name: str, s: dict) -> None:
    if not s:
        return
    print(f"  {name:<26}{s['n']:>6}"
          + f"{s['hold']:>+8.2f}%{HOLD:>6}日{s['hold_per_day']:>8.3f}%"
          + "".join(f"{s[f'tp{int(p)}']:>+8.2f}%{s[f'tp{int(p)}_day']:>6.1f}日"
                    f"{s[f'tp{int(p)}_per_day']:>8.3f}%" for p in TARGETS))


def main() -> int:
    d = prepare()
    log(f"対象 {len(d):,}件 / {d['Date'].min().date()}〜{d['Date'].max().date()}")
    paths = forward_paths()
    d = attach(d, paths)
    cov = d["entry"].notna().mean()
    log(f"値動きが付いた行 {cov*100:.1f}%（entry が取れない = 期間の末尾）")
    d = d[d["entry"].notna() & d["r"].notna()]

    for c in ("lgbm", "xgb", "cat"):
        d[f"p_{c}"] = d[f"s_{c}"].rank(pct=True)
    d["p_min"] = d[["p_lgbm", "p_xgb", "p_cat"]].min(axis=1)

    print("\n" + HEAD)
    print(line("母集団（全ブレイク）", stats(d)))
    print(line("3モデル 90以上（全日・全件）", stats(d[d[RULE]])))
    for m in (10, 20, 26):
        print(line(f"  発火{m}件以上・全件", stats(d[d[RULE] & (d["n_break"] >= m)])))
    for m, k in ((20, 1), (20, 2), (26, 1), (26, 2)):
        sub = d[d[RULE] & (d["n_break"] >= m)].sort_values(["Date", "p_min"], ascending=[True, False])
        sub = sub.assign(rank=sub.groupby("Date").cumcount() + 1)
        print(line(f"  発火{m}件以上・上位{k}件", stats(sub[sub["rank"] <= k])))

    print("\n=== 1日あたりの収益率（資金が1〜2銘柄しか無いときはこれが効く）===")
    print(f"  {'条件':<26}{'件数':>6}{'持ち切り':>9}{'日数':>7}{'1日':>8}"
          + "".join(f"{'+'+str(int(p))+'%利確':>9}{'日数':>7}{'1日':>8}" for p in TARGETS))
    for name, sub in (("母集団", d), ("3モデル 90以上（全日）", d[d[RULE]]),
                      ("  発火20件以上・全件", d[d[RULE] & (d["n_break"] >= 20)])):
        perday(name, stats(sub))
    for m, k in ((20, 1), (20, 2)):
        sub = d[d[RULE] & (d["n_break"] >= m)].sort_values(["Date", "p_min"], ascending=[True, False])
        sub = sub.assign(rank=sub.groupby("Date").cumcount() + 1)
        perday(f"  発火{m}件以上・上位{k}件", stats(sub[sub["rank"] <= k]))

    print("\n=== ばらつき（3モデル 90以上・発火20件以上）===")
    s0 = stats(d[d[RULE] & (d["n_break"] >= 20)])
    print(f"  {'出口':<16}{'平均':>9}{'標準偏差':>10}{'下位10%':>9}{'勝率':>7}")
    print(f"  {'持ち切り20日':<16}{s0['hold']:>+8.2f}%{s0['hold_sd']:>9.2f}{s0['hold_p10']:>+8.2f}%"
          f"{s0['hold_win']:>6.0f}%")
    for p in TARGETS:
        k = int(p)
        print(f"  {'+'+str(k)+'%で利確':<16}{s0[f'tp{k}']:>+8.2f}%{s0[f'tp{k}_sd']:>9.2f}"
              f"{s0[f'tp{k}_p10']:>+8.2f}%{s0[f'tp{k}_win']:>6.0f}%")

    print("\n=== 到達までの日数の分布（3モデル 90以上・発火20件以上）===")
    g = d[d[RULE] & (d["n_break"] >= 20)]
    for p in TARGETS:
        k = int(p)
        dd = g.loc[g[f"hit{k}"], f"day{k}"]
        if not len(dd):
            continue
        print(f"  +{k}%: 到達 {len(dd)}件 / {len(g)}件（{len(dd)/len(g)*100:.1f}%）"
              f" 中央値 {dd.median():.0f}営業日 / 25-75% {dd.quantile(.25):.0f}〜{dd.quantile(.75):.0f} / "
              f"5営業日以内 {(dd <= 5).mean()*100:.0f}% / 10営業日以内 {(dd <= 10).mean()*100:.0f}%")

    print("\n=== 到達した銘柄を持ち切っていたら（早売りの損得）===")
    print(f"  {'条件':<26}{'+10%到達':>9}{'利確 +10%':>10}{'持ち切りなら':>12}{'差':>8}")
    for name, sub in (("3モデル 90以上（全日）", d[d[RULE]]),
                      ("  発火20件以上", d[d[RULE] & (d["n_break"] >= 20)]),
                      ("母集団", d)):
        hit = sub[sub["hit10"]]
        if len(hit) < 10:
            continue
        print(f"  {name:<26}{len(hit):>8}件{10.0:>+9.2f}%{hit['r'].mean():>+11.2f}%"
              f"{10.0-hit['r'].mean():>+7.2f}pt")

    keep = ["Code", "Date", "n_break", RULE, "r", "entry", "max_gain",
            "hit5", "day5", "tp5", "hit10", "day10", "tp10"]
    d[keep].to_csv(os.path.join(OOF_DIR, "e32_takeprofit.csv"), index=False)
    log(f"記録: {OOF_DIR}/e32_takeprofit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
