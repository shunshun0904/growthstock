#!/usr/bin/env python3
"""
実験30: 候補がまとまって出る日と、ほとんど出ない日で、精度と利益は違うか。

問い（運用者の仮説）
------------------
「地合いが弱くて発火が少ない日のブレイクは、それでも抜けてきたのだから
ポテンシャルが大きいのではないか。モデルはそれを捉えているか」

3つの層で測る
-----------
  母集団   その日にブレイクした全銘柄。仮説そのものの検証
           （発火の少ない日のブレイクは、本当に良いのか）
  選定     基準（lgbm 単体 95以上 / 3モデル 90以上）を満たした銘柄
  最良1件  その日のスコア最上位を1銘柄だけ買う（本番の順位と同じ LightGBM で並べる）
           ・基準なし: 毎営業日1件
           ・基準あり: 基準を満たす日だけ1件

日で切る軸
---------
  発火数   その日の母集団の件数（1〜3 / 4〜7 / 8〜14 / 15〜25 / 26件〜）
  地合い   TOPIX の200日線からの乖離（5分位）。発火数と地合いは相関するので
           両方で切って、どちらが効いているのかを見る

同じ日の銘柄は地合いを共有するので、**まず日ごとに平均してから**日をまたいで
平均する（1日1観測）。標準誤差も日の数で出す。

スコアは実験27 の腕 B1（本番と同じ153列・本番のパラメータ・種3つの平均）。
結果は research/_data/oof/e30_*.csv。
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
from e25_auc_noise import average  # noqa: E402
from e27_timing_multi import OOF_DIR, log  # noqa: E402

SEEDS3 = (42, 7, 123)
OUTCOME = "ret_o1_20"
#: 発火数の帯。中央値10件・上位10%が26件以上（docs/MODEL_SELECTION_EDA.md）
BREAK_BINS = [0, 3, 7, 14, 25, 10_000]
BREAK_JA = ["1〜3件", "4〜7件", "8〜14件", "15〜25件", "26件〜"]


def load_scores() -> pd.DataFrame:
    base = None
    for a in OR.BOOST:
        files = [os.path.join(OOF_DIR, f"e27_{a}_B1_s{s}.parquet") for s in SEEDS3]
        if not all(map(os.path.exists, files)):
            raise SystemExit(f"e27_{a}_B1_s*.parquet がありません（先に実験27）")
        o = average([pd.read_parquet(f) for f in files])
        cols = ["Code", "Date", "fold", "label", OUTCOME, "ret_o1_40", "score"]
        o = o[cols].rename(columns={"score": f"s_{a}"})
        base = o if base is None else base.merge(
            o[["Code", "Date", f"s_{a}"]], on=["Code", "Date"], how="inner")
    base["Date"] = pd.to_datetime(base["Date"])
    return base


def add_rules(df: pd.DataFrame) -> pd.DataFrame:
    """運用の2基準。しきい値はそれより前の窓の分布（運用そのもの）。"""
    for rname, models, pct in OR.RULES:
        flag = np.zeros(len(df), dtype=bool)
        for f in sorted(df["fold"].unique()):
            prev = df[df["fold"] < f]
            if len(prev) < 500:
                continue
            m = (df["fold"] == f).to_numpy()
            ok = m.copy()
            for a in models:
                ok &= (df[f"s_{a}"] > np.percentile(prev[f"s_{a}"], pct)).to_numpy()
            flag |= ok
        df[rname] = flag
    df["usable"] = df["fold"] > df["fold"].min()      # 窓1は参照分布が無い
    return df


def day_table(df: pd.DataFrame) -> pd.DataFrame:
    """1日1行に畳む。"""
    d = df[df["usable"]].copy()
    d["r"] = pd.to_numeric(d[OUTCOME], errors="coerce")
    rows = []
    for date, g in d.groupby("Date"):
        g = g.sort_values("s_lgbm", ascending=False)
        rec = {"Date": date, "n_break": len(g),
               "uni_ret": g["r"].mean() * 100, "uni_label": g["label"].mean(),
               "top1_ret": g["r"].iloc[0] * 100, "top1_label": float(g["label"].iloc[0]),
               "top1_score_rank": 1.0}
        for rname, _, _ in OR.RULES:
            sel = g[g[rname]]
            rec[f"n_{rname}"] = len(sel)
            rec[f"ret_{rname}"] = sel["r"].mean() * 100 if len(sel) else np.nan
            rec[f"label_{rname}"] = sel["label"].mean() if len(sel) else np.nan
            rec[f"top1_ret_{rname}"] = sel["r"].iloc[0] * 100 if len(sel) else np.nan
            rec[f"top1_label_{rname}"] = float(sel["label"].iloc[0]) if len(sel) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def mean_se(v: pd.Series):
    v = v.dropna()
    if not len(v):
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan, len(v)


def show(days: pd.DataFrame, by: str, order: list, title: str) -> pd.DataFrame:
    print(f"\n=== {title} ===")
    print(f"  {'帯':<12}{'日数':>5}{'発火/日':>8}"
          f"{'母集団 ret20':>13}{'SE':>6}{'母集団 正例率':>13}"
          f"{'最良1件 ret20':>14}{'SE':>6}{'正例率':>8}{'超過':>8}")
    rows = []
    for k in order:
        g = days[days[by] == k]
        if not len(g):
            continue
        um, us, _ = mean_se(g["uni_ret"])
        tm, ts, n1 = mean_se(g["top1_ret"])
        rec = {"bin": k, "days": len(g), "breaks": g["n_break"].mean(),
               "uni_ret": um, "uni_se": us, "uni_label": g["uni_label"].mean(),
               "top1_ret": tm, "top1_se": ts, "top1_label": g["top1_label"].mean(),
               "top1_excess": tm - um}
        rows.append(rec)
        print(f"  {str(k):<12}{len(g):>5}{g['n_break'].mean():>8.1f}"
              f"{um:>+12.2f}%{us:>6.2f}{g['uni_label'].mean()*100:>12.1f}%"
              f"{tm:>+13.2f}%{ts:>6.2f}{g['top1_label'].mean()*100:>7.1f}%{tm-um:>+7.2f}pt")
    return pd.DataFrame(rows)


def show_rule(days: pd.DataFrame, by: str, order: list, rname: str) -> pd.DataFrame:
    print(f"\n  --- {rname} ---")
    print(f"  {'帯':<12}{'出た日':>7}{'出た率':>7}{'選定/日':>8}"
          f"{'選定 ret20':>12}{'SE':>6}{'正例率':>8}"
          f"{'最良1件 ret20':>14}{'SE':>6}{'正例率':>8}{'その日の母集団比':>10}")
    rows = []
    for k in order:
        g = days[days[by] == k]
        hit = g[g[f"n_{rname}"] > 0]
        if not len(hit):
            continue
        sm, ss, _ = mean_se(hit[f"ret_{rname}"])
        tm, ts, _ = mean_se(hit[f"top1_ret_{rname}"])
        um, _, _ = mean_se(hit["uni_ret"])
        rows.append({"bin": k, "days": len(g), "hit_days": len(hit),
                     "hit_rate": len(hit) / len(g), "sel_per_day": hit[f"n_{rname}"].mean(),
                     "sel_ret": sm, "sel_se": ss, "sel_label": hit[f"label_{rname}"].mean(),
                     "top1_ret": tm, "top1_se": ts, "top1_label": hit[f"top1_label_{rname}"].mean(),
                     "top1_excess": tm - um})
        print(f"  {str(k):<12}{len(hit):>7}{len(hit)/len(g)*100:>6.0f}%"
              f"{hit[f'n_{rname}'].mean():>8.1f}{sm:>+11.2f}%{ss:>6.2f}"
              f"{hit[f'label_{rname}'].mean()*100:>7.1f}%"
              f"{tm:>+13.2f}%{ts:>6.2f}{hit[f'top1_label_{rname}'].mean()*100:>7.1f}%"
              f"{tm-um:>+9.2f}pt")
    return pd.DataFrame(rows)


def supplement(df: pd.DataFrame) -> None:
    """
    銘柄単位の補足。
      1. モデルは発火の少ない日の銘柄をどう採点しているか
      2. リターンの分布（ポテンシャル = 上側の裾、下側の裾）
      3. 最大上昇（ref_rise: 先60営業日の終値の最大）と持ち切り（ref_end: 60営業日後）
    """
    d = df[df["usable"]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    fr = lab.frame()
    fr["Date"] = pd.to_datetime(fr["Date"])
    d = d.merge(fr[["Code", "Date", "ref_rise", "ref_end"]], on=["Code", "Date"], how="left")
    d = d.merge(d.groupby("Date").size().rename("n_break"), on="Date")
    d["bin"] = pd.cut(d["n_break"], BREAK_BINS, labels=BREAK_JA)
    d["r"] = pd.to_numeric(d[OUTCOME], errors="coerce") * 100
    d["rise"] = pd.to_numeric(d["ref_rise"], errors="coerce") * 100
    d["end"] = pd.to_numeric(d["ref_end"], errors="coerce") * 100
    d["s_pct"] = d["s_lgbm"].rank(pct=True) * 100

    print("\n=== モデルは発火の少ない日の銘柄をどう採点しているか（銘柄単位）===")
    print(f"  {'帯':<10}{'銘柄数':>8}{'スコアの%点':>12}{'20日リターン':>13}{'正例率':>8}")
    for k in BREAK_JA:
        g = d[d["bin"] == k]
        print(f"  {k:<10}{len(g):>8,}{g['s_pct'].mean():>11.1f}%{g['r'].mean():>+12.2f}%"
              f"{g['label'].mean()*100:>7.1f}%")
    print(f"  スコアの百分位と log(その日の発火数) の相関 "
          f"{np.corrcoef(d['s_pct'], np.log(d['n_break']))[0, 1]:+.3f}")

    for title, sub in (("母集団", d), ("3モデル 90以上", d[d["3モデル 90以上"]])):
        print(f"\n=== リターンの分布と、最大上昇 対 持ち切り（{title}）===")
        print(f"  {'帯':<10}{'銘柄数':>8}{'20日 平均':>10}{'中央値':>8}{'+10%超':>8}{'-10%割れ':>9}"
              f"{'最大上昇 平均':>13}{'+30%超':>8}{'持ち切り':>9}{'最大−持ち切り':>13}")
        for k in BREAK_JA:
            g = sub[sub["bin"] == k]
            if len(g) < 20:
                print(f"  {k:<10}{len(g):>8,}  （件数不足）")
                continue
            print(f"  {k:<10}{len(g):>8,}{g['r'].mean():>+9.2f}%{g['r'].median():>+7.2f}%"
                  f"{(g['r'] > 10).mean()*100:>7.1f}%{(g['r'] < -10).mean()*100:>8.1f}%"
                  f"{g['rise'].mean():>+12.2f}%{(g['rise'] > 30).mean()*100:>7.1f}%"
                  f"{g['end'].mean():>+8.2f}%{g['rise'].mean()-g['end'].mean():>+12.2f}pt")


def main() -> int:
    os.makedirs(OOF_DIR, exist_ok=True)
    df = add_rules(load_scores())
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    mkt = (frame[["Date", "topix_ma200_gap", "topix_ret_20", "topix_vol_20"]]
           .drop_duplicates("Date"))
    days = day_table(df).merge(mkt, on="Date", how="left")
    days["break_bin"] = pd.cut(days["n_break"], BREAK_BINS, labels=BREAK_JA)
    days["mkt_bin"] = pd.qcut(days["topix_ma200_gap"], 5,
                              labels=["地合い 最弱", "弱", "中", "強", "最強"])
    log(f"対象 {len(days):,}営業日 / 発火 {days['n_break'].sum():,}件 "
        f"（1日 {days['n_break'].mean():.1f}件 / 中央値 {days['n_break'].median():.0f}）")
    print(f"  発火数と TOPIX 乖離の相関 {days['n_break'].corr(days['topix_ma200_gap']):+.3f}")
    days.to_csv(os.path.join(OOF_DIR, "e30_days.csv"), index=False)

    for by, order, title in (("break_bin", BREAK_JA, "その日の発火数で切る"),
                             ("mkt_bin", list(days["mkt_bin"].cat.categories),
                              "地合い（TOPIX の200日線からの乖離）で切る")):
        t = show(days, by, order, title
                 + "（最良1件は基準なし・毎営業日1件。超過＝その日の母集団平均との差）")
        t.to_csv(os.path.join(OOF_DIR, f"e30_{by}_universe.csv"), index=False)
        for rname, _, _ in OR.RULES:
            r = show_rule(days, by, order, rname)
            r.to_csv(os.path.join(OOF_DIR, f"e30_{by}_{rname.split()[0]}.csv"), index=False)

    # 発火数と地合いのどちらが効いているか（地合いを固定して発火数で切る）
    print("\n=== 地合いを固定して発火数で切る（母集団 ret20、日平均）===")
    piv = days.pivot_table(index="mkt_bin", columns="break_bin", values="uni_ret",
                           aggfunc="mean", observed=False)
    cnt = days.pivot_table(index="mkt_bin", columns="break_bin", values="uni_ret",
                           aggfunc="size", observed=False)
    print("  " + "".join(f"{c:>12}" for c in piv.columns))
    for i in piv.index:
        print(f"  {str(i):<10}" + "".join(
            f"{piv.loc[i, c]:>+9.2f}%({int(cnt.loc[i, c]) if pd.notna(cnt.loc[i, c]) else 0:>3})"
            if pd.notna(piv.loc[i, c]) else f"{'-':>12}" for c in piv.columns))
    piv.to_csv(os.path.join(OOF_DIR, "e30_pivot_universe.csv"))
    supplement(df)
    log(f"記録: {OOF_DIR}/e30_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
