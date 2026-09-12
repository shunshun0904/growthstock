#!/usr/bin/env python3
"""
実験06: 「スコアが過去分布の上位X%」で買う運用を測る。

測る対象
--------
画面の棒グラフ（pctHistorical）が示しているのは

    pctHistorical = 過去スコア分布のうち、この銘柄のスコアより低い割合 × 100

で、predict_daily.py が research/model/oof.parquet から作っている。
「80を超えたら買う」は、その日の1位かどうかに関係なく、スコアの水準で
絶対的に切る運用。日によって買わない日もあれば複数買う日もある。

毎日1位（実験05 で +0.62pt）とは別の戦略なので、別に測る。

先読みを避ける
-------------
本番は OOF 全期間の分布でパーセンタイルを出している。過去を評価するときに
これを使うと、未来のスコア分布を見てしきい値を決めることになる。
ここでは窓ごとに「それより前の窓のスコアだけ」を参照分布にする。
窓1 は参照が無いので対象外（結果は窓2〜10 の集計）。

参考として、本番と同じ「全期間分布」で切った場合も併記する。
両者の差が大きければ、画面に出ている pctHistorical 自体が
やや楽観側に寄っているということになる。

比較相手
--------
  同じ窓の全候補の平均         買わない日も含めて、母集団をそのまま買った場合
  買った日の候補平均           「買う日を選ぶ力」を除いて「銘柄を選ぶ力」だけを見る
この2つを分けないと、タイミングの寄与と銘柄選定の寄与が混ざる。

安定性
------
実験05 で、pooled の上位5%は44%が窓9 に集中していた。日をまたぐ絶対
しきい値は同じ弱さを持つので、窓ごとの内訳を必ず出す。
どこか1つの窓に偏っていたら、その優位は再現しない。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lab  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PCTS = (70, 80, 85, 90, 95)


def load(name: str) -> pd.DataFrame:
    p = os.path.join(OOF_DIR, f"{name}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"{p} がありません。先に e05_pooling.py を回してください")
    return pd.read_parquet(p)


def walk_forward_threshold(oof: pd.DataFrame, pct: float) -> pd.DataFrame:
    """
    窓ごとに、それより前の窓のスコア分布で切る。

    その時点で手に入る情報だけを使う。窓1 は参照分布が無いので落とす。
    """
    out = []
    for f in sorted(oof["fold"].unique()):
        ref = oof.loc[oof["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        thr = float(np.percentile(ref, pct))
        cur = oof[oof["fold"] == f]
        sel = cur[cur["score"] > thr].copy()
        sel["thr"] = thr
        out.append(sel)
    return pd.concat(out, ignore_index=True) if out else oof.iloc[:0].copy()


def whole_period_threshold(oof: pd.DataFrame, pct: float) -> pd.DataFrame:
    """本番と同じ切り方（全期間の分布）。先読みを含む参考値。"""
    thr = float(np.percentile(oof["score"].to_numpy(), pct))
    return oof[oof["score"] > thr].copy()


def report(oof: pd.DataFrame, sel: pd.DataFrame, universe: pd.DataFrame) -> dict:
    """universe は比較の母数（同じ窓の全候補）。"""
    e = pd.to_numeric(sel["ref_end"], errors="coerce")
    all_end = pd.to_numeric(universe["ref_end"], errors="coerce").mean()
    # 「買った日」に限った候補平均。買う日を選ぶ力を除いた比較相手
    traded_days = set(pd.Index(sel["Date"]).unique())
    same_days = universe[universe["Date"].isin(traded_days)]
    same_end = pd.to_numeric(same_days["ref_end"], errors="coerce").mean()
    n_days_all = universe["Date"].nunique()
    return {
        "n": len(sel), "days": sel["Date"].nunique(), "days_all": n_days_all,
        "per_day": len(sel) / max(1, sel["Date"].nunique()),
        "end": float(e.mean()) * 100,
        "vs_all": float(e.mean() - all_end) * 100,
        "vs_same_day": float(e.mean() - same_end) * 100,
        "pos": float(sel["label"].mean()) * 100,
        "win": float((e > 0).mean()) * 100,
    }


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "base"
    oof = load(name)
    print(f"モデル: {name} / out-of-fold {len(oof):,}件 "
          f"/ 窓 {oof['fold'].min()}〜{oof['fold'].max()}")
    print(f"母集団の実収益 {pd.to_numeric(oof['ref_end']).mean()*100:+.2f}% "
          f"/ 正例率 {oof['label'].mean()*100:.1f}%")

    # 窓1 は参照分布が無いので、比較の母数も窓2以降に揃える
    uni = oof[oof["fold"] > oof["fold"].min()]
    print(f"評価対象: 窓{uni['fold'].min()}〜{uni['fold'].max()} "
          f"({len(uni):,}件 / {uni['Date'].nunique():,}日)")
    print()

    print("=== 窓ごとに、それより前の窓の分布で切る（先読みなし）===")
    print(f"  {'しきい値':>8}{'取引数':>8}{'取引日':>8}{'/全日':>8}{'1日あたり':>10}"
          f"{'実収益':>10}{'対全候補':>10}{'対同日候補':>11}{'正例率':>8}{'勝率':>8}")
    rows = {}
    for p in PCTS:
        sel = walk_forward_threshold(oof, p)
        if sel.empty:
            print(f"  上位{100-p:>2}%   該当なし")
            continue
        s = report(oof, sel, uni)
        rows[p] = (sel, s)
        print(f"  上位{100-p:>2}%{s['n']:>8,}{s['days']:>8,}{s['days_all']:>8,}"
              f"{s['per_day']:>10.2f}{s['end']:>+9.2f}%{s['vs_all']:>+9.2f}pt"
              f"{s['vs_same_day']:>+10.2f}pt{s['pos']:>7.1f}%{s['win']:>7.1f}%")

    print()
    print("=== 参考: 本番と同じ全期間分布で切る（先読みを含む）===")
    print(f"  {'しきい値':>8}{'取引数':>8}{'実収益':>10}{'対全候補':>10}{'勝率':>8}")
    for p in PCTS:
        sel = whole_period_threshold(oof, p)
        s = report(oof, sel, oof)
        print(f"  上位{100-p:>2}%{s['n']:>8,}{s['end']:>+9.2f}%"
              f"{s['vs_all']:>+9.2f}pt{s['win']:>7.1f}%")

    print()
    for pct in (80, 90, 95):
        if pct not in rows:
            continue
        print(f"=== 安定性: pctHistorical > {pct}（上位{100-pct}%）の窓ごとの内訳 ===")
        sel, _ = rows[pct]
        print(f"  {'窓':>4}{'取引数':>8}{'シェア':>8}{'実収益':>10}"
              f"{'同窓の全候補':>13}{'差':>9}{'勝率':>8}")
        for f in sorted(sel["fold"].unique()):
            g = sel[sel["fold"] == f]
            u = uni[uni["fold"] == f]
            ge = pd.to_numeric(g["ref_end"], errors="coerce")
            ue = pd.to_numeric(u["ref_end"], errors="coerce")
            print(f"  {int(f):>4}{len(g):>8,}{len(g)/len(sel)*100:>7.0f}%"
                  f"{ge.mean()*100:>+9.2f}%{ue.mean()*100:>+12.2f}%"
                  f"{(ge.mean()-ue.mean())*100:>+8.2f}pt{(ge>0).mean()*100:>7.1f}%")
        wins = sum(1 for f in sel["fold"].unique()
                   if pd.to_numeric(sel[sel["fold"] == f]["ref_end"]).mean()
                   > pd.to_numeric(uni[uni["fold"] == f]["ref_end"]).mean())
        n_f = sel["fold"].nunique()
        print(f"  -> 同窓の全候補を上回った窓: {wins}/{n_f}"
              f" / 最大シェアの窓 {sel['fold'].value_counts().max()/len(sel)*100:.0f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
