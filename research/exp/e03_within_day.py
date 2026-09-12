#!/usr/bin/env python3
"""
実験03: 日付内で動く情報と、日付間でしか動かない情報を分ける。

動機（実験02 の SHAP から）
--------------------------
寄与の 22.6% が market グループに行っている。ところがこのグループは
topix_ret_120 / nk225_ret_120 / gold_ret_120 のように**同じ日なら全銘柄で
同じ値**なので、その日の候補どうしの順位付けには一切効かない。
日全体のスコアを上下させているだけ。日付内AUCが 0.595 と低いのはこれが理由。

さらに危ない点がある。母集団は約1,949日しかなく、ラベルが60営業日先を見るので
隣接する日は強く相関する。日付単位の実効標本数は数十しかない。
そこに日付定数の特徴量を11列も与えると、「どの相場局面か」を覚えるだけの
余地が大きい。gold_ret_120 が寄与3位というのは、その兆候に見える。

一方、価格・ブレイク・需給・業種指数の特徴量は日付内で銘柄ごとに変わるのに、
日付内順位化が掛かっていない（既存の add_cross_sectional_ranks は
RAW_FOR_RANK = 財務系だけを対象にしている）。絶対値のままだと
「その日の地合いぶんの下駄」が乗ったまま比較されることになる。

試す形
------
  base            現行151列
  -market         market グループ(11列)を落とす
  +rank_px        価格・ブレイク・流動性・需給・業種指数を日付内順位で追加
  -market+rank_px 両方

指標の読み方
-----------
  日付内AUC   その日の候補の順位付け。運用の「どれを買うか」に直結
  上位5%収益  日をまたいだ選択も含む。「そもそも買う日か」も効く
market を落とすと後者だけ下がる、という形になるはず。両方見ないと判断を誤る。

すべて5種平均（実験00 のノイズ床対策）。単一種だと上位5%収益は
レンジ0.81pt 動くので、それ未満の差は読めない。
"""
from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

#: 日付内で銘柄ごとに変わるのに、順位化されていないグループ
PX_GROUPS = ("price", "breakout", "liquidity", "supply", "volume",
             "sector_index", "scale")


def add_within_day_ranks(df: pd.DataFrame, cols: List[str]) -> tuple:
    """
    同じ日付内のパーセンタイル順位(0〜1)を列として足す。

    既存の add_cross_sectional_ranks と同じ作法にする。
      ・元の列は残す（絶対値と順位のどちらが効くかを比べたいので）
      ・欠測は欠測のまま。0.5 で埋めると「中位だった」という
        観測していない情報を与えることになる
      ・その日に有効な値が2件未満なら順位は定義できないので欠測
    """
    target = [c for c in cols if F.group_of(c) in PX_GROUPS]
    out = df.copy()
    g = out.groupby("Date", sort=False)
    added = []
    for c in target:
        valid = g[c].transform("count") >= 2
        out[f"{c}_dr"] = g[c].rank(pct=True, method="average").where(valid)
        added.append(f"{c}_dr")
    return out, added


#: 選抜は3種で回す（標準誤差 0.31/√3 = 0.18pt、検出できる差は約0.5pt）。
#: 勝った案だけ5種で確認する。全部を5種で回すと4設定で13分かかり、
#: 試行回数が稼げない。コンペと同じで、粗く振るってから精査する。
SCREEN_SEEDS = (42, 7, 123)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="種の数（3=選抜 / 5=確認）")
    args = ap.parse_args()
    seeds = lab.SEEDS[:args.seeds] if args.seeds != 3 else SCREEN_SEEDS

    df = lab.frame()
    cols = F.columns("all")
    market = [c for c in cols if F.group_of(c) == "market"]
    df, rank_cols = add_within_day_ranks(df, cols)
    no_market = [c for c in cols if c not in set(market)]

    print(f"特徴量 {len(cols)}列 / market {len(market)}列 / 追加した日付内順位 {len(rank_cols)}列")
    print(f"種 {len(seeds)}個: {seeds}")
    print(f"market: {', '.join(market)}")
    print()

    runs = {
        "base": cols,
        "-market": no_market,
        "+rank_px": cols + rank_cols,
        "-market+rank_px": no_market + rank_cols,
    }
    results = {}
    for name, use in runs.items():
        results[name] = lab.run_multi(df, lambda s: lab.lgbm(seed=s),
                                      seeds=seeds, cols=use, name=name)
        sp = lab.spread(results[name])
        print(f"  {name:<18} {len(use):>3}列  種ごと上位5%収益 "
              f"平均{sp['mean']:+.2f}% (SD {sp['sd']:.2f}) -> 種平均 {sp['avg_of_ensemble']:+.2f}%")

    print()
    print(lab.table(results))
    print()
    print("=== ベースラインとの対比較（5種平均どうし）===")
    base = results["base"]
    for name, r in results.items():
        if name == "base":
            continue
        c = lab.compare(base, r)
        lo, hi = c["end_ci"]
        print(f"{name:<18} 上位5%収益 {c['end_diff']:+.2f}pt [{lo:+.2f}, {hi:+.2f}] "
              f"改善確率{c['p_better']*100:>3.0f}%  | PR-AUC {c['pr_auc_diff']:+.4f} "
              f"| 日付内AUC {c['auc_in_day_diff']:+.4f}")

    print()
    print("=== 種平均そのものの効果（アンサンブルとして）===")
    for name, r in results.items():
        sp = lab.spread(r)
        gain = sp["avg_of_ensemble"] - sp["mean"]
        print(f"  {name:<18} 単一種の平均 {sp['mean']:+.2f}%  -> 5種平均 "
              f"{sp['avg_of_ensemble']:+.2f}%  ({gain:+.2f}pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
