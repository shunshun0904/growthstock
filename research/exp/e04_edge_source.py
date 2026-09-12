#!/usr/bin/env python3
"""
実験04: エッジは「いつ買うか」から来ているのか「何を買うか」から来ているのか。

実験03 で分かったこと（仮説が外れた）
------------------------------------
market グループ（同じ日なら全銘柄で同じ値の11列）を落とすと、
上位5%収益が +5.43% -> +3.71% に落ちた。対母集団の優位は
+2.02pt -> +0.29pt と、ほぼ消える。改善確率0%。

「日付内順位に効かない列だから外していい」という私の読みは逆だった。
ただし外して ROC-AUC は上がっている（0.5911 -> 0.5993）。
つまり market 列は、ラベルの当て方ではなく**実収益の取り方**に効いている。

機構の見立て: market 列は水準を上下させるだけでなく、木の中で
「どの局面か」を分岐させ、銘柄側の特徴量の使われ方を切り替えている。
だから日付内AUC まで下がる（0.5975 -> 0.5847）。単なる下駄ではない。

ここで確かめること
------------------
上位5%という指標は、日をまたいだ選択（いつ買うか）と日の中の選択
（何を買うか）が混ざっている。運用は毎晩4〜5件の候補から選ぶので、
この2つは別の能力であり、別々に測らないと打ち手を間違える。

  market_only  11列だけ。銘柄の情報を一切持たない。
               これで上位5%のエッジが出るなら、それはタイミングの力。
  stock_only   140列。日付定数を持たない。銘柄選定の力だけ。
  base         両方。

指標
----
  上位5%収益    日をまたぐ選択込み（いつ買うか + 何を買うか）
  毎日1位       毎日その日の1位を1件買う。日数が固定されるので
                タイミングの力が消え、銘柄選定の腕だけが残る
  対その日平均  毎日1位 − その日の候補平均。これが銘柄選定の純粋な取り分
  日付内AUC     同上をラベル基準で見たもの

market_only の「毎日1位」は、その日の全候補が同スコアになるため
事実上ランダムな1件になる。対その日平均がゼロ近辺になるはずで、
これが分解が効いていることの確認になる。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    market = [c for c in cols if F.group_of(c) == "market"]
    stock = [c for c in cols if c not in set(market)]

    runs = {
        "base(151)": cols,
        "market_only(11)": market,
        "stock_only(140)": stock,
    }
    results = {}
    for name, use in runs.items():
        results[name] = lab.run_multi(df, lambda s: lab.lgbm(seed=s),
                                      seeds=SCREEN_SEEDS, cols=use, name=name)
        print(f"  {name:<18} 完了")

    print()
    print(lab.table(results))
    print()
    print("=== 読み方 ===")
    b = results["base(151)"].metrics
    m = results["market_only(11)"].metrics
    s = results["stock_only(140)"].metrics
    print(f"  母集団の実収益                 {b['base_end']:+.2f}%")
    print()
    print(f"  上位5%収益（いつ買うか + 何を買うか）")
    print(f"     base         {b['end_5']:+.2f}%  (対母集団 {b['lift_5']:+.2f}pt)")
    print(f"     market_only  {m['end_5']:+.2f}%  (対母集団 {m['lift_5']:+.2f}pt)"
          f"  <- タイミングだけで取れる分")
    print(f"     stock_only   {s['end_5']:+.2f}%  (対母集団 {s['lift_5']:+.2f}pt)"
          f"  <- 銘柄選定だけで取れる分")
    print()
    print(f"  毎日1位を買う（日数固定 = 銘柄選定の腕だけ）")
    for name, r in results.items():
        x = r.metrics
        print(f"     {name:<18} {x['pick1_end']:+.2f}%  対その日平均 "
              f"{x['pick1_lift']:+.2f}pt  勝率 {x['pick1_win']*100:.1f}%  "
              f"({x['pick1_n']}日)")
    print()
    print("=== ベースラインとの対比較 ===")
    base = results["base(151)"]
    for name, r in results.items():
        if name.startswith("base"):
            continue
        c = lab.compare(base, r)
        lo, hi = c["end_ci"]
        print(f"  {name:<18} 上位5%収益 {c['end_diff']:+.2f}pt [{lo:+.2f}, {hi:+.2f}] "
              f"改善確率{c['p_better']*100:>3.0f}%  | 日付内AUC {c['auc_in_day_diff']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
