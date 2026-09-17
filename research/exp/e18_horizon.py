#!/usr/bin/env python3
"""
実験18: 目的変数を 60営業日 から 20営業日 に短くしてよいか。

なぜ測るか
--------
運用の手仕舞いが1ヶ月前後なのに、ラベルの基準点が3ヶ月先にあった。
「売ったあとに起きたこと」で正例・負例を決めていたことになる。
短くすること自体は運用の要請だが、**精度を落とさずにできるか**は別の話で、
測らずに本番の学習へ入れるわけにはいかない。

何と何を比べるか
--------------
  旧   h=60 / MA20>=MA60   これまでの本番
  中間 h=20 / MA20>=MA60   ホライズンだけ縮めた状態（トレンド条件が事実上無効）
  新   h=20 / MA5>=MA20    ホライズンに合わせて移動平均も縮めた状態
  無   h=20 / 条件なし      トレンド条件を明示的に外した状態

「中間」を置くのは、新旧の差が「ホライズンを縮めたこと」と
「トレンド条件を直したこと」のどちらから来ているかを分けるため。
2つ同時に変えて良くなっても、どちらが効いたのか分からない。

「無」を後から足したのは、3条件の結果で「中間」がいちばん良かったため。
「中間」は MA20>=MA60 が h=20 では削るのが2件しかなく、実質トレンド条件なし
のはず——だが「はず」で結論を出さない。同じかどうかを測る。

物差し
-----
  ret_o1_20  翌営業日の寄り買い・20営業日後の5日平均終値売り（新しい既定）
  ret_o1_40  同・40営業日（これまでの既定。過去の記録と繋げるため併記）

物差しはラベルに依存しないので、3条件を同じ2つの物差しで測れる。
ただし **20日と40日の数字どうしを直接比べてはいけない**（保有期間が違えば
リターンの大きさが違うのは当たり前）。見るのは「同じ物差しの中での条件間の差」。

公平にするための細工
------------------
1. **評価する行を揃える**。h=60 のラベルは直近60営業日ぶんが未確定なので、
   そのままだと h=20 のほうが新しいデータを多く使えてしまう。
   3条件すべてでラベルが確定している行だけに絞って比較する。
   （データ量が増えること自体は h=20 の利点だが、それはラベルの質とは
   別の話なので、混ぜずに最後に別途書く。）
2. **窓を揃える**。lab.folds のエンバーゴは B.RISE_HORIZON から取るので、
   条件ごとに変わると窓の形まで変わってしまう。実験中は厳しいほう（60）に
   固定して、全条件で同一の窓・同一の訓練行にする。
   本番の h=20 は 20営業日のエンバーゴで走るので、この実験は h=20 側に
   やや不利な（保守的な）見積りになる。

判定の足切り
-----------
実験11 のノイズ床（同一設定・種5個）
  窓平均 レンジ 0.143pt  <- 最も解像度が高い
採否は z = 差 / √(SE_a² + SE_b²) > 2。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import sweep_design as S  # noqa: E402

SEEDS = (42, 7, 123)
OUTCOMES = ("ret_o1_20", "ret_o1_40")
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")

#: 比較する条件。(名前, RiseConfig)
ARMS = [
    ("旧   h=60 / MA20>=MA60", B.RiseConfig(horizon=60, trend_short=20,
                                            trend_long=60)),
    ("中間 h=20 / MA20>=MA60", B.RiseConfig(horizon=20, trend_short=20,
                                            trend_long=60)),
    ("新   h=20 / MA5>=MA20", B.RiseConfig(horizon=20, trend_short=5,
                                           trend_long=20)),
    # 「中間」は MA20>=MA60 が h=20 では事実上無効（削るのが2件）なので、
    # 実質「トレンド条件なし」と同じはず。同じかどうかは推測せずに測る。
    ("無   h=20 / 条件なし", B.RiseConfig(horizon=20, require_uptrend=False)),
]


def label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """各条件ぶんのラベルを作って df に横付けする。"""
    import glob

    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel = S.Panels(bars).get(B.HIGH_WINDOW)
    for i, (name, cfg) in enumerate(ARMS):
        lb = S.labels_for(panel, cfg)[["Code", "Date", "label"]]
        lb = lb.rename(columns={"label": f"y{i}"})
        df = df.merge(lb, on=["Code", "Date"], how="left")
        n = int(df[f"y{i}"].notna().sum())
        print(f"  {name}: 確定 {n:,}件 / 正例率 "
              f"{df[f'y{i}'].mean()*100:.2f}%")
    return df


def edge(oof: pd.DataFrame, outcome: str) -> dict:
    m = lab.threshold_edge(oof, outcome=outcome)
    m["se"] = m["thr_fold_sd"] / np.sqrt(max(1, m["thr_folds"]))
    return m


def main() -> int:
    # 窓（エンバーゴ）を全条件で揃える。細工2 の理由はモジュール docstring。
    B.RISE_HORIZON = 60

    df = lab.frame()
    cols = F.columns("all")
    os.makedirs(OOF_DIR, exist_ok=True)

    print(f"母集団 {len(df):,}件")
    print(f"ラベルを{len(ARMS)}条件ぶん作り直す")
    df = label_columns(df)

    ycols = [f"y{i}" for i in range(len(ARMS))]
    both = df[ycols].notna().all(axis=1)
    df = df[both].reset_index(drop=True)
    print(f"\n全条件でラベルが確定している行 {len(df):,}件に絞って比較する")
    print(f"期間 {df['Date'].min().date()} 〜 {df['Date'].max().date()}")
    for i, (name, _) in enumerate(ARMS):
        print(f"  {name}: 正例率 {df[f'y{i}'].mean()*100:.2f}%")
    print(f"\n種 {SEEDS} の確率平均 / 物差し {OUTCOMES}")
    print("エンバーゴは全条件で60営業日に固定（窓を揃えるため）")

    runs = {}
    for i, (name, cfg) in enumerate(ARMS):
        p = os.path.join(OOF_DIR, f"h_{i}.parquet")
        if os.path.exists(p):
            runs[name] = lab.attach_outcomes(pd.read_parquet(p), df)
            print(f"\n  {name}: 保存済みを読む")
            continue
        sub = df.copy()
        sub["label"] = sub[f"y{i}"]
        print(f"\n  {name}: 学習")
        r = lab.run_multi(sub, lambda s: lab.lgbm(seed=s), seeds=SEEDS,
                          cols=cols, name=f"h{i}")
        r.oof.to_parquet(p, index=False)
        runs[name] = r.oof
        print(f"    out-of-fold {len(r.oof):,}件")

    for outcome in OUTCOMES:
        print(f"\n=== 物差し {outcome} ===")
        print(f"  {'条件':<24}{'窓平均':>10}{'標準誤差':>10}"
              f"{'勝ち窓':>9}{'最悪の窓':>11}{'取引数':>9}")
        ms = {}
        for name, oof in runs.items():
            m = edge(oof, outcome)
            ms[name] = m
            print(f"  {name:<24}{m['thr_fold_mean']:>+9.2f}pt{m['se']:>10.2f}"
                  f"{m['thr_folds_won']:>6}/{m['thr_folds']:<2}"
                  f"{m['thr_worst']:>+10.2f}pt{m['thr_n']:>9,}")

        print("  --- 差の検定（足切り z>2）---")
        pairs = [(0, 1, "ホライズンを縮めた効果（60 -> 20、トレンド条件は据え置き）"),
                 (1, 2, "トレンド条件を直した効果（MA20>=MA60 -> MA5>=MA20）"),
                 (1, 3, "「中間」は本当にトレンド条件なしと同じか"),
                 (3, 2, "トレンド条件なし -> MA5>=MA20 を足す効果"),
                 (0, 2, "合計（旧 -> 新）")]
        names = [n for n, _ in ARMS]
        for a, b, note in pairs:
            ma, mb = ms[names[a]], ms[names[b]]
            d = mb["thr_fold_mean"] - ma["thr_fold_mean"]
            se = np.sqrt(ma["se"] ** 2 + mb["se"] ** 2)
            z = d / se if se else float("nan")
            v = "採用可" if z > 2 else ("要確認" if z > 1 else
                                     ("差なし" if z > -1 else "悪化"))
            print(f"    {note}")
            print(f"      差 {d:+.2f}pt / 合成SE {se:.2f} / z {z:+.2f} → {v}")

    # データ量の利点は別枠。ラベルの質とは別の話なので混ぜない。
    print("\n=== 参考: ラベルが確定する行数（絞る前）===")
    full = lab.frame()
    full = label_columns(full)
    for i, (name, _) in enumerate(ARMS):
        n = int(full[f"y{i}"].notna().sum())
        last = full.loc[full[f"y{i}"].notna(), "Date"].max()
        print(f"  {name}: {n:,}件 / 末尾 {pd.Timestamp(last).date()}")
    print("  短いホライズンはラベルが早く確定するぶん、直近のデータを使える。")
    print("  上の比較では揃えて消してあるので、この利点は上の数字に入っていない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
