#!/usr/bin/env python3
"""
実験02: 何がスコアを上げ下げしているのか。当たりと外れは何が違うのか。

出すもの
--------
1. グローバル寄与        平均|SHAP| で、どの特徴がスコアを動かしているか
2. 上位を押し上げる要因  スコア上位5%の行で、SHAP の符号つき平均
3. 下位に落とす要因      スコア下位5%の行で、同じもの
4. 当たり外れの違い      上位5%の中で label=1 と label=0 の特徴値を比べる
5. 取りこぼしの正体      下位50%に居る正例は何が違うのか
6. 分離可能性の検定      「高スコアの中で当たりと外れを見分けられるか」を
                         別のモデルで直接学習して測る。これが効かないなら、
                         残差は特徴量の不足ではなく本質的な不確実性。

SHAP は out-of-fold で取る。全期間モデルの自己説明を見ると、
訓練データを覚えた結果を「要因」と読んでしまう。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402

TOP_PCT = 5
N_SHOW = 18


def oof_shap(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    各行の SHAP を、その行を訓練に使っていないモデルから取る。

    LightGBM の pred_contrib は対数オッズ空間の寄与を返す。
    最後の列は期待値（base value）なので特徴量ぶんだけ切り出す。
    """
    import lightgbm as lgb

    params = {k: v for k, v in tuning.params_for("all").items()
              if not k.startswith("_")}
    params.update(verbose=-1, random_state=lab.SEED)
    d = pd.to_datetime(df["Date"])
    lb = df["label"].notna()
    parts = []
    for f in lab.folds(df):
        tr = df[(d <= f.train_end) & lb]
        te = df[(d >= f.test_start) & (d <= f.test_end) & lb]
        if len(te) < lab.MIN_TEST or len(tr) < lab.MIN_TRAIN:
            continue
        ytr = tr["label"].to_numpy(dtype=int)
        m = lgb.LGBMClassifier(**params, scale_pos_weight=lab._spw(ytr))
        m.fit(tr[cols].to_numpy(dtype=float), ytr)
        Xte = te[cols].to_numpy(dtype=float)
        contrib = m.booster_.predict(Xte, pred_contrib=True)[:, :len(cols)]
        part = te[["Code", "Date", "label", "ref_end"]].reset_index(drop=True)
        part["score"] = m.predict_proba(Xte)[:, 1]
        parts.append(pd.concat([part, pd.DataFrame(contrib, columns=cols)], axis=1))
        print(f"  窓{f.index} SHAP {len(te):,}件")
    return pd.concat(parts, ignore_index=True)


def _fmt(name: str) -> str:
    return f"{name[:26]:<27}[{F.group_of(name)[:12]:<12}]"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    sh = oof_shap(df, cols)
    C = sh[cols]

    section("1. グローバル寄与（平均|SHAP|・対数オッズ空間）")
    glob = C.abs().mean().sort_values(ascending=False)
    tot = glob.sum()
    for c, v in glob.head(N_SHOW).items():
        print(f"  {_fmt(c)} {v:>7.4f}  シェア {v/tot*100:>4.1f}%")
    print()
    grp = pd.Series({c: F.group_of(c) for c in cols})
    by_g = pd.DataFrame({"v": glob, "g": grp}).groupby("g")["v"].sum().sort_values(ascending=False)
    print("  --- グループ別の合計シェア ---")
    for g, v in by_g.head(12).items():
        print(f"    {g:<18} {v/tot*100:>5.1f}%  {'#'*int(v/tot*60)}")

    n_top = max(1, int(len(sh) * TOP_PCT / 100))
    top = sh.nlargest(n_top, "score")
    bot = sh.nsmallest(n_top, "score")

    # 特徴量の実値は1度だけ結合しておく（列ごとに merge すると遅いうえ読みにくい）
    vals = df[["Code", "Date"] + cols]
    top_v = top[["Code", "Date"]].merge(vals, on=["Code", "Date"], how="left")
    pop_med = df[cols].median()

    section(f"2. スコア上位{TOP_PCT}%を押し上げている要因（符号つき平均SHAP）")
    st = top[cols].mean().sort_values(ascending=False)
    print(f"  {'特徴量':<41}{'SHAP':>9}{'母集団中央':>12}{'上位中央':>12}")
    print("  [押し上げ]")
    for c, v in st.head(10).items():
        print(f"    {_fmt(c)}{v:>+9.4f}{pop_med[c]:>12.3g}{top_v[c].median():>12.3g}")
    print("  [それでも押し下げているもの]")
    for c, v in st.tail(5).items():
        print(f"    {_fmt(c)}{v:>+9.4f}{pop_med[c]:>12.3g}{top_v[c].median():>12.3g}")

    section(f"3. スコア下位{TOP_PCT}%を落としている要因")
    sb = bot[cols].mean().sort_values()
    for c, v in sb.head(10).items():
        print(f"    {_fmt(c)} {v:>+7.4f}")

    section(f"4. 上位{TOP_PCT}%の中で、当たりと外れは何が違うか")
    merged = top[["Code", "Date", "label", "score", "ref_end"]].merge(
        df[["Code", "Date"] + cols], on=["Code", "Date"], how="left")
    hit = merged[merged["label"] == 1]
    mis = merged[merged["label"] == 0]
    print(f"  上位{TOP_PCT}% {len(merged):,}件  当たり {len(hit):,}件 ({len(hit)/len(merged)*100:.1f}%) "
          f"/ 外れ {len(mis):,}件")
    print(f"  実収益  当たり {hit['ref_end'].mean()*100:+.2f}%  外れ {mis['ref_end'].mean()*100:+.2f}%")
    print()
    rows = []
    for c in cols:
        a, b = hit[c].dropna(), mis[c].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if not np.isfinite(pooled) or pooled == 0:
            continue
        rows.append((c, (a.median() - b.median()), (a.mean() - b.mean()) / pooled,
                     a.median(), b.median()))
    rows.sort(key=lambda r: -abs(r[2]))
    print(f"  {'特徴量':<41}{'当たり':>11}{'外れ':>11}{'効果量d':>9}")
    for c, _, d, am, bm in rows[:N_SHOW]:
        print(f"  {_fmt(c)}{am:>11.3g}{bm:>11.3g}{d:>+9.3f}")

    section("5. 取りこぼし（スコア下位50%に居る正例）は何が違うか")
    med = sh["score"].median()
    # sh は SHAP 値を特徴量と同じ列名で持っている。実値と結合する前に
    # 識別子だけに絞らないと、同名列がぶつかって _x/_y に化ける
    low = sh.loc[sh["score"] <= med, ["Code", "Date", "label", "score"]].merge(
        vals, on=["Code", "Date"], how="left")
    lh, lm = low[low["label"] == 1], low[low["label"] == 0]
    print(f"  下位50% {len(low):,}件  うち正例 {len(lh):,}件 ({len(lh)/len(low)*100:.1f}%)")
    rows2 = []
    for c in cols:
        a, b = lh[c].dropna(), lm[c].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if not np.isfinite(pooled) or pooled == 0:
            continue
        rows2.append((c, (a.mean() - b.mean()) / pooled, a.median(), b.median()))
    rows2.sort(key=lambda r: -abs(r[1]))
    print(f"  {'特徴量':<41}{'正例':>11}{'負例':>11}{'効果量d':>9}")
    for c, d, am, bm in rows2[:12]:
        print(f"  {_fmt(c)}{am:>11.3g}{bm:>11.3g}{d:>+9.3f}")

    section("6. 高スコア群の中で、当たりと外れを見分けられるか")
    print("  上位20%だけを取り出し、そこで label を直接予測する。")
    print("  ここで AUC が 0.5 付近なら、残差は特徴量の不足ではなく本質的な不確実性。")
    print()
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    n20 = int(len(sh) * 0.20)
    sub = sh.nlargest(n20, "score")[["Code", "Date", "label"]].merge(
        vals, on=["Code", "Date"], how="left")
    # 時系列で前7割を訓練、後ろ3割を検証にする。ランダム分割だと
    # 未来を見て過去を当てることになり、判別できて当然になる
    sub = sub.sort_values("Date").reset_index(drop=True)
    cut = int(len(sub) * 0.7)
    tr, te = sub.iloc[:cut], sub.iloc[cut:]
    params = {k: v for k, v in tuning.params_for("all").items()
              if not k.startswith("_")}
    params.update(verbose=-1, random_state=lab.SEED)
    y = tr["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**params, scale_pos_weight=lab._spw(y))
    m.fit(tr[cols].to_numpy(dtype=float), y)
    p = m.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
    auc = roc_auc_score(te["label"].to_numpy(dtype=int), p)
    print(f"  上位20% {len(sub):,}件（訓練{len(tr):,} / 検証{len(te):,}）")
    print(f"  当たり外れの判別 AUC = {auc:.4f}")
    print(f"  {'-> まだ見分けられる余地がある' if auc > 0.55 else '-> ほぼ見分けられない（本質的な不確実性）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
