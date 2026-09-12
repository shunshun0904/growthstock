#!/usr/bin/env python3
"""
実験05: 上位5%のエッジは実力か、プーリングの artifact か。

実験04 で出た矛盾
----------------
  market_only(11列)   上位5%収益 +0.65%  対母集団 -2.76pt   日付内AUC 0.5000
  stock_only(140列)   上位5%収益 +3.71%  対母集団 +0.29pt   日付内AUC 0.5847
  base(151列)         上位5%収益 +5.43%  対母集団 +2.02pt   日付内AUC 0.5975

日付定数の11列だけのモデルは、単独では母集団を2.76pt も下回る。
タイミングを当てるどころか外している。なのに、その11列を base から
落とすと 1.73pt 下がる。単独では有害なものが、組み合わせると効く。

日付内AUC がちょうど 0.5000 なのは分解が効いている証拠
（同じ日の全候補が同スコアなので順位が付かない）。

疑っていること
-------------
上位5%は「全10窓・全期間のスコアを1つの列に並べて上から5%」で選んでいる。
窓ごとに検証期間が違い、正例率も相場も違う。**スコアが期間をまたいで
可比でないなら、この選び方は「生スコアが高く出た期間」を選んでいるだけ**で、
銘柄を選ぶ力とは関係ない。

market 列はモデルに「いまどの局面か」を教えるので、期間ごとの水準を
揃える働きをする。落とすと期間をまたいだ比較が壊れ、上位5%が
でたらめな期間に偏る。それだけで 1.73pt 下がって見える可能性がある。

これが本当なら、+2.02pt という看板の数字は実力ではない。

測り方
------
同じ out-of-fold のスコアに対して、3通りの選び方を当てる。

  pooled      現行。全期間のスコアをそのまま並べて上位5%
  per_fold    窓ごとにスコアをパーセンタイル化してから並べて上位5%
              -> 窓をまたいだ非可比性だけを消す
  within_day  日付ごとにパーセンタイル化してから並べて上位5%
              -> 日をまたぐ情報を全部消す。純粋に銘柄選定だけ

base と stock_only の差が
  per_fold で消えるなら   -> 窓をまたいだ非可比性の artifact だった
  within_day で消えるなら -> 日をまたぐ選択（タイミング）の寄与だった
  どちらでも残るなら      -> 銘柄選定に効いている（交互作用として本物）

out-of-fold は保存する。以降の分析を学習なしで回せるようにするため。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402
import lab  # noqa: E402

SCREEN_SEEDS = (42, 7, 123)
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
K = 5


def get_oof(df: pd.DataFrame, name: str, cols) -> pd.DataFrame:
    """out-of-fold を作る。作ってあれば読む（学習をやり直さない）。"""
    os.makedirs(OOF_DIR, exist_ok=True)
    p = os.path.join(OOF_DIR, f"{name}.parquet")
    if os.path.exists(p):
        print(f"  {name:<18} 保存済みを読む")
        return pd.read_parquet(p)
    r = lab.run_multi(df, lambda s: lab.lgbm(seed=s), seeds=SCREEN_SEEDS,
                      cols=cols, name=name)
    r.oof.to_parquet(p, index=False)
    print(f"  {name:<18} 学習して保存")
    return r.oof


def select(oof: pd.DataFrame, mode: str, k: int = K) -> pd.DataFrame:
    """3通りの選び方。返すのは選ばれた行。"""
    n = max(1, int(len(oof) * k / 100))
    if mode == "pooled":
        key = oof["score"]
    elif mode == "per_fold":
        key = oof.groupby("fold")["score"].rank(pct=True)
    elif mode == "within_day":
        key = oof.groupby("Date")["score"].rank(pct=True)
    else:
        raise ValueError(mode)
    return oof.assign(_k=key).nlargest(n, "_k")


def stats(oof: pd.DataFrame, sel: pd.DataFrame) -> dict:
    e = pd.to_numeric(sel["ref_end"], errors="coerce")
    base = pd.to_numeric(oof["ref_end"], errors="coerce").mean()
    return {"n": len(sel), "end": float(e.mean()) * 100,
            "lift": float(e.mean() - base) * 100,
            "pos": float(sel["label"].mean()) * 100,
            "win": float((e > 0).mean()) * 100,
            "n_days": int(sel["Date"].nunique())}


def ci_diff(a: pd.DataFrame, b: pd.DataFrame, mode: str, n_boot: int = 2000) -> tuple:
    """同じ日付ブロックで再抽出して、b - a の区間を出す。"""
    sa, sb = select(a, mode), select(b, mode)
    rng = np.random.default_rng(lab.SEED)
    days = pd.unique(a["Date"])
    ia = {d: g["ref_end"].to_numpy(dtype=float) for d, g in sa.groupby("Date")}
    ib = {d: g["ref_end"].to_numpy(dtype=float) for d, g in sb.groupby("Date")}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(days, len(days), replace=True)
        va = np.concatenate([ia[d] for d in pick if d in ia]) if len(pick) else np.array([])
        vb = np.concatenate([ib[d] for d in pick if d in ib]) if len(pick) else np.array([])
        draws[i] = (np.nanmean(vb) if len(vb) else np.nan) \
            - (np.nanmean(va) if len(va) else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo) * 100, float(hi) * 100, float(np.nanmean(draws > 0))


def main() -> int:
    df = lab.frame()
    cols = F.columns("all")
    market = [c for c in cols if F.group_of(c) == "market"]
    stock = [c for c in cols if c not in set(market)]

    oofs = {
        "base(151)": get_oof(df, "base", cols),
        "stock_only(140)": get_oof(df, "stock_only", stock),
        "market_only(11)": get_oof(df, "market_only", market),
    }

    for mode, title in (("pooled", "現行：全期間のスコアをそのまま並べる"),
                        ("per_fold", "窓ごとに順位化してから並べる"),
                        ("within_day", "日付ごとに順位化してから並べる")):
        print()
        print("=" * 76)
        print(f"[{mode}] {title}")
        print("=" * 76)
        print(f"  {'モデル':<20}{'件数':>7}{'日数':>7}{'実収益':>10}{'対母集団':>10}"
              f"{'正例率':>8}{'勝率':>8}")
        for name, o in oofs.items():
            s = stats(o, select(o, mode))
            print(f"  {name:<20}{s['n']:>7,}{s['n_days']:>7,}{s['end']:>+9.2f}%"
                  f"{s['lift']:>+9.2f}pt{s['pos']:>7.1f}%{s['win']:>7.1f}%")
        print()
        a = oofs["base(151)"]
        for name in ("stock_only(140)", "market_only(11)"):
            lo, hi, p = ci_diff(a, oofs[name], mode)
            d = stats(oofs[name], select(oofs[name], mode))["end"] - \
                stats(a, select(a, mode))["end"]
            print(f"    {name:<20} base との差 {d:+.2f}pt [{lo:+.2f}, {hi:+.2f}] "
                  f"改善確率 {p*100:>3.0f}%")

    print()
    print("=" * 76)
    print("選ばれた行が、どの窓に偏っているか")
    print("=" * 76)
    for name, o in oofs.items():
        print(f"  [{name}]")
        for mode in ("pooled", "per_fold", "within_day"):
            c = select(o, mode)["fold"].value_counts().sort_index()
            share = (c / c.sum() * 100).round(0).astype(int)
            print(f"    {mode:<11} " + " ".join(f"窓{k}:{v:>2}%" for k, v in share.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
