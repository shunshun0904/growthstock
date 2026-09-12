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
    """
    3通りの選び方。返すのは選ばれた行。

    同点は必ず乱数で割る。スコアやその順位が同じ行を nlargest に渡すと
    行順（=日付順）で切れてしまい、「古い日だけ」を測ることになる。
    実際 within_day を大域しきい値で切ったとき、選ばれた行が窓1〜7 に
    限られ、窓8〜10 が丸ごと欠けていた。
    """
    rng = np.random.default_rng(lab.SEED)
    o = oof.assign(_tb=rng.random(len(oof)))

    if mode == "within_day":
        # 日付内順位は1位が全部 1.0 で並ぶので大域のしきい値では切れない。
        # 各日の1位を取る。運用（毎晩1件買う）と同じ形であり、
        # 日数が固定されるのでタイミングの力が消え、銘柄選定だけが残る。
        return (o.sort_values(["score", "_tb"])
                 .groupby("Date", sort=False).tail(1))

    n = max(1, int(len(oof) * k / 100))
    if mode == "pooled":
        o["_k"] = o["score"]
    elif mode == "per_fold":
        o["_k"] = o.groupby("fold")["score"].rank(pct=True)
    else:
        raise ValueError(mode)
    return o.sort_values(["_k", "_tb"], ascending=False).head(n)


def benchmark(oof: pd.DataFrame, mode: str) -> float:
    """
    その選び方に対して「腕がない場合」の水準。

    pooled / per_fold は全行から選ぶので、母集団の平均が比較相手。
    within_day は毎日1件選ぶので、比較相手は「毎日でたらめに1件選ぶ」
    = 日ごとの平均を日で等重み平均したもの。母集団の平均と比べると、
    候補数の多い日に重みが寄ってしまい比較にならない。
    """
    if mode == "within_day":
        return float(oof.groupby("Date")["ref_end"].mean().mean())
    return float(pd.to_numeric(oof["ref_end"], errors="coerce").mean())


def stats(oof: pd.DataFrame, sel: pd.DataFrame, mode: str) -> dict:
    e = pd.to_numeric(sel["ref_end"], errors="coerce")
    base = benchmark(oof, mode)
    return {"n": len(sel), "end": float(e.mean()) * 100,
            "lift": float(e.mean() - base) * 100,
            "pos": float(sel["label"].mean()) * 100,
            "win": float((e > 0).mean()) * 100,
            "n_days": int(sel["Date"].nunique())}


def _buckets(sel: pd.DataFrame, days: pd.Index) -> list:
    """
    選ばれた行を日付ごとの配列に振り分ける。

    日付は整数コードで扱う。pd.unique(Series) は numpy.datetime64 を返すのに
    groupby のキーは pandas.Timestamp で、両者はハッシュが一致しないことがある。
    dict 引きにすると全部外れて静かに空になる（実際それで落ちた）ので、
    pd.Index.get_indexer で位置に直し、型に依存しない形にする。
    """
    out = [[] for _ in range(len(days))]
    if len(sel):
        idx = days.get_indexer(pd.Index(sel["Date"]))
        v = pd.to_numeric(sel["ref_end"], errors="coerce").to_numpy(dtype=float)
        for i, j in enumerate(idx):
            if j >= 0:
                out[j].append(v[i])
    return [np.asarray(x, dtype=float) for x in out]


def ci_diff(a: pd.DataFrame, b: pd.DataFrame, mode: str, n_boot: int = 2000) -> tuple:
    """
    同じ日付ブロックで再抽出して、b - a の区間を出す。

    日付単位で振り直すのは、同じ日の銘柄が地合いを共有するため。
    行単位で振ると独立標本を仮定することになり区間が不当に狭くなる。
    両モデルで同じ日の並びを使うので、差は対になっている。
    """
    days = pd.Index(pd.unique(a["Date"]))
    ba = _buckets(select(a, mode), days)
    bb = _buckets(select(b, mode), days)
    rng = np.random.default_rng(lab.SEED)
    draws = np.full(n_boot, np.nan)
    for i in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        va = [ba[j] for j in pick if ba[j].size]
        vb = [bb[j] for j in pick if bb[j].size]
        if not va or not vb:
            continue
        draws[i] = np.nanmean(np.concatenate(vb)) - np.nanmean(np.concatenate(va))
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
            s = stats(o, select(o, mode), mode)
            print(f"  {name:<20}{s['n']:>7,}{s['n_days']:>7,}{s['end']:>+9.2f}%"
                  f"{s['lift']:>+9.2f}pt{s['pos']:>7.1f}%{s['win']:>7.1f}%")
        print()
        a = oofs["base(151)"]
        for name in ("stock_only(140)", "market_only(11)"):
            lo, hi, p = ci_diff(a, oofs[name], mode)
            d = stats(oofs[name], select(oofs[name], mode), mode)["end"] - \
                stats(a, select(a, mode), mode)["end"]
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
