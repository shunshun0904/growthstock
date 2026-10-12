#!/usr/bin/env python3
"""
実験84: 6本とも95以上（1.2σ・1.5σ）の中で、6本の平均の最上位2件を月に2銘柄選んだらどうなるか。収益の分位点と、6本の平均の
分位ごとの成績も。

運用者の問い（2026-10-12）「さらに六本の平均で最上位の2銘柄した場合はどうなりますか？ただ最上位の場合の分位点も見たいです」
「月に２銘柄です。それと現在の売りロジックも+10%指値です」。

選び方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  A15  = 1.2σ・1.5σ の lgbm・xgb・cat の6本とも 95以上（実験82・83 と同じ）。R0 = 1.2σ の3本が95以上（今の目安）
  avg6 = その6本の百分位の平均（丸める前の値で並べ、同点は lgbm のスコア順）
  出口: +10% の指値・届かなければ 20営業日目の終値（実験83 の tp10c。運用者の今の売り）。参考に +20%・5日平均と指値なし
  A. 後から見た月の上位2件（月末にならないと決まらない。実験79・80 の「月に2銘柄」と同じ作り）
       主: A15 の中で avg6 の上位2件
       比べる: 今のやり方 = R0 の中で今の並べ方（1.2σ の3本の最小。同点は lgbm のスコア順）の上位2件 / A15 の中で今の並べ方の
              上位2件 / R0 の中で avg6 の上位2件 / A15 から無作為に月2件（2,000回）
       判定: 主 − 今のやり方 の窓ごとの差の平均が正・上の窓が過半、かつ 主 が無作為に月2件より上（p < 0.05）
  B. その日のうちに決められる形（先着2件）
       主: A15 で avg6 が線 T 以上の候補を、その月の先着2件（同じ日は avg6 の高い順）。T は窓ごとに、訓練（その窓より前）で
           A15 の行が月に平均2件になる高さ（訓練の分布だけで決める）
       比べる: 今のやり方 = R0 の先着2件（同じ日は今の並べ方の順） / A15 の先着2件（線なし）
       判定: 主 − 今のやり方 の窓ごとの差の平均が正・上の窓が過半
  分位点: 選んだ取引の収益の 10 / 25 / 50 / 75 / 90% 点。選んだ行・線の avg6 の水準（同じ窓の候補全体の中で上位何%か）
  avg6 の四分位: 窓ごとに A15 の行を avg6 の順位で4つに分け、成績を比べる（上の四分位 − 下の四分位、窓ごと。見るだけ）
  スコアの下限（運用者「どの程度のスコアの下限なのか知りたい」）: 各月の A15 の6本の平均（画面の丸め）の1番目と2番目
       （2番目 = その月の上位2件に入る下限）の月ごとの散らばり。前向きと、画面と同じ全期間の百分位の両方。先着2件の線
       （前向きは窓ごと、画面の物差しでは全期間で A15 が月平均2件になる高さ）と、線の高さごとの月あたりの件数・成績（見るだけ。
       線はここから選ばない）
  百分位は前向き（実験79 の window_split、丸めは画面と同じ）。テスト窓 3〜11。参考に全期間の百分位（A だけ）

公開ログには集計だけ（群ごとの件数・平均・分位点、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e84_six_mean_top2.py
    args="--draws 2000"       （既定）

結果（2026-10-12、run 38204561654、a42f9a2、54秒。docs/MODEL_ADOPTION_RULES.md §41。出口はどれも +10%・20日目の終値）
------------------------------------------------------------------
  A. 後から見た月の上位2件: A15 で6本の平均 65件（39か月）+2.30%（中央 +5.41・10% 点 −8.88）、今のやり方（R0 で3本の最小）
     74件 +2.00%（中央 +3.54・10% 点 −8.62）。差 +0.15 ± 0.76pt（9窓中4）、無作為に月2件 +2.01%（p = 0.36）→ 通らない
  B. 先着2件: A15 で線以上 56件（34か月）+0.51%（中央 +2.65・10% 点 −10.38・勝率 57.1%）、今のやり方（R0 の先着2件）
     74件 −0.26%（中央 +2.55・10% 点 −13.24・−10%割れ 17.6%）。差 +0.47 ± 0.98pt（9窓中5）→ 決めた条件は通るが差はぶれの内
  月2件に絞ると、A15 全体（333件 +3.59%）より1取引の平均がはっきり下がる（候補の多い時期の良い取引を取りこぼす）
  四分位: A15 の中で6本の平均が高いほど良いとは言えない（上 − 下 +0.10 ± 3.87pt）
  スコアの下限（各月の A15 の2番目）: 前向き 中央 99.0（10〜90% 点 97.7〜99.8）、画面の物差し 中央 98.8（98.0〜99.7）。
     画面の物差しで全期間に月平均2件になる線は 98.3（候補全体の上位 0.82%）。ただし A15 が1件も無い月が 65か月中34
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

MODELS = E79.MODELS
SIX = E82.TREES + E82.T15                      # lgbm, xgb, cat, lgbm_k15, xgb_k15, cat_k15
FLAG = "A15"
BASE = "R0"
MAIN = "tp10c"
REF_EXITS = ("tp20a", "c20")
TOP_N = 2
DRAWS = 2000
QS = (10, 25, 50, 75, 90)
SEED = 42
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


# --------------------------------------------------------------------------- #
# 6本の平均
# --------------------------------------------------------------------------- #

def avg6_of(P: np.ndarray, models=MODELS) -> np.ndarray:
    idx = [list(models).index(a) for a in SIX]
    return np.asarray(P, dtype=float)[:, idx].mean(axis=1)


def build(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """実験83 の行に、6本の平均 avg6（丸める前）・avg6r（画面と同じ丸めの値の平均）と、同じ窓の候補全体の中での上位何%（avg6_top）を足す。"""
    ft = E83.build(df, parts)
    idx = [list(models).index(a) for a in SIX]
    ft["avg6"] = np.concatenate([avg6_of(P, models) for _, P in parts])
    ft["avg6r"] = np.concatenate([E82.rounded(np.asarray(P, dtype=float)[:, idx]).mean(axis=1) for _, P in parts])
    # 同じ窓の候補全体の中で、avg6 がそれより高い行の割合（%）。0 に近いほど最上位
    ft["avg6_top"] = ft.groupby("window")["avg6"].rank(ascending=False, method="min").sub(1).div(
        ft.groupby("window")["avg6"].transform("size")).mul(100)
    return ft


def qtiles(x) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(x):
        return {f"p{q}": float("nan") for q in QS} | {"n": 0, "mean": float("nan"), "min": float("nan"),
                                                      "max": float("nan")}
    out = {f"p{q}": float(np.percentile(x, q)) for q in QS}
    out.update(n=int(len(x)), mean=float(x.mean()), min=float(x.min()), max=float(x.max()))
    return out


def fmt_q(q: dict) -> str:
    return "  ".join(f"{q_}% {q[f'p{q_}']:+6.2f}" for q_ in QS) + f"  最悪 {q['min']:+.1f}%"


# --------------------------------------------------------------------------- #
# 選び方
# --------------------------------------------------------------------------- #

def months_of(x: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(x["Date"]).dt.to_period("M")


def monthly_by(ft: pd.DataFrame, flag: str, key: str, n: int = TOP_N) -> pd.DataFrame:
    """後から見た月の上位: 各月、flag を満たす行を key の高い順（同点は lgbm のスコア順）に上位 n 件。"""
    sub = ft[ft[flag].to_numpy(dtype=bool)].assign(_m=months_of)
    sub = sub.sort_values(["_m", key, "score"], ascending=[True, False, False], kind="mergesort")
    return sub.groupby("_m", sort=False).head(n).reset_index(drop=True)


def first_by(ft: pd.DataFrame, mask: np.ndarray, key: str, n: int = TOP_N) -> pd.DataFrame:
    """先着: 各月、mask の行を日付の早い順（同じ日は key の高い順、同点は lgbm のスコア順）に n 件。その日のうちに決められる。"""
    sub = ft[mask].assign(_m=months_of)
    sub = sub.sort_values(["_m", "Date", key, "score"], ascending=[True, True, False, False], kind="mergesort")
    return sub.groupby("_m", sort=False).head(n).reset_index(drop=True)


def random_monthly(ft: pd.DataFrame, flag: str, col: str, n: int = TOP_N, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """各月、flag を満たす行から無作為に n 件（足りなければ全部）選んだときの、選んだ行の col の平均（draws 回）。"""
    rng = np.random.default_rng(seed)
    sub = ft[ft[flag].to_numpy(dtype=bool)]
    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    months = months_of(sub).to_numpy()
    groups = [np.where(months == m)[0] for m in pd.unique(months)]
    out = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([g if len(g) <= n else rng.choice(g, n, replace=False) for g in groups])
        out[i] = np.nanmean(vals[pick])
    return out


def train_thresholds(df: pd.DataFrame, split: dict, models=MODELS, flag: str = FLAG, per_month: float = TOP_N) -> dict:
    """
    テスト窓ごとに、訓練の行（その窓より前。百分位も訓練の分布で付けたもの）で flag の行が月に平均 per_month 件になる
    avg6 の線 T。{窓: {T, 訓練の月数, 訓練の flag の件数, 訓練の候補全体で T 以上の割合（%）}}。
    """
    out = {}
    for w, (tr, te, Ptr, Pte) in split.items():
        fl = E82.flags_of(E82.rounded(Ptr), models)[flag]
        a6 = avg6_of(Ptr, models)
        months = int(pd.to_datetime(df.iloc[tr]["Date"]).dt.to_period("M").nunique())
        k = int(round(per_month * months))
        cand = np.sort(a6[fl])[::-1]
        if not len(cand) or k < 1:
            T = float("inf")
        else:
            T = float(cand[min(k, len(cand)) - 1])
        out[int(w)] = {"T": T, "months": months, "n_flag": int(fl.sum()), "k": k,
                       "share_all": float((a6 >= T).mean() * 100) if np.isfinite(T) else 0.0}
    return out


def threshold_mask(ft: pd.DataFrame, thr: dict, flag: str = FLAG) -> np.ndarray:
    T = ft["window"].map(lambda w: thr.get(int(w), {}).get("T", float("inf"))).to_numpy(dtype=float)
    return ft[flag].to_numpy(dtype=bool) & (ft["avg6"].to_numpy(dtype=float) >= T)


def window_diff(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    ma, mb = a.groupby("window")[col].mean(), b.groupby("window")[col].mean()
    j = ma.index.intersection(mb.index)
    return summarize((ma[j] - mb[j]).to_numpy(dtype=float))


def verdict(d: dict, p: float | None = None) -> tuple[bool, str]:
    need = d["n"] // 2 + 1
    pos = bool(d["n"] and d["mean"] > 0)
    maj = bool(d["n"] and d["wins"] >= need)
    ok = pos and maj and (p is None or p < 0.05)
    txt = f"差の平均が正 {'○' if pos else '×'} / 上の窓が過半 {'○' if maj else '×'}（{need}窓以上）"
    if p is not None:
        txt += f" / 無作為より上 p < 0.05 {'○' if p < 0.05 else '×'}"
    return ok, txt


def group_line(name: str, g: pd.DataFrame, months: int, width: int = 38) -> str:
    r = pd.to_numeric(g[MAIN], errors="coerce").dropna()
    hit = g.loc[r.index, "hit10"].astype(bool).mean() * 100 if len(r) else float("nan")
    refs = "  ".join(f"{E83.EXIT_JA[c].split('（')[0]} {pd.to_numeric(g[c], errors='coerce').mean():+.2f}%"
                     for c in REF_EXITS)
    return (f"  {LT._l(name, width)}{len(g):>5,}件（月 {len(g) / months:4.2f}件）  +10% {r.mean():+6.2f}%（中央 {r.median():+5.2f}）"
            f"  勝率 {(r > 0).mean() * 100:4.1f}%  到達 {hit:4.1f}%  −10%割れ {(r < -10).mean() * 100:4.1f}%  ｜ 参考 {refs}")


CUT_GRID = (97.0, 97.5, 98.0, 98.5, 99.0, 99.5)


def month_cutoffs(fr: pd.DataFrame, flag: str = FLAG, col: str = "avg6r") -> pd.DataFrame:
    """各月の flag の行の col の1番目・2番目（2番目 = その月の上位2件に入る下限）と件数。flag の行が無い月は出さない。"""
    sub = fr[fr[flag].to_numpy(dtype=bool)].assign(_m=months_of)
    rows = []
    for m, g in sub.groupby("_m"):
        v = np.sort(pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float))[::-1]
        if not len(v):
            continue
        rows.append({"_m": m, "window": int(g["window"].iloc[0]), "n": int(len(v)), "first": float(v[0]),
                     "second": float(v[1]) if len(v) > 1 else float("nan")})
    return pd.DataFrame(rows, columns=["_m", "window", "n", "first", "second"])


def full_threshold(fr: pd.DataFrame, flag: str = FLAG, col: str = "avg6r", per_month: float = TOP_N) -> dict:
    """fr の全期間で flag の行が月に平均 per_month 件になる col の線（画面の物差しの線に使う）と、候補全体で線以上の割合（%）。"""
    months = E82.n_months(fr)
    v = np.sort(pd.to_numeric(fr.loc[fr[flag].to_numpy(dtype=bool), col], errors="coerce").dropna().to_numpy())[::-1]
    k = int(round(per_month * months))
    if not len(v) or k < 1:
        return {"T": float("inf"), "months": months, "k": k, "share_all": 0.0}
    T = float(v[min(k, len(v)) - 1])
    return {"T": T, "months": months, "k": k, "n_flag": int(len(v)),
            "share_all": float((pd.to_numeric(fr[col], errors="coerce") >= T).mean() * 100)}


def grid_rows(fr: pd.DataFrame, months: int, grid=CUT_GRID, flag: str = FLAG, col: str = "avg6r") -> list:
    """線（col）の高さごとに、flag かつ線以上の行の月あたりの件数・1件以上ある月の割合・先着2件の +10% の成績。"""
    out = []
    for t in grid:
        mask = fr[flag].to_numpy(dtype=bool) & (pd.to_numeric(fr[col], errors="coerce").to_numpy(dtype=float) >= t)
        picks = first_by(fr, mask, "avg6")
        r = pd.to_numeric(picks[MAIN], errors="coerce").dropna()
        out.append({"t": t, "per_month": float(mask.sum() / months) if months else float("nan"),
                    "months_any": int(picks["_m"].nunique()) if len(picks) else 0, "n_picks": int(len(picks)),
                    "mean": float(r.mean()) if len(r) else float("nan"), "med": float(r.median()) if len(r) else float("nan"),
                    "win": float((r > 0).mean() * 100) if len(r) else float("nan"),
                    "p10": float(np.percentile(r, 10)) if len(r) else float("nan")})
    return out


def print_cutoffs(fr: pd.DataFrame, label: str) -> dict:
    months = E82.n_months(fr)
    mc = month_cutoffs(fr)
    two = mc.dropna(subset=["second"])
    q1, q2 = qtiles(mc["first"]), qtiles(two["second"])
    n0 = months - len(mc)
    n1 = int((mc["n"] == 1).sum())
    print(f"  {label}: {months}か月のうち A15 が2件以上の月 {len(two)}・1件の月 {n1}・0件の月 {n0}")
    print(f"    各月の1番目（最上位）: " + "  ".join(f"{q_}% {q1[f'p{q_}']:.1f}" for q_ in QS)
          + f"  最小 {q1['min']:.1f}・最大 {q1['max']:.1f}")
    print(f"    各月の2番目（上位2件に入る下限）: " + "  ".join(f"{q_}% {q2[f'p{q_}']:.1f}" for q_ in QS)
          + f"  最小 {q2['min']:.1f}・最大 {q2['max']:.1f}")
    byw = two.groupby("window")["second"].median()
    print("    窓ごとの2番目の中央: " + "  ".join(f"窓{int(w)} {v:.1f}" for w, v in byw.items()))
    return {"months": months, "two": int(len(two)), "one": n1, "zero": int(n0), "first": q1, "second": q2,
            "second_by_window": {int(w): float(v) for w, v in byw.items()}}


def quartiles_in_window(ft: pd.DataFrame, flag: str, key: str = "avg6", min_rows: int = 8) -> pd.Series:
    """窓ごとに flag の行を key の順位で4つ（0 = 下 〜 3 = 上）に分ける。行が少ない窓は NaN。"""
    out = pd.Series(np.nan, index=ft.index)
    sub = ft[ft[flag].to_numpy(dtype=bool)]
    for _, g in sub.groupby("window"):
        if len(g) < min_rows:
            continue
        out.loc[g.index] = pd.qcut(g[key].rank(method="first"), 4, labels=False)
    return out


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def report(ft: pd.DataFrame, thr: dict | None = None, ff: pd.DataFrame | None = None, draws: int = DRAWS,
           seed: int = SEED) -> dict:
    summary: dict = {}
    months = E82.n_months(ft)
    a15 = ft[ft[FLAG].to_numpy(dtype=bool)]
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月。A15（6本とも95以上）{len(a15):,}行・R0（1.2σ の3本）"
          f"{int(ft[BASE].sum()):,}行。出口はどれも +10% の指値・届かなければ20日目の終値（運用者の今の売り）")
    print(f"  A15 の6本の平均（画面の丸め）: 10% 点 {np.percentile(a15['avg6r'], 10):.1f}・中央 {np.median(a15['avg6r']):.1f}・"
          f"90% 点 {np.percentile(a15['avg6r'], 90):.1f}")
    summary["a15_avg6r"] = qtiles(a15["avg6r"])

    # ---- A. 後から見た月の上位2件
    print(f"\n■ 1. 後から見た月の上位{TOP_N}件（月末にならないと決まらない）")
    sel = {"top6": ("A15 で6本の平均の上位2件（主）", monthly_by(ft, FLAG, "avg6")),
           "cur": ("今のやり方: R0 で3本の最小の上位2件", monthly_by(ft, BASE, "pmin3")),
           "a15m": ("A15 で3本の最小の上位2件", monthly_by(ft, FLAG, "pmin3")),
           "r06": ("R0 で6本の平均の上位2件", monthly_by(ft, BASE, "avg6"))}
    print(group_line("A15 全体（参考）", a15, months))
    for k, (name, g) in sel.items():
        print(group_line(name, g, months))
    top6, cur = sel["top6"][1], sel["cur"][1]
    rnd = random_monthly(ft, FLAG, MAIN, draws=draws, seed=seed)
    m6 = float(pd.to_numeric(top6[MAIN], errors="coerce").mean())
    p_rnd = float((np.sum(rnd >= m6) + 1) / (len(rnd) + 1))
    print(f"  A15 から無作為に月{TOP_N}件（{draws}回）: 平均 {rnd.mean():+.2f}%（5〜95% {np.percentile(rnd, 5):+.2f} 〜 "
          f"{np.percentile(rnd, 95):+.2f}%）。主がそれ以上になる割合 p = {p_rnd:.3f}")
    d_cur = window_diff(top6, cur, MAIN)
    ok_a, txt_a = verdict(d_cur, p_rnd)
    print(f"  窓ごとの差: 主 − 今のやり方 {fmt_s(d_cur)}、主 − A15 で3本の最小の上位2件 "
          f"{fmt_s(window_diff(top6, sel['a15m'][1], MAIN))}、R0 で6本の平均 − 今のやり方 "
          f"{fmt_s(window_diff(sel['r06'][1], cur, MAIN))}")
    print(f"  判定: {'通る' if ok_a else '通らない'}（{txt_a}）")
    print("  収益の分位点（+10% の指値・20日目の終値。上の方は +10% で止まる）")
    for k, (name, g) in sel.items():
        print(f"    {LT._l(name, 38)}{fmt_q(qtiles(g[MAIN]))}")
    print(f"    {LT._l('A15 全体（参考）', 38)}{fmt_q(qtiles(a15[MAIN]))}")
    print("  収益の分位点（参考: 指値なし・20日目の終値。値動きそのものの散らばり）")
    for k in ("top6", "cur"):
        print(f"    {LT._l(sel[k][0], 38)}{fmt_q(qtiles(sel[k][1]['c20']))}")
    lv, tp = qtiles(top6["avg6r"]), qtiles(top6["avg6_top"])
    print(f"  主で選んだ行の6本の平均（画面の丸め）: 10% 点 {lv['p10']:.1f}・中央 {lv['p50']:.1f}・90% 点 {lv['p90']:.1f}。"
          f"同じ窓の候補全体の中で上位 {tp['p50']:.2f}%（中央。10〜90% 点 {tp['p10']:.2f}〜{tp['p90']:.2f}%）")
    summary["A"] = {k: {"n": int(len(g)), "q": qtiles(g[MAIN]),
                        "ref": {c: float(pd.to_numeric(g[c], errors="coerce").mean()) for c in REF_EXITS}}
                    for k, (_, g) in sel.items()}
    summary["A"].update(random={"mean": float(rnd.mean()), "p": p_rnd}, vs_cur=d_cur, passed=ok_a, level=lv, top_pct=tp)

    # ---- B. その日のうちに決められる形（先着2件）
    if thr:
        print(f"\n■ 2. その日のうちに決められる形（各月の先着{TOP_N}件。線は訓練の期間で A15 が月{TOP_N}件になる高さ）")
        Ts = np.array([v["T"] for v in thr.values() if np.isfinite(v["T"])])
        sh = np.array([v["share_all"] for v in thr.values() if np.isfinite(v["T"])])
        print(f"  線（6本の平均、丸める前）: 窓ごとに {Ts.min():.2f}〜{Ts.max():.2f}（中央 {np.median(Ts):.2f}）。訓練の候補全体で"
              f"線以上は上位 {np.median(sh):.2f}%（中央。{sh.min():.2f}〜{sh.max():.2f}%）")
        mth = threshold_mask(ft, thr)
        per_w = ft.assign(_ok=mth, _m=months_of).groupby("window").agg(n=("_ok", "sum"), m=("_m", "nunique"))
        print("  窓ごとの線と、テストで線以上の A15 の月あたりの件数: " + "  ".join(
            f"窓{int(w)} {thr[int(w)]['T']:.2f}（{per_w.loc[w, 'n'] / per_w.loc[w, 'm']:.1f}件）" for w in per_w.index
            if int(w) in thr))
        selb = {"thr": ("A15 で線以上の先着2件（主）", first_by(ft, threshold_mask(ft, thr), "avg6")),
                "cur": ("今のやり方: R0 の先着2件", first_by(ft, ft[BASE].to_numpy(dtype=bool), "pmin3")),
                "a15": ("A15 の先着2件（線なし）", first_by(ft, ft[FLAG].to_numpy(dtype=bool), "avg6"))}
        for k, (name, g) in selb.items():
            print(group_line(name, g, months))
            nm = g["_m"].nunique() if len(g) else 0
            print(f"      買った月 {nm}/{months}か月  分位点 {fmt_q(qtiles(g[MAIN]))}")
        d_b = window_diff(selb["thr"][1], selb["cur"][1], MAIN)
        ok_b, txt_b = verdict(d_b)
        print(f"  窓ごとの差: 主 − 今のやり方 {fmt_s(d_b)}、A15 の先着2件 − 今のやり方 "
              f"{fmt_s(window_diff(selb['a15'][1], selb['cur'][1], MAIN))}")
        print(f"  判定: {'通る' if ok_b else '通らない'}（{txt_b}）")
        summary["B"] = {k: {"n": int(len(g)), "months": int(g["_m"].nunique()) if len(g) else 0, "q": qtiles(g[MAIN])}
                        for k, (_, g) in selb.items()}
        summary["B"].update(thresholds=thr, vs_cur=d_b, passed=ok_b)

    # ---- 3. avg6 の四分位
    print("\n■ 3. A15 の中の6本の平均の四分位（窓ごとに順位で4つ。見るだけ）")
    qt = quartiles_in_window(ft, FLAG)
    summary["quartiles"] = {}
    names = ("下（0〜25%）", "25〜50%", "50〜75%", "上（75〜100%）")
    for k in range(4):
        g = ft[qt == k]
        if not len(g):
            continue
        q = qtiles(g[MAIN])
        summary["quartiles"][k] = q | {"avg6r_med": float(np.median(g["avg6r"]))}
        print(group_line(f"{names[k]}（6本の平均 中央 {np.median(g['avg6r']):.1f}）", g, months, width=36))
        print(f"      分位点 {fmt_q(q)}")
    dq = window_diff(ft[qt == 3], ft[qt == 0], MAIN)
    summary["quartiles"]["top_minus_low"] = dq
    print(f"  上の四分位 − 下の四分位（窓ごと）: {fmt_s(dq)}")

    # ---- 5（先に作る）. スコアの下限
    cut = {"forward": None, "full": None}

    # ---- 4. 全期間の百分位（参考）
    if ff is not None and len(ff):
        mf = E82.n_months(ff)
        print("\n■ 4. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・後から見た月の上位2件")
        t6f, tcf = monthly_by(ff, FLAG, "avg6"), monthly_by(ff, BASE, "pmin3")
        print(group_line("A15 で6本の平均の上位2件", t6f, mf))
        print(group_line("今のやり方: R0 で3本の最小の上位2件", tcf, mf))
        print(f"    分位点（主）{fmt_q(qtiles(t6f[MAIN]))}")
        summary["full"] = {"top6": qtiles(t6f[MAIN]), "cur": qtiles(tcf[MAIN]), "vs_cur": window_diff(t6f, tcf, MAIN)}
        print(f"    窓ごとの差（主 − 今のやり方）{fmt_s(summary['full']['vs_cur'])}")

    print("\n■ 5. スコアの下限（6本の平均。画面の丸めの値の平均）")
    print("  後から見た月の上位2件に入る下限 = 各月の A15 の2番目の値")
    cut["forward"] = print_cutoffs(ft, "前向きの百分位")
    if ff is not None and len(ff):
        cut["full"] = print_cutoffs(ff, "画面と同じ全期間の百分位（窓 1〜11）")
        tf = full_threshold(ff)
        mf = E82.n_months(ff)
        print(f"  先着2件の線を画面の物差しで1本にすると（全期間で A15 が月平均2件になる高さ）: 6本の平均 {tf['T']:.1f} 以上"
              f"（候補全体の上位 {tf['share_all']:.2f}%）")
        cut["full_T"] = tf
        print("  線の高さごとの、A15 かつ線以上の月あたりの件数と先着2件の +10% の成績（画面の物差し・全期間。見るだけ。線はここから選ばない）")
        rows = grid_rows(ff, mf)
        cut["grid_full"] = rows
        for r in rows:
            print(f"    {r['t']:.1f} 以上: 月 {r['per_month']:4.2f}件・買える月 {r['months_any']}/{mf}・先着2件 {r['n_picks']}件  "
                  f"1取引 {r['mean']:+6.2f}%（中央 {r['med']:+5.2f}・10% 点 {r['p10']:+6.2f}）  勝率 {r['win']:4.1f}%")
    print("  線の高さごとの、同じ表（前向きの百分位。見るだけ）")
    rows_f = grid_rows(ft, months)
    cut["grid_forward"] = rows_f
    for r in rows_f:
        print(f"    {r['t']:.1f} 以上: 月 {r['per_month']:4.2f}件・買える月 {r['months_any']}/{months}・先着2件 {r['n_picks']}件  "
              f"1取引 {r['mean']:+6.2f}%（中央 {r['med']:+5.2f}・10% 点 {r['p10']:+6.2f}）  勝率 {r['win']:4.1f}%")
    summary["cutoff"] = cut

    print("\n■ まとめ（出口はどれも +10% の指値・20日目の終値）")
    A = summary["A"]
    print(f"  A. 後から見た月の上位2件: A15 で6本の平均 {A['top6']['q']['mean']:+.2f}%（中央 {A['top6']['q']['p50']:+.2f}・"
          f"10% 点 {A['top6']['q']['p10']:+.2f}・{A['top6']['n']}件）、今のやり方 {A['cur']['q']['mean']:+.2f}%（中央 "
          f"{A['cur']['q']['p50']:+.2f}・10% 点 {A['cur']['q']['p10']:+.2f}）。差 {fmt_s(A['vs_cur'])}、無作為より上 "
          f"p = {A['random']['p']:.3f} → {'通る' if A['passed'] else '通らない'}")
    if "B" in summary:
        B = summary["B"]
        print(f"  B. 先着2件: A15 で線以上 {B['thr']['q']['mean']:+.2f}%（{B['thr']['n']}件・{B['thr']['months']}か月）、"
              f"今のやり方 {B['cur']['q']['mean']:+.2f}%（{B['cur']['n']}件）。差 {fmt_s(B['vs_cur'])} → "
              f"{'通る' if B['passed'] else '通らない'}")
    print(f"  四分位: 上 − 下 {fmt_s(summary['quartiles']['top_minus_low'])}")
    c = summary["cutoff"]
    for key, name in (("forward", "前向き"), ("full", "画面の物差し")):
        if c.get(key):
            q2 = c[key]["second"]
            print(f"  上位2件に入る下限（{name}）: 中央 {q2['p50']:.1f}（10〜90% 点 {q2['p10']:.1f}〜{q2['p90']:.1f}）")
    if c.get("full_T"):
        print(f"  先着2件の線（画面の物差し、月平均2件）: {c['full_T']['T']:.1f} 以上")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験84: 6本とも95以上の中で6本の平均の上位2件（月に2銘柄）")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件（モデルごとの件数 {sorted(set(info0['rows'].values()))}）")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=(10.0, 20.0))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・5日平均と ret_o1_20 の一致 {chk['hold5_match']:,}/"
        f"{chk['hold5_both']:,}・到達の順序の食い違い {chk['order_violations']}件")
    split = E79.window_split(df, folds)
    ft = build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    thr = train_thresholds(df, split)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    ff = build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    summary = report(ft, thr=thr, ff=ff, draws=args.draws)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e84_summary.json")

    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=conv)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
