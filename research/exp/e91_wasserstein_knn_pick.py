#!/usr/bin/env python3
"""
実験91: 今の規則（1.5σ の3本がすべて95以上）を満たした候補が同じ日に2件以上あるとき、ワッサーステイン距離の k 近傍で
1件を選ぶと、今の並べ方・無作為に選ぶより良いか（今のモデルの後工程の検証。本番は変えない）。

運用者の案（2026-10-14）「新高値ブレイクでの期待値の推定にワッサーステイン距離を使えないか」。後工程として足す考え
「今のモデルのあと工程として、追加するつもりでしたので、私もそのイメージでした。今の選定基準の条件満たした上で最後の2択の
ようなイメージです」。定義は運用者が選んだ（AskUserQuestion）: 2択の中身 = 同じ日の候補から選ぶ / 似た過去の事例を探す範囲 =
惜しい帯から上 / 比べる分布 = 60日のリターンの形と出来高の比率の分布 / 進め方 = 台本を書いて回す。

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  行: 12本の out-of-fold・前向きの百分位（テスト窓 3〜11。実験88 と同じ作り）。出口は +10% の指値・届かなければ 20営業日目の
      終値（tp10c）
  候補: 1.5σ の lgbm_k15・xgb_k15・cat_k15 の百分位（画面と同じ丸め）がすべて95以上の行
  判断日: 候補が同じ日に2件以上ある日（収益と2つの分布が出せる候補だけ数える）
  分布（ブレイクの日を含む直近60営業日。その銘柄の日足だけで作る）
    R = 日次リターン（調整後終値の前日比。vol_20d と同じ作り）を平均0・標準偏差1にそろえたもの（形だけ。位置と尺度は
        モデルが持っている）。60日のうち50日以上の値が要る
    V = log(その日の出来高 ÷ その前の60営業日の平均出来高)。出来高は調整後（無ければ素の値）で、0 の日は使わない。50日以上
    どちらも 60点の分位関数（確率 (j+0.5)/60）で持ち、W1 = 分位関数の差の絶対値の平均（1次元の1次ワッサーステイン距離。
    値が60個どうしならちょうど一致する）
  距離: W1(R)/s_R + W1(V)/s_V。s は窓ごとに、テスト窓の始まりより前に結果が出そろった近傍の母集団から無作為に取った
        2,000組の W1 の中央値（尺度をそろえるだけ。収益は使わない）
  近傍の母集団（惜しい帯から上）: その窓の百分位の物差し（前向きの百分位と同じ訓練の分布）で 1.5σ の3本の最小が85以上の行
        のうち、結果が出そろった行（買った日を1日目として20営業日目 = その銘柄の日足で i+20 本目の日が、判断日より前）
  見積もり: 距離の近い50件の tp10c の平均（k-NN の期待値）。母集団が100件に満たない日は判断日から外す（件数を出す）
  選び方（判断日ごとに1件）
    K（主）: 見積もりが最も高い候補（同点は今の並べ方）
    M（今の並べ方）: 3本の最小が高い順・同点は基準モデルのスコア順（画面の orderRows / strategySignal と同じ）
    R（無作為）: 同じ日の候補から1件を無作為に（2,000回）
  判定（「後工程として効く」= 次の3つがそろう）
    1. K − M の窓ごとの差（窓の中の判断日の差の平均。判断日が3日以上ある窓）の平均が正
    2. 上の窓が過半
    3. K の平均が無作為の平均の分布の上側 5% に入る（片側 p < 0.05。p = 無作為の平均が K 以上になる割合）
    そろわなければ「今の並べ方・無作為と見分けられない」
  参考（判定に使わない）
    - 分布を片方だけにした距離（R だけ・V だけ）、近傍の数 20件・100件
    - 判断日の中の順位相関（候補3件以上の日）: 見積もりと実際の収益（並べ替え p）、見積もりと3本の最小、見積もりと vol_20d
      （見積もりが合意の高さやボラの言い直しになっていないか）
    - 今の並べ方 M と無作為の比較（p）、後から見た最良の1件（上限の目安）、判断日の候補全体の平均（無作為の期待値）
    - 見積もりの水準: 選んだ候補の見積もりの平均と実際の平均、近傍の −10%割れの割合と実際の割合
  出すもの: 件数・平均・中央・勝率・+10% に届く割合・−10%割れ

公開ログには集計だけ（件数・平均・割合・窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e91_wasserstein_knn_pick.py
    args="--draws 2000"       （既定）

結果（2026-10-14、run 38518952184、f413c6f、62秒。docs/MODEL_ADOPTION_RULES.md §51）
------------------------------------------------------------------
  判定: 今の並べ方・無作為と見分けられない。判断日96日（今の規則の行 559件。判断日の候補は中央3件・最大17件。窓5・6・9 に
        72日）。K（リターンの形 + 出来高の比率・近い50件）+3.61%（中央 +6.63・勝率 70.8%・−10%割れ 5.2%）、M（今の並べ方）
        +3.99%（中央 +6.26・71.9%・5.2%）、無作為 +3.49%（5〜95% 点 +2.54〜+4.36）。窓ごとの差 K − M +0.14 ± 0.64pt
        （上の窓 4/7）で条件1・2 は満たすが、無作為より上 p = 0.426 で条件3 を満たさない。全部を足した K − M は −0.38pt
        （判断日の多い窓5・9 が負）
  参考: リターンの形だけ +4.04%（p = 0.163）・出来高の比率だけ +4.22%（p = 0.090）・近い20件 +3.77%（p = 0.316）・
        近い100件 +4.78%（K − M +1.30 ± 0.99pt・p = 0.008。4つ並べた参考の1つを結果を見てから選ぶ形で、近傍の数に
        対して向きもそろわない）
  日の中の順位相関（候補3件以上の58日）: 見積もりと実際の収益 +0.027（両側 p = 0.727）・3本の最小 +0.008・vol_20d +0.051
  M も無作為と見分けられない（p = 0.190）。見積もりの水準は低め（K の見積もり +1.81% 対 実際 +3.61%）
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
import models as M  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

RULE_MODELS = E82.T15                 # lgbm_k15・xgb_k15・cat_k15（今の規則の3本）
LINE = 95.0                           # 今の規則の線
POOL_LINE = 85.0                      # 近傍の母集団: 惜しい帯から上
WIN = 60                              # 分布を取る営業日数（ブレイクの日を含む）
MIN_OBS = 50                          # そのうち値が要る日数
VOL_BASE = 60                         # 出来高の比率の分母（その前の60営業日の平均）
LEVELS = 60                           # 分位関数を取る点の数
HOLD = LT.HOLD                        # 20営業日（結果が出そろう日）
TARGET = 10.0                         # +10% の指値
TP = "tp10c"                          # +10% の指値・届かなければ 20営業日目の終値（%）
HIT = "hit10"
K_MAIN = 50                           # 近傍の数（主）
K_REF = (20, 100)                     # 近傍の数（参考）
MIN_POOL = 2 * K_MAIN                 # 近傍の母集団がこれに満たない日は判断日から外す
N_PAIRS = 2000                        # 尺度をそろえる中央値に使う組の数
MIN_DAYS_WIN = 3                      # 窓ごとの差に数える窓の、判断日の最小
DRAWS = 2000                          # 無作為に選ぶ回数
RHO_PERM = 500                        # 順位相関の並べ替えの回数
SEED = 42
#: (記号, 表示)。RV が主
VARIANTS = (("RV", "リターンの形 + 出来高の比率"), ("R", "リターンの形だけ"), ("V", "出来高の比率だけ"))
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


# --------------------------------------------------------------------------- #
# 分布と W1
# --------------------------------------------------------------------------- #

def probs(levels: int = LEVELS) -> np.ndarray:
    return (np.arange(levels) + 0.5) / levels


def quantile_vector(x, levels: int = LEVELS) -> np.ndarray:
    """経験分布の分位関数を確率 (j+0.5)/levels で取る（値がちょうど levels 個なら、並べ替えた値そのもの）。"""
    xs = np.sort(np.asarray(x, dtype=float))
    idx = np.minimum((probs(levels) * len(xs)).astype(int), len(xs) - 1)
    return xs[idx]


def w1(q, Q) -> np.ndarray:
    """分位関数 q（levels,）と Q（n × levels）の W1（1次元の1次ワッサーステイン距離 = 分位関数の差の絶対値の平均）。"""
    return np.abs(np.asarray(Q, dtype=float) - np.asarray(q, dtype=float)).mean(axis=-1)


def _col(g: pd.DataFrame, main: str, fallback: str) -> np.ndarray:
    """調整後の列を優先し、欠けていれば素の列で埋める（build_dataset と同じ扱い）。"""
    a = pd.to_numeric(g[main], errors="coerce") if main in g.columns else pd.Series(np.nan, index=g.index)
    if fallback in g.columns:
        a = a.fillna(pd.to_numeric(g[fallback], errors="coerce"))
    return a.to_numpy(dtype=float)


def event_distributions(bars: pd.DataFrame, keys: pd.DataFrame, win: int = WIN, base: int = VOL_BASE,
                        min_obs: int = MIN_OBS, levels: int = LEVELS, hold: int = HOLD) -> dict:
    """
    keys の (Code, Date) ごとに、その日を含む直近 win 営業日の
      QR: 日次リターン（調整後終値の前日比）を平均0・標準偏差1にそろえた分布の分位関数（形だけ）
      QV: log(その日の出来高 ÷ その前の base 営業日の平均出来高) の分布の分位関数
    を返す（keys と同じ並び。値が min_obs 日に満たない・標準偏差が0 なら NaN）。
    done: 結果が出そろう日 = その銘柄の日足で i + hold 本目の日（買った日 i+1 を1日目として hold 日目）。無ければ NaT。
    """
    n = len(keys)
    QR = np.full((n, levels), np.nan)
    QV = np.full((n, levels), np.nan)
    done = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    k = pd.DataFrame({"Code": keys["Code"].astype(str).to_numpy(),
                      "Date": pd.to_datetime(keys["Date"]).to_numpy(dtype="datetime64[ns]"),
                      "_i": np.arange(n)})
    want = {c: (g["Date"].to_numpy(dtype="datetime64[ns]"), g["_i"].to_numpy()) for c, g in k.groupby("Code", sort=False)}
    b = bars[bars["Code"].astype(str).isin(want.keys())].copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"], kind="mergesort")
    min_base = int(round(base * 0.8))
    for code, g in b.groupby("Code", sort=False):
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        close = _col(g, "AdjC", "C")
        vol = _col(g, "AdjVo", "Vo")
        ret = np.full(len(close), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[1:] = close[1:] / close[:-1] - 1.0
        v = pd.Series(np.where(vol > 0, vol, np.nan))
        prev = v.shift(1).rolling(base, min_periods=min_base).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(v.to_numpy() / prev)
        nd = len(dates)
        ds, rows = want[code]
        for d, r_i in zip(ds, rows):
            i = int(np.searchsorted(dates, d))
            if i >= nd or dates[i] != d:
                continue
            if i + hold < nd:
                done[r_i] = dates[i + hold]
            lo = i - win + 1
            if lo < 1:
                continue
            r = ret[lo:i + 1]
            r = r[np.isfinite(r)]
            if len(r) >= min_obs:
                sd = r.std()
                if sd > 0:
                    QR[r_i] = quantile_vector((r - r.mean()) / sd, levels)
            x = lr[lo:i + 1]
            x = x[np.isfinite(x)]
            if len(x) >= min_obs:
                QV[r_i] = quantile_vector(x, levels)
    return {"QR": QR, "QV": QV, "done": done}


# --------------------------------------------------------------------------- #
# 百分位（窓ごとの物差し）と近傍
# --------------------------------------------------------------------------- #

def window_min(df: pd.DataFrame, split: dict, models=RULE_MODELS) -> dict:
    """
    {窓: (行番号, 3本の最小)}。窓 w までの全行に、窓 w の訓練の分布（前向きの百分位と同じ物差し）で付けた百分位
    （画面と同じ丸め）の最小。欠けがあれば NaN（満たさない扱い）。
    """
    w_arr = df["window"].to_numpy()
    out = {}
    for w, (tr, te, Ptr, Pte) in split.items():
        U = np.where((w_arr > 0) & (w_arr <= w))[0]
        R = E82.rounded(E79.pct_matrix(df.iloc[U], models, ref=df.iloc[tr]))
        out[w] = (U, R.min(axis=1))
    return out


def pair_median(Q: np.ndarray, n_pairs: int = N_PAIRS, seed: int = SEED) -> float:
    """Q の行から無作為に取った組（行が少なければ全部の組）の W1 の中央値。2行未満なら NaN。"""
    n = len(Q)
    if n < 2:
        return float("nan")
    if n * (n - 1) // 2 <= n_pairs:
        a, b = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        a = rng.integers(0, n, size=n_pairs)
        b = rng.integers(0, n - 1, size=n_pairs)
        b = b + (b >= a)                                   # a と違う行
    return float(np.median(np.abs(Q[a] - Q[b]).mean(axis=1)))


def done_before(done_sorted: np.ndarray, day) -> int:
    """結果が出そろう日の昇順の配列で、day より前に出そろった行の数（同じ日に出そろう行は含まない）。"""
    return int(np.searchsorted(done_sorted, np.datetime64(pd.Timestamp(day), "ns"), side="left"))


def knn(D: np.ndarray, vals: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    """距離 D の近い k 件の vals の平均と、その行の位置。"""
    k = int(min(k, len(D)))
    nn = np.argpartition(D, k - 1)[:k]
    return float(vals[nn].mean()), nn


def collect(df: pd.DataFrame, dist: dict, split: dict, starts: dict, line: float = LINE, pool_line: float = POOL_LINE,
            k_main: int = K_MAIN, k_ref=K_REF, min_pool: int = MIN_POOL, n_pairs: int = N_PAIRS,
            seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """
    判断日の候補ごとに、k-NN の見積もり（距離の形 × 近傍の数）を付けた表と、件数の記録を返す。
    starts: {窓: テスト窓の始まり}。尺度をそろえる中央値には、それより前に結果が出そろった行だけを使う。
    """
    QR, QV, done = dist["QR"], dist["QV"], dist["done"]
    tp = pd.to_numeric(df[TP], errors="coerce").to_numpy(dtype=float)
    hit = df[HIT].to_numpy(dtype=bool) if HIT in df.columns else tp >= TARGET
    ok = np.isfinite(tp) & np.isfinite(QR).all(axis=1) & np.isfinite(QV).all(axis=1)
    dates = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[ns]")
    w_arr = df["window"].to_numpy()
    score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    vol = (pd.to_numeric(df["vol_20d"], errors="coerce").to_numpy(dtype=float) if "vol_20d" in df.columns
           else np.full(len(df), np.nan))
    mins = window_min(df, split)
    ks = (k_main,) + tuple(k for k in k_ref if k != k_main)
    info = {"rule_rows": 0, "rule_ok": 0, "days2": 0, "days_small_pool": 0, "days": 0, "pool": [], "norm": {},
            "per_window": {}}
    recs = []
    for w in sorted(split):
        U, mn = mins[w]
        rule = (w_arr[U] == w) & (mn >= line)
        info["rule_rows"] += int(rule.sum())
        cand = U[rule & ok[U]]
        info["rule_ok"] += int(len(cand))
        mn_of = dict(zip(U.tolist(), mn.tolist()))
        pool = U[(mn >= pool_line) & ok[U] & ~np.isnat(done[U])]
        pool = pool[np.argsort(done[pool], kind="mergesort")]
        pdone = done[pool]
        base = pool[: done_before(pdone, starts[w])]
        sR = pair_median(QR[base], n_pairs, seed + int(w))
        sV = pair_median(QV[base], n_pairs, seed + 100 + int(w))
        sR = sR if np.isfinite(sR) and sR > 0 else 1.0
        sV = sV if np.isfinite(sV) and sV > 0 else 1.0
        info["norm"][int(w)] = {"base": int(len(base)), "sR": sR, "sV": sV}
        n_days = 0
        cdates = dates[cand]
        for d in np.unique(cdates):
            idx = cand[cdates == d]
            if len(idx) < 2:
                continue
            info["days2"] += 1
            P = pool[: done_before(pdone, d)]
            if len(P) < min_pool:
                info["days_small_pool"] += 1
                continue
            info["days"] += 1
            n_days += 1
            info["pool"].append(int(len(P)))
            tpP = tp[P]
            QRP, QVP = QR[P], QV[P]
            for i in idx:
                dR = w1(QR[i], QRP) / sR
                dV = w1(QV[i], QVP) / sV
                rec = {"Date": pd.Timestamp(d), "window": int(w), "row": int(i), "min15": float(mn_of[int(i)]),
                       "score": float(score[i]), TP: float(tp[i]), HIT: bool(hit[i]), "vol_20d": float(vol[i]),
                       "pool_n": int(len(P))}
                for name, D in (("RV", dR + dV), ("R", dR), ("V", dV)):
                    for kk in ks:
                        if name != "RV" and kk != k_main:
                            continue
                        est, nn = knn(D, tpP, kk)
                        rec[f"est_{name}_{kk}"] = est
                        if name == "RV" and kk == k_main:
                            rec["nn_bad10"] = float((tpP[nn] < -10).mean() * 100)
                recs.append(rec)
        info["per_window"][int(w)] = n_days
    return pd.DataFrame(recs), info


# --------------------------------------------------------------------------- #
# 選び方と判定
# --------------------------------------------------------------------------- #

def day_picks(c: pd.DataFrame, est: str) -> pd.DataFrame:
    """判断日ごとに、見積もり est の最上位（K）・今の並べ方の最上位（M）・候補全体の平均・後から見た最良。"""
    out = []
    if c.empty:
        return pd.DataFrame(out)
    for d, g in c.groupby("Date", sort=True):
        if len(g) < 2:
            continue
        gk = g.sort_values([est, "min15", "score"], ascending=False, kind="mergesort")
        gm = g.sort_values(["min15", "score"], ascending=False, kind="mergesort")
        out.append({"Date": d, "window": int(g["window"].iat[0]), "n": int(len(g)),
                    "K": float(gk[TP].iat[0]), "K_hit": bool(gk[HIT].iat[0]), "K_est": float(gk[est].iat[0]),
                    "K_bad_est": float(gk["nn_bad10"].iat[0]) if "nn_bad10" in gk.columns else float("nan"),
                    "M": float(gm[TP].iat[0]), "M_hit": bool(gm[HIT].iat[0]), "M_est": float(gm[est].iat[0]),
                    "all": float(g[TP].mean()), "best": float(g[TP].max()),
                    "same": bool(gk["row"].iat[0] == gm["row"].iat[0])})
    return pd.DataFrame(out)


def random_means(c: pd.DataFrame, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """判断日ごとに候補から1件を無作為に選んだときの、判断日全体の平均（draws 回ぶん）。"""
    groups = [g[TP].to_numpy(dtype=float) for _, g in c.groupby("Date", sort=True) if len(g) >= 2] if len(c) else []
    if not groups:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    tot = np.zeros(draws)
    for v in groups:
        tot += v[rng.integers(0, len(v), size=draws)]
    return tot / len(groups)


def p_upper(null: np.ndarray, obs: float) -> float:
    """無作為の分布で、実際（obs）以上になる割合（片側）。"""
    if not len(null) or not np.isfinite(obs):
        return float("nan")
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def pick_stats(pk: pd.DataFrame, col: str) -> dict:
    r = pk[col].to_numpy(dtype=float)
    n = int(len(r))
    nan = float("nan")
    hit = pk[f"{col}_hit"].to_numpy(dtype=bool) if f"{col}_hit" in pk.columns else r >= TARGET
    return {"n": n, "mean": float(r.mean()) if n else nan, "med": float(np.median(r)) if n else nan,
            "win": float((r > 0).mean() * 100) if n else nan, "hit": float(hit.mean() * 100) if n else nan,
            "bad10": float((r < -10).mean() * 100) if n else nan,
            "se": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else nan}


def window_diff(pk: pd.DataFrame, a: str = "K", b: str = "M", min_days: int = MIN_DAYS_WIN) -> dict:
    """窓ごとに「判断日の a − b の平均」。判断日が min_days 日以上ある窓だけ数える。"""
    vals, per = [], {}
    for w, g in pk.groupby("window"):
        if len(g) >= min_days:
            v = float((g[a] - g[b]).mean())
            vals.append(v)
            per[int(w)] = {"days": int(len(g)), "diff": v}
    s = summarize(np.array(vals, dtype=float))
    s["per_window"] = per
    return s


def evaluate(c: pd.DataFrame, est: str, draws: int = DRAWS, seed: int = SEED, min_days: int = MIN_DAYS_WIN) -> dict:
    """K（見積もり est の最上位）を、今の並べ方 M と無作為に比べる。判定の3条件も。"""
    pk = day_picks(c, est)
    if pk.empty:
        return {"days": 0, "effective": False, "conds": {}}
    null = random_means(c, draws, seed)
    k_mean, m_mean = float(pk["K"].mean()), float(pk["M"].mean())
    wd = window_diff(pk, "K", "M", min_days)
    p_k, p_m = p_upper(null, k_mean), p_upper(null, m_mean)
    conds = {"mean_pos": bool(wd["n"] > 0 and wd["mean"] > 0),
             "majority": bool(wd["n"] > 0 and wd["wins"] * 2 > wd["n"]),
             "p_random": bool(np.isfinite(p_k) and p_k < 0.05)}
    return {"days": int(len(pk)), "K": pick_stats(pk, "K"), "M": pick_stats(pk, "M"),
            "all": float(pk["all"].mean()), "best": float(pk["best"].mean()),
            "random_mean": float(null.mean()) if len(null) else float("nan"),
            "random_q": [float(np.quantile(null, 0.05)), float(np.quantile(null, 0.95))] if len(null) else [],
            "p_K": p_k, "p_M": p_m, "window": wd, "conds": conds, "effective": bool(all(conds.values())),
            "same_share": float(pk["same"].mean() * 100),
            "K_est": float(pk["K_est"].mean()), "M_est": float(pk["M_est"].mean()),
            "K_bad_est": float(np.nanmean(pk["K_bad_est"])) if pk["K_bad_est"].notna().any() else float("nan")}


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def within_day_rho(c: pd.DataFrame, a: str, b: str, min_n: int = 3, perm: int = 0, seed: int = SEED) -> dict:
    """候補が min_n 件以上の判断日ごとの、a と b の順位相関の平均。perm > 0 なら b を日の中で並べ替えた分布と比べる。"""
    parts = []
    if len(c):
        for _, g in c.groupby("Date", sort=True):
            x = pd.to_numeric(g[a], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(g[b], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= min_n:
                parts.append((x[ok], y[ok]))
    obs = np.array([_rho(x, y) for x, y in parts], dtype=float)
    good = np.isfinite(obs)
    if not good.any():
        return {"n": 0}
    m = float(obs[good].mean())
    res = {"n": int(good.sum()), "mean": m, "pos": int((obs[good] > 0).sum())}
    if perm:
        rng = np.random.default_rng(seed)
        null = np.empty(perm)
        for k in range(perm):
            null[k] = np.nanmean([_rho(x, y[rng.permutation(len(y))]) for x, y in parts])
        res["p_two"] = float((np.sum(np.abs(null) >= abs(m)) + 1) / (perm + 1))
        res["p_upper"] = float((np.sum(null >= m) + 1) / (perm + 1))
    return res


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict, width: int = 34) -> str:
    if not s.get("n"):
        return f"  {LT._l(name, width)}   0日"
    return (f"  {LT._l(name, width)}{s['n']:>4}日  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}・SE {s['se']:.2f}）  "
            f"勝率 {s['win']:4.1f}%  到達 {s['hit']:4.1f}%  −10%割れ {s['bad10']:4.1f}%")


def mark(b: bool) -> str:
    return "○" if b else "×"


def report(c: pd.DataFrame, info: dict, draws: int = DRAWS, seed: int = SEED, k_main: int = K_MAIN, k_ref=K_REF,
           perm: int = RHO_PERM, min_pool: int = MIN_POOL) -> dict:
    summary: dict = {"info": {k: v for k, v in info.items() if k != "pool"}}
    main_est = f"est_RV_{k_main}"
    pool = np.array(info.get("pool", []), dtype=float)
    print("\n■ 0. 件数（前向き・テスト窓ごとの物差し）")
    print(f"  今の規則の行 {info['rule_rows']}件（収益と2つの分布が出せる行 {info['rule_ok']}件）。候補が2件以上の日 "
          f"{info['days2']}日、うち近傍の母集団が{min_pool}件に満たず外した日 {info['days_small_pool']}日 → 判断日 {info['days']}日")
    if len(c):
        per_day = c.groupby("Date").size()
        print(f"  判断日の候補の数: 中央 {per_day.median():.0f}件（最大 {per_day.max()}件・2件の日 {int((per_day == 2).sum())}日）")
        summary["cands_per_day"] = {"median": float(per_day.median()), "max": int(per_day.max()),
                                    "two": int((per_day == 2).sum())}
    print("  窓ごとの判断日: " + "  ".join(f"窓{w} {n}日" for w, n in info.get("per_window", {}).items()))
    if len(pool):
        print(f"  近傍の母集団の大きさ: 最小 {pool.min():.0f} / 中央 {np.median(pool):.0f} / 最大 {pool.max():.0f}件")
    print("  尺度（窓ごとの W1 の中央値。リターンの形 / 出来高の比率）: "
          + "  ".join(f"窓{w} {v['sR']:.3f} / {v['sV']:.3f}" for w, v in info.get("norm", {}).items()))

    print(f"\n■ 1. 判定（主: リターンの形 + 出来高の比率・近い{k_main}件）")
    ev = evaluate(c, main_est, draws, seed)
    summary["main"] = ev
    if not ev["days"]:
        print("  判断日が0日 → 判定できない")
    else:
        print(line("K: k-NN の見積もりの最上位", ev["K"]))
        print(line("M: 今の並べ方（3本の最小）", ev["M"]))
        print(f"  判断日の候補全体の平均（無作為の期待値） {ev['all']:+.2f}%・後から見た最良の1件（上限の目安） {ev['best']:+.2f}%")
        print(f"  K と M が同じ候補を選んだ日 {ev['same_share']:.1f}%")
        wd = ev["window"]
        print(f"  窓ごとの差（K − M、判断日{MIN_DAYS_WIN}日以上の窓）: {fmt_s(wd)}")
        print("    " + ("  ".join(f"窓{w} {v['diff']:+.2f}pt（{v['days']}日）" for w, v in wd["per_window"].items()) or "—"))
        rq = ev["random_q"]
        print(f"  無作為（{draws:,}回）: 平均 {ev['random_mean']:+.2f}%（5〜95% 点 {rq[0]:+.2f}〜{rq[1]:+.2f}）。"
              f"K の平均 {ev['K']['mean']:+.2f}% 以上になる割合 p = {ev['p_K']:.3f}")
        cd = ev["conds"]
        verdict = "後工程として効く" if ev["effective"] else "今の並べ方・無作為と見分けられない"
        print(f"  判定: {verdict}（K − M の窓の平均が正 {mark(cd['mean_pos'])} / 上の窓が過半 {mark(cd['majority'])} / "
              f"無作為より上 p < 0.05 {mark(cd['p_random'])}）")

    print("\n■ 2. 参考: 距離の形と近傍の数（判定に使わない）")
    summary["variants"] = {}
    rows = [(f"est_{name}_{k_main}", f"{ja}・近い{k_main}件") for name, ja in VARIANTS[1:]]
    rows += [(f"est_RV_{k}", f"{VARIANTS[0][1]}・近い{k}件") for k in k_ref if k != k_main]
    for col, name in rows:
        if col not in c.columns:
            continue
        e = evaluate(c, col, draws, seed)
        summary["variants"][col] = e
        if e["days"]:
            print(f"  {LT._l(name, 44)}K {e['K']['mean']:+6.2f}%・K − M {fmt_s(e['window'])}・無作為より上 p = {e['p_K']:.3f}")

    print("\n■ 3. 参考: 判断日の中の順位相関（候補3件以上の日。判定に使わない）")
    summary["rho"] = {}
    for b, name, pm in ((TP, "見積もりと実際の収益", perm), ("min15", "見積もりと3本の最小", 0), ("vol_20d", "見積もりと vol_20d", 0)):
        r = within_day_rho(c, main_est, b, perm=pm, seed=seed) if main_est in c.columns else {"n": 0}
        summary["rho"][b] = r
        if r.get("n"):
            extra = f"・並べ替え 両側 p = {r['p_two']:.3f}" if "p_two" in r else ""
            print(f"  {LT._l(name, 24)}平均 {r['mean']:+.3f}（正の日 {r['pos']}/{r['n']}）{extra}")
        else:
            print(f"  {LT._l(name, 24)}—（順位相関を出せる日が無い）")

    print("\n■ 4. 参考: 今の並べ方と、見積もりの水準（判定に使わない）")
    if ev["days"]:
        print(f"  今の並べ方 M の平均 {ev['M']['mean']:+.2f}%: 無作為より上になる割合 p = {ev['p_M']:.3f}")
        print(f"  K の見積もりの平均 {ev['K_est']:+.2f}% 対 実際 {ev['K']['mean']:+.2f}%・M の見積もり {ev['M_est']:+.2f}% 対 実際 "
              f"{ev['M']['mean']:+.2f}%")
        print(f"  K の近傍の −10%割れ {ev['K_bad_est']:.1f}% 対 実際 {ev['K']['bad10']:.1f}%")

    print("\n■ まとめ")
    if ev["days"]:
        print(f"  判断日 {ev['days']}日: K {ev['K']['mean']:+.2f}%・M {ev['M']['mean']:+.2f}%・無作為 {ev['random_mean']:+.2f}%。"
              f"K − M {fmt_s(ev['window'])}・無作為より上 p = {ev['p_K']:.3f} → "
              f"{'後工程として効く' if ev['effective'] else '今の並べ方・無作為と見分けられない'}")
    else:
        print("  判断日が0日 → 判定できない")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def attach_vol(df: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """データセットの vol_20d を足す（見積もりがボラの言い直しになっていないかを見るだけ）。"""
    if "vol_20d" not in fr.columns:
        return df.assign(vol_20d=np.nan)
    f = fr[["Code", "Date", "vol_20d"]].copy()
    f["Code"] = f["Code"].astype(str)
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.drop_duplicates(["Code", "Date"])
    return df.merge(f, on=["Code", "Date"], how="left")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験91: 同じ日の候補から1件を、ワッサーステイン距離の k 近傍で選ぶ")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    df = attach_vol(df, fr)
    df["score"] = pd.to_numeric(df[f"score_{M.BASELINE}"], errors="coerce")
    del fr
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=250),
                        extra=("C", "AdjVo", "Vo"))
    df, chk = E83.attach_exits(df, bars, targets=(TARGET,))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行")
    dist = event_distributions(bars, df[["Code", "Date"]])
    del bars
    log(f"分布: リターンの形 {int(np.isfinite(dist['QR']).all(axis=1).sum()):,}行・出来高の比率 "
        f"{int(np.isfinite(dist['QV']).all(axis=1).sum()):,}行・結果の出そろう日 {int((~np.isnat(dist['done'])).sum()):,}行"
        f"（全 {len(df):,}行）")
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    c, info = collect(df, dist, split, starts)
    log(f"今の規則の行 {info['rule_rows']}件（実験88・probe_k15_baseline は 559件）・判断日 {info['days']}日")
    summary = report(c, info, draws=args.draws)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e91_summary.json")

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
