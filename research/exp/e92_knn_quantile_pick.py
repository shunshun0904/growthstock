#!/usr/bin/env python3
"""
実験92: 同じ日の候補から1件を、ワッサーステイン距離の k 近傍の「収益の下側25%点」で選び、分布で判定する（検証だけ。本番は変えない）。

運用者の指摘（2026-10-15）「平均での比較はあまり意味がないように思えます。分布のほうが良い気がします。…少なくとも平均だけの
大小の比較をして良し悪しを決めるのは本質ではないですね」。平均の差は分布関数の差の面積の「上回る側 − 下回る側」で、良い側と
悪い側を相殺した量（実験91 の cdf_areas の恒等式）。実験90 でも平均は M が上・中央値は K が上と向きが食い違っていた。運用者の
選択（AskUserQuestion）: 次の一手 = 実験92（近傍の分布で選び、分布で判定）、分布の物差し = 下側25%点（選び方にも判定にも同じ物差し）。

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  行・候補・判断日・分布・距離・近傍の母集団: 実験90 と同じ（research/exp/e90_wasserstein_knn_pick.py。前向きの百分位・+10% の
      指値・届かなければ20営業日目の終値、1.5σ の3本がすべて95以上の候補が同じ日に2件以上ある日、リターンの形 + 出来高の比率の
      W1、惜しい帯から上の結果が出そろった事例、近い50件）
  見積もり: 近傍50件の tp10c の **下側25%点**（主）。参考に 中央値・−10% 割れの割合・平均（実験90 の K と同じ）、25%点の近い100件
  選び方（判断日ごとに1件）
    Q（主）: 近傍の25%点が最も高い候補（同点は今の並べ方）
    M: 今の並べ方（3本の最小の順・同点は基準モデルのスコア順。画面と同じ）
    R: 無作為（判断日ごとに1件。2,000回）
  判定（「分布で見て後工程として効く」= 3つがそろう）
    1. Q の取引の25%点 ≥ M の取引の25%点
    2. Q の25%点が無作為の25%点の分布の上側5%に入る（片側 p < 0.05。p = 無作為の25%点が Q 以上になる割合）
    3. 下回らない: Q の 10・50・75%点がどれも無作為の同じ分位点の中央以上（下の裾を良くするために上を捨てていないか）
    そろわなければ「今の並べ方・無作為と見分けられない」。判断日は窓に偏る（実験90: 3つの窓に72日）ので窓ごとの差は出さず、
    対の差と日の復元抽出で散らばりを見る
  参考（判定に使わない）
    - 対の差（同じ日の Q − M）: 25%点・中央値・75%点、Q が上の日・同じ候補の日・下の日の割合。日を復元抽出した 2,000回の
      「25%点の差」の 5〜95% 点。判断日を前半・後半に分けたときの 25%点の差
    - 優越からのずれ: Q と M、Q と無作為（2,000回の選択をすべて合わせた分布）の分布関数の差の面積（下側 = Q が下回る量・
      上側 = 上回る量。実験91 の cdf_areas）。平均の差 = 上側 − 下側
    - ほかの物差しで選ぶ形（近傍の中央値・−10% 割れの少なさ・平均 = 実験90 の K・25%点の近い100件）の同じ指標
    - 実験90 の見直し（見るだけ。結果を見た後の再集計）: 平均で選んだ K・M・無作為の 10・25・50・75%点・−10% 割れ・+10% 到達と面積
  出すもの: 件数・分位点・割合・面積。銘柄・日付ごとの値は出さない

    exp=e92_knn_quantile_pick.py
    args="--draws 2000"       （既定）

結果（2026-10-15、run 38733526689、282dcda、30秒。docs/MODEL_ADOPTION_RULES.md §52）
------------------------------------------------------------------
  判定: 分布で見て後工程として効く（先に決めた3条件がそろった）。判断日96日（実験90 と同じ）。
      Q（近傍50件の25%点の最上位）25%点 +0.21・10%点 −4.89・中央 +7.53・平均 +4.62%・−10%割れ 3.1%・到達 47.9%
      M（今の並べ方）25%点 −0.90・10%点 −5.13・中央 +6.26・平均 +3.99%・−10%割れ 5.2%・到達 42.7%
      無作為（2,000回を合わせた分布）25%点 −1.04・中央 +4.96・平均 +3.49%。Q の25%点が無作為の25%点（5〜95% 点 −2.01〜−0.18）
      より上になる p = 0.021。10・50・75%点も無作為の中央以上（−4.89 対 −4.97・+7.53 対 +4.89・+10 対 +10）
  優越からのずれ: Q 対 M 下側 0.06 / 上側 0.69pt、Q 対 無作為 下側 0.01 / 上側 1.13pt（ほぼ優越）
  ただし: 対の差（Q − M）は 25%点 −0.13 / 中央 0 / 75%点 +1.88pt で、Q が上の日 29%・下 25%。日の復元抽出の25%点の差は
      5〜95% 点 −1.06〜+2.49pt で0をまたぐ。前半 +0.03pt・後半 +1.37pt。同じ候補を選んだ日 33.3%
  参考: 近傍の中央値で選ぶ p = 0.136（条件 ○××）、−10% 割れの少なさ p = 0.897（×××。Q 対 M 下側 1.52pt）、平均（実験90 の K）
      p = 0.397（○××）、25%点の近い100件 p = 0.090（○×○。向きは同じ）
  実験90 の見直し（見るだけ）: 平均で選んだ K は M より下の裾が重い（K 対 M 下側 0.46 / 上側 0.08pt。25%点 −0.89 対 −0.90・
      10%点 −5.16 対 −5.13・中央 +6.63 対 +6.26）
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
import e83_tp10_rule as E83  # noqa: E402
import e90_wasserstein_knn_pick as E90  # noqa: E402
import e91_expectation_monitor as E91  # noqa: E402

LINE, POOL_LINE = E90.LINE, E90.POOL_LINE
K_MAIN = E90.K_MAIN                   # 50
K_REF = (100,)
MIN_POOL, N_PAIRS = E90.MIN_POOL, E90.N_PAIRS
TARGET, TP, HIT = E90.TARGET, E90.TP, E90.HIT
PCTS = (10, 25, 50, 75)
MAIN_P = 25                           # 主の物差し: 下側25%点
DRAWS = 2000
BOOT = 2000
SEED = 42
#: (見積もりの列, 表示, 小さいほど良いか)
SELECTORS = (("p25", "近傍の25%点", False), ("p50", "近傍の中央値", False), ("bad", "近傍の −10% 割れの少なさ", True),
             ("mean", "近傍の平均（実験90 の K）", False))
OUT_DIR = E79.OUT_DIR

log = E79.log
nan = float("nan")


# --------------------------------------------------------------------------- #
# 近傍の分布の指標
# --------------------------------------------------------------------------- #

def neighbor_stats(D: np.ndarray, vals: np.ndarray, k: int) -> dict:
    """距離 D の近い k 件の vals の 25%点・中央値・−10% 割れの割合（%）・平均。"""
    k = int(min(k, len(D)))
    nn = np.argpartition(D, k - 1)[:k]
    v = vals[nn]
    return {"p25": float(np.quantile(v, 0.25)), "p50": float(np.median(v)), "bad": float((v < -10).mean() * 100),
            "mean": float(v.mean())}


def collect(df: pd.DataFrame, dist: dict, split: dict, starts: dict, line: float = LINE, pool_line: float = POOL_LINE,
            k_main: int = K_MAIN, k_ref=K_REF, min_pool: int = MIN_POOL, n_pairs: int = N_PAIRS,
            seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """実験90 の collect と同じ作りで、近傍の分布の指標（<指標>_<近傍の数>）を候補ごとに付ける。"""
    QR, QV, done = dist["QR"], dist["QV"], dist["done"]
    tp = pd.to_numeric(df[TP], errors="coerce").to_numpy(dtype=float)
    hit = df[HIT].to_numpy(dtype=bool) if HIT in df.columns else tp >= TARGET
    ok = np.isfinite(tp) & np.isfinite(QR).all(axis=1) & np.isfinite(QV).all(axis=1)
    dates = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[ns]")
    w_arr = df["window"].to_numpy()
    score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    vol = (pd.to_numeric(df["vol_20d"], errors="coerce").to_numpy(dtype=float) if "vol_20d" in df.columns
           else np.full(len(df), np.nan))
    mins = E90.window_min(df, split)
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
        base = pool[: E90.done_before(pdone, starts[w])]
        sR = E90.pair_median(QR[base], n_pairs, seed + int(w))
        sV = E90.pair_median(QV[base], n_pairs, seed + 100 + int(w))
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
            P = pool[: E90.done_before(pdone, d)]
            if len(P) < min_pool:
                info["days_small_pool"] += 1
                continue
            info["days"] += 1
            n_days += 1
            info["pool"].append(int(len(P)))
            tpP, QRP, QVP = tp[P], QR[P], QV[P]
            for i in idx:
                D = E90.w1(QR[i], QRP) / sR + E90.w1(QV[i], QVP) / sV
                rec = {"Date": pd.Timestamp(d), "window": int(w), "row": int(i), "min15": float(mn_of[int(i)]),
                       "score": float(score[i]), TP: float(tp[i]), HIT: bool(hit[i]), "vol_20d": float(vol[i]),
                       "pool_n": int(len(P))}
                for kk in ks:
                    for name, v in neighbor_stats(D, tpP, kk).items():
                        rec[f"{name}_{kk}"] = v
                recs.append(rec)
        info["per_window"][int(w)] = n_days
    return pd.DataFrame(recs), info


# --------------------------------------------------------------------------- #
# 選び方と判定
# --------------------------------------------------------------------------- #

def day_picks(c: pd.DataFrame, key: str, ascending: bool = False) -> pd.DataFrame:
    """判断日ごとに、見積もり key の最上位（Q。同点は今の並べ方）と今の並べ方の最上位（M）。日付の順。"""
    out = []
    if c.empty:
        return pd.DataFrame(out)
    for d, g in c.groupby("Date", sort=True):
        if len(g) < 2:
            continue
        gq = g.sort_values([key, "min15", "score"], ascending=[ascending, False, False], kind="mergesort")
        gm = g.sort_values(["min15", "score"], ascending=False, kind="mergesort")
        out.append({"Date": d, "window": int(g["window"].iat[0]), "n": int(len(g)),
                    "Q": float(gq[TP].iat[0]), "Q_hit": bool(gq[HIT].iat[0]), "Q_est": float(gq[key].iat[0]),
                    "M": float(gm[TP].iat[0]), "M_hit": bool(gm[HIT].iat[0]), "M_est": float(gm[key].iat[0]),
                    "all": float(g[TP].mean()), "best": float(g[TP].max()),
                    "same": bool(gq["row"].iat[0] == gm["row"].iat[0])})
    return pd.DataFrame(out)


def random_draws(c: pd.DataFrame, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """判断日ごとに候補から1件を無作為に選んだ収益の行列（draws × 判断日）。実験90 の random_means と同じ乱数の順。"""
    groups = [g[TP].to_numpy(dtype=float) for _, g in c.groupby("Date", sort=True) if len(g) >= 2] if len(c) else []
    rng = np.random.default_rng(seed)
    mat = np.empty((draws, len(groups)))
    for j, v in enumerate(groups):
        mat[:, j] = v[rng.integers(0, len(v), size=draws)]
    return mat


def qs(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {p: (float(np.quantile(x, p / 100)) if len(x) else nan) for p in PCTS}


def dist_stats(x, hit=None) -> dict:
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    if not n:
        return {"n": 0}
    h = np.asarray(hit, dtype=bool) if hit is not None else x >= TARGET - 1e-9
    s = {f"p{p}": v for p, v in qs(x).items()}
    s.update(n=n, mean=float(x.mean()), win=float((x > 0).mean() * 100), hit=float(h.mean() * 100),
             bad10=float((x < -10).mean() * 100))
    return s


def evaluate(c: pd.DataFrame, key: str, ascending: bool = False, draws: int = DRAWS, seed: int = SEED,
             boot: int = BOOT) -> dict:
    """Q（見積もり key の最上位）を、今の並べ方 M と無作為に分布で比べる。判定の3条件も。"""
    pk = day_picks(c, key, ascending)
    if pk.empty:
        return {"days": 0, "effective": False, "conds": {}}
    mat = random_draws(c, draws, seed)
    Q, Mv = pk["Q"].to_numpy(dtype=float), pk["M"].to_numpy(dtype=float)
    qQ, qM = qs(Q), qs(Mv)
    null = {p: np.quantile(mat, p / 100, axis=1) for p in PCTS}
    pooled = mat.ravel()
    p_q, p_m = E90.p_upper(null[MAIN_P], qQ[MAIN_P]), E90.p_upper(null[MAIN_P], qM[MAIN_P])
    conds = {"not_below_M": bool(qQ[MAIN_P] >= qM[MAIN_P]),
             "p_random": bool(np.isfinite(p_q) and p_q < 0.05),
             "no_worse_elsewhere": bool(all(qQ[p] >= float(np.median(null[p])) for p in PCTS if p != MAIN_P))}
    d = Q - Mv
    rng = np.random.default_rng(seed + 1)
    n = len(Q)
    bt = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, n, size=n)
        bt[b] = np.quantile(Q[idx], MAIN_P / 100) - np.quantile(Mv[idx], MAIN_P / 100)
    half = n // 2
    halves = []
    for a, b in ((0, half), (half, n)):
        if b - a >= 4:
            halves.append({"days": b - a, "diff": float(np.quantile(Q[a:b], MAIN_P / 100) - np.quantile(Mv[a:b], MAIN_P / 100))})
    return {"days": int(n), "Q": dist_stats(Q, pk["Q_hit"]), "M": dist_stats(Mv, pk["M_hit"]), "R": dist_stats(pooled),
            "null_q": {p: {"p5": float(np.quantile(null[p], 0.05)), "p50": float(np.median(null[p])),
                           "p95": float(np.quantile(null[p], 0.95))} for p in PCTS},
            "p_Q": p_q, "p_M": p_m, "conds": conds, "effective": bool(all(conds.values())),
            "same_share": float(pk["same"].mean() * 100), "all": float(pk["all"].mean()), "best": float(pk["best"].mean()),
            "Q_est": float(pk["Q_est"].mean()), "M_est": float(pk["M_est"].mean()),
            "pair": {"p25": float(np.quantile(d, 0.25)), "p50": float(np.median(d)), "p75": float(np.quantile(d, 0.75)),
                     "up": float((d > 0).mean() * 100), "same": float((d == 0).mean() * 100), "down": float((d < 0).mean() * 100)},
            "boot": {"p5": float(np.quantile(bt, 0.05)), "p50": float(np.median(bt)), "p95": float(np.quantile(bt, 0.95))},
            "halves": halves,
            "area_M": {k: v for k, v in E91.cdf_areas(Q, Mv, ranges=()).items() if k != "down_parts"},
            "area_R": {k: v for k, v in E91.cdf_areas(Q, pooled, ranges=()).items() if k != "down_parts"}}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict, width: int = 30) -> str:
    if not s.get("n"):
        return f"  {LT._l(name, width)}   0件"
    return (f"  {LT._l(name, width)}10%点 {s['p10']:+6.2f}  25%点 {s['p25']:+6.2f}  中央 {s['p50']:+6.2f}  75%点 {s['p75']:+6.2f}  "
            f"平均 {s['mean']:+5.2f}%  −10%割れ {s['bad10']:4.1f}%  到達 {s['hit']:4.1f}%")


def mark(b: bool) -> str:
    return "○" if b else "×"


def fmt_cond(e: dict) -> str:
    nq = e["null_q"]
    c = e["conds"]
    others = "・".join(f"{p}%点 {e['Q'][f'p{p}']:+.2f} 対 {nq[p]['p50']:+.2f}" for p in PCTS if p != MAIN_P)
    return (f"1. Q の{MAIN_P}%点 {e['Q'][f'p{MAIN_P}']:+.2f} ≥ M の {e['M'][f'p{MAIN_P}']:+.2f} {mark(c['not_below_M'])} / "
            f"2. 無作為の{MAIN_P}%点（5〜95% 点 {nq[MAIN_P]['p5']:+.2f}〜{nq[MAIN_P]['p95']:+.2f}）より上 p = {e['p_Q']:.3f} "
            f"{mark(c['p_random'])} / 3. 下回らない（{others}）{mark(c['no_worse_elsewhere'])}")


def fmt_pair(e: dict, name: str = "Q") -> str:
    pr, bt = e["pair"], e["boot"]
    hv = "・".join(f"{h['diff']:+.2f}pt（{h['days']}日）" for h in e["halves"]) or "—"
    return (f"対の差（{name} − M）25%点 {pr['p25']:+.2f} / 中央 {pr['p50']:+.2f} / 75%点 {pr['p75']:+.2f}pt、{name} が上の日 {pr['up']:.0f}%・"
            f"同じ {pr['same']:.0f}%・下 {pr['down']:.0f}%。日の復元抽出の{MAIN_P}%点の差 5〜95% 点 {bt['p5']:+.2f}〜{bt['p95']:+.2f}pt。"
            f"前半・後半の{MAIN_P}%点の差 {hv}")


def fmt_area(e: dict, name: str = "Q") -> str:
    am, ar = e["area_M"], e["area_R"]
    return (f"優越からのずれ: {name} 対 M 下側 {am['down']:.2f} / 上側 {am['up']:.2f}pt、{name} 対 無作為 下側 {ar['down']:.2f} / "
            f"上側 {ar['up']:.2f}pt（平均の差 = 上側 − 下側）")


def report(c: pd.DataFrame, info: dict, draws: int = DRAWS, seed: int = SEED, k_main: int = K_MAIN, k_ref=K_REF,
           min_pool: int = MIN_POOL) -> dict:
    summary: dict = {"info": {k: v for k, v in info.items() if k != "pool"}}
    pool = np.array(info.get("pool", []), dtype=float)
    print("\n■ 0. 件数（実験90 と同じ作り）")
    print(f"  今の規則の行 {info['rule_rows']}件（収益と2つの分布が出せる行 {info['rule_ok']}件）。候補が2件以上の日 "
          f"{info['days2']}日、うち近傍の母集団が{min_pool}件に満たず外した日 {info['days_small_pool']}日 → 判断日 {info['days']}日")
    if len(c):
        per_day = c.groupby("Date").size()
        print(f"  判断日の候補の数: 中央 {per_day.median():.0f}件（最大 {per_day.max()}件・2件の日 {int((per_day == 2).sum())}日）")
    if len(pool):
        print(f"  近傍の母集団の大きさ: 最小 {pool.min():.0f} / 中央 {np.median(pool):.0f} / 最大 {pool.max():.0f}件")

    main_key = f"p{MAIN_P}_{k_main}"
    print(f"\n■ 1. 判定（主: 近傍{k_main}件の{MAIN_P}%点で選ぶ Q。分布で比べる）")
    ev = evaluate(c, main_key, False, draws, seed)
    summary["main"] = ev
    if not ev["days"]:
        print("  判断日が0日 → 判定できない")
        verdict = "判定できない"
    else:
        print(line(f"Q: 近傍の{MAIN_P}%点の最上位（{ev['days']}日）", ev["Q"]))
        print(line(f"M: 今の並べ方（{ev['days']}日）", ev["M"]))
        print(line(f"無作為（{draws:,}回を合わせた分布）", ev["R"]))
        print(f"  Q と M が同じ候補を選んだ日 {ev['same_share']:.1f}%。候補全体の平均 {ev['all']:+.2f}%・後から見た最良 {ev['best']:+.2f}%")
        print(f"  条件: {fmt_cond(ev)}")
        verdict = "分布で見て後工程として効く" if ev["effective"] else "今の並べ方・無作為と見分けられない"
        print(f"  判定: {verdict}")
        print(f"  参考: {fmt_pair(ev)}")
        print(f"  参考: {fmt_area(ev)}")
        print(f"  参考: M の{MAIN_P}%点が無作為より上になる p = {ev['p_M']:.3f}。Q の見積もり（近傍の{MAIN_P}%点）の平均 "
              f"{ev['Q_est']:+.2f} 対 実際の{MAIN_P}%点 {ev['Q'][f'p{MAIN_P}']:+.2f}")

    print("\n■ 2. 参考: ほかの物差しで選ぶ形（判定に使わない）")
    summary["ref"] = {}
    rows = [(f"{key}_{k_main}", f"{ja}・近い{k_main}件", asc) for key, ja, asc in SELECTORS if key != f"p{MAIN_P}"]
    rows += [(f"p{MAIN_P}_{k}", f"近傍の{MAIN_P}%点・近い{k}件", False) for k in k_ref if k != k_main]
    for col, name, asc in rows:
        if col not in c.columns:
            continue
        e = evaluate(c, col, asc, draws, seed)
        summary["ref"][col] = e
        if e["days"]:
            cd = e["conds"]
            print(f"  {LT._l(name, 40)}Q の{MAIN_P}%点 {e['Q'][f'p{MAIN_P}']:+6.2f}（M {e['M'][f'p{MAIN_P}']:+6.2f}）・無作為より上 p = {e['p_Q']:.3f}・"
                  f"条件 {mark(cd['not_below_M'])}{mark(cd['p_random'])}{mark(cd['no_worse_elsewhere'])}・"
                  f"Q 対 M 下側 {e['area_M']['down']:.2f} / 上側 {e['area_M']['up']:.2f}pt・平均 {e['Q']['mean']:+.2f}%")

    print("\n■ 3. 実験90 の見直し（見るだけ。結果を見た後の再集計）: 平均で選んだ K・M・無作為の分布")
    ek = summary["ref"].get(f"mean_{k_main}")
    if ek and ek["days"]:
        print(line(f"K: 近傍の平均の最上位（{ek['days']}日）", ek["Q"]))
        print(line(f"M: 今の並べ方（{ek['days']}日）", ek["M"]))
        print(line(f"無作為（{draws:,}回を合わせた分布）", ek["R"]))
        print(f"  {fmt_area(ek, 'K')}")
        print(f"  {fmt_pair(ek, 'K')}")

    print("\n■ まとめ")
    if ev["days"]:
        print(f"  判断日 {ev['days']}日: {MAIN_P}%点 Q {ev['Q'][f'p{MAIN_P}']:+.2f} / M {ev['M'][f'p{MAIN_P}']:+.2f} / 無作為の中央 "
              f"{ev['null_q'][MAIN_P]['p50']:+.2f}（p = {ev['p_Q']:.3f}）→ {verdict}")
    else:
        print("  判断日が0日 → 判定できない")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ（実験90 と同じデータ）。手数料・税・すべりは含まない")
    summary["verdict"] = verdict
    return summary


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験92: 同じ日の候補から1件を、近傍の収益の25%点で選び、分布で判定する")
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
    df = E90.attach_vol(df, fr)
    df["score"] = pd.to_numeric(df[f"score_{M.BASELINE}"], errors="coerce")
    del fr
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=250),
                        extra=("C", "AdjVo", "Vo"))
    df, chk = E83.attach_exits(df, bars, targets=(TARGET,))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行")
    dist = E90.event_distributions(bars, df[["Code", "Date"]])
    del bars
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    c, info = collect(df, dist, split, starts)
    log(f"今の規則の行 {info['rule_rows']}件（実験90 は 559件）・判断日 {info['days']}日（実験90 は 96日）")
    summary = report(c, info, draws=args.draws)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e92_summary.json")

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
