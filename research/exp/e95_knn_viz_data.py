#!/usr/bin/env python3
"""
実験95（可視化の材料）: 後工程（ワッサーステイン距離の k 近傍の収益の下側25%点で、同じ日の候補から1件を選ぶ。実験92・94）を
解釈するための集計を、候補の線 95 と 90 の両方で出す（判定はしない。本番・画面は変えない）。

運用者の依頼（2026-10-15）「ワッサーステインのこの今のロジックによる選定を解釈のためにも、散布図やヒストグラム等で可視化して
ほしいです。今のダッシュボードに追加せず、まずはアーティファクトで作成してほしいです」。

公開ログには集計だけ（度数・分位点・日の中の順位の度数・平均の形・マス目の件数）。銘柄・日付・1行ごとの値は出さない。
地図（実験79）と同じく、度数が MIN_CELL 未満のマス目は出さない（0 にして、消した件数を記す）。集計は @viz で始まる行に
JSON を分けて出す（手元で図にするため。e79 の @tag と同じ扱い）。

出すもの（線ごと）
  counts     件数（行・判断日・1日の候補数の度数・窓ごとの判断日と Q/M の平均）
  picks      Q（近傍の25%点の最上位）・M（今の並べ方の先頭）・K（近傍の平均の最上位）・無作為（2,000回を合わせた分布）の
             収益の度数分布・分位関数（2.5%刻み）・分布関数の差の面積・判定の条件・対の差（Q − M）の度数分布
  estimate   見積もり（近傍の25%点）と実際の収益: 見積もりの十分位ごとの実際の分布、2次元の度数（マス目）、日の中の順位の
             突き合わせ（見積もりの順位 × 実際の順位）、日の中の順位相関
  shapes     Q が選んだ候補・M が選んだ候補・候補全体・母集団（+10% に届いた行 / −10% 割れの行 / 全部）の、標準化した
             日次リターンの分位関数と出来高の比率の分位関数の平均（60点）
  distances  候補から近い50件までの距離（リターンの形・出来高の比率の成分）の度数分布
  selectors  物差しごと（25%点・中央値・平均・−10% 割れ・25%点の近い100件）の Q の分布と判定の条件

    exp=e95_knn_viz_data.py
    args="--draws 2000"       （既定）
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
import e92_knn_quantile_pick as E92  # noqa: E402

LINES = (95.0, 90.0)                   # 候補の線（実験92 / 実験94）
POOL_LINE = E92.POOL_LINE              # 近傍の母集団（85以上）
KS = (E92.K_MAIN,) + E92.K_REF         # 近傍の数（50・100）
K_MAIN = E92.K_MAIN
TP, HIT, TARGET = E92.TP, E92.HIT, E92.TARGET
DRAWS, SEED = E92.DRAWS, E92.SEED
MIN_CELL = 5                           # マス目に出す最小件数
BINS_RET = [-40.0, -30.0, -25.0, -20.0, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.01]
BINS_PAIR = [-30.0, -20.0, -15.0, -10.0, -5.0, -2.0, -0.005, 0.005, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
BINS_DIST = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
PCT_CURVE = [p / 2 for p in range(5, 196, 5)]         # 2.5, 5, ..., 97.5
Y_EDGES = [-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.01]
N_X = 8
MAX_RANK = 4
#: (見積もりの列, 表示, 小さいほど良いか)
SELECTORS = (("p25_50", "neighbor p25 (k=50)", False), ("p50_50", "neighbor median (k=50)", False),
             ("mean_50", "neighbor mean (k=50)", False), ("bad_50", "fewest -10% in neighbors (k=50)", True),
             ("p25_100", "neighbor p25 (k=100)", False))
CHUNK = 1500
OUT_DIR = E79.OUT_DIR
log = E79.log


# --------------------------------------------------------------------------- #
# 候補ごとの近傍（実験92 の collect に、距離の成分を足したもの）
# --------------------------------------------------------------------------- #

def collect(df: pd.DataFrame, dist: dict, split: dict, starts: dict, line: float, pool_line: float = POOL_LINE,
            ks=KS, min_pool: int = E92.MIN_POOL, n_pairs: int = E92.N_PAIRS, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """実験92 の collect と同じ候補・距離・近傍。近傍50件までの距離の成分（dR_nn・dV_nn・d_nn の平均）も付ける。"""
    QR, QV, done = dist["QR"], dist["QV"], dist["done"]
    tp = pd.to_numeric(df[TP], errors="coerce").to_numpy(dtype=float)
    hit = df[HIT].to_numpy(dtype=bool) if HIT in df.columns else tp >= TARGET
    ok = np.isfinite(tp) & np.isfinite(QR).all(axis=1) & np.isfinite(QV).all(axis=1)
    dates = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[ns]")
    w_arr = df["window"].to_numpy()
    score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    mins = E90.window_min(df, split)
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
                dR = E90.w1(QR[i], QRP) / sR
                dV = E90.w1(QV[i], QVP) / sV
                D = dR + dV
                rec = {"Date": pd.Timestamp(d), "window": int(w), "row": int(i), "min15": float(mn_of[int(i)]),
                       "score": float(score[i]), TP: float(tp[i]), HIT: bool(hit[i]), "pool_n": int(len(P))}
                for kk in ks:
                    for name, v in E92.neighbor_stats(D, tpP, kk).items():
                        rec[f"{name}_{kk}"] = v
                k0 = int(min(K_MAIN, len(D)))
                nn = np.argpartition(D, k0 - 1)[:k0]
                rec["dR_nn"] = float(dR[nn].mean())
                rec["dV_nn"] = float(dV[nn].mean())
                rec["d_nn"] = float(D[nn].mean())
                recs.append(rec)
        info["per_window"][int(w)] = n_days
    return pd.DataFrame(recs), info


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def hist(x, bins) -> list:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [0] * (len(bins) - 1)
    x = np.clip(x, bins[0], bins[-1] - 1e-9)
    return [int(v) for v in np.histogram(x, bins=bins)[0]]


def curve(x) -> list:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None] * len(PCT_CURVE)
    return [round(float(np.quantile(x, p / 100)), 3) for p in PCT_CURVE]


def pick_rows(c: pd.DataFrame, key: str, ascending: bool = False) -> pd.DataFrame:
    """判断日ごとの Q（見積もり key の最上位）と M（今の並べ方の先頭）の行番号・収益・到達・3本の最小（e92.day_picks と同じ並べ方）。"""
    out = []
    if c.empty:
        return pd.DataFrame(out)
    for d, g in c.groupby("Date", sort=True):
        if len(g) < 2:
            continue
        gq = g.sort_values([key, "min15", "score"], ascending=[ascending, False, False], kind="mergesort")
        gm = g.sort_values(["min15", "score"], ascending=False, kind="mergesort")
        out.append({"Date": d, "window": int(g["window"].iat[0]), "n": int(len(g)),
                    "Q_row": int(gq["row"].iat[0]), "Q": float(gq[TP].iat[0]), "Q_hit": bool(gq[HIT].iat[0]),
                    "Q_min15": float(gq["min15"].iat[0]), "Q_est": float(gq[key].iat[0]),
                    "M_row": int(gm["row"].iat[0]), "M": float(gm[TP].iat[0]), "M_hit": bool(gm[HIT].iat[0]),
                    "M_min15": float(gm["min15"].iat[0])})
    return pd.DataFrame(out)


def calibration(c: pd.DataFrame, key: str, n_bins: int = 10) -> list:
    """見積もり key の分位（十分位）ごとの、実際の収益の分布。"""
    if c.empty:
        return []
    est = pd.to_numeric(c[key], errors="coerce").to_numpy(dtype=float)
    tp = pd.to_numeric(c[TP], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(est) & np.isfinite(tp)
    est, tp = est[ok], tp[ok]
    if len(est) < n_bins:
        return []
    order = np.argsort(est, kind="mergesort")
    groups = np.array_split(order, n_bins)
    out = []
    for i, g in enumerate(groups):
        e, r = est[g], tp[g]
        out.append({"bin": i + 1, "n": int(len(g)), "est_lo": round(float(e.min()), 3), "est_hi": round(float(e.max()), 3),
                    "est_mean": round(float(e.mean()), 3), "real_mean": round(float(r.mean()), 3),
                    "real_p25": round(float(np.quantile(r, 0.25)), 3), "real_p50": round(float(np.median(r)), 3),
                    "hit": round(float((r >= TARGET - 1e-9).mean() * 100), 1), "bad10": round(float((r < -10).mean() * 100), 1)})
    return out


def density2d(c: pd.DataFrame, key: str, n_x: int = N_X, y_edges=Y_EDGES, min_cell: int = MIN_CELL) -> dict:
    """見積もり（x。分位で n_x 等分）× 実際の収益（y。固定の刻み）の度数。min_cell 未満のマス目は 0 にして消した件数を記す。"""
    if c.empty:
        return {"x_edges": [], "y_edges": list(y_edges), "counts": [], "suppressed_cells": 0, "suppressed_rows": 0}
    est = pd.to_numeric(c[key], errors="coerce").to_numpy(dtype=float)
    tp = pd.to_numeric(c[TP], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(est) & np.isfinite(tp)
    est, tp = est[ok], tp[ok]
    x_edges = [float(np.quantile(est, q)) for q in np.linspace(0, 1, n_x + 1)]
    ix = np.clip(np.searchsorted(x_edges, est, side="right") - 1, 0, n_x - 1)
    y = np.clip(tp, y_edges[0], y_edges[-1] - 1e-9)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2)
    counts = np.zeros((n_x, len(y_edges) - 1), dtype=int)
    for a, b in zip(ix, iy):
        counts[a, b] += 1
    small = (counts > 0) & (counts < min_cell)
    out = {"x_edges": [round(v, 3) for v in x_edges], "y_edges": list(y_edges),
           "counts": np.where(small, 0, counts).tolist(), "suppressed_cells": int(small.sum()),
           "suppressed_rows": int(counts[small].sum()), "n": int(len(est))}
    return out


def rank_matrix(c: pd.DataFrame, key: str, max_rank: int = MAX_RANK) -> dict:
    """
    判断日の中の、見積もりの順位（1 = 最上位）× 実際の収益の順位（1 = 最高。同点は同じ順位）の度数。候補が 2件 / 3件 / 4件以上の日に
    分けて数える（順位は max_rank で切る）。
    """
    classes = {"2": np.zeros((max_rank, max_rank), dtype=int), "3": np.zeros((max_rank, max_rank), dtype=int),
               "4+": np.zeros((max_rank, max_rank), dtype=int)}
    days = {"2": 0, "3": 0, "4+": 0}
    if c.empty:
        return {"classes": {k: v.tolist() for k, v in classes.items()}, "days": days, "max_rank": max_rank}
    for _, g in c.groupby("Date", sort=True):
        n = len(g)
        if n < 2:
            continue
        cls = "2" if n == 2 else ("3" if n == 3 else "4+")
        days[cls] += 1
        re = (-pd.to_numeric(g[key], errors="coerce")).rank(method="min").to_numpy()
        rr = (-pd.to_numeric(g[TP], errors="coerce")).rank(method="min").to_numpy()
        for a, b in zip(re, rr):
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            ia, ib = int(min(a, max_rank)) - 1, int(min(b, max_rank)) - 1
            classes[cls][ia, ib] += 1
    return {"classes": {k: v.tolist() for k, v in classes.items()}, "days": days, "max_rank": max_rank}


def shape_mean(Q: np.ndarray, rows) -> dict:
    rows = np.asarray(rows, dtype=int)
    if not len(rows):
        return {"n": 0, "mean": None}
    sub = Q[rows]
    sub = sub[np.isfinite(sub).all(axis=1)]
    if not len(sub):
        return {"n": 0, "mean": None}
    return {"n": int(len(sub)), "mean": [round(float(v), 4) for v in sub.mean(axis=0)]}


def shapes(c: pd.DataFrame, pk: pd.DataFrame, dist: dict, pool_rows: dict) -> dict:
    """グループごとの、標準化した日次リターンの分位関数（QR）と出来高の比率の分位関数（QV）の平均。"""
    groups = {"picked_Q": pk["Q_row"].to_numpy() if len(pk) else np.array([], dtype=int),
              "picked_M": pk["M_row"].to_numpy() if len(pk) else np.array([], dtype=int),
              "candidates": c["row"].to_numpy() if len(c) else np.array([], dtype=int)}
    groups.update(pool_rows)
    out = {"probs": [round((j + 0.5) / E90.LEVELS, 4) for j in range(E90.LEVELS)]}
    for name, rows in groups.items():
        out[name] = {"QR": shape_mean(dist["QR"], rows), "QV": shape_mean(dist["QV"], rows)}
    return out


def pool_groups(df: pd.DataFrame, dist: dict, pool_line: float = POOL_LINE) -> dict:
    """母集団（3本の最小が pool_line 以上・収益と分布の出せる行）の、+10% に届いた行 / −10% 割れの行 / 全部の行番号。"""
    if "min15" not in df.columns:
        return {"pool_hit": np.array([], dtype=int), "pool_loss": np.array([], dtype=int), "pool_all": np.array([], dtype=int)}
    tp = pd.to_numeric(df[TP], errors="coerce").to_numpy(dtype=float)
    mn = pd.to_numeric(df["min15"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(tp) & (mn >= pool_line) & np.isfinite(dist["QR"]).all(axis=1) & np.isfinite(dist["QV"]).all(axis=1)
    rows = np.where(ok)[0]
    return {"pool_hit": rows[tp[rows] >= TARGET - 1e-9], "pool_loss": rows[tp[rows] < -10], "pool_all": rows}


def stats_of(x, hit=None) -> dict:
    s = E92.dist_stats(x, hit)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}


def per_window(pk: pd.DataFrame, min_days_q: int = 10) -> list:
    out = []
    if pk.empty:
        return out
    for w, g in pk.groupby("window"):
        rec = {"window": int(w), "days": int(len(g)), "Q_mean": round(float(g["Q"].mean()), 3),
               "M_mean": round(float(g["M"].mean()), 3), "same": round(float((g["Q_row"] == g["M_row"]).mean() * 100), 1)}
        if len(g) >= min_days_q:
            rec["Q_p25"] = round(float(np.quantile(g["Q"], 0.25)), 3)
            rec["M_p25"] = round(float(np.quantile(g["M"], 0.25)), 3)
        out.append(rec)
    return out


def summarize_line(c: pd.DataFrame, info: dict, df: pd.DataFrame, dist: dict, draws: int = DRAWS, seed: int = SEED) -> dict:
    """線ごとの集計。"""
    key = f"p25_{K_MAIN}"
    out: dict = {"counts": {k: v for k, v in info.items() if k != "pool"}}
    if len(info.get("pool", [])):
        p = np.array(info["pool"], dtype=float)
        out["counts"]["pool_size"] = {"min": int(p.min()), "median": int(np.median(p)), "max": int(p.max())}
    if c.empty:
        out["picks"] = {}
        return out
    per_day = c.groupby("Date").size()
    out["counts"]["cands_per_day"] = {"2": int((per_day == 2).sum()), "3": int((per_day == 3).sum()),
                                      "4": int((per_day == 4).sum()), "5-7": int(((per_day >= 5) & (per_day <= 7)).sum()),
                                      "8+": int((per_day >= 8).sum())}
    pk = pick_rows(c, key)
    pkK = pick_rows(c, f"mean_{K_MAIN}")
    mat = E92.random_draws(c, draws, seed)
    pooled = mat.ravel()
    evQ = E92.evaluate(c, key, False, draws, seed)
    evK = E92.evaluate(c, f"mean_{K_MAIN}", False, draws, seed)
    Q, Mv, Kv = pk["Q"].to_numpy(dtype=float), pk["M"].to_numpy(dtype=float), pkK["Q"].to_numpy(dtype=float)
    d = Q - Mv
    out["counts"]["per_window"] = per_window(pk)
    out["picks"] = {
        "bins": BINS_RET, "pct": PCT_CURVE,
        "Q": {"stats": stats_of(Q, pk["Q_hit"]), "hist": hist(Q, BINS_RET), "curve": curve(Q)},
        "M": {"stats": stats_of(Mv, pk["M_hit"]), "hist": hist(Mv, BINS_RET), "curve": curve(Mv)},
        "K": {"stats": stats_of(Kv, pkK["Q_hit"]), "hist": hist(Kv, BINS_RET), "curve": curve(Kv)},
        "R": {"stats": stats_of(pooled), "hist": hist(pooled, BINS_RET), "curve": curve(pooled), "draws": int(draws)},
        "all": {"stats": stats_of(c[TP].to_numpy(dtype=float), c[HIT].to_numpy(dtype=bool))},
        "areas": {"Q_vs_M": E91.cdf_areas(Q, Mv, ranges=()), "Q_vs_R": E91.cdf_areas(Q, pooled, ranges=()),
                  "K_vs_M": E91.cdf_areas(Kv, Mv, ranges=()), "K_vs_R": E91.cdf_areas(Kv, pooled, ranges=())},
        "null_q": evQ["null_q"], "p_Q": evQ["p_Q"], "p_M": evQ["p_M"], "p_K": evK["p_Q"],
        "conds_Q": evQ["conds"], "conds_K": evK["conds"], "effective_Q": evQ["effective"],
        "same_share": evQ["same_share"], "boot": evQ["boot"], "halves": evQ["halves"],
        "pair": {"bins": BINS_PAIR, "hist": hist(d, BINS_PAIR), "up": round(float((d > 0).mean() * 100), 1),
                 "same": round(float((d == 0).mean() * 100), 1), "down": round(float((d < 0).mean() * 100), 1),
                 "p25": round(float(np.quantile(d, 0.25)), 3), "p50": round(float(np.median(d)), 3),
                 "p75": round(float(np.quantile(d, 0.75)), 3)},
        "Q_min15": {"ge95": round(float((pk["Q_min15"] >= 95).mean() * 100), 1)},
        "M_min15": {"ge95": round(float((pk["M_min15"] >= 95).mean() * 100), 1)},
    }
    for k in ("areas",):
        for name, a in out["picks"][k].items():
            out["picks"][k][name] = {kk: round(float(vv), 4) for kk, vv in a.items() if kk != "down_parts"}
    rho = E90.within_day_rho(c, key, TP, perm=500, seed=seed)
    out["estimate"] = {"key": key, "calibration": calibration(c, key), "density": density2d(c, key),
                       "rank": rank_matrix(c, key), "rho": rho,
                       "est_vs_real": {"est_mean_all": round(float(c[key].mean()), 3),
                                       "real_p25_all": round(float(np.quantile(c[TP], 0.25)), 3)}}
    out["shapes"] = shapes(c, pk, dist, pool_groups(df, dist))
    out["distances"] = {"bins": BINS_DIST, "dR": hist(c["dR_nn"], BINS_DIST), "dV": hist(c["dV_nn"], BINS_DIST),
                        "d": hist(c["d_nn"], [v * 2 for v in BINS_DIST]), "d_bins": [v * 2 for v in BINS_DIST],
                        "share_R": round(float((c["dR_nn"] / (c["dR_nn"] + c["dV_nn"])).mean()), 3),
                        "norm": info.get("norm", {})}
    sel = []
    for col, name, asc in SELECTORS:
        if col not in c.columns:
            continue
        e = E92.evaluate(c, col, asc, draws, seed)
        sel.append({"key": col, "name": name, "ascending": asc, "Q": stats_of(pick_rows(c, col, asc)["Q"]),
                    "p_Q": round(float(e["p_Q"]), 4), "conds": e["conds"], "effective": e["effective"],
                    "area_M": {k: round(float(v), 4) for k, v in e["area_M"].items()}, "same_share": round(float(e["same_share"]), 1)})
    out["selectors"] = sel
    return out


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def _js(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), default=_js, allow_nan=False)


_CHUNK_RE = re.compile(r"@viz(\d{4})\|")


def chunk_lines(s: str, size: int = CHUNK) -> list:
    """JSON の文字列を「@viz0001|」のように番号付きの行に分ける（ログの1行が長くなりすぎないように。番号でつなぎ直す）。"""
    return [f"@viz{i:04d}|" + s[j:j + size] for i, j in enumerate(range(0, len(s), size))]


def parse_chunks(lines) -> dict:
    """番号付きの @viz の行をつなぎ直して JSON に戻す（手元で図にするとき用）。見出しや時刻の前置きは無視する。"""
    parts = {}
    for ln in lines:
        m = _CHUNK_RE.search(ln.rstrip("\r\n"))
        if m:
            parts[int(m.group(1))] = ln.rstrip("\r\n")[m.end():]
    return json.loads("".join(parts[k] for k in sorted(parts))) if parts else {}


def print_brief(line: float, s: dict) -> None:
    pk = s.get("picks", {})
    if not pk:
        print(f"  線 {line:.0f}: 判断日 0日")
        return
    q, m, r = pk["Q"]["stats"], pk["M"]["stats"], pk["R"]["stats"]
    print(f"  線 {line:.0f}: 判断日 {q['n']}日。25%点 Q {q['p25']:+.2f} / M {m['p25']:+.2f} / 無作為 {r['p25']:+.2f}（p = {pk['p_Q']:.3f}）、"
          f"平均 Q {q['mean']:+.2f} / M {m['mean']:+.2f} / 無作為 {r['mean']:+.2f}。Q 対 M 下側 {pk['areas']['Q_vs_M']['down']:.2f} / "
          f"上側 {pk['areas']['Q_vs_M']['up']:.2f}pt。マス目で消した件数 {s['estimate']['density']['suppressed_rows']}"
          f"（{s['estimate']['density']['suppressed_cells']}マス）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験95: 後工程の解釈のための集計（可視化の材料）")
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
    df = E91.attach_forward(df, split)
    out = {"meta": {"made": pd.Timestamp.today().strftime("%Y-%m-%d"), "draws": int(args.draws), "k_main": K_MAIN,
                    "pool_line": POOL_LINE, "min_cell": MIN_CELL, "levels": E90.LEVELS, "rows_total": int(len(df))},
           "lines": {}}
    print("\n■ 集計（判定はしない）")
    for line in LINES:
        c, info = collect(df, dist, split, starts, line=line)
        s = summarize_line(c, info, df, dist, draws=args.draws)
        out["lines"][f"{line:.0f}"] = s
        print_brief(line, s)
    js = to_json(out)
    print(f"\n■ 集計の JSON {len(js):,}文字を、番号付きの @viz の行（{CHUNK} 文字ずつ）に分けて出す（集計だけ。銘柄・日付ごとの値は無い）")
    for ln in chunk_lines(js):
        print(ln)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e95_viz.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(js)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
