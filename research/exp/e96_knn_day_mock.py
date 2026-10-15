#!/usr/bin/env python3
"""
実験96（可視化の材料・2）: ダッシュボード案の模型のために、線95（1.5σ の3本がすべて95以上）の判断日のうち直近の「候補2件・3件・
4件以上」の日を1日ずつ取り、候補ごとの近傍の集計を、銘柄・日付を伏せて出す（候補は今の並べ方の順に A・B・C…）。判定はしない。
本番・画面は変えない。

運用者の依頼（2026-10-15）「ダッシュボードに載せるならどういった図になりますか？ 候補点同士で相対的にどっちが良いのかわかるような
可視化をしてほしいです。散布図で、より対象の群に近い等など」。

出すもの（集計だけ。銘柄コード・日付・1行ごとの収益は出さない）
  days[]        kind（2 / 3 / 4+）・候補の数・候補ごとに:
                  label（A, B, …。今の並べ方＝3本の最小の順）・min15（0.1刻み）・est（近傍50件の収益の 10/25/50/75/90%点・+10% 到達・
                  −10% 割れの割合）・d_nn（近傍50件までの距離の成分の平均）・cloud（近傍50件の「距離 × 収益の帯」の度数）・
                  to_hit / to_loss（母集団の +10% に届いた行・−10% 割れの行の平均の形までの距離の成分）・outcome_band（その候補の
                  実際の収益の帯）・picked_Q / picked_M
  context       全候補（線95）の見積もりの25%点の十分位の境、見積もり（25%点）× 近傍の −10% 割れの割合 の2次元の度数
                （MIN_CELL 未満は消す）、母集団の +10% 到達の行と −10% 割れの行の平均の形どうしの距離

    exp=e96_knn_day_mock.py
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
import e92_knn_quantile_pick as E92  # noqa: E402
import e95_knn_viz_data as E95  # noqa: E402

LINE = E92.LINE                        # 95
POOL_LINE = E92.POOL_LINE              # 85
K = E92.K_MAIN                         # 50
TP, HIT, TARGET = E92.TP, E92.HIT, E92.TARGET
KEY = f"p{E92.MAIN_P}_{K}"
MIN_CELL = E95.MIN_CELL
D_BINS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
Y_BANDS = [-40.0, -10.0, 0.0, 5.0, 10.0 - 1e-9, 10.01]       # 帯: −10 未満 / −10〜0 / 0〜+5 / +5〜+10 未満 / +10 到達
Y_LABELS = ["-10 未満", "-10〜0", "0〜+5", "+5〜+10 未満", "+10 到達"]
MAX_SHOW = 6                           # 4件以上の日は、見やすさのため候補がこれ以下の日を優先する
OUT_DIR = E79.OUT_DIR
log = E79.log


def band_of(v: float, hit: bool) -> str:
    if hit or v >= TARGET - 1e-9:
        return Y_LABELS[4]
    if v < -10:
        return Y_LABELS[0]
    if v < 0:
        return Y_LABELS[1]
    if v < 5:
        return Y_LABELS[2]
    return Y_LABELS[3]


def pick_days(c: pd.DataFrame) -> dict:
    """直近の候補2件・3件・4件以上（候補 MAX_SHOW 件以下を優先）の日。{kind: 日付}。"""
    out = {}
    if c.empty:
        return out
    per = c.groupby("Date").size().sort_index()
    for kind, m in (("2", per == 2), ("3", per == 3)):
        idx = per.index[m.to_numpy()]
        if len(idx):
            out[kind] = idx[-1]
    big = per[(per >= 4) & (per <= MAX_SHOW)]
    if big.empty:
        big = per[per >= 4]
    if not big.empty:
        out["4+"] = big.index[-1]
    return out


def day_detail(df: pd.DataFrame, dist: dict, split: dict, starts: dict, day, group_means: dict,
               line: float = LINE, pool_line: float = POOL_LINE, k: int = K) -> dict:
    """1つの判断日の候補ごとの近傍の集計（実験92 の collect と同じ候補・距離・近傍）。"""
    QR, QV, done = dist["QR"], dist["QV"], dist["done"]
    tp = pd.to_numeric(df[TP], errors="coerce").to_numpy(dtype=float)
    hit = df[HIT].to_numpy(dtype=bool)
    ok = np.isfinite(tp) & np.isfinite(QR).all(axis=1) & np.isfinite(QV).all(axis=1)
    dates = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[ns]")
    w_arr = df["window"].to_numpy()
    score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
    d64 = np.datetime64(pd.Timestamp(day), "ns")
    w = int(w_arr[dates == d64][0])
    U, mn = E90.window_min(df, split)[w]
    mn_of = dict(zip(U.tolist(), mn.tolist()))
    cand = U[(w_arr[U] == w) & (mn >= line) & ok[U] & (dates[U] == d64)]
    pool = U[(mn >= pool_line) & ok[U] & ~np.isnat(done[U])]
    pool = pool[np.argsort(done[pool], kind="mergesort")]
    pdone = done[pool]
    base = pool[: E90.done_before(pdone, starts[w])]
    sR = E90.pair_median(QR[base], E92.N_PAIRS, E92.SEED + w)
    sV = E90.pair_median(QV[base], E92.N_PAIRS, E92.SEED + 100 + w)
    sR = sR if np.isfinite(sR) and sR > 0 else 1.0
    sV = sV if np.isfinite(sV) and sV > 0 else 1.0
    P = pool[: E90.done_before(pdone, d64)]
    tpP, hitP, QRP, QVP = tp[P], hit[P], QR[P], QV[P]
    rows = []
    for i in cand:
        dR = E90.w1(QR[i], QRP) / sR
        dV = E90.w1(QV[i], QVP) / sV
        D = dR + dV
        kk = int(min(k, len(D)))
        nn = np.argpartition(D, kk - 1)[:kk]
        v, hv, dn = tpP[nn], hitP[nn], D[nn]
        counts = np.zeros((len(D_BINS) - 1, len(Y_BANDS) - 1), dtype=int)
        ix = np.clip(np.searchsorted(D_BINS, np.clip(dn, D_BINS[0], D_BINS[-1] - 1e-9), side="right") - 1, 0, len(D_BINS) - 2)
        iy = np.clip(np.searchsorted(Y_BANDS, np.clip(v, Y_BANDS[0], Y_BANDS[-1] - 1e-9), side="right") - 1, 0, len(Y_BANDS) - 2)
        iy = np.where(hv, len(Y_BANDS) - 2, iy)
        for a, b in zip(ix, iy):
            counts[a, b] += 1
        rows.append({"row": int(i), "min15": round(float(mn_of[int(i)]), 1), "score": float(score[i]),
                     "est": {"p10": round(float(np.quantile(v, 0.10)), 2), "p25": round(float(np.quantile(v, 0.25)), 2),
                             "p50": round(float(np.median(v)), 2), "p75": round(float(np.quantile(v, 0.75)), 2),
                             "p90": round(float(np.quantile(v, 0.90)), 2), "hit": round(float(hv.mean() * 100), 1),
                             "bad": round(float((v < -10).mean() * 100), 1), "n": int(kk)},
                     "d_nn": {"R": round(float(dR[nn].mean()), 3), "V": round(float(dV[nn].mean()), 3)},
                     "cloud": {"d_bins": D_BINS, "y_labels": Y_LABELS, "counts": counts.tolist()},
                     "to_hit": {"R": round(float(np.abs(QR[i] - group_means["hit_QR"]).mean() / sR), 3),
                                "V": round(float(np.abs(QV[i] - group_means["hit_QV"]).mean() / sV), 3)},
                     "to_loss": {"R": round(float(np.abs(QR[i] - group_means["loss_QR"]).mean() / sR), 3),
                                 "V": round(float(np.abs(QV[i] - group_means["loss_QV"]).mean() / sV), 3)},
                     "outcome_band": band_of(float(tp[i]), bool(hit[i]))})
    # 今の並べ方（3本の最小・同点は基準モデルのスコア）の順に A, B, C…
    rows.sort(key=lambda r: (-r["min15"], -r["score"]))
    for j, r in enumerate(rows):
        r["label"] = chr(ord("A") + j)
        r["picked_M"] = j == 0
    best = max(range(len(rows)), key=lambda j: (rows[j]["est"]["p25"], rows[j]["min15"], rows[j]["score"])) if rows else None
    for j, r in enumerate(rows):
        r["picked_Q"] = j == best
        del r["row"], r["score"]
    return {"n": len(rows), "pool_n": int(len(P)), "scale": {"R": round(sR, 4), "V": round(sV, 4)}, "candidates": rows}


def context(c: pd.DataFrame, group_means: dict, min_cell: int = MIN_CELL) -> dict:
    """全候補の見積もり（25%点）の十分位の境と、見積もり × 近傍の −10% 割れ の2次元の度数。"""
    out = {}
    if c.empty:
        return out
    est = pd.to_numeric(c[KEY], errors="coerce").to_numpy(dtype=float)
    bad = pd.to_numeric(c[f"bad_{K}"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(est) & np.isfinite(bad)
    est, bad = est[ok], bad[ok]
    out["p25_deciles"] = [round(float(np.quantile(est, q)), 2) for q in np.linspace(0, 1, 11)]
    xe = [float(np.quantile(est, q)) for q in np.linspace(0, 1, 7)]
    ye = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 100.0]
    ix = np.clip(np.searchsorted(xe, est, side="right") - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, bad, side="right") - 1, 0, len(ye) - 2)
    counts = np.zeros((len(xe) - 1, len(ye) - 1), dtype=int)
    for a, b in zip(ix, iy):
        counts[a, b] += 1
    small = (counts > 0) & (counts < min_cell)
    out["density"] = {"x_edges": [round(v, 2) for v in xe], "y_edges": ye, "counts": np.where(small, 0, counts).tolist(),
                      "suppressed_cells": int(small.sum()), "suppressed_rows": int(counts[small].sum()), "n": int(len(est))}
    out["group_gap"] = {"QR": round(float(np.abs(group_means["hit_QR"] - group_means["loss_QR"]).mean()), 4),
                        "QV": round(float(np.abs(group_means["hit_QV"] - group_means["loss_QV"]).mean()), 4)}
    return out


def group_shapes(df: pd.DataFrame, dist: dict) -> dict:
    g = E95.pool_groups(df, dist)
    out = {}
    for name, key in (("hit", "pool_hit"), ("loss", "pool_loss")):
        rows = g[key]
        out[f"{name}_QR"] = dist["QR"][rows].mean(axis=0) if len(rows) else np.zeros(E90.LEVELS)
        out[f"{name}_QV"] = dist["QV"][rows].mean(axis=0) if len(rows) else np.zeros(E90.LEVELS)
        out[f"{name}_n"] = int(len(rows))
    return out


def build(df: pd.DataFrame, dist: dict, split: dict, starts: dict, c: pd.DataFrame) -> dict:
    gm = group_shapes(df, dist)
    days = []
    for kind, day in pick_days(c).items():
        d = day_detail(df, dist, split, starts, day, gm, line=LINE, pool_line=POOL_LINE)   # 線は呼ぶ時点の値（テストで差し替える）
        d["kind"] = kind
        days.append(d)
    return {"days": days, "context": context(c, gm), "group_n": {"hit": gm["hit_n"], "loss": gm["loss_n"]}}


def print_brief(out: dict) -> None:
    for d in out["days"]:
        labs = " ".join(f"{r['label']}{'*' if r['picked_Q'] else ''}(25%点 {r['est']['p25']:+.1f}・到達 {r['est']['hit']:.0f}%・{r['outcome_band']})"
                        for r in d["candidates"])
        print(f"  候補{d['kind']}件の日: 母集団 {d['pool_n']}件  {labs}")
    cx = out.get("context", {})
    if cx:
        print(f"  母集団の +10% 到達の行と −10% 割れの行の平均の形の距離: リターンの形 {cx['group_gap']['QR']}・出来高の比率 {cx['group_gap']['QV']}"
              f"（候補から近傍50件までの距離は 0.5〜1.0 程度）。2次元の度数で消したマス目 {cx['density']['suppressed_cells']}（{cx['density']['suppressed_rows']}件）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験96: ダッシュボード案の模型のための、判断日の候補ごとの近傍の集計（銘柄・日付は伏せる）")
    ap.parse_args(argv)
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
    dist = E90.event_distributions(bars, df[["Code", "Date"]])
    del bars
    split = E79.window_split(df, folds)
    starts = {f.index: pd.Timestamp(f.test_start) for f in folds}
    df = E91.attach_forward(df, split)
    c, info = E95.collect(df, dist, split, starts, line=LINE)
    log(f"線 {LINE:.0f}以上の候補 {len(c)}件・判断日 {info['days']}日（実験92・95 は 96日）")
    out = build(df, dist, split, starts, c)
    out["meta"] = {"made": pd.Timestamp.today().strftime("%Y-%m-%d"), "line": LINE, "k": K, "pool_line": POOL_LINE, "min_cell": MIN_CELL}
    print("\n■ 模型にする日（銘柄・日付は伏せる。* は近傍の25%点の最上位）")
    print_brief(out)
    js = E95.to_json(out)
    print(f"\n■ 集計の JSON {len(js):,}文字を、番号付きの @viz の行に分けて出す")
    for ln in E95.chunk_lines(js):
        print(ln)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e96_day_mock.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(js)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
