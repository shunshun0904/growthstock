#!/usr/bin/env python3
"""
実験80: 「合意の上位で、1.5σ の4本が 1.2σ の4本より高い行」の前向き検証。上位2.5% での検証と、地図の位置で選ぶ案の
前向き検証も一緒に。

運用者の指示（2026-10-12）「1.5σ の4本が高い行の前向き検証をお願いします。それと、上位2.5% での検証もお願いします」。
運用者の案（同日）「もはやクラスタリングは不要でシンプルにリターンで色付けて、対象日付の予測サンプルがどこに来るかで
判断するのもありかな」。

きっかけ
-------
実験79（docs/MODEL_ADOPTION_RULES.md §35）で、合意（12本の百分位の平均）の上位10% の中を「1.5σ の4本の平均百分位 −
1.2σ（本番）の4本の平均百分位」の三分位で分けると、窓ごとの上 − 下が +2.71 ± 1.09pt（10窓中9）と目立った。ただし
三分位の並びが単調でなく（+1.79 / +4.08 / +2.28%）、6つの差を同時に見た中の1つで、百分位も全期間の分布で付けていた。

規則（先に決めたもの。ここから動かさない）
------------------------------------
  合意の上位帯の中で、d15 = 1.5σ の4本の平均百分位 − 1.2σ の4本の平均百分位 > 0 の行
  帯: 合意の上位10% と 2.5%（5% は参考）。帯の閾値は訓練の合意の分布から（テスト窓の分布は使わない）
  百分位・標準化は訓練（テスト窓より前の窓。手前20営業日は外す）の分布で付ける（実験79 の window_split と同じ）
  物差し: 窓ごとに「d15 > 0 の行の平均実収益 − d15 ≤ 0 の行の平均実収益」。判定は 平均が正・並べ替え検定の p < 0.05・
         上の窓が過半の3つ。並べ替え検定は、窓ごとに帯の中で d15 > 0 の印を並べ替える（件数はそのまま）
  参考（判定に使わない）: 1.8σ の4本 − 1.2σ の4本（d18）、d15 が訓練の帯の上の三分位以上（実験79 の切り方に近い）

注意: 仮説は同じ期間（2021〜2026年の out-of-fold）で見つけたので、前向きの形で測り直しても独立の確かめにはならない。
前向きにするのは、百分位・閾値に先の情報を使わないため。独立の確かめは、これからの予測（ライブ）で行う。

地図で選ぶ案（運用者の案）
----------------------
訓練の窓だけで PCA の地図（実験79 と同じ 20×20 のマス目、20件以上のマス目だけ）を作り、マス目ごとの過去の平均実収益を
テスト窓の行に当て、高い順に選ぶ（同点は合意の高い順）。同じ考え方をなめらかにした「12次元で近い200件の過去の平均
実収益」（k 近傍）でも選ぶ。どちらも、同じ件数を合意の上位帯で選んだときと比べる。t-SNE は新しい点を置けないので
使わない。

月に2銘柄: 合意の上位2件、d15 > 0 の中の合意の上位2件、地図の値の上位2件、k 近傍の上位2件、lgbm の上位2件。

公開ログには集計だけ（帯・群ごとの件数と平均）。

    exp=e80_k15_rule_forward.py
    args="--perm 2000"        （既定）
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
import models as M  # noqa: E402
import e79_model_clusters as E79  # noqa: E402

OUTCOME = E79.OUTCOME
MODELS = E79.MODELS
BASE4, K15, K18 = E79.BASE4, E79.K15, E79.K18
BANDS = (0.10, 0.05, 0.025)
MAIN_BANDS = (0.10, 0.025)          # 判定する帯（5% は参考）
N_PERM = 2000
MIN_GROUP = 5                       # 窓ごとの比較で、群に要る件数
KNN_K = 200
SEED = 42
OUT_DIR = E79.OUT_DIR
EXTRA_COLS = ("future_rise", "label_k12", "label_k15", "label_k18", "ret_o1_40")

log = E79.log
nmean = E79.nmean
summarize = E79.summarize
fmt_s = E79.fmt_s


def band_name(b: float) -> str:
    return f"上位{b * 100:g}%"


def group_mean(P: np.ndarray, grp, models=MODELS) -> np.ndarray:
    idx = {a: i for i, a in enumerate(models)}
    return P[:, [idx[a] for a in grp]].mean(axis=1)


def forward_frame(df: pd.DataFrame, split: dict, models=MODELS, bands=BANDS, knn_k: int = KNN_K,
                  grid_n: int = E79.GRID_N, min_cell: int = E79.MIN_CELL) -> tuple[pd.DataFrame, dict]:
    """
    前向き: テスト窓ごとに、訓練の分布で付けた百分位から 合意・d15・d18・lgbm の百分位・帯の印・地図の値・k 近傍の値を
    作ったテストの行。地図の値・k 近傍の値は「過去の平均実収益」（%）。
    """
    from sklearn.neighbors import KNeighborsRegressor
    i_lgbm = list(models).index(M.BASELINE)
    r_all = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    keep = ["Date", "window", OUTCOME] + [c for c in EXTRA_COLS if c in df.columns]
    parts, info = [], {}
    for w, (tr, te, Ptr, Pte) in split.items():
        cons_tr, cons_te = Ptr.mean(axis=1), Pte.mean(axis=1)
        out = df.iloc[te][keep].copy().reset_index(drop=True)
        out["cons"] = cons_te
        out["lgbm"] = Pte[:, i_lgbm]
        out["d15"] = group_mean(Pte, K15, models) - group_mean(Pte, BASE4, models)
        out["d18"] = group_mean(Pte, K18, models) - group_mean(Pte, BASE4, models)
        d15_tr = group_mean(Ptr, K15, models) - group_mean(Ptr, BASE4, models)
        for b in bands:
            thr = float(np.quantile(cons_tr, 1 - b))
            out[f"in_{b}"] = cons_te >= thr
            band_tr = cons_tr >= thr
            cut = float(np.quantile(d15_tr[band_tr], 2 / 3)) if band_tr.sum() >= 3 else np.inf
            out[f"d15hi_{b}"] = out["d15"].to_numpy() >= cut
        # 地図（PCA）: 訓練だけで作り、テストの行を置く
        Ztr, mu, sd = E79.standardize(Ptr)
        Zte = (Pte - mu) / sd
        Etr, comps, _ = E79.pca_fit(Ztr)
        Ete = (Zte - Ztr.mean(axis=0)) @ comps.T
        ex, ey = E79.grid_edges(Etr[:, 0], grid_n), E79.grid_edges(Etr[:, 1], grid_n)

        def cell(E):
            ix = np.clip(np.searchsorted(ex, E[:, 0], side="right") - 1, 0, grid_n - 1)
            iy = np.clip(np.searchsorted(ey, E[:, 1], side="right") - 1, 0, grid_n - 1)
            return ix * grid_n + iy

        rtr = r_all[tr]
        cells = pd.DataFrame({"c": cell(Etr[:, :2]), "r": rtr}).dropna()
        agg = cells.groupby("c")["r"].agg(["size", "mean"])
        agg = agg[agg["size"] >= min_cell]
        out["map"] = pd.Series(cell(Ete[:, :2])).map(agg["mean"]).to_numpy(dtype=float) * 100
        ok = ~np.isnan(rtr)
        knn = KNeighborsRegressor(n_neighbors=int(min(knn_k, ok.sum()))).fit(Ztr[ok], rtr[ok])
        out["knn"] = knn.predict(Zte) * 100
        info[int(w)] = {"n_train": int(len(tr)), "n_test": int(len(te)), "map_cells": int(len(agg)),
                        "map_cover": float(np.isfinite(out["map"]).mean())}
        parts.append(out)
    ft = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return ft, info


def band_summary(ft: pd.DataFrame, band: float) -> dict:
    """合意の上位帯そのもの: 成績、窓ごとの 帯 − 窓の全体、帯 − lgbm の上位（同じ件数）。"""
    sub = ft[ft[f"in_{band}"]]
    d_all, d_lgbm, ns = [], [], []
    for _, g in ft.groupby("window"):
        m = g[f"in_{band}"].to_numpy()
        n = int(m.sum())
        if n < MIN_GROUP:
            continue
        r = pd.to_numeric(g[OUTCOME], errors="coerce").to_numpy(dtype=float)
        bm = nmean(r[m])
        d_all.append((bm - nmean(r)) * 100)
        top_l = np.argsort(-g["lgbm"].to_numpy(dtype=float), kind="mergesort")[:n]
        d_lgbm.append((bm - nmean(r[top_l])) * 100)
        ns.append(n)
    return {"stats": E79.stats(sub), "vs_all": summarize(np.array(d_all)), "vs_lgbm": summarize(np.array(d_lgbm)),
            "n_per_window": float(np.mean(ns)) if ns else float("nan")}


def rule_test(ft: pd.DataFrame, band: float, col: str = "d15", flag: str | None = None, n_perm: int = N_PERM,
              seed: int = SEED, min_group: int = MIN_GROUP) -> dict:
    """
    帯の中で、印（既定は col > 0。flag を渡せばその列）の行 − それ以外の行の平均実収益を窓ごとに。並べ替え検定は、窓ごとに
    帯の中で印を並べ替える（件数はそのまま）。p = 並べ替えた平均が実際の平均以上だった割合（片側）。
    """
    rng = np.random.default_rng(seed)
    sub = ft[ft[f"in_{band}"]].reset_index(drop=True)
    f = (sub[col].to_numpy(dtype=float) > 0) if flag is None else sub[flag].to_numpy(dtype=bool)
    r = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy(dtype=float)
    w = sub["window"].to_numpy()
    groups = []
    for ww in np.unique(w):
        m = np.where((w == ww) & ~np.isnan(r))[0]
        if f[m].sum() >= min_group and (~f[m]).sum() >= min_group:
            groups.append(m)

    def stat(ff):
        return np.array([(r[g][ff[g]].mean() - r[g][~ff[g]].mean()) * 100 for g in groups], dtype=float)

    d = stat(f)
    s = summarize(d)
    p = float("nan")
    if groups and n_perm:
        perm = np.empty(n_perm)
        for i in range(n_perm):
            fp = f.copy()
            for g in groups:
                fp[g] = f[g][rng.permutation(len(g))]
            perm[i] = stat(fp).mean()
        p = float((np.sum(perm >= s["mean"]) + 1) / (n_perm + 1))
    return {"band": band, "col": col if flag is None else flag, "summary": s, "p": p, "diffs": d.tolist(),
            "windows": [int(w[g[0]]) for g in groups], "share": float(f.mean()) if len(f) else float("nan"),
            "yes": E79.stats(sub[f]), "no": E79.stats(sub[~f])}


def judge(res: dict) -> dict:
    s = res["summary"]
    need = s["n"] // 2 + 1
    out = {"positive": bool(s["n"] and s["mean"] > 0), "p_ok": bool(np.isfinite(res["p"]) and res["p"] < 0.05),
           "majority": bool(s["n"] and s["wins"] >= need), "need": need}
    out["pass"] = out["positive"] and out["p_ok"] and out["majority"]
    return out


def selection_test(ft: pd.DataFrame, band: float, keys=("map", "knn", "lgbm")) -> dict:
    """窓ごとに、合意の上位帯と同じ件数を key の高い順（同点は合意の高い順）で選んだときの平均実収益 − 帯の平均。"""
    diffs = {k: [] for k in keys}
    over = {k: [] for k in keys}
    picks = {k: [] for k in ("cons",) + tuple(keys)}
    for _, g in ft.groupby("window"):
        m = g[f"in_{band}"].to_numpy()
        n = int(m.sum())
        if n < MIN_GROUP:
            continue
        r = pd.to_numeric(g[OUTCOME], errors="coerce").to_numpy(dtype=float)
        base = nmean(r[m])
        picks["cons"].append(g[m])
        cons = g["cons"].to_numpy(dtype=float)
        for k in keys:
            v = g[k].to_numpy(dtype=float)
            v = np.where(np.isnan(v), -np.inf, v)
            order = np.lexsort((-cons, -v))
            sel = order[:n]
            diffs[k].append((nmean(r[sel]) - base) * 100)
            over[k].append(float(m[sel].mean()))
            picks[k].append(g.iloc[sel])
    out = {"cons": {"stats": E79.stats(pd.concat(picks["cons"])) if picks["cons"] else None}}
    for k in keys:
        out[k] = {"vs_cons": summarize(np.array(diffs[k])), "overlap": float(np.mean(over[k])) if over[k] else np.nan,
                  "stats": E79.stats(pd.concat(picks[k])) if picks[k] else None}
    return out


MONTHLY = (
    ("合意の上位2件", "cons", None),
    ("d15 > 0 の中で合意の上位2件", "cons", "d15"),
    ("地図の値の上位2件", "map", None),
    ("近い200件の値の上位2件", "knn", None),
    ("lgbm の上位2件", "lgbm", None),
)


def monthly(ft: pd.DataFrame, n: int = E79.MONTHLY_N) -> dict:
    """月に n 銘柄: 並べる値の高い順（同点は合意の高い順）に各月の上位 n 件。"""
    out = {}
    base_by_w = None
    for name, key, cond in MONTHLY:
        sub = ft if cond is None else ft[ft[cond].to_numpy(dtype=float) > 0]
        sub = sub.assign(_k=sub[key].astype(float).fillna(-np.inf),
                         _m=pd.to_datetime(sub["Date"]).dt.to_period("M"))
        sub = sub.sort_values(["_m", "_k", "cons"], ascending=[True, False, False], kind="mergesort")
        picks = sub.groupby("_m", sort=False).head(n)
        st = E79.stats(picks)
        st["months"] = int(picks["_m"].nunique())
        by_w = picks.groupby("window")[OUTCOME].mean()
        if base_by_w is None:
            base_by_w = by_w
            st["vs_cons"] = None
        else:
            j = base_by_w.index.intersection(by_w.index)
            st["vs_cons"] = summarize(((by_w[j] - base_by_w[j]) * 100).to_numpy())
        out[name] = st
    return out


def line(name: str, s: dict) -> str:
    return (f"  {name:<22}{s['n']:>6,}件  実収益 {s['ret']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:5.1f}%  "
            f"−10%未満 {s['bad10']:4.1f}%  上昇率 {s.get('rise', np.nan):5.1f}%  正例率 {s.get('pos12', np.nan):4.1f}%")


def report(ft: pd.DataFrame, info: dict, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    summary: dict = {"windows": sorted(info)}
    print(f"\n■ 0. 前向きの窓（テスト窓 {len(info)}つ。訓練 = それより前の窓、テスト窓の手前20営業日は外す）")
    for w, x in info.items():
        g = ft[ft["window"] == w]
        print(f"  窓{w:>2}: 訓練 {x['n_train']:>6,}件 / テスト {x['n_test']:>5,}件（{pd.to_datetime(g['Date']).min().date()} 〜 "
              f"{pd.to_datetime(g['Date']).max().date()}）  地図のマス目 {x['map_cells']}・テストの行に値が付いた割合 "
              f"{x['map_cover'] * 100:.0f}%")
    print(line("テストの全行", E79.stats(ft)))

    print("\n■ 1. 合意の上位帯そのもの（帯の閾値は訓練の合意の分布から）")
    summary["bands"] = {}
    for b in BANDS:
        bs = band_summary(ft, b)
        summary["bands"][str(b)] = bs
        tag = "" if b in MAIN_BANDS else "（参考）"
        print(line(f"{band_name(b)}{tag}", bs["stats"]) + f"  1窓あたり {bs['n_per_window']:.0f}件")
        print(f"      窓の全体との差 {fmt_s(bs['vs_all'])}  lgbm の上位（同じ件数）との差 {fmt_s(bs['vs_lgbm'])}")

    print(f"\n■ 2. 規則「d15 = 1.5σ の4本の平均百分位 − 1.2σ の4本の平均百分位 > 0」（先に決めたもの。並べ替え {n_perm}回）")
    print("  判定: 平均が正・p < 0.05・上の窓が過半。帯は上位10% と 2.5%（5% は参考）")
    summary["rule"] = {}
    for b in BANDS:
        res = rule_test(ft, b, "d15", n_perm=n_perm, seed=seed)
        jd = judge(res)
        summary["rule"][str(b)] = {**{k: v for k, v in res.items() if k not in ("yes", "no")},
                                   "yes": res["yes"], "no": res["no"], "judge": jd}
        tag = "" if b in MAIN_BANDS else "（参考）"
        verdict = ("通る" if jd["pass"] else "通らない") if b in MAIN_BANDS else "—"
        print(f"\n  {band_name(b)}{tag}: d15 > 0 の割合 {res['share'] * 100:.0f}%")
        print(line("d15 > 0", res["yes"]))
        print(line("d15 ≤ 0", res["no"]))
        print(f"      窓ごとの差 {fmt_s(res['summary'])}  並べ替え p = {res['p']:.3f}  判定: {verdict}"
              f"（平均が正 {'○' if jd['positive'] else '×'} / p < 0.05 {'○' if jd['p_ok'] else '×'} / "
              f"過半 {'○' if jd['majority'] else '×'}（{jd['need']}窓以上））")
        if b in MAIN_BANDS:
            print("      窓ごと: " + "  ".join(f"窓{w} {d:+.2f}" for w, d in zip(res["windows"], res["diffs"])))

    print("\n  参考（判定に使わない）")
    summary["rule_ref"] = {}
    for b in MAIN_BANDS:
        for col, flag, name in (("d18", None, "1.8σ の4本 − 1.2σ の4本 > 0"),
                                ("d15", f"d15hi_{b}", "d15 が訓練の帯の上の三分位以上")):
            res = rule_test(ft, b, col, flag=flag, n_perm=n_perm, seed=seed)
            summary["rule_ref"][f"{name}_{b}"] = {"summary": res["summary"], "p": res["p"], "share": res["share"],
                                                  "yes": res["yes"], "no": res["no"]}
            print(f"  {band_name(b)} {name}: 割合 {res['share'] * 100:.0f}%  {res['yes']['ret']:+.2f}% 対 "
                  f"{res['no']['ret']:+.2f}%  窓ごとの差 {fmt_s(res['summary'])}  p = {res['p']:.3f}")

    print("\n■ 3. 地図の位置で選ぶ案（訓練だけで作った PCA の地図のマス目の過去の平均 / 12次元で近い200件の過去の平均）")
    print("  合意の上位帯と同じ件数を、値の高い順に選んだときの平均実収益 − 帯の平均（窓ごと）。重なり = 選んだ行のうち帯にも入る割合")
    summary["select"] = {}
    for b in BANDS:
        st = selection_test(ft, b)
        summary["select"][str(b)] = st
        tag = "" if b in MAIN_BANDS else "（参考）"
        print(f"  {band_name(b)}{tag}: 帯 {st['cons']['stats']['ret']:+.2f}%")
        for k, nm in (("map", "地図のマス目"), ("knn", "近い200件"), ("lgbm", "lgbm の上位")):
            x = st[k]
            print(f"      {nm:<10} {x['stats']['ret']:+.2f}%  帯との差 {fmt_s(x['vs_cons'])}  重なり {x['overlap'] * 100:.0f}%")

    print("\n■ 4. 月に2銘柄（テスト窓・前向きの百分位。差は合意の上位2件との窓ごとの差）")
    mo = monthly(ft)
    summary["monthly"] = mo
    for name, s in mo.items():
        extra = f"  合意の上位2件との差 {fmt_s(s['vs_cons'])}" if s.get("vs_cons") else ""
        print(line(f"{name}（{s['months']}か月）", s) + extra)

    print("\n■ まとめ")
    for b in MAIN_BANDS:
        r = summary["rule"][str(b)]
        print(f"  {band_name(b)} の d15 規則: 窓ごとの差 {fmt_s(r['summary'])}、p = {r['p']:.3f} → "
              f"{'通る' if r['judge']['pass'] else '通らない'}")
    for b in MAIN_BANDS:
        st = summary["select"][str(b)]
        print(f"  {band_name(b)} を地図のマス目で選ぶ − 合意の上位帯: {fmt_s(st['map']['vs_cons'])}、"
              f"近い200件: {fmt_s(st['knn']['vs_cons'])}")
    print("  注意: 仮説は同じ期間（実験79）で見つけたので、独立の確かめにはならない。独立の確かめはこれからの予測で")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験80: 1.5σ の4本が高い行の前向き検証・上位2.5%・地図で選ぶ案")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    split = E79.window_split(df, folds)
    ft, info = forward_frame(df, split)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(info)}つ）")
    summary = report(ft, info, n_perm=args.perm)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e80_summary.json")

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
