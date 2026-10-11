#!/usr/bin/env python3
"""
実験81: 12モデルの予測値を「教師なし異常検知」のように使えるかの前向き検証。

運用者の案（2026-10-12）「クラスタリングは不要で、シンプルにリターンで色付けて、対象日付の予測サンプルがどこに来るかで
判断する。いわゆる教師なし異常検知的な使い方」。

地図の位置で判断するには、読み方が2つある。
  (1) 位置の過去の収益（その辺りの過去の点の平均収益）で判断する → 実験80 の §3（地図のマス目・近い200件）
  (2) 位置の「珍しさ」（過去の点が少ない所に来たか）で判断する → この実験。異常検知の本来の使い方（収益は使わない）

珍しさの物差し（どれも訓練の窓だけで作ってテスト窓の行に当てる。百分位・標準化も訓練の分布で。実験79 の window_split）
  近傍距離: 12次元（訓練で標準化）で近い20件までの平均距離
  孤立森: Isolation Forest（200本）の異常度
  地図の疎ら: 訓練だけで作った PCA の地図（20×20）で、その行が落ちたマス目の訓練の件数の少なさ（運用者が地図で見る
            「点の少ない所」）
  形の珍しさ: 12本から行ごとの平均を引いた形（並び方だけ）のマハラノビス距離。合意の高さそのものの珍しさを除く

注意: 12次元そのままの珍しさは「合意が極端に高い・低い」ことでも大きくなる（分布の端）。合意の上位帯は端なので、
それだけで珍しく見える。形の珍しさは、合意の高さを除いた並び方の珍しさ。各物差しと合意の順位相関も出す。

物差し: 帯（全行・合意の上位10%・上位2.5%。帯の閾値は訓練の合意の分布から）の中で、珍しさの三分位（切り目は訓練の
帯の中の分布から。上 = 最も珍しい）ごとの実収益・勝率・−10%未満・上昇率と、窓ごとの「上 − 下」。

公開ログには集計だけ（帯・三分位ごとの件数と平均）。

    exp=e81_model_anomaly.py
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
import e79_model_clusters as E79  # noqa: E402

OUTCOME = E79.OUTCOME
MODELS = E79.MODELS
BANDS = (1.0, 0.10, 0.025)          # 全行・上位10%・上位2.5%
SCORES = (("knn", "近傍距離"), ("iso", "孤立森"), ("sparse", "地図の疎ら"), ("shape", "形の珍しさ"))
KNN_K = 20
ISO_TREES = 200
MIN_GROUP = 5
SEED = 42
OUT_DIR = E79.OUT_DIR
EXTRA_COLS = ("future_rise", "label_k12", "ret_o1_40")

log = E79.log
nmean = E79.nmean
summarize = E79.summarize
fmt_s = E79.fmt_s


def band_name(b: float) -> str:
    return "全行" if b >= 1.0 else f"上位{b * 100:g}%"


def mahalanobis(C: np.ndarray, mean: np.ndarray, inv: np.ndarray) -> np.ndarray:
    X = C - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, inv, X), 0.0))


def scores_for_window(Ptr: np.ndarray, Pte: np.ndarray, k: int = KNN_K, seed: int = SEED,
                      grid_n: int = E79.GRID_N) -> tuple[dict, dict]:
    """訓練とテストの珍しさ（大きいほど珍しい）。どれも訓練だけで作る。"""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    Ztr, mu, sd = E79.standardize(Ptr)
    Zte = (Pte - mu) / sd
    kk = int(min(k, len(Ztr) - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Ztr)
    d_tr, _ = nn.kneighbors(Ztr)                 # 自分自身（距離 0）を除く
    d_te, _ = nn.kneighbors(Zte, n_neighbors=kk)
    iso = IsolationForest(n_estimators=ISO_TREES, random_state=seed).fit(Ztr)
    Etr, comps, _ = E79.pca_fit(Ztr)
    Ete = (Zte - Ztr.mean(axis=0)) @ comps.T
    ex, ey = E79.grid_edges(Etr[:, 0], grid_n), E79.grid_edges(Etr[:, 1], grid_n)

    def cell(E):
        ix = np.clip(np.searchsorted(ex, E[:, 0], side="right") - 1, 0, grid_n - 1)
        iy = np.clip(np.searchsorted(ey, E[:, 1], side="right") - 1, 0, grid_n - 1)
        return ix * grid_n + iy

    counts = pd.Series(cell(Etr[:, :2])).value_counts()
    sp_tr = -pd.Series(cell(Etr[:, :2])).map(counts).fillna(0).to_numpy(dtype=float)
    sp_te = -pd.Series(cell(Ete[:, :2])).map(counts).fillna(0).to_numpy(dtype=float)
    Ctr = Ptr - Ptr.mean(axis=1, keepdims=True)
    Cte = Pte - Pte.mean(axis=1, keepdims=True)
    m = Ctr.mean(axis=0)
    inv = np.linalg.pinv(np.cov(Ctr - m, rowvar=False))
    tr = {"knn": d_tr[:, 1:].mean(axis=1), "iso": -iso.score_samples(Ztr), "sparse": sp_tr,
          "shape": mahalanobis(Ctr, m, inv)}
    te = {"knn": d_te.mean(axis=1), "iso": -iso.score_samples(Zte), "sparse": sp_te,
          "shape": mahalanobis(Cte, m, inv)}
    return tr, te


def anomaly_frame(df: pd.DataFrame, split: dict, bands=BANDS, models=MODELS) -> pd.DataFrame:
    """前向きのテストの行に、合意・珍しさ・帯の印・帯ごとの珍しさの三分位（0 = 普通 … 2 = 最も珍しい）を付ける。"""
    keep = ["Date", "window", OUTCOME] + [c for c in EXTRA_COLS if c in df.columns]
    parts = []
    for w, (tr, te, Ptr, Pte) in split.items():
        s_tr, s_te = scores_for_window(Ptr, Pte)
        cons_tr, cons_te = Ptr.mean(axis=1), Pte.mean(axis=1)
        out = df.iloc[te][keep].copy().reset_index(drop=True)
        out["cons"] = cons_te
        for name, _ in SCORES:
            out[name] = s_te[name]
        for b in bands:
            thr = -np.inf if b >= 1.0 else float(np.quantile(cons_tr, 1 - b))
            btr, bte = cons_tr >= thr, cons_te >= thr
            out[f"in_{b}"] = bte
            for name, _ in SCORES:
                t = np.full(len(te), np.nan)
                if btr.sum() >= 3:
                    cuts = np.quantile(s_tr[name][btr], [1 / 3, 2 / 3])
                    t[bte] = np.digitize(s_te[name][bte], cuts)
                out[f"t_{name}_{b}"] = t
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tercile_report(af: pd.DataFrame, band: float, name: str) -> dict:
    """帯の中で、珍しさの三分位ごとの成績、窓ごとの 上 − 下、窓ごとの珍しさと実収益の順位相関、珍しさと合意の順位相関。"""
    sub = af[af[f"in_{band}"]]
    t = sub[f"t_{name}_{band}"].to_numpy(dtype=float)
    st = [E79.stats(sub[t == q]) for q in range(3)]
    r = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy(dtype=float)
    w = sub["window"].to_numpy()
    diffs, rhos = [], []
    for ww in np.unique(w):
        m = w == ww
        hi, lo = r[m & (t == 2)], r[m & (t == 0)]
        hi, lo = hi[~np.isnan(hi)], lo[~np.isnan(lo)]
        if len(hi) >= MIN_GROUP and len(lo) >= MIN_GROUP:
            diffs.append((hi.mean() - lo.mean()) * 100)
        x = sub.loc[m, name].to_numpy(dtype=float)
        if m.sum() >= 10:
            rhos.append(pd.Series(x).corr(pd.Series(r[m]), method="spearman"))
    rho_cons = pd.Series(sub[name].to_numpy(dtype=float)).corr(pd.Series(sub["cons"].to_numpy(dtype=float)),
                                                                 method="spearman")
    return {"terciles": st, "hi_lo": summarize(np.array(diffs)), "rho_ret": summarize(np.array(rhos)),
            "rho_cons": float(rho_cons), "shares": [float((t == q).mean()) for q in range(3)]}


def report(af: pd.DataFrame) -> dict:
    summary: dict = {"n": int(len(af)), "windows": sorted(int(x) for x in af["window"].unique())}
    print(f"\n■ 0. 前向きのテストの行 {len(af):,}件（テスト窓 {af['window'].nunique()}つ。訓練 = それより前の窓、手前20営業日は外す）")
    print("  珍しさは訓練だけで作る（収益は使わない）。三分位の切り目も訓練の帯の中の分布から。上 = 最も珍しい")
    summary["bands"] = {}
    for b in BANDS:
        sub = af[af[f"in_{b}"]]
        print(f"\n■ {band_name(b)}（{len(sub):,}件）  帯の全体: 実収益 {nmean(sub[OUTCOME]) * 100:+.2f}%")
        print("  物差し        合意との順位相関  普通 / 中 / 珍しい: 実収益（件数の割合）             勝率 普通 / 珍しい  "
              "−10%未満 普通 / 珍しい  上昇率 普通 / 珍しい  珍しい − 普通（窓ごと）  珍しさと実収益の順位相関（窓ごと）")
        summary["bands"][str(b)] = {}
        for name, ja in SCORES:
            x = tercile_report(af, b, name)
            summary["bands"][str(b)][name] = x
            st = x["terciles"]
            sh = x["shares"]
            print(f"  {ja:<8}  {x['rho_cons']:>+8.2f}        "
                  f"{st[0]['ret']:+.2f}% / {st[1]['ret']:+.2f}% / {st[2]['ret']:+.2f}%"
                  f"（{sh[0] * 100:.0f} / {sh[1] * 100:.0f} / {sh[2] * 100:.0f}%）  "
                  f"{st[0]['win']:.0f} / {st[2]['win']:.0f}%  {st[0]['bad10']:.1f} / {st[2]['bad10']:.1f}%  "
                  f"{st[0].get('rise', np.nan):.1f} / {st[2].get('rise', np.nan):.1f}%  {fmt_s(x['hi_lo'])}  "
                  f"{x['rho_ret']['mean']:+.3f} ± {x['rho_ret']['se']:.3f}")

    print("\n■ まとめ（珍しい − 普通、窓ごと）")
    for b in BANDS:
        parts = []
        for name, ja in SCORES:
            s = summary["bands"][str(b)][name]["hi_lo"]
            parts.append(f"{ja} {s['mean']:+.2f} ± {s['se']:.2f}pt（{s['wins']}/{s['n']}）")
        print(f"  {band_name(b)}: " + "  ".join(parts))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験81: 12モデルの予測値を教師なし異常検知のように使えるか")
    ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    split = E79.window_split(df, folds)
    af = anomaly_frame(df, split)
    log(f"前向きの行 {len(af):,}件")
    summary = report(af)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e81_summary.json")

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
