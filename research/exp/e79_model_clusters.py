#!/usr/bin/env python3
"""
実験79: 12モデルの予測値の傾向分析（クラスタリング）。

運用者の指示（2026-10-12）「12個のモデルの予測値を使った傾向分析に入ってください。クラスタリングの件です。
2次元に削減した後にリターン（上昇率）の大小で色付けてもよいですね」。その前の指示（2026-10-11）「12モデルの
予測値自体を特徴量として使うイメージ。クラスター数は決めていないが、リターンが大きくなるクラスターが見つかれば
かなり儲けもん」。

何を見るか
---------
本番の4モデル（lgbm / xgb / cat / logit）とラベル違いの8本（到達しきい値 1.5σ / 1.8σ × 4学習器）の out-of-fold
スコア（週次の再学習が Release に上げたもの。本番と同じ窓 36ヶ月 / 6ヶ月 / 6ヶ月、各 16,320件）を、画面と同じ
「そのモデル自身の過去スコア分布での百分位」（pctHistorical）に直し、1行（銘柄 × 日付）を 12次元の点にする。
12本の百分位の平均を「合意」と呼ぶ。

  1. 12本の相関と主成分（PCA の寄与率と負荷量）、合意の十分位ごとの成績
  2. 2次元の地図: PCA の第1・第2主成分と t-SNE。地図をマス目に切り、マス目ごとの件数・平均実収益・平均上昇率・
     正例率・勝率を @grid の行に出す（手元で色付けして描く）。第1主成分の十分位の中で第2・第3主成分の上下を比べる
  3. k-means（12次元を標準化。k = 3〜12）。塊ごとの成績と、同じ窓で合意の高さが同じ行から見込める収益との差
     （形の効果。合意の高さで説明できる分を引いた残り）
  4. 合意の上位10% の中の「ずれの型」: 行ごとに12本の平均を引いた形で k-means。分かりやすい差（木 − ロジスティック、
     1.8σ − 1.2σ、1.5σ − 1.2σ、12本のばらつき、12本の最小）の三分位も
  5. 前向きの確かめ: 過去の窓だけで百分位・標準化・k-means を作り、過去の実収益が最も高い塊を選んで次の窓で測る。
     同じ件数を合意の上位で選んだとき・lgbm の上位で選んだときと比べる。プラセボは「合意の高さが同じ行の中で過去の
     実収益を並べ替えて選ぶ」（形の情報だけ消す）。上位10% の中の型も同じ手順で
  6. 月に2銘柄: 合意の上位2件、lgbm の上位2件、前向きに選んだ塊・型の中の合意の上位2件

物差しは実収益 ret_o1_20（翌営業日の寄りで買い、20営業日後の5日平均終値で売る。lab.OUTCOME）、上昇率 future_rise
（翌営業日の寄りから20営業日の間の最高終値まで。ラベルの到達判定に使う値）、ラベル（1.2σ / 1.5σ / 1.8σ）。

公開ログに出すのは集計だけ（マス目は MIN_CELL 件以上、塊・型・分位は件数つきの平均）。銘柄・日付・1行ごとの値は
出さない。@ で始まる行は手元で図にするための集計（同じく集計だけ）。

    exp=e79_model_clusters.py
    args="--k 8 --placebo 200"          （既定）
    args="--no-tsne --placebo 20"        （軽く回す）

結果（2026-10-12、run 38183373898、216秒。docs/MODEL_ADOPTION_RULES.md §35）
--------------------------------------------------------------------
  12本の順位相関 0.61〜0.96（中央 0.78）。PC1 79.9%（合意）/ PC2 6.6%（ロジスティック 対 木）/ PC3 3.7%（1.8σ 対 1.2σ）
  合意の十分位 1 → 10: 実収益 +0.01% → +2.72%（7〜9 は +1% 前後、10 で跳ねる）、正例率 6.2% → 36.5%、上昇率 7.3〜9.0%（横ばい）
  合意の上位3割で PC2 / PC3 の上 − 下: +0.34 ± 0.56pt / +0.36 ± 0.42pt（どちらも 5/11）
  k = 8 の最良の塊は「12本がそろって高い」塊（+2.25%、同じ件数を合意の上位で +2.26%、形の効果 −0.06 ± 0.16pt）
  前向き: 選んだ塊 − 合意の上位 k = 6 −0.45 / k = 8 −1.09 / k = 12 −0.61pt、プラセボ以上 0.71〜0.88
  上位10% の型 − 上位10% 全体 −0.51 ± 0.97pt（プラセボ以上 0.47）
  月に2銘柄: 合意の上位2件 +1.68%、lgbm の上位2件 +0.70%（差 −1.19 ± 1.05pt）
  → 並び方の塊は足しにならない。効くのは12本がそろって高いか（合意）だけ
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
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import models as M  # noqa: E402
import walkforward as WF  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

DATA_DIR = lab.DATA_DIR
OUT_DIR = os.path.join(DATA_DIR, "oof")
OUTCOME = lab.OUTCOME                                   # ret_o1_20
#: 実験79〜88 の4学習器（当時の 1.2σ の4本）。2026-11-02 の予測からロジスティック回帰を毎日の採点から外したが
#: （models.RETIRED）、実験は記録どおり logit を含む4本のまま（models.ALGOS に頼らない）
ALGOS4 = ("lgbm", "xgb", "cat", "logit")
#: 当時の画面の並び（1.2σ の4 → 1.5σ の4 → 1.8σ の4）
MODELS = ALGOS4 + tuple(M.VARIANTS)
#: ラベルの列をどの out-of-fold から取るか（ラベル違いの OOF の label はそのモデルのラベル）
LABEL_FROM = {M.BASELINE: "label_k12", "lgbm_k15": "label_k15", "lgbm_k18": "label_k18"}
TREES = tuple(a for a in MODELS if M.base_algo(a) in ("lgbm", "xgb", "cat"))
LOGITS = tuple(a for a in MODELS if M.base_algo(a) == "logit")
BASE4 = ALGOS4
K15 = tuple(M.variants_of(1.5))
K18 = tuple(M.variants_of(1.8))

MIN_CELL = 20                 # 地図のマス目に出す最小件数（それ未満は出さない）
GRID_N = 20                   # 地図のマス目（縦横）
KS = tuple(range(3, 13))
K_MAIN = 8
WF_KS = (4, 6, 8, 12)
TOP_Q = 0.90                  # 合意の上位10%
TYPE_K = 4
MIN_TRAIN_WINDOWS = 2         # 前向き: 訓練に窓がいくつ要るか
MIN_PICK_ROWS = 50            # 前向き: 選べる塊の最小件数（訓練）
MIN_PICK_SHARE = 0.02         # 同、訓練の行に占める割合
MIN_TYPE_ROWS = 30
MIN_TYPE_SHARE = 0.05
N_PLACEBO = 200
CONS_BINS = 10                # プラセボで「合意の高さが同じ」とみなす分位の数
MIN_WIN_ROWS = 10             # 窓ごとの比較で、1窓に要る件数
SEED = 42
MONTHLY_N = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def short(a: str) -> str:
    return M.SHORT.get(a, a)


def emit(tag: str, *vals) -> None:
    """手元で図にするための集計の行（@tag,値,...）。"""
    out = []
    for v in vals:
        if isinstance(v, (float, np.floating)):
            out.append("" if not np.isfinite(v) else f"{float(v):.4g}")
        else:
            out.append(str(v))
    print("@" + tag + "," + ",".join(out))


def nmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #

def oof_path(algo: str, data_dir: str = DATA_DIR) -> str:
    """Release（data-raw）のアセット名。基準モデルは oof.parquet、ほかは <名前>_oof.parquet。"""
    return os.path.join(data_dir, "oof.parquet" if algo == M.BASELINE else f"{algo}_oof.parquet")


def load_oofs(data_dir: str = DATA_DIR, models=MODELS) -> tuple[pd.DataFrame, dict]:
    """
    12本の out-of-fold を (Code, Date) で横に並べる。スコアは score_<名前>、ラベルは label_k12 / k15 / k18、
    実収益は基準モデルの OUTCOME。info にモデルごとの件数・重複・欠測と、突き合わせた件数を入れる。
    """
    info = {"rows": {}, "dups": {}, "nan_scores": {}}
    wide = None
    for a in models:
        p = oof_path(a, data_dir)
        if not os.path.exists(p):
            raise SystemExit(f"{a} の out-of-fold がありません（{os.path.basename(p)}）。"
                             "週次の再学習が Release に上げたものを使う")
        o = pd.read_parquet(p)
        need = {"Code", "Date", "score"}
        if a in LABEL_FROM:
            need.add("label")
        if a == M.BASELINE:
            need.add(OUTCOME)
        miss = sorted(need - set(o.columns))
        if miss:
            raise SystemExit(f"{a} の out-of-fold に列がありません: {miss}")
        o = o.copy()
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        info["rows"][a] = int(len(o))
        info["dups"][a] = int(o.duplicated(["Code", "Date"]).sum())
        o = o.drop_duplicates(["Code", "Date"], keep="first")
        info["nan_scores"][a] = int(o["score"].isna().sum())
        keep, ren = ["Code", "Date", "score"], {"score": f"score_{a}"}
        if a in LABEL_FROM:
            keep.append("label")
            ren["label"] = LABEL_FROM[a]
        if a == M.BASELINE:
            keep.append(OUTCOME)
        part = o[keep].rename(columns=ren)
        wide = part if wide is None else wide.merge(part, on=["Code", "Date"], how="inner")
    info["joined"] = int(len(wide))
    wide = wide.dropna(subset=[f"score_{a}" for a in models])
    info["kept"] = int(len(wide))
    return wide.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True), info


def attach_outcomes(wide: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """上昇率（future_rise）と40営業日の実収益（ret_o1_40）を足す。"""
    cols = [c for c in ("future_rise", "ret_o1_40") if c in frame.columns]
    f = frame[["Code", "Date"] + cols].copy()
    f["Code"] = f["Code"].astype(str)
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.drop_duplicates(["Code", "Date"])
    return wide.merge(f, on=["Code", "Date"], how="left")


def make_windows(dates: pd.Series) -> list:
    """本番の out-of-fold と同じ窓（train_multi.oof_scores / train_production.oof_scores と同じ引数）。"""
    return WF.make_folds(pd.to_datetime(dates), min_train_months=OOF_MIN_TRAIN_MONTHS,
                         test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                         embargo_days=B.RISE_HORIZON)


def assign_windows(dates: pd.Series, folds: list) -> np.ndarray:
    """各行がどのテスト窓に入るか（窓の番号。どれにも入らなければ 0）。"""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    w = np.zeros(len(d), dtype=int)
    for f in folds:
        m = (d >= np.datetime64(pd.Timestamp(f.test_start))) & (d <= np.datetime64(pd.Timestamp(f.test_end)))
        w[m] = f.index
    return w


# --------------------------------------------------------------------------- #
# 百分位・標準化・成績
# --------------------------------------------------------------------------- #

def pct_within(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ref の中で x より低い割合 × 100（画面の pctHistorical と同じく、同点は数えない）。"""
    if len(ref_sorted) == 0:
        return np.full(len(x), np.nan)
    return np.searchsorted(ref_sorted, x, side="left") / len(ref_sorted) * 100.0


def pct_matrix(rows: pd.DataFrame, models=MODELS, ref: pd.DataFrame | None = None) -> np.ndarray:
    """各モデルのスコアを ref（既定は rows 自身）の分布での百分位に直した (行 × モデル) の行列。"""
    ref = rows if ref is None else ref
    cols = []
    for a in models:
        r = np.sort(ref[f"score_{a}"].to_numpy(dtype=float))
        cols.append(pct_within(r, rows[f"score_{a}"].to_numpy(dtype=float)))
    return np.column_stack(cols)


def standardize(P: np.ndarray, mu=None, sd=None):
    mu = np.nanmean(P, axis=0) if mu is None else mu
    sd = np.nanstd(P, axis=0) if sd is None else sd
    sd = np.where(sd > 0, sd, 1.0)
    return (P - mu) / sd, mu, sd


def stats(d: pd.DataFrame) -> dict:
    """行の集まりの成績（% 表示の数）。"""
    r = pd.to_numeric(d[OUTCOME], errors="coerce").dropna()
    have = len(r) > 0
    out = {"n": int(len(d)),
           "ret": float(r.mean()) * 100 if have else np.nan,
           "med": float(r.median()) * 100 if have else np.nan,
           "win": float((r > 0).mean()) * 100 if have else np.nan,
           "hit10": float((r >= 0.10).mean()) * 100 if have else np.nan,
           "bad10": float((r < -0.10).mean()) * 100 if have else np.nan}
    if "future_rise" in d.columns:
        x = pd.to_numeric(d["future_rise"], errors="coerce").dropna()
        out["rise"] = float(x.mean()) * 100 if len(x) else np.nan
        out["rise_med"] = float(x.median()) * 100 if len(x) else np.nan
    if "ret_o1_40" in d.columns:
        x = pd.to_numeric(d["ret_o1_40"], errors="coerce").dropna()
        out["ret40"] = float(x.mean()) * 100 if len(x) else np.nan
    for c in ("label_k12", "label_k15", "label_k18"):
        if c in d.columns:
            y = pd.to_numeric(d[c], errors="coerce").dropna()
            out["pos" + c[-2:]] = float(y.mean()) * 100 if len(y) else np.nan
    return out


def window_diffs(df: pd.DataFrame, mask: np.ndarray, base: np.ndarray | None = None,
                 min_rows: int = MIN_WIN_ROWS) -> np.ndarray:
    """窓ごとに「mask の行の平均実収益 − base の行（既定は窓の全行）の平均実収益」（pt）。少ない窓は落とす。"""
    r = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    w = df["window"].to_numpy()
    base = np.ones(len(df), dtype=bool) if base is None else base
    out = []
    for ww in np.unique(w):
        m = w == ww
        a = r[m & mask]
        b = r[m & base]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) >= min_rows and len(b) >= min_rows:
            out.append((a.mean() - b.mean()) * 100)
    return np.array(out, dtype=float)


def window_means(df: pd.DataFrame, vals: np.ndarray, mask: np.ndarray, min_rows: int = MIN_WIN_ROWS) -> np.ndarray:
    """窓ごとに vals の mask の行の平均。少ない窓は落とす。"""
    w = df["window"].to_numpy()
    out = []
    for ww in np.unique(w):
        v = vals[(w == ww) & mask]
        v = v[~np.isnan(v)]
        if len(v) >= min_rows:
            out.append(v.mean())
    return np.array(out, dtype=float)


def summarize(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = int(len(d))
    return {"n": n, "mean": float(d.mean()) if n else float("nan"),
            "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "wins": int((d > 0).sum())}


def fmt_s(s: dict) -> str:
    if not s["n"]:
        return "—"
    return f"{s['mean']:+.2f} ± {s['se']:.2f}pt（上の窓 {s['wins']}/{s['n']}）"


def resid_within_window(df: pd.DataFrame, cons: np.ndarray, bins: int = 10) -> np.ndarray:
    """
    各行の実収益から「同じ窓で合意の高さが同じ行（合意を bins 分位に切る）の平均」を引いた残り（pt）。
    塊の形が、合意の高さで説明できる分のほかに収益と結びつくかを見る。
    """
    r = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    w = df["window"].to_numpy()
    for ww in np.unique(w):
        m = np.where(w == ww)[0]
        if len(m) < bins * 2:
            continue
        q = pd.qcut(pd.Series(cons[m]), bins, labels=False, duplicates="drop").to_numpy()
        e = pd.Series(r[m]).groupby(q).transform("mean").to_numpy()
        out[m] = (r[m] - e) * 100
    return out


def profile_text(vals: np.ndarray, models=MODELS) -> str:
    """12本の値を「本番 a/b/c/d  1.5σ ...  1.8σ ...」の形に（並びは LGB/XGB/CAT/LR）。"""
    idx = {a: i for i, a in enumerate(models)}
    parts = []
    for name, grp in (("本番", BASE4), ("1.5σ", K15), ("1.8σ", K18)):
        parts.append(name + " " + "/".join(f"{vals[idx[a]]:.0f}" for a in grp))
    return "  ".join(parts)


def profile_signed(vals: np.ndarray, models=MODELS) -> str:
    idx = {a: i for i, a in enumerate(models)}
    parts = []
    for name, grp in (("本番", BASE4), ("1.5σ", K15), ("1.8σ", K18)):
        parts.append(name + " " + "/".join(f"{vals[idx[a]]:+.0f}" for a in grp))
    return "  ".join(parts)


def high_low(vals: np.ndarray, models=MODELS, n: int = 2) -> str:
    o = np.argsort(vals)
    lo = ", ".join(short(models[i]) for i in o[:n])
    hi = ", ".join(short(models[i]) for i in o[::-1][:n])
    return f"高い {hi} / 低い {lo}"


# --------------------------------------------------------------------------- #
# 次元削減・地図・塊
# --------------------------------------------------------------------------- #

def pca_fit(Z: np.ndarray, n: int = 6):
    """PCA。第1主成分は負荷量の和が正（みんなが高いほど正）、ほかは絶対値最大の負荷量が正になる向きにそろえる。"""
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(n, Z.shape[1]), random_state=SEED).fit(Z)
    comps = p.components_.copy()
    for i in range(len(comps)):
        s = np.sign(comps[i].sum()) if i == 0 else np.sign(comps[i][np.argmax(np.abs(comps[i]))])
        comps[i] *= s if s != 0 else 1.0
    E = (Z - p.mean_) @ comps.T
    return E, comps, p.explained_variance_ratio_


def tsne_embed(Z: np.ndarray, perplexity: float = 50.0, seed: int = SEED) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(Z)


def grid_edges(x: np.ndarray, n: int = GRID_N, q: float = 0.005) -> np.ndarray:
    """両端 0.5% を外した範囲を n 等分（外れた行は端のマス目に入れる）。"""
    lo, hi = np.nanquantile(x, [q, 1 - q])
    if not hi > lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, n + 1)


def grid_cells(E2: np.ndarray, df: pd.DataFrame, ex: np.ndarray, ey: np.ndarray,
               min_n: int = MIN_CELL) -> tuple[pd.DataFrame, dict]:
    """地図のマス目ごとの件数・平均実収益・平均上昇率・正例率・勝率（%）。min_n 件未満のマス目は出さない。"""
    ix = np.clip(np.searchsorted(ex, E2[:, 0], side="right") - 1, 0, len(ex) - 2)
    iy = np.clip(np.searchsorted(ey, E2[:, 1], side="right") - 1, 0, len(ey) - 2)
    r = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    rise = (pd.to_numeric(df["future_rise"], errors="coerce").to_numpy(dtype=float) * 100
            if "future_rise" in df.columns else np.full(len(df), np.nan))
    g = pd.DataFrame({"ix": ix, "iy": iy, "ret": r * 100, "rise": rise,
                      "pos": pd.to_numeric(df["label_k12"], errors="coerce").to_numpy(dtype=float) * 100,
                      "win": np.where(np.isnan(r), np.nan, (r > 0) * 100.0)})
    agg = (g.groupby(["ix", "iy"])
           .agg(n=("ret", "size"), ret=("ret", "mean"), rise=("rise", "mean"),
                pos=("pos", "mean"), win=("win", "mean"))
           .reset_index())
    shown = agg[agg["n"] >= min_n].reset_index(drop=True)
    hidden = agg[agg["n"] < min_n]
    return shown, {"cells": int(len(shown)), "hidden_cells": int(len(hidden)),
                   "hidden_rows": int(hidden["n"].sum())}


def kmeans_fit(Z: np.ndarray, k: int, seed: int = SEED):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)


def order_map(labels: np.ndarray, key: np.ndarray, k: int) -> np.ndarray:
    """塊の番号を key（合意）の平均の低い順 0..k-1 に付け直す表（元の番号 → 新しい番号）。空の塊は最後。"""
    means = np.array([np.nanmean(key[labels == c]) if np.any(labels == c) else np.inf for c in range(k)])
    order = np.argsort(means, kind="mergesort")
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap


def terciles(x: np.ndarray) -> np.ndarray:
    """三分位（0 / 1 / 2）。"""
    return pd.qcut(pd.Series(x), 3, labels=False, duplicates="drop").to_numpy()


# --------------------------------------------------------------------------- #
# 前向き（過去の窓だけで決めて、次の窓で測る）
# --------------------------------------------------------------------------- #

def window_split(df: pd.DataFrame, folds: list, models=MODELS) -> dict:
    """
    テスト窓ごとに、訓練（それより前の窓。テスト窓の始まりの手前 RISE_HORIZON 営業日ぶんは結果が出ていないので
    外す）と、訓練の分布で付け直した百分位を作る。{窓: (訓練の行番号, テストの行番号, 訓練の百分位, テストの百分位)}
    """
    w = df["window"].to_numpy()
    d = pd.to_datetime(df["Date"]).to_numpy()
    start = {f.index: pd.Timestamp(f.test_start) for f in folds}
    gap = pd.Timedelta(days=int(round(B.RISE_HORIZON * WF.TRADING_TO_CALENDAR)))
    present = sorted(int(x) for x in np.unique(w) if x > 0)
    out = {}
    for ww in present:
        prior = [x for x in present if x < ww]
        if len(prior) < MIN_TRAIN_WINDOWS:
            continue
        cut = np.datetime64(start[ww] - gap)
        tr = np.where(np.isin(w, prior) & (d <= cut))[0]
        te = np.where(w == ww)[0]
        if len(tr) < 200 or len(te) < 20:
            continue
        ref = df.iloc[tr]
        out[ww] = (tr, te, pct_matrix(ref, models, ref=ref), pct_matrix(df.iloc[te], models, ref=ref))
    return out


def _cum_top_means(r: np.ndarray, order: np.ndarray) -> np.ndarray:
    """order の順に上から n 件取ったときの平均実収益（n = 1..）。欠測は飛ばす。"""
    x = r[order]
    ok = ~np.isnan(x)
    cs = np.cumsum(np.where(ok, x, 0.0))
    cn = np.cumsum(ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)


def _placebo_groups(cons_tr: np.ndarray, w_tr: np.ndarray, bins: int = CONS_BINS) -> list:
    q = pd.qcut(pd.Series(cons_tr), bins, labels=False, duplicates="drop").to_numpy()
    groups = []
    for ww in np.unique(w_tr):
        for b in np.unique(q[~pd.isna(q)]):
            g = np.where((w_tr == ww) & (q == b))[0]
            if len(g) > 1:
                groups.append(g)
    return groups


def wf_clusters(df: pd.DataFrame, split: dict, k: int, models=MODELS, n_placebo: int = N_PLACEBO,
                seed: int = SEED) -> dict:
    """
    前向き: 訓練の窓だけで標準化と k-means を作り、訓練の実収益が最も高い塊（MIN_PICK_* 以上）を選んで、テスト窓で
    その塊に入った行を測る。比べる相手は同じ件数を「合意の上位」「lgbm の上位」で選んだとき。
    プラセボ: 訓練の実収益を「同じ窓・合意の同じ分位」の中で並べ替えてから塊を選ぶ（形の情報だけを消す）。
    """
    rng = np.random.default_rng(seed)
    r_all = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    w_all = df["window"].to_numpy()
    i_lgbm = list(models).index(M.BASELINE)
    rows, pick_mask = [], np.zeros(len(df), dtype=bool)
    plc_cons, plc_all = [], []
    for ww, (tr, te, Ptr, Pte) in split.items():
        Ztr, mu, sd = standardize(Ptr)
        Zte = (Pte - mu) / sd
        km = kmeans_fit(Ztr, k, seed)
        ltr, lte = km.labels_, km.predict(Zte)
        rtr, rte = r_all[tr], r_all[te]
        sizes = np.bincount(ltr, minlength=k)
        ok = sizes >= max(MIN_PICK_ROWS, MIN_PICK_SHARE * len(tr))
        if not ok.any():
            continue
        members = [ltr == c for c in range(k)]

        def best_of(rv):
            m = np.array([nmean(rv[members[c]]) if ok[c] else -np.inf for c in range(k)])
            m = np.where(np.isnan(m), -np.inf, m)
            return int(np.argmax(m))

        b = best_of(rtr)
        pick = lte == b
        n_p = int(pick.sum())
        cons_tr, cons_te = Ptr.mean(axis=1), Pte.mean(axis=1)
        cum_cons = _cum_top_means(rte, np.argsort(-cons_te, kind="mergesort"))
        cum_lgbm = _cum_top_means(rte, np.argsort(-Pte[:, i_lgbm], kind="mergesort"))
        ret_all = nmean(rte)
        rec = {"window": int(ww), "n_train": int(len(tr)), "n_test": int(len(te)), "n_pick": n_p,
               "best_train_ret": nmean(rtr[members[b]]) * 100, "best_train_cons": nmean(cons_tr[members[b]]),
               "best_test_cons": nmean(cons_te[pick]) if n_p else np.nan,
               "ret_pick": nmean(rte[pick]) * 100 if n_p else np.nan,
               "ret_cons": cum_cons[n_p - 1] * 100 if n_p else np.nan,
               "ret_lgbm": cum_lgbm[n_p - 1] * 100 if n_p else np.nan,
               "ret_all": ret_all * 100}
        rows.append(rec)
        pick_mask[te[pick]] = True
        # プラセボ
        te_counts = np.bincount(lte, minlength=k)
        te_means = np.array([nmean(rte[lte == c]) if te_counts[c] else np.nan for c in range(k)])
        groups = _placebo_groups(cons_tr, w_all[tr])
        pc, pa = np.full(n_placebo, np.nan), np.full(n_placebo, np.nan)
        for s in range(n_placebo):
            rp = rtr.copy()
            for g in groups:
                rp[g] = rp[rng.permutation(g)]
            bb = best_of(rp)
            nb = int(te_counts[bb])
            if nb:
                pa[s] = (te_means[bb] - ret_all) * 100
                pc[s] = (te_means[bb] - cum_cons[nb - 1]) * 100
        plc_cons.append(pc)
        plc_all.append(pa)
    res = pd.DataFrame(rows)
    out = {"k": k, "rows": res, "pick_mask": pick_mask}
    if len(res):
        d_cons = (res["ret_pick"] - res["ret_cons"]).to_numpy()
        d_all = (res["ret_pick"] - res["ret_all"]).to_numpy()
        d_lgbm = (res["ret_pick"] - res["ret_lgbm"]).to_numpy()
        out.update({"vs_cons": summarize(d_cons), "vs_all": summarize(d_all), "vs_lgbm": summarize(d_lgbm)})
        if n_placebo and plc_cons:
            pm_cons = np.nanmean(np.vstack(plc_cons), axis=0)       # プラセボの1回ごとに窓の平均
            pm_all = np.nanmean(np.vstack(plc_all), axis=0)
            act_c, act_a = out["vs_cons"]["mean"], out["vs_all"]["mean"]
            out["placebo"] = {"n": int(n_placebo),
                              "vs_cons_mean": float(np.nanmean(pm_cons)),
                              "vs_cons_p": float(np.mean(pm_cons >= act_c)) if np.isfinite(act_c) else np.nan,
                              "vs_all_mean": float(np.nanmean(pm_all)),
                              "vs_all_p": float(np.mean(pm_all >= act_a)) if np.isfinite(act_a) else np.nan}
    return out


def wf_types(df: pd.DataFrame, split: dict, models=MODELS, k: int = TYPE_K, n_placebo: int = N_PLACEBO,
             seed: int = SEED) -> dict:
    """
    前向き（合意の上位10% の中の型）: 訓練の合意の上位10% の閾値をテストにも当て、上位の行の「12本から平均を引いた
    形」を k-means で型に分ける。訓練の実収益が最も高い型をテストで測り、テストの上位10% 全体・同じ件数を合意の
    上位で選んだときと比べる。プラセボは訓練の上位の行の実収益を窓の中で並べ替えて型を選ぶ。
    """
    rng = np.random.default_rng(seed + 1)
    r_all = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    w_all = df["window"].to_numpy()
    rows, pick_mask, plc = [], np.zeros(len(df), dtype=bool), []
    for ww, (tr, te, Ptr, Pte) in split.items():
        cons_tr, cons_te = Ptr.mean(axis=1), Pte.mean(axis=1)
        thr = float(np.quantile(cons_tr, TOP_Q))
        Ttr, Tte = cons_tr >= thr, cons_te >= thr
        if Ttr.sum() < 100 or Tte.sum() < MIN_WIN_ROWS:
            continue
        prof_tr = Ptr[Ttr] - Ptr[Ttr].mean(axis=1, keepdims=True)
        prof_te = Pte[Tte] - Pte[Tte].mean(axis=1, keepdims=True)
        km = kmeans_fit(prof_tr, k, seed)
        ltr, lte = km.labels_, km.predict(prof_te)
        rtr, rte = r_all[tr][Ttr], r_all[te][Tte]
        sizes = np.bincount(ltr, minlength=k)
        ok = sizes >= max(MIN_TYPE_ROWS, MIN_TYPE_SHARE * Ttr.sum())
        if not ok.any():
            continue
        members = [ltr == c for c in range(k)]

        def best_of(rv):
            m = np.array([nmean(rv[members[c]]) if ok[c] else -np.inf for c in range(k)])
            m = np.where(np.isnan(m), -np.inf, m)
            return int(np.argmax(m))

        b = best_of(rtr)
        pick = lte == b
        n_p = int(pick.sum())
        cum_cons = _cum_top_means(rte, np.argsort(-cons_te[Tte], kind="mergesort"))
        ret_T = nmean(rte)
        rows.append({"window": int(ww), "n_top_test": int(Tte.sum()), "n_pick": n_p,
                     "best_center": km.cluster_centers_[b].tolist(),
                     "ret_pick": nmean(rte[pick]) * 100 if n_p else np.nan,
                     "ret_top": ret_T * 100,
                     "ret_cons": cum_cons[n_p - 1] * 100 if n_p else np.nan})
        te_idx = te[Tte]
        pick_mask[te_idx[pick]] = True
        te_counts = np.bincount(lte, minlength=k)
        te_means = np.array([nmean(rte[lte == c]) if te_counts[c] else np.nan for c in range(k)])
        wtr = w_all[tr][Ttr]
        groups = [np.where(wtr == x)[0] for x in np.unique(wtr)]
        pa = np.full(n_placebo, np.nan)
        for s in range(n_placebo):
            rp = rtr.copy()
            for g in groups:
                if len(g) > 1:
                    rp[g] = rp[rng.permutation(g)]
            bb = best_of(rp)
            if te_counts[bb]:
                pa[s] = (te_means[bb] - ret_T) * 100
        plc.append(pa)
    res = pd.DataFrame(rows)
    out = {"k": k, "rows": res, "pick_mask": pick_mask}
    if len(res):
        out["vs_top"] = summarize((res["ret_pick"] - res["ret_top"]).to_numpy())
        out["vs_cons"] = summarize((res["ret_pick"] - res["ret_cons"]).to_numpy())
        if n_placebo and plc:
            pm = np.nanmean(np.vstack(plc), axis=0)
            act = out["vs_top"]["mean"]
            out["placebo"] = {"n": int(n_placebo), "vs_top_mean": float(np.nanmean(pm)),
                              "vs_top_p": float(np.mean(pm >= act)) if np.isfinite(act) else np.nan}
    return out


def monthly_picks(sub: pd.DataFrame, key: str, n: int = MONTHLY_N) -> pd.DataFrame:
    """各月の key の上位 n 件（月に n 銘柄買う運用）。"""
    d = sub.copy()
    d["_m"] = pd.to_datetime(d["Date"]).dt.to_period("M")
    d = d.sort_values(["_m", key], ascending=[True, False], kind="mergesort")
    return d.groupby("_m", sort=False).head(n).reset_index(drop=True)


def monthly_compare(df: pd.DataFrame, split: dict, masks: dict, models=MODELS) -> dict:
    """月に2銘柄: 前向きの百分位で、合意の上位2件 / lgbm の上位2件 / 塊・型の中の合意の上位2件。"""
    i_lgbm = list(models).index(M.BASELINE)
    cons = np.full(len(df), np.nan)
    lg = np.full(len(df), np.nan)
    for ww, (tr, te, Ptr, Pte) in split.items():
        cons[te] = Pte.mean(axis=1)
        lg[te] = Pte[:, i_lgbm]
    base = df.assign(_cons=cons, _lgbm=lg)
    base = base[~np.isnan(cons)]
    out = {"合意の上位2件": monthly_picks(base, "_cons"), "lgbm の上位2件": monthly_picks(base, "_lgbm")}
    for name, m in masks.items():
        sel = base[m[~np.isnan(cons)]]
        out[name] = monthly_picks(sel, "_cons")
    return out


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line_stats(name: str, s: dict, extra: str = "") -> str:
    return (f"  {name:<14}{s['n']:>7,}件  実収益 {s['ret']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:5.1f}%  "
            f"+10%以上 {s['hit10']:4.1f}%  −10%未満 {s['bad10']:4.1f}%  上昇率 {s.get('rise', np.nan):5.1f}%  "
            f"正例率 {s.get('pos12', np.nan):4.1f}%{extra}")


def analyze(df: pd.DataFrame, *, info: dict | None = None, folds: list | None = None, k_main: int = K_MAIN,
            ks=KS, wf_ks=WF_KS, tsne: bool = True, perplexity: float = 50.0, n_placebo: int = N_PLACEBO,
            seed: int = SEED, models=MODELS) -> dict:
    """df: 12本のスコア・ラベル・実収益・window（テスト窓の番号）を持つ行。報告を書いて要約を返す。"""
    summary: dict = {"models": list(models), "k_main": k_main}
    df = df[df["window"] > 0].reset_index(drop=True)
    r = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy(dtype=float)
    allst = stats(df)
    summary["all"] = allst

    # ---- 0. データ
    print("\n■ 0. データ（本番の4モデル + ラベル違いの8本の out-of-fold。本番と同じ窓）")
    if info:
        for a in models:
            print(f"  {short(a):<5} {info['rows'].get(a, 0):>7,}件  重複 {info['dups'].get(a, 0)}  "
                  f"スコア欠測 {info['nan_scores'].get(a, 0)}")
        print(f"  12本を (Code, Date) で突き合わせた行 {info['joined']:,} / スコアがそろった行 {info['kept']:,}")
    wn = df.groupby("window")
    print(f"  使う行 {len(df):,}件 / 窓 {df['window'].nunique()}つ / "
          f"{pd.to_datetime(df['Date']).min().date()} 〜 {pd.to_datetime(df['Date']).max().date()}")
    for ww, g in wn:
        print(f"    窓{ww:>2}: {len(g):>5,}件  {pd.to_datetime(g['Date']).min().date()} 〜 "
              f"{pd.to_datetime(g['Date']).max().date()}  実収益 {nmean(g[OUTCOME]) * 100:+.2f}%")
    print(line_stats("全体", allst))
    emit("meta", len(df), df["window"].nunique(), allst["ret"], allst.get("rise", np.nan), allst.get("pos12", np.nan),
         allst["win"])

    # ---- 1. 相関・主成分・合意
    P = pct_matrix(df, models)
    cons = P.mean(axis=1)
    sc = df[[f"score_{a}" for a in models]]
    S = sc.rank().corr().to_numpy()
    print("\n■ 1. 12本のスコアの順位相関（Spearman。全行）")
    print("        " + "".join(f"{short(a):>6}" for a in models))
    for i, a in enumerate(models):
        print(f"  {short(a):<6}" + "".join(f"{S[i, j]:>6.2f}" for j in range(len(models))))
        emit("corr", short(a), *[float(S[i, j]) for j in range(len(models))])
    off = S[np.triu_indices(len(models), 1)]
    print(f"  2本ずつの相関: 最小 {off.min():.2f} / 中央 {np.median(off):.2f} / 最大 {off.max():.2f}")
    summary["corr"] = {"min": float(off.min()), "median": float(np.median(off)), "max": float(off.max())}

    Z, mu, sd = standardize(P)
    E, comps, evr = pca_fit(Z)
    print("\n  主成分（12本の百分位を標準化。寄与率 / 累積）: "
          + "  ".join(f"PC{i + 1} {v * 100:.1f}%" for i, v in enumerate(evr))
          + f"  （累積 PC1〜3 {evr[:3].sum() * 100:.1f}%）")
    print("  負荷量            PC1     PC2     PC3")
    for j, a in enumerate(models):
        print(f"    {short(a):<6}{comps[0][j]:>+10.3f}{comps[1][j]:>+8.3f}{comps[2][j]:>+8.3f}")
        emit("pca_load", short(a), float(comps[0][j]), float(comps[1][j]), float(comps[2][j]))
    for i, v in enumerate(evr):
        emit("pca_var", i + 1, float(v))
    summary["pca"] = {"evr": [float(v) for v in evr],
                      "load": {short(a): [float(comps[i][j]) for i in range(3)] for j, a in enumerate(models)}}

    print("\n  合意（12本の百分位の平均）の十分位ごと（1 = 低い）")
    dec = pd.qcut(pd.Series(cons), 10, labels=False, duplicates="drop").to_numpy()
    summary["decile"] = []
    for d in range(int(np.nanmax(dec)) + 1):
        m = dec == d
        s = stats(df[m])
        print(line_stats(f"十分位{d + 1}（{cons[m].min():.0f}〜{cons[m].max():.0f}）", s))
        emit("decile", d + 1, s["n"], s["ret"], s.get("rise", np.nan), s.get("pos12", np.nan), s["win"],
             float(cons[m].mean()))
        summary["decile"].append(s)
    top = dec == np.nanmax(dec)
    print(f"  最上位十分位 − 窓の全体（窓ごと）: {fmt_s(summarize(window_diffs(df, top)))}")

    # ---- 2. 2次元の地図
    print("\n■ 2. 2次元の地図（マス目 {0}×{0}。{1}件未満のマス目は出さない。@grid の行が手元で描く集計）".format(
        GRID_N, MIN_CELL))
    embeds = {"pca": E[:, :2]}
    if tsne:
        t0 = time.time()
        embeds["tsne"] = tsne_embed(Z, perplexity=perplexity, seed=seed)
        log(f"t-SNE（perplexity {perplexity:g}）{time.time() - t0:.0f}秒")
    summary["grid"] = {}
    for name, E2 in embeds.items():
        ex, ey = grid_edges(E2[:, 0]), grid_edges(E2[:, 1])
        cells, gi = grid_cells(E2, df, ex, ey)
        summary["grid"][name] = gi
        print(f"  {name}: 出すマス目 {gi['cells']} / 出さないマス目 {gi['hidden_cells']}（{gi['hidden_rows']:,}件）")
        emit("edges", name, "x", *[float(v) for v in ex])
        emit("edges", name, "y", *[float(v) for v in ey])
        for _, c in cells.iterrows():
            emit("grid", name, int(c["ix"]), int(c["iy"]), int(c["n"]), float(c["ret"]), float(c["rise"]),
                 float(c["pos"]), float(c["win"]))
        # 地図の上の合意の並び（色の読み方の手がかり）: 軸ごとの相関
        for ax in (0, 1):
            rho = pd.Series(E2[:, ax]).corr(pd.Series(cons), method="spearman")
            print(f"    軸{ax + 1} と合意の順位相関 {rho:+.3f}")
            emit("axis_cons", name, ax + 1, float(rho))

    # 第1主成分の十分位の中で、第2・第3主成分の上下を比べる（同じ合意の高さで「形」が効くか）
    print("\n  第1主成分（≒合意）の十分位の中で、第2・第3主成分の三分位を比べる（実収益 %。下 / 中 / 上、上 − 下）")
    d1 = pd.qcut(pd.Series(E[:, 0]), 10, labels=False, duplicates="drop").to_numpy()
    summary["pc_within"] = {}
    for j in (1, 2):
        t = np.full(len(E), np.nan)
        for d in np.unique(d1):
            m = d1 == d
            t[m] = terciles(E[m, j])
        print(f"   PC{j + 1}:")
        for d in np.unique(d1):
            m = d1 == d
            v = [nmean(r[m & (t == q)]) * 100 for q in range(3)]
            print(f"    PC1 十分位{d + 1:>2}: {v[0]:+6.2f} / {v[1]:+6.2f} / {v[2]:+6.2f}  （上 − 下 {v[2] - v[0]:+.2f}）")
            emit("pc_within", j + 1, int(d) + 1, *v)
        hi_top = (d1 >= 7) & (t == 2)
        lo_top = (d1 >= 7) & (t == 0)
        s = summarize(window_diffs(df, hi_top, base=lo_top))
        print(f"    PC1 の上位3十分位で、PC{j + 1} の上 − 下（窓ごと）: {fmt_s(s)}")
        summary["pc_within"][f"pc{j + 1}"] = s

    # ---- 3. k-means
    print(f"\n■ 3. k-means（12本の百分位を標準化。塊の番号は合意の低い順）")
    resid = resid_within_window(df, cons)
    order_c = np.argsort(-cons, kind="mergesort")
    cum_c = _cum_top_means(r, order_c)
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z), size=min(4000, len(Z)), replace=False)
    print("  k   シルエット  最も実収益の高い塊: 件数 / 合意 / 実収益 / 形の効果（窓ごと） / 同じ件数を合意の上位で")
    summary["ks"] = {}
    fits = {}
    for k in ks:
        km = kmeans_fit(Z, k, seed)
        remap = order_map(km.labels_, cons, k)
        lab_k = remap[km.labels_]
        fits[k] = (km, remap, lab_k)
        sil = float(silhouette_score(Z[sub], lab_k[sub])) if k > 1 else np.nan
        rets = np.array([nmean(r[lab_k == c]) for c in range(k)])
        b = int(np.nanargmax(rets))
        m = lab_k == b
        nb = int(m.sum())
        rs = summarize(window_means(df, resid, m))
        print(f"  {k:>2}  {sil:>8.3f}    {nb:>6,}件 / {cons[m].mean():5.1f} / {rets[b] * 100:+.2f}% / {fmt_s(rs)}"
              f" / {cum_c[nb - 1] * 100:+.2f}%")
        summary["ks"][k] = {"silhouette": sil, "best": int(b), "n": nb, "cons": float(cons[m].mean()),
                            "ret": float(rets[b] * 100), "resid": rs, "cons_top_same_n": float(cum_c[nb - 1] * 100)}
        emit("kbest", k, sil, nb, float(cons[m].mean()), float(rets[b] * 100), rs["mean"], rs["se"],
             float(cum_c[nb - 1] * 100))

    if k_main not in fits:
        km = kmeans_fit(Z, k_main, seed)
        remap = order_map(km.labels_, cons, k_main)
        fits[k_main] = (km, remap, remap[km.labels_])
    km, remap, lab_k = fits[k_main]
    print(f"\n  k = {k_main} の塊（中心は各モデルの百分位の平均。並びは LGB/XGB/CAT/LR）")
    tsne_xy = embeds.get("tsne")
    summary["clusters"] = []
    for c in range(k_main):
        m = lab_k == c
        s = stats(df[m])
        cen = P[m].mean(axis=0)
        rs = summarize(window_means(df, resid, m))
        wd = summarize(window_diffs(df, m))
        same = cum_c[int(m.sum()) - 1] * 100 if m.any() else np.nan
        print(f"  C{c + 1}: 合意 {cons[m].mean():5.1f}  {profile_text(cen)}")
        print(line_stats("", s, extra=f"  1.5σ 正例率 {s.get('pos15', np.nan):4.1f}%  1.8σ {s.get('pos18', np.nan):4.1f}%"))
        print(f"      窓の全体との差 {fmt_s(wd)}  形の効果 {fmt_s(rs)}  同じ件数を合意の上位で {same:+.2f}%")
        tx = float(tsne_xy[m, 0].mean()) if tsne_xy is not None else np.nan
        ty = float(tsne_xy[m, 1].mean()) if tsne_xy is not None else np.nan
        emit("cluster", k_main, c + 1, s["n"], float(cons[m].mean()), s["ret"], s["med"], s["win"], s["bad10"],
             s.get("rise", np.nan), s.get("pos12", np.nan), rs["mean"], rs["se"], rs["wins"], rs["n"],
             float(E[m, 0].mean()), float(E[m, 1].mean()), tx, ty, same)
        emit("centroid", k_main, c + 1, *[float(v) for v in cen])
        summary["clusters"].append({"c": c + 1, **s, "cons": float(cons[m].mean()), "centroid": cen.tolist(),
                                    "resid": rs, "vs_window": wd, "cons_top_same_n": float(same)})

    # ---- 4. 合意の上位10% の中の型
    thr = float(np.quantile(cons, TOP_Q))
    T = cons >= thr
    PT = P[T]
    dT = df[T].reset_index(drop=True)
    rT = r[T]
    consT = cons[T]
    print(f"\n■ 4. 合意の上位10%（合意 {thr:.1f} 以上、{int(T.sum()):,}件）の中の「ずれの型」")
    print(line_stats("上位10% 全体", stats(dT)))
    prof = PT - PT.mean(axis=1, keepdims=True)
    kmT = kmeans_fit(prof, TYPE_K, seed)
    sizes = np.bincount(kmT.labels_, minlength=TYPE_K)
    order_t = np.argsort(-sizes, kind="mergesort")
    remap_t = np.empty(TYPE_K, dtype=int)
    remap_t[order_t] = np.arange(TYPE_K)
    labT = remap_t[kmT.labels_]
    residT = resid[T]
    summary["types"] = []
    for c in range(TYPE_K):
        m = labT == c
        s = stats(dT[m])
        cen = prof[m].mean(axis=0)
        wd = summarize(window_diffs(dT, m))
        rs = summarize(window_means(dT, residT, m, min_rows=5))
        print(f"  型{c + 1}: {int(m.sum()):,}件 合意 {consT[m].mean():5.1f}  {high_low(cen)}")
        print(f"      平均との差 {profile_signed(cen)}")
        print(line_stats("", s))
        print(f"      上位10% 全体との差（窓ごと） {fmt_s(wd)}  形の効果 {fmt_s(rs)}")
        emit("type", c + 1, s["n"], float(consT[m].mean()), s["ret"], s["win"], s["bad10"], s.get("rise", np.nan),
             s.get("pos12", np.nan), wd["mean"], wd["se"], wd["wins"], wd["n"], *[float(v) for v in cen])
        summary["types"].append({"t": c + 1, **s, "cons": float(consT[m].mean()), "profile": cen.tolist(),
                                 "vs_top": wd, "resid": rs})

    idx = {a: i for i, a in enumerate(models)}

    def grp_mean(Pm, grp):
        return Pm[:, [idx[a] for a in grp]].mean(axis=1)

    contrasts = {
        "木 − ロジスティック": grp_mean(PT, TREES) - grp_mean(PT, LOGITS),
        "1.8σ − 1.2σ（4本ずつ）": grp_mean(PT, K18) - grp_mean(PT, BASE4),
        "1.5σ − 1.2σ（4本ずつ）": grp_mean(PT, K15) - grp_mean(PT, BASE4),
        "12本のばらつき（標準偏差）": PT.std(axis=1),
        "12本の最小": PT.min(axis=1),
        "lgbm − 他11本": PT[:, idx[M.BASELINE]] - np.delete(PT, idx[M.BASELINE], axis=1).mean(axis=1),
    }
    print("\n  分かりやすい差の三分位（上位10% の中。実収益 %: 下 / 中 / 上、窓ごとの 上 − 下）")
    summary["contrasts"] = {}
    for name, x in contrasts.items():
        t = terciles(x)
        v = [stats(dT[t == q]) for q in range(3)]
        s = summarize(window_diffs(dT, t == 2, base=(t == 0), min_rows=5))
        cut = np.nanquantile(x, [1 / 3, 2 / 3])
        print(f"  {name:<20} 区切り {cut[0]:+.1f} / {cut[1]:+.1f}: {v[0]['ret']:+.2f} / {v[1]['ret']:+.2f} / "
              f"{v[2]['ret']:+.2f}%  勝率 {v[0]['win']:.0f} / {v[1]['win']:.0f} / {v[2]['win']:.0f}%  "
              f"−10%未満 {v[0]['bad10']:.1f} / {v[1]['bad10']:.1f} / {v[2]['bad10']:.1f}%  上 − 下 {fmt_s(s)}")
        emit("contrast", name, *[v[q]["ret"] for q in range(3)], *[v[q]["win"] for q in range(3)],
             *[v[q]["bad10"] for q in range(3)], s["mean"], s["se"], s["wins"], s["n"])
        summary["contrasts"][name] = {"ret": [v[q]["ret"] for q in range(3)], "hi_lo": s,
                                      "cuts": [float(cut[0]), float(cut[1])]}

    # ---- 5. 前向き
    print(f"\n■ 5. 前向きの確かめ（訓練 = それより前の窓、テスト窓の手前 {B.RISE_HORIZON}営業日ぶんは外す。百分位も訓練の分布で。"
          f"プラセボ {n_placebo}回）")
    split = window_split(df, folds or [], models) if folds else {}
    summary["wf"] = {}
    wf_main = None
    if not split:
        print("  前向きに使える窓がありません")
    else:
        print(f"  テスト窓 {len(split)}つ（{', '.join(str(w) for w in split)}）")
        print("  k   選んだ塊の件数/窓  合意  実収益 − 合意の上位（同じ件数）  − 窓の全体  − lgbm の上位  "
              "プラセボ（形の情報を消す）: 合意の上位との差の平均 / これ以上の割合")
        for k in sorted(set(wf_ks) | {k_main}):
            res = wf_clusters(df, split, k, models, n_placebo=n_placebo, seed=seed)
            rows = res["rows"]
            if not len(rows):
                continue
            pl = res.get("placebo", {})
            print(f"  {k:>2}  {rows['n_pick'].mean():>8.0f}       {rows['best_test_cons'].mean():5.1f}  "
                  f"{fmt_s(res['vs_cons'])}  {fmt_s(res['vs_all'])}  {fmt_s(res['vs_lgbm'])}  "
                  f"{pl.get('vs_cons_mean', np.nan):+.2f}pt / {pl.get('vs_cons_p', np.nan):.2f}")
            emit("wf", k, float(rows["n_pick"].mean()), float(rows["best_test_cons"].mean()),
                 res["vs_cons"]["mean"], res["vs_cons"]["se"], res["vs_cons"]["wins"], res["vs_cons"]["n"],
                 res["vs_all"]["mean"], res["vs_all"]["se"], res["vs_lgbm"]["mean"], res["vs_lgbm"]["se"],
                 pl.get("vs_cons_mean", np.nan), pl.get("vs_cons_p", np.nan))
            summary["wf"][k] = {key: res[key] for key in ("vs_cons", "vs_all", "vs_lgbm") if key in res}
            summary["wf"][k]["placebo"] = pl
            summary["wf"][k]["n_pick_mean"] = float(rows["n_pick"].mean())
            if k == k_main:
                wf_main = res
        if wf_main is not None and len(wf_main["rows"]):
            print(f"\n  k = {wf_main['k']} の窓ごと（実収益 %）")
            for _, x in wf_main["rows"].iterrows():
                print(f"    窓{int(x['window']):>2}: 選んだ塊 {int(x['n_pick']):>4}件（訓練の合意 {x['best_train_cons']:.0f}・"
                      f"訓練の実収益 {x['best_train_ret']:+.2f}%）  テスト {x['ret_pick']:+.2f}%  合意の上位 "
                      f"{x['ret_cons']:+.2f}%  lgbm の上位 {x['ret_lgbm']:+.2f}%  全体 {x['ret_all']:+.2f}%")
        wt = wf_types(df, split, models, n_placebo=n_placebo, seed=seed)
        if len(wt["rows"]):
            pl = wt.get("placebo", {})
            print(f"\n  上位10% の中の型（k = {TYPE_K}）: 選んだ型 {wt['rows']['n_pick'].mean():.0f}件/窓  "
                  f"− 上位10% 全体 {fmt_s(wt['vs_top'])}  − 同じ件数を合意の上位で {fmt_s(wt['vs_cons'])}  "
                  f"プラセボ {pl.get('vs_top_mean', np.nan):+.2f}pt / これ以上 {pl.get('vs_top_p', np.nan):.2f}")
            for _, x in wt["rows"].iterrows():
                cen = np.asarray(x["best_center"], dtype=float)
                print(f"    窓{int(x['window']):>2}: 型 {int(x['n_pick']):>3}件（{high_low(cen)}）  {x['ret_pick']:+.2f}%  "
                      f"上位10% 全体 {x['ret_top']:+.2f}%  合意の上位 {x['ret_cons']:+.2f}%")
            emit("wf_type", float(wt["rows"]["n_pick"].mean()), wt["vs_top"]["mean"], wt["vs_top"]["se"],
                 wt["vs_top"]["wins"], wt["vs_top"]["n"], wt["vs_cons"]["mean"], wt["vs_cons"]["se"],
                 pl.get("vs_top_mean", np.nan), pl.get("vs_top_p", np.nan))
            summary["wf_type"] = {key: wt[key] for key in ("vs_top", "vs_cons") if key in wt}
            summary["wf_type"]["placebo"] = pl

        # ---- 6. 月に2銘柄
        masks = {}
        if wf_main is not None:
            masks[f"塊（k={wf_main['k']}）の中で合意の上位2件"] = wf_main["pick_mask"]
        if len(wt["rows"]):
            masks["上位10% の型の中で合意の上位2件"] = wt["pick_mask"]
        mp = monthly_compare(df, split, masks, models)
        print(f"\n■ 6. 月に{MONTHLY_N}銘柄（テスト窓 {len(split)}つ・前向きの百分位）")
        base_name = "合意の上位2件"
        base_by_w = mp[base_name].groupby("window")[OUTCOME].mean()
        summary["monthly"] = {}
        for name, picks in mp.items():
            s = stats(picks)
            months = int(picks["_m"].nunique()) if len(picks) else 0
            if name != base_name and len(picks):
                cur = picks.groupby("window")[OUTCOME].mean()
                j = base_by_w.index.intersection(cur.index)
                dd = summarize(((cur[j] - base_by_w[j]) * 100).to_numpy())
                extra = f"  合意の上位2件との差 {fmt_s(dd)}"
            else:
                dd, extra = None, ""
            print(f"  {name}（{months}か月）")
            print(line_stats("", s, extra=f"  40営業日 {s.get('ret40', np.nan):+.2f}%" + extra))
            summary["monthly"][name] = {**s, "months": months, "vs_cons": dd}
            emit("monthly", name, s["n"], months, s["ret"], s["med"], s["win"], s["hit10"], s["bad10"],
                 s.get("ret40", np.nan), s.get("pos12", np.nan),
                 dd["mean"] if dd else np.nan, dd["se"] if dd else np.nan)

    # ---- まとめ
    print("\n■ まとめ（数字だけ。読み方は docs/MODEL_ADOPTION_RULES.md §35）")
    print(f"  12本の相関 中央 {summary['corr']['median']:.2f}（最小 {summary['corr']['min']:.2f}）。"
          f"PC1 {summary['pca']['evr'][0] * 100:.1f}% / PC2 {summary['pca']['evr'][1] * 100:.1f}% / "
          f"PC3 {summary['pca']['evr'][2] * 100:.1f}%")
    dtop = summary["decile"][-1]
    print(f"  合意の最上位十分位: 実収益 {dtop['ret']:+.2f}%（全体 {allst['ret']:+.2f}%）、上昇率 {dtop.get('rise', np.nan):.1f}%"
          f"（全体 {allst.get('rise', np.nan):.1f}%）、正例率 {dtop.get('pos12', np.nan):.1f}%（全体 {allst.get('pos12', np.nan):.1f}%）")
    for j in ("pc2", "pc3"):
        if j in summary.get("pc_within", {}):
            print(f"  PC1 の上位3十分位で {j.upper()} の上 − 下: {fmt_s(summary['pc_within'][j])}")
    best = max(summary["clusters"], key=lambda c: c["ret"] if np.isfinite(c["ret"]) else -np.inf)
    print(f"  k = {k_main} で実収益の最も高い塊 C{best['c']}: {best['n']:,}件・合意 {best['cons']:.1f}・実収益 {best['ret']:+.2f}%"
          f"・形の効果 {fmt_s(best['resid'])}・同じ件数を合意の上位で {best['cons_top_same_n']:+.2f}%")
    if summary.get("wf"):
        for k, v in summary["wf"].items():
            pl = v.get("placebo", {})
            print(f"  前向き k = {k}: 選んだ塊 − 合意の上位 {fmt_s(v['vs_cons'])}（プラセボ {pl.get('vs_cons_mean', np.nan):+.2f}pt、"
                  f"これ以上 {pl.get('vs_cons_p', np.nan):.2f}）")
    if summary.get("wf_type"):
        v = summary["wf_type"]
        print(f"  前向き 上位10% の型: − 上位10% 全体 {fmt_s(v['vs_top'])}（プラセボ {v['placebo'].get('vs_top_mean', np.nan):+.2f}pt、"
              f"これ以上 {v['placebo'].get('vs_top_p', np.nan):.2f}）")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験79: 12モデルの予測値の傾向分析（クラスタリング）")
    ap.add_argument("--k", type=int, default=K_MAIN, help="塊の表を詳しく出す k")
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--perplexity", type=float, default=50.0)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info = load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info['kept']:,}件")
    fr = lab.frame()
    df = attach_outcomes(wide, fr)
    folds = make_windows(fr["Date"])
    df["window"] = assign_windows(df["Date"], folds)
    n0 = int((df["window"] == 0).sum())
    if n0:
        log(f"どのテスト窓にも入らない行 {n0:,}件（外す）")
    summary = analyze(df, info=info, folds=folds, k_main=args.k, tsne=not args.no_tsne,
                      perplexity=args.perplexity, n_placebo=args.placebo)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e79_summary.json")

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
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
