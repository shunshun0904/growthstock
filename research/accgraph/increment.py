#!/usr/bin/env python3
"""
EDINET の明細を足した上積みを、2段構えで測る。

## なぜ2段にするのか

切り分け（`diagnose.py`、2026-09-22）で次の2つが分かった。

  - 明細ありの群（106社）だけで学習すると件数が足りず、J-Quants の
    ノードだけでも AUC が 0.50 前後まで落ちる
  - 全体で学習して明細ノードを混ぜると、明細は9割以上が欠測のまま
    学習される。ロジットでは明細ありの群の AUC がかえって下がった
    （latest_jq 0.536 -> latest 0.508）

どちらの測り方でも、明細の上積みは取り出せない。そこで役割を分ける。

  1段目  J-Quants のノードだけで、全体（約5.7万件）で学習する。
         土台の信号はここで決まる
  2段目  明細ありの群だけで、1段目の予測を**固定したまま**、明細の
         特徴量で補正する分だけを学習する

2段目が学ぶのは補正の分だけなので、群が小さくても学習する量が少ない。
正則化を強めると補正はゼロに縮み、1段目の予測そのものに戻る
（一様な予測に戻るのではない）。明細に情報が無ければ、害は小さく収まる。

## 比べるもの

同じ2段目の仕組みで、明細の特徴量を**入れる版と入れない版**を作り、
明細ありの群のテスト窓で AUC を比べる。入れない版は「群に合わせて
1段目を較正し直しただけ」のもので、較正し直しの効果と明細の効果を分ける
対照になる。

## 時間の順序

1段目の予測は、訓練にも使う行について必ずアウトオブサンプルでなければ
ならない。訓練に使った行の予測は当たりすぎていて、2段目がそれを
信じすぎるように学ぶ。

ここでは1段目を、テスト窓を1年ずつ進める walk-forward で最初の2年から
回し、各行に「その行より前だけで学習したモデルの予測」を付ける。
2段目のフォールド k は、テスト窓 k より前（purge / embargo 済み）の
明細ありの行だけで学習する。未来の情報はどこにも入らない。

## 判定基準（結果を見る前に決めておく）

明細ありの群のテスト窓で、差 = AUC(明細あり版) − AUC(明細なし版) を取り、

  明細が効く      差の95%区間が 0 を上回り、かつ偽の明細より大きい
  明細が害になる  差の95%区間が 0 を下回り、かつ偽の明細より小さい
  差が見えない    それ以外

区間は発表日単位のブートストラップで出す。これはテスト行の引き直しだけで、
「別の訓練データなら別の補正になる」揺れを含まない。そこで明細の特徴量を
明細ありの行の間で入れ替えた**偽の明細**で2段目を学習し直し（19回）、
本物の差が偽の差をすべて上回るか（片側 p = 0.05）も併せて見る。
合成データ（明細に情報が無い）で、区間だけだと「効く」と出た例があった。

判定に使うのは明細の特徴量を絞った版（compact）だけ。
全部入れた版（full）は参考として並べ、偽の明細は回さない。

  python3 research/accgraph/increment.py
  -> docs/ACCGRAPH_INCREMENT.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, RESEARCH)

from accgraph import baselines, build, diagnose, schema, splits  # noqa: E402

OUT_MD = os.path.join(ROOT, "docs", "ACCGRAPH_INCREMENT.md")
OUT_JSON = os.path.join(build.OUT_DIR, "increment.json")
CLASSES = np.array([0, 1, 2])

#: 1段目。切り分けで使ったものと同じ（明細ありの群で 0.536 / 0.522）
STAGE1_FEATURES = "latest_jq"
STAGE1_MODELS = ("logit", "lgbm")

#: 1段目の walk-forward の最初の訓練期間。2段目の訓練に使える行を
#: 増やすため、評価（48か月）より短くして前から予測を付けておく
STAGE1_MIN_TRAIN_MONTHS = 24

#: 明細の特徴量を絞った版で使うノード特徴量。年1回の書類なので、
#: 四半期の変化・傾き・ばらつき（qoq_sym / slope4 / vol4）は大半が0、
#: span は定数、会社予想（fcst_*）は J-Quants 側にしか無い。
#: log_size / scaled は規模で、1段目がすでに持っている
COMPACT_NODE_FEATURES = ("to_sales", "to_assets", "yoy_sym", "sign", "is_missing")
#: エッジは比率だけ。符号の一致と両端の有無は、ノードの sign / is_missing と重なる
COMPACT_EDGE_FEATURES = ("ratio",)

#: 補正の強さの候補。大きいほど補正がゼロ（=1段目のまま）に縮む。
#: 最大値は、明細に情報が無いときに実質ゼロへ潰せる大きさにしておく
LAMBDA_GRID = (10.0, 1.0, 0.3, 0.1, 0.03, 0.01)
#: 補正の強さを選ぶための、訓練行の中の時系列ブロック数
LAMBDA_CV_BLOCKS = 4

#: 標準化した明細の特徴量の上下限。比率は裾が重く、1件の外れ値で補正が振れる
CLIP_Z = 5.0

#: 2段目の訓練に要る最小件数。これを割るフォールドは比較から外す
MIN_STAGE2_ROWS = 300

#: 偽の明細で学習し直す回数。本物が19回すべてを上回れば片側 p = 0.05
N_PLACEBO = 19


# --------------------------------------------------------------------------- #
# 明細の特徴量
# --------------------------------------------------------------------------- #

def _is_edinet_node(j: int) -> bool:
    return schema.NODES[j].source_table == "edinet"


def _edge_touches_edinet(e) -> bool:
    return (_is_edinet_node(schema.NODE_INDEX[e.src])
            or _is_edinet_node(schema.NODE_INDEX[e.dst]))


def edinet_block(node_feat: np.ndarray, edge_feat: np.ndarray,
                 compact: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    当該四半期（t=0）の明細ノードと、明細ノードを端点に持つエッジの特徴量。

    `latest` から `latest_jq` を引いたものと同じ範囲を取る。compact は
    上の COMPACT_* に絞り、書類の古さ（age_years）を1列だけ足す。
    age_years は明細ノードの間で同じ値なので、全ノードぶん持つと重複する。
    """
    node_idx = [j for j in range(len(schema.NODES)) if _is_edinet_node(j)]
    edge_idx = [i for i, e in enumerate(schema.EDGES) if _edge_touches_edinet(e)]
    nfeat = COMPACT_NODE_FEATURES if compact else tuple(schema.NODE_FEATURES)
    efeat = COMPACT_EDGE_FEATURES if compact else tuple(schema.EDGE_FEATURES)
    nf_pos = [schema.NODE_FEATURES.index(f) for f in nfeat]
    ef_pos = [schema.EDGE_FEATURES.index(f) for f in efeat]

    parts, names = [], []
    nf = node_feat[:, 0][:, node_idx][:, :, nf_pos]
    parts.append(nf.reshape(len(nf), -1))
    names += [f"{schema.NODE_IDS[j]}.{f}" for j in node_idx for f in nfeat]
    ef = edge_feat[:, 0][:, edge_idx][:, :, ef_pos]
    parts.append(ef.reshape(len(ef), -1))
    names += [f"{schema.EDGES[i].src}->{schema.EDGES[i].dst}.{f}"
              for i in edge_idx for f in efeat]
    if compact:
        age = node_feat[:, 0, node_idx[0], schema.NODE_FEATURES.index("age_years")]
        parts.append(age[:, None])
        names.append("edinet.age_years")
    X = np.concatenate([p.astype(np.float64) for p in parts], axis=1)
    assert X.shape[1] == len(names)
    return X, names


def standardize(Z_tr: np.ndarray, Z_te: np.ndarray, clip: float = CLIP_Z
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    訓練行の平均と標準偏差で標準化し、欠測は0（=訓練の平均）で埋める。

    欠測かどうかは is_missing が別に持っている。訓練で分散が無い列は
    情報が無いので0にする（テストで値があっても使わない）。
    """
    mu = np.nanmean(np.where(np.isfinite(Z_tr), Z_tr, np.nan), axis=0)
    sd = np.nanstd(np.where(np.isfinite(Z_tr), Z_tr, np.nan), axis=0)
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12)
    mu = np.where(ok, mu, 0.0)
    sd = np.where(ok, sd, 1.0)

    def f(Z):
        out = (Z - mu) / sd
        out = np.where(np.isfinite(out), out, 0.0)
        out[:, ~ok] = 0.0
        return np.clip(out, -clip, clip)

    return f(Z_tr), f(Z_te)


# --------------------------------------------------------------------------- #
# 2段目: 1段目の予測を固定した多クラスロジスティック回帰
# --------------------------------------------------------------------------- #

def _log_softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    return a - np.log(np.exp(a).sum(axis=1, keepdims=True))


def fit_offset_logit(Z: np.ndarray, y: np.ndarray, offset: np.ndarray,
                     lam: float, classes: np.ndarray = CLASSES
                     ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    logit = offset + Z W + b を学ぶ。offset（1段目の対数確率）の係数は1で固定。

    損失は平均交差エントロピー + lam/2 * |W|^2。切片 b は正則化しない
    （群のクラス比に合わせるだけなので、縮める理由が無い）。
    W を0に縮めると 1段目の予測（+ 切片）に戻る。

    Z の列が0本なら切片だけを学ぶ。明細を入れない対照はこれ。

    W = U / sqrt(lam) と置き直して U について解く。そのままだと lam が
    大きいとき W の方向だけ曲率が桁違いに大きくなり、切片が収束する前に
    最適化が止まる（lam = 1e6 で予測が 0.02 ずれた）。
    """
    from scipy.optimize import minimize

    n, d = Z.shape
    K = len(classes)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    s = 1.0 / np.sqrt(lam)
    Zs = Z * s

    def obj(theta):
        U = theta[: d * K].reshape(d, K)
        b = theta[d * K:]
        logp = _log_softmax(offset + Zs @ U + b)
        loss = -(Y * logp).sum() / n + 0.5 * float((U ** 2).sum())
        G = (np.exp(logp) - Y) / n
        gU = Zs.T @ G + U
        return loss, np.concatenate([gU.ravel(), G.sum(axis=0)])

    res = minimize(obj, np.zeros(d * K + K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    W = res.x[: d * K].reshape(d, K) * s
    b = res.x[d * K:]

    def predict(Z_new: np.ndarray, offset_new: np.ndarray) -> np.ndarray:
        return np.exp(_log_softmax(offset_new + Z_new @ W + b))

    return predict


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def choose_lambda(Z: np.ndarray, y: np.ndarray, offset: np.ndarray,
                  dates: pd.Series, ready: pd.Series,
                  grid: Sequence[float] = LAMBDA_GRID,
                  n_blocks: int = LAMBDA_CV_BLOCKS,
                  min_rows: int = 50) -> Tuple[float, Dict[str, float]]:
    """
    補正の強さを、訓練行の中だけの前向き検証で選ぶ。

    訓練行を発表日順に n_blocks 個に分け、ブロック b を検証、それより前
    （ラベルが検証の開始前に確定したもの）を訓練にして対数損失を測る。
    テスト窓は一切見ない。同点なら強い正則化（=1段目に近い方）を採る。
    検証が1本も組めなければ最大値を返す。
    """
    d = pd.to_datetime(dates).to_numpy()
    r = pd.to_datetime(ready).fillna(pd.Timestamp.max).to_numpy()
    # 行数で等分し、同じ日の行は同じブロックに入れる
    uniq, counts = np.unique(d, return_counts=True)
    cum = np.cumsum(counts) / counts.sum()
    block_of_date = np.minimum((cum * n_blocks - 1e-9).astype(int), n_blocks - 1)
    block = block_of_date[np.searchsorted(uniq, d)]

    losses = {float(g): [] for g in grid}
    for b in range(1, n_blocks):
        va = block == b
        if not va.any():
            continue
        start = d[va].min()
        tr = (block < b) & (r < start)
        if tr.sum() < min_rows or len(np.unique(y[va])) < len(CLASSES):
            continue
        for g in grid:
            f = fit_offset_logit(Z[tr], y[tr], offset[tr], g)
            losses[float(g)].append(_logloss(y[va], f(Z[va], offset[va])))
    mean = {g: (float(np.mean(v)) if v else float("nan")) for g, v in losses.items()}
    valid = {g: v for g, v in mean.items() if np.isfinite(v)}
    if not valid:
        return float(max(grid)), mean
    best = min(valid.values())
    # 同点（1e-6 以内）なら大きい方
    lam = max(g for g, v in valid.items() if v <= best + 1e-6)
    return lam, mean


# --------------------------------------------------------------------------- #
# 2段構え
# --------------------------------------------------------------------------- #

def two_stage(y: np.ndarray, edinet: np.ndarray, p1: np.ndarray,
              Z: np.ndarray, dates: pd.Series, ready: pd.Series,
              outer_tr: List[np.ndarray], outer_te: List[np.ndarray],
              min_stage2_rows: int = MIN_STAGE2_ROWS,
              grid: Sequence[float] = LAMBDA_GRID) -> Dict:
    """
    外側のフォールドごとに2段目を学習し、明細ありのテスト行に予測を付ける。

    p1 は1段目のアウトオブサンプル予測（無い行は NaN）。
    2段目の訓練は「明細あり かつ p1 がある かつ 外側の訓練窓」の行。
    外側の訓練窓は purge / embargo 済みなので、テスト窓の価格は入らない。

    返り値の preds は raw（1段目のまま）/ base（明細なしの2段目）/
    edinet（明細ありの2段目）。比較から外したフォールドの行は NaN。
    """
    n = len(y)
    has_p1 = np.isfinite(p1).all(axis=1)
    offset = np.log(np.clip(np.where(has_p1[:, None], p1, 1.0 / 3), 1e-6, None))
    preds = {k: np.full((n, len(CLASSES)), np.nan) for k in ("raw", "base", "edinet")}
    folds = []
    for k, (tr, te) in enumerate(zip(outer_tr, outer_te)):
        tr2 = tr & edinet & has_p1
        te2 = te & edinet & has_p1
        info = {"fold": k, "n_train": int(tr2.sum()), "n_test": int(te2.sum()),
                "used": False, "lambda": None, "cv_loss": None}
        if tr2.sum() < min_stage2_rows or te2.sum() == 0 \
                or len(np.unique(y[tr2])) < len(CLASSES):
            folds.append(info)
            continue
        Ztr, Zte = standardize(Z[tr2], Z[te2])
        lam, cv = choose_lambda(Ztr, y[tr2], offset[tr2],
                                dates[tr2], ready[tr2], grid=grid)
        empty_tr = np.zeros((int(tr2.sum()), 0))
        empty_te = np.zeros((int(te2.sum()), 0))
        f_base = fit_offset_logit(empty_tr, y[tr2], offset[tr2], lam)
        f_ed = fit_offset_logit(Ztr, y[tr2], offset[tr2], lam)
        preds["raw"][te2] = p1[te2]
        preds["base"][te2] = f_base(empty_te, offset[te2])
        preds["edinet"][te2] = f_ed(Zte, offset[te2])
        info.update(used=True, **{"lambda": lam, "cv_loss": cv})
        folds.append(info)
    return {"preds": preds, "folds": folds}


def bootstrap_models(y: np.ndarray, preds: Dict[str, np.ndarray],
                     dates: pd.Series, pairs: List[Tuple[str, str]],
                     n_boot: int = 1000, seed: int = 0) -> Dict:
    """
    同じ行に付けた複数の予測の AUC と、その差の区間。

    差は同じ引き直しの上で取る（対になった比較）。引き直しは発表日単位。
    """
    rng = np.random.default_rng(seed)
    _, blocks = diagnose._date_blocks(dates)
    point = {k: diagnose.macro_auc(y, p) for k, p in preds.items()}
    draws = {k: [] for k in preds}
    diffs = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        vals = {k: diagnose.macro_auc(y[idx], p[idx]) for k, p in preds.items()}
        for k, v in vals.items():
            if np.isfinite(v):
                draws[k].append(v)
        for a, b in pairs:
            if np.isfinite(vals[a]) and np.isfinite(vals[b]):
                diffs[f"{a}-{b}"].append(vals[a] - vals[b])

    def ci(v):
        if len(v) < n_boot * 0.5:
            return [float("nan"), float("nan")]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    return {
        "auc": {k: {"point": point[k], "ci": ci(draws[k])} for k in preds},
        "diff": {key: {"point": point[key.split("-")[0]] - point[key.split("-")[1]],
                       "ci": ci(v)} for key, v in diffs.items()},
    }


def placebo_diffs(y: np.ndarray, edinet: np.ndarray, p1: np.ndarray,
                  Z: np.ndarray, dates: pd.Series, ready: pd.Series,
                  outer_tr: List[np.ndarray], outer_te: List[np.ndarray],
                  ok: np.ndarray, n: int = N_PLACEBO, seed: int = 0,
                  min_stage2_rows: int = MIN_STAGE2_ROWS) -> List[float]:
    """
    明細の特徴量を明細ありの行の間で入れ替えて2段目を学習し直し、
    AUC(明細あり版) − AUC(明細なし版) を n 回ぶん返す。

    入れ替えで明細とラベルの対応は切れるが、値の分布はそのまま残る。
    本物の差がこれを上回らなければ、補正の学習の揺れと区別できない。
    評価する行（ok）は本物と同じにそろえる。
    """
    rng = np.random.default_rng(seed + 1)
    rows = np.flatnonzero(edinet)
    out = []
    for _ in range(n):
        Zp = Z.copy()
        Zp[rows] = Z[rng.permutation(rows)]
        ts = two_stage(y, edinet, p1, Zp, dates, ready, outer_tr, outer_te,
                       min_stage2_rows=min_stage2_rows)
        pe, pb = ts["preds"]["edinet"][ok], ts["preds"]["base"][ok]
        out.append(diagnose.macro_auc(y[ok], pe) - diagnose.macro_auc(y[ok], pb))
    return out


def permutation_p(observed: float, placebo: Sequence[float],
                  side: str = "greater") -> float:
    """偽の明細の差に対する片側 p 値。(1 + 本物以上の回数) / (1 + 回数)。"""
    v = np.asarray([x for x in placebo if np.isfinite(x)], dtype=float)
    if len(v) == 0 or not np.isfinite(observed):
        return float("nan")
    hits = (v >= observed).sum() if side == "greater" else (v <= observed).sum()
    return float((1 + hits) / (1 + len(v)))


def verdict(diff_ci: Sequence[float], p_greater: Optional[float] = None,
            p_less: Optional[float] = None, alpha: float = 0.05
            ) -> Tuple[str, str]:
    """
    事前に決めた基準で判定する。

    p_greater / p_less は偽の明細に対する片側 p 値。None なら区間だけで
    判定する（参考として並べる full 用）。
    """
    lo, hi = diff_ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "判定できない", "区間を出せるだけの件数が無い"
    if lo > 0.0:
        if p_greater is None or p_greater <= alpha:
            return "明細が効く", "明細を入れた版の AUC が、入れない版を有意に上回る"
        return "差が見えない", (f"区間は0を上回るが、偽の明細でも同じくらいの差が出る"
                             f"（p = {p_greater:.2f}）。補正の学習の揺れと区別できない")
    if hi < 0.0:
        if p_less is None or p_less <= alpha:
            return "明細が害になる", "明細を入れた版の AUC が、入れない版を有意に下回る"
        return "差が見えない", (f"区間は0を下回るが、偽の明細でも同じくらいの差が出る"
                             f"（p = {p_less:.2f}）。補正の学習の揺れと区別できない")
    return "差が見えない", (f"差の95%区間が0をまたぐ。上積みがあるとしても "
                         f"{hi:+.3f} 程度まで")


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def stage1_oof(X: np.ndarray, y: np.ndarray, model: str,
               folds_tr: List[np.ndarray], folds_te: List[np.ndarray],
               seed: int = 0) -> np.ndarray:
    """1段目。全体で学習し、各テスト窓に確率を付ける（それ以外は NaN）。"""
    out = np.full((len(y), len(CLASSES)), np.nan)
    fit = baselines.MODELS[model]
    for tr, te in zip(folds_tr, folds_te):
        m = fit(X[tr], y[tr], CLASSES, seed=seed)
        out[te] = baselines.align_proba(m, X[te], CLASSES)
    return out


def run(data_dir: str = build.OUT_DIR, *, benchmark: str = "topix",
        horizon: int = 20, max_horizon: int = 20,
        models=STAGE1_MODELS, n_boot: int = 1000,
        min_train_months: int = 48,
        stage1_min_train_months: int = STAGE1_MIN_TRAIN_MONTHS,
        test_months: int = 12, step_months: int = 12,
        min_test_rows: int = 200, min_stage2_rows: int = MIN_STAGE2_ROWS,
        n_placebo: int = N_PLACEBO, seed: int = 0) -> Dict:
    gap = min_train_months - stage1_min_train_months
    if gap < 0 or gap % step_months:
        raise SystemExit(
            "1段目の最初の訓練期間は、評価のものより step_months の整数倍だけ"
            f"短くしてください（今は {stage1_min_train_months} と {min_train_months}、"
            f"刻み {step_months}）。ずれると評価のテスト窓に1段目の予測が付かない")
    meta, nf, ef, pm = build.load(data_dir, liquid_only=True)
    if "has_edinet" not in meta.columns:
        raise SystemExit("has_edinet がありません。build.py を回し直してください")
    y_col = f"y_{benchmark}_{horizon}d"
    usable = meta[y_col].to_numpy() >= 0
    meta = meta[usable].reset_index(drop=True)
    nf, ef, pm = nf[usable], ef[usable], pm[usable]
    y = meta[y_col].to_numpy().astype(int)
    edinet = meta["has_edinet"].to_numpy().astype(bool)
    dates, ready = meta["entry_date"], meta["label_ready_date"]
    print(f"[inc] 全体 {len(meta):,}件 / 明細あり {int(edinet.sum()):,}件 "
          f"({meta.loc[edinet, 'Code'].nunique():,}社)")

    wf = dict(test_months=test_months, step_months=step_months,
              embargo_trading_days=max_horizon, min_test_rows=min_test_rows)
    # 評価のフォールドは切り分けと同じ（最初の訓練48か月）
    outer, outer_tr, outer_te = splits.walk_forward(
        meta, min_train_months=min_train_months, **wf)
    # 1段目は24か月から回し、2段目の訓練行にも予測を付ける。
    # 月の刻みが同じなので、48か月以降のテスト窓は評価のものと一致する
    inner, inner_tr, inner_te = splits.walk_forward(
        meta, min_train_months=stage1_min_train_months, **wf)
    inner_keys = {f.test_start for f in inner}
    missing = [f.test_start for f in outer if f.test_start not in inner_keys]
    if missing:
        raise SystemExit(f"1段目のフォールドが評価のテスト窓を覆っていません: {missing}")
    print("[inc] 評価の分割\n" + splits.describe(outer))
    print(f"[inc] 1段目の分割: {len(inner)}本（{inner[0].test_start} から）")

    X1, _ = baselines.flatten(nf, ef, pm, meta, kind=STAGE1_FEATURES)
    variants = {"compact": edinet_block(nf, ef, compact=True),
                "full": edinet_block(nf, ef, compact=False)}
    for v, (Z, names) in variants.items():
        print(f"[inc] 明細の特徴量 {v}: {len(names)}列")

    in_outer_test = np.zeros(len(meta), dtype=bool)
    for m in outer_te:
        in_outer_test |= m

    results = {}
    pairs = [("edinet", "base"), ("base", "raw"), ("edinet", "raw")]
    for mname in models:
        p1 = stage1_oof(X1, y, mname, inner_tr, inner_te, seed=seed)
        has = np.isfinite(p1).all(axis=1)
        print(f"[inc] 1段目 {mname}: 予測が付いた行 {int(has.sum()):,} "
              f"（うち明細あり {int((has & edinet).sum()):,}）")
        for v, (Z, names) in variants.items():
            ts = two_stage(y, edinet, p1, Z, dates, ready, outer_tr, outer_te,
                           min_stage2_rows=min_stage2_rows)
            ok = np.isfinite(ts["preds"]["edinet"]).all(axis=1)
            res = bootstrap_models(
                y[ok], {k: p[ok] for k, p in ts["preds"].items()},
                dates[ok], pairs, n_boot=n_boot, seed=seed)
            key = f"{mname}/{v}"
            obs = res["diff"]["edinet-base"]["point"]
            if v == "compact" and n_placebo > 0:
                plc = placebo_diffs(y, edinet, p1, Z, dates, ready,
                                    outer_tr, outer_te, ok, n=n_placebo,
                                    seed=seed, min_stage2_rows=min_stage2_rows)
                pg, pl = permutation_p(obs, plc, "greater"), permutation_p(obs, plc, "less")
            else:
                plc, pg, pl = [], None, None
            v_label, why = verdict(res["diff"]["edinet-base"]["ci"], pg, pl)
            res.update(verdict=v_label, why=why, folds=ts["folds"],
                       n_test=int(ok.sum()),
                       codes=int(meta.loc[ok, "Code"].nunique()),
                       n_features=len(names), placebo=plc,
                       p_greater=pg, p_less=pl)
            results[key] = res
            a, dd = res["auc"], res["diff"]["edinet-base"]
            print(f"  {mname:<6} {v:<8} raw {a['raw']['point']:.4f} / "
                  f"base {a['base']['point']:.4f} / edinet {a['edinet']['point']:.4f}"
                  f"  差 {dd['point']:+.4f} [{dd['ci'][0]:+.3f}, {dd['ci'][1]:+.3f}]"
                  + (f" 偽の明細 最大 {max(plc):+.4f} / p {pg:.2f}" if plc else "")
                  + f" -> {v_label}")

    return {
        "benchmark": benchmark, "horizon": horizon, "n_boot": n_boot,
        "stage1_features": STAGE1_FEATURES,
        "stage1_min_train_months": stage1_min_train_months,
        "folds": [f.to_dict() for f in outer],
        "n_edinet_rows": int(edinet.sum()),
        "n_edinet_codes": int(meta.loc[edinet, "Code"].nunique()),
        "n_edinet_test_rows": int((edinet & in_outer_test).sum()),
        "features": {v: names for v, (_, names) in variants.items()},
        "lambda_grid": list(LAMBDA_GRID),
        "n_placebo": n_placebo,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def _ci(c) -> str:
    if c is None or any(x is None or not np.isfinite(x) for x in c):
        return "—"
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def _auc_ci(c) -> str:
    if c is None or any(x is None or not np.isfinite(x) for x in c):
        return "—"
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def to_markdown(d: Dict) -> str:
    lines = [
        "# 会計フローグラフ — EDINET の明細の上積み（2段構え）",
        "",
        "`research/accgraph/increment.py` の出力。**実行した結果のみ**を載せる。",
        "",
        "## やったこと",
        "",
        "明細ありの群だけで学習すると件数が足りず、全体に混ぜると明細は9割以上が",
        "欠測のまま学習される（`docs/ACCGRAPH_DIAGNOSE.md`）。そこで役割を分けた。",
        "",
        f"1. **1段目**: J-Quants のノードだけ（`{d['stage1_features']}`）で、全体で学習する",
        "2. **2段目**: 明細ありの群だけで、1段目の予測を固定したまま、明細の特徴量で"
        "補正する分だけを学習する",
        "",
        "2段目は正則化を強めると補正がゼロに縮み、1段目の予測に戻る。"
        "補正の強さは訓練行の中だけの前向き検証で選び、テスト窓は見ない。",
        "",
        f"- ベンチマーク: {'TOPIX控除' if d['benchmark'] == 'topix' else '業種指数控除'}"
        f" / {d['horizon']}営業日",
        f"- 明細ありの群: {d['n_edinet_rows']:,}件・{d['n_edinet_codes']:,}社"
        f"（うち評価のテスト窓に {d['n_edinet_test_rows']:,}件）",
        f"- 1段目の walk-forward: 最初の訓練 {d['stage1_min_train_months']}か月から"
        "（2段目の訓練行にもアウトオブサンプルの予測を付けるため）",
        f"- 補正の強さの候補: {', '.join(str(g) for g in d['lambda_grid'])}"
        "（大きいほど1段目に近い）",
        f"- 区間: 発表日単位のブートストラップ {d['n_boot']:,}回の95%区間",
        "",
        "## 比べるもの",
        "",
        "| 名前 | 中身 |",
        "| --- | --- |",
        "| 1段目のまま | 全体で学習した J-Quants のモデルの予測 |",
        "| 明細なしの2段目 | 同じ仕組みで、明細の特徴量を入れず切片だけ学習したもの（群に合わせた較正し直し） |",
        "| 明細ありの2段目 | 同じ仕組みで、明細の特徴量を入れたもの |",
        "",
        "明細の効果は **明細ありの2段目 − 明細なしの2段目** で測る。"
        "較正し直しの効果と明細の効果を分けるため。",
        "",
        "## 判定基準（結果を見る前に決めたもの）",
        "",
        "差 = AUC(明細ありの2段目) − AUC(明細なしの2段目)。",
        "",
        "| 判定 | 条件 |",
        "| --- | --- |",
        "| 明細が効く | 差の95%区間が 0 を上回り、かつ偽の明細の差をすべて上回る |",
        "| 明細が害になる | 差の95%区間が 0 を下回り、かつ偽の明細の差をすべて下回る |",
        "| 差が見えない | それ以外 |",
        "",
        f"偽の明細は、明細の特徴量を明細ありの行の間で入れ替えたもの（{d['n_placebo']}回）。"
        "区間はテスト行の引き直しだけで、補正の学習そのものの揺れを含まないので、"
        "これで補う。",
        "",
        "判定に使うのは明細の特徴量を絞った版（compact）だけ。全部入れた版（full）は参考で、"
        "偽の明細は回さない。",
        "",
        "## 結果",
        "",
        "| 1段目 | 明細の特徴量 | 件数 | 1段目のまま | 明細なしの2段目 | 明細ありの2段目 "
        "| 明細あり − なし | 偽の明細の差（最小〜最大） | 判定 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, r in d["results"].items():
        m, v = key.split("/")
        a = r["auc"]
        dd = r["diff"]["edinet-base"]
        label = f"**{r['verdict']}**" if v == "compact" else f"（参考）{r['verdict']}"
        plc = [x for x in r.get("placebo", []) if x is not None and np.isfinite(x)]
        plc_s = f"{min(plc):+.4f}〜{max(plc):+.4f}" if plc else "—"
        lines.append(
            f"| {m} | {v}（{r['n_features']}列） | {r['n_test']:,} "
            f"| {a['raw']['point']:.4f} {_auc_ci(a['raw']['ci'])} "
            f"| {a['base']['point']:.4f} {_auc_ci(a['base']['ci'])} "
            f"| {a['edinet']['point']:.4f} {_auc_ci(a['edinet']['ci'])} "
            f"| {dd['point']:+.4f} {_ci(dd['ci'])} | {plc_s} | {label} |")

    lines += [
        "",
        "## 判定の根拠",
        "",
    ]
    for key, r in d["results"].items():
        lines.append(f"- `{key}`: **{r['verdict']}** — {r['why']}")

    lines += [
        "",
        "## 較正し直しの効果（参考）",
        "",
        "| 1段目 | 明細の特徴量 | 明細なしの2段目 − 1段目のまま |",
        "| --- | --- | ---: |",
    ]
    for key, r in d["results"].items():
        m, v = key.split("/")
        dd = r["diff"]["base-raw"]
        lines.append(f"| {m} | {v} | {dd['point']:+.4f} {_ci(dd['ci'])} |")

    lines += [
        "",
        "## フォールドごとの2段目",
        "",
        "補正の強さが毎回最大（10.0）なら、訓練行の中の検証で明細に情報が"
        "見つからなかったということ。",
        "",
        "| 1段目 | 明細の特徴量 | フォールド | 2段目の訓練件数 | テスト件数 | 選ばれた強さ |",
        "| --- | --- | :-: | ---: | ---: | ---: |",
    ]
    for key, r in d["results"].items():
        m, v = key.split("/")
        for f in r["folds"]:
            lam = "（件数不足で除外）" if not f["used"] else f"{f['lambda']:g}"
            lines.append(f"| {m} | {v} | {f['fold']} | {f['n_train']:,} "
                         f"| {f['n_test']:,} | {lam} |")

    lines += [
        "",
        "## 明細の特徴量（compact）",
        "",
        ", ".join(f"`{n}`" for n in d["features"]["compact"]),
        "",
        "## 読み方",
        "",
        "- 「1段目のまま」は切り分け（`docs/ACCGRAPH_DIAGNOSE.md`）の `latest_jq` の"
        "「明細あり」列と一致するはず。評価のテスト窓も、その窓の予測を出すモデルの"
        "訓練データも同じ。ずれていれば、2段目の件数不足で外したフォールドがある"
        "（上のフォールド表で分かる）か、どこかが食い違っている。",
        "- 差が見えないときは、区間の上限を見る。上限が小さければ、少なくとも"
        "それより大きな上積みは今の件数でも否定できている。",
        "- 明細ありの群は本流の母集団の順に取得しているので、大型・人気株に偏っている。"
        "ここでの結果は、この群についてのもの。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EDINET の明細の上積みを2段構えで測る")
    ap.add_argument("--data-dir", default=build.OUT_DIR)
    ap.add_argument("--benchmark", choices=["topix", "sector"], default="topix")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-train-months", type=int, default=48)
    ap.add_argument("--stage1-min-train-months", type=int,
                    default=STAGE1_MIN_TRAIN_MONTHS)
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--step-months", type=int, default=12)
    ap.add_argument("--min-test-rows", type=int, default=200)
    ap.add_argument("--min-stage2-rows", type=int, default=MIN_STAGE2_ROWS)
    ap.add_argument("--n-placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--models", nargs="*", default=list(STAGE1_MODELS))
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args(argv)

    d = run(args.data_dir, benchmark=args.benchmark, horizon=args.horizon,
            models=tuple(args.models), n_boot=args.n_boot,
            min_train_months=args.min_train_months,
            stage1_min_train_months=args.stage1_min_train_months,
            test_months=args.test_months, step_months=args.step_months,
            min_test_rows=args.min_test_rows,
            min_stage2_rows=args.min_stage2_rows, n_placebo=args.n_placebo)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(d))
    print(f"[inc] 書き出し {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
