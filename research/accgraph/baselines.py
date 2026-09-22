#!/usr/bin/env python3
"""
ベースラインモデル。

GNN を作る前に、グラフを「ただの数値の並び」に潰した単純なモデルが
どこまで届くかを測る。GNN が有効だと言えるのは、同じ特徴量・同じ分割で
ここを上回ったときだけである。上回らなければ、グラフ構造は
この問題に効いていないという結論になる。

特徴量の潰し方は3通り用意する。

  latest   当該四半期のグラフだけ（時系列を使わない）
  seq      過去8四半期ぶんを全部つなげる（GNN と同じ情報量）
  nodes    ノード特徴量だけ（エッジ特徴量の寄与を分離する）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema

#: サンプルごとの文脈。グラフの外側にあるが、決算の読み方を左右する
CONTEXT_COLS = ("quarter", "seq_len", "log_turnover")


def context_features(meta: pd.DataFrame, period_mask: np.ndarray
                     ) -> Tuple[np.ndarray, List[str]]:
    """決算の四半期・系列の長さ・流動性。いずれも発表時点で確定している。"""
    q = pd.to_numeric(meta["quarter"], errors="coerce").fillna(0).to_numpy()
    onehot = np.stack([(q == k).astype(np.float64) for k in (1, 2, 3, 4)], axis=1)
    seq_len = period_mask.sum(axis=1).astype(np.float64)[:, None]
    tv = pd.to_numeric(meta["turnover_ma20"], errors="coerce").to_numpy()
    log_tv = np.log1p(np.nan_to_num(tv, nan=0.0))[:, None]
    names = [f"quarter_{k}" for k in (1, 2, 3, 4)] + ["seq_len", "log_turnover"]
    return np.concatenate([onehot, seq_len, log_tv], axis=1), names


def rank_within_date(X: np.ndarray, dates: pd.Series) -> np.ndarray:
    """
    同じ発表日の中で各列を順位（0〜1）に直す。

    ## なぜ要るのか

    実データで測ると、単変量の情報係数の上位が軒並み `log_size`（金額の
    対数＝企業規模）だった。AUC 0.55 の正体が規模効果なら、会計フローを
    グラフにした意味はそこには無い。

    同じ日に決算を出した銘柄どうしで順位に直すと、

      - その日の地合い（市場全体の動き）が全銘柄に共通なので消える
      - 規模の絶対水準が消え、「その日の中で相対的に大きいか」だけが残る

    ので、会計構造そのものに情報があるかを分離して測れる。

    同順位は平均を取らず、安定ソートの並び順で割り振る（2回の argsort）。
    単調変換なので順位相関やAUCは変わらず、日付ごとの行数が小さいので
    総当たりのランク付けより桁違いに速い。
    """
    d = pd.to_datetime(dates).to_numpy()
    order = np.argsort(d, kind="stable")
    ds = d[order]
    # 日付の切れ目
    bounds = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1], True])
    out = np.empty_like(X, dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        n = len(idx)
        if n == 1:
            out[idx] = 0.5
            continue
        blk = X[idx]
        r = blk.argsort(axis=0, kind="stable").argsort(axis=0, kind="stable")
        out[idx] = r / (n - 1)
    return out


def flatten(node_feat: np.ndarray, edge_feat: np.ndarray,
            period_mask: np.ndarray, meta: pd.DataFrame,
            kind: str = "seq") -> Tuple[np.ndarray, List[str]]:
    """
    グラフ系列を1本のベクトルに潰す。

    末尾が `_rank` のセットは、同じ発表日の中で順位に直したもの。
    規模と地合いを抜いても情報が残るかを測るために使う。
    """
    rank = kind.endswith("_rank")
    if rank:
        kind = kind[: -len("_rank")]
    if kind not in ("latest", "seq", "nodes"):
        raise ValueError(f"未知の特徴量セット: {kind}")

    t_slice = slice(0, 1) if kind == "latest" else slice(None)
    nf = node_feat[:, t_slice]
    parts = [nf.reshape(len(nf), -1)]
    names = [f"n[{t}]{schema.NODE_IDS[j]}.{f}"
             for t in range(nf.shape[1])
             for j in range(nf.shape[2])
             for f in schema.NODE_FEATURES]

    if kind != "nodes":
        ef = edge_feat[:, t_slice]
        parts.append(ef.reshape(len(ef), -1))
        names += [f"e[{t}]{schema.EDGES[j].src}->{schema.EDGES[j].dst}.{f}"
                  for t in range(ef.shape[1])
                  for j in range(ef.shape[2])
                  for f in schema.EDGE_FEATURES]

    # 各四半期が存在したかどうかも情報。0埋めと「値が0」を区別させる
    parts.append(period_mask[:, t_slice].astype(np.float64))
    names += [f"period_mask[{t}]" for t in range(period_mask[:, t_slice].shape[1])]

    ctx, ctx_names = context_features(meta, period_mask)
    parts.append(ctx)
    names += list(ctx_names)

    X = np.concatenate([p.astype(np.float64) for p in parts], axis=1)
    assert X.shape[1] == len(names), (X.shape, len(names))
    if rank:
        X = rank_within_date(X, meta["entry_date"])
        names = [f"rank({n})" for n in names]
    return X, names


# --------------------------------------------------------------------------- #
# モデル
# --------------------------------------------------------------------------- #

@dataclass
class FittedModel:
    name: str
    predict_proba: Callable[[np.ndarray], np.ndarray]
    classes: np.ndarray


def _priors(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    return counts / max(counts.sum(), 1.0)


def fit_majority(X: np.ndarray, y: np.ndarray, classes: np.ndarray, **_) -> FittedModel:
    """訓練期間のクラス比率をそのまま返すだけ。情報ゼロの基準線。"""
    p = _priors(y, classes)
    return FittedModel("majority", lambda Z: np.tile(p, (len(Z), 1)), classes)


def fit_logit(X: np.ndarray, y: np.ndarray, classes: np.ndarray,
              seed: int = 0, **_) -> FittedModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, random_state=seed, n_jobs=1),
    )
    pipe.fit(X, y)
    return FittedModel("logit", pipe.predict_proba, pipe.classes_)


def fit_lgbm(X: np.ndarray, y: np.ndarray, classes: np.ndarray,
             seed: int = 0, feature_names: Optional[List[str]] = None,
             **_) -> FittedModel:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(classes),
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(X, y)
    return FittedModel("lgbm", clf.predict_proba, clf.classes_)


def fit_mlp(X: np.ndarray, y: np.ndarray, classes: np.ndarray,
            seed: int = 0, **_) -> FittedModel:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                      max_iter=200, early_stopping=True, n_iter_no_change=10,
                      random_state=seed),
    )
    pipe.fit(X, y)
    return FittedModel("mlp", pipe.predict_proba, pipe.classes_)


MODELS: Dict[str, Callable[..., FittedModel]] = {
    "majority": fit_majority,
    "logit": fit_logit,
    "lgbm": fit_lgbm,
    "mlp": fit_mlp,
}


def align_proba(model: FittedModel, X: np.ndarray,
                classes: np.ndarray) -> np.ndarray:
    """
    訓練期間に出現しなかったクラスがあっても、列を必ず classes の並びに揃える。

    揃えないと、フォールドごとに列の意味が変わって混同行列が混ざる。
    """
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    pos = {c: i for i, c in enumerate(model.classes)}
    for j, c in enumerate(classes):
        if c in pos:
            out[:, j] = p[:, pos[c]]
    s = out.sum(axis=1, keepdims=True)
    return np.divide(out, np.where(s > 0, s, 1.0))
