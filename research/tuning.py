#!/usr/bin/env python3
"""
LightGBM のハイパーパラメータを Optuna で探索する。

## 最重要: テスト期間を見ないこと

探索は「良さそうなパラメータを選ぶ」作業なので、
評価に使う期間のデータを一度でも見ればリークになる。
ウォークフォワードは9つのテスト窓（2021-06〜2025-11）を持つため、
探索に使ってよいのは **最初のテスト窓より前** のデータだけになる。

    |--- 探索に使ってよい ---|--エンバーゴ--|-- F1テスト --|-- F2テスト --| ...
                            ↑
                     ここより後は一切見ない

そのうえで、探索用データをさらに時系列で内側訓練 / 内側検証に割り、
内側検証の PR-AUC を最大化する。ここもランダム分割は使わない
（同一銘柄の隣接月が強く相関するため、必ず楽観的な数字が出る）。

## なぜフォールドごとに探索しないか

フォールドごとに探索するのが理想だが、
9フォールド × 11プリセット × 試行回数 の学習が必要で現実的な時間に収まらない。
代わりに「プリセットごとに1回、最初のテスト窓より前のデータだけで探索」する。
全フォールドで同じパラメータを使うので比較の条件も揃う。

探索結果は research/_data/lgbm_params.json に保存し、
単一分割とウォークフォワードで共有する。毎回探索し直さない。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_data", "lgbm_params.json")

# 探索しないもの（固定）。再現性と、木の本数は early stopping で決めるため。
FIXED = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 2000,      # 上限。実際の本数は early stopping が決める
    "random_state": 0,
    "n_jobs": -1,
    "verbose": -1,
}

# 探索しなかった場合に使う既定値。控えめな正則化。
DEFAULT_PARAMS = {
    **FIXED,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


def chronological_split(df: pd.DataFrame, valid_frac: float = 0.25
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    時系列で内側訓練 / 内側検証に割る。

    ランダム分割にすると、同一銘柄の隣接月が両側に入って
    検証が簡単になりすぎ、必ず楽観的なパラメータが選ばれる。
    """
    d = pd.to_datetime(df["Date"])
    cut = d.quantile(1.0 - valid_frac)
    inner_train = df[d <= cut]
    inner_valid = df[d > cut]
    return inner_train, inner_valid


def scale_pos_weight(y: np.ndarray) -> float:
    pos = max(1, int(np.sum(y)))
    return float((len(y) - pos) / pos)


def _fit_one(params: Dict, tr: pd.DataFrame, va: pd.DataFrame,
             cols: List[str]) -> Tuple[float, int]:
    """内側検証の PR-AUC と、early stopping が決めた木の本数を返す。"""
    import lightgbm as lgb

    Xtr, ytr = tr[cols].to_numpy(dtype=float), tr["label"].to_numpy(dtype=int)
    Xva, yva = va[cols].to_numpy(dtype=float), va["label"].to_numpy(dtype=int)
    model = lgb.LGBMClassifier(**params,
                               scale_pos_weight=scale_pos_weight(ytr))
    # eval_set は 4.7 で非推奨。eval_X / eval_y を使う
    model.fit(
        Xtr, ytr,
        eval_X=Xva, eval_y=yva,
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best_iter = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
    score = float(average_precision_score(yva, model.predict_proba(Xva)[:, 1]))
    return score, best_iter


def tune(df: pd.DataFrame, cols: List[str], *, n_trials: int = 30,
         seed: int = 0, verbose: bool = True) -> Dict:
    """
    Optuna で探索する。df は「テスト窓より前」のデータだけを渡すこと。

    返すのは LGBMClassifier にそのまま渡せる辞書。
    n_estimators は early stopping が決めた本数に置き換えてある。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va = chronological_split(df)
    if len(va) == 0 or va["label"].nunique() < 2 or tr["label"].nunique() < 2:
        if verbose:
            print("  [tune] 内側検証に両クラスが揃わないため既定値を使う")
        return dict(DEFAULT_PARAMS)

    def objective(trial):
        params = {
            **FIXED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2,
                                                 log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,
                                                   log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        score, best_iter = _fit_one(params, tr, va, cols)
        trial.set_user_attr("best_iteration", best_iter)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**FIXED, **study.best_params, "subsample_freq": 1}
    # 木の本数は探索対象にせず、early stopping が決めた本数を使う
    best["n_estimators"] = max(
        50, int(study.best_trial.user_attrs.get("best_iteration", 200)))
    if verbose:
        print(f"  [tune] {n_trials}試行 / 内側検証PR-AUC {study.best_value:.4f} "
              f"/ 木 {best['n_estimators']}本")
    return best


def load_params(path: str = PARAMS_PATH) -> Dict[str, Dict]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_params(params: Dict[str, Dict], path: str = PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, ensure_ascii=False, indent=2)


def params_for(preset: str, store: Optional[Dict[str, Dict]] = None) -> Dict:
    """プリセット名から学習用パラメータを返す。無ければ既定値。"""
    store = load_params() if store is None else store
    got = store.get(preset)
    return {**dict(DEFAULT_PARAMS), **got} if got else dict(DEFAULT_PARAMS)
