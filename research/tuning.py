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

# 探索結果はリポジトリに残す。
# research/_data/ は .gitignore されているので、そこに置くと
# ワークフローのコンテナが終わった時点で消え、毎回探索し直しになる。
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lgbm_params.json")

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


def time_series_folds(df: pd.DataFrame, n_splits: int = 5,
                      embargo_days: int = 60
                      ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    時系列の k 分割。訓練窓を伸ばしながら検証窓を前に進める。

    通常の KFold は使えない。行をシャッフルすると同じ銘柄の隣接期間が
    訓練と検証の両方に入り、検証が簡単になりすぎて必ず楽観的な
    パラメータが選ばれる。

    1つの分割で決めるとその期間の癖を拾うので、複数の期間で平均する。
    分割は「件数で等分」する（期間で等分すると、サンプルの少ない
    初期の窓が極端に小さくなる）。

    訓練と検証の間にはエンバーゴを置く。ラベルが先 embargo_days 営業日の
    情報を含むため、隣接させると訓練側のラベルが検証期間に食い込む。
    """
    d = pd.to_datetime(df["Date"])
    edges = [d.quantile(k / (n_splits + 1)) for k in range(1, n_splits + 2)]
    embargo = pd.Timedelta(days=int(round(embargo_days * 1.45)))
    out: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        va_lo, va_hi = edges[i], edges[i + 1]
        tr = df[d <= va_lo - embargo]
        va = df[(d > va_lo) & (d <= va_hi)]
        if len(tr) == 0 or len(va) == 0:
            continue
        if tr["label"].nunique() < 2 or va["label"].nunique() < 2:
            continue
        out.append((tr, va))
    return out


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
         seed: int = 0, verbose: bool = True, n_splits: int = 5,
         embargo_days: int = 60) -> Dict:
    """
    Optuna で探索する。df は「テスト窓より前」のデータだけを渡すこと。

    評価は時系列 k 分割の平均 PR-AUC。
    1つの分割で決めていたときは、その期間の癖を拾う恐れがあった。

    返すのは LGBMClassifier にそのまま渡せる辞書。
    探索の記録は tune_cv() で別に取る（返り値に混ぜると、
    そのまま LGBMClassifier に渡したときに未知の引数で落ちる）。
    n_estimators は各分割の early stopping が決めた本数の中央値。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = time_series_folds(df, n_splits=n_splits, embargo_days=embargo_days)
    if not folds:
        if verbose:
            print("  [tune] 分割を作れないため既定値を使う")
        return dict(DEFAULT_PARAMS)
    if verbose:
        print(f"  [tune] 時系列{len(folds)}分割 "
              + " / ".join(f"訓練{len(t):,}→検証{len(v):,}(正例{int(v['label'].sum())})"
                           for t, v in folds))

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
        scores, iters = [], []
        for tr, va in folds:
            sc, it = _fit_one(params, tr, va, cols)
            scores.append(sc)
            iters.append(it)
        trial.set_user_attr("best_iteration", int(np.median(iters)))
        trial.set_user_attr("fold_scores", [round(s, 4) for s in scores])
        # 分割ごとのばらつきが大きい設定は、たまたま当たっただけの可能性がある。
        # 平均で選ぶが、ばらつきも残して後から見られるようにする
        trial.set_user_attr("score_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**FIXED, **study.best_params, "subsample_freq": 1}
    # 木の本数は探索対象にせず、early stopping が決めた本数を使う
    best["n_estimators"] = max(
        50, int(study.best_trial.user_attrs.get("best_iteration", 200)))
    at = study.best_trial.user_attrs
    global LAST_CV
    LAST_CV = {"n_splits": len(folds), "mean_pr_auc": round(study.best_value, 4),
               "std": round(float(at.get("score_std", 0.0)), 4),
               "fold_scores": at.get("fold_scores", []),
               "n_trials": n_trials}
    if verbose:
        folds_txt = " ".join(f"{x:.4f}" for x in at.get("fold_scores", []))
        print(f"  [tune] {n_trials}試行 / {len(folds)}分割平均PR-AUC "
              f"{study.best_value:.4f} (±{at.get('score_std', 0):.4f}) "
              f"/ 木 {best['n_estimators']}本")
        print(f"  [tune] 分割ごと: {folds_txt}")
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


#: 直近の tune() が使った分割の記録。tune() の返り値には混ぜない。
LAST_CV: Dict = {}


def params_for(preset: str, store: Optional[Dict[str, Dict]] = None) -> Dict:
    """
    プリセット名から学習用パラメータを返す。無ければ既定値。

    `_` で始まるキーは探索の記録（CV スコアなど）で、LightGBM には渡さない。
    """
    store = load_params() if store is None else store
    got = store.get(preset)
    if not got:
        return dict(DEFAULT_PARAMS)
    clean = {k: v for k, v in got.items() if not k.startswith("_")}
    return {**dict(DEFAULT_PARAMS), **clean}
