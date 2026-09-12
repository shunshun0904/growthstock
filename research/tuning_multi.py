#!/usr/bin/env python3
"""
複数アルゴリズムのパラメータ探索。

なぜ別モジュールにするか
----------------------
tuning.py は本番の学習パス（retrain-weekly.yml → run_tuning.py →
train_production.py）に組み込まれていて動いている。そこに分岐を増やすと、
本番の探索を壊す危険がある。ここは新しいモデルを足すための場所にして、
tuning.py からは分割の作り方（year_folds）と重みの付け方だけを借りる。

lgbm と条件を完全に揃える
------------------------
アルゴリズムを比べるのに片方だけ探索済みだと、差が手法の差なのか
探索労力の差なのか分からない。以下を全モデルで共通にする。

  分割        年 × 時価総額帯で層別・日付単位で分割・5分割
              （tuning.year_folds(scheme="year_cap_date") と同じ）
  目的関数    分割平均の PR-AUC
  試行数      50
  木の本数    200 で固定（lgbm の SEARCH_N_ESTIMATORS と同じ）
              early stopping で決めると検証窓のばらつきが本数に乗り、
              試行ごとに「別の大きさのモデル」を比べることになる
  探索期間    ホールドアウト（直近12ヶ月）より手前だけ
              （train_model.holdout_bounds から導く。手で書かない）

探索するモデル
-------------
  lgbm     LightGBM。本番と同じ。条件を揃えるため再探索する
  xgb      XGBoost。欠損はネイティブ。depth-wise で木の形が違う
  cat      CatBoost。欠損と正則化の扱いが違う
  rf       ランダムフォレスト。欠損は中央値補完（Pipeline で fold 内に閉じる）
  logit    ロジスティック回帰。中央値補完 + 欠損指示子 + 標準化
           （分位変換は窓ごとの安定性を損なうと実測で分かっている）

保存先は research/multi_params.json。本番が読む lgbm_params.json とは
別ファイルにして、探索が失敗しても本番が動き続けるようにする。
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import tuning

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "multi_params.json")

#: 木の本数。lgbm の探索と同じ値に固定する
N_ESTIMATORS = tuning.SEARCH_N_ESTIMATORS      # 200
SEED = 0
ALGOS = ("lgbm", "xgb", "cat", "rf", "logit")


# --------------------------------------------------------------------------- #
# アルゴリズムごとの探索空間
# --------------------------------------------------------------------------- #

def _space_lgbm(trial) -> Dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300,
                                               log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _space_xgb(trial) -> Dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # depth-wise なので葉数ではなく深さ。6 を中心に振る
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0,
                                                log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def _space_cat(trial) -> Dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # CatBoost は深さを 6〜8 に取るのが定石。16 まで許すと学習が極端に遅い
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0,
                                               log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature",
                                                   0.0, 2.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128,
                                                                   254]),
    }


def _space_rf(trial) -> Dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50,
                                              log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50,
                                               log=True),
        "max_features": trial.suggest_float("max_features", 0.05, 0.8),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]),
    }


def _space_logit(trial) -> Dict:
    return {
        # 正則化の強さが本命。実効標本数が小さいので強めも試せる幅を取る
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    }


SPACES: Dict[str, Callable] = {
    "lgbm": _space_lgbm, "xgb": _space_xgb, "cat": _space_cat,
    "rf": _space_rf, "logit": _space_logit,
}


# --------------------------------------------------------------------------- #
# アルゴリズムごとの学習
# --------------------------------------------------------------------------- #

def build(algo: str, params: Dict, y: np.ndarray):
    """パラメータから学習器を組む。不均衡の補正も algo ごとに違う。"""
    spw = tuning.scale_pos_weight(y)
    p = dict(params)

    if algo == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary", boosting_type="gbdt",
            n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1,
            verbose=-1, subsample_freq=1, scale_pos_weight=spw, **p)

    if algo == "xgb":
        import xgboost as xgbm
        return xgbm.XGBClassifier(
            n_estimators=N_ESTIMATORS, tree_method="hist", n_jobs=-1,
            random_state=SEED, eval_metric="logloss",
            scale_pos_weight=spw, **p)

    if algo == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=N_ESTIMATORS, random_seed=SEED, verbose=0,
            allow_writing_files=False, scale_pos_weight=spw, **p)

    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        # 欠損を扱えないので中央値で埋める。Pipeline なので fold の中で
        # 訓練側だけから中央値が決まり、検証側の分布は漏れない
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                   random_state=SEED, **p))

    if algo == "logit":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # l1 は liblinear/saga のみ。saga は収束が遅いので liblinear にする
        solver = "liblinear" if p.get("penalty") == "l1" else "lbfgs"
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED, solver=solver, **p))

    raise ValueError(algo)


def fit_eval(algo: str, params: Dict, tr: pd.DataFrame, va: pd.DataFrame,
             cols: List[str]) -> Tuple[float, float]:
    """内側検証の PR-AUC と ROC-AUC。目的関数は PR-AUC（lgbm と同じ）。"""
    Xtr = tr[cols].to_numpy(dtype=float)
    ytr = tr["label"].to_numpy(dtype=int)
    Xva = va[cols].to_numpy(dtype=float)
    yva = va["label"].to_numpy(dtype=int)
    m = build(algo, params, ytr)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(Xva)[:, 1]
    return (float(average_precision_score(yva, prob)),
            float(roc_auc_score(yva, prob)))


# --------------------------------------------------------------------------- #
# 探索
# --------------------------------------------------------------------------- #

def tune(algo: str, df: pd.DataFrame, cols: List[str], *, n_trials: int = 50,
         n_splits: int = 5, seed: int = SEED, verbose: bool = True) -> Dict:
    """
    1アルゴリズムを探索する。df はホールドアウトより手前だけを渡すこと。

    分割は tuning.year_folds の year_cap_date（年×時価総額帯で層別・
    日付単位で分割）。日付単位にするのは、同じ日の銘柄が地合いを共有する
    ため。行単位で切ると同じ日が訓練と検証に分かれ、検証が楽になる。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = tuning.year_folds(df, n_splits=n_splits, seed=seed,
                              by_year=True, by_cap=True, group_by_date=True)
    if not folds:
        raise SystemExit("分割を作れません")
    if verbose:
        print(f"  [{algo}] {len(folds)}分割 / 木{N_ESTIMATORS}本固定 / "
              f"{n_trials}試行")
        print("  " + " / ".join(
            f"訓練{len(t):,}→検証{len(v):,}(正例{int(v['label'].sum())})"
            for t, v in folds))

    def objective(trial):
        params = SPACES[algo](trial)
        prs, rocs = [], []
        for tr, va in folds:
            pr, roc = fit_eval(algo, params, tr, va, cols)
            prs.append(pr)
            rocs.append(roc)
        trial.set_user_attr("fold_scores", [round(x, 4) for x in prs])
        trial.set_user_attr("roc_auc", float(np.mean(rocs)))
        # 分割ごとのばらつきが大きい設定は、たまたま当たっただけの可能性がある。
        # 平均で選ぶが、ばらつきも残して後から見られるようにする
        trial.set_user_attr("score_std", float(np.std(prs)))
        return float(np.mean(prs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    at = study.best_trial.user_attrs
    out = {
        "params": dict(study.best_params),
        "_cv": {
            "algo": algo, "scheme": "year_cap_date", "n_splits": len(folds),
            "n_trials": n_trials, "n_estimators": N_ESTIMATORS,
            "mean_pr_auc": round(study.best_value, 4),
            "std": round(float(at.get("score_std", 0.0)), 4),
            "fold_scores": at.get("fold_scores", []),
            "mean_roc_auc": round(float(at.get("roc_auc", float("nan"))), 4),
            "base_rate": round(float(np.mean(
                [v["label"].mean() for _, v in folds])), 4),
            "train_rows": int(len(df)),
            "train_to": str(pd.to_datetime(df["Date"]).max().date()),
        },
    }
    if verbose:
        cv = out["_cv"]
        print(f"  [{algo}] PR-AUC {cv['mean_pr_auc']:.4f} "
              f"(±{cv['std']:.4f}) / ROC-AUC {cv['mean_roc_auc']:.4f} / "
              f"正例率 {cv['base_rate']*100:.1f}%")
        print(f"  [{algo}] {out['params']}")
    return out


def load(path: str = PARAMS_PATH) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(store: Dict[str, Dict], path: str = PARAMS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2, sort_keys=True)


def params_for(algo: str, store: Optional[Dict[str, Dict]] = None) -> Dict:
    """探索済みパラメータを返す。無ければ空（各 build の既定に任せる）。"""
    s = load() if store is None else store
    return dict(s.get(algo, {}).get("params", {}))
