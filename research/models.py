#!/usr/bin/env python3
"""
本番で走らせる複数モデルの、学習・保存・読込を1箇所にまとめる。

なぜ複数モデルか
---------------
アンサンブル（機械的に混ぜて1つのスコアにする）はしない。運用は手動の
指値なので、**複数モデルのスコアを画面に並べて人間が統合判断する**。

実測でその価値は裏付けられている。

  lgbm と logit のスコア相関   0.412（Spearman）
  上位10%に選ばれた銘柄の重複  12%のみ

ほぼ別の銘柄を選んでいるので、並べて見る意味がある。逆に RF は lgbm との
相関 0.858 で冗長だったため外した（速度も最遅、運用指標も最下位）。

モデル間でスコアは比較できない
---------------------------
学習器が違えばスコアのスケールも意味も違う。画面に出すのは
**各モデル自身の過去スコア分布での位置（pctHistorical）** に揃える。
いま棒グラフに使っている指標と同じもの。

保存形式
-------
全モデル joblib で統一する。lgbm だけ model.txt も書く（既存の
predict_daily.py と retrain-weekly.yml が読んでいるため。後方互換）。

  research/model/
    model.txt              lgbm の booster（既存・後方互換）
    meta.json              lgbm のメタ（既存・後方互換）
    oof.parquet            lgbm の out-of-fold（既存・後方互換）
    models/<algo>/
      model.joblib         学習済みモデル
      meta.json            較正表・スコア帯・パラメータ・学習日
      oof.parquet          out-of-fold のスコア
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning_multi as TM  # noqa: E402

#: 画面に並べる順。RF は lgbm との相関 0.858 で冗長なため入れない
ALGOS = ("lgbm", "xgb", "cat", "logit", "mlp")

#: 基準モデル。日次予測の順位・スコア帯・較正・SHAP・追跡ファイルは
#: すべてこのモデルのもの。research/model/{model.txt, meta.json, oof.parquet}
#: に置かれ、train_production.py が作る
BASELINE = "lgbm"

#: train_multi.py が作るモデル。基準モデルは含めない。
#:
#: なぜ含めないか（実測）
#: --------------------
#: 同じ LightGBM を別のパラメータで当てはめると、内側検証の PR-AUC は
#: ほぼ同じ（本番 0.4016 / 探索 0.4029）なのに、選ぶ銘柄が大きく変わる。
#:
#:   Spearman 相関            0.863
#:   上位10%の重複            52.8%（= 47%は別の銘柄）
#:   パーセンタイルの絶対差   中央 8.6pt / 90%分位 25.8pt / 最大 62.7pt
#:
#: 画面に「LightGBM」が2本あって45ptずれていると、どちらを見ればよいのか
#: 分からない。順位・帯・SHAP の基準になっているのは本番モデルなので、
#: 画面の LightGBM も本番モデルに一本化する。
EXTRA = tuple(a for a in ALGOS if a != BASELINE)

#: 画面に出す日本語名。棒グラフのラベルに使う
JA = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "logit": "ロジスティック回帰",
    "mlp": "ニューラルネット",
}

#: 棒グラフ・スプレッドシートの列名に使う短い記号。
#:
#: 画面（src/lib/predictions.js）とスプレッドシート（export_sheets.py）で
#: 同じ記号を使いたいので、ここを正本にして predictions.json に載せる。
#: JS 側にも同じ表があるが、payload の short を優先して読む
SHORT = {
    "lgbm": "LGB", "xgb": "XGB", "cat": "CAT",
    "logit": "LR", "mlp": "NN", "rf": "RF",
}

#: それぞれ何を見ているか。画面の説明に使う
NOTE = {
    "lgbm": "勾配ブースティング（葉単位で成長）。本番の基準モデル",
    "xgb": "勾配ブースティング（深さ単位で成長）。木の形が違う",
    "cat": "勾配ブースティング（欠損と正則化の扱いが違う）",
    "logit": "線形モデル。特徴量の足し算で判断するので、木とは外す銘柄が違う",
    "mlp": "ニューラルネット。滑らかな交互作用を見る。木とも線形とも違う",
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def sub_dir(algo: str, root: str = MODEL_DIR) -> str:
    return os.path.join(root, "models", algo)


def fit(algo: str, X: np.ndarray, y: np.ndarray, cols: List[str],
        params: Optional[Dict] = None):
    """
    探索済みパラメータで学習する。

    モデルの定義は tuning_multi.build に置いてある。探索と本番で同じ
    ものを使うため、ここでは組み立てを委ねる（食わせる形がずれない）。
    """
    p = TM.params_for(algo) if params is None else params
    m = TM.build(algo, p, y, cols)
    m.fit(X, y)
    return m


def predict(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def save(algo: str, model, meta: Dict, oof: pd.DataFrame,
         root: str = MODEL_DIR) -> str:
    """1モデルぶんを保存する。"""
    import joblib

    d = sub_dir(algo, root)
    os.makedirs(d, exist_ok=True)
    joblib.dump(model, os.path.join(d, "model.joblib"), compress=3)
    import json
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    oof.to_parquet(os.path.join(d, "oof.parquet"), index=False,
                   compression="zstd")
    return d


def load(algo: str, root: str = MODEL_DIR):
    """
    保存済みモデルとメタを返す。無ければ (None, None)。

    まだ学習していないモデルがあっても予測は動かす（画面はあるモデルだけ
    並べる）。1モデルの学習が失敗した週に、日次予測まで止めないため。
    """
    import json

    import joblib

    d = sub_dir(algo, root)
    mp = os.path.join(d, "model.joblib")
    tp = os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None, None
    with open(tp, encoding="utf-8") as fh:
        meta = json.load(fh)
    return joblib.load(mp), meta


def available(root: str = MODEL_DIR) -> List[str]:
    """保存済みのモデル名を ALGOS の順で返す。"""
    return [a for a in ALGOS
            if os.path.exists(os.path.join(sub_dir(a, root), "model.joblib"))]


def hist_scores(algo: str, root: str = MODEL_DIR) -> Optional[np.ndarray]:
    """
    そのモデルの過去スコア分布（out-of-fold）。pctHistorical の基準。

    モデルごとに別の分布を使う。スコアのスケールが違うので、
    共通の分布で位置を出すと意味が混ざる。
    """
    p = os.path.join(sub_dir(algo, root), "oof.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p, columns=["score"])["score"].to_numpy()


def pct_historical(score: float, hist: Optional[np.ndarray]) -> Optional[float]:
    """過去分布のうち、このスコアより低い割合 × 100。"""
    if hist is None or not len(hist) or not np.isfinite(score):
        return None
    return round(float((hist < score).mean()) * 100, 1)
