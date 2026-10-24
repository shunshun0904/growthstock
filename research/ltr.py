#!/usr/bin/env python3
"""
ランキング学習（LambdaRank）を本番の脇に置く。**買いの規則は変えない。**

運用者の選択（2026-10-24）「まず並べて見る」: 実験114b で「両方使う形」（今の規則を満たし、かつその日の候補の中で
ランキング学習の順位が上半分）が先に決めた判定を3通りとも満たした（docs/MODEL_ADOPTION_RULES.md §78-8）。
ただし差の幅が大きく、基準モデル1本の速報では差がほぼ0だったので、規則は変えずに画面と実運用の追跡に並べて見る（§78-9）。

何か
  実験114 の L と同じもの。LightGBM の LGBMRanker で、同じ日の候補を1つの組にし、段階は同じ日の中の
  「翌営業日の寄りで買い、20営業日目の終値で売った収益」の5分位（0〜4。利得は段階そのもの）。列は基準モデルと同じ
  （features.DEFAULT_PRESET。2026-10-24 から360列）。木の形は実験114 の探索の結果をそのまま使い（research/ltr_params.json）、
  毎週は探索しない。種は 42（実験114b で確かめた L と同じ）。当てはめは実験114 の fit_score と同じ作り
  （tests/test_ltr.py が同じスコアになることを確かめる）

画面と追跡での使い方
  毎日の候補を採点し、その日の候補の中の百分位（day_pct。0 = 最下位 / 1 = 最上位。候補が1件の日は 0.5。実験114b と同じ）を
  出す。「両方使う形」= 今の規則（1.2σ の3本がすべて95以上）かつ day_pct が DAY_UPPER 以上。参考に並べるだけで、買う銘柄は
  今の規則のまま

置き場
  research/model/models/ltr/{model.joblib, meta.json, oof.parquet}（ほかの追加モデルと同じ。Release には ltr_model.joblib /
  ltr_oof.parquet の名前で上がり、meta.json は git に残る）。models.LINEUP には入れない（分類ではないので predict_proba が
  無く、百分位の表・合意・一致度には数えない）
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_track as LT  # noqa: E402
import tuning  # noqa: E402

NAME = "ltr"
JA = "ランキング学習"
SHORT = "LTR"
NOTE = ("LambdaRank。同じ日の候補の中で、翌営業日の寄りで買い20営業日目の終値で売った収益の順番を学ぶ（実験114）。"
        "その日の中の順位だけを使い、買いの規則には使わない（参考。両方使う形）")
N_GRADES = 5
SEED = 42
#: 「両方使う形」の線: その日の候補の中の百分位（0〜1）がこれ以上なら上半分（実験114b の DAY_UPPER と同じ）
DAY_UPPER = 0.5
R20 = "ret_c20"
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "ltr_params.json")
#: 探索・当てはめで LightGBM に渡さない鍵（目的関数・不均衡の補正・種は、ここで決める）
_DROP = ("objective", "scale_pos_weight", "random_state")


# --------------------------------------------------------------------------- #
# 木の形
# --------------------------------------------------------------------------- #

def load_params(path: str = PARAMS_PATH) -> Dict:
    """ltr_params.json の中身（params と、探索した列の指紋 _features_sig・列数 _n_features・出どころ _source）。"""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def params_mismatch(rec: Dict, cols: Sequence[str]) -> Optional[str]:
    """学習する列 cols で探索した木の形でなければ、その理由（ほかのモデルと同じ確かめ。tuning.tuned_mismatch）。"""
    if not rec or not rec.get("params"):
        return "木の形が無い"
    return tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), list(cols))


def ranker_params(params: Dict, seed: int = SEED) -> Dict:
    """LGBMRanker に渡す形（実験114 の fit_score と同じ: 記録用の鍵を外し、種・出力の静けさ・並列を決める）。"""
    p = {k: v for k, v in params.items() if not str(k).startswith("_") and k not in _DROP}
    p["random_state"] = int(seed)
    p.setdefault("verbose", -1)
    p.setdefault("n_jobs", -1)
    return p


# --------------------------------------------------------------------------- #
# 段階と、その日の中の百分位
# --------------------------------------------------------------------------- #

def within_date_pct(dates, values) -> np.ndarray:
    """
    同じ日の中の百分位（0〜1。同じ値は平均の順位。1件だけの日は 0.5）。実験52 の within_date_pct と同じ式
    （tests/test_ltr.py が一致を確かめる）。学習の段階と、毎日の候補の「その日の中の順位」の両方に使う。
    """
    s = pd.Series(np.asarray(values, dtype=float), index=pd.RangeIndex(len(values)))
    g = s.groupby(pd.Series(dates).to_numpy())
    n = g.transform("size")
    rk = g.rank(method="average")
    return np.where(n > 1, (rk - 1) / (n - 1), 0.5).astype(float)


def grades(dates, r) -> np.ndarray:
    """同じ日の中の収益の5分位（0〜4 の整数。1件だけの日は真ん中の 2）。実験52 の within_date_grade と同じ。"""
    p = within_date_pct(dates, r)
    return np.minimum((p * N_GRADES).astype(int), N_GRADES - 1)


def attach_r20(ds: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    翌営業日の寄りで買い、20営業日目の終値で売った収益（比）を R20 の列に足す。live_track.forward の ret_close
    （実験114 の attach_exit と同じ計算）。日足が無い・寄り付かない・20営業日に満たない行は欠測。
    """
    keys = ds[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    keys["Date"] = pd.to_datetime(keys["Date"])
    fw = LT.forward(bars, keys, target=LT.TAKE_PROFIT, hold=LT.HOLD)
    r = pd.to_numeric(fw["ret_close"], errors="coerce").to_numpy(dtype=float) / 100.0
    out = ds.copy()
    out[R20] = r
    return out


# --------------------------------------------------------------------------- #
# 当てはめ
# --------------------------------------------------------------------------- #

def fit(tr: pd.DataFrame, cols: Sequence[str], params: Dict, seed: int = SEED):
    """
    tr（ラベルと R20 がある行）で学習する。LGBMRanker は同じ組（日付）が続いて並ぶことを要るので (Date, Code) の順に
    並べ、組の大きさを日ごとの件数にする（実験114 の fit_score の L と同じ）。
    """
    import lightgbm as lgb

    cols = list(cols)
    d = tr.sort_values(["Date", "Code"], kind="mergesort")
    y = grades(d["Date"], d[R20].to_numpy(dtype=float))
    sizes = d.groupby("Date", sort=True).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(N_GRADES)), **ranker_params(params, seed))
    m.fit(d[cols].to_numpy(dtype=float), y, group=sizes)
    return m


def predict(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def trainable(ds: pd.DataFrame) -> pd.Series:
    """学習・採点に使える行（ラベルと R20 がある行。実験114 の L と同じ）。"""
    return ds["label"].notna() & ds[R20].notna()


def oof_scores(ds: pd.DataFrame, cols: Sequence[str], params: Dict, seed: int = SEED) -> pd.DataFrame:
    """
    各行を「その行より前のデータだけで学習したモデル」で採点する。窓は train_production.oof_scores と同じ
    （36ヶ月 / 6ヶ月 / 6ヶ月、エンバーゴ = ラベル確定に要る営業日数。実験114 のずらし0と同じ窓）。
    学習・採点するのはラベルと R20 がある行。画面の百分位（pctHistorical）の基準と、実験114 との突き合わせに使う。
    """
    import build_dataset as B
    import walkforward as WF
    from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS

    d = pd.to_datetime(ds["Date"])
    folds = WF.make_folds(d, min_train_months=OOF_MIN_TRAIN_MONTHS, test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS, embargo_days=B.RISE_HORIZON)
    ok = trainable(ds)
    parts = []
    for f in folds:
        tr = ds[(d <= pd.Timestamp(f.train_end)) & ok]
        te = ds[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & ok]
        if len(te) < 200 or len(tr) < 1000:
            continue
        m = fit(tr, cols, params, seed)
        part = te[["Code", "Date", "label", R20]].copy()
        part["score"] = predict(m, te[list(cols)].to_numpy(dtype=float))
        part["fold"] = int(f.index)
        parts.append(part)
    if not parts:
        raise SystemExit("out-of-fold を作れません。期間が短すぎます")
    return pd.concat(parts, ignore_index=True)


def ndcg_at_k(dates, score, gain, k: int = 5, min_n: int = 5) -> float:
    """日付内 NDCG@k の平均（候補 min_n 件以上の日。実験114 の ndcg_at_k と同じ）。学習の確かめに使う。"""
    f = pd.DataFrame({"d": np.asarray(dates), "s": np.asarray(score, dtype=float),
                      "g": np.asarray(gain, dtype=float)})
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    vals = []
    for _, g in f.groupby("d", sort=True):
        if len(g) < min_n:
            continue
        gv = g["g"].to_numpy()
        top = gv[np.argsort(-g["s"].to_numpy(), kind="mergesort")][:k]
        ideal = np.sort(gv)[::-1][:k]
        idcg = float((ideal * disc[:len(ideal)]).sum())
        if idcg <= 0:
            continue
        vals.append(float((top * disc[:len(top)]).sum()) / idcg)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# 毎日の採点
# --------------------------------------------------------------------------- #

def load(model_dir: str):
    """保存済みのランキング学習（model, meta）。無ければ (None, None)（画面と追跡は出さないだけで、予測は止めない）。"""
    import models as M

    d = M.sub_dir(NAME, model_dir)
    mp, tp = os.path.join(d, "model.joblib"), os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None, None
    import joblib

    with open(tp, encoding="utf-8") as fh:
        meta = json.load(fh)
    return joblib.load(mp), meta


def day_fields(dates, scores, hist: Optional[np.ndarray] = None) -> List[Dict]:
    """
    候補ごとの画面の値: score・dayPct（その日の中の百分位 0〜100、小数1桁）・dayRank（その日の中の順位。1 = 最上位）・
    nDay（その日の候補数）・upper（dayPct が DAY_UPPER 以上か）・pctHistorical（OOF の分布での位置。無ければ None）。
    """
    import models as M

    dates = pd.Series(pd.to_datetime(pd.Series(dates)).to_numpy())
    sc = np.asarray(scores, dtype=float)
    p = within_date_pct(dates, sc)
    s = pd.Series(sc)
    rank = s.groupby(dates.to_numpy()).rank(ascending=False, method="min").to_numpy()
    n = s.groupby(dates.to_numpy()).transform("size").to_numpy()
    out = []
    for i in range(len(sc)):
        out.append({"score": round(float(sc[i]), 6), "dayPct": round(float(p[i]) * 100, 1),
                    "dayRank": int(rank[i]), "nDay": int(n[i]), "upper": bool(p[i] >= DAY_UPPER),
                    # ほかのモデルと同じ物差し（models.pct_historical: 過去の分布のうち、このスコアより低い割合）
                    "pctHistorical": M.pct_historical(float(sc[i]), hist)})
    return out
