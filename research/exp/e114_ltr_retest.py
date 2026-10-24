#!/usr/bin/env python3
"""
実験114: ランキング学習（LambdaRank）を今の条件で確かめ直す（検証だけ。基準モデル・規則・画面は変えない）。

運用者の依頼（2026-10-24）「今の目的変数の作り方的に、予測スコアの順と、リターンの順が一致していないということなので、
以前に実施した、ランキング学習の検証を、今このタイミングで改めて実施したいです。あのときからは、バグも直しましたし、
特徴量も増えたので。」→ AskUserQuestion: 学ぶ並び「今の売り方の収益」・木の形「専用に探索する」・判定「前回決めた基準」。
同じ日の運用者の指摘「今の売り方の収益だと、ランキングにならない気がします。10%で売るので。単純に、20営業日後のリターン
（その日の終値ベース）で良い気がします」（実測: 今の売り方では同じ日の候補の 31.1% が +10% ちょうどで同点、20営業日目の
終値では 0.11%）→ AskUserQuestion: 測り方「寄りで買い20日目の終値」・判定の収益「今の売り方の損益」・比較の腕「20日の終値だけ」。

前回（実験52 の L、2026-09-26、docs/MODEL_ADOPTION_RULES.md §15）から変えたところ
  - データ: 分割・併合の後の調整後終値の直し（2026-10-16 に全期間を取り直し）の後のもの
  - 列: 206 / 212列 → 360列（features.DEFAULT_PRESET）
  - 学ぶ並び: ret_o1_20（20営業日目までの5日平均終値）の日付内5分位 → 翌営業日の寄りで買い20営業日目の終値で売った収益
    （live_track の ret_close。今の売り方で指値に届かなかったときの売りと同じ値）の日付内5分位
  - 木の形: 分類器の探索結果の流用 → L 専用に探索（目的は日付内の上位5件の当たり NDCG@5）
  - 基準モデル（C）も 360列で探索し直す。本番の 1.2σ の LightGBM は 2026-10-18 に 239列で探索・学習したもので、360列の
    最初の探索は日曜 10/25 の週次。C はその週次と同じ関数・同じ設定（run_tuning.py → tuning.tune）で探索する

腕（LightGBM・木200本）
  C  分類（基準モデルと同じ作り: binary + scale_pos_weight、ラベルは 1.2σ。train_production.oof_scores と同じ当てはめ）
  L  LambdaRank（LGBMRanker）。同じ日の候補を1つのグループにし、段階は同じ日の中の20営業日目の終値の収益の5分位（0〜4。
     利得は段階そのもの。実験52 と同じ作り）

探索（どちらも本番の探索と同じ枠組み: ホールドアウトより前・年×時価総額帯で層別・日付単位の5分割・種0・50試行・木200本固定・
探索空間は tuning.tune と同じ）
  C  目的は PR-AUC（本番どおり。tuning.tune をそのまま呼ぶ）
  L  目的は検証分割の日付内 NDCG@5（候補5件以上の日の平均。利得は20営業日目の終値の段階）
  探索の後に、両方の木の形を同じ5分割で NDCG@5・日付内の順位相関（20営業日目の終値）・分割の中の上位10% の平均収益
  （今の売り方）・PR-AUC でも測って並べる
  （並べ替えた順位で測った NDCG@5 も「でたらめな順」の基準として出す）

評価（基準モデルの out-of-fold と同じ窓を 0/2/4か月ずらした3通り × 種3つ（42 / 7 / 123）。種のスコアは平均。
収益は今の売り方（翌営業日の寄りで買い、+10% の指値、届かなければ20営業日目の終値）で、−10% 未満は「−10% より悪い」。
学ぶ並びの20営業日目の終値の平均も参考に並べる）
  1. 【判定】窓の中の上位10%（件数をそろえる）: 窓ごとにスコアの上から窓の行数の10% を取り、平均収益・−10% 未満の割合
     （窓の平均。実験52 の「窓の中の上位10%」と同じ数え方）。L − C の窓ごとの差も
  2. 窓の中の上位5%（同じ）
  3. 基準モデルの95以上と同じ件数: 窓ごとに C が「前の窓までの分布の上位5%」を超えた件数と同じ件数を L の上から取る
     （運用の形の代わり。今の規則は XGBoost・CatBoost も要るので、ここでは基準モデル1本の線で件数を決める）
  4. 前の窓までの分布の上位5% / 10% を超えたら買う（閾値ルール。参考）
  5. 日付内: 1位・上位2件の平均収益（発火8件以上の日）、日付内の順位相関（スコアと今の売り方の収益・20営業日目の終値。
     5件以上の日）
  6. 参考: 1.2σ のラベルで測った PR-AUC / ROC-AUC / 日内AUC、上位10% のボラの帯

判定（結果を見る前に決めたもの。§15 で決めた基準を今の売り方で測る）
  3つのずらし方のすべてで、L の「窓の中の上位10%（件数をそろえる）」の平均収益が C より高く、かつ −10% 未満の割合が C より
  低いとき「基準を満たす」。満たしても採用するかは運用者が決める（検証だけ）

公開ログには集計だけ（件数・平均・割合・AUC・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e114_*。

    exp=e114_ltr_retest.py  args="--trials 50 --shifts 0,2,4 --seeds 3"
    --skip-tune は e114_params.json が同じデータ・同じ列で作られていればそれを読む
    --rule / --quick は実験114b（今の規則の形 / 基準モデル1本の速報）を回す（research/exp/e114b_ltr_rule.py）
    --show-params は置き場の探索の結果を丸めずに出す。--check-ltr は本番の脇に置いたランキング学習が L と同じかを確かめる

結果（2026-10-24、run 40070769689、f5d652a、実験の部分 62分。docs/MODEL_ADOPTION_RULES.md §78）
------------------------------------------------------------------
  確かめ: 5日平均と ret_o1_20 が 22,012 / 22,012 行で一致。同点 今の売り方 31.1% / 20営業日目の終値 0.11%
  探索: C の分割平均 PR-AUC 0.3476、L の NDCG@5 0.6587（でたらめな順 0.5998、C の木の形で 0.6396）
  判定: 3通りすべてで基準を満たす。窓の中の上位10%（件数そろえ）C +1.24 / +1.09 / +1.11% → L +1.47 / +1.69 / +1.20%、
        −10% 未満 10.3 / 10.8 / 10.8% → 9.5 / 8.5 / 9.4%。窓ごとの差 +0.23 ± 0.21（6/11）/ +0.60 ± 0.35（8/11）/ +0.09 ± 0.42pt（4/10）
  基準モデルの95以上と同じ件数: +0.70 / +0.13 / −0.00pt。L の上位10% は低ボラに偏らない（最も低い帯 29% 対 C 52〜56%）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import tuning  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

PREFIX = "e114"
OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
OUTCOME = lab.OUTCOME                   # ret_o1_20（出口の計算の確かめと参考）
TARGET = float(LT.TAKE_PROFIT)          # +10% の指値
HOLD = int(LT.HOLD)                     # 20営業日
RET = "ret_tp"                          # 今の売り方の収益（比。指値に届けば 0.10）。判定の物差し
R20 = "ret_c20"                         # 翌営業日の寄りで買い20営業日目の終値で売った収益（比）。L が学ぶ並び
HITDAY = "hitday_tp"                    # 指値に届いた営業日（1 = 買った日。届かなければ空）。3枠の形の売買（実験114b）に使う
HIT = "hit_tp"                          # 指値に届いたか
N_GRADES = E52.N_GRADES                 # 段階の数（5）
NDCG_K = 5
MIN_DAY = 5                             # NDCG・順位相関に使う日の最小の候補数
MIN_BREAKS = E52.MIN_BREAKS             # 日付内の1位・上位2件に使う日の最小の候補数（8）
BAD = -0.10                             # 大負けの線
SEED = 0                                # 探索の種（本番と同じ）
N_SPLITS = 5
SEEDS = E27.SEEDS3                      # 42 / 7 / 123（実験52 と同じ）
ARMS = ("C", "L")
LABELS = {"C": "C 基準モデル（分類）", "L": "L LambdaRank（20日目の終値）"}
TOP_FRACS = (0.10, 0.05)
#: 木の形として記録・表示する鍵
TREE_KEYS = ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
             "reg_alpha", "reg_lambda")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 収益と段階
# --------------------------------------------------------------------------- #

def attach_exit(df: pd.DataFrame, bars: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    今の売り方の収益 RET（比）・指値に届いたかの印 HIT・20営業日目の終値で売った収益 R20（比）を足す。計算は実験83 の
    attach_exits（live_track.forward）をそのまま使う（買値は翌営業日の寄り、1〜20営業日目の高値が +10% に届けば +10%、
    届かなければ20営業日目の終値。R20 は指値に関係なく20営業日目の終値）。
    返す確かめ（chk）の hold5_match / hold5_both は「同じ計算の5日平均が ret_o1_20 と一致した行 / 両方ある行」。
    """
    base = df[["Code", "Date", OUTCOME]].copy()
    out, chk = E83.attach_exits(base, bars, targets=(TARGET,), hold=HOLD)
    k = E83.tp_key(TARGET)
    m = out[["Code", "Date", f"tp{k}c", f"hit{k}", f"hitday{k}", "c20"]].rename(
        columns={f"tp{k}c": RET, f"hit{k}": HIT, f"hitday{k}": HITDAY, "c20": R20})
    m[RET] = pd.to_numeric(m[RET], errors="coerce") / 100.0
    m[R20] = pd.to_numeric(m[R20], errors="coerce") / 100.0
    res = df.merge(m, on=["Code", "Date"], how="left", validate="one_to_one")
    return res, chk


def grades(dates, r) -> np.ndarray:
    """同じ日の中の収益の5分位（0〜4 の整数。1件だけの日は真ん中の 2）。実験52 の within_date_grade と同じ。"""
    return E52.within_date_grade(pd.Series(np.asarray(dates)), np.asarray(r, dtype=float), N_GRADES)


def ndcg_at_k(dates, score, gain, k: int = NDCG_K, min_n: int = MIN_DAY) -> float:
    """
    日付内 NDCG@k の平均（候補 min_n 件以上の日だけ。理想の並びの DCG が 0 の日は除く）。利得は gain そのもの
    （LGBMRanker の label_gain = 0〜4 と同じ）、割引は 1 / log2(順位 + 1)。同じスコアは入力の順。
    """
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
# 当てはめ
# --------------------------------------------------------------------------- #

def tree_only(params: Dict) -> Dict:
    """木の形だけ（目的関数・不均衡の補正・種は腕が決めるので外す）。"""
    return {k: v for k, v in params.items()
            if not str(k).startswith("_") and k not in ("objective", "scale_pos_weight", "random_state")}


def fit_score(arm: str, params: Dict, tr: pd.DataFrame, te: pd.DataFrame, cols: Sequence[str],
              seed: int) -> np.ndarray:
    """腕ごとに tr で学習して te を採点する。"""
    import lightgbm as lgb

    p = tree_only(params)
    p["random_state"] = int(seed)
    p.setdefault("verbose", -1)
    p.setdefault("n_jobs", -1)
    cols = list(cols)
    Xte = te[cols].to_numpy(dtype=float)
    if arm == "C":
        # train_production.oof_scores と同じ（binary + その窓の訓練の正例率から作る scale_pos_weight）
        y = tr["label"].to_numpy(dtype=int)
        m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=tuning.scale_pos_weight(y), **p)
        m.fit(tr[cols].to_numpy(dtype=float), y)
        return m.predict_proba(Xte)[:, 1]
    if arm == "L":
        # LGBMRanker はグループ（日付）が続いて並ぶことを要る。(Date, Code) の順に固定する（実験52 と同じ）
        d = tr.sort_values(["Date", "Code"], kind="mergesort")
        y = grades(d["Date"], d[R20].to_numpy(dtype=float))
        sizes = d.groupby("Date", sort=True).size().to_numpy()
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(N_GRADES)), **p)
        m.fit(d[cols].to_numpy(dtype=float), y, group=sizes)
        return m.predict(Xte)
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
# 探索
# --------------------------------------------------------------------------- #

def cv_metrics(arm: str, params: Dict, folds, cols: Sequence[str], seed: int = SEED) -> Dict:
    """
    探索の5分割それぞれで学習→検証し、NDCG@5・日付内の順位相関（どちらも20営業日目の終値）・分割の中の上位10% の平均収益
    （今の売り方）・PR-AUC（分割の平均）。
    """
    from sklearn.metrics import average_precision_score

    nd, rho, top, pr = [], [], [], []
    for tr, va in folds:
        tr_ = tr[tr[R20].notna()] if arm == "L" else tr
        va_ = va[va[RET].notna() & va[R20].notna()]
        s = fit_score(arm, params, tr_, va_, cols, seed)
        r20 = va_[R20].to_numpy(dtype=float)
        nd.append(ndcg_at_k(va_["Date"], s, grades(va_["Date"], r20)))
        rho.append(E53.date_rho(va_["Date"], s, r20, min_n=MIN_DAY))
        top.append(E53.top_mean(s, va_[RET].to_numpy(dtype=float))["ret"])
        pr.append(float(average_precision_score(va_["label"].to_numpy(dtype=int), s)))
    return {"ndcg": float(np.mean(nd)), "ndcg_folds": [round(x, 4) for x in nd],
            "rho": float(np.nanmean(rho)), "top10": float(np.mean(top)), "pr_auc": float(np.mean(pr))}


def random_ndcg(folds, seed: int = SEED) -> float:
    """でたらめな順で測った NDCG@5（探索の目的の値の基準）。"""
    rng = np.random.default_rng(seed)
    vals = []
    for _, va in folds:
        va_ = va[va[RET].notna() & va[R20].notna()]
        r20 = va_[R20].to_numpy(dtype=float)
        vals.append(ndcg_at_k(va_["Date"], rng.random(len(va_)), grades(va_["Date"], r20)))
    return float(np.mean(vals))


def tune_c(sub: pd.DataFrame, cols: Sequence[str], n_trials: int) -> Dict:
    """基準モデルの探索。日曜の週次（run_tuning.py）と同じ呼び方。"""
    t0 = time.time()
    params = tuning.tune(sub, list(cols), n_trials=n_trials, n_splits=N_SPLITS, embargo_days=EMBARGO_DAYS,
                         scheme=tuning.PRODUCTION_CV, model="classifier", objective=tuning.PRODUCTION_OBJECTIVE)
    return {"params": params, "_cv": dict(tuning.LAST_CV), "_minutes": round((time.time() - t0) / 60, 1)}


def tune_l(sub_l: pd.DataFrame, cols: Sequence[str], n_trials: int, folds) -> Dict:
    """LambdaRank の探索（探索空間は tuning.tune と同じ。目的は検証分割の日付内 NDCG@5 の分割平均）。"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()

    def objective(trial):
        p = E53.search_params(trial)
        m = cv_metrics("L", p, folds, cols)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["ndcg"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {"boosting_type": "gbdt", "n_estimators": tuning.SEARCH_N_ESTIMATORS, "subsample_freq": 1,
            "n_jobs": -1, "verbose": -1, **study.best_params}
    at = study.best_trial.user_attrs
    return {"params": best,
            "_cv": {"objective": f"検証分割の日付内 NDCG@{NDCG_K}（候補{MIN_DAY}件以上の日。利得は20営業日目の終値の段階 "
                                 f"0〜{N_GRADES - 1}）",
                    "n_splits": len(folds), "n_trials": n_trials, "best_trial": int(study.best_trial.number),
                    "ndcg": round(float(study.best_value), 4), "ndcg_folds": at.get("ndcg_folds"),
                    "rho": round(float(at.get("rho", np.nan)), 4), "top10": round(float(at.get("top10", np.nan)), 5),
                    "pr_auc": round(float(at.get("pr_auc", np.nan)), 4),
                    "trials": [{"n": t.number, "ndcg": round(float(t.value), 4)} for t in study.trials]},
            "_minutes": round((time.time() - t0) / 60, 1)}


def data_sig(cols: Sequence[str], sub: pd.DataFrame, n_trials: int) -> Dict:
    """探索の結果を使い回してよいかの目印（列・探索期間・行数・試行数が同じなら同じ）。"""
    return {"features_sig": F.signature(list(cols)), "n_features": len(cols), "rows": int(len(sub)),
            "train_to": str(pd.to_datetime(sub["Date"]).max().date()), "n_trials": int(n_trials),
            "ret_rows": int(sub[RET].notna().sum()), "r20_rows": int(sub[R20].notna().sum()), "target": R20}


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

def run_tag(arm: str, params: Dict, cols: Sequence[str], df: pd.DataFrame) -> str:
    """保存名に入れる目印（木の形・列・データが変わったら別の名前にして、古い out-of-fold を読まない）。"""
    key = {"arm": arm, "params": {k: tree_only(params)[k] for k in sorted(tree_only(params))},
           "cols": F.signature(list(cols)), "rows": int(len(df)), "last": str(df["Date"].max().date()),
           "ret_rows": int(df[RET].notna().sum()), "r20_rows": int(df[R20].notna().sum()), "target": R20, "v": 2}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_arm(df: pd.DataFrame, cols: Sequence[str], arm: str, params: Dict, shift: int,
            seeds: Sequence[int], out_dir: str = OOF_DIR) -> pd.DataFrame:
    """
    腕・ずらし方ごとの out-of-fold（種の平均）。窓は基準モデルの out-of-fold と同じ作り（e41.folds_for。ずらし0 は
    train_production.oof_scores と同じ窓）。学習: C は 1.2σ のラベルがある行（本番と同じ）、L はさらに20営業日目の終値が
    ある行。採点: 窓の中のラベル・今の売り方の収益・20営業日目の終値がある行（両腕で同じ行）。
    """
    folds = E41.folds_for(df["Date"], shift)
    d = pd.to_datetime(df["Date"])
    keep = [c for c in ("Code", "Date", "label", RET, HIT, R20, OUTCOME, "vol_20d") if c in df.columns]
    tag = run_tag(arm, params, cols, df)
    parts = []
    for sd in seeds:
        path = os.path.join(out_dir, f"{PREFIX}_{arm}_{tag}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        rows = []
        for f in folds:
            trm = (d <= pd.Timestamp(f.train_end)) & df["label"].notna()
            if arm == "L":
                trm &= df[R20].notna()
            tem = ((d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                   & df["label"].notna() & df[RET].notna() & df[R20].notna())
            tr, te = df[trm], df[tem]
            if len(te) < 200 or len(tr) < 1000:
                continue
            part = te[keep].copy()
            part["score"] = fit_score(arm, params, tr, te, cols, sd)
            part["fold"] = int(f.index)
            rows.append(part)
        if not rows:
            raise SystemExit(f"腕{arm} ずらし{shift}か月: 窓を作れない（期間が短すぎる）")
        o = pd.concat(rows, ignore_index=True)
        os.makedirs(out_dir, exist_ok=True)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  腕{arm} ずらし{shift}か月 種{sd}: {o['fold'].nunique()}窓・{len(o):,}件 {time.time() - t0:.0f}秒")
    ref = parts[0][["Code", "Date"]].astype(str).to_numpy()
    for x in parts[1:]:
        if not np.array_equal(ref, x[["Code", "Date"]].astype(str).to_numpy()):
            raise SystemExit(f"腕{arm} ずらし{shift}か月: 種によって行が違う")
    o = parts[0].copy()
    o["score"] = np.mean([x["score"].to_numpy(dtype=float) for x in parts], axis=0)
    o["Date"] = pd.to_datetime(o["Date"])
    return o


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def _stats(sel: pd.DataFrame) -> Dict:
    r = sel[RET].to_numpy(dtype=float)
    return {"n": int(len(sel)), "ret": float(np.mean(r)) if len(r) else np.nan,
            "bad": float(np.mean(r < BAD)) if len(r) else np.nan,
            "hit": float(sel[HIT].astype(float).mean()) if len(sel) else np.nan,
            "pos": float(sel["label"].mean()) if len(sel) else np.nan,
            "r20": float(sel[R20].mean()) if (len(sel) and R20 in sel.columns) else np.nan}


def top_within(o: pd.DataFrame, frac: float) -> pd.DataFrame:
    """窓ごとにスコアの上から窓の行数の frac（四捨五入・最低1件）を取る。同じ行の腕どうしなら件数は同じ。"""
    rows = []
    for f, g in o.groupby("fold", sort=True):
        n = max(1, int(round(len(g) * frac)))
        sel = g.sort_values("score", ascending=False, kind="mergesort").head(n)
        rows.append({"fold": int(f), **_stats(sel)})
    return pd.DataFrame(rows, columns=["fold", "n", "ret", "bad", "hit", "pos", "r20"])


def forward_counts(o: pd.DataFrame, pct: float = 95.0, min_ref: int = 500) -> Dict[int, int]:
    """窓ごとに、前の窓までのスコアの分布の pct 分位を超えた件数（窓1など参照が min_ref 件未満の窓は数えない）。"""
    out = {}
    for f in sorted(o["fold"].unique()):
        ref = o.loc[o["fold"] < f, "score"].to_numpy(dtype=float)
        if len(ref) < min_ref:
            continue
        cur = o[o["fold"] == f]
        out[int(f)] = int((cur["score"].to_numpy(dtype=float) > np.percentile(ref, pct)).sum())
    return out


def matched(o: pd.DataFrame, counts: Dict[int, int]) -> pd.DataFrame:
    """窓ごとに counts 件だけスコアの上から取る（件数を基準モデルの線にそろえた選定）。"""
    rows = []
    for f, n in sorted(counts.items()):
        g = o[o["fold"] == f]
        if n <= 0 or not len(g):
            continue
        sel = g.sort_values("score", ascending=False, kind="mergesort").head(n)
        rows.append({"fold": int(f), **_stats(sel)})
    return pd.DataFrame(rows, columns=["fold", "n", "ret", "bad", "hit", "pos", "r20"])


def threshold_picks(o: pd.DataFrame, pct: float, min_ref: int = 500) -> pd.DataFrame:
    """前の窓までの分布の pct 分位を超えたら買う（閾値ルール。運用の形。参照の足りない窓は対象外）。"""
    sel = []
    for f in sorted(o["fold"].unique()):
        ref = o.loc[o["fold"] < f, "score"].to_numpy(dtype=float)
        if len(ref) < min_ref:
            continue
        cur = o[o["fold"] == f]
        sel.append(cur[cur["score"].to_numpy(dtype=float) > np.percentile(ref, pct)])
    return pd.concat(sel) if sel else o.iloc[:0]


def within_date(o: pd.DataFrame) -> Dict:
    """
    その日の1位・上位2件の平均収益（候補 MIN_BREAKS 件以上の日。今の売り方）と、日付内の順位相関（MIN_DAY 件以上の日。
    今の売り方の収益 rho と20営業日目の終値 rho20）。
    """
    top1, top2 = [], []
    for _, g in o.groupby("Date", sort=True):
        if len(g) < MIN_BREAKS:
            continue
        r = g.sort_values("score", ascending=False, kind="mergesort")[RET].to_numpy(dtype=float)
        top1.append(r[0])
        top2.append(np.mean(r[:2]))
    return {"top1": float(np.mean(top1)) if top1 else np.nan, "top2": float(np.mean(top2)) if top2 else np.nan,
            "days": len(top1),
            "rho": E53.date_rho(o["Date"], o["score"].to_numpy(dtype=float), o[RET].to_numpy(dtype=float),
                                min_n=MIN_DAY),
            "rho20": (E53.date_rho(o["Date"], o["score"].to_numpy(dtype=float), o[R20].to_numpy(dtype=float),
                                   min_n=MIN_DAY) if R20 in o.columns else np.nan)}


def vol_mix(o: pd.DataFrame, frac: float = 0.10) -> pd.DataFrame:
    """窓の中の上位 frac のボラの帯の構成と帯ごとの平均収益（帯の境界は実験52 と同じ）。"""
    sel = []
    for _, g in o.groupby("fold", sort=True):
        n = max(1, int(round(len(g) * frac)))
        sel.append(g.sort_values("score", ascending=False, kind="mergesort").head(n))
    s = pd.concat(sel)
    band = pd.cut(s["vol_20d"], E52.VOL_EDGES, labels=E52.VOL_NAMES, include_lowest=True)
    t = s.groupby(band, observed=False).agg(n=(RET, "size"), ret=(RET, "mean"))
    t["share"] = t["n"] / max(1, t["n"].sum())
    return t


def fold_diff(base: pd.DataFrame, cur: pd.DataFrame, key: str = "ret") -> Dict:
    """窓ごとの差（cur − base）の平均・標準誤差・上回った窓の数。"""
    m = base.merge(cur, on="fold", suffixes=("_c", "_a"))
    dv = (m[f"{key}_a"] - m[f"{key}_c"]).to_numpy(dtype=float)
    if not len(dv):
        return {"mean": np.nan, "se": np.nan, "won": 0, "n": 0, "each": []}
    se = float(dv.std(ddof=1) / np.sqrt(len(dv))) if len(dv) > 1 else np.nan
    return {"mean": float(dv.mean()), "se": se, "won": int((dv > 0).sum()), "n": int(len(dv)),
            "each": [round(float(x), 4) for x in dv]}


def verdict(top10: Dict[int, Dict[str, pd.DataFrame]]) -> Dict:
    """
    判定（結果を見る前に決めたもの）: すべてのずらし方で、L の窓の中の上位10% の平均収益（窓の平均）が C より高く、
    −10% 未満の割合（窓の平均）が C より低い。
    """
    rows = []
    for sh in sorted(top10):
        c, l_ = top10[sh]["C"], top10[sh]["L"]
        rc, rl = float(c["ret"].mean()), float(l_["ret"].mean())
        bc, bl = float(c["bad"].mean()), float(l_["bad"].mean())
        rows.append({"shift": int(sh), "ret_C": rc, "ret_L": rl, "bad_C": bc, "bad_L": bl,
                     "ret_ok": bool(rl > rc), "bad_ok": bool(bl < bc), "ok": bool(rl > rc and bl < bc)})
    return {"rows": rows, "pass": bool(rows) and all(r["ok"] for r in rows)}


def _pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:+.{nd}f}%" if np.isfinite(x) else "—"


def _share(x: float) -> str:
    return f"{x * 100:.1f}%" if np.isfinite(x) else "—"


def _tree_line(params: Dict) -> str:
    return " / ".join(f"{k} {params[k]:.4g}" if isinstance(params[k], float) else f"{k} {params[k]}"
                      for k in TREE_KEYS if k in params)


# --------------------------------------------------------------------------- #

def load_frame(cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.Timestamp, Dict]:
    """データセット（実収益付き）に今の売り方の収益を足す。探索の打ち切り日は run_tuning.py と同じ（データセットの全日付から）。"""
    full = lab.frame()
    miss = [c for c in cols if c not in full.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:6]}。research/build_dataset.py を回し直してください")
    full = full.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    cutoff, _, _ = holdout_bounds(full["Date"], HOLDOUT_MONTHS, EMBARGO_DAYS)
    bars = LT.load_bars(lab.DATA_DIR, start=full["Date"].min() - pd.Timedelta(days=10))
    full, chk = attach_exit(full, bars)
    return full, pd.Timestamp(cutoff), chk


def show_params(path: str = PARAMS_PATH) -> int:
    """探索の結果（C と L の木の形・探索の目印）を JSON で出す（小数は丸めない。試行ごとの記録は出さない）。"""
    if not os.path.exists(path):
        raise SystemExit(f"実験114 の探索の結果が無い（{path}）")
    with open(path, encoding="utf-8") as fh:
        store = json.load(fh)
    out = {"_sig": store.get("_sig"), "_cutoff": store.get("_cutoff")}
    for arm in ARMS:
        rec = store.get(arm) or {}
        cv = {k: v for k, v in (rec.get("_cv") or {}).items() if k != "trials"}
        out[arm] = {"params": rec.get("params"), "_cv": cv}
    print("[探索の結果] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験114: ランキング学習を今の条件で確かめ直す（検証だけ）")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--rule", action="store_true",
                    help="実験114b（今の規則の形で確かめる）を回す。実験114 の探索の結果と L の out-of-fold を同じ置き場から読む")
    ap.add_argument("--quick", action="store_true",
                    help="実験114b の速報（基準モデル1本＋ランキング学習で両方使う形）を回す。C と L の out-of-fold を同じ置き場から読む")
    ap.add_argument("--show-params", action="store_true",
                    help="置き場にある探索の結果（C と L の木の形。小数を丸めない）を出して終わる。ランキング学習を本番の脇に"
                         "写すため（research/ltr_params.json。運用者の選択「まず並べて見る」2026-10-24）")
    ap.add_argument("--check-ltr", action="store_true",
                    help="本番の脇に置いたランキング学習（Release の ltr_oof.parquet・research/ltr_params.json）が、"
                         "実験114 の L（種42・ずらし0）と同じかを確かめる（research/exp/e114b_ltr_rule.py の check_ltr）")
    args = ap.parse_args(argv)
    if args.show_params:
        return show_params()
    if args.check_ltr:
        import e114b_ltr_rule as RB
        return RB.main(["--check-ltr"])
    if args.rule or args.quick:
        import e114b_ltr_rule as RB
        return RB.main(["--quick"] if args.quick else [])
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = SEEDS[:args.seeds]
    cols = F.columns(F.DEFAULT_PRESET)
    os.makedirs(OOF_DIR, exist_ok=True)
    # 同じ日の収益がすべて同じ（指値の +10% ちょうどばかりの日など）と順位相関が定まらない。その日は数えない（警告は出さない）
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    t_all = time.time()
    full, cutoff, chk = load_frame(cols)
    df = full[full["label"].notna()].reset_index(drop=True)
    print("=" * 96)
    print(f"実験114 ランキング学習を今の条件で確かめ直す（検証だけ）: {len(df):,}件（ラベルあり）・"
          f"{df['Date'].nunique():,}日・{df['Date'].min().date()} 〜 {df['Date'].max().date()} / "
          f"{F.DEFAULT_PRESET} {len(cols)}列（指紋 {F.signature(cols)}）/ 種{len(seeds)}つ / ずらし {shifts}か月")
    print(f"  判定の収益: 翌営業日の寄りで買い、+{TARGET:g}% の指値、届かなければ{HOLD}営業日目の終値（live_track と同じ）/ "
          f"L が学ぶ並び: 翌営業日の寄りで買い{HOLD}営業日目の終値の、同じ日の中の{N_GRADES}分位")
    print("=" * 96)

    # ---- 確かめ: 出口の計算 ----
    with_ret = int(df[RET].notna().sum())
    match = chk["hold5_match"] / max(1, chk["hold5_both"])
    log(f"[確かめ] 収益がある行 {with_ret:,} / {len(df):,}（無いのは寄り付かず・日足なし・20営業日に満たない行）・"
        f"+{TARGET:g}% に届いた {df[HIT].mean() * 100:.1f}% / 同じ計算の5日平均が {OUTCOME} と一致 "
        f"{chk['hold5_match']:,} / {chk['hold5_both']:,} 行（{match * 100:.2f}%）")
    if chk["hold5_both"] and match < 0.99:
        raise SystemExit("出口の計算が ret_o1_20 と合わない（日足と学習データの株価がずれている）")
    lab_rows = df[df[RET].notna() & df[R20].notna()]
    ties = {c: float(lab_rows.groupby(["Date", c]).size().pipe(lambda s: s[s > 1].sum()) / len(lab_rows))
            for c in (RET, R20)}
    log(f"[確かめ] 同じ日の中でほかの候補とちょうど同じ値の行: 今の売り方 {ties[RET] * 100:.1f}% / "
        f"{HOLD}営業日目の終値 {ties[R20] * 100:.2f}%（L は後者で学ぶ）")
    g_all = grades(lab_rows["Date"], lab_rows[R20].to_numpy(dtype=float))
    sz = lab_rows.groupby("Date").size()
    log(f"[確かめ] 段階の割合（全期間・同じ日の中の{HOLD}営業日目の終値の5分位）: "
        + " / ".join(f"{k} {np.mean(g_all == k) * 100:.1f}%" for k in range(N_GRADES))
        + f" ｜ 1日の候補 中央 {int(sz.median())}件・{MIN_DAY}件以上の日 {np.mean(sz >= MIN_DAY) * 100:.1f}%"
          f"（行の {sz[sz >= MIN_DAY].sum() / sz.sum() * 100:.1f}%）")
    f0 = E41.folds_for(df["Date"], 0)
    log(f"[確かめ] ずらし0の窓 {len(f0)}本（基準モデルの out-of-fold と同じ作り）: "
        f"最初のテスト {pd.Timestamp(f0[0].test_start).date()} 〜 最後 {pd.Timestamp(f0[-1].test_end).date()}")

    # ---- 探索 ----
    sub = full[full["Date"] <= cutoff].reset_index(drop=True)
    if sub["label"].isna().any():
        raise SystemExit(f"探索の期間にラベルの無い行がある（{int(sub['label'].isna().sum())}行）")
    sub_l = sub[sub[RET].notna() & sub[R20].notna()].reset_index(drop=True)
    sig = data_sig(cols, sub, args.trials)
    log(f"[探索] 期間 〜{cutoff.date()}（run_tuning.py と同じ打ち切り）: {len(sub):,}件・正例率 {sub['label'].mean() * 100:.1f}%"
        f" / L は収益がある {len(sub_l):,}件")
    store = None
    if args.skip_tune and os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("_sig") == sig:
            store = prev
            log("[探索] 保存済みの探索を読む（同じデータ・同じ列・同じ試行数）")
        else:
            log("[探索] 保存済みの探索は別のデータ・列のものなので探索し直す")
    folds_l = tuning.cv_folds(sub_l, tuning.PRODUCTION_CV, n_splits=N_SPLITS, seed=SEED, embargo_days=EMBARGO_DAYS)
    if not folds_l:
        raise SystemExit("探索の分割を作れない")
    if store is None:
        log(f"[探索] C（本番と同じ tuning.tune・目的 {tuning.PRODUCTION_OBJECTIVE}・{args.trials}試行）")
        rec_c = tune_c(sub, cols, args.trials)
        log(f"[探索] C {rec_c['_minutes']}分 / 分割平均 PR-AUC {rec_c['_cv'].get('mean_pr_auc')} / {_tree_line(rec_c['params'])}")
        log(f"[探索] L（LambdaRank・目的 NDCG@{NDCG_K}・{args.trials}試行）")
        rec_l = tune_l(sub_l, cols, args.trials, folds_l)
        log(f"[探索] L {rec_l['_minutes']}分 / 分割平均 NDCG@{NDCG_K} {rec_l['_cv']['ndcg']} / {_tree_line(rec_l['params'])}")
        store = {"_sig": sig, "_cutoff": str(cutoff.date()), "C": rec_c, "L": rec_l}
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2, default=str)
    params = {"C": store["C"]["params"], "L": store["L"]["params"]}
    print(f"\n■ 探索の結果を同じ5分割で並べる（分割の平均。でたらめな順の NDCG@{NDCG_K} {random_ndcg(folds_l):.4f}）")
    print(f"  {'腕':<22}{'NDCG@5':>9}{'日付内ρ':>9}{'上位10%':>10}{'PR-AUC':>9}  木の形"
          f"（NDCG と ρ は{HOLD}営業日目の終値、上位10% は今の売り方の平均収益）")
    cvc = {}
    for arm in ARMS:
        m = cv_metrics(arm, params[arm], folds_l, cols)
        cvc[arm] = m
        print(f"  {LABELS[arm]:<20}{m['ndcg']:>9.4f}{m['rho']:>+9.3f}{_pct(m['top10']):>10}{m['pr_auc']:>9.4f}  "
              f"{_tree_line(params[arm])}")

    # ---- out-of-fold と評価 ----
    summary: Dict = {"sig": sig, "cutoff": str(cutoff.date()), "cv": cvc, "shifts": {}}
    top10: Dict[int, Dict[str, pd.DataFrame]] = {}
    for sh in shifts:
        res = {arm: oof_arm(df, cols, arm, params[arm], sh, seeds) for arm in ARMS}
        if not np.array_equal(res["C"][["Code", "Date"]].astype(str).to_numpy(),
                              res["L"][["Code", "Date"]].astype(str).to_numpy()):
            raise SystemExit(f"ずらし{sh}か月: 腕によって採点した行が違う")
        o_c = res["C"]
        print(f"\n■ ずらし{sh}か月（{o_c['fold'].nunique()}窓・{len(o_c):,}件・今の売り方の平均 {_pct(o_c[RET].mean())}・"
              f"−10% 未満 {_share(np.mean(o_c[RET] < BAD))}・{HOLD}営業日目の終値の平均 {_pct(o_c[R20].mean())}）")
        sh_out: Dict = {}
        top10[sh] = {}
        for frac in TOP_FRACS:
            tag = f"top{int(round(frac * 100))}"
            print(f"  ◆ 窓の中の上位{frac * 100:.0f}%（件数をそろえる）: 件数 / 平均収益（窓の平均）/ −10%未満 / +{TARGET:g}%到達"
                  f" / 1.2σ の正例率 / 参考 {HOLD}営業日目の終値 ｜ L − C（窓ごと）")
            base = top_within(o_c, frac)
            for arm in ARMS:
                cur = base if arm == "C" else top_within(res[arm], frac)
                line = (f"    {LABELS[arm]:<20}{int(cur['n'].sum()):>6}件 {_pct(cur['ret'].mean()):>8} "
                        f"{_share(cur['bad'].mean()):>6} {_share(cur['hit'].mean()):>6} {_share(cur['pos'].mean()):>6} "
                        f"{_pct(cur['r20'].mean()):>8}")
                if arm != "C":
                    dd = fold_diff(base, cur)
                    db = fold_diff(base, cur, "bad")
                    line += (f" ｜ 収益 {dd['mean'] * 100:+.2f} ± {dd['se'] * 100:.2f}pt（上の窓 {dd['won']}/{dd['n']}）・"
                             f"−10%未満 {db['mean'] * 100:+.1f}pt")
                    sh_out[f"{tag}_diff"] = dd
                print(line)
                sh_out[f"{tag}_{arm}"] = {"n": int(cur["n"].sum()), "ret": float(cur["ret"].mean()),
                                          "bad": float(cur["bad"].mean()), "hit": float(cur["hit"].mean()),
                                          "r20": float(cur["r20"].mean()),
                                          "folds": json.loads(cur.to_json(orient="records"))}
                if frac == 0.10:
                    top10[sh][arm] = cur
        cnt = forward_counts(o_c, 95.0)
        print(f"  ◆ 基準モデルの95以上と同じ件数（窓ごとに C が前の窓までの分布の上位5% を超えた件数。{sum(cnt.values())}件・"
              f"{len(cnt)}窓）: 平均収益 / −10%未満 / +{TARGET:g}%到達")
        mc = matched(o_c, cnt)
        for arm in ARMS:
            cur = mc if arm == "C" else matched(res[arm], cnt)
            line = (f"    {LABELS[arm]:<20}{int(cur['n'].sum()):>6}件 {_pct(cur['ret'].mean()):>8} "
                    f"{_share(cur['bad'].mean()):>6} {_share(cur['hit'].mean()):>6}")
            if arm != "C":
                dd = fold_diff(mc, cur)
                line += f" ｜ 収益 {dd['mean'] * 100:+.2f} ± {dd['se'] * 100:.2f}pt（上の窓 {dd['won']}/{dd['n']}）"
                sh_out["c95_diff"] = dd
            print(line)
            sh_out[f"c95_{arm}"] = {"n": int(cur["n"].sum()), "ret": float(cur["ret"].mean()),
                                    "bad": float(cur["bad"].mean())}
        print("  ◆ 閾値ルール（前の窓までの分布の上位5% / 10% を超えたら買う。参考）: 件数 / 平均収益（全件）/ −10%未満")
        for pct in (95.0, 90.0):
            for arm in ARMS:
                s = threshold_picks(res[arm], pct)
                st = _stats(s)
                print(f"    上位{100 - pct:.0f}% {LABELS[arm]:<20}{st['n']:>6}件 {_pct(st['ret']):>8} {_share(st['bad']):>6}")
                sh_out[f"thr{int(100 - pct)}_{arm}"] = st
        print(f"  ◆ 日付内: 1位 / 上位2件の平均収益（候補{MIN_BREAKS}件以上の日・今の売り方）/ 順位相関（{MIN_DAY}件以上の日。"
              f"今の売り方 / {HOLD}営業日目の終値） ｜ 参考 PR-AUC / ROC-AUC / 日内AUC（1.2σ のラベル）")
        for arm in ARMS:
            w = within_date(res[arm])
            m = E52.label_metrics(res[arm])
            print(f"    {LABELS[arm]:<20}{_pct(w['top1']):>8} {_pct(w['top2']):>8}（{w['days']}日） "
                  f"{w['rho']:>+7.3f} / {w['rho20']:>+7.3f} ｜ {m['pr']:.4f} {m['roc']:.4f} {m['day_auc']:.4f}")
            sh_out[f"within_{arm}"] = w
            sh_out[f"label_{arm}"] = {k: float(v) for k, v in m.items()}
        print("  ◆ 窓の中の上位10% のボラの帯（20日ボラ%: 割合 / 今の売り方の平均収益）")
        for arm in ARMS:
            t = vol_mix(res[arm])
            print(f"    {LABELS[arm]:<20}" + " | ".join(f"{i} {r['share'] * 100:.0f}% {_pct(r['ret'], 1)}"
                                                       for i, r in t.iterrows()))
        summary["shifts"][str(sh)] = sh_out

    # ---- まとめと判定 ----
    v = verdict(top10)
    summary["verdict"] = v
    print(f"\n■ 判定（先に決めた基準: すべてのずらし方で、L の窓の中の上位10% の平均収益が C より高く、−10% 未満の割合が C より低い）")
    for r in v["rows"]:
        print(f"  ずらし{r['shift']}か月: 平均収益 C {_pct(r['ret_C'])} / L {_pct(r['ret_L'])}（{'○' if r['ret_ok'] else '×'}）・"
              f"−10%未満 C {_share(r['bad_C'])} / L {_share(r['bad_L'])}（{'○' if r['bad_ok'] else '×'}）")
    print(f"  → {'基準を満たす（採用するかは運用者が決める）' if v['pass'] else '基準を満たさない'}")
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
