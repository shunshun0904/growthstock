#!/usr/bin/env python3
"""
実験128: 「半年（125営業日）以内に買値の2倍」のラベルで、本番と同じ5分割の探索（373列・50試行）をし直してから、列の区分ごとの
重要度（gain の割合）を測り直す（検証だけ。本番のモデル・ラベル・パラメータ・買いの規則・画面は変えない）。

運用者の依頼（2026-11-07。実験127 の結果を見て）
  「ラベルは正例率から、先半年で2倍にしましょう。それと5cvのパラメーターチューニングもお願いします。」
  「ファンダメンタルズが思ったより効かなかったのは、目的変数の違うモデルでチューニングしたものを使ったからだと思います。
    直近的には、先半年だと決算を少なくとも2回はまたいでいるので、影響がでないとおかしいです。」
  選んだもの（AskUserQuestion）: 5つへの分け方は「本番と同じ（日付ごとランダム）」（ほかに「期間で5つ・前後半年を外す」
  「前進分割（直近の半年×5本）」）。
  追加の依頼（同日・回す前）「それと、閾値をかなり上げた場合の、Precisionをみたいです。少ない弾数で伸びのよいお宝銘柄を当てるのが
  目的ですので。」→ 参考に「しきい値を上げたときの的中率」を出す（下）。
  追加の指示（同日・結果が出る前）「B：2倍のラベル × +10% 用の木（前回の実験127と同じ組み合わせ） この比較はいらないです。検証は
  いまlgbmだけでお願いします。」→ 比べるのは本番の lgbm（A）と 2倍用に探索した LightGBM（C）の2つだけにする（いったん回した
  run 42373190959 は B を含んでいたので、結果が出る前に止めた）。
  追加の依頼（同日・run 42375605469 の結果を見て）「例えば、上位1%で残りの26サンプルが、2倍には届かなくとも最終的にどこまで上昇
  したのかは気になります。半年で+30%や+50%でも十分なので。」→ 的中率の表に、選んだ銘柄（全体と、2倍に届かなかった銘柄）の行方を
  足す（下の「行方」）。探索はやり直さず、前の回の探索したパラメータを --params-from で読み（列の指紋・ラベル・境界が同じことを
  確かめる）、選ばれた銘柄が前の回と同じかを、上位の割合×木×期間ごとの件数・的中で確かめる（--prev）。

ラベル: 実験127 の「2倍（半年以内）」と同じ（買値は翌営業日の寄り。t+1〜t+125 営業日の終値の最大が買値の2倍以上なら正例。
  売買が止まった銘柄は止まるまでで判定。先が足りない行は使わない）。使う行は +10% と 2倍の両方が分かる新高値の候補。

探索（本番の run_tuning.py と同じ作り。違うのはラベルと、ホールドアウトとのあいだのエンバーゴだけ）
  tuning.tune を本番と同じ設定で呼ぶ: 分割 year_cap_date（年×時価総額帯×その日の正例数で層別し、日付単位で5分割）・50試行・
  目的は分割の平均 PR-AUC・木200本固定・探索範囲も本番のまま・TPE の種 0・373列（features.DEFAULT_PRESET）。
  探索に使うのはホールドアウト（直近12か月）より前の行だけ（本番と同じく train_model.holdout_bounds で切る）。エンバーゴは
  ラベルの窓の125営業日（本番は +10% の窓の20営業日。窓より短いと、ホールドアウトの答えが探索側の行の答えに入る）。直近12か月の
  起点は、2倍の窓が最後まで揃っている最後の日（それより後の行は売買が止まった銘柄だけなので、ホールドアウトに入れない）。
  分け方の注意（運用者が選んだうえで）: 日付単位のランダムな分割では、同じ銘柄の隣り合う日（半年先の答えはほぼ同じ）が学習と
  検証に分かれるので、CV の値は楽観側に出る。当たり具合はホールドアウトの値を参考に並べる。
  探索したパラメータは research/_data/oof/e128_params.json にだけ書く（本番の research/lgbm_params.json には書かない）。

比べる2つ（同じ行・同じ373列・期間×種の15組。学習の仕方は実験127 と同じ）
  A  +10%（本番の lgbm）  本番のラベル・本番の373列のパラメータ（research/lgbm_params.json）
  C  2倍（探索した木）    2倍のラベル・この回で探索したパラメータ（LightGBM）

読み方（回す前に決めたもの。採用の判定ではない。どれも割合を%の小数1桁に丸め、期間×種の15組それぞれで C と A を比べる）
  1. 運用者の見立て「目的変数に合わせて探索すれば決算が効いてくる（半年なら決算を2回はまたぐ）」: 決算・財務＋バリュエーション
     （fund+val）の割合
     - 15組すべてで C > A →「仮説どおり（2倍（探索した木）のほうが決算・財務＋バリュエーションの割合が高い）」
     - 15組すべてで C < A →「仮説と逆（…低い）」
     - それ以外 →「はっきりしない」
  2. もとの仮説「本番のモデルよりテクニカルの重要度が下がる」: テクニカルの割合（実験127 と同じ読み方）
     - 15組すべてで C < A →「仮説どおり」・すべてで C > A →「仮説と逆」・それ以外 →「はっきりしない」
  見るだけ: 値動きの荒さの5列（VOL_COLS。実験127 でテクニカルの増えた分のほぼすべて）の合計、区分・グループ・列ごとの割合、
  探索したパラメータ。

参考（読み方には入れない）
  - 同じ5分割での CV の値（楽観側に出る）: 探索した木を探索と同じ分割で計算し直し、探索の記録と合うかを見る
  - ホールドアウト（直近12か月）: 探索に使った期間の行で学習し（種ごと）、2倍のラベルの PR-AUC（正例率の何倍か）と ROC-AUC を
    A（+10% のラベルで学習した本番の lgbm の順位で2倍を当てる）と C で比べる
  - しきい値を上げたときの的中率（2倍に届いた割合）: A・C のスコア（種の平均）の上位 PREC_TOPS で切り、件数・的中・的中率
    （正例率の何倍か）・伸び（半年以内の終値の最高値 ÷ 買値 − 1）の中央値。数え方は行ごとと、銘柄ごと（しきい値を初めて越えた
    日に1回だけ買うとみなす。的中率に95%の幅（Wilson））の2通り。期間はホールドアウトと、前進分割 2021〜2025年（学習はテストの
    年の始まりの PURGE_DAYS 暦日前より前の行。年ごとに上位を切ってまとめる。C の木は探索に使った期間の行で探索したので、その期間に
    入る年の C は楽観側）。上位の割合はその期間の行のスコアの順位で切るので、そのまま本番のしきい値にはならない
  - 行方（銘柄ごと。選んだ銘柄の全体と、2倍に届かなかった銘柄）: 半年以内の終値の最高値までの上がり幅と、半年後（125営業日目。
    それより前に売買が止まった銘柄は最後の終値）の終値での損益。どちらも買値（翌営業日の寄り）に対する割合で、中央値・+30% 以上・
    +50% 以上の数、半年後の損益は平均とマイナスの数も

確かめ（合わなければ止まる）
  実験127 と同じ（+10% のラベルが本番の定義・買値と20営業日の窓が本番のラベルと合う・373列すべてに区分が付く・本番のパラメータが
  373列で探索したもの）。加えて、探索に渡すラベルが2倍のラベルであること・探索の行がホールドアウトの起点からエンバーゴぶん前で
  終わること・探索が分割を作れたこと・伸びに使う最高値の倍率から作り直した「2倍に届いたか」がラベルとすべての行で合うこと・
  半年後の終値の倍率がすべての行にあり、最高値の倍率を超えないこと。

公開ログには集計だけ（件数・割合・区分名・列名・指標・パラメータ）。銘柄ごとの値は出さない。結果は research/_data/oof/e128_results.json。

    exp=e128_double_half_year_tuned.py  args=""   （既定は --n-trials 50 --seeds 0,1,2）
    前の回の探索を使う: args="--params-from research/_data/oof/e128_params.json --prev research/_data/oof/e128_results.json"
    （Run Experiment は research/_data/oof を同じ台本の前の回から引き継ぐ）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
import lab  # noqa: E402
import label_specs as LS  # noqa: E402
import train_model  # noqa: E402
import tuning  # noqa: E402
import e127_triple_one_year as E  # noqa: E402

#: 2倍（半年以内）。列名・窓の営業日数・倍率（実験127 と同じ）
TARGET = ("x2", 125, 2.0)
N_TRIALS = 50
N_SPLITS = 5
SCHEME = tuning.PRODUCTION_CV
OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
HOLDOUT_MONTHS = train_model.HOLDOUT_MONTHS
#: ホールドアウトと探索の行のあいだに空ける営業日（ラベルの窓の長さ。本番の EMBARGO_DAYS も窓の長さ）
EMBARGO_DAYS = TARGET[1]
SEEDS = E.SEEDS
WINDOW_ENDS = E.WINDOW_ENDS
#: 値動きの荒さを測る5列（実験127 でテクニカルの割合が増えた分のほぼすべて。§89-3）
VOL_COLS = ("tx_vol_parkinson_20", "tx_vol_gk_20", "tx_natr_14", "tx_vol_60", "vol_20d")
OUT = os.path.join(lab.DATA_DIR, "oof", "e128_results.json")
PARAMS_OUT = os.path.join(lab.DATA_DIR, "oof", "e128_params.json")
PARAMS_KEY = "x2_half_year"
#: 比べる2つ（記号・ラベルの列・木・名前）。運用者の指示で、2倍のラベル × +10% 用の木（実験127 の作り）は比べない
ARMS = (("A", "p10", "prod", "+10%（本番の lgbm）"),
        ("C", "x2", "tuned", "2倍（探索した木）"))
ARM_NAME = {a: name for a, _, _, name in ARMS}
#: しきい値を上げたときの的中率（参考）で見る上位の割合
PREC_TOPS = (0.10, 0.05, 0.02, 0.01, 0.005)
#: 行方で数える上がり幅（+30%・+50%）
GAIN_LEVELS = (0.30, 0.50)
#: 前進分割（参考）のテストの年と、学習とテストのあいだの暦日（125営業日×1.5＋5＝193日。実験127 と同じ式）
OOF_YEARS = (2021, 2022, 2023, 2024, 2025)
PURGE_DAYS = int(np.ceil(TARGET[1] * E.PURGE_PER_DAY)) + E.PURGE_PAD
#: 当たり具合を比べる木（記号・学習のラベル・名前）。A は +10% のラベルで学習した本番の lgbm の順位で2倍を当てる
SCORE_ARMS = (("A", "p10", "+10%で学習（本番の lgbm）"), ("C", TARGET[0], "2倍（探索した木）"))
SCORE_NAME = {a: name for a, _, name in SCORE_ARMS}
#: 探索するパラメータ（tuning.tune の探索範囲と同じ並び）と表示名
SEARCHED = (("learning_rate", "学習率"), ("num_leaves", "葉"), ("min_child_samples", "葉の最小件数"),
            ("subsample", "行の抜き取り"), ("colsample_bytree", "列の抜き取り"),
            ("reg_alpha", "L1"), ("reg_lambda", "L2"))


# --------------------------------------------------------------------------- #
# 探索に使う行・ホールドアウト
# --------------------------------------------------------------------------- #
def split_for_tuning(df: pd.DataFrame, key: str = TARGET[0], holdout_months: int = HOLDOUT_MONTHS,
                     embargo_days: int = EMBARGO_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    探索に使う行（ホールドアウトの起点からエンバーゴぶん前まで）・ホールドアウトの行・境界。

    直近 holdout_months か月の終わりは、key の窓が最後まで揃っている最後の日（それより後の行は売買が止まった銘柄だけ）。
    境界は本番と同じ train_model.holdout_bounds で決める（エンバーゴの営業日→暦日の換算も同じ）。
    """
    full = df[f"{key}_have"].fillna(False).astype(bool)
    if not full.any():
        raise SystemExit("ラベルの窓が最後まで揃っている行がない")
    d = pd.to_datetime(df["Date"])
    last_full = pd.Timestamp(d[full].max())
    train_end, test_start, dmax = train_model.holdout_bounds(d[d <= last_full], holdout_months, embargo_days)
    tune_rows = df[d <= train_end]
    hold = df[(d >= test_start) & (d <= dmax)]
    if tune_rows.empty or hold.empty:
        raise SystemExit("探索の行かホールドアウトの行が空")
    if pd.to_datetime(tune_rows["Date"]).max() + pd.Timedelta(
            days=int(round(embargo_days * train_model.TRADING_TO_CALENDAR))) > test_start:
        raise SystemExit("探索の行がホールドアウトの起点に近すぎる（エンバーゴが足りない）")
    bounds = {"train_end": str(train_end.date()), "test_start": str(test_start.date()), "last_full": str(dmax.date()),
              "embargo_days": int(embargo_days), "embargo_calendar_days": int((test_start - train_end).days),
              "embargo_rows": int(((d > train_end) & (d < test_start)).sum()),
              "after_last_full": int((d > dmax).sum())}
    return tune_rows, hold, bounds


def tuning_frame(tune_rows: pd.DataFrame, key: str = TARGET[0]) -> pd.DataFrame:
    """
    探索に渡す行。tuning.tune は label の列を学ぶので、2倍のラベルに差し替える（差し替えないと本番の +10% の探索になる）。
    """
    t = tune_rows.copy()
    t["label"] = pd.to_numeric(t[key], errors="raise").astype(int)
    if not np.array_equal(t["label"].to_numpy(), tune_rows[key].to_numpy(dtype=int)):
        raise SystemExit("探索に渡すラベルが2倍のラベルと合わない")
    if t["label"].nunique() < 2:
        raise SystemExit("探索の行に正例か負例がない")
    return t


def run_tuning(tdf: pd.DataFrame, cols: Sequence[str], n_trials: int) -> Tuple[Dict, Dict]:
    """本番と同じ設定で探索する（分割 year_cap_date・目的 PR-AUC・木の本数と探索範囲は tuning.tune のまま・種 0）。"""
    tuning.LAST_CV = {}
    params = tuning.tune(tdf, list(cols), n_trials=int(n_trials), n_splits=N_SPLITS, scheme=SCHEME,
                         objective=OBJECTIVE, seed=0)
    record = dict(tuning.LAST_CV)
    if not record:
        raise SystemExit("探索の分割を作れなかった（既定のパラメータが返った）")
    if record.get("scheme") != SCHEME or record.get("objective") != OBJECTIVE:
        raise SystemExit(f"探索の設定が本番と違う: {record.get('scheme')} / {record.get('objective')}")
    p = {k: v for k, v in params.items() if not str(k).startswith("_")}
    p["verbose"] = -1
    return p, record


def cv_scores(tdf: pd.DataFrame, cols: Sequence[str], params_by_arm: Dict[str, Dict]) -> Dict:
    """探索と同じ5分割（同じ種）で、木ごとの PR-AUC・ROC-AUC を計算し直す（参考。楽観側に出る）。"""
    folds = tuning.cv_folds(tdf, SCHEME, n_splits=N_SPLITS, seed=0)
    out: Dict = {"base_rate": [float(va["label"].mean()) for _, va in folds], "arms": {}}
    for arm, p in params_by_arm.items():
        prs, rocs = [], []
        for tr, va in folds:
            pr, roc, _ = tuning._fit_one(p, tr, va, list(cols), early_stopping=False)
            prs.append(pr)
            rocs.append(roc)
        out["arms"][arm] = {"pr_auc": float(np.mean(prs)), "fold_pr_auc": [float(x) for x in prs],
                            "roc_auc": float(np.mean(rocs)),
                            "lift": float(np.mean([a / b for a, b in zip(prs, out["base_rate"])]))}
    return out


# --------------------------------------------------------------------------- #
# 当たり具合（参考）
# --------------------------------------------------------------------------- #
def forward_outcomes(bars: pd.DataFrame, rows: pd.DataFrame, horizon: int = TARGET[1]) -> pd.DataFrame:
    """
    rows の各行の、買値（翌営業日の寄り）に対する倍率を2つ（rows と同じ並び）:
      peak_ratio  t+1〜t+horizon 営業日の終値の最高値 ÷ 買値（2倍のラベルと同じ値）
      end_ratio   半年後の終値 ÷ 買値。t+horizon 営業日目の終値（その日に値が無ければ直前の終値）。それより前に売買が止まった
                  銘柄は最後の終値。
                  先が足りず、売買も止まっていない行は NaN
    窓は実験127 の price_columns・forward_frame と同じ（銘柄ごとの行で数える）。日足の重ねは build_frame と同じく後ろの行を残す。
    """
    prices = E.price_columns(bars).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    fr = E.forward_frame(prices, horizon)
    close = prices.groupby("Code", sort=False)["close"]
    # その日に終値が無い（売買が無かった）ときは、その日までの直近の終値（銘柄をまたがない）
    filled = close.ffill()
    end_close = filled.groupby(prices["Code"], sort=False).shift(-horizon).where(
        fr["have_forward"], close.transform("last").where(fr["stopped"]))
    fr["peak_ratio"] = fr["fmax"] / fr["entry"]
    fr["end_ratio"] = end_close / fr["entry"]
    m = rows[["Code", "Date"]].merge(fr[["Code", "Date", "peak_ratio", "end_ratio"]], on=["Code", "Date"], how="left",
                                     validate="one_to_one")
    return m[["peak_ratio", "end_ratio"]].reset_index(drop=True)


def peak_ratio(bars: pd.DataFrame, rows: pd.DataFrame, horizon: int = TARGET[1]) -> np.ndarray:
    """rows の各行の t+1〜t+horizon 営業日の終値の最高値 ÷ 買値（2倍のラベルと同じ窓・同じ値）。"""
    return forward_outcomes(bars, rows, horizon)["peak_ratio"].to_numpy(dtype=float)


def ratio_mismatch(df: pd.DataFrame, key: str = TARGET[0], multiple: float = TARGET[2]) -> float:
    """最高値の倍率から作り直した「届いたか」がラベルと合わない行の割合（倍率の無い行も合わないと数える）。"""
    r = pd.Series(pd.to_numeric(df["peak_ratio"], errors="coerce").to_numpy(dtype=float))
    if r.empty:
        return 0.0
    hit = LS.at_least(r, multiple).fillna(False).astype(int).to_numpy()
    bad = (hit != df[key].to_numpy(dtype=int)) | r.isna().to_numpy()
    return float(bad.mean())


def arm_scores(train_rows: pd.DataFrame, test_rows: pd.DataFrame, cols: Sequence[str],
               arms: Dict[str, Tuple[str, Dict]], seeds: Sequence[int]) -> Dict[str, List[np.ndarray]]:
    """arms = {記号: (学習のラベルの列, 木のパラメータ)}。種ごとに学習し、テストの行のスコアを返す（片方のクラスしか無い木は飛ばす）。"""
    Xtr = train_rows[list(cols)].to_numpy(dtype=float)
    Xte = test_rows[list(cols)].to_numpy(dtype=float)
    out: Dict[str, List[np.ndarray]] = {}
    for arm, (label, p) in arms.items():
        ytr = train_rows[label].to_numpy(dtype=int)
        if not 0 < ytr.sum() < len(ytr):
            continue
        out[arm] = [E.fit(Xtr, ytr, p, int(s)).predict_proba(Xte)[:, 1] for s in seeds]
    return out


def holdout_metrics(hold: pd.DataFrame, scores: Dict[str, List[np.ndarray]], n_train: int,
                    key: str = TARGET[0]) -> Optional[Dict]:
    """ホールドアウトで key のラベルを当てた PR-AUC（種ごと・平均）と ROC-AUC（参考）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = hold[key].to_numpy(dtype=int)
    if not scores or not 0 < y.sum() < len(y):
        return None
    base = float(y.mean())
    d = pd.to_datetime(hold["Date"])
    out: Dict = {"n_train": int(n_train), "n_test": int(len(hold)), "pos_test": int(y.sum()), "base": base,
                 "from": str(d.min().date()), "to": str(d.max().date()), "arms": {}}
    for arm, per_seed in scores.items():
        aps = [float(average_precision_score(y, s)) for s in per_seed]
        rocs = [float(roc_auc_score(y, s)) for s in per_seed]
        out["arms"][arm] = {"pr_auc": float(np.mean(aps)), "pr_auc_by_seed": aps, "lift": float(np.mean(aps)) / base,
                            "roc_auc": float(np.mean(rocs))}
    return out


def top_picks(frame: pd.DataFrame, scores: np.ndarray, q: float) -> pd.DataFrame:
    """スコアの上位 q の行（件数は round(q×行数)、少なくとも1。同点は元の並びの順）。"""
    k = max(1, int(round(float(q) * len(frame))))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k]
    return frame.iloc[order]


def first_cross(picks: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに、しきい値を初めて越えた日の行だけ（その日に1回だけ買うとみなす）。期間（period の列）ごとに数える。"""
    return picks.sort_values("Date", kind="mergesort").drop_duplicates(["period", "Code"], keep="first")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """的中率の95%の幅（Wilson）。件数0は幅なし（NaN）。"""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def outcome_stats(rows: pd.DataFrame) -> Dict:
    """
    選んだ行の行方: 半年以内の終値の最高値までの上がり幅（peak）と、半年後の終値での損益（end）。どちらも買値に対する割合。
    中央値、+30%・+50%（GAIN_LEVELS）以上の数、半年後の損益の平均とマイナスの数。end_ratio の列が無ければ peak だけ。
    """
    n = int(len(rows))
    nan = float("nan")
    peak = rows["peak_ratio"].to_numpy(dtype=float) - 1.0
    end = (rows["end_ratio"].to_numpy(dtype=float) - 1.0) if "end_ratio" in rows.columns else np.full(n, np.nan)
    has_end = n > 0 and not np.isnan(end).all()
    out: Dict = {"n": n,
                 "peak_median": float(np.nanmedian(peak)) if n else nan,
                 "end_median": float(np.nanmedian(end)) if has_end else nan,
                 "end_mean": float(np.nanmean(end)) if has_end else nan,
                 "end_lt0": int(np.sum(end < 0))}
    for lv in GAIN_LEVELS:
        tag = int(round(lv * 100))
        out[f"peak_ge_{tag}"] = int(np.sum(peak >= lv - 1e-12))
        out[f"end_ge_{tag}"] = int(np.sum(end >= lv - 1e-12))
    return out


def pick_stats(picks: pd.DataFrame, key: str = TARGET[0]) -> Dict:
    """選んだ行の件数・的中・的中率（95%の幅）・伸び（最高値の倍率 − 1）の中央値と、行方（全体・2倍に届かなかった行）。"""
    y = picks[key].to_numpy(dtype=int)
    n, k = int(len(y)), int(y.sum())
    lo, hi = wilson(k, n)
    gain = picks["peak_ratio"].to_numpy(dtype=float) - 1.0
    return {"n": n, "hits": k, "precision": (k / n) if n else float("nan"), "lo": lo, "hi": hi,
            "median_gain": float(np.nanmedian(gain)) if n else float("nan"),
            "all": outcome_stats(picks), "miss": outcome_stats(picks[y == 0])}


def precision_table(parts: Sequence[Tuple[pd.DataFrame, np.ndarray]], tops: Sequence[float] = PREC_TOPS,
                    key: str = TARGET[0]) -> List[Dict]:
    """parts = [(期間の行（period・Date・Code・key・peak_ratio の列）, スコア)]。期間ごとに上位 q を切り、まとめて数える。"""
    out = []
    for q in tops:
        picks = pd.concat([top_picks(f, s, q) for f, s in parts], ignore_index=True)
        out.append({"top": float(q), "rows": pick_stats(picks, key), "stocks": pick_stats(first_cross(picks), key)})
    return out


def score_frame(rows: pd.DataFrame, period: str, key: str = TARGET[0]) -> pd.DataFrame:
    keep = ["Date", "Code", key, "peak_ratio"] + (["end_ratio"] if "end_ratio" in rows.columns else [])
    return rows[keep].assign(period=str(period))


def load_params_from(path: str, sig: str, bounds: Dict) -> Tuple[Dict, Dict]:
    """
    前の回の探索したパラメータ（e128_params.json）を読む。列の指紋・ラベル・ホールドアウトの境界が今の回と同じでなければ止める
    （違えば、前の回とは別のデータか別の作りで探索したもの）。
    """
    if not os.path.exists(path):
        raise SystemExit(f"前の回の探索結果がない: {path}（探索をやり直すなら --params-from を付けない）")
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh).get(PARAMS_KEY) or {}
    if rec.get("_features_sig") != sig:
        raise SystemExit(f"前の回の探索結果の列の指紋が違う: {rec.get('_features_sig')} / {sig}")
    want = {"key": TARGET[0], "horizon": TARGET[1], "multiple": TARGET[2]}
    if rec.get("_label") != want:
        raise SystemExit(f"前の回の探索結果のラベルが違う: {rec.get('_label')} / {want}")
    old = rec.get("_bounds") or {}
    for k in ("train_end", "test_start", "last_full"):
        if old.get(k) != bounds.get(k):
            raise SystemExit(f"前の回と境界が違う（{k}: {old.get(k)} / {bounds.get(k)}）。データが変わったので探索をやり直す")
    params = {k: v for k, v in rec.items() if not str(k).startswith("_")}
    params["verbose"] = -1
    return params, dict(rec.get("_cv") or {})


def compare_prev(prev: Optional[Dict], prec: Dict[str, Dict]) -> Tuple[int, int]:
    """前の回の的中の表と、上位の割合×木×期間ごとの件数・的中（行ごと・銘柄ごと）が同じか。(同じ数, 比べた数)。"""
    old = (prev or {}).get("precision") or {}
    same = total = 0
    for part in ("holdout", "oof"):
        for arm, rows in (prec.get(part) or {}).items():
            olds = (old.get(part) or {}).get(arm) or []
            for i, r in enumerate(rows):
                total += 1
                o = olds[i] if i < len(olds) else None
                if o and float(o.get("top", -1)) == float(r["top"]) and all(
                        int(r[kind][f]) == int(o[kind][f]) for kind in ("rows", "stocks") for f in ("n", "hits")):
                    same += 1
    return same, total


def oof_parts(df: pd.DataFrame, cols: Sequence[str], arms: Dict[str, Tuple[str, Dict]], seeds: Sequence[int],
              years: Sequence[int] = OOF_YEARS, purge_days: int = PURGE_DAYS,
              key: str = TARGET[0]) -> Tuple[Dict[str, List], List[Dict]]:
    """前進分割（参考）: テストの年ごとに、年の始まりの purge_days 暦日前より前の行で学習し、その年の行のスコア（種の平均）を出す。"""
    parts: Dict[str, List] = {arm: [] for arm in arms}
    years_info: List[Dict] = []
    for y0 in years:
        start = pd.Timestamp(f"{y0}-01-01")
        te = df[(df["Date"] >= start) & (df["Date"] < pd.Timestamp(f"{y0 + 1}-01-01"))]
        tr = df[df["Date"] < start - pd.Timedelta(days=int(purge_days))]
        if te.empty or tr.empty:
            continue
        sc = arm_scores(tr, te, cols, arms, seeds)
        frame = score_frame(te, str(y0), key)
        for arm, per_seed in sc.items():
            parts[arm].append((frame, np.mean(per_seed, axis=0)))
        years_info.append({"year": int(y0), "n_train": int(len(tr)), "n_test": int(len(te)),
                           "pos_test": int(te[key].sum()), "arms": sorted(sc)})
    return parts, years_info


# --------------------------------------------------------------------------- #
# 重要度
# --------------------------------------------------------------------------- #
def importance_arms(df: pd.DataFrame, cols: Sequence[str], prod: Dict, tuned: Dict, seeds: Sequence[int],
                    ends: Sequence[str], log=print) -> List[Dict]:
    """期間×種ごとに A（+10%・本番の lgbm の木）・C（2倍・探索した木）を学習し、列ごとの割合を返す。"""
    key = TARGET[0]
    # ここでの名前は窓ごとの正例の件数の表示にだけ使う（ラベルの名前）
    runs_prod = E.importance_runs(df, cols, prod, seeds, ends, [("p10", "+10%")], log=log)
    runs_tuned = E.importance_runs(df, cols, tuned, seeds, ends, [(key, "2倍")], log=log)
    tuned_by = {(r["end"], r["seed"]): r for r in runs_tuned}
    runs = []
    for r in runs_prod:
        t = tuned_by.get((r["end"], r["seed"]))
        if t is not None:
            runs.append({"end": r["end"], "seed": r["seed"], "n": r["n"], "A": r["p10"], "C": t[key]})
    return runs


def vol_share(col_share: Dict[str, float]) -> float:
    """値動きの荒さの5列の割合の合計（%）。"""
    return 100.0 * sum(float(col_share.get(c, 0.0)) for c in VOL_COLS)


def fund_verdict(pairs: Sequence[Tuple[float, float]], name: str = "2倍（探索した木）") -> str:
    """pairs = [(A の fund+val %, C の fund+val %)]。小数1桁に丸めて、すべての組で同じ向きのときだけ決める。"""
    if not pairs:
        return "比べられない（組が無い）"
    if all(round(c, 1) > round(a, 1) for a, c in pairs):
        return f"仮説どおり（{name}のほうが決算・財務＋バリュエーションの割合が高い）"
    if all(round(c, 1) < round(a, 1) for a, c in pairs):
        return f"仮説と逆（{name}のほうが決算・財務＋バリュエーションの割合が低い）"
    return "はっきりしない"


def summarize(runs: List[Dict], cmap: Dict[str, str], cols: Sequence[str]) -> Dict:
    groups = F.column_groups()
    gmap = {c: (groups.get(c) or "?") for c in cols}
    rows = []
    for r in runs:
        row = {"end": r["end"], "seed": r["seed"], "n": r["n"]}
        for arm, _, _, _ in ARMS:
            row[arm] = {"gain": E.by_key(r[arm]["gain"], cmap), "split": E.by_key(r[arm]["split"], cmap),
                        "pos": r[arm]["pos"], "vol": vol_share(r[arm]["gain"])}
        rows.append(row)

    def tech(row, arm):
        return 100 * row[arm]["gain"].get("tech", 0.0)

    def fund_val(row, arm):
        return 100 * (row[arm]["gain"].get("fund", 0.0) + row[arm]["gain"].get("val", 0.0))

    pairs = {"fund_val_CA": [(fund_val(r, "A"), fund_val(r, "C")) for r in rows],
             "tech_CA": [(tech(r, "A"), tech(r, "C")) for r in rows],
             "vol_CA": [(r["A"]["vol"], r["C"]["vol"]) for r in rows]}

    def mean_over(level: str, arm: str, kind: str = "gain") -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for r in runs:
            if level == "category":
                d = E.by_key(r[arm][kind], cmap)
            elif level == "group":
                d = E.by_key(r[arm][kind], gmap)
            else:
                d = r[arm][kind]
            for k, v in d.items():
                acc.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    return {"rows": rows, "pairs": pairs,
            "readings": {"fund": fund_verdict(pairs["fund_val_CA"], ARM_NAME["C"]),
                         "tech": E.verdict(pairs["tech_CA"], ARM_NAME["C"])},
            "look": {"vol_CA": E.direction(pairs["vol_CA"], ARM_NAME["C"])},
            "mean": {arm: {"category": mean_over("category", arm), "category_split": mean_over("category", arm, "split"),
                           "group": mean_over("group", arm), "column": mean_over("column", arm),
                           "vol": float(np.mean([r[arm]["vol"] for r in rows])) if rows else 0.0}
                     for arm, _, _, _ in ARMS}}


# --------------------------------------------------------------------------- #
# 表示
# --------------------------------------------------------------------------- #
def fmt_param(name: str, v) -> str:
    if v is None:
        return "—"
    if name in ("num_leaves", "min_child_samples"):
        return f"{int(v)}"
    if name in ("reg_alpha", "reg_lambda"):
        return f"{float(v):.2e}"
    return f"{float(v):.4g}"


def spread(pairs: Sequence[Tuple[float, float]]) -> str:
    diffs = [b - a for a, b in pairs]
    if not diffs:
        return "（組なし）"
    return f"平均 {np.mean(diffs):+.1f}pt（最小 {min(diffs):+.1f}・最大 {max(diffs):+.1f}）"


def report(summary: Dict, cv: Optional[Dict], hold: Optional[Dict], log=print) -> None:
    m = summary["mean"]
    log("\n" + "=" * 110)
    log(f"区分ごとの重要度の割合（gain。期間×種の{len(summary['rows'])}組の平均。括弧は split＝分岐に使われた回数の割合）")
    log("=" * 110)
    log("  " + "区分".ljust(30) + "".join(f"{ARM_NAME[a]:>22}" for a, _, _, _ in ARMS) + f"{'差 C−A':>12}")
    for cat, cname in E.CATEGORIES:
        cells = []
        for a, _, _, _ in ARMS:
            g = 100 * m[a]["category"].get(cat, 0.0)
            sp = 100 * m[a]["category_split"].get(cat, 0.0)
            cells.append(f"{g:>13.1f}%（{sp:4.1f}%）")
        ca = 100 * (m["C"]["category"].get(cat, 0.0) - m["A"]["category"].get(cat, 0.0))
        log("  " + cname.ljust(28) + "".join(cells) + f"{ca:>+10.1f}pt")
    log("  " + "値動きの荒さの5列（テクニカルの内数）".ljust(23)
        + "".join(f"{m[a]['vol']:>21.1f}%" for a, _, _, _ in ARMS)
        + f"{m['C']['vol'] - m['A']['vol']:>+10.1f}pt")

    log("\n組ごと（決算・財務＋バリュエーション＝fund+val と テクニカル の割合 gain %）")
    for i, r in enumerate(summary["rows"]):
        fv_a, fv_c = summary["pairs"]["fund_val_CA"][i]
        te_a, te_c = summary["pairs"]["tech_CA"][i]
        log(f"  〜{r['end']:<11} 種{r['seed']} {r['n']:>7,}行 正例 +10% {r['A']['pos']:,}・2倍 {r['C']['pos']:,}"
            f" | fund+val A {fv_a:.1f}% → C {fv_c:.1f}%（{fv_c - fv_a:+.1f}pt）"
            f" | テクニカル A {te_a:.1f}% → C {te_c:.1f}%（{te_c - te_a:+.1f}pt）"
            f" | 荒さ5列 A {r['A']['vol']:.1f}% → C {r['C']['vol']:.1f}%")
    p = summary["pairs"]
    log(f"  fund+val の差 C−A: {spread(p['fund_val_CA'])} / テクニカルの差 C−A: {spread(p['tech_CA'])}")
    log(f"  見るだけ: 荒さ5列の差 C−A {spread(p['vol_CA'])} → {summary['look']['vol_CA']}")
    log("\n読み方（回す前に決めたもの。C と A）:")
    log(f"  1. 2倍用に探索した木では本番の lgbm より決算が効いてくるか（fund+val）: {summary['readings']['fund']}")
    log(f"  2. 本番の lgbm よりテクニカルの重要度が下がるか（テクニカル）: {summary['readings']['tech']}")

    for a, _, _, name in ARMS:
        top = sorted(m[a]["group"].items(), key=lambda kv: -kv[1])[:12]
        log(f"\nグループごとの割合（gain・平均）の上位12: {name}")
        log("  " + " / ".join(f"{g} {100 * v:.1f}%" for g, v in top))
    for a, _, _, name in ARMS:
        top = sorted(m[a]["column"].items(), key=lambda kv: -kv[1])[:15]
        log(f"\n列ごとの割合（gain・平均）の上位15: {name}")
        log("  " + " / ".join(f"{c} {100 * v:.1f}%" for c, v in top))

    if cv:
        log("\n参考: 探索と同じ5分割での2倍のラベルの当たり具合（日付単位のランダムな分割なので楽観側に出る）")
        rates = " ".join(f"{100 * b:.2f}%" for b in cv["base_rate"])
        log(f"  検証の正例率: {rates}")
        for arm in ("C",):
            c = cv["arms"].get(arm)
            if c:
                folds = " ".join(f"{x:.4f}" for x in c["fold_pr_auc"])
                log(f"  {ARM_NAME[arm]}: PR-AUC {c['pr_auc']:.4f}（正例率の {c['lift']:.2f}倍）/ ROC-AUC {c['roc_auc']:.4f}"
                    f" / 分割ごと {folds}")
    if hold:
        log(f"\n参考: ホールドアウト（{hold['from']}〜{hold['to']}・{hold['n_test']:,}行・2倍の正例 {hold['pos_test']:,}"
            f"（{100 * hold['base']:.2f}%）。学習は探索に使った {hold['n_train']:,}行）")
        for arm, _, name in SCORE_ARMS:
            c = hold["arms"].get(arm)
            if c:
                seeds = " ".join(f"{x:.4f}" for x in c["pr_auc_by_seed"])
                log(f"  {arm} {name}: PR-AUC {c['pr_auc']:.4f}（正例率の {c['lift']:.2f}倍）/ ROC-AUC {c['roc_auc']:.4f}"
                    f" / 種ごと {seeds}")


def pct(v: float, digits: int = 1) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:.{digits}f}%"


def precision_lines(tables: Dict[str, List[Dict]], base: float, log=print) -> None:
    """上位の割合ごとに A・C を並べる（行ごと | 銘柄ごと）。"""
    first = next(iter(tables.values()), None)
    if not first:
        return
    for i, row in enumerate(first):
        head = f"上位 {100 * row['top']:g}%"
        for arm, _, name in SCORE_ARMS:
            t = tables.get(arm)
            if not t:
                continue
            r, s_ = t[i]["rows"], t[i]["stocks"]
            lift = (r["precision"] / base) if base > 0 and r["n"] else float("nan")
            log(f"    {head:<8} {arm} {name:<14} 行 {r['n']:>5,}件・的中 {r['hits']:>4,}（{pct(r['precision'])}・正例率の {lift:.1f}倍）"
                f"・伸び {100 * r['median_gain']:+.0f}% | 銘柄 {s_['n']:>4,}・的中 {s_['hits']:>3,}（{pct(s_['precision'])}"
                f"、95%の幅 {pct(s_['lo'])}〜{pct(s_['hi'])}）・伸び {100 * s_['median_gain']:+.0f}%")
            head = ""


def fate_text(st: Dict) -> str:
    def p(v):
        if v is None or not np.isfinite(v):
            return "—"
        return "±0%" if abs(100 * v) < 0.5 else f"{100 * v:+.0f}%"
    return (f"最高値まで 中央値 {p(st['peak_median'])}・+30%以上 {st['peak_ge_30']}・+50%以上 {st['peak_ge_50']}"
            f" | 半年後の終値 中央値 {p(st['end_median'])}・平均 {p(st['end_mean'])}・+30%以上 {st['end_ge_30']}"
            f"・+50%以上 {st['end_ge_50']}・マイナス {st['end_lt0']}")


def fate_lines(tables: Dict[str, List[Dict]], log=print) -> None:
    """銘柄ごとの行方（選んだ銘柄の全体と、2倍に届かなかった銘柄）。"""
    first = next(iter(tables.values()), None)
    if not first or "all" not in first[0]["stocks"]:
        return
    log("    行方（銘柄ごと。最高値まで＝半年以内の終値の最高値の上がり幅、半年後の終値＝125営業日目の終値での損益。どちらも買値に対して）")
    for i, row in enumerate(first):
        head = f"上位 {100 * row['top']:g}%"
        for arm, _, name in SCORE_ARMS:
            t = tables.get(arm)
            if not t:
                continue
            st = t[i]["stocks"]
            log(f"    {head:<8} {arm} {name:<14} 全体 {st['all']['n']:>4,}銘柄: {fate_text(st['all'])}")
            log(f"    {'':<8} {'':<1} {'':<14} 外れ {st['miss']['n']:>4,}銘柄: {fate_text(st['miss'])}")
            head = ""


def report_precision(prec_hold: Dict[str, List[Dict]], hold: Optional[Dict], prec_oof: Dict[str, List[Dict]],
                     oof_years: List[Dict], bounds: Dict, log=print) -> None:
    log("\n参考: しきい値を上げたときの的中率（2倍に届いた割合）。上位の割合は、その期間の行のスコア（種の平均）の順位で切る")
    log("  伸び＝半年以内の終値の最高値 ÷ 買値 − 1 の中央値。銘柄ごと＝しきい値を初めて越えた日に1回だけ買うとみなした数え方"
        "（的中率の95%の幅は Wilson）")
    if hold and prec_hold:
        log(f"  [ホールドアウト {hold['from']}〜{hold['to']}・{hold['n_test']:,}行・2倍の正例 {hold['pos_test']:,}"
            f"（{100 * hold['base']:.2f}%）。学習は探索に使った {hold['n_train']:,}行]")
        precision_lines(prec_hold, hold["base"], log)
        fate_lines(prec_hold, log)
    if prec_oof and oof_years:
        n = sum(y["n_test"] for y in oof_years)
        pos = sum(y["pos_test"] for y in oof_years)
        base = pos / n if n else float("nan")
        log(f"  [前進分割 {oof_years[0]['year']}〜{oof_years[-1]['year']}年・{n:,}行・2倍の正例 {pos:,}（{100 * base:.2f}%）。"
            f"学習はテストの年の始まりの{PURGE_DAYS}暦日前まで・年ごとに上位を切ってまとめる。C の木は 〜{bounds['train_end']} の行で"
            "探索したので、その期間に入る年の C は楽観側]")
        log("    年ごと（学習・テスト・2倍の正例）: " + " / ".join(
            f"{y['year']}: {y['n_train']:,}・{y['n_test']:,}・{y['pos_test']:,}" for y in oof_years))
        precision_lines(prec_oof, base, log)
        fate_lines(prec_oof, log)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None, log=print) -> int:
    ap = argparse.ArgumentParser(description="実験128: 半年以内に2倍のラベルで5分割の探索をし直し、重要度の割合を測り直す（検証だけ）")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--ends", default=",".join(WINDOW_ENDS))
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--params-out", default=PARAMS_OUT)
    ap.add_argument("--params-from", default="",
                    help="探索をやり直さず、前の回の e128_params.json からパラメータを読む（列の指紋・ラベル・境界が同じことを確かめる）")
    ap.add_argument("--prev", default="", help="前の回の e128_results.json。的中の件数が同じか（同じ銘柄が選ばれたか）を確かめる")
    args = ap.parse_args(argv)
    t0 = time.time()
    key, horizon, multiple = TARGET
    xname = E.target_name(horizon, multiple)

    cols = F.columns(F.DEFAULT_PRESET)
    cmap = E.category_map(cols)
    prod = E.production_params(cols)
    sig = F.signature(list(cols))
    log(f"[setup] 列 {F.DEFAULT_PRESET}（{len(cols)}列・指紋 {sig}）/ 本番の木 {prod.get('n_estimators')}本"
        f" / 学習率 {prod.get('learning_rate'):.4g} / 葉 {prod.get('num_leaves')}（本番の lgbm_params.json）")
    counts: Dict[str, int] = {}
    for c in cols:
        counts[cmap[c]] = counts.get(cmap[c], 0) + 1
    log("[setup] 区分ごとの列の数: " + " / ".join(f"{name} {counts.get(cat, 0)}" for cat, name in E.CATEGORIES))
    log(f"[setup] ラベル: +10%（本番） / {xname}。比べる木: " + " / ".join(f"{a} {name}" for a, _, _, name in ARMS))

    prev = None
    if args.prev:
        if os.path.exists(args.prev):
            with open(args.prev, encoding="utf-8") as fh:
                prev = json.load(fh)
        else:
            log(f"[warn] 前の回の結果がない（{os.path.basename(args.prev)}）。同じ銘柄が選ばれたかは確かめられない")

    ds, bars = E.load_inputs(args.data_dir)
    df, info = E.build_frame(ds, bars, [TARGET], cols, log=log)
    outcomes = forward_outcomes(bars, df, horizon)
    df["peak_ratio"] = outcomes["peak_ratio"].to_numpy(dtype=float)
    df["end_ratio"] = outcomes["end_ratio"].to_numpy(dtype=float)
    del bars, ds, outcomes
    bad = ratio_mismatch(df, key, multiple)
    log(f"[check] 半年以内の終値の最高値 ÷ 買値 から作り直した「2倍に届いたか」がラベルと合わない行 {100 * bad:.3f}%（許容 0）")
    if bad > 0:
        raise SystemExit("伸びに使う最高値の倍率が2倍のラベルと合わない")
    no_end = int(df["end_ratio"].isna().sum())
    over = int((df["end_ratio"] > df["peak_ratio"] + 1e-12).sum())
    log(f"[check] 半年後の終値の倍率が無い行 {no_end:,} / 最高値の倍率を超える行 {over:,}（どちらも許容 0）")
    if no_end or over:
        raise SystemExit("半年後の終値の倍率が無いか、最高値の倍率を超える行がある")
    t = info["targets"][key]
    log(f"[label] 候補 {info['candidates']:,}行のうち、+10% と {xname} の両方が分かる行 {info['known']:,}"
        f"（{info['first']}〜{info['last']}）/ 先が足りない行 {t['unknown_recent']:,}・ほかの理由で分からない行 "
        f"{t['unknown_other']:,} / 使う行のうち、売買が止まって窓の途中で分かった行 {t['stopped_known']:,}（うち正例 {t['stopped_pos']:,}）")
    n = max(len(df), 1)
    log(f"[label] 正例（使う行）: +10%（本番） {int(df['p10'].sum()):,}（{100 * df['p10'].sum() / n:.2f}%） / {xname} "
        f"{int(df[key].sum()):,}（{100 * df[key].sum() / n:.2f}%）/ {xname} の正例のうち +10% も正例 "
        f"{int((df[key].astype(bool) & df['p10'].astype(bool)).sum()):,}")
    log("[label] 年ごと（行・正例 +10%・2倍）: "
        + " / ".join(f"{y}: {v['n']:,}・{v['p10']:,}・{v[key]:,}" for y, v in info["years"].items()))

    tune_rows, hold, bounds = split_for_tuning(df, key)
    tdf = tuning_frame(tune_rows, key)
    log(f"\n[tune] 探索に使う行: 〜{bounds['train_end']}（{len(tdf):,}行・{xname} の正例 {int(tdf['label'].sum()):,}"
        f"（{100 * tdf['label'].mean():.2f}%）。+10% の正例率 {100 * tdf['p10'].mean():.2f}% とは別のラベル）")
    log(f"[tune] ホールドアウト: {bounds['test_start']}〜{bounds['last_full']}（{len(hold):,}行）/ あいだ {bounds['embargo_days']}営業日"
        f"（{bounds['embargo_calendar_days']}暦日・{bounds['embargo_rows']:,}行は使わない）/ 窓の揃った最後の日より後の行"
        f" {bounds['after_last_full']:,}（売買が止まった銘柄だけ。ホールドアウトに入れない）")
    t1 = time.time()
    if args.params_from:
        tuned, record = load_params_from(args.params_from, sig, bounds)
        log(f"[tune] 探索はやり直さず、前の回の探索したパラメータを使う（{os.path.basename(args.params_from)}。列の指紋・ラベル・"
            f"境界が同じ。前の回の探索: 分割 {record.get('scheme')}・{record.get('n_splits')}分割・{record.get('n_trials')}試行）")
    else:
        log(f"[tune] 本番と同じ設定: 分割 {SCHEME}・{N_SPLITS}分割・{args.n_trials}試行・目的 {tuning.OBJECTIVE_JA[OBJECTIVE]}")
        tuned, record = run_tuning(tdf, cols, args.n_trials)
    log(f"[tune] {(time.time() - t1) / 60:.1f}分 / 探索した木: "
        + " / ".join(f"{ja} {fmt_param(k, prod.get(k))} → {fmt_param(k, tuned.get(k))}" for k, ja in SEARCHED)
        + f" / 木 {tuned.get('n_estimators')}本（左が本番の +10% の木）")
    if record.get("n_splits") != N_SPLITS:
        log(f"[warn] 分割が {record.get('n_splits')} 本になった（正例か負例が無い分割は飛ばされる）")
    cv = cv_scores(tdf, cols, {"C": tuned})
    log(f"[tune] 探索の記録の最良の PR-AUC {float(record.get('mean_pr_auc', float('nan'))):.4f}"
        f"・同じ分割で計算し直した値 {cv['arms']['C']['pr_auc']:.4f}")

    os.makedirs(os.path.dirname(args.params_out), exist_ok=True)
    with open(args.params_out, "w", encoding="utf-8") as fh:
        json.dump({PARAMS_KEY: {**tuned, "_cv": record, "_n_features": len(cols), "_features_sig": sig,
                                "_preset": F.DEFAULT_PRESET, "_label": {"key": key, "horizon": horizon,
                                                                        "multiple": multiple},
                                "_bounds": bounds}}, fh, ensure_ascii=False, indent=1)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ends = [e.strip() for e in args.ends.split(",") if e.strip()]
    log(f"\n[fit] 期間 {len(ends)}つ × 種 {len(seeds)}つ × 木2つ（A・C）")
    runs = importance_arms(df, cols, prod, tuned, seeds, ends, log=log)
    summary = summarize(runs, cmap, cols)
    arms_fit = {"A": ("p10", prod), "C": (key, tuned)}
    log(f"\n[score] ホールドアウトと前進分割（{OOF_YEARS[0]}〜{OOF_YEARS[-1]}年）で A・C を学習（種 {len(seeds)}つ）")
    hs = arm_scores(tune_rows, hold, cols, arms_fit, seeds)
    hold_res = holdout_metrics(hold, hs, len(tune_rows), key)
    hframe = score_frame(hold, "hold", key)
    prec_hold = {arm: precision_table([(hframe, np.mean(per_seed, axis=0))]) for arm, per_seed in hs.items()}
    oparts, oyears = oof_parts(df, cols, arms_fit, seeds)
    prec_oof = {arm: precision_table(parts) for arm, parts in oparts.items() if parts}
    if args.prev:
        same, total = compare_prev(prev, {"holdout": prec_hold, "oof": prec_oof})
        log(f"[check] 前の回と同じ銘柄が選ばれたか（上位の割合×木×期間ごとの件数・的中。行ごと・銘柄ごと）: 一致 {same} / {total}"
            + ("" if total and same == total else "（一致しないので、前の回の表とは別の選ばれ方。下の行方は今回の選び方のもの）"))
    report(summary, cv, hold_res, log=log)
    report_precision(prec_hold, hold_res, prec_oof, oyears, bounds, log=log)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"info": info, "bounds": bounds, "arms": [list(a) for a in ARMS], "tuned": tuned,
                   "prod": prod, "tuning_record": record, "cv": cv, "holdout": hold_res,
                   "precision": {"tops": list(PREC_TOPS), "holdout": prec_hold, "oof": prec_oof, "oof_years": oyears,
                                 "purge_days": PURGE_DAYS},
                   "readings": summary["readings"], "look": summary["look"], "pairs": summary["pairs"],
                   "mean": summary["mean"], "seeds": seeds, "ends": ends}, fh, ensure_ascii=False, indent=1)
    log(f"\n[done] {args.out} / {args.params_out} / {(time.time() - t0) / 60:.1f}分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
