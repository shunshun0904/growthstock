#!/usr/bin/env python3
"""
LightGBM のハイパーパラメータを Optuna で探索する。

## 最重要: ホールドアウトを見ないこと

探索は「良さそうなパラメータを選ぶ」作業なので、
評価に使う期間のデータを一度でも見ればリークになる。
探索に使ってよいのは **ホールドアウト（直近1年）より前** だけ。
境界は train_model.holdout_bounds から機械的に取る（run_tuning.py）。

    |--- 探索に使ってよい ---|--エンバーゴ--|--- ホールドアウト ---|
                            ↑
                     ここより後は一切見ない

## 分割は「年の束 × ラベル」で層別する

時系列分割はこの規模では推定が安定しなかった（実測で分割ごとの
PR-AUC が 0.036〜0.230 と6倍以上ばらついた）。年で層別すれば
局面の当たり外れがフォールド間で相殺される。

層にラベルも入れる理由: PR-AUC の下限は正例率そのものなので、
フォールド間で正例率がずれると、スコアの差が実力の差なのか
正例率の差なのか分からなくなる。年だけで層別すると、正例率が
1割程度の問題ではフォールドごとの正例数が実際に偏る。
揃っていることは実行のたびに記録する（LAST_CV の fold_pos_rate）。

引き換えに、訓練と検証が同じ期間を含むので、この CV スコア自体は
将来性能の推定にはならない（楽観側に出る）。パラメータを選ぶためだけに
使い、実力の判定はホールドアウトで行う。

## なぜフォールドごとに探索しないか

フォールドごとに探索するのが理想だが、学習回数が現実的な時間に収まらない。
代わりに「プリセットごとに1回、ホールドアウトより前のデータだけで探索」する。
全フォールドで同じパラメータを使うので比較の条件も揃う。

探索結果は research/lgbm_params.json に保存する。毎回探索し直さない。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 探索結果はリポジトリに残す。
# research/_data/ は .gitignore されているので、そこに置くと
# ワークフローのコンテナが終わった時点で消え、毎回探索し直しになる。
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lgbm_params.json")

#: 探索中の木の本数。固定する。
#:
#: early stopping で本数を決めると、検証窓のばらつきがそのまま本数に乗り、
#: 試行ごとに「別の大きさのモデル」を比べることになる。
#: 本数を固定すれば、比べているのは残りのパラメータの違いだけになる。
#:
#: 学習にもこの本数がそのまま使われる（fit_models は探索済みの
#: n_estimators を読む）。本数を変えたら、以前の探索結果と混ぜて
#: 比べないこと。木の数が違うモデルの比較になる。
SEARCH_N_ESTIMATORS = 200

# 探索しないもの（固定）。再現性のため。
FIXED = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": SEARCH_N_ESTIMATORS,
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


def _year_groups(years: pd.Series, labels: pd.Series, n_splits: int) -> pd.Series:
    """
    年を、正例・負例とも n_splits 件以上になるように束ねる。

    StratifiedKFold は、どの層も分割数以上の件数を要求する。
    サンプルの少ない年（初期は決算4期分の履歴が要るぶん少ない）を
    そのまま層にすると落ちるので、隣の年に寄せる。
    """
    order = sorted(years.unique())
    label = labels.astype(int)
    groups: Dict[int, str] = {}
    cur: List[int] = []
    for y in order:
        cur.append(y)
        m = years.isin(cur)
        if int(label[m].sum()) >= n_splits and int((1 - label[m]).sum()) >= n_splits:
            name = f"{cur[0]}" if len(cur) == 1 else f"{cur[0]}-{cur[-1]}"
            for yy in cur:
                groups[yy] = name
            cur = []
    if cur:
        # 末尾が足りなければ直前の束に混ぜる
        last = groups[order[len(groups) - 1]] if groups else f"{cur[0]}-{cur[-1]}"
        for yy in cur:
            groups[yy] = last
    return years.map(groups)


#: 層別に使う時価総額の帯の数。全期間を通した分位で切る。
#:
#: 日付ごとの分位にしてはいけない。どの日も各帯が均等になるので、
#: 層別しているつもりで何も揃えていないことになる。
CAP_BANDS = 5
CAP_COL = "log_market_cap"


def _cap_bands(df: pd.DataFrame, n_bands: int = CAP_BANDS) -> pd.Series:
    """時価総額（対数）を全期間の分位で帯に分ける。欠測は独立した帯にする。"""
    if CAP_COL not in df.columns:
        return pd.Series("na", index=df.index)
    v = pd.to_numeric(df[CAP_COL], errors="coerce")
    band = pd.qcut(v, n_bands, labels=False, duplicates="drop")
    return band.astype("Int64").astype(str).fillna("na")


def _composition_spread(frames: List[pd.DataFrame], key: pd.Series) -> float:
    """
    フォールド間で key の構成比が最大どれだけ違うか（0〜1）。

    key は全行に対して与える（フォールド内で切り直さないこと）。
    フォールド内で分位を取ると定義上どのフォールドも均等になり、
    層別しているつもりで何も測っていないことになる。
    """
    if not frames:
        return float("nan")
    m = pd.DataFrame([key.loc[f.index].value_counts(normalize=True)
                      for f in frames]).fillna(0.0)
    return float((m.max() - m.min()).max())


def _coarsen(levels: List[pd.Series], n_splits: int) -> pd.Series:
    """
    細かい層から順に使い、分割数に満たない層だけを1段粗い層に落とす。

    層を細かくするほど各フォールドの構成は揃うが、StratifiedKFold は
    件数が分割数未満の層で落ちる。全体を粗くすると、細かく取れる部分まで
    損をする。足りない層だけを落とせば両方を取れる。

    levels は粗い順に渡す（最後が最も細かい）。
    先頭は必ず全行を賄える粗さにすること（実際にはラベルだけの層）。
    """
    out = levels[0].astype(str)
    for lv in levels[1:]:
        cand = lv.astype(str)
        big = cand.groupby(cand).transform("size") >= n_splits
        out = cand.where(big, out)

    # 細かい層に抜けていった結果、残った粗い層のほうが分割数を割ることがある。
    # 例: 粗い層に100件あり、96件が細かい層へ移ると粗い層は4件になる。
    # StratifiedKFold はそこで落ちる（警告を出して分割が偏る）。
    # 割ってしまった層は最大の層に寄せる。
    for _ in range(len(levels) + 2):
        size = out.groupby(out).transform("size")
        if bool((size >= n_splits).all()):
            break
        biggest = out.value_counts().index[0]
        out = out.where(size >= n_splits, biggest)
    return out


#: 日付単位で分けるときの、1日あたり正例数のバケット境界。
#: 平均7.2銘柄/日なので正例は0〜3件が大半。細かく割ると層が薄くなる。
POS_BUCKETS = (0, 1, 2, 3)


def _date_strata(df: pd.DataFrame, n_splits: int, by_year: bool,
                 by_cap: bool) -> pd.Series:
    """
    日付ごとの層を作る（索引は日付）。

    日付単位で分けるなら、層も日付単位で持たなければならない。
    ラベルや時価総額は日付の中で変わるので、行単位の層をそのまま
    StratifiedGroupKFold に渡しても満たせず、結果として何も揃わない
    （実測で年構成のずれが 0.062pt -> 8.820pt に悪化した）。

    日付レベルで意味を持つのは
      ・その日の正例数（正例率を揃えるため）
      ・その日が属する年（局面を揃えるため）
      ・その日の規模の中心（規模構成を揃えるため）

    粗いほうから順に積んで _coarsen に渡す。一気に細かくすると、
    薄い層が「正例数だけ」まで落ちて年の情報ごと失われる
    （実測で年構成のずれが 2.725pt に留まっていた原因）。
    """
    g = df.groupby("Date", sort=True)
    npos = g["label"].sum().astype(int)

    # 積む順が結果を決める。薄い層は1段粗いほうへ落ちるので、
    # 最後まで残したい軸を先に置く。年は局面の当たり外れを相殺する
    # いちばん効く軸なので先頭に置く（正例数を先頭にすると、
    # 薄い層が年の情報ごと落ちて年構成のずれが3倍になった）。
    levels: List[pd.Series] = []
    if by_year:
        row_years = pd.to_datetime(df["Date"]).dt.year
        mapping = dict(zip(row_years,
                           _year_groups(row_years, df["label"].astype(int),
                                        n_splits)))
        years = pd.to_datetime(pd.Series(g["Date"].first())).dt.year
        levels.append("y" + years.map(mapping).astype(str))
    # 1日あたりの正例数。多すぎる日はまとめる
    pos = "p" + np.minimum(npos, POS_BUCKETS[-1]).astype(str)
    levels.append((levels[-1] + "_" + pos) if levels else pos)
    if by_cap:
        band = _cap_bands(df)
        mode = band.groupby(df["Date"], sort=True).agg(
            lambda t: t.value_counts().index[0]).astype(str)
        levels.append(levels[-1] + "_c" + mode)
    return _coarsen(levels, n_splits)


def year_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0,
               by_cap: bool = False, by_year: bool = True,
               group_by_date: bool = False
               ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    層別した k 分割。各フォールドが同じ構成になるようにする。

    層に入れる軸:
      ラベル      必ず入れる。PR-AUC の下限は正例率そのものなので、
                  分割間で正例率がずれるとスコアの差が実力の差なのか
                  正例率の差なのか分からなくなる
      年（束）    by_year=True。時系列分割はこの規模では推定が安定しなかった
                  （実測で分割ごとの PR-AUC が 0.036〜0.230 と6倍以上ばらついた）。
                  同じ年構成にすれば局面の当たり外れが相殺される
      時価総額帯  by_cap=True。正例率が規模で 1.8倍違う（〜100億 10.36% /
                  3000億〜 5.74%）ので、規模構成がフォールドで違うと
                  難易度も違ってくる

    年を時価総額帯で「置き換え」ないこと。置き換えると局面の当たり外れが
    相殺されなくなり、時系列分割で起きたばらつきが戻る。足すのが正しい。

    group_by_date=True にすると、同じ日のサンプルを必ず同じフォールドに入れる。
    行単位で切ると、ある日の7銘柄が訓練側と検証側に分かれる。
    「同じ日の銘柄を並べ替える」ことを学習する LTR ではその日の答えの一部を
    訓練で見ることになり、成立しない。pointwise でも同じ日の銘柄は
    地合いを共有するので、分けると検証が楽になり CV が楽観側に出る。
    引き換えに、日付をまたげないぶん層の揃い方は緩くなる（近似になる）。

    層を細かくするほど構成は揃うが、StratifiedKFold は件数が分割数未満の
    層で落ちる。足りない層だけ1段粗い層に落とす（_coarsen）。

    引き換えに、訓練と検証が同じ期間を含むので、この CV スコア自体は
    将来性能の推定にはならない（楽観側に出る）。
    パラメータを選ぶためだけに使い、実力の判定はホールドアウトで行う。
    """
    from sklearn.model_selection import StratifiedKFold

    label = df["label"].astype(int)
    # 粗い順に積む。先頭は必ず全行を賄えるもの（ラベルだけ）
    levels: List[pd.Series] = [label.astype(str)]
    if by_year:
        years = pd.to_datetime(df["Date"]).dt.year
        yg = _year_groups(years, label, n_splits).astype(str)
        levels.append(yg + "_" + label.astype(str))
    if by_cap:
        cap = _cap_bands(df).astype(str)
        prev = levels[-1]
        levels.append(prev + "_cap" + cap)
    strata = _coarsen(levels, n_splits)

    if group_by_date:
        # 日付を1件とみなして層別し、日付ごとまとめてフォールドに入れる。
        # 層も日付単位で作る（行単位の層を渡しても満たせない）
        ds = _date_strata(df, n_splits, by_year=by_year, by_cap=by_cap)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        dates = ds.index.to_numpy()
        by_date = {d: i for i, d in enumerate(dates)}
        pos = df["Date"].map(by_date).to_numpy()
        parts = ((np.flatnonzero(np.isin(pos, tr)),
                  np.flatnonzero(np.isin(pos, va)))
                 for tr, va in skf.split(dates, ds.to_numpy()))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        parts = splitter.split(df, strata)

    out: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for tr_idx, va_idx in parts:
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        if tr["label"].nunique() < 2 or va["label"].nunique() < 2:
            continue
        out.append((tr, va))
    return out


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


def _fit_ranker(params: Dict, tr: pd.DataFrame, va: pd.DataFrame,
                cols: List[str]) -> Tuple[float, float, int]:
    """
    LGBMRanker（lambdarank）を日付グループで学習し、検証窓で測る。

    目的関数は pointwise と同じ PR-AUC にする。lambdarank が最適化する
    のは日付内の順位だが、探索の指標まで変えると
    「順位を直接最適化したから良くなった」のか
    「別の指標で選んだから良くなった」のか分からなくなる。
    日付内での実力は評価側で別に測る（train_model.within_date_auc）。
    """
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if k != "objective"}
    model = lgb.LGBMRanker(objective="lambdarank", **p)
    d = tr.sort_values("Date", kind="stable")
    sizes = d.groupby("Date", sort=True).size().to_numpy()
    model.fit(d[cols].to_numpy(dtype=float),
              d["label"].to_numpy(dtype=int), group=sizes)
    yva = va["label"].to_numpy(dtype=int)
    sc = model.predict(va[cols].to_numpy(dtype=float))
    return (float(average_precision_score(yva, sc)),
            float(roc_auc_score(yva, sc)), int(params["n_estimators"]))


def _fit_one(params: Dict, tr: pd.DataFrame, va: pd.DataFrame,
             cols: List[str], early_stopping: bool = True
             ) -> Tuple[float, float, int]:
    """
    内側検証の PR-AUC・ROC-AUC と、early stopping が決めた木の本数を返す。

    探索の目的関数は PR-AUC。正例率が7%台なので、ROC-AUC だと
    負例側の並びの差が支配的になり、上位の精度が上がらなくても数字が動く。
    ROC-AUC は「見るため」に併記する（目的関数にはしない）。
    """
    import lightgbm as lgb

    Xtr, ytr = tr[cols].to_numpy(dtype=float), tr["label"].to_numpy(dtype=int)
    Xva, yva = va[cols].to_numpy(dtype=float), va["label"].to_numpy(dtype=int)
    model = lgb.LGBMClassifier(**params,
                               scale_pos_weight=scale_pos_weight(ytr))
    if early_stopping:
        # eval_set は 4.7 で非推奨。eval_X / eval_y を使う
        model.fit(
            Xtr, ytr,
            eval_X=Xva, eval_y=yva,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
    else:
        # 本数を固定して学習する。検証データは評価にだけ使う
        model.fit(Xtr, ytr)
    best_iter = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
    prob = model.predict_proba(Xva)[:, 1]
    pr = float(average_precision_score(yva, prob))
    roc = float(roc_auc_score(yva, prob))
    return pr, roc, best_iter


def tune(df: pd.DataFrame, cols: List[str], *, n_trials: int = 30,
         seed: int = 0, verbose: bool = True, n_splits: int = 5,
         embargo_days: int = 60, scheme: str = "year",
         model: str = "classifier") -> Dict:
    """
    Optuna で探索する。df は「テスト窓より前」のデータだけを渡すこと。

    評価は時系列 k 分割の平均 PR-AUC。
    1つの分割で決めていたときは、その期間の癖を拾う恐れがあった。

    model="ranker" にすると LGBMRanker（lambdarank）を日付グループで学習する。
    そのときは分割も日付単位でなければならない（同じ日が両側にあると、
    その日の答えの一部を訓練で見ることになる）。

    返すのは LGBMClassifier にそのまま渡せる辞書。
    探索の記録は tune_cv() で別に取る（返り値に混ぜると、
    そのまま LGBMClassifier に渡したときに未知の引数で落ちる）。
    n_estimators は各分割の early stopping が決めた本数の中央値。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if model == "ranker" and scheme != "year_cap_date":
        raise SystemExit(
            "LTR は日付単位の分割が必須です（--cv year_cap_date）。"
            f"指定: {scheme}。同じ日が訓練と検証に分かれると、"
            "その日の答えの一部を訓練で見ることになります")
    if scheme in ("year", "year_cap", "cap", "year_cap_date"):
        folds = year_folds(df, n_splits=n_splits, seed=seed,
                           by_year=scheme in ("year", "year_cap",
                                              "year_cap_date"),
                           by_cap=scheme in ("year_cap", "cap",
                                             "year_cap_date"),
                           group_by_date=scheme == "year_cap_date")
    elif scheme == "timeseries":
        folds = time_series_folds(df, n_splits=n_splits,
                                  embargo_days=embargo_days)
    else:
        raise SystemExit(
            f"未知の分割方式: {scheme}"
            "（year / year_cap / year_cap_date / cap / timeseries）")
    if not folds:
        if verbose:
            print("  [tune] 分割を作れないため既定値を使う")
        return dict(DEFAULT_PARAMS)
    if verbose:
        ja = {"year": "年で層別", "year_cap": "年×時価総額帯で層別",
              "year_cap_date": "年×時価総額帯で層別・日付単位で分割",
              "cap": "時価総額帯で層別", "timeseries": "時系列"}[scheme]
        print(f"  [tune] {ja}{len(folds)}分割 / 木{SEARCH_N_ESTIMATORS}本固定")
        print("  [tune] " + " / ".join(
            f"訓練{len(t):,}→検証{len(v):,}(正例{int(v['label'].sum())})"
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
        prs, rocs, iters = [], [], []
        for tr, va in folds:
            if model == "ranker":
                pr, roc, it = _fit_ranker(params, tr, va, cols)
            else:
                pr, roc, it = _fit_one(params, tr, va, cols,
                                       early_stopping=False)
            prs.append(pr)
            rocs.append(roc)
            iters.append(it)
        trial.set_user_attr("best_iteration", int(np.median(iters)))
        trial.set_user_attr("fold_scores", [round(x, 4) for x in prs])
        trial.set_user_attr("fold_roc", [round(x, 4) for x in rocs])
        trial.set_user_attr("roc_auc", float(np.mean(rocs)))
        trial.set_user_attr("roc_std", float(np.std(rocs)))
        # 分割ごとのばらつきが大きい設定は、たまたま当たっただけの可能性がある。
        # 平均で選ぶが、ばらつきも残して後から見られるようにする
        trial.set_user_attr("score_std", float(np.std(prs)))
        return float(np.mean(prs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**FIXED, **study.best_params, "subsample_freq": 1}
    # 木の本数は探索中ずっと固定なので、そのまま採用する
    best["n_estimators"] = SEARCH_N_ESTIMATORS
    at = study.best_trial.user_attrs
    global LAST_CV
    LAST_CV = {"scheme": scheme, "model": model,
               "n_estimators": SEARCH_N_ESTIMATORS,
               "n_splits": len(folds), "mean_pr_auc": round(study.best_value, 4),
               "std": round(float(at.get("score_std", 0.0)), 4),
               "fold_scores": at.get("fold_scores", []),
               "mean_roc_auc": round(float(at.get("roc_auc", float("nan"))), 4),
               "roc_std": round(float(at.get("roc_std", 0.0)), 4),
               "fold_roc": at.get("fold_roc", []),
               "base_rate": round(float(np.mean(
                   [v["label"].mean() for _, v in folds])), 4),
               # フォールドごとの正例率も残す。層は「年の束 × ラベル」なので
               # ここが揃っているはずだが、記録が無いと後から確かめられない。
               # 揃っていなければ、その探索の PR-AUC は分割間で比較できていない
               # （PR-AUC の下限は正例率そのものなので、正例率がずれると
               #   スコアの差が実力の差なのか正例率の差なのか分からなくなる）。
               "fold_pos_rate": [round(float(v["label"].mean()), 4)
                                 for _, v in folds],
               "fold_pos_rate_spread": round(float(
                   max(v["label"].mean() for _, v in folds)
                   - min(v["label"].mean() for _, v in folds)), 4),
               # 規模構成がフォールド間でどれだけ違うか（構成比の最大差、pt）。
               # 合成データでは 2.803pt -> 0.233pt に締まることを確認したが、
               # 実データで確認する手段が無かった。ここに残せば毎回検証できる。
               "fold_cap_spread_pt": round(_composition_spread(
                   [v for _, v in folds], _cap_bands(df)) * 100, 3),
               "n_trials": n_trials}
    if verbose:
        pr_txt = " ".join(f"{x:.4f}" for x in at.get("fold_scores", []))
        roc_txt = " ".join(f"{x:.4f}" for x in at.get("fold_roc", []))
        print(f"  [tune] {n_trials}試行 / {len(folds)}分割 "
              f"PR-AUC {study.best_value:.4f} (±{at.get('score_std', 0):.4f}) "
              f"/ ROC-AUC {at.get('roc_auc', float('nan')):.4f} "
              f"(±{at.get('roc_std', 0):.4f}) / 木 {best['n_estimators']}本")
        print(f"  [tune] 分割ごと PR-AUC : {pr_txt}")
        print(f"  [tune] 分割ごと ROC-AUC: {roc_txt}")
        rates = [v["label"].mean() for _, v in folds]
        print(f"  [tune] 検証窓の正例率 : "
              + " ".join(f"{r*100:.2f}%" for r in rates)
              + f"（幅 {(max(rates)-min(rates))*100:.3f}pt）")
        print(f"  [tune] 訓練窓の正例率 : "
              + " ".join(f"{t['label'].mean()*100:.2f}%" for t, _ in folds))
        print(f"  [tune] 検証窓の構成ずれ: "
              f"時価総額帯 {LAST_CV['fold_cap_spread_pt']:.3f}pt")
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
