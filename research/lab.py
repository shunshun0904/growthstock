#!/usr/bin/env python3
"""
精度を上げるための実験台。

分析コンペとして詰めるには、思いつきを片端から試すより先に
「何と何を比べたのか」が毎回同じである土台が要る。ここで固定するのは3つ。

1. 分割      本番の out-of-fold と同じウォークフォワード（36ヶ月 / 6ヶ月 / 6ヶ月、
             エンバーゴ = ラベル確定に要る60営業日）。ここを実験ごとに変えると、
             良くなったのが手法なのか分割なのか分からなくなる。
2. 指標      ラベル基準（PR-AUC / ROC-AUC / 日付内AUC / P@5%）と、
             ラベル非依存の実収益（ref_end）の両方を必ず一緒に出す。
             この母集団では両者が一致しないことが分かっているため
             （docs/MODEL_DESIGN_WALKFORWARD.md）、片方だけ見ると判断を誤る。
3. 比較      marginal な信頼区間の重なりでは有意性を判定できない。
             同じ行での差を取り、日付ブロックブートストラップで区間を出す。
             同じ日の銘柄は地合いを共有するので、行単位の再抽出では
             区間が不当に狭くなる。

使い方:

    import lab
    df = lab.frame()                       # ref_end 付きデータセット（キャッシュ）
    base = lab.run(df, lab.lgbm())         # 既定の LightGBM
    exp  = lab.run(df, lab.lgbm(num_leaves=63))
    print(lab.table({"baseline": base, "leaves63": exp}))
    print(lab.compare(base, exp))          # 対比較

`fit` は (X_train, y_train) -> (X_test -> スコア) を返す関数なら何でもよい。
LightGBM 以外のアルゴリズム、アンサンブル、スタッキングも同じ台に載る。
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
DATASET = os.path.join(DATA_DIR, "dataset.parquet")
#: ref_end を毎回作り直すと bars を全部読み直すことになるので置いておく
CACHE = os.path.join(DATA_DIR, "lab.parquet")

#: 本番の out-of-fold と同じ分割。ここを勝手に変えない
MIN_TRAIN_MONTHS = 36
TEST_MONTHS = 6
STEP_MONTHS = 6
MIN_TEST = 200
MIN_TRAIN = 1000

SEED = 42
#: 上位何%を「買う」と見なして評価するか。運用は日に数件なので上位は薄く取る
TOP_PCTS = (1, 5, 10)


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #

def frame(rebuild: bool = False) -> pd.DataFrame:
    """
    データセットに実収益（ref_end / ref_rise）を付けて返す。

    ref_end はラベル定義に一切依存しない物差し。
    ラベルを変える実験をしても、これだけは同じ数字であり続ける。
    """
    if not rebuild and os.path.exists(CACHE) \
            and os.path.getmtime(CACHE) >= os.path.getmtime(DATASET):
        return pd.read_parquet(CACHE)

    import sweep_design as S

    ds = pd.read_parquet(DATASET)
    ds["Date"] = pd.to_datetime(ds["Date"])
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ref = S.reference_outcome(S.Panels(bars).get(B.HIGH_WINDOW))
    out = ds.merge(ref, on=["Code", "Date"], how="left")
    out.to_parquet(CACHE, index=False, compression="zstd")
    return out


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def folds(df: pd.DataFrame) -> List[Fold]:
    """本番 out-of-fold と同じ窓を返す。"""
    import walkforward as WF

    raw = WF.make_folds(pd.to_datetime(df["Date"]),
                        min_train_months=MIN_TRAIN_MONTHS,
                        test_months=TEST_MONTHS, step_months=STEP_MONTHS,
                        embargo_days=B.RISE_HORIZON)
    return [Fold(f.index, pd.Timestamp(f.train_end), pd.Timestamp(f.test_start),
                 pd.Timestamp(f.test_end)) for f in raw]


# --------------------------------------------------------------------------- #
# 学習器（fit ファクトリ）
# --------------------------------------------------------------------------- #

Fitted = Callable[[np.ndarray], np.ndarray]
Fit = Callable[[np.ndarray, np.ndarray], Fitted]


def _spw(y: np.ndarray) -> float:
    """正例が少ないので重みで釣り合わせる。本番と同じ扱い。"""
    pos = float((y == 1).sum())
    return float((len(y) - pos) / pos) if pos else 1.0


def lgbm(params: Optional[Dict] = None, *, balance: bool = True,
         preset: str = "all", seed: Optional[int] = None) -> Fit:
    """
    本番と同じ LightGBM。params 未指定なら探索済みのものを使う。

    seed を渡すと乱数種を上書きする。探索済みパラメータは自前の
    random_state を持っている（探索時のもの）ので、setdefault では
    上書きされない。種を振る実験では必ず明示的に渡すこと。
    """
    import lightgbm as lgb
    import tuning

    p = dict(tuning.params_for(preset) if params is None else params)
    p = {k: v for k, v in p.items() if not k.startswith("_")}
    p.setdefault("verbose", -1)
    if seed is not None:
        p["random_state"] = seed
    else:
        p.setdefault("random_state", SEED)

    def fit(X: np.ndarray, y: np.ndarray) -> Fitted:
        kw = dict(p)
        if balance:
            kw["scale_pos_weight"] = _spw(y)
        m = lgb.LGBMClassifier(**kw)
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


def xgb(seed: int = SEED, **kw) -> Fit:
    """XGBoost。欠損は LightGBM と同様にネイティブで扱える。"""
    import xgboost as xgbm

    p = dict(n_estimators=600, learning_rate=0.05, max_depth=6,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
             tree_method="hist", n_jobs=-1, random_state=seed,
             eval_metric="logloss", **kw)

    def fit(X, y):
        m = xgbm.XGBClassifier(**p, scale_pos_weight=_spw(y))
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


def cat(seed: int = SEED, **kw) -> Fit:
    """CatBoost。欠損の扱いと正則化の仕方が LightGBM と違うので、
    アンサンブルの相方として誤りの出方が変わることを期待する。"""
    from catboost import CatBoostClassifier

    p = dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
             random_seed=seed, verbose=0, allow_writing_files=False, **kw)

    def fit(X, y):
        m = CatBoostClassifier(**p, scale_pos_weight=_spw(y))
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


def rf(seed: int = SEED, **kw) -> Fit:
    """
    ランダムフォレスト。欠損を扱えないので中央値で埋める。

    埋める値は訓練側だけで決めて検証側に当てる（Pipeline が fold の中で
    fit されるので、検証側の分布は漏れない）。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    def fit(X, y):
        m = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   max_features="sqrt", n_jobs=-1,
                                   class_weight="balanced_subsample",
                                   random_state=seed, **kw))
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


def logit(seed: int = SEED, *, scaler: str = "quantile", **kw) -> Fit:
    """
    ロジスティック回帰。木と違って前処理が結果を左右するので、
    欠損補完とスケール変換をここで試せるようにしてある。

      quantile … 分位変換で正規分布に寄せる。外れ値に強い
      standard … 平均0分散1。外れ値の影響が残る

    株価系の特徴量は裾が重いので quantile を既定にした。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    tf = (QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                              random_state=seed)
          if scaler == "quantile" else StandardScaler())

    def fit(X, y):
        m = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), tf,
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                               random_state=seed, **kw))
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]

    return fit


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    name: str
    oof: pd.DataFrame
    metrics: Dict = field(default_factory=dict)
    #: run_multi のときだけ、種ごとの結果が入る
    per_seed: List["Result"] = field(default_factory=list)


def run(df: pd.DataFrame, fit: Fit, *, cols: Optional[Sequence[str]] = None,
        name: str = "model", prep=None, quiet: bool = True) -> Result:
    """
    ウォークフォワードで out-of-fold のスコアを作り、指標まで出す。

    prep は (train_df, test_df) -> (X_train, X_test) の前処理。
    欠損補完やスケール変換を「訓練側だけで決めて検証側に当てる」ために
    ここに挟む。fold の中で呼ぶので、検証側の情報が訓練に漏れない。
    """
    cols = list(cols if cols is not None else F.columns("all"))
    d = pd.to_datetime(df["Date"])
    lab = df["label"].notna()
    parts = []
    for f in folds(df):
        tr = df[(d <= f.train_end) & lab]
        te = df[(d >= f.test_start) & (d <= f.test_end) & lab]
        if len(te) < MIN_TEST or len(tr) < MIN_TRAIN:
            continue
        if prep is None:
            Xtr = tr[cols].to_numpy(dtype=float)
            Xte = te[cols].to_numpy(dtype=float)
        else:
            Xtr, Xte = prep(tr, te, cols)
        ytr = tr["label"].to_numpy(dtype=int)
        predict = fit(Xtr, ytr)
        part = te[["Code", "Date", "label", "ref_end", "ref_rise"]].copy()
        part["score"] = np.asarray(predict(Xte), dtype=float)
        part["fold"] = f.index
        parts.append(part)
        if not quiet:
            print(f"  窓{f.index} {f.test_start.date()}〜{f.test_end.date()}: "
                  f"訓練{len(tr):,} / 検証{len(te):,}")
    if not parts:
        raise SystemExit("out-of-fold を作れません")
    oof = pd.concat(parts, ignore_index=True)
    return Result(name=name, oof=oof, metrics=metrics(oof))


#: 種平均に使う乱数種。実験00 で測ったばらつきはこの5種によるもの
SEEDS = (42, 7, 123, 2024, 31337)


def run_multi(df: pd.DataFrame, make_fit: Callable[[int], Fit], *,
              seeds: Sequence[int] = SEEDS, cols: Optional[Sequence[str]] = None,
              name: str = "model", prep=None) -> Result:
    """
    種を変えて走らせ、スコアを平均する。

    種平均は3つを同時にやる。

      1. 乱数の揺れを均す。実験00 で測ったとおり、単一種だと上位5%収益は
         レンジ0.81pt も動く。5種平均なら標準誤差が 1/√5 になり、
         検出できる効果が 0.8pt -> 0.4pt に下がる。
      2. それ自体がアンサンブル。同じ学習器でも種が違えば誤りの出方が違うので、
         平均すると分散が落ちる。単一種より良くなるのが普通。
      3. 種ごとの結果も残すので、この設定自体のばらつきが後から出せる。

    平均は確率の単純平均。同じ学習器・同じ目的関数なのでスケールが揃っており、
    順位平均にする理由がない（較正の意味も保たれる）。
    """
    per_seed: List[Result] = []
    for s in seeds:
        per_seed.append(run(df, make_fit(s), cols=cols, name=f"{name}#{s}",
                            prep=prep))
    key = ["Code", "Date", "label", "ref_end", "ref_rise", "fold"]
    acc = per_seed[0].oof[key].copy()
    acc["score"] = np.mean([r.oof["score"].to_numpy() for r in per_seed], axis=0)
    return Result(name=name, oof=acc, metrics=metrics(acc), per_seed=per_seed)


def spread(res: Result, key: str = "end_5") -> Dict:
    """種ごとのばらつきを返す（run_multi の結果にだけ意味がある）。"""
    rs = res.per_seed
    if not rs:
        return {}
    v = np.array([r.metrics[key] for r in rs], dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "avg_of_ensemble": float(res.metrics[key])}


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def _auc_in_day(oof: pd.DataFrame) -> float:
    """
    日付内 AUC。同じ日の候補の中で正例を上位に置けているか。

    運用では「その日の候補のどれを買うか」を決めるので、
    日をまたいだ順位より、この指標のほうが決定に近い。
    正例と負例が両方ある日だけで計算する（片方しかない日は定義できない）。
    """
    from sklearn.metrics import roc_auc_score

    vals, weights = [], []
    for _, g in oof.groupby("Date"):
        y = g["label"].to_numpy(dtype=int)
        if 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, g["score"].to_numpy()))
            weights.append(len(g))
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=weights))


#: しきい値運用の既定。画面の pctHistorical がこれを超えたら買う想定
THR_PCT = 90


def threshold_edge(oof: pd.DataFrame, pct: float = THR_PCT) -> Dict:
    """
    「スコアが過去分布の上位(100-pct)%なら買う」運用を測る。

    これが運用そのものの指標。日付内の順位は使わない（買うかどうかは
    その日の1位かではなく、スコアの水準で決めるため）。

    しきい値は窓ごとに「それより前の窓のスコア分布」から出す。全期間の
    分布を使うと未来を見てしきい値を決めることになり、実測で約0.9pt
    楽観側に出た。窓1 は参照分布が無いので対象外。

    再現性を必ず一緒に返す。平均が良くても、勝ちが1つの窓に偏っていれば
    その優位は再現しない（実測では pooled 上位5%の44%が窓9 に集中していた）。
    """
    folds = sorted(oof["fold"].unique())
    picks, per_fold = [], []
    uni = oof[oof["fold"] > folds[0]]
    for f in folds:
        ref = oof.loc[oof["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        thr = float(np.percentile(ref, pct))
        cur = oof[oof["fold"] == f]
        sel = cur[cur["score"] > thr]
        if not len(sel):
            continue
        picks.append(sel)
        se = pd.to_numeric(sel["ref_end"], errors="coerce").mean()
        ce = pd.to_numeric(cur["ref_end"], errors="coerce").mean()
        per_fold.append(float(se - ce))
    if not picks:
        return {"thr_n": 0, "thr_lift": float("nan"), "thr_folds_won": 0,
                "thr_folds": 0, "thr_end": float("nan"),
                "thr_win": float("nan"), "thr_rate": 0.0,
                "thr_lift_same_day": float("nan"), "thr_worst": float("nan")}
    sel = pd.concat(picks, ignore_index=True)
    e = pd.to_numeric(sel["ref_end"], errors="coerce")
    all_end = pd.to_numeric(uni["ref_end"], errors="coerce").mean()
    same = uni[uni["Date"].isin(set(pd.Index(sel["Date"]).unique()))]
    same_end = pd.to_numeric(same["ref_end"], errors="coerce").mean()
    return {
        "thr_n": int(len(sel)),
        "thr_rate": float(len(sel) / len(uni)),
        "thr_end": float(e.mean()) * 100,
        "thr_lift": float(e.mean() - all_end) * 100,
        "thr_lift_same_day": float(e.mean() - same_end) * 100,
        "thr_win": float((e > 0).mean()),
        "thr_folds_won": int(sum(1 for x in per_fold if x > 0)),
        "thr_folds": int(len(per_fold)),
        "thr_worst": float(min(per_fold)) * 100,
    }


def metrics(oof: pd.DataFrame) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = oof["label"].to_numpy(dtype=int)
    s = oof["score"].to_numpy(dtype=float)
    end = pd.to_numeric(oof["ref_end"], errors="coerce")
    out = {
        "n": int(len(oof)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "auc_in_day": _auc_in_day(oof),
        "base_end": float(end.mean()) * 100,
    }
    for k in TOP_PCTS:
        t = oof.nlargest(max(1, int(len(oof) * k / 100)), "score")
        te = pd.to_numeric(t["ref_end"], errors="coerce")
        out[f"p_at_{k}"] = float(t["label"].mean())
        out[f"end_{k}"] = float(te.mean()) * 100
        out[f"lift_{k}"] = out[f"end_{k}"] - out["base_end"]

    # 運用そのものの指標。毎日「その日の1位」を1つ買ったらどうなるか。
    # 上位k% は日をまたいだ選択（いつ買うか）と日の中の選択（何を買うか）が
    # 混ざっている。こちらは日数を固定するので、銘柄選定の腕だけが出る。
    #
    # 同点は乱数で割る。idxmax は最初の行を返すので、スコアが同点のとき
    # 行順（=コード順）で決まってしまう。日付定数の特徴量だけのモデルは
    # その日の全候補が同スコアになるため、これをやると「毎日いちばん小さい
    # コードを買う」戦略を測ることになり、偶然の偏りが実力に見える。
    rng = np.random.default_rng(SEED)
    tb = oof.assign(_tb=rng.random(len(oof)))
    best = (tb.sort_values(["score", "_tb"])
              .groupby("Date", sort=False).tail(1))
    # 同点がどれだけあったかも出す。多ければ pick1 は読めない
    top_by_day = oof.groupby("Date")["score"].transform("max")
    out["pick1_tied"] = float((oof["score"] >= top_by_day - 1e-12)
                              .groupby(oof["Date"]).mean().mean())
    be = pd.to_numeric(best["ref_end"], errors="coerce")
    out["pick1_end"] = float(be.mean()) * 100
    out["pick1_pos"] = float(best["label"].mean())
    out["pick1_win"] = float((be > 0).mean())
    out["pick1_n"] = int(len(best))
    # 同じ日数だけランダムに1件選んだ場合（＝銘柄選定をしない場合）との差
    out["pick1_lift"] = out["pick1_end"] - float(
        oof.groupby("Date")["ref_end"].mean().mean()) * 100

    # 運用の的。しきい値運用の優位と、その再現性
    if "fold" in oof.columns:
        out.update(threshold_edge(oof))
    return out


# --------------------------------------------------------------------------- #
# 比較
# --------------------------------------------------------------------------- #

def _block_ci(values: np.ndarray, dates: np.ndarray, n: int = 2000,
              seed: int = SEED) -> tuple:
    """
    日付ブロックブートストラップ。

    同じ日の銘柄は地合いを共有するので、行単位で再抽出すると
    独立標本を仮定することになり区間が不当に狭くなる。日ごと再抽出する。
    """
    rng = np.random.default_rng(seed)
    uniq = pd.unique(dates)
    idx = {d: np.where(dates == d)[0] for d in uniq}
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[i] = values[np.concatenate([idx[d] for d in pick])].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def compare(a: Result, b: Result, *, k: int = 5, seed: int = SEED) -> Dict:
    """
    同じ行の上で b − a を測る。

    それぞれの marginal な区間が重なっていても差が有意なことはあるし、
    逆もある。対にして差そのものの区間を出さないと判定できない。
    """
    key = ["Code", "Date"]
    m = a.oof[key + ["label", "ref_end", "score"]].merge(
        b.oof[key + ["score"]], on=key, suffixes=("_a", "_b"))
    if len(m) != len(a.oof) or len(m) != len(b.oof):
        raise ValueError(f"行が揃わない: a={len(a.oof)} b={len(b.oof)} 共通={len(m)}")

    dates = m["Date"].to_numpy()
    out = {"n": int(len(m)), "k_pct": k}

    # 上位k% の実収益の差
    n_top = max(1, int(len(m) * k / 100))
    ea = pd.to_numeric(m.nlargest(n_top, "score_a")["ref_end"], errors="coerce").mean()
    eb = pd.to_numeric(m.nlargest(n_top, "score_b")["ref_end"], errors="coerce").mean()
    out["end_a"] = float(ea) * 100
    out["end_b"] = float(eb) * 100
    out["end_diff"] = out["end_b"] - out["end_a"]

    # 行ごとの差で区間を出せるのは順位に依らない量。
    # 実収益は「上位k%」の取り方で行が入れ替わるため、
    # 選ばれた集合ごと日付ブロックで再抽出する
    sel_a = m.nlargest(n_top, "score_a")
    sel_b = m.nlargest(n_top, "score_b")
    rng = np.random.default_rng(seed)
    uniq = pd.unique(dates)
    ia = {d: np.where(sel_a["Date"].to_numpy() == d)[0] for d in pd.unique(sel_a["Date"])}
    ib = {d: np.where(sel_b["Date"].to_numpy() == d)[0] for d in pd.unique(sel_b["Date"])}
    va = pd.to_numeric(sel_a["ref_end"], errors="coerce").to_numpy()
    vb = pd.to_numeric(sel_b["ref_end"], errors="coerce").to_numpy()
    draws = np.empty(2000)
    for i in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ja = np.concatenate([ia[d] for d in pick if d in ia]) if len(pick) else np.array([], int)
        jb = np.concatenate([ib[d] for d in pick if d in ib]) if len(pick) else np.array([], int)
        draws[i] = (np.nanmean(vb[jb]) if len(jb) else np.nan) \
            - (np.nanmean(va[ja]) if len(ja) else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    out["end_ci"] = (float(lo) * 100, float(hi) * 100)
    out["p_better"] = float(np.nanmean(draws > 0))

    for key_, fn in (("pr_auc", "pr"), ("roc_auc", "roc"), ("auc_in_day", "day")):
        out[f"{key_}_a"] = a.metrics[key_]
        out[f"{key_}_b"] = b.metrics[key_]
        out[f"{key_}_diff"] = b.metrics[key_] - a.metrics[key_]
    return out


#: 表に出す列。運用の的（しきい値運用）を左に置き、ラベル基準は参考として右に。
#: 日付内AUC と 毎日1位 は運用と無関係なので既定の表からは外した
#: （metrics には残してあるので必要なら見られる）。
COLS = [("thr_lift", "しきい値優位", "{:+.2f}pt"),
        ("thr_lift_same_day", "対同日候補", "{:+.2f}pt"),
        ("thr_folds_won", "勝った窓", "{:.0f}"), ("thr_worst", "最悪の窓", "{:+.2f}pt"),
        ("thr_end", "実収益", "{:+.2f}%"), ("thr_win", "勝率", "{:.1%}"),
        ("thr_n", "取引数", "{:,.0f}"),
        ("pr_auc", "PR-AUC", "{:.4f}"), ("roc_auc", "ROC-AUC", "{:.4f}")]


def table(results: Dict[str, Result]) -> str:
    """結果を1枚の表にする。"""
    head = f"{'実験':<22}" + "".join(f"{t:>12}" for _, t, _ in COLS)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        row = f"{name[:21]:<22}"
        for k, _, fmt in COLS:
            v = r.metrics.get(k)
            row += f"{(fmt.format(v) if v is not None else '-'):>12}"
        lines.append(row)
    return "\n".join(lines)
