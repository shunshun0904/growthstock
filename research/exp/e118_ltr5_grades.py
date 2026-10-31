#!/usr/bin/env python3
"""
実験118: 直近5営業日の組のランキング学習（W5。本番の ltr5）の段階を 5 → 10 にしたときの影響（検証だけ。本番の ltr5・買いの
規則・画面は変えない）。

運用者の依頼（2026-10-31）
  「label_gain：0〜4 の5段階。翌営業日の寄りで買い、20営業日目の終値で売ったときの収益を、日（または組）の中で5段階に分けたもの
    直近5営業日モデルは、銘柄数が多くなると思うので、10段階にできるならそっちの方が良いと思いました。可能なら、まず検証から
    実施してください。変更による影響具合をみたいです。」（docs/MODEL_ADOPTION_RULES.md §88）

組・収益・窓は実験117 と同じ（e117 の rolling_frame・W.attach_week_exit・E41.folds_for の 0 / 2 / 4か月ずらし。判断日の翌営業日の
寄りで買い、+10% の指値、届かなければ20営業日目の終値）。学ぶのは判断日がテスト窓より前の組の行、採点するのはテスト窓の組の行で、
窓ごと（6か月ごと）に学び直す（本番は週ごとに学び直す）。

腕（学ぶ並びの段階の数と木の形だけが違う。どれも LGBMRanker・360列＋値動き4列・種42・木200本・label_gain は段階そのもの）
  G5    今の W5: 組の中の20営業日目の終値の5分位（段階 0〜4）。木の形は本番の research/ltr5_params.json
  G10   10分位（段階 0〜9）。木の形は G5 と同じ（段階の数の効き目だけを分ける）
  G10T  10分位で探索し直した木の形（本番の探索と同じ: run_tuning.py と同じ打ち切り・暦月の塊の5分割・前後4営業日を外す・TPE・
        種0・50試行。目的は検証分割の組の中の NDCG@5 で、利得を10段階にしたもの）。本番を10段階にしたら毎週の探索がこうなる形
  D     今のランキング学習（日次。research/ltr_params.json）。見るだけの表で3条件の D1（その日の候補の中の1位）を付けるためだけ

比べるもの（ずらし方ごと。採点するのはテスト窓の組の行のうち収益がある行）
  1. 組の1位（同じスコアは銘柄コードの順）を判断日の翌営業日の寄りで買ったとき（毎日1件。持っている銘柄の重複は問わない）の
     今の売り方の収益（全件・窓の平均）・+10% 到達・−10% 未満・20営業日目の終値。上位2件も。G5 との窓ごとの差
  2. 組の中の順位相関（今の売り方の収益 / 20営業日目の終値。組ごとの Spearman の平均。候補5件以上の組）と、G5 との窓ごとの差
  3. 参考: 組の中の NDCG@5 を2つの物差しで（利得が5段階 / 10段階）。それぞれの物差しで学んだ腕に有利なので、読み方には入れない
  4. 変わり方: 組の1位が G5 と違う判断日の割合、上位2件の顔ぶれが違う割合、組の中の G5 のスコアとの順位相関の平均
  5. 組の大きさ（その判断日の組の行の数。3分位）ごとの、組の1位の平均収益と G5 との差（運用者の見立て「銘柄数が多くなると
     思うので」）。組の大きさの分布（中央値・10件以上 / 20件以上の組の割合）と、10段階の段階ごとの行の割合
  6. 実験117 の毎日の買い（上位2件のうち、今の買いの規則 R0 = 1.2σ の木3本がすべて95以上を満たす未購入。数える判断日は R0 の
     印が組のすべての行に付いた日）の件数・平均収益（全件・窓の平均）・−10% 未満と、画面の3枠で回した取引
  7. 見るだけ: 3条件（日次の1位・組の1位・R0）・4条件（＋1.5σ の木3本 R15）・組の1位＋R0 の件数と平均収益（腕ごとに組の1位を
     入れ替えて数える。数える判断日は R0 と R15 の印が組のすべての行に付いた日。持っている銘柄は買い増さない）

読み方（回す前に決めたもの。採用の判定ではない。入れるかは運用者が決める）。G10 と G10T を別々に G5 と比べる
  - 3通りのずらしすべてで、(a) 組の1位の今の売り方の収益の平均（窓の平均。%の小数2桁に丸めて）が G5 より高く、かつ (b) 組の中の
    順位相関（今の売り方の収益。小数3桁に丸めて）が G5 より高い →「10段階のほうが良い」
  - 3通りすべてで (a)・(b) とも G5 より低い →「10段階のほうが悪い」
  - それ以外 →「はっきりとは変わらない」

確かめ（合わなければ止まる）
  - 実験117 と同じ組の確かめ（日次の出口の計算・組の中身・値動きの列・判断日に入った候補の収益）
  - 段階: この台本の5段階の付け方が、本番（ltr.grades）と組のすべての行で同じ。10段階は 0〜9
  - G5 の当てはめ: ずらし0の最初の窓の訓練で、この台本の5段階の当てはめのスコアが本番の当てはめ（ltr.fit・ltr.predict）と同じ
  - 木の形: 本番の ltr5_params.json が 360列＋値動き4列で探索したものであること（ltr.params_mismatch）
  - 知らせるだけ（止めない）: 本番の ltr5_params.json の探索の行の数・期間が、この回の探索の行と同じか

公開ログには集計だけ（件数・平均・割合・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e118_*（探索の結果と
out-of-fold は置き場に残し、同じデータ・同じ設定なら次の回は読む）。

    exp=e118_ltr5_grades.py  args=""   （既定は --trials 50 --shifts 0,2,4 --top 2）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import ltr as LTR  # noqa: E402
import ltr5 as L5  # noqa: E402
import tuning  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402
import e114_ltr_retest as E  # noqa: E402
import e115_weekly_ltr as W  # noqa: E402
import e117_rolling5_ltr as R  # noqa: E402

PREFIX = "e118"
OOF_DIR = E.OOF_DIR
PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")

RET, HIT, HITDAY, R20 = E.RET, E.HIT, E.HITDAY, E.R20
K = R.K                           # 組の長さ（直近5営業日）
MOVE_COLS = R.MOVE_COLS
TOP_N = R.TOP_N                   # 毎日見る上位の件数（実験117 と同じ 2）
SEED = L5.SEED                    # 学習の種（本番と同じ 42）
TUNE_SEED = L5.TUNE_SEED          # 探索の種（本番と同じ 0）
NDCG_K = L5.NDCG_K
MIN_POOL = L5.MIN_POOL            # NDCG・順位相関に使う組の最小の行数
MIN_TEST, MIN_TRAIN = R.MIN_TEST, R.MIN_TRAIN
KEEP = R.KEEP_R
ARMS = ("G5", "G10", "G10T")
GRADES = {"G5": 5, "G10": 10, "G10T": 10}
LABELS = {"G5": "G5   今の W5（5段階・本番の木の形）", "G10": "G10  10段階（木の形は G5 と同じ）",
          "G10T": "G10T 10段階（10段階で探索し直した木の形）"}
R0_TREES = tuple(R.R0_TREES)      # 1.2σ の木3本（今の買いの規則）
R15_TREES = tuple(R.RULE_TREES)   # 1.5σ の木3本（4条件）
SIZE_BINS = R.POOL_BINS           # 小・中・大
MARK_SETS = ("C3", "C4", "W1R0")
MARK_JA = {"C3": "3条件（日次の1位・組の1位・R0）", "C4": "4条件（3条件＋R15）", "W1R0": "組の1位＋R0（日次の順位は問わない）"}
READ_JA = {"better": "10段階のほうが良い", "worse": "10段階のほうが悪い", "none": "はっきりとは変わらない"}
FIT_TOL = 1e-9                    # G5 の当てはめと本番の当てはめのスコアの差の許し

log = E.log


# --------------------------------------------------------------------------- #
# 段階・当てはめ・探索
# --------------------------------------------------------------------------- #

def grades(dates, r, n: int) -> np.ndarray:
    """組（日付）の中の収益の n 分位（0〜n−1 の整数。1件だけの組は真ん中）。n = 5 で本番の ltr.grades と同じ式。"""
    p = LTR.within_date_pct(dates, r)
    return np.minimum((p * n).astype(int), n - 1)


def fit_rank(params: Dict, tr: pd.DataFrame, te: pd.DataFrame, cols: Sequence[str], n: int,
             seed: int = SEED) -> np.ndarray:
    """
    tr の組の中の20営業日目の終値の n 分位を学ぶ LambdaRank（label_gain = 0〜n−1）で te を採点する。n = 5 で本番の
    ltr.fit・ltr.predict と同じ（(Date, Code) の順に並べ、組の大きさは日ごとの件数）。
    """
    import lightgbm as lgb

    cols = list(cols)
    d = tr.sort_values(["Date", "Code"], kind="mergesort")
    y = grades(d["Date"], d[R20].to_numpy(dtype=float), n)
    sizes = d.groupby("Date", sort=True).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(n)), **LTR.ranker_params(params, seed))
    m.fit(d[cols].to_numpy(dtype=float), y, group=sizes)
    return np.asarray(m.predict(te[cols].to_numpy(dtype=float)), dtype=float)


def tune_g(sub: pd.DataFrame, cols_w: Sequence[str], folds, n: int, n_trials: int = L5.TUNE_TRIALS,
           seed: int = TUNE_SEED) -> Dict:
    """
    木の形の探索（本番の ltr5.tune と同じ: 探索空間 ltr.search_space、各試行で分割ごとに学習→検証の組の中の NDCG@5、その平均を
    最大にする。TPE の種・探索中の当てはめの種はどれも seed）。違うのは段階の数 n（学ぶ並びと NDCG の利得の両方）。n = 5 で
    本番と同じ結果になる。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols_w = list(cols_w)
    t0 = time.time()
    if not folds:
        raise SystemExit("探索の分割を作れない（期間が短すぎる）")

    def objective(trial):
        p = LTR.search_space(trial)
        nd = []
        for tr, va in folds:
            s = fit_rank(p, tr, va, cols_w, n, seed)
            nd.append(LTR.ndcg_at_k(va["Date"], s, grades(va["Date"], va[R20].to_numpy(dtype=float), n),
                                    NDCG_K, MIN_POOL))
        trial.set_user_attr("ndcg_folds", [round(x, 4) for x in nd])
        return float(np.mean(nd))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {"boosting_type": "gbdt", "n_estimators": tuning.SEARCH_N_ESTIMATORS, "subsample_freq": 1,
            "n_jobs": -1, "verbose": -1, **study.best_params}
    d = pd.to_datetime(sub["Date"])
    return {"params": best,
            "_cv": {"objective": (f"検証分割の組の中の NDCG@{NDCG_K}（候補{MIN_POOL}件以上の組。利得は組の中の20営業日目の"
                                  f"終値の段階 0〜{n - 1}）"),
                    "grades": int(n), "n_splits": len(folds), "n_trials": int(n_trials), "seed": int(seed),
                    "best_trial": int(study.best_trial.number), "ndcg": round(float(study.best_value), 4),
                    "ndcg_folds": study.best_trial.user_attrs.get("ndcg_folds"),
                    "n_rows": int(len(sub)), "n_days": int(d.nunique()),
                    "train_from": str(d.min().date()), "train_to": str(d.max().date()),
                    "minutes": round((time.time() - t0) / 60, 1)}}


def check_fit(rl: pd.DataFrame, cols_w: Sequence[str], params: Dict, n_test: int = 2000) -> Dict:
    """ずらし0の最初の窓の訓練で、この台本の5段階の当てはめと本番の当てはめ（ltr.fit・ltr.predict）のスコアが同じか。"""
    dr = pd.to_datetime(rl["Date"])
    for f in E41.folds_for(rl["Date"], 0):
        lo, hi, end = pd.Timestamp(f.test_start), pd.Timestamp(f.test_end), pd.Timestamp(f.train_end)
        tr = rl[dr <= end]
        te = rl[(dr >= lo) & (dr <= hi)]
        if len(te) < MIN_TEST or len(tr) < MIN_TRAIN:
            continue
        te = te.head(n_test)
        mine = fit_rank(params, tr, te, cols_w, 5)
        prod = LTR.predict(LTR.fit(tr, cols_w, params, SEED), te[list(cols_w)].to_numpy(dtype=float))
        diff = float(np.max(np.abs(mine - prod))) if len(te) else 0.0
        return {"fold": int(f.index), "train": int(len(tr)), "rows": int(len(te)), "max_diff": diff,
                "same": bool(diff <= FIT_TOL)}
    return {"fold": None, "train": 0, "rows": 0, "max_diff": float("nan"), "same": False}


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

def run_tag(params: Dict[str, Dict], cols: Sequence[str], cols_w: Sequence[str], rl: pd.DataFrame,
            daily: pd.DataFrame) -> str:
    """保存名に入れる目印（木の形・段階・列・データが変わったら別の名前にして、古い out-of-fold を読まない）。"""
    key = {"params": {a: {k: p[k] for k in sorted(p)} for a, p in params.items()}, "grades": dict(GRADES),
           "cols": F.signature(list(cols)), "cols_w": F.signature(list(cols_w)),
           "rl_rows": int(len(rl)), "rl_last": str(pd.to_datetime(rl["Date"]).max().date()),
           "rl_r20": int(rl[R20].notna().sum()), "d_rows": int(len(daily)),
           "d_last": str(pd.to_datetime(daily["Date"]).max().date()), "d_r20": int(daily[R20].notna().sum()),
           "seed": SEED, "k": K, "v": 1}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_shift(rl: pd.DataFrame, daily: pd.DataFrame, cols: Sequence[str], cols_w: Sequence[str],
              params: Dict[str, Dict], shift: int, out_dir: str = OOF_DIR) -> pd.DataFrame:
    """
    1つのずらし方の out-of-fold（実験117 の oof_shift と同じ窓・同じ行）。組の行（収益があるもの）を3つの腕と D で採点した表
    （score_G5・score_G10・score_G10T・score_D・fold）。G5・G10・G10T は判断日が訓練の最終日以前の組の行で、D は候補の日が訓練の
    最終日以前の日次の行で学ぶ。保存済みなら読む。
    """
    tag = run_tag(params, cols, cols_w, rl, daily)
    path = os.path.join(out_dir, f"{PREFIX}_pool_{tag}_sh{shift}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    folds = E41.folds_for(rl["Date"], shift)
    dr = pd.to_datetime(rl["Date"])
    dd = pd.to_datetime(daily["Date"])
    has_r20 = daily[R20].notna()
    rows = []
    for f in folds:
        lo, hi, end = pd.Timestamp(f.test_start), pd.Timestamp(f.test_end), pd.Timestamp(f.train_end)
        te = rl[(dr >= lo) & (dr <= hi)]
        tr_r = rl[dr <= end]
        if len(te) < MIN_TEST or len(tr_r) < MIN_TRAIN:
            continue
        tr_l = daily[(dd <= end) & has_r20]
        part = te[list(KEEP)].copy()
        for a in ARMS:
            part[f"score_{a}"] = fit_rank(params[a], tr_r, te, cols_w, GRADES[a])
        part["score_D"] = W.fit_rank(params["D"], tr_l, te, cols)
        part["fold"] = int(f.index)
        rows.append(part)
    if not rows:
        raise SystemExit(f"ずらし{shift}か月: 窓を作れない（期間が短すぎる）")
    o = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    o.to_parquet(path, index=False)
    log(f"  ずらし{shift}か月: {o['fold'].nunique()}窓・組の行 {len(o):,}件・{o['Date'].nunique():,}日 {time.time() - t0:.0f}秒")
    return o


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def top_rows(o: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
    """判断日ごとの組の上位 n 件（スコアの高い順。同じスコアは銘柄コードの順）。"""
    s = o.sort_values(["Date", score, "Code"], ascending=[True, False, True], kind="mergesort")
    return s.groupby("Date", sort=True).head(n)


def rank_stats(o: pd.DataFrame, score: str) -> Dict:
    """組の中の NDCG@5（利得が5段階 / 10段階）と順位相関（今の売り方の収益 / 20営業日目の終値）。"""
    s = o[score].to_numpy(dtype=float)
    r20 = o[R20].to_numpy(dtype=float)
    return {"ndcg5": LTR.ndcg_at_k(o["Date"], s, grades(o["Date"], r20, 5), NDCG_K, MIN_POOL),
            "ndcg10": LTR.ndcg_at_k(o["Date"], s, grades(o["Date"], r20, 10), NDCG_K, MIN_POOL),
            "rho": E53.date_rho(o["Date"], s, o[RET].to_numpy(dtype=float), min_n=MIN_POOL),
            "rho20": E53.date_rho(o["Date"], s, r20, min_n=MIN_POOL)}


def fold_rho(o: pd.DataFrame, score: str) -> pd.DataFrame:
    """窓ごとの組の中の順位相関（今の売り方の収益。候補 MIN_POOL 件以上の組）。"""
    rows = [{"fold": int(f), "rho": E53.date_rho(g["Date"], g[score].to_numpy(dtype=float),
                                                 g[RET].to_numpy(dtype=float), min_n=MIN_POOL)}
            for f, g in o.groupby("fold", sort=True)]
    return pd.DataFrame(rows, columns=["fold", "rho"])


def change_stats(o: pd.DataFrame, arm: str, base: str = "G5") -> Dict:
    """組の1位・上位2件の顔ぶれが base と違う判断日の割合と、組の中の base のスコアとの順位相関の平均。"""
    t1a = top_rows(o, f"score_{arm}", 1).set_index("Date")["Code"]
    t1b = top_rows(o, f"score_{base}", 1).set_index("Date")["Code"].reindex(t1a.index)
    t2a = top_rows(o, f"score_{arm}", 2).groupby("Date", sort=True)["Code"].apply(frozenset)
    t2b = top_rows(o, f"score_{base}", 2).groupby("Date", sort=True)["Code"].apply(frozenset).reindex(t2a.index)
    rho = E53.date_rho(o["Date"], o[f"score_{arm}"].to_numpy(dtype=float), o[f"score_{base}"].to_numpy(dtype=float),
                       min_n=MIN_POOL)
    return {"days": int(len(t1a)), "top1_changed": float((t1a != t1b).mean()) if len(t1a) else np.nan,
            "top2_changed": float((t2a != t2b).mean()) if len(t2a) else np.nan, "rho_base": float(rho)}


def size_stats(o: pd.DataFrame) -> Dict:
    """
    組の大きさ（判断日の組の行の数。収益がある行）の分布と、3分位（分位点ちょうどの日は小さい側。e117.size_bins）ごとの
    腕ごとの組の1位の平均収益と G5 との差。
    """
    size = o.groupby("Date", sort=True).size()
    bins, cuts = R.size_bins(size.to_numpy())
    b = pd.Series(bins, index=size.index)
    tops = {a: top_rows(o, f"score_{a}", 1).set_index("Date") for a in ARMS}
    out = {"days": int(len(size)), "q": [float(size.quantile(q)) for q in (0.25, 0.5, 0.75)] if len(size) else [],
           "ge10": float((size >= 10).mean()) if len(size) else np.nan,
           "ge20": float((size >= 20).mean()) if len(size) else np.nan,
           "cuts": [float(x) for x in cuts], "bins": []}
    for k, nm in enumerate(SIZE_BINS):
        idx = b.index[b.to_numpy() == k]
        sz = size.reindex(idx)
        row = {"name": nm, "days": int(len(idx)), "n_min": int(sz.min()) if len(sz) else None,
               "n_max": int(sz.max()) if len(sz) else None, "ret": {}, "diff": {}}
        for a in ARMS:
            row["ret"][a] = float(tops[a][RET].reindex(idx).mean()) if len(idx) else np.nan
        for a in ARMS[1:]:
            row["diff"][a] = row["ret"][a] - row["ret"]["G5"]
        out["bins"].append(row)
    return out


def flagged(o: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """cols の印が組のすべての行に付いた判断日の組だけ（印の無い候補が上位に来た日を、条件を満たさない扱いにしないため）。"""
    ok = np.ones(len(o), dtype=bool)
    for c in cols:
        ok &= o[c].notna().groupby(o["Date"]).transform("all").to_numpy(dtype=bool)
    return o[ok].reset_index(drop=True)


def mark_buys(o: pd.DataFrame, days, arm: str, hold: int = E.HOLD) -> Dict[str, pd.DataFrame]:
    """
    3条件（日次の1位・組の1位・R0）・4条件（＋R15）・組の1位＋R0 を、腕 arm の組の中の順位で数える（実験117 の triple_buys と
    同じ作り: 候補の日の行（r5_days = 0）に、その日の候補の中の D の順位と組全体の中の腕の順位を付ける。どちらも同じスコアは
    銘柄コードの順。候補の日の翌営業日の寄りで買い、持っている銘柄は買い増さない）。o には r0・r15 の印が要る。
    """
    cal = W.calendar(days)
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    x = o.reset_index(drop=True).copy()
    s = x.sort_values(["Date", f"score_{arm}", "Code"], ascending=[True, False, True], kind="mergesort")
    x["w_rank"] = 0
    x.loc[s.index, "w_rank"] = (s.groupby("Date", sort=False).cumcount() + 1).to_numpy()
    s = x[x["r5_days"] == 0].sort_values(["Date", "score_D", "Code"], ascending=[True, False, True], kind="mergesort")
    x["d_rank"] = 0
    x.loc[s.index, "d_rank"] = (s.groupby("Date", sort=False).cumcount() + 1).to_numpy()
    day0 = (x["r5_days"] == 0).to_numpy()
    r0 = (pd.to_numeric(x["r0"], errors="coerce") == 1.0).to_numpy()
    r15 = (pd.to_numeric(x["r15"], errors="coerce") == 1.0).to_numpy()
    d1 = (x["d_rank"] == 1).to_numpy()
    w1 = (x["w_rank"] == 1).to_numpy()
    masks = {"C3": day0 & d1 & w1 & r0, "C4": day0 & d1 & w1 & r0 & r15, "W1R0": day0 & w1 & r0}
    out: Dict[str, pd.DataFrame] = {}
    for k, m in masks.items():
        sel = x[m].sort_values(["Date", "Code"], kind="mergesort")
        out[k] = R._take(sel, np.ones(len(sel), dtype=bool), pos, hold)
    return out


def mark_stat(b: pd.DataFrame, months: int) -> Dict:
    st = W.pick_stats(b)
    r = b[RET].to_numpy(dtype=float)
    st["pm"] = float(len(b) / months) if months else np.nan
    st["se"] = float(np.std(r, ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    st["win"] = float(np.mean(r > 0)) if len(r) else np.nan
    return st


def evaluate(o: pd.DataFrame, days, top_n: int = TOP_N) -> Dict:
    """
    1つのずらし方の集計。o は組の out-of-fold に R0 の印（r0）と R15 の印（r15）を足したもの。組の1位・上位 top_n 件・組の中の
    並び・変わり方・組の大きさは収益がある組の行すべてで、R0 で絞った毎日の買いは R0 の印が組のすべてに付いた判断日で、3条件・
    4条件は R0 と R15 の印が組のすべてに付いた判断日で数える。
    """
    res: Dict = {"windows": int(o["fold"].nunique()) if len(o) else 0, "rows": int(len(o)),
                 "days": int(o["Date"].nunique()) if len(o) else 0, "months": R.n_months(o),
                 "top1": {}, "top2": {}, "rank": {}, "diff_top1": {}, "diff_top1_bad": {}, "diff_rho": {}, "change": {}}
    t1 = {a: top_rows(o, f"score_{a}", 1) for a in ARMS}
    pw1 = {a: W.per_window(t) for a, t in t1.items()}
    fr = {a: fold_rho(o, f"score_{a}") for a in ARMS}
    for a in ARMS:
        res["top1"][a] = W.pick_stats(t1[a])
        res["top2"][a] = W.pick_stats(top_rows(o, f"score_{a}", top_n))
        res["rank"][a] = rank_stats(o, f"score_{a}")
    for a in ARMS[1:]:
        res["diff_top1"][a] = E.fold_diff(pw1["G5"], pw1[a])
        res["diff_top1_bad"][a] = E.fold_diff(pw1["G5"], pw1[a], "bad")
        res["diff_rho"][a] = E.fold_diff(fr["G5"], fr[a], "rho")
        res["change"][a] = change_stats(o, a)
    res["size"] = size_stats(o)
    of = flagged(o, ("r0",))
    nm = R.n_months(of)
    buys = {a: R.daily_buys(of, f"score_{a}", days, top_n, need="r0") for a in ARMS}
    res["rule"] = {"days": int(of["Date"].nunique()) if len(of) else 0, "months": nm,
                   "buys": {a: W.pick_stats(b) for a, b in buys.items()},
                   "slots": {a: R.slot_trades(b, days, nm) for a, b in buys.items()}}
    of2 = flagged(o, ("r0", "r15"))
    nm2 = R.n_months(of2)
    res["marks"] = {"days": int(of2["Date"].nunique()) if len(of2) else 0, "months": nm2,
                    "sets": {a: {k: mark_stat(b, nm2) for k, b in mark_buys(of2, days, a).items()} for a in ARMS}}
    return res


def reading(rows: List[Dict]) -> Dict:
    """
    読み方（回す前に決めたもの。採用の判定ではない）。rows はずらし方ごとの evaluate の結果（shift を足したもの）。腕（G10・G10T）
    ごとに、すべてのずらしで (a) 組の1位の平均収益（窓の平均。%の小数2桁に丸めて）と (b) 組の中の順位相関（今の売り方の収益。
    小数3桁に丸めて）がどちらも G5 より高ければ better、どちらも低ければ worse、それ以外は none。
    """
    out: Dict = {}
    for a in ARMS[1:]:
        each = []
        for r in rows:
            ra, rb = round(r["top1"][a]["ret_w"] * 100, 2), round(r["top1"]["G5"]["ret_w"] * 100, 2)
            pa, pb = round(r["rank"][a]["rho"], 3), round(r["rank"]["G5"]["rho"], 3)
            each.append({"shift": r["shift"], "ret": [rb, ra], "rho": [pb, pa],
                         "up": bool(ra > rb and pa > pb), "down": bool(ra < rb and pa < pb)})
        read = ("better" if each and all(x["up"] for x in each)
                else "worse" if each and all(x["down"] for x in each) else "none")
        out[a] = {"read": read, "rows": each}
    return out


# --------------------------------------------------------------------------- #
# 表示（公開ログ。集計だけ）
# --------------------------------------------------------------------------- #

_p, _s = W._p, W._s


def _pt(x: float) -> str:
    return f"{x * 100:+.2f}pt" if np.isfinite(x) else "—"


def _d(d: Dict) -> str:
    """窓ごとの差（E.fold_diff）の表示。"""
    if not d.get("n"):
        return "—"
    se = f" ± {d['se'] * 100:.2f}" if np.isfinite(d.get("se", np.nan)) else ""
    return f"{d['mean'] * 100:+.2f}pt{se}（上回った窓 {d['won']}/{d['n']}）"


def _dr(d: Dict) -> str:
    if not d.get("n"):
        return "—"
    se = f" ± {d['se']:.3f}" if np.isfinite(d.get("se", np.nan)) else ""
    return f"{d['mean']:+.3f}{se}（上回った窓 {d['won']}/{d['n']}）"


def report(sh: int, res: Dict, top_n: int) -> None:
    print(f"\n■ ずらし{sh}か月: {res['windows']}窓・判断日 {res['days']:,}日・組の行 {res['rows']:,}件（{res['months']}か月）")
    print("  組の1位を毎日1件（判断日の翌営業日の寄りで買う。持っている銘柄の重複は問わない）: 件数 / 平均収益（全件・窓の平均）/ "
          "+10%到達 / −10%未満 / 20営業日目の終値")
    for a in ARMS:
        t = res["top1"][a]
        print(f"    {LABELS[a]:<28} {t['n']:>6,}件 {_p(t['ret']):>8} / {_p(t['ret_w']):>8}  {_s(t['hit']):>6}  "
              f"{_s(t['bad']):>6}  {_p(t['r20']):>8}")
    print(f"  上位{top_n}件:")
    for a in ARMS:
        t = res["top2"][a]
        print(f"    {LABELS[a]:<28} {t['n']:>6,}件 {_p(t['ret']):>8} / {_p(t['ret_w']):>8}  {_s(t['hit']):>6}  "
              f"{_s(t['bad']):>6}  {_p(t['r20']):>8}")
    print("  組の中の並び: NDCG@5（利得 5段階 / 10段階。参考）・順位相関（今の売り方 / 20営業日目の終値）")
    for a in ARMS:
        k = res["rank"][a]
        print(f"    {LABELS[a]:<28} {k['ndcg5']:.4f} / {k['ndcg10']:.4f}   {k['rho']:+.4f} / {k['rho20']:+.4f}")
    for a in ARMS[1:]:
        c = res["change"][a]
        print(f"  {a} − G5（窓ごとの差）: 組の1位の平均収益 {_d(res['diff_top1'][a])}・−10%未満 "
              f"{_d(res['diff_top1_bad'][a])}・順位相関 {_dr(res['diff_rho'][a])}")
        print(f"    変わり方: 組の1位が G5 と違う判断日 {_s(c['top1_changed'])}・上位2件の顔ぶれが違う {_s(c['top2_changed'])}・"
              f"組の中の G5 のスコアとの順位相関 {c['rho_base']:+.3f}")
    z = res["size"]
    q = z["q"]
    print(f"  組の大きさ（判断日の組の行の数）: 25% / 中央 / 75% 点 {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}件・10件以上の組 "
          f"{_s(z['ge10'])}・20件以上 {_s(z['ge20'])} ｜ 3分位ごとの組の1位の平均収益（G5 / G10 / G10T。括弧は G5 との差）")
    for b in z["bins"]:
        print(f"    {b['name']} {b['days']:>5}日 {b['n_min']}〜{b['n_max']}件: {_p(b['ret']['G5'])} / {_p(b['ret']['G10'])}"
              f"（{_pt(b['diff']['G10'])}）/ {_p(b['ret']['G10T'])}（{_pt(b['diff']['G10T'])}）")
    u = res["rule"]
    print(f"  今の規則（R0 = 1.2σ の3本95以上）で絞った毎日の買い（上位{top_n}件のうち未購入。数える判断日 {u['days']:,}日・"
          f"{u['months']}か月）: 件数 / 平均収益（全件・窓の平均）/ −10%未満 ｜ 3枠の取引 件数・平均・資金全体の月あたり")
    for a in ARMS:
        t, sl = u["buys"][a], u["slots"][a]
        print(f"    {LABELS[a]:<28} {t['n']:>5}件 {_p(t['ret']):>8} / {_p(t['ret_w']):>8}  {_s(t['bad']):>6} ｜ "
              f"{sl['n']:>4}件 {_p(sl['ret']):>8} {_p(sl['capital_pm']):>8}")
    m = res["marks"]
    print(f"  見るだけ: 3条件・4条件（数える判断日 {m['days']:,}日・{m['months']}か月。件数 / 月あたり / 平均収益 ± 標準誤差 / "
          f"勝率 / −10%未満）")
    for k in MARK_SETS:
        print(f"    {MARK_JA[k]}")
        for a in ARMS:
            t = m["sets"][a][k]
            se = f" ± {t['se'] * 100:.2f}" if np.isfinite(t["se"]) else ""
            print(f"      {LABELS[a]:<28} {t['n']:>4}件 {t['pm']:.2f}件/月 {_p(t['ret'])}{se}  {_s(t['win'])}  "
                  f"{_s(t['bad'])}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験118: 直近5営業日の組のランキング学習の段階を 5 → 10 にした影響（検証だけ）")
    ap.add_argument("--trials", type=int, default=L5.TUNE_TRIALS)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    t_all = time.time()
    cols = list(F.columns(F.DEFAULT_PRESET))
    cols_w = cols + list(MOVE_COLS)
    rec5 = L5.load_params()
    why = LTR.params_mismatch(rec5, cols_w)
    if why:
        raise SystemExit(f"本番の ltr5_params.json が {len(cols_w)}列（360列＋値動き4列）で探索したものではない（{why}）")
    rec_d = W.daily_ltr_params(cols)
    os.makedirs(OOF_DIR, exist_ok=True)

    full, bars, cutoff, chk = W.load_all(cols)
    days = W.calendar(bars["Date"].unique())
    rl, info = R.rolling_frame(full, bars, days, cols)
    rl, xinfo = W.attach_week_exit(rl, bars, days)
    cm = R.check_move_cols(rl, bars, days)
    cp = R.check_pools(rl, full, days)
    sd = R.check_same_day(rl)
    del bars
    daily = full[full["label"].notna()].reset_index(drop=True)
    del full

    print("=" * 100)
    print(f"実験118 直近{K}営業日の組のランキング学習の段階 5 → 10（検証だけ）: 日次の候補 {info['daily_rows']:,}件 → 組の行 "
          f"{info['rows']:,}件・判断日 {info['days']:,}日（{info['first']} 〜 {info['last']}）/ {F.DEFAULT_PRESET} {len(cols)}列＋"
          f"値動き{len(MOVE_COLS)}列 / ずらし {shifts}か月")
    print(f"  収益: 判断日の翌営業日の寄りで買い、+{E.TARGET:g}% の指値、届かなければ{E.HOLD}営業日目の終値（今の売り方）/ "
          f"学ぶ並び: 組の中の{E.HOLD}営業日目の終値の5分位（G5）・10分位（G10・G10T）")
    print("=" * 100)
    log(f"[確かめ] 日次の出口の計算: 同じ計算の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,} 行")
    if chk["hold5_both"] and chk["hold5_match"] / chk["hold5_both"] < 0.99:
        raise SystemExit("出口の計算が ret_o1_20 と合わない（日足と学習データの株価がずれている）")
    log(f"[確かめ] 組の中身をデータセットから直接選び直す: {cp['same']} / {cp['days']} 日が同じ ｜ 値動き4列と r5_days を日足から"
        f"直接計算し直す: {cm['same']} / {cm['rows']} 行が同じ ｜ 判断日に入った候補の収益の合わない行 {sd['mismatch']}")
    if cp["same"] != cp["days"]:
        raise SystemExit("組の中身が、データセットから直接選び直したものと合わない")
    if sd["mismatch"]:
        raise SystemExit("判断日に入った候補で、組の収益が日次の収益と合わない（収益の付け方がずれている）")
    if cm["same"] != cm["rows"]:
        raise SystemExit("値動きの列が日足から直接計算した値と合わない")
    rule0, r0info = R.daily_rule(daily, models=R0_TREES)
    if not r0info.get("found"):
        raise SystemExit("本番の out-of-fold（1.2σ の木3本）が置き場に無いので、今の規則の印（R0）を作れない")
    rule15, r15info = R.daily_rule(daily, models=R15_TREES)
    if not r15info.get("found"):
        raise SystemExit("本番の out-of-fold（1.5σ の木3本）が置き場に無いので、R15 の印を作れない")
    log(f"[確かめ] 印（本番の out-of-fold・実験82 と同じ作り）: R0（1.2σ の3本）{r0info['r15']:,}件・前向きの百分位が付く "
        f"{r0info['with_pct']:,}行 ｜ R15（1.5σ の3本）{r15info['r15']:,}件・{r15info['with_pct']:,}行")

    rl_eval = rl[rl[RET].notna() & rl[R20].notna()].reset_index(drop=True)
    r20 = rl_eval[R20].to_numpy(dtype=float)
    g5 = grades(rl_eval["Date"], r20, 5)
    if not np.array_equal(g5, LTR.grades(rl_eval["Date"], r20)):
        raise SystemExit("5段階の段階の付け方が本番（ltr.grades）と合わない")
    g10 = grades(rl_eval["Date"], r20, 10)
    if g10.min() < 0 or g10.max() > 9:
        raise SystemExit("10段階の段階が 0〜9 に収まらない")
    size = rl_eval.groupby("Date").size()
    gshare = [float(np.mean(g10 == j)) for j in range(10)]
    log(f"[確かめ] 段階: 5段階は本番（ltr.grades）と {len(g5):,} / {len(g5):,} 行が同じ ｜ 10段階の段階ごとの行の割合 "
        + " / ".join(f"{j} {x * 100:.1f}%" for j, x in enumerate(gshare)))
    log(f"[確かめ] 組の大きさ（収益がある行）: 判断日 {len(size):,}日・中央 {size.median():.0f}件・10件以上の組 "
        f"{_s(float((size >= 10).mean()))}・20件以上 {_s(float((size >= 20).mean()))}")

    # ---- 探索（G10T） ----
    sub, no_label = R.search_frame(rl, cutoff)
    sig = {**R.search_sig(cols_w, sub, args.trials), "grades": 10, "exp": PREFIX}
    log(f"[探索] 期間 〜{cutoff.date()}（run_tuning.py と同じ打ち切り）: 組の行 {len(sub):,}件・{sub['Date'].nunique()}日"
        f"（層別に使うラベルが無くて外した {no_label}件）")
    prod_cv = rec5.get("_cv") or {}
    same_rows = (prod_cv.get("n_rows") == int(len(sub))
                 and prod_cv.get("train_to") == str(pd.to_datetime(sub["Date"]).max().date()))
    log(f"[確かめ] G5 の木の形（本番の ltr5_params.json。{rec5.get('_source', '出どころの記録なし')}）: 探索の行 "
        f"{prod_cv.get('n_rows', '?')}件・〜{prod_cv.get('train_to', '?')} → この回の探索の行と"
        f"{'同じ' if same_rows else '違う（本番の探索とこの回で探索の行が違う）'}")
    store = None
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("_sig") == sig:
            store = prev
            log("[探索] 保存済みの探索を読む（同じデータ・同じ列・同じ試行数）")
    folds_s = R.search_folds(sub, days)
    if not folds_s:
        raise SystemExit("探索の分割を作れない")
    if store is None:
        log(f"[探索] G10T（LambdaRank・組 = 直近{K}営業日・段階 0〜9・目的は検証分割の組の中の NDCG@{NDCG_K}（利得 10段階）・"
            f"{args.trials}試行・5分割）")
        rec10 = tune_g(sub, cols_w, folds_s, 10, args.trials)
        store = {"_sig": sig, "_cutoff": str(cutoff.date()), "G10T": rec10}
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2, default=str)
        log(f"[探索] G10T {rec10['_cv']['minutes']}分 / 分割平均 NDCG@{NDCG_K}（利得 10段階）{rec10['_cv']['ndcg']}"
            f"（最良 {rec10['_cv']['best_trial']}試行目）")
    params = {"G5": rec5["params"], "G10": rec5["params"], "G10T": store["G10T"]["params"], "D": rec_d["params"]}
    print("\n■ 木の形")
    print(f"  G5・G10（本番の ltr5_params.json）: {E._tree_line(params['G5'])}")
    print(f"  G10T（10段階で探索し直したもの。探索 〜{store['_cutoff']}）: {E._tree_line(params['G10T'])}")
    print(f"  D 今のランキング学習（日次。research/ltr_params.json）: {E._tree_line(params['D'])}")

    fit_chk = check_fit(rl_eval, cols_w, params["G5"])
    log(f"[確かめ] G5 の当てはめと本番の当てはめ（ltr.fit・ltr.predict）: 窓 {fit_chk['fold']}・訓練 {fit_chk['train']:,}行・"
        f"採点 {fit_chk['rows']:,}行・スコアの差の最大 {fit_chk['max_diff']:.1e}")
    if not fit_chk["same"]:
        raise SystemExit("この台本の5段階の当てはめが本番の当てはめと合わない")

    # ---- out-of-fold と集計 ----
    summary: Dict = {"info": info, "exit": xinfo, "check_cols": cm, "check_pools": cp, "check_same_day": sd,
                     "rule_info": {"r0": r0info, "r15": r15info}, "cutoff": str(cutoff.date()), "sig": sig,
                     "grade_share10": gshare, "prod_params_same_rows": bool(same_rows), "check_fit": fit_chk,
                     "params": {a: dict(p) for a, p in params.items()},
                     "search_cv": {k: v for k, v in store["G10T"]["_cv"].items()}, "top": args.top, "shifts": {}}
    rows = []
    for sh in shifts:
        o = oof_shift(rl_eval, daily, cols, cols_w, params, sh)
        o = R.attach_rule(o, rule0, ("r0", "pmin0"))
        o = R.attach_rule(o, rule15, ("r15", "pmin"))
        res = evaluate(o, days, args.top)
        report(sh, res, args.top)
        summary["shifts"][str(sh)] = res
        rows.append({"shift": sh, **res})

    rd = reading(rows)
    summary["reading"] = rd
    print("\n■ 読み方（回す前に決めたもの。採用の判定ではない）: (a) 組の1位の平均収益（窓の平均）と (b) 組の中の順位相関"
          "（今の売り方の収益）が、すべてのずらしで G5 より高いか")
    for a, r in rd.items():
        print(f"  {a}: " + " ／ ".join(
            f"ずらし{x['shift']}か月 (a) {x['ret'][1]:+.2f}% 対 {x['ret'][0]:+.2f}%・(b) {x['rho'][1]:+.3f} 対 {x['rho'][0]:+.3f}"
            f"（{'↑' if x['up'] else '↓' if x['down'] else '－'}）" for x in r["rows"]) + f" → {READ_JA[r['read']]}")
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
