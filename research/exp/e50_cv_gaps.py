#!/usr/bin/env python3
"""
実験50: 5分割 CV で、3モデル（lgbm / xgb / cat）の「大外れ」の傾向を調べる。

運用者の依頼（2026-09-26）:
  「oofではない5cvの期間で調査してほしいです。各3モデル（lgbm,xgb,cat）ごとに予測スコアが
  異様に低いけど、本当は正解だったもの、逆の「予測スコアが異様に高いけど、本当は負例」の
  ようなものを見つけて、どんな傾向があるか分析してほしいです。実際に特徴量としての傾向も
  知りたいですし、それらに共通してどの特徴が効いているのか、ギャップが大きいものは他の
  サンプルと比較してなにが違うのかなど（例えばですが、横方向にみた際に異様に欠損が多いなど。
  つまり特徴量の多くは欠損など。）」

■ データと分割（本番のパラメータ探索と同じ）
  データセット（lab.frame）をホールドアウトより前で切り（train_model.holdout_bounds）、
  tuning.year_folds(year_cap_date = 年×1日の正例数×時価総額帯で層別・日付単位, 5分割, 種0) で
  分ける。lgbm の探索（run_tuning --cv year_cap_date）と xgb / cat の探索（tuning_multi.tune）は
  同じ種・同じ方式なので、3モデルとも同じ5分割になる。各分割の検証側は「全期間から選んだ
  日付の集まり」で、連続した期間ではない（同じ日の銘柄は必ず同じ分割に入る）。

■ パラメータ（--tune）
  本番と同じ手順で 206列（features.DEFAULT_PRESET）で探索する。lgbm は tuning.tune（50試行）、
  xgb / cat は tuning_multi.tune（50試行）。本番の設定ファイル（research/lgbm_params.json・
  multi_params.json）には書かない。書くのは research/_data/oof/e50_params.json だけ。
  ログに「E50_PARAMS {...}」の1行で出す（手元の --params に渡す）。

■ CV スコア
  各分割を、残り4分割で学習したモデルで予測する（どのサンプルにも CV スコアが1つ付く）。
  学習は本番と同じ作り（lgbm: LGBMClassifier + scale_pos_weight、xgb / cat: tuning_multi.build）。
  分割ごとの PR-AUC が探索の記録（fold_scores）と一致するかも出す。
  SHAP（木の寄与。対数オッズ）を検証側で出す。

■ 群（分割の中のスコアの順位で決める。スコアの水準はモデル・分割で違うので順位で見る）
  FN_ext  正例なのに下位10%（スコアが異様に低い）
  FP_ext  負例なのに上位5%（スコアが異様に高い。運用で買う側）
  TP_top  正例で上位5%（高スコアで当たり）   … FP_ext の比べ相手（同じスコア帯）
  TN_bot  負例で下位10%（低スコアで正しい）   … FN_ext の比べ相手（同じスコア帯）

■ 比べるもの（どれも分割ごとに出し、5分割で向きがそろうかを見る）
  1. 行の欠損（206列のうち欠損の割合。特徴量の大区分ごとにも）
  2. 列ごとの違い: 欠損率の差と、値の順位の差（AUC = 群の値が相手より大きい確率。0.5 で差なし）
  3. SHAP の平均: どの列がスコアを押し下げた／押し上げたか。3モデル共通のもの
  4. ラベルのどの条件で落ちたか（到達 +1.2σ / 終盤 +0.5倍 / MA5>=MA20）と、境目までの距離
  5. 実収益（ret_o1_20）、時期・規模・上場年数の構成、3モデルの重なり

  ログには件数・割合・順位・効果量・集計した収益率だけを出す（Actions のログは公開される）。
  銘柄ごとの一覧は research/_data/oof/e50_cv.parquet に書き、--examples で手元でだけ見る
  （GITHUB_ACTIONS の上では --examples を拒む）。

  python3 research/exp/e50_cv_gaps.py --tune              # Actions（探索 + 分析）
  python3 research/exp/e50_cv_gaps.py --params p.json     # 手元（分析だけ）
  python3 research/exp/e50_cv_gaps.py --params p.json --examples 15
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
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))

import build_dataset as B  # noqa: E402
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "e50_params.json")
CV_PATH = os.path.join(OOF_DIR, "e50_cv.parquet")
N_SPLITS = 5
N_TRIALS = 50
SEED = 0                      # tuning.tune の既定・tuning_multi.SEED と同じ
ALGOS = ("lgbm", "xgb", "cat")
OUTCOME = lab.OUTCOME         # ret_o1_20

#: 群の境目（分割の中のスコアの順位）
LOW = 0.10                    # 下位10%
HIGH = 0.95                   # 上位5%（運用の lgbm95 と同じ帯）
GROUPS = ("FN_ext", "FP_ext", "TP_top", "TN_bot")
GROUP_JA = {"FN_ext": "正例なのに下位10%", "FP_ext": "負例なのに上位5%",
            "TP_top": "正例で上位5%", "TN_bot": "負例で下位10%"}

#: 大区分。画面の寄与と同じ括り（地合いだけ独立。predict_daily.GROUP_JA と同じ）
_G2JA = {g: ja for ja, groups, _ in FD.MACRO for g in groups}
_G2JA["market"] = "地合い（市場環境）"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# データと分割
# --------------------------------------------------------------------------- #

def tuning_frame() -> Tuple[pd.DataFrame, List[str], pd.Timestamp]:
    """本番の探索と同じ期間（ホールドアウトより前）。(Date, Code) の順（run_tuning と同じ）。"""
    df = lab.frame()
    cols = F.columns(F.DEFAULT_PRESET)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{F.DEFAULT_PRESET} の列がデータセットにありません: {missing[:5]}")
    df = df.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)
    sub = df[(d <= cutoff) & df["label"].notna()].reset_index(drop=True)
    return sub, cols, cutoff


def folds_of(sub: pd.DataFrame) -> np.ndarray:
    """各行の分割番号（0〜4）。本番の探索と同じ year_folds(year_cap_date)。"""
    folds = tuning.year_folds(sub, n_splits=N_SPLITS, seed=SEED,
                              by_year=True, by_cap=True, group_by_date=True)
    fold = np.full(len(sub), -1, dtype=int)
    for k, (_, va) in enumerate(folds):
        fold[va.index.to_numpy()] = k
    if (fold < 0).any():
        raise SystemExit(f"どの分割の検証にも入らない行が {int((fold < 0).sum())}件")
    return fold


# --------------------------------------------------------------------------- #
# 探索（本番と同じ手順・206列）
# --------------------------------------------------------------------------- #

def tune_all(sub: pd.DataFrame, cols: List[str], n_trials: int) -> Dict:
    out: Dict = {"_features_sig": F.signature(cols), "_n_features": len(cols),
                 "_n_trials": n_trials, "_rows": int(len(sub)),
                 "_train_to": str(pd.to_datetime(sub["Date"]).max().date())}
    t0 = time.time()
    lg = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS, seed=SEED,
                     embargo_days=EMBARGO_DAYS, scheme="year_cap_date", model="classifier",
                     verbose=True)
    out["lgbm"] = {"params": lg, "_cv": dict(tuning.LAST_CV),
                   "_minutes": round((time.time() - t0) / 60, 1)}
    log(f"lgbm 探索 {out['lgbm']['_minutes']}分 / CV PR-AUC {tuning.LAST_CV['mean_pr_auc']}")
    for algo in ("xgb", "cat"):
        t0 = time.time()
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS, seed=SEED)
        rec["_minutes"] = round((time.time() - t0) / 60, 1)
        out[algo] = rec
        log(f"{algo} 探索 {rec['_minutes']}分 / CV PR-AUC {rec['_cv']['mean_pr_auc']}")
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    return out


# --------------------------------------------------------------------------- #
# CV スコアと SHAP
# --------------------------------------------------------------------------- #

def _model(algo: str, rec: Dict, ytr: np.ndarray):
    if algo == "lgbm":
        import lightgbm as lgb
        # tuning._fit_one と同じ（探索の最良パラメータ + scale_pos_weight）
        return lgb.LGBMClassifier(**rec["params"],
                                  scale_pos_weight=tuning.scale_pos_weight(ytr))
    return TM.build(algo, rec["params"], ytr)


def _shap(algo: str, model, X: np.ndarray) -> np.ndarray:
    """列ごとの寄与（対数オッズ）。最後の列（基準値）は落とす。"""
    if algo == "lgbm":
        c = model.booster_.predict(X, pred_contrib=True)
    elif algo == "xgb":
        import xgboost as xgbm
        c = model.get_booster().predict(xgbm.DMatrix(X, missing=np.nan), pred_contribs=True)
    else:
        from catboost import Pool
        c = model.get_feature_importance(Pool(X), type="ShapValues")
    return np.asarray(c, dtype=float)[:, :-1]


def cv_scores(sub: pd.DataFrame, cols: List[str], fold: np.ndarray, params: Dict
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[float]]]:
    X = sub[cols].to_numpy(dtype=float)
    y = sub["label"].to_numpy(dtype=int)
    scores, shaps, fold_pr = {}, {}, {}
    for algo in ALGOS:
        s = np.full(len(sub), np.nan)
        sh = np.full((len(sub), len(cols)), np.nan)
        prs = []
        t0 = time.time()
        for k in range(N_SPLITS):
            va = fold == k
            m = _model(algo, params[algo], y[~va])
            m.fit(X[~va], y[~va])
            s[va] = m.predict_proba(X[va])[:, 1]
            sh[va] = _shap(algo, m, X[va])
            prs.append(float(average_precision_score(y[va], s[va])))
        scores[algo], shaps[algo], fold_pr[algo] = s, sh, prs
        log(f"{algo} CV {time.time()-t0:.0f}秒")
    return scores, shaps, fold_pr


def within_fold_rank(score: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """分割の中のスコアの順位（0〜1。大きいほど高スコア）。"""
    r = pd.Series(score).groupby(fold).rank(pct=True, method="average")
    return r.to_numpy()


def groups_of(label: np.ndarray, rank: np.ndarray,
              low: float = LOW, high: float = HIGH) -> Dict[str, np.ndarray]:
    pos = label == 1
    return {"FN_ext": pos & (rank <= low), "FP_ext": ~pos & (rank >= high),
            "TP_top": pos & (rank >= high), "TN_bot": ~pos & (rank <= low)}


# --------------------------------------------------------------------------- #
# 比べる道具（値そのものは出さない。欠損率・順位・効果量だけ）
# --------------------------------------------------------------------------- #

def rank_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a = b)。欠測は除く。片方が空なら nan。"""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([a, b])).rank(method="average").to_numpy()
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(u / (len(a) * len(b)))


def feature_diff(X: np.ndarray, a: np.ndarray, b: np.ndarray, cols: Sequence[str],
                 fold: Optional[np.ndarray] = None, min_n: int = 20) -> pd.DataFrame:
    """
    群 a と相手 b の列ごとの違い。miss_a/miss_b は欠損率、auc は値の順位の差
    （欠測を除く）。fold を渡すと、分割ごとの向き（auc>0.5 か）が全体と同じだった
    分割の数（agree / n_folds）も出す。
    """
    rows = []
    for j, c in enumerate(cols):
        xa, xb = X[a, j], X[b, j]
        na, nb = int((~np.isnan(xa)).sum()), int((~np.isnan(xb)).sum())
        auc = rank_auc(xa, xb) if min(na, nb) >= min_n else float("nan")
        rec = {"col": c, "miss_a": float(np.isnan(xa).mean()) if len(xa) else np.nan,
               "miss_b": float(np.isnan(xb).mean()) if len(xb) else np.nan,
               "n_a": na, "n_b": nb, "auc": auc}
        if fold is not None and not np.isnan(auc):
            agree, nf = 0, 0
            for k in np.unique(fold):
                fa, fb = X[a & (fold == k), j], X[b & (fold == k), j]
                ak = rank_auc(fa, fb)
                if np.isnan(ak) or min((~np.isnan(fa)).sum(), (~np.isnan(fb)).sum()) < 5:
                    continue
                nf += 1
                agree += int((ak > 0.5) == (auc > 0.5))
            rec.update(agree=agree, n_folds=nf)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["miss_diff"] = out["miss_a"] - out["miss_b"]
    out["effect"] = (out["auc"] - 0.5).abs()
    return out


def stratified_diff(X: np.ndarray, a: np.ndarray, b: np.ndarray, cols: Sequence[str],
                    strata: np.ndarray, min_n: int = 10) -> pd.DataFrame:
    """
    帯（strata）の中だけで比べた順位の差。帯ごとの AUC を、帯の小さいほうの群の件数で
    重み付けして平均する。帯の効き目（例: ボラの高低）を除いた違いを見るため。
    same = 全体と同じ向きだった帯の数 / 使えた帯の数。
    """
    rows = []
    for j, c in enumerate(cols):
        aucs, ws = [], []
        for s_ in np.unique(strata[~pd.isna(strata)]):
            m = strata == s_
            xa, xb = X[a & m, j], X[b & m, j]
            na, nb = int((~np.isnan(xa)).sum()), int((~np.isnan(xb)).sum())
            if min(na, nb) < min_n:
                continue
            aucs.append(rank_auc(xa, xb))
            ws.append(min(na, nb))
        if not aucs:
            rows.append({"col": c, "auc": np.nan, "same": 0, "n_strata": 0})
            continue
        auc = float(np.average(aucs, weights=ws))
        same = sum((x > 0.5) == (auc > 0.5) for x in aucs)
        rows.append({"col": c, "auc": auc, "same": same, "n_strata": len(aucs)})
    out = pd.DataFrame(rows)
    out["effect"] = (out["auc"] - 0.5).abs()
    return out


def col_family(cols: Sequence[str]) -> Dict[str, str]:
    """列 -> 大区分（画面の寄与と同じ名前）。"""
    g_of: Dict[str, str] = {}
    for g, cs in F.GROUPS.items():
        for c in cs:
            g_of.setdefault(c, g)
    return {c: _G2JA.get(g_of.get(c, ""), "その他") for c in cols}


def label_parts(sub: pd.DataFrame) -> pd.DataFrame:
    """
    ラベルの3条件（build_dataset.attach_rise_label と同じ式）と境目までの距離。
      reach  = 期間内の最高終値の上昇 / 到達に要る上昇（+1.2σ）。1 以上で到達
      end    = 終盤5日平均の水準 / 終盤に要る水準（到達しきい値の半分）。1 以上で合格
      trend  = 期間末に MA5 >= MA20
    """
    need, end_need = B.rise_thresholds(sub["vol_20d"], B.DEFAULT_RISE)
    need, end_need = need.to_numpy(), end_need.to_numpy()
    reach = sub["future_rise"].to_numpy(dtype=float) / need
    end = sub["end_level"].to_numpy(dtype=float) / end_need
    trend = sub["uptrend_end"].to_numpy(dtype=float) == 1.0
    ok = (reach >= 1.0) & (end >= 1.0) & trend
    return pd.DataFrame({"reach": reach, "end": end, "trend": trend, "rebuilt": ok})


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{x*100:.1f}%"


def _ja(c: str, width: int = 22) -> str:
    """列名と日本語名の頭（ログを読みやすくする。値は出さない）"""
    ja = FD.COL_JA.get(c, "")
    ja = ja.split("。")[0][:width]
    return f"{c}（{ja}）" if ja else c


def show_summary(sub, fold, scores, fold_pr, params, grp_by):
    y = sub["label"].to_numpy(dtype=int)
    print("\n■ 1. CV の成績と群の大きさ（5分割・検証側。分割ごとに出して平均 ± 標準偏差）")
    print(f"  サンプル {len(sub):,}件 / 正例率 {y.mean()*100:.2f}% / 期間 "
          f"{pd.to_datetime(sub['Date']).min().date()} 〜 {pd.to_datetime(sub['Date']).max().date()}")
    print("  分割ごとの件数: " + " / ".join(
        f"{int((fold == k).sum()):,}件(正例 {int(y[fold == k].sum())})" for k in range(N_SPLITS)))
    for algo in ALGOS:
        rec = params[algo]
        tuned = (rec.get("_cv") or {}).get("fold_scores")
        prs = fold_pr[algo]
        rocs = [roc_auc_score(y[fold == k], scores[algo][fold == k]) for k in range(N_SPLITS)]
        g = grp_by[algo]
        prec = []
        for k in range(N_SPLITS):
            m = fold == k
            top = g["TP_top"][m].sum() + g["FP_ext"][m].sum()
            prec.append(g["TP_top"][m].sum() / top if top else np.nan)
        low_rate = [g["FN_ext"][fold == k].sum() / (g["FN_ext"][fold == k].sum()
                                                    + g["TN_bot"][fold == k].sum())
                    for k in range(N_SPLITS)]
        print(f"\n  [{algo}] PR-AUC {np.mean(prs):.4f} ± {np.std(prs):.4f} / "
              f"ROC-AUC {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
        print("    分割ごとの PR-AUC: " + " / ".join(f"{v:.4f}" for v in prs))
        if tuned:
            same = all(abs(a - b) < 5e-4 for a, b in zip(prs, tuned))
            print("    探索の記録     : " + " / ".join(f"{v:.4f}" for v in tuned)
                  + ("（一致）" if same else "（不一致）"))
        print(f"    上位5% の的中率 {np.nanmean(prec)*100:.1f}%（分割ごと "
              + " / ".join(_pct(v) for v in prec) + "）")
        print(f"    下位10% の正例率 {np.mean(low_rate)*100:.1f}%（分割ごと "
              + " / ".join(_pct(v) for v in low_rate) + "）")
        print("    群の件数: " + " / ".join(f"{GROUP_JA[k]} {int(g[k].sum())}" for k in GROUPS))


def show_overlap(grp_by):
    print("\n■ 2. 3モデルの重なり（同じサンプルを大外れにしたか。Jaccard = 共通 / 合併）")
    for key in ("FN_ext", "FP_ext"):
        sets = {a: set(np.flatnonzero(grp_by[a][key])) for a in ALGOS}
        both = []
        for i, a in enumerate(ALGOS):
            for b in ALGOS[i + 1:]:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                both.append(f"{a}×{b} {inter / union:.2f}" if union else f"{a}×{b} —")
        all3 = set.intersection(*sets.values())
        any1 = set.union(*sets.values())
        print(f"  {GROUP_JA[key]}: " + " / ".join(both)
              + f" / 3モデルとも {len(all3)}件（どれか1つ {len(any1)}件）")


def row_missing(X: np.ndarray) -> np.ndarray:
    return np.isnan(X).mean(axis=1)


def show_missing(sub, X, cols, fold, grp_by, fam, ranks):
    print("\n■ 3. 行の欠損（206列のうち欠損の割合）。群ごとの中央値 / 平均 / 半分以上欠損の行の割合")
    mf = row_missing(X)
    print(f"  全体: 中央値 {np.median(mf)*100:.1f}% / 平均 {mf.mean()*100:.1f}% / "
          f"半分以上欠損 {np.mean(mf >= 0.5)*100:.1f}%")
    # 欠損の多さと、正例率・スコアの順位の関係（行は3モデル共通）
    tab = pd.DataFrame({"bin": pd.qcut(mf, 5, duplicates="drop"),
                        "y": sub["label"].to_numpy(dtype=int),
                        **{a: ranks[a] for a in ALGOS}})
    print("  欠損の多さ（5等分）ごとの正例率と、各モデルのスコアの平均順位（0.5 が真ん中）:")
    for b, g in tab.groupby("bin", observed=True):
        print(f"    欠損 {b.left*100:5.1f}〜{b.right*100:5.1f}%: {len(g):,}件 / 正例率 "
              f"{g['y'].mean()*100:.1f}% / 平均順位 "
              + " ".join(f"{a} {g[a].mean():.2f}" for a in ALGOS))
    for algo in ALGOS:
        g = grp_by[algo]
        print(f"  [{algo}] " + " / ".join(
            f"{GROUP_JA[k]} 中央値 {np.median(mf[g[k]])*100:.1f}%・平均 {mf[g[k]].mean()*100:.1f}%"
            f"・半分以上 {np.mean(mf[g[k]] >= 0.5)*100:.0f}%" for k in GROUPS if g[k].any()))
        # 分割ごとの向き: FN_ext の中央値 > TN_bot の中央値 か / FP_ext > TP_top か
        for a_key, b_key in (("FN_ext", "TN_bot"), ("FP_ext", "TP_top")):
            ks = [k for k in range(N_SPLITS)
                  if (g[a_key] & (fold == k)).any() and (g[b_key] & (fold == k)).any()]
            higher = sum(np.median(mf[g[a_key] & (fold == k)]) > np.median(mf[g[b_key] & (fold == k)])
                         for k in ks)
            print(f"      {GROUP_JA[a_key]} の欠損の中央値が {GROUP_JA[b_key]} より大きい分割: "
                  f"{higher}/{len(ks)}")
    # 大区分ごとの欠損率
    fams = sorted(set(fam.values()), key=lambda f: list(fam.values()).index(f))
    print("  大区分ごとの欠損率（群 − 比べ相手、pt）。|差| が 5pt 以上だけ")
    for algo in ALGOS:
        g = grp_by[algo]
        lines = []
        for f in fams:
            idx = [j for j, c in enumerate(cols) if fam[c] == f]
            m = np.isnan(X[:, idx]).mean(axis=1)
            for a_key, b_key in (("FN_ext", "TN_bot"), ("FP_ext", "TP_top")):
                if g[a_key].any() and g[b_key].any():
                    d = (m[g[a_key]].mean() - m[g[b_key]].mean()) * 100
                    if abs(d) >= 5:
                        lines.append(f"{f}: {GROUP_JA[a_key]} {m[g[a_key]].mean()*100:.0f}% vs "
                                     f"{GROUP_JA[b_key]} {m[g[b_key]].mean()*100:.0f}%（{d:+.0f}pt）")
        print(f"  [{algo}] " + ("\n          ".join(lines) if lines else "5pt 以上の差なし"))


def show_feature_diffs(X, cols, fold, grp_by, y, top: int = 10):
    print("\n■ 4. 列ごとの違い（AUC = 群の値が相手より大きい確率。0.5 で差なし。欠損は除いて比べる。"
          "向き = 5分割のうち全体と同じ向きだった分割の数）")
    pairs = (("FP_ext", "TP_top", "高スコア帯で外れた負例 vs 当たった正例"),
             ("FN_ext", "TN_bot", "低スコア帯で見逃した正例 vs 正しく外した負例"),
             ("FN_ext", "rest_pos", "見逃した正例 vs ほかの正例"),
             ("FP_ext", "rest_neg", "高スコアの負例 vs ほかの負例"))
    common: Dict[str, List[pd.DataFrame]] = {}
    for a_key, b_key, title in pairs:
        print(f"\n  ◆ {title}")
        for algo in ALGOS:
            g = grp_by[algo]
            a = g[a_key]
            if b_key == "rest_pos":
                b = (y == 1) & ~a
            elif b_key == "rest_neg":
                b = (y == 0) & ~a
            else:
                b = g[b_key]
            d = feature_diff(X, a, b, cols, fold)
            d["algo"] = algo
            common.setdefault(title, []).append(d)
            dd = d.dropna(subset=["auc"]).sort_values("effect", ascending=False).head(top)
            print(f"    [{algo}] {int(a.sum())}件 vs {int(b.sum())}件")
            for _, r in dd.iterrows():
                print(f"      AUC {r['auc']:.2f}（向き {int(r['agree'])}/{int(r['n_folds'])}）"
                      f" 欠損 {r['miss_a']*100:.0f}% vs {r['miss_b']*100:.0f}%  {_ja(r['col'])}")
            mm = d.assign(am=d["miss_diff"].abs()).sort_values("am", ascending=False).head(5)
            mm = mm[mm["am"] >= 0.05]
            if len(mm):
                print("      欠損率の差が大きい列: " + " / ".join(
                    f"{r['col']} {r['miss_a']*100:.0f}% vs {r['miss_b']*100:.0f}%"
                    for _, r in mm.iterrows()))
    print("\n  ◆ 3モデル共通（3モデルとも |AUC−0.5| ≥ 0.08・同じ向き・5分割のうち4以上で同じ向き）")
    for title, ds in common.items():
        m = ds[0][["col"]].copy()
        ok = np.ones(len(m), dtype=bool)
        sign = np.sign(ds[0]["auc"].to_numpy() - 0.5)
        for d in ds:
            ok &= (d["effect"].to_numpy() >= 0.08) & (np.sign(d["auc"].to_numpy() - 0.5) == sign) \
                & (d["agree"].fillna(0).to_numpy() >= 4)
        m["auc_mean"] = np.mean([d["auc"].to_numpy() for d in ds], axis=0)
        m = m[ok].assign(e=lambda t: (t["auc_mean"] - 0.5).abs()).sort_values("e", ascending=False)
        print(f"    {title}: {len(m)}列")
        for _, r in m.head(12).iterrows():
            print(f"      AUC {r['auc_mean']:.2f}  {_ja(r['col'])}")


def show_shap(shaps, cols, fold, grp_by, fam, top: int = 8):
    print("\n■ 5. SHAP（寄与。対数オッズ）。群の平均寄与から、分割全体の平均寄与を引いたもの。"
          "FN_ext で負に大きい列 = スコアを押し下げた理由、FP_ext で正に大きい列 = 押し上げた理由")
    per: Dict[str, Dict[str, pd.Series]] = {"FN_ext": {}, "FP_ext": {}}
    for algo in ALGOS:
        sh = shaps[algo]
        base = pd.DataFrame(sh, columns=cols).groupby(fold).mean()   # 分割ごとの平均
        g = grp_by[algo]
        for key, sign in (("FN_ext", 1), ("FP_ext", -1)):
            m = g[key]
            if not m.any():
                continue
            rel = pd.DataFrame(sh[m], columns=cols) - base.loc[fold[m]].to_numpy()
            mean = rel.mean()
            per[key][algo] = mean
            s = mean.sort_values(ascending=(sign == 1)).head(top)
            # 分割ごとの向き
            agree = {c: sum(np.sign(rel[fold[m] == k][c].mean()) == np.sign(mean[c])
                            for k in range(N_SPLITS) if (fold[m] == k).any()) for c in s.index}
            print(f"  [{algo}] {GROUP_JA[key]}（{int(m.sum())}件）: " + " / ".join(
                f"{c} {v:+.2f}（{agree[c]}/5）" for c, v in s.items()))
            fam_sum = mean.groupby(pd.Series(fam)).sum().sort_values(ascending=(sign == 1))
            print("      大区分の合計: " + " / ".join(f"{f} {v:+.2f}" for f, v in fam_sum.head(5).items()))
    print("  3モデル共通（3モデルとも同じ向きで上位15列に入るもの）")
    for key, sign in (("FN_ext", 1), ("FP_ext", -1)):
        if len(per[key]) < 3:
            continue
        tops = [set(v.sort_values(ascending=(sign == 1)).head(15).index) for v in per[key].values()]
        both = set.intersection(*tops)
        avg = pd.concat(per[key].values(), axis=1).mean(axis=1)
        ordered = avg[list(both)].sort_values(ascending=(sign == 1))
        print(f"    {GROUP_JA[key]}: " + (" / ".join(f"{_ja(c, 14)} {v:+.2f}"
                                                    for c, v in ordered.items()) or "なし"))


def show_label_parts(parts: pd.DataFrame, sub, grp_by):
    y = sub["label"].to_numpy(dtype=int)
    same = float((parts["rebuilt"].to_numpy() == (y == 1)).mean())
    print(f"\n■ 6. ラベルのどの条件で落ちたか・境目までの距離（ラベルの組み直しの一致率 {same*100:.2f}%）")
    reach, end, trend = parts["reach"].to_numpy(), parts["end"].to_numpy(), parts["trend"].to_numpy()
    for algo in ALGOS:
        g = grp_by[algo]
        fp, tn = g["FP_ext"], ~(y == 1) & ~g["FP_ext"]
        for name, m in (("負例なのに上位5%", fp), ("ほかの負例", tn)):
            n = int(m.sum())
            if not n:
                continue
            no_reach = np.mean(reach[m] < 1)
            faded = np.mean((reach[m] >= 1) & (end[m] < 1))
            no_trend = np.mean((reach[m] >= 1) & (end[m] >= 1) & ~trend[m])
            near = np.mean((reach[m] >= 0.8) & (reach[m] < 1))
            print(f"  [{algo}] {name}（{n}件）: 未到達 {no_reach*100:.0f}%（うち8割以上まで届いた "
                  f"{near*100:.0f}%）/ 到達したが終盤で失速 {faded*100:.0f}% / "
                  f"到達・終盤合格だが MA5<MA20 {no_trend*100:.0f}% / 到達率の中央値 "
                  f"{np.nanmedian(reach[m]):.2f}")
        fn, tp = g["FN_ext"], g["TP_top"]
        for name, m in (("正例なのに下位10%", fn), ("正例で上位5%", tp)):
            if m.any():
                print(f"  [{algo}] {name}（{int(m.sum())}件）: 到達率の中央値 {np.nanmedian(reach[m]):.2f} / "
                      f"到達率 1.0〜1.2（境目のすぐ上）{np.mean((reach[m] < 1.2))*100:.0f}% / "
                      f"終盤の水準の中央値 {np.nanmedian(end[m]):.2f}")


def show_outcomes(sub, grp_by):
    print(f"\n■ 7. 実収益（{OUTCOME}: 翌営業日の寄りで買い20営業日後の終値）の分布")
    r = sub[OUTCOME].to_numpy(dtype=float)
    for algo in ALGOS:
        g = grp_by[algo]
        print(f"  [{algo}] " + " / ".join(
            f"{GROUP_JA[k]}: 中央値 {np.nanmedian(r[g[k]])*100:+.1f}%・マイナス "
            f"{np.nanmean(r[g[k]] < 0)*100:.0f}%" for k in GROUPS if g[k].any()))


def vol_bins(sub) -> pd.Series:
    return pd.qcut(sub["vol_20d"], 5, duplicates="drop")


def show_vol_bands(sub, ranks, grp_by):
    """ボラ正規化ラベル（到達しきい値 = 1.2σ）とスコアの関係を、ボラの5等分で見る。"""
    print("\n■ 9. 自分のボラ（vol_20d。直近20営業日の日次リターンの標準偏差）の5等分ごと")
    print("  到達しきい値は 1.2σ × √20 なので、ボラが高いほど正例になるのに要る上昇が大きい")
    y = sub["label"].to_numpy(dtype=int)
    r = sub[OUTCOME].to_numpy(dtype=float)
    need, _ = B.rise_thresholds(sub["vol_20d"], B.DEFAULT_RISE)
    t = pd.DataFrame({"bin": vol_bins(sub), "y": y, "r": r, "need": need.to_numpy(),
                      **{f"rk_{a}": ranks[a] for a in ALGOS},
                      **{f"fn_{a}": grp_by[a]["FN_ext"] for a in ALGOS},
                      **{f"fp_{a}": grp_by[a]["FP_ext"] for a in ALGOS}})
    for b, g in t.groupby("bin", observed=True):
        pos = g["y"] == 1
        print(f"  ボラ {b.left:.2f}〜{b.right:.2f}%（必要な上昇の中央値 {g['need'].median()*100:.0f}%）: "
              f"{len(g):,}件 / 正例率 {g['y'].mean()*100:.1f}% / 正例の実収益の中央値 "
              f"{np.nanmedian(g.loc[pos, 'r'])*100:+.1f}% / 平均順位 "
              + " ".join(f"{a} {g[f'rk_{a}'].mean():.2f}" for a in ALGOS)
              + " / 正例なのに下位10% " + "・".join(str(int(g[f"fn_{a}"].sum())) for a in ALGOS)
              + " / 負例なのに上位5% " + "・".join(str(int(g[f"fp_{a}"].sum())) for a in ALGOS)
              + "（lgbm・xgb・cat）")


def show_years(sub, X, grp_by):
    print("\n■ 10. 年ごと（件数・正例率・行の欠損の中央値・上位5%に入った件数と的中率・正例なのに下位10%の件数）")
    d = pd.to_datetime(sub["Date"]).dt.year.to_numpy()
    y = sub["label"].to_numpy(dtype=int)
    mf = row_missing(X)
    for yr in np.unique(d):
        m = d == yr
        prec = []
        for a in ALGOS:
            g = grp_by[a]
            top = (g["TP_top"] | g["FP_ext"]) & m
            prec.append(f"{a} {int(top.sum())}件 {g['TP_top'][m].sum() / top.sum() * 100:.0f}%"
                        if top.sum() else f"{a} —")
        print(f"  {yr}: {int(m.sum()):,}件 / 正例率 {y[m].mean()*100:.1f}% / 欠損 {np.median(mf[m])*100:.1f}% / "
              f"上位5% " + " ".join(prec) + " / 正例なのに下位10% "
              + "・".join(str(int(grp_by[a]["FN_ext"][m].sum())) for a in ALGOS))


def show_within_vol(sub, X, cols, grp_by, top: int = 8):
    print("\n■ 11. 同じボラの帯の中で比べた違い（ボラの効き目を除く。帯ごとの AUC を件数で重み付け。"
          "両方の群が10件以上ある帯だけ使い、2帯以上で比べられた列だけ出す。"
          "（a/b）= 使えた b 帯のうち全体と同じ向きだった数）")
    strata = vol_bins(sub).astype(str).to_numpy()
    pairs = (("FP_ext", "TP_top", "高スコア帯で外れた負例 vs 当たった正例"),
             ("FN_ext", "TN_bot", "低スコア帯で見逃した正例 vs 正しく外した負例"))
    for a_key, b_key, title in pairs:
        print(f"  ◆ {title}")
        per = []
        for algo in ALGOS:
            g = grp_by[algo]
            d = stratified_diff(X, g[a_key], g[b_key], cols, strata)
            d.loc[d["n_strata"] < 2, ["auc", "effect"]] = np.nan   # 1帯だけの比べは読まない
            per.append(d.set_index("col"))
            dd = d.dropna(subset=["auc"]).sort_values("effect", ascending=False).head(top)
            print(f"    [{algo}] " + " / ".join(
                f"{r['col']} {r['auc']:.2f}（{int(r['same'])}/{int(r['n_strata'])}）"
                for _, r in dd.iterrows()))
        both = pd.concat([p_["auc"] for p_ in per], axis=1)
        sgn = np.sign(both - 0.5)
        ok = (both.sub(0.5).abs() >= 0.07).all(axis=1) & (sgn.nunique(axis=1) == 1)
        common = both[ok].mean(axis=1).sort_values(key=lambda v: -(v - 0.5).abs())
        print("    3モデル共通（|AUC−0.5| ≥ 0.07・同じ向き）: " + (" / ".join(
            f"{_ja(c, 14)} {v:.2f}" for c, v in common.head(10).items()) or "なし"))


def show_composition(sub, fold, grp_by):
    print("\n■ 8. 構成（群の中の割合 − 全体の割合、pt）。年・時価総額帯・上場年数")
    d = pd.to_datetime(sub["Date"])
    keys = {"年": d.dt.year.astype(str),
            "時価総額帯": sub["cap_band"].astype("Int64").astype(str),
            # 欠損は「データの初日（2016-10）より前から上場で、初日から5年たつまで」= 2021-10 まで
            "上場年数": pd.cut(sub["listing_years"], [-0.01, 1, 2, 3, 4.99, 5.01],
                            labels=["〜1年", "1〜2年", "2〜3年", "3〜5年", "5年以上"])
                         .astype(str).replace("nan", "不明（2021-10まで・古くからの上場）")}
    for algo in ALGOS:
        g = grp_by[algo]
        print(f"  [{algo}]")
        for name, key in keys.items():
            allp = key.value_counts(normalize=True)
            parts = []
            for gk in ("FN_ext", "FP_ext"):
                if not g[gk].any():
                    continue
                gp = key[g[gk]].value_counts(normalize=True).reindex(allp.index).fillna(0)
                diff = ((gp - allp) * 100).sort_values()
                big = diff[diff.abs() >= 4]
                parts.append(f"{GROUP_JA[gk]}: " + (", ".join(
                    f"{i} {v:+.0f}pt" for i, v in big.items()) or "4pt 以上の偏りなし"))
            print(f"    {name}: " + " | ".join(parts))
    # 月の偏り（3モデル共通の大外れで、多い月）。上位5%は月で件数が違うので、
    # 「その月に3モデルとも上位5%に入った件数」と、そのうちの負例の割合も出す
    ym = d.dt.to_period("M").astype(str)
    top3 = np.logical_and.reduce([grp_by[a]["TP_top"] | grp_by[a]["FP_ext"] for a in ALGOS])
    for gk in ("FN_ext", "FP_ext"):
        m = np.logical_and.reduce([grp_by[a][gk] for a in ALGOS])
        if not m.any():
            continue
        exp = ym.value_counts(normalize=True) * m.sum()
        got = ym[m].value_counts()
        ex = (got - exp.reindex(got.index)).sort_values(ascending=False).head(6)
        extra = ""
        print(f"  3モデルとも{GROUP_JA[gk]}（{int(m.sum())}件）が多い月: " + " / ".join(
            f"{i} {int(got[i])}件（期待 {exp[i]:.1f}"
            + (f"・その月に3モデルとも上位5% {int(top3[ym == i].sum())}件" if gk == "FP_ext" else "")
            + "）" for i in ex.index) + extra)


# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tune", action="store_true", help="206列で3モデルを探索してから分析する")
    ap.add_argument("--params", help="探索済みのパラメータ（e50_params.json の形）")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--examples", type=int, default=0,
                    help="銘柄ごとの大外れを N件ずつ出す（手元だけ。Actions では拒む）")
    args = ap.parse_args(argv)
    if args.examples and os.environ.get("GITHUB_ACTIONS") == "true":
        raise SystemExit("--examples は銘柄ごとの値を出すので、公開ログ（Actions）では使えません")

    sub, cols, cutoff = tuning_frame()
    log(f"探索期間 〜{cutoff.date()} / {len(sub):,}件 / 正例率 {sub['label'].mean()*100:.2f}% / "
        f"{F.DEFAULT_PRESET}（{len(cols)}列・指紋 {F.signature(cols)}）")
    if args.tune:
        params = tune_all(sub, cols, args.trials)
    elif args.params:
        with open(args.params, encoding="utf-8") as fh:
            params = json.load(fh)
    else:
        raise SystemExit("--tune か --params を指定してください")
    if params.get("_features_sig") not in (None, F.signature(cols)):
        raise SystemExit(f"パラメータは別の列で探索したもの（{params.get('_features_sig')}）")
    print("E50_PARAMS " + json.dumps(
        {a: params[a] for a in ALGOS} | {k: v for k, v in params.items() if k.startswith("_")},
        ensure_ascii=False, default=str, separators=(",", ":")), flush=True)

    fold = folds_of(sub)
    scores, shaps, fold_pr = cv_scores(sub, cols, fold, params)
    y = sub["label"].to_numpy(dtype=int)
    ranks = {a: within_fold_rank(scores[a], fold) for a in ALGOS}
    grp_by = {a: groups_of(y, ranks[a]) for a in ALGOS}
    X = sub[cols].to_numpy(dtype=float)
    fam = col_family(cols)
    parts = label_parts(sub)

    show_summary(sub, fold, scores, fold_pr, params, grp_by)
    show_overlap(grp_by)
    show_missing(sub, X, cols, fold, grp_by, fam, ranks)
    show_feature_diffs(X, cols, fold, grp_by, y)
    show_shap(shaps, cols, fold, grp_by, fam)
    show_label_parts(parts, sub, grp_by)
    show_outcomes(sub, grp_by)
    show_composition(sub, fold, grp_by)
    show_vol_bands(sub, ranks, grp_by)
    show_years(sub, X, grp_by)
    show_within_vol(sub, X, cols, grp_by)

    # 銘柄ごとの一覧（手元で読む。ログには出さない）
    out = sub[["Date", "Code", "label", OUTCOME, "cap_band", "listing_years"]].copy()
    out["fold"] = fold
    out["miss_frac"] = row_missing(X)
    out = pd.concat([out, parts[["reach", "end", "trend"]]], axis=1)
    for a in ALGOS:
        out[f"score_{a}"] = scores[a]
        out[f"rank_{a}"] = ranks[a]
        for k in GROUPS:
            out[f"{k}_{a}"] = grp_by[a][k]
        top = np.argsort(-np.abs(shaps[a]), axis=1)[:, :5]
        out[f"shap_top_{a}"] = [",".join(f"{cols[j]}:{shaps[a][i, j]:+.2f}" for j in row)
                                for i, row in enumerate(top)]
    os.makedirs(OOF_DIR, exist_ok=True)
    out.to_parquet(CV_PATH, index=False)
    for a in ALGOS:
        np.save(os.path.join(OOF_DIR, f"e50_shap_{a}.npy"), shaps[a].astype(np.float32))
    log(f"銘柄ごとの一覧を {CV_PATH} に書いた（{len(out):,}行）")

    if args.examples:
        show_examples(out, args.examples)
    return 0


def show_examples(out: pd.DataFrame, n: int) -> None:
    """手元だけ。大外れを銘柄ごとに出す（3モデルとも大外れのものを先に）。"""
    for key in ("FN_ext", "FP_ext"):
        m = np.logical_and.reduce([out[f"{key}_{a}"].to_numpy() for a in ALGOS])
        rows = out[m].copy()
        rows["avg_rank"] = rows[[f"rank_{a}" for a in ALGOS]].mean(axis=1)
        rows = rows.sort_values("avg_rank", ascending=(key == "FN_ext")).head(n)
        print(f"\n■ 例: 3モデルとも{GROUP_JA[key]}（{int(m.sum())}件のうち {len(rows)}件）")
        for _, r in rows.iterrows():
            print(f"  {str(r['Date'])[:10]} {r['Code']} 順位 "
                  + "/".join(f"{r[f'rank_{a}']*100:.0f}" for a in ALGOS)
                  + f" 実収益 {r[OUTCOME]*100:+.1f}% 到達率 {r['reach']:.2f} 終盤 {r['end']:.2f} "
                  f"欠損 {r['miss_frac']*100:.0f}% | lgbm寄与 {r['shap_top_lgbm']}")


if __name__ == "__main__":
    raise SystemExit(main())
