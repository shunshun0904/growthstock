#!/usr/bin/env python3
"""
実験53: 収益を目的にした腕（R 回帰 / Q 日付内順位の回帰）に、腕ごとのパラメータ探索を与える。

実験52（docs/MODEL_ADOPTION_RULES.md §15）は木の形を本番の分類器の探索結果から流用したので、
R / Q は不利な条件の見積もりだった。ここでは 212列（実験51 の6列込み。実験52 で Q が一番
良かった列）で、R と Q それぞれに専用の探索をしてから、同じ 3切り方 × 種3つで比べる。

探索（本番の探索と同じ枠組み）
  - 期間はホールドアウトより前（train_model.holdout_bounds）、年×時価総額帯で層別・日付単位の
    5分割・種0（tuning.year_folds の year_cap_date。実験50 と同じ分割）
  - 探索空間は tuning.tune と同じ（学習率は木200本に合わせた範囲、葉 7〜127、min_child 10〜300、
    subsample / colsample / reg_alpha / reg_lambda）。木は200本固定
  - 目的は「分割の中の上位10%（件数は分割ごとに同じ割合）の平均収益（ret_o1_20）」の5分割平均。
    採否の物差し（§15）と同じにする。PR-AUC は目的が違うので使わない。日付内の順位相関と
    −10% 未満の割合は記録だけ
  - 探索の結果は research/_data/oof/e53_params.json（本番の設定には書かない）

評価の腕（すべて 212列・LightGBM・木200本）
  C   分類（本番の木の形）                … 実験52 と同じ。比べる基準
  R0  収益の回帰（本番の木の形）           … 実験52 と同じ
  Q0  日付内順位の回帰（本番の木の形）     … 実験52 と同じ
  R1  収益の回帰（R の探索結果）
  Q1  日付内順位の回帰（Q の探索結果）

評価（実収益。実験52 の物差しに「発火数を C にそろえる」を足す）
  1. 閾値ルール（前の窓のスコア分布の上位5% / 10% を超えたら買う。運用の形）
  2. **発火数を C にそろえる**: 窓ごとに C が閾値ルールで買った件数と同じ件数を、その腕のスコア
     の上から取る。実験52 で Q が閾値ルールで勝ったのは発火が C の 1/3〜1/4 だったからか、
     順位が良いからかを切り分ける
  3. 窓の中の上位10%（件数をそろえる）と C との差（窓ごと）
  4. 日付内: 1位・上位2件の平均収益、順位相関。参考に PR-AUC など
  5. 上位10% の中身（ボラの帯）

  python3 research/exp/e53_objective_tuning.py [--trials 50] [--shifts 0,2,4] [--seeds 3]
                                               [--arms C,R0,Q0,R1,Q1] [--skip-tune]
  --skip-tune は e53_params.json が既にあるときだけ使える。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e52_return_objective as E52  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "e53_params.json")
OUTCOME = lab.OUTCOME
PRESET = "all_plus_prog_listing_vol"
N_SPLITS = 5
N_TRIALS = 50
SEED = 0
#: 探索の目的: 分割の中の上位 (100−TOP_PCT)% の平均収益
TOP_PCT = 90
#: 探索する腕（実験52 の腕の記号）
TUNED = ("R", "Q")
#: 評価の腕 → (実験52 の腕, 探索結果を使うか)
ARMS: Dict[str, Tuple[str, bool]] = {"C": ("C", False), "R0": ("R", False), "Q0": ("Q", False),
                                     "R1": ("R", True), "Q1": ("Q", True)}
LABELS = {"C": "C 分類（本番の木）", "R0": "R0 回帰（本番の木）", "Q0": "Q0 日付内順位（本番の木）",
          "R1": "R1 回帰（専用の探索）", "Q1": "Q1 日付内順位（専用の探索）"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 探索
# --------------------------------------------------------------------------- #

def tuning_frame(cols: List[str]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """本番の探索と同じ期間（ホールドアウトより前）。(Date, Code) の順。収益が無い行は外す。"""
    df = lab.frame()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{PRESET} の列がデータセットにありません: {missing[:5]}。"
                         "research/build_dataset.py を回し直してください")
    df = df.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)
    sub = df[(d <= cutoff) & df["label"].notna() & df[OUTCOME].notna()].reset_index(drop=True)
    sub["Date"] = pd.to_datetime(sub["Date"])
    return sub, cutoff


def top_mean(score: np.ndarray, ret: np.ndarray, pct: float = TOP_PCT) -> Dict:
    """スコア上位 (100−pct)% の平均収益・−10% 未満の割合・件数（1つの分割の中で）。"""
    thr = np.percentile(score, pct)
    sel = ret[score > thr]
    if not len(sel):
        return {"ret": np.nan, "bad": np.nan, "n": 0}
    return {"ret": float(np.nanmean(sel)), "bad": float(np.nanmean(sel < -0.10)), "n": int(len(sel))}


def date_rho(dates: pd.Series, score: np.ndarray, ret: np.ndarray, min_n: int = 5) -> float:
    """日付内の順位相関（Spearman）の平均。min_n 件以上の日だけ。"""
    from scipy.stats import spearmanr
    f = pd.DataFrame({"d": pd.Series(dates).to_numpy(), "s": score, "r": ret})
    out = []
    for _, g in f.groupby("d"):
        if len(g) >= min_n and g["r"].notna().sum() >= min_n:
            out.append(spearmanr(g["s"], g["r"], nan_policy="omit").correlation)
    return float(np.nanmean(out)) if out else np.nan


def search_params(trial, n_trees: int = tuning.SEARCH_N_ESTIMATORS) -> Dict:
    """tuning.tune と同じ探索空間（目的関数は腕が決めるので入れない）。"""
    lo, hi = tuning.lr_range(n_trees)
    return {
        "boosting_type": "gbdt",
        "n_estimators": n_trees,
        "learning_rate": trial.suggest_float("learning_rate", lo, hi, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": -1,
    }


def cv_objective(arm: str, params: Dict, folds, cols: List[str]) -> Dict:
    """5分割それぞれで学習→検証し、上位10% の平均収益（分割平均）などを返す。"""
    rets, bads, rhos = [], [], []
    for tr, va in folds:
        s = E52.fit_predict(arm, tr, va, cols, SEED, params)
        r = va[OUTCOME].to_numpy(dtype=float)
        t = top_mean(s, r)
        rets.append(t["ret"])
        bads.append(t["bad"])
        rhos.append(date_rho(va["Date"], s, r))
    return {"top_ret": float(np.mean(rets)), "top_ret_folds": [round(x, 5) for x in rets],
            "top_ret_std": float(np.std(rets)), "bad": float(np.mean(bads)),
            "rho": float(np.nanmean(rhos))}


def tune_arm(arm: str, sub: pd.DataFrame, cols: List[str], n_trials: int) -> Dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = tuning.year_folds(sub, n_splits=N_SPLITS, seed=SEED, by_year=True, by_cap=True,
                              group_by_date=True)
    if not folds:
        raise SystemExit("分割を作れない")

    def objective(trial):
        p = search_params(trial)
        m = cv_objective(arm, p, folds, cols)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["top_ret"]

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {"boosting_type": "gbdt", "n_estimators": tuning.SEARCH_N_ESTIMATORS, "subsample_freq": 1,
            "n_jobs": -1, **study.best_params}
    at = study.best_trial.user_attrs
    # 本番の木の形を同じ物差しで測る（探索が本当に改善したかの基準）
    base = cv_objective(arm, E52.tree_params(SEED), folds, cols)
    rec = {"params": best,
           "_cv": {"objective": f"分割の中の上位{100-TOP_PCT}% の平均収益", "n_splits": len(folds),
                   "n_trials": n_trials, "best_trial": int(study.best_trial.number),
                   "top_ret": round(at["top_ret"], 5), "top_ret_std": round(at["top_ret_std"], 5),
                   "top_ret_folds": at["top_ret_folds"], "bad": round(at["bad"], 4),
                   "rho": round(at["rho"], 4),
                   "prod_tree": {k: round(v, 5) if isinstance(v, float) else v for k, v in base.items()},
                   "fold_pos_rate": [round(float(v["label"].mean()), 4) for _, v in folds],
                   "trials": [{"n": t.number, "top_ret": round(t.value, 5),
                               "rho": round(t.user_attrs.get("rho", np.nan), 4)}
                              for t in study.trials]},
           "_minutes": round((time.time() - t0) / 60, 1)}
    return rec


def tune_all(sub: pd.DataFrame, cols: List[str], n_trials: int) -> Dict:
    out: Dict = {"_features_sig": F.signature(cols), "_n_features": len(cols),
                 "_n_trials": n_trials, "_rows": int(len(sub)),
                 "_train_to": str(sub["Date"].max().date()), "_objective_pct": TOP_PCT}
    for arm in TUNED:
        rec = tune_arm(arm, sub, cols, n_trials)
        out[arm] = rec
        cv = rec["_cv"]
        log(f"腕{arm} 探索 {rec['_minutes']}分 / 上位10% の平均収益 {cv['top_ret']*100:+.2f}%"
            f"（本番の木 {cv['prod_tree']['top_ret']*100:+.2f}%）/ 日付内 ρ {cv['rho']:+.3f}"
            f"（本番の木 {cv['prod_tree']['rho']:+.3f}）/ −10%未満 {cv['bad']*100:.1f}%")
        print("  木の形: " + " / ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                        for k, v in rec["params"].items()
                                        if k in ("learning_rate", "num_leaves", "min_child_samples",
                                                 "subsample", "colsample_bytree", "reg_alpha",
                                                 "reg_lambda")))
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    return out


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def threshold_counts(o_c: pd.DataFrame, pct: float) -> Dict[int, int]:
    """C が閾値ルール（前の窓の分布の上位 (100−pct)%）で買った件数（窓ごと）。"""
    out = {}
    for f in sorted(o_c["fold"].unique()):
        ref = o_c.loc[o_c["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        cur = o_c[o_c["fold"] == f]
        out[int(f)] = int((cur["score"] > np.percentile(ref, pct)).sum())
    return out


def matched_picks(o: pd.DataFrame, counts: Dict[int, int]) -> pd.DataFrame:
    """窓ごとに counts 件だけ、その腕のスコアの上から取る（発火数を C にそろえた選定）。"""
    rows = []
    for f, n in counts.items():
        cur = o[o["fold"] == f]
        if n <= 0 or not len(cur):
            continue
        sel = cur.sort_values("score", ascending=False, kind="mergesort").head(n)
        r = pd.to_numeric(sel[OUTCOME], errors="coerce")
        rows.append({"fold": int(f), "n": int(len(sel)), "ret": float(r.mean()),
                     "bad": float((r < -0.10).mean()), "pos": float(sel["label"].mean())})
    return pd.DataFrame(rows, columns=["fold", "n", "ret", "bad", "pos"])


def _diff_line(base: pd.DataFrame, cur: pd.DataFrame) -> str:
    m = base.merge(cur, on="fold", suffixes=("_c", "_a"))
    if not len(m):
        return ""
    d = (m["ret_a"] - m["ret_c"]).to_numpy() * 100
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return (f" ｜ {d.mean():+.2f} ± {se:.2f}pt（上の窓 {(d > 0).sum()}/{len(d)}）: "
            + " ".join(f"{x:+.1f}" for x in d))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験53: R / Q に専用のパラメータ探索")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--skip-tune", action="store_true", help="e53_params.json を読むだけ")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]
    bad_arms = [a for a in arms if a not in ARMS]
    if bad_arms:
        raise SystemExit(f"知らない腕: {bad_arms}（使えるもの {list(ARMS)}）")

    cols = F.columns(PRESET)
    tag = f"_{F.signature(cols)}"
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print(f"実験53 収益を目的にした腕に専用の探索（{PRESET} {len(cols)}列（指紋 {F.signature(cols)}）"
          f" / 種{len(seeds)}つ / ずらし {shifts}か月 / 探索 {args.trials}試行）")
    print("=" * 78)

    # ---- 探索 ----
    if args.skip_tune:
        if not os.path.exists(PARAMS_PATH):
            raise SystemExit(f"--skip-tune には {PARAMS_PATH} が要ります")
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            tuned = json.load(fh)
        if tuned.get("_features_sig") != F.signature(cols):
            raise SystemExit("e53_params.json の列の指紋が今の212列と違います。探索し直してください")
        print(f"[tune] {PARAMS_PATH} を読む（{tuned['_n_trials']}試行）")
    else:
        sub, cutoff = tuning_frame(cols)
        print(f"[tune] 期間 〜{cutoff.date()} / {len(sub):,}件 / 正例率 {sub['label'].mean()*100:.2f}% / "
              f"目的: 分割の中の上位{100-TOP_PCT}% の平均収益（{N_SPLITS}分割・種{SEED}）")
        tuned = tune_all(sub, cols, args.trials)
    print("\n■ 探索の結果（5分割の平均。本番の木は同じ分割で測り直したもの）")
    print(f"  {'腕':<4}{'上位10% 平均収益':>16}{'本番の木':>10}{'日付内ρ':>9}{'本番の木':>9}{'−10%未満':>9}{'本番の木':>9}")
    for arm in TUNED:
        cv = tuned[arm]["_cv"]
        pt = cv["prod_tree"]
        print(f"  {arm:<4}{cv['top_ret']*100:>+15.2f}% {pt['top_ret']*100:>+9.2f}% "
              f"{cv['rho']:>+8.3f} {pt['rho']:>+8.3f} {cv['bad']*100:>8.1f}% {pt['bad']*100:>8.1f}%")

    # ---- 評価 ----
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    edges, rows = [], []
    for sh in shifts:
        res = {}
        for a in arms:
            base_arm, use_tuned = ARMS[a]
            params = tuned[base_arm]["params"] if use_tuned else None
            res[a] = E52.oof_arm(df, cols, base_arm, sh, seeds, tag, params=params, name=a,
                                 prefix="e53")
        print(f"\n■ ずらし{sh}か月")
        for pct in (95, 90):
            print(f"  ◆ 上位{100-pct}%（前の窓の分布の閾値）: 件数 / 平均 / 全体との差 / "
                  f"窓ごとの差 ± SE（上の窓 / 最悪） / −10%未満")
            for a in arms:
                e = lab.threshold_edge(res[a], pct=pct, outcome=OUTCOME)
                pf = E52._per_fold(res[a], pct, within=False)
                bad = float((pf["bad"] * pf["n"]).sum() / pf["n"].sum()) if pf["n"].sum() else np.nan
                se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
                print(f"    {LABELS[a]:<26}{e['thr_n']:>6}件 {e['thr_end']:>+7.2f}% {e['thr_lift']:>+7.2f}pt "
                      f" {e['thr_fold_mean']:>+6.2f} ± {se:.2f}（{e['thr_folds_won']}/{e['thr_folds']} / "
                      f"{e['thr_worst']:+.2f}） {bad*100:>5.1f}%")
                edges.append({"shift": sh, "arm": a, "kind": f"thr{pct}", "bad": bad, **e})
            if "C" not in arms:
                continue
            counts = threshold_counts(res["C"], pct)
            base = matched_picks(res["C"], counts)
            print(f"  ◆ 発火数を C にそろえる（C の上位{100-pct}% と同じ件数を各窓で上から取る）: "
                  f"件数 / 平均収益 / −10%未満 / 正例率 ｜ C との差（窓ごと、pt）")
            for a in arms:
                cur = matched_picks(res[a], counts)
                n = int(cur["n"].sum())
                ret = float((cur["ret"] * cur["n"]).sum() / n) if n else np.nan
                bad = float((cur["bad"] * cur["n"]).sum() / n) if n else np.nan
                pos = float((cur["pos"] * cur["n"]).sum() / n) if n else np.nan
                line = f"    {LABELS[a]:<26}{n:>6}件 {ret*100:>+7.2f}% {bad*100:>5.1f}% {pos*100:>4.0f}%"
                if a != "C":
                    line += _diff_line(base, cur)
                print(line)
                edges.append({"shift": sh, "arm": a, "kind": f"match{pct}", "bad": bad,
                              "thr_n": n, "thr_end": ret * 100, "thr_fold_mean": cur["ret"].mean() * 100})
        print("  ◆ 窓の中の上位10%（件数をそろえる）: 平均収益 / −10%未満 / 正例率 ｜ C との差（窓ごと、pt）")
        base = E52._per_fold(res["C"], 90, within=True) if "C" in arms else None
        for a in arms:
            cur = E52._per_fold(res[a], 90, within=True)
            line = (f"    {LABELS[a]:<26}{cur['ret'].mean()*100:>+6.2f}% {cur['bad'].mean()*100:>5.1f}% "
                    f"{cur['pos'].mean()*100:>4.0f}%")
            if base is not None and a != "C":
                line += _diff_line(base, cur)
            print(line)
            edges.append({"shift": sh, "arm": a, "kind": "top10_within", "bad": cur["bad"].mean(),
                          "thr_fold_mean": cur["ret"].mean() * 100, "thr_n": int(cur["n"].sum())})
        print(f"  ◆ 日付内（発火{E52.MIN_BREAKS}件以上の日）: 1位の平均収益 / 上位2件の平均 / 順位相関 / 日数"
              " ｜ 参考 PR-AUC / ROC-AUC / 日内AUC")
        for a in arms:
            w = E52.within_date_metrics(res[a])
            m = E52.label_metrics(res[a])
            print(f"    {LABELS[a]:<26}{w['top1']:>+7.2f}% {w['top2']:>+7.2f}% {w['rho']:>+6.3f} "
                  f"{w['days']:>5}日 ｜ {m['pr']:.4f} {m['roc']:.4f} {m['day_auc']:.4f}")
            rows.append({"shift": sh, "arm": a, **w, **m})
        print("  ◆ 上位10% の中身（ボラの帯ごと: 割合 / 平均収益 / 正例率）")
        for a in arms:
            t = E52.picks_by_vol(res[a], 90)
            if len(t):
                print(f"    {LABELS[a]:<26}" + " | ".join(
                    f"{i} {r['share']*100:.0f}% {r['ret']*100:+.1f}% {r['pos']*100:.0f}%"
                    for i, r in t.iterrows()))

    ed = pd.DataFrame(edges)
    ed.to_csv(os.path.join(OOF_DIR, "e53_edges.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e53_within_date.csv"), index=False)
    if len(shifts) > 1 and len(ed):
        print(f"\n■ 切り方{len(shifts)}通りをまとめて（切り方ごと / 平均）")
        for kind, ja in (("thr95", "上位5%（閾値）全体との差 pt"), ("thr90", "上位10%（閾値）全体との差 pt")):
            print(f"  {ja} / −10%未満")
            for a in arms:
                g = ed[(ed["arm"] == a) & (ed["kind"] == kind)]
                print(f"    {LABELS[a]:<26}" + " / ".join(f"{v:+.2f}" for v in g["thr_lift"])
                      + f"（平均 {g['thr_lift'].mean():+.2f}）/ {g['bad'].mean()*100:.1f}%")
        for kind, ja in (("match95", "発火数を C の上位5% にそろえた平均収益"),
                         ("match90", "発火数を C の上位10% にそろえた平均収益"),
                         ("top10_within", "窓の中の上位10%（件数をそろえる）の平均収益")):
            g = ed[ed["kind"] == kind]
            if not len(g):
                continue
            print(f"  {ja} / −10%未満")
            for a in arms:
                h = g[g["arm"] == a]
                print(f"    {LABELS[a]:<26}" + " / ".join(f"{v:+.2f}%" for v in h["thr_fold_mean"])
                      + f"（平均 {h['thr_fold_mean'].mean():+.2f}%）/ {h['bad'].mean()*100:.1f}%")
        r = pd.DataFrame(rows)
        print("  日付内の1位の平均収益（切り方の平均）: " + " / ".join(
            f"{LABELS[a]} {r[r['arm'] == a]['top1'].mean():+.2f}%" for a in arms))
    log(f"記録: {OOF_DIR}/e53_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
