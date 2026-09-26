#!/usr/bin/env python3
"""
実験55: ラベルの σ を長い窓（60日 / 120日）にしたときの、探索し直し・3モデル・運用規則での確かめ。

実験54（docs/MODEL_ADOPTION_RULES.md §16）で σ60 のラベル（L1）が、本番の分類器・本番の木の形の
まま、選定の実収益を3つの切り方すべてで上げた。ラベルの変更は本番の全経路に及ぶので、§7 の手順
どおり (1) 探索し直し（B2）でも同じ向きか、(2) xgb / cat でも同じ向きか、(3) 運用規則（3モデル
90以上・lgbm95、live_track の取引: TOP_K 2 / SLOTS 3 / 利確 20% / 20日）でも良いか、を測る。
同じ向きにさらに進めた σ120 も腕に入れる。運用者の承認（2026-09-26「実験55いいですね」）。

腕（ラベル）: L0 1.2σ20（現行） / L1 1.2σ60 / L4 1.2σ120。k は 1.2 のまま（実験54 で σ60 は
k=1.2 で正例率が変わらなかった。σ120 の正例率はここで出す）。ラベルの引き直しは実験54 と同じ。

学習（本番の206列）
  lgbm B1  本番の木の形（実験54 と同じ）
  lgbm B2  そのラベルで探索し直した木の形（tuning.tune: 年×時価総額帯で層別・日付単位の5分割・
           種0・50試行・PR-AUC。本番の週次探索と同じ）
  xgb / cat  本番の木の形（追加モデルの探索し直しはしない。3モデル規則の確認用）

評価（3切り方 × 種3つ。全腕でラベルが確定している行。正例率・PR-AUC は現行ラベル L0 で測る）
  1. lgbm B1 / B2: 実験54 と同じ実収益（閾値ルール・発火数を L0-B1 にそろえる・窓の中の上位10%・
     −10% 未満・ボラの帯）と、参考の PR-AUC（自分のラベルで測ったものも並べる）
  2. xgb / cat: 窓の中の上位10%（件数そろえ）の平均収益と −10% 未満、L0 との窓ごとの差
  3. 運用規則: lgbm95 / 3モデル90 の選定（ops_rule）と、live_track の取引（ab_oof.rule_trades）
     を B1 の組（lgbm B1 + xgb + cat）と B2 の組（lgbm B2 + xgb + cat）で。
     --consensus 90,95,98 で 3モデル合議の百分位を変えたときも出す（運用者の問い 2026-09-26
     「σ60でも、3モデル合議にするとどうなりますか？（3モデルとも95,98で）」）

  python3 research/exp/e55_label_sigma_full.py [--trials 50] [--shifts 0,2,4] [--seeds 3]
                                               [--arms L0,L1,L4] [--algos lgbm,xgb,cat] [--skip-tune]
                                               [--consensus 90,95,98]
  結果は research/_data/oof/e55_*。探索の結果は e55_params.json（本番の設定には書かない）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import ops_rule as OR  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402
import e54_label_sigma as E54  # noqa: E402
from e25_auc_noise import average  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
PARAMS_PATH = os.path.join(OOF_DIR, "e55_params.json")
OUTCOME = lab.OUTCOME
N_SPLITS = 5
N_TRIALS = 50
SEED = 0
ARMS: Dict[str, str] = {"L0": "sigma20", "L1": "sigma60", "L4": "sigma120"}
LABELS = {"L0": "L0 1.2σ20（現行）", "L1": "L1 1.2σ60", "L4": "L4 1.2σ120"}
FITS = ("B1", "B2")
FIT_JA = {"B1": "本番の木", "B2": "探索し直し"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 探索し直し（B2）
# --------------------------------------------------------------------------- #

def tune_label(df: pd.DataFrame, ycol: str, cols: List[str], n_trials: int) -> Dict:
    """そのラベルで本番と同じ探索（ホールドアウトより前・year_cap_date 5分割・PR-AUC）。"""
    d = pd.to_datetime(df["Date"])
    cutoff, _, _ = holdout_bounds(d, HOLDOUT_MONTHS, EMBARGO_DAYS)
    sub = df[(d <= cutoff) & df[ycol].notna()].copy()
    sub["label"] = sub[ycol]
    sub = sub.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    t0 = time.time()
    params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS, seed=SEED,
                         embargo_days=EMBARGO_DAYS, scheme="year_cap_date", model="classifier",
                         verbose=True)
    return {"params": params, "_cv": dict(tuning.LAST_CV), "_rows": int(len(sub)),
            "_pos_rate": round(float(sub["label"].mean()), 4), "_train_to": str(cutoff.date()),
            "_minutes": round((time.time() - t0) / 60, 1)}


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

def oof(df: pd.DataFrame, cols: List[str], arm: str, fit: str, algo: str, shift: int, seeds,
        params: Dict) -> pd.DataFrame:
    """腕（ラベル）× 木の形 × モデル × 切り方の out-of-fold（種の平均）。保存済みなら読む。"""
    folds = E41.folds_for(df["Date"], shift)
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"e55_{arm}_{fit}_{algo}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds(algo, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  腕{arm} {fit} {algo} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def rule_trades_at(oofs: Dict[str, pd.DataFrame], bars: pd.DataFrame, bar_days, agree: float) -> pd.DataFrame:
    """ab_oof.rule_trades と同じ取引を、3モデル合議の百分位 agree を変えて取る。"""
    base = None
    for a in L.BOOST:
        o = oofs[a][["Code", "Date", "fold", "label", "score"]].rename(columns={"score": f"s_{a}"})
        o[f"p_{a}"] = L.pct_of(o[f"s_{a}"].to_numpy(), o[f"s_{a}"].to_numpy())
        base = o if base is None else base.merge(o.drop(columns=["label", "fold"]),
                                                 on=["Code", "Date"], how="inner")
    base["score"] = base["s_lgbm"]
    _, _, picks = L.decide(base, agree=agree)
    pf = L.forward(bars, picks)
    sim = L.simulate(pf, bar_days)
    return sim[sim["taken"] == L.TAKEN].copy()


def with_base_label(o: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """評価用: label を現行ラベル（y_L0）に置き換え、vol_20d を付ける。自分のラベルは own_label。"""
    m = o.rename(columns={"label": "own_label"}).merge(
        df[["Code", "Date", "y_L0", "vol_20d"]], on=["Code", "Date"], how="left")
    return m.rename(columns={"y_L0": "label"})


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験55: ラベルの σ（60日 / 120日）の確かめ")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--algos", default=",".join(L.BOOST))
    ap.add_argument("--skip-tune", action="store_true", help="e55_params.json を読むだけ")
    ap.add_argument("--consensus", default="90", help="3モデル合議の百分位（例 90,95,98）")
    args = ap.parse_args(argv)
    pcts = [float(x) for x in args.consensus.split(",") if x.strip()]
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad or "L0" not in arms:
        raise SystemExit(f"腕は {list(ARMS)} から、L0 を含めて指定してください: {arms}")
    if "lgbm" not in algos:
        raise SystemExit("lgbm は必須です")

    cols = F.columns(F.DEFAULT_PRESET)
    df = lab.frame()
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print(f"実験55 ラベルの σ（60日 / 120日）を探索し直し・3モデル・運用規則で確かめる"
          f"（{len(df):,}件 / {F.DEFAULT_PRESET} {len(cols)}列 / 種{len(seeds)}つ / ずらし {shifts}か月 / "
          f"モデル {algos} / 探索 {args.trials}試行）")
    print("=" * 78)

    # ---- ラベル ----
    sg = E54.sigmas_from_bars(E54.load_bars())
    df = df.merge(sg, on=["Code", "Date"], how="left")
    if int(((df["sigma20"] - df["vol_20d"]).abs() > 1e-6).sum()):
        raise SystemExit("bars から作った σ20 がデータセットの vol_20d と一致しません")
    y0 = E54.relabel(df, E54.need_from_sigma(df["sigma20"], E54.K0))
    if int((y0 != df["label"]).sum()):
        raise SystemExit("引き直した L0 が本番のラベルと違います")
    for a in arms:
        df[f"y_{a}"] = E54.relabel(df, E54.need_from_sigma(df[ARMS[a]], E54.K0))
    keep = df[[f"y_{a}" for a in arms]].notna().all(axis=1)
    print(f"\n■ ラベル（全腕で確定している {int(keep.sum()):,}件 / 落ちた {int((~keep).sum()):,}件）")
    df = df[keep].reset_index(drop=True)
    print(f"  {'腕':<20}{'正例率':>8}  ボラの帯ごとの正例率（低→高）  L0 と違う行  正例の実収益 中央値 / 平均")
    for a in arms:
        pv = E54.pos_by_vol(df[f"y_{a}"], df["vol_20d"])
        ch = float((df[f"y_{a}"] != df["y_L0"]).mean())
        pos = df.loc[df[f"y_{a}"] == 1, OUTCOME]
        print(f"  {LABELS[a]:<20}{df[f'y_{a}'].mean()*100:>7.2f}%  " + " / ".join(f"{v*100:.1f}%" for v in pv)
              + f"  {ch*100:>5.1f}%  {pos.median()*100:+.1f}% / {pos.mean()*100:+.1f}%")

    # ---- 探索し直し ----
    if args.skip_tune:
        if not os.path.exists(PARAMS_PATH):
            raise SystemExit(f"--skip-tune には {PARAMS_PATH} が要ります")
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            tuned = json.load(fh)
        missing = [a for a in arms if a not in tuned]
        if missing or tuned.get("_features_sig") != F.signature(cols):
            raise SystemExit(f"e55_params.json に足りない腕 {missing} か、列の指紋が違います")
        print(f"\n[tune] {PARAMS_PATH} を読む")
    else:
        tuned = {"_features_sig": F.signature(cols), "_n_features": len(cols), "_n_trials": args.trials}
        for a in arms:
            print(f"\n[tune] 腕{a}（{LABELS[a]}）を探索し直す")
            tuned[a] = tune_label(df, f"y_{a}", cols, args.trials)
            log(f"腕{a} 探索 {tuned[a]['_minutes']}分 / CV PR-AUC {tuned[a]['_cv']['mean_pr_auc']}"
                f"（正例率 {tuned[a]['_pos_rate']*100:.2f}%）")
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(tuned, fh, ensure_ascii=False, indent=2, default=str)
    print("\n■ 探索し直しの結果（5分割の CV。PR-AUC は自分のラベルで測ったもので、腕の間では比べない）")
    for a in arms:
        p = tuned[a]["params"]
        cv = tuned[a]["_cv"]
        print(f"  {LABELS[a]:<20} PR-AUC {cv['mean_pr_auc']:.4f} ± {cv.get('std', 0):.4f} / ROC {cv['mean_roc_auc']:.4f} ｜ "
              + " / ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in p.items() if k in ("learning_rate", "num_leaves", "min_child_samples",
                                                          "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")))

    # ---- out-of-fold と評価 ----
    prod = {a: E27.prod_params(a)["params"] for a in algos}
    edges, rows, trades = [], [], []
    bars = bar_days = None
    if set(L.BOOST) <= set(algos):
        bars = L.load_bars(lab.DATA_DIR, start=pd.Timestamp("2021-01-01"))
        bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    for sh in shifts:
        res: Dict[str, Dict[str, pd.DataFrame]] = {}
        for a in arms:
            d = df.copy()
            d["label"] = d[f"y_{a}"]
            res[a] = {}
            for fit in FITS:
                params = prod["lgbm"] if fit == "B1" else tuned[a]["params"]
                res[a][f"lgbm_{fit}"] = oof(d, cols, a, fit, "lgbm", sh, seeds, params)
            for algo in algos:
                if algo != "lgbm":
                    res[a][algo] = oof(d, cols, a, "B1", algo, sh, seeds, prod[algo])
        ev = {a: {k: with_base_label(o, df) for k, o in m.items()} for a, m in res.items()}
        print(f"\n■ ずらし{sh}か月")
        for fit in FITS:
            key = f"lgbm_{fit}"
            print(f"  ◇ lgbm {fit}（{FIT_JA[fit]}）")
            for pct in (95, 90):
                print(f"  ◆ 上位{100-pct}%（前の窓の分布の閾値）: 件数 / 平均 / 全体との差 / "
                      f"窓ごとの差 ± SE（上の窓 / 最悪） / −10%未満")
                for a in arms:
                    o = ev[a][key]
                    e = lab.threshold_edge(o, pct=pct, outcome=OUTCOME)
                    pf = E52._per_fold(o, pct, within=False)
                    bd = float((pf["bad"] * pf["n"]).sum() / pf["n"].sum()) if pf["n"].sum() else np.nan
                    se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
                    print(f"    {LABELS[a]:<20}{e['thr_n']:>6}件 {e['thr_end']:>+7.2f}% {e['thr_lift']:>+7.2f}pt "
                          f" {e['thr_fold_mean']:>+6.2f} ± {se:.2f}（{e['thr_folds_won']}/{e['thr_folds']} / "
                          f"{e['thr_worst']:+.2f}） {bd*100:>5.1f}%")
                    edges.append({"shift": sh, "arm": a, "fit": key, "kind": f"thr{pct}", "bad": bd, **e})
                counts = E53.threshold_counts(ev["L0"]["lgbm_B1"], pct)
                base = E53.matched_picks(ev["L0"]["lgbm_B1"], counts)
                print(f"  ◆ 発火数を L0-B1 の上位{100-pct}% にそろえる: 件数 / 平均収益 / −10%未満 / 正例率(L0) ｜ L0-B1 との差（窓ごと、pt）")
                for a in arms:
                    cur = E53.matched_picks(ev[a][key], counts)
                    n = int(cur["n"].sum())
                    ret = float((cur["ret"] * cur["n"]).sum() / n) if n else np.nan
                    bd = float((cur["bad"] * cur["n"]).sum() / n) if n else np.nan
                    pos = float((cur["pos"] * cur["n"]).sum() / n) if n else np.nan
                    line = f"    {LABELS[a]:<20}{n:>6}件 {ret*100:>+7.2f}% {bd*100:>5.1f}% {pos*100:>4.0f}%"
                    if not (a == "L0" and fit == "B1"):
                        line += E53._diff_line(base, cur)
                    print(line)
                    edges.append({"shift": sh, "arm": a, "fit": key, "kind": f"match{pct}", "bad": bd,
                                  "thr_n": n, "thr_end": ret * 100, "thr_fold_mean": cur["ret"].mean() * 100})
        print("  ◆ 窓の中の上位10%（件数をそろえる）: 平均収益 / −10%未満 / 正例率(L0) ｜ L0 の同じモデルとの差（窓ごと、pt）")
        keys = [f"lgbm_{f}" for f in FITS] + [a for a in algos if a != "lgbm"]
        for key in keys:
            base = E52._per_fold(ev["L0"][key], 90, within=True)
            for a in arms:
                cur = E52._per_fold(ev[a][key], 90, within=True)
                line = (f"    {key:<8}{LABELS[a]:<20}{cur['ret'].mean()*100:>+6.2f}% {cur['bad'].mean()*100:>5.1f}% "
                        f"{cur['pos'].mean()*100:>4.0f}%")
                if a != "L0":
                    line += E53._diff_line(base, cur)
                print(line)
                edges.append({"shift": sh, "arm": a, "fit": key, "kind": "top10_within", "bad": cur["bad"].mean(),
                              "thr_fold_mean": cur["ret"].mean() * 100, "thr_n": int(cur["n"].sum())})
        print("  ◆ 日付内（発火8件以上の日）: 1位 / 上位2件 / 順位相関 ｜ PR-AUC（L0 のラベル） / PR-AUC（自分のラベル）")
        for key in keys:
            for a in arms:
                o = ev[a][key]
                w = E52.within_date_metrics(o)
                m = E52.label_metrics(o)
                own = E52.label_metrics(o.assign(label=o["own_label"]))
                print(f"    {key:<8}{LABELS[a]:<20}{w['top1']:>+7.2f}% {w['top2']:>+7.2f}% {w['rho']:>+6.3f} ｜ "
                      f"{m['pr']:.4f} / {own['pr']:.4f}")
                rows.append({"shift": sh, "arm": a, "fit": key, **w, "pr_L0": m["pr"], "pr_own": own["pr"]})
        print("  ◆ 上位10% の中身（lgbm B2、ボラの帯ごと: 割合 / 平均収益 / 正例率(L0)）")
        for a in arms:
            t = E52.picks_by_vol(ev[a]["lgbm_B2"], 90)
            if len(t):
                print(f"    {LABELS[a]:<20}" + " | ".join(
                    f"{i} {r['share']*100:.0f}% {r['ret']*100:+.1f}% {r['pos']*100:.0f}%" for i, r in t.iterrows()))
        if bars is not None:
            for fit in FITS:
                print(f"  ◆ 運用規則（lgbm {fit} + xgb + cat）。3モデル合議の百分位 {pcts}")
                for a in arms:
                    oofs = {"lgbm": ev[a][f"lgbm_{fit}"], "xgb": ev[a]["xgb"], "cat": ev[a]["cat"]}
                    OR.report(oofs, label=f"    {LABELS[a]} / {FIT_JA[fit]}")
                    lg = oofs["lgbm"]
                    n_days = sum(1 for d in bar_days if lg["Date"].min() <= d <= lg["Date"].max())
                    for pct in pcts:
                        if pct != 90.0:
                            c = OR.consensus(oofs, pct, models=L.BOOST)
                            print(OR.fmt(f"3モデル {pct:g}以上", c))
                        t = rule_trades_at(oofs, bars, bar_days, agree=pct)
                        st = L.trade_stats(t, n_days)
                        per = t.groupby("fold")["ret"].mean() if len(t) else pd.Series(dtype=float)
                        print(f"      live_track（合議 {pct:g}）: 取引 {st['n']} / 1取引 {st['mean']:+.2f}% / 勝率 {st['win']:.0f}% / "
                              f"月あたり {st['monthly']:+.2f}% / 窓の平均の最小 {(per.min() if len(per) else np.nan):+.2f}% / "
                              f"平均が正の窓 {int((per > 0).sum())}/{len(per)}")
                        trades.append({"shift": sh, "arm": a, "fit": fit, "agree": pct,
                                       **{k: st[k] for k in ("n", "mean", "win", "monthly")},
                                       "worst_fold": float(per.min()) if len(per) else np.nan,
                                       "pos_folds": int((per > 0).sum()), "folds": int(len(per))})

    ed = pd.DataFrame(edges)
    ed.to_csv(os.path.join(OOF_DIR, "e55_edges.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e55_within_date.csv"), index=False)
    if trades:
        pd.DataFrame(trades).to_csv(os.path.join(OOF_DIR, "e55_trades.csv"), index=False)
    if len(shifts) > 1 and len(ed):
        print(f"\n■ 切り方{len(shifts)}通りをまとめて（切り方ごと / 平均）")
        for kind, ja in (("thr95", "上位5%（閾値）全体との差 pt"), ("thr90", "上位10%（閾値）全体との差 pt"),
                         ("match90", "発火数を L0-B1 の上位10% にそろえた平均収益"),
                         ("top10_within", "窓の中の上位10%（件数そろえ）の平均収益")):
            print(f"  {ja} / −10%未満")
            for fit in sorted(ed.loc[ed["kind"] == kind, "fit"].unique()):
                for a in arms:
                    g = ed[(ed["arm"] == a) & (ed["kind"] == kind) & (ed["fit"] == fit)]
                    col = "thr_lift" if kind.startswith("thr") else "thr_fold_mean"
                    unit = "" if kind.startswith("thr") else "%"
                    print(f"    {fit:<8}{LABELS[a]:<20}" + " / ".join(f"{v:+.2f}{unit}" for v in g[col])
                          + f"（平均 {g[col].mean():+.2f}{unit}）/ {g['bad'].mean()*100:.1f}%")
        if trades:
            t = pd.DataFrame(trades)
            for pct in pcts:
                print(f"  live_track の取引（3モデル合議 {pct:g}。切り方ごとの 取引数 1取引の平均 / 月あたり / 平均が正の窓）")
                for fit in FITS:
                    for a in arms:
                        g = t[(t["arm"] == a) & (t["fit"] == fit) & (t["agree"] == pct)]
                        print(f"    {fit:<8}{LABELS[a]:<20}" + " / ".join(
                            f"{int(r['n'])}件 {r['mean']:+.2f}% {r['monthly']:+.2f}% {r['pos_folds']}/{r['folds']}"
                            for _, r in g.iterrows())
                            + f"（取引 {int(g['n'].sum())} / 1取引の平均 {g['mean'].mean():+.2f}% / 月 {g['monthly'].mean():+.2f}%）")
    log(f"記録: {OOF_DIR}/e55_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
