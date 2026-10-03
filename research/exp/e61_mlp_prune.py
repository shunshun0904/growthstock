#!/usr/bin/env python3
"""
実験61: MLP の「効かない列」を削って学習し直すとどうなるか。

運用者の問い（2026-10-03、実験60 の報告を受けて）「MLP の効かない列を削って再学習したらどうなりますか」

「効かない列」の決め方は1つではないので、3通りを腕にする。P（全239列）と同じ手順（本番の週次実行と同じ
探索 Optuna 50試行 × 5分割 → 本番と同じ out-of-fold → 32窓）で比べ、採否は §7 に準じて「OOF と窓の過半で
上回る」。前処理は本番の MLP と同じ v1。木・ロジスティック回帰は触らない。

腕（列の選び方。探索に使う期間の行と、本番の MLP だけで決める。先の期間の値は使わない）
  P  全239列（いまの本番）
  A  過大評価の列だけ外す: 本番 MLP の寄与（積分勾配）の上位30列のうち、単独の ROC-AUC が 0.48〜0.52 の列
     （実験60 §4 の13列に相当）
  B  単独で効かない列を全部外す: 単独の ROC-AUC が 0.48〜0.52 の列（探索に使う期間で 182列 → 残り 57列）。
     決算の変化・水準・連続改善の列はほぼ全部ここに入る（単独では効かず、組み合わせで効いている列）
  C  MLP が使っていない列を外す: 本番 MLP の列ごとの置換の崩れ（1 − 順位相関）が小さい下位半分（約120列）
  At / Ct  A / C と同じ決め方を、本番 MLP の代わりに「探索に使う期間だけで学習した MLP」（P のパラメータ・種42）
     で行う。本番 MLP は評価する期間（OOF・窓の検証側）も学習に含むので、A / C の列の選び方には先読みが混じる
     （ラベルは使わないが、モデルが評価期間を見ている）。At / Ct はそれが無い。採用の判断は Ct で行う
結果（2026-10-03、docs/MODEL_ADOPTION_RULES.md §23）: C は 32窓で PR 25/32・ROC 29/32 と上がって見えたが、先読みを
除いた Ct では PR 17/32（+0.0013）・ROC 19/32（+0.0030）で P と区別がつかない → 採用しない（239列のまま）。
評価は実験58 と同じ作り（e25_auc_noise.metrics、ab_oof.auc_by_window）。種は窓では3つ（42, 7, 123）、
本番と同じ OOF は 42。記録は research/_data/oof/e61_*（列の一覧は e61_cols_{腕}.json）。
本番の設定（research/multi_params.json、features の preset）には書かない。

使い方
    python3 research/exp/e61_mlp_prune.py
    python3 research/exp/e61_mlp_prune.py --n-trials 1 --shifts 0 --seeds 1 --release-dir /tmp/q   # 試運転
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import models as M  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e60_model_tendency as E60  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGO = "mlp"
PREP = "v1"
N_SPLITS = 5
ARMS = ("P", "A", "B", "C", "At", "Ct")
LABELS = {"P": "P 全239列", "A": "A 過大評価の列を外す", "B": "B 単独で効かない列を全部外す",
          "C": "C MLP が使っていない下位半分を外す",
          "At": "At 過大評価の列を外す（探索期間だけで学習した MLP で選ぶ）",
          "Ct": "Ct MLP が使っていない下位半分を外す（探索期間だけで学習した MLP で選ぶ）"}
#: 本番 MLP ではなく、探索に使う期間だけで学習した MLP で列を選ぶ腕 → 選び方の元になる腕
TUNE_SELECT = {"At": "A", "Ct": "C"}
WEAK_AUC = 0.02
TOP_N = 30
PROD_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- #
# 列の選び方
# ---------------------------------------------------------------- #

def weak_columns(cols, X: np.ndarray, y: np.ndarray, thr: float = WEAK_AUC):
    """単独の ROC-AUC が 0.5 ± thr の列（NaN を除いて計算。出せない列は弱いとみなさない）。"""
    auc = E60.univariate_auc(X, y)
    return [c for c, a in zip(cols, auc) if np.isfinite(a) and abs(a - 0.5) < thr], auc


def select(cols, X: np.ndarray, y: np.ndarray, model, *, top_n: int = TOP_N, seed: int = 0):
    """
    腕ごとの列。戻り値 {腕: 列のリスト}（順は本番の列の順のまま）と、判断に使った表。
    model は本番の MLP（前処理つき Pipeline）。
    """
    weak, auc = weak_columns(cols, X, y)
    weak_set = set(weak)
    # A: 本番 MLP の寄与（|積分勾配| の平均）の上位 top_n のうち弱い列
    attr = E60.attributions(ALGO, model, X, cols)
    share = pd.Series(np.abs(attr).mean(axis=0), index=cols)
    top = share.sort_values(ascending=False).head(top_n).index
    over = [c for c in cols if c in set(top) and c in weak_set]
    # C: 列ごとの置換の崩れが小さい下位半分
    base = E60.raw_score(ALGO, model, X)
    perm = E60.permutation_reliance(ALGO, model, X, base, {c: [j] for j, c in enumerate(cols)}, seed=seed)
    ps = pd.Series(perm)
    cut = ps.median()
    unused = [c for c in cols if ps[c] <= cut]
    out = {"P": list(cols),
           "A": [c for c in cols if c not in set(over)],
           "B": [c for c in cols if c not in weak_set],
           "C": [c for c in cols if c not in set(unused)]}
    table = pd.DataFrame({"col": cols, "uni_auc": auc, "share": share.to_numpy(), "perm": ps.reindex(cols).to_numpy(),
                          "weak": [c in weak_set for c in cols], "over": [c in set(over) for c in cols],
                          "unused": [c in set(unused) for c in cols]})
    return out, table


def fit_mlp(df: pd.DataFrame, cols: list, params: dict, seed: int = PROD_SEED):
    """探索に使う期間の行だけで MLP を1つ学習する（列の選び方に先読みを入れないため）。本番と同じ組み方・前処理。"""
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    TM.SEED = seed                              # build() が random_state に使う（e41.oof_folds と同じ）
    try:
        return M.fit(ALGO, X, y, cols, params=params, prep=PREP)
    finally:
        TM.SEED = 0


# ---------------------------------------------------------------- #
# 探索と OOF（実験58 と同じ作り。腕 = 列の組）
# ---------------------------------------------------------------- #

def tune_arm(arm: str, tune_df: pd.DataFrame, cols: list, n_trials: int) -> dict:
    path = os.path.join(OOF_DIR, f"e61_params_{arm}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("_cv", {}).get("n_trials") == n_trials and rec.get("_cv", {}).get("n_cols") == len(cols):
            log(f"  [{arm}] 探索済みを読む（CV PR-AUC {rec['_cv'].get('mean_pr_auc')}）")
            return rec
    t0 = time.time()
    TM.STUDY_DB = os.path.join(OOF_DIR, f"e61_optuna_{arm}.db")
    rec = TM.tune(ALGO, tune_df, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False)
    rec["_cv"]["seconds"] = round(time.time() - t0)
    rec["_cv"]["n_cols"] = len(cols)
    rec["_cv"]["preprocess"] = PREP
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    log(f"  [{arm}] 探索 {time.time() - t0:.0f}秒 / {len(cols)}列 / CV PR-AUC {rec['_cv']['mean_pr_auc']:.4f} "
        f"(±{rec['_cv']['std']:.4f}) / ROC {rec['_cv']['mean_roc_auc']:.4f} / {rec['params']}")
    return rec


def oof_arm(arm: str, df: pd.DataFrame, cols: list, params: dict, shift: int, seeds) -> pd.DataFrame:
    folds = E41.folds_for(df["Date"], shift)
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"e61_{arm}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        o = E41.oof_folds(ALGO, df, cols, params, sd, folds)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"    {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time() - t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def pair_line(name: str, a: np.ndarray, b: np.ndarray, fmt: str = "{:.4f}") -> str:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return (f"  {name:<22}{fmt.format(a.mean()):>9}{fmt.format(b.mean()):>9}"
            f"{('+' if d.mean() >= 0 else '') + fmt.format(d.mean()):>10}{fmt.format(se):>9}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験61: MLP の効かない列を削って学習し直す")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--release-dir", default=lab.DATA_DIR, help="本番 MLP（<dir>/mlp_model.joblib）の場所")
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(lab.DATA_DIR), "model"))
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]
    if "P" not in arms or any(a not in ARMS for a in arms):
        raise SystemExit(f"腕は {list(ARMS)} から、P を含めて: {arms}")
    os.makedirs(OOF_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print("実験61 MLP の「効かない列」を削って学習し直す（本番と同じ探索 → OOF → 32窓）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 / 前処理 {PREP} / 探索 {args.n_trials}試行 × {N_SPLITS}分割 / "
          f"ずらし {shifts}か月 / 種 {seeds}")
    for a in arms:
        print(f"  {LABELS[a]}")
    print("=" * 78)

    raw = (pd.read_parquet(os.path.join(lab.DATA_DIR, "dataset.parquet"))
           .sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True))
    dates = pd.to_datetime(raw["Date"])
    cutoff, _, _ = T.holdout_bounds(dates, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    tune_df = raw[dates <= cutoff].reset_index(drop=True)
    log(f"[tune] 全体 {len(raw):,}件 / 探索に使う期間 〜{cutoff.date()}（{len(tune_df):,}件 / "
        f"正例率 {tune_df['label'].mean() * 100:.2f}%）")

    # ---- 列の選び方（探索に使う期間 + 本番 MLP）----
    models = E60.load_models(args.release_dir, args.model_dir)
    mlp_model, mlp_meta = models[ALGO]
    if list(mlp_meta.get("features") or []) != list(cols):
        raise SystemExit("本番 MLP の meta の列が preset と違います")
    Xt = tune_df[cols].to_numpy(dtype=float)
    yt = tune_df["label"].to_numpy(dtype=int)
    arm_cols, table = select(cols, Xt, yt, mlp_model)
    table.to_csv(os.path.join(OOF_DIR, "e61_selection.csv"), index=False)
    if any(a in TUNE_SELECT for a in arms):
        # 先読みの無い選び方: 探索に使う期間だけで学習した MLP（P のパラメータ・種42）で A / C と同じ決め方
        rec_p = tune_arm("P", tune_df, cols, args.n_trials)
        t0 = time.time()
        model_t = fit_mlp(tune_df, cols, rec_p["params"])
        arm_t, table_t = select(cols, Xt, yt, model_t)
        table_t.to_csv(os.path.join(OOF_DIR, "e61_selection_t.csv"), index=False)
        for a, src in TUNE_SELECT.items():
            arm_cols[a] = arm_t[src]
        log(f"  探索期間だけで学習した MLP で列を選び直した（{time.time() - t0:.0f}秒）: "
            + " / ".join(f"{a} と {src} の重なり {len(set(arm_cols[a]) & set(arm_cols[src]))}列"
                         f"（{a} {len(arm_cols[a])}列・{src} {len(arm_cols[src])}列）"
                         for a, src in TUNE_SELECT.items()))
    for a in arms:
        with open(os.path.join(OOF_DIR, f"e61_cols_{a}.json"), "w", encoding="utf-8") as fh:
            json.dump(arm_cols[a], fh, ensure_ascii=False, indent=0)
    g_of = F.column_groups()
    print("\n■ 0. 腕ごとの列（探索に使う期間の単独 AUC と、MLP の寄与・置換で決めた）")
    for a in arms:
        dropped = [c for c in cols if c not in set(arm_cols[a])]
        fam = pd.Series([E60.FAMILY_OF.get(g_of.get(c, ""), "その他") for c in dropped]).value_counts()
        print(f"  {LABELS[a]}: 残す {len(arm_cols[a])}列 / 外す {len(dropped)}列"
              + (f"（外した大区分: " + " / ".join(f"{k} {v}" for k, v in fam.head(6).items()) + "）" if dropped else ""))
        # 列名だけなので公開ログに出してよい（値は出さない）。採用するときはこの一覧を preset に固定する
        if dropped:
            print("     外す列: " + ", ".join(dropped))
            print("     残す列: " + ", ".join(arm_cols[a]))
    log(f"  前処理後の入力の列数: " + " / ".join(
        f"{a} {int(TM._preprocess(arm_cols[a], PREP).fit_transform(tune_df[arm_cols[a]].to_numpy(dtype=float)).shape[1])}"
        for a in arms))

    recs = {a: tune_arm(a, tune_df, arm_cols[a], args.n_trials) for a in arms}
    del raw, tune_df, Xt

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'腕':<4}{'列数':>5}{'PR-AUC':>9}{'±SD':>8}{'ROC':>9}{'秒':>7}  パラメータ")
    for a in arms:
        cv = recs[a]["_cv"]
        print(f"  {a:<4}{len(arm_cols[a]):>5}{cv['mean_pr_auc']:>9.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>9.4f}"
              f"{cv.get('seconds', 0):>7}  {recs[a]['params']}")

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log(f"データ {len(df):,}件 / 正例率 {df['label'].mean() * 100:.2f}% / "
        f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}")

    summary = {"cols": {a: len(arm_cols[a]) for a in arms},
               "tuning": {a: recs[a]["_cv"] | {"params": recs[a]["params"]} for a in arms},
               "production_oof": {}}
    print("\n■ 2. 本番と同じ作りの out-of-fold（ずらし0・種42）")
    print(f"  {'腕':<4}{'PR-AUC':>9}{'リフト':>7}{'ROC':>9}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}")
    prod = {}
    for a in arms:
        o = oof_arm(a, df, arm_cols[a], recs[a]["params"], 0, (PROD_SEED,))
        prod[a] = o
        m = metrics(o)
        summary["production_oof"][a] = {k: float(v) for k, v in m.items()}
        print(f"  {a:<4}{m['pr_auc']:>9.4f}{m['lift']:>6.2f}x{m['roc_auc']:>9.4f}"
              f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
              f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt")
    mp = metrics(prod["P"])
    for a in arms:
        if a == "P":
            continue
        m = metrics(prod[a])
        print(f"  {a}−P {m['pr_auc'] - mp['pr_auc']:>+9.4f}{'':>7}{m['roc_auc'] - mp['roc_auc']:>+9.4f}"
              f"{m['day_auc'] - mp['day_auc']:>+8.4f}")

    # MLP は種で ±0.01 動く（実験58）。本番と同じ作りの OOF を種3つの平均でも出す（ずらし0の窓の計算を共用）
    if len(seeds) > 1:
        print(f"\n■ 2b. 本番と同じ作りの out-of-fold（ずらし0・種{seeds} の平均）")
        print(f"  {'腕':<4}{'PR-AUC':>9}{'リフト':>7}{'ROC':>9}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}")
        prod3 = {}
        summary["production_oof_3seeds"] = {}
        for a in arms:
            o = oof_arm(a, df, arm_cols[a], recs[a]["params"], 0, seeds)
            prod3[a] = o
            m = metrics(o)
            summary["production_oof_3seeds"][a] = {k: float(v) for k, v in m.items()}
            print(f"  {a:<4}{m['pr_auc']:>9.4f}{m['lift']:>6.2f}x{m['roc_auc']:>9.4f}"
                  f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                  f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt")
        mp3 = metrics(prod3["P"])
        for a in arms:
            if a == "P":
                continue
            m = metrics(prod3[a])
            print(f"  {a}−P {m['pr_auc'] - mp3['pr_auc']:>+9.4f}{'':>7}{m['roc_auc'] - mp3['roc_auc']:>+9.4f}"
                  f"{m['day_auc'] - mp3['day_auc']:>+8.4f}")

    rows = []
    for sh in shifts:
        print(f"\n■ 3. 分離力（窓ごと。ずらし{sh}か月）")
        print(f"  {'':<22}{'P':>9}{'腕':>9}{'腕−P':>10}{'SE':>9}{'腕が上':>9}")
        res = {a: oof_arm(a, df, arm_cols[a], recs[a]["params"], sh, seeds) for a in arms}
        wp = AB.auc_by_window(res["P"])
        for a in arms:
            if a == "P":
                continue
            m = wp.merge(AB.auc_by_window(res[a]), on="fold", suffixes=("_p", "_a"))
            print(pair_line(f"{a} PR-AUC", m["pr_p"].to_numpy(), m["pr_a"].to_numpy()))
            print(pair_line(f"{a} ROC-AUC", m["roc_p"].to_numpy(), m["roc_a"].to_numpy()))
            for _, r in m.iterrows():
                rows.append({"shift": sh, "arm": a, "fold": int(r["fold"]), "pr_p": r["pr_p"], "pr_a": r["pr_a"],
                             "roc_p": r["roc_p"], "roc_a": r["roc_a"]})
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e61_auc_by_window.csv"), index=False)

    if rows:
        s = pd.DataFrame(rows)
        print("\n■ 4. 切り方3通りをまとめた窓ごとの 腕−P")
        print(f"  {'':<16}{'窓の数':>7}{'腕−P の平均':>12}{'SE':>9}{'腕が上':>9}")
        summary["windows"] = {}
        for a, g in s.groupby("arm"):
            for met in ("pr", "roc"):
                d = (g[f"{met}_a"] - g[f"{met}_p"]).to_numpy()
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                print(f"  {a + ' ' + met.upper():<16}{len(d):>7}{d.mean():>+12.4f}{se:>9.4f}{int((d > 0).sum()):>5}/{len(d)}")
                summary["windows"][f"{a}_{met}"] = {"n": int(len(d)), "mean_diff": float(d.mean()),
                                                   "se": float(se), "wins": int((d > 0).sum())}
    with open(os.path.join(OOF_DIR, "e61_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OOF_DIR}/e61_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
