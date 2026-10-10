#!/usr/bin/env python3
"""
実験72: 非線形時系列解析の 32列だけで LightGBM を学習すると、どのくらい当たるか。

運用者の問い（2026-10-10）「このセッションで作成した特徴量は何個ありますか？ それのみを使ってモデルを作った時に
どのくらいの精度が出るのか知りたい」。確認（AskUserQuestion）で LightGBM だけ / 比較の相手は本番239列と正例率 /
物差しは分離力 + 運用の選定の的中率、を選択。

腕（LightGBM。3切り方 × 種3つ。実験66 と同じ out-of-fold の作り）:
  T   本番の239列・本番の木の形（実験66 の保存済みを読む。無ければ計算する）
  N   非線形32列（nonlinear_features.NL_COLS）だけ・本番の木の形
  Nt  非線形32列だけ・32列に合わせて探索し直した木の形（tuning.tune、50試行 × 5分割 year_cap_date、
      ホールドアウトより前の行。実験66 の tune_arm と同じ）
物差し:
  1. 窓ごとの PR-AUC / ROC-AUC と腕の差（SE・上の窓）。下限は正例率（でたらめな順位の PR-AUC ≒ 正例率、ROC-AUC 0.5）
  2. 運用の選定（lgbm 単体 95 / 90。前の窓の分布の百分位）で選ばれた行の 正例率・平均収益・勝率・+10% 以上・−10% 未満
     （ラベルは全腕で同じなので、正例率をそのまま比べられる）
  3. 32列の中でどれが効くか（Nt の木の形で、ホールドアウトより前の行で 1 回学習した gain 重要度）

  python3 research/exp/e72_nl_only.py [--shifts 0,2,4] [--seeds 3] [--n-trials 50]
  結果は research/_data/oof/e72_*（auc_by_window / precision / importance）。本番の設定には書かない。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import nonlinear_features as NL  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import ops_rule as OR  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e66_nonlinear_screen as E66  # noqa: E402
import e70_label_sigma_dfa_full as E70  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TAG = "e72"
LABELS = {"T": "T 本番の239列", "N": "N 非線形32列だけ", "Nt": "Nt 32列だけ・探索し直し"}
#: 運用の選定（LightGBM 単体。ops_rule.RULES の 1 つ目と、同じ作りの 90）
RULES = (("lgbm 単体 95以上", 95.0), ("lgbm 単体 90以上", 90.0))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def positive_rate_by_window(o: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの正例率（でたらめな順位の PR-AUC の目安）。"""
    return o.groupby("fold")["label"].mean().rename("pos").reset_index()


def auc_table(res: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """腕ごとの窓ごとの PR-AUC / ROC-AUC を 1 つの表に（fold / pos / pr_<腕> / roc_<腕>）。全腕にある窓だけ。"""
    first = next(iter(res))
    tab = positive_rate_by_window(res[first])
    for arm, o in res.items():
        w = AB.auc_by_window(o).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
        tab = tab.merge(w, on="fold", how="inner")
    return tab


def pair_summary(tab: pd.DataFrame, a: str, b: str) -> Dict[str, dict]:
    """腕 b − 腕 a の窓ごとの差（PR-AUC / ROC-AUC）: 平均・SE・上の窓。"""
    out = {}
    for k in ("pr", "roc"):
        d = (tab[f"{k}_{b}"] - tab[f"{k}_{a}"]).to_numpy(dtype=float)
        out[k] = {"a": float(tab[f"{k}_{a}"].mean()), "b": float(tab[f"{k}_{b}"].mean()),
                  "diff": float(d.mean()),
                  "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                  "won": int((d > 0).sum()), "n": int(len(d))}
    return out


def precision_rows(res: Dict[str, pd.DataFrame], rules=RULES) -> pd.DataFrame:
    """腕ごと・選定ごとに、選ばれた行の正例率・平均収益・勝率・+10% 以上・−10% 未満（実験70 の band_stats）。"""
    recs = []
    for arm, o in res.items():
        for name, pct in rules:
            c = OR.consensus({"lgbm": o}, pct, models=("lgbm",), keep=True)
            if not c.get("n"):
                recs.append({"arm": arm, "rule": name, "n": 0})
                continue
            st = E70.band_stats(c["rows"])
            st.update({"arm": arm, "rule": name, "rate": float(c["rate"]),
                       "lift20": float(c["ret_o1_20"]["fold_mean"]),
                       "won": int(c["ret_o1_20"]["won"]), "folds": int(c["ret_o1_20"]["n_folds"])})
            recs.append(st)
    cols = ["arm", "rule", "n", "rate", "hit_own", "ret20", "win", "big_up", "big_down", "lift20", "won", "folds"]
    t = pd.DataFrame(recs)
    return t.reindex(columns=[c for c in cols if c in t.columns])


def importance(df: pd.DataFrame, cols: Sequence[str], params: dict, train_end) -> pd.Series:
    """ホールドアウトより前の行で 1 回学習した gain 重要度（合計 1 に正規化、降順）。"""
    import lightgbm as lgb

    sub = df[(pd.to_datetime(df["Date"]) <= pd.Timestamp(train_end)) & df["label"].notna()]
    y = sub["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": 0, "verbose": -1},
                           scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(sub[list(cols)].to_numpy(dtype=float), y)
    imp = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=list(cols), dtype=float)
    return (imp / imp.sum()).sort_values(ascending=False)


PREC_HEADER = (f"  {'腕':<26}{'規則':<16}{'件数':>7}{'選定率':>7}{'正例率':>8}{'ret20':>8}{'勝率':>7}"
               f"{'+10%以上':>9}{'-10%未満':>9}{'窓平均超過':>10}{'勝ち窓':>8}")


def fmt_prec(r: pd.Series, label: str) -> str:
    if not r.get("n"):
        return f"  {label:<26}{r['rule']:<16}（選定なし）"
    rate = f"{r['rate']*100:>6.1f}%" if np.isfinite(r.get("rate", np.nan)) else f"{'':>7}"
    lift = f"{r['lift20']:>+9.2f}pt" if np.isfinite(r.get("lift20", np.nan)) else f"{'':>11}"
    return (f"  {label:<26}{r['rule']:<16}{int(r['n']):>7,}{rate}{r['hit_own']*100:>7.1f}%"
            f"{r['ret20']:>+7.2f}%{r['win']*100:>6.0f}%{r['big_up']*100:>8.1f}%{r['big_down']*100:>8.1f}%"
            f"{lift}{int(r['won']):>5}/{int(r['folds']):<3}")


def show_pairs(tab: pd.DataFrame, pairs=(("T", "N"), ("T", "Nt"), ("N", "Nt"))) -> None:
    print(f"  {'':<22}{'a':>9}{'b':>9}{'b − a':>10}{'SE':>9}{'上の窓':>9}")
    for a, b in pairs:
        s = pair_summary(tab, a, b)
        for k, ja in (("pr", "PR-AUC"), ("roc", "ROC-AUC")):
            v = s[k]
            print(f"  {b + ' − ' + a + ' ' + ja:<22}{v['a']:>9.4f}{v['b']:>9.4f}{v['diff']:>+10.4f}{v['se']:>9.4f}"
                  f"{v['won']:>5}/{v['n']:<3}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験72: 非線形32列だけの LightGBM")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=E66.N_TRIALS)
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    os.makedirs(OOF_DIR, exist_ok=True)

    df = E66.load_frame()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    base = F.columns(F.DEFAULT_PRESET)
    nl = list(NL.NL_COLS)
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    n_trees = E66.PROD_TREES
    lr = tuning.lr_range(n_trees)
    print("=" * 78)
    print(f"実験72 非線形32列だけの LightGBM（種{len(seeds)}つ・ずらし {shifts}か月・木 {n_trees}本・タグ {TAG}）: "
          f"{len(df):,}件 / 正例率 {df['label'].mean()*100:.2f}% / 探索は 〜{train_end.date()} {len(sub):,}件")
    for arm, cols in (("T", base), ("N", nl), ("Nt", nl)):
        print(f"  {LABELS[arm]:<30}{len(cols)}列  指紋 {F.signature(cols)}")
    print("=" * 78)

    prod = E27.prod_params("lgbm")["params"]
    rec = E66.tune_arm("lgbm", "N", sub, nl, n_trees, lr, train_end, args.n_trials)
    tuned = dict(rec["params"])
    cv = rec.get("_cv", {})
    print(f"\n■ 32列での探索し直し（{args.n_trials}試行 × {E66.N_SPLITS}分割、木 {n_trees}本、学習率 {lr[0]:g}〜{lr[1]:g}）: "
          f"所要 {rec.get('_minutes')}分 / CV PR-AUC {cv.get('mean_pr_auc')} ± {cv.get('std')} / ROC {cv.get('mean_roc_auc')} "
          f"/ 正例率 {cv.get('base_rate')}")
    keys = ["learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    for name, p in (("本番の木の形（239列で探索）", prod), ("32列で探索し直し", tuned)):
        print(f"  {name:<24}" + " / ".join(f"{k} {p[k]:.4g}" if isinstance(p.get(k), float) else f"{k} {p.get(k)}"
                                          for k in keys if k in p))

    imp = importance(df, nl, tuned, train_end)
    imp.rename("gain_share").to_csv(os.path.join(OOF_DIR, f"{TAG}_importance.csv"))
    print(f"\n■ 32列の gain 重要度（Nt の木の形、〜{train_end.date()} の行で 1 回学習。上位 12）")
    for c, v in imp.head(12).items():
        print(f"  {c:<24}{v*100:>6.1f}%  {NL.NL_DESC.get(c, '')}")
    print(f"  上位 5 列で {imp.head(5).sum()*100:.0f}%、上位 10 列で {imp.head(10).sum()*100:.0f}%")

    tabs, precs = [], []
    for sh in shifts:
        res = {"T": AB.oof_arm(E66.tag_for(n_trees), df, base, "T", sh, ["lgbm"], seeds)["lgbm"],
               "N": AB.oof_arm(TAG, df, nl, "N", sh, ["lgbm"], seeds)["lgbm"],
               "Nt": E66.oof_arm_params(TAG, df, nl, "Nt", sh, "lgbm", seeds, tuned)}
        tab = auc_table(res)
        tab.insert(0, "shift", sh)
        tabs.append(tab)
        print(f"\n■ ずらし{sh}か月: 窓ごとの PR-AUC / ROC-AUC（pos = 正例率 = でたらめな順位の PR-AUC の目安）")
        print(f"  {'窓':>4}{'pos':>8}{'PR T':>9}{'PR N':>9}{'PR Nt':>9}{'ROC T':>9}{'ROC N':>9}{'ROC Nt':>9}")
        for _, r in tab.iterrows():
            print(f"  {int(r['fold']):>4}{r['pos']:>8.4f}{r['pr_T']:>9.4f}{r['pr_N']:>9.4f}{r['pr_Nt']:>9.4f}"
                  f"{r['roc_T']:>9.4f}{r['roc_N']:>9.4f}{r['roc_Nt']:>9.4f}")
        print(f"  {'平均':>4}{tab['pos'].mean():>8.4f}{tab['pr_T'].mean():>9.4f}{tab['pr_N'].mean():>9.4f}{tab['pr_Nt'].mean():>9.4f}"
              f"{tab['roc_T'].mean():>9.4f}{tab['roc_N'].mean():>9.4f}{tab['roc_Nt'].mean():>9.4f}")
        show_pairs(tab)
        prec = precision_rows(res)
        prec.insert(0, "shift", sh)
        precs.append(prec)
        print(f"  ◆ 運用の選定（前の窓の分布の百分位。正例率は共通のラベル）")
        print(PREC_HEADER)
        for _, r in prec.iterrows():
            print(fmt_prec(r, LABELS[r["arm"]]))

    allt = pd.concat(tabs, ignore_index=True)
    allt.to_csv(os.path.join(OOF_DIR, f"{TAG}_auc_by_window.csv"), index=False)
    allp = pd.concat(precs, ignore_index=True)
    allp.to_csv(os.path.join(OOF_DIR, f"{TAG}_precision.csv"), index=False)
    print(f"\n■ 切り方{len(shifts)}通りをまとめて（{len(allt)}窓）")
    print(f"  {'':<22}{'正例率':>9}{'T':>9}{'N':>9}{'Nt':>9}")
    print(f"  {'PR-AUC の平均':<22}{allt['pos'].mean():>9.4f}{allt['pr_T'].mean():>9.4f}{allt['pr_N'].mean():>9.4f}{allt['pr_Nt'].mean():>9.4f}")
    print(f"  {'ROC-AUC の平均':<22}{0.5:>9.4f}{allt['roc_T'].mean():>9.4f}{allt['roc_N'].mean():>9.4f}{allt['roc_Nt'].mean():>9.4f}")
    print(f"  {'PR ÷ 正例率（倍率）':<22}{1.0:>9.2f}{(allt['pr_T']/allt['pos']).mean():>9.2f}"
          f"{(allt['pr_N']/allt['pos']).mean():>9.2f}{(allt['pr_Nt']/allt['pos']).mean():>9.2f}")
    print(f"  正例率を上回る窓: N {int((allt['pr_N'] > allt['pos']).sum())}/{len(allt)}、"
          f"Nt {int((allt['pr_Nt'] > allt['pos']).sum())}/{len(allt)}、T {int((allt['pr_T'] > allt['pos']).sum())}/{len(allt)}")
    show_pairs(allt)
    s = E70.summarize_precision(allp)
    print(f"  ◆ 運用の選定（{len(shifts)}切り方の合計。件数で重み付け、勝ち窓は合計）")
    print(PREC_HEADER)
    for _, r in s.iterrows():
        print(fmt_prec(r, LABELS[r["arm"]]))
    log(f"記録: {OOF_DIR}/{TAG}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
