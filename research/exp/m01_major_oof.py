#!/usr/bin/env python3
"""
本ブレイク予測モデルの最初の測定（ブランチ claude/major-breakout-model）。

運用者の依頼（2026-09-23）
  「次は、今の特徴量（205列）でこのラベルを学習し、今のモデルと同じ窓で分離力を
   測るところから始められます。上記をお願いします。」

ラベル  research/major/label_eda.py の major_label。60営業日以内に一度でも終値が +50%、
        かつ 120営業日以内に一度でも終値が 2倍（買値 = 翌営業日の寄り）。買った当日の
        終値で +50% に届いたものは負例。120営業日先まで上場していない行は判定できないので外す
列      本番の205列（features.DEFAULT_PRESET = all_plus）
窓      今のモデルの out-of-fold と同じテスト窓（36か月 / 6か月 / 6か月。train_production の OOF_*）
学習    **テスト開始の121営業日前までの行だけ**。ラベルが120営業日先まで見るので、それより
        後の行を入れるとテスト期間の値動きが答えに混ざる。今のモデル（20営業日のラベル）は
        20営業日あけているだけなので、ここだけ違う
モデル  LightGBM / XGBoost / CatBoost。パラメータは本番の今の値（新しいラベル用には探索していない）。
        正例の重みは学習データの正例率から（tuning.scale_pos_weight。本番と同じ）。種3つの確率平均
指標    PR-AUC（全窓まとめ）と正例率で割った値（リフト）、ROC-AUC、日内AUC、窓ごと、
        窓の中で上位 1 / 5 / 10% に入った行の正例率と、その後の値動き
比べる  (1) 今のモデル（20営業日のラベルで学習した LightGBM、本番のパラメータ・同じ窓）の
            スコアで新しいラベルを並べた場合
        (2) 列1本だけで並べた場合（向きは先に決める: 日次ボラ 高い順 / 時価総額 小さい順 /
            直近20日の上昇率 高い順 / 6か月前の高値からの位置 低い順）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "major"))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import walkforward as WF  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import label_eda as LE  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGOS = ("lgbm", "xgb", "cat")
SEEDS = (42, 7, 123)
PURGE = LE.DAYS_2 + 1                       # テスト開始の何営業日前までの行で学習するか
PREFIX = "m01"                              # 保存するファイルの頭（--quick では別にする）
TOPS = (0.01, 0.05, 0.10)
MIN_TEST, MIN_TRAIN, MIN_TRAIN_POS = 200, 1000, 20
#: 列1本の比べ物。向き（+1 = 大きいほど正例らしい）は結果を見る前に決めた
SINGLE = (("vol_20d", +1, "日次ボラが高い順"), ("log_market_cap", -1, "時価総額が小さい順"),
          ("ret_20d", +1, "直近20日の上昇率が高い順"), ("r_high_6m", -1, "6か月前の高値からの位置が低い順"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_plan(dates: pd.Series, cal: pd.DatetimeIndex) -> list:
    """今のモデルと同じテスト窓と、新しいラベルで学習に使える最後の日（テスト開始の PURGE 営業日前）。"""
    folds = WF.make_folds(pd.to_datetime(dates), min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    plan = []
    for f in folds:
        k = int(cal.searchsorted(pd.Timestamp(f.test_start)))    # テスト開始以降で最初の営業日
        plan.append((f, cal[k - PURGE] if k - PURGE >= 0 else None))
    return plan


def oof(algo: str, seed: int, d: pd.DataFrame, cols: list, params: dict, plan: list,
        label: str, cut_of) -> pd.DataFrame:
    """
    plan の窓で out-of-fold を作る。学習は cut_of(fold, cutoff) の日までの label が付いた行、
    テストは新しいラベル（y）が付いた行。学習器の組み方は実験41 と同じ。
    """
    import lightgbm as lgb
    import models as M
    import tuning
    import tuning_multi as TM

    p = params["params"] if isinstance(params.get("params"), dict) else params
    parts = []
    try:
        for f, cut in plan:
            last = cut_of(f, cut)
            if last is None:
                continue
            tr = d[(d["Date"] <= last) & d[label].notna()]
            te = d[(d["Date"] >= pd.Timestamp(f.test_start)) & (d["Date"] <= pd.Timestamp(f.test_end))
                   & d["y"].notna()]
            if len(te) < MIN_TEST or len(tr) < MIN_TRAIN or tr[label].sum() < MIN_TRAIN_POS:
                continue
            Xtr, ytr = tr[cols].to_numpy(dtype=float), tr[label].to_numpy(dtype=int)
            Xte = te[cols].to_numpy(dtype=float)
            if algo == "lgbm":
                m = lgb.LGBMClassifier(**{**p, "random_state": seed},
                                       scale_pos_weight=tuning.scale_pos_weight(ytr))
                m.fit(Xtr, ytr)
                sc = m.predict_proba(Xte)[:, 1]
            else:
                TM.SEED = seed                  # build() が random_state に使う
                m = M.fit(algo, Xtr, ytr, cols, params=p)
                sc = M.predict(m, Xte)
            part = te[["Code", "Date"]].copy()
            part["score"] = sc
            part["fold"] = f.index
            part["n_train"], part["pos_train"] = len(tr), int(ytr.sum())
            parts.append(part)
    finally:
        TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


def averaged(algo: str, tag: str, d: pd.DataFrame, cols: list, params: dict, plan: list,
             label: str, cut_of) -> pd.DataFrame:
    """種3つの確率平均。種ごとに保存し、2回目以降は読む。"""
    outs = []
    for sd in SEEDS:
        path = os.path.join(OOF_DIR, f"{PREFIX}_{algo}_{tag}_s{sd}.parquet")
        if os.path.exists(path):
            o = pd.read_parquet(path)
            o["Date"] = pd.to_datetime(o["Date"])
            log(f"  [{algo}/{tag}] 種{sd} 保存済みを読む")
        else:
            t0 = time.time()
            o = oof(algo, sd, d, cols, params, plan, label, cut_of)
            o.to_parquet(path, index=False)
            log(f"  [{algo}/{tag}] 種{sd} {time.time() - t0:.0f}秒（{o['fold'].nunique()}窓・{len(o):,}行）")
        outs.append(o.set_index(["Code", "Date"]))
    base = outs[0].copy()
    base["score"] = np.mean([o["score"].reindex(base.index).to_numpy() for o in outs], axis=0)
    return base.reset_index()


def pct_in_fold(o: pd.DataFrame) -> np.ndarray:
    """窓の中での順位（0〜1、大きいほど上位）。窓ごとにモデルが違うので、上位X%は窓の中で切る。"""
    return o.groupby("fold")["score"].rank(pct=True).to_numpy()


def measure(o: pd.DataFrame, y: np.ndarray, extra: pd.DataFrame) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = o["score"].to_numpy(dtype=float)
    base = float(y.mean())
    pr = float(average_precision_score(y, s))
    day = lab.auc_in_day(pd.DataFrame({"Date": o["Date"], "label": y, "score": s}))
    r = {"n": len(y), "pos": int(y.sum()), "base": base, "pr": pr, "lift": pr / base,
         "roc": float(roc_auc_score(y, s)), "day": float(day)}
    q = pct_in_fold(o)
    for t in TOPS:
        m = q > 1 - t
        r[f"top{int(t*100)}"] = float(y[m].mean())
        r[f"top{int(t*100)}_n"] = int(m.sum())
    m10 = q > 0.9
    for c in ("ret_o1_60", "ret_o1_120", "mx"):
        r[f"{c}_top10"] = float(np.nanmean(extra[c].to_numpy(dtype=float)[m10]))
        r[f"{c}_all"] = float(np.nanmean(extra[c].to_numpy(dtype=float)))
    return r


HEAD = (f"  {'並べ方':<34}{'件数':>7}{'正例':>6}{'正例率':>8}{'PR-AUC':>9}{'リフト':>8}{'ROC-AUC':>9}"
        f"{'日内AUC':>9}{'上位1%':>8}{'上位5%':>8}{'上位10%':>8}")


def line(name: str, r: dict) -> str:
    return (f"  {name:<34}{r['n']:>7,}{r['pos']:>6}{r['base']*100:>7.2f}%{r['pr']:>9.4f}{r['lift']:>7.2f}x"
            f"{r['roc']:>9.4f}{r['day']:>9.4f}{r['top1']*100:>7.1f}%{r['top5']*100:>7.1f}%{r['top10']*100:>7.1f}%")


def main(argv=None) -> int:
    global ALGOS, SEEDS, PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true",
                    help="手元の動作確認用。LightGBM・種1つだけ（結果は別の名前で保存し、本番の測定には使わない）")
    args = ap.parse_args(argv)
    if args.quick:
        ALGOS, SEEDS, PREFIX = ("lgbm",), (42,), "m01quick"
    os.makedirs(OOF_DIR, exist_ok=True)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    bars = LE.load_bars()
    cal = pd.DatetimeIndex(np.sort(bars["Date"].unique()))
    lf = LE.label_frame(df[["Code", "Date"]], bars)
    df["y"] = lf["y_major"].to_numpy()
    fw = LE.forward_closes(bars, df[["Code", "Date"]])
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        df["mx"] = np.nanmax(fw["C"] / fw["entry"][:, None], axis=1) - 1.0     # 120営業日のうちの最高値（終値）
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    miss = [c for c in F.columns(F.DEFAULT_PRESET) if c not in df.columns]
    if miss:
        raise SystemExit(f"本番の列がデータセットに無い: {miss[:5]} ほか{len(miss)}列")

    plan = fold_plan(df["Date"], cal)
    print("=== 0. 前提 ===")
    print(f"  列: {F.DEFAULT_PRESET}（{len(cols)}列） / 母集団 {len(df):,}件 / 新しいラベルが付く行 "
          f"{int(df['y'].notna().sum()):,}件（正例 {int(np.nansum(df['y']))}件）")
    print(f"  {'窓':>3} {'テスト':<24}{'今のモデルの学習の最後':>14}{'新しいラベルの学習の最後':>16}")
    for f, cut in plan:
        print(f"  {f.index:>3} {f.test_start}〜{f.test_end}  {f.train_end:>14}"
              f"  {str(cut.date()) if cut is not None else '-':>16}")
    print(f"  新しいラベルは、テスト開始の{PURGE}営業日前までの行で学習する（120営業日先まで見るラベルなので）")

    # --- 新しいラベルで学習 --- #
    scores = {}
    for a in ALGOS:
        par = E27.prod_params(a)
        log(f"[{a}] 本番のパラメータ（{len(par['params'])}個）で学習する")
        scores[a] = averaged(a, "major", df, cols, par, plan, "y", lambda f, cut: cut)
    # --- 今のモデル（20営業日のラベル）を同じ窓で。学習の区切りは今のモデルと同じ --- #
    ref = averaged("lgbm", "cur", df, cols, E27.prod_params("lgbm"), plan, "label",
                   lambda f, cut: pd.Timestamp(f.train_end))

    # 比べる行を揃える（全部のモデルに点がある行）
    key = ["Code", "Date"]
    common = scores["lgbm"][key]
    for a in ALGOS[1:]:
        common = common.merge(scores[a][key], on=key)
    common = common.merge(ref[key], on=key)
    common = common.merge(df[key + ["y", "ret_o1_60", "ret_o1_120", "mx"] + [c for c, _, _ in SINGLE]],
                          on=key, how="left")
    folds_of = scores["lgbm"].set_index(key)["fold"]
    common["fold"] = folds_of.reindex(pd.MultiIndex.from_frame(common[key])).to_numpy()
    y = common["y"].to_numpy(dtype=int)

    def on_common(o: pd.DataFrame) -> pd.DataFrame:
        s = o.set_index(key)["score"].reindex(pd.MultiIndex.from_frame(common[key])).to_numpy()
        return common[key + ["fold"]].assign(score=s)

    res, rows = {}, {}
    for a in ALGOS:
        rows[a] = on_common(scores[a])
    ens = common[key + ["fold"]].copy()
    ens["score"] = np.mean([pct_in_fold(rows[a]) for a in ALGOS], axis=0)
    rows["ens"] = ens
    rows["ref"] = on_common(ref)
    for c, sign, _ in SINGLE:
        rows[c] = common[key + ["fold"]].assign(score=sign * common[c].to_numpy(dtype=float))

    names = {"lgbm": "LightGBM（新しいラベル）", "xgb": "XGBoost（新しいラベル）", "cat": "CatBoost（新しいラベル）",
             "ens": "3モデルの平均順位", "ref": "今のモデル（20日ラベルの LightGBM）"}
    names.update({c: f"列1本: {lab_}" for c, _, lab_ in SINGLE})
    from sklearn.metrics import roc_auc_score

    print(f"\n=== 1. 分離力（全窓まとめ。{len(common):,}行・正例 {int(y.sum())}件）===")
    print(HEAD)
    models_ = [a for a in ALGOS] + ["ens", "ref"]
    for k in models_ + [c for c, _, _ in SINGLE]:
        o = rows[k]
        ok = np.isfinite(o["score"].to_numpy(dtype=float))
        r = measure(o[ok], y[ok], common[ok])
        res[k] = r
        print(line(names[k], r))
    print("  リフト = PR-AUC ÷ 正例率（1.00x = 無作為に並べたのと同じ）。上位X% = 窓の中で上位X%に入った行の正例率。")
    print("  日内AUC = 同じ日の候補どうしの順位（正例と負例が両方ある日だけ）。列1本は欠測の行を除いて測る。")

    print("\n=== 2. 上位10%に入った行のその後（買値から。窓の中で上位10%）===")
    print(f"  {'並べ方':<34}{'60日後(5日平均)':>14}{'120日後(5日平均)':>16}{'120日の最高値':>14}")
    r0 = res["lgbm"]
    print(f"  {'（全体）':<34}{r0['ret_o1_60_all']*100:>+13.1f}%{r0['ret_o1_120_all']*100:>+15.1f}%{r0['mx_all']*100:>+13.1f}%")
    for k in models_:
        r = res[k]
        print(f"  {names[k]:<34}{r['ret_o1_60_top10']*100:>+13.1f}%{r['ret_o1_120_top10']*100:>+15.1f}%"
              f"{r['mx_top10']*100:>+13.1f}%")

    print("\n=== 3. 窓ごと（3モデルの平均順位 / LightGBM / 今のモデル）===")
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"  {'窓':>3} {'テスト':<24}{'学習':>7}{'学習の正例':>9}{'件数':>6}{'正例':>5}{'正例率':>8}"
          f"{'平均順位 リフト':>14}{'ROC':>7}{'上位10%':>8}{'LGBM リフト':>12}{'今のモデル リフト':>15}")
    per = []
    for f, cut in plan:
        m = (common["fold"] == f.index).to_numpy()
        if m.sum() == 0:
            continue
        yy = y[m]
        tr_info = scores["lgbm"][scores["lgbm"]["fold"] == f.index].iloc[0]
        row = {"fold": f.index, "n": int(m.sum()), "pos": int(yy.sum())}
        if 0 < yy.sum() < len(yy):
            for k in ("ens", "lgbm", "ref"):
                s = rows[k]["score"].to_numpy(dtype=float)[m]
                row[f"{k}_lift"] = average_precision_score(yy, s) / yy.mean()
                row[f"{k}_roc"] = roc_auc_score(yy, s)
            q = rows["ens"]["score"].to_numpy(dtype=float)[m]
            q = pd.Series(q).rank(pct=True).to_numpy()
            row["ens_top10"] = yy[q > 0.9].mean()
        per.append(row)
        fmt = lambda k: f"{row[k]:.2f}x" if k in row else "-"   # noqa: E731
        print(f"  {f.index:>3} {f.test_start}〜{f.test_end}{int(tr_info['n_train']):>7,}{int(tr_info['pos_train']):>9}"
              f"{row['n']:>6}{row['pos']:>5}{yy.mean()*100:>7.2f}%{fmt('ens_lift'):>14}"
              f"{row.get('ens_roc', float('nan')):>7.3f}{row.get('ens_top10', float('nan'))*100:>7.1f}%"
              f"{fmt('lgbm_lift'):>12}{fmt('ref_lift'):>15}")
    pf = pd.DataFrame(per)
    for k in ("ens", "lgbm", "ref"):
        v = pf[f"{k}_lift"].dropna()
        print(f"  {names[k]}: 窓平均リフト {v.mean():.2f}x / 1倍を超えた窓 {(v > 1).sum()}/{len(v)} / 最低 {v.min():.2f}x")
    print("  学習 = その窓で学習に使った行数（テスト開始の121営業日前まで）。件数の少ない窓のリフトは大きく揺れる。")

    print("\n=== 4. 日次ボラの近い銘柄どうしで比べると（ボラの五分位ごと）===")
    print("  列1本の日次ボラが強いので、モデルがボラ以外で当てているかを見る。五分位は比べる行全体で切る。")
    vol = common["vol_20d"].to_numpy(dtype=float)
    okv = np.isfinite(vol)
    qv = pd.qcut(pd.Series(vol[okv]).rank(method="first"), 5, labels=False).to_numpy()
    band = np.full(len(vol), -1)
    band[okv] = qv
    print(f"  {'五分位':<8}{'日次ボラ':>14}{'件数':>7}{'正例':>6}{'正例率':>8}"
          + "".join(f"{names[k][:10] + ' ROC':>18}" for k in models_) + f"{'ボラ1本 ROC':>13}")
    acc = {k: [] for k in models_ + ["vol_20d"]}
    for b_ in range(5):
        m = band == b_
        yy = y[m]
        lo_, hi_ = np.nanmin(vol[m]), np.nanmax(vol[m])
        cells = ""
        for k in models_ + ["vol_20d"]:
            sc = rows[k]["score"].to_numpy(dtype=float)[m]
            ok = np.isfinite(sc)
            v = roc_auc_score(yy[ok], sc[ok]) if 0 < yy[ok].sum() < ok.sum() else float("nan")
            acc[k].append((v, int(m.sum())))
            cells += f"{v:>18.3f}" if k != "vol_20d" else f"{v:>13.3f}"
        print(f"  {'Q' + str(b_ + 1):<8}{lo_:>6.2f}〜{hi_:>5.2f}%{int(m.sum()):>7,}{int(yy.sum()):>6}"
              f"{yy.mean()*100:>7.2f}%{cells}")
    def wavg(v):
        v = [(a, n) for a, n in v if np.isfinite(a)]
        return sum(a * n for a, n in v) / sum(n for _, n in v) if v else float("nan")
    print("  五分位の中の ROC-AUC（件数で重み付け）: "
          + " / ".join(f"{names[k]} {wavg(acc[k]):.3f}" for k in models_ + ["vol_20d"]))
    qe = pct_in_fold(rows["ens"])
    print(f"  3モデルの平均順位で上位10%に入った行の日次ボラ（中央値） {np.nanmedian(vol[qe > 0.9]):.2f}%"
          f" / 全体 {np.nanmedian(vol):.2f}%")

    pd.DataFrame(res).T.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_summary.csv"))
    pf.to_csv(os.path.join(OOF_DIR, f"{PREFIX}_per_window.csv"), index=False)
    log(f"記録: {OOF_DIR}/{PREFIX}_summary.csv / {PREFIX}_per_window.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
