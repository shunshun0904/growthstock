#!/usr/bin/env python3
"""
実験78: 目的変数の到達しきい値 k（必要な上昇率 = k × σ20 × √20）を 1.2σ から 1.5σ にすると、lgbm の選定はどう変わるか
（lgbm だけ。腕ごとに Optuna の探索（50試行 × 5分割 year_cap_date）をし直す。本番は変えない。検証だけ）。

運用者の指示（2026-10-11）「数打てば当たる作戦ではないので、該当銘柄数は今より少し少なくても問題ない。1.5σ にしたら
どうなるかみたい。これだ！というのを月に2銘柄くらい買えれば十分。lgbm だけ Optuna による 5cv パラメータチューニングを。
目的変数に連動して修正が必要な特徴量があればそちらも。本番は何も変更しない」。

1.2σ の根拠（docs/MODEL_DESIGN_SWEEP.md・MODEL_DESIGN_WALKFORWARD.md、2026-09-17 ごろ。当時は h=60・終値基準）:
k = 1.0 / 1.2 / 1.4 / 1.6σ の掃引で、上位5% の実収益 − 全件 は +5.62 / +6.16 / +5.67 / +5.84pt とどれも有意で、
1.2σ が最も良かったので採用。差は小さい（k を上げると正例率が下がり、残った正例は極端になる）。

ラベルの引き直し: データセットに残る future_rise / end_level / uptrend_end から、k だけ変えて合成し直す
（build_dataset.rise_thresholds と attach_rise_label の合成と同じ。終盤の必要水準は到達の 0.5倍のまま → 1.5σ なら 0.75σ、
MA5>=MA20 は同じ）。k=1.2 の引き直しが本番のラベルと全行一致することを最初に確かめる。
**特徴量に目的変数の k に連動するものは無い**（rise_need / end_need は FUTURE_COLS で特徴量に入れていない。vol_20d は
特徴量だが σ そのものでラベルの k に依らない。エンバーゴは RISE_HORIZON = 20 に連動し、k では変わらない）。

腕（ラベルだけ違う。列は本番の239列）
  K12  k = 1.2σ（現行）。探索し直し
  K15  k = 1.5σ。探索し直し
  （--ks で増やせる。例: 1.2,1.5,1.8）

評価（ラベルが違うと PR-AUC は比べられないので、実収益で比べる。実験54 と同じ物差し + 運用者の「月に2銘柄」）
  1. ラベル: 正例率・必要な上昇率の中央値・ボラの帯ごとの正例率・K12 と違う行・正例の実収益
  2. 本番と同じ OOF（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）: 自分のラベルでの PR-AUC・リフト・ROC と、
     相手のラベルで測った PR-AUC・ROC（1.5σ で学んだモデルが 1.2σ の正例も上に並べるか）
  3. 閾値ルール（前の窓の分布の上位5% / 10%）: 件数・平均収益・全体との差・窓ごとの差・−10% 未満・月あたりの発火数
  4. 発火数を K12 にそろえる / 窓の中の上位10%（件数そろえ）/ 日付内の1位・上位2件
  5. 月に2銘柄（各月のスコア上位2件）: 平均収益（20・40営業日）・勝率・+10% 以上の割合・−10% 未満の割合・
     K12 のラベルでの正例率。窓ごとの K15 − K12 の差
  3〜5 は窓のずらし 0/2/4か月 × 種3つの平均（実験68 と同じ作り）。

    exp=e78_label_k.py args="--ks 1.2,1.5,1.8"                 （lgbm）
    exp=e78_label_k.py args="--ks 1.2,1.5,1.8 --algos xgb"     （運用者の指示 2026-10-11「他のモデルも」）
    exp=e78_label_k.py args="--ks 1.2,1.5,1.8 --algos cat"
試運転: exp=e78_label_k.py args="--tag e78smoke --n-trials 2 --shifts 0"

結果（2026-10-11、run 38136357178 / 38139194190、docs/MODEL_ADOPTION_RULES.md §34）: K15 は正例率 13.0%（現行 17.0%）。
現行のラベルで測った 32窓の PR-AUC は K15−K12 +0.0075 ± 0.0019（26/32）、ROC は +0.0008（変わらず）。実収益は上位5% の
閾値ルールで差が無く（+0.36pt、14/29）、上位10% 以下・日付内1位・月に2銘柄（+2.16pt、19/32）では K15 が上。
2回目（run 38139829499、--ks 1.2,1.5,1.8）: 1.8σ は現行のラベルで 1.2σ と変わらず（PR +0.0012 ± 0.0021、ROC −0.0057）、
1.8σ 自身のラベルでも 1.5σ のモデルが上（+0.0117 対 +0.0100）。腕どうしの順位相関は 0.84〜0.89 だが上位5% の組の重なりは
55〜64%。1.5σ が頃合い。本番は変えない（判断は運用者）。

公開ログには件数・割合・日付・精度だけを出す。本番の設定（research/lgbm_params.json、build_dataset.VOL_NORM_K、
features.py）には書かない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import e52_return_objective as E52  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402

log = E68.log
DEFAULT_ALGOS = "lgbm"                  # --algos で xgb / cat / logit も（E68.ALGOS）
OUTCOME = lab.OUTCOME                       # ret_o1_20
K0 = float(B.VOL_NORM_K)                    # 1.2
DEFAULT_KS = "1.2,1.5"
BASE = "K12"
MONTHLY_N = 2                               # 運用者の「月に2銘柄」
TAG = "e78"


def arm_name(k: float) -> str:
    return f"K{round(k * 10):02d}"


def arm_label(k: float) -> str:
    return f"{arm_name(k)} {k:.1f}σ" + ("（現行）" if abs(k - K0) < 1e-9 else "")


# --------------------------------------------------------------------------- #
# ラベルの引き直し
# --------------------------------------------------------------------------- #
def relabel(df: pd.DataFrame, k: float) -> tuple[pd.Series, pd.Series]:
    """k × σ20 × √20 でラベルを引き直す（attach_rise_label の合成と同じ）。戻り値: (ラベル, 必要な上昇率)。"""
    cfg = B.RiseConfig(vol_norm_k=float(k))
    need, end_need = B.rise_thresholds(df["vol_20d"], cfg)
    need.index = df.index
    end_need.index = df.index
    ok = ((df["future_rise"] >= need) & (df["end_level"] >= end_need) & (df["uptrend_end"] == 1.0))
    determined = df["label"].notna() & need.notna()
    return ok.astype(float).where(determined), need


def build_labels(df: pd.DataFrame, ks: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """各 k のラベル列 y_<腕> と必要な上昇率 need_<腕> を付け、腕ごとの表を返す。"""
    rows = []
    for k in ks:
        a = arm_name(k)
        y, need = relabel(df, k)
        df[f"y_{a}"] = y
        df[f"need_{a}"] = need
        rows.append({"arm": a, "k": float(k), "pos_rate": float(y.mean()), "n_pos": int(y.sum()),
                     "need_median": float(need[y.notna()].median())})
    return df, pd.DataFrame(rows)


def pos_by_vol(y: pd.Series, vol: pd.Series) -> pd.Series:
    band = pd.cut(vol, E52.VOL_EDGES, labels=E52.VOL_NAMES, include_lowest=True)
    return y.groupby(band, observed=False).mean()


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #
def monthly_top(o: pd.DataFrame, n: int = MONTHLY_N) -> pd.DataFrame:
    """各月のスコア上位 n 件（月に n 銘柄買う運用）。"""
    d = o.copy()
    d["_m"] = pd.to_datetime(d["Date"]).dt.to_period("M")
    d = d.sort_values(["_m", "score"], ascending=[True, False], kind="mergesort")
    return d.groupby("_m", sort=False).head(n).reset_index(drop=True)


def monthly_stats(picks: pd.DataFrame, label_col: str = "label") -> dict:
    r20 = pd.to_numeric(picks[OUTCOME], errors="coerce")
    r40 = pd.to_numeric(picks["ret_o1_40"], errors="coerce") if "ret_o1_40" in picks.columns else pd.Series(dtype=float)
    out = {"n": int(len(picks)), "months": int(picks["_m"].nunique()) if "_m" in picks.columns else 0,
           "ret20_mean": float(r20.mean()) * 100 if len(r20) else np.nan,
           "ret20_median": float(r20.median()) * 100 if len(r20) else np.nan,
           "win": float((r20 > 0).mean()) if len(r20) else np.nan,
           "hit10": float((r20 >= 0.10).mean()) if len(r20) else np.nan,
           "bad10": float((r20 < -0.10).mean()) if len(r20) else np.nan,
           "ret40_mean": float(r40.mean()) * 100 if len(r40.dropna()) else np.nan,
           "pos": float(pd.to_numeric(picks[label_col], errors="coerce").mean()) if label_col in picks.columns else np.nan}
    return out


def monthly_by_fold(picks: pd.DataFrame) -> pd.DataFrame:
    g = picks.groupby("fold")
    return pd.DataFrame({"n": g.size(), "ret": g[OUTCOME].mean()}).reset_index()


def cross_label(o: pd.DataFrame, label_cols: list) -> dict:
    """1つのモデルのスコアを、各腕のラベルで測る（PR-AUC・リフト・ROC）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    s = o["score"].to_numpy(dtype=float)
    for c in label_cols:
        y = pd.to_numeric(o[c], errors="coerce")
        m = y.notna().to_numpy()
        yy = y[m].to_numpy(dtype=int)
        if len(yy) == 0 or yy.min() == yy.max():
            out[c] = {"pr": np.nan, "lift": np.nan, "roc": np.nan, "rate": np.nan}
            continue
        pr = float(average_precision_score(yy, s[m]))
        rate = float(yy.mean())
        out[c] = {"pr": pr, "lift": pr / rate, "roc": float(roc_auc_score(yy, s[m])), "rate": rate}
    return out


def windows_cross(o: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """窓ごとの PR-AUC / ROC-AUC を label_col のラベルで測る（MIN_WINDOW_ROWS 未満・1クラスの窓は落とす）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for f, g in o.groupby("fold"):
        y = pd.to_numeric(g[label_col], errors="coerce")
        m = y.notna().to_numpy()
        yy = y[m].to_numpy(dtype=int)
        if len(yy) < E68.MIN_WINDOW_ROWS or yy.min() == yy.max():
            continue
        sc = g.loc[m, "score"].to_numpy(dtype=float)
        rows.append({"fold": int(f), "pr": float(average_precision_score(yy, sc)), "roc": float(roc_auc_score(yy, sc)),
                     "rate": float(yy.mean())})
    return pd.DataFrame(rows, columns=["fold", "pr", "roc", "rate"])


def rank_agreement(o_a: pd.DataFrame, o_b: pd.DataFrame, pcts=(95, 90), n_month: int = MONTHLY_N) -> dict:
    """
    2つの腕のスコアの一致（同じ行で）。全体の順位相関（Spearman）、日付内の順位相関の平均（発火 MIN_BREAKS 件以上の日）、
    窓の中の上位 (100−pct)%（腕 a で選んだ組）が腕 b でも上位に入る割合と、その組の腕 b での百分位（窓の中）の中央値、
    月に n 銘柄（腕 a の各月の上位 n 件）が腕 b の各月の上位 n 件と重なる割合。
    """
    from scipy.stats import spearmanr
    m = o_a[["Code", "Date", "fold", "score"]].merge(o_b[["Code", "Date", "score"]], on=["Code", "Date"],
                                                   suffixes=("_a", "_b"))
    out = {"n": int(len(m))}
    if len(m) < 10:
        return out
    out["rho_all"] = float(spearmanr(m["score_a"], m["score_b"]).correlation)
    rhos = []
    for _, g in m.groupby("Date"):
        if len(g) >= E52.MIN_BREAKS:
            rhos.append(spearmanr(g["score_a"], g["score_b"]).correlation)
    out["rho_day"] = float(np.nanmean(rhos)) if rhos else np.nan
    out["days"] = len(rhos)
    for pct in pcts:
        inside, pct_b = [], []
        for _, g in m.groupby("fold"):
            if len(g) < E68.MIN_WINDOW_ROWS:
                continue
            ta = g["score_a"] > np.percentile(g["score_a"], pct)
            tb = g["score_b"] > np.percentile(g["score_b"], pct)
            if not ta.any():
                continue
            inside.append(float((ta & tb).sum() / ta.sum()))
            rb = g["score_b"].rank(pct=True)
            pct_b.append(float(rb[ta].median()))
        out[f"top{100-pct}_in_b"] = float(np.mean(inside)) if inside else np.nan
        out[f"top{100-pct}_pct_b"] = float(np.mean(pct_b)) if pct_b else np.nan
    pa = monthly_top(o_a, n_month)[["Code", "Date"]]
    pb = monthly_top(o_b, n_month)[["Code", "Date"]]
    both = pa.merge(pb, on=["Code", "Date"])
    out["month_overlap"] = float(len(both) / len(pa)) if len(pa) else np.nan
    return out


def attach_eval_cols(o: pd.DataFrame, df: pd.DataFrame, arms: list) -> pd.DataFrame:
    """OOF に vol_20d と各腕のラベル（y_<腕>）を付ける（評価用。Code・Date で結合）。"""
    cols = ["Code", "Date", "vol_20d"] + [f"y_{a}" for a in arms]
    return o.merge(df[cols], on=["Code", "Date"], how="left")


def diff_by_fold(base: pd.DataFrame, cur: pd.DataFrame, col: str = "ret") -> dict:
    m = base.merge(cur, on="fold", suffixes=("_b", "_c"))
    if not len(m):
        return {"n": 0, "mean": np.nan, "se": np.nan, "wins": 0}
    d = (m[f"{col}_c"] - m[f"{col}_b"]).to_numpy(dtype=float) * 100
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"n": int(len(d)), "mean": float(d.mean()), "se": float(se), "wins": int((d > 0).sum()),
            "per_fold": [float(x) for x in d]}


def fmt_diff(d: dict) -> str:
    if not d.get("n"):
        return ""
    return f" ｜ {d['mean']:+.2f} ± {d['se']:.2f}pt（上の窓 {d['wins']}/{d['n']}）"


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #
def prepare(ks: list, rebuild: bool = False):
    pf, ps = E68.fpath("frame.parquet"), E68.fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        have = [float(k) for k in stamp.get("ks", [])]
        missing = [float(k) for k in ks if not any(abs(float(k) - h) < 1e-9 for h in have)]
        if missing:
            # 表の版（built_utc）はそのまま。足りない k のラベル列だけ同じ表から引き直す（保存済みの探索・OOF はそのまま使える）
            df, tab = build_labels(df, missing)
            stamp["ks"] = have + missing
            stamp["labels"] = (stamp.get("labels") or []) + tab.to_dict(orient="records")
            df.to_parquet(pf, index=False)
            with open(ps, "w", encoding="utf-8") as fh:
                json.dump(stamp, fh, ensure_ascii=False, indent=1)
            log(f"[data] 保存済みの表（{stamp['built_utc']} 作成）に k={missing} のラベルを足した")
        else:
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
        return df, stamp
    os.makedirs(E68.OOF_DIR, exist_ok=True)
    t0 = time.time()
    df = lab.frame(rebuild=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df[df["label"].notna()].reset_index(drop=True)
    need = ["vol_20d", "future_rise", "end_level", "uptrend_end"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss}")
    y0, _ = relabel(df, K0)
    diff = int((y0 != df["label"].astype(float)).sum())
    if diff:
        raise SystemExit(f"k={K0} の引き直しが本番のラベルと {diff}件違う。合成の式がずれている")
    df, tab = build_labels(df, ks)
    log(f"[data] lab.frame {len(df):,}行 / k={K0} の引き直しは本番のラベルと全行一致 / {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_max": str(df["Date"].max().date()), "ks": [float(k) for k in ks],
             "labels": tab.to_dict(orient="records")}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    return df, stamp


def label_report(df: pd.DataFrame, ks: list) -> dict:
    arms = [arm_name(k) for k in ks]
    print(f"\n■ 1. ラベル（{len(df):,}行・{df['Date'].min().date()} 〜 {df['Date'].max().date()}。"
          f"現行 {B.DEFAULT_RISE.name}）")
    print(f"  {'腕':<16}{'正例率':>8}{'正例':>8}{'必要な上昇率の中央値':>12}  ボラの帯ごとの正例率（低→高）  "
          f"{BASE} と違う行  正例の実収益20 中央値 / 平均")
    out = {}
    for k in ks:
        a = arm_name(k)
        y = df[f"y_{a}"]
        pv = pos_by_vol(y, df["vol_20d"])
        ch = float((y != df[f"y_{BASE}"]).mean())
        r = pd.to_numeric(df.loc[y == 1, OUTCOME], errors="coerce")
        out[a] = {"k": float(k), "pos_rate": float(y.mean()), "n_pos": int(y.sum()),
                  "need_median": float(df[f"need_{a}"].median()), "pos_by_vol": [float(v) for v in pv],
                  "changed_vs_base": ch, "pos_ret20_median": float(r.median()) * 100, "pos_ret20_mean": float(r.mean()) * 100}
        print(f"  {arm_label(k):<16}{y.mean()*100:>7.2f}%{int(y.sum()):>8,}{df[f'need_{a}'].median()*100:>+11.1f}%  "
              + " / ".join(f"{v*100:.1f}%" for v in pv) + f"  {ch*100:>5.1f}%  {r.median()*100:+.1f}% / {r.mean()*100:+.1f}%")
    return out


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def run(algo: str, df: pd.DataFrame, ks: list, cols: list, cutoff, n_trials: int, stamp: dict, shifts: list,
        compute: bool = True) -> dict | None:
    arms = [arm_name(k) for k in ks]
    frames = {}
    for a in arms:
        f = df.copy()
        f["label"] = f[f"y_{a}"]
        f.attrs["built_utc"] = stamp["built_utc"]
        frames[a] = f
    recs, params = {}, {}
    for a in arms:
        recs[a] = E68.tune(algo, a, frames[a], cols, cutoff, n_trials, stamp, compute)
        if recs[a] is None:
            return None
        params[a] = recs[a]["params"]
    prod = {}
    for a in arms:
        o = E68.oof(algo, a, frames[a], cols, params[a], 0, (E68.PROD_SEED,), compute)
        if o is None:
            return None
        prod[a] = attach_eval_cols(o, df, arms)
    shifted = {}
    for sh in shifts:
        for a in arms:
            o = E68.oof(algo, a, frames[a], cols, params[a], sh, E68.SEEDS[algo], compute)
            if o is None:
                return None
            shifted[(sh, a)] = attach_eval_cols(o, df, arms)
    return {"recs": recs, "params": params, "prod": prod, "shifted": shifted}


def report(res: dict, df: pd.DataFrame, ks: list, shifts: list, stamp: dict, algo: str = "lgbm") -> dict:
    arms = [arm_name(k) for k in ks]
    names = {arm_name(k): arm_label(k) for k in ks}
    ycols = [f"y_{a}" for a in arms]
    summary = {"algo": algo, "data": stamp, "labels": label_report(df, ks), "tuning": {}, "production_oof": {}, "windows": {},
               "rank": {}, "shifts": {}, "monthly": {}, "summary": {}}

    print(f"\n■ 2. [{algo}] 探索の CV（{E68.CV_SCHEME} 5分割・自分のラベル。楽観側に出る）と本番と同じ OOF（ずらし0・種42）")
    print(f"  {'腕':<16}{'CV PR':>8}{'±SD':>8}{'CV ROC':>8}{'秒':>6}  本番と同じ OOF: "
          + "  ".join(f"[{names[b]}のラベル] PR / リフト / ROC" for b in arms))
    for a in arms:
        cv = res["recs"][a]["_cv"]
        summary["tuning"][a] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        summary["tuning"][a]["params"] = res["params"][a]
        cl = cross_label(res["prod"][a], ycols)
        summary["production_oof"][a] = {c: cl[c] for c in ycols}
        summary["production_oof"][a]["day_auc"] = float(lab.auc_in_day(res["prod"][a]))
        print(f"  {names[a]:<16}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
              f"{res['recs'][a].get('_seconds', 0):>6}  "
              + "  ".join(f"{cl[c]['pr']:.4f} / {cl[c]['lift']:.2f}x / {cl[c]['roc']:.4f}" for c in ycols))
    for a in arms:
        print(f"  {names[a]:<16}パラメータ {E68.short(res['params'][a])}")

    # 窓ごとの分離力（各腕のラベルで測る。ラベルが同じなら腕どうしを比べられる）
    rows_auc = []
    for sh in shifts:
        for a in arms:
            for lc in ycols:
                w = windows_cross(res["shifted"][(sh, a)], lc)
                for _, r in w.iterrows():
                    rows_auc.append({"shift": sh, "arm": a, "label": lc, "fold": int(r["fold"]), "pr": float(r["pr"]),
                                     "roc": float(r["roc"]), "rate": float(r["rate"])})
    auc = pd.DataFrame(rows_auc)
    if len(auc):
        auc.to_csv(E68.path(f"{algo}_auc_by_window.csv"), index=False)
        print(f"\n■ 2b. 窓ごとの分離力（ずらし {'/'.join(map(str, shifts))}か月 × 種3つの平均。同じラベルで測った {BASE} との差）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for lc in ycols:
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                b = auc[(auc["arm"] == BASE) & (auc["label"] == lc)].set_index(["shift", "fold"])[met].rename("b")
                for a in arms:
                    if a == BASE:
                        continue
                    c = auc[(auc["arm"] == a) & (auc["label"] == lc)].set_index(["shift", "fold"])[met].rename("c")
                    m = pd.concat([b, c], axis=1).dropna()
                    if not len(m):
                        continue
                    line, st = E68.pair(f"[{lc[2:]}のラベル] {nm} {a}−{BASE}", m["b"].to_numpy(), m["c"].to_numpy())
                    print(line)
                    summary["windows"][f"{lc}_{met}_{a}-{BASE}"] = st

    # 腕どうしのスコアの一致（運用者の仮説: 厳しいラベルで上位なら、緩いラベルのモデルでも上位）
    if len(arms) > 1:
        print(f"\n■ 2c. 腕どうしのスコアの一致（ずらし {'/'.join(map(str, shifts))}か月の平均。a で選んだ組が b でどう見えるか）")
        print(f"  {'a → b':<14}{'順位相関 全体':>10}{'日付内':>8}{'a 上位5% が b 上位5% にも':>14}{'同 b での百分位':>10}"
              f"{'a 上位10% が b 上位10% にも':>15}{'同 百分位':>9}{'月2件の重なり':>10}")
        rows_rank = []
        for i, a in enumerate(arms):
            for b in arms:
                if a == b:
                    continue
                vals = [rank_agreement(res["shifted"][(sh, a)], res["shifted"][(sh, b)]) for sh in shifts]
                keys = ("rho_all", "rho_day", "top5_in_b", "top5_pct_b", "top10_in_b", "top10_pct_b", "month_overlap")
                avg = {k: float(np.nanmean([v.get(k, np.nan) for v in vals])) for k in keys}
                avg["n"] = int(np.mean([v.get("n", 0) for v in vals]))
                rows_rank.append({"a": a, "b": b, **avg})
                summary["rank"][f"{a}->{b}"] = avg
                print(f"  {a + ' → ' + b:<14}{avg['rho_all']:>10.3f}{avg['rho_day']:>8.3f}{avg['top5_in_b']*100:>13.0f}%"
                      f"{avg['top5_pct_b']:>10.3f}{avg['top10_in_b']*100:>14.0f}%{avg['top10_pct_b']:>9.3f}"
                      f"{avg['month_overlap']*100:>9.0f}%")
        pd.DataFrame(rows_rank).to_csv(E68.path(f"{algo}_rank_agreement.csv"), index=False)

    rows_thr, rows_month = [], []
    for sh in shifts:
        o = {a: res["shifted"][(sh, a)] for a in arms}
        # 評価のラベルは現行（K12）でそろえる。自分のラベルは y_<腕> 列で見られる
        for a in arms:
            o[a] = o[a].copy()
            o[a]["label"] = o[a][f"y_{BASE}"]
        months = int(pd.to_datetime(o[BASE]["Date"]).dt.to_period("M").nunique())
        print(f"\n■ ずらし{sh}か月（種3つの平均。{len(o[BASE]):,}件・{months}か月）")
        for pct in (95, 90):
            print(f"  ◆ 上位{100-pct}%（前の窓の分布の閾値）: 件数（月あたり）/ 平均収益 / 全体との差 / "
                  f"窓ごとの差 ± SE（上の窓 / 最悪）/ −10%未満 / 正例率({BASE})")
            base_pf = E52._per_fold(o[BASE], pct, within=False)
            for a in arms:
                e = lab.threshold_edge(o[a], pct=pct, outcome=OUTCOME)
                pf = E52._per_fold(o[a], pct, within=False)
                bad = float((pf["bad"] * pf["n"]).sum() / pf["n"].sum()) if pf["n"].sum() else np.nan
                pos = float((pf["pos"] * pf["n"]).sum() / pf["n"].sum()) if pf["n"].sum() else np.nan
                se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"])) if e["thr_folds"] else np.nan
                line = (f"    {names[a]:<16}{e['thr_n']:>6}件（{e['thr_n'] / max(1, months):.1f}/月） {e['thr_end']:>+7.2f}% "
                        f"{e['thr_lift']:>+7.2f}pt  {e['thr_fold_mean']:>+6.2f} ± {se:.2f}（{e['thr_folds_won']}/{e['thr_folds']} / "
                        f"{e['thr_worst']:+.2f}） {bad*100:>5.1f}% {pos*100:>4.0f}%")
                d = diff_by_fold(base_pf, pf) if a != BASE else {}
                print(line + fmt_diff(d))
                rows_thr.append({"shift": sh, "arm": a, "kind": f"thr{pct}", "n": int(e["thr_n"]),
                                 "per_month": float(e["thr_n"] / max(1, months)), "ret": float(e["thr_end"]),
                                 "lift": float(e["thr_lift"]), "fold_mean": float(e["thr_fold_mean"]), "bad": bad, "pos": pos,
                                 "diff_mean": d.get("mean", np.nan), "diff_se": d.get("se", np.nan),
                                 "diff_wins": d.get("wins", 0), "diff_n": d.get("n", 0)})
            counts = E53.threshold_counts(o[BASE], pct)
            base_m = E53.matched_picks(o[BASE], counts)
            print(f"  ◆ 発火数を {BASE} の上位{100-pct}% にそろえる（各窓で同じ件数を上から）: 件数 / 平均収益 / −10%未満 / 正例率({BASE})")
            for a in arms:
                cur = E53.matched_picks(o[a], counts)
                n = int(cur["n"].sum())
                ret = float((cur["ret"] * cur["n"]).sum() / n) if n else np.nan
                bd = float((cur["bad"] * cur["n"]).sum() / n) if n else np.nan
                pos = float((cur["pos"] * cur["n"]).sum() / n) if n else np.nan
                d = diff_by_fold(base_m, cur) if a != BASE else {}
                print(f"    {names[a]:<16}{n:>6}件 {ret*100:>+7.2f}% {bd*100:>5.1f}% {pos*100:>4.0f}%" + fmt_diff(d))
                rows_thr.append({"shift": sh, "arm": a, "kind": f"match{pct}", "n": n, "ret": ret * 100, "bad": bd, "pos": pos,
                                 "fold_mean": float(cur["ret"].mean()) * 100 if len(cur) else np.nan,
                                 "diff_mean": d.get("mean", np.nan), "diff_se": d.get("se", np.nan),
                                 "diff_wins": d.get("wins", 0), "diff_n": d.get("n", 0)})
        print(f"  ◆ 窓の中の上位10%（件数をそろえる）: 平均収益 / −10%未満 / 正例率({BASE})")
        base_w = E52._per_fold(o[BASE], 90, within=True)
        for a in arms:
            cur = E52._per_fold(o[a], 90, within=True)
            d = diff_by_fold(base_w, cur) if a != BASE else {}
            print(f"    {names[a]:<16}{cur['ret'].mean()*100:>+6.2f}% {cur['bad'].mean()*100:>5.1f}% {cur['pos'].mean()*100:>4.0f}%"
                  + fmt_diff(d))
            rows_thr.append({"shift": sh, "arm": a, "kind": "top10_within", "n": int(cur["n"].sum()),
                             "ret": float(cur["ret"].mean()) * 100, "bad": float(cur["bad"].mean()), "pos": float(cur["pos"].mean()),
                             "fold_mean": float(cur["ret"].mean()) * 100,
                             "diff_mean": d.get("mean", np.nan), "diff_se": d.get("se", np.nan),
                             "diff_wins": d.get("wins", 0), "diff_n": d.get("n", 0)})
        print(f"  ◆ 日付内（発火{E52.MIN_BREAKS}件以上の日）: 1位の平均収益 / 上位2件の平均 / 順位相関 / 日数")
        for a in arms:
            w = E52.within_date_metrics(o[a])
            print(f"    {names[a]:<16}{w['top1']:>+7.2f}% {w['top2']:>+7.2f}% {w['rho']:>+6.3f} {w['days']:>5}日")
            rows_thr.append({"shift": sh, "arm": a, "kind": "within_date", **w})
        print(f"  ◆ 月に{MONTHLY_N}銘柄（各月のスコア上位{MONTHLY_N}件）: 件数 / 平均収益20 / 中央値 / 勝率 / +10%以上 / −10%未満 / "
              f"平均収益40 / 正例率({BASE}) ｜ 窓ごとの差")
        base_mo = monthly_by_fold(monthly_top(o[BASE]))
        for a in arms:
            picks = monthly_top(o[a])
            st = monthly_stats(picks, "label")
            d = diff_by_fold(base_mo, monthly_by_fold(picks)) if a != BASE else {}
            print(f"    {names[a]:<16}{st['n']:>5}件 {st['ret20_mean']:>+6.2f}% {st['ret20_median']:>+6.2f}% {st['win']*100:>5.1f}% "
                  f"{st['hit10']*100:>5.1f}% {st['bad10']*100:>5.1f}% {st['ret40_mean']:>+6.2f}% {st['pos']*100:>4.0f}%" + fmt_diff(d))
            rows_month.append({"shift": sh, "arm": a, **st, "diff_mean": d.get("mean", np.nan), "diff_se": d.get("se", np.nan),
                               "diff_wins": d.get("wins", 0), "diff_n": d.get("n", 0)})
        print("  ◆ 上位10% の中身（ボラの帯ごと: 割合 / 平均収益 / 正例率(K12)）")
        for a in arms:
            t = E52.picks_by_vol(o[a], 90)
            if len(t):
                print(f"    {names[a]:<16}" + " | ".join(
                    f"{i} {r['share']*100:.0f}% {r['ret']*100:+.1f}% {r['pos']*100:.0f}%" for i, r in t.iterrows()))

    thr = pd.DataFrame(rows_thr)
    mo = pd.DataFrame(rows_month)
    if len(thr):
        thr.to_csv(E68.path(f"{algo}_edges.csv"), index=False)
    if len(mo):
        mo.to_csv(E68.path(f"{algo}_monthly.csv"), index=False)
    summary["shifts"] = thr.to_dict(orient="records") if len(thr) else []
    summary["monthly"] = mo.to_dict(orient="records") if len(mo) else []

    if len(shifts) > 1 and len(thr):
        print(f"\n■ まとめ（切り方{len(shifts)}通りの平均。差は {BASE} に対する窓ごとの差の平均）")
        for kind, ja in (("thr95", "上位5%（閾値）"), ("thr90", "上位10%（閾値）"), ("match95", "発火数を上位5% にそろえ"),
                         ("match90", "発火数を上位10% にそろえ"), ("top10_within", "窓の中の上位10%")):
            g = thr[thr["kind"] == kind]
            if not len(g):
                continue
            print(f"  {ja}: 腕ごとの 平均収益 / −10%未満 / 月あたり件数 / 差（対 {BASE}）")
            for a in arms:
                h = g[g["arm"] == a]
                pm = f"{h['per_month'].mean():.1f}/月 " if "per_month" in h.columns and h["per_month"].notna().any() else ""
                dm = (f"{h['diff_mean'].mean():+.2f}pt（上の窓 {int(h['diff_wins'].sum())}/{int(h['diff_n'].sum())}）"
                      if a != BASE else "—")
                print(f"    {names[a]:<16}{h['ret'].mean():>+6.2f}% {h['bad'].mean()*100:>5.1f}% {pm:>8} {dm}")
                summary["summary"][f"{kind}_{a}"] = {"ret": float(h["ret"].mean()), "bad": float(h["bad"].mean()),
                                                     "diff_mean": float(h["diff_mean"].mean()) if a != BASE else 0.0,
                                                     "diff_wins": int(h["diff_wins"].sum()), "diff_n": int(h["diff_n"].sum())}
        if len(mo):
            print(f"  月に{MONTHLY_N}銘柄: 平均収益20 / 勝率 / +10%以上 / −10%未満 / 差（対 {BASE}）")
            for a in arms:
                h = mo[mo["arm"] == a]
                dm = (f"{h['diff_mean'].mean():+.2f}pt（上の窓 {int(h['diff_wins'].sum())}/{int(h['diff_n'].sum())}）"
                      if a != BASE else "—")
                print(f"    {names[a]:<16}{h['ret20_mean'].mean():>+6.2f}% {h['win'].mean()*100:>5.1f}% {h['hit10'].mean()*100:>5.1f}% "
                      f"{h['bad10'].mean()*100:>5.1f}% {dm}")
                summary["summary"][f"monthly_{a}"] = {"ret20": float(h["ret20_mean"].mean()), "win": float(h["win"].mean()),
                                                      "hit10": float(h["hit10"].mean()), "bad10": float(h["bad10"].mean()),
                                                      "diff_mean": float(h["diff_mean"].mean()) if a != BASE else 0.0,
                                                      "diff_wins": int(h["diff_wins"].sum()), "diff_n": int(h["diff_n"].sum())}
    with open(E68.path(f"{algo}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験78: 目的変数の到達しきい値 k を 1.2σ → 1.5σ にした lgbm（探索し直し）")
    ap.add_argument("--ks", default=DEFAULT_KS, help="比べる k（σ の倍数）。現行 1.2 は必ず含める")
    ap.add_argument("--algos", default=DEFAULT_ALGOS, help="この回に計算するモデル（lgbm / xgb / cat / logit。, 区切り）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--budget-min", type=float, default=290.0)
    args = ap.parse_args(argv)
    ks = [float(x) for x in args.ks.split(",") if x.strip()]
    if not any(abs(k - K0) < 1e-9 for k in ks):
        raise SystemExit(f"現行の k={K0} は比べる基準なので外せません")
    if len({arm_name(k) for k in ks}) != len(ks):
        raise SystemExit("k が近すぎて腕の名前が重なります（0.1 刻みで）")
    E68.TAG = args.tag
    E68.RUN = f"{args.tag}_k"
    E68.DEADLINE = time.time() + args.budget_min * 60 if args.budget_min > 0 else None
    E68.TIMED_OUT = False
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in E68.ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}（使えるもの {E68.ALGOS}）")
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験78 目的変数の到達しきい値 k（1.2σ → 1.5σ / 1.8σ）を変えたモデル（腕ごとに探索し直し・本番と同じ OOF・32窓・実収益）")
    print(f"  腕: " + " / ".join(arm_label(k) for k in ks) + f" / 列 {F.DEFAULT_PRESET} {len(cols)} / "
          f"探索 {args.n_trials}試行 × {E68.N_SPLITS}分割（{E68.CV_SCHEME}）/ ずらし {shifts}か月 / この回のモデル {algos} / "
          f"種 {E68.SEEDS} / 保存 {E68.RUN}_*")
    print("  特徴量は変えない（目的変数の k に連動する特徴量は無い。rise_need / end_need は特徴量に入れていない）")
    print("=" * 78)

    df, stamp = prepare(ks, args.rebuild)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df.attrs["built_utc"] = stamp["built_utc"]
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"  行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行")
    for algo in algos:
        t0 = time.time()
        res = run(algo, df, ks, cols, cutoff, args.n_trials, stamp, shifts)
        log(f"[{algo}] {'そろった' if res else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
        if res is None:
            label_report(df, ks)
            log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
            break
        report(res, df, ks, shifts, stamp, algo)
    log(f"記録: {E68.OOF_DIR}/{E68.RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
