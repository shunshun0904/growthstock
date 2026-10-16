#!/usr/bin/env python3
"""
実験100: 基準日より前のストップ高（research/price_limits.py で日足から判定。時点で分かる情報）の4列を本番の239列に足すと、
LightGBM の探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓と、**高スコアの行の成績**がどう変わるか（実験89 と同じ手順。
腕は「列」の違い。学習器は LightGBM だけ）。本番・画面は変えない。

運用者の指示（2026-10-16、実験99 §59 の「過去のストップ高は届きやすいが平均は低い」を見て）「今目的変数はより多くの上昇で
ラベル付けしているので、高予測スコアはやはりここに寄るのではないかと思ってます。平均というよりもやはり高スコアでどうなるかなので、
検証する価値はあるかなと。lgbmのみで良いのでパラメータチューニング含めた5cv+oof評価をお願いします」。

足す列（基準日 t の時点で分かるものだけ。price_limits.limit_events。どれも日足から）
  ul_t           基準日そのものがストップ高か（0/1）。実験99 では、この行の 33.6% が買ったあと20営業日にもストップ高
  ul_past20_n    前20営業日のストップ高の日数
  ul_past250_n   前250営業日のストップ高の日数。実験99 では、1回以上の候補の 16.5% が買ったあともストップ高（無い候補 2.6%）
  ul_days_since  前250営業日の中で最後のストップ高から基準日までの営業日数（無ければ欠測）
  結果の列 ul_fwd（買ったあと20営業日にストップ高があったか）は高スコアの行の集計にだけ使い、特徴量には入れない

腕（データは同じ1本。列だけが違う。実験89 と同じ）
  A  いまの本番（239列）。A の列で探索
  B  239 + 4列で探索し直し → 日曜の週次実行がやること
  C  239 + 4列を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P  対照: 4列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を A のパラメータで
  C−A が「列を足すことの効果」、P−A が「列数が増えただけの幅」、B−A が「再学習で起きること」、B−C が「探索し直しの揺れ」

ラベル（--k。1回の実行で1つ。両方回す）
  1.5  到達しきい値 1.5σ（既定）。2026-10-12 からの選定「1.5σ の3本がすべて95以上」のモデルのラベル
  1.2  1.2σ。本番の基準モデルのラベル

判定（回す前に決めた。2026-10-16）
  判定1（分離力。§7 の LightGBM 1本分）: 32窓の PR-AUC で、C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半。
      §7 は木3モデル中2つ以上の票で決めるので、LightGBM だけでは採否は決まらない。満たせば xgb・cat でも同じ手順を回して票を
      取る（運用者が決める）。満たさなければ、この列は閉じる
  判定2（高スコア。運用者の関心）: 窓ごとに、スコアが窓の上位5%の行の正例率で、同じ3条件（C−A の平均が正・P−A を上回る・
      C が上の窓が過半）
  参考（判定に使わない）: 上位5%・10% の行の ret_o1_20 の平均と、買ったあとのストップ高の割合（ul_fwd）。本番と同じ OOF の
      PR-AUC・リフト・ROC・日内 AUC・上位10% の超過収益（e25.metrics）。探索の CV。B−A・B−C。窓ごとの値の一覧
  評価の行は全行（4列はどの行にも付く。前250営業日の日足が無い行だけ回数が少なめに出る）

出すもの: 列の付き方（件数・割合だけ）、探索の CV（分割ごと）、本番と同じ OOF、窓ごとの差の平均・SE・上の窓の数、判定1・2、
窓ごとの値の一覧。公開ログには件数・割合・日付・精度だけ（銘柄・値は出さない）。

    exp=e100_limit_up_ab.py args="--k 1.5"
    exp=e100_limit_up_ab.py args="--k 1.2"
充足だけ見る: exp=e100_limit_up_ab.py args="--dry"
試運転:       exp=e100_limit_up_ab.py args="--tag e100smoke --n-trials 2 --shifts 0"

Run Experiment は同じ実験を同時に1本しか走らせないので、--k 1.5 と --k 1.2 は1本ずつ起動する。表（4列を付けた frame）は
1回目に作って research/_data/oof/<tag>_* に置き、2回目は actions/cache で引き継いだそれを使う（どちらも同じ行で比べる）。
--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。
"""
from __future__ import annotations

import argparse
import hashlib
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
import live_track as LT  # noqa: E402
import price_limits as PL  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, keys_of, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 本番の retrain-weekly.yml と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
ARMS = ("A", "B", "C", "P")
COLS_OF = {"A": "base", "B": "new", "C": "new", "P": "perm"}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}
LABELS = {"A": "A いまの本番（239列）", "B": "B 239 + ストップ高 4列・探索し直し",
          "C": "C 239 + ストップ高 4列・A のパラメータ", "P": "P 対照（4列を日付内で入れ替え）・A のパラメータ"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 足す列（時点で分かるものだけ）と、結果の列（特徴量にしない）
FEATURES = ("ul_t", "ul_past20_n", "ul_past250_n", "ul_days_since")
OUTCOME_COL = "ul_fwd"
RENAME = {"ul_past_n": "ul_past250_n"}
HOLD = LT.HOLD
PAST = 250
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261016
MIN_WINDOW_ROWS = 100
KS = (1.5, 1.2)
#: 判定2 の「高スコア」= 窓の中でスコアが上位この割合
TOP_Q = 0.05
TOP_QS = (0.05, 0.10)
TAG = "e100"
RUN = "e100_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約（ラベルごと）。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def k_tag(k: float) -> str:
    return f"k{int(round(k * 10)):02d}"


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


def with_label(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """k のラベルに引き直した表（1.2σ はデータセットのラベルのまま）。ラベルの決まらない行は落とす。"""
    if abs(k - B.VOL_NORM_K) < 1e-9:
        return df[df["label"].notna()].reset_index(drop=True)
    out = df.copy()
    out["label"] = B.rise_label_for_k(out, k)
    return out[out["label"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 列を付ける
# --------------------------------------------------------------------------- #
def attach_features(keys: pd.DataFrame, bars: pd.DataFrame, hold: int = HOLD, past: int = PAST) -> tuple[pd.DataFrame, dict]:
    """
    (Code, Date) ごとに、足す4列と結果の列 ul_fwd を日足から付けて返す（keys と同じ行の並び）。記録に判定の集計を入れる。
    """
    flags = PL.flag_limits(bars)
    info = PL.flags_summary(flags)
    arrays = PL.code_arrays(flags)
    ev = PL.limit_events(arrays, keys[["Code", "Date"]], hold=hold, past=past).rename(columns=RENAME)
    cols = list(FEATURES) + [OUTCOME_COL, "past_bars"]
    out = keys[["Code", "Date"]].copy()
    out["Code"] = out["Code"].astype(str)
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.merge(ev[["Code", "Date"] + cols], on=["Code", "Date"], how="left")
    info["attached"] = int(out["ul_past250_n"].notna().sum())
    info["past_full"] = int((out["past_bars"] >= past).sum())
    return out.drop(columns=["past_bars"]), info


def prepare(rebuild: bool = False):
    """lab.frame に4列と ul_fwd を付けた表（全行。ラベルは 1.2σ のまま）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    bars = LT.load_bars(lab.DATA_DIR, start=frame["Date"].min() - pd.Timedelta(days=500), extra=("H", "L", "C"))
    feat, info = attach_features(frame[["Code", "Date"]], bars, HOLD, PAST)
    df = pd.concat([frame.reset_index(drop=True), feat.drop(columns=["Code", "Date"]).reset_index(drop=True)], axis=1)
    yr = df["Date"].dt.year
    log(f"[data] 4列を付けた {len(df):,}行（付いた行 {info['attached']:,}・前250営業日がそろう行 {info['past_full']:,}）/ "
        f"日足 {info['rows']:,}行・ストップ高の日 {info['ul']:,} / {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())},
             "ul_t_by_year": {str(y): int(((df["ul_t"] == 1) & (yr == y)).sum()) for y in sorted(yr.unique())},
             "past_any_by_year": {str(y): int(((df["ul_past250_n"] > 0) & (yr == y)).sum()) for y in sorted(yr.unique())},
             "flags": info}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def coverage_report(df: pd.DataFrame, stamp: dict, k: float) -> dict:
    """列の付き方と、列の値ごとのラベルの正例率・買ったあとのストップ高の割合（件数と割合だけ）。"""
    fl = stamp.get("flags", {})
    print(f"\n■ 0. 列の付き方（表は {stamp['built_utc']} 作成・{stamp['rows']:,}行）")
    if fl:
        print(f"  日足 {fl.get('rows', 0):,}行のうち判定できた {fl.get('known_share', float('nan')):.1f}%・ストップ高の日 {fl.get('ul', 0):,}"
              f"（高値がちょうど上限 {fl.get('exact_among_ul', float('nan')):.1f}%）。4列の付いた行 {fl.get('attached', 0):,}・"
              f"前250営業日がそろう行 {fl.get('past_full', 0):,}")
    yrs = sorted(stamp["rows_by_year"])
    print("  年別（全行・基準日がストップ高の行・前250営業日に1回以上の行）: " + "  ".join(
        f"{y} {stamp['rows_by_year'][y]:,}・{stamp['ul_t_by_year'].get(y, 0):,}・{stamp['past_any_by_year'].get(y, 0):,}"
        for y in yrs))
    out = {"label_k": k}
    lab_ = df["label"]
    ulf = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    print(f"  ラベル {k:.1f}σ の正例率と、買ったあとのストップ高の割合（列の値ごと）:")
    groups = (("ul_t = 1", df["ul_t"] == 1), ("ul_t = 0", df["ul_t"] == 0),
              ("ul_past250_n ≥ 1", df["ul_past250_n"] >= 1), ("ul_past250_n = 0", df["ul_past250_n"] == 0),
              ("ul_past20_n ≥ 1", df["ul_past20_n"] >= 1), ("ul_days_since ≤ 20", df["ul_days_since"] <= 20),
              ("全行", pd.Series(True, index=df.index)))
    for name, m in groups:
        m = m.fillna(False) & lab_.notna()
        r = lab_[m].mean() if m.any() else float("nan")
        u = ulf[m].mean() if ulf[m].notna().any() else float("nan")
        print(f"    {name:<18}{int(m.sum()):>7,}行（{m.sum() / max(lab_.notna().sum(), 1) * 100:5.1f}%）  正例率 {r*100:5.1f}%  "
              f"買ったあとのストップ高 {u*100:5.1f}%")
        out[name] = {"n": int(m.sum()), "pos_rate": float(r), "ul_fwd": float(u)}
    desc = df[list(FEATURES)].describe().loc[["mean", "50%", "max"]]
    print("  4列の平均 / 中央 / 最大: " + "  ".join(f"{c} {desc.loc['mean', c]:.2f} / {desc.loc['50%', c]:.0f} / {desc.loc['max', c]:.0f}"
                                            for c in FEATURES))
    out["missing_days_since"] = float(df["ul_days_since"].isna().mean())
    return out


# --------------------------------------------------------------------------- #
# 探索（本番の週次実行と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, compute: bool = True):
    p = path(f"params_{ALGO}_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not compute or not time_left():
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    log(f"  [{ALGO} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
    params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                         embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier",
                         verbose=False)
    rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [{ALGO} {arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）"
        f" / ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #
def oof(colset: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds, compute: bool = True):
    """列の組（base / new / perm）× パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{ALGO}_c{colset}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and (not compute or not time_left()):
        return None
    folds = E41.folds_for(frame["Date"], shift) if missing else None
    parts = []
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o = E41.oof_folds(ALGO, frame, cols, params, sd, folds)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {ALGO} 列{colset} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def with_outcome(o: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """out-of-fold に結果の列 ul_fwd を (Code, Date) で付ける。"""
    if OUTCOME_COL in o.columns:
        return o
    f = frame[["Code", "Date", OUTCOME_COL]].drop_duplicates(["Code", "Date"])
    return o.merge(f, on=["Code", "Date"], how="left")


def windows_on(o: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率。MIN_WINDOW_ROWS 未満の窓は落とす。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    if len(o) == 0:
        return pd.DataFrame(columns=cols)
    n = o.groupby("fold").size()
    sub = o[o["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def top_by_window(o: pd.DataFrame, q: float = TOP_Q) -> pd.DataFrame:
    """
    窓ごとに、スコアが窓の上位 q の行（スコアの (1−q) 分位以上。同点は含む）の件数・正例率・ret_o1_20 の平均・買ったあとの
    ストップ高の割合。MIN_WINDOW_ROWS 未満の窓は落とす。
    """
    rows = []
    for f, g in o.groupby("fold"):
        if len(g) < MIN_WINDOW_ROWS:
            continue
        s = g["score"].to_numpy(dtype=float)
        top = g[s >= np.quantile(s, 1.0 - q)]
        r20 = pd.to_numeric(top["ret_o1_20"], errors="coerce") if "ret_o1_20" in top.columns else pd.Series(dtype=float)
        ulf = pd.to_numeric(top[OUTCOME_COL], errors="coerce") if OUTCOME_COL in top.columns else pd.Series(dtype=float)
        rows.append({"fold": int(f), "n_top": int(len(top)), "pos_top": float(top["label"].mean()),
                     "ret_top": float(r20.mean() * 100) if r20.notna().any() else float("nan"),
                     "ul_top": float(ulf.mean() * 100) if ulf.notna().any() else float("nan")})
    return pd.DataFrame(rows, columns=["fold", "n_top", "pos_top", "ret_top", "ul_top"])


def run(frames: dict, colsets: dict, cutoff, n_trials: int, stamp: dict, shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(arm, frames[COLS_OF[arm]], colsets[COLS_OF[arm]], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod, wins, tops = {}, {}, {}
    for arm in ARMS:
        cs = COLS_OF[arm]
        o = oof(cs, frames[cs], colsets[cs], params[arm], 0, (PROD_SEED,), compute)
        if o is None:
            return None
        prod[arm] = with_outcome(o, frames[cs])
    for sh in shifts:
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = oof(cs, frames[cs], colsets[cs], params[arm], sh, SEEDS, compute)
            if o is None:
                return None
            o = with_outcome(o, frames[cs])
            wins[(sh, arm)] = windows_on(o)
            tops[(sh, arm)] = {q: top_by_window(o, q) for q in TOP_QS}
    return {"recs": recs, "params": params, "prod": prod, "wins": wins, "tops": tops}


def window_table(results: dict, shifts: list) -> pd.DataFrame:
    """窓ごとの値（ずらし × 窓）。4腕がそろった窓だけ。PR / ROC と、上位5%・10% の正例率・収益・ストップ高。"""
    rows = []
    for sh in shifts:
        w = None
        for arm in ARMS:
            x = results["wins"][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            if arm != "A":
                x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
            for q in TOP_QS:
                t = results["tops"][(sh, arm)][q]
                qs = f"q{int(round(q * 100)):02d}"
                t = t.rename(columns={c: f"{c}_{qs}_{arm}" for c in ("n_top", "pos_top", "ret_top", "ul_top")})
                x = x.merge(t, on="fold", how="inner")
            w = x if w is None else w.merge(x, on="fold")
        if w is None or len(w) == 0:
            continue
        for _, r in w.iterrows():
            rec = {"shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"], "n": int(r["n"]),
                   "pos": float(r["pos"])}
            for arm in ARMS:
                rec[f"pr_{arm}"] = float(r[f"pr_{arm}"])
                rec[f"roc_{arm}"] = float(r[f"roc_{arm}"])
                for q in TOP_QS:
                    qs = f"q{int(round(q * 100)):02d}"
                    for c in ("n_top", "pos_top", "ret_top", "ul_top"):
                        rec[f"{c}_{qs}_{arm}"] = float(r[f"{c}_{qs}_{arm}"])
            rows.append(rec)
    return pd.DataFrame(rows)


def judge_lines(name: str, s: pd.DataFrame, col: str) -> tuple[dict, str]:
    """列 col（腕ごとの値）で §7 の3条件。"""
    j = judge7((s[f"{col}_C"] - s[f"{col}_A"]).to_numpy(dtype=float), (s[f"{col}_P"] - s[f"{col}_A"]).to_numpy(dtype=float))
    line = (f"  {name:<10}C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / P−A {j['mean_pa']:>+8.4f} "
            f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
            f"→ {'満たす' if j['pass'] else '満たさない'}")
    return j, line


def report(results: dict, shifts: list, stamp: dict, cover: dict, k: float) -> dict:
    print("\n" + "=" * 78)
    print(f"ラベル {k:.1f}σ / LightGBM / 列: {', '.join(FEATURES)}")
    summary = {"data": stamp, "coverage": cover, "label_k": k, "perm_seed": PERM_SEED, "features": list(FEATURES),
               "tuning": {}, "production_oof": {}, "windows": {}, "judge1": {}, "judge2": {}, "tops": {}, "per_window": []}
    r = results

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC")
    for arm in ("A", "B"):
        cv = r["recs"][arm]["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}{r['recs'][arm].get('_seconds', 0):>6}  "
              + " / ".join(f"{v:.4f}" for v in fs))
        summary["tuning"][arm] = {kk: cv.get(kk) for kk in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
    pa, pb = r["recs"]["A"]["params"], r["recs"]["B"]["params"]
    fa = r["recs"]["A"]["_cv"].get("fold_scores") or []
    fb = r["recs"]["B"]["_cv"].get("fold_scores") or []
    if fa and len(fa) == len(fb):
        d = np.array(fb) - np.array(fa)
        print(f"  B−A {d.mean():>+8.4f}{'':>8}{r['recs']['B']['_cv']['mean_roc_auc'] - r['recs']['A']['_cv']['mean_roc_auc']:>+8.4f}"
              f"{'':>6}  分割ごと " + " / ".join(f"{v:+.4f}" for v in d) + f"（B が上 {(d > 0).sum()}/{len(d)}）")
    print("  選ばれたパラメータ: " + ("A と B で同じ" if params_hash(pa) == params_hash(pb) else f"A {short(pa)} / B {short(pb)}"))

    print(f"\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。全行）")
    print(f"  {'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
    ms = {}
    for arm in ARMS:
        o = r["prod"][arm]
        m = metrics(o)
        ms[arm] = m
        summary["production_oof"][arm] = {kk: float(v) for kk, v in m.items()}
        print(f"  {arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
              f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
              f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
    for x, y in PAIRS:
        print(f"  {y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
              f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    s = window_table(r, shifts)
    if not len(s):
        print("\n  窓がそろっていない")
        return summary
    s.to_csv(path("by_window.csv"), index=False)
    print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り・種3つの平均。{len(s)}窓）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for met, nm in (("pr", "PR"), ("roc", "ROC")):
        for x, y in PAIRS:
            line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(dtype=float), s[f"{met}_{y}"].to_numpy(dtype=float))
            print(line)
            summary["windows"][f"{met}_{y}-{x}"] = st

    print("\n■ 4. 判定1（§7 の LightGBM 1本分。窓の PR-AUC）: 平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
    j1, line = judge_lines("PR-AUC", s, "pr")
    print(line)
    summary["judge1"] = j1
    print("  → " + ("満たす。§7 の票は木3モデル中2つ以上なので、xgb・cat でも同じ手順を回すかを運用者が決める" if j1["pass"]
                   else "満たさない。LightGBM では列を足す効果が対照と見分けられない"))

    print(f"\n■ 5. 判定2（高スコア。窓の中でスコアが上位 {TOP_Q*100:.0f}% の行の正例率）: 同じ3条件")
    for q in TOP_QS:
        qs = f"q{int(round(q * 100)):02d}"
        head = f"上位{q*100:.0f}%"
        print(f"  {head}（1窓あたり {s[f'n_top_{qs}_A'].mean():.0f}件）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for c, nm in (("pos_top", "正例率"), ("ret_top", "ret20 %"), ("ul_top", "ストップ高 %")):
            for x, y in PAIRS[:3]:
                a_ = s[f"{c}_{qs}_{x}"].to_numpy(dtype=float)
                b_ = s[f"{c}_{qs}_{y}"].to_numpy(dtype=float)
                ok = np.isfinite(a_) & np.isfinite(b_)
                if ok.sum() < 2:
                    continue
                line, st = pair(f"{nm} {y}−{x}", a_[ok], b_[ok])
                print(line)
                summary["tops"][f"{qs}_{c}_{y}-{x}"] = st
        j, line = judge_lines(f"正例率 {head}", s, f"pos_top_{qs}")
        summary["judge2"][qs] = j
        print(line + ("（判定2）" if abs(q - TOP_Q) < 1e-9 else "（参考）"))
    j2 = summary["judge2"][f"q{int(round(TOP_Q * 100)):02d}"]
    print("  → 判定2: " + ("満たす" if j2["pass"] else "満たさない"))

    print(f"\n■ 6. 窓ごとの値（PR-AUC と上位{TOP_Q*100:.0f}% の正例率。差は「後 − 前」）")
    for sh in shifts:
        g = s[s["shift"] == sh].sort_values("fold")
        if not len(g):
            continue
        qs = f"q{int(round(TOP_Q * 100)):02d}"
        print(f"  ずらし{sh}か月（{len(g)}窓）")
        print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'A':>8}{'B':>8}{'C':>8}{'P':>8}{'C−A':>9}{'P−A':>9}"
              f"{'上位A':>7}{'上位C':>7}{'上位P':>7}")
        for _, rr in g.iterrows():
            print(f"    {int(rr['fold']):>2}  {pd.Timestamp(rr['start']).date()}〜{pd.Timestamp(rr['end']).date()}"
                  f"{int(rr['n']):>7,}{rr['pos']*100:>6.1f}%{rr['pr_A']:>8.4f}{rr['pr_B']:>8.4f}{rr['pr_C']:>8.4f}{rr['pr_P']:>8.4f}"
                  f"{rr['pr_C'] - rr['pr_A']:>+9.4f}{rr['pr_P'] - rr['pr_A']:>+9.4f}"
                  f"{rr[f'pos_top_{qs}_A']*100:>6.1f}%{rr[f'pos_top_{qs}_C']*100:>6.1f}%{rr[f'pos_top_{qs}_P']*100:>6.1f}%")
    summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n■ まとめ（ラベル {k:.1f}σ）: 判定1（PR-AUC）{'満たす' if j1['pass'] else '満たさない'}（C−A {j1['mean_ca']:+.4f}・P−A "
          f"{j1['mean_pa']:+.4f}・上の窓 {j1['wins']}/{j1['n']}）/ 判定2（上位{TOP_Q*100:.0f}% の正例率）"
          f"{'満たす' if j2['pass'] else '満たさない'}（C−A {j2['mean_ca']*100:+.2f}pt・P−A {j2['mean_pa']*100:+.2f}pt・上の窓 "
          f"{j2['wins']}/{j2['n']}）")
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験100: 基準日より前のストップ高の4列を本番の239列に足す前後（LightGBM）")
    ap.add_argument("--k", type=float, default=KS[0], help="ラベルの到達しきい値（σ の倍数。1.5 か 1.2）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    if not any(abs(args.k - k) < 1e-9 for k in KS):
        raise SystemExit(f"--k は {KS} のどれか（実際: {args.k}）")
    TAG = args.tag
    RUN = f"{TAG}_{k_tag(args.k)}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    new = list(FEATURES)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"本番の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new, "perm": base + new}

    print("=" * 78)
    print("実験100 基準日より前のストップ高の4列を本番の239列に足す前後（LightGBM。5分割 CV の探索 + OOF + 32窓・対照 P・高スコアの行）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + {', '.join(new)} → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  ラベル {args.k:.1f}σ / 探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / "
          f"種 {SEEDS} / 対照の種 {PERM_SEED} / 高スコア = 窓の上位 {TOP_Q*100:.0f}% / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in colsets["new"] + [OUTCOME_COL] if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    df = with_label(full, args.k)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cover = coverage_report(df, stamp, args.k)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（ラベル {args.k:.1f}σ の正例率 {df['label'].mean()*100:.1f}%・{df['Date'].min().date()} 〜 "
          f"{df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では {(df['Date'] <= cutoff).sum():,}行")
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: {len(new)}列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm}

    t0 = time.time()
    results = run(frames, colsets, cutoff, args.n_trials, stamp, shifts, compute=True)
    log(f"[{ALGO}] {'そろった' if results else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    if results:
        report(results, shifts, stamp, cover, args.k)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
