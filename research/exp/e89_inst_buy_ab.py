#!/usr/bin/env python3
"""
実験89: 機関投資家（保有目的が運用）の直近の買いのフラグ2列（research/inst_features.py の inst_buy_20・inst_buy_60）
を本番の239列に足すと、木3モデル + ロジスティック回帰の探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓が
どう変わるか（実験67・68 と同じ手順。腕は「列」の違い）。

運用者の依頼（2026-10-13）「特徴量として、機関投資家による買いが直近であったかどうかのフラグを入れたいです」と、
定義の回答（同日。大量保有報告書の実測 research/exp/probe_lvs_holders.py を見て）: 範囲は「運用目的のみ」、買いは
「割合が増えた・新規5%超」、直近は20日と60日を「どちらも特徴量として追加する。」。列の中身は research/inst_features.py。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定
（research/features.py・research/lgbm_params.json・research/multi_params.json）には書かない。

腕（データは同じ1本。列だけが違う）
  A  いまの本番（239列）。A の列で探索
  B  239 + 2列で探索し直し → 日曜の週次実行がやること
  C  239 + 2列を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P  対照: 2列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を A のパラメータで。§7 の「対照 P の改善を
     上回る」に使う
  C−A が「列を足すことの効果」、P−A が「列数が増えただけの幅」、B−A が「再学習で起きること」、B−C が「探索し直しの揺れ」

ラベル（--k。1回の実行で1つ。両方回す）
  1.5  到達しきい値 1.5σ（既定）。2026-10-12 からの選定「1.5σ の3本がすべて95以上」のモデルのラベル
  1.2  1.2σ。本番の基準モデル（画面の帯・順位・シート）と、§7 がもともと対象にしているモデルのラベル
  ラベル以外（列・探索・窓・種）は同じ。1.5σ はデータセットの材料から引き直す（B.rise_label_for_k。週次の学習と同じ）

行
  学習は全行（本番と同じ。大量保有報告書の取り込みの始まりより前の行は inst_buy_* が欠測で、モデルは欠測として扱う）。
  評価は2通り:
    covered  inst_buy_60 が分かる行（取り込みの始まりから60暦日がそろった後）→ 判定はこちら（回す前に決めた）
    all      全行（参考）

判定（回す前に決めた。2026-10-13）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）を、ラベルごと・評価の行ごとに当てる: 木3モデル（lgbm / xgb / cat）
  のうち2つ以上で、窓ごとの PR-AUC の差 C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）。
  本番に入れる案として運用者に出すのは、1.5σ と 1.2σ の両方の covered で §7 を満たすとき。片方だけなら、結果を見せて
  運用者が決める。logit は同じ表に並べる（§7 の票には入れない。決定的なので種は1つ）。B は日曜がやることの参考。

出すもの: 列の付き方（件数・割合だけ）、探索の CV（分割ごと）、本番と同じ OOF（種42・ずらし0）、窓ごとの差の平均・SE・
上の窓の数、§7 の判定、**窓ごとの値の一覧**（運用者の指示「oof だけでなく、各cv（異なる窓）でもみたいです」。窓の期間・
件数・正例率・A/B/C/P の PR-AUC と差）。公開ログには件数・割合・日付・精度だけ（銘柄・保有者の名前・値は出さない）。

    exp=e89_inst_buy_ab.py args="--k 1.5 --algos lgbm"
    exp=e89_inst_buy_ab.py args="--k 1.5 --algos xgb"
    exp=e89_inst_buy_ab.py args="--k 1.5 --algos cat"
    exp=e89_inst_buy_ab.py args="--k 1.5 --algos logit"
    （同じく --k 1.2 で4回）
充足だけ見る: exp=e89_inst_buy_ab.py args="--dry"
試運転:       exp=e89_inst_buy_ab.py args="--tag e89smoke --algos lgbm --n-trials 2 --shifts 0"

Run Experiment は同じ実験を同時に1本しか走らせず、待ちに置けるのも1本だけ（3本目を入れると待ちの1本が取り消される）
なので、1本ずつ起動する。表（inst_buy_* を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、以降の回は
actions/cache で引き継いだそれを使う（どの回も同じ行で比べる）。--budget-min（既定 290分）を過ぎたら新しい計算を
始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。
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
import extra_features as XF  # noqa: E402
import features as F  # noqa: E402
import inst_features as I  # noqa: E402
import lab  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, keys_of, on_rows, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 本番の retrain-weekly.yml と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGOS = ("lgbm", "xgb", "cat", "logit")
#: §7 の票は木3モデル。logit は同じ表に並べて参考にする
JUDGE_ALGOS = ("lgbm", "xgb", "cat")
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123), "logit": (42,)}
PROD_SEED = 42
ARMS = ("A", "B", "C", "P")
COLS_OF = {"A": "base", "B": "new", "C": "new", "P": "perm"}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A"}
LABELS = {"A": "A いまの本番（239列）", "B": "B 239 + inst 2列・探索し直し",
          "C": "C 239 + inst 2列・A のパラメータ", "P": "P 対照（inst を日付内で入れ替え）・A のパラメータ"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261013
MIN_WINDOW_ROWS = 100
KS = (1.5, 1.2)
#: 判定に使う評価の行（回す前に決めた）
PRIMARY_ROWS = "covered"
#: 列が分かる行の目印（窓の長いほう。こちらが分かれば短いほうも分かる）
COVER_COL = "inst_buy_60"
TAG = "e89"
RUN = "e89_k15"
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


def eval_keysets(df: pd.DataFrame) -> dict:
    """評価する行の組。全行と、inst_buy_60 が分かる行。"""
    return {"all": None, "covered": keys_of(df[df[COVER_COL].notna()])}


def with_label(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """k のラベルに引き直した表（1.2σ はデータセットのラベルのまま）。ラベルの決まらない行は落とす。"""
    if abs(k - B.VOL_NORM_K) < 1e-9:
        return df[df["label"].notna()].reset_index(drop=True)
    out = df.copy()
    out["label"] = B.rise_label_for_k(out, k)
    return out[out["label"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に inst_buy_* を付けた表（全行。ラベルは 1.2σ のまま）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}・"
            f"列の分かる行 {stamp['covered']:,}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    inst, st = I.attach(frame[["Code", "Date"]], XF.DATA_DIR)
    if not st.get("events"):
        raise SystemExit("運用目的の保有者の買いを含む報告書が0件。大量保有報告書（lvshld）の取り込みを確かめる")
    df = pd.concat([frame.reset_index(drop=True), inst.reset_index(drop=True)], axis=1)
    have = df[COVER_COL].notna()
    yr = df["Date"].dt.year
    log(f"[data] inst_buy_* を付けた {len(df):,}行（列の分かる行 {have.sum():,}・{have.mean()*100:.1f}%）/ "
        f"{time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "covered": int(have.sum()), "covered_from": str(df.loc[have, "Date"].min().date()) if have.any() else None,
             "covered_by_year": {str(y): int((have & (yr == y)).sum()) for y in sorted(yr.unique())},
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())},
             "ones_by_year": {c: {str(y): int(((df[c] == 1) & (yr == y)).sum()) for y in sorted(yr.unique())}
                              for c in I.COLUMNS},
             "inst": st}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def coverage_report(df: pd.DataFrame, stamp: dict, k: float) -> dict:
    """列の付き方（年別）と、1 の行の正例率（件数と割合だけ）。"""
    st = stamp.get("inst", {})
    print(f"\n■ 0. 列の付き方（表は {stamp['built_utc']} 作成）")
    print(I.describe(st, stamp["rows"]))
    print(f"  列の分かる行（{COVER_COL} が欠測でない）: {stamp['covered']:,} / {stamp['rows']:,}"
          f"（{stamp['covered_from']} から）")
    yrs = sorted(stamp["rows_by_year"])
    print("  年別（分かる行 / 全行・1 の行 20日 / 60日）: " + "  ".join(
        f"{y} {stamp['covered_by_year'].get(y, 0):,}/{stamp['rows_by_year'][y]:,}・"
        f"{stamp['ones_by_year']['inst_buy_20'].get(y, 0):,}/{stamp['ones_by_year']['inst_buy_60'].get(y, 0):,}"
        for y in yrs))
    out = {"label_k": k}
    lab_ = df["label"]
    print(f"  ラベル {k:.1f}σ の正例率（列の値ごと。分かる行の中で）:")
    for c in I.COLUMNS:
        m = df[c].notna() & lab_.notna()
        on, off = m & (df[c] == 1), m & (df[c] == 0)
        r1 = lab_[on].mean() if on.any() else float("nan")
        r0 = lab_[off].mean() if off.any() else float("nan")
        print(f"    {c}: 1 の行 {int(on.sum()):,}（{on.sum() / max(m.sum(), 1) * 100:.1f}%）の正例率 {r1*100:.1f}% / "
              f"0 の行 {int(off.sum()):,} の正例率 {r0*100:.1f}%")
        out[c] = {"ones": int(on.sum()), "zeros": int(off.sum()), "pos_rate_1": float(r1), "pos_rate_0": float(r0)}
    return out


# --------------------------------------------------------------------------- #
# 探索（本番の週次実行と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(algo: str, arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int,
         stamp: dict, compute: bool = True):
    p = path(f"params_{algo}_{arm}.json")
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
    if algo == "lgbm":
        log(f"  [{algo} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
        params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier",
                             verbose=False)
        rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    else:
        log(f"  [{algo} {arm}] 探索（tuning_multi.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列・"
            f"前処理 {TM.preprocess_version(algo) or 'なし'}）")
        TM.STUDY_DB = path(f"optuna_{arm}.db")
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False, scheme=CV_SCHEME,
                      objective=CV_OBJECTIVE)
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [{algo} {arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）"
        f" / ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #
def oof(algo: str, colset: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        compute: bool = True):
    """列の組（base / new / perm）× パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_c{colset}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
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
        o = E41.oof_folds(algo, frame, cols, params, sd, folds)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} 列{colset} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame, keys: pd.MultiIndex | None) -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率。keys があればその行だけで、MIN_WINDOW_ROWS 未満の窓は落とす。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    sub = o if keys is None else on_rows(o, keys)
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def run(algo: str, frames: dict, keysets: dict, colsets: dict, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frames[COLS_OF[arm]], colsets[COLS_OF[arm]], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod = {rs: {} for rs in keysets}
    for arm in ARMS:
        cs = COLS_OF[arm]
        o = oof(algo, cs, frames[cs], colsets[cs], params[arm], 0, (PROD_SEED,), compute)
        if o is None:
            return None
        for rs, keys in keysets.items():
            prod[rs][arm] = o if keys is None else on_rows(o, keys)
    wins = {rs: {} for rs in keysets}
    for sh in shifts:
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = oof(algo, cs, frames[cs], colsets[cs], params[arm], sh, SEEDS[algo], compute)
            if o is None:
                return None
            for rs, keys in keysets.items():
                wins[rs][(sh, arm)] = windows_on(o, keys)
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


def window_table(results: dict, algos: list, shifts: list, rs: str) -> pd.DataFrame:
    """窓ごとの値（モデル × ずらし × 窓）。4腕がそろった窓だけ。"""
    rows = []
    for a in algos:
        for sh in shifts:
            w = None
            for arm in ARMS:
                x = results[a]["wins"][rs][(sh, arm)]
                x = x.rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                if arm != "A":
                    x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
                w = x if w is None else w.merge(x, on="fold")
            if w is None or len(w) == 0:
                continue
            for _, r in w.iterrows():
                rows.append({"algo": a, "shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"],
                             "n": int(r["n"]), "pos": float(r["pos"]),
                             **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    return pd.DataFrame(rows)


def report(results: dict, shifts: list, stamp: dict, keysets: dict, cover: dict, k: float) -> dict:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"ラベル {k:.1f}σ / 結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "coverage": cover, "label_k": k, "perm_seed": PERM_SEED, "primary_rows": PRIMARY_ROWS,
               "tuning": {}, "production_oof": {}, "windows": {}, "judge7": {}, "per_window": {}}
    titles = {"all": "全行", "covered": f"列の分かる行（{stamp.get('covered_from')} から）"}
    sec = {rs: "ab"[i] for i, rs in enumerate(keysets)}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC / パラメータ")
    for a in algos:
        r = results[a]
        for arm in ("A", "B"):
            cv = r["recs"][arm]["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{r['recs'][arm].get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {kk: cv.get(kk) for kk in
                                               ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        pa, pb = r["recs"]["A"]["params"], r["recs"]["B"]["params"]
        fa = r["recs"]["A"]["_cv"].get("fold_scores") or []
        fb = r["recs"]["B"]["_cv"].get("fold_scores") or []
        if fa and len(fa) == len(fb):
            d = np.array(fb) - np.array(fa)
            print(f"  {a:<7}B−A {d.mean():>+8.4f}{'':>8}"
                  f"{r['recs']['B']['_cv']['mean_roc_auc'] - r['recs']['A']['_cv']['mean_roc_auc']:>+8.4f}"
                  f"{'':>6}  分割ごと " + " / ".join(f"{v:+.4f}" for v in d)
                  + f"（B が上 {(d > 0).sum()}/{len(d)}）")
        print(f"  {'':<7}選ばれたパラメータ: " + ("A と B で同じ" if params_hash(pa) == params_hash(pb)
                                              else f"A {short(pa)} / B {short(pb)}"))

    for rs in keysets:
        print(f"\n■ 2{sec[rs]}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。{titles[rs]}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {kk: float(v) for kk, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    for rs in keysets:
        s = window_table(results, algos, shifts, rs)
        tables[rs] = s
        if not len(s):
            continue
        s.to_csv(path(f"auc_by_window_{rs}.csv"), index=False)
        note = f"・{MIN_WINDOW_ROWS}件以上の窓" if keysets[rs] is not None else ""
        print(f"\n■ 3{sec[rs]}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。"
              f"logit は種1つ、ほかは種3つの平均。{titles[rs]}{note}）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = st
        print(f"\n■ 4{sec[rs]}. §7（2026-09-26 改訂）の判定（窓の PR-AUC。{titles[rs]}"
              f"{'。判定に使う行（回す前に決めた）' if rs == PRIMARY_ROWS else '。参考'}）: "
              "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
        for a in algos:
            g = s[s["algo"] == a]
            j = judge7((g["pr_C"] - g["pr_A"]).to_numpy(), (g["pr_P"] - g["pr_A"]).to_numpy())
            summary["judge7"][f"{a}_{rs}"] = j
            ref = "" if a in JUDGE_ALGOS else "（参考。§7 の票には入れない）"
            print(f"  {a:<7}C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / "
                  f"P−A {j['mean_pa']:>+8.4f} {'○' if j['beats_placebo'] else '×'} / "
                  f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} "
                  f"→ {'満たす' if j['pass'] else '満たさない'}{ref}")
        judged = [a for a in algos if a in JUDGE_ALGOS]
        passed = sum(int(summary["judge7"][f"{a}_{rs}"]["pass"]) for a in judged)
        complete = all(a in algos for a in JUDGE_ALGOS)
        verdict = (("満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）") if complete
                   else f"{passed}/{len(judged)} モデルが満たす（木3モデルがそろってから判定）")
        summary["judge7"][f"verdict_{rs}"] = {"passed": passed, "of": len(judged), "complete": complete,
                                              "reference": [a for a in algos if a not in JUDGE_ALGOS]}
        print(f"  → §7（ラベル {k:.1f}σ・{titles[rs]}）: {verdict}")

    for rs in keysets:
        s = tables[rs]
        if not len(s):
            continue
        print(f"\n■ 5{sec[rs]}. 窓ごとの値（{titles[rs]}。PR-AUC。差は「後 − 前」。運用者の指示「各cv（異なる窓）でもみたい」）")
        for a in algos:
            for sh in shifts:
                g = s[(s["algo"] == a) & (s["shift"] == sh)].sort_values("fold")
                if not len(g):
                    continue
                print(f"  {a}・ずらし{sh}か月（{len(g)}窓）")
                print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}{'A':>8}{'B':>8}{'C':>8}{'P':>8}"
                      f"{'C−A':>9}{'P−A':>9}{'B−A':>9}")
                for _, r in g.iterrows():
                    print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                          f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%{r['pr_A']:>8.4f}{r['pr_B']:>8.4f}{r['pr_C']:>8.4f}"
                          f"{r['pr_P']:>8.4f}{r['pr_C'] - r['pr_A']:>+9.4f}{r['pr_P'] - r['pr_A']:>+9.4f}"
                          f"{r['pr_B'] - r['pr_A']:>+9.4f}")
        summary["per_window"][rs] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験89: 機関投資家（運用目的）の直近の買いのフラグ2列を本番の239列に足す前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--k", type=float, default=KS[0], help="ラベルの到達しきい値（σ の倍数。1.5 か 1.2）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0,
                    help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    if not any(abs(args.k - k) < 1e-9 for k in KS):
        raise SystemExit(f"--k は {KS} のどれか（実際: {args.k}）")
    TAG = args.tag
    RUN = f"{TAG}_{k_tag(args.k)}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    new = list(I.COLUMNS)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"本番の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new, "perm": base + new}

    print("=" * 78)
    print("実験89 機関投資家（運用目的）の直近の買いのフラグ2列を本番の239列に足す前後"
          "（5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + {', '.join(new)} → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  ラベル {args.k:.1f}σ / この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割"
          f"（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / 対照の種 {PERM_SEED} / "
          f"判定の行 {PRIMARY_ROWS} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    miss = [c for c in colsets["new"] if c not in full.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    df = with_label(full, args.k)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cover = coverage_report(df, stamp, args.k)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    have = df[COVER_COL].notna()
    print(f"\n  この回の行 {len(df):,}（ラベル {args.k:.1f}σ の正例率 {df['label'].mean()*100:.1f}%・列の分かる行 "
          f"{have.sum():,}・{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行（分かる行 {(have & (df['Date'] <= cutoff)).sum():,}）")
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: {len(new)}列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm}
    keysets = eval_keysets(df)

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {'そろった' if results[a] else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    report(results, shifts, stamp, keysets, cover, args.k)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
