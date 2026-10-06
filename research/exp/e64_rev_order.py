#!/usr/bin/env python3
"""
実験64: 業績予想の修正の並べ替えを決定的にした直し（2026-10-06）で、5モデルの探索（5分割 CV）・
本番と同じ作りの out-of-fold・32窓がどう変わるか。

運用者の依頼（2026-10-06）「並べ替えを (開示日, 開示時刻, 開示番号) の安定ソートにして決定的にする。
これでお願いします。変更の影響具合の検証をしたいので、これまで通り、日曜の再学習の前に、5cv+oof検証で
どのくらい精度に影響があるかは確認したいです。」

直したこと（build_dataset.forecast_revisions）: 同じ銘柄が同じ日に業績予想の修正を2件以上出したとき、
各行に付く修正と向きの比較相手が、安定でない並べ替えの偶然で決まっていた。行を (開示日, 開示時刻,
開示番号) で安定に並べ、その日の最後の開示を付ける。値が変わりうるのは rev_pct / rev_up / rev_up_n_250 /
rev_dn_n_250 で、同じ日に2件以上の修正がある日（131日、修正の (銘柄, 開示日) の 0.47%）に当たる行だけ
（docs/DATA_TIMING.md「2026-10-05 の一致チェックで出た rev_pct / rev_up の欠測」）。

腕（生データは同じ。forecast_revisions の作りだけが違う。列は本番の239列）
  A  直す前の作り（この台本に写した旧版 forecast_revisions_old）。A のデータで探索
  B  直した後の作り（いまの build_dataset）。B のデータで探索 → 日曜の週次実行がやること
  C  直した後の作りを、A のパラメータで（探索なし）→ データの違いだけの効果
  B−A が「日曜の再学習で起きること」、C−A が「直しそのものの効果」、B−C が「探索し直しの揺れ」

探索は本番の週次実行と同じ関数・条件（50試行 × 5分割 year_cap_date・ホールドアウトより前・種0）:
  LightGBM は run_tuning.py と同じ tuning.tune（木200本・学習率の範囲は本数から）、
  ほか4モデルは e15_tune_all.py と同じ tuning_multi.tune（前処理は本番の設定。logit は v2）

評価（実験57・58 と同じ作り）
  0. データの違い: 行の出入りと、値が変わった列・行の数（探索の期間 / OOF の期間）
  1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）: PR-AUC ± SD・ROC・分割ごとの PR-AUC・所要
  2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）:
     PR-AUC / リフト / ROC-AUC / 日内 AUC / 上位10% の ret_o1_20 の超過（e25_auc_noise.metrics）
  3. 窓ごと: 境界を 0/2/4か月ずらした3通り（計32窓）。種は logit 1つ（決定的）、ほか3つの平均。
     B−A / C−A / B−C の平均・SE・上の窓の数

Actions の1回の上限（330分）に収まるよう、モデルを分けて回す。データセット（両方の腕）は1回目に作って
research/_data/oof/e64_* に置き、以降の回はそれを使う（その間に増えた日のデータは混ぜない）。途中の結果も
同じ場所に置き、run-experiment のキャッシュで次の回に引き継ぐ。表は、結果のそろったモデルをすべて出す。
    exp=e64_rev_order.py args="--algos lgbm,logit,mlp"
    exp=e64_rev_order.py args="--algos xgb"
    exp=e64_rev_order.py args="--algos cat"

本番の設定（research/lgbm_params.json、research/multi_params.json）には書かない。

試運転:
    python3 research/exp/e64_rev_order.py --algos logit --n-trials 2 --shifts 0
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
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = "year_cap_date"          # 本番の retrain-weekly.yml と同じ
ALGOS = ("lgbm", "xgb", "cat", "logit", "mlp")
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123),
         "logit": (42,), "mlp": (42, 7, 123)}
PROD_SEED = 42
ARMS = ("A", "B", "C")
DATA_OF = {"A": "A", "B": "B", "C": "B"}       # どのデータセット
PARAMS_OF = {"A": "A", "B": "B", "C": "A"}     # どの探索のパラメータ
LABELS = {"A": "A 直す前", "B": "B 直した後・探索し直し", "C": "C 直した後・A のパラメータ"}
PAIRS = (("A", "B"), ("A", "C"), ("C", "B"))
#: 値が変わりうる列（features.REDEFINED に入れた列）
REV_COLS = ("rev_pct", "rev_up", "rev_up_n_250", "rev_dn_n_250")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OOF_DIR, f"e64_{name}")


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# 直す前の作り（build_dataset.forecast_revisions の 2026-10-06 までの版をそのまま写したもの）
# --------------------------------------------------------------------------- #
def forecast_revisions_old(samples: pd.DataFrame, fins: pd.DataFrame) -> pd.DataFrame:
    """2026-10-06 までの forecast_revisions（同じ日の並びが安定でない並べ替えで決まる版）。"""
    want = F.GROUPS.get("revision", [])
    out = pd.DataFrame(np.nan, index=samples.index, columns=want, dtype=float)
    if "DocType" not in fins.columns:
        return out
    f = fins.copy()
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f.dropna(subset=["DiscDate", "Code"]).sort_values(["Code", "DiscDate"])
    dt_ = f["DocType"].astype(str)
    is_earn = dt_.str.contains("EarnForecastRevision", na=False)
    is_div = dt_.str.contains("DividendForecastRevision", na=False)

    if {"FOP", "CurFYSt"} <= set(f.columns):
        fop = pd.to_numeric(f["FOP"], errors="coerce")
        prev = fop.groupby([f["Code"], f["CurFYSt"]], sort=False).shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["_rev_pct"] = np.where(prev > 0, fop / prev * 100.0 - 100.0, np.nan)
    else:
        f["_rev_pct"] = np.nan

    left = samples[["Code", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    left = left.sort_values("Date")

    for flag, prefix in ((is_earn, "rev"), (is_div, "divrev")):
        ev = f.loc[flag, ["Code", "DiscDate", "_rev_pct"]].sort_values("DiscDate")
        if not len(ev):
            continue
        ev = ev.rename(columns={"DiscDate": f"_{prefix}_d",
                                "_rev_pct": f"_{prefix}_pct"})
        m = pd.merge_asof(left, ev, left_on="Date", right_on=f"_{prefix}_d",
                          by="Code", direction="backward", allow_exact_matches=True)
        m = m.sort_values("_i")
        days = (m["Date"] - m[f"_{prefix}_d"]).dt.days
        out[f"days_since_{prefix}"] = np.clip(days.to_numpy(), 0, B.REV_CLIP)
        if prefix == "rev":
            out["rev_pct"] = m[f"_{prefix}_pct"].to_numpy()
            out["rev_up"] = np.where(np.isfinite(out["rev_pct"]),
                                     (out["rev_pct"] > 0).astype(float), np.nan)

    ev = f.loc[is_earn, ["Code", "DiscDate", "_rev_pct"]].copy()
    if len(ev):
        for w in B.REV_WINDOWS:
            out[f"rev_n_{w}"] = B._events_in_window(samples, ev, "DiscDate", w)
        up = ev[ev["_rev_pct"] > 0]
        dn = ev[ev["_rev_pct"] < 0]
        if len(up):
            out["rev_up_n_250"] = B._events_in_window(samples, up, "DiscDate", 250)
        if len(dn):
            out["rev_dn_n_250"] = B._events_in_window(samples, dn, "DiscDate", 250)
    return out[want]


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def build_with(func, out_path: str) -> pd.DataFrame:
    """build_dataset.build を、forecast_revisions だけ差し替えて回す（終わったら戻す）。"""
    keep = B.forecast_revisions
    B.forecast_revisions = func
    try:
        return B.build(lab.DATA_DIR, out_path)
    finally:
        B.forecast_revisions = keep


def same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """値が同じか（欠測どうしは同じ。数値でない列はそのまま比べる）。"""
    try:
        x = pd.to_numeric(a, errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(b, errors="raise").to_numpy(dtype=float)
        return np.isclose(x, y, rtol=1e-12, atol=0.0, equal_nan=True)
    except (TypeError, ValueError):
        xa, xb = a.astype(object).to_numpy(), b.astype(object).to_numpy()
        return np.array([(u == v) or (pd.isna(u) and pd.isna(v)) for u, v in zip(xa, xb)])


def prepare(rebuild: bool = False):
    """
    腕 A / B の表（lab.frame と同じ列: データセット + 実収益）を返す。

    B は lab.frame()（このワークフローの Build dataset が直した後のコードで作った dataset.parquet から）。
    同じコードでもう一度作った B と全列一致することを確かめる（作りが決定的か）。A は旧版で作った
    データセットの列で、B の表の特徴量の列を差し替えたもの（行・ラベル・実収益は B と同じ）。
    """
    pa, pb, ps = path("frame_A.parquet"), path("frame_B.parquet"), path("stamp.json")
    if not rebuild and all(os.path.exists(p) for p in (pa, pb, ps)):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・"
            f"〜{stamp['date_max']}）")
        return pd.read_parquet(pa), pd.read_parquet(pb), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    fb = lab.frame(rebuild=True)
    fb["Date"] = pd.to_datetime(fb["Date"])
    fb["Code"] = fb["Code"].astype(str)
    log(f"[data] B（lab.frame、直した後のコード）{len(fb):,}行 × {fb.shape[1]}列 / {time.time()-t0:.0f}秒")

    t0 = time.time()
    db = build_with(B.forecast_revisions, path("dataset_B.parquet"))
    log(f"[data] B をもう一度作った {len(db):,}行 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    da = build_with(forecast_revisions_old, path("dataset_A.parquet"))
    log(f"[data] A（旧版の forecast_revisions）{len(da):,}行 / {time.time()-t0:.0f}秒")
    for d in (da, db):
        d["Date"] = pd.to_datetime(d["Date"])
        d["Code"] = d["Code"].astype(str)

    # 行がそろっているか（forecast_revisions は行を増減させない）
    kb = list(zip(fb["Code"], fb["Date"]))
    for name, d in (("B（もう一度）", db), ("A", da)):
        if list(zip(d["Code"], d["Date"])) != kb:
            raise SystemExit(f"{name} の行（銘柄・日付・並び）が lab.frame と違う")
        if not same(d["label"], fb["label"]).all():
            raise SystemExit(f"{name} のラベルが lab.frame と違う")
    feat = [c for c in db.columns if c in fb.columns and c not in ("Code", "Date")]
    redo = [c for c in feat if not same(db[c], fb[c]).all()]
    print(f"  作りが決定的か: 直した後のコードで2回作ったデータセットの {len(feat)}列のうち、"
          f"値が違う列 {len(redo)}本" + (f" → {redo[:12]}" if redo else "（全列一致）"))

    fa = fb.copy()
    changed = [c for c in feat if not same(da[c], fb[c]).all()]
    for c in changed:
        fa[c] = da[c].to_numpy()
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(fb)), "date_max": str(fb["Date"].max().date()),
             "changed_columns": changed, "nondeterministic_columns": redo}
    fa.to_parquet(pa, index=False)
    fb.to_parquet(pb, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    import glob
    for p in glob.glob(path("dataset_*")):     # キャッシュを小さくする（表に入れたので要らない）
        os.remove(p)
    return fa, fb, stamp


def show_diff(fa: pd.DataFrame, fb: pd.DataFrame, stamp: dict, cols: list, cutoff, oof_start) -> dict:
    print("\n■ 0. データの違い（A 直す前 / B 直した後。行・ラベル・実収益は同じ）")
    if stamp.get("nondeterministic_columns"):
        print(f"  ※ 直した後のコードで2回作って値が違った列: {stamp['nondeterministic_columns']}")
    d = fb["Date"]
    in_tune = (d <= cutoff).to_numpy()
    in_oof = ((d >= oof_start) & fb["label"].notna()).to_numpy()
    print(f"  行 {len(fb):,}（探索の期間 〜{cutoff.date()}: {int(in_tune.sum()):,} / "
          f"OOF の検証の期間 {oof_start.date()}〜: {int(in_oof.sum()):,}）")
    print(f"  {'列':<22}{'本番の列':>8}{'違う行':>8}{'探索の期間':>10}{'OOF の期間':>10}"
          f"{'A だけ欠測':>10}{'B だけ欠測':>10}")
    out = {}
    changed = stamp.get("changed_columns", [])
    for c in sorted(set(changed) | set(REV_COLS)):
        if c not in fb.columns:
            continue
        diff = ~same(fa[c], fb[c])
        a_nan = pd.isna(fa[c]).to_numpy() & ~pd.isna(fb[c]).to_numpy()
        b_nan = ~pd.isna(fa[c]).to_numpy() & pd.isna(fb[c]).to_numpy()
        out[c] = {"rows": int(diff.sum()), "tune": int((diff & in_tune).sum()),
                  "oof": int((diff & in_oof).sum()), "a_nan": int(a_nan.sum()), "b_nan": int(b_nan.sum()),
                  "in_production": c in cols}
        print(f"  {c:<22}{'○' if c in cols else '':>8}{out[c]['rows']:>8,}{out[c]['tune']:>10,}"
              f"{out[c]['oof']:>10,}{out[c]['a_nan']:>10,}{out[c]['b_nan']:>10,}")
    others = [c for c in changed if c not in REV_COLS]
    print(f"  値が変わった列 {len(changed)}本（本番の239列に入るもの "
          f"{len([c for c in changed if c in cols])}本）" + (f" / 修正の4列以外: {others}" if others else ""))
    rows_any = np.zeros(len(fb), dtype=bool)
    for c in changed:
        rows_any |= ~same(fa[c], fb[c])
    print(f"  どれかの列が違う行: {int(rows_any.sum()):,}（{rows_any.mean()*100:.2f}%）/ "
          f"探索の期間 {int((rows_any & in_tune).sum()):,} / OOF の期間 {int((rows_any & in_oof).sum()):,}")
    out["_rows_any"] = int(rows_any.sum())
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
    if not compute:
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    if algo == "lgbm":
        log(f"  [{algo} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件）")
        params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, model="classifier",
                             verbose=False)
        rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    else:
        log(f"  [{algo} {arm}] 探索（tuning_multi.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・"
            f"前処理 {TM.preprocess_version(algo) or 'なし'}）")
        TM.STUDY_DB = path(f"optuna_{arm}.db")   # 腕ごとに分ける（同じ study 名になるため）
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS, verbose=False)
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
def oof(algo: str, data: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        compute: bool = True):
    """
    データ（A / B）・パラメータ・ずらし・種ごとの out-of-fold（種の平均）。保存済みなら読む。
    compute=False なら無いとき None。名前は腕でなく「データ × パラメータ」で付けるので、A と B の探索が
    同じパラメータを選んだときは、B と C（同じデータ・同じパラメータ）が同じ計算を共有する。
    """
    ph = params_hash(params)
    files = [path(f"{algo}_d{data}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    if not compute and not all(os.path.exists(f) for f in files):
        return None
    folds = E41.folds_for(frame["Date"], shift)
    parts = []
    for sd, f in zip(seeds, files):
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
            continue
        t0 = time.time()
        o = E41.oof_folds(algo, frame, cols, params, sd, folds)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} データ{data} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def pair(name: str, a: np.ndarray, b: np.ndarray) -> tuple:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    line = (f"  {name:<20}{a.mean():>9.4f}{b.mean():>9.4f}{d.mean():>+10.4f}{se:>9.4f}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}{int((d == 0).sum()):>6}")
    return line, {"n": int(len(d)), "mean_diff": float(d.mean()), "se": float(se),
                  "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}


def run(algo: str, frames: dict, cols: list, cutoff, n_trials: int, stamp: dict, shifts: list,
        compute: bool) -> dict | None:
    """1モデルぶん。compute=False なら保存済みだけで表を作る（そろっていなければ None）。"""
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frames[arm], cols, cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod = {}
    for arm in ARMS:
        prod[arm] = oof(algo, DATA_OF[arm], frames[DATA_OF[arm]], cols, params[arm], 0, (PROD_SEED,), compute)
        if prod[arm] is None:
            return None
    wins = {}
    for sh in shifts:
        for arm in ARMS:
            o = oof(algo, DATA_OF[arm], frames[DATA_OF[arm]], cols, params[arm], sh, SEEDS[algo], compute)
            if o is None:
                return None
            wins[(sh, arm)] = AB.auc_by_window(o)
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


def report(results: dict, shifts: list, diff: dict) -> None:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": diff, "tuning": {}, "production_oof": {}, "windows": {}}

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'±SD':>8}{'ROC':>8}{'秒':>6}  分割ごとの PR-AUC / パラメータ")
    for a in algos:
        r = results[a]
        for arm in ("A", "B"):
            cv = r["recs"][arm]["_cv"]
            fs = cv.get("fold_scores") or []
            print(f"  {a:<7}{arm:<4}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
                  f"{r['recs'][arm].get('_seconds', 0):>6}  " + " / ".join(f"{v:.4f}" for v in fs))
            summary["tuning"][f"{a}_{arm}"] = {k: cv.get(k) for k in
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

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）")
    print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}")
    for a in algos:
        ms = {}
        for arm in ARMS:
            m = metrics(results[a]["prod"][arm])
            ms[arm] = m
            summary["production_oof"][f"{a}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                  f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                  f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt")
        for x, y in PAIRS:
            print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                  f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                  f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    rows = []
    for a in algos:
        for sh in shifts:
            w = None
            for arm in ARMS:
                x = results[a]["wins"][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                w = x if w is None else w.merge(x, on="fold")
            for _, r in w.iterrows():
                rows.append({"algo": a, "shift": sh, "fold": int(r["fold"]),
                             **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
    s = pd.DataFrame(rows)
    if len(s):
        s.to_csv(path("auc_by_window.csv"), index=False)
        print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り。"
              "logit は種1つ、ほかは種3つの平均）")
        print(f"  {'':<20}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}"] = st
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)


#: 表に出さないパラメータ（どの探索でも同じ値）
_FIXED_KEYS = {"objective", "boosting_type", "n_jobs", "verbose", "random_state", "subsample_freq"}


def short(p: dict) -> str:
    """パラメータを短く（小数は有効4桁）。"""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "{" + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(p.items()) if k not in _FIXED_KEYS) + "}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験64: 修正の並べ替えを決定的にした直しの前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--rebuild", action="store_true", help="データの表を作り直す（以前の結果は使えなくなる）")
    args = ap.parse_args(argv)
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験64 業績予想の修正の並べ替えを決定的にした直しの前後（5分割 CV の探索 + OOF + 32窓）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 指紋 {F.signature(cols)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {SEEDS}")
    print("=" * 78)

    fa, fb, stamp = prepare(args.rebuild)
    for f in (fa, fb):
        f["Date"] = pd.to_datetime(f["Date"])
        f["Code"] = f["Code"].astype(str)
    miss = [c for c in cols if c not in fb.columns]
    if miss:
        raise SystemExit(f"データに無い列: {miss[:8]}")
    cutoff, _, _ = T.holdout_bounds(fb["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    oof_start = pd.Timestamp(min(f.test_start for f in E41.folds_for(fb["Date"], 0)))
    diff = show_diff(fa, fb, stamp, cols, cutoff, oof_start)
    frames = {"A": fa, "B": fb}

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, cols, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {time.time()-t0:.0f}秒")
    report(results, shifts, diff)
    log(f"記録: {OOF_DIR}/e64_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
