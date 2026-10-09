#!/usr/bin/env python3
"""
実験67: 日証金の特徴量（research/jsf_features.py、17列）を本番の239列に足すと、ツリー系3モデルの探索（5分割 CV）・
本番と同じ作りの out-of-fold・32窓がどう変わるか（実験65 と同じ手順。腕は「列」の違い）。
2026-10-09 の運用者の決定で、実験はツリー系3種（lgbm / xgb / cat）だけにした（logit / mlp は画面からも外した）。

運用者の指示（2026-10-09）「先に日証金（データが完全で、20営業日の目的変数に近い日次の需給）」。
EDA は実験66（research/exp/e66_jsf_eda.py）。

腕（データは同じ1本。列だけが違う）
  A  いまの本番（239列）。A の列で探索
  B  239 + jsf（255列）で探索し直し → 日曜の週次実行がやること
  C  239 + jsf を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  B−A が「再学習で起きること」、C−A が「列を足すことの効果」、B−C が「探索し直しの揺れ」

**比べる行は2通り出す。** 全行（jsf は 2023年10月より前の行で欠測。モデルは欠測として扱う）と、
jsf の付いた行だけ（2023年10月以降の貸借銘柄。列の効果が出うる行）。窓ごとの比較は、付いた行が
100件以上あり正例と負例の両方がある窓だけ（ずらし 0/2/4か月で、2023年10月以降の窓）。

探索・評価は実験64・65 と同じ（50試行 × 5分割 year_cap_date・本番と同じ OOF・32窓。logit は種1つ、
ほかは種3つの平均）。

Actions の1回の上限（330分）に収まるよう、モデルを分けて回す。表（jsf を付けた frame）は1回目に作って
research/_data/oof/<tag>_* に置き、以降の回はそれを使う。
    exp=e67_jsf_ab.py args="--algos lgbm"
    exp=e67_jsf_ab.py args="--algos xgb"
    exp=e67_jsf_ab.py args="--algos cat"
試運転: exp=e67_jsf_ab.py args="--tag e67smoke --algos logit --n-trials 2 --shifts 0"

最初の回（run 37878860125、lgbm・logit・mlp。列は変換なし）は、logit・mlp の OOF が −0.02〜−0.04 落ちた。
裾の重い列（出来高比・逆日歩）が標準化した値に効いたためとみて、jsf_features.py で asinh / log1p に変え、
--rebuild で表を作り直して回し直した（木のモデルは単調変換に不変）。

公開ログには件数・割合・日付・精度だけを出す（日証金の値は出さない。docs/DATA_JSF.md の利用条件）。
本番の設定（research/lgbm_params.json、research/multi_params.json、features.py）には書かない。
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
import features as F  # noqa: E402
import jsf_features as JF  # noqa: E402
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
ALGOS = ("lgbm", "xgb", "cat")        # 2026-10-09 運用者の決定: 実験はツリー系3種だけ（logit / mlp は外した）
SEEDS = {"lgbm": (42, 7, 123), "xgb": (42, 7, 123), "cat": (42, 7, 123),
         "logit": (42,), "mlp": (42, 7, 123)}
PROD_SEED = 42
ARMS = ("A", "B", "C")
COLS_OF = {"A": "base", "B": "jsf", "C": "jsf"}      # どの列の組
PARAMS_OF = {"A": "A", "B": "B", "C": "A"}           # どの探索のパラメータ
LABELS = {"A": "A いまの本番（239列）", "B": "B 239 + jsf・探索し直し",
          "C": "C 239 + jsf・A のパラメータ"}
PAIRS = (("A", "B"), ("A", "C"), ("C", "B"))
ROWSETS = ("all", "covered")
MIN_WINDOW_ROWS = 100
TAG = "e67"
KEY = ["Code", "Date"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def params_hash(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:8]


def keys_of(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df["Code"].astype(str), pd.to_datetime(df["Date"])])


def on_rows(o: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    return o[keys_of(o).isin(keys)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def prepare(rebuild: bool = False):
    """lab.frame に jsf_* を付けた表と記録。"""
    pf, ps = path("frame.parquet"), path("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}・"
            f"jsf の付いた行 {stamp['covered']:,}）")
        return pd.read_parquet(pf), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    df = JF.build(frame, lab.DATA_DIR)
    have = df["jsf_ratio"].notna()
    yr = df["Date"].dt.year
    log(f"[data] jsf を付けた {len(df):,}行（付いた行 {have.sum():,}・{have.mean()*100:.1f}%）/ {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows": int(len(df)), "date_max": str(df["Date"].max().date()),
             "covered": int(have.sum()), "covered_from": str(df.loc[have, "Date"].min().date()),
             "covered_by_year": {str(y): int((have & (yr == y)).sum()) for y in sorted(yr.unique())},
             "jsf_columns": JF.columns("all")}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    return df, stamp


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
        log(f"  [{algo} {arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
        params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=T.EMBARGO_DAYS, scheme=CV_SCHEME, model="classifier",
                             verbose=False)
        rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    else:
        log(f"  [{algo} {arm}] 探索（tuning_multi.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列・"
            f"前処理 {TM.preprocess_version(algo) or 'なし'}）")
        TM.STUDY_DB = path(f"optuna_{arm}.db")
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
def oof(algo: str, colset: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        compute: bool = True):
    """列の組（base / jsf）× パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_c{colset}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
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
        log(f"    {algo} 列{colset} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame, keys: pd.MultiIndex | None) -> pd.DataFrame:
    """窓ごとの AUC。keys があればその行だけで、MIN_WINDOW_ROWS 未満の窓は落とす。"""
    sub = o if keys is None else on_rows(o, keys)
    if len(sub) == 0:
        return pd.DataFrame(columns=["fold", "pr", "roc"])
    n = sub.groupby("fold").size()
    sub = sub[sub["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["fold", "pr", "roc"])
    return AB.auc_by_window(sub)


def pair(name: str, a: np.ndarray, b: np.ndarray) -> tuple:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    line = (f"  {name:<24}{a.mean():>9.4f}{b.mean():>9.4f}{d.mean():>+10.4f}{se:>9.4f}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}{int((d == 0).sum()):>6}")
    return line, {"n": int(len(d)), "mean_diff": float(d.mean()), "se": float(se),
                  "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}


def run(algo: str, frame: pd.DataFrame, keysets: dict, colsets: dict, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frame, colsets[COLS_OF[arm]], cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod = {rs: {} for rs in ROWSETS}
    for arm in ARMS:
        o = oof(algo, COLS_OF[arm], frame, colsets[COLS_OF[arm]], params[arm], 0, (PROD_SEED,), compute)
        if o is None:
            return None
        for rs in ROWSETS:
            prod[rs][arm] = o if keysets[rs] is None else on_rows(o, keysets[rs])
    wins = {rs: {} for rs in ROWSETS}
    for sh in shifts:
        for arm in ARMS:
            o = oof(algo, COLS_OF[arm], frame, colsets[COLS_OF[arm]], params[arm], sh, SEEDS[algo], compute)
            if o is None:
                return None
            for rs in ROWSETS:
                wins[rs][(sh, arm)] = windows_on(o, keysets[rs])
    return {"recs": recs, "params": params, "prod": prod, "wins": wins}


def report(results: dict, shifts: list, stamp: dict) -> None:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "tuning": {}, "production_oof": {}, "windows": {}}

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

    for rs, title in (("all", "全行"), ("covered", "jsf の付いた行だけ")):
        print(f"\n■ 2{'a' if rs == 'all' else 'b'}. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・"
              f"ずらし0・種42。{title}）")
        print(f"  {'':<7}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
        for a in algos:
            ms = {}
            for arm in ARMS:
                o = results[a]["prod"][rs][arm]
                m = metrics(o)
                ms[arm] = m
                summary["production_oof"][f"{a}_{arm}_{rs}"] = {k: float(v) for k, v in m.items()}
                print(f"  {a:<7}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}"
                      f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+10.2f}pt"
                      f"{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}{m['ret_o1_20_worst']:>+8.2f}pt"
                      f"{len(o):>8,}")
            for x, y in PAIRS:
                print(f"  {a:<7}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}"
                      f"{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}"
                      f"{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    for rs, title in (("all", "全行"), ("covered", f"jsf の付いた行だけ・{MIN_WINDOW_ROWS}件以上の窓")):
        rows = []
        for a in algos:
            for sh in shifts:
                w = None
                for arm in ARMS:
                    x = results[a]["wins"][rs][(sh, arm)].rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
                    w = x if w is None else w.merge(x, on="fold")
                if w is None or len(w) == 0:
                    continue
                for _, r in w.iterrows():
                    rows.append({"algo": a, "shift": sh, "fold": int(r["fold"]),
                                 **{f"{m}_{arm}": float(r[f"{m}_{arm}"]) for m in ("pr", "roc") for arm in ARMS}})
        s = pd.DataFrame(rows)
        if not len(s):
            continue
        s.to_csv(path(f"auc_by_window_{rs}.csv"), index=False)
        print(f"\n■ 3{'a' if rs == 'all' else 'b'}. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした"
              f"{len(shifts)}通り。logit は種1つ、ほかは種3つの平均。{title}）")
        print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
        for a in algos:
            g = s[s["algo"] == a]
            for met, nm in (("pr", "PR"), ("roc", "ROC")):
                for x, y in PAIRS:
                    line, st = pair(f"{a} {nm} {y}−{x}", g[f"{met}_{x}"].to_numpy(), g[f"{met}_{y}"].to_numpy())
                    print(line)
                    summary["windows"][f"{a}_{met}_{y}-{x}_{rs}"] = st
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)


_FIXED_KEYS = {"objective", "boosting_type", "n_jobs", "verbose", "random_state", "subsample_freq"}


def short(p: dict) -> str:
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "{" + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(p.items()) if k not in _FIXED_KEYS) + "}"


def main(argv=None) -> int:
    global TAG
    ap = argparse.ArgumentParser(description="実験67: 日証金の特徴量を本番の239列に足す前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    args = ap.parse_args(argv)
    TAG = args.tag
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    jsf = JF.columns("all")
    colsets = {"base": base, "jsf": base + jsf}

    print("=" * 78)
    print("実験67 日証金の特徴量を本番の239列に足す前後（5分割 CV の探索 + OOF + 32窓）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + jsf {len(jsf)}列 "
          f"（{', '.join(jsf)}）")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {SEEDS} / 保存 {TAG}_*")
    print("=" * 78)

    frame, stamp = prepare(args.rebuild)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    frame.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    miss = [c for c in colsets["jsf"] if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    cutoff, _, _ = T.holdout_bounds(frame["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    have = frame["jsf_ratio"].notna()
    print(f"  jsf の付いた行 {have.sum():,} / {len(frame):,}（{have.mean()*100:.1f}%。"
          f"{frame.loc[have, 'Date'].min().date()} 〜）。探索の期間（〜{cutoff.date()}）では "
          f"{(have & (frame['Date'] <= cutoff)).sum():,}行")
    keysets = {"all": None, "covered": keys_of(frame[have])}

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frame, keysets, colsets, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {time.time()-t0:.0f}秒")
    report(results, shifts, stamp)
    log(f"記録: {OOF_DIR}/{TAG}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
