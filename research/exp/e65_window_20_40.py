#!/usr/bin/env python3
"""
実験65: 目的変数の判定窓を「21〜40営業日目」にしたモデル（新ラベル）を本番と同じ手順で作り、
現行モデルと併用したとき（両方で発火した銘柄だけを買う）の収益を測る。

運用者の依頼（2026-10-07）
  「今の目的変数の期間を変更したモデルを追加で作ってほしいです。目的変数の期間を次の日寄り付き始値から
   20営業日の中で、という範囲にしてるはずですが、20〜40営業日というので目的変数を定義してモデルを
   構築してほしいです。検証方法や手法、特徴量等、サンプル母集団の定義は一切変えず、目的変数の定義のみ
   変更して検証してみてほしいです。もちろん銘柄選定の方法も同じです。最終的には、今のモデルと併用して、
   両方で発火した銘柄を選定した場合に収益率が上がっているかの確認がしたいです。」

運用者の選択（同日）
  判定窓    t+21〜t+40（今の窓 t+1〜t+20 のすぐ後の20営業日）
  しきい値  1.2σ、σ = vol_20d × √20（必要な上昇幅は今と同じ）
  探索      5モデルとも新ラベルで探索し直す（日曜の再学習と同じ手順）
  売り方    +20% の指値・届かなければ 20営業日目の終値（主）。満期を 40営業日にした版も並べる

新ラベル = build_dataset.RiseConfig(start=21, horizon=40, sigma_days=20)
  基準      t+1 の寄り（分割調整後の始値。今と同じ）
  到達      t+21〜t+40 の終値の最大 ≥ 基準 × (1 + 1.2σ)
  終盤      t+40 の5日平均終値 ≥ 基準 × (1 + 0.6σ)
  トレンド  t+40 で MA5 ≥ MA20（MA20 は t+21〜t+40 = 判定窓そのもの）
  確定      先 40営業日ぶんの行があること（今は 20）

変えないもの
  母集団（データセットの行。本番の build_dataset.py そのまま。新ラベルが未確定の行は学習に使わない）、
  特徴量（features.DEFAULT_PRESET）、5モデルの作り（本番の LightGBM と tuning_multi.build）、
  探索の手順（年×時価総額帯で層別・日付単位の5分割・50試行・木200本・ホールドアウトより前）、
  OOF の窓（36ヶ月 / 6ヶ月 / 6ヶ月）、選定の規則（実験59: 百分位は過去分布、全5モデル 95以上、
  1日1件、枠3、翌営業日の寄りで 100株）。
  エンバーゴは「ラベル確定に要る営業日数」という今の規則のまま、値が 20 → 40 になる。make_folds は
  訓練の終わりを固定してテスト窓の始まりを後ろへずらすので、新ラベルの OOF は約1か月遅く始まる。
  比べるのは両方の OOF がある期間だけ。

段（--stage）
  label  データセットの行に新ラベルを付ける。現行ラベルを同じ関数で引き直し、本番のラベルと全行一致する
         ことを確かめてから使う
  tune   新ラベルで LightGBM（run_tuning.py と同じ）と xgb / cat / logit / mlp（e15_tune_all.py と同じ）を
         探索する。結果は <out>/params_<algo>.json、Optuna の記録は <out>/optuna_<algo>.db（本番の
         lgbm_params.json / multi_params.json / optuna_multi.db は触らない）
  oof    現行（本番のパラメータ・エンバーゴ20）と新ラベル（探索したパラメータ・エンバーゴ40）で
         5モデルずつ OOF を作る。現行の OOF は Release の OOF（本番の週次学習）と照合する
  sim    実験59 と同じ規則で 現行のみ / 新のみ / 両方 を比べる（売り方 2通り）。候補単位（枠なし）の
         比較、分離力、年ごとも出す
  all    上を順に（label → tune → oof → sim）
  robust 頑健性の確認。OOF の窓の境界をずらした版（--stage oof --shift 2 / 4 で先に作る。実験26・41・55 の
         「切り方」と同じ）でも sim を回し、切り方ごとの要の数字を並べる。パラメータはずらし0 と同じもの

出力は <out>（既定 research/_data/oof/e65/、.gitignore 済み）。取引の一覧（trades.csv）は銘柄コードと
買値を含むので公開の場には出さない。ログに出すのは件数・割合・金額の合計だけ。

    python3 research/exp/e65_window_20_40.py --stage all
    python3 research/exp/e65_window_20_40.py --stage tune --algos logit     # 1モデルだけ探索
    python3 research/exp/e65_window_20_40.py --stage oof --shift 2          # 窓を2か月ずらした OOF
    python3 research/exp/e65_window_20_40.py --stage robust --shifts 0,2,4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import e59_unit_sim as U  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import models as M  # noqa: E402
import train_model as T  # noqa: E402
import train_multi as TMU  # noqa: E402
import train_production as TP  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402

OUT_DIR = os.path.join(lab.DATA_DIR, "oof", "e65")

#: 新ラベル（運用者の選択 2026-10-07）。基準・k・終盤・トレンドの式は今のまま
NEW_RISE = B.RiseConfig(start=21, horizon=40, sigma_days=20)
#: エンバーゴ = ラベル確定に要る営業日数（今の規則。値だけ 20 → 40）
EMBARGO_NEW = NEW_RISE.horizon
ALGOS = tuple(M.ALGOS)               # lgbm / xgb / cat / logit / mlp
N_TRIALS = 50                        # 日曜の再学習と同じ
N_SPLITS = 5
HOLDS = (20, 40)                     # 売り方: 満期 20営業日（主）/ 40営業日（併記）
TARGET = L.TAKE_PROFIT               # +20% の指値
SLOTS = L.SLOTS                      # 枠3
UNIT = U.UNIT                        # 100株
AGREES = (95.0, 90.0)                # 全5モデル 95以上（運用者の規則）と、参考の 90以上
CUR = tuple(f"cur_{a}" for a in ALGOS)
NEW = tuple(f"new_{a}" for a in ALGOS)
#: (名前, 合議に使うモデル, 百分位の下限)
PATTERNS = tuple(
    (f"{tag} 全5モデル{agree:.0f}以上", models, agree)
    for agree in AGREES
    for tag, models in (("現行", CUR), ("新", NEW), ("両方", CUR + NEW)))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preset() -> str:
    return F.DEFAULT_PRESET


def feature_cols() -> List[str]:
    return list(F.columns(preset()))


# ---------------------------------------------------------------- #
# label: データセットの行に新ラベルを付ける
# ---------------------------------------------------------------- #

LABEL_COLS = ["label", "future_rise", "end_level", "uptrend_end", "entry_price",
              "rise_need", "end_need"]


def labels_for(panel: pd.DataFrame, cfg: B.RiseConfig) -> pd.DataFrame:
    """
    price_panel の (Code, Date, close, open, vol_20d) に cfg のラベルを付けた表。
    build_dataset.attach_rise_label をそのまま使う（ラベルの定義を2箇所に持たない）。
    """
    d = B.attach_rise_label(panel[["Code", "Date", "close", "open", "vol_20d"]].copy(), cfg)
    d = d[["Code", "Date"] + LABEL_COLS].copy()
    # label / uptrend_end は True / False / NaN の object になるので、1.0 / 0.0 / NaN にそろえる
    for c in ("label", "uptrend_end"):
        d[c] = pd.to_numeric(d[c].map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0, 1.0: 1.0, 0.0: 0.0}),
                             errors="coerce")
    return d


def _same(a: pd.Series, b: pd.Series, tol: float = 1e-9) -> pd.Series:
    """両方 NaN か、差が tol 以下なら True。"""
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    return (a.isna() & b.isna()) | ((a - b).abs() <= tol * np.maximum(1.0, b.abs()))


def check_current(ds: pd.DataFrame, cur: pd.DataFrame) -> Dict:
    """
    現行ラベルの引き直し（cur）が、データセットに入っている本番のラベルと全行一致するか。
    一致しなければ止める（同じ作りで新ラベルを付けた、と言えなくなる）。
    """
    m = ds.merge(cur, on=["Code", "Date"], how="left", suffixes=("", "_re"), validate="one_to_one")
    out = {"n": int(len(m))}
    for c in ("label", "future_rise", "end_level", "uptrend_end", "entry_price"):
        if c not in ds.columns:
            continue
        ok = _same(m[c].astype(float), m[f"{c}_re"].astype(float))
        out[c] = int((~ok).sum())
    bad = {k: v for k, v in out.items() if k != "n" and v}
    if bad:
        raise SystemExit(f"現行ラベルの引き直しが本番のラベルと一致しません（食い違う行数）: {bad}")
    return out


def funnel(t: pd.DataFrame, cfg: B.RiseConfig) -> List[str]:
    """確定した行について、条件を重ねたときの正例数。"""
    d = t[t["label"].notna()]
    n = len(d)
    reach = d["future_rise"] >= d["rise_need"]
    end = d["end_level"] >= d["end_need"]
    up = d["uptrend_end"].astype(float) == 1.0
    lines = [f"確定 {n:,}件"]
    for name, mask in (("到達", reach), ("＋終盤", reach & end), ("＋トレンド（= 正例）", reach & end & up)):
        lines.append(f"{name} {int(mask.sum()):,}件（{mask.mean() * 100:.2f}%）")
    return lines


def label_stage(data_dir: str, dataset: str, out_dir: str) -> pd.DataFrame:
    ds = pd.read_parquet(dataset, columns=["Code", "Date", "label", "future_rise", "end_level",
                                           "uptrend_end", "entry_price"])
    ds["Date"] = pd.to_datetime(ds["Date"])
    log(f"データセット {len(ds):,}行 / {ds['Date'].min().date()}〜{ds['Date'].max().date()} / "
        f"現行ラベルの正例率 {ds['label'].mean() * 100:.2f}%")
    bars = B.load_parts("bars", data_dir)
    log(f"日足 {len(bars):,}行。price_panel を作る（ラベルの入力 close / open / vol_20d は build と同じもの）")
    panel = B.price_panel(bars)[["Code", "Date", "close", "open", "vol_20d"]].copy()
    del bars
    log("現行ラベルを引き直す（本番と一致するかの検査）")
    cur = labels_for(panel, B.DEFAULT_RISE)
    chk = check_current(ds, cur)
    log(f"一致: {chk['n']:,}行すべて（label / future_rise / end_level / uptrend_end / entry_price）")
    del cur
    log(f"新ラベルを付ける: {NEW_RISE.name}")
    new = labels_for(panel, NEW_RISE)
    del panel
    t = ds[["Code", "Date", "label"]].rename(columns={"label": "label_cur"}).merge(
        new, on=["Code", "Date"], how="left", validate="one_to_one")
    t = t.rename(columns={c: f"{c}_new" for c in LABEL_COLS})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "labels.parquet")
    t.to_parquet(path, index=False, compression="zstd")

    # --- 要約 --- #
    f = t.rename(columns={f"{c}_new": c for c in LABEL_COLS})
    print("\n■ 新ラベル（データセットの行）")
    for line in funnel(f, NEW_RISE):
        print(f"  {line}")
    det = t["label_new"].notna()
    last = t.loc[det, "Date"].max()
    tail = t["Date"] > last
    print(f"  未確定 {int((~det).sum()):,}行（うち末尾 {last.date()} より後 {int((~det & tail).sum()):,}行、"
          f"それより前 {int((~det & ~tail).sum()):,}行 = 21〜40営業日目の間に日足が途切れた銘柄など）")
    both = t[det]
    ct = pd.crosstab(both["label_cur"].astype(int), both["label_new"].astype(int))
    print(f"  現行ラベル × 新ラベル（両方確定 {len(both):,}行）:")
    for i in (0, 1):
        row = [int(ct.loc[i, j]) if (i in ct.index and j in ct.columns) else 0 for j in (0, 1)]
        print(f"    現行 {i}: 新 0 = {row[0]:,} / 新 1 = {row[1]:,}")
    agree = float((both["label_cur"].astype(int) == both["label_new"].astype(int)).mean() * 100)
    print(f"    一致 {agree:.2f}% / 正例率 現行 {both['label_cur'].mean() * 100:.2f}% / "
          f"新 {both['label_new'].mean() * 100:.2f}%")
    y = both.groupby(both["Date"].dt.year).agg(n=("label_new", "size"), cur=("label_cur", "mean"),
                                               new=("label_new", "mean"))
    print("  年ごとの正例率（現行 → 新）: " + " ".join(
        f"{yy}:{r.cur * 100:.1f}→{r.new * 100:.1f}%" for yy, r in y.iterrows()))
    log(f"書いた: {path}")
    return t


def load_labels(out_dir: str) -> pd.DataFrame:
    path = os.path.join(out_dir, "labels.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} がありません。先に --stage label を回す")
    t = pd.read_parquet(path)
    t["Date"] = pd.to_datetime(t["Date"])
    return t


def with_new_label(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    frame（lab.frame() = データセット + 実収益）の label を新ラベルに差し替え、新ラベルが未確定の行を落とす。
    行の順は frame のまま（学習データの並び = (Date, Code)。build_dataset.canonical_order）。
    現行ラベルは label_cur に残す。
    """
    f = frame.merge(labels[["Code", "Date", "label_new"]], on=["Code", "Date"], how="left",
                    validate="one_to_one")
    f = f.rename(columns={"label": "label_cur"})
    f = f[f["label_new"].notna()].reset_index(drop=True)
    f["label"] = f.pop("label_new").astype(int)
    return f


# ---------------------------------------------------------------- #
# tune: 新ラベルで5モデルを探索する（本番と同じ手順）
# ---------------------------------------------------------------- #

def params_path(out_dir: str, algo: str) -> str:
    return os.path.join(out_dir, f"params_{algo}.json")


def load_new_params(out_dir: str, algo: str) -> Dict:
    p = params_path(out_dir, algo)
    if not os.path.exists(p):
        raise SystemExit(f"{p} がありません。先に --stage tune --algos {algo} を回す")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def tune_window(fw: pd.DataFrame, embargo_days: int = EMBARGO_NEW) -> pd.DataFrame:
    """探索に使う行: ホールドアウト（直近12か月）とエンバーゴより前（run_tuning / e15 と同じ境界の取り方）。"""
    d = pd.to_datetime(fw["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, embargo_days)
    return fw[d <= train_end]


def tune_one(algo: str, sub: pd.DataFrame, cols: List[str], out_dir: str,
             n_trials: int = N_TRIALS) -> Dict:
    """
    1モデルを探索して <out>/params_<algo>.json に書く。
      lgbm      run_tuning.py と同じ（tuning.tune、year_cap_date・5分割・classifier。行は (Date, Code) の順）
      それ以外  e15_tune_all.py と同じ（tuning_multi.tune）。Optuna の記録は <out>/optuna_<algo>.db
    """
    t0 = time.time()
    if algo == "lgbm":
        s = sub.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
        params = tuning.tune(s, cols, n_trials=n_trials, n_splits=N_SPLITS,
                             embargo_days=EMBARGO_NEW, scheme="year_cap_date", model="classifier")
        rec = {**params, "_cv": dict(tuning.LAST_CV), "_n_features": len(cols),
               "_features_sig": F.signature(cols), "_preset": preset()}
    else:
        # 本番の試行 DB（optuna_multi.db）に新ラベルの試行を混ぜない
        TM.STUDY_DB = os.path.join(out_dir, f"optuna_{algo}.db")
        rec = TM.tune(algo, sub, cols, n_trials=n_trials, n_splits=N_SPLITS)
        rec["_cv"]["preset"] = preset()
    rec["_e65"] = {"label": NEW_RISE.name, "embargo_days": EMBARGO_NEW,
                   "train_to": str(pd.to_datetime(sub["Date"]).max().date()),
                   "rows": int(len(sub)), "positive_rate": round(float(sub["label"].mean()), 4),
                   "secs": round(time.time() - t0)}
    os.makedirs(out_dir, exist_ok=True)
    with open(params_path(out_dir, algo), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return rec


def tune_stage(frame: pd.DataFrame, labels: pd.DataFrame, algos: Sequence[str], out_dir: str,
               n_trials: int = N_TRIALS) -> None:
    cols = feature_cols()
    fw = with_new_label(frame, labels)
    sub = tune_window(fw)
    log(f"探索に使う行: {len(sub):,}件 / {sub['Date'].min().date()}〜{sub['Date'].max().date()} / "
        f"正例率 {sub['label'].mean() * 100:.2f}% / 列 {preset()}（{len(cols)}列） / 各{n_trials}試行")
    for algo in algos:
        log(f"[{algo}] 探索を始める")
        rec = tune_one(algo, sub, cols, out_dir, n_trials)
        cv = rec["_cv"]
        log(f"[{algo}] {rec['_e65']['secs']}秒 / 探索CV PR-AUC {cv['mean_pr_auc']:.4f}"
            f"（±{cv['std']:.4f}）/ ROC-AUC {cv['mean_roc_auc']:.4f}")


# ---------------------------------------------------------------- #
# oof: 現行と新ラベルで5モデルずつ
# ---------------------------------------------------------------- #

def oof_path(out_dir: str, key: str, shift: int = 0) -> str:
    """OOF の置き場所。ずらし0 は oof_<key>.parquet、ずらしたものは oof_<key>_sh<か月>.parquet。"""
    return os.path.join(out_dir, f"oof_{key}{f'_sh{shift}' if shift else ''}.parquet")


@contextmanager
def shifted_folds(shift_months: int):
    """
    OOF の窓の境界を shift_months か月後ろにずらす（実験26・41・55 の「切り方」と同じ。make_folds に渡す
    日付の頭を切るので、訓練は期間の最初から使う）。0 なら何もしない。本番の oof_scores
    （train_production / train_multi）をそのまま使うため、walkforward.make_folds を一時的に差し替える。
    """
    import walkforward as WF
    if not shift_months:
        yield
        return
    orig = WF.make_folds

    def make_folds(dates, **kw):
        d = pd.to_datetime(pd.Series(dates))
        return orig(d[d >= d.min() + pd.DateOffset(months=shift_months)], **kw)

    WF.make_folds = make_folds
    try:
        yield
    finally:
        WF.make_folds = orig


def current_params(algo: str, cols: List[str]) -> Optional[Dict]:
    """本番のパラメータ（lgbm は lgbm_params.json の本番の列の鍵、ほかは multi_params.json）。"""
    if algo == "lgbm":
        rec = tuning.load_params().get(preset())
        if not rec:
            raise SystemExit(f"本番の LightGBM のパラメータ（{preset()}）がありません")
        why = tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), cols)
        if why or rec.get("n_estimators") != tuning.SEARCH_N_ESTIMATORS:
            raise SystemExit(f"本番の LightGBM のパラメータが今の列・本数と合いません: {why}")
        return tuning.params_for(preset())
    why = TMU.untuned([algo], TM.load(), cols)
    if why:
        raise SystemExit(f"本番の {algo} のパラメータが今の列と合いません: {why[algo]}")
    return None      # train_multi.oof_scores が multi_params.json から読む（本番と同じ経路）


def new_params(algo: str, cols: List[str], out_dir: str) -> Dict:
    rec = load_new_params(out_dir, algo)
    if algo == "lgbm":
        why = tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), cols)
        if why:
            raise SystemExit(f"新ラベルの LightGBM のパラメータが今の列と合いません: {why}")
        # train_production と同じく、既定値に探索した値を重ねる
        return tuning.params_for(preset(), store={preset(): rec})
    cv = rec.get("_cv", {})
    why = tuning.tuned_mismatch(cv.get("features_sig"), cv.get("n_features"), cols)
    if why:
        raise SystemExit(f"新ラベルの {algo} のパラメータが今の列と合いません: {why}")
    return dict(rec["params"])


def make_oof(algo: str, ds: pd.DataFrame, cols: List[str], params: Optional[Dict],
             embargo_days: Optional[int]) -> pd.DataFrame:
    """本番と同じ OOF（lgbm は train_production.oof_scores、ほかは train_multi.oof_scores）。"""
    if algo == "lgbm":
        return TP.oof_scores(ds, cols, params, embargo_days=embargo_days)
    return TMU.oof_scores(algo, ds, cols, params=params, embargo_days=embargo_days)


def release_oof(data_dir: str, algo: str) -> Optional[pd.DataFrame]:
    """Release（data-raw）から落とした本番の OOF。無ければ None。"""
    p = L.find_oof(algo, data_dir, L.MODEL_DIR)
    if p is None:
        return None
    o = pd.read_parquet(p)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def compare_release(mine: pd.DataFrame, rel: Optional[pd.DataFrame]) -> str:
    if rel is None:
        return "Release の OOF なし（照合しない）"
    m = mine.merge(rel[["Code", "Date", "score"]], on=["Code", "Date"], how="inner", suffixes=("", "_rel"))
    if not len(m):
        return "Release の OOF と共通の行なし"
    diff = (m["score"] - m["score_rel"]).abs()
    corr = float(np.corrcoef(m["score"], m["score_rel"])[0, 1]) if len(m) > 2 else float("nan")
    return (f"Release と共通 {len(m):,}行（自分 {len(mine):,} / Release {len(rel):,}）: "
            f"一致（差 1e-6 以下）{(diff <= 1e-6).mean() * 100:.1f}% / 相関 {corr:.5f} / 最大差 {diff.max():.2e}")


def oof_stage(frame: pd.DataFrame, labels: pd.DataFrame, algos: Sequence[str], which: Sequence[str],
              data_dir: str, out_dir: str, shift: int = 0) -> None:
    """
    現行と新ラベルの OOF を作る。shift（か月）を渡すと窓の境界をずらす（頑健性の確認用。shifted_folds）。
    現行の OOF はずらし0 のときだけ Release の OOF と照合する。
    """
    cols = feature_cols()
    os.makedirs(out_dir, exist_ok=True)
    tag = f"ずらし{shift}か月 " if shift else ""
    if "cur" in which:
        log(f"{tag}現行: {len(frame):,}行 / 正例率 {frame['label'].mean() * 100:.2f}% / エンバーゴ {B.RISE_HORIZON}")
        for algo in algos:
            t0 = time.time()
            with shifted_folds(shift):
                o = make_oof(algo, frame, cols, current_params(algo, cols), None)
            o.to_parquet(oof_path(out_dir, f"cur_{algo}", shift), index=False, compression="zstd")
            om = TP.oof_metrics(o)
            log(f"{tag}[cur_{algo}] {time.time() - t0:.0f}秒 / OOF {len(o):,}行 {o['Date'].min().date()}〜"
                f"{o['Date'].max().date()} / PR-AUC {om['prAuc']:.4f} ROC-AUC {om['rocAuc']:.4f} "
                f"日内AUC {om['aucInDay']:.4f}")
            if not shift:
                log(f"[cur_{algo}] {compare_release(o, release_oof(data_dir, algo))}")
    if "new" in which:
        fw = with_new_label(frame, labels)
        log(f"{tag}新: {len(fw):,}行 / 正例率 {fw['label'].mean() * 100:.2f}% / エンバーゴ {EMBARGO_NEW}")
        for algo in algos:
            t0 = time.time()
            with shifted_folds(shift):
                o = make_oof(algo, fw, cols, new_params(algo, cols, out_dir), EMBARGO_NEW)
            o.to_parquet(oof_path(out_dir, f"new_{algo}", shift), index=False, compression="zstd")
            om = TP.oof_metrics(o)
            log(f"{tag}[new_{algo}] {time.time() - t0:.0f}秒 / OOF {len(o):,}行 {o['Date'].min().date()}〜"
                f"{o['Date'].max().date()} / PR-AUC {om['prAuc']:.4f} ROC-AUC {om['rocAuc']:.4f} "
                f"日内AUC {om['aucInDay']:.4f}")


# ---------------------------------------------------------------- #
# sim: 実験59 と同じ規則で 現行のみ / 新のみ / 両方
# ---------------------------------------------------------------- #

def load_oofs(out_dir: str, keys: Sequence[str] = CUR + NEW, shift: int = 0) -> Dict[str, pd.DataFrame]:
    oofs = {}
    for k in keys:
        p = oof_path(out_dir, k, shift)
        if not os.path.exists(p):
            raise SystemExit(f"{p} がありません。先に --stage oof{f' --shift {shift}' if shift else ''} を回す")
        o = pd.read_parquet(p)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        oofs[k] = o
    return oofs


def rows_with_pct(oofs: Dict[str, pd.DataFrame], min_hist: int = U.MIN_HIST) -> pd.DataFrame:
    """
    百分位（過去分布 = その日より前の、そのモデル自身の OOF の分布。実験59 の本命と同じ）を
    モデルごとに付け、現行 LightGBM の OOF の行に左結合で揃える。
    新ラベルの OOF に無い行（新ラベルが未確定）は新モデルの百分位が NaN になり、新・両方の基準は
    満たさない（実験59 と同じく、欠けたモデルがあれば「判定できない」）。
    label_cur / label_new / ret_o1_20 / ret_o1_40 は持ち回す。score は現行 LightGBM（同点の並べ替え用）。
    """
    base = None
    for k, o in oofs.items():
        t = o[["Code", "Date", "score"]].rename(columns={"score": f"s_{k}"}).copy()
        t[f"p_{k}"] = U.pct_expanding(t["Date"], t[f"s_{k}"].to_numpy(dtype=float), min_hist)
        if k == "cur_lgbm":
            t["label_cur"] = o["label"].to_numpy()
            for c in ("ret_o1_20",):
                if c in o:
                    t[c] = o[c].to_numpy()
        if k == "new_lgbm":
            t["label_new"] = o["label"].to_numpy()
        base = t if base is None else base.merge(t, on=["Code", "Date"], how="left",
                                                 validate="one_to_one")
    base["score"] = base["s_cur_lgbm"]
    return base.sort_values(["Date", "Code"]).reset_index(drop=True)


def common_period(base: pd.DataFrame, oofs: Dict[str, pd.DataFrame], bar_days: Sequence,
                  hold: int = max(HOLDS)):
    """
    比べる選定日の範囲 [lo, hi]。
      lo  10モデルすべての百分位が出ている最初の日（新ラベルの OOF は約1か月遅く始まり、過去分布に
          MIN_HIST 行が要る）
      hi  どちらの OOF にもある最後の日と、最も長い満期（40営業日）の出口まで日足が揃う日の早いほう
    売り方 2通りで同じ範囲を使う（同じ候補で比べる）。
    """
    pc = [c for c in base.columns if c.startswith("p_")]
    full = base[pc].notna().all(axis=1)
    lo = base.loc[full, "Date"].min()
    hi = min(o["Date"].max() for o in oofs.values())
    cut = U.complete_cutoff(bar_days, hold)
    if cut is not None:
        hi = min(hi, pd.Timestamp(cut))
    return pd.Timestamp(lo), pd.Timestamp(hi)


def candidate_returns(base: pd.DataFrame, px: "U.PriceGrid", holds: Sequence[int] = HOLDS,
                      target: float = TARGET) -> pd.DataFrame:
    """
    候補1件ずつの、規則どおりの収益（枠の制約なし）。買うのは選定日の翌営業日の寄り
    （U.simulate_arm と同じ）、出口は U.rule_exit（+20% は買った日から高値、届かなければ満期の終値）。
    列 ret_<hold>（%）/ hit_<hold>（+20% 到達）。寄り付かない・先の日足が無い候補は NaN。
    """
    idx = {pd.Timestamp(d): i for i, d in enumerate(px.days)}
    T_ = len(px.days)
    out = {f"ret_{h}": np.full(len(base), np.nan) for h in holds}
    out.update({f"hit_{h}": np.full(len(base), np.nan) for h in holds})
    for n, (code, date) in enumerate(zip(base["Code"].astype(str), base["Date"])):
        i = idx.get(pd.Timestamp(date))
        col = px.col.get(code)
        if i is None or col is None or i + 1 >= T_:
            continue
        buy = i + 1
        entry = px.adjo[buy, col]
        if not (np.isfinite(entry) and entry > 0):
            continue
        for h in holds:
            _, ret, status = U.rule_exit(px, col, buy, float(entry), hold=h, target=target)
            if status in ("+20%到達", "満了"):
                out[f"ret_{h}"][n] = ret
                out[f"hit_{h}"][n] = 1.0 if status == "+20%到達" else 0.0
    return pd.DataFrame(out, index=base.index)


def cluster_se(x: pd.Series, groups: pd.Series) -> float:
    """
    平均の標準誤差を、同じ日の候補をまとめて出す（同じ日の銘柄は地合いを共有するので、
    行ごとの SE は小さく出すぎる）。SE = √Σ_g (Σ_{i∈g}(x_i − x̄))² / n。
    """
    x = pd.to_numeric(x, errors="coerce")
    ok = x.notna()
    x, g = x[ok], groups[ok]
    n = len(x)
    if n < 2:
        return float("nan")
    s = (x - x.mean()).groupby(g.to_numpy()).sum()
    return float(math.sqrt(float((s ** 2).sum())) / n)


def group_stats(d: pd.DataFrame, hold: int) -> Dict:
    r = pd.to_numeric(d[f"ret_{hold}"], errors="coerce")
    ok = r.notna()
    r = r[ok]
    n = int(len(r))
    return {"n": n, "days": int(d.loc[ok, "Date"].nunique()),
            "mean": float(r.mean()) if n else float("nan"),
            "median": float(r.median()) if n else float("nan"),
            "win": float((r > 0).mean() * 100) if n else float("nan"),
            "hit": float(pd.to_numeric(d.loc[ok, f"hit_{hold}"]).mean() * 100) if n else float("nan"),
            "loss10": float((r <= -10).mean() * 100) if n else float("nan"),
            "se": cluster_se(r, d.loc[ok, "Date"]),
            "pos_cur": float(pd.to_numeric(d.loc[ok, "label_cur"], errors="coerce").mean() * 100) if n else float("nan"),
            "pos_new": float(pd.to_numeric(d.loc[ok, "label_new"], errors="coerce").mean() * 100) if n else float("nan")}


def zdiff(a: Dict, b: Dict) -> float:
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2) if np.isfinite(a["se"]) and np.isfinite(b["se"]) else float("nan")
    return (a["mean"] - b["mean"]) / se if se and np.isfinite(se) and se > 0 else float("nan")


def candidate_groups(c: pd.DataFrame, agree: float) -> Dict[str, pd.DataFrame]:
    """候補単位の組。基準は合議の百分位の最小（1つでも欠ければ満たさない）。"""
    pc = c[[f"p_{k}" for k in CUR]].min(axis=1, skipna=False)
    pn = c[[f"p_{k}" for k in NEW]].min(axis=1, skipna=False)
    a = (pc >= agree).fillna(False)
    b = (pn >= agree).fillna(False)
    return {"母集団（期間内の候補すべて）": c,
            f"現行 {agree:.0f}以上": c[a],
            f"  両方 {agree:.0f}以上": c[a & b],
            f"  現行だけ（新は {agree:.0f}未満）": c[a & ~b],
            f"新 {agree:.0f}以上": c[b],
            f"  新だけ（現行は {agree:.0f}未満）": c[b & ~a]}


def separation(oofs: Dict[str, pd.DataFrame], base: pd.DataFrame) -> List[str]:
    """分離力（自分のラベルで、OOF 全体）と、期間内の共通の行での現行・新の重なり。"""
    from sklearn.metrics import roc_auc_score
    lines = [f"  {'モデル':<6}{'現行 PR':>9}{'÷正例率':>8}{'ROC':>8}{'日内':>8}   "
             f"{'新 PR':>8}{'÷正例率':>8}{'ROC':>8}{'日内':>8}   {'順位相関':>8}{'95の重なり':>10}"
             f"{'新→現行ラベル ROC':>18}"]
    for a in ALGOS:
        mc = TP.oof_metrics(oofs[f"cur_{a}"])
        mn = TP.oof_metrics(oofs[f"new_{a}"])
        sc, sn = base[f"s_cur_{a}"], base[f"s_new_{a}"]
        ok = sc.notna() & sn.notna()
        rho = float(sc[ok].rank().corr(sn[ok].rank())) if ok.sum() > 2 else float("nan")
        c95 = (base[f"p_cur_{a}"] >= 95) & ok
        n95 = (base[f"p_new_{a}"] >= 95) & ok
        ov = float((c95 & n95).sum() / max(int(c95.sum()), 1) * 100)
        y = pd.to_numeric(base.loc[ok, "label_cur"], errors="coerce")
        roc_x = float(roc_auc_score(y.astype(int), sn[ok])) if y.nunique() == 2 else float("nan")
        lines.append(f"  {a:<6}{mc['prAuc']:>9.4f}{mc['prAucOverBase']:>8.2f}{mc['rocAuc']:>8.4f}"
                     f"{mc['aucInDay']:>8.4f}   {mn['prAuc']:>8.4f}{mn['prAucOverBase']:>8.2f}"
                     f"{mn['rocAuc']:>8.4f}{mn['aucInDay']:>8.4f}   {rho:>8.3f}{ov:>9.1f}%{roc_x:>18.4f}")
    return lines


def sim_stage(data_dir: str, out_dir: str, min_hist: int = U.MIN_HIST, shift: int = 0) -> Dict:
    """
    実験59 と同じ規則の模擬と、候補単位の比較。shift（か月）はずらした窓で作った OOF を読む。
    戻り値は頑健性の要約（robust_stage）に使う {"lo", "hi", "summary", "cands"}。
    """
    oofs = load_oofs(out_dir, shift=shift)
    base = rows_with_pct(oofs, min_hist)
    lo_oof = min(o["Date"].min() for o in oofs.values())
    bars = L.load_bars(data_dir, start=lo_oof - pd.Timedelta(days=10), extra=("O",))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    lo, hi = common_period(base, oofs, bar_days)
    years = max((hi - lo).days, 1) / U.YEAR_DAYS
    c = base[(base["Date"] >= lo) & (base["Date"] <= hi)].reset_index(drop=True)

    print("=" * 110)
    print("実験65 目的変数の判定窓 21〜40営業日目（新）× 現行の併用: 両方で発火した銘柄を買うと収益は上がるか"
          + (f"（OOF の窓を {shift}か月ずらした版）" if shift else ""))
    print("=" * 110)
    print(f"  新ラベル: {NEW_RISE.name}（エンバーゴ {EMBARGO_NEW}営業日）")
    print(f"  現行ラベル: {B.DEFAULT_RISE.name}（エンバーゴ {B.RISE_HORIZON}営業日）")
    for k in ("cur_lgbm", "new_lgbm"):
        o = oofs[k]
        print(f"  OOF {k[:3]}: {len(o):,}行 / {o['Date'].min().date()}〜{o['Date'].max().date()}")
    print(f"  比べる選定日: {lo.date()}〜{hi.date()}（{years:.2f}年 / 候補 {len(c):,}件 / "
          f"{c['Date'].nunique():,}営業日。10モデルの百分位がそろい、満期40営業日の出口まで日足がある範囲）")
    miss = int(c[[f"p_{k}" for k in NEW]].isna().any(axis=1).sum())
    print(f"  新モデルの百分位が無い候補（新ラベルが未確定で OOF に無い）: {miss:,}件 → 新・両方の基準は満たさない")
    print(f"  規則: 百分位は過去分布、合議のモデルすべてが基準以上、1日1件（合議の最小の百分位が最大）、枠{SLOTS}、"
          f"翌営業日の寄りで{UNIT}株、+{TARGET:.0f}% の指値 / 満期の終値。乗り換えなし（実験59 の腕 A）")

    print("\n■ 分離力（自分のラベル・OOF 全体）と、比べる期間での現行・新の重なり")
    for line in separation(oofs, c):
        print(line)
    print("  （95の重なり = 現行で95以上の候補のうち、同じアルゴリズムの新でも95以上の割合。"
          "新→現行ラベル ROC = 新モデルのスコアで現行ラベルを測った ROC-AUC）")

    # --- 枠つきの模擬（実験59 と同じ） --- #
    sels = []
    for name, models, agree in PATTERNS:
        r, _, picks = L.decide(c, agree=agree, min_break=0, top_k=1, models=models)
        sels.append((name, r[r["passed"]], picks))
    codes = set(c["Code"].astype(str))
    px = U.PriceGrid(bars, bar_days, codes)
    summary, by_year, trades = [], [], []
    for hold in HOLDS:
        print(f"\n■ 枠{SLOTS}・1日1件の模擬（売り方: +{TARGET:.0f}% の指値 / {hold}営業日目の終値）")
        print(U.HEAD)
        for name, passed, picks in sels:
            t, counts, _ = U.simulate_arm(picks, px, arm="A", slots=SLOTS, hold=hold, target=TARGET, unit=UNIT)
            done = t[t["status"].isin(U.DONE)].copy() if len(t) else U.empty_trades()
            curve = U.capital_curve(done, bar_days) if len(done) else pd.Series(dtype=float)
            s = U.stats(done, curve, lo, hi, slots=SLOTS, counts=counts, n_pass=len(passed))
            print(U.line(f"{name}（基準 {len(passed)}件）", s))
            summary.append({"hold": hold, "pattern": name, **s})
            for y in range(lo.year, hi.year + 1):
                ylo, yhi = max(pd.Timestamp(f"{y}-01-01"), lo), min(pd.Timestamp(f"{y}-12-31"), hi)
                dy = done[done["buy_date"].dt.year == y] if len(done) else done
                st = U.stats(dy, curve, ylo, yhi, slots=SLOTS,
                             counts={"n_full": sum(1 for d in counts["full_days"] if d.year == y)},
                             n_pass=int((passed["Date"].dt.year == y).sum()))
                by_year.append({"hold": hold, "pattern": name, "year": y, **st})
            if len(done):
                tt = done.copy()
                tt.insert(0, "pattern", name)
                tt.insert(0, "hold", hold)
                trades.append(tt)
        print(f"  （列は実験59 と同じ。年間 = 取引 ÷ {years:.2f}年。1取引% の SE は取引どうしを独立と見た値）")

    for hold in HOLDS:
        print(f"\n■ 年ごと（+{TARGET:.0f}% / {hold}営業日、買った日の年。候補 = 基準を満たした件数）")
        for name, *_ in PATTERNS[:3]:
            rows = [r for r in by_year if r["hold"] == hold and r["pattern"] == name]
            print(f"\n  {name}")
            print(U.YEAR_HEAD)
            for r in rows:
                print(U.year_line(r["year"], r))

    # --- 候補単位（枠なし） --- #
    rets = candidate_returns(c, px)
    cc = pd.concat([c, rets], axis=1)
    cand_rows = []
    for hold in HOLDS:
        print(f"\n■ 候補単位（枠なし・同じ日の複数件もすべて数える。+{TARGET:.0f}% / {hold}営業日）")
        print(f"  {'組':<34}{'件数':>7}{'日数':>6}{'平均%':>8}{'SE':>7}{'中央%':>8}{'勝率':>7}"
              f"{'+20%到達':>9}{'−10%以下':>9}{'正例率 現行':>11}{'正例率 新':>10}")
        for agree in AGREES:
            gs = candidate_groups(cc, agree)
            st = {k: group_stats(v, hold) for k, v in gs.items()}
            for k, s in st.items():
                if agree != AGREES[0] and k.startswith("母集団"):
                    continue
                cand_rows.append({"hold": hold, "agree": agree, "group": k.strip(), **s})
                print(f"  {L._l(k, 34)}{s['n']:>7,}{s['days']:>6,}{U._pct(s['mean']):>8}"
                      f"{U._pct(s['se'], '{:.2f}'):>7}{U._pct(s['median']):>8}{U._pct(s['win'], '{:.0f}%'):>7}"
                      f"{U._pct(s['hit'], '{:.0f}%'):>9}{U._pct(s['loss10'], '{:.1f}%'):>9}"
                      f"{U._pct(s['pos_cur'], '{:.1f}%'):>11}{U._pct(s['pos_new'], '{:.1f}%'):>10}")
            keys = list(st)
            both, only = st[keys[2]], st[keys[3]]
            print(f"  → 両方 − 現行だけ（{agree:.0f}）: {both['mean'] - only['mean']:+.2f}pt "
                  f"（z = {zdiff(both, only):+.2f}）/ 両方 − 現行{agree:.0f}以上 全体: "
                  f"{both['mean'] - st[keys[1]]['mean']:+.2f}pt")
        print("  （SE は同じ日の候補をまとめた値。z = 差 ÷ √(SE₁² + SE₂²)。足切りは |z| > 2）")

    os.makedirs(out_dir, exist_ok=True)
    sfx = f"_sh{shift}" if shift else ""
    pd.DataFrame(summary).drop(columns=["full_days"], errors="ignore").to_csv(
        os.path.join(out_dir, f"summary{sfx}.csv"), index=False)
    pd.DataFrame(by_year).to_csv(os.path.join(out_dir, f"by_year{sfx}.csv"), index=False)
    pd.DataFrame(cand_rows).to_csv(os.path.join(out_dir, f"candidates{sfx}.csv"), index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(os.path.join(out_dir, f"trades{sfx}.csv"), index=False)
    log(f"書いた: {out_dir}/summary{sfx}.csv / by_year{sfx}.csv / candidates{sfx}.csv / trades{sfx}.csv")
    return {"lo": lo, "hi": hi, "summary": summary, "cands": cand_rows}


def robust_stage(data_dir: str, out_dir: str, shifts: Sequence[int], min_hist: int = U.MIN_HIST) -> None:
    """
    窓の切り方（ずらし0 / 2 / 4か月）ごとに sim を回し、要の数字を1つの表に並べる。
    ずらした OOF は --stage oof --shift N で先に作っておく。
    """
    res = {sh: sim_stage(data_dir, out_dir, min_hist, sh) for sh in shifts}
    print("\n" + "=" * 110)
    print(f"■ 切り方 {len(shifts)}通り（OOF の窓を {'/'.join(str(s) for s in shifts)}か月ずらす）の要約")
    print("=" * 110)
    for hold in HOLDS:
        for agree in AGREES:
            print(f"\n  枠{SLOTS}・1日1件（+{TARGET:.0f}% / {hold}営業日・全5モデル{agree:.0f}以上）: 取引 / 1取引% (SE) / 総損益万円")
            for sh in shifts:
                r = res[sh]
                cells = []
                for tag in ("現行", "新", "両方"):
                    s = next(x for x in r["summary"] if x["hold"] == hold
                             and x["pattern"] == f"{tag} 全5モデル{agree:.0f}以上")
                    cells.append(f"{tag} {s['n']:>3}件 {U._pct(s['mean']):>7}% ({U._pct(s['se'], '{:.2f}')}) "
                                 f"{U._man(s['pnl_total']):>4}")
                print(f"    ずらし{sh}（{r['lo'].date()}〜{r['hi'].date()}）: " + " / ".join(cells))
            print(f"  候補単位（+{TARGET:.0f}% / {hold}営業日・{agree:.0f}以上）: 両方 − 現行だけ")
            for sh in shifts:
                rows = {x["group"]: x for x in res[sh]["cands"] if x["hold"] == hold and x["agree"] == agree}
                both = rows[f"両方 {agree:.0f}以上"]
                only = rows[f"現行だけ（新は {agree:.0f}未満）"]
                allc = rows[f"現行 {agree:.0f}以上"]
                print(f"    ずらし{sh}: 現行 {allc['n']:>4}件 {U._pct(allc['mean'])}% / 両方 {both['n']:>4}件 "
                      f"{U._pct(both['mean'])}% / 現行だけ {only['n']:>4}件 {U._pct(only['mean'])}% / "
                      f"差 {both['mean'] - only['mean']:+.2f}pt（z = {zdiff(both, only):+.2f}）")


# ---------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験65: 目的変数の判定窓 21〜40営業日目のモデルと現行の併用")
    ap.add_argument("--stage", default="all", choices=["label", "tune", "oof", "sim", "robust", "all"])
    ap.add_argument("--algos", default=",".join(ALGOS), help="tune / oof で扱うモデル（カンマ区切り）")
    ap.add_argument("--which", default="cur,new", help="oof で作る側（cur / new）")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--shift", type=int, default=0,
                    help="oof / sim: OOF の窓の境界を何か月ずらすか（頑健性の確認。0 が本番と同じ窓）")
    ap.add_argument("--shifts", default="0,2,4", help="robust: 並べる切り方（か月）")
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--dataset", default=lab.DATASET)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-hist", type=int, default=U.MIN_HIST)
    args = ap.parse_args(argv)
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}（{ALGOS}）")
    which = [w for w in args.which.split(",") if w]
    t0 = time.time()
    if args.stage in ("label", "all"):
        label_stage(args.data_dir, args.dataset, args.out_dir)
    if args.stage in ("tune", "oof", "all"):
        frame = lab.frame()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Code"] = frame["Code"].astype(str)
        labels = load_labels(args.out_dir)
        if args.stage in ("tune", "all"):
            tune_stage(frame, labels, algos, args.out_dir, args.n_trials)
        if args.stage in ("oof", "all"):
            oof_stage(frame, labels, algos, which, args.data_dir, args.out_dir, args.shift)
    if args.stage in ("sim", "all"):
        sim_stage(args.data_dir, args.out_dir, args.min_hist, args.shift)
    if args.stage == "robust":
        robust_stage(args.data_dir, args.out_dir,
                     [int(x) for x in args.shifts.split(",") if x.strip()], args.min_hist)
    log(f"終わり（{(time.time() - t0) / 60:.1f}分）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
