#!/usr/bin/env python3
"""
実験65: 普通株以外（優先株など）を学習データ・推論データからそもそも外すと、5モデルの探索（5分割 CV）・
本番と同じ作りの out-of-fold・32窓がどう変わるか。

運用者の方針（2026-10-08）「優先株の除外は、そもそも学習データ、推論データから除外してくださいという意味
でした。なので、いつも通り、5cv+oof の検証から始めて下さい。」
きっかけ: 10/8 に伊藤園（優先株式, 25935）が候補に出た。決算は会社（普通株のコード）に付くので、優先株の行は
決算の特徴量がすべて欠損のまま採点されていた（docs/OPERATIONS.md「優先株など普通株以外を母集団から外した」）。

外し方（build_dataset.drop_non_common_shares、EXCLUDE_NON_COMMON）: J-Quants の5桁コードの末尾が 0 でない銘柄
を、ETF・REIT と同じ場所（横断面の順位を付ける前）で母集団から外す。外した行が無くなるうえ、その行と同じ日の
ほかの銘柄の順位（*_r）も変わる。

腕（生データは同じ。列は本番の239列）
  A  いまの本番（外さない）。A のデータで探索
  B  外す。B のデータで探索 → 日曜の週次実行がやること
  C  外したデータを、A のパラメータで（探索なし）→ データの違いだけの効果
  B−A が「日曜の再学習で起きること」、C−A が「外すことそのものの効果」、B−C が「探索し直しの揺れ」

**比べる行は、A と B の両方にある行（= B の行）だけ。** B には外した銘柄の行が無いので、A の予測からも
その行を除いて比べる（学習には A は外した行も使っている。それが違いの中身）。参考に、A を自分の行すべてで
測った値も出す。

探索は本番の週次実行と同じ関数・条件（50試行 × 5分割 year_cap_date・ホールドアウトより前・種0）:
  LightGBM は run_tuning.py と同じ tuning.tune、ほか4モデルは tuning_multi.tune（前処理は本番の設定）。
  探索の CV は腕ごとの行（A は外す銘柄の行も含む）で測るので、厳密には同じ行どうしではない（パラメータ選び用）

評価（実験57・58・64 と同じ作り）
  0. データの違い: 外した行（探索の期間 / OOF の期間・正例）と、残った行で値が変わった列・行の数
  1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用）: PR-AUC ± SD・ROC・分割ごとの PR-AUC・所要
  2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）: 共通の行で
     PR-AUC / リフト / ROC-AUC / 日内 AUC / 上位10% の ret_o1_20 の超過（e25_auc_noise.metrics）
  3. 窓ごと: 境界を 0/2/4か月ずらした3通り（計32窓）。種は logit 1つ（決定的）、ほか3つの平均。
     共通の行で B−A / C−A / B−C の平均・SE・上の窓の数

Actions の1回の上限（330分）に収まるよう、モデルを分けて回す（実験64 と同じ）。データの表（両方の腕）は
1回目に作って research/_data/oof/<tag>_* に置き、以降の回はそれを使う。途中の結果も同じ場所に置き、
run-experiment のキャッシュで次の回に引き継ぐ。表は、結果のそろったモデルをすべて出す。
    exp=e65_non_common.py args="--algos lgbm,logit,mlp"
    exp=e65_non_common.py args="--algos xgb"
    exp=e65_non_common.py args="--algos cat"

試運転（本番の回とキャッシュを混ぜないよう、名前を変える）:
    exp=e65_non_common.py args="--tag e65smoke --algos logit --n-trials 2 --shifts 0"

公開ログには件数・割合・日付・精度だけを出す（銘柄のコードや名前は出さない）。
本番の設定（research/lgbm_params.json、research/multi_params.json、build_dataset.EXCLUDE_NON_COMMON）には書かない。

結果（2026-10-08、run 37770920219 / 37770967228 / 37786306233、docs/MODEL_ADOPTION_RULES.md §26）: 外れるのは
4行（1銘柄）で、本番の239列の値は変わらない。データだけの差（C−A）は5モデルとも 32窓で ±1.75 SE 以内。探索し直し
（B−C）の揺れのほうが大きく、向きはそろわない → 運用者の方針どおり外す（本番の EXCLUDE_NON_COMMON を True にした）。
そのため、この台本で A（外さない）を作り直すことはもうできない（prepare が止まる。記録は §26 と Actions の成果物）。
"""
from __future__ import annotations

import argparse
import glob
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
DATA_OF = {"A": "A", "B": "B", "C": "B"}       # どのデータの表
PARAMS_OF = {"A": "A", "B": "B", "C": "A"}     # どの探索のパラメータ
LABELS = {"A": "A いまの本番（外さない）", "B": "B 普通株以外を外す・探索し直し",
          "C": "C 外したデータ・A のパラメータ"}
PAIRS = (("A", "B"), ("A", "C"), ("C", "B"))
TAG = "e65"                                    # 保存するファイルの頭（--tag で変える）
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
    """out-of-fold のうち、keys（共通の行）にある行だけ。"""
    return o[keys_of(o).isin(keys)].reset_index(drop=True)


def is_non_common(code: pd.Series) -> np.ndarray:
    """drop_non_common_shares と同じ決まり（5桁で末尾が 0 でない）。"""
    c = code.astype(str).str.strip()
    return ((c.str.len() == 5) & ~c.str.endswith("0")).to_numpy()


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
def build_with(exclude: bool, out_path: str) -> pd.DataFrame:
    """build_dataset.build を EXCLUDE_NON_COMMON を切り替えて回す（終わったら戻す）。"""
    keep = B.EXCLUDE_NON_COMMON
    B.EXCLUDE_NON_COMMON = exclude
    try:
        d = B.build(lab.DATA_DIR, out_path)
    finally:
        B.EXCLUDE_NON_COMMON = keep
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    return d


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
    腕 A / B の表（lab.frame と同じ列: データセット + 実収益）と記録を返す。

    A は lab.frame()（このワークフローの Build dataset がいまの本番のコード = 外さない で作った dataset.parquet
    から）。同じコードでもう一度作った A と全列一致することを確かめる（作りが決定的か）。B は外して作った
    データセットに、A の実収益の列を (銘柄, 日付) で付けたもの。B の行が「A の行から外す銘柄の行を除いたもの」
    （並びも同じ）で、残った行のラベルが A と同じことを確かめる。
    """
    pa, pb, ps = path("frame_A.parquet"), path("frame_B.parquet"), path("stamp.json")
    if not rebuild and all(os.path.exists(p) for p in (pa, pb, ps)):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・A {stamp['rows_A']:,}行 / "
            f"B {stamp['rows_B']:,}行・〜{stamp['date_max']}）")
        return pd.read_parquet(pa), pd.read_parquet(pb), stamp
    os.makedirs(OOF_DIR, exist_ok=True)
    if B.EXCLUDE_NON_COMMON:
        raise SystemExit("本番の EXCLUDE_NON_COMMON が True になっている。A（外さない）が作れない")
    t0 = time.time()
    fa = lab.frame(rebuild=True)
    fa["Date"] = pd.to_datetime(fa["Date"])
    fa["Code"] = fa["Code"].astype(str)
    log(f"[data] A（lab.frame、いまの本番 = 外さない）{len(fa):,}行 × {fa.shape[1]}列 / {time.time()-t0:.0f}秒")

    t0 = time.time()
    da = build_with(False, path("dataset_A.parquet"))
    log(f"[data] A をもう一度作った {len(da):,}行 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    db = build_with(True, path("dataset_B.parquet"))
    log(f"[data] B（普通株以外を外す）{len(db):,}行 / {time.time()-t0:.0f}秒")

    # A をもう一度作ったものが lab.frame と行・全列で一致するか（作りが決定的か）
    if list(zip(da["Code"], da["Date"])) != list(zip(fa["Code"], fa["Date"])):
        raise SystemExit("A をもう一度作った行（銘柄・日付・並び）が lab.frame と違う")
    feat = [c for c in da.columns if c in fa.columns and c not in KEY]
    redo = [c for c in feat if not same(da[c], fa[c]).all()]
    print(f"  作りが決定的か: いまの本番のコードで2回作ったデータセットの {len(feat)}列のうち、"
          f"値が違う列 {len(redo)}本" + (f" → {redo[:12]}" if redo else "（全列一致）"))

    # B の行 = A の行から外す銘柄の行を除いたもの（並びも同じ）
    drop = is_non_common(fa["Code"])
    kept = fa.loc[~drop, KEY].reset_index(drop=True)
    if list(zip(db["Code"], db["Date"])) != list(zip(kept["Code"], kept["Date"])):
        raise SystemExit("B の行が「A の行から普通株以外を除いたもの」と違う")
    out_cols = [c for c in fa.columns if c not in db.columns]
    fb = db.merge(fa[KEY + out_cols], on=KEY, how="left", validate="one_to_one")
    fak = fa.loc[~drop].reset_index(drop=True)
    if not same(fb["label"], fak["label"]).all():
        raise SystemExit("残った行のラベルが A と B で違う")
    feat_b = [c for c in db.columns if c in fa.columns and c not in KEY + ["label"]]
    changed = [c for c in feat_b if not same(fb[c], fak[c]).all()]

    lab_drop = fa.loc[drop, "label"]
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
             "rows_A": int(len(fa)), "rows_B": int(len(fb)), "date_max": str(fa["Date"].max().date()),
             "dropped_rows": int(drop.sum()), "dropped_codes": int(fa.loc[drop, "Code"].nunique()),
             "dropped_dates": sorted(str(d.date()) for d in fa.loc[drop, "Date"].unique()),
             "dropped_pos": int((lab_drop == 1).sum()), "dropped_labeled": int(lab_drop.notna().sum()),
             "changed_columns": changed, "nondeterministic_columns": redo}
    fa.to_parquet(pa, index=False)
    fb.to_parquet(pb, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1)
    for p in glob.glob(path("dataset_*")):     # キャッシュを小さくする（表に入れたので要らない）
        os.remove(p)
    return fa, fb, stamp


def show_diff(fa: pd.DataFrame, fb: pd.DataFrame, stamp: dict, cols: list, cutoff, oof_start) -> dict:
    print("\n■ 0. データの違い（A いまの本番 / B 普通株以外を外す）")
    if stamp.get("nondeterministic_columns"):
        print(f"  ※ いまの本番のコードで2回作って値が違った列: {stamp['nondeterministic_columns']}")
    drop = is_non_common(fa["Code"])
    d = fa["Date"]
    in_tune = (d <= cutoff).to_numpy()
    in_oof = ((d >= oof_start) & fa["label"].notna()).to_numpy()
    print(f"  A {len(fa):,}行 → B {len(fb):,}行。外した行 {int(drop.sum()):,}（{stamp['dropped_codes']}銘柄・"
          f"{len(stamp['dropped_dates'])}日）/ 探索の期間 {int((drop & in_tune).sum()):,} / "
          f"OOF の検証の期間 {int((drop & in_oof).sum()):,}")
    print(f"  外した行のうちラベルが確定 {stamp['dropped_labeled']:,}・正例 {stamp['dropped_pos']:,}")
    if stamp["dropped_dates"]:
        print(f"  外した行の日付: {stamp['dropped_dates'][0]} 〜 {stamp['dropped_dates'][-1]}")

    # 残った行で値が変わった列（同じ日の順位 *_r のはず）
    fak = fa.loc[~drop].reset_index(drop=True)
    db_ = fb["Date"]
    k_tune = (db_ <= cutoff).to_numpy()
    k_oof = ((db_ >= oof_start) & fb["label"].notna()).to_numpy()
    out = {"dropped_rows": int(drop.sum()), "dropped_tune": int((drop & in_tune).sum()),
           "dropped_oof": int((drop & in_oof).sum()), "columns": {}}
    changed = stamp.get("changed_columns", [])
    print(f"  残った行（{len(fb):,}）で値が変わった列 {len(changed)}本（本番の239列に入るもの "
          f"{len([c for c in changed if c in cols])}本）")
    if changed:
        print(f"  {'列':<28}{'本番の列':>8}{'違う行':>8}{'探索の期間':>10}{'OOF の期間':>10}")
    rows_any = np.zeros(len(fb), dtype=bool)
    for c in changed:
        diff = ~same(fa.loc[~drop, c].reset_index(drop=True), fb[c])
        rows_any |= diff
        out["columns"][c] = {"rows": int(diff.sum()), "tune": int((diff & k_tune).sum()),
                             "oof": int((diff & k_oof).sum()), "in_production": c in cols}
        print(f"  {c:<28}{'○' if c in cols else '':>8}{int(diff.sum()):>8,}{int((diff & k_tune).sum()):>10,}"
              f"{int((diff & k_oof).sum()):>10,}")
    print(f"  どれかの列が違う残った行: {int(rows_any.sum()):,}（{rows_any.mean()*100:.2f}%）/ "
          f"探索の期間 {int((rows_any & k_tune).sum()):,} / OOF の期間 {int((rows_any & k_oof).sum()):,}")
    out["rows_any"] = int(rows_any.sum())
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
    compute=False なら無いとき None。名前は腕でなく「データ × パラメータ」で付ける（A と B の探索が同じ
    パラメータを選んだときは、B と C が同じ計算を共有する）。
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


def same_folds(fa: pd.DataFrame, fb: pd.DataFrame, shifts: list) -> None:
    """A と B で窓の境界が同じか（違えば比べられないので止める）。"""
    for sh in sorted(set([0] + list(shifts))):
        xa = [(f.train_start, f.train_end, f.test_start, f.test_end) for f in E41.folds_for(fa["Date"], sh)]
        xb = [(f.train_start, f.train_end, f.test_start, f.test_end) for f in E41.folds_for(fb["Date"], sh)]
        if xa != xb:
            raise SystemExit(f"ずらし{sh}か月の窓の境界が A と B で違う（A {len(xa)} / B {len(xb)}窓）")
    print(f"  窓の境界は A と B で同じ（ずらし {sorted(set([0] + list(shifts)))}か月）")


def pair(name: str, a: np.ndarray, b: np.ndarray) -> tuple:
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    line = (f"  {name:<20}{a.mean():>9.4f}{b.mean():>9.4f}{d.mean():>+10.4f}{se:>9.4f}"
            f"{int((d > 0).sum()):>5}/{len(d):<4}{int((d == 0).sum()):>6}")
    return line, {"n": int(len(d)), "mean_diff": float(d.mean()), "se": float(se),
                  "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}


def run(algo: str, frames: dict, keys: pd.MultiIndex, cols: list, cutoff, n_trials: int, stamp: dict,
        shifts: list, compute: bool) -> dict | None:
    """1モデルぶん。compute=False なら保存済みだけで表を作る（そろっていなければ None）。"""
    recs = {}
    for arm in ("A", "B"):
        recs[arm] = tune(algo, arm, frames[arm], cols, cutoff, n_trials, stamp, compute)
        if recs[arm] is None:
            return None
    params = {arm: recs[PARAMS_OF[arm]]["params"] for arm in ARMS}
    prod, prod_all_a = {}, None
    for arm in ARMS:
        o = oof(algo, DATA_OF[arm], frames[DATA_OF[arm]], cols, params[arm], 0, (PROD_SEED,), compute)
        if o is None:
            return None
        if arm == "A":
            prod_all_a = o
        prod[arm] = on_rows(o, keys)
    wins = {}
    for sh in shifts:
        for arm in ARMS:
            o = oof(algo, DATA_OF[arm], frames[DATA_OF[arm]], cols, params[arm], sh, SEEDS[algo], compute)
            if o is None:
                return None
            wins[(sh, arm)] = AB.auc_by_window(on_rows(o, keys))
    return {"recs": recs, "params": params, "prod": prod, "prod_all_a": prod_all_a, "wins": wins}


def report(results: dict, shifts: list, diff: dict) -> None:
    algos = [a for a in ALGOS if results.get(a)]
    missing = [a for a in ALGOS if not results.get(a)]
    print("\n" + "=" * 78)
    print(f"結果のそろったモデル: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": diff, "tuning": {}, "production_oof": {}, "windows": {}}

    print("\n■ 1. 探索の CV（層別5分割。楽観側に出る。パラメータ選び用。腕ごとの行で測る）")
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

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。共通の行）")
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
        ma = metrics(results[a]["prod_all_a"])
        summary["production_oof"][f"{a}_A_all_rows"] = {k: float(v) for k, v in ma.items()}
        print(f"  {a:<7}参考 A を自分の行すべてで（{len(results[a]['prod_all_a']):,}件）: PR-AUC {ma['pr_auc']:.4f} / "
              f"ROC {ma['roc_auc']:.4f} / 日内 {ma['day_auc']:.4f}")

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
              "logit は種1つ、ほかは種3つの平均。共通の行）")
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
    global TAG
    ap = argparse.ArgumentParser(description="実験65: 普通株以外を学習・推論データから外す前後")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算するモデル")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="データの表を作り直す（以前の結果は使えなくなる）")
    args = ap.parse_args(argv)
    TAG = args.tag
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らないモデル: {bad}")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験65 普通株以外（優先株など）を学習・推論データから外す前後（5分割 CV の探索 + OOF + 32窓）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 指紋 {F.signature(cols)}")
    for arm in ARMS:
        print(f"  {LABELS[arm]}")
    print(f"  この回に計算: {', '.join(algos)} / 探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}）/ "
          f"窓のずらし {shifts}か月 / 種 {SEEDS} / 保存 {TAG}_*")
    print("=" * 78)

    fa, fb, stamp = prepare(args.rebuild)
    for f in (fa, fb):
        f["Date"] = pd.to_datetime(f["Date"])
        f["Code"] = f["Code"].astype(str)
    miss = [c for c in cols if c not in fb.columns]
    if miss:
        raise SystemExit(f"データに無い列: {miss[:8]}")
    cutoff, _, _ = T.holdout_bounds(fa["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    cutoff_b, _, _ = T.holdout_bounds(fb["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    if cutoff != cutoff_b:
        raise SystemExit(f"探索の期間の終わりが A と B で違う（{cutoff.date()} / {cutoff_b.date()}）")
    same_folds(fa, fb, shifts)
    oof_start = pd.Timestamp(min(f.test_start for f in E41.folds_for(fa["Date"], 0)))
    diff = show_diff(fa, fb, stamp, cols, cutoff, oof_start)
    frames = {"A": fa, "B": fb}
    keys = keys_of(fb)

    results = {}
    for a in ALGOS:
        t0 = time.time()
        results[a] = run(a, frames, keys, cols, cutoff, args.n_trials, stamp, shifts, compute=a in algos)
        if a in algos:
            log(f"[{a}] {time.time()-t0:.0f}秒")
    report(results, shifts, diff)
    log(f"記録: {OOF_DIR}/{TAG}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
