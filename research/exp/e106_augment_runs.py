#!/usr/bin/env python3
"""
実験106（データ拡張の案1）: 同じ上昇局面の2日目以降の更新日（いまは BREAKOUT_COOLDOWN で捨てている実在の行）を学習の側だけに
足すと、LightGBM（1.5σ）の out-of-fold の分離力は上がるか。本番と同じ作りの out-of-fold・32窓で、今のラベルの PR-AUC（§7 と
同じ物差し）で比べる（検証だけ）。

運用者の依頼（2026-10-18）「金融時系列データのオーグメンテーション手法について」→ 調査（非公開の Artifact「ブレイク予測の
データ拡張」https://claude.ai/artifact/QmsTiMPjrMAUwQa98AEsQa ）→ 案0 の数え上げ（probe_augment_rows.py。run 39115463637。
docs/MODEL_ADOPTION_RULES.md §67）: 足せる行 123,081（いまの 5.61倍）・正例率 1.5σ 13.0%（いまも 13.0%）。ラベル期間の重なりで
数えた実効の行数は 21,957 → 41,004（1.87倍）で、増える分は足せる行のある局面の数（19,078）とほぼ同じ。足す行は base_length で
いまの行と完全に分かれる（KS 1.000）→ 運用者の選択（AskUserQuestion）: 「案1 を比べる」・足す行の重みは「局面ごとに合計1」。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです」→ 本番の設定（research/features.py・research/lgbm_params.json・
build_dataset.py）には書かない。

足す行
  78週高値の更新日のうち、新規のブレイクではない日（直前20営業日に更新がある日）。除外の条件・特徴量は本番と同じ（本番の
  build_dataset.build() を probe_augment_rows の差し替え（mark_new_highs だけ）で回した表。作り方・置き場所は probe と同じ
  pa0_*）。ラベルは rise_label_for_k の 1.5σ（いまの行と同じ式）。局面（新規のブレイクから、同じ銘柄の次の新規のブレイクの
  前までの更新日）は probe_augment_rows.assign_clusters で付ける。局面の新規の行が学習データに無い足す行（流動性などで新規の
  日だけ落ちた）も実在の行なので足す。

腕（評価はどの腕も、いまの行（新規のブレイク）・今のラベル 1.5σ。窓はいまの行の日付で作る。パラメータはすべて A のもの）
  A  いまの行だけ・重みなし。249列（回した時点の既定）で探索（Optuna 50試行 × 5分割。週次の再学習と同じ関数・条件）
  G  いまの行（重み1）+ 足す行。足す行は窓ごとの学習の期間（日付が学習の最終日以前。エンバーゴはそのまま）の中だけで、重みは
     その窓の学習の中で局面ごとに合計1（同じ局面の足す行が k行なら各 1/k）                    ← 判定
  D  対照: G と同じ件数を、その窓の学習のいまの行から無作為に（重複あり）複製して足す。重みの合計は G の足す行と同じ
     （1行あたり 合計 ÷ 件数）。行と重みは増えるが、新しい行・ラベルは無い。種は PERM_SEED + 100 × ずらし + 窓の番号
  P  対照: G の足す行のラベルを、同じ日付の足す行の中で入れ替えたもの（重みは G と同じ）。行の特徴量と日ごとの正例の数は
     そのままで、銘柄ごとのラベルの情報は無い。種は PERM_SEED + 100 × ずらし
  正例の重み scale_pos_weight は、重みを掛けた負例 ÷ 正例（重みなしの A は 負例数 ÷ 正例数。実験104 と同じ）

判定（回す前に決めた。2026-10-18）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）の1モデル分の3条件を、32窓（ずらし 0/2/4か月 × 種3つの平均）の今のラベルの
  PR-AUC に当てる: 窓ごとの G−A の平均が正・対照の平均を上回る（対照は P−A と D−A の大きいほう。つまり両方を上回る）・G が上の
  窓が過半（n//2 + 1 以上）。木3モデルの票は数えられない（lgbm だけ）ので、本番に入れるかは結果を見て運用者が決める
  （実験101〜104 と同じ扱い）。

出すもの（公開ログには件数・割合・分位・日付・精度だけ。銘柄・値は出さない）
  0. 確かめ（更新日すべての表の新規の行が、いまの学習データと銘柄×日付・249列・ラベル（1.2σ・1.5σ）ですべて一致。base_length
     での判定と本番の印が一致）と、足す行の数・局面・正例率・P の入れ替え
  1. 探索の CV（A）
  2. 本番と同じ作りの out-of-fold（ずらし0・種42）: 腕ごとの PR-AUC・リフト・ROC・日内 AUC など、年ごとの PR-AUC ÷ 正例率、
     最後の窓の学習の行と足す行の重みの合計
  3. 32窓の差（PR-AUC・ROC）の平均・SE・上の窓
  4. §7 の判定
  5. 窓の中の上位2% の Precision（今のラベル）・+10% の指値に届いた割合・1取引の平均（実験83 の出口。参考）
  6. 列の重要度（gain の割合。本番と同じ作りの out-of-fold の窓の平均）: 土台の4列と、A・G の上位10列
  7. 窓ごとの値の一覧

    exp=e106_augment_runs.py                       ← 回すのはこの1本（lgbm 1.5σ）
足す行の表だけ作って確かめる: exp=e106_augment_runs.py args="--dry"
試運転:                       exp=e106_augment_runs.py args="--tag e106smoke --n-trials 2 --shifts 0"

Run Experiment は同じ実験を同時に1本しか走らせない。表は1回目に作って research/_data/oof/ に置き（評価の行と足す行は
<tag>_*、更新日すべての表は probe と同じ pa0_*）、以降の回は actions/cache で引き継いだそれを使う。--budget-min（既定 290分）を
過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。本番と同じ作りの out-of-fold を
4腕そろえてから、ずらしごとに A・G・P・D の順で回す。
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
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e104_market_drift as E104  # noqa: E402
import probe_augment_rows as PA  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 週次の再学習（retrain-weekly.yml）と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
#: ラベルの到達しきい値（σ の倍数）。§62〜§66 と同じく 1.5σ だけ
K = 1.5
#: 列の組。回した時点の既定（249列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent"
#: 対照 D の複製・P の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261018
ARMS = ("A", "G", "P", "D")
#: 判定の対照（大きいほうを使う）
CONTROLS = ("P", "D")
LABELS = {"A": "A  いまの行だけ・重みなし・探索したパラメータ（基準）",
          "G": "G  いまの行 + 足す行（局面ごとに重みの合計1）・A のパラメータ",
          "P": "P  対照: G の足す行のラベルを同じ日付の中で入れ替え・A のパラメータ",
          "D": "D  対照: G と同じ件数・同じ重みの合計を、いまの行の無作為な複製で足す・A のパラメータ"}
#: 表に出す差（「後 − 前」）
PAIRS = (("A", "G"), ("A", "P"), ("A", "D"), ("P", "G"), ("D", "G"))
#: 足す行といまの行を分ける土台の列（案0 で KS が 0.3 を超えた4列）
BASE_COLS = ("base_length", "broken_high_age", "base_depth", "pullback_depth")
TOP_IMP = 10
#: 特徴量の一致を見るときの許し（同じ関数・同じデータなので本来は完全に一致する）
RTOL, ATOL = 1e-9, 1e-12
#: +10% の指値（実験83・運用の今の出口）
TP = LT.TAKE_PROFIT
TAG = "e106"
RUN = "e106_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（評価の行・足す行）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """探索・out-of-fold・要約。"""
    return os.path.join(OOF_DIR, f"{RUN}_{name}")


def time_left() -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min を過ぎたので、新しい計算は始めない（保存済みの分だけで報告。続きは次の回）")
    return False


# --------------------------------------------------------------------------- #
# 表（いまの行と、更新日すべての表から作る足す行）
# --------------------------------------------------------------------------- #
def same_values(a, b, rtol: float = RTOL, atol: float = ATOL) -> np.ndarray:
    """行ごとに a と b が同じか（欠測どうしは同じ）。"""
    x = pd.to_numeric(pd.Series(a).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(b).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol)


def compare_fresh(ds: pd.DataFrame, fresh: pd.DataFrame, cols: list) -> dict:
    """
    いまの学習データ（ds）と、更新日すべての表の新規の行（fresh）を銘柄×日付で突き合わせ、行・列（cols と label・y15）が
    すべて一致するかを数える。bad_cols は値が食い違った列と行の数。
    """
    kd, kf = PA.keys_of(ds), PA.keys_of(fresh)
    out = {"ds": int(len(ds)), "fresh": int(len(fresh)), "only_ds": int((~kd.isin(kf)).sum()),
           "only_fresh": int((~kf.isin(kd)).sum()), "dup": int(kd.duplicated().sum() + kf.duplicated().sum())}
    use = ["Code", "Date"] + list(cols) + ["label", "y15"]
    m = ds[use].merge(fresh[use], on=["Code", "Date"], how="inner", suffixes=("", "_t"))
    bad = {}
    for c in list(cols) + ["label", "y15"]:
        n = int((~same_values(m[c], m[f"{c}_t"])).sum())
        if n:
            bad[c] = n
    out["matched"] = int(len(m))
    out["bad_cols"] = bad
    return out


def check_fresh_or_stop(c: dict) -> None:
    print(f"  確かめ: 更新日すべての表の新規の行（base_length が{PA.COOLDOWN + 1}以上か欠測）{c['fresh']:,}行 と "
          f"いまの学習データ {c['ds']:,}行: 銘柄×日付が一致 {c['matched']:,} / 表だけ {c['only_fresh']:,} / "
          f"dataset だけ {c['only_ds']:,} / 重なり {c['dup']:,}・値が食い違う列 {len(c['bad_cols'])}（249列とラベル 1.2σ・{K:.1f}σ）")
    if c["only_ds"] or c["only_fresh"] or c["dup"] or c["bad_cols"]:
        worst = sorted(c["bad_cols"].items(), key=lambda kv: -kv[1])[:8]
        raise SystemExit("更新日すべての表の新規の行が、いまの学習データと合わない（足す行の特徴量の作りがずれている）: "
                         + ", ".join(f"{k} {v:,}行" for k, v in worst))


def assemble(frame: pd.DataFrame, al: pd.DataFrame, updates: pd.DataFrame, cols: list) -> tuple:
    """
    frame（lab.frame = いまの学習データ）と al（更新日すべての表）・updates（控えた更新日）から、
    評価の行（いまの行。label は 1.5σ）と足す行（Code・Date・cols・label（1.5σ）・run（局面）・offset）を作り、確かめを数える。
    """
    frame, al, updates = frame.copy(), al.copy(), updates.copy()
    for x in (frame, al, updates):                  # 突き合わせの鍵をそろえる（parquet の時刻の単位の違いで外れないように）
        x["Code"] = x["Code"].astype(str)
        x["Date"] = pd.to_datetime(x["Date"]).astype("datetime64[ns]")
    frame["y15"] = B.rise_label_for_k(frame, K)
    al["y15"] = B.rise_label_for_k(al, K)
    is_fresh = PA.fresh_by_base_length(al["base_length"])
    checks = {"fresh": compare_fresh(frame, al[is_fresh], cols)}
    check_fresh_or_stop(checks["fresh"])

    u = updates[["Code", "Date", "fresh"]].copy()
    m = al[["Code", "Date"]].assign(_bl=is_fresh).merge(u, on=["Code", "Date"], how="left")
    if len(m) != len(al):
        raise SystemExit("控えた更新日との突き合わせで行が増えた（更新日の Code・Date が重なっている）")
    known = m["fresh"].notna()
    flag = {"rows": int(len(m)), "same": int((known & (m["_bl"] == m["fresh"].astype(bool))).sum()),
            "diff": int((known & (m["_bl"] != m["fresh"].astype(bool))).sum()), "no_update": int((~known).sum())}
    checks["flag"] = flag
    print(f"  確かめ: base_length での判定と本番の印（直前{PA.COOLDOWN}営業日に更新が無い日）: 一致 {flag['same']:,} / "
          f"食い違い {flag['diff']:,} / 控えた更新日に無い行 {flag['no_update']:,}")
    if flag["diff"] or flag["no_update"]:
        raise SystemExit("base_length での判定が本番の印と合わない")

    cl = PA.assign_clusters(updates)[["Code", "Date", "cluster", "offset"]]
    ad = al.loc[~is_fresh, ["Code", "Date"] + list(cols) + ["y15"]].merge(cl, on=["Code", "Date"], how="left")
    if len(ad) != int((~is_fresh).sum()):
        raise SystemExit("局面の突き合わせで行が増えた")
    no_run = ad["cluster"].isna() | (ad["cluster"] < 0)
    no_label = ad["y15"].isna()
    fresh_runs = set(cl.merge(frame[["Code", "Date"]], on=["Code", "Date"], how="inner")["cluster"].astype(int))
    ad = ad[~no_run & ~no_label].rename(columns={"y15": "label", "cluster": "run"}).reset_index(drop=True)
    ad["run"] = ad["run"].astype(np.int64)
    checks["added"] = {"rows": int((~is_fresh).sum()), "no_run": int(no_run.sum()), "no_label": int(no_label.sum()),
                       "used": int(len(ad)), "runs": int(ad["run"].nunique()),
                       "orphan": int((~ad["run"].isin(fresh_runs)).sum())}
    fr = frame[["Code", "Date"] + list(cols) + ["y15", "ret_o1_20", "ret_o1_40"]].rename(columns={"y15": "label"})
    fr = fr[fr["label"].notna()].reset_index(drop=True)
    checks["eval_rows"] = int(len(fr))
    return fr, ad, checks


def prepare(cols: list, rebuild: bool = False):
    """評価の行（いまの行。今のラベル 1.5σ と出口つき）と足す行の表・記録。1回目に作って、以降の回は同じものを使う。"""
    pf, pad, ps = fpath("fresh.parquet"), fpath("added.parquet"), fpath("stamp.json")
    if not rebuild and all(os.path.exists(p) for p in (pf, pad, ps)):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        fr, ad = pd.read_parquet(pf), pd.read_parquet(pad)
        miss = [c for c in cols if c not in fr.columns or c not in ad.columns]
        if not miss:
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・いまの行 {stamp['rows']:,}・足す行 "
                f"{stamp['added']:,}・〜{stamp['date_max']}）")
            return fr, ad, stamp
        log(f"[data] 保存済みの表に無い列がある（{miss[:5]}）ので作り直す")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    miss = [c for c in cols + ["vol_20d", "future_rise", "end_level", "uptrend_end", "label", "ret_o1_20", "ret_o1_40"]
            if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")

    t0 = time.time()
    t = PA.build_tables(lab.DATA_DIR, with_close=False, rebuild=rebuild)
    al = t["all"]
    miss = [c for c in cols + ["vol_20d", "future_rise", "end_level", "uptrend_end", "label"] if c not in al.columns]
    if miss:
        raise SystemExit(f"更新日すべての表に無い列: {miss[:8]}")
    log(f"[data] 更新日すべての表 {len(al):,}行 / {time.time()-t0:.0f}秒")
    print("\n■ 確かめ（足す行は本番の build_dataset.build() の差し替えで作った表から取る）")
    fr, ad, checks = assemble(frame, al, t["updates"], cols)
    del frame, al, t

    # --- 出口（実験83 と同じ +10% の指値。参考の物差し）--- #
    t0 = time.time()
    bars_px = LT.load_bars(lab.DATA_DIR, start=fr["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(fr[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(fr) or not (ex["Code"].to_numpy() == fr["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    k = E83.tp_key(TP)
    fr["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float)
    fr["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行・20日目の終値あり "
          f"{chk['with_close']:,} / {chk['rows']:,}行 → {'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}"
          f"（{time.time()-t0:.0f}秒）")
    if not exit_ok:
        fr["hit10"] = np.nan
        fr["tp10c"] = np.nan

    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(fr)), "added": int(len(ad)),
             "runs": int(ad["run"].nunique()), "date_min": str(fr["Date"].min().date()),
             "date_max": str(fr["Date"].max().date()), "checks": checks, "cols": len(cols), "signature": F.signature(cols)}
    fr.to_parquet(pf, index=False)
    ad.to_parquet(pad, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return fr, ad, stamp


# --------------------------------------------------------------------------- #
# 足す行の重み・対照
# --------------------------------------------------------------------------- #
def run_weights(runs) -> np.ndarray:
    """局面ごとに合計1の重み（同じ局面の k行に各 1/k）。"""
    r = pd.Series(np.asarray(runs))
    return (1.0 / r.map(r.value_counts()).to_numpy(dtype=float)) if len(r) else np.zeros(0)


def shuffle_within_date(labels, dates, seed: int) -> np.ndarray:
    """ラベルを同じ日付の行の中で入れ替える（日付ごとの正例の数は変わらない。1行だけの日はそのまま）。"""
    y = np.asarray(labels, dtype=float)
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = y.copy()
    if not len(y):
        return out
    rng = np.random.default_rng(seed)
    order = np.argsort(d, kind="mergesort")
    ds = d[order]
    cut = np.r_[0, np.flatnonzero(ds[1:] != ds[:-1]) + 1, len(ds)]
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a > 1:
            idx = order[a:b]
            out[idx] = y[rng.permutation(idx)]
    return out


def training_set(arm: str, Xf: np.ndarray, yf: np.ndarray, fmask: np.ndarray, Xa: np.ndarray, ya: np.ndarray,
                 amask: np.ndarray, runs: np.ndarray, rng: np.random.Generator) -> tuple:
    """
    窓の学習の行（いまの行 fmask・足す行 amask）から、腕の学習の行列・ラベル・重み（A は None）と大きさを返す。
    G・P は足す行（P のラベルは呼ぶ側が入れ替えたもの）を局面ごとに合計1の重みで、D はいまの行の無作為な複製を
    G と同じ件数・同じ重みの合計で足す。いまの行の重みは 1。
    """
    X0, y0 = Xf[fmask], yf[fmask]
    if arm == "A":
        return X0, y0, None, {"rows": int(len(y0)), "extra": 0, "extra_weight": 0.0}
    w_add = run_weights(runs[amask])
    n_add, total = int(len(w_add)), float(w_add.sum())
    if arm in ("G", "P"):
        X1, y1, w1 = Xa[amask], ya[amask], w_add      # D は足す行の数と局面だけを使う（Xa・ya は見ない）
    elif arm == "D":
        idx = rng.integers(0, len(y0), n_add)
        X1, y1 = X0[idx], y0[idx]
        w1 = np.full(n_add, total / n_add) if n_add else np.zeros(0)
    else:
        raise ValueError(f"知らない腕: {arm}")
    X = np.vstack([X0, X1])
    y = np.r_[y0, y1]
    w = np.r_[np.ones(len(y0)), w1]
    return X, y, w, {"rows": int(len(y0)), "extra": n_add, "extra_weight": total}


# --------------------------------------------------------------------------- #
# 探索（週次の再学習と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, compute: bool = True):
    p = path("params_A.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not compute or not time_left():
        return None
    sub = frame.loc[(frame["Date"] <= cutoff) & frame["label"].notna()].copy()
    t0 = time.time()
    log(f"  [A] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
    params = tuning.tune(sub, cols, n_trials=n_trials, n_splits=N_SPLITS, embargo_days=T.EMBARGO_DAYS,
                         scheme=CV_SCHEME, objective=CV_OBJECTIVE, model="classifier", verbose=False)
    rec = {"params": tuning.params_for("x", store={"x": params}), "_cv": dict(tuning.LAST_CV)}
    rec["_seconds"] = round(time.time() - t0)
    rec["_n_trials"] = n_trials
    rec["_cutoff"] = str(cutoff.date())
    rec["_data"] = stamp["built_utc"]
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, default=float)
    cv = rec["_cv"]
    log(f"  [A] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）/ ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold（E41.oof_folds の LightGBM と同じ組み方に、足す行と重みを足したもの）
# --------------------------------------------------------------------------- #
OOF_KEEP = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40", "hit10", "tp10c"]


def oof_arm(fr: pd.DataFrame, ad: pd.DataFrame, cols: list, params: dict, seed: int, folds, arm: str,
            shift: int = 0, perm_base: int = PERM_SEED, with_importance: bool = False) -> tuple:
    """
    folds で out-of-fold を作る（評価はいまの行だけ）。A はいまの行だけ・重みなしで、E41.oof_folds の LightGBM と同じ結果になる。
    返すのは (out-of-fold, 窓ごとの学習の大きさ, 列の重要度（gain の割合の窓の平均。with_importance のときだけ）)。
    """
    import lightgbm as lgb

    if isinstance(params, dict) and isinstance(params.get("params"), dict):
        params = params["params"]
    d = pd.to_datetime(fr["Date"]).to_numpy()
    Xf = fr[cols].to_numpy(dtype=float)
    yf = fr["label"].to_numpy(dtype=float)
    ok = np.isfinite(yf)
    Xa = ya = None
    if arm == "A" or not len(ad):
        da, runs = np.zeros(0, dtype="datetime64[ns]"), np.zeros(0, dtype=np.int64)
    else:
        da = pd.to_datetime(ad["Date"]).to_numpy()
        runs = ad["run"].to_numpy()
        if arm in ("G", "P"):
            Xa = ad[cols].to_numpy(dtype=float)
            ya = ad["label"].to_numpy(dtype=float)
            if arm == "P":
                ya = shuffle_within_date(ya, ad["Date"], perm_base + 100 * shift)
    parts, sizes, imps = [], [], []
    for f in folds:
        end = np.datetime64(pd.Timestamp(f.train_end))
        tr = (d <= end) & ok
        te = (d >= np.datetime64(pd.Timestamp(f.test_start))) & (d <= np.datetime64(pd.Timestamp(f.test_end))) & ok
        if te.sum() < 200 or tr.sum() < 1000:
            continue
        rng = np.random.default_rng(perm_base + 100 * shift + int(f.index))
        X, y, w, info = training_set(arm, Xf, yf, tr, Xa, ya, da <= end, runs, rng)
        y = y.astype(int)
        m = lgb.LGBMClassifier(**{**params, "random_state": seed}, scale_pos_weight=E104.weighted_spw(y, w))
        if w is None:
            m.fit(X, y)
        else:
            m.fit(X, y, sample_weight=w)
        part = fr.loc[te, [c for c in OOF_KEEP if c in fr.columns]].copy()
        part["score"] = m.predict_proba(Xf[te])[:, 1]
        part["fold"] = f.index
        parts.append(part)
        sizes.append({"fold": int(f.index), "train_end": str(f.train_end), **info})
        if with_importance:
            g = m.booster_.feature_importance(importance_type="gain").astype(float)
            imps.append(g / g.sum() if g.sum() > 0 else g)
    imp = dict(zip(cols, np.mean(imps, axis=0).tolist())) if imps else {}
    return pd.concat(parts, ignore_index=True), sizes, imp


def oof(arm: str, fr: pd.DataFrame, ad: pd.DataFrame, cols: list, params: dict, shift: int, seeds,
        info_out: dict | None = None):
    """腕 × パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。info_out を渡すと学習の大きさ・重要度も。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(fr.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{arm}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    infos = [f[: -len(".parquet")] + "_info.json" for f in files]
    want_info = info_out is not None
    done = [os.path.exists(f) and (not want_info or os.path.exists(i)) for f, i in zip(files, infos)]
    if not all(done) and not time_left():
        return None
    folds = E41.folds_for(fr["Date"], shift) if not all(done) else None
    parts, recs = [], []
    for sd, f, i, ok in zip(seeds, files, infos, done):
        if ok:
            parts.append(pd.read_parquet(f))
            if want_info:
                with open(i, encoding="utf-8") as fh:
                    recs.append(json.load(fh))
            continue
        if not time_left():
            return None
        t0 = time.time()
        o, sizes, imp = oof_arm(fr, ad, cols, params, sd, folds, arm, shift, PERM_SEED, want_info)
        o.to_parquet(f, index=False)
        rec = {"sizes": sizes, "imp": imp, "seconds": round(time.time() - t0)}
        if want_info:
            with open(i, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, default=float)
            recs.append(rec)
        parts.append(o)
        log(f"    {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    if want_info and recs:
        info_out[arm] = recs[0]
    return o


def run(arms: list, fr: pd.DataFrame, ad: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict,
        shifts: list) -> dict:
    """本番と同じ作りの out-of-fold を先に全腕、そのあとずらしごとに A・G・P・D。時間切れの腕は結果に入らない。"""
    res = {"recs": {}, "params": {}, "prod": {}, "oofs": {}, "info": {}}
    res["recs"]["A"] = tune(fr, cols, cutoff, n_trials, stamp)
    if res["recs"]["A"] is None:
        return res
    params = res["recs"]["A"]["params"]
    for arm in arms:
        res["params"][arm] = params
        o = oof(arm, fr, ad, cols, params, 0, (PROD_SEED,), res["info"])
        if o is not None:
            res["prod"][arm] = o
    for sh in shifts:
        for arm in arms:
            o = oof(arm, fr, ad, cols, params, sh, SEEDS)
            if o is not None:
                res["oofs"][(sh, arm)] = o
    return res


def complete_arms(res: dict, shifts: list) -> list:
    return [a for a in ARMS if a in res["prod"] and all((sh, a) in res["oofs"] for sh in shifts)]


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #
def data_report(fr: pd.DataFrame, ad: pd.DataFrame, stamp: dict) -> dict:
    """評価の行と足す行の数・局面・正例率・P の入れ替え（件数・割合・分位だけ）。"""
    c = stamp.get("checks", {})
    a = c.get("added", {})
    print(f"\n■ 0. 表（いまの行 {len(fr):,}・{fr['Date'].min().date()}〜{fr['Date'].max().date()}。足す行 {len(ad):,}。{K:.1f}σ）")
    if c.get("fresh"):
        f = c["fresh"]
        print(f"  確かめ（表を作った回）: 新規の行と dataset.parquet の一致 {f['matched']:,} / 表だけ {f['only_fresh']} / "
              f"dataset だけ {f['only_ds']} / 値が食い違う列 {len(f['bad_cols'])}・base_length の判定と本番の印の食い違い "
              f"{c.get('flag', {}).get('diff', '—')}")
    if a:
        print(f"  足せる行（除外の後）{a['rows']:,} → 局面の無い行 {a['no_run']:,}・ラベルの決まらない行 {a['no_label']:,} を除いて "
              f"{a['used']:,}（局面 {a['runs']:,}。局面の新規の行が学習データに無い行 {a['orphan']:,}）")
    per = ad.groupby("run").size()
    q = per.quantile([0.5, 0.75, 0.9, 0.99]) if len(per) else pd.Series(dtype=float)
    print(f"  1局面の足す行: 分位 50/75/90/99% = {' / '.join(f'{v:.0f}' for v in q)}・最大 {int(per.max()) if len(per) else 0}")
    print(f"  正例率（{K:.1f}σ）: いまの行 {fr['label'].mean()*100:.1f}% / 足す行 {ad['label'].mean()*100:.1f}%"
          f"（局面ごとに合計1の重みで {np.average(ad['label'], weights=run_weights(ad['run']))*100:.1f}%）")
    yr = sorted(set(fr["Date"].dt.year) | set(ad["Date"].dt.year))
    print("  年ごとの行（いま / 足す）: " + " ".join(
        f"{y}:{int((fr['Date'].dt.year == y).sum()):,}/{int((ad['Date'].dt.year == y).sum()):,}" for y in yr))
    n_day = ad.groupby("Date")["label"].transform("size")
    moved = shuffle_within_date(ad["label"], ad["Date"], PERM_SEED)
    out = {"eval_rows": int(len(fr)), "added": int(len(ad)), "runs": int(ad["run"].nunique()),
           "pos_eval": float(fr["label"].mean()), "pos_added": float(ad["label"].mean()),
           "on_multi_row_dates": float((n_day > 1).mean()) if len(ad) else float("nan"),
           "p_changed": float((moved != ad["label"].to_numpy(dtype=float)).mean()) if len(ad) else float("nan")}
    print(f"  P の入れ替え: 同じ日付に足す行が2行以上ある行 {out['on_multi_row_dates']*100:.1f}%・ラベルが変わった行 "
          f"{out['p_changed']*100:.1f}%（ずらし0 の種）")
    return out


def report(res: dict, shifts: list, stamp: dict, data_out: dict, cols: list) -> dict:
    arms = complete_arms(res, shifts)
    missing = [a for a in ARMS if a not in arms]
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 結果のそろった腕: {', '.join(arms) or 'なし'}" + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "table": data_out, "label_k": K, "perm_seed": PERM_SEED, "tuning": {},
               "production_oof": {}, "by_year": [], "train_size": {}, "windows": {}, "judge7": {}, "top": {},
               "importance": {}, "per_window": []}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    rec = res["recs"].get("A")
    if rec:
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  A   {cv['mean_pr_auc']:>8.4f} ±{cv['std']:.4f}  ROC {cv['mean_roc_auc']:.4f}  {rec.get('_seconds', 0)}秒  "
              f"分割ごと " + " / ".join(f"{v:.4f}" for v in fs))
        summary["tuning"]["A"] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        print(f"      パラメータ {short(rec['params'])}")
    if "A" not in arms:
        print("  基準 A の out-of-fold がまだそろっていない（続きは次の回）")
        with open(path("summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
        return summary

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。評価はいまの行・今のラベル）")
    print(f"  {'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
    ms = {}
    for arm in arms:
        o = res["prod"][arm]
        m = metrics(o)
        ms[arm] = m
        summary["production_oof"][arm] = {k: float(v) for k, v in m.items()}
        print(f"  {arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
              f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
              f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
    for x, y in PAIRS:
        if x in ms and y in ms:
            print(f"  {y}−{x:<4}{ms[y]['pr_auc'] - ms[x]['pr_auc']:>+6.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
                  f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")
    yt = E104.year_table(res, arms)
    summary["by_year"] = yt
    print("  年ごとの PR-AUC ÷ 正例率（今のラベル。本番と同じ作りの OOF）")
    print(f"    {'年':<6}{'件数':>7}{'正例率':>8}" + "".join(f"{a:>7}" for a in arms))
    for r in yt:
        print(f"    {r['year']:<6}{r['n']:>7,}{r['pos']*100:>7.1f}%" + "".join(f"{r[a]:>7.2f}" for a in arms))
    print("  学習の大きさ（本番と同じ作りの OOF。最初の窓 → 最後の窓）: いまの行 + 足す行（足す行の重みの合計）")
    for arm in arms:
        sz = res["info"].get(arm, {}).get("sizes") or []
        if not sz:
            continue
        a, b = sz[0], sz[-1]
        summary["train_size"][arm] = {"first": a, "last": b}
        print(f"    {arm:<3} {a['rows']:,} + {a['extra']:,}（{a['extra_weight']:,.0f}） → {b['rows']:,} + {b['extra']:,}"
              f"（{b['extra_weight']:,.0f}）・{len(sz)}窓")

    s = E104.window_table(res, arms, shifts)
    s.to_csv(path("auc_by_window.csv"), index=False)
    print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種3つの平均。評価はいまの行・今のラベル）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for met, nm in (("pr", "PR"), ("roc", "ROC")):
        for x, y in PAIRS:
            if x in arms and y in arms:
                line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                print(line)
                summary["windows"][f"{met}_{y}-{x}"] = st

    print("\n■ 4. §7（2026-09-26 改訂）の1モデル分の3条件（32窓の今のラベルの PR-AUC。回す前に決めた）: "
          "平均の差が正 / 対照の差（P−A と D−A の大きいほう）を上回る / 上の窓が過半")
    if all(a in arms for a in ("A", "G") + CONTROLS):
        d_ga = (s["pr_G"] - s["pr_A"]).to_numpy()
        js = {p: judge7(d_ga, (s[f"pr_{p}"] - s["pr_A"]).to_numpy()) for p in CONTROLS}
        ctrl = max(CONTROLS, key=lambda p: js[p]["mean_pa"])
        j = js[ctrl]
        summary["judge7"] = {"arm": "G", "placebo": ctrl, **j,
                             "controls": {p: js[p]["mean_pa"] for p in CONTROLS}}
        print(f"  G−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / 対照 {ctrl}−A {j['mean_pa']:>+8.4f}"
              f"（P−A {js['P']['mean_pa']:+.4f}・D−A {js['D']['mean_pa']:+.4f}）{'○' if j['beats_placebo'] else '×'} / "
              f"上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）{'○' if j['majority'] else '×'} → "
              f"{'満たす' if j['pass'] else '満たさない'}")
    else:
        print(f"  腕がそろっていない（必要: A・G・{'・'.join(CONTROLS)}）")

    tt = E104.top_table(res, arms, shifts)
    if len(tt):
        tt.to_csv(path("top_by_window.csv"), index=False)
        has_exit = bool(tt[[f"hit10_{a}" for a in arms]].notna().any().any())
        mets = (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)) if has_exit \
            else (("prec", "Precision", 100),)
        print(f"\n■ 5. 窓の中の上位{E104.TOP_SHARE*100:g}%（参考。{len(tt)}窓。件数は窓ごとに切り上げ・合計 {int(tt['k'].sum()):,}件）: "
              "Precision（今のラベル）/ +10% に届いた割合 / 1取引の平均（%。届けば +10、届かなければ20日目の終値）"
              + ("" if has_exit else "。出口の確かめが合わなかったので Precision だけ"))
        for arm in arms:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = (f"  {arm:<4}{r['prec']*100:>7.1f}%" + (f"{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%" if has_exit else ""))
            if arm != "A":
                for m, nm, sc in mets:
                    d = (tt[f"{m}_{arm}"] - tt[f"{m}_A"]).to_numpy(dtype=float) * sc
                    d = d[np.isfinite(d)]
                    if len(d) > 1:
                        line += f" ｜ {nm} {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f}（上 {int((d > 0).sum())}/{len(d)}）"
                        r[f"d_{m}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                       "wins": int((d > 0).sum()), "n": int(len(d))}
            summary["top"][arm] = r
            print(line)

    print("\n■ 6. 列の重要度（gain の割合。本番と同じ作りの OOF（ずらし0・種42）の窓の平均）")
    imps = {a: res["info"].get(a, {}).get("imp") or {} for a in arms}
    if all(imps.values()):
        summary["importance"] = {a: {c: imps[a].get(c, 0.0) for c in BASE_COLS} for a in arms}
        print(f"  土台の4列（{'・'.join(BASE_COLS)}）の合計: " + " / ".join(
            f"{a} {sum(imps[a].get(c, 0.0) for c in BASE_COLS)*100:.1f}%" for a in arms))
        for c in BASE_COLS:
            print(f"    {c:<18}" + " / ".join(f"{a} {imps[a].get(c, 0.0)*100:.2f}%" for a in arms))
        for a in ("A", "G"):
            if a in imps:
                top = sorted(imps[a].items(), key=lambda kv: -kv[1])[:TOP_IMP]
                summary["importance"][f"top_{a}"] = dict(top)
                print(f"  {a} の上位{TOP_IMP}列: " + " ・ ".join(f"{c} {v*100:.1f}%" for c, v in top))
    else:
        print("  重要度がそろっていない")

    print("\n■ 7. 窓ごとの値（今のラベルの PR-AUC。差は「後 − 前」）")
    for sh in shifts:
        g = s[s["shift"] == sh].sort_values("fold")
        if not len(g):
            continue
        print(f"  ずらし{sh}か月（{len(g)}窓）")
        print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}" + "".join(f"{a:>8}" for a in arms)
              + "".join(f"{a + '−A':>9}" for a in arms if a != "A"))
        for _, r in g.iterrows():
            print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                  f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%" + "".join(f"{r[f'pr_{a}']:>8.4f}" for a in arms)
                  + "".join(f"{r[f'pr_{a}'] - r['pr_A']:>+9.4f}" for a in arms if a != "A"))
    summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験106: 同じ上昇局面の2日目以降の更新日を学習だけに足す（lgbm 1.5σ。検証だけ）")
    ap.add_argument("--arms", default=",".join(ARMS), help="この回に計算する腕（A は必ず含める）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="表を作って確かめを出したら終わる")
    args = ap.parse_args(argv)
    TAG = args.tag
    RUN = f"{TAG}_k{int(round(K * 10)):02d}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"知らない腕: {bad}（使えるもの {ARMS}）")
    if "A" not in arms:
        raise SystemExit("基準 A は外せません")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(BASE_PRESET)

    print("=" * 78)
    print(f"実験106 データ拡張の案1: 同じ上昇局面の2日目以降の更新日を学習だけに足す（lgbm {K:.1f}σ・本番と同じ作りの OOF + "
          "32窓・対照 P / D）")
    print(f"  列: {BASE_PRESET} {len(cols)}列 指紋 {F.signature(cols)}")
    for a in ARMS:
        print(f"  {LABELS[a]}{'' if a in arms else '（この回は計算しない）'}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / "
          f"対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    fr, ad, stamp = prepare(cols, args.rebuild)
    for x in (fr, ad):
        x["Date"] = pd.to_datetime(x["Date"])
        x["Code"] = x["Code"].astype(str)
    fr.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    data_out = data_report(fr, ad, stamp)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(fr["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  評価の行 {len(fr):,}（{fr['Date'].min().date()} 〜 {fr['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(fr['Date'] <= cutoff).sum():,}行")
    t0 = time.time()
    res = run(arms, fr, ad, cols, cutoff, args.n_trials, stamp, shifts)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒 / そろった腕 {', '.join(complete_arms(res, shifts)) or 'なし'}")
    report(res, shifts, stamp, data_out, cols)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
