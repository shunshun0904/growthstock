#!/usr/bin/env python3
"""
実験112: Chronos-2（TSFM）の予測を3列にして本番の360列に足すと、木3モデル（lgbm・xgb・cat）の本番と同じ作りの
out-of-fold・32窓の分離力（PR-AUC）が上がるか（§7。腕は「列」の違い）。検証だけ。

運用者の依頼（2026-10-23）「こちらの論文（TSFM）を使った金融時系列リターン予測の論文ですが、私が運用している本運用モデルに
導入できないか考えたいです」（Noguer i Alonso・Pereira Franklin「Pretrained Time-Series Foundation Models for Financial Return
Forecasting」arXiv 2606.27100）→ 調べた答え（docs/MODEL_ADOPTION_RULES.md §75）: 置き換え・混ぜるは勧めない（論文自身が売買の
優位の証拠ではないと書き、比べた相手は1銘柄ごとに一から学ぶ深層学習で、木のモデルは入っていない。実験109 でも Chronos-2 は
基準モデルを下回った）。まだ試していない入れ方で筋が通るのは「TSFM の予測を列として足す」だけ（実験109 の株価だけの腕 Z0 は
基準モデルを下回ったが、当てる力はあった: lift 1.33・95%区間の下端 1.28）→ 運用者の選択（AskUserQuestion）「列として足す
検証」・記録は「docs に残す」。運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」
→ 本番の設定（features.py・lgbm_params.json・multi_params.json）と build_dataset.py には書かない。

足す3列（回す前に決めた。2026-10-23）
  Chronos-2（amazon/chronos-2。実験109 と同じ版 REVISION に固定）を学習せずに（ゼロショット）、実験109 の腕 Z0 と同じ入力で読む:
  各行 t の銘柄の日足（price_panel。分割を調整した終値）の t までの直近 256行の終値の対数、先は20行（ラベルの数え方と同じ）。
    c2_p_end   20日目の終値が必要上昇率（ラベルと同じ式・1.2σ×√20。どのラベルの回でも 1.2σ の値）を超える確率
               （実験109 の判定の点数 p_end と同じ作り。E109.scores）
    c2_med     20日目の中央値 − log C[t]（20営業日の対数リターンの中央値）
    c2_width   20日目の 90% 点 − 10% 点（20営業日の対数リターンの 80% 区間の幅）
  Chronos-2 は学習しないので、学習の行も評価の行も同じ作りで値が付く（目的変数は見ない）。t の終値までしか使わない
  （日次の予測は引けの後に回るので、本番でも同じ値が作れる）。

腕（データは同じ1本。列だけが違う。パラメータは3つの腕で同じ）
  A  本番の360列（features.DEFAULT_PRESET）
  C  360列 + 3列
  P  対照: 360列 + 3列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED。実験51・56・67・68・105 と同じ作り）
  パラメータ: 本番のモデルがいま使っているもの（research/model/meta.json・research/model/models/<名前>/meta.json の params。
  239列で探索したもの。本番の学習は 2026-10-17・18。360列での探索は週末 10/24・10/25 の再学習から）。LightGBM は本番の組み方
  （tuning_multi.build）と同じ固定の値（木200本など）で補う（full_lgbm）

ラベル（--k）
  1.2  基準のラベル（データセットの label）。§7 の判定はこの回で行う
  1.5  参考: 今の買いの規則の3本（L15・X15・C15）のラベルとパラメータ。同じ集計を参考に出し、あわせて規則の行（1.5σ の3本の
       前向きの百分位がすべて95以上・テスト窓 3〜11）の件数・+10% の指値の成績を A と C で並べる

判定（回す前に決めた。2026-10-23）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）: 木3モデル（lgbm・xgb・cat）のうち2つ以上で、窓ごとの PR-AUC の差 C−A の
  平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）。窓は本番の out-of-fold と同じ作り（36/6/6か月・
  エンバーゴ20営業日）の境界を 0/2/4か月ずらした3通り（あわせて約32窓）、点数は種3つ（42・7・123）の平均。1.2σ のラベルで
  判定し、本番に入れるかは結果を見て運用者が決める（実験101〜105 と同じ扱い）。

参考（判定には使わない）
  0. 3列の付き方（欠測・分位）、3列だけでの PR-AUC・lift・ROC-AUC（全行・実験109 のテストの期間・モデルの公開
     （2025年10月）より前と後。後の行で当てる力が落ちていないか = 学習データとの重なりで良く見えていないかの目安）、本番の
     360列との順位相関の絶対値の最大（重なりの多い列）
  0b. 学習の寄与（gain。探索の期間の行で lgbm の C を1本）の3列の割合と順位
  1. 本番と同じ作りの out-of-fold（ずらし0・種42）
  2. 窓ごとの差（C−A・P−A・C−P。PR-AUC と ROC-AUC）
  4. 窓の中の上位2% の Precision・+10% の指値に届いた割合・1取引の平均（実験83 の出口）
  5. （1.5σ の回）規則の行の件数・+10% に届いた割合・1取引の平均（A と C）

確かめ（合わなければ止める）
  - 日足と学習データの行の並び・調整が同じこと（E105.check_alignment。close_position が全行一致）
  - 1.2σ に引き直したラベルがデータセットの label と全行一致すること（E105.check_label）
  - Chronos-2 の版が REVISION であること（Hugging Face から取った版を突き合わせる）
  - 部品: tests/test_e112_chronos2_cols.py

回し方（Run Experiment。同じ実験は同時に1本だけ）
    exp=e112_chronos2_cols.py args="--algos lgbm --shifts 0 --seeds 42"   ← 1回目: 通しの確かめ（3列を全行に付けて表を保存）
    exp=e112_chronos2_cols.py                                            ← 1.2σ（判定）
    exp=e112_chronos2_cols.py args="--k 1.5"                             ← 1.5σ（参考）
  表（3列と出口を付けた frame）は1回目に作って research/_data/oof/e112_* に置き、以降の回は actions/cache で引き継いだそれを
  使う（データセットは作り直すたびに値がわずかに違い、作り直すと out-of-fold の名前が変わるため。実験105 と同じ）。
  --budget-min（既定 290分）を過ぎたら新しい計算を始めずに報告して終わる（保存済みの out-of-fold は次の回が続きから使う）。

公開ログには件数・割合・日付・精度・列名だけを出す（銘柄・値は出さない）。
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
import tuning_multi as TM  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e105_technical_ab as E105  # noqa: E402
import e109_chronos2 as E109  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
TAG = "e112"
RUN = "e112_k12"
ALGOS = ("lgbm", "xgb", "cat")
SEEDS = (42, 7, 123)
PROD_SEED = 42
SHIFTS = (0, 2, 4)
#: ラベルの到達しきい値（σ の倍数）。1.2 が基準のラベル（判定）、1.5 は今の買いの規則の3本のラベル（参考）
K_DEFAULT = 1.2
K = K_DEFAULT
NEW_COLS = ("c2_p_end", "c2_med", "c2_width")
ARMS = ("A", "C", "P")
COLS_OF = {"A": "base", "C": "new", "P": "perm"}
LABELS = {"A": "A  本番の360列", "C": "C  360列 + Chronos-2 の3列", "P": "P  対照（3列を日付内で入れ替え）"}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"))
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261023
#: 実験109 が読んだ版（run 39563953576 のログ）。本番に入れるなら同じく版を固定する
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
MODEL_ID = E109.MODEL_ID
CONTEXT = E109.CONTEXT
HORIZON = E109.HORIZON
CHUNK_ROWS = E109.CHUNK_ROWS
BATCH = E109.BATCH_ITEMS_Z0
POST_RELEASE = E109.POST_RELEASE
#: 実験109 のテストの期間（窓1 の初日）。3列だけの PR-AUC を実験109 の Z0（0.2380）と見比べる
E109_TEST_START = pd.Timestamp("2021-05-02")
TP = LT.TAKE_PROFIT
#: 規則の行（1.5σ の回の参考）: 3本の前向きの百分位がすべてこの値以上。百分位はそれより前の窓の点数の分布で付ける
RULE_LINE = 95.0
RULE_FIRST_WINDOW = 3
#: 本番の LightGBM の組み方（tuning_multi.build）の固定の値。探索の記録（params）には入っていないものがある
LGBM_FIXED = {"objective": "binary", "boosting_type": "gbdt", "n_estimators": TM.N_ESTIMATORS, "n_jobs": -1,
              "verbose": -1, "subsample_freq": 1}
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cpath(name: str) -> str:
    """表と Chronos-2 の分位点（ラベルによらない）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def path(name: str) -> str:
    """out-of-fold・要約（ラベルごと）。"""
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
# パラメータ（本番のモデルがいま使っているもの）
# --------------------------------------------------------------------------- #

def model_name(algo: str, k: float) -> str:
    """ラベル k の本番のモデルの名前（1.2σ は学習器の名前のまま、1.5σ は <学習器>_k15）。"""
    if abs(k - 1.2) < 1e-9:
        return algo
    if abs(k - 1.5) < 1e-9:
        return f"{algo}_k15"
    raise ValueError(f"このラベルの本番のモデルは無い: {k}")


def meta_path(name: str, root: str = MODEL_DIR) -> str:
    """本番のモデルの記録。基準モデル（1.2σ の LightGBM）は model/meta.json、ほかは model/models/<名前>/meta.json。"""
    return os.path.join(root, "meta.json") if name == "lgbm" else os.path.join(root, "models", name, "meta.json")


def full_lgbm(p: dict) -> dict:
    """LightGBM のパラメータを、本番の組み方（tuning_multi.build）の固定の値で補う（探索の値が先に勝つ）。"""
    return {**LGBM_FIXED, **dict(p)}


def prod_params(algo: str, k: float, root: str = MODEL_DIR) -> tuple[dict, dict]:
    """本番のモデルが学習に使ったパラメータと、その出どころ（名前・列の組・学習の日・列の数・指紋）。"""
    name = model_name(algo, k)
    with open(meta_path(name, root), encoding="utf-8") as fh:
        meta = json.load(fh)
    p = dict(meta["params"])
    if algo == "lgbm":
        p = full_lgbm(p)
    info = {"model": name, "preset": meta.get("preset"), "trained": str(meta.get("trainedAt", ""))[:10],
            "n_features": len(meta.get("features") or []), "hash": params_hash(p)}
    return p, info


# --------------------------------------------------------------------------- #
# Chronos-2 の3列
# --------------------------------------------------------------------------- #

def level_index(levels, x: float) -> int:
    """分位点の水準のうち x のもの（無ければ止める）。"""
    lv = np.asarray(levels, dtype=float)
    i = int(np.argmin(np.abs(lv - x)))
    if abs(lv[i] - x) > 1e-9:
        raise SystemExit(f"分位点の水準に {x} が無い: {list(lv)}")
    return i


def c2_columns(q: np.ndarray, levels, last_log: np.ndarray, need: np.ndarray) -> pd.DataFrame:
    """
    q: (n, Q, H) の終値の対数の分位点、last_log: log C[t]、need: 必要上昇率（比。0.1 = +10%）-> 3列（行の並びのまま）。
    20日目の分位点は小さい順に並べ直してから引く（交差を直す。E109.scores と同じ）。
    """
    sc = E109.scores(q, levels, last_log, need)
    qs = np.sort(np.asarray(q, dtype=float)[:, :, -1], axis=1)
    i10, i50, i90 = (level_index(levels, x) for x in (0.1, 0.5, 0.9))
    return pd.DataFrame({"c2_p_end": sc["p_end"].to_numpy(dtype=float),
                         "c2_med": qs[:, i50] - last_log,
                         "c2_width": qs[:, i90] - qs[:, i10]})


def load_pipeline():
    """amazon/chronos-2 を版 REVISION で読む。取った版が違えば止める。"""
    import torch
    from huggingface_hub import snapshot_download
    from chronos import Chronos2Pipeline

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    local = snapshot_download(MODEL_ID, revision=REVISION)
    got = os.path.basename(os.path.normpath(local))
    if got != REVISION:
        raise SystemExit(f"Chronos-2 の版が違う: {got}（{REVISION} のはず）")
    pipe = Chronos2Pipeline.from_pretrained(local)
    pipe.model.eval()
    return pipe, got


def z0_quantiles(rows: pd.DataFrame, panel: pd.DataFrame) -> tuple[np.ndarray, list, dict]:
    """
    rows（Code・Date。日足の行）の Chronos-2 の分位点（n, Q, HORIZON）と水準。入力は実験109 の腕 Z0 と同じ（E109.make_items）。
    CHUNK_ROWS 行ごとに保存し、保存済みは読む（束の名前は行・入力の値・版で決まる。E109.chunk_key）。
    """
    start, end = E109.context_bounds(panel, rows, CONTEXT)
    close = panel["close"].to_numpy(dtype=float)
    meta = {"arm": "Z0", "ctx": CONTEXT, "h": HORIZON, "model": MODEL_ID, "rev": REVISION}
    pipe, levels, parts = None, None, []
    info = {"chunks": 0, "computed": 0, "seconds": 0.0}
    for st in range(0, len(rows), CHUNK_ROWS):
        sl = np.arange(st, min(st + CHUNK_ROWS, len(rows)))
        items = E109.make_items(close, start[sl], end[sl])
        p = cpath(f"z0_{E109.chunk_key(meta, rows.iloc[sl][['Code', 'Date']], items)}.npz")
        info["chunks"] += 1
        if os.path.exists(p):
            z = np.load(p)
            q, lv = z["q"], [float(x) for x in z["levels"]]
        else:
            if pipe is None:
                E109.ensure_chronos()
                t0 = time.time()
                pipe, rev = load_pipeline()
                log(f"[chronos] {MODEL_ID}（commit {rev}）を読んだ / {time.time()-t0:.0f}秒・分位点 {len(pipe.quantiles)}個")
            t0 = time.time()
            q = E109.forecast(pipe, items, BATCH)
            lv = [float(x) for x in pipe.quantiles]
            np.savez_compressed(p, q=q, levels=np.asarray(lv, dtype=float))
            dt = time.time() - t0
            info["computed"] += len(sl)
            info["seconds"] += dt
            log(f"  Chronos-2 {sl[-1] + 1:,} / {len(rows):,}行 {dt:.0f}秒（1行 {dt / len(sl):.3f}秒）")
        if levels is None:
            levels = lv
        elif lv != levels:
            raise SystemExit("束によって分位点の水準が違う")
        parts.append(q)
    return np.concatenate(parts, axis=0), levels, info


# --------------------------------------------------------------------------- #
# 表（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #

def prepare(cols: list, rebuild: bool = False, dry: bool = False):
    """lab.frame に3列と出口（+10% の指値）を付けた表（ラベルの決まる全行。ラベルはデータセットの 1.2σ のまま）と記録。"""
    pf, ps = cpath("frame.parquet"), cpath("stamp.json")
    if not rebuild and not dry and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        miss = [c for c in cols + list(NEW_COLS) if c not in df.columns]
        if not miss:
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
            return df, stamp
        log(f"[data] 保存済みの表に無い列がある（{miss[:5]}）ので作り直す")
    os.makedirs(OOF_DIR, exist_ok=True)
    t0 = time.time()
    frame = lab.frame(rebuild=True)
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    frame = frame[frame["label"].notna()].reset_index(drop=True)
    need_cols = cols + ["vol_20d", "close_position", "ret_o1_20", "ret_o1_40", "future_rise", "end_level", "uptrend_end"]
    miss = [c for c in need_cols if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    dup = [c for c in NEW_COLS if c in frame.columns]
    if dup:
        raise SystemExit(f"足す列の名前が表に既にある: {dup}")
    log(f"[data] lab.frame {len(frame):,}行（{frame['Date'].min().date()}〜{frame['Date'].max().date()}）/ {time.time()-t0:.0f}秒")

    t0 = time.time()
    panel = E105.build_panel(lab.DATA_DIR, frame["Code"].unique())
    # 文脈は日足の連続した行で切る（E109.context_bounds・make_items）ので、銘柄・日付の順を自分で固定して確かめる
    panel = panel.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    if panel.duplicated(["Code", "Date"]).any():
        raise SystemExit("日足の Code・Date が重なっている")
    log(f"[data] 日足 {len(panel):,}行（{panel['Code'].nunique():,}銘柄）/ {time.time()-t0:.0f}秒")
    print("\n■ 確かめ（日足の並び・調整が学習データと同じか）")
    checks = {"alignment": E105.check_alignment(frame, panel)}
    start, end = E109.context_bounds(panel, frame, CONTEXT)
    span = end - start + 1
    checks["context"] = {"median": int(np.median(span)), "short_share": float((span < CONTEXT).mean())}
    print(f"  文脈の長さ（行）: 中央値 {int(np.median(span))}・{CONTEXT} 行に満たない行 {(span < CONTEXT).mean()*100:.1f}%")
    if dry:
        print("\n[dry] Chronos-2 の手前で止める")
        return frame, None

    t0 = time.time()
    q, levels, info = z0_quantiles(frame, panel)
    close = panel["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        last_log = np.log(close[end])
    del panel
    need, _ = B.rise_thresholds(frame["vol_20d"], B.DEFAULT_RISE)
    c2 = c2_columns(q, levels, last_log, need.to_numpy(dtype=float))
    crossed = float((np.diff(q[:, :, -1], axis=1) < 0).any(axis=1).mean())
    checks["chronos"] = {"model": MODEL_ID, "revision": REVISION, "levels": levels, "rows": int(len(frame)),
                         "computed_rows": int(info["computed"]), "seconds": round(info["seconds"]),
                         "crossed_share": crossed, "nan_rows": int(c2.isna().any(axis=1).sum())}
    log(f"[chronos] 3列 {len(frame):,}行（今回計算 {info['computed']:,}行・{info['seconds']:.0f}秒）/ {time.time()-t0:.0f}秒"
        f"（欠測のある行 {checks['chronos']['nan_rows']:,}・20日目の分位点が交差した行 {crossed*100:.1f}%）")
    df = pd.concat([frame.reset_index(drop=True), c2], axis=1)

    t0 = time.time()
    bars_px = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(df[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(df) or not (ex["Code"].to_numpy() == df["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    k = E83.tp_key(TP)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    df["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float) if exit_ok else np.nan
    df["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float) if exit_ok else np.nan
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行 → "
          f"{'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}（{time.time()-t0:.0f}秒）")

    yr = df["Date"].dt.year
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())}, "checks": checks,
             "missing": {c: int(df[c].isna().sum()) for c in NEW_COLS}}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


# --------------------------------------------------------------------------- #
# 3列の付き方（参考）
# --------------------------------------------------------------------------- #

def standalone(y: np.ndarray, s: np.ndarray) -> dict:
    """1列だけを点数にしたときの PR-AUC・lift・ROC-AUC（欠測の行は除く）。"""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < 50 or y.min() == y.max():
        return {"n": int(len(y)), "pr": float("nan"), "lift": float("nan"), "roc": float("nan")}
    pr, roc = E109.pr_roc(y, s)
    return {"n": int(len(y)), "pr": pr, "lift": pr / float(y.mean()), "roc": roc}


def column_report(df: pd.DataFrame, base: list) -> dict:
    """3列の欠測・分位、3列だけの当てる力（期間ごと）、本番の列との順位相関の絶対値の最大（件数・割合・精度だけ）。"""
    out = {"columns": {}, "standalone": {}, "overlap": {}}
    print(f"\n■ 0. Chronos-2 の3列の付き方（{len(df):,}行。分位は 5% / 50% / 95%）")
    for c in NEW_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        qv = v.quantile([0.05, 0.5, 0.95])
        out["columns"][c] = {"missing": float(v.isna().mean()), "q05": float(qv[0.05]), "q50": float(qv[0.5]),
                             "q95": float(qv[0.95])}
        print(f"  {c:<10} 欠測 {v.isna().mean()*100:5.1f}%  {qv[0.05]:>9.4f} / {qv[0.5]:>9.4f} / {qv[0.95]:>9.4f}")
    y_all = df["label"].to_numpy(dtype=int)
    d = pd.to_datetime(df["Date"])
    periods = {"全行": np.ones(len(df), dtype=bool), f"実験109 のテストの期間（{E109_TEST_START.date()} 以降）": d >= E109_TEST_START,
               f"公開より前（〜{(POST_RELEASE - pd.Timedelta(days=1)).date()}）": d < POST_RELEASE,
               f"公開より後（{POST_RELEASE.date()} 以降）": d >= POST_RELEASE}
    print(f"  3列だけを点数にしたとき（{K:.1f}σ のラベル。PR-AUC / lift / ROC-AUC。c2_width は狭いほど上の向きでも出す）")
    for name, m in periods.items():
        m = np.asarray(m)
        line, rec = [], {}
        for c in NEW_COLS:
            s = pd.to_numeric(df.loc[m, c], errors="coerce").to_numpy(dtype=float)
            r = standalone(y_all[m], s)
            rec[c] = r
            line.append(f"{c} {r['pr']:.4f} / {r['lift']:.2f} / {r['roc']:.4f}")
        r = standalone(y_all[m], -pd.to_numeric(df.loc[m, "c2_width"], errors="coerce").to_numpy(dtype=float))
        rec["-c2_width"] = r
        line.append(f"−c2_width {r['pr']:.4f} / {r['lift']:.2f} / {r['roc']:.4f}")
        out["standalone"][name] = {"rows": int(m.sum()), "pos": float(y_all[m].mean()), **rec}
        print(f"  {name}（{int(m.sum()):,}行・正例率 {y_all[m].mean()*100:.1f}%）: " + "  ".join(line))
    ranks = df[list(base) + list(NEW_COLS)].rank()
    for c in NEW_COLS:
        s = ranks[list(base)].corrwith(ranks[c]).abs().dropna().sort_values(ascending=False)
        top = [(k, float(v)) for k, v in s.head(5).items()]
        out["overlap"][c] = {"top": top, "n_ge_0.9": int((s >= 0.9).sum()), "n_ge_0.7": int((s >= 0.7).sum())}
        print(f"  {c} と本番の列の順位相関の絶対値: 0.9 以上 {int((s >= 0.9).sum())}列・0.7 以上 {int((s >= 0.7).sum())}列・"
              "上位 " + " / ".join(f"{k} {v:.2f}" for k, v in top))
    inner = ranks[list(NEW_COLS)].corr()
    out["inner"] = {f"{a}~{b}": float(inner.loc[a, b]) for i, a in enumerate(NEW_COLS) for b in NEW_COLS[i + 1:]}
    print("  3列どうしの順位相関: " + " / ".join(f"{k} {v:+.2f}" for k, v in out["inner"].items()))
    return out


def gain_report(df: pd.DataFrame, base: list, params: dict, cutoff) -> dict:
    """探索の期間の行で lgbm の C（360 + 3列）を1本学習し、3列の寄与（gain）の割合と順位（参考）。"""
    import lightgbm as lgb

    cols = list(base) + list(NEW_COLS)
    sub = df[df["Date"] <= cutoff]
    y = sub["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": PROD_SEED}, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(sub[cols].to_numpy(dtype=float), y)
    gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols, dtype=float)
    tot = float(gain.sum()) or 1.0
    rank = gain.rank(ascending=False, method="min")
    out = {c: {"share": float(gain[c] / tot), "rank": int(rank[c])} for c in NEW_COLS}
    out["_total_share"] = float(gain[list(NEW_COLS)].sum() / tot)
    print(f"\n■ 0b. 学習の寄与（gain。探索の期間 〜{cutoff.date()} の {len(sub):,}行で lgbm の C を1本。参考）: 3列で "
          f"{out['_total_share']*100:.1f}%（" + " / ".join(f"{c} {out[c]['share']*100:.2f}%・{out[c]['rank']}位"
                                                  for c in NEW_COLS) + f"。{len(cols)}列の中の順位）")
    return out


# --------------------------------------------------------------------------- #
# out-of-fold（本番と同じ作り。E41.oof_folds）
# --------------------------------------------------------------------------- #

def oof(algo: str, colset: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds):
    """学習器 × 列の組（base / new / perm）× ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{algo}_c{colset}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing and not time_left():
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
        o = o.merge(frame[["Code", "Date", "hit10", "tp10c"]], on=["Code", "Date"], how="left")
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {algo} 列{colset} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def run(algos, frames: dict, colsets: dict, params: dict, shifts, seeds) -> dict:
    """学習器ごとに、本番と同じ作り（ずらし0・種42）と、ずらし × 種の平均を、腕 A・C・P で。時間切れの分は入らない。"""
    res = {}
    for algo in algos:
        r = {"prod": {}, "oofs": {}}
        for arm in ARMS:
            cs = COLS_OF[arm]
            o = oof(algo, cs, frames[cs], colsets[cs], params[algo], 0, (PROD_SEED,))
            if o is not None:
                r["prod"][arm] = o
        for sh in shifts:
            for arm in ARMS:
                cs = COLS_OF[arm]
                o = oof(algo, cs, frames[cs], colsets[cs], params[algo], sh, seeds)
                if o is not None:
                    r["oofs"][(sh, arm)] = o
        res[algo] = r
    return res


def complete(r: dict, shifts) -> bool:
    return all(a in r["prod"] for a in ARMS) and all((sh, a) in r["oofs"] for sh in shifts for a in ARMS)


# --------------------------------------------------------------------------- #
# 規則の行（1.5σ の回の参考）
# --------------------------------------------------------------------------- #

def forward_pct(o: pd.DataFrame, first: int = RULE_FIRST_WINDOW) -> np.ndarray:
    """窓ごとに、それより前の窓の点数の分布での百分位（画面と同じ LT.pct_of）。first より前の窓は欠測。"""
    s = o["score"].to_numpy(dtype=float)
    f = o["fold"].to_numpy()
    out = np.full(len(o), np.nan)
    for w in np.unique(f):
        if w < first:
            continue
        hist = s[f < w]
        if not len(hist):
            continue
        m = f == w
        out[m] = LT.pct_of(s[m], hist)
    return out


def rule_rows(oofs: dict, line: float = RULE_LINE, first: int = RULE_FIRST_WINDOW) -> pd.DataFrame:
    """学習器 -> out-of-fold（ずらし0・種の平均）の前向きの百分位が、すべて line 以上の行（3本に共通の行の中で）。"""
    base = None
    for a, o in oofs.items():
        keep = ["Code", "Date", "fold", "label", "hit10", "tp10c"] if base is None else ["Code", "Date"]
        x = o[keep].copy()
        x[f"pct_{a}"] = forward_pct(o, first)
        base = x if base is None else base.merge(x, on=["Code", "Date"], how="inner")
    pc = base[[f"pct_{a}" for a in oofs]].to_numpy(dtype=float)
    ok = np.isfinite(pc).all(axis=1)
    ok[ok] = pc[ok].min(axis=1) >= line
    return base[ok].reset_index(drop=True)


def rule_stats(g: pd.DataFrame) -> dict:
    t = pd.to_numeric(g["tp10c"], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    return {"n": int(len(g)), "days": int(g["Date"].nunique()), "pos": float(g["label"].mean()) if len(g) else float("nan"),
            "hit10": float(pd.to_numeric(g["hit10"], errors="coerce").mean()) if len(g) else float("nan"),
            "tp10c": float(t.mean()) if len(t) else float("nan"),
            "se": float(t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else float("nan")}


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def report(res: dict, shifts, seeds, stamp: dict, extra: dict, pinfo: dict) -> dict:
    algos = [a for a in ALGOS if a in res and complete(res[a], shifts)]
    missing = [a for a in ALGOS if a not in algos]
    full_design = tuple(shifts) == SHIFTS and tuple(seeds) == SEEDS
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 結果のそろった学習器: {', '.join(algos) or 'なし'}"
          + (f" / まだ: {', '.join(missing)}" if missing else "")
          + ("" if full_design else f"（この回は ずらし {list(shifts)}・種 {list(seeds)} だけ。判定の形ではない）"))
    summary = {"data": stamp, **extra, "label_k": K, "perm_seed": PERM_SEED, "params": pinfo, "shifts": list(shifts),
               "seeds": list(seeds), "full_design": full_design, "production_oof": {}, "windows": {}, "judge7": {},
               "top": {}, "per_window": {}}

    print("\n■ 1. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）")
    print(f"  {'':<6}{'腕':<4}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
    for a in algos:
        ms = {}
        for arm in ARMS:
            o = res[a]["prod"][arm]
            m = metrics(o)
            ms[arm] = m
            summary["production_oof"][f"{a}_{arm}"] = {k: float(v) for k, v in m.items()}
            print(f"  {a:<6}{arm:<4}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
                  f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
        for x, y in PAIRS:
            print(f"  {a:<6}{y}−{x} {ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
                  f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    tables = {}
    print(f"\n■ 2. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種{len(seeds)}つの平均）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for a in algos:
        s = E105.window_table(res[a], list(ARMS), shifts)
        tables[a] = s
        s.to_csv(path(f"{a}_auc_by_window.csv"), index=False)
        for met, nm in (("pr", "PR"), ("roc", "ROC")):
            for x, y in PAIRS:
                line, st = pair(f"{a} {nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                print(line)
                summary["windows"][f"{a}_{met}_{y}-{x}"] = st

    print(f"\n■ 3. §7（2026-09-26 改訂）の判定（窓の PR-AUC。回す前に決めた。{K:.1f}σ）: "
          "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
    for a in algos:
        s = tables[a]
        j = judge7((s["pr_C"] - s["pr_A"]).to_numpy(), (s["pr_P"] - s["pr_A"]).to_numpy())
        summary["judge7"][a] = j
        print(f"  {a:<6}C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / P−A {j['mean_pa']:>+8.4f} "
              f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
              f"{'○' if j['majority'] else '×'} → {'満たす' if j['pass'] else '満たさない'}")
    passed = sum(int(summary["judge7"][a]["pass"]) for a in algos)
    done = len(algos) == len(ALGOS) and full_design
    if done:
        verdict = "満たす（木3モデル中2つ以上）" if passed >= 2 else "満たさない（木3モデル中2つ未満）"
    else:
        verdict = f"{passed}/{len(algos)} が満たす（木3モデル・ずらし3通り・種3つがそろってから判定）"
    summary["judge7"]["verdict"] = {"passed": passed, "of": len(algos), "complete": done, "text": verdict}
    print(f"  → §7: {verdict}" + ("" if abs(K - 1.2) < 1e-9 else "（1.5σ は参考。判定は 1.2σ の回）"))

    print("\n■ 4. 窓の中の上位2%（参考）: Precision / +10% に届いた割合 / 1取引の平均（%。届けば +10、届かなければ20日目の終値）")
    for a in algos:
        tt = E105.top_table(res[a], list(ARMS), shifts)
        if not len(tt):
            continue
        tt.to_csv(path(f"{a}_top_by_window.csv"), index=False)
        summary["top"][a] = {}
        for arm in ARMS:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = f"  {a:<6}{arm:<4}{r['prec']*100:>7.1f}%{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%"
            if arm != "A":
                for m, nm, sc in (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)):
                    d = (tt[f"{m}_{arm}"] - tt[f"{m}_A"]).to_numpy(dtype=float) * sc
                    d = d[np.isfinite(d)]
                    if len(d) > 1:
                        se = d.std(ddof=1) / np.sqrt(len(d))
                        line += f" ｜ {nm} {d.mean():+.2f} ± {se:.2f}（上 {int((d > 0).sum())}/{len(d)}）"
                        r[f"d_{m}"] = {"mean": float(d.mean()), "se": float(se), "wins": int((d > 0).sum()), "n": int(len(d))}
            summary["top"][a][arm] = r
            print(line + f"（{len(tt)}窓・{int(tt['k'].sum()):,}件）")

    if abs(K - 1.5) < 1e-9 and len(algos) == len(ALGOS) and 0 in shifts:
        print(f"\n■ 5. 規則の行（参考。1.5σ の3本の前向きの百分位がすべて{RULE_LINE:g}以上・テスト窓 {RULE_FIRST_WINDOW}〜・ずらし0・種"
              f"{len(seeds)}つの平均。届けば +10%、届かなければ20日目の終値）")
        picks, summary["rule"] = {}, {}
        for arm in ("A", "C"):
            picks[arm] = rule_rows({a: res[a]["oofs"][(0, arm)] for a in ALGOS})
            st = rule_stats(picks[arm])
            summary["rule"][arm] = st
            print(f"  {arm}  {st['n']:>5,}件（{st['days']:,}日）  正例率 {st['pos']*100:5.1f}%  +10% {st['hit10']*100:5.1f}%  "
                  f"1取引 {st['tp10c']:+.2f}%（± {st['se']:.2f}）")
        ka = set(zip(picks["A"]["Code"], picks["A"]["Date"]))
        kc = set(zip(picks["C"]["Code"], picks["C"]["Date"]))
        summary["rule"]["overlap"] = {"both": len(ka & kc), "only_A": len(ka - kc), "only_C": len(kc - ka)}
        print(f"  重なり: 両方 {len(ka & kc):,}件・A だけ {len(ka - kc):,}件・C だけ {len(kc - ka):,}件")

    print("\n■ 6. 窓ごとの PR-AUC の差（C−A / P−A。ずらしごと）")
    for a in algos:
        s = tables[a]
        summary["per_window"][a] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
        for sh in shifts:
            g = s[s["shift"] == sh].sort_values("fold")
            if not len(g):
                continue
            print(f"  {a:<6}ずらし{sh}か月（{len(g)}窓）C−A " + " ".join(f"{v:+.3f}" for v in g["pr_C"] - g["pr_A"])
                  + " ｜ P−A " + " ".join(f"{v:+.3f}" for v in g["pr_P"] - g["pr_A"]))
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE, K
    ap = argparse.ArgumentParser(description="実験112: Chronos-2 の予測の3列を本番の360列に足す前後（木3モデル・§7。検証だけ）")
    ap.add_argument("--k", type=float, default=K_DEFAULT, choices=(1.2, 1.5), help="ラベルの到達しきい値（σ の倍数）。既定 1.2")
    ap.add_argument("--algos", default=",".join(ALGOS), help="この回に計算する学習器")
    ap.add_argument("--shifts", default=",".join(map(str, SHIFTS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の out-of-fold は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="Chronos-2 の手前まで（行・文脈の長さ・パラメータ）")
    args = ap.parse_args(argv)
    TAG = args.tag
    K = float(args.k)
    RUN = f"{TAG}_k{int(round(K * 10)):02d}"
    if args.budget_min > 0:
        DEADLINE = time.time() + args.budget_min * 60
    algos = [a for a in args.algos.split(",") if a]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise SystemExit(f"知らない学習器: {bad}（使えるもの {ALGOS}）")
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    base = F.columns(F.DEFAULT_PRESET)
    new = list(NEW_COLS)
    params, pinfo = {}, {}
    for a in ALGOS:
        params[a], pinfo[a] = prod_params(a, K)

    print("=" * 78)
    print(f"実験112 Chronos-2 の予測の3列を本番の360列に足す前後（{K:.1f}σ・木3モデル・OOF + 窓・対照 P）")
    print(f"  列: {F.DEFAULT_PRESET} {len(base)}列 指紋 {F.signature(base)} / + {', '.join(new)} → {len(base) + len(new)}列")
    print(f"  Chronos-2: {MODEL_ID}（commit {REVISION}）・文脈 {CONTEXT}行・先 {HORIZON}行・ゼロショット")
    for a in ARMS:
        print(f"  {LABELS[a]}")
    for a in ALGOS:
        i = pinfo[a]
        print(f"  パラメータ {a}: 本番の {i['model']}（{i['preset']}・{i['n_features']}列・学習 {i['trained']}・指紋 {i['hash']}）"
              f"{'' if a in algos else '（この回は計算しない）'}  {short(params[a])}")
    print(f"  窓のずらし {shifts}か月 / 種 {seeds} / 対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(base, args.rebuild, args.dry)
    if args.dry:
        return 0
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    label_check = E105.check_label(full, K)
    df = E105.with_label(full, K)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（ラベル {K:.1f}σ の正例率 {df['label'].mean()*100:.1f}%・{df['Date'].min().date()} 〜 "
          f"{df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では {(df['Date'] <= cutoff).sum():,}行")
    extra = {"columns": column_report(df, base), "label_check": label_check}
    extra["gain"] = gain_report(df, base, params["lgbm"], cutoff)
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: 3列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm}
    colsets = {"base": base, "new": base + new, "perm": base + new}
    t0 = time.time()
    res = run(algos, frames, colsets, params, shifts, seeds)
    log(f"[oof] 計算 {time.time()-t0:.0f}秒 / そろった学習器 {', '.join(a for a in algos if complete(res[a], shifts)) or 'なし'}")
    # この回に計算しない学習器も、保存済みがそろっていれば集計に入れる（読むだけ。足りなければ入らない）
    for a in ALGOS:
        if a not in res:
            r = run([a], frames, colsets, params, shifts, seeds) if _saved_only(a, frames, colsets, params, shifts, seeds) else {}
            res.update(r)
    report(res, shifts, seeds, stamp, extra, pinfo)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


def _saved_only(algo: str, frames: dict, colsets: dict, params: dict, shifts, seeds) -> bool:
    """algo の out-of-fold がすべて保存済みか（この回に計算しない学習器を、読むだけで集計に入れられるか）。"""
    for arm in ARMS:
        cs = COLS_OF[arm]
        fr = frames[cs]
        ph = params_hash(params[algo]) + "_d" + hashlib.sha1(str(fr.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
        want = [(0, PROD_SEED)] + [(sh, sd) for sh in shifts for sd in seeds]
        if not all(os.path.exists(path(f"{algo}_c{cs}_{ph}_sh{sh}_s{sd}.parquet")) for sh, sd in want):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
