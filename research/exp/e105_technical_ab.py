#!/usr/bin/env python3
"""
実験105: テクニカル指標111列（research/technicals.py。トレンド系41・オシレーター26・ボラティリティ14・出来高系18・ローソク足12）を
今の249列に足すと、LightGBM（1.5σ）の探索（5分割 CV）・本番と同じ作りの out-of-fold・32窓がどう変わるか（実験101〜103 と
同じ手順。腕は「列」の違い）。検証だけ。

運用者の依頼（2026-10-18）「いま株価や出来高から作られている特徴量は27列だけです。以前にもっと、ありとあらゆるテクニカルの
手法、オシレーター、トレンド系全てを使って特徴量を生成してほしい旨伝えた気がするのですが、改めて作成してほしいです。100個
くらいは作れるはずです」→ 調べた答え: 株価・出来高から作った列は 27列（高値への近さ3・ブレイクの性質5・出来高1・売買代金1・
時価総額1・ボラの分解6・抜けた土台の形2・下げの深さ3・直近の増減5）で、RSI・MACD などの一般的なテクニカル指標は無い。以前の
依頼はコード・文書・この作業の会話（2026-10-05〜）には見当たらなかった → 111列の案を示し、運用者の決定（AskUserQuestion）
「lgbm の 1.5σ だけ」。運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→
本番の設定（research/features.py・research/lgbm_params.json）と build_dataset.py には書かない。111列はこの台本の中で日足から
作る（research/technicals.compute。売買の成立した行だけで数え、t の引けまでの値だけを使う。定義と確かめは
tests/test_technicals.py）。運用者の指示「パラメータチューニングする際も全特徴量を使ったモデルでお願いします」→ 探索は
249 + 111 = 360列で行う（腕 B）。

腕（データは同じ1本。列だけが違う）
  A   今の249列（all_plus_prog_listing_vol_book_shape_dd_recent。回した時点の既定）。A の列で探索
  B   249 + 111列で探索し直し → 入れたときに日曜の週次の再学習がやること
  C   249 + 111列を A のパラメータで（探索なし）→ 列を足したことそのものの効果
  P   対照: 111列の値を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を A のパラメータで。§7 の「対照 P の改善を上回る」
  G_* 参考: 系統ごとに 249 + その系統の列を A のパラメータで（G_trend 41・G_oscillator 26・G_volatility 14・G_volume 18・
      G_candle 12）。どの系統が効くかを見る（判定には使わない）
  C−A が「111列を足すことの効果」、P−A が「列数が増えただけの幅」、C−P が「111列そのものの情報」、B−A が「再学習で起きること」

ラベル: 1.5σ（運用者の指示で lgbm の 1.5σ だけ）。行: 学習も評価も全行

判定（回す前に決めた。2026-10-18）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）の1モデル分の3条件を、32窓（ずらし 0/2/4か月 × 種3つの平均）の PR-AUC に
  当てる: 窓ごとの C−A の平均が正・対照 P−A の平均を上回る・C が上の窓が過半（n//2 + 1 以上）。木3モデルの票は数えられないので、
  本番に入れるかは結果を見て運用者が決める（実験101〜104 と同じ扱い）。あわせて C−P・B−A・系統ごとの G−A を出す。

結果（2026-10-18。run 39118153360。docs/MODEL_ADOPTION_RULES.md §68）: 確かめはすべて一致（学習データの行 21,957 / 21,957行が
日足にあり close_position も一致、111列がすべて欠測の行 0）。32窓の PR-AUC で C−A +0.0016（SE 0.0020、上の窓 18/32）・
対照 P−A −0.0016 → §7 の1モデル分の3条件を満たす（差は SE の 0.8倍）。C−P +0.0032（SE 0.0018、22/32）で111列そのものの
情報が見え、探索し直した B−A は +0.0051（SE 0.0021、22/32）。系統ごとの腕はどれも単独ではほぼ変わらない（−0.0015〜+0.0014）。
窓の中の上位2% の +10% 到達は A 40.6% → C 38.6%・B 39.3%（Precision は B 34.7% と +2.6pt）。寄与（gain）は111列で 22%。

1.2σ の回（2026-10-18 追加。運用者の指示「lgbm 1.2σでも、確認したいです」。回す前に決めた）
  同じ表・同じ腕の作りで、ラベルだけ 1.2σ（データセットの label。基準モデルの目的変数）に替える（--k 1.2。保存は e105_k12_*。
  表は 1.5σ の回と同じもの（e105_frame.parquet）を使う）。腕は A・C・P・B（系統ごとの腕は 1.5σ でどれも単独ではほぼ変わらな
  かったので省く）。判定は 1.5σ と同じ §7 の1モデル分の3条件を 1.2σ のラベルの32窓の PR-AUC に当てる。最初に、1.2σ に引き直した
  ラベルがデータセットの label と全行一致することを確かめ、違えば止める（check_label）。
  結果（run 39125832391。§68）: ラベルは 21,957 / 21,957行で一致。32窓の PR-AUC で C−A +0.0046（SE 0.0020、上の窓 17/32）・
  対照 P−A +0.0018 → §7 の1モデル分の3条件を満たす（上の窓は境目）。C−P +0.0028（23/32）。探索し直した B−A は +0.0011（19/32）、
  B−C −0.0035（9/32）。窓の中の上位2% は B が +10% 到達 34.7% → 39.2%・1取引の平均 +0.78% → +1.39% だが、対照 P も平均 +1.32%。

出すもの（公開ログには件数・割合・日付・精度だけ。銘柄・値は出さない）
  0. 111列の付き方（欠測の割合・分位）、既存の249列との順位相関の最大（重なりの多い列の数）、学習の寄与（gain）の系統ごとの割合と
     上位の列
  1. 探索の CV（A・B）  2. 本番と同じ作りの OOF（ずらし0・種42）  3. 32窓の差  4. §7 の判定
  5. 窓の中の上位2% の Precision・+10% の指値に届いた割合・1取引の平均（実験83 の出口。参考）  6. 窓ごとの値

    exp=e105_technical_ab.py                       ← lgbm 1.5σ（run 39118153360）
    exp=e105_technical_ab.py args="--k 1.2 --arms A,C,P,B"   ← lgbm 1.2σ
列の付き方だけ見る: exp=e105_technical_ab.py args="--dry"
試運転:             exp=e105_technical_ab.py args="--tag e105smoke --n-trials 2 --shifts 0 --arms A,C,P"

Run Experiment は同じ実験を同時に1本しか走らせない。表（111列を付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、
以降の回は actions/cache で引き継いだそれを使う。本番導入の第1段階（2026-10-18、§68）からはデータセットにも111列があるので、
表を作り直すときは、この台本の計算と全行一致することを確かめてから使う（compare_columns）。--budget-min（既定 290分）を過ぎたら
新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。判定の腕（A・C・P）を先に、B と系統ごとの腕を
後に計算する。
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
import technicals as TX  # noqa: E402
import train_model as T  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e44_shortsale import permuted  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 週次の再学習（retrain-weekly.yml）と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
#: ラベルの到達しきい値（σ の倍数）。既定は 1.5σ（運用者の指示）。--k 1.2 で基準モデルの 1.2σ（2026-10-18 追加）
K_DEFAULT = 1.5
K = K_DEFAULT
#: 比べる土台。回した時点の既定（249列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent"
NEW_COLS = tuple(TX.COLUMNS)
#: 系統ごとの参考の腕
GROUP_ARMS = {f"G_{g}": g for g in TX.GROUPS}
ARMS = ("A", "C", "P", "B") + tuple(GROUP_ARMS)
#: 判定に使う腕（先に計算する）
PRIMARY = ("A", "C", "P")
COLS_OF = {"A": "base", "B": "new", "C": "new", "P": "perm", **{a: a for a in GROUP_ARMS}}
PARAMS_OF = {"A": "A", "B": "B", "C": "A", "P": "A", **{a: "A" for a in GROUP_ARMS}}
LABELS = {"A": "A  今の249列（探索した A のパラメータ）", "B": "B  249 + 111列・探索し直し",
          "C": "C  249 + 111列・A のパラメータ", "P": "P  対照（111列を日付内で入れ替え）・A のパラメータ",
          **{a: f"{a} 参考: 249 + {TX.GROUP_JA[g]} {len(TX.GROUPS[g])}列・A のパラメータ" for a, g in GROUP_ARMS.items()}}
PAIRS = (("A", "C"), ("A", "P"), ("P", "C"), ("A", "B"), ("C", "B")) + tuple(("A", a) for a in GROUP_ARMS)
#: 対照の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261020
MIN_WINDOW_ROWS = 100
#: 重なりの多い列の目安（既存の列との順位相関の絶対値）
OVERLAP = (0.9, 0.95)
TP = LT.TAKE_PROFIT
TAG = "e105"
RUN = "e105_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（ラベルによらない）。"""
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


def check_label(full: pd.DataFrame, k: float) -> dict | None:
    """
    k が基準モデルの倍数（DEFAULT_RISE.vol_norm_k = 1.2）のとき、引き直したラベルがデータセットの label と全行一致するか
    （決まる・決まらないも含めて）。違えば止める。ほかの k では何もしない（None）。
    """
    if abs(float(k) - float(B.DEFAULT_RISE.vol_norm_k)) > 1e-9:
        return None
    a = B.rise_label_for_k(full, k)
    b = pd.to_numeric(full["label"], errors="coerce")
    both = a.notna() & b.notna()
    out = {"rows": int(len(full)), "both": int(both.sum()), "same": int((a[both] == b[both]).sum()),
           "defined_mismatch": int((a.notna() != b.notna()).sum())}
    print(f"  確かめ: {k:.1f}σ に引き直したラベルとデータセットの label の一致 {out['same']:,} / {out['both']:,}行"
          f"（決まる・決まらないの食い違い {out['defined_mismatch']:,}行）")
    if out["same"] != out["both"] or out["defined_mismatch"]:
        raise SystemExit(f"{k:.1f}σ に引き直したラベルがデータセットの label と合わない")
    return out


def with_label(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """k のラベルに引き直した表（データセットの材料から。週次の学習と同じ rise_label_for_k）。決まらない行は落とす。"""
    out = df.copy()
    out["label"] = B.rise_label_for_k(out, k)
    return out[out["label"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
PANEL_COLS = ["Code", "Date", "open", "high", "low", "close", "vol", "trading_value"]


def build_panel(data_dir: str, codes) -> pd.DataFrame:
    """build_dataset.build と同じ作りの日足（price_panel。分割を調整した値）から、指標に要る列だけ。codes の銘柄だけ残す。"""
    bars = B.load_parts("bars", data_dir)
    segments = B.load_market_segments(data_dir)
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    p = B.price_panel(bars, start=start)
    del bars
    p["Code"] = p["Code"].astype(str)
    p = p.loc[p["Code"].isin(set(codes)), PANEL_COLS].reset_index(drop=True)
    p["Date"] = p["Date"].astype("datetime64[ns]")
    return p


def attach_technicals(keys: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """keys（Code, Date）の行ごとに111列（keys の並びのまま）。日足に無い行は欠測。銘柄を分けて計算する（technicals.compute_at）。"""
    return TX.compute_at(panel, keys)


def compare_columns(have: pd.DataFrame, want: pd.DataFrame, cols=NEW_COLS, rtol: float = 1e-6, atol: float = 1e-9) -> dict:
    """
    列ごとに、欠測の位置か値が食い違う行の数。have・want は同じ行の並び。値の許しは学習と予測の突き合わせ
    （check_train_serve.compare）と同じ相対 1e-6・絶対 1e-9。technicals.compute_at は銘柄の組ごとに数えるので、組が違うと
    移動和の丸めが変わり、同じ行でも相対 1e-10 ほど違う（作りもので最大 3e-10。欠測の位置は変わらない）。
    """
    out = {}
    for c in cols:
        a = pd.to_numeric(have[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(want[c], errors="coerce").to_numpy(dtype=float)
        na, nb = np.isnan(a), np.isnan(b)
        both = ~na & ~nb
        close = np.zeros(len(a), dtype=bool)
        close[both] = np.isclose(a[both], b[both], rtol=rtol, atol=atol)
        out[c] = int(((na != nb) | (both & ~close)).sum())
    return out


def check_alignment(frame: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """日足から数え直した当日の値幅の中の位置（close_position）が、学習データの値と全行一致するか（並び・調整の確かめ）。"""
    p = panel[["Code", "Date", "high", "low", "close"]].copy()
    rng = p["high"] - p["low"]
    p["cp"] = np.where(rng > 0, (p["close"] - p["low"]) / rng * 100.0, np.nan)
    m = frame[["Code", "Date", "close_position"]].merge(p[["Code", "Date", "cp", "close"]], on=["Code", "Date"], how="left")
    if len(m) != len(frame):
        raise SystemExit("日足の Code・Date が重なっている")
    a = pd.to_numeric(m["close_position"], errors="coerce").to_numpy(dtype=float)
    b = m["cp"].to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    out = {"rows": int(len(m)), "in_panel": int(m["close"].notna().sum()),
           "same": int((both & (np.abs(a - b) < 1e-9)).sum()), "both": int(both.sum()),
           "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum())}
    print(f"  確かめ: 学習データの行のうち日足に終値がある {out['in_panel']:,} / {out['rows']:,}行・"
          f"close_position の一致 {out['same']:,} / {out['both']:,}行（欠測の食い違い {out['nan_mismatch']:,}行）")
    if out["in_panel"] != out["rows"] or out["same"] != out["both"] or out["nan_mismatch"]:
        raise SystemExit("日足と学習データの行が合わない（並びか調整の作りが違う）")
    return out


def prepare(cols: list, rebuild: bool = False):
    """lab.frame に111列と出口（+10% の指値）を付けた表（全行。ラベルはデータセットの 1.2σ のまま）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
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
    miss = [c for c in cols + ["close_position", "ret_o1_20", "ret_o1_40"] if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    have = [c for c in NEW_COLS if c in frame.columns]
    if have and len(have) != len(NEW_COLS):
        raise SystemExit(f"学習データに111列の一部だけがある: {have[:5]}")
    log(f"[data] lab.frame {len(frame):,}行 × {frame.shape[1]}列 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    panel = build_panel(lab.DATA_DIR, frame["Code"].unique())
    log(f"[data] 日足 {len(panel):,}行（表の {frame['Code'].nunique():,}銘柄）/ {time.time()-t0:.0f}秒")
    print("\n■ 確かめ（日足の並び・調整が学習データと同じか）")
    checks = {"alignment": check_alignment(frame, panel)}
    t0 = time.time()
    tx = attach_technicals(frame[["Code", "Date"]], panel)
    del panel
    all_nan = int(tx.isna().all(axis=1).sum())
    checks["rows_all_nan"] = all_nan
    log(f"[data] 111列 / {time.time()-t0:.0f}秒（111列がすべて欠測の行 {all_nan:,}）")
    if have:
        # 本番導入の第1段階（2026-10-18、§68）からはデータセットにも111列がある（build_dataset.attach_technicals）。
        # この台本の計算と全行一致することを確かめてから、台本の値を使う
        bad = compare_columns(frame[list(NEW_COLS)].reset_index(drop=True), tx)
        n_bad = sum(bad.values())
        checks["dataset_columns"] = {"rows": int(len(frame)), "bad_cells": int(n_bad),
                                     "bad_columns": int(sum(v > 0 for v in bad.values()))}
        print(f"  確かめ: データセットの111列とこの台本の計算の一致 {len(frame):,}行 × 111列 → 食い違い {n_bad:,}セル"
              f"（{checks['dataset_columns']['bad_columns']}列）")
        if n_bad:
            raise SystemExit("データセットの111列がこの台本の計算と合わない: "
                             + ", ".join(f"{c} {v}" for c, v in bad.items() if v)[:300])
        frame = frame.drop(columns=list(NEW_COLS))
    df = pd.concat([frame.reset_index(drop=True), tx], axis=1)

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


def coverage_report(df: pd.DataFrame, base: list) -> dict:
    """111列の欠測の割合と分位、既存の249列との順位相関の最大（件数と割合・相関だけ）。"""
    out = {"columns": {}, "overlap": {}}
    print(f"\n■ 0. 111列の付き方（{len(df):,}行。分位は 5% / 50% / 95%）")
    for g, cols in TX.GROUPS.items():
        print(f"  ◆ {TX.GROUP_JA[g]}（{len(cols)}列）")
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            q = v.quantile([0.05, 0.5, 0.95]) if v.notna().any() else pd.Series([np.nan] * 3, index=[0.05, 0.5, 0.95])
            out["columns"][c] = {"missing": float(v.isna().mean()), "q05": float(q[0.05]), "q50": float(q[0.5]),
                                 "q95": float(q[0.95])}
            print(f"    {c:<24}{TX.JA[c]:<40} 欠測 {v.isna().mean()*100:5.1f}%  {q[0.05]:>10.3g} / {q[0.5]:>10.3g} / "
                  f"{q[0.95]:>10.3g}")
    ranks = df[base + list(NEW_COLS)].rank()
    corr = ranks.corr()
    best = {}
    for c in NEW_COLS:
        s = corr.loc[c, base].abs()
        s = s[s.notna()]
        best[c] = (s.idxmax(), float(s.max())) if len(s) else (None, float("nan"))
    inner = corr.loc[list(NEW_COLS), list(NEW_COLS)].abs().to_numpy()
    iu = np.triu_indices(len(NEW_COLS), 1)
    pairs = inner[iu]
    for th in OVERLAP:
        n_base = sum(1 for c in NEW_COLS if best[c][1] >= th)
        n_pair = int(np.sum(pairs >= th))
        out["overlap"][str(th)] = {"vs_base": n_base, "pairs_within": n_pair}
        print(f"  既存の249列のどれかと順位相関の絶対値が {th} 以上の列 {n_base} / 111・111列どうしで {th} 以上の組 {n_pair}")
    top = sorted(NEW_COLS, key=lambda c: -best[c][1] if np.isfinite(best[c][1]) else 0)[:10]
    print("  既存の列といちばん重なる10列: " + " / ".join(f"{c}↔{best[c][0]} {best[c][1]:.2f}" for c in top))
    out["max_abs_rho_vs_base"] = {c: {"col": best[c][0], "rho": best[c][1]} for c in NEW_COLS}
    return out


def gain_report(df: pd.DataFrame, base: list, params: dict, cutoff) -> dict:
    """探索の期間の全行で C（249 + 111列・A のパラメータ）を1本学習し、寄与（gain）の系統ごとの割合と上位の列（参考）。"""
    import lightgbm as lgb

    cols = base + list(NEW_COLS)
    sub = df[df["Date"] <= cutoff]
    y = sub["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": PROD_SEED}, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(sub[cols].to_numpy(dtype=float), y)
    gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols)
    tot = float(gain.sum()) or 1.0
    share = {"既存の249列": float(gain[base].sum() / tot)}
    for g, gc in TX.GROUPS.items():
        share[TX.GROUP_JA[g]] = float(gain[gc].sum() / tot)
    print(f"\n■ 0b. 学習の寄与（gain）の割合（探索の期間 〜{cutoff.date()} の {len(sub):,}行で C を1本。参考）: "
          + " / ".join(f"{k} {v*100:.1f}%" for k, v in share.items()))
    new_rank = gain[list(NEW_COLS)].sort_values(ascending=False)
    all_rank = gain.rank(ascending=False)
    print("  111列のうち寄与の大きい15列（全360列の中の順位）: " + " / ".join(
        f"{c} {int(all_rank[c])}位" for c in new_rank.index[:15]))
    return {"share": share, "top_new": {c: {"gain_share": float(gain[c] / tot), "rank": int(all_rank[c])}
                                        for c in new_rank.index[:30]}}


# --------------------------------------------------------------------------- #
# 探索（週次の再学習と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict):
    p = path(f"params_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not time_left():
        return None
    sub = frame[(frame["Date"] <= cutoff) & frame["label"].notna()]
    t0 = time.time()
    log(f"  [{arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列）")
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
    log(f"  [{arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）/ ROC {cv['mean_roc_auc']:.4f}")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold（本番と同じ作り。E41.oof_folds）
# --------------------------------------------------------------------------- #
def oof(arm: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds):
    """腕 × パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{ALGO}_c{COLS_OF[arm]}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
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
        o = E41.oof_folds(ALGO, frame, cols, params, sd, folds)
        o = o.merge(frame[["Code", "Date", "hit10", "tp10c"]], on=["Code", "Date"], how="left")
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの AUC と、窓の期間・件数・正例率（MIN_WINDOW_ROWS 未満の窓は落とす）。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    n = o.groupby("fold").size()
    sub = o[o["fold"].isin(n[n >= MIN_WINDOW_ROWS].index)]
    if not len(sub):
        return pd.DataFrame(columns=cols)
    w = AB.auc_by_window(sub)
    if not len(w):
        return pd.DataFrame(columns=cols)
    g = sub.groupby("fold")
    info = pd.DataFrame({"start": g["Date"].min(), "end": g["Date"].max(), "n": g.size(),
                         "pos": g["label"].mean()}).reset_index()
    return w.merge(info, on="fold", how="left")[cols]


def run(arms: list, frames: dict, colsets: dict, cutoff, n_trials: int, stamp: dict, shifts: list) -> dict:
    """判定の腕（A・C・P）を先に、B と系統ごとの腕を後に計算する。時間切れの腕は結果に入らない。"""
    res = {"recs": {}, "params": {}, "prod": {}, "oofs": {}}
    res["recs"]["A"] = tune("A", frames["base"], colsets["base"], cutoff, n_trials, stamp)
    if res["recs"]["A"] is None:
        return res
    first = [a for a in PRIMARY if a in arms]
    later = [a for a in ARMS if a in arms and a not in PRIMARY]
    for group in (first, later):
        if "B" in group:
            res["recs"]["B"] = tune("B", frames["new"], colsets["new"], cutoff, n_trials, stamp)
        for arm in group:
            rec = res["recs"].get(PARAMS_OF[arm])
            if rec is None:
                continue
            res["params"][arm] = rec["params"]
            cs = COLS_OF[arm]
            o = oof(arm, frames[cs], colsets[cs], rec["params"], 0, (PROD_SEED,))
            if o is not None:
                res["prod"][arm] = o
        for sh in shifts:
            for arm in group:
                if arm not in res["params"]:
                    continue
                cs = COLS_OF[arm]
                o = oof(arm, frames[cs], colsets[cs], res["params"][arm], sh, SEEDS)
                if o is not None:
                    res["oofs"][(sh, arm)] = o
    return res


def complete_arms(res: dict, shifts: list) -> list:
    return [a for a in ARMS if a in res["prod"] and all((sh, a) in res["oofs"] for sh in shifts)]


def window_table(res: dict, arms: list, shifts: list) -> pd.DataFrame:
    rows = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = windows_on(res["oofs"][(sh, arm)]).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
            if w is not None:
                x = x[["fold", f"pr_{arm}", f"roc_{arm}"]]
            w = x if w is None else w.merge(x, on="fold")
        if w is None or not len(w):
            continue
        for _, r in w.iterrows():
            rows.append({"shift": sh, "fold": int(r["fold"]), "start": r["start"], "end": r["end"], "n": int(r["n"]),
                         "pos": float(r["pos"]), **{f"{m}_{a}": float(r[f"{m}_{a}"]) for m in ("pr", "roc") for a in arms}})
    return pd.DataFrame(rows)


def top_table(res: dict, arms: list, shifts: list) -> pd.DataFrame:
    rows = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = top_by_window(res["oofs"][(sh, arm)]).rename(columns={c: f"{c}_{arm}" for c in ("prec", "hit10", "tp10c")})
            x = x.drop(columns=["n", "k"]) if w is not None else x
            w = x if w is None else w.merge(x, on="fold")
        if w is not None and len(w):
            rows += [{"shift": sh, **r} for r in w.to_dict("records")]
    return pd.DataFrame(rows)


def report(res: dict, shifts: list, stamp: dict, extra: dict) -> dict:
    arms = complete_arms(res, shifts)
    missing = [a for a in ARMS if a not in arms]
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 結果のそろった腕: {', '.join(arms) or 'なし'}" + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, **extra, "label_k": K, "perm_seed": PERM_SEED, "tuning": {}, "production_oof": {},
               "windows": {}, "judge7": {}, "top": {}, "per_window": []}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用）")
    for arm in ("A", "B"):
        rec = res["recs"].get(arm)
        if not rec:
            continue
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f} ±{cv['std']:.4f}  ROC {cv['mean_roc_auc']:.4f}  {rec.get('_seconds', 0)}秒  "
              f"分割ごと " + " / ".join(f"{v:.4f}" for v in fs))
        print(f"      パラメータ {short(rec['params'])}")
        summary["tuning"][arm] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
    if "A" not in arms:
        print("  基準 A の out-of-fold がまだそろっていない（続きは次の回）")
        with open(path("summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
        return summary

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42）")
    print(f"  {'腕':<14}{'PR-AUC':>8}{'リフト':>7}{'ROC':>8}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}{'件数':>8}")
    ms = {}
    for arm in arms:
        o = res["prod"][arm]
        m = metrics(o)
        ms[arm] = m
        summary["production_oof"][arm] = {k: float(v) for k, v in m.items()}
        print(f"  {arm:<14}{m['pr_auc']:>8.4f}{m['lift']:>6.2f}x{m['roc_auc']:>8.4f}{m['day_auc']:>8.4f}"
              f"{m['ret_o1_20_mean']:>+10.2f}pt{int(m['ret_o1_20_won']):>4}/{int(m['ret_o1_20_n']):<2}"
              f"{m['ret_o1_20_worst']:>+8.2f}pt{len(o):>8,}")
    for x, y in PAIRS:
        if x in ms and y in ms:
            print(f"  {y + '−' + x:<14}{ms[y]['pr_auc'] - ms[x]['pr_auc']:>+8.4f}{'':>7}{ms[y]['roc_auc'] - ms[x]['roc_auc']:>+8.4f}"
                  f"{ms[y]['day_auc'] - ms[x]['day_auc']:>+8.4f}{ms[y]['ret_o1_20_mean'] - ms[x]['ret_o1_20_mean']:>+10.2f}pt")

    s = window_table(res, arms, shifts)
    s.to_csv(path("auc_by_window.csv"), index=False)
    print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種3つの平均）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for met, nm in (("pr", "PR"), ("roc", "ROC")):
        for x, y in PAIRS:
            if x in arms and y in arms:
                line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                print(line)
                summary["windows"][f"{met}_{y}-{x}"] = st

    print("\n■ 4. §7（2026-09-26 改訂）の1モデル分の3条件（32窓の PR-AUC。回す前に決めた）: "
          "平均 C−A が正 / 対照 P−A を上回る / C が上の窓が過半")
    if all(a in arms for a in ("A", "C", "P")):
        j = judge7((s["pr_C"] - s["pr_A"]).to_numpy(), (s["pr_P"] - s["pr_A"]).to_numpy())
        summary["judge7"] = j
        print(f"  lgbm {K:.1f}σ: C−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / P−A {j['mean_pa']:>+8.4f} "
              f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
              f"{'○' if j['majority'] else '×'} → {'満たす' if j['pass'] else '満たさない'}")
    else:
        print("  腕がそろっていない（A・C・P）")

    tt = top_table(res, arms, shifts)
    if len(tt):
        tt.to_csv(path("top_by_window.csv"), index=False)
        has_exit = bool(tt[[f"hit10_{a}" for a in arms]].notna().any().any())
        mets = (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)) if has_exit \
            else (("prec", "Precision", 100),)
        print(f"\n■ 5. 窓の中の上位2%（参考。{len(tt)}窓・合計 {int(tt['k'].sum()):,}件）: Precision / +10% に届いた割合 / "
              "1取引の平均（%。届けば +10、届かなければ20日目の終値）")
        for arm in arms:
            r = {m: float(tt[f"{m}_{arm}"].mean()) for m in ("prec", "hit10", "tp10c")}
            line = f"  {arm:<14}{r['prec']*100:>7.1f}%" + (f"{r['hit10']*100:>8.1f}%{r['tp10c']:>+8.2f}%" if has_exit else "")
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

    show = [a for a in ("A", "B", "C", "P") if a in arms] + [a for a in GROUP_ARMS if a in arms]
    print("\n■ 6. 窓ごとの値（PR-AUC。差は「後 − 前」）")
    for sh in shifts:
        g = s[s["shift"] == sh].sort_values("fold")
        if not len(g):
            continue
        print(f"  ずらし{sh}か月（{len(g)}窓）")
        print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}" + "".join(f"{a[:8]:>9}" for a in show)
              + f"{'C−A':>9}{'P−A':>9}")
        for _, r in g.iterrows():
            print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                  f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%" + "".join(f"{r[f'pr_{a}']:>9.4f}" for a in show)
                  + (f"{r['pr_C'] - r['pr_A']:>+9.4f}" if "C" in arms else "")
                  + (f"{r['pr_P'] - r['pr_A']:>+9.4f}" if "P" in arms else ""))
    summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE, K
    ap = argparse.ArgumentParser(description="実験105: テクニカル指標111列を249列に足す前後（lgbm 1.5σ・1.2σ。検証だけ）")
    ap.add_argument("--k", type=float, default=K_DEFAULT, choices=(1.2, 1.5), help="ラベルの到達しきい値（σ の倍数）。既定 1.5")
    ap.add_argument("--arms", default=",".join(ARMS), help="この回に計算する腕（A は必ず含める）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="111列の付き方だけ出して終わる")
    args = ap.parse_args(argv)
    TAG = args.tag
    K = float(args.k)
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
    base = F.columns(BASE_PRESET)
    new = list(NEW_COLS)
    dup = sorted(set(base) & set(new))
    if dup:
        raise SystemExit(f"今の列と重なる: {dup}")
    colsets = {"base": base, "new": base + new, "perm": base + new,
               **{a: base + list(TX.GROUPS[g]) for a, g in GROUP_ARMS.items()}}

    print("=" * 78)
    print(f"実験105 テクニカル指標111列を今の249列に足す前後（lgbm {K:.1f}σ・5分割 CV の探索 + OOF + 32窓・対照 P）")
    print(f"  列: {BASE_PRESET} {len(base)}列 指紋 {F.signature(base)} / + 111列 → {len(base) + len(new)}列 "
          f"指紋 {F.signature(base + new)}")
    print("  系統: " + " / ".join(f"{TX.GROUP_JA[g]} {len(c)}" for g, c in TX.GROUPS.items()))
    for a in ARMS:
        print(f"  {LABELS[a]}{'' if a in arms else '（この回は計算しない）'}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / "
          f"対照の種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    full, stamp = prepare(base, args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    label_check = check_label(full, K)
    df = with_label(full, K)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    extra = {"coverage": coverage_report(df, base), "label_check": label_check}
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（ラベル {K:.1f}σ の正例率 {df['label'].mean()*100:.1f}%・{df['Date'].min().date()} 〜 "
          f"{df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では {(df['Date'] <= cutoff).sum():,}行")
    if args.dry:
        return 0
    t0 = time.time()
    perm = permuted(df, new, seed=PERM_SEED)
    perm.attrs["built_utc"] = stamp["built_utc"]
    log(f"[data] 対照 P: 111列を日付内で入れ替えた（種 {PERM_SEED}）/ {time.time()-t0:.0f}秒")
    frames = {"base": df, "new": df, "perm": perm, **{a: df for a in GROUP_ARMS}}
    t0 = time.time()
    res = run(arms, frames, colsets, cutoff, args.n_trials, stamp, shifts)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒 / そろった腕 {', '.join(complete_arms(res, shifts)) or 'なし'}")
    if res["recs"].get("A"):
        extra["gain"] = gain_report(df, base, res["recs"]["A"]["params"], cutoff)
    report(res, shifts, stamp, extra)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
