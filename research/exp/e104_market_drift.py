#!/usr/bin/env python3
"""
実験104: 「長期的な市場の伸び」の影響を除いて学習すると、LightGBM（1.5σ）の out-of-fold の分離力は上がるか。
2つのやり方を、本番と同じ作りの out-of-fold・32窓で、今のラベルの PR-AUC（§7 と同じ物差し）で比べる（検証だけ）。

運用者の仮説（2026-10-18）「oofでの精度が下がるのは、株式市場の成長性、資本主義社会を前提とした物価の上昇などのマクロ経済
要因というか、長期的な市場の成長を考慮できていないからだと思います。その影響を除いてモデリングする方法を知りたいです。
株式市場の成長性といいますか。年々ブレイクするサンプル数も増えているはずなので。おそらく正例数も？かもしれません。」
→ 年ごとの実データ（基準モデル（1.2σ の LightGBM）の OOF。1日あたりの件数・正例率・PR-AUC）を見せた: 件数も正例率も年々
増える形ではなく市場の局面で上下し、分離力が低いのは弱い年（2022・2026）。そのうえで5つのやり方を示した →
運用者の決定（AskUserQuestion）: 「目的変数を TOPIX 超過に」と「古い年の重みを下げる」の2つを、「lgbm の 1.5σ だけ」で。
重みの半減期は「2年を主、1年・4年を参考」。判定は「今のラベルの PR-AUC（§7 と同じ）」。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」→ 本番の設定
（research/features.py・research/lgbm_params.json・build_dataset.py）には書かない。

部A: 目的変数を TOPIX 超過に
  今のラベル（1.5σ）: t+1 の寄りで買って、t+1〜t+20 の終値の最大が k·σ20·√20 以上上がり、t+20 の5日平均が必要な上昇の
  半分以上残り、t+20 で MA5 >= MA20 か（build_dataset.attach_rise_label）。
  超過ラベル: 同じ関数を「株価 ÷ TOPIX」の値動きに当てる（終値は TOPIX の終値で、寄りは TOPIX の始値で割る）。到達は
  「買値からの株価の伸び ÷ 同じ期間の TOPIX の伸び」で測ることになる。必要な上昇率の σ20 は銘柄自身の値のまま（しきい値は
  今のラベルと同じで、その間の市場の動きだけを除く）。TOPIX は indices のコード 0000 の始値・終値（topix_*.parquet の値と
  共通の日すべてで一致することを確かめる）。
  対照 PX: TOPIX の値動きを 250営業日（約1年）前にずらしてから割る。揺れの大きさと長い期間の伸びは TOPIX と同じだが、
  その期間の市場の動きとは関係が無い。X−A が PX−A を上回れば「その期間の市場の動きを除いたこと」の効果。
  学習のラベルだけを変え、評価はどの腕も今のラベル（1.5σ）で測る（ラベルが同じなので PR-AUC を比べられる）。

部B: 古い年の重みを下げる
  学習の行ごとの重み w = 0.5^(経過年数 / 半減期)。経過年数は窓ごとの学習の最終日（train_end）からの暦日 ÷ 365.25。
  窓ごとに平均1にそろえる（探索したパラメータの葉の重みの下限などが同じ大きさで効くように）。正例の重み
  scale_pos_weight は重みを掛けた負例 ÷ 正例（重みなしの腕の「負例数 ÷ 正例数」と同じ考え方）。
  対照 PW: W2 の重みを窓ごとの学習の行の中で入れ替えたもの（種 PERM_SEED + 100 × ずらし + 窓の番号）。重みの分布
  （実効の行数の減り方）は同じで、新しさとは関係が無い。

腕（行は同じ。今のラベル・TOPIX 超過・対照の3つのラベルがすべて決まる行だけ）
  A   今のラベル（1.5σ）・重みなし。249列（回した時点の既定）で探索（Optuna 50試行 × 5分割。週次の再学習と同じ関数・条件）
  X   TOPIX 超過のラベルで学習・A のパラメータ                                  ← 部A の判定
  PX  対照: 1年ずらした TOPIX で割ったラベルで学習・A のパラメータ              ← 部A の対照
  W2  今のラベル・半減期2年の重み・A のパラメータ                              ← 部B の判定
  PW  対照: W2 の重みを学習の行の中で入れ替え・A のパラメータ                  ← 部B の対照
  XB  TOPIX 超過のラベルで探索し直し（参考。採用したら週次の再学習がやること）
  W1  半減期1年（参考）
  W4  半減期4年（参考）

判定（回す前に決めた。2026-10-18）
  docs/MODEL_ADOPTION_RULES.md §7（2026-09-26 改訂）の1モデル分の3条件を、32窓（ずらし 0/2/4か月 × 種3つの平均）の今の
  ラベルの PR-AUC に当てる。
    部A: 窓ごとの X−A の平均が正・対照 PX−A の平均を上回る・X が上の窓が過半（n//2 + 1 以上）
    部B: 窓ごとの W2−A の平均が正・対照 PW−A の平均を上回る・W2 が上の窓が過半
  木3モデルの票は数えられない（lgbm だけ）ので、本番に入れるかは結果を見て運用者が決める（実験101〜103 と同じ扱い）。
  XB・W1・W4 は参考（判定には使わない）。

出すもの（公開ログには件数・割合・日付・精度だけ。銘柄・値は出さない）
  0. ラベルの年ごとの正例率（今・超過・対照）、その期間の TOPIX の騰落の平均、ラベルが入れ替わった行の割合、月ごとの
     正例率と TOPIX の騰落の相関、TOPIX の騰落の区切りごとの正例率
  1. 探索の CV（A・XB）
  2. 本番と同じ作りの out-of-fold（ずらし0・種42）: 腕ごとの PR-AUC・リフト・ROC・日内 AUC など、年ごとの PR-AUC ÷ 正例率
  3. 32窓の差（PR-AUC・ROC）の平均・SE・上の窓。参考に超過ラベルで測った PR-AUC
  4. §7 の判定（部A・部B）
  5. 窓の中の上位2% の Precision（今のラベル）・+10% の指値に届いた割合・1取引の平均（実験83 の出口。参考）
  6. 窓ごとの値の一覧

    exp=e104_market_drift.py                       ← 回すのはこの1本（lgbm 1.5σ）
ラベルの付き方だけ見る: exp=e104_market_drift.py args="--dry"
試運転:                 exp=e104_market_drift.py args="--tag e104smoke --n-trials 2 --shifts 0 --arms A,X,PX,W2,PW"

Run Experiment は同じ実験を同時に1本しか走らせない。表（3つのラベルと出口を付けた frame）は1回目に作って
research/_data/oof/<tag>_* に置き、以降の回は actions/cache で引き継いだそれを使う。--budget-min（既定 290分）を
過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が続きから使う）。判定の腕（A・X・PX・W2・PW）を
先に計算し、参考の腕（XB・W1・W4）は後に回す。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace

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
import ab_oof as AB  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e25_auc_noise import average, metrics  # noqa: E402
from e68_edinet_ab import judge7, pair, params_hash, short  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
N_SPLITS = 5
CV_SCHEME = tuning.PRODUCTION_CV       # 週次の再学習（retrain-weekly.yml）と同じ（year_cap_date）
CV_OBJECTIVE = tuning.PRODUCTION_OBJECTIVE
ALGO = "lgbm"
SEEDS = (42, 7, 123)
PROD_SEED = 42
#: ラベルの到達しきい値（σ の倍数）。運用者の指示で 1.5σ だけ
K = 1.5
#: 列の組。回した時点の既定（249列）を名前で固定する
BASE_PRESET = "all_plus_prog_listing_vol_book_shape_dd_recent"
#: indices の中の TOPIX のコード
TOPIX_CODE = "0000"
#: 対照 PX で TOPIX をずらす営業日数（約1年。回す前に決めた）
PLACEBO_LAG = 250
#: 重みの半減期（年）。W2 が判定、W1・W4 は参考
HALF_LIVES = {"W1": 1.0, "W2": 2.0, "W4": 4.0}
#: 対照 PW の入れ替えの種（結果を見る前に決めた）
PERM_SEED = 20261018
ARMS = ("A", "X", "PX", "W2", "PW", "XB", "W1", "W4")
#: 判定に使う腕（先に計算する）と参考の腕
PRIMARY = ("A", "X", "PX", "W2", "PW")
REFERENCE = ("XB", "W1", "W4")
#: 学習に使うラベルの列（評価はどの腕も今のラベル = label）
TRAIN_LABEL = {"A": "y_cur", "X": "y_x", "PX": "y_px", "XB": "y_x",
               "W2": "y_cur", "PW": "y_cur", "W1": "y_cur", "W4": "y_cur"}
#: どの探索のパラメータを使うか
PARAMS_OF = {"A": "A", "X": "A", "PX": "A", "XB": "XB", "W2": "A", "PW": "A", "W1": "A", "W4": "A"}
#: 学習の行の重み（None は重みなし。("decay", 半減期) / ("shuffle", 半減期) = その重みを入れ替えたもの）
WEIGHT_OF = {"A": None, "X": None, "PX": None, "XB": None,
             "W2": ("decay", HALF_LIVES["W2"]), "PW": ("shuffle", HALF_LIVES["W2"]),
             "W1": ("decay", HALF_LIVES["W1"]), "W4": ("decay", HALF_LIVES["W4"])}
LABELS = {"A": "A  今のラベル（1.5σ）・重みなし・探索したパラメータ（基準）",
          "X": "X  TOPIX 超過のラベルで学習・A のパラメータ",
          "PX": "PX 対照: 1年ずらした TOPIX で割ったラベルで学習・A のパラメータ",
          "W2": "W2 古い年の重みを下げる（半減期2年）・A のパラメータ",
          "PW": "PW 対照: W2 の重みを学習の行の中で入れ替え・A のパラメータ",
          "XB": "XB TOPIX 超過のラベルで探索し直し（参考）",
          "W1": "W1 半減期1年（参考）",
          "W4": "W4 半減期4年（参考）"}
#: §7 の判定（回す前に決めた）: 部 → (判定の腕, 対照の腕, 名前)
JUDGES = {"A": ("X", "PX", "部A 目的変数を TOPIX 超過に"),
          "B": ("W2", "PW", "部B 古い年の重みを下げる（半減期2年）")}
#: 表に出す差（「後 − 前」）
PAIRS = (("A", "X"), ("A", "PX"), ("PX", "X"), ("A", "XB"),
         ("A", "W2"), ("A", "PW"), ("PW", "W2"), ("A", "W1"), ("A", "W4"))
MIN_WINDOW_ROWS = 100
#: 窓の中の上位の割合（運用者が見てきた上位2%。参考）
TOP_SHARE = 0.02
#: +10% の指値（実験83・運用の今の出口）
TP = LT.TAKE_PROFIT
#: TOPIX の騰落の区切り（%。t+1 の寄り → t+20 の引け。回す前に決めた）
TFWD_EDGES = (-np.inf, -5.0, -2.0, 0.0, 2.0, 5.0, np.inf)
TAG = "e104"
RUN = "e104_k15"
DEADLINE: float | None = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fpath(name: str) -> str:
    """表（3つのラベルと出口を付けた frame）。"""
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
# ラベル（build_dataset.attach_rise_label をそのまま使う）
# --------------------------------------------------------------------------- #
LABEL_INPUTS = ["Code", "Date", "close", "open", "vol_20d"]


def label_on(panel: pd.DataFrame, k: float) -> pd.DataFrame:
    """
    panel（銘柄・日付の順に並んだ日足。Code・Date・close・open・vol_20d）に、到達しきい値 k のラベルを付けて
    Code・Date・label を返す。ラベルの式は build_dataset.attach_rise_label（k 以外は本番の設定）。
    """
    cfg = replace(B.DEFAULT_RISE, vol_norm_k=float(k))
    out = B.attach_rise_label(panel[LABEL_INPUTS].copy(), cfg)
    return out[["Code", "Date", "label"]]


def topix_series(indices: pd.DataFrame, topix: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """
    indices からコード 0000（TOPIX）の始値・終値（日付の順）。topix（Date・topix）を渡すと、共通の日の終値の一致を数える。
    始値・終値が欠測か 0 以下の日、日付の重なりは止める。
    """
    ix = indices.copy()
    ix["Code"] = ix["Code"].astype(str).str.strip()
    t = ix[ix["Code"] == TOPIX_CODE][["Date", "O", "C"]].copy()
    if not len(t):
        raise SystemExit(f"indices に TOPIX（コード {TOPIX_CODE}）がありません")
    t["Date"] = pd.to_datetime(t["Date"]).astype("datetime64[ns]")
    for c in ("O", "C"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    dup = int(t["Date"].duplicated().sum())
    if dup:
        raise SystemExit(f"TOPIX の日付が重なっている: {dup}日")
    bad = int((~(t["O"] > 0) | ~(t["C"] > 0)).sum())
    if bad:
        raise SystemExit(f"TOPIX の始値・終値が欠測か 0 以下の日がある: {bad}日")
    t = t.sort_values("Date").reset_index(drop=True)
    info = {"days": int(len(t)), "first": str(t["Date"].min().date()), "last": str(t["Date"].max().date())}
    if topix is not None:
        r = topix[["Date", "topix"]].copy()
        r["Date"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
        r["topix"] = pd.to_numeric(r["topix"], errors="coerce")
        m = t.merge(r.dropna().drop_duplicates("Date", keep="last"), on="Date", how="inner")
        diff = float((m["C"] - m["topix"]).abs().max()) if len(m) else float("nan")
        info.update({"topix_common_days": int(len(m)), "topix_max_abs_diff": diff})
        if not len(m) or not diff < 1e-6:
            raise SystemExit(f"indices の TOPIX の終値が topix の値と合わない（共通 {len(m):,}日・差の最大 {diff}）")
    return t, info


def lagged(tp: pd.DataFrame, lag: int) -> pd.DataFrame:
    """TOPIX の始値・終値を、TOPIX の営業日で lag 日前の値に置き換えたもの（最初の lag 日は欠測）。"""
    out = tp[["Date", "O", "C"]].sort_values("Date").reset_index(drop=True)
    if lag:
        out[["O", "C"]] = out[["O", "C"]].shift(lag)
    return out


def relative_panel(panel: pd.DataFrame, tp: pd.DataFrame) -> pd.DataFrame:
    """
    株価 ÷ TOPIX の日足（終値は TOPIX の終値で、寄りは TOPIX の始値で割る）。vol_20d は銘柄自身の値のまま。
    TOPIX の最後の日より後の行は落とす（その先のラベルは「決まらない」になる）。行の並びは panel のまま。
    """
    p = panel.loc[panel["Date"] <= tp["Date"].max(), LABEL_INPUTS]
    m = p.merge(tp[["Date", "O", "C"]], on="Date", how="left")
    if len(m) != len(p):
        raise SystemExit("TOPIX との突き合わせで行が増えた（TOPIX の日付が重なっている）")
    m["close"] = m["close"] / m["C"]
    m["open"] = m["open"] / m["O"]
    return m[LABEL_INPUTS]


def missing_topix_days(panel_dates: pd.Series, tp: pd.DataFrame, start) -> list:
    """日足にあって TOPIX に無い日（start 以降、TOPIX の最後の日まで）。ここが空でないと超過ラベルが黙って欠ける。"""
    d = pd.to_datetime(pd.Series(pd.unique(panel_dates))).astype("datetime64[ns]")
    d = d[(d >= pd.Timestamp(start)) & (d <= tp["Date"].max())]
    have = set(tp["Date"])
    return sorted(x for x in d if x not in have)


def topix_forward(dates: pd.Series, tp: pd.DataFrame, horizon: int = B.RISE_HORIZON) -> np.ndarray:
    """日付 t ごとの TOPIX の騰落（%。t+1 の始値 → t+horizon の終値。TOPIX の営業日で数える）。t が TOPIX に無ければ欠測。"""
    t = tp.sort_values("Date").reset_index(drop=True)
    pos = pd.Series(np.arange(len(t)), index=t["Date"])
    p = pd.to_datetime(dates).astype("datetime64[ns]").map(pos).to_numpy(dtype=float)
    o, c = t["O"].to_numpy(dtype=float), t["C"].to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & (p + horizon < len(t))
    i = p[ok].astype(int)
    out[ok] = (c[i + horizon] / o[i + 1] - 1.0) * 100.0
    return out


def compare_labels(keys: pd.DataFrame, want: pd.Series, lbl: pd.DataFrame) -> dict:
    """keys（Code・Date）の行ごとに、want と lbl（Code・Date・label）のラベルが値も欠測もそろうか。"""
    m = keys[["Code", "Date"]].merge(lbl, on=["Code", "Date"], how="left")
    if len(m) != len(keys):
        raise SystemExit("ラベルの突き合わせで行が増えた（日足の Code・Date が重なっている）")
    a = pd.to_numeric(pd.Series(want).reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(m["label"], errors="coerce").to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    return {"rows": int(len(a)), "same": int((both & (a == b)).sum()), "both": int(both.sum()),
            "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum())}


def check_or_stop(name: str, c: dict) -> None:
    print(f"  確かめ: {name} 一致 {c['same']:,} / {c['both']:,}行（欠測の食い違い {c['nan_mismatch']:,}行・全 {c['rows']:,}行）")
    if c["same"] != c["both"] or c["nan_mismatch"]:
        raise SystemExit(f"{name} が合わない（ラベルの作り方がずれている）")


# --------------------------------------------------------------------------- #
# 学習の行の重み
# --------------------------------------------------------------------------- #
def decay_weights(dates, train_end, half_life: float) -> np.ndarray:
    """w = 0.5^(train_end からの経過年数 / 半減期)。平均1にそろえる。"""
    age = (pd.Timestamp(train_end) - pd.to_datetime(pd.Series(dates))).dt.days.to_numpy(dtype=float) / 365.25
    if np.any(age < 0):
        raise ValueError("学習の最終日より後の行がある")
    w = 0.5 ** (age / float(half_life))
    return w / w.mean()


def fold_weights(dates, train_end, weight, seed: int) -> np.ndarray | None:
    """weight（WEIGHT_OF の値）に従う学習の行の重み。("shuffle", h) は半減期 h の重みを seed で入れ替えたもの。"""
    if weight is None:
        return None
    mode, hl = weight
    w = decay_weights(dates, train_end, hl)
    if mode == "decay":
        return w
    if mode == "shuffle":
        return np.random.default_rng(seed).permutation(w)
    raise ValueError(f"知らない重み: {mode}")


def weighted_spw(y: np.ndarray, w: np.ndarray | None) -> float:
    """正例の重み（scale_pos_weight）。重みなしは tuning.scale_pos_weight と同じ。重みありは重みを掛けた負例 ÷ 正例。"""
    if w is None:
        return tuning.scale_pos_weight(y)
    y = np.asarray(y, dtype=int)
    pos = float(np.sum(w[y == 1]))
    return float(np.sum(w[y == 0]) / pos) if pos > 0 else 1.0


# --------------------------------------------------------------------------- #
# データ（1回目に作って、以降の回は同じものを使う）
# --------------------------------------------------------------------------- #
KEEP = ["label", "y_cur", "y_x", "y_px", "ret_o1_20", "ret_o1_40", "hit10", "tp10c", "topix_fwd"]


def price_panel_for_labels(data_dir: str, codes) -> pd.DataFrame:
    """build_dataset.build と同じ作りの日足（price_panel）から、ラベルに要る列だけ。codes の銘柄だけ残す。"""
    bars = B.load_parts("bars", data_dir)
    segments = B.load_market_segments(data_dir)
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    p = B.price_panel(bars, start=start)
    del bars
    p["Code"] = p["Code"].astype(str)
    p = p.loc[p["Code"].isin(set(codes)), LABEL_INPUTS].reset_index(drop=True)
    p["Date"] = p["Date"].astype("datetime64[ns]")
    return p


def prepare(cols: list, rebuild: bool = False):
    """今のラベル（1.5σ）・TOPIX 超過・対照の3つのラベルと出口を付けた表（3つがすべて決まる行だけ）と記録。"""
    pf, ps = fpath("frame.parquet"), fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        miss = [c for c in cols if c not in df.columns]
        if not miss:
            log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
            return df, stamp
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
    checks = {}

    # --- 日足からラベルを作り直せることの確かめ（1.2σ = データセットのラベル、1.5σ = rise_label_for_k）--- #
    t0 = time.time()
    panel = price_panel_for_labels(lab.DATA_DIR, frame["Code"].unique())
    dup = int(panel.duplicated(["Code", "Date"]).sum())
    if dup:
        raise SystemExit(f"日足の Code・Date が重なっている: {dup:,}行")
    log(f"[data] 日足 {len(panel):,}行（表の {frame['Code'].nunique():,}銘柄）/ {time.time()-t0:.0f}秒")
    keys = frame[["Code", "Date"]]
    y12 = frame["label"]
    y15 = B.rise_label_for_k(frame, K)
    print("\n■ 確かめ（ラベルを日足から作り直す。超過ラベルは同じ関数に「株価 ÷ TOPIX」を渡すだけ）")
    checks["label_12"] = compare_labels(keys, y12, label_on(panel, B.VOL_NORM_K))
    check_or_stop(f"{B.VOL_NORM_K:.1f}σ（データセットのラベル）", checks["label_12"])
    checks["label_15"] = compare_labels(keys, y15, label_on(panel, K))
    check_or_stop(f"{K:.1f}σ（rise_label_for_k）", checks["label_15"])

    # --- TOPIX --- #
    tp, tinfo = topix_series(B.load_parts("indices", lab.DATA_DIR), B.load_parts("topix", lab.DATA_DIR))
    checks["topix"] = tinfo
    print(f"  確かめ: TOPIX（indices の {TOPIX_CODE}）{tinfo['days']:,}日（{tinfo['first']}〜{tinfo['last']}）/ topix の値と"
          f"共通 {tinfo['topix_common_days']:,}日で終値が一致（差の最大 {tinfo['topix_max_abs_diff']:.2g}）")
    lost = missing_topix_days(panel["Date"], tp, frame["Date"].min())
    checks["missing_topix_days"] = len(lost)
    print(f"  確かめ: 表の最初の日〜TOPIX の最後の日のうち、日足にあって TOPIX に無い日 {len(lost)}日")
    if lost:
        raise SystemExit(f"TOPIX の欠けた日がある（{len(lost)}日。最初 {lost[0].date()}・最後 {lost[-1].date()}）。"
                         "超過ラベルが黙って欠けるので止める")
    after = int((panel["Date"] > tp["Date"].max()).sum())
    checks["rows_after_topix"] = after
    tl = lagged(tp, PLACEBO_LAG)
    first_ok = tl.loc[tl["C"].notna(), "Date"].min()
    checks["placebo_first"] = str(pd.Timestamp(first_ok).date())
    print(f"  対照の TOPIX（{PLACEBO_LAG}営業日前の値）は {first_ok.date()} から（表の最初の日 {frame['Date'].min().date()}）")
    if pd.Timestamp(first_ok) > frame["Date"].min():
        raise SystemExit("対照の TOPIX が表の最初の日に間に合わない")

    t0 = time.time()
    lx = label_on(relative_panel(panel, tp), K).rename(columns={"label": "y_x"})
    lpx = label_on(relative_panel(panel, tl), K).rename(columns={"label": "y_px"})
    del panel
    df = frame.assign(y_cur=y15.to_numpy())
    df = df.merge(lx, on=["Code", "Date"], how="left").merge(lpx, on=["Code", "Date"], how="left")
    if len(df) != len(frame):
        raise SystemExit("超過ラベルの突き合わせで行が増えた")
    log(f"[data] 超過・対照のラベル / {time.time()-t0:.0f}秒（TOPIX の最後の日より後の日足 {after:,}行は使わない）")

    # --- 3つのラベルがすべて決まる行 --- #
    cur = df["y_cur"].notna()
    both = cur & df["y_x"].notna() & df["y_px"].notna()
    checks["rows"] = {"frame": int(len(df)), "cur": int(cur.sum()), "x_missing_on_cur": int((cur & df["y_x"].isna()).sum()),
                      "px_missing_on_cur": int((cur & df["y_px"].isna()).sum()), "used": int(both.sum())}
    print(f"  行: 表 {len(df):,} / 今のラベルが決まる {int(cur.sum()):,} / うち超過が決まらない "
          f"{checks['rows']['x_missing_on_cur']:,}・対照が決まらない {checks['rows']['px_missing_on_cur']:,} → 使う {int(both.sum()):,}")
    df = df[both].reset_index(drop=True)
    df["label"] = df["y_cur"]

    # --- 出口（実験83 と同じ +10% の指値。参考の物差し）と、その期間の TOPIX の騰落（ラベルの診断用）--- #
    t0 = time.time()
    bars_px = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(df[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    if len(ex) != len(df) or not (ex["Code"].to_numpy() == df["Code"].to_numpy()).all():
        raise SystemExit("出口の突き合わせで行の並びが変わった")
    k = E83.tp_key(TP)
    df["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float)
    df["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    checks["exits"] = {**chk, "ok": bool(exit_ok)}
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行・20日目の終値あり "
          f"{chk['with_close']:,} / {chk['rows']:,}行 → {'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}"
          f"（{time.time()-t0:.0f}秒）")
    if not exit_ok:
        df["hit10"] = np.nan
        df["tp10c"] = np.nan
    df["topix_fwd"] = topix_forward(df["Date"], tp)
    checks["topix_fwd_missing"] = int(df["topix_fwd"].isna().sum())

    overlap = sorted(set(cols) & set(KEEP))
    if overlap:
        raise SystemExit(f"特徴量とラベル・出口の列が重なっている: {overlap}")
    df = df[["Code", "Date"] + cols + KEEP].copy()
    yr = df["Date"].dt.year
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "rows_by_year": {str(y): int((yr == y).sum()) for y in sorted(yr.unique())}, "checks": checks,
             "cols": len(cols), "signature": F.signature(cols)}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=str)
    return df, stamp


def label_report(df: pd.DataFrame) -> dict:
    """ラベルの年ごとの正例率と、その期間の TOPIX の騰落との関係（件数・割合・相関だけ）。"""
    out = {"by_year": [], "by_tfwd": [], "corr_month": {}}
    names = (("y_cur", "今"), ("y_x", "超過"), ("y_px", "対照"))
    print(f"\n■ 0. ラベル（{len(df):,}行・{df['Date'].min().date()}〜{df['Date'].max().date()}。{K:.1f}σ。"
          "TOPIX は t+1 の寄り → t+20 の引け）")
    print(f"  {'年':<6}{'行':>7}{'今':>8}{'超過':>8}{'対照':>8}{'TOPIX 平均':>11}{'今≠超過':>9}{'今≠対照':>9}")
    for y, g in list(df.groupby(df["Date"].dt.year)) + [("全体", df)]:
        r = {"year": str(y), "n": int(len(g)), **{c: float(g[c].mean()) for c, _ in names},
             "tfwd": float(g["topix_fwd"].mean()), "chg_x": float((g["y_cur"] != g["y_x"]).mean()),
             "chg_px": float((g["y_cur"] != g["y_px"]).mean())}
        out["by_year"].append(r)
        print(f"  {str(y):<6}{r['n']:>7,}{r['y_cur']*100:>7.1f}%{r['y_x']*100:>7.1f}%{r['y_px']*100:>7.1f}%"
              f"{r['tfwd']:>+10.2f}%{r['chg_x']*100:>8.1f}%{r['chg_px']*100:>8.1f}%")
    band = pd.cut(df["topix_fwd"], bins=list(TFWD_EDGES))
    print("  TOPIX の騰落の区切りごとの正例率（今 / 超過 / 対照）: ", end="")
    parts = []
    for iv, g in df.groupby(band, observed=False):
        r = {"band": str(iv), "n": int(len(g)), **{c: float(g[c].mean()) if len(g) else float("nan") for c, _ in names}}
        out["by_tfwd"].append(r)
        parts.append(f"{iv} {r['n']:,}行 {r['y_cur']*100:.1f} / {r['y_x']*100:.1f} / {r['y_px']*100:.1f}%")
    print("  ".join(parts))
    m = df.groupby(df["Date"].dt.to_period("M")).agg(n=("y_cur", "size"), tfwd=("topix_fwd", "mean"),
                                                    **{c: (c, "mean") for c, _ in names})
    m = m[m["n"] >= 20]
    for c, ja in names:
        out["corr_month"][c] = float(m[c].corr(m["tfwd"])) if len(m) > 2 else float("nan")
    print(f"  月ごとの正例率と TOPIX の騰落の相関（20行以上の {len(m)}か月）: "
          + " / ".join(f"{ja} {out['corr_month'][c]:+.2f}" for c, ja in names))
    return out


# --------------------------------------------------------------------------- #
# 探索（週次の再学習と同じ関数・条件）
# --------------------------------------------------------------------------- #
def tune(arm: str, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, compute: bool = True):
    p = path(f"params_{arm}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        if (rec.get("_n_trials") == n_trials and rec.get("_cutoff") == str(cutoff.date())
                and rec.get("_data") == stamp["built_utc"]):
            return rec
    if not compute or not time_left():
        return None
    sub = frame.loc[frame["Date"] <= cutoff].copy()
    sub["label"] = sub[TRAIN_LABEL[arm]]
    sub = sub[sub["label"].notna()]
    t0 = time.time()
    log(f"  [{arm}] 探索（tuning.tune・{n_trials}試行 × {N_SPLITS}分割・{len(sub):,}件・{len(cols)}列・"
        f"学習のラベル {TRAIN_LABEL[arm]}）")
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
    log(f"  [{arm}] {rec['_seconds']}秒 / CV PR-AUC {cv['mean_pr_auc']:.4f}（±{cv['std']:.4f}）/ ROC {cv['mean_roc_auc']:.4f}"
        f"（{TRAIN_LABEL[arm]} で測った値）")
    return rec


# --------------------------------------------------------------------------- #
# out-of-fold（E41.oof_folds の LightGBM と同じ組み方に、学習のラベルと重みを足したもの）
# --------------------------------------------------------------------------- #
OOF_KEEP = ["Code", "Date", "label", "y_x", "ret_o1_20", "ret_o1_40", "hit10", "tp10c"]


def oof_arm(frame: pd.DataFrame, cols: list, params: dict, seed: int, folds, train_col: str = "label",
            weight=None, perm_base: int = PERM_SEED) -> pd.DataFrame:
    """
    folds で out-of-fold を作る。学習は train_col のラベル（決まる行だけ）、評価の label はそのまま持ち出す。
    weight は WEIGHT_OF の値（None なら重みなし。E41.oof_folds の LightGBM と同じ結果になる）。
    入れ替え（"shuffle"）の種は perm_base + 窓の番号。
    """
    import lightgbm as lgb

    if isinstance(params, dict) and isinstance(params.get("params"), dict):
        params = params["params"]
    d = pd.to_datetime(frame["Date"])
    ok_tr = frame[train_col].notna()
    parts = []
    for f in folds:
        tr = frame[(d <= pd.Timestamp(f.train_end)) & ok_tr]
        te = frame[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & frame["label"].notna()]
        if len(te) < 200 or len(tr) < 1000:
            continue
        Xtr = tr[cols].to_numpy(dtype=float)
        ytr = tr[train_col].to_numpy(dtype=int)
        w = fold_weights(tr["Date"], f.train_end, weight, perm_base + int(f.index))
        m = lgb.LGBMClassifier(**{**params, "random_state": seed}, scale_pos_weight=weighted_spw(ytr, w))
        if w is None:
            m.fit(Xtr, ytr)
        else:
            m.fit(Xtr, ytr, sample_weight=w)
        part = te[[c for c in OOF_KEEP if c in te.columns]].copy()
        part["score"] = m.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
        part["fold"] = f.index
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def oof(arm: str, frame: pd.DataFrame, cols: list, params: dict, shift: int, seeds, compute: bool = True):
    """腕 × パラメータ × ずらし × 種の out-of-fold（種の平均）。保存済みなら読む。"""
    ph = params_hash(params) + "_d" + hashlib.sha1(str(frame.attrs.get("built_utc", "")).encode()).hexdigest()[:6]
    files = [path(f"{arm}_{ph}_sh{shift}_s{sd}.parquet") for sd in seeds]
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
        o = oof_arm(frame, cols, params, sd, folds, TRAIN_LABEL[arm], WEIGHT_OF[arm], PERM_SEED + 100 * shift)
        o.to_parquet(f, index=False)
        parts.append(o)
        log(f"    {arm} ずらし{shift}か月 種{sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
    o = average(parts)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    return o


def windows_on(o: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """窓ごとの AUC（label_col のラベルで測る）と、窓の期間・件数・正例率。MIN_WINDOW_ROWS 未満の窓は落とす。"""
    cols = ["fold", "pr", "roc", "start", "end", "n", "pos"]
    sub = o.assign(label=o[label_col]) if label_col != "label" else o
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


def top_by_window(o: pd.DataFrame, share: float = TOP_SHARE) -> pd.DataFrame:
    """窓ごとの上位 share（件数は切り上げ）の Precision（今のラベル）・+10% に届いた割合・1取引の平均（%）。"""
    rows = []
    for f, g in o.groupby("fold"):
        n = len(g)
        if n < MIN_WINDOW_ROWS:
            continue
        k = max(1, int(np.ceil(share * n)))
        t = g.iloc[np.argsort(-g["score"].to_numpy(dtype=float), kind="mergesort")[:k]]
        rows.append({"fold": int(f), "n": n, "k": k, "prec": float(t["label"].mean()),
                     "hit10": float(pd.to_numeric(t["hit10"], errors="coerce").mean()),
                     "tp10c": float(pd.to_numeric(t["tp10c"], errors="coerce").mean())})
    return pd.DataFrame(rows, columns=["fold", "n", "k", "prec", "hit10", "tp10c"])


def run(arms: list, frame: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, shifts: list) -> dict:
    """判定の腕を先に、参考の腕を後に計算する。時間切れの腕は結果に入らない。"""
    res = {"recs": {}, "params": {}, "prod": {}, "oofs": {}}
    res["recs"]["A"] = tune("A", frame, cols, cutoff, n_trials, stamp)
    if res["recs"]["A"] is None:
        return res
    for group in ([a for a in PRIMARY if a in arms], [a for a in REFERENCE if a in arms]):
        if "XB" in group:
            # XB の探索は判定の腕の out-of-fold の後に回す
            res["recs"]["XB"] = tune("XB", frame, cols, cutoff, n_trials, stamp)
        for arm in group:
            rec = res["recs"].get(PARAMS_OF[arm])
            if rec is None:
                continue
            res["params"][arm] = rec["params"]
            o = oof(arm, frame, cols, rec["params"], 0, (PROD_SEED,))
            if o is not None:
                res["prod"][arm] = o
        for sh in shifts:
            for arm in group:
                if arm not in res["params"]:
                    continue
                o = oof(arm, frame, cols, res["params"][arm], sh, SEEDS)
                if o is not None:
                    res["oofs"][(sh, arm)] = o
    return res


def complete_arms(res: dict, shifts: list) -> list:
    return [a for a in ARMS if a in res["prod"] and all((sh, a) in res["oofs"] for sh in shifts)]


def window_table(res: dict, arms: list, shifts: list, label_col: str = "label") -> pd.DataFrame:
    """窓ごとの値（ずらし × 窓）。arms がそろった窓だけ。"""
    rows = []
    for sh in shifts:
        w = None
        for arm in arms:
            x = windows_on(res["oofs"][(sh, arm)], label_col).rename(columns={"pr": f"pr_{arm}", "roc": f"roc_{arm}"})
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
            x = top_by_window(res["oofs"][(sh, arm)]).rename(
                columns={c: f"{c}_{arm}" for c in ("prec", "hit10", "tp10c")})
            x = x.drop(columns=["n", "k"]) if w is not None else x
            w = x if w is None else w.merge(x, on="fold")
        if w is None or not len(w):
            continue
        rows += [{"shift": sh, **r} for r in w.to_dict("records")]
    return pd.DataFrame(rows)


def year_table(res: dict, arms: list) -> list:
    """本番と同じ作りの OOF の年ごとの PR-AUC ÷ 正例率（今のラベル）。"""
    from sklearn.metrics import average_precision_score

    rows = []
    base = res["prod"]["A"]
    for y, g in base.groupby(base["Date"].dt.year):
        r = {"year": int(y), "n": int(len(g)), "pos": float(g["label"].mean())}
        for arm in arms:
            h = res["prod"][arm]
            h = h[h["Date"].dt.year == y]
            yy = h["label"].to_numpy(dtype=int)
            r[arm] = (float(average_precision_score(yy, h["score"]) / yy.mean())
                      if len(yy) and 0 < yy.sum() < len(yy) else float("nan"))
        rows.append(r)
    return rows


def report(res: dict, shifts: list, stamp: dict, lab_out: dict) -> dict:
    arms = complete_arms(res, shifts)
    missing = [a for a in ARMS if a not in arms]
    print("\n" + "=" * 78)
    print(f"ラベル {K:.1f}σ / 結果のそろった腕: {', '.join(arms) or 'なし'}" + (f" / まだ: {', '.join(missing)}" if missing else ""))
    summary = {"data": stamp, "labels": lab_out, "label_k": K, "perm_seed": PERM_SEED, "placebo_lag": PLACEBO_LAG,
               "half_lives": HALF_LIVES, "tuning": {}, "production_oof": {}, "by_year": [], "windows": {},
               "windows_excess_label": {}, "judge7": {}, "top": {}, "per_window": []}

    print(f"\n■ 1. 探索の CV（{CV_SCHEME}・{N_SPLITS}分割。楽観側に出る。パラメータ選び用。自分の学習のラベルで測った値）")
    for arm in ("A", "XB"):
        rec = res["recs"].get(arm)
        if not rec:
            continue
        cv = rec["_cv"]
        fs = cv.get("fold_scores") or []
        print(f"  {arm:<4}{cv['mean_pr_auc']:>8.4f} ±{cv['std']:.4f}  ROC {cv['mean_roc_auc']:.4f}  {rec.get('_seconds', 0)}秒  "
              f"分割ごと " + " / ".join(f"{v:.4f}" for v in fs))
        summary["tuning"][arm] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        print(f"      パラメータ {short(rec['params'])}")
    if "A" not in arms:
        print("  基準 A の out-of-fold がまだそろっていない（続きは次の回）")
        with open(path("summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
        return summary

    print("\n■ 2. 本番と同じ作りの out-of-fold（36/6/6か月・エンバーゴ20営業日・ずらし0・種42。評価は今のラベル）")
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
    yt = year_table(res, arms)
    summary["by_year"] = yt
    print("  年ごとの PR-AUC ÷ 正例率（今のラベル。本番と同じ作りの OOF）")
    print(f"    {'年':<6}{'件数':>7}{'正例率':>8}" + "".join(f"{a:>7}" for a in arms))
    for r in yt:
        print(f"    {r['year']:<6}{r['n']:>7,}{r['pos']*100:>7.1f}%" + "".join(f"{r[a]:>7.2f}" for a in arms))

    s = window_table(res, arms, shifts)
    s.to_csv(path("auc_by_window.csv"), index=False)
    print(f"\n■ 3. 窓ごと（境界を {'/'.join(map(str, shifts))}か月ずらした{len(shifts)}通り × 種3つの平均。評価は今のラベル）")
    print(f"  {'':<24}{'前':>9}{'後':>9}{'差の平均':>10}{'SE':>9}{'上の窓':>9}{'同じ':>6}")
    for met, nm in (("pr", "PR"), ("roc", "ROC")):
        for x, y in PAIRS:
            if x in arms and y in arms:
                line, st = pair(f"{nm} {y}−{x}", s[f"{met}_{x}"].to_numpy(), s[f"{met}_{y}"].to_numpy())
                print(line)
                summary["windows"][f"{met}_{y}-{x}"] = st
    sx = window_table(res, arms, shifts, label_col="y_x")
    print("  参考: TOPIX 超過のラベルで測った PR-AUC")
    for x, y in (("A", "X"), ("A", "PX"), ("A", "XB")):
        if x in arms and y in arms and len(sx):
            line, st = pair(f"超過で PR {y}−{x}", sx[f"pr_{x}"].to_numpy(), sx[f"pr_{y}"].to_numpy())
            print(line)
            summary["windows_excess_label"][f"pr_{y}-{x}"] = st

    print("\n■ 4. §7（2026-09-26 改訂）の1モデル分の3条件（32窓の今のラベルの PR-AUC。回す前に決めた）: "
          "平均の差が正 / 対照の差を上回る / 上の窓が過半")
    for part, (c, p, name) in JUDGES.items():
        if not all(a in arms for a in ("A", c, p)):
            print(f"  {name}: 腕がそろっていない（{c}・{p}）")
            continue
        j = judge7((s[f"pr_{c}"] - s["pr_A"]).to_numpy(), (s[f"pr_{p}"] - s["pr_A"]).to_numpy())
        summary["judge7"][part] = {"arm": c, "placebo": p, **j}
        print(f"  {name}: {c}−A {j['mean_ca']:>+8.4f} {'○' if j['positive'] else '×'} / 対照 {p}−A {j['mean_pa']:>+8.4f} "
              f"{'○' if j['beats_placebo'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
              f"{'○' if j['majority'] else '×'} → {'満たす' if j['pass'] else '満たさない'}")

    tt = top_table(res, arms, shifts)
    if len(tt):
        tt.to_csv(path("top_by_window.csv"), index=False)
        has_exit = bool(tt[[f"hit10_{a}" for a in arms]].notna().any().any())
        mets = (("prec", "Precision", 100), ("hit10", "+10%", 100), ("tp10c", "平均", 1)) if has_exit \
            else (("prec", "Precision", 100),)
        print(f"\n■ 5. 窓の中の上位{TOP_SHARE*100:g}%（参考。{len(tt)}窓。件数は窓ごとに切り上げ・合計 {int(tt['k'].sum()):,}件）: "
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

    print("\n■ 6. 窓ごとの値（今のラベルの PR-AUC。差は「後 − 前」）")
    for sh in shifts:
        g = s[s["shift"] == sh].sort_values("fold")
        if not len(g):
            continue
        print(f"  ずらし{sh}か月（{len(g)}窓）")
        print(f"    {'窓':>2}  {'期間':<23}{'件数':>7}{'正例率':>7}" + "".join(f"{a:>8}" for a in arms)
              + "".join(f"{c + '−A':>9}" for c, _, _ in JUDGES.values() if c in arms))
        for _, r in g.iterrows():
            print(f"    {int(r['fold']):>2}  {pd.Timestamp(r['start']).date()}〜{pd.Timestamp(r['end']).date()}"
                  f"{int(r['n']):>7,}{r['pos']*100:>6.1f}%" + "".join(f"{r[f'pr_{a}']:>8.4f}" for a in arms)
                  + "".join(f"{r[f'pr_{c}'] - r['pr_A']:>+9.4f}" for c, _, _ in JUDGES.values() if c in arms))
    summary["per_window"] = s.assign(start=s["start"].astype(str), end=s["end"].astype(str)).to_dict("records")
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    return summary


def main(argv=None) -> int:
    global TAG, RUN, DEADLINE
    ap = argparse.ArgumentParser(description="実験104: TOPIX 超過のラベル・古い年の重みを下げる（lgbm 1.5σ。検証だけ）")
    ap.add_argument("--arms", default=",".join(ARMS), help="この回に計算する腕（A は必ず含める）")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--tag", default=TAG, help="保存するファイルの頭（試運転は本番の回と分ける）")
    ap.add_argument("--rebuild", action="store_true", help="表を作り直す（以前の結果は使えなくなる）")
    ap.add_argument("--budget-min", type=float, default=290.0, help="この分数を過ぎたら新しい計算を始めない（0 で無制限）")
    ap.add_argument("--dry", action="store_true", help="ラベルの付き方だけ出して終わる")
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
    print("実験104 市場の伸びの影響を除く: 目的変数を TOPIX 超過に（部A）・古い年の重みを下げる（部B）"
          f"（lgbm {K:.1f}σ・本番と同じ作りの OOF + 32窓・対照 PX / PW）")
    print(f"  列: {BASE_PRESET} {len(cols)}列 指紋 {F.signature(cols)}")
    for a in ARMS:
        print(f"  {LABELS[a]}{'' if a in arms else '（この回は計算しない）'}")
    print(f"  探索 {args.n_trials}試行 × {N_SPLITS}分割（{CV_SCHEME}・{CV_OBJECTIVE}）/ 窓のずらし {shifts}か月 / 種 {SEEDS} / "
          f"対照の TOPIX のずらし {PLACEBO_LAG}営業日 / 重みの入れ替えの種 {PERM_SEED} / 上限 {args.budget_min:g}分 / 保存 {RUN}_*")
    print("=" * 78)

    df, stamp = prepare(cols, args.rebuild)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df.attrs["built_utc"] = stamp["built_utc"]      # out-of-fold の名前に表の版を入れる（作り直しの前の結果を混ぜない）
    lab_out = label_report(df)
    if args.dry:
        return 0
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"\n  この回の行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行")
    t0 = time.time()
    res = run(arms, df, cols, cutoff, args.n_trials, stamp, shifts)
    log(f"[{ALGO}] 計算 {time.time()-t0:.0f}秒 / そろった腕 {', '.join(complete_arms(res, shifts)) or 'なし'}")
    report(res, shifts, stamp, lab_out)
    if TIMED_OUT:
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    log(f"記録: {OOF_DIR}/{RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
