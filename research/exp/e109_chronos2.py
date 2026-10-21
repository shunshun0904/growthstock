#!/usr/bin/env python3
"""
実験109: Amazon の時系列の基盤モデル Chronos-2（TSFM）で、78週高値の更新日の候補（データセットの行。目的変数は 1.2σ）が
どれだけ当たるかを、基準モデル（1.2σ の LightGBM）と同じ行・同じ窓で比べる。検証だけ（本番の設定・データセットの作りには書かない）。

運用者の依頼（2026-10-21）「amazonのクロノスv2（TSFM）による予測精度を検証したいです。特徴量が多すぎる場合は、gbdtでの重要度が
高い上位30変数くらいでまず試してみてほしいです」。調べたこと（2026-10-21）: Chronos-2 は Amazon の 120M パラメータのモデル
（arXiv 2510.15821。Hugging Face の amazon/chronos-2。公式パッケージ chronos-forecasting 2.3.2 の Chronos2Pipeline）。過去の値だけ
分かる共変量を入れて予測できる。系列ごとに窓の中の平均・標準偏差でそろえてから読む（chronos_bolt.InstanceNorm）ので、共変量の
値の水準（PER が 15 か 40 か）は伝わらず、窓の中での動きだけが伝わる。この作業の環境からは Hugging Face に繋がらない（403）。
→ 運用者の選択（AskUserQuestion）: 実行場所「GitHub Actions で回す」・上位30列「毎日動く列から30本」・比べ方「単体で比べる」。
運用者の方針「いきなり変えるよりも、まずは検証してほしいです。本番導入するかはその結果次第です」。

行と窓
  データセットの行（78週高値の新規の更新日 t）。窓は本番と同じ前進分割（36/6/6か月、ずらし0。E41.folds_for）。各窓のテストの行を
  3つの腕で採点して比べる。Chronos-2 は学習しない（ゼロショット）ので、窓は比べる単位としてだけ使う。

腕
  A    基準モデル: LightGBM・1.2σ・本番の既定の360列・本番のパラメータ（E27.prod_params）。窓ごとに学習し、種3つ（42・7・123）の平均
  Z0   Chronos-2 株価だけ（共変量なし）
  Z30  Chronos-2 ＋ 上位30列を過去の共変量に

上位30列（窓ごと。回す前に決めた）
  その窓の学習データで学習した A の3つの種の gain の平均で並べ、毎日（または毎週）値が変わる列（ELIGIBLE）の上から30本。評価の
  期間を列選びに使わない（実験61 の「評価期間を学習済みのモデルで列を選んだことによる見かけ」を避ける）。
  回をまたぐとき（2026-10-21 に足した）: データセットは作り直すたびに値がわずかに違い（同じ生データでも指紋が変わる。試運転と
  本番の1回目で確かめた）、A を学習し直すと gain の30位前後が入れ替わる窓がある。その窓は束の名前が変わって前の回の分位点が
  使えない。そこで今回と、保存してある回ごとの A の gain の上位列を候補にし、窓ごとに保存済みの束の行が多いものを使う（同じなら
  今回の列）。どれも「その窓の学習データで学習した A の gain の上位」で、評価の行の結果は選び方に入らない。どの回の列かは ■ 0b。
  ELIGIBLE（159列）: テクニカル指標111・市場環境11・業種指数4・直近の増減5・高値への近さ3・ボラの分解6・出来高の傾き1・信用倍率1・
  需給（空売り比率・投資部門別）12、ブレイクの性質のうち毎日の値がある4（break_margin・close_position・ret_20d・vol_20d）、
  売買代金1（log_trading_value）。
  外した列: 更新日の行にしか値が無い列（base_length・broken_high_age・prev_nh_ratio・base_depth・pullback_depth・max_dd_120）、
  決算・開示・配当・保有の報告・予想の修正・開示からの日数・業種や市場の区分、時価総額（株数が決算の値）・バリュエーション（株価と
  決算の値の比）、信用規制（公表の日にしか変わらない）。これらは窓の中では段か直線にしかならず、水準は正規化で消える。

予測と点数（回す前に決めた。2026-10-21）
  目的の系列: その銘柄の日足（price_panel。分割を調整した終値）の t までの直近 CONTEXT=256 行（約1年）の終値の対数。共変量は
  同じ行の値（データセットと同じ関数で作る。covariates）。先は20行（t+1〜t+20。ラベルの数え方と同じ）。分位点はモデルの持つ全部。
  判定に使う点数: p_end = P(log C[t+20] − log C[t] ≥ log(1 + 必要上昇率))。必要上昇率はラベルと同じ式（rise_thresholds。1.2σ×√20）。
  分位点の値を小さい順に並べ直し（交差を直す）、間は直線で補い、外側は端の確率で止める（prob_above）。
  参考の点数（判定には使わない）: p_reach（t+1〜t+20 のうち最大の到達確率）、med（20日目の中央値の上昇 ÷ log(1 + 必要上昇率)）、
  p10（+10% に届く確率の最大）。

判定（回す前に決めた。2026-10-21）
  1. Chronos-2 と基準モデル: 窓ごとの PR-AUC（1.2σ）の差 Z − A の平均・SE・上の窓の数と、テストの全行を日付で束ねたブートストラップ
     （2,000回）での PR-AUC の差の 95% 区間。上回る = 平均が正・区間の下端が正・上の窓が過半。下回る = 平均が負・区間の上端が負・
     上の窓が過半に届かない。ほかは「決められない」。
  2. Chronos-2 に当てる力があるか: Z の PR-AUC ÷ 正例率（lift）と ROC-AUC の 95% 区間（同じブートストラップ）の下端が 1・0.5 を
     超えるか。
  3. 共変量の効果: Z30 − Z0 を 1 と同じ作りで。
  参考: ROC-AUC・日付内 AUC、窓の中の上位2% の Precision・+10% の指値に届いた割合・1取引の平均（実験83 の出口）、モデルの公開
  （2025年10月）より後の行（2025-11-01 以降）だけの集計（Chronos-2 の学習データと重ならない期間）、予測の当たり具合（20日目の
  10〜90% の区間に実際の終値が入った割合、p_end の平均と実際に20日目の終値が必要上昇率を超えた割合）。

確かめ（合わなければ止める）
  - 基準モデルの学習が実験の共通部品（E41.oof_folds）と同じ点数を出すこと（最初の窓・種42）
  - 共変量の値: 候補の行 t の値が、データセットの ELIGIBLE の全列と一致すること（相対 1e-6・絶対 1e-9。e105.compare_columns）
  - 日足と学習データの行の並び・調整が同じこと（e105.check_alignment）

出すもの（公開ログには件数・割合・日付・精度・列名だけ。銘柄・値は出さない）

    exp=e109_chronos2.py                       本番の回。--budget-min までに終わらない束は計算せずに終わる（続きは次の回が使う）
    exp=e109_chronos2.py args="--smoke"        試運転（窓ごとに先頭 SMOKE_ROWS 行。速さを測る。保存は e109smoke_*）
手元（Chronos-2 の手前まで。共変量の確かめ・上位30列・文脈の大きさ）: python3 research/exp/e109_chronos2.py --dry

Run Experiment は同じ実験を同時に1本しか走らせない。途中の結果（基準モデルの out-of-fold・共変量の表・Chronos-2 の分位点）は
research/_data/oof/<tag>_* に置き、actions/cache で次の回に引き継ぐ。成果物にはしない（銘柄ごとの値を含むため）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import extra_features as EF  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import technicals as TX  # noqa: E402
import trading_calendar  # noqa: E402
import tuning  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
from e104_market_drift import top_by_window  # noqa: E402
from e105_technical_ab import check_alignment, compare_columns  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
TAG = "e109"
SEEDS = (42, 7, 123)
SHIFT = 0
TOP_N = 30
CONTEXT = 256
HORIZON = 20
TP = LT.TAKE_PROFIT
#: Chronos-2 の公開（arXiv 2510.15821 は 2025年10月）より後の行。学習データと重ならない期間として別に集計する
POST_RELEASE = pd.Timestamp("2025-11-01")
N_BOOT = 2000
BOOT_SEED = 20261021
CHUNK_ROWS = 512
SMOKE_ROWS = 64
#: 1回の計算に入れる行の数（Chronos-2 の batch_size は系列の本数なので、Z30 は 31倍して渡す）
BATCH_ITEMS_Z0 = 256
BATCH_ITEMS_Z30 = 16
MODEL_ID = "amazon/chronos-2"
#: ランナーで入れる版（2週間以上前に出たもの。chronos-forecasting 2.3.2 は 2026-09-08）。torch は CPU 版の置き場から
PIP_TORCH = "torch==2.14.0"
PIP_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
PIP_CHRONOS = ["chronos-forecasting==2.3.2", "transformers==5.16.1", "accelerate==1.15.0", "einops==0.8.2"]

#: 毎日（または毎週）値が変わる列の系統（features.GROUPS の名前）。決め方は冒頭の説明
ELIGIBLE_GROUPS = ("tech_indicators", "market", "sector_index", "recent_change", "price", "vol_factors", "volume",
                   "supply", "flow")
#: 系統としては外したが、毎日の値がある列
ELIGIBLE_EXTRA = ("break_margin", "close_position", "ret_20d", "vol_20d", "log_trading_value")
#: 日足（price_panel・add_breakout_context）にそのままある列
PANEL_FEATURES = ("r_high", "r_high_3m", "r_high_6m", "vol_20d", "ret_20d", "break_margin", "close_position",
                  "ret_1d", "ret_2d", "gap_open", "ret_5d", "vol_chg_1d", "volume_trend",
                  "vol_rel_long", "vol_rel_short", "vol_updown", "vol_gap_ratio")
ARMS = ("A", "Z0", "Z30")
LABELS = {"A": "A   基準モデル（1.2σ の LightGBM・360列）", "Z0": "Z0  Chronos-2 株価だけ",
          "Z30": "Z30 Chronos-2 ＋ 上位30列"}

DEADLINE: Optional[float] = None
TIMED_OUT = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    """Chronos-2 の分位点・共変量（試運転は e109smoke_*）。"""
    return os.path.join(OOF_DIR, f"{TAG}_{name}")


def apath(name: str) -> str:
    """基準モデルの out-of-fold と gain（試運転と本番の回で同じものを使う）。"""
    return os.path.join(OOF_DIR, f"e109_{name}")


def time_left(need_sec: float = 0.0) -> bool:
    global TIMED_OUT
    if DEADLINE is None or time.time() + need_sec < DEADLINE:
        return True
    if not TIMED_OUT:
        TIMED_OUT = True
        log("[time] --budget-min までに終わらない束は計算しない（収まる小さな束は計算する。保存済みの分で報告。続きは次の回）")
    return False


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# 列
# --------------------------------------------------------------------------- #

def eligible_columns(cols: Sequence[str]) -> List[str]:
    """cols のうち、毎日（または毎週）値が変わる列（並びは cols のまま）。"""
    return [c for c in cols if F.group_of(c) in ELIGIBLE_GROUPS or c in ELIGIBLE_EXTRA]


def top_eligible(gain: pd.Series, eligible: Sequence[str], n: int = TOP_N) -> List[str]:
    """gain（列名 -> 値）の大きい順に、eligible の列から n 本。同じ値は列名の順（決定的にする）。"""
    g = gain.reindex(list(eligible)).fillna(0.0)
    order = sorted(g.index, key=lambda c: (-float(g[c]), c))
    return order[:n]


def tops_candidates(eligible: Sequence[str], n: int = TOP_N) -> Dict[str, Dict[int, List[str]]]:
    """
    保存してある A の gain（回ごと。apath の A_<指紋>_gain.parquet）それぞれの、窓ごとの上位 n 列。毎日動く列がすべて
    gain に載っているものだけ（列の組が違う回の gain は使わない）。名前は <指紋>。
    """
    out = {}
    head, tail = apath("A_"), "_gain.parquet"
    for p in sorted(glob.glob(apath("A_*_gain.parquet"))):
        g = pd.read_parquet(p)
        if not {"col", "gain", "fold"} <= set(g.columns) or not set(eligible) <= set(g["col"]):
            continue
        out[p[len(head):-len(tail)]] = {int(f): top_eligible(x.set_index("col")["gain"], eligible, n)
                                        for f, x in g.groupby("fold")}
    return out


def choose_tops(current: Dict[int, List[str]], cands: Dict[str, Dict[int, List[str]]], saved) -> tuple:
    """
    窓ごとに使う上位列。今回の A の列と、保存してある回ごとの候補のうち、保存済みの束の行（saved(窓, 列) -> 行の数）が
    いちばん多いもの。同じなら今回の列、その次は候補の名前の順。戻り値は (窓 -> 列, 窓 -> (どの回の列か, 保存済みの行))。

    データセットを作り直すたびに値がわずかに違い（同じ生データでも指紋が変わる）、A を学習し直すと gain の30位前後が
    入れ替わる窓がある。その窓は束の名前（入力の値から作る）が変わり、前の回の分位点が使えなくなるので、列はどれも
    「その窓の学習データで学習した A の gain の上位」のまま、保存済みの多いほうを使う（評価の行は列選びに使わない）。
    """
    chosen, why = {}, {}
    for f, cur in current.items():
        opts, seen = [], []
        for name, cols in [("今回", cur)] + [(k, v[f]) for k, v in sorted(cands.items()) if f in v]:
            if cols not in seen:
                seen.append(cols)
                opts.append((name, cols, saved(f, cols)))
        name, cols, n_saved = max(opts, key=lambda t: t[2])     # 同じ行の数なら先に並べたもの（今回の列）
        chosen[f] = cols
        why[f] = (name, int(n_saved))
    return chosen, why


# --------------------------------------------------------------------------- #
# 基準モデル（A）: 本番と同じ作りの out-of-fold と、窓ごとの gain
# --------------------------------------------------------------------------- #

def fold_rows(df: pd.DataFrame, f) -> Optional[tuple]:
    """窓の学習・テストの行（E41.oof_folds と同じ切り方）。行が足りなければ None。"""
    d = pd.to_datetime(df["Date"])
    ok = df["label"].notna()
    tr = df[(d <= pd.Timestamp(f.train_end)) & ok]
    te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & ok]
    if len(te) < 200 or len(tr) < 1000:
        return None
    return tr, te


def fold_fit(df: pd.DataFrame, cols: List[str], params: dict, seed: int, f) -> Optional[tuple]:
    """
    1つの窓の学習と予測（E41.oof_folds の LightGBM と同じ組み方）。戻り値は (テストの行, 点数, gain) か、行が足りなければ None。
    """
    import lightgbm as lgb

    r = fold_rows(df, f)
    if r is None:
        return None
    tr, te = r
    ytr = tr["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**{**params, "random_state": seed}, scale_pos_weight=tuning.scale_pos_weight(ytr))
    m.fit(tr[cols].to_numpy(dtype=float), ytr)
    sc = m.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
    gain = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols, dtype=float)
    return te, sc, gain


def arm_a(df: pd.DataFrame, cols: List[str], folds) -> tuple:
    """A の out-of-fold（種3つの平均）と、窓ごとの gain（種3つの平均）。保存済みなら読む。"""
    params = E27.prod_params("lgbm")["params"]
    fp = AB.fingerprint(df, cols) + "_" + digest(params)
    po, pg = apath(f"A_{fp}_oof.parquet"), apath(f"A_{fp}_gain.parquet")
    if os.path.exists(po) and os.path.exists(pg):
        log(f"[A] 保存済みの out-of-fold と gain を読む（{fp}）")
        return pd.read_parquet(po), pd.read_parquet(pg)
    # 確かめ: 最初の窓・種42 で、実験の共通部品と同じ点数になること
    first = next(f for f in folds if fold_rows(df, f) is not None)
    want = E41.oof_folds("lgbm", df, cols, params, SEEDS[0], [first])
    got = fold_fit(df, cols, params, SEEDS[0], first)
    diff = float(np.max(np.abs(want["score"].to_numpy() - got[1])))
    print(f"  確かめ: 基準モデルの学習が E41.oof_folds と同じ点数か（窓{first.index}・種{SEEDS[0]}）"
          f" {len(got[1]):,}行の差の最大 {diff:.2e}")
    if diff > 1e-12:
        raise SystemExit("基準モデルの学習が実験の共通部品と合わない")
    rows, gains = [], []
    for f in folds:
        t0 = time.time()
        sc, gs, te = [], [], None
        for sd in SEEDS:
            r = fold_fit(df, cols, params, sd, f)
            if r is None:
                break
            te = r[0]
            sc.append(r[1])
            gs.append(r[2])
        if te is None:
            continue
        part = te[["Code", "Date", "label"]].copy()
        part["score"] = np.mean(sc, axis=0)
        part["fold"] = f.index
        rows.append(part)
        g = pd.concat(gs, axis=1).mean(axis=1).rename("gain").reset_index().rename(columns={"index": "col"})
        g["fold"] = f.index
        gains.append(g)
        log(f"  A 窓{f.index}（{f.test_start}〜{f.test_end}）: テスト {len(part):,}行 {time.time()-t0:.0f}秒")
    o = pd.concat(rows, ignore_index=True)
    gn = pd.concat(gains, ignore_index=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    o.to_parquet(po, index=False)
    gn.to_parquet(pg, index=False)
    return o, gn


# --------------------------------------------------------------------------- #
# 日足と共変量（データセットと同じ関数）
# --------------------------------------------------------------------------- #

def build_panel(data_dir: str, codes) -> pd.DataFrame:
    """
    build_dataset.build と同じ作りの日足（price_panel → mark_new_highs → attach_rise_label → add_breakout_context）。
    codes の銘柄だけ、要る列だけ残す（銘柄・日付の順）。
    """
    bars = B.load_parts("bars", data_dir)
    segments = B.load_market_segments(data_dir)
    gm = B.general_market_start(bars, segments)
    start = gm.loc[gm["tpm"]].set_index("Code")["start"] if B.GENERAL_MARKET_START else None
    p = B.price_panel(bars, start=start)
    del bars
    p = B.mark_new_highs(p)
    p = B.attach_rise_label(p)
    p = B.add_breakout_context(p)
    keep = list(dict.fromkeys(["Code", "Date", "open", "high", "low", "close", "vol", "trading_value", "tv_ma20"]
                              + list(PANEL_FEATURES)))
    miss = [c for c in keep if c not in p.columns and c != "trading_value"]
    if miss:
        raise SystemExit(f"日足に無い列: {miss}")
    p["Code"] = p["Code"].astype(str)
    p = p.loc[p["Code"].isin(set(codes)), [c for c in keep if c in p.columns]]
    p["Date"] = pd.to_datetime(p["Date"]).astype("datetime64[ns]")
    p = p.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    if p.duplicated(["Code", "Date"]).any():
        raise SystemExit("日足の Code・Date が重なっている")
    return p


def load_sources(data_dir: str) -> dict:
    """共変量に要る日次・週次のデータ（build_dataset.build と同じ読み方）。"""
    bars = B.load_parts("bars", data_dir)
    topix = B.load_parts("topix", data_dir)
    env = B.market_environment(bars, topix)
    del bars
    try:
        indices = B.load_parts("indices", data_dir)
    except SystemExit:
        indices = pd.DataFrame(columns=["Date", "Code", "C"])
    try:
        margin = B.load_parts("margin", data_dir)
    except SystemExit:
        margin = pd.DataFrame(columns=["Date", "Code", "LongVol", "ShrtVol"])
    import glob
    mh_paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if not mh_paths:
        raise SystemExit("master_hist が無い（業種指数の列が作れない）")
    mh = pd.concat([pd.read_parquet(x) for x in mh_paths], ignore_index=True)
    mh["Date"] = pd.to_datetime(mh["Date"])
    mh = (mh[["Date", "Code", "S33"]].dropna(subset=["Date", "Code"])
          .sort_values("Date").drop_duplicates(["Date", "Code"], keep="last"))
    return {"env": env, "indices": indices, "margin": margin, "mh": mh,
            "days": trading_calendar.load(data_dir).days, "data_dir": data_dir}


def covariates(keys: pd.DataFrame, cols: Sequence[str], panel: pd.DataFrame, src: dict,
               dtype=np.float64) -> pd.DataFrame:
    """
    keys（Code・Date。日足の行）の cols の値（keys の並びのまま）。build_dataset.build の samples に付けるのと同じ関数・同じ順で
    付ける（日足の列 → 市場環境 → 業種 → 業種指数 → 市場・業種とのボラの比 → 信用倍率 → 需給 → テクニカル指標）。
    """
    cols = list(cols)
    k = keys[["Code", "Date"]].copy().reset_index(drop=True)
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"]).astype("datetime64[ns]")
    n = len(k)
    pcols = ["Code", "Date", "tv_ma20"] + list(PANEL_FEATURES)
    s = k.merge(panel[pcols], on=["Code", "Date"], how="left")
    if len(s) != n:
        raise SystemExit("日足との突き合わせで行が増えた")
    s["_i"] = np.arange(n)
    s["log_trading_value"] = np.log1p(s["tv_ma20"])
    s = pd.merge_asof(s.sort_values("Date"), src["env"], on="Date", direction="backward")
    s = pd.merge_asof(s.sort_values("Date"), src["mh"], on="Date", by="Code", direction="backward")
    s = s.sort_values("_i").reset_index(drop=True)
    s = B.attach_sector_index(s, src["indices"])
    s = B.attach_vol_vs_market(s)
    s = B.attach_credit_ratio(s, src["margin"], src["days"]).sort_values("_i").reset_index(drop=True)
    for fn in (EF.short_ratio, EF.investor_types):
        part = fn(s, src["data_dir"])
        for c in part.columns:
            s[c] = part[c].to_numpy()
    out = {}
    tx_cols = [c for c in cols if c in TX.COLUMNS]
    for c in cols:
        if c in tx_cols:
            continue
        if c not in s.columns:
            raise SystemExit(f"共変量の列が作れない: {c}")
        out[c] = pd.to_numeric(s[c], errors="coerce").to_numpy(dtype=dtype)
    if tx_cols:
        tp = panel[[c for c in B.TECH_PANEL_COLS if c in panel.columns]]
        vals = {c: np.full(n, np.nan, dtype=dtype) for c in tx_cols}
        codes = k["Code"].to_numpy()
        pcodes = panel["Code"].to_numpy()
        order = [c for c in pd.unique(pcodes) if c in set(codes)]
        for st in range(0, len(order), B.TECH_CODES_PER_CHUNK):
            chunk = set(order[st:st + B.TECH_CODES_PER_CHUNK])
            sel = np.isin(codes, list(chunk))
            sub = tp.loc[np.isin(pcodes, list(chunk))].reset_index(drop=True)
            tx = TX.compute_at(sub, k.loc[sel, ["Code", "Date"]])
            for c in tx_cols:
                vals[c][sel] = tx[c].to_numpy(dtype=dtype)
        out.update(vals)
    frame = pd.DataFrame({c: out[c] for c in cols})
    return frame.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# 文脈（各行の t までの直近 CONTEXT 行）
# --------------------------------------------------------------------------- #

def context_bounds(panel: pd.DataFrame, rows: pd.DataFrame, context: int) -> tuple:
    """
    rows（Code・Date）の各行について、日足の中の (始めの行, t の行) を返す（どちらも panel の位置。同じ銘柄の中だけ）。
    panel は Code・Date の順に並び、rows の (Code, Date) は panel に必ずあること（無ければ止める）。
    """
    key = pd.MultiIndex.from_frame(panel[["Code", "Date"]])
    pos = key.get_indexer(pd.MultiIndex.from_frame(
        rows[["Code", "Date"]].assign(Code=rows["Code"].astype(str),
                                      Date=pd.to_datetime(rows["Date"]).astype("datetime64[ns]"))))
    if (pos < 0).any():
        raise SystemExit(f"日足に無い行がある: {int((pos < 0).sum()):,}行")
    codes = panel["Code"].to_numpy()
    first = pd.Series(np.arange(len(panel))).groupby(codes).transform("min").to_numpy()
    start = np.maximum(pos - context + 1, first[pos])
    return start, pos


# --------------------------------------------------------------------------- #
# 点数（分位点 -> 確率）
# --------------------------------------------------------------------------- #

def prob_above(q: np.ndarray, levels: Sequence[float], x: np.ndarray) -> np.ndarray:
    """
    q: (n, Q)（1つの先の行の分位点の値）、levels: (Q,)、x: (n,) -> P(Y ≥ x)。分位点の値は小さい順に並べ直す（交差を直す）。
    間は直線で補い、いちばん外の分位点の外は端の確率で止める（下に外れれば 1 − levels[0]、上に外れれば 1 − levels[-1]）。
    """
    lv = np.asarray(levels, dtype=float)
    qs = np.sort(np.asarray(q, dtype=float), axis=1)
    out = np.full(len(qs), np.nan)
    for i in range(len(qs)):
        if np.isfinite(x[i]) and np.isfinite(qs[i]).all():
            out[i] = 1.0 - float(np.interp(x[i], qs[i], lv))
    return out


def scores(qt: np.ndarray, levels: Sequence[float], last_log: np.ndarray, need: np.ndarray) -> pd.DataFrame:
    """
    qt: (n, Q, H) の終値の対数の分位点。last_log: log C[t]、need: 必要上昇率（比。0.1 = +10%）。
    戻り値: p_end（判定）・p_reach・med・p10 と、外側で止めた行の印（clip_hi: x が最上位の分位点より上）。
    """
    n, nq, h = qt.shape
    lv = list(levels)
    thr = last_log + np.log1p(need)
    thr10 = last_log + np.log1p(TP / 100.0)
    p_h = np.column_stack([prob_above(qt[:, :, j], lv, thr) for j in range(h)])
    p10 = np.column_stack([prob_above(qt[:, :, j], lv, thr10) for j in range(h)])
    q_sorted = np.sort(qt[:, :, h - 1], axis=1)
    mid = lv.index(0.5) if 0.5 in lv else int(np.argmin(np.abs(np.asarray(lv) - 0.5)))
    with np.errstate(divide="ignore", invalid="ignore"):
        med = (q_sorted[:, mid] - last_log) / np.log1p(need)
    return pd.DataFrame({"p_end": p_h[:, -1], "p_reach": np.nanmax(p_h, axis=1), "med": med,
                         "p10": np.nanmax(p10, axis=1), "clip_hi": thr > q_sorted[:, -1],
                         "crossed": (np.diff(qt[:, :, h - 1], axis=1) < 0).any(axis=1)})


# --------------------------------------------------------------------------- #
# Chronos-2
# --------------------------------------------------------------------------- #

def ensure_chronos() -> None:
    """chronos-forecasting と torch（CPU 版）が無ければ入れる（ランナーで1回だけ。research/requirements.txt には入れない）。"""
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    log("[pip] torch（CPU 版）と chronos-forecasting を入れる")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", PIP_TORCH, "--index-url", PIP_TORCH_INDEX])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PIP_CHRONOS])


def load_pipeline():
    """amazon/chronos-2 を読む。取った版（Hugging Face の commit）を返す。"""
    import torch
    from huggingface_hub import snapshot_download
    from chronos import Chronos2Pipeline

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    local = snapshot_download(MODEL_ID)
    pipe = Chronos2Pipeline.from_pretrained(local)
    pipe.model.eval()
    return pipe, os.path.basename(os.path.normpath(local))


def forecast(pipe, items: list, batch_series: int) -> np.ndarray:
    """items（Chronos2Pipeline.predict の入力）の分位点（n, Q, HORIZON）。"""
    import torch

    with torch.no_grad():
        out = pipe.predict(items, prediction_length=HORIZON, batch_size=batch_series, context_length=CONTEXT)
    arr = np.stack([o.detach().cpu().numpy()[0] for o in out]).astype(np.float32)
    if arr.shape[1:] != (len(pipe.quantiles), HORIZON):
        raise SystemExit(f"予測の形が想定と違う: {arr.shape}")
    return arr


def make_items(panel_close: np.ndarray, start: np.ndarray, end: np.ndarray, cov: Optional[np.ndarray] = None,
               cov_idx: Sequence[int] = (), cov_names: Sequence[str] = (), cov_row: Optional[np.ndarray] = None) -> list:
    """
    行ごとの入力。目的の系列は終値の対数（日足の start〜end 行）。cov があれば同じ日足の行の共変量（cov_row で cov の行に、
    cov_idx で列に引く。名前は cov_names）。
    """
    items = []
    for s, e in zip(start, end):
        with np.errstate(divide="ignore", invalid="ignore"):
            tgt = np.log(panel_close[s:e + 1]).astype(np.float32)
        tgt[~np.isfinite(tgt)] = np.nan              # 売買の無い日・0 以下の終値は欠測として渡す（Chronos-2 は欠測を伏せて読む）
        if cov is None:
            items.append(tgt)
            continue
        r = cov_row[s:e + 1]
        if (r < 0).any():
            raise SystemExit("共変量の表に無い日足の行がある")
        block = cov[r][:, list(cov_idx)]
        items.append({"target": tgt,
                      "past_covariates": {c: block[:, j].astype(np.float32) for j, c in enumerate(cov_names)}})
    return items


def chunk_key(meta: dict, keys: pd.DataFrame, items: list) -> str:
    """
    保存の名前。束の行（Code・Date）と、モデルに渡す入力の値そのもの（目的の系列・共変量の名前と値）と meta から作る。
    日々の取り込みで行が増えても、分割で過去の調整後の値が直っても、入力が同じ束はそのまま使え、違う束だけ計算し直す。
    """
    h = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode())
    h.update(pd.util.hash_pandas_object(keys.reset_index(drop=True), index=False).to_numpy().tobytes())
    for it in items:
        if isinstance(it, dict):
            h.update(np.ascontiguousarray(it["target"]).tobytes())
            for c, v in it["past_covariates"].items():
                h.update(c.encode())
                h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(np.ascontiguousarray(it).tobytes())
    return h.hexdigest()[:16]


def arm_chunks(arm: str, rev: str, rows: pd.DataFrame, close: np.ndarray, start: np.ndarray, end: np.ndarray,
               tops: Dict[int, List[str]], cov_tab: Optional[np.ndarray], cov_cols: List[str],
               cov_row: Optional[np.ndarray], folds=None):
    """
    腕の束（窓ごと・CHUNK_ROWS 行ごと）を順に: (窓, 束の頭, 窓の行数, 行の位置, 入力, 保存の名前)。folds で窓を絞れる。
    """
    meta = {"arm": arm, "ctx": CONTEXT, "h": HORIZON, "model": MODEL_ID, "rev": rev}
    for fold, idx in rows.groupby("fold").indices.items():
        if folds is not None and int(fold) not in folds:
            continue
        idx = np.sort(idx)
        cols = tops[int(fold)] if arm == "Z30" else []
        for st in range(0, len(idx), CHUNK_ROWS):
            sl = idx[st:st + CHUNK_ROWS]
            if arm == "Z30":
                items = make_items(close, start[sl], end[sl], cov_tab, [cov_cols.index(c) for c in cols], cols, cov_row)
            else:
                items = make_items(close, start[sl], end[sl])
            p = path(f"{arm}_{chunk_key(meta, rows.iloc[sl][['Code', 'Date']], items)}.npz")
            yield int(fold), st, len(idx), sl, items, p


def saved_rows(rev: str, rows: pd.DataFrame, close: np.ndarray, start: np.ndarray, end: np.ndarray, fold: int,
               cols: List[str], cov_tab: np.ndarray, cov_cols: List[str], cov_row: np.ndarray) -> int:
    """窓 fold を列 cols で Z30 にしたときの束のうち、分位点が保存済みの行の数（計算はしない）。"""
    n = 0
    for _f, _st, _n, sl, _items, p in arm_chunks("Z30", rev, rows, close, start, end, {fold: cols}, cov_tab, cov_cols,
                                                 cov_row, folds={fold}):
        if os.path.exists(p):
            n += len(sl)
    return n


def run_arm(arm: str, pipe, rev: str, rows: pd.DataFrame, panel: pd.DataFrame, start: np.ndarray, end: np.ndarray,
            tops: Dict[int, List[str]], cov_tab: Optional[np.ndarray], cov_cols: List[str],
            cov_row: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    腕（Z0 / Z30）の分位点を、窓ごと・CHUNK_ROWS 行ごとに計算して保存する（保存済みは読む）。全部そろえば (行, 分位点) を返す。
    """
    close = panel["close"].to_numpy(dtype=float)
    parts, missing = [], 0
    for fold, st, n_fold, sl, items, p in arm_chunks(arm, rev, rows, close, start, end, tops, cov_tab, cov_cols,
                                                     cov_row):
        if os.path.exists(p):
            z = np.load(p)
            parts.append((sl, z["q"]))
            continue
        if pipe is None or not time_left(EST_SEC.get(arm, 0.0) * len(sl) * 1.3):
            missing += len(sl)
            continue
        bs = (len(tops[fold]) + 1) * BATCH_ITEMS_Z30 if arm == "Z30" else BATCH_ITEMS_Z0
        t0 = time.time()
        q = forecast(pipe, items, bs)
        np.savez_compressed(p, q=q)
        parts.append((sl, q))
        dt = time.time() - t0
        EST_SEC[arm] = dt / len(sl)
        log(f"  {arm} 窓{fold} の {st + len(sl):,} / {n_fold:,}行 {dt:.0f}秒（1行 {dt / len(sl):.3f}秒）")
    if missing:
        log(f"[{arm}] まだの行 {missing:,}（次の回で続ける）")
        return None
    order = np.concatenate([p[0] for p in parts])
    q = np.concatenate([p[1] for p in parts])
    if len(order) != len(rows) or len(np.unique(order)) != len(rows):
        raise SystemExit(f"{arm}: 分位点の行が採点する行とそろわない（{len(order):,} / {len(rows):,}）")
    out = np.empty_like(q)
    out[order] = q
    return out


EST_SEC: Dict[str, float] = {}


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #

def pr_roc(y: np.ndarray, s: np.ndarray) -> tuple:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if y.min() == y.max():
        return np.nan, np.nan
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def by_window(o: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """窓ごとの PR-AUC・ROC-AUC（腕ごと）。o は fold・label・s_<腕> を持つ。"""
    rows = []
    for f, g in o.groupby("fold"):
        y = g["label"].to_numpy(dtype=int)
        r = {"fold": int(f), "n": len(g), "pos": float(y.mean())}
        for a in arms:
            r[f"pr_{a}"], r[f"roc_{a}"] = pr_roc(y, g[f"s_{a}"].to_numpy(dtype=float))
        rows.append(r)
    return pd.DataFrame(rows)


def boot(o: pd.DataFrame, arms: Sequence[str], n: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """日付で束ねたブートストラップ。回ごとの PR-AUC・ROC-AUC・正例率（腕ごと）。"""
    rng = np.random.default_rng(seed)
    dates = o["Date"].to_numpy()
    uniq, inv = np.unique(dates, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    y_all = o["label"].to_numpy(dtype=int)
    s_all = {a: o[f"s_{a}"].to_numpy(dtype=float) for a in arms}
    rows = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        y = y_all[idx]
        r = {"pos": float(y.mean())}
        for a in arms:
            r[f"pr_{a}"], r[f"roc_{a}"] = pr_roc(y, s_all[a][idx])
        rows.append(r)
    return pd.DataFrame(rows)


def ci(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def verdict(mean: float, lo: float, hi: float, wins: int, n: int) -> str:
    """判定 1・3 の言い方（回す前に決めた）。"""
    if mean > 0 and lo > 0 and wins > n / 2:
        return "上回る"
    if mean < 0 and hi < 0 and wins < n / 2:
        return "下回る"
    return "決められない"


def report(o: pd.DataFrame, arms: Sequence[str], tops: Dict[int, List[str]], diag: Dict[str, pd.DataFrame]) -> dict:
    """判定と参考の集計を出す（件数・割合・精度だけ）。"""
    out = {"rows": int(len(o)), "folds": int(o["fold"].nunique()), "pos_rate": float(o["label"].mean())}
    w = by_window(o, arms)
    bt = boot(o, arms)
    print(f"\n■ 1. 窓ごとの PR-AUC（1.2σ のラベル。{len(w)}窓・{len(o):,}行・正例率 {o['label'].mean()*100:.1f}%）")
    print(f"  {'窓':<4}{'行':>7}{'正例率':>8}" + "".join(f"{a:>9}" for a in arms))
    for _, r in w.iterrows():
        print(f"  {int(r['fold']):<4}{int(r['n']):>7,}{r['pos']*100:>7.1f}%" + "".join(f"{r[f'pr_{a}']:>9.4f}" for a in arms))
    whole = {a: pr_roc(o["label"].to_numpy(dtype=int), o[f"s_{a}"].to_numpy(dtype=float)) for a in arms}
    print("\n■ 2. 全行（日付で束ねたブートストラップ 2,000回の 95% 区間）")
    print(f"  {'腕':<34}{'PR-AUC':>9}{'区間':>18}{'lift':>7}{'lift の区間':>16}{'ROC-AUC':>9}{'区間':>18}{'日付内AUC':>10}")
    out["arms"] = {}
    for a in arms:
        pr, roc = whole[a]
        lo, hi = ci(bt[f"pr_{a}"])
        llo, lhi = ci(bt[f"pr_{a}"] / bt["pos"])
        rlo, rhi = ci(bt[f"roc_{a}"])
        day = lab.auc_in_day(o.assign(score=o[f"s_{a}"]))
        lift = pr / o["label"].mean()
        print(f"  {LABELS[a]:<34}{pr:>9.4f}  [{lo:.4f}, {hi:.4f}]{lift:>7.2f}  [{llo:.2f}, {lhi:.2f}]"
              f"{roc:>9.4f}  [{rlo:.4f}, {rhi:.4f}]{day:>10.4f}")
        out["arms"][a] = {"pr": pr, "pr_ci": [lo, hi], "lift": lift, "lift_ci": [llo, lhi], "roc": roc,
                          "roc_ci": [rlo, rhi], "day_auc": day,
                          "skill": bool(llo > 1.0 and rlo > 0.5) if a != "A" else None}
    print("\n■ 3. 判定（回す前に決めた）")
    out["pairs"] = {}
    for b, a in (("A", "Z0"), ("A", "Z30"), ("Z0", "Z30")):
        if a not in arms or b not in arms:
            continue
        d = (w[f"pr_{a}"] - w[f"pr_{b}"]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        lo, hi = ci(bt[f"pr_{a}"] - bt[f"pr_{b}"])
        wins = int((d > 0).sum())
        v = verdict(float(d.mean()), lo, hi, wins, len(d))
        print(f"  {a} − {b}: 窓ごとの PR-AUC の差 平均 {d.mean():+.4f}（SE {se:.4f}・上の窓 {wins}/{len(d)}）・"
              f"全行の差の95%区間 [{lo:+.4f}, {hi:+.4f}] → {a} は {b} を「{v}」")
        out["pairs"][f"{a}-{b}"] = {"mean": float(d.mean()), "se": se, "wins": wins, "n": int(len(d)),
                                    "ci": [lo, hi], "verdict": v}
    for a in arms:
        if a == "A":
            continue
        s = out["arms"][a]
        print(f"  {a} に当てる力があるか: lift の区間の下端 {s['lift_ci'][0]:.2f}（1 を超えるか）・ROC-AUC の区間の下端 "
              f"{s['roc_ci'][0]:.4f}（0.5 を超えるか）→ {'ある' if s['skill'] else '言えない'}")

    # 参考: 窓の中の上位2%
    print("\n■ 4. 参考: 窓の中の上位2%（Precision・+10% の指値に届いた割合・1取引の平均）")
    out["top2"] = {}
    for a in arms:
        t = top_by_window(o.assign(score=o[f"s_{a}"]))
        if not len(t):
            continue
        out["top2"][a] = {"k": int(t["k"].sum()), "prec": float(t["prec"].mean()), "hit10": float(t["hit10"].mean()),
                          "tp10c": float(t["tp10c"].mean())}
        r = out["top2"][a]
        print(f"  {LABELS[a]:<34} {r['k']:>4}件  Precision {r['prec']*100:5.1f}%  +10% {r['hit10']*100:5.1f}%  "
              f"1取引 {r['tp10c']:+.2f}%")

    # 参考: 公開より後の期間
    post = o[o["Date"] >= POST_RELEASE]
    out["post"] = {"rows": int(len(post))}
    if len(post) >= 200 and post["label"].nunique() == 2:
        bp = boot(post, arms)
        print(f"\n■ 5. 参考: モデルの公開より後（{POST_RELEASE.date()} 以降。{len(post):,}行・正例率 "
              f"{post['label'].mean()*100:.1f}%）")
        for a in arms:
            pr, roc = pr_roc(post["label"].to_numpy(dtype=int), post[f"s_{a}"].to_numpy(dtype=float))
            lo, hi = ci(bp[f"pr_{a}"])
            out["post"][a] = {"pr": pr, "pr_ci": [lo, hi], "roc": roc}
            print(f"  {LABELS[a]:<34}PR-AUC {pr:.4f} [{lo:.4f}, {hi:.4f}]  ROC-AUC {roc:.4f}")
        for b, a in (("A", "Z0"), ("A", "Z30"), ("Z0", "Z30")):
            if a in arms and b in arms:
                lo, hi = ci(bp[f"pr_{a}"] - bp[f"pr_{b}"])
                print(f"  {a} − {b}: PR-AUC の差の95%区間 [{lo:+.4f}, {hi:+.4f}]")

    # 参考: 予測の当たり具合
    print("\n■ 6. 参考: 予測の当たり具合（Chronos-2）")
    out["calib"] = {}
    for a, dg in diag.items():
        r = {"cover_10_90": float(dg["covered"].mean()), "p_end_mean": float(dg["p_end"].mean()),
             "hit_end_rate": float(dg["hit_end"].mean()), "clip_hi": float(dg["clip_hi"].mean()),
             "crossed": float(dg["crossed"].mean())}
        out["calib"][a] = r
        print(f"  {a}: 20日目の 10〜90% の区間に実際が入った割合 {r['cover_10_90']*100:.1f}%（80% が目安）・"
              f"p_end の平均 {r['p_end_mean']*100:.1f}% / 実際に20日目の終値が必要上昇率を超えた割合 {r['hit_end_rate']*100:.1f}%・"
              f"必要上昇率が最上位の分位点より上の行 {r['clip_hi']*100:.1f}%・分位点が交差した行 {r['crossed']*100:.1f}%")

    # 参考の点数
    print("\n■ 7. 参考の点数（判定には使わない。全行の PR-AUC / ROC-AUC）")
    for a, dg in diag.items():
        y = o["label"].to_numpy(dtype=int)
        line = []
        for k in ("p_reach", "med", "p10"):
            pr, roc = pr_roc(y, dg[k].to_numpy(dtype=float))
            line.append(f"{k} {pr:.4f} / {roc:.4f}")
        print(f"  {a}: " + "  ".join(line))
    return out


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    global TAG, DEADLINE
    ap = argparse.ArgumentParser(description="実験109: Chronos-2 と基準モデルを同じ行・同じ窓で比べる")
    ap.add_argument("--tag", default="e109")
    ap.add_argument("--arms", default="A,Z0,Z30")
    ap.add_argument("--smoke", action="store_true", help=f"窓ごとに先頭 {SMOKE_ROWS} 行だけ（速さを測る）")
    ap.add_argument("--dry", action="store_true", help="Chronos-2 の手前まで（共変量の確かめ・上位30列）")
    ap.add_argument("--budget-min", type=float, default=290.0)
    args = ap.parse_args(argv)
    TAG = args.tag + ("smoke" if args.smoke and not args.tag.endswith("smoke") else "")
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad or "A" not in arms:
        raise SystemExit(f"腕は {ARMS} から（A は必ず入れる）: {bad}")
    if args.budget_min:
        DEADLINE = time.time() + args.budget_min * 60
    os.makedirs(OOF_DIR, exist_ok=True)

    t0 = time.time()
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame["Code"] = frame["Code"].astype(str)
    frame = frame[frame["label"].notna()].reset_index(drop=True)
    cols = F.columns(F.DEFAULT_PRESET)
    miss = [c for c in cols + ["vol_20d", "close_position", "ret_o1_20"] if c not in frame.columns]
    if miss:
        raise SystemExit(f"表に無い列: {miss[:8]}")
    elig = eligible_columns(cols)
    log(f"[data] {len(frame):,}行（{frame['Date'].min().date()}〜{frame['Date'].max().date()}）・{len(cols)}列"
        f"（うち毎日動く列 {len(elig)}）/ {time.time()-t0:.0f}秒")
    folds = E41.folds_for(frame["Date"], SHIFT)

    # --- A ---
    oA, gain = arm_a(frame, cols, folds)
    tops = {}
    for f, g in gain.groupby("fold"):
        tops[int(f)] = top_eligible(g.set_index("col")["gain"], elig)
    union = sorted({c for v in tops.values() for c in v}, key=lambda c: cols.index(c))
    print(f"\n■ 0. 窓ごとの上位{TOP_N}列（A の gain・毎日動く列から）: {len(tops)}窓で使う列は合わせて {len(union)}本")
    cnt = pd.Series([c for v in tops.values() for c in v]).value_counts()
    for c in union:
        print(f"  {c:<26}{F.group_of(c):<16}{int(cnt[c]):>3}窓")
    share = gain.groupby("col")["gain"].sum()
    print(f"  A の gain のうち毎日動く列の割合 {share.reindex(elig).sum() / share.sum() * 100:.1f}%")
    # 前の回の A の上位列も候補にする（窓ごとに、保存済みの束が多いほうを Chronos-2 の前で選ぶ。choose_tops）
    cands = tops_candidates(elig) if "Z30" in arms else {}
    union = sorted({c for v in [tops, *cands.values()] for lst in v.values() for c in lst}, key=lambda c: cols.index(c))
    if cands:
        diff = {k: sum(v.get(f) != t for f, t in tops.items()) for k, v in cands.items()}
        print(f"  保存してある回ごとの A の上位列: {len(cands)}回（今回と違う窓の数 "
              + "・".join(f"{k} {n}" for k, n in diff.items()) + f"）→ 共変量はその和 {len(union)}列で作る")

    rows = oA.copy()
    rows["Date"] = pd.to_datetime(rows["Date"]).astype("datetime64[ns]")
    rows["Code"] = rows["Code"].astype(str)
    if args.smoke:
        rows = rows.groupby("fold").head(SMOKE_ROWS).reset_index(drop=True)
    take = list(dict.fromkeys(["Code", "Date", "vol_20d", "close_position", "ret_o1_20"] + elig))
    rows = rows.merge(frame[take], on=["Code", "Date"], how="left", validate="one_to_one")
    need, _ = B.rise_thresholds(rows["vol_20d"], B.DEFAULT_RISE)
    rows["need"] = need.to_numpy(dtype=float)
    log(f"[rows] 採点する行 {len(rows):,}（{rows['fold'].nunique()}窓）")

    # --- 日足・共変量 ---
    t0 = time.time()
    panel = build_panel(lab.DATA_DIR, rows["Code"].unique())
    log(f"[panel] 日足 {len(panel):,}行（{panel['Code'].nunique():,}銘柄）/ {time.time()-t0:.0f}秒")
    checks = {"alignment": check_alignment(rows, panel)}
    src = load_sources(lab.DATA_DIR)
    t0 = time.time()
    at_t = covariates(rows[["Code", "Date"]], elig, panel, src)
    bad = compare_columns(rows[elig].reset_index(drop=True), at_t, cols=elig)
    n_bad = sum(bad.values())
    checks["covariates_at_t"] = {"rows": int(len(rows)), "columns": len(elig), "bad_cells": int(n_bad)}
    print(f"  確かめ: 候補の行 t の共変量とデータセットの値の一致 {len(rows):,}行 × {len(elig)}列 → 食い違い {n_bad:,}セル"
          f"（{time.time()-t0:.0f}秒）")
    if n_bad:
        raise SystemExit("共変量の値がデータセットと合わない: "
                         + ", ".join(f"{c} {v}" for c, v in bad.items() if v)[:400])
    start, end = context_bounds(panel, rows, CONTEXT)
    span = end - start + 1
    print(f"  文脈の長さ（行）: 中央値 {int(np.median(span))}・{CONTEXT} 行に満たない行 {(span < CONTEXT).mean()*100:.1f}%")
    need_rows = np.unique(np.concatenate([np.arange(s, e + 1) for s, e in zip(start, end)]))
    log(f"[cov] 文脈の日足 {len(need_rows):,}行に {len(union)}列を付ける")
    cov_tab, cov_row = None, None
    if "Z30" in arms:
        cp = path(f"cov_{digest({'rows': AB.fingerprint(rows, ['need']), 'ctx': CONTEXT, 'cols': union})}.npy")
        if os.path.exists(cp):
            cov_tab = np.load(cp)
            log(f"[cov] 保存済みの共変量を読む（{cov_tab.shape[0]:,}行 × {cov_tab.shape[1]}列）")
        else:
            t0 = time.time()
            cov_tab = covariates(panel.loc[need_rows, ["Code", "Date"]], union, panel, src,
                                 dtype=np.float32).to_numpy(dtype=np.float32)
            np.save(cp, cov_tab)
            log(f"[cov] {cov_tab.shape[0]:,}行 × {cov_tab.shape[1]}列 / {time.time()-t0:.0f}秒")
        cov_row = np.full(len(panel), -1, dtype=np.int64)
        cov_row[need_rows] = np.arange(len(need_rows))
        miss_share = float(np.isnan(cov_tab).mean())
        print(f"  共変量の欠測の割合（文脈の全セル）{miss_share*100:.1f}%")
    if args.dry:
        tok = (np.ceil(span / 16) + 3)
        print(f"\n[dry] Chronos-2 の手前で止める。1行の系列の長さ（16行で1つ）: 中央値 {np.median(tok):.0f}・"
              f"Z30 は {len(tops[next(iter(tops))]) + 1}本の系列")
        return 0

    # --- 出口（+10% の指値）---
    bars_px = LT.load_bars(lab.DATA_DIR, start=rows["Date"].min() - pd.Timedelta(days=10))
    bars_px["Date"] = bars_px["Date"].astype("datetime64[ns]")
    ex, chk = E83.attach_exits(rows[["Code", "Date", "ret_o1_20"]], bars_px, targets=(TP,))
    del bars_px
    k = E83.tp_key(TP)
    exit_ok = chk["hold5_match"] == chk["hold5_both"] and chk["order_violations"] == 0
    rows["hit10"] = ex[f"hit{k}"].to_numpy(dtype=float) if exit_ok else np.nan
    rows["tp10c"] = pd.to_numeric(ex[f"tp{k}c"], errors="coerce").to_numpy(dtype=float) if exit_ok else np.nan
    print(f"  確かめ: 出口の5日平均と ret_o1_20 の一致 {chk['hold5_match']:,} / {chk['hold5_both']:,}行 → "
          f"{'一致' if exit_ok else '合わない（上位2% の +10% は出さない）'}")

    # --- Chronos-2 ---
    rows["s_A"] = rows["score"]
    z_arms = [a for a in arms if a != "A"]
    pipe, rev = None, "none"
    ensure_chronos()
    t0 = time.time()
    pipe, rev = load_pipeline()
    log(f"[chronos] {MODEL_ID}（commit {rev}）を読んだ / {time.time()-t0:.0f}秒・分位点 {len(pipe.quantiles)}個"
        f"（{pipe.quantiles[0]}〜{pipe.quantiles[-1]}）")
    close = panel["close"].to_numpy(dtype=float)
    last_log = np.log(close[end])
    fwd = end + HORIZON
    same = (fwd < len(panel)) & (panel["Code"].to_numpy()[np.minimum(fwd, len(panel) - 1)] == rows["Code"].to_numpy())
    real_end = np.where(same, np.log(close[np.minimum(fwd, len(panel) - 1)]), np.nan)
    src_tops = {}
    if "Z30" in z_arms:
        memo = {}

        def saved(f, lst):
            k = (f, tuple(lst))
            if k not in memo:
                memo[k] = saved_rows(rev, rows, close, start, end, f, lst, cov_tab, union, cov_row)
            return memo[k]

        tops, src_tops = choose_tops(tops, cands, saved)
        n_fold = rows.groupby("fold").size()
        print(f"\n■ 0b. Z30 が窓ごとに使う上位{TOP_N}列（今回か前の回の A の gain の上位のうち、保存済みの束が多いほう）")
        for f in sorted(tops):
            name, n_saved = src_tops[f]
            print(f"  窓{f:<3} {name:<18} 保存済みの束 {n_saved:>6,} / {int(n_fold[f]):,}行")
    diag, done = {}, ["A"]
    for a in z_arms:
        q = run_arm(a, pipe, rev, rows, panel, start, end, tops, cov_tab, union, cov_row)
        if q is None:
            continue
        sc = scores(q, pipe.quantiles, last_log, rows["need"].to_numpy(dtype=float))
        rows[f"s_{a}"] = sc["p_end"].to_numpy()
        lv = list(pipe.quantiles)
        qs = np.sort(q[:, :, -1], axis=1)
        lo_i, hi_i = lv.index(0.1), lv.index(0.9)
        sc["covered"] = (real_end >= qs[:, lo_i]) & (real_end <= qs[:, hi_i])
        sc["hit_end"] = real_end - last_log >= np.log1p(rows["need"].to_numpy(dtype=float))
        diag[a] = sc
        done.append(a)
        if args.smoke:
            n = len(rows)
            print(f"  [smoke] {a}: 1行 {EST_SEC.get(a, float('nan')):.3f}秒 → 全行（約 {len(oA):,}行）で "
                  f"約 {EST_SEC.get(a, 0) * len(oA) / 3600:.1f}時間（今回 {n:,}行）")
    if len(done) < 2:
        log("[report] Chronos-2 の腕がまだそろっていない（次の回で続ける）")
        return 0
    res = report(rows, done, tops, diag)
    res.update({"checks": checks, "tops": tops, "tops_source": src_tops, "model": MODEL_ID, "revision": rev,
                "context": CONTEXT,
                "smoke": bool(args.smoke), "built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())})
    with open(path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
