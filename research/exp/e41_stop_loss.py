#!/usr/bin/env python3
"""
実験41: 「3モデル 90以上」で選んだ銘柄は、選んだ後にどう動くか（損切りの設計用）。

運用者の問い（2026-09-23）
  「3モデル 90以上（運用の基準）で選んだ場合、oof の正解率的に3銘柄のうち
    2つは20営業日後に、所望の水準に到達していないということかと思います。
    残り2つがどんな傾向、最大上昇率、最大下落率、20営業日以降も保有した
    場合の期待利益率等が知りたいです。というのも、ここからは損切りライン
    について、設計したいからです。」

選定は本番と同じ作り
  スコア  実験39 の腕 B2（本番の205列 all_plus・**205列で探索したパラメータ**・
          種3つの平均）。2026-09-27 の週次再学習から本番はこの形になる
          （探索は学習と同じ列で行う。運用者の指示 2026-09-23）。
          --arm B1 なら本番のパラメータのまま（実験39 の腕 B1）
  基準    3モデルすべてが、それより前の窓のスコア分布で90パーセンタイル超
          （ops_rule.consensus をそのまま使う）

ラベルの外れ方で4つに分ける（ラベルは基準日 t の終値から測る）
  正例              到達・終盤・トレンドの3条件をすべて満たした
  到達→終盤失速     20営業日以内に必要な上昇率に届いたが、t+20 の5日平均が
                    終盤の必要水準（到達しきい値の半分）を割った
  到達→トレンド割れ 終盤の水準は保ったが、t+20 で MA5 < MA20
  未到達            20営業日以内に必要な上昇率に届かなかった

値動きは**買値（翌営業日の寄り AdjO[t+1]）**から測る。物差し ret_o1_20 と同じ入り口。
  1日目 = 買った日（t+1）。足は銘柄ごとの並びで数える（lab.realized_returns と同じ）。
  売買が成立しなかった日も1日と数え、その日の値は欠測として扱う
  最大上昇率  1〜N日目の高値の最大 / 買値 − 1
  最大下落率  1〜N日目の安値の最小 / 買値 − 1
  N日保有     N日目の5日平均終値 / 買値 − 1（= ret_o1_N。表から作り直して一致を確かめる）

損切りの約定の仮定
  逆指値（場中）  安値が 買値×(1−X) に触れたらその値で約定。
                  寄りの時点で既に下回っていたら（窓を開けて割った）寄値で約定
                  ＝損は X より大きくなる
  終値で判定      終値が 買値×(1−X) 以下で引けたら、次に寄りが付いた日の寄値で売る
  利確 +20%       高値が 買値×1.20 に触れたらちょうど +20%（窓で飛び越えても +20%。
                  実験32 と同じ保守側）
  同じ日に損切りと利確の両方に触れたら、損切りが先だったとみなす（保守側）
  どれにも掛からなければ N日保有（ret_o1_N）
  手数料・スリッページ・税は入れない

使い方
  python3 research/exp/e41_stop_loss.py              # 腕 B2（205列で探索し直す。Actions 向け・約1時間）
  python3 research/exp/e41_stop_loss.py --arm B1     # 腕 B1（本番のパラメータのまま）
  python3 research/exp/e41_stop_loss.py --scores e27 # 手元の動作確認（保存済みの153列のスコア）

結果は research/_data/oof/e41_*.csv。**本番の設定には書かない。**
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import ops_rule as OR  # noqa: E402
import train_model as T  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e39_extra_multi as E39  # noqa: E402
from e25_auc_noise import average  # noqa: E402

OOF_DIR = E27.OOF_DIR
log = E39.log

RULE = "3モデル 90以上"
RULE_MODELS, RULE_PCT = {n: (m, p) for n, m, p in OR.RULES}[RULE]

K = 60                                     # 追う日数（60日版の最大上昇・最大下落まで）
HOLDS = (20, 40, 60)                       # 保有日数の候補（ret_o1_N と同じ出口）
CLASSES = ("正例", "到達→終盤失速", "到達→トレンド割れ", "未到達")
STOPS = (3, 5, 7, 8, 10, 12, 15, 20)       # 損切り（買値から %）
SIGMAS = (0.5, 0.75, 1.0, 1.25, 1.5)       # ラベルと同じ σ（vol_20d/100 × √20）の倍数
TP = 0.20                                  # 利確。実験32 / PLAYBOOK と同じ +20%
TOUCH = (3, 5, 7, 10, 15, 20)              # 「−X% に触れた銘柄はその後どうなったか」
DAYS = (1, 3, 5, 10, 15, 20, 40, 60)       # 値動きの途中経過を見る日
TSTOP_DAYS = (3, 5, 10)                    # 日数で切る: k日目の終値で判定
TSTOP_LEVELS = (-0.05, -0.03, 0.0)         # その時点で 買値+θ を下回っていたら切る
SHIFTS = (0, 2, 4)                         # §10 窓の境界をずらす月数（out-of-fold の切り方）
BIG = -0.10                                # 大負けの線（20営業日保有の収益）
SCREEN_Z = 3.0                             # 大負けの共通点として拾う |z|（205列を両側で測るので2では緩い）


# ---------------------------------------------------------------------- #
# 集計の小道具（NaN は落として数える）
# ---------------------------------------------------------------------- #

def _fin(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def mean(a) -> float:
    a = _fin(a)
    return float(a.mean()) if len(a) else float("nan")


def med(a) -> float:
    a = _fin(a)
    return float(np.median(a)) if len(a) else float("nan")


def qt(a, p: float) -> float:
    a = _fin(a)
    return float(np.percentile(a, p)) if len(a) else float("nan")


def win(a) -> float:
    a = _fin(a)
    return float((a > 0).mean()) if len(a) else float("nan")


def pc(x: float, w: int = 8, d: int = 2, sign: bool = True) -> str:
    """比率を % で。符号付き。"""
    if x is None or not np.isfinite(x):
        return f"{'-':>{w}}"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return f"{s:>{w}}"


def num(x: float, w: int = 6, d: int = 0) -> str:
    if x is None or not np.isfinite(x):
        return f"{'-':>{w}}"
    return f"{x:>{w}.{d}f}"


# ---------------------------------------------------------------------- #
# スコアと選定
# ---------------------------------------------------------------------- #

def load_scores(kind: str, df: pd.DataFrame, arm: str = "B2") -> dict:
    if kind == "e27":
        # 手元の動作確認用（153列の旧スコア・保存済み）。結論には使わない
        out = {}
        for a in RULE_MODELS:
            files = [os.path.join(OOF_DIR, f"e27_{a}_B1_s{s}.parquet") for s in E27.SEEDS3]
            missing = [p for p in files if not os.path.exists(p)]
            if missing:
                raise SystemExit(f"{missing[0]} がありません")
            out[a] = average([pd.read_parquet(p) for p in files])
        log("スコア: 実験27 の腕 B1（153列・動作確認用。結論には使わない）")
        return out
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    if len(cols) != len(F.columns(F.DEFAULT_PRESET)):
        raise SystemExit(f"{F.DEFAULT_PRESET} の列がデータセットに揃っていない")
    if arm == "B1":
        log(f"スコア: 実験39 の腕 B1（{F.DEFAULT_PRESET} {len(cols)}列・本番のパラメータ・"
            f"種 {E27.SEEDS3}）")
        return {a: E39.oof_arm(a, "B1", df, cols, E27.prod_params(a)) for a in RULE_MODELS}
    par = arm_params(df, cols, arm)
    log(f"スコア: 実験39 の腕 B2（{F.DEFAULT_PRESET} {len(cols)}列で探索したパラメータ・"
        f"種 {E27.SEEDS3}）")
    return {a: E39.oof_arm(a, "B2", df, cols, par[a]) for a in RULE_MODELS}


def arm_params(df: pd.DataFrame, cols: list, arm: str) -> dict:
    """
    腕のパラメータ。B1 は本番のパラメータ、B2 は205列で探索し直したもの
    （実験39 と同じ作法。探索はホールドアウトより手前だけ、5分割・50試行・
    year_cap_date。探索済みなら保存から読む）。本番の週次再学習は B2 の形。
    """
    if arm == "B1":
        return {a: E27.prod_params(a) for a in RULE_MODELS}
    d = pd.to_datetime(df["Date"])
    train_end, _, _ = T.holdout_bounds(d, T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    sub = df[(d <= train_end) & df["label"].notna()]
    return {a: E39.tune_for(a, sub, cols, F.DEFAULT_PRESET) for a in RULE_MODELS}


# ---------------------------------------------------------------------- #
# 窓の切り方を変えた out-of-fold（§10）
# ---------------------------------------------------------------------- #

def folds_for(dates: pd.Series, shift_months: int):
    """
    本番の out-of-fold と同じ作り（36ヶ月 / 6ヶ月 / 6ヶ月、エンバーゴ20営業日）で、
    窓の境界だけ shift_months か月後ろにずらす（実験26 と同じやり方）。
    訓練は従来どおり期間の最初から使う。
    """
    import walkforward as WF
    from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS

    d = pd.to_datetime(dates)
    if shift_months:
        d = d[d >= d.min() + pd.DateOffset(months=shift_months)]
    return WF.make_folds(d, min_train_months=OOF_MIN_TRAIN_MONTHS,
                         test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                         embargo_days=B.RISE_HORIZON)


def oof_folds(algo: str, df: pd.DataFrame, cols: list, params: dict, seed: int,
              folds) -> pd.DataFrame:
    """
    folds で out-of-fold を作る。学習器は実験39 と同じ（LightGBM は
    e19_freshdata.oof_for、他は e27_timing_multi.oof_multi と同じ組み方）。
    """
    import lightgbm as lgb
    import models as M
    import tuning
    import tuning_multi as TM

    if isinstance(params, dict) and isinstance(params.get("params"), dict):
        params = params["params"]
    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "label", "ret_o1_20", "ret_o1_40"]
    parts = []
    try:
        for f in folds:
            tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
            te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                    & df["label"].notna()]
            if len(te) < 200 or len(tr) < 1000:
                continue
            Xtr = tr[cols].to_numpy(dtype=float)
            ytr = tr["label"].to_numpy(dtype=int)
            Xte = te[cols].to_numpy(dtype=float)
            if algo == "lgbm":
                m = lgb.LGBMClassifier(**{**params, "random_state": seed},
                                       scale_pos_weight=tuning.scale_pos_weight(ytr))
                m.fit(Xtr, ytr)
                sc = m.predict_proba(Xte)[:, 1]
            else:
                TM.SEED = seed                  # build() が random_state に使う
                m = M.fit(algo, Xtr, ytr, cols, params=params)
                sc = M.predict(m, Xte)
            part = te[[c for c in keep if c in te.columns]].copy()
            part["score"] = sc
            part["fold"] = f.index
            parts.append(part)
    finally:
        TM.SEED = 0
    return pd.concat(parts, ignore_index=True)


def shifted_scores(df: pd.DataFrame, cols: list, par: dict, shift: int, arm: str) -> dict:
    """窓を shift か月ずらした out-of-fold（種3つの平均）。保存済みなら読む。"""
    folds = folds_for(df["Date"], shift)
    out = {}
    for a in RULE_MODELS:
        parts = []
        for sd in E27.SEEDS3:
            path = os.path.join(OOF_DIR, f"e41_{a}_{arm}_sh{shift}_s{sd}.parquet")
            if os.path.exists(path):
                parts.append(pd.read_parquet(path))
                continue
            t0 = time.time()
            o = oof_folds(a, df, cols, par[a], sd, folds)
            o.to_parquet(path, index=False)
            parts.append(o)
            log(f"    {a} ずらし{shift}か月 種 {sd}: {len(o):,}件 / {time.time()-t0:.0f}秒")
        out[a] = average(parts)
    return out


def classify(d: pd.DataFrame):
    """ラベルの外れ方。rise_thresholds はラベルを作った式そのもの。"""
    need, end_need = B.rise_thresholds(d["vol_20d"])
    need = need.to_numpy(dtype=float)
    end_need = end_need.to_numpy(dtype=float)
    reach = d["future_rise"].to_numpy(dtype=float) >= need
    endok = d["end_level"].to_numpy(dtype=float) >= end_need
    up = d["uptrend_end"].to_numpy(dtype=float) == 1.0
    y = d["label"].to_numpy(dtype=float)
    cls = np.select([y == 1, reach & ~endok, reach & ~up, ~reach],
                    list(CLASSES), default="判定不能")
    rebuilt = (reach & endok & up).astype(float)
    return cls, need, end_need, rebuilt


# ---------------------------------------------------------------------- #
# 値動きの表
# ---------------------------------------------------------------------- #

def load_bars(codes) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    cols = ["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC"]
    b = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b[b["Code"].isin(set(codes))]
    b = b.sort_values(["Code", "Date"]).reset_index(drop=True)
    if b.duplicated(["Code", "Date"]).any():
        raise SystemExit("bars に (Code, Date) の重複がある")
    return b


def forward(b: pd.DataFrame, keys: pd.DataFrame, k: int = K) -> dict:
    """
    keys の各行（基準日 t）について、1〜k 日目（t+1 〜 t+k）の四本値を
    (行, 日) の表にする。列 j が j+1 日目。先の足が無いところは NaN。
    """
    pos = b[["Code", "Date"]].assign(_i=np.arange(len(b)))
    i0 = keys[["Code", "Date"]].merge(pos, on=["Code", "Date"], how="left")["_i"]
    if i0.isna().any():
        raise SystemExit(f"bars に無い基準日が {int(i0.isna().sum())}件")
    i0 = i0.to_numpy(dtype=np.int64)
    left = b.groupby("Code", sort=False).cumcount(ascending=False).to_numpy()
    step = np.arange(1, k + 1)
    ok = step[None, :] <= left[i0][:, None]
    idx = np.where(ok, i0[:, None] + step[None, :], 0)
    out = {"ok": ok}
    for name, col in (("O", "AdjO"), ("H", "AdjH"), ("L", "AdjL"), ("C", "AdjC")):
        a = b[col].to_numpy(dtype=float)[idx]
        a[~ok] = np.nan
        out[name] = a
    e = out["O"][:, 0]
    out["entry"] = e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for h in HOLDS:
            # h日目の5日平均終値（t+h-4 〜 t+h）。lab.realized_returns の
            # rolling(5, min_periods=1) と同じく欠測は飛ばして平均する
            x = np.nanmean(out["C"][:, h - 5:h], axis=1)
            out[f"x{h}"] = np.where(ok[:, h - 1], x / e - 1.0, np.nan)
    return out


def excursions(P: dict, n: int) -> pd.DataFrame:
    """1〜n 日目の最大上昇・最大下落とその日、高値をつけるまでの最大下落。"""
    e = P["entry"]
    H, L, C = P["H"][:, :n], P["L"][:, :n], P["C"][:, :n]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hmax = np.nanmax(H, axis=1)
        lmin = np.nanmin(L, axis=1)
        cmax = np.nanmax(C, axis=1)
        cmin = np.nanmin(C, axis=1)
        dmax = np.where(np.isnan(H), -np.inf, H).argmax(axis=1) + 1
        dmin = np.where(np.isnan(L), np.inf, L).argmin(axis=1) + 1
        before = np.where(np.arange(n)[None, :] < dmax[:, None], L, np.nan)
        pre = np.nanmin(before, axis=1)
    return pd.DataFrame({
        f"mfe{n}": hmax / e - 1.0, f"mae{n}": lmin / e - 1.0,
        f"cmfe{n}": cmax / e - 1.0, f"cmae{n}": cmin / e - 1.0,
        f"day_mfe{n}": dmax.astype(float), f"day_mae{n}": dmin.astype(float),
        f"pre_peak_dd{n}": pre / e - 1.0,
    })


def touch_day(P: dict, x: float, n: int = 20) -> np.ndarray:
    """安値が 買値×(1−x) に最初に触れた日（触れなければ NaN）。"""
    L = P["L"][:, :n]
    hit = L <= (P["entry"] * (1.0 - x))[:, None]
    first = hit.argmax(axis=1) + 1.0
    return np.where(hit.any(axis=1), first, np.nan)


# ---------------------------------------------------------------------- #
# 損切り・利確の模擬
# ---------------------------------------------------------------------- #

def simulate(P: dict, hold: int, stop=None, tp=None, mode: str = "intraday",
             tstop=None):
    """
    stop:  損切り幅（0.08 = −8%）。行ごとの配列でもよい（σ 基準）
    mode:  "intraday" 逆指値 / "close" 終値で判定して次の寄りで売る
    tstop: (k, θ) 日数で切る。k日目の終値が 買値×(1+θ) を下回っていたら、
           次に寄りが付いた日の寄値で売る
    戻り値: 収益, 手仕舞った日, 理由（0 期間満了 / 1 損切り / 2 利確 / 3 日数で切った）,
            同じ日に損切りと利確の両方に触れた件数
    """
    e = P["entry"]
    O, H, L, C = (P[c][:, :hold] for c in "OHLC")
    m = len(e)
    S = e * (1.0 - np.asarray(stop, dtype=float)) if stop is not None else None
    T = e * (1.0 + tp) if tp is not None else None
    ret = np.full(m, np.nan)
    day = np.full(m, np.nan)
    why = np.zeros(m, dtype=np.int8)
    done = np.zeros(m, dtype=bool)
    pend = np.zeros(m, dtype=bool)
    tpend = np.zeros(m, dtype=bool)
    none = np.zeros(m, dtype=bool)
    both = 0
    with np.errstate(invalid="ignore"):
        for k in range(hold):
            o, h, lo, c = O[:, k], H[:, k], L[:, k], C[:, k]
            if tstop is not None:
                ex = ~done & tpend & np.isfinite(o)
                ret[ex] = o[ex] / e[ex] - 1.0
                day[ex] = k + 1
                why[ex] = 3
                done |= ex
            if S is not None:
                if mode == "close":
                    ex = ~done & pend & np.isfinite(o)          # 前日までに終値で割った
                else:
                    ex = ~done & (o <= S)                       # 窓を開けて割った（1日目は o=買値なので起きない）
                ret[ex] = o[ex] / e[ex] - 1.0
                day[ex] = k + 1
                why[ex] = 1
                done |= ex
            hs = (~done & (lo <= S)) if (S is not None and mode == "intraday") else none
            ht = (~done & (h >= T)) if T is not None else none
            both += int((hs & ht).sum())
            if hs.any():
                ret[hs] = S[hs] / e[hs] - 1.0
                day[hs] = k + 1
                why[hs] = 1
            ht = ht & ~hs
            if ht.any():
                ret[ht] = tp
                day[ht] = k + 1
                why[ht] = 2
            done |= hs | ht
            if S is not None and mode == "close":
                pend = ~done & (pend | (c <= S))
            if tstop is not None and k + 1 == tstop[0]:
                tpend = ~done & (c < e * (1.0 + tstop[1]))
    rest = ~done
    ret[rest] = P[f"x{hold}"][rest]
    day[rest] = hold
    return ret, day, why, both


def fold_diff(ret, base, fold) -> tuple:
    """窓ごとの平均の差（損切りあり − なし）。勝ち窓 / 窓数 / 最悪の窓。"""
    t = pd.DataFrame({"r": ret, "b": base, "f": fold}).dropna()
    g = t.groupby("f")[["r", "b"]].mean()
    diff = (g["r"] - g["b"]).to_numpy()
    if not len(diff):
        return 0, 0, float("nan")
    return int((diff > 1e-12).sum()), int(len(diff)), float(diff.min())


def summarize(name: str, ret, day, why, both, base, y, fold) -> dict:
    stopped = (why == 1) | (why == 3)            # 値段で切った / 日数で切った
    won, nf, worst_f = fold_diff(ret, base, fold)
    return {
        "rule": name, "n": int(len(ret)),
        "stop_rate": float(stopped.mean()), "tp_rate": float((why == 2).mean()),
        "stop_ret": mean(ret[stopped]) if stopped.any() else float("nan"),
        "mean": mean(ret), "diff": mean(ret) - mean(base), "median": med(ret),
        "win": win(ret), "p10": qt(ret, 10), "worst": float(np.nanmin(ret)),
        "sd": float(np.nanstd(ret, ddof=1)), "days": mean(day),
        "per_day": mean(ret) / mean(day),
        "pos_cut": float(stopped[y == 1].mean()) if (y == 1).any() else float("nan"),
        # 損切りした銘柄のうち、持ち切ったほうが良かった割合（裏目）と、その平均の効果
        "regret": float((base[stopped] > ret[stopped]).mean()) if stopped.any() else float("nan"),
        "effect_stopped": mean(ret[stopped] - base[stopped]) if stopped.any() else float("nan"),
        "won": won, "n_folds": nf, "worst_fold": worst_f, "both": both,
    }


SWEEP_HEAD = (f"  {'規則':<22}{'損切り':>7}{'切った時':>8}{'利確':>6}{'平均':>8}{'差':>8}{'中央値':>8}{'勝率':>6}"
              f"{'下位10%':>8}{'最悪':>8}{'日数':>6}{'1日':>8}{'正例切り':>8}{'裏目':>6}"
              f"{'効果':>8}{'勝ち窓':>7}{'最悪窓':>8}")


def sweep_line(s: dict) -> str:
    return (f"  {s['rule']:<22}{s['stop_rate']*100:>6.1f}%{pc(s['stop_ret'])}{s['tp_rate']*100:>5.1f}%"
            f"{pc(s['mean'])}{pc(s['diff'])}{pc(s['median'])}{s['win']*100:>5.0f}%"
            f"{pc(s['p10'])}{pc(s['worst'], d=1)}{s['days']:>6.1f}{pc(s['per_day'], d=3)}"
            f"{s['pos_cut']*100:>7.1f}%{s['regret']*100 if np.isfinite(s['regret']) else float('nan'):>5.0f}%"
            f"{pc(s['effect_stopped'])}{s['won']:>4}/{s['n_folds']:<2}{pc(s['worst_fold'])}")


# ---------------------------------------------------------------------- #
# 大負けの共通点（§9）
# ---------------------------------------------------------------------- #

def date_constant(frame: pd.DataFrame, cols: list) -> set:
    """
    同じ日ならどの銘柄でも同じ値の列（銘柄ではなく時期を表す列）。
    すべての日で1種類の値しか無いこと。まれにしか立たない旗（ほとんどの日で
    全銘柄 0）を拾わないよう、「ほとんどの日」ではなく「すべての日」で見る。
    """
    g = frame.groupby("Date")
    multi = g.size() >= 2
    out = set()
    for c in cols:
        nu = g[c].nunique(dropna=True)[multi]
        if len(nu) and bool((nu <= 1).all()) and frame[c].nunique(dropna=True) > 1:
            out.add(c)
    return out


def loser_screen(u: pd.DataFrame, feats: list, big: float = BIG) -> pd.DataFrame:
    """
    母集団で、列の上位10% / 下位10% の大負け率が、その窓の大負け率より
    どれだけ高いかを窓ごとに測り、窓をまたいだ 平均 ÷ 標準誤差 を z とする。
    （実験23・40 の両側スクリーニングと同じ形。物差しを「大負けしたか」にしただけ）
    """
    loss = (u["ret_o1_20"].to_numpy(dtype=float) <= big).astype(float)
    fo = u["fold"].to_numpy()
    rows = []
    for f in feats:
        x = pd.to_numeric(u[f], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 1000 or np.nanstd(x[ok]) == 0:
            continue
        ex = {"top": [], "bot": []}
        for k in np.unique(fo):
            w = ok & (fo == k)
            if w.sum() < 200:
                continue
            xw, lw = x[w], loss[w]
            hi, lo = np.percentile(xw, 90), np.percentile(xw, 10)
            for side, m in (("top", xw >= hi), ("bot", xw <= lo)):
                # 同じ値が多い列で「上位10%」が半分を超えるなら、上位とは言えない
                if 20 <= m.sum() <= 0.5 * w.sum():
                    ex[side].append(lw[m].mean() - lw.mean())
        rec = {"feature": f, "coverage": float(ok.mean())}
        for side, arr in ex.items():
            arr = np.asarray(arr)
            if len(arr) >= 5 and arr.std(ddof=1) > 0:
                rec[f"{side}_pt"] = float(arr.mean() * 100)
                rec[f"{side}_z"] = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))
                rec[f"{side}_pos"] = int((arr > 0).sum())
                rec[f"{side}_n"] = int(len(arr))
        rows.append(rec)
    return pd.DataFrame(rows)


def exclusion(s: pd.DataFrame, ref: pd.DataFrame, f: str, side: str) -> np.ndarray:
    """
    選んだ銘柄のうち、列 f が「それより前の窓の母集団」の上位10%（または
    下位10%）に入るものを True にする。しきい値に先の値は使わない。
    値が無い行は外さない。
    """
    out = np.zeros(len(s), dtype=bool)
    x = pd.to_numeric(s[f], errors="coerce").to_numpy(dtype=float)
    fo = s["fold"].to_numpy()
    for k in np.unique(fo):
        prev = pd.to_numeric(ref.loc[ref["fold"] < k, f], errors="coerce").dropna()
        if len(prev) < 500:
            continue
        m = fo == k
        if side == "top":
            out[m] = x[m] >= np.percentile(prev, 90)
        else:
            out[m] = x[m] <= np.percentile(prev, 10)
    return out


def topix_forward(h: int = 20) -> pd.Series:
    """日付 -> その日の終値から h 営業日後の終値までの TOPIX の騰落。"""
    paths = sorted(glob.glob(os.path.join(lab.DATA_DIR, "topix_*.parquet")))
    if not paths:
        return pd.Series(dtype=float)
    t = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    t["Date"] = pd.to_datetime(t["Date"])
    t = t.drop_duplicates("Date").sort_values("Date").set_index("Date")["topix"].astype(float)
    return t.shift(-h) / t - 1.0


def loser_section(s: pd.DataFrame, u: pd.DataFrame, ref: pd.DataFrame, feats: list) -> None:
    import feature_dict as FD

    def say(f: str) -> str:
        if f == "entry_gap":
            return "買った日の寄りの窓（前日終値比。寄りの瞬間に分かる）"
        return FD.describe(f)

    lose = s["ret_o1_20"] <= BIG
    print(f"\n=== 9. 大負けの共通点（20営業日保有で {BIG*100:.0f}% 以下）===")
    print(f"  選んだ銘柄 {len(s):,}件のうち 大負け {int(lose.sum())}件（{lose.mean()*100:.1f}%）"
          f" / −15%以下 {int((s['ret_o1_20'] <= -0.15).sum())}件"
          f" / −20%以下 {int((s['ret_o1_20'] <= -0.20).sum())}件"
          f"。母集団では {(u['ret_o1_20'] <= BIG).mean()*100:.1f}%")

    print("\n  9a. 時期に固まっているか（相場全体の下げと重なっていないか）")
    tf = topix_forward(20)
    tx = s["Date"].map(tf)
    down = tx <= -0.05
    print(f"    同じ20営業日に TOPIX が −5% 以上下げていた割合: 大負け {down[lose].mean()*100:.1f}%"
          f" / 選んだ銘柄全体 {down.mean()*100:.1f}%（TOPIX の値がある行で）")
    print(f"    同じ期間の TOPIX の騰落（中央値）: 大負け {pc(med(tx[lose]), 0)}"
          f" / 大負け以外 {pc(med(tx[~lose]), 0)}")
    ym = pd.to_datetime(s["Date"]).dt.to_period("M")
    cnt = ym[lose].value_counts()
    top5 = cnt.head(5)
    print(f"    大負けの多い月: " + " / ".join(f"{k} {v}件" for k, v in top5.items())
          + f"（上位5か月で {top5.sum()/max(lose.sum(), 1)*100:.0f}%。大負けがあった月は {len(cnt)}か月）")
    same_day = s[lose].groupby("Date").size()
    print(f"    同じ日に買った銘柄が2件以上そろって大負け: {int(same_day[same_day >= 2].sum())}件"
          f"（{len(same_day[same_day >= 2])}日）")

    print(f"\n  9b. 母集団（{len(u):,}件・窓ごと）で、大負けが多い側（|z| > {SCREEN_Z:.0f}）")
    print("      上位/下位 = その列の上位10% / 下位10%。差 = その側の大負け率 − 窓全体の大負け率（窓平均）")
    scr = loser_screen(u, feats)
    const = date_constant(u, feats)
    hits = []
    for _, r in scr.iterrows():
        for side in ("top", "bot"):
            z = r.get(f"{side}_z", np.nan)
            if np.isfinite(z) and z > SCREEN_Z:
                hits.append((r["feature"], side, float(z), float(r[f"{side}_pt"]),
                             int(r[f"{side}_pos"]), int(r[f"{side}_n"])))
    hits.sort(key=lambda h: -h[2])
    vol = pd.to_numeric(u["vol_20d"], errors="coerce")
    n_const = sum(1 for h in hits if h[0] in const)
    print(f"  該当 {len(hits)}（列×側）。うち日内一定の列 {n_const}。z の大きい順に上位20を出す"
          "（全件は e41_loser_screen.csv）")
    print(f"  {'列':<24}{'側':>4}{'差':>8}{'z':>7}{'窓':>7}{'ボラ相関':>8}  {'選んだ銘柄の中で':<22}  説明")
    for f, side, z, pt, pos, n in hits[:20]:
        m = exclusion(s, ref, f, side)
        tag = "［日内一定］" if f in const else ""
        lr = s.loc[m, "ret_o1_20"]
        inner = (f"{int(m.sum()):>4}件 大負け率 {(lr <= BIG).mean()*100 if len(lr) else float('nan'):>5.1f}%"
                 if m.sum() else "   該当なし")
        rho = pd.to_numeric(u[f], errors="coerce").corr(vol, method="spearman")
        print(f"  {f:<24}{'上位' if side == 'top' else '下位':>4}{pt:>+7.1f}pt{z:>7.1f}{pos:>4}/{n:<2}"
              f"{rho:>+8.2f}  {inner:<22}  {tag}{say(f)[:30]}")
    if not hits:
        print("    （該当なし）")
    print("  ボラ相関 = 日次ボラ（vol_20d）との順位相関。大きいものは「値動きが荒い」の言い換え")
    pd.DataFrame(scr).to_csv(os.path.join(OOF_DIR, "e41_loser_screen.csv"), index=False)

    stock = [(f, side) for f, side, _, _, _, _ in hits if f not in const][:6]
    print("\n  9c. 銘柄ごとの列で、候補から外したら（しきい値はそれより前の窓の母集団の上位/下位10%）")
    print(f"  {'外す条件':<30}{'外す':>6}{'大負け':>10}{'正例':>10}{'外した分の20日':>14}"
          f"{'残りの20日':>11}{'差':>8}{'下位10%':>9}{'最悪':>8}{'勝ち窓':>7}")
    base = s["ret_o1_20"].to_numpy(dtype=float)
    y = s["label"].to_numpy(dtype=float)
    fo = s["fold"].to_numpy()
    lb = lose.to_numpy()

    def line(name: str, m: np.ndarray) -> None:
        keep = ~m
        # 窓ごとに「外したあとの平均 − 外さない平均」
        won = nf = 0
        for k in np.unique(fo):
            w = fo == k
            if (w & keep).sum():
                nf += 1
                won += int(np.nanmean(base[w & keep]) - np.nanmean(base[w]) > 1e-12)
        print(f"  {name:<30}{int(m.sum()):>5}件"
              f"{int((m & lb).sum()):>5}/{int(lb.sum()):<4}{int((m & (y == 1)).sum()):>5}/{int((y == 1).sum()):<4}"
              f"{pc(mean(base[m]), 14)}{pc(mean(base[keep]), 11)}{pc(mean(base[keep]) - mean(base))}"
              f"{pc(qt(base[keep], 10), 9)}{pc(float(np.nanmin(base[keep])), 8, 1)}{won:>4}/{nf:<2}")

    print(f"  {'（外さない）':<30}{0:>5}件{0:>5}/{int(lb.sum()):<4}{0:>5}/{int((y == 1).sum()):<4}"
          f"{'-':>14}{pc(mean(base), 11)}{pc(0.0)}{pc(qt(base, 10), 9)}{pc(float(np.nanmin(base)), 8, 1)}")
    masks = []
    for f, side in stock:
        m = exclusion(s, ref, f, side)
        masks.append(m)
        line(f"{f} {'上位' if side == 'top' else '下位'}10%", m)
    if len(masks) >= 2:
        line("上の2つのどちらか", masks[0] | masks[1])
    if len(masks) >= 3:
        line("上の3つのどれか", masks[0] | masks[1] | masks[2])
    # 検定の結果にかかわらず、よく使われる2つの規則も測る
    print("  --- よく使われる規則（スクリーニングとは別に、決め打ちで測る）---")
    line("寄りの窓が上位10%（飛びつかない）", exclusion(s, ref, "entry_gap", "top"))
    dte = pd.to_numeric(s.get("days_to_earn"), errors="coerce").to_numpy(dtype=float)
    line("保有期間中に決算予定（28暦日以内）", np.isfinite(dte) & (dte <= 28))
    print("  大負け = 外した中の大負けの数 / 全体の大負け。正例も同じ（外すと当たりも一緒に失う）。")
    print("  勝ち窓 = 窓ごとの平均が「外さない」を上回った窓の数。")
    print("  決算予定は、その日までに公表済みの予定があるものだけ（無い行は外さない）。")

    print("\n  9d. 大負けとそれ以外の中央値（選んだ銘柄。9c の列）")
    print(f"  {'列':<24}{'大負け':>12}{'それ以外':>12}  説明")
    for f, side in stock:
        x = pd.to_numeric(s[f], errors="coerce")
        print(f"  {f:<24}{med(x[lose]):>12.4g}{med(x[~lose]):>12.4g}  {say(f)[:40]}")


# ---------------------------------------------------------------------- #
# 窓ごと・切り方ごとの損切りの効果（§10）
# ---------------------------------------------------------------------- #

#: §10 で窓ごとに並べる規則（20営業日保有）。stop="1s" はラベルと同じ σ の1倍
RULES10 = (
    ("−5%", {"stop": 0.05}), ("−8%", {"stop": 0.08}), ("−10%", {"stop": 0.10}),
    ("−15%", {"stop": 0.15}), ("−20%", {"stop": 0.20}), ("−1σ", {"stop": "1s"}),
    ("終値−10%", {"stop": 0.10, "mode": "close"}), ("利確20%", {"tp": TP}),
    ("−10%+利確", {"stop": 0.10, "tp": TP}), ("5日目<0", {"tstop": (5, 0.0)}),
)


def pattern_rows(oofs_k: dict, df: pd.DataFrame, bars: pd.DataFrame):
    """切り方ごとの選定（3モデル 90以上）と、その値動きの表（20日先まで揃う行だけ）。"""
    r = OR.consensus(oofs_k, RULE_PCT, models=RULE_MODELS, keep=True)
    sel = r["rows"].copy()
    folds = sorted(int(f) for f in sel["fold"].unique())
    pool = oofs_k[RULE_MODELS[0]]
    nb = pool[pool["fold"].isin(folds)].groupby("Date").size()
    sel = sel.merge(df[["Code", "Date", "vol_20d"]], on=["Code", "Date"], how="left")
    sel["n_break"] = sel["Date"].map(nb).to_numpy()
    sel["sigma20"] = sel["vol_20d"] / 100.0 * np.sqrt(B.RISE_HORIZON)
    P = forward(bars, sel)
    fin = np.isfinite(P["x20"]) & np.isfinite(P["entry"])
    P = {k: (v[fin] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    return sel[fin].reset_index(drop=True), P, r


def rule_returns(P: dict, sig: np.ndarray) -> dict:
    """RULES10 の各規則の収益（行ごと）。規則は行ごとに独立なので、絞り込みは後から行を選べばよい。"""
    out = {"なし": P["x20"]}
    for name, kw in RULES10:
        kw = dict(kw)
        if isinstance(kw.get("stop"), str):
            kw["stop"] = sig
        out[name], _, _, _ = simulate(P, 20, **kw)
    return out


def window_table(title: str, fold: np.ndarray, rets: dict, ranges: dict) -> dict:
    """窓ごとに「規則の平均 − 損切りなしの平均」（pt）を並べる。戻り値は規則ごとの窓の差。"""
    base = rets["なし"]
    print(f"\n  [{title}] {len(base):,}件 / 窓 {len(np.unique(fold))}")
    print(f"  {'窓':>3} {'期間':<22}{'件数':>5}{'損切りなし':>9}{'最悪':>8} |"
          + "".join(f"{n:>9}" for n, _ in RULES10))
    agg = {n: [] for n, _ in RULES10}
    for k in sorted(np.unique(fold)):
        m = fold == k
        row = (f"  {int(k):>3} {ranges.get(int(k), ''):<22}{int(m.sum()):>5}{pc(mean(base[m]), 9)}"
               f"{pc(float(np.nanmin(base[m])), 8, 1)} |")
        for n, _ in RULES10:
            dl = mean(rets[n][m]) - mean(base[m])
            agg[n].append(dl)
            row += f"{dl*100:>+9.2f}"
        print(row)
    print(f"  {'':>3} {'全体':<22}{len(base):>5}{pc(mean(base), 9)}{pc(float(np.nanmin(base)), 8, 1)} |"
          + "".join(f"{(mean(rets[n]) - mean(base))*100:>+9.2f}" for n, _ in RULES10))
    print(f"  {'':>3} {'勝ち窓':<22}{'':>22} |"
          + "".join(f"{sum(x > 1e-12 for x in agg[n]):>6}/{len(agg[n]):<2}" for n, _ in RULES10))
    print(f"  {'':>3} {'最悪の窓':<22}{'':>22} |"
          + "".join(f"{min(agg[n])*100:>+9.2f}" for n, _ in RULES10))
    print(f"  {'':>3} {'最良の窓':<22}{'':>22} |"
          + "".join(f"{max(agg[n])*100:>+9.2f}" for n, _ in RULES10))
    return agg


# ---------------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scores", choices=("e39", "e27"), default="e39",
                    help="e39: 本番と同じ205列のスコア（既定） / e27: 手元の動作確認用の旧スコア")
    ap.add_argument("--arm", choices=("B2", "B1"), default="B2",
                    help="B2: 205列で探索したパラメータ（本番の形・既定） / B1: 本番のパラメータのまま")
    ap.add_argument("--shifts", default=",".join(str(k) for k in SHIFTS),
                    help="§10 で窓の境界をずらす月数（カンマ区切り）。0 は本番と同じ窓")
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    oofs = load_scores(args.scores, df, args.arm)
    for a in oofs:
        oofs[a]["Date"] = pd.to_datetime(oofs[a]["Date"])

    r = OR.consensus(oofs, RULE_PCT, models=RULE_MODELS, keep=True)
    if not r.get("n"):
        raise SystemExit("選定が0件")
    sel = r["rows"]
    folds = sorted(int(f) for f in sel["fold"].unique())
    pool = oofs[RULE_MODELS[0]]
    pool = pool[pool["fold"].isin(folds)][["Code", "Date", "fold"]].copy()
    pool = pool.merge(pool.groupby("Date").size().rename("n_break"), on="Date")
    pool["selected"] = pool.set_index(["Code", "Date"]).index.isin(
        sel.set_index(["Code", "Date"]).index)

    # ラベルの部品と実収益は今のデータセットから引く（out-of-fold と同じ行）
    extra = ["label", "vol_20d", "future_rise", "end_level", "uptrend_end",
             "entry_gap"] + [f"ret_o1_{h}" for h in lab.HORIZONS]
    d = pool.merge(df[["Code", "Date"] + extra], on=["Code", "Date"], how="left")
    d = d.merge(sel[["Code", "Date"] + [f"s_{a}" for a in RULE_MODELS]],
                on=["Code", "Date"], how="left")
    # 3モデルの最小順位（PLAYBOOK の並べ方。実験32 と同じく母集団の中の順位）
    for a in RULE_MODELS:
        sc = oofs[a][["Code", "Date", "score"]].rename(columns={"score": f"p_{a}"})
        d = d.merge(sc, on=["Code", "Date"], how="left")
        d[f"p_{a}"] = d[f"p_{a}"].rank(pct=True)
    d["p_min"] = d[[f"p_{a}" for a in RULE_MODELS]].min(axis=1)
    cls, need, end_need, rebuilt = classify(d)
    d["cls"] = cls
    d["need"] = need
    d["sigma20"] = d["vol_20d"] / 100.0 * np.sqrt(B.RISE_HORIZON)

    log("値動きの表を作る")
    bars = load_bars(d["Code"].unique())
    P_all = forward(bars, d)
    ex20 = excursions(P_all, 20)
    ex60 = excursions(P_all, 60)
    for c in ex20.columns:
        d[c] = ex20[c].to_numpy()
    for c in ex60.columns:
        d[c] = ex60[c].to_numpy()
    for h in HOLDS:
        d[f"x{h}"] = P_all[f"x{h}"]
    d["entry"] = P_all["entry"]
    for k in DAYS:
        d[f"c{k}"] = P_all["C"][:, k - 1] / P_all["entry"] - 1.0

    # ------------------------------------------------------------------ #
    print("\n=== 0. 前提の確認 ===")
    print(f"  スコア: {'実験27 の旧スコア（動作確認用）' if args.scores == 'e27' else '腕 ' + args.arm}")
    print(f"  基準 {RULE}: 選定 {r['n']:,}件（評価した窓の {r['rate']*100:.1f}%）/ "
          f"正例率 {r['label_rate']*100:.1f}% / ret_o1_20 平均 {r['ret20']:+.2f}%")
    print(f"  評価した窓 {folds[0]}〜{folds[-1]}（{len(folds)}窓。窓1は比べる過去が無いので除く）/ "
          f"母集団 {len(d):,}件 / 期間 {d['Date'].min().date()}〜{d['Date'].max().date()}")
    y = d["label"].to_numpy(dtype=float)
    det = np.isfinite(y)
    mis = int((rebuilt[det] != y[det]).sum())
    print(f"  ラベルを部品（到達・終盤・トレンド）から組み直した結果と食い違う行: {mis}件"
          f" / 判定不能に落ちた行: {int((d['cls'] == '判定不能').sum())}件")
    xr = d["x20"] - d["ret_o1_20"]
    both_fin = d["x20"].notna() & d["ret_o1_20"].notna()
    print(f"  値動きの表から作った20日保有と ret_o1_20 の差: 最大 {np.abs(xr[both_fin]).max():.2e}"
          f"（{int(both_fin.sum()):,}件）/ 片方だけ欠測 {int((d['x20'].notna() ^ d['ret_o1_20'].notna()).sum())}件")
    print(f"  買値（翌日の寄り）が付かない行: {int(d['entry'].isna().sum())}件（以下の集計から外す）")

    s = d[d["selected"] & d["entry"].notna()].reset_index(drop=True)
    # その日の選定の中での順位（並べ替えはしない。値動きの表と行を揃えたままにする）
    s["rank_day"] = s.groupby("Date")["p_min"].rank(ascending=False, method="first")
    u = d[d["entry"].notna()].reset_index(drop=True)
    groups = [("母集団（全ブレイク）", u), ("3モデル 90以上（全体）", s)]
    groups += [(f"  {c}", s[s["cls"] == c]) for c in CLASSES]
    groups += [("  正例以外（まとめ）", s[s["cls"] != "正例"])]

    # ------------------------------------------------------------------ #
    print(f"\n=== 1. ラベルの外れ方（選んだ {len(s):,}件の内訳）===")
    print("  ラベルは基準日の終値から、収益は翌日の寄り（買値）から測る。")
    print(f"  {'区分':<22}{'件数':>6}{'構成比':>7}{'必要上昇':>9}{'最大上昇(終値)':>12}"
          f"{'寄りの窓':>9}{'日次ボラ':>8}{'20日':>9}{'中央値':>9}{'勝率':>6}{'40日':>9}{'60日':>9}")
    for name, g in groups:
        share = f"{'-':>7}" if g is u else f"{len(g)/max(len(s), 1)*100:>6.1f}%"
        print(f"  {name:<22}{len(g):>6}{share}"
              f"{pc(med(g['need']), 9, 1, False)}{pc(med(g['future_rise']), 12, 1)}"
              f"{pc(med(g['entry_gap']), 9)}{med(g['vol_20d']):>7.2f}%"
              f"{pc(mean(g['ret_o1_20']), 9)}{pc(med(g['ret_o1_20']), 9)}{win(g['ret_o1_20'])*100:>5.0f}%"
              f"{pc(mean(g['ret_o1_40']), 9)}{pc(mean(g['ret_o1_60']), 9)}")
    print("  必要上昇・最大上昇(終値)・寄りの窓・日次ボラは中央値。20/40/60日は平均（ret_o1_N）。")
    print(f"  ※ 60日は先の足が揃っている行だけ（{int(s['ret_o1_60'].notna().sum()):,}件）")

    # ------------------------------------------------------------------ #
    print("\n=== 2a. 最大上昇率（買値から・20営業日以内の高値）===")
    print(f"  {'区分':<22}{'件数':>6}{'平均':>8}{'25%':>8}{'中央値':>8}{'75%':>8}{'90%':>8}"
          f"{'その日':>7}{'終値ベース':>10}{'+10%到達':>9}{'+20%到達':>9}")
    for name, g in groups:
        print(f"  {name:<22}{len(g):>6}{pc(mean(g['mfe20']))}{pc(qt(g['mfe20'], 25))}"
              f"{pc(med(g['mfe20']))}{pc(qt(g['mfe20'], 75))}{pc(qt(g['mfe20'], 90))}"
              f"{num(med(g['day_mfe20']), 6)}日{pc(med(g['cmfe20']), 10)}"
              f"{(g['mfe20'] >= 0.10).mean()*100:>8.1f}%{(g['mfe20'] >= 0.20).mean()*100:>8.1f}%")
    print("  その日 = 最大上昇をつけた日（中央値。1日目＝買った日）。終値ベースは中央値。")

    print("\n=== 2b. 最大下落率（買値から・20営業日以内の安値）===")
    print(f"  {'区分':<22}{'件数':>6}{'平均':>8}{'10%':>8}{'25%':>8}{'中央値':>8}{'その日':>7}"
          f"{'終値ベース':>10}{'σ単位(中央)':>11}{'高値前の押し':>11}{'その10%':>9}")
    for name, g in groups:
        sig = g["mae20"] / g["sigma20"]
        print(f"  {name:<22}{len(g):>6}{pc(mean(g['mae20']))}{pc(qt(g['mae20'], 10))}"
              f"{pc(qt(g['mae20'], 25))}{pc(med(g['mae20']))}{num(med(g['day_mae20']), 6)}日"
              f"{pc(med(g['cmae20']), 10)}{num(med(sig), 10, 2)}σ"
              f"{pc(med(g['pre_peak_dd20']), 11)}{pc(qt(g['pre_peak_dd20'], 10), 9)}")
    print("  σ単位 = 最大下落 ÷ ラベルと同じ σ（日次ボラ×√20）。高値前の押し = 最大上昇をつける"
          "までに付けた最安値（中央値と下位10%）。")

    print("\n=== 2c. 20営業日以内に安値が −X% に触れた割合 ===")
    print(f"  {'区分':<22}{'件数':>6}" + "".join(f"{'−'+str(x)+'%':>8}" for x in TOUCH))
    for name, g in groups:
        print(f"  {name:<22}{len(g):>6}"
              + "".join(f"{(g['mae20'] <= -x/100).mean()*100:>7.1f}%" for x in TOUCH))

    print("\n=== 2d. 60営業日まで見たとき ===")
    print(f"  {'区分':<22}{'件数':>6}{'最大上昇':>9}{'その日':>7}{'最大下落':>9}{'その日':>7}"
          f"{'+20%到達':>9}{'−10%接触':>9}{'−20%接触':>9}")
    for name, g in groups:
        g = g[g["x60"].notna()]
        print(f"  {name:<22}{len(g):>6}{pc(med(g['mfe60']), 9)}{num(med(g['day_mfe60']), 6)}日"
              f"{pc(med(g['mae60']), 9)}{num(med(g['day_mae60']), 6)}日"
              f"{(g['mfe60'] >= 0.20).mean()*100:>8.1f}%{(g['mae60'] <= -0.10).mean()*100:>8.1f}%"
              f"{(g['mae60'] <= -0.20).mean()*100:>8.1f}%")
    print("  中央値。60日先の足が揃っている行だけ。")

    print("\n=== 2e. 途中経過（買値からの終値の騰落。平均 / 中央値）===")
    print(f"  {'区分':<22}" + "".join(f"{str(k)+'日目':>15}" for k in DAYS))
    for name, g in groups:
        print(f"  {name:<22}" + "".join(f"{pc(mean(g[f'c{k}']), 8, 1)}/{pc(med(g[f'c{k}']), 6, 1)}"
                                        for k in DAYS))

    print("\n=== 2f. ボラ（日次）の三分位ごと（選んだ銘柄の中で切る）===")
    s["vol_bin"] = pd.qcut(s["vol_20d"], 3, labels=["低", "中", "高"])
    print(f"  {'ボラ':<6}{'範囲':>14}{'件数':>6}{'正例率':>7}{'最大上昇':>9}{'最大下落':>9}"
          f"{'正例の下落':>10}{'正例以外の下落':>13}{'20日':>9}{'1σ(20日)':>10}")
    for b_, g in s.groupby("vol_bin", observed=True):
        pos = g[g["cls"] == "正例"]
        neg = g[g["cls"] != "正例"]
        print(f"  {b_:<6}{g['vol_20d'].min():>6.2f}〜{g['vol_20d'].max():>5.2f}%{len(g):>6}"
              f"{(g['cls'] == '正例').mean()*100:>6.1f}%{pc(med(g['mfe20']), 9)}{pc(med(g['mae20']), 9)}"
              f"{pc(med(pos['mae20']), 10)}{pc(med(neg['mae20']), 13)}{pc(mean(g['ret_o1_20']), 9)}"
              f"{pc(med(g['sigma20']), 10, 1, False)}")
    print("  最大上昇・最大下落は20営業日以内の中央値。")

    # ------------------------------------------------------------------ #
    print("\n=== 3a. 20営業日を過ぎても持ったら（ret_o1_N。先の足が60日揃っている行）===")
    print(f"  {'区分':<22}{'件数':>6}" + "".join(f"{str(h)+'日 平均':>10}{'中央値':>8}{'勝率':>6}"
                                               for h in HOLDS)
          + f"{'追加20→40':>10}{'追加20→60':>10}")
    for name, g in groups:
        g = g[g["x60"].notna()]
        a40 = (1 + g["ret_o1_40"]) / (1 + g["ret_o1_20"]) - 1
        a60 = (1 + g["ret_o1_60"]) / (1 + g["ret_o1_20"]) - 1
        print(f"  {name:<22}{len(g):>6}"
              + "".join(f"{pc(mean(g[f'ret_o1_{h}']), 10)}{pc(med(g[f'ret_o1_{h}']), 8, 1)}"
                        f"{win(g[f'ret_o1_{h}'])*100:>5.0f}%" for h in HOLDS)
              + f"{pc(mean(a40), 10)}{pc(mean(a60), 10)}")
    print("  追加20→N = 20日目に売らずに N日目まで持ったぶんの騰落（平均）。")
    g120 = s[s["ret_o1_120"].notna()]
    print(f"  参考: 120日保有（先の足が揃う {len(g120):,}件）平均 {pc(mean(g120['ret_o1_120']), 0)}"
          f" / 中央値 {pc(med(g120['ret_o1_120']), 0)} / 勝率 {win(g120['ret_o1_120'])*100:.0f}%")

    print("\n=== 3b. 20日目の損益で分けたとき、その先を持つ価値 ===")
    print(f"  {'区分':<22}{'20日目':>8}{'件数':>6}{'20日':>9}{'追加20→40':>10}{'中央値':>8}{'勝率':>6}"
          f"{'追加20→60':>10}{'中央値':>8}{'勝率':>6}")
    for name, g in [("3モデル 90以上（全体）", s), ("  正例", s[s["cls"] == "正例"]),
                    ("  正例以外（まとめ）", s[s["cls"] != "正例"])]:
        g = g[g["x60"].notna()]
        for tag, m in (("プラス", g["ret_o1_20"] > 0), ("マイナス", g["ret_o1_20"] <= 0)):
            h = g[m]
            a40 = (1 + h["ret_o1_40"]) / (1 + h["ret_o1_20"]) - 1
            a60 = (1 + h["ret_o1_60"]) / (1 + h["ret_o1_20"]) - 1
            print(f"  {name:<22}{tag:>8}{len(h):>6}{pc(mean(h['ret_o1_20']), 9)}"
                  f"{pc(mean(a40), 10)}{pc(med(a40), 8)}{win(a40)*100:>5.0f}%"
                  f"{pc(mean(a60), 10)}{pc(med(a60), 8)}{win(a60)*100:>5.0f}%")

    # ------------------------------------------------------------------ #
    print("\n=== 4. 20営業日以内に −X% に触れた銘柄は、その後どうなったか（選んだ銘柄）===")
    P = forward(bars, s)
    print(f"  {'安値が触れた':<12}{'件数':>6}{'割合':>7}{'触れた日':>8}{'うち正例':>8}"
          f"{'買値に戻った':>11}{'20日プラス':>10}{'20日平均':>9}{'40日平均':>9}{'触れず20日':>11}")
    for x in TOUCH:
        td = touch_day(P, x / 100.0)
        hit = np.isfinite(td)
        g = s[hit]
        # 触れた日より後（その日を含まない）に高値が買値以上に戻ったか
        after = np.where(np.arange(20)[None, :] >= td[hit][:, None], P["H"][hit, :20], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            back = np.nanmax(after, axis=1) >= P["entry"][hit]
        print(f"  {'−'+str(x)+'%':<12}{int(hit.sum()):>6}{hit.mean()*100:>6.1f}%"
              f"{num(med(td[hit]), 7)}日{(g['cls'] == '正例').mean()*100:>7.1f}%"
              f"{back.mean()*100:>10.1f}%{(g['ret_o1_20'] > 0).mean()*100:>9.1f}%"
              f"{pc(mean(g['ret_o1_20']), 9)}{pc(mean(g['ret_o1_40']), 9)}"
              f"{pc(mean(s.loc[~hit, 'ret_o1_20']), 11)}")
    print("  買値に戻った = 触れた日の翌日以降、20日目までに高値が買値以上をつけた割合。")

    # ------------------------------------------------------------------ #
    fin = np.isfinite(P["x20"])
    P20 = {k: (v[fin] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    s20 = s[fin].reset_index(drop=True)
    base = P20["x20"]
    y20 = s20["label"].to_numpy(dtype=float)
    f20 = s20["fold"].to_numpy()
    sig = s20["sigma20"].to_numpy(dtype=float)
    rows = []
    print(f"\n=== 5. 損切りの掃引（20営業日保有・{len(s20):,}件）===")
    print("  損切り/利確 = 発動した割合。差 = 損切りなしとの平均の差。正例切り = 正例のうち損切りに"
          "掛かった割合。")
    print("  裏目 = 損切りした銘柄のうち、持ち切ったほうが良かった割合。効果 = 損切りした銘柄での"
          "平均の差（損切り − 持ち切り）。")
    print("  勝ち窓 = 窓ごとの平均が損切りなしを上回った窓の数。最悪窓 = 窓ごとの差の最小。")
    for tp in (None, TP):
        tag = "" if tp is None else "・利確+20%"
        print(f"\n  --- 損切り{tag} ---")
        print(SWEEP_HEAD)
        specs = [("なし", None, "intraday")]
        specs += [(f"逆指値 −{x}%", x / 100.0, "intraday") for x in STOPS]
        specs += [(f"逆指値 −{m}σ", m * sig, "intraday") for m in SIGMAS]
        specs += [(f"終値 −{x}%", x / 100.0, "close") for x in STOPS]
        specs += [(f"終値 −{m}σ", m * sig, "close") for m in SIGMAS]
        for name, stop, mode in specs:
            ret, day, why, both = simulate(P20, 20, stop=stop, tp=tp, mode=mode)
            st = summarize(name + tag, ret, day, why, both, base, y20, f20)
            st.update({"hold": 20, "mode": mode, "tp": tp or 0.0})
            rows.append(st)
            print(sweep_line(st))
        if tp is not None:
            nb = sum(r_["both"] for r_ in rows if r_["tp"] == tp)
            print(f"  （同じ日に損切りと利確の両方に触れた延べ件数 {nb}件。すべて損切り扱い）")
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e41_sweep.csv"), index=False)
    print(f"\n  σ の目安: 選んだ銘柄の 1σ(20日) 中央値 {med(sig)*100:.1f}%"
          f"（25% {qt(sig, 25)*100:.1f}% / 75% {qt(sig, 75)*100:.1f}%）")

    print("\n=== 5b. 日数で切る（k日目の終値が 買値+θ を下回っていたら、翌営業日の寄りで売る）===")
    print("  値段の損切りは掛けない（最後の3行だけ 逆指値 −10% を重ねる）。20営業日保有・利確なし。")
    print(SWEEP_HEAD)
    tspecs = [(f"{k}日目 {th*100:+.0f}%未満", (k, th), None)
              for k in TSTOP_DAYS for th in TSTOP_LEVELS]
    tspecs += [(f"{k}日目 {0:+.0f}%未満+逆指値−10%", (k, 0.0), 0.10) for k in TSTOP_DAYS]
    for name, ts, stop in tspecs:
        ret, day, why, both = simulate(P20, 20, stop=stop, tstop=ts)
        st = summarize(name, ret, day, why, both, base, y20, f20)
        st.update({"hold": 20, "mode": "time", "tp": 0.0})
        rows.append(st)
        print(sweep_line(st))
    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, "e41_sweep.csv"), index=False)
    print("  損切り・切った時 = 日数で切った（と逆指値に掛かった）割合と、そのときの平均収益。")

    # ------------------------------------------------------------------ #
    fin60 = np.isfinite(P["x60"])
    P60 = {k: (v[fin60] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    s60 = s[fin60].reset_index(drop=True)
    sig60 = s60["sigma20"].to_numpy(dtype=float)
    cand = [("なし", None), ("−5%", 0.05), ("−8%", 0.08), ("−10%", 0.10), ("−12%", 0.12),
            ("−15%", 0.15), ("−20%", 0.20), ("−1.0σ", 1.0 * sig60), ("−1.5σ", 1.5 * sig60)]
    combos = [(h, tp) for h in HOLDS for tp in (None, TP)]
    print(f"\n=== 6. 持ち期間 × 利確 × 損切り（逆指値。先の足が60日揃っている {len(s60):,}件で揃えて比べる）===")
    for metric, label in (("mean", "平均"), ("per_day", "1日あたり"), ("p10", "下位10%"),
                          ("win", "勝率")):
        print(f"\n  [{label}]")
        print(f"  {'損切り':<10}" + "".join(f"{str(h)+'日'+('・利確' if tp else ''):>12}"
                                          for h, tp in combos))
        for name, stop in cand:
            cells = []
            for h, tp in combos:
                ret, day, why, both = simulate(P60, h, stop=stop, tp=tp)
                v = {"mean": mean(ret), "per_day": mean(ret) / mean(day), "p10": qt(ret, 10),
                     "win": win(ret)}[metric]
                cells.append(pc(v, 12, 3 if metric == "per_day" else 2,
                                sign=metric != "win"))
            print(f"  {name:<10}" + "".join(cells))

    # ------------------------------------------------------------------ #
    print("\n=== 7. 年ごと（20営業日保有・逆指値。平均）===")
    s20["year"] = pd.to_datetime(s20["Date"]).dt.year
    yr_specs = [("なし", None, None), ("−8%", 0.08, None), ("−10%", 0.10, None),
                ("−1.0σ", 1.0 * sig, None), ("なし・利確", None, TP),
                ("−10%・利確", 0.10, TP), ("−1.0σ・利確", 1.0 * sig, TP)]
    yr = {}
    for name, stop, tp in yr_specs:
        ret, _, _, _ = simulate(P20, 20, stop=stop, tp=tp)
        yr[name] = ret
    print(f"  {'年':<6}{'件数':>6}{'正例率':>7}" + "".join(f"{n_:>12}" for n_, _, _ in yr_specs))
    for yv, idx in s20.groupby("year").groups.items():
        idx = np.asarray(list(idx))
        print(f"  {yv:<6}{len(idx):>6}{np.nanmean(y20[idx])*100:>6.1f}%"
              + "".join(pc(mean(yr[n_][idx]), 12) for n_, _, _ in yr_specs))

    # ------------------------------------------------------------------ #
    print("\n=== 8. 運用の絞り込み（PLAYBOOK: 発火20件以上・3モデルの最小順位で上位1〜2件）でも同じか ===")
    hot = (s20["n_break"] >= 20).to_numpy()
    rk = s20["rank_day"].to_numpy()
    subsets = [("全日・全件（5 と同じ）", np.ones(len(s20), dtype=bool)),
               ("発火20件以上・全件", hot),
               ("発火20件以上・上位2件", hot & (rk <= 2)),
               ("発火20件以上・上位1件", hot & (rk <= 1))]
    print(f"  {'絞り込み':<24}{'件数':>6}{'正例率':>7}{'未到達':>7}{'最大上昇':>9}{'最大下落':>9}"
          f"{'−5%接触':>8}{'−10%接触':>9}{'20日':>9}{'勝率':>6}")
    for name, m in subsets:
        g = s20[m]
        print(f"  {name:<24}{len(g):>6}{g['label'].mean()*100:>6.1f}%"
              f"{(g['cls'] == '未到達').mean()*100:>6.1f}%{pc(med(g['mfe20']), 9)}{pc(med(g['mae20']), 9)}"
              f"{(g['mae20'] <= -0.05).mean()*100:>7.1f}%{(g['mae20'] <= -0.10).mean()*100:>8.1f}%"
              f"{pc(mean(g['ret_o1_20']), 9)}{win(g['ret_o1_20'])*100:>5.0f}%")
    print("  最大上昇・最大下落は20営業日以内の中央値。")
    rules8 = [("なし", None, None, None), ("逆指値 −5%", 0.05, None, None),
              ("逆指値 −8%", 0.08, None, None), ("逆指値 −10%", 0.10, None, None),
              ("逆指値 −15%", 0.15, None, None), ("逆指値 −1.0σ", "1s", None, None),
              ("終値 −10%", 0.10, None, None), ("なし・利確+20%", None, TP, None),
              ("逆指値 −10%・利確+20%", 0.10, TP, None), ("5日目 +0%未満", None, None, (5, 0.0))]
    for name, m in subsets[1:]:
        print(f"\n  --- {name}（{int(m.sum())}件）---")
        print(SWEEP_HEAD)
        Pm = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in P20.items()}
        for rn, stop, tp, ts in rules8:
            st_ = sig[m] if isinstance(stop, str) else stop
            mode = "close" if rn.startswith("終値") else "intraday"
            ret, day, why, both = simulate(Pm, 20, stop=st_, tp=tp, mode=mode, tstop=ts)
            print(sweep_line(summarize(rn, ret, day, why, both, base[m], y20[m], f20[m])))
    print("  件数が少ないので窓ごとの勝ち負けは粗い（上位1件は1窓あたり十数件）。")

    # ------------------------------------------------------------------ #
    # §9 大負けの共通点。特徴量は本番の205列 + 買う瞬間に分かる寄りの窓
    allf = list(dict.fromkeys([c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
                              + ["entry_gap"]))
    # しきい値を引く母集団は窓1も含めた out-of-fold の全行（窓2 のしきい値に窓1 を使う）
    ref = oofs[RULE_MODELS[0]][["Code", "Date", "fold"]].merge(
        df[["Code", "Date"] + allf], on=["Code", "Date"], how="left")
    add = ["Code", "Date"] + [c for c in allf if c not in s.columns]
    loser_section(s.merge(df[add], on=["Code", "Date"], how="left"),
                  u.merge(df[add], on=["Code", "Date"], how="left"), ref, allf)

    # ------------------------------------------------------------------ #
    # §10 窓ごと・out-of-fold の切り方ごと。運用者の指示（2026-09-23）
    # 「oof だけでなく、各cv（異なる窓）でもみたいです。１パターンのoofで戦略を
    #   決めても絶対にうまくいかないので（過適合）」
    shifts = [int(x) for x in args.shifts.split(",") if x.strip() != ""]
    if args.scores == "e27" and any(shifts):
        log("  --scores e27 では窓をずらした out-of-fold を作らない（ずらし0か月だけ）")
        shifts = [0]
    print("\n=== 10. 窓ごと・out-of-fold の切り方ごと（1通りの切り方に合わせ込んでいないか）===")
    print("  切り方 = 窓の境界を " + " / ".join(f"{k}か月" for k in shifts) + " ずらした out-of-fold。"
          "ずらしたものは3モデルとも学習し直す（パラメータは同じ）")
    print("  表の値 = その窓の「規則の平均 − 損切りなしの平均」（pt）。20営業日保有。")
    print("  列: −X% = 逆指値 / −1σ = ラベルと同じ σ の逆指値 / 終値−10% = 終値で判定して翌寄り /"
          " 利確20% = +20% 利確のみ / 5日目<0 = 5日目の終値が買値未満なら翌寄り")
    ok0 = folds_for(df["Date"], 0)
    got = oofs[RULE_MODELS[0]].groupby("fold")["Date"].agg(["min", "max"])
    bad = [f.index for f in ok0 if f.index in got.index
           and not (pd.Timestamp(f.test_start) <= got.loc[f.index, "min"]
                    and got.loc[f.index, "max"] <= pd.Timestamp(f.test_end))]
    print(f"  窓の境界の確認（ずらし0か月 = 本番と同じ窓か）: "
          f"{'一致' if not bad else '不一致の窓 ' + str(bad)}")
    cols10 = [c for c in F.columns(F.DEFAULT_PRESET) if c in df.columns]
    par10 = arm_params(df, cols10, args.arm) if any(shifts) else None
    pats = {}
    for k in shifts:
        o = oofs if k == 0 else shifted_scores(df, cols10, par10, k, args.arm)
        for a in o:
            o[a]["Date"] = pd.to_datetime(o[a]["Date"])
        pats[k] = o
    codes = set().union(*[set(o[RULE_MODELS[0]]["Code"]) for o in pats.values()])
    bars10 = bars if codes <= set(bars["Code"].unique()) else load_bars(codes)
    summ = {}
    for k in shifts:
        sel_k, P_k, r_k = pattern_rows(pats[k], df, bars10)
        rng = {f.index: f"{f.test_start}〜{f.test_end}" for f in folds_for(df["Date"], k)}
        rets = rule_returns(P_k, sel_k["sigma20"].to_numpy(dtype=float))
        fo = sel_k["fold"].to_numpy()
        print(f"\n  --- ずらし{k}か月: 選定 {r_k['n']:,}件 / 正例率 {r_k['label_rate']*100:.1f}% / "
              f"ret_o1_20 {r_k['ret20']:+.2f}% ---")
        summ[(k, "全日・全件")] = window_table(f"ずらし{k}か月・全日・全件", fo, rets, rng)
        hot = (sel_k["n_break"] >= 20).to_numpy()
        summ[(k, "発火20件以上")] = window_table(
            f"ずらし{k}か月・発火20件以上・全件", fo[hot], {n: v[hot] for n, v in rets.items()}, rng)

    print("\n  --- 10c. まとめ: 規則ごとに、切り方×絞り込みの 勝ち窓 と 窓の差の平均（pt）---")
    keys = list(summ)
    print(f"  {'規則':<12}" + "".join(f"{('ずらし' + str(k) + ' ' + g)[:14]:>18}" for k, g in keys)
          + f"{'勝ち窓の合計':>14}")
    for n, _ in RULES10:
        cells, won, tot = [], 0, 0
        for key in keys:
            v = summ[key][n]
            w = sum(x > 1e-12 for x in v)
            won, tot = won + w, tot + len(v)
            cells.append(f"{w:>5}/{len(v):<3}{np.mean(v)*100:>+8.2f}  ")
        print(f"  {n:<12}" + "".join(f"{c:>18}" for c in cells) + f"{won:>9}/{tot:<4}")
    print("  勝ち窓 = 損切りなしを上回った窓の数。窓の差の平均 = 窓ごとの差を窓の数で平均したもの"
          "（件数では重み付けしない）。")

    # ------------------------------------------------------------------ #
    keep = (["Code", "Date", "fold", "label", "cls", "need", "vol_20d", "sigma20",
             "entry_gap", "n_break", "p_min", "rank_day"] + [f"s_{a}" for a in RULE_MODELS]
            + [f"ret_o1_{h}" for h in lab.HORIZONS]
            + list(ex20.columns) + list(ex60.columns) + [f"c{k}" for k in DAYS])
    out = s[keep].copy()
    for name, stop, tp in (("sl10", 0.10, None), ("sl1s", 1.0, None), ("sl10_tp20", 0.10, TP)):
        st = stop * s["sigma20"].to_numpy(dtype=float) if name == "sl1s" else stop
        ret, day, why, _ = simulate(P, 20, stop=st, tp=tp)
        out[f"{name}_ret"] = ret
        out[f"{name}_day"] = day
        out[f"{name}_why"] = why
    out.to_csv(os.path.join(OOF_DIR, "e41_rows.csv"), index=False)
    log(f"記録: {OOF_DIR}/e41_sweep.csv / e41_rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
