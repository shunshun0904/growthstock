#!/usr/bin/env python3
"""
実験131: 買った後の「持ち続けるか・売るか」を強化学習（決定木の Q 学習）で決めると、今の売りのルールより 1取引の損益が
上がるか（検証だけ。本番の売りのルール・買いの規則・モデル・画面は変えない）。

運用者の依頼（2026-11-13）「買う銘柄は、今のモデルで上位を選ぶ方法で良いのですが、売るタイミング（トレード戦略）は強化学習で
前日の終値の判断でホールドか買うかを予測する、エージェントを構築したいです。…買った後、ホールドするか売るかのタイミングを
最適化したいだけです。」
運用者の選択（同日、AskUserQuestion。回す前に決めたもの）
  +10% の指値      両方作って比べる（指値を残す版 L・指値なし版 N）
  最長の保有        20営業日（今のルールと同じ。20営業日目まで持っていたら、その日の終値で必ず売る）
  目的              1取引の損益（売値 ÷ 買値 − 1 の平均。手数料・税・すべりは入れない。これまでの実験と同じ）
  学習の取引        候補すべてで学習（新高値の候補すべてを「買ったとして」学習。評価は買いの規則を満たした行で）
  学習方法          決定木の Q 学習（Fitted Q Iteration。LightGBM）
  判断の材料        値動きと買いのスコア（下の STATE_COLS。29個）
  判定              買いの規則を満たした行で3条件（下）

取引の約束（live_track.forward・実験83 と同じ）
  買い: 選定日（新高値の日。0日目）の翌営業日（1日目）の寄り。日数は買った日を1日目として数える
  判断: k日目（k = 1〜19）の引けの後に、その日までに分かる情報だけで「持ち続ける」か「売る」を決める。売ると決めたら
        k+1 日目の寄りで売る。20日目まで持っていたら 20日目の終値で売る
  指値を残す版 L: +10% の指値を毎日置く。k日目の高値が買値の 1.10倍以上なら +10% ちょうどで売れたと数える（寄りで +10% を
        超えて始まった日も +10%。控えめな側）。エージェントが寄りで売る日に寄りが +10% 以上なら、それも +10% で数える
  指値なし版 N: 指値は置かない。売るのはエージェントの判断か、20日目の終値だけ
  損益は % （売値 ÷ 買値 − 1）。1〜20日目に値が付かない日（始値・高値・安値・終値のどれかが無い）がある取引は外す（件数を出す）

学習（決定木の Q 学習。最終日から1日ずつ遡る Fitted Q Iteration）
  売買は株価を動かさない前提なので、過去の値動きの上で「売った場合」と「持ち続けた場合」の両方の結果が分かる。
  k = 19, 18, …, 1 の順に、k日目の引けの状態 s から
      D_k = （持ち続けて、k+1 日目以降は学習した方針に従ったときの実現損益） − （k+1 日目の寄りで売った損益）
  を LightGBM の回帰で当てる（1日につき1本）。方針は「D_k の予測 > 0 なら持ち続ける」。
  「k+1 日目以降の方針」に使う予測は、学習の取引を選定日の四半期で5つに分けた交差（その取引を使わずに学習したモデルの予測）。
  同じ取引で学習したモデルの予測をそのまま使うと、先の結果を覚えた方針で先の損益を置くことになり、持ち続ける側が実際より
  良く見える（Longstaff–Schwartz 法で知られた偏り）ため。評価に使うモデルは、学習の取引すべてで学習し直したもの
  木の設定は固定（PARAMS。探索しない）。種は 42 の1つ

評価（前向き。評価する期間より前に終わった取引だけで学習する）
  窓: 本番の out-of-fold と同じ（36か月 / 6か月 / 6か月）。テスト窓ごとに、20日目がテスト窓の始まりより前の取引すべてで学習し、
      テスト窓の取引に当てる
  買いの規則を満たした行（R0）: +10% の木3本（lgbm・xgb・cat）の out-of-fold のスコアを、テスト窓より前の窓の分布で百分位に
      直し（実験79 の window_split。丸めは画面と同じ小数1桁）、3本とも 95以上。テスト窓は百分位を付けられる 3つ目以降
  買いのスコア（状態）: 6本の out-of-fold のスコアそのもの（その行より前のデータだけで学習したモデルの採点。2021年5月より前の
      行には無いので欠測のまま渡す。参考に、スコアを状態に入れない版も並べる）

比べるもの（同じ行で）
  今のルール   +10% の指値、届かなければ 20日目の終値（live_track の出口。含み損の持ち越しは売る日が決まらないので、
               20日目の終値で売ったとして数える）
  エージェント L / N
  参考（判定に使わない）: 指値なし・20日目の終値、損切り（終値が買値の −8% 以下なら翌朝売る）＋指値、10日目の引けで見切って
               翌朝売る＋指値、スコアを状態に入れないエージェント L / N、上限（先が見えたときにいちばん高く売れた場合）

判定（回す前に決めた。エージェント L・N それぞれ）
  買いの規則を満たした行で、エージェントの損益 − 今のルールの損益を窓ごとに平均する（その窓に MIN_GROUP 行以上）。
    1. 窓ごとの差の平均が正
    2. 並べ替え検定で p < 0.05（選定日ごとに差の符号を入れ替える。片側。N_PERM 回）
    3. エージェントが上の窓が過半（n // 2 + 1 以上）
  3つとも満たせば「通る」。上位10%（lgbm の前向きの百分位 90以上）と全候補の行は参考に並べる。
  2つの版を別々に判定するので、どちらも効かなくても偶然で片方が通る見込みは約 10%。通っても本番に入れるかは運用者が決める

確かめ（合わなければ止まる）
  - 「いつも持ち続ける」方針の損益が、live_track.forward の値（指値あり = ret、指値なし = ret_close）と評価の全行で一致する
  - 買値が live_track.forward の entry と一致する
  - 状態は k日目までの値だけから作る（tests/test_e131_exit_rl.py: k日目より後の値を変えても状態が変わらない）
  - 上限（先が見えた場合）がどの方針よりも低くならない

公開ログには集計だけ（件数・平均・割合・窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e131_exit_rl.py                       （既定。--perm 2000）
    試運転: exp=e131_exit_rl.py args="--max-windows 2 --no-ref"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOLD = 20                 # 最長の保有（営業日。買った日が1日目）
LIMIT = 10.0              # 指値（%）
LEVEL_EPS = 1e-9          # 指値の比べの許容（live_track.LEVEL_EPS と同じ）
PRE = 5                   # 買う前の終値を何日ぶん持つか（3日・5日の変化に使う）
VOL_PRE = 20              # 買う前の平均出来高の日数（選定日まで）
STOP_REF = -8.0           # 参考の損切り（%。終値で判断）
HOLD10 = 10               # 参考の見切り（10日目の引けで判断）
N_FOLDS = 5               # 学習の中の交差の数（選定日の四半期で分ける）
MIN_FIT_ROWS = 300        # これ未満なら、その日のモデルは作らない（持ち続ける）
MIN_GROUP = 5             # 窓ごとの比較で、その窓に要る行の数（実験80・83 と同じ）
N_PERM = 2000
SEED = 42
LINE = 95.0               # 買いの規則の線（百分位）
TOP_LINE = 90.0           # 参考: 上位10%
RULE_MODELS = ("lgbm", "xgb", "cat")
SCORE_MODELS = ("lgbm", "xgb", "cat", "lgbm_k15", "xgb_k15", "cat_k15")
#: 木の設定（固定。探索しない）
PARAMS = {"objective": "regression", "n_estimators": 150, "learning_rate": 0.05, "num_leaves": 15,
          "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
          "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1}

PRICE_COLS = ["ret_now", "max_high", "min_low", "dd_peak", "ret_1d", "ret_3d", "ret_5d", "range_pos", "gap_today",
              "intraday", "vol_ratio", "vol_ratio3", "entry_gap", "to_limit", "up_share", "rvol", "new_high_today",
              "days_since_peak", "ret_vs_signal", "topix_since", "topix_1d", "excess", "vol_20d"]
SCORE_COLS = [f"score_{a}" for a in SCORE_MODELS]
STATE_COLS = PRICE_COLS + SCORE_COLS
VARIANTS = ("L", "N")
VARIANT_JA = {"L": "指値を残す版", "N": "指値なし版"}
#: 試運転（2026-11-14 の run 43457622788。--max-windows 2）で結果を見た窓。追加の作り B はこの後に足したので、
#: B の判定は、これらの窓を除いた形も並べる
TRIAL_WINDOWS = (10, 11)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 値動きの道筋（取引ごとに −PRE〜HOLD 日目）
# --------------------------------------------------------------------------- #
class Paths:
    """
    取引（Code, Date = 選定日）ごとの値動き。配列は (取引, PRE + 1 + HOLD) で、列 j は j − PRE 日目（0 = 選定日、1 = 買った日）。
    O・H・L・C・V は分割調整後。T は TOPIX の終値（0〜HOLD 日目。列 d は d 日目）。entry は 1日目の寄り。
    """

    def __init__(self, keys: pd.DataFrame, O, H, L, C, V, T, pre_vol, vol_20d, end_date, scores: Optional[pd.DataFrame] = None):
        self.keys = keys.reset_index(drop=True)
        self.O, self.H, self.L, self.C, self.V, self.T = (np.asarray(x, dtype=float) for x in (O, H, L, C, V, T))
        self.pre_vol = np.asarray(pre_vol, dtype=float)
        self.vol_20d = np.asarray(vol_20d, dtype=float)
        self.end_date = pd.to_datetime(pd.Series(end_date)).to_numpy()
        self.entry = self.O[:, PRE + 1].copy()
        self.scores = (scores.reset_index(drop=True) if scores is not None
                       else pd.DataFrame({c: np.full(len(self.keys), np.nan) for c in SCORE_COLS}))

    def __len__(self) -> int:
        return len(self.keys)

    def take(self, idx) -> "Paths":
        idx = np.asarray(idx)
        return Paths(self.keys.iloc[idx], self.O[idx], self.H[idx], self.L[idx], self.C[idx], self.V[idx], self.T[idx],
                     self.pre_vol[idx], self.vol_20d[idx], self.end_date[idx], self.scores.iloc[idx])

    # ---- 損益（%）。d は日（1〜HOLD） ----
    def r_open(self, d: int) -> np.ndarray:
        return (self.O[:, PRE + d] / self.entry - 1.0) * 100.0

    def r_close(self, d: int) -> np.ndarray:
        return (self.C[:, PRE + d] / self.entry - 1.0) * 100.0

    def hit(self, d: int) -> np.ndarray:
        """d日目の高値が指値（買値 × 1.10）に届いたか。"""
        level = self.entry * (1.0 + LIMIT / 100.0) * (1.0 - LEVEL_EPS)
        return self.H[:, PRE + d] >= level

    def sell_pay(self, k: int, variant: str) -> np.ndarray:
        """k日目の引けに売ると決めたときの損益（k+1 日目の寄り）。指値を残す版は +10% で頭打ち。"""
        r = self.r_open(k + 1)
        return np.minimum(r, LIMIT) if variant == "L" else r


def build_paths(keys: pd.DataFrame, bars: pd.DataFrame, topix: pd.DataFrame,
                vol_20d: Optional[Sequence[float]] = None) -> Tuple[Paths, Dict]:
    """
    keys（Code, Date）の取引の値動きを日足から切り出す。使えない取引は外し、理由ごとの件数を info に入れる:
      no_bar（選定日の日足が無い）・short（20日目まで日足が無い）・gap（1〜20日目に値の付かない日がある・買値が無い）
    戻りの Paths.keys は使える取引だけ（元の並びのまま。元の行番号は列 _row）。
    """
    keys = keys[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    keys["Date"] = pd.to_datetime(keys["Date"])
    keys["_row"] = np.arange(len(keys))
    v20 = np.full(len(keys), np.nan) if vol_20d is None else np.asarray(vol_20d, dtype=float)
    width = PRE + 1 + HOLD
    n = len(keys)
    O = np.full((n, width), np.nan)
    H = np.full((n, width), np.nan)
    L = np.full((n, width), np.nan)
    C = np.full((n, width), np.nan)
    V = np.full((n, width), np.nan)
    D = np.full((n, HOLD + 1), np.datetime64("NaT"), dtype="datetime64[ns]")
    pre_vol = np.full(n, np.nan)
    status = np.full(n, "no_bar", dtype=object)
    b = bars[bars["Code"].isin(set(keys["Code"]))].sort_values(["Code", "Date"])
    offs = np.arange(-PRE, HOLD + 1)
    by_code = keys.groupby("Code").indices
    for code, g in b.groupby("Code", sort=False):
        rows = by_code.get(code)
        if rows is None or not len(rows):
            continue
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        arr = {c: g[c].to_numpy(dtype=float) for c in ("AdjO", "AdjH", "AdjL", "AdjC", "AdjVo")}
        m = len(dates)
        want = keys.loc[rows, "Date"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(dates, want)
        for r, i, d in zip(rows, pos, want):
            if i >= m or dates[i] != d:
                continue
            if i + HOLD >= m:
                status[r] = "short"
                continue
            idx = i + offs
            ok = idx >= 0
            for name, dst in (("AdjO", O), ("AdjH", H), ("AdjL", L), ("AdjC", C), ("AdjVo", V)):
                dst[r, ok] = arr[name][idx[ok]]
            D[r] = dates[i:i + HOLD + 1]
            lo = max(0, i - VOL_PRE + 1)
            pv = arr["AdjVo"][lo:i + 1]
            pv = pv[np.isfinite(pv)]
            pre_vol[r] = pv.mean() if len(pv) >= VOL_PRE // 2 else np.nan
            status[r] = "ok"
    days = slice(PRE + 1, PRE + HOLD + 1)
    full = (np.isfinite(O[:, days]).all(axis=1) & np.isfinite(H[:, days]).all(axis=1)
            & np.isfinite(L[:, days]).all(axis=1) & np.isfinite(C[:, days]).all(axis=1)
            & np.isfinite(C[:, PRE]) & (O[:, PRE + 1] > 0))
    status[(status == "ok") & ~full] = "gap"
    keep = np.flatnonzero(status == "ok")
    tp = topix.copy()
    tp["Date"] = pd.to_datetime(tp["Date"])
    tp = tp.dropna(subset=["topix"]).drop_duplicates("Date", keep="last").set_index("Date")["topix"].astype(float)
    T = np.full((n, HOLD + 1), np.nan)
    if len(keep):
        flat = pd.DatetimeIndex(D[keep].ravel())
        T[keep] = tp.reindex(flat).to_numpy().reshape(len(keep), HOLD + 1)
    info = {"rows": int(n), "ok": int(len(keep)), **{k: int((status == k).sum()) for k in ("no_bar", "short", "gap")}}
    P = Paths(keys.iloc[keep], O[keep], H[keep], L[keep], C[keep], V[keep], T[keep], pre_vol[keep], v20[keep],
              D[keep][:, HOLD])
    return P, info


# --------------------------------------------------------------------------- #
# 状態（k日目の引けまでに分かる値だけ）
# --------------------------------------------------------------------------- #
def state_at(P: Paths, k: int) -> pd.DataFrame:
    """
    k日目（1〜HOLD−1）の引けの状態。**k日目より後の列には触らない**（最初に k日目までを切り出す）。
    損益・変化は % ではなく比（0.05 = +5%）。木は単調変換に不変なので尺度は効かない。
    """
    if not 1 <= k <= HOLD - 1:
        raise ValueError(f"k は 1〜{HOLD - 1}: {k}")
    end = PRE + k + 1
    O, H, L, C, V = P.O[:, :end], P.H[:, :end], P.L[:, :end], P.C[:, :end], P.V[:, :end]
    T = P.T[:, :k + 1]
    entry = O[:, PRE + 1]
    j = PRE + k                                   # k日目の列
    held = slice(PRE + 1, PRE + k + 1)            # 1〜k日目
    c = C[:, j]
    hi = H[:, held].max(axis=1)
    lo = L[:, held].min(axis=1)
    s = pd.DataFrame(index=np.arange(len(P)))
    s["ret_now"] = c / entry - 1.0
    s["max_high"] = hi / entry - 1.0
    s["min_low"] = lo / entry - 1.0
    s["dd_peak"] = c / hi - 1.0
    s["ret_1d"] = c / C[:, j - 1] - 1.0
    s["ret_3d"] = c / C[:, j - 3] - 1.0
    s["ret_5d"] = c / C[:, j - 5] - 1.0
    rng = H[:, j] - L[:, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        s["range_pos"] = np.where(rng > 0, (c - L[:, j]) / rng, np.nan)
        s["gap_today"] = O[:, j] / C[:, j - 1] - 1.0
        s["intraday"] = c / O[:, j] - 1.0
        pv = np.where(P.pre_vol > 0, P.pre_vol, np.nan)
        s["vol_ratio"] = V[:, j] / pv
        v3 = V[:, max(PRE + 1, j - 2):j + 1]                          # 直近3日（買ってから3日に満たなければある分）
        cnt = np.isfinite(v3).sum(axis=1)
        s["vol_ratio3"] = np.where(cnt > 0, np.nansum(v3, axis=1) / np.maximum(cnt, 1), np.nan) / pv
        s["entry_gap"] = entry / C[:, PRE] - 1.0
        s["to_limit"] = entry * (1.0 + LIMIT / 100.0) / c - 1.0
        dc = C[:, PRE + 1:end] / C[:, PRE:end - 1] - 1.0              # 1〜k日目の日々の変化
        s["up_share"] = (dc > 0).mean(axis=1)
        s["rvol"] = dc.std(axis=1, ddof=1) if k >= 2 else np.nan
        prev_hi = H[:, PRE + 1:j].max(axis=1) if k >= 2 else np.full(len(P), -np.inf)
        s["new_high_today"] = (H[:, j] >= prev_hi).astype(float)
        s["days_since_peak"] = (k - 1 - H[:, held].argmax(axis=1)).astype(float)
        s["ret_vs_signal"] = c / C[:, PRE] - 1.0
        s["topix_since"] = T[:, k] / T[:, 0] - 1.0
        s["topix_1d"] = T[:, k] / T[:, k - 1] - 1.0
    s["excess"] = s["ret_now"] - s["topix_since"]
    s["vol_20d"] = P.vol_20d
    for col in SCORE_COLS:
        s[col] = P.scores[col].to_numpy(dtype=float)
    return s[STATE_COLS]


# --------------------------------------------------------------------------- #
# 方針に従って売買する
# --------------------------------------------------------------------------- #
HoldFn = Callable[[int, np.ndarray], np.ndarray]


def simulate(P: Paths, hold_fn: HoldFn, variant: str) -> pd.DataFrame:
    """
    方針 hold_fn(k, 行番号) -> 持ち続けるか（真偽）で売買した結果。列: ret（%）, exit_day（売った日。寄りで売れば k+1）,
    how（limit = 指値 / agent = 方針で売った / end = 20日目の終値）。
    """
    if variant not in VARIANTS:
        raise KeyError(variant)
    n = len(P)
    ret = np.full(n, np.nan)
    day = np.zeros(n, dtype=int)
    how = np.full(n, "", dtype=object)
    alive = np.ones(n, dtype=bool)
    for k in range(1, HOLD + 1):
        if variant == "L":
            h = alive & P.hit(k)
            ret[h], day[h], how[h] = LIMIT, k, "limit"
            alive &= ~h
        if k == HOLD:
            ret[alive], day[alive], how[alive] = P.r_close(HOLD)[alive], HOLD, "end"
            break
        rows = np.flatnonzero(alive)
        if not len(rows):
            continue
        keep = np.asarray(hold_fn(k, rows), dtype=bool)
        if keep.shape != rows.shape:
            raise ValueError("hold_fn の戻りの長さが違う")
        sell = rows[~keep]
        ret[sell], day[sell], how[sell] = P.sell_pay(k, variant)[sell], k + 1, "agent"
        alive[sell] = False
    return pd.DataFrame({"ret": ret, "exit_day": day, "how": how})


def always_hold(k: int, rows: np.ndarray) -> np.ndarray:
    return np.ones(len(rows), dtype=bool)


def oracle(P: Paths, variant: str) -> np.ndarray:
    """先が見えたときにいちばん高く売れた場合の損益（%）。どの方針もこれを超えない。"""
    opens = np.column_stack([P.r_open(d) for d in range(2, HOLD + 1)])
    best = np.maximum(opens.max(axis=1), P.r_close(HOLD))
    if variant == "N":
        return best
    hit = np.column_stack([P.hit(d) for d in range(1, HOLD + 1)])
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, HOLD + 1)      # 指値に届いた日（無ければ 21）
    # 指値に届く日より前（その日の寄りを含む）に寄りで売る場合の最大（+10% で頭打ち）
    days = np.arange(2, HOLD + 1)[None, :]
    before = np.where(days <= first[:, None], np.minimum(opens, LIMIT), -np.inf).max(axis=1)
    return np.where(any_hit, LIMIT, np.maximum(before, P.r_close(HOLD)))


# --------------------------------------------------------------------------- #
# 学習（最終日から遡る Fitted Q Iteration）
# --------------------------------------------------------------------------- #
def fold_groups(dates: pd.Series, n_folds: int = N_FOLDS) -> np.ndarray:
    """選定日の四半期の通し番号 % n_folds。隣り合う日（値動きが重なる取引）を同じ組に入れる。"""
    d = pd.to_datetime(pd.Series(dates))
    return ((d.dt.year * 4 + (d.dt.quarter - 1)) % n_folds).to_numpy()


def _fit(X: np.ndarray, y: np.ndarray, seed: int, params: Optional[dict] = None):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(**{**(params or PARAMS), "random_state": seed})
    m.fit(X, y)
    return m


class Agent:
    """k日目（1〜HOLD−1）ごとのモデル。models[k] が無い日は持ち続ける。"""

    def __init__(self, variant: str, cols: List[str]):
        self.variant = variant
        self.cols = list(cols)
        self.models: Dict[int, object] = {}
        self.info: Dict[int, Dict] = {}
        #: 学習の取引の上での成績（交差の予測で売買したときの平均 value・いつも持ち続けたときの平均 base。%）
        self.train: Dict = {}

    def hold_fn(self, P: Paths) -> HoldFn:
        cache: Dict[int, pd.DataFrame] = {}

        def fn(k: int, rows: np.ndarray) -> np.ndarray:
            m = self.models.get(k)
            if m is None:
                return np.ones(len(rows), dtype=bool)
            if k not in cache:
                cache[k] = state_at(P, k)[self.cols]
            return m.predict(cache[k].iloc[rows].to_numpy(dtype=float)) > 0.0
        return fn


def fit_agent(P: Paths, variant: str, cols: Sequence[str] = STATE_COLS, seed: int = SEED,
              params: Optional[dict] = None, n_folds: int = N_FOLDS, min_rows: int = MIN_FIT_ROWS) -> Agent:
    """
    学習の取引 P から方針を作る。k = HOLD−1, …, 1 の順に、D_k = 持ち続けた先の実現損益 − 売った損益 を回帰する。
    先の実現損益は、k+1 日目以降を「交差の予測で決めた方針」に従ったときの値（その取引を使わずに学習したモデルの予測）。
    """
    agent = Agent(variant, list(cols))
    n = len(P)
    groups = fold_groups(P.keys["Date"], n_folds)
    hits = np.column_stack([P.hit(d) for d in range(1, HOLD + 1)]) if variant == "L" else np.zeros((n, HOLD), dtype=bool)
    hit_by = np.cumsum(hits, axis=1) > 0                 # 列 d−1: d日目までに指値に届いたか
    v_next = P.r_close(HOLD)                             # 20日目の引けの判断の時点の値（= 20日目の終値で売る）
    for k in range(HOLD - 1, 0, -1):
        y_sell = P.sell_pay(k, variant)
        y_hold = np.where(hits[:, k], LIMIT, v_next)     # hits[:, k] = k+1 日目に指値に届く
        alive = ~hit_by[:, k - 1]                        # k日目までに指値に届いていない
        d = y_hold - y_sell
        rows = np.flatnonzero(alive & np.isfinite(d))
        hold = np.ones(n, dtype=bool)
        rec = {"alive": int(alive.sum()), "fit_rows": int(len(rows))}
        if len(rows) >= min_rows:
            X = state_at(P, k)[agent.cols].to_numpy(dtype=float)
            pred = np.full(n, np.nan)
            for g in range(n_folds):
                te = rows[groups[rows] == g]
                tr = rows[groups[rows] != g]
                if not len(te):
                    continue
                if len(tr) < min_rows:
                    pred[te] = np.inf                    # 学習できない組は持ち続ける
                    continue
                pred[te] = _fit(X[tr], d[tr], seed, params).predict(X[te])
            agent.models[k] = _fit(X[rows], d[rows], seed, params)
            hold[rows] = pred[rows] > 0.0
            rec["hold_share"] = float(hold[rows].mean())
            rec["d_mean"] = float(d[rows].mean())
        agent.info[k] = rec
        v_next = np.where(hold, y_hold, y_sell)          # k日目の引けの判断の時点の値（方針に従った実現損益）
    # 学習の取引の上での成績: 1日目に指値に届いた取引は +10%、ほかは 1日目の引けの判断の時点の値
    start = np.where(hits[:, 0], LIMIT, v_next)
    base = simulate(P, always_hold, variant)["ret"].to_numpy()
    agent.train = {"n": int(n), "value": float(np.mean(start)) if n else float("nan"),
                   "base": float(np.mean(base)) if n else float("nan"),
                   "hold_share": float(np.mean([x["hold_share"] for x in agent.info.values() if "hold_share" in x]))
                   if agent.models else float("nan")}
    return agent


# --------------------------------------------------------------------------- #
# 追加の作り B（案）: 今のルールを土台にした方針の改善。売ると決める線を学習の取引の中で選ぶ
# --------------------------------------------------------------------------- #
#: 売ると決める線 δ の候補（%）。「持ち続けた場合の損益 − 売った場合の損益」の予測が −δ より悪いときだけ売る。inf = 売らない
DELTAS = (float("inf"), 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0)


class Improver:
    """
    今のルール（売らない。指値を残す版は指値だけ）を土台にした方針。k日目の引けに、
        A_k = （このあとも売らずに今のルールを続けたときの損益） − （k+1 日目の寄りで売った損益）
    の予測が −delta より悪いときだけ売る。delta は学習の取引の上で、交差の予測で売買したときの平均の損益がいちばん高いものを
    DELTAS から選ぶ（同じなら大きい方 = 売らない側）。delta = inf なら今のルールと同じ。
    """

    def __init__(self, variant: str, cols: List[str]):
        self.variant = variant
        self.cols = list(cols)
        self.models: Dict[int, object] = {}
        self.delta = float("inf")
        self.train: Dict = {}

    def hold_fn(self, P: Paths) -> HoldFn:
        cache: Dict[int, pd.DataFrame] = {}

        def fn(k: int, rows: np.ndarray) -> np.ndarray:
            m = self.models.get(k)
            if m is None or not np.isfinite(self.delta):
                return np.ones(len(rows), dtype=bool)
            if k not in cache:
                cache[k] = state_at(P, k)[self.cols]
            return m.predict(cache[k].iloc[rows].to_numpy(dtype=float)) >= -self.delta
        return fn


def fit_improver(P: Paths, variant: str, cols: Sequence[str] = STATE_COLS, seed: int = SEED,
                 params: Optional[dict] = None, n_folds: int = N_FOLDS, min_rows: int = MIN_FIT_ROWS,
                 deltas: Sequence[float] = DELTAS) -> Improver:
    """
    学習の取引 P から Improver を作る。目的変数は今のルールを続けたときの実現損益との差なので、ほかの日のモデルの予測に
    依らない（日ごとに独立に学習できる）。線 delta を選ぶときの売買は交差の予測（その取引を使わずに学習したモデルの予測）で行う。
    """
    imp = Improver(variant, list(cols))
    n = len(P)
    groups = fold_groups(P.keys["Date"], n_folds)
    hits = np.column_stack([P.hit(d) for d in range(1, HOLD + 1)]) if variant == "L" else np.zeros((n, HOLD), dtype=bool)
    hit_by = np.cumsum(hits, axis=1) > 0
    close_end = P.r_close(HOLD)
    oof = {k: np.full(n, np.inf) for k in range(1, HOLD)}        # 予測の無い行は持ち続ける
    for k in range(1, HOLD):
        g0 = np.where(hits[:, k:].any(axis=1), LIMIT, close_end)  # k+1 日目以降に指値に届けば +10%、届かなければ20日目の終値
        a = g0 - P.sell_pay(k, variant)
        rows = np.flatnonzero(~hit_by[:, k - 1] & np.isfinite(a))
        if len(rows) < min_rows:
            continue
        X = state_at(P, k)[imp.cols].to_numpy(dtype=float)
        for g in range(n_folds):
            te = rows[groups[rows] == g]
            tr = rows[groups[rows] != g]
            if len(te) and len(tr) >= min_rows:
                oof[k][te] = _fit(X[tr], a[tr], seed, params).predict(X[te])
        imp.models[k] = _fit(X[rows], a[rows], seed, params)
    table = []
    best = None
    for dl in deltas:
        res = simulate(P, lambda k, rows, _d=dl: oof[k][rows] >= -_d, variant)
        rec = {"delta": float(dl), "value": float(res["ret"].mean()) if n else float("nan"),
               "sold": float((res["how"] == "agent").mean()) if n else float("nan")}
        table.append(rec)
        if best is None or rec["value"] > best["value"] + 1e-12:   # 同じなら先（= 大きい線。売らない側）を残す
            best = rec
    base = next((r["value"] for r in table if not np.isfinite(r["delta"])), float("nan"))
    imp.delta = best["delta"] if (best is not None and imp.models) else float("inf")
    imp.train = {"n": int(n), "value": best["value"] if best else float("nan"), "base": base,
                 "delta": imp.delta, "sold": best["sold"] if best else float("nan"), "table": table}
    return imp


# --------------------------------------------------------------------------- #
# 評価の集計
# --------------------------------------------------------------------------- #
def stats(ret: pd.Series, how: Optional[pd.Series] = None, day: Optional[pd.Series] = None) -> Dict:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    nan = float("nan")
    out = {"n": int(len(r)), "mean": float(r.mean()) if len(r) else nan, "med": float(r.median()) if len(r) else nan,
           "win": float((r > 0).mean() * 100) if len(r) else nan, "bad10": float((r < -10).mean() * 100) if len(r) else nan,
           "se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else nan}
    if how is not None and len(r):
        h = how.loc[r.index]
        out["limit"] = float((h == "limit").mean() * 100)
        out["agent"] = float((h == "agent").mean() * 100)
    if day is not None and len(r):
        out["days"] = float(pd.to_numeric(day.loc[r.index]).mean())
    return out


def window_diffs(ev: pd.DataFrame, a: str, b: str, min_rows: int = MIN_GROUP) -> pd.DataFrame:
    """同じ行で a − b の平均を窓ごとに（その窓に min_rows 行以上）。"""
    d = (ev[a] - ev[b]).rename("d")
    g = pd.concat([ev["window"], d], axis=1).dropna()
    out = g.groupby("window")["d"].agg(["size", "mean"]).reset_index().rename(columns={"size": "n"})
    return out[out["n"] >= min_rows].reset_index(drop=True)


def _window_stat(win: np.ndarray, d: np.ndarray, windows: np.ndarray) -> float:
    return float(np.mean([d[win == w].mean() for w in windows]))


def judge(ev: pd.DataFrame, agent: str, base: str, n_perm: int = N_PERM, seed: int = SEED,
          min_rows: int = MIN_GROUP) -> Dict:
    """判定の3条件（窓ごとの差の平均が正・選定日ごとの符号の入れ替えで p < 0.05・上の窓が過半）。"""
    wd = window_diffs(ev, agent, base, min_rows)
    out = {"windows": int(len(wd)), "rows": int(wd["n"].sum()) if len(wd) else 0, "mean": float("nan"), "p": float("nan"),
           "wins": 0, "need": 0, "positive": False, "significant": False, "majority": False, "pass": False}
    if not len(wd):
        return out
    sub = ev[ev["window"].isin(wd["window"])]
    d = (sub[agent] - sub[base]).to_numpy(dtype=float)
    ok = np.isfinite(d)
    d, win = d[ok], sub["window"].to_numpy()[ok]
    codes, inv = np.unique(pd.to_datetime(sub["Date"]).to_numpy()[ok], return_inverse=True)
    windows = wd["window"].to_numpy()
    obs = _window_stat(win, d, windows)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        sign = rng.choice((-1.0, 1.0), size=len(codes))[inv]
        if _window_stat(win, d * sign, windows) >= obs - 1e-12:
            ge += 1
    wins = int((wd["mean"] > 0).sum())
    need = len(wd) // 2 + 1
    out.update(mean=obs, se=float(wd["mean"].std(ddof=1) / np.sqrt(len(wd))) if len(wd) > 1 else float("nan"),
               p=(1 + ge) / (1 + n_perm), wins=wins, need=need, positive=bool(obs > 0),
               majority=bool(wins >= need), dates=int(len(codes)))
    out["significant"] = bool(out["p"] < 0.05)
    out["pass"] = bool(out["positive"] and out["significant"] and out["majority"])
    return out


def judge_text(j: Dict) -> str:
    if not j["windows"]:
        return "測れない（行の足りる窓が無い）"
    mark = lambda b: "○" if b else "×"  # noqa: E731
    return (f"窓ごとの差の平均 {j['mean']:+.2f}pt {mark(j['positive'])} / 並べ替え p = {j['p']:.3f} {mark(j['significant'])} / "
            f"上の窓 {j['wins']}/{j['windows']}（{j['need']} 以上）{mark(j['majority'])} → {'通る' if j['pass'] else '通らない'}"
            f"（{j['rows']:,}行・選定日 {j.get('dates', 0):,}日）")


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #
def load_scores(data_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """+10%・+15% の木6本の out-of-fold を (Code, Date) で横に並べる（学習の記録と同じファイルかは flat_oof_path が確かめる）。"""
    import model_artifacts as MA
    wide, info = None, {}
    for a in SCORE_MODELS:
        if not (MA.directory(a) / "meta.json").is_file():
            raise SystemExit(f"{a} の学習の記録（meta.json）が無い（固定 +10%・+15% の out-of-fold かを確かめられない）")
        o = pd.read_parquet(MA.flat_oof_path(a, data_dir))
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        info[a] = {"rows": int(len(o)), "dups": int(o.duplicated(["Code", "Date"]).sum())}
        part = o.drop_duplicates(["Code", "Date"])[["Code", "Date", "score"]].rename(columns={"score": f"score_{a}"})
        wide = part if wide is None else wide.merge(part, on=["Code", "Date"], how="outer")
    return wide.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True), info


def rule_flags(wide: pd.DataFrame, frame_dates: pd.Series) -> Tuple[pd.DataFrame, list]:
    """
    テスト窓ごとの前向きの百分位（その窓より前の窓の分布）で、買いの規則（3本とも LINE 以上）と上位10% の印を付ける。
    戻り: 行（Code, Date, window, r0, top10）と窓の一覧。百分位を付けられない窓の行は入らない。
    """
    import e79_model_clusters as E79
    import e82_and_rule_variants as E82
    df = wide.dropna(subset=[f"score_{a}" for a in RULE_MODELS]).reset_index(drop=True)
    folds = E79.make_windows(frame_dates)
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    split = E79.window_split(df, folds, models=RULE_MODELS)
    parts = []
    for w, (_, te, _, Pte) in split.items():
        R = E82.rounded(Pte)
        part = df.iloc[te][["Code", "Date", "window"]].copy()
        part["r0"] = (R >= LINE).all(axis=1)
        part["top10"] = R[:, RULE_MODELS.index("lgbm")] >= TOP_LINE
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Code", "Date", "window", "r0", "top10"])
    return out, folds


def check_against_live_track(P: Paths, bars: pd.DataFrame, rows: np.ndarray) -> Dict:
    """「いつも持ち続ける」方針の損益と買値が live_track.forward と一致するか（rows の取引で）。違えば止める。"""
    import live_track as LT
    sub = P.take(rows)
    fw = LT.forward(bars, sub.keys[["Code", "Date"]], target=LIMIT, hold=HOLD, fallback="close")
    if len(fw) != len(sub):
        raise SystemExit("live_track.forward の行の数が合わない")
    hl = simulate(sub, always_hold, "L")["ret"].to_numpy()
    hn = simulate(sub, always_hold, "N")["ret"].to_numpy()
    out = {"rows": int(len(sub)),
           "entry": int(np.isclose(fw["entry"].to_numpy(dtype=float), sub.entry, rtol=1e-12, atol=0).sum()),
           "limit": int(np.isclose(fw["ret"].to_numpy(dtype=float), hl, rtol=0, atol=1e-9).sum()),
           "close": int(np.isclose(fw["ret_close"].to_numpy(dtype=float), hn, rtol=0, atol=1e-9).sum())}
    print(f"  確かめ: live_track.forward と一致 — 買値 {out['entry']:,} / 指値あり {out['limit']:,} / 指値なし {out['close']:,}"
          f"（{out['rows']:,}行）")
    if not (out["entry"] == out["limit"] == out["close"] == out["rows"]):
        raise SystemExit("「いつも持ち続ける」方針の損益が live_track.forward と合わない")
    return out


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
POLICIES = (("rule", "今のルール（+10% の指値・20日目の終値）"), ("agent_L", "エージェント・指値を残す版"),
            ("agent_N", "エージェント・指値なし版"),
            ("agentb_L", "追加の作り B（方針の改善）・指値を残す版"), ("agentb_N", "追加の作り B（方針の改善）・指値なし版"),
            ("c20", "参考: 指値なし・20日目の終値"),
            ("stop_L", f"参考: 損切り {STOP_REF:g}%（終値）＋指値"), ("hold10_L", f"参考: {HOLD10}日目で見切り＋指値"),
            ("agentp_L", "参考: スコアなしのエージェント・指値を残す版"), ("agentp_N", "参考: スコアなしのエージェント・指値なし版"),
            ("oracle_L", "上限（先が見えた場合）・指値あり"), ("oracle_N", "上限（先が見えた場合）・指値なし"))


def run_windows(P: Paths, flags: pd.DataFrame, folds: list, with_ref: bool = True, max_windows: int = 0,
                seed: int = SEED, with_b: bool = False, train_only: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    テスト窓ごとに学習して当てる。戻り: 評価の行（方針ごとの損益・売った日・売り方）と学習の記録。
    train_only なら学習だけして、テスト窓の取引には当てない（評価の行は空。学習の中の確かめだけを見るとき）。
    """
    key = P.keys[["Code", "Date"]].assign(_p=np.arange(len(P)))
    fl = flags.merge(key, on=["Code", "Date"], how="inner")
    start = {f.index: pd.Timestamp(f.test_start) for f in folds}
    windows = sorted(int(w) for w in fl["window"].unique())
    if max_windows:
        windows = windows[-max_windows:]
    out, record = [], {}
    end = pd.to_datetime(pd.Series(P.end_date)).to_numpy()
    for w in windows:
        t0 = time.time()
        te = fl[fl["window"] == w]
        tr = np.flatnonzero(end < np.datetime64(start[w]))
        Ptr = P.take(tr)
        rec = {"train": int(len(Ptr)), "test": int(len(te)), "train_to": str(pd.Timestamp(end[tr].max()).date()) if len(tr) else "",
               "test_from": str(start[w].date())}
        Pte = None if train_only else P.take(te["_p"].to_numpy())
        ev = te[["Code", "Date", "window", "r0", "top10"]].reset_index(drop=True)
        arms = [("agent", STATE_COLS)] + ([("agentp", PRICE_COLS)] if with_ref else [])
        for name, cols in arms:
            for v in VARIANTS:
                ag = fit_agent(Ptr, v, cols, seed)
                rec[f"{name}_{v}"] = {"models": len(ag.models), "hold_share": ag.train["hold_share"], "train": ag.train}
                if name == "agent":
                    rec[f"gain_{v}"] = gain_share(ag)
                if Pte is not None:
                    res = simulate(Pte, ag.hold_fn(Pte), v)
                    ev[f"{name}_{v}"], ev[f"{name}_{v}_day"], ev[f"{name}_{v}_how"] = res["ret"], res["exit_day"], res["how"]
        if with_b:
            for v in VARIANTS:
                im = fit_improver(Ptr, v, STATE_COLS, seed)
                rec[f"agentb_{v}"] = {"models": len(im.models), "train": im.train}
                if Pte is not None:
                    res = simulate(Pte, im.hold_fn(Pte), v)
                    ev[f"agentb_{v}"], ev[f"agentb_{v}_day"], ev[f"agentb_{v}_how"] = res["ret"], res["exit_day"], res["how"]
        if Pte is None:
            rec["seconds"] = round(time.time() - t0)
            record[w] = rec
            log(f"  窓{w}: 学習 {rec['train']:,}件（20日目が 〜{rec['train_to']}）だけ（テスト窓には当てない）/ {rec['seconds']}秒")
            continue
        for v, col in (("L", "rule"), ("N", "c20")):
            res = simulate(Pte, always_hold, v)
            ev[col], ev[f"{col}_day"], ev[f"{col}_how"] = res["ret"], res["exit_day"], res["how"]
        st = {k: state_at(Pte, k)["ret_now"].to_numpy() for k in range(1, HOLD)}
        res = simulate(Pte, lambda k, rows: st[k][rows] * 100.0 > STOP_REF, "L")
        ev["stop_L"], ev["stop_L_day"], ev["stop_L_how"] = res["ret"], res["exit_day"], res["how"]
        res = simulate(Pte, lambda k, rows: np.full(len(rows), k < HOLD10), "L")
        ev["hold10_L"], ev["hold10_L_day"], ev["hold10_L_how"] = res["ret"], res["exit_day"], res["how"]
        for v in VARIANTS:
            ev[f"oracle_{v}"] = oracle(Pte, v)
        rec["seconds"] = round(time.time() - t0)
        record[w] = rec
        out.append(ev)
        log(f"  窓{w}: 学習 {rec['train']:,}件（20日目が 〜{rec['train_to']}）→ テスト {rec['test']:,}件（{rec['test_from']} 〜）/ {rec['seconds']}秒")
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), record


def gain_share(ag: Agent) -> Dict[str, float]:
    """日ごとのモデルの寄与（gain）を合わせた割合（列ごと）。"""
    tot = pd.Series(0.0, index=ag.cols)
    for m in ag.models.values():
        g = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=ag.cols)
        s = float(g.sum())
        if s > 0:
            tot += g / s
    s = float(tot.sum())
    return {c: float(v / s) for c, v in tot.items()} if s > 0 else {}


ARM_JA = (("agent", "決めた作り A"), ("agentp", "参考: スコアなしの A"), ("agentb", "追加の作り B"))


def train_report(record: Dict) -> None:
    """学習の取引の上での成績（テスト窓の取引は使わない）。交差の予測で売買したときの1取引の損益と、いつも持ち続けたとき。"""
    print("\n■ 学習の中の確かめ（テスト窓の取引は使わない。学習の取引を、その取引を使わずに学習したモデルの予測で売買したときの"
          "1取引の損益。「持ち続ける」は指値を残す版 = 今のルール、指値なし版 = 20日目の終値）")
    for w, rec in record.items():
        print(f"  窓{w}（学習 {rec['train']:,}件・20日目が 〜{rec['train_to']}）")
        for name, ja in ARM_JA:
            for v in VARIANTS:
                r = rec.get(f"{name}_{v}")
                if not r or "train" not in r:
                    continue
                t = r["train"]
                line = (f"    {ja}・{VARIANT_JA[v]}: 方針 {t['value']:+.2f}% / 持ち続ける {t['base']:+.2f}% / "
                        f"差 {t['value'] - t['base']:+.2f}pt")
                if name == "agentb":
                    dl = t["delta"]
                    line += (f"・選んだ線 {'売らない' if not np.isfinite(dl) else f'{dl:g}%'}（方針で売った割合 {t['sold'] * 100:.0f}%）"
                             "｜線ごとの差 " + " / ".join(
                                 f"{'売らない' if not np.isfinite(x['delta']) else format(x['delta'], 'g')} {x['value'] - t['base']:+.2f}"
                                 for x in t["table"]))
                else:
                    line += f"・持ち続けると決めた割合（日ごとの平均）{t['hold_share'] * 100:.0f}%"
                print(line)


def check_oracle(ev: pd.DataFrame) -> int:
    """上限がどの方針よりも低くなっていないか（低い行の数）。"""
    bad = 0
    for col, _ in POLICIES:
        if col.startswith("oracle") or col not in ev.columns:
            continue
        v = "N" if col.endswith("_N") or col == "c20" else "L"
        bad += int((ev[col] > ev[f"oracle_{v}"] + 1e-9).sum())
    return bad


def report(ev: pd.DataFrame, record: Dict, n_perm: int = N_PERM, seed: int = SEED) -> Dict:
    summary: Dict = {"groups": {}, "judge": {}, "windows": {}, "sell_days": {}, "record": record}
    groups = (("r0", "買いの規則を満たした行（3本とも95以上）"), ("top10", "参考: 上位10%（lgbm の百分位 90以上）"),
              ("all", "参考: 全候補の行"))
    ev = ev.copy()
    ev["all"] = True
    months = max(1, ev["Date"].dt.to_period("M").nunique())
    print(f"\n■ 1. 評価の行: {len(ev):,}件・テスト窓 {ev['window'].nunique()}つ（{ev['Date'].min().date()} 〜 {ev['Date'].max().date()}・"
          f"{months}か月）/ 規則を満たした行 {int(ev['r0'].sum()):,}（月 {ev['r0'].sum() / months:.1f}件）/ 上位10% {int(ev['top10'].sum()):,}")
    print("\n■ 2. 成績（1取引の損益 %。同じ行で。指値 = +10% の指値で売れた割合、方針 = エージェント・参考の規則が売った割合、"
          "日数 = 売るまでの営業日）")
    for flag, ja in groups:
        g = ev[ev[flag].astype(bool)]
        print(f"  ◆ {ja}: {len(g):,}件")
        print(f"    {'':<44}{'平均':>8}{'中央':>8}{'勝率':>7}{'−10%割れ':>9}{'指値':>7}{'方針':>7}{'日数':>7}{'SE':>7}")
        summary["groups"][flag] = {}
        for col, name in POLICIES:
            if col not in g.columns:
                continue
            how = g[f"{col}_how"] if f"{col}_how" in g.columns else None
            day = g[f"{col}_day"] if f"{col}_day" in g.columns else None
            s = stats(g[col], how, day)
            summary["groups"][flag][col] = s
            f = lambda k, fmt="{:>6.1f}%": fmt.format(s[k]) if k in s and np.isfinite(s[k]) else f"{'—':>7}"  # noqa: E731
            print(f"    {name:<44}{s['mean']:>+7.2f}%{s['med']:>+7.2f}%{s['win']:>6.1f}%{s['bad10']:>8.1f}%{f('limit')}{f('agent')}"
                  f"{f('days', '{:>7.1f}')}{s['se']:>7.2f}")
    print("\n■ 3. 判定（回す前に決めた。エージェントの損益 − 今のルールの損益を窓ごとに。3つとも満たせば通る）")
    for flag, ja in groups:
        summary["judge"][flag] = {}
        for v in VARIANTS:
            j = judge(ev[ev[flag].astype(bool)], f"agent_{v}", "rule", n_perm, seed)
            summary["judge"][flag][v] = j
            tag = "判定" if flag == "r0" else "参考"
            print(f"  [{tag}] {ja}・{VARIANT_JA[v]}: {judge_text(j)}")
    has_b = "agentb_L" in ev.columns
    if has_b:
        print("  追加の作り B（試運転の後に足した案。同じ3条件。エージェント B の損益 − 今のルールの損益）")
        for flag, ja in groups:
            for v in VARIANTS:
                j = judge(ev[ev[flag].astype(bool)], f"agentb_{v}", "rule", n_perm, seed)
                summary["judge"][flag][f"B_{v}"] = j
                print(f"  [{'追加' if flag == 'r0' else '参考'}] {ja}・{VARIANT_JA[v]}: {judge_text(j)}")
    rest = ev[~ev["window"].isin(TRIAL_WINDOWS)]
    if len(rest) and len(rest) != len(ev):
        print(f"  試運転で見た窓（{'・'.join(map(str, TRIAL_WINDOWS))}）を除いた形（買いの規則を満たした行）")
        summary["judge"]["r0_untried"] = {}
        for name, ja in (("agent", "決めた作り A"),) + ((("agentb", "追加の作り B"),) if has_b else ()):
            for v in VARIANTS:
                j = judge(rest[rest["r0"].astype(bool)], f"{name}_{v}", "rule", n_perm, seed)
                summary["judge"]["r0_untried"][f"{name}_{v}"] = j
                print(f"    {ja}・{VARIANT_JA[v]}: {judge_text(j)}")
    print("  参考: 同じ形で、ほかの方針 − 今のルール（買いの規則を満たした行）")
    for col in ("c20", "stop_L", "hold10_L", "agentp_L", "agentp_N"):
        if col in ev.columns:
            j = judge(ev[ev["r0"].astype(bool)], col, "rule", n_perm, seed)
            summary["judge"].setdefault("ref", {})[col] = j
            print(f"    {dict(POLICIES)[col]}: {judge_text(j)}")
    print("\n■ 4. 窓ごとの差（買いの規則を満たした行。エージェント − 今のルール、pt）")
    r0 = ev[ev["r0"].astype(bool)]
    wl, wn = window_diffs(r0, "agent_L", "rule", 1), window_diffs(r0, "agent_N", "rule", 1)
    tab = wl.merge(wn, on=["window", "n"], suffixes=("_L", "_N"))
    if has_b:
        bl = window_diffs(r0, "agentb_L", "rule", 1).rename(columns={"mean": "mean_BL"})
        bn = window_diffs(r0, "agentb_N", "rule", 1).rename(columns={"mean": "mean_BN"})
        tab = tab.merge(bl, on=["window", "n"], how="left").merge(bn, on=["window", "n"], how="left")
    span = ev.groupby("window")["Date"].agg(["min", "max"])
    print(f"    {'窓':>3}  {'期間':<23}{'件数':>6}{'A 指値を残す版':>13}{'A 指値なし版':>12}"
          + (f"{'B 指値を残す版':>13}{'B 指値なし版':>12}" if has_b else ""))
    for _, r in tab.iterrows():
        w = int(r["window"])
        print(f"    {w:>3}  {span.loc[w, 'min'].date()}〜{span.loc[w, 'max'].date()}{int(r['n']):>6}{r['mean_L']:>+12.2f}{r['mean_N']:>+12.2f}"
              + (f"{r['mean_BL']:>+12.2f}{r['mean_BN']:>+12.2f}" if has_b else "")
              + ("  （5件未満。判定には入らない）" if r["n"] < MIN_GROUP else "")
              + ("  （試運転で見た窓）" if w in TRIAL_WINDOWS else ""))
        summary["windows"][w] = {"n": int(r["n"]), "L": float(r["mean_L"]), "N": float(r["mean_N"]),
                                 **({"BL": float(r["mean_BL"]), "BN": float(r["mean_BN"])} if has_b else {})}
    print("\n■ 5. エージェントの売り方（買いの規則を満たした行。売った日ごとの割合 %。寄りで売った日）")
    for name, ja in (("agent", "A"),) + ((("agentb", "B"),) if has_b else ()):
        for v in VARIANTS:
            d = r0.loc[r0[f"{name}_{v}_how"] == "agent", f"{name}_{v}_day"]
            share = (d.value_counts(normalize=True).sort_index() * 100) if len(d) else pd.Series(dtype=float)
            summary["sell_days"][f"{name}_{v}"] = {int(k): float(x) for k, x in share.items()}
            early = float(share[share.index <= 5].sum()) if len(share) else float("nan")
            print(f"  {ja}・{VARIANT_JA[v]}: 方針で売った {len(d):,}件 / {len(r0):,}件・うち 2〜5日目 {early:.0f}%・"
                  + " ".join(f"{int(k)}日 {x:.0f}" for k, x in share.items()))
    print("\n■ 6. 学習の記録（窓ごと）と、判断に効いた材料（寄与 gain の割合。日ごとのモデルの平均。最後の窓）")
    for w, rec in record.items():
        print(f"    窓{w}: 学習 {rec['train']:,}件 → テスト {rec['test']:,}件・学習の中で持ち続けると決めた割合（日ごとの平均） "
              + " / ".join(f"{VARIANT_JA[v]} {rec[f'agent_{v}']['hold_share'] * 100:.0f}%" for v in VARIANTS))
    train_report(record)
    if record:
        last = record[max(record)]
        for v in VARIANTS:
            g = sorted(last.get(f"gain_{v}", {}).items(), key=lambda kv: -kv[1])
            sc = sum(x for c, x in g if c in SCORE_COLS)
            print(f"  {VARIANT_JA[v]}: 買いのスコア6本の合計 {sc * 100:.0f}% / 上位 " + " / ".join(f"{c} {x * 100:.0f}%" for c, x in g[:10]))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験131: 売りのタイミングの強化学習（決定木の Q 学習。検証だけ）")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--max-windows", type=int, default=0, help="試運転: 最後の n 窓だけ（0 で全部）")
    ap.add_argument("--no-ref", action="store_true", help="スコアなしのエージェント（参考）を作らない")
    ap.add_argument("--with-b", action="store_true", help="追加の作り B（今のルールを土台にした方針の改善）も作る")
    ap.add_argument("--train-only", action="store_true",
                    help="学習だけして、テスト窓の取引には当てない（学習の中の確かめだけを出す。評価の結果は見ない）")
    args = ap.parse_args(argv)
    import build_dataset as B
    import lab
    import live_track as LT
    t0 = time.time()
    print("=" * 78)
    print("実験131 売りのタイミングの強化学習（決定木の Q 学習）: 持ち続けるか・翌朝の寄りで売るかを毎日の引けで決める")
    print(f"  最長 {HOLD}営業日 / 指値 +{LIMIT:g}% を残す版 L・指値なし版 N / 目的 1取引の損益 / 状態 {len(STATE_COLS)}個"
          f"（値動き {len(PRICE_COLS)}・買いのスコア {len(SCORE_COLS)}）")
    print(f"  学習: 候補すべて・最終日から遡る Fitted Q Iteration（LightGBM {PARAMS['n_estimators']}本・葉 {PARAMS['num_leaves']}・"
          f"交差 {N_FOLDS}つ・種 {SEED}）/ 判定: 買いの規則を満たした行で3条件（並べ替え {args.perm}回）")
    print("=" * 78)
    fr = lab.frame()
    fr["Code"] = fr["Code"].astype(str)
    fr["Date"] = pd.to_datetime(fr["Date"])
    wide, info = load_scores(lab.DATA_DIR)
    log("買いのモデルの out-of-fold: " + " / ".join(f"{a} {v['rows']:,}行" for a, v in info.items())
        + f"（重複 {sum(v['dups'] for v in info.values())}）")
    flags, folds = rule_flags(wide, fr["Date"])
    log(f"前向きの百分位を付けた行 {len(flags):,}（テスト窓 {flags['window'].nunique()}つ）・規則を満たした行 {int(flags['r0'].sum()):,}")
    bars = LT.load_bars(lab.DATA_DIR, start=fr["Date"].min() - pd.Timedelta(days=90), extra=("AdjL", "AdjVo"))
    topix = B.load_parts("topix", lab.DATA_DIR)
    keys = fr[["Code", "Date", "vol_20d"]].drop_duplicates(["Code", "Date"]).reset_index(drop=True)
    P, pinfo = build_paths(keys, bars, topix, keys["vol_20d"].to_numpy(dtype=float))
    sc = P.keys[["Code", "Date"]].merge(wide, on=["Code", "Date"], how="left")
    P.scores = sc[SCORE_COLS].reset_index(drop=True)
    print("\n■ 0. データ")
    print(f"  候補 {pinfo['rows']:,}件 → 使える取引 {pinfo['ok']:,}件（選定日の日足なし {pinfo['no_bar']:,}・20日目まで日足が無い "
          f"{pinfo['short']:,}・値の付かない日がある {pinfo['gap']:,}）/ 買いのスコアがある取引 {int(P.scores['score_lgbm'].notna().sum()):,}")
    ev_rows = P.keys[["Code", "Date"]].assign(_p=np.arange(len(P))).merge(flags[["Code", "Date"]], on=["Code", "Date"])["_p"].to_numpy()
    print(f"  評価の行のうち使える取引 {len(ev_rows):,} / {len(flags):,}")
    chk = check_against_live_track(P, bars, ev_rows)
    del bars
    ev, record = run_windows(P, flags, folds, with_ref=not args.no_ref, max_windows=args.max_windows,
                             with_b=args.with_b, train_only=args.train_only)
    os.makedirs(os.path.join(lab.DATA_DIR, "oof"), exist_ok=True)
    if args.train_only:
        train_report(record)
        path = os.path.join(lab.DATA_DIR, "oof", "e131_train_only.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"record": record, "paths": pinfo, "check": chk, "params": PARAMS, "deltas": list(DELTAS)}, fh,
                      ensure_ascii=False, indent=1, default=str)
        log(f"学習の中の確かめだけ（テスト窓の取引には当てていない）。記録: {path} / 全体 {time.time() - t0:.0f}秒")
        return 0
    if not len(ev):
        raise SystemExit("評価の行が無い")
    bad = check_oracle(ev)
    print(f"  確かめ: 上限（先が見えた場合）より高い損益の行 {bad}")
    if bad:
        raise SystemExit("上限より高い損益がある（損益の数え方が合っていない）")
    summary = report(ev, record, args.perm)
    summary.update(paths=pinfo, check=chk, params=PARAMS, state=STATE_COLS)
    os.makedirs(os.path.join(lab.DATA_DIR, "oof"), exist_ok=True)
    path = os.path.join(lab.DATA_DIR, "oof", "e131_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
