#!/usr/bin/env python3
"""
非線形時系列解析の特徴量（候補の群 `nl_*`、32列）。

母集団は 78週ぶりの高値更新日なので、どの行も「高値からの距離」は同じで、残る違いは
「どんな値動きを経てここへ来たか」である。本番の239列のうち株価系列そのものから作る列は
r_high×3・breakout 5列・vol_factors 6列だけで、いずれも**分散（ボラ）か水準**の指標。
5モデルの最大の依存は「自分の平常より静かな銘柄」（vol_rel_long）だが、同じスコア帯・
同じボラ帯の中では当たりと外れを分ける列がほとんど無い（docs/MODEL_ADOPTION_RULES.md §13）。
ここで足すのは、分散では見えない**系列の構造**（持続性・複雑さ・非対称性・再帰性・
価格と出来高の結合）を測る列。

  持続性（長期記憶）   DFA の α（Peng ら 1994）、分散比（Lo & MacKinlay 1988）、自己相関
  複雑さ（エントロピー）順列エントロピー（Bandt & Pompe 2002）、サンプルエントロピー
                       （Richman & Moorman 2000）、Lempel-Ziv 複雑さ（Kaspar & Schuster 1987）、
                       スペクトルエントロピー
  非線形性・非対称性   時間反転非対称性と c3（Schreiber & Schmitz 1997）、歪度・尖度、
                       BDS 統計量（Brock, Dechert, Scheinkman & LeBaron 1996）
  再帰性（位相空間）   リカレンス定量化解析の DET / LAM / ENTR（Webber & Zbilut 1994、
                       Marwan ら 2007）。埋め込みは Takens（1981）
  経路の形             Higuchi のフラクタル次元（1988）、Kaufman の効率比、MA20 の横切り回数、
                       ゼロリターン日の割合（Lesmond, Ogden & Trzcinka 1999）
  出来高・売買代金     売買代金系列の DFA / 順列エントロピー / サンプルエントロピー、
                       バースト性（Goh & Barabási 2008）、|リターン| と売買代金の相互情報量、
                       リターンと売買代金の変化の順位相関

すべて**基準日 t までの足だけ**で作る（先読み無し。tests/test_nonlinear_features.py で固定）。
窓は直近 60 / 120 / 250 営業日。欠測の足は窓から落とし、残りが窓の 8 割に満たなければ欠測。
値が無い・定義できないときは NaN（0 で埋めない。実測値と欠測を混ぜない）。

依存は numpy / pandas だけ（research/requirements.txt の範囲）。1行あたり 120×120 程度の
距離行列を数枚作るので、全行（約2.2万）で数分。銘柄ごとに並列化できる。

使い方（データセットの行に列を足す）:

    import nonlinear_features as NL
    nl = NL.attach_nonlinear(samples[["Code", "Date"]], bars, workers=4)
    samples = samples.merge(nl, on=["Code", "Date"], how="left")

    python3 research/nonlinear_features.py            # dataset.parquet の行ぶんを計算して保存
      -> research/_data/nonlinear_features.parquet
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
OUT_PATH = os.path.join(DATA_DIR, "nonlinear_features.parquet")

#: 窓（営業日）
W_SHORT, W_MID, W_LONG = 60, 120, 250
#: 窓の中で値がある足がこの割合に満たなければ、その窓の列は欠測
MIN_COVERAGE = 0.8
#: 1行に要る足の本数（250 リターン = 251 本の終値）
N_BARS = W_LONG + 1

#: 候補の列（順序は固定。features.GROUPS に入れるときもこの並び）
NL_COLS: List[str] = [
    # --- 持続性（長期記憶）: 日次ログリターン r ---
    "nl_dfa_r120", "nl_dfa_r250", "nl_dfa_abs120", "nl_vr5_120",
    "nl_acf1_60", "nl_acf_sq1_60",
    # --- 複雑さ（エントロピー） ---
    "nl_pe3_60", "nl_pe4_120", "nl_sampen_120", "nl_lz_120", "nl_spec_ent_120",
    # --- 非線形性・非対称性 ---
    "nl_tra1_120", "nl_c3_120", "nl_skew_60", "nl_kurt_60", "nl_bds2_120",
    # --- 再帰性（位相空間、m=3） ---
    "nl_rqa_det_120", "nl_rqa_lam_120", "nl_rqa_entr_120",
    # --- 経路の形（終値） ---
    "nl_higuchi_120", "nl_er_60", "nl_ma20x_120", "nl_zero_ret_60",
    # --- 自分の過去との差 ---
    "nl_pe3_60_vs_prior", "nl_dfa_r120_vs_prior",
    # --- 出来高・売買代金 ---
    "nl_tv_dfa_120", "nl_tv_pe3_60", "nl_tv_sampen_120", "nl_tv_burst_60",
    "nl_pv_mi_60", "nl_pv_rho_60", "nl_pv_absrho_60",
]

#: 列の意味（feature_dict / docs 用）
NL_DESC: Dict[str, str] = {
    "nl_dfa_r120": "日次ログリターンの DFA 指数 α（120日）。0.5 がランダム、大きいほど持続（トレンド）、小さいほど反転",
    "nl_dfa_r250": "同 250日",
    "nl_dfa_abs120": "|リターン| の DFA 指数 α（120日）。ボラの塊（クラスタリング）の強さ",
    "nl_vr5_120": "分散比 Var(5日リターン)÷(5×Var(1日))（120日）。1 がランダムウォーク、>1 持続、<1 反転",
    "nl_acf1_60": "リターンの1次自己相関（60日）",
    "nl_acf_sq1_60": "リターンの2乗の1次自己相関（60日）。ARCH 効果",
    "nl_pe3_60": "リターンの順列エントロピー（m=3、60日、0〜1）。1 がランダム、小さいほど順序に型がある",
    "nl_pe4_120": "同 m=4、120日",
    "nl_sampen_120": "標準化リターンのサンプルエントロピー（m=2、r=0.2σ、120日）。小さいほど規則的",
    "nl_lz_120": "リターンの符号列の Lempel-Ziv 複雑さ（正規化、120日）。1 付近がランダム",
    "nl_spec_ent_120": "リターンのスペクトルエントロピー（120日、0〜1）。1 がホワイトノイズ",
    "nl_tra1_120": "時間反転非対称性 E[r(t+1)²r(t) − r(t+1)r(t)²]/σ³（120日）。0 から離れるほど非可逆（非線形）",
    "nl_c3_120": "c3 統計量 E[r(t+2)r(t+1)r(t)]/σ³（120日）。非線形性の指標",
    "nl_skew_60": "リターンの歪度（60日）",
    "nl_kurt_60": "リターンの超過尖度（60日）。跳びの多さ",
    "nl_bds2_120": "BDS 統計量（m=2、ε=σ、120日）。i.i.d. からのずれ（非線形依存）の z 値",
    "nl_rqa_det_120": "リカレンスプロットの決定性 DET（m=3、再帰率 10%、120日）。対角線に乗る割合",
    "nl_rqa_lam_120": "同 層流性 LAM。縦線に乗る割合（同じ状態に留まる＝張り付き）",
    "nl_rqa_entr_120": "同 対角線の長さ分布のエントロピー ENTR",
    "nl_higuchi_120": "ログ終値の Higuchi フラクタル次元（kmax=10、120日）。1 が滑らか、2 に近いほどギザギザ",
    "nl_er_60": "Kaufman の効率比 |終値の差| ÷ Σ|日々の差|（60日）。1 が一直線、0 が往復",
    "nl_ma20x_120": "終値が20日移動平均を横切った回数 ÷ 120（120日）。もみ合いの度合い",
    "nl_zero_ret_60": "リターンがちょうど 0 の日の割合（60日）。値が張り付いている（TOB・不出来）",
    "nl_pe3_60_vs_prior": "nl_pe3_60 − その前の 190日の順列エントロピー。最近になって型が出てきたか",
    "nl_dfa_r120_vs_prior": "nl_dfa_r120 − その前の 120日の α。持続性が最近強まったか",
    "nl_tv_dfa_120": "ログ売買代金の DFA 指数 α（120日）。売買代金の持続性",
    "nl_tv_pe3_60": "ログ売買代金の順列エントロピー（m=3、60日）",
    "nl_tv_sampen_120": "標準化ログ売買代金のサンプルエントロピー（m=2、r=0.2σ、120日）",
    "nl_tv_burst_60": "売買代金のバースト性 (σ−μ)/(σ+μ)（60日）。−1 周期的、0 ポアソン、1 に近いほど突発",
    "nl_pv_mi_60": "|リターン| とログ売買代金の相互情報量（3×3 分位ビン、60日、0〜1）。価格と出来高の結合",
    "nl_pv_rho_60": "リターンと売買代金の変化（ログ差）の順位相関（60日）。上げの日に商いが増えるか",
    "nl_pv_absrho_60": "|リターン| とログ売買代金の順位相関（60日）。動いた日に商いが伴うか",
}
assert set(NL_DESC) == set(NL_COLS) and len(NL_COLS) == len(set(NL_COLS))

_LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# 個々の指標（1次元の numpy 配列を受け取り float を返す。定義できなければ NaN）
# --------------------------------------------------------------------------- #

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _enough(x: np.ndarray, w: int) -> bool:
    return len(x) >= max(8, int(math.ceil(w * MIN_COVERAGE)))


def dfa_alpha(x: np.ndarray, scales: Sequence[int]) -> float:
    """
    DFA-1（Peng ら 1994）のスケーリング指数 α。

    プロファイル（平均を引いた累積和）を長さ s の箱に切り、各箱を1次で除いた残差の
    RMS F(s) を取り、log F(s) 対 log s の傾きを返す。箱は前からと後ろからの両方を使う。
    白色雑音で 0.5、ランダムウォークの増分の累積（= ランダムウォーク）で 1.5。
    """
    x = _finite(x)
    n = len(x)
    if n < 16 or np.std(x) == 0:
        return float("nan")
    y = np.cumsum(x - x.mean())
    ss, fs = [], []
    for s in scales:
        nb = n // s
        if nb < 2:
            continue
        t = np.arange(s, dtype=float)
        t -= t.mean()
        tt = (t * t).sum()
        f2 = 0.0
        for seg in (y[:nb * s].reshape(nb, s), y[n - nb * s:].reshape(nb, s)):
            slope = (seg * t).sum(axis=1) / tt
            resid = seg - seg.mean(axis=1, keepdims=True) - slope[:, None] * t
            f2 += (resid ** 2).mean()
        f = math.sqrt(f2 / 2.0)
        if f > 0:
            ss.append(math.log(s))
            fs.append(math.log(f))
    if len(ss) < 3:
        return float("nan")
    return float(np.polyfit(ss, fs, 1)[0])


DFA_SCALES_120 = (4, 5, 6, 8, 10, 12, 15, 20, 24, 30)
DFA_SCALES_250 = (4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 62)


def variance_ratio(x: np.ndarray, q: int = 5) -> float:
    """Lo & MacKinlay（1988）の分散比 Var(q日和)÷(q×Var(1日))。重なりのある q 日和を使う。"""
    x = _finite(x)
    if len(x) < 4 * q:
        return float("nan")
    v1 = x.var(ddof=1)
    if v1 <= 0:
        return float("nan")
    xq = np.convolve(x, np.ones(q), mode="valid")
    return float(xq.var(ddof=1) / (q * v1))


def autocorr(x: np.ndarray, lag: int = 1) -> float:
    x = _finite(x)
    if len(x) <= lag + 2:
        return float("nan")
    x = x - x.mean()
    d = (x * x).sum()
    if d <= 0:
        return float("nan")
    return float((x[lag:] * x[:-lag]).sum() / d)


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    Bandt & Pompe（2002）の順列エントロピー。log(m!) で割って 0〜1。

    同値は現れた順（stable な argsort）で順位を決める。ほとんどの日がリターン 0 の系列は
    1つの型に集中して 0 に近づく（張り付きとして正しい読み）。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x) - (m - 1) * tau
    if n < 2 * math.factorial(m):
        return float("nan")
    idx = np.arange(n)[:, None] + np.arange(m)[None, :] * tau
    pats = np.argsort(x[idx], axis=1, kind="stable")
    codes = (pats * (m ** np.arange(m))).sum(axis=1)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Richman & Moorman（2000）のサンプルエントロピー −log(A/B)。
    x は標準化してから r を σ の倍数として使う。σ=0 なら定義しない（NaN）。
    """
    x = _finite(x)
    n = len(x)
    if n < 4 * (m + 1):
        return float("nan")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    z = (x - x.mean()) / sd
    n_t = n - m          # 長さ m と m+1 の両方で同じ本数のテンプレートを使う

    def pairs(mm: int) -> float:
        X = np.lib.stride_tricks.sliding_window_view(z, mm)[:n_t]
        D = np.abs(X[:, None, :] - X[None, :, :]).max(axis=-1)
        return float(((D <= r).sum() - n_t) / 2.0)

    B = pairs(m)
    A = pairs(m + 1)
    if A <= 0 or B <= 0:
        return float("nan")
    return float(-math.log(A / B))


def lempel_ziv(bits: np.ndarray) -> float:
    """
    Kaspar & Schuster（1987）の手順で数えた Lempel-Ziv 複雑さ c(n) を n/log2(n) で正規化。
    ランダムな0/1列で 1 付近、周期的な列で 0 に近づく。
    """
    s = [int(b) for b in np.asarray(bits).ravel() if b == 0 or b == 1]
    n = len(s)
    if n < 16:
        return float("nan")
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return float(c / (n / (math.log(n) / _LOG2)))


def spectral_entropy(x: np.ndarray) -> float:
    """ピリオドグラム（直流を除く）の正規化シャノンエントロピー。白色雑音で 1 に近い。"""
    x = _finite(x)
    if len(x) < 16 or np.std(x) == 0:
        return float("nan")
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    psd = psd[1:]
    tot = psd.sum()
    if tot <= 0:
        return float("nan")
    p = psd / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(psd)))


def time_reversal_asymmetry(x: np.ndarray, lag: int = 1) -> float:
    """Schreiber & Schmitz（1997）の時間反転非対称性 E[x(t+τ)²x(t) − x(t+τ)x(t)²] / σ³。"""
    x = _finite(x)
    if len(x) <= lag + 8:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    a, b = x[lag:], x[:-lag]
    return float(np.mean(a * a * b - a * b * b) / sd ** 3)


def c3(x: np.ndarray, lag: int = 1) -> float:
    """Schreiber & Schmitz（1997）の c3 = E[x(t+2τ)x(t+τ)x(t)] / σ³。"""
    x = _finite(x)
    if len(x) <= 2 * lag + 8:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.mean(x[2 * lag:] * x[lag:-lag] * x[:-2 * lag]) / sd ** 3)


def skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    x = _finite(x)
    if len(x) < 8:
        return float("nan"), float("nan")
    sd = x.std()
    if sd <= 0:
        return float("nan"), float("nan")
    z = (x - x.mean()) / sd
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def bds_statistic(x: np.ndarray, m: int = 2, eps_sd: float = 1.0) -> float:
    """
    BDS 統計量（Brock, Dechert, Scheinkman & LeBaron 1996）、m=2、ε = eps_sd × σ。

    W = √n (C_m(ε) − C_1(ε)^m) / σ_m。i.i.d. なら漸近的に N(0,1)。正に大きいほど
    近い値が続く（非線形依存・ボラの塊）。C と K（3点が互いに ε 内にある確率）は
    statsmodels.tsa.stattools.bds と同じ数え方（自分自身の対は除く）。
    """
    x = _finite(x)
    n = len(x)
    if n < 40:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    eps = eps_sd * sd
    I = (np.abs(x[:, None] - x[None, :]) <= eps)
    np.fill_diagonal(I, False)
    nf = n - m + 1                       # 埋め込み後の点の数。C_1 もこの点数で測る
    Ifull = I[-nf:, -nf:] if False else I[(n - nf):, (n - nf):]
    c1 = Ifull.sum() / (nf * (nf - 1))
    # m=2: 隣り合う2点がどちらも ε 内
    J = I[:-1, :-1] & I[1:, 1:]
    cm = J.sum() / (nf * (nf - 1))
    rows = Ifull.sum(axis=1).astype(float)
    k = ((rows ** 2).sum() - Ifull.sum()) / (nf * (nf - 1) * (nf - 2))
    sig2 = 4.0 * (k ** m + 2.0 * sum(k ** (m - j) * c1 ** (2 * j) for j in range(1, m))
                  + (m - 1) ** 2 * c1 ** (2 * m) - m * m * k * c1 ** (2 * m - 2))
    if not np.isfinite(sig2) or sig2 <= 0:
        return float("nan")
    return float(math.sqrt(nf) * (cm - c1 ** m) / math.sqrt(sig2))


def _run_lengths(rows: np.ndarray) -> np.ndarray:
    """2値行列の各行の True の連の長さをまとめて返す。"""
    if rows.size == 0:
        return np.zeros(0, dtype=int)
    p = np.pad(rows.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(p, axis=1)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts


def rqa(x: np.ndarray, m: int = 3, tau: int = 1, rr: float = 0.10,
        lmin: int = 2) -> Tuple[float, float, float]:
    """
    リカレンス定量化解析（Webber & Zbilut 1994、Marwan ら 2007）。

    標準化した x を Takens 埋め込み（m, τ）し、チェビシェフ距離の下位 rr 分位を再帰の
    しきい値にする（再帰率を固定し、銘柄間でボラの違いが混ざらないようにする）。
    主対角線は除く。返すのは DET（長さ lmin 以上の対角線に乗る再帰点の割合）、
    LAM（同じく縦線）、ENTR（対角線の長さ分布のシャノンエントロピー、nat）。
    """
    x = _finite(x)
    n = len(x) - (m - 1) * tau
    if n < 30:
        return (float("nan"),) * 3
    sd = x.std(ddof=1)
    if sd <= 0:
        return (float("nan"),) * 3
    z = (x - x.mean()) / sd
    idx = np.arange(n)[:, None] + np.arange(m)[None, :] * tau
    X = z[idx]
    D = np.abs(X[:, None, :] - X[None, :, :]).max(axis=-1)
    iu = np.triu_indices(n, k=1)
    eps = np.quantile(D[iu], rr)
    R = D <= eps
    np.fill_diagonal(R, False)
    # 対角線（上三角。k 本目の対角線を行 k-1 に並べる）
    kk, ii = np.meshgrid(np.arange(1, n), np.arange(n - 1), indexing="ij")
    valid = ii + kk < n
    Dg = np.zeros((n - 1, n - 1), dtype=bool)
    Dg[valid] = R[ii[valid], ii[valid] + kk[valid]]
    dl = _run_lengths(Dg)
    tot = dl.sum()
    if tot <= 0:
        return (float("nan"),) * 3
    long = dl[dl >= lmin]
    det = float(long.sum() / tot)
    if len(long):
        _, cnt = np.unique(long, return_counts=True)
        p = cnt / cnt.sum()
        entr = float(-(p * np.log(p)).sum())
    else:
        entr = 0.0
    vl = _run_lengths(R.T)        # 列方向の連 = 縦線
    vtot = vl.sum()
    lam = float(vl[vl >= lmin].sum() / vtot) if vtot > 0 else float("nan")
    return det, lam, entr


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi（1988）のフラクタル次元。1 が滑らかな曲線、2 に近いほど荒い。"""
    x = _finite(x)
    n = len(x)
    if n < 4 * kmax:
        return float("nan")
    lk, ks = [], []
    for k in range(1, kmax + 1):
        lm = []
        for m0 in range(k):
            idx = np.arange(m0, n, k)
            if len(idx) < 2:
                continue
            lm.append(np.abs(np.diff(x[idx])).sum() * (n - 1) / ((len(idx) - 1) * k) / k)
        if lm and np.mean(lm) > 0:
            lk.append(math.log(np.mean(lm)))
            ks.append(math.log(1.0 / k))
    if len(ks) < 3:
        return float("nan")
    return float(np.polyfit(ks, lk, 1)[0])


def efficiency_ratio(close: np.ndarray) -> float:
    """Kaufman の効率比 |終点 − 始点| ÷ Σ|日々の差|。1 が一直線、0 が往復。"""
    c = _finite(close)
    if len(c) < 8:
        return float("nan")
    path = np.abs(np.diff(c)).sum()
    return float(abs(c[-1] - c[0]) / path) if path > 0 else float("nan")


def ma_crossings(close: np.ndarray, window: int = 20, span: int = W_MID) -> float:
    """終値が window 日移動平均を横切った回数 ÷ span。"""
    c = _finite(close)
    if len(c) < window + span // 2:
        return float("nan")
    ma = np.convolve(c, np.ones(window) / window, mode="valid")
    diff = c[window - 1:] - ma
    diff = diff[-span:]
    s = np.sign(diff)
    s = s[s != 0]
    if len(s) < 8:
        return float("nan")
    return float((np.diff(s) != 0).sum() / span)


def burstiness(v: np.ndarray) -> float:
    """Goh & Barabási（2008）のバースト性 (σ−μ)/(σ+μ)。"""
    v = _finite(v)
    if len(v) < 8:
        return float("nan")
    mu, sd = v.mean(), v.std(ddof=1)
    return float((sd - mu) / (sd + mu)) if (sd + mu) > 0 else float("nan")


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average").to_numpy(dtype=float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 3) -> float:
    """分位ビン（既定 3×3）で離散化した相互情報量。log(bins) で割って 0〜1。"""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6 * bins:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    n = ok.sum()
    qa = np.minimum((ra - 1) * bins // n, bins - 1).astype(int)
    qb = np.minimum((rb - 1) * bins // n, bins - 1).astype(int)
    tab = np.zeros((bins, bins))
    np.add.at(tab, (qa, qb), 1.0)
    p = tab / n
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    return float(max(mi, 0.0) / math.log(bins))


# --------------------------------------------------------------------------- #
# 1行ぶんの窓 → 32列
# --------------------------------------------------------------------------- #

def _tail(x: np.ndarray, w: int) -> np.ndarray:
    """
    暦どおりの末尾 w 本から欠測を落としたもの。値のある足が w の 8 割に満たなければ空。

    「直近の w 本の値がある足」ではない。長く売買の無い銘柄で1年前の足まで遡って
    埋めると、基準日の性質ではなくなる。
    """
    f = _finite(np.asarray(x, dtype=float)[-w:])
    return f if _enough(f, w) else np.zeros(0)


def features_for_window(close: np.ndarray, value: np.ndarray) -> Dict[str, float]:
    """
    基準日 t を末尾とする終値と売買代金の窓（最大 N_BARS 本）から 32列を作る。

    close : 分割調整後の終値（欠測は NaN）
    value : 売買代金（円。欠測は NaN、売買が無い日は 0 でも NaN でもよい）
    """
    out = {c: float("nan") for c in NL_COLS}
    close = np.asarray(close, dtype=float)[-N_BARS:]
    value = np.asarray(value, dtype=float)[-N_BARS:]
    if len(close) < 2:
        return out
    # 位置をそろえたまま扱う（i 番目は基準日から数えて同じ日）。欠測の日のリターンは NaN、
    # 欠測の翌日のリターンは欠測をまたいだ複数日ぶんの変化になる
    okc = np.isfinite(close) & (close > 0)
    lc = np.where(okc, np.log(np.where(okc, close, 1.0)), np.nan)
    lc_ff = pd.Series(lc).ffill().to_numpy(dtype=float)
    r = np.diff(lc_ff)
    r[~okc[1:]] = np.nan                 # その日に値が無ければリターンも無い
    r60, r120, r250 = _tail(r, W_SHORT), _tail(r, W_MID), _tail(r, W_LONG)

    if len(r120):
        out["nl_dfa_r120"] = dfa_alpha(r120, DFA_SCALES_120)
        out["nl_dfa_abs120"] = dfa_alpha(np.abs(r120), DFA_SCALES_120)
        out["nl_vr5_120"] = variance_ratio(r120, 5)
        out["nl_pe4_120"] = permutation_entropy(r120, m=4)
        out["nl_sampen_120"] = sample_entropy(r120, m=2, r=0.2)
        out["nl_lz_120"] = lempel_ziv((r120 > 0).astype(int))
        out["nl_spec_ent_120"] = spectral_entropy(r120)
        out["nl_tra1_120"] = time_reversal_asymmetry(r120, 1)
        out["nl_c3_120"] = c3(r120, 1)
        out["nl_bds2_120"] = bds_statistic(r120, m=2, eps_sd=1.0)
        det, lam, entr = rqa(r120, m=3, tau=1, rr=0.10)
        out["nl_rqa_det_120"], out["nl_rqa_lam_120"], out["nl_rqa_entr_120"] = det, lam, entr
    if len(r250):
        out["nl_dfa_r250"] = dfa_alpha(r250, DFA_SCALES_250)
    if len(r60):
        out["nl_acf1_60"] = autocorr(r60, 1)
        out["nl_acf_sq1_60"] = autocorr(r60 ** 2, 1)
        out["nl_pe3_60"] = permutation_entropy(r60, m=3)
        out["nl_skew_60"], out["nl_kurt_60"] = skew_kurt(r60)
        out["nl_zero_ret_60"] = float((r60 == 0).mean())

    # 自分の過去との差（前の窓は「直近の窓を除いた、その前」）
    if len(r) >= W_LONG and len(r60):
        prior = _finite(r[-W_LONG:-W_SHORT])
        if _enough(prior, W_LONG - W_SHORT):
            out["nl_pe3_60_vs_prior"] = out["nl_pe3_60"] - permutation_entropy(prior, m=3)
    if len(r) >= 2 * W_MID and len(r120):
        prior = _finite(r[-2 * W_MID:-W_MID])
        if _enough(prior, W_MID):
            out["nl_dfa_r120_vs_prior"] = out["nl_dfa_r120"] - dfa_alpha(prior, DFA_SCALES_120)

    # 経路の形（終値そのもの）
    c120 = _tail(lc, W_MID + 1)
    if len(c120):
        out["nl_higuchi_120"] = higuchi_fd(c120, kmax=10)
    c60 = _tail(lc, W_SHORT + 1)
    if len(c60):
        out["nl_er_60"] = efficiency_ratio(np.exp(c60))
    c140 = _tail(lc, W_MID + 20)
    if len(c140):
        out["nl_ma20x_120"] = ma_crossings(np.exp(c140), window=20, span=W_MID)

    # 売買代金（売買の無い日は欠測として落とす。位置は終値と同じ）
    pos = np.isfinite(value) & (value > 0)
    lv = np.where(pos, np.log(np.where(pos, value, 1.0)), np.nan)
    lv120, lv60 = _tail(lv, W_MID), _tail(lv, W_SHORT)
    if len(lv120):
        out["nl_tv_dfa_120"] = dfa_alpha(lv120, DFA_SCALES_120)
        out["nl_tv_sampen_120"] = sample_entropy(lv120, m=2, r=0.2)
    if len(lv60):
        out["nl_tv_pe3_60"] = permutation_entropy(lv60, m=3)
    raw60 = np.where(np.isfinite(value), value, 0.0)[-W_SHORT:]
    if len(raw60) >= int(W_SHORT * MIN_COVERAGE) and okc[-W_SHORT:].sum() >= int(W_SHORT * MIN_COVERAGE):
        out["nl_tv_burst_60"] = burstiness(raw60)
    # 価格と出来高の結合（同じ日の対で。lv[1:] が r と同じ日）
    if len(r) >= W_SHORT:
        rr = r[-W_SHORT:]
        lv_same = lv[1:][-W_SHORT:]
        dlv = np.diff(lv)[-W_SHORT:]
        out["nl_pv_mi_60"] = mutual_information(np.abs(rr), lv_same, bins=3)
        out["nl_pv_absrho_60"] = spearman(np.abs(rr), lv_same)
        out["nl_pv_rho_60"] = spearman(rr, dlv)
    for k, v in out.items():
        if v is not None and not np.isfinite(v):
            out[k] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# データセットの行に付ける
# --------------------------------------------------------------------------- #

def _bars_by_code(bars: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    b = bars.copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str).str.strip()
    close = b["AdjC"].fillna(b["C"]) if "AdjC" in b.columns else b["C"]
    b["_close"] = pd.to_numeric(close, errors="coerce")
    if "Va" in b.columns:
        b["_value"] = pd.to_numeric(b["Va"], errors="coerce")
    else:
        b["_value"] = pd.to_numeric(b["C"], errors="coerce") * pd.to_numeric(b["Vo"], errors="coerce")
    b = b.sort_values(["Code", "Date"])
    out = {}
    for code, g in b.groupby("Code", sort=False):
        out[code] = (g["Date"].to_numpy(dtype="datetime64[ns]"),
                     g["_close"].to_numpy(dtype=float), g["_value"].to_numpy(dtype=float))
    return out


def _compute_code(args) -> List[dict]:
    code, dates, close, value, targets = args
    rows = []
    for d in targets:
        pos = int(np.searchsorted(dates, d))
        if pos >= len(dates) or dates[pos] != d:
            rec = {c: float("nan") for c in NL_COLS}
        else:
            lo = max(0, pos - N_BARS + 1)
            rec = features_for_window(close[lo:pos + 1], value[lo:pos + 1])
        rec["Code"] = code
        rec["Date"] = d
        rows.append(rec)
    return rows


def attach_nonlinear(keys: pd.DataFrame, bars: pd.DataFrame, workers: int = 1,
                     log=None) -> pd.DataFrame:
    """
    keys（Code / Date）の各行に、その日までの足から作った 32列を付けて返す。
    bars は Date / Code / C（AdjC があれば優先）/ Va（無ければ C×Vo）を持つ日次バー。
    """
    k = keys[["Code", "Date"]].drop_duplicates().copy()
    k["Code"] = k["Code"].astype(str).str.strip()
    k["Date"] = pd.to_datetime(k["Date"]).to_numpy(dtype="datetime64[ns]")
    by = _bars_by_code(bars)
    jobs = []
    for code, g in k.groupby("Code", sort=False):
        targets = np.sort(g["Date"].to_numpy(dtype="datetime64[ns]"))
        if code in by:
            dates, close, value = by[code]
        else:
            dates, close, value = (np.zeros(0, dtype="datetime64[ns]"), np.zeros(0), np.zeros(0))
        jobs.append((code, dates, close, value, targets))
    t0 = time.time()
    rows: List[dict] = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, part in enumerate(ex.map(_compute_code, jobs, chunksize=8)):
                rows.extend(part)
                if log and (i + 1) % 500 == 0:
                    log(f"  {i + 1:,}/{len(jobs):,} 銘柄 {time.time() - t0:.0f}秒")
    else:
        for i, j in enumerate(jobs):
            rows.extend(_compute_code(j))
            if log and (i + 1) % 500 == 0:
                log(f"  {i + 1:,}/{len(jobs):,} 銘柄 {time.time() - t0:.0f}秒")
    out = pd.DataFrame(rows, columns=["Code", "Date"] + NL_COLS)
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def load_bars(data_dir: str = DATA_DIR, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or ["Date", "Code", "C", "AdjC", "Vo", "Va"]
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_[0-9]*.parquet")))
    if not paths:
        raise SystemExit(f"bars_*.parquet がありません: {data_dir}")
    parts = []
    for p in paths:
        df = pd.read_parquet(p)
        parts.append(df[[c for c in cols if c in df.columns]])
    return pd.concat(parts, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="非線形時系列解析の特徴量（nl_*）を計算して保存する")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--limit", type=int, default=0, help="先頭 n 行だけ（動作確認用）")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    ds = pd.read_parquet(args.dataset, columns=["Code", "Date"])
    if args.limit:
        ds = ds.head(args.limit)
    log(f"対象 {len(ds):,}行 / 銘柄 {ds['Code'].nunique():,}")
    bars = load_bars(args.data_dir)
    log(f"日次バー {len(bars):,}行")
    out = attach_nonlinear(ds, bars, workers=args.workers, log=log)
    miss = out[NL_COLS].isna().mean() * 100
    log("欠測率: " + " ".join(f"{c}={v:.1f}%" for c, v in miss.items() if v > 5)
        + (" （5% 超の列なし）" if (miss <= 5).all() else ""))
    out.to_parquet(args.out, index=False)
    log(f"保存: {args.out} ({len(out):,}行 × {len(NL_COLS)}列)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
