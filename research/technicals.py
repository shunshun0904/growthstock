#!/usr/bin/env python3
"""
テクニカル指標（株価・出来高から作る111列。実験105 で検証。本番の列の組にはまだ入れていない）。

運用者の依頼（2026-10-18）「いま株価や出来高から作られている特徴量は27列だけです。ありとあらゆるテクニカルの手法、
オシレーター、トレンド系全てを使って特徴量を生成してほしい。100個くらいは作れるはず」。

入力（compute の panel）: 銘柄・日付の順に並んだ日足。Code・Date・open・high・low・close・vol（分割を調整した値。
build_dataset.price_panel の列）と、あれば trading_value（売買代金。素の終値 × 出来高）。

数え方
- **売買の成立した行（終値がある行）だけで数える。** n 日 = その銘柄の n 回の取引日。終値の無い行（売買不成立）は計算に
  使わず、出力も欠測。始値・高値・安値が欠けた行は終値で埋め、高値は始値・終値以上、安値は以下にそろえる
- どの値も **t の引けまでの値だけ** で作る（t より先は見ない。tests/test_technicals.py が「先の行を消しても値が変わらない」
  ことを確かめる）。銘柄をまたいだ計算はしない（同じく「1銘柄ずつ計算しても同じ」ことを確かめる）
- 窓の長さ n の指標は、その銘柄の取引日が n 日そろってから値を出す（それまでは欠測）。指数平滑（EMA・Wilder の平滑）は
  最初の値から始め（pandas の ewm、adjust=False）、平滑の長さ分の取引日がたまるまでは欠測にする
- 値段の単位が残る指標（MACD・ATR・移動平均との差など）は終値で割って %（銘柄どうしで比べられるように）。出来高の単位が
  残る指標は出来高の合計・平均で割る
- 名前はすべて tx_ で始める（既存の列と重ならない）

系統（GROUPS）: trend 41 / oscillator 26 / volatility 14 / volume 18 / candle 12 = 111列。定義は下の各関数と JA の名前。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

GROUPS: Dict[str, List[str]] = {
    "trend": (
        [f"tx_ma_gap_{n}" for n in (5, 10, 25, 50, 75, 100, 200)]
        + [f"tx_ma_slope_{n}" for n in (25, 75, 200)]
        + ["tx_ma_ratio_5_25", "tx_ma_ratio_25_75", "tx_ma_ratio_50_200", "tx_perfect_order", "tx_ma_cross_days",
           "tx_macd", "tx_macd_signal", "tx_macd_hist", "tx_adx_14", "tx_pdi_14", "tx_mdi_14",
           "tx_aroon_up_25", "tx_aroon_down_25", "tx_psar_gap", "tx_psar_days",
           "tx_ichi_tk", "tx_ichi_kijun_gap", "tx_ichi_cloud_pos", "tx_ichi_cloud_thick", "tx_ichi_chikou",
           "tx_ichi_sanyaku", "tx_linreg_slope_20", "tx_linreg_slope_60", "tx_linreg_r2_60", "tx_kaufman_er_20",
           "tx_trix_15", "tx_vortex_pos_14", "tx_vortex_neg_14", "tx_supertrend_gap",
           "tx_donchian_pos_20", "tx_donchian_pos_55"]),
    "oscillator": [
        "tx_rsi_9", "tx_rsi_14", "tx_rsi_25", "tx_stoch_k", "tx_stoch_d", "tx_stoch_rsi_14", "tx_williams_r_14",
        "tx_cci_20", "tx_roc_10", "tx_roc_60", "tx_roc_120", "tx_roc_250", "tx_ultimate_osc", "tx_cmo_14",
        "tx_tsi", "tx_dpo_20", "tx_kst", "tx_awesome_osc", "tx_rci_9", "tx_rci_26", "tx_psy_12", "tx_connors_rsi",
        "tx_fisher_10", "tx_elder_bull", "tx_elder_bear", "tx_bop_14"],
    "volatility": [
        "tx_natr_14", "tx_atr_ratio_14_50", "tx_bb_pctb", "tx_bb_width", "tx_bb_width_rank_120", "tx_keltner_pos",
        "tx_squeeze", "tx_vol_parkinson_20", "tx_vol_gk_20", "tx_vol_60", "tx_ulcer_14", "tx_chaikin_vol_10",
        "tx_mass_index_25", "tx_chop_14"],
    "volume": [
        "tx_obv_chg_20", "tx_obv_ma_gap", "tx_ad_ma_gap", "tx_cmf_20", "tx_chaikin_osc", "tx_mfi_14", "tx_force_13",
        "tx_eom_14", "tx_vwap_gap_20", "tx_vwap_gap_60", "tx_pvt_chg_20", "tx_vr_25", "tx_vol_osc_5_20",
        "tx_vol_ratio_60_250", "tx_up_vol_share_20", "tx_klinger", "tx_nvi_chg_60", "tx_tv_ratio_20_120"],
    "candle": [
        "tx_body_ratio", "tx_upper_shadow", "tx_lower_shadow", "tx_bull_share_20", "tx_candle_streak",
        "tx_close_streak", "tx_body_mean_20", "tx_gap_up_count_20", "tx_range_atr", "tx_engulfing", "tx_nr7",
        "tx_three_soldiers"],
}
COLUMNS: List[str] = [c for cols in GROUPS.values() for c in cols]
GROUP_OF: Dict[str, str] = {c: g for g, cols in GROUPS.items() for c in cols}
GROUP_JA = {"trend": "トレンド系", "oscillator": "オシレーター", "volatility": "ボラティリティ", "volume": "出来高系",
            "candle": "ローソク足"}

JA: Dict[str, str] = {
    **{f"tx_ma_gap_{n}": f"{n}日移動平均乖離率（%）" for n in (5, 10, 25, 50, 75, 100, 200)},
    **{f"tx_ma_slope_{n}": f"{n}日移動平均の傾き（5取引日前比 %）" for n in (25, 75, 200)},
    "tx_ma_ratio_5_25": "5日 ÷ 25日移動平均（%）", "tx_ma_ratio_25_75": "25日 ÷ 75日移動平均（%）",
    "tx_ma_ratio_50_200": "50日 ÷ 200日移動平均（%）", "tx_perfect_order": "パーフェクトオーダーの段数（5>25>75>200、0〜3）",
    "tx_ma_cross_days": "25日と75日の移動平均が交差してからの取引日数（上なら＋）",
    "tx_macd": "MACD（12・26）÷ 終値（%）", "tx_macd_signal": "MACD シグナル（9）÷ 終値（%）",
    "tx_macd_hist": "MACD ヒストグラム ÷ 終値（%）", "tx_adx_14": "ADX（14）", "tx_pdi_14": "+DI（14）", "tx_mdi_14": "−DI（14）",
    "tx_aroon_up_25": "アルーン・アップ（25）", "tx_aroon_down_25": "アルーン・ダウン（25）",
    "tx_psar_gap": "パラボリック SAR との距離（%）", "tx_psar_days": "パラボリック SAR が反転してからの取引日数（上昇なら＋）",
    "tx_ichi_tk": "一目均衡表 転換線 − 基準線（終値比 %）", "tx_ichi_kijun_gap": "一目均衡表 基準線との乖離（%）",
    "tx_ichi_cloud_pos": "一目均衡表 雲との位置（%。雲の中は 0）", "tx_ichi_cloud_thick": "一目均衡表 雲の厚さ（先行1 − 先行2、終値比 %）",
    "tx_ichi_chikou": "一目均衡表 遅行スパン（25取引日前の終値比 %）", "tx_ichi_sanyaku": "一目均衡表 三役好転の数（0〜3）",
    "tx_linreg_slope_20": "20日の対数価格の回帰の傾き（%／日）", "tx_linreg_slope_60": "60日の対数価格の回帰の傾き（%／日）",
    "tx_linreg_r2_60": "60日の対数価格の回帰の決定係数", "tx_kaufman_er_20": "効率レシオ（カウフマン、20）",
    "tx_trix_15": "TRIX（15）", "tx_vortex_pos_14": "ボルテックス VI+（14）", "tx_vortex_neg_14": "ボルテックス VI−（14）",
    "tx_supertrend_gap": "スーパートレンド（10・3）との距離（%）",
    "tx_donchian_pos_20": "ドンチャン・チャネルの中の位置（20、%）", "tx_donchian_pos_55": "ドンチャン・チャネルの中の位置（55、%）",
    "tx_rsi_9": "RSI（9）", "tx_rsi_14": "RSI（14）", "tx_rsi_25": "RSI（25）",
    "tx_stoch_k": "ストキャスティクス スロー %K（14・3）", "tx_stoch_d": "ストキャスティクス スロー %D（3）",
    "tx_stoch_rsi_14": "ストキャスティクス RSI（14）", "tx_williams_r_14": "ウィリアムズ %R（14）", "tx_cci_20": "CCI（20）",
    **{f"tx_roc_{n}": f"ROC（{n}日、%）" for n in (10, 60, 120, 250)},
    "tx_ultimate_osc": "アルティメット・オシレーター（7・14・28）", "tx_cmo_14": "CMO（14）", "tx_tsi": "TSI（25・13）",
    "tx_dpo_20": "DPO（20）÷ 終値（%）", "tx_kst": "KST", "tx_awesome_osc": "オーサム・オシレーター ÷ 終値（%）",
    "tx_rci_9": "RCI（9）", "tx_rci_26": "RCI（26）", "tx_psy_12": "サイコロジカルライン（12）",
    "tx_connors_rsi": "コナーズ RSI（3・2・100）", "tx_fisher_10": "フィッシャー・トランスフォーム（10）",
    "tx_elder_bull": "エルダー・レイ ブル・パワー ÷ 終値（%）", "tx_elder_bear": "エルダー・レイ ベア・パワー ÷ 終値（%）",
    "tx_bop_14": "バランス・オブ・パワー（14）",
    "tx_natr_14": "ATR（14）÷ 終値（%）", "tx_atr_ratio_14_50": "ATR（14）÷ ATR（50）",
    "tx_bb_pctb": "ボリンジャーバンド %B（20・2σ）", "tx_bb_width": "ボリンジャーバンドの幅（20・2σ、%）",
    "tx_bb_width_rank_120": "バンド幅の過去120日の中の順位（%）", "tx_keltner_pos": "ケルトナー・チャネルの中の位置（%）",
    "tx_squeeze": "スクイーズ（ボリンジャーがケルトナーの内側なら 1）", "tx_vol_parkinson_20": "パーキンソンのボラ（20、%）",
    "tx_vol_gk_20": "ガーマン・クラスのボラ（20、%）", "tx_vol_60": "60日のボラ（日次リターンの標準偏差、%）",
    "tx_ulcer_14": "アルサー・インデックス（14）", "tx_chaikin_vol_10": "チャイキン・ボラティリティ（10、%）",
    "tx_mass_index_25": "マス・インデックス（25）", "tx_chop_14": "チョピネス・インデックス（14）",
    "tx_obv_chg_20": "OBV の20日の変化 ÷ 20日の出来高（%）", "tx_obv_ma_gap": "OBV − その20日移動平均 ÷ 20日の出来高（%）",
    "tx_ad_ma_gap": "A/D ライン − その20日移動平均 ÷ 20日の出来高（%）", "tx_cmf_20": "チャイキン・マネーフロー（20）",
    "tx_chaikin_osc": "チャイキン・オシレーター ÷ 10日の平均出来高", "tx_mfi_14": "MFI（14）",
    "tx_force_13": "フォース・インデックス（13）÷（終値 × 20日の平均出来高）（%）",
    "tx_eom_14": "イーズ・オブ・ムーブメント（14、終値・平均出来高で割った形）",
    "tx_vwap_gap_20": "20日 VWAP との乖離（%）", "tx_vwap_gap_60": "60日 VWAP との乖離（%）",
    "tx_pvt_chg_20": "PVT の20日の変化 ÷ 20日の出来高（%）", "tx_vr_25": "出来高レシオ VR（25、%）",
    "tx_vol_osc_5_20": "出来高オシレーター（5日 ÷ 20日平均、%）", "tx_vol_ratio_60_250": "出来高の60日平均 ÷ 250日平均",
    "tx_up_vol_share_20": "上昇日の出来高の割合（20、%）", "tx_klinger": "クリンガー・オシレーター（34・55）÷ 20日の平均出来高",
    "tx_nvi_chg_60": "NVI の60日の変化（%）", "tx_tv_ratio_20_120": "売買代金の20日平均 ÷ 120日平均",
    "tx_body_ratio": "当日の実体 ÷ 値幅（陽線＋）", "tx_upper_shadow": "当日の上ヒゲ ÷ 値幅", "tx_lower_shadow": "当日の下ヒゲ ÷ 値幅",
    "tx_bull_share_20": "20日の陽線の割合（%）", "tx_candle_streak": "陽線・陰線の連続日数（陽線＋）",
    "tx_close_streak": "終値の連続上昇・下落日数（上昇＋）", "tx_body_mean_20": "20日の実体 ÷ 値幅の平均",
    "tx_gap_up_count_20": "20日の上への窓開けの回数", "tx_range_atr": "当日の値幅 ÷ ATR（14）",
    "tx_engulfing": "包み足（陽の包み足 1・陰の包み足 −1）", "tx_nr7": "NR7（7日で最も狭い値幅なら 1）",
    "tx_three_soldiers": "3日続けて陽線で高値・安値・終値を切り上げ（赤三兵に近い形なら 1）",
}
assert len(COLUMNS) == 111 and len(set(COLUMNS)) == 111 and set(JA) == set(COLUMNS), "列の一覧と名前がずれている"

LN2 = float(np.log(2.0))


class _Panel:
    """銘柄ごとに続いて並んだ配列の操作（行は銘柄・日付の順）。窓が銘柄をまたぐ行は欠測にする。"""

    def __init__(self, codes: np.ndarray):
        codes = np.asarray(codes)
        self.n = len(codes)
        new = np.ones(self.n, dtype=bool)
        if self.n:
            new[1:] = codes[1:] != codes[:-1]
        self.gid = np.cumsum(new) - 1
        self.starts = np.flatnonzero(new)
        self.pos = np.arange(self.n) - self.starts[self.gid] if self.n else np.zeros(0, dtype=int)

    def mask(self, x: np.ndarray, need: int) -> np.ndarray:
        """銘柄の中の位置が need 未満（取引日が need + 1 日そろう前）の行を欠測にする。"""
        out = np.asarray(x, dtype=float).copy()
        out[self.pos < need] = np.nan
        return out

    def shift(self, x, k: int = 1) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.full(self.n, np.nan)
        if 0 < k < self.n:
            out[k:] = x[:-k]
        out[self.pos < k] = np.nan
        return out

    def roll(self, x, n: int, fn: str = "mean", **kw) -> np.ndarray:
        """直近 n 行（当日を含む）の集計。銘柄の取引日が n 日そろうまでは欠測。"""
        r = getattr(pd.Series(np.asarray(x, dtype=float)).rolling(n, min_periods=n), fn)(**kw)
        return self.mask(r.to_numpy(dtype=float), n - 1)

    def ewm(self, x, *, span: float | None = None, alpha: float | None = None) -> np.ndarray:
        """銘柄ごとの指数平滑（adjust=False。最初の値から始める）。"""
        s = pd.Series(np.asarray(x, dtype=float))
        e = s.groupby(self.gid, sort=False).ewm(span=span, alpha=alpha, adjust=False).mean()
        return e.reset_index(level=0, drop=True).sort_index().to_numpy(dtype=float)

    def cumsum(self, x) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=float))
        c = np.cumsum(x)
        base = np.r_[0.0, c][self.starts]
        return c - base[self.gid]

    def windows(self, x, n: int, fn) -> np.ndarray:
        """直近 n 行の窓ごとに fn（窓の2次元配列 → 1次元）を当てる。銘柄をまたぐ窓は欠測。"""
        x = np.asarray(x, dtype=float)
        out = np.full(self.n, np.nan)
        if self.n < n:
            return out
        chunk = max(10_000, 20_000_000 // n)
        for s in range(n - 1, self.n, chunk):
            e = min(self.n, s + chunk)
            w = sliding_window_view(x[s - n + 1:e], n)
            out[s:e] = fn(w)
        return self.mask(out, n - 1)

    def runlen(self, sign: np.ndarray) -> np.ndarray:
        """同じ符号が続いた行数（変わった行が 1）× 符号。符号 0 は 0、欠測は欠測。"""
        s = np.asarray(sign, dtype=float)
        prev = self.shift(s, 1)
        start = (self.pos == 0) | ~(s == prev)
        run = np.cumsum(start) - 1
        first = np.flatnonzero(start)
        count = np.arange(self.n) - first[run] + 1 if self.n else np.zeros(0)
        out = count * s
        out[np.isnan(s)] = np.nan
        return out


def _div(a, b, fill=np.nan) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b != 0, a / b, fill)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def _gap(a, b) -> np.ndarray:
    """(a ÷ b − 1) × 100。b が 0 以下・欠測なら欠測。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, (a / b - 1.0) * 100.0, np.nan)


def _rsi(P: _Panel, x: np.ndarray, n: int) -> np.ndarray:
    """Wilder の RSI（平均の上げ幅・下げ幅を alpha = 1/n で平滑）。上げも下げも 0 なら 50。"""
    d = x - P.shift(x, 1)
    gain = P.ewm(np.where(d > 0, d, np.where(np.isnan(d), np.nan, 0.0)), alpha=1.0 / n)
    loss = P.ewm(np.where(d < 0, -d, np.where(np.isnan(d), np.nan, 0.0)), alpha=1.0 / n)
    tot = gain + loss
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(tot > 0, 100.0 * gain / tot, 50.0)
    return P.mask(out, n)


def _psar(P: _Panel, H: np.ndarray, L: np.ndarray, step: float = 0.02, cap: float = 0.2):
    """Wilder のパラボリック SAR（加速 0.02 ずつ、上限 0.2）。各取引日の SAR（反転した日は前のトレンドの極値）と向き（+1/−1）。"""
    n = P.n
    sar = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    hs, ls, pos = H.tolist(), L.tolist(), P.pos.tolist()
    up, s, ep, af = True, 0.0, 0.0, step
    for i in range(n):
        if pos[i] == 0:
            up, s, ep, af = True, ls[i], hs[i], step
            sar[i], trend[i] = s, 1.0
            continue
        s = s + af * (ep - s)
        if up:
            s = min(s, ls[i - 1], ls[i - 2]) if pos[i] >= 2 else min(s, ls[i - 1])
            if ls[i] < s:
                # 反転の日の SAR は前の上昇の極値（その日と前の日の高値を下回らない。TA-Lib と同じ）
                up, s, ep, af = False, max(ep, hs[i], hs[i - 1]), ls[i], step
            elif hs[i] > ep:
                ep, af = hs[i], min(af + step, cap)
        else:
            s = max(s, hs[i - 1], hs[i - 2]) if pos[i] >= 2 else max(s, hs[i - 1])
            if hs[i] > s:
                up, s, ep, af = True, min(ep, ls[i], ls[i - 1]), hs[i], step
            elif ls[i] < ep:
                ep, af = ls[i], min(af + step, cap)
        sar[i], trend[i] = s, (1.0 if up else -1.0)
    return sar, trend


def _supertrend(P: _Panel, H, L, C, atr, mult: float = 3.0) -> np.ndarray:
    """
    スーパートレンドの線（中値 ± ATR の mult 倍の帯。上昇中は下の帯、下落中は上の帯）。ATR が出るまでは欠測。
    帯は前の日の終値が前の日の帯の内側にある間は狭まる向きにだけ動き、向きは前の日の帯を終値が抜けたら変わる
    （TradingView の Supertrend と同じ作り）。最初の日は終値が中値以上なら上昇から始める。
    """
    n = P.n
    st = np.full(n, np.nan)
    hl2 = (H + L) / 2.0
    bub, blb = (hl2 + mult * atr).tolist(), (hl2 - mult * atr).tolist()
    cs, pos = C.tolist(), P.pos.tolist()
    ok = (~np.isnan(atr)).tolist()
    fub = flb = 0.0
    up = True
    started = False
    for i in range(n):
        if pos[i] == 0:
            started = False
        if not ok[i]:
            continue
        if not started:
            fub, flb = bub[i], blb[i]
            up = cs[i] >= (bub[i] + blb[i]) / 2.0
            started = True
        else:
            ub1, lb1 = fub, flb
            flb = max(blb[i], lb1) if cs[i - 1] > lb1 else blb[i]
            fub = min(bub[i], ub1) if cs[i - 1] < ub1 else bub[i]
            if not up and cs[i] > ub1:
                up = True
            elif up and cs[i] < lb1:
                up = False
        st[i] = flb if up else fub
    return st


def _fisher(P: _Panel, mp: np.ndarray, n: int = 10) -> np.ndarray:
    """Ehlers のフィッシャー・トランスフォーム（中値 (H+L)/2 の n 日の位置を 0.33/0.67 で平滑し、±0.999 で頭打ち）。"""
    hh = P.roll(mp, n, "max")
    ll = P.roll(mp, n, "min")
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rng > 0, 2.0 * ((mp - ll) / rng - 0.5), 0.0)
    raw[np.isnan(rng)] = np.nan
    out = np.full(P.n, np.nan)
    r, pos = raw.tolist(), P.pos.tolist()
    v = f = 0.0
    for i in range(P.n):
        if pos[i] == 0:
            v = f = 0.0
        if r[i] != r[i]:            # 欠測（窓がそろう前）
            continue
        v = 0.33 * r[i] + 0.67 * v
        v = 0.999 if v > 0.99 else (-0.999 if v < -0.99 else v)
        f = 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * f
        out[i] = f
    return out


def _rank_corr_time(w: np.ndarray) -> np.ndarray:
    """窓ごとの、時間の並びと値の順位の相関（同じ値は平均順位。RCI ÷ 100）。"""
    from scipy.stats import rankdata

    r = rankdata(w, axis=1, method="average")
    t = np.arange(1, w.shape[1] + 1, dtype=float)
    rc = r - r.mean(axis=1, keepdims=True)
    tc = t - t.mean()
    den = np.sqrt((rc ** 2).sum(axis=1) * (tc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (rc * tc).sum(axis=1) / den, 0.0)


def _age_of_extreme(w: np.ndarray, kind: str) -> np.ndarray:
    """窓の中で最大（最小）を付けた一番新しい行が何行前か（0 = 当日）。"""
    rev = w[:, ::-1]
    return (np.argmax(rev, axis=1) if kind == "max" else np.argmin(rev, axis=1)).astype(float)


def compute(panel: pd.DataFrame) -> pd.DataFrame:
    """
    panel の行ごとに111列（COLUMNS の順）。行の並び・index は panel のまま。売買の成立していない行は欠測。
    panel は銘柄・日付の順に並んでいること（同じ銘柄の行が続いていて、日付が昇順）。
    """
    need = ["Code", "Date", "open", "high", "low", "close", "vol"]
    miss = [c for c in need if c not in panel.columns]
    if miss:
        raise ValueError(f"日足に無い列: {miss}")
    codes_all = panel["Code"].astype(str).to_numpy()
    dates_all = pd.to_datetime(panel["Date"]).to_numpy()
    if len(panel) > 1:
        same = codes_all[1:] == codes_all[:-1]
        if (same & (dates_all[1:] <= dates_all[:-1])).any():
            raise ValueError("日足が銘柄・日付の順に並んでいない（同じ銘柄の中で日付が増えていない行がある）")
        starts = np.r_[True, ~same]
        if len(np.unique(codes_all[starts])) != int(starts.sum()):
            raise ValueError("同じ銘柄の行が続いていない")
    C0 = pd.to_numeric(panel["close"], errors="coerce").to_numpy(dtype=float)
    traded = np.isfinite(C0) & (C0 > 0)
    idx = np.flatnonzero(traded)
    C = C0[idx]

    def col(name):
        return pd.to_numeric(panel[name], errors="coerce").to_numpy(dtype=float)[idx]

    O, H, L, V = col("open"), col("high"), col("low"), col("vol")
    O = np.where(np.isfinite(O) & (O > 0), O, C)
    H = np.where(np.isfinite(H) & (H > 0), H, C)
    L = np.where(np.isfinite(L) & (L > 0), L, C)
    H = np.maximum.reduce([H, O, C])
    L = np.minimum.reduce([L, O, C])
    V = np.where(np.isfinite(V) & (V >= 0), V, 0.0)
    TV = (pd.to_numeric(panel["trading_value"], errors="coerce").to_numpy(dtype=float)[idx]
          if "trading_value" in panel.columns else C * V)
    P = _Panel(codes_all[idx])
    f: Dict[str, np.ndarray] = {}

    Cp, Hp, Lp, Op = P.shift(C, 1), P.shift(H, 1), P.shift(L, 1), P.shift(O, 1)
    dC = C - Cp
    tr = np.where(np.isnan(Cp), H - L, np.maximum.reduce([H - L, np.abs(H - Cp), np.abs(L - Cp)]))
    rng = H - L
    atr = {n: P.ewm(tr, alpha=1.0 / n) for n in (10, 14, 20, 50)}

    # ------------------------------------------------------------------ トレンド系
    sma = {n: P.roll(C, n) for n in (5, 10, 25, 50, 75, 100, 200)}
    for n in (5, 10, 25, 50, 75, 100, 200):
        f[f"tx_ma_gap_{n}"] = _gap(C, sma[n])
    for n in (25, 75, 200):
        f[f"tx_ma_slope_{n}"] = _gap(sma[n], P.shift(sma[n], 5))
    f["tx_ma_ratio_5_25"] = _gap(sma[5], sma[25])
    f["tx_ma_ratio_25_75"] = _gap(sma[25], sma[75])
    f["tx_ma_ratio_50_200"] = _gap(sma[50], sma[200])
    po = ((sma[5] > sma[25]).astype(float) + (sma[25] > sma[75]) + (sma[75] > sma[200]))
    po[np.isnan(sma[200]) | np.isnan(sma[75]) | np.isnan(sma[25]) | np.isnan(sma[5])] = np.nan
    f["tx_perfect_order"] = po
    f["tx_ma_cross_days"] = P.runlen(np.sign(sma[25] - sma[75]))

    ema12, ema26 = P.ewm(C, span=12), P.ewm(C, span=26)
    macd = ema12 - ema26
    sig = P.ewm(macd, span=9)
    f["tx_macd"] = P.mask(macd / C * 100.0, 25)
    f["tx_macd_signal"] = P.mask(sig / C * 100.0, 34)
    f["tx_macd_hist"] = P.mask((macd - sig) / C * 100.0, 34)

    upm, dnm = H - Hp, Lp - L
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    mdm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    pdm[np.isnan(Hp)] = np.nan
    mdm[np.isnan(Hp)] = np.nan
    pdi = 100.0 * _div(P.ewm(pdm, alpha=1 / 14), atr[14])
    mdi = 100.0 * _div(P.ewm(mdm, alpha=1 / 14), atr[14])
    dx = 100.0 * _div(np.abs(pdi - mdi), pdi + mdi, fill=0.0)
    f["tx_pdi_14"] = P.mask(pdi, 14)
    f["tx_mdi_14"] = P.mask(mdi, 14)
    f["tx_adx_14"] = P.mask(P.ewm(dx, alpha=1 / 14), 27)

    f["tx_aroon_up_25"] = 100.0 * (25.0 - P.windows(H, 26, lambda w: _age_of_extreme(w, "max"))) / 25.0
    f["tx_aroon_down_25"] = 100.0 * (25.0 - P.windows(L, 26, lambda w: _age_of_extreme(w, "min"))) / 25.0

    sar, ptrend = _psar(P, H, L)
    f["tx_psar_gap"] = P.mask((C - sar) / C * 100.0, 4)
    f["tx_psar_days"] = P.mask(P.runlen(ptrend), 4)

    tenkan = (P.roll(H, 9, "max") + P.roll(L, 9, "min")) / 2.0
    kijun = (P.roll(H, 26, "max") + P.roll(L, 26, "min")) / 2.0
    span_b = (P.roll(H, 52, "max") + P.roll(L, 52, "min")) / 2.0
    # 先行スパンは「当日を含めて26本先」に描く → t の雲は 25取引日前に計算した値。遅行スパンも同じく25取引日前と比べる
    sa, sb = P.shift((tenkan + kijun) / 2.0, 25), P.shift(span_b, 25)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    top[np.isnan(sa) | np.isnan(sb)] = np.nan
    bot[np.isnan(sa) | np.isnan(sb)] = np.nan
    cloud = np.where(C > top, (C - top) / C, np.where(C < bot, (C - bot) / C, 0.0)) * 100.0
    cloud[np.isnan(top)] = np.nan
    c25 = P.shift(C, 25)
    f["tx_ichi_tk"] = (tenkan - kijun) / C * 100.0
    f["tx_ichi_kijun_gap"] = _gap(C, kijun)
    f["tx_ichi_cloud_pos"] = cloud
    f["tx_ichi_cloud_thick"] = (sa - sb) / C * 100.0
    f["tx_ichi_chikou"] = _gap(C, c25)
    san = (tenkan > kijun).astype(float) + (C > top) + (C > c25)
    san[np.isnan(tenkan) | np.isnan(kijun) | np.isnan(top) | np.isnan(c25)] = np.nan
    f["tx_ichi_sanyaku"] = san

    y = np.log(C)
    x = P.pos.astype(float)
    for n in (20, 60):
        cov = pd.Series(y).rolling(n, min_periods=n).cov(pd.Series(x), ddof=0).to_numpy(dtype=float)
        cov = P.mask(cov, n - 1)
        varx = (n * n - 1) / 12.0
        f[f"tx_linreg_slope_{n}"] = cov / varx * 100.0
        if n == 60:
            vary = P.roll(y, n, "var", ddof=0)
            f["tx_linreg_r2_60"] = _div(cov * cov, varx * vary, fill=0.0)
    f["tx_kaufman_er_20"] = _div(np.abs(C - P.shift(C, 20)), P.mask(P.roll(np.abs(dC), 20, "sum"), 20), fill=0.0)
    e1 = P.ewm(C, span=15)
    e3 = P.ewm(P.ewm(e1, span=15), span=15)
    f["tx_trix_15"] = P.mask(_gap(e3, P.shift(e3, 1)), 44)
    trs = P.mask(P.roll(tr, 14, "sum"), 14)
    f["tx_vortex_pos_14"] = _div(P.roll(np.abs(H - Lp), 14, "sum"), trs)
    f["tx_vortex_neg_14"] = _div(P.roll(np.abs(L - Hp), 14, "sum"), trs)
    st = _supertrend(P, H, L, C, P.mask(atr[10], 9))
    f["tx_supertrend_gap"] = (C - st) / C * 100.0
    for n in (20, 55):
        hh, ll = P.roll(H, n, "max"), P.roll(L, n, "min")
        f[f"tx_donchian_pos_{n}"] = 100.0 * _div(C - ll, hh - ll, fill=0.5)

    # ------------------------------------------------------------------ オシレーター
    rsi14 = _rsi(P, C, 14)
    for n in (9, 14, 25):
        f[f"tx_rsi_{n}"] = rsi14 if n == 14 else _rsi(P, C, n)
    hh14, ll14 = P.roll(H, 14, "max"), P.roll(L, 14, "min")
    fast_k = 100.0 * _div(C - ll14, hh14 - ll14, fill=0.5)
    slow_k = P.roll(fast_k, 3)
    f["tx_stoch_k"] = slow_k
    f["tx_stoch_d"] = P.roll(slow_k, 3)
    rlo, rhi = P.roll(rsi14, 14, "min"), P.roll(rsi14, 14, "max")
    f["tx_stoch_rsi_14"] = 100.0 * _div(rsi14 - rlo, rhi - rlo, fill=0.5)
    f["tx_williams_r_14"] = -100.0 * _div(hh14 - C, hh14 - ll14, fill=0.5)
    tp = (H + L + C) / 3.0
    tp_sma = P.roll(tp, 20)
    mad = P.windows(tp, 20, lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1))
    f["tx_cci_20"] = _div(tp - tp_sma, 0.015 * mad, fill=0.0)
    for n in (10, 60, 120, 250):
        f[f"tx_roc_{n}"] = _gap(C, P.shift(C, n))
    lo_b = np.where(np.isnan(Cp), L, np.minimum(L, Cp))
    hi_b = np.where(np.isnan(Cp), H, np.maximum(H, Cp))
    bp, tru = C - lo_b, hi_b - lo_b
    avg = {n: _div(P.roll(bp, n, "sum"), P.roll(tru, n, "sum"), fill=0.5) for n in (7, 14, 28)}
    f["tx_ultimate_osc"] = 100.0 * (4 * avg[7] + 2 * avg[14] + avg[28]) / 7.0
    ups = P.mask(P.roll(np.where(dC > 0, dC, 0.0), 14, "sum"), 14)
    dns = P.mask(P.roll(np.where(dC < 0, -dC, 0.0), 14, "sum"), 14)
    f["tx_cmo_14"] = 100.0 * _div(ups - dns, ups + dns, fill=0.0)
    m = np.where(np.isnan(dC), np.nan, dC)
    num = P.ewm(P.ewm(m, span=25), span=13)
    den = P.ewm(P.ewm(np.abs(m), span=25), span=13)
    f["tx_tsi"] = P.mask(100.0 * _div(num, den, fill=0.0), 38)
    f["tx_dpo_20"] = (P.shift(C, 11) - P.roll(C, 20)) / C * 100.0
    roc = {n: _gap(C, P.shift(C, n)) for n in (10, 15, 20, 30)}
    f["tx_kst"] = (P.roll(roc[10], 10) + 2 * P.roll(roc[15], 10) + 3 * P.roll(roc[20], 10) + 4 * P.roll(roc[30], 15))
    mp = (H + L) / 2.0
    f["tx_awesome_osc"] = (P.roll(mp, 5) - P.roll(mp, 34)) / C * 100.0
    for n in (9, 26):
        f[f"tx_rci_{n}"] = 100.0 * P.windows(C, n, _rank_corr_time)
    f["tx_psy_12"] = P.mask(100.0 * P.roll((dC > 0).astype(float), 12, "sum") / 12.0, 12)
    streak = P.runlen(np.sign(dC))
    rocp = _gap(C, Cp)
    pr = P.windows(rocp, 101, lambda w: (w[:, :-1] < w[:, -1:]).sum(axis=1).astype(float))
    pr[np.isnan(rocp)] = np.nan
    f["tx_connors_rsi"] = P.mask((_rsi(P, C, 3) + _rsi(P, streak, 2) + pr) / 3.0, 101)
    f["tx_fisher_10"] = _fisher(P, mp, 10)
    ema13 = P.ewm(C, span=13)
    f["tx_elder_bull"] = P.mask((H - ema13) / C * 100.0, 12)
    f["tx_elder_bear"] = P.mask((L - ema13) / C * 100.0, 12)
    f["tx_bop_14"] = P.roll(_div(C - O, rng, fill=0.0), 14)

    # ------------------------------------------------------------------ ボラティリティ
    f["tx_natr_14"] = P.mask(atr[14] / C * 100.0, 13)
    f["tx_atr_ratio_14_50"] = P.mask(_div(atr[14], atr[50]), 49)
    m20 = P.roll(C, 20)
    sd20 = P.roll(C, 20, "std", ddof=0)
    f["tx_bb_pctb"] = 100.0 * _div(C - (m20 - 2 * sd20), 4 * sd20, fill=0.5)
    width = _div(4 * sd20, m20) * 100.0
    f["tx_bb_width"] = width
    f["tx_bb_width_rank_120"] = 100.0 * P.roll(width, 120, "rank", pct=True)
    mid = P.ewm(C, span=20)
    a10 = P.mask(atr[10], 9)
    f["tx_keltner_pos"] = P.mask(100.0 * _div(C - (mid - 2 * a10), 4 * a10, fill=0.5), 19)
    a20 = P.mask(atr[20], 19)
    sq = ((m20 + 2 * sd20 < m20 + 1.5 * a20) & (m20 - 2 * sd20 > m20 - 1.5 * a20)).astype(float)
    sq[np.isnan(sd20) | np.isnan(a20)] = np.nan
    f["tx_squeeze"] = sq
    lhl = np.log(H / L)
    f["tx_vol_parkinson_20"] = np.sqrt(P.roll(lhl ** 2, 20) / (4.0 * LN2)) * 100.0
    gk = 0.5 * lhl ** 2 - (2.0 * LN2 - 1.0) * np.log(C / O) ** 2
    f["tx_vol_gk_20"] = np.sqrt(np.clip(P.roll(gk, 20), 0.0, None)) * 100.0
    f["tx_vol_60"] = P.mask(P.roll(rocp, 60, "std"), 60)
    mx14 = P.roll(C, 14, "max")
    ddp = (C - mx14) / mx14 * 100.0
    f["tx_ulcer_14"] = np.sqrt(P.roll(ddp ** 2, 14))
    ehl = P.ewm(rng, span=10)
    f["tx_chaikin_vol_10"] = P.mask(_gap(ehl, P.shift(ehl, 10)), 19)
    e9 = P.ewm(rng, span=9)
    e99 = P.ewm(e9, span=9)
    f["tx_mass_index_25"] = P.mask(P.roll(_div(e9, e99, fill=1.0), 25, "sum"), 41)
    hl14 = P.roll(H, 14, "max") - P.roll(L, 14, "min")
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = 100.0 * np.log10(_div(P.roll(tr, 14, "sum"), hl14)) / np.log10(14.0)
    f["tx_chop_14"] = P.mask(chop, 14)

    # ------------------------------------------------------------------ 出来高系
    v20 = P.roll(V, 20, "sum")
    sgn = np.sign(dC)
    sv = np.where(np.isnan(sgn), 0.0, sgn) * V
    obv = P.cumsum(sv)
    f["tx_obv_chg_20"] = P.mask(100.0 * _div(P.roll(sv, 20, "sum"), v20), 20)
    f["tx_obv_ma_gap"] = 100.0 * _div(obv - P.roll(obv, 20), v20)
    mfm = _div((C - L) - (H - C), rng, fill=0.0)
    mfv = mfm * V
    ad = P.cumsum(mfv)
    f["tx_ad_ma_gap"] = 100.0 * _div(ad - P.roll(ad, 20), v20)
    f["tx_cmf_20"] = _div(P.roll(mfv, 20, "sum"), v20)
    f["tx_chaikin_osc"] = P.mask(_div(P.ewm(ad, span=3) - P.ewm(ad, span=10), P.roll(V, 10)), 9)
    tpp = P.shift(tp, 1)
    rmf = tp * V
    pos_mf = P.roll(np.where(tp > tpp, rmf, 0.0), 14, "sum")
    neg_mf = P.roll(np.where(tp < tpp, rmf, 0.0), 14, "sum")
    f["tx_mfi_14"] = P.mask(100.0 * _div(pos_mf, pos_mf + neg_mf, fill=0.5), 14)
    vma20 = P.roll(V, 20)
    fi = P.ewm(np.where(np.isnan(dC), np.nan, dC * V), span=13)
    f["tx_force_13"] = P.mask(100.0 * _div(fi, C * vma20), 19)
    # 中値の移動（終値比 %）× 値幅（終値比 %）÷ 出来高（20日平均比）。値幅 0 の日は 0
    dmid = (mp - P.shift(mp, 1)) / C * 100.0
    f["tx_eom_14"] = P.mask(P.roll(_div(dmid * (rng / C * 100.0), _div(V, vma20), fill=0.0), 14), 33)
    for n in (20, 60):
        vw = _div(P.roll(tp * V, n, "sum"), P.roll(V, n, "sum"))
        f[f"tx_vwap_gap_{n}"] = _gap(C, vw)
    f["tx_pvt_chg_20"] = P.mask(_div(P.roll(np.where(np.isnan(rocp), 0.0, rocp) * V, 20, "sum"), v20), 20)
    up_v = P.roll(np.where(dC > 0, V, 0.0), 25, "sum")
    dn_v = P.roll(np.where(dC < 0, V, 0.0), 25, "sum")
    fl_v = P.roll(np.where(dC == 0, V, 0.0), 25, "sum")
    vr = 100.0 * _div(up_v + fl_v / 2.0, dn_v + fl_v / 2.0, fill=np.inf)
    f["tx_vr_25"] = P.mask(np.minimum(vr, 1000.0), 25)
    f["tx_vol_osc_5_20"] = _gap(P.roll(V, 5), vma20)
    f["tx_vol_ratio_60_250"] = _div(P.roll(V, 60), P.roll(V, 250))
    f["tx_up_vol_share_20"] = P.mask(100.0 * _div(P.roll(np.where(dC > 0, V, 0.0), 20, "sum"), v20), 20)
    hlc = H + L + C
    ktr = np.where(hlc > P.shift(hlc, 1), 1.0, -1.0)
    ktr[P.pos == 0] = np.nan
    kv = ktr * V
    f["tx_klinger"] = P.mask(_div(P.ewm(kv, span=34) - P.ewm(kv, span=55), vma20), 55)
    lr = np.where(V < P.shift(V, 1), np.log1p(np.where(np.isnan(rocp), 0.0, rocp) / 100.0), 0.0)
    f["tx_nvi_chg_60"] = P.mask((np.exp(P.roll(lr, 60, "sum")) - 1.0) * 100.0, 60)
    f["tx_tv_ratio_20_120"] = _div(P.roll(TV, 20), P.roll(TV, 120))

    # ------------------------------------------------------------------ ローソク足
    f["tx_body_ratio"] = _div(C - O, rng, fill=0.0)
    f["tx_upper_shadow"] = _div(H - np.maximum(O, C), rng, fill=0.0)
    f["tx_lower_shadow"] = _div(np.minimum(O, C) - L, rng, fill=0.0)
    f["tx_bull_share_20"] = 100.0 * P.roll((C > O).astype(float), 20)
    f["tx_candle_streak"] = P.runlen(np.sign(C - O))
    f["tx_close_streak"] = streak
    f["tx_body_mean_20"] = P.roll(_div(np.abs(C - O), rng, fill=0.0), 20)
    f["tx_gap_up_count_20"] = P.mask(P.roll((L > Hp).astype(float), 20, "sum"), 20)
    f["tx_range_atr"] = P.mask(_div(rng, atr[14]), 13)
    bull = (Cp < Op) & (C > O) & (O < Cp) & (C > Op)
    bear = (Cp > Op) & (C < O) & (O > Cp) & (C < Op)
    eng = bull.astype(float) - bear.astype(float)
    eng[np.isnan(Cp)] = np.nan
    f["tx_engulfing"] = eng
    f["tx_nr7"] = P.mask((rng <= P.roll(rng, 7, "min")).astype(float), 6)
    yang = C > O
    step_up = (C > Cp) & (H > Hp) & (L > Lp) & yang
    three = (step_up & (P.shift(step_up.astype(float), 1) == 1) & (P.shift(step_up.astype(float), 2) == 1)).astype(float)
    f["tx_three_soldiers"] = P.mask(three, 3)

    out = pd.DataFrame(np.nan, index=panel.index, columns=COLUMNS)
    vals = np.column_stack([np.asarray(f[c], dtype=float) for c in COLUMNS]) if len(idx) else np.zeros((0, len(COLUMNS)))
    vals[~np.isfinite(vals)] = np.nan
    out.iloc[idx, :] = vals
    return out


def compute_at(panel: pd.DataFrame, keys: pd.DataFrame, codes_per_chunk: int = 300) -> pd.DataFrame:
    """
    keys（Code・Date）の行ごとの111列（keys の並びのまま。日足に無い行は欠測）。全銘柄の日足（約800万行）で一度に compute すると
    111列 × 全行がメモリに乗るので、銘柄を codes_per_chunk ずつに分けて計算し、keys の行だけを残す（銘柄をまたぐ計算は無いので
    分けても値は同じ。tests/test_technicals.py が確かめる）。panel は compute と同じ並び。
    """
    k = keys[["Code", "Date"]].copy()
    k["Code"] = k["Code"].astype(str)
    k["Date"] = pd.to_datetime(k["Date"]).astype("datetime64[ns]")
    codes = panel["Code"].astype(str).to_numpy()
    want = set(k["Code"])
    order = pd.unique(codes)
    order = [c for c in order if c in want]
    parts = []
    for s in range(0, len(order), codes_per_chunk):
        chunk = set(order[s:s + codes_per_chunk])
        sub = panel.loc[np.isin(codes, list(chunk))].reset_index(drop=True)
        tx = compute(sub)
        tx.insert(0, "Code", sub["Code"].astype(str).to_numpy())
        tx.insert(1, "Date", pd.to_datetime(sub["Date"]).astype("datetime64[ns]").to_numpy())
        parts.append(k[k["Code"].isin(chunk)].merge(tx, on=["Code", "Date"], how="inner"))
    if parts:
        got = pd.concat(parts, ignore_index=True)
    else:                                                       # keys が無い・日足に無い銘柄だけ（型をそろえて空の表）
        got = pd.DataFrame({"Code": pd.Series(dtype=str), "Date": pd.Series(dtype="datetime64[ns]"),
                            **{c: pd.Series(dtype=float) for c in COLUMNS}})
    if got.duplicated(["Code", "Date"]).any():
        raise ValueError("日足の Code・Date が重なっている")
    m = k.merge(got, on=["Code", "Date"], how="left")
    if len(m) != len(k):
        raise ValueError("指標の突き合わせで行が増えた")
    return m[COLUMNS].astype(float).reset_index(drop=True)
