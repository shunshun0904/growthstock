#!/usr/bin/env python3
"""
probe: テクニカル指標111列（research/technicals.py）と今の株価・出来高の27列が、1列ずつで今の目的変数（1.5σ）をどれだけ
分けるか（検証だけ。集計だけ出す。本番は変えない）。

運用者の問い（2026-10-18）「仮説としては今の目的変数の予測にはどのテクニカル指標が効くと思いますか？ 実際にグラフで可視化して
ください」→ 仮説を結果を見る前にこの docstring と HYPOTHESIS に固定し、実データで1列ずつの分け方を数える。グラフは手元で
この台本の出力（集計だけ）から作る。

仮説（2026-10-18、結果を見る前。根拠は今までの実測）
  根拠1: ボラが低いほど正例が多い（1.5σ のボラの帯ごとの正例率 21.3 / 14.5 / 11.5 / 10.1 / 7.4%（実験78）、vol_20d の
         日付内 AUC 0.40（docs/MODEL_WITHIN_DATE.md））。到達の必要幅が σ で決まるので、荒い銘柄は 1.5σ×√20 を上げ切って保ちにくい
  根拠2: 大きく跳ねた抜けほど正例が少ない（実験103: 前日比 15% 超 6.4%・0〜2% 16.2%、5営業日の騰落 25% 超 7.1%、ret_20d・
         volume_trend・break_margin・close_position の日付内 AUC が 0.43〜0.46）
  根拠3: 3・6か月前から高値圏にいた銘柄ほど正例が多い（r_high_3m / 6m の日付内 AUC 0.53）。V字の戻りより、高値圏で
         もみ合ってからの抜けが続きやすい
  → 予想（1列ずつ・日付内 AUC で、0.5 からの差の大きさ: 強 0.05 以上・中 0.02〜0.05・弱 0.01〜0.02・無 0.01 未満）
  H1 ボラの水準（ATR÷終値・パーキンソン・ガーマン・クラス・60日のボラ・バンドの幅）: 高いほど正例が少ない。強
     （ただし vol_20d の言い換えで、足しても上積みは小さい）
  H2 短期の過熱（5〜25日の乖離率・ROC10・RSI・CCI・%B・VWAP との乖離・エルダーのブル・パワー・当日の値幅 ÷ ATR・出来高の急増）:
     高いほど正例が少ない。中。ストキャスティクス・%R は抜けた日に高値圏で張り付くので弱
  H3 長めの上げ幅（ROC60/120/250・100・200日の乖離率・雲との距離・遅行スパン・スーパートレンドとの距離）: 高いほど正例が少ない
     （V字の戻り）。弱〜中。上げの滑らかさ（60日の決定係数）・長いトレンドの続き（200日線の傾き・パーフェクトオーダー・交差から
     の日数）は高いほど正例が多い。弱
  H4 出来高の蓄積（OBV・A/D・CMF・チャイキン・オシレーター・上昇日の出来高の割合・NVI・クリンガー）: 弱いか無い（向きは
     決めない）
  H5 この母集団ではほぼ効かない: アルーン・アップ（抜けた日は25日の最高値なのでほぼ 100）・ドンチャンの位置・三役好転（抜けた日は
     ほぼ 3）・包み足・NR7・赤三兵（まれ）・ADX・雲の厚さ
  列ごとの予想は HYPOTHESIS（"+" 高いほど正例が多い / "-" 少ない / "0" 効かない or 向きを決めない、強さ）。

数えるもの（行: 学習データの全行。ラベルは 1.5σ（rise_label_for_k）。参考に 1.2σ と +10% の指値への到達）
  auc_day   日付内 AUC（同じ日の候補どうしで、値の高い方が正例か。正例・負例の両方がある日だけ・日の件数で重み付け。
            lab._auc_in_day と同じ。0.5 が分離力ゼロ、0.5 より大きければ高いほど正例が多い）← 主
  auc_all   全行の ROC-AUC（日をまたぐ。相場の局面の差も入る）
  auc_day_12 / auc_day_hit   1.2σ のラベル / +10% の到達での日付内 AUC
  auc_early / auc_late       2018-04〜2021-12 / 2022-01〜 の日付内 AUC（向きが続くか）
  q / h     5分位（全行で切る）ごとの 1.5σ の正例率・+10% の到達率
  rho_vol   vol_20d との順位相関（ボラの言い換えかどうか）
公開ログには件数・割合・相関・AUC・分位の境目だけ（銘柄・行ごとの値は出さない）。機械が読む行は「SIG<TAB>…」。

    exp=probe_technical_signal.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import technicals as TX  # noqa: E402
import e105_technical_ab as E105  # noqa: E402

K = 1.5
#: 今の株価・出来高から作った27列（比べる相手）
PRICE_COLS = ["r_high", "r_high_3m", "r_high_6m", "base_length", "break_margin", "close_position", "ret_20d", "vol_20d",
              "volume_trend", "log_trading_value", "log_market_cap", "vol_rel_long", "vol_rel_mkt", "vol_rel_sector",
              "vol_updown", "vol_rel_short", "vol_gap_ratio", "broken_high_age", "prev_nh_ratio", "base_depth",
              "pullback_depth", "max_dd_120", "ret_1d", "ret_2d", "gap_open", "ret_5d", "vol_chg_1d"]
SPLIT = pd.Timestamp("2022-01-01")
STRENGTH = {"強": 0.05, "中": 0.02, "弱": 0.01, "無": 0.0}


def _h(sign: str, strength: str, *cols):
    return {c: (sign, strength) for c in cols}


#: 列ごとの予想（結果を見る前に固定。2026-10-18）。符号: "+" 高いほど正例が多い / "-" 少ない / "0" 効かない・向きを決めない
HYPOTHESIS = {
    # H1 ボラの水準
    **_h("-", "強", "tx_natr_14", "tx_vol_parkinson_20", "tx_vol_gk_20", "tx_vol_60", "tx_bb_width"),
    **_h("-", "中", "tx_ulcer_14"),
    **_h("-", "弱", "tx_atr_ratio_14_50", "tx_bb_width_rank_120", "tx_chaikin_vol_10", "tx_mass_index_25"),
    **_h("+", "弱", "tx_squeeze"),
    **_h("0", "無", "tx_chop_14"),
    # H2 短期の過熱
    **_h("-", "中", "tx_ma_gap_5", "tx_ma_gap_10", "tx_ma_gap_25", "tx_roc_10", "tx_rsi_9", "tx_rsi_14", "tx_cci_20",
         "tx_bb_pctb", "tx_vwap_gap_20", "tx_elder_bull", "tx_range_atr", "tx_vol_osc_5_20", "tx_keltner_pos",
         "tx_connors_rsi", "tx_linreg_slope_20", "tx_ichi_kijun_gap"),
    **_h("-", "弱", "tx_rsi_25", "tx_stoch_k", "tx_stoch_d", "tx_stoch_rsi_14", "tx_williams_r_14", "tx_cmo_14",
         "tx_ultimate_osc", "tx_psy_12", "tx_rci_9", "tx_fisher_10", "tx_elder_bear", "tx_awesome_osc", "tx_macd",
         "tx_macd_signal", "tx_macd_hist", "tx_tsi", "tx_kst", "tx_dpo_20", "tx_trix_15", "tx_ichi_tk", "tx_ma_ratio_5_25",
         "tx_ma_ratio_25_75", "tx_ma_slope_25", "tx_kaufman_er_20", "tx_vortex_pos_14", "tx_pdi_14", "tx_psar_gap",
         "tx_body_ratio", "tx_candle_streak", "tx_close_streak", "tx_gap_up_count_20", "tx_vr_25", "tx_force_13",
         "tx_mfi_14", "tx_pvt_chg_20", "tx_vol_ratio_60_250", "tx_tv_ratio_20_120"),
    **_h("+", "弱", "tx_upper_shadow", "tx_vortex_neg_14", "tx_mdi_14"),
    # H3 長めの上げ幅（V字の戻り）と、上げの滑らかさ・長いトレンドの続き
    **_h("-", "中", "tx_roc_60", "tx_roc_120", "tx_ma_gap_100", "tx_ma_gap_200", "tx_ichi_cloud_pos", "tx_ichi_chikou",
         "tx_supertrend_gap", "tx_vwap_gap_60"),
    **_h("-", "弱", "tx_roc_250", "tx_ma_gap_50", "tx_ma_gap_75", "tx_linreg_slope_60"),
    **_h("+", "弱", "tx_linreg_r2_60", "tx_ma_slope_200", "tx_perfect_order", "tx_ma_cross_days", "tx_psar_days"),
    # H4 出来高の蓄積（向きは決めない）
    **_h("0", "無", "tx_obv_chg_20", "tx_obv_ma_gap", "tx_ad_ma_gap", "tx_cmf_20", "tx_chaikin_osc", "tx_eom_14",
         "tx_up_vol_share_20", "tx_klinger", "tx_nvi_chg_60"),
    # H5 この母集団ではほぼ効かない（値が偏る・まれ）
    **_h("0", "無", "tx_aroon_up_25", "tx_aroon_down_25", "tx_donchian_pos_20", "tx_donchian_pos_55", "tx_ichi_sanyaku",
         "tx_engulfing", "tx_nr7", "tx_three_soldiers", "tx_adx_14", "tx_ichi_cloud_thick", "tx_ma_slope_75",
         "tx_ma_ratio_50_200", "tx_rci_26", "tx_bop_14", "tx_lower_shadow", "tx_bull_share_20", "tx_body_mean_20"),
}
assert set(HYPOTHESIS) == set(TX.COLUMNS), f"予想の無い列: {sorted(set(TX.COLUMNS) - set(HYPOTHESIS))}"


def auc_in_day(values: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    """
    日付内 AUC（lab._auc_in_day と同じ: 正例・負例の両方がある日だけ、日の件数で重み付けした平均）。値が欠測の行は除く。
    日ごとの AUC は平均順位から（同じ値は半分ずつ数える。roc_auc_score と同じ）。
    """
    d = pd.DataFrame({"v": np.asarray(values, dtype=float), "y": np.asarray(y, dtype=float), "d": dates})
    d = d[d["v"].notna() & d["y"].notna()]
    if not len(d):
        return float("nan")
    d["r"] = d.groupby("d")["v"].rank(method="average")
    g = d.groupby("d")
    n = g.size()
    npos = g["y"].sum()
    s = d[d["y"] == 1].groupby("d")["r"].sum().reindex(n.index, fill_value=0.0)
    nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    if not ok.any():
        return float("nan")
    auc = (s[ok] - npos[ok] * (npos[ok] + 1) / 2.0) / (npos[ok] * nneg[ok])
    return float(np.average(auc, weights=n[ok]))


def auc_all(values: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    v = np.asarray(values, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(v) & np.isfinite(yy)
    if m.sum() < 10 or yy[m].min() == yy[m].max():
        return float("nan")
    return float(roc_auc_score(yy[m].astype(int), v[m]))


def quintile_rates(values: np.ndarray, y: np.ndarray, hit: np.ndarray) -> dict:
    """5分位（全行で切る。同じ値の多い列は切れる数だけ）ごとの正例率・+10% の到達率・件数と、分位の境目。"""
    v = pd.Series(np.asarray(values, dtype=float))
    m = v.notna()
    out = {"edges": [], "q": [], "h": [], "n": []}
    if m.sum() < 50:
        return out
    try:
        cut, edges = pd.qcut(v[m], 5, retbins=True, duplicates="drop")
    except ValueError:
        return out
    yy = pd.Series(np.asarray(y, dtype=float))[m]
    hh = pd.Series(np.asarray(hit, dtype=float))[m]
    for iv in cut.cat.categories:
        sel = (cut == iv).to_numpy()
        out["q"].append(float(yy[sel].mean()))
        out["h"].append(float(hh[sel].mean()))
        out["n"].append(int(sel.sum()))
    out["edges"] = [float(x) for x in edges]
    return out


def signal_table(df: pd.DataFrame, cols: list, kinds: dict) -> pd.DataFrame:
    """列ごとの物差し（docstring の「数えるもの」）。df は y15・y12・hit10・Date・vol_20d を持つこと。"""
    dates = df["Date"].to_numpy()
    early = (df["Date"] < SPLIT).to_numpy()
    vol_rank = df["vol_20d"].rank()
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        qr = quintile_rates(v, df["y15"].to_numpy(), df["hit10"].to_numpy())
        hyp = HYPOTHESIS.get(c, ("", ""))
        rows.append({
            "col": c, "kind": kinds[c], "missing": float(np.isnan(v).mean()),
            "auc_day": auc_in_day(v, df["y15"].to_numpy(), dates),
            "auc_all": auc_all(v, df["y15"].to_numpy()),
            "auc_day_12": auc_in_day(v, df["y12"].to_numpy(), dates),
            "auc_day_hit": auc_in_day(v, df["hit10"].to_numpy(), dates),
            "auc_early": auc_in_day(v[early], df["y15"].to_numpy()[early], dates[early]),
            "auc_late": auc_in_day(v[~early], df["y15"].to_numpy()[~early], dates[~early]),
            "rho_vol": float(pd.Series(v).rank().corr(vol_rank)),
            "q": qr["q"], "h": qr["h"], "qn": qr["n"], "edges": qr["edges"],
            "hyp_sign": hyp[0], "hyp_strength": hyp[1]})
    out = pd.DataFrame(rows)
    out["dev"] = out["auc_day"] - 0.5
    return out


def hypothesis_check(t: pd.DataFrame) -> dict:
    """予想（向き・強さ）と結果の突き合わせ。向き: 予想が + / − の列で、日付内 AUC の向きが合うか。強さ: 帯が合うか。"""
    tx = t[t["kind"] != "既存"].copy()

    def band(x):
        a = abs(x)
        return "強" if a >= 0.05 else ("中" if a >= 0.02 else ("弱" if a >= 0.01 else "無"))

    tx["band"] = tx["dev"].map(band)
    signed = tx[tx["hyp_sign"].isin(["+", "-"])]
    hit_dir = ((signed["hyp_sign"] == "+") & (signed["dev"] > 0)) | ((signed["hyp_sign"] == "-") & (signed["dev"] < 0))
    zero = tx[tx["hyp_sign"] == "0"]
    return {"signed": int(len(signed)), "direction_ok": int(hit_dir.sum()),
            "zero": int(len(zero)), "zero_small": int((zero["dev"].abs() < 0.02).sum()),
            "band_ok": int((tx["band"] == tx["hyp_strength"]).sum()), "all": int(len(tx))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="probe: テクニカル指標111列と株価・出来高の27列の、1列ずつの分け方（集計だけ）")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    base = F.columns(E105.BASE_PRESET)
    missing = [c for c in PRICE_COLS if c not in base]
    if missing:
        raise SystemExit(f"株価・出来高の27列が今の列に無い: {missing}")
    full, stamp = E105.prepare(base, args.rebuild)
    full["Date"] = pd.to_datetime(full["Date"])
    y15 = B.rise_label_for_k(full, K)
    keep = y15.notna().to_numpy()
    df = full[keep].reset_index(drop=True)
    df["y15"] = y15[keep].to_numpy()
    df["y12"] = full.loc[keep, "label"].to_numpy(dtype=float)
    print("=" * 100)
    print(f"probe: テクニカル指標111列と株価・出来高の27列を1列ずつ（行 {len(df):,}・{df['Date'].min().date()}〜"
          f"{df['Date'].max().date()}・1.5σ の正例率 {df['y15'].mean()*100:.1f}%・1.2σ {df['y12'].mean()*100:.1f}%・"
          f"+10% の到達 {df['hit10'].mean()*100:.1f}%。表は {stamp['built_utc']} 作成）")
    print("=" * 100)
    kinds = {**{c: TX.GROUP_OF[c] for c in TX.COLUMNS}, **{c: "既存" for c in PRICE_COLS}}
    t = signal_table(df, list(TX.COLUMNS) + PRICE_COLS, kinds)
    # 機械が読む行（集計だけ）
    for _, r in t.iterrows():
        print("SIG\t" + "\t".join([
            r["col"], r["kind"], f"{r['missing']:.4f}", f"{r['auc_day']:.4f}", f"{r['auc_all']:.4f}",
            f"{r['auc_day_12']:.4f}", f"{r['auc_day_hit']:.4f}", f"{r['auc_early']:.4f}", f"{r['auc_late']:.4f}",
            f"{r['rho_vol']:.3f}", ",".join(f"{x:.4f}" for x in r["q"]), ",".join(f"{x:.4f}" for x in r["h"]),
            ",".join(str(x) for x in r["qn"]), ",".join(f"{x:.4g}" for x in r["edges"]), r["hyp_sign"], r["hyp_strength"]]))
    # 人が読む表
    t = t.sort_values("dev", key=lambda s: -s.abs())
    print("\n■ 日付内 AUC の 0.5 からの差が大きい順（1.5σ。+ は高いほど正例が多い）上位40")
    print(f"  {'列':<24}{'系統':<12}{'日付内':>8}{'全行':>8}{'1.2σ':>8}{'+10%':>8}{'前半':>8}{'後半':>8}{'ボラとの相関':>10}  予想")
    for _, r in t.head(40).iterrows():
        print(f"  {r['col']:<24}{r['kind']:<12}{r['dev']:>+8.3f}{r['auc_all'] - 0.5:>+8.3f}{r['auc_day_12'] - 0.5:>+8.3f}"
              f"{r['auc_day_hit'] - 0.5:>+8.3f}{r['auc_early'] - 0.5:>+8.3f}{r['auc_late'] - 0.5:>+8.3f}{r['rho_vol']:>+10.2f}"
              f"  {r['hyp_sign']}{r['hyp_strength']}")
    print("\n■ 系統ごとの |日付内 AUC − 0.5| の中央値と最大")
    for k, g in t.groupby("kind"):
        print(f"  {k:<12}{len(g):>4}列  中央値 {g['dev'].abs().median():.3f}  最大 {g['dev'].abs().max():.3f}（{g.iloc[0]['col']}）")
    hc = hypothesis_check(t)
    print(f"\n■ 予想との突き合わせ: 向きを決めた {hc['signed']}列のうち向きが合った {hc['direction_ok']}列 / "
          f"「効かない」とした {hc['zero']}列のうち差が 0.02 未満 {hc['zero_small']}列 / 強さの帯が合った {hc['band_ok']} / {hc['all']}列")
    print(f"\n（{time.time()-t0:.0f}秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
