#!/usr/bin/env python3
"""
新高値ブレイクアウト予測の学習データセットを構築する。

設計は docs/MODEL_DESIGN.md を参照。要点:

  * サンプリング : 銘柄 × 月末営業日
  * 特徴量       : 基準日 t までに「開示済み」のデータのみ（先読みなし）
  * ラベル       : [t+20営業日, t+120営業日] にブレイクアウトが起きたか
  * 未確定       : t+140営業日ぶんのデータが無いサンプルは捨てる（正例/負例に混ぜない）

入力  : research/_data/{bars,fins,margin,topix}_*.parquet  (jq_bulk.py が生成)
出力  : research/_data/dataset.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

# --- ラベル定義のパラメータ (docs/MODEL_DESIGN.md §2.1) --- #
# 既定値。LabelConfig で上書きできる（定義の比較検証のため）。
# 10定義を実測比較したうえで採用した定義 E（docs/MODEL_RESULTS 参照）:
#   52週高値 / ホライズン 1〜6ヶ月 / 定着20日-8% / ブレイク60営業日後も水準維持
# 「+60日後維持」が分離度に最も効いた（+2.2pt -> +3.8pt、全10定義中で最高）。
# 定着日数の延長(40日)は分離度がむしろ悪化し、78週高値も効果が薄かった。
HORIZON_START = 20      # 予測ホライズンの開始（営業日）= 約1ヶ月先
HORIZON_END = 120       # 予測ホライズンの終了（営業日）= 約6ヶ月先
HOLD_DAYS = 20          # ブレイク後の定着を見る日数
HOLD_DRAWDOWN = 0.92    # ブレイク時終値の-8%を割らないこと
VOL_MULTIPLE = 1.5      # ブレイク日の出来高が20日平均の何倍以上か
HIGH_WINDOW = 245       # 52週 ≒ 245営業日 (78週なら 368)
SUSTAIN_DAYS = 60       # ブレイク60営業日後の水準を見る
SUSTAIN_RATIO = 1.0     # ブレイク時終値を下回らないこと


@dataclass(frozen=True)
class LabelConfig:
    """ラベル定義。定義を変えて比較できるようにパラメータ化してある。"""
    high_window: int = HIGH_WINDOW
    horizon_start: int = HORIZON_START
    horizon_end: int = HORIZON_END
    hold_days: int = HOLD_DAYS
    hold_drawdown: float = HOLD_DRAWDOWN
    vol_multiple: float = VOL_MULTIPLE
    max_rhigh_at_t: float = 95.0
    min_trading_value: float = 0.5
    # --- 定着条件（任意）---
    # ブレイクから sustain_days 営業日後の終値が、
    # ブレイク時終値の sustain_ratio 倍以上であることを要求する。
    # hold（期間中の最安値が -x% を割らない）が「急落しないこと」を見るのに対し、
    # sustain は「一定期間後も水準を保っていること」を見る。失速を除外できる。
    sustain_days: int = SUSTAIN_DAYS   # 0 なら条件なし
    sustain_ratio: float = SUSTAIN_RATIO

    @property
    def name(self) -> str:
        weeks = round(self.high_window / 245 * 52)
        m0 = round(self.horizon_start / 20)
        m1 = round(self.horizon_end / 20)
        dd = round((1 - self.hold_drawdown) * 100)
        base = f"{weeks}週 / {m0}〜{m1}ヶ月 / 定着{self.hold_days}日-{dd}%"
        if self.sustain_days:
            pct = round((self.sustain_ratio - 1) * 100)
            sign = "+" if pct > 0 else ""
            base += f" +{self.sustain_days}日後{sign}{pct}%"
        return base

    @property
    def forward_needed(self) -> int:
        """ラベル確定に必要な将来営業日数。"""
        return self.horizon_end + max(self.hold_days, self.sustain_days)


DEFAULT_LABEL = LabelConfig()

# --- 除外条件 (docs/MODEL_DESIGN.md §2.2) --- #
MAX_RHIGH_AT_T = 95.0   # 基準日ですでに高値圏の銘柄は対象外
MIN_TRADING_VALUE = 0.5 # 20日平均売買代金の下限（億円）


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #

def load_parts(prefix: str, data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not paths:
        raise SystemExit(f"{prefix} の parquet が {data_dir} にありません。先に jq_bulk.py を実行してください")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"[load] {prefix}: {len(df):,}行 ({len(paths)}ファイル)")
    return df


# --------------------------------------------------------------------------- #
# 株価系の特徴量とラベル（銘柄ごとに時系列で算出）
# --------------------------------------------------------------------------- #

def price_panel(bars: pd.DataFrame, cfg: LabelConfig = DEFAULT_LABEL) -> pd.DataFrame:
    """
    銘柄ごとに時系列指標を算出する。

    調整後（Adj*）を優先して使う。株式分割をまたぐと素の価格では
    52週高値が不連続になり、偽のブレイクを大量に生むため。
    """
    df = bars.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

    close = df["AdjC"].fillna(df["C"])
    high = df["AdjH"].fillna(df["H"])
    vol = df["AdjVo"].fillna(df["Vo"])
    df["close"] = close
    df["high"] = high
    df["vol"] = vol
    # 売買代金は実際の円建て金額なので素の終値×出来高を使う（仕様書 §3.2-4）
    df["trading_value"] = df["C"] * df["Vo"] / 1e8

    g = df.groupby("Code", sort=False)

    # --- 52週高値（当日を含む / 含まない の2種類が要る） --- #
    # 含む  : 基準日時点の高値接近率 R_high の分母
    # 含まない: ブレイク判定（「それまでの高値」を上抜けたか）の基準
    w = cfg.high_window
    df["high52w"] = g["high"].transform(
        lambda s: s.rolling(w, min_periods=w).max()
    )
    df["high52w_prior"] = g["high"].transform(
        lambda s: s.rolling(w, min_periods=w).max().shift(1)
    )
    df["r_high"] = df["close"] / df["high52w"] * 100.0

    # --- 出来高モメンタム（当日を除く直前20日平均との比） --- #
    df["vol_ma20"] = g["vol"].transform(
        lambda s: s.rolling(20, min_periods=15).mean().shift(1)
    )
    df["volume_trend"] = df["vol"] / df["vol_ma20"] * 100.0

    # --- 流動性フィルタ用 --- #
    df["tv_ma20"] = g["trading_value"].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )

    # --- 過去の R_high（ベース形成の推移を見る） --- #
    df["r_high_3m"] = g["r_high"].shift(60)
    df["r_high_6m"] = g["r_high"].shift(120)

    return df


def breakout_flags(df: pd.DataFrame, cfg: LabelConfig = DEFAULT_LABEL) -> pd.DataFrame:
    """
    各営業日が「ブレイクアウト日」かどうかを判定する。

    3条件すべてを満たす日のみ True:
      1. 終値が それまでの52週高値 を上抜け
      2. 出来高が20日平均の VOL_MULTIPLE 倍以上
      3. 以降 hold_days 営業日、ブレイク時終値から hold_drawdown を割らない
      4. (任意) sustain_days 営業日後も、ブレイク時終値の sustain_ratio 倍以上
    """
    g = df.groupby("Code", sort=False)

    cond_high = df["close"] > df["high52w_prior"]
    cond_vol = df["vol"] >= df["vol_ma20"] * cfg.vol_multiple

    # 3. 定着: 未来 HOLD_DAYS 日の終値の最小値。
    #    逆順 rolling で「t+1 〜 t+HOLD_DAYS」の最小値を得る。
    hd = cfg.hold_days

    def future_min(s: pd.Series) -> pd.Series:
        return s[::-1].rolling(hd, min_periods=hd).min()[::-1].shift(-1)

    df["future_min_close"] = g["close"].transform(future_min)
    cond_hold = df["future_min_close"] >= df["close"] * cfg.hold_drawdown

    is_bo = cond_high & cond_vol & cond_hold
    undetermined = df["future_min_close"].isna()

    # 4. 水準維持（任意）: sustain_days 営業日後もブレイク時終値の水準を保っているか
    if cfg.sustain_days:
        sd = cfg.sustain_days
        df["sustain_close"] = g["close"].transform(lambda s: s.shift(-sd))
        cond_sustain = df["sustain_close"] >= df["close"] * cfg.sustain_ratio
        is_bo = is_bo & cond_sustain
        undetermined = undetermined | df["sustain_close"].isna()

    # 判定に必要な将来データが無い（データ末尾）日は判定不能として NaN にする
    df["is_breakout"] = is_bo.where(~undetermined)

    return df


def attach_labels(df: pd.DataFrame, cfg: LabelConfig = DEFAULT_LABEL) -> pd.DataFrame:
    """
    基準日 t のラベル = [t+HORIZON_START, t+HORIZON_END] にブレイク日が1つでもあるか。

    ラベル確定には t+HORIZON_END+HOLD_DAYS 営業日ぶんのデータが必要。
    足りない場合は NaN のままにし、後段で確実に除外する。
    """
    g = df.groupby("Code", sort=False)
    window = cfg.horizon_end - cfg.horizon_start + 1
    need = cfg.forward_needed   # ラベル確定に必要な将来営業日数

    def forward_any(s: pd.Series) -> pd.Series:
        # 逆順 rolling max で [t, t+window-1] の最大値 -> shift で [t+START, t+END] にずらす
        return s[::-1].rolling(window, min_periods=1).max()[::-1].shift(-cfg.horizon_start)

    df["label"] = g["is_breakout"].transform(forward_any)

    # 将来データが足りない行は「未確定」。0（起きなかった）にしてはいけない。
    df["_pos_from_end"] = g.cumcount(ascending=False)
    df.loc[df["_pos_from_end"] < need, "label"] = np.nan
    df = df.drop(columns=["_pos_from_end"])

    return df


# --------------------------------------------------------------------------- #
# 財務系の特徴量（過去3決算）
# --------------------------------------------------------------------------- #

def quarterize_panel(fins: pd.DataFrame) -> pd.DataFrame:
    """
    累計ベースの決算を単一四半期に差分展開し、前年同期比を付ける。

    scripts/jquants_data_fetcher.py の quarterize() と同じ考え方を
    pandas でベクトル化したもの（4,441銘柄 × 10年を回すため）。
    """
    q_map = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}
    df = fins.copy()
    df["quarter"] = df["CurPerType"].map(q_map)
    # 実績値を持つ開示のみ（業績予想の修正だけの開示を除く）
    has_actual = df[["Sales", "OP", "NP", "EPS"]].notna().any(axis=1)
    df = df[df["quarter"].notna() & has_actual].copy()
    df["quarter"] = df["quarter"].astype(int)
    df["DiscDate"] = pd.to_datetime(df["DiscDate"])

    # 同一(銘柄, 会計年度, 四半期)の重複開示は最後の開示を採用（訂正を反映）
    df = (df.sort_values(["Code", "CurFYSt", "quarter", "DiscDate", "DiscTime"])
            .drop_duplicates(["Code", "CurFYSt", "quarter"], keep="last"))

    df = df.sort_values(["Code", "CurFYSt", "quarter"]).reset_index(drop=True)
    grp = df.groupby(["Code", "CurFYSt"], sort=False)

    # 会計年度内で1つ前の四半期との差分を取る（1Q は累計=単期）
    for src, dst in [("Sales", "q_sales"), ("OP", "q_op"), ("NP", "q_np"), ("EPS", "q_eps")]:
        prev_val = grp[src].shift(1)
        prev_q = grp["quarter"].shift(1)
        contiguous = prev_q == df["quarter"] - 1
        df[dst] = np.where(
            df["quarter"] == 1, df[src],
            np.where(contiguous, df[src] - prev_val, np.nan),
        )

    # 進捗率: 当期累計営業利益 / 通期会社予想営業利益
    df["progress_rate"] = np.where(
        (df["FOP"] > 0) & df["OP"].notna(), df["OP"] / df["FOP"] * 100.0, np.nan
    )
    df["progress_vs_base"] = df["progress_rate"] - df["quarter"] * 25.0

    # 営業利益率（単一四半期）
    df["op_margin"] = np.where(
        df["q_sales"] > 0, df["q_op"] / df["q_sales"] * 100.0, np.nan
    )

    # --- 前年同期比（同じ四半期どうしを比較） --- #
    df = df.sort_values(["Code", "quarter", "CurFYSt"]).reset_index(drop=True)
    by_cq = df.groupby(["Code", "quarter"], sort=False)
    for src, dst in [("q_sales", "sales_growth"), ("q_eps", "eps_growth")]:
        prev = by_cq[src].shift(1)
        # 前年が0以下なら成長率は定義できない（赤字→黒字を+1000%等と表現しない）
        df[dst] = np.where(prev > 0, (df[src] - prev) / prev * 100.0, np.nan)

    # 自己資本・株数から時価総額を出すための情報も残す
    df["shares_out"] = df["ShOutFY"] - df["TrShFY"].fillna(0)
    df.loc[df["shares_out"] <= 0, "shares_out"] = np.nan

    df = df.sort_values(["Code", "DiscDate"]).reset_index(drop=True)

    # --- 直近3決算をラグ列として横に並べる --- #
    axes = ["eps_growth", "sales_growth", "ROE", "op_margin"]
    g2 = df.groupby("Code", sort=False)
    for a in axes:
        df[f"{a}_q0"] = df[a]
        df[f"{a}_q1"] = g2[a].shift(1)
        df[f"{a}_q2"] = g2[a].shift(2)
        # 変化と傾き（CANSLIM の核心は水準より「加速」）
        df[f"{a}_chg"] = df[f"{a}_q0"] - df[f"{a}_q2"]
        df[f"{a}_chg1"] = df[f"{a}_q0"] - df[f"{a}_q1"]
        df[f"{a}_slope"] = (df[f"{a}_q0"] - df[f"{a}_q2"]) / 2.0

    keep = (["Code", "DiscDate", "quarter", "progress_vs_base", "shares_out"]
            + [c for a in axes for c in
               (f"{a}_q0", f"{a}_q1", f"{a}_q2", f"{a}_chg", f"{a}_chg1", f"{a}_slope")])
    return df[keep]


# --------------------------------------------------------------------------- #
# 横断面正規化
# --------------------------------------------------------------------------- #

def add_cross_sectional_ranks(df: pd.DataFrame, cols: List[str],
                              date_col: str = "Date") -> pd.DataFrame:
    """
    各列を「同じ日付内でのパーセンタイル順位」(0〜1) に変換した列を追加する。

    元の列は残す。絶対値と順位のどちらが効くかを比較できるようにするため。
    追加される列名は `<元の列>_r`。

    欠測はそのまま欠測にする。0.5 等で埋めると「中位だった」という
    観測していない情報を与えることになるため。
    その日に有効な値が2件未満なら順位が定義できないので欠測にする。
    """
    out = df.copy()
    g = out.groupby(date_col, sort=False)
    added = 0
    for c in cols:
        if c not in out.columns:
            continue
        ranked = g[c].rank(pct=True, method="average")   # 0〜1、NaN は NaN のまま
        valid = g[c].transform("count") >= 2
        out[f"{c}_r"] = ranked.where(valid)
        added += 1
    print(f"[rank] {added}列の順位版を追加")
    return out


# --------------------------------------------------------------------------- #
# 組み立て
# --------------------------------------------------------------------------- #

def build(data_dir: str, out_path: str) -> pd.DataFrame:
    bars = load_parts("bars", data_dir)
    fins = load_parts("fins", data_dir)
    topix = load_parts("topix", data_dir)
    try:
        margin = load_parts("margin", data_dir)
    except SystemExit:
        print("[warn] margin データなし。需給軸は欠測として扱います")
        margin = pd.DataFrame(columns=["Date", "Code", "LongVol", "ShrtVol"])

    print("\n[panel] 株価系の指標を算出")
    df = price_panel(bars)
    print("[panel] ブレイクアウト日を判定")
    df = breakout_flags(df)
    print("[panel] ラベルを付与")
    df = attach_labels(df)

    # --- 月末営業日だけをサンプルとして抜き出す --- #
    df["ym"] = df["Date"].dt.to_period("M")
    is_month_end = df.groupby(["Code", "ym"], sort=False)["Date"].transform("max") == df["Date"]
    samples = df[is_month_end].copy()
    print(f"[sample] 月末サンプル: {len(samples):,}行")

    # --- 除外条件 --- #
    before = len(samples)
    samples = samples[samples["label"].notna()]
    print(f"[filter] ラベル未確定を除外: {before:,} -> {len(samples):,}")

    before = len(samples)
    samples = samples[samples["high52w"].notna()]
    print(f"[filter] 52週高値が未定義(上場直後)を除外: {before:,} -> {len(samples):,}")

    before = len(samples)
    samples = samples[samples["r_high"] < MAX_RHIGH_AT_T]
    print(f"[filter] 基準日ですでに高値圏(R_high>={MAX_RHIGH_AT_T})を除外: {before:,} -> {len(samples):,}")

    before = len(samples)
    samples = samples[samples["tv_ma20"] >= MIN_TRADING_VALUE]
    print(f"[filter] 低流動性(20日平均売買代金<{MIN_TRADING_VALUE}億円)を除外: {before:,} -> {len(samples):,}")

    # --- 財務をマージ（開示日ベースの point-in-time） --- #
    print("\n[merge] 財務情報を開示日ベースで結合")
    q = quarterize_panel(fins)
    samples = samples.sort_values("Date")
    q = q.sort_values("DiscDate")
    samples = pd.merge_asof(
        samples, q,
        left_on="Date", right_on="DiscDate", by="Code",
        direction="backward",   # 基準日までに開示済みの直近決算のみ
        allow_exact_matches=True,
    )
    # 決算が古すぎる（1年以上前）場合は使わない
    stale = (samples["Date"] - samples["DiscDate"]).dt.days > 365
    fin_cols = [c for c in q.columns if c not in ("Code", "DiscDate")]
    samples.loc[stale, fin_cols] = np.nan
    print(f"[merge] 決算が1年以上古いサンプル: {int(stale.sum()):,}件を欠測扱い")

    # --- 時価総額 --- #
    samples["market_cap"] = samples["close"] * samples["shares_out"] / 1e8

    # --- 信用倍率 --- #
    if len(margin):
        margin = margin.copy()
        margin["Date"] = pd.to_datetime(margin["Date"])
        margin["credit_ratio"] = np.where(
            margin["ShrtVol"] > 0, margin["LongVol"] / margin["ShrtVol"], np.nan
        )
        m = margin[["Date", "Code", "credit_ratio"]].sort_values("Date")
        samples = pd.merge_asof(
            samples.sort_values("Date"), m,
            on="Date", by="Code", direction="backward",
            # 信用残は週次公表。3週間以上前の値は古すぎるので使わない
            tolerance=pd.Timedelta("21D"),
        )
    else:
        samples["credit_ratio"] = np.nan

    # --- 市場環境（TOPIX） --- #
    print("[merge] TOPIX の市場環境特徴量を結合")
    tp = topix.copy()
    tp["Date"] = pd.to_datetime(tp["Date"])
    tp = tp.sort_values("Date").reset_index(drop=True)
    tp["topix_ret_20"] = tp["topix"].pct_change(20) * 100
    tp["topix_ret_120"] = tp["topix"].pct_change(120) * 100
    samples = pd.merge_asof(
        samples.sort_values("Date"), tp[["Date", "topix_ret_20", "topix_ret_120"]],
        on="Date", direction="backward",
    )

    # --- 最終的な特徴量セット --- #
    samples["log_trading_value"] = np.log1p(samples["tv_ma20"])
    samples["log_market_cap"] = np.log1p(samples["market_cap"])

    # --- 横断面正規化 ---
    # 絶対値のままだと相場局面に依存する。上昇局面では全銘柄の R_high が高くなるため、
    # 「R_high が 87%」の意味が期間によって変わってしまう。
    # 同じ日付内での順位（パーセンタイル）に直すと、
    # 「その時点で全銘柄中どのくらいの位置か」という局面に依らない量になる。
    #
    # 実測で訓練期間の正例率 6.19% に対しテスト期間 21.66% と3倍以上ずれており、
    # 絶対値の特徴量では学習が成立していなかった（docs/MODEL_RESULTS.md 参照）。
    print("\n[rank] 横断面正規化（同一日付内のパーセンタイル順位）")
    samples = add_cross_sectional_ranks(samples, features.RAW_FOR_RANK)

    # 特徴量の一覧は features.py が持つ。データセットには全部作っておき、
    # どれを使うかは学習時にプリセットで選ぶ（特徴量の実験を回しやすくするため）。
    feature_cols = features.all_columns()
    missing = [c for c in feature_cols if c not in samples.columns]
    if missing:
        raise SystemExit(f"features.py が要求する列がありません: {missing}")
    meta_cols = ["Code", "Date", "close", "high52w", "tv_ma20", "market_cap", "label"]

    out = samples[meta_cols + feature_cols].copy()
    out["label"] = out["label"].astype(int)

    print(f"\n[result] {len(out):,}サンプル / 特徴量{len(feature_cols)}個")
    print(f"[result] 正例率: {out['label'].mean()*100:.2f}%  ({int(out['label'].sum()):,}件)")
    print(f"[result] 期間: {out['Date'].min().date()} 〜 {out['Date'].max().date()}")
    print(f"[result] 銘柄数: {out['Code'].nunique():,}")

    miss = out[feature_cols].isna().mean().sort_values(ascending=False)
    print("\n[欠測率の高い特徴量]")
    for name, rate in miss.head(8).items():
        print(f"  {name:<24} {rate*100:5.1f}%")

    # どの定義で作ったデータセットかを残す。あとから追跡できないと混乱するため。
    meta = {
        "labelConfig": {
            "high_window": DEFAULT_LABEL.high_window,
            "horizon": [DEFAULT_LABEL.horizon_start, DEFAULT_LABEL.horizon_end],
            "hold_days": DEFAULT_LABEL.hold_days,
            "hold_drawdown": DEFAULT_LABEL.hold_drawdown,
            "vol_multiple": DEFAULT_LABEL.vol_multiple,
            "sustain_days": DEFAULT_LABEL.sustain_days,
            "sustain_ratio": DEFAULT_LABEL.sustain_ratio,
            "name": DEFAULT_LABEL.name,
            "forward_needed": DEFAULT_LABEL.forward_needed,
        },
        "n": int(len(out)), "positiveRate": round(float(out["label"].mean()), 4),
        "features": feature_cols,
        "from": str(out["Date"].min().date()), "to": str(out["Date"].max().date()),
    }
    with open(os.path.join(os.path.dirname(out_path), "dataset_meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[label] 定義: {DEFAULT_LABEL.name} "
          f"(ラベル確定に将来 {DEFAULT_LABEL.forward_needed} 営業日)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    print(f"\n[done] {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)")
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ブレイクアウト予測の学習データを構築する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "dataset.parquet"))
    args = ap.parse_args(argv)
    build(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
