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
#
# 一時 78週 + 小型株限定に変更したが、定義Eに戻した。
# あの変更は「決算特徴量に予測力が無い」という結論への対応だったが、
# その結論自体が決算データの欠損（ROE_chg が94.4%欠測）によるもので、
# 前提が成り立っていなかった。データを直したうえで元の定義から測り直す。
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
# 時価総額の帯（億円）。基準日時点で判定する。None なら絞らない。
#
# 一時 50〜300億円に絞ったが、解除した。
# あの変更は「決算特徴量に予測力が無い」という結論への対応だったが、
# その結論は決算データの欠損によるもので前提が成り立っていなかった。
# 母集団を狭めるとサンプルが142,000 -> 31,859まで減り検出力も落ちる。
# データを直したうえで、まず全銘柄で測り直す。
# 絞りたくなったらここに数値を入れれば戻せる。
MIN_MARKET_CAP = None
MAX_MARKET_CAP = None

# PER / PBR の上限。これを超えたら分母が丸め誤差レベルとみなし欠測にする。
# 逆数（earnings_yield / book_yield）は分母が株価なので発散せず、そちらは残す。
PER_MAX = 1000.0
PBR_MAX = 1000.0
PEG_MAX = 100.0     # PER/成長率。成長率が極小だと発散する


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
    # 未調整の終値。バリュエーションと時価総額に使う。
    #
    # close は分割調整後（AdjC）で、株価位置とブレイク判定にはこちらが要る
    # （調整しないと分割日に偽のブレイクが大量に出る）。
    # 一方 EPS / BPS / 株数は「開示時点のまま」で分割調整されていないため、
    # 調整後株価と組み合わせると分割をまたいだ時点で比率がずれる。
    # 実測では earnings_yield が最大 +1276%（EPSが株価の12.7倍）まで出ていた。
    df["close_raw"] = df["C"]
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

def clip_divergent(s, lo: float, hi: float, name: str = ""):
    """
    比率の発散を止める。範囲外は欠測にし、件数を出す。

    分母が丸め誤差レベルだと比率は桁外れの値になる（実測で
    payout_ratio が 460,000%、guidance_op_growth が 96,285% まで出た）。
    そういう値は情報ではなく雑音なので落とす。

    上限で切り捨てるのではなく欠測にするのは、
    切り捨てると「上限にへばりついた実在の値」に見えてしまうため。

    範囲は「実在しうるか」で決める。ROE のように 100% を超えることが
    実際にある指標を、比率だからと機械的に切ってはいけない。
    """
    v = pd.to_numeric(s, errors="coerce")
    bad = v.notna() & ((v < lo) | (v > hi) | ~np.isfinite(v))
    n = int(bad.sum())
    if n and name:
        print(f"[clip] {name}: 範囲外 {n:,}件を欠測に "
              f"（{lo:g} 〜 {hi:g} の外）")
    return v.mask(bad)


def _lag_available(df: pd.DataFrame, col: str, n: int,
                   max_gap_days: int = 800) -> pd.Series:
    """
    銘柄ごとに「n個前の "値がある" 開示」の値を返す。

    開示単位の shift(n) だと、間に値の無い開示が1つ挟まった時点で切れる。
    ROE は通期開示にしか入らないため、shift(1) はほぼ NaN になっていた
    （実測: q0 がある行のうち q1 もあるのは 6.5%。
      結果 ROE_chg は 94.4% が欠測し、
      「決算に予測力が無い」という結論の根拠になってしまっていた）。
    値のある開示だけを詰めてからずらす。

    古すぎる開示との比較は意味が無いので、開示日の間隔に上限を置く。
    既定の800日は、年1回しか出ない項目の2期前（≒730日）まで許す値。

    過去方向にしかずらさないので先読みは起きない。
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df[col].notna()
    if not valid.any():
        return out
    sub = df.loc[valid]
    g = sub.groupby("Code", sort=False)
    prev_val = g[col].shift(n)
    prev_date = g["DiscDate"].shift(n)
    gap = (sub["DiscDate"] - prev_date).dt.days
    out.loc[valid] = prev_val.where(gap <= max_gap_days)
    return out


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
    #
    # 経常利益とキャッシュフローも同じ累計ベースなので、ここで一緒に展開する。
    # これらを後段（DiscDate でソートし直した後）でやると、
    # 会計年度内の並びが崩れたグループを使うことになり、行がずれる。
    cumulative = [("Sales", "q_sales"), ("OP", "q_op"), ("NP", "q_np"),
                  ("EPS", "q_eps"), ("OdP", "q_odp"),
                  ("CFO", "q_cfo"), ("CFI", "q_cfi"), ("CFF", "q_cff")]
    for src, dst in cumulative:
        if src not in df.columns:
            df[dst] = np.nan
            continue
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
        # 従来の成長率。前年が0以下だと定義できない
        # （赤字 -> 黒字を +1000% のように表現しないため）
        df[dst] = np.where(prev > 0, (df[src] - prev) / prev * 100.0, np.nan)
        # 対称変化率。分母を |今期|+|前期| にすることで
        # 前年が赤字でも定義でき、値は -100〜+100 に収まる。
        #
        # 前年が0以下のときに欠測にする扱いは、赤字企業を丸ごと捨てていた。
        # 小型株は赤字企業の比率が高く、しかも赤字->黒字転換は
        # 株価が最も動くイベントなので、そこを落とすのは損失が大きい
        # （実測で eps_growth の充足率は32.4%しかなかった）。
        denom = df[src].abs() + prev.abs()
        df[f"{dst}_sym"] = np.where(denom > 0,
                                    (df[src] - prev) / denom * 100.0, np.nan)
        # 赤字 -> 黒字の転換そのものをフラグとして持つ
        df[f"{dst}_turn"] = np.where(prev.notna() & df[src].notna(),
                                     ((prev <= 0) & (df[src] > 0)).astype(float),
                                     np.nan)

    # 自己資本・株数から時価総額を出すための情報も残す
    df["shares_out"] = df["ShOutFY"] - df["TrShFY"].fillna(0)
    df.loc[df["shares_out"] <= 0, "shares_out"] = np.nan

    df = df.sort_values(["Code", "DiscDate"]).reset_index(drop=True)

    # --- ROE を TTM で補完 --- #
    # V2 の ROE は通期開示にしか入っていない
    # （実測: 1Q/2Q/3Q/4Q すべて 0.0%、FY のみ 61.1%。
    #   docs/MODEL_FUNDAMENTAL_COVERAGE.md 参照）。
    # そのままでは四半期サンプルで 23.5% しか埋まらない。
    # scripts/jquants_data_fetcher.py は既に TTM 補完を持っているのに、
    # 研究側のパイプラインだけ提供値をそのまま使っていた。
    g_code = df.groupby("Code", sort=False)
    ttm_np = g_code["q_np"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    roe_ttm = np.where(df["Eq"] > 0, ttm_np / df["Eq"] * 100.0, np.nan)
    df["roe_basis"] = np.where(
        df["ROE"].notna(), "provided",
        np.where(np.isfinite(roe_ttm), "ttm", "none"))
    # ROE/ROA は 100% 超が実在するので上限は広く取る。
    # ただし自己資本が極小だと桁外れになる（実測で ROE -178,300%）
    df["ROE"] = clip_divergent(
        df["ROE"].where(df["ROE"].notna(), pd.Series(roe_ttm, index=df.index)),
        -500.0, 500.0, "ROE")

    # --- ROA / BPS / 自己資本比率 --- #
    # 方針: API が返す比率をそのまま使わず、充足率の高い素の項目から計算する。
    # 実測（docs/DATA_FIELDS.md）:
    #   Eq 94.9% / TA 94.9% / ShEq 94.7% / EPS 94.8% / ShOutFY 94.9%
    #   一方 ROE 32.4%（通期のみ） / BPS 46.7%（ほぼ通期のみ） / NCROE 0.0%
    #   ROA と PER と PBR は項目として存在しない。
    ttm_np_roa = g_code["q_np"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    df["ROA"] = clip_divergent(
        pd.Series(np.where(df["TA"] > 0, ttm_np_roa / df["TA"] * 100.0, np.nan),
                  index=df.index), -500.0, 500.0, "ROA")

    # PER 用の12ヶ月EPS。単期EPSの4期和。
    # 赤字（0以下）でも値は残す。PER は後段で符号を見て扱う
    df["eps_ttm"] = g_code["q_eps"].transform(lambda s: s.rolling(4, min_periods=4).sum())

    # BPS は提供値が46.7%しか無いので、株主資本と株数から作る（約94%）。
    # 提供値があるときはそれを優先し、無いところだけ埋める。
    sh = df["ShOutFY"] - df["TrShFY"].fillna(0) if "TrShFY" in df.columns else df["ShOutFY"]
    sh = sh.where(sh > 0)
    eq_for_bps = df["ShEq"] if "ShEq" in df.columns else df["Eq"]
    bps_calc = eq_for_bps / sh
    df["BPS"] = (df["BPS"] if "BPS" in df.columns
                 else pd.Series(np.nan, index=df.index))
    df["bps_basis"] = np.where(df["BPS"].notna(), "provided",
                               np.where(bps_calc.notna(), "calc", "none"))
    df["BPS"] = df["BPS"].where(df["BPS"].notna(), bps_calc)

    # 自己資本比率。常に Eq / TA から計算する（単位を揃えるため）。
    #
    # API の EqAR を優先していたが、EqAR は比率（0〜1）で返り、
    # 計算側は % だったため単位が混在していた。
    # 実測で中央値 0.53（比率）と最大 79.3（%）が同居しており、
    # 同じ列に2つの尺度が混ざっていた。
    # Eq(94.9%) と TA(94.9%) は EqAR(94.8%) と充足率が変わらないので、
    # 提供値を使う利点が無い。
    df["equity_ratio"] = np.where(df["TA"] > 0, df["Eq"] / df["TA"] * 100.0, np.nan)

    # --- 配当・キャッシュフロー・その他の比率 --- #
    # 使う項目は docs/DATA_FIELDS.md の実測値に基づく。
    # 存在しない項目は作らない（EV/EBITDA は有利子負債の項目が無いため不可）。
    def ttm(col: str):
        """単期の値を4期合計して12ヶ月ぶんにする。"""
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return g_code[col].transform(lambda s: s.rolling(4, min_periods=4).sum())

    def col(name: str):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    sales_ttm = ttm("q_sales")
    op_ttm = ttm("q_op")
    np_ttm = ttm("q_np")
    df["sales_ttm"] = sales_ttm

    # 経常利益（88.9%）の差分展開も上のループで済ませてある（q_odp）
    odp_ttm = ttm("q_odp")

    # 利益率（TTM ベース。単期だと季節性で振れる）
    # 利益率は売上が極小の会社で発散する（実測で -166,800% まで出た）
    df["net_margin"] = clip_divergent(
        pd.Series(np.where(sales_ttm > 0, np_ttm / sales_ttm * 100.0, np.nan),
                  index=df.index), -500.0, 100.0, "net_margin")
    df["ordinary_margin"] = clip_divergent(
        pd.Series(np.where(sales_ttm > 0, odp_ttm / sales_ttm * 100.0, np.nan),
                  index=df.index), -500.0, 100.0, "ordinary_margin")

    # 総資産回転率
    df["asset_turnover"] = np.where(col("TA") > 0, sales_ttm / col("TA"), np.nan)

    # --- キャッシュフロー --- #
    # 累計からの差分展開は上のループで済ませてある（q_cfo / q_cfi / q_cff）。
    #
    # TTM（4期合計）にすると充足率が 50.8% -> 11.5% まで落ちる。
    # キャッシュフロー計算書を四半期ごとに出す会社が少なく、
    # 半期・通期しか出さない会社では4期が揃わないため。
    # 代わりに「開示時点の累計値」をそのまま使う。
    # 累計は期首からの積み上げなので、利益側も同じ累計と比べれば整合する。
    cfo_cum = col("CFO")
    cfi_cum = col("CFI")
    df["cfo_cum"] = cfo_cum
    # フリーCF = 営業CF + 投資CF（投資CFは通常負なので加算でよい）
    df["fcf_cum"] = cfo_cum + cfi_cum
    # 利益の質: 営業CFが営業利益をどれだけ裏付けているか。
    # 分母も同じ期間の累計（OP）にそろえる
    df["cfo_to_op"] = clip_divergent(
        pd.Series(np.where(col("OP") > 0, cfo_cum / col("OP") * 100.0, np.nan),
                  index=df.index), -1000.0, 2000.0, "cfo_to_op")
    # アクルーアル: 利益と営業CFの乖離。大きいほど利益の質が低い。
    # 分子は同じ累計期間の純利益にそろえる
    df["accruals"] = clip_divergent(
        pd.Series(np.where(col("TA") > 0,
                           (col("NP") - cfo_cum) / col("TA") * 100.0, np.nan),
                  index=df.index), -200.0, 200.0, "accruals")

    # --- 配当 --- #
    # 会社予想の年間配当（57.8%）を優先し、無ければ実績（32.4%）
    div = col("FDivAnn")
    div = div.where(div.notna(), col("DivAnn"))
    df["dps"] = div
    df["has_dividend"] = np.where(div.notna(), (div > 0).astype(float), np.nan)
    # 配当性向。提供値（22.7%）が無ければ EPS から計算
    payout_calc = np.where(df["EPS"] > 0, div / df["EPS"] * 100.0, np.nan)
    df["payout_ratio"] = clip_divergent(
        col("PayoutRatioAnn").where(col("PayoutRatioAnn").notna(),
                                    pd.Series(payout_calc, index=df.index)),
        -100.0, 1000.0, "payout_ratio")

    # --- 会社予想（今期の伸び見通し）--- #
    # 予想営業利益 / 前期実績営業利益。1を超えれば増益見通し
    prev_op_ttm = g_code["q_op"].transform(
        lambda s: s.shift(4).rolling(4, min_periods=4).sum())
    df["guidance_op_growth"] = clip_divergent(
        pd.Series(np.where(prev_op_ttm > 0,
                           col("FOP") / prev_op_ttm * 100.0 - 100.0, np.nan),
                  index=df.index), -100.0, 1000.0, "guidance_op_growth")
    # 予想の修正: 同じ会計年度で前回開示の予想と比べて何%動いたか。
    # 上方修正は「プラスアルファの好材料」そのもの
    prev_fop = df.groupby(["Code", "CurFYSt"], sort=False)["FOP"].shift(1) \
        if "FOP" in df.columns else pd.Series(np.nan, index=df.index)
    df["guidance_revision"] = clip_divergent(
        pd.Series(np.where(prev_fop > 0,
                           col("FOP") / prev_fop * 100.0 - 100.0, np.nan),
                  index=df.index), -100.0, 1000.0, "guidance_revision")

    # --- 直近4決算をラグ列として横に並べる --- #
    # 52週高値のブレイクは、3〜4決算続けて好調な銘柄で起きる。
    # レーダーチャートを複数時点で重ねて表示しているのも、
    # 1時点の形ではなく「推移」を見るため。特徴量も推移を持つ必要がある。
    axes = ["eps_growth", "sales_growth", "eps_growth_sym", "sales_growth_sym",
            "ROE", "ROA", "op_margin", "equity_ratio"]
    for a in axes:
        df[f"{a}_q0"] = df[a]
        for k in (1, 2, 3):
            df[f"{a}_q{k}"] = _lag_available(df, a, k)

        q0, q1, q2, q3 = (df[f"{a}_q{k}"] for k in range(4))

        # --- 決算をまたぐ各段の差分 --- #
        # chg1 だけでは「直近1回の変化」しか見えない。
        # 各段の差を持つことで「毎回伸びているか」を表現できる。
        #
        # 線形モデルは q1 と q2 から差を作れるが、決定木は個別の列で分岐するので
        # 差を作れない。連言条件（3期とも増加）はそもそも水準の線形結合では
        # 表現できないため、明示的に列として持たせる。
        df[f"{a}_chg1"] = q0 - q1     # 前回 -> 今回
        df[f"{a}_chg2"] = q1 - q2     # 2回前 -> 前回
        df[f"{a}_chg3"] = q2 - q3     # 3回前 -> 2回前
        # 2期ぶん・3期ぶんの変化
        df[f"{a}_chg"] = q0 - q2
        df[f"{a}_chg_3q"] = q0 - q3
        df[f"{a}_slope"] = (q0 - q2) / 2.0
        # 加速: 変化そのものが増えているか（CANSLIM の核心）
        df[f"{a}_accel"] = df[f"{a}_chg1"] - df[f"{a}_chg2"]

        # --- 連続性 --- #
        # 「何期続けて伸びているか」「何期プラスを保っているか」。
        # 欠測は数えず、有効な期が2つ未満なら NaN にする
        levels = [q0, q1, q2, q3]
        avail = pd.concat([s.notna() for s in levels], axis=1).sum(axis=1)
        enough = avail >= 2

        # 直近から数えて何段連続で増加しているか（0〜3）
        steps = [df[f"{a}_chg1"], df[f"{a}_chg2"], df[f"{a}_chg3"]]
        up = pd.Series(0.0, index=df.index)
        alive = pd.Series(True, index=df.index)
        for s in steps:
            inc = (s > 0).fillna(False) & s.notna()
            up = up + (alive & inc).astype(float)
            alive = alive & inc
        df[f"{a}_up_streak"] = up.where(enough)

        # 有効な期のうち、水準がプラスだった割合（0〜1）
        pos = pd.concat([(s > 0) & s.notna() for s in levels], axis=1).sum(axis=1)
        df[f"{a}_pos_ratio"] = (pos / avail.where(avail > 0)).where(enough)

    keep = (["Code", "DiscDate", "quarter", "progress_vs_base", "shares_out",
             "eps_growth_turn", "sales_growth_turn", "BPS", "bps_basis", "roe_basis",
             # 配当・キャッシュフロー・会社予想・その他の比率
             "dps", "has_dividend", "payout_ratio",
             "sales_ttm", "cfo_cum", "fcf_cum", "cfo_to_op", "accruals",
             "net_margin", "ordinary_margin", "asset_turnover",
             "guidance_op_growth", "guidance_revision"]
            + [c for c in ("EPS", "eps_ttm") if c in df.columns]
            + [c for a in axes for c in (
                f"{a}_q0", f"{a}_q1", f"{a}_q2", f"{a}_q3",
                f"{a}_chg1", f"{a}_chg2", f"{a}_chg3",
                f"{a}_chg", f"{a}_chg_3q", f"{a}_slope", f"{a}_accel",
                f"{a}_up_streak", f"{a}_pos_ratio")])
    return df[[c for c in keep if c in df.columns]]


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
    # 時価総額は未調整終値 × 開示時点の株数。
    # 調整後株価を使うと、後年の分割ぶんだけ過小評価される
    samples["market_cap"] = samples["close_raw"] * samples["shares_out"] / 1e8

    # --- バリュエーション --- #
    # PER / PBR は API に項目が無い（docs/DATA_FIELDS.md の実測）ので、
    # 基準日の株価と決算値から作る。株価は基準日 t のもの、
    # 決算は t 以前に開示されたものだけを使っている（merge_asof）ので先読みは無い。
    #
    # 赤字のとき PER は負になり「割安」と誤読される。
    # 逆数の益回り（EPS/株価）にすれば符号がそのまま意味を持ち、
    # 赤字企業も連続量として扱える。PER 自体は黒字のときだけ持つ。
    px = samples["close_raw"]
    samples["earnings_yield"] = np.where(px > 0, samples["eps_ttm"] / px * 100.0,
                                         np.nan)
    # 純資産倍率の逆数。BPS が負（債務超過）でも意味を保つ
    samples["book_yield"] = np.where(px > 0, samples["BPS"] / px, np.nan)

    # PER / PBR は分母が小さいと発散する。
    # 実測で per の最大が 4.0e17 まで出ていた（EPS が丸め誤差レベル）。
    # 逆数側（益回り・純資産倍率の逆数）は分母が株価なので発散せず、
    # そちらを特徴量として持っている。比率側は解釈用と割り切り、
    # 実在しうる範囲を超えたものは欠測にする。
    # --- 株価との比で作る指標 --- #
    mc = samples["market_cap"]          # 億円
    # PSR = 時価総額 / 売上高(TTM)。売上は円なので億円に直す
    sales_oku = samples["sales_ttm"] / 1e8
    samples["psr"] = clip_divergent(
        pd.Series(np.where(sales_oku > 0, mc / sales_oku, np.nan),
                  index=samples.index), 0.0, 1000.0, "psr")
    samples["sales_yield"] = np.where(mc > 0, sales_oku / mc * 100.0, np.nan)
    # キャッシュフロー利回り
    samples["cfo_yield"] = clip_divergent(
        pd.Series(np.where(mc > 0, samples["cfo_cum"] / 1e8 / mc * 100.0, np.nan),
                  index=samples.index), -500.0, 500.0, "cfo_yield")
    samples["fcf_yield"] = clip_divergent(
        pd.Series(np.where(mc > 0, samples["fcf_cum"] / 1e8 / mc * 100.0, np.nan),
                  index=samples.index), -500.0, 500.0, "fcf_yield")
    # 配当利回り
    samples["div_yield"] = np.where(px > 0, samples["dps"] / px * 100.0, np.nan)

    per = np.where(samples["eps_ttm"] > 0, px / samples["eps_ttm"], np.nan)
    pbr = np.where(samples["BPS"] > 0, px / samples["BPS"], np.nan)
    samples["per"] = np.where(np.isfinite(per) & (per <= PER_MAX), per, np.nan)
    samples["pbr"] = np.where(np.isfinite(pbr) & (pbr <= PBR_MAX), pbr, np.nan)
    # PEG = PER / EPS成長率(%)。成長に対して株価が割高か。
    # 成長率が0以下だと意味を持たない（負のPEGは「割安」ではない）ので欠測にする。
    # 成長率が極端に小さいと発散するため、PER と同じ考え方で上限を置く。
    growth = samples["eps_growth_q0"]
    peg = np.where((samples["per"] > 0) & (growth > 0), samples["per"] / growth,
                   np.nan)
    samples["peg"] = np.where(np.isfinite(peg) & (peg <= PEG_MAX), peg, np.nan)
    n_peg = int((np.isfinite(peg) & (peg > PEG_MAX)).sum())
    if n_peg:
        print(f"[filter] peg > {PEG_MAX:g} を欠測に: {n_peg:,}件（成長率が極小）")

    for name, arr, cap in (("per", per, PER_MAX), ("pbr", pbr, PBR_MAX)):
        n = int((np.isfinite(arr) & (arr > cap)).sum())
        if n:
            print(f"[filter] {name} > {cap:g} を欠測に: {n:,}件"
                  f"（分母が丸め誤差レベル。逆数側は残している）")

    # --- 時価総額の帯で絞る（設定されている場合のみ）--- #
    # 基準日時点で判定する。将来の時価総額は使わない。
    if MIN_MARKET_CAP is not None or MAX_MARKET_CAP is not None:
        lo = MIN_MARKET_CAP if MIN_MARKET_CAP is not None else -np.inf
        hi = MAX_MARKET_CAP if MAX_MARKET_CAP is not None else np.inf
        before = len(samples)
        samples = samples[samples["market_cap"].between(lo, hi)]
        print(f"[filter] 時価総額 {lo}〜{hi}億円の外を除外: "
              f"{before:,} -> {len(samples):,}")

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

    # --- 業種・市場区分を時点別に結合 --- #
    # 最新のマスタを過去のサンプルに当てると先読みになる。
    # とくに市場区分は2022年4月の東証再編で全銘柄が変わっているため、
    # 2018年のサンプルに現在の区分を付けるのは誤り。
    # 月次スナップショットを merge_asof で「その時点で有効だった区分」に合わせる。
    mh_paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if mh_paths:
        mh = pd.concat([pd.read_parquet(x) for x in mh_paths], ignore_index=True)
        mh["Date"] = pd.to_datetime(mh["Date"])
        keep_mh = [c for c in ("Date", "Code", "S33", "S17", "ScaleCat", "Mkt")
                   if c in mh.columns]
        mh = (mh[keep_mh].dropna(subset=["Date", "Code"])
              .sort_values("Date").drop_duplicates(["Date", "Code"], keep="last"))
        print(f"[merge] 業種・市場区分を時点別に結合 ({len(mh):,}行 / "
              f"{mh['Date'].nunique()}時点)")
        samples = pd.merge_asof(
            samples.sort_values("Date"), mh,
            on="Date", by="Code", direction="backward")
        for c in ("S33", "S17", "ScaleCat", "Mkt"):
            if c in samples.columns:
                # カテゴリは数値コードにする（LightGBM はそのまま分岐できる）
                samples[f"{c.lower()}_code"] = pd.to_numeric(samples[c],
                                                             errors="coerce")
    else:
        print("[merge] master_hist が無いため業種・市場区分は付与しない")
    for c in ("s33", "s17", "scalecat", "mkt"):
        if f"{c}_code" not in samples.columns:
            samples[f"{c}_code"] = np.nan

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
    # 特徴量ではないが検証に要る列も残す。
    # eps_ttm と BPS は per / pbr の分母なので、
    # 「per × earnings_yield == 100」のような恒等式の検査に必要
    # （research/validate_metrics.py）。
    # *_basis は提供値と計算値のどちらを使ったかの記録。
    meta_cols = ["Code", "Date", "close", "close_raw", "high52w", "tv_ma20",
                 "market_cap", "label",
                 "eps_ttm", "BPS", "roe_basis", "bps_basis"]
    meta_cols = [c for c in meta_cols if c in samples.columns]

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
