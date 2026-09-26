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
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 画面のデータ取得（進捗期待の基準の規則を共有する）。research の後ろに置き、名前で隠さない
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "scripts"))
import availability as AV  # noqa: E402
import features  # noqa: E402
import extra_features  # noqa: E402
import jquants_data_fetcher as JF  # noqa: E402
import trading_calendar  # noqa: E402

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
def _sweep_override(name: str, default, cast):
    """
    設計の掃引（research/sweep_design.py）から母集団の定義を差し替えるための口。

    環境変数 SWEEP_<name> が入っているときだけ効く。既定の挙動は変えない。
    掃引は「52週にすると母集団が増えるが分離力はどうなるか」のように、
    データセットごと作り直さないと測れない軸があるので必要になる。

    黙って効くと事故になる（気づかないまま別の母集団で学習してしまう）ので、
    効いたときは必ず標準出力に出し、dataset_meta.json にも残す。
    """
    raw = os.environ.get(f"SWEEP_{name}")
    if raw is None or raw.strip() == "":
        return default
    val = cast(raw.strip())
    SWEEP_OVERRIDES[name] = val
    print(f"[sweep] {name} を環境変数で {default} -> {val} に差し替え")
    return val


#: 実際に効いた差し替え。既定で回すかぎり空のまま。
SWEEP_OVERRIDES: dict = {}

#: True にすると、ラベルが確定していない行も残す。
#:
#: 学習では必ず False。ラベルの無い行を混ぜると訓練できない。
#: True にするのは日次予測のときだけで、予測したい行（今日のブレイク）は
#: 定義上まだラベルが無いため。build_dataset.py --keep-unlabeled で立つ。
KEEP_UNLABELED = False

# 78週 ≒ 368営業日。52週(245日)から広げた。
# 52週だと「1年前の高値を1円抜いただけ」も母集団に入り、
# 抜けた水準の重みが軽い。1年半ぶりの高値なら上値の戻り売りが薄い。
HIGH_WINDOW = _sweep_override("HIGH_WINDOW", 368, int)
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

# --------------------------------------------------------------------------- #
# 母集団の定義
# --------------------------------------------------------------------------- #
# "breakout" … 52週高値を更新した日 × 銘柄。運用の形に合わせた設定。
#              その日の夜にモデルを回して「この更新は伸びるか」を判定する。
# "month_end" … 月末営業日 × 銘柄（従来）。
#              「まだ抜けていない銘柄が先1〜6ヶ月で抜けるか」を予測していた。
#
# breakout に変えた理由:
#   従来の基準 r_high（52週高値への近さ）はゴールまでの距離を測っているだけで、
#   ラベル（52週高値を抜けるか）とほぼ同語反復だった。
#   実測でも r_high の Lift は層0で2.29倍、層4（高値至近）で1.05倍と、
#   距離を揃えると何も予測できていなかった（docs/MODEL_STRATIFIED.md）。
#   高値更新日に母集団を固定すれば距離は全銘柄で同じになり、この問題が消える。
POPULATION = "breakout"

# 高値更新の判定は高値ベース（終値ベースではなく）。
# 場中に52週高値を超えた時点で候補になる、という運用の形に合わせる。
BREAKOUT_ON_HIGH = True

# 同じ上昇局面の連続した更新日を1件にまとめる。
# 高値更新が10日続くと、ほぼ同じ特徴量・重なるラベルのサンプルが10件でき、
# 件数が水増しされるうえサンプル間が独立でなくなる。
# 直前 BREAKOUT_COOLDOWN 営業日に更新が無い日だけを「新規のブレイク」とみなす。
BREAKOUT_COOLDOWN = _sweep_override("BREAKOUT_COOLDOWN", 20, int)

# --- 目的変数（母集団が breakout のとき）--- #
# 基準の価格（翌営業日の寄り。LABEL_ENTRY）から、先 RISE_HORIZON 営業日以内に
# RISE_THRESHOLD 以上上昇したか（2026-09-25 までは更新日の終値が基準だった）。
# 終値ベースで測る（高値ベースだと「一瞬触れただけ」を正例にしてしまう）。
# 20営業日 ≒ 1ヶ月。60（約3ヶ月）から短くした。
#
# 理由は運用側にある。実際に手仕舞うのが1ヶ月前後なのに、ラベルの基準点が
# 3ヶ月先にあると「売ったあとに起きたこと」で正例・負例を決めることになる。
# 売買の回転を上げたいなら、ラベルの見る先も持ち期間に合わせる必要がある。
#
# しきい値は k×σ で σ = vol_20d/100 × √horizon なので、期間を縮めると
# 必要な上昇率も自動で √(20/60) = 0.577 倍になる（別途調整は要らない）。
# 終盤条件（END_RATIO）とトレンド条件も同じ t+horizon 時点を見るので、
# この1行で3条件すべての基準点が t+60 から t+20 に移る。
RISE_HORIZON = 20        # 営業日。約1ヶ月
RISE_THRESHOLD = 0.20    # +20%。VOL_NORM_K が None のときだけ効く

# --- 上昇を測る基準の価格 --- #
# 2026-09-25 から **翌営業日の寄り付き（分割調整後の始値 AdjO[t+1]）**（運用者の決定）。
# 前は高値を更新した日の終値（close[t]）だった。候補が分かるのは終値が出た後で、
# 実際に買えるのは翌営業日の寄りなので、収益の計算（lab.realized_returns の ret_o1_*、
# 買い = AdjO[t+1]）と同じ基準にそろえる。変わるのは到達・終盤（・維持日数）の比率の
# 分母だけで、見る窓（t+1〜t+horizon の終値）、しきい値（k×σ）、トレンド条件は変えない。
# 翌営業日に寄りが付かない行は、収益の計算と同じく判定できない（未確定）とする。
# "close" にすると前の定義に戻る（掃引の物差し sweep_design.REF_RISE はこちらのまま）。
LABEL_ENTRY = "next_open"
LABEL_ENTRIES = ("next_open", "close")

# --- 到達しきい値を銘柄自身のボラティリティで測る --- #
#
# 固定の +20% は銘柄ごとの難易度がまったく揃っていない。実測の期間σは
#   20営業日 中央  8.7% / p5 3.4% / p95 27.6%   （8.0倍の開き）
#   60営業日 中央 15.0% / p5 6.0% / p95 47.8%   （8.0倍の開き）
# なので、同じ +20% が静かな銘柄と荒い銘柄で何倍も難易度が違う
# （60営業日なら静かな銘柄に 3.3σ、荒い銘柄に 0.42σ）。
# 期間を縮めても開きは 8 倍のまま変わらないので、正規化の必要性も変わらない。
# 正例になりやすさが定義の時点で銘柄ごとに何倍も違っていた。
#
# その結果、モデルは「上がる銘柄を当てる係」ではなく
# 「荒い銘柄を選ぶ係」になっていた。実測（docs/MODEL_DESIGN_SWEEP.md）:
#   固定+20% の上位5%   日次ボラ中央 3.32%（母集団 1.78%）
#   同じ期間に高ボラ順で機械的に買うと 実収益 -21.15pt / 勝率25.9%
#
# しきい値を k×σ（σ = vol_20d/100 × √horizon）にすると選ぶ銘柄が反転し、
# 10窓のウォークフォワードでも実収益の差が +5.02pt [+3.19,+5.96] と有意になった
# （docs/MODEL_DESIGN_WALKFORWARD.md）。k は 1.0〜1.6σ のどれでも有意で、
# 1.2σ が最も良かったので採用する。
#
# None にすると固定 RISE_THRESHOLD に戻る。
VOL_NORM_K: Optional[float] = 1.2

# --- 継続の軸 --- #
# 「到達したか」だけだと、一瞬吹き上げてすぐ下落トレンドに入った銘柄も
# 正例になる。それはモメンタムではないので、続いたことを条件に加える。
# 3つとも独立に効かせられる（0 / None で無効）。
#
#   keep_days  … +threshold の水準を通算何営業日保ったか。瞬間的なヒゲを外す
#   end_ratio  … ホライズン終盤の水準（基準の価格比。LABEL_ENTRY）。失速・往って来いを外す
#   uptrend    … ホライズン終了時点で短期移動平均 >= 長期移動平均。
#                下落トレンドに転換したものを外す
# しきい値は8定義の比較とチャートの目視で決めた（緩い案 = 定義F）。
#
# 一度は厳しいほう（定義G: 維持20日 / 終盤+15%）を採用したが、正例率が
# 7.49% まで落ちて 1,316件しか残らず、ウォークフォワードの各窓に
# 正例が14〜110件しか入らなくなった。判定が数十件の増減で揺れる。
# 緩い案なら 11.55% / 2,030件で、正例が1.5倍になる。
# 差の714件はチャートで見ると「+20%に届いたあと10〜20日で失速した」群で、
# 買えないほど悪いわけではない。まずは推定を安定させることを優先する。
KEEP_DAYS = 0            # 維持日数の条件。0 なら課さない
END_RATIO = 0.10         # t+horizon 後もまだ +10%以上（ボラ正規化時は比例させる）
END_WINDOW = 5           # 終盤の水準は5営業日平均で見る（1日の綾を拾わない）

# トレンド条件の移動平均。**RISE_HORIZON に合わせて選ぶ必要がある**。
#
# 長期側は「判定期間そのもの」、短期側はその終盤だけを見る長さにする。
# こうすると条件の意味は「窓の終わりにかけてまだ上にいるか」になる。
#
#   h=60 のとき MA20/MA60 … MA20 は [t+41, t+60]（終盤1/3）、MA60 は [t+1, t+60]
#   h=20 のとき MA5/MA20  … MA5  は [t+16, t+20]（終盤1/4）、MA20 は [t+1, t+20]
#
# h を 20 にしたとき MA20/MA60 のままにしていたら、条件がほぼ無効になっていた。
# MA60(t+20) は [t-39, t+20] の平均で、大半がブレイク**前**の安い期間になる。
# 基準日は78週高値の更新日なので、MA20(t+20)（すべてブレイク後）がそれを
# 下回るには相当な下落が要る。実測で 5,671件から削るのが**2件（0.04%）**
# だけだった（h=60 では473件削っていた）。条件として機能していないのに
# 名前だけ残る状態だったので、長さを h に合わせた。
TREND_SHORT = 5          # 短期移動平均（営業日）= 判定期間の終盤
TREND_LONG = 20          # 長期移動平均（営業日）= 判定期間そのもの（= RISE_HORIZON）
REQUIRE_UPTREND = True

# 窓の中で実際に値がある割合の下限。
# 売買が成立しない日は高値が欠測になる（実測で全行の約3%）。
# 欠測を許さないと1つの欠測が窓ぶんの判定を潰すが、
# 許しすぎると長期休止銘柄の「数点だけの高値」を基準にしてしまう。
MIN_WINDOW_COVERAGE = 0.5


def _flag(raw: str) -> bool:
    """「1 / true / on / yes」を True にする（環境変数からの差し替え用）。"""
    return raw.strip().lower() in ("1", "true", "on", "yes")


#: TOKYO PRO MARKET（プロ向け市場）の市場区分名（master_hist の MktNm）
TPM_NAME = "TOKYO PRO MARKET"

#: 78週の窓（HIGH_WINDOW 本）を、**一般市場に移ってからの行だけ**で数えるか
#: （運用者の選択 A、2026-09-25）。
#:
#: J-Quants は TOKYO PRO MARKET の銘柄にも日足の行を持つが、取引がほとんど無く
#: 値はほぼ空（いま TPM の188銘柄で、全期間 120,064行のうち値のある行は 666）。
#: 一般市場へ移った銘柄は、その空の行も「368本」に数えられ、窓の半分以上に値が
#: あれば通ってしまう。5537 は実際には約9か月（188日）の高値が「78週高値」として
#: 扱われ、2026-09-24 の予測候補に入った。
#: 「最初に値が付いた日から数える」だけでは直らない: 移った9銘柄のうち8銘柄は、
#: TPM に上場した日などに値のある行を持っている（5537 も 2023-11-29 に1行）。
#: そこで master_hist で最後に TPM だった月末を調べ、その後で最初に値が付いた日から
#: 数える（general_market_start）。
#:
#: 2026-09-25 から True（運用者の決定。データの誤りを直すもので、分離力の足切りでは
#: 決めない）。実験47（1回目）の実測: 学習データの行は 21,867 -> 21,864（3行減る）、
#: 共通の行で値が変わった列は0本。画面の78週高値も同じ切り替えに従う
#: （general_market_start.json の enabled）
GENERAL_MARKET_START = _sweep_override("GENERAL_MARKET_START", True, _flag)

#: 上場からの年数（listing_years、実験47の候補）の打ち止め（年）。
#: J-Quants は 2016-10 より前が見えないので、それより前から上場している銘柄の年数は
#: 分からない。5年で打ち止めにすれば、データの初日から5年たった 2021-10 以降は
#: 古い銘柄も「5」と確定する（運用者の選択 ②、2026-09-25。①の3年から変えた）。
#: 実測（手元の学習データ 21,856行）: 値あり 70.6%（5年未満 6.1%）。欠測は 2018〜2021年の
#: 古い銘柄に偏る（年ごとの値あり 2018 1.4% / 2019 3.5% / 2020 7.8% / 2021 23.5% / 2022〜 100%）
#: ので、欠測がそのまま「時期」の目印になる。実験47の対照 P（同じ日の中で入れ替え）は
#: 時期の情報を残して銘柄との対応だけを壊すので、L − P でその分を差し引いて読む
LISTING_CAP_YEARS = 5.0

# --- 決算の完全性で母集団を絞る --- #
# 決算の「変化」で判断させるなら、変化が作れないレコードを混ぜても
# 欠測を学習させるだけになる。作れない行は最初から外す。
#
# 候補は狭いものから広いものまであり、残る件数と正例率が変わる。
# どれを採るかは実測を見て決めるので、候補は全部数えて出す。
FUND_REQUIREMENT_SETS = {
    "none": [],
    # 直近1回の変化（売上とEPS）
    "growth1": ["eps_growth_chg1", "sales_growth_chg1"],
    # 2段の変化。「毎回伸びているか」を表せる最小構成
    "growth2": ["eps_growth_chg1", "eps_growth_chg2",
                "sales_growth_chg1", "sales_growth_chg2"],
    # 収益性の変化も要求する
    "profit2": ["eps_growth_chg1", "eps_growth_chg2",
                "sales_growth_chg1", "sales_growth_chg2",
                "ROE_chg1", "ROE_chg2", "op_margin_chg1", "op_margin_chg2"],
    # 4期そろい（3段の変化がすべて作れる）
    #
    # 注意: eps_growth / sales_growth は「前年同期が0以下だと欠測」という
    # 定義なので、これを要求すると前年4期すべて黒字だった会社しか残らない。
    # 実測でその副作用が出ている（EDA):
    #   eps_growth_turn      全10,116行が 0（赤字->黒字が1件も無い）
    #   eps_growth_sym_q0    max 99.939 で 100 に届かない（同上）
    #   同 p5 = -100          黒字->赤字転落は5%以上ある（非対称に落ちている）
    "full4": ["eps_growth_chg1", "eps_growth_chg2", "eps_growth_chg3",
              "sales_growth_chg1", "sales_growth_chg2", "sales_growth_chg3",
              "ROE_chg1", "ROE_chg2", "op_margin_chg1", "op_margin_chg2"],
    # full4 と同じ強さの要求を、対称変化率（_sym）で掛けたもの。
    # _sym は分母が |今期|+|前期| なので前年が赤字でも定義でき、
    # 赤字->黒字転換の会社を落とさない。
    # ROE_chg / op_margin_chg は水準の差分なので符号に依存せず、そのまま。
    "full4_sym": ["eps_growth_sym_chg1", "eps_growth_sym_chg2", "eps_growth_sym_chg3",
                  "sales_growth_sym_chg1", "sales_growth_sym_chg2",
                  "sales_growth_sym_chg3",
                  "ROE_chg1", "ROE_chg2", "op_margin_chg1", "op_margin_chg2"],
}
#: 実際に適用する要求。FUND_REQUIREMENT_SETS のキー。
#:
#: full4_sym = 売上・EPS の3段の差分（対称変化率）と、ROE・営業利益率の
#: 2段の差分がすべて作れること。要求の強さは full4 と同じ10列。
#:
#: full4 から full4_sym に変えた理由:
#:   full4 は eps_growth / sales_growth（前年同期が0以下だと欠測）を要求するので、
#:   前年4期すべて黒字だった会社しか残らなかった。実測でその副作用が出ていた。
#:     eps_growth_turn / sales_growth_turn  全10,116行が 0（赤字->黒字が1件も無い）
#:     equity_ratio_pos_ratio               全行 1.0
#:     eps_growth_sym_q0 の max             99.939（+100 は前期<0<今期のときだけ）
#:     同 p5                                -100.000（黒字->赤字転落は5%以上ある）
#:   赤字企業を落とさないために _sym を用意したのに、別の入口で同じことが
#:   起きていた。赤字->黒字転換は株価が最も動くイベントなので損失が大きい。
#:
#: 実測（絞る前 17,580件 / 正例率 11.55%）:
#:     full4      10,116件 (57.5%) 正例率 11.64%
#:     full4_sym  12,484件 (71.0%) 正例率 11.82%   ← 採用
#:   +2,368件（+23.4%）戻るが正例率はほぼ動かないので、絞り方を変えても
#:   正例側に偏りは入らない。
#:
#: 赤字銘柄を候補から外したいなら、この列の欠測に頼るのではなく
#: FUND_QUALITY（増収増益フィルタ）で明示的に指定すること。
#:
#: いまは "none"（絞らない）にしてある。
#:   full4_sym は 17,580 -> 12,484 と母集団の29.0%を落としていた。
#:   落ちた5,096行は「決算の変化が作れない」だけで、
#:   株価・需給・地合いの特徴量は揃っている。捨てるには惜しい。
#:   欠測を学習させる懸念は fund_complete フラグで明示することに置き換えた
#:   （列ごとにばらばらに欠測を学ぶより、1本のフラグのほうが素直）。
#: 効果は all（fund_complete 込み）で測る。戻すなら "full4_sym" に書き換える。
FUND_REQUIREMENT = "none"

# --- 決算の中身で母集団を絞る --- #
# 決算を特徴量として薄く効かせるより、対象を選ぶ側に使う。
# 「増収増益が続いている銘柄の高値更新だけを見る」という形。
#
# 効いたかどうかは正例率で分かる。絞って正例率が上がるなら、
# モデルを作る前の段階で優位が取れている。上がらないなら、
# 高値更新という事実に決算の良さがすでに織り込まれているということ。
def _up(d, col, n):
    """col の直近 n 期がすべてプラスか。"""
    import functools
    import operator
    return functools.reduce(operator.and_,
                            (d[f"{col}_q{k}"] > 0 for k in range(n)))


FUND_QUALITY_SETS = {
    "none": None,
    # 直近1期が増収増益
    "up1": lambda d: _up(d, "sales_growth", 1) & _up(d, "eps_growth", 1),
    # 2期連続で増収増益
    "up2": lambda d: _up(d, "sales_growth", 2) & _up(d, "eps_growth", 2),
    # 3期連続で増収増益
    "up3": lambda d: _up(d, "sales_growth", 3) & _up(d, "eps_growth", 3),
    # 2期連続の増収増益 かつ EPS成長が加速している
    "up2_accel": lambda d: (_up(d, "sales_growth", 2) & _up(d, "eps_growth", 2)
                            & (d["eps_growth_chg1"] > 0)),
    # 増益かつ営業利益率も改善
    "up2_margin": lambda d: (_up(d, "sales_growth", 2) & _up(d, "eps_growth", 2)
                             & (d["op_margin_chg1"] > 0)),
}
#: 実際に適用する条件。FUND_QUALITY_SETS のキー。
FUND_QUALITY = "none"


@dataclass(frozen=True)
class RiseConfig:
    """母集団が breakout のときの目的変数。定義を変えて比較できる形にしてある。"""
    horizon: int = RISE_HORIZON
    threshold: float = RISE_THRESHOLD
    keep_days: int = KEEP_DAYS              # 0 なら条件なし
    end_ratio: Optional[float] = END_RATIO  # None なら条件なし
    end_window: int = END_WINDOW
    trend_short: int = TREND_SHORT
    trend_long: int = TREND_LONG
    require_uptrend: bool = REQUIRE_UPTREND
    #: 到達しきい値を k×σ で測る。None なら固定の threshold（VOL_NORM_K 参照）
    vol_norm_k: Optional[float] = VOL_NORM_K
    #: 上昇を測る基準の価格（LABEL_ENTRY 参照）。"next_open" = AdjO[t+1]、"close" = close[t]
    entry: str = LABEL_ENTRY

    @property
    def normalised(self) -> bool:
        return self.vol_norm_k is not None

    @property
    def name(self) -> str:
        m = round(self.horizon / 20)
        if self.normalised:
            base = f"{m}ヶ月内+{self.vol_norm_k:.1f}σ"
        else:
            base = f"{m}ヶ月内+{self.threshold*100:.0f}%"
        if self.keep_days:
            base += f" / 維持{self.keep_days}日"
        if self.end_ratio is not None:
            # ボラ正規化のときは終盤も同じ比率で伸縮する
            if self.normalised:
                base += f" / 終盤+{self.end_ratio / self.threshold:.2f}倍"
            else:
                base += f" / 終盤+{self.end_ratio*100:.0f}%"
        if self.require_uptrend:
            base += f" / MA{self.trend_short}>=MA{self.trend_long}"
        if self.entry == "next_open":
            base += " / 翌営業日寄り基準"
        return base


DEFAULT_RISE = RiseConfig()

#: 未来の値から作った列。特徴量に混ぜたらリークになる。
#: meta として持ち出すが、features に紛れ込んでいないか build() で必ず確認する。
#: rise_need / end_need は未来の値ではないが、ラベル定義そのものの列なので
#: 特徴量には入れない（vol_20d の単調変換で、特徴量としては冗長でもある）。
FUTURE_COLS = ["label", "future_max_close", "future_rise",
               "keep_days_cnt", "end_level", "uptrend_end",
               "rise_need", "end_need", "entry_price"]

# --- 除外条件 (docs/MODEL_DESIGN.md §2.2) --- #
MAX_RHIGH_AT_T = 95.0   # 基準日ですでに高値圏の銘柄は対象外

# 20日平均売買代金の下限（億円）。None なら絞らない。
#
# 0.1億円（1,000万円/日）。実測のはしご（report_liquidity_threshold）:
#
#   下限        残存      構成比   正例率
#   なし      27,237件   100.0%   10.87%
#   0.05億円  24,654件    90.5%   11.32%
#   0.1億円   22,909件    84.1%   11.51%   ← 採用
#   0.3億円   19,531件    71.7%   11.68%
#   0.5億円   17,580件    64.5%   11.55%   ← 変更前
#   1.0億円   14,418件    52.9%   11.34%
#   3.0億円    9,379件    34.4%   11.57%
#
# 帯ごとに逆算すると、1,000万円未満の2,583件は正例率が約6.6%、
# 0.05〜0.1億円の1,745件は約8.8%（はしごの丸めから導いた概算）。
# ここだけ全体の10.87%より明確に低い。
#
# 0.5億円だと母集団の35.5%を落としていた。0.1億円なら84.1%が残り、
# 変更前より+30%多い。正例率も 11.55% -> 11.51% とほぼ変わらない。
#
# 完全に外さない理由は運用側にある。売買代金が小さい銘柄は
# 実際には買えない（スプレッド・約定不能・スリッページ）ので、
# 統計が良くなっても利益に直結しない。
MIN_TRADING_VALUE = _sweep_override("MIN_TRADING_VALUE", 0.1, float)
#: 残存件数と正例率を出す閾値の候補（億円）。None は「絞らない」
LIQUIDITY_LADDER = (None, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0)


def _mkt_codes(raw: str) -> tuple:
    """「109,105」のような文字列を市場区分コードの組にする。空/none で無効。"""
    if raw.strip().lower() in ("", "none", "off", "-"):
        return ()
    return tuple(int(x) for x in raw.replace(" ", "").split(",") if x)


# 母集団から外す市場区分コード（時点別 mkt_code）。空タプルなら外さない。
#
# 109 =「その他」。J-Quants の市場区分でここに入るのは ETF / ETN / REIT /
# インフラファンドで、事業会社の株式は1件も入らない（実測: 546銘柄すべて
# 33業種が「その他」）。
#
# 外す根拠は research/exp/e17_etf.py の実測（3シード・ret_o1_40・
# しきい値はウォークフォワード）:
#
#   条件                            窓平均    SE   勝ち窓  最悪の窓  取引数
#   A  学習=全部 / 評価=全部（従来） +0.80pt  0.32   7/9   -0.73pt  2,000
#   A' 学習=全部 / 評価=株式        +1.17pt  0.26   9/9   +0.17pt  1,856
#   B  学習=株式 / 評価=株式        +1.28pt  0.33   9/9   +0.22pt  1,898
#
# A'-B = +0.11pt (z=0.27) で、学習から外すかどうかは差が無い。効くのは
# 「候補から外す」ほう（A-A' = +0.37pt, z=0.90）。z<2 なのでノイズ床は
# 越えていないが、最悪の窓が負から正に変わる点は一貫している。
#
# ETF が上位に溜まる理由は分散の小ささにある。母集団の 7.6% しか無いのに
# 上位10%の 31.0% を占め（4.1倍）、正例率は 32.0%（株式 19.9%）。
# ラベルは vol_20d で正規化した +1.2σ なので、日次ボラ 0.97%（株式 2.04%）の
# ETF は同じ σ に届く実際の値幅が半分で済む。ラベルは当たるが、
# 中央値 +2.79% / 平均 +2.38% と上値も薄い（株式は中央 +2.24% / 平均 +3.58%）。
# 値幅を取りにいく運用とは噛み合わないので母集団から外す。
EXCLUDE_MKT_CODES = _sweep_override("EXCLUDE_MKT_CODES", (109,), _mkt_codes)

# 時価総額の帯（億円）。フラグ列 cap_band の境界。
#
# 端は固定値にする。分位点で切ると全期間の分布を見ることになり、
# 基準日から見て未来の情報が混ざる。
# 区切りは EDA のラベル内訳（〜100 / 100〜300 / 300〜1000 / 1000〜3000 / 3000〜）
# と同じにして、集計とモデルで別の帯を使わないようにする。
CAP_BAND_EDGES = (100.0, 300.0, 1000.0, 3000.0)
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
    """
    年ごとの parquet を1つに繋ぐ。

    列の集合がファイル間でずれていると、concat は足りない列を黙って
    NaN で埋める。行数は揃っているのに値だけが無い、という形になり、
    行数を数えるだけでは気づけない。ずれていたら必ず出す。
    """
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not paths:
        raise SystemExit(f"{prefix} の parquet が {data_dir} にありません。先に jq_bulk.py を実行してください")
    frames, cols = [], {}
    for path in paths:
        f = pd.read_parquet(path)
        frames.append(f)
        cols[os.path.basename(path)] = set(f.columns)
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {prefix}: {len(df):,}行 ({len(paths)}ファイル)")

    for name, cs in cols.items():
        missing = sorted(set(df.columns) - cs)
        if missing:
            print(f"[warn] {name}: 他のファイルにあって、このファイルに無い列 "
                  f"{missing}（concat で NaN 埋めになる）")
    return df


# --------------------------------------------------------------------------- #
# 株価系の特徴量とラベル（銘柄ごとに時系列で算出）
# --------------------------------------------------------------------------- #

def load_market_segments(data_dir: str) -> Optional[pd.DataFrame]:
    """月末ごとの市場区分（master_hist の Date / Code / MktNm）。無ければ None。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if not paths:
        return None
    mh = pd.concat([pd.read_parquet(x, columns=["Date", "Code", "MktNm"]) for x in paths],
                   ignore_index=True)
    mh["Date"] = pd.to_datetime(mh["Date"])
    mh["Code"] = mh["Code"].astype(str)
    return mh.dropna(subset=["Date", "Code"])


def general_market_start(bars: pd.DataFrame, segments: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    銘柄ごとの「一般市場で数え始める日」と、上場日がデータの中で分かるか。

    返す列（1銘柄1行）
      Code
      first_row  最初の日足の行の日
      tpm        TOKYO PRO MARKET にいたことがある（segments の月末のどこかで）
      start      一般市場で数え始める日。TPM にいたことが無い銘柄は first_row（今までどおり）。
                 TPM から一般市場へ移った銘柄（最後に TPM だった月末より後の月末に、
                 TPM 以外の区分で載っている）は、**最後に TPM だった月末より後で、最初に
                 値が付いた日**。まだ TPM にいる銘柄と、TPM のまま消えた銘柄は NaT
                 （一般市場では数え始めていない。TPM の最後の数週に約定があっても移った
                 とはみなさない。実測で 311A・5073 がこの形）
      known      上場日（一般市場に出た日）がデータの中にある。TPM から移った銘柄と、
                 最初の行がデータの初日より後の銘柄（新規上場）。データの初日から
                 行がある銘柄は、それより前から上場していて上場日は分からない

    月末の区分で判定するので、移った月の途中の約定は「TPM だった月末」の後なら
    数える（実測では移った後の最初の値は移った日そのもの。5537 は 2025-12-15）。
    """
    d = pd.to_datetime(bars["Date"])
    high = bars["AdjH"].fillna(bars["H"]) if "AdjH" in bars.columns else bars["H"]
    x = pd.DataFrame({"Code": bars["Code"].astype(str).to_numpy(), "Date": d.to_numpy(),
                      "has": high.notna().to_numpy()})
    first_row = x.groupby("Code")["Date"].min()
    out = pd.DataFrame({"first_row": first_row, "tpm": False, "start": first_row})
    if segments is not None and len(segments):
        seg = segments.assign(Code=segments["Code"].astype(str))
        is_tpm = seg["MktNm"].astype(str) == TPM_NAME
        last_tpm = seg[is_tpm].groupby("Code")["Date"].max()
        codes = out.index.intersection(last_tpm.index)
        out.loc[codes, "tpm"] = True
        # 最後に TPM だった月末より後に、TPM 以外の区分で載った銘柄だけが「移った」
        later = seg[~is_tpm & seg["Code"].isin(codes)]
        later = later[later["Date"].to_numpy() > later["Code"].map(last_tpm).to_numpy()]
        moved = set(later["Code"])
        px = x[x["has"] & x["Code"].isin(moved)]
        px = px[px["Date"].to_numpy() > px["Code"].map(last_tpm).to_numpy()]
        after = px.groupby("Code")["Date"].min()
        out.loc[codes, "start"] = after.reindex(codes).to_numpy()
    data_start = x["Date"].min()
    out["known"] = ((out["tpm"] & out["start"].notna())
                    | (~out["tpm"] & (out["first_row"] > data_start)))
    out.index.name = "Code"
    return out.reset_index()


def write_general_market_start(gm: pd.DataFrame, path: str, enabled: bool) -> None:
    """
    画面のデータ取得（scripts/jquants_data_fetcher.py --general-market-start）が読む一覧。

    TPM にいたことがある銘柄だけを書く（それ以外は今までどおり最初の行から数える）。
    enabled は GENERAL_MARKET_START。False のあいだは画面も今までどおり（モデルと
    同じ切り替えに従う）。
    """
    t = gm[gm["tpm"]]
    payload = {
        "enabled": bool(enabled),
        "note": "TOKYO PRO MARKET にいたことがある銘柄の、一般市場で数え始める日"
                "（null はまだ TPM）。research/build_dataset.general_market_start",
        "start": {str(c): (None if pd.isna(v) else pd.Timestamp(v).date().isoformat())
                  for c, v in zip(t["Code"], t["start"])},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"[gm] 一般市場で数え始める日の一覧: TPM にいたことがある {len(t):,}銘柄"
          f"（うち移った {int(t['start'].notna().sum()):,}）/ enabled={bool(enabled)} -> {path}")


def listing_years(codes: pd.Series, dates: pd.Series, gm: pd.DataFrame,
                  cap: float = LISTING_CAP_YEARS) -> np.ndarray:
    """
    一般市場に上場（TPM から移行）してからの年数。cap 年で打ち止め（運用者の選択 ②）。

    上場日が分からない銘柄（データの初日より前から上場）は、データの初日から
    cap 年たつまでは欠測、その後は cap。まだ TPM にいる銘柄は欠測。
    上場日は上場したその日に分かることなので、先読みにはならない。
    """
    m = gm.set_index("Code")
    data_start = pd.to_datetime(m["first_row"]).min()
    dts = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    cs = pd.Series(codes).astype(str).reset_index(drop=True)
    listed = pd.to_datetime(cs.map(m["start"].where(m["known"])))
    yrs = (dts - listed).dt.days / 365.25
    since_start = (dts - data_start).dt.days / 365.25
    tpm_now = cs.map(m["tpm"] & m["start"].isna()).fillna(False).astype(bool)
    out = np.where(listed.notna(), np.minimum(yrs, cap),
                   np.where(since_start >= cap, cap, np.nan))
    out = np.where(tpm_now.to_numpy(), np.nan, out)
    # 上場前（まだ行が無い日）は作らない
    return np.where(np.asarray(yrs.fillna(0.0)) < 0, np.nan, out).astype(float)


def add_vol_factors(df: pd.DataFrame, ret1: pd.Series) -> pd.DataFrame:
    """
    vol_20d を「どんな種類のボラか」に分解した列（実験51、2026-09-26 運用者の依頼）。

    vol_20d は特徴量であると同時にラベルの到達しきい値（1.2σ×√20）の σ でもあるので、
    モデルは「ボラが高い＝正例になりにくい」をほぼしきい値経由で学ぶ（実験50）。
    水準そのものは vol_20d に任せ、ここでは比だけを持つ。
      vol_rel_long   vol_20d ÷ vol_120d。いま異常に荒いのか、もともと荒い銘柄なのか
      vol_rel_short  vol_5d ÷ vol_20d。ブレイク直前に膨らんだか、落ち着いてきたか
      vol_updown     上昇日の半分散の平方根 ÷ 下落日の同じもの（20日）。上に跳ねて荒いのか、
                     投げられて荒いのか。片方に日が無ければ欠測
      vol_gap_ratio  寄り付きギャップの σ ÷ 日中レンジの σ（20日）。材料で飛んだのか、
                     日中の売買で動いたのか。ギャップ = 始値 ÷ 前日終値 − 1、
                     日中レンジ = log(高値 ÷ 安値)
    市場・業種との比（vol_rel_mkt / vol_rel_sector）は指数を結合した後で作る
    （attach_vol_vs_market）。すべて当日までの値だけで作る（先読みなし）。
    """
    codes = df["Code"]
    gr = ret1.groupby(codes, sort=False)
    vol_120 = gr.transform(lambda s: s.rolling(120, min_periods=90).std()) * 100.0
    vol_5 = gr.transform(lambda s: s.rolling(5, min_periods=4).std()) * 100.0
    df["vol_rel_long"] = _safe_ratio(df["vol_20d"], vol_120)
    df["vol_rel_short"] = _safe_ratio(vol_5, df["vol_20d"])

    up = ret1.clip(lower=0.0) ** 2
    dn = ret1.clip(upper=0.0) ** 2
    up_sd = np.sqrt(up.groupby(codes, sort=False).transform(
        lambda s: s.rolling(20, min_periods=15).mean()))
    dn_sd = np.sqrt(dn.groupby(codes, sort=False).transform(
        lambda s: s.rolling(20, min_periods=15).mean()))
    df["vol_updown"] = _safe_ratio(up_sd, dn_sd)

    prev_close = df.groupby("Code", sort=False)["close"].shift(1)
    if "open" not in df.columns:
        # 始値の無い足（テストの合成データなど）ではギャップは作れない。欠測にする
        df["vol_gap_ratio"] = np.nan
        return df
    gap = df["open"] / prev_close - 1.0
    hl = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.log(np.where((hl[0] > 0) & (hl[1] > 0), hl[0] / hl[1], np.nan))
    rng = pd.Series(rng, index=df.index)
    gap_sd = gap.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=15).std())
    rng_sd = rng.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=15).std())
    df["vol_gap_ratio"] = _safe_ratio(gap_sd, rng_sd)
    return df


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a ÷ b。分母が 0・欠測なら欠測（0 で割った無限大を残さない）。"""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b.where(b > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def attach_vol_vs_market(samples: pd.DataFrame) -> pd.DataFrame:
    """
    vol_20d を市場・業種のボラで割った比（実験51）。市場環境と業種指数を結合した後に呼ぶ。
      vol_rel_mkt     vol_20d ÷ topix_vol_20
      vol_rel_sector  vol_20d ÷ sector_vol_20（業種指数の20日ボラ。無ければ欠測）
    """
    samples["vol_rel_mkt"] = _safe_ratio(samples["vol_20d"], samples.get("topix_vol_20"))
    sec = samples["sector_vol_20"] if "sector_vol_20" in samples.columns else np.nan
    samples["vol_rel_sector"] = _safe_ratio(samples["vol_20d"], pd.Series(sec, index=samples.index))
    for c in ("vol_rel_mkt", "vol_rel_sector"):
        print(f"[merge] {c}: 欠測 {samples[c].isna().mean()*100:.1f}%")
    return samples


def price_panel(bars: pd.DataFrame, cfg: LabelConfig = DEFAULT_LABEL,
                start: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    銘柄ごとに時系列指標を算出する。

    調整後（Adj*）を優先して使う。株式分割をまたぐと素の価格では
    52週高値が不連続になり、偽のブレイクを大量に生むため。

    start（銘柄コード -> 数え始める日。NaT はまだ数え始めない）を渡すと、78週の窓の
    「HIGH_WINDOW 本の履歴」をその日からの行だけで数える（GENERAL_MARKET_START）。
    start に無い銘柄は今までどおり最初の行から数える。
    """
    df = bars.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

    close = df["AdjC"].fillna(df["C"])
    # 始値は目的変数の基準（翌営業日の寄り、LABEL_ENTRY）にだけ使う。特徴量には使わない
    if "O" in df.columns:
        df["open"] = (df["AdjO"].fillna(df["O"]) if "AdjO" in df.columns else df["O"])
    high = df["AdjH"].fillna(df["H"])
    low = df["AdjL"].fillna(df["L"]) if "AdjL" in df.columns else df["L"]
    vol = df["AdjVo"].fillna(df["Vo"])
    df["close"] = close
    df["high"] = high
    df["low"] = low
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

    # --- ボラティリティ（20営業日の日次リターン標準偏差、%） --- #
    # もとは add_breakout_context で作っていたが、ラベルより後に走るので
    # 「到達しきい値を銘柄自身のσで測る」定義に使えなかった。
    # 定義を2箇所に持つと必ずずれるので、ここ1箇所で作って両方が使う。
    _ret1 = df["close"] / g["close"].shift(1) - 1.0
    df["vol_20d"] = _ret1.groupby(df["Code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=15).std()) * 100.0
    df = add_vol_factors(df, _ret1)

    # --- 52週高値（当日を含む / 含まない の2種類が要る） --- #
    # 含む  : 基準日時点の高値接近率 R_high の分母
    # 含まない: ブレイク判定（「それまでの高値」を上抜けたか）の基準
    # min_periods は「窓の中の非欠測の数」で判定される。
    # w を指定すると、窓の中に欠測が1つでもあれば結果が欠測になる。
    # 高値が欠測の日（売買が成立していない日）は実測で全行の約3%あり、
    # 1つの欠測が後続 w 行（78週窓なら約1年半）の判定を丸ごと潰していた。
    # 実際 2021年は高値更新日が0件になり、原因はこれだった。
    #
    # 意図は「w 営業日ぶんの履歴があること」であって
    # 「窓の中に欠測が1つも無いこと」ではない。2つに分けて表す。
    #   min_periods=1  … 手元にある高値の最大を取る
    #   age >= w       … w 行ぶんの履歴がたまっているか
    #   coverage       … 窓の中で実際に値がある割合（休止銘柄を弾く）
    w = cfg.high_window
    age = g.cumcount()
    if start is not None and len(start):
        # 一般市場で数え始める前の行（TOKYO PRO MARKET の時期）は履歴に数えない
        listed = df["Code"].astype(str).isin(start.index)
        s0 = pd.to_datetime(df["Code"].astype(str).map(start))
        began = ~listed | (s0.notna() & (df["Date"] >= s0))
        age = began.astype(int).groupby(df["Code"], sort=False).cumsum() - 1
        age = age.where(began, -1)
    hi = g["high"].transform(lambda s: s.rolling(w, min_periods=1).max())
    cov = g["high"].transform(lambda s: s.rolling(w, min_periods=1).count())
    enough = (age >= w - 1) & (cov >= w * MIN_WINDOW_COVERAGE)
    df["high52w"] = hi.where(enough)
    df["high52w_prior"] = hi.shift(1).where(enough.shift(1, fill_value=False)
                                            & (age >= w))
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


def mark_new_highs(df: pd.DataFrame, cooldown: int = BREAKOUT_COOLDOWN,
                   on_high: bool = BREAKOUT_ON_HIGH) -> pd.DataFrame:
    """
    52週高値を更新した日に印を付ける。

    判定は「それまでの52週高値」との比較（high52w_prior は当日を含まない）。
    当日を含む high52w と比べると、自分自身と比べることになって常に成立する。

    連続した更新日は1件にまとめる。
    高値更新が10日続くと、ほぼ同じ特徴量・重なるラベルのサンプルが10件でき、
    件数が水増しされるうえサンプル間が独立でなくなるため。
    直前 cooldown 営業日に更新が無い日だけを「新規のブレイク」とみなす。
    """
    px = df["high"] if on_high else df["close"]
    is_new_high = (px > df["high52w_prior"]) & df["high52w_prior"].notna()
    df["is_new_high"] = is_new_high

    g = df.groupby("Code", sort=False)
    # 直前 cooldown 営業日（当日を除く）に更新があったか
    recent = g["is_new_high"].transform(
        lambda s: s.shift(1).rolling(cooldown, min_periods=1).max())
    df["is_fresh_break"] = is_new_high & (recent.fillna(0) == 0)
    return df


def _days_above(close: np.ndarray, horizon: int, level) -> np.ndarray:
    """
    各 t について、t+1 〜 t+horizon の終値が close[t]*level 以上だった日数。

    しきい値が行ごと（close[t] 倍）に動くので、固定値の rolling では書けない。
    銘柄1本ぶんの窓行列を作って一気に数える。

    level はスカラーでも行ごとの配列でもよい。配列にするのは
    「到達しきい値を銘柄自身のσで測る」定義のため（VOL_NORM_K 参照）。
    level が欠測の行は 0 ではなく NaN にする。0 にすると
    「一度も届かなかった」と「そもそも判定できない」が混ざる。
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n <= horizon:
        return out
    win = np.lib.stride_tricks.sliding_window_view(close, horizon)  # win[k] = close[k:k+horizon]
    fwd = win[1:]              # t 行目 = close[t+1 : t+1+horizon]
    m = fwd.shape[0]           # = n - horizon
    lv = np.asarray(level, dtype=float)
    thr = close[:m] * (lv if lv.ndim == 0 else lv[:m])
    cnt = (fwd >= thr[:, None]).sum(axis=1).astype(float)
    out[:m] = np.where(np.isnan(thr), np.nan, cnt)
    return out


def _by_code(df: pd.DataFrame, values: np.ndarray, fn,
             level: Optional[np.ndarray] = None) -> np.ndarray:
    """
    銘柄ごとの連続区間に fn を適用する。df は Code,Date でソート済みが前提。

    level を渡すと同じ区間で切って第2引数に渡す。行ごとに動くしきい値を
    銘柄の区間に正しく対応させるため（丸ごと渡すと位置がずれる）。
    """
    out = np.full(len(df), np.nan)
    codes = df["Code"].to_numpy()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for a, b in zip(starts, ends):
        out[a:b] = fn(values[a:b]) if level is None else fn(values[a:b], level[a:b])
    return out


#: /equities/valuation から使う列。運用者の判断（2026-09-22、実験38）で
#: **PER / PBR / ROE は API 側に寄せる**。両方は持たない。
#:
#: 根拠（実験38 の実測、突き合わせ 21,856行）
#:   内部整合   PER × EPS = 終値 の誤差中央値 0.02%（1%以内 100%）
#:              PBR × BPS = 終値 の誤差中央値 0.15%（1%以内  98%）
#:   充足率     PER 82.0% -> 91.1%（まともな増分 +9.1pt）
#:              ROE 92.6% -> 99.1%（まともな増分 +6.5pt）
#:              PBR は 99.6% で横ばい
#: ROE は「精度の差」ではなく**定義の差**（API は EPS/BPS の1株ベース、
#: 自前は TTM純利益/自己資本）。運用者が API を選んだ。
#:
#: API の ROE は**小数**（0.0791 = 7.91%）なので100倍して % に直す。
API_VALUATION_SCALE = {"PER": 1.0, "PBR": 1.0, "ROE": 100.0}

#: どの指標を API に寄せるか。実験で出どころを A/B するために外から
#: 差し替えられるようにしてある（本番はすべて True）。
#: 実験40 の実測で、API の ROE に替えると ROE 一族の検出力が落ちた
#: （ROE_accel の下位z が -2.51 -> +0.19 と符号ごと消えるなど）。
#: 充足率は 92.6% -> 99.7% に上がるので、どちらを取るかはモデルでの
#: 評価（docs/MODEL_ADOPTION_RULES.md §7）で決める。
USE_API_VALUATION = {"per": True, "pbr": True, "roe": True}


def api_valuation(keys: pd.DataFrame, date_col: str = "Date",
                  data_dir: str = DATA_DIR,
                  asof: bool = False) -> pd.DataFrame:
    """
    (Code, date_col) に /equities/valuation の PER / PBR / ROE を合わせる。

    asof=True なら merge_asof の backward で「その日までに出ている直近」を
    取る（開示日が非営業日のことがあるため）。False なら同日の完全一致。

    先読みにはならない。valuation はその日の終値から作られる当日の値で、
    決算パネル側は DiscDate <= サンプル日 でしか結合されない。

    戻り値の列は api_per / api_pbr / api_roe。無ければ全 NaN。
    """
    out = pd.DataFrame(index=keys.index,
                       columns=["api_per", "api_pbr", "api_roe"], dtype=float)
    paths = sorted(glob.glob(os.path.join(data_dir, "valuation_[0-9]*.parquet")))
    if not paths:
        return out
    want = ["Code", "Date"] + list(API_VALUATION_SCALE)
    v = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    v = v[[c for c in want if c in v.columns]].copy()
    if not {"Code", "Date"} <= set(v.columns):
        return out
    v["Code"] = v["Code"].astype(str)
    v["Date"] = pd.to_datetime(v["Date"], errors="coerce")
    v = v.dropna(subset=["Code", "Date"])
    for c, k in API_VALUATION_SCALE.items():
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce") * k
    v = v.rename(columns={c: f"api_{c.lower()}" for c in API_VALUATION_SCALE})

    left = keys[["Code", date_col]].copy()
    left["Code"] = left["Code"].astype(str)
    left[date_col] = pd.to_datetime(left[date_col], errors="coerce")
    left["_i"] = np.arange(len(left))
    if asof:
        m = pd.merge_asof(left.sort_values(date_col), v.sort_values("Date"),
                          left_on=date_col, right_on="Date", by="Code",
                          direction="backward", allow_exact_matches=True,
                          tolerance=pd.Timedelta(days=10))
        m = m.sort_values("_i")
    else:
        m = left.merge(v.rename(columns={"Date": date_col}),
                       on=["Code", date_col], how="left")
    for c in out.columns:
        if c in m.columns:
            out[c] = pd.to_numeric(m[c], errors="coerce").to_numpy()
    return out


def rise_thresholds(vol_20d, cfg: RiseConfig = DEFAULT_RISE):
    """
    cfg が課す (到達しきい値, 終盤の必要水準) を行ごとに返す。

    固定%なら全行同じ値、ボラ正規化なら銘柄自身の期間σの k 倍。
    ラベルを作る側（attach_rise_label）と、データセットから引き直す側
    （export_label_samples.verdict）が同じ式を使うための1箇所。
    2箇所に書くと必ずずれる。実際、ボラ正規化に変えたとき引き直し側が
    固定値のままで 2,364件食い違った（テストと assert が捕まえた）。

    vol_20d は %（日次リターン標準偏差×100）で渡すこと。
    """
    vol = pd.to_numeric(pd.Series(vol_20d), errors="coerce")
    if cfg.normalised:
        need = cfg.vol_norm_k * vol / 100.0 * np.sqrt(cfg.horizon)
    else:
        need = pd.Series(float(cfg.threshold), index=vol.index)
    if cfg.end_ratio is None:
        end_need = pd.Series(np.nan, index=vol.index)
    elif cfg.normalised:
        end_need = (cfg.end_ratio / cfg.threshold) * need
    else:
        end_need = pd.Series(float(cfg.end_ratio), index=vol.index)
    return need, end_need


def attach_rise_label(df: pd.DataFrame, cfg: RiseConfig = DEFAULT_RISE) -> pd.DataFrame:
    """
    「上がったか」だけでなく「続いたか」も条件にする。

    到達（従来）:
        基準の価格から、先 horizon 営業日以内に threshold 以上上昇したか。
        基準は cfg.entry（既定は翌営業日の寄り AdjO[t+1]。"close" なら更新日の終値）。
        上昇は終値で測る。高値ベースだと「一瞬触れただけ」も正例になる。

    継続（追加）:
        到達しても、その後すぐ下落トレンドに入るならモメンタムとは言えない。
        次の3つで「続いたこと」を要求する。どれも0/Noneで無効化できる。
          keep_days … +threshold 以上で引けた日が通算 keep_days 日以上
          end_ratio … 終盤 end_window 日平均が基準の価格の +end_ratio 以上
          uptrend   … t+horizon 時点で MA(trend_short) >= MA(trend_long)

    先の営業日が足りない末尾は NaN（判定不能）にする。False にはしない。
    ここを False にすると「起きなかった」と「まだ分からない」が混ざる。
    """
    g = df.groupby("Code", sort=False)
    h = cfg.horizon

    # 高値窓と同じ理由で min_periods は 1 にする。
    # 先の窓に売買不成立の日が1つあるだけでラベルが未確定になっていた。
    # 「先 h 営業日ぶんの行があるか」は末尾からの位置で別に判定する。
    def future_max(s: pd.Series) -> pd.Series:
        # t+1 〜 t+horizon の終値の最大
        return s[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)

    have_forward = g.cumcount(ascending=False) >= h
    df["future_max_close"] = g["close"].transform(future_max).where(have_forward)

    # --- 基準の価格（LABEL_ENTRY）--- #
    if cfg.entry not in LABEL_ENTRIES:
        raise SystemExit(f"RiseConfig.entry は {LABEL_ENTRIES} のどれか: {cfg.entry!r}")
    if cfg.entry == "next_open":
        if "open" not in df.columns:
            raise SystemExit("open がありません。翌営業日の寄りを基準にするには "
                             "price_panel の始値が要ります")
        # 翌営業日の行の始値。収益の計算（lab.realized_returns の entry =
        # AdjO.shift(-1)）と同じ取り方で、寄りが付かなかった日は欠測のまま
        entry = g["open"].shift(-1)
    else:
        entry = df["close"]
    df["entry_price"] = entry
    ratio = df["future_max_close"] / entry - 1.0
    df["future_rise"] = ratio

    # --- 到達に必要な上昇率 --- #
    # 固定なら全行同じ。ボラ正規化なら銘柄自身の期間σの k 倍。
    # 実際に課したしきい値を列として残す。残さないと、後段
    # （ファネル・ラベル比較・チャート）が cfg.threshold を見てしまい、
    # 行ごとに違うしきい値を1つの数字で語ることになる。
    if cfg.normalised and "vol_20d" not in df.columns:
        raise SystemExit(
            "vol_20d がありません。ボラ正規化ラベルには price_panel が必要です")
    need, end_need = rise_thresholds(
        df["vol_20d"] if cfg.normalised else pd.Series(np.nan, index=df.index), cfg)
    need.index = df.index
    end_need.index = df.index
    df["rise_need"] = need
    hit = ratio >= need

    close = df["close"].to_numpy(dtype=float)

    # --- 維持日数 --- #
    # _days_above は close[t] に level を掛けて比べるので、基準の価格に換算して渡す
    if cfg.keep_days:
        df["keep_days_cnt"] = _by_code(
            df, close, lambda a, lv: _days_above(a, h, lv),
            level=((1.0 + need) * (entry / df["close"])).to_numpy(dtype=float))
    else:
        df["keep_days_cnt"] = np.nan

    # --- 終盤の水準 --- #
    # t+horizon 時点での「直近 end_window 日平均終値」。
    # 1日だけの値だと、たまたまその日が押し目でも失格になってしまう。
    ma_end = g["close"].transform(
        lambda s: s.rolling(cfg.end_window, min_periods=1).mean())
    end_close = ma_end.groupby(df["Code"], sort=False).shift(-h)
    df["end_level"] = end_close / entry - 1.0

    # --- トレンド --- #
    ma_s = g["close"].transform(
        lambda s: s.rolling(cfg.trend_short, min_periods=1).mean())
    ma_l = g["close"].transform(
        lambda s: s.rolling(cfg.trend_long, min_periods=1).mean())
    # 移動平均は履歴がたまってから使う（min_periods を外したぶんここで担保する）
    _age = g.cumcount()
    ma_s = ma_s.where(_age >= cfg.trend_short - 1)
    ma_l = ma_l.where(_age >= cfg.trend_long - 1)
    up = (ma_s >= ma_l).where(ma_s.notna() & ma_l.notna())
    df["uptrend_end"] = up.groupby(df["Code"], sort=False).shift(-h)

    # --- 終盤に必要な水準 --- #
    # 固定なら end_ratio そのまま。ボラ正規化なら到達しきい値と同じ比率で伸縮
    # （+20%に対する+10% = 半分、という関係を σ 版でも保つ）。rise_thresholds 参照
    df["end_need"] = end_need

    # --- 合成 --- #
    ok = hit.copy()
    if cfg.keep_days:
        ok &= df["keep_days_cnt"] >= cfg.keep_days
    if cfg.end_ratio is not None:
        ok &= df["end_level"] >= df["end_need"]
    if cfg.require_uptrend:
        ok &= df["uptrend_end"] == 1.0

    # 判定に使う将来値が1つでも欠けていれば未確定。
    # 課していない条件の入力までは要求しない
    # （終盤条件を切っているのに終盤の値が無いから未確定、では筋が通らない）。
    determined = df["future_max_close"].notna()
    if cfg.entry == "next_open":
        # 翌営業日に寄りが付かなければ買えない。収益の計算と同じく判定しない
        determined &= entry.notna()
    if cfg.normalised:
        # しきい値そのものが作れない行（上場直後などで vol_20d が欠測）は
        # 「起きなかった」ではなく「判定できない」
        determined &= need.notna()
    if cfg.end_ratio is not None:
        determined &= end_close.notna()
    if cfg.require_uptrend:
        determined &= df["uptrend_end"].notna()
    df["label"] = ok.where(determined)
    return df


def report_fund_completeness(samples: pd.DataFrame) -> None:
    """
    決算の変化がどれだけ作れるかを、要求の強さごとに数える。

    絞ってから測るのでは「絞った結果どうなるか」が分からない。
    絞る前に、候補ごとの残存数と正例率を並べて出す。
    """
    n = len(samples)
    lab = samples["label"]
    # 赤字->黒字転換が何件残るかも一緒に出す。
    # full4 は eps_growth（前年同期が0以下だと欠測）を要求していたため、
    # 転換した会社を1件残らず落としていた。件数を毎回出しておけば、
    # 同じことが起きたときに残存件数と正例率だけでは見えない差に気づける。
    turn = pd.to_numeric(samples.get("eps_growth_turn"), errors="coerce")
    print(f"[fund] 決算の完全性で絞った場合の残存（絞る前 {n:,}件 / "
          f"正例率 {lab.mean()*100:.2f}% / 黒字転換 {int((turn == 1).sum()):,}件）")
    for name, cols in FUND_REQUIREMENT_SETS.items():
        have = [c for c in cols if c in samples.columns]
        if len(have) != len(cols):
            print(f"  {name:<9} 列が足りない: {sorted(set(cols) - set(have))}")
            continue
        m = samples[have].notna().all(axis=1) if have else pd.Series(True, index=samples.index)
        k = int(m.sum())
        rate = lab[m].mean() * 100 if k else float("nan")
        n_turn = int((turn[m] == 1).sum()) if turn is not None else 0
        mark = " ← 採用" if name == FUND_REQUIREMENT else ""
        print(f"  {name:<9} {k:>7,}件 ({k/n*100:5.1f}%) 正例率 {rate:5.2f}% "
              f"/ 黒字転換 {n_turn:>5,}件 / 要求{len(cols)}列{mark}")


def report_fund_quality(samples: pd.DataFrame) -> None:
    """
    決算の中身で絞ったときに、正例率がどう動くかを候補ごとに出す。

    ここで正例率が上がるなら、モデル以前の段階で優位が取れている。
    上がらないなら、高値更新という事実に決算の良さが織り込まれている。
    絞る前に全候補を数える。
    """
    n = len(samples)
    base = samples["label"].mean() * 100
    print(f"[quality] 決算の中身で絞った場合（絞る前 {n:,}件 / 正例率 {base:.2f}%）")
    for name, fn in FUND_QUALITY_SETS.items():
        if fn is None:
            continue
        try:
            m = fn(samples).fillna(False)
        except KeyError as exc:
            print(f"  {name:<11} 列が無い: {exc}")
            continue
        k = int(m.sum())
        if k == 0:
            print(f"  {name:<11} 該当なし")
            continue
        rate = samples.loc[m, "label"].mean() * 100
        mark = " ← 採用" if name == FUND_QUALITY else ""
        print(f"  {name:<11} {k:>7,}件 ({k/n*100:5.1f}%) 正例率 {rate:5.2f}% "
              f"（{rate - base:+.2f}pt / {rate/base:.2f}倍）{mark}")


def apply_fund_quality(samples: pd.DataFrame) -> pd.DataFrame:
    """FUND_QUALITY に従って、決算の中身で母集団を絞る。"""
    fn = FUND_QUALITY_SETS[FUND_QUALITY]
    if fn is None:
        return samples
    before = len(samples)
    out = samples[fn(samples).fillna(False)].copy()
    print(f"[filter] 決算の中身で絞る（{FUND_QUALITY}）: {before:,} -> {len(out):,} "
          f"/ 正例率 {out['label'].mean()*100:.2f}%")
    return out


def report_liquidity_threshold(samples: pd.DataFrame) -> None:
    """
    流動性の下限を変えたときの残存件数と正例率を出す。

    絞ってから測るのでは「絞った結果どうなるか」しか分からない。
    絞る前に、候補ごとの残存数と正例率を並べて出す
    （決算フィルタの report_fund_completeness と同じやり方）。
    """
    n = len(samples)
    if n == 0 or "tv_ma20" not in samples.columns:
        return
    base = float(samples["label"].mean() * 100)
    print(f"[liq] 流動性の下限ごとの残存（絞る前 {n:,}件 / 正例率 {base:.2f}%）")
    for thr in LIQUIDITY_LADDER:
        keep = samples if thr is None else samples[samples["tv_ma20"] >= thr]
        name = "なし" if thr is None else f"{thr}億円"
        rate = float(keep["label"].mean() * 100) if len(keep) else float("nan")
        print(f"  {name:>8}  {len(keep):>7,}件 ({len(keep) / n * 100:5.1f}%) "
              f"正例率 {rate:5.2f}%")


def fund_complete_flag(samples: pd.DataFrame) -> pd.Series:
    """
    決算の変化（full4_sym の10列）がすべて作れる行かどうか。

    FUND_REQUIREMENT で絞らない場合、決算が欠測の行が混ざる。
    LightGBM は欠測をそのまま扱えるが、「欠測かどうか」は
    列ごとにばらばらに学習される。1本のフラグにしておけば、
    どちらの母集団の行なのかをモデルが直接見られる。

    絞る／絞らないに関わらず作る。絞った場合は全行 1 になる。
    """
    cols = FUND_REQUIREMENT_SETS["full4_sym"]
    missing = sorted(set(cols) - set(samples.columns))
    if missing:
        raise SystemExit(f"[fatal] fund_complete の材料がありません: {missing}")
    return samples[cols].notna().all(axis=1).astype(float)


def cap_band(market_cap: pd.Series) -> pd.Series:
    """
    時価総額の帯（0=最小 … 4=最大）。欠測は欠測のまま返す。

    log_market_cap（連続値）が既にあるので、木にとって情報は増えない。
    帯は log_market_cap の単調な階段関数であり、帯での分割は
    log_market_cap での分割で必ず再現できる。

    それでも持たせるのは2点のため:
      ・CV の層別（tuning._cap_bands）と EDA の内訳（eda_stats）が
        同じ帯を使うようになる。3か所で違う切り方をしていると、
        同じ「規模」という言葉が別のものを指す
      ・浅い木でも他の特徴量との交互作用を作りやすい

    層別評価（stratified_eval）は日付ごとの分位で切る。あちらは
    「同じ日の中で規模を揃えて比べる」ための層で、目的が違う。
    """
    filled = pd.to_numeric(market_cap, errors="coerce")
    band = pd.Series(np.digitize(filled.fillna(-1.0), CAP_BAND_EDGES),
                     index=filled.index, dtype=float)
    return band.where(filled.notna())


def apply_fund_requirement(samples: pd.DataFrame) -> pd.DataFrame:
    """FUND_REQUIREMENT に従って、変化が作れない行を落とす。"""
    cols = FUND_REQUIREMENT_SETS[FUND_REQUIREMENT]
    if not cols:
        return samples
    have = [c for c in cols if c in samples.columns]
    missing = sorted(set(cols) - set(have))
    if missing:
        raise SystemExit(f"[fatal] 要求列がデータにありません: {missing}")
    before = len(samples)
    out = samples[samples[have].notna().all(axis=1)].copy()
    print(f"[filter] 決算の変化が作れない行を除外（{FUND_REQUIREMENT}）: "
          f"{before:,} -> {len(out):,}")
    return out


def report_rise_funnel(samples: pd.DataFrame,
                       cfg: RiseConfig = DEFAULT_RISE) -> None:
    """継続条件をどれだけ課したか、条件ごとに何件落ちたかを出す。

    「正例が減った」で終わらせず、どの条件がどれだけ効いたかを見えるようにする。
    条件は順に重ねるので、各行の残数は「そこまでの全条件を満たした件数」。
    """
    d = samples[samples["label"].notna()]
    if d.empty:
        return
    n = len(d)
    # しきい値は行ごとに違いうるので、cfg の数字ではなく実際に課した列で判定する
    need = (d["rise_need"] if "rise_need" in d.columns
            else pd.Series(float(cfg.threshold), index=d.index))
    lab = (f"{cfg.vol_norm_k:.1f}σ" if cfg.normalised
           else f"{cfg.threshold*100:.0f}%")
    steps = [(f"到達（{cfg.horizon}営業日以内に +{lab}）", d["future_rise"] >= need)]
    if cfg.keep_days:
        steps.append((f"維持（+{lab}以上で引けた日 >= {cfg.keep_days}日）",
                      d["keep_days_cnt"] >= cfg.keep_days))
    if cfg.end_ratio is not None:
        end_need = (d["end_need"] if "end_need" in d.columns
                    else pd.Series(float(cfg.end_ratio), index=d.index))
        ratio_lab = (f"{cfg.end_ratio / cfg.threshold:.2f}倍" if cfg.normalised
                     else f"+{cfg.end_ratio*100:.0f}%")
        steps.append((f"終盤（{cfg.horizon}営業日後の{cfg.end_window}日平均 >= "
                      f"{ratio_lab}）",
                      d["end_level"] >= end_need))
    if cfg.require_uptrend:
        steps.append((f"トレンド（MA{cfg.trend_short} >= MA{cfg.trend_long}）",
                      d["uptrend_end"] == 1.0))

    print(f"[label] ラベル確定 {n:,}件について、条件を重ねたときの正例数")
    mask = pd.Series(True, index=d.index)
    prev = n
    for name, cond in steps:
        mask = mask & cond.fillna(False)
        k = int(mask.sum())
        print(f"  {name:<46} {k:>7,}件 ({k/n*100:5.2f}%)  -{prev-k:,}")
        prev = k


def add_breakout_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    ブレイクそのものの性質を表す特徴量。

    母集団を高値更新日にすると、どの銘柄も「高値からの距離」は同じになる。
    残る違いは「どういう抜け方をしたか」なので、それを列にする。
    ここは推測なので、効くかどうかは日付内診断と層別評価で測る。
    """
    g = df.groupby("Code", sort=False)

    # ベースの長さ: 前回の52週高値更新から何営業日空いたか。
    # 長く保ち合ってからの初回ブレイクほど強い、という仮説を測れるようにする
    idx = pd.Series(np.arange(len(df)), index=df.index)
    last_nh = idx.where(df["is_new_high"].fillna(False)).groupby(
        df["Code"], sort=False).ffill().shift(1)
    df["base_length"] = (idx - last_nh).where(df["is_new_high"].fillna(False))

    # 抜けの大きさ: それまでの高値をどれだけ上回ったか（%）
    df["break_margin"] = (df["close"] / df["high52w_prior"] - 1.0) * 100.0

    # 当日の値幅の中で終値がどこにあるか。
    # 高値引けなら強い、上ヒゲなら弱い
    rng = df["high"] - df["low"]
    df["close_position"] = np.where(rng > 0, (df["close"] - df["low"]) / rng * 100.0,
                                    np.nan)

    # 直近20営業日の上昇率。すでに走った後か、静かなところからの初動か
    df["ret_20d"] = (df["close"] / g["close"].shift(20) - 1.0) * 100.0
    # vol_20d は price_panel で作る（ラベルのしきい値に使うので、
    # ラベル計算より前に存在している必要がある）。ここでは作り直さない。
    if "vol_20d" not in df.columns:
        raise SystemExit("vol_20d がありません。price_panel を先に通してください")
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


def encode_category(s: pd.Series, name: str = "") -> pd.Series:
    """
    マスタのカテゴリ列を数値コードにする。

    数値で来る列（S33/S17/Mkt は数字コード）はそのまま数値化する。
    数値でない列（ScaleCat は "TOPIX Small 2" のような文字列）を
    pd.to_numeric に通すと**全件 NaN の空列**になる。
    実際それで scalecat_code が100%欠測だった。列は届いていたのに、
    数値化に失敗していることを誰も見ていなかった。

    文字列だった場合は、値を並べて安定した整数に割り当てる。
    並びは辞書順で固定する（実行ごとに変わるとモデルが再現しなくなる）。
    どの値が来ているかは必ず出す。順序に意味を持たせたければ、
    実際の値を見てから明示的に対応表を書く。
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().any():
        return v
    vals = sorted(x for x in s.dropna().unique())
    if not vals:
        print(f"[warn] {name}: 値が1つも無い。欠測のままにする")
        return pd.Series(np.nan, index=s.index)
    print(f"[merge] {name} は数値でないため符号化する: "
          + ", ".join(f"{i}={x}" for i, x in enumerate(vals)))
    code = {x: i for i, x in enumerate(vals)}
    return s.map(code).astype("float64")


def drop_excluded_markets(samples: pd.DataFrame,
                          codes: tuple = None) -> pd.DataFrame:
    """
    市場区分で母集団から外す。既定は ETF・REIT 等（EXCLUDE_MKT_CODES）。

    mkt_code は master_hist を結合してからでないと分からないので、
    ほかの除外条件とは別の場所から呼ぶことになる。
    理由と実測は EXCLUDE_MKT_CODES のコメントに書いた。
    """
    codes = EXCLUDE_MKT_CODES if codes is None else tuple(codes)
    if not codes:
        print("[filter] 市場区分による除外はしない（EXCLUDE_MKT_CODES が空）")
        return samples
    before = len(samples)
    # isin は欠測を False にするので、市場区分が付かなかった行は残る。
    # master_hist が無い環境で母集団ごと消えるのを避けるため。
    out = samples[~samples["mkt_code"].isin(codes)]
    n_unknown = int(out["mkt_code"].isna().sum())
    names = ", ".join(str(c) for c in codes)
    print(f"[filter] ETF・REIT等(mkt_code in {{{names}}})を除外: "
          f"{before:,} -> {len(out):,}")
    if n_unknown:
        print(f"[filter] 市場区分が付かなかった {n_unknown:,}件は残した")
    return out


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


#: 開示からの日数の上限。これより古い開示は「止まっている」として同じ値にする
TIMING_CLIP_ANY = 400
TIMING_CLIP_FY = 800


#: 予想修正からの日数の上限。これより古ければ「無い」と同じ扱いにする
REV_CLIP = 400
#: 予想修正の件数を数える窓（暦日）
REV_WINDOWS = (60, 250)


def forecast_revisions(samples: pd.DataFrame, fins: pd.DataFrame) -> pd.DataFrame:
    """
    業績予想・配当予想の**修正イベント**を各行に付ける。

    なぜ要るか
    --------
    `/fins/summary` の DocType には決算短信以外が混ざっている（全期間）:

      EarnForecastRevision          24,293  業績予想の修正
      DividendForecastRevision       4,174  配当予想の修正
      REITEarnForecastRevision          612
      REITDividendForecastRevision       38

    取り込みには入っていたが、特徴量としては一度も使っていなかった。
    追加の取得はいらない。

    既存の `guidance_revision`（FOP の前回開示比＝修正の**幅**）とは別物で、
    こちらは修正**イベントの発生とタイミング**を見る。`days_since_disc` は
    実績（Sales か NP）のある開示だけを数えているので、修正だけの開示は
    そもそも勘定に入っていない。

    時点整合
    ------
    開示日（DiscDate）で merge_asof の backward。当日の開示は 0 日
    （決算短信は 18:00 過ぎに載り、予測はその後に走る。docs/OPERATIONS.md）。

    向きの出し方
    ----------
    修正行の FOP を、同じ事業年度（CurFYSt）の**直前の開示**の FOP と比べる。
    上方修正なら正、下方修正なら負。前の予想が無ければ欠測（0 にしない）。
    """
    # 列構成は取り込みの状況で変えない。features.all_columns() が要求する
    # 列が欠けると build_dataset ごと落ちる（SystemExit）
    want = features.GROUPS.get("revision", [])
    out = pd.DataFrame(np.nan, index=samples.index, columns=want, dtype=float)
    if "DocType" not in fins.columns:
        return out
    f = fins.copy()
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f.dropna(subset=["DiscDate", "Code"]).sort_values(["Code", "DiscDate"])
    dt_ = f["DocType"].astype(str)
    is_earn = dt_.str.contains("EarnForecastRevision", na=False)
    is_div = dt_.str.contains("DividendForecastRevision", na=False)

    # 向き: 同じ事業年度で、直前の開示の予想と比べる
    if {"FOP", "CurFYSt"} <= set(f.columns):
        fop = pd.to_numeric(f["FOP"], errors="coerce")
        prev = fop.groupby([f["Code"], f["CurFYSt"]], sort=False).shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["_rev_pct"] = np.where(prev > 0, fop / prev * 100.0 - 100.0, np.nan)
    else:
        f["_rev_pct"] = np.nan

    left = samples[["Code", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    left = left.sort_values("Date")

    for flag, prefix in ((is_earn, "rev"), (is_div, "divrev")):
        ev = f.loc[flag, ["Code", "DiscDate", "_rev_pct"]].sort_values("DiscDate")
        if not len(ev):
            continue
        ev = ev.rename(columns={"DiscDate": f"_{prefix}_d",
                                "_rev_pct": f"_{prefix}_pct"})
        m = pd.merge_asof(left, ev, left_on="Date", right_on=f"_{prefix}_d",
                          by="Code", direction="backward", allow_exact_matches=True)
        m = m.sort_values("_i")
        days = (m["Date"] - m[f"_{prefix}_d"]).dt.days
        out[f"days_since_{prefix}"] = np.clip(days.to_numpy(), 0, REV_CLIP)  # noqa: E501
        if prefix == "rev":
            out["rev_pct"] = m[f"_{prefix}_pct"].to_numpy()
            # 向きだけを取り出す。幅が極端でも 1 / 0 に潰れる
            out["rev_up"] = np.where(np.isfinite(out["rev_pct"]),
                                     (out["rev_pct"] > 0).astype(float), np.nan)

    # 件数。上方・下方を分けて数える（回数そのものが材料になる）
    ev = f.loc[is_earn, ["Code", "DiscDate", "_rev_pct"]].copy()
    if len(ev):
        for w in REV_WINDOWS:
            out[f"rev_n_{w}"] = _events_in_window(samples, ev, "DiscDate", w)
        up = ev[ev["_rev_pct"] > 0]
        dn = ev[ev["_rev_pct"] < 0]
        if len(up):
            out["rev_up_n_250"] = _events_in_window(samples, up, "DiscDate", 250)
        if len(dn):
            out["rev_dn_n_250"] = _events_in_window(samples, dn, "DiscDate", 250)
    return out[want]


def _events_in_window(samples: pd.DataFrame, events: pd.DataFrame,
                      date_col: str, days: int) -> np.ndarray:
    """(Code, Date) ごとに、過去 days 暦日に起きた events の件数。"""
    left = samples[["Code", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    ev = events[["Code", date_col]].copy()
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce")
    ev = ev.dropna(subset=[date_col]).sort_values(date_col)
    ev["_c"] = ev.groupby("Code", sort=False).cumcount() + 1
    hi = pd.merge_asof(left.sort_values("Date"), ev, left_on="Date",
                       right_on=date_col, by="Code", direction="backward",
                       allow_exact_matches=True).sort_values("_i")["_c"].to_numpy()
    lo_left = left.assign(_lo=left["Date"] - pd.Timedelta(days=days))
    lo = pd.merge_asof(lo_left.sort_values("_lo"), ev, left_on="_lo",
                       right_on=date_col, by="Code", direction="backward",
                       allow_exact_matches=True).sort_values("_i")["_c"].to_numpy()
    hi = np.where(np.isfinite(hi), hi, 0.0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    return hi - lo


def disclosure_timing(samples: pd.DataFrame, fins: pd.DataFrame) -> pd.DataFrame:
    """
    直近の決算開示からの日数（days_since_disc）と、直近の通期開示からの日数
    （days_since_fy）を各行に付ける。実験24〜26 と同じ定義（docs/FEATURE_IDEAS_EDINET.md）。

    - 実績値（Sales か NP）のある開示だけを数える。予想修正だけの開示は除く
    - 当日の開示は 0 日（allow_exact_matches=True）。財務の結合と同じ作法で、
      決算短信は 18:00 過ぎに J-Quants に載り、予測はその後に走る
      （docs/OPERATIONS.md の実測）
    - 開示が1つも無い行は NaN。上限より古い開示は上限の値
    - 行の順序は元のまま返す
    """
    disc = fins.loc[fins[["Sales", "NP"]].notna().any(axis=1),
                    ["Code", "DiscDate", "CurPerType"]].copy()
    disc["DiscDate"] = pd.to_datetime(disc["DiscDate"])
    disc = disc.dropna(subset=["DiscDate"])
    left = samples[["Code", "Date"]].copy()
    left["_i"] = np.arange(len(left))
    left["Date"] = pd.to_datetime(left["Date"])
    left = left.sort_values("Date")
    any_ = (disc[["Code", "DiscDate"]].sort_values("DiscDate")
            .rename(columns={"DiscDate": "AnyDisc"}))
    fy = (disc.loc[disc["CurPerType"] == "FY", ["Code", "DiscDate"]]
          .sort_values("DiscDate").rename(columns={"DiscDate": "FyDisc"}))
    m = pd.merge_asof(left, any_, left_on="Date", right_on="AnyDisc", by="Code",
                      direction="backward", allow_exact_matches=True)
    m = pd.merge_asof(m, fy, left_on="Date", right_on="FyDisc", by="Code",
                      direction="backward", allow_exact_matches=True)
    m = m.sort_values("_i")
    out = samples.copy()
    out["days_since_disc"] = ((m["Date"] - m["AnyDisc"]).dt.days.astype(float)
                              .clip(upper=TIMING_CLIP_ANY).to_numpy())
    out["days_since_fy"] = ((m["Date"] - m["FyDisc"]).dt.days.astype(float)
                            .clip(upper=TIMING_CLIP_FY).to_numpy())
    return out


# --------------------------------------------------------------------------- #
# 進捗期待（画面の8軸と同じ定義。2026-09-25、運用者の指示で特徴量にも入れる候補）
# --------------------------------------------------------------------------- #

#: 前年同期の進捗の下限（均等ペース Q×25% に対する割合）。画面のデータ取得の値を
#: そのまま使う（同じ定数を2か所に書くと、片方だけ直す事故が起きる）
PROGRESS_FLOOR = JF.PROGRESS_FLOOR

#: 順位を付けるのに要る、それより前の開示の数（同じ四半期・同じ物差し）。
#: 少ないと順位が荒れる。データの最初（2016-10）の直後だけ欠測になる
PROGRESS_PCT_MIN_HISTORY = 200


def seasonal_progress(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    決算の行ごとに、進捗率の基準（前年同期の進捗）・物差し・比率を返す（df と同じ index）。

    画面（scripts/jquants_data_fetcher.progress_benchmark）と同じ規則:
      基準 = 前年の同じ四半期までの累計営業利益 ÷ 前年の通期実績（%）。
             前年度はちょうど1年前に始まったものだけ（決算期の変更で長さが違う年は使わない）
             Q×12.5% 未満なら Q×12.5%（'floor'）、前年の数字が無い・0以下なら Q×25%（'linear'）
      比率 = 進捗率 ÷ 基準（1.0 = 例年どおりのペース）
    本決算（4Q）と、進捗率が無い行は欠測。

    df:  重複を除いた決算（Code / CurFYSt / quarter / DiscDate / progress_rate）
    raw: 重複を除く前の、実績のある開示（Code / CurFYSt / quarter / DiscDate / DiscTime / OP）

    **前年の数字は、その行の開示日までに出ていた版を使う。** df は同じ四半期の
    重複を「最後の開示」に絞ってあるので、そこから前年を引くと、この行より後に
    出た前年の訂正を先に見てしまう（先読み）。
    """
    q = df["quarter"].to_numpy(dtype=int)
    prev = (pd.to_datetime(df["CurFYSt"], errors="coerce")
            - pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    r = raw.copy()
    r["CurFYSt"] = pd.to_datetime(r["CurFYSt"], errors="coerce").dt.strftime("%Y-%m-%d")
    r["DiscDate"] = pd.to_datetime(r["DiscDate"])
    r["quarter"] = r["quarter"].astype(int)
    r = r.dropna(subset=["CurFYSt", "DiscDate"])
    order = ["DiscDate"] + (["DiscTime"] if "DiscTime" in r.columns else [])
    r = r.sort_values(order, kind="mergesort")[["Code", "CurFYSt", "quarter", "DiscDate", "OP"]]
    r["OP"] = pd.to_numeric(r["OP"], errors="coerce")

    def as_of(quarters: np.ndarray) -> np.ndarray:
        """前年度の指定の四半期の累計営業利益（その行の開示日までに出ていた最後の版）。"""
        left = pd.DataFrame({"_i": np.arange(len(df)), "Code": df["Code"].to_numpy(),
                             "CurFYSt": prev.to_numpy(), "quarter": quarters,
                             "DiscDate": pd.to_datetime(df["DiscDate"]).to_numpy()})
        left = left.dropna(subset=["CurFYSt", "DiscDate"]).sort_values("DiscDate",
                                                                        kind="mergesort")
        m = pd.merge_asof(left, r, on="DiscDate", by=["Code", "CurFYSt", "quarter"],
                          direction="backward", allow_exact_matches=True)
        v = np.full(len(df), np.nan)
        v[m["_i"].to_numpy()] = m["OP"].to_numpy(dtype=float)
        return v

    part = as_of(q)                                  # 前年の同じ四半期までの累計
    whole = as_of(np.full(len(df), 4, dtype=int))    # 前年の通期
    even = q * 25.0
    floor = even * PROGRESS_FLOOR
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where((part > 0) & (whole > 0), part / whole * 100.0, np.nan)
    rate = pd.to_numeric(df["progress_rate"], errors="coerce").to_numpy(dtype=float)
    has = np.isfinite(rate) & np.isin(q, (1, 2, 3))
    seasonal = np.isfinite(share)
    bench = np.where(seasonal, np.maximum(share, floor), even)
    basis = np.where(seasonal, np.where(share < floor, "floor", "seasonal"), "linear")
    return pd.DataFrame({
        "progress_bench": np.where(has, bench, np.nan),
        "progress_basis": pd.Series(basis).where(has, None).to_numpy(),
        "progress_ratio": np.where(has, rate / bench, np.nan),
    }, index=df.index)


def progress_percentile(dates, quarter, basis, ratio,
                        min_history: int = PROGRESS_PCT_MIN_HISTORY) -> np.ndarray:
    """
    進捗の比率の順位（0〜100）。同じ四半期・同じ物差し（前年同期 / Q×25%）の、
    **その開示日より前の**開示の中で数える。同じ日の開示は数えず、同順位は半分ずつ数える。

    画面の点数（src/lib/scoring.js の PROGRESS_TABLE）は 2016〜2026年の全期間で作った
    固定の表で順位を付ける。学習に使う列でそれをすると、過去の行を未来の分布で
    順位付けることになる（未来の情報の混入）ので、ここではその日までの開示だけを使う。
    それより前の開示が min_history 件に満たないうちは欠測。
    """
    ratio = np.asarray(ratio, dtype=float)
    quarter = np.asarray(quarter)
    table = np.where(np.isin(np.asarray(basis, dtype=object), ["seasonal", "floor"]),
                     "seasonal", "linear")
    days = pd.to_datetime(pd.Series(dates)).to_numpy()
    ok = np.isfinite(ratio) & np.isin(quarter, (1, 2, 3))
    out = np.full(len(ratio), np.nan)
    for qq in (1, 2, 3):
        for t in ("seasonal", "linear"):
            idx = np.where(ok & (quarter == qq) & (table == t))[0]
            if not len(idx):
                continue
            vals, rank = np.unique(ratio[idx], return_inverse=True)
            n = len(vals)
            tree = np.zeros(n + 1, dtype=np.int64)       # 値の順位ごとの件数（Fenwick 木）

            def below(k: int) -> int:                    # 順位 k 未満の件数
                s = 0
                while k > 0:
                    s += tree[k]
                    k -= k & -k
                return int(s)

            seq = idx[np.argsort(days[idx], kind="mergesort")]
            pos = {int(i): int(rk) for i, rk in zip(idx, rank)}
            total, i = 0, 0
            while i < len(seq):
                j = i
                while j < len(seq) and days[seq[j]] == days[seq[i]]:
                    j += 1
                if total >= min_history:                 # 同じ日の開示どうしは数えない
                    for s_ in seq[i:j]:
                        k = pos[int(s_)]
                        less = below(k)
                        eq = below(k + 1) - less
                        out[s_] = (less + 0.5 * eq) / total * 100.0
                for s_ in seq[i:j]:
                    k = pos[int(s_)] + 1
                    while k <= n:
                        tree[k] += 1
                        k += k & -k
                total += j - i
                i = j
    return out


def attach_fins(samples: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """
    決算（quarterize_panel の出力）を、基準日までに開示済みの直近のものとして結合する。

    **同じ日に複数の期が出ることがある**（古い期の出し直しと新しい期が同じ日。
    実測で決算の行の 0.87%、学習データの 37行・0.17%）。merge_asof は同じ開示日の
    中では最後の行を取るので、会計期間の順（事業年度の開始日・四半期）に並べて
    いちばん新しい期を最後に置く。画面のデータ取得の「最新 = 会計期間が
    いちばん新しい四半期」と同じ規則。2026-09-25 まで開示日だけで並べていて
    （安定でない並べ替え）、どちらの期の決算が付くかが並びまかせだった。
    """
    s = samples.sort_values("Date", kind="mergesort")
    qq = q.sort_values(["DiscDate", "CurFYSt", "quarter"], kind="mergesort")
    out = pd.merge_asof(
        s, qq,
        left_on="Date", right_on="DiscDate", by="Code",
        direction="backward",   # 基準日までに開示済みの直近決算のみ
        allow_exact_matches=True,
    )
    return out.drop(columns=["CurFYSt"])


def quarterize_panel(fins: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
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
    # 重複を除く前の開示。進捗の基準（前年同期）を「その日までに出ていた版」で引くため
    raw_op = df[[c for c in ("Code", "CurFYSt", "quarter", "DiscDate", "DiscTime", "OP")
                 if c in df.columns]].copy()

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
    # 画面の「進捗期待」と同じ定義（前年同期の進捗で割った比率）。実験46の候補
    sp = seasonal_progress(df, raw_op)
    for c in sp.columns:
        df[c] = sp[c]

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
    # --- ROE の出どころを API に寄せる（運用者の判断・実験38）--- #
    #
    # **ここ1か所で差し替える。** ROE_q0/q1/q2/q3 も ROE_chg* も
    # ROE_up_streak も、すべてこの列を四半期でずらして作る。
    # ROE_q0 だけ API に替えると、q1 以降は財務諸表ベースのままになり、
    # ROE_chg1 = q0 - q1 が**別の定義どうしの引き算**になってしまう。
    # 出どころは1つに揃えること。
    #
    # API が無い行は、これまでどおり 提供値 -> TTM の順で埋める。
    if USE_API_VALUATION.get("roe", True):
        api_roe = api_valuation(df, "DiscDate", data_dir=data_dir, asof=True)["api_roe"]
    else:
        api_roe = pd.Series(np.nan, index=range(len(df)), dtype=float)
    api_roe.index = df.index
    df["roe_basis"] = np.where(
        api_roe.notna(), "api",
        np.where(df["ROE"].notna(), "provided",
                 np.where(np.isfinite(roe_ttm), "ttm", "none")))
    # ROE/ROA は 100% 超が実在するので上限は広く取る。
    # ただし自己資本が極小だと桁外れになる（実測で ROE -178,300%）。
    # API 側もクリップしていない（実測で最小 -8,671%）ので同じ扱いにする
    df["ROE"] = clip_divergent(
        api_roe.where(api_roe.notna(),
                      df["ROE"].where(df["ROE"].notna(),
                                      pd.Series(roe_ttm, index=df.index))),
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
    # 予想営業利益 / 直前に終わった事業年度の実績。1を超えれば増益見通し。
    #
    # **通期決算では FOP が空で、翌期予想は NxFOP に入る。** 実測:
    #
    #   FOP   の充足  1Q 91.5% / 2Q 93.2% / 3Q 92.2% / FY  0.0%
    #   NxFOP の充足  1Q  0.0% /                       FY 87.8%
    #
    # そのため guidance_op_growth は通期行で必ず欠測になり、全体の充足が
    # 50% 止まりだった。この指標は両側スクリーニングで**下位10%が
    # z = -3.68（11窓中10窓で悪い）**と、測った中でいちばん強い
    # （docs/DATA_FIELDS.md / 実験37）。穴を塞ぐ価値がある。
    #
    # 通期の NxFOP と、その次の1Q の FOP は **92.7% が完全一致**（13,185組を
    # 実測）。同じ事業年度の予想を指しているので、繋いでよい。
    #
    # **分母も変える。** FOP は「進行中の事業年度」の予想で、その1年前は
    # shift(4) した4期和。NxFOP は「次の事業年度」の予想なので、比べる相手は
    # いま締めた事業年度＝shift しない4期和。ここを揃えないと、通期行だけ
    # 2年ぶんの伸びを見ることになる。
    prev_op_ttm = g_code["q_op"].transform(
        lambda s: s.shift(4).rolling(4, min_periods=4).sum())
    cur_op_ttm = g_code["q_op"].transform(
        lambda s: s.rolling(4, min_periods=4).sum())
    is_fy = df["CurPerType"].eq("FY").to_numpy() if "CurPerType" in df.columns \
        else np.zeros(len(df), dtype=bool)
    has_nx = "NxFOP" in df.columns
    fop_eff = np.where(is_fy & has_nx, col("NxFOP"), col("FOP"))
    den = np.where(is_fy & has_nx, cur_op_ttm, prev_op_ttm)
    # どちらの予想を使ったかを残す。後から充足の出どころを追えるように
    df["guidance_basis"] = np.where(
        ~np.isfinite(fop_eff), "none",
        np.where(is_fy & has_nx, "NxFOP", "FOP"))
    df["guidance_op_growth"] = clip_divergent(
        pd.Series(np.where(den > 0, fop_eff / den * 100.0 - 100.0, np.nan),
                  index=df.index), -100.0, 1000.0, "guidance_op_growth")
    n_nx = int((df["guidance_basis"] == "NxFOP").sum())
    print(f"[guidance] 予想の出どころ: FOP "
          f"{int((df['guidance_basis'] == 'FOP').sum()):,}行 / "
          f"NxFOP {n_nx:,}行（通期の翌期予想）/ "
          f"無し {int((df['guidance_basis'] == 'none').sum()):,}行")
    print(f"[guidance] guidance_op_growth の充足 "
          f"{df['guidance_op_growth'].notna().mean()*100:.1f}%")
    # 予想の修正: 同じ会計年度で前回開示の予想と比べて何%動いたか。
    # 上方修正は「プラスアルファの好材料」そのもの。
    #
    # **ここは NxFOP で埋めない。** 通期行の NxFOP は「次の事業年度」の
    # 最初の予想で、同じ CurFYSt の中の前回（3Q）の予想とは別の年度を
    # 指している。埋めると、年度をまたいだ差を「修正」として出してしまう。
    # 新しい年度の最初の予想に「修正」は定義できないので、欠測が正しい。
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
        # かつて {a}_slope = (q0-q2)/2 を持っていたが、chg の定数倍でしかなく
        # EDA で8軸すべて chg と r=1.000 だった。情報が無いので作らない。
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

    # 比率の順位（その開示日より前の開示だけで数える）。行の並びに依らない
    df["progress_pct"] = progress_percentile(df["DiscDate"], df["quarter"].to_numpy(),
                                             df["progress_basis"].to_numpy(),
                                             df["progress_ratio"].to_numpy())
    print(f"[progress] 前年同期の基準 {int(df['progress_basis'].isin(['seasonal', 'floor']).sum()):,}行"
          f"（うち下限 {int((df['progress_basis'] == 'floor').sum()):,}）/ Q×25% "
          f"{int((df['progress_basis'] == 'linear').sum()):,}行 / 順位あり "
          f"{int(df['progress_pct'].notna().sum()):,}行")

    keep = (["Code", "DiscDate", "CurFYSt", "quarter", "progress_vs_base", "shares_out",
             "progress_ratio", "progress_pct", "progress_basis",
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
                f"{a}_chg", f"{a}_chg_3q", f"{a}_accel",
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
# 市場環境（地合い）
# --------------------------------------------------------------------------- #

# 地合いの軸に使う上場商品。コードは推測ではなく、/equities/master が返した
# 名称から特定した（research/probe_market_data.py の実測 / docs/MARKET_DATA.md）。
#
#   13210  ＮＥＸＴ ＦＵＮＤＳ 日経２２５連動型上場投信
#          2016-10-03〜 / 出来高0の日 0.0% / 売買代金の中央値 78.5億円
#   15400  純金上場信託（現物国内保管型）
#          2016-10-03〜 / 出来高0の日 0.0% / 売買代金の中央値  6.3億円
#   25160  東証グロース250ＥＴＦ
#          2018-02-01〜 / 出来高0の日 0.0% / 売買代金の中央値  4.5億円
#
# なぜ指数ではなく ETF なのか:
#   指数は /indices/bars/daily?code=... で引ける（79件が存在する）が、
#   レスポンスに名称が無く（列は Code/Date/O/H/L/C）、
#   どのコードが何なのかを確定できない。既知は TOPIX=0000 だけ。
#   ETF は名称で特定でき、しかも /equities/bars/daily に含まれるので
#   取得を増やさずに済む（日次バーは毎日全4,441銘柄を取っている）。
#
# 出来高0の日が1日も無いことを実測済み。出来高0の日があると
# 前日終値が据え置かれ、リターンが人為的に0になる。
MACRO_ETFS = {
    "nk225": "13210",
    "gold": "15400",
    "growth250": "25160",
}


def _macro_series(bars: pd.DataFrame, label: str, code: str) -> Optional[pd.DataFrame]:
    """
    日次バーから1銘柄の終値系列を取り出し、20日・120日リターンにする。

    水準そのものは特徴量にしない。TOPIX 2,700 という値は
    「2024年」とほぼ同義で、モデルが相場局面を暗記する入口になる。

    分割・併合をまたぐので終値は調整後（AdjC）を使う。
    """
    sub = bars[bars["_code"] == code]
    if sub.empty:
        print(f"[warn] 市場環境: コード {code}（{label}）が日次バーに無い。この軸は欠測になる")
        return None
    col = "AdjC" if "AdjC" in sub.columns and sub["AdjC"].notna().any() else "C"
    s = sub[["Date", col]].rename(columns={col: label}).copy()
    s["Date"] = pd.to_datetime(s["Date"])
    s[label] = pd.to_numeric(s[label], errors="coerce")
    s = (s.dropna().sort_values("Date")
         .drop_duplicates("Date", keep="last").reset_index(drop=True))
    if s.empty:
        print(f"[warn] 市場環境: コード {code}（{label}）の終値が全て欠測")
        return None
    out = pd.DataFrame({"Date": s["Date"]})
    out[f"{label}_ret_20"] = s[label].pct_change(20) * 100
    out[f"{label}_ret_120"] = s[label].pct_change(120) * 100
    print(f"[merge] 市場環境 {label}({code}) [{col}]: {len(s):,}日 "
          f"{s['Date'].min().date()}〜{s['Date'].max().date()}")
    return out


def market_environment(bars: pd.DataFrame, topix: pd.DataFrame) -> pd.DataFrame:
    """
    日付ごとの市場環境を1枚にまとめる。

    ここで作る列はすべて「その日は全銘柄同じ値」である。
    したがって同じ日の銘柄の順位付けには寄与せず、
    日ごとの正例率の水準を動かすだけになる。
    横断面正規化（RAW_FOR_RANK）の対象から外してあるのはそのため。

    列数を増やしすぎると危ない。母集団は1,739日しかなく、しかも
    ラベルが60日先を見るので隣り合う日は強く相関する。
    日付単位の実効的な標本数は数十しかない。
    効いているかどうかは all と all_no_market の差で測る。
    """
    tp = topix.copy()
    tp["Date"] = pd.to_datetime(tp["Date"])
    tp["topix"] = pd.to_numeric(tp["topix"], errors="coerce")
    tp = (tp.dropna().sort_values("Date")
          .drop_duplicates("Date", keep="last").reset_index(drop=True))
    env = pd.DataFrame({"Date": tp["Date"]})
    env["topix_ret_20"] = tp["topix"].pct_change(20) * 100
    env["topix_ret_120"] = tp["topix"].pct_change(120) * 100
    # 局面の「荒さ」。リターンとは別の軸で、下げそのものより
    # 荒れているかどうかがブレイクの続きやすさに効く、という仮説
    env["topix_vol_20"] = tp["topix"].pct_change().rolling(20).std() * 100
    # 長期の傾き。20日・120日リターンは直近の勢いしか見ていない
    env["topix_ma200_gap"] = (tp["topix"] / tp["topix"].rolling(200).mean() - 1) * 100

    # 日次バーは1,000万行規模なので、コードの文字列化と絞り込みは1度だけ。
    # 銘柄ごとに astype(str) を呼ぶと軸の数だけ全件を走査することになる
    code_col = bars["Code"].astype(str).str.strip()
    picked = bars[code_col.isin(set(MACRO_ETFS.values()))].copy()
    picked["_code"] = code_col[code_col.isin(set(MACRO_ETFS.values()))]

    for label, code in MACRO_ETFS.items():
        part = _macro_series(picked, label, code)
        if part is None:
            env[f"{label}_ret_20"] = np.nan
            env[f"{label}_ret_120"] = np.nan
            continue
        env = env.merge(part, on="Date", how="outer")

    env = env.sort_values("Date").reset_index(drop=True)
    # 金とTOPIXの差。リスクオフの度合い。
    # 木は特徴量どうしの引き算ができない（分割しかしない）ので、
    # 両方を入れておくだけでは差を見たことにならない
    env["risk_off_20"] = env["gold_ret_20"] - env["topix_ret_20"]
    return env


# --------------------------------------------------------------------------- #
# 業種指数
# --------------------------------------------------------------------------- #

#: 指数コード -> 東証33業種コード -> 業種名。
#:
#: /indices/bars/daily は名称を返さない（列は Code/Date/O/H/L/C）ので、
#: research/identify_indices.py で実測して同定した（docs/INDEX_MAPPING.md）。
#: 手元の株価から業種別の時価総額加重リターンを組み、日次リターンの相関を取った。
#:
#: 「指数コードの16進の並び = 業種コードの昇順の並び」という規則で、
#: 33対中19対が仮説の指す業種と1位一致した（偶然なら期待値1）。
#: 並びも東証33業種の標準的な順序と一致している。
#: 下のコメントは各対の相関と順位（1位なら直接確認できた対）。
#:
#: 1位一致しなかった14対は、手元の照合用系列が数銘柄しか無い業種に
#: 集中している（ゴム製品2銘柄 / 海運業2 / 水産・農林業4 など）。
#: 指数側の問題ではない。特徴量には API が返す本物の指数を使うので、
#: 照合用系列の薄さは特徴量の質に影響しない。
SECTOR_INDEX: List[tuple] = [
    ("0040", "0050", "水産・農林業"),        # r=0.4628 順位8
    ("0041", "1050", "鉱業"),                # r=0.9997 順位1
    ("0042", "2050", "建設業"),              # r=0.6564 順位1
    ("0043", "3050", "食料品"),              # r=0.6334 順位1
    ("0044", "3100", "繊維製品"),            # r=0.6189 順位3
    ("0045", "3150", "パルプ・紙"),          # r=0.5080 順位1
    ("0046", "3200", "化学"),                # r=0.7225 順位3
    ("0047", "3250", "医薬品"),              # r=0.5801 順位1
    ("0048", "3300", "石油･石炭製品"),       # r=0.9979 順位1
    ("0049", "3350", "ゴム製品"),            # r=0.4622 順位16
    ("004A", "3400", "ガラス･土石製品"),     # r=0.6850 順位4
    ("004B", "3450", "鉄鋼"),                # r=0.6263 順位2
    ("004C", "3500", "非鉄金属"),            # r=0.7268 順位1
    ("004D", "3550", "金属製品"),            # r=0.7320 順位2
    ("004E", "3600", "機械"),                # r=0.8262 順位1
    ("004F", "3650", "電気機器"),            # r=0.8921 順位1
    ("0050", "3700", "輸送用機器"),          # r=0.8099 順位1
    ("0051", "3750", "精密機器"),            # r=0.7471 順位1
    ("0052", "3800", "その他製品"),          # r=0.6995 順位1
    ("0053", "4050", "電気･ガス業"),         # r=0.3490 順位20
    ("0054", "5050", "陸運業"),              # r=0.4546 順位8
    ("0055", "5100", "海運業"),              # r=0.2941 順位21
    ("0056", "5150", "空運業"),              # r=0.9996 順位1
    ("0057", "5200", "倉庫･運輸関連業"),     # r=0.4242 順位24
    ("0058", "5250", "情報･通信業"),         # r=0.7450 順位1
    ("0059", "6050", "卸売業"),              # r=0.8202 順位1
    ("005A", "6100", "小売業"),              # r=0.7777 順位1
    ("005B", "7050", "銀行業"),              # r=0.9989 順位1
    ("005C", "7100", "証券･商品先物取引業"), # r=0.7574 順位2
    ("005D", "7150", "保険業"),              # r=0.6598 順位2
    ("005E", "7200", "その他金融業"),        # r=0.5285 順位18
    ("005F", "8050", "不動産業"),            # r=0.7512 順位1
    ("0060", "9050", "サービス業"),          # r=0.8475 順位1
]
assert len(SECTOR_INDEX) == 33, "東証33業種と数が合っていない"
#: 業種コード -> 指数コード
S33_TO_INDEX = {s33: ix for ix, s33, _ in SECTOR_INDEX}


def sector_index_returns(indices: pd.DataFrame) -> pd.DataFrame:
    """
    業種指数の20日・120日リターン（列=業種コード、行=日付）。

    指数は分割の概念が無いので調整の区別は要らない。
    水準そのものは使わない。TOPIX と同じ理由で、水準は年号とほぼ同義になる。
    """
    df = indices.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str).str.strip()
    df["C"] = pd.to_numeric(df["C"], errors="coerce")
    df = (df.dropna(subset=["Date", "Code", "C"])
          .drop_duplicates(["Date", "Code"], keep="last"))
    wide = df.pivot(index="Date", columns="Code", values="C").sort_index()

    missing = [ix for ix in S33_TO_INDEX.values() if ix not in wide.columns]
    if missing:
        print(f"[warn] 業種指数のうち取得できていないコード: {missing}")

    out = {}
    for s33, ix in S33_TO_INDEX.items():
        if ix not in wide.columns:
            continue
        col = wide[ix]
        out[(s33, 20)] = col.pct_change(20, fill_method=None) * 100
        out[(s33, 120)] = col.pct_change(120, fill_method=None) * 100
        # 業種指数の20日ボラ（%）。銘柄のボラを業種で割る比（実験51）に使う
        out[(s33, "vol20")] = col.pct_change(fill_method=None).rolling(20, min_periods=15).std() * 100
    if not out:
        return pd.DataFrame(index=wide.index)
    res = pd.DataFrame(out)
    res.columns = pd.MultiIndex.from_tuples(res.columns, names=["s33", "win"])
    print(f"[merge] 業種指数: {len(S33_TO_INDEX) - len(missing)}業種 / "
          f"{len(wide):,}日 {wide.index.min().date()}〜{wide.index.max().date()}")
    return res


def attach_sector_index(samples: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """
    各サンプルに「その銘柄の業種の指数」を当てる。

    業種は時点別（master_hist から merge_asof 済みの S33 列）。
    最新の業種を過去に当てると、業種変更をまたいだところで別の業種の
    指数が付く。

    ここで作る列は市場環境（market グループ）と違って
    **日付内で銘柄ごとに値が変わる**。同じ日でも業種が違えば違う値になり、
    日付内の順位付けに効く。
    """
    cols = ["sector_ret_20", "sector_ret_120", "rel_sector_20", "sector_vs_topix_20",
            "sector_vol_20"]
    if indices is None or indices.empty or "S33" not in samples.columns:
        if "S33" not in samples.columns:
            print("[warn] S33 が無いので業種指数は付与しない")
        for c in cols:
            samples[c] = np.nan
        return samples

    ret = sector_index_returns(indices)
    if ret.empty:
        for c in cols:
            samples[c] = np.nan
        return samples

    # 日付 × 業種で引く。merge_asof でなく reindex にするのは、
    # 指数の営業日と株価の営業日が同じだから（どちらも東証の暦）。
    # ずれた場合は欠測になり、下の欠測率で気づける
    s33 = samples["S33"].astype(str).str.strip()
    dates = pd.to_datetime(samples["Date"])
    for win, name in ((20, "sector_ret_20"), (120, "sector_ret_120"), ("vol20", "sector_vol_20")):
        sub = ret.xs(win, axis=1, level="win")
        idx = sub.index.get_indexer(dates)
        vals = np.full(len(samples), np.nan)
        colpos = {c: i for i, c in enumerate(sub.columns)}
        arr = sub.to_numpy()
        cpos = s33.map(colpos).to_numpy()
        ok = (idx >= 0) & pd.notna(cpos)
        if ok.any():
            vals[ok] = arr[idx[ok], cpos[ok].astype(int)]
        samples[name] = vals

    # 銘柄自身のリターンから業種ぶんを引く。
    # 木は特徴量どうしの引き算ができないので、差は明示的に列にする
    samples["rel_sector_20"] = samples["ret_20d"] - samples["sector_ret_20"]
    # その業種が市場に対して強いか
    samples["sector_vs_topix_20"] = samples["sector_ret_20"] - samples["topix_ret_20"]

    for c in cols:
        miss = float(samples[c].isna().mean() * 100)
        print(f"[merge] 業種指数 {c}: 欠測 {miss:.1f}%")
    return samples


# --------------------------------------------------------------------------- #
# 組み立て
# --------------------------------------------------------------------------- #

def attach_credit_ratio(samples: pd.DataFrame, margin: pd.DataFrame,
                        days=None) -> pd.DataFrame:
    """
    信用倍率（信用買残 ÷ 信用売残）を、**公表日**で結合する。

    週次の信用残は基準日（通常は金曜）の翌週の第2営業日に公表される
    （availability.MARGIN_PUBLISH_BD）。2026-09-24 まで基準日で結合していて、
    金曜・月曜のブレイク（母集団の 38.5%）で公表前の値を学習に使っていた。
    予測の時点では同じ新しさの値は手に入らない。

    古すぎる値は使わない: 公表日から 21日より前のものは欠測にする
    （基準日で21日としていたのと同じ考え方を、公表日で数え直した）。
    """
    if not len(margin):
        out = samples.copy()
        out["credit_ratio"] = np.nan
        return out
    m = margin[["Date", "Code", "LongVol", "ShrtVol"]].copy()
    m["Date"] = pd.to_datetime(m["Date"])
    m["Code"] = m["Code"].astype(str)
    m["credit_ratio"] = np.where(m["ShrtVol"] > 0, m["LongVol"] / m["ShrtVol"], np.nan)
    m["_avail"] = AV.margin_available(m, days)
    m = m.dropna(subset=["_avail"]).sort_values("_avail")[["_avail", "Code", "credit_ratio"]]
    # 並びは従来（基準日で結合していたとき）と同じく Date 順で返す。
    # 行の並びが変わると学習の結果も僅かに動き、結合の修正の効果と混ざる
    left = samples.drop(columns=["credit_ratio"], errors="ignore").sort_values("Date")
    out = pd.merge_asof(left, m, left_on="Date", right_on="_avail", by="Code",
                        direction="backward", allow_exact_matches=True,
                        tolerance=pd.Timedelta("21D"))
    return out.drop(columns=["_avail"])


#: 学習データの行の並び。build() はこの順に並べてから書く
ROW_ORDER = ["Date", "Code"]


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    学習データの行を (Date, Code) の順に並べる（2026-09-25、運用者の了承）。

    学習は行を並べ替えずに使い、LightGBM・XGBoost は行を間引く（subsample）ので、
    並びが変わると同じ種でも別の行を引く。これまで並びは結合の順しだいで、
    コードを直すたびに変わり、中身が同じでも結果が種を変えたのと同じくらい
    動いていた（docs/MODEL_ADOPTION_RULES.md §10 実験46 の読み方 4）。
    同じキーの行が残っても並びが決まるよう、安定な並べ替えにする。
    """
    key = pd.DataFrame({"d": pd.to_datetime(df["Date"]).to_numpy(),
                        "c": df["Code"].astype(str).to_numpy()})
    order = key.sort_values(["d", "c"], kind="mergesort").index.to_numpy()
    return df.iloc[order].reset_index(drop=True)


def build(data_dir: str, out_path: str) -> pd.DataFrame:
    bars = load_parts("bars", data_dir)
    fins = load_parts("fins", data_dir)
    topix = load_parts("topix", data_dir)
    try:
        indices = load_parts("indices", data_dir)
    except SystemExit:
        print("[warn] indices データなし。業種指数は欠測として扱います")
        indices = pd.DataFrame(columns=["Date", "Code", "C"])
    try:
        margin = load_parts("margin", data_dir)
    except SystemExit:
        print("[warn] margin データなし。需給軸は欠測として扱います")
        margin = pd.DataFrame(columns=["Date", "Code", "LongVol", "ShrtVol"])

    # 年別の行数。年ごとの parquet を1つ落とすと、その年が丸ごと消えるだけでなく
    # 前後の年の rolling も壊れる。集計結果の年別内訳を読む前に、
    # 入力そのものが揃っているかを確かめられるようにしておく。
    _d = pd.to_datetime(bars["Date"])
    _by = _d.dt.year.value_counts().sort_index()
    print("[input] 日次バーの年別行数: "
          + " ".join(f"{y}:{n:,}" for y, n in _by.items()))
    # 行数が揃っていても、年をまたいで銘柄コードが噛み合っていなければ
    # 各コードの履歴が分断され、rolling が成立しなくなる。
    # 年ごとの銘柄数と、前年と共通のコード数を見る。
    _codes = {int(y): set(g) for y, g in bars.assign(_y=_d.dt.year)
              .groupby("_y")["Code"]}
    _prev = None
    _row = []
    for y in sorted(_codes):
        cur = _codes[y]
        ov = f"/前年と共通 {len(cur & _prev):,}" if _prev else ""
        _row.append(f"{y}:{len(cur):,}{ov}")
        _prev = cur
    print("[input] 年別の銘柄数: " + " ".join(_row))
    # 月別の行数。年内の穴（特定の月だけ無い）は年別集計では見えない
    _m = _d.dt.to_period("M").value_counts().sort_index()
    _gap = [str(k) for k, v in _m.items() if v < _m.median() * 0.5]
    print(f"[input] 月別行数の中央値 {int(_m.median()):,} / "
          f"半分未満の月: {', '.join(_gap) if _gap else 'なし'}")
    # 行があっても値が無ければ同じこと。列が欠けたファイルが1つ混ざると、
    # concat がその列を NaN で埋めるので、行数の確認だけでは見つからない。
    for c in ("H", "AdjH", "C", "AdjC", "Vo", "AdjVo"):
        if c not in bars.columns:
            print(f"[warn] 日次バーに {c} 列が無い")
            continue
        miss = bars[c].isna().groupby(_d.dt.year).mean() * 100
        bad = [f"{y}:{v:.0f}%" for y, v in miss.items() if v > 5]
        if bad:
            print(f"[warn] {c} の欠測が多い年: " + " ".join(bad))

    # --- 一般市場で数え始める日（TOKYO PRO MARKET からの移行）と上場日 --- #
    segments = load_market_segments(data_dir)
    if segments is None:
        print("[gm] master_hist が無い。TPM の判定はせず、上場日はデータの行から決める")
    gm = general_market_start(bars, segments)
    write_general_market_start(
        gm, os.path.splitext(out_path)[0] + "_general_market_start.json",
        GENERAL_MARKET_START)
    start = (gm.loc[gm["tpm"]].set_index("Code")["start"] if GENERAL_MARKET_START else None)
    print(f"[gm] 78週の窓を一般市場に移ってからの行で数える: {GENERAL_MARKET_START}")

    print("\n[panel] 株価系の指標を算出")
    df = price_panel(bars, start=start)
    if POPULATION == "breakout":
        print("[panel] 52週高値の更新日を判定")
        df = mark_new_highs(df)
        print("[panel] ブレイク後の上昇ラベルを付与")
        df = attach_rise_label(df)
        df = add_breakout_context(df)

        n_all = int(df["is_new_high"].sum())
        samples = df[df["is_fresh_break"] == True].copy()   # noqa: E712
        weeks = round(HIGH_WINDOW / 245 * 52)
        print(f"[sample] {weeks}週高値の更新日: {n_all:,}行")
        print(f"[sample] うち新規のブレイク（直前{BREAKOUT_COOLDOWN}営業日に更新なし）: "
              f"{len(samples):,}行")
        print(f"[label] 定義: {DEFAULT_RISE.name}")
        for tag, sub in (("高値更新日", df[df["is_new_high"] == True]),   # noqa: E712
                         ("新規ブレイク", samples)):
            c = sub.groupby(sub["Date"].dt.year).size()
            print(f"[sample] {tag}の年別: "
                  + " ".join(f"{y}:{n:,}" for y, n in c.items()))
        # 更新日が0の年があったとき、どこで止まっているのかを分ける。
        # 「値が無い(high)」「基準が作れない(prior)」「届かなかった(比率)」は別物。
        _y = df["Date"].dt.year
        print(f"[diag] dtype high={df['high'].dtype} "
              f"prior={df['high52w_prior'].dtype} date={df['Date'].dtype}")
        rows = []
        for y, gy in df.groupby(_y):
            ratio = (gy["high"] / gy["high52w_prior"]).replace(
                [np.inf, -np.inf], np.nan)
            mx = f"{ratio.max():.3f}" if ratio.notna().any() else "—"
            rows.append(f"{y}:行{len(gy):,}/high有{gy['high'].notna().sum():,}"
                        f"/prior有{gy['high52w_prior'].notna().sum():,}/最大比{mx}")
        print("[diag] 年別: " + "  ".join(rows))
        # 全期間そろっている1銘柄を1本追う。全体の集計では
        # 「新規上場が多いだけ」と区別がつかない。
        code = df["Code"].value_counts().idxmax()
        one = df[df["Code"] == code]
        per = one.groupby(one["Date"].dt.year).agg(
            n=("high", "size"), high=("high", "count"),
            prior=("high52w_prior", "count"))
        print(f"[diag] 最長銘柄 {code}（{len(one):,}行）: "
              + " ".join(f"{y}:{r.n}/{r.high}/{r.prior}"
                         for y, r in per.iterrows())
              + "  （行/high有/prior有）")
        report_rise_funnel(samples)
    else:
        print("[panel] ブレイクアウト日を判定")
        df = breakout_flags(df)
        print("[panel] ラベルを付与")
        df = attach_labels(df)
        df["ym"] = df["Date"].dt.to_period("M")
        is_month_end = (df.groupby(["Code", "ym"], sort=False)["Date"]
                        .transform("max") == df["Date"])
        samples = df[is_month_end].copy()
        print(f"[sample] 月末サンプル: {len(samples):,}行")

    # --- 除外条件 --- #
    before = len(samples)
    if KEEP_UNLABELED:
        # 予測用。今日のブレイクはラベルが確定していない（先 RISE_HORIZON 営業日ぶんの
        # 値動きがまだ無い）ので、落とすと予測したい行が消える。
        n_un = int(samples["label"].isna().sum())
        print(f"[filter] ラベル未確定を残す（予測用）: {n_un:,}件が未確定のまま")
    else:
        samples = samples[samples["label"].notna()]
        print(f"[filter] ラベル未確定を除外: {before:,} -> {len(samples):,}")

    before = len(samples)
    samples = samples[samples["high52w"].notna()]
    print(f"[filter] 52週高値が未定義(上場直後)を除外: {before:,} -> {len(samples):,}")

    if POPULATION != "breakout":
        # 高値更新日を母集団にする場合、全件が定義上ここに引っかかるので掛けない
        before = len(samples)
        samples = samples[samples["r_high"] < MAX_RHIGH_AT_T]
        print(f"[filter] 基準日ですでに高値圏(R_high>={MAX_RHIGH_AT_T})を除外: "
              f"{before:,} -> {len(samples):,}")

    report_liquidity_threshold(samples)
    if MIN_TRADING_VALUE is None:
        print("[filter] 流動性の下限は設定していない（MIN_TRADING_VALUE=None）")
    else:
        before = len(samples)
        samples = samples[samples["tv_ma20"] >= MIN_TRADING_VALUE]
        print(f"[filter] 低流動性(20日平均売買代金<{MIN_TRADING_VALUE}億円)を除外: "
              f"{before:,} -> {len(samples):,}")

    # --- 財務をマージ（開示日ベースの point-in-time） --- #
    print("\n[merge] 財務情報を開示日ベースで結合")
    # data_dir を渡す（渡さないと既定の research/_data のバリュエーションを読み、
    # 別の場所のデータで作ったときに API の ROE が抜けていた。2026-09-25）
    q = quarterize_panel(fins, data_dir)
    samples = attach_fins(samples, q)
    # 決算が古すぎる（1年以上前）場合は使わない
    stale = (samples["Date"] - samples["DiscDate"]).dt.days > 365
    fin_cols = [c for c in q.columns if c not in ("Code", "DiscDate", "CurFYSt")]
    samples.loc[stale, fin_cols] = np.nan
    print(f"[merge] 決算が1年以上古いサンプル: {int(stale.sum()):,}件を欠測扱い")

    # --- 開示からの日数（実験24〜26 で採用。docs/MODEL_ADOPTION_RULES.md §6） --- #
    samples = disclosure_timing(samples, fins)
    print(f"[merge] 開示からの日数: 中央値 {samples['days_since_disc'].median():.0f}日 / "
          f"欠測 {samples['days_since_disc'].isna().mean()*100:.1f}%")

    # --- 予想の修正イベント（DocType。取り込み済みで未使用だった） --- #
    rev = forecast_revisions(samples, fins)
    if rev.shape[1]:
        dup = [c for c in rev.columns if c in samples.columns]
        if dup:
            print(f"[rev] 既存と同名の列は捨てる: {dup}")
            rev = rev.drop(columns=dup)
        samples = pd.concat([samples, rev], axis=1)
        cov = ", ".join(f"{c} {samples[c].notna().mean()*100:.0f}%"
                        for c in rev.columns)
        print(f"[rev] 予想修正 {rev.shape[1]}列を追加: {cov}")

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

    # --- PER / PBR の出どころを API に寄せる（運用者の判断・実験38）--- #
    #
    # 自前は「株価 ÷ eps_ttm」で、eps_ttm は4四半期そろったときだけ作る。
    # そこで落ちる行が API では埋まる（実測 PER 82.0% -> 91.1%、
    # うち負でも極端でもない**まともな増分が +9.1pt**）。
    # API 値はクリップされていない（実測 PER 最大 18,920）ので、
    # 自前と同じ上限を掛けてから使う。API が無い行は自前で埋める。
    av = api_valuation(samples, "Date", data_dir=data_dir)
    per = np.where(samples["eps_ttm"] > 0, px / samples["eps_ttm"], np.nan)
    pbr = np.where(samples["BPS"] > 0, px / samples["BPS"], np.nan)
    for name, own, cap in (("per", per, PER_MAX), ("pbr", pbr, PBR_MAX)):
        a = (av[f"api_{name}"].to_numpy() if USE_API_VALUATION.get(name, True)
             else np.full(len(samples), np.nan))
        merged = np.where(np.isfinite(a) & (a > 0), a, own)
        samples[name] = np.where(np.isfinite(merged) & (merged <= cap),
                                 merged, np.nan)
        samples[f"{name}_basis"] = np.where(
            np.isfinite(a) & (a > 0), "api",
            np.where(np.isfinite(own), "own", "none"))
        n_api = int((np.isfinite(a) & (a > 0)).sum())
        n_own = int(np.isfinite(own).sum())
        print(f"[valuation] {name}: API {n_api:,}行 / 自前 {n_own:,}行 "
              f"-> 採用 {int(samples[name].notna().sum()):,}行")
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

    for name, cap in (("per", PER_MAX), ("pbr", PBR_MAX)):
        a = (av[f"api_{name}"].to_numpy() if USE_API_VALUATION.get(name, True)
             else np.full(len(samples), np.nan))
        own = per if name == "per" else pbr
        merged = np.where(np.isfinite(a) & (a > 0), a, own)
        n = int((np.isfinite(merged) & (merged > cap)).sum())
        if n:
            print(f"[filter] {name} > {cap:g} を欠測に: {n:,}件"
                  f"（分母が丸め誤差レベル。逆数側は残している）")

    # --- 2026-09-22 に足した8本から作る特徴量 --- #
    #
    # 取り込みが届いていない種別は列が空になるだけで、ここは落ちない。
    # 時価総額・株価・per・ROE_q0 を使うので、バリュエーションの後に置く。
    # 引数の data_dir を渡す（既定の場所を決め打ちすると、別の場所のデータで
    # 作ったつもりの追加特徴量が、既定の場所から読まれていた）
    extra = extra_features.attach(samples, data_dir)
    if extra.shape[1]:
        dup = [c for c in extra.columns if c in samples.columns]
        if dup:
            # 同名の列を黙って上書きしない。気づけない形で値が変わる
            print(f"[extra] 既存と同名の列は捨てる: {dup}")
            extra = extra.drop(columns=dup)
        samples = pd.concat([samples, extra], axis=1)
        print(f"[extra] {extra.shape[1]}列を追加")

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
    samples = attach_credit_ratio(samples, margin, trading_calendar.load(data_dir).days)

    # --- 業種・市場区分を時点別に結合 --- #
    # 最新のマスタを過去のサンプルに当てると先読みになる。
    # とくに市場区分は2022年4月の東証再編で全銘柄が変わっているため、
    # 2018年のサンプルに現在の区分を付けるのは誤り。
    # 月次スナップショットを merge_asof で「その時点で有効だった区分」に合わせる。
    mh_paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if mh_paths:
        mh = pd.concat([pd.read_parquet(x) for x in mh_paths], ignore_index=True)
        mh["Date"] = pd.to_datetime(mh["Date"])
        # 実際に来ている項目名を必ず出す。
        # "ScaleCat" を実測せずに書いたせいで100%欠測の空列を作っていた。
        # 名前を決め打ちすると、外したときに静かに空列になって気づけない。
        # 実際に来ている項目名を必ず出す。
        # 名前を決め打ちすると、外したときに静かに空列になって気づけない。
        print(f"[merge] master_hist の項目: {sorted(mh.columns)}")
        want = ("S33", "S17", "Mkt", "ScaleCat")
        found = [c for c in want if c in mh.columns]
        for c in want:
            if c not in found:
                print(f"[warn] master_hist に {c} が無い。{c.lower()}_code は欠測になる")
        mh = (mh[["Date", "Code"] + found].dropna(subset=["Date", "Code"])
              .sort_values("Date").drop_duplicates(["Date", "Code"], keep="last"))
        print(f"[merge] 業種・市場区分を時点別に結合 ({len(mh):,}行 / "
              f"{mh['Date'].nunique()}時点)")
        samples = pd.merge_asof(
            samples.sort_values("Date"), mh,
            on="Date", by="Code", direction="backward")
        for c in found:
            # カテゴリは数値コードにする（LightGBM はそのまま分岐できる）
            samples[f"{c.lower()}_code"] = encode_category(samples[c], c)
    else:
        print("[merge] master_hist が無いため業種・市場区分は付与しない")
    for c in ("s33", "s17", "mkt", "scalecat"):
        if f"{c}_code" not in samples.columns:
            samples[f"{c}_code"] = np.nan

    # --- ETF・REIT を母集団から外す --- #
    # 市場区分は master_hist を結合してからでないと分からないので、
    # ほかの除外条件（上の「除外条件」ブロック）とは離れてここに置く。
    samples = drop_excluded_markets(samples)

    # --- 上場からの年数（実験47の候補。本番の列には入れていない） --- #
    samples["listing_years"] = listing_years(samples["Code"], samples["Date"], gm)
    _ly = samples["listing_years"]
    print(f"[gm] 上場からの年数: 値あり {_ly.notna().mean()*100:.1f}% / "
          f"打ち止め（{LISTING_CAP_YEARS:g}年）{(_ly >= LISTING_CAP_YEARS).mean()*100:.1f}% / "
          f"{LISTING_CAP_YEARS:g}年未満 {(_ly < LISTING_CAP_YEARS).mean()*100:.1f}%")

    # --- 市場環境（地合い） --- #
    print("[merge] 市場環境の特徴量を結合")
    env = market_environment(bars, topix)
    samples = pd.merge_asof(
        samples.sort_values("Date"), env, on="Date", direction="backward",
    )
    for c in [c for c in env.columns if c != "Date"]:
        miss = float(samples[c].isna().mean() * 100)
        if miss > 0:
            print(f"[merge] 市場環境 {c}: 欠測 {miss:.1f}%")

    # --- 業種指数 --- #
    # market の後に置く。rel_sector_20 は ret_20d を、
    # sector_vs_topix_20 は topix_ret_20 を使うので、両方が揃ってからでないと作れない
    print("[merge] 業種指数を結合")
    samples = attach_sector_index(samples, indices)
    samples = attach_vol_vs_market(samples)

    # --- 最終的な特徴量セット --- #
    samples["log_trading_value"] = np.log1p(samples["tv_ma20"])
    samples["log_market_cap"] = np.log1p(samples["market_cap"])
    # 母集団を広げると規模と決算の揃い方の分布が変わる。
    # どちらの群の行なのかをモデルが直接見られるようにフラグで持たせる
    samples["cap_band"] = cap_band(samples["market_cap"])
    samples["fund_complete"] = fund_complete_flag(samples)
    _cb = samples["cap_band"].value_counts(dropna=False).sort_index()
    print("[flag] 時価総額の帯: " + " ".join(
        f"{'欠測' if pd.isna(k) else int(k)}:{v:,}" for k, v in _cb.items()))
    print(f"[flag] 決算の変化が全て作れる行: "
          f"{samples['fund_complete'].sum():,.0f} / {len(samples):,} "
          f"({samples['fund_complete'].mean() * 100:.1f}%)")

    # --- 横断面正規化 ---
    # 絶対値のままだと相場局面に依存する。上昇局面では全銘柄の R_high が高くなるため、
    # 「R_high が 87%」の意味が期間によって変わってしまう。
    # 同じ日付内での順位（パーセンタイル）に直すと、
    # 「その時点で全銘柄中どのくらいの位置か」という局面に依らない量になる。
    #
    # 実測で訓練期間の正例率 6.19% に対しテスト期間 21.66% と3倍以上ずれており、
    # 絶対値の特徴量では学習が成立していなかった（docs/MODEL_RESULTS.md 参照）。
    # --- 決算の完全性で絞る --- #
    # 順位を付ける前に絞る。順位は「その日の中で何位か」なので、
    # 絞ったあとの母集団の中で付け直さないと意味がずれる。
    report_fund_completeness(samples)
    samples = apply_fund_requirement(samples)
    report_fund_quality(samples)
    samples = apply_fund_quality(samples)

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
    # ラベルの材料も持ち出す。EDA で将来リターンの分布を見たり、
    # 「どの条件で正例から外れたか」を数えたりするのに要る。
    # 未来の情報なので特徴量には絶対に入れない（下でチェックする）。
    meta_cols = ["Code", "Date", "close", "close_raw", "high52w", "tv_ma20",
                 "market_cap", "label",
                 "future_rise", "keep_days_cnt", "end_level", "uptrend_end",
                 "eps_ttm", "BPS", "roe_basis", "bps_basis",
                 # PER / PBR / ROE を API に寄せた（実験38）。
                 # どの行がどちらから来たかを追えるようにしておく
                 "per_basis", "pbr_basis",
                 # 進捗の基準の物差し（前年同期 / 下限 / Q×25%。実験46）
                 "progress_basis",
                 # 目的変数の基準の価格（翌営業日の寄り。LABEL_ENTRY）
                 "entry_price"]
    meta_cols = [c for c in meta_cols if c in samples.columns]

    # 未来から作った列が特徴量に混ざるとリークで結果が無意味になる。
    # 名前の付け替えで事故が起きうるので、機械的に止める。
    leak = sorted(set(FUTURE_COLS) & set(feature_cols))
    if leak:
        raise SystemExit(f"[fatal] 未来の列が特徴量に含まれています: {leak}")

    out = samples[meta_cols + feature_cols].copy()
    # 予測用は未確定（NaN）が残るので int にできない。
    # Int64（欠測を持てる整数）にして、学習側では notna() で弾く
    out["label"] = (out["label"].astype("Int64") if KEEP_UNLABELED
                    else out["label"].astype(int))
    # 行の並びを固定する（canonical_order の説明）
    out = canonical_order(out)

    print(f"\n[result] {len(out):,}サンプル / 特徴量{len(feature_cols)}個")
    _lab = out["label"].dropna()
    if len(_lab):
        print(f"[result] 正例率: {_lab.mean()*100:.2f}%  ({int(_lab.sum()):,}件) "
              f"/ ラベル未確定 {int(out['label'].isna().sum()):,}件")
    else:
        print(f"[result] ラベルは全行未確定（予測用）: {len(out):,}件")
    print(f"[result] 期間: {out['Date'].min().date()} 〜 {out['Date'].max().date()}")
    print(f"[result] 銘柄数: {out['Code'].nunique():,}")

    miss = out[feature_cols].isna().mean().sort_values(ascending=False)
    print("\n[欠測率の高い特徴量]")
    for name, rate in miss.head(8).items():
        print(f"  {name:<24} {rate*100:5.1f}%")

    # どの定義で作ったデータセットかを残す。あとから追跡できないと混乱するため。
    #
    # 目的変数は母集団で切り替わる。breakout なら attach_rise_label（RiseConfig）、
    # month_end なら attach_labels（LabelConfig）。ここで常に LabelConfig を
    # 書いていたので、breakout で作ったデータセットの meta に**使っていない定義**が
    # 入っていた。追跡のために置いてある欄が、追跡を誤らせていた。
    if POPULATION == "breakout":
        label_cfg = {
            "kind": "rise",
            "population": POPULATION,
            "high_window": HIGH_WINDOW,
            "horizon": DEFAULT_RISE.horizon,
            "vol_norm_k": DEFAULT_RISE.vol_norm_k,
            # vol_norm_k が None のときだけ効く固定しきい値
            "threshold": DEFAULT_RISE.threshold,
            "keep_days": DEFAULT_RISE.keep_days,
            "end_ratio": DEFAULT_RISE.end_ratio,
            "end_window": DEFAULT_RISE.end_window,
            "require_uptrend": DEFAULT_RISE.require_uptrend,
            "trend_short": DEFAULT_RISE.trend_short,
            "trend_long": DEFAULT_RISE.trend_long,
            # 上昇を測る基準の価格（next_open = 翌営業日の寄り / close = 更新日の終値）
            "entry": DEFAULT_RISE.entry,
            "name": DEFAULT_RISE.name,
            "forward_needed": DEFAULT_RISE.horizon,
        }
    else:
        label_cfg = {
            "kind": "breakout_within_horizon",
            "population": POPULATION,
            "high_window": DEFAULT_LABEL.high_window,
            "horizon": [DEFAULT_LABEL.horizon_start, DEFAULT_LABEL.horizon_end],
            "hold_days": DEFAULT_LABEL.hold_days,
            "hold_drawdown": DEFAULT_LABEL.hold_drawdown,
            "vol_multiple": DEFAULT_LABEL.vol_multiple,
            "sustain_days": DEFAULT_LABEL.sustain_days,
            "sustain_ratio": DEFAULT_LABEL.sustain_ratio,
            "name": DEFAULT_LABEL.name,
            "forward_needed": DEFAULT_LABEL.forward_needed,
        }
    meta = {
        "labelConfig": label_cfg,
        "n": int(len(out)),
        "positiveRate": (None if out["label"].notna().sum() == 0
                         else round(float(out["label"].mean()), 4)),
        "features": feature_cols,
        "from": str(out["Date"].min().date()), "to": str(out["Date"].max().date()),
        # 既定で回すかぎり空。掃引で母集団を差し替えたときだけ中身が入る。
        # これが空でないデータセットを既定の学習に使ってはいけない
        "sweepOverrides": dict(SWEEP_OVERRIDES),
        # 予測用に作ったデータセットかどうか。True のものを学習に使ってはいけない
        "keepUnlabeled": bool(KEEP_UNLABELED),
        "nUnlabeled": int(out["label"].isna().sum()),
        "cooldown": BREAKOUT_COOLDOWN,
        "minTradingValue": MIN_TRADING_VALUE,
        "excludeMktCodes": list(EXCLUDE_MKT_CODES),
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[label] 定義: {label_cfg['name']} "
          f"(ラベル確定に将来 {label_cfg['forward_needed']} 営業日)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    print(f"\n[done] {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)")
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ブレイクアウト予測の学習データを構築する")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "dataset.parquet"))
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="ラベル未確定の行も残す（日次予測用。学習には使わない）")
    args = ap.parse_args(argv)
    if args.keep_unlabeled:
        globals()["KEEP_UNLABELED"] = True
        print("[mode] ラベル未確定の行を残す（予測用データセット）")
    build(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
