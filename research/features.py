#!/usr/bin/env python3
"""
特徴量セットの定義。

特徴量を足して何パターンか試すことを前提に、**グループ単位で名前を付けて**
組み合わせられるようにしてある。新しい特徴量を試す手順は:

  1. 下の GROUPS に列名を足す（build_dataset.py が列を作る）
  2. データセットを再構築する（Release から読むので数分・API取得なし）
  3. train_model.py --features <セット名> で比較する

セットの追加は PRESETS に1行足すだけ。学習側のコードは触らない。
"""
from __future__ import annotations

from typing import Dict, List

# 決算の軸。build_dataset.py が各軸について q0/q1/q2/chg/chg1/slope を作る。
#
# 充足率は docs/DATA_FIELDS.md と docs/MODEL_FUNDAMENTAL_COVERAGE.md を参照。
# eps_growth / sales_growth は「前年が0以下だと欠測」という定義のため
# 赤字企業が丸ごと落ちる。同じ内容を負値でも定義できる形にした
# *_sym（対称変化率）を併置し、どちらが効くか比較できるようにする。
FUND_AXES = ["eps_growth", "sales_growth", "eps_growth_sym", "sales_growth_sym",
             "ROE", "ROA", "op_margin", "equity_ratio"]

#: 特徴量のグループ。キーがグループ名、値が列名。
GROUPS: Dict[str, List[str]] = {
    # --- 決算: 直近の水準 ---
    "fund_level": [f"{a}_q0" for a in FUND_AXES],
    # --- 決算: 1〜3期前の水準 ---
    "fund_lag": [f"{a}_{q}" for a in FUND_AXES for q in ("q1", "q2", "q3")],
    # --- 決算: 変化と傾き（CANSLIM の核心は水準より加速）---
    # chg1/chg2/chg3 は決算をまたぐ各段の差。
    # 「直近1回だけ伸びた」と「3期続けて伸びている」を区別するために各段を持つ。
    # 決定木は q1 と q2 から差を作れないため、明示的に列にする必要がある。
    "fund_trend": [f"{a}_{s}" for a in FUND_AXES
                   for s in ("chg1", "chg2", "chg3", "chg", "chg_3q",
                             "slope", "accel")],
    # --- 決算: 連続性 ---
    # 何期続けて伸びているか / 何期プラスを保っているか。
    # 連言条件（3期とも増加）は水準の線形結合では表現できない。
    "fund_streak": [f"{a}_{s}" for a in FUND_AXES
                    for s in ("up_streak", "pos_ratio")],
    # --- 株価位置 ---
    # 母集団を52週高値の更新日にすると r_high は全件ほぼ100になり使えなくなる。
    # ラグ版（3ヶ月前・6ヶ月前の位置）は「どこから上がってきたか」を表すので残る。
    "price": ["r_high", "r_high_3m", "r_high_6m"],
    # --- ブレイクの性質 ---
    # 高値更新日を母集団にしたので「どう抜けたか」が主役になる。
    # 効くかどうかは日付内診断と層別評価で測る（現時点では仮説）。
    "breakout": ["base_length", "break_margin", "close_position",
                 "ret_20d", "vol_20d"],
    # --- 出来高 ---
    "volume": ["volume_trend"],
    # --- 流動性・サイズ ---
    "liquidity": ["log_trading_value", "log_market_cap"],
    # --- 需給 ---
    "supply": ["credit_ratio"],
    # --- 決算進捗 ---
    "progress": ["progress_vs_base"],
    # --- バリュエーション ---
    # PER/PBR は API に無いので株価と決算から作った（docs/DATA_FIELDS.md）。
    # 逆数（益回り・純資産倍率の逆数）も持つ。赤字や債務超過で
    # PER/PBR が負になると「割安」と誤読されるが、逆数なら符号が意味を保つ。
    "valuation": ["per", "pbr", "earnings_yield", "book_yield",
                  "psr", "sales_yield", "peg"],
    # --- 配当 ---
    # 優待は J-Quants に項目が無いので入れられない
    # （docs/DATA_FIELDS.md の111項目を機械的に走査して0件）
    "dividend": ["div_yield", "payout_ratio", "has_dividend"],
    # --- キャッシュフロー（充足率 約51%）---
    # 利益の質。利益は出ているが営業CFが伴わない銘柄を分ける
    "cashflow": ["cfo_yield", "fcf_yield", "cfo_to_op", "accruals"],
    # --- 収益性・効率 ---
    "efficiency": ["net_margin", "ordinary_margin", "asset_turnover"],
    # --- 会社予想 ---
    # 「プラスアルファの好材料」に最も近い。上方修正はそれ自体が材料
    "guidance": ["guidance_op_growth", "guidance_revision"],
    # --- 業種・市場区分（時点別）---
    # 最新のマスタを過去に当てると先読みになるため月次スナップショットを使う
    "sector": ["s33_code", "s17_code", "scalecat_code", "mkt_code"],
    # --- 黒字転換 ---
    # 赤字->黒字は小型株で株価が最も動くイベントだが、
    # 従来の成長率定義では欠測として捨てられていた
    "turnaround": ["eps_growth_turn", "sales_growth_turn"],
    # --- 市場環境（これを外すとモデルは相場局面を暗記しやすくなる）---
    "market": ["topix_ret_20", "topix_ret_120"],
}

#: 横断面正規化（同じ日付内でのパーセンタイル順位）を作る対象の列。
#: 絶対値のままだと相場局面に依存するため、順位に直して局面依存を消す。
#: 実測で訓練 6.19% / テスト 21.66% と正例率が3倍以上ずれており、
#: 絶対値の特徴量では学習が成立していなかった。
RAW_FOR_RANK: List[str] = [
    c for g in ("fund_level", "fund_lag", "fund_trend", "fund_streak", "price",
                "breakout", "volume", "liquidity", "supply", "progress",
                "valuation", "dividend", "cashflow", "efficiency", "guidance")
    for c in GROUPS[g]
]

#: 順位版のグループ。列名は元の列に `_r` を付けたもの。
# 順位版のグループは自動生成する。
# 対象は RAW_FOR_RANK と揃える。ここがずれると
# 「順位列が存在しないプリセット」が黙って出来てしまう。
RANKED_GROUPS = ("fund_level", "fund_lag", "fund_trend", "fund_streak", "price",
                 "breakout", "volume", "liquidity", "supply", "progress",
                 "valuation", "dividend", "cashflow", "efficiency", "guidance")
GROUPS.update({
    f"{g}_rank": [f"{c}_r" for c in GROUPS[g]] for g in RANKED_GROUPS
})
assert set(RAW_FOR_RANK) == {c for g in RANKED_GROUPS for c in GROUPS[g]}, \
    "RAW_FOR_RANK と RANKED_GROUPS がずれている"

#: 「全部入り」に含めるグループ。
#: `all` と `rank_all` はここから作る。片方に足し忘れる事故を防ぐため、
#: グループ名を書く場所を1箇所に絞る。
ALL_GROUPS: List[str] = [
    "fund_level", "fund_lag", "fund_trend", "fund_streak", "price", "breakout",
    "volume", "liquidity", "supply", "progress", "valuation", "dividend",
    "cashflow", "efficiency", "guidance", "sector", "turnaround", "market",
]

#: 実験用のプリセット。グループ名の並びで指定する。
PRESETS: Dict[str, List[str]] = {
    # 素朴なベースライン。株価位置だけ
    "price_only": ["price"],
    # テクニカル・需給のみ（決算を使わない）
    "technical": ["price", "breakout", "volume", "liquidity", "supply", "market"],
    # ブレイクの性質だけ。高値更新日を母集団にしたときの素朴なベースライン
    "breakout_only": ["breakout"],
    # 決算のみ（株価を使わない）。従来比較用
    "fundamental": ["fund_level", "fund_lag", "fund_trend", "fund_streak", "progress"],
    # 決算は直近の水準だけ（ラグと傾きを落とす）
    "fund_simple": ["fund_level", "price", "volume", "liquidity", "supply",
                    "progress", "market"],
    # 全部（絶対値のみ。順位版は別プリセットで比較する）
    "all": ALL_GROUPS,
    # バリュエーションのみ
    "valuation_only": ["valuation"],
    # 決算 + バリュエーション + 黒字転換（株価位置を使わない）
    "fundamental_v2": ["fund_level", "fund_lag", "fund_trend", "fund_streak",
                       "progress", "valuation", "dividend", "cashflow",
                       "efficiency", "guidance", "turnaround"],
    # 決算まわり全部 + 業種。算出できるファンダメンタルズを可能な限り入れる
    "fundamental_v3": ["fund_level", "fund_lag", "fund_trend", "fund_streak",
                       "progress", "valuation", "dividend", "cashflow",
                       "efficiency", "guidance", "sector", "turnaround"],
    # 今回足したものだけ。既存の決算軸を抜いた効果を見る
    "extras_only": ["valuation", "dividend", "cashflow", "efficiency",
                    "guidance", "sector"],
    # 市場環境を抜いた全部。
    # all との差が「相場局面をどれだけ暗記していたか」の目安になる
    "all_no_market": [g for g in GROUPS
                      if g != "market" and not g.endswith("_rank")],

    # --- 横断面正規化版（同じ日付内でのパーセンタイル順位）---
    # 株価位置の順位だけ。price_only と直接比較する
    "rank_price_only": ["price_rank"],
    # テクニカル・需給の順位版
    "rank_technical": ["price_rank", "volume_rank", "liquidity_rank",
                       "supply_rank", "market"],
    # 決算の順位版のみ
    "rank_fundamental": ["fund_level_rank", "fund_lag_rank", "fund_trend_rank",
                         "fund_streak_rank", "progress_rank"],
    # 全部の順位版。ALL_GROUPS から機械的に導出する。
    # グループ名を2箇所に手書きしていたため、`all` に足したときに追随せず
    # 列数がずれた。今度は同じ一覧から作る。
    "rank_all": [f"{g}_rank" if f"{g}_rank" in GROUPS else g for g in ALL_GROUPS],
    # 決算 + バリュエーションの順位版
    "rank_fundamental_v2": ["fund_level_rank", "fund_lag_rank", "fund_trend_rank",
                            "fund_streak_rank", "progress_rank",
                            "valuation_rank", "dividend_rank", "cashflow_rank",
                            "efficiency_rank", "guidance_rank", "turnaround"],
    # 絶対値と順位の両方（順位が絶対値に上乗せの情報を持つかを見る）
    "raw_and_rank": ["fund_level", "fund_trend", "price", "volume", "liquidity",
                     "supply", "progress", "market",
                     "fund_level_rank", "fund_trend_rank", "price_rank",
                     "volume_rank", "liquidity_rank", "supply_rank",
                     "progress_rank"],
}

DEFAULT_PRESET = "all"


def columns(preset: str) -> List[str]:
    """プリセット名から列名の一覧を返す。"""
    if preset not in PRESETS:
        raise KeyError(f"未知のプリセット: {preset}. 利用可能: {sorted(PRESETS)}")
    out: List[str] = []
    for g in PRESETS[preset]:
        out.extend(GROUPS[g])
    # 順序を保ったまま重複を除く
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def all_columns() -> List[str]:
    """データセットに作るべき全列。build_dataset.py が使う。

    どのプリセットからでも参照されうる列の和集合を返す。
    `columns("all")` ではないことに注意。"all" は絶対値のみのプリセットであり、
    それを使うと横断面正規化した `*_r` 列がデータセットから落ちてしまう。
    """
    out: List[str] = []
    seen = set()
    for preset in PRESETS:
        for c in columns(preset):
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def describe(preset: str) -> str:
    groups = PRESETS[preset]
    return f"{preset} ({len(columns(preset))}列): " + " + ".join(groups)
