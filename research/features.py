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
    # --- 決算: 1期前・2期前の水準 ---
    "fund_lag": [f"{a}_{q}" for a in FUND_AXES for q in ("q1", "q2")],
    # --- 決算: 変化と傾き（CANSLIM の核心は水準より加速）---
    "fund_trend": [f"{a}_{s}" for a in FUND_AXES for s in ("chg", "chg1", "slope")],
    # --- 株価位置 ---
    "price": ["r_high", "r_high_3m", "r_high_6m"],
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
    "valuation": ["per", "pbr", "earnings_yield", "book_yield"],
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
    c for g in ("fund_level", "fund_lag", "fund_trend", "price", "volume",
                "liquidity", "supply", "progress", "valuation")
    for c in GROUPS[g]
]

#: 順位版のグループ。列名は元の列に `_r` を付けたもの。
# 順位版のグループは自動生成する。
# 対象は RAW_FOR_RANK と揃える。ここがずれると
# 「順位列が存在しないプリセット」が黙って出来てしまう。
RANKED_GROUPS = ("fund_level", "fund_lag", "fund_trend", "price", "volume",
                 "liquidity", "supply", "progress", "valuation")
GROUPS.update({
    f"{g}_rank": [f"{c}_r" for c in GROUPS[g]] for g in RANKED_GROUPS
})
assert set(RAW_FOR_RANK) == {c for g in RANKED_GROUPS for c in GROUPS[g]}, \
    "RAW_FOR_RANK と RANKED_GROUPS がずれている"

#: 実験用のプリセット。グループ名の並びで指定する。
PRESETS: Dict[str, List[str]] = {
    # 素朴なベースライン。株価位置だけ
    "price_only": ["price"],
    # テクニカル・需給のみ（決算を使わない）
    "technical": ["price", "volume", "liquidity", "supply", "market"],
    # 決算のみ（株価を使わない）。従来比較用
    "fundamental": ["fund_level", "fund_lag", "fund_trend", "progress"],
    # 決算は直近の水準だけ（ラグと傾きを落とす）
    "fund_simple": ["fund_level", "price", "volume", "liquidity", "supply",
                    "progress", "market"],
    # 全部（絶対値のみ。順位版は別プリセットで比較する）
    "all": ["fund_level", "fund_lag", "fund_trend", "price", "volume",
            "liquidity", "supply", "progress", "valuation", "turnaround",
            "market"],
    # バリュエーションのみ
    "valuation_only": ["valuation"],
    # 決算 + バリュエーション + 黒字転換（株価位置を使わない）
    "fundamental_v2": ["fund_level", "fund_lag", "fund_trend", "progress",
                       "valuation", "turnaround"],
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
                         "progress_rank"],
    # 全部の順位版
    "rank_all": ["fund_level_rank", "fund_lag_rank", "fund_trend_rank",
                 "price_rank", "volume_rank", "liquidity_rank",
                 "supply_rank", "progress_rank", "valuation_rank",
                 "turnaround", "market"],
    # 決算 + バリュエーションの順位版
    "rank_fundamental_v2": ["fund_level_rank", "fund_lag_rank", "fund_trend_rank",
                            "progress_rank", "valuation_rank", "turnaround"],
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
