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

FUND_AXES = ["eps_growth", "sales_growth", "ROE", "op_margin"]

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
    # --- 市場環境（これを外すとモデルは相場局面を暗記しやすくなる）---
    "market": ["topix_ret_20", "topix_ret_120"],
}

#: 実験用のプリセット。グループ名の並びで指定する。
PRESETS: Dict[str, List[str]] = {
    # 素朴なベースライン。株価位置だけ
    "price_only": ["price"],
    # テクニカル・需給のみ（決算を使わない）
    "technical": ["price", "volume", "liquidity", "supply", "market"],
    # 決算のみ（株価を使わない）
    "fundamental": ["fund_level", "fund_lag", "fund_trend", "progress"],
    # 決算は直近の水準だけ（ラグと傾きを落とす）
    "fund_simple": ["fund_level", "price", "volume", "liquidity", "supply",
                    "progress", "market"],
    # 全部
    "all": list(GROUPS.keys()),
    # 市場環境を抜いた全部。
    # all との差が「相場局面をどれだけ暗記していたか」の目安になる
    "all_no_market": [g for g in GROUPS if g != "market"],
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
    """データセットに作るべき全列。build_dataset.py が使う。"""
    return columns("all")


def describe(preset: str) -> str:
    groups = PRESETS[preset]
    return f"{preset} ({len(columns(preset))}列): " + " + ".join(groups)
