#!/usr/bin/env python3
"""
生データの保存と差分取得の管理。

J-Quants からの全期間取得は約2.5時間かかる。毎回取り直すのは無駄なので、
取得済みの生データを GitHub Release に置き、次回は**まだ取っていない日だけ**を取る。

  初回        : 2.5時間（変わらない）
  日次更新    : 約5秒（1営業日ぶんのみ）
  定義の再検証: 0秒（取得不要）

保存先は GitHub Release（タグ `data-raw`）。外部の認証情報が不要で永続する。
アップロード/ダウンロードは Actions の `gh` CLI が行い、
本モジュールは「どの日を取得済みか」を manifest で管理する。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, Iterable, List, Set

MANIFEST_NAME = "manifest.json"

#: manifest で「取得済みの日」を管理するデータ種別。
#: master は日付ごとの蓄積ではなく毎回最新に上書きするため、ここには含めない。
KINDS = ("bars", "fins", "margin", "topix")


def manifest_path(data_dir: str) -> str:
    return os.path.join(data_dir, MANIFEST_NAME)


def load_manifest(data_dir: str) -> Dict[str, Dict]:
    """
    取得済みの日付を記録した manifest を読む。無ければ空で返す。

    「その日を取得しに行ったか」を記録する（行数ではなく）。
    財務のようにその日に開示が0件でも「取得済み」であり、再取得の必要はないため。
    """
    p = manifest_path(data_dir)
    if not os.path.exists(p):
        return {k: {"fetched_days": []} for k in KINDS}
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    for k in KINDS:
        m.setdefault(k, {"fetched_days": []})
    return m


def save_manifest(data_dir: str, manifest: Dict[str, Dict]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for k in KINDS:
        days = sorted(set(manifest.get(k, {}).get("fetched_days", [])))
        manifest[k] = {"fetched_days": days, "count": len(days),
                       "from": days[0] if days else None,
                       "to": days[-1] if days else None}
    # master は日付リストを持たない（毎回最新に上書き）ので、そのまま残す
    manifest["updatedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(manifest_path(data_dir), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def fetched_days(manifest: Dict[str, Dict], kind: str) -> Set[str]:
    return set(manifest.get(kind, {}).get("fetched_days", []))


def missing_days(manifest: Dict[str, Dict], kind: str,
                 candidates: Iterable[dt.date]) -> List[dt.date]:
    """まだ取得していない営業日だけを返す。"""
    have = fetched_days(manifest, kind)
    return [d for d in candidates if d.isoformat() not in have]


def mark_fetched(manifest: Dict[str, Dict], kind: str, days: Iterable[dt.date]) -> None:
    cur = set(manifest.setdefault(kind, {"fetched_days": []})["fetched_days"])
    cur.update(d.isoformat() for d in days)
    manifest[kind]["fetched_days"] = sorted(cur)


def summarize(manifest: Dict[str, Dict]) -> str:
    lines = []
    for k in KINDS:
        d = manifest.get(k, {})
        days = d.get("fetched_days", [])
        if days:
            lines.append(f"  {k:<8} {len(days):>5}日  {days[0]} 〜 {days[-1]}")
        else:
            lines.append(f"  {k:<8} {'0':>5}日  (未取得)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 年別ファイルへの分割保存
# --------------------------------------------------------------------------- #

def year_path(data_dir: str, kind: str, year: int) -> str:
    return os.path.join(data_dir, f"{kind}_{year}.parquet")


def merge_into_years(data_dir: str, kind: str, new_df, date_col: str = "Date") -> List[str]:
    """
    新しく取得したデータを年別 parquet にマージする。

    年で分けるのは、更新時に触るファイルを最小限にするため
    （今年ぶんだけ書き換えれば済み、過去年は再アップロード不要）。
    """
    import pandas as pd

    if new_df is None or len(new_df) == 0:
        return []
    df = new_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    written: List[str] = []

    for year, part in df.groupby(df[date_col].dt.year):
        p = year_path(data_dir, kind, int(year))
        if os.path.exists(p):
            old = pd.read_parquet(p)
            old[date_col] = pd.to_datetime(old[date_col])
            part = pd.concat([old, part], ignore_index=True)
        # 同じ日・同じ銘柄の重複は後勝ち（訂正を反映）
        subset = [c for c in (date_col, "Code") if c in part.columns]
        if subset:
            part = part.drop_duplicates(subset=subset, keep="last")
        part = part.sort_values(subset or [date_col]).reset_index(drop=True)
        part.to_parquet(p, index=False, compression="zstd")
        written.append(p)

    return written
