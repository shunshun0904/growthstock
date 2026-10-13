#!/usr/bin/env python3
"""
機関投資家（保有目的が運用）の直近の買いのフラグ。実験89（research/exp/e89_inst_buy_ab.py）の候補で、
本番の列（features.GROUPS・features.DEFAULT_PRESET）には入れていない。

運用者の依頼（2026-10-13）「特徴量として、機関投資家による買いが直近であったかどうかのフラグを入れたいです」と、
大量保有報告書の実測（research/exp/probe_lvs_holders.py）を見たうえでの定義の回答（同日）:

  機関投資家  大量保有報告書（J-Quants /edinet/large-volume-shareholders、保存名 lvshld）の保有者（Hldrs）のうち、
             保有目的（HldgPurp）が運用（純投資・投資一任・顧客の資産）の保有者だけ（「運用目的のみ」）。
             区分は probe と同じ正規表現・同じ順（証券業務・貸借・担保 → 重要提案 → 経営・安定株主・政策投資 →
             運用。先に当たった区分に入れる。運用者に見せた件数はこの区分のもの）
  買い        その保有者の保有割合（ShsRatio）が前回（ShsRatioLast）より増えた、または前回が無い・0 で今回が正
             （新たに5%を超えた）（「割合が増えた・新規5%超」= probe の D1）
  直近        報告書の提出日（SubDate）が、その行の日付から数えて20暦日以内・60暦日以内（当日を含む）
             → 「どちらも特徴量として追加する。」（2列とも足す）

列
  inst_buy_20  直近20暦日に、運用目的の保有者の買いを含む報告書が1件でもあれば 1、なければ 0
  inst_buy_60  同じく60暦日

決め事
  - 提出日で結合する。提出日の当日の行から使う（extra_features.large_volume の lvs_* と同じ。
    availability.RULES["lvshld"]: 16:00〜17:31 JST に少しずつ出る）。数え方は lvs_n_* と同じ
    extra_features._count_in_window（その日を含む過去 N 暦日。提出は休日も届く）
  - 訂正報告書（DocTypeCode 360）は数えない。元の報告の訂正なので、同じ取引を遅れてもう一度数えることになる
  - 1報告書 = 1件（共同保有者のうち運用目的の買いが何人いても1件）
  - 取り込みの始まり（lvshld の最初の提出日。実測 2021-07-01）より前と、始まりから窓の日数がそろう前の行は
    NaN。報告が無かったのか取り込んでいないのか区別できないため（0 では埋めない）
  - 今回の割合が読めない保有者は買いにしない

本番に入れるなら（実験89 の結果と運用者の判断のあと）: extra_features.BUILDERS と features.GROUPS に足し、
feature_dict に日本語名を足す。当日の夜の予測は 17:31 JST より前の取り込みだと当日の提出の一部を取りこぼす
（学習は全部を見る）。lvs_* と同じ性質で、翌日の取り直しで埋まる。

公開ログには件数・割合・日付だけを出す（銘柄・保有者の名前・値は出さない）。
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extra_features as XF  # noqa: E402

#: 保有目的（HldgPurp）の文の区分（上から順に当てる）。research/exp/probe_lvs_holders.py の PURPOSE_CLASSES と同じ
#: （tests/test_inst_features.py で同じであることを確かめる）
PURPOSE_CLASSES = [
    ("証券業務・貸借・担保", re.compile(r"証券業務|商品在庫|トレーディング|ディーリング|自己勘定|貸借|担保")),
    ("重要提案", re.compile(r"重要提案")),
    ("経営・安定株主・政策投資", re.compile(r"経営|安定株主|政策投資|支配|業務提携|資本提携|取引関係")),
    ("運用（純投資・投資一任・顧客の資産）", re.compile(r"純投資|投資一任|顧客|運用|信託財産|ポートフォリオ")),
]
INVEST = "運用（純投資・投資一任・顧客の資産）"
#: 訂正報告書の書類の種類のコード（数えない）
CORRECTION_DOC_TYPE = 360
WINDOWS: Tuple[int, ...] = (20, 60)
COLUMNS = [f"inst_buy_{w}" for w in WINDOWS]


def purpose_class(text) -> str:
    """保有目的の文の区分（probe_lvs_holders.purpose_class と同じ）。"""
    if not isinstance(text, str) or not text.strip():
        return "（空）"
    for label, pat in PURPOSE_CLASSES:
        if pat.search(text):
            return label
    return "その他"


def _ratio(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if v is not None and np.isfinite(v) else np.nan


def is_buy(h: dict) -> bool:
    """保有者の割合が前回より増えた、または前回が無い・0 で今回が正（新たに5%超）。"""
    now, last = _ratio(h.get("ShsRatio")), _ratio(h.get("ShsRatioLast"))
    if not np.isfinite(now):
        return False
    if not np.isfinite(last) or last == 0:
        return now > 0
    return now > last


def report_has_invest_buy(cell) -> bool:
    """報告書の保有者の一覧（Hldrs）に、運用目的の保有者の買いが1人でもいるか。"""
    return any(purpose_class(h.get("HldgPurp")) == INVEST and is_buy(h) for h in XF._holders(cell))


def buy_events(d: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    運用目的の保有者の買いを含む報告書の (Code, SubDate)。訂正は除き、1報告書 = 1件。
    返す記録は件数だけ（ログに出してよいもの）。
    """
    stats: Dict = {"reports": int(len(d))}
    empty = pd.DataFrame({"Code": pd.Series(dtype=str), "SubDate": pd.Series(dtype="datetime64[ns]")})
    if not len(d) or "Hldrs" not in d.columns or "SubDate" not in d.columns or "Code" not in d.columns:
        stats.update(corrections=0, events=0)
        return empty, stats
    x = d
    if "DocTypeCode" in x.columns:
        corr = pd.to_numeric(x["DocTypeCode"], errors="coerce") == CORRECTION_DOC_TYPE
        stats["corrections"] = int(corr.sum())
        x = x[~corr]
    else:
        stats["corrections"] = 0
    hit = np.fromiter((report_has_invest_buy(c) for c in x["Hldrs"]), dtype=bool, count=len(x))
    keep = [c for c in ("DocId", "Code", "SubDate") if c in x.columns]
    ev = x.loc[hit, keep].copy()
    ev["Code"] = ev["Code"].astype(str)
    ev["SubDate"] = pd.to_datetime(ev["SubDate"], errors="coerce").dt.normalize()
    ev = ev.dropna(subset=["Code", "SubDate"])
    ev = ev.drop_duplicates(keep)
    stats["events"] = int(len(ev))
    stats["events_by_year"] = {int(y): int(n) for y, n in sorted(Counter(ev["SubDate"].dt.year).items())}
    return ev[["Code", "SubDate"]].reset_index(drop=True), stats


def flags(samples: pd.DataFrame, events: pd.DataFrame, start: Optional[pd.Timestamp],
          windows: Sequence[int] = WINDOWS) -> pd.DataFrame:
    """
    (Code, Date) ごとの inst_buy_<w>（その日を含む過去 w 暦日に買いの報告があれば 1、なければ 0）。
    start（取り込みの始まり）から w 暦日がそろう前の行は NaN。start が None なら全行 NaN（取り込みが無い）。
    """
    out = pd.DataFrame(index=samples.index)
    dates = pd.to_datetime(samples["Date"]).dt.normalize().to_numpy()
    for w in windows:
        if start is None:
            out[f"inst_buy_{w}"] = np.nan
            continue
        if len(events):
            n = XF._count_in_window(samples, events.assign(_one=1), "SubDate", w)
        else:
            n = np.zeros(len(samples))
        v = (np.asarray(n, dtype=float) > 0).astype(float)
        # 窓の最初の日（Date − w + 1）が取り込みの始まりより前なら、まだ分からない
        v[dates < np.datetime64(pd.Timestamp(start).normalize() + pd.Timedelta(days=w - 1))] = np.nan
        out[f"inst_buy_{w}"] = v
    return out


def attach(samples: pd.DataFrame, data_dir: str = XF.DATA_DIR,
           windows: Sequence[int] = WINDOWS) -> Tuple[pd.DataFrame, Dict]:
    """samples（Code, Date）に inst_buy_* を付けた列（samples と同じ行数・同じ並び）と、件数の記録。"""
    d = XF._read("lvshld", data_dir)
    if not len(d) or "SubDate" not in d.columns:
        return flags(samples, pd.DataFrame(columns=["Code", "SubDate"]), None, windows), {"reports": 0}
    sub = pd.to_datetime(d["SubDate"], errors="coerce")
    start = sub.min().normalize() if sub.notna().any() else None
    ev, stats = buy_events(d)
    out = flags(samples, ev, start, windows)
    stats["start"] = str(start.date()) if start is not None else None
    stats["last"] = str(sub.max().date()) if sub.notna().any() else None
    for c in out.columns:
        v = out[c]
        stats[f"{c}_known"] = int(v.notna().sum())
        stats[f"{c}_ones"] = int((v == 1).sum())
    return out, stats


def describe(stats: Dict, n_rows: int) -> str:
    """ログ用の1行ずつ（件数・割合・日付だけ）。"""
    lines = [f"[inst] 大量保有報告書 {stats.get('reports', 0):,}件（提出日 {stats.get('start')} 〜 {stats.get('last')}）・"
             f"訂正 {stats.get('corrections', 0):,}件を除く・運用目的の保有者の買いを含む報告書 {stats.get('events', 0):,}件"]
    by = stats.get("events_by_year") or {}
    if by:
        lines.append("[inst] 年ごと: " + " / ".join(f"{y}年 {n:,}" for y, n in by.items()))
    for c in COLUMNS:
        k, o = stats.get(f"{c}_known", 0), stats.get(f"{c}_ones", 0)
        lines.append(f"[inst] {c}: 分かる行 {k:,} / {n_rows:,}（{k / n_rows * 100 if n_rows else 0:.1f}%）・"
                     f"そのうち 1 の行 {o:,}（{o / k * 100 if k else 0:.1f}%）")
    return "\n".join(lines)
