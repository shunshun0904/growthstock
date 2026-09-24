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
import glob
import json
import os
from typing import Dict, Iterable, List, Optional, Set

MANIFEST_NAME = "manifest.json"

#: manifest で「取得済みの日」を管理するデータ種別。
#: master は日付ごとの蓄積ではなく毎回最新に上書きするため、ここには含めない。
KINDS = ("bars", "fins", "margin", "topix", "indices", "master_hist")


def manifest_path(data_dir: str) -> str:
    return os.path.join(data_dir, MANIFEST_NAME)


def load_manifest(data_dir: str) -> Dict[str, Dict]:
    """
    取得済みの日付を記録した manifest を読む。無ければ空で返す。

    「その日を取得しに行ったか」を記録する（行数ではなく）。
    財務のようにその日に開示が0件でも「取得済み」であり、再取得の必要はないため。
    公表前に叩いてしまった日だけは例外扱いにする（confirmed_days）。
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


#: 「その日ぶんがまだ公表されていないかもしれない」とみなす日数。
#: 当日を1日目と数えるので、1 は「当日だけ」の意味。
#:
#: 取得済みの記録は**行数ではなく「叩きに行ったか」**で付けている
#: （開示0件の日を毎回叩き直さないため）。だが公表前に叩いた日まで
#: それで記録すると、**その日のデータは二度と取りに行かない**。
#: missing_days は「候補 − 記録」なので、一度記録した日は候補から消える。
#:
#: 取り込みを 21:30 JST から 16:05 JST に前倒ししたことで、これが
#: 実害になった。実測（research/probe_update_time.py, 2026-09-18）:
#:
#:   bars / master_hist  16:00 までに出ている（初回ポーリング時点で既にあり）
#:   indices / topix     16:30
#:   fins                18:00
#:   margin              週次。金曜ぶんが翌週に出る（20:30 までには出ない）
#:
#: margin だけ長いのは、これが既に起きていたため。Release の保存データを
#: 確認すると、記録は 2026-09-14 まであるのに実データは 2026-08-28 が
#: 最後だった（2026-09-18 時点）。公表前に叩いた週が取得済みとして
#: 記録され、そのまま失われている。戻すには --forget を使う。
PUBLISH_LAG_DAYS = {
    "bars": 1,
    "indices": 1,
    "topix": 1,
    "fins": 1,
    "master_hist": 1,
    "margin": 10,   # 営業日3日ぶんの公表ラグ＋連休ぶんの余裕
}
DEFAULT_PUBLISH_LAG_DAYS = 1


def confirmed_days(kind: str, requested: Iterable[dt.date],
                   got: Iterable[dt.date], today: dt.date) -> List[dt.date]:
    """
    叩きに行った日のうち、**取得済みとして記録してよい日**だけを返す。

    行が取れた日は当然よい。取れなかった日は「本当に0件」か
    「まだ公表されていない」かを区別できないので、公表ラグの中に
    入っている日は記録しない（次回また取りに行く）。

    公表ラグを過ぎても0件なら、それは本当に0件（祝日・開示なし）なので
    記録する。ここで記録しないと、その日を永久に叩き続けることになる。
    """
    lag = PUBLISH_LAG_DAYS.get(kind, DEFAULT_PUBLISH_LAG_DAYS)
    have = {d for d in got}
    return [d for d in requested
            if d in have or (today - d).days >= lag]


def forget_days(manifest: Dict[str, Dict], kind: str,
                since: Optional[dt.date] = None) -> List[str]:
    """
    取得記録だけを外す。**保存済みの parquet は消さない。**

    `reset_kind` は保存データごと消して全期間を取り直す。公表前に叩いて
    しまった数日を取り直したいだけのときには重すぎるうえ、取り直しの
    途中で落ちると保存データが欠けたまま残る。

    こちらは記録を外すだけなので、次の差分取得が同じ日を取りに行き、
    `merge_into_years` が既存ファイルに上書きマージする。取れなければ
    現状のまま何も変わらない。外した日のリストを返す。
    """
    cur = sorted(fetched_days(manifest, kind))
    keep, dropped = [], []
    for d in cur:
        if since is None or dt.date.fromisoformat(d) >= since:
            dropped.append(d)
        else:
            keep.append(d)
    manifest.setdefault(kind, {})["fetched_days"] = keep
    return dropped


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

def reset_kind(data_dir: str, manifest: Dict[str, Dict], kind: str) -> List[str]:
    """
    ある種別の保存済みデータと取得記録を消す。

    列を増やしたときに使う。取得済みの parquet には新しい列が入っていないが、
    manifest 上は「取得済み」なので incremental では永久に取り直されない。
    その状態を解消するために、その種別だけを取得前の状態に戻す。

    他の種別（株価など）には触らない。株価は取り直すと数時間かかる。
    """
    removed = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{kind}_*.parquet"))):
        os.remove(path)
        removed.append(os.path.basename(path))
    manifest.pop(kind, None)
    return removed


def year_path(data_dir: str, kind: str, year: int) -> str:
    return os.path.join(data_dir, f"{kind}_{year}.parquet")


#: 年別ファイルにマージするとき「同じ行」とみなすキー（種別ごと）。
#:
#: 2026-09-24 まで全種別で (日付列, Code) を使っていて、同じ日に同じ銘柄の行が
#: 複数あるデータを黙って1行に潰していた。取り込みと同じ引数で8日ぶんを叩いた
#: 実測（research/probe_dedupe_keys.py）で捨てていた割合は、業種別の空売り比率
#: 97%（銘柄の列が無いので「同じ日」で潰れ、1日34行が1行になっていた）、大量保有
#: 報告書 15%、大株主 8%、決算発表予定 6%、決算 3%。
#:
#: どのキーも API の応答で一意になることを実測で確かめたもの。**ここに無い種別は
#: 保存しない**（KeyError）。種別を足したら、ここにキーを書いてテストで確かめる。
#: None は行そのもの（全列）で見分ける。報告に ID が無い空売り残高報告は、
#: 同じ報告者・同じ計算日でも比率の違う行があり、どの列の組でも一意にならない。
ROW_KEYS: Dict[str, Optional[List[str]]] = {
    "bars": ["Date", "Code"],
    "fins": ["DiscNo"],
    "margin": ["Date", "Code"],
    "topix": ["Date"],
    "indices": ["Date", "Code"],
    "master_hist": ["Date", "Code"],
    "valuation": ["Date", "Code"],
    "shortratio": ["Date", "S33"],
    "marginalert": ["PubDate", "Code"],
    "earndate": ["PubDate", "Code", "FQName", "SchDate"],
    "lvshld": ["DocId"],
    "mjrshld": ["DocId"],
    "xhold": ["DocId"],
    "shortsale": None,
}


def _comparable(df):
    """入れ子の列（list / dict / ndarray）を文字列にした写し。行の比較だけに使う。"""
    import numpy as np

    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(
                lambda v: isinstance(v, (list, dict, tuple, np.ndarray))).any():
            out[c] = out[c].map(lambda v: repr(v.tolist() if isinstance(v, np.ndarray) else v))
    return out


def row_key(kind: str, df) -> List[str]:
    """その種別の行のキー（列名の並び）。登録が無い・列が無いなら例外。"""
    if kind not in ROW_KEYS:
        raise KeyError(f"{kind}: 行のキーが決まっていない（data_store.ROW_KEYS に足す）")
    key = ROW_KEYS[kind]
    if key is None:
        return list(df.columns)
    missing = [c for c in key if c not in df.columns]
    if missing:
        raise KeyError(f"{kind}: キーの列が無い {missing}（列: {list(df.columns)[:12]}）")
    return list(key)


def merge_into_years(data_dir: str, kind: str, new_df, date_col: str = "Date") -> List[str]:
    """
    新しく取得したデータを年別 parquet にマージする。

    年で分けるのは、更新時に触るファイルを最小限にするため
    （今年ぶんだけ書き換えれば済み、過去年は再アップロード不要）。

    同じキーの行は後勝ち（訂正を反映）。キーは種別ごと（ROW_KEYS）。
    **今回取った行どうしでキーが重なり、ほかの列が違うなら保存しない**
    （ValueError）。キーの決め方が誤っていて、別々の行を潰すことになるため。
    まったく同じ行の重なりは1行にする。
    """
    import pandas as pd

    if new_df is None or len(new_df) == 0:
        return []
    df = new_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    key = row_key(kind, df)

    cmp_new = _comparable(df)
    df = df.loc[~cmp_new.duplicated()]                     # まったく同じ行は1行に
    cmp_new = cmp_new.loc[df.index]
    clash = cmp_new.duplicated(subset=key, keep=False)
    if clash.any():
        raise ValueError(
            f"{kind}: 今回取った行の中でキー {key} が重なり、ほかの列が違う行が "
            f"{int(clash.sum())}行ある。別々の行を潰すことになるので保存しない"
            f"（data_store.ROW_KEYS を見直す）")

    written: List[str] = []
    for year, part in df.groupby(df[date_col].dt.year):
        p = year_path(data_dir, kind, int(year))
        if os.path.exists(p):
            old = pd.read_parquet(p)
            old[date_col] = pd.to_datetime(old[date_col])
            part = pd.concat([old, part], ignore_index=True)
        # 同じキーの行は後勝ち（訂正を反映）
        cmp = _comparable(part)
        part = part.loc[~cmp.duplicated(subset=key, keep="last")]
        order = [c for c in (date_col, "Code") if c in part.columns]
        part = part.sort_values(order, kind="stable").reset_index(drop=True)
        part.to_parquet(p, index=False, compression="zstd")
        written.append(p)

    return written
