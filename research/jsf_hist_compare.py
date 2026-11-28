#!/usr/bin/env python3
"""
日証金の過去ファイル（銘柄ごとの過去3年）を取り直したあと、取り直す前の控えと今のファイルを比べる
（docs/DATA_JSF.md「取り直し（2026-11）」）。

運用者の選択（2026-11-12）「全銘柄の過去を取り直す」「控えを残して上書き」。上書きは申込日・銘柄が重なれば
新しく取った値を残す（jsf_fetch.merge_store）。日証金が過去の値を直していなければ、重なる行は同じ値のはず。
違う行があれば、どの列で何行・いつの申込日かを数える。

出すのは件数・日付・割合だけ（値は出さない。日証金は私的利用の範囲で、Actions のログは公開）。

  1. 行: 控え・今・重なり（申込日×銘柄）、控えにしか無い行・今にしか無い行
  2. 列ごとの食い違い（重なる行）。控えの各銘柄の最終日（取った日の速報のことがある）は分けて数える
  3. 抜けた5日（申込日 10/23・10/26・10/27・10/30・11/2。毎日の確報が取れなかった日）が埋まったか
  4. 列の性質: 文字の列が銘柄の中で日によって変わるか（取った時点の値を全行に付けた列なら、どの銘柄でも
     1つの値しか無く、控えと今で違う）

  $ python3 research/jsf_hist_compare.py [--old 控え] [--new 今]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
BACKUP_NAME = "backup_jsf_hist_20261112.parquet"
KEY = ["申込日", "code"]
#: 毎日の確報が取れなかった申込日（docs/DATA_JSF.md「取り込みの欠け」）
GAP_DAYS = ("2026-10-23", "2026-10-26", "2026-10-27", "2026-10-30", "2026-11-02")
#: 比べない列（取った時刻などの記録）
SKIP = {"_fetched_at", "_last_modified"}
#: 性質を見る文字の列（銘柄の中で日によって変わるか）
TEXT_NATURE = ("貸借区分", "市場区分", "制限措置", "直近制限措置", "直近臨時措置", "直近特別措置", "直後基準日")


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["申込日"] = pd.to_datetime(df["申込日"])
    df["code"] = df["code"].astype(str)
    return df


def _norm_text(s: pd.Series) -> pd.Series:
    t = s.astype("string").fillna("").str.strip()
    return t.replace({"－": "", "-": "", "―": "", "ー": ""})


def same_values(a: pd.Series, b: pd.Series) -> pd.Series:
    """2つの列（同じ行の並び）が同じ値か。欠測どうしは同じ。数の列は誤差 1e-9 まで同じとみなす。"""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x = a.astype(float).to_numpy()
        y = b.astype(float).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=0.0, atol=1e-9, equal_nan=False)
        return pd.Series(both_nan | close)
    if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b):
        x = pd.to_datetime(a, errors="coerce")
        y = pd.to_datetime(b, errors="coerce")
        return pd.Series(((x == y) | (x.isna() & y.isna())).to_numpy())
    return pd.Series((_norm_text(a) == _norm_text(b)).to_numpy())


def compare(old: pd.DataFrame, new: pd.DataFrame) -> Dict:
    """控え（old）と今（new）を比べた件数の集計（値は入れない）。"""
    out: Dict = {}
    for name, df in (("old", old), ("new", new)):
        out[f"dup_{name}"] = int(df.duplicated(KEY).sum())
    old = old.drop_duplicates(KEY, keep="last")
    new = new.drop_duplicates(KEY, keep="last")
    out["rows_old"], out["rows_new"] = len(old), len(new)
    out["codes_old"], out["codes_new"] = int(old["code"].nunique()), int(new["code"].nunique())
    out["span_old"] = (old["申込日"].min(), old["申込日"].max())
    out["span_new"] = (new["申込日"].min(), new["申込日"].max())

    last_old = old.groupby("code")["申込日"].max().rename("_last_old")
    m = old.merge(new, on=KEY, how="outer", suffixes=("_o", "_n"), indicator=True)
    out["overlap"] = int((m["_merge"] == "both").sum())
    only_old = m[m["_merge"] == "left_only"]
    only_new = m[m["_merge"] == "right_only"]
    first_new = new.groupby("code")["申込日"].min()
    oo_first = only_old["code"].map(first_new)
    out["only_old"] = len(only_old)
    out["only_old_before_new_window"] = int((only_old["申込日"] < oo_first).sum())
    out["only_old_no_new_code"] = int(oo_first.isna().sum())
    out["only_old_inside"] = out["only_old"] - out["only_old_before_new_window"] - out["only_old_no_new_code"]
    on_last = only_new["code"].map(last_old)
    out["only_new"] = len(only_new)
    out["only_new_after_old_end"] = int((only_new["申込日"] > on_last).sum())
    out["only_new_new_code"] = int(on_last.isna().sum())
    out["only_new_inside"] = out["only_new"] - out["only_new_after_old_end"] - out["only_new_new_code"]

    both = m[m["_merge"] == "both"].copy()
    both["_is_last_old"] = both["申込日"].eq(both["code"].map(last_old))
    cols = [c for c in old.columns if c in new.columns and c not in KEY and c not in SKIP]
    per: List[Dict] = []
    for c in cols:
        same = same_values(both[f"{c}_o"], both[f"{c}_n"]).to_numpy()
        diff = ~same
        last = both["_is_last_old"].to_numpy()
        d_body = diff & ~last
        rec = {"col": c, "diff": int(d_body.sum()), "diff_last": int((diff & last).sum())}
        if rec["diff"]:
            dd = both.loc[d_body, ["申込日", "code"]]
            rec["dates"] = (dd["申込日"].min(), dd["申込日"].max())
            rec["codes"] = int(dd["code"].nunique())
        per.append(rec)
    out["columns"] = per
    out["overlap_last_rows"] = int(both["_is_last_old"].sum())

    gaps = []
    for d in GAP_DAYS:
        t = pd.Timestamp(d)
        gaps.append({"day": d, "old": int((old["申込日"] == t).sum()), "new": int((new["申込日"] == t).sum())})
    out["gaps"] = gaps

    nature = []
    for c in TEXT_NATURE:
        rec = {"col": c}
        for name, df in (("old", old), ("new", new)):
            if c not in df.columns:
                rec[name] = None
                continue
            t = _norm_text(df[c])
            k = t.groupby(df["code"]).nunique()
            rec[name] = {"codes": int(len(k)), "varies": int((k > 1).sum()),
                         "filled": float((t != "").mean())}
        nature.append(rec)
    out["nature"] = nature
    return out


def _d(x) -> str:
    return "-" if x is None or pd.isna(x) else pd.Timestamp(x).date().isoformat()


def report(r: Dict) -> None:
    print(f"[compare] 控え {r['rows_old']:,}行・{r['codes_old']:,}銘柄（申込日 {_d(r['span_old'][0])} 〜 "
          f"{_d(r['span_old'][1])}） / 今 {r['rows_new']:,}行・{r['codes_new']:,}銘柄（{_d(r['span_new'][0])} 〜 "
          f"{_d(r['span_new'][1])}） / 重なり {r['overlap']:,}行（うち控えの各銘柄の最終日 {r['overlap_last_rows']:,}行）")
    if r["dup_old"] or r["dup_new"]:
        print(f"[compare] 申込日×銘柄の重複: 控え {r['dup_old']:,} / 今 {r['dup_new']:,}")
    print(f"[compare] 控えにしか無い行 {r['only_old']:,}（今の窓より前 {r['only_old_before_new_window']:,}・"
          f"今に無い銘柄 {r['only_old_no_new_code']:,}・窓の中 {r['only_old_inside']:,}）")
    print(f"[compare] 今にしか無い行 {r['only_new']:,}（控えの最終日より後 {r['only_new_after_old_end']:,}・"
          f"控えに無い銘柄 {r['only_new_new_code']:,}・控えの期間の中 {r['only_new_inside']:,}）")
    bad = [c for c in r["columns"] if c["diff"]]
    print(f"[compare] 重なる行で値が違う列: {len(bad)} / {len(r['columns'])}列（控えの各銘柄の最終日を除く）")
    for c in r["columns"]:
        if c["diff"] or c["diff_last"]:
            extra = (f"・申込日 {_d(c['dates'][0])} 〜 {_d(c['dates'][1])}・{c['codes']:,}銘柄" if c["diff"] else "")
            print(f"  {c['col']}: 違う {c['diff']:,}行{extra} / 最終日の行で違う {c['diff_last']:,}行")
    for g in r["gaps"]:
        print(f"[compare] 抜けた日 {g['day']}: 控え {g['old']:,}銘柄 → 今 {g['new']:,}銘柄")
    for n in r["nature"]:
        parts = []
        for name in ("old", "new"):
            v = n[name]
            parts.append(f"{'控え' if name == 'old' else '今'} " + ("列なし" if v is None else
                         f"日によって変わる銘柄 {v['varies']:,} / {v['codes']:,}（値あり {v['filled'] * 100:.1f}%）"))
        print(f"[compare] {n['col']}: " + " / ".join(parts))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="日証金の過去ファイルの控えと今を比べる（件数だけ）")
    ap.add_argument("--old", default=os.path.join(DATA_DIR, BACKUP_NAME))
    ap.add_argument("--new", default=os.path.join(DATA_DIR, "jsf_hist.parquet"))
    a = ap.parse_args(argv)
    for p in (a.old, a.new):
        if not os.path.exists(p):
            print(f"[compare] ファイルが無い: {os.path.basename(p)}")
            return 1
    report(compare(_load(a.old), _load(a.new)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
