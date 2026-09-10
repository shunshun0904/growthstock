#!/usr/bin/env python3
"""
指数コードが「どの業種の指数か」を実測で同定する。

なぜ必要か:
  /indices/bars/daily?date=... はその日の全指数（実測79件）を返すが、
  レスポンスに**名称が無い**（列は Code / Date / O / H / L / C）。
  既知は TOPIX = 0000 だけで、残りが何かは分からない。

  コード帯の件数（0040〜0060 が33件、0080〜0090 が17件）は
  東証33業種／17業種の構成と一致するが、それは状況証拠にすぎない。
  「たぶん業種指数だろう」で特徴量を作ると、間違った業種を
  当てたまま学習することになる。

やること:
  手元の株価データから業種ごとの**時価総額加重リターン**を組み、
  各指数の日次リターンと相関を取る。最も相関の高い業種を割り当てる。

  同定できたと言えるのは、1位の相関が高く、かつ2位との差が
  はっきりしているときだけ。両方を出して判断できるようにする。
  対照として「全銘柄」も入れる。0000（TOPIX）が全銘柄に最も
  近くならなければ、この手続き自体が信用できない。

  相関は**日次リターン**で取る。水準どうしだと、上昇トレンドを
  共有しているだけで高い相関が出る。

出力: research/index_mapping.json と docs/INDEX_MAPPING.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jquants_data_fetcher import (  # noqa: E402
    JQuantsClient, JQuantsError, resolve_api_key,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "_data")
OUT_JSON = os.path.join(HERE, "index_mapping.json")
OUT_MD = os.path.join(ROOT, "docs", "INDEX_MAPPING.md")

EARLIEST_DATE = "2016-10-01"
#: 指数の一覧を引く基準日。この日に存在した指数がすべて返る
INDEX_ASOF = "2024-05-15"

#: 同定できたとみなす条件。
#: 相関だけ高くても、2位と僅差なら「どちらか分からない」ということ。
MIN_CORR = 0.80
MIN_GAP = 0.05

#: 業種の系列を組むときに残す市場区分（**名称**の断片）。
#:
#: master_hist の Mkt は数値コード（実測で 101〜113 の8種）であって名称ではない。
#: 最初これを名称と決めつけて絞り込み、1,000万行が0行になった。
#: 名称の列（MktNm）がある場合だけ使い、無ければ絞らない。
#:
#: 絞らなくても実害は小さい。ETF/ETN には決算が無いので株数が付かず、
#: 時価総額の重みが作れずに落ちる。
DOMESTIC_MARKETS = ("プライム", "スタンダード", "グロース", "市場第一部",
                    "市場第二部", "マザーズ", "JASDAQ")
#: 市場区分の名称が入っている可能性のある列。実測で見つかったものを使う
MARKET_NAME_COLS = ("MktNm", "MarketCodeName")

#: 業種コード -> 名称 / 銘柄数。master_hist が S33Nm / S17Nm を持っていたので、
#: コードだけで報告せずに済む（実測: master_hist の項目一覧）。
SECTOR_NAMES: Dict[str, Dict[str, str]] = {}
SECTOR_COUNTS: Dict[str, Dict[str, Optional[int]]] = {}


def label(axis: str, code: str) -> str:
    """業種コードに名称を添える。名称が無ければコードだけ返す。"""
    if code == "ALL":
        return "全銘柄"
    return SECTOR_NAMES.get(axis, {}).get(code, code)


# --------------------------------------------------------------------------- #
# 指数
# --------------------------------------------------------------------------- #

#: 取得した指数リターンの置き場。解析でつまずいても API を叩き直さないため。
#: 「取得結果は解析より先に保存する」は probe_fins_fields.py で一度学んだこと。
INDEX_CACHE = os.path.join(DATA_DIR, "index_returns.parquet")


def fetch_index_returns(client: JQuantsClient, end: str) -> pd.DataFrame:
    """全指数の日次リターン（列=コード、行=日付）。"""
    if os.path.exists(INDEX_CACHE):
        cached = pd.read_parquet(INDEX_CACHE)
        cached.index = pd.to_datetime(cached.index)
        print(f"[index] キャッシュを使用: {INDEX_CACHE} "
              f"({cached.shape[1]}系列 / {cached.shape[0]}日)")
        return cached
    codes = sorted({str(r.get("Code"))
                    for r in client.get_paginated(
                        "/indices/bars/daily", {"date": INDEX_ASOF})})
    print(f"[index] {INDEX_ASOF} に存在した指数: {len(codes)}件")
    series: Dict[str, pd.Series] = {}
    for i, code in enumerate(codes, 1):
        try:
            rows = client.get_paginated(
                "/indices/bars/daily",
                {"code": code, "from": EARLIEST_DATE, "to": end})
        except JQuantsError as exc:
            print(f"  {code}: 取得不可 {str(exc)[:80]}")
            continue
        if not rows:
            continue
        d = pd.DataFrame.from_records(rows)[["Date", "C"]]
        d["Date"] = pd.to_datetime(d["Date"])
        d["C"] = pd.to_numeric(d["C"], errors="coerce")
        d = d.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
        series[code] = d.set_index("Date")["C"].pct_change()
        if i % 20 == 0 or i == len(codes):
            print(f"  {i}/{len(codes)} 取得")
    out = pd.DataFrame(series).sort_index()
    print(f"[index] {out.shape[1]}系列 / {out.shape[0]}日 "
          f"{out.index.min().date()}〜{out.index.max().date()}")
    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_parquet(INDEX_CACHE)
    print(f"[index] 保存: {INDEX_CACHE}")
    return out


# --------------------------------------------------------------------------- #
# 業種の系列を手元のデータから組む
# --------------------------------------------------------------------------- #

def _load(prefix: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, f"{prefix}_*.parquet")))
    if not paths:
        raise SystemExit(f"{prefix}_*.parquet が {DATA_DIR} にありません")
    frames = [pd.read_parquet(p, columns=cols) if cols else pd.read_parquet(p)
              for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {prefix}: {len(df):,}行 ({len(paths)}ファイル)")
    return df


def sector_returns(axis: str) -> pd.DataFrame:
    """
    業種ごとの時価総額加重リターン（列=業種コード、行=日付）。

    重みは**前日**の時価総額を使う。当日の時価総額で重み付けすると、
    その日に上がった銘柄の重みが上がり、指数のリターンが過大になる。

    時価総額は未調整終値 × 開示時点の株数。調整後終値を使うと、
    後年の分割ぶんだけ過去の時価総額が小さく出る。
    リターンのほうは分割をまたぐので調整後（AdjC）で計算する。

    業種は月次スナップショット（master_hist）から時点別に当てる。
    最新のマスタを過去に当てると、2022年4月の東証再編や業種変更を
    またいだところで別の業種の銘柄が混ざる。
    """
    bars = _load("bars", ["Date", "Code", "C", "AdjC"])
    bars["Date"] = pd.to_datetime(bars["Date"])
    bars["Code"] = bars["Code"].astype(str)
    for c in ("C", "AdjC"):
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars = (bars.dropna(subset=["Date", "Code"])
            .sort_values(["Code", "Date"]).reset_index(drop=True))

    # 株数（開示時点のもの）。時価総額の重みに要る
    fins = _load("fins")
    share_col = next((c for c in ("ShOutFY", "ShOut", "SharesOut")
                      if c in fins.columns), None)
    if share_col is None:
        raise SystemExit(f"[fatal] 株数の項目がありません。列: {sorted(fins.columns)[:40]}")
    f = fins[["Code", "DiscDate", share_col]].copy()
    f["Code"] = f["Code"].astype(str)
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f[share_col] = pd.to_numeric(f[share_col], errors="coerce")
    f = (f.dropna().sort_values("DiscDate")
         .drop_duplicates(["Code", "DiscDate"], keep="last"))
    bars = pd.merge_asof(
        bars.sort_values("Date"), f.rename(columns={share_col: "shares"}),
        left_on="Date", right_on="DiscDate", by="Code", direction="backward")

    # 業種（時点別）
    mh = _load("master_hist")
    mh["Date"] = pd.to_datetime(mh["Date"])
    mh["Code"] = mh["Code"].astype(str)
    print(f"[sector] master_hist の項目: {sorted(mh.columns)}")
    name_col = next((c for c in MARKET_NAME_COLS if c in mh.columns), None)
    # 業種の名称。master_hist が持っているので、コードだけで報告せずに済む
    sector_name_col = f"{axis}Nm" if f"{axis}Nm" in mh.columns else None
    keep = [c for c in (axis, name_col, sector_name_col) if c and c in mh.columns]
    if axis not in keep:
        raise SystemExit(f"[fatal] master_hist に {axis} がありません。列: {sorted(mh.columns)}")
    mh = (mh[["Date", "Code"] + keep].dropna(subset=["Date", "Code"])
          .sort_values("Date").drop_duplicates(["Date", "Code"], keep="last"))
    bars = pd.merge_asof(bars.sort_values("Date"), mh,
                         on="Date", by="Code", direction="backward")

    if name_col and name_col in bars.columns:
        before = len(bars)
        mk = bars[name_col].astype(str)
        kept = bars[mk.apply(lambda m: any(k in m for k in DOMESTIC_MARKETS))]
        # 絞った結果が空なら、絞り方が間違っている。
        # 黙って0行のまま進むと、後段が意味不明な失敗をする
        if kept.empty:
            print(f"[warn] {name_col} で絞ると0行になった（値の例: "
                  f"{sorted(mk.unique())[:5]}）。絞らずに進む")
        else:
            bars = kept
            print(f"[sector] 内国株に限定: {before:,} -> {len(bars):,}行")
    else:
        print("[sector] 市場区分の名称が無いので絞らない"
              "（ETF/ETN は決算が無く株数が付かないので、重みの段階で落ちる）")

    bars = bars.dropna(subset=[axis])
    bars = bars.sort_values(["Code", "Date"])
    g = bars.groupby("Code", sort=False)
    # fill_method=None が要る。既定の ffill は欠測日を前日値で埋めてから
    # 変化率を取るので、値が無い日をまたいだところで 0% のリターンが
    # 作られ、系列がなまる
    bars["ret"] = g["AdjC"].pct_change(fill_method=None)
    # 前日の時価総額を重みにする
    bars["w"] = (g["C"].shift(1) * g["shares"].shift(1))
    bars = bars.dropna(subset=["ret", "w"])
    bars = bars[bars["w"] > 0]

    if bars.empty:
        raise SystemExit(f"[fatal] {axis} の系列を作れる行が残っていません。"
                         "業種・株数・リターンのどれかが全滅している")

    bars["wr"] = bars["w"] * bars["ret"]
    grp = bars.groupby(["Date", axis], sort=True)
    agg = grp[["wr", "w"]].sum()
    wide = (agg["wr"] / agg["w"]).unstack(axis).sort_index()
    wide.columns = [str(c) for c in wide.columns]

    # 対照として全銘柄（=TOPIX に対応するはず）も作る
    allg = bars.groupby("Date", sort=True)[["wr", "w"]].sum()
    wide["ALL"] = (allg["wr"] / allg["w"])

    if sector_name_col:
        names = (bars.dropna(subset=[sector_name_col])
                 .groupby(axis)[sector_name_col].agg(
                     lambda t: t.value_counts().index[0]))
        SECTOR_NAMES[axis] = {str(k): str(v) for k, v in names.items()}

    n = bars.groupby(["Date", axis], sort=True).size().unstack(axis)
    med = n.stack().median()
    print(f"[sector] {axis}: {wide.shape[1] - 1}業種 / {wide.shape[0]}日 "
          f"（1業種あたりの銘柄数 中央値 "
          f"{'—' if pd.isna(med) else int(med)}）")
    # 銘柄数が極端に少ない業種は、指数と対応させても偶然の相関になりやすい。
    # 相関を見る前に気づけるよう、全業種ぶん出す
    SECTOR_COUNTS[axis] = {str(k): (None if pd.isna(v) else int(v))
                           for k, v in n.median().items()}
    thin = n.median().sort_values().head(8)
    print("[sector] 銘柄数が少ない業種: " + " / ".join(
        f"{k}({label(axis, str(k))}):{'—' if pd.isna(v) else int(v)}"
        for k, v in thin.items()))
    return wide


# --------------------------------------------------------------------------- #
# 突き合わせ
# --------------------------------------------------------------------------- #

def match(idx: pd.DataFrame, sec: pd.DataFrame, min_days: int = 500) -> List[dict]:
    """各指数コードについて、最も相関の高い業種と2位を出す。"""
    common = idx.index.intersection(sec.index)
    print(f"[match] 共通の日付: {len(common):,}日")
    a, b = idx.loc[common], sec.loc[common]
    out = []
    for code in a.columns:
        x = a[code]
        ok = x.notna()
        if int(ok.sum()) < min_days:
            out.append({"index": code, "days": int(ok.sum()), "note": "日数不足"})
            continue
        corr = b.loc[ok].corrwith(x[ok]).dropna().sort_values(ascending=False)
        if corr.empty:
            out.append({"index": code, "days": int(ok.sum()), "note": "相関を計算できない"})
            continue
        best = corr.index[0]
        second = corr.index[1] if len(corr) > 1 else None
        gap = float(corr.iloc[0] - corr.iloc[1]) if len(corr) > 1 else float("nan")
        out.append({
            "index": code,
            "days": int(ok.sum()),
            "best": str(best),
            "corr": round(float(corr.iloc[0]), 4),
            "second": None if second is None else str(second),
            "secondCorr": None if second is None else round(float(corr.iloc[1]), 4),
            "gap": round(gap, 4),
            "confident": bool(corr.iloc[0] >= MIN_CORR and gap >= MIN_GAP),
        })
    return out


#: 業種として扱わないコード。実測で「その他」に相当し、銘柄数が1しかない
NON_SECTOR_CODES = {"9999", "99"}


def _hex_runs(codes: List[str]) -> List[List[str]]:
    """4桁16進として連続しているコードの並びを取り出す。"""
    vals = []
    for c in codes:
        try:
            vals.append((int(c, 16), c))
        except ValueError:
            continue
    vals.sort()
    runs: List[List[str]] = []
    for v, c in vals:
        if runs and int(runs[-1][-1], 16) + 1 == v:
            runs[-1].append(c)
        else:
            runs.append([c])
    return runs


def test_offset(idx: pd.DataFrame, sec: pd.DataFrame, axis: str) -> dict:
    """
    「指数コードの並び順が業種コードの並び順と一致する」という仮説を検証する。

    最初の実測で、相関がはっきり出た4件がすべて
    「指数コードの16進オフセット = 業種コードの昇順の位置」に乗っていた。
    偶然そうなる確率は低いが、乗っている4件だけを見て決めるのは
    後知恵の当てはめになる。

    そこで仮説を先に固定し、**全対**について
      ・仮説が指す業種との相関
      ・その業種が何位か（1位なら仮説どおり）
    を出す。何位かまで出せば、相関が低い業種でも仮説が当たっているのか
    外れているのかを分けて読める。

    対象は「業種の数とちょうど同じ長さの連番」。長さが違う並びに
    当てはめると、どこを起点にしても何かしら当たってしまう。
    """
    sectors = [c for c in sec.columns
               if c != "ALL" and c not in NON_SECTOR_CODES]
    sectors.sort()
    n = len(sectors)
    runs = [r for r in _hex_runs(list(idx.columns)) if len(r) == n]
    print(f"[offset] {axis}: 業種 {n}個 / 長さの一致する連番 {len(runs)}本")
    out = {"axis": axis, "nSectors": n, "runs": []}
    common = idx.index.intersection(sec.index)
    a, b = idx.loc[common], sec.loc[common]
    for run in runs:
        pairs, rank1 = [], 0
        for k, code in enumerate(run):
            x = a[code]
            ok = x.notna()
            corr = b.loc[ok].corrwith(x[ok]).dropna().sort_values(ascending=False)
            want = sectors[k]
            if want not in corr.index:
                pairs.append({"index": code, "sector": want, "corr": None,
                              "rank": None})
                continue
            rank = int(list(corr.index).index(want)) + 1
            rank1 += rank == 1
            pairs.append({
                "index": code, "sector": want, "sectorName": label(axis, want),
                "corr": round(float(corr[want]), 4), "rank": rank,
                "nStocks": SECTOR_COUNTS.get(axis, {}).get(want),
            })
        got = [p["corr"] for p in pairs if p["corr"] is not None]
        med = float(np.median(got)) if got else float("nan")
        print(f"[offset] {run[0]}〜{run[-1]}: 1位一致 {rank1}/{len(run)} "
              f"/ 相関の中央値 {med:.4f}")
        out["runs"].append({"from": run[0], "to": run[-1],
                            "rank1": rank1, "n": len(run),
                            "medianCorr": round(med, 4), "pairs": pairs})
    return out


def write_md(res: dict) -> None:
    L: List[str] = []
    L.append("# 指数コードの同定（実測）")
    L.append("")
    L.append("`research/identify_indices.py` の出力。**実測値のみ**を記載する。")
    L.append("")
    L.append("`/indices/bars/daily` は名称を返さない（列は Code / Date / O / H / L / C）。")
    L.append("そこで手元の株価データから業種ごとの時価総額加重リターンを組み、")
    L.append("各指数の**日次リターン**との相関で対応を決めた。")
    L.append("水準どうしの相関は上昇トレンドを共有しているだけで高く出るので使わない。")
    L.append("")
    L.append(f"- 実測日時: {res['measuredAt']}")
    L.append(f"- 同定の条件: 相関 >= {MIN_CORR} かつ 2位との差 >= {MIN_GAP}")
    L.append("")
    L.append("## 対照 — TOPIX(0000) が全銘柄に当たるか")
    L.append("")
    ctrl = res.get("control")
    if ctrl:
        L.append(f"`0000` の最良一致: **{ctrl['best']}**（相関 {ctrl['corr']}）"
                 f" / 2位 {ctrl['second']}（{ctrl['secondCorr']}）")
        L.append("")
        L.append("これが `ALL`（全銘柄の時価総額加重）にならなければ、"
                 "業種の組み方そのものが間違っている。")
    else:
        L.append("`0000` を測れなかった。")
    L.append("")

    for axis, label_axis in (("S33", "33業種"), ("S17", "17業種")):
        rows = res["axes"].get(axis, [])
        ok = [r for r in rows if r.get("confident")]
        L.append(f"## {label_axis}（`{axis}`）との突き合わせ")
        L.append("")
        L.append(f"指数 {len(rows)}件のうち、条件を満たしたのは **{len(ok)}件**。")
        L.append("")
        cnts = (res.get("sectors") or {}).get(axis, {})
        thin = sorted(((v.get("stocks") or 0), k, v.get("name", k))
                      for k, v in cnts.items() if k != "ALL")[:6]
        if thin:
            L.append("銘柄数が少ない業種は、指数と対応させても偶然の相関になりやすい: "
                     + " / ".join(f"{nm}(`{k}`) {n}銘柄" for n, k, nm in thin))
            L.append("")
        L.append("| 指数 | 最良一致 | 相関 | 2位 | 相関 | 差 | 判定 |")
        L.append("| --- | --- | ---: | --- | ---: | ---: | :---: |")
        for r in sorted(rows, key=lambda x: -(x.get("corr") or -1)):
            if "best" not in r:
                L.append(f"| `{r['index']}` | — | — | — | — | — | {r.get('note','')} |")
                continue
            bn = r.get("bestName") or r["best"]
            sn = r.get("secondName") or r.get("second")
            L.append(f"| `{r['index']}` | {bn} (`{r['best']}`) | {r['corr']} | "
                     f"{sn} (`{r['second']}`) | {r['secondCorr']} | {r['gap']} | "
                     f"{'○' if r['confident'] else '×'} |")
        L.append("")

        off = (res.get("offset") or {}).get(axis)
        if off and off.get("runs"):
            L.append(f"### 並び順の仮説（{label_axis})")
            L.append("")
            L.append("「指数コードの16進の並び = 業種コードの昇順の並び」を先に固定し、")
            L.append("**全対**について仮説が指す業種との相関と、その業種が何位かを出す。")
            L.append("相関が低い業種でも、1位なら仮説どおりに当たっている。")
            L.append("")
            for run in off["runs"]:
                L.append(f"`{run['from']}`〜`{run['to']}`: "
                         f"**1位一致 {run['rank1']}/{run['n']}** "
                         f"/ 相関の中央値 {run['medianCorr']}")
                L.append("")
                L.append("| 指数 | 仮説が指す業種 | 相関 | 順位 | 銘柄数 |")
                L.append("| --- | --- | ---: | ---: | ---: |")
                for pr in run["pairs"]:
                    nm = pr.get("sectorName") or pr["sector"]
                    L.append(f"| `{pr['index']}` | {nm} (`{pr['sector']}`) | "
                             f"{pr.get('corr', '—')} | {pr.get('rank', '—')} | "
                             f"{pr.get('nStocks', '—')} |")
                L.append("")

    L.append("## 読み方")
    L.append("")
    L.append("- 相関が高くても2位と僅差なら同定できていない。業種どうしは")
    L.append("  もともと相関するので、1位だけを見て決めてはいけない")
    L.append("- 業種コードの意味は `research/build_dataset.py` の `encode_category` と")
    L.append("  `master_hist` の生の値を参照。ここでは API が返した値をそのまま使う")
    L.append("- 手元の系列は東証の指数と算出方法が違う（浮動株調整をしていない、")
    L.append("  構成銘柄の出入りを再現していない）。完全一致は期待できない")
    L.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[write] {OUT_MD}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="指数コードを業種と突き合わせて同定する")
    ap.add_argument("--axes", nargs="+", default=["S33", "S17"])
    args = ap.parse_args(argv)

    end = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    client = JQuantsClient(resolve_api_key(), pause=0.05)

    print("[1] 指数の日次リターンを取得")
    idx = fetch_index_returns(client, end)

    res = {"measuredAt": dt.datetime.now(dt.timezone.utc).isoformat(),
           "minCorr": MIN_CORR, "minGap": MIN_GAP, "axes": {}}
    for axis in args.axes:
        print(f"\n[2] {axis} の業種別リターンを組む")
        sec = sector_returns(axis)
        print(f"[3] {axis} と突き合わせ")
        rows = match(idx, sec)
        res["axes"][axis] = rows
        for r in rows:
            if r.get("best"):
                r["bestName"] = label(axis, r["best"])
                r["secondName"] = label(axis, r["second"]) if r.get("second") else None
        print(f"[4] {axis}: 並び順の仮説を検証")
        res.setdefault("offset", {})[axis] = test_offset(idx, sec, axis)
        res.setdefault("sectors", {})[axis] = {
            c: {"name": label(axis, c), "stocks": SECTOR_COUNTS.get(axis, {}).get(c)}
            for c in sorted(sec.columns)}
        hit = [r for r in rows if r.get("confident")]
        print(f"  条件を満たした指数: {len(hit)}/{len(rows)}件")
        for r in sorted(hit, key=lambda x: -x["corr"])[:10]:
            print(f"    {r['index']} -> {r['best']:>6}  r={r['corr']:.4f} "
                  f"(2位 {r['second']} {r['secondCorr']:.4f} / 差 {r['gap']:.4f})")
        if axis == "S33":
            res["control"] = next((r for r in rows if r["index"] == "0000"), None)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\n[write] {OUT_JSON}")
    write_md(res)

    ctrl = res.get("control")
    if ctrl and ctrl.get("best") != "ALL":
        print(f"[warn] 対照が通っていない。0000 の最良一致は {ctrl.get('best')} "
              f"（ALL であるべき）。業種の組み方を疑うこと")
    return 0


if __name__ == "__main__":
    sys.exit(main())
