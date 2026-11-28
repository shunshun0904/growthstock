#!/usr/bin/env python3
"""
probe: 決算の出し直し（同じ期の後からの版）で、学習データの財務の列が「その日に分かっていた値」から外れる行の数と、
値の変わり方（検証だけ。本番は変えない）。

運用者の選択「まず影響を測る」（2026-10-29）。build_dataset.quarterize_panel は同じ（銘柄, 事業年度, 四半期）の開示を
最後の版だけ残し（訂正を反映）、attach_fins はその版の開示日から結合する。そのため元の版の開示日から出し直しの日までの
行は、その期の決算が無いものとして1つ前の期（1年以上前なら欠測）が付き、前年同期比などの基準も後から出た版の値になる。
予測の時点では元の版を使っているので、学習と予測で値が食い違う（2026-10-28 の学習と予測の一致チェックで見つけた）。

数え方
  今の作り   build_dataset.build と同じ（全部の決算で quarterize_panel → attach_fins → 1年以上古い決算は欠測）。
             データセットの値と一致することを先に確かめる（一致しない列は数えない）
  その日の版 行の日付までに開示された決算だけで quarterize_panel を作り、同じように結合した値。銘柄ごとに、決算の開示が
             増える日で区切って作る（同じ区間の行は同じ値になる）
  当たる行   ある期の最初の版の開示日 ≤ 行の日付 < その期の最後の版の開示日、となる期がその銘柄にある行
  確かめ     当たらない行から選んだ行をその日の版で作り直しても変わらないこと（変わる列は、全銘柄の中の順位のように
             銘柄ごとの作り直しでは測れない列なので数えない）
出すもの: 当たる行の数と割合（年ごと・正例の割合）、当たる行のうち値が変わった行と種類、列ごとの変わった行・欠測になった行・
値が戻った行・相対差の中央値（本番の列の組に入るかも）。銘柄名・値は出さない。

    exp=probe_fins_versions.py
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import sys
import tempfile
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as BD  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402

DATA_DIR = lab.DATA_DIR
#: build_dataset.build と同じ（決算が1年以上古い行は財務の列を欠測にする）
STALE_DAYS = 365
#: quarterize_panel と同じ四半期の番号
Q_MAP = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}
#: 当たらない行から選んで、その日の版で作り直しても変わらないかを確かめる行の数
N_CHECK = 500
SEED = 20261029
RTOL = 1e-9
KEYS = ("Code", "DiscDate", "CurFYSt")
#: 全銘柄の中の順位（progress_pct_point_in_time）。銘柄ごとの作り直しでは母集団が1銘柄になって測れないので数えない
CROSS_SECTIONAL = ("progress_pct",)


def statement_versions(fins: pd.DataFrame) -> pd.DataFrame:
    """quarterize_panel が使う決算（実績のある四半期の開示）の版: Code・CurFYSt・quarter・DiscDate。"""
    q = fins["CurPerType"].map(Q_MAP)
    has = fins[["Sales", "OP", "NP", "EPS"]].notna().any(axis=1)
    f = fins[q.notna() & has]
    return pd.DataFrame({"Code": f["Code"].astype(str).to_numpy(), "CurFYSt": f["CurFYSt"].astype(str).to_numpy(),
                         "quarter": q[f.index].astype(int).to_numpy(),
                         "DiscDate": pd.to_datetime(f["DiscDate"]).to_numpy()})


def reissue_windows(ver: pd.DataFrame) -> pd.DataFrame:
    """同じ期の版が別の日に出た期の [最初の版の開示日, 最後の版の開示日)。この間の行は、今の作りでは最後の版しか見えない。"""
    g = ver.groupby(["Code", "CurFYSt", "quarter"])["DiscDate"].agg(["min", "max"]).reset_index()
    g = g[g["max"] > g["min"]]
    return g.rename(columns={"min": "start", "max": "end"})[["Code", "start", "end"]].reset_index(drop=True)


def in_windows(rows: pd.DataFrame, win: pd.DataFrame) -> np.ndarray:
    """rows（Code・Date）のうち、同じ銘柄の窓 [start, end) に入る行。"""
    r = pd.DataFrame({"_i": np.arange(len(rows)), "Code": rows["Code"].astype(str).to_numpy(),
                      "Date": pd.to_datetime(rows["Date"]).to_numpy()})
    m = r.merge(win, on="Code", how="inner")
    hit = m.loc[(m["Date"] >= m["start"]) & (m["Date"] < m["end"]), "_i"].unique()
    out = np.zeros(len(rows), dtype=bool)
    out[hit] = True
    return out


def panel(fins: pd.DataFrame, adj: Optional[pd.DataFrame], data_dir: str = DATA_DIR) -> pd.DataFrame:
    """quarterize_panel（銘柄ごとに何度も呼ぶので、ログと警告は出さない）。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return BD.quarterize_panel(fins, data_dir, adj=adj)


def split_valuation(codes, data_dir: str, root: str) -> Dict[str, str]:
    """
    quarterize_panel が読むバリュエーション（data_dir の valuation_*.parquet。API の ROE）を、銘柄ごとの置き場に分ける。
    銘柄ごとに作り直すたびに全銘柄分（約90MB）を読み直すと1回数秒かかるので、その銘柄の行だけを置いた置き場を渡す
    （api_valuation は銘柄で結合するので、ほかの銘柄の行が無くても値は同じ）。quarterize_panel が data_dir を使うのは
    api_valuation だけ。置き場にバリュエーションが無ければ、どの銘柄も data_dir のまま。
    """
    codes = sorted(set(map(str, codes)))
    paths = sorted(glob.glob(os.path.join(data_dir, "valuation_[0-9]*.parquet")))
    if not paths:
        return {c: data_dir for c in codes}
    want = set(codes)
    parts = []
    for p in paths:
        v = pd.read_parquet(p)
        v = v[v["Code"].astype(str).isin(want)]
        if len(v):
            parts.append(v)
    v = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    by = dict(tuple(v.groupby(v["Code"].astype(str), sort=False))) if len(v) else {}
    out = {}
    for c in codes:
        d = os.path.join(root, c)
        os.makedirs(d, exist_ok=True)
        if c in by:
            by[c].to_parquet(os.path.join(d, "valuation_0.parquet"), index=False)
        out[c] = d
    return out


def attach(rows: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """build_dataset.build と同じ結合（attach_fins）と、1年以上古い決算の欠測。rows の順で返す。"""
    r = pd.DataFrame({"Code": rows["Code"].astype(str).to_numpy(), "Date": pd.to_datetime(rows["Date"]).to_numpy(),
                      "_i": np.arange(len(rows))})
    qq = q.copy()
    qq["Code"] = qq["Code"].astype(str)
    qq["DiscDate"] = pd.to_datetime(qq["DiscDate"])
    out = BD.attach_fins(r, qq)
    fin_cols = [c for c in q.columns if c not in KEYS]
    stale = (out["Date"] - out["DiscDate"]).dt.days > STALE_DAYS
    out.loc[stale, fin_cols] = np.nan
    return out.sort_values("_i").reset_index(drop=True)


def as_of_values(rows: pd.DataFrame, fins: pd.DataFrame, adj: Optional[pd.DataFrame],
                 data_dir: str = DATA_DIR, dirs: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    その日の版: 行ごとに、その日までに開示された決算だけで quarterize_panel を作り、build と同じく結合した値。
    銘柄ごとに、決算の開示が増える日で区切って作る（同じ区間の行は同じ）。rows の順で返す（決算の無い行は欠測）。
    dirs（銘柄 -> バリュエーションの置き場。split_valuation）があれば、銘柄ごとにその置き場を使う。
    """
    r = pd.DataFrame({"Code": rows["Code"].astype(str).to_numpy(), "Date": pd.to_datetime(rows["Date"]).to_numpy(),
                      "_i": np.arange(len(rows))})
    codes = set(r["Code"])
    f = fins[fins["Code"].astype(str).isin(codes)]
    f = f.assign(Code=f["Code"].astype(str), _dd=pd.to_datetime(f["DiscDate"]))
    by_code = dict(tuple(f.groupby("Code", sort=False)))
    adj_by = {}
    if adj is not None:
        a = adj[adj["Code"].astype(str).isin(codes)]
        adj_by = dict(tuple(a.assign(Code=a["Code"].astype(str)).groupby("Code", sort=False)))
    parts: List[pd.DataFrame] = [r.iloc[:0][["_i"]]]
    for code, rc in r.groupby("Code", sort=False):
        fc = by_code.get(code)
        if fc is None:
            parts.append(rc[["_i"]])
            continue
        dates = np.sort(fc["_dd"].unique())
        pos = np.searchsorted(dates, rc["Date"].to_numpy(), side="right") - 1
        for p, rr in rc.groupby(pos):
            if p < 0:
                parts.append(rr[["_i"]])
                continue
            q = panel(fc[fc["_dd"] <= dates[p]].drop(columns=["_dd"]), adj_by.get(code),
                      (dirs or {}).get(code, data_dir))
            got = attach(rr, q)
            got["_i"] = rr["_i"].to_numpy()
            parts.append(got)
    return pd.concat(parts, ignore_index=True).sort_values("_i").reset_index(drop=True)


def same_values(a: pd.Series, b: pd.Series) -> np.ndarray:
    """同じ値か（どちらも欠測なら同じ。数値は相対 1e-9 まで同じとみなす）。"""
    an, bn = a.isna().to_numpy(), b.isna().to_numpy()
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    num = (x.notna() | an).all() and (y.notna() | bn).all()
    if num:
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        eq = np.isclose(xv, yv, rtol=RTOL, atol=0.0)
    else:
        eq = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
    return (an & bn) | (~an & ~bn & eq)


def compare(cur: pd.DataFrame, pit: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """列ごとの変わり方（変わった行・今の作りで欠測になった行・値が戻った行・両方に値があって違う行の相対差の中央値）。"""
    out = []
    for c in cols:
        a, b = cur[c].reset_index(drop=True), pit[c].reset_index(drop=True)
        same = same_values(a, b)
        an, bn = a.isna().to_numpy(), b.isna().to_numpy()
        both = ~an & ~bn & ~same
        rel = np.nan
        x, y = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float), pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
        ok = both & np.isfinite(x) & np.isfinite(y)
        if ok.any():
            rel = float(np.median(np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-12)))
        out.append({"col": c, "changed": int((~same).sum()), "to_nan": int((an & ~bn).sum()),
                    "from_nan": int((~an & bn).sum()), "both": int(both.sum()), "rel_med": rel})
    return pd.DataFrame(out)


def kinds(cur: pd.DataFrame, pit: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """
    変わった行の種類: none（変わらない）/ lost（今の作りでは決算が付かない・1年以上古いが、その日の版では付く）/
    older（今の作りではより古い開示が付く）/ base（付く開示は同じ日だが値が違う。前の期などの基準が後の版）/ other。
    """
    changed = np.zeros(len(cur), dtype=bool)
    for c in cols:
        changed |= ~same_values(cur[c].reset_index(drop=True), pit[c].reset_index(drop=True))
    cur_has = cur[list(cols)].notna().any(axis=1).to_numpy()
    pit_has = pit[list(cols)].notna().any(axis=1).to_numpy()
    dc, dp = pd.to_datetime(cur["DiscDate"]).to_numpy(), pd.to_datetime(pit["DiscDate"]).to_numpy()
    out = np.full(len(cur), "other", dtype=object)
    out[~changed] = "none"
    out[changed & ~cur_has & pit_has] = "lost"
    older = changed & cur_has & pit_has & (dc < dp)
    out[older] = "older"
    out[changed & cur_has & pit_has & (dc == dp)] = "base"
    return pd.Series(out)


def load_adj(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """quarterize_panel に渡す分割の調整の倍率の元（build と同じ close_raw = C、close = AdjC、無ければ C）。"""
    parts = []
    for p in sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet"))):
        b = pd.read_parquet(p, columns=["Date", "Code", "C", "AdjC"])
        parts.append(pd.DataFrame({"Code": b["Code"].astype(str).to_numpy(), "Date": pd.to_datetime(b["Date"]).to_numpy(),
                                   "close_raw": pd.to_numeric(b["C"], errors="coerce").to_numpy(),
                                   "close": pd.to_numeric(b["AdjC"], errors="coerce")
                                   .fillna(pd.to_numeric(b["C"], errors="coerce")).to_numpy()}))
    if not parts:
        raise SystemExit(f"bars の parquet が {data_dir} に無い")
    return pd.concat(parts, ignore_index=True)


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.2f}%" if d else "—"


def main(argv=None) -> int:
    t0 = time.time()
    ds = pd.read_parquet(lab.DATASET)
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds = ds.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        fins = BD.load_parts("fins", DATA_DIR)
    adj = load_adj()
    ver = statement_versions(fins)
    win = reissue_windows(ver)
    n_groups = ver.groupby(["Code", "CurFYSt", "quarter"]).ngroups
    print("=" * 96)
    print(f"probe 決算の出し直しと学習データ: 学習データ {len(ds):,}件・{ds['Date'].nunique():,}日・"
          f"{ds['Date'].min().date()} 〜 {ds['Date'].max().date()} / 決算 {len(fins):,}行（実績のある四半期の開示 "
          f"{len(ver):,}件・{n_groups:,}期）。別の日に出し直された期 {len(win):,}（{_pct(len(win), n_groups)}）")
    print("=" * 96)

    # ---- 今の作り（build と同じ）と、データセットとの一致 ----
    q_all = panel(fins, adj)
    cur_all = attach(ds, q_all)
    cols_q = [c for c in q_all.columns if c not in KEYS]
    in_ds = [c for c in cols_q if c in ds.columns]
    match = {c: float(same_values(cur_all[c], ds[c]).mean()) for c in in_ds}
    good = [c for c in in_ds if match[c] >= 0.999]
    bad = sorted((c for c in in_ds if match[c] < 0.999), key=lambda c: match[c])
    print(f"[確かめ1] 今の作りの再現: 決算から作る列 {len(cols_q)}列のうちデータセットにある {len(in_ds)}列。"
          f"全行の 99.9% 以上で一致 {len(good)}列" + (f" / 一致しない（数えない）{len(bad)}列: "
                                              + ", ".join(f"{c} {match[c] * 100:.1f}%" for c in bad[:12])
                                              if bad else ""))

    # ---- 当たる行 ----
    hit = in_windows(ds, win)
    n_hit = int(hit.sum())
    lab_ok = ds["label"].notna()
    pos_all = float(ds.loc[lab_ok, "label"].mean())
    pos_hit = float(ds.loc[hit & lab_ok, "label"].mean()) if (hit & lab_ok).any() else float("nan")
    by_year = pd.Series(hit).groupby(ds["Date"].dt.year.to_numpy()).agg(["sum", "size"])
    print(f"[当たる行] {n_hit:,}件（{_pct(n_hit, len(ds))}）。正例の割合 当たる行 {pos_hit * 100:.1f}% / 全体 "
          f"{pos_all * 100:.1f}%")
    print("  年ごと: " + " / ".join(f"{y} {int(r['sum'])}件（{_pct(int(r['sum']), int(r['size']))}）"
                                    for y, r in by_year.iterrows()))

    # ---- 確かめ2: 当たらない行は、その日の版で作り直しても変わらない ----
    cols = [c for c in good if c not in CROSS_SECTIONAL]
    rng = np.random.default_rng(SEED)
    rest = np.flatnonzero(~hit)
    pick = np.sort(rng.choice(rest, size=min(N_CHECK, len(rest)), replace=False))
    idx = np.flatnonzero(hit)
    tmp = tempfile.TemporaryDirectory()
    t1 = time.time()
    dirs = split_valuation(ds["Code"].iloc[np.concatenate([pick, idx])], DATA_DIR, tmp.name)
    print(f"[準備] バリュエーションを銘柄ごとの置き場に分けた: {len(dirs):,}銘柄（{time.time() - t1:.0f}秒）")
    t1 = time.time()
    chk_cur = cur_all.iloc[pick].reset_index(drop=True)
    chk_pit = as_of_values(ds.iloc[pick], fins, adj, dirs=dirs).reindex(columns=chk_cur.columns)
    kd_chk = kinds(chk_cur, chk_pit, cols)
    n_out = int((kd_chk != "none").sum())
    cmp_chk = compare(chk_cur, chk_pit, cols)
    top_out = cmp_chk.loc[cmp_chk["changed"] > 0].sort_values("changed", ascending=False)["col"].tolist()
    print(f"[確かめ2] 当たらない行から {len(pick)}件をその日の版で作り直した（{time.time() - t1:.0f}秒。"
          f"{', '.join(CROSS_SECTIONAL)} は全銘柄の中の順位なので数えない）: 変わった行 {n_out}件"
          + (f"（列: {', '.join(top_out[:8])}。当たる行の決め方の外で変わる行がある。当たらない行 {len(rest):,}件に"
             f"引き直すと約 {n_out / len(pick) * len(rest):,.0f}件）" if n_out else "（当たる行の決め方で漏れは無い）"))

    # ---- 当たる行の変わり方 ----
    t2 = time.time()
    cur = cur_all.iloc[idx].reset_index(drop=True)
    pit = as_of_values(ds.iloc[idx], fins, adj, dirs=dirs).reindex(columns=cur.columns)
    tmp.cleanup()
    kd = kinds(cur, pit, cols)
    vc = kd.value_counts()
    n_changed = int((kd != "none").sum())
    print(f"[変わり方] 当たる行 {len(idx):,}件をその日の版で作り直した（{time.time() - t2:.0f}秒）: どれかの列が変わった行 "
          f"{n_changed:,}件（当たる行の {_pct(n_changed, len(idx))}・学習データの {_pct(n_changed, len(ds))}）")
    names = {"lost": "今の作りでは決算が付かない（1年以上古い・無い）が、その日の版では付く",
             "older": "今の作りではより古い開示が付く",
             "base": "付く開示は同じだが値が違う（前の期・前年同期の基準が後の版）", "other": "その他"}
    for k in ("lost", "older", "base", "other"):
        print(f"  {names[k]}: {int(vc.get(k, 0)):,}件")
    prod = set(F.columns(F.DEFAULT_PRESET))
    cmp_hit = compare(cur, pit, cols).sort_values("changed", ascending=False)
    in_prod = [c for c in cols if c in prod]
    print(f"[列ごと] 数える {len(cols)}列のうち本番の列の組（{F.DEFAULT_PRESET}・{len(prod)}列）に入る {len(in_prod)}列。"
          "変わった行の多い順（当たる行のうち）")
    print(f"  {'列':<28}{'本番':>4}{'変わった':>9}{'欠測に':>8}{'値が戻る':>9}{'値が違う':>9}{'相対差の中央値':>15}")
    for _, r in cmp_hit.head(30).iterrows():
        if r["changed"] == 0:
            break
        rel = f"{r['rel_med'] * 100:.1f}%" if np.isfinite(r["rel_med"]) else "—"
        print(f"  {r['col']:<28}{'○' if r['col'] in prod else '':>4}{int(r['changed']):>9,}{int(r['to_nan']):>8,}"
              f"{int(r['from_nan']):>9,}{int(r['both']):>9,}{rel:>15}")
    any_prod = np.zeros(len(cur), dtype=bool)
    for c in in_prod:
        any_prod |= ~same_values(cur[c], pit[c])
    print(f"[全体] 本番の列の組に入る列のどれかが変わる行: {int(any_prod.sum()):,}件（学習データの "
          f"{_pct(int(any_prod.sum()), len(ds))}）。時価総額で割る列など、ここで変わった列から後で作る列も同じ行で変わる")
    print(f"[done] {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
