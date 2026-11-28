#!/usr/bin/env python3
"""
probe: 窓に決算が入る行が上に抜けやすいのは「方向」（決算が平均して良い）か「分散」（上下どちらにも大きく動く）か。
モデルは決算前の行を高く付けているか。窓の中の決算は実際に上方修正だったか。

運用者の問い（2026-10-17）「予測時点では、数日後に発表される決算が基本的には好業績で、+要因だと、結果的に
それを予測しているような感じですよね？」に答える。probe_label_earn（run 39012793757）の続き。

§1 方向か分散か … 窓（t+1〜t+20 営業日）の終値の最大・最小・最後（基準は翌営業日の寄り）を日足から作り、
   決算短信の開示が窓に入る行と入らない行で、上に抜ける割合・下に抜ける割合（同じ kσ）・20日目の収益の分布を比べる。
   +10% の指値・20日目の終値の規則（live_track.forward）の収益も比べる
§2 モデルの見方 … 規則の3本（lgbm_k15 / xgb_k15 / cat_k15）の OOF の百分位を、決算までの日数（days_to_earn、その日
   までに公表済みの予定）の区分ごとに。規則（3本 95 以上）を通る行のうち決算が窓に入る割合、通った行の正例率と収益を
   決算の有無で。lgbm_k15 の TreeSHAP で days_to_earn の寄与の大きさ（順位）と向き
§3 窓の中の決算の中身 … 窓に入る最初の決算短信で、通期の営業利益予想（FOP）が同じ事業年度の直前の開示から上方・
   据え置き・下方のどれか（比べられない = 新年度の最初の予想か予想なし）。ラベル別の割合と、区分ごとの正例率・収益・
   開示をまたぐ2日の値動き

出すのは件数・割合・平均・分位だけ。銘柄名・値は出さない（Actions のログは公開）。

  $ python3 research/exp/probe_earn_direction.py [--data-dir research/_data]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import probe_label_earn as P  # noqa: E402
import trading_calendar as TC  # noqa: E402

DATA_DIR = lab.DATA_DIR
RULE = tuple(LT.K15)                    # 集計したときの規則（1.5σ の3本。2026-10-23 からは 1.2σ の3本なので記録どおり固定）
LINE = float(LT.AGREE_PCT)
AHEAD_BINS: Sequence[Tuple[str, float, float]] = (
    ("0〜7日", 0, 7), ("8〜14日", 7, 14), ("15〜28日（窓の中）", 14, 28), ("29〜60日", 28, 60), ("61日〜", 60, 1e9))
REV_CATS = ("上方", "据え置き", "下方", "比べられない")


# --------------------------------------------------------------------------- #
# 日足から窓の値動き
# --------------------------------------------------------------------------- #

def load_bars(data_dir: str, codes: set) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit(f"bars_*.parquet が {data_dir} にありません")
    parts = []
    for p in paths:
        d = pd.read_parquet(p, columns=["Date", "Code", "AdjO", "AdjH", "AdjC"])
        d["Code"] = d["Code"].astype(str).str.strip()
        parts.append(d[d["Code"].isin(codes)])
    b = pd.concat(parts, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    for c in ("AdjO", "AdjH", "AdjC"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.sort_values(["Code", "Date"]).drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)


def window_stats(bars: pd.DataFrame, keys: pd.DataFrame, h: int,
                 react_dates: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    keys（Code, Date）ごとに、翌営業日の寄り（entry）と窓 t+1〜t+h の終値の最大・最小・最後。
    react_dates（開示日、NaT 可）を渡すと、開示日をまたぐ2日の値動き（開示日の前の営業日の終値 → 次の営業日の終値）も付ける。
    行の順序は keys のまま。窓が足りない行は NaN。
    """
    n = len(keys)
    out = {k: np.full(n, np.nan) for k in ("entry", "w_max", "w_min", "w_end", "react")}
    kd = keys["Date"].to_numpy(dtype="datetime64[ns]")
    rd = (np.asarray(react_dates, dtype="datetime64[ns]") if react_dates is not None
          else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"))
    by_code: Dict[str, List[int]] = {}
    for i, c in enumerate(keys["Code"].to_numpy()):
        by_code.setdefault(c, []).append(i)
    for code, g in bars.groupby("Code", sort=False):
        rows = by_code.get(code)
        if not rows:
            continue
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        o = g["AdjO"].to_numpy(dtype=float)
        c = g["AdjC"].to_numpy(dtype=float)
        m = len(dates)
        for i in rows:
            j = int(np.searchsorted(dates, kd[i]))
            if j >= m or dates[j] != kd[i] or j + h >= m:
                continue
            seg = c[j + 1:j + h + 1]
            out["entry"][i] = o[j + 1]
            out["w_max"][i] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
            out["w_min"][i] = np.nanmin(seg) if np.isfinite(seg).any() else np.nan
            out["w_end"][i] = c[j + h]
            if not np.isnat(rd[i]):
                k = int(np.searchsorted(dates, rd[i]))      # 開示日（か、その次の営業日）
                if 1 <= k < m - 1 and np.isfinite(c[k - 1]) and c[k - 1] > 0:
                    out["react"][i] = c[k + 1] / c[k - 1] - 1.0
    res = pd.DataFrame(out, index=keys.index)
    ok = np.isfinite(res["entry"]) & (res["entry"] > 0)
    res["rise"] = np.where(ok, res["w_max"] / res["entry"] - 1.0, np.nan)
    res["fall"] = np.where(ok, res["w_min"] / res["entry"] - 1.0, np.nan)
    res["ret"] = np.where(ok, res["w_end"] / res["entry"] - 1.0, np.nan)
    return res


# --------------------------------------------------------------------------- #
# 表
# --------------------------------------------------------------------------- #

def _p(x) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.1f}%"


def _q(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def dist_row(name: str, ret: np.ndarray, rise: np.ndarray, fall: np.ndarray, need: np.ndarray) -> str:
    m = np.isfinite(ret) & np.isfinite(need)
    r, u, d, nd = ret[m], rise[m], fall[m], need[m]
    return (f"  {name:22s}{m.sum():8,d}{np.mean(u >= nd) * 100:8.1f}%{np.mean(d <= -nd) * 100:8.1f}%"
            f"{np.mean(r) * 100:+8.2f}{np.median(r) * 100:+8.2f}{_q(r, 10) * 100:+8.1f}{_q(r, 90) * 100:+8.1f}"
            f"{np.mean(r > 0) * 100:7.1f}%")


def print_direction(flag: np.ndarray, ws: pd.DataFrame, need: Dict[str, np.ndarray], title: str) -> None:
    print(f"\n■ {title}")
    for kname, nd in need.items():
        print(f"  --- 上下に抜ける割合のしきい値 {kname}")
        print(f"  {'':22s}{'行':>8s}{'上に抜け':>9s}{'下に抜け':>9s}{'平均%':>8s}{'中央%':>8s}{'P10%':>8s}{'P90%':>8s}{'勝率':>8s}")
        for name, m in (("決算が窓に入る", flag), ("入らない", ~flag)):
            print(dist_row(name, ws["ret"].to_numpy()[m], ws["rise"].to_numpy()[m], ws["fall"].to_numpy()[m], nd[m]))


def print_rule_outcome(fw: pd.DataFrame, flag: np.ndarray, title: str) -> None:
    """+10% の指値・20日目の終値の規則の収益（live_track.forward の ret, status）。"""
    print(f"  --- {title}: +10% の指値・20日で手仕舞い")
    print(f"  {'':22s}{'行':>8s}{'指値到達':>9s}{'平均%':>8s}{'中央%':>8s}{'P10%':>8s}{'勝率':>8s}")
    for name, m in (("決算が窓に入る", flag), ("入らない", ~flag)):
        f = fw[m]
        done = f[f["status"].isin([LT.HIT, "満了"])]
        r = pd.to_numeric(done["ret"], errors="coerce").to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        hit = float((done["status"] == LT.HIT).mean()) if len(done) else float("nan")
        print(f"  {name:22s}{len(done):8,d}{hit * 100:8.1f}%{np.mean(r):+8.2f}{np.median(r):+8.2f}{_q(r, 10):+8.1f}"
              f"{np.mean(r > 0) * 100:7.1f}%")


def ahead_bin(dte: np.ndarray) -> np.ndarray:
    lab_ = np.array(["予定不明"] * len(dte), dtype=object)
    for name, lo, hi in AHEAD_BINS:
        m = np.isfinite(dte) & (dte > lo if lo > 0 else dte >= lo) & (dte <= hi)
        lab_[m] = name
    return lab_


def print_model_by_ahead(bins: np.ndarray, pcts: Dict[str, np.ndarray], passed: np.ndarray, y: np.ndarray,
                         ret: np.ndarray, contrib: Optional[np.ndarray]) -> None:
    names = [b[0] for b in AHEAD_BINS] + ["予定不明"]
    head = f"  {'決算までの日数':20s}{'行':>8s}" + "".join(f"{'中央 ' + a[:-4]:>12s}" for a in RULE) + \
        f"{'3本95以上':>10s}{'正例率1.5σ':>11s}{'平均%':>8s}" + ("" if contrib is None else f"{'SHAP寄与':>9s}")
    print(head)
    for name in names:
        m = bins == name
        if not m.any():
            continue
        line = f"  {name:20s}{m.sum():8,d}" + "".join(f"{np.nanmedian(pcts[a][m]):12.1f}" for a in RULE)
        line += f"{np.mean(passed[m]) * 100:9.2f}%{np.nanmean(y[m]) * 100:10.1f}%{np.nanmean(ret[m]) * 100:+8.2f}"
        if contrib is not None:
            line += f"{np.nanmean(contrib[m]):+9.3f}"
        print(line)


def shap_days_to_earn(df_full: pd.DataFrame, data_dir: str, algo: str = "lgbm_k15") -> Tuple[Optional[np.ndarray], Optional[str]]:
    """lgbm_k15 の TreeSHAP（対数オッズ）。days_to_earn の列の寄与と、平均|寄与| の順位の説明文。失敗したら理由を返す。"""
    try:
        import e100_line_drift as E100
        model = E100.load_model(algo, data_dir)
        cols = E100.features_of(algo)
        missing = [c for c in cols if c not in df_full.columns]
        if missing:
            return None, f"データセットに無い列 {len(missing)}個"
        X = df_full[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        C = E100.contrib_matrix(model, X)[:, :-1]
        mean_abs = np.abs(C).mean(axis=0)
        order = np.argsort(-mean_abs)
        rank = {cols[i]: r + 1 for r, i in enumerate(order)}
        j = cols.index("days_to_earn")
        top = ", ".join(f"{cols[i]}({mean_abs[i]:.3f})" for i in order[:8])
        note = (f"days_to_earn の平均|寄与| {mean_abs[j]:.3f}（{len(cols)}列中 {rank['days_to_earn']}位）。"
                f"上位8列: {top}")
        return C[:, j], note
    except Exception as exc:  # noqa: BLE001  モデルが無い・形式が違う
        return None, f"TreeSHAP は出せなかった: {str(exc)[:80]}"


# --------------------------------------------------------------------------- #
# 窓の中の決算の中身（予想修正の向き）
# --------------------------------------------------------------------------- #

def disclosure_table(fins: pd.DataFrame) -> pd.DataFrame:
    """
    決算短信（実績値あり）ごとの、通期の営業利益予想（FOP）の直前の開示（同じ事業年度）比と期の種類。
    並びは forecast_revisions と同じ（DISC_ORDER）。同じ日の複数の開示は最後の1件。
    """
    f = fins.copy()
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f.dropna(subset=["DiscDate", "Code"])
    order = ["Code", "DiscDate"]
    for c in B.DISC_ORDER[1:]:
        if c in f.columns:
            f[f"_ord_{c}"] = f[c].astype("string")
            order.append(f"_ord_{c}")
    f = f.sort_values(order, kind="mergesort", na_position="last").reset_index(drop=True)
    if {"FOP", "CurFYSt"} <= set(f.columns):
        fop = pd.to_numeric(f["FOP"], errors="coerce")
        prev = fop.groupby([f["Code"], f["CurFYSt"]], sort=False).shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["rev_pct"] = np.where((prev > 0) & fop.notna(), fop / prev * 100.0 - 100.0, np.nan)
    else:
        f["rev_pct"] = np.nan
    has_result = f[["Sales", "NP"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    d = f.loc[has_result, ["Code", "DiscDate", "rev_pct"] + (["CurPerType"] if "CurPerType" in f.columns else [])]
    d = d.drop_duplicates(["Code", "DiscDate"], keep="last").reset_index(drop=True)
    cat = np.where(d["rev_pct"] > 0, "上方", np.where(d["rev_pct"] < 0, "下方",
                   np.where(d["rev_pct"] == 0, "据え置き", "比べられない")))
    d["rev_cat"] = cat
    return d


def print_content(cat: np.ndarray, per: np.ndarray, labels: Dict[str, np.ndarray], ret: np.ndarray,
                  react: np.ndarray, inwin: np.ndarray) -> None:
    print("\n■ §3 窓の中の最初の決算短信の中身（通期の営業利益予想の、同じ事業年度の直前の開示比）")
    n_in = int(inwin.sum())
    print(f"  窓に決算短信が入る行 {n_in:,}。区分の内訳: " +
          "、".join(f"{c} {int((cat[inwin] == c).sum()):,}（{(cat[inwin] == c).mean() * 100:.1f}%）" for c in REV_CATS))
    if per is not None:
        vc = pd.Series(per[inwin]).value_counts()
        print("  期の種類: " + "、".join(f"{k} {int(v):,}" for k, v in vc.items()))
    for lname, y in labels.items():
        print(f"  --- ラベル {lname}: 窓に入る行のうち、区分の割合（ラベル別）と区分ごとの正例率")
        print(f"  {'区分':12s}{'ラベル1':>9s}{'ラベル0':>9s}{'差(pt)':>8s}{'行':>8s}{'正例率':>8s}{'20日%':>8s}{'開示をまたぐ2日%':>16s}")
        pos, neg = inwin & (y == 1), inwin & (y == 0)
        for c in REV_CATS:
            m = inwin & (cat == c)
            p1 = float((cat[pos] == c).mean()) if pos.any() else float("nan")
            p0 = float((cat[neg] == c).mean()) if neg.any() else float("nan")
            print(f"  {c:12s}{p1 * 100:8.1f}%{p0 * 100:8.1f}%{(p1 - p0) * 100:+8.1f}{m.sum():8,d}"
                  f"{np.nanmean(y[m]) * 100:7.1f}%{np.nanmean(ret[m]) * 100:+8.2f}{np.nanmean(react[m]) * 100:+16.2f}")
        m = ~inwin
        print(f"  {'（窓に無し）':12s}{'':>9s}{'':>9s}{'':>8s}{m.sum():8,d}{np.nanmean(y[m]) * 100:7.1f}%{np.nanmean(ret[m]) * 100:+8.2f}")


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--horizon", type=int, default=B.DEFAULT_RISE.horizon)
    args = ap.parse_args(argv)
    h = int(args.horizon)

    full = pd.read_parquet(os.path.join(args.data_dir, "dataset.parquet"))
    missing = [c for c in P.NEED if c not in full.columns]
    if missing:
        raise SystemExit(f"データセットに要る列がありません: {missing}")
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str).str.strip()
    df = full[P.NEED + (["days_to_earn"] if "days_to_earn" in full.columns else [])].copy()
    print(f"データセット {len(df):,}行（{df['Date'].min().date()}〜{df['Date'].max().date()}）、判定の窓 {h}営業日")

    cal = TC.load(args.data_dir)
    if not (cal and cal.covers(df["Date"].min().date(), (df["Date"].max() + pd.Timedelta(days=60)).date())):
        raise SystemExit("営業日カレンダーが範囲を覆っていません")
    tdays = np.array(sorted(cal.days), dtype="datetime64[D]")
    start, end = P.window_bounds(df["Date"].to_numpy(), tdays, h)
    have_end = ~np.isnat(end)

    fins = B.load_parts("fins", args.data_dir)
    fins["Code"] = fins["Code"].astype(str).str.strip()
    disc = disclosure_table(fins)
    nxt, _same = P.next_disclosure(df, disc[["Code", "DiscDate"]])
    inwin = P.in_window(nxt, end)
    span = (end - df["Date"].to_numpy(dtype="datetime64[D]")).astype(float)
    dte = (pd.to_numeric(df["days_to_earn"], errors="coerce").to_numpy(dtype=float)
           if "days_to_earn" in df.columns else np.full(len(df), np.nan))
    pre = np.isfinite(dte) & have_end & (dte <= span) & (dte > 0)
    print(f"決算短信（実績値あり、同じ日は最後の1件）{len(disc):,}件。窓に入る行 {int(inwin.sum()):,}"
          f"（{inwin.mean() * 100:.1f}%）、公表済みの予定日が窓に入る行 {int(pre.sum()):,}（{pre.mean() * 100:.1f}%）")

    # §1 日足から窓の値動き
    bars = load_bars(args.data_dir, set(df["Code"]))
    print(f"日足 {len(bars):,}行（データセットの銘柄 {bars['Code'].nunique():,}）")
    react_dates = np.where(inwin, nxt, np.datetime64("NaT"))
    ws = window_stats(bars, df[["Code", "Date"]], h, react_dates)
    okw = np.isfinite(ws["ret"].to_numpy())
    print(f"窓の値動きが決まる行 {int(okw.sum()):,} / {len(df):,}")
    need: Dict[str, np.ndarray] = {}
    for k in (1.2, 1.5):
        nd, _ = B.rise_thresholds(df["vol_20d"], replace(B.DEFAULT_RISE, vol_norm_k=float(k)))
        need[f"{k:.1f}σ"] = np.asarray(nd, dtype=float)
    print_direction(inwin & okw, ws, need, "§1 方向か分散か（基準は翌営業日の寄り、窓は t+1〜t+20 の終値）")
    # 検算: データセットの future_rise（同じ定義）と日足から作った rise が一致するか
    fr = pd.to_numeric(df["future_rise"], errors="coerce").to_numpy(dtype=float)
    both = okw & np.isfinite(fr)
    diff = np.abs(ws["rise"].to_numpy()[both] - fr[both])
    print(f"  検算: データセットの future_rise と日足から作った上昇率の差が 1e-6 超の行 {int((diff > 1e-6).sum()):,} / {int(both.sum()):,}")
    fw = LT.forward(bars, df[["Code", "Date"]], target=LT.TAKE_PROFIT, hold=LT.HOLD, fallback="close")
    print_rule_outcome(fw, inwin, "§1 実運用の規則")

    # ラベル
    labels = {"1.2σ（本番）": pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=float),
              "1.5σ": B.rise_label_for_k(df, 1.5).to_numpy(dtype=float)}
    y15 = labels["1.5σ"]
    ret = ws["ret"].to_numpy()

    # §2 モデルの見方
    print("\n■ §2 モデルの見方（規則の3本の OOF の百分位、days_to_earn = その日までに公表済みの予定日までの暦日）")
    oof = LT.oof_rows(args.data_dir, LT.MODEL_DIR, RULE)
    key = df[["Code", "Date"]].copy()
    key["_i"] = np.arange(len(key))
    j = key.merge(oof[["Code", "Date"] + [f"p_{a}" for a in RULE]], on=["Code", "Date"], how="left")
    j = j.sort_values("_i")
    pcts = {a: j[f"p_{a}"].to_numpy(dtype=float) for a in RULE}
    has_oof = np.all(np.column_stack([np.isfinite(pcts[a]) for a in RULE]), axis=1)
    passed = np.all(np.column_stack([np.nan_to_num(pcts[a], nan=-1.0) >= LINE for a in RULE]), axis=1)
    print(f"  OOF が揃う行 {int(has_oof.sum()):,} / {len(df):,}、規則（3本 {LINE:.0f} 以上）を通る行 {int(passed.sum()):,}"
          f"（{passed[has_oof].mean() * 100:.2f}%）")
    contrib, note = shap_days_to_earn(full, args.data_dir)
    bins = ahead_bin(dte)
    m = has_oof
    print_model_by_ahead(bins[m], {a: pcts[a][m] for a in RULE}, passed[m], y15[m], ret[m],
                         None if contrib is None else contrib[m])
    print(f"  TreeSHAP（lgbm_k15、対数オッズ）: {note}")
    for name, flag in (("事前: 予定日が窓に入る", pre), ("事後: 決算短信が窓に入る", inwin)):
        a, b = m & flag, m & ~flag
        print(f"  {name}: 行の割合 {flag[m].mean() * 100:.1f}% → 通った行の中では {flag[m & passed].mean() * 100:.1f}%"
              f"（通る率 {passed[a].mean() * 100:.2f}% / {passed[b].mean() * 100:.2f}%）")
    print(f"  --- 通った行の成績（決算の有無。事後）")
    print(f"  {'':22s}{'行':>8s}{'正例率1.5σ':>11s}{'20日 平均%':>11s}{'中央%':>8s}{'+10%規則 平均%':>14s}{'指値到達':>9s}")
    for name, flag in (("決算が窓に入る", inwin), ("入らない", ~inwin)):
        mm = passed & flag
        done = fw[mm & fw["status"].isin([LT.HIT, "満了"]).to_numpy()]
        r = pd.to_numeric(done["ret"], errors="coerce").to_numpy(dtype=float)
        hit = float((done["status"] == LT.HIT).mean()) if len(done) else float("nan")
        print(f"  {name:22s}{mm.sum():8,d}{np.nanmean(y15[mm]) * 100:10.1f}%{np.nanmean(ret[mm]) * 100:+11.2f}"
              f"{np.nanmedian(ret[mm]) * 100:+8.2f}{np.nanmean(r):+14.2f}{hit * 100:8.1f}%")

    # §3 窓の中の決算の中身
    left = pd.DataFrame({"Code": df["Code"].to_numpy(), "DiscDate": pd.to_datetime(nxt), "_i": np.arange(len(df))})
    got = left.merge(disc, on=["Code", "DiscDate"], how="left").sort_values("_i")
    cat = got["rev_cat"].fillna("比べられない").to_numpy(dtype=object)
    per = got["CurPerType"].to_numpy(dtype=object) if "CurPerType" in got.columns else None
    print_content(cat, per, labels, ret, ws["react"].to_numpy(), inwin & okw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
