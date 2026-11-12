#!/usr/bin/env python3
"""
日証金（日本証券金融、taisyaku.jp）の需給の特徴量・作り直し版（v2。研究用。docs/DATA_JSF.md「特徴量の作り直し（v2）」）。

運用者の依頼（2026-11-12）「（過去3年が）そろい…であれば、特徴量として、作成してほしいです」「列方向、横方向、銘柄間での
バリデーションチェックも欠かさずに」。運用者の選択（同日）: 373列で検証し直す・10/9 の17列＋同じ銘柄の過去との比較（60日の
位置の3列）・制限措置は毎日の写しから・貸借区分は J-Quants の信用区分から・窓は営業日で数える。

10/9 の作り（research/jsf_features.py。実験66・67・72 と、結果を見る前に決めた実験69 が使う）は変えない。こちらは別の列名
（jsf2_*）で作る。

v1 との違い
  1. 窓を営業日で数える。銘柄ごとに、取引所の営業日の格子（最初の申込日〜最後の申込日）に並べてから数える。毎日の確報が欠けた
     日は欠測のまま（v1 は申込日の行の数で数えたので、欠けた日の分だけ窓が黙って長くなった）:
       5・20日の変化   両端の日がそろったときだけ
       5日の新規の合計 5日がそろったときだけ
       20日の逆日歩の日数・最大  20日のうち10日以上あるとき（v1 と同じ最低の日数）
       60日の位置      60日のうち40日以上あるとき
  2. 制限措置: 銘柄ごとの過去ファイルにその申込日の行があればその値（v1 と同じ）。無い申込日（過去ファイルを取った後）は、毎日の
     制限措置一覧の写し（jsf_restrict.parquet）から作る。写しの日と申込日の対応（RESTRICT_SNAP_SHIFT 営業日）は、過去ファイルと
     重なる日で確かめてから決める（決まるまでは None = 写しを使わず欠測）。写しが無い日は欠測（0 にしない）
  3. 貸借区分: J-Quants の銘柄一覧（master_hist_*.parquet）の信用区分 Mrgn（2 貸借 → 1、1 信用・3 その他 → 0）を、申込日に
     「その日に有効だった区分」で合わせる（build_dataset の業種・市場区分と同じ merge_asof）。過去ファイルの「貸借区分」は使わない
  4. 足す3列: 融資残高・貸株残高・貸借倍率の、その銘柄の直近60営業日（その申込日を含む）の中での位置。
     位置 = (順位 − 1) / (値の数 − 1)。順位は小さい方から、同じ値は平均の順位。0 = 窓の中で最も低い、1 = 最も高い

変えないこと: 取引日 T の行には、申込日が T より前の最新の値（8暦日以上前は欠測。v1 の attach と同じ。確報が欠けた日の翌日の
行は、その前の申込日の値になる。jsf2_lag で分かる）。出来高比・asinh・log1p の形と、20日平均出来高の取り方。

列（prefix jsf2_。日証金の値が無い行は欠測のまま）
  jsf2_ratio, jsf2_ratio_chg5, jsf2_ratio_chg20
  jsf2_loan_v, jsf2_stock_v, jsf2_net_v
  jsf2_loan_chg5_v, jsf2_loan_chg20_v, jsf2_stock_chg5_v, jsf2_stock_chg20_v
  jsf2_long_new5_v, jsf2_short_new5_v
  jsf2_fee, jsf2_fee_days20, jsf2_fee_max20
  jsf2_restrict, jsf2_lendable
  jsf2_loan_pos60, jsf2_stock_pos60, jsf2_ratio_pos60
  確認用: jsf2_lag（使った申込日から T までの暦日）
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import jsf_features as V1

DATA_DIR = V1.DATA_DIR
MAX_STALE = V1.MAX_STALE
VOL_WINDOW = V1.VOL_WINDOW
#: 写しの日 D の制限措置一覧が、どの申込日の状態にあたるか（申込日 = D から何営業日後か）。過去ファイルと重なる日で確かめて
#: 運用者が決めるまでは None（写しを使わない）
RESTRICT_SNAP_SHIFT: Optional[int] = None
POS_WINDOW = 60
POS_MIN = 40
FEE_WINDOW = 20
FEE_MIN = 10

BASE_COLS = ["jsf2_ratio", "jsf2_ratio_chg5", "jsf2_ratio_chg20",
             "jsf2_loan_v", "jsf2_stock_v", "jsf2_net_v",
             "jsf2_loan_chg5_v", "jsf2_loan_chg20_v", "jsf2_stock_chg5_v", "jsf2_stock_chg20_v",
             "jsf2_long_new5_v", "jsf2_short_new5_v",
             "jsf2_fee", "jsf2_fee_days20", "jsf2_fee_max20",
             "jsf2_restrict", "jsf2_lendable",
             "jsf2_loan_pos60", "jsf2_stock_pos60", "jsf2_ratio_pos60"]
CHECK_COLS = ["jsf2_lag"]


def columns(group: str = "all") -> List[str]:
    """特徴量の列名。all = 20列、check = 確認用、both = 両方。"""
    if group == "all":
        return list(BASE_COLS)
    if group == "check":
        return list(CHECK_COLS)
    if group == "both":
        return list(BASE_COLS) + list(CHECK_COLS)
    raise KeyError(group)


# ---------------------------------------------------------------------- #
# 営業日・写し・信用区分
# ---------------------------------------------------------------------- #

def trading_days(data_dir: str = DATA_DIR) -> pd.DatetimeIndex:
    """取引所の営業日（J-Quants のカレンダー。取り込みが保存したもの）。無ければ止める（平日で推し量らない）。"""
    import trading_calendar
    cal = trading_calendar.load(data_dir)
    if not cal.days:
        raise SystemExit("営業日のカレンダー（calendar.parquet）が無い。日証金の窓を営業日で数えられない")
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in cal.days))


def load_snapshots(path: str) -> pd.DataFrame:
    """制限措置一覧の写し -> (snap_date, code, level)。同じ写しの同じ銘柄は最も重い措置。"""
    r = pd.read_parquet(path)
    out = pd.DataFrame({"snap_date": pd.to_datetime(r["snap_date"]).dt.normalize(),
                        "code": r["code"].astype(str),
                        "level": V1.restrict_level(r["実施措置"]).to_numpy()})
    return out.groupby(["snap_date", "code"], as_index=False)["level"].max()


def snapshot_levels(snaps: pd.DataFrame, days: pd.DatetimeIndex, shift: int) -> pd.DataFrame:
    """
    写しの日 D を、D から shift 営業日後の申込日 A に対応させる。戻り値 (app_date, code, level) と、写しのある申込日の一覧
    （attrs["covered"]）。写しに載っていない銘柄は、写しのある申込日なら 0（措置なし）。それは panel_v2 が付ける。
    """
    pos = {d: i for i, d in enumerate(days)}
    def to_app(d):
        i = pos.get(pd.Timestamp(d))
        if i is None:
            return pd.NaT
        j = i + shift
        return days[j] if 0 <= j < len(days) else pd.NaT
    s = snaps.copy()
    s["app_date"] = s["snap_date"].map(to_app)
    s = s.dropna(subset=["app_date"])
    out = s[["app_date", "code", "level"]].reset_index(drop=True)
    out.attrs["covered"] = sorted(s["app_date"].unique())
    return out


def load_margin_category(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """J-Quants の銘柄一覧の信用区分 -> (Date, Code, lendable)。2 貸借 → 1、1 信用・3 その他 → 0、ほかは欠測。"""
    paths = sorted(glob.glob(os.path.join(data_dir, "master_hist_*.parquet")))
    if not paths:
        raise SystemExit("master_hist_*.parquet が無い（貸借区分を J-Quants の信用区分から作れない）")
    m = pd.concat([pd.read_parquet(p, columns=["Date", "Code", "Mrgn"]) for p in paths], ignore_index=True)
    return margin_from_master(m)


def margin_from_master(m: pd.DataFrame) -> pd.DataFrame:
    code = m["Mrgn"].astype("string").str.strip()
    lend = pd.Series(np.nan, index=m.index, dtype=float)
    lend[code == "2"] = 1.0
    lend[code.isin(["1", "3"])] = 0.0
    out = pd.DataFrame({"Date": pd.to_datetime(m["Date"]), "Code": m["Code"].astype(str), "lendable": lend})
    return (out.dropna(subset=["Date"]).sort_values("Date")
               .drop_duplicates(["Date", "Code"], keep="last").reset_index(drop=True))


# ---------------------------------------------------------------------- #
# パネル（申込日×銘柄。営業日の格子に並べる）
# ---------------------------------------------------------------------- #

def panel_v2(hist: Optional[pd.DataFrame], daily: Optional[pd.DataFrame], days: pd.DatetimeIndex,
             snaps: Optional[pd.DataFrame] = None, shift: Optional[int] = RESTRICT_SNAP_SHIFT,
             margin: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    hist（銘柄ごとの過去）と daily（毎日の確報）を1本にし、銘柄ごとに営業日の格子に並べる。
      - 同じ申込日なら daily（確報）を残す（v1 と同じ）。品貸料率の年率換算は daily の行に無ければ hist の同じ日の値
      - 観測のある行（observed=True）で品貸料率の無い日は 0（逆日歩なし）。格子で足した行（確報が欠けた日）は欠測
      - restrict: hist にその日の行があればその値。無ければ写し（shift が決まっているとき）。どちらも無ければ欠測
      - lendable: J-Quants の信用区分（margin）を申込日で merge_asof（後ろ向き）
    """
    parts = [p for p in (hist, daily) if p is not None and len(p)]
    if not parts:
        raise ValueError("日証金のデータが無い")
    allp = pd.concat(parts, ignore_index=True, sort=False)
    allp["app_date"] = pd.to_datetime(allp["app_date"]).dt.normalize()
    allp["Code"] = allp["Code"].astype(str)
    allp = allp.sort_values(["Code", "app_date", "source"])          # daily < hist → keep="first" で daily
    if hist is not None and len(hist):
        h = hist[["app_date", "Code", "restrict", "fee_ann"]].copy()
        h["app_date"] = pd.to_datetime(h["app_date"]).dt.normalize()
        h["Code"] = h["Code"].astype(str)
        h = h.drop_duplicates(["app_date", "Code"], keep="last")
        h = h.rename(columns={"restrict": "_r_hist", "fee_ann": "_f_hist"})
        h["_in_hist"] = True
        allp = allp.merge(h, on=["app_date", "Code"], how="left")
    else:
        allp["_r_hist"] = np.nan
        allp["_f_hist"] = np.nan
        allp["_in_hist"] = False
    p = allp.drop_duplicates(["Code", "app_date"], keep="first").reset_index(drop=True)
    p["fee_ann"] = p["fee_ann"].fillna(p["_f_hist"])
    p["_in_hist"] = p["_in_hist"].eq(True)                         # 合わなかった行は欠測 → False
    keep = ["app_date", "Code", "source", "loan_new", "loan_ret", "loan_bal", "stock_new", "stock_ret",
            "stock_bal", "net_bal", "price", "fee_ann", "_r_hist", "_in_hist"]
    p = p[[c for c in keep if c in p.columns]]

    # 営業日の格子
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days).normalize())))
    bad = ~p["app_date"].isin(days)
    if bad.any():
        raise ValueError(f"営業日でない申込日の行が {int(bad.sum()):,} 行ある（カレンダーと合わない）")
    span = p.groupby("Code")["app_date"].agg(["min", "max"])
    lo = np.searchsorted(days.values, span["min"].values)
    hi = np.searchsorted(days.values, span["max"].values)
    codes = np.repeat(span.index.values, hi - lo + 1)
    idx = np.concatenate([np.arange(a, b + 1) for a, b in zip(lo, hi)]) if len(lo) else np.array([], int)
    grid = pd.DataFrame({"Code": codes, "app_date": days.values[idx]})
    g = grid.merge(p, on=["Code", "app_date"], how="left")
    g["observed"] = g["source"].notna()
    g["source"] = g["source"].fillna("missing")
    g["_in_hist"] = g["_in_hist"].eq(True)
    g.loc[g["observed"], "fee_ann"] = g.loc[g["observed"], "fee_ann"].fillna(0.0)

    # 制限措置: hist → 写し → 欠測
    g["restrict"] = g["_r_hist"].where(g["_in_hist"])
    g["restrict_src"] = np.where(g["_in_hist"] & g["restrict"].notna(), "hist", "")
    if snaps is not None and shift is not None and len(snaps):
        sl = snapshot_levels(snaps, days, shift)
        covered = set(pd.DatetimeIndex(sl.attrs["covered"]))
        lv = sl.rename(columns={"code": "Code", "level": "_r_snap"})
        g = g.merge(lv, on=["app_date", "Code"], how="left")
        on_snap = g["app_date"].isin(covered)
        g["_r_snap"] = g["_r_snap"].where(~on_snap | g["_r_snap"].notna(), 0.0)   # 写しにあって載っていない = 措置なし
        use = g["restrict"].isna() & on_snap & g["observed"]
        g.loc[use, "restrict"] = g.loc[use, "_r_snap"]
        g.loc[use, "restrict_src"] = "snap"
        g = g.drop(columns=["_r_snap"])
    g.loc[~g["observed"], "restrict"] = np.nan

    # 貸借区分: J-Quants の信用区分（その日に有効だった区分）
    if margin is not None and len(margin):
        mg = margin.rename(columns={"Date": "app_date"}).sort_values("app_date")
        left = g[["Code", "app_date"]].reset_index().sort_values("app_date")
        mm = pd.merge_asof(left, mg[["app_date", "Code", "lendable"]], on="app_date", by="Code",
                           direction="backward").set_index("index").sort_index()
        g["lendable"] = mm["lendable"].where(g["observed"])
    else:
        g["lendable"] = np.nan
    g = g.drop(columns=["_r_hist", "_in_hist"])
    return g.sort_values(["Code", "app_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------- #
# 特徴量（格子の上で。1行 = 1営業日なので、行で数える窓 = 営業日の窓）
# ---------------------------------------------------------------------- #

def _pos(s: pd.Series, window: int = POS_WINDOW, min_periods: int = POS_MIN) -> pd.Series:
    """銘柄ごとの直近 window 営業日（その日を含む）の中での位置 (順位 − 1) / (値の数 − 1)。同じ値は平均の順位。"""
    r = s.rolling(window, min_periods=min_periods).rank(method="average")
    n = s.rolling(window, min_periods=min_periods).count()
    out = (r - 1.0) / (n - 1.0)
    return out.where(n > 1)


def features_v2(g: pd.DataFrame, vol: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """格子（panel_v2）の上で jsf2_* を作る。値は申込日のもの（取引日への付け替えは attach_v2）。"""
    g = g.sort_values(["Code", "app_date"]).reset_index(drop=True)
    by = g["Code"]
    out = g[["Code", "app_date", "observed"]].copy()
    loan = g["loan_bal"].astype(float)
    stock = g["stock_bal"].astype(float)
    ratio = np.log((loan.clip(lower=0) + 1.0) / (stock.clip(lower=0) + 1.0))
    out["jsf2_ratio"] = ratio
    for n in (5, 20):
        out[f"jsf2_ratio_chg{n}"] = ratio - ratio.groupby(by, sort=False).shift(n)
    if vol is not None and len(vol):
        v = vol.rename(columns={"Date": "app_date"}).copy()
        v["app_date"] = pd.to_datetime(v["app_date"])
        v["Code"] = v["Code"].astype(str)
        v = v.sort_values("app_date")
        pv = pd.merge_asof(g[["Code", "app_date"]].reset_index().sort_values("app_date"), v,
                           on="app_date", by="Code", direction="backward",
                           tolerance=pd.Timedelta(days=10)).set_index("index").sort_index()
        av = pv["avg_vol"].where(pv["avg_vol"] > 0)
    else:
        av = pd.Series(np.nan, index=g.index)
    out["jsf2_loan_v"] = np.arcsinh(loan / av)
    out["jsf2_stock_v"] = np.arcsinh(stock / av)
    out["jsf2_net_v"] = np.arcsinh((loan - stock) / av)
    for n in (5, 20):
        out[f"jsf2_loan_chg{n}_v"] = np.arcsinh((loan - loan.groupby(by, sort=False).shift(n)) / av)
        out[f"jsf2_stock_chg{n}_v"] = np.arcsinh((stock - stock.groupby(by, sort=False).shift(n)) / av)
    for col, name in (("loan_new", "jsf2_long_new5_v"), ("stock_new", "jsf2_short_new5_v")):
        tot = g[col].astype(float).groupby(by, sort=False).transform(lambda s: s.rolling(5, min_periods=5).sum())
        out[name] = np.arcsinh(tot / av)
    fee = g["fee_ann"].astype(float).clip(lower=0.0)               # 格子で足した日は欠測のまま
    out["jsf2_fee"] = np.log1p(fee)
    pos_fee = (fee > 0).astype(float).where(fee.notna())
    out["jsf2_fee_days20"] = pos_fee.groupby(by, sort=False).transform(
        lambda s: s.rolling(FEE_WINDOW, min_periods=FEE_MIN).sum())
    out["jsf2_fee_max20"] = np.log1p(fee.groupby(by, sort=False).transform(
        lambda s: s.rolling(FEE_WINDOW, min_periods=FEE_MIN).max()))
    out["jsf2_restrict"] = g["restrict"].astype(float)
    out["jsf2_lendable"] = g["lendable"].astype(float)
    out["jsf2_loan_pos60"] = loan.groupby(by, sort=False).transform(_pos)
    out["jsf2_stock_pos60"] = stock.groupby(by, sort=False).transform(_pos)
    out["jsf2_ratio_pos60"] = ratio.groupby(by, sort=False).transform(_pos)
    # 格子で足した日（確報が欠けた日）の行は、どの列も欠測（その日の値は分からない）
    for c in BASE_COLS:
        out.loc[~out["observed"], c] = np.nan
    return out


def attach_v2(frame: pd.DataFrame, feats: pd.DataFrame, max_stale: int = MAX_STALE) -> pd.DataFrame:
    """取引日 T の行（Code, Date）に、申込日が T より前の最新の観測の値を付ける（v1 の attach と同じ決め方）。"""
    obs = feats[feats["observed"]].drop(columns=["observed"])
    left = frame[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    right = obs.rename(columns={"app_date": "Date"}).copy()
    right["Code"] = right["Code"].astype(str)
    right["_app"] = right["Date"]
    m = pd.merge_asof(left.sort_values("Date"), right.sort_values("Date"), on="Date", by="Code",
                      direction="backward", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=max_stale - 1))
    m = m.sort_values("_i").set_index(frame.index)
    m["jsf2_lag"] = (m["Date"] - m["_app"]).dt.days.astype(float)
    cols = columns("both")
    add = m[cols].astype(float)
    return pd.concat([frame.drop(columns=[c for c in cols if c in frame.columns]), add], axis=1)


def inputs(data_dir: str = DATA_DIR, with_daily: bool = True):
    """研究用: data_dir から (hist, daily, snaps, margin, days) を読む。"""
    hist = V1.load_hist(os.path.join(data_dir, "jsf_hist.parquet"))
    daily = None
    if with_daily and os.path.exists(os.path.join(data_dir, "jsf_balance.parquet")):
        daily = V1.load_daily(os.path.join(data_dir, "jsf_balance.parquet"),
                              os.path.join(data_dir, "jsf_lending.parquet"))
    sp = os.path.join(data_dir, "jsf_restrict.parquet")
    snaps = load_snapshots(sp) if os.path.exists(sp) else None
    return hist, daily, snaps, load_margin_category(data_dir), trading_days(data_dir)


def build(frame: pd.DataFrame, data_dir: str = DATA_DIR, shift: Optional[int] = RESTRICT_SNAP_SHIFT,
          codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """研究用の入口: data_dir の日証金・日足・銘柄一覧・カレンダーから jsf2_* を作って frame に付ける。"""
    hist, daily, snaps, margin, days = inputs(data_dir)
    if codes is not None:
        keep = set(map(str, codes))
        hist = hist[hist["Code"].isin(keep)]
        daily = daily[daily["Code"].isin(keep)] if daily is not None else None
    g = panel_v2(hist, daily, days, snaps, shift, margin)
    vol = V1.avg_volume(V1.load_bars(data_dir, codes=g["Code"].unique()))
    return attach_v2(frame, features_v2(g, vol))
