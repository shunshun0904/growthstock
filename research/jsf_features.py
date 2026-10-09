#!/usr/bin/env python3
"""
日証金（日本証券金融、taisyaku.jp）の需給データから特徴量を作る（研究用。docs/DATA_JSF.md）。

元データ（research/_data/。Release data-jsf から取る。jsf_fetch.py が作る）
  jsf_hist.parquet     銘柄ごとの過去3年（申込日×銘柄、東証）。最後の日は取った当日の速報
  jsf_balance.parquet  DATA ページの残高（申込日×銘柄×取引所区分名）。前営業日の申込日の確報。
                       同じ申込日なら daily（確報）を hist より優先する
  jsf_lending.parquet  品貸料率（逆日歩）の一覧（申込日×銘柄×取引所区分）
  jsf_restrict.parquet 制限措置等の一覧の写し（使わない。hist の「制限措置」列が日ごとの状態）

時点の扱い（docs/DATA_JSF.md「時点の扱い」）
  取引日 T の行には、申込日が T より前の最新の確報を使う（原則 T の前営業日。確報は T の昼に
  出るので夜の予測に間に合う）。T 当日の速報は使わない（学習は確報、予測は速報、という食い違いを
  作らない）。申込日が T より MAX_STALE 暦日以上前なら欠測（古い値を引きずらない）。

列（prefix jsf_。値の無い銘柄＝貸借銘柄でない・2023年秋より前は欠測のまま）
  jsf_ratio          貸借倍率の対数 log((融資残高+1)/(貸株残高+1))。週次の信用倍率（credit_ratio）の日次・制度信用版
  jsf_ratio_chg5/20  jsf_ratio の 5 / 20 申込日の変化
  jsf_loan_v         融資残高 ÷ 20日平均出来高（何日ぶんの出来高か）
  jsf_stock_v        貸株残高 ÷ 20日平均出来高
  jsf_net_v          （融資残高 − 貸株残高）÷ 20日平均出来高（差引残高）
  jsf_loan_chg5/20_v  融資残高の 5 / 20 申込日の増減 ÷ 20日平均出来高（買い方の積み増し）
  jsf_stock_chg5/20_v 貸株残高の 5 / 20 申込日の増減 ÷ 20日平均出来高（売り方の積み増し）
  jsf_long_new5_v    直近5申込日の融資新規の合計 ÷ 20日平均出来高
  jsf_short_new5_v   直近5申込日の貸株新規の合計 ÷ 20日平均出来高
  jsf_fee            品貸料率（年率換算 %）。貸借銘柄で逆日歩が無い日は 0
  jsf_fee_days20     直近20申込日のうち逆日歩が付いた日数
  jsf_fee_max20      直近20申込日の品貸料率（年率換算 %）の最大
  jsf_restrict       制限措置の段階（0 なし / 1 注意喚起 / 2 申込制限 / 3 申込停止。hist の文言から）
  jsf_lendable       貸借銘柄なら 1（制度信用の売りができる）、融資銘柄（買いだけ）なら 0。hist の「貸借区分」から
  jsf_lag            使った申込日から T までの暦日（確認用。原則 1〜4）

20日平均出来高は J-Quants の日足（bars_*.parquet の Vo）を、申込日と同じ日まで（T より前）で取る。
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "_data")
HIST = os.path.join(DATA_DIR, "jsf_hist.parquet")
BALANCE = os.path.join(DATA_DIR, "jsf_balance.parquet")
LENDING = os.path.join(DATA_DIR, "jsf_lending.parquet")

#: 申込日が T よりこの暦日以上前なら使わない（連休明けでも 1〜4 日。5営業日 ≒ 7暦日）
MAX_STALE = 8
#: 出来高の平均の窓（営業日）
VOL_WINDOW = 20

#: 共通の列名 <- 元の列名（hist / daily で見出しが違う）
HIST_COLS = {"loan_new": "融資新規（株）", "loan_ret": "融資返済（株）", "loan_bal": "融資残高（株）",
             "stock_new": "貸株新規（株）", "stock_ret": "貸株返済（株）", "stock_bal": "貸株残高（株）",
             "net_bal": "差引残高（株）", "price": "貸借値段（円）",
             "fee_yen": "品貸料率（品貸日数分/円）", "fee_days": "品貸日数",
             "fee_ann": "品貸料率（年率換算/％）", "restrict": "制限措置", "lendable": "貸借区分"}
DAILY_COLS = {"loan_new": "融資新規株数", "loan_ret": "融資返済株数", "loan_bal": "融資残高株数",
              "stock_new": "貸株新規株数", "stock_ret": "貸株返済株数", "stock_bal": "貸株残高株数",
              "net_bal": "差引残高株数"}
LENDING_COLS = {"price": "貸借値段（円）", "fee_yen": "当日品貸料率（円）", "fee_days": "当日品貸日数"}

#: 制限措置の文言 -> 段階。文言はサイトのもの（docs/DATA_JSF.md）。合わないものは 0 にせず欠測
RESTRICT_LEVELS = (("停止", 3), ("制限", 2), ("注意", 1))

BASE_COLS = ["jsf_ratio", "jsf_ratio_chg5", "jsf_ratio_chg20",
             "jsf_loan_v", "jsf_stock_v", "jsf_net_v",
             "jsf_loan_chg5_v", "jsf_loan_chg20_v", "jsf_stock_chg5_v", "jsf_stock_chg20_v",
             "jsf_long_new5_v", "jsf_short_new5_v",
             "jsf_fee", "jsf_fee_days20", "jsf_fee_max20", "jsf_restrict", "jsf_lendable"]
CHECK_COLS = ["jsf_lag"]


def columns(group: str = "all") -> List[str]:
    """特徴量の列名。all = 本番候補の16本、check = 確認用、both = 両方。"""
    if group == "all":
        return list(BASE_COLS)
    if group == "check":
        return list(CHECK_COLS)
    if group == "both":
        return list(BASE_COLS) + list(CHECK_COLS)
    raise KeyError(group)


# ---------------------------------------------------------------------- #
# 読み込み: hist と daily を共通の列にそろえ、申込日×銘柄の1本のパネルにする
# ---------------------------------------------------------------------- #

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def restrict_level(text: pd.Series) -> pd.Series:
    """制限措置の文言 -> 0/1/2/3。空・「－」は 0（措置なし）。知らない文言は欠測。"""
    t = text.astype("string").fillna("").str.strip()
    out = pd.Series(np.nan, index=text.index, dtype=float)
    none = t.isin(["", "－", "-", "―", "ー"])
    out[none] = 0.0
    for word, level in RESTRICT_LEVELS:
        hit = t.str.contains(word, na=False) & out.isna()
        out[hit] = float(level)
    return out


def lendable_flag(text: pd.Series) -> pd.Series:
    """貸借区分の文言 -> 1（貸借: 融資も貸株もできる）/ 0（貸借融資: 融資だけ、非貸借）。空は欠測。"""
    t = text.astype("string").fillna("").str.strip()
    out = pd.Series(np.nan, index=text.index, dtype=float)
    out[t == "貸借"] = 1.0
    out[t.isin(["貸借融資", "非貸借", "融資"])] = 0.0
    return out


def _tse(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """取引所区分の列があれば東証の行だけ（無ければそのまま）。"""
    if col in df.columns:
        s = df[col].astype("string").fillna("")
        tse = s.str.contains("東証", na=False)
        if tse.any():
            return df[tse]
    return df


def load_hist(path: str = HIST) -> pd.DataFrame:
    """jsf_hist.parquet -> 共通の列。"""
    h = pd.read_parquet(path)
    out = pd.DataFrame({"app_date": pd.to_datetime(h["申込日"]), "Code": h["code"].astype(str)})
    for k, c in HIST_COLS.items():
        if c not in h.columns:
            raise KeyError(f"jsf_hist に列が無い: {c}（ある列: {list(h.columns)[:30]}）")
        out[k] = (restrict_level(h[c]) if k == "restrict" else lendable_flag(h[c]) if k == "lendable"
                  else _num(h[c]))
    out["source"] = "hist"
    return out.dropna(subset=["app_date"]).reset_index(drop=True)


def load_daily(balance: str = BALANCE, lending: str = LENDING) -> pd.DataFrame:
    """jsf_balance（確報）+ jsf_lending -> 共通の列。品貸料率は年率換算（円/株/日 ÷ 貸借値段 × 365）。"""
    b = _tse(pd.read_parquet(balance), "取引所区分名")
    out = pd.DataFrame({"app_date": pd.to_datetime(b["申込日"]), "Code": b["code"].astype(str)})
    for k, c in DAILY_COLS.items():
        if c not in b.columns:
            raise KeyError(f"jsf_balance に列が無い: {c}（ある列: {list(b.columns)[:30]}）")
        out[k] = _num(b[c].to_numpy())
    out = out.dropna(subset=["app_date"]).drop_duplicates(["app_date", "Code"], keep="last")
    for k in ("price", "fee_yen", "fee_days", "fee_ann"):
        out[k] = np.nan
    out["restrict"] = np.nan                      # daily には日ごとの制限措置の列が無い
    out["lendable"] = np.nan                      # 貸借区分も無い（panel で hist の値を引き継ぐ）
    if lending and os.path.exists(lending):
        ln = _tse(pd.read_parquet(lending), "取引所区分")
        ld = pd.DataFrame({"app_date": pd.to_datetime(ln["貸借申込日"]), "Code": ln["code"].astype(str)})
        for k, c in LENDING_COLS.items():
            ld[k] = _num(ln[c].to_numpy()) if c in ln.columns else np.nan
        ld = ld.dropna(subset=["app_date"]).drop_duplicates(["app_date", "Code"], keep="last")
        ld["fee_ann"] = annualize_fee(ld["fee_yen"], ld["fee_days"], ld["price"])
        out = out.drop(columns=["price", "fee_yen", "fee_days", "fee_ann"]).merge(
            ld, on=["app_date", "Code"], how="left")
    out["source"] = "daily"
    return out.reset_index(drop=True)


def annualize_fee(fee_yen: pd.Series, fee_days: pd.Series, price: pd.Series) -> pd.Series:
    """品貸料率（品貸日数ぶんの円/株）を年率換算 % にする（hist の「年率換算」と同じ考え方）。"""
    days = _num(fee_days).where(lambda s: s > 0, 1.0)
    out = _num(fee_yen) / days / _num(price) * 365.0 * 100.0
    return out.where(_num(price) > 0)


def panel(hist: Optional[pd.DataFrame] = None, daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    hist と daily を1本にする（申込日×銘柄で一意。同じ申込日なら daily（確報）を優先）。
    hist にしか無い列（制限措置・年率換算の品貸料率）は、daily の行には hist の同じ日の値を補う。
    逆日歩の無い日の fee_ann は 0（その銘柄に残高の行がある = 貸借銘柄）。
    """
    parts = [p for p in (hist, daily) if p is not None and len(p)]
    if not parts:
        raise ValueError("日証金のデータが無い")
    allp = pd.concat(parts, ignore_index=True, sort=False)
    allp = allp.sort_values(["Code", "app_date", "source"])      # daily < hist の順 → keep="first" で daily
    # daily の行に hist の制限措置・年率換算を補う（同じ申込日・銘柄）
    if hist is not None and daily is not None and len(hist) and len(daily):
        fill = hist[["app_date", "Code", "restrict", "fee_ann", "lendable"]].drop_duplicates(["app_date", "Code"])
        allp = allp.merge(fill.rename(columns={"restrict": "_r", "fee_ann": "_f", "lendable": "_l"}),
                          on=["app_date", "Code"], how="left")
        allp["restrict"] = allp["restrict"].fillna(allp["_r"])
        allp["fee_ann"] = allp["fee_ann"].fillna(allp["_f"])
        allp["lendable"] = allp["lendable"].fillna(allp["_l"])
        allp = allp.drop(columns=["_r", "_f", "_l"])
    out = allp.drop_duplicates(["Code", "app_date"], keep="first").reset_index(drop=True)
    out["fee_ann"] = out["fee_ann"].fillna(0.0)
    # 貸借区分は滅多に変わらないので、hist の無い日（daily だけの日）は銘柄ごとに前の値を引き継ぐ
    out["lendable"] = out.groupby("Code", sort=False)["lendable"].ffill()
    return out


# ---------------------------------------------------------------------- #
# 出来高（J-Quants の日足）
# ---------------------------------------------------------------------- #

def avg_volume(bars: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """銘柄ごとの出来高の移動平均（その日まで window 営業日）。列: Date, Code, avg_vol。"""
    b = bars[["Date", "Code", "Vo"]].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    b["Vo"] = _num(b["Vo"])
    b = b.dropna(subset=["Date"]).sort_values(["Code", "Date"])
    b["avg_vol"] = (b.groupby("Code")["Vo"]
                     .transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).mean()))
    return b[["Date", "Code", "avg_vol"]]


def load_bars(data_dir: str = DATA_DIR, codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise SystemExit("bars_*.parquet がありません")
    frames = []
    for p in paths:
        f = pd.read_parquet(p, columns=["Date", "Code", "Vo"])
        if codes is not None:
            f = f[f["Code"].astype(str).isin(set(codes))]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------- #
# 特徴量（申込日の系列の上で作る）
# ---------------------------------------------------------------------- #

def features_on_panel(p: pd.DataFrame, vol: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    申込日×銘柄のパネルに jsf_* を付ける（この時点では「申込日」の値。取引日への付け替えは attach）。
    変化は「申込日の行数」で数える（申込日 = 営業日なので営業日の数と同じ）。
    """
    p = p.sort_values(["Code", "app_date"]).reset_index(drop=True)
    g = p.groupby("Code", sort=False)
    out = p[["Code", "app_date"]].copy()
    out["jsf_ratio"] = np.log((p["loan_bal"].clip(lower=0) + 1.0) / (p["stock_bal"].clip(lower=0) + 1.0))
    for n in (5, 20):
        out[f"jsf_ratio_chg{n}"] = out["jsf_ratio"] - out.groupby(p["Code"], sort=False)["jsf_ratio"].shift(n)
    if vol is not None and len(vol):
        v = vol.rename(columns={"Date": "app_date"})
        v = v.sort_values("app_date")
        pv = pd.merge_asof(p[["Code", "app_date"]].reset_index().sort_values("app_date"), v,
                           on="app_date", by="Code", direction="backward",
                           tolerance=pd.Timedelta(days=10)).set_index("index").sort_index()
        av = pv["avg_vol"].where(pv["avg_vol"] > 0)
    else:
        av = pd.Series(np.nan, index=p.index)
    out["jsf_loan_v"] = p["loan_bal"] / av
    out["jsf_stock_v"] = p["stock_bal"] / av
    out["jsf_net_v"] = (p["loan_bal"] - p["stock_bal"]) / av
    for n in (5, 20):
        out[f"jsf_loan_chg{n}_v"] = (p["loan_bal"] - g["loan_bal"].shift(n)) / av
        out[f"jsf_stock_chg{n}_v"] = (p["stock_bal"] - g["stock_bal"].shift(n)) / av
    out["jsf_long_new5_v"] = g["loan_new"].transform(lambda s: s.rolling(5, min_periods=3).sum()) / av
    out["jsf_short_new5_v"] = g["stock_new"].transform(lambda s: s.rolling(5, min_periods=3).sum()) / av
    fee = p["fee_ann"].fillna(0.0)
    out["jsf_fee"] = fee
    out["jsf_fee_days20"] = (fee > 0).astype(float).groupby(p["Code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).sum())
    out["jsf_fee_max20"] = fee.groupby(p["Code"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).max())
    out["jsf_restrict"] = p["restrict"]
    out["jsf_lendable"] = p["lendable"]
    return out


def attach(frame: pd.DataFrame, feats: pd.DataFrame, max_stale: int = MAX_STALE) -> pd.DataFrame:
    """
    取引日 T の行（Code, Date）に、申込日が T より前（T を含まない）の最新の値を付ける。
    申込日が T より max_stale 暦日以上前なら欠測。jsf_lag = T − 申込日（暦日）。
    """
    left = frame[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    right = feats.rename(columns={"app_date": "Date"}).copy()
    right["Code"] = right["Code"].astype(str)
    right["_app"] = right["Date"]
    m = pd.merge_asof(left.sort_values("Date"), right.sort_values("Date"), on="Date", by="Code",
                      direction="backward", allow_exact_matches=False,
                      tolerance=pd.Timedelta(days=max_stale - 1))
    m = m.sort_values("_i").set_index(frame.index)
    m["jsf_lag"] = (m["Date"] - m["_app"]).dt.days.astype(float)
    cols = columns("both")
    add = m[cols].astype(float)
    return pd.concat([frame.drop(columns=[c for c in cols if c in frame.columns]), add], axis=1)


def build(frame: pd.DataFrame, data_dir: str = DATA_DIR, with_daily: bool = True) -> pd.DataFrame:
    """研究用の入口: data_dir の日証金と日足から jsf_* を作って frame に付ける。"""
    hist = load_hist(os.path.join(data_dir, "jsf_hist.parquet"))
    daily = None
    if with_daily and os.path.exists(os.path.join(data_dir, "jsf_balance.parquet")):
        daily = load_daily(os.path.join(data_dir, "jsf_balance.parquet"),
                           os.path.join(data_dir, "jsf_lending.parquet"))
    p = panel(hist, daily)
    vol = avg_volume(load_bars(data_dir, codes=p["Code"].unique()))
    return attach(frame, features_on_panel(p, vol))
