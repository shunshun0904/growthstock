#!/usr/bin/env python3
"""
2026-09-22 に取り込みへ足した8本から特徴量を作る。

出どころは J-Quants スタンダードで使えるのに叩いていなかったエンドポイント
（`docs/DATA_FIELDS.md` の実測）。**1本ずつ検証せず、全部入れてから
3モデル×5分割チューニング＋OOF で一括評価する**（運用者の判断）。

  valuation    /equities/valuation              銘柄×日
  lvshld       /edinet/large-volume-shareholders 大量保有報告書（提出日）
  mjrshld      /edinet/major-shareholders        大株主（提出日、年1回）
  xhold        /edinet/cross-shareholdings       政策保有（提出日、年1回）
  shortratio   /markets/short-ratio              市場全体（日次）
  marginalert  /markets/margin-alert             銘柄×公表日
  earndate     /fins/earnings-date               次の決算予定（公表日）
  investor     /equities/investor-types          市場全体（週次）

時点整合の決め事
--------------
**「その行をいつ知りえたか」で結合する。** EDINET は提出日（SubDate）、
信用規制は公表日（PubDate）、決算予定は公表日（PubDate）。
merge_asof の backward で、その日までに出ているものだけを見る。

決算予定の SchDate は**中央値64日先**なので、これを結合キーにすると
未来を見る。実測で確かめてある（jq_bulk.DAILY_KINDS のコメント）。

値が無ければ NaN のまま。0 で埋めない（実測値と欠測を混ぜない）。
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

#: 信託銀行の名寄せ。大株主に占める信託口は、指数連動・機関投資家の保有の代理。
#: 個人（創業家）の保有と分けたいので、名前で見分ける
TRUST_PAT = re.compile(r"信託|カストディ|マスタートラス|ＴＨＥ　ＢＡＮＫ|STATE STREET|"
                       r"JPMORGAN|BNY|NORTHERN TRUST|SSBTC", re.I)
#: 法人らしさ。これに当たらない大株主を個人とみなす（粗い代理）
CORP_PAT = re.compile(r"株式会社|\(株\)|（株）|有限会社|合同会社|財団|組合|"
                      r"CO\.|LTD|INC|CORP|LLC|GMBH|S\.A\.|N\.V\.|LIMITED|COMPANY|"
                      r"BANK|TRUST|FUND|PARTNERS|CAPITAL|HOLDINGS", re.I)


def _read(kind: str, data_dir: str = DATA_DIR,
          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """kind_YYYY.parquet か kind.parquet を読む。無ければ空。"""
    paths = sorted(glob.glob(os.path.join(data_dir, f"{kind}_[0-9]*.parquet")))
    one = os.path.join(data_dir, f"{kind}.parquet")
    if not paths and os.path.exists(one):
        paths = [one]
    if not paths:
        return pd.DataFrame()
    out = []
    for p in paths:
        try:
            out.append(pd.read_parquet(p, columns=columns) if columns
                       else pd.read_parquet(p))
        except Exception:                                    # noqa: BLE001
            out.append(pd.read_parquet(p))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _asof(left: pd.DataFrame, right: pd.DataFrame, right_on: str,
          by_code: bool = True, exact: bool = True) -> pd.DataFrame:
    """
    samples の (Code, Date) に、その日までに出ている right を付ける。

    **行の順序は元のまま返す。** 並べ替えたまま返すと、呼び出し側で
    列を代入したときに静かにずれる。
    """
    l = left.copy()
    l["_i"] = np.arange(len(l))
    l["Date"] = pd.to_datetime(l["Date"])
    l = l.sort_values("Date")
    r = right.copy()
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce")
    r = r.dropna(subset=[right_on]).sort_values(right_on)
    kw = {"by": "Code"} if by_code else {}
    m = pd.merge_asof(l, r, left_on="Date", right_on=right_on,
                      direction="backward", allow_exact_matches=exact, **kw)
    return m.sort_values("_i").drop(columns=["_i"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. バリュエーション（銘柄×日。当日の値がそのまま使える）
# --------------------------------------------------------------------------- #

#: 自前計算と重なる列。どちらを採るかは実験38 の実測で決める
VALUATION_OVERLAP = ["PER", "PBR", "ROE"]
#: 自前に対応が無い列
VALUATION_NEW = ["FwdEPS", "FwdPER", "FwdROE"]


def valuation(samples: pd.DataFrame, data_dir: str = DATA_DIR,
              use_overlap: bool = False) -> pd.DataFrame:
    """
    予想ベースの指標を付ける。

    use_overlap=True のときだけ PER/PBR/ROE も持ってくる。既定で持たないのは
    自前計算と二重になるため（運用者の指示で**両方は持たない**）。
    """
    v = _read("valuation", data_dir)
    if not len(v):
        return pd.DataFrame(index=samples.index)
    v["Code"] = v["Code"].astype(str)
    cols = VALUATION_NEW + (VALUATION_OVERLAP if use_overlap else [])
    cols = [c for c in cols if c in v.columns]
    if not cols:
        return pd.DataFrame(index=samples.index)
    v = v[["Code", "Date"] + cols].copy()
    v["Date"] = pd.to_datetime(v["Date"])
    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"])
    m = left.merge(v, on=["Code", "Date"], how="left")

    out = pd.DataFrame(index=samples.index)
    for c in cols:
        out[f"jq_{c.lower()}"] = _num(m[c]).to_numpy()

    # **API の ROE / FwdROE は小数**（0.0791 = 7.91%）。
    # 実測（2026-09-22、valuation_2025）で FwdEPS/BPS の中央値 0.0791 が
    # API の FwdROE 0.0791 と一致することを確かめた。自前の ROE_q0 は
    # % 単位（中央値 5.71 = 5.71%）なので、ここで % に揃える。
    #
    # 揃えずに引き算していたため jq_roe_gap が実質 -ROE_q0 になり、
    # ROE_q0 との相関が **-0.9999** という、ほぼ同じ列が2本ある状態に
    # なっていた（実験40 の冗長検出が拾った）。
    for c in ("jq_roe", "jq_fwdroe"):
        if c in out.columns:
            out[c] = out[c] * 100.0

    # 予想と実績の差。株価が織り込んでいる「これからの伸び」
    if "FwdEPS" in cols and "close_raw" in samples.columns:
        px = _num(samples["close_raw"]).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["jq_fwd_earnings_yield"] = np.where(
                px > 0, _num(m["FwdEPS"]).to_numpy() / px * 100.0, np.nan)
    if "FwdROE" in cols and "ROE_q0" in samples.columns:
        # 上で % に揃えた out["jq_fwdroe"] を使う。m["FwdROE"] は小数のまま
        out["jq_roe_gap"] = (out["jq_fwdroe"].to_numpy()
                             - _num(samples["ROE_q0"]).to_numpy())
    if "FwdPER" in cols and "per" in samples.columns:
        per = _num(samples["per"]).to_numpy()
        fwd = _num(m["FwdPER"]).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["jq_per_gap"] = np.where((per > 0) & (fwd > 0), per / fwd, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 2. 大量保有報告書（提出日で結合。「誰かが5%超を買い増した」を拾う）
# --------------------------------------------------------------------------- #

LVS_WINDOWS = (20, 60, 250)


def large_volume(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    直近の大量保有報告書の保有比率・前回比と、過去N日の提出回数。

    `TotalShsRatioLast` があるので「買い増したのか減らしたのか」が分かる。
    報告義務は5%超の取得・1%以上の増減なので、**提出そのものがイベント**。
    """
    d = _read("lvshld", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or "SubDate" not in d.columns:
        return out
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    d["SubDate"] = pd.to_datetime(d["SubDate"], errors="coerce")
    d = d.dropna(subset=["SubDate", "Code"])
    for c in ("TotalShsRatio", "TotalShsRatioLast", "TotalShsHeld"):
        if c in d.columns:
            d[c] = _num(d[c])
    d = d.sort_values("SubDate")

    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    keep = ["Code", "SubDate"] + [c for c in ("TotalShsRatio", "TotalShsRatioLast")
                                  if c in d.columns]
    m = _asof(left, d[keep].rename(columns={"SubDate": "LvsDate"}), "LvsDate")
    out["lvs_days"] = (pd.to_datetime(m["Date"]) - m["LvsDate"]).dt.days.to_numpy()
    if "TotalShsRatio" in m.columns:
        out["lvs_ratio"] = _num(m["TotalShsRatio"]).to_numpy() * 100.0
        if "TotalShsRatioLast" in m.columns:
            out["lvs_ratio_chg"] = (out["lvs_ratio"]
                                    - _num(m["TotalShsRatioLast"]).to_numpy() * 100.0)

    # 過去N営業日の提出回数。日数ではなく暦日で数える（提出は休日も届く）
    ev = d[["Code", "SubDate"]].assign(_one=1)
    for w in LVS_WINDOWS:
        cnt = _count_in_window(samples, ev, "SubDate", w)
        out[f"lvs_n_{w}"] = cnt
    return out


def _count_in_window(samples: pd.DataFrame, events: pd.DataFrame,
                     date_col: str, days: int) -> np.ndarray:
    """(Code, Date) ごとに、過去 days 暦日に起きた events の件数。"""
    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    ev = events.copy()
    ev["Code"] = ev["Code"].astype(str)
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce")
    ev = ev.dropna(subset=[date_col]).sort_values(date_col)
    # 累積件数の差で数える: [Date-days, Date] の件数 = C(Date) - C(Date-days-1)
    ev = ev.assign(_c=ev.groupby("Code").cumcount() + 1)
    hi = pd.merge_asof(left.sort_values("Date"), ev[["Code", date_col, "_c"]],
                       left_on="Date", right_on=date_col, by="Code",
                       direction="backward", allow_exact_matches=True)
    lo_left = left.sort_values("Date").assign(
        _lo=lambda x: x["Date"] - pd.Timedelta(days=days))
    lo = pd.merge_asof(lo_left.sort_values("_lo"), ev[["Code", date_col, "_c"]],
                       left_on="_lo", right_on=date_col, by="Code",
                       direction="backward", allow_exact_matches=True)
    a = hi.sort_values("_i")["_c"].to_numpy()
    b = lo.sort_values("_i")["_c"].to_numpy()
    a = np.where(np.isfinite(a), a, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    return a - b


# --------------------------------------------------------------------------- #
# 3. 大株主（有報ベース。年1回だが、誰が持っているかは構造的な情報）
# --------------------------------------------------------------------------- #

def _holders(cell) -> List[dict]:
    """Hldrs は list of dict。

    **parquet から読み戻すと numpy.ndarray で来る。** pyarrow の list 型は
    ndarray に復元されるため、list/tuple だけを見ていると全部取りこぼす。
    実測（2026-09-22、mjrshld 77,730行を取り込んだ直後）で mjr_* 7列が
    **全欠測**になった。データは1行も欠けていないのに 0% だった。

    文字列で保存された版も読めるようにしてある（取り込み経路によっては
    入れ子が str 化されることがある）。
    """
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, (list, tuple)):
        return [h for h in cell if isinstance(h, dict)]
    if isinstance(cell, str) and cell.strip().startswith("["):
        import ast as _ast
        try:
            v = _ast.literal_eval(cell)
            return [h for h in v if isinstance(h, dict)]
        except (ValueError, SyntaxError):
            return []
    return []


def _holder_stats(cell) -> dict:
    """
    大株主10名から、保有の集中度と主体の内訳を出す。

    ShsRatio は小数（0.0919 = 9.19%）。合計が1を超えることは無いはずだが、
    自己株の扱いで前後するので丸めない。
    """
    hs = _holders(cell)
    if not hs:
        return {}
    rat, trust, indiv = [], 0.0, 0.0
    for h in hs:
        r = pd.to_numeric(h.get("ShsRatio"), errors="coerce")
        if not np.isfinite(r):
            continue
        rat.append(float(r))
        name = str(h.get("HldrName") or "")
        if TRUST_PAT.search(name):
            trust += float(r)
        elif not CORP_PAT.search(name):
            indiv += float(r)
    if not rat:
        return {}
    rat = np.array(rat, dtype=float)
    return {"mjr_top1": rat.max() * 100.0,
            "mjr_top10": rat.sum() * 100.0,
            "mjr_n": float(len(rat)),
            # 上位の偏り。1位が突出しているか、横並びか
            "mjr_conc": float(rat.max() / rat.sum()) if rat.sum() > 0 else np.nan,
            "mjr_trust": trust * 100.0,
            "mjr_indiv": indiv * 100.0}


MJR_COLS = ["mjr_top1", "mjr_top10", "mjr_n", "mjr_conc", "mjr_trust", "mjr_indiv"]


def major_holders(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """直近の有報の大株主10名から、集中度・信託比率・個人比率。"""
    d = _read("mjrshld", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or "Hldrs" not in d.columns or "SubDate" not in d.columns:
        return out
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    d["SubDate"] = pd.to_datetime(d["SubDate"], errors="coerce")
    d = d.dropna(subset=["SubDate", "Code"]).sort_values("SubDate")
    stats = pd.DataFrame([_holder_stats(c) for c in d["Hldrs"]], index=d.index)
    d = pd.concat([d[["Code", "SubDate"]], stats], axis=1)
    d = d.rename(columns={"SubDate": "MjrDate"})

    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    m = _asof(left, d, "MjrDate")
    out["mjr_days"] = (pd.to_datetime(m["Date"]) - m["MjrDate"]).dt.days.to_numpy()
    for c in MJR_COLS:
        if c in m.columns:
            out[c] = _num(m[c]).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 4. 政策保有株（売却はガバナンス改善のシグナルとして読まれる）
# --------------------------------------------------------------------------- #

def _xh(cell, key) -> float:
    """Report は dict。_holders と同じ理由で、来うる形を広めに受ける。"""
    if isinstance(cell, np.ndarray) and cell.size == 1:
        cell = cell.item()
    if isinstance(cell, str) and cell.strip().startswith("{"):
        import ast as _ast
        try:
            cell = _ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return np.nan
    if not isinstance(cell, dict):
        return np.nan
    return pd.to_numeric(cell.get(key), errors="coerce")


XH_COLS = ["xh_iss", "xh_bookval", "xh_dec_amt", "xh_inc_cost", "xh_net"]


def cross_holdings(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """上場株の政策保有の銘柄数・簿価と、その期の増減。"""
    d = _read("xhold", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or "Report" not in d.columns or "SubDate" not in d.columns:
        return out
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    d["SubDate"] = pd.to_datetime(d["SubDate"], errors="coerce")
    d = d.dropna(subset=["SubDate", "Code"]).sort_values("SubDate")
    rep = d["Report"]
    d["xh_iss"] = [_xh(c, "ListedIss") for c in rep]
    d["xh_bookval"] = [_xh(c, "ListedBookVal") for c in rep]
    d["xh_dec_amt"] = [_xh(c, "ListedDecSaleAmt") for c in rep]
    d["xh_inc_cost"] = [_xh(c, "ListedIncAcqCost") for c in rep]
    d["xh_net"] = d["xh_inc_cost"] - d["xh_dec_amt"]
    d = d[["Code", "SubDate"] + XH_COLS].rename(columns={"SubDate": "XhDate"})

    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    m = _asof(left, d, "XhDate")
    out["xh_days"] = (pd.to_datetime(m["Date"]) - m["XhDate"]).dt.days.to_numpy()
    for c in XH_COLS:
        out[c] = _num(m[c]).to_numpy()
    # 時価総額との比。規模で割らないと大型株ばかり大きく出る。
    # 接尾辞は `_mc`。`_r` はこのリポジトリで「同日内の順位」を表す
    # 予約語なので使わない（features.RAW_FOR_RANK / RANKED_GROUPS）
    if "market_cap" in samples.columns:
        mc = _num(samples["market_cap"]).to_numpy() * 1e8      # 億円 -> 円
        with np.errstate(divide="ignore", invalid="ignore"):
            out["xh_bookval_mc"] = np.where(mc > 0, out["xh_bookval"] / mc * 100.0,
                                           np.nan)
            out["xh_net_mc"] = np.where(mc > 0, out["xh_net"] / mc * 100.0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 5. 信用規制（規制がかかると値動きの性質が変わる）
# --------------------------------------------------------------------------- #

def margin_alert(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """直近の信用規制の公表からの日数と、そのときの残高の比率。"""
    d = _read("marginalert", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or "PubDate" not in d.columns:
        return out
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    d["PubDate"] = pd.to_datetime(d["PubDate"], errors="coerce")
    d = d.dropna(subset=["PubDate", "Code"]).sort_values("PubDate")
    num_cols = [c for c in ("LongOutRatio", "ShrtOutRatio", "SLRatio")
                if c in d.columns]
    keep = ["Code", "PubDate"] + num_cols
    m = _asof(samples[["Code", "Date"]].assign(Code=lambda x: x["Code"].astype(str)),
              d[keep].rename(columns={"PubDate": "AlertDate"}), "AlertDate")
    out["alert_days"] = (pd.to_datetime(m["Date"]) - m["AlertDate"]).dt.days.to_numpy()
    for c in num_cols:
        out[f"alert_{c.lower()}"] = _num(m[c]).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 6. 次の決算まで何日か（days_since_disc の裏返し）
# --------------------------------------------------------------------------- #

def earnings_ahead(samples: pd.DataFrame, data_dir: str = DATA_DIR,
                   clip: int = 120) -> pd.DataFrame:
    """
    その時点で**公表済みの**予定から、次の決算発表まで何日か。

    予定日（SchDate）は中央値64日先。公表日（PubDate）で「その日までに
    知っていた予定」だけに絞ったうえで、Date より後の SchDate の最小を取る。
    ここを間違えると未来を見る。
    """
    d = _read("earndate", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or not {"PubDate", "SchDate", "Code"} <= set(d.columns):
        return out
    d = d.copy()
    d["Code"] = d["Code"].astype(str)
    for c in ("PubDate", "SchDate"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d.dropna(subset=["PubDate", "SchDate", "Code"])

    left = samples[["Code", "Date"]].copy()
    left["Code"] = left["Code"].astype(str)
    left["Date"] = pd.to_datetime(left["Date"])
    left["_i"] = np.arange(len(left))
    # 銘柄ごとに、公表済み かつ 当日以降 の予定の最小を取る
    j = left.merge(d[["Code", "PubDate", "SchDate"]], on="Code", how="left")
    ok = (j["PubDate"] <= j["Date"]) & (j["SchDate"] >= j["Date"])
    nxt = j[ok].groupby("_i")["SchDate"].min()
    sch = pd.Series(np.nan, index=left["_i"].to_numpy(), dtype="datetime64[ns]")
    sch.loc[nxt.index] = nxt.to_numpy()
    days = (sch.to_numpy() - left["Date"].to_numpy()) / np.timedelta64(1, "D")
    out["days_to_earn"] = np.clip(days.astype(float), 0, clip)
    return out


# --------------------------------------------------------------------------- #
# 7〜8. 市場全体（空売り比率・投資部門別）。銘柄ごとではなく日付ごと
# --------------------------------------------------------------------------- #

def short_ratio(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """市場全体の空売り比率（実測では S33=9999 の1系列だけ）。"""
    d = _read("shortratio", data_dir)
    out = pd.DataFrame(index=samples.index)
    need = {"Date", "SellExShortVa", "ShrtWithResVa", "ShrtNoResVa"}
    if not len(d) or not need <= set(d.columns):
        return out
    d = d.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    if "S33" in d.columns:                       # 全市場の行だけ使う
        d = d[_num(d["S33"]) == 9999]
    d = d.dropna(subset=["Date"]).sort_values("Date")
    tot = _num(d["SellExShortVa"]) + _num(d["ShrtWithResVa"]) + _num(d["ShrtNoResVa"])
    with np.errstate(divide="ignore", invalid="ignore"):
        d["short_ratio"] = np.where(tot > 0,
                                    (_num(d["ShrtWithResVa"])
                                     + _num(d["ShrtNoResVa"])) / tot * 100.0, np.nan)
    d["short_ratio_20"] = d["short_ratio"] - d["short_ratio"].rolling(20).mean()
    keep = d[["Date", "short_ratio", "short_ratio_20"]].rename(
        columns={"Date": "SrDate"})
    m = _asof(samples[["Code", "Date"]], keep, "SrDate", by_code=False)
    for c in ("short_ratio", "short_ratio_20"):
        out[c] = _num(m[c]).to_numpy()
    return out


#: 投資部門別で見る主体。Bal = 差引（買い − 売り）
INVESTOR_GROUPS = {"Frgn": "inv_foreign", "InvTr": "inv_trust", "Ind": "inv_indiv",
                   "Prop": "inv_prop", "BusCo": "inv_busco"}


def investor_types(samples: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    市場全体の投資部門別売買。週次なので、その週までに出ているものを使う。

    金額そのものではなく**売買代金に対する差引の比**にする。規模が年々
    変わるので、生の金額だと年代の代理になってしまう。
    """
    d = _read("investor", data_dir)
    out = pd.DataFrame(index=samples.index)
    if not len(d) or "EnDate" not in d.columns:
        return out
    d = d.copy()
    d["EnDate"] = pd.to_datetime(d["EnDate"], errors="coerce")
    d = d.dropna(subset=["EnDate"])
    # 市場区分ごとに行があるなら、日付でまとめる（全市場の合計にする）
    num = [c for c in d.columns if d[c].dtype != object and c != "EnDate"]
    d = d.groupby("EnDate", as_index=False)[num].sum(min_count=1).sort_values("EnDate")
    tot = sum((_num(d.get(f"{g}Buy", 0)) + _num(d.get(f"{g}Sell", 0)))
              for g in INVESTOR_GROUPS)
    keep = pd.DataFrame({"InvDate": d["EnDate"]})
    for g, name in INVESTOR_GROUPS.items():
        bal = _num(d.get(f"{g}Bal"))
        if bal is None or not bal.notna().any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            keep[name] = np.where(tot > 0, bal / tot * 100.0, np.nan)
        keep[f"{name}_4w"] = pd.Series(keep[name]).rolling(4).mean().to_numpy()
    if keep.shape[1] <= 1:
        return out
    m = _asof(samples[["Code", "Date"]], keep, "InvDate", by_code=False)
    for c in keep.columns:
        if c == "InvDate":
            continue
        out[c] = _num(m[c]).to_numpy()
    return out


#: build_dataset から呼ぶ順。戻り値はすべて samples と同じ行数・同じ並び
BUILDERS = [
    ("valuation", valuation),
    ("lvshld", large_volume),
    ("mjrshld", major_holders),
    ("xhold", cross_holdings),
    ("marginalert", margin_alert),
    ("earndate", earnings_ahead),
    ("shortratio", short_ratio),
    ("investor", investor_types),
]


def expected_columns() -> List[str]:
    """
    このモジュールが作るべき列の一覧。正本は features.GROUPS。

    build_dataset は features.all_columns() の全列が揃っていることを
    要求する（揃っていなければ SystemExit）。取り込みが届いていない種別を
    「列ごと作らない」にすると、**データセットの構築そのものが落ちる**。
    実際それで日次予測が止まる状態になった（2026-09-22）。
    """
    import features as _F
    mine = ("fwd", "holders_lvs", "holders_major", "holders_cross",
            "margin_alert", "earn_ahead", "flow")
    return [c for g in mine for c in _F.GROUPS.get(g, ())]


def attach(samples: pd.DataFrame, data_dir: str = DATA_DIR,
           verbose: bool = True) -> pd.DataFrame:
    """
    8本ぶんの特徴量を付けた列を返す（samples と同じ行数・同じ並び）。

    1本が落ちても他は作る。**取り込みがまだ届いていない種別は全欠測の列**に
    なる。0 では埋めない。

    列ごと落とさないのは、データセットの列構成を取り込みの進み具合で
    変えないため。欠測は「まだ分からない」であって、列が無いのとは違う。
    木モデルは全欠測の列を無視するだけなので害は無い。
    """
    parts = []
    for name, fn in BUILDERS:
        try:
            part = fn(samples, data_dir)
        except Exception as exc:                             # noqa: BLE001
            if verbose:
                print(f"[warn] extra_features {name} を作れませんでした（続行）: "
                      f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        if part is None or not part.shape[1]:
            if verbose:
                print(f"[extra] {name:<12} 列なし（取り込み待ち）")
            continue
        if len(part) != len(samples):
            if verbose:
                print(f"[warn] {name}: 行数が合わない "
                      f"({len(part)} != {len(samples)})。捨てる")
            continue
        parts.append(part)
        if verbose:
            cov = part.notna().mean().mean() * 100
            print(f"[extra] {name:<12} {part.shape[1]:>3}列  平均充足 {cov:>5.1f}%")
    out = (pd.concat(parts, axis=1) if parts
           else pd.DataFrame(index=samples.index))
    # 届いていない種別ぶんを全欠測で埋め、列構成を常に同じにする
    want = expected_columns()
    absent = [c for c in want if c not in out.columns]
    if absent:
        out = pd.concat(
            [out, pd.DataFrame(np.nan, index=samples.index, columns=absent)],
            axis=1)
        if verbose:
            print(f"[extra] 取り込み待ちの {len(absent)}列は全欠測で置く"
                  f"（列を落とすとデータセットの構築が落ちる）")
    return out[want]
