#!/usr/bin/env python3
"""
決算開示を「その開示の時点で見えていた値」に組み直す（point-in-time / as-of）。

## なぜ専用に作るのか

research/build_dataset.py の quarterize_panel() は
`drop_duplicates(..., keep="last")` で **その期の最終訂正値** を採る。
最新時点の分析には正しいが、過去8期の系列を作るには使えない。
2020年1Qの数値が2021年に訂正されていた場合、2020年1Q時点のグラフに
2021年の情報が混ざる。これは未来情報のリークである。

ここでは各開示を基準日として、その日までに開示された版だけで系列を組む。

## 組み方

  1. 開示の「版」表を作る（同じ期の訂正も別の行として残す）
  2. 銘柄ごとに会計期間（CurPerEn）を並べ、スロット番号を振る
  3. 各開示を基準（アンカー）として、スロットを L 期さかのぼった行を並べる
  4. merge_asof で「基準日以前に開示された最新の版」を引く

アンカーは各期の **最初の開示** に限る。訂正開示そのものを予測対象にすると、
同じ決算に対して複数のサンプルができてしまうため。訂正は履歴側にだけ効く。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import schema

#: グラフ系列の長さ（四半期）。要件のフェーズ1は過去8四半期。
SEQ_LEN = 8

#: 前年同期比を作るには、系列の一番古い期からさらに4期前が要る。
#: さらに累計→単期の差分展開に1期前が要るので、合計で +5 期さかのぼる。
EXTRA_LAGS = 5
N_LAGS = SEQ_LEN + EXTRA_LAGS

#: 会計期間の並びから「前年同期」「前四半期」と認めてよい日数の幅。
#: 決算期変更（ChgFYEnd）で期の長さが変わることがあるため、
#: 単純に4つ前・1つ前を信用せず、期末日の間隔で確かめる。
YOY_DAYS = (300, 430)
QOQ_DAYS = (60, 130)

#: /fins/summary から読む項目。
RAW_FIELDS: Tuple[str, ...] = tuple(
    sorted(set(schema.CUMULATIVE_FIELDS) | set(schema.STOCK_FIELDS))
)
#: 通期の会社予想。進捗乖離に使う。
FORECAST_FIELDS: Tuple[str, ...] = tuple(
    sorted({n.forecast_field for n in schema.NODES if n.forecast_field})
)

_QUARTER_MAP = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}


# --------------------------------------------------------------------------- #
# 版の表
# --------------------------------------------------------------------------- #

def prepare_versions(fins: pd.DataFrame) -> pd.DataFrame:
    """
    /fins/summary の生データから「開示の版」表を作る。

    訂正開示も落とさずに残す。どの版が基準日時点で最新だったかは
    あとで merge_asof が決める。
    """
    need = {"Code", "DiscDate", "CurPerType", "CurPerEn", "CurFYSt"}
    missing = need - set(fins.columns)
    if missing:
        raise ValueError(f"/fins/summary に必要な列がありません: {sorted(missing)}")

    df = fins.copy()
    df["Code"] = df["Code"].astype(str).str.strip()
    df["quarter"] = df["CurPerType"].astype(str).str.strip().map(_QUARTER_MAP)
    df["per_end"] = pd.to_datetime(df["CurPerEn"], errors="coerce")
    df["fy_start"] = df["CurFYSt"].astype(str).str.strip()
    disc_date = pd.to_datetime(df["DiscDate"], errors="coerce")
    df["disc_date"] = disc_date

    # 同じ日に訂正が出ることがあるので、時刻まで見て順序を決める。
    # 時刻が無い行は 00:00 とみなす（同日なら訂正が後に来るとは限らないが、
    # 同日内の順序はどちらを採っても未来情報にはならない）。
    t = pd.to_timedelta(
        df.get("DiscTime", pd.Series("", index=df.index)).astype(str).str.strip(),
        errors="coerce",
    ).fillna(pd.Timedelta(0))
    df["disc_ts"] = disc_date + t

    for col in RAW_FIELDS + FORECAST_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 実績値を持つ開示のみ。業績予想の修正だけの開示は会計フローを持たない。
    has_actual = df[list(schema.CUMULATIVE_FIELDS) + list(schema.STOCK_FIELDS)].notna().any(axis=1)
    before = len(df)
    df = df[df["quarter"].notna() & df["per_end"].notna()
            & df["disc_date"].notna() & has_actual].copy()
    df["quarter"] = df["quarter"].astype(int)

    cols = (["Code", "quarter", "fy_start", "per_end", "disc_date", "disc_ts"]
            + list(RAW_FIELDS) + list(FORECAST_FIELDS))
    df = df[cols].sort_values(["Code", "per_end", "disc_ts"]).reset_index(drop=True)
    print(f"[panel] 開示の版: {len(df):,}行（生データ {before:,}行から実績のみ抽出）")
    return df


def period_slots(versions: pd.DataFrame) -> pd.DataFrame:
    """
    銘柄ごとに会計期間を期末日の順に並べ、スロット番号を振る。

    期の同一性は (Code, per_end) で決める。通期開示（FY）と第4四半期開示は
    同じ期末日を指すので、ここで同じスロットに入る。
    """
    per = (versions.groupby(["Code", "per_end"], as_index=False)
           .agg(quarter=("quarter", "first"), fy_start=("fy_start", "first")))
    per = per.sort_values(["Code", "per_end"]).reset_index(drop=True)
    per["slot"] = per.groupby("Code").cumcount()
    return per


def anchors(versions: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """
    予測の基準となる開示イベント（各期の最初の開示）を取り出す。
    """
    first = (versions.sort_values(["Code", "per_end", "disc_ts"])
             .groupby(["Code", "per_end"], as_index=False).first())
    a = first.merge(periods[["Code", "per_end", "slot"]], on=["Code", "per_end"],
                    how="left", validate="one_to_one")
    a = a.sort_values(["disc_date", "Code"]).reset_index(drop=True)
    a["anchor_id"] = np.arange(len(a), dtype=np.int64)
    print(f"[panel] アンカー（各期の最初の開示）: {len(a):,}件 "
          f"{a['disc_date'].min().date()}〜{a['disc_date'].max().date()}")
    return a[["anchor_id", "Code", "per_end", "slot", "quarter", "fy_start",
              "disc_date", "disc_ts"]]


# --------------------------------------------------------------------------- #
# as-of 展開
# --------------------------------------------------------------------------- #

def asof_lags(anchor_df: pd.DataFrame, versions: pd.DataFrame,
              periods: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    各アンカーについて、L 期前（L = 0..n_lags-1）の会計期間の値を
    「アンカーの開示時点で見えていた版」で引く。

    返すのは (anchor_id, lag) をキーとする縦持ちの表。
    該当する期が存在しない（上場が浅い）場合は値がすべて欠測になる。
    """
    slot_map = periods.set_index(["Code", "slot"])["per_end"]

    parts = []
    for lag in range(n_lags):
        p = anchor_df[["anchor_id", "Code", "slot", "disc_ts"]].copy()
        p["lag"] = lag
        p["target_slot"] = p["slot"] - lag
        idx = pd.MultiIndex.from_arrays([p["Code"], p["target_slot"]])
        p["per_end"] = slot_map.reindex(idx).to_numpy()
        parts.append(p.drop(columns=["slot", "target_slot"]))
    left = pd.concat(parts, ignore_index=True)
    left["per_end"] = pd.to_datetime(left["per_end"])

    # 期が存在しない行は as-of の対象にならない。分けておく
    ok = left["per_end"].notna()
    todo = left[ok].sort_values("disc_ts").reset_index(drop=True)

    right = versions.sort_values("disc_ts").reset_index(drop=True)
    value_cols = list(RAW_FIELDS) + list(FORECAST_FIELDS)
    right_small = right[["Code", "per_end", "disc_ts", "quarter", "fy_start",
                         "disc_date"] + value_cols].copy()
    right_small = right_small.rename(columns={"disc_date": "src_disc_date"})
    for c in value_cols:
        right_small[c] = right_small[c].astype("float64")

    merged = pd.merge_asof(
        todo, right_small,
        left_on="disc_ts", right_on="disc_ts",
        by=["Code", "per_end"], direction="backward", allow_exact_matches=True,
    )

    # 期が無かった行を戻して、(anchor_id, lag) が必ず n_lags 本そろうようにする。
    # 列の型を merged に合わせておく（合わせないと concat が dtype を推定し直し、
    # 数値列が object になって以降の計算が黙って壊れる）
    blanks = left[~ok].copy()
    for c in ["quarter", "src_disc_date"] + value_cols:
        blanks[c] = pd.Series(np.nan, index=blanks.index,
                              dtype=merged[c].dtype if c in merged else "float64")
    blanks["fy_start"] = pd.Series("", index=blanks.index, dtype=object)
    out = pd.concat([merged, blanks], ignore_index=True) if len(blanks) else merged
    out = out.sort_values(["anchor_id", "lag"]).reset_index(drop=True)

    n_expect = len(anchor_df) * n_lags
    if len(out) != n_expect:
        raise AssertionError(
            f"as-of 展開の行数が合いません: {len(out):,} != {n_expect:,}。"
            "同じ (Code, per_end, disc_ts) の版が重複している可能性があります"
        )
    return out


# --------------------------------------------------------------------------- #
# 縦持ち → [アンカー, ラグ] の行列
# --------------------------------------------------------------------------- #

def _matrix(long: pd.DataFrame, col: str, n_anchors: int, n_lags: int,
            dtype=np.float64) -> np.ndarray:
    """(anchor_id, lag) 順に並んだ縦持ちを [n_anchors, n_lags] に畳む。"""
    v = long[col].to_numpy()
    if v.dtype == object or str(v.dtype).startswith("datetime"):
        v = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy()
    return v.astype(dtype).reshape(n_anchors, n_lags)


def _shift_str(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, "", dtype=a.dtype)
    if k < a.shape[1]:
        out[:, :a.shape[1] - k] = a[:, k:]
    return out


def single_period(cum: np.ndarray, quarter: np.ndarray, fy: np.ndarray,
                  per_end_days: np.ndarray, max_back: int = 3):
    """
    累計値 [n_anchors, n_lags] を「1四半期あたりの金額」と「対象期数」に分ける。

    ## なぜ単純な前四半期との差ではだめか

    キャッシュフローは 2Q と 通期 しか開示されない（実測で 1Q 9.9% / 3Q 8.2%、
    docs/DATA_FIELDS.md）。前四半期との差を取る作りにすると、

      2Q  … 1Q の累計が無いので差が取れない → 欠測
      FY  … 3Q の累計が無いので差が取れない → 欠測

    となり、**CF ノードが全期間まるごと欠測になる**。実際そうなっていた。

    そこで「同じ会計年度内で、値を持つ直近の先行開示」との差を取る。

      2Q（1Qが無い） … 累計そのもの = 上期の値、対象期数 2
      FY（3Qが無い） … FY - 2Q = 下期の値、対象期数 2
      2Q（1Qがある） … 2Q - 1Q = 当四半期の値、対象期数 1

    値は対象期数で割って「1四半期あたり」に揃える。揃えないと、
    上期の営業CFと単一四半期の売上高を同じ土俵で比べることになる。
    対象期数そのものも特徴量として持たせるので、
    「これは半期を2で割った値だ」という情報は失われない。
    """
    val = cum.copy()
    # 同一年度に先行する開示が無ければ、累計そのものが当期までの値。
    # その場合の対象期数は四半期番号（2Q なら2期ぶん）
    span = quarter.astype(np.float64).copy()
    found = np.zeros(cum.shape, dtype=bool)

    for k in range(1, max_back + 1):
        prev = shift_lag(cum, k)
        prev_q = shift_lag(quarter, k)
        prev_fy = _shift_str(fy, k)
        prev_end = shift_lag(per_end_days, k)
        step = quarter - prev_q
        gap = per_end_days - prev_end
        cand = (~found & ~np.isnan(cum) & ~np.isnan(prev)
                & (prev_fy == fy) & (step > 0)
                # 期末日の間隔が期数と釣り合っていること。
                # 決算期変更で期の長さが変わった行を混ぜない
                & (gap >= QOQ_DAYS[0] * step * 0.7)
                & (gap <= QOQ_DAYS[1] * step))
        val = np.where(cand, cum - prev, val)
        span = np.where(cand, step, span)
        found |= cand

    val = np.where(np.isnan(cum), np.nan, val)
    span = np.where(np.isnan(val) | (span <= 0), np.nan, span)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_quarter = val / span
    return per_quarter, span


def _sym_change(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """
    対称変化率 (cur - prev) / (|cur| + |prev|)。値域は -1〜+1。

    通常の成長率は前期が0以下だと定義できず、赤字企業をまるごと落とす。
    赤字→黒字の転換は決算で最も株価が動く場面なので、そこを欠測にしない。
    """
    denom = np.abs(cur) + np.abs(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, (cur - prev) / denom, np.nan)
    return np.where(np.isnan(cur) | np.isnan(prev), np.nan, out)


def shift_lag(a: np.ndarray, k: int) -> np.ndarray:
    """
    lag 軸を k だけ古い側にずらす（lag L の位置に lag L+k の値を置く）。

    はみ出す側（最も古いラグ）は欠測で埋める。ここを 0 で埋めると
    「履歴が無い」と「値が 0」が区別できなくなる。
    """
    if a.dtype.kind != "f":
        raise TypeError(f"浮動小数の行列にのみ使えます: dtype={a.dtype}")
    out = np.full_like(a, np.nan)
    if k < a.shape[1]:
        out[:, :a.shape[1] - k] = a[:, k:]
    return out


def build_asof_matrices(anchor_df: pd.DataFrame, versions: pd.DataFrame,
                        periods: pd.DataFrame,
                        n_lags: int = N_LAGS) -> Dict[str, np.ndarray]:
    """
    as-of 展開から、以降の特徴量計算に必要な行列一式を作る。

    返す辞書の各値は [n_anchors, n_lags] の行列。
      q_<FIELD>    … 1四半期あたりに直したフロー項目
      span_<FIELD> … その値が何四半期ぶんか（CF は 2 になることが多い）
      s_<FIELD>    … 期末残高（そのまま）
      cum_<FIELD>  … 累計（進捗率の計算に使う）
      f_<FIELD>   … 通期の会社予想
      quarter, per_end_days, available
    """
    long = asof_lags(anchor_df, versions, periods, n_lags=n_lags)
    n = len(anchor_df)

    quarter = _matrix(long, "quarter", n, n_lags)
    fy = long["fy_start"].fillna("").to_numpy().reshape(n, n_lags)

    # 期末日を「日数」に直す。期の間隔（前年同期か・前四半期か）の判定に使う。
    # NaT を int64 にすると最小値になって巨大な間隔に見えるので、欠測へ戻す
    pe = pd.to_datetime(long["per_end"])
    per_end_days = (pe.astype("int64").to_numpy() / 86_400_000_000_000.0
                    ).reshape(n, n_lags)
    per_end_days = np.where(pe.isna().to_numpy().reshape(n, n_lags),
                            np.nan, per_end_days)

    out: Dict[str, np.ndarray] = {
        "quarter": quarter,
        "per_end_days": per_end_days,
        # その期の値がアンカー時点で1つでも見えていたか
        "available": (~long["src_disc_date"].isna().to_numpy()).reshape(n, n_lags),
    }

    for f in schema.CUMULATIVE_FIELDS:
        cum = _matrix(long, f, n, n_lags)
        out[f"cum_{f}"] = cum
        out[f"q_{f}"], out[f"span_{f}"] = single_period(
            cum, quarter, fy, per_end_days)
    for f in schema.STOCK_FIELDS:
        v = _matrix(long, f, n, n_lags)
        out[f"s_{f}"] = v
        # 期末残高は「その時点の値」なので対象期数は常に1期ぶん
        out[f"span_{f}"] = np.where(np.isnan(v), np.nan, 1.0)
    for f in FORECAST_FIELDS:
        out[f"f_{f}"] = _matrix(long, f, n, n_lags)

    return out


def yoy_valid(per_end_days: np.ndarray) -> np.ndarray:
    """lag L と lag L+4 が前年同期の関係にあるか。"""
    gap = per_end_days - shift_lag(per_end_days, 4)
    return (gap >= YOY_DAYS[0]) & (gap <= YOY_DAYS[1])


def qoq_valid(per_end_days: np.ndarray) -> np.ndarray:
    """lag L と lag L+1 が前四半期の関係にあるか。"""
    gap = per_end_days - shift_lag(per_end_days, 1)
    return (gap >= QOQ_DAYS[0]) & (gap <= QOQ_DAYS[1])


def sym_change_lag(values: np.ndarray, k: int, valid: np.ndarray) -> np.ndarray:
    """lag L と lag L+k の対称変化率。valid が False の位置は欠測。"""
    prev = shift_lag(values, k)
    out = _sym_change(values, prev)
    return np.where(valid, out, np.nan)
