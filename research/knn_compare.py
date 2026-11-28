#!/usr/bin/env python3
"""
候補の比較（似た過去の事例）: 日次予測の候補ごとに、ワッサーステイン距離の k 近傍（近い50件の過去の事例）の収益の分布を
集計して、画面の「候補の比較」タブ（src/views/KnnCompareView.jsx）に渡す（public/data/predictions.json の knn）。

運用者の指示（2026-10-15）「別タブとして、左から2番目の位置に追加して下さい。3枚で良いです」（実験97 の模型の3枚:
候補の位置（散布図）・似た過去の事例の収益の幅・近傍50件の「距離 × 収益」。docs/MODEL_ADOPTION_RULES.md §57・§58）。

**参考の表示であって、選定には使わない。** 近傍の25%点で選ぶ後工程（実験93、§53）は同じ期間のデータでは先に決めた判定を
通ったが裏付けは弱く（§53・§55）、前向きの確かめ直し（実験94、§54。判定は 2027-10-15 以降）が通るまでは、今日の戦略の
並べ方（規則の3本の最小。2026-10-30 から 1.2σ の3本。2026-10-23〜10-29 も同じ。2026-10-29〜10-30 は 1.5σ の3本）のまま。

作り（実験91・93・97 と同じ。research/exp/e91_wasserstein_knn_pick.py。下の関数は同じ式を本番に写したもので、
tests/test_knn_compare.py が実験の関数と同じ値になることを確かめる）
  分布: ブレイクの日を含む直近60営業日の、日次リターン（調整後終値の前日比）を平均0・標準偏差1にそろえた分布（R）と、
        log(その日の出来高 ÷ その前の60営業日の平均出来高)（V）。それぞれ60点の分位関数で持ち、W1 = 分位関数の差の絶対値の平均
  距離: W1(R)/sR + W1(V)/sV。s は母集団から無作為に取った2,000組の W1 の中央値（尺度をそろえるだけ。収益は使わない）
  母集団（惜しい帯から上）: 今の規則の3本（RULE_MODELS。2026-10-30 から 1.2σ の lgbm・xgb・cat。2026-10-23〜10-29 も同じ。
        実験91・93・97 と 2026-10-23 より前・2026-10-29〜10-30 の画面は 1.5σ の lgbm_k15・xgb_k15・cat_k15）の out-of-fold の行のうち、画面と同じ百分位（それぞれの
        out-of-fold 全体の分布での位置。live_track.pct_of）の最小が85以上で、結果が出そろった行（買った日を1日目として
        20営業日目の日足が、予測日より前）。結果 = 翌営業日の寄りで買い、+10% の指値に届けば +10、届かなければ
        20営業日目の終値（live_track.forward。今日の戦略の売りの規則と同じ）
  見積もり: 距離の近い50件の収益の 10/25/50/75/90% 点、+10% 到達と −10% 割れの割合、近傍までの距離の成分の平均、
        「距離 × 収益の帯」の度数（雲）。その日の候補すべてに付ける（画面で「規則を満たす候補 / 惜しい候補まで / 全候補」を選ぶ）
  文脈（散布図の背景）: 母集団のうち最小が95以上の行（過去の候補）に、その行の日より前に結果が出そろった母集団で同じ見積もりを
        付けた分布。25%点の十分位の境と、25%点 × −10% 割れの2次元の度数（MIN_CELL 件未満のマス目は消す）
  過去の日: その日に出した値のまま（predict_daily.freeze_past_days と同じ考え方。前の predictions.json の knn.byDay を引き継ぐ）。
        文脈と尺度は毎回作り直す
出すのは集計だけ。過去の事例の銘柄・日付は出さない（候補の銘柄コードは predictions.json の candidates と同じもの）。
作れなくても日次予測は止めない（build は例外を外に出さず、その日を「出せない」として残す）。
"""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_track as LT  # noqa: E402

RULE_MODELS = tuple(LT.RULE_MODELS)   # 今の規則の3本（2026-10-30 から lgbm・xgb・cat。画面の strategy.js の RULE_MODELS と同じ）
POOL_LINE = 85.0                      # 近傍の母集団: 規則の3本の最小が85以上（惜しい帯から上）
CONTEXT_LINE = LT.AGREE_PCT           # 文脈（過去の候補）: 今の規則の線 95
K = 50                                # 近傍の数
MIN_POOL = 2 * K                      # 母集団がこれに満たない日は見積もりを出さない
WIN = 60                              # 分布を取る営業日数（ブレイクの日を含む）
MIN_OBS = 50                          # そのうち値が要る日数
VOL_BASE = 60                         # 出来高の比率の分母（その前の60営業日の平均）
LEVELS = 60                           # 分位関数を取る点の数
HOLD = LT.HOLD                        # 20営業日（結果が出そろう日）
TARGET = LT.TAKE_PROFIT               # +10% の指値
N_PAIRS = 2000                        # 尺度をそろえる中央値に使う組の数
SEED = 42
D_BINS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]                     # 雲の横（距離）の刻み
BANDS = ["-10 未満", "-10〜0", "0〜+5", "+5〜+10 未満", "+10 到達"]       # 雲の縦（収益の帯）
Y_BANDS = [-40.0, -10.0, 0.0, 5.0, TARGET - 1e-9, TARGET + 0.01]
Y_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 100.0]                        # 文脈の縦（近傍の −10% 割れの割合）の刻み
MIN_CELL = 5                          # 文脈の2次元の度数で消す閾（公開ログ・公開ファイルの方針と同じ）
BARS_BACK_DAYS = 250                  # 日足を読み始める日（母集団の最初の日のこれだけ前。60 + 60 営業日ぶんの余裕）
VERSION = 1


def log(msg: str) -> None:
    print(f"[knn] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 分布と W1（実験91 と同じ式）
# --------------------------------------------------------------------------- #

def probs(levels: int = LEVELS) -> np.ndarray:
    return (np.arange(levels) + 0.5) / levels


def quantile_vector(x, levels: int = LEVELS) -> np.ndarray:
    """経験分布の分位関数を確率 (j+0.5)/levels で取る（値がちょうど levels 個なら、並べ替えた値そのもの）。"""
    xs = np.sort(np.asarray(x, dtype=float))
    idx = np.minimum((probs(levels) * len(xs)).astype(int), len(xs) - 1)
    return xs[idx]


def w1(q, Q) -> np.ndarray:
    """分位関数 q（levels,）と Q（n × levels）の W1（1次元の1次ワッサーステイン距離 = 分位関数の差の絶対値の平均）。"""
    return np.abs(np.asarray(Q, dtype=float) - np.asarray(q, dtype=float)).mean(axis=-1)


def _col(g: pd.DataFrame, main: str, fallback: str) -> np.ndarray:
    """調整後の列を優先し、欠けていれば素の列で埋める（build_dataset と同じ扱い）。"""
    a = pd.to_numeric(g[main], errors="coerce") if main in g.columns else pd.Series(np.nan, index=g.index)
    if fallback in g.columns:
        a = a.fillna(pd.to_numeric(g[fallback], errors="coerce"))
    return a.to_numpy(dtype=float)


def event_distributions(bars: pd.DataFrame, keys: pd.DataFrame, win: int = WIN, base: int = VOL_BASE,
                        min_obs: int = MIN_OBS, levels: int = LEVELS, hold: int = HOLD) -> dict:
    """
    keys の (Code, Date) ごとに、その日を含む直近 win 営業日の
      QR: 日次リターン（調整後終値の前日比）を平均0・標準偏差1にそろえた分布の分位関数（形だけ）
      QV: log(その日の出来高 ÷ その前の base 営業日の平均出来高) の分布の分位関数
    を返す（keys と同じ並び。値が min_obs 日に満たない・標準偏差が0 なら NaN）。
    done: 結果が出そろう日 = その銘柄の日足で i + hold 本目の日（買った日 i+1 を1日目として hold 日目）。無ければ NaT。
    """
    n = len(keys)
    QR = np.full((n, levels), np.nan)
    QV = np.full((n, levels), np.nan)
    done = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    k = pd.DataFrame({"Code": keys["Code"].astype(str).to_numpy(),
                      "Date": pd.to_datetime(keys["Date"]).to_numpy(dtype="datetime64[ns]"),
                      "_i": np.arange(n)})
    want = {c: (g["Date"].to_numpy(dtype="datetime64[ns]"), g["_i"].to_numpy()) for c, g in k.groupby("Code", sort=False)}
    b = bars[bars["Code"].astype(str).isin(want.keys())].copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"], kind="mergesort")
    min_base = int(round(base * 0.8))
    for code, g in b.groupby("Code", sort=False):
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        close = _col(g, "AdjC", "C")
        vol = _col(g, "AdjVo", "Vo")
        ret = np.full(len(close), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[1:] = close[1:] / close[:-1] - 1.0
        v = pd.Series(np.where(vol > 0, vol, np.nan))
        prev = v.shift(1).rolling(base, min_periods=min_base).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(v.to_numpy() / prev)
        nd = len(dates)
        ds, rows = want[code]
        for d, r_i in zip(ds, rows):
            i = int(np.searchsorted(dates, d))
            if i >= nd or dates[i] != d:
                continue
            if i + hold < nd:
                done[r_i] = dates[i + hold]
            lo = i - win + 1
            if lo < 1:
                continue
            r = ret[lo:i + 1]
            r = r[np.isfinite(r)]
            if len(r) >= min_obs:
                sd = r.std()
                if sd > 0:
                    QR[r_i] = quantile_vector((r - r.mean()) / sd, levels)
            x = lr[lo:i + 1]
            x = x[np.isfinite(x)]
            if len(x) >= min_obs:
                QV[r_i] = quantile_vector(x, levels)
    return {"QR": QR, "QV": QV, "done": done}


def pair_median(Q: np.ndarray, n_pairs: int = N_PAIRS, seed: int = SEED) -> float:
    """Q の行から無作為に取った組（行が少なければ全部の組）の W1 の中央値。2行未満なら NaN。"""
    n = len(Q)
    if n < 2:
        return float("nan")
    if n * (n - 1) // 2 <= n_pairs:
        a, b = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        a = rng.integers(0, n, size=n_pairs)
        b = rng.integers(0, n - 1, size=n_pairs)
        b = b + (b >= a)                                   # a と違う行
    return float(np.median(np.abs(Q[a] - Q[b]).mean(axis=1)))


def done_before(done_sorted: np.ndarray, day) -> int:
    """結果が出そろう日の昇順の配列で、day より前に出そろった行の数（同じ日に出そろう行は含まない）。"""
    return int(np.searchsorted(done_sorted, np.datetime64(pd.Timestamp(day), "ns"), side="left"))


def scale_of(Q: np.ndarray, seed: int) -> float:
    """尺度（無作為な2件の距離の中央値）。出せなければ 1.0（割らない）。"""
    s = pair_median(Q, N_PAIRS, seed)
    return s if np.isfinite(s) and s > 0 else 1.0


# --------------------------------------------------------------------------- #
# 母集団: out-of-fold の行（惜しい帯から上）と、その結果
# --------------------------------------------------------------------------- #

def load_pool(model_dir: str, oof_dir: Optional[str] = None, models: Sequence[str] = RULE_MODELS,
              pool_line: float = POOL_LINE) -> pd.DataFrame:
    """
    規則の3本の out-of-fold を (Code, Date) でそろえ、画面と同じ百分位（それぞれの out-of-fold 全体の分布での位置）の最小
    min15 を付けて、pool_line 以上の行を返す。列: Code, Date, min15。out-of-fold が無ければ FileNotFoundError。
    """
    oof_dir = oof_dir or model_dir
    base = None
    for a in models:
        path = LT.find_oof(a, oof_dir, model_dir)
        if path is None:
            raise FileNotFoundError(f"{a} の out-of-fold がありません（{oof_dir} / {model_dir}）")
        o = pd.read_parquet(path, columns=["Code", "Date", "score"])
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        hist = o["score"].to_numpy(dtype=float)                   # 画面の物差し（models.hist_scores と同じ全行）
        o = o[np.isfinite(hist)].drop_duplicates(["Code", "Date"], keep="first").copy()
        o[f"p_{a}"] = LT.pct_of(o["score"].to_numpy(dtype=float), hist)
        o = o.drop(columns=["score"])
        base = o if base is None else base.merge(o, on=["Code", "Date"], how="inner")
    base["min15"] = base[[f"p_{a}" for a in models]].min(axis=1)
    pool = base[base["min15"] >= pool_line][["Code", "Date", "min15"]].reset_index(drop=True)
    return pool


def load_bars(data_dir: str, codes: Iterable[str], start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    日足（live_track.load_bars と同じ列 + 素の終値・出来高）。codes の銘柄だけ読む（母集団と候補の銘柄だけで足りる。
    全銘柄を読むと数年ぶんで数百 MB になる）。start の年より前のファイルは読まない。
    """
    codes = sorted({str(c) for c in codes})
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if start is not None:
        paths = [p for p in paths
                 if os.path.basename(p)[5:9].isdigit() and int(os.path.basename(p)[5:9]) >= start.year]
    if not paths:
        raise FileNotFoundError(f"日足（bars_*.parquet）がありません: {data_dir}")
    import pyarrow.parquet as pq

    need = ["Date", "Code", "AdjO", "AdjH", "AdjC"]
    optional = ["C", "AdjVo", "Vo"]          # 素の終値・出来高（調整後が欠けた日の埋め合わせ。無いファイルは NaN）
    parts = []
    for p in paths:
        have = set(pq.read_schema(p).names)
        cols = need + [c for c in optional if c in have]
        part = pd.read_parquet(p, columns=cols, filters=[("Code", "in", codes)])
        for c in optional:
            if c not in part.columns:
                part[c] = np.nan
        parts.append(part)
    b = pd.concat(parts, ignore_index=True)
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    return b


def attach_outcomes(pool: pd.DataFrame, bars: pd.DataFrame, target: float = TARGET, hold: int = HOLD) -> pd.DataFrame:
    """母集団の行に収益 tp（+target に届けば target、届かなければ hold 営業日目の終値。%）と到達 hit を付け、出そろった行だけ返す。"""
    if pool.empty:
        return pool.assign(tp=np.nan, hit=False)
    fw = LT.forward(bars, pool[["Code", "Date"]], target=target, hold=hold, fallback="close")
    out = pool.merge(fw[["Code", "Date", "ret", "status"]], on=["Code", "Date"], how="left")
    out["tp"] = pd.to_numeric(out["ret"], errors="coerce")
    out["hit"] = out["status"].eq(LT.HIT)
    out = out[np.isfinite(out["tp"].to_numpy(dtype=float))].drop(columns=["ret", "status"])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 近傍の集計
# --------------------------------------------------------------------------- #

def neighbor_summary(D: np.ndarray, tp: np.ndarray, hit: np.ndarray, dR: np.ndarray, dV: np.ndarray, k: int = K) -> dict:
    """距離 D の近い k 件の収益の分位点・割合、距離の成分の平均、雲（距離 × 収益の帯）の度数。"""
    kk = int(min(k, len(D)))
    nn = np.argpartition(D, kk - 1)[:kk]
    v, hv, dn = tp[nn], hit[nn], D[nn]
    counts = np.zeros((len(D_BINS) - 1, len(BANDS)), dtype=int)
    ix = np.clip(np.searchsorted(D_BINS, np.clip(dn, D_BINS[0], D_BINS[-1] - 1e-9), side="right") - 1, 0, len(D_BINS) - 2)
    iy = np.clip(np.searchsorted(Y_BANDS, np.clip(v, Y_BANDS[0], Y_BANDS[-1] - 1e-9), side="right") - 1, 0, len(Y_BANDS) - 2)
    iy = np.where(hv, len(BANDS) - 1, iy)
    for a, b in zip(ix, iy):
        counts[a, b] += 1
    return {"est": {"p10": round(float(np.quantile(v, 0.10)), 2), "p25": round(float(np.quantile(v, 0.25)), 2),
                    "p50": round(float(np.median(v)), 2), "p75": round(float(np.quantile(v, 0.75)), 2),
                    "p90": round(float(np.quantile(v, 0.90)), 2),
                    "hit": round(float(hv.mean() * 100), 1), "bad": round(float((v < -10).mean() * 100), 1), "n": int(kk)},
            "dNn": {"R": round(float(dR[nn].mean()), 3), "V": round(float(dV[nn].mean()), 3)},
            "cloud": counts.tolist()}


def context_of(p25: np.ndarray, bad: np.ndarray, min_cell: int = MIN_CELL) -> Optional[dict]:
    """過去の候補の見積もりの分布（散布図の背景）: 25%点の十分位の境と、25%点 × −10% 割れの2次元の度数（min_cell 未満は消す）。"""
    ok = np.isfinite(p25) & np.isfinite(bad)
    p25, bad = p25[ok], bad[ok]
    if len(p25) < min_cell:
        return None
    xe = [float(np.quantile(p25, q)) for q in np.linspace(0, 1, 7)]
    ix = np.clip(np.searchsorted(xe, p25, side="right") - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(Y_EDGES, bad, side="right") - 1, 0, len(Y_EDGES) - 2)
    counts = np.zeros((len(xe) - 1, len(Y_EDGES) - 1), dtype=int)
    for a, b in zip(ix, iy):
        counts[a, b] += 1
    small = (counts > 0) & (counts < min_cell)
    return {"n": int(len(p25)),
            "p25Deciles": [round(float(np.quantile(p25, q)), 2) for q in np.linspace(0, 1, 11)],
            "density": {"xEdges": [round(v, 2) for v in xe], "yEdges": Y_EDGES, "counts": np.where(small, 0, counts).tolist(),
                        "suppressedCells": int(small.sum()), "suppressedRows": int(counts[small].sum())}}


def candidate_min15(row: dict, models: Sequence[str] = RULE_MODELS) -> Optional[float]:
    """候補の規則の3本の最小（画面の strategy.js の minPct と同じ。1本でも欠けていれば None）。名前の min15 は 1.5σ の3本だったころのまま。"""
    per = row.get("byModel") or {}
    vals = []
    for a in models:
        v = (per.get(a) or {}).get("pctHistorical")
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    return min(vals)


def header(days: Sequence[str]) -> dict:
    return {"version": VERSION, "k": K, "poolLine": POOL_LINE, "contextLine": CONTEXT_LINE, "minPool": MIN_POOL,
            "target": TARGET, "hold": HOLD, "win": WIN, "models": list(RULE_MODELS),
            "dBins": D_BINS, "bands": BANDS, "minCell": MIN_CELL, "context": None, "byDay": {},
            "note": (f"参考の表示。候補ごとに、60日のリターンの形と出来高の比率が似た過去の事例（{LT.RULE_JA}の最小が"
                     f"{POOL_LINE:g}以上で結果が出そろった行から近い50件）の収益の分布。選定には使わない（今日の戦略の並べ方のまま。"
                     "実験93・94）。"
                     "過去の日は、その日に出した値のまま")}


def carry_over(prev: Optional[dict], days: Sequence[str], frozen: Iterable[str], reason: Optional[str] = None) -> dict:
    """前の knn（無ければ空）から、固定する日の分だけ引き継いだ骨組み。採点した日は reason があれば「出せない」として置く。"""
    frozen = set(frozen)
    out = header(days)
    prev_by_day = (prev or {}).get("byDay") or {}
    prev_cx = (prev or {}).get("context")
    for d in days:
        if d in frozen and d in prev_by_day:
            out["byDay"][d] = prev_by_day[d]
        elif d not in frozen and reason:
            out["byDay"][d] = {"status": "unavailable", "reason": reason}
    if prev_cx:
        out["context"] = prev_cx
    return out


def compute(rows: List[dict], days: Sequence[str], data_dir: str, model_dir: str, oof_dir: Optional[str] = None,
            k: int = K, min_pool: int = MIN_POOL, pool_line: float = POOL_LINE, context_line: float = CONTEXT_LINE) -> dict:
    """
    days（採点した日。古い順）の候補 rows（predict_daily の行。jqCode・date・byModel）に近傍の集計を付ける。
    戻り値: {"context", "byDay": {日: {...}}, "poolN", "scale"}。例外はそのまま外へ（build が受ける）。
    """
    t0 = time.time()
    days = [str(d) for d in days]
    pool = load_pool(model_dir, oof_dir, pool_line=pool_line)
    cand_rows = [x for x in rows if x.get("date") in set(days)]
    codes = set(pool["Code"].astype(str)) | {str(x["jqCode"]) for x in cand_rows}
    first = min([pd.Timestamp(d) for d in days] + ([pool["Date"].min()] if len(pool) else []))
    bars = load_bars(data_dir, codes, start=first - pd.Timedelta(days=BARS_BACK_DAYS))
    pool = attach_outcomes(pool, bars)
    dist = event_distributions(bars, pool[["Code", "Date"]])
    ok = np.isfinite(dist["QR"]).all(axis=1) & np.isfinite(dist["QV"]).all(axis=1) & ~np.isnat(dist["done"])
    pool = pool[ok].reset_index(drop=True)
    QR, QV, done = dist["QR"][ok], dist["QV"][ok], dist["done"][ok]
    order = np.argsort(done, kind="mergesort")
    pool, QR, QV, done = pool.iloc[order].reset_index(drop=True), QR[order], QV[order], done[order]
    tp = pool["tp"].to_numpy(dtype=float)
    hit = pool["hit"].to_numpy(dtype=bool)
    min15 = pool["min15"].to_numpy(dtype=float)
    pdates = pd.to_datetime(pool["Date"]).to_numpy(dtype="datetime64[ns]")
    d0 = pd.Timestamp(days[0])
    n_base = done_before(done, d0)
    sR = scale_of(QR[:n_base], SEED) if n_base >= 2 else 1.0
    sV = scale_of(QV[:n_base], SEED + 100) if n_base >= 2 else 1.0
    out = {"poolN": int(n_base), "scale": {"R": round(sR, 4), "V": round(sV, 4)}, "context": None, "byDay": {}}
    log(f"母集団（最小 {pool_line:.0f}以上・結果が出そろった行）{n_base:,}件（{d0.date()} より前）・"
        f"尺度 リターンの形 {sR:.4f}・出来高の比率 {sV:.4f}")

    # 文脈: 過去の候補（最小が context_line 以上の行）に、その行の日より前の母集団で同じ見積もりを付ける
    cx_rows = np.where((min15[:n_base] >= context_line))[0]
    p25s, bads = [], []
    for i in cx_rows:
        n_i = done_before(done, pdates[i])
        if n_i < min_pool:
            continue
        D = w1(QR[i], QR[:n_i]) / sR + w1(QV[i], QV[:n_i]) / sV
        kk = int(min(k, n_i))
        nn = np.argpartition(D, kk - 1)[:kk]
        v = tp[nn]
        p25s.append(float(np.quantile(v, 0.25)))
        bads.append(float((v < -10).mean() * 100))
    out["context"] = context_of(np.array(p25s, dtype=float), np.array(bads, dtype=float))
    log(f"文脈（過去の候補 最小 {context_line:.0f}以上）{len(p25s)}件" + (
        f"・消したマス目 {out['context']['density']['suppressedCells']}" if out["context"] else "・背景は出せない"))

    for d in days:
        today = [x for x in cand_rows if x["date"] == d]
        n_d = done_before(done, pd.Timestamp(d))
        blk = {"status": "ok", "poolN": int(n_d), "n": 0, "candidates": {}}
        if not today:
            blk["status"] = "no_candidates"
            out["byDay"][d] = blk
            continue
        if n_d < min_pool:
            blk["status"] = "pool_small"
            out["byDay"][d] = blk
            continue
        keys = pd.DataFrame({"Code": [str(x["jqCode"]) for x in today], "Date": [pd.Timestamp(d)] * len(today)})
        cd = event_distributions(bars, keys)
        P_tp, P_hit, P_QR, P_QV = tp[:n_d], hit[:n_d], QR[:n_d], QV[:n_d]
        for j, x in enumerate(today):
            m = candidate_min15(x)
            if m is None or not (np.isfinite(cd["QR"][j]).all() and np.isfinite(cd["QV"][j]).all()):
                continue
            dR = w1(cd["QR"][j], P_QR) / sR
            dV = w1(cd["QV"][j], P_QV) / sV
            rec = neighbor_summary(dR + dV, P_tp, P_hit, dR, dV, k)
            rec["min15"] = round(m, 1)
            blk["candidates"][str(x["jqCode"])] = rec
        blk["n"] = len(blk["candidates"])
        out["byDay"][d] = blk
        log(f"{d}: 候補 {len(today)}件のうち {blk['n']}件に近傍の集計（母集団 {n_d:,}件）")
    log(f"{time.time() - t0:.0f}秒")
    return out


def build(rows: List[dict], days: Sequence[str], data_dir: str, model_dir: str, prev: Optional[dict] = None,
          frozen: Iterable[str] = (), oof_dir: Optional[str] = None, **kw) -> dict:
    """
    predictions.json の knn を作る。days は画面に出す日（古い順）、frozen はそのうち前の値のまま固定する日
    （predict_daily.freeze_past_days の frozen）。固定する日は prev（前の knn）の byDay を引き継ぎ、残りの日を集計する。
    失敗しても例外を出さず、採点した日を「出せない」（status: unavailable）として返す（日次予測を止めない）。
    """
    frozen = set(frozen)
    scored = [d for d in days if d not in frozen]
    out = carry_over(prev, days, frozen)
    if not scored:
        return out
    try:
        got = compute(rows, scored, data_dir, model_dir, oof_dir, **kw)
    except Exception as exc:                     # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        log(f"近傍の集計を作れないので、採点した日は出せない: {reason}")
        return carry_over(prev, days, frozen, reason=reason[:300])
    out["context"] = got["context"]
    out["poolN"], out["scale"] = got["poolN"], got["scale"]
    for d in scored:
        out["byDay"][d] = got["byDay"].get(d) or {"status": "no_candidates", "poolN": got["poolN"], "n": 0, "candidates": {}}
        out["byDay"][d]["scale"] = got["scale"]
    return out


def summary_line(knn: Optional[dict], days: Sequence[str]) -> str:
    """ログ用の1行。"""
    if not knn:
        return "近傍の集計なし"
    parts = []
    for d in days:
        b = (knn.get("byDay") or {}).get(d)
        if not b:
            parts.append(f"{d} なし")
        elif b.get("status") == "ok":
            parts.append(f"{d} {b.get('n', 0)}件（母集団 {b.get('poolN', 0):,}）")
        else:
            parts.append(f"{d} {b.get('status')}")
    cx = knn.get("context")
    return "・".join(parts) + (f"／文脈 {cx['n']}件" if cx else "／文脈なし")
