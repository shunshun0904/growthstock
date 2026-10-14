#!/usr/bin/env python3
"""
実験91: 期待値の見張り（ワッサーステイン距離）を、過去の前向きのデータで確かめる（検証だけ。本番・画面は変えない）。

運用者の希望（2026-10-14）「直接的な予測というより、補助的にでも良いので、使いたいですね」。実験90 で同じ日の候補を
ワッサーステイン距離の k 近傍で選んでも効かなかった（§50）ので、当てる道具ではなく見張る道具として使う案を出し、運用者が
選んだ（AskUserQuestion）: 進める案 = 期待値の見張り / 進め方 = 過去データで確かめてから。

見張りの中身（実運用の追跡に足す形を、そのまま過去に当てる）
------------------------------------------
  行: 実験88・90 と同じ（12本の out-of-fold・前向きの百分位・テスト窓 3〜11・+10% の指値・届かなければ 20営業日目の終値
      tp10c）。規則の行 = 1.5σ の lgbm_k15・xgb_k15・cat_k15 の百分位（画面と同じ丸め）がすべて95以上
  結果が出そろう日: その銘柄の日足でブレイクの日から20本目の日（実験90 と同じ）
  時点 t: 各月末
    直近の塊 Q_t = 結果が出そろった日が t の前3か月に入る規則の行
    過去 R_t   = 結果が出そろった日が t の3か月前以前の規則の行
    全体 B_t   = t までに結果が出そろった規則の行（R_t と Q_t を合わせたもの）
  物差し（収益の分布関数 F で。単位は %pt）
    W1     = ∫|F_Q − F_R| dx（平均の差は必ずこれ以下）
    下側   = ∫max(F_Q − F_R, 0) dx（直近が過去より下に寄った量。平均の悪化は必ずこれ以下）
    上側   = ∫max(F_R − F_Q, 0) dx。W1 = 下側 + 上側、平均(R) − 平均(Q) = 下側 − 上側
    下側の内訳: 収益が −10% 未満 / −10〜0% / 0% 以上（+10% に届く分を含む）の範囲ごとの面積
  合図: 下側が、R_t から Q_t と同じ件数を復元抽出で2,000回取ったときの下側の分布の90%点を超える月末（Q_t 10件以上・
        R_t 50件以上の月末だけ判定する）
  期待値の下限の目安: 下限_t = B_t の平均 − ε_t。ε_t = t までの月末 s（合図を判定した月末）の下側の80%点。6つ未満なら出さない。
        分布が過去の普通のずれの範囲で動く限り、次の塊の平均はこれを下回らない（平均の悪化 ≤ 下側）

確かめ（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  評価する月末: B_t が50件以上・下限が出せる・次の3か月（選定日が t の翌日〜3か月後）の規則の行が10件以上。次の3か月の
      収益が出そろっていない月末は数えない
  主（下限の当たり）: 次の3か月の規則の行の平均が下限_t 以上だった月末の割合。比べる相手は、同じ月末の SE の下限
      （B_t の平均 − 0.8416 × B_t の標準偏差 ÷ √次の件数。片側80%）
      判定: W の下限の当たりが70%以上、かつ SE の下限の当たり以上 → 「期待値の下限の目安として使える」。評価できる月末が
      6つ未満なら「判定できない」
  副（合図）: 合図を判定した月末のうち次の3か月が5件以上の月末で、次の3か月の平均を合図あり・なしで比べる（差 = あり − なし）。
      並べ替え検定は、月末の並びの上で合図の印を巡回してずらす（すべてのずらし方。片側 p = ずらした差が実際以下の割合）
      判定: 差 < 0 かつ p < 0.05 → 「注意の合図として使える」。合図の月末が3つ未満、またはすべてが合図なら「判定できない」
  参考（判定に使わない）
    - 普通のぶれの幅: 月末ごとの W1・下側・上側の中央・80%点・最大、合図の出た月末の割合、下側の内訳の割合
    - 月末ごとの一覧（件数・平均・W1・下側とその内訳・合図・2つの下限・次の件数と平均）。月の単位の集計だけ
    - 塊を6か月にした形、3本の最小が85以上の行（惜しい帯から上）で同じ確かめ
    - 入力のずれ: 選定日が t の前3か月の規則の行と、それより前の規則の行の、vol_20d・ret_20d・log_market_cap・
      log_trading_value・3本の百分位の平均の W1 ÷ 過去の標準偏差。列の平均と次の3か月の平均の順位相関
  出すもの: 件数・平均・割合・月の単位の一覧。銘柄・日付ごとの値は出さない

    exp=e91_expectation_monitor.py
    args="--draws 2000"       （既定）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

RULE_MODELS = E82.T15                 # lgbm_k15・xgb_k15・cat_k15（今の規則の3本）
LINE = 95.0                           # 今の規則の線
BAND_LINE = 85.0                      # 参考: 惜しい帯から上
BLOCK = 3                             # 直近の塊・次の塊の長さ（か月）
BLOCK_REF = 6                         # 参考の塊の長さ（か月）
MIN_Q = 10                            # 合図を判定する直近の塊の最小件数
MIN_R = 50                            # 過去・全体の最小件数
MIN_NEXT = 10                         # 下限の当たりに数える次の塊の最小件数
MIN_NEXT_ALARM = 5                    # 合図の比べに数える次の塊の最小件数
MIN_EPS = 6                           # 下限の目安に要る、過去の下側の値の数
EPS_Q = 0.80                          # 下限の目安: 過去の下側の80%点
Z80 = 0.8416                          # SE の下限: 片側80%
ALARM_Q = 0.90                        # 合図: 無作為の下側の90%点を超える
COVER_LINE = 0.70                     # 主の判定: 下限の当たりがこれ以上
MIN_COVER_N = 6                       # 主の判定に要る、評価できる月末の数
HOLD = LT.HOLD                        # 20営業日
TARGET = 10.0
TP = "tp10c"
DRAWS = 2000
SEED = 42
#: (下限, 上限, 表示)。下側の内訳
RANGES = ((-np.inf, -10.0, "−10% 未満"), (-10.0, 0.0, "−10〜0%"), (0.0, np.inf, "0% 以上"))
#: (列, 表示)。入力のずれ
FEATS = (("vol_20d", "20日ボラ"), ("ret_20d", "20日上昇率"), ("log_market_cap", "時価総額（対数）"),
         ("log_trading_value", "売買代金（対数）"), ("pct15", "3本の百分位の平均"))
OUT_DIR = E79.OUT_DIR

log = E79.log
nan = float("nan")


# --------------------------------------------------------------------------- #
# 分布の比べ方
# --------------------------------------------------------------------------- #

def _finite_sorted(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sort(x[np.isfinite(x)])


def cdf_areas(q, r, ranges=RANGES) -> dict:
    """
    1次元の経験分布 q（直近）と r（過去）の分布関数の差の面積（%pt）。
      w1 = ∫|F_q − F_r|、down = ∫max(F_q − F_r, 0)（q が下に寄った量）、up = ∫max(F_r − F_q, 0)
      平均(r) − 平均(q) = down − up、w1 = down + up。down_parts は ranges の範囲ごとの down
    """
    q, r = _finite_sorted(q), _finite_sorted(r)
    if not len(q) or not len(r):
        return {"w1": nan, "down": nan, "up": nan, "down_parts": [nan] * len(ranges)}
    allv = np.sort(np.concatenate([q, r]))
    x0, x1 = allv[:-1], allv[1:]
    width = x1 - x0
    d = np.searchsorted(q, x0, side="right") / len(q) - np.searchsorted(r, x0, side="right") / len(r)
    dn, upv = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    parts = [float(np.sum(dn * np.clip(np.minimum(x1, hi) - np.maximum(x0, lo), 0.0, None))) for lo, hi, _ in ranges]
    return {"w1": float(np.sum((dn + upv) * width)), "down": float(np.sum(dn * width)), "up": float(np.sum(upv * width)),
            "down_parts": parts}


def null_down(r, n: int, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """r から n 件を復元抽出したときの down(標本, r) の分布（直近が過去と同じ分布から来たときのぶれ）。"""
    r = _finite_sorted(r)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for k in range(draws):
        out[k] = cdf_areas(r[rng.integers(0, len(r), size=n)], r, ranges=())["down"]
    return out


# --------------------------------------------------------------------------- #
# 行の準備
# --------------------------------------------------------------------------- #

def finalize_dates(bars: pd.DataFrame, keys: pd.DataFrame, hold: int = HOLD) -> np.ndarray:
    """(Code, Date) ごとに、その銘柄の日足でブレイクの日から hold 本目の日（結果が出そろう日）。無ければ NaT。"""
    n = len(keys)
    done = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    k = pd.DataFrame({"Code": keys["Code"].astype(str).to_numpy(),
                      "Date": pd.to_datetime(keys["Date"]).to_numpy(dtype="datetime64[ns]"), "_i": np.arange(n)})
    want = {c: (g["Date"].to_numpy(dtype="datetime64[ns]"), g["_i"].to_numpy()) for c, g in k.groupby("Code", sort=False)}
    b = bars[bars["Code"].astype(str).isin(want.keys())][["Code", "Date"]].copy()
    b["Code"] = b["Code"].astype(str)
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values(["Code", "Date"], kind="mergesort")
    for code, g in b.groupby("Code", sort=False):
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        ds, rows = want[code]
        i = np.searchsorted(dates, ds)
        ok = i < len(dates)
        ok[ok] = dates[i[ok]] == ds[ok]
        j = i + hold
        ok &= j < len(dates)
        done[rows[ok]] = dates[j[ok]]
    return done


def attach_forward(df: pd.DataFrame, split: dict, models=RULE_MODELS) -> pd.DataFrame:
    """テスト窓の行に、前向きの百分位（画面と同じ丸め）の 3本の最小 min15 と平均 pct15 を付ける。ほかの行は NaN。"""
    out = df.copy()
    mn = np.full(len(df), np.nan)
    av = np.full(len(df), np.nan)
    for w, (tr, te, Ptr, Pte) in split.items():
        R = E82.rounded(E79.pct_matrix(df.iloc[te], models, ref=df.iloc[tr]))
        mn[te] = R.min(axis=1)
        av[te] = R.mean(axis=1)
    out["min15"] = mn
    out["pct15"] = av
    return out


def attach_feats(df: pd.DataFrame, fr: pd.DataFrame, feats=FEATS) -> pd.DataFrame:
    """データセットの列（入力のずれを見るだけ）を足す。無い列は NaN。"""
    cols = [c for c, _ in feats if c in fr.columns and c not in df.columns]
    out = df
    if cols:
        f = fr[["Code", "Date"] + cols].copy()
        f["Code"] = f["Code"].astype(str)
        f["Date"] = pd.to_datetime(f["Date"])
        out = df.merge(f.drop_duplicates(["Code", "Date"]), on=["Code", "Date"], how="left")
    for c, _ in feats:
        if c not in out.columns and c != "pct15":
            out[c] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 見張り
# --------------------------------------------------------------------------- #

def month_ends(start, end) -> list:
    if pd.isna(start) or pd.isna(end) or pd.Timestamp(end) < pd.Timestamp(start):
        return []
    return list(pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end), freq="ME"))


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else nan


def monitor(pop: pd.DataFrame, block: int = BLOCK, draws: int = DRAWS, seed: int = SEED, ends=None,
            data_end=None) -> pd.DataFrame:
    """
    pop: 見張る行（Date・done・tp10c）。各月末 t の、直近の塊と過去の比べ（W1・下側・上側・内訳）・合図・2つの下限・
    次の塊の件数と平均の表。data_end: 収益が出そろっている最後の選定日（t + block がこれを越える月末は次の塊を数えない）。
    """
    d = pop[np.isfinite(pd.to_numeric(pop[TP], errors="coerce").to_numpy(dtype=float))]
    date = pd.to_datetime(d["Date"]).to_numpy(dtype="datetime64[ns]")
    done = pd.to_datetime(d["done"]).to_numpy(dtype="datetime64[ns]")
    tp = pd.to_numeric(d[TP], errors="coerce").to_numpy(dtype=float)
    if ends is None:
        fin = done[~np.isnat(done)]
        ends = month_ends(pd.Timestamp(fin.min()), pd.Timestamp(fin.max())) if len(fin) else []
    rows, hist = [], []
    for k, t in enumerate(ends):
        t = pd.Timestamp(t)
        t64 = np.datetime64(t, "ns")
        lo64 = np.datetime64(t - pd.DateOffset(months=block), "ns")
        hi = t + pd.DateOffset(months=block)
        isB = done <= t64                                  # NaT との比べは False
        isQ = isB & (done > lo64)
        isR = done <= lo64
        rec = {"t": t, "n_B": int(isB.sum()), "n_Q": int(isQ.sum()), "n_R": int(isR.sum()),
               "mean_B": _mean(tp[isB]), "mean_Q": _mean(tp[isQ]), "mean_R": _mean(tp[isR]),
               "sd_B": float(np.std(tp[isB], ddof=1)) if isB.sum() > 1 else nan,
               "w1": nan, "down": nan, "up": nan, "down_parts": [nan] * len(RANGES), "null_pct": nan, "alarm": nan,
               "eps": nan, "floor_w": nan, "floor_se": nan, "n_next": nan, "mean_next": nan}
        if rec["n_Q"] >= MIN_Q and rec["n_R"] >= MIN_R:
            a = cdf_areas(tp[isQ], tp[isR])
            null = null_down(tp[isR], rec["n_Q"], draws, seed + k)
            rec.update(w1=a["w1"], down=a["down"], up=a["up"], down_parts=a["down_parts"],
                       null_pct=float((null < a["down"]).mean() * 100),
                       alarm=float(a["down"] > np.quantile(null, ALARM_Q)))
            hist.append(a["down"])
        if len(hist) >= MIN_EPS and rec["n_B"] >= MIN_R:
            rec["eps"] = float(np.quantile(hist, EPS_Q))
            rec["floor_w"] = rec["mean_B"] - rec["eps"]
        if data_end is None or hi <= pd.Timestamp(data_end):
            isN = (date > t64) & (date <= np.datetime64(hi, "ns"))
            rec["n_next"] = int(isN.sum())
            rec["mean_next"] = _mean(tp[isN])
            if rec["n_next"] and np.isfinite(rec["sd_B"]):
                rec["floor_se"] = rec["mean_B"] - Z80 * rec["sd_B"] / np.sqrt(rec["n_next"])
        rows.append(rec)
    return pd.DataFrame(rows)


def coverage(tab: pd.DataFrame, min_next: int = MIN_NEXT, min_b: int = MIN_R) -> dict:
    """主の確かめ: 次の塊の平均が下限以上だった月末の割合（W の下限・SE の下限）。"""
    if tab.empty:
        return {"n": 0, "usable": False, "judgeable": False}
    m = (tab["floor_w"].notna() & tab["floor_se"].notna() & (tab["n_next"].fillna(0) >= min_next)
         & (tab["n_B"] >= min_b))
    g = tab[m]
    if g.empty:
        return {"n": 0, "usable": False, "judgeable": False}
    cw = float((g["mean_next"] >= g["floor_w"]).mean())
    cs = float((g["mean_next"] >= g["floor_se"]).mean())
    return {"n": int(len(g)), "cover_w": cw, "cover_se": cs,
            "floor_w_med": float(g["floor_w"].median()), "floor_se_med": float(g["floor_se"].median()),
            "gap_w_med": float((g["mean_next"] - g["floor_w"]).median()),
            "gap_se_med": float((g["mean_next"] - g["floor_se"]).median()),
            "eps_med": float(g["eps"].median()), "floor_w_neg": float((g["floor_w"] < 0).mean()),
            "next_mean_med": float(g["mean_next"].median()),
            "judgeable": bool(len(g) >= MIN_COVER_N),
            "usable": bool(len(g) >= MIN_COVER_N and cw >= COVER_LINE and cw >= cs)}


def alarm_test(tab: pd.DataFrame, min_next: int = MIN_NEXT_ALARM) -> dict:
    """副の確かめ: 合図あり・なしの月末の、次の塊の平均の差。合図の印を月末の並びの上で巡回してずらす（すべて）。"""
    if tab.empty:
        return {"n": 0, "n_alarm": 0, "judgeable": False, "useful": False}
    g = tab[tab["alarm"].notna() & (tab["n_next"].fillna(0) >= min_next)].reset_index(drop=True)
    a = g["alarm"].to_numpy(dtype=float) > 0.5
    y = g["mean_next"].to_numpy(dtype=float)
    n, na = int(len(g)), int(a.sum())
    res = {"n": n, "n_alarm": na, "judgeable": False, "useful": False}
    if na < 3 or na == n:
        return res
    obs = float(y[a].mean() - y[~a].mean())
    null = np.array([y[np.roll(a, s)].mean() - y[~np.roll(a, s)].mean() for s in range(1, n)], dtype=float)
    p = float((np.sum(null <= obs) + 1) / (len(null) + 1))
    res.update(judgeable=True, mean_alarm=float(y[a].mean()), mean_quiet=float(y[~a].mean()), diff=obs, p=p,
               useful=bool(obs < 0 and p < 0.05))
    return res


def input_drift(pop: pd.DataFrame, ends, block: int = BLOCK, feats=FEATS) -> pd.DataFrame:
    """月末ごとに、選定日が直前 block か月の行と、それより前の行の特徴量の W1 ÷ 過去の標準偏差。"""
    date = pd.to_datetime(pop["Date"]).to_numpy(dtype="datetime64[ns]")
    out = []
    for t in ends:
        t = pd.Timestamp(t)
        t64 = np.datetime64(t, "ns")
        lo64 = np.datetime64(t - pd.DateOffset(months=block), "ns")
        isQ = (date > lo64) & (date <= t64)
        isR = date <= lo64
        rec = {"t": t}
        vals = []
        for c, _ in feats:
            x = pd.to_numeric(pop[c], errors="coerce").to_numpy(dtype=float) if c in pop.columns else np.full(len(pop), np.nan)
            q, r = x[isQ & np.isfinite(x)], x[isR & np.isfinite(x)]
            v = nan
            if len(q) >= MIN_Q and len(r) >= MIN_R and np.std(r) > 0:
                v = cdf_areas(q, r, ranges=())["w1"] / float(np.std(r))
            rec[c] = v
            if np.isfinite(v):
                vals.append(v)
        rec["mean"] = float(np.mean(vals)) if vals else nan
        out.append(rec)
    return pd.DataFrame(out)


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return nan
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return nan
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def _q(x, q) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else nan


def fmt_cov(c: dict) -> str:
    if not c.get("n"):
        return "評価できる月末が無い"
    return (f"{c['n']}か月: W の下限の当たり {c['cover_w'] * 100:.0f}%・SE の下限の当たり {c['cover_se'] * 100:.0f}%"
            f"（下限の中央 W {c['floor_w_med']:+.2f}% / SE {c['floor_se_med']:+.2f}%・ε の中央 {c['eps_med']:.2f}pt・"
            f"W の下限が0%未満の月末 {c['floor_w_neg'] * 100:.0f}%）")


def fmt_alarm(a: dict) -> str:
    if not a.get("judgeable"):
        return f"判定できない（月末 {a.get('n', 0)}・合図 {a.get('n_alarm', 0)}）"
    return (f"月末 {a['n']}（合図 {a['n_alarm']}）: 次の3か月の平均 合図あり {a['mean_alarm']:+.2f}% / なし {a['mean_quiet']:+.2f}%・"
            f"差 {a['diff']:+.2f}pt・片側 p = {a['p']:.3f}")


def report(rule: pd.DataFrame, band: pd.DataFrame | None = None, data_end=None, draws: int = DRAWS,
           seed: int = SEED) -> dict:
    summary: dict = {}
    fin = pd.to_datetime(rule["done"]).dropna()
    ends = month_ends(fin.min(), fin.max()) if len(fin) else []
    tab = monitor(rule, BLOCK, draws, seed, ends=ends, data_end=data_end)
    ok = tab[tab["alarm"].notna()] if len(tab) else tab
    n_rows = int(np.isfinite(pd.to_numeric(rule[TP], errors="coerce")).sum())
    print("\n■ 0. 件数（今の規則・前向き）")
    print(f"  規則の行 {len(rule)}件（収益の出せる行 {n_rows}件）。月末 {len(ends)}（"
          f"{ends[0].strftime('%Y-%m') if ends else '—'}〜{ends[-1].strftime('%Y-%m') if ends else '—'}）、うち直近の塊と過去を"
          f"比べられた月末 {len(ok)}。次の塊の収益が出そろう最後の選定日 "
          f"{pd.Timestamp(data_end).strftime('%Y-%m-%d') if data_end is not None else '—'}")
    summary["counts"] = {"rows": int(len(rule)), "rows_tp": n_rows, "month_ends": len(ends), "compared": int(len(ok))}

    print(f"\n■ 1. 普通のぶれの幅（直近{BLOCK}か月 対 それより前。%pt）")
    if len(ok):
        for col, name in (("w1", "W1"), ("down", "下側"), ("up", "上側")):
            v = ok[col].to_numpy(dtype=float)
            print(f"  {LT._l(name, 6)}中央 {np.median(v):.2f}・80%点 {_q(v, 0.8):.2f}・最大 {np.max(v):.2f}")
        parts = np.array(ok["down_parts"].tolist(), dtype=float)
        tot = parts.sum(axis=1)
        share = np.where(tot[:, None] > 0, parts / np.where(tot[:, None] > 0, tot[:, None], 1.0), np.nan)
        print("  下側の内訳（月末ごとの割合の中央）: " + " / ".join(
            f"{name} {np.nanmedian(share[:, i]) * 100:.0f}%" for i, (_, _, name) in enumerate(RANGES)))
        print(f"  合図の出た月末 {int((ok['alarm'] > 0.5).sum())}/{len(ok)}")
        summary["spread"] = {c: {"median": float(ok[c].median()), "p80": _q(ok[c], 0.8), "max": float(ok[c].max())}
                             for c in ("w1", "down", "up")}
        summary["alarm_share"] = float((ok["alarm"] > 0.5).mean())
    else:
        print("  比べられた月末が無い")

    print(f"\n■ 2. 主: 期待値の下限の当たり（次の{BLOCK}か月の平均が下限以上だった月末の割合）")
    cov = coverage(tab)
    summary["main"] = cov
    print(f"  {fmt_cov(cov)}")
    if cov.get("n"):
        print(f"  次の{BLOCK}か月の平均の中央 {cov['next_mean_med']:+.2f}%。下限との差の中央 W {cov['gap_w_med']:+.2f}pt / "
              f"SE {cov['gap_se_med']:+.2f}pt")
    if not cov.get("judgeable"):
        verdict_main = f"判定できない（評価できる月末が{MIN_COVER_N}つ未満）"
    else:
        verdict_main = "期待値の下限の目安として使える" if cov.get("usable") else "下限の目安としての裏付けは無い"
    print(f"  判定: {verdict_main}（W の当たり {COVER_LINE * 100:.0f}%以上 "
          f"{'○' if cov.get('n') and cov['cover_w'] >= COVER_LINE else '×'} / SE の当たり以上 "
          f"{'○' if cov.get('n') and cov['cover_w'] >= cov['cover_se'] else '×'}）")

    print(f"\n■ 3. 副: 合図（下側が無作為の{ALARM_Q * 100:.0f}%点を超えた月末）と次の{BLOCK}か月の平均")
    at = alarm_test(tab)
    summary["alarm"] = at
    print(f"  {fmt_alarm(at)}")
    if not at.get("judgeable"):
        verdict_alarm = "判定できない"
    else:
        verdict_alarm = "注意の合図として使える" if at["useful"] else "注意の合図としての裏付けは無い"
    print(f"  判定: {verdict_alarm}")

    print("\n■ 4. 参考: 月末ごとの一覧（月の単位の集計だけ）")
    print("  月末     過去 直近  過去の平均 直近の平均   W1  下側（−10%未満/−10〜0/0以上）  上側 無作為の何%点 合図"
          "  W の下限 SE の下限 次の件数 次の平均")
    for _, r in tab.iterrows():
        if not r["n_B"]:
            continue
        p = r["down_parts"]
        parts = "/".join(f"{x:.2f}" for x in p) if np.all(np.isfinite(p)) else "—"
        alarm = "—" if not np.isfinite(r["alarm"]) else ("●" if r["alarm"] > 0.5 else "・")

        def f(v, fmt):
            return format(v, fmt) if np.isfinite(v) else "—"
        print(f"  {r['t'].strftime('%Y-%m')}  {r['n_R']:>4} {r['n_Q']:>4}  {f(r['mean_R'], '+6.2f'):>8}  "
              f"{f(r['mean_Q'], '+6.2f'):>8}  {f(r['w1'], '5.2f'):>5}  {f(r['down'], '5.2f'):>5}（{parts}）  "
              f"{f(r['up'], '5.2f'):>5}  {f(r['null_pct'], '5.1f'):>5}  {alarm}  {f(r['floor_w'], '+6.2f'):>7}  "
              f"{f(r['floor_se'], '+6.2f'):>7}  {f(r['n_next'], '.0f'):>4}  {f(r['mean_next'], '+6.2f'):>7}")

    print("\n■ 5. 参考: 塊の長さと見張る行を変えた形（判定に使わない）")
    summary["ref"] = {}
    variants = [("rule6", f"今の規則・{BLOCK_REF}か月の塊", rule, BLOCK_REF)]
    if band is not None and len(band):
        variants += [("band3", f"惜しい帯から上（3本の最小{BAND_LINE:.0f}以上）・{BLOCK}か月", band, BLOCK),
                     ("band6", f"惜しい帯から上・{BLOCK_REF}か月", band, BLOCK_REF)]
    for key, name, pop, blk in variants:
        fp = pd.to_datetime(pop["done"]).dropna()
        e2 = month_ends(fp.min(), fp.max()) if len(fp) else []
        t2 = monitor(pop, blk, draws, seed, ends=e2, data_end=data_end)
        c2, a2 = coverage(t2), alarm_test(t2)
        summary["ref"][key] = {"main": c2, "alarm": a2}
        print(f"  {name}")
        print(f"    下限: {fmt_cov(c2)}")
        print(f"    合図: {fmt_alarm(a2)}")

    print(f"\n■ 6. 参考: 入力のずれ（選定日が直近{BLOCK}か月の規則の行 対 それより前。W1 ÷ 過去の標準偏差）")
    dr = input_drift(rule, ends)
    summary["input"] = {}
    for c, name in FEATS:
        v = dr[c].to_numpy(dtype=float) if c in dr.columns else np.array([])
        if np.isfinite(v).any():
            print(f"  {LT._l(name, 20)}中央 {np.nanmedian(v):.2f}・80%点 {_q(v, 0.8):.2f}・最大 {np.nanmax(v):.2f}")
            summary["input"][c] = {"median": float(np.nanmedian(v)), "p80": _q(v, 0.8), "max": float(np.nanmax(v))}
        else:
            print(f"  {LT._l(name, 20)}—")
    if len(dr) and len(tab):
        m = tab[["t", "mean_next", "n_next"]].merge(dr[["t", "mean"]], on="t", how="inner")
        m = m[m["n_next"].fillna(0) >= MIN_NEXT_ALARM]
        rho = spearman(m["mean"], m["mean_next"])
        summary["input"]["rho_next"] = rho
        print(f"  列の平均のずれと次の{BLOCK}か月の平均の順位相関: {rho:+.3f}（{len(m)}か月）" if np.isfinite(rho)
              else "  列の平均のずれと次の平均の順位相関: —")

    print("\n■ まとめ")
    print(f"  主: {verdict_main}（{fmt_cov(cov)}）")
    print(f"  副: {verdict_alarm}（{fmt_alarm(at)}）")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。月末の評価は3か月ずつ重なるので独立ではない。手数料・税・すべりは含まない")
    summary["verdict"] = {"main": verdict_main, "alarm": verdict_alarm}
    summary["table"] = tab.assign(t=tab["t"].dt.strftime("%Y-%m")).to_dict(orient="records") if len(tab) else []
    return summary


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験91: 期待値の見張り（ワッサーステイン距離）を過去の前向きのデータで確かめる")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    df = attach_feats(df, fr)
    del fr
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=(TARGET,))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行")
    df["done"] = finalize_dates(bars, df[["Code", "Date"]])
    del bars
    split = E79.window_split(df, folds)
    df = attach_forward(df, split)
    tpv = pd.to_numeric(df[TP], errors="coerce")
    data_end = pd.to_datetime(df.loc[tpv.notna(), "Date"]).max()
    rule = df[df["min15"] >= LINE].reset_index(drop=True)
    band = df[df["min15"] >= BAND_LINE].reset_index(drop=True)
    log(f"今の規則の行 {len(rule)}件（実験88・90 は 559件）・惜しい帯から上 {len(band)}件・収益の出そろう最後の選定日 "
        f"{data_end.date()}")
    summary = report(rule, band, data_end=data_end, draws=args.draws)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e91_summary.json")

    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=conv)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
