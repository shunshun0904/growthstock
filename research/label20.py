#!/usr/bin/env python3
"""
ラベルの地平を60営業日から20営業日へ変えたときに何が入れ替わるかを測る。

なぜ変えるのか
--------------
出口の実測（docs/EXIT_TIMING.md）で、TOPIX を上回る超過リターンは
20〜30営業日でピークを打ち、それ以降は消えていた。優位性は約1ヶ月で尽きる。
ところがラベルは「3ヶ月後も維持しているか」を当てに行っている。
当てに行く先と、実際に取れる期間がずれている。

もう一つ、60営業日（約3ヶ月）の判定期間には四半期決算がほぼ必ず1回入る。
「3ヶ月後の水準」は次の決算の結果をかなりの部分で当てに行っていることになる。

変えるもの
----------
  地平        60営業日 -> 20営業日
  到達しきい値 1.2σ√60 -> 1.2σ√20（σ=3%/日なら +27.9% -> +16.1%）
  終盤        t+56〜t+60 の5日平均 -> t+16〜t+20 の5日平均（どちらも到達の半分）
  トレンド    t+60 で MA20>=MA60 -> t+20 で MA5>=MA20

**持続の条件は落とさない。** 突発的な高騰を正例にしないための条件であり、
地平を縮めてもそこは変えない。移動平均だけは、20営業日で判定するのに
MA20>=MA60 では動かなすぎるので短い組に替える。

この文書の対象は「入れ替わったもの」である。学習はまだしない。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
from build_dataset import RiseConfig  # noqa: E402

#: 現行（比較の基準）
CFG_60 = B.DEFAULT_RISE

#: 新案。形は現行のまま、地平だけ縮める。
#: 移動平均は MA20>=MA60 だと20営業日ではほとんど動かないので MA5>=MA20 にする。
CFG_20 = RiseConfig(horizon=20, trend_short=5, trend_long=20)

#: attach_rise_label が付ける列。2つの定義で引き直すため、間で消す
LABEL_COLS = ("label", "future_max_close", "future_rise", "rise_need",
              "keep_days_cnt", "end_level", "end_need", "uptrend_end")

#: チャートに出す範囲（基準日の前後、営業日）
BEFORE, AFTER = 60, 80


def labels_for(panel: pd.DataFrame, cfg: RiseConfig,
               mask: np.ndarray) -> pd.DataFrame:
    """
    1つの定義でラベルを引き直し、対象行だけ取り出す。

    attach_rise_label は渡した DataFrame に列を足す（入力を書き換える）。
    2つの定義で続けて呼ぶので、取り出したあとに必ず消す。
    消さないと2回目が1回目の列を上書きし、どちらの結果か分からなくなる。
    """
    p = B.attach_rise_label(panel, cfg)
    cols = [c for c in LABEL_COLS if c in p.columns]
    out = p.loc[mask, cols].copy()
    p.drop(columns=cols, inplace=True)
    return out


def conditions(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    ラベルを構成する3条件を、それぞれ独立に取り出す。

    どの条件で落ちたかが分からないと、定義を変えた効果を説明できない。
    """
    reached = (pd.to_numeric(df["future_rise"], errors="coerce")
               >= pd.to_numeric(df["rise_need"], errors="coerce"))
    end_ok = (pd.to_numeric(df["end_level"], errors="coerce")
              >= pd.to_numeric(df["end_need"], errors="coerce"))
    trend_ok = pd.to_numeric(df["uptrend_end"], errors="coerce") == 1.0
    return {"reached": reached.fillna(False),
            "end_ok": end_ok.fillna(False),
            "trend_ok": trend_ok.fillna(False),
            "determined": df["label"].notna()}


def funnel(df: pd.DataFrame, name: str) -> Dict:
    """到達 -> 終盤 -> トレンド と絞っていったときの件数。"""
    c = conditions(df)
    det = c["determined"]
    n = int(det.sum())
    r = c["reached"] & det
    e = r & c["end_ok"]
    t = e & c["trend_ok"]
    lab = (df["label"] == True) & det   # noqa: E712

    def pct(x):
        return round(100.0 * x / n, 2) if n else float("nan")

    return {"name": name, "n_determined": n,
            "reached": int(r.sum()), "reached_pct": pct(int(r.sum())),
            "reached_end": int(e.sum()), "reached_end_pct": pct(int(e.sum())),
            "reached_end_trend": int(t.sum()),
            "reached_end_trend_pct": pct(int(t.sum())),
            "positive": int(lab.sum()), "positive_rate": pct(int(lab.sum())),
            # 落ちた理由の内訳（到達したのに落ちたもの）
            "dropped_by_end": int((r & ~c["end_ok"]).sum()),
            "dropped_by_trend": int((e & ~c["trend_ok"]).sum())}


def transition(l60: pd.Series, l20: pd.Series) -> Dict:
    """
    2つの定義の正負がどう入れ替わったか。

    どちらも判定できた行だけを数える。片方が判定不能な行を混ぜると、
    「入れ替わった」と「まだ分からない」が同じ枠に入る。
    """
    both = l60.notna() & l20.notna()
    a = (l60 == True) & both   # noqa: E712
    b = (l20 == True) & both   # noqa: E712
    n = int(both.sum())
    # 「どちらも判定できた行」に限る。both を掛け忘れると、
    # 判定不能な行がまるごと neg_neg に落ちて件数が水増しされる。
    cells = {
        "pos_pos": int((a & b).sum()),
        "pos_neg": int((a & ~b & both).sum()),
        "neg_pos": int((~a & b & both).sum()),
        "neg_neg": int((~a & ~b & both).sum()),
    }
    out = {"n": n, **cells}
    for k, v in cells.items():
        out[f"{k}_pct"] = round(100.0 * v / n, 2) if n else float("nan")
    out["flipped"] = cells["pos_neg"] + cells["neg_pos"]
    out["flipped_pct"] = (round(100.0 * out["flipped"] / n, 2)
                          if n else float("nan"))
    return out


def masks(l60: pd.Series, l20: pd.Series) -> Dict[str, np.ndarray]:
    """遷移の4群。チャートの抽出と素性の集計で同じ切り方を使う。"""
    both = (l60.notna() & l20.notna()).to_numpy()
    a = ((l60 == True).to_numpy()) & both   # noqa: E712
    b = ((l20 == True).to_numpy()) & both   # noqa: E712
    return {"pos_pos": a & b, "pos_neg": a & ~b & both,
            "neg_pos": ~a & b & both, "neg_neg": ~a & ~b & both}


BUCKET_JA = {
    "pos_pos": "両方とも正例",
    "pos_neg": "60日では正例 → 20日では負例",
    "neg_pos": "60日では負例 → 20日では正例",
    "neg_neg": "両方とも負例",
}


def profile(ev: pd.DataFrame, d60: pd.DataFrame, d20: pd.DataFrame,
            m: np.ndarray) -> Dict:
    """1つの群の素性。何が入れ替わったのかを数字で言えるようにする。"""
    def mean(s, scale=1.0):
        v = pd.to_numeric(s, errors="coerce").to_numpy()[m]
        v = v[np.isfinite(v)]
        return round(float(v.mean()) * scale, 2) if v.size else None

    def median(s, scale=1.0):
        v = pd.to_numeric(s, errors="coerce").to_numpy()[m]
        v = v[np.isfinite(v)]
        return round(float(np.median(v)) * scale, 2) if v.size else None

    return {
        "n": int(m.sum()),
        "vol_20d": mean(ev.get("vol_20d")),
        "need_60": mean(d60["rise_need"], 100.0),
        "need_20": mean(d20["rise_need"], 100.0),
        "max_gain_60": median(d60["future_rise"], 100.0),
        "max_gain_20": median(d20["future_rise"], 100.0),
        "end_level_60": median(d60["end_level"], 100.0),
        "end_level_20": median(d20["end_level"], 100.0),
    }


def by_group(ev: pd.DataFrame, l60: pd.Series, l20: pd.Series,
             key: pd.Series, label: str) -> List[Dict]:
    """群ごとの正例率。片方だけで見ると、どこが動いたのか分からない。"""
    both = l60.notna() & l20.notna()
    df = pd.DataFrame({"key": key, "l60": (l60 == True), "l20": (l20 == True),  # noqa: E712
                       "ok": both})
    out = []
    for k, g in df[df["ok"]].groupby("key", sort=True):
        if len(g) < 30:
            continue
        out.append({"group": str(k), "label": label, "n": int(len(g)),
                    "rate_60": round(100.0 * float(g["l60"].mean()), 2),
                    "rate_20": round(100.0 * float(g["l20"].mean()), 2),
                    "diff": round(100.0 * float(g["l20"].mean() - g["l60"].mean()), 2)})
    return out


def vol_bands(vol: pd.Series, n: int = 5) -> pd.Series:
    """ボラの分位帯。正例率はボラで大きく歪むので、帯を揃えて見る。"""
    v = pd.to_numeric(vol, errors="coerce")
    try:
        q = pd.qcut(v, n, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(["—"] * len(v), index=v.index)
    edges = [v[q == i].min() for i in range(int(np.nanmax(q)) + 1)]
    names = []
    for i in q:
        if not np.isfinite(i):
            names.append("欠測")
        else:
            i = int(i)
            hi = edges[i + 1] if i + 1 < len(edges) else np.inf
            names.append(f"{i+1}. σ {edges[i]:.1f}〜{hi:.1f}%"
                         if np.isfinite(hi) else f"{i+1}. σ {edges[i]:.1f}%〜")
    return pd.Series(names, index=v.index)


# --------------------------------------------------------------------------- #
# チャート
# --------------------------------------------------------------------------- #

def build_case(panel: pd.DataFrame, code: str, t_date: pd.Timestamp,
               name: str, bucket: str) -> Optional[Dict]:
    """1件ぶんのチャートデータ。2つの定義の線を両方引く。"""
    g = panel[panel["Code"] == code].reset_index(drop=True)
    idx = g.index[g["Date"] == t_date]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    lo = max(0, i - BEFORE)
    hi = min(len(g), i + AFTER + 1)
    win = g.iloc[lo:hi]
    t_pos = i - lo

    close = g["close"]
    # 移動平均は全系列で計算してから窓を切る（窓の中だけだと頭が欠ける）
    mas = {p: close.rolling(p, min_periods=p).mean() for p in (5, 20, 60)}
    close_t = float(g.iloc[i]["close"])
    vol_t = float(g.iloc[i].get("vol_20d", np.nan))

    def r(v, d=2):
        return None if (v is None or not np.isfinite(v)) else round(float(v), d)

    def horizon_block(cfg: RiseConfig) -> Dict:
        need, end_need = B.rise_thresholds(pd.Series([vol_t]), cfg)
        nd, en = float(need.iloc[0]), float(end_need.iloc[0])
        h_lo, h_hi = i + 1, min(len(g) - 1, i + cfg.horizon)
        fwd = close.iloc[h_lo:h_hi + 1] if h_lo <= h_hi else pd.Series(dtype=float)
        mx = float(fwd.max()) if len(fwd) else np.nan
        end_lo = max(h_lo, h_hi - cfg.end_window + 1)
        ew = close.iloc[end_lo:h_hi + 1] if h_lo <= h_hi else pd.Series(dtype=float)
        end_level = (float(ew.mean()) / close_t - 1) if len(ew) else np.nan
        ms, ml = mas[cfg.trend_short], mas[cfg.trend_long]
        trend = None
        if h_lo <= h_hi and np.isfinite(ms.iloc[h_hi]) and np.isfinite(ml.iloc[h_hi]):
            trend = bool(ms.iloc[h_hi] >= ml.iloc[h_hi])
        hit = None
        if len(fwd) and np.isfinite(nd):
            above = np.where(fwd.to_numpy() >= close_t * (1 + nd))[0]
            if len(above):
                hit = int(h_lo + above[0] - lo)
        reached = bool(np.isfinite(mx) and mx >= close_t * (1 + nd))
        end_ok = bool(np.isfinite(end_level) and end_level >= en)
        return {
            "horizon": cfg.horizon,
            "trendPair": [cfg.trend_short, cfg.trend_long],
            "span": [t_pos + 1, t_pos + cfg.horizon],
            "target": r(close_t * (1 + nd), 1),
            "endLine": r(close_t * (1 + en), 1),
            "thresholdPct": r(nd * 100, 1),
            "endThresholdPct": r(en * 100, 1),
            "maxGainPct": r((mx / close_t - 1) * 100 if np.isfinite(mx) else np.nan, 1),
            "endLevelPct": r(end_level * 100, 1),
            "hitPos": hit,
            "reached": reached, "endOk": end_ok, "trendOk": trend,
            "label": bool(reached and end_ok and (trend is True)),
        }

    def col(nm, d=1):
        return [r(v, d) for v in win[nm]] if nm in win.columns else None

    return {
        "code": code, "name": name, "bucket": bucket,
        "t": t_date.date().isoformat(), "tPos": t_pos,
        "closeAtT": r(close_t, 1),
        "volAtT": r(vol_t, 2),
        "dates": [d.date().isoformat() for d in win["Date"]],
        "close": col("close"), "high": col("high"), "low": col("low"),
        "volume": [r(v, 0) for v in win["vol"]] if "vol" in win.columns else None,
        "ma5": [r(v, 1) for v in mas[5].iloc[lo:hi]],
        "ma20": [r(v, 1) for v in mas[20].iloc[lo:hi]],
        "ma60": [r(v, 1) for v in mas[60].iloc[lo:hi]],
        "h60": horizon_block(CFG_60),
        "h20": horizon_block(CFG_20),
    }


def pick(ev: pd.DataFrame, m: np.ndarray, n: int, seed: int = 0) -> List[int]:
    """
    年をまたいで散らして抽出する。

    無作為だと件数の多い年に偏る。一つの局面だけを見て
    「定義変更はこういうものだ」と判断してしまうのを避ける。
    """
    idx = np.where(m)[0]
    if len(idx) <= n:
        return idx.tolist()
    years = pd.to_datetime(ev["Date"]).dt.year.to_numpy()[idx]
    rng = np.random.default_rng(seed)
    picked: List[int] = []
    uy = sorted(set(years.tolist()))
    per = max(1, n // max(1, len(uy)))
    for y in uy:
        cand = idx[years == y]
        take = min(per, len(cand))
        picked += rng.choice(cand, size=take, replace=False).tolist()
    rest = [i for i in idx if i not in set(picked)]
    if len(picked) < n and rest:
        picked += rng.choice(rest, size=min(n - len(picked), len(rest)),
                             replace=False).tolist()
    return sorted(picked[:n])
