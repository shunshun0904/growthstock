#!/usr/bin/env python3
"""
実験103: 売りの指値を +10%（今）から +15% に上げたほうが妥当か。今の規則・画面の戦略の形・前向きの百分位で、資金全体の
月の収益を比べる。検証だけ。本番（live_track.TAKE_PROFIT・src/lib/strategy.js の takeProfit）は変えない。

運用者の問い（2026-10-17）「指値10%ではなく、15%のほうが妥当なのかを知りたいです」。運用者の選択（同じ日、回す前）: 実際の
買い方は画面の戦略どおり（その日の上位2件まで・同時に3件まで・売れたら枠が空く）→ 資金全体の月の収益で判定し、1件あたりの
収益は参考に出す。

前の結果（どれも今の規則での判定ではない）
  実験83（6本の規則 A15・333件）の参考の表: 1件あたり +10% +3.59% / +15% +4.27% / +20% +4.61% / 指値なし +5.12%
  実験87（21,600通りの組み合わせ）: 後知恵の上位は +15% が多いが、前半で選んだ最良は後半に +6.43 → +0.66% と落ち、前半と後半の
      順位相関は +0.17（過去の成績で細かく選んでも、ほとんどは偶然の当たり）
  §46 運用者の決定（2026-10-12）: 規則は 1.5σ の3本 95以上、売りの指値は +10%

先に決めた比べ方（2026-10-17。結果を見てから動かさない）
------------------------------------------------
  行: 前向きの百分位（実験79 の window_split。テスト窓 3〜11・53か月）。規則 = 1.5σ の3本（lgbm_k15・xgb_k15・cat_k15）が
      すべて 95以上（画面と同じ丸め）
  買い方（画面の戦略。2026-10-17 の live_track・strategy.js と同じ）: 発火の数では見送らない・その日の上位2件（3本の最小の
      高い順、同点は lgbm のスコア順）・同時に3件まで・先着（枠が埋まっていれば見送る）・翌営業日の寄りで買う。指値に届けば
      届いた日の翌営業日から、届かなければ 20営業日の翌営業日から枠が空く（実験87 の simulate）。1件に資金の 1/3、複利にしない。
      手数料・税・すべりは含まない
  売り: A = +10% の指値（今）/ B = +15% の指値。買った日を1日目として 1〜20営業日目の高値が買値の 1.10倍 / 1.15倍以上に
      なったら指値ちょうどで売る（寄りで超えて始まった日も指値で数える。控えめな側）。届かなければ 20営業日目の終値で売る
  物差し（主）: 資金全体に対する月の損益（%）= その月（選定日の月）に買った取引の収益 × 1/3 の合計（取引の無い月は 0）。
      テスト窓ごとに月の平均を出し、B − A の窓ごとの差（9窓）を見る。A と B では枠の空き方が違うので、買えた取引も違う
  判定: B − A の窓ごとの差の **平均 > 0・B が上の窓が過半（5/9 以上）・符号の入れ替えの片側 p < 0.05**（9窓の差の符号を
      全 512 通り入れ替える正確な並べ替え）の3つがそろえば「+15% のほうが妥当」→ 指値を変える案を運用者に出す。
      そろわなければ「+10% のまま」
  参考（判定に使わない）
    1. 3枠で買えた取引の中身（件数・1取引の平均・中央・勝率・到達・−10% 割れ・持った日数・最悪の月・累計の落ち込み）
    2. 1件あたり: 規則の行すべて（同じ行で売りだけ違う）の tp15c − tp10c を窓ごとに。1日あたり（収益の合計 ÷ 持った日数の合計）
    3. 差の分解（1件あたり）: +15% に届いた行（+5pt）/ +10% に届いたが +15% に届かない行（20日目の終値 − 10）/
       +10% に届かない行（0）
    4. 水準の形: +10 / +15 / +20% / 指値なし（20日目の終値）。水準はここから選ばない（見るだけ）
    5. 前半（窓3〜6）・後半（窓7〜11）、月ごとの差（53か月）
    6. ボラの帯（その行の 1.5σ = 1.5 × vol_20d × √20 が 10% 未満 / 10〜15% / 15% 以上）ごとの1件あたりの差（探索的）
  確かめ: 規則の行が 559件（実験88・99 と同じ）、A の3枠の形が §47（実験88）の 128件・1取引 +2.19%・月 +1.76% と同じ、
      自前の回し方が実験87 の run_one と同じ取引になること、出口の到達の順序の食い違い 0件

公開ログには集計だけ（件数・平均・割合・窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e103_tp15_vs_tp10.py

結果（2026-10-17、手元の run・e0211a4・22秒。Actions run 39053324210 で確かめる。docs/MODEL_ADOPTION_RULES.md §63）
------------------------------------------------------------------
  確かめ: 規則の行 559件、+10% の3枠の形 128件・1取引 +2.19%・月 +1.76%（§47 と一致）、実験87 の run_one と同じ取引、
          到達の順序の食い違い 0件
  判定: 資金全体の月の平均 +10% +1.76% / +15% +1.62%。B − A の窓ごとの差 −0.21 ± 0.60pt（上 5・下 4 / 9・p = 0.625）→ +10% のまま
  仕組み: 1取引の平均はほぼ同じ（+2.19 / +2.22%）。+15% は長く持つ（15.2 → 17.9営業日）ので3枠が空きにくく、取引が 128 → 116件
          に減る（月の差 −0.14pt = 取引が減った分 −0.165 + 1取引の平均の分 +0.021）
  1件あたり（規則の行 559件・参考）: +10% +3.09% / +15% +3.58% / +20% +3.71% / 指値なし +4.22%、窓ごとの差 +0.10 ± 0.31pt（p = 0.377）。
          分解: +15% にも届く 23.1%（+1.15pt）・+10% だけ届く 17.4%（20日目の終値 +6.13%、−0.67pt）・届かない 59.6%
  参考の形: 指値なしは月 +2.62% と最も高いが、中央値 +0.79%・勝率 55.1%・最悪の月 −17.62%（水準はここから選ばない）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402

MODELS = E79.MODELS
RULE_CODE = "K15"                   # 1.5σ の3本（実験87 の組の記号）
RULE_LINE = 95.0                    # 3本すべてがこの百分位以上
MIN_BREAK = 1                       # 発火の数では見送らない（live_track.SKIP_BREAKS。2026-10-12 に 8 → 1）
TOP_K = 2                           # その日に買うのは上位2件まで
SLOTS = 3                           # 同時に持つのは3件まで（1件に資金の 1/3）
HOLD = 20                           # 届かなければ 20営業日目の終値で売る
TP_A = 10.0                         # 今の指値
TP_B = 15.0                         # 比べる指値
LEVELS = (10.0, 15.0, 20.0, None)   # 参考の形（None = 指値なし・20日目の終値）。水準はここから選ばない
ALPHA = 0.05                        # 符号の入れ替えの片側 p の線
H1 = (3, 4, 5, 6)                   # 前半のテスト窓
H2 = (7, 8, 9, 10, 11)              # 後半のテスト窓
VOL_BANDS = ((0.0, 10.0, "1.5σ が 10% 未満"), (10.0, 15.0, "1.5σ が 10〜15%"), (15.0, math.inf, "1.5σ が 15% 以上"))
RULE_ROWS = 559                     # 実験88・99 の規則の行
CHECK_A = {"n": 128, "trade_mean": 2.19, "avg": 1.76}   # §47（実験88）: 3枠・発火を問わない・+10%
OUT_DIR = E79.OUT_DIR

log = E79.log


# --------------------------------------------------------------------------- #
# 列と取引
# --------------------------------------------------------------------------- #

def level_cols(t):
    """水準 t（None = 指値なし）の (収益の列, 到達の印の列, 到達日の列)。"""
    if t is None:
        return "c20", None, None
    k = E83.tp_key(t)
    return f"tp{k}c", f"hit{k}", f"hitday{k}"


def level_name(t) -> str:
    return "指値なし（20日目の終値）" if t is None else f"+{t:.0f}% の指値"


def attach_extra(ft: pd.DataFrame, df: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """前向きの行に 20日目の終値 c20 と、その行の 1.5σ の到達線（%）need15 を付ける（行の数は変えない）。"""
    key = ["Code", "Date"]

    def norm(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x["Code"] = x["Code"].astype(str)
        x["Date"] = pd.to_datetime(x["Date"])
        return x.drop_duplicates(key)

    ex = norm(df[key + ["c20"]])
    vol = norm(fr[key + ["vol_20d"]]).reset_index(drop=True)
    need, _ = B.rise_thresholds(vol["vol_20d"], replace(B.DEFAULT_RISE, vol_norm_k=1.5))
    vol["need15"] = np.asarray(need, dtype=float) * 100.0
    out = ft.copy()
    out["Code"] = out["Code"].astype(str)
    out["Date"] = pd.to_datetime(out["Date"])
    n0 = len(out)
    out = out.drop(columns=[c for c in ("c20", "need15") if c in out.columns])
    out = out.merge(ex, on=key, how="left").merge(vol[key + ["need15"]], on=key, how="left")
    if len(out) != n0:
        raise SystemExit(f"行の数が変わった: {n0} → {len(out)}")
    return out


def screen_trades(ft: pd.DataFrame, bar_days, t, slots: int = SLOTS, top_k: int = TOP_K, code: str = RULE_CODE,
                  line: float = RULE_LINE, min_break: int = MIN_BREAK, hold: int = HOLD) -> pd.DataFrame:
    """画面の戦略の形で、水準 t（None = 指値なし）で売るときに買えた取引の行（実験87 の run_one と同じ回し方）。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    c = E87.candidates(ft, code, line, min_break)
    c = c[c["rank"] <= top_k]
    day = np.array([idx.get(pd.Timestamp(d), -1) for d in c["Date"]], dtype=float)
    c, day = c[day >= 0], day[day >= 0]
    rc, hc, dc = level_cols(t)
    ret = pd.to_numeric(c[rc], errors="coerce").to_numpy(dtype=float)
    hit = c[hc].to_numpy(dtype=bool) if hc else np.zeros(len(c), dtype=bool)
    hd = pd.to_numeric(c[dc], errors="coerce").to_numpy(dtype=float) if dc else np.full(len(c), np.nan)
    m = pd.to_datetime(c["Date"]).dt.to_period("M").map(lambda p: p.ordinal).to_numpy()
    tk = E87.simulate(day, ret, hit, hd, m, slots, 0, hold=hold)
    return c.iloc[tk].copy()


def held_days(g: pd.DataFrame, t, hold: int = HOLD) -> np.ndarray:
    """持った営業日数（買った日 = 1日目。届けば到達日、届かなければ hold）。"""
    _, hc, dc = level_cols(t)
    if hc is None:
        return np.full(len(g), float(hold))
    return np.where(g[hc].to_numpy(dtype=bool), pd.to_numeric(g[dc], errors="coerce").to_numpy(dtype=float), float(hold))


def trade_stats(g: pd.DataFrame, t, hold: int = HOLD) -> dict:
    """取引（または行）の1件あたりの要約。1日あたり = 収益の合計 ÷ 持った日数の合計。"""
    rc, hc, _ = level_cols(t)
    r = pd.to_numeric(g[rc], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(r)
    r, days = r[ok], held_days(g, t, hold)[ok]
    hit = g[hc].to_numpy(dtype=bool)[ok] if hc else np.zeros(len(r), dtype=bool)
    n = int(len(r))
    nan = float("nan")
    if not n:
        return {"n": 0, "mean": nan, "med": nan, "win": nan, "hit": nan, "bad": nan, "days": nan, "per_day": nan}
    return {"n": n, "mean": float(r.mean()), "med": float(np.median(r)), "win": float((r > 0).mean() * 100),
            "hit": float(hit.mean() * 100), "bad": float((r < -10).mean() * 100), "days": float(days.mean()),
            "per_day": float(r.sum() / days.sum()) if days.sum() > 0 else nan}


# --------------------------------------------------------------------------- #
# 月の損益と判定
# --------------------------------------------------------------------------- #

def monthly_pnl(trades: pd.DataFrame, t, pos: dict, n_months: int, slots: int = SLOTS) -> np.ndarray:
    """資金全体に対する月の損益（%）。月は選定日の月、1件に資金の 1/slots、取引の無い月は 0。"""
    if not len(trades):
        return np.zeros(n_months)
    rc = level_cols(t)[0]
    mp = pd.to_datetime(trades["Date"]).dt.to_period("M").map(pos).to_numpy(dtype=int)
    return np.bincount(mp, weights=pd.to_numeric(trades[rc], errors="coerce").to_numpy(dtype=float) / slots,
                       minlength=n_months)


def trade_counts(trades: pd.DataFrame, pos: dict, n_months: int) -> np.ndarray:
    """月ごとに買った取引の数（選定日の月）。"""
    if not len(trades):
        return np.zeros(n_months, dtype=int)
    return np.bincount(pd.to_datetime(trades["Date"]).dt.to_period("M").map(pos).to_numpy(dtype=int), minlength=n_months)


def window_means(monthly: np.ndarray, month_win: np.ndarray, windows) -> np.ndarray:
    """テスト窓ごとの月の平均。"""
    return np.array([monthly[month_win == w].mean() if (month_win == w).any() else np.nan for w in windows])


def mean_of(x) -> float:
    """平均（空なら NaN。警告を出さない）。"""
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if len(x) else float("nan")


def max_drawdown(monthly: np.ndarray) -> float:
    cum = np.cumsum(monthly)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float((peak - cum).max()) if len(cum) else float("nan")


def sign_flip_p(d) -> float:
    """片側 p: 差の符号を全通り入れ替えたとき、平均が観測以上になる割合（観測そのものも数える）。"""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError(f"全列挙は 20 個まで: {n}")
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    means = (signs * np.abs(d)).mean(axis=1)
    return float((means >= d.mean() - 1e-12).mean())


def paired(d) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = int(len(d))
    nan = float("nan")
    return {"n": n, "mean": float(d.mean()) if n else nan,
            "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else nan,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": sign_flip_p(d) if n else nan}


def judge(d) -> dict:
    """先に決めた3条件: 平均 > 0・B が上の窓が過半・符号の入れ替えの片側 p < ALPHA。"""
    s = paired(d)
    n = s["n"]
    need = n // 2 + 1
    j = {**s, "need": need, "positive": bool(n and s["mean"] > 0), "majority": bool(n and s["wins"] >= need),
         "p_ok": bool(n and s["p"] < ALPHA)}
    j["pass"] = j["positive"] and j["majority"] and j["p_ok"]
    return j


def fmt_pair(s: dict, unit: str = "pt") -> str:
    if not s.get("n"):
        return "（比べられる窓なし）"
    return (f"{s['mean']:+.2f} ± {s['se']:.2f}{unit}（上 {s['wins']}・下 {s['losses']} / {s['n']}・"
            f"符号の入れ替え p = {s['p']:.3f}）")


# --------------------------------------------------------------------------- #
# 参考の集計
# --------------------------------------------------------------------------- #

def decompose(g: pd.DataFrame, a: float = TP_A, b: float = TP_B) -> dict:
    """1件あたりの差 tp_b − tp_a を、+b% に届いた行・+a% には届いたが +b% には届かない行・+a% に届かない行に分ける。"""
    ka, kb = E83.tp_key(a), E83.tp_key(b)
    ra = pd.to_numeric(g[f"tp{ka}c"], errors="coerce").to_numpy(dtype=float)
    rb = pd.to_numeric(g[f"tp{kb}c"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(rb)
    ha, hb = g[f"hit{ka}"].to_numpy(dtype=bool)[ok], g[f"hit{kb}"].to_numpy(dtype=bool)[ok]
    c20 = pd.to_numeric(g["c20"], errors="coerce").to_numpy(dtype=float)[ok]
    d = rb[ok] - ra[ok]
    n = int(len(d))
    nan = float("nan")
    if not n:
        return {"n": 0}
    top, mid, low = hb, ha & ~hb, ~ha
    return {"n": n, "share_top": float(top.mean() * 100), "share_mid": float(mid.mean() * 100),
            "share_low": float(low.mean() * 100), "gain": float(d[top].sum() / n), "loss": float(d[mid].sum() / n),
            "rest": float(d[low].sum() / n), "mid_c20": float(np.nanmean(c20[mid])) if mid.any() else nan,
            "per_top": float(b - a), "diff": float(d.mean()), "order_bad": int((hb & ~ha).sum())}


def vol_bands(g: pd.DataFrame, a: float = TP_A, b: float = TP_B) -> list:
    """その行の 1.5σ の帯ごとの1件あたり（探索的）。"""
    ka, kb = E83.tp_key(a), E83.tp_key(b)
    v = pd.to_numeric(g["need15"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for lo, hi, name in VOL_BANDS:
        s = g[(v >= lo) & (v < hi)]
        ra = pd.to_numeric(s[f"tp{ka}c"], errors="coerce")
        rb = pd.to_numeric(s[f"tp{kb}c"], errors="coerce")
        ok = ra.notna() & rb.notna()
        rows.append({"band": name, "n": int(ok.sum()),
                     "need_med": float(np.nanmedian(v[(v >= lo) & (v < hi)])) if len(s) else float("nan"),
                     "hit_a": float(s[f"hit{ka}"].mean() * 100) if len(s) else float("nan"),
                     "hit_b": float(s[f"hit{kb}"].mean() * 100) if len(s) else float("nan"),
                     "a": float(ra[ok].mean()) if ok.any() else float("nan"),
                     "b": float(rb[ok].mean()) if ok.any() else float("nan"),
                     "diff": float((rb[ok] - ra[ok]).mean()) if ok.any() else float("nan")})
    return rows


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def report(ft: pd.DataFrame, bar_days, chk: dict | None = None, check_a: dict | None = CHECK_A,
           rule_rows: int | None = RULE_ROWS) -> dict:
    allm, pos, month_win = E87.months_index(ft)
    n_months = len(allm)
    windows = sorted(int(w) for w in set(month_win.tolist()) if w > 0)
    rule = ft[pd.to_numeric(ft[f"min_{RULE_CODE}"], errors="coerce").to_numpy(dtype=float) >= RULE_LINE]
    summary: dict = {"rows": int(len(ft)), "rule_rows": int(len(rule)), "months": int(n_months), "windows": windows,
                     "design": {"rule": f"{RULE_CODE} >= {RULE_LINE}", "top_k": TOP_K, "slots": SLOTS,
                                "min_break": MIN_BREAK, "hold": HOLD, "a": TP_A, "b": TP_B, "alpha": ALPHA}}

    sims = {}
    for t in LEVELS:
        tr = screen_trades(ft, bar_days, t)
        mon = monthly_pnl(tr, t, pos, n_months)
        sims[t] = {"trades": tr, "monthly": mon, "wm": window_means(mon, month_win, windows)}

    # 確かめ
    checks = {"rule_rows": int(len(rule)), "rule_rows_expected": rule_rows}
    for t in (TP_A, TP_B):
        r1 = E87.run_one(ft, bar_days, RULE_CODE, RULE_LINE, MIN_BREAK, TOP_K, SLOTS, 0, t)
        mine = sims[t]["trades"]
        checks[f"same_as_e87_{E83.tp_key(t)}"] = bool(
            list(zip(r1["Code"].astype(str), pd.to_datetime(r1["Date"])))
            == list(zip(mine["Code"].astype(str), pd.to_datetime(mine["Date"]))))
    sa = trade_stats(sims[TP_A]["trades"], TP_A)
    checks["a_now"] = {"n": sa["n"], "trade_mean": round(sa["mean"], 2), "avg": round(float(sims[TP_A]["monthly"].mean()), 2)}
    checks["a_expected"] = check_a
    checks["a_match"] = bool(check_a is not None and checks["a_now"] == check_a)
    if chk:
        checks["order_violations"] = int(chk.get("order_violations", -1))
    summary["checks"] = checks

    print(f"\n■ 0. 前向き {len(ft):,}行・{n_months}か月・テスト窓 {len(windows)}つ（窓{windows[0]}〜{windows[-1]}）。"
          f"規則（1.5σ の3本 {RULE_LINE:.0f}以上）の行 {len(rule)}件" + (f"（実験88・99 は {rule_rows}件）" if rule_rows else ""))
    print(f"  確かめ: +10% の3枠の形 {sa['n']}件・1取引 {sa['mean']:+.2f}%・月 {sims[TP_A]['monthly'].mean():+.2f}%"
          + (f"（§47 は {check_a['n']}件・{check_a['trade_mean']:+.2f}%・月 {check_a['avg']:+.2f}% → "
             f"{'一致' if checks['a_match'] else '食い違い'}）" if check_a else "")
          + f" / 実験87 の run_one と同じ取引: +10% {'○' if checks['same_as_e87_10'] else '×'}・"
            f"+15% {'○' if checks['same_as_e87_15'] else '×'}"
          + (f" / 出口の到達の順序の食い違い {checks['order_violations']}件" if chk else ""))

    # ■ 1. 判定
    d_win = sims[TP_B]["wm"] - sims[TP_A]["wm"]
    j = judge(d_win)
    summary["judge"] = j
    h1, h2 = np.isin(month_win, H1), np.isin(month_win, H2)
    print(f"\n■ 1. 判定（画面の戦略の形: 上位{TOP_K}件・同時に{SLOTS}件・発火を問わない。資金全体に対する月の損益。"
          f"B − A のテスト窓ごとの差）: 平均 > 0 / B が上の窓が過半 / 符号の入れ替え p < {ALPHA}")
    for t, tag in ((TP_A, "A"), (TP_B, "B")):
        mon = sims[t]["monthly"]
        print(f"  {tag} {level_name(t):<10} 月の平均 {mon.mean():+.2f}%（前半 {mean_of(mon[h1]):+.2f} / 後半 {mean_of(mon[h2]):+.2f}）"
              f"・取引 {len(sims[t]['trades'])}件")
    print(f"  B − A: {fmt_pair(j)}")
    verdict = "+15% のほうが妥当（指値を変える案を運用者に出す）" if j["pass"] else "+10% のまま"
    print(f"  → 平均>0 {'○' if j['positive'] else '×'} / 上の窓 {j['wins']}/{j['n']}（{j['need']} 以上）"
          f"{'○' if j['majority'] else '×'} / p < {ALPHA} {'○' if j['p_ok'] else '×'} → {verdict}")

    # ■ 2. 3枠で買えた取引の中身
    print(f"\n■ 2. 3枠で買えた取引の中身（+20%・指値なしは参考の形。水準はここから選ばない）")
    print(f"  {'売り':<16}{'取引':>5}{'1取引':>8}{'中央':>8}{'勝率':>7}{'到達':>7}{'−10%割れ':>9}{'日数':>6}"
          f"{'月の平均':>9}{'最悪の月':>9}{'落ち込み':>9}")
    summary["screen"] = {}
    for t in LEVELS:
        s = trade_stats(sims[t]["trades"], t)
        mon = sims[t]["monthly"]
        row = {**s, "avg": float(mon.mean()), "avg_h1": float(mean_of(mon[h1])), "avg_h2": float(mean_of(mon[h2])),
               "worst": float(mon.min()), "max_dd": max_drawdown(mon),
               "zero_months": int((trade_counts(sims[t]["trades"], pos, n_months) == 0).sum())}
        summary["screen"][level_name(t)] = row
        print(f"  {level_name(t):<14}{s['n']:>5}{s['mean']:>+8.2f}{s['med']:>+8.2f}{s['win']:>6.1f}%{s['hit']:>6.1f}%"
              f"{s['bad']:>8.1f}%{s['days']:>6.1f}{row['avg']:>+8.2f}%{row['worst']:>+8.2f}%{row['max_dd']:>8.1f}%")

    # ■ 3. 1件あたり（規則の行すべて）
    print(f"\n■ 3. 1件あたり（規則の行すべて {len(rule)}件。同じ行で売りだけ違う。枠の制約なし。参考）")
    print(f"  {'売り':<16}{'件数':>5}{'平均':>8}{'中央':>8}{'勝率':>7}{'到達':>7}{'−10%割れ':>9}{'日数':>6}{'1日あたり':>10}")
    summary["per_trade"] = {}
    for t in LEVELS:
        s = trade_stats(rule, t)
        summary["per_trade"][level_name(t)] = s
        print(f"  {level_name(t):<14}{s['n']:>5}{s['mean']:>+8.2f}{s['med']:>+8.2f}{s['win']:>6.1f}%{s['hit']:>6.1f}%"
              f"{s['bad']:>8.1f}%{s['days']:>6.1f}{s['per_day']:>+9.3f}%")
    ka, kb = E83.tp_key(TP_A), E83.tp_key(TP_B)
    dr = (pd.to_numeric(rule[f"tp{kb}c"], errors="coerce") - pd.to_numeric(rule[f"tp{ka}c"], errors="coerce"))
    d_rule = np.array([dr[rule["window"] == w].mean() for w in windows])
    pr = paired(d_rule)
    summary["per_trade_diff"] = pr
    print(f"  +15% − +10%（窓ごと）: {fmt_pair(pr)}")

    # ■ 4. 差の分解
    dec = decompose(rule)
    summary["decompose"] = dec
    print(f"\n■ 4. 1件あたりの差の分解（規則の行 {dec.get('n', 0)}件）: +15% − +10% = {dec.get('diff', float('nan')):+.2f}pt")
    if dec.get("n"):
        print(f"  +15% にも届いた行      {dec['share_top']:5.1f}%  1行 +{dec['per_top']:.0f}pt → 平均に {dec['gain']:+.2f}pt")
        print(f"  +10% だけ届いた行      {dec['share_mid']:5.1f}%  20日目の終値の平均 {dec['mid_c20']:+.2f}%"
              f"（1行 {dec['mid_c20'] - TP_A:+.2f}pt）→ 平均に {dec['loss']:+.2f}pt")
        print(f"  +10% に届かない行      {dec['share_low']:5.1f}%  差 0 → 平均に {dec['rest']:+.2f}pt"
              f"（到達の順序の食い違い {dec['order_bad']}件）")

    # ■ 5. 前半・後半と月ごと
    dm = sims[TP_B]["monthly"] - sims[TP_A]["monthly"]
    pm_all = {"n": int(len(dm)), "mean": float(dm.mean()), "se": float(dm.std(ddof=1) / math.sqrt(len(dm))) if len(dm) > 1 else float("nan"),
              "wins": int((dm > 0).sum()), "losses": int((dm < 0).sum()), "ties": int((dm == 0).sum())}
    summary["monthly_diff"] = pm_all
    summary["halves"] = {"h1": {"a": float(mean_of(sims[TP_A]["monthly"][h1])), "b": float(mean_of(sims[TP_B]["monthly"][h1])),
                                "months": int(h1.sum())},
                         "h2": {"a": float(mean_of(sims[TP_A]["monthly"][h2])), "b": float(mean_of(sims[TP_B]["monthly"][h2])),
                                "months": int(h2.sum())}}
    print(f"\n■ 5. 前半・後半と月ごと（資金全体の月の損益）")
    for k, name in (("h1", f"前半（窓{H1[0]}〜{H1[-1]}）"), ("h2", f"後半（窓{H2[0]}〜{H2[-1]}）")):
        h = summary["halves"][k]
        print(f"  {name:<16}{h['months']:>3}か月  +10% {h['a']:+.2f}% / +15% {h['b']:+.2f}% / 差 {h['b'] - h['a']:+.2f}pt")
    print(f"  月ごとの差（{pm_all['n']}か月）: {pm_all['mean']:+.2f} ± {pm_all['se']:.2f}pt・+15% が上 {pm_all['wins']}か月・"
          f"下 {pm_all['losses']}か月・同じ {pm_all['ties']}か月")

    # ■ 6. ボラの帯
    vb = vol_bands(rule)
    summary["vol_bands"] = vb
    print(f"\n■ 6. その行の 1.5σ の帯ごと（1件あたり、規則の行。探索的。判定に使わない）")
    print(f"  {'帯':<16}{'件数':>5}{'1.5σ 中央':>10}{'+10%到達':>9}{'+15%到達':>9}{'+10%':>8}{'+15%':>8}{'差':>8}")
    for r in vb:
        print(f"  {r['band']:<14}{r['n']:>5}{r['need_med']:>9.1f}%{r['hit_a']:>8.1f}%{r['hit_b']:>8.1f}%"
              f"{r['a']:>+8.2f}{r['b']:>+8.2f}{r['diff']:>+8.2f}")

    # ■ 7. 窓ごとの値
    print(f"\n■ 7. テスト窓ごとの値（資金全体の月の平均）")
    print(f"  {'窓':>3}{'月数':>5}{'取引 +10%':>10}{'取引 +15%':>10}{'+10%':>9}{'+15%':>9}{'差':>9}")
    summary["per_window"] = []
    for k, w in enumerate(windows):
        na = int((sims[TP_A]["trades"]["window"] == w).sum())
        nb = int((sims[TP_B]["trades"]["window"] == w).sum())
        rec = {"window": int(w), "months": int((month_win == w).sum()), "n_a": na, "n_b": nb,
               "a": float(sims[TP_A]["wm"][k]), "b": float(sims[TP_B]["wm"][k]), "diff": float(d_win[k])}
        summary["per_window"].append(rec)
        print(f"  {w:>3}{rec['months']:>5}{na:>10}{nb:>10}{rec['a']:>+8.2f}%{rec['b']:>+8.2f}%{rec['diff']:>+8.2f}pt")

    print(f"\n■ まとめ: +15% − +10%（資金全体の月の平均、窓ごと）{fmt_pair(j)} → {verdict}"
          f" ｜ 1件あたり（規則の行）{fmt_pair(pr)}")
    summary["verdict"] = verdict
    return summary


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def save(summary: dict, name: str = "e103_summary.json") -> str:
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=conv)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験103: +15% の指値は +10% より妥当か（今の規則・画面の戦略の形・前向き）")
    ap.parse_args(argv)
    t0 = time.time()
    if (LT.TAKE_PROFIT, LT.TOP_K, LT.SLOTS, LT.SKIP_BREAKS, LT.AGREE_PCT) != (TP_A, TOP_K, SLOTS, MIN_BREAK, RULE_LINE):
        log("注意: 今の live_track の設定がこの実験の前提（2026-10-17）と違う。前提どおりに回す")
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件・"
        "到達の割合 " + "・".join(f"+{k}% {v:.1f}%" for k, v in chk["hit_share"].items()))
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = E87.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    ft = attach_extra(ft, df, fr)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。1.5σ の到達線が付いた行 {int(ft['need15'].notna().sum()):,}")
    summary = report(ft, bar_days, chk)
    summary["exit_check"] = chk
    log(f"記録: {save(summary)} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
