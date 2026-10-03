#!/usr/bin/env python3
"""
実験59: 選び方4通りで、条件を満たす銘柄を 1単元（100株）ずつ買ったときの損益と資産回転率。

運用者の依頼（2026-10-03）
  「銘柄の選び方を
     - gbdt3モデルで97.5%以上の銘柄
     - gbdt3モデルで95%以上の銘柄
     - 全5モデルで95%以上の銘柄
     - 全5モデルで90%以上の銘柄
   で4パターンで、条件を満たす銘柄を1単元（100株）を購入した場合の損益シミュレーションを
   してほしいです。同時保有数は３銘柄なので、資産回転率も指標に入れたいです。」

何を測るか
  本番モデル5つ（LightGBM / XGBoost / CatBoost / ロジスティック回帰 / MLP）の out-of-fold
  （Release の oof.parquet / <algo>_oof.parquet。その行より前のデータだけで学習した採点）に
  選び方を当て、翌営業日の寄りで 100株 買い、+20% の指値か 20営業日目の終値で売る。
  同時に持つのは 3銘柄まで（先着順・乗り換えなし。実験33〜35、実運用の追跡と同じ約束）。
  判断・値動き・枠の関数は research/live_track.py のものをそのまま使う。

選び方（百分位 = そのモデルの OOF の分布のうち、そのスコアより低い割合 × 100。画面と同じ式）
  GBDT3 97.5以上     ブースティング3モデルすべてが 97.5 以上
  GBDT3 95以上       同 95 以上
  全5モデル 95以上   5モデルすべてが 95 以上
  全5モデル 90以上   同 90 以上
  （参考）現行の規則  GBDT3 90以上・発火8件以上の日だけ・日内上位2件（画面の規則そのまま）
  4つの選び方は発火数の足切りを置かず、その日の候補を p_min（合議の最小の百分位）の高い順に
  空き枠のぶんだけ買う（1日に最大3件）。

百分位の取り方は2通り出す
  過去分布（本命）   その日より前の OOF の分布で百分位を出す（過去の行が 500 未満の日は判定しない）。
                     画面が「その時点の本番モデルの OOF」で百分位を出すのと同じ立場で、先の値を見ない
  全期間分布（参考） OOF 全体の分布で百分位を出す（実運用の追跡 research/live_track.py の見込みと
                     同じ約束）。後半ほどスコアが高いので、選定が 2024〜25年に偏る

数え方
  買値        翌営業日の始値。買付額は **調整前の始値 × 100株**（分割調整後の値だと、後で分割した
              銘柄の当時の金額を小さく見積もる）。円の損益 = 買付額 × 収益率（値動きは分割調整後）
  出口        +20% は 1営業日目（買った日）から高値で見る。届けば +20% ちょうど（指値）。
              届かなければ 20営業日目の終値。研究の物差し ret_o1_20（5日平均終値）での平均% も併記
  手数料・税・配当・スリッページは含まない。終わっていない玉（保有中）は成績に数えない
  投下資本    その営業日に持っていた玉の買付額の合計。平均（空き枠は 0 と数える）と
              最大（= 必要資金の目安）
  回転率      年間の買付額 ÷ 必要資金（最大同時投下資本）。用意した資金が年に何回まわるか。
              ÷ 平均投下資本 も出す（こちらは保有期間の短さだけを映す）
  稼働率      枠が埋まっていた営業日の割合（Σ保有日数 ÷ 枠数 ÷ 営業日数）

出力  research/_data/oof/e59_summary.csv（選び方 × 百分位の取り方）、e59_by_year.csv、
      e59_trades.csv（取引の一覧。銘柄コードと買値を含むので、公開の場には出さない）。
      本番の設定は何も変えない。ログに出すのは件数・割合・金額の合計だけ。

使い方
    python3 research/exp/e59_unit_sim.py
    python3 research/exp/e59_unit_sim.py --oof-dir research/_data --hows past
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_track as L  # noqa: E402

OOF_DIR = os.path.join(L.DATA_DIR, "oof")

UNIT = 100            # 1単元の株数
MIN_HIST = 500        # 過去分布で百分位を出すのに要る過去の行数（約 40営業日ぶん）
YEAR_DAYS = 365.25    # 「年間」に直すときの日数（暦の日数で割る）

#: (名前, 合議に使うモデル, 百分位の下限)。発火数の足切りなし、1日に最大 枠数 件
PATTERNS = (
    ("GBDT3 97.5以上", L.BOOST, 97.5),
    ("GBDT3 95以上", L.BOOST, 95.0),
    ("全5モデル 95以上", L.ALL5, 95.0),
    ("全5モデル 90以上", L.ALL5, 90.0),
)
#: 参考: 画面の規則そのまま（GBDT3 90以上・発火 8件以上・上位 2件）
REFERENCE = ("参考: 現行の規則（GBDT3 90・発火8件以上・上位2件）", L.BOOST, L.AGREE_PCT)
HOWS = {"past": "過去分布", "whole": "全期間分布"}
DONE = ("+20%到達", "満了")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- #
# 百分位
# ---------------------------------------------------------------- #

def pct_expanding(dates, scores, min_hist: int = MIN_HIST) -> np.ndarray:
    """
    その日より前（同じ日は含まない）の分布で百分位。丸めは live_track.pct_of と同じ
    （小数1桁、Python の round）。過去の行が min_hist 未満の日と、スコアが NaN の行は NaN。
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    s = np.asarray(scores, dtype=float)
    out = np.full(len(s), np.nan)
    order = np.argsort(d, kind="stable")
    hist = np.array([], dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and d[order[j]] == d[order[i]]:
            j += 1
        idx = order[i:j]
        if len(hist) >= min_hist:
            k = np.searchsorted(hist, s[idx], side="left")
            vals = np.array([round(float(x / len(hist)) * 100, 1) for x in k])
            vals[~np.isfinite(s[idx])] = np.nan
            out[idx] = vals
        new = s[idx]
        hist = np.sort(np.concatenate([hist, new[np.isfinite(new)]]))
        i = j
    return out


def pct_whole(scores) -> np.ndarray:
    """OOF 全体の分布で百分位（live_track.pct_of）。スコアが NaN の行は NaN。"""
    s = np.asarray(scores, dtype=float)
    out = L.pct_of(s, s[np.isfinite(s)]).astype(float)
    out[~np.isfinite(s)] = np.nan
    return out


def load_oofs(oof_dir: str, model_dir: str, algos: Sequence[str] = L.ALL5) -> Dict[str, pd.DataFrame]:
    """本番モデルの OOF を algo ごとに読む（置き場所の探し方は live_track.find_oof）。"""
    oofs = {}
    for a in algos:
        path = L.find_oof(a, oof_dir, model_dir)
        if path is None:
            raise SystemExit(f"{a} の OOF が見つかりません（{oof_dir} / {model_dir}）")
        o = pd.read_parquet(path)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        oofs[a] = o
    return oofs


def rows_with_pct(oofs: Dict[str, pd.DataFrame], how: str, min_hist: int = MIN_HIST) -> pd.DataFrame:
    """
    5モデルの OOF を LightGBM の行に揃え（左結合。欠けたモデルの百分位は NaN のまま →
    そのモデルを使う選び方では「判定できない」）、s_<algo> / p_<algo> を付ける。
    """
    base = None
    for a, o in oofs.items():
        keep = ["Code", "Date", "score"] + ([c for c in ("label", "ret_o1_20") if c in o]
                                            if a == "lgbm" else [])
        o = o[keep].rename(columns={"score": f"s_{a}"}).copy()
        s = o[f"s_{a}"].to_numpy(dtype=float)
        o[f"p_{a}"] = pct_whole(s) if how == "whole" else pct_expanding(o["Date"], s, min_hist)
        if base is None:
            base = o
        else:
            base = base.merge(o, on=["Code", "Date"], how="left")
    base["score"] = base["s_lgbm"]
    return base.sort_values(["Date", "Code"]).reset_index(drop=True)


# ---------------------------------------------------------------- #
# 取引: 1単元の買付額と円の損益
# ---------------------------------------------------------------- #

def exec_ret(status: pd.Series, ret_close: pd.Series, target: float = L.TAKE_PROFIT) -> pd.Series:
    """規則どおりの出口の収益率（%）。+20% は指値ちょうど、満了は 20営業日目の終値。他は NaN。"""
    r = pd.Series(np.nan, index=status.index, dtype=float)
    r[status == "+20%到達"] = target
    m = status == "満了"
    r[m] = pd.to_numeric(ret_close[m], errors="coerce")
    return r


def held_days(status: pd.Series, hit_day: pd.Series, hold: int = L.HOLD) -> pd.Series:
    """枠をふさいだ営業日数（買った日を 1 と数える）。+20% なら届いた日まで、満了なら hold。"""
    d = pd.Series(np.nan, index=status.index, dtype=float)
    d[status == "+20%到達"] = pd.to_numeric(hit_day[status == "+20%到達"], errors="coerce")
    d[status == "満了"] = float(hold)
    return d


def attach_capital(taken: pd.DataFrame, raw_open: pd.DataFrame, unit: int = UNIT) -> pd.DataFrame:
    """
    買った取引に、調整前の始値（買った日）× 株数 の買付額と、円の損益を付ける。
    raw_open: 列 Code / Date / O（調整前の始値）。
    """
    t = taken.merge(raw_open.rename(columns={"Date": "buy_date", "O": "raw_open"}),
                    on=["Code", "buy_date"], how="left")
    t["capital"] = t["raw_open"] * unit
    t["ret_exec"] = exec_ret(t["status"], t["ret_close"])
    t["days"] = held_days(t["status"], t["hit_day"])
    t["pnl"] = t["capital"] * t["ret_exec"] / 100.0           # 規則どおりの出口（終値）
    t["pnl_ma5"] = t["capital"] * pd.to_numeric(t["ret"], errors="coerce") / 100.0  # 研究の物差し
    return t


def empty_trades() -> pd.DataFrame:
    """取引が1件も無いときの空の表（stats が読む列だけ持つ）。"""
    cols = {c: pd.Series(dtype=float) for c in ("ret_exec", "capital", "pnl", "days", "ret")}
    cols["status"] = pd.Series(dtype=object)
    return pd.DataFrame(cols)


def capital_curve(done: pd.DataFrame, bar_days: Sequence) -> pd.Series:
    """
    営業日ごとの投下資本（その日に持っていた玉の買付額の合計）。
    買った日から売った日まで（両端を含む）。live_track.simulate が枠を空ける日
    （買い + 保有日数）の前日までと同じ。
    """
    idx = {pd.Timestamp(d): i for i, d in enumerate(bar_days)}
    diff = np.zeros(len(bar_days) + 1)
    for r in done.itertuples(index=False):
        if not (np.isfinite(r.capital) and np.isfinite(r.days)):
            continue
        b = idx[pd.Timestamp(r.buy_date)]
        e = min(b + int(r.days) - 1, len(bar_days) - 1)
        diff[b] += r.capital
        diff[e + 1] -= r.capital
    return pd.Series(np.cumsum(diff)[:len(bar_days)], index=pd.DatetimeIndex(bar_days))


def stats(done: pd.DataFrame, curve: pd.Series, lo, hi, *, slots: int, n_full: int = 0,
          n_pass: int = 0, n_open: int = 0) -> dict:
    """
    期間 [lo, hi]（日付）の成績。done は終わった取引（status が DONE）だけ。
    curve は capital_curve（期間で切る）。年間は暦の日数で割る。
    """
    if len(curve):
        c = curve[(curve.index >= pd.Timestamp(lo)) & (curve.index <= pd.Timestamp(hi))]
    else:
        c = curve
    n_days = int(len(c))
    years = max((pd.Timestamp(hi) - pd.Timestamp(lo)).days, 1) / YEAR_DAYS
    r = pd.to_numeric(done["ret_exec"], errors="coerce")
    ok = r.notna()
    r = r[ok]
    n = int(len(r))
    cap = done.loc[ok, "capital"]
    pnl = done.loc[ok, "pnl"]
    has_cap = cap.notna()
    buy_total = float(cap[has_cap].sum())
    cap_max = float(c.max()) if n_days else float("nan")
    cap_avg = float(c.mean()) if n_days else float("nan")
    held = pd.to_numeric(done.loc[ok, "days"], errors="coerce")
    out = {
        "n": n, "per_year": n / years, "years": years, "n_days": n_days,
        "n_pass": int(n_pass), "n_full": int(n_full), "n_open": int(n_open),
        "win": float((r > 0).mean() * 100) if n else float("nan"),
        "hit": float((done.loc[ok, "status"] == "+20%到達").mean() * 100) if n else float("nan"),
        "mean": float(r.mean()) if n else float("nan"),
        "median": float(r.median()) if n else float("nan"),
        "se": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "worst": float(r.min()) if n else float("nan"),
        "mean_ma5": float(pd.to_numeric(done.loc[ok, "ret"], errors="coerce").mean()) if n else float("nan"),
        "n_no_capital": int((~has_cap).sum()),
        "pnl_total": float(pnl[has_cap].sum()),
        "pnl_per_trade": float(pnl[has_cap].mean()) if has_cap.any() else float("nan"),
        "pnl_per_year": float(pnl[has_cap].sum()) / years,
        "pnl_worst": float(pnl[has_cap].min()) if has_cap.any() else float("nan"),
        "buy_total": buy_total,
        "buy_median": float(cap[has_cap].median()) if has_cap.any() else float("nan"),
        "buy_max": float(cap[has_cap].max()) if has_cap.any() else float("nan"),
        "cap_max": cap_max, "cap_avg": cap_avg,
        "turnover": (buy_total / years / cap_max) if cap_max and cap_max > 0 else float("nan"),
        "turnover_avg": (buy_total / years / cap_avg) if cap_avg and cap_avg > 0 else float("nan"),
        "roi_max": (float(pnl[has_cap].sum()) / years / cap_max * 100) if cap_max and cap_max > 0 else float("nan"),
        "roi_avg": (float(pnl[has_cap].sum()) / years / cap_avg * 100) if cap_avg and cap_avg > 0 else float("nan"),
        "util": float(held.sum() / (slots * n_days) * 100) if n_days else float("nan"),
        "hold_mean": float(held.mean()) if n else float("nan"),
    }
    return out


# ---------------------------------------------------------------- #
# 1つの選び方を通す
# ---------------------------------------------------------------- #

def run_pattern(base: pd.DataFrame, bars: pd.DataFrame, bar_days: Sequence, raw_open: pd.DataFrame,
                *, models: Sequence[str], agree: float, min_break: int, top_k: int, slots: int,
                unit: int = UNIT):
    """decide → forward → simulate → 買付額。戻り値 (基準を満たした行, 枠に入れた全行, 買った取引)。"""
    r, _, picks = L.decide(base, agree=agree, min_break=min_break, top_k=top_k, models=models)
    passed = r[r["passed"] & r["buyable_day"]]
    if picks.empty:
        return passed, picks, picks
    fwd = L.forward(bars, picks)
    sim = L.simulate(fwd, bar_days, slots=slots)
    taken = attach_capital(sim[sim["taken"] == L.TAKEN].copy(), raw_open, unit)
    return passed, sim, taken


def complete_cutoff(bar_days: Sequence, hold: int = L.HOLD):
    """選定日がこの日以前なら 20営業日の出口まで日足が揃っている（live_track と同じ）。"""
    return bar_days[-1 - hold] if len(bar_days) > hold else None


def _man(v) -> str:
    """円を万円で（小数なし、カンマ区切り）。−5千円〜0円は「-0」ではなく「0」。"""
    if v is None or not np.isfinite(v):
        return "-"
    out = f"{v / 1e4:,.0f}"
    return "0" if out == "-0" else out


def _pct(v, fmt="{:+.2f}") -> str:
    return "-" if v is None or not np.isfinite(v) else fmt.format(v)


NAME_W = 44
COLS = (("取引", 5), ("年間", 6), ("勝率", 6), ("到達", 6), ("1取引%", 8), ("SE", 6), ("総損益万円", 11),
        ("年万円", 8), ("買付中央", 9), ("必要資金", 9), ("平均投下", 9), ("回転", 6), ("稼働", 6), ("保有日", 7))
HEAD = "  " + L._l("選び方", NAME_W) + "".join(L._r(h, w) for h, w in COLS)


def line(name: str, s: dict) -> str:
    if not s["n"]:
        return f"  {L._l(name, NAME_W)}（取引なし。基準を満たした {s['n_pass']}件）"
    vals = (s["n"], f"{s['per_year']:.1f}", _pct(s["win"], "{:.0f}%"), _pct(s["hit"], "{:.0f}%"),
            _pct(s["mean"]), _pct(s["se"], "{:.2f}"), _man(s["pnl_total"]), _man(s["pnl_per_year"]),
            _man(s["buy_median"]), _man(s["cap_max"]), _man(s["cap_avg"]), _pct(s["turnover"], "{:.1f}"),
            _pct(s["util"], "{:.0f}%"), _pct(s["hold_mean"], "{:.1f}"))
    return "  " + L._l(name, NAME_W) + "".join(L._r(v, w) for v, (_, w) in zip(vals, COLS))


YCOLS = (("候補", 6), ("取引", 5), ("満杯", 5), ("勝率", 6), ("1取引%", 8), ("損益万円", 9), ("買付万円", 10),
         ("必要資金", 9), ("回転", 6), ("稼働", 6))
YEAR_HEAD = "    " + L._l("年", 6) + "".join(L._r(h, w) for h, w in YCOLS)


def year_line(y, s: dict) -> str:
    if not s["n"]:
        vals = (s["n_pass"], 0, s["n_full"]) + ("-",) * 7
    else:
        vals = (s["n_pass"], s["n"], s["n_full"], _pct(s["win"], "{:.0f}%"), _pct(s["mean"]),
                _man(s["pnl_total"]), _man(s["buy_total"]), _man(s["cap_max"]),
                _pct(s["turnover"], "{:.1f}"), _pct(s["util"], "{:.0f}%"))
    return "    " + L._l(str(y), 6) + "".join(L._r(v, w) for v, (_, w) in zip(vals, YCOLS))


# ---------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験59: 選び方4通り × 1単元の損益と回転率")
    ap.add_argument("--data-dir", default=L.DATA_DIR)
    ap.add_argument("--oof-dir", default=L.DATA_DIR,
                    help="Release から落とした OOF（oof.parquet / <algo>_oof.parquet）の場所")
    ap.add_argument("--model-dir", default=L.MODEL_DIR)
    ap.add_argument("--slots", type=int, default=L.SLOTS)
    ap.add_argument("--unit", type=int, default=UNIT, help="1単元の株数")
    ap.add_argument("--min-hist", type=int, default=MIN_HIST, help="過去分布に要る過去の行数")
    ap.add_argument("--hows", default="past,whole", help="百分位の取り方（past / whole）")
    ap.add_argument("--prefix", default="e59", help="出力ファイル名の頭")
    args = ap.parse_args(argv)
    hows = [h for h in args.hows.split(",") if h]
    bad = [h for h in hows if h not in HOWS]
    if bad:
        raise SystemExit(f"百分位の取り方は {list(HOWS)} から: {bad}")
    os.makedirs(OOF_DIR, exist_ok=True)

    oofs = load_oofs(args.oof_dir, args.model_dir)
    lo_oof = min(o["Date"].min() for o in oofs.values())
    hi_oof = max(o["Date"].max() for o in oofs.values())
    meta_path = os.path.join(args.model_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    bars = L.load_bars(args.data_dir, start=lo_oof - pd.Timedelta(days=10), extra=("O",))
    bar_days = sorted(pd.Timestamp(d) for d in bars["Date"].unique())
    raw_open = bars[["Code", "Date", "O"]]
    cutoff = complete_cutoff(bar_days)
    if cutoff is None:
        raise SystemExit("日足が足りません")
    # 成績に数える期間: OOF の最初の日 〜 min(OOF の最後の日, 出口まで日足が揃う最後の選定日)
    lo = lo_oof
    hi = min(hi_oof, cutoff)

    print("=" * 100)
    print(f"実験59 選び方4通り × 1単元（{args.unit}株）× 枠{args.slots}: 損益と資産回転率")
    print("=" * 100)
    print(f"  OOF {lo_oof.date()}〜{hi_oof.date()}（{len(oofs['lgbm']):,}行 / "
          f"{oofs['lgbm']['Date'].nunique():,}営業日）、モデル {len(oofs)}つ"
          + (f"、学習 {str(meta.get('trainedAt', ''))[:10]} / {meta.get('preset', '')}" if meta else ""))
    print(f"  日足 〜{bar_days[-1].date()}。成績に数える選定日 {lo.date()}〜{hi.date()}"
          f"（20営業日の出口まで日足が揃う範囲）")
    print(f"  出口: +{L.TAKE_PROFIT:.0f}% の指値 / {L.HOLD}営業日目の終値。手数料・税・配当は含まない。"
          f"年間は暦の日数（{YEAR_DAYS}日）で割る")

    summary, by_year, trades = [], [], []
    for how in hows:
        log(f"百分位 {HOWS[how]} を付ける")
        base = rows_with_pct(oofs, how, args.min_hist)
        base = base[(base["Date"] >= lo) & (base["Date"] <= hi)].reset_index(drop=True)
        print(f"\n■ 百分位の取り方: {HOWS[how]}"
              + ("（その日より前の分布。先の値を見ない。本命）" if how == "past"
                 else "（OOF 全体の分布。実運用の追跡と同じ約束。参考）"))
        print(HEAD)
        runs = [(nm, md, ag, 0, args.slots) for nm, md, ag in PATTERNS]
        runs.append((REFERENCE[0], REFERENCE[1], REFERENCE[2], L.SKIP_BREAKS, L.TOP_K))
        for name, models, agree, min_break, top_k in runs:
            passed, sim, taken = run_pattern(base, bars, bar_days, raw_open, models=models, agree=agree,
                                             min_break=min_break, top_k=top_k, slots=args.slots,
                                             unit=args.unit)
            n_pass = int(len(passed))
            if taken.empty:
                s = stats(empty_trades(), pd.Series(dtype=float), lo, hi, slots=args.slots, n_pass=n_pass)
                print(line(name, s))
                summary.append({"how": HOWS[how], "pattern": name, "agree": agree, **s})
                continue
            done = taken[taken["status"].isin(DONE)].copy()
            n_open = int((~taken["status"].isin(DONE)).sum())
            n_full = int((sim["taken"] == L.FULL).sum())
            curve = capital_curve(done, bar_days)
            s = stats(done, curve, lo, hi, slots=args.slots, n_full=n_full, n_pass=n_pass, n_open=n_open)
            print(line(name, s))
            summary.append({"how": HOWS[how], "pattern": name, "agree": agree, **s})
            # 年ごと（買った日の年）
            for y in range(lo.year, hi.year + 1):
                ylo = max(pd.Timestamp(f"{y}-01-01"), lo)
                yhi = min(pd.Timestamp(f"{y}-12-31"), hi)
                dy = done[done["buy_date"].dt.year == y]
                sy = sim[sim["Date"].dt.year == y]
                st = stats(dy, curve, ylo, yhi, slots=args.slots,
                           n_full=int((sy["taken"] == L.FULL).sum()),
                           n_pass=int((passed["Date"].dt.year == y).sum()))
                by_year.append({"how": HOWS[how], "pattern": name, "year": y, **st})
            t = done.copy()
            t.insert(0, "pattern", name)
            t.insert(0, "how", HOWS[how])
            trades.append(t)
        print(f"  （取引 = 枠に入って終わった取引。年間 = 取引 ÷ 年数。到達 = +{L.TAKE_PROFIT:.0f}% に届いた割合。"
              f"買付中央 = 1取引の買付額の中央値。必要資金 = 最大同時投下資本。")
        print(f"    回転 = 年間買付額 ÷ 必要資金。稼働 = 枠が埋まっていた営業日の割合。万円）")

    # ---- 年ごと（本命の取り方だけ表にする。全部は CSV）----
    main_how = HOWS[hows[0]]
    print(f"\n■ 年ごと（{main_how}、買った日の年。候補 = 基準を満たした件数、満杯 = 枠が無くて見送った件数）")
    for name, *_ in list(PATTERNS) + [REFERENCE]:
        rows = [r for r in by_year if r["how"] == main_how and r["pattern"] == name]
        if not rows:
            continue
        print(f"\n  {name}")
        print(YEAR_HEAD)
        for r in rows:
            print(year_line(r["year"], r))

    # ---- 補足 ----
    print("\n■ 補足")
    for r in summary:
        if r["n"]:
            print(f"  {r['how']} / {r['pattern']}: 満了を5日平均終値で売った場合の1取引 {_pct(r['mean_ma5'])}%"
                  f"（終値 {_pct(r['mean'])}%）、最悪の1取引 {_pct(r['worst'])}% / {_man(r['pnl_worst'])}万円、"
                  f"1取引の買付額の最大 {_man(r['buy_max'])}万円、途中の玉 {r['n_open']}件、"
                  f"回転（÷平均投下資本）{_pct(r['turnover_avg'], '{:.1f}')}、"
                  f"年あたり損益 ÷ 必要資金 {_pct(r['roi_max'], '{:+.1f}')}%"
                  + (f"、買付額が取れない {r['n_no_capital']}件" if r["n_no_capital"] else ""))

    pd.DataFrame(summary).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_summary.csv"), index=False)
    pd.DataFrame(by_year).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_by_year.csv"), index=False)
    if trades:
        cols = ["how", "pattern", "Code", "Date", "buy_date", "exit_date", "status", "days", "rank", "slot",
                "p_min", "entry", "raw_open", "capital", "ret_exec", "ret", "ret_close", "pnl", "pnl_ma5"]
        pd.concat(trades, ignore_index=True)[cols].to_csv(
            os.path.join(OOF_DIR, f"{args.prefix}_trades.csv"), index=False)
    log(f"書いた: {OOF_DIR}/{args.prefix}_summary.csv / _by_year.csv / _trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
