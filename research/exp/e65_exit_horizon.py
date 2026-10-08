#!/usr/bin/env python3
"""
実験65: 20営業日を越えて持つ出口は、今の売りのルールより期待値が高いか。

運用者の問い（2026-10-08）
  今の売りのルール（docs/OPERATIONS.md「売りのルール」）は
    - 買った日を1日目として 20営業日以内に、高値が 買値×1.10 に届いたらその値で売る（指値）
    - 届かなければ 20日目の終値で売る。ただし終値が買値を下回っていたら売らずに持ち越す
  ブレイク銘柄なので「+10% に届いても売らずに持つ」「20日目にプラスでも持ち続ける」もあり得る。
  期待値では、売らずに持つほうが良いのかを確かめたい。

何を測るか
  本番5モデル（LightGBM / XGBoost / CatBoost / ロジスティック回帰 / MLP）の out-of-fold
  （Release の oof.parquet / <algo>_oof.parquet）に、画面と同じ「過去分布の百分位」
  （実験59 の pct_expanding）を付けて、2つの選定で銘柄を選ぶ。
    現行       全5モデル 95以上・その日のベスト1件（運用者の決定 2026-10-03。実験59 と同じ）
    GBDT3 90   ブースティング3モデルすべて 90以上（実験41 と同じ基準。件数が多いので再現性の確認用）
  選んだ銘柄を翌営業日の寄り（AdjO[t+1]）で買い、1〜120日目の四本値（実験41 の forward）で
  出口の規則を当てる。規則は exit_rule で組み合わせる（約定の仮定は実験41 と同じ保守側）。
    指値      1〜tp_days 日目に高値が 買値×(1+tp) に届けば、ちょうど +tp で売る（飛び越えても）
    decide    decide 日目の終値が買値以上なら plus の扱い（sell = その終値で売る / hold = 持ち続ける）、
              買値未満なら minus の扱い（sell = その終値で売る / carry = 持ち越す）
    持ち越し  翌日から 買値×(1+be) の指値（プラ転でちょうど be で売る）
    上限      cap 日目の終値で必ず売る
  比べる出口
    20日目の終値 / +10%指値・20日目 / 現行（+10%指値・20日目プラスで売り・マイナスは建値まで、上限 40/60/120）
    指値なしで20日目プラスで売り（マイナスは建値まで）/ +10%到達後も持つ（20日目プラスは上限まで持つ）
    40・60日目の終値 / +10%・+20%・+30% の指値を 40 日（60 日）置く
  1取引ごと（上限の日 120 まで値動きが揃う取引だけ。全出口で同じ取引）と、枠3 の模擬
  （実験41 の slot_sim。資金は買うたびに「現金＋建玉の簿価」の 1/3、月利は実現ベース）で出す。
  再現性は、本番の out-of-fold と同じ窓（36/6/6か月・エンバーゴ20営業日）ごとと、年ごとで見る。
  窓の切り方をずらした確認（実験41 §6）は、5モデルを学習し直さないとできないのでここではしない。

  手数料・税・配当・スリッページは含まない。+X% の指値は「高値が届けば約定」。持ち越した玉の
  プラ転も「高値が建値に届けば約定」。上限の日・decide の日は終値で売れるとみなす。

使い方
    python3 research/exp/e65_exit_horizon.py [--oof-dir research/_data] [--k 120]
結果は research/_data/oof/e65_*.csv。**本番の設定には書かない。**
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import live_track as L  # noqa: E402
import walkforward as WF  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e59_unit_sim as E59  # noqa: E402

OOF_DIR = os.path.join(L.DATA_DIR, "oof")

K = 120                      # 追う日数（上限の最長）
DECIDE = 20                  # 今のルールで終値を見る日（買った日 = 1日目）
TP = 0.10                    # 今のルールの指値
CAPS = (40, 60, 120)         # 持ち越しの上限
SLOTS = 3
SLOT_LIST = (1, 2, 3, 5)      # 枠の数を変えたときの確認
MIN_HIST = E59.MIN_HIST
BASE = "20日目の終値"         # 窓ごとの勝ち負けの比べ相手
CURRENT = "現行（+10%指値・20日目プラスで売り・マイナスは建値まで）上限60日"
YEAR_DAYS = 365.25

#: 選定。(名前, 合議のモデル, 百分位, 1取引ごとの top_k, 枠の模擬の (min_break, top_k))
SELECTIONS = (
    ("現行: 全5モデル 95以上・1日1件", L.ALL5, 95.0, 1, (0, 1)),
    ("GBDT3 90以上（全件）", L.BOOST, 90.0, 10 ** 6, (L.SKIP_BREAKS, L.TOP_K)),
)

WHY = {1: "指値", 2: "decide日プラス", 3: "プラ転", 4: "上限", 6: "decide日マイナス"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- #
# 出口の規則
# ---------------------------------------------------------------------- #

def exit_rule(P: dict, *, cap: int, tp: Optional[float] = None, tp_days: Optional[int] = None,
              decide: int = DECIDE, plus: str = "sell", minus: str = "sell", be: float = 0.0):
    """
    値動きの表 P（実験41 の forward。O/H/L/C は (行, 日) で列 j が j+1 日目、entry は1日目の始値）に
    出口の規則を当てる。戻り値は (収益率, 手仕舞った日, 理由)。収益率は小数（0.1 = +10%）。

      1〜tp_days 日目      高値が 買値×(1+tp) 以上なら、ちょうど +tp で売る（tp=None なら指値なし）
      decide 日目の終値    買値以上: plus が "sell" ならその終値で売る、"hold" なら持ち続ける
                           買値未満: minus が "sell" ならその終値で売る、"carry" なら持ち越す
      持ち越した玉         decide+1 日目から、高値が 買値×(1+be) 以上なら ちょうど be で売る
      cap 日目             まだ持っていれば終値で売る（終値が無い日は、それまでの最後の終値）
      同じ日に複数が当たれば、指値 → プラ転 → 終値 の順（実験41 の plan_exit と同じ）

    理由: 1 指値 / 2 decide 日目にプラスで売った / 3 プラ転で売った / 4 上限の日に売った /
          6 decide 日目にマイナスで売った。cap 日目まで終値が1つも無ければ NaN（数えない）。
    """
    if plus not in ("sell", "hold") or minus not in ("sell", "carry"):
        raise ValueError(f"plus は sell/hold、minus は sell/carry: {plus!r} {minus!r}")
    if decide > cap:
        raise ValueError(f"decide ({decide}) は cap ({cap}) 以下")
    tp_days = decide if tp_days is None else tp_days
    e = P["entry"]
    H, C = P["H"], P["C"]
    m = len(e)
    cap = min(cap, H.shape[1])
    T = e * (1.0 + tp) if tp is not None else None
    BE = e * (1.0 + be)
    ret = np.full(m, np.nan)
    day = np.full(m, np.nan)
    why = np.zeros(m, dtype=np.int8)
    done = ~np.isfinite(e)                 # 買えなかった行は最初から対象外
    carried = np.zeros(m, dtype=bool)
    lastc = np.full(m, np.nan)

    def settle(mask, val, k, w):
        if mask.any():
            ret[mask] = val[mask] if isinstance(val, np.ndarray) else val
            day[mask] = k + 1
            why[mask] = w
            done[mask] = True

    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(cap):
            h, c = H[:, k], C[:, k]
            if T is not None and k < tp_days:
                settle(~done & (h >= T), tp, k, 1)
            if k >= decide:
                settle(~done & carried & (h >= BE), be, k, 3)
            lastc = np.where(np.isfinite(c), c, lastc)
            if k + 1 == decide:
                known = ~done & np.isfinite(lastc)
                up, dn = known & (lastc >= e), known & (lastc < e)
                if plus == "sell":
                    settle(up, lastc / e - 1.0, k, 2)
                if minus == "sell":
                    settle(dn, lastc / e - 1.0, k, 6)
                else:
                    carried |= dn
            if k + 1 == cap:
                settle(~done & np.isfinite(lastc), lastc / e - 1.0, k, 4)
    return ret, day, why


def rules() -> List[Tuple[str, dict]]:
    """比べる出口（名前, exit_rule の引数）。順番は表の並び。"""
    out = [(BASE, dict(cap=DECIDE)),
           ("+10%指値・20日目", dict(cap=DECIDE, tp=TP))]
    for c in CAPS:
        out.append((f"現行（+10%指値・20日目プラスで売り・マイナスは建値まで）上限{c}日",
                    dict(cap=c, tp=TP, minus="carry")))
    for c in CAPS:
        out.append((f"指値なし・20日目プラスで売り・マイナスは建値まで 上限{c}日",
                    dict(cap=c, minus="carry")))
    for c in (40, 60):
        out.append((f"+10%指値・20日目プラスは持ち続け・マイナスは建値まで 上限{c}日",
                    dict(cap=c, tp=TP, plus="hold", minus="carry")))
    for c in (40, 60):
        out.append((f"指値なし・20日目プラスは持ち続け・マイナスは建値まで 上限{c}日",
                    dict(cap=c, plus="hold", minus="carry")))
    for c in (40, 60):
        out.append((f"{c}日目の終値", dict(cap=c, decide=c)))
    out += [("+10%指値・40日目", dict(cap=40, tp=TP, tp_days=40, decide=40)),
            ("+20%指値・40日目", dict(cap=40, tp=0.20, tp_days=40, decide=40)),
            ("+30%指値・40日目", dict(cap=40, tp=0.30, tp_days=40, decide=40)),
            ("+30%指値・60日目", dict(cap=60, tp=0.30, tp_days=60, decide=60))]
    return out


# ---------------------------------------------------------------------- #
# 窓（本番の out-of-fold と同じ切り方）
# ---------------------------------------------------------------------- #

def fold_of(dates: pd.Series, train_from: str, train_to: str) -> np.ndarray:
    """
    本番の OOF と同じ窓（36/6/6か月・エンバーゴ RISE_HORIZON）に各行を割り当てる。
    Release の OOF には fold 列が無いので、meta.json の trainFrom / trainTo から作り直す。
    どの窓にも入らない行は 0。
    """
    fs = WF.make_folds(pd.Series(pd.to_datetime([train_from, train_to])),
                       min_train_months=36, test_months=6, step_months=6,
                       embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = np.zeros(len(d), dtype=int)
    for f in fs:
        m = (d >= np.datetime64(pd.Timestamp(f.test_start))) & (d <= np.datetime64(pd.Timestamp(f.test_end)))
        out[m] = f.index
    return out


# ---------------------------------------------------------------------- #
# 集計
# ---------------------------------------------------------------------- #

def _mean(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def arm_stats(ret: np.ndarray, day: np.ndarray, why: np.ndarray, base: np.ndarray,
              fold: np.ndarray) -> dict:
    """1取引ごとの成績（ret は小数）。base は比べ相手（同じ行）の収益。"""
    ok = np.isfinite(ret) & np.isfinite(base)
    r, d, w, b, f = ret[ok] * 100, day[ok], why[ok], base[ok] * 100, fold[ok]
    n = int(len(r))
    diff = r - b
    per_fold = []
    for k in np.unique(f):
        m = (f == k)
        if m.sum() >= 5:
            per_fold.append(float(diff[m].mean()))
    pf = np.array(per_fold)
    out = {"n": n,
           "mean": float(r.mean()) if n else float("nan"),
           "median": float(np.median(r)) if n else float("nan"),
           "sd": float(r.std(ddof=1)) if n > 1 else float("nan"),
           "se": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
           "win": float((r > 0).mean() * 100) if n else float("nan"),
           "p10": float(np.quantile(r, 0.1)) if n else float("nan"),
           "worst": float(r.min()) if n else float("nan"),
           "lt_m10": float((r < -10).mean() * 100) if n else float("nan"),
           "days": float(d.mean()) if n else float("nan"),
           "per_day": float(r.mean() / d.mean()) if n else float("nan"),
           "diff": float(diff.mean()) if n else float("nan"),
           "diff_se": float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
           "folds_won": int((pf > 0).sum()), "folds": int(len(pf)),
           "fold_diff_mean": float(pf.mean()) if len(pf) else float("nan"),
           "fold_diff_se": float(pf.std(ddof=1) / math.sqrt(len(pf))) if len(pf) > 1 else float("nan"),
           "worst_fold": float(pf.min()) if len(pf) else float("nan")}
    for code, name in WHY.items():
        out[f"why_{code}"] = float((w == code).mean() * 100) if n else float("nan")
        out[f"why_{code}_mean"] = float(r[w == code].mean()) if (w == code).any() else float("nan")
    out["carried"] = out["why_3"] + out["why_4"] if n else float("nan")   # 持ち越した割合（プラ転 + 上限）
    return out


def path_stats(P: dict, sel: pd.DataFrame, k_max: int) -> List[dict]:
    """
    選んだ銘柄のその後（買値からの終値、%）。全体 / 20日目の損益で分けて / 20日以内に +10% に
    届いたかで分けて。実験41 §3 と同じ読み方で、今の選定で出し直す。
    """
    e = P["entry"]
    C, H = P["C"], P["H"]
    with np.errstate(invalid="ignore", divide="ignore"):
        close_at = {h: C[:, h - 1] / e - 1.0 for h in (5, 10, 20, 40, 60, 120) if h <= k_max}
        hit10_20 = np.nanmax(H[:, :20], axis=1) >= e * (1.0 + TP)
        reach = {}
        for x in (0.10, 0.20, 0.30):
            for n in (20, 40, 60):
                reach[(x, n)] = np.nanmax(H[:, :n], axis=1) >= e * (1.0 + x)
    c20 = close_at[20]
    groups = [("全体", np.ones(len(e), dtype=bool)),
              ("20日目プラス", c20 >= 0), ("20日目マイナス", c20 < 0),
              ("20日以内に+10%到達", hit10_20), ("20日以内に+10%未到達", ~hit10_20),
              ("+10%到達かつ20日目プラス", hit10_20 & (c20 >= 0)),
              ("+10%未到達かつ20日目プラス", ~hit10_20 & (c20 >= 0))]
    rows = []
    for name, m in groups:
        row = {"group": name, "n": int(m.sum())}
        for h, v in close_at.items():
            row[f"c{h}"] = _mean(v[m]) * 100
            row[f"c{h}_med"] = float(np.nanmedian(v[m]) * 100) if m.any() else float("nan")
            row[f"c{h}_win"] = float(np.nanmean(v[m] > 0) * 100) if m.any() else float("nan")
        if 40 in close_at:
            row["add_20_40"] = _mean((close_at[40] - c20)[m]) * 100
            row["add_20_40_win"] = float(np.nanmean(((close_at[40] - c20) > 0)[m]) * 100) if m.any() else float("nan")
        if 60 in close_at:
            row["add_20_60"] = _mean((close_at[60] - c20)[m]) * 100
        for (x, n), v in reach.items():
            row[f"reach{int(x*100)}_{n}"] = float(np.nanmean(v[m]) * 100) if m.any() else float("nan")
        rows.append(row)
    return rows


def after_hit_stats(P: dict, tp: float = TP, decide: int = DECIDE) -> dict:
    """
    decide 日以内に高値が 買値×(1+tp) に届いた玉が、その後どこまで行ったか（運用者の問い 2026-10-08
    「+10% に到達しても放置していれば +15% に届く可能性が高いということか」への答え）。
      reach{X}_{N}  届いた日から N 日目までに高値が +X% に届いた割合
      end20_ge15 / end20_ge10 / end20_0_10 / end20_lt0  20日目の終値の位置（買値比）
      diff20  「+tp で売った」と「20日目の終値」の差（放置 − 売却）の平均 pt、win20 は放置が勝った割合
      pull_*  届いた日から20日目までの安値の最小（買値比）
    """
    e = P["entry"]
    H, C, Lo = P["H"], P["C"], P["L"]
    with np.errstate(invalid="ignore", divide="ignore"):
        touched = H[:, :decide] >= e[:, None] * (1.0 + tp)
        hit = touched.any(axis=1) & np.isfinite(e)
        day = np.where(hit, touched.argmax(axis=1) + 1, np.nan)
        idx = np.where(hit)[0]
        amax20 = np.full(len(e), np.nan)
        amax40 = np.full(len(e), np.nan)
        amin20 = np.full(len(e), np.nan)
        for i in idx:
            d = int(day[i]) - 1
            amax20[i] = np.nanmax(H[i, d:decide]) / e[i] - 1.0
            amax40[i] = np.nanmax(H[i, d:40]) / e[i] - 1.0
            amin20[i] = np.nanmin(Lo[i, d:decide]) / e[i] - 1.0
        c20 = C[:, decide - 1] / e - 1.0
        c40 = C[:, 39] / e - 1.0
    g = hit
    n = int(g.sum())
    if not n:
        return {"n": 0}
    pc = lambda m: float(np.nanmean(m[g]) * 100)
    out = {"n": n, "share": float(n / np.isfinite(e).sum() * 100), "day_med": float(np.nanmedian(day[g])),
           "reach15_20": pc(amax20 >= 0.15), "reach20_20": pc(amax20 >= 0.20),
           "reach15_40": pc(amax40 >= 0.15), "reach20_40": pc(amax40 >= 0.20), "reach30_40": pc(amax40 >= 0.30),
           "end20_ge15": pc(c20 >= 0.15), "end20_ge10": pc(c20 >= tp), "end20_0_10": pc((c20 >= 0) & (c20 < tp)),
           "end20_lt0": pc(c20 < 0), "end20_mean": pc(c20), "end20_med": float(np.nanmedian(c20[g]) * 100),
           "end20_p10": float(np.nanquantile(c20[g], 0.1) * 100), "end20_worst": float(np.nanmin(c20[g]) * 100),
           "diff20": float(np.nanmean(c20[g] - tp) * 100), "win20": pc(c20 >= tp),
           "end40_ge15": pc(c40 >= 0.15), "end40_ge10": pc(c40 >= tp), "end40_lt0": pc(c40 < 0),
           "end40_mean": pc(c40), "end40_med": float(np.nanmedian(c40[g]) * 100),
           "end40_p10": float(np.nanquantile(c40[g], 0.1) * 100),
           "pull_med": float(np.nanmedian(amin20[g]) * 100), "pull_p25": float(np.nanquantile(amin20[g], 0.25) * 100),
           "pull_p10": float(np.nanquantile(amin20[g], 0.1) * 100),
           "pull_lt5": pc(amin20 < 0.05), "pull_lt0": pc(amin20 < 0.0)}
    for lab, m in (("early", g & (day <= 10)), ("late", g & (day > 10))):
        out[f"{lab}_n"] = int(m.sum())
        if m.sum():
            out[f"{lab}_end20"] = float(np.nanmean(c20[m]) * 100)
            out[f"{lab}_ge10"] = float(np.mean(c20[m] >= tp) * 100)
            out[f"{lab}_reach15_20"] = float(np.mean(amax20[m] >= 0.15) * 100)
            out[f"{lab}_end40"] = float(np.nanmean(c40[m]) * 100)
    return out


def by_year(ret: np.ndarray, dates: pd.Series) -> Dict[int, Tuple[int, float]]:
    y = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
    out = {}
    for yy in np.unique(y):
        m = (y == yy) & np.isfinite(ret)
        if m.sum():
            out[int(yy)] = (int(m.sum()), float(ret[m].mean() * 100))
    return out


# ---------------------------------------------------------------------- #
# 印字
# ---------------------------------------------------------------------- #

def pc(x, w=8, d=2, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return f"{'-':>{w}}"
    return f"{x:{'+' if sign else ''}{w}.{d}f}"


ARM_HEAD = (f"  {'出口':<60}{'平均':>8}{'中央値':>8}{'勝率':>6}{'SD':>7}{'下位10%':>9}{'最悪':>8}{'−10%未満':>9}"
            f"{'保有日':>7}{'1日あたり':>9}{'指値':>6}{'持越し':>7}{'プラ転':>7}{'上限':>6}{'上限の平均':>10}"
            f"{'差':>7}{'対SE':>6}{'勝ち窓':>7}{'窓差':>7}{'最悪窓':>8}")


def arm_line(name: str, s: dict) -> str:
    z = s["diff"] / s["diff_se"] if s.get("diff_se") and np.isfinite(s["diff_se"]) and s["diff_se"] > 0 else float("nan")
    return (f"  {name:<60}{pc(s['mean'])}{pc(s['median'])}{s['win']:>5.0f}%{pc(s['sd'], 7, 2, False)}"
            f"{pc(s['p10'], 9)}{pc(s['worst'], 8, 1)}{s['lt_m10']:>8.1f}%{s['days']:>7.1f}{pc(s['per_day'], 9, 3)}"
            f"{s['why_1']:>5.1f}%{s['carried']:>6.1f}%{s['why_3']:>6.1f}%{s['why_4']:>5.1f}%{pc(s['why_4_mean'], 10, 1)}"
            f"{pc(s['diff'], 7)}{pc(z, 6, 1)}{s['folds_won']:>4}/{s['folds']:<2}{pc(s['fold_diff_mean'], 7)}"
            f"{pc(s['worst_fold'], 8)}")


SLOT_HEAD = (f"  {'出口':<60}{'取引':>5}{'見送り':>6}{'月の取引':>8}{'保有日':>7}{'稼働率':>7}{'1取引':>8}{'月利平均':>9}"
             f"{'月利中央':>9}{'最悪の月':>9}{'年率':>8}{'最大DD':>8}{'最悪':>8}")


def slot_line(name: str, r: dict) -> str:
    if not r.get("taken"):
        return f"  {name:<60} （取引なし）"
    return (f"  {name:<60}{r['taken']:>5}{r['skipped']:>6}{r['per_month']:>8.2f}{r['days']:>7.1f}{r['util']*100:>6.0f}%"
            f"{pc(r['mean']*100, 8)}{pc(r['m_mean']*100, 9)}{pc(r['m_med']*100, 9)}{pc(r['m_worst']*100, 9, 1)}"
            f"{pc(r['cagr']*100, 8, 1)}{pc(r['mdd']*100, 8, 1)}{pc(r['worst']*100, 8, 1)}")


PATH_HEAD = (f"  {'区分':<26}{'件数':>6}{'5日':>8}{'10日':>8}{'20日':>8}{'40日':>8}{'60日':>8}{'120日':>8}"
             f"{'20→40':>8}{'勝率':>6}{'20→60':>8}{'+10%/20':>8}{'+20%/40':>8}{'+30%/40':>8}{'+30%/60':>8}")


def path_line(r: dict) -> str:
    return (f"  {r['group']:<26}{r['n']:>6}{pc(r.get('c5'))}{pc(r.get('c10'))}{pc(r.get('c20'))}{pc(r.get('c40'))}"
            f"{pc(r.get('c60'))}{pc(r.get('c120'))}{pc(r.get('add_20_40'))}{r.get('add_20_40_win', float('nan')):>5.0f}%"
            f"{pc(r.get('add_20_60'))}{r.get('reach10_20', float('nan')):>7.1f}%{r.get('reach20_40', float('nan')):>7.1f}%"
            f"{r.get('reach30_40', float('nan')):>7.1f}%{r.get('reach30_60', float('nan')):>7.1f}%")


# ---------------------------------------------------------------------- #
# 本体
# ---------------------------------------------------------------------- #

def select(base: pd.DataFrame, models: Sequence[str], pct: float, top_k: int,
           min_break: int = 0) -> pd.DataFrame:
    """live_track.decide（画面と同じ判断）で選ぶ。1日 top_k 件、発火 min_break 件以上の日だけ。"""
    _, _, picks = L.decide(base, agree=pct, min_break=min_break, top_k=top_k, models=models)
    return picks.sort_values(["Date", "rank"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験65: 20営業日を越えて持つ出口の期待値")
    ap.add_argument("--data-dir", default=L.DATA_DIR)
    ap.add_argument("--oof-dir", default=L.DATA_DIR)
    ap.add_argument("--model-dir", default=L.MODEL_DIR)
    ap.add_argument("--k", type=int, default=K, help="追う日数（上限の最長）")
    ap.add_argument("--min-hist", type=int, default=MIN_HIST)
    ap.add_argument("--slots", type=int, default=SLOTS)
    ap.add_argument("--prefix", default="e65")
    args = ap.parse_args(argv)
    os.makedirs(OOF_DIR, exist_ok=True)
    k_max = args.k

    oofs = E59.load_oofs(args.oof_dir, args.model_dir)
    meta_path = os.path.join(args.model_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    train_from = meta.get("trainFrom", "2018-04-03")
    lo_oof = min(o["Date"].min() for o in oofs.values())
    hi_oof = max(o["Date"].max() for o in oofs.values())
    train_to = meta.get("trainTo", str(hi_oof.date()))
    log(f"OOF {lo_oof.date()}〜{hi_oof.date()}（{len(oofs['lgbm']):,}行）。窓は {train_from} 起点")

    base = E59.rows_with_pct(oofs, "past", args.min_hist)
    base["fold"] = fold_of(base["Date"], train_from, train_to)
    codes = set(base["Code"].astype(str))
    bars = E41.load_bars(codes)
    cal = pd.Series(np.sort(bars["Date"].unique()))
    log(f"日足 {len(bars):,}行（{len(codes):,}銘柄、〜{pd.Timestamp(cal.iloc[-1]).date()}）")

    print("=" * 120)
    print(f"実験65 20営業日を越えて持つ出口の期待値（1取引ごと・枠{args.slots}の模擬）")
    print("=" * 120)
    print("  約定: 指値・プラ転はちょうどその値（高値が届けば）。decide の日・上限の日は終値。手数料・税・配当は含まない。")
    print(f"  1取引ごとの表は、上限の日（{k_max}日）まで値動きが揃う取引だけ（全出口で同じ取引）。")
    print(f"  差・対SE・勝ち窓・窓差・最悪窓は「{BASE}」と比べたもの（同じ取引の差。窓は本番の OOF と同じ切り方）。")

    arms_out, folds_out, years_out, slots_out, paths_out, after_out = [], [], [], [], [], []
    for name, models, pct, top_k, (mb, tk) in SELECTIONS:
        sel = select(base, models, pct, top_k)
        if not len(sel):
            print(f"\n■ {name}: 選定なし")
            continue
        keys = sel[["Code", "Date"]].copy()
        keys["Code"] = keys["Code"].astype(str)
        P = E41.forward(bars, keys, k=k_max)
        full = P["ok"][:, k_max - 1] & np.isfinite(P["entry"])
        Pf = {a: (v[full] if isinstance(v, np.ndarray) else v) for a, v in P.items()}
        sf = sel[full].reset_index(drop=True)
        fold = sf["fold"].to_numpy()
        span = f"{pd.Timestamp(sf['Date'].min()).date()}〜{pd.Timestamp(sf['Date'].max()).date()}"
        print(f"\n■ {name}: 基準を満たした {len(sel):,}件（{pd.Timestamp(sel['Date'].min()).date()}〜"
              f"{pd.Timestamp(sel['Date'].max()).date()}）、うち {k_max}日まで揃う {len(sf):,}件（{span}）")

        res = {n: exit_rule(Pf, **kw) for n, kw in rules()}
        b_ret = res[BASE][0]
        print(f"\n  --- 1取引ごと（{len(sf):,}件）---")
        print(ARM_HEAD)
        for n, (r_, d_, w_) in res.items():
            s = arm_stats(r_, d_, w_, b_ret, fold)
            print(arm_line(n, s))
            arms_out.append({"selection": name, "rule": n, **s})
            for k in np.unique(fold):
                m = (fold == k) & np.isfinite(r_) & np.isfinite(b_ret)
                if m.sum():
                    folds_out.append({"selection": name, "rule": n, "fold": int(k), "n": int(m.sum()),
                                      "mean": float(r_[m].mean() * 100), "base": float(b_ret[m].mean() * 100),
                                      "days": float(d_[m].mean())})
            for yy, (n_y, m_y) in by_year(r_, sf["Date"]).items():
                years_out.append({"selection": name, "rule": n, "year": yy, "n": n_y, "mean": m_y})
        print("  （指値 = 指値で売れた割合、持越し = 20日目にマイナスで持ち越した割合、プラ転 = 建値で売れた割合、"
              "上限 = 上限の日まで戻らず売った割合とその平均。差 = 比べ相手との同じ取引の差、対SE = 差 ÷ 差の標準誤差）")

        print(f"\n  --- 選んだ銘柄のその後（買値からの終値の平均 %、{len(sf):,}件）---")
        print(PATH_HEAD)
        for r in path_stats(Pf, sf, k_max):
            print(path_line(r))
            paths_out.append({"selection": name, **r})
        print("  （20→40 = 20日目から40日目までの追加分の平均と、その勝率。+10%/20 = 20日以内に高値が +10% に届いた割合、"
              "+20%/40 = 40日以内に +20%、+30%/40 = 40日以内に +30%、+30%/60 = 60日以内に +30%）")

        a = after_hit_stats(Pf)
        after_out.append({"selection": name, **a})
        if a["n"]:
            print(f"\n  --- 20日以内に +{TP*100:.0f}% に届いた {a['n']}件（{a['share']:.1f}%、届いた日の中央値 {a['day_med']:.0f}日目）のその後 ---")
            print(f"  届いた後さらに: +15% に20日以内 {a['reach15_20']:.1f}% / +20% に20日以内 {a['reach20_20']:.1f}% / "
                  f"+15% に40日以内 {a['reach15_40']:.1f}% / +20% に40日以内 {a['reach20_40']:.1f}% / +30% に40日以内 {a['reach30_40']:.1f}%")
            print(f"  20日目の終値: +15%以上 {a['end20_ge15']:.1f}% / +10%以上 {a['end20_ge10']:.1f}% / 0〜+10% {a['end20_0_10']:.1f}% / "
                  f"マイナス {a['end20_lt0']:.1f}%。平均 {a['end20_mean']:+.1f}% 中央値 {a['end20_med']:+.1f}% 下位10% {a['end20_p10']:+.1f}% 最悪 {a['end20_worst']:+.1f}%")
            print(f"  放置 − +10%で売却（20日目）: 平均 {a['diff20']:+.1f}pt、放置が勝った割合 {a['win20']:.0f}%")
            print(f"  40日目の終値: +15%以上 {a['end40_ge15']:.1f}% / +10%以上 {a['end40_ge10']:.1f}% / マイナス {a['end40_lt0']:.1f}%。"
                  f"平均 {a['end40_mean']:+.1f}% 中央値 {a['end40_med']:+.1f}% 下位10% {a['end40_p10']:+.1f}%")
            print(f"  届いた後の押し（安値の最小、買値比）: 中央値 {a['pull_med']:+.1f}% / 下位25% {a['pull_p25']:+.1f}% / 下位10% {a['pull_p10']:+.1f}%。"
                  f"+5% を割る {a['pull_lt5']:.1f}% / 買値を割る {a['pull_lt0']:.1f}%")
            for lab, ja in (("early", "10日目までに届いた"), ("late", "11〜20日目に届いた")):
                if a.get(f"{lab}_n"):
                    print(f"  {ja} {a[f'{lab}_n']}件: 20日目 平均 {a[f'{lab}_end20']:+.1f}%、+10%以上で終える {a[f'{lab}_ge10']:.0f}%、"
                          f"+15% に20日以内 {a[f'{lab}_reach15_20']:.0f}%、40日目 平均 {a[f'{lab}_end40']:+.1f}%")

        # 年ごと（主な出口）
        main_rules = [BASE, "+10%指値・20日目", CURRENT,
                      "指値なし・20日目プラスは持ち続け・マイナスは建値まで 上限40日", "40日目の終値", "+30%指値・40日目"]
        print("\n  --- 年ごと（買った日の年。件数 / 1取引の平均 %）---")
        yrs = sorted({r["year"] for r in years_out if r["selection"] == name})
        print(f"  {'出口':<60}" + "".join(f"{y:>14}" for y in yrs))
        for n in main_rules:
            cells = []
            for y in yrs:
                hit = [r for r in years_out if r["selection"] == name and r["rule"] == n and r["year"] == y]
                cells.append(f"{hit[0]['n']:>4} {hit[0]['mean']:>+8.2f}" if hit else f"{'-':>14}")
            print(f"  {n:<60}" + "".join(f"{c:>14}" for c in cells))

        # 枠の模擬
        sig = select(base, models, pct, tk, min_break=mb)
        sig = sig.merge(sf[["Code", "Date"]].assign(i=np.arange(len(sf))), on=["Code", "Date"], how="inner")
        sig = sig.sort_values(["Date", "rank"]).reset_index(drop=True)
        print(f"\n  --- 枠{args.slots}の模擬（1日 {tk}件・発火 {mb}件以上の日、候補 {len(sig):,}件。実験41 の slot_sim）---")
        print(SLOT_HEAD)
        for n, (r_, d_, w_) in res.items():
            r = E41.slot_sim(sig, cal, r_, d_, w_, slots=args.slots)
            print(slot_line(n, r))
            slots_out.append({"selection": name, "rule": n, **{k: v for k, v in r.items()}})
        print("  （見送り = 枠が無くて見送った候補。月利は実現ベース（売った月に数える。持っている間は簿価）。"
              "年率 = 月利の複利を年に直したもの。1本あたりの取引が少ないので、水準ではなく順序を見る）")

        # 枠の数を変えても順序が同じかを見る（主な出口だけ）。年率 / 取引数
        print("\n  --- 枠の数を変えたとき（年率 % / 取引数。主な出口だけ）---")
        print(f"  {'出口':<60}" + "".join(f"{'枠' + str(k):>16}" for k in SLOT_LIST))
        for n in main_rules:
            r_, d_, w_ = res[n]
            cells = []
            for k in SLOT_LIST:
                r = E41.slot_sim(sig, cal, r_, d_, w_, slots=k)
                slots_out.append({"selection": name, "rule": n, "slots_alt": k,
                                  **{kk: v for kk, v in r.items()}})
                cells.append(f"{r['cagr']*100:>+8.1f} /{r['taken']:>4}" if r.get("taken") else f"{'-':>16}")
            print(f"  {n:<60}" + "".join(f"{c:>16}" for c in cells))

    pd.DataFrame(arms_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_arms.csv"), index=False)
    pd.DataFrame(folds_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_folds.csv"), index=False)
    pd.DataFrame(years_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_years.csv"), index=False)
    pd.DataFrame(slots_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_slots.csv"), index=False)
    pd.DataFrame(paths_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_paths.csv"), index=False)
    pd.DataFrame(after_out).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_after10.csv"), index=False)
    log(f"書いた: {OOF_DIR}/{args.prefix}_arms.csv / _folds.csv / _years.csv / _slots.csv / _paths.csv / _after10.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
