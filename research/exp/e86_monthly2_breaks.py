#!/usr/bin/env python3
"""
実験86: 月に2銘柄の先着買いを「発火8件以上の日だけ」に絞ったらどうなるか。

実験85 の報告（2026-10-12）で、月2件の先着買いは画面の規則の「発火8件以上」を使っておらず、候補の少ない静かな日にも買う
ことを書いた。運用者の返事「「月2件＋発火8件以上の日だけ買う」を、比べ方を先に決めて確かめましょうか？ お願いします」。

規則と比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  発火 = その日の候補の数（out-of-fold の行数。画面・live_track.decide と同じ数え方）。発火8件以上 = LT.SKIP_BREAKS 以上
  R0 = 1.2σ の lgbm・xgb・cat の3本とも95以上（運用者の目安）。百分位は前向き（実験79 の window_split、丸めは画面と同じ）。
  テスト窓 3〜11。出口: +10% の指値・届かなければ 20営業日目の終値（運用者の今の売り。実験83 の tp10c）
  主: R0 の先着2件・発火8件以上の日だけ（各月、発火8件以上の日の R0 の行を日付の早い順に。同じ日は3本の最小の順、同点は
      lgbm のスコア順。月に2件まで）
  比べる（今のやり方）: R0 の先着2件（発火を問わない。実験84・85 と同じ）
  判定（3つとも満たせば通る）
    1. 主 − 今のやり方 の窓ごとの差（1取引の平均）の平均が正
    2. 上の窓が過半
    3. 発火の印をずらした絞り込みより上（p < 0.05）: 窓ごとに、候補のある日の並びの上で「発火8件以上」の印を巡回して
       ずらし（ずらす幅は窓ごとに無作為。印の付く日の数はそのまま、続く日のかたまりも窓の端で回り込む所のほかはそのまま）、
       同じ選び方をしたときの窓ごとの差の平均が実際以上になる割合（2,000回）。買える日を減らして月の後ろへずらすこと自体の
       効果と、発火の多さの効果を分ける
  参考（判定に使わない）
    - 今のやり方で買った取引のうち、発火8件未満の日に買ったものの件数と成績
    - R0 の行全体で、発火8件以上の日 − 8件未満の日（窓ごと。並べ替え p と銘柄単位の p。実験82 の split_test）
    - 同じ絞り込みを A15（6本とも95以上）の先着2件と、実験85 の段階的（16日から緩める）に
    - 画面の売買の形（発火8件以上の日に上位2件・3枠・先着順）を R0 で（実験83 の trade_sim。お金の入れ方が違うので
      1取引の成績と件数を並べる）
    - 発火の線を 1（絞らない）/ 4 / 8 / 12 / 16 / 20 件にしたときの R0 の先着2件（見るだけ。線はここから選ばない）
    - 全期間の百分位（画面の物差し）で、主・今のやり方・画面の売買の形
  出すもの: 件数・買った月（2件 / 1件 / 0件）・1取引の平均・中央・勝率・到達・−10%割れ・分位点・月あたりの損益の合計
    （1件を同じ金額で買ったときの、1件の金額に対する%。取引の収益の合計 ÷ 月数）
  確かめ: 発火の数え方（out-of-fold の行数）を、データセットの日ごとの行数と突き合わせる

公開ログには集計だけ（件数・平均・分位点、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e86_monthly2_breaks.py
    args="--draws 2000"       （既定）

結果（2026-10-12、run 38207442369、3a18209、59秒。docs/MODEL_ADOPTION_RULES.md §43。+10% の指値・20日目の終値）
------------------------------------------------------------------
  確かめ: 発火（out-of-fold の行数）がデータセットの日ごとの行数と一致 1,054/1,054日。前向きの R0 543件（実験82・83 と同じ）
  候補のある日 1,054日のうち発火8件以上 718日（68.1%）。R0 の行の 93.2%（543件中506件）は発火8件以上の日に出る
  今のやり方（R0 の先着2件）: 74件（2件の月 32・1件 10・0件 11）−0.26%（中央 +2.55）。うち発火8件未満の日 12件 −2.02%
  主（発火8件以上の日だけ）: 67件（29・9・15）+0.32%（中央 +2.71・10% 点 −11.38・−10%割れ 13.4%）
  窓ごとの差 −0.57 ± 1.66pt（9窓中4）。印をずらした絞り込みは差の平均 +0.73pt、実際以上が p = 0.861 → 通らない
  参考: R0 の行全体で 8件以上の日 506件 +3.25%・8件未満の日 37件 +1.28%（比べられる窓は2つ、p = 0.38）。
        同じ絞り込みを A15 の先着2件に −0.70 ± 1.34pt、段階的に +0.02 ± 0.69pt。発火の線 4〜20件で向きがそろわない
        画面の売買の形（R0・3枠）93件 +2.33%（中央 +4.29）、今のやり方との差 +0.98 ± 2.00pt（9窓中8。先に決めた比較ではない）
        画面の物差し: 今のやり方 63件 +1.63%、発火8件以上の日だけ 56件 +2.43%（差 +0.38 ± 0.63pt、9窓中3）、3枠 80件 +3.67%
  → 発火8件未満の日の取引は74件中12件と少なく、外しても窓ごとの成績は上がらない。日を無作為に外したのと見分けられない
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
import e84_six_mean_top2 as E84  # noqa: E402
import e85_tier_cascade as E85  # noqa: E402

MODELS = E79.MODELS
MAIN = "tp10c"
TOP_N = 2
MIN_BREAK = LT.SKIP_BREAKS      # 8
DRAWS = 2000
SEED = 42
GRID = (1, 4, 8, 12, 16, 20)
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s
qtiles, months_of, window_diff, first_by = E84.qtiles, E84.months_of, E84.window_diff, E84.first_by


# --------------------------------------------------------------------------- #
# 発火と選び方
# --------------------------------------------------------------------------- #

def add_breaks(ft: pd.DataFrame, min_break: int = MIN_BREAK) -> pd.DataFrame:
    """発火（その日の候補の数 = 行数。live_track.decide と同じ）n_break と、発火 min_break 件以上の印 hot を足す。"""
    n = ft.groupby("Date")["Code"].transform("size").to_numpy()
    return ft.assign(n_break=n, hot=n >= min_break)


def first_hot(ft: pd.DataFrame, flag: str = "R0", key: str = "pmin3", hot: np.ndarray | None = None,
              n: int = TOP_N) -> pd.DataFrame:
    """各月、flag の行のうち印 hot（既定は ft["hot"]）の日の行を先着 n 件（同じ日は key の高い順、同点は lgbm のスコア順）。"""
    h = ft["hot"].to_numpy(dtype=bool) if hot is None else np.asarray(hot, dtype=bool)
    return first_by(ft, ft[flag].to_numpy(dtype=bool) & h, key, n)


def day_index(ft: pd.DataFrame) -> list:
    """窓ごとの (窓, 行番号, 行ごとの日の通し番号, 日ごとの印)。日は、その窓で候補のある日を日付の順に並べたもの。"""
    dates = pd.to_datetime(ft["Date"]).to_numpy()
    w = ft["window"].to_numpy()
    hot = ft["hot"].to_numpy(dtype=bool)
    out = []
    for ww in np.unique(w):
        m = np.where(w == ww)[0]
        days, inv = np.unique(dates[m], return_inverse=True)
        dh = np.zeros(len(days), dtype=bool)
        dh[inv] = hot[m]
        out.append((int(ww), m, inv, dh))
    return out


def rotate_hot(n_rows: int, idx: list, shifts: dict) -> np.ndarray:
    """窓ごとに日ごとの印を shifts[窓] 日だけ巡回してずらし、行ごとの印に戻す（ずらしても印の付く日の数は変わらない）。"""
    out = np.zeros(n_rows, dtype=bool)
    for ww, m, inv, dh in idx:
        out[m] = np.roll(dh, int(shifts.get(ww, 0)))[inv]
    return out


def shift_test(ft: pd.DataFrame, cur: pd.DataFrame, flag: str = "R0", key: str = "pmin3", draws: int = DRAWS,
               seed: int = SEED) -> dict:
    """
    判定3: 実際の絞り込み（主 − 今のやり方 の窓ごとの差の平均）と、印を窓ごとに無作為に巡回してずらしたときの同じ値。
    p = ずらしたほうが実際以上になる割合（片側）。
    """
    obs = window_diff(first_hot(ft, flag, key), cur, MAIN)
    idx = day_index(ft)
    rng = np.random.default_rng(seed)
    null = np.full(draws, np.nan)
    for k in range(draws):
        shifts = {ww: int(rng.integers(len(dh))) for ww, _, _, dh in idx}
        null[k] = window_diff(first_hot(ft, flag, key, hot=rotate_hot(len(ft), idx, shifts)), cur, MAIN)["mean"]
    ok = null[np.isfinite(null)]
    p = float((np.sum(ok >= obs["mean"]) + 1) / (len(ok) + 1)) if obs["n"] else float("nan")
    return {"obs": obs, "p": p, "draws": int(len(ok)),
            "null_mean": float(ok.mean()) if len(ok) else float("nan"),
            "null_p05": float(np.percentile(ok, 5)) if len(ok) else float("nan"),
            "null_p95": float(np.percentile(ok, 95)) if len(ok) else float("nan")}


def verdict(d: dict, p: float) -> tuple[bool, str]:
    need = d["n"] // 2 + 1
    pos = bool(d["n"] and d["mean"] > 0)
    maj = bool(d["n"] and d["wins"] >= need)
    p_ok = bool(np.isfinite(p) and p < 0.05)
    txt = (f"差の平均が正 {'○' if pos else '×'} / 上の窓が過半 {'○' if maj else '×'}（{need}窓以上） / "
           f"印をずらした絞り込みより上 p < 0.05 {'○' if p_ok else '×'}")
    return pos and maj and p_ok, txt


def break_check(ft: pd.DataFrame, counts: pd.Series) -> dict:
    """発火（out-of-fold の行数）と、データセットの日ごとの行数 counts（index = 日付）の突き合わせ。"""
    per_day = ft.groupby(pd.to_datetime(ft["Date"]))["n_break"].first()
    ref = counts.reindex(per_day.index)
    same = int((ref.to_numpy() == per_day.to_numpy()).sum())
    flip = int(((ref.to_numpy() >= MIN_BREAK) != (per_day.to_numpy() >= MIN_BREAK)).sum())
    return {"days": int(len(per_day)), "same": same, "flip": flip,
            "missing": int(ref.isna().sum())}


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def describe(name: str, g: pd.DataFrame, months: int, width: int = 44) -> dict:
    """実験85 の describe に、月あたりの損益の合計と、発火8件未満の日の取引を足す。"""
    out = E85.describe(name, g, months, width=width)
    r = pd.to_numeric(g[MAIN], errors="coerce").dropna()
    quiet = g[~g["hot"].to_numpy(dtype=bool)] if len(g) else g
    rq = pd.to_numeric(quiet[MAIN], errors="coerce").dropna()
    out["sum_pm"] = float(r.sum() / months) if months else float("nan")
    out["quiet_n"] = int(len(quiet))
    out["quiet_mean"] = float(rq.mean()) if len(rq) else float("nan")
    print(f"      月あたりの損益の合計 {out['sum_pm']:+.2f}%（1件の金額に対する%）・発火{MIN_BREAK}件未満の日の取引 "
          f"{len(quiet)}件" + (f"（1取引 {rq.mean():+.2f}%）" if len(rq) else ""))
    return out


def sim_line(name: str, st: dict, months: int, width: int = 44) -> dict:
    """画面の売買の形（3枠）の1取引の成績。"""
    r = np.asarray(st.get("rets", []), dtype=float)
    q = qtiles(r)
    out = {"picks": int(st.get("picks", 0)), "n": int(st.get("n", 0)), "q": q,
           "win": float(st.get("win", float("nan"))), "hit": float(st.get("hit", float("nan"))),
           "bad10": float(st.get("bad10", float("nan"))), "sum_pm": float(r.sum() / months) if months and len(r) else float("nan"),
           "by_window": st.get("by_window", {})}
    if not out["n"]:
        print(f"  {LT._l(name, width)}取引なし")
        return out
    print(f"  {LT._l(name, width)}{out['n']:>4}件（候補 {out['picks']}件）  +10% {q['mean']:+6.2f}%（中央 {q['p50']:+5.2f}）  勝率 "
          f"{out['win']:4.1f}%  到達 {out['hit']:4.1f}%  −10%割れ {out['bad10']:4.1f}%")
    print(f"      分位点 {E84.fmt_q(q)}")
    print(f"      月あたりの損益の合計 {out['sum_pm']:+.2f}%（1件の金額に対する%。3枠なので同時に持つのは3件まで）")
    return out


def by_window_diff(bw: dict, cur: pd.DataFrame) -> dict:
    mc = cur.groupby("window")[MAIN].mean()
    j = sorted(set(int(w) for w in bw) & set(int(w) for w in mc.index))
    return summarize(np.array([bw[w] - mc.loc[w] for w in j], dtype=float))


def report(ft: pd.DataFrame, bar_days=None, ff: pd.DataFrame | None = None, draws: int = DRAWS, seed: int = SEED,
           check: dict | None = None) -> dict:
    summary: dict = {}
    months = E82.n_months(ft)
    days = ft.groupby(pd.to_datetime(ft["Date"]))["hot"].first()
    r0 = ft["R0"].to_numpy(dtype=bool)
    hot = ft["hot"].to_numpy(dtype=bool)
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月（{pd.to_datetime(ft['Date']).min().date()} 〜 "
          f"{pd.to_datetime(ft['Date']).max().date()}）。出口はどれも +10% の指値・届かなければ20日目の終値")
    print(f"  候補のある日 {len(days):,}日のうち発火{MIN_BREAK}件以上 {int(days.sum()):,}日（{days.mean() * 100:.1f}%）。"
          f"R0 の行 {int(r0.sum())}件のうち発火{MIN_BREAK}件以上の日 {int((r0 & hot).sum())}件"
          f"（{(r0 & hot).sum() / max(r0.sum(), 1) * 100:.1f}%）")
    if check:
        print(f"  確かめ: 発火（out-of-fold の行数）がデータセットの行数と一致する日 {check['same']:,}/{check['days']:,}・"
              f"8件の線の上下が入れ替わる日 {check['flip']}・データセットに無い日 {check['missing']}")
        summary["break_check"] = check
    summary["days"] = {"all": int(len(days)), "hot": int(days.sum()), "r0": int(r0.sum()), "r0_hot": int((r0 & hot).sum())}

    # ---- 1. 判定
    print(f"\n■ 1. 月に{TOP_N}件の先着（判定）")
    cur = first_by(ft, r0, "pmin3")
    main = first_hot(ft)
    summary["cur"] = describe("今のやり方: R0 の先着2件", cur, months)
    summary["main"] = describe(f"主: R0 の先着2件・発火{MIN_BREAK}件以上の日だけ", main, months)
    st = shift_test(ft, cur, draws=draws, seed=seed)
    d = st["obs"]
    ok, txt = verdict(d, st["p"])
    print(f"  窓ごとの差（主 − 今のやり方）: {fmt_s(d)}")
    print(f"  印をずらした絞り込み（{st['draws']}回）: 差の平均 {st['null_mean']:+.2f}pt（5〜95% {st['null_p05']:+.2f} 〜 "
          f"{st['null_p95']:+.2f}）。実際以上になる割合 p = {st['p']:.3f}")
    print(f"  判定: {'通る' if ok else '通らない'}（{txt}）")
    summary.update(main_vs_cur=d, shift=st, passed=ok)

    # ---- 2. 仕組み
    print(f"\n■ 2. 参考: R0 の行全体で、発火{MIN_BREAK}件以上の日 − 8件未満の日（窓ごと。並べ替え {draws}回）")
    for name, m in ((f"発火{MIN_BREAK}件以上の日", r0 & hot), (f"発火{MIN_BREAK}件未満の日", r0 & ~hot)):
        x = pd.to_numeric(ft.loc[m, MAIN], errors="coerce").dropna()
        print(f"  {LT._l(name, 20)}{len(x):>4}件（月 {len(x) / months:.1f}件）  +10% {x.mean():+6.2f}%（中央 {x.median():+5.2f}）"
              f"  勝率 {(x > 0).mean() * 100:4.1f}%")
    res = E82.split_test(ft, "hot", "R0", col=MAIN, n_perm=draws, seed=seed)
    rc = E82.split_test(ft, "hot", "R0", col=MAIN, n_perm=draws, seed=seed, unit="code")
    print(f"  窓ごとの差 {fmt_s(res['summary'])}  並べ替え p = {res['p']:.3f}  銘柄単位 {fmt_s(rc['summary'])} p = {rc['p']:.3f}"
          f"  （両側に5件以上ある窓だけ）")
    summary["rows"] = {"row": {k: res[k] for k in ("summary", "p", "n_yes", "n_no")},
                       "code": {k: rc[k] for k in ("summary", "p")}}

    # ---- 3. ほかの選び方に同じ絞り込み
    print(f"\n■ 3. 参考: 同じ絞り込みを、ほかの月{TOP_N}件の選び方に")
    a15 = ft["A15"].to_numpy(dtype=bool)
    t = E85.tiers(ft)
    t_hot = t.assign(tier=np.where(t["hot"].to_numpy(dtype=bool), t["tier"].to_numpy(), 0))
    sel = {"a15": ("A15 の先着2件", first_by(ft, a15, "avg6")),
           "a15_hot": (f"A15 の先着2件・発火{MIN_BREAK}件以上の日だけ", first_hot(ft, "A15", "avg6")),
           "cas": ("段階的・16日から緩める（実験85 の主）", E85.cascade(t)),
           "cas_hot": (f"段階的・発火{MIN_BREAK}件以上の日だけ", E85.cascade(t_hot))}
    summary["other"] = {k: describe(name, g, months) for k, (name, g) in sel.items()}
    for a, b in (("a15_hot", "a15"), ("cas_hot", "cas")):
        dd = window_diff(sel[a][1], sel[b][1], MAIN)
        dc = window_diff(sel[a][1], cur, MAIN)
        summary["other"][f"{a}-{b}"], summary["other"][f"{a}-cur"] = dd, dc
        print(f"  {sel[a][0]}: 絞らない形との差 {fmt_s(dd)}、今のやり方との差 {fmt_s(dc)}")

    # ---- 4. 画面の売買の形
    if bar_days is not None and len(bar_days):
        n_days = LT.n_bar_days(bar_days, pd.to_datetime(ft["Date"]).min(), pd.to_datetime(ft["Date"]).max())
        print(f"\n■ 4. 参考: 画面の売買の形（発火{LT.SKIP_BREAKS}件以上の日に上位{LT.TOP_K}件・{LT.SLOTS}枠・先着順）を R0 で")
        sim = E83.trade_sim(ft, "R0", MAIN, "hit10", "hitday10", bar_days, n_days, keep_trades=True)
        summary["daily"] = sim_line("R0・3枠", sim, months)
        dw = by_window_diff(summary["daily"]["by_window"], cur)
        dm = by_window_diff(summary["daily"]["by_window"], main)
        summary["daily"].update(vs_cur=dw, vs_main=dm)
        print(f"  窓ごとの差: 今のやり方との差 {fmt_s(dw)}、主との差 {fmt_s(dm)}")

    # ---- 5. 発火の線ごと
    print(f"\n■ 5. 参考: 発火の線ごとの R0 の先着{TOP_N}件（見るだけ。線はここから選ばない）")
    grid = []
    nb = ft["n_break"].to_numpy()
    nb_day = ft.groupby(pd.to_datetime(ft["Date"]))["n_break"].first()
    for b in GRID:
        g = first_by(ft, r0 & (nb >= b), "pmin3")
        x = pd.to_numeric(g[MAIN], errors="coerce").dropna()
        per_m = g.groupby("_m").size() if len(g) else pd.Series(dtype=int)
        row = {"min_break": b, "day_share": float((nb_day >= b).mean() * 100), "n": int(len(g)), "zero": int(months - len(per_m)), "mean": float(x.mean()) if len(x) else float("nan"),
               "med": float(x.median()) if len(x) else float("nan"),
               "win": float((x > 0).mean() * 100) if len(x) else float("nan"),
               "sum_pm": float(x.sum() / months) if months else float("nan"),
               "vs_cur": window_diff(g, cur, MAIN)}
        grid.append(row)
        print(f"  発火{b:>2}件以上（候補のある日の {row['day_share']:5.1f}%）: {row['n']:>3}件・0件の月 {row['zero']:>2}  1取引 "
              f"{row['mean']:+6.2f}%（中央 {row['med']:+5.2f}）  勝率 {row['win']:4.1f}%  月あたりの合計 {row['sum_pm']:+5.2f}%  "
              f"今のやり方との差 {fmt_s(row['vs_cur'])}")
    summary["grid"] = grid

    # ---- 6. 画面の物差し
    if ff is not None and len(ff):
        mf = E82.n_months(ff)
        print(f"\n■ 6. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・窓 {int(ff['window'].min())}〜{int(ff['window'].max())}"
              f"・{mf}か月")
        cur_f = first_by(ff, ff["R0"].to_numpy(dtype=bool), "pmin3")
        main_f = first_hot(ff)
        summary["full"] = {"cur": describe("今のやり方: R0 の先着2件", cur_f, mf),
                           "main": describe(f"R0 の先着2件・発火{MIN_BREAK}件以上の日だけ", main_f, mf)}
        summary["full"]["main_vs_cur"] = window_diff(main_f, cur_f, MAIN)
        print(f"  窓ごとの差（発火{MIN_BREAK}件以上の日だけ − 今のやり方）: {fmt_s(summary['full']['main_vs_cur'])}")
        if bar_days is not None and len(bar_days):
            nd = LT.n_bar_days(bar_days, pd.to_datetime(ff["Date"]).min(), pd.to_datetime(ff["Date"]).max())
            simf = E83.trade_sim(ff, "R0", MAIN, "hit10", "hitday10", bar_days, nd, keep_trades=True)
            summary["full"]["daily"] = sim_line("画面の売買の形（R0・3枠）", simf, mf)

    print("\n■ まとめ（+10% の指値・20日目の終値。前向きの百分位）")
    c, m = summary["cur"], summary["main"]
    print(f"  今のやり方: {c['n']}件（0件の月 {c['zero']}）・1取引 {c['q']['mean']:+.2f}%（中央 {c['q']['p50']:+.2f}）・"
          f"うち発火{MIN_BREAK}件未満の日 {c['quiet_n']}件（{c['quiet_mean']:+.2f}%）")
    print(f"  主（発火{MIN_BREAK}件以上の日だけ）: {m['n']}件（0件の月 {m['zero']}）・1取引 {m['q']['mean']:+.2f}%（中央 "
          f"{m['q']['p50']:+.2f}）。差 {fmt_s(d)}・印をずらした絞り込みより上 p = {st['p']:.3f} → {'通る' if ok else '通らない'}")
    if "daily" in summary:
        dl = summary["daily"]
        print(f"  画面の売買の形（R0・3枠）: {dl['n']}件・1取引 {dl['q']['mean']:+.2f}%（中央 {dl['q']['p50']:+.2f}）")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験86: 月2件の先着買いを発火8件以上の日だけに絞る")
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
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = E83.attach_exits(df, bars, targets=(10.0, 20.0))
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・5日平均と ret_o1_20 の一致 {chk['hold5_match']:,}/"
        f"{chk['hold5_both']:,}・到達の順序の食い違い {chk['order_violations']}件")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = add_breaks(E85.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]))
    ff = add_breaks(E85.build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))]))
    counts = pd.to_datetime(fr["Date"]).value_counts()
    check = break_check(ft, counts)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    summary = report(ft, bar_days=bar_days, ff=ff, draws=args.draws, check=check)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e86_summary.json")

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
