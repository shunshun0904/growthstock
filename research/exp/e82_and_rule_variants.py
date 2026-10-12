#!/usr/bin/env python3
"""
実験82: 「1.2σ（本番）の3本がすべて 95以上」の中で、1.5σ・1.8σ の3本もすべて 95以上の銘柄は、やはり儲かるか。

運用者の指示（2026-10-12）「1.2だけでなく、1.5や1.8の方でも、&条件で95以上ならそっちの銘柄の方がやはり利益は
高いですか？検証してほしいです」。

規則（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  R0  = 1.2σ（本番）の lgbm・xgb・cat がすべて 95以上（運用者の買いの目安）
  A15 = R0 かつ 1.5σ の lgbm_k15・xgb_k15・cat_k15 もすべて 95以上
  A18 = R0 かつ 1.8σ の lgbm_k18・xgb_k18・cat_k18 もすべて 95以上
  主の比較: R0 の中で「A15 の行 − A15 でない行」と「A18 の行 − A18 でない行」
  百分位: 前向き（実験79 の window_split。テスト窓より前の窓のスコア分布での位置。テスト窓の手前の20営業日ぶんは
          外す）。丸めは画面と同じ（小数1桁、Python の round。live_track.pct_of と同じ）。テスト窓は 3〜11 の9つ
  物差し: 実収益 ret_o1_20（判定に使う）。並べて運用の出口（+20% の指値、届かなければ 20営業日。live_track.forward）の
          収益・勝率・−10%割れ・+20% 到達と、月あたりの件数・銘柄数
  判定: 窓ごとの差の平均が正・並べ替え検定の p < 0.05（窓ごとに R0 の中で印を並べ替える。片側）・上の窓が過半
        （実験80 と同じ3つ）。同じ銘柄が続けて候補に出るので、銘柄単位（同じ窓・同じ銘柄・同じ側の行を1つにまとめる）
        の p も出し、0.05 未満でなければ「頑健でない」と書く
  参考（判定に使わない）
    - 9本（R0 かつ 1.5σ・1.8σ の両方）
    - ロジスティック回帰も入れた4本ずつ（1.2σ の4本 / それに 1.5σ の4本 / 1.8σ の4本）
    - 1.5σ / 1.8σ の3本だけで 95以上（1.2σ を問わない。規則の置き換え）。R0 との窓ごとの差
    - 全期間の百分位（画面の pctHistorical と同じ物差し。先の分布も含むので甘い）。窓 1〜11
    - A15 / A18 と同じ件数を、R0 の中で12本の合意の高い順に選んだとき（画面に出した合意で足りるか）
    - 運用の通りの売買: 発火8件以上の日に、規則を満たす候補を 1.2σ の3本の最小の百分位の高い順に上位2件
      （同点は lgbm のスコア順）、3枠・先着順、+20% の指値・20営業日（live_track.simulate / trade_stats）
    - 月に2銘柄（各月、規則を満たす行を同じ並べ方で上位2件）

注意: 規則の候補は運用者の発案で、確かめられるのは同じ期間（2021〜2026年の out-of-fold）だけ。前向きにするのは
百分位に先の情報を使わないため。独立の確かめは、これからの予測（ライブ）で行う。

公開ログには集計だけ（群ごとの件数・平均・割合、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e82_and_rule_variants.py
    args="--perm 2000"        （既定）

結果（2026-10-12、run 38187609971、06f6e46、40秒。docs/MODEL_ADOPTION_RULES.md §39）
------------------------------------------------------------------
  データ: 12本 16,320件。運用の出口は全行に付き、日足から作った20営業日の5日平均と ret_o1_20 が全行で一致（差 0.000pt）。
          前向き 14,382行（テスト窓 3〜11、2022-05〜2026-09、53か月）。丸めで 95 の線の上に入った行 4件
  R0 543件（月 10.2件）: 実収益 +4.08%・+20%指値 +3.62%
  A15 333件（R0 の 61%）+4.64% / +4.40%、残り 210件 +3.20% / +2.37%
      → 窓ごとの差 +0.60 ± 2.28pt（6窓中4）・p = 0.36、+20%指値 +1.02 ± 1.61pt・p = 0.25 → 通らない
  A18 279件（51%）+4.49% / +4.37%、残り 264件 +3.66% / +2.83%
      → −3.07 ± 1.93pt（8窓中4）・p = 0.93、+20%指値 −1.04 ± 1.44pt・p = 0.74 → 通らない
  銘柄単位の p もほぼ同じ（A15 0.36、A18 0.92）。9本・4本版・置き換え（1.5σ / 1.8σ の3本だけ）も通らない
  運用の通りの売買（3枠）: R0 82取引 1取引 +2.68%・月 +1.44%、A15 74取引 +3.12%・月 +1.52%（取引の窓ごとの差
      −0.80 ± 1.43pt）、A18 65取引 +2.92%・月 +1.25%（−1.81 ± 1.36pt）
  全期間の百分位（参考。画面の物差し）: A15 − 残り +1.95 ± 2.06pt・p = 0.13、+20%指値 +2.47 ± 1.67pt・p = 0.052
  → 全部を足した平均では高く見えるが、窓ごとに比べると向きが窓によって逆になり、先に決めた判定は通らない
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
import models as M  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e80_k15_rule_forward as E80  # noqa: E402

OUTCOME = E79.OUTCOME
MODELS = E79.MODELS
LINE = 95.0
TREES = ("lgbm", "xgb", "cat")
T15 = tuple(a for a in E79.K15 if M.base_algo(a) in TREES)
T18 = tuple(a for a in E79.K18 if M.base_algo(a) in TREES)
BASE4, K15, K18 = E79.BASE4, E79.K15, E79.K18

#: (名前, 表示, すべて LINE 以上が要るモデル)
RULES = (
    ("R0", "1.2σ の3本", TREES),
    ("A15", "R0 + 1.5σ の3本", TREES + T15),
    ("A18", "R0 + 1.8σ の3本", TREES + T18),
    ("A9", "R0 + 1.5σ・1.8σ の6本", TREES + T15 + T18),
    ("R0_4", "1.2σ の4本（logit も）", BASE4),
    ("A15_4", "4本 + 1.5σ の4本", BASE4 + K15),
    ("A18_4", "4本 + 1.8σ の4本", BASE4 + K18),
    ("K15", "1.5σ の3本だけ", T15),
    ("K18", "1.8σ の3本だけ", T18),
)
LABEL = {k: v for k, v, _ in RULES}
MAIN = (("A15", "R0"), ("A18", "R0"))                          # 判定する比較（印, 母集団）
REF = (("A9", "R0"), ("A15_4", "R0_4"), ("A18_4", "R0_4"))      # 参考（同じ形の比較）
SWAP = ("K15", "K18")                                           # 置き換え（R0 と比べる。重なるので検定しない）
TRADE_RULES = ("R0", "A15", "A18", "A9")
N_PERM = 2000
MIN_GROUP = 5                       # 窓ごとの比較で、群に要る件数（実験80 と同じ）
MONTHLY_N = 2
SEED = 42
OUT_DIR = E79.OUT_DIR

log = E79.log
nmean = E79.nmean
summarize = E79.summarize
fmt_s = E79.fmt_s


# --------------------------------------------------------------------------- #
# 百分位・印
# --------------------------------------------------------------------------- #

def rounded(P: np.ndarray) -> np.ndarray:
    """画面の pctHistorical と同じ丸め（小数1桁、Python の round。live_track.pct_of と同じ）。"""
    P = np.asarray(P, dtype=float)
    flat = [round(float(v), 1) if np.isfinite(v) else np.nan for v in P.ravel()]
    return np.array(flat, dtype=float).reshape(P.shape)


def flags_of(R: np.ndarray, models=MODELS, rules=RULES, line: float = LINE) -> dict:
    """規則ごとの印（丸めた百分位がすべて line 以上）。NaN は満たさない扱い。"""
    idx = {a: i for i, a in enumerate(models)}
    return {key: (R[:, [idx[a] for a in grp]] >= line).all(axis=1) for key, _, grp in rules}


def build_frame(df: pd.DataFrame, parts: list, models=MODELS) -> pd.DataFrame:
    """
    parts: [(行番号, 丸める前の百分位の行列)]。1行ごとに 規則の印・1.2σ の3本の最小（丸め後）・合意（12本の平均）・
    lgbm のスコア・実収益（%）・運用の出口の収益（%）を持つ表。
    """
    i = {a: k for k, a in enumerate(models)}
    outs = []
    for rows, P in parts:
        R = rounded(P)
        src = df.iloc[rows]
        g = src[["Code", "Date", "window"]].copy().reset_index(drop=True)
        g["ret20"] = pd.to_numeric(src[OUTCOME], errors="coerce").to_numpy(dtype=float) * 100
        for c in ("ret_tp", "status", "hit_day"):
            g[c] = src[c].to_numpy() if c in src.columns else np.nan
        g["score"] = src[f"score_{M.BASELINE}"].to_numpy(dtype=float)
        g["pmin3"] = R[:, [i[a] for a in TREES]].min(axis=1)
        g["pmin3_raw"] = np.asarray(P, dtype=float)[:, [i[a] for a in TREES]].min(axis=1)
        g["cons"] = np.asarray(P, dtype=float).mean(axis=1)
        for k, v in flags_of(R, models).items():
            g[k] = v
        outs.append(g)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 成績と比べ方
# --------------------------------------------------------------------------- #

def n_months(ft: pd.DataFrame) -> int:
    return int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique()) if len(ft) else 0


def group_stats(g: pd.DataFrame, months: float) -> dict:
    """群の成績（% の数）。hit20 は運用の出口が付いた行のうち +20% に届いた割合。"""
    r = pd.to_numeric(g["ret20"], errors="coerce").dropna()
    t = pd.to_numeric(g["ret_tp"], errors="coerce")
    ok = t.notna()
    t = t[ok]
    codes = g.assign(_m=pd.to_datetime(g["Date"]).dt.to_period("M")).drop_duplicates(["Code", "_m"])
    nan = float("nan")
    return {
        "n": int(len(g)),
        "per_month": len(g) / months if months else nan,
        "codes_per_month": len(codes) / months if months else nan,
        "ret": float(r.mean()) if len(r) else nan,
        "med": float(r.median()) if len(r) else nan,
        "win": float((r > 0).mean() * 100) if len(r) else nan,
        "bad10": float((r < -10).mean() * 100) if len(r) else nan,
        "tp": float(t.mean()) if len(t) else nan,
        "tp_win": float((t > 0).mean() * 100) if len(t) else nan,
        "tp_bad10": float((t < -10).mean() * 100) if len(t) else nan,
        "hit20": float((g.loc[ok, "status"] == "+20%到達").mean() * 100) if len(t) else nan,
    }


def split_test(ft: pd.DataFrame, flag: str, within: str, col: str = "ret20", n_perm: int = N_PERM,
               seed: int = SEED, min_group: int = MIN_GROUP, unit: str = "row") -> dict:
    """
    within の行の中で「flag の行 − それ以外の行」の平均（col）を窓ごとに。並べ替え検定は、窓ごとに within の中で印を
    並べ替える（件数はそのまま。片側: 並べ替えた平均が実際の平均以上だった割合）。
    unit="code" は、同じ窓・同じ銘柄・同じ側の行を1つ（平均）にまとめてから同じことをする。
    """
    sub = ft[ft[within].to_numpy(dtype=bool)]
    sub = sub[np.isfinite(pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float))]
    if unit == "code":
        sub = sub.groupby(["window", "Code", flag], as_index=False)[col].mean()
    f = sub[flag].to_numpy(dtype=bool)
    r = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    w = sub["window"].to_numpy()
    groups = []
    for ww in np.unique(w):
        m = np.where(w == ww)[0]
        if f[m].sum() >= min_group and (~f[m]).sum() >= min_group:
            groups.append(m)

    def stat(ff):
        return np.array([r[g][ff[g]].mean() - r[g][~ff[g]].mean() for g in groups], dtype=float)

    d = stat(f)
    s = summarize(d)
    p = float("nan")
    if groups and n_perm:
        rng = np.random.default_rng(seed)
        perm = np.empty(n_perm)
        for k in range(n_perm):
            fp = f.copy()
            for g in groups:
                fp[g] = f[g][rng.permutation(len(g))]
            perm[k] = stat(fp).mean()
        p = float((np.sum(perm >= s["mean"]) + 1) / (n_perm + 1))
    return {"flag": flag, "within": within, "col": col, "unit": unit, "summary": s, "p": p,
            "diffs": d.tolist(), "windows": [int(w[g[0]]) for g in groups],
            "n_yes": int(f.sum()), "n_no": int((~f).sum()), "share": float(f.mean()) if len(f) else float("nan")}


judge = E80.judge


def vs_all(ft: pd.DataFrame, flag: str, base: str, col: str = "ret20", min_group: int = MIN_GROUP) -> dict:
    """窓ごとに「flag の行の平均 − base の行の平均」（重なってよい。置き換えの比較と、R0 全体との差に使う）。"""
    d = []
    for _, g in ft.groupby("window"):
        r = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        a, b = r[g[flag].to_numpy(dtype=bool)], r[g[base].to_numpy(dtype=bool)]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) >= min_group and len(b) >= min_group:
            d.append(a.mean() - b.mean())
    return summarize(np.array(d, dtype=float))


def vs_consensus(ft: pd.DataFrame, flag: str, within: str, col: str = "ret20", min_group: int = MIN_GROUP) -> dict:
    """窓ごとに、within の中で flag と同じ件数を合意（12本の平均）の高い順に選んだときとの差（flag − 合意）と重なり。"""
    d, over = [], []
    for _, g in ft[ft[within].to_numpy(dtype=bool)].groupby("window"):
        f = g[flag].to_numpy(dtype=bool)
        n = int(f.sum())
        if n < min_group or n == len(g):
            continue
        r = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        top = np.argsort(-g["cons"].to_numpy(dtype=float), kind="mergesort")[:n]
        d.append(nmean(r[f]) - nmean(r[top]))
        over.append(float(f[top].mean()))
    return {"summary": summarize(np.array(d, dtype=float)), "overlap": float(np.mean(over)) if over else float("nan")}


# --------------------------------------------------------------------------- #
# 運用の通りの売買・月に2銘柄
# --------------------------------------------------------------------------- #

def daily_picks(ft: pd.DataFrame, flag: str, min_break: int = LT.SKIP_BREAKS, top_k: int = LT.TOP_K) -> pd.DataFrame:
    """
    画面の判断（live_track.decide）と同じ並べ方: 発火（その日の候補の数）が min_break 以上の日に、flag を満たす候補を
    1.2σ の3本の最小の百分位の高い順（同点は lgbm のスコア順）に上位 top_k 件。
    """
    r = ft.copy()
    r["n_break"] = r.groupby("Date")["Code"].transform("size")
    ok = r[flag].to_numpy(dtype=bool) & (r["n_break"] >= min_break).to_numpy()
    p = r[ok].sort_values(["Date", "pmin3", "score"], ascending=[True, False, False], kind="mergesort")
    p["rank"] = p.groupby("Date").cumcount() + 1
    return p[p["rank"] <= top_k].reset_index(drop=True)


def trade_sim(ft: pd.DataFrame, flag: str, bar_days, n_days: int) -> dict:
    """運用の通り（上位2件・3枠・先着順・+20% の指値・20営業日）に売買したときの成績と、取引の窓ごとの平均。"""
    picks = daily_picks(ft, flag)
    if not len(picks):
        return {"picks": 0, "n": 0, "by_window": {}}
    sim = LT.simulate(picks.rename(columns={"ret_tp": "ret"}), bar_days, slots=LT.SLOTS, hold=LT.HOLD)
    taken = sim[sim["taken"] == LT.TAKEN]
    st = LT.trade_stats(taken, n_days, slots=LT.SLOTS)
    st["picks"] = int(len(picks))
    st["full"] = int((sim["taken"] == LT.FULL).sum())
    st["ret20"] = float(pd.to_numeric(taken["ret20"], errors="coerce").mean()) if len(taken) else float("nan")
    st["bad10"] = float((pd.to_numeric(taken["ret"], errors="coerce").dropna() < -10).mean() * 100) \
        if len(taken) else float("nan")
    st["by_window"] = {int(w): float(pd.to_numeric(g["ret"], errors="coerce").mean())
                       for w, g in taken.groupby("window")}
    return st


def monthly_top(ft: pd.DataFrame, flag: str, n: int = MONTHLY_N) -> pd.DataFrame:
    """各月、flag を満たす行を 1.2σ の3本の最小の百分位の高い順（同点は lgbm のスコア順）に上位 n 件。"""
    sub = ft[ft[flag].to_numpy(dtype=bool)].assign(_m=lambda x: pd.to_datetime(x["Date"]).dt.to_period("M"))
    sub = sub.sort_values(["_m", "pmin3", "score"], ascending=[True, False, False], kind="mergesort")
    return sub.groupby("_m", sort=False).head(n).reset_index(drop=True)


def window_diff_means(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    """窓ごとの平均の差（a − b）。両方にある窓だけ。"""
    ma = a.groupby("window")[col].mean()
    mb = b.groupby("window")[col].mean()
    j = ma.index.intersection(mb.index)
    return summarize((ma[j] - mb[j]).to_numpy(dtype=float))


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict) -> str:
    return (f"  {LT._l(name, 34)}{s['n']:>6,}件（月 {s['per_month']:5.1f}件・{s['codes_per_month']:4.1f}銘柄）  "
            f"実収益 {s['ret']:+6.2f}%（中央 {s['med']:+5.2f}）勝率 {s['win']:4.1f}%  −10%割れ {s['bad10']:4.1f}%  ｜ "
            f"+20%指値 {s['tp']:+6.2f}%  勝率 {s['tp_win']:4.1f}%  +20%到達 {s['hit20']:4.1f}%  −10%割れ {s['tp_bad10']:4.1f}%")


def verdict_text(jd: dict) -> str:
    return (f"{'通る' if jd['pass'] else '通らない'}（平均が正 {'○' if jd['positive'] else '×'} / "
            f"p < 0.05 {'○' if jd['p_ok'] else '×'} / 過半 {'○' if jd['majority'] else '×'}（{jd['need']}窓以上））")


def compare_block(ft: pd.DataFrame, flag: str, within: str, n_perm: int, seed: int, detail: bool) -> dict:
    out = {}
    for col, nm in (("ret20", "実収益"), ("ret_tp", "+20%指値")):
        res = split_test(ft, flag, within, col=col, n_perm=n_perm, seed=seed)
        rc = split_test(ft, flag, within, col=col, n_perm=n_perm, seed=seed, unit="code")
        jd = judge(res)
        out[col] = {"row": res, "code": {k: rc[k] for k in ("summary", "p", "n_yes", "n_no")}, "judge": jd,
                    "vs_within": vs_all(ft, flag, within, col=col)}
        print(f"    {nm:<8} 窓ごとの差 {fmt_s(res['summary'])}  並べ替え p = {res['p']:.3f}  "
              f"銘柄単位 {fmt_s(rc['summary'])} p = {rc['p']:.3f}")
        print(f"             {LABEL[within]} 全体との差 {fmt_s(out[col]['vs_within'])}  判定（行単位）: {verdict_text(jd)}")
        if detail:
            print("             窓ごと: " + "  ".join(f"窓{w} {d:+.2f}" for w, d in zip(res["windows"], res["diffs"])))
    return out


def report(ft: pd.DataFrame, bar_days=None, ff: pd.DataFrame | None = None, n_perm: int = N_PERM,
           seed: int = SEED) -> dict:
    summary: dict = {}
    months = n_months(ft)
    wins = sorted(int(w) for w in ft["window"].unique())
    print(f"\n■ 0. 前向きのテスト窓 {len(wins)}つ（{wins[0]}〜{wins[-1]}）・{len(ft):,}行・{months}か月"
          f"（{pd.to_datetime(ft['Date']).min().date()} 〜 {pd.to_datetime(ft['Date']).max().date()}）")
    edge = int(((ft["pmin3_raw"] < LINE) & (ft["pmin3"] >= LINE)).sum())
    print(f"  画面と同じ丸め（小数1桁）で 95 の線の上に入った行（丸める前は 95 未満）: {edge}件")
    for key, _, _ in RULES:
        for base in ("R0", "R0_4"):
            if key.endswith("_4") == (base == "R0_4") and key not in SWAP:
                assert not (ft[key] & ~ft[base]).any(), f"{key} が {base} の外にある"
    summary["windows"] = wins
    summary["months"] = months

    print("\n■ 1. 群ごとの成績（前向きの百分位。実収益 = ret_o1_20、+20%指値 = 運用の出口）")
    summary["groups"] = {"all": group_stats(ft, months)}
    print(line("テストの全行", summary["groups"]["all"]))
    for key, label, _ in RULES:
        m = ft[key].to_numpy(dtype=bool)
        summary["groups"][key] = group_stats(ft[m], months)
        print(line(f"{key} {label}", summary["groups"][key]))
        if key in ("A15", "A18", "A9"):
            summary["groups"][f"R0_not_{key}"] = group_stats(ft[ft["R0"].to_numpy(dtype=bool) & ~m], months)
            print(line(f"  R0 で {key} でない", summary["groups"][f"R0_not_{key}"]))

    print(f"\n■ 2. 判定（先に決めた比較。R0 の中で 印の行 − それ以外。並べ替え {n_perm}回）")
    print("  判定: 窓ごとの差の平均が正・p < 0.05・上の窓が過半。銘柄単位 = 同じ窓・同じ銘柄・同じ側の行を1つにまとめた形")
    summary["main"] = {}
    for flag, within in MAIN:
        r0 = ft[ft[within].to_numpy(dtype=bool)]
        print(f"\n  {flag}（{LABEL[flag]}）: R0 の {r0[flag].mean() * 100:.0f}%")
        summary["main"][flag] = compare_block(ft, flag, within, n_perm, seed, detail=True)

    print("\n■ 3. 参考（判定に使わない）")
    summary["ref"] = {}
    for flag, within in REF:
        sub = ft[ft[within].to_numpy(dtype=bool)]
        print(f"\n  {flag}（{LABEL[flag]}）の中 対 {within}（{LABEL[within]}）の残り: {within} の "
              f"{sub[flag].mean() * 100 if len(sub) else float('nan'):.0f}%")
        summary["ref"][flag] = compare_block(ft, flag, within, n_perm, seed, detail=False)
    print("\n  置き換え（1.2σ を問わない）: R0 との窓ごとの差（重なるので検定しない）")
    summary["swap"] = {}
    for key in SWAP:
        k = ft[key].to_numpy(dtype=bool)
        over = float((k & ft["R0"].to_numpy(dtype=bool)).sum() / k.sum()) if k.sum() else float("nan")
        v20, vtp = vs_all(ft, key, "R0", "ret20"), vs_all(ft, key, "R0", "ret_tp")
        summary["swap"][key] = {"ret20": v20, "ret_tp": vtp, "overlap": over}
        print(f"    {key}（{LABEL[key]}）: 実収益 {fmt_s(v20)}  +20%指値 {fmt_s(vtp)}  R0 にも入る割合 {over * 100:.0f}%")
    print("\n  同じ件数を R0 の中で12本の合意の高い順に選んだときとの差（印 − 合意。重なり = 合意で選んだ行のうち印も付く割合）")
    summary["vs_cons"] = {}
    for flag, within in MAIN:
        v20, vtp = vs_consensus(ft, flag, within, "ret20"), vs_consensus(ft, flag, within, "ret_tp")
        summary["vs_cons"][flag] = {"ret20": v20, "ret_tp": vtp}
        print(f"    {flag}: 実収益 {fmt_s(v20['summary'])}  +20%指値 {fmt_s(vtp['summary'])}  重なり {v20['overlap'] * 100:.0f}%")

    if bar_days is not None and len(bar_days):
        n_days = LT.n_bar_days(bar_days, pd.to_datetime(ft["Date"]).min(), pd.to_datetime(ft["Date"]).max())
        print(f"\n■ 4. 運用の通りの売買（発火{LT.SKIP_BREAKS}件以上の日に上位{LT.TOP_K}件・{LT.SLOTS}枠・先着順・"
              f"+{LT.TAKE_PROFIT:g}% の指値・{LT.HOLD}営業日。{n_days}営業日）")
        print("  月の収益 = 取引の収益の合計 ÷ 枠数 ÷ 月数（1取引に資金の1/3。複利にしない。手数料・税は含まない）")
        summary["trade"] = {}
        for key in TRADE_RULES:
            st = trade_sim(ft, key, bar_days, n_days)
            summary["trade"][key] = st
            if not st.get("n"):
                print(f"  {key:<4} 取引なし")
                continue
            print(f"  {LT._l(key + ' ' + LABEL[key], 30)}候補 {st['picks']:>4}件 → 買えた {st['n']:>4}件（枠が満杯 {st['full']}件・"
                  f"月 {st['per_month']:.1f}件）  1取引 {st['mean']:+6.2f}%（中央 {st['median']:+5.2f}）  "
                  f"勝率 {st['win']:4.1f}%  +20%到達 {st['hit']:4.1f}%  −10%割れ {st['bad10']:4.1f}%  "
                  f"月の収益 {st['monthly']:+5.2f}%  （同じ取引の実収益 {st['ret20']:+5.2f}%）")
        base = summary["trade"].get("R0", {}).get("by_window", {})
        for key in TRADE_RULES[1:]:
            bw = summary["trade"].get(key, {}).get("by_window", {})
            j = sorted(set(base) & set(bw))
            s = summarize(np.array([bw[w] - base[w] for w in j], dtype=float))
            summary["trade"][key]["vs_R0"] = s
            print(f"      {key} − R0（取引の窓ごとの平均の差）: {fmt_s(s)}")

    print(f"\n■ 5. 月に{MONTHLY_N}銘柄（各月、規則を満たす行を 1.2σ の3本の最小の百分位の高い順に）")
    summary["monthly"] = {}
    base_m = monthly_top(ft, "R0")
    for key in TRADE_RULES:
        mt = monthly_top(ft, key)
        st = group_stats(mt, months)
        st["months"] = int(mt["Date"].pipe(pd.to_datetime).dt.to_period("M").nunique()) if len(mt) else 0
        if key != "R0":
            st["vs_R0"] = window_diff_means(mt, base_m, "ret20")
            st["vs_R0_tp"] = window_diff_means(mt, base_m, "ret_tp")
        summary["monthly"][key] = st
        extra = (f"  R0 の上位{MONTHLY_N}件との差 実収益 {fmt_s(st['vs_R0'])} / +20%指値 {fmt_s(st['vs_R0_tp'])}"
                 if key != "R0" else "")
        print(line(f"{key}（{st['months']}か月）", st) + extra)

    if ff is not None and len(ff):
        mf = n_months(ff)
        print(f"\n■ 6. 参考: 全期間の百分位（画面の pctHistorical と同じ物差し。先の分布も含む）・窓 "
              f"{int(ff['window'].min())}〜{int(ff['window'].max())}・{len(ff):,}行")
        summary["full"] = {"groups": {}, "main": {}}
        for key in ("R0", "A15", "A18", "A9"):
            summary["full"]["groups"][key] = group_stats(ff[ff[key].to_numpy(dtype=bool)], mf)
            print(line(f"{key} {LABEL[key]}", summary["full"]["groups"][key]))
        for flag, within in MAIN:
            res = split_test(ff, flag, within, n_perm=n_perm, seed=seed)
            rt = split_test(ff, flag, within, col="ret_tp", n_perm=n_perm, seed=seed)
            summary["full"]["main"][flag] = {"ret20": res, "ret_tp": rt}
            print(f"    {flag} − R0 の残り: 実収益 {fmt_s(res['summary'])} p = {res['p']:.3f}  "
                  f"+20%指値 {fmt_s(rt['summary'])} p = {rt['p']:.3f}")

    print("\n■ まとめ")
    for flag, _ in MAIN:
        x = summary["main"][flag]
        g, ng = summary["groups"][flag], summary["groups"][f"R0_not_{flag}"]
        robust = x["ret20"]["code"]["p"] < 0.05
        print(f"  {flag}（{LABEL[flag]}）: 実収益 {g['ret']:+.2f}% 対 残り {ng['ret']:+.2f}%（月 {g['per_month']:.1f}件 対 "
              f"{ng['per_month']:.1f}件）、窓ごとの差 {fmt_s(x['ret20']['row']['summary'])}、p = "
              f"{x['ret20']['row']['p']:.3f} → {'通る' if x['ret20']['judge']['pass'] else '通らない'}"
              + ("" if not x["ret20"]["judge"]["pass"] else ("（銘柄単位でも p < 0.05）" if robust
                                                           else "（銘柄単位では p ≥ 0.05。頑健でない）")))
        print(f"      +20%指値: {g['tp']:+.2f}% 対 {ng['tp']:+.2f}%、窓ごとの差 {fmt_s(x['ret_tp']['row']['summary'])}、"
              f"p = {x['ret_tp']['row']['p']:.3f}")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。独立の確かめはこれからの予測で")
    return summary


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

def attach_exit(df: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """運用の出口（+20% の指値・20営業日）の収益 ret_tp（%）と status / hit_day を足す。ret_o1_20 との一致も数える。"""
    fw = LT.forward(bars, df[["Code", "Date"]], target=LT.TAKE_PROFIT, hold=LT.HOLD)
    fw = fw[["Code", "Date", "ret", "ret_hold", "status", "hit_day"]].rename(columns={"ret": "ret_tp"})
    out = df.merge(fw, on=["Code", "Date"], how="left")
    a = pd.to_numeric(out["ret_hold"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(out[OUTCOME], errors="coerce").to_numpy(dtype=float) * 100
    both = np.isfinite(a) & np.isfinite(b)
    chk = {"rows": int(len(out)), "with_exit": int(np.isfinite(out["ret_tp"].to_numpy(dtype=float)).sum()),
           "both": int(both.sum()), "match": int((np.abs(a[both] - b[both]) < 0.01).sum()),
           "max_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan"),
           "status": out["status"].value_counts(dropna=False).to_dict()}
    return out, chk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験82: 1.2σ の3本が95以上の中で、1.5σ・1.8σ の3本も95以上の行")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)
    t0 = time.time()
    wide, info0 = E79.load_oofs()
    log(f"out-of-fold 12本を突き合わせた: {info0['kept']:,}件（モデルごとの件数 {sorted(set(info0['rows'].values()))}）")
    fr = lab.frame()
    df = E79.attach_outcomes(wide, fr)
    folds = E79.make_windows(fr["Date"])
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(df["Date"]).min() - pd.Timedelta(days=10))
    df, chk = attach_exit(df, bars)
    log(f"運用の出口: {chk['with_exit']:,}/{chk['rows']:,}行に付いた。20営業日の5日平均（ret_hold）と ret_o1_20 が "
        f"0.01pt 以内で一致 {chk['match']:,}/{chk['both']:,}行（最大の差 {chk['max_diff']:.3f}pt）")
    log("  状態: " + "・".join(f"{k} {v:,}" for k, v in chk["status"].items()))
    if chk["both"] == 0 or chk["match"] / chk["both"] < 0.99:
        log("[warn] 日足から作った ret_hold と ret_o1_20 の一致が 99% に届かない。日足とデータセットの食い違いを疑う")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = build_frame(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    ff = build_frame(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    summary = report(ft, bar_days=bar_days, ff=ff, n_perm=args.perm)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e82_summary.json")

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
