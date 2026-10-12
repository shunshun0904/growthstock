#!/usr/bin/env python3
"""
実験88: 発火数（その日の候補の数）とリターンの関係を、今の規則（1.5σ の3本がすべて95以上・+10% の指値）で測り直す。

運用者の依頼（2026-10-12）「「発火7件以下の日を見送る線をどうするか。」上記ですが、リターンと発火数の検証は改めて実施して
ほしいです」。画面の見送りの線（発火7件以下の日は買わない）は、以前の規則（1.2σ の3本 90以上、153列のころの OOF）の
142件 −0.11% で決めたもの。今の規則では見るだけの集計で 22件 +5.37%（画面の物差し）と少なかった（§46）。

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  規則の行: 1.5σ の lgbm_k15・xgb_k15・cat_k15 の百分位（画面と同じ丸め）がすべて95以上。出口は +10% の指値・届かなければ
  20営業日目の終値（tp10c）。発火 = その日の候補の数（行数。画面・live_track と同じ）。百分位は前向き（主。テスト窓 3〜11）と
  画面の物差し（参考）
  判定（見送りの線を今の規則で支えるか）: 規則の行の中で D = 発火7件以下の日の平均 − 8件以上の日の平均（全部を足した平均）。
      並べ替え検定は、テスト窓ごとに「7件以下」の印を並べ替える（件数はそのまま。2,000回）。片側 p = 並べ替えた D が実際以下に
      なる割合。銘柄単位（同じ窓・同じ銘柄・同じ側の行を平均して1つ）でも同じ p。
      「支える」= D < 0・行の p < 0.05・銘柄単位の p < 0.05 の3つ。そうでなければ「支えない」（7件以下の件数も並べる）
  参考（判定に使わない）
    - 帯ごとの成績（1〜7 / 8〜19 / 20件以上、細かく 1〜3 / 4〜7 / 8〜11 / 12〜19 / 20〜29 / 30〜49 / 50件以上）を、規則の行と
      候補全体で。前向きと画面の物差し
    - 傾向: テスト窓ごとに、規則の行の発火数と収益の順位相関（スピアマン）。窓の平均・正の窓の数・並べ替え p（窓ごとに収益を
      並べ替える。両側）
    - 同じ判定を、同じ月の中で並べ替えた形（月ごとの地合いをそろえる）・候補全体・以前の規則（1.2σ の3本 90以上。以前の
      142件 −0.11% の測り直し）でも
    - 20件以上（本命）− 8〜19件 の差と並べ替え p
    - 画面の3枠の形（今の規則・上位2件・同時に3件）で、発火8件以上の日だけ買う形と、発火を問わない形の取引の成績
      （1取引の平均・資金全体の月・窓ごとの差）
  出すもの: 件数・平均・中央・勝率・+10% に届く割合・−10%割れ

公開ログには集計だけ（件数・平均・割合）。銘柄・日付ごとの値は出さない。

    exp=e88_breaks_returns.py
    args="--perm 2000"        （既定）

結果（2026-10-12、run 38217093189、30e5c90、34秒。docs/MODEL_ADOPTION_RULES.md §47）
------------------------------------------------------------------
  判定: 見送りの線を支えない。今の規則（前向き 559件）のうち発火7件以下の日は 36件で +3.71%（中央 +10.00・勝率 77.8%・
        −10%割れ 11.1%）、8件以上の日は 523件で +3.05%（中央 +5.00・勝率 67.5%・−10%割れ 5.0%）。D = +0.65pt（7件以下のほうが
        上）、片側 p = 0.880・銘柄単位 0.876・同じ月の中 0.940。窓ごとの差は9窓中6で正（窓ごとの件数は 1〜8件）
  帯（今の規則・前向き）: 1〜7件 36件 +3.71% / 8〜19件 209件 +2.07% / 20件以上 314件 +3.71%。細かく 1〜3件は2件で −11.34%
        （件数が少なすぎる）・4〜7件 34件 +4.59%
  以前の規則（1.2σ の3本 90以上）: 1〜7件 79件 +0.87% / 8〜19件 429件 +1.79% / 20件以上 658件 +2.88%。D = −1.57pt だが、窓ごとに
        並べ替えると片側 p = 0.494（7件以下の日は収益の低い窓に多いだけで、同じ窓の中では差が無い）
  順位相関（窓ごと）: 今の規則 −0.09（正の窓 3/8、p = 0.110）・以前の規則 −0.00・候補全体 −0.05（正の窓 1/9、p = 0.002）
  20件以上 − 8〜19件（今の規則）: +1.64pt（両側 p = 0.163）
  画面の3枠の形（今の規則・上位2件・同時に3件・+10%）: 発火8件以上の日だけ 118件 +1.82%（資金全体の月 +1.35%）、発火を問わない
        128件 +2.19%（月 +1.76%。7件以下の日の取引 12件）。窓ごとの差（8件以上だけ − 問わない）−0.92 ± 0.69pt（9窓中2で上）
  画面の物差し: 1〜7件 22件 +5.37% / 8〜19件 140件 +1.51% / 20件以上 192件 +4.89%。D = +1.91pt・片側 p = 0.973
  注意: 両側 p は「並べ替えた D の絶対値が実際以上」の割合。窓で並べ替えると D の分布の中心が 0 からずれるので、両側 p は参考
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
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402

MODELS = E79.MODELS
#: (記号, 表示, 最小の列, 線)。None は候補全体
RULES = (("K15", "今の規則: 1.5σ の3本 95以上", "min_K15", 95.0),
         ("T3_90", "以前の規則: 1.2σ の3本 90以上", "min_T3", 90.0),
         ("ALL", "候補全体", None, None))
SKIP = LT.PREV_SKIP_BREAKS                  # 8: これ未満は見送り（実験のときの画面の線。2026-10-12 に外した）
STRONG = LT.STRONG_BREAKS                   # 20: 本命
BANDS = ((1, SKIP - 1, f"1〜{SKIP - 1}件"), (SKIP, STRONG - 1, f"{SKIP}〜{STRONG - 1}件"), (STRONG, None, f"{STRONG}件以上"))
FINE = ((1, 3), (4, 7), (8, 11), (12, 19), (20, 29), (30, 49), (50, None))
N_PERM = 2000
SEED = 42
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


def rule_mask(ft: pd.DataFrame, col: str | None, line: float | None) -> np.ndarray:
    if col is None:
        return np.ones(len(ft), dtype=bool)
    return ft[col].to_numpy(dtype=float) >= line


def stats(g: pd.DataFrame) -> dict:
    r = pd.to_numeric(g["tp10c"], errors="coerce").dropna()
    n = int(len(r))
    nan = float("nan")
    hit = g.loc[r.index, "hit10"].astype(bool) if n else pd.Series(dtype=bool)
    return {"n": n, "mean": float(r.mean()) if n else nan, "med": float(r.median()) if n else nan,
            "win": float((r > 0).mean() * 100) if n else nan, "hit": float(hit.mean() * 100) if n else nan,
            "bad10": float((r < -10).mean() * 100) if n else nan,
            "se": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else nan}


def band_mask(nb: np.ndarray, lo: int, hi: int | None) -> np.ndarray:
    return (nb >= lo) & ((nb <= hi) if hi is not None else True)


def strat_perm(v: np.ndarray, f: np.ndarray, strata: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """
    D = 平均(v[f]) − 平均(v[~f])（全部を足した平均）。strata ごとに f を並べ替えた D の分布と比べる。
    p_less = 並べ替えた D が実際以下になる割合（片側。「f の側が悪い」）、p_two = 絶対値で実際以上（両側）。
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=bool)
    s = np.asarray(strata)
    ok = np.isfinite(v)
    v, f, s = v[ok], f[ok], s[ok]
    nan = float("nan")
    if f.sum() == 0 or (~f).sum() == 0:
        return {"D": nan, "p_less": nan, "p_two": nan, "n_yes": int(f.sum()), "n_no": int((~f).sum())}
    obs = float(v[f].mean() - v[~f].mean())
    groups = [np.where(s == u)[0] for u in np.unique(s)]
    groups = [g for g in groups if 0 < f[g].sum() < len(g)]          # 両側がある層だけ並べ替える（他は動かない）
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        fp = f.copy()
        for g in groups:
            fp[g] = f[g][rng.permutation(len(g))]
        null[k] = v[fp].mean() - v[~fp].mean()
    return {"D": obs, "p_less": float((np.sum(null <= obs) + 1) / (n_perm + 1)),
            "p_two": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)),
            "n_yes": int(f.sum()), "n_no": int((~f).sum()), "strata_used": len(groups)}


def code_level(g: pd.DataFrame, flag: np.ndarray, strata_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同じ層・同じ銘柄・同じ側の行を平均して1つにした (値, 印, 層)。"""
    x = pd.DataFrame({"v": pd.to_numeric(g["tp10c"], errors="coerce").to_numpy(dtype=float), "f": flag,
                      "s": g[strata_col].to_numpy(), "c": g["Code"].to_numpy()}).dropna(subset=["v"])
    a = x.groupby(["s", "c", "f"], as_index=False)["v"].mean()
    return a["v"].to_numpy(), a["f"].to_numpy(dtype=bool), a["s"].to_numpy()


def skip_test(g: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """g（規則の行）で「発火 SKIP 未満」の日 − それ以外。窓で並べ替え（主）・銘柄単位・月で並べ替え（参考）。"""
    nb = g["n_break"].to_numpy()
    f = nb < SKIP
    v = pd.to_numeric(g["tp10c"], errors="coerce").to_numpy(dtype=float)
    row = strat_perm(v, f, g["window"].to_numpy(), n_perm, seed)
    cv, cf, cs = code_level(g, f, "window")
    code = strat_perm(cv, cf, cs, n_perm, seed)
    month = strat_perm(v, f, pd.to_datetime(g["Date"]).dt.to_period("M").astype(str).to_numpy(), n_perm, seed)
    per_w = {}
    for w, x in g.groupby("window"):
        xf = x["n_break"].to_numpy() < SKIP
        r = pd.to_numeric(x["tp10c"], errors="coerce").to_numpy(dtype=float)
        if xf.any() and (~xf).any():
            per_w[int(w)] = {"n_skip": int(xf.sum()), "diff": float(np.nanmean(r[xf]) - np.nanmean(r[~xf]))}
    return {"row": row, "code": code, "month": month, "per_window": per_w,
            "support": bool(np.isfinite(row["D"]) and row["D"] < 0 and row["p_less"] < 0.05 and code["p_less"] < 0.05)}


def band_test(g: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """20件以上 − 8〜19件（発火 SKIP 以上の日の中で）。窓で並べ替え、両側。"""
    nb = g["n_break"].to_numpy()
    sub = g[nb >= SKIP]
    f = sub["n_break"].to_numpy() >= STRONG
    return strat_perm(pd.to_numeric(sub["tp10c"], errors="coerce").to_numpy(dtype=float), f, sub["window"].to_numpy(),
                      n_perm, seed)


def spearman_by_window(g: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED, min_rows: int = 10) -> dict:
    """テスト窓ごとの、発火数と収益の順位相関。窓の平均と、窓ごとに収益を並べ替えたときの平均の分布（両側 p）。"""
    parts = []
    for _, x in g.groupby("window"):
        r = pd.to_numeric(x["tp10c"], errors="coerce")
        ok = r.notna().to_numpy()
        if ok.sum() >= min_rows and x["n_break"].nunique() > 1:
            parts.append((x["n_break"].to_numpy(dtype=float)[ok], r.to_numpy(dtype=float)[ok]))

    def rho(a, b):
        ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
        if np.std(ra) == 0 or np.std(rb) == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    obs = np.array([rho(a, b) for a, b in parts], dtype=float)
    if not len(obs):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = np.nanmean([rho(a, b[rng.permutation(len(b))]) for a, b in parts])
    m = float(np.nanmean(obs))
    return {"n": int(len(obs)), "mean": m, "pos": int((obs > 0).sum()),
            "p_two": float((np.sum(np.abs(null) >= abs(m)) + 1) / (n_perm + 1)), "per_window": obs.tolist()}


def trade_form(ft: pd.DataFrame, bar_days, min_break: int, months: int) -> tuple[dict, pd.DataFrame]:
    """画面の3枠の形（今の規則・上位2件・同時に3件・+10%）で、発火 min_break 件以上の日だけ買う。"""
    t = E87.run_one(ft, bar_days, "K15", 95.0, min_break, LT.TOP_K, LT.SLOTS, 0, 10.0)
    s = stats(t)
    r = pd.to_numeric(t["tp10c"], errors="coerce").dropna()
    s["capital_pm"] = float(r.sum() / LT.SLOTS / months) if months else float("nan")
    s["quiet"] = int((t["n_break"].to_numpy() < SKIP).sum()) if len(t) else 0
    return s, t


def line(name: str, s: dict, width: int = 26) -> str:
    return (f"  {LT._l(name, width)}{s['n']:>5}件  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}・SE {s['se']:.2f}）  勝率 "
            f"{s['win']:4.1f}%  到達 {s['hit']:4.1f}%  −10%割れ {s['bad10']:4.1f}%")


def fmt_p(t: dict, one_sided: bool = True) -> str:
    if not np.isfinite(t.get("D", np.nan)):
        return "—（片側が0件）"
    one = f"片側 p = {t['p_less']:.3f}・" if one_sided else ""
    return f"D {t['D']:+.2f}pt（{t['n_yes']}件 対 {t['n_no']}件）・{one}両側 p = {t['p_two']:.3f}"


def report(ft: pd.DataFrame, bar_days=None, ff: pd.DataFrame | None = None, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    summary: dict = {}
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    days = ft.groupby(pd.to_datetime(ft["Date"]))["n_break"].first()
    k = rule_mask(ft, "min_K15", 95.0)
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月。候補のある日 {len(days):,}日のうち発火{SKIP - 1}件以下 {int((days < SKIP).sum())}日"
          f"（{(days < SKIP).mean() * 100:.1f}%）・{STRONG}件以上 {int((days >= STRONG).sum())}日。今の規則の行 {int(k.sum())}件のうち"
          f"発火{SKIP - 1}件以下の日 {int((k & (ft['n_break'].to_numpy() < SKIP)).sum())}件")
    summary["days"] = {"all": int(len(days)), "quiet": int((days < SKIP).sum()), "strong": int((days >= STRONG).sum())}

    # ---- 1. 判定
    g = ft[k]
    st = skip_test(g, n_perm, seed)
    print(f"\n■ 1. 判定（今の規則・前向き）: 発火{SKIP - 1}件以下の日 − {SKIP}件以上の日")
    for name, m in ((f"発火{SKIP - 1}件以下の日", g["n_break"].to_numpy() < SKIP), (f"発火{SKIP}件以上の日", g["n_break"].to_numpy() >= SKIP)):
        print(line(name, stats(g[m])))
    print(f"  行: {fmt_p(st['row'])}（窓ごとに並べ替え）")
    print(f"  銘柄単位: {fmt_p(st['code'])}")
    print(f"  参考・同じ月の中で並べ替え: {fmt_p(st['month'])}")
    print("  窓ごと（両側に1件以上ある窓）: " + ("  ".join(f"窓{w} {v['diff']:+.2f}pt（{v['n_skip']}件）" for w, v in st["per_window"].items())
                                          or "—"))
    print(f"  判定: {'見送りの線を支える' if st['support'] else '見送りの線を支えない'}（D < 0 {'○' if np.isfinite(st['row']['D']) and st['row']['D'] < 0 else '×'} / "
          f"行の p < 0.05 {'○' if st['row']['p_less'] < 0.05 else '×'} / 銘柄単位の p < 0.05 {'○' if st['code']['p_less'] < 0.05 else '×'}）")
    summary["main"] = st

    # ---- 2. 帯ごとの成績
    print("\n■ 2. 発火の帯ごとの成績（前向き）")
    summary["bands"] = {}
    for key, name, col, ln in RULES:
        gg = ft[rule_mask(ft, col, ln)]
        nb = gg["n_break"].to_numpy()
        print(f"  {name}")
        summary["bands"][key] = {}
        for lo, hi, bname in BANDS:
            s = stats(gg[band_mask(nb, lo, hi)])
            summary["bands"][key][bname] = s
            print(line("  " + bname, s))
        fine = []
        for lo, hi in FINE:
            s = stats(gg[band_mask(nb, lo, hi)])
            bn = f"{lo}〜{hi}件" if hi else f"{lo}件以上"
            fine.append(f"{bn} {s['n']}件 {s['mean']:+.2f}%" if s["n"] else f"{bn} 0件")
            summary["bands"][key][f"fine_{lo}"] = s
        print("    細かく: " + " / ".join(fine))

    # ---- 3. 傾向
    print("\n■ 3. 傾向: テスト窓ごとの、発火数と収益の順位相関（スピアマン。前向き）")
    summary["trend"] = {}
    for key, name, col, ln in RULES:
        sp = spearman_by_window(ft[rule_mask(ft, col, ln)], n_perm=min(n_perm, 500), seed=seed)
        summary["trend"][key] = sp
        if sp.get("n"):
            print(f"  {LT._l(name, 30)}窓の平均 {sp['mean']:+.3f}（正の窓 {sp['pos']}/{sp['n']}）・両側 p = {sp['p_two']:.3f}")

    # ---- 4. 同じ判定をほかの行で
    print(f"\n■ 4. 参考: 同じ判定（発火{SKIP - 1}件以下 − {SKIP}件以上）をほかの行で")
    summary["other"] = {}
    for key, name, col, ln in RULES[1:]:
        t = skip_test(ft[rule_mask(ft, col, ln)], n_perm, seed)
        summary["other"][key] = t
        print(f"  {LT._l(name, 30)}行 {fmt_p(t['row'])}")
        print(f"  {LT._l('', 30)}銘柄単位 {fmt_p(t['code'])}・同じ月の中 {fmt_p(t['month'])}")

    # ---- 5. 本命の線
    print(f"\n■ 5. 参考: {STRONG}件以上（本命）− {SKIP}〜{STRONG - 1}件（今の規則・前向き）")
    bt = band_test(g, n_perm, seed)
    summary["strong"] = bt
    print(f"  {fmt_p(bt, one_sided=False)}")

    # ---- 6. 3枠の形
    if bar_days is not None and len(bar_days):
        print(f"\n■ 6. 参考: 画面の3枠の形（今の規則・上位{LT.TOP_K}件・同時に{LT.SLOTS}件・+10%）で、発火{SKIP}件以上の日だけ / 発火を問わない")
        s8, t8 = trade_form(ft, bar_days, SKIP, months)
        s1, t1 = trade_form(ft, bar_days, 1, months)
        for name, s in ((f"発火{SKIP}件以上の日だけ（今）", s8), ("発火を問わない", s1)):
            print(line(name, s, 30) + f"  資金全体で月 {s['capital_pm']:+.2f}%（発火{SKIP - 1}件以下の日の取引 {s['quiet']}件）")
        m8, m1 = t8.groupby("window")["tp10c"].mean(), t1.groupby("window")["tp10c"].mean()
        j = m8.index.intersection(m1.index)
        dw = summarize((m8[j] - m1[j]).to_numpy(dtype=float))
        print(f"  窓ごとの差（発火{SKIP}件以上だけ − 問わない、1取引の平均）: {fmt_s(dw)}")
        summary["trade"] = {"skip": s8, "all": s1, "diff": dw}

    # ---- 7. 画面の物差し
    if ff is not None and len(ff):
        print("\n■ 7. 参考: 画面の物差し（全期間の百分位）")
        kf = rule_mask(ff, "min_K15", 95.0)
        gf = ff[kf]
        sf = skip_test(gf, n_perm, seed)
        summary["full"] = {"main": sf}
        for lo, hi, bname in BANDS:
            s = stats(gf[band_mask(gf["n_break"].to_numpy(), lo, hi)])
            summary["full"][bname] = s
            print(line(bname, s))
        print(f"  判定と同じ形: 行 {fmt_p(sf['row'])}・銘柄単位 {fmt_p(sf['code'])}")

    print("\n■ まとめ")
    r = st["row"]
    print(f"  今の規則（前向き）: 発火{SKIP - 1}件以下の日 {r['n_yes']}件、差 D {r['D']:+.2f}pt・片側 p = {r['p_less']:.3f}"
          f"（銘柄単位 {st['code']['p_less']:.3f}）→ {'見送りの線を支える' if st['support'] else '見送りの線を支えない'}")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験88: 発火数とリターン（今の規則で測り直す）")
    ap.add_argument("--perm", type=int, default=N_PERM)
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
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = E87.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    ff = E87.build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。今の規則の行 {int(rule_mask(ft, 'min_K15', 95.0).sum())}件"
        f"（probe_k15_baseline は 559件）")
    summary = report(ft, bar_days=bar_days, ff=ff, n_perm=args.perm)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e88_summary.json")

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
