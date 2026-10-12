#!/usr/bin/env python3
"""
実験83: 「1.2σ・1.5σ の6本とも 95以上」の銘柄を、+10% の指値（届かなければ 20営業日目の終値で売る）で売買したらどうなるか。

運用者の問い（2026-10-12）「1.2σ、1.5σ の6本とも95以上の場合に、+10%で指値を置いた場合どうなりますか？現実的にはこの
ラインかなと。指値に到達しない場合には、20営業日終値で売ります」。

規則と出口（先に決めたもの。結果を見てから動かさない）
------------------------------------------------
  A15 = 1.2σ の lgbm・xgb・cat と 1.5σ の lgbm_k15・xgb_k15・cat_k15 の6本とも 95以上（実験82 の A15 と同じ）
  R0  = 1.2σ の3本だけ 95以上（今の目安）
  出口 tp10c: 翌営業日の寄りで買い、1〜20営業日目（1 = 買った日）の高値が買値の 1.10倍以上になったら +10% ちょうどで売る
             （寄りで +10% を超えて始まった日も +10% で数える。控えめな側）。届かなければ 20営業日目の終値で売る
             （live_track.forward の ret_close）。手数料・税・すべりは含まない
  百分位: 前向き（実験79 の window_split。テスト窓より前の窓の分布、丸めは画面と同じ小数1桁）。テスト窓 3〜11
  判定（主）: R0 の中で A15 − それ以外（出口 tp10c）を窓ごとに。平均が正・並べ替え p < 0.05・上の窓が過半（実験80・82 と
             同じ3つ）。銘柄単位の p も出す
  参考（判定に使わない）
    - 出口の比べ（同じ行で）: +20% の指値・20日目の終値（tp20c）、今の運用の +20% の指値・5日平均（tp20a）、指値なし・
      20日目の終値（c20）。A15 の行で tp10c − ほかの出口を窓ごとに
    - 指値の水準 +5 / +10 / +15 / +20%（届かなければ 20日目の終値）。水準はここから選ばない（見るだけ）
    - 運用の通りの売買（発火8件以上の日に上位2件・3枠・先着順）を R0 / A15 × 出口 tp10c / tp20a で
    - 全期間の百分位（画面の物差し。先の分布も含むので甘い）
  前の結果: 実験41 §8 では「+20% の指値を置かずに 20日目の終値で売る」ほうが1取引でも枠3の年率でも上、PLAYBOOK の
  「+5% / +10% の早い利確」は平均が 1.0〜1.9pt 下がった（どちらも今の6本の条件より前の規則）

公開ログには集計だけ（群ごとの件数・平均・割合、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e83_tp10_rule.py
    args="--perm 2000"        （既定）

結果（2026-10-12、run 38202969447、1173b7f、56秒。docs/MODEL_ADOPTION_RULES.md §40）
------------------------------------------------------------------
  データ: 12本 16,320件。20日目の終値は 16,319行に付き、5日平均は ret_o1_20 と全行で一致、指値の到達の順序の食い違い 0件。
          全行で指値に届いた割合 +5% 57.2%・+10% 31.7%・+15% 18.5%・+20% 11.5%
  A15（333件・月 6.3件）× +10%: 1取引 +3.59%（中央 +5.96）・勝率 70.9%・到達 43.2%・−10%割れ 5.1%
    同じ行で +20%・5日平均（今の出口）+4.40%、+20%・20日目の終値 +4.61%、指値なし +5.12%
    +10% − 今の出口 −0.19 ± 0.40pt、+10% − 指値なし −1.12 ± 0.54pt（窓ごと）
  R0（543件）× +10%: +3.11%・勝率 68.9%。A15 − R0 の残り +0.66 ± 1.16pt（6窓中4）・p = 0.31 → 通らない
  3枠で売買: A15 × +10% 84取引・1取引 +2.45%（中央 +5.82）・勝率 63.1%・月 +1.35%、A15 × +20% 月 +1.52%、
             今の運用（R0 × +20%・5日平均）月 +1.44%、R0 × 指値なし 月 +1.64%
  A15 × +10% の窓ごとの平均は9窓中7でプラス。ただし行の 67% が2つの窓（窓6・窓9）に集まる
  → +10% の指値は勝率と中央値を上げるが、平均は +20% よりわずかに、指値なしより約1pt 下がる。月の収益はほぼ同じ
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

OUTCOME = E79.OUTCOME
MODELS = E79.MODELS
TARGETS = (5.0, 10.0, 15.0, 20.0)       # 指値の水準（参考の表）。主は 10
MAIN_TP = 10.0
HOLD = LT.HOLD                          # 20営業日
#: (列, 表示, 到達の印の列, 到達日の列)
EXITS = (
    ("tp10c", "+10% 指値・20日目の終値", "hit10", "hitday10"),
    ("tp20c", "+20% 指値・20日目の終値", "hit20", "hitday20"),
    ("tp20a", "+20% 指値・5日平均（今の運用）", "hit20", "hitday20"),
    ("c20", "指値なし・20日目の終値", None, None),
)
EXIT_JA = {c: ja for c, ja, _, _ in EXITS}
MAIN_EXIT = "tp10c"
RULES = ("R0", "A15")
N_PERM = E82.N_PERM
SEED = E82.SEED
MIN_GROUP = E82.MIN_GROUP
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


def tp_key(t: float) -> str:
    return f"{int(round(t))}"


# --------------------------------------------------------------------------- #
# 出口
# --------------------------------------------------------------------------- #

def attach_exits(df: pd.DataFrame, bars: pd.DataFrame, targets=TARGETS, hold: int = HOLD) -> tuple[pd.DataFrame, dict]:
    """
    指値の水準ごとに、到達の印 hit<k>・到達日 hitday<k>（1 = 買った日）・収益 tp<k>c（到達なら +k%、届かなければ 20営業日目の
    終値）を足す。c20（20日目の終値）・hold5（20日目の5日平均 = ret_o1_20）・tp20a（+20% 指値・5日平均）も。収益は %。
    live_track.forward の status は水準によらず「+20%到達」と書くので、到達は hit_day の有無で決める。
    """
    keys = df[["Code", "Date"]]
    out = df.copy()
    for t in targets:
        k = tp_key(t)
        fw = LT.forward(bars, keys, target=t, hold=hold)
        fw = fw[["Code", "Date", "hit_day", "ret_close", "ret_hold"]].rename(
            columns={"hit_day": f"hitday{k}", "ret_close": f"_c20_{k}", "ret_hold": f"_h5_{k}"})
        out = out.merge(fw, on=["Code", "Date"], how="left")
        out[f"hit{k}"] = out[f"hitday{k}"].notna()
        if "c20" not in out.columns:
            out["c20"] = pd.to_numeric(out[f"_c20_{k}"], errors="coerce")
            out["hold5"] = pd.to_numeric(out[f"_h5_{k}"], errors="coerce")
        out[f"tp{k}c"] = np.where(out[f"hit{k}"], t, out["c20"])
        if abs(t - 20.0) < 1e-9:
            out["tp20a"] = np.where(out[f"hit{k}"], t, out["hold5"])
        out = out.drop(columns=[f"_c20_{k}", f"_h5_{k}"])
    # 確かめ: 高い指値に届いた行は低い指値にも同じ日以前に届いている。5日平均は ret_o1_20 と一致する
    ks = [tp_key(t) for t in sorted(targets)]
    bad_order = 0
    for lo, hi in zip(ks, ks[1:]):
        h_hi = out[f"hit{hi}"].to_numpy()
        bad_order += int((h_hi & ~out[f"hit{lo}"].to_numpy()).sum())
        both = h_hi & out[f"hit{lo}"].to_numpy()
        bad_order += int((out.loc[both, f"hitday{lo}"].to_numpy() > out.loc[both, f"hitday{hi}"].to_numpy()).sum())
    a = out["hold5"].to_numpy(dtype=float)
    b = pd.to_numeric(out[OUTCOME], errors="coerce").to_numpy(dtype=float) * 100
    both = np.isfinite(a) & np.isfinite(b)
    chk = {"rows": int(len(out)), "with_close": int(np.isfinite(out["c20"].to_numpy(dtype=float)).sum()),
           "hold5_match": int((np.abs(a[both] - b[both]) < 0.01).sum()), "hold5_both": int(both.sum()),
           "order_violations": bad_order,
           "hit_share": {tp_key(t): float(out[f"hit{tp_key(t)}"].mean() * 100) for t in targets}}
    return out, chk


EXTRA = ("c20", "hold5", "tp20a") + tuple(f"{p}{tp_key(t)}{s}" for t in TARGETS
                                          for p, s in (("tp", "c"), ("hit", ""), ("hitday", "")))


def build(df: pd.DataFrame, parts: list) -> pd.DataFrame:
    """実験82 の行（規則の印・1.2σ の3本の最小・合意・lgbm のスコア）に、出口の列を足す。"""
    ft = E82.build_frame(df, parts)
    cols = [c for c in EXTRA if c in df.columns]
    extra = pd.concat([df.iloc[rows][cols].reset_index(drop=True) for rows, _ in parts], ignore_index=True)
    for c in cols:
        ft[c] = extra[c].to_numpy()
    return ft


# --------------------------------------------------------------------------- #
# 成績
# --------------------------------------------------------------------------- #

def exit_stats(g: pd.DataFrame, col: str, hit: str | None, months: float) -> dict:
    """出口 col の成績（%）。hit は指値に届いた割合（指値なしは NaN）。"""
    r = pd.to_numeric(g[col], errors="coerce")
    ok = r.notna()
    r = r[ok]
    nan = float("nan")
    return {"n": int(len(g)), "per_month": len(g) / months if months else nan,
            "mean": float(r.mean()) if len(r) else nan, "med": float(r.median()) if len(r) else nan,
            "win": float((r > 0).mean() * 100) if len(r) else nan,
            "hit": float(g.loc[ok, hit].astype(bool).mean() * 100) if (hit and len(r)) else nan,
            "bad10": float((r < -10).mean() * 100) if len(r) else nan,
            "worst": float(r.min()) if len(r) else nan,
            "se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else nan}


def paired_by_window(g: pd.DataFrame, a: str, b: str, min_rows: int = MIN_GROUP) -> dict:
    """同じ行で出口 a − 出口 b の平均を窓ごとに。"""
    d = []
    for _, x in g.groupby("window"):
        v = (pd.to_numeric(x[a], errors="coerce") - pd.to_numeric(x[b], errors="coerce")).dropna()
        if len(v) >= min_rows:
            d.append(float(v.mean()))
    return summarize(np.array(d, dtype=float))


def window_means(g: pd.DataFrame, col: str) -> list:
    """窓ごとの (窓, 件数, 平均)。"""
    return [(int(w), int(len(x)), float(pd.to_numeric(x[col], errors="coerce").mean()))
            for w, x in g.groupby("window")]


def trade_sim(ft: pd.DataFrame, flag: str, col: str, hit: str | None, hitday: str | None, bar_days, n_days: int,
              key: str = "pmin3", keep_trades: bool = False) -> dict:
    """
    運用の通りの売買（発火8件以上の日に上位2件・3枠・先着順）。出口 col。指値に届いた取引は到達日の翌営業日から枠が空き
    （live_track.simulate の「+20%到達」の扱いを、どの水準の指値にも使う）、届かなければ 20営業日の翌営業日から空く。
    key はその日の並べ方（既定は今の画面と同じ 1.2σ の3本の最小）。keep_trades なら買えた取引の収益（rets）も返す。
    """
    picks = E82.daily_picks(ft, flag, key=key)
    if not len(picks):
        return {"picks": 0, "n": 0, "by_window": {}}
    p = picks.copy()
    p["ret"] = pd.to_numeric(p[col], errors="coerce")
    hit_v = p[hit].astype(bool).to_numpy() if hit else np.zeros(len(p), dtype=bool)
    p["status"] = np.where(p["ret"].isna(), "日足なし", np.where(hit_v, "+20%到達", "満了"))
    p["hit_day"] = p[hitday].to_numpy(dtype=float) if hitday else np.nan
    sim = LT.simulate(p, bar_days, slots=LT.SLOTS, hold=HOLD)
    taken = sim[sim["taken"] == LT.TAKEN]
    st = LT.trade_stats(taken, n_days, slots=LT.SLOTS)
    st["picks"] = int(len(picks))
    st["full"] = int((sim["taken"] == LT.FULL).sum())
    r = pd.to_numeric(taken["ret"], errors="coerce").dropna()
    st["bad10"] = float((r < -10).mean() * 100) if len(r) else float("nan")
    st["by_window"] = {int(w): float(pd.to_numeric(x["ret"], errors="coerce").mean()) for w, x in taken.groupby("window")}
    if keep_trades:
        st["rets"] = r.to_numpy(dtype=float)
    return st


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict, width: int = 40) -> str:
    hit = "" if not np.isfinite(s["hit"]) else f"  指値到達 {s['hit']:4.1f}%"
    return (f"  {LT._l(name, width)}{s['n']:>6,}件（月 {s['per_month']:4.1f}件）  1取引 {s['mean']:+6.2f}%"
            f"（中央 {s['med']:+5.2f}・SE {s['se']:.2f}）  勝率 {s['win']:4.1f}%{hit}  −10%割れ {s['bad10']:4.1f}%  "
            f"最悪 {s['worst']:+6.1f}%")


def report(ft: pd.DataFrame, bar_days=None, ff: pd.DataFrame | None = None, n_perm: int = N_PERM,
           seed: int = SEED) -> dict:
    summary: dict = {}
    months = E82.n_months(ft)
    wins = sorted(int(w) for w in ft["window"].unique())
    print(f"\n■ 0. 前向きのテスト窓 {len(wins)}つ（{wins[0]}〜{wins[-1]}）・{len(ft):,}行・{months}か月"
          f"（{pd.to_datetime(ft['Date']).min().date()} 〜 {pd.to_datetime(ft['Date']).max().date()}）")
    assert not (ft["A15"] & ~ft["R0"]).any(), "A15 が R0 の外にある"
    summary["windows"], summary["months"] = wins, months

    print("\n■ 1. 群ごと・出口ごとの成績（前向きの百分位。手数料・税・すべりは含まない）")
    groups = {"all": ("テストの全行", np.ones(len(ft), dtype=bool)),
              "R0": ("R0 1.2σ の3本が95以上", ft["R0"].to_numpy(dtype=bool)),
              "A15": ("A15 1.2σ・1.5σ の6本とも95以上", ft["A15"].to_numpy(dtype=bool)),
              "R0_not_A15": ("R0 で A15 でない", ft["R0"].to_numpy(dtype=bool) & ~ft["A15"].to_numpy(dtype=bool))}
    summary["groups"] = {}
    for key, (name, m) in groups.items():
        print(f"  {name}")
        summary["groups"][key] = {}
        for col, ja, hit, _ in EXITS:
            s = exit_stats(ft[m], col, hit, months)
            summary["groups"][key][col] = s
            print(line(f"  {ja}", s))

    print(f"\n■ 2. 判定（先に決めた比較。R0 の中で A15 − それ以外、出口 {EXIT_JA[MAIN_EXIT]}。並べ替え {n_perm}回）")
    res = E82.split_test(ft, "A15", "R0", col=MAIN_EXIT, n_perm=n_perm, seed=seed)
    rc = E82.split_test(ft, "A15", "R0", col=MAIN_EXIT, n_perm=n_perm, seed=seed, unit="code")
    jd = E82.judge(res)
    va = E82.vs_all(ft, "A15", "R0", col=MAIN_EXIT)
    summary["main"] = {"row": res, "code": {k: rc[k] for k in ("summary", "p")}, "judge": jd, "vs_R0": va}
    print(f"  窓ごとの差 {fmt_s(res['summary'])}  並べ替え p = {res['p']:.3f}  銘柄単位 {fmt_s(rc['summary'])} "
          f"p = {rc['p']:.3f}")
    print(f"  R0 全体との差 {fmt_s(va)}  判定（行単位）: {E82.verdict_text(jd)}")
    print("  窓ごと: " + "  ".join(f"窓{w} {d:+.2f}" for w, d in zip(res["windows"], res["diffs"])))

    print(f"\n■ 3. 出口の比べ（同じ行で {EXIT_JA[MAIN_EXIT]} − ほかの出口。窓ごと）")
    summary["exit_diff"] = {}
    for key in ("A15", "R0"):
        g = ft[groups[key][1]]
        for col, ja, _, _ in EXITS[1:]:
            s = paired_by_window(g, MAIN_EXIT, col)
            summary["exit_diff"][f"{key}_{col}"] = s
            print(f"  {key:<4} − {LT._l(ja, 30)} {fmt_s(s)}")

    print("\n■ 4. 指値の水準（参考。届かなければ 20日目の終値。水準はここから選ばない）")
    summary["targets"] = {}
    for key in ("A15", "R0"):
        g = ft[groups[key][1]]
        for t in TARGETS:
            k = tp_key(t)
            s = exit_stats(g, f"tp{k}c", f"hit{k}", months)
            summary["targets"][f"{key}_{k}"] = s
            print(line(f"{key} +{k}% 指値", s, width=18))
        s = exit_stats(g, "c20", None, months)
        summary["targets"][f"{key}_none"] = s
        print(line(f"{key} 指値なし", s, width=18))

    if bar_days is not None and len(bar_days):
        n_days = LT.n_bar_days(bar_days, pd.to_datetime(ft["Date"]).min(), pd.to_datetime(ft["Date"]).max())
        print(f"\n■ 5. 運用の通りの売買（発火{LT.SKIP_BREAKS}件以上の日に上位{LT.TOP_K}件・{LT.SLOTS}枠・先着順。"
              f"{n_days}営業日）")
        print("  月の収益 = 取引の収益の合計 ÷ 枠数 ÷ 月数（1取引に資金の1/3。複利にしない。手数料・税は含まない）")
        summary["trade"] = {}
        for key in RULES:
            for col, ja, hit, hd in (EXITS[0], EXITS[2], EXITS[3]):
                st = trade_sim(ft, key, col, hit, hd, bar_days, n_days)
                summary["trade"][f"{key}_{col}"] = st
                if not st.get("n"):
                    print(f"  {key} {ja}: 取引なし")
                    continue
                print(f"  {LT._l(key + ' ' + ja, 36)}候補 {st['picks']:>4}件 → 買えた {st['n']:>4}件（月 {st['per_month']:.1f}件）"
                      f"  1取引 {st['mean']:+6.2f}%（中央 {st['median']:+5.2f}）  勝率 {st['win']:4.1f}%  "
                      f"指値到達 {st['hit']:4.1f}%  −10%割れ {st['bad10']:4.1f}%  月の収益 {st['monthly']:+5.2f}%")
        for a, b in ((f"A15_{MAIN_EXIT}", f"R0_{MAIN_EXIT}"), (f"A15_{MAIN_EXIT}", "A15_tp20a"),
                     (f"R0_{MAIN_EXIT}", "R0_tp20a")):
            wa, wb = summary["trade"][a].get("by_window", {}), summary["trade"][b].get("by_window", {})
            j = sorted(set(wa) & set(wb))
            s = summarize(np.array([wa[w] - wb[w] for w in j], dtype=float))
            summary["trade"][f"{a}-{b}"] = s
            print(f"      {a} − {b}（取引の窓ごとの平均の差）: {fmt_s(s)}")

    print(f"\n■ 6. 窓ごとの A15（{EXIT_JA[MAIN_EXIT]}）")
    wm = window_means(ft[groups["A15"][1]], MAIN_EXIT)
    summary["a15_windows"] = wm
    print("  " + "  ".join(f"窓{w} {m:+.2f}%（{n}件）" for w, n, m in wm))
    print(f"  平均がプラスの窓 {sum(1 for _, _, m in wm if m > 0)}/{len(wm)}")

    if ff is not None and len(ff):
        mf = E82.n_months(ff)
        print(f"\n■ 7. 参考: 全期間の百分位（画面の物差し。先の分布も含む）・窓 {int(ff['window'].min())}〜"
              f"{int(ff['window'].max())}・{len(ff):,}行")
        summary["full"] = {}
        for key, name in (("R0", "R0"), ("A15", "A15")):
            g = ff[ff[key].to_numpy(dtype=bool)]
            for col, ja, hit, _ in (EXITS[0], EXITS[2]):
                s = exit_stats(g, col, hit, mf)
                summary["full"][f"{key}_{col}"] = s
                print(line(f"{name} {ja}", s))
        r = E82.split_test(ff, "A15", "R0", col=MAIN_EXIT, n_perm=n_perm, seed=seed)
        summary["full"]["main"] = r
        print(f"    A15 − R0 の残り（{EXIT_JA[MAIN_EXIT]}）: {fmt_s(r['summary'])} p = {r['p']:.3f}")

    print("\n■ まとめ")
    a = summary["groups"]["A15"][MAIN_EXIT]
    r0 = summary["groups"]["R0"][MAIN_EXIT]
    cur = summary["groups"]["A15"]["tp20a"]
    print(f"  A15 を {EXIT_JA[MAIN_EXIT]} で: 1取引 {a['mean']:+.2f}%（中央 {a['med']:+.2f}）・勝率 {a['win']:.1f}%・"
          f"指値到達 {a['hit']:.1f}%・−10%割れ {a['bad10']:.1f}%・月 {a['per_month']:.1f}件")
    print(f"  同じ出口の R0: {r0['mean']:+.2f}%・勝率 {r0['win']:.1f}%。A15 − R0 の残り {fmt_s(summary['main']['row']['summary'])}、"
          f"p = {summary['main']['row']['p']:.3f} → {'通る' if summary['main']['judge']['pass'] else '通らない'}")
    print(f"  A15 の今の出口（+20%・5日平均）: {cur['mean']:+.2f}%・勝率 {cur['win']:.1f}%。同じ行で +10% − 今の出口 "
          f"{fmt_s(summary['exit_diff']['A15_tp20a'])}")
    if "trade" in summary:
        t1, t0 = summary["trade"][f"A15_{MAIN_EXIT}"], summary["trade"]["R0_tp20a"]
        print(f"  3枠で売買: A15 × +10% 月の収益 {t1.get('monthly', float('nan')):+.2f}%（{t1.get('n', 0)}取引）、"
              f"今の運用（R0 × +20%・5日平均）{t0.get('monthly', float('nan')):+.2f}%（{t0.get('n', 0)}取引）")
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験83: 6本とも95以上 × +10% の指値・20日目の終値")
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
    df, chk = attach_exits(df, bars)
    log(f"出口: 20日目の終値が付いた行 {chk['with_close']:,}/{chk['rows']:,}。5日平均と ret_o1_20 の一致 "
        f"{chk['hold5_match']:,}/{chk['hold5_both']:,}。指値の到達の順序の食い違い {chk['order_violations']}件")
    log("  指値に届いた行の割合（全行）: " + "・".join(f"+{k}% {v:.1f}%" for k, v in chk["hit_share"].items()))
    if chk["order_violations"] or chk["hold5_both"] == 0 or chk["hold5_match"] / chk["hold5_both"] < 0.99:
        log("[warn] 出口の確かめに食い違いがある。日足とデータセットの食い違いを疑う")
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    split = E79.window_split(df, folds)
    ft = build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）")
    ff = build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    summary = report(ft, bar_days=bar_days, ff=ff, n_perm=args.perm)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e83_summary.json")

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
