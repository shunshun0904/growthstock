#!/usr/bin/env python3
"""
1.5σ の3本（lgbm_k15・xgb_k15・cat_k15）がすべて95以上を、選定のベースラインにしたら過去はどうだったか（見るだけ）。

運用者の話（2026-10-12）「1.5σの3モデルで95以上を選定のベースラインにしようかなと思います。売りの指値は10%です」と依頼
「両方お願いします。今日の戦略パネルも変えてください」。今の目安（1.2σ の3本がすべて95以上）と並べて出す。判定はしない
（どちらが良いかは実験82・87 の通り、過去のデータでは見分けられない）。画面の戦略パネルの文言に使う数字もここで測る。

出すもの（出口はどれも +10% の指値・届かなければ20営業日目の終値）
  0. 確かめ: 前向きの 1.2σ の3本 95以上は 543件、画面の3枠の形は 93件・+2.33%（実験83・86・87 と同じ）
  1. 月ごとの件数（前向き・画面の物差し）と、0件の月・続いた最長・年ごと。グラフ用の csv の行
  2. 重なり: 1.5σ の3本 95以上のうち、1.2σ の3本も95以上の割合（と逆）
  3. +10% の成績: 行全体 / 月2件の先着 / 画面の3枠の形（発火8件以上の日に上位2件・同時に3件）。前向きと画面の物差し。
     窓ごとの差（1.5σ − 1.2σ。見るだけ）
  4. 画面の文言用（画面の物差し・1.5σ の3本 95以上）: 行全体で +10% に届く割合と届いた日の分位（1 = 買った日）、
     発火の帯（20件以上 / 8〜19件 / 7件以下）ごとの成績、惜しい帯（3本の最小 85〜95）の幅（最大 − 最小 5 未満 / 5 以上）ごと
     の成績、3枠の形の1取引・月あたり（1件に資金の1/3）

公開ログには集計だけ（件数・平均・割合・分位）。銘柄・日付ごとの値は出さない。

    exp=probe_k15_baseline.py

結果（2026-10-12、run 38213352870、372f7e8、54秒。docs/MODEL_ADOPTION_RULES.md §46）
------------------------------------------------------------------
  確かめ: 前向きの 1.2σ の3本 95以上 543件・画面の3枠の形 93件・+2.33%（実験83・86・87 と同じ）
  件数: 1.5σ の3本 95以上は前向き 559件（0件の月 4 / 53）・画面の物差し 354件（22 / 65）。1.2σ の3本は 543件（11）・371件（29）。
        重なりは 1.5σ の 60%（前向き）
  +10%（前向き）: 1.5σ の3本 行全体 +3.09%・月2件の先着 90件 +0.24%・3枠 118件 +1.82%（資金全体で月 +1.35%）。
        1.2σ の3本 +3.12%・74件 −0.26%・93件 +2.33%（月 +1.36%）。窓ごとの差（1.5σ − 1.2σ）−0.11 / +0.25 / −0.16pt
  +10%（画面の物差し）: 1.5σ の3本 +3.58%・77件 +0.26%・107件 +1.51%（月 +0.83%）。1.2σ の3本 +3.60%・63件 +1.63%・
        80件 +3.67%（月 +1.51%）。窓ごとの差 −0.53 ± 0.70 / −3.65 ± 1.79（9窓中1）/ −4.26 ± 2.09pt（9窓中2）
  画面の文言用（画面の物差し・1.5σ の3本 95以上）: 届く 44.1%（届いた日の中央 9営業日）。3枠で買えた取引は 40.2%・中央 7営業日。
        発火20件以上 192件 +4.89%・8〜19件 140件 +1.51%・7件以下 22件 +5.37%。惜しい帯（最小 85〜95・発火8件以上）
        幅5未満 369件 +1.05%・幅5以上 584件 +2.08%
"""
from __future__ import annotations

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
import e87_monthly_return_search as E87  # noqa: E402
import probe_monthly_counts as PMC  # noqa: E402

MODELS = E79.MODELS
LINE = 95.0
RULES = (("K15", "1.5σ の3本 95以上"), ("T3", "1.2σ の3本 95以上"))
NEAR_LO = 85.0
WIDE = 5.0                      # 惜しい帯で「1つだけ低い」とみなす幅（画面の nearMisses と同じ）
BANDS = ((20, None, "発火20件以上"), (8, 19, "発火8〜19件"), (0, 7, "発火7件以下"))
CSV_COLS = ("month", "breaks", "f_k15", "f_t3", "s_k15", "s_t3")
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s


def add_spread(ft: pd.DataFrame, parts: list, code: str = "K15", models=MODELS) -> pd.DataFrame:
    """組 code の百分位（画面と同じ丸め）の最大を max_<code> として足す（E87.build と同じ行の順）。"""
    grp = dict((c, g) for c, _, g in E87.SETS)[code]
    idx = [list(models).index(a) for a in grp]
    mx = np.concatenate([E82.rounded(np.asarray(P, dtype=float)[:, idx]).max(axis=1) for _, P in parts])
    return ft.assign(**{f"max_{code}": mx})


def rule_mask(ft: pd.DataFrame, code: str, line: float = LINE) -> np.ndarray:
    return ft[f"min_{code}"].to_numpy(dtype=float) >= line


def stats(g: pd.DataFrame, months: int | None = None) -> dict:
    """+10% の成績（%）。hit = 指値に届いた割合、day = 届いた日の 25/50/75% 点（1 = 買った日）。"""
    r = pd.to_numeric(g["tp10c"], errors="coerce").dropna()
    n = int(len(r))
    nan = float("nan")
    hit = g.loc[r.index, "hit10"].astype(bool) if n else pd.Series(dtype=bool)
    d = pd.to_numeric(g.loc[r.index][hit.to_numpy()]["hitday10"], errors="coerce").dropna() if n else pd.Series(dtype=float)
    out = {"n": n, "mean": float(r.mean()) if n else nan, "med": float(r.median()) if n else nan,
           "win": float((r > 0).mean() * 100) if n else nan, "hit": float(hit.mean() * 100) if n else nan,
           "bad10": float((r < -10).mean() * 100) if n else nan,
           "day25": float(np.percentile(d, 25)) if len(d) else nan, "day50": float(np.percentile(d, 50)) if len(d) else nan,
           "day75": float(np.percentile(d, 75)) if len(d) else nan}
    if months:
        per_m = g.assign(_m=pd.to_datetime(g["Date"]).dt.to_period("M")).groupby("_m").size() if n else pd.Series(dtype=int)
        out.update(per_month=n / months, two=int((per_m >= 2).sum()), one=int((per_m == 1).sum()),
                   zero=int(months - (per_m >= 1).sum()))
    return out


def fmt_line(name: str, s: dict, width: int = 34) -> str:
    extra = f"（2件の月 {s['two']}・1件 {s['one']}・0件 {s['zero']}）" if "two" in s else ""
    return (f"  {LT._l(name, width)}{s['n']:>5}件{extra}  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:4.1f}%  "
            f"到達 {s['hit']:4.1f}%（届いた日の中央 {s['day50']:.0f}営業日）  −10%割れ {s['bad10']:4.1f}%")


def window_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ma = a.groupby("window")["tp10c"].mean()
    mb = b.groupby("window")["tp10c"].mean()
    j = ma.index.intersection(mb.index)
    return summarize((ma[j] - mb[j]).to_numpy(dtype=float))


def monthly_counts(ft: pd.DataFrame) -> pd.DataFrame:
    """月ごとの件数（取引の無い月も含む）: 発火（候補全体）・各規則の行。"""
    d = pd.to_datetime(ft["Date"])
    allm = pd.period_range(d.min().to_period("M"), d.max().to_period("M"), freq="M")
    m = d.dt.to_period("M")
    out = pd.DataFrame(index=allm)
    out["breaks"] = m.value_counts().reindex(allm, fill_value=0).astype(int)
    for code, _ in RULES:
        out[code] = m[rule_mask(ft, code)].value_counts().reindex(allm, fill_value=0).astype(int)
    return out


def scale_block(ft: pd.DataFrame, bar_days, name: str) -> dict:
    """1つの物差し（前向き / 画面）での 1〜3。"""
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    out: dict = {"months": months}
    tbl = monthly_counts(ft)
    print(f"\n■ 1. 月ごとの件数（{name}・{months}か月）")
    for code, ja in RULES:
        z = PMC.zero_runs(tbl[code].astype(float))
        yr = tbl.index.year
        per_year = "  ".join(f"{y}年 {int((tbl.loc[yr == y, code] == 0).sum())}（{int((yr == y).sum())}か月）"
                             for y in sorted(set(yr)))
        tail = f"・最長 {z['longest']}か月（{z['longest_from']} 〜 {z['longest_to']}）" if z["longest"] else ""
        print(f"  {LT._l(ja, 22)}合計 {int(tbl[code].sum()):>4}件（月 {tbl[code].mean():.1f}件）・0件の月 {z['zero']} / "
              f"{z['months']}{tail}")
        print(f"    年ごとの0件の月: {per_year}")
        out[f"count_{code}"] = {"total": int(tbl[code].sum()), "zero": z["zero"], "longest": z["longest"]}
    out["table"] = tbl

    k, t = rule_mask(ft, "K15"), rule_mask(ft, "T3")
    print(f"\n■ 2. 重なり（{name}）: 1.5σ の3本 95以上 {int(k.sum())}件のうち 1.2σ の3本も95以上 {int((k & t).sum())}件"
          f"（{(k & t).sum() / max(k.sum(), 1) * 100:.1f}%）。1.2σ の3本 95以上 {int(t.sum())}件のうち 1.5σ も "
          f"{(k & t).sum() / max(t.sum(), 1) * 100:.1f}%")
    out["overlap"] = {"k15": int(k.sum()), "t3": int(t.sum()), "both": int((k & t).sum())}

    print(f"\n■ 3. +10% の成績（{name}）")
    res = {}
    for code, ja in RULES:
        rows = ft[rule_mask(ft, code)]
        first2 = E87.run_one(ft, bar_days, code, LINE, 1, 3, 5, 2, 10.0)
        screen = E87.run_one(ft, bar_days, code, LINE, LT.PREV_SKIP_BREAKS, LT.TOP_K, LT.SLOTS, 0, 10.0)
        res[code] = {"rows": rows, "first2": first2, "screen": screen}
        s_rows, s_f2, s_sc = stats(rows, months), stats(first2, months), stats(screen, months)
        r_sc = pd.to_numeric(screen["tp10c"], errors="coerce").dropna()
        s_sc["capital_pm"] = float(r_sc.sum() / LT.SLOTS / months) if months else float("nan")
        print(f"  {ja}")
        print(fmt_line("  行全体", s_rows))
        print(fmt_line("  月2件の先着", s_f2))
        print(fmt_line("  画面の3枠の形", s_sc) + f"  資金全体で月 {s_sc['capital_pm']:+.2f}%（1件に1/3）")
        out[code] = {"rows": s_rows, "first2": s_f2, "screen": s_sc}
    print("  窓ごとの差（1.5σ − 1.2σ。見るだけ）: "
          + "、".join(f"{lab_} {fmt_s(window_diff(res['K15'][k_], res['T3'][k_]))}"
                     for k_, lab_ in (("rows", "行全体"), ("first2", "月2件の先着"), ("screen", "画面の3枠の形"))))
    out["diff"] = {k_: window_diff(res["K15"][k_], res["T3"][k_]) for k_ in ("rows", "first2", "screen")}
    return out


def panel_block(ff: pd.DataFrame, months: int) -> dict:
    """画面の文言用の数字（画面の物差し・1.5σ の3本 95以上）。"""
    out: dict = {}
    k = rule_mask(ff, "K15")
    nb = ff["n_break"].to_numpy()
    print("\n■ 4. 画面の文言用（画面の物差し・1.5σ の3本 95以上）")
    s_all = stats(ff[k])
    print(f"  行全体 {s_all['n']}件: +10% に届く {s_all['hit']:.1f}%・届いた日 25/50/75% 点 {s_all['day25']:.0f} / "
          f"{s_all['day50']:.0f} / {s_all['day75']:.0f}営業日（1 = 買った日）・1件 {s_all['mean']:+.2f}%（中央 {s_all['med']:+.2f}）・"
          f"勝率 {s_all['win']:.1f}%・−10%割れ {s_all['bad10']:.1f}%")
    out["all"] = s_all
    out["bands"] = {}
    for lo, hi, name in BANDS:
        m = k & (nb >= lo) & ((nb <= hi) if hi is not None else True)
        s = stats(ff[m])
        out["bands"][name] = s
        print(f"  {LT._l(name, 14)}{s['n']:>4}件  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:4.1f}%  "
              f"−10%割れ {s['bad10']:4.1f}%")
    mn = ff["min_K15"].to_numpy(dtype=float)
    sp = ff["max_K15"].to_numpy(dtype=float) - mn
    near = (mn >= NEAR_LO) & (mn < LINE) & (nb >= LT.PREV_SKIP_BREAKS)
    out["near"] = {}
    for name, m in (("惜しい帯・幅5未満", near & (sp < WIDE)), ("惜しい帯・幅5以上", near & (sp >= WIDE))):
        s = stats(ff[m])
        out["near"][name] = s
        print(f"  {LT._l(name, 18)}{s['n']:>4}件  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:4.1f}%  "
              f"−10%割れ {s['bad10']:4.1f}%（3本の最小 {NEAR_LO:g}〜{LINE:g}・発火{LT.PREV_SKIP_BREAKS}件以上の日）")
    return out


def main(argv=None) -> int:
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
    parts_f = [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()]
    parts_s = [(np.arange(len(df)), E79.pct_matrix(df, MODELS))]
    ft = add_spread(E87.build(df, parts_f), parts_f)
    ff = add_spread(E87.build(df, parts_s), parts_s)

    # 確かめ
    sc = E87.run_one(ft, bar_days, "T3", LINE, LT.PREV_SKIP_BREAKS, LT.TOP_K, LT.SLOTS, 0, 10.0)
    print(f"■ 0. 確かめ: 前向きの 1.2σ の3本 95以上 {int(rule_mask(ft, 'T3').sum())}件（実験82・83 は 543件）、画面の3枠の形 "
          f"{len(sc)}件・+{pd.to_numeric(sc['tp10c']).mean():.2f}%（実験83・87 は 93件・+2.33%）")

    fw = scale_block(ft, bar_days, "前向き")
    sw = scale_block(ff, bar_days, "画面の物差し")
    pn = panel_block(ff, sw["months"])

    print("\n■ 5. グラフ用（csv。月ごとの件数）")
    tf, ts = fw.pop("table"), sw.pop("table")
    print("csv:" + ",".join(CSV_COLS))
    for m in ts.index:
        f_k = "" if m not in tf.index else str(int(tf.loc[m, "K15"]))
        f_t = "" if m not in tf.index else str(int(tf.loc[m, "T3"]))
        print(f"csv:{m},{int(ts.loc[m, 'breaks'])},{f_k},{f_t},{int(ts.loc[m, 'K15'])},{int(ts.loc[m, 'T3'])}")

    print("\n■ まとめ")
    for key, name in (("前向き", fw), ("画面の物差し", sw)):
        k_, t_ = name["K15"], name["T3"]
        print(f"  {key}: 1.5σ の3本 95以上 {name['count_K15']['total']}件（0件の月 {name['count_K15']['zero']} / {name['months']}）・"
              f"月2件の先着 {k_['first2']['mean']:+.2f}%・3枠 {k_['screen']['mean']:+.2f}%、"
              f"1.2σ の3本 95以上 {name['count_T3']['total']}件（0件の月 {name['count_T3']['zero']}）・月2件の先着 "
              f"{t_['first2']['mean']:+.2f}%・3枠 {t_['screen']['mean']:+.2f}%")
    print("  注意: 見るだけ。どちらが良いかは過去のデータでは見分けられない（実験82・87）。手数料・税・すべりは含まない")
    log(f"全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
