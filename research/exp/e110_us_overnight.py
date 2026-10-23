#!/usr/bin/env python3
"""
実験110: 前夜の米国の下落と、今の規則の買いの成績（検証だけ）。

運用者の依頼（2026-10-23）「全体が悪い日の購入は避けたいので、朝8時に、深夜〜明け方の米国株市場の動向をまとめて…
おおきく米国株市場が下落した日はその日の買いは控えるべきだと思うので」。朝のまとめ（平日 07:31 の定時）の目安
「NASDAQ 総合が −2.0% 以下、または S&P 500 が −1.5% 以下」（運用者の決定）の日に買った分が、本当にほかの日より悪いかを
過去の買いで確かめる（運用者の選択「確かめる」）。画面・規則・朝のまとめは変えない。

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  行: 実験88 と同じ今の規則の行。1.5σ の lgbm_k15・xgb_k15・cat_k15 の前向きの百分位（画面と同じ丸め）がすべて95以上、
      テスト窓 3〜11。収益は +10% の指値・届かなければ20営業日目の終値（tp10c）。買うのは選定日 D の次の営業日 B の寄り
  前夜の米国: B の寄りより前に引けた米国の取引日のうち最後の日 U。米国の日付 U の引け（16:00 ET）は日本時間で U+1日の
      5:00（夏時間）/ 6:00 なので、D ≤ U ≤ B−1日。D より前の米国の日は D の終値に織り込み済みなので入れない。間に米国の
      取引日が無い（米国の休場）ときは印なし。東証の連休で間に2日以上あるときも U の1日だけで決める（朝のまとめと同じ）
  印: U の前日比（終値、小数2桁に丸める）が NASDAQ 総合 ≤ −2.0% または S&P 500 ≤ −1.5%
  判定: D = 印の日の買いの平均 − それ以外の平均（tp10c、全部を足した平均）。並べ替えは日単位（同じ日の買いは同じ地合いを
      受けるので行ごとには並べ替えない）: テスト窓ごとに、規則の行がある買う日 B の印を日の間で並べ替える（印の日数は
      そのまま。2,000回）。片側 p = 並べ替えた D が実際以下になる割合
      「見送りを支える」= D < 0 かつ 片側 p < 0.05。そうでなければ「支えない」
  参考（判定に使わない）
    - 件数・平均・中央・勝率・+10% に届く割合・−10%割れ。見送った場合（印の日の買いを除いた平均・除いた件数と合計）
    - 前夜の S&P 500 の帯ごと（−1.5% 以下 / −1.5〜−0.5 / −0.5〜+0.5 / +0.5% 超）の成績
    - 別の目安（S&P −2% か NASDAQ −3%・S&P −1.5% だけ・NASDAQ −2% だけ）で同じ比べ方
    - 候補全体（規則を問わない前向きの全行）で同じ比べ方（行が多い）
    - 寄りの値ざや（B の寄り ÷ D の終値 − 1）の平均を、印の日とそれ以外で（印の日は安く買えているか）
    - テスト窓ごとの差、画面の3枠の形（今の規則・上位2件・同時に3件・+10%）で印の日だけ買わない形と、全部買う形

データ: 米国の日足（S&P 500・NASDAQ 総合の終値）は FMP（運用者の契約の MCP）から手元に取ったもの。FMP のデータなので
リポジトリ・Actions・Release には置かず、--us-sp / --us-nq で手元のファイル（[{symbol, date, price, volume}, ...]）を渡す。
ほかは実験88 と同じ（data-raw の out-of-fold・日足からデータセットを作る）。公開の場所には集計だけ（件数・平均・割合）。
銘柄・日付ごとの値は出さない。

    python3 research/exp/e110_us_overnight.py --us-sp <S&P 500 の JSON> --us-nq <NASDAQ 総合の JSON>
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
import e88_breaks_returns as E88  # noqa: E402

#: 朝のまとめの目安（運用者の決定 2026-10-23）。前日比 %（小数2桁に丸めて比べる）
SP_LINE = -1.5
NQ_LINE = -2.0
#: 参考の目安（名前, S&P の線, NASDAQ の線）。None はその指数を見ない
ALT = (("S&P −2% か NASDAQ −3%", -2.0, -3.0), ("S&P −1.5% だけ", -1.5, None), ("NASDAQ −2% だけ", None, -2.0))
#: 前夜の S&P 500 の帯（参考）。(下, 上] の %、None は端なし
SP_BANDS = ((None, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, None))
N_PERM = 2000
SEED = 42
OUT_DIR = E79.OUT_DIR
US_COLS = ("buy", "us_date", "sp", "nq", "n_us", "sp_cum", "nq_cum")

log = E79.log
stats = E88.stats
line = E88.line


# --------------------------------------------------------------------------- #
# 米国の日足と、前夜の対応
# --------------------------------------------------------------------------- #

def load_us(sp_path: str, nq_path: str) -> pd.DataFrame:
    """FMP の日足2つ（S&P 500・NASDAQ 総合）から、米国の取引日ごとの前日比（%）。列 us_date・sp・nq。"""

    def one(path: str, sym: str) -> pd.Series:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        x = pd.DataFrame(rows)
        miss = sorted({"symbol", "date", "price"} - set(x.columns))
        if miss or not len(x):
            raise SystemExit(f"{os.path.basename(path)}: 列がありません {miss}")
        if set(x["symbol"]) != {sym}:
            raise SystemExit(f"{os.path.basename(path)}: {sym} 以外の行があります")
        d = pd.to_datetime(x["date"])
        if d.duplicated().any():
            raise SystemExit(f"{os.path.basename(path)}: 日付の重複 {int(d.duplicated().sum())}件")
        px = pd.to_numeric(x["price"], errors="coerce")
        if not (px > 0).all():
            raise SystemExit(f"{os.path.basename(path)}: 終値が正でない行 {int((~(px > 0)).sum())}件")
        return pd.Series(px.to_numpy(dtype=float), index=d.to_numpy()).sort_index()

    sp, nq = one(sp_path, "^GSPC"), one(nq_path, "^IXIC")
    if not sp.index.equals(nq.index):
        raise SystemExit(f"S&P 500 と NASDAQ の日付が合いません（片方だけ {len(sp.index.symmetric_difference(nq.index))}日）")
    out = pd.DataFrame({"us_date": sp.index, "sp": sp.pct_change().to_numpy() * 100,
                        "nq": nq.pct_change().to_numpy() * 100}).iloc[1:]
    return out.reset_index(drop=True)


def next_day(dates, bar_days) -> np.ndarray:
    """選定日の次の営業日（bar_days の中で dates より後の最初の日。無ければ NaT）。"""
    bd = np.sort(np.asarray(pd.to_datetime(pd.Index(bar_days)).values, dtype="datetime64[ns]"))
    d = np.asarray(pd.to_datetime(pd.Series(dates)).values, dtype="datetime64[ns]")
    i = np.searchsorted(bd, d, side="right")
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (i < len(bd)) & ~np.isnat(d)
    out[ok] = bd[i[ok]]
    return out


def overnight(sig, buy, us: pd.DataFrame) -> pd.DataFrame:
    """
    買う日 B ごとの前夜の米国。D ≤ U ≤ B−1日 の米国の取引日のうち最後の日 U と、その前日比（%）。
    列: us_date（無ければ NaT）, sp, nq（U の前日比）, n_us（その間の米国の取引日の数）, sp_cum, nq_cum（その間の累積 %）。
    """
    ud = np.asarray(us["us_date"].values, dtype="datetime64[ns]")
    if len(ud) > 1 and not (np.diff(ud) > np.timedelta64(0, "ns")).all():
        raise ValueError("米国の日付が昇順でない")
    sp = us["sp"].to_numpy(dtype=float)
    nq = us["nq"].to_numpy(dtype=float)
    csp = np.concatenate([[0.0], np.cumsum(np.log1p(sp / 100))])
    cnq = np.concatenate([[0.0], np.cumsum(np.log1p(nq / 100))])
    d = np.asarray(pd.to_datetime(pd.Series(sig)).values, dtype="datetime64[ns]")
    b = np.asarray(pd.to_datetime(pd.Series(buy)).values, dtype="datetime64[ns]")
    ok = ~np.isnat(d) & ~np.isnat(b)
    lo = np.searchsorted(ud, d, side="left")
    hi = np.searchsorted(ud, b - np.timedelta64(1, "D"), side="right")
    hi = np.where(ok, np.maximum(hi, lo), lo)
    n = hi - lo
    has = n > 0
    j = np.clip(hi - 1, 0, max(len(ud) - 1, 0))
    nan = np.nan
    out = pd.DataFrame({
        "us_date": np.where(has, ud[j] if len(ud) else np.datetime64("NaT"), np.datetime64("NaT")),
        "sp": np.where(has, sp[j] if len(ud) else nan, nan),
        "nq": np.where(has, nq[j] if len(ud) else nan, nan),
        "n_us": n.astype(int),
        "sp_cum": np.where(has, (np.exp(csp[hi] - csp[lo]) - 1) * 100, nan),
        "nq_cum": np.where(has, (np.exp(cnq[hi] - cnq[lo]) - 1) * 100, nan),
    })
    return out


def attach_us(ft: pd.DataFrame, bar_days, us: pd.DataFrame) -> pd.DataFrame:
    """ft に買う日 buy（市場の次の営業日）と前夜の米国の列を足す。"""
    out = ft.copy()
    out["buy"] = next_day(out["Date"], bar_days)
    ov = overnight(out["Date"], out["buy"], us)
    for c in ov.columns:
        out[c] = ov[c].to_numpy()
    return out


def flag(sp, nq, sp_line: float | None = SP_LINE, nq_line: float | None = NQ_LINE) -> np.ndarray:
    """前日比（%）を小数2桁に丸めて、sp ≤ sp_line または nq ≤ nq_line。NaN（米国の休場）は印なし。"""
    sp = np.round(np.asarray(sp, dtype=float), 2)
    nq = np.round(np.asarray(nq, dtype=float), 2)
    f = np.zeros(len(sp), dtype=bool)
    if sp_line is not None:
        f |= np.nan_to_num(sp, nan=np.inf) <= sp_line
    if nq_line is not None:
        f |= np.nan_to_num(nq, nan=np.inf) <= nq_line
    return f


# --------------------------------------------------------------------------- #
# 日単位の並べ替え検定
# --------------------------------------------------------------------------- #

def day_perm(v, f, day, strata, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """
    D = 平均(v[f]) − 平均(v[~f])（全部を足した平均）。層（テスト窓）ごとに、日の印を日の間で並べ替え（印の日数はそのまま）、
    行は日の印を受け継ぐ。p_less = 並べ替えた D が実際以下になる割合（片側。「印の日が悪い」）、p_two = 絶対値で実際以上。
    同じ層・同じ日で印が食い違う行があれば ValueError。
    """
    x = pd.DataFrame({"v": np.asarray(v, dtype=float), "f": np.asarray(f, dtype=bool),
                      "d": pd.to_datetime(pd.Series(day)).to_numpy(), "s": np.asarray(strata)})
    x = x[np.isfinite(x["v"].to_numpy()) & x["d"].notna().to_numpy()].reset_index(drop=True)   # 収益・買う日の無い行は外す
    nan = float("nan")
    gd = x.groupby(["s", "d"], sort=True)["f"]
    mixed = int((gd.nunique() > 1).sum())
    if mixed:
        raise ValueError(f"同じ日で印が食い違う: {mixed}日")
    days = gd.first().reset_index()
    base = {"n_yes": int(x["f"].sum()), "n_no": int((~x["f"]).sum()),
            "days_yes": int(days["f"].sum()), "days_no": int((~days["f"]).sum())}
    if base["n_yes"] == 0 or base["n_no"] == 0:
        return {"D": nan, "p_less": nan, "p_two": nan, **base, "strata_used": 0}
    did = x.groupby(["s", "d"], sort=True).ngroup().to_numpy()
    v_ = x["v"].to_numpy()
    fd = days["f"].to_numpy(dtype=bool)
    sd = days["s"].to_numpy()
    obs = float(v_[fd[did]].mean() - v_[~fd[did]].mean())
    groups = [np.where(sd == u)[0] for u in pd.unique(sd)]
    groups = [g for g in groups if 0 < fd[g].sum() < len(g)]          # 両側の日がある層だけ並べ替える
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        fp = fd.copy()
        for g in groups:
            fp[g] = fd[g][rng.permutation(len(g))]
        fr = fp[did]
        null[k] = v_[fr].mean() - v_[~fr].mean() if fr.any() and (~fr).any() else 0.0
    return {"D": obs, "p_less": float((np.sum(null <= obs) + 1) / (n_perm + 1)),
            "p_two": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)), **base, "strata_used": len(groups)}


def fmt_p(t: dict) -> str:
    if not np.isfinite(t.get("D", np.nan)):
        return "—（片側が0件）"
    return (f"D {t['D']:+.2f}pt（{t['n_yes']}件・{t['days_yes']}日 対 {t['n_no']}件・{t['days_no']}日）・"
            f"片側 p = {t['p_less']:.3f}・両側 p = {t['p_two']:.3f}")


def band_mask(x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    x = np.round(np.asarray(x, dtype=float), 2)
    m = np.isfinite(x)
    if lo is not None:
        m &= x > lo
    if hi is not None:
        m &= x <= hi
    return m


def band_name(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return f"{hi:+.1f}% 以下"
    if hi is None:
        return f"{lo:+.1f}% 超"
    return f"{lo:+.1f}〜{hi:+.1f}%"


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def trade(ft: pd.DataFrame, bar_days, months: int) -> dict:
    """画面の3枠の形（今の規則・上位2件・同時に3件・+10%・発火を問わない）の取引の成績。"""
    t = E87.run_one(ft, bar_days, "K15", 95.0, 1, LT.TOP_K, LT.SLOTS, 0, 10.0)
    s = stats(t)
    r = pd.to_numeric(t["tp10c"], errors="coerce").dropna()
    s["capital_pm"] = float(r.sum() / LT.SLOTS / months) if months else float("nan")
    return s


def report(ft: pd.DataFrame, bar_days, us: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """ft は attach_us 済み（buy・us_date・sp・nq・n_us・sp_cum・nq_cum の列）。gap の列があれば寄りの値ざやも出す。"""
    miss = [c for c in US_COLS if c not in ft.columns]
    if miss:
        raise ValueError(f"attach_us の列がありません: {miss}")
    summary: dict = {}
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    k = E88.rule_mask(ft, "min_K15", 95.0)
    f_all = flag(ft["sp"], ft["nq"])
    us_f = flag(us["sp"], us["nq"])
    in_span = (us["us_date"] >= pd.to_datetime(ft["Date"]).min()) & (us["us_date"] <= pd.to_datetime(ft["buy"]).max())
    print(f"\n■ 0. データ: 米国の日足 {len(us):,}日（{us['us_date'].min().date()}〜{us['us_date'].max().date()}）。"
          f"目安に当たる米国の日 {int(us_f.sum())}日（{us_f.mean() * 100:.1f}%）・前向きの期間では "
          f"{int(us_f[in_span.to_numpy()].sum())}/{int(in_span.sum())}日")
    nus = ft["n_us"].to_numpy()
    print(f"  前向き {len(ft):,}行・{months}か月。今の規則の行 {int(k.sum())}件。前夜の米国が無い行（米国の休場）"
          f"{int((nus == 0).sum())}件・間に米国の日が2日以上（東証の連休）{int((nus >= 2).sum())}件・買う日が無い行 "
          f"{int(pd.isna(ft['buy']).sum())}件")
    summary["data"] = {"us_days": int(len(us)), "us_flag_days": int(us_f.sum()), "rows": int(len(ft)),
                       "rule_rows": int(k.sum()), "no_us": int((nus == 0).sum()), "multi_us": int((nus >= 2).sum())}

    # ---- 1. 判定
    g = ft[k]
    fg = f_all[k]
    v = pd.to_numeric(g["tp10c"], errors="coerce").to_numpy(dtype=float)
    main = day_perm(v, fg, g["buy"], g["window"].to_numpy(), n_perm, seed)
    support = bool(np.isfinite(main["D"]) and main["D"] < 0 and main["p_less"] < 0.05)
    print(f"\n■ 1. 判定（今の規則・前向き）: 前夜の米国が目安（NASDAQ ≤ {NQ_LINE:+.1f}% か S&P ≤ {SP_LINE:+.1f}%）の日 − それ以外")
    s_yes, s_no = stats(g[fg]), stats(g[~fg])
    print(line("目安に当たる日の買い", s_yes))
    print(line("それ以外の日の買い", s_no))
    print(f"  日単位の並べ替え（窓ごと）: {fmt_p(main)}")
    print(f"  判定: {'見送りを支える' if support else '見送りを支えない'}（D < 0 かつ 片側 p < 0.05 で支える）")
    summary["main"] = {"yes": s_yes, "no": s_no, "test": main, "support": support}

    # ---- 2. 見送った場合
    s_all = stats(g)
    kept = stats(g[~fg])
    r_yes = v[fg & np.isfinite(v)]
    print("\n■ 2. 目安の日だけ買わなかった場合（今の規則・1取引の平均）")
    print(line("全部買う", s_all))
    print(line("目安の日は買わない", kept))
    print(f"  買わなかった取引 {len(r_yes)}件・その合計 {r_yes.sum():+.1f}pt（1取引 {r_yes.mean() if len(r_yes) else np.nan:+.2f}%）。"
          f"平均の差 {kept['mean'] - s_all['mean']:+.2f}pt")
    summary["skip"] = {"all": s_all, "kept": kept, "dropped_n": int(len(r_yes)), "dropped_sum": float(r_yes.sum())}

    # ---- 3. 帯
    print("\n■ 3. 前夜の S&P 500 の前日比の帯ごと（今の規則・参考）")
    bands = {}
    for lo, hi in SP_BANDS:
        m = band_mask(g["sp"], lo, hi)
        s = stats(g[m])
        bands[band_name(lo, hi)] = s
        print(line(f"S&P {band_name(lo, hi)}", s))
    s = stats(g[g["n_us"].to_numpy() == 0])
    bands["米国の休場"] = s
    print(line("米国の休場（前夜なし）", s) if s["n"] else "  米国の休場（前夜なし）       0件")
    summary["bands"] = bands

    # ---- 4. 別の目安
    print("\n■ 4. 別の目安（今の規則・参考）")
    alt = {}
    for name, sl, nl in ALT:
        fa = flag(g["sp"], g["nq"], sl, nl)
        t = day_perm(v, fa, g["buy"], g["window"].to_numpy(), n_perm, seed)
        alt[name] = {"yes": stats(g[fa]), "test": t}
        print(line(f"{name}", stats(g[fa])))
        print(f"    {fmt_p(t)}")
    summary["alt"] = alt

    # ---- 5. 候補全体
    va = pd.to_numeric(ft["tp10c"], errors="coerce").to_numpy(dtype=float)
    full = day_perm(va, f_all, ft["buy"], ft["window"].to_numpy(), n_perm, seed)
    print("\n■ 5. 候補全体（規則を問わない前向きの全行・参考）")
    print(line("目安に当たる日", stats(ft[f_all])))
    print(line("それ以外の日", stats(ft[~f_all])))
    print(f"  日単位の並べ替え（窓ごと）: {fmt_p(full)}")
    summary["full"] = {"yes": stats(ft[f_all]), "no": stats(ft[~f_all]), "test": full}

    # ---- 6. 寄りの値ざや
    if "gap" in ft.columns:
        gp = pd.to_numeric(g["gap"], errors="coerce").to_numpy(dtype=float)
        gy, gn = gp[fg & np.isfinite(gp)], gp[~fg & np.isfinite(gp)]
        print("\n■ 6. 寄りの値ざや（B の寄り ÷ D の終値 − 1。今の規則・参考）")
        print(f"  目安に当たる日 {len(gy)}件 平均 {gy.mean() if len(gy) else np.nan:+.2f}%（中央 "
              f"{np.median(gy) if len(gy) else np.nan:+.2f}）・それ以外 {len(gn)}件 平均 "
              f"{gn.mean() if len(gn) else np.nan:+.2f}%（中央 {np.median(gn) if len(gn) else np.nan:+.2f}）")
        summary["gap"] = {"yes_n": int(len(gy)), "yes_mean": float(gy.mean()) if len(gy) else None,
                          "no_n": int(len(gn)), "no_mean": float(gn.mean()) if len(gn) else None}

    # ---- 7. 窓ごと
    print("\n■ 7. テスト窓ごとの差（今の規則・目安の日 − それ以外・参考）")
    per_w = {}
    for w, x in g.groupby("window"):
        xf = flag(x["sp"], x["nq"])
        r = pd.to_numeric(x["tp10c"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(r)
        if (xf & ok).any() and (~xf & ok).any():
            per_w[int(w)] = {"n_yes": int((xf & ok).sum()), "n_no": int((~xf & ok).sum()),
                             "diff": float(r[xf & ok].mean() - r[~xf & ok].mean())}
    pos = sum(1 for d in per_w.values() if d["diff"] > 0)
    print("  " + " / ".join(f"窓{w} {d['diff']:+.2f}pt（{d['n_yes']}件）" for w, d in per_w.items()))
    print(f"  目安の日のほうが良い窓 {pos}/{len(per_w)}")
    summary["per_window"] = per_w

    # ---- 8. 画面の3枠の形
    if bar_days is not None:
        t_all = trade(ft, bar_days, months)
        t_skip = trade(ft[~f_all], bar_days, months)
        print("\n■ 8. 画面の3枠の形（今の規則・上位2件・同時に3件・+10%・参考）")
        print(line("全部買う", t_all) + f"  資金全体の月 {t_all['capital_pm']:+.2f}%")
        print(line("目安の日は買わない", t_skip) + f"  資金全体の月 {t_skip['capital_pm']:+.2f}%")
        summary["trade"] = {"all": t_all, "skip": t_skip}

    print(f"\n■ まとめ: {'見送りを支える' if support else '見送りを支えない'}。目安の日の買い {s_yes['n']}件 "
          f"{s_yes['mean']:+.2f}%・それ以外 {s_no['n']}件 {s_no['mean']:+.2f}%（D {main['D']:+.2f}pt・片側 p = "
          f"{main['p_less']:.3f}）")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験110: 前夜の米国の下落と今の規則の買いの成績")
    ap.add_argument("--us-sp", required=True, help="S&P 500（^GSPC）の日足 JSON")
    ap.add_argument("--us-nq", required=True, help="NASDAQ 総合（^IXIC）の日足 JSON")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)
    t0 = time.time()
    us = load_us(args.us_sp, args.us_nq)
    log(f"米国の日足 {len(us):,}日（{us['us_date'].min().date()}〜{us['us_date'].max().date()}）")
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
    ft = attach_us(ft, bar_days, us)
    k = E88.rule_mask(ft, "min_K15", 95.0)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。今の規則の行 {int(k.sum())}件（実験88 は 559件）")
    # 確かめ: 規則の行の買う日（live_track の buy_date = その銘柄の次の日足）と市場の次の営業日、寄りの値ざや
    fw = LT.forward(bars, ft.loc[k, ["Code", "Date"]], target=10.0)[["Code", "Date", "buy_date", "entry"]]
    ft = ft.merge(fw, on=["Code", "Date"], how="left")
    cl = bars[["Code", "Date", "AdjC"]].rename(columns={"AdjC": "_c0"})
    ft = ft.merge(cl, on=["Code", "Date"], how="left")
    ft["gap"] = (pd.to_numeric(ft["entry"], errors="coerce") / pd.to_numeric(ft["_c0"], errors="coerce") - 1) * 100
    kk = E88.rule_mask(ft, "min_K15", 95.0)
    bd = pd.to_datetime(ft.loc[kk, "buy_date"])
    diff = int((bd.notna() & (bd.to_numpy() != pd.to_datetime(ft.loc[kk, "buy"]).to_numpy())).sum())
    log(f"規則の行の買う日: 銘柄の次の日足と市場の次の営業日が違う行 {diff}件・買う日の無い行 {int(bd.isna().sum())}件")
    summary = report(ft.drop(columns=["_c0"]), bar_days, us, n_perm=args.perm)
    summary["exit_check"] = chk
    summary["buy_day_mismatch"] = diff
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e110_summary.json")

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
