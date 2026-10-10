#!/usr/bin/env python3
"""
実験70: 「際どい」候補は、その後どうだったか。日証金で分けると違うか（本番の239列のモデルの OOF で）。

運用者（2026-10-10）: 「特徴量としての追加ではなく、モデルの後工程として日証金や EDINET DB を使い、
買うか見送るかの判断が際どい時に補助的に使う」。際どい範囲は「全部」（3モデルの最小が 90〜95、
2モデルが95以上で残り1つが95未満、画面の惜しい候補 85〜90）。例: xgb 96・cat 98・logit 90以上・lgbm 85。
運用の線は「3モデル（lgbm・xgb・cat）がすべて 95以上」。

何を見るか（先に決めた。結果を見てから分け方を足さない）
--------
1. 形ごとの成績（OOF の窓2以降。窓1は百分位の参照分布が無い）
   百分位は本番と同じ基準: その行より前の窓のスコア分布での位置（実験36 と同じ。丸めは画面と同じ）
   形（SHAPES）: 3つとも95以上 / 2つが95以上で残り1つが 90〜95・85〜90・85未満 /
                 95以上は1つ以下で最小が 90〜95・85〜90 / それ以外
   2つが95以上の形は、残り1つ（最下位）がどのモデルか、logit が90以上か、でも分ける
2. 際どい候補（上の和集合）を日証金で分ける（日証金は2023年10月からなので、それ以降の行だけ）
   向きは実験66 で決めたもの（実験69 と同じ）:
     融資残高の20日変化（jsf_loan_chg20_v）が上位10% → 良い側
     直近20日の逆日歩の日数（jsf_fee_days20）が上位10%（かつ1日以上）→ 悪い側
   上位10% の線は、その月より前の母集団（日証金の付いた行）の分布で引く（先の情報を使わない）
物差し: +10% の指値（20営業日以内に高値が買値の +10% に届けば +10%、届かなければ ret_o1_20。実験32 と同じ）と
持ち切り（ret_o1_20）。勝率・+10%到達率・−10%割れ。SE は銘柄ごとにまとめて計算する（同じ銘柄が続けて
候補に出るので、行を独立に数えると SE が小さく出る）。

注意: 日証金で分けた側は行が少なく（際どい候補 × 2023年10月以降）、差が出なくても「効かない」とは言えない。
悪い方向に効いていないかを見るのが目的。公開ログには集計だけを出す（銘柄名・銘柄ごとの値は出さない）。

  exp=e70_borderline.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import walkforward as WF  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

TREES = ("lgbm", "xgb", "cat")
LINE = 95.0            # 運用の線（3つとも これ以上で買う）
NEAR_LO = 85.0         # 際どい範囲の下（画面の惜しい候補の下限）
MIN_REF = 500          # 百分位の参照分布に要る行（実験36 と同じ）
JSF_MIN_REF = 300      # 日証金の線を引くのに要る、その月より前の行
JSF_TOP = 90.0         # 日証金の上位10%
OUT = os.path.join(lab.DATA_DIR, "oof", "e70_borderline.json")

#: 形の名前と表示（並び順もこのとおり）
SHAPES = [
    ("all95", "3つとも95以上（線の上）"),
    ("two95_hi", "2つ95以上・残り 90〜95"),
    ("two95_mid", "2つ95以上・残り 85〜90"),
    ("two95_lo", "2つ95以上・残り 85未満"),
    ("min90", "95以上は1つ以下・最小 90〜95"),
    ("min85", "95以上は1つ以下・最小 85〜90"),
    ("other", "それ以外"),
]
BORDER = ("two95_hi", "two95_mid", "two95_lo", "min90", "min85")


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- #
# 百分位と形
# ---------------------------------------------------------------------- #

def assign_folds(dates: pd.Series, ref_dates: pd.Series) -> np.ndarray:
    """本番の OOF と同じ作りの窓（データセットの日付から作る）を各行に当てる。どの窓にも無ければ 0。"""
    folds = WF.make_folds(pd.to_datetime(ref_dates), min_train_months=OOF_MIN_TRAIN_MONTHS,
                          test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                          embargo_days=B.RISE_HORIZON)
    d = pd.to_datetime(dates).to_numpy()
    out = np.zeros(len(d), dtype=int)
    for f in folds:
        out[(d >= np.datetime64(f.test_start)) & (d <= np.datetime64(f.test_end))] = f.index
    return out


def hist_pct(d: pd.DataFrame, algos: Sequence[str], min_ref: int = MIN_REF) -> pd.DataFrame:
    """
    本番と同じ百分位 hp_<algo>: その行の窓より前の窓（fold が小さい行）のスコア分布での位置。
    丸めは画面と同じ（live_track.pct_of）。参照が min_ref 行に満たない窓は NaN。
    """
    d = d.copy()
    for a in algos:
        out = np.full(len(d), np.nan)
        for f in sorted(x for x in d["fold"].unique() if x > 0):
            prev = d.loc[(d["fold"] > 0) & (d["fold"] < f), f"s_{a}"].to_numpy(dtype=float)
            prev = prev[np.isfinite(prev)]
            if len(prev) < min_ref:
                continue
            m = (d["fold"] == f).to_numpy()
            out[m] = LT.pct_of(d.loc[m, f"s_{a}"].to_numpy(dtype=float), prev)
        d[f"hp_{a}"] = out
    return d


def shape_of(p: pd.DataFrame, line: float = LINE, lo: float = NEAR_LO) -> pd.Series:
    """3モデル（TREES）の百分位 hp_* から形（SHAPES のキー）。1つでも NaN なら NaN。"""
    v = p[[f"hp_{a}" for a in TREES]].to_numpy(dtype=float)
    ok = np.isfinite(v).all(axis=1)
    n95 = (v >= line).sum(axis=1)
    mn = np.where(ok, np.nanmin(np.where(np.isfinite(v), v, np.inf), axis=1), np.nan)
    out = np.where(n95 == 3, "all95",
          np.where(n95 == 2, np.where(mn >= 90, "two95_hi", np.where(mn >= lo, "two95_mid", "two95_lo")),
          np.where(mn >= 90, "min90", np.where(mn >= lo, "min85", "other"))))
    return pd.Series(np.where(ok, out, None), index=p.index, dtype=object)


def laggard_of(p: pd.DataFrame) -> pd.Series:
    """3モデルのうち百分位がいちばん低いモデル（同点は TREES の先）。1つでも NaN なら None。"""
    v = p[[f"hp_{a}" for a in TREES]].to_numpy(dtype=float)
    ok = np.isfinite(v).all(axis=1)
    idx = np.argmin(np.where(np.isfinite(v), v, np.inf), axis=1)
    return pd.Series([TREES[i] if k else None for i, k in zip(idx, ok)], index=p.index, dtype=object)


# ---------------------------------------------------------------------- #
# 集計
# ---------------------------------------------------------------------- #

def cluster_se(x, groups) -> float:
    """平均の SE を、同じ groups（銘柄）の行をまとめて計算する（CR1: G/(G−1) を掛ける）。"""
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 2:
        return np.nan
    s = pd.Series(x - x.mean()).groupby(g).sum().to_numpy()
    G = len(s)
    if G < 2:
        return np.nan
    return float(np.sqrt((s ** 2).sum() * G / (G - 1)) / n)


def summarize(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"n": 0}
    return {"n": n, "codes": int(g["Code"].nunique()),
            "tp10": float(g["tp10"].mean()), "tp10_se": cluster_se(g["tp10"], g["Code"]),
            "hold": float(g["r"].mean()), "hold_se": cluster_se(g["r"], g["Code"]),
            "win": float((g["tp10"] > 0).mean() * 100), "hit10": float(g["hit10"].mean() * 100),
            "lose10": float((g["r"] < -10).mean() * 100), "label": float(g["label"].mean() * 100)}


def diff(a: dict, b: dict, key: str = "tp10") -> dict:
    """a − b と、その SE（両群を独立とみなす）・z。"""
    if not a.get("n") or not b.get("n"):
        return {"d": np.nan, "se": np.nan, "z": np.nan}
    dv = a[key] - b[key]
    se = float(np.sqrt(a[f"{key}_se"] ** 2 + b[f"{key}_se"] ** 2))
    return {"d": dv, "se": se, "z": dv / se if se > 0 else np.nan}


HEAD = (f"  {'':<30}{'件数':>6}{'銘柄':>6}{'+10%指値':>9}{'SE':>6}{'持ち切り':>9}{'SE':>6}"
        f"{'勝率':>6}{'+10%到達':>8}{'−10%割れ':>8}{'正例率':>7}")


def line(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {name:<30}{0:>6}"
    if s["n"] < 10:
        return f"  {name:<30}{s['n']:>6}{s['codes']:>6}   （件数が少なすぎる）"
    return (f"  {name:<30}{s['n']:>6}{s['codes']:>6}{s['tp10']:>+8.2f}%{s['tp10_se']:>6.2f}"
            f"{s['hold']:>+8.2f}%{s['hold_se']:>6.2f}{s['win']:>5.0f}%{s['hit10']:>7.0f}%"
            f"{s['lose10']:>7.1f}%{s['label']:>6.0f}%")


# ---------------------------------------------------------------------- #
# 日証金
# ---------------------------------------------------------------------- #

def past_threshold(dates: pd.Series, x: pd.Series, q: float = JSF_TOP,
                   min_rows: int = JSF_MIN_REF) -> pd.Series:
    """各行に「その月より前の行の x の q 分位」を付ける。前の行が min_rows に満たない月は NaN。"""
    month = pd.to_datetime(dates).dt.to_period("M")
    xv = pd.to_numeric(x, errors="coerce")
    ok = np.isfinite(xv)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for m in sorted(month.unique()):
        prev = xv[ok & (month < m)]
        if len(prev) >= min_rows:
            out[month == m] = float(np.percentile(prev, q))
    return out


def jsf_flags(d: pd.DataFrame) -> pd.DataFrame:
    """
    jsf_good: 融資残高の20日変化が、その月より前の母集団の上位10% 以上（かつ増加）
    jsf_bad:  逆日歩の日数が、その月より前の母集団の上位10% 以上（かつ1日以上）
    線が引けない行（日証金が無い・前の行が足りない）は jsf_ok = False。
    """
    d = d.copy()
    cov = d["jsf_ratio"].notna()
    t_loan = past_threshold(d.loc[cov, "Date"], d.loc[cov, "jsf_loan_chg20_v"])
    t_fee = past_threshold(d.loc[cov, "Date"], d.loc[cov, "jsf_fee_days20"])
    d["t_loan"] = t_loan.reindex(d.index)
    d["t_fee"] = t_fee.reindex(d.index)
    loan = pd.to_numeric(d["jsf_loan_chg20_v"], errors="coerce")
    fee = pd.to_numeric(d["jsf_fee_days20"], errors="coerce")
    d["jsf_ok"] = cov & d["t_loan"].notna() & d["t_fee"].notna() & loan.notna() & fee.notna()
    d["jsf_good"] = d["jsf_ok"] & (loan >= d["t_loan"]) & (loan > 0)
    d["jsf_bad"] = d["jsf_ok"] & (fee >= np.maximum(d["t_fee"], 1.0))
    return d


# ---------------------------------------------------------------------- #

def load() -> tuple:
    """本番の OOF（木3つ + あれば logit）と、結果・日証金を付けた表。"""
    from e32_takeprofit import attach, forward_paths
    import jsf_features as JF

    algos = list(TREES)
    has_logit = LT.find_oof("logit", lab.DATA_DIR, os.path.join(os.path.dirname(HERE), "model")) is not None
    if has_logit:
        algos.append("logit")
    base = None
    for a in algos:
        path = LT.find_oof(a, lab.DATA_DIR, os.path.join(os.path.dirname(HERE), "model"))
        if path is None:
            raise SystemExit(f"{a} の OOF が無い（Release data-raw の oof.parquet / {a}_oof.parquet）")
        o = pd.read_parquet(path)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        keep = ["Code", "Date", "score"] + (["label", "ret_o1_20"] if a == "lgbm" else [])
        o = o[keep].rename(columns={"score": f"s_{a}"})
        base = o if base is None else base.merge(o, on=["Code", "Date"], how="inner")
    frame = lab.frame()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Code"] = frame["Code"].astype(str)
    base["fold"] = assign_folds(base["Date"], frame["Date"])
    base = hist_pct(base, algos)
    base["n_break"] = base.groupby("Date")["Code"].transform("size")
    base["r"] = pd.to_numeric(base["ret_o1_20"], errors="coerce") * 100
    d = attach(base, forward_paths())
    d = d[d["entry"].notna() & d["r"].notna()].copy()
    jsf = JF.build(frame[["Code", "Date"]].copy(), lab.DATA_DIR)
    cols = ["jsf_ratio", "jsf_loan_chg20_v", "jsf_fee_days20"]
    d = d.merge(jsf[["Code", "Date"] + cols], on=["Code", "Date"], how="left")
    return d.reset_index(drop=True), has_logit


def main(argv=None) -> int:
    d, has_logit = load()
    d["shape"] = shape_of(d)
    d["laggard"] = laggard_of(d)
    d = d[d["shape"].notna()].copy()
    log(f"対象 {len(d):,}行（OOF の窓2以降）/ {d['Date'].min().date()}〜{d['Date'].max().date()} / "
        f"logit の OOF {'あり' if has_logit else 'なし'}")
    res: Dict[str, object] = {"rows": int(len(d)), "from": str(d["Date"].min().date()),
                              "to": str(d["Date"].max().date()), "has_logit": has_logit}

    # 1. 形ごと
    for title, sub, key in (("全日", d, "shape_all"), ("発火8件以上の日", d[d["n_break"] >= 8], "shape_8")):
        print(f"\n=== 1. 形ごとの成績（{title}。百分位は前の窓の分布）===")
        print(HEAD)
        res[key] = {}
        for k, nm in SHAPES:
            s = summarize(sub[sub["shape"] == k])
            res[key][k] = s
            print(line(nm, s))
        s = summarize(sub[sub["shape"].isin(BORDER)])
        res[key]["border"] = s
        print(line("（際どい候補の合計）", s))
        print(line("（母集団）", summarize(sub)))

    # 1b. 2つ95以上の形を、最下位のモデルと logit で分ける
    print("\n=== 1b. 2つが95以上の形: 最下位のモデル・logit で分ける（全日）===")
    print(HEAD)
    two = d[d["shape"].isin(("two95_hi", "two95_mid", "two95_lo"))]
    res["two95"] = {}
    for k, nm in (("two95_hi", "残り 90〜95"), ("two95_mid", "残り 85〜90"), ("two95_lo", "残り 85未満")):
        g = two[two["shape"] == k]
        for a in TREES:
            s = summarize(g[g["laggard"] == a])
            res["two95"][f"{k}_{a}"] = s
            print(line(f"{nm}・最下位 {a}", s))
    if has_logit:
        print()
        for nm, m in (("2つ95以上・logit 90以上", two["hp_logit"] >= 90),
                      ("2つ95以上・logit 90未満", two["hp_logit"] < 90),
                      ("最下位 lgbm・logit 90以上", (two["laggard"] == "lgbm") & (two["hp_logit"] >= 90)),
                      ("最下位 lgbm・logit 90未満", (two["laggard"] == "lgbm") & (two["hp_logit"] < 90))):
            s = summarize(two[m])
            res["two95"][nm] = s
            print(line(nm, s))

    # 1c. 年ごとの件数（運用で際どい候補がどれくらい出るか）
    print("\n=== 1c. 年ごとの件数（行 / 日）===")
    d["year"] = d["Date"].dt.year
    res["per_year"] = {}
    for y, g in d.groupby("year"):
        b = g[g["shape"].isin(BORDER)]
        a95 = g[g["shape"] == "all95"]
        res["per_year"][int(y)] = {"border_rows": int(len(b)), "border_days": int(b["Date"].nunique()),
                                   "all95_rows": int(len(a95)), "days": int(g["Date"].nunique())}
        print(f"  {y}: 際どい {len(b):>4}行・{b['Date'].nunique():>3}日 / 3つとも95以上 {len(a95):>4}行 "
              f"/ 候補のあった日 {g['Date'].nunique():>3}")

    # 2. 際どい候補を日証金で分ける
    j = jsf_flags(d)
    jb = j[j["shape"].isin(BORDER) & j["jsf_ok"]]
    print(f"\n=== 2. 際どい候補を日証金で分ける（線を引けた行 {len(jb):,}・"
          f"{jb['Date'].min().date() if len(jb) else '-'}〜。向きは実験66 で決めたもの）===")
    print(HEAD)
    # (記録の鍵, 表示, 行の選び方)。鍵は重ならないようにする
    groups = [
        ("good", "融資の20日変化 上位10%（良い側）", jb["jsf_good"]),
        ("not_good", "  融資 上位10% 以外", ~jb["jsf_good"]),
        ("bad", "逆日歩の日数 上位10%（悪い側）", jb["jsf_bad"]),
        ("not_bad", "  逆日歩 上位10% 以外", ~jb["jsf_bad"]),
        ("good_only", "良い側だけ", jb["jsf_good"] & ~jb["jsf_bad"]),
        ("neither", "どちらでもない", ~jb["jsf_good"] & ~jb["jsf_bad"]),
    ]
    res["jsf"] = {}
    for key_, nm, m in groups:
        s = summarize(jb[m])
        res["jsf"][key_] = s
        print(line(nm, s))
    for nm, (a, b) in (("良い側 − それ以外", ("good", "not_good")),
                       ("悪い側 − それ以外", ("bad", "not_bad"))):
        for key in ("tp10", "hold"):
            dd = diff(res["jsf"][a], res["jsf"][b], key)
            res["jsf"][f"{nm}_{key}"] = dd
            print(f"  {nm}（{'+10%指値' if key == 'tp10' else '持ち切り'}）: {dd['d']:+.2f}pt ± {dd['se']:.2f}"
                  f"（z {dd['z']:+.2f}）")
    print("  ※ 行が少ないので、差が出なくても「効かない」とは言えない。向きが決めたとおりかを見る")

    # 参考: 線の上（3つとも95以上）と母集団でも同じ分け方（日証金が効くなら、ここでも同じ向きのはず）
    print("\n  参考: 同じ分け方を 3つとも95以上 と 母集団 で")
    print(HEAD)
    for nm, sub in (("3つとも95以上", j[(j["shape"] == "all95") & j["jsf_ok"]]), ("母集団", j[j["jsf_ok"]])):
        for lab_, m in (("良い側", sub["jsf_good"]), ("悪い側", sub["jsf_bad"]),
                        ("どちらでもない", ~sub["jsf_good"] & ~sub["jsf_bad"])):
            s = summarize(sub[m])
            res["jsf"][f"{nm}_{lab_}"] = s
            print(line(f"{nm}・{lab_}", s))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda x: None if x is None else float(x))
    log(f"記録: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
