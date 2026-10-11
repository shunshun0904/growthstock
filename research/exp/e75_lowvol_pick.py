#!/usr/bin/env python3
"""
実験75: モデルが選んだ候補の中で、日次ボラが低い銘柄を選ぶと +20% 指値の収益は上がるか（Jev を使わない）

経緯（2026-10-11）
------------------
実験73・74 で、Jev の答えは state に渡した 20日リターン・日次ボラの言い換えに近く（Spearman 0.78〜0.92 / 0.44〜0.55）、
モデルが陽性と見た候補（3モデルとも90以上）の中では「答えが高い（荒い・直前に上がった）銘柄ほど正例率と +20% 指値の
収益が低い」向きが出た。同じ集合で正例を分ける ROC-AUC は日次ボラ1列が 0.312（低いほど当たる。反転で 0.69）で、Jev の
どの問い（0.367〜0.489）より強い。
運用者の読み: 「3モデル95以上で買う候補でも、荒い銘柄は逆に振れる。同日に 95 未満でも低い銘柄があればそちらを選ぶ使い方は
あり得る」。→ Jev ではなく、Jev に渡している日次ボラ（vol_20d）そのもので、同日の選び方の規則を全期間の OOF で比べる。
Jev を経由する理由は無い（費用ゼロ・決定的・モデル版に依らない・全期間で測れる）。

運用者の決定（2026-10-11、AskUserQuestion）: 規則は「3モデル90以上の中で日次ボラが最も低い1銘柄」と「3モデル95以上を優先し、
無い日は 90以上の低ボラ1銘柄」。基準はいまの規則（3モデル95以上を全部、3モデル90以上を全部）。主の物差しは +20% 指値の
収益の平均。直近1年は実験74 の控え（Jev の答え）も並べる。API の課金なし。

先に決めること（結果を見てから分け方を足さない）
------------------------------------------------
対象: 本番の OOF（lgbm / xgb / cat。logit があれば百分位の有無の条件にだけ使う）の全期間で、3モデルの百分位（その行より前の
      窓の分布。実験70 と同じ）と +20% の結果（実験32 の値動きの表）が付く行。その日の候補 = 3モデルとも90以上の行（画面の線）。
規則（1日ごとに選ぶ。候補が無い日は買わない）:
  基準 all95     3モデルとも95以上を全部買う（運用者の線）
  基準 all90     3モデルとも90以上を全部買う（画面の線）
  基準 top1      90以上の中で 3モデルの最小が最大の1銘柄（画面の並び順。同点は LightGBM のスコア）= 画面の規則の topK=1
  基準 top2      同じく上位2銘柄（src/lib/strategy.js STRATEGY.topK = 2）
  規則 lowvol1   90以上の中で日次ボラ（vol_20d）が最も低い1銘柄（同点は 3モデルの最小が高いほう）
  規則 pref95    95以上がある日はその中の top1、無い日は 90以上の lowvol1
  参考 lowvol2   90以上の中で日次ボラが低い2銘柄
  参考 lowvol95  95以上がある日はその中の低ボラ1銘柄、無い日は 90以上の lowvol1
  参考 lowjev1   90以上の中で Jev（R: +10% 到達の確率、prod 腕）が最も低い1銘柄。実験74 の控えがある行だけ（直近1年）
物差し: 主 = +20% 指値の収益（tp20。20営業日以内に届けば +20%、届かなければ持ち切り ret_o1_20）の1件あたり平均。
      副 = +10% 指値（tp10）、持ち切り（r）、勝率（tp20 > 0）、+20% 到達率、正例率（+1.2σ）、−10% 割れ（r < −10）の率。
SE:   1件あたり平均の SE は銘柄と日付の2方向でまとめる（CR1。V = V_銘柄 + V_日付 − V_iid。負なら大きいほう）。
      規則同士の差は同じ日の対で取る（d_t = 規則A のその日の選んだ行の tp20 の平均 − 規則B の同じもの。両方が選んだ日だけ。
      同じ銘柄を選んだ日は差 0 として数に入れる。運用上その日は違いが出ない）。SE は月でまとめる（20営業日の保有が重なる
      隣の日の相関を吸う）。違う銘柄を選んだ日だけの差も出す。年ごとの差の符号も出す。
日の形: 発火数（n_break）が STRATEGY.skipBreaks（8）未満の日を見送ったとき（live_track.SKIP_BREAKS）の副表。
集合の中の並べる物差し（実験74 §3 の確かめ。全期間）: 3モデル90以上 / 95以上 / LightGBM 95以上 / 90以上 / 母集団 の中で、
      vol_20d・ret_20d・3モデルの最小・LightGBM の百分位（直近1年は Jev R・C も）の、正例と tp20>0 の ROC-AUC、
      真陽性−偽陽性 の差（E73.tp_fp_gap。SE は銘柄ごと）、vol_20d の3分位ごとの正例率・tp20・−10% 割れ。
採否を検討する条件（全部そろったとき。1つの OOF で決めない。満たしても前向きの記録で確かめてから選定の規則に入れる）:
  (1) lowvol1 または pref95 の tp20 の平均が top1 より高く、同じ日の対の差（月まとめ SE）で z ≥ 2
  (2) その差が年ごとに過半の年で正
  (3) −10% 割れの率が top1 より高くない
  (4) 直近1年の窓で、その規則の tp20 の平均が lowjev1 より低くない（Jev を使う理由が無いことの確認）
記録: 集計だけをログに出す（銘柄ごとの値は出さない）。JSON は research/_data/oof/e75_lowvol_pick.json（キャッシュに乗るだけ）。

  exp=e75_lowvol_pick.py  args="--dry-run"   reuse=e74_jev_questions.py   # 件数と規則ごとの件数だけ
  exp=e75_lowvol_pick.py                     reuse=e74_jev_questions.py   # 全期間（既定）

reuse は run-experiment.yml の入力。実験74 のキャッシュ（Jev の控え e74_jev_answers.parquet と値動きの表）を先に戻す。
無くても動く（§6 を飛ばす）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)
sys.path.insert(0, HERE)
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import e70_borderline as E70  # noqa: E402
import e73_jev_oof as E73  # noqa: E402
import e74_jev_questions as E74  # noqa: E402

log = E70.log
TREES = E70.TREES
LINE_HI = 95.0                 # 運用者の線（3つとも これ以上）
LINE_LO = 90.0                 # 画面の線（3つとも これ以上で候補）
SKIP_BREAKS = LT.SKIP_BREAKS   # 発火数がこれ未満の日は見送り（src/lib/strategy.js STRATEGY.skipBreaks と同じ）
MIN_ROWS = E73.MIN_ROWS
MIN_TERCILE = E73.MIN_TERCILE
METRIC = "tp20"
OUT = os.path.join(lab.DATA_DIR, "oof", "e75_lowvol_pick.json")
JEV_ARM = "prod"
#: 実験74 の控えの列 → この実験の列
JEV_COLS = {"r_rise10": "jev_r", "c_label": "jev_c"}

#: 規則（表に出す順）。先頭の語が 基準 / 規則 / 参考
RULES: Dict[str, str] = {
    "all95": "基準 3モデルとも95以上を全部",
    "all90": "基準 3モデルとも90以上を全部",
    "top1": "基準 90以上の中で3モデルの最小が最大の1銘柄",
    "top2": "基準 同じく上位2銘柄（topK=2）",
    "lowvol1": "規則 90以上の中で日次ボラが最も低い1銘柄",
    "pref95": "規則 95以上がある日はその top1、無い日は lowvol1",
    "lowvol2": "参考 90以上の中で日次ボラが低い2銘柄",
    "lowvol95": "参考 95以上がある日はその中の低ボラ1、無い日は lowvol1",
    "lowjev1": "参考 90以上の中で Jev(+10%到達) が最も低い1銘柄",
}
MAIN_RULES = ("lowvol1", "pref95")
BASE_RULE = "top1"
#: 同じ日の対で比べる組（A − B）
PAIRS = (("lowvol1", "top1"), ("pref95", "top1"), ("lowvol95", "top1"), ("lowvol2", "top2"),
         ("lowvol1", "all90"), ("pref95", "all95"))
JEV_RULES = ("top1", "lowvol1", "lowjev1", "pref95")
JEV_PAIRS = (("lowvol1", "lowjev1"), ("pref95", "lowjev1"), ("lowjev1", "top1"), ("lowvol1", "top1"))
#: 集合の中で並べる物差し
MEASURES = {"vol_20d": "日次ボラ vol_20d（低いほど）", "ret_20d": "20日リターン ret_20d", "hp_min": "3モデルの最小",
            "hp_lgbm": "LightGBM の百分位"}
JEV_MEASURES = {"jev_r": "Jev R +10%到達の確率", "jev_c": "Jev C 正例の確率"}


# ---------------------------------------------------------------------- #
# 部品（データ無しで単体テストできる）
# ---------------------------------------------------------------------- #

def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """hp_min・候補の印（cand90 / cand95）・答え合わせの列・年と月を足す。"""
    hp = [f"hp_{a}" for a in TREES]
    for c in hp + ["vol_20d", "tp20", "tp10", "hit20", "r", "label", "s_lgbm", "n_break"]:
        if c not in d.columns:
            raise SystemExit(f"{c} が無い（E73.load_rows の表を渡す）")
    d = d.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    d["hp_min"] = d[hp].min(axis=1)
    d["cand90"] = (d[hp] >= LINE_LO).all(axis=1)
    d["cand95"] = (d[hp] >= LINE_HI).all(axis=1)
    d["y_c"] = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int) == 1
    d["y_b"] = pd.to_numeric(d["tp20"], errors="coerce") > 0
    d["lose10"] = pd.to_numeric(d["r"], errors="coerce") < -10
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.strftime("%Y-%m")
    return d


def _rank_in_day(d: pd.DataFrame, mask: pd.Series, by: List[str], ascending: List[bool]) -> pd.Series:
    """mask の行を日付ごとに by で並べた順位（1 から）。mask 外は NaN。同点は by の後ろの列、最後は元の並び。"""
    sub = d.loc[mask, ["Date"] + by]
    sub = sub.sort_values(["Date"] + by, ascending=[True] + list(ascending), kind="mergesort", na_position="last")
    r = sub.groupby("Date").cumcount() + 1
    out = pd.Series(np.nan, index=d.index)
    out.loc[r.index] = r.to_numpy(dtype=float)
    return out


def pick(d: pd.DataFrame, rule: str, jev_col: str = "jev_r") -> pd.Series:
    """規則で買う行（bool）。1日ごとに選ぶ。候補が無い日は買わない。"""
    c90, c95 = d["cand90"].astype(bool), d["cand95"].astype(bool)
    has_vol = d["vol_20d"].notna()
    by_score, asc_score = ["hp_min", "s_lgbm"], [False, False]
    by_vol, asc_vol = ["vol_20d", "hp_min"], [True, False]
    if rule == "all95":
        return c95
    if rule == "all90":
        return c90
    if rule in ("top1", "top2"):
        return _rank_in_day(d, c90, by_score, asc_score) <= (1 if rule == "top1" else 2)
    if rule in ("lowvol1", "lowvol2"):
        return _rank_in_day(d, c90 & has_vol, by_vol, asc_vol) <= (1 if rule == "lowvol1" else 2)
    if rule in ("pref95", "lowvol95"):
        day95 = d["Date"].isin(d.loc[c95, "Date"].unique())
        if rule == "pref95":
            a = _rank_in_day(d, c95, by_score, asc_score) <= 1
        else:
            a = _rank_in_day(d, c95 & has_vol, by_vol, asc_vol) <= 1
        b = _rank_in_day(d, c90 & has_vol & ~day95, by_vol, asc_vol) <= 1
        return a | b
    if rule == "lowjev1":
        if jev_col not in d.columns:
            return pd.Series(False, index=d.index)
        return _rank_in_day(d, c90 & d[jev_col].notna(), [jev_col, "hp_min"], [True, False]) <= 1
    raise ValueError(f"知らない規則: {rule}")


def two_way_se(x, g1, g2) -> float:
    """
    平均の SE を、g1（銘柄）と g2（日付）の2方向でまとめる（CR1）。
    V = V_g1 + V_g2 − V_(g1×g2)。負になれば大きいほうを使う。片方しか計算できなければそれを使う。
    """
    x = np.asarray(x, dtype=float)
    g1 = np.asarray(g1).astype(str)
    g2 = np.asarray(g2).astype(str)
    ok = np.isfinite(x)
    x, g1, g2 = x[ok], g1[ok], g2[ok]
    n = len(x)
    if n < 2:
        return np.nan
    e = pd.Series(x - x.mean())

    def var(groups) -> float:
        s = e.groupby(groups).sum().to_numpy()
        G = len(s)
        if G < 2:
            return np.nan
        return float((s ** 2).sum() * G / (G - 1) / n ** 2)

    v1, v2 = var(g1), var(g2)
    v12 = var(np.char.add(np.char.add(g1, "|"), g2))
    if np.isfinite(v1) and np.isfinite(v2):
        v = v1 + v2 - v12 if np.isfinite(v12) else max(v1, v2)
        if not v > 0:
            v = max(v1, v2)
    elif np.isfinite(v1) or np.isfinite(v2):
        v = v1 if np.isfinite(v1) else v2
    else:
        return np.nan
    return float(np.sqrt(v))


def summarize(sel: pd.DataFrame) -> Dict:
    """選んだ行の成績。主の物差しは tp20 の平均（SE は銘柄×日付の2方向。銘柄だけのものも）。"""
    if len(sel) == 0:
        return {"n": 0}
    out = {"n": int(len(sel)), "days": int(sel["Date"].nunique()), "codes": int(sel["Code"].nunique()),
           "tp20": float(sel["tp20"].mean()), "tp20_se": two_way_se(sel["tp20"], sel["Code"], sel["Date"]),
           "tp20_se_code": E70.cluster_se(sel["tp20"], sel["Code"]),
           "tp10": float(sel["tp10"].mean()), "hold": float(sel["r"].mean()),
           "win20": float(sel["y_b"].mean() * 100), "hit20": float(sel["hit20"].astype(bool).mean() * 100),
           "label": float(sel["y_c"].mean() * 100), "lose10": float(sel["lose10"].mean() * 100),
           "vol": float(pd.to_numeric(sel["vol_20d"], errors="coerce").mean()),
           "hp_min": float(sel["hp_min"].mean())}
    return out


def day_values(sel: pd.DataFrame, metric: str = METRIC) -> pd.Series:
    """規則が選んだ行の、日付ごとの平均（1銘柄の規則ならその行の値）。"""
    return sel.groupby("Date")[metric].mean()


def paired_diff(d: pd.DataFrame, pa: pd.Series, pb: pd.Series, metric: str = METRIC) -> Dict:
    """
    同じ日の対で A − B。両方が選んだ日だけ。同じ銘柄（の集合）を選んだ日は差 0 として数に入れる。
    SE は月でまとめる（E70.cluster_se）。違う銘柄を選んだ日だけの差も出す。年ごとの差も。
    """
    a, b = day_values(d[pa], metric), day_values(d[pb], metric)
    days = a.index.intersection(b.index).sort_values()
    if len(days) == 0:
        return {"n": 0}
    diff = (a.loc[days] - b.loc[days]).to_numpy(dtype=float)
    sa = d[pa].groupby("Date")["Code"].apply(frozenset).loc[days]
    sb = d[pb].groupby("Date")["Code"].apply(frozenset).loc[days]
    same = (sa == sb).to_numpy()
    months = pd.Series(days).dt.strftime("%Y-%m").to_numpy()
    years = pd.Series(days).dt.year.to_numpy()
    se = E70.cluster_se(diff, months)
    se_iid = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    dd, dm = diff[~same], months[~same]
    out = {"n": int(len(days)), "d": float(diff.mean()), "se": se,
           "z": float(diff.mean() / se) if np.isfinite(se) and se > 0 else np.nan, "se_iid": se_iid,
           "same": float(same.mean() * 100), "n_diff": int(len(dd)),
           "d_diff": float(dd.mean()) if len(dd) else np.nan,
           "se_diff": E70.cluster_se(dd, dm) if len(dd) > 1 else np.nan, "years": {}}
    for y in sorted(set(years.tolist())):
        m = years == y
        out["years"][int(y)] = {"n": int(m.sum()), "d": float(diff[m].mean())}
    out["years_pos"] = int(sum(1 for v in out["years"].values() if v["d"] > 0))
    out["years_n"] = int(len(out["years"]))
    return out


def terciles(sub: pd.DataFrame, col: str) -> List[Dict]:
    """col の3分位ごとの件数・範囲・正例率・tp20（SE は銘柄ごと）・−10% 割れ・+20% 到達。"""
    g = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
    if len(g) < MIN_TERCILE:
        return []
    try:
        q = pd.qcut(g[col], 3, labels=False, duplicates="drop")
    except ValueError:
        return []
    out = []
    for t in sorted(q.dropna().unique()):
        s = g[q == t]
        out.append({"t": int(t), "lo": float(s[col].min()), "hi": float(s[col].max()), "n": int(len(s)),
                    "label": float(s["y_c"].mean() * 100), "tp20": float(s["tp20"].mean()),
                    "tp20_se": E70.cluster_se(s["tp20"], s["Code"]), "lose10": float(s["lose10"].mean() * 100),
                    "hit20": float(s["hit20"].astype(bool).mean() * 100)})
    return out


def measures_in_sets(d: pd.DataFrame, measures: Dict[str, str], sets: Sequence[Tuple[str, pd.Series]]) -> Dict:
    """集合ごとに、物差しの ROC-AUC（正例 / tp20>0）と 真陽性−偽陽性 の差、vol_20d の3分位。"""
    res: Dict[str, Dict] = {}
    for name, mask in sets:
        sub = d[mask]
        res[name] = {"n": int(len(sub)), "measures": {}, "vol_terciles": terciles(sub, "vol_20d")}
        for col in measures:
            if col not in sub.columns:
                continue
            g = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
            if len(g) < MIN_ROWS:
                continue
            res[name]["measures"][col] = {
                "n": int(len(g)),
                "auc_c": E73.roc_auc(g["y_c"], g[col]), "auc_b": E73.roc_auc(g["y_b"], g[col]),
                "gap_c": E73.tp_fp_gap(g, col, "y_c"), "gap_b": E73.tp_fp_gap(g, col, "y_b")}
    return res


def load_jev(d: pd.DataFrame, path: str = E74.CACHE, arm: str = JEV_ARM) -> Tuple[pd.DataFrame, Dict]:
    """実験74 の控え（prod 腕、qset prec_v1）から Jev の答えを (Code, Date) で付ける。無ければそのまま。"""
    info: Dict[str, object] = {"path": path, "rows": 0}
    for c in JEV_COLS.values():
        d[c] = np.nan
    if not os.path.exists(path):
        log(f"Jev の控えが無い（{path}）。§6 は飛ばす")
        return d, info
    try:
        c = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        log(f"Jev の控えを読めない: {type(exc).__name__}。§6 は飛ばす")
        return d, info
    need = ["arm", "Code", "Date", "qset"] + list(JEV_COLS)
    if any(k not in c.columns for k in need):
        log("Jev の控えの列が足りない。§6 は飛ばす")
        return d, info
    c = c[(c["arm"].astype(str) == arm) & (c["qset"].astype(str) == E74.QSET)].copy()
    c["Date"] = pd.to_datetime(c["Date"])
    c["Code"] = c["Code"].astype(str)
    c = c.drop_duplicates(["Code", "Date"], keep="first")
    m = c[["Code", "Date"] + list(JEV_COLS)].rename(columns=JEV_COLS)
    d = d.drop(columns=list(JEV_COLS.values())).merge(m, on=["Code", "Date"], how="left")
    ok = d["jev_r"].notna()
    info.update({"rows": int(ok.sum()), "arm": arm, "qset": E74.QSET,
                 "from": str(d.loc[ok, "Date"].min().date()) if ok.any() else None,
                 "to": str(d.loc[ok, "Date"].max().date()) if ok.any() else None})
    log(f"Jev の控え: {arm} 腕 {int(ok.sum()):,}行に答えが付いた（{info['from']}〜{info['to']}）")
    return d, info


# ---------------------------------------------------------------------- #
# 表
# ---------------------------------------------------------------------- #

HEAD = (f"  {'規則':<44}{'件数':>6}{'日数':>6}{'銘柄':>6}{'+20%指値':>9}{'SE2':>6}{'SE銘柄':>7}"
        f"{'+10%指値':>9}{'持ち切り':>9}{'勝率':>6}{'+20%到達':>9}{'正例率':>7}{'−10%割れ':>9}{'平均ボラ':>8}{'最小百分位':>8}")


def line(name: str, s: Dict) -> str:
    if not s.get("n"):
        return f"  {name:<44}{0:>6}"
    if s["n"] < MIN_ROWS:
        return f"  {name:<44}{s['n']:>6}   （件数が少なすぎる）"
    return (f"  {name:<44}{s['n']:>6}{s['days']:>6}{s['codes']:>6}{s['tp20']:>+8.2f}%{s['tp20_se']:>6.2f}"
            f"{s['tp20_se_code']:>7.2f}{s['tp10']:>+8.2f}%{s['hold']:>+8.2f}%{s['win20']:>5.0f}%"
            f"{s['hit20']:>8.1f}%{s['label']:>6.1f}%{s['lose10']:>8.1f}%{s['vol']:>7.2f}%{s['hp_min']:>8.1f}")


def pair_line(a: str, b: str, p: Dict) -> str:
    if not p.get("n"):
        return f"  {a} − {b}: 重なる日が無い"
    yrs = " ".join(f"{y}:{v['d']:+.2f}({v['n']})" for y, v in p["years"].items())
    return (f"  {a:<8}− {b:<8} 日数 {p['n']:>5}  差 {p['d']:>+6.2f}pt  SE(月) {p['se']:>5.2f}  z {p['z']:>+5.2f}"
            f"  SE(iid) {p['se_iid']:>5.2f}  同じ銘柄の日 {p['same']:>4.0f}%  違う日だけ {p['n_diff']:>4}日 "
            f"差 {p['d_diff']:>+6.2f} SE {p['se_diff']:>5.2f}  年ごと +{p['years_pos']}/{p['years_n']} [{yrs}]")


def _measure_lines(m: Dict[str, Dict], labels: Dict[str, str]) -> None:
    print(f"  {'物差し':<30}{'件数':>7}{'AUC(正例)':>10}{'AUC(tp20>0)':>12}{'正例: 真陽性−偽陽性':>20}{'z':>7}"
          f"{'tp20>0: 真−偽':>16}{'z':>7}")
    for col, v in m.items():
        gc, gb = v["gap_c"], v["gap_b"]
        print(f"  {labels.get(col, col):<30}{v['n']:>7}{v['auc_c']:>10.3f}{v['auc_b']:>12.3f}"
              f"{gc['d']:>+14.2f} ± {gc['se']:<5.2f}{gc['z']:>+5.1f}"
              f"{gb['d']:>+10.2f} ± {gb['se']:<5.2f}{gb['z']:>+5.1f}")


def _tercile_lines(rows: List[Dict], unit: str = "%") -> None:
    if not rows:
        print("    （3分位に割る行が足りない）")
        return
    print(f"    {'3分位（vol_20d）':<22}{'件数':>6}{'正例率':>8}{'+20%指値':>9}{'SE':>6}{'+20%到達':>9}{'−10%割れ':>9}")
    names = ("低い 1/3", "中 1/3", "高い 1/3")
    for r in rows:
        nm = names[r["t"]] if r["t"] < 3 else str(r["t"])
        print(f"    {nm:<10}{r['lo']:>5.2f}〜{r['hi']:<5.2f}{unit} {r['n']:>5}{r['label']:>7.1f}%{r['tp20']:>+8.2f}%"
              f"{r['tp20_se']:>6.2f}{r['hit20']:>8.1f}%{r['lose10']:>8.1f}%")


def _js(o):
    """json.dump の default。numpy の値・日付・集合を素の型に。"""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, (set, frozenset)):
        return sorted(str(x) for x in o)
    return str(o)


# ---------------------------------------------------------------------- #
# 集計
# ---------------------------------------------------------------------- #

def report(d: pd.DataFrame, jev_info: Dict, dry_run: bool = False) -> Dict:
    res: Dict[str, object] = {"rows": int(len(d)), "codes": int(d["Code"].nunique()),
                              "from": str(d["Date"].min().date()), "to": str(d["Date"].max().date()),
                              "days": int(d["Date"].nunique()), "jev": {"info": jev_info}}
    c90, c95 = d["cand90"], d["cand95"]
    print(f"\n=== 0. 対象 ===\n  {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {d['Date'].nunique():,}日 / "
          f"{res['from']}〜{res['to']}")
    print(f"  3モデルとも90以上（候補）: {int(c90.sum()):,}行 / {d.loc[c90, 'Date'].nunique():,}日   "
          f"3モデルとも95以上: {int(c95.sum()):,}行 / {d.loc[c95, 'Date'].nunique():,}日")
    print(f"  発火数 {SKIP_BREAKS} 以上の日: {d.loc[d['n_break'] >= SKIP_BREAKS, 'Date'].nunique():,}日 / 全 {d['Date'].nunique():,}日")
    yr = d.groupby("year").agg(rows=("Code", "size"), c90=("cand90", "sum"), c95=("cand95", "sum"),
                               tp20=("tp20", "mean"), label=("y_c", "mean"))
    print(f"  {'年':<6}{'行':>8}{'90以上':>8}{'95以上':>8}{'母集団 tp20':>12}{'正例率':>8}")
    for y, r in yr.iterrows():
        print(f"  {y:<6}{int(r['rows']):>8,}{int(r['c90']):>8,}{int(r['c95']):>8,}{r['tp20']:>+11.2f}%{r['label'] * 100:>7.1f}%")
    res["years"] = {int(y): {"rows": int(r["rows"]), "c90": int(r["c90"]), "c95": int(r["c95"])} for y, r in yr.iterrows()}

    picks = {rule: pick(d, rule) for rule in RULES if rule != "lowjev1"}
    if dry_run:
        print("\n[dry-run] 規則ごとの件数:")
        for rule, p in picks.items():
            print(f"  {rule:<10}{int(p.sum()):>7}行 / {d.loc[p, 'Date'].nunique():>5}日")
        res["dry_run"] = {rule: int(p.sum()) for rule, p in picks.items()}
        return res

    # 1. 規則ごとの成績
    print("\n=== 1. 規則ごとの成績（全期間。1件あたり。SE2 = 銘柄×日付の2方向、SE銘柄 = 銘柄だけ）===")
    print(HEAD)
    res["rules"] = {}
    for rule, p in picks.items():
        s = summarize(d[p])
        res["rules"][rule] = s
        print(line(RULES[rule], s))

    # 2. 規則同士の差（同じ日の対）
    print(f"\n=== 2. 規則同士の差（同じ日の対。主 = +20% 指値。SE は月でまとめる。同じ銘柄を選んだ日は差 0）===")
    res["pairs"] = {}
    for a, b in PAIRS:
        p = paired_diff(d, picks[a], picks[b])
        res["pairs"][f"{a}-{b}"] = p
        print(pair_line(a, b, p))

    # 3. 発火数の線を掛けたとき
    gate = d["n_break"] >= SKIP_BREAKS
    print(f"\n=== 3. 発火数 {SKIP_BREAKS} 未満の日を見送ったとき（運用の見送り線）===")
    print(HEAD)
    res["gated"] = {"rules": {}, "pairs": {}}
    for rule, p in picks.items():
        s = summarize(d[p & gate])
        res["gated"]["rules"][rule] = s
        print(line(RULES[rule], s))
    for a, b in PAIRS[:2]:
        p = paired_diff(d, picks[a] & gate, picks[b] & gate)
        res["gated"]["pairs"][f"{a}-{b}"] = p
        print(pair_line(a, b, p))

    # 4. 集合の中で並べる物差し
    print("\n=== 4. 選定の集合の中で並べる物差し（全期間。AUC 0.5 未満 = 低いほど当たる。差の SE は銘柄ごと）===")
    sets = E73.selections(d)
    ms = measures_in_sets(d, MEASURES, sets)
    res["measures"] = ms
    for name, _ in sets:
        v = ms[name]
        print(f"  --- {name}: {v['n']:,}行 ---")
        _measure_lines(v["measures"], MEASURES)
        _tercile_lines(v["vol_terciles"])

    # 5. 直近1年（Jev の控えがある行）
    print("\n=== 5. 直近1年（実験74 の控えに Jev の答えがある行）: Jev・日次ボラ・20日リターンを同じ土俵で ===")
    res["jev"].update({"rules": {}, "pairs": {}, "measures": {}})
    has = d["jev_r"].notna()
    if has.sum() < MIN_ROWS:
        print("  Jev の控えが無いか少ないので飛ばす")
    else:
        start = d.loc[has, "Date"].min()
        w = d[d["Date"] >= start].copy()
        # その日の候補（90以上）に1つでも答えの無い行がある日は外す（lowjev1 だけ不利にならないように）
        full = w[w["cand90"]].groupby("Date")["jev_r"].apply(lambda s: s.notna().all())
        ok_days = set(full[full].index)
        dropped = int((~full).sum())
        w = w[w["Date"].isin(ok_days) | ~w["cand90"]].copy()
        print(f"  窓 {start.date()}〜{w['Date'].max().date()}: {len(w):,}行 / 候補のある日 {len(ok_days):,}日"
              f"（答えの欠けた候補がある日 {dropped} を外した）")
        wp = {rule: pick(w, rule) for rule in JEV_RULES}
        print(HEAD)
        for rule in JEV_RULES:
            s = summarize(w[wp[rule]])
            res["jev"]["rules"][rule] = s
            print(line(RULES[rule], s))
        for a, b in JEV_PAIRS:
            p = paired_diff(w, wp[a], wp[b])
            res["jev"]["pairs"][f"{a}-{b}"] = p
            print(pair_line(a, b, p))
        sets_w = [(n, m) for n, m in E73.selections(w) if n.startswith("3モデルとも") or n == "母集団"]
        mw = measures_in_sets(w, {**MEASURES, **JEV_MEASURES}, sets_w)
        res["jev"]["measures"] = mw
        for name, _ in sets_w:
            print(f"  --- {name}: {mw[name]['n']:,}行 ---")
            _measure_lines(mw[name]["measures"], {**MEASURES, **JEV_MEASURES})
        rho = w.loc[w["cand90"] & w["jev_r"].notna(), ["jev_r", "jev_c", "vol_20d", "ret_20d", "hp_min"]].corr(method="spearman")
        res["jev"]["rho_cand90"] = {k: {kk: float(vv) for kk, vv in r.items()} for k, r in rho.to_dict().items()}
        print("  候補（90以上）の中の Spearman: " + ", ".join(
            f"Jev R×{c} {rho.loc['jev_r', c]:+.2f}" for c in ("vol_20d", "ret_20d", "hp_min")) +
            f", Jev C×vol_20d {rho.loc['jev_c', 'vol_20d']:+.2f}")

    # 6. 採否の条件
    print("\n=== 6. 採否を検討する条件（冒頭に先に書いたもの）===")
    res["verdict"] = verdict(res)
    for rule, v in res["verdict"].items():
        print(f"  [{rule}] " + "  ".join(f"({i}) {'○' if ok else '×'} {txt}" for i, (ok, txt) in enumerate(v["conds"], 1))
              + f"  → {'全部そろった' if v['all'] else 'そろわない'}")
    return res


def verdict(res: Dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    base = (res.get("rules") or {}).get(BASE_RULE, {})
    for rule in MAIN_RULES:
        p = (res.get("pairs") or {}).get(f"{rule}-{BASE_RULE}", {})
        s = (res.get("rules") or {}).get(rule, {})
        c1 = bool(p.get("n") and np.isfinite(p.get("z", np.nan)) and p["d"] > 0 and p["z"] >= 2)
        c2 = bool(p.get("n") and p.get("years_n") and p["years_pos"] > p["years_n"] / 2)
        c3 = bool(s.get("n") and base.get("n") and s["lose10"] <= base["lose10"])
        jr = (res.get("jev") or {}).get("rules") or {}
        c4 = bool(jr.get(rule, {}).get("n") and jr.get("lowjev1", {}).get("n")
                  and jr[rule]["tp20"] >= jr["lowjev1"]["tp20"])
        conds = [(c1, f"{BASE_RULE} との対の差 {p.get('d', np.nan):+.2f}pt z {p.get('z', np.nan):+.2f}（正で z ≥ 2）"),
                 (c2, f"年ごとの差が正の年 {p.get('years_pos', 0)}/{p.get('years_n', 0)}（過半）"),
                 (c3, f"−10% 割れ {s.get('lose10', np.nan):.1f}% 対 {BASE_RULE} {base.get('lose10', np.nan):.1f}%（高くない）"),
                 (c4, "直近1年の tp20 が lowjev1 以上" + (
                     f"（{jr[rule]['tp20']:+.2f}% 対 {jr['lowjev1']['tp20']:+.2f}%）"
                     if jr.get(rule, {}).get("n") and jr.get("lowjev1", {}).get("n") else "（Jev の控えが無い）"))]
        out[rule] = {"conds": conds, "all": all(c for c, _ in conds)}
    return out


# ---------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験75: 選んだ候補の中で日次ボラが低い銘柄を選ぶ")
    ap.add_argument("--days", type=int, default=0, help="最新日から何暦日を対象にするか（0 = 全期間）")
    ap.add_argument("--jev-arm", default=JEV_ARM, help="実験74 の控えのどの腕を使うか（prod / anon）")
    ap.add_argument("--dry-run", action="store_true", help="件数と規則ごとの件数だけ")
    args = ap.parse_args(argv)

    d, algos, _frame = E73.load_rows(args.days or 36500)
    d = prepare(d)
    log(f"対象 {len(d):,}行 / {d['Code'].nunique():,}銘柄 / {d['Date'].min().date()}〜{d['Date'].max().date()} "
        f"/ モデル {', '.join(algos)}")
    d, jev_info = load_jev(d, arm=args.jev_arm)
    res = report(d, jev_info, dry_run=args.dry_run)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=_js)
    log(f"記録: {OUT}（集計だけ。銘柄ごとの値は出さない）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
