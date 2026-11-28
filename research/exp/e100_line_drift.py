#!/usr/bin/env python3
"""
実験100: 本番モデルの線（百分位 95）が OOF より緩い原因を切り分ける（検証だけ。規則・モデル・画面は変えない）

背景（実運用の追跡 2026-10-16、run 38904278747）: 今の規則の記録 10/13〜10/16 の候補 64件で、lgbm_k15 が95以上の
割合 23.4%（OOF では定義上 5%）、3本すべて95以上 7.8%（OOF の同じ長さの期間の 90% 点 4.6%）。9/23 の初回（以前の
規則）でも同じ向きだった。画面の百分位は「本番モデル（全期間で学習）のスコアが、そのモデルの OOF（前進分割。
3年学習 → 6か月採点を半年ずつ進めたもの。2021-05〜）の分布のどこにあるか」なので、ずれの元は次のどれか。

  (1) 分布の形: 候補全体が上にずれている（水準）のか、両端が太いだけ（分散）か。画面の 10/13〜16 の百分位から
  (2) OOF 自身の時期ごとのずれ: 前進分割の直近の窓（学習が多い・相場が違う）が、プールした分布の中で高めに出ているか。
      出ていれば「プールした分布が古い」が原因の一部。暦月（10月）の偏りも同じ表で見る
  (3) 特徴量の寄与: 本番 lgbm_k15 の TreeSHAP（pred_contrib。画面の「効いた特徴量」と同じ）で、実運用の行と OOF の期間の行の
      寄与の平均の差を列ごとに出す。日内で一定の列（地合い: 日経・TOPIX・金・投資部門別）と、決算の暦（days_to_earn・
      days_since_disc）と、それ以外（銘柄固有）に分ける。反事実: 地合いの列を過去の中央値（全期間 / 10月だけ）に
      置き換えて採点し直し、中央百分位がどれだけ下がるか
  (4) その地合いは OOF で元が取れていたか: 寄与の差が大きい地合いの列の五分位ごとに、OOF の正例率・95以上の割合・
      規則の行の20営業日の収益
  (5) 学習と予測の特徴量のずれ: 日次予測の train/serve の検査（10/16: 149行中 lvs_ratio の1行だけ）に加え、学習の後の
      期間（trainTo の翌日〜）を学習側の特徴量（dataset_predict）で採点し、実運用の行（live_features）の採点と比べる。
      あわせて学習後の各週の百分位（4営業日だけが高いのか、学習の後ずっと高いのか）
  (6) 基準の分布を直近に限ると: OOF の直近12か月 / 最後の窓 / 2026年 / 10月の行だけを基準にした百分位

判定（先に決める）:
  形:      中央百分位 > 65 かつ 10以下の割合 < 5% → 「水準の上方シフト」（分散ではない）
  OOF:     最後の窓の lgbm_k15 95以上の割合 > 10%（定義の2倍）→ 「OOF の中でも直近は高い（プールした分布が古い）」
  地合い:  反事実 A（地合いの列を全期間の中央値に）で中央百分位の 50 との差が半分以下に縮む → 「地合いの特徴量が主因」。
           反事実 B（A ＋ 決算の暦）でさらに縮めば暦の寄与もある
  元:      地合いの列で、いまの値が入る五分位の OOF 正例率が全体より 3pt 以上高い → 「その地合いでは OOF でも正例が多かった
           （線が緩いのは見込みと整合）」。そうでなければ「地合いの過信（OOF で元が取れていない）」
  ずれ:    live_features と dataset_predict の同じ行の百分位の差の中央 < 1pt → 「学習と予測のずれは原因ではない」

回し方: Run Experiment（exp=e100_line_drift.py）。Release の data-raw にあるもの（3本のモデル・OOF・live_features）と、
作ったデータセットを使う。学習後の期間の採点のために dataset_predict（--keep-unlabeled）を作り直す（約2分）。
公開ログには件数・割合・特徴量名と地合いの列の中央値だけ出す（銘柄ごとの値は出さない）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import models as M  # noqa: E402
import feature_dict as FD  # noqa: E402
import e79_model_clusters as E79  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = lab.DATA_DIR
OUT_DIR = E79.OUT_DIR
RULE_MODELS = LT.K15                    # ("lgbm_k15", "xgb_k15", "cat_k15")。実験のときの規則（2026-10-23 からは 1.2σ の3本なので記録どおり固定）
MAIN = "lgbm_k15"                       # 寄与の分解に使う（画面の「効いた特徴量」と同じ TreeSHAP）
LINE = LT.AGREE_PCT                     # 95
LIVE_FROM, LIVE_TO = "2026-10-13", "2026-10-16"   # 今の規則の記録（既定。--live-from / --live-to）
CAL_COLS = ("days_to_earn", "days_since_disc")     # 決算の暦の列
TOP_FEATURES = 12                       # 寄与の差を出す列の数
N_QUANT = 5                             # 地合いの列の五分位
PRED_PATH = os.path.join(ROOT, "public", "data", "predictions.json")
log = E79.log


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #

def model_path(algo: str, data_dir: str = DATA_DIR) -> Optional[str]:
    """Release のアセット名（<algo>_model.joblib）か、日次予測の並べ方（model/models/<algo>/model.joblib）。"""
    cands = [os.path.join(data_dir, f"{algo}_model.joblib"),
             os.path.join(M.sub_dir(algo), "model.joblib")]
    return next((p for p in cands if os.path.exists(p)), None)


def load_model(algo: str, data_dir: str = DATA_DIR):
    import joblib

    p = model_path(algo, data_dir)
    if p is None:
        raise SystemExit(f"{algo} のモデルがありません（{data_dir}）。週次の再学習が Release に上げたものを使う")
    return joblib.load(p)


def features_of(algo: str) -> List[str]:
    """学習に使った列（git 側の meta.json）。"""
    with open(os.path.join(M.sub_dir(algo), "meta.json"), encoding="utf-8") as fh:
        return list(json.load(fh)["features"])


def load_oof(algo: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    p = LT.find_oof(algo, data_dir, LT.MODEL_DIR)
    if p is None:
        raise SystemExit(f"{algo} の OOF がありません（{data_dir}）")
    o = pd.read_parquet(p)
    o["Code"] = o["Code"].astype(str)
    o["Date"] = pd.to_datetime(o["Date"])
    return o.drop_duplicates(["Code", "Date"], keep="first").reset_index(drop=True)


def load_oof_wide(data_dir: str = DATA_DIR, models: Sequence[str] = RULE_MODELS) -> pd.DataFrame:
    """規則の3本の OOF を (Code, Date) で揃え、画面と同じプールした百分位 p_<algo> を付ける。
    label_k15 は lgbm_k15 の自分のラベル、label（1.2σ）と ret_o1_20 は本番の oof.parquet から。"""
    base = load_oof(M.BASELINE, data_dir)
    keep = ["Code", "Date"] + [c for c in ("label", lab.OUTCOME) if c in base.columns]
    wide = base[keep].copy()
    for a in models:
        o = load_oof(a, data_dir)
        cols = ["Code", "Date", "score"] + (["label"] if (a == MAIN and "label" in o.columns) else [])
        o = o[cols].rename(columns={"score": f"s_{a}", "label": "label_k15"})
        wide = wide.merge(o, on=["Code", "Date"], how="inner")
    for a in models:
        wide[f"p_{a}"] = LT.pct_of(wide[f"s_{a}"].to_numpy(), wide[f"s_{a}"].to_numpy())
    return wide.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


def live_rows(live_path: str, cols: Sequence[str], lo: str, hi: str) -> pd.DataFrame:
    """日次予測が控えた特徴量（最初の予測で凍結）のうち、期間内で cols をすべて控えた行。"""
    lv = pd.read_parquet(live_path)
    lv["Date"] = pd.to_datetime(lv["Date"])
    lv["Code"] = lv["Code"].astype(str)
    lv = lv[(lv["Date"] >= pd.Timestamp(lo)) & (lv["Date"] <= pd.Timestamp(hi))]
    if "_features" in lv.columns:
        need = set(cols)
        ok = lv["_features"].fillna("").map(lambda s: need <= set(s.split(",")))
        lv = lv[ok]
    miss = [c for c in cols if c not in lv.columns]
    if miss:
        raise SystemExit(f"控えに無い列: {miss[:5]}…（{len(miss)}列）")
    return lv.drop_duplicates(["Code", "Date"]).reset_index(drop=True)


def ensure_predict_dataset(path: str, data_dir: str = DATA_DIR) -> str:
    """学習後の行（ラベル未確定）も含むデータセット。無ければ build_dataset --keep-unlabeled で作る。"""
    if os.path.exists(path):
        return path
    log("dataset_predict を作る（build_dataset --keep-unlabeled。約2分）")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "research", "build_dataset.py"),
                        "--keep-unlabeled", "--out", path, "--data-dir", data_dir],
                       capture_output=True, text=True, cwd=ROOT)
    tail = "\n".join((r.stdout or "").splitlines()[-6:])
    print(tail)
    if r.returncode != 0 or not os.path.exists(path):
        raise SystemExit(f"dataset_predict を作れません（exit {r.returncode}）\n{(r.stderr or '')[-800:]}")
    return path


# --------------------------------------------------------------------------- #
# 採点・百分位・印
# --------------------------------------------------------------------------- #

def score_all(models: Dict[str, object], X: np.ndarray) -> Dict[str, np.ndarray]:
    return {a: M.predict(m, X) for a, m in models.items()}


def pct_all(scores: Dict[str, np.ndarray], refs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """モデルごとに、基準の分布（そのモデルの OOF のスコア）の中の百分位（画面と同じ丸め）。"""
    return {a: LT.pct_of(s, refs[a]) for a, s in scores.items()}


def pass_mask(pcts: Dict[str, np.ndarray], models: Sequence[str] = RULE_MODELS, line: float = LINE) -> np.ndarray:
    P = np.column_stack([np.asarray(pcts[a], dtype=float) for a in models])
    return np.all(np.nan_to_num(P, nan=-1.0) >= line, axis=1)


def dist(p: np.ndarray) -> dict:
    """百分位の分布の要約（件数・中央・両端の割合）。"""
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if not len(p):
        return {"n": 0}
    return {"n": int(len(p)), "mean": float(p.mean()), "median": float(np.median(p)),
            "le5": float((p <= 5).mean() * 100), "le10": float((p <= 10).mean() * 100),
            "ge90": float((p >= 90).mean() * 100), "ge95": float((p >= 95).mean() * 100)}


def summary_of(pcts: Dict[str, np.ndarray], main: str = MAIN) -> dict:
    d = dist(pcts[main])
    d["pass"] = float(pass_mask(pcts).mean() * 100) if d.get("n") else np.nan
    d["ge95_by_model"] = {a: float((np.asarray(p, dtype=float) >= LINE).mean() * 100) for a, p in pcts.items()}
    return d


# --------------------------------------------------------------------------- #
# 地合いの列（日内で一定）と寄与の分解
# --------------------------------------------------------------------------- #

def market_wide(df: pd.DataFrame, cols: Sequence[str], min_days: int = 20) -> List[str]:
    """日内で値が一定の列（= 地合い。どの銘柄にも同じ値が入る）。欠測は無視し、値のある日が min_days 未満の列は数えない。"""
    out = []
    g = df.groupby("Date")
    for c in cols:
        if c not in df.columns:
            continue
        nun = g[c].nunique(dropna=True)
        has = nun[nun > 0]
        if len(has) >= min_days and int(nun.max()) <= 1:
            out.append(c)
    return out


def contrib_matrix(model, X: np.ndarray) -> np.ndarray:
    """TreeSHAP（対数オッズ）。最後の列が基準値。"""
    booster = getattr(model, "booster_", model)
    return np.asarray(booster.predict(X, pred_contrib=True), dtype=float)


def decompose(model, cols: Sequence[str], X_live: np.ndarray, X_hist: np.ndarray) -> Tuple[pd.DataFrame, dict]:
    """列ごとの寄与の平均（実運用 − 過去）。差の合計 = 生スコア（対数オッズ）の平均の差。"""
    cl, ch = contrib_matrix(model, X_live), contrib_matrix(model, X_hist)
    live_m, hist_m = cl[:, :-1].mean(axis=0), ch[:, :-1].mean(axis=0)
    tbl = pd.DataFrame({"col": list(cols), "live": live_m, "hist": hist_m, "diff": live_m - hist_m})
    tbl["abs"] = tbl["diff"].abs()
    tbl = tbl.sort_values("abs", ascending=False).reset_index(drop=True)
    info = {"base_live": float(cl[:, -1].mean()), "base_hist": float(ch[:, -1].mean()),
            "raw_live": float(cl.sum(axis=1).mean()), "raw_hist": float(ch.sum(axis=1).mean())}
    return tbl, info


def group_sums(tbl: pd.DataFrame, mkt: Sequence[str], cal: Sequence[str] = CAL_COLS) -> dict:
    mkt, cal = set(mkt), set(cal)
    g = {"地合い（日内で一定）": 0.0, "決算の暦": 0.0, "銘柄固有（それ以外）": 0.0}
    for r in tbl.itertuples(index=False):
        if r.col in mkt:
            g["地合い（日内で一定）"] += float(r.diff)
        elif r.col in cal:
            g["決算の暦"] += float(r.diff)
        else:
            g["銘柄固有（それ以外）"] += float(r.diff)
    return g


def replace_cols(X_df: pd.DataFrame, values: Dict[str, float]) -> pd.DataFrame:
    """反事実: 指定の列を与えた値に置き換える（他の列はそのまま）。"""
    out = X_df.copy()
    for c, v in values.items():
        if c in out.columns:
            out[c] = v
    return out


def daily_medians(hist: pd.DataFrame, cols: Sequence[str], month: Optional[int] = None,
                  day_hi: int = 31) -> Dict[str, float]:
    """地合いの列の過去の中央値（日ごとの値を1日1つにしてから。month を渡すとその月の day_hi 日までの日だけ）。"""
    cols = [c for c in cols if c in hist.columns]
    d = hist[["Date"] + list(cols)].drop_duplicates("Date")
    if month is not None:
        d = d[(d["Date"].dt.month == month) & (d["Date"].dt.day <= day_hi)]
    return {c: float(d[c].median()) for c in cols if c in d.columns and d[c].notna().any()}


def value_pct(hist_vals: np.ndarray, v: float) -> float:
    h = np.asarray(hist_vals, dtype=float)
    h = h[np.isfinite(h)]
    if not len(h) or not np.isfinite(v):
        return np.nan
    return float((h < v).mean() * 100)


# --------------------------------------------------------------------------- #
# 表
# --------------------------------------------------------------------------- #

def _rate(x) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean() * 100) if len(x) else np.nan


def period_row(g: pd.DataFrame) -> dict:
    """OOF の一部分の要約: 件数、lgbm_k15 の中央百分位、95以上の割合（3本）、3本すべて95以上の割合、正例率、規則の行の収益。"""
    pcts = {a: g[f"p_{a}"].to_numpy() for a in RULE_MODELS}
    ps = pass_mask(pcts)
    out = {"n": int(len(g)), "median": float(np.median(g[f"p_{MAIN}"])) if len(g) else np.nan,
           "pass": float(ps.mean() * 100) if len(g) else np.nan,
           "pos15": _rate(g["label_k15"]) if "label_k15" in g else np.nan,
           "pos12": _rate(g["label"]) if "label" in g else np.nan,
           "n_pass": int(ps.sum())}
    for a in RULE_MODELS:
        out[f"ge95_{a}"] = float((g[f"p_{a}"] >= LINE).mean() * 100) if len(g) else np.nan
    if lab.OUTCOME in g:
        r = pd.to_numeric(g.loc[ps, lab.OUTCOME], errors="coerce").dropna()
        out["ret_pass"] = float(r.mean() * 100) if len(r) else np.nan
    return out


def window_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, g in oof.groupby("window"):
        if w == 0:
            continue
        rows.append({"window": int(w), "from": g["Date"].min().date(), "to": g["Date"].max().date(), **period_row(g)})
    return pd.DataFrame(rows)


def year_table(oof: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"year": int(y), **period_row(g)} for y, g in oof.groupby(oof["Date"].dt.year)])


def month_table(oof: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"month": int(m), **period_row(g)} for m, g in oof.groupby(oof["Date"].dt.month)])


def quintile_table(oof: pd.DataFrame, col: str, n: int = N_QUANT) -> pd.DataFrame:
    """地合いの列の分位ごと（行で等分）の OOF の成績。"""
    d = oof[oof[col].notna()].copy()
    if len(d) < n * 20:
        return pd.DataFrame()
    d["q"] = pd.qcut(d[col].rank(method="first"), n, labels=False)
    rows = []
    for q, g in d.groupby("q"):
        rows.append({"q": int(q) + 1, "lo": float(g[col].min()), "hi": float(g[col].max()), **period_row(g)})
    return pd.DataFrame(rows)


def weekly_table(rows: pd.DataFrame, pcts: Dict[str, np.ndarray]) -> pd.DataFrame:
    """学習後の期間を週ごとに（月曜始まり）。"""
    d = rows[["Date"]].copy()
    for a, p in pcts.items():
        d[f"p_{a}"] = np.asarray(p, dtype=float)
    d["week"] = d["Date"].dt.to_period("W-SUN").dt.start_time
    out = []
    for w, g in d.groupby("week"):
        pc = {a: g[f"p_{a}"].to_numpy() for a in pcts}
        s = summary_of(pc)
        out.append({"week": w.date(), "days": int(g["Date"].nunique()), "n": s["n"], "median": s["median"],
                    "ge95": s["ge95"], "pass": s["pass"]})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #

def judge(live: dict, last_window: dict, cf_a: dict, cf_b: dict, top_q: Optional[dict], overall_pos: float,
          skew_median: float) -> List[Tuple[str, str, str]]:
    out = []
    med = live.get("median", np.nan)
    shape = bool(np.isfinite(med) and med > 65 and live.get("le10", 100) < 5)
    out.append(("形", "水準の上方シフト" if shape else "上方シフトとは言えない（裾か、ずれ無し）",
                f"中央 {med:.1f}・10以下 {live.get('le10', np.nan):.1f}%"))
    lw = last_window.get(f"ge95_{MAIN}", np.nan)
    out.append(("OOF の直近の窓", "直近は高い（プールした分布が古い）" if (np.isfinite(lw) and lw > 10) else "直近も定義どおり（古さは主因でない）",
                f"最後の窓の {MAIN} 95以上 {lw:.1f}%（定義 5%）"))
    gap0 = med - 50 if np.isfinite(med) else np.nan
    ga = cf_a.get("median", np.nan) - 50
    gb = cf_b.get("median", np.nan) - 50
    mkt_main = bool(np.isfinite(gap0) and np.isfinite(ga) and gap0 > 0 and ga <= gap0 / 2)
    out.append(("地合いの特徴量", "主因（置き換えで差が半分以下）" if mkt_main else "主因ではない（置き換えても残る）",
                f"中央 {med:.1f} → 反事実A {cf_a.get('median', np.nan):.1f} → 反事実B {cf_b.get('median', np.nan):.1f}"))
    cal_more = bool(np.isfinite(ga) and np.isfinite(gb) and gb <= ga - 5)
    out.append(("決算の暦", "寄与あり（A→B でさらに 5pt 以上縮む）" if cal_more else "寄与は小さい",
                f"反事実A {cf_a.get('median', np.nan):.1f} → B {cf_b.get('median', np.nan):.1f}"))
    if top_q and np.isfinite(top_q.get("pos15", np.nan)):
        gain = top_q["pos15"] - overall_pos
        out.append(("地合いの元", "OOF でも正例が多かった（見込みと整合）" if gain >= 3 else "地合いの過信（OOF で元が取れていない）",
                    f"いまの値が入る五分位の正例率 {top_q['pos15']:.1f}% vs 全体 {overall_pos:.1f}%（{gain:+.1f}pt）"))
    else:
        out.append(("地合いの元", "比べられない", "五分位が作れない"))
    out.append(("学習と予測のずれ", "原因ではない" if (np.isfinite(skew_median) and skew_median < 1) else "ずれがある（要調査）",
                f"同じ行の百分位の差の中央 {skew_median:.2f}pt"))
    return out


# --------------------------------------------------------------------------- #
# 表示
# --------------------------------------------------------------------------- #

def _f(v, fmt="{:.1f}", na="-"):
    try:
        return na if v is None or not np.isfinite(float(v)) else fmt.format(float(v))
    except (TypeError, ValueError):
        return na


def print_dist(title: str, d: dict) -> None:
    print(f"  {title:28s} n={d.get('n', 0):4d}  中央 {_f(d.get('median'))}  平均 {_f(d.get('mean'))}  "
          f"5以下 {_f(d.get('le5'))}%  10以下 {_f(d.get('le10'))}%  90以上 {_f(d.get('ge90'))}%  95以上 {_f(d.get('ge95'))}%"
          + (f"  3本とも95以上 {_f(d.get('pass'))}%" if "pass" in d else ""))


def print_period_table(tbl: pd.DataFrame, key: str, label: str) -> None:
    if tbl.empty:
        print("  （行が無い）")
        return
    has_ret = "ret_pass" in tbl.columns
    print(f"  {label:14s}{'件数':>7s}{'中央%':>7s}{'L15≥95':>8s}{'X15≥95':>8s}{'C15≥95':>8s}{'3本≥95':>8s}{'正例1.5σ':>9s}{'正例1.2σ':>9s}"
          + (f"{'規則の行':>7s}{'20日%':>8s}" if has_ret else ""))
    for rd in tbl.to_dict("records"):
        name = rd[key] if key != "window" else f"窓{rd['window']} {rd['from']}〜"
        line = (f"  {str(name):14s}{int(rd['n']):>7,d}{_f(rd['median']):>7s}{_f(rd['ge95_lgbm_k15']):>8s}{_f(rd['ge95_xgb_k15']):>8s}"
                f"{_f(rd['ge95_cat_k15']):>8s}{_f(rd['pass']):>8s}{_f(rd['pos15']):>9s}{_f(rd['pos12']):>9s}")
        if has_ret:
            line += f"{int(rd['n_pass']):>7d}{_f(rd.get('ret_pass'), '{:+.2f}'):>8s}"
        print(line)


def report(res: dict) -> None:
    print("=" * 100)
    print(f"実験100: 本番モデルの線が OOF より緩い原因（{res['live_from']}〜{res['live_to']} の候補 {res['n_live']}件。"
          f"規則の3本 {' / '.join(RULE_MODELS)}、線 {LINE:g}）")
    print("=" * 100)

    print("\n■ 0. 画面の百分位の分布（予測ファイル。モデルごとの OOF に対する位置）")
    for a, d in res["screen"].items():
        print_dist(a, d)
    print(f"  控えた特徴量から採点し直した百分位との差（{MAIN}）: 最大 {_f(res['rescore_maxdiff'], '{:.2f}')}pt（0 なら画面と同じ計算）")

    print("\n■ 1. OOF の中の時期ごとのずれ（百分位は OOF 全体の分布で付ける。95以上の割合は OOF 全体では定義上 5%）")
    print("  前進分割の窓ごと")
    print_period_table(res["windows"], "window", "窓")
    print("  年ごと")
    print_period_table(res["years"], "year", "年")
    print("  暦月ごと（年をまとめて。10月の偏りを見る）")
    print_period_table(res["months"], "month", "月")

    print(f"\n■ 2. 寄与の分解（{MAIN} の TreeSHAP、対数オッズ。実運用の行 − OOF の期間の行）")
    print(f"  生スコア（対数オッズ）の平均: 実運用 {_f(res['dec_info']['raw_live'], '{:+.3f}')} / 過去 {_f(res['dec_info']['raw_hist'], '{:+.3f}')}"
          f"（差 {_f(res['dec_info']['raw_live'] - res['dec_info']['raw_hist'], '{:+.3f}')}）")
    print("  群ごとの差の合計:")
    for k, v in res["groups"].items():
        print(f"    {k:22s} {v:+.3f}")
    print(f"  地合いの列（日内で一定）: {len(res['mkt_cols'])}列 = {', '.join(res['mkt_cols'])}")
    print(f"  {'列':24s}{'差':>8s}{'実運用':>8s}{'過去':>8s}  {'いまの値(中央)':>14s}{'過去の中央':>10s}{'過去の中の位置':>12s}  日本語名")
    for r in res["top"]:
        print(f"  {r['col']:24s}{r['diff']:>+8.3f}{r['live']:>+8.3f}{r['hist']:>+8.3f}  {_f(r['v_live'], '{:.3g}'):>14s}"
              f"{_f(r['v_hist'], '{:.3g}'):>10s}{_f(r['v_pct']) + '%':>12s}  {r['name']}{'  ［地合い］' if r['mkt'] else ''}{'  ［暦］' if r['cal'] else ''}")
    print("  反事実（実運用の行の列を置き換えて3本で採点し直す）:")
    print_dist("そのまま", res["cf"]["そのまま"])
    print_dist("A 地合い→全期間の中央値", res["cf"]["A"])
    print_dist("B A＋決算の暦→中央値", res["cf"]["B"])
    print_dist("C 地合い→過去の10月の中央値", res["cf"]["C"])

    print("\n■ 3. その地合いは OOF で元が取れていたか（寄与の差が大きい地合いの列の五分位。行で等分）")
    for col, q in res["quintiles"].items():
        print(f"  {col}（{FD.name_ja(col)}）: いまの値 {_f(res['live_vals'].get(col), '{:.3g}')} = 過去の {_f(res['live_pct'].get(col))}% 点。"
              f"全体の正例率 1.5σ {_f(res['overall_pos'])}%")
        if q.empty:
            print("    （五分位が作れない）")
            continue
        print(f"    {'分位':6s}{'範囲':>22s}{'件数':>7s}{'中央%':>7s}{'L15≥95':>8s}{'3本≥95':>8s}{'正例1.5σ':>9s}{'正例1.2σ':>9s}{'規則の行':>8s}{'20日%':>8s}")
        for rd in q.to_dict("records"):
            mark = " ←いま" if res["live_q"].get(col) == rd["q"] else ""
            print(f"    {int(rd['q']):<6d}{_f(rd['lo'], '{:.3g}') + '〜' + _f(rd['hi'], '{:.3g}'):>22s}{int(rd['n']):>7,d}{_f(rd['median']):>7s}"
                  f"{_f(rd['ge95_lgbm_k15']):>8s}{_f(rd['pass']):>8s}{_f(rd['pos15']):>9s}{_f(rd['pos12']):>9s}{int(rd['n_pass']):>8d}"
                  f"{_f(rd.get('ret_pass'), '{:+.2f}'):>8s}{mark}")

    print(f"\n■ 4. 学習の後の期間（trainTo {res['train_to']} の翌日〜）を本番モデルで採点（学習側の特徴量。週ごと）")
    w = res["weekly"]
    if w.empty:
        print("  （行が無い）")
    else:
        print(f"  {'週（月曜）':12s}{'日数':>5s}{'件数':>6s}{'中央%':>7s}{'L15≥95':>8s}{'3本≥95':>8s}")
        for rd in w.to_dict("records"):
            print(f"  {str(rd['week']):12s}{int(rd['days']):>5d}{int(rd['n']):>6d}{_f(rd['median']):>7s}{_f(rd['ge95']):>8s}{_f(rd['pass']):>8s}")
    sk = res["skew"]
    print(f"  学習と予測のずれ（同じ行を控えの特徴量と学習側の特徴量で採点。{MAIN} の百分位の差 = 控え − 学習側）: "
          f"{sk.get('n', 0)}行、中央 {_f(sk.get('median'), '{:+.2f}')}pt、絶対値の中央 {_f(sk.get('abs_median'), '{:.2f}')}pt、最大 {_f(sk.get('abs_max'), '{:.2f}')}pt")

    print("\n■ 5. 基準の分布を直近に限ると（実運用の行の百分位。各モデルの OOF の一部を基準にする）")
    for name, d in res["refs"].items():
        print_dist(name, d)

    print("\n■ 6. 判定（先に決めた基準）")
    for name, verdict, detail in res["judge"]:
        print(f"  {name:12s} {verdict}  （{detail}）")
    print("\n■ 注意")
    print("  ・寄与の分解は本番モデル（全期間で学習）の TreeSHAP。過去の行は学習に使った行なので、その寄与は当てはめ後の値。"
          "実運用の行との差を読むための物差しで、過去の行の予測力を測るものではない")
    print("  ・反事実は列の置き換えで、列どうしの整合（例: 日経の20日と120日）は崩れる。大きさの目安として読む")
    print("  ・本番は何も変えない。線や基準の分布を変えるかは、この結果を見て運用者が決める")


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #

def screen_dists(pred_path: str, lo: str, hi: str, models: Sequence[str]) -> Dict[str, dict]:
    with open(pred_path, encoding="utf-8") as fh:
        d = json.load(fh)
    rows = [r for r in d.get("candidates", []) if lo <= str(r.get("date", "")) <= hi]
    out = {}
    pcts = {}
    for a in models:
        p = np.array([r["byModel"][a]["pctHistorical"] for r in rows
                      if a in (r.get("byModel") or {}) and r["byModel"][a].get("pctHistorical") is not None], dtype=float)
        pcts[a] = p
        out[a] = dist(p)
    if all(len(pcts[a]) == len(rows) for a in RULE_MODELS) and rows:
        out[RULE_MODELS[0]]["pass"] = float(pass_mask(pcts).mean() * 100)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本番モデルの線が OOF より緩い原因を切り分ける")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--live-from", default=LIVE_FROM)
    ap.add_argument("--live-to", default=LIVE_TO)
    ap.add_argument("--predict-dataset", default="", help="dataset_predict.parquet の場所（既定: <data-dir>/dataset_predict.parquet）")
    ap.add_argument("--no-rebuild", action="store_true", help="dataset_predict が無くても作らない（§4 を飛ばす）")
    ap.add_argument("--json", default=os.path.join(OUT_DIR, "e100_summary.json"))
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)

    cols = features_of(MAIN)
    for a in RULE_MODELS:
        if features_of(a) != cols:
            raise SystemExit(f"{a} の列が {MAIN} と違う")
    log(f"列 {len(cols)}、モデル {RULE_MODELS} を読む")
    models = {a: load_model(a, args.data_dir) for a in RULE_MODELS}
    oof = load_oof_wide(args.data_dir)
    refs_all = {a: oof[f"s_{a}"].to_numpy(dtype=float) for a in RULE_MODELS}
    log(f"OOF {len(oof):,}行 {oof['Date'].min().date()}〜{oof['Date'].max().date()}")

    # データセット（学習側の特徴量。OOF の期間の行を「過去」とする）
    ds = pd.read_parquet(os.path.join(args.data_dir, "dataset.parquet"))
    ds["Code"] = ds["Code"].astype(str)
    ds["Date"] = pd.to_datetime(ds["Date"])
    folds = E79.make_windows(ds["Date"])
    oof["window"] = E79.assign_windows(oof["Date"], folds)
    hist = ds.merge(oof[["Code", "Date"]], on=["Code", "Date"], how="inner").reset_index(drop=True)
    log(f"過去の行（データセット ∩ OOF）{len(hist):,}行")
    train_to = str(oof["Date"].max().date())
    meta_main = json.load(open(os.path.join(M.sub_dir(MAIN), "meta.json"), encoding="utf-8"))
    train_to = str(meta_main.get("trainTo") or train_to)

    # 実運用の行（控えた特徴量）
    live = live_rows(os.path.join(args.data_dir, "live_features.parquet"), cols, args.live_from, args.live_to)
    log(f"実運用の行 {len(live)}件（{args.live_from}〜{args.live_to}、{live['Date'].nunique()}日）")
    X_live = live[cols].to_numpy(dtype=float)
    sc_live = score_all(models, X_live)
    pc_live = pct_all(sc_live, refs_all)

    # §0 画面の分布と、採点し直しの一致
    screen = screen_dists(PRED_PATH, args.live_from, args.live_to, list(RULE_MODELS) + ["lgbm", "xgb", "cat"])
    with open(PRED_PATH, encoding="utf-8") as fh:
        pj = json.load(fh)
    disp = {(str(r["code"])[:4], str(r["date"])): r["byModel"][MAIN]["pctHistorical"]
            for r in pj.get("candidates", []) if MAIN in (r.get("byModel") or {})}
    diffs = [abs(float(pc_live[MAIN][i]) - float(disp[k])) for i, k in
             enumerate(zip(live["Code"].str[:4], live["Date"].dt.strftime("%Y-%m-%d"))) if k in disp and disp[k] is not None]
    rescore_maxdiff = max(diffs) if diffs else np.nan

    # §1 OOF の時期ごと
    windows, years, months = window_table(oof), year_table(oof), month_table(oof)

    # §2 寄与の分解と反事実
    mkt = market_wide(hist, cols)
    X_hist = hist[cols].to_numpy(dtype=float)
    tbl, dec_info = decompose(models[MAIN], cols, X_live, X_hist)
    groups = group_sums(tbl, mkt)
    live_df = live[cols].copy()
    top = []
    for r in tbl.head(TOP_FEATURES).itertuples(index=False):
        v_live = float(np.nanmedian(live_df[r.col].to_numpy(dtype=float))) if live_df[r.col].notna().any() else np.nan
        hv = (hist.drop_duplicates("Date") if r.col in mkt else hist)[r.col].to_numpy(dtype=float)
        top.append({"col": r.col, "name": FD.name_ja(r.col), "diff": float(r.diff), "live": float(r.live), "hist": float(r.hist),
                    "v_live": v_live, "v_hist": float(np.nanmedian(hv)) if np.isfinite(hv).any() else np.nan,
                    "v_pct": value_pct(hv, v_live), "mkt": r.col in mkt, "cal": r.col in CAL_COLS})
    med_all = daily_medians(hist, mkt)
    med_oct = daily_medians(hist, mkt, month=10, day_hi=20)
    med_cal = {c: float(np.nanmedian(hist[c].to_numpy(dtype=float))) for c in CAL_COLS if c in hist.columns}
    cf = {"そのまま": summary_of(pc_live)}
    for name, vals in (("A", med_all), ("B", {**med_all, **med_cal}), ("C", med_oct)):
        Xc = replace_cols(live_df, vals)[cols].to_numpy(dtype=float)
        cf[name] = summary_of(pct_all(score_all(models, Xc), refs_all))

    # §3 地合いの列の五分位
    mkt_top = [r["col"] for r in top if r["mkt"]][:3]
    hist_oof = oof.merge(hist[["Code", "Date"] + mkt_top], on=["Code", "Date"], how="inner") if mkt_top else oof.iloc[:0]
    quintiles, live_vals, live_pct, live_q = {}, {}, {}, {}
    for c in mkt_top:
        q = quintile_table(hist_oof, c)
        quintiles[c] = q
        v = float(np.nanmedian(live_df[c].to_numpy(dtype=float)))
        live_vals[c], live_pct[c] = v, value_pct(hist.drop_duplicates("Date")[c].to_numpy(dtype=float), v)
        if not q.empty and np.isfinite(v):
            hit = q[(q["lo"] <= v) & (v <= q["hi"])]
            live_q[c] = int(hit["q"].iloc[0]) if len(hit) else (int(q["q"].max()) if v > q["hi"].max() else int(q["q"].min()))
    overall_pos = _rate(oof["label_k15"]) if "label_k15" in oof else np.nan

    # §4 学習後の期間（学習側の特徴量）と、控えとの差
    weekly, skew = pd.DataFrame(), {}
    pred_path = args.predict_dataset or os.path.join(args.data_dir, "dataset_predict.parquet")
    if not args.no_rebuild or os.path.exists(pred_path):
        pred_path = ensure_predict_dataset(pred_path, args.data_dir)
        pr = pd.read_parquet(pred_path, columns=["Code", "Date"] + cols)
        pr["Code"] = pr["Code"].astype(str)
        pr["Date"] = pd.to_datetime(pr["Date"])
        pr = pr[pr["Date"] > pd.Timestamp(train_to)].reset_index(drop=True)
        if len(pr):
            pc_pr = pct_all(score_all(models, pr[cols].to_numpy(dtype=float)), refs_all)
            weekly = weekly_table(pr, pc_pr)
            both = live[["Code", "Date"]].copy()
            both["p_live"] = pc_live[MAIN]
            prd = pr[["Code", "Date"]].copy()
            prd["p_pred"] = pc_pr[MAIN]
            both = both.merge(prd, on=["Code", "Date"], how="inner")
            if len(both):
                d = (both["p_live"] - both["p_pred"]).to_numpy(dtype=float)
                skew = {"n": int(len(d)), "median": float(np.median(d)), "abs_median": float(np.median(np.abs(d))),
                        "abs_max": float(np.abs(d).max())}

    # §5 基準を直近に
    last_w = int(oof["window"].max())
    subsets = {"全体（いまの画面）": np.ones(len(oof), dtype=bool),
               "直近12か月": (oof["Date"] > oof["Date"].max() - pd.Timedelta(days=365)).to_numpy(),
               "最後の窓": (oof["window"] == last_w).to_numpy(),
               "2026年": (oof["Date"].dt.year == 2026).to_numpy(),
               "10月の行（全年）": (oof["Date"].dt.month == 10).to_numpy()}
    refs_out = {}
    for name, m in subsets.items():
        refs = {a: oof.loc[m, f"s_{a}"].to_numpy(dtype=float) for a in RULE_MODELS}
        refs_out[f"{name}（{int(m.sum()):,}行）"] = summary_of(pct_all(sc_live, refs))

    # §6 判定
    lw_row = windows[windows["window"] == last_w].iloc[0].to_dict() if len(windows) else {}
    top_q = None
    if mkt_top and not quintiles[mkt_top[0]].empty and mkt_top[0] in live_q:
        q = quintiles[mkt_top[0]]
        top_q = q[q["q"] == live_q[mkt_top[0]]].iloc[0].to_dict()
    verdicts = judge(cf["そのまま"], lw_row, cf["A"], cf["B"], top_q, overall_pos, skew.get("abs_median", np.nan))

    res = {"live_from": args.live_from, "live_to": args.live_to, "n_live": int(len(live)), "screen": screen,
           "rescore_maxdiff": rescore_maxdiff, "windows": windows, "years": years, "months": months,
           "dec_info": dec_info, "groups": groups, "mkt_cols": mkt, "top": top, "cf": cf, "quintiles": quintiles,
           "live_vals": live_vals, "live_pct": live_pct, "live_q": live_q, "overall_pos": overall_pos,
           "train_to": train_to, "weekly": weekly, "skew": skew, "refs": refs_out, "judge": verdicts}
    report(res)

    def plain(o):
        if isinstance(o, pd.DataFrame):
            return json.loads(o.to_json(orient="records", date_format="iso"))
        if isinstance(o, dict):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(plain(res), fh, ensure_ascii=False, indent=1)
    log(f"要約を書いた: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
