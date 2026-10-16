#!/usr/bin/env python3
"""
実験102: 目的変数の定義を変える（「到達のみ」「+10% の指値に届く」= ストップ高のような急騰のあと落ちた行も正例にする）と、
LightGBM の高スコア帯の収益（運用と同じ出口: +10% の指値に届けば +10、届かなければ20営業日目の終値）は、現行の選定のラベル
（1.5σ の3条件）より上がるか。学習器は LightGBM だけ、腕ごとに探索し直す。検証だけ。本番は変えない。

運用者の指示（2026-10-16、実験99〜101 で「基準日より前のストップ高の列は高スコア帯に効かない。ラベルが往って来いを外す作りなので
正例に結びつかない」を見て）「であれば、目的変数の定義を変えて、そういったものも、正例としたいです。まずは検証からで。本番は
触らずに。その方がリターンが上がるのであれば、目的変数の定義の変更はありです」。

腕（ラベルだけ違う。列は本番の239列。腕ごとに Optuna の探索（50試行 × 5分割 year_cap_date）をし直す。運用者の指示（2026-10-16）
「検証は lgbm の 1.5σ のみで良い」で、1.2σ の腕（CUR12・HIT12）は外した。TP10 は運用者の選択で残す）
  CUR15  現行の選定のラベル: 1.5σ 到達 + 終盤（到達の 0.5倍）+ トレンド（MA5 ≥ MA20）。基準（2026-10-12 からの規則のモデル）
  HIT15  到達のみ 1.5σ（終盤・トレンドの条件なし）。往って来い（ストップ高のあと落ちた行）も正例
  TP10   +10% の指値に届く（翌営業日の寄りで買い、20営業日以内に高値が買値の 1.1倍以上）= 今日の戦略の売りの規則そのもの
  ラベルの引き直しは build_dataset.rise_label_for_k（終盤・トレンドは RiseConfig で外す）。TP10 は live_track.forward と同じ
  判定（e83.attach_exits の hit10）。どのラベルも、現行のラベルが決まる行（20営業日の結果がそろう行）だけで決める

追加の腕（運用者の指示 2026-10-16「目的変数の定義変更は、ストップ高の履歴を特徴量に入れると効果があるので、それを入れた検証も
実施してください」。2回目の run で足す。1回目の探索・OOF は cache で引き継ぐ）
  ラベル L ∈ {HIT15, TP10} ごとに、実験101 の7列（ul_t・ul_past20_n・ul_past60_n・ul_past120_n・ul_past250_n・ul_days_since・
  ul_decay。基準日の時点で分かるストップ高の履歴）を本番の239列に足した腕:
  L_ULB  239 + 7列で探索し直し（Optuna 50試行 × 5分割）
  L_ULC  239 + 7列を L のパラメータで（探索なし）→ 列を足したことそのものの効果
  L_ULP  対照: 7列を同じ日付の行どうしで入れ替えたもの（種 PERM_SEED）を L のパラメータで
  判定（回す前に決めた）: L ごとに、窓の中の上位5% の tp10c の窓ごとの差で、C−A（L_ULC − L）の平均が正・対照 P−A（L_ULP − L）の
      平均を上回る・C が上の窓が過半（17/32 以上）の3つ（実験100 の判定の物差しを収益にしたもの）で「列の効果あり」。
      参考に自分のラベルで測った窓の PR-AUC の C−A・P−A、B−A（再学習で起きること）、ほかの選び方の差

評価（出口は運用と同じ tp10c。選び方は LightGBM 1本で代える）
  選び方: 窓の中の上位5%（「1.5σ の3本 95以上」の代わり）・上位10%・前の窓の分布の上位5%（閾値。窓1は対象外）・月に2銘柄
  物差し: tp10c の平均（主）、ret_o1_20 の平均、+10% に届く割合、−10% 割れ（tp10c < −10）、買ってからのストップ高（price_limits）、
      現行（CUR15）のラベルの正例率。ずらし 0/2/4か月 × 種3つの平均で 32窓
  判定（回す前に決めた。2026-10-16）: 腕ごとに、窓の中の上位5% の tp10c の平均の CUR15 に対する窓ごとの差（32窓）が
      **平均 > 0・上の窓が過半（17/32 以上）・z = 平均 ÷ SE ≥ 1.64** の3つをそろえれば「リターンが上がる（支持）」。
      HIT15・TP10 のそれぞれに当てる。支持した腕があれば、本番に入れる案（3モデル・ラベル違いのモデルの置き換え）を運用者に出す。
      無ければ、この線は閉じる
  参考（判定に使わない）: 上位10%・閾値・月2銘柄の同じ差、相手のラベルで測った PR-AUC（HIT の正例も現行の正例を上に並べるか）、
      探索の CV、ラベルの正例率・重なり・正例の収益、窓ごとの値の一覧

出すもの: 件数・割合・収益・精度だけ（銘柄・値は出さない）。本番の設定（build_dataset の RiseConfig・lgbm_params.json・
tuning_multi.VARIANTS）には書かない。

    exp=e102_label_hit_ab.py                      （1回目: CUR15・HIT15・TP10）
    exp=e102_label_hit_ab.py args="--with-ul"     （2回目: + ストップ高の履歴の7列の腕。1回目の探索・OOF は cache で引き継ぐ）
充足だけ見る: exp=e102_label_hit_ab.py args="--dry"
試運転:       exp=e102_label_hit_ab.py args="--tag e102smoke --n-trials 2 --shifts 0 --with-ul"

表（出口・ストップ高・各ラベルを付けた frame）は1回目に作って research/_data/oof/<tag>_* に置き、続きの回は actions/cache で
引き継いだそれを使う。--budget-min（既定 290分）を過ぎたら新しい計算を始めずに終わる（保存済みの探索・out-of-fold は次の回が
続きから使う）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import price_limits as PL  # noqa: E402
import train_model as T  # noqa: E402
import e68_edinet_ab as E68  # noqa: E402
import e78_label_k as E78  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

log = E68.log
ALGO = "lgbm"
ARMS = ("CUR15", "HIT15", "TP10")
BASE = "CUR15"
JUDGED = ("HIT15", "TP10")
#: ラベルの定義（kind=sigma: k × σ の到達。end / trend は終盤・トレンドの条件。kind=tp: +target% の指値に届く）。
#: CUR12 は引き直しの確かめ（本番のラベルと一致するか）にだけ使う。HIT12 は腕には入れない（運用者の指示 2026-10-16）
LABEL_DEFS = {
    "CUR15": {"kind": "sigma", "k": 1.5, "end": True, "trend": True},
    "CUR12": {"kind": "sigma", "k": 1.2, "end": True, "trend": True},
    "HIT15": {"kind": "sigma", "k": 1.5, "end": False, "trend": False},
    "HIT12": {"kind": "sigma", "k": 1.2, "end": False, "trend": False},
    "TP10": {"kind": "tp", "target": 10.0},
}
NAMES = {"CUR15": "CUR15 現行 1.5σ 3条件", "CUR12": "CUR12 1.2σ 3条件（参考）", "HIT15": "HIT15 到達のみ 1.5σ",
         "HIT12": "HIT12 到達のみ 1.2σ", "TP10": "TP10 +10% の指値に届く"}
#: ストップ高の履歴の7列（実験101 と同じ。基準日の時点で分かる）を足す腕。L_ULB 探索し直し / L_ULC L のパラメータ / L_ULP 対照
UL_FEATURES = ("ul_t", "ul_past20_n", "ul_past60_n", "ul_past120_n", "ul_past250_n", "ul_days_since", "ul_decay")
UL_LABELS = ("HIT15", "TP10")
UL_KINDS = ("B", "C", "P")
UL_ARMS = tuple(f"{L}_UL{x}" for L in UL_LABELS for x in UL_KINDS)
UL_RENAME = {"ul_past_n": "ul_past250_n"}
PERM_SEED = 20261018
for _L in UL_LABELS:
    NAMES[f"{_L}_ULB"] = f"{_L}+UL 7列・探索し直し"
    NAMES[f"{_L}_ULC"] = f"{_L}+UL 7列・{_L} のパラメータ"
    NAMES[f"{_L}_ULP"] = f"{_L}+UL 対照（入れ替え）"


def arm_spec(arm: str) -> dict:
    """腕 → (ラベル, 列の組, パラメータの出どころ, 探索するか)。"""
    if arm in ARMS:
        return {"label": arm, "colset": "base", "params_from": arm, "tune": True}
    L, kind = arm.rsplit("_UL", 1)
    if L not in UL_LABELS or kind not in UL_KINDS:
        raise KeyError(f"知らない腕: {arm}")
    return {"label": L, "colset": "perm" if kind == "P" else "ul", "params_from": arm if kind == "B" else L,
            "tune": kind == "B"}
TARGET = LT.TAKE_PROFIT                  # 10.0
HOLD = LT.HOLD                           # 20
OUTCOME = "tp10c"                        # 主の物差し（%）
OUTCOME2 = "ret_o1_20"                   # 参考（比。表では % に直す）
TOP_QS = (0.05, 0.10)
MAIN_Q = 0.05
THR_PCT = 95
MONTHLY_N = 2
Z_MIN = 1.64
MIN_WINDOW_ROWS = E68.MIN_WINDOW_ROWS
SHIFTS = "0,2,4"
TAG = "e102"


# --------------------------------------------------------------------------- #
# ラベル
# --------------------------------------------------------------------------- #
def relabel(df: pd.DataFrame, spec: dict) -> pd.Series:
    """定義 spec のラベル（1/0。現行のラベルが決まらない行は NaN）。"""
    determined = df["label"].notna()
    if spec["kind"] == "sigma":
        cfg = replace(B.DEFAULT_RISE, end_ratio=(B.END_RATIO if spec["end"] else None), require_uptrend=bool(spec["trend"]))
        y = B.rise_label_for_k(df, float(spec["k"]), cfg)
        return y.where(determined)
    if spec["kind"] == "tp":
        k = E83.tp_key(float(spec["target"]))
        hit = df[f"hit{k}"].astype(bool)
        ok = determined & df["c20"].notna()
        return hit.astype(float).where(ok)
    raise KeyError(f"知らないラベルの種類: {spec}")


def build_labels(df: pd.DataFrame, arms=ARMS) -> tuple[pd.DataFrame, list]:
    rows = []
    for a in arms:
        y = relabel(df, LABEL_DEFS[a])
        df[f"y_{a}"] = y
        rows.append({"arm": a, "pos_rate": float(y.mean()), "n_pos": int(y.sum()), "n": int(y.notna().sum())})
    return df, rows


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #
def attach_ul(df: pd.DataFrame, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """買ったあとのストップ高 ul_fwd（結果。特徴量にしない）と、ストップ高の履歴の7列（特徴量の候補）を日足から付ける。"""
    if bars is None:
        bars = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=500), extra=("H", "L", "C"))
    flags = PL.flag_limits(bars)
    ev = PL.limit_events(PL.code_arrays(flags), df[["Code", "Date"]], hold=HOLD, past=250).rename(columns=UL_RENAME)
    cols = ["ul_fwd"] + list(UL_FEATURES)
    out = df.drop(columns=[c for c in cols if c in df.columns])
    return out.merge(ev[["Code", "Date"] + cols], on=["Code", "Date"], how="left")


def prepare(rebuild: bool = False):
    pf, ps = E68.fpath("frame.parquet"), E68.fpath("stamp.json")
    if not rebuild and os.path.exists(pf) and os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            stamp = json.load(fh)
        df = pd.read_parquet(pf)
        log(f"[data] 1回目に作った表を使う（{stamp['built_utc']} 作成・{stamp['rows']:,}行・〜{stamp['date_max']}）")
        if any(c not in df.columns for c in UL_FEATURES):
            # 1回目の表にはストップ高の履歴の7列が無い。同じ行に足すだけなので表の版（built_utc）は変えない
            # （保存済みの探索・OOF はそのまま使える）
            df["Date"] = pd.to_datetime(df["Date"])
            df["Code"] = df["Code"].astype(str)
            df = attach_ul(df)
            df.to_parquet(pf, index=False)
            stamp["ul_added_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
            with open(ps, "w", encoding="utf-8") as fh:
                json.dump(stamp, fh, ensure_ascii=False, indent=1, default=float)
            log(f"[data] 保存済みの表にストップ高の履歴の7列を足した（版は {stamp['built_utc']} のまま）")
        return df, stamp
    os.makedirs(E68.OOF_DIR, exist_ok=True)
    t0 = time.time()
    df = lab.frame(rebuild=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df[df["label"].notna()].reset_index(drop=True)
    need = ["vol_20d", "future_rise", "end_level", "uptrend_end", OUTCOME2]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss}")
    y0 = relabel(df, LABEL_DEFS["CUR12"])
    diff = int((y0 != df["label"].astype(float)).sum())
    if diff:
        raise SystemExit(f"1.2σ の3条件の引き直しが本番のラベルと {diff}件違う。合成の式がずれている")
    log(f"[data] lab.frame {len(df):,}行（ラベルの決まる行）/ 1.2σ の引き直しは本番のラベルと全行一致 / {time.time()-t0:.0f}秒")
    t0 = time.time()
    bars = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=500), extra=("H", "L", "C"))
    df, chk = E83.attach_exits(df, bars, targets=(TARGET,), hold=HOLD)
    df = attach_ul(df, bars)
    df, tab = build_labels(df)
    log(f"[data] 出口（20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達 {chk['hit_share'][E83.tp_key(TARGET)]:.1f}%）と"
        f"ストップ高、{len(ARMS)}本のラベルを付けた / {time.time()-t0:.0f}秒")
    stamp = {"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rows": int(len(df)),
             "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
             "exit_check": {k: (v if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()}) for k, v in chk.items()},
             "labels": tab}
    df.to_parquet(pf, index=False)
    with open(ps, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, ensure_ascii=False, indent=1, default=float)
    return df, stamp


def label_report(df: pd.DataFrame) -> dict:
    """ラベルごとの正例率・重なり・正例の収益（件数と割合だけ）。"""
    print(f"\n■ 1. ラベル（{len(df):,}行・{df['Date'].min().date()} 〜 {df['Date'].max().date()}。出口 = +10% の指値・届かなければ"
          f"20営業日目の終値）")
    print(f"  {'腕':<26}{'正例率':>7}{'正例':>7}  {BASE}と同じ  {BASE}の正例を含む  正例の tp10c / ret20 平均  正例の買ってからストップ高  "
          f"正例の −10%割れ")
    out = {}
    yb = df[f"y_{BASE}"]
    tp = pd.to_numeric(df[OUTCOME], errors="coerce")
    r20 = pd.to_numeric(df[OUTCOME2], errors="coerce") * 100
    ul = pd.to_numeric(df["ul_fwd"], errors="coerce")
    for a in ARMS:
        y = df[f"y_{a}"]
        m = (y == 1) & tp.notna()
        same = float((y == yb).mean())
        cover = float(((y == 1) & (yb == 1)).sum() / max(1, (yb == 1).sum()))
        out[a] = {"pos_rate": float(y.mean()), "n_pos": int(y.sum()), "same_as_base": same, "covers_base_pos": cover,
                  "pos_tp10c": float(tp[m].mean()), "pos_ret20": float(r20[m].mean()), "pos_ul_fwd": float(ul[m].mean()),
                  "pos_bad10": float((tp[m] < -10).mean())}
        print(f"  {NAMES[a]:<26}{y.mean()*100:>6.1f}%{int(y.sum()):>7,}  {same*100:>8.1f}%  {cover*100:>12.1f}%  "
              f"{tp[m].mean():>+6.2f}% / {r20[m].mean():>+6.2f}%  {ul[m].mean()*100:>18.1f}%  {(tp[m] < -10).mean()*100:>8.1f}%")
    print(f"  全行: tp10c {tp.mean():+.2f}%・ret20 {r20.mean():+.2f}%・+10% に届く {float(df[f'hit{E83.tp_key(TARGET)}'].mean())*100:.1f}%・"
          f"買ってからストップ高 {ul.mean()*100:.1f}%")
    return out


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #
EVAL_COLS = [OUTCOME, "hit10", "ul_fwd", "vol_20d"]


def attach_eval(o: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Code", "Date"] + [c for c in EVAL_COLS if c in df.columns and c not in o.columns] + [f"y_{a}" for a in ARMS]
    return o.merge(df[cols].drop_duplicates(["Code", "Date"]), on=["Code", "Date"], how="left")


def stats_of(sel: pd.DataFrame) -> dict:
    tp = pd.to_numeric(sel[OUTCOME], errors="coerce")
    r20 = pd.to_numeric(sel[OUTCOME2], errors="coerce") * 100
    ul = pd.to_numeric(sel["ul_fwd"], errors="coerce") if "ul_fwd" in sel.columns else pd.Series(dtype=float)
    nan = float("nan")
    return {"n": int(len(sel)), "tp": float(tp.mean()) if tp.notna().any() else nan,
            "ret20": float(r20.mean()) if r20.notna().any() else nan,
            "hit": float(sel["hit10"].astype(float).mean() * 100) if "hit10" in sel.columns and len(sel) else nan,
            "bad": float((tp < -10).mean() * 100) if tp.notna().any() else nan,
            "ul": float(ul.mean() * 100) if len(ul) and ul.notna().any() else nan,
            "pos_base": float(pd.to_numeric(sel[f"y_{BASE}"], errors="coerce").mean() * 100) if f"y_{BASE}" in sel.columns else nan}


def top_within(o: pd.DataFrame, q: float) -> pd.DataFrame:
    """窓の中でスコアが上位 q の行（(1−q) 分位以上）の成績。MIN_WINDOW_ROWS 未満の窓は落とす。"""
    rows = []
    for f, g in o.groupby("fold"):
        if len(g) < MIN_WINDOW_ROWS:
            continue
        s = g["score"].to_numpy(dtype=float)
        rows.append({"fold": int(f), **stats_of(g[s >= np.quantile(s, 1.0 - q)])})
    return pd.DataFrame(rows)


def threshold_prior(o: pd.DataFrame, pct: float = THR_PCT) -> pd.DataFrame:
    """前の窓のスコアの分布の上位 (100−pct)% を閾値にして選ぶ（運用の形。参照が500行未満の窓は対象外）。"""
    rows = []
    for f in sorted(o["fold"].unique()):
        ref = o.loc[o["fold"] < f, "score"].to_numpy(dtype=float)
        if len(ref) < 500:
            continue
        cur = o[o["fold"] == f]
        sel = cur[cur["score"].to_numpy(dtype=float) > np.percentile(ref, pct)]
        if len(sel):
            rows.append({"fold": int(f), **stats_of(sel)})
    return pd.DataFrame(rows)


def monthly(o: pd.DataFrame, n: int = MONTHLY_N) -> pd.DataFrame:
    """各月のスコア上位 n 件を、窓ごとにまとめた成績。"""
    picks = E78.monthly_top(o, n)
    rows = [{"fold": int(f), **stats_of(g)} for f, g in picks.groupby("fold")]
    return pd.DataFrame(rows)


FORMS = (("top05", f"窓の中の上位5%", lambda o: top_within(o, 0.05)),
         ("top10", "窓の中の上位10%", lambda o: top_within(o, 0.10)),
         ("thr95", "前の窓の分布の上位5%（閾値）", lambda o: threshold_prior(o, THR_PCT)),
         ("month2", f"月に{MONTHLY_N}銘柄", lambda o: monthly(o, MONTHLY_N)))
MAIN_FORM = "top05"


def paired(base: pd.DataFrame, cur: pd.DataFrame, col: str = "tp") -> dict:
    """窓ごとの差（cur − base）の平均・SE・上の窓・z。(shift, fold) でそろえる。"""
    m = base.merge(cur, on=["shift", "fold"], suffixes=("_b", "_c"))
    d = (m[f"{col}_c"] - m[f"{col}_b"]).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    n = int(len(d))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "wins": 0, "z": float("nan"), "per_window": []}
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(d.mean())
    return {"n": n, "mean": mean, "se": se, "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "z": float(mean / se) if n > 1 and se > 0 else float("nan"), "per_window": [float(x) for x in d]}


def judge(d: dict) -> dict:
    n = d.get("n", 0)
    need = n // 2 + 1
    out = {"n": n, "need": need, "positive": bool(n and d["mean"] > 0), "majority": bool(n and d["wins"] >= need),
           "z_ok": bool(n and np.isfinite(d["z"]) and d["z"] >= Z_MIN)}
    out["pass"] = out["positive"] and out["majority"] and out["z_ok"]
    return out


def fmt_pair(d: dict) -> str:
    if not d.get("n"):
        return "—"
    return f"{d['mean']:+.2f} ± {d['se']:.2f}pt（上の窓 {d['wins']}/{d['n']}・z {d['z']:+.2f}）"


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, cols: list, cutoff, n_trials: int, stamp: dict, shifts: list, compute: bool = True,
        arms: tuple = ARMS) -> dict | None:
    """腕ごとに（必要なら）探索し、本番と同じ OOF（ずらし0・種42）と、ずらし × 種3つの OOF を作る。保存済みは読む。"""
    colsets = {"base": list(cols), "ul": list(cols) + list(UL_FEATURES), "perm": list(cols) + list(UL_FEATURES)}
    perm = None
    frames = {}
    for a in arms:
        sp = arm_spec(a)
        if sp["colset"] == "perm":
            if perm is None:
                from e44_shortsale import permuted
                perm = permuted(df, list(UL_FEATURES), seed=PERM_SEED)
            f = perm.copy()
        else:
            f = df.copy()
        f["label"] = f[f"y_{sp['label']}"]
        f.attrs["built_utc"] = stamp["built_utc"]
        frames[a] = f
    recs, params = {}, {}
    for a in arms:
        sp = arm_spec(a)
        if not sp["tune"]:
            continue
        recs[a] = E68.tune(ALGO, a, frames[a], colsets[sp["colset"]], cutoff, n_trials, stamp, compute)
        if recs[a] is None:
            return None
        params[a] = recs[a]["params"]
    for a in arms:
        src = arm_spec(a)["params_from"]
        if src not in params:
            return None                      # 元のラベルの探索がまだ
        params[a] = params[src]
    prod = {}
    for a in arms:
        sp = arm_spec(a)
        o = E68.oof(ALGO, a, frames[a], colsets[sp["colset"]], params[a], 0, (E68.PROD_SEED,), compute)
        if o is None:
            return None
        prod[a] = attach_eval(o, df)
    shifted = {}
    for sh in shifts:
        for a in arms:
            sp = arm_spec(a)
            o = E68.oof(ALGO, a, frames[a], colsets[sp["colset"]], params[a], sh, E68.SEEDS[ALGO], compute)
            if o is None:
                return None
            shifted[(sh, a)] = attach_eval(o, df)
    return {"recs": recs, "params": params, "prod": prod, "shifted": shifted, "arms": tuple(arms)}


def form_tables(res: dict, shifts: list) -> dict:
    """選び方 × 腕 → 窓ごとの成績（shift, fold 付き）。"""
    out = {}
    arms = res.get("arms", ARMS)
    for key, _, fn in FORMS:
        out[key] = {}
        for a in arms:
            parts = []
            for sh in shifts:
                t = fn(res["shifted"][(sh, a)])
                if len(t):
                    t = t.assign(shift=sh)
                    parts.append(t)
            out[key][a] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["shift", "fold", "n", "tp"])
    return out


def report(res: dict, df: pd.DataFrame, shifts: list, stamp: dict) -> dict:
    arms = tuple(res.get("arms", ARMS))
    ycols = [f"y_{a}" for a in ARMS]
    summary = {"data": {k: v for k, v in stamp.items() if k != "labels"}, "labels": label_report(df), "tuning": {},
               "production_oof": {}, "forms": {}, "judge": {}, "judge_ul": {}, "per_window": {}, "arms": list(arms)}

    print(f"\n■ 2. 探索の CV（{E68.CV_SCHEME} 5分割・自分のラベル。楽観側に出る。探索しない腕は元のラベルの値）と本番と同じ OOF"
          f"（ずらし0・種42）の PR-AUC を各ラベルで")
    print(f"  {'腕':<30}{'CV PR':>8}{'±SD':>8}{'CV ROC':>8}{'秒':>6}  " + "  ".join(f"[{a}] PR/リフト" for a in ARMS))
    for a in arms:
        src = a if a in res["recs"] else arm_spec(a)["params_from"]
        cv = res["recs"][src]["_cv"]
        summary["tuning"][a] = {k: cv.get(k) for k in ("mean_pr_auc", "std", "mean_roc_auc", "fold_scores")}
        summary["tuning"][a]["params"] = res["params"][a]
        summary["tuning"][a]["tuned"] = a in res["recs"]
        cl = E78.cross_label(res["prod"][a], ycols)
        summary["production_oof"][a] = {c: cl[c] for c in ycols}
        print(f"  {NAMES[a]:<30}{cv['mean_pr_auc']:>8.4f}{cv['std']:>8.4f}{cv['mean_roc_auc']:>8.4f}"
              f"{res['recs'][src].get('_seconds', 0) if a in res['recs'] else 0:>6}  "
              + "  ".join(f"{cl[f'y_{b}']['pr']:.4f}/{cl[f'y_{b}']['lift']:.2f}x" for b in ARMS))
    for a in arms:
        if a in res["recs"]:
            print(f"  {NAMES[a]:<30}パラメータ {E68.short(res['params'][a])}")

    tables = form_tables(res, shifts)
    print(f"\n■ 3. 選び方ごとの成績（ずらし {'/'.join(map(str, shifts))}か月 × 種3つの平均。出口 = +10% の指値・届かなければ20日目の終値。"
          f"差は {BASE} に対する窓ごとの差）")
    for key, name, _ in FORMS:
        print(f"  ◆ {name}: 1窓の件数 / tp10c / ret20 / +10% 到達 / −10%割れ / 買ってからストップ高 / {BASE}の正例率 ｜ tp10c の差 ｜ ret20 の差"
              f"（差は {BASE} に対して。UL の腕は元のラベルの腕に対して）")
        base = tables[key][BASE]
        summary["forms"][key] = {}
        for a in arms:
            ref = BASE if a in ARMS else arm_spec(a)["label"]
            base = tables[key][ref]
            t = tables[key][a]
            if not len(t):
                print(f"    {NAMES[a]:<26}（窓なし）")
                continue
            d_tp = paired(base, t, "tp") if a != BASE else {}
            d_r = paired(base, t, "ret20") if a != BASE else {}
            if a != BASE and not len(base):
                d_tp, d_r = {"n": 0}, {"n": 0}
            summary["forms"][key][a] = {"n_per_window": float(t["n"].mean()), "tp": float(t["tp"].mean()), "ret20": float(t["ret20"].mean()),
                                        "hit": float(t["hit"].mean()), "bad": float(t["bad"].mean()), "ul": float(t["ul"].mean()),
                                        "pos_base": float(t["pos_base"].mean()), "windows": int(len(t)),
                                        "diff_tp": {k: v for k, v in d_tp.items() if k != "per_window"},
                                        "diff_ret20": {k: v for k, v in d_r.items() if k != "per_window"}}
            print(f"    {NAMES[a]:<30}{t['n'].mean():>5.0f}件 {t['tp'].mean():>+6.2f}% {t['ret20'].mean():>+6.2f}% {t['hit'].mean():>5.1f}% "
                  f"{t['bad'].mean():>5.1f}% {t['ul'].mean():>5.1f}% {t['pos_base'].mean():>5.1f}%（{len(t)}窓）"
                  + (f" ｜ {fmt_pair(d_tp)} ｜ {fmt_pair(d_r)}" if a != BASE else ""))

    print(f"\n■ 4. 判定（窓の中の上位5% の tp10c の平均。{BASE} に対する窓ごとの差）: 平均 > 0 / 上の窓が過半 / z ≥ {Z_MIN}")
    base = tables[MAIN_FORM][BASE]
    for a in ARMS:
        if a == BASE:
            continue
        d = paired(base, tables[MAIN_FORM][a], "tp")
        j = judge(d)
        summary["judge"][a] = {**j, "diff": {k: v for k, v in d.items() if k != "per_window"}}
        ref = "（参考）" if a not in JUDGED else ""
        print(f"  {NAMES[a]:<26}{fmt_pair(d)}  平均>0 {'○' if j['positive'] else '×'} / 上の窓 {d.get('wins', 0)}/{d.get('n', 0)}"
              f"（{j['need']} 以上）{'○' if j['majority'] else '×'} / z {'○' if j['z_ok'] else '×'} → "
              f"{'リターンが上がる（支持）' if j['pass'] else '支持しない'}{ref}")
    passed = [a for a in JUDGED if summary["judge"].get(a, {}).get("pass")]
    print("  → " + (f"支持した腕: {', '.join(passed)}。本番に入れる案を運用者に出す" if passed else "支持した腕は無い。この線は閉じる"))

    ul_done = [L for L in UL_LABELS if all(f"{L}_UL{x}" in arms for x in UL_KINDS)]
    if ul_done:
        print(f"\n■ 4b. 判定（ストップ高の履歴の7列。窓の中の上位5% の tp10c。元のラベルの腕に対する窓ごとの差）: "
              "C−A の平均が正 / 対照 P−A を上回る / C が上の窓が過半")
        for L in ul_done:
            base_t = tables[MAIN_FORM][L]
            d_c = paired(base_t, tables[MAIN_FORM][f"{L}_ULC"], "tp")
            d_p = paired(base_t, tables[MAIN_FORM][f"{L}_ULP"], "tp")
            d_b = paired(base_t, tables[MAIN_FORM][f"{L}_ULB"], "tp")
            n = d_c.get("n", 0)
            need = n // 2 + 1
            j = {"n": n, "need": need, "positive": bool(n and d_c["mean"] > 0),
                 "beats_placebo": bool(n and d_c["mean"] > d_p["mean"]), "majority": bool(n and d_c["wins"] >= need)}
            j["pass"] = j["positive"] and j["beats_placebo"] and j["majority"]
            # 参考: 自分のラベルで測った窓の PR-AUC
            pr = {}
            for kind in ("B", "C", "P"):
                parts_b, parts_c = [], []
                for sh in shifts:
                    wb = E78.windows_cross(res["shifted"][(sh, L)], f"y_{L}").assign(shift=sh)
                    wc = E78.windows_cross(res["shifted"][(sh, f"{L}_UL{kind}")], f"y_{L}").assign(shift=sh)
                    parts_b.append(wb)
                    parts_c.append(wc)
                m = pd.concat(parts_b).merge(pd.concat(parts_c), on=["shift", "fold"], suffixes=("_b", "_c"))
                d = (m["pr_c"] - m["pr_b"]).to_numpy(dtype=float)
                pr[kind] = {"mean": float(d.mean()) if len(d) else float("nan"),
                            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                            "wins": int((d > 0).sum()), "n": int(len(d))}
            summary["judge_ul"][L] = {**j, "diff_c": {k: v for k, v in d_c.items() if k != "per_window"},
                                      "diff_p": {k: v for k, v in d_p.items() if k != "per_window"},
                                      "diff_b": {k: v for k, v in d_b.items() if k != "per_window"}, "pr": pr}
            print(f"  {L}: C−A {fmt_pair(d_c)} / P−A {fmt_pair(d_p)} / B−A {fmt_pair(d_b)}")
            print(f"       平均>0 {'○' if j['positive'] else '×'} / 対照を上回る {'○' if j['beats_placebo'] else '×'} / 上の窓 "
                  f"{d_c.get('wins', 0)}/{n}（{need} 以上）{'○' if j['majority'] else '×'} → "
                  f"{'列の効果あり' if j['pass'] else '列の効果なし'}")
            print("       参考: 自分のラベルの窓の PR-AUC " + " / ".join(
                f"{k}−A {v['mean']:+.4f} ± {v['se']:.4f}（{v['wins']}/{v['n']}）" for k, v in pr.items()))

    print(f"\n■ 5. 窓ごとの値（窓の中の上位5% の tp10c。{BASE} と各腕、差）")
    for sh in shifts:
        b = tables[MAIN_FORM][BASE]
        b = b[b["shift"] == sh].sort_values("fold")
        if not len(b):
            continue
        print(f"  ずらし{sh}か月（{len(b)}窓）")
        print(f"    {'窓':>2}{'件数':>6}" + "".join(f"{a:>10}" for a in ARMS) + "".join(f"{a + '−' + BASE:>13}" for a in ARMS if a != BASE)
              + ("  （UL の腕は summary.json / by_window.csv）" if len(arms) > len(ARMS) else ""))
        for _, rb in b.iterrows():
            vals, diffs = [], []
            for a in ARMS:
                t = tables[MAIN_FORM][a]
                r = t[(t["shift"] == sh) & (t["fold"] == rb["fold"])]
                v = float(r["tp"].iloc[0]) if len(r) else float("nan")
                vals.append(v)
                if a != BASE:
                    diffs.append(v - float(rb["tp"]))
            print(f"    {int(rb['fold']):>2}{int(rb['n']):>6}" + "".join(f"{v:>+9.2f}%" for v in vals) + "".join(f"{d:>+12.2f}pt" for d in diffs))
    rows = []
    for key, _, _ in FORMS:
        for a in arms:
            t = tables[key][a]
            for _, r in t.iterrows():
                rows.append({"form": key, "arm": a, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}})
    summary["per_window"] = rows
    pd.DataFrame(rows).to_csv(E68.path("by_window.csv"), index=False)
    with open(E68.path("summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    print("\n■ まとめ: " + " / ".join(
        f"{a} {fmt_pair(summary['judge'][a]['diff']) if a in summary['judge'] else '—'}"
        f"{'（支持）' if summary['judge'].get(a, {}).get('pass') else ''}" for a in ARMS if a != BASE)
        + ("" if not summary["judge_ul"] else " ｜ UL の列: " + " / ".join(
            f"{L} C−A {fmt_pair(v['diff_c'])}{'（効果あり）' if v['pass'] else ''}" for L, v in summary["judge_ul"].items())))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験102: 目的変数の定義（到達のみ・+10% の指値）を変えた LightGBM の高スコア帯の収益")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--shifts", default=SHIFTS)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--budget-min", type=float, default=290.0)
    ap.add_argument("--dry", action="store_true", help="ラベルの付き方だけ出して終わる")
    ap.add_argument("--with-ul", action="store_true", help="ストップ高の履歴の7列を足した腕（HIT15・TP10 の B/C/P）も計算する")
    args = ap.parse_args(argv)
    arms = ARMS + (UL_ARMS if args.with_ul else ())
    E68.TAG = args.tag
    E68.RUN = f"{args.tag}_lgbm"
    E68.DEADLINE = time.time() + args.budget_min * 60 if args.budget_min > 0 else None
    E68.TIMED_OUT = False
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    cols = F.columns(F.DEFAULT_PRESET)

    print("=" * 78)
    print("実験102 目的変数の定義を変えたモデル（到達のみ・+10% の指値）の高スコア帯の収益（LightGBM・腕ごとに探索し直し・本番と同じ OOF・32窓）")
    print("  腕: " + " / ".join(NAMES[a] for a in arms) + f" / 列 {F.DEFAULT_PRESET} {len(cols)} / 探索 {args.n_trials}試行 × {E68.N_SPLITS}分割"
          f"（{E68.CV_SCHEME}）/ ずらし {shifts}か月 / 種 {E68.SEEDS[ALGO]} / 判定 = 窓の中の上位5% の tp10c（平均>0・過半・z≥{Z_MIN}）/ "
          f"保存 {E68.RUN}_*")
    print("=" * 78)

    df, stamp = prepare(args.rebuild)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df.attrs["built_utc"] = stamp["built_utc"]
    if args.dry:
        label_report(df)
        return 0
    cutoff, _, _ = T.holdout_bounds(df["Date"], T.HOLDOUT_MONTHS, T.EMBARGO_DAYS)
    print(f"  行 {len(df):,}（{df['Date'].min().date()} 〜 {df['Date'].max().date()}）。探索の期間（〜{cutoff.date()}）では "
          f"{(df['Date'] <= cutoff).sum():,}行")
    t0 = time.time()
    res = run(df, cols, cutoff, args.n_trials, stamp, shifts, arms=arms)
    log(f"[{ALGO}] {'そろった' if res else '途中（続きは次の回）'} / {time.time()-t0:.0f}秒")
    if res is None:
        label_report(df)
        log("時間の上限で止めた。同じ引数でもう1回回すと、保存済みの探索・out-of-fold の続きから")
    else:
        report(res, df, shifts, stamp)
    log(f"記録: {E68.OOF_DIR}/{E68.RUN}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
