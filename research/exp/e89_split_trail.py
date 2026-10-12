#!/usr/bin/env python3
"""
実験89: 分割決済＋トレーリングストップ。「6本とも95以上」（A15）の銘柄を2単位（200株）買い、前半1単位は +10% の指値、
後半1単位は買った後の高値から W% 下げたら売る（上限 40営業日）と、今の出口よりよいか。

運用者の問い（2026-10-13、音声）「200株買って100株は10%で指値を入れておいて、100株はそのまま持ち続ける。高値から
下がったら、高値からマイナス何パーかのところで、もう100株も利確する」。きっかけは実験83 の「+10% の指値は勝率と中央値を
上げるが、平均は指値なしより約1pt 下がる」（docs/MODEL_ADOPTION_RULES.md §40）と、ドライランで +10% で売った銘柄が
その日のうちに決算で +19% まで上げたこと。実験41 §9・§11 では、選ばれた銘柄は +10% に届いた後も伸び、狭い逆指値
（−5% / −8%）は成績を下げ、−15% の逆指値の発動は取引の約5% だった。幅 W は 7 / 10 / 15% の3水準（運用者「そのままで
OK」）、後半の上限は 40営業日（「40日まででお願いします」）。

規則と出口（先に決めたもの。結果を見てから動かさない）
------------------------------------------------
  A15  1.2σ の lgbm・xgb・cat と 1.5σ の lgbm_k15・xgb_k15・cat_k15 の6本とも 95以上（実験82・83 の A15 と同じ）
  R0   1.2σ の3本だけ 95以上（参考。実験83 と同じ）
  買い 翌営業日の寄りで2単位（1日目 = 買った日。足は銘柄ごとの並びで数える。live_track.forward と同じ）
  前半1単位 tp10c（実験83 と同じ）: 1〜20営業日目の高値が買値の 1.10倍以上なら +10% ちょうど（寄りで超えて始まっても
           +10%。控えめな側）。届かなければ 20営業日目の終値
  後半1単位 trail<W>（W = 7 / 10 / 15）: 買った後の高値の最高値（ハイウォーターマーク。買値から始める）から W% 下げたら
           売る。上限 40営業日（掛からなければ 40営業日目の終値）
    約定（実験41 の逆指値と同じ保守側。tests/test_e89_split_trail.py で固定）
      k日目の線 =（k−1日目までの高値の最高値。1日目は買値）×（1 − W%）。k日目の高値で線を上げるのは k+1日目から
                  （日足では高値と安値の順番が分からないので、同じ日に高値を更新してから線まで下げた分は数えない）
      寄りの時点で線以下（窓を開けて割った）なら寄値で約定（損は W% より大きくなる）。場中に安値が線に触れたら線ちょうど
      売買が成立しなかった日（値が欠測）は判定しない（高値の最高値も変えない）
  1取引の収益 split<W> = (tp10c + trail<W>) / 2（2単位の平均）。手数料・税・すべりは含まない
  比べる出口 tp10c（全部 +10%。今の運用 = live_track の TAKE_PROFIT 10・FALLBACK close）、tp20a（+20% の指値・届かなければ
           20営業日目の5日平均。2026-10-12 までの運用）、c20（指値なし・20日目の終値）、c40（指値なし・40日目の終値）
  比べる行 40営業日の結果が出ている行（選定日が日足の最後の日の 40営業日より前）で、どの出口の収益も欠けていない行
           （40日目の足が無い・終値が欠測の行は全部の出口から外し、件数をログに出す）。どの出口も同じ行で比べる
  百分位   前向き（実験79 の window_split。テスト窓より前の窓の分布での位置。丸めは画面と同じ小数1桁）
  窓の切り方 ずらし0か月 = 本番の窓・Release の out-of-fold（実験83 と同じ行）。ずらし2・4か月 = 窓の境界をずらして6本を
           学習し直した out-of-fold（実験41 §10・実験26 と同じ folds_for。列・パラメータ・種は本番と同じ: 基準の lgbm は
           train_production と同じ lgbm_params.json、ほかは train_multi と同じ multi_params.json・tuning_multi.SEED）。
           ずらした窓も回すときは、参考に、ずらし0か月も同じ作り方で学習し直す（0r。作り方の違いと、ずらしの違いを分けるため）
  判定（主） ずらし0か月の A15 で split<W> − tp10c（同じ行）を窓ごとに。平均が正・並べ替え p < 0.05・上の窓が過半
           （実験80・82・83 と同じ3つ）。並べ替えは、窓ごとに行の差の符号をでたらめに反転する（同じ行の2つの出口の
           入れ替え。片側）。銘柄単位（同じ窓・同じ銘柄の行を1つにまとめる）の p も出す。
           W ごとに出し、W はここから選ばない。ずらし2・4か月でも差の平均が正で上の窓が過半かを添える。3水準とも
           ずらし0か月で通り、ずらし2・4か月でも同じ向きなら「幅によらず今の出口より上」、一部だけなら「幅しだい」
           幅の安定（見るだけ）: 切り方 0・2・4か月のうち差の平均が正で上の窓が過半だった数の多い順、同じなら
           いちばん悪い切り方の差の平均の高い順に並べる
  参考（判定に使わない）
    - 同じ行で split<W> − c20 / c40 / tp20a、trail<W> − c40（後半だけ）、tp10c − c20（実験83 の再現）
    - トレーリングの発動率・発動までの日数・発動した単位の平均・寄りで割った割合・−10%割れ
    - 3枠の模擬（運用の通り: 発火の数は問わず上位2件・先着順。live_track の SKIP_BREAKS / TOP_K / SLOTS。並べ方は
      規則のモデルの最小の百分位の高い順で、A15 は今の画面と同じ 1.5σ の3本、R0 は 1.2σ の3本。同点は lgbm のスコア順）。
      資金は3単位。1単位の出口は1銘柄に1単位（資金の 1/3）、分割は1銘柄に2単位（資金の 2/3）を使うので、同時に持てる銘柄が
      減る（集中度。前半が売れた日の翌営業日から1単位が空く）。比べに、資金を6単位（3銘柄 × 2単位。1銘柄に資金の 1/3）に
      した分割も出す。月の収益 = 単位ごとの収益の合計 ÷ 単位数 ÷ 月数（実験83 と同じ。複利にしない）。年率は、約定のたびに
      その時点の資金（現金 + 持っている玉の簿価）を単位で割って入れ、複利で回した月末の資金から（実験41 §11 と同じ作り。
      持っている玉の含み損益は売るまで数えないので、最大DD は売った後の資金で測る）。最大DD・月利の平均 /
      中央 / 最悪・稼働率（使っている単位の割合）・持っている銘柄数の平均・2銘柄以上を持つ日の割合も
    - 窓ごとの A15（split<W>・tp10c）
    - 実験83 と同じ行（20日目まで）の A15 × tp10c（再現の確かめ。実験83 は 333件・+3.59%・勝率 70.9%・到達 43.2%）
  前の結果: 実験83 の A15 × +10% は 1取引 +3.59%（中央 +5.96）、指値なし（c20）+5.12%、+10% − 指値なし −1.12 ± 0.54pt。
  実験41 §9・§11 では選ばれた銘柄の 20日目の平均が、+10% 到達の後も +12.6〜13.1%

公開ログには集計だけ（群ごとの件数・平均・割合、窓ごとの差）。銘柄・日付ごとの値は出さない。

    exp=e89_split_trail.py
    args="--perm 2000"                  （既定。ずらし 0・2・4 か月と、学習し直した 0 か月 = 0r）
    args="--shifts 0 --perm 200"        （軽く回す。Release の out-of-fold だけ。学習し直さない）
    args="--no-retrain0"                （学習し直した 0 か月 = 0r を省く）
  学習し直した out-of-fold は research/_data/oof/e89_<モデル>_sh<月>_<指紋>.parquet に置く（run-experiment.yml の
  actions/cache で次の実行に引き継ぐ。指紋はパラメータ・種・列・データセットの中身（学習の列とラベルの列のハッシュ）から
  作るので、どれかが変われば作り直す）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import models as M  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402

OUTCOME = E79.OUTCOME
TREES = E82.TREES                       # 1.2σ の3本
T15 = E82.T15                           # 1.5σ の3本
SIX = TREES + T15                       # A15 に要る6本
LINE = E82.LINE                         # 95
WIDTHS = (7.0, 10.0, 15.0)              # トレーリングの幅（%）。ここから選ばない
HOLD = LT.HOLD                          # 前半の上限（20営業日）
CAP = 40                                # 後半の上限（40営業日）
TP = 10.0                               # 前半の指値（%）
MAIN_EXIT = "tp10c"
RULES = ("A15", "R0")
RULE_KEY = {"A15": "pmin15", "R0": "pmin3"}     # 3枠の模擬の並べ方（規則のモデルの最小の百分位）
SHIFTS = (0, 2, 4)
UNITS = LT.SLOTS                        # 資金の単位（3）。分割は1銘柄に2単位
N_PERM = E82.N_PERM
SEED = E82.SEED
MIN_GROUP = E82.MIN_GROUP
EPS = LT.LEVEL_EPS
OUT_DIR = E79.OUT_DIR

log = E79.log
summarize = E79.summarize
fmt_s = E79.fmt_s
NAN = float("nan")


def wk(w: float) -> str:
    return f"{int(round(w))}"


WKEYS = tuple(wk(w) for w in WIDTHS)
#: (列, 表示, 売った日の列)。売った日は 1 = 買った日
BASE_EXITS = (
    ("tp10c", "全部 +10%（今の運用）", "d_tp10c"),
    ("tp20a", "+20%・5日平均（10/12 まで）", "d_tp20a"),
    ("c20", "指値なし・20日目の終値", "d_c20"),
    ("c40", "指値なし・40日目の終値", "d_c40"),
)
SPLIT_EXITS = tuple((f"split{k}", f"分割 +10% / 高値−{k}%", None) for k in WKEYS)
TRAIL_EXITS = tuple((f"trail{k}", f"後半だけ 高値−{k}%", f"tday{k}") for k in WKEYS)
EXITS = BASE_EXITS + SPLIT_EXITS + TRAIL_EXITS
EXIT_JA = {c: ja for c, ja, _ in EXITS}
EXIT_COLS = tuple(dict.fromkeys(
    tuple(c for c, _, _ in EXITS) + ("hit10", "hitday10", "hit20", "hitday20", "hold5")
    + tuple(d for _, _, d in BASE_EXITS)
    + tuple(f"{p}{k}" for k in WKEYS for p in ("tday", "tfire", "tgap"))))


def fnum(v) -> float:
    """数に直す（欠測・None・pd.NA は NaN）。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


# --------------------------------------------------------------------------- #
# 後半1単位: 高値からのトレーリング
# --------------------------------------------------------------------------- #

def trail_one(o, h, lo, c, entry: float, w: float) -> tuple:
    """
    1取引の後半1単位。o / h / lo / c は 1日目（買った日）〜上限の日の値（長さ = 上限の日数）。
    戻り値 (収益 %, 売った日（1 = 買った日）, 線で売ったか, 寄りで割って寄値で売ったか)。
    上限の日まで掛からず、上限の日の終値が欠測なら収益は NaN。
    """
    f = 1.0 - w / 100.0
    hwm = float(entry)
    n = len(o)
    for k in range(n):
        line = hwm * f
        ok, hk, lk = o[k], h[k], lo[k]
        if np.isfinite(ok) and ok <= line * (1.0 + EPS):
            return (ok / entry - 1.0) * 100.0, k + 1, True, True
        if np.isfinite(lk) and lk <= line * (1.0 + EPS):
            return (line / entry - 1.0) * 100.0, k + 1, True, False
        if np.isfinite(hk) and hk > hwm:
            hwm = float(hk)
    last = c[n - 1] if n else NAN
    return ((last / entry - 1.0) * 100.0 if np.isfinite(last) else NAN), n, False, False


def trail_exits(bars: pd.DataFrame, keys: pd.DataFrame, widths=WIDTHS, cap: int = CAP) -> pd.DataFrame:
    """
    (Code, Date) ごと（Date は選定日）に、後半1単位の収益 trail<k>（%）・売った日 tday<k>・線で売った印 tfire<k>・寄りで割った
    印 tgap<k> と、40営業日目の終値で売った収益 c40（%）。上限の日の足が無い行（まだ途中・上場廃止など）は全部 NaN。
    bars には AdjO / AdjH / AdjL / AdjC が要る。
    """
    need = keys[["Code", "Date"]].drop_duplicates()
    want = need.groupby("Code")["Date"].apply(list).to_dict()
    b = bars[bars["Code"].isin(want)].sort_values(["Code", "Date"])
    ks = [wk(w) for w in widths]
    out = []
    for code, g in b.groupby("Code", sort=False):
        dates = g["Date"].to_numpy(dtype="datetime64[ns]")
        o = g["AdjO"].to_numpy(dtype=float)
        h = g["AdjH"].to_numpy(dtype=float)
        lo = g["AdjL"].to_numpy(dtype=float)
        c = g["AdjC"].to_numpy(dtype=float)
        n = len(dates)
        for d in want[code]:
            rec = {"Code": code, "Date": d, "c40": NAN}
            for k in ks:
                rec.update({f"trail{k}": NAN, f"tday{k}": NAN, f"tfire{k}": False, f"tgap{k}": False})
            i = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(d), "ns")))
            if i >= n or dates[i] != np.datetime64(pd.Timestamp(d), "ns") or i + cap >= n:
                out.append(rec)
                continue
            entry = o[i + 1]
            if not (np.isfinite(entry) and entry > 0):
                out.append(rec)
                continue
            sl = slice(i + 1, i + cap + 1)
            if np.isfinite(c[i + cap]):
                rec["c40"] = (c[i + cap] / entry - 1.0) * 100.0
            for w, k in zip(widths, ks):
                r, day, fired, gap = trail_one(o[sl], h[sl], lo[sl], c[sl], entry, w)
                rec.update({f"trail{k}": r, f"tday{k}": float(day), f"tfire{k}": fired, f"tgap{k}": gap})
            out.append(rec)
    cols = ["Code", "Date", "c40"] + [f"{p}{k}" for k in ks for p in ("trail", "tday", "tfire", "tgap")]
    f = pd.DataFrame(out, columns=cols)
    for k in ks:
        f[f"tfire{k}"] = f[f"tfire{k}"].astype(bool)
        f[f"tgap{k}"] = f[f"tgap{k}"].astype(bool)
    return keys[["Code", "Date"]].merge(f, on=["Code", "Date"], how="left")


def attach_all(df: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    出口の列を足す。実験83 の attach_exits（+10% / +20% の指値・c20・hold5・tp20a）に、c40・trail<k>・split<k> と、
    3枠の模擬に使う売った日（d_<出口>。1 = 買った日）を足す。確かめ（chk）も返す。
    """
    out, chk = E83.attach_exits(df, bars, targets=(TP, 20.0))
    tr = trail_exits(bars, out[["Code", "Date"]])
    tr = tr.drop_duplicates(["Code", "Date"])
    out = out.merge(tr, on=["Code", "Date"], how="left")
    hold = float(HOLD)
    out["d_tp10c"] = np.where(out["hit10"].astype(bool), pd.to_numeric(out["hitday10"], errors="coerce"), hold)
    out["d_tp20a"] = np.where(out["hit20"].astype(bool), pd.to_numeric(out["hitday20"], errors="coerce"), hold)
    out["d_c20"] = hold
    out["d_c40"] = float(CAP)
    for k in WKEYS:
        out[f"split{k}"] = (pd.to_numeric(out["tp10c"], errors="coerce")
                            + pd.to_numeric(out[f"trail{k}"], errors="coerce")) / 2.0
        out[f"tfire{k}"] = out[f"tfire{k}"].fillna(False).astype(bool)
        out[f"tgap{k}"] = out[f"tgap{k}"].fillna(False).astype(bool)
    # 確かめ: 幅が狭いほど早く（同じ日か前に）売っている。線で売った単位は −W% より下で売っていない（寄りで割った日を除く）
    order = 0
    for a, b_ in zip(WKEYS, WKEYS[1:]):
        da = pd.to_numeric(out[f"tday{a}"], errors="coerce").to_numpy(dtype=float)
        db = pd.to_numeric(out[f"tday{b_}"], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(da) & np.isfinite(db)
        order += int((da[both] > db[both]).sum())
    below = 0
    for w, k in zip(WIDTHS, WKEYS):
        r = pd.to_numeric(out[f"trail{k}"], errors="coerce").to_numpy(dtype=float)
        m = out[f"tfire{k}"].to_numpy(dtype=bool) & ~out[f"tgap{k}"].to_numpy(dtype=bool) & np.isfinite(r)
        below += int((r[m] < -w - 1e-6).sum())
    chk.update({"with_c40": int(np.isfinite(pd.to_numeric(out["c40"], errors="coerce")).sum()),
                "trail_order_violations": order, "trail_below_line": below})
    return out, chk


# --------------------------------------------------------------------------- #
# 行の表（前向きの百分位・規則の印・出口）
# --------------------------------------------------------------------------- #

def build_rows(df: pd.DataFrame, parts: list, models=SIX) -> pd.DataFrame:
    """
    parts: [(行番号, 丸める前の百分位の行列（列は models の順）)]。1行ごとに 規則の印（A15 / R0）・1.2σ の3本の最小（pmin3）・
    1.5σ の3本の最小（pmin15。丸め後）・lgbm のスコア・実収益（%）・出口の列を持つ表（実験82 の build_frame の6本版）。
    """
    i = {a: k for k, a in enumerate(models)}
    outs = []
    for rows, P in parts:
        R = E82.rounded(P)
        src = df.iloc[rows]
        g = src[["Code", "Date", "window"]].copy().reset_index(drop=True)
        g["ret20"] = pd.to_numeric(src[OUTCOME], errors="coerce").to_numpy(dtype=float) * 100
        g["score"] = src[f"score_{M.BASELINE}"].to_numpy(dtype=float)
        g["pmin3"] = R[:, [i[a] for a in TREES]].min(axis=1)
        g["pmin15"] = R[:, [i[a] for a in T15]].min(axis=1)
        g["R0"] = (R[:, [i[a] for a in TREES]] >= LINE).all(axis=1)
        g["A15"] = (R[:, [i[a] for a in SIX]] >= LINE).all(axis=1)
        for c in EXIT_COLS:
            if c in src.columns:
                g[c] = src[c].to_numpy()
        outs.append(g)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 成績・比べ方
# --------------------------------------------------------------------------- #

def exit_stats(g: pd.DataFrame, col: str, months: float) -> dict:
    """出口 col の成績（%）。前半の指値の到達（hit）は tp10c / split だけ。トレーリングは発動の割合などを足す。"""
    hit = "hit10" if (col == "tp10c" or col.startswith("split")) else ("hit20" if col == "tp20a" else None)
    s = E83.exit_stats(g, col, hit if hit in g.columns else None, months)
    k = col[5:] if col.startswith(("split", "trail")) else None
    if k is not None and f"tfire{k}" in g.columns:
        r = pd.to_numeric(g[col], errors="coerce")
        ok = r.notna().to_numpy()
        fire = g[f"tfire{k}"].to_numpy(dtype=bool)[ok]
        gap = g[f"tgap{k}"].to_numpy(dtype=bool)[ok]
        day = pd.to_numeric(g[f"tday{k}"], errors="coerce").to_numpy(dtype=float)[ok]
        tr = pd.to_numeric(g[f"trail{k}"], errors="coerce").to_numpy(dtype=float)[ok]
        s.update({"fire": float(fire.mean() * 100) if len(fire) else NAN,
                  "gap": float(gap.mean() * 100) if len(gap) else NAN,
                  "fire_day": float(np.nanmean(day[fire])) if fire.any() else NAN,
                  "fire_ret": float(np.nanmean(tr[fire])) if fire.any() else NAN,
                  "keep_ret": float(np.nanmean(tr[~fire])) if (~fire).any() else NAN,
                  "days": float(np.nanmean(day)) if len(day) else NAN})
    return s


def paired_test(g: pd.DataFrame, a: str, b: str, n_perm: int = N_PERM, seed: int = SEED, unit: str = "row",
                min_rows: int = MIN_GROUP) -> dict:
    """
    同じ行で出口 a − 出口 b を窓ごとに平均する（実験83 の paired_by_window と同じ）。並べ替え検定は、窓ごとに行の差の符号を
    でたらめに反転する（同じ行の2つの出口の入れ替え。統計量は窓ごとの平均の平均。片側: 並べ替えた値が実際の値以上だった割合）。
    unit="code" は、同じ窓・同じ銘柄の行を1つ（平均）にまとめてから同じことをする。
    """
    x = g[["window", "Code"]].copy()
    x["d"] = pd.to_numeric(g[a], errors="coerce").to_numpy(dtype=float) - \
        pd.to_numeric(g[b], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x["d"].to_numpy(dtype=float))]
    if unit == "code":
        x = x.groupby(["window", "Code"], as_index=False)["d"].mean()
    groups = [(int(w), y["d"].to_numpy(dtype=float)) for w, y in x.groupby("window") if len(y) >= min_rows]
    d = np.array([v.mean() for _, v in groups], dtype=float)
    s = summarize(d)
    p = NAN
    if groups and n_perm:
        rng = np.random.default_rng(seed)
        tot = np.zeros(n_perm)
        for _, v in groups:
            signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(v)))
            tot += (signs * v).mean(axis=1)
        perm = tot / len(groups)
        p = float((np.sum(perm >= s["mean"] - 1e-12) + 1) / (n_perm + 1))
    return {"a": a, "b": b, "unit": unit, "summary": s, "p": p, "diffs": d.tolist(),
            "windows": [w for w, _ in groups], "rows": int(len(x))}


judge = E82.judge
verdict_text = E82.verdict_text


def same_way(res: dict) -> bool:
    """ずらした窓で「差の平均が正・上の窓が過半」か（p は問わない）。"""
    s = res["summary"]
    return bool(s["n"] and s["mean"] > 0 and s["wins"] >= s["n"] // 2 + 1)


# --------------------------------------------------------------------------- #
# 3枠の模擬（単位で数える）
# --------------------------------------------------------------------------- #

def legs_of(col: str) -> list:
    """出口ごとの単位の組 [(単位数, 収益の列, 売った日の列)]。分割は前半 tp10c と後半 trail<k> の2単位。"""
    if col.startswith("split"):
        k = col[5:]
        return [(1, "tp10c", "d_tp10c"), (1, f"trail{k}", f"tday{k}")]
    if col.startswith("trail"):
        k = col[5:]
        return [(1, col, f"tday{k}")]
    return [(1, col, f"d_{col}")]


def unit_sim(picks: pd.DataFrame, bar_days, legs: list, units: int = UNITS, *, n_days: int | None = None,
             eq_span: tuple | None = None, occ_span: tuple | None = None) -> dict:
    """
    先着順の模擬（live_track.simulate と同じ約束を単位に広げたもの）。資金は units 単位。1銘柄で legs の単位数の合計を使い、
    足りなければ見送る（保有中の玉は切らない）。単位は「売った日の翌営業日」から空く（売った日 = 買った日 + 日数 − 1）。
    picks は選定日・順位の順。収益や売った日が欠測の候補は数えない（日足なし）。
      月の収益 = 単位ごとの収益の合計 ÷ units ÷ 月数（n_days ÷ 21。実験83 と同じ。複利にしない）
      年率 = 約定のたびに その時点の資金（現金 + 持っている玉の簿価）× 単位数 ÷ units を入れ、売ったら戻す複利の資金の
             月末の値から（実験41 §11 と同じ作り）。eq_span（最初と最後の日）の月で数える（出口どうしで同じ月数にする）
      集中度 = occ_span（最初と最後の日）の営業日ごとに、使っている単位・持っている銘柄の数
    """
    cal = [pd.Timestamp(d) for d in bar_days]
    idx = {d: i for i, d in enumerate(cal)}
    need = int(sum(u for u, _, _ in legs))
    free = [-1.0] * units
    cash = 1.0
    open_legs: list = []
    events: list = []
    taken, skipped, nodata = [], 0, 0

    def close_until(t: float) -> None:
        nonlocal cash
        due = sorted((q for q in open_legs if q["free"] <= t), key=lambda q: q["free"])
        for q in due:
            open_legs.remove(q)
            cash += q["cost"] * (1.0 + q["ret"] / 100.0)
            events.append((q["exit"], cash + sum(z["cost"] for z in open_legs)))

    trail_col = legs[-1][1] if legs[-1][1].startswith("trail") else None
    for r in picks.sort_values(["Date", "rank"], kind="mergesort").to_dict("records"):
        di = idx.get(pd.Timestamp(r["Date"]))
        rets = [fnum(r.get(c)) for _, c, _ in legs]
        days = [fnum(r.get(dc)) for _, _, dc in legs]
        if di is None or not all(np.isfinite(v) for v in rets + days):
            nodata += 1
            continue
        buy = di + 1
        if buy + int(max(days)) - 1 >= len(cal):
            nodata += 1
            continue
        close_until(buy)
        empty = [s for s in range(units) if free[s] <= buy]
        if len(empty) < need:
            skipped += 1
            continue
        eq = cash + sum(q["cost"] for q in open_legs)
        j = 0
        pos = []
        for (u, _, _), ret, day in zip(legs, rets, days):
            day = int(day)
            for s in empty[j:j + u]:
                free[s] = buy + day
            j += u
            cost = eq * u / units
            cash -= cost
            q = {"cost": cost, "ret": float(ret), "free": buy + day, "exit": buy + day - 1, "buy": buy, "u": u}
            open_legs.append(q)
            pos.append(q)
        taken.append({"Date": r["Date"], "window": r.get("window"), "buy": buy,
                      "last": max(q["exit"] for q in pos),
                      "ret": float(sum(q["u"] * q["ret"] for q in pos) / need),
                      "units_ret": float(sum(q["u"] * q["ret"] for q in pos)),
                      "legs": [(q["u"], q["buy"], q["exit"]) for q in pos],
                      "hit": float(bool(r.get("hit10", False))) if legs[0][1] == "tp10c" else NAN,
                      "fire": float(bool(r.get(f"tfire{trail_col[5:]}", False))) if trail_col else NAN})
    close_until(math.inf)
    out = {"picks": int(len(picks)), "n": int(len(taken)), "skipped": int(skipped), "nodata": int(nodata),
           "need_units": need, "units": units}
    if not taken:
        return out
    t = pd.DataFrame(taken)
    r = t["ret"].to_numpy(dtype=float)
    months = (max(n_days, 1) / LT.MONTH) if n_days else NAN
    out.update({"mean": float(r.mean()), "median": float(np.median(r)), "win": float((r > 0).mean() * 100),
                "bad10": float((r < -10).mean() * 100), "worst": float(r.min()),
                "se": float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else NAN,
                "per_month": float(len(t) / months) if months == months else NAN,
                "monthly": float(t["units_ret"].sum() / units / months) if months == months else NAN,
                "by_window": {int(w): float(x["ret"].mean()) for w, x in t.groupby("window")}})
    h = pd.to_numeric(t["hit"], errors="coerce")
    if h.notna().any():
        out["hit"] = float(h.mean() * 100)
    fi = pd.to_numeric(t["fire"], errors="coerce")
    if fi.notna().any():
        out["fire"] = float(fi.mean() * 100)
    cal64 = np.array(cal, dtype="datetime64[ns]")

    def bounds(sp):
        a, b = (pd.Timestamp(sp[0]), pd.Timestamp(sp[1])) if sp else (cal[int(t["buy"].min())],
                                                                      cal[int(t["last"].max())])
        lo_ = int(np.searchsorted(cal64, np.datetime64(a, "ns")))
        hi_ = int(np.searchsorted(cal64, np.datetime64(b, "ns"), side="right")) - 1
        return lo_, max(hi_, lo_)

    # 集中度（occ_span の営業日で数える）
    lo, hi = bounds(occ_span)
    used = np.zeros(hi - lo + 1)
    held = np.zeros(hi - lo + 1)
    for legs_ in t["legs"]:
        first, last = min(b for _, b, _ in legs_), max(e for _, _, e in legs_)
        a, b = max(first, lo), min(last, hi)
        if a <= b:
            held[a - lo:b - lo + 1] += 1
        for u, b0, e0 in legs_:
            a, b = max(b0, lo), min(e0, hi)
            if a <= b:
                used[a - lo:b - lo + 1] += u
    out.update({"util": float(used.mean() / units * 100), "held": float(held.mean()),
                "held2": float((held >= 2).mean() * 100), "idle": float((held == 0).mean() * 100),
                "max_weight": float(need / units * 100)})
    # 複利の資金（eq_span の月で数える。最後の玉を売った月より短くはしない）
    lo, hi = bounds(eq_span)
    hi = max(hi, int(t["last"].max()))
    ev = pd.DataFrame(events, columns=["i", "equity"])
    ev["month"] = pd.to_datetime(pd.Series([cal[int(i)] for i in ev["i"]])).dt.to_period("M").to_numpy()
    mons = pd.period_range(cal[lo].to_period("M"), cal[hi].to_period("M"), freq="M")
    eq = ev.groupby("month")["equity"].last().reindex(mons).ffill().fillna(1.0)
    mret = eq / eq.shift(1).fillna(1.0) - 1.0
    nm = len(mons)
    peak = eq.cummax()
    out.update({"months": int(nm), "m_mean": float(mret.mean() * 100), "m_med": float(mret.median() * 100),
                "m_worst": float(mret.min() * 100), "cagr": float((eq.iloc[-1] ** (12.0 / nm) - 1.0) * 100),
                "mdd": float((eq / peak - 1.0).min() * 100), "final": float(eq.iloc[-1])})
    return out


def sim_block(ft: pd.DataFrame, rule: str, bar_days, n_days: int, eq_span: tuple, occ_span: tuple) -> dict:
    """運用の通りの候補（発火の数は問わず上位2件）を、出口ごとに3単位（分割は6単位も）で回す。"""
    picks = E82.daily_picks(ft, rule, min_break=LT.SKIP_BREAKS, top_k=LT.TOP_K, key=RULE_KEY[rule])
    res = {}
    kw = {"n_days": n_days, "eq_span": eq_span, "occ_span": occ_span}
    for col, _, _ in BASE_EXITS + SPLIT_EXITS:
        res[col] = unit_sim(picks, bar_days, legs_of(col), UNITS, **kw)
    for col, _, _ in SPLIT_EXITS:
        res[col + "_6u"] = unit_sim(picks, bar_days, legs_of(col), UNITS * 2, **kw)
    return res


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def line(name: str, s: dict, width: int = 32, split: bool = False) -> str:
    hit = "" if not np.isfinite(s.get("hit", NAN)) else f"  {'前半の +10% 到達' if split else '指値到達'} {s['hit']:4.1f}%"
    fire = "" if not np.isfinite(s.get("fire", NAN)) else (
        f"  発動 {s['fire']:4.1f}%（寄りで割った {s['gap']:3.1f}%・平均 {s['fire_day']:4.1f}日目・"
        f"発動の単位 {s['fire_ret']:+5.2f}%・残りの単位 {s['keep_ret']:+5.2f}%）")
    return (f"  {LT._l(name, width)}{s['n']:>6,}件  1取引 {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}・SE {s['se']:.2f}）"
            f"  勝率 {s['win']:4.1f}%  −10%割れ {s['bad10']:4.1f}%  最悪 {s['worst']:+6.1f}%{hit}{fire}")


def sim_line(name: str, st: dict, width: int = 34) -> str:
    if not st.get("n"):
        return f"  {LT._l(name, width)}取引なし（候補 {st.get('picks', 0)}件）"
    return (f"  {LT._l(name, width)}候補 {st['picks']:>4} → 買えた {st['n']:>4}（見送り {st['skipped']:>3}）"
            f"  1取引 {st['mean']:+6.2f}%（中央 {st['median']:+5.2f}）  勝率 {st['win']:4.1f}%  −10%割れ {st['bad10']:4.1f}%"
            f"  月の収益 {st['monthly']:+5.2f}%  年率 {st['cagr']:+6.1f}%  最大DD {st['mdd']:+6.1f}%"
            f"  月利 平均 {st['m_mean']:+5.2f}・中央 {st['m_med']:+5.2f}・最悪 {st['m_worst']:+6.1f}%"
            f"  稼働 {st['util']:3.0f}%  銘柄 {st['held']:.2f}（2銘柄以上 {st['held2']:3.0f}%・0 {st['idle']:3.0f}%）"
            f"  1銘柄の比率 {st['max_weight']:.0f}%")


def run_report(tag: str, ft: pd.DataFrame, bar_days, n_perm: int, seed: int, full: bool,
               ft20: pd.DataFrame | None = None) -> dict:
    """1つの窓の切り方の報告。full なら参考まで全部、そうでなければ判定の差・群の成績・3枠（A15）だけ。"""
    s: dict = {"tag": tag}
    months = E82.n_months(ft)
    wins = sorted(int(w) for w in ft["window"].unique())
    lo_d, hi_d = pd.to_datetime(ft["Date"]).min(), pd.to_datetime(ft["Date"]).max()
    s.update({"windows": wins, "months": months, "rows": int(len(ft)), "span": [str(lo_d.date()), str(hi_d.date())]})
    print(f"\n━━━━ 窓の切り方: {tag} ━━━━")
    print(f"■ {tag}-0. 前向きのテスト窓 {len(wins)}つ（{wins[0]}〜{wins[-1]}）・{len(ft):,}行・{months}か月"
          f"（{lo_d.date()} 〜 {hi_d.date()}。40営業日の結果が出ている選定日まで）")
    assert not (ft["A15"] & ~ft["R0"]).any(), "A15 が R0 の外にある"
    if ft20 is not None and len(ft20):
        a20 = ft20[ft20["A15"].to_numpy(dtype=bool)]
        e83 = exit_stats(a20, "tp10c", E82.n_months(ft20))
        s["e83_check"] = e83
        print(f"  確かめ（実験83 と同じ行 = 20日目まで）: A15 × 全部+10% {e83['n']}件・1取引 {e83['mean']:+.2f}%"
              f"（中央 {e83['med']:+.2f}）・勝率 {e83['win']:.1f}%・到達 {e83['hit']:.1f}%"
              f"  （実験83: 333件・+3.59%・中央 +5.96・70.9%・43.2%）")

    groups = {"A15": ft["A15"].to_numpy(dtype=bool), "R0": ft["R0"].to_numpy(dtype=bool)}
    print(f"\n■ {tag}-1. 出口ごとの成績（同じ行。手数料・税・すべりは含まない）")
    s["groups"] = {}
    for key in (RULES if full else ("A15",)):
        g = ft[groups[key]]
        s["groups"][key] = {}
        print(f"  {key}（{int(groups[key].sum()):,}行・月 {groups[key].sum() / max(months, 1):.1f}件）")
        for col, ja, _ in EXITS:
            st = exit_stats(g, col, months)
            s["groups"][key][col] = st
            print(line(ja, st, split=col.startswith("split")))

    a15 = ft[groups["A15"]]
    print(f"\n■ {tag}-2. 判定の差: A15 で 分割 − 全部+10%（同じ行・窓ごと。並べ替え {n_perm}回。W は選ばない）")
    s["main"] = {}
    for k in WKEYS:
        res = paired_test(a15, f"split{k}", MAIN_EXIT, n_perm=n_perm, seed=seed)
        rc = paired_test(a15, f"split{k}", MAIN_EXIT, n_perm=n_perm, seed=seed, unit="code")
        jd = judge(res)
        s["main"][k] = {"row": res, "code": {x: rc[x] for x in ("summary", "p", "rows")}, "judge": jd,
                        "same_way": same_way(res)}
        print(f"  高値−{k:>2}%: 窓ごとの差 {fmt_s(res['summary'])}  p = {res['p']:.3f}  銘柄単位 {fmt_s(rc['summary'])} "
              f"p = {rc['p']:.3f}  → {verdict_text(jd)}")
        if full:
            print("      窓ごと: " + "  ".join(f"窓{w} {d:+.2f}" for w, d in zip(res["windows"], res["diffs"])))

    if full:
        print(f"\n■ {tag}-3. 参考: ほかの出口との差（同じ行・窓ごと。判定に使わない）")
        s["ref"] = {}
        pairs = [(f"split{k}", b) for k in WKEYS for b in ("c20", "c40", "tp20a")]
        pairs += [(f"trail{k}", "c40") for k in WKEYS] + [("tp10c", "c20"), ("c40", "c20")]
        for key in RULES:
            g = ft[groups[key]]
            for a, b in pairs:
                res = paired_test(g, a, b, n_perm=0)
                s["ref"][f"{key}_{a}-{b}"] = res["summary"]
                print(f"  {key:<4} {LT._l(EXIT_JA[a], 24)} − {LT._l(EXIT_JA[b], 26)} {fmt_s(res['summary'])}")

    n_days = LT.n_bar_days(bar_days, lo_d, hi_d)
    eq_span = (lo_d, pd.Timestamp(bar_days[-1]))
    occ_span = (lo_d, hi_d)
    print(f"\n■ {tag}-4. 3枠の模擬（{LT.breaks_txt(LT.SKIP_BREAKS)}上位{LT.TOP_K}件・先着順。資金 {UNITS}単位。"
          f"分割は1銘柄 2単位 = 資金の 2/{UNITS}。{n_days}営業日）")
    print("  月の収益 = 単位ごとの収益の合計 ÷ 単位数 ÷ 月数（複利にしない）。年率・最大DD = 約定ごとに資金（簿価）を単位で"
          "割り当てた複利（全出口で同じ月数）。稼働 = 使っている単位の割合。銘柄 = 1日に持っている銘柄の数の平均"
          "（選定日の期間で数える）")
    s["sim"] = {}
    for key in (RULES if full else ("A15",)):
        res = sim_block(ft, key, bar_days, n_days, eq_span, occ_span)
        s["sim"][key] = res
        print(f"  {key}（並べ方: {'1.5σ の3本' if RULE_KEY[key] == 'pmin15' else '1.2σ の3本'}の最小の百分位）")
        for col, ja, _ in BASE_EXITS + SPLIT_EXITS:
            print(sim_line(f"{ja}" + ("（資金3単位）" if col.startswith("split") else ""), res[col]))
        for col, ja, _ in SPLIT_EXITS:
            print(sim_line(f"{ja}（資金6単位・1銘柄1/3）", res[col + "_6u"]))
        base = res[MAIN_EXIT].get("by_window", {})
        for col, _, _ in SPLIT_EXITS:
            for suf in ("", "_6u"):
                bw = res[col + suf].get("by_window", {})
                j = sorted(set(base) & set(bw))
                d = summarize(np.array([bw[w] - base[w] for w in j], dtype=float))
                res[f"{col}{suf}-{MAIN_EXIT}"] = d
                print(f"      {col}{suf} − {MAIN_EXIT}（取引の窓ごとの平均の差）: {fmt_s(d)}")

    if full:
        print(f"\n■ {tag}-5. 窓ごとの A15")
        s["a15_windows"] = {}
        for col in (MAIN_EXIT,) + tuple(f"split{k}" for k in WKEYS):
            wm = E83.window_means(a15, col)
            s["a15_windows"][col] = wm
            print(f"  {LT._l(EXIT_JA[col], 24)}" + "  ".join(f"窓{w} {m:+.2f}%（{n}件）" for w, n, m in wm)
                  + f"  プラスの窓 {sum(1 for _, _, m in wm if m > 0)}/{len(wm)}")
    return s


def verdict_of(out: dict, have_main: bool, have_shifted: bool) -> str:
    """
    幅ごとの結果（main_pass: ずらし0か月で通る / shifted_same_way: ずらした窓でも同じ向き。回していなければ None）から
    判定の文言を作る（先に決めた形。docstring の「判定（主）」）。
    """
    if not have_main:
        return "判定なし（ずらし0か月 = 本番の窓を回していない）"
    ok = [k for k in WKEYS if out[k]["main_pass"] and out[k]["shifted_same_way"] is not False]
    if not ok:
        return "通らない（今の出口 = 全部+10% より上とは言えない）"
    if len(ok) == len(WKEYS):
        if have_shifted:
            return "幅によらず今の出口（全部+10%）より上（3水準ともずらし0か月で通り、ずらした窓でも同じ向き）"
        return "ずらし0か月では3水準とも通る（ずらした窓は回していないので、幅によらずとはまだ言えない）"
    tail = "" if have_shifted else "。ずらした窓は回していない"
    return "幅しだい（" + "・".join(f"−{k}%" for k in ok) + f" だけ通る。W はここから選ばない{tail}）"


def stability(runs: dict) -> list:
    """
    幅の安定の並び（見るだけ。選ばない）。窓の切り方 0・2・4か月（学習し直した 0r は除く）で、(1) 差の平均が正で上の窓が過半
    だった切り方の数の多い順、(2) いちばん悪い切り方の差の平均の高い順。[(幅, 数, 切り方の数, いちばん悪い差)]
    """
    tags = [t for t in runs if t != "0r"]
    rows = []
    for k in WKEYS:
        res = [runs[t]["main"][k]["row"] for t in tags]
        good = sum(1 for r in res if same_way(r))
        means = [r["summary"]["mean"] for r in res if r["summary"]["n"]]
        rows.append((k, good, len(res), min(means) if means else NAN))
    return sorted(rows, key=lambda x: (-x[1], -(x[3] if np.isfinite(x[3]) else -np.inf)))


def final_summary(runs: dict) -> dict:
    """窓の切り方をまたいだ判定と、まとめの表示。"""
    print("\n■ まとめ（A15。分割 − 全部+10%、同じ行・窓ごと）")
    tags = list(runs)
    head = "  幅        " + "".join(f"{LT._l('ずらし' + t, 34)}" for t in tags)
    print(head)
    out = {}
    shifted = [t for t in tags if t not in ("0", "0r")]
    for k in WKEYS:
        cells = []
        for t in tags:
            m = runs[t]["main"][k]
            sm = m["row"]["summary"]
            cells.append(LT._l(f"{sm['mean']:+.2f} ± {sm['se']:.2f}（{sm['wins']}/{sm['n']}）p={m['row']['p']:.3f}", 34))
        print(f"  高値−{k:>2}%  " + "".join(cells))
        main_pass = bool(runs["0"]["main"][k]["judge"]["pass"]) if "0" in runs else False
        robust = all(runs[t]["main"][k]["same_way"] for t in shifted) if shifted else None
        out[k] = {"main_pass": main_pass, "shifted_same_way": robust}
    verdict = verdict_of(out, "0" in runs, bool(shifted))
    print(f"  判定: {verdict}")
    for k in WKEYS:
        o = out[k]
        print(f"    −{k}%: ずらし0か月 {'通る' if o['main_pass'] else '通らない'} / ずらした窓で同じ向き "
              f"{'—（回していない）' if o['shifted_same_way'] is None else ('○' if o['shifted_same_way'] else '×')}")
    st = stability(runs)
    print("  幅の安定（見るだけ。選ばない。差が正・上の窓が過半だった切り方の数 → いちばん悪い切り方の差の順）: "
          + " ＞ ".join(f"−{k}%（{g}/{n}・最悪 {w:+.2f}pt）" for k, g, n, w in st))
    if "0" in runs and "groups" in runs["0"] and "sim" in runs["0"]:
        g = runs["0"]["groups"]["A15"]
        print(f"  ずらし0か月の A15: 全部+10% {g['tp10c']['mean']:+.2f}%・"
              + "・".join(f"分割−{k}% {g['split' + k]['mean']:+.2f}%" for k in WKEYS)
              + f"・指値なし20日 {g['c20']['mean']:+.2f}%・40日 {g['c40']['mean']:+.2f}%")
        sim = runs["0"]["sim"]["A15"]
        print("  3枠の月の収益（A15）: " + "・".join(
            f"{c} {sim[c].get('monthly', NAN):+.2f}%" for c in ("tp10c", "c20", "c40") + tuple(f"split{k}" for k in WKEYS))
              + "（分割は資金3単位）")
        print("                       " + "・".join(
            f"split{k}（6単位）{sim['split' + k + '_6u'].get('monthly', NAN):+.2f}%" for k in WKEYS))
    print("  注意: 確かめられるのは同じ期間の out-of-fold だけ。手数料・税・すべりは含まない。W は選ばない（見るだけ）")
    return {"by_width": out, "verdict": verdict, "stability": st}


# --------------------------------------------------------------------------- #
# 窓をずらした out-of-fold（6本を本番と同じ作りで学習し直す）
# --------------------------------------------------------------------------- #

def prod_params(algo: str) -> tuple[dict, int]:
    """
    本番と同じパラメータと種。基準モデル（lgbm）は train_production と同じ（lgbm_params.json の DEFAULT_PRESET の鍵。
    random_state は tuning.FIXED の値）、ほかは train_multi と同じ（multi_params.json・tuning_multi.SEED）。
    """
    if algo == M.BASELINE:
        p = tuning.params_for(F.DEFAULT_PRESET)
        return p, int(p.get("random_state", 0))
    return TM.params_for(algo), int(TM.SEED)


def check_params(cols: list) -> list:
    """
    学習し直す6本に、学習する列と同じ列で探索したパラメータがあるか（本番の train_production / train_multi と同じ確かめ）。
    無ければ止める（既定値へ黙って落とさない）。列が違うものは知らせる（戻り値の文言）。
    """
    notes = []
    rec = tuning.load_params().get(F.DEFAULT_PRESET)
    if not rec:
        raise SystemExit(f"lgbm のパラメータ {F.DEFAULT_PRESET} がありません（{tuning.PARAMS_PATH}）")
    why = tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), cols)
    if why:
        notes.append(f"lgbm: {why}")
    if rec.get("n_estimators") != tuning.SEARCH_N_ESTIMATORS:
        notes.append(f"lgbm: 木 {rec.get('n_estimators')}本で探索（今は {tuning.SEARCH_N_ESTIMATORS}本）")
    store = TM.load()
    for a in SIX:
        if a == M.BASELINE:
            continue
        if a not in store:
            raise SystemExit(f"{a} の探索済みパラメータがありません（multi_params.json）")
        cv = store[a].get("_cv", {})
        why = tuning.tuned_mismatch(cv.get("features_sig"), cv.get("n_features"), cols)
        if why:
            notes.append(f"{a}: {why}")
        if TM.is_tree(a) and cv.get("n_estimators") != TM.N_ESTIMATORS:
            notes.append(f"{a}: 木 {cv.get('n_estimators')}本で探索（今は {TM.N_ESTIMATORS}本）")
    return notes


def data_signature(fr: pd.DataFrame, cols: list) -> str:
    """データセットの中身の指紋（学習に使う列・ラベルの引き直しに使う列）。保存済みの out-of-fold を使い回してよいかの鍵。"""
    keep = ["Code", "Date", "label"] + [c for c in ("vol_20d", "future_rise", "end_level", "uptrend_end", OUTCOME)
                                       if c in fr.columns] + list(cols)
    keep = list(dict.fromkeys(c for c in keep if c in fr.columns))
    h = pd.util.hash_pandas_object(fr[keep], index=False).to_numpy()
    return hashlib.sha1(h.tobytes()).hexdigest()[:10]


def frame_for(fr: pd.DataFrame, algo: str) -> pd.DataFrame:
    """ラベル違いのモデルは k のラベルに引き直す（train_multi.train_one と同じ。判定できない行は NaN で、学習・採点から外れる）。"""
    k = M.label_k(algo)
    if k is None:
        return fr
    d = fr.copy()
    d["label"] = B.rise_label_for_k(d, k)
    return d


def fingerprint(algo: str, params: dict, seed: int, cols: list, data_sig: str, shift: int) -> str:
    key = json.dumps({"algo": algo, "params": params, "seed": seed, "cols": F.signature(cols), "data": data_sig,
                      "k": M.label_k(algo), "shift": shift}, sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def shifted_wide(fr: pd.DataFrame, cols: list, shift: int, oof_fn=None, folds_fn=None, out_dir: str = OUT_DIR,
                 models=SIX, cache: bool = True) -> tuple[pd.DataFrame, list, dict]:
    """
    窓の境界を shift か月ずらして、models を本番と同じ作りで学習し直した out-of-fold を (Code, Date) で横に並べる
    （スコアは score_<名前>、実収益は基準モデルの行の OUTCOME）。cache なら保存済みを読み、無ければ作って保存する。
    oof_fn / folds_fn は既定で実験41 の oof_folds / folds_for（テストで差し替える）。
    """
    if oof_fn is None or folds_fn is None:
        import e41_stop_loss as E41          # 重い import（学習器）なので、学習し直すときだけ
        oof_fn = oof_fn or E41.oof_folds
        folds_fn = folds_fn or E41.folds_for
    folds = folds_fn(fr["Date"], shift)
    info = {"secs": {}, "rows": {}, "cached": []}
    dsig = data_signature(fr, cols) if cache else ""
    wide = None
    for a in models:
        params, seed = prod_params(a)
        path = os.path.join(out_dir, f"e89_{a}_sh{shift}_{fingerprint(a, params, seed, cols, dsig, shift)}.parquet")
        t0 = time.time()
        if cache and os.path.exists(path):
            o = pd.read_parquet(path)
            info["cached"].append(a)
        else:
            o = oof_fn(a, frame_for(fr, a), cols, params, seed, folds)
            if cache:
                os.makedirs(out_dir, exist_ok=True)
                o.to_parquet(path, index=False)
        info["secs"][a] = round(time.time() - t0)
        info["rows"][a] = int(len(o))
        o = o.copy()
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        o = o.drop_duplicates(["Code", "Date"], keep="first")
        keep = ["Code", "Date", "score"] + ([OUTCOME] if a == M.BASELINE and OUTCOME in o.columns else [])
        part = o[keep].rename(columns={"score": f"score_{a}"})
        wide = part if wide is None else wide.merge(part, on=["Code", "Date"], how="inner")
        log(f"    {a} ずらし{shift}か月: {len(o):,}件 / {info['secs'][a]}秒" + ("（保存済み）" if a in info["cached"] else ""))
    wide = wide.dropna(subset=[f"score_{a}" for a in models])
    info["kept"] = int(len(wide))
    return wide.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True), folds, info


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #

#: 比べる行に、すべて値があること（どの出口も同じ行で比べる）
COMPARE_COLS = tuple(c for c, _, _ in EXITS)


def complete_rows(ft20: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, dict]:
    """
    比べる行: 選定日が cutoff（日足の最後の日の 40営業日前）以前で、どの出口の収益も欠けていない行。
    40日目の足が無い（上場廃止など）・20日目の終値が欠測の行は、どの出口からも外す。
    """
    d = pd.to_datetime(ft20["Date"])
    late = (d > pd.Timestamp(cutoff)).to_numpy()
    vals = np.column_stack([pd.to_numeric(ft20[c], errors="coerce").to_numpy(dtype=float) for c in COMPARE_COLS])
    full = np.isfinite(vals).all(axis=1)
    keep = ~late & full
    info = {"before": int(len(ft20)), "late": int(late.sum()), "missing": int((~late & ~full).sum()),
            "missing_a15": int((~late & ~full & ft20["A15"].to_numpy(dtype=bool)).sum()), "kept": int(keep.sum())}
    return ft20[keep].reset_index(drop=True), info


def prepare(wide: pd.DataFrame, folds: list, bars: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """1つの窓の切り方の行の表（比べる行 ft と、実験83 と同じ 20日目までの行 ft20）。"""
    df = wide.copy()
    df["window"] = E79.assign_windows(df["Date"], folds)
    df = df[df["window"] > 0].reset_index(drop=True)
    df, chk = attach_all(df, bars)
    split = E79.window_split(df, folds, models=SIX)
    ft20 = build_rows(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    ft, rows = complete_rows(ft20, cutoff)
    chk["test_windows"] = len(split)
    chk["compare_rows"] = rows
    return ft, ft20, chk


def make_plan(shifts, no_retrain0: bool = False) -> list:
    """
    回す窓の切り方 [(名前, ずらす月数, 学習し直すか)]。0 は Release の out-of-fold（学習し直さない）。学習し直した 0 か月（0r）は、
    ずらした窓（2・4か月）も回すときだけ足す（作り方の違いとずらしの違いを分けるための参考なので）。
    """
    shifts = list(dict.fromkeys(int(s) for s in shifts))
    plan = []
    if 0 in shifts:
        plan.append(("0", 0, False))
        if not no_retrain0 and any(s != 0 for s in shifts):
            plan.append(("0r", 0, True))
    plan += [(str(s), s, True) for s in shifts if s != 0]
    return plan


def log_check(tag: str, chk: dict) -> None:
    log(f"[{tag}] 出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}・40日目の終値 {chk['with_c40']:,}。5日平均と "
        f"ret_o1_20 の一致 {chk['hold5_match']:,}/{chk['hold5_both']:,}。指値の順序の食い違い {chk['order_violations']}件・"
        f"トレーリングの幅の順序の食い違い {chk['trail_order_violations']}件・線より下で売った単位（寄りで割った日を除く）"
        f"{chk['trail_below_line']}件")
    if chk["order_violations"] or chk["trail_order_violations"] or chk["trail_below_line"] or chk["hold5_both"] == 0 \
            or chk["hold5_match"] / chk["hold5_both"] < 0.99:
        log(f"[warn] [{tag}] 出口の確かめに食い違いがある。日足とデータセットの食い違いを疑う")


def conv(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験89: 分割決済（前半 +10% の指値・後半 高値からのトレーリング）")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--shifts", default=",".join(str(s) for s in SHIFTS),
                    help="窓の境界をずらす月数（カンマ区切り）。0 は本番の窓（Release の out-of-fold）")
    ap.add_argument("--no-retrain0", action="store_true", help="学習し直した 0 か月（参考）を省く")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip() != ""]
    t0 = time.time()
    fr = lab.frame()
    bars = LT.load_bars(lab.DATA_DIR, start=pd.to_datetime(fr["Date"]).min() - pd.Timedelta(days=10), extra=("AdjL",))
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    cutoff = bar_days[-(CAP + 1)]
    log(f"日足 {bar_days[0].date()} 〜 {bar_days[-1].date()}。40営業日の結果が出ている選定日は {cutoff.date()} まで")

    plan = make_plan(shifts, args.no_retrain0)
    log("窓の切り方: " + "・".join(f"{t}（{'学習し直す' if re else 'Release'}）" for t, _, re in plan))
    cols = None
    if any(re for _, _, re in plan):
        cols = F.columns(F.DEFAULT_PRESET)
        miss = [c for c in cols if c not in fr.columns]
        if miss:
            raise SystemExit(f"{F.DEFAULT_PRESET} の列がデータセットに揃っていない（{len(miss)}列。例 {miss[:5]}）")
        log(f"学習し直す6本: {', '.join(SIX)}（{F.DEFAULT_PRESET} {len(cols)}列・指紋 {F.signature(cols)}。"
            f"種: lgbm {prod_params(M.BASELINE)[1]} / ほか {TM.SEED}）")
        for note in check_params(cols):
            log(f"[warn] 探索した列と学習する列が違う: {note}")

    runs, infos = {}, {}
    for tag, shift, retrain in plan:
        if retrain:
            log(f"[{tag}] 窓を {shift}か月ずらした out-of-fold を作る")
            wide, folds, info = shifted_wide(fr, cols, shift)
            info["shift"] = shift
        else:
            wide, info0 = E79.load_oofs()
            log(f"[{tag}] Release の out-of-fold 12本を突き合わせた: {info0['kept']:,}件（使うのは6本）")
            folds = E79.make_windows(fr["Date"])
            info = {"kept": info0["kept"], "shift": 0}
        ft, ft20, chk = prepare(wide, folds, bars, cutoff)
        info["exit_check"] = chk
        log_check(tag, chk)
        rw = chk["compare_rows"]
        log(f"[{tag}] 前向きの行 {rw['before']:,}件（テスト窓 {chk['test_windows']}つ）→ 比べる行 {rw['kept']:,}件"
            f"（A15 {int(ft['A15'].sum())}件・R0 {int(ft['R0'].sum())}件）。外した行: 40営業日の結果がまだ {rw['late']:,}件・"
            f"どれかの出口が欠測 {rw['missing']:,}件（うち A15 {rw['missing_a15']}件）")
        runs[tag] = run_report(tag, ft, bar_days, args.perm, SEED, full=(tag == "0"),
                               ft20=ft20 if tag == "0" else None)
        infos[tag] = info
    summary = {"runs": runs, "info": infos, "cutoff": str(cutoff.date())}
    summary["final"] = final_summary(runs)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e89_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=conv)
    log(f"記録: {path} / 全体 {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
