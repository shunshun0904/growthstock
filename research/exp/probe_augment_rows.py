#!/usr/bin/env python3
"""
probe（データ拡張の案0）: いまは学習から外している実在の行を、学習の側だけに足すとしたら、何行・どんな行になるかを数える。
モデルは学習しない（数えるだけ。検証だけ。本番の設定・データセットは変えない）。

運用者の依頼（2026-10-18）「金融時系列データのオーグメンテーション手法について」→ 調査（運用者の選択: 範囲「このモデル
への適用」・狙い「サンプル数・過学習」・出力「共有用のページ」）。調査のページ（非公開の Artifact「ブレイク予測のデータ拡張」
https://claude.ai/artifact/QmsTiMPjrMAUwQa98AEsQa ）の結論: ジッター・スケーリング・タイムワープ・GAN などの一般的な時系列の
データ拡張は見送り、いまは捨てている実在の行を学習の側だけに足す案（案1・案2）を試す。その前に、足せる行の数・正例率・
ラベル期間の重なりを数える（案0）。運用者の選択（AskUserQuestion）「案0の数え上げを作る」。

足す候補
  案1  連続した更新日: 78週高値の更新日のうち、新規のブレイクではない日（直前20営業日に更新がある日。いまは
       BREAKOUT_COOLDOWN で捨てている）。除外の条件（ラベル確定・78週高値の定義・流動性・ETF/REIT・普通株以外）は本番と同じ
  案2  終値の母集団だけにある行: 「終値 > それまでの終値の最大」の新規のブレイク（実験90 の close）のうち、いまの学習データに
       無い行

作り方（除外の条件と特徴量を本番とそろえるため、本番の build_dataset.build() をそのまま呼ぶ）
  案1  更新日の印（mark_new_highs）だけを呼ぶ間だけ差し替え、更新日すべてを母集団に残す（終わったら必ず戻す）。新規の
       ブレイクかどうかは base_length（1つ前の更新日からの営業日数）で分かる: 21以上か欠測なら新規（docs/MODEL_DESIGN.md
       「base_length の定義」）。新規の行は、ワークフローが既定で作った dataset.parquet と行がすべて一致するはず（確かめて出す）。
       差し替えの中で、本番の印（直前20営業日に更新が無い日）と銘柄の中の行の位置を控える（上昇局面とラベル期間の重なりに使う）
  案2  BREAKOUT_REF=close / BREAKOUT_ON_HIGH=False（実験90 と同じやり方）

言葉
  上昇局面  新規のブレイクから、同じ銘柄の次の新規のブレイクの前までの更新日（足す行はどれか1つの局面に入る）
  一意性    同じ銘柄の行どうしでラベルの判定期間（t+1〜t+20 営業日）が重なる数で割った、行ごとの平均（López de Prado
            『Advances in Financial Machine Learning』第4章 Sample Weights の average uniqueness）。合計が実効の行数の目安

出すもの（公開ログには件数・割合・分位・日付だけ。銘柄・値は出さない）
  0. 確かめ（新規の行と dataset.parquet の一致、base_length での判定と本番の印の一致）と、更新日の行数（除外の前後）
  1. 案1 の行数（年ごと）と正例率（1.2σ・1.5σ・1.8σ。年ごと・上昇局面の中の位置ごと）
  2. 上昇局面の大きさ
  3. ラベル期間の重なり（一意性・実効の行数）
  4. 足す行のラベルと、同じ局面の新規の行のラベルの一致
  5. 列の分布の違い（主な列の分位と、本番の249列それぞれの KS 統計量）
  6. 案2 の行数・正例率・いまの行との近さ・案1 との重なり・実効の行数

    exp=probe_augment_rows.py                 （表を作って数える。build を2回回す）
    exp=probe_augment_rows.py args="--no-close"   （案1 だけ。build は1回）

作った表は research/_data/oof/pa0_* に置き、次の回は actions/cache で引き継いだそれを使う（dataset.parquet の行数・期間・
正例数が変わっていたら作り直す。--rebuild で必ず作り直す）。

結果（2026-10-18。run 39115463637。docs/MODEL_ADOPTION_RULES.md §67）: 確かめはすべて一致（新規の行 21,957 が dataset.parquet と
銘柄×日付ですべて一致、base_length での判定と本番の印が 145,038行すべてで一致）。
  案1  足せる行 123,081（いまの 5.61倍）・正例率 1.5σ 13.0%（いま 13.0%）。ラベル期間の重なりで数えた実効の行数は
       21,957 → 41,004（1.87倍）で、増える分は足せる行のある局面の数（19,078）とほぼ同じ。足す行は base_length で新規の行と
       完全に分かれる（KS 1.000）が、本番の249列の KS の中央値は 0.027
  案2  close だけの行 10,153（0.46倍）・正例率 1.5σ 13.9%。8割がいまの行の前後20営業日以内にあり、実効の行数は 25,427（+15.8%）
本番は変えていない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402

DATA_DIR = B.DATA_DIR
TAG = "pa0"
#: ラベルのしきい値（本番の12本と同じ3つ）。1.2σ はデータセットのラベル、ほかは rise_label_for_k で引き直す
KS = (1.2, 1.5, 1.8)
#: 正例率を年ごと・位置ごとに並べるラベル（今の規則のモデルのラベル）
MAIN_K = 1.5
#: ラベルの判定期間（営業日）。重なりの計算に使う
HORIZON = B.RISE_HORIZON
#: 新規のブレイクとみなす間隔（営業日）。base_length がこれより大きいか欠測なら新規
COOLDOWN = B.BREAKOUT_COOLDOWN
#: 上昇局面の中の位置（新規のブレイクから何営業日目か）の区切り。(下限, 上限]
OFFSET_BINS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, None))
#: 分位を並べる主な列（無い列は飛ばす）
KEY_COLS = ("base_length", "broken_high_age", "pullback_depth", "break_margin", "ret_1d", "ret_5d", "ret_20d",
            "vol_20d", "volume_trend", "r_high_3m")
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
CLUSTER_QS = (0.5, 0.75, 0.9, 0.99)
TOP_KS = 15


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oof_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "oof")


def fpath(data_dir: str, name: str) -> str:
    return os.path.join(oof_dir(data_dir), f"{TAG}_{name}")


def ycol(k: float) -> str:
    return f"y{int(round(k * 10)):02d}"


def norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Code"] = out["Code"].astype(str)
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def keys_of(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[["Code", "Date"]])


# --------------------------------------------------------------------------- #
# 表を作る（本番の build をそのまま呼ぶ。設定は呼ぶ間だけ書き換えて、必ず戻す）
# --------------------------------------------------------------------------- #
@contextmanager
def all_update_days(capture: dict):
    """
    build() の間だけ、78週高値の更新日すべてを母集団に残す（mark_new_highs の差し替え）。

    本番の印（直前 COOLDOWN 営業日に更新が無い日 = 新規のブレイク）と、銘柄の中の行の位置（price_panel の並び = 銘柄・日付順）
    を capture["updates"]（更新日だけ）に、日足にある日を capture["days"] に控える。差し替えた表には SWEEP_OVERRIDES の
    印が meta.json に残る（既定の学習に使ってはいけない表の目印）。
    """
    orig = B.mark_new_highs
    saved = dict(B.SWEEP_OVERRIDES)

    def mark(df: pd.DataFrame, cooldown: int | None = None, on_high: bool | None = None) -> pd.DataFrame:
        out = orig(df, cooldown, on_high)
        nh = out["is_new_high"].fillna(False).astype(bool)
        pos = out.groupby("Code", sort=False).cumcount()
        capture["updates"] = pd.DataFrame({
            "Code": out.loc[nh, "Code"].astype(str).to_numpy(),
            "Date": pd.to_datetime(out.loc[nh, "Date"]).to_numpy(),
            "pos": pos[nh].to_numpy(dtype=np.int64),
            "fresh": out.loc[nh, "is_fresh_break"].astype(bool).to_numpy()})
        capture["days"] = np.sort(pd.to_datetime(out["Date"]).unique())
        out["is_fresh_break"] = nh
        return out

    B.mark_new_highs = mark
    B.SWEEP_OVERRIDES["ALL_UPDATE_DAYS"] = True
    try:
        yield capture
    finally:
        B.mark_new_highs = orig
        B.SWEEP_OVERRIDES.clear()
        B.SWEEP_OVERRIDES.update(saved)


@contextmanager
def close_population():
    """build() の間だけ、母集団を「終値 > それまでの終値の最大」にする（実験90 の close と同じ書き換え）。"""
    saved = (B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH, dict(B.SWEEP_OVERRIDES))
    B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH = "close", False
    B.SWEEP_OVERRIDES["BREAKOUT_REF"] = "close"
    B.SWEEP_OVERRIDES["BREAKOUT_ON_HIGH"] = False
    try:
        yield
    finally:
        B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH = saved[0], saved[1]
        B.SWEEP_OVERRIDES.clear()
        B.SWEEP_OVERRIDES.update(saved[2])


def load_dataset(data_dir: str) -> pd.DataFrame:
    p = os.path.join(data_dir, "dataset.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"{p} がありません（ワークフローの Build dataset が作る）")
    return norm(pd.read_parquet(p))


def stamp_of(ds: pd.DataFrame) -> dict:
    return {"n": int(len(ds)), "from": str(ds["Date"].min().date()), "to": str(ds["Date"].max().date()),
            "positives": int(pd.to_numeric(ds["label"], errors="coerce").sum()), "features": len(F.all_columns()),
            "cooldown": COOLDOWN, "horizon": HORIZON}


def build_tables(data_dir: str = DATA_DIR, with_close: bool = True, rebuild: bool = False) -> dict:
    """いまの学習データと、更新日すべてを残した表（と控えた更新日）・終値の母集団の表。作ったものは次の回に使う。"""
    if (B.BREAKOUT_REF, B.BREAKOUT_ON_HIGH, B.SWEEP_OVERRIDES) != ("high", True, {}):
        raise SystemExit("build_dataset の既定が本番と違う（SWEEP_* の環境変数が入っている？）: "
                         f"{B.BREAKOUT_REF} / {B.BREAKOUT_ON_HIGH} / {B.SWEEP_OVERRIDES}")
    ds = load_dataset(data_dir)
    st = stamp_of(ds)
    os.makedirs(oof_dir(data_dir), exist_ok=True)
    p = {name: fpath(data_dir, f"{name}.parquet") for name in ("all", "updates", "days", "close")}
    ps = fpath(data_dir, "stamp.json")
    old = None
    if os.path.exists(ps):
        with open(ps, encoding="utf-8") as fh:
            old = json.load(fh)
    if rebuild or old != st:
        for f in list(p.values()) + [ps]:     # 古いデータで作った表を次の回に使わない
            if os.path.exists(f):
                os.remove(f)
    if not all(os.path.exists(p[n]) for n in ("all", "updates", "days")):
        t0 = time.time()
        cap: dict = {}
        with all_update_days(cap):
            B.build(data_dir, p["all"])
        if "updates" not in cap:
            raise SystemExit("mark_new_highs の差し替えが build の中で呼ばれなかった")
        cap["updates"].to_parquet(p["updates"], index=False)
        pd.DataFrame({"Date": cap["days"]}).to_parquet(p["days"], index=False)
        with open(ps, "w", encoding="utf-8") as fh:  # 案2 の表を作る途中で落ちても、次の回はこの表を使う
            json.dump(st, fh, ensure_ascii=False, indent=1)
        log(f"[build] 更新日すべてを残した表を作った（{time.time() - t0:.0f}秒）")
    else:
        log(f"[build] 前の回に作った表を使う（dataset.parquet {st['n']:,}行・{st['to']} まで）")
    if with_close and not os.path.exists(p["close"]):
        t0 = time.time()
        with close_population():
            B.build(data_dir, p["close"])
        log(f"[build] 終値の母集団の表を作った（{time.time() - t0:.0f}秒）")
    up = pd.read_parquet(p["updates"])
    up["Code"] = up["Code"].astype(str)
    up["Date"] = pd.to_datetime(up["Date"])
    return {"ds": ds, "all": norm(pd.read_parquet(p["all"])), "updates": up,
            "days": pd.to_datetime(pd.read_parquet(p["days"])["Date"]).to_numpy(),
            "close": norm(pd.read_parquet(p["close"])) if with_close else None}


# --------------------------------------------------------------------------- #
# 数える部品（重い計算を回さずにテストできる形）
# --------------------------------------------------------------------------- #
def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    """KS の3つのラベルの列（y12・y15・y18）を足す。1.2σ はデータセットのラベル、ほかは rise_label_for_k で引き直す。"""
    out = df.copy()
    out[ycol(B.VOL_NORM_K)] = pd.to_numeric(out["label"], errors="coerce").astype(float)
    for k in KS:
        if abs(k - B.VOL_NORM_K) > 1e-9:
            out[ycol(k)] = B.rise_label_for_k(out, k)
    return out


def fresh_by_base_length(base_length: pd.Series, cooldown: int = COOLDOWN) -> np.ndarray:
    """新規のブレイクか（1つ前の更新日から cooldown 営業日より後か、前の更新日がデータに無い）。"""
    bl = pd.to_numeric(base_length, errors="coerce")
    return (bl.isna() | (bl > cooldown)).to_numpy()


def assign_clusters(updates: pd.DataFrame) -> pd.DataFrame:
    """
    更新日（Code・Date・pos・fresh）に上昇局面を付ける。局面は新規のブレイクで始まり、同じ銘柄の次の新規のブレイクの前で
    終わる。cluster（通し番号。新規のブレイクより前の更新日は -1）・offset（局面の新規のブレイクから何営業日目か）・
    size（局面の更新日の数。新規を含む）。
    """
    u = updates.sort_values(["Code", "pos"]).reset_index(drop=True)
    cid = u.groupby("Code", sort=False)["fresh"].cumsum().astype(int)
    lab_ = u["Code"].astype(str) + "#" + cid.astype(str)
    u["cluster"] = np.where(cid > 0, pd.factorize(lab_)[0], -1)
    first = u.groupby("cluster")["pos"].transform("min")
    u["offset"] = np.where(cid > 0, u["pos"] - first, np.nan)
    u["size"] = np.where(cid > 0, u.groupby("cluster")["pos"].transform("size"), np.nan)
    return u


def avg_uniqueness(codes: Sequence, pos: Sequence, horizon: int = HORIZON) -> np.ndarray:
    """
    行ごとの平均の一意性。行 i の判定期間 [pos_i + 1, pos_i + horizon] の各日について、同じ銘柄でその日を判定期間に含む行の
    数 c_t を数え、1/c_t の平均を取る（重なりが無ければ 1）。
    """
    codes = np.asarray(codes).astype(str)
    pos = np.asarray(pos, dtype=np.int64)
    out = np.full(len(pos), np.nan)
    if not len(pos):
        return out
    order = np.lexsort((pos, codes))
    cs, ps = codes[order], pos[order]
    cut = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    for a, b in zip(cut[:-1], cut[1:]):
        p = ps[a:b]
        base = int(p.min()) + 1
        n_days = int(p.max()) + horizon - base + 2
        diff = np.zeros(n_days + 1)
        np.add.at(diff, p + 1 - base, 1.0)
        np.add.at(diff, p + horizon + 1 - base, -1.0)
        conc = np.cumsum(diff)[:n_days]
        inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
        pre = np.r_[0.0, np.cumsum(inv)]
        lo = p + 1 - base
        out[order[a:b]] = (pre[lo + horizon] - pre[lo]) / horizon
    return out


def ks_stat(a: Iterable, b: Iterable) -> float:
    """2つの標本の Kolmogorov–Smirnov 統計量（累積分布の差の最大。欠測は除く）。どちらかが空なら NaN。"""
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return float("nan")
    v = np.concatenate([a, b])
    fa = np.searchsorted(a, v, side="right") / len(a)
    fb = np.searchsorted(b, v, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def nearby(only: pd.DataFrame, other: pd.DataFrame, days: int = COOLDOWN) -> pd.DataFrame:
    """only の各行について、other に同じ銘柄の行が 直前 / 直後 days 営業日以内（mpos の差）にあるか（before・after の列）。"""
    o = only[["Code", "mpos"]].reset_index().sort_values("mpos")
    t = other[["Code", "mpos"]].sort_values("mpos").assign(_hit=True)
    kw = dict(on="mpos", by="Code", tolerance=days, allow_exact_matches=False)
    back = pd.merge_asof(o, t, direction="backward", **kw).set_index("index")["_hit"]
    fwd = pd.merge_asof(o, t, direction="forward", **kw).set_index("index")["_hit"]
    out = only.copy()
    out["before"] = back.reindex(only.index).notna().to_numpy()
    out["after"] = fwd.reindex(only.index).notna().to_numpy()
    return out


def market_pos(dates: pd.Series, days: np.ndarray) -> np.ndarray:
    """日足にある日の通し番号（銘柄の休みは数えるので、銘柄の中の行の位置の近似）。"""
    return np.searchsorted(np.asarray(days, dtype="datetime64[ns]"), pd.to_datetime(dates).to_numpy(), side="left")


def rate(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce")
    return float(y.mean()) if y.notna().any() else float("nan")


def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def qline(s: pd.Series, qs: Sequence[float] = QS, nd: int = 1) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if not len(v):
        return "—"
    return " / ".join(f"{x:,.{nd}f}" for x in v.quantile(list(qs)))


def times(a: float, b: float) -> str:
    return f"{a / b:.2f}倍" if b else "—"


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #
def report(t: dict) -> dict:
    """表（build_tables の戻り値）から数えて出す。要約（公開してよい数だけ）を返す。"""
    summ: dict = {}
    ds = with_labels(t["ds"])
    al = with_labels(t["all"])
    up = assign_clusters(t["updates"])
    days = t["days"]
    yk = ycol(MAIN_K)

    # ■ 0. 確かめ
    print("\n■ 0. 表と確かめ")
    print(f"  いまの学習データ（dataset.parquet）: {len(ds):,}行 / {ds['Date'].min().date()}〜{ds['Date'].max().date()} / "
          "正例率 " + "・".join(f"{k}σ {pct(rate(ds[ycol(k)]))}" for k in KS))
    al["fresh_bl"] = fresh_by_base_length(al["base_length"])
    m = al.merge(up[["Code", "Date", "pos", "fresh", "cluster", "offset", "size"]], on=["Code", "Date"], how="left")
    matched = m["pos"].notna()
    fr = m[m["fresh_bl"]]
    ad = m[~m["fresh_bl"]].copy()
    kd, kf = keys_of(ds), keys_of(fr)
    n_both = int(kf.isin(kd).sum())
    n_only_all, n_only_ds = int(len(kf) - n_both), int((~kd.isin(kf)).sum())
    agree = int((m.loc[matched, "fresh"].astype(bool) == m.loc[matched, "fresh_bl"]).sum())
    print(f"  更新日をすべて残した表（build_dataset.build・mark_new_highs だけ差し替え）: {len(al):,}行")
    print(f"    新規のブレイク（base_length が{COOLDOWN + 1}以上か欠測）: {len(fr):,}行 → dataset.parquet と一致 {n_both:,} / "
          f"表だけ {n_only_all:,} / dataset だけ {n_only_ds:,}（どちらも 0 のはず）")
    print(f"    base_length での判定と本番の印（直前{COOLDOWN}営業日に更新が無い日）: 一致 {agree:,} / "
          f"食い違い {int(matched.sum()) - agree:,} / 控えた更新日に無い行 {int((~matched).sum()):,}")
    n_up, n_up_fresh = len(up), int(up["fresh"].sum())
    print(f"  更新日（日足のパネル・除外の前）: {n_up:,}行（新規 {n_up_fresh:,} / 新規ではない {n_up - n_up_fresh:,}）→ "
          f"除外の後: 新規 {len(fr):,}（{pct(len(fr) / n_up_fresh if n_up_fresh else np.nan)}）/ 新規ではない {len(ad):,}"
          f"（{pct(len(ad) / (n_up - n_up_fresh) if n_up > n_up_fresh else np.nan)}）")
    summ["check"] = {"dataset_rows": int(len(ds)), "all_rows": int(len(al)), "fresh_rows": int(len(fr)),
                     "fresh_in_dataset": n_both, "fresh_only_table": n_only_all, "dataset_only": n_only_ds,
                     "flag_agree": agree, "flag_disagree": int(matched.sum()) - agree, "unmatched": int((~matched).sum()),
                     "updates_panel": int(n_up), "updates_panel_fresh": n_up_fresh}

    # ■ 1. 案1 の行数と正例率
    print("\n■ 1. 案1（連続した更新日）の行数と正例率")
    print(f"  全体: いまの行 {len(fr):,} / 足せる行 {len(ad):,}（いまの行の {times(len(ad), len(fr))}）")
    print("  ラベル別の正例率（いま / 足す）: " + " / ".join(
        f"{k}σ {pct(rate(fr[ycol(k)]))}・{pct(rate(ad[ycol(k)]))}" for k in KS))
    print(f"  年ごと（行数 いま / 足す・倍率 ・ 正例率 {MAIN_K}σ いま / 足す）:")
    by_year = {}
    for y in sorted(set(m["Date"].dt.year)):
        a, b = fr[fr["Date"].dt.year == y], ad[ad["Date"].dt.year == y]
        by_year[str(y)] = {"fresh": int(len(a)), "added": int(len(b)), "pos_fresh": rate(a[yk]), "pos_added": rate(b[yk])}
        print(f"    {y}: {len(a):,} / {len(b):,}（{times(len(b), len(a))}）・{pct(rate(a[yk]))} / {pct(rate(b[yk]))}")
    print(f"  上昇局面の中の位置（足す行。新規のブレイクから何営業日目か）: 行数・正例率 {MAIN_K}σ")
    by_off = {}
    for lo, hi in OFFSET_BINS:
        sel = (ad["offset"] > lo) & ((ad["offset"] <= hi) if hi is not None else True)
        name = f"{lo + 1}〜{hi}" if hi is not None else f"{lo + 1}〜"
        by_off[name] = {"rows": int(sel.sum()), "pos": rate(ad.loc[sel, yk])}
        print(f"    {name:>7}: {int(sel.sum()):,}行・{pct(rate(ad.loc[sel, yk]))}")
    summ["plan1"] = {"fresh": int(len(fr)), "added": int(len(ad)),
                     "pos": {str(k): {"fresh": rate(fr[ycol(k)]), "added": rate(ad[ycol(k)])} for k in KS},
                     "by_year": by_year, "by_offset": by_off}

    # ■ 2. 上昇局面
    print("\n■ 2. 上昇局面（新規のブレイクから、同じ銘柄の次の新規のブレイクの前までの更新日）")
    n_cl = int(up.loc[up["cluster"] >= 0, "cluster"].nunique())
    fresh_cl = set(fr.loc[fr["cluster"] >= 0, "cluster"].astype(int))
    print(f"  局面の数（パネル）: {n_cl:,}（うち新規の行が学習データにある {len(fresh_cl):,}）")
    sizes = up.loc[up["cluster"] >= 0].groupby("cluster")["pos"].size()
    print(f"  1局面の更新日の数（パネル・新規を含む）: 分位 {'/'.join(f'{int(q * 100)}%' for q in CLUSTER_QS)} = "
          f"{qline(sizes, CLUSTER_QS, 0)} / 最大 {int(sizes.max()) if len(sizes) else 0:,}")
    per_cl = ad.loc[ad["cluster"] >= 0].groupby("cluster").size()
    orphan = int((~ad["cluster"].isin(fresh_cl)).sum())
    print(f"  足せる行のある局面: {len(per_cl):,}（いまの行の {pct(len(per_cl) / len(fr) if len(fr) else np.nan)}）。"
          f"1局面の足せる行の数: 分位 {qline(per_cl, CLUSTER_QS, 0)} / 最大 {int(per_cl.max()) if len(per_cl) else 0:,}")
    print(f"  足せる行のうち、局面の新規の行が学習データに無い行（流動性などで新規の行だけが落ちた）: {orphan:,}"
          f"（{pct(orphan / len(ad) if len(ad) else np.nan)}）")
    summ["clusters"] = {"panel": n_cl, "with_fresh_in_dataset": len(fresh_cl), "with_added": int(len(per_cl)),
                        "added_per_cluster_q": {str(q): float(per_cl.quantile(q)) for q in CLUSTER_QS} if len(per_cl) else {},
                        "orphan_added": orphan}

    # ■ 3. ラベル期間の重なり
    print(f"\n■ 3. ラベル期間の重なり（同じ銘柄で判定期間 t+1〜t+{HORIZON} が重なる行の数で割った平均の一意性。"
          "López de Prado 第4章）")
    um = m[matched]
    u_f = avg_uniqueness(um.loc[um["fresh_bl"], "Code"], um.loc[um["fresh_bl"], "pos"])
    u_all = avg_uniqueness(um["Code"], um["pos"])
    is_f = um["fresh_bl"].to_numpy()
    print(f"  いまの行だけ: 一意性の平均 {np.nanmean(u_f):.3f}（最小 {np.nanmin(u_f) if len(u_f) else np.nan:.3f}）・"
          f"実効の行数 {np.nansum(u_f):,.0f}")
    print(f"  いまの行 + 足せる行: {len(um):,}行 → 実効の行数 {np.nansum(u_all):,.0f}（いまの行の "
          f"{times(np.nansum(u_all), np.nansum(u_f))}。行数では {times(len(um), int(is_f.sum()))}）")
    print(f"    一意性の平均: いまの行 {np.nanmean(u_all[is_f]):.3f} / 足す行 "
          f"{np.nanmean(u_all[~is_f]) if (~is_f).any() else np.nan:.3f}")
    print(f"  局面ごとに足す行の重みの合計を1にするなら、足す行の重みの合計 = 足せる行のある局面の数 {len(per_cl):,}"
          f"（いまの行の {pct(len(per_cl) / len(fr) if len(fr) else np.nan)}）")
    summ["uniqueness"] = {"fresh_mean": float(np.nanmean(u_f)) if len(u_f) else None,
                          "effective_fresh": float(np.nansum(u_f)), "effective_union": float(np.nansum(u_all)),
                          "union_rows": int(len(um)), "mean_fresh_in_union": float(np.nanmean(u_all[is_f])),
                          "mean_added_in_union": float(np.nanmean(u_all[~is_f])) if (~is_f).any() else None}

    # ■ 4. ラベルの一致
    print("\n■ 4. 足す行と、同じ局面の新規の行のラベル（局面の新規の行が学習データにある足す行だけ）")
    agree_out = {}
    for k in (MAIN_K, B.VOL_NORM_K):
        c = ycol(k)
        head = fr.loc[fr["cluster"] >= 0, ["cluster", c]].rename(columns={c: "_head"})
        j = ad[["cluster", c]].merge(head, on="cluster", how="inner").dropna()
        p1, p0 = j.loc[j["_head"] == 1, c], j.loc[j["_head"] == 0, c]
        agree_out[str(k)] = {"head_pos": {"n": int(len(p1)), "rate": rate(p1)}, "head_neg": {"n": int(len(p0)), "rate": rate(p0)}}
        print(f"  {k}σ: 新規が正例 → 足す行の正例率 {pct(rate(p1))}（{len(p1):,}行）/ 新規が負例 → {pct(rate(p0))}（{len(p0):,}行）")
    summ["label_agree"] = agree_out

    # ■ 5. 列の分布
    print("\n■ 5. 列の分布（新規の行 と 足せる行）")
    print(f"  主な列の分位（{'/'.join(f'{int(q * 100)}%' for q in QS)}）: いま | 足す")
    for c in KEY_COLS:
        if c in m.columns:
            print(f"    {c:<16} {qline(fr[c])} | {qline(ad[c])}")
    cols = [c for c in F.columns(F.DEFAULT_PRESET) if c in m.columns]
    ks = pd.Series({c: ks_stat(fr[c], ad[c]) for c in cols}).dropna().sort_values(ascending=False)
    if len(ks):
        print(f"  本番の{len(cols)}列の KS 統計量（2つの累積分布の差の最大。0 なら同じ・1 なら重ならない）: 中央値 "
              f"{ks.median():.3f}・0.1 を超える列 {int((ks > 0.1).sum())}・0.3 を超える列 {int((ks > 0.3).sum())}")
        print(f"  上位{min(TOP_KS, len(ks))}列（KS / 欠測 いま・足す）:")
        for c, v in ks.head(TOP_KS).items():
            print(f"    {c:<24} {v:.3f} / {pct(fr[c].isna().mean())}・{pct(ad[c].isna().mean())}")
    summ["ks"] = {"median": float(ks.median()) if len(ks) else None, "over_0_1": int((ks > 0.1).sum()),
                  "over_0_3": int((ks > 0.3).sum()), "top": {c: float(v) for c, v in ks.head(TOP_KS).items()}}

    # ■ 6. 案2
    cl = t.get("close")
    if cl is None:
        print("\n■ 6. 案2（終値の母集団だけにある行）: 作っていない（--no-close）")
        return summ
    print("\n■ 6. 案2（終値の母集団だけにある行）")
    cl = with_labels(cl)
    co = cl[~keys_of(cl).isin(kd)].copy()
    print(f"  終値の母集団（BREAKOUT_REF=close・ON_HIGH=False。新規のブレイクだけ）: {len(cl):,}行。いまの行と共通 "
          f"{len(cl) - len(co):,} / close だけ {len(co):,}（いまの行の {times(len(co), len(ds))}）")
    print("  ラベル別の正例率（いま / close だけ）: " + " / ".join(
        f"{k}σ {pct(rate(ds[ycol(k)]))}・{pct(rate(co[ycol(k)]))}" for k in KS))
    print(f"  年ごと（close だけの行数・正例率 {MAIN_K}σ）: " + " ".join(
        f"{y}:{int((co['Date'].dt.year == y).sum()):,}・{pct(rate(co.loc[co['Date'].dt.year == y, yk]))}"
        for y in sorted(set(co["Date"].dt.year))))
    dsp = ds.assign(mpos=market_pos(ds["Date"], days))
    co["mpos"] = market_pos(co["Date"], days)
    nb = nearby(co, dsp)
    near = {}
    for name, sel in (("直前だけ", nb["before"] & ~nb["after"]), ("直後だけ", ~nb["before"] & nb["after"]),
                      ("両方", nb["before"] & nb["after"]), ("どちらも無い", ~nb["before"] & ~nb["after"])):
        near[name] = {"rows": int(sel.sum()), "pos": rate(nb.loc[sel, yk])}
    print(f"  いまの行との近さ（同じ銘柄のいまの行が 直前 / 直後 {COOLDOWN}営業日以内に）: " + " / ".join(
        f"{n} {v['rows']:,}（{pct(v['pos'])}）" for n, v in near.items()))
    ov = int(keys_of(co).isin(keys_of(ad)).sum())
    print(f"  案1 の足せる行と同じ銘柄×日付: {ov:,}（close だけの行の {pct(ov / len(co) if len(co) else np.nan)}）")
    un = pd.concat([dsp[["Code", "mpos"]], co[["Code", "mpos"]]], ignore_index=True)
    u_ds = avg_uniqueness(dsp["Code"], dsp["mpos"])
    u_un = avg_uniqueness(un["Code"], un["mpos"])
    print(f"  いまの行 + close だけの行: {len(un):,}行 → 実効の行数 {np.nansum(u_un):,.0f}（いまの行だけ {np.nansum(u_ds):,.0f}。"
          "営業日は日足にある日の通し番号で数えた）")
    summ["plan2"] = {"close_rows": int(len(cl)), "close_only": int(len(co)),
                     "pos": {str(k): rate(co[ycol(k)]) for k in KS}, "near": near, "overlap_plan1": ov,
                     "effective_fresh": float(np.nansum(u_ds)), "effective_union": float(np.nansum(u_un))}
    return summ


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--no-close", action="store_true", help="案2（終値の母集団）を作らない")
    ap.add_argument("--rebuild", action="store_true", help="前の回の表を使わずに作り直す")
    args = ap.parse_args(argv)
    t0 = time.time()
    tables = build_tables(args.data_dir, with_close=not args.no_close, rebuild=args.rebuild)
    summ = report(tables)
    os.makedirs(oof_dir(args.data_dir), exist_ok=True)
    with open(fpath(args.data_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1, default=str)
    log(f"[done] {time.time() - t0:.0f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
