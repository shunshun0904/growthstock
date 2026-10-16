#!/usr/bin/env python3
"""
実験99: 候補の「ストップ高」の頻度 — 今の規則（1.5σ の3本がすべて95以上）の行は、同じ窓・同じ時価総額の帯の候補より、買った
あと20営業日の間にストップ高になりやすいか（判定）。時価総額の帯・3本の最小の帯ごとの割合、基準日より前のストップ高の回数、
収益との関係（参考）。本番・画面は変えない。

運用者の問い（2026-10-16）「予測モデルによる銘柄候補で選んだ、特に小型株（時価総額300〜400億円以内くらい）が結果的にストップ高に
なっている割合が高い。これは過去の傾向から分かるか。過去にストップ高があったかどうかは過去の日足から分かるか。モデルがちゃんと
見極めて、結果的にストップ高になりそうな銘柄を選定できているのなら、バックテストでも検証したい」。

ストップ高の判定（research/price_limits.py）
------------------------------------------
  保存済みの日足には J-Quants のフラグ（UL / LL）が無いので、東証の制限値幅の表から判定する。基準値段 = 前営業日の終値（売買の
  無い日は最後の終値。分割は AdjC ÷ C の比で直す）、制限値幅 = 表の値。「ストップ高」= その日の高値が 基準値段 + 制限値幅 以上
  （最終気配が基準値段になった日・制限値幅の4倍拡大の日も含む）。張り付き = 終値が同じ水準以上。ストップ安 = 安値が
  基準値段 − 制限値幅 以下。実測（2023〜2026 の日足 387万行）: 高値がちょうど上限の日 9,303、上限を超える日 854（0.02%）。
  API のフラグとの一致は research/probe_limit_flags.py で測る（鍵が要るので Actions で回す）

比べ方（先に決めたもの。結果を見てから動かさない）
------------------------------------------
  行: 12本の out-of-fold を突き合わせた候補（78週高値の更新日）。百分位は前向き（テスト窓 3〜11。それより前の窓の分布で付け直す。
      実験88 と同じ）。規則の行 = 1.5σ の lgbm_k15・xgb_k15・cat_k15 の百分位（画面と同じ丸め）の最小が 95 以上
  結果: ul_fwd = 買った日（翌営業日 = 1日目）から20営業日目までにストップ高の日が1日でもあれば 1、無ければ 0（20営業日の日足が
      そろわない行は数えない）
  層: テスト窓 × 時価総額の帯（基準日の時価総額（億円）。build_dataset.CAP_BAND_EDGES の 100 / 300 / 1,000 / 3,000 の5帯。
      時価総額の無い行は外す）
  判定1（全体）: D = 層ごとの「規則の行の ul_fwd の割合 − それ以外の候補の割合」を規則の行の数で重み付けした平均（pt）。層の中で
      比べるので、規則の行が小型株に偏っていても帯の違いそのものでは差が出ない。並べ替え検定は層の中で規則の印を並べ替える
      （2,000回。片側 p = 並べ替えた D が実際以上になる割合）。銘柄単位（同じ層・同じ銘柄・同じ側の行を平均して1つ）でも同じ p。
      「支持」= D > 0・行の p < 0.05・銘柄単位の p < 0.05 の3つ。そうでなければ「支持しない」
  判定2（小型株）: 時価総額 1,000億円未満（帯 0〜2）の行だけで、判定1 と同じ3条件（線は 2026-10-16 の運用者の選択。初案は 300億円
      未満だったが、回す前に 1,000億円未満に決めた）
  参考（判定に使わない）
    - 帯ごとの割合（規則の行・それ以外・候補全体）と差。細かい帯（〜100 / 100〜300 / 300〜500 / 500〜1,000 / 1,000〜3,000 / 3,000〜）
    - 候補全体を 3本の最小の帯（〜70 / 70〜85 / 85〜95 / 95〜）× 時価総額の帯で
    - 張り付き（終値）・ストップ安・20営業日の中のストップ高の日数・最初の日、基準日そのものがストップ高の割合
    - 基準日より前250営業日のストップ高の回数（規則の行とそれ以外）と、回数の有無で分けた ul_fwd の割合（候補全体・帯ごと）
    - 収益との関係（規則の行: ストップ高あり/なしの +10% の出口の平均・到達の割合、到達の日がストップ高の日である割合）
    - 年ごと、以前の規則（1.2σ の3本 90以上）、画面の物差し（全期間の百分位）
  出すもの: 件数・割合・平均。公開ログには集計だけ。銘柄・日付ごとの値は出さない

    exp=e99_limit_up.py
    args="--perm 2000"        （既定）

結果（2026-10-16、run 38844915486、009943d、51秒。docs/MODEL_ADOPTION_RULES.md §59）
------------------------------------------------------------------
  判定1: 支持しない。規則の行 559件のうち買ったあと20営業日にストップ高があったのは 3.9%、同じ窓・同じ帯のそれ以外の候補は 5.7%。
        層の中の差 D −2.74pt・片側 p = 0.998（両側 0.007）、銘柄単位 D −3.02pt・片側 p = 1.000（両側 0.003）。向きは逆で、規則の行の
        ほうがストップ高になりにくい
  判定2（1,000億円未満）: 支持しない。447件 4.9% 対 8,753件 8.0%、D −3.02pt・片側 p = 0.995（両側 0.014）、銘柄単位 −3.33pt
  帯ごと（規則 / それ以外）: 〜100億 7.0% / 19.0%（100件）、100〜300億 3.1% / 6.4%（194件）、300〜1,000億 5.9% / 3.4%（153件。
        細かく 300〜500億 7.1% / 3.8%（70件・5件）、500〜1,000億 4.8% / 3.1%）、1,000〜3,000億 0.0% / 2.1%（60件）、3,000億〜 0.0% / 1.6%
  3本の最小の帯（候補全体）: 〜70 6.3% / 70〜85 3.7% / 85〜95 4.0% / 95〜 3.9%。〜100億だけ見ると 23.6% → 7.0% と、モデルが高く見る
        ほどストップ高は減る
  過去: 基準日そのものがストップ高だった行は規則の行 0.2%（それ以外 7.6%）。前250営業日のストップ高は回数の平均 0.42 対 0.49・
        1回以上 22.5% 対 22.2% で違いが無い。候補全体では、前250営業日にストップ高が有った行 3,195件の 16.5% があとの20営業日にも
        ストップ高（無かった行 11,187件は 2.6%。どの帯でも4〜6倍）、基準日がストップ高の行は 33.6%（それ以外 3.5%）
  収益（規則の行）: あとにストップ高があった 22件は全部 +10% に届いた。無かった 537件は +2.81%（到達 38.0%・−10%割れ 5.6%）。
        到達した 226件のうち到達の日がストップ高の日 8.0%。候補全体では あり 816件 +6.54% / なし 13,565件 +0.55%
  その他: 張り付き 3.4% / 4.4%、ストップ安 1.4% / 2.4%、最初の日の中央 10日目 / 3日目。年ごとの規則の行 3〜7%。画面の物差しでも
        同じ向き（354件 3.7% 対 5.7%、D −3.48pt・両側 p = 0.011）
  読み方: モデルの選定は「ストップ高になりそうな銘柄」を拾ってはいない。むしろ避けている（基準日がストップ高の候補をほぼ選ばず、
        目的変数も終盤・トレンドの条件で往って来いを外す作り）。小型株の候補がストップ高になった印象は、件数の少ない帯（300〜500億
        70件のうち 5件）の偶然の範囲。過去のストップ高は先のストップ高をよく当てる（時点で分かる情報）が、それが目的変数・収益に
        効くかは別の問い → 過去の有無で分けた収益（上の参考。2回目の run で足した）を見て、A/B（§7 の手順）を回すかを運用者が決める
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
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import price_limits as PL  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e83_tp10_rule as E83  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402

MODELS = E79.MODELS
#: (記号, 表示, 最小の列, 線)
RULE = ("K15", "今の規則: 1.5σ の3本 95以上", "min_K15", 95.0)
PREV = ("T3_90", "以前の規則: 1.2σ の3本 90以上", "min_T3", 90.0)
HOLD = LT.HOLD                              # 20営業日（買った日 = 1日目）
PAST = 250                                  # 基準日より前に数える営業日
MAIN = "ul_fwd"                             # 判定に使う結果（高値がストップ高の水準に達した日があるか）
CAP_EDGES = B.CAP_BAND_EDGES                # (100, 300, 1000, 3000) 億円
CAP_LABELS = ("〜100億", "100〜300億", "300〜1,000億", "1,000〜3,000億", "3,000億〜")
FINE_EDGES = (100.0, 300.0, 500.0, 1000.0, 3000.0)
FINE_LABELS = ("〜100億", "100〜300億", "300〜500億", "500〜1,000億", "1,000〜3,000億", "3,000億〜")
SMALL_CAP_MAX = 1000.0                      # 判定2 の「小型株」= 時価総額がこれ未満（億円。2026-10-16 の運用者の選択）
PCT_BANDS = ((0.0, 70.0, "〜70"), (70.0, 85.0, "70〜85"), (85.0, 95.0, "85〜95"), (95.0, 101.0, "95〜"))
N_PERM = 2000
SEED = 42
OUT_DIR = E79.OUT_DIR

log = E79.log

# --------------------------------------------------------------------------- #
# 列を付ける
# --------------------------------------------------------------------------- #


def fine_band(market_cap) -> pd.Series:
    """細かい時価総額の帯（0〜5）。欠測は欠測のまま。"""
    x = pd.to_numeric(pd.Series(market_cap), errors="coerce")
    band = pd.Series(np.digitize(x.fillna(-1.0), FINE_EDGES), index=x.index, dtype=float)
    return band.where(x.notna())


def attach_caps(df: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """基準日の時価総額（億円。データセットの market_cap）と帯を付ける。"""
    f = frame[["Code", "Date", "market_cap"]].copy()
    f["Code"] = f["Code"].astype(str)
    f["Date"] = pd.to_datetime(f["Date"])
    f = f.drop_duplicates(["Code", "Date"])
    out = df.drop(columns=[c for c in ("market_cap", "cap_band", "fine_band") if c in df.columns])
    out = out.merge(f, on=["Code", "Date"], how="left")
    out["cap_band"] = B.cap_band(out["market_cap"]).to_numpy()
    out["fine_band"] = fine_band(out["market_cap"]).to_numpy()
    return out


def attach_limits(df: pd.DataFrame, arrays: dict, hold: int = HOLD, past: int = PAST) -> pd.DataFrame:
    """ストップ高の列（price_limits.limit_events）と、+10% に届いた日がストップ高の日か（hit_on_ul）を付ける。"""
    ev = PL.limit_events(arrays, df[["Code", "Date"]], hold=hold, past=past)
    out = df.drop(columns=[c for c in ev.columns if c not in ("Code", "Date") and c in df.columns])
    out = out.merge(ev, on=["Code", "Date"], how="left")
    if "hitday10" in out.columns:
        out["hit_on_ul"] = PL.day_flag(arrays, out["Code"].to_numpy(), out["Date"].to_numpy(),
                                       pd.to_numeric(out["hitday10"], errors="coerce").to_numpy(dtype=float))
    return out


def rule_mask(ft: pd.DataFrame, col: str | None, line: float | None) -> np.ndarray:
    if col is None:
        return np.ones(len(ft), dtype=bool)
    return ft[col].to_numpy(dtype=float) >= line


# --------------------------------------------------------------------------- #
# 比べ方
# --------------------------------------------------------------------------- #


def strat_perm(v, f, strata, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """
    D = 層ごとの（印の側の平均 − 残りの平均）を印の側の件数で重み付けした平均。両側のある層だけで数える。
    並べ替えは層の中で印を入れ替える。p_more = 並べ替えた D が実際以上になる割合（片側。「印の側が高い」）、
    p_two = 絶対値で実際以上。D_pooled は全部を足した平均の差（帯の偏りを含むので参考）。
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=bool)
    s = np.asarray(strata)
    ok = np.isfinite(v)
    v, f, s = v[ok], f[ok], s[ok]
    nan = float("nan")
    base = {"D": nan, "D_pooled": nan, "p_more": nan, "p_two": nan, "n_yes": int(f.sum()), "n_no": int((~f).sum()),
            "strata_used": 0, "rate_yes": nan, "rate_no": nan}
    if f.sum() == 0 or (~f).sum() == 0:
        return base
    base["rate_yes"] = float(v[f].mean() * 100)
    base["rate_no"] = float(v[~f].mean() * 100)
    base["D_pooled"] = float((v[f].mean() - v[~f].mean()) * 100)
    _, inv = np.unique(s, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(inv.max() + 1)]
    groups = [g for g in groups if 0 < f[g].sum() < len(g)]
    base["strata_used"] = len(groups)
    if not groups:
        return base
    wts = np.array([f[g].sum() for g in groups], dtype=float)

    def stat(ff: np.ndarray) -> float:
        d = np.array([v[g][ff[g]].mean() - v[g][~ff[g]].mean() for g in groups])
        return float((d * wts).sum() / wts.sum() * 100)

    obs = stat(f)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        fp = f.copy()
        for g in groups:
            fp[g] = f[g][rng.permutation(len(g))]
        null[k] = stat(fp)
    base.update(D=obs, p_more=float((np.sum(null >= obs) + 1) / (n_perm + 1)),
                p_two=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)))
    return base


def code_level(g: pd.DataFrame, flag: np.ndarray, strata: np.ndarray, col: str = MAIN):
    """同じ層・同じ銘柄・同じ側の行を平均して1つにした (値, 印, 層)。"""
    x = pd.DataFrame({"v": pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float), "f": np.asarray(flag, dtype=bool),
                      "s": np.asarray(strata), "c": g["Code"].to_numpy()}).dropna(subset=["v"])
    a = x.groupby(["s", "c", "f"], as_index=False)["v"].mean()
    return a["v"].to_numpy(), a["f"].to_numpy(dtype=bool), a["s"].to_numpy()


def strata_of(g: pd.DataFrame) -> np.ndarray:
    """層 = テスト窓 × 時価総額の帯。"""
    return (g["window"].to_numpy(dtype=int) * 10 + g["cap_band"].to_numpy(dtype=float).astype(int))


def rule_test(ft: pd.DataFrame, col: str = MAIN, rule=RULE, sub: np.ndarray | None = None,
              n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """規則の行 − それ以外の候補（同じ窓・同じ帯の中で）。行と銘柄単位の並べ替え検定、帯ごとの割合、判定。"""
    g = ft[ft["cap_band"].notna() & ft["window"].notna()]
    if sub is not None:
        g = g[np.asarray(sub, dtype=bool)[ft.index.get_indexer(g.index)] if len(sub) == len(ft) else sub]
    f = rule_mask(g, rule[2], rule[3])
    s = strata_of(g)
    v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    row = strat_perm(v, f, s, n_perm, seed)
    cv, cf, cs = code_level(g, f, s, col)
    code = strat_perm(cv, cf, cs, n_perm, seed)
    bands = {}
    for b, name in enumerate(CAP_LABELS):
        m = g["cap_band"].to_numpy(dtype=float) == b
        yes = v[m & f]
        no = v[m & ~f]
        yes, no = yes[np.isfinite(yes)], no[np.isfinite(no)]
        bands[name] = {"n_yes": int(len(yes)), "n_no": int(len(no)),
                       "rate_yes": float(yes.mean() * 100) if len(yes) else float("nan"),
                       "rate_no": float(no.mean() * 100) if len(no) else float("nan")}
        bands[name]["diff"] = bands[name]["rate_yes"] - bands[name]["rate_no"]
    support = bool(np.isfinite(row["D"]) and row["D"] > 0 and row["p_more"] < 0.05 and code["p_more"] < 0.05)
    return {"row": row, "code": code, "bands": bands, "support": support, "n_rows": int(len(g))}


# --------------------------------------------------------------------------- #
# 参考の集計
# --------------------------------------------------------------------------- #


def rate(x) -> float:
    r = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(r.mean() * 100) if len(r) else float("nan")


def n_ok(x) -> int:
    return int(pd.to_numeric(pd.Series(x), errors="coerce").notna().sum())


def band_rates(g: pd.DataFrame, band_col: str, labels, f: np.ndarray, col: str = MAIN) -> dict:
    """帯ごとに、印の行・それ以外・全部の割合と件数。"""
    out = {}
    b = g[band_col].to_numpy(dtype=float)
    v = g[col]
    for k, name in enumerate(labels):
        m = b == k
        out[name] = {"n_yes": n_ok(v[m & f]), "rate_yes": rate(v[m & f]), "n_no": n_ok(v[m & ~f]), "rate_no": rate(v[m & ~f]),
                     "n_all": n_ok(v[m]), "rate_all": rate(v[m])}
    out["全体"] = {"n_yes": n_ok(v[f]), "rate_yes": rate(v[f]), "n_no": n_ok(v[~f]), "rate_no": rate(v[~f]),
                 "n_all": n_ok(v), "rate_all": rate(v)}
    return out


def pct_band_table(g: pd.DataFrame, col: str = MAIN, pct_col: str = RULE[2]) -> dict:
    """3本の最小の帯 × 時価総額の帯の割合（候補全体）。"""
    p = g[pct_col].to_numpy(dtype=float)
    b = g["cap_band"].to_numpy(dtype=float)
    out = {}
    for lo, hi, name in PCT_BANDS:
        m = (p >= lo) & (p < hi)
        row = {"全体": {"n": n_ok(g[col][m]), "rate": rate(g[col][m])}}
        for k, bname in enumerate(CAP_LABELS):
            mm = m & (b == k)
            row[bname] = {"n": n_ok(g[col][mm]), "rate": rate(g[col][mm])}
        out[name] = row
    return out


def past_table(g: pd.DataFrame, f: np.ndarray, col: str = MAIN) -> dict:
    """基準日より前のストップ高（回数）と、その有無で分けた ul_fwd の割合。"""
    past = pd.to_numeric(g["ul_past_n"], errors="coerce")
    ul_t = pd.to_numeric(g["ul_t"], errors="coerce")
    out = {}
    for name, m in (("規則の行", f), ("それ以外", ~f), ("候補全体", np.ones(len(g), dtype=bool))):
        p = past[m].dropna()
        out[name] = {"n": int(len(p)), "mean_past": float(p.mean()) if len(p) else float("nan"),
                     "share_past_any": float((p > 0).mean() * 100) if len(p) else float("nan"),
                     "share_past_3plus": float((p >= 3).mean() * 100) if len(p) else float("nan"),
                     "rate_ul_t": rate(ul_t[m]), "n_ul_t": n_ok(ul_t[m])}
    # 回数の有無で分けた ul_fwd（候補全体・帯ごと）
    has = (past > 0).to_numpy()
    known = past.notna().to_numpy()
    b = g["cap_band"].to_numpy(dtype=float)
    lift = {"全体": {"n_has": n_ok(g[col][known & has]), "rate_has": rate(g[col][known & has]),
                   "n_none": n_ok(g[col][known & ~has]), "rate_none": rate(g[col][known & ~has])}}
    for k, bname in enumerate(CAP_LABELS):
        mm = known & (b == k)
        lift[bname] = {"n_has": n_ok(g[col][mm & has]), "rate_has": rate(g[col][mm & has]),
                       "n_none": n_ok(g[col][mm & ~has]), "rate_none": rate(g[col][mm & ~has])}
    # 基準日そのものがストップ高の行の ul_fwd
    t1 = (ul_t == 1).to_numpy()
    t0 = (ul_t == 0).to_numpy()
    lift["基準日がストップ高"] = {"n_has": n_ok(g[col][t1]), "rate_has": rate(g[col][t1]),
                           "n_none": n_ok(g[col][t0]), "rate_none": rate(g[col][t0])}
    return {"groups": out, "lift": lift}


def returns_table(g: pd.DataFrame, f: np.ndarray, col: str = MAIN) -> dict:
    """規則の行の収益（+10% の指値・20日目の終値）を、ストップ高の有無で分ける。到達の日がストップ高の日である割合も。"""
    r = g[f]
    ul = pd.to_numeric(r[col], errors="coerce")
    tp = pd.to_numeric(r["tp10c"], errors="coerce")
    hit = r["hit10"].astype(bool) if "hit10" in r.columns else pd.Series(False, index=r.index)
    out = {}
    for name, m in (("ストップ高あり", (ul == 1).to_numpy()), ("ストップ高なし", (ul == 0).to_numpy())):
        t = tp[m].dropna()
        out[name] = {"n": int(len(t)), "mean": float(t.mean()) if len(t) else float("nan"),
                     "med": float(t.median()) if len(t) else float("nan"),
                     "win": float((t > 0).mean() * 100) if len(t) else float("nan"),
                     "hit": float(hit[m].mean() * 100) if m.sum() else float("nan"),
                     "bad10": float((t < -10).mean() * 100) if len(t) else float("nan")}
    hu = pd.to_numeric(r["hit_on_ul"], errors="coerce") if "hit_on_ul" in r.columns else pd.Series(dtype=float)
    hu = hu[hit.to_numpy() if len(hu) else []] if len(hu) else hu
    out["到達の日がストップ高"] = {"n": int(hu.notna().sum()), "share": rate(hu)}
    # 候補全体でも同じ内訳（参考）
    ul_all = pd.to_numeric(g[col], errors="coerce")
    tp_all = pd.to_numeric(g["tp10c"], errors="coerce")
    out["候補全体"] = {name: {"n": int(tp_all[m].notna().sum()), "mean": float(tp_all[m].mean())}
                   for name, m in (("ストップ高あり", (ul_all == 1).to_numpy()), ("ストップ高なし", (ul_all == 0).to_numpy()))}
    # 基準日より前250営業日のストップ高の有無（時点で分かる情報）で分けた収益。**結果を見てから足した参考**（1回目の run
    # 38844915486 には無い。過去の回数を特徴量にする A/B（§7）を回すかの材料。判定には使わない）
    past = pd.to_numeric(g["ul_past_n"], errors="coerce")
    hit_all = g["hit10"].astype(bool) if "hit10" in g.columns else pd.Series(False, index=g.index)
    out["過去"] = {}
    for label, grp in (("候補全体", np.ones(len(g), dtype=bool)), ("規則の行", np.asarray(f, dtype=bool))):
        for name, m in (("過去あり", (past > 0).to_numpy()), ("過去なし", (past == 0).to_numpy())):
            mm = grp & m
            t = tp_all[mm].dropna()
            out["過去"][f"{label}_{name}"] = {"n": int(len(t)), "mean": float(t.mean()) if len(t) else float("nan"),
                                           "hit": float(hit_all[mm].mean() * 100) if mm.sum() else float("nan"),
                                           "ul_fwd": rate(ul_all[mm])}
    return out


def year_table(g: pd.DataFrame, f: np.ndarray, col: str = MAIN) -> dict:
    y = pd.to_datetime(g["Date"]).dt.year.to_numpy()
    out = {}
    for yy in sorted(set(y.tolist())):
        m = y == yy
        out[int(yy)] = {"n_yes": n_ok(g[col][m & f]), "rate_yes": rate(g[col][m & f]),
                        "n_no": n_ok(g[col][m & ~f]), "rate_no": rate(g[col][m & ~f])}
    return out


def extras(g: pd.DataFrame, f: np.ndarray) -> dict:
    """張り付き・ストップ安・日数・最初の日（規則の行・それ以外）。"""
    out = {}
    for name, m in (("規則の行", f), ("それ以外", ~f)):
        x = g[m]
        n_days = pd.to_numeric(x["ul_fwd_n"], errors="coerce").dropna()
        first = pd.to_numeric(x["ul_fwd_first"], errors="coerce").dropna()
        out[name] = {"n": n_ok(x[MAIN]), "ul": rate(x[MAIN]), "ul_close": rate(x["ulc_fwd"]), "ll": rate(x["ll_fwd"]),
                     "days_mean": float(n_days.mean()) if len(n_days) else float("nan"),
                     "days_2plus": float((n_days >= 2).mean() * 100) if len(n_days) else float("nan"),
                     "first_med": float(first.median()) if len(first) else float("nan"),
                     "first_le5": float((first <= 5).mean() * 100) if len(first) else float("nan")}
    return out


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #


def fmt_t(t: dict) -> str:
    if not np.isfinite(t.get("D", np.nan)):
        return f"—（片側が0件。{t.get('n_yes', 0)}件 対 {t.get('n_no', 0)}件）"
    return (f"D {t['D']:+.2f}pt（層の中の差。{t['n_yes']}件 {t['rate_yes']:.1f}% 対 {t['n_no']}件 {t['rate_no']:.1f}%・"
            f"全部を足した差 {t['D_pooled']:+.2f}pt）・片側 p = {t['p_more']:.3f}・両側 p = {t['p_two']:.3f}・層 {t['strata_used']}")


def band_lines(bands: dict, width: int = 16) -> list[str]:
    return [f"  {LT._l(name, width)}規則 {b['n_yes']:>4}件 {b['rate_yes']:5.1f}%  それ以外 {b['n_no']:>5}件 {b['rate_no']:5.1f}%  "
            f"差 {b['diff']:+5.1f}pt" for name, b in bands.items()]


def verdict(t: dict) -> str:
    r, c = t["row"], t["code"]
    d_ok = np.isfinite(r["D"]) and r["D"] > 0
    return (f"{'支持する' if t['support'] else '支持しない'}（D > 0 {'○' if d_ok else '×'} / 行の p < 0.05 "
            f"{'○' if r['p_more'] < 0.05 else '×'} / 銘柄単位の p < 0.05 {'○' if c['p_more'] < 0.05 else '×'}）")


def report(ft: pd.DataFrame, ff: pd.DataFrame | None = None, flags_info: dict | None = None,
           n_perm: int = N_PERM, seed: int = SEED) -> dict:
    summary: dict = {}
    f = rule_mask(ft, RULE[2], RULE[3])
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    ok = pd.to_numeric(ft[MAIN], errors="coerce").notna().to_numpy()
    has_cap = ft["cap_band"].notna().to_numpy()
    print(f"\n■ 0. 前向き {len(ft):,}行・{months}か月・テスト窓 {ft['window'].nunique()}つ。規則の行 {int(f.sum())}件。"
          f"20営業日の結果がある行 {int(ok.sum()):,}（{ok.mean() * 100:.1f}%）・時価総額の帯がある行 {int(has_cap.sum()):,}"
          f"（{has_cap.mean() * 100:.1f}%）")
    if flags_info:
        print(f"  日足の判定: {flags_info['rows']:,}行のうち判定できた {flags_info['known_share']:.1f}%・ストップ高の日 "
              f"{flags_info['ul']:,}（判定できた日の {flags_info['ul_share']:.2f}%。高値がちょうど上限 {flags_info['exact_among_ul']:.1f}%）・"
              f"張り付き {flags_info['ul_close']:,}・ストップ安 {flags_info['ll']:,}")
    counts = ft.loc[has_cap, "cap_band"].astype(int).value_counts().sort_index()
    print("  時価総額の帯（候補全体 / 規則の行）: " + "  ".join(
        f"{CAP_LABELS[int(k)]} {int(n)} / {int((f & (ft['cap_band'].to_numpy(dtype=float) == k)).sum())}" for k, n in counts.items()))
    summary["rows"] = {"all": int(len(ft)), "rule": int(f.sum()), "with_outcome": int(ok.sum()), "with_cap": int(has_cap.sum()),
                       "months": months, "flags": flags_info}

    # ---- 1. 判定1
    t1 = rule_test(ft, MAIN, RULE, n_perm=n_perm, seed=seed)
    print(f"\n■ 1. 判定1（全体）: 規則の行 − それ以外の候補（同じ窓・同じ時価総額の帯）。結果 = 翌営業日から20営業日目までにストップ高")
    print(f"  行: {fmt_t(t1['row'])}")
    print(f"  銘柄単位: {fmt_t(t1['code'])}")
    for line in band_lines(t1["bands"]):
        print(line)
    print(f"  判定1: {verdict(t1)}")
    summary["main"] = t1

    # ---- 2. 判定2
    small = ft["market_cap"].to_numpy(dtype=float) < SMALL_CAP_MAX
    t2 = rule_test(ft, MAIN, RULE, sub=small, n_perm=n_perm, seed=seed)
    print(f"\n■ 2. 判定2（小型株 = 時価総額 {SMALL_CAP_MAX:.0f}億円未満）: 同じ比べ方")
    print(f"  行: {fmt_t(t2['row'])}")
    print(f"  銘柄単位: {fmt_t(t2['code'])}")
    print(f"  判定2: {verdict(t2)}")
    summary["small"] = t2

    # ---- 3. 帯ごと（細かく）・以前の規則
    print("\n■ 3. 参考: 細かい時価総額の帯ごとの割合（前向き）")
    fine = band_rates(ft, "fine_band", FINE_LABELS, f)
    for name, b in fine.items():
        print(f"  {LT._l(name, 16)}規則 {b['n_yes']:>4}件 {b['rate_yes']:5.1f}%  それ以外 {b['n_no']:>5}件 {b['rate_no']:5.1f}%  "
              f"候補全体 {b['n_all']:>5}件 {b['rate_all']:5.1f}%")
    fp = rule_mask(ft, PREV[2], PREV[3])
    prev = band_rates(ft, "cap_band", CAP_LABELS, fp)
    print(f"  {PREV[1]}: 全体 規則 {prev['全体']['n_yes']}件 {prev['全体']['rate_yes']:.1f}% 対 それ以外 {prev['全体']['n_no']}件 "
          f"{prev['全体']['rate_no']:.1f}%・帯ごとの差 " + " / ".join(
              f"{name} {b['rate_yes'] - b['rate_no']:+.1f}pt" for name, b in prev.items() if name != "全体"))
    summary["fine"] = fine
    summary["prev"] = prev

    # ---- 4. 3本の最小の帯
    print("\n■ 4. 参考: 3本の最小の帯 × 時価総額の帯（候補全体・前向き。ストップ高の割合）")
    pb = pct_band_table(ft)
    print("  " + LT._l("3本の最小", 10) + "".join(LT._r(name, 14) for name in ("全体",) + CAP_LABELS))
    for name, row in pb.items():
        print("  " + LT._l(name, 10) + "".join(LT._r(f"{c['rate']:.1f}%({c['n']})" if c["n"] else "—", 14)
                                            for c in [row["全体"]] + [row[b] for b in CAP_LABELS]))
    summary["pct_bands"] = pb

    # ---- 5. 張り付き・ストップ安・日数・最初の日・過去
    print("\n■ 5. 参考: 張り付き・ストップ安・日数・最初の日（前向き）")
    ex = extras(ft, f)
    for name, e in ex.items():
        print(f"  {LT._l(name, 10)}{e['n']:>5}件  ストップ高 {e['ul']:5.1f}%  張り付き {e['ul_close']:5.1f}%  ストップ安 {e['ll']:5.1f}%  "
              f"日数の平均 {e['days_mean']:.2f}（2日以上 {e['days_2plus']:.1f}%）  最初の日の中央 {e['first_med']:.0f}日目"
              f"（5日目まで {e['first_le5']:.1f}%）")
    pt = past_table(ft, f)
    print("  基準日より前250営業日のストップ高:")
    for name, p in pt["groups"].items():
        print(f"    {LT._l(name, 10)}{p['n']:>5}件  回数の平均 {p['mean_past']:.2f}  1回以上 {p['share_past_any']:.1f}%  3回以上 "
              f"{p['share_past_3plus']:.1f}%  基準日そのものがストップ高 {p['rate_ul_t']:.1f}%（{p['n_ul_t']}件）")
    print("  前250営業日にストップ高が有った行 / 無かった行の、買ったあとのストップ高の割合（候補全体）:")
    for name, lf in pt["lift"].items():
        print(f"    {LT._l(name, 16)}有り {lf['n_has']:>5}件 {lf['rate_has']:5.1f}%  無し {lf['n_none']:>5}件 {lf['rate_none']:5.1f}%")
    summary["extras"] = ex
    summary["past"] = pt

    # ---- 6. 収益
    print("\n■ 6. 参考: 規則の行の収益（+10% の指値・届かなければ20日目の終値）を、買ったあとのストップ高の有無で")
    rt = returns_table(ft, f)
    for name in ("ストップ高あり", "ストップ高なし"):
        s = rt[name]
        print(f"  {LT._l(name, 12)}{s['n']:>4}件  +10% {s['mean']:+6.2f}%（中央 {s['med']:+5.2f}）  勝率 {s['win']:4.1f}%  "
              f"到達 {s['hit']:4.1f}%  −10%割れ {s['bad10']:4.1f}%")
    print(f"  到達した行のうち、到達の日がストップ高の日: {rt['到達の日がストップ高']['share']:.1f}%（{rt['到達の日がストップ高']['n']}件）")
    print("  候補全体: " + " / ".join(f"{k} {v['n']}件 {v['mean']:+.2f}%" for k, v in rt["候補全体"].items()))
    print("  前250営業日のストップ高の有無で分けた収益（結果を見てから足した参考。判定には使わない）:")
    for label in ("候補全体", "規則の行"):
        print("    " + LT._l(label, 10) + "  ".join(
            f"{name} {v['n']}件 {v['mean']:+.2f}%（到達 {v['hit']:.1f}%・あとのストップ高 {v['ul_fwd']:.1f}%）"
            for name in ("過去あり", "過去なし") for k, v in rt["過去"].items() if k == f"{label}_{name}"))
    summary["returns"] = rt

    # ---- 7. 年ごと
    print("\n■ 7. 参考: 年ごと（前向き。規則の行 / それ以外）")
    yt = year_table(ft, f)
    print("  " + "  ".join(f"{y} {v['rate_yes']:.0f}%({v['n_yes']}) / {v['rate_no']:.0f}%({v['n_no']})" for y, v in yt.items()))
    summary["years"] = yt

    # ---- 8. 画面の物差し
    if ff is not None and len(ff):
        print("\n■ 8. 参考: 画面の物差し（全期間の百分位）で判定1 と同じ比べ方")
        tf = rule_test(ff, MAIN, RULE, n_perm=n_perm, seed=seed)
        print(f"  行: {fmt_t(tf['row'])}")
        print(f"  銘柄単位: {fmt_t(tf['code'])}")
        for line in band_lines(tf["bands"]):
            print(line)
        print(f"  同じ形の判定: {verdict(tf)}")
        summary["full"] = tf

    print("\n■ まとめ")
    r1, r2 = t1["row"], t2["row"]
    print(f"  判定1（全体）: 規則の行 {r1['n_yes']}件 {r1['rate_yes']:.1f}% 対 それ以外 {r1['rate_no']:.1f}%、層の中の差 D {r1['D']:+.2f}pt・"
          f"片側 p = {r1['p_more']:.3f}（銘柄単位 {t1['code']['p_more']:.3f}）→ {'支持する' if t1['support'] else '支持しない'}")
    print(f"  判定2（{SMALL_CAP_MAX:.0f}億円未満）: 規則の行 {r2['n_yes']}件 {r2['rate_yes']:.1f}% 対 それ以外 {r2['rate_no']:.1f}%、"
          f"D {r2['D']:+.2f}pt・片側 p = {r2['p_more']:.3f}（銘柄単位 {t2['code']['p_more']:.3f}）→ "
          f"{'支持する' if t2['support'] else '支持しない'}")
    print("  注意: ストップ高は保存済みの四本値と制限値幅の表から判定したもの（API のフラグではない）。確かめられるのは同じ期間の "
          "out-of-fold だけ")
    return summary


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験99: 候補のストップ高の頻度（今の規則 対 それ以外。時価総額の帯の中で）")
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
    first = pd.to_datetime(df["Date"]).min()
    bars = LT.load_bars(lab.DATA_DIR, start=first - pd.Timedelta(days=500), extra=("H", "L", "C"))
    log(f"日足 {len(bars):,}行（{bars['Date'].min().date()} 〜 {bars['Date'].max().date()}）")
    df, chk = E83.attach_exits(df, bars, targets=E87.TARGETS)
    log(f"出口: 20日目の終値 {chk['with_close']:,}/{chk['rows']:,}行・到達の順序の食い違い {chk['order_violations']}件")
    flags = PL.flag_limits(bars)
    info = PL.flags_summary(flags)
    arrays = PL.code_arrays(flags)
    log(f"ストップ高の判定: {info['rows']:,}行・判定できた {info['known_share']:.1f}%・ストップ高の日 {info['ul']:,}"
        f"（ちょうど {info['exact_among_ul']:.1f}%）")
    split = E79.window_split(df, folds)
    ft = E87.build(df, [(te, Pte) for _, (tr, te, Ptr, Pte) in split.items()])
    ff = E87.build(df, [(np.arange(len(df)), E79.pct_matrix(df, MODELS))])
    ft = attach_caps(attach_limits(ft, arrays), fr)
    ff = attach_caps(attach_limits(ff, arrays), fr)
    log(f"前向きの行 {len(ft):,}件（テスト窓 {len(split)}つ）。規則の行 {int(rule_mask(ft, RULE[2], RULE[3]).sum())}件"
        f"（実験88 は 559件）")
    summary = report(ft, ff=ff, flags_info=info, n_perm=args.perm)
    summary["exit_check"] = chk
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "e99_summary.json")

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
