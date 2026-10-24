#!/usr/bin/env python3
"""
運用者の仮説（2026-10-24）「信用倍率、出来高増加率、価格の増加率、ボラティリティ、移動平均あたりの変数だけで、ある程度予想
できませんか？ 上記5変数でクラスタリングし、各クラスタごとで、目的変数の比率などをみてみたいです。おそらくある特定のクラスタ
だけで、ラベル1が多くなるみたいなことはあるのかなと。逆にラベル0が多くなる等も。」（見るだけ。本番は変えない）

5変数（運用者の選択。AskUserQuestion）:
  信用倍率 = credit_ratio（買残 ÷ 売残）・出来高増加率 = volume_trend（その日の出来高 ÷ 直前20日平均、%）・
  価格の増加率 = ret_20d（20日上昇率、%）・ボラティリティ = vol_20d（20日の日次リターンの標準偏差、%）・
  移動平均 = tx_ma_ratio_25_75（25日線 ÷ 75日線 − 1、%）

比べ方（回す前に決めた）
  1. 母集団: データセットの全行（78週高値の更新日）。信用倍率が無い行はクラスタに入れず「信用倍率なし」の組として別に数える。
     ほかの4変数のどれかが無い行は外して件数だけ出す
  2. 変換: 5変数とも百分位（0〜1。同点は中央の順位）。裾の重い信用倍率・出来高比が距離を占めないように
  3. クラスタ: k-means（k = 8、初期値 20 通り、乱数 0）。k = 5・12 も同じ形で回し、見え方が k で変わらないかを見る
  4. 組ごと（全期間）: 件数・割合・正例率（Wilson の 95% 区間）・全体の正例率との比（リフト）・20営業日以内の終値の最高値で
     +10% 以上の割合（目的変数の到達ラインは20日ボラで決まるので、それによらない見方）・規則どおりの収益（+10% の指値・
     届かなければ20営業日目の終値。%）の平均・5変数の中央値（元の単位）と百分位の低・中・高
  5. 安定: 前半（〜2022-06）と後半（2022-07〜）の正例率と、年ごとにその年の全体の正例率を上回った年の数
  6. 前向き（当てる力）: 本番の out-of-fold と同じ窓（学習36か月から、テスト6か月ずつ前へ。学習はテスト窓の始まりの
     20営業日ぶん手前まで）で、窓ごとに学習期間だけで百分位・k-means・組ごとの正例率を作り、テスト窓の行をその組の学習期間の
     正例率で採点する。同じ行で、5変数だけの LightGBM（既定のパラメータ）と本番の 1.2σ の LightGBM（360列）の out-of-fold を
     並べて ROC-AUC・PR-AUC を窓ごとに出す。あわせて「学習期間で一番正例率が高かった組（学習100行以上）」がテスト窓でも
     高いか、同じ件数を本番モデル・5変数 LightGBM のスコアの上位から取ったときの正例率と比べる（一番低かった組も同じ）
  7. （1回目を回した後に足した見方）本番の 1.2σ の LightGBM の out-of-fold の上位5%（窓ごとの順位）が、全期間の組（k = 8）の
     どこにいるか。本番モデルがこの5変数の傾向をすでに取り込んでいるかを見る

公開ログには集計だけ（件数・割合・中央値・AUC）。銘柄・日付ごとの値は出さない。

    python3 research/exp/probe_cluster5.py [--data-dir research/_data] [--no-exits]

結果（2026-10-24、手元・70秒。データセット 2018-04-03〜2026-09-16 の 21,975行。docs/MODEL_ADOPTION_RULES.md §76）
------------------------------------------------------------------
  全体: 正例率 17.0%（前半 14.2% / 後半 18.6%）・+10% 到達 26.0%・規則どおりの収益 +0.46%（21,955行。欠測で外した 20行）
  組（k = 8・正例率の高い順。正例率 [95%区間] → リフト・+10% 到達・規則どおりの収益・年で上回った数。形は中央値）
    C1 2,476行 24.0% [22.3–25.7] → 1.41・18.0%・+1.08%・8/9  信用倍率 17倍・出来高 2.0倍・20日 +7%・ボラ 1.3%・25/75 +1.7%
    C2 2,395行 23.3% [21.6–25.0] → 1.37・22.2%・+0.77%・9/9  9倍・1.3倍・+6%・1.5%・+6.4%
    C3 3,092行 18.5% [17.1–19.9] → 1.08・11.3%・−0.41%・5/9  0.8倍・1.2倍・+6%・1.3%・+3.0%
    信用なし 3,876行 17.5% [16.3–18.7] → 1.02・36.0%・+0.81%・6/9  （信用倍率なし）・4.1倍・+16%・3.0%・+4.6%
    C4 2,325行 14.4% [13.0–15.9] → 0.84・41.0%・+0.45%・2/9  15倍・3.6倍・+23%・3.8%・+8.1%
    C5 2,051行 13.7% [12.3–15.3] → 0.80・18.2%・−0.33%・1/9  1.2倍・3.4倍・+8%・1.9%・+3.1%
    C6 1,907行 13.6% [12.1–15.2] → 0.80・28.4%・−0.04%・2/9  1.4倍・1.4倍・+16%・2.5%・+8.3%
    C7 1,775行 12.7% [11.3–14.4] → 0.75・23.2%・+0.49%・2/9  2.8倍・1.3倍・+15%・2.3%・+0.8%
    C8 2,058行 11.8% [10.5–13.3] → 0.69・34.0%・+1.19%・1/9  13倍・6.0倍・+20%・3.5%・+0.5%
  前向き（本番の OOF と同じ 11 窓・16,340行・正例率 17.9%）: ROC-AUC 組 0.565 ± 0.028・5変数 LightGBM 0.581 ± 0.041・
    本番 0.639 ± 0.041、PR-AUC 0.193・0.206・0.246。学習期間で一番高い組はテスト窓で 23.8%（全体 17.8%・11/11 窓で上回る）、
    同じ件数を本番の上位から取ると 28.4%・5変数 LightGBM は 22.9%。一番低い組は 13.7%（全体 18.1%・10/11 窓で下回る）、
    本番の下位同数 5.2%・5変数 LightGBM 9.4%
  本番の上位5%（窓ごと・812行・正例率 32.5%）: C1・C2 が 44.4%（OOF 全体では 26.5%）、C4〜C8 は 17.6%（全体では 42.6%）
  k = 5・12 でも同じ形: 一番高い組 23.8% / 26.0%（信用倍率 高・静か）、一番低い組 12.1% / 10.9%（出来高急増・急騰・ボラ高）、
    前向きの ROC-AUC 0.560 / 0.565
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import walkforward as WF  # noqa: E402
from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS  # noqa: E402

#: (列, 表示, 単位)
VARS = (
    ("credit_ratio", "信用倍率", "倍"),
    ("volume_trend", "出来高の20日平均比", "%"),
    ("ret_20d", "20日上昇率", "%"),
    ("vol_20d", "20日ボラ", "%"),
    ("tx_ma_ratio_25_75", "25日÷75日", "%"),
)
COLS = [c for c, _, _ in VARS]
CREDIT = "credit_ratio"
K_MAIN = 8
K_ALT = (5, 12)
N_INIT = 20
SEED = 0
SPLIT = pd.Timestamp("2022-07-01")      # 前半 / 後半
REACH = 0.10                            # future_rise（比）で +10%
TARGET = LT.TAKE_PROFIT                 # 規則の指値 +10%
HOLD = LT.HOLD                          # 20営業日
LEVELS = ((0.35, "低"), (0.65, "中"), (math.inf, "高"))   # 組の中央の百分位 → 低・中・高
NO_CREDIT = 0                           # 「信用倍率なし」の組（クラスタは 1〜k）
DROPPED = -1                            # ほかの変数が欠けて外した行
MIN_YEAR_ROWS = 20                      # 年ごとの比べに数える組の最小の件数
MIN_TRAIN_GROUP = 100                   # 前向きで「一番高い・低い組」に選べる学習の最小の件数
TOP_SHARE = 0.05                        # 本番モデルの上位（窓ごとの順位）がどの組にいるかを見る割合
OUT_DIR = os.path.join(lab.DATA_DIR, "oof")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 読み込み・百分位・クラスタ
# --------------------------------------------------------------------------- #

def load(data_dir: str) -> pd.DataFrame:
    cols = ["Code", "Date", "label", "future_rise"] + COLS
    df = pd.read_parquet(os.path.join(data_dir, "dataset.parquet"), columns=cols)
    df["Code"] = df["Code"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["label"].notna()].reset_index(drop=True)
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項の割合の 95% 区間（Wilson）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def pct_refs(rows: pd.DataFrame) -> dict:
    """百分位の物差し（列ごとの有限な値を並べたもの）。"""
    return {c: np.sort(pd.to_numeric(rows[c], errors="coerce").dropna().to_numpy(dtype=float)) for c in COLS}


def pct_apply(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ref の中での x の百分位（0〜1。同点は中央の順位）。NaN は NaN のまま。"""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if len(ref) == 0:
        return out
    lo = np.searchsorted(ref, x[ok], side="left")
    hi = np.searchsorted(ref, x[ok], side="right")
    out[ok] = (lo + hi) / 2.0 / len(ref)
    return out


def pct_matrix(rows: pd.DataFrame, refs: dict) -> np.ndarray:
    return np.column_stack([pct_apply(refs[c], pd.to_numeric(rows[c], errors="coerce").to_numpy(dtype=float))
                            for c in COLS])


def groups_of(rows: pd.DataFrame, X: np.ndarray, km) -> np.ndarray:
    """組の番号: 信用倍率なし = 0、ほかの変数が欠ける = -1、それ以外は k-means のクラスタ + 1。"""
    g = np.full(len(rows), DROPPED, dtype=int)
    credit = np.isfinite(pd.to_numeric(rows[CREDIT], errors="coerce").to_numpy(dtype=float))
    others = np.isfinite(X[:, 1:]).all(axis=1)
    g[~credit & others] = NO_CREDIT
    full = credit & others
    if full.any():
        g[full] = km.predict(X[full]) + 1
    return g


def fit_km(X: np.ndarray, k: int, seed: int = SEED):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def cluster_all(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """全期間で百分位とクラスタを作る（記述用）。戻り値: 組の番号・百分位の行列。"""
    refs = pct_refs(df)
    X = pct_matrix(df, refs)
    full = np.isfinite(X).all(axis=1)
    km = fit_km(X[full], k)
    return groups_of(df, X, km), X


def renumber_by_rate(df: pd.DataFrame, g: np.ndarray) -> np.ndarray:
    """クラスタの番号を正例率の高い順に 1〜k に付け直す（信用倍率なし 0・外した -1 はそのまま）。"""
    y = df["label"].to_numpy(dtype=float)
    ids = sorted(int(x) for x in np.unique(g) if x > 0)
    rate = {i: y[g == i].mean() for i in ids}
    order = sorted(ids, key=lambda i: (-rate[i], i))
    m = {old: new for new, old in enumerate(order, start=1)}
    return np.array([m.get(int(x), int(x)) for x in g], dtype=int)


# --------------------------------------------------------------------------- #
# 組ごとの表（全期間）
# --------------------------------------------------------------------------- #

def level(p: float) -> str:
    for hi, name in LEVELS:
        if p < hi:
            return name
    return LEVELS[-1][1]


def nan_median(x: np.ndarray) -> float:
    """有限な値の中央値（無ければ NaN。信用倍率なしの組の信用倍率など）。"""
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def describe(df: pd.DataFrame, g: np.ndarray, X: np.ndarray, rule_ret: np.ndarray | None = None) -> dict:
    """組ごとの件数・正例率・リフト・到達・収益・中央値・安定。"""
    y = df["label"].to_numpy(dtype=float)
    fr = pd.to_numeric(df["future_rise"], errors="coerce").to_numpy(dtype=float)
    d = df["Date"].to_numpy(dtype="datetime64[ns]")
    yr = df["Date"].dt.year.to_numpy()
    kept = g != DROPPED
    base = float(y[kept].mean())
    first = d < np.datetime64(SPLIT)
    year_rate = {int(a): float(y[kept & (yr == a)].mean()) for a in np.unique(yr[kept])}
    out = {"base": base, "rows": int(kept.sum()), "dropped": int((~kept).sum()),
           "base_first": float(y[kept & first].mean()), "base_second": float(y[kept & ~first].mean()),
           "reach_all": float(np.nanmean(fr[kept] >= REACH)),
           "ret_all": (float(np.nanmean(rule_ret[kept])) if rule_ret is not None else None),
           "groups": []}
    for gid in sorted(int(x) for x in np.unique(g) if x != DROPPED):
        m = g == gid
        n = int(m.sum())
        pos = int(y[m].sum())
        lo, hi = wilson(pos, n)
        row = {"id": gid, "n": n, "share": n / out["rows"], "pos": pos, "rate": pos / n, "lo": lo, "hi": hi,
               "lift": (pos / n) / base if base > 0 else float("nan"),
               "reach10": float(np.nanmean(fr[m] >= REACH)),
               "ret": (float(np.nanmean(rule_ret[m])) if rule_ret is not None and np.isfinite(rule_ret[m]).any()
                       else None),
               "median": {c: nan_median(pd.to_numeric(df.loc[m, c], errors="coerce").to_numpy(dtype=float))
                          for c in COLS},
               "level": {}}
        for j, c in enumerate(COLS):
            p = nan_median(X[m, j])
            row["level"][c] = level(p) if np.isfinite(p) else "-"
        for name, mm in (("first", m & first), ("second", m & ~first)):
            row[f"n_{name}"] = int(mm.sum())
            row[f"rate_{name}"] = float(y[mm].mean()) if mm.any() else float("nan")
        above = counted = 0
        for a, ra in year_rate.items():
            mm = m & (yr == a)
            if mm.sum() >= MIN_YEAR_ROWS:
                counted += 1
                above += int(y[mm].mean() > ra)
        row["years_above"], row["years_counted"] = above, counted
        out["groups"].append(row)
    # 確かめ: 件数と正例の数が全体と合う
    assert sum(r["n"] for r in out["groups"]) == out["rows"], "組の件数の合計が行数と合わない"
    assert sum(r["pos"] for r in out["groups"]) == int(y[kept].sum()), "組の正例の合計が全体と合わない"
    return out


# --------------------------------------------------------------------------- #
# 前向き
# --------------------------------------------------------------------------- #

def make_windows(dates: pd.Series) -> list:
    """本番の out-of-fold と同じ窓（train_production.oof_scores と同じ引数）。"""
    return WF.make_folds(pd.to_datetime(dates), min_train_months=OOF_MIN_TRAIN_MONTHS,
                         test_months=OOF_TEST_MONTHS, step_months=OOF_STEP_MONTHS,
                         embargo_days=B.RISE_HORIZON)


def top_rate(y: np.ndarray, s: np.ndarray, n: int) -> float:
    """スコアの上位 n 行の正例率（同点は行の順で切る。n = 0 なら NaN）。"""
    if n <= 0:
        return float("nan")
    idx = np.argsort(-s, kind="stable")[:n]
    return float(y[idx].mean())


def bottom_rate(y: np.ndarray, s: np.ndarray, n: int) -> float:
    if n <= 0:
        return float("nan")
    idx = np.argsort(s, kind="stable")[:n]
    return float(y[idx].mean())


def forward(df: pd.DataFrame, oof: pd.DataFrame | None, k: int, with_gbm: bool = True) -> dict:
    """窓ごとに学習期間だけで作った組の正例率で採点し、5変数 LightGBM・本番の OOF と同じ行で比べる。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = df["Date"]
    folds = make_windows(d)
    y_all = df["label"].to_numpy(dtype=float)
    base_score = None
    if oof is not None:
        o = oof[["Code", "Date", "score"]].copy()
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        base_score = df[["Code", "Date"]].merge(o.drop_duplicates(["Code", "Date"]), on=["Code", "Date"],
                                                how="left")["score"].to_numpy(dtype=float)
    wins = []
    for f in folds:
        tr = np.where((d <= pd.Timestamp(f.train_end)).to_numpy())[0]
        te = np.where(((d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))).to_numpy())[0]
        if len(te) < 200 or len(tr) < 1000:
            continue
        rtr, rte = df.iloc[tr], df.iloc[te]
        refs = pct_refs(rtr)
        Xtr, Xte = pct_matrix(rtr, refs), pct_matrix(rte, refs)
        full = np.isfinite(Xtr).all(axis=1)
        km = fit_km(Xtr[full], k)
        gtr, gte = groups_of(rtr, Xtr, km), groups_of(rte, Xte, km)
        ytr, yte = y_all[tr], y_all[te]
        rate_tr = {int(i): float(ytr[gtr == i].mean()) for i in np.unique(gtr)}
        n_tr = {int(i): int((gtr == i).sum()) for i in np.unique(gtr)}
        overall_tr = float(ytr.mean())
        s_cl = np.array([rate_tr.get(int(i), overall_tr) if i != DROPPED else overall_tr for i in gte])
        scores = {"cluster": s_cl}
        if with_gbm:
            import lightgbm as lgb
            gbm = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
            gbm.fit(rtr[COLS].to_numpy(dtype=float), ytr.astype(int))
            scores["gbm5"] = gbm.predict_proba(rte[COLS].to_numpy(dtype=float))[:, 1]
        if base_score is not None:
            scores["base"] = base_score[te]
        ok = np.ones(len(te), dtype=bool)
        for s in scores.values():
            ok &= np.isfinite(s)
        yy = yte[ok]
        rec = {"window": int(f.index), "test": f"{f.test_start}〜{f.test_end}", "n": int(ok.sum()),
               "n_test": int(len(te)), "pos_rate": float(yy.mean()) if ok.any() else float("nan"), "auc": {}, "pr": {}}
        if ok.sum() >= 50 and 0 < yy.sum() < len(yy):
            for name, s in scores.items():
                rec["auc"][name] = float(roc_auc_score(yy, s[ok]))
                rec["pr"][name] = float(average_precision_score(yy, s[ok]))
        # 学習期間で一番高い・低い組（クラスタだけ。学習 MIN_TRAIN_GROUP 行以上）
        cand = [i for i in rate_tr if i > 0 and n_tr[i] >= MIN_TRAIN_GROUP]
        if cand:
            best = max(cand, key=lambda i: (rate_tr[i], -i))
            worst = min(cand, key=lambda i: (rate_tr[i], i))
            gok = gte[ok]
            for tag, gid, pick in (("best", best, top_rate), ("worst", worst, bottom_rate)):
                mm = gok == gid
                nn = int(mm.sum())
                rec[tag] = {"train_rate": rate_tr[gid], "train_n": n_tr[gid], "n": nn,
                            "rate": float(yy[mm].mean()) if nn else float("nan"),
                            "same_n": {name: pick(yy, s[ok], nn) for name, s in scores.items() if name != "cluster"}}
        wins.append(rec)
    return {"k": k, "windows": wins, "summary": summarize_forward(wins)}


def summarize_forward(wins: list) -> dict:
    out = {"n_windows": len(wins), "rows": int(sum(w["n"] for w in wins))}
    names = sorted({n for w in wins for n in w["auc"]})
    for metric in ("auc", "pr"):
        out[metric] = {}
        for name in names:
            v = np.array([w[metric][name] for w in wins if name in w[metric]])
            out[metric][name] = {"mean": float(v.mean()) if len(v) else float("nan"),
                                 "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}
    out["pos_rate"] = float(np.average([w["pos_rate"] for w in wins], weights=[w["n"] for w in wins])) if wins else float("nan")
    for tag in ("best", "worst"):
        ws = [w for w in wins if tag in w and w[tag]["n"] > 0]
        n = sum(w[tag]["n"] for w in ws)
        pos = sum(w[tag]["rate"] * w[tag]["n"] for w in ws)
        rec = {"windows": len(ws), "n": int(n), "rate": (pos / n) if n else float("nan"),
               "base": (sum(w["pos_rate"] * w[tag]["n"] for w in ws) / n) if n else float("nan"),
               "beats_base": sum(1 for w in ws if (w[tag]["rate"] > w["pos_rate"]) == (tag == "best")),
               "same_n": {}}
        for name in sorted({nm for w in ws for nm in w[tag]["same_n"]}):
            vals = [(w[tag]["same_n"][name], w[tag]["n"]) for w in ws if np.isfinite(w[tag]["same_n"][name])]
            nn = sum(b for _, b in vals)
            rec["same_n"][name] = (sum(a * b for a, b in vals) / nn) if nn else float("nan")
        out[tag] = rec
    return out


def model_top(df: pd.DataFrame, g: np.ndarray, oof: pd.DataFrame, top: float = TOP_SHARE) -> dict:
    """
    本番の 1.2σ の LightGBM の out-of-fold の上位 top（窓ごとの順位）が、全期間の組のどこにいるか。
    組ごとに、out-of-fold の行に占める割合・上位に占める割合・上位の中の正例率。
    """
    o = oof[["Code", "Date", "score"]].copy()
    o["Code"] = o["Code"].astype(str)
    o["Date"] = pd.to_datetime(o["Date"])
    m = df[["Code", "Date", "label"]].assign(g=g).merge(o.drop_duplicates(["Code", "Date"]), on=["Code", "Date"],
                                                        how="inner")
    m = m[m["g"] != DROPPED].reset_index(drop=True)
    win = np.zeros(len(m), dtype=int)
    for f in make_windows(df["Date"]):
        mm = ((m["Date"] >= pd.Timestamp(f.test_start)) & (m["Date"] <= pd.Timestamp(f.test_end))).to_numpy()
        win[mm] = f.index
    m["win"] = win
    m = m[m["win"] > 0]
    m["top"] = m.groupby("win")["score"].rank(ascending=False, pct=True) <= top
    t = m[m["top"]]
    out = {"top": top, "rows": int(len(m)), "top_rows": int(len(t)), "top_rate": float(t["label"].mean()) if len(t) else float("nan"),
           "groups": []}
    for gid in sorted(int(x) for x in m["g"].unique()):
        a = m["g"] == gid
        b = t["g"] == gid
        out["groups"].append({"id": gid, "share_all": float(a.mean()), "share_top": float(b.mean()) if len(t) else float("nan"),
                              "n_top": int(b.sum()), "rate_top": float(t.loc[b, "label"].mean()) if b.any() else float("nan")})
    return out


# --------------------------------------------------------------------------- #
# 出口（規則どおりの収益）
# --------------------------------------------------------------------------- #

def rule_returns(df: pd.DataFrame, data_dir: str) -> np.ndarray:
    """+10% の指値・届かなければ20営業日目の終値の収益（%）。日足が足りない行は NaN。"""
    bars = LT.load_bars(data_dir, start=df["Date"].min() - pd.Timedelta(days=10))
    fw = LT.forward(bars, df[["Code", "Date"]], target=TARGET, hold=HOLD, fallback="close")
    r = df[["Code", "Date"]].merge(fw[["Code", "Date", "ret", "status"]], on=["Code", "Date"], how="left")
    done = r["status"].isin(["満了", "指値到達"]).to_numpy()
    return np.where(done, pd.to_numeric(r["ret"], errors="coerce").to_numpy(dtype=float), np.nan)


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def fmt_pct(x: float, nd: int = 1) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def fmt_num(x: float, nd: int = 2) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def profile(row: dict) -> str:
    return "・".join(f"{ja}{row['level'][c]}" for c, ja, _ in VARS if row["level"].get(c, "-") != "-")


def print_describe(desc: dict, k: int) -> None:
    log(f"\n■ 組ごと（全期間・k = {k}）: {desc['rows']:,}行（外した行 {desc['dropped']}）・全体の正例率 {fmt_pct(desc['base'])}"
        f"（前半 {fmt_pct(desc['base_first'])} / 後半 {fmt_pct(desc['base_second'])}）・+10% 到達 {fmt_pct(desc['reach_all'])}"
        + (f"・規則どおりの収益 {desc['ret_all']:+.2f}%" if desc.get("ret_all") is not None else ""))
    log("  組   件数(割合)       正例率 [95%区間]      リフト  +10%到達  収益     前半 / 後半      年(上回り/数)  形")
    for r in desc["groups"]:
        name = "信用なし" if r["id"] == NO_CREDIT else f"C{r['id']}"
        ret = f"{r['ret']:+.2f}%" if r.get("ret") is not None else "-"
        log(f"  {name:<6} {r['n']:>5,}({r['share'] * 100:4.1f}%)  {fmt_pct(r['rate'])} [{fmt_pct(r['lo'])}–{fmt_pct(r['hi'])}]"
            f"  {r['lift']:.2f}  {fmt_pct(r['reach10'])}  {ret:>7}  {fmt_pct(r['rate_first'])} / {fmt_pct(r['rate_second'])}"
            f"  {r['years_above']}/{r['years_counted']}  {profile(r)}")
    log("  中央値（元の単位）:")
    for r in desc["groups"]:
        name = "信用なし" if r["id"] == NO_CREDIT else f"C{r['id']}"
        log(f"  {name:<6} " + "  ".join(f"{ja} {fmt_num(r['median'][c], 1)}{unit}" for c, ja, unit in VARS))


def print_forward(fw: dict) -> None:
    s = fw["summary"]
    log(f"\n■ 前向き（k = {fw['k']}・本番の out-of-fold と同じ窓 {s['n_windows']} 本・{s['rows']:,}行・正例率 {fmt_pct(s['pos_rate'])}）")
    ja = {"cluster": "組の正例率", "gbm5": "5変数 LightGBM", "base": "本番 1.2σ LightGBM（360列）"}
    for name in s["auc"]:
        a, p = s["auc"][name], s["pr"][name]
        log(f"  {ja.get(name, name):<28} ROC-AUC {a['mean']:.3f} ± {a['sd']:.3f}  PR-AUC {p['mean']:.3f} ± {p['sd']:.3f}（{a['n']}窓）")
    log("  窓ごと（ROC-AUC 組 / 5変数 / 本番、正例率）:")
    for w in fw["windows"]:
        au = w["auc"]
        log(f"   窓{w['window']:>2} {w['test']}  {w['n']:>5,}行  "
            + " / ".join(fmt_num(au.get(n, float('nan')), 3) for n in ("cluster", "gbm5", "base"))
            + f"  正例率 {fmt_pct(w['pos_rate'])}")
    for tag, ja_tag in (("best", "学習期間で一番高かった組"), ("worst", "学習期間で一番低かった組")):
        b = s.get(tag)
        if not b or not b["windows"]:
            continue
        same = "・".join(f"{ja.get(n, n)}の{'上位' if tag == 'best' else '下位'}同数 {fmt_pct(v)}" for n, v in b["same_n"].items())
        log(f"  {ja_tag}: テスト窓で {b['n']:,}行・正例率 {fmt_pct(b['rate'])}（同じ窓の全体 {fmt_pct(b['base'])}・"
            f"{'上回った' if tag == 'best' else '下回った'}窓 {b['beats_base']}/{b['windows']}）。{same}")


def print_model_top(mt: dict) -> None:
    log(f"\n■ 本番 1.2σ LightGBM の上位 {mt['top'] * 100:.0f}%（窓ごと・{mt['top_rows']:,}行・正例率 {fmt_pct(mt['top_rate'])}）"
        f"がどの組にいるか（out-of-fold {mt['rows']:,}行に占める割合 → 上位に占める割合・上位の中の正例率）")
    for r in mt["groups"]:
        name = "信用なし" if r["id"] == NO_CREDIT else f"C{r['id']}"
        log(f"  {name:<6} {fmt_pct(r['share_all'])} → {fmt_pct(r['share_top'])}（{r['n_top']}行・{fmt_pct(r['rate_top'])}）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="5変数のクラスタごとの正例率（見るだけ）")
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--no-exits", action="store_true", help="規則どおりの収益（日足が要る）を出さない")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "probe_cluster5.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    df = load(args.data_dir)
    log(f"[probe] データセット {len(df):,}行・{df['Date'].min().date()}〜{df['Date'].max().date()}・正例率 "
        f"{fmt_pct(float(df['label'].mean()))}・信用倍率なし {fmt_pct(float(df[CREDIT].isna().mean()))}")
    rule_ret = None if args.no_exits else rule_returns(df, args.data_dir)
    if rule_ret is not None:
        log(f"[probe] 規則どおりの収益が出た行 {int(np.isfinite(rule_ret).sum()):,} / {len(df):,}")
    path = LT.find_oof("lgbm", args.data_dir, os.path.join(os.path.dirname(args.data_dir), "model"))
    oof = pd.read_parquet(path) if path else None
    log(f"[probe] 本番 1.2σ の LightGBM の out-of-fold: {len(oof):,}行" if oof is not None else "[probe] out-of-fold なし")
    result = {"vars": COLS, "k_main": K_MAIN, "describe": {}, "forward": {}}
    for k in (K_MAIN,) + K_ALT:
        g, X = cluster_all(df, k)
        g = renumber_by_rate(df, g)
        desc = describe(df, g, X, rule_ret)
        fw = forward(df, oof, k)
        result["describe"][k], result["forward"][k] = desc, fw
        if k == K_MAIN:
            print_describe(desc, k)
            print_forward(fw)
            if oof is not None:
                result["model_top"] = model_top(df, g, oof)
                print_model_top(result["model_top"])
        else:
            cl = [r for r in desc["groups"] if r["id"] > 0]
            hi, lo = cl[0], cl[-1]
            a = fw["summary"]["auc"].get("cluster", {})
            b = fw["summary"].get("best", {})
            log(f"\n■ k = {k}: 一番高い組 {fmt_pct(hi['rate'])}（{hi['n']:,}行・リフト {hi['lift']:.2f}・{profile(hi)}）、"
                f"一番低い組 {fmt_pct(lo['rate'])}（{lo['n']:,}行・{lo['lift']:.2f}・{profile(lo)}）。前向きの ROC-AUC "
                f"{fmt_num(a.get('mean', float('nan')), 3)}・学習期間で一番高かった組のテスト窓の正例率 {fmt_pct(b.get('rate', float('nan')))}"
                f"（全体 {fmt_pct(b.get('base', float('nan')))}）")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1, default=float)
    log(f"\n[probe] {args.out} に書いた（{time.time() - t0:.0f}秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
