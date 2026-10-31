#!/usr/bin/env python3
"""
直近5営業日の組のランキング学習（W5）を本番の脇に置く。**買いの規則は変えない。**

運用者の選択（2026-10-30）「画面に並べて確かめる」: 実験117 の W5（判断日ごとに、その日までの直近5営業日の候補を1つの組にして
並べる LambdaRank）を、日曜の週次で毎週探索・学習し、毎日の予測で「直近5営業日の中の順位」と3条件・4条件の印を予測タブと
スプレッドシートに出す。買う銘柄は今の規則のまま（1.5σ の木3本がすべて95以上）。これからの成績を記録して確かめる
（docs/MODEL_ADOPTION_RULES.md §82-6）。

何か（実験117 の W5 と同じ作り。tests/test_ltr5.py が実験117 の関数と同じ結果になることを確かめる）
  組      判断日 t ごとに、候補の日が t の4営業日前〜t のデータセットの行（78週高値の更新日）。同じ銘柄が組の中に2回以上入って
          いれば、最後の候補の日の行を残す。営業日は日足の取引日の暦で数える
  列      基準モデルと同じ列（features.DEFAULT_PRESET。候補の日の値）＋候補の日から判断日までの値動き4列
            r5_days  候補の日から判断日までの営業日数（0 = 判断日に入った候補。0〜4）
            r5_ret   候補の日の終値 → 判断日の終値の騰落（比。調整後終値）
            r5_high  候補の日の翌営業日〜判断日の高値の最大 ÷ 候補の日の終値 − 1（判断日に入った候補は欠測）
            r5_low   同じ期間の安値の最小 ÷ 候補の日の終値 − 1（同じ）
  段階    組の中の「判断日の翌営業日の寄りで買い、20営業日目の終値で売った収益」の5分位（0〜4。ltr.grades と同じ式で、組 =
          判断日）。当てはめは ltr.fit と同じ（LGBMRanker・組は Date = 判断日・種42）
  探索    実験117 の W5 と同じ: 探索の期間は run_tuning.py と同じ打ち切り（ltr.search_cutoff）、50試行・木200本・TPE・種0・
          探索空間は ltr.search_space。5分割は判断日を暦月の塊にまとめ、年で層別して塊ごとに分ける（同じ月の判断日は同じ側）。
          訓練側からは、検証側の判断日の前後4営業日の判断日を外す（同じ候補が両側の組に入らないように）。目的は検証分割の組の
          中の NDCG@5（候補5件以上の組）の分割平均

毎日の使い方（research/predict_daily.py の attach_ltr5）
  予測の日（asOf）の組を作り（その日までの直近5営業日の候補）、採点して組の中の順位（同じスコアは銘柄コードの順。実験117 と
  同じ）を付ける。予測の日の候補の行に ltr5（score・rank・nPool・pct）と、3条件・4条件の印（conds）を足す
    D1  その日の候補の中で、今のランキング学習（日次。research/ltr.py）が1位（同じスコアは銘柄コードの順）
    W1  直近5営業日の組の中で W5 が1位
    R0  1.2σ の木3本の百分位がすべて95以上（今日の戦略の規則。2026-10-30 から）
    R15 1.5σ の木3本の百分位がすべて95以上
    3条件 = D1 かつ W1 かつ R0、4条件 = 3条件 かつ R15（見るだけ）。4条件は実験117 の参考の表 §82-4 と同じ組み合わせ。3条件は
    2026-11-02 の予測から、運用者の選択「すぐ 1.2σ に替える」（2026-10-31。docs/MODEL_ADOPTION_RULES.md §87）で R15 → R0 に
    した（実験117 では測っていない組み合わせ）。§82-6 で決めた定義は 3条件 = D1・W1・R15、4条件 = ＋R0 で、4条件は同じ

置き場
  research/model/models/ltr5/{model.joblib, meta.json, oof.parquet}（ほかの追加モデルと同じ。Release には ltr5_model.joblib /
  ltr5_oof.parquet の名前で上がり、meta.json は git に残る）。models.LINEUP には入れない（分類ではないので、百分位の表・合意・
  一致度には数えない）
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_track as LT  # noqa: E402
import ltr  # noqa: E402
import tuning  # noqa: E402

NAME = "ltr5"
JA = "直近5営業日のランキング学習"
SHORT = "LTR5"
NOTE = ("LambdaRank。判断日ごとに、その日までの直近5営業日の候補を1つの組にして、判断日の翌営業日の寄りで買い20営業日目の"
        "終値で売った収益の順番を学ぶ（実験117 の W5）。組の中の順位だけを使い、買いの規則には使わない（参考。3条件・4条件）")
K = 5
MOVE_COLS = ("r5_days", "r5_ret", "r5_high", "r5_low")
#: 組の行で足す列（データセットに同じ名前の列があれば止める）
ADDED = MOVE_COLS + ("cand_date",)
SEED = ltr.SEED
R20 = ltr.R20
RET = "ret_tp"                   # 今の売り方の収益（+10% に届けば 0.10、届かなければ20営業日目の終値。比。実験114 と同じ名前）
HIT = "hit_tp"                   # 指値に届いたか（実験114 と同じ名前）
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltr5_params.json")
TUNE_TRIALS = ltr.TUNE_TRIALS
TUNE_SPLITS = ltr.TUNE_SPLITS
TUNE_SEED = ltr.TUNE_SEED
NDCG_K = ltr.NDCG_K
MIN_POOL = ltr.MIN_DAY           # NDCG に使う組の最小の行数
PURGE = K - 1                    # 探索の分割で、検証側の判断日の前後この営業日数の判断日を訓練から外す
TUNE_OBJECTIVE = (f"検証分割の組の中の NDCG@{NDCG_K}（候補{MIN_POOL}件以上の組。利得は組の中の20営業日目の終値の段階 "
                  f"0〜{ltr.N_GRADES - 1}）")
#: 3条件・4条件の線（今日の戦略の規則と同じ。画面の百分位 = 小数1桁に丸めた値で比べる）
LINE = 95.0
R15_TREES = ("lgbm_k15", "xgb_k15", "cat_k15")
R0_TREES = ("lgbm", "xgb", "cat")
#: 値動きの列を日足から直接計算し直して確かめる行の数（学習のとき）
N_CHECK = 300


# --------------------------------------------------------------------------- #
# 組
# --------------------------------------------------------------------------- #

def calendar(days) -> pd.DatetimeIndex:
    """取引日の暦（重複を除いて並べたもの。実験115 の calendar と同じ）。"""
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(days))).drop_duplicates().sort_values())


def pool_frame(cand: pd.DataFrame, bars: pd.DataFrame, cal, cols: Sequence[str], k: int = K,
               decision_days=None, keep: Sequence[str] = ()) -> pd.DataFrame:
    """
    候補の行（データセットの行。Code・Date と cols）から組の行を作る（実験117 の rolling_frame と同じ作り）。判断日 t ごとに、
    候補の日が t の (k−1) 営業日前〜t の行を集め、同じ (銘柄, 判断日) は最後の候補の日の行だけ残す。Date を判断日にし、元の
    候補の日を cand_date に残し、値動き4列を足す。(Date, Code) の順。

    decision_days を渡すと、その判断日の組だけを作る（毎日の予測では予測の日だけ）。keep の列（label など）も残す。
    bars は日足（Code・Date・AdjH・AdjL・AdjC）。cal は取引日の暦（calendar）。
    """
    clash = [c for c in ADDED if c in cand.columns]
    if clash:
        raise SystemExit(f"候補の行に組の行で足す列と同じ名前の列がある: {clash}")
    cal = calendar(cal)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    want = ["Code", "Date"] + [c for c in list(keep) + list(cols) if c not in ("Code", "Date")]
    d = cand[[c for c in dict.fromkeys(want) if c in cand.columns]].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    off = int((~d["Date"].isin(cal)).sum())
    if off:
        raise SystemExit(f"候補の行に取引日の暦に無い日付の行がある（{off}行）")
    d["_p"] = d["Date"].map(pos).to_numpy(dtype=np.int64)
    only = None
    if decision_days is not None:
        dd = pd.to_datetime(pd.Series(np.asarray(decision_days)))
        miss = int((~dd.isin(cal)).sum())
        if miss:
            raise SystemExit(f"判断日が取引日の暦に無い（{miss}日）")
        only = set(int(pos[pd.Timestamp(x)]) for x in dd)
    parts = []
    for a in range(k):
        x = d[d["_p"] + a < len(cal)].copy()
        x["_t"] = x["_p"] + a
        if only is not None:
            x = x[x["_t"].isin(only)]
        parts.append(x)
    r = pd.concat(parts, ignore_index=True)
    r = r.sort_values(["Code", "_t", "_p"], kind="mergesort").drop_duplicates(["Code", "_t"], keep="last")
    r["cand_date"] = r["Date"]
    r["Date"] = cal[r["_t"].to_numpy()]
    r["r5_days"] = (r["_t"] - r["_p"]).astype(np.int64)
    r = r.drop(columns=["_p", "_t"]).sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    if not len(r):
        for c in MOVE_COLS[1:]:
            r[c] = np.array([], dtype=float)
        return r
    p0 = r["cand_date"].map(pos).to_numpy(dtype=np.int64)

    # 値動き: 候補の日・判断日の終値と、その間（候補の日の翌取引日〜判断日）の高値の最大・安値の最小
    px = bars[bars["Code"].astype(str).isin(set(r["Code"]))][["Code", "Date", "AdjH", "AdjL", "AdjC"]].copy()
    px["Date"] = pd.to_datetime(px["Date"])
    px["Code"] = px["Code"].astype(str)
    rid = np.arange(len(r))
    codes = r["Code"].to_numpy()

    def close_at(dates) -> np.ndarray:
        q = pd.DataFrame({"Code": codes, "Date": pd.to_datetime(dates), "_rid": rid})
        m = q.merge(px[["Code", "Date", "AdjC"]], on=["Code", "Date"], how="left", validate="many_to_one")
        return m.sort_values("_rid")["AdjC"].to_numpy(dtype=float)

    c0 = close_at(r["cand_date"].to_numpy())
    c1 = close_at(r["Date"].to_numpy())
    span = r["r5_days"].to_numpy()
    seg = [pd.DataFrame({"_rid": rid[span >= j], "Code": codes[span >= j], "Date": cal[p0[span >= j] + j]})
           for j in range(1, int(span.max(initial=0)) + 1)]
    hmax = np.full(len(r), np.nan)
    lmin = np.full(len(r), np.nan)
    if seg:
        s = pd.concat(seg, ignore_index=True).merge(px[["Code", "Date", "AdjH", "AdjL"]], on=["Code", "Date"],
                                                    how="left", validate="many_to_one")
        agg = s.groupby("_rid").agg(h=("AdjH", "max"), l=("AdjL", "min"))
        hmax = agg["h"].reindex(rid).to_numpy(dtype=float)
        lmin = agg["l"].reindex(rid).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r["r5_ret"] = np.where(c0 > 0, c1 / c0 - 1.0, np.nan)
        r["r5_high"] = np.where(c0 > 0, hmax / c0 - 1.0, np.nan)
        r["r5_low"] = np.where(c0 > 0, lmin / c0 - 1.0, np.nan)

    if r.duplicated(["Code", "Date"]).any():
        raise SystemExit("組の行に同じ (銘柄, 判断日) の行が2つ以上ある")
    if ((r["r5_days"] < 0) | (r["r5_days"] > k - 1)).any() or (r["cand_date"] > r["Date"]).any():
        raise SystemExit(f"候補の日が判断日の0〜{k - 1}営業日前に入らない行がある")
    return r


def attach_exit(pool: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    組の行（Date = 判断日）に、判断日の翌営業日の寄りで買う収益を足す（実験115 の attach_week_exit と同じ式。live_track.forward）:
    R20 = 20営業日目の終値の収益（比）、RET = +10% に届けば 0.10・届かなければ R20、HIT = 届いたか。
    """
    keys = pool[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    keys["Date"] = pd.to_datetime(keys["Date"])
    fw = LT.forward(bars, keys, target=LT.TAKE_PROFIT, hold=LT.HOLD)
    for c in ("hit_day", "ret_close"):
        if c not in fw.columns:                  # すべて日足なしのとき
            fw[c] = np.nan
    fw = fw[["Code", "Date", "hit_day", "ret_close"]].copy()
    fw["Code"] = fw["Code"].astype(str)
    fw["Date"] = pd.to_datetime(fw["Date"])
    m = keys.merge(fw, on=["Code", "Date"], how="left", validate="one_to_one")
    hit = m["hit_day"].notna().to_numpy()
    c20 = pd.to_numeric(m["ret_close"], errors="coerce").to_numpy(dtype=float)
    out = pool.copy()
    out[R20] = c20 / 100.0
    out[RET] = np.where(hit, LT.TAKE_PROFIT, c20) / 100.0
    out[HIT] = hit
    return out


def check_move_cols(pool: pd.DataFrame, bars: pd.DataFrame, cal, n: int = N_CHECK, seed: int = 0) -> Dict:
    """
    値動き4列と r5_days を、無作為に選んだ n 行について銘柄の日足を直接切り出して計算し直し、同じかを数える（実験117 の
    check_move_cols と同じ。欠測どうしは同じ、値は差が 1e-12 以下なら同じ）。
    """
    cal = calendar(cal)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), size=min(n, len(pool)), replace=False)) if len(pool) else np.array([], int)
    sub = pool.iloc[idx][["Code", "Date", "cand_date"] + list(MOVE_COLS)]
    b = bars[bars["Code"].astype(str).isin(set(sub["Code"]))].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b["Code"] = b["Code"].astype(str)
    groups = {c: g.set_index("Date").sort_index() for c, g in b.groupby("Code", sort=False)}
    same = 0
    for r in sub.itertuples(index=False):
        g = groups.get(r.Code)
        c0 = float(g["AdjC"].get(r.cand_date, np.nan)) if g is not None else np.nan
        c1 = float(g["AdjC"].get(r.Date, np.nan)) if g is not None else np.nan
        win = g[(g.index > r.cand_date) & (g.index <= r.Date)] if g is not None else None
        want = {"r5_days": float(((cal > r.cand_date) & (cal <= r.Date)).sum()),
                "r5_ret": c1 / c0 - 1.0 if c0 > 0 else np.nan,
                "r5_high": float(win["AdjH"].max()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan,
                "r5_low": float(win["AdjL"].min()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan}
        ok = True
        for key, v in want.items():
            got = float(getattr(r, key))
            if np.isnan(v) and np.isnan(got):
                continue
            if np.isnan(v) or np.isnan(got) or abs(v - got) > 1e-12:
                ok = False
        same += int(ok)
    return {"rows": int(len(sub)), "same": int(same)}


def check_same_day(pool: pd.DataFrame, daily: pd.DataFrame) -> Dict:
    """
    判断日に入った候補（r5_days = 0）は、判断日 = 候補の日なので、組の R20 が日次の R20（ltr.attach_r20。候補の日の翌営業日の
    寄りで買い20営業日目の終値）と同じはず（欠測の有無も含めて）。合わない行を数える（実験117 の check_same_day と同じ考え方）。
    """
    z = pool[pool["r5_days"] == 0][["Code", "Date", R20]]
    d = daily[["Code", "Date", R20]].rename(columns={R20: "_d"}).copy()
    d["Code"] = d["Code"].astype(str)
    d["Date"] = pd.to_datetime(d["Date"])
    m = z.merge(d, on=["Code", "Date"], how="left", validate="one_to_one")
    x = pd.to_numeric(m[R20], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(m["_d"], errors="coerce").to_numpy(dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    bad = int((np.isfinite(x) != np.isfinite(y)).sum() + (np.abs(x[both] - y[both]) > 1e-12).sum())
    return {"rows": int(len(z)), "mismatch": bad}


def load_pool(dataset: str, data_dir: str, cols: Sequence[str]
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, Dict]:
    """
    学習・探索に使う組の行（R20・RET・label 付き。(Date, Code) の順）と、日次の候補（ltr.load_frame の形）・取引日の暦・
    確かめの数。
    train_ltr5.py と tune_ltr5.py が共通に使う。確かめ（値動きの列を日足から直接計算し直す・判断日に入った候補の R20 が日次と
    同じ）が合わなければ止める。
    """
    daily = ltr.load_frame(dataset, data_dir, cols)
    bars = LT.load_bars(data_dir, start=daily["Date"].min() - pd.Timedelta(days=10), extra=["AdjL"])
    cal = calendar(bars["Date"].unique())
    pool = pool_frame(daily, bars, cal, cols, keep=("label",))
    pool = attach_exit(pool, bars)
    cm = check_move_cols(pool, bars, cal)
    sd = check_same_day(pool, daily)
    size = pool.groupby("Date").size()
    info = {"daily_rows": int(len(daily)), "rows": int(len(pool)), "days": int(pool["Date"].nunique()),
            "per_day_median": float(size.median()) if len(size) else float("nan"),
            "age_share": [float((pool["r5_days"] == j).mean()) if len(pool) else float("nan") for j in range(K)],
            "with_r20": int(pool[R20].notna().sum()), "check_cols": cm, "check_same_day": sd}
    if cm["same"] != cm["rows"]:
        raise SystemExit(f"値動きの列が日足から直接計算した値と合わない（{cm['same']} / {cm['rows']} 行）")
    if sd["mismatch"]:
        raise SystemExit(f"判断日に入った候補で、組の R20 が日次の R20 と合わない（{sd['mismatch']}行）")
    return pool, daily, cal, info


def trainable(pool: pd.DataFrame) -> pd.Series:
    """学習・採点に使える組の行（今の売り方の収益と20営業日目の終値がある行。実験117 の rl_eval と同じ）。"""
    return pool[RET].notna() & pool[R20].notna()


# --------------------------------------------------------------------------- #
# 探索（実験117 の W5 と同じ）
# --------------------------------------------------------------------------- #

def search_rows(pool: pd.DataFrame, cutoff) -> Tuple[pd.DataFrame, int]:
    """
    探索に使う組の行（判断日が打ち切り以前・収益がある・ラベルがある行。(Date, Code) の順）と、ラベルが無くて外した行の数
    （実験117 の search_frame と同じ）。
    """
    sub = pool[(pd.to_datetime(pool["Date"]) <= pd.Timestamp(cutoff)) & pool[RET].notna() & pool[R20].notna()]
    no_label = int(sub["label"].isna().sum()) if "label" in sub.columns else 0
    if "label" in sub.columns:
        sub = sub[sub["label"].notna()]
    return sub.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True), no_label


def _year_groups(years: np.ndarray, n_splits: int) -> np.ndarray:
    """月の塊の年。月が n_splits 未満の年は隣の年とまとめる（実験117 の _year_groups と同じ）。"""
    uniq = sorted(set(int(y) for y in years))
    cnt = {y: int((years == y).sum()) for y in uniq}
    groups: List[List[int]] = []
    cur: List[int] = []
    n = 0
    for y in uniq:
        cur.append(y)
        n += cnt[y]
        if n >= n_splits:
            groups.append(cur)
            cur, n = [], 0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    gid = {y: i for i, g in enumerate(groups) for y in g}
    return np.array([gid[int(y)] for y in years])


def search_folds(sub: pd.DataFrame, cal, n_splits: int = TUNE_SPLITS, seed: int = TUNE_SEED,
                 purge: int = PURGE) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    探索の分割（実験117 の search_folds と同じ）。判断日を暦月の塊にまとめ、年で層別して塊ごとに n_splits 分割する（同じ月の
    判断日は同じ側）。訓練側からは、検証側の判断日の前後 purge 営業日以内の判断日を外す。
    """
    from sklearn.model_selection import StratifiedKFold

    cal = calendar(cal)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    d = pd.to_datetime(sub["Date"])
    month = d.dt.to_period("M")
    months = np.array(sorted(month.unique()))
    if len(months) < n_splits:
        return []
    years = np.array([m.year for m in months])
    strata = _year_groups(years, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = d.map(pos).to_numpy(dtype=np.int64)
    out = []
    for _, va_m in skf.split(months, strata):
        va = month.isin(months[va_m]).to_numpy()
        near = np.zeros(len(cal) + 2 * purge + 1, dtype=bool)
        vp = np.unique(p[va])
        for j in range(-purge, purge + 1):
            near[np.clip(vp + j, 0, len(cal) - 1)] = True
        tr = (~va) & (~near[p])
        out.append((sub[tr], sub[va]))
    return out


def tune(sub: pd.DataFrame, cols_w: Sequence[str], folds, n_trials: int = TUNE_TRIALS,
         seed: int = TUNE_SEED) -> Dict:
    """
    木の形を探索する（実験117 の W5 = 実験114 の tune_l と同じ: 探索空間 ltr.search_space、各試行で分割ごとに学習→検証の組の
    中の NDCG@5、その平均を最大にする。TPE の種・探索中の当てはめの種はどれも seed）。

    返すもの: params（ltr5_params.json の params にそのまま入る形）・_cv（目的・分割・試行数・最良の試行・分割ごとの値・
    探索に使った行と期間・分）・_trials。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cols_w = list(cols_w)
    t0 = time.time()
    if not folds:
        raise SystemExit("探索の分割を作れない（期間が短すぎる）")

    def objective(trial):
        p = ltr.search_space(trial)
        nd = []
        for tr, va in folds:
            m = ltr.fit(tr, cols_w, p, seed)
            s = ltr.predict(m, va[cols_w].to_numpy(dtype=float))
            nd.append(ltr.ndcg_at_k(va["Date"], s, ltr.grades(va["Date"], va[R20].to_numpy(dtype=float)),
                                    NDCG_K, MIN_POOL))
        trial.set_user_attr("ndcg_folds", [round(x, 4) for x in nd])
        return float(np.mean(nd))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {"boosting_type": "gbdt", "n_estimators": tuning.SEARCH_N_ESTIMATORS, "subsample_freq": 1,
            "n_jobs": -1, "verbose": -1, **study.best_params}
    d = pd.to_datetime(sub["Date"])
    return {"params": best,
            "_cv": {"objective": TUNE_OBJECTIVE, "scheme": f"month_purged{PURGE}", "n_splits": len(folds),
                    "n_trials": int(n_trials), "seed": int(seed), "best_trial": int(study.best_trial.number),
                    "ndcg": round(float(study.best_value), 4),
                    "ndcg_folds": study.best_trial.user_attrs.get("ndcg_folds"),
                    "n_rows": int(len(sub)), "n_days": int(d.nunique()),
                    "train_from": str(d.min().date()), "train_to": str(d.max().date()),
                    "minutes": round((time.time() - t0) / 60, 1)},
            "_trials": [{"n": int(t.number), "ndcg": round(float(t.value), 4)} for t in study.trials]}


# --------------------------------------------------------------------------- #
# 当てはめ・out-of-fold
# --------------------------------------------------------------------------- #

def load_params(path: str = PARAMS_PATH) -> Dict:
    """ltr5_params.json の中身（params と、探索した列の指紋 _features_sig・列数 _n_features・出どころ _source）。"""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def oof_scores(pool: pd.DataFrame, cols_w: Sequence[str], params: Dict, seed: int = SEED) -> pd.DataFrame:
    """
    組の各行を「その判断日より前のデータだけで学習したモデル」で採点する（実験117 の oof_shift のずらし0と同じ窓: 36ヶ月 /
    6ヶ月 / 6ヶ月、エンバーゴ = ラベル確定に要る営業日数。判断日で分ける）。学習・採点するのは trainable の行。
    """
    import build_dataset as B
    import walkforward as WF
    from train_production import OOF_MIN_TRAIN_MONTHS, OOF_STEP_MONTHS, OOF_TEST_MONTHS

    ok = trainable(pool)
    p = pool[ok].reset_index(drop=True)
    d = pd.to_datetime(p["Date"])
    folds = WF.make_folds(d, min_train_months=OOF_MIN_TRAIN_MONTHS, test_months=OOF_TEST_MONTHS,
                          step_months=OOF_STEP_MONTHS, embargo_days=B.RISE_HORIZON)
    parts = []
    for f in folds:
        tr = p[d <= pd.Timestamp(f.train_end)]
        te = p[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))]
        if len(te) < 200 or len(tr) < 1000:
            continue
        m = ltr.fit(tr, cols_w, params, seed)
        part = te[["Code", "Date", "cand_date", "r5_days", R20, RET, HIT]].copy()
        part["score"] = ltr.predict(m, te[list(cols_w)].to_numpy(dtype=float))
        part["fold"] = int(f.index)
        parts.append(part)
    if not parts:
        raise SystemExit("out-of-fold を作れません。期間が短すぎます")
    return pd.concat(parts, ignore_index=True)


def pool_ranks(dates, codes, scores) -> np.ndarray:
    """組の中の順位（1 = 最上位。同じスコアは銘柄コードの順。実験117 の triple_buys の w5_rank と同じ）。"""
    f = pd.DataFrame({"d": pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(),
                      "c": pd.Series(np.asarray(codes)).astype(str).to_numpy(),
                      "s": np.asarray(scores, dtype=float)})
    s = f.sort_values(["d", "s", "c"], ascending=[True, False, True], kind="mergesort")
    rank = np.zeros(len(f), dtype=np.int64)
    rank[s.index.to_numpy()] = (s.groupby("d", sort=False).cumcount() + 1).to_numpy()
    return rank


def load(model_dir: str):
    """保存済みのモデル（model, meta）。無ければ (None, None)（画面は出さないだけで、予測は止めない）。"""
    import models as M

    d = M.sub_dir(NAME, model_dir)
    mp, tp = os.path.join(d, "model.joblib"), os.path.join(d, "meta.json")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return None, None
    import joblib

    with open(tp, encoding="utf-8") as fh:
        meta = json.load(fh)
    return joblib.load(mp), meta


# --------------------------------------------------------------------------- #
# 3条件・4条件の印（毎日の予測）
# --------------------------------------------------------------------------- #

def all_at_least(per: Dict, algos: Sequence[str], line: float = LINE) -> Optional[bool]:
    """per（byModel）で algos の百分位がすべて line 以上か。どれかの値が無ければ None（判定しない）。"""
    vals = []
    for a in algos:
        v = (per.get(a) or {}).get("pctHistorical")
        if v is None or not np.isfinite(float(v)):
            return None
        vals.append(float(v))
    return bool(all(v >= line for v in vals))


def conds(d1: Optional[bool], w1: Optional[bool], r15: Optional[bool], r0: Optional[bool]) -> Dict:
    """
    3条件（D1・W1・R0。R0 は今日の戦略の規則 = 1.2σ の木3本）と4条件（＋R15）。どれかが決まらなければ、その組み合わせは None。
    2026-11-02 の予測から（運用者の選択 2026-10-31「すぐ 1.2σ に替える」。§87）。§82-6 で決めた定義は 3条件 = D1・W1・R15、
    4条件 = ＋R0（4条件は同じ）。
    """
    c3 = None if None in (d1, w1, r0) else bool(d1 and w1 and r0)
    c4 = None if c3 is None or r15 is None else bool(c3 and r15)
    return {"d1": d1, "w1": w1, "r15": r15, "r0": r0, "c3": c3, "c4": c4}
