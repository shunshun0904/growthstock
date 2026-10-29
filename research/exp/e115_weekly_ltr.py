#!/usr/bin/env python3
"""
実験115: ランキング学習を「週ごとの組」で学ぶ（週次のランキング学習。検証だけ。基準モデル・買いの規則・画面・本番のランキング
学習は変えない）。

運用者の依頼（2026-10-29）「追加で検証したいアイデアがあります。ランキング学習の粒度を週次単位での予測にしたいです。推論の候補は、
その週の月曜〜金曜日までで今のサンプルに入ってきたものが推論対象です。ただ、金曜日の夜に推論する場合、月曜のサンプルは一番情報が
リッチなので、金曜日から月曜まで遡るにつれ、保守的になりやすい=木曜日や金曜日に追加された銘柄のほうがランキングとして上位に
来やすくなる？かもしれませんが。」
AskUserQuestion（同じ日）: 候補の日から週末までの値動きを特徴量に「使う」/ 何と比べるか「今の運用はそのまま残しこれは、実験として
まずは始めてください。」/ 1週に買う件数「上位2件」（勧めた）。docs/MODEL_ADOPTION_RULES.md §79。

週の組（金曜の夜にその週の候補を並べ、上位2件を翌営業日の寄りで買う形）
  - 週: 月〜金（取引日の暦。祝日で金曜が休みなら木曜が週の最後の取引日 = 週末）。データの最後の週はまだ終わっていないかも
    しれないので外す（収益もまだ決まらない）
  - 候補: その週のどこかの日にデータセットに入った行（78週高値の更新日）。同じ銘柄が週に2回以上入ったら最後の日の行を残す
  - 特徴量: 残した行の 360列（features.DEFAULT_PRESET。候補の日の値）＋週末までの値動き4列
      wk_days  候補の日から週末までの営業日数（取引日の暦で数える。0 = 週末に入った候補）
      wk_ret   候補の日の終値 → 週末の終値の騰落（比。調整後終値）
      wk_high  候補の日の翌営業日〜週末の高値の最大 ÷ 候補の日の終値 − 1（週末に入った候補は欠測）
      wk_low   同じ期間の安値の最小 ÷ 候補の日の終値 − 1（同じ）
  - 収益: 週末の翌営業日の寄りで買い、+10% の指値、届かなければ20営業日目の終値（今の売り方。live_track.forward に選定日 =
    週末として渡すのと同じ）。学ぶ並びは週の中の「翌営業日の寄りで買い20営業日目の終値」の5分位（実験114 の L と同じ作りで、
    組が日 → 週）

腕（どれも同じ週の候補の行を採点し、週ごとにスコアの上から2件を買う）
  W   週次のランキング学習（LGBMRanker。組 = 週。360列＋値動き4列。種42）。木の形は実験114 の L と同じ探索（探索の期間は
      run_tuning.py と同じ打ち切り・年×時価総額帯で層別した週単位の5分割・種0・TPE・50試行・木200本・目的は検証分割の週の中の
      NDCG@5）
  WX  参考: W から値動き4列を外したもの（360列だけ。木の形は W と同じ）。週の組で学ぶこと自体と、値動きの列の効き目を分ける
  D   今のランキング学習（日次。research/ltr_params.json の木の形・種42。日ごとの組で学んだモデルで残した行を採点）
  B   基準モデル（1.2σ の LightGBM。lgbm_params.json の 360列の木の形。本番の out-of-fold と同じ当てはめ）
  窓: 基準モデルの out-of-fold と同じ作り（36か月 / 6か月 / 6か月、エンバーゴ20営業日。e41.folds_for）を週末の日付で作り、
  0 / 2 / 4か月ずらした3通り。W・WX は週末が窓の訓練の最終日以前の週の行で、D・B は候補の日が訓練の最終日以前の日次の行
  （D はラベルと20営業日目の終値がある行、B はラベルがある行。どちらも本番と同じ）で学ぶ。採点は週末がテスト窓に入る週の行

比べるもの（ずらし方ごと）
  1. 週ごとの上位2件: 件数 / 平均収益（全件・窓の平均）/ −10% 未満（全件・窓の平均）/ +10% 到達 / 20営業日目の終値の平均。
     W − B・W − D・W − WX の窓ごとの差。候補全体の平均（でたらめに2件買うのと同じ）
  2. 週の中の並び: NDCG@5（週の中の20営業日目の終値の段階）・週の中の順位相関（今の売り方の収益 / 20営業日目の終値）
  3. 曜日（運用者の見立て「木・金に入った候補が上に来やすいのでは」）: 候補の日の曜日ごとの候補の割合・週明けに買った平均収益・
     参考に候補の翌営業日に買った平均収益、腕ごとの週の中の百分位の平均、上位2件に占める割合
  4. 参考: 今の運用に近い毎日の買い（基準モデル1本の前向きの百分位95以上を、候補の翌営業日の寄りで買う。今の規則は1.2σ の3本
     なので近い形）の件数・平均収益・−10% 未満と、同じ窓での週ごとの上位2件

読み方（回す前に決めたもの。採用の判定ではない。入れるかは運用者が決める）
  - 3通りのずらしすべてで、W の上位2件の平均収益（窓の平均）が B と D の上位2件の両方より高く、かつ −10% 未満の割合（窓の平均）
    が両方より低い →「週ごとに学ぶ並べ方は、今のスコアで週の候補を並べるより良い」。それ以外 →「はっきりとは良くない」
  - 曜日: W の上位2件に占める木・金の候補の割合 − 候補全体に占める木・金の割合（ポイント。小数1桁に丸める）が、3通りすべてで
    +5ポイント以上 →「見立てどおり木・金に寄る」、すべてで −5ポイント以下 →「見立てと逆に木・金を避ける」、それ以外 →
    「はっきりとは寄らない」

確かめ（合わなければ止まる）
  - 週の候補は (銘柄, 週末) で1行・候補の日は週末以前で同じ週・データセットの日付はすべて取引日
  - 週末に入った候補（wk_days = 0）は、週の収益が日次の収益（実験114 の attach_exit）とすべて同じ（選定日が同じなので）
  - 値動き4列と wk_days を、無作為に選んだ行で銘柄の日足から直接計算し直した値と同じ
  - 参考に数えるもの: 週の収益がある行のうち、買った日が週末の翌取引日の割合（売買停止などで遅れる行がある）

公開ログには集計だけ（件数・平均・割合・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e115_*。

    exp=e115_weekly_ltr.py  args=""      （既定は --trials 50 --shifts 0,2,4 --top 2。探索の結果と out-of-fold は置き場に
                                            残し、同じデータ・同じ設定なら次の回は読む）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import ltr as LTR  # noqa: E402
import tuning  # noqa: E402
from train_model import EMBARGO_DAYS, HOLDOUT_MONTHS, holdout_bounds  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e53_objective_tuning as E53  # noqa: E402
import e114_ltr_retest as E  # noqa: E402

PREFIX = "e115"
OOF_DIR = E.OOF_DIR
PARAMS_PATH = os.path.join(OOF_DIR, f"{PREFIX}_params.json")
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
RET, HIT, HITDAY, R20 = E.RET, E.HIT, E.HITDAY, E.R20
#: 候補の翌営業日に買った場合の収益（日次。実験114 の attach_exit。確かめと参考）
D_RET, D_HIT, D_R20 = "d_ret", "d_hit", "d_r20"
WEEK_COLS = ("wk_days", "wk_ret", "wk_high", "wk_low")
#: 週の候補の行に足す列（データセットに同じ名前の列があれば止める）
ADDED = WEEK_COLS + (D_RET, D_HIT, D_R20, "cand_date", "wk_hits", "buy_date")
TOP_N = 2                         # 1週に買う件数（運用者の選択「上位2件」）
SEED = LTR.SEED                   # ランキング学習の種（本番と同じ 42）
LINE = 95.0                       # 参考の毎日の買いの線（前向きの百分位）
TF_LINE = 5.0                     # 曜日の読み方の線（木・金の割合の差、ポイント）
BAD = E.BAD
MIN_TEST, MIN_TRAIN = 200, 1000   # 窓を使う最小の件数（実験114 の oof_arm と同じ）
N_CHECK = 300                     # 値動きの列を直接計算し直す行の数
ARMS = ("W", "WX", "D", "B")
LABELS = {"W": "W 週次のランキング学習", "WX": "WX 週次（値動きの列なし）",
          "D": "D 今のランキング学習（日次）", "B": "B 基準モデル（1.2σ）"}
WEEKDAYS = ("月", "火", "水", "木", "金")
THU_FRI = (3, 4)
OBJECTIVE_JA = (f"検証分割の週の中の NDCG@{E.NDCG_K}（候補{E.MIN_DAY}件以上の週。利得は週の中の20営業日目の終値の段階 "
                f"0〜{E.N_GRADES - 1}）")

log = E.log


# --------------------------------------------------------------------------- #
# 週の組
# --------------------------------------------------------------------------- #

def calendar(days) -> pd.DatetimeIndex:
    """取引日の暦（重複を除いて並べたもの）。"""
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(days))).drop_duplicates().sort_values())


def week_ends(days) -> pd.Series:
    """
    取引日ごとの、その週（月〜日。取引は月〜金）の最後の取引日（週末）。index は取引日。最後の週は、データがその週の途中で
    終わっているかもしれないので外す（次の週の取引日がある週だけ）。
    """
    cal = calendar(days)
    if not len(cal):
        return pd.Series(dtype="datetime64[ns]")
    per = cal.to_period("W-SUN")
    end = pd.Series(cal, index=cal).groupby(np.asarray(per)).transform("max")
    done = np.asarray(per < per.max())
    return pd.Series(end.to_numpy()[done], index=cal[done])


def weekly_frame(full: pd.DataFrame, bars: pd.DataFrame, days) -> Tuple[pd.DataFrame, Dict]:
    """
    日次の候補（データセットの行。今の売り方の収益 RET・HIT・HITDAY・R20 が付いたもの）から週の候補を作る。Date を週末に置き換え、
    元の候補の日を cand_date に残す。同じ (銘柄, 週) は最後の候補の日の行だけ残し、その週に入った日数を wk_hits に。値動き4列を
    足す。日次の収益は d_ret・d_hit・d_r20 に名前を変えて残す（確かめと参考）。(Date, Code) の順。
    """
    clash = [c for c in ADDED if c in full.columns]
    if clash:
        raise SystemExit(f"データセットに週の候補で足す列と同じ名前の列がある: {clash}")
    cal = calendar(days)
    wmap = week_ends(cal)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    d = full.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    off = int((~d["Date"].isin(cal)).sum())
    if off:
        raise SystemExit(f"データセットに取引日の暦に無い日付の行がある（{off}行）")
    d["_wend"] = d["Date"].map(wmap)
    open_week = int(d["_wend"].isna().sum())
    d = d[d["_wend"].notna()].rename(columns={RET: D_RET, HIT: D_HIT, R20: D_R20})
    d = d.drop(columns=[HITDAY], errors="ignore").sort_values(["Code", "_wend", "Date"], kind="mergesort")
    d["wk_hits"] = d.groupby(["Code", "_wend"])["Date"].transform("size").astype(int)
    w = d.drop_duplicates(["Code", "_wend"], keep="last").copy()
    w["cand_date"] = w["Date"]
    w["Date"] = pd.to_datetime(w["_wend"])
    w = w.drop(columns=["_wend"]).sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    p0 = w["cand_date"].map(pos).to_numpy(dtype=np.int64)
    p1 = w["Date"].map(pos).to_numpy(dtype=np.int64)
    w["wk_days"] = (p1 - p0).astype(np.int64)

    # 値動き: 候補の日・週末の終値と、その間（候補の日の翌取引日〜週末）の高値の最大・安値の最小
    px = bars[bars["Code"].isin(set(w["Code"]))][["Code", "Date", "AdjH", "AdjL", "AdjC"]].copy()
    px["Date"] = pd.to_datetime(px["Date"])
    rid = np.arange(len(w))
    codes = w["Code"].to_numpy()

    def close_at(dates) -> np.ndarray:
        k = pd.DataFrame({"Code": codes, "Date": pd.to_datetime(dates), "_rid": rid})
        m = k.merge(px[["Code", "Date", "AdjC"]], on=["Code", "Date"], how="left", validate="many_to_one")
        return m.sort_values("_rid")["AdjC"].to_numpy(dtype=float)

    c0 = close_at(w["cand_date"].to_numpy())
    c1 = close_at(w["Date"].to_numpy())
    span = w["wk_days"].to_numpy()
    seg = [pd.DataFrame({"_rid": rid[span >= k], "Code": codes[span >= k], "Date": cal[p0[span >= k] + k]})
           for k in range(1, int(span.max(initial=0)) + 1)]
    hmax = np.full(len(w), np.nan)
    lmin = np.full(len(w), np.nan)
    if seg:
        s = pd.concat(seg, ignore_index=True).merge(px[["Code", "Date", "AdjH", "AdjL"]], on=["Code", "Date"],
                                                    how="left", validate="many_to_one")
        agg = s.groupby("_rid").agg(h=("AdjH", "max"), l=("AdjL", "min"))
        hmax = agg["h"].reindex(rid).to_numpy(dtype=float)
        lmin = agg["l"].reindex(rid).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w["wk_ret"] = np.where(c0 > 0, c1 / c0 - 1.0, np.nan)
        w["wk_high"] = np.where(c0 > 0, hmax / c0 - 1.0, np.nan)
        w["wk_low"] = np.where(c0 > 0, lmin / c0 - 1.0, np.nan)

    if w.duplicated(["Code", "Date"]).any():
        raise SystemExit("週の候補に同じ (銘柄, 週末) の行が2つ以上ある")
    if (w["wk_days"] < 0).any() or (w["cand_date"].dt.to_period("W-SUN") != w["Date"].dt.to_period("W-SUN")).any():
        raise SystemExit("週末が候補の日より前か、候補の日と同じ週にない行がある")
    size = w.groupby("Date").size()
    wd = w["cand_date"].dt.weekday
    info = {"daily_rows": int(len(full)), "open_week_rows": open_week, "rows": int(len(w)),
            "weeks": int(w["Date"].nunique()), "multi_share": float((w["wk_hits"] >= 2).mean()) if len(w) else np.nan,
            "dropped_repeats": int(len(full) - open_week - len(w)),
            "per_week_median": float(size.median()) if len(size) else np.nan,
            "weeks_ge5": float((size >= E.MIN_DAY).mean()) if len(size) else np.nan,
            "weekday_share": [float((wd == k).mean()) if len(w) else np.nan for k in range(5)],
            "wk_days_share": [float((w["wk_days"] == k).mean()) if len(w) else np.nan for k in range(5)],
            "first": str(w["Date"].min().date()) if len(w) else None,
            "last": str(w["Date"].max().date()) if len(w) else None}
    return w, info


def check_week_cols(w: pd.DataFrame, bars: pd.DataFrame, days, n: int = N_CHECK, seed: int = 0) -> Dict:
    """
    値動き4列と wk_days を、無作為に選んだ n 行について銘柄の日足を直接切り出して計算し直し、同じかを数える（欠測どうしは同じ、
    値は差が 1e-12 以下なら同じ）。
    """
    cal = calendar(days)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(w), size=min(n, len(w)), replace=False)) if len(w) else np.array([], dtype=int)
    sub = w.iloc[idx][["Code", "Date", "cand_date"] + list(WEEK_COLS)]
    b = bars[bars["Code"].isin(set(sub["Code"]))].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    groups = {c: g.set_index("Date").sort_index() for c, g in b.groupby("Code", sort=False)}
    same = 0
    worst = 0.0
    for r in sub.itertuples(index=False):
        g = groups.get(r.Code)
        c0 = float(g["AdjC"].get(r.cand_date, np.nan)) if g is not None else np.nan
        c1 = float(g["AdjC"].get(r.Date, np.nan)) if g is not None else np.nan
        win = g[(g.index > r.cand_date) & (g.index <= r.Date)] if g is not None else None
        want = {"wk_days": float(((cal > r.cand_date) & (cal <= r.Date)).sum()),
                "wk_ret": c1 / c0 - 1.0 if c0 > 0 else np.nan,
                "wk_high": float(win["AdjH"].max()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan,
                "wk_low": float(win["AdjL"].min()) / c0 - 1.0 if (win is not None and len(win) and c0 > 0) else np.nan}
        ok = True
        for k, v in want.items():
            got = float(getattr(r, k))
            if np.isnan(v) and np.isnan(got):
                continue
            if np.isnan(v) or np.isnan(got) or abs(v - got) > 1e-12:
                ok = False
                if np.isfinite(v) and np.isfinite(got):
                    worst = max(worst, abs(v - got))
                else:
                    worst = float("inf")
        same += int(ok)
    return {"rows": int(len(sub)), "same": int(same), "max_diff": float(worst)}


def attach_week_exit(w: pd.DataFrame, bars: pd.DataFrame, days) -> Tuple[pd.DataFrame, Dict]:
    """
    週の候補（Date = 週末）に、週末の翌営業日の寄りで買う今の売り方の収益を足す。計算は実験114 の attach_exit（実験83 の
    attach_exits → live_track.forward）と同じ式: RET = +10% に届けば 0.10、届かなければ20営業日目の終値の収益（比）、HIT・HITDAY・
    R20（20営業日目の終値の収益）。買った日 buy_date も残し、週末の翌取引日に買えた割合を数える。
    """
    keys = w[["Code", "Date"]].copy()
    keys["Code"] = keys["Code"].astype(str)
    keys["Date"] = pd.to_datetime(keys["Date"])
    fw = LT.forward(bars, keys, target=E.TARGET, hold=E.HOLD)
    for c in ("hit_day", "ret_close", "buy_date"):
        if c not in fw.columns:                  # すべて日足なしのとき
            fw[c] = np.nan
    fw = fw[["Code", "Date", "hit_day", "ret_close", "buy_date"]].copy()
    fw["Code"] = fw["Code"].astype(str)
    fw["Date"] = pd.to_datetime(fw["Date"])
    m = keys.merge(fw, on=["Code", "Date"], how="left", validate="one_to_one")
    hit = m["hit_day"].notna().to_numpy()
    c20 = pd.to_numeric(m["ret_close"], errors="coerce").to_numpy(dtype=float)
    out = w.copy()
    out[RET] = np.where(hit, E.TARGET, c20) / 100.0
    out[HIT] = hit
    out[HITDAY] = pd.to_numeric(m["hit_day"], errors="coerce").to_numpy(dtype=float)
    out[R20] = c20 / 100.0
    out["buy_date"] = pd.to_datetime(m["buy_date"]).to_numpy()
    cal = calendar(days)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    p = out["Date"].map(pos).to_numpy(dtype=np.int64)
    nxt = cal.to_numpy()[np.minimum(p + 1, len(cal) - 1)]
    nxt[p + 1 >= len(cal)] = np.datetime64("NaT")
    lab_ = out[R20].notna().to_numpy()
    on_time = lab_ & (out["buy_date"].to_numpy(dtype="datetime64[ns]") == nxt)
    info = {"rows": int(len(out)), "with_r20": int(lab_.sum()), "with_ret": int(out[RET].notna().sum()),
            "hit_share": float(np.mean(hit[lab_])) if lab_.any() else np.nan,
            "next_day_share": float(on_time.sum() / max(1, lab_.sum())), "late": int(lab_.sum() - on_time.sum())}
    return out, info


def check_same_day(w: pd.DataFrame) -> Dict:
    """週末に入った候補（wk_days = 0）は、週の収益と日次の収益が同じはず（欠測の有無も含めて）。合わない行を数える。"""
    z = w[w["wk_days"] == 0]

    def differ(a: str, b: str) -> int:
        x = pd.to_numeric(z[a], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(z[b], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(x) & np.isfinite(y)
        return int((np.isfinite(x) != np.isfinite(y)).sum() + (np.abs(x[both] - y[both]) > 1e-12).sum())

    out = {"rows": int(len(z)), "with_ret": int(z[RET].notna().sum()),
           "ret": differ(RET, D_RET), "r20": differ(R20, D_R20),
           "hit": int((z[HIT].astype(bool).to_numpy() != z[D_HIT].astype(bool).to_numpy()).sum())}
    out["mismatch"] = out["ret"] + out["r20"] + out["hit"]
    return out


# --------------------------------------------------------------------------- #
# 木の形
# --------------------------------------------------------------------------- #

def base_params(cols: Sequence[str], store: Dict | None = None) -> Dict:
    """基準モデル（1.2σ の LightGBM）の木の形。学習する列で探索したものでなければ止める（実験114b の prod_params と同じ確かめ）。"""
    store = tuning.load_params() if store is None else store
    rec = store.get(F.DEFAULT_PRESET) or {}
    why = tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), list(cols)) if rec else "探索の結果が無い"
    if why:
        raise SystemExit(f"基準モデルの木の形が {len(cols)}列で探索したものではない（{why}）")
    return tuning.params_for(F.DEFAULT_PRESET, store)


def daily_ltr_params(cols: Sequence[str], path: str | None = None) -> Dict:
    """今のランキング学習（日次）の木の形の記録（research/ltr_params.json）。学習する列で探索したものでなければ止める。"""
    rec = LTR.load_params(path or LTR.PARAMS_PATH)
    why = LTR.params_mismatch(rec, cols)
    if why:
        raise SystemExit(f"今のランキング学習の木の形が {len(cols)}列で探索したものではない（{why}）")
    return rec


def search_frame(wk: pd.DataFrame, cutoff) -> Tuple[pd.DataFrame, int]:
    """探索に使う週の候補（週末が打ち切り以前・収益がある・層別に使うラベルがある行）と、ラベルが無くて外した行の数。"""
    sub = wk[(pd.to_datetime(wk["Date"]) <= pd.Timestamp(cutoff)) & wk[RET].notna() & wk[R20].notna()]
    no_label = int(sub["label"].isna().sum())
    sub = sub[sub["label"].notna()].sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    return sub, no_label


def search_sig(cols_w: Sequence[str], sub: pd.DataFrame, n_trials: int) -> Dict:
    """探索の結果を使い回してよいかの目印（列・行・期間・試行数が同じなら同じ）。"""
    return {"features_sig": F.signature(list(cols_w)), "n_features": len(cols_w), "rows": int(len(sub)),
            "weeks": int(sub["Date"].nunique()), "train_to": str(pd.to_datetime(sub["Date"]).max().date()),
            "n_trials": int(n_trials), "target": R20, "group": "week", "v": 1}


def search_folds(sub: pd.DataFrame):
    """探索の5分割（本番の探索と同じ year_cap_date。Date = 週末なので、同じ週の候補は同じ側に入る）。"""
    return tuning.cv_folds(sub, tuning.PRODUCTION_CV, n_splits=E.N_SPLITS, seed=E.SEED, embargo_days=EMBARGO_DAYS)


# --------------------------------------------------------------------------- #
# out-of-fold
# --------------------------------------------------------------------------- #

def fit_rank(params: Dict, tr: pd.DataFrame, te: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """ランキング学習（実験114 の L と同じ当てはめ。組は tr の Date: 週の行なら週、日次の行なら日）。"""
    return E.fit_score("L", params, tr, te, cols, SEED)


def fit_base(params: Dict, tr: pd.DataFrame, tes: Sequence[pd.DataFrame], cols: Sequence[str]) -> List[np.ndarray]:
    """基準モデル（本番の out-of-fold と同じ: binary + 訓練の正例率から作る scale_pos_weight）。1回学習して tes をそれぞれ採点。"""
    import lightgbm as lgb

    cols = list(cols)
    y = tr["label"].to_numpy(dtype=int)
    m = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(y))
    m.fit(tr[cols].to_numpy(dtype=float), y)
    return [m.predict_proba(t[cols].to_numpy(dtype=float))[:, 1] if len(t) else np.array([]) for t in tes]


KEEP_W = ("Code", "Date", "cand_date", "wk_days", "wk_hits", RET, HIT, R20, D_RET, D_R20)
KEEP_D = ("Code", "Date", RET, HIT, R20)


def run_tag(params: Dict[str, Dict], cols: Sequence[str], cols_w: Sequence[str], wk: pd.DataFrame,
            daily: pd.DataFrame) -> str:
    """保存名に入れる目印（木の形・列・データが変わったら別の名前にして、古い out-of-fold を読まない）。"""
    key = {"params": {a: {k: p[k] for k in sorted(p)} for a, p in params.items()},
           "cols": F.signature(list(cols)), "cols_w": F.signature(list(cols_w)),
           "wk_rows": int(len(wk)), "wk_last": str(pd.to_datetime(wk["Date"]).max().date()),
           "wk_r20": int(wk[R20].notna().sum()), "d_rows": int(len(daily)),
           "d_last": str(pd.to_datetime(daily["Date"]).max().date()), "d_r20": int(daily[R20].notna().sum()),
           "seed": SEED, "v": 1}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_shift(wk: pd.DataFrame, daily: pd.DataFrame, cols: Sequence[str], cols_w: Sequence[str],
              params: Dict[str, Dict], shift: int, out_dir: str = OOF_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1つのずらし方の out-of-fold。週の候補（収益があるもの）を4本で採点した表（score_W・score_WX・score_D・score_B・fold）と、
    参考の毎日の買いに使う日次の行の基準モデルのスコアの表。保存済みなら読む。
    wk は週の候補（今の売り方の収益と20営業日目の終値がある行）、daily は日次の候補（ラベルがある行）。
    """
    tag = run_tag(params, cols, cols_w, wk, daily)
    pw = os.path.join(out_dir, f"{PREFIX}_week_{tag}_sh{shift}.parquet")
    pdl = os.path.join(out_dir, f"{PREFIX}_day_{tag}_sh{shift}.parquet")
    if os.path.exists(pw) and os.path.exists(pdl):
        return pd.read_parquet(pw), pd.read_parquet(pdl)
    t0 = time.time()
    folds = E41.folds_for(wk["Date"], shift)
    dw = pd.to_datetime(wk["Date"])
    dd = pd.to_datetime(daily["Date"])
    has_r20 = daily[R20].notna()
    d_eval = daily[RET].notna() & daily[R20].notna()
    rows_w, rows_d = [], []
    for f in folds:
        lo, hi, end = pd.Timestamp(f.test_start), pd.Timestamp(f.test_end), pd.Timestamp(f.train_end)
        te = wk[(dw >= lo) & (dw <= hi)]
        tr_w = wk[dw <= end]
        if len(te) < MIN_TEST or len(tr_w) < MIN_TRAIN:
            continue
        tr_l = daily[(dd <= end) & has_r20]
        tr_c = daily[dd <= end]
        te_d = daily[(dd >= lo) & (dd <= hi) & d_eval]
        part = te[list(KEEP_W)].copy()
        part["score_W"] = fit_rank(params["W"], tr_w, te, cols_w)
        part["score_WX"] = fit_rank(params["W"], tr_w, te, cols)
        part["score_D"] = fit_rank(params["D"], tr_l, te, cols)
        sb, sd = fit_base(params["B"], tr_c, [te, te_d], cols)
        part["score_B"] = sb
        part["fold"] = int(f.index)
        rows_w.append(part)
        dpart = te_d[list(KEEP_D)].copy()
        dpart["score_B"] = sd
        dpart["fold"] = int(f.index)
        rows_d.append(dpart)
    if not rows_w:
        raise SystemExit(f"ずらし{shift}か月: 窓を作れない（期間が短すぎる）")
    o = pd.concat(rows_w, ignore_index=True)
    od = pd.concat(rows_d, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    o.to_parquet(pw, index=False)
    od.to_parquet(pdl, index=False)
    log(f"  ずらし{shift}か月: {o['fold'].nunique()}窓・週の候補 {len(o):,}件・日次 {len(od):,}件 {time.time() - t0:.0f}秒")
    return o, od


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def top_per_week(o: pd.DataFrame, score: str, n: int = TOP_N) -> pd.DataFrame:
    """週ごとにスコアの上から n 件（候補が n 件に満たない週は全部）。同じスコアは銘柄コードの順。"""
    s = o.sort_values(["Date", score, "Code"], ascending=[True, False, True], kind="mergesort")
    return s.groupby("Date", sort=True).head(n)


def per_window(sel: pd.DataFrame) -> pd.DataFrame:
    """窓ごとの件数・平均収益・−10% 未満の割合・+10% 到達・20営業日目の終値の平均（選んだ行のある窓だけ）。"""
    cols = ["fold", "n", "ret", "bad", "hit", "r20"]
    if not len(sel):
        return pd.DataFrame(columns=cols)
    g = sel.groupby("fold", sort=True)
    out = pd.DataFrame({"n": g.size(), "ret": g[RET].mean(),
                        "bad": g[RET].apply(lambda x: float(np.mean(x.to_numpy(dtype=float) < BAD))),
                        "hit": g[HIT].apply(lambda x: float(np.mean(x.astype(float)))),
                        "r20": g[R20].mean()}).reset_index()
    return out[cols]


def pick_stats(sel: pd.DataFrame) -> Dict:
    """選んだ行の件数・組（週 / 日）の数・平均収益（全件 / 窓の平均）・−10% 未満（全件 / 窓の平均）・+10% 到達・20営業日目の終値。"""
    pw = per_window(sel)
    r = sel[RET].to_numpy(dtype=float)
    return {"n": int(len(sel)), "groups": int(sel["Date"].nunique()) if len(sel) else 0, "windows": int(len(pw)),
            "ret": float(np.mean(r)) if len(r) else np.nan,
            "ret_w": float(pw["ret"].mean()) if len(pw) else np.nan,
            "bad": float(np.mean(r < BAD)) if len(r) else np.nan,
            "bad_w": float(pw["bad"].mean()) if len(pw) else np.nan,
            "hit": float(np.mean(sel[HIT].astype(float))) if len(sel) else np.nan,
            "r20": float(sel[R20].mean()) if len(sel) else np.nan}


def ranking_metrics(o: pd.DataFrame, score: str) -> Dict:
    """週の中の NDCG@5（週の中の20営業日目の終値の段階）と、週の中の順位相関（今の売り方の収益 / 20営業日目の終値）。"""
    s = o[score].to_numpy(dtype=float)
    r20 = o[R20].to_numpy(dtype=float)
    return {"ndcg": E.ndcg_at_k(o["Date"], s, E.grades(o["Date"], r20)),
            "rho": E53.date_rho(o["Date"], s, o[RET].to_numpy(dtype=float), min_n=E.MIN_DAY),
            "rho20": E53.date_rho(o["Date"], s, r20, min_n=E.MIN_DAY)}


def weekday_stats(o: pd.DataFrame, picks: Dict[str, pd.DataFrame]) -> Dict:
    """
    候補の日の曜日（月〜金）ごとに: 候補の件数・割合、週明けに買った平均収益、参考に候補の翌営業日に買った平均収益、腕ごとの
    週の中の百分位（0 = 最下位 / 1 = 最上位）の平均、上位 n 件に占める割合。木・金の割合（候補全体 / 腕ごとの上位 n 件）も。
    """
    wd = pd.to_datetime(o["cand_date"]).dt.weekday.to_numpy()
    n = len(o)
    out: Dict = {"n": [int((wd == k).sum()) for k in range(5)],
                 "share": [float((wd == k).mean()) if n else np.nan for k in range(5)],
                 "ret": [float(o.loc[wd == k, RET].mean()) if (wd == k).any() else np.nan for k in range(5)],
                 "d_ret": [float(o.loc[wd == k, D_RET].mean()) if (wd == k).any() else np.nan for k in range(5)],
                 "pct": {}, "pick_share": {}, "tf_pick": {}}
    out["tf_cand"] = float(np.isin(wd, THU_FRI).mean()) if n else np.nan
    for a, p in picks.items():
        pct = LTR.within_date_pct(o["Date"], o[f"score_{a}"].to_numpy(dtype=float))
        out["pct"][a] = [float(pct[wd == k].mean()) if (wd == k).any() else np.nan for k in range(5)]
        pwd = pd.to_datetime(p["cand_date"]).dt.weekday.to_numpy()
        out["pick_share"][a] = [float((pwd == k).mean()) if len(p) else np.nan for k in range(5)]
        out["tf_pick"][a] = float(np.isin(pwd, THU_FRI).mean()) if len(p) else np.nan
    return out


def daily_reference(od: pd.DataFrame, line: float = LINE) -> Tuple[Dict, List[int]]:
    """参考の毎日の買い: 窓ごとに、前の窓までの基準モデルのスコアの分布の line 分位を超えた行（e114.threshold_picks）。"""
    s = E.threshold_picks(od.rename(columns={"score_B": "score"}), line)
    st = pick_stats(s)
    return st, sorted(int(x) for x in s["fold"].unique())


def evaluate(o: pd.DataFrame, od: pd.DataFrame, top_n: int = TOP_N) -> Dict:
    """1つのずらし方の集計（腕ごとの上位 n 件・候補全体・窓ごとの差・週の中の並び・曜日・参考の毎日の買い）。"""
    picks = {a: top_per_week(o, f"score_{a}", top_n) for a in ARMS}
    res: Dict = {"windows": int(o["fold"].nunique()), "rows": int(len(o)), "weeks": int(o["Date"].nunique()),
                 "all": pick_stats(o), "top": {a: pick_stats(p) for a, p in picks.items()},
                 "rank": {a: ranking_metrics(o, f"score_{a}") for a in ARMS}, "diff": {}, "diff_bad": {}}
    base = {a: per_window(p) for a, p in picks.items()}
    for b in ("B", "D", "WX"):
        res["diff"][b] = E.fold_diff(base[b], base["W"])
        res["diff_bad"][b] = E.fold_diff(base[b], base["W"], "bad")
    res["weekday"] = weekday_stats(o, picks)
    ref, wins = daily_reference(od)
    res["daily_ref"] = ref
    res["daily_ref_windows"] = wins
    res["top_ref_windows"] = {a: pick_stats(p[p["fold"].isin(wins)]) for a, p in picks.items()}
    return res


def reading(rows: List[Dict]) -> Dict:
    """
    読み方（回す前に決めたもの。採用の判定ではない）。rows はずらし方ごとの evaluate の結果（shift を足したもの）。
      better: すべてのずらし方で、W の上位 n 件の平均収益（窓の平均）が B・D の両方より高く、−10% 未満（窓の平均）が両方より低い
      lean: W の上位 n 件の木・金の割合 − 候補全体の木・金の割合（ポイント。小数1桁に丸める）が、すべてで +TF_LINE 以上なら
            "thu_fri"、すべてで −TF_LINE 以下なら "avoid"、それ以外は "none"
    """
    each = []
    for r in rows:
        w, b, d = r["top"]["W"], r["top"]["B"], r["top"]["D"]
        ok = bool(w["ret_w"] > b["ret_w"] and w["ret_w"] > d["ret_w"] and w["bad_w"] < b["bad_w"] and w["bad_w"] < d["bad_w"])
        # 表に出す桁（小数1桁）に丸めてから線と比べる（表の値と読み方を食い違わせない）
        tf = round(float((r["weekday"]["tf_pick"]["W"] - r["weekday"]["tf_cand"]) * 100.0), 1)
        each.append({"shift": int(r["shift"]), "better": ok, "tf_gap": tf})
    gaps = [x["tf_gap"] for x in each]
    if gaps and all(g >= TF_LINE for g in gaps):
        lean = "thu_fri"
    elif gaps and all(g <= -TF_LINE for g in gaps):
        lean = "avoid"
    else:
        lean = "none"
    return {"rows": each, "better": bool(each) and all(x["better"] for x in each), "lean": lean}


LEAN_JA = {"thu_fri": "見立てどおり木・金に寄る", "avoid": "見立てと逆に木・金を避ける", "none": "はっきりとは寄らない"}


# --------------------------------------------------------------------------- #

def load_all(cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict]:
    """
    データセット（実収益付き）に日次の今の売り方の収益（実験114 の attach_exit）を足したもの・日足（安値も）・探索の打ち切り日
    （run_tuning.py と同じ。データセットの全日付から）・出口の計算の確かめ。
    """
    full = lab.frame()
    miss = [c for c in cols if c not in full.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:6]}。research/build_dataset.py を回し直してください")
    full = full.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full["Code"] = full["Code"].astype(str)
    cutoff, _, _ = holdout_bounds(full["Date"], HOLDOUT_MONTHS, EMBARGO_DAYS)
    bars = LT.load_bars(lab.DATA_DIR, start=full["Date"].min() - pd.Timedelta(days=10), extra=["AdjL"])
    full, chk = E.attach_exit(full, bars)
    return full, bars, pd.Timestamp(cutoff), chk


def _p(x: float, nd: int = 2) -> str:
    return f"{x * 100:+.{nd}f}%" if np.isfinite(x) else "—"


def _s(x: float) -> str:
    return f"{x * 100:.1f}%" if np.isfinite(x) else "—"


def _d(dd: Dict) -> str:
    if not dd.get("n"):
        return "—"
    return f"{dd['mean'] * 100:+.2f} ± {dd['se'] * 100:.2f}pt（上の窓 {dd['won']}/{dd['n']}）"


def report(sh: int, res: Dict, top_n: int) -> None:
    """1つのずらし方の表（公開ログ。集計だけ）。"""
    a0 = res["all"]
    print(f"\n■ ずらし{sh}か月（{res['windows']}窓・{res['weeks']}週・週の候補 {res['rows']:,}件・候補全体の平均 {_p(a0['ret'])}"
          f"（窓の平均 {_p(a0['ret_w'])}）・−10% 未満 {_s(a0['bad'])}）")
    print(f"  ◆ 週ごとの上位{top_n}件（週明けの寄りで買う）: 件数 / 平均収益（全件・窓の平均）/ −10% 未満（全件・窓の平均）/ "
          f"+10%到達 / 20営業日目の終値 ｜ 週の中の NDCG@{E.NDCG_K} / 順位相関（今の売り方・20営業日目の終値）")
    for a in ARMS:
        t, k = res["top"][a], res["rank"][a]
        print(f"    {LABELS[a]:<24}{t['n']:>5}件 {_p(t['ret']):>8} {_p(t['ret_w']):>8} {_s(t['bad']):>6} {_s(t['bad_w']):>6} "
              f"{_s(t['hit']):>6} {_p(t['r20']):>8} ｜ {k['ndcg']:.4f} / {k['rho']:+.3f} / {k['rho20']:+.3f}")
    print(f"    {'候補全体（でたらめに選ぶのと同じ）':<22}{a0['n']:>5}件 {_p(a0['ret']):>8} {_p(a0['ret_w']):>8} "
          f"{_s(a0['bad']):>6} {_s(a0['bad_w']):>6} {_s(a0['hit']):>6} {_p(a0['r20']):>8}")
    print("  ◆ 窓ごとの差（W − 比べる腕。平均収益 / −10% 未満）: " + " / ".join(
        f"W − {b} {_d(res['diff'][b])}・−10%未満 {res['diff_bad'][b]['mean'] * 100:+.1f}pt" if res['diff'][b].get('n')
        else f"W − {b} —" for b in ("B", "D", "WX")))
    wd = res["weekday"]
    print("  ◆ 候補の日の曜日: 候補の割合 / 週明けに買った平均収益 / 参考 候補の翌営業日に買った平均収益")
    print("    " + " | ".join(f"{WEEKDAYS[k]} {_s(wd['share'][k])} {_p(wd['ret'][k])} {_p(wd['d_ret'][k])}" for k in range(5)))
    print("  ◆ 腕ごとの週の中の百分位の平均（曜日別。0.5 が真ん中）と、上位の件数に占める曜日の割合 ｜ 木・金の割合（上位 / 候補全体）")
    for a in ARMS:
        print(f"    {a:<3}" + " ".join(f"{WEEKDAYS[k]}{wd['pct'][a][k]:.2f}" for k in range(5)) + " ｜ "
              + " ".join(f"{WEEKDAYS[k]}{_s(wd['pick_share'][a][k])}" for k in range(5))
              + f" ｜ {_s(wd['tf_pick'][a])} / {_s(wd['tf_cand'])}")
    ref, rw = res["daily_ref"], res["daily_ref_windows"]
    print(f"  ◆ 参考: 今の運用に近い毎日の買い（基準モデル1本の前向きの百分位{LINE:.0f}以上・候補の翌営業日の寄り。{len(rw)}窓）: "
          f"{ref['n']}件 {_p(ref['ret'])}（窓の平均 {_p(ref['ret_w'])}）・−10% 未満 {_s(ref['bad'])}（{_s(ref['bad_w'])}）・"
          f"+10%到達 {_s(ref['hit'])} ｜ 同じ窓の週ごとの上位{top_n}件: " + " / ".join(
              f"{a} {res['top_ref_windows'][a]['n']}件 {_p(res['top_ref_windows'][a]['ret_w'])}" for a in ("W", "B", "D")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験115: 週次のランキング学習（検証だけ）")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)

    t_all = time.time()
    cols = list(F.columns(F.DEFAULT_PRESET))
    cols_w = cols + list(WEEK_COLS)
    p_base = base_params(cols)
    rec_d = daily_ltr_params(cols)
    os.makedirs(OOF_DIR, exist_ok=True)

    full, bars, cutoff, chk = load_all(cols)
    days = calendar(bars["Date"].unique())
    wk, info = weekly_frame(full, bars, days)
    wk, xinfo = attach_week_exit(wk, bars, days)
    cw = check_week_cols(wk, bars, days)
    sd = check_same_day(wk)
    del bars
    daily = full[full["label"].notna()].reset_index(drop=True)
    del full

    print("=" * 100)
    print(f"実験115 週次のランキング学習（検証だけ）: 日次の候補 {info['daily_rows']:,}件 → 週の候補 {info['rows']:,}件・"
          f"{info['weeks']:,}週（{info['first']} 〜 {info['last']}）/ {F.DEFAULT_PRESET} {len(cols)}列＋値動き{len(WEEK_COLS)}列 / "
          f"ずらし {shifts}か月 / 週ごとの上位{args.top}件")
    print(f"  収益: 週末の翌営業日の寄りで買い、+{E.TARGET:g}% の指値、届かなければ{E.HOLD}営業日目の終値（今の売り方）/ "
          f"W が学ぶ並び: 週の中の{E.HOLD}営業日目の終値の{E.N_GRADES}分位")
    print("=" * 100)
    log(f"[確かめ] 日次の出口の計算: 同じ計算の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,} 行")
    if chk["hold5_both"] and chk["hold5_match"] / chk["hold5_both"] < 0.99:
        raise SystemExit("出口の計算が ret_o1_20 と合わない（日足と学習データの株価がずれている）")
    log(f"[確かめ] 週の組: 最後の週（終わっていないかもしれない）で外した日次の候補 {info['open_week_rows']}件・同じ週に2回以上"
        f"入った銘柄の重なりで外した行 {info['dropped_repeats']:,}件（週の候補の {_s(info['multi_share'])} が2日以上入った銘柄）・"
        f"1週の候補 中央 {info['per_week_median']:.0f}件・{E.MIN_DAY}件以上の週 {_s(info['weeks_ge5'])}")
    log("[確かめ] 候補の日の曜日: " + " / ".join(f"{WEEKDAYS[k]} {_s(info['weekday_share'][k])}" for k in range(5))
        + " ｜ 週末までの営業日数: " + " / ".join(f"{k}日 {_s(info['wk_days_share'][k])}" for k in range(5)))
    log(f"[確かめ] 週の収益: 20営業日目の終値がある {xinfo['with_r20']:,} / {xinfo['rows']:,}件・+{E.TARGET:g}% 到達 "
        f"{_s(xinfo['hit_share'])}・週末の翌取引日に買えた {_s(xinfo['next_day_share'])}（遅れた {xinfo['late']}件）")
    log(f"[確かめ] 週末に入った候補（wk_days = 0）の週の収益と日次の収益: {sd['rows']:,}件中 合わない行 今の売り方 {sd['ret']}・"
        f"20営業日目の終値 {sd['r20']}・到達の印 {sd['hit']}")
    log(f"[確かめ] 値動き4列と wk_days を日足から直接計算し直す: {cw['same']} / {cw['rows']} 行が同じ（差の最大 {cw['max_diff']:.1e}）")
    if sd["mismatch"]:
        raise SystemExit("週末に入った候補で、週の収益が日次の収益と合わない（週の収益の付け方がずれている）")
    if cw["same"] != cw["rows"]:
        raise SystemExit("値動きの列が日足から直接計算した値と合わない")

    # ---- 探索 ----
    sub, no_label = search_frame(wk, cutoff)
    sig = search_sig(cols_w, sub, args.trials)
    log(f"[探索] 期間 〜{cutoff.date()}（run_tuning.py と同じ打ち切り）: 週の候補 {len(sub):,}件・{sub['Date'].nunique()}週"
        f"（層別に使うラベルが無くて外した {no_label}件）")
    store = None
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("_sig") == sig:
            store = prev
            log("[探索] 保存済みの探索を読む（同じデータ・同じ列・同じ試行数）")
    folds_s = search_folds(sub)
    if not folds_s:
        raise SystemExit("探索の分割を作れない")
    if store is None:
        log(f"[探索] W（LambdaRank・組 = 週・目的 {OBJECTIVE_JA}・{args.trials}試行）")
        rec_w = E.tune_l(sub, cols_w, args.trials, folds_s)
        rec_w["_cv"]["objective"] = OBJECTIVE_JA
        store = {"_sig": sig, "_cutoff": str(cutoff.date()), "W": rec_w}
        with open(PARAMS_PATH, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2, default=str)
        log(f"[探索] W {rec_w['_minutes']}分 / 分割平均 NDCG@{E.NDCG_K} {rec_w['_cv']['ndcg']}（最良 {rec_w['_cv']['best_trial']}"
            f"試行目）")
    params = {"W": store["W"]["params"], "D": rec_d["params"], "B": p_base}
    print("\n■ 木の形")
    print(f"  W  週次（探索 〜{store['_cutoff']}）: {E._tree_line(params['W'])}")
    print(f"  D  今のランキング学習（research/ltr_params.json・探索 〜{rec_d.get('_cutoff', '?')}）: {E._tree_line(params['D'])}")
    print(f"  B  基準モデル（lgbm_params.json）: {E._tree_line(params['B'])}")
    rnd = E.random_ndcg(folds_s)
    cvm = {"W": E.cv_metrics("L", params["W"], folds_s, cols_w), "WX": E.cv_metrics("L", params["W"], folds_s, cols)}
    print(f"  探索の5分割で（分割の平均）: 週の中の NDCG@{E.NDCG_K} W {cvm['W']['ndcg']:.4f} / WX（値動きの列なし）"
          f"{cvm['WX']['ndcg']:.4f} / でたらめな順 {rnd:.4f} ｜ 週の中の順位相関（20営業日目の終値）W {cvm['W']['rho']:+.3f} / "
          f"WX {cvm['WX']['rho']:+.3f}")

    # ---- out-of-fold と集計 ----
    wk_eval = wk[wk[RET].notna() & wk[R20].notna()].reset_index(drop=True)
    summary: Dict = {"info": info, "exit": xinfo, "check_cols": cw, "check_same_day": sd, "sig": sig,
                     "cutoff": str(cutoff.date()), "params": {a: {k: v for k, v in p.items()} for a, p in params.items()},
                     "search_cv": {k: v for k, v in store["W"]["_cv"].items() if k != "trials"},
                     "cv": cvm, "random_ndcg": rnd, "top": args.top, "shifts": {}}
    rows = []
    for sh in shifts:
        o, od = oof_shift(wk_eval, daily, cols, cols_w, params, sh)
        res = evaluate(o, od, args.top)
        report(sh, res, args.top)
        summary["shifts"][str(sh)] = res
        rows.append({"shift": sh, **res})

    rd = reading(rows)
    summary["reading"] = rd
    print("\n■ 読み方（回す前に決めたもの。採用の判定ではない）")
    for x, r in zip(rd["rows"], rows):
        t = r["top"]
        print(f"  ずらし{x['shift']}か月: 上位{args.top}件の平均収益（窓の平均）W {_p(t['W']['ret_w'])} / B {_p(t['B']['ret_w'])} / "
              f"D {_p(t['D']['ret_w'])}・−10% 未満 W {_s(t['W']['bad_w'])} / B {_s(t['B']['bad_w'])} / D {_s(t['D']['bad_w'])}"
              f"（{'○' if x['better'] else '×'}）・木・金の割合の差 {x['tf_gap']:+.1f}pt")
    print(f"  → 並べ方: {'週ごとに学ぶ並べ方は、今のスコアで週の候補を並べるより良い' if rd['better'] else 'はっきりとは良くない'}"
          f" ／ 曜日: {LEAN_JA[rd['lean']]}")
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
