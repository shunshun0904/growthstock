#!/usr/bin/env python3
"""
実験114b: 実験114 のランキング学習（L）を、今の規則の形で確かめる（検証だけ。本番は変えない）。

運用者の選択（2026-10-24、実験114 の結果を見て）「今の規則の形で確かめる」: 今の規則（1.2σ の3本 95以上）の買いと、ランキング
学習を使う規則（単独で95以上・今の規則かつランキングが上位）を、同じ期間・同じ件数で比べる。docs/MODEL_ADOPTION_RULES.md §78-5。

モデル（どれも 360列 = features.DEFAULT_PRESET）
  lgbm / xgb / cat  1.2σ の3本。木の形は本番の探索の結果（lgbm_params.json の DEFAULT_PRESET の鍵・multi_params.json）で、
                    360列で探索したものでなければ止める（週次の再学習の後に回す。10/25 の回は 10/24 に前倒しした）。
                    当てはめは本番の out-of-fold と同じ（train_production.oof_scores / train_multi.oof_scores と同じ作り・同じ種）
  ltr               実験114 の L（LGBMRanker。段階は同じ日の中の20営業日目の終値の5分位）。木の形は実験114 の探索の結果
                    （e114_params.json）、種は 42 の1つ（実験114 の out-of-fold をそのまま読む。本番の3本と同じく種1つ）
窓: 基準モデルの out-of-fold と同じ窓を 0 / 2 / 4か月ずらした3通り（e41.folds_for）。4本とも同じ窓・同じ行を採点する
百分位: 前向き（テスト窓より前の窓のスコアの分布での位置。e79.window_split と同じ作りで、最初の2窓は参照が足りないので外す）。
        画面と同じ小数1桁に丸める（e82.rounded）
確かめ: ずらし0の lgbm / xgb / cat のスコアが、週次の再学習が Release（data-raw）に上げた out-of-fold（oof.parquet /
        xgb_oof.parquet / cat_oof.parquet）と同じか（突き合わせた行数・差の最大・一致した行の割合）

買い方（収益は今の売り方: 翌営業日の寄りで買い、+10% の指値、届かなければ20営業日目の終値）
  R0   今の規則: 1.2σ の3本の百分位がすべて95以上
  L95  ランキング学習の百分位が95以上
  R0L  今の規則を満たし、かつランキング学習の百分位が50以上（上半分。日付をまたいだ百分位）
  R0D  【判定】今の規則を満たし、かつその日の候補の中でランキング学習の順位が上半分（同じ日の中の百分位が 0.5 以上。候補が
       1件の日は通る）。分類（3本）で日付をまたいで絞り、ランキング学習でその日の中の順番を付ける「両方使う形」
  LN   同じ件数: 窓ごとに R0 と同じ件数を、ランキング学習のスコアの上から取る
  ・それぞれの件数・平均収益（全件 / 窓の平均）・−10% 未満・+10% 到達・1件も無い月と、R0 との窓ごとの差
  ・R0 の中を、ランキング学習の上半分 / 下半分に分けた平均収益（日付をまたいだ百分位で / その日の中の順位で）
  ・日付をまたいだ力: 候補5件以上の日ごとに、4本それぞれの百分位の平均と、その日の候補の平均収益（今の売り方 / 20営業日目の
    終値）の、日をまたいだ順位相関。日ごとの平均百分位の上位20% の日と下位20% の日の平均収益（運用者の見立て「その日の中の
    ランキングは日付をまたいだ値上がりしやすさを踏まえていない」を数字で見る）
  ・3枠の形（同時に3銘柄まで・その日の上位2件・発火を問わない / 8件以上の日。実験87・88 の simulate）: R0 を今の並び
    （3本の最小の百分位の順、同点は基準モデルのスコア）で / R0 をランキング学習のスコアの順で / R0D を今の並びで / L95 を
    ランキング学習の順で、の取引数・1取引の平均・資金全体の月の損益

判定（回す前に決めたもの。運用者の選択「両方使う形」2026-10-24）
  R0D が 0 / 2 / 4か月の3通りすべてで、R0 より平均収益（窓の平均）が高く、かつ −10% 未満の割合（窓の平均）が低いとき
  「両方使う形で今の規則を上回る」。LN で同じ判定をしたものも参考に並べる。満たしても規則を変えるかは運用者が決める
  （最初は LN で判定する案だったが、回す前に運用者の指摘「その日の中での相対的なランキングだと、日付をまたいで見た場合に、
  より値上げする確率が高い銘柄を踏まえたランキングではない」「両者は補完し合うものなので、両方使うのがベスト」を受けて変えた）

速報（運用者の選択「今すぐ速報＋今夜本番の形」2026-10-24。§78-6）
  1.2σ の3本の代わりに基準モデル1本（実験114 の C: 360列で探索した 1.2σ の LightGBM。本番の 1.2σ の3本はまだ 239列で
  学習したもの）を使い、同じ買い方・同じ比べ方をする。C と L はどちらも実験114 の out-of-fold（種3つの平均。実験114 の
  集計と同じ値になるかを確かめる）をそのまま読み、参考に種42の1つずつでも同じ判定をする。R0 は「基準モデル 95以上」、
  R0D は「基準モデル 95以上 かつ その日の中でランキング学習が上半分」と読み替える。判定の形は上と同じ（R0D が3通りすべてで
  R0 より平均収益（窓の平均）が高く、−10% 未満（窓の平均）が低い）。結果は research/_data/oof/e114q_summary.json

公開ログには集計だけ（件数・平均・割合・木の形）。銘柄ごとの値は出さない。結果は research/_data/oof/e114b_*。

    exp=e114_ltr_retest.py  args="--rule"     （速報は args="--quick"）
    （実験114 の探索の結果と L の out-of-fold を Actions のキャッシュから読むため、実験114 の名前で回す。手元なら
     python3 research/exp/e114b_ltr_rule.py [--quick]）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as LT  # noqa: E402
import models as M  # noqa: E402
import tuning  # noqa: E402
import tuning_multi as TM  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e79_model_clusters as E79  # noqa: E402
import e82_and_rule_variants as E82  # noqa: E402
import e87_monthly_return_search as E87  # noqa: E402
import e114_ltr_retest as E  # noqa: E402

PREFIX = "e114b"
OOF_DIR = E.OOF_DIR
SUMMARY_PATH = os.path.join(OOF_DIR, f"{PREFIX}_summary.json")
TREES = ("lgbm", "xgb", "cat")
LTR = "ltr"
MODELS4 = TREES + (LTR,)
LINE = 95.0                    # 今の規則の線
UPPER = 50.0                   # 「ランキング学習が上位」= 上半分
L_SEED = 42                    # L は実験114 の種 42 の out-of-fold（種1つ）
TOP_K = LT.TOP_K               # その日の上位何件まで（2）
SLOTS = LT.SLOTS               # 同時に持つ銘柄（3）
BUSY = LT.PREV_SKIP_BREAKS     # 発火8件以上の日
BAD = E.BAD
RET, HIT, HITDAY, R20 = E.RET, E.HIT, E.HITDAY, E.R20
PROD_OOF = {"lgbm": "oof.parquet", "xgb": "xgb_oof.parquet", "cat": "cat_oof.parquet"}
RULES = (("R0", "今の規則（1.2σ の3本 95以上）"), ("L95", "ランキング学習 95以上"),
         ("R0L", "今の規則 かつ ランキング学習 50以上"), ("R0D", "今の規則 かつ その日の中でランキング学習が上半分"),
         ("LN", "今の規則と同じ件数をランキング学習の上から"))
#: 判定に使う買い方（運用者の選択「両方使う形」）と、参考に同じ判定をする買い方
PRIMARY = "R0D"
SECONDARY = "LN"
DAY_UPPER = 0.5                # その日の中の百分位（0 = 最下位 / 1 = 最上位）がこれ以上なら上半分
CROSS_TOP = 0.2                # 日付をまたいだ力: 平均百分位の上位 / 下位何割の日を比べるか
RULE_JA = dict(RULES)
#: 速報: 基準モデル1本（実験114 の C を lgbm の名前で並べる）＋ランキング学習。買い方の鍵は同じで、R0 を読み替える
QUICK_TREES = ("lgbm",)
QUICK_MODELS = QUICK_TREES + (LTR,)
QUICK_RULES = (("R0", "基準モデル 95以上"), ("L95", "ランキング学習 95以上"),
               ("R0L", "基準モデル 95以上 かつ ランキング学習 50以上"),
               ("R0D", "基準モデル 95以上 かつ その日の中でランキング学習が上半分"),
               ("LN", "基準モデル 95以上と同じ件数をランキング学習の上から"))
QUICK_SUMMARY_PATH = os.path.join(OOF_DIR, "e114q_summary.json")

log = E.log


# --------------------------------------------------------------------------- #
# 本番の3本
# --------------------------------------------------------------------------- #

def prod_params(cols: Sequence[str], lgbm_store: Dict | None = None, multi_store: Dict | None = None) -> Dict[str, Dict]:
    """
    本番の3本の木の形。学習する列（cols）で探索したものでなければ止める（train_production.py / e15_tune_all.py と同じ確かめ）。
    """
    cols = list(cols)
    sig = F.signature(cols)
    store = tuning.load_params() if lgbm_store is None else lgbm_store
    rec = store.get(F.DEFAULT_PRESET) or {}
    why = tuning.tuned_mismatch(rec.get("_features_sig"), rec.get("_n_features"), cols) if rec else "探索の結果が無い"
    if why:
        raise SystemExit(f"基準モデル（lgbm）の木の形が {len(cols)}列で探索したものではない（{why}）。"
                         "週次の再学習（360列の最初の探索）の後に回す")
    out = {"lgbm": tuning.params_for(F.DEFAULT_PRESET, store)}
    ms = TM.load() if multi_store is None else multi_store
    for a in ("xgb", "cat"):
        cv = (ms.get(a) or {}).get("_cv", {})
        if cv.get("features_sig") != sig:
            raise SystemExit(f"{a} の木の形が {len(cols)}列で探索したものではない（{cv.get('n_features', '?')}列・"
                             f"指紋 {cv.get('features_sig', 'なし')}）。週次の再学習の後に回す")
        out[a] = TM.params_for(a, ms)
    return out


def fit_tree(algo: str, params: Dict, tr: pd.DataFrame, te: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """本番の out-of-fold と同じ当てはめ（lgbm は train_production.oof_scores、xgb / cat は train_multi.oof_scores）。"""
    cols = list(cols)
    X = tr[cols].to_numpy(dtype=float)
    y = tr["label"].to_numpy(dtype=int)
    Xte = te[cols].to_numpy(dtype=float)
    if algo == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params, scale_pos_weight=tuning.scale_pos_weight(y))
        m.fit(X, y)
        return m.predict_proba(Xte)[:, 1]
    m = M.fit(algo, X, y, cols, params=params)
    return M.predict(m, Xte)


def eval_mask(df: pd.DataFrame) -> pd.Series:
    """採点する行（実験114 の oof_arm と同じ: ラベル・今の売り方の収益・20営業日目の終値がある行）。"""
    return df["label"].notna() & df[RET].notna() & df[R20].notna()


def tree_tag(algo: str, params: Dict, cols: Sequence[str], df: pd.DataFrame) -> str:
    key = {"algo": algo, "params": {k: params[k] for k in sorted(params)}, "cols": F.signature(list(cols)),
           "rows": int(len(df)), "last": str(pd.to_datetime(df["Date"]).max().date()),
           "eval_rows": int(eval_mask(df).sum()), "v": 1}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


def oof_tree(df: pd.DataFrame, cols: Sequence[str], algo: str, params: Dict, shift: int,
             out_dir: str = OOF_DIR) -> pd.DataFrame:
    """本番の3本の1つの out-of-fold（学習はラベルのある行 = 本番と同じ。採点は eval_mask の行）。保存済みなら読む。"""
    path = os.path.join(out_dir, f"{PREFIX}_{algo}_{tree_tag(algo, params, cols, df)}_sh{shift}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    folds = E41.folds_for(df["Date"], shift)
    d = pd.to_datetime(df["Date"])
    ev = eval_mask(df)
    rows = []
    for f in folds:
        tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna()]
        te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end)) & ev]
        if len(te) < 200 or len(tr) < 1000:
            continue
        part = te[["Code", "Date"]].copy()
        part["score"] = fit_tree(algo, params, tr, te, cols)
        part["fold"] = int(f.index)
        rows.append(part)
    if not rows:
        raise SystemExit(f"{algo} ずらし{shift}か月: 窓を作れない")
    o = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    o.to_parquet(path, index=False)
    log(f"  {algo} ずらし{shift}か月: {o['fold'].nunique()}窓・{len(o):,}件 {time.time() - t0:.0f}秒")
    return o


def prod_check(o: pd.DataFrame, algo: str, data_dir: str = lab.DATA_DIR, tol: float = 1e-6) -> Dict:
    """ずらし0の out-of-fold が、週次の再学習が上げた本番の out-of-fold と同じか。"""
    p = os.path.join(data_dir, PROD_OOF[algo])
    if not os.path.exists(p):
        return {"algo": algo, "found": False}
    prod = pd.read_parquet(p, columns=["Code", "Date", "score"])
    prod["Code"] = prod["Code"].astype(str)
    prod["Date"] = pd.to_datetime(prod["Date"])
    ours = o[["Code", "Date", "score"]].copy()
    ours["Code"] = ours["Code"].astype(str)
    ours["Date"] = pd.to_datetime(ours["Date"])
    m = ours.merge(prod, on=["Code", "Date"], suffixes=("", "_prod"))
    diff = np.abs(m["score"].to_numpy(dtype=float) - m["score_prod"].to_numpy(dtype=float))
    return {"algo": algo, "found": True, "ours": int(len(ours)), "prod": int(len(prod)), "matched": int(len(m)),
            "max_diff": float(diff.max()) if len(m) else float("nan"),
            "same_share": float(np.mean(diff <= tol)) if len(m) else float("nan")}


# --------------------------------------------------------------------------- #
# 前向きの百分位と規則
# --------------------------------------------------------------------------- #

def wide(parts: Dict[str, pd.DataFrame], df: pd.DataFrame, models: Sequence[str] = MODELS4) -> pd.DataFrame:
    """models（既定は4本）の out-of-fold を (Code, Date) で横に並べ、収益の列を足す。採点した行が違えば止める。"""
    keys = None
    base = None
    for a in models:
        o = parts[a][["Code", "Date", "score", "fold"]].copy()
        o["Code"] = o["Code"].astype(str)
        o["Date"] = pd.to_datetime(o["Date"])
        o = o.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)
        k = o[["Code", "Date", "fold"]].astype(str).to_numpy()
        if keys is None:
            keys = k
            base = o.rename(columns={"score": f"score_{a}", "fold": "window"})
        else:
            if not np.array_equal(keys, k):
                raise SystemExit(f"{a} の採点した行・窓がほかのモデルと違う")
            base[f"score_{a}"] = o["score"].to_numpy(dtype=float)
    src = df[["Code", "Date", "label", RET, HIT, HITDAY, R20, "vol_20d"]].copy()
    src["Code"] = src["Code"].astype(str)
    src["Date"] = pd.to_datetime(src["Date"])
    out = base.merge(src, on=["Code", "Date"], how="left", validate="one_to_one")
    out["window"] = out["window"].astype(int)
    return out


def forward_frame(w: pd.DataFrame, folds, models: Sequence[str] = MODELS4,
                  trees: Sequence[str] = TREES) -> pd.DataFrame:
    """
    前向きの百分位（e79.window_split。画面と同じ丸め）と、trees（今の規則に使う分類。既定は 1.2σ の3本、速報は基準モデル
    1本）の最小・その日の発火数を足す。
    """
    split = E79.window_split(w, folds, models=list(models))
    outs = []
    for ww in sorted(split):
        _, te, _, Pte = split[ww]
        R = E82.rounded(Pte)
        g = w.iloc[te].copy()
        for k, a in enumerate(models):
            g[f"pct_{a}"] = R[:, k]
        outs.append(g)
    if not outs:
        raise SystemExit("前向きの百分位を付けられる窓が無い")
    ft = pd.concat(outs, ignore_index=True)
    ft["min_t3"] = ft[[f"pct_{a}" for a in trees]].min(axis=1, skipna=False)
    ft["n_break"] = ft.groupby("Date")["Code"].transform("size").to_numpy()
    # その日の候補の中でのランキング学習の百分位（0 = 最下位 / 1 = 最上位。候補が1件の日は 0.5。実験52・114 の段階と同じ作り）
    ft["day_pct_ltr"] = E.E52.within_date_pct(ft["Date"], ft["score_ltr"].to_numpy(dtype=float))
    return ft


def rule_masks(ft: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    R0 / L95 / R0L / R0D / LN の印（NaN の百分位は満たさない）。R0D はその日の中のランキング学習の百分位が DAY_UPPER 以上。
    LN は窓ごとに R0 と同じ件数をランキング学習の上から。
    """
    t3 = np.nan_to_num(ft["min_t3"].to_numpy(dtype=float), nan=-1.0) >= LINE
    lp = np.nan_to_num(ft["pct_ltr"].to_numpy(dtype=float), nan=-1.0)
    dp = ft["day_pct_ltr"].to_numpy(dtype=float)
    out = {"R0": t3, "L95": lp >= LINE, "R0L": t3 & (lp >= UPPER), "R0D": t3 & (dp >= DAY_UPPER)}
    ln = np.zeros(len(ft), dtype=bool)
    win = ft["window"].to_numpy()
    for ww in np.unique(win):
        idx = np.where(win == ww)[0]
        n = int(t3[idx].sum())
        if n <= 0:
            continue
        sc = ft["score_ltr"].to_numpy(dtype=float)[idx]
        order = idx[np.argsort(-sc, kind="mergesort")]
        ln[order[:n]] = True
    out["LN"] = ln
    return out


def per_window(ft: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """窓ごとの件数・平均収益・−10% 未満の割合（選んだ行のある窓だけ）。"""
    s = ft[mask]
    if not len(s):
        return pd.DataFrame(columns=["window", "n", "ret", "bad"])
    g = s.groupby("window")[RET]
    return pd.DataFrame({"n": g.size(), "ret": g.mean(),
                         "bad": g.apply(lambda x: float(np.mean(x.to_numpy(dtype=float) < BAD)))}).reset_index()


def rule_stats(ft: pd.DataFrame, mask: np.ndarray) -> Dict:
    s = ft[mask]
    r = s[RET].to_numpy(dtype=float)
    pw = per_window(ft, mask)
    months = pd.to_datetime(ft["Date"]).dt.to_period("M")
    pm = months[mask].value_counts()
    allm = pd.period_range(months.min(), months.max(), freq="M")
    return {"n": int(len(s)), "ret": float(np.mean(r)) if len(r) else float("nan"),
            "ret_w": float(pw["ret"].mean()) if len(pw) else float("nan"),
            "bad": float(np.mean(r < BAD)) if len(r) else float("nan"),
            "bad_w": float(pw["bad"].mean()) if len(pw) else float("nan"),
            "hit": float(s[HIT].astype(float).mean()) if len(s) else float("nan"),
            "zero_months": int(sum(1 for m in allm if pm.get(m, 0) == 0)), "months": int(len(allm))}


def window_diff(ft: pd.DataFrame, base: np.ndarray, cur: np.ndarray, key: str = "ret") -> Dict:
    """窓ごとの差（cur − base）の平均・標準誤差・上回った窓の数（両方に選んだ行がある窓だけ）。"""
    a, b = per_window(ft, base), per_window(ft, cur)
    m = a.merge(b, on="window", suffixes=("_b", "_c"))
    dv = (m[f"{key}_c"] - m[f"{key}_b"]).to_numpy(dtype=float)
    if not len(dv):
        return {"mean": float("nan"), "se": float("nan"), "won": 0, "n": 0}
    se = float(dv.std(ddof=1) / np.sqrt(len(dv))) if len(dv) > 1 else float("nan")
    return {"mean": float(dv.mean()), "se": se, "won": int((dv > 0).sum()), "n": int(len(dv))}


# --------------------------------------------------------------------------- #
# 3枠の形
# --------------------------------------------------------------------------- #

def ordered_candidates(ft: pd.DataFrame, mask: np.ndarray, order: str, min_break: int = 1,
                       top_k: int = TOP_K) -> pd.DataFrame:
    """
    その日の候補を並べて上位 top_k 件まで。order="min" は今の並び（3本の最小の百分位の高い順、同点は基準モデルのスコア。
    e87.candidates と同じ）、order="ltr" はランキング学習のスコアの高い順。
    """
    c = ft[mask & (ft["n_break"].to_numpy() >= min_break)].copy()
    if order == "min":
        c = c.sort_values(["Date", "min_t3", "score_lgbm"], ascending=[True, False, False], kind="mergesort")
    elif order == "ltr":
        c = c.sort_values(["Date", "score_ltr"], ascending=[True, False], kind="mergesort")
    else:
        raise ValueError(order)
    c["rank"] = c.groupby("Date").cumcount().to_numpy() + 1
    return c[c["rank"] <= top_k]


def slot_trades(c: pd.DataFrame, bar_days, slots: int = SLOTS) -> pd.DataFrame:
    """3枠の形で買えた取引（e87.simulate: 翌営業日に買い、指値に届いた翌営業日か20営業日の翌営業日から枠が空く）。"""
    idx = {pd.Timestamp(d): k for k, d in enumerate(bar_days)}
    day = np.array([idx.get(pd.Timestamp(d), -1) for d in c["Date"]], dtype=float)
    keep = day >= 0
    c, day = c[keep], day[keep]
    ret = c[RET].to_numpy(dtype=float) * 100.0
    hit = c[HIT].fillna(False).to_numpy(dtype=bool)
    hitday = pd.to_numeric(c[HITDAY], errors="coerce").to_numpy(dtype=float)
    month = pd.to_datetime(c["Date"]).dt.to_period("M").map(lambda p: p.ordinal).to_numpy()
    taken = E87.simulate(day, ret, hit, hitday, month, slots, 0)
    return c.iloc[taken]


def slot_stats(t: pd.DataFrame, months: int, slots: int = SLOTS) -> Dict:
    r = t[RET].to_numpy(dtype=float) * 100.0
    return {"n": int(len(t)), "mean": float(np.mean(r)) if len(r) else float("nan"),
            "win": float(np.mean(r > 0) * 100) if len(r) else float("nan"),
            "capital_pm": float(r.sum() / slots / months) if months else float("nan")}


def verdict(rows: List[Dict], key: str = PRIMARY) -> Dict:
    """
    判定（回す前に決めたもの）: すべてのずらし方で key（既定は R0D）の平均収益（窓の平均）が R0 より高く、−10% 未満
    （窓の平均）が低い。
    """
    out = []
    for r in rows:
        ok_r = bool(r[key]["ret_w"] > r["R0"]["ret_w"])
        ok_b = bool(r[key]["bad_w"] < r["R0"]["bad_w"])
        out.append({"shift": int(r["shift"]), "key": key, "ret_R0": r["R0"]["ret_w"], "ret_X": r[key]["ret_w"],
                    "bad_R0": r["R0"]["bad_w"], "bad_X": r[key]["bad_w"], "ret_ok": ok_r, "bad_ok": ok_b,
                    "ok": ok_r and ok_b})
    return {"key": key, "rows": out, "pass": bool(out) and all(x["ok"] for x in out)}


def cross_date(ft: pd.DataFrame, models: Sequence[str] = MODELS4, min_n: int = E.MIN_DAY,
               top: float = CROSS_TOP) -> Dict[str, Dict]:
    """
    日付をまたいだ力: 候補 min_n 件以上の日ごとに、各モデルの百分位の平均とその日の候補の平均収益の、日をまたいだ順位相関
    （今の売り方 rho / 20営業日目の終値 rho20）と、平均百分位の上位 top の日・下位 top の日の平均収益（今の売り方）。
    """
    from scipy.stats import spearmanr

    size = ft.groupby("Date")["Code"].transform("size").to_numpy()
    sub = ft[size >= min_n]
    if not len(sub):
        return {}
    agg = sub.groupby("Date").agg(ret=(RET, "mean"), r20=(R20, "mean"),
                                  **{f"m_{a}": (f"pct_{a}", "mean") for a in models})
    out = {}
    for a in models:
        x = agg[f"m_{a}"]
        q = x.rank(pct=True)
        out[a] = {"days": int(len(agg)),
                  "rho": float(spearmanr(x, agg["ret"], nan_policy="omit").correlation),
                  "rho20": float(spearmanr(x, agg["r20"], nan_policy="omit").correlation),
                  "top_days": float(agg.loc[q > 1 - top, "ret"].mean()),
                  "bottom_days": float(agg.loc[q <= top, "ret"].mean())}
    return out


def _p(x: float, nd: int = 2) -> str:
    return f"{x * 100:+.{nd}f}%" if np.isfinite(x) else "—"


def _s(x: float) -> str:
    return f"{x * 100:.1f}%" if np.isfinite(x) else "—"


def report_shift(ft: pd.DataFrame, bar_days, rules: Sequence = RULES, models: Sequence[str] = MODELS4,
                 order_ja: str = "今の並び") -> tuple:
    """
    1つのずらし方の表を出して集計を返す（買い方ごと・R0 の中をランキング学習で分けた平均・日付をまたいだ力・3枠の形）。
    rules は (鍵, 日本語) の並び、models は日付をまたいだ力に並べるモデル、order_ja は order="min" の並びの呼び名。
    """
    rule_ja = dict(rules)
    masks = rule_masks(ft)
    print("  ◆ 買い方ごと: 件数 / 平均収益（全件・窓の平均）/ −10% 未満（全件・窓の平均）/ +10% 到達 / 1件も無い月 ｜ R0 との差（窓ごと）")
    st = {}
    for key, ja in rules:
        s = rule_stats(ft, masks[key])
        st[key] = s
        line = (f"    {key:<4}{ja:<28}{s['n']:>6}件 {_p(s['ret']):>8} {_p(s['ret_w']):>8} {_s(s['bad']):>6} "
                f"{_s(s['bad_w']):>6} {_s(s['hit']):>6} {s['zero_months']:>3} / {s['months']}月")
        if key != "R0":
            dd = window_diff(ft, masks["R0"], masks[key])
            db = window_diff(ft, masks["R0"], masks[key], "bad")
            line += (f" ｜ {dd['mean'] * 100:+.2f} ± {dd['se'] * 100:.2f}pt（上の窓 {dd['won']}/{dd['n']}）・"
                     f"−10%未満 {db['mean'] * 100:+.1f}pt")
            s["diff"] = dd
            s["diff_bad"] = db
        print(line)
    lp = np.nan_to_num(ft["pct_ltr"].to_numpy(dtype=float), nan=-1.0)
    up, lowr = masks["R0"] & (lp >= UPPER), masks["R0"] & (lp < UPPER)
    su, sl = rule_stats(ft, up), rule_stats(ft, lowr)
    print(f"  ◆ {rule_ja['R0']}の中をランキング学習で分ける: 上半分 {su['n']}件 {_p(su['ret'])}（−10%未満 {_s(su['bad'])}）/ "
          f"下半分 {sl['n']}件 {_p(sl['ret'])}（{_s(sl['bad'])}）")
    st["R0_upper"], st["R0_lower"] = su, sl
    dp = ft["day_pct_ltr"].to_numpy(dtype=float)
    du, dl = rule_stats(ft, masks["R0"] & (dp >= DAY_UPPER)), rule_stats(ft, masks["R0"] & (dp < DAY_UPPER))
    print(f"  ◆ {rule_ja['R0']}の中を、その日の中のランキング学習の順位で分ける: 上半分 {du['n']}件 {_p(du['ret'])}"
          f"（−10%未満 {_s(du['bad'])}）/ 下半分 {dl['n']}件 {_p(dl['ret'])}（{_s(dl['bad'])}）")
    st["R0_day_upper"], st["R0_day_lower"] = du, dl
    cd = cross_date(ft, models)
    if cd:
        print(f"  ◆ 日付をまたいだ力（候補{E.MIN_DAY}件以上の {next(iter(cd.values()))['days']}日）: 日ごとの平均百分位と、"
              f"その日の候補の平均収益の順位相関（今の売り方 / 20営業日目の終値）・平均百分位の上位{CROSS_TOP * 100:.0f}% の日 / "
              f"下位{CROSS_TOP * 100:.0f}% の日の平均収益")
        for a in models:
            c = cd[a]
            print(f"    {a:<5} {c['rho']:>+7.3f} / {c['rho20']:>+7.3f} ｜ {_p(c['top_days'])} / {_p(c['bottom_days'])}")
    st["cross_date"] = cd
    months = int(pd.to_datetime(ft["Date"]).dt.to_period("M").nunique())
    print(f"  ◆ 3枠の形（同時に{SLOTS}銘柄・その日の上位{TOP_K}件）: 取引数 / 1取引の平均 / 勝率 / 資金全体の月の損益（{months}か月）")
    slots_out = {}
    for mb, mb_ja in ((1, "発火を問わない"), (BUSY, f"発火{BUSY}件以上の日")):
        for name, key, order in ((f"R0・{order_ja}", "R0", "min"), ("R0・ランキングの並び", "R0", "ltr"),
                                 (f"R0D・{order_ja}", "R0D", "min"), ("L95・ランキングの並び", "L95", "ltr")):
            t = slot_trades(ordered_candidates(ft, masks[key], order, min_break=mb), bar_days)
            ss = slot_stats(t, months)
            slots_out[f"{key}_{order}_{mb}"] = ss
            print(f"    {mb_ja:<14}{name:<22}{ss['n']:>5}取引 {ss['mean']:>+7.2f}% {ss['win']:>5.1f}% "
                  f"{ss['capital_pm']:>+7.2f}%")
    return st, slots_out


def print_verdict(v: Dict, v2: Dict, rules: Sequence = RULES, base_ja: str = "今の規則",
                  pass_ja: str = "両方使う形で今の規則を上回る（規則を変えるかは運用者が決める）",
                  fail_ja: str = "両方使う形では今の規則を上回らない") -> None:
    rule_ja = dict(rules)
    print(f"\n■ 判定（先に決めた基準: すべてのずらし方で、{PRIMARY}（{rule_ja[PRIMARY]}）の平均収益（窓の平均）が{base_ja}より高く、"
          "−10% 未満（窓の平均）が低い）")
    for r in v["rows"]:
        print(f"  ずらし{r['shift']}か月: 平均収益 R0 {_p(r['ret_R0'])} / {PRIMARY} {_p(r['ret_X'])}（{'○' if r['ret_ok'] else '×'}）・"
              f"−10%未満 R0 {_s(r['bad_R0'])} / {PRIMARY} {_s(r['bad_X'])}（{'○' if r['bad_ok'] else '×'}）")
    print(f"  → {pass_ja if v['pass'] else fail_ja}")
    print(f"  参考（{SECONDARY}: {rule_ja[SECONDARY]} で同じ判定）: " + " / ".join(
        f"ずらし{r['shift']} {'○' if r['ok'] else '×'}" for r in v2["rows"]) + f" → {'満たす' if v2['pass'] else '満たさない'}")


def cached_seeds(df: pd.DataFrame, cols: Sequence[str], arm: str, params: Dict, shift: int, seeds: Sequence[int],
                 out_dir: str = OOF_DIR) -> int:
    """実験114 の out-of-fold が置き場にある種の数（無い種は E.oof_arm が作り直すので時間がかかる）。"""
    tag = E.run_tag(arm, params, cols, df)
    return int(sum(os.path.exists(os.path.join(out_dir, f"{E.PREFIX}_{arm}_{tag}_sh{shift}_s{sd}.parquet"))
                   for sd in seeds))


def check_114(res: Dict[str, pd.DataFrame], prev: Dict | None, shift: int) -> Dict[str, Dict]:
    """
    読んだ out-of-fold から実験114 の判定の数（窓の中の上位10% の件数と平均収益）を計算し直し、実験114 の集計
    （e114_summary.json）と同じかを見る。同じなら、実験114 で評価したものと同じ out-of-fold を読んでいる。
    """
    rec = ((prev or {}).get("shifts") or {}).get(str(shift)) or {}
    out = {}
    for arm, o in res.items():
        cur = E.top_within(o, 0.10)
        got = {"n": int(cur["n"].sum()), "ret": float(cur["ret"].mean())}
        want = rec.get(f"top10_{arm}")
        same = bool(want is not None and int(want["n"]) == got["n"] and abs(float(want["ret"]) - got["ret"]) < 1e-12)
        out[arm] = {**got, "found": want is not None, "same": same,
                    "ret_114": float(want["ret"]) if want is not None else float("nan")}
    return out


def quick_main(shifts: Sequence[int]) -> int:
    """
    速報: 1.2σ の3本の代わりに基準モデル1本（実験114 の C）を使い、ランキング学習（実験114 の L）と「両方使う形」を確かめる。
    どちらも実験114 の out-of-fold（種3つの平均）を読む。参考に種42の1つずつでも同じ判定をする。
    """
    t_all = time.time()
    cols = F.columns(F.DEFAULT_PRESET)
    if not os.path.exists(E.PARAMS_PATH):
        raise SystemExit(f"実験114 の探索の結果が無い（{E.PARAMS_PATH}）。先に実験114 を回す")
    with open(E.PARAMS_PATH, encoding="utf-8") as fh:
        store = json.load(fh)
    params = {"C": store["C"]["params"], "L": store["L"]["params"]}
    prev = None
    if os.path.exists(E.SUMMARY_PATH):
        with open(E.SUMMARY_PATH, encoding="utf-8") as fh:
            prev = json.load(fh)
    full, cutoff, chk = E.load_frame(cols)
    df = full[full["label"].notna()].reset_index(drop=True)
    sig_now = E.data_sig(cols, full[full["Date"] <= cutoff], store["_sig"]["n_trials"])
    same_data = store.get("_sig") == sig_now
    bars = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=10))
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    del bars
    seeds = tuple(E.SEEDS)

    print("=" * 96)
    print(f"実験114b 速報 基準モデル1本＋ランキング学習で「両方使う形」を確かめる（検証だけ）: {len(df):,}件・"
          f"{df['Date'].nunique():,}日・{df['Date'].min().date()} 〜 {df['Date'].max().date()} / {len(cols)}列 / "
          f"ずらし {list(shifts)}か月")
    print(f"  収益: 今の売り方（翌営業日の寄りで買い、+{E.TARGET:g}% の指値、届かなければ{E.HOLD}営業日目の終値）")
    print(f"  基準モデル（lgbm の名前で並べる）: 実験114 の C（360列で探索した 1.2σ の LightGBM）木の形: {E._tree_line(params['C'])}")
    print(f"  ランキング学習（ltr）: 実験114 の L 木の形: {E._tree_line(params['L'])}")
    print(f"  どちらも実験114 の out-of-fold（種 {' / '.join(str(s) for s in seeds)} の平均）。参考に種{L_SEED}の1つずつでも同じ判定 / "
          f"{'実験114 と同じデータ' if same_data else '実験114 のときとデータが違う（out-of-fold を作り直す）'}")
    print("=" * 96)
    log(f"[確かめ] 同じ計算の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,} 行")

    summary: Dict = {"mode": "quick", "shifts": {}, "same_data": same_data, "seeds": list(seeds)}
    vrows, vrows1 = [], []
    for sh in shifts:
        have = {arm: cached_seeds(df, cols, arm, params[arm], sh, seeds) for arm in ("C", "L")}
        log(f"[確かめ] ずらし{sh}か月: 置き場にある実験114 の out-of-fold C {have['C']} / {len(seeds)}・"
            f"L {have['L']} / {len(seeds)}（無い種は作り直す）")
        res = {arm: E.oof_arm(df, cols, arm, params[arm], sh, seeds) for arm in ("C", "L")}
        c114 = check_114(res, prev, sh)
        log(f"[確かめ] 実験114 の判定の数（窓の中の上位10%）を読み直した out-of-fold で計算し直す: " + " / ".join(
            f"{arm} {c['n']:,}件 {_p(c['ret'])}（実験114 {_p(c['ret_114'])}・"
            f"{'一致' if c['same'] else ('不一致' if c['found'] else '集計なし')}）" for arm, c in c114.items()))
        folds = E41.folds_for(df["Date"], sh)
        ft = forward_frame(wide({"lgbm": res["C"], LTR: res["L"]}, df, QUICK_MODELS), folds,
                           QUICK_MODELS, QUICK_TREES)
        print(f"\n■ ずらし{sh}か月（前向きの百分位を付けた {ft['window'].nunique()}窓・{len(ft):,}件・"
              f"{ft['Date'].min().date()} 〜 {ft['Date'].max().date()}・全体の平均 {_p(np.nanmean(ft[RET]))}・"
              f"−10% 未満 {_s(np.nanmean(ft[RET] < BAD))}）")
        st, slots_out = report_shift(ft, bar_days, QUICK_RULES, QUICK_MODELS, "基準モデルの並び")
        one = {arm: E.oof_arm(df, cols, arm, params[arm], sh, (L_SEED,)) for arm in ("C", "L")}
        ft1 = forward_frame(wide({"lgbm": one["C"], LTR: one["L"]}, df, QUICK_MODELS), folds,
                            QUICK_MODELS, QUICK_TREES)
        m1 = rule_masks(ft1)
        st1 = {k: rule_stats(ft1, m1[k]) for k in ("R0", PRIMARY, SECONDARY)}
        print(f"  ◆ 参考（種{L_SEED}の1つずつ）: 件数 / 平均収益（窓の平均）/ −10% 未満（窓の平均）: " + " / ".join(
            f"{k} {s['n']}件 {_p(s['ret_w'])} {_s(s['bad_w'])}" for k, s in st1.items()))
        summary["shifts"][str(sh)] = {"stats": st, "slots": slots_out, "seed1": st1, "check114": c114,
                                      "windows": int(ft["window"].nunique()), "rows": int(len(ft))}
        vrows.append({"shift": sh, **{k: st[k] for k in ("R0", PRIMARY, SECONDARY)}})
        vrows1.append({"shift": sh, **st1})

    v, v2, v1 = verdict(vrows, PRIMARY), verdict(vrows, SECONDARY), verdict(vrows1, PRIMARY)
    summary["verdict"], summary["verdict_ref"], summary["verdict_seed1"] = v, v2, v1
    print_verdict(v, v2, QUICK_RULES, "基準モデル 95以上",
                  "速報では、両方使う形が基準モデル 95以上を上回る（今の規則の形は週次の再学習の後に確かめる）",
                  "速報では、両方使う形は基準モデル 95以上を上回らない（今の規則の形は週次の再学習の後に確かめる）")
    print(f"  参考（種{L_SEED}の1つずつで {PRIMARY} の判定）: " + " / ".join(
        f"ずらし{r['shift']} {'○' if r['ok'] else '×'}" for r in v1["rows"]) + f" → {'満たす' if v1['pass'] else '満たさない'}")
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(QUICK_SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {QUICK_SUMMARY_PATH} （{summary['minutes']}分）")
    return 0


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験114b: ランキング学習を今の規則の形で確かめる（検証だけ）")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--quick", action="store_true",
                    help="速報: 1.2σ の3本の代わりに基準モデル1本（実験114 の C）で両方使う形を確かめる")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    if args.quick:
        return quick_main(shifts)

    t_all = time.time()
    cols = F.columns(F.DEFAULT_PRESET)
    params = prod_params(cols)                       # 360列の週次の再学習の前なら、ここで止まる
    if not os.path.exists(E.PARAMS_PATH):
        raise SystemExit(f"実験114 の探索の結果が無い（{E.PARAMS_PATH}）。先に実験114 を回す")
    with open(E.PARAMS_PATH, encoding="utf-8") as fh:
        store = json.load(fh)
    params["ltr"] = store["L"]["params"]
    full, cutoff, chk = E.load_frame(cols)
    df = full[full["label"].notna()].reset_index(drop=True)
    sig_now = E.data_sig(cols, full[full["Date"] <= cutoff], store["_sig"]["n_trials"])
    same_data = store.get("_sig") == sig_now
    bars = LT.load_bars(lab.DATA_DIR, start=df["Date"].min() - pd.Timedelta(days=10))
    bar_days = list(pd.to_datetime(np.sort(bars["Date"].unique())))
    del bars

    print("=" * 96)
    print(f"実験114b ランキング学習を今の規則の形で確かめる（検証だけ）: {len(df):,}件・{df['Date'].nunique():,}日・"
          f"{df['Date'].min().date()} 〜 {df['Date'].max().date()} / {len(cols)}列 / ずらし {shifts}か月")
    print(f"  収益: 今の売り方（翌営業日の寄りで買い、+{E.TARGET:g}% の指値、届かなければ{E.HOLD}営業日目の終値）")
    for a in MODELS4:
        print(f"  {a:<5} 木の形: {E._tree_line(params[a])}")
    print(f"  ランキング学習の木の形は実験114 の探索（{store.get('_cutoff')} まで）。"
          f"{'同じデータ' if same_data else '実験114 のときとデータが違う（木の形はそのまま使う）'}")
    print("=" * 96)
    log(f"[確かめ] 同じ計算の5日平均が ret_o1_20 と一致 {chk['hold5_match']:,} / {chk['hold5_both']:,} 行")

    summary: Dict = {"shifts": {}, "same_data": same_data}
    vrows = []
    for sh in shifts:
        parts = {a: oof_tree(df, cols, a, params[a], sh) for a in TREES}
        lo = E.oof_arm(df, cols, "L", params["ltr"], sh, (L_SEED,))
        parts[LTR] = lo
        sh_out: Dict = {}
        if sh == 0:
            checks = [prod_check(parts[a], a) for a in TREES]
            sh_out["prod_check"] = checks
            for c in checks:
                if not c["found"]:
                    log(f"[確かめ] {c['algo']}: 本番の out-of-fold（{PROD_OOF[c['algo']]}）が置き場に無いので比べない")
                else:
                    log(f"[確かめ] {c['algo']} ずらし0 対 本番の out-of-fold: 突き合わせ {c['matched']:,} / {c['ours']:,} 行"
                        f"（本番 {c['prod']:,}行）・差の最大 {c['max_diff']:.2e}・一致 {c['same_share'] * 100:.2f}%")
        folds = E41.folds_for(df["Date"], sh)
        w = wide(parts, df)
        ft = forward_frame(w, folds)
        print(f"\n■ ずらし{sh}か月（前向きの百分位を付けた {ft['window'].nunique()}窓・{len(ft):,}件・"
              f"{ft['Date'].min().date()} 〜 {ft['Date'].max().date()}・全体の平均 {_p(np.nanmean(ft[RET]))}・"
              f"−10% 未満 {_s(np.nanmean(ft[RET] < BAD))}）")
        st, slots_out = report_shift(ft, bar_days)
        sh_out.update({"stats": st, "slots": slots_out, "windows": int(ft["window"].nunique()), "rows": int(len(ft))})
        summary["shifts"][str(sh)] = sh_out
        vrows.append({"shift": sh, **{k: st[k] for k in ("R0", PRIMARY, SECONDARY)}})

    v = verdict(vrows, PRIMARY)
    v2 = verdict(vrows, SECONDARY)
    summary["verdict"], summary["verdict_ref"] = v, v2
    print_verdict(v, v2)
    summary["minutes"] = round((time.time() - t_all) / 60, 1)
    os.makedirs(OOF_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"記録: {OOF_DIR}/{PREFIX}_* （{summary['minutes']}分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
