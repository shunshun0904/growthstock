#!/usr/bin/env python3
"""
実験52: 目的を「確率」から「収益」に変えると、上位の実収益はどう変わるか。

運用者の指示（2026-09-26）: 実験50 で、見逃した正例（正例なのに下位10%）は実収益の中央値が
+21〜26% と最も儲かる群だった。理由は、ラベルの到達しきい値が 1.2σ×√20 でボラに比例し、
分類器は「届く確率」しか見ないので「届いたときの大きさ」が大きい側（高ボラ）が必ず下に来る
こと。「(b) 目的を確率から収益に変える（回帰、または ret_o1_20 の順位で学ぶ LTR）」を選んだ。

腕（すべて LightGBM・木200本。木の形のパラメータは本番の探索結果を読むだけで、目的関数だけ
差し替える。回帰・LTR 向けの探索はしていない＝控えめな見積もり）
  C  分類（本番そのもの。objective=binary + scale_pos_weight）。スコアは確率
  R  収益の回帰（objective=huber）。目的は ret_o1_20 を訓練側の 1〜99% で刈り込んだもの
  Q  日付内の収益の順位の回帰（objective=regression）。目的は同じ日の中の ret_o1_20 の
     百分位（0〜1）。「その日の候補の中でどれが上か」だけを学ぶ
  L  LTR（LGBMRanker・lambdarank）。段階は同じ日の中の ret_o1_20 の5分位（0〜4）。
     日付をグループにする

評価（PR-AUC は参考にとどめ、実収益で比べる）
  1. 「スコアが過去分布の上位5% / 10% なら買う」の実収益（lab.threshold_edge。しきい値は
     前の窓のスコア分布から。運用の lgbm95 と同じ形）: 選ばれた件数・平均・全体との差・
     窓ごとの差の平均 ± SE・上の窓・最悪の窓・−10% 未満の割合
  2. 日付内: その日の1位の平均収益、上位2件の平均収益（発火8件以上の日だけ）、
     日付内の順位相関（Spearman、5件以上の日の平均）
  3. 上位10% の中身: ボラの帯の構成と帯ごとの平均収益（分類は低ボラ帯に偏る。それが変わるか）
  4. 参考: PR-AUC / ROC-AUC / 日内AUC（ラベルで測ったもの）
どれも本番と同じ窓（36/6/6か月・エンバーゴ20営業日）を 0/2/4か月ずらした3通り × 種3つの
平均で、窓ごとに出す。

  --shifts 0,2,4  --seeds 3  --arms C,R,Q,L  [--preset all_plus_prog_listing_vol]
  結果は research/_data/oof/e52_*（--preset を変えたときは e52_<列の指紋>_*）。
  本番の設定には書かない。212列（実験51 の6列を足したもの）は e52b_return_objective_vol.py から
  同じ台本を呼ぶ（Actions の同時実行の組が実験名ごとなので、別名にして並行して回す）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import tuning  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
OUTCOME = lab.OUTCOME                       # ret_o1_20
ARMS = ("C", "R", "Q", "L")
LABELS = {"C": "C 分類（本番）", "R": "R 収益の回帰（huber）",
          "Q": "Q 日付内の収益順位の回帰", "L": "L LTR（収益の日付内5分位）"}
#: 回帰の目的の刈り込み（訓練側の分位）。数件の +300% に木を取られないため
CLIP_Q = (0.01, 0.99)
#: LTR の段階数（同じ日の中の収益の分位）
N_GRADES = 5
#: 日付内の指標に使う最小の発火数（運用の見送り基準 SKIP_BREAKS と同じ8件）
MIN_BREAKS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 目的の作り方
# --------------------------------------------------------------------------- #

def clipped_return(r: np.ndarray, lo_hi=CLIP_Q):
    """訓練側の分位で刈り込む（境界は訓練側だけから決める。返り値は (値, 下限, 上限)）。"""
    lo, hi = np.nanquantile(r, lo_hi[0]), np.nanquantile(r, lo_hi[1])
    return np.clip(r, lo, hi), float(lo), float(hi)


def within_date_pct(dates: pd.Series, r: np.ndarray) -> np.ndarray:
    """同じ日の中の収益の百分位（0〜1。1件だけの日は 0.5）。"""
    s = pd.Series(r, index=pd.RangeIndex(len(r)))
    g = s.groupby(pd.Series(dates).to_numpy())
    n = g.transform("size")
    rk = g.rank(method="average")
    return np.where(n > 1, (rk - 1) / (n - 1), 0.5).astype(float)


def within_date_grade(dates: pd.Series, r: np.ndarray, n_grades: int = N_GRADES) -> np.ndarray:
    """同じ日の中の収益の分位（0〜n_grades−1 の整数）。LTR の段階。"""
    p = within_date_pct(dates, r)
    return np.minimum((p * n_grades).astype(int), n_grades - 1)


_PROD: Dict = {}


def tree_params(seed: int) -> Dict:
    """本番の探索結果から、目的関数と不均衡の補正を除いた木の形（読むのは1回だけ）。"""
    if not _PROD:
        _PROD.update(E27.prod_params("lgbm")["params"])
    rec = _PROD
    p = {k: v for k, v in rec.items()
         if not k.startswith("_") and k not in ("objective", "scale_pos_weight")}
    p["random_state"] = seed
    p["verbose"] = -1
    return p


def fit_predict(arm: str, tr: pd.DataFrame, te: pd.DataFrame, cols: List[str], seed: int,
                params: Dict | None = None) -> np.ndarray:
    """
    腕ごとに学習して te のスコアを返す。params を渡すとその木の形を使う（実験53 の腕ごとの
    探索結果。objective / scale_pos_weight は腕が決めるので、渡されても外す）。
    """
    import lightgbm as lgb

    if params is None:
        p = tree_params(seed)
    else:
        p = {k: v for k, v in params.items()
             if not k.startswith("_") and k not in ("objective", "scale_pos_weight")}
        p["random_state"] = seed
        p["verbose"] = -1
    Xtr = tr[cols].to_numpy(dtype=float)
    Xte = te[cols].to_numpy(dtype=float)
    r = tr[OUTCOME].to_numpy(dtype=float)
    if arm == "C":
        y = tr["label"].to_numpy(dtype=int)
        m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=tuning.scale_pos_weight(y), **p)
        m.fit(Xtr, y)
        return m.predict_proba(Xte)[:, 1]
    if arm == "R":
        y, _, _ = clipped_return(r)
        m = lgb.LGBMRegressor(objective="huber", **p)
        m.fit(Xtr, y)
        return m.predict(Xte)
    if arm == "Q":
        y = within_date_pct(tr["Date"], r)
        m = lgb.LGBMRegressor(objective="regression", **p)
        m.fit(Xtr, y)
        return m.predict(Xte)
    if arm == "L":
        # (Date, Code) の順に固定する。日付の中の並びが入力しだいだと、行の抽出の乱数が
        # 変わって結果が再現しない（学習データと同じ canonical_order）
        d = tr.sort_values(["Date", "Code"], kind="mergesort")
        y = within_date_grade(d["Date"], d[OUTCOME].to_numpy(dtype=float))
        sizes = d.groupby("Date", sort=True).size().to_numpy()
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(N_GRADES)), **p)
        m.fit(d[cols].to_numpy(dtype=float), y, group=sizes)
        return m.predict(Xte)
    raise ValueError(arm)


def oof_arm(df: pd.DataFrame, cols: List[str], arm: str, shift: int, seeds,
            tag: str = "", params: Dict | None = None, name: str | None = None,
            prefix: str = "e52") -> pd.DataFrame:
    """
    腕・切り方ごとの out-of-fold（種の平均）。保存済みなら読む。tag は列の違い。
    params / name は実験53（腕ごとに探索した木の形で同じ腕を回す。保存名は name）。
    """
    name = name or arm
    folds = E41.folds_for(df["Date"], shift)
    d = pd.to_datetime(df["Date"])
    keep = ["Code", "Date", "label", OUTCOME, "ret_o1_40", "vol_20d"]
    parts = []
    for sd in seeds:
        path = os.path.join(OOF_DIR, f"{prefix}{tag}_{name}_sh{shift}_s{sd}.parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        rows = []
        for f in folds:
            tr = df[(d <= pd.Timestamp(f.train_end)) & df["label"].notna() & df[OUTCOME].notna()]
            te = df[(d >= pd.Timestamp(f.test_start)) & (d <= pd.Timestamp(f.test_end))
                    & df["label"].notna()]
            if len(te) < 200 or len(tr) < 1000:
                continue
            part = te[keep].copy()
            part["score"] = fit_predict(arm, tr, te, cols, sd, params)
            part["fold"] = f.index
            rows.append(part)
        o = pd.concat(rows, ignore_index=True)
        o.to_parquet(path, index=False)
        parts.append(o)
        log(f"  腕{name} ずらし{shift}か月 種{sd}: {len(o):,}件 {time.time()-t0:.0f}秒")
    o = parts[0].copy()
    o["score"] = np.mean([x["score"].to_numpy(dtype=float) for x in parts], axis=0)
    o["Date"] = pd.to_datetime(o["Date"])
    return o


# --------------------------------------------------------------------------- #
# 評価
# --------------------------------------------------------------------------- #

def within_date_metrics(o: pd.DataFrame) -> Dict:
    """その日の1位・上位2件の平均収益（発火 MIN_BREAKS 件以上の日）と、日付内の順位相関。"""
    from scipy.stats import spearmanr

    top1, top2, rho = [], [], []
    for _, g in o.groupby("Date"):
        if len(g) < MIN_BREAKS:
            continue
        s = g.sort_values("score", ascending=False)
        r = pd.to_numeric(s[OUTCOME], errors="coerce").to_numpy()
        top1.append(r[0])
        top2.append(np.nanmean(r[:2]))
        if len(g) >= 5 and np.isfinite(r).sum() >= 5:
            rho.append(spearmanr(s["score"].to_numpy(), r, nan_policy="omit").correlation)
    return {"top1": float(np.nanmean(top1)) * 100 if top1 else np.nan,
            "top2": float(np.nanmean(top2)) * 100 if top2 else np.nan,
            "rho": float(np.nanmean(rho)) if rho else np.nan, "days": len(top1)}


def label_metrics(o: pd.DataFrame) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = o["label"].to_numpy(dtype=int)
    return {"pr": average_precision_score(y, o["score"]), "roc": roc_auc_score(y, o["score"]),
            "day_auc": lab.auc_in_day(o)}


def picks_by_vol(o: pd.DataFrame, pct: float) -> pd.DataFrame:
    """上位 (100−pct)% に選ばれた件のボラの帯の構成と帯ごとの平均収益（しきい値は前の窓から）。"""
    folds = sorted(o["fold"].unique())
    sel = []
    for f in folds:
        ref = o.loc[o["fold"] < f, "score"].to_numpy()
        if len(ref) < 500:
            continue
        cur = o[o["fold"] == f]
        sel.append(cur[cur["score"] > np.percentile(ref, pct)])
    if not sel:
        return pd.DataFrame()
    s = pd.concat(sel)
    band = pd.cut(s["vol_20d"], VOL_EDGES, labels=VOL_NAMES, include_lowest=True)
    t = s.groupby(band, observed=False).agg(n=(OUTCOME, "size"), ret=(OUTCOME, "mean"),
                                             pos=("label", "mean"))
    t["share"] = t["n"] / t["n"].sum()
    return t


#: ボラの帯（実験50 の5等分の境界。全期間の分位。表示用なので固定でよい）
VOL_EDGES = [0.0, 1.31, 1.77, 2.42, 3.73, 1e9]
VOL_NAMES = ["〜1.31", "〜1.77", "〜2.42", "〜3.73", "3.73〜"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験52: 目的を確率から収益に変える")
    ap.add_argument("--shifts", default="0,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--preset", default=F.DEFAULT_PRESET,
                    help="列のプリセット。既定は本番の206列。実験51 の6列を足すなら all_plus_prog_listing_vol")
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = E27.SEEDS3[:args.seeds]
    arms = [a for a in args.arms.split(",") if a]

    cols = F.columns(args.preset)
    # 本番の列以外は、保存する名前に列の指紋を入れて混ざらないようにする
    tag = "" if args.preset == F.DEFAULT_PRESET else f"_{F.signature(cols)}"
    df = lab.frame()
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"データセットに無い列: {miss[:6]}。research/build_dataset.py を回し直してください")
    df = df[df["label"].notna()].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    os.makedirs(OOF_DIR, exist_ok=True)
    print("=" * 78)
    print(f"実験52 目的を確率から収益に変える（{len(df):,}件 / {args.preset} {len(cols)}列"
          f"（指紋 {F.signature(cols)}） / "
          f"種{len(seeds)}つ / ずらし {shifts}か月）")
    print(f"  目的の収益: {OUTCOME}（翌営業日の寄りで買い、20営業日後の終値で売る）。"
          f"回帰は訓練側の {CLIP_Q[0]*100:.0f}〜{CLIP_Q[1]*100:.0f}% で刈り込み / LTR の段階は日付内の{N_GRADES}分位")
    p = tree_params(0)
    print("  木の形（本番の探索結果）: " + " / ".join(
        f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
        for k, v in p.items() if k in ("n_estimators", "learning_rate", "num_leaves",
                                       "min_child_samples", "subsample", "colsample_bytree",
                                       "reg_alpha", "reg_lambda")))
    print("=" * 78)

    rows, edges = [], []
    for sh in shifts:
        res = {arm: oof_arm(df, cols, arm, sh, seeds, tag) for arm in arms}
        print(f"\n■ ずらし{sh}か月")
        for pct in (95, 90):
            print(f"  ◆ 上位{100-pct}%（スコアが前の窓の分布の上位{100-pct}%なら買う）: "
                  f"件数 / 平均 / 全体との差 / 窓ごとの差 ± SE（上の窓 / 最悪） / −10%未満")
            for arm in arms:
                o = res[arm]
                e = lab.threshold_edge(o, pct=pct, outcome=OUTCOME)
                # −10% 未満の割合
                folds = sorted(o["fold"].unique())
                sel = []
                for f in folds:
                    ref = o.loc[o["fold"] < f, "score"].to_numpy()
                    if len(ref) >= 500:
                        cur = o[o["fold"] == f]
                        sel.append(cur[cur["score"] > np.percentile(ref, pct)])
                s = pd.concat(sel) if sel else o.iloc[:0]
                bad = float((pd.to_numeric(s[OUTCOME], errors="coerce") < -0.10).mean()) if len(s) else np.nan
                se = e["thr_fold_sd"] / np.sqrt(max(1, e["thr_folds"]))
                print(f"    {LABELS[arm]:<24}{e['thr_n']:>6}件 {e['thr_end']:>+7.2f}% "
                      f"{e['thr_lift']:>+7.2f}pt  {e['thr_fold_mean']:>+6.2f} ± {se:.2f}"
                      f"（{e['thr_folds_won']}/{e['thr_folds']} / {e['thr_worst']:+.2f}） "
                      f"{bad*100:>5.1f}%")
                edges.append({"shift": sh, "arm": arm, "pct": pct, "bad": bad, **e})
        print(f"  ◆ 日付内（発火{MIN_BREAKS}件以上の日）: 1位の平均収益 / 上位2件の平均 / 順位相関 / 日数"
              " ｜ 参考 PR-AUC / ROC-AUC / 日内AUC")
        for arm in arms:
            w = within_date_metrics(res[arm])
            m = label_metrics(res[arm])
            print(f"    {LABELS[arm]:<24}{w['top1']:>+7.2f}% {w['top2']:>+7.2f}% {w['rho']:>+6.3f} "
                  f"{w['days']:>5}日 ｜ {m['pr']:.4f} {m['roc']:.4f} {m['day_auc']:.4f}")
            rows.append({"shift": sh, "arm": arm, **w, **m})
        print(f"  ◆ 上位10% の中身（ボラの帯ごと: 割合 / 平均収益 / 正例率）")
        for arm in arms:
            t = picks_by_vol(res[arm], 90)
            if len(t):
                print(f"    {LABELS[arm]:<24}" + " | ".join(
                    f"{i} {r['share']*100:.0f}% {r['ret']*100:+.1f}% {r['pos']*100:.0f}%"
                    for i, r in t.iterrows()))
        # 件数をそろえた比べ: 窓の中の上位10%（スコアの分布が腕で違っても、選ぶ件数は同じ）
        print("  ◆ 窓の中の上位10%（件数をそろえる）: 平均収益 / −10%未満 / 正例率 ｜ C との差（窓ごと、pt）")
        base = _per_fold(res["C"], 90, within=True) if "C" in arms else None
        for arm in arms:
            cur = _per_fold(res[arm], 90, within=True)
            line = (f"    {LABELS[arm]:<24}{cur['ret'].mean()*100:>+6.2f}% {cur['bad'].mean()*100:>5.1f}% "
                    f"{cur['pos'].mean()*100:>4.0f}%")
            if base is not None and arm != "C":
                m = base.merge(cur, on="fold", suffixes=("_c", "_a"))
                d = (m["ret_a"] - m["ret_c"]).to_numpy() * 100
                line += (f" ｜ {d.mean():+.2f} ± {d.std(ddof=1)/np.sqrt(len(d)):.2f}pt"
                         f"（上の窓 {(d > 0).sum()}/{len(d)}）: " + " ".join(f"{x:+.1f}" for x in d))
            print(line)
            edges.append({"shift": sh, "arm": arm, "pct": "top10_within", "bad": cur["bad"].mean(),
                          "thr_lift": np.nan, "thr_fold_mean": cur["ret"].mean() * 100,
                          "thr_n": int(cur["n"].sum())})

    pd.DataFrame(rows).to_csv(os.path.join(OOF_DIR, f"e52{tag}_within_date.csv"), index=False)
    ed = pd.DataFrame(edges)
    ed.to_csv(os.path.join(OOF_DIR, f"e52{tag}_edges.csv"), index=False)
    if len(shifts) > 1 and len(ed):
        print(f"\n■ 切り方{len(shifts)}通りをまとめて")
        for pct in (95, 90):
            print(f"  上位{100-pct}%: 全体との差（切り方ごと）/ 窓ごとの差の平均 / −10%未満")
            for arm in arms:
                g = ed[(ed["arm"] == arm) & (ed["pct"].astype(str) == str(pct))]
                print(f"    {LABELS[arm]:<24}" + " / ".join(f"{v:+.2f}" for v in g["thr_lift"])
                      + f" pt（平均 {g['thr_lift'].mean():+.2f}）/ {g['thr_fold_mean'].mean():+.2f}pt / "
                      f"{g['bad'].mean()*100:.1f}%")
        print("  窓の中の上位10%（件数をそろえる）の平均収益（切り方の平均）: " + " / ".join(
            f"{LABELS[a]} {ed[(ed['arm'] == a) & (ed['pct'] == 'top10_within')]['thr_fold_mean'].mean():+.2f}%"
            for a in arms))
        r = pd.DataFrame(rows)
        print("  日付内の1位の平均収益（切り方の平均）: " + " / ".join(
            f"{LABELS[a]} {r[r['arm'] == a]['top1'].mean():+.2f}%" for a in arms))
    log(f"記録: {OOF_DIR}/e52{tag}_*")
    return 0


def _per_fold(o: pd.DataFrame, pct: float, within: bool = False) -> pd.DataFrame:
    """
    窓ごとの上位 (100−pct)% の平均収益。within=False はしきい値を前の窓から取る（運用の形。
    窓1は対象外）、True は窓の中の分位で選ぶ（件数をそろえた比べ。窓1も入る）。
    """
    rows = []
    for f in sorted(o["fold"].unique()):
        cur = o[o["fold"] == f]
        if within:
            thr = np.percentile(cur["score"].to_numpy(), pct)
        else:
            ref = o.loc[o["fold"] < f, "score"].to_numpy()
            if len(ref) < 500:
                continue
            thr = np.percentile(ref, pct)
        sel = cur[cur["score"] > thr]
        if len(sel):
            r = pd.to_numeric(sel[OUTCOME], errors="coerce")
            rows.append({"fold": int(f), "n": int(len(sel)), "ret": float(r.mean()),
                         "bad": float((r < -0.10).mean()), "pos": float(sel["label"].mean())})
    return pd.DataFrame(rows, columns=["fold", "n", "ret", "bad", "pos"])


if __name__ == "__main__":
    raise SystemExit(main())
