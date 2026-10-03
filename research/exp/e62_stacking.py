#!/usr/bin/env python3
"""
実験62: 本番5モデル（LightGBM / XGBoost / CatBoost / ロジスティック回帰 / MLP）のスタッキング。

運用者の依頼（2026-10-03）「5個のモデルのスタッキングを実施してほしいです」。
これまでの運用は「5モデルのスコアを並べ、合議（全5モデル 95以上など）で選ぶ」で、スコアを混ぜる
アンサンブルはしていない（docs/MODEL_LINEUP.md）。ここでは 5モデルのスコアを入力にした2段目（meta）を
学習し、単体・順位平均・いまの合議の規則と、本番と同じ OOF と 32窓で比べる。

先読みを入れない作り（実験61 §23 の教訓）
  1段目: 5モデルそれぞれの out-of-fold（本番と同じ 36/6/6か月・エンバーゴ20営業日の窓。切り方3通り × 種3つ）。
         パラメータは本番のもの（lgbm_params.json / multi_params.json）、前処理も本番と同じ
  2段目: 窓 k を採点する meta は、**窓 k の訓練終了日（train_end）以前の行の OOF スコアとラベルだけ**で学習する
         （それより前の窓の OOF。窓 k−1 の末尾のエンバーゴ分も除く）。本番でやるなら「毎週、保存してある
         5モデルの OOF で meta を学習し、その日の5モデルのスコアに当てる」なので、同じ形になる
  meta の入力は 5モデルのスコアだけ（元の特徴量は使わない）。入力の作り方は3通り
    logit   log(p / (1−p))（スケールは meta の学習行で標準化）
    日内百分位  同じ日の候補の中での百分位（画面でも出せる）
    過去百分位  その日より前の OOF 分布での百分位（画面の「位置%」と同じ。実験59 の pct_expanding。
                過去の行が PCT_MIN_HIST 未満の日は欠損で、その行は meta の学習・採点から外れる）
  meta の学習器は ロジスティック回帰（L2）と、小さな LightGBM（葉4・200本。交互作用を見るため）

比べるもの
  単体5つ / 順位平均（日内百分位の平均。学習なし）/ LR×logit / LR×日内百分位 / LR×過去百分位 /
  LGBM×logit / LR×logit（木3つだけ。線形・MLP を足す価値を見る）
指標
  §1 本番と同じ OOF（ずらし0・種3つの平均）: PR-AUC・リフト・ROC・日内 AUC・上位10% の超過・勝窓・最悪
  §2 32窓（切り方3通り × 窓。種3つの平均のスコア）: 方法 − LightGBM、方法 − 順位平均（平均・SE・上回った窓）
  §3 運用の規則: 「全5モデル 95以上・1日1件」と「スタック 95以上・1日1件」、発火数をそろえた正例率・ret20
  §4 meta の中身: LR の標準化係数（どのモデルをどれだけ重く見るか）、LGBM meta の寄与
meta が学習できる窓（学習行 MIN_META 以上）だけで比べる。単体も同じ窓で測る。
記録は research/_data/oof/e62_*。本番の設定には何も書かない。

結果（2026-10-03、run 37105947657、docs/MODEL_ADOPTION_RULES.md §24）: 学習した meta は窓ごとに LightGBM 単体を
PR +0.002〜0.003 上回る（17/26窓）が、本番と同じ OOF では下回り、「全5モデル 95以上」に発火数をそろえた差は
ノイズの範囲（正例率 +2.9pt / ret20 +0.9pt、SE 程度）→ 採用を勧めない。

使い方
    python3 research/exp/e62_stacking.py
    python3 research/exp/e62_stacking.py --shifts 0 --seeds 1      # 試運転
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import models as M  # noqa: E402
import tuning_multi as TM  # noqa: E402
import ab_oof as AB  # noqa: E402
import e27_timing_multi as E27  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import e59_unit_sim as E59  # noqa: E402
from e25_auc_noise import metrics  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGOS = tuple(L.ALL5)
BOOST = tuple(L.BOOST)
SHIFTS = (0, 2, 4)
SEEDS = E27.SEEDS3
MIN_META = 1000          # meta を学習するのに要る行数。これ未満の窓は（単体も）比べない
PCT_MIN_HIST = 250       # 過去百分位の入力に要る過去の行数（約 20営業日ぶん。規則の判定は実験59 と同じ 500）
RULE_PCT = 95.0          # 運用の規則「95以上」
EPS = 1e-6
KEEP = ["Code", "Date", "fold", "label", "ret_o1_20", "ret_o1_40"]

#: (鍵, 表示名, 入力, 学習器)。学習器 None は学習なし
METHODS: Tuple[Tuple[str, str, str, str], ...] = (
    ("rank_avg", "順位平均", "rank", ""),
    ("lr_logit", "LR×logit", "logit", "lr"),
    ("lr_rank", "LR×日内百分位", "rank", "lr"),
    ("lr_pct", "LR×過去百分位", "pct", "lr"),
    ("lgbm_logit", "LGBM×logit", "logit", "lgbm"),
    ("lr_logit3", "LR×logit（木3つ）", "logit3", "lr"),
)
NAME = {k: n for k, n, _, _ in METHODS}
NAME.update({a: M.JA.get(a, a) for a in ALGOS})
META_LGBM = dict(n_estimators=200, learning_rate=0.03, num_leaves=4, min_child_samples=200,
                 subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
                 n_jobs=-1, verbose=-1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- #
# 1段目: 5モデルの OOF（本番のパラメータ・前処理・窓）
# ---------------------------------------------------------------- #

def base_oof(algo: str, df: pd.DataFrame, cols: list, shift: int, seed: int, folds) -> pd.DataFrame:
    path = os.path.join(OOF_DIR, f"e62_base_{algo}_sh{shift}_s{seed}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    params = E27.prod_params(algo)["params"]
    o = E41.oof_folds(algo, df, cols, params, seed, folds)
    o["Date"] = pd.to_datetime(o["Date"])
    o["Code"] = o["Code"].astype(str)
    o.to_parquet(path, index=False)
    log(f"    {algo} ずらし{shift}か月 種{seed}: {len(o):,}件 / {time.time() - t0:.0f}秒")
    return o


def merge_base(oofs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """5モデルの OOF を (Code, Date, fold) で内部結合し s_<algo> を並べる。行の組は全モデルで同じはず。"""
    base = None
    for a, o in oofs.items():
        keep = [c for c in KEEP if c in o.columns]
        part = o[keep + ["score"]].rename(columns={"score": f"s_{a}"})
        if base is None:
            base = part
            n0 = len(part)
        else:
            base = base.merge(part[["Code", "Date", "fold", f"s_{a}"]], on=["Code", "Date", "fold"], how="inner")
            if len(base) != n0:
                raise SystemExit(f"{a} の OOF の行が他と違う（{len(base):,} vs {n0:,}）")
    base["Date"] = pd.to_datetime(base["Date"])
    return base.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- #
# meta の入力
# ---------------------------------------------------------------- #

def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def within_date_pct(frame: pd.DataFrame, col: str) -> np.ndarray:
    """同じ日の候補の中での百分位（(順位 − 0.5) / 件数。0〜1、日の平均は 0.5）。"""
    r = frame.groupby("Date")[col].rank(method="average")
    n = frame.groupby("Date")[col].transform("size")
    return ((r - 0.5) / n).to_numpy(dtype=float)


def features(frame: pd.DataFrame, kind: str) -> Tuple[np.ndarray, List[str]]:
    """kind ごとの入力行列（行 × モデル）と列名。どれも「その行までに分かる値」だけで作る。"""
    algos = BOOST if kind == "logit3" else ALGOS
    if kind in ("logit", "logit3"):
        X = np.column_stack([logit(frame[f"s_{a}"].to_numpy(dtype=float)) for a in algos])
    elif kind == "rank":
        X = np.column_stack([within_date_pct(frame, f"s_{a}") for a in algos])
    elif kind == "pct":
        X = np.column_stack([E59.pct_expanding(frame["Date"], frame[f"s_{a}"].to_numpy(dtype=float),
                                               PCT_MIN_HIST) / 100.0 for a in algos])
    else:
        raise ValueError(kind)
    return X, list(algos)


# ---------------------------------------------------------------- #
# 2段目: 時間順の meta（窓 k は train_end_k 以前の OOF だけで学習）
# ---------------------------------------------------------------- #

def fit_meta(learner: str, X: np.ndarray, y: np.ndarray, seed: int = 0):
    if learner == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    elif learner == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**META_LGBM, random_state=seed)
    else:
        raise ValueError(learner)
    m.fit(X, y)
    return m


def meta_weights(learner: str, model, names: List[str]) -> Dict[str, float]:
    """LR は標準化した入力への係数、LGBM は gain の割合。"""
    if learner == "lr":
        coef = model[-1].coef_.ravel()
        return {n: float(c) for n, c in zip(names, coef)}
    g = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    g = g / g.sum() if g.sum() > 0 else g
    return {n: float(c) for n, c in zip(names, g)}


def train_rows(frame: pd.DataFrame, k: int, train_end: pd.Timestamp) -> np.ndarray:
    """窓 k の meta の学習に使ってよい行: 前の窓の行で、日付が窓 k の訓練終了日以前のもの。"""
    return ((frame["fold"].to_numpy() < k) & (frame["Date"].to_numpy() <= np.datetime64(train_end))
            & frame["label"].notna().to_numpy())


def stack_scores(frame: pd.DataFrame, X: np.ndarray, names: List[str], learner: str,
                 train_end: Dict[int, pd.Timestamp], *, min_train: int = MIN_META, seed: int = 0):
    """
    窓ごとに時間順で meta を学習して採点する。
    戻り値: (スコア。学習できない窓は NaN, 同じ meta を学習行に当てた分布での百分位（0〜100）,
             {窓: 学習行数}, {窓: 重み})
    百分位は「本番で meta を毎週学習し直し、その meta の OOF（学習行）の分布でその日のスコアの位置を出す」形。
    """
    y_all = frame["label"].to_numpy(dtype=float)
    f = frame["fold"].to_numpy()
    out = np.full(len(frame), np.nan)
    pct_same = np.full(len(frame), np.nan)
    n_train: Dict[int, int] = {}
    weights: Dict[int, Dict[str, float]] = {}
    ok_row = np.isfinite(X).all(axis=1)
    for k in sorted(set(int(v) for v in f)):
        tr = train_rows(frame, k, train_end[k]) & ok_row
        n = int(tr.sum())
        n_train[k] = n
        if n < min_train or y_all[tr].min() == y_all[tr].max():
            continue
        m = fit_meta(learner, X[tr], y_all[tr].astype(int), seed)
        te = (f == k) & ok_row
        out[te] = np.asarray(m.predict_proba(X[te]), dtype=float)[:, 1]
        hist = np.asarray(m.predict_proba(X[tr]), dtype=float)[:, 1]
        pct_same[te] = L.pct_of(out[te], hist)
        weights[k] = meta_weights(learner, m, names)
    return out, pct_same, n_train, weights


def rank_average(frame: pd.DataFrame) -> np.ndarray:
    X, _ = features(frame, "rank")
    return X.mean(axis=1)


def add_stacks(frame: pd.DataFrame, train_end: Dict[int, pd.Timestamp], *, seed: int = 0,
               min_train: int = MIN_META):
    """frame に m_<鍵> を足す。戻り値: (frame, {鍵: {窓: 学習行数}}, {鍵: {窓: 重み}})"""
    frame = frame.copy()
    n_train: Dict[str, Dict[int, int]] = {}
    weights: Dict[str, Dict[int, Dict[str, float]]] = {}
    cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for key, _, kind, learner in METHODS:
        if not learner:
            frame[f"m_{key}"] = rank_average(frame)
            continue
        if kind not in cache:
            cache[kind] = features(frame, kind)
        X, names = cache[kind]
        sc, q, nt, w = stack_scores(frame, X, names, learner, train_end, min_train=min_train, seed=seed)
        frame[f"m_{key}"] = sc
        frame[f"q_{key}"] = q                          # 同じ meta の学習行の分布での百分位
        n_train[key] = nt
        weights[key] = w
    return frame, n_train, weights


def evaluated_folds(n_train: Dict[str, Dict[int, int]], min_train: int = MIN_META) -> List[int]:
    """全部の学習する方法で meta が学習できた窓（学習行は方法によらず同じなので、実質は行数の条件）。"""
    keys = list(n_train)
    folds = sorted(set(k for d in n_train.values() for k in d))
    return [k for k in folds if all(n_train[m].get(k, 0) >= min_train for m in keys)]


def average_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """種ごとの frame（行は同じ）のスコア列を平均する。"""
    base = frames[0].copy()
    for fr in frames[1:]:
        if not (fr["Code"].to_numpy() == base["Code"].to_numpy()).all() or \
           not (fr["Date"].to_numpy() == base["Date"].to_numpy()).all():
            raise SystemExit("種ごとの OOF の行が揃っていません")
    for c in [c for c in base.columns if c[:2] in ("s_", "m_", "q_")]:
        base[c] = np.mean([fr[c].to_numpy(dtype=float) for fr in frames], axis=0)
    return base


# ---------------------------------------------------------------- #
# 指標
# ---------------------------------------------------------------- #

def score_frame(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    o = frame[[c for c in KEEP if c in frame.columns]].copy()
    o["score"] = frame[col].to_numpy(dtype=float)
    return o[np.isfinite(o["score"])]


def columns_to_compare() -> List[str]:
    return [f"s_{a}" for a in ALGOS] + [f"m_{k}" for k, _, _, _ in METHODS]


def label_of(col: str) -> str:
    return NAME[col[2:]]


def pct_cols(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """運用の規則用: 列ごとに過去分布の百分位 p_<鍵> を足す（実験59 と同じ pct_expanding）。"""
    out = frame.copy()
    for c in cols:
        out[f"p_{c[2:]}"] = E59.pct_expanding(out["Date"], out[c].to_numpy(dtype=float))
        if f"q_{c[2:]}" in out.columns:
            out[f"p_{c[2:]}_same"] = out[f"q_{c[2:]}"].to_numpy(dtype=float)
    out["score"] = out["s_lgbm"]
    return out


def rule_picks(base: pd.DataFrame, models: Sequence[str], pct: float = RULE_PCT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """「models の百分位がすべて pct 以上」を満たした行と、1日1件の買い（live_track.decide）。"""
    r, _, picks = L.decide(base, agree=pct, min_break=0, top_k=1, models=models)
    return r[r["passed"]], picks


def rule_stats(rows: pd.DataFrame) -> dict:
    if len(rows) == 0:
        return {"n": 0, "pos": float("nan"), "ret20": float("nan"), "med20": float("nan")}
    return {"n": int(len(rows)), "pos": float(rows["label"].mean() * 100),
            "ret20": float(rows["ret_o1_20"].mean() * 100), "med20": float(rows["ret_o1_20"].median() * 100)}


def matched(base: pd.DataFrame, pcol: str, n_by_fold: Dict[int, int]) -> pd.DataFrame:
    """窓ごとに、基準の規則と同じ件数だけ pcol の高い順に取る（同点は LightGBM のスコア順）。"""
    parts = []
    for k, n in n_by_fold.items():
        g = base[base["fold"] == k].sort_values([pcol, "score"], ascending=[False, False])
        parts.append(g.head(int(n)))
    return pd.concat(parts) if parts else base.iloc[:0]


def matched_picks(picks: pd.DataFrame, pcol: str, n_by_fold: Dict[int, int]) -> pd.DataFrame:
    """1日1件の買いの中から、窓ごとに基準と同じ件数だけ pcol の高い順に取る。"""
    parts = []
    for k, n in n_by_fold.items():
        g = picks[picks["fold"] == k].sort_values([pcol, "score"], ascending=[False, False])
        parts.append(g.head(int(n)))
    return pd.concat(parts) if parts else picks.iloc[:0]


def fmt_rule(name: str, s: dict, ref: dict | None = None) -> str:
    d = "" if ref is None or not np.isfinite(s["pos"]) else f"  正例率 {s['pos'] - ref['pos']:+.1f}pt / ret20 {s['ret20'] - ref['ret20']:+.2f}pt"
    return (f"  {L._l(name, 34)}{s['n']:>6}件  正例率 {s['pos']:>5.1f}%  ret20 平均 {s['ret20']:>+6.2f}%  "
            f"中央値 {s['med20']:>+6.2f}%{d}")


def se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


# ---------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験62: 5モデルのスタッキング（時間順の meta）")
    ap.add_argument("--shifts", default=",".join(str(s) for s in SHIFTS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--min-meta", type=int, default=MIN_META)
    args = ap.parse_args(argv)
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    seeds = SEEDS[:args.seeds]
    os.makedirs(OOF_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    cols = F.columns(F.DEFAULT_PRESET)
    print("=" * 78)
    print("実験62 5モデルのスタッキング（1段目 = 本番と同じ OOF、2段目 = 窓より前の OOF だけで学習する meta）")
    print(f"  列: {F.DEFAULT_PRESET} {len(cols)}列 / ずらし {shifts}か月 / 種 {seeds} / meta の最少学習行 {args.min_meta:,}")
    print("  前処理: " + " / ".join(f"{a} {TM.preprocess_version(a) or '木'}" for a in ALGOS))
    for key, name, kind, learner in METHODS:
        print(f"  {name}: 入力 {kind} / 学習器 {learner or 'なし'}")
    print("=" * 78)

    df = lab.frame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log(f"データ {len(df):,}件 / 正例率 {df['label'].mean() * 100:.2f}% / "
        f"{df['Date'].min().date()} 〜 {df['Date'].max().date()}")

    per_shift: Dict[int, pd.DataFrame] = {}
    per_shift_seed: Dict[Tuple[int, int], pd.DataFrame] = {}
    folds_ok: Dict[int, List[int]] = {}
    all_weights: Dict[str, List[Dict[str, float]]] = {}
    summary: dict = {"shifts": shifts, "seeds": list(seeds), "min_meta": args.min_meta, "folds": {}}

    print("\n■ 0. 1段目の OOF と、meta の学習行（窓ごと。時間順）")
    for sh in shifts:
        folds = E41.folds_for(df["Date"], sh)
        train_end = {f.index: pd.Timestamp(f.train_end) for f in folds}
        frames = []
        for sd in seeds:
            oofs = {a: base_oof(a, df, cols, sh, sd, folds) for a in ALGOS}
            frame = merge_base(oofs)
            frame, n_train, weights = add_stacks(frame, train_end, seed=sd, min_train=args.min_meta)
            frame.to_parquet(os.path.join(OOF_DIR, f"e62_stack_sh{sh}_s{sd}.parquet"), index=False)
            frames.append(frame)
            per_shift_seed[(sh, sd)] = frame
            if sd == seeds[0]:
                folds_ok[sh] = evaluated_folds(n_train, args.min_meta)
                print(f"  ずらし{sh}か月: {len(frame):,}件 / 窓 {len(train_end)} / meta を比べる窓 {folds_ok[sh]}")
                for k in sorted(train_end):
                    nt = n_train["lr_logit"].get(k, 0)
                    nte = int((frame["fold"] == k).sum())
                    print(f"    窓{k:>2}: 訓練終了 {train_end[k].date()} / meta の学習行 {nt:>6,} / 採点 {nte:>5,}"
                          + ("" if nt >= args.min_meta else "  （学習行が足りないので比べない）"))
                summary["folds"][str(sh)] = {"evaluated": folds_ok[sh],
                                             "n_train": {str(k): v for k, v in n_train["lr_logit"].items()}}
            for key, w in weights.items():
                for k in folds_ok[sh]:
                    if k in w:
                        all_weights.setdefault(key, []).append(w[k])
        per_shift[sh] = average_frames(frames)

    # ---------------- §1 本番と同じ OOF ---------------- #
    cmp_cols = columns_to_compare()
    if 0 in per_shift:
        fr = per_shift[0]
        ev = fr[fr["fold"].isin(folds_ok[0])]
        print(f"\n■ 1. 本番と同じ作りの out-of-fold（ずらし0・種{seeds} の平均・meta のある窓 {folds_ok[0]}・{len(ev):,}件）")
        print(f"  {L._l('方法', 22)}{'PR-AUC':>9}{'リフト':>7}{'ROC':>9}{'日内':>8}{'上位10%超過':>12}{'勝窓':>7}{'最悪':>9}")
        summary["production_oof"] = {}
        base_m = None
        for c in cmp_cols:
            o = score_frame(ev, c)
            m = metrics(o)
            summary["production_oof"][c[2:]] = {k: float(v) for k, v in m.items()}
            if c == "s_lgbm":
                base_m = m
            print(f"  {L._l(label_of(c), 22)}{m['pr_auc']:>9.4f}{m['lift']:>6.2f}x{m['roc_auc']:>9.4f}"
                  f"{m['day_auc']:>8.4f}{m['ret_o1_20_mean']:>+11.2f}pt{m['ret_o1_20_won']:>4}/{m['ret_o1_20_n']:<3}"
                  f"{m['ret_o1_20_worst']:>+8.2f}pt")
        print(f"  {L._l('方法 − LightGBM', 22)}{'PR':>9}{'':>7}{'ROC':>9}{'日内':>8}")
        for c in cmp_cols:
            if c == "s_lgbm":
                continue
            m = summary["production_oof"][c[2:]]
            print(f"  {L._l(label_of(c), 22)}{m['pr_auc'] - base_m['pr_auc']:>+9.4f}{'':>7}"
                  f"{m['roc_auc'] - base_m['roc_auc']:>+9.4f}{m['day_auc'] - base_m['day_auc']:>+8.4f}")

    # ---------------- §2 窓ごと ---------------- #
    print("\n■ 2. 窓ごと（切り方3通り。種の平均のスコア。meta のある窓だけ）")
    rows = []
    for sh in shifts:
        fr = per_shift[sh]
        ev = fr[fr["fold"].isin(folds_ok[sh])]
        per_col = {c: AB.auc_by_window(score_frame(ev, c)).set_index("fold") for c in cmp_cols}
        for k in folds_ok[sh]:
            row = {"shift": sh, "fold": k}
            for c in cmp_cols:
                if k in per_col[c].index:
                    row[f"pr_{c[2:]}"] = float(per_col[c].loc[k, "pr"])
                    row[f"roc_{c[2:]}"] = float(per_col[c].loc[k, "roc"])
            rows.append(row)
    win = pd.DataFrame(rows)
    win.to_csv(os.path.join(OOF_DIR, "e62_windows.csv"), index=False)
    summary["windows"] = {}
    for ref in ("lgbm", "rank_avg"):
        print(f"  方法 − {NAME[ref]}（{len(win)}窓）")
        print(f"  {L._l('', 22)}{'PR の差':>9}{'SE':>8}{'上':>8}{'ROC の差':>11}{'SE':>8}{'上':>8}")
        for c in cmp_cols:
            key = c[2:]
            if key == ref:
                continue
            dpr = (win[f"pr_{key}"] - win[f"pr_{ref}"]).to_numpy(dtype=float)
            droc = (win[f"roc_{key}"] - win[f"roc_{ref}"]).to_numpy(dtype=float)
            dpr, droc = dpr[np.isfinite(dpr)], droc[np.isfinite(droc)]
            summary["windows"][f"{key}-{ref}"] = {"n": int(len(dpr)), "pr_mean": float(dpr.mean()), "pr_se": se(dpr),
                                                  "pr_up": int((dpr > 0).sum()), "roc_mean": float(droc.mean()),
                                                  "roc_se": se(droc), "roc_up": int((droc > 0).sum())}
            print(f"  {L._l(NAME[key], 22)}{dpr.mean():>+9.4f}{se(dpr):>8.4f}{(dpr > 0).sum():>5}/{len(dpr):<3}"
                  f"{droc.mean():>+11.4f}{se(droc):>8.4f}{(droc > 0).sum():>5}/{len(droc):<3}")
    # 切り方ごとの内訳（LightGBM との差）
    print("  切り方ごと（方法 − LightGBM の PR / ROC の平均と上回った窓）")
    for c in cmp_cols:
        key = c[2:]
        if key == "lgbm":
            continue
        parts = []
        for sh in shifts:
            w = win[win["shift"] == sh]
            dpr = (w[f"pr_{key}"] - w["pr_lgbm"]).to_numpy(dtype=float)
            droc = (w[f"roc_{key}"] - w["roc_lgbm"]).to_numpy(dtype=float)
            parts.append(f"ずらし{sh} PR {dpr.mean():+.4f} {(dpr > 0).sum()}/{len(dpr)}・ROC {droc.mean():+.4f} {(droc > 0).sum()}/{len(droc)}")
        print(f"  {L._l(NAME[key], 22)}" + " / ".join(parts))

    # ---------------- §3 運用の規則 ---------------- #
    print(f"\n■ 3. 運用の規則（百分位 {RULE_PCT:.0f} 以上・1日1件。meta のある窓だけ。切り方3通りの行を合算）")
    rule_rows = []
    pooled: Dict[str, List[pd.DataFrame]] = {}
    for sh in shifts:
        fr = per_shift[sh]
        base = pct_cols(fr, cmp_cols)                     # 百分位の過去分布は meta の無い窓も含めて積む
        base = base[base["fold"].isin(folds_ok[sh])].reset_index(drop=True)
        rows_a, picks_a = rule_picks(base, ALGOS)
        n_rows = rows_a.groupby("fold").size().to_dict()
        n_picks = picks_a.groupby("fold").size().to_dict()
        sets = {"全5モデル 95以上": picks_a,
                "GBDT3 95以上": rule_picks(base, BOOST)[1],
                "LightGBM 95以上": rule_picks(base, ("lgbm",))[1]}
        for key, name, _, learner in METHODS:
            picks_m = rule_picks(base, (key,))[1]
            sets[f"{name}|past"] = picks_m
            if learner:
                sets[f"{name}|same"] = rule_picks(base, (f"{key}_same",))[1]
            sets[f"{name}|rows"] = matched(base, f"p_{key}", n_rows)
            sets[f"{name}|picks"] = matched_picks(picks_m, f"p_{key}", n_picks)
        for name, rows_ in sets.items():
            pooled.setdefault(name, []).append(rows_)
            rule_rows.append({"shift": sh, "rule": name, **rule_stats(rows_)})
    pd.DataFrame(rule_rows).to_csv(os.path.join(OOF_DIR, "e62_rules.csv"), index=False)
    stats = {name: rule_stats(pd.concat(parts)) for name, parts in pooled.items()}
    summary["rules"] = stats
    ref = stats["全5モデル 95以上"]

    def cell(name: str, with_n: bool = True) -> str:
        st = stats.get(name)
        if st is None or st["n"] == 0:
            return f"{'—':>22}" if with_n else f"{'—':>16}"
        body = f"{st['pos']:>5.1f}% {st['ret20']:>+6.2f}%"
        return (f"{st['n']:>6}件 " if with_n else "  ") + body
    print("  基準（1日1件）                       件数  正例率  ret20平均")
    for name in ("全5モデル 95以上", "GBDT3 95以上", "LightGBM 95以上"):
        st = stats[name]
        print(f"  {L._l(name, 34)}{st['n']:>6}件  {st['pos']:>5.1f}%  {st['ret20']:>+6.2f}%")
    print(f"  {L._l('方法', 22)}{'過去分布 95以上・1日1件':>24}{'同じmeta分布 95以上・1日1件':>26}"
          f"{'件数そろえ(行)':>16}{'件数そろえ(1日1件)':>18}")
    for key, name, _, learner in METHODS:
        print(f"  {L._l(name, 22)}{cell(f'{name}|past'):>24}{cell(f'{name}|same') if learner else '—':>26}"
              f"{cell(f'{name}|rows', False):>16}{cell(f'{name}|picks', False):>18}")
    print(f"  （基準の全5モデル 95以上: 正例率 {ref['pos']:.1f}% / ret20 {ref['ret20']:+.2f}%。"
          "「過去分布」はその日より前の OOF 分布での百分位、「同じmeta分布」はその窓の meta を学習行に当てた分布での百分位。"
          "「件数そろえ(行)」は窓ごとに全5モデル 95以上を満たした行数と同じ数を百分位の高い順に取る。"
          "「件数そろえ(1日1件)」は 1日1件の買いの数をそろえる。ret20 は翌営業日の寄りで買い 20営業日の収益 %）")

    # ---------------- §4 meta の中身 ---------------- #
    print("\n■ 4. meta の重み（meta のある窓・種の平均。LR は標準化した入力への係数、LGBM は gain の割合）")
    summary["weights"] = {}
    for key, name, kind, learner in METHODS:
        if not learner or key not in all_weights:
            continue
        w = pd.DataFrame(all_weights[key]).mean()
        summary["weights"][key] = {k: float(v) for k, v in w.items()}
        print(f"  {L._l(name, 22)}" + "  ".join(f"{M.SHORT.get(a, a)} {w[a]:+.3f}" for a in w.index))

    with open(os.path.join(OOF_DIR, "e62_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)
    log(f"記録: {OOF_DIR}/e62_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
