#!/usr/bin/env python3
"""
実験60: 本番5モデル（LightGBM / XGBoost / CatBoost / ロジスティック回帰 / MLP）の傾向。

運用者の依頼（2026-10-03）
  「この５個のモデルの、傾向を知りたいです。例えば lgbm は比較的 xxx の特徴量を過大評価する傾向がある、
   MLP は比較的 yyyy の特徴量を過大評価する傾向がある、のような知見がほしいです。必要であれば shap や
   クラスタリングなどを算出してもらっても良いです。」

何を出すか
  1. 頼り方（寄与）   各モデルが「どの列・どの区分にどれだけ頼っているか」。寄与は対数オッズの単位で揃える
       木（lgbm / xgb / cat）  TreeSHAP（pred_contrib / ShapValues。モデル固有の厳密な分解）
       ロジスティック回帰      係数 × 標準化後の値（線形モデルではこれが寄与そのもの）
       MLP                     積分勾配（基準 = 標準化後の 0 = 訓練の平均。32段。合計が出力の差に一致する）
     前処理後の列（one-hot・欠損指示子）は元の列に戻して足す。対応は前処理の出力名か、1列ずつ揺らして求める
  2. 共通の物差し     置換で順位がどれだけ崩れるか（区分ごと・列ごと）。区分の列をまとめて行の間で入れ替えて
     採点し直し、元のスコアとの順位相関の落ち（1 − ρ）を「その区分への依存」とする。ラベルを使わないので
     学習に使った行でも偏らず、5モデルを同じ物差しで比べられる
  3. 過大評価         頼っている割に、その列単独の分離力（ラベルに対する AUC）が弱い列。
     「他の4モデルより重く見ている区分」（区分の寄与の割合 − 5モデルの平均）も出す
  4. 一致と不一致     OOF（Release の oof.parquet / <algo>_oof.parquet）でスコアの順位相関、上位10% の重なり、
     階層クラスタリング。lgbm と意見が割れた行（片方だけ上位10%）で、どちらが当たったか
  5. 上位銘柄の性格   各モデルの上位10%（OOF）が、代表的な列で母集団のどこに偏るか（同じ日の中の百分位の平均）
  6. 型（クラスタ）   5モデルの上位10%（和集合）を代表的な列で k-means（k=5）に分け、型ごとの特徴と
     各モデルの偏り、OOF の正例率を出す

データ
  本番モデルは Release（data-raw）の model.txt / <algo>_model.joblib。meta.json（git）の列で揃える。
  寄与と置換は学習に使った行（dataset.parquet の訓練期間）で出す。OOF の分析は Release の OOF。
  ログに出すのは列名・区分名と割合・件数だけ（生データの値は出さない）。

出力  research/_data/oof/e60_reliance.csv（モデル × 列）、e60_family.csv（モデル × 大区分）、
      e60_group.csv（モデル × 中区分）、e60_agreement.csv、e60_profile.csv、e60_disagree.csv、
      e60_clusters.csv、e60_summary.json。本番の設定は何も変えない。

使い方
    python3 research/exp/e60_model_tendency.py                       # Actions（Release の配置）
    python3 research/exp/e60_model_tendency.py --release-dir /tmp/q --quick   # 手元の試運転
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_dict as FD  # noqa: E402
import features as F  # noqa: E402
import lab  # noqa: E402
import live_track as L  # noqa: E402
import models as M  # noqa: E402
import tuning_multi as TM  # noqa: E402

OOF_DIR = os.path.join(lab.DATA_DIR, "oof")
ALGOS = ("lgbm", "xgb", "cat", "logit", "mlp")
TREES = ("lgbm", "xgb", "cat")
IG_STEPS = 32
TOP_SHARE = 0.10          # 「上位」= OOF スコアの上位10%
N_REP = 24                # 性格・型に使う代表的な列の数
K_CLUSTERS = 5
WEAK_AUC = 0.02           # 単独の分離力が弱い = |AUC − 0.5| がこれ未満

#: 中区分 → 大区分（画面の寄与と同じ名前。research/feature_dict.MACRO）
FAMILY_OF = {g: ja for ja, groups, _ in FD.MACRO for g in groups}
FAMILY_OF["market"] = "地合い（市場環境）"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ja(col: str, width: int = 18) -> str:
    """列名と日本語名の頭。"""
    d = FD.describe(col)
    d = d.split("。")[0].split("（")[0][:width]
    return f"{col}（{d}）" if d and d != "（説明未登録）" else col


# ---------------------------------------------------------------- #
# モデルとデータ
# ---------------------------------------------------------------- #

def load_models(release_dir: str, model_dir: str) -> Dict[str, Tuple[object, Dict]]:
    """
    本番5モデル。Release の配置（<release_dir>/model.txt, <algo>_model.joblib）を優先し、
    無ければ予測の配置（<model_dir>/model.txt, models/<algo>/model.joblib）。meta は git の model_dir。
    """
    import joblib
    import lightgbm as lgb
    out = {}
    for a in ALGOS:
        if a == "lgbm":
            cands = [os.path.join(release_dir, "model.txt"), os.path.join(model_dir, "model.txt")]
            meta_path = os.path.join(model_dir, "meta.json")
        else:
            cands = [os.path.join(release_dir, f"{a}_model.joblib"),
                     os.path.join(model_dir, "models", a, "model.joblib")]
            meta_path = os.path.join(model_dir, "models", a, "meta.json")
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None or not os.path.exists(meta_path):
            raise SystemExit(f"{a} のモデルか meta が見つかりません: {cands} / {meta_path}")
        meta = json.load(open(meta_path, encoding="utf-8"))
        model = lgb.Booster(model_file=path) if a == "lgbm" else joblib.load(path)
        out[a] = (model, meta)
    return out


def raw_score(algo: str, model, X: np.ndarray) -> np.ndarray:
    """確率（lgbm は Booster なので predict がそのまま確率）。"""
    if algo == "lgbm":
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


# ---------------------------------------------------------------- #
# 寄与（対数オッズ）
# ---------------------------------------------------------------- #

def tree_shap(algo: str, model, X: np.ndarray) -> np.ndarray:
    """木の SHAP（最後の基準値の列は落とす）。"""
    if algo == "lgbm":
        c = model.predict(X, pred_contrib=True)
    elif algo == "xgb":
        import xgboost as xgbm
        c = model.get_booster().predict(xgbm.DMatrix(X, missing=np.nan), pred_contribs=True)
    else:
        from catboost import Pool
        c = model.get_feature_importance(Pool(X), type="ShapValues")
    return np.asarray(c, dtype=float)[:, :-1]


def transform(ct, X: np.ndarray) -> np.ndarray:
    Z = ct.transform(X)
    return np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)


def output_owners(ct, X: np.ndarray, cols: Sequence[str], n_probe: int = 2000,
                  seed: int = 0) -> np.ndarray:
    """
    前処理の出力列が元のどの列から来たか（index）。出力名（<branch>__<col>...）から読めるものは
    それで、読めない列は1列ずつ揺らして求める。どちらでも決まらない列は -1（寄与は捨てる。
    one-hot で揺らした行に無いカテゴリの列など、揺らした行では値が 0 の列）。
    """
    Z0 = transform(ct, X[:n_probe])
    own = np.full(Z0.shape[1], -1, dtype=int)
    pos = {c: i for i, c in enumerate(cols)}
    try:
        names = list(ct.get_feature_names_out())
    except Exception:
        names = []
    if len(names) == Z0.shape[1]:
        for k, nm in enumerate(names):
            body = nm.split("__", 1)[1] if "__" in nm else nm
            if body.startswith("missingindicator_"):
                body = body[len("missingindicator_"):]
            if body in pos:
                own[k] = pos[body]
                continue
            # one-hot: <col>_<category>
            for c in cols:
                if body.startswith(c + "_"):
                    own[k] = pos[c]
                    break
    if (own == -1).any():
        rng = np.random.default_rng(seed)
        Xb = X[:n_probe].copy()
        for j, c in enumerate(cols):
            if not (own == -1).any():
                break
            Xp = Xb.copy()
            col = Xp[:, j]
            fin = np.isfinite(col)
            if c in TM.CATEGORICAL:
                Xp[:, j] = np.roll(col, 1)
                Xp[: len(Xp) // 4, j] = np.nan
            else:
                sd = float(np.nanstd(col)) if fin.any() else 1.0
                sd = sd if sd > 0 else 1.0
                half = len(Xp) // 2
                Xp[:half, j] = np.where(fin[:half], col[:half] + sd + 1.0, 0.0)
                Xp[half:, j] = np.nan
            Zp = transform(ct, Xp)
            a = np.nan_to_num(Zp, nan=-9e9)
            b = np.nan_to_num(Z0, nan=-9e9)
            diff = np.any(~np.isclose(a, b), axis=0)
            own[diff & (own == -1)] = j
    return own


def fold_to_columns(contrib_z: np.ndarray, own: np.ndarray, n_cols: int) -> np.ndarray:
    """前処理後の列の寄与を、元の列ごとに足す。持ち主が無い列は捨てる。"""
    out = np.zeros((contrib_z.shape[0], n_cols))
    for k in range(contrib_z.shape[1]):
        if own[k] >= 0:
            out[:, own[k]] += contrib_z[:, k]
    return out


def mlp_logit_and_grad(clf, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    scikit-learn の MLPClassifier（relu、出力 logistic）の、出力の対数オッズとその入力勾配。
    forward: a0 = Z; s_l = a_l W_l + b_l; a_{l+1} = relu(s_l)（隠れ層）; out = a_last W + b（対数オッズ）。
    """
    if clf.activation != "relu" or clf.out_activation_ != "logistic":
        raise ValueError(f"想定外の MLP: activation={clf.activation} out={clf.out_activation_}")
    W, B = clf.coefs_, clf.intercepts_
    a = Z
    pres = []
    for l in range(len(W) - 1):
        s = a @ W[l] + B[l]
        pres.append(s)
        a = np.maximum(s, 0.0)
    out = (a @ W[-1] + B[-1])[:, 0]
    G = np.repeat(W[-1].T, len(Z), axis=0)                 # d out / d a_last
    for l in range(len(W) - 2, -1, -1):
        G = (G * (pres[l] > 0)) @ W[l].T
    return out, G


def integrated_gradients(clf, Z: np.ndarray, steps: int = IG_STEPS,
                         baseline: Optional[np.ndarray] = None) -> np.ndarray:
    """積分勾配（基準は 0 = 標準化後の訓練平均）。合計が f(Z) − f(基準) に一致する。"""
    base = np.zeros_like(Z) if baseline is None else np.broadcast_to(baseline, Z.shape)
    diff = Z - base
    acc = np.zeros_like(Z)
    for k in range(1, steps + 1):
        alpha = (k - 0.5) / steps
        _, G = mlp_logit_and_grad(clf, base + alpha * diff)
        acc += G
    return diff * acc / steps


def attributions(algo: str, model, X: np.ndarray, cols: Sequence[str], *, ig_steps: int = IG_STEPS,
                 chunk: int = 4000) -> np.ndarray:
    """行 × 元の列 の寄与（対数オッズ）。"""
    if algo in TREES:
        return np.vstack([tree_shap(algo, model, X[i:i + chunk]) for i in range(0, len(X), chunk)])
    ct = model.steps[0][1]
    clf = model.steps[-1][1]
    own = output_owners(ct, X, cols)
    outs = []
    for i in range(0, len(X), chunk):
        Z = transform(ct, X[i:i + chunk])
        if algo == "logit":
            cz = Z * clf.coef_[0][None, :]
        else:
            cz = integrated_gradients(clf, Z, steps=ig_steps)
        outs.append(fold_to_columns(cz, own, len(cols)))
    return np.vstack(outs)


# ---------------------------------------------------------------- #
# 共通の物差し: 置換で順位がどれだけ崩れるか
# ---------------------------------------------------------------- #

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def permutation_reliance(algo: str, model, X: np.ndarray, base_score: np.ndarray,
                         blocks: Dict[str, List[int]], seed: int = 0) -> Dict[str, float]:
    """blocks（名前 → 列 index のリスト）ごとに、列をまとめて行の間で入れ替えて採点し、1 − ρ を返す。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    out = {}
    for name, idx in blocks.items():
        Xp = X.copy()
        Xp[:, idx] = X[perm][:, idx]
        out[name] = 1.0 - spearman(base_score, raw_score(algo, model, Xp))
    return out


# ---------------------------------------------------------------- #
# 単独の分離力
# ---------------------------------------------------------------- #

def univariate_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """列ごとの ROC-AUC（NaN の行は除く。片方のクラスしか無ければ NaN）。0.5 からの距離で強さを見る。"""
    from sklearn.metrics import roc_auc_score
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        v = X[:, j]
        m = np.isfinite(v)
        if m.sum() < 100 or len(np.unique(y[m])) < 2 or np.nanstd(v[m]) == 0:
            continue
        out[j] = roc_auc_score(y[m], v[m])
    return out


# ---------------------------------------------------------------- #
# OOF: 一致・不一致・性格・型
# ---------------------------------------------------------------- #

def load_oofs(oof_dir: str, model_dir: str) -> pd.DataFrame:
    """5モデルの OOF を (Code, Date) で揃える（label は lgbm の OOF）。"""
    base = None
    for a in ALGOS:
        path = L.find_oof(a, oof_dir, model_dir)
        if path is None:
            raise SystemExit(f"{a} の OOF が見つかりません（{oof_dir} / {model_dir}）")
        o = pd.read_parquet(path)
        o["Date"] = pd.to_datetime(o["Date"])
        o["Code"] = o["Code"].astype(str)
        keep = ["Code", "Date", "score"] + ([c for c in ("label", "ret_o1_20") if c in o] if a == "lgbm" else [])
        o = o[keep].rename(columns={"score": f"s_{a}"})
        base = o if base is None else base.merge(o, on=["Code", "Date"], how="inner")
    return base


def within_date_pct(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    同じ日の候補の中での百分位。(順位 − 0.5) ÷ その日の件数 で、偏りが無ければ平均がちょうど 0.5 になる
    （rank(pct=True) は (n+1)/(2n) に寄るので、1日の件数が少ないと 0.5 を超えてしまう）。NaN はそのまま。
    """
    g = df.groupby("Date")[list(cols)]
    return (g.rank(method="average") - 0.5) / g.transform("count")


def hier_order(corr: pd.DataFrame) -> List[str]:
    """階層クラスタリング（平均連結）の併合の順を文字で返す。"""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    d = 1.0 - corr.to_numpy()
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    names = list(corr.index)
    groups = {i: [n] for i, n in enumerate(names)}
    lines = []
    for k, (i, j, dist, _) in enumerate(Z):
        gi, gj = groups.pop(int(i)), groups.pop(int(j))
        groups[len(names) + k] = gi + gj
        lines.append(f"{'+'.join(gi)} と {'+'.join(gj)}（距離 {dist:.3f} = 1 − 相関）")
    return lines


# ---------------------------------------------------------------- #
# 表示
# ---------------------------------------------------------------- #

def _r(v, w):
    return L._r(v, w)


def _l(v, w):
    return L._l(v, w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="実験60: 本番5モデルの傾向")
    ap.add_argument("--data-dir", default=lab.DATA_DIR)
    ap.add_argument("--release-dir", default=lab.DATA_DIR, help="Release を落とした場所（model.txt など）")
    ap.add_argument("--oof-dir", default="", help="OOF の場所（既定: --release-dir）")
    ap.add_argument("--model-dir", default=M.MODEL_DIR, help="meta.json の場所（git）")
    ap.add_argument("--ig-steps", type=int, default=IG_STEPS)
    ap.add_argument("--quick", action="store_true", help="試運転: 行を減らす")
    ap.add_argument("--prefix", default="e60")
    args = ap.parse_args(argv)
    oof_dir = args.oof_dir or args.release_dir
    os.makedirs(OOF_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    models = load_models(args.release_dir, args.model_dir)
    cols = F.columns(F.DEFAULT_PRESET)
    for a, (_, meta) in models.items():
        feats = meta.get("features") or []
        if list(feats) != list(cols):
            raise SystemExit(f"{a} の meta の列（{len(feats)}）が本番の preset（{len(cols)}）と違います")
    g_of = F.column_groups()
    group = [g_of.get(c, "") for c in cols]
    family = [FAMILY_OF.get(g, "その他") for g in group]
    lgbm_meta = models["lgbm"][1]

    ds = pd.read_parquet(os.path.join(args.data_dir, "dataset.parquet"))
    ds["Date"] = pd.to_datetime(ds["Date"])
    ds["Code"] = ds["Code"].astype(str)
    ds = ds[ds["label"].notna()]
    lo, hi = pd.Timestamp(lgbm_meta["trainFrom"]), pd.Timestamp(lgbm_meta["trainTo"])
    ds = ds[(ds["Date"] >= lo) & (ds["Date"] <= hi)].reset_index(drop=True)
    if args.quick:
        ds = ds.sample(min(len(ds), 3000), random_state=0).sort_values("Date").reset_index(drop=True)
    X = ds[cols].to_numpy(dtype=float)
    y = ds["label"].to_numpy(dtype=int)
    recent = (ds["Date"] > hi - pd.Timedelta(days=365)).to_numpy()

    print("=" * 100)
    print("実験60 本番5モデルの傾向: 頼り方（寄与）・置換の物差し・過大評価・一致と不一致・上位銘柄の性格・型")
    print("=" * 100)
    print(f"  モデル 学習 {str(lgbm_meta.get('trainedAt', ''))[:10]} / {lgbm_meta.get('preset')} {len(cols)}列 / "
          f"logit 前処理 {models['logit'][1].get('preprocess') or 'v1（meta に記録なし）'} / "
          f"mlp 前処理 {models['mlp'][1].get('preprocess') or 'v1（meta に記録なし）'}")
    print(f"  寄与と置換に使う行: 学習に使った {len(ds):,}件（{lo.date()}〜{hi.date()}。直近1年 {int(recent.sum()):,}件）"
          + ("【試運転: 行を減らしている】" if args.quick else ""))

    # ---- 1. 寄与 ----
    attr: Dict[str, np.ndarray] = {}
    scores: Dict[str, np.ndarray] = {}
    for a, (model, _) in models.items():
        t0 = time.time()
        attr[a] = attributions(a, model, X, cols, ig_steps=args.ig_steps)
        scores[a] = raw_score(a, model, X)
        extra = ""
        if a == "mlp":
            clf = model.steps[-1][1]
            Z = transform(model.steps[0][1], X[:500])
            f1, _ = mlp_logit_and_grad(clf, Z)
            f0, _ = mlp_logit_and_grad(clf, np.zeros_like(Z))
            tot = integrated_gradients(clf, Z, steps=args.ig_steps).sum(axis=1)
            extra = f"（積分勾配の合計と f(Z)−f(0) の差: 最大 {np.abs(tot - (f1 - f0)).max():.4f}）"
        if a == "logit":
            p = scores[a][:500]
            lo_ = np.log(p / (1 - p))
            Z = transform(model.steps[0][1], X[:500])
            clf = model.steps[-1][1]
            pred = Z @ clf.coef_[0] + clf.intercept_[0]
            extra = f"（係数×値の合計 + 切片 と decision の差: 最大 {np.abs(pred - lo_).max():.4f}）"
        log(f"{a}: 寄与 {attr[a].shape} / {time.time() - t0:.0f}秒 {extra}")

    uni = univariate_auc(X, y)
    strength = np.abs(uni - 0.5)

    rel_rows = []
    fam_share: Dict[str, pd.Series] = {}
    grp_share: Dict[str, pd.Series] = {}
    for a in ALGOS:
        A = attr[a]
        mean_abs = np.abs(A).mean(axis=0)
        share = mean_abs / mean_abs.sum()
        mean_abs_recent = np.abs(A[recent]).mean(axis=0) if recent.any() else mean_abs
        share_recent = mean_abs_recent / mean_abs_recent.sum()
        # 向き: 列の値と寄与の順位相関（+ なら値が高いほどスコアを押し上げる）
        sign = np.array([spearman(X[:, j], A[:, j]) if np.isfinite(X[:, j]).sum() > 100 else np.nan
                         for j in range(len(cols))])
        fs = pd.Series(share, index=cols).groupby(pd.Series(family, index=cols)).sum()
        gs = pd.Series(share, index=cols).groupby(pd.Series(group, index=cols)).sum()
        fam_share[a], grp_share[a] = fs, gs
        for j, c in enumerate(cols):
            rel_rows.append({"model": a, "col": c, "group": group[j], "family": family[j],
                             "mean_abs": float(mean_abs[j]), "share": float(share[j]),
                             "share_recent": float(share_recent[j]), "sign": float(sign[j]),
                             "uni_auc": float(uni[j]) if np.isfinite(uni[j]) else float("nan"),
                             "uni_strength": float(strength[j]) if np.isfinite(strength[j]) else float("nan")})
    rel = pd.DataFrame(rel_rows)
    rel["rank"] = rel.groupby("model")["share"].rank(ascending=False, method="first")

    # ---- 2. 置換の物差し ----
    blocks_g = {g: [j for j, gg in enumerate(group) if gg == g] for g in dict.fromkeys(group)}
    blocks_c = {c: [j] for j, c in enumerate(cols)}
    perm_g: Dict[str, Dict[str, float]] = {}
    perm_c: Dict[str, Dict[str, float]] = {}
    for a, (model, _) in models.items():
        t0 = time.time()
        perm_g[a] = permutation_reliance(a, model, X, scores[a], blocks_g)
        perm_c[a] = permutation_reliance(a, model, X, scores[a], blocks_c)
        log(f"{a}: 置換 {len(blocks_g)}区分 + {len(blocks_c)}列 / {time.time() - t0:.0f}秒")
    rel["perm"] = [perm_c[r.model][r.col] for r in rel.itertuples()]

    fam_tbl = pd.DataFrame(fam_share).fillna(0.0)
    fam_tbl["平均"] = fam_tbl.mean(axis=1)
    fam_tbl = fam_tbl.sort_values("平均", ascending=False)
    grp_tbl = pd.DataFrame(grp_share).fillna(0.0)
    grp_tbl["平均"] = grp_tbl.mean(axis=1)
    perm_tbl = pd.DataFrame(perm_g).fillna(0.0)
    perm_fam = perm_tbl.groupby(pd.Series({g: FAMILY_OF.get(g, "その他") for g in perm_tbl.index})).sum()

    print("\n■ 1. 大区分ごとの寄与の割合（|寄与| の平均の割合。学習に使った全行。%）と、5モデルの平均との差")
    print("  " + _l("大区分", 22) + "".join(_r(a, 8) for a in ALGOS) + _r("平均", 8) + "   他より重く見ている（差 +2pt 以上）")
    for fam, row in fam_tbl.iterrows():
        over = [f"{a} {100 * (row[a] - row['平均']):+.0f}" for a in ALGOS if row[a] - row["平均"] >= 0.02]
        print("  " + _l(fam, 22) + "".join(_r(f"{100 * row[a]:.1f}", 8) for a in ALGOS)
              + _r(f"{100 * row['平均']:.1f}", 8) + "   " + " / ".join(over))

    print("\n■ 2. 置換の物差し: 大区分の列をまとめて入れ替えたときの順位の崩れ（1 − 順位相関 ×100。大きいほど依存）")
    pf = perm_fam.copy()
    pf["平均"] = pf.mean(axis=1)
    pf = pf.sort_values("平均", ascending=False)
    print("  " + _l("大区分", 22) + "".join(_r(a, 8) for a in ALGOS) + _r("平均", 8))
    for fam, row in pf.iterrows():
        print("  " + _l(fam, 22) + "".join(_r(f"{100 * row[a]:.1f}", 8) for a in ALGOS) + _r(f"{100 * row['平均']:.1f}", 8))

    print("\n■ 3. モデルごとの上位15列（寄与の割合%、向き、置換の崩れ×100、単独の AUC）")
    for a in ALGOS:
        sub = rel[rel["model"] == a].sort_values("share", ascending=False).head(15)
        print(f"  ▼ {M.JA[a]}")
        for r in sub.itertuples():
            d = "+" if r.sign > 0.1 else ("−" if r.sign < -0.1 else "±")
            weak = " ←単独では弱い" if np.isfinite(r.uni_strength) and r.uni_strength < WEAK_AUC else ""
            print(f"    {_l(ja(r.col), 44)} {100 * r.share:5.1f}%  {d}  置換 {100 * r.perm:4.1f}  "
                  f"AUC {r.uni_auc:.3f}{weak}")

    print("  （注意: s33_code 等のカテゴリ列は one-hot の多数の列の寄与の合計なので、線形・MLP では大きく出やすい。"
          "依存の実際は「置換」の値で読む）")
    print(f"\n■ 4. 過大評価の候補: 寄与の上位30列のうち、単独の分離力が弱い（|AUC−0.5| < {WEAK_AUC}）列")
    for a in ALGOS:
        sub = rel[rel["model"] == a].sort_values("share", ascending=False).head(30)
        weak = sub[sub["uni_strength"] < WEAK_AUC]
        rho = spearman(sub["share"].to_numpy(), sub["uni_strength"].fillna(0).to_numpy())
        print(f"  {M.JA[a]}: {len(weak)}列（上位30列の寄与と単独の分離力の順位相関 {rho:+.2f}）: "
              + " / ".join(f"{ja(r.col, 10)} {100 * r.share:.1f}%" for r in weak.itertuples()))

    print("\n■ 5. 他の4モデルより重く見ている中区分（寄与の割合の差 +1.5pt 以上）と、軽く見ている中区分（−1.5pt 以下）")
    for a in ALGOS:
        d = (grp_tbl[a] - grp_tbl[[b for b in ALGOS if b != a]].mean(axis=1)) * 100
        up = d[d >= 1.5].sort_values(ascending=False)
        dn = d[d <= -1.5].sort_values()
        print(f"  {M.JA[a]}: 重い " + (" / ".join(f"{g} {v:+.1f}" for g, v in up.items()) or "なし")
              + " ｜ 軽い " + (" / ".join(f"{g} {v:+.1f}" for g, v in dn.items()) or "なし"))

    # ---- OOF ----
    oof = load_oofs(oof_dir, args.model_dir)
    oof = oof.merge(ds[["Code", "Date"] + cols], on=["Code", "Date"], how="inner")
    if args.quick:
        # 日ごとの百分位を使うので、行ではなく日を間引く
        days = pd.Series(oof["Date"].unique())
        keep = days.sample(min(len(days), 300), random_state=0)
        oof = oof[oof["Date"].isin(keep)].reset_index(drop=True)
    corr = oof[[f"s_{a}" for a in ALGOS]].corr(method="spearman")
    corr.index = corr.columns = list(ALGOS)
    top = {a: oof[f"s_{a}"] >= oof[f"s_{a}"].quantile(1 - TOP_SHARE) for a in ALGOS}
    print(f"\n■ 6. 一致（OOF {len(oof):,}件）: スコアの順位相関と、上位10% の重なり（Jaccard）")
    print("  " + _l("", 8) + "".join(_r(a, 8) for a in ALGOS) + "     上位10% の重なり")
    jac_rows = []
    for a in ALGOS:
        js = []
        for b in ALGOS:
            inter = int((top[a] & top[b]).sum())
            union = int((top[a] | top[b]).sum())
            jac = inter / union if union else float("nan")
            js.append(jac)
            jac_rows.append({"a": a, "b": b, "spearman": float(corr.loc[a, b]), "jaccard_top": jac})
        print("  " + _l(a, 8) + "".join(_r(f"{corr.loc[a, b]:.2f}", 8) for b in ALGOS)
              + "     " + " ".join(f"{b} {j:.2f}" for b, j in zip(ALGOS, js) if b != a))
    print("  階層クラスタリング（平均連結）の併合の順: " + " → ".join(hier_order(corr)))

    print("\n■ 7. 不一致: LightGBM と意見が割れた行（片方だけ上位10%、もう片方は下位50%）で、どちらが当たったか（OOF の正例率）")
    base_rate = float(oof["label"].mean())
    dis_rows = []
    for b in ALGOS[1:]:
        low_b = oof[f"s_{b}"] <= oof[f"s_{b}"].median()
        low_a = oof["s_lgbm"] <= oof["s_lgbm"].median()
        only_a = top["lgbm"] & low_b
        only_b = top[b] & low_a
        both = top["lgbm"] & top[b]
        r = {"other": b, "n_only_lgbm": int(only_a.sum()), "rate_only_lgbm": float(oof.loc[only_a, "label"].mean()) if only_a.any() else float("nan"),
             "n_only_other": int(only_b.sum()), "rate_only_other": float(oof.loc[only_b, "label"].mean()) if only_b.any() else float("nan"),
             "n_both": int(both.sum()), "rate_both": float(oof.loc[both, "label"].mean()) if both.any() else float("nan")}
        dis_rows.append(r)
        print(f"  lgbm vs {b}: lgbm だけ上位 {r['n_only_lgbm']}件 正例率 {100 * r['rate_only_lgbm']:.1f}% ｜ "
              f"{b} だけ上位 {r['n_only_other']}件 {100 * r['rate_only_other']:.1f}% ｜ "
              f"両方上位 {r['n_both']}件 {100 * r['rate_both']:.1f}%（母集団 {100 * base_rate:.1f}%）")

    # 代表的な列: 5モデル平均の寄与の割合が高い順に、中区分あたり2列まで。
    # 同じ日の候補で値が同じ列（地合い・市場全体の需給など日付の列）は「候補の中での位置」が出ないので外す。
    # すでに選んだ列と順位相関 0.9 超の列（ほぼ同じもの）も外す
    avg_share = rel.groupby("col")["share"].mean().sort_values(ascending=False)
    numeric = [c for c in cols if c not in TM.CATEGORICAL]
    pct_all = within_date_pct(oof, numeric)
    within_sd = pct_all.std()
    rep, per_g = [], {}
    for c in avg_share.index:
        g = g_of.get(c, "")
        if per_g.get(g, 0) >= 2 or c in TM.CATEGORICAL or not (within_sd.get(c, 0) > 0.05):
            continue
        # 「日の中の位置」がすでに選んだ列とほぼ同じ（例: vol_rel_mkt は vol_20d ÷ 日付の定数）なら外す
        if any(abs(spearman(pct_all[c].fillna(0.5).to_numpy(), pct_all[r].fillna(0.5).to_numpy())) > 0.9
               for r in rep):
            continue
        rep.append(c)
        per_g[g] = per_g.get(g, 0) + 1
        if len(rep) >= N_REP:
            break
    pct = pct_all[rep]
    print(f"\n■ 8. 上位10% の性格: 代表的な {len(rep)}列の「同じ日の候補の中での百分位」の平均（0.5 = 母集団と同じ）。"
          f"5モデルで最も偏る列から")
    prof = pd.DataFrame({a: pct[top[a]].mean() for a in ALGOS})
    prof["平均"] = prof.mean(axis=1)
    prof["ばらつき"] = prof[list(ALGOS)].std(axis=1)
    prof = prof.reindex((prof["平均"] - 0.5).abs().sort_values(ascending=False).index)
    print("  " + _l("列", 40) + "".join(_r(a, 7) for a in ALGOS) + _r("平均", 7) + _r("SD", 6))
    for c, row in prof.iterrows():
        print("  " + _l(ja(c, 14), 40) + "".join(_r(f"{row[a]:.2f}", 7) for a in ALGOS)
              + _r(f"{row['平均']:.2f}", 7) + _r(f"{row['ばらつき']:.2f}", 6))
    print("  モデル間で最も割れる列（SD の大きい順）: " + " / ".join(
        f"{ja(c, 10)}（{'・'.join(f'{a} {prof.loc[c, a]:.2f}' for a in ALGOS)}）"
        for c in prof["ばらつき"].sort_values(ascending=False).head(5).index))

    # ---- 型 ----
    from sklearn.cluster import KMeans
    union = pd.Series(False, index=oof.index)
    for a in ALGOS:
        union |= top[a]
    P = pct.loc[union].fillna(0.5)
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=0).fit(P.to_numpy())
    lab_ = pd.Series(km.labels_, index=P.index)
    print(f"\n■ 9. 型: 5モデルの上位10% の和集合（{int(union.sum()):,}件）を代表的な列で k-means（k={K_CLUSTERS}）。"
          f"型ごとの特徴（百分位の平均が 0.5 から遠い列）、各モデルの上位の内訳、OOF の正例率")
    cl_rows = []
    for k in range(K_CLUSTERS):
        m = lab_ == k
        cen = P[m].mean()
        dev = (cen - 0.5).sort_values(key=np.abs, ascending=False).head(4)
        shares = {a: float((top[a] & union)[m[m].index].sum() / max(int(top[a].sum()), 1)) for a in ALGOS}
        rate = float(oof.loc[m[m].index, "label"].mean())
        cl_rows.append({"cluster": k, "n": int(m.sum()), "label_rate": rate,
                        "features": "; ".join(f"{c} {v:+.2f}" for c, v in dev.items()), **{f"share_{a}": shares[a] for a in ALGOS}})
        print(f"  型{k + 1}（{int(m.sum())}件、正例率 {100 * rate:.0f}%）: "
              + " / ".join(f"{ja(c, 10)} {v:+.2f}" for c, v in dev.items()))
        print("      各モデルの上位10% に占める割合: " + " ".join(f"{a} {100 * shares[a]:.0f}%" for a in ALGOS))

    # ---- 保存 ----
    rel.to_csv(os.path.join(OOF_DIR, f"{args.prefix}_reliance.csv"), index=False)
    fam_tbl.to_csv(os.path.join(OOF_DIR, f"{args.prefix}_family.csv"))
    grp_tbl.join(perm_tbl.add_prefix("perm_")).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_group.csv"))
    pd.DataFrame(jac_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_agreement.csv"), index=False)
    prof.to_csv(os.path.join(OOF_DIR, f"{args.prefix}_profile.csv"))
    pd.DataFrame(dis_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_disagree.csv"), index=False)
    pd.DataFrame(cl_rows).to_csv(os.path.join(OOF_DIR, f"{args.prefix}_clusters.csv"), index=False)
    summary = {"trainedAt": lgbm_meta.get("trainedAt"), "n_rows": int(len(ds)), "n_oof": int(len(oof)),
               "family_share": {a: {k: float(v) for k, v in fam_share[a].items()} for a in ALGOS},
               "perm_family": {a: {k: float(v) for k, v in perm_fam[a].items()} for a in ALGOS},
               "spearman": {a: {b: float(corr.loc[a, b]) for b in ALGOS} for a in ALGOS},
               "rep_cols": rep}
    json.dump(summary, open(os.path.join(OOF_DIR, f"{args.prefix}_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"書いた: {OOF_DIR}/{args.prefix}_*.csv / _summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
