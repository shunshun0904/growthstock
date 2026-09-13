#!/usr/bin/env python3
"""
会計フローグラフの EDA 集計。

描画はしない。集計結果を JSON にして research/accgraph/eda_report.py が組版する。
データセットは CI 側にしかないので、集計と描画を分けておかないと
手元で図を直すたびに再集計が要る。

見るもの:

  1. 取得可能性 — どのノードが実質空か。CF の半期開示がどこまで効いているか
  2. 目的変数  — 超過リターンの分布とクラス比が、年・四半期・業種・流動性・
                 決算の混雑度でどう偏るか。ベンチマーク控除が効いているか
  3. 特徴量    — 欠損・外れ値・定数列・冗長な組み合わせ・単変量の情報量
  4. グラフ構造— 会計恒等式が実データで成立しているか。CF の符号パターン

  python3 research/accgraph/eda.py
  -> research/_data/accgraph/eda.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from accgraph import build, labels as L, schema  # noqa: E402

OUT_JSON = os.path.join(build.OUT_DIR, "eda.json")

N_BINS = 36
#: 相関がこれを超える組を「冗長かもしれない」として報告する
CORR_WARN = 0.95
#: 相関行列に使う行数の上限。総当たりは列数の2乗 × 行数に比例するので、
#: 10万行 × 234列をそのまま回すと数十分かかる。冗長性の判定に精度は要らない
CORR_SAMPLE = 30_000
#: 情報係数を出すのに必要な最小サンプル数。これ未満は順位相関が運で決まる
IC_MIN_N = 500
#: 流動性の帯（20日平均売買代金・億円）。README §4.2 の機関投資家参入度に合わせる
TURNOVER_BANDS = [(0, 1, "1億円未満"), (1, 5, "1〜5億円"), (5, 10, "5〜10億円"),
                  (10, 30, "10〜30億円"), (30, np.inf, "30億円以上")]
#: 同じ日に何社が決算を出したか（決算の混雑度）
CROWDING_BANDS = [(0, 10, "10社以下"), (10, 50, "11〜50社"),
                  (50, 200, "51〜200社"), (200, np.inf, "201社以上")]

#: 分布を並べるノード特徴量（当該四半期 T=0）。全部出すと読めない
SHOWCASE_FEATURES = ["scaled", "yoy_sym", "qoq_sym", "slope4", "vol4", "fcst_gap"]
#: 分布を並べるノード
SHOWCASE_NODES = ["sales", "op", "ordinary_profit", "net_income",
                  "cfo", "cfi", "fcf", "total_assets", "equity", "cash"]


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #

def _pct(mask: np.ndarray) -> float:
    return float(np.mean(mask) * 100) if len(mask) else float("nan")


def histogram(v: np.ndarray, bins: int = N_BINS) -> Optional[Dict]:
    """
    ビンごとの件数。外れ値で潰れないよう 1%〜99% で範囲を切り、
    範囲外の件数は別に持つ（描画側で「範囲外 N件」と出せるように）。
    """
    x = np.asarray(v, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    lo, hi = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    inside = x[(x >= lo) & (x <= hi)]
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    return {"lo": lo, "hi": hi, "counts": counts.tolist(),
            "outside": int(len(x) - len(inside)), "n": int(len(x))}


def quantiles(v: np.ndarray) -> Dict[str, Optional[float]]:
    x = np.asarray(v, dtype="float64")
    x = x[np.isfinite(x)]
    if not len(x):
        return {k: None for k in ("p01", "p25", "p50", "p75", "p99", "mean", "std")}
    q = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {"p01": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p99": float(q[4]),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def band_of(values: np.ndarray, bands) -> np.ndarray:
    """値を帯の名前に割り当てる。欠測は None。"""
    out = np.full(len(values), None, dtype=object)
    for lo, hi, name in bands:
        m = np.isfinite(values) & (values >= lo) & (values < hi)
        out[m] = name
    return out


def class_share(y: np.ndarray) -> Dict[str, float]:
    """3クラスの構成比（%）。欠測（-1）は除く。"""
    v = y[y >= 0]
    if not len(v):
        return {n: float("nan") for n in L.CLASS_NAMES}
    return {L.CLASS_NAMES[c]: float(np.mean(v == c) * 100) for c in (0, 1, 2)}


def _group_share(meta: pd.DataFrame, y: np.ndarray, key: pd.Series,
                 excess: np.ndarray, min_n: int = 30) -> Dict[str, Dict]:
    """グループごとのクラス比・件数・超過リターン中央値。"""
    out: Dict[str, Dict] = {}
    for name, idx in key.groupby(key).groups.items():
        if name is None or (isinstance(name, float) and np.isnan(name)):
            continue
        pos = meta.index.get_indexer(idx)
        yy = y[pos]
        if int((yy >= 0).sum()) < min_n:
            continue
        ex = excess[pos]
        ex = ex[np.isfinite(ex)]
        out[str(name)] = {
            "n": int((yy >= 0).sum()),
            "share": class_share(yy),
            "median_excess": float(np.median(ex)) if len(ex) else None,
            "mean_excess": float(np.mean(ex)) if len(ex) else None,
        }
    return out


# --------------------------------------------------------------------------- #
# 1. 取得可能性
# --------------------------------------------------------------------------- #

def availability(meta: pd.DataFrame, node_feat: np.ndarray,
                 edge_feat: np.ndarray, period_mask: np.ndarray) -> Dict:
    mi = schema.NODE_FEATURES.index("is_missing")
    sp = schema.NODE_FEATURES.index("span")
    present = 1.0 - node_feat[:, 0, :, mi]          # [n, n_nodes]
    span = node_feat[:, 0, :, sp]

    q = pd.to_numeric(meta["quarter"], errors="coerce").to_numpy()
    year = pd.to_datetime(meta["entry_date"]).dt.year.to_numpy()
    s33 = meta["s33"].astype(str).to_numpy() if "s33" in meta.columns else None

    nodes = []
    for j, node in enumerate(schema.NODES):
        row = {
            "id": node.id, "name_ja": node.name_ja, "statement": node.statement,
            "source": node.source, "kind": node.kind,
            "overall": _pct(present[:, j] > 0),
            "by_quarter": {str(k): _pct(present[q == k, j] > 0)
                           for k in (1, 2, 3, 4) if (q == k).any()},
            "by_year": {str(int(yv)): _pct(present[year == yv, j] > 0)
                        for yv in sorted(set(year.tolist()))},
        }
        # 何四半期ぶんの値か（CF は半期がまとまって 2 になる）
        s = span[present[:, j] > 0, j]
        row["median_span"] = float(np.median(s)) if len(s) else None
        row["half_year_share"] = _pct(s >= 1.5) if len(s) else None
        nodes.append(row)

    bp = schema.EDGE_FEATURES.index("both_present")
    edges = [{"src": e.src, "dst": e.dst, "kind": e.kind,
              "both_present": _pct(edge_feat[:, 0, i, bp] > 0)}
             for i, e in enumerate(schema.EDGES)]

    lens = period_mask.sum(axis=1)
    out = {
        "nodes": nodes,
        "edges": edges,
        "seq_len_hist": {str(k): int((lens == k).sum())
                         for k in range(period_mask.shape[1] + 1)},
        "full_seq_pct": _pct(lens == period_mask.shape[1]),
        "samples_by_year": {str(int(yv)): int((year == yv).sum())
                            for yv in sorted(set(year.tolist()))},
        "samples_by_quarter": {str(k): int((q == k).sum()) for k in (1, 2, 3, 4)},
    }
    if s33 is not None:
        vc = pd.Series(s33).value_counts()
        out["samples_by_sector"] = {str(k): int(v) for k, v in vc.items()
                                    if k not in ("nan", "None")}
    return out


# --------------------------------------------------------------------------- #
# 2. 目的変数
# --------------------------------------------------------------------------- #

def label_stats(meta: pd.DataFrame, horizons=(5, 10, 20)) -> Dict:
    out: Dict = {"horizons": {}}
    year = pd.to_datetime(meta["entry_date"]).dt.year.astype(str)
    quarter = pd.to_numeric(meta["quarter"], errors="coerce").astype("Int64").astype(str)
    tv = pd.to_numeric(meta["turnover_ma20"], errors="coerce").to_numpy()
    turnover = pd.Series(band_of(tv, TURNOVER_BANDS), index=meta.index)
    crowd_n = meta.groupby("disc_date")["anchor_id"].transform("size").to_numpy()
    crowding = pd.Series(band_of(crowd_n.astype(float), CROWDING_BANDS),
                         index=meta.index)
    sector = (meta["s33"].astype(str) if "s33" in meta.columns
              else pd.Series("不明", index=meta.index))

    for h in horizons:
        block: Dict = {}
        for bench in ("topix", "sector"):
            ycol, ecol = f"y_{bench}_{h}d", f"excess_{bench}_{h}d"
            if ycol not in meta.columns:
                continue
            y = meta[ycol].to_numpy().astype(int)
            ex = pd.to_numeric(meta[ecol], errors="coerce").to_numpy()
            block[bench] = {
                "n": int((y >= 0).sum()),
                "share": class_share(y),
                "quantiles": quantiles(ex),
                "hist": histogram(ex),
            }
        raw = pd.to_numeric(meta.get(f"ret_{h}d"), errors="coerce").to_numpy()
        block["raw"] = {"quantiles": quantiles(raw), "hist": histogram(raw)}
        out["horizons"][str(h)] = block

    # 偏りは主保有期間（20営業日・TOPIX控除）で見る
    h = max(horizons)
    ycol, ecol = f"y_topix_{h}d", f"excess_topix_{h}d"
    if ycol in meta.columns:
        y = meta[ycol].to_numpy().astype(int)
        ex = pd.to_numeric(meta[ecol], errors="coerce").to_numpy()
        out["primary_horizon"] = h
        out["by_year"] = _group_share(meta, y, year, ex)
        out["by_quarter"] = _group_share(meta, y, quarter, ex)
        out["by_sector"] = _group_share(meta, y, sector, ex, min_n=100)
        out["by_turnover"] = _group_share(meta, y, turnover, ex)
        out["by_crowding"] = _group_share(meta, y, crowding, ex)
        # ベンチマーク控除が実際に何を消しているか
        raw = pd.to_numeric(meta.get(f"ret_{h}d"), errors="coerce").to_numpy()
        ok = np.isfinite(raw) & np.isfinite(ex)
        out["benchmark_effect"] = {
            "raw_std": float(np.std(raw[ok], ddof=1)) if ok.sum() > 1 else None,
            "excess_std": float(np.std(ex[ok], ddof=1)) if ok.sum() > 1 else None,
            "raw_mean": float(np.mean(raw[ok])) if ok.any() else None,
            "excess_mean": float(np.mean(ex[ok])) if ok.any() else None,
            # 年ごとの平均が 0 から離れていれば、その年は市場全体が動いている
            "raw_mean_by_year": {k: float(np.mean(raw[ok][year[ok].to_numpy() == k]))
                                 for k in sorted(set(year[ok].tolist()))},
            "excess_mean_by_year": {k: float(np.mean(ex[ok][year[ok].to_numpy() == k]))
                                    for k in sorted(set(year[ok].tolist()))},
        }
    return out


# --------------------------------------------------------------------------- #
# 3. 特徴量
# --------------------------------------------------------------------------- #

def _t0_frame(node_feat: np.ndarray) -> pd.DataFrame:
    """当該四半期（T=0）のノード特徴量を [n, 18*13] の表にする。"""
    n = len(node_feat)
    flat = node_feat[:, 0].reshape(n, -1)
    cols = [f"{nid}.{f}" for nid in schema.NODE_IDS
            for f in schema.NODE_FEATURES]
    return pd.DataFrame(flat, columns=cols)


def feature_stats(meta: pd.DataFrame, node_feat: np.ndarray,
                  corr_sample: int = CORR_SAMPLE, seed: int = 0) -> Dict:
    df = _t0_frame(node_feat)
    mi = schema.NODE_FEATURES.index("is_missing")
    missing = node_feat[:, 0, :, mi]                 # [n, n_nodes]

    stats: Dict[str, Dict] = {}
    constant: List[str] = []
    for j, nid in enumerate(schema.NODE_IDS):
        m = missing[:, j] > 0
        for f in schema.NODE_FEATURES:
            col = f"{nid}.{f}"
            v = df[col].to_numpy()
            # 欠測の期は 0 で埋めてある。0 を分布に混ぜると山が立つので外す
            vv = v if f == "is_missing" else v[~m]
            s = quantiles(vv)
            s["missing_pct"] = _pct(m)
            s["n_unique"] = int(len(np.unique(np.round(vv, 6)))) if len(vv) else 0
            # クリップの上限・下限に張り付いた割合。発散を疑う手がかり
            s["clipped_pct"] = (_pct(np.abs(vv) >= build.CLIP - 1e-9)
                                if len(vv) else float("nan"))
            stats[col] = s
            if s["n_unique"] <= 1:
                constant.append(col)

    # --- 冗長な組み合わせ --- #
    # 欠測を 0 埋めしたまま相関を取ると、欠測パターンが同じ列どうしが
    # 高く出る。欠測を除いた行だけで取る
    use = [c for c in df.columns if c not in constant]
    valid = pd.DataFrame(np.repeat(missing == 0, len(schema.NODE_FEATURES), axis=1),
                         columns=df.columns)
    rows = np.arange(len(df))
    if len(rows) > corr_sample:
        rows = np.sort(np.random.default_rng(seed).choice(
            len(df), corr_sample, replace=False))
    masked = df.iloc[rows][use].where(valid.iloc[rows][use])
    corr = masked.corr(min_periods=200).abs()
    red = []
    cols = list(corr.columns)
    arr = corr.to_numpy()
    for i in range(len(cols)):
        for k in range(i + 1, len(cols)):
            c = arr[i, k]
            if np.isfinite(c) and c >= CORR_WARN:
                red.append({"a": cols[i], "b": cols[k], "corr": float(c)})
    red.sort(key=lambda r: -r["corr"])

    hist = {}
    for nid in SHOWCASE_NODES:
        for f in SHOWCASE_FEATURES:
            col = f"{nid}.{f}"
            if col not in df.columns:
                continue
            j = schema.NODE_INDEX[nid]
            v = df[col].to_numpy()[missing[:, j] == 0]
            h = histogram(v)
            if h:
                hist[col] = h

    return {"stats": stats, "constant": constant, "redundant": red[:40],
            "hist": hist, "n_columns": int(df.shape[1]),
            "corr_rows": int(len(rows))}


def information_coefficient(meta: pd.DataFrame, node_feat: np.ndarray,
                            horizon: int = 20, bench: str = "topix",
                            top: int = 25, min_n: int = IC_MIN_N) -> Dict:
    """
    各特徴量と超過リターンの順位相関（情報係数）。

    単変量でどれだけ効くかの目安。モデルの結論ではないが、
    ここが全部ゼロ付近なら、グラフにする以前に材料が無いということ。
    順位相関にするのは、外れ値1件で符号が変わらないようにするため。
    """
    ecol = f"excess_{bench}_{horizon}d"
    if ecol not in meta.columns:
        return {}
    y = pd.to_numeric(meta[ecol], errors="coerce")
    df = _t0_frame(node_feat)
    mi = schema.NODE_FEATURES.index("is_missing")
    missing = node_feat[:, 0, :, mi]

    rows = []
    ry_all = y.rank()
    for j, nid in enumerate(schema.NODE_IDS):
        keep = (missing[:, j] == 0) & y.notna().to_numpy()
        if keep.sum() < min_n:
            continue
        for f in schema.NODE_FEATURES:
            if f in ("is_missing", "fcst_avail"):
                continue
            col = f"{nid}.{f}"
            v = df.loc[keep, col]
            if v.nunique() <= 1:
                continue
            # 母集団を揃えるため、欠測を除いた行の中で順位を取り直す
            ic = float(v.rank().corr(ry_all[keep].rank()))
            if np.isfinite(ic):
                rows.append({"col": col, "ic": ic, "n": int(keep.sum())})
    rows.sort(key=lambda r: -abs(r["ic"]))
    return {"horizon": horizon, "benchmark": bench, "top": rows[:top],
            "n_evaluated": len(rows), "min_n": int(min_n),
            "max_abs_ic": float(max((abs(r["ic"]) for r in rows), default=0.0))}


# --------------------------------------------------------------------------- #
# 4. グラフ構造
# --------------------------------------------------------------------------- #

def structure_stats(meta: pd.DataFrame, node_feat: np.ndarray,
                    edge_feat: np.ndarray) -> Dict:
    mi = schema.NODE_FEATURES.index("is_missing")
    sc = schema.NODE_FEATURES.index("scaled")
    sg = schema.NODE_FEATURES.index("sign")
    ri = schema.EDGE_FEATURES.index("ratio")
    ss = schema.EDGE_FEATURES.index("same_sign")
    bp = schema.EDGE_FEATURES.index("both_present")

    # --- CF の符号パターン --- #
    # 営業+ / 投資- / 財務- は「本業で稼ぎ、投資し、返済している」型。
    # 会計フローグラフが本当に会計の型を捉えているなら、ここに偏りが出る
    idx = [schema.NODE_INDEX[k] for k in ("cfo", "cfi", "cff")]
    signs = node_feat[:, 0, idx, sg]
    present = (node_feat[:, 0, idx, mi] == 0).all(axis=1)
    pat = {}
    if present.any():
        s = signs[present]
        keys = ["".join("+" if x > 0 else "-" if x < 0 else "0" for x in row)
                for row in s]
        vc = pd.Series(keys).value_counts()
        pat = {str(k): {"n": int(v), "pct": float(v / len(keys) * 100)}
               for k, v in vc.items()}

    # --- エッジごとの比率 --- #
    edges = []
    for i, e in enumerate(schema.EDGES):
        ok = edge_feat[:, 0, i, bp] > 0
        r = edge_feat[ok, 0, i, ri]
        edges.append({
            "src": e.src, "dst": e.dst, "kind": e.kind,
            "n": int(ok.sum()),
            "same_sign_pct": _pct(edge_feat[ok, 0, i, ss] > 0) if ok.any() else None,
            "ratio": quantiles(r),
        })

    # --- 利益の質 — 営業CF と 営業利益 の比 --- #
    # 減価償却費が取れないので発生主義との差を分解はできないが、
    # 「営業利益ほどには現金が入っていない」かどうかは見える
    jo, jc = schema.NODE_INDEX["op"], schema.NODE_INDEX["cfo"]
    ok = (node_feat[:, 0, jo, mi] == 0) & (node_feat[:, 0, jc, mi] == 0)
    op_s = node_feat[ok, 0, jo, sc]
    cfo_s = node_feat[ok, 0, jc, sc]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.abs(op_s) > 1e-6, cfo_s / op_s, np.nan)
    quality = {"n": int(ok.sum()), "quantiles": quantiles(ratio),
               "hist": histogram(np.clip(ratio, -5, 5))}

    return {"cf_sign_patterns": pat, "edges": edges, "cfo_to_op": quality,
            "cf_available_pct": _pct(present)}


# --------------------------------------------------------------------------- #

def sanitize(obj):
    """
    NaN / Inf を None に、numpy のスカラを素の Python 型に落とす。

    json.dump は既定で NaN をそのまま書くが、それは JSON の仕様外で、
    読み手（JavaScript の JSON.parse など）が構文エラーで落ちる。
    集計に欠測が出るのは正常なので、書き出す前にここで畳む。
    """
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run(data_dir: str = build.OUT_DIR, liquid_only: bool = True,
        horizon: int = 20, ic_min_n: int = IC_MIN_N) -> Dict:
    meta, nf, ef, pm = build.load(data_dir, liquid_only=liquid_only)
    meta = meta.reset_index(drop=True)
    print(f"[eda] {len(meta):,}サンプル / {meta['Code'].nunique():,}銘柄")

    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_samples": int(len(meta)),
        "n_codes": int(meta["Code"].nunique()),
        "seq_len": int(pm.shape[1]),
        "liquid_only": bool(liquid_only),
        "period": [str(pd.to_datetime(meta["entry_date"]).min().date()),
                   str(pd.to_datetime(meta["entry_date"]).max().date())],
        "schema": {
            "n_nodes": schema.N_NODES, "n_edges": schema.N_EDGES,
            "node_features": schema.NODE_FEATURES,
        },
    }
    print("[eda] 1/4 取得可能性")
    out["availability"] = availability(meta, nf, ef, pm)
    print("[eda] 2/4 目的変数")
    out["label"] = label_stats(meta)
    print("[eda] 3/4 特徴量")
    out["features"] = feature_stats(meta, nf)
    out["ic"] = information_coefficient(meta, nf, horizon=horizon,
                                        min_n=ic_min_n)
    print("[eda] 4/4 グラフ構造")
    out["structure"] = structure_stats(meta, nf, ef)
    return sanitize(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="会計フローグラフの EDA 集計")
    ap.add_argument("--data-dir", default=build.OUT_DIR)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--all-stocks", action="store_true",
                    help="流動性フィルタを外す")
    ap.add_argument("--ic-min-n", type=int, default=IC_MIN_N,
                    help="情報係数を出すのに必要な最小サンプル数")
    args = ap.parse_args(argv)

    d = run(args.data_dir, liquid_only=not args.all_stocks, horizon=args.horizon,
            ic_min_n=args.ic_min_n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[eda] 書き出し {args.out} "
          f"({os.path.getsize(args.out) / 1000:.0f}KB)")
    if d.get("ic"):
        print(f"[eda] 単変量の情報係数 最大 |IC| = {d['ic']['max_abs_ic']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
