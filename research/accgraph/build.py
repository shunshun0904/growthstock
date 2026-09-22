#!/usr/bin/env python3
"""
会計フローグラフ系列データセットの構築。

  research/_data/*.parquet（生データ）
      -> research/_data/accgraph/accgraph_<年>.npz   グラフのテンソル
         research/_data/accgraph/meta.parquet        1サンプル1行のメタ・ラベル

グラフはスキーマが全サンプルで同一（ノード18・エッジ21の固定）なので、
サンプルごとに構造を持つ必要が無く、テンソル1本で表せる。

  node_feat  [n, T, 29, 14]   四半期ごと・ノードごとの特徴量
  edge_feat  [n, T, 37,  3]   四半期ごと・エッジごとの特徴量
  period_mask[n, T]           その四半期の決算が実在し、基準日時点で見えていたか

T は直近から過去へ向かう並び（T=0 が当該決算）。PyTorch Geometric へは
edge_index（schema.edge_index()）をそのまま渡せる。

実行:
  python3 research/accgraph/build.py                 # 全期間
  python3 research/accgraph/build.py --from 2020-01-01
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)
sys.path.insert(0, os.path.dirname(RESEARCH))

from accgraph import edinet, labels as L  # noqa: E402
from accgraph import panel, schema  # noqa: E402

DATA_DIR = os.path.join(RESEARCH, "_data")
OUT_DIR = os.path.join(DATA_DIR, "accgraph")

#: 比率系の特徴量を打ち切る範囲。売上がほぼ0の四半期で比が発散するため。
CLIP = 20.0
#: エッジの比率を打ち切る範囲
CLIP_RATIO = 20.0
EPS = 1e-9


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #

def load_parts(prefix: str, data_dir: str, required: bool = True
               ) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.parquet")))
    if not paths:
        if required:
            raise SystemExit(
                f"{prefix} の parquet が {data_dir} にありません。"
                "先に research/jq_bulk.py を実行してください")
        print(f"[load] {prefix}: 見つからないので使わない")
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"[load] {prefix}: {len(df):,}行 ({len(paths)}ファイル)")
    return df


# --------------------------------------------------------------------------- #
# ノードの金額を組み立てる
# --------------------------------------------------------------------------- #

_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(msg)


def _field_matrix(mats: Dict[str, np.ndarray], field: str,
                  source_table: str = "fins") -> np.ndarray:
    """
    項目名から、1四半期あたりの金額（フロー）または期末残高（ストック）を返す。

    EDINET 由来の項目は年次なので、edinet.asof_matrices が既に4で割って
    1四半期あたりに直してある。ここでは引くだけ。
    """
    if source_table == "edinet":
        key = f"e_{field}"
        if key not in mats:
            # EDINET を結合していない呼び出し（粗いグラフだけを見たいとき）。
            # 落とさずに欠測で返す。黙って 0 にはしない
            _warn_once(f"[build] {key} が無いので {field} は欠測にする")
            return np.full_like(mats["quarter"], np.nan)
        return mats[key]
    if field in schema.CUMULATIVE_FIELDS:
        return mats[f"q_{field}"]
    if field in schema.STOCK_FIELDS:
        return mats[f"s_{field}"]
    raise KeyError(f"{field} は累計・ストックのどちらにも分類されていない")


def node_amounts(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """[n, n_lags, n_nodes] の金額行列。開示値と恒等式による計算値の両方。"""
    n, n_lags = mats["quarter"].shape
    out = np.full((n, n_lags, schema.N_NODES), np.nan)
    for j, node in enumerate(schema.NODES):
        if node.source == "disclosed":
            out[:, :, j] = _field_matrix(mats, node.field, node.source_table)
            continue
        # 恒等式による計算値。構成項目が1つでも欠測なら結果も欠測にする。
        # 欠測を 0 とみなすと「負債 = 総資産」のような嘘の値ができる
        plus, minus = node.derive
        acc = np.zeros((n, n_lags))
        for f in plus:
            v = _field_matrix(mats, f, node.source_table)
            acc = acc + np.nan_to_num(v)
            acc = np.where(np.isnan(v), np.nan, acc)
        for f in minus:
            v = _field_matrix(mats, f, node.source_table)
            acc = acc - np.nan_to_num(v)
            acc = np.where(np.isnan(v), np.nan, acc)
        out[:, :, j] = acc
    return out


def node_spans(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    [n, n_lags, n_nodes] の「その金額が何四半期ぶんか」。

    計算値ノードは構成項目の期数が揃っている前提だが、揃わないことも
    あり得る（片方だけ半期開示など）。その場合は長いほうを採る。
    値そのものは1四半期あたりに直してあるので足し引きしてよいが、
    「どの粒度の値か」は長いほうに引きずられるため。
    """
    n, n_lags = mats["quarter"].shape
    out = np.full((n, n_lags, schema.N_NODES), np.nan)
    for j, node in enumerate(schema.NODES):
        fields = ([node.field] if node.field
                  else list(node.derive[0]) + list(node.derive[1]))
        acc = None
        for f in fields:
            if node.source_table == "edinet":
                # 年次のフローは1年＝4四半期ぶん。ストックは期末残高なので1
                v = np.where(np.isnan(_field_matrix(mats, f, "edinet")), np.nan,
                             schema.ANNUAL_TO_QUARTER
                             if f in schema.EDINET_FLOW_FIELDS else 1.0)
            else:
                v = mats[f"span_{f}"]
            acc = v if acc is None else np.fmax(acc, v)
        out[:, :, j] = acc
    return out


def node_cumulative(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """進捗乖離に使う累計値。開示値ノードのみ、他は欠測。"""
    n, n_lags = mats["quarter"].shape
    out = np.full((n, n_lags, schema.N_NODES), np.nan)
    for j, node in enumerate(schema.NODES):
        if node.source == "disclosed" and node.field in schema.CUMULATIVE_FIELDS:
            out[:, :, j] = mats[f"cum_{node.field}"]
    return out


# --------------------------------------------------------------------------- #
# 系列の統計量
# --------------------------------------------------------------------------- #

def _lag_window(a: np.ndarray, width: int) -> np.ndarray:
    """[n, n_lags, ...] から、各位置について過去 width 期を並べた軸を足す。"""
    n_lags = a.shape[1]
    out = np.full((width,) + a.shape, np.nan)
    for k in range(width):
        out[k, :, :n_lags - k] = a[:, k:]
    return out


def slope_and_vol(scaled: np.ndarray, width: int = 4):
    """
    直近 width 期の傾きとばらつき。

    傾きは「1四半期あたりの変化」。時間は新しいほうを正にとるので、
    過去へさかのぼる軸 k に対して t = -k を回帰の説明変数にする。
    3期以上そろっていないと傾きは出さない（2点だと外れ値と区別できない）。
    """
    win = _lag_window(scaled, width)              # [width, n, n_lags, n_nodes]
    t = (-np.arange(width, dtype=np.float64)).reshape(
        (width,) + (1,) * (scaled.ndim))
    m = ~np.isnan(win)
    cnt = m.sum(axis=0)
    y = np.where(m, win, 0.0)
    tt = np.broadcast_to(t, win.shape)
    tm = np.where(m, tt, 0.0)

    sx = tm.sum(axis=0)
    sy = y.sum(axis=0)
    sxx = (tm * tm).sum(axis=0)
    sxy = (tm * y).sum(axis=0)
    denom = cnt * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((cnt >= 3) & (np.abs(denom) > EPS),
                         (cnt * sxy - sx * sy) / denom, np.nan)
        mean = np.where(cnt >= 3, sy / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt >= 3,
                       ((np.where(m, win - mean, 0.0) ** 2).sum(axis=0)
                        / np.maximum(cnt, 1)), np.nan)
    return slope, np.sqrt(var)


# --------------------------------------------------------------------------- #
# 特徴量
# --------------------------------------------------------------------------- #

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(b) > EPS, a / b, np.nan)
    return np.clip(out, -CLIP, CLIP)


def node_features(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """[n, n_lags, n_nodes, len(NODE_FEATURES)] を作る。"""
    amount = node_amounts(mats)
    cum = node_cumulative(mats)
    n, n_lags, n_nodes = amount.shape

    # 正規化の分母。PL・CF のフローは売上高、BS のストックは総資産で割る。
    # 分母を項目ごとに変えると企業間で比較できなくなるので、ノードごとに固定する
    sales = np.abs(mats["q_Sales"])[:, :, None]
    assets = np.abs(mats["s_TA"])[:, :, None]
    use_assets = np.array(
        [nd.scale_by == schema.SCALE_ASSETS for nd in schema.NODES]
    ).reshape(1, 1, n_nodes)
    denom = np.where(use_assets, assets, sales)

    scaled = _safe_div(amount, denom)
    to_sales = _safe_div(amount, sales)
    to_assets = _safe_div(amount, assets)

    with np.errstate(invalid="ignore"):
        log_size = np.sign(amount) * np.log1p(np.abs(amount) / 1e6)

    yoy_ok = panel.yoy_valid(mats["per_end_days"])[:, :, None]
    qoq_ok = panel.qoq_valid(mats["per_end_days"])[:, :, None]
    yoy = _sym_change_3d(amount, 4, yoy_ok)
    qoq = _sym_change_3d(amount, 1, qoq_ok)

    slope, vol = slope_and_vol(scaled, width=4)

    sign = np.where(np.isnan(amount), np.nan, np.sign(amount))
    span = np.where(np.isnan(amount), np.nan, node_spans(mats))

    # その値が何年前の書類か。四半期のノードは当期そのものなので 0。
    # EDINET のノードは年1回しか出ないので、最大で1年ぶん古くなる
    is_edinet = np.array([nd.source_table == "edinet" for nd in schema.NODES]
                         ).reshape(1, 1, n_nodes)
    age = mats.get("e_age_years")
    age_years = np.where(is_edinet,
                         age[:, :, None] if age is not None else np.nan, 0.0)
    age_years = np.where(np.isnan(amount), np.nan, age_years)

    # --- 通期会社予想に対する進捗の乖離 --- #
    # 「今の累計 / 通期予想」が「経過四半期 / 4」をどれだけ上回っているか。
    # アナリスト予想は J-Quants に無いので、会社予想で代用する。
    quarter = mats["quarter"][:, :, None]
    fcst = np.full_like(amount, np.nan)
    for j, node in enumerate(schema.NODES):
        if node.forecast_field:
            fcst[:, :, j] = mats[f"f_{node.forecast_field}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        progress = np.where(np.abs(fcst) > EPS, cum / fcst, np.nan)
    fcst_gap = np.clip(progress - quarter / 4.0, -CLIP, CLIP)
    fcst_avail = (~np.isnan(fcst_gap)).astype(np.float64)

    is_missing = np.isnan(amount).astype(np.float64)

    stack = {
        "scaled": scaled, "log_size": log_size, "to_sales": to_sales,
        "to_assets": to_assets, "yoy_sym": yoy, "qoq_sym": qoq,
        "slope4": slope, "vol4": vol, "sign": sign, "span": span,
        "age_years": age_years,
        "fcst_gap": fcst_gap, "fcst_avail": fcst_avail, "is_missing": is_missing,
    }
    feats = np.stack([stack[name] for name in schema.NODE_FEATURES], axis=-1)
    # 欠測は 0 で置く。どこが欠測かは is_missing が持っているので、
    # 0 埋めしても「値が 0」と混同されない
    return np.nan_to_num(feats, nan=0.0, posinf=CLIP, neginf=-CLIP)


def _sym_change_3d(a: np.ndarray, k: int, valid: np.ndarray) -> np.ndarray:
    """[n, n_lags, n_nodes] に対する対称変化率。"""
    prev = np.full_like(a, np.nan)
    if k < a.shape[1]:
        prev[:, :a.shape[1] - k] = a[:, k:]
    denom = np.abs(a) + np.abs(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > EPS, (a - prev) / denom, np.nan)
    out = np.where(np.isnan(a) | np.isnan(prev), np.nan, out)
    return np.where(valid, out, np.nan)


def edge_features(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """[n, n_lags, n_edges, len(EDGE_FEATURES)] を作る。"""
    amount = node_amounts(mats)
    src = np.array([schema.NODE_INDEX[e.src] for e in schema.EDGES])
    dst = np.array([schema.NODE_INDEX[e.dst] for e in schema.EDGES])
    a_src = amount[:, :, src]
    a_dst = amount[:, :, dst]

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip(np.abs(a_src) / (np.abs(a_dst) + EPS), 0.0, CLIP_RATIO)
    same_sign = np.where(np.isnan(a_src) | np.isnan(a_dst), np.nan,
                         (np.sign(a_src) == np.sign(a_dst)).astype(np.float64))
    both = (~np.isnan(a_src) & ~np.isnan(a_dst)).astype(np.float64)

    stack = {"ratio": ratio, "same_sign": same_sign, "both_present": both}
    feats = np.stack([stack[name] for name in schema.EDGE_FEATURES], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=CLIP_RATIO, neginf=0.0)


# --------------------------------------------------------------------------- #
# 組み立て
# --------------------------------------------------------------------------- #

def build(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR,
          date_from: Optional[str] = None, date_to: Optional[str] = None,
          cfg: L.LabelConfig = L.DEFAULT_LABEL,
          seq_len: int = panel.SEQ_LEN) -> pd.DataFrame:
    fins = load_parts("fins", data_dir)
    bars = load_parts("bars", data_dir)
    indices = load_parts("indices", data_dir, required=False)
    topix = load_parts("topix", data_dir, required=False)
    master_hist = load_parts("master_hist", data_dir, required=False)

    versions = panel.prepare_versions(fins)
    periods = panel.period_slots(versions)
    anchor_df = panel.anchors(versions, periods)

    if date_from:
        anchor_df = anchor_df[anchor_df["disc_date"] >= pd.Timestamp(date_from)]
    if date_to:
        anchor_df = anchor_df[anchor_df["disc_date"] <= pd.Timestamp(date_to)]
    anchor_df = anchor_df.reset_index(drop=True)
    print(f"[build] 対象アンカー {len(anchor_df):,}件")

    meta = L.build_labels(anchor_df, bars, indices, topix, master_hist, cfg)

    # ラベルが1つも付かない開示は学習にも評価にも使えない。
    # 落とした件数は report_universe が出しているので、ここでは静かに落とす
    h = cfg.primary_horizon
    keep = (meta[f"y_topix_{h}d"] >= 0) | (meta[f"y_sector_{h}d"] >= 0)
    meta = meta[keep].reset_index(drop=True)
    print(f"[build] ラベルが付いたサンプル {len(meta):,}件")
    if meta.empty:
        raise SystemExit("ラベルの付いたサンプルがありません")

    meta["liquid"] = (meta["turnover_ma20"] >= cfg.min_turnover_oku).fillna(False)
    print(f"[build] うち流動性 {cfg.min_turnover_oku}億円以上: "
          f"{int(meta['liquid'].sum()):,}件")

    kept = anchor_df.merge(meta[["anchor_id"]], on="anchor_id", how="inner")
    kept = kept.sort_values("anchor_id").reset_index(drop=True)
    meta = meta.sort_values("anchor_id").reset_index(drop=True)

    mats = panel.build_asof_matrices(kept, versions, periods)

    # EDINET の明細を、各四半期の開示時点で読めた有報から当てる。
    # 無ければ明細ノードが欠測になるだけで、粗いグラフはそのまま成立する
    fin_edinet = edinet.load(data_dir)
    ed_panel = edinet.annual_panel(fin_edinet) if fin_edinet is not None else None
    mats.update(edinet.asof_matrices(kept, mats["disc_date_days"], ed_panel))

    nf, ef = features_in_chunks(mats, seq_len)
    period_mask = (mats["available"] & ~np.isnan(mats["per_end_days"]))[:, :seq_len]

    report_node_coverage(nf, period_mask, meta)
    print(f"[build] node_feat {nf.shape} / edge_feat {ef.shape} "
          f"(float32 で {(nf.nbytes + ef.nbytes) / 1e6:.0f}MB)")
    print(f"[build] 系列の充足: 8期そろっているサンプル "
          f"{float((period_mask.sum(axis=1) == seq_len).mean()) * 100:.1f}%")

    os.makedirs(out_dir, exist_ok=True)
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    meta["year"] = pd.to_datetime(meta["entry_date"]).dt.year

    for year, part in meta.groupby("year"):
        rows = part["row"].to_numpy()
        np.savez_compressed(
            os.path.join(out_dir, f"accgraph_{int(year)}.npz"),
            node_feat=nf[rows],
            edge_feat=ef[rows],
            period_mask=period_mask[rows],
            anchor_id=part["anchor_id"].to_numpy(),
        )
    meta.drop(columns=["row"]).to_parquet(
        os.path.join(out_dir, "meta.parquet"), index=False)

    with open(os.path.join(out_dir, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "seq_len": seq_len,
            "nodes": [n.id for n in schema.NODES],
            "node_names_ja": [n.name_ja for n in schema.NODES],
            "edges": [[e.src, e.dst, e.kind] for e in schema.EDGES],
            "edge_index": schema.edge_index(),
            "node_features": schema.NODE_FEATURES,
            "node_constants": schema.NODE_CONSTANTS,
            "node_constant_values": schema.node_constants(),
            "edge_features": schema.EDGE_FEATURES,
            "edge_constants": schema.EDGE_CONSTANTS,
            "edge_constant_values": schema.edge_constants(),
            "label": {
                "horizons": list(cfg.horizons),
                "primary_horizon": cfg.primary_horizon,
                "up_threshold": cfg.up_threshold,
                "down_threshold": cfg.down_threshold,
                "class_names": list(L.CLASS_NAMES),
            },
        }, fh, ensure_ascii=False, indent=2)
    print(f"[build] 保存先 {out_dir}")
    return meta


#: 特徴量を一度に計算するアンカー数。
#:
#: 中間配列は [アンカー, ラグ13, ノード29] の倍精度が特徴量の数だけ並ぶ。
#: 14万件を一度に回すと 7GB を超えてランナーが落ちる。行で分けて、
#: 各塊を単精度に落としてから繋ぐ。結果は分けないときと同じ。
CHUNK_ROWS = 20_000


def features_in_chunks(mats: Dict[str, np.ndarray], seq_len: int,
                       chunk_rows: int = CHUNK_ROWS):
    """アンカーを塊に分けてノード・エッジ特徴量を作る（メモリのため）。"""
    n = mats["quarter"].shape[0]
    # 塊を溜めてから繋ぐと、繋ぐ瞬間に同じ大きさの配列が2本並ぶ。
    # 先に器を作って埋める
    nf = np.empty((n, seq_len, schema.N_NODES, len(schema.NODE_FEATURES)),
                  dtype=np.float32)
    ef = np.empty((n, seq_len, schema.N_EDGES, len(schema.EDGE_FEATURES)),
                  dtype=np.float32)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        part = {k: v[start:stop] for k, v in mats.items()}
        nf[start:stop] = node_features(part)[:, :seq_len]
        ef[start:stop] = edge_features(part)[:, :seq_len]
        if n > chunk_rows:
            print(f"[build] 特徴量 {stop:,}/{n:,}件")
    return nf, ef


def report_node_coverage(node_feat: np.ndarray, period_mask: np.ndarray,
                         meta: pd.DataFrame) -> None:
    """
    ノードごとの充足率を当該四半期（T=0）で出す。

    どのノードが実質的に空かは、ここを見ないと分からない。
    契約で取れない項目を「作った」つもりになっていないかの確認でもある。
    """
    mi = schema.NODE_FEATURES.index("is_missing")
    present = 1.0 - node_feat[:, 0, :, mi]
    print("[build] 当該四半期のノード充足率")
    q = pd.to_numeric(meta["quarter"], errors="coerce").to_numpy()
    for j, node in enumerate(schema.NODES):
        by_q = " / ".join(
            f"{k}Q {float(present[q == k, j].mean()) * 100:4.1f}%"
            if (q == k).any() else f"{k}Q    -"
            for k in (1, 2, 3, 4))
        kind = "開示" if node.source == "disclosed" else "計算"
        src = "EDINET" if node.source_table == "edinet" else "JQ    "
        print(f"  {node.id:<22}{src} {kind} "
              f"全体 {float(present[:, j].mean()) * 100:5.1f}%"
              f"  ({by_q})")


# --------------------------------------------------------------------------- #
# 読み出し
# --------------------------------------------------------------------------- #

def load(out_dir: str = OUT_DIR, liquid_only: bool = True):
    """
    保存したデータセットを (meta, node_feat, edge_feat, period_mask) で返す。

    流動性で絞る場合は、**年ごとのファイルを読んだ時点で落とす**。
    全件を繋いでから絞ると、一瞬だけ数GBの配列が2本並んでランナーが落ちる。
    """
    meta = pd.read_parquet(os.path.join(out_dir, "meta.parquet"))
    paths = sorted(glob.glob(os.path.join(out_dir, "accgraph_*.npz")))
    if not paths:
        raise SystemExit(f"{out_dir} にデータセットがありません")

    if liquid_only and "liquid" in meta.columns:
        meta = meta[meta["liquid"].to_numpy().astype(bool)].reset_index(drop=True)
    want = pd.Series(np.arange(len(meta)), index=meta["anchor_id"].to_numpy())

    nf, ef, pm, pos = [], [], [], []
    for p in paths:
        z = np.load(p)
        ids = z["anchor_id"]
        dest = want.reindex(ids).to_numpy()
        keep = ~pd.isna(dest)
        if not keep.any():
            continue
        nf.append(z["node_feat"][keep])
        ef.append(z["edge_feat"][keep])
        pm.append(z["period_mask"][keep])
        pos.append(dest[keep].astype(int))
    if not nf:
        raise SystemExit("meta と npz の anchor_id が1件も対応していません")

    nf = np.concatenate(nf)
    ef = np.concatenate(ef)
    pm = np.concatenate(pm)
    pos = np.concatenate(pos)
    if len(pos) != len(meta):
        raise AssertionError(
            f"meta {len(meta):,}件に対して npz から拾えたのは {len(pos):,}件")

    order = np.argsort(pos, kind="stable")
    return meta, nf[order], ef[order], pm[order]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="会計フローグラフ系列データセットの構築")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    args = ap.parse_args(argv)

    print(schema.summary())
    build(args.data_dir, args.out_dir, args.date_from, args.date_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
