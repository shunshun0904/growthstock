#!/usr/bin/env python3
"""Build the independent TOPIX Core30 galaxy atlas from *all* saved daily bars.

Usage:
    python3 scripts/build_core30_galaxy.py --data-dir research/_data \
        --output dist/galaxy/data.json

No day sampling is performed. Points are individual days with valid adjusted
closes. A common PCA maps their 32 technical indicators to local 3D offsets;
UMAP maps stock-level feature summaries to galaxy centers. Neighbors are ranked
in the uncompressed summary space, not from apparent screen distances.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import technicals as TX  # noqa: E402

# JPX TOPIX New Index Series, effective 2026-10-30.
CORE30 = [
    ("2914", "日本たばこ産業", "食品"),
    ("4063", "信越化学工業", "化学"),
    ("4502", "武田薬品工業", "医薬品"),
    ("5803", "フジクラ", "非鉄金属"),
    ("6098", "リクルートHD", "サービス"),
    ("6501", "日立製作所", "電気機器"),
    ("6503", "三菱電機", "電気機器"),
    ("6752", "パナソニックHD", "電気機器"),
    ("6758", "ソニーグループ", "電気機器"),
    ("6857", "アドバンテスト", "電気機器"),
    ("6861", "キーエンス", "電気機器"),
    ("6981", "村田製作所", "電気機器"),
    ("7011", "三菱重工業", "機械"),
    ("7203", "トヨタ自動車", "輸送用機器"),
    ("7741", "HOYA", "精密機器"),
    ("7974", "任天堂", "その他製品"),
    ("8001", "伊藤忠商事", "卸売業"),
    ("8031", "三井物産", "卸売業"),
    ("8035", "東京エレクトロン", "電気機器"),
    ("8053", "住友商事", "卸売業"),
    ("8058", "三菱商事", "卸売業"),
    ("8306", "三菱UFJ FG", "銀行"),
    ("8316", "三井住友FG", "銀行"),
    ("8411", "みずほFG", "銀行"),
    ("8766", "東京海上HD", "保険"),
    ("9432", "NTT", "通信"),
    ("9433", "KDDI", "通信"),
    ("9434", "ソフトバンク", "通信"),
    ("9983", "ファーストリテイリング", "小売業"),
    ("9984", "ソフトバンクグループ", "通信"),
]

FEATURES = [
    "tx_ma_gap_5", "tx_ma_gap_25", "tx_ma_gap_75", "tx_ma_gap_200",
    "tx_ma_slope_25", "tx_ma_slope_75", "tx_ma_ratio_5_25", "tx_macd_hist",
    "tx_adx_14", "tx_pdi_14", "tx_mdi_14", "tx_ichi_cloud_pos",
    "tx_linreg_slope_20", "tx_kaufman_er_20", "tx_donchian_pos_20",
    "tx_rsi_9", "tx_rsi_14", "tx_stoch_k", "tx_williams_r_14",
    "tx_roc_10", "tx_roc_60", "tx_cmo_14", "tx_natr_14", "tx_bb_pctb",
    "tx_bb_width", "tx_vol_60", "tx_chop_14", "tx_cmf_20",
    "tx_mfi_14", "tx_vol_osc_5_20", "tx_vwap_gap_20",
    "tx_tv_ratio_20_120",
]


def load_bars(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "bars_*.parquet")))
    if not paths:
        raise FileNotFoundError("bars_*.parquet がありません: " + data_dir)
    valid = {code for code, _, _ in CORE30}
    parquet_codes = [v for c in valid for v in (c, c + "0")]
    frames = []
    for path in paths:
        # Read just the 30 companies, not all ~4,000 Japanese stocks.
        try:
            p = pd.read_parquet(path, filters=[("Code", "in", parquet_codes)])
        except (TypeError, ValueError, OSError) as exc:
            raise RuntimeError(f"{path} の対象銘柄フィルターを読めません: {exc}") from exc
        if p.empty:
            print("No Core30 bars in", os.path.basename(path), file=sys.stderr)
            continue
        p["Code"] = p["Code"].astype(str).str.strip().str[:4]
        p = p[p["Code"].isin(valid)]
        frames.append(p)
    if not frames:
        raise ValueError("Core30の日足データが取得できません")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Code", "Date"])
    df = df.drop_duplicates(["Code", "Date"], keep="last").reset_index(drop=True)
    return df


def adjusted(p: pd.DataFrame, adjusted_col: str, raw_col: str) -> pd.Series:
    raw = pd.to_numeric(p[raw_col], errors="coerce")
    if adjusted_col in p:
        adj = pd.to_numeric(p[adjusted_col], errors="coerce")
        return adj.where(np.isfinite(adj) & (adj > 0), raw)
    return raw


def stock_panel(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    # Keep every day with a real traded closing price. No n-day stepping.
    p["close"] = adjusted(p, "AdjC", "C")
    p["open"] = adjusted(p, "AdjO", "O")
    p["high"] = adjusted(p, "AdjH", "H")
    p["low"] = adjusted(p, "AdjL", "L")
    p["vol"] = adjusted(p, "AdjVo", "Vo")
    p["trading_value"] = pd.to_numeric(p.get("Va", pd.Series(np.nan, index=p.index)),
                                        errors="coerce")
    p = p[np.isfinite(p["close"]) & (p["close"] > 0)].copy().reset_index(drop=True)
    if p.empty:
        return p
    out = TX.compute(p)
    for col in FEATURES:
        p[col] = pd.to_numeric(out[col], errors="coerce").to_numpy()
    return p


def generate(data_dir: str) -> dict:
    import umap

    bars = load_bars(data_dir)
    panels = []
    for code, name, sector in CORE30:
        p = stock_panel(bars[bars["Code"] == code])
        if len(p) < 252:
            raise ValueError(f"{code} {name}: 観測点が不足しています（{len(p)}件）")
        if p["Date"].duplicated().any():
            raise AssertionError(code + ": 重複日付")
        panels.append(p)
        print(f"{code} {name}: {len(p)} trading days ({p['Date'].iloc[0].date()} ~ {p['Date'].iloc[-1].date()})")

    # Shared transformation across all stocks/days; missing *indicator values*
    # in lookback warm-up rows are imputed, but no price points are discarded.
    sizes = [len(p) for p in panels]
    X = np.vstack([p[FEATURES].to_numpy(dtype=float) for p in panels])
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0
    X = np.where(np.isfinite(X), X, med)
    Z = np.clip(StandardScaler().fit_transform(X), -8.0, 8.0)
    zblocks = np.split(Z, np.cumsum(sizes)[:-1])
    pca = PCA(n_components=3, random_state=42)
    local = pca.fit_transform(Z)
    localblocks = np.split(local, np.cumsum(sizes)[:-1])
    radius = np.percentile(np.linalg.norm(local, axis=1), 95)
    point_scale = 3.7 / max(radius, 1e-6)

    # Summary over every available year, 32 indicators x [mean, std, 75th percentile].
    summary = np.asarray([
        np.concatenate([z.mean(axis=0), z.std(axis=0), np.percentile(z, 75, axis=0)])
        for z in zblocks
    ])
    summary_scaled = StandardScaler().fit_transform(summary)
    summary_dist = pairwise_distances(summary_scaled)
    centers = umap.UMAP(n_components=3, n_neighbors=8, min_dist=0.35,
                        random_state=42, init="spectral").fit_transform(summary_scaled)
    centers = centers - centers.mean(axis=0)
    center_dist = pairwise_distances(centers)
    np.fill_diagonal(center_dist, np.inf)
    nearest_gap = np.median(center_dist.min(axis=1))
    centers *= 22.0 / max(nearest_gap, 1e-6)

    stocks = []
    all_dates = []
    total_points = 0
    for i, ((code, name, sector), p, local3) in enumerate(zip(CORE30, panels, localblocks)):
        # Translate each PCA cloud to its UMAP galaxy center, preserving the
        # shared PCA axes and scaling; do not manufacture sampling/extra dates.
        offsets = (local3 - local3.mean(axis=0)) * point_scale
        values = p["close"].to_numpy(dtype=float)
        rets = np.zeros(len(values), dtype=float)
        rets[1:] = np.where(values[:-1] > 0,
                            (values[1:] / values[:-1] - 1) * 100, 0)
        dates = p["Date"].dt.strftime("%Y%m%d").astype(int).tolist()
        all_dates.extend([dates[0], dates[-1]])
        order = np.argsort(summary_dist[i] + np.eye(len(CORE30))[i] * 1e8)
        neighbors = [
            {"code": CORE30[j][0], "distance": round(float(summary_dist[i, j]), 4)}
            for j in order[:5]
        ]
        points = [
            [int(date), round(float(price), 3),
             round(float(x), 3), round(float(y), 3), round(float(z), 3),
             round(float(ret), 3)]
            for date, price, (x, y, z), ret in zip(dates, values, offsets, rets)
        ]
        if len(points) != len(p):
            raise AssertionError("日足が間引かれています: " + code)
        total_points += len(points)
        stocks.append({
            "code": code, "name": name, "sector": sector,
            "center": [round(float(x), 3) for x in centers[i]],
            "neighbors": neighbors, "count": len(points),
            "from": dates[0], "to": dates[-1], "points": points,
        })

    if len(stocks) != 30:
        raise AssertionError("Core30の銘柄数が30ではありません")
    return {
        "meta": {
            "universe": "TOPIX Core30",
            "constituentsAsOf": "2026-10-30",
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stockCount": len(stocks), "pointCount": total_points,
            "from": min(all_dates), "to": max(all_dates),
            "features": FEATURES, "featureCount": len(FEATURES),
            "localEmbedding": "shared PCA 3D (all daily points)",
            "galaxyEmbedding": "UMAP 3D (96-dimensional stock summaries)",
            "neighborDistance": "Euclidean in standardized 96D stock summaries",
            "pcaExplainedVariance": [
                round(float(x), 4) for x in pca.explained_variance_ratio_
            ],
            "notes": "全営業日の実測終値を使用。指標の初期欠測は特徴量中央値で補完。表示上の銀河間距離は絶対的な類似度を表さない。",
        },
        "stocks": stocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "research", "_data"))
    parser.add_argument("--output", default=os.path.join(ROOT, "public", "galaxy", "data.json"))
    args = parser.parse_args()
    result = generate(args.data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print("Galaxy data written:", args.output,
          result["meta"]["stockCount"], "stocks",
          result["meta"]["pointCount"], "daily points")


if __name__ == "__main__":
    main()
