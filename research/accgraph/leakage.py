#!/usr/bin/env python3
"""
未来情報リークの検査。

「リークしていないつもり」は検査ではない。実際に破れる形の主張を並べ、
1つでも破れたら止める。予測精度が想定より高いときに真っ先に疑う場所なので、
モデルを触る前に必ず通す。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import panel, schema


class LeakCheckError(AssertionError):
    pass


def _fail(results: List[Dict], name: str, ok: bool, detail: str) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def check_labels(meta: pd.DataFrame, horizons=(5, 10, 20)) -> List[Dict]:
    """ラベル側の時間整合。"""
    res: List[Dict] = []
    d = pd.to_datetime(meta["disc_date"])
    e = pd.to_datetime(meta["entry_date"])

    bad = int((e <= d).sum())
    _fail(res, "エントリーは発表日より後の営業日", bad == 0,
          f"発表日以前にエントリーしている行 {bad}件")

    if "label_ready_date" in meta.columns:
        r = pd.to_datetime(meta["label_ready_date"])
        bad = int((r < e).sum())
        _fail(res, "ラベル確定日はエントリー日以降", bad == 0,
              f"逆転している行 {bad}件")

    # 同じ (銘柄, 期) が2回以上サンプルになっていないか。
    # 訂正開示をアンカーにすると同じ決算が二重に入り、
    # 訓練とテストの両方に同じ事象が現れる
    dup = int(meta.duplicated(["Code", "per_end"]).sum())
    _fail(res, "同じ銘柄・同じ会計期間のサンプルは1件", dup == 0,
          f"重複 {dup}件")

    for h in horizons:
        col = f"excess_topix_{h}d"
        if col not in meta.columns:
            continue
        v = meta[col].dropna()
        if v.empty:
            continue
        # 超過リターンが ±100% を超えるのは、分割調整の失敗や
        # ベンチマークの取り違えを疑う水準
        extreme = int((v.abs() > 1.0).sum())
        _fail(res, f"超過リターン({h}営業日)が現実的な範囲",
              extreme / len(v) < 0.01,
              f"|超過| > 100% が {extreme}件 ({extreme / len(v) * 100:.2f}%)")
    return res


def check_asof(versions: pd.DataFrame, periods: pd.DataFrame,
               anchor_df: pd.DataFrame, n_lags: int = panel.N_LAGS,
               sample: int = 2000, seed: int = 0) -> List[Dict]:
    """
    as-of 展開が「基準日より後に開示された版」を拾っていないか。

    無作為に抽出したアンカーについて、引かれた版の開示日が
    アンカーの開示日以下であることを直接確かめる。
    """
    res: List[Dict] = []
    rng = np.random.default_rng(seed)
    take = anchor_df if len(anchor_df) <= sample else anchor_df.iloc[
        np.sort(rng.choice(len(anchor_df), sample, replace=False))]
    long = panel.asof_lags(take.reset_index(drop=True), versions, periods,
                           n_lags=n_lags)
    src = pd.to_datetime(long["src_disc_date"])
    anchor = pd.to_datetime(long["disc_ts"]).dt.normalize()
    future = int((src.notna() & (src > anchor)).sum())
    _fail(res, "履歴に使う開示は基準日以前のものだけ", future == 0,
          f"基準日より後の開示を引いた行 {future}件 / {len(long):,}行")

    # 期の並びが逆行していないか（lag が大きいほど古い期であること）
    pe = pd.to_datetime(long["per_end"])
    piv = pe.to_numpy().reshape(len(take), n_lags)
    # 欠測（存在しない期）は NaN のまま差分を取り、nansum で無視する
    diffs = np.diff(piv.astype("float64"), axis=1)
    bad = int(np.nansum(diffs > 0))
    _fail(res, "ラグが大きいほど古い会計期間", bad == 0,
          f"順序が逆転している箇所 {bad}件")
    return res


def check_features(node_feat: np.ndarray, period_mask: np.ndarray,
                   meta: pd.DataFrame) -> List[Dict]:
    """特徴量テンソル側の検査。"""
    res: List[Dict] = []
    n, t, n_nodes, n_feat = node_feat.shape
    _fail(res, "テンソルの形がスキーマと一致",
          n_nodes == schema.N_NODES and n_feat == len(schema.NODE_FEATURES),
          f"node_feat={node_feat.shape} / スキーマ=({schema.N_NODES}, "
          f"{len(schema.NODE_FEATURES)})")
    _fail(res, "サンプル数が meta と一致", n == len(meta),
          f"node_feat {n} vs meta {len(meta)}")

    bad = int(np.isnan(node_feat).sum() + np.isinf(node_feat).sum())
    _fail(res, "NaN / Inf が残っていない", bad == 0, f"{bad}箇所")

    # 欠測の期は、is_missing 以外の特徴量が 0 で埋まっていること。
    # ここに値が残っていると「存在しない四半期」から学習してしまう
    mi = schema.NODE_FEATURES.index("is_missing")
    others = [i for i in range(n_feat) if i != mi]
    absent = ~period_mask
    if absent.any():
        vals = node_feat[absent][:, :, others]
        _fail(res, "存在しない四半期の特徴量は0", float(np.abs(vals).max()) == 0.0,
              f"最大 |値| = {float(np.abs(vals).max()):.6g}")
    else:
        _fail(res, "存在しない四半期の特徴量は0", True, "該当なし")

    # 当該決算（T=0）が欠測のサンプルは、そもそも予測の材料が無い
    if t > 0:
        no_current = int((~period_mask[:, 0]).sum())
        _fail(res, "当該四半期が必ず存在する", no_current == 0,
              f"当該四半期が欠測のサンプル {no_current}件")
    return res


def run_all(meta: pd.DataFrame, node_feat: Optional[np.ndarray] = None,
            period_mask: Optional[np.ndarray] = None,
            versions: Optional[pd.DataFrame] = None,
            periods: Optional[pd.DataFrame] = None,
            anchor_df: Optional[pd.DataFrame] = None,
            raise_on_fail: bool = True) -> List[Dict]:
    res = check_labels(meta)
    if node_feat is not None and period_mask is not None:
        res += check_features(node_feat, period_mask, meta)
    if versions is not None and periods is not None and anchor_df is not None:
        res += check_asof(versions, periods, anchor_df)

    print("[leak] リーク検査")
    for r in res:
        mark = "OK  " if r["ok"] else "NG  "
        print(f"  {mark}{r['check']}: {r['detail']}")
    failed = [r for r in res if not r["ok"]]
    if failed and raise_on_fail:
        raise LeakCheckError(
            "リーク検査に失敗しました: "
            + " / ".join(f"{r['check']} ({r['detail']})" for r in failed))
    return res
