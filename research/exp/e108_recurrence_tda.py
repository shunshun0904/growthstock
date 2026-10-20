#!/usr/bin/env python3
"""
実験108: リカレンス・ネットワークの「今日のまわり」と「今日までの軌跡」の形を持続ホモロジー（トポロジカルデータ解析）で数え、
その数字が +10% 到達を当てるか。検証だけ（画面・research/recurrence_net.py は変えない）。

運用者の依頼（2026-10-20）「この幾何構造に対して、トポロジカルデータアナリシスを適用し銘柄の定量評価や、該当日付近辺の構造の
定量評価をしたいです」→ 運用者の選択（AskUserQuestion）: 使い道「まず検証する」、数える範囲「今日のまわり（状態が近い日）」
「今日までの軌跡（時間が近い日）」（銘柄全体の形は選ばれなかった）。docs/MODEL_ADOPTION_RULES.md §71。

やること（実験107 と同じ対象・同じ手順で、似た日の代わりに形の数字を出す）
  対象の日: 全銘柄（一般市場。build_panel と同じ）の78週高値の更新日（is_fresh_break）のうち、20営業日後までの日足があり
           結果が分かる日。その日までのノード（9列がそろう日）が MIN_NODES（100）日以上ある日
  状態: その日 t までのノードだけで9列を変換・標準化し（recurrence_net.transform_states）、1/20σ 刻みに丸めた点（画面と同じ）
  今日のまわり: 今日と、タイラー窓（20ノード）より離れたノードのうち距離の近い LOCAL（100）日の点の雲（今日を入れて101点）
  今日までの軌跡: 直近 TRAJ（60）ノードの点の雲（時間で続いた日。輪は状態がぐるっと回って戻った跡）
  持続ホモロジー: ユークリッド距離の Vietoris–Rips 複体の H0（塊）・H1（輪）（ripser。Bauer 2021）。寿命 = 消えた距離 − 生まれた距離
  当てる量 y: t の +10% 到達（翌営業日の始値で買い、t+1〜t+20 の高値が 1.10 倍に届いたか。recurrence_net.forward_moves の
             rise ≥ 10。実験107 と同じ）

数える4つの量（回す前に決めた。どれも σ の単位）
  local_h1_total  今日のまわりの輪（H1）の寿命の合計
  local_h0_gap    今日のまわりの塊（H0）の寿命の最大（いちばん大きな割れ目。Vietoris–Rips の H0 の死は最小全域木の辺の長さ）
  traj_h1_land    軌跡の輪のランドスケープの L1 ノルム（すべての段の和 = Σ 寿命² / 4。Gidea・Katz arXiv:1703.04385 の量。
                  Bubenik 2015 のランドスケープ）
  traj_h1_max     軌跡の輪の寿命の最大（周期らしさ。Perea・Harer 2015 の最大の持続）

判定（回す前に決めた。2026-10-20。運用者への報告にも書いた）
  主な物差しは、同じ銘柄の日どうしの組だけで数える AUC（実験107 で「どの銘柄か」の影響が大きかったので、銘柄の中で当たるかを
  主にした）。4つ同時に見るので、銘柄を選び直すブートストラップ（1,000回・種 0）の区間を 98.75%（1 − 0.05/4）にする。区間が
  0.5 をまたがず、かつ AUC が 0.52 以上か 0.48 以下（向きは問わない）なら「効く」→ 画面に出すかは運用者が決める。
  参考（判定には使わない）: 全日まとめた AUC、5つに分けた段ごとの実際の到達、全期間の到達率・5番目に近い日までの距離（画面の
  珍しさのもと）との順位相関

結果（2026-10-20。手元で回した。日足は Release data-raw の 2026-10-19 時点。docs/MODEL_ADOPTION_RULES.md §71-2）: 評価した日
28,222（4,428銘柄・2018-04〜2026-09）。銘柄の中だけの AUC は 今日のまわり・輪の合計 0.5116（98.75% 区間 0.4990〜0.5242）・最大の
割れ目 0.5183（0.5044〜0.5309。0.5 からの離れが 0.02 に届かない）・軌跡・ランドスケープ L1 0.4966・最大の輪 0.4979 → 4つとも
「効かない」。最大の割れ目は5番目に近い日までの距離（画面の珍しさのもと）と順位相関 +0.77 で、その距離の AUC 0.5180 とほぼ同じ。

出すもの（公開ログには件数・割合・AUC だけ。銘柄・値は出さない）

    python3 research/exp/e108_recurrence_tda.py --data-dir research/_data --out research/_data/oof/e108_rows.parquet
ripser が要る（pip install ripser。本番の research/requirements.txt には入れない）。集計の関数は ripser なしで読める
（tests/test_e108_recurrence_tda.py）。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import recurrence_net as RN  # noqa: E402

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)    # build_panel が列を1本ずつ足す警告（結果は同じ）

COLS: Tuple[str, ...] = tuple(RN.FEATURES)       # 9列（運用者の決定 §70-5 で9列のまま）
TH, MIN_NODES = RN.THEILER, RN.MIN_NODES
LOCAL = 100
TRAJ = 60
KTH = 5
QUANTITIES = ("local_h1_total", "local_h0_gap", "traj_h1_land", "traj_h1_max")
LABELS = {"local_h1_total": "今日のまわり・輪の合計", "local_h0_gap": "今日のまわり・最大の割れ目",
          "traj_h1_land": "軌跡・輪のランドスケープ L1", "traj_h1_max": "軌跡・最大の輪"}
CHUNK = 100
SEED = 0
N_BOOT = 1000
LEVEL = 1 - 0.05 / len(QUANTITIES)              # 98.75%
EFFECT = 0.02                                    # AUC が 0.5 からこれ以上離れていること


def log(msg: str) -> None:
    print(f"[e108] {msg}", flush=True)


def summaries(h0: np.ndarray, h1: np.ndarray) -> Dict[str, float]:
    """
    持続図（H0・H1。行 = [生まれた距離, 消えた距離]）の数字。H0 の消えない1本（無限）は数えない。
      h0_gap   H0 の寿命の最大（いちばん大きな割れ目）
      h1_total H1 の寿命の合計
      h1_max   H1 の寿命の最大
      h1_land  H1 のランドスケープの L1 ノルム（すべての段の和）= Σ 寿命² / 4（三角形の面積の和）
    輪が無ければ H1 の数字は 0。
    """
    h0 = np.asarray(h0, dtype=float).reshape(-1, 2)
    h1 = np.asarray(h1, dtype=float).reshape(-1, 2)
    l0 = h0[np.isfinite(h0[:, 1]), 1] - h0[np.isfinite(h0[:, 1]), 0]
    l1 = h1[np.isfinite(h1[:, 1]), 1] - h1[np.isfinite(h1[:, 1]), 0]
    return {"h0_gap": float(l0.max()) if len(l0) else 0.0,
            "h1_total": float(l1.sum()), "h1_max": float(l1.max()) if len(l1) else 0.0,
            "h1_land": float((l1 ** 2).sum() / 4.0)}


def local_index(s2: np.ndarray, m: int, size: int = LOCAL) -> np.ndarray:
    """今日（最後のノード）と、タイラー窓より離れたノードのうち距離の2乗 s2 の近い size 日（同じ距離なら古い日が先）の位置。"""
    pos = np.arange(m)
    c = pos[pos < m - 1 - TH]
    near = c[np.lexsort((c, s2[c]))][:size]
    return np.r_[near, m - 1]


def traj_index(m: int, size: int = TRAJ) -> np.ndarray:
    """直近 size ノードの位置（今日を含む）。"""
    return np.arange(max(0, m - size), m)


def persistence(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """点の雲の Vietoris–Rips の持続図（H0・H1）。"""
    from ripser import ripser
    dg = ripser(np.asarray(X, dtype=float), maxdim=1)["dgms"]
    return dg[0], dg[1]


def day_features(X: np.ndarray, ok: np.ndarray, t: int) -> Optional[Dict[str, float]]:
    """パネルの行 t を今日として、t までのノードで形の数字を出す。ノードが MIN_NODES 未満・t がノードでなければ None。"""
    idx = np.flatnonzero(ok[:t + 1])
    if len(idx) < MIN_NODES or idx[-1] != t:
        return None
    Z, _ = RN.transform_states(X[idx], COLS)
    zq = RN.quantize(Z).astype(np.int64)
    m = len(idx)
    s2 = ((zq - zq[-1]) ** 2).sum(1)
    pts = zq / float(RN.Q)
    lo = summaries(*persistence(pts[local_index(s2, m)]))
    tr = summaries(*persistence(pts[traj_index(m)]))
    pos = np.arange(m)
    c = pos[pos < m - 1 - TH]
    kth = float(np.sqrt(np.sort(s2[c])[KTH - 1]) / RN.Q) if len(c) >= KTH else np.nan
    return {"local_h1_total": lo["h1_total"], "local_h0_gap": lo["h0_gap"],
            "traj_h1_land": tr["h1_land"], "traj_h1_max": tr["h1_max"], "kth": kth, "_idx": idx}


def run_chunk(args) -> Tuple[List[dict], float]:
    """1チャンクの銘柄について、更新日ごとの行（y・形の数字・全期間の到達率・5番目に近い日までの距離）を返す。"""
    codes, data_dir = args
    t0 = time.time()
    segments = B.load_market_segments(data_dir)
    bars = RN.load_bars(data_dir, codes)
    panel = RN.build_panel(bars, segments)
    bars_by = {c: g for c, g in bars.groupby("Code", sort=False)}
    rows: List[dict] = []
    for code, g in panel.groupby("Code", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        fw = RN.forward_moves(bars_by[code], g["Date"])
        hit = np.where(np.isfinite(fw["rise"]), (fw["rise"] >= 10 - 1e-9).astype(float), np.nan)
        X = g[list(COLS)].to_numpy(float)
        ok = np.isfinite(X).all(1)
        fresh = g["is_fresh_break"].fillna(False).to_numpy(bool)
        for t in np.flatnonzero(fresh & ok & np.isfinite(hit)):
            f = day_features(X, ok, t)
            if f is None:
                continue
            idx = f.pop("_idx")
            past = idx[: max(0, len(idx) - 1 - TH)]          # 21ノード以上前（結果が t までに分かる）
            hp = hit[past]
            f["base"] = float(np.nanmean(hp)) if np.isfinite(hp).any() else np.nan
            rows.append({"code": code, "date": g["Date"].iloc[t], "y": hit[t], **f})
    return rows, time.time() - t0


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC（Mann–Whitney。同じ値は 0.5）。"""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def within_parts(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """銘柄ごとの「銘柄の中だけの AUC」の分子と分母（同じ銘柄の 到達した日 × 届かなかった日 の組の数。同じ値は 0.5）。"""
    nums, dens = [], []
    for _, g in df.groupby("code", sort=False):
        s = g[col].to_numpy(float)
        ok = np.isfinite(s)
        y, s = g["y"].to_numpy(float)[ok], s[ok]
        n1 = y.sum()
        n0 = len(y) - n1
        if not n1 or not n0:
            continue
        r = pd.Series(s).rank(method="average").to_numpy()
        nums.append(r[y == 1].sum() - n1 * (n1 + 1) / 2)
        dens.append(n1 * n0)
    return np.asarray(nums, dtype=float), np.asarray(dens, dtype=float)


def within_auc(df: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = SEED,
               level: float = LEVEL) -> Tuple[float, float, float]:
    """銘柄の中だけの AUC と、銘柄を選び直すブートストラップの level 区間。"""
    num, den = within_parts(df, col)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(den), np.full(len(den), 1 / len(den)), size=n_boot).astype(float)
    b = (w @ num) / (w @ den)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(b, [a, 100 - a])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def verdict(a: float, lo: float, hi: float) -> bool:
    """区間が 0.5 をまたがず、AUC が 0.5 から EFFECT 以上離れていれば「効く」（向きは問わない）。"""
    return (lo > 0.5 or hi < 0.5) and abs(a - 0.5) >= EFFECT


def report(df: pd.DataFrame) -> None:
    y = df["y"].to_numpy(float)
    log(f"評価した日 {len(df):,}・銘柄 {df['code'].nunique():,}・期間 {df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}・"
        f"+10% 到達 {y.mean():.1%}")
    for q in QUANTITIES:
        a, lo, hi = within_auc(df, q)
        log(f"{LABELS[q]}: 銘柄の中だけの AUC {a:.4f}（{LEVEL:.2%} 区間 {lo:.4f}〜{hi:.4f}）→ {'効く' if verdict(a, lo, hi) else '効かない'}・"
            f"全日まとめた AUC {auc(y, df[q].to_numpy()):.4f}")
        bins = pd.qcut(df[q].rank(method="first"), 5, labels=False)
        log(f"  5つの段（小 → 大）の実際の到達: " + "・".join(f"{df.loc[bins == b, 'y'].mean():.1%}" for b in range(5)))
        log(f"  順位相関: 全期間の到達率 {df[q].corr(df['base'], method='spearman'):+.3f}・"
            f"5番目に近い日までの距離 {df[q].corr(df['kth'], method='spearman'):+.3f}")
    for ref, lab in (("base", "全期間の到達率"), ("kth", "5番目に近い日までの距離")):
        a, lo, hi = within_auc(df, ref)
        log(f"参考・{lab}: 銘柄の中だけの AUC {a:.4f}（{lo:.4f}〜{hi:.4f}）・全日まとめた AUC {auc(y, df[ref].to_numpy()):.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_data"))
    ap.add_argument("--out", default=None, help="評価した日の行（parquet）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="試運転: 先頭の銘柄数（0 = 全部）")
    ap.add_argument("--rows", default=None, help="前に書いた行（--out）から表だけを出し直す（計算しない）")
    args = ap.parse_args(argv)
    if args.rows:
        report(pd.read_parquet(args.rows))
        return 0
    import pyarrow.parquet as pq
    codes = sorted({c for p in glob.glob(os.path.join(args.data_dir, "bars_*.parquet"))
                    for c in pq.read_table(p, columns=["Code"]).column("Code").to_pylist()})
    if args.limit:
        codes = codes[:args.limit]
    chunks = [(codes[i:i + CHUNK], args.data_dir) for i in range(0, len(codes), CHUNK)]
    log(f"銘柄 {len(codes):,}・チャンク {len(chunks)}・並列 {args.workers}")
    t0 = time.time()
    rows: List[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (r, _) in enumerate(ex.map(run_chunk, chunks), 1):
            rows.extend(r)
            if i % 5 == 0 or i == len(chunks):
                log(f"{i}/{len(chunks)} チャンク・{len(rows):,}日・{time.time() - t0:.0f}秒")
    df = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_parquet(args.out, index=False)
    report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
