#!/usr/bin/env python3
"""
実験111: リカレンス・ネットワークの「今日に似た日」を、今のユークリッド距離ではなくマハラノビス距離（縮小推定つき）で選ぶと、
同じ銘柄の中でどの更新日が +10% に届くかをよく当てるか。検証だけ（画面・research/recurrence_net.py は変えない）。

運用者の依頼（2026-10-23）「リカレンスネットワークにおける、近傍点の距離の算出方法を見直してほしいです。今はユークリッド距離に
なってませんか？」。今の距離は、9列を列の型で変換し銘柄ごとに平均0・標準偏差1にそろえた値（1/20σ 刻み）のユークリッド距離
（標準化ユークリッド距離。recurrence_net.sq_dist・画面の src/lib/recurrence.js の sqDist）。手元の確かめ（無作為の254銘柄。
docs/MODEL_ADOPTION_RULES.md §74）では、トレンド系の5列（−DI・VI+・基準線乖離・TSI・ストキャス%K）が銘柄の中で互いに相関
0.61〜0.80 で、ユークリッド距離ではこの向きが約5倍の重みになる（主成分の1つ目が散らばりの 45%・実質の次元 3.6）。
運用者の選択（AskUserQuestion）: 比べる距離「マハラノビス（縮小推定つき）」・判定の物差し「同じ銘柄の中の AUC」。

やること（実験107 と同じ手順を、距離だけ替えて過去の日に当てる）
  対象の日: 全銘柄（一般市場。build_panel と同じ）の78週高値の更新日（is_fresh_break）のうち、20営業日後までの日足があり結果が
           分かる日で、その日までのノード（9列がそろう日）が MIN_NODES（100）日以上ある日。実験107 は111列もそろう日に絞ったが、
           ここは9列だけなので日が少し多い
  状態: t までのノードだけで9列を変換・標準化する（recurrence_net.transform_states。今と同じ）
  E（今）: 1/20σ 刻みに丸めた状態の距離の2乗で、タイラー窓（20ノード）より離れたノードを近い順に。同じ距離なら古い日が先
           （recurrence_net.knn_order と同じ）
  M（マハラノビス）: 同じ状態 Z（t までのノード）の共分散を Ledoit–Wolf の縮小推定（sklearn.covariance.ledoit_wolf）で求め、
           その逆平方根で白色化した座標 Z Σ̂^(−1/2)（|値| は Z_CAP で切る）を同じ 1/20 刻みに丸め、E と同じ手順で近い日を選ぶ。
           白色化した座標のユークリッド距離は、縮小した共分散でのマハラノビス距離と同じ（画面に入れるなら、ファイルの状態を
           この座標に替えるだけで画面の式は今のまま）。縮小推定にするのは、散らばりの小さい向き（9列の重なりの残り。散らばりの
           1〜2%）の雑音を、ほかの向きと同じ重みまで大きくしないため
  当てる量 y: t の +10% 到達（翌営業日の始値で買い、t+1〜t+20 の高値が 1.10 倍に届いたか。実験107 と同じ）
  使う量: 似た5日の +10% 到達率（画面のタイル）。参考に似た10日と、全期間（t の21ノード以上前までの到達率）

判定（回す前に決めた。2026-10-23）
  主な物差しは同じ銘柄の中だけの AUC（同じ銘柄の「届いた日 × 届かなかった日」の組だけを数える。同じ値は 0.5。実験107 の参考の
  物差しと同じ）で、似た5日の到達率の M − E。銘柄を選び直すブートストラップ（1,000回・種 0）の 95% 区間の下が 0 より上なら
  「マハラノビスの方が当てる」→ 画面に入れるかは運用者が決める。区間が 0 をまたぐ・E が上ならユークリッドのまま。
  参考（判定には使わない）: 全日まとめた AUC（実験107 の主な物差し。どの銘柄かでほぼ決まる）、似た10日、似た5日の E と M の
  重なり、似た5日がいくつの時期に分かれるか、縮小の強さ、前半（〜2021）・後半（2022〜）

確かめ（回す前）: E の似た日の到達率が、実験107 の9列の行（2026-10-19 時点の日足）と同じ日で一致するか（--check-e107）

出すもの（公開ログには件数・割合・AUC だけ。銘柄・値は出さない）

    python3 research/exp/e111_recurrence_metric.py --data-dir research/_data --out research/_data/oof/e111_rows.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import recurrence_net as RN  # noqa: E402

COLS: Tuple[str, ...] = tuple(RN.FEATURES)       # 9列（§70-5 で9列のまま）
assert len(COLS) == 9, COLS
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)    # build_panel が列を1本ずつ足す警告（結果は同じ）
TH, MIN_NODES = RN.THEILER, RN.MIN_NODES
KS = (5, 10)
ARMS = ("E", "M")
CHUNK = 100
SEED = 0
N_BOOT = 1000


def log(msg: str) -> None:
    print(f"[e111] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 距離（E: 今のユークリッド / M: 縮小推定の共分散で白色化した座標のユークリッド = マハラノビス）
# --------------------------------------------------------------------------- #

def whitener(Z: np.ndarray) -> Tuple[np.ndarray, float]:
    """Z（ノード × 9）の共分散を Ledoit–Wolf で縮小し、その逆平方根（対称）と縮小の強さ（0〜1）を返す。"""
    S, shrink = ledoit_wolf(np.asarray(Z, dtype=float))
    vals, vecs = np.linalg.eigh(S)
    W = (vecs * (1.0 / np.sqrt(np.maximum(vals, 1e-12)))) @ vecs.T
    return W, float(shrink)


def arm_states(Z: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
    """腕ごとの丸めた座標（整数。1/Q 刻み）。E は今の状態そのもの、M は白色化した座標（|値| ≤ Z_CAP）。"""
    W, shrink = whitener(Z)
    return {"E": RN.quantize(Z).astype(np.int64),
            "M": RN.quantize(np.clip(Z @ W, -RN.Z_CAP, RN.Z_CAP)).astype(np.int64)}, shrink


def nearest(zq: np.ndarray, kmax: int = max(KS), theiler: int = TH) -> np.ndarray:
    """最後のノード（今日）の近い順の近傍（ノードの位置）。タイラー窓より離れたノードだけ、同じ距離なら古い日が先（knn_order と同じ）。"""
    m = len(zq)
    s2 = ((zq - zq[-1]) ** 2).sum(1)
    pos = np.arange(m)
    c = pos[pos < m - 1 - theiler]
    return c[np.lexsort((c, s2[c]))][:kmax]


def neighbor_sets(X: np.ndarray, ok: np.ndarray, t: int) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], float]]:
    """
    パネルの行 t を今日として、t までのノード（ok の行）で腕ごとに近い日を選ぶ。
    戻り値: (ノードの行番号, {腕: 近い順の行番号を max(KS) 日まで}, 縮小の強さ)。ノードが MIN_NODES 未満・t がノードでなければ None。
    """
    idx = np.flatnonzero(ok[:t + 1])
    if len(idx) < MIN_NODES or idx[-1] != t:
        return None
    Z, _ = RN.transform_states(X[idx], COLS)
    zs, shrink = arm_states(Z)
    return idx, {a: idx[nearest(zs[a])] for a in ARMS}, shrink


def run_chunk(args) -> Tuple[List[dict], float]:
    """1チャンクの銘柄について、更新日ごとの行（y と腕ごとの量）を返す。"""
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
            got = neighbor_sets(X, ok, t)
            if got is None:
                continue
            idx, nb, shrink = got
            past = idx[: max(0, len(idx) - 1 - TH)]              # 21ノード以上前（結果が t までに分かる）
            hp = hit[past]
            rec = {"code": code, "date": g["Date"].iloc[t], "y": hit[t], "shrink": shrink, "n_nodes": len(idx),
                   "base": float(np.nanmean(hp)) if np.isfinite(hp).any() else np.nan,
                   "ov5": len(set(nb["E"][:5]) & set(nb["M"][:5])) / 5}
            for a in ARMS:
                for k in KS:
                    h = hit[nb[a][:k]]
                    rec[f"p{k}_{a}"] = float(np.nanmean(h)) if np.isfinite(h).any() else np.nan
                rec[f"per_{a}"] = RN.periods_of(np.searchsorted(idx, nb[a][:5]))
            rows.append(rec)
    return rows, time.time() - t0


# --------------------------------------------------------------------------- #
# AUC（実験107 と同じ式。実験107 の台本は読み込むと RN.FEATURES を111列に替えるので、ここに置く）
# --------------------------------------------------------------------------- #

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC（Mann–Whitney。同じ値は 0.5）。"""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def boot_diff(df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    """全日まとめた AUC(b) − AUC(a) と、銘柄を選び直すブートストラップの 95% 区間。"""
    y = df["y"].to_numpy(float)
    sa, sb = df[a].to_numpy(float), df[b].to_numpy(float)
    groups = [np.asarray(v) for v in df.groupby("code").indices.values()]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(auc(y[take], sb[take]) - auc(y[take], sa[take]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return auc(y, sb) - auc(y, sa), float(lo), float(hi)


def within_parts(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    銘柄ごとの「銘柄の中だけの AUC」の分子（列ごと）と分母（届いた日 × 届かなかった日 の組の数）。同じ銘柄の日どうしの組だけを
    数え、同じ値は 0.5。どの列にも値のある日だけを使う（列どうしで同じ組を比べる）。
    """
    nums, dens = [], []
    for _, g in df.groupby("code", sort=False):
        s = g[list(cols)].to_numpy(float)
        ok = np.isfinite(s).all(1)
        y, s = g["y"].to_numpy(float)[ok], s[ok]
        n1 = y.sum()
        n0 = len(y) - n1
        if not n1 or not n0:
            continue
        r = pd.DataFrame(s).rank(method="average").to_numpy()
        nums.append(r[y == 1].sum(0) - n1 * (n1 + 1) / 2)
        dens.append(n1 * n0)
    return np.asarray(nums, dtype=float).reshape(-1, len(cols)), np.asarray(dens, dtype=float)


def boot_within(df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float, float, float]:
    """銘柄の中だけの AUC（a・b）と、差 b − a・銘柄を選び直すブートストラップの 95% 区間。数えられる組が無ければ NaN。"""
    num, den = within_parts(df, (a, b))
    if not len(den):
        return (float("nan"),) * 5
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(den), np.full(len(den), 1 / len(den)), size=n_boot).astype(float)   # 銘柄ごとの選ばれた回数
    d = (w @ num[:, 1] - w @ num[:, 0]) / (w @ den)
    lo, hi = np.percentile(d, [2.5, 97.5])
    au = num.sum(0) / den.sum()
    return float(au[0]), float(au[1]), float(au[1] - au[0]), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 報告
# --------------------------------------------------------------------------- #

def report(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    y = df["y"].to_numpy(float)
    log(f"評価した日 {len(df):,}・銘柄 {df['code'].nunique():,}・期間 {df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}・"
        f"+10% 到達 {y.mean():.1%}")
    out: dict = {"days": int(len(df)), "codes": int(df["code"].nunique()), "y": float(y.mean())}
    # 主な判定: 同じ銘柄の中だけの AUC（似た5日）の M − E
    aE, aM, d, lo, hi = boot_within(df, "p5_E", "p5_M", n_boot=n_boot)
    better = bool(lo > 0)
    log(f"■ 判定（同じ銘柄の中の AUC・似た5日）: ユークリッド {aE:.4f}・マハラノビス {aM:.4f}・差 {d:+.4f}"
        f"（95% 区間 {lo:+.4f}〜{hi:+.4f}）→ {'マハラノビスの方が当てる' if better else 'ユークリッドのまま'}")
    out["main"] = {"E": aE, "M": aM, "diff": d, "lo": lo, "hi": hi, "better": better}
    # 参考
    e10, m10, d10, lo10, hi10 = boot_within(df, "p10_E", "p10_M", n_boot=n_boot)
    base_w, _, _, _, _ = boot_within(df, "base", "p5_E", n_boot=10)
    log(f"参考・同じ銘柄の中の AUC: 似た10日 E {e10:.4f}・M {m10:.4f}（差 {d10:+.4f}、{lo10:+.4f}〜{hi10:+.4f}）・全期間 {base_w:.4f}")
    out["within10"] = {"E": e10, "M": m10, "diff": d10, "lo": lo10, "hi": hi10}
    for a in ARMS:
        log(f"参考・全日まとめた AUC（{a}）: 似た5日 {auc(y, df[f'p5_{a}'].to_numpy()):.4f}・似た10日 "
            f"{auc(y, df[f'p10_{a}'].to_numpy()):.4f}・似た5日の時期の数 平均 {df[f'per_{a}'].mean():.2f}")
    log(f"参考・全日まとめた AUC（全期間）: {auc(y, df['base'].to_numpy()):.4f}")
    dd, dlo, dhi = boot_diff(df, "p5_E", "p5_M", n_boot=n_boot)
    log(f"参考・全日まとめた AUC の差 M − E（似た5日）: {dd:+.4f}（95% 区間 {dlo:+.4f}〜{dhi:+.4f}）")
    out["pooled"] = {"diff": dd, "lo": dlo, "hi": dhi,
                     **{f"p5_{a}": auc(y, df[f"p5_{a}"].to_numpy()) for a in ARMS}, "base": auc(y, df["base"].to_numpy())}
    log(f"参考・似た5日の E と M の重なり 平均 {df['ov5'].mean():.2f}（5日とも同じ {(df['ov5'] == 1).mean():.1%}・"
        f"1日も同じでない {(df['ov5'] == 0).mean():.1%}）・縮小の強さ 中央 {df['shrink'].median():.3f}"
        f"（{df['shrink'].quantile(0.1):.3f}〜{df['shrink'].quantile(0.9):.3f}）")
    out["overlap5"] = float(df["ov5"].mean())
    out["shrink_median"] = float(df["shrink"].median())
    for a in ARMS:
        p = df[f"p5_{a}"]
        parts = []
        for lab, m in (("0%", p < 0.1), ("20%", (p >= 0.1) & (p < 0.3)), ("40%以上", p >= 0.3)):
            parts.append(f"{lab} {m.sum():,}日 → 到達 {df.loc[m, 'y'].mean():.1%}")
        log(f"参考・{a}: 似た5日の到達率の段ごとの実際の到達: " + "・".join(parts))
    for lab, m in (("〜2021", df["date"] < "2022-01-01"), ("2022〜", df["date"] >= "2022-01-01")):
        sub = df[m]
        e_, m_, d_, lo_, hi_ = boot_within(sub, "p5_E", "p5_M", n_boot=200)
        log(f"参考・{lab}（{len(sub):,}日）: 同じ銘柄の中の AUC 似た5日 E {e_:.4f}・M {m_:.4f}（差 {d_:+.4f}、"
            f"{lo_:+.4f}〜{hi_:+.4f}・200回）")
    return out


def check_e107(df: pd.DataFrame, path: str) -> dict:
    """E の量が、実験107 の9列の行（同じ銘柄・同じ日）と一致するか。"""
    old = pd.read_parquet(path)
    m = df.merge(old[["code", "date", "y", "base_9", "p5_9", "p10_9"]], on=["code", "date"], how="inner")
    res = {"common": int(len(m)), "e107_rows": int(len(old)), "e111_rows": int(len(df))}
    for a, b in (("y", "y_y"), ("base", "base_9"), ("p5_E", "p5_9"), ("p10_E", "p10_9")):
        x = m[a if a != "y" else "y_x"].to_numpy(float)
        z = m[b].to_numpy(float)
        same = np.isclose(x, z, rtol=0, atol=1e-12) | (np.isnan(x) & np.isnan(z))
        res[a] = float(same.mean()) if len(m) else float("nan")
    log(f"確かめ・実験107 の行と同じ日 {res['common']:,}（実験107 {res['e107_rows']:,}・今回 {res['e111_rows']:,}）: 一致 "
        f"y {res['y']:.2%}・全期間 {res['base']:.2%}・似た5日 {res['p5_E']:.2%}・似た10日 {res['p10_E']:.2%}")
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_data"))
    ap.add_argument("--out", default=None, help="評価した日の行（parquet）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="試運転: 先頭の銘柄数（0 = 全部）")
    ap.add_argument("--rows", default=None, help="前に書いた行（--out）から表だけを出し直す（計算しない）")
    ap.add_argument("--check-e107", default=None, help="実験107 の行（e107_rows.parquet）。E の量が同じ日で一致するか")
    args = ap.parse_args(argv)
    if args.rows:
        df = pd.read_parquet(args.rows)
    else:
        import glob
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
            for i, (r, _sec) in enumerate(ex.map(run_chunk, chunks), 1):
                rows.extend(r)
                if i % 5 == 0 or i == len(chunks):
                    log(f"{i}/{len(chunks)} チャンク・{len(rows):,}日・{time.time() - t0:.0f}秒")
        df = pd.DataFrame(rows)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            df.to_parquet(args.out, index=False)
    if args.check_e107:
        check_e107(df, args.check_e107)
    report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
