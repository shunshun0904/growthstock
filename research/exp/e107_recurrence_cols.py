#!/usr/bin/env python3
"""
実験107: リカレンス・ネットワークの「今日に似た日」を、テクニカル指標111列すべてで選ぶと、今の9列より結果をよく当てるか。
検証だけ（画面・research/recurrence_net.py の FEATURES は変えない）。

運用者の依頼（2026-10-20）「全テクニカル系特徴量を使って再描画して欲しいです」→ 手元で111列版の網を描いて見比べた
（docs/MODEL_ADOPTION_RULES.md §70）。今の候補6件では、似た5日がほぼ入れ替わり（9列と同じ日は平均 0.5日）、似た日の
+10% 到達は 9列 0/30・111列 5/30 だったが、似た5日は同じ時期の続いた日が多く件数が少ないので判断できない →
運用者の選択（AskUserQuestion）「大きく検証する」。

やること（画面が候補の日に作るのと同じ手順を、過去の日に当てる）
  対象の日: 全銘柄（一般市場。build_panel と同じ）の78週高値の更新日（is_fresh_break。モデルの母集団。網の「今日」はいつも
           この日）のうち、20営業日後までの日足があり結果が分かる日。その日までのノード（列がそろう日）が9列・111列とも
           MIN_NODES（100）日以上ある日だけ（両方で同じ日を比べる）
  似た日: その日 t までのノードだけで列を変換・標準化し（recurrence_net.transform_states）、1/20σ 刻みに丸め、タイラー窓
          （20ノード）より離れたノードのうち距離の近い k 日（同じ距離なら古い日が先）。似た日はいつも t より21ノード以上前
          なので、結果は t の時点で分かっている（先を見ない）
  当てる量 y: t の +10% 到達（翌営業日の始値で買い、t+1〜t+20 の高値が 1.10 倍に届いたか。recurrence_net.forward_moves の
             rise ≥ 10。画面の「結果」と同じ判定）
  使う量: 似た5日の +10% 到達率（画面のタイル「似た日の +10% 到達」）。参考に k=10、その銘柄の t の21ノード以上前までの
         到達率（画面の「全期間」にあたる）、似た日の到達率 − 全期間（銘柄の癖を引いたもの）

判定（回す前に決めた。2026-10-20。運用者への報告にも書いた）
  主な物差しは、似た5日の到達率で y を当てる AUC（全日をまとめて。同じ値は 0.5 で数える）。111列 − 9列 の AUC の差に、
  銘柄を選び直すブートストラップ（1,000回・種 0）で 95% 区間を付け、区間の下が 0 より上なら「111列の方が当てる」→ 本番に
  入れるかは運用者が決める。区間が 0 をまたぐ・9列が上なら 9列のまま。

確かめ（回す前）: この台本の近傍の選び方が、本番のファイル（9列版・111列版。2026-10-13〜19 の候補6件）の今日の似た日と
12/12 で一致し、+10% 到達の判定も 60/60 で一致した。

結果（2026-10-20。手元で回した。日足は Release data-raw の 2026-10-19 時点。docs/MODEL_ADOPTION_RULES.md §70）: 評価した日
28,222（4,428銘柄・2018-04〜2026-09）、+10% 到達 29.8%。AUC（全日まとめて）似た5日 9列 0.5955・111列 0.6173、差 +0.0218（95% 区間
+0.0147〜+0.0291）→ 判定は「111列の方が当てる」。ただし全期間の到達率だけで 0.6768 とどちらより高く、参考の銘柄の中だけの AUC は
9列 0.5022・111列 0.4897（差 −0.0125、区間 −0.0233〜−0.0020）で、どちらもその銘柄の中でどの日が届くかは当てない。
運用者の決定（2026-10-20）「9列のまま」→ 画面・recurrence_net.py の FEATURES は変えない。

出すもの（公開ログには件数・割合・AUC だけ。銘柄・値は出さない）
  評価した日の数・銘柄の数・y の割合、AUC（9列・111列。k=5・10、全期間、似た日 − 全期間）、差と 95% 区間、似た日の到達率の
  段（0%・20%・40%以上）ごとの実際の到達率、似た5日がいくつの時期に分かれるか、前半（〜2021）と後半（2022〜）、参考に同じ銘柄の
  日どうしだけの AUC（試運転で全期間の到達率だけの AUC が高かったので足した。判定には使わない）

    python3 research/exp/e107_recurrence_cols.py --data-dir research/_data --out research/_data/oof/e107_rows.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dataset as B  # noqa: E402
import recurrence_net as RN  # noqa: E402
import technicals as TX  # noqa: E402

NINE: Tuple[str, ...] = tuple(RN.FEATURES)
ALL: Tuple[str, ...] = tuple(TX.COLUMNS)
assert set(NINE) <= set(ALL) and len(ALL) == 111
RN.FEATURES = ALL                     # build_panel が111列すべてを付けるように（9列はその一部。この台本の中だけ）
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)    # build_panel が列を1本ずつ足す警告（結果は同じ）
TH, MIN_NODES = RN.THEILER, RN.MIN_NODES
KS = (5, 10)
ARMS = (("9", NINE), ("111", ALL))
CHUNK = 100
SEED = 0
N_BOOT = 1000


def log(msg: str) -> None:
    print(f"[e107] {msg}", flush=True)


def neighbors(X: np.ndarray, ok: np.ndarray, t: int, cols: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    パネルの行 t を今日として、t までのノード（ok の行）で画面と同じ手順で近い日を選ぶ。
    戻り値: (ノードの行番号, 近い順の行番号を max(KS) 日まで)。ノードが MIN_NODES 未満・t がノードでなければ (None, None)。
    """
    idx = np.flatnonzero(ok[:t + 1])
    if len(idx) < MIN_NODES or idx[-1] != t:
        return None, None
    Z, _ = RN.transform_states(X[idx], cols)
    zq = RN.quantize(Z).astype(np.int64)
    m = len(idx)
    s2 = ((zq - zq[-1]) ** 2).sum(1)
    pos = np.arange(m)
    c = pos[pos < m - 1 - TH]                         # タイラー窓: 20 ノードより離れたノードだけ
    o = c[np.lexsort((c, s2[c]))][:max(KS)]          # 距離の2乗の小さい順、同じなら古い日が先（knn_order と同じ）
    return idx, idx[o]


def run_chunk(args) -> Tuple[List[dict], float]:
    """1チャンクの銘柄について、更新日ごとの行（y と9列・111列の量）を返す。"""
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
        mats = {name: g[list(cols)].to_numpy(float) for name, cols in ARMS}
        oks = {name: np.isfinite(X).all(1) for name, X in mats.items()}
        fresh = g["is_fresh_break"].fillna(False).to_numpy(bool)
        for t in np.flatnonzero(fresh & oks["9"] & oks["111"] & np.isfinite(hit)):
            rec = {"code": code, "date": g["Date"].iloc[t], "y": hit[t]}
            for name, cols in ARMS:
                idx, nb = neighbors(mats[name], oks[name], t, cols)
                if idx is None:
                    break
                past = idx[: max(0, len(idx) - 1 - TH)]          # 21ノード以上前（結果が t までに分かる）
                hp = hit[past]
                rec[f"base_{name}"] = float(np.nanmean(hp)) if np.isfinite(hp).any() else np.nan
                for k in KS:
                    h = hit[nb[:k]]
                    rec[f"p{k}_{name}"] = float(np.nanmean(h)) if np.isfinite(h).any() else np.nan
                rec[f"per_{name}"] = RN.periods_of(np.searchsorted(idx, nb[:5]))
            else:
                rows.append(rec)
    return rows, time.time() - t0


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC（Mann–Whitney。同じ値は 0.5）。"""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def boot_diff(df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    """AUC(b) − AUC(a) と、銘柄を選び直すブートストラップの 95% 区間。"""
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
    銘柄ごとの「銘柄の中だけの AUC」の分子（列ごと）と分母（到達した日 × 届かなかった日 の組の数）。同じ銘柄の日どうしの組だけを
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
    return np.asarray(nums, dtype=float), np.asarray(dens, dtype=float)


def boot_within(df: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float, float, float]:
    """銘柄の中だけの AUC（a・b）と、差 b − a・銘柄を選び直すブートストラップの 95% 区間。"""
    num, den = within_parts(df, (a, b))
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(den), np.full(len(den), 1 / len(den)), size=n_boot).astype(float)   # 銘柄ごとの選ばれた回数
    d = (w @ num[:, 1] - w @ num[:, 0]) / (w @ den)
    lo, hi = np.percentile(d, [2.5, 97.5])
    au = num.sum(0) / den.sum()
    return float(au[0]), float(au[1]), float(au[1] - au[0]), float(lo), float(hi)


def report(df: pd.DataFrame) -> None:
    y = df["y"].to_numpy(float)
    log(f"評価した日 {len(df):,}・銘柄 {df['code'].nunique():,}・期間 {df['date'].min():%Y-%m-%d}〜{df['date'].max():%Y-%m-%d}・"
        f"+10% 到達 {y.mean():.1%}")
    for name, _ in ARMS:
        df[f"rel_{name}"] = df[f"p5_{name}"] - df[f"base_{name}"]
        log(f"{name}列: AUC 似た5日 {auc(y, df[f'p5_{name}'].to_numpy()):.4f}・似た10日 {auc(y, df[f'p10_{name}'].to_numpy()):.4f}・"
            f"全期間 {auc(y, df[f'base_{name}'].to_numpy()):.4f}・似た5日−全期間 {auc(y, df[f'rel_{name}'].to_numpy()):.4f}・"
            f"似た5日の時期の数 平均 {df[f'per_{name}'].mean():.2f}")
    for a, b, what in (("p5_9", "p5_111", "似た5日"), ("p10_9", "p10_111", "似た10日"), ("rel_9", "rel_111", "似た5日−全期間")):
        d, lo, hi = boot_diff(df, a, b)
        log(f"差 111列−9列（{what}）: {d:+.4f}（95% 区間 {lo:+.4f}〜{hi:+.4f}）")
    for name, _ in ARMS:
        p = df[f"p5_{name}"]
        parts = []
        for lab, m in (("0%", p < 0.1), ("20%", (p >= 0.1) & (p < 0.3)), ("40%以上", p >= 0.3)):
            parts.append(f"{lab} {m.sum():,}日 → 到達 {df.loc[m, 'y'].mean():.1%}")
        log(f"{name}列: 似た5日の到達率の段ごとの実際の到達: " + "・".join(parts))
    for lab, m in (("〜2021", df["date"] < "2022-01-01"), ("2022〜", df["date"] >= "2022-01-01")):
        sub = df[m]
        ys = sub["y"].to_numpy(float)
        log(f"{lab}（{len(sub):,}日）: AUC 似た5日 9列 {auc(ys, sub['p5_9'].to_numpy()):.4f}・111列 {auc(ys, sub['p5_111'].to_numpy()):.4f}")
    # 参考（試運転を見てから足した。判定には使わない）: 全期間の到達率（どの銘柄か）だけで AUC が高かったので、同じ銘柄の日どうし
    # だけで比べる（その銘柄の中で、似た日の到達率が高い日ほど到達しやすいか）
    a9, a111, d, lo, hi = boot_within(df, "p5_9", "p5_111")
    ab, _, _, _, _ = boot_within(df, "base_9", "base_111", n_boot=10)
    log(f"参考・銘柄の中だけの AUC: 9列 似た5日 {a9:.4f}・111列 似た5日 {a111:.4f}・全期間 {ab:.4f}")
    log(f"参考・銘柄の中だけの AUC の差 111列−9列（似た5日）: {d:+.4f}（95% 区間 {lo:+.4f}〜{hi:+.4f}）")


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
    import glob
    codes = sorted({c for p in glob.glob(os.path.join(args.data_dir, "bars_*.parquet"))
                    for c in pq.read_table(p, columns=["Code"]).column("Code").to_pylist()})
    if args.limit:
        codes = codes[:args.limit]
    chunks = [(codes[i:i + CHUNK], args.data_dir) for i in range(0, len(codes), CHUNK)]
    log(f"銘柄 {len(codes):,}・チャンク {len(chunks)}・並列 {args.workers}")
    t0 = time.time()
    rows: List[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (r, sec) in enumerate(ex.map(run_chunk, chunks), 1):
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
