#!/usr/bin/env python3
"""
列ごとの検定（research/exp/e20_annual_trajectory.py の screen）で、偶然 |z| を超える本数の見込み。

z は窓ごとの超過の平均 / 窓の SE なので、窓が n 本なら自由度 n−1 の t 分布。正規分布の 5% で数えると
窓が少ないとき過小になる（2026-10-10、EDINET の 523列・窓11本で「5% ≈ 26本」と出したが、t 分布では 38本）。
"""
import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import numpy as np  # noqa: E402

from e20_annual_trajectory import OUTCOME, TOP_PCT, chance_hits, screen  # noqa: E402


class TestChanceHits(unittest.TestCase):
    def test_eleven_windows_is_wider_than_normal(self):
        res = pd.DataFrame({"n_win": [11] * 523, "abs_z": [0.0] * 523})
        n, p, n_win = chance_hits(res)
        self.assertEqual(n_win, 11)
        self.assertAlmostEqual(p, 0.0734, places=3)        # 正規分布なら 0.0455
        self.assertAlmostEqual(n, 38.4, places=1)
        n3, p3, _ = chance_hits(res, 3.0)
        self.assertAlmostEqual(n3, 7.0, places=1)

    def test_many_windows_approaches_normal(self):
        res = pd.DataFrame({"n_win": [200] * 100, "abs_z": [0.0] * 100})
        _, p, _ = chance_hits(res)
        self.assertAlmostEqual(p, 0.0468, places=3)

    def test_unmeasured_rows_and_no_windows(self):
        res = pd.DataFrame({"n_win": [6, 6, None], "abs_z": [0.0, 0.0, None]})
        n, p, n_win = chance_hits(res)
        self.assertEqual(n_win, 6)
        self.assertAlmostEqual(p, 0.1019, places=3)
        self.assertEqual(chance_hits(pd.DataFrame({"abs_z": [0.0]}))[1], 0.05)   # 窓の列が無ければ従来の 5%


def screen_before_2026_10_10(frame, feats, windows):
    """window_stats を切り出す前の screen（2026-10-10 まで）。切り出しで値が変わらないことの比較用。"""
    from sklearn.metrics import roc_auc_score

    y = frame["label"].to_numpy(dtype=float)
    r = frame[OUTCOME].to_numpy(dtype=float)
    d = frame["Date"].to_numpy()
    rows = []
    for f in feats:
        x = frame[f].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
        cov = ok.mean()
        if ok.sum() < 500 or len(np.unique(y[ok])) < 2 or np.nanstd(x[ok]) == 0:
            rows.append({"feature": f, "coverage": cov, "n": int(ok.sum())})
            continue
        pooled = roc_auc_score(y[ok], x[ok])
        aucs, edges = [], []
        for (s, e) in windows:
            w = ok & (d >= s) & (d <= e)
            if w.sum() < 100 or len(np.unique(y[w])) < 2:
                continue
            xw, yw, rw = x[w], y[w], r[w]
            aucs.append(roc_auc_score(yw, xw))
            thr = np.nanpercentile(xw, TOP_PCT)
            top = xw >= thr
            if top.sum() >= 5:
                edges.append((rw[top].mean() - rw.mean()) * 100)
        aucs, edges = np.array(aucs), np.array(edges)
        se_e = edges.std(ddof=1) / np.sqrt(len(edges)) if len(edges) > 1 else np.nan
        rows.append({
            "feature": f, "coverage": cov, "n": int(ok.sum()),
            "auc_pooled": pooled,
            "auc_win": aucs.mean() if len(aucs) else np.nan,
            "auc_se": aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else np.nan,
            "auc_win_gt05": int((aucs > 0.5).sum()), "n_win": len(aucs),
            "edge_pt": edges.mean() if len(edges) else np.nan,
            "edge_se": se_e,
            "edge_z": edges.mean() / se_e if len(edges) > 1 and se_e > 0 else np.nan,
            "edge_win_pos": int((edges > 0).sum()),
        })
    out = pd.DataFrame(rows)
    out["abs_z"] = out["edge_z"].abs()
    return out.sort_values("abs_z", ascending=False).reset_index(drop=True)


class TestScreenUnchanged(unittest.TestCase):
    """2026-10-10 に窓ごとの計算（window_stats）を切り出した。実験22・66 の数字が変わらないこと。"""

    def test_same_numbers_as_before(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2021-01-04", "2023-12-29")
        n = len(dates) * 6
        df = pd.DataFrame({"Date": np.repeat(dates, 6)})
        r = rng.normal(0, 0.08, n)
        df[OUTCOME] = r
        df["label"] = (r > 0.05).astype(float)
        df["cont"] = rng.normal(0, 1, n) + 3 * r                       # 効く列
        df["noise"] = rng.normal(0, 1, n)
        df["ties"] = rng.integers(0, 3, n).astype(float)                # 同じ値が多い列
        df["sparse"] = np.where(rng.random(n) < 0.4, rng.normal(0, 1, n), np.nan)
        df["const"] = 1.0                                                # 測れない列
        df.loc[rng.random(n) < 0.05, OUTCOME] = np.nan
        windows = [(np.datetime64(s), np.datetime64(e)) for s, e in
                   [("2021-01-04", "2021-06-30"), ("2021-07-01", "2021-12-31"), ("2022-01-01", "2022-01-20"),
                    ("2022-01-21", "2022-12-31"), ("2023-01-01", "2023-12-29")]]
        feats = ["cont", "noise", "ties", "sparse", "const"]
        pd.testing.assert_frame_equal(screen(df, feats, windows), screen_before_2026_10_10(df, feats, windows))


if __name__ == "__main__":
    unittest.main()
