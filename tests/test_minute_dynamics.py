"""分足の価格帯別出来高の推移の特徴量（research/minute_dynamics.py。実験133）のテスト。

守りたいこと:
- 距離（フィッシャー・ラオ・ヘリンジャー・ワッサースタイン）とエントロピー・70%範囲が定義どおり
- 日ごとの分布は、分割・併合の係数で D の値の基準にそろえる（価格は掛け、株数は割る）
- 重心が上へ動けば mean_shift・poc_shift が正、日ごとの形が変わらなければ速さは 0
- 行の並び・件数を変えず、20営業日がそろわない行は欠測のまま。D より後の分足を消しても D の行は同じ
"""

from pathlib import Path
import json
import math
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "research", ROOT / "tests/fixtures"):
    sys.path.insert(0, str(extra))

import minute_dynamics as MD  # noqa: E402
import minute_features as MF  # noqa: E402
from volume_profile_inputs import write_inputs  # noqa: E402

NB = len(MD.FINE_EDGES) - 1


def at(rel, qty=1.0):
    """D の終値からの距離 rel の帯に qty を置いた分布。"""
    h = np.zeros(NB)
    h[int(np.digitize([rel], MD.FINE_EDGES)[0]) - 1] += qty
    return h


class DistanceTest(unittest.TestCase):
    def test_distances_and_entropy(self):
        p = at(-0.035) + at(-0.015)
        p = p / p.sum()
        self.assertAlmostEqual(MD.fisher_rao(p, p), 0.0, places=6)
        self.assertAlmostEqual(MD.hellinger(p, p), 0.0, places=6)
        q = at(0.05)
        self.assertAlmostEqual(MD.fisher_rao(at(-0.05), q), math.pi)          # 重ならない分布
        self.assertAlmostEqual(MD.hellinger(at(-0.05), q), 1.0)
        self.assertAlmostEqual(MD.wasserstein1(at(-0.015), at(-0.005)), 0.01)  # 1つ隣の帯へ動かす
        self.assertAlmostEqual(MD.wasserstein1(at(-0.055), at(-0.005)), 0.05)
        self.assertAlmostEqual(MD.entropy(p), math.log(2))
        self.assertAlmostEqual(MD.entropy(at(0.0)), 0.0)

    def test_value_area_and_coarse(self):
        p = np.zeros(NB)
        for rel, w in ((-0.045, .1), (-0.035, .2), (-0.025, .4), (-0.015, .2), (-0.005, .1)):
            p += at(rel, w)
        lo, hi = MD.value_area(p)
        self.assertAlmostEqual(lo, -0.04)                                       # −4%〜−1% で 80%
        self.assertAlmostEqual(hi, -0.01)
        c = MD.coarse(p)
        self.assertEqual(len(c), 8)
        self.assertAlmostEqual(c.sum(), 1.0)
        self.assertAlmostEqual(c[3], .7)                                         # −5〜−2%
        self.assertAlmostEqual(c[4], .3)                                         # −2〜0%

    def test_day_hist_uses_the_d_basis(self):
        pts = np.array([[200.0, 10.0, 2000.0]])                                 # 分割前の日: 200円・10株
        h = MD.day_hist(pts, 0.5, 100.0)                                        # 1:2 の分割 → 100円・20株
        self.assertAlmostEqual(h.sum(), 20.0)
        self.assertEqual(int(np.argmax(h)), int(np.digitize([0.0], MD.FINE_EDGES)[0]) - 1)


class FeaturesTest(unittest.TestCase):
    def series(self, prior_rel, recent_rel, jitter=0.0, seed=0):
        rng = np.random.default_rng(seed)
        hists, buckets = [], []
        for t in range(MD.WINDOW):
            rel = recent_rel if t >= MD.WINDOW - MD.RECENT else prior_rel
            h = at(rel, 100.0) + (at(rel + 0.01, jitter * rng.random()) if jitter else 0)
            hists.append(h)
            buckets.append(np.array([10.0, 40.0, 40.0, 10.0 if t < MD.WINDOW - MD.RECENT else 30.0]))
        return hists, buckets

    def test_shift_up_and_constant_shape(self):
        f = MD.features(*self.series(-0.055, -0.005))
        self.assertGreater(f["mean_shift"], 0.04)
        self.assertAlmostEqual(f["poc_shift"], 0.05, places=6)
        self.assertAlmostEqual(f["w1_recent_prior"], 0.05, places=6)
        self.assertAlmostEqual(f["fr_recent_prior"], math.pi)
        self.assertAlmostEqual(f["speed_prior"], 0.0)                            # その前の15日は同じ形
        self.assertGreater(f["close_share_change"], 0)                           # 引けの割合が増えた
        self.assertAlmostEqual(sum(f[f"delta_b{i}"] for i in range(8)), 0.0)
        self.assertAlmostEqual(sum(f[f"shape5_b{i}"] for i in range(8)), 1.0)
        self.assertEqual(set(f), {c.replace("vp_dyn_", "") for c in MD.COLUMNS})
        same = MD.features(*self.series(-0.025, -0.025))
        self.assertAlmostEqual(same["fr_recent_prior"], 0.0, places=6)
        self.assertAlmostEqual(same["speed_recent"], 0.0, places=6)
        self.assertAlmostEqual(same["fisher_info_recent"], 0.0, places=6)

    def test_needs_twenty_days_with_trades(self):
        hists, buckets = self.series(-0.02, -0.01)
        self.assertIsNone(MD.features(hists[1:], buckets[1:]))
        empty = [np.zeros(NB)] * MD.RECENT
        self.assertIsNone(MD.features(hists[:-MD.RECENT] + empty, buckets))   # 直近に約定が無い


def shaped_days(locs, scales, base=None, shares=None):
    """日ごとに、同じ形（base の点と株数）を位置 locs[t]・広がり scales[t] で動かした (価格, 株数)。"""
    base = np.array([-1.0, -0.5, 0.0, 0.5, 3.0]) if base is None else np.asarray(base, dtype=float)
    shares = np.array([10.0, 30.0, 40.0, 15.0, 5.0]) if shares is None else np.asarray(shares, dtype=float)
    base = base - (shares @ base) / shares.sum()                               # 株数で重みを付けた平均を 0 に
    return [(loc + sc * base, shares.copy()) for loc, sc in zip(locs, scales)]


class ShapeOnlyTest(unittest.TestCase):
    def test_same_shape_has_zero_shape_distance_whatever_the_location_and_scale(self):
        locs = [100.0] * 15 + [112.0] * 5                                     # 直近に 12% 上へ動いた
        scales = [1.0] * 15 + [2.0] * 5                                        # 広がりは2倍
        days = shaped_days(locs, scales)
        f = MD.shape_features(days, close=113.0)
        self.assertAlmostEqual(f["shape_fr_recent_prior"], 0.0, places=6)
        self.assertAlmostEqual(f["shape_speed_recent"], 0.0, places=6)
        self.assertAlmostEqual(f["shape_speed_prior"], 0.0, places=6)
        self.assertAlmostEqual(f["last_day_shape_fr"], 0.0, places=6)
        self.assertAlmostEqual(f["shape_info_recent"], 0.0, places=6)
        self.assertAlmostEqual(f["shape_entropy_change"], 0.0, places=6)
        self.assertAlmostEqual(f["skew_change"], 0.0, places=6)
        self.assertGreater(f["skew_recent"], 0.5)                               # 上側に裾が長い形
        w0, b0 = np.array([10, 30, 40, 15, 5.0]), np.array([-1, -.5, 0, .5, 3.0])
        sd_base = math.sqrt(w0 @ (b0 - (w0 @ b0) / w0.sum()) ** 2 / w0.sum())     # 株数で重みを付けた標準偏差
        self.assertAlmostEqual(f["sigma_recent"], 2.0 * sd_base / 112.0, places=6)
        self.assertAlmostEqual(f["sigma_log_ratio"], math.log((2.0 / 112.0) / (1.0 / 100.0)), places=6)
        self.assertAlmostEqual(f["vwap_drift_sigma"], 12.0 / sd_base, places=6)
        self.assertAlmostEqual(f["z_close_recent"], 1.0 / (2.0 * sd_base), places=6)
        self.assertAlmostEqual(f["vwap_up_share"], 1 / 19)                      # 上がったのは1日だけ
        # D の終値からの距離の帯で測る前回の距離は、値が動いただけで大きくなる
        hists = [MD.day_hist(np.column_stack([p, w, p * w]), 1.0, 113.0) for p, w in days]
        old = MD.features(hists, [np.array([1.0, 1, 1, 1])] * 20)
        self.assertGreater(old["fr_recent_prior"], 2.5)

    def test_shape_change_is_measured(self):
        days = shaped_days([100.0] * 20, [1.0] * 20)
        flip = shaped_days([100.0] * 5, [1.0] * 5, base=[-3.0, -0.5, 0.0, 0.5, 1.0], shares=[5.0, 15.0, 40.0, 30.0, 10.0])
        f = MD.shape_features(days[:15] + flip, close=100.0)                    # 直近は下側に裾が長い形
        self.assertGreater(f["shape_fr_recent_prior"], 0.5)
        self.assertLess(f["skew_change"], -1.0)
        self.assertGreater(f["last_day_shape_fr"], -1e-12)
        self.assertGreater(f["shape_speed_recent"], 0.0)                       # 形の変わり目が直近にある
        self.assertAlmostEqual(f["shape_speed_prior"], 0.0, places=6)

    def test_flat_days_and_missing_days(self):
        days = shaped_days([100.0] * 20, [1.0] * 20)
        days[3] = (np.array([100.0]), np.array([50.0]))                        # ストップ高の張り付き（1つの値だけ）
        days[7] = None                                                        # 売買の無い日
        f = MD.shape_features(days, close=100.0)
        self.assertTrue(all(np.isfinite(v) for v in f.values()))                 # 1つの値だけの日・売買の無い日があっても値が出る
        self.assertGreater(f["shape_speed_prior"], 0.0)                         # 張り付いた日の前後で形が変わる
        self.assertIsNone(MD.shape_features(days[:-5] + [None] * 5, close=100.0))   # 直近に約定が無い


class DynamicsOnStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        write_inputs(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, rows, minute_dir=None):
        keys = pd.DataFrame(rows, columns=["Code", "Date"])
        return MD.dynamics(keys, str(minute_dir or self.root / "minute"), str(self.root / "raw"))

    def test_rows_keep_order_and_missing_stays_missing(self):
        rows = [("123A0", "2025-11-18"), ("77770", "2025-11-18"), ("123A0", "2025-10-02"), ("99990", "2025-11-18")]
        out, report = self.build(rows)
        self.assertEqual(list(out["Code"]), [r[0] for r in rows])
        self.assertTrue(out.loc[[0, 3], MD.COLUMNS].notna().all().all())
        self.assertTrue(out.loc[1, MD.COLUMNS].isna().all())                   # 日足の無い銘柄
        self.assertTrue(out.loc[2, MD.COLUMNS].isna().all())                   # 20営業日がそろわない
        self.assertEqual(report["attached"], 2)
        self.assertEqual(report["status"]["ok"], 2)
        self.assertEqual(report["shape_attached"], 2)
        self.assertTrue(out.loc[[0, 3], MD.SHAPE_DYN_COLUMNS].notna().all().all())
        self.assertTrue(out.loc[[1, 2], MD.SHAPE_DYN_COLUMNS].isna().all().all())
        # 20日の形は minute_features の分布の形（帯で按分）に近い（こちらは価格ごとの株数をそのまま置く）
        shaped, _ = MF.build(pd.DataFrame(rows, columns=["Code", "Date"]), str(self.root / "minute"),
                             str(self.root / "raw"), with_shape=True)
        ours = out.loc[0, [f"vp_dyn_shape20_b{i}" for i in range(8)]].to_numpy(dtype=float)
        theirs = shaped.loc[0, MF.shape_columns(20)].to_numpy(dtype=float)
        self.assertLess(np.abs(ours - theirs).max(), 0.1)

    def test_later_days_do_not_change_earlier_rows(self):
        rows = [("123A0", "2025-11-14"), ("99990", "2025-11-14")]
        full, _ = self.build(rows)
        cut = Path(self.temp.name) / "cut"
        shutil.copytree(self.root / "minute", cut)
        manifest = json.loads((cut / "minute_manifest.json").read_text())
        manifest["days"] = {d: v for d, v in manifest["days"].items() if d <= "2025-11-14"}
        (cut / "minute_manifest.json").write_text(json.dumps(manifest))
        frame = pd.read_parquet(cut / "minute_202511.parquet")
        frame[pd.to_datetime(frame["Date"]) <= "2025-11-14"].to_parquet(cut / "minute_202511.parquet", index=False)
        early, _ = self.build(rows, cut)
        pd.testing.assert_frame_equal(full, early)


if __name__ == "__main__":
    unittest.main()
