#!/usr/bin/env python3
"""
実験106（research/exp/e106_augment_runs.py）の部品。データには触らない（小さな作りものの表で確かめる）。

- 表: 更新日すべての表の新規の行がいまの学習データと行・列・ラベルですべて一致すれば通り、1か所でもずれれば止まる。
  base_length での判定と本番の印の食い違いも止まる。足す行には局面が付き、局面の新規の行が学習データに無い行を数える
- 重み: 局面ごとに合計1（窓の学習に入った行の中で）。D は G と同じ件数・同じ重みの合計で、足す行はいまの行の複製。
  P の入れ替えは日付ごとの正例の数を変えず、1行だけの日はそのまま、種が同じなら同じ
- out-of-fold: A は E41.oof_folds（本番と同じ作り）と同じスコア。G・P・D は A と違い、同じ引数なら同じ。評価の行はどの腕も同じ
- 報告の一通り（作りものの out-of-fold で最後まで通る）

  python3 tests/test_e106_augment_runs.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import build_dataset as B  # noqa: E402
import e106_augment_runs as E  # noqa: E402
import e41_stop_loss as E41  # noqa: E402
import features as F  # noqa: E402
import probe_augment_rows as PA  # noqa: E402


# --------------------------------------------------------------------------- #
# 表（assemble・compare_fresh）
# --------------------------------------------------------------------------- #
TCOLS = ["f0", "f1", "f2", "base_length"]


def synth_tables(seed=0, drop_fresh=0):
    """
    更新日すべての表 al（新規 = base_length が21以上か欠測）・控えた更新日 updates・いまの学習データ frame。
    drop_fresh > 0 なら、新規の行をその数だけ al と frame の両方から落とす（流動性などで新規の日だけ落ちた形。updates には残す）。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=260)
    rows = []
    for c in ("10010", "10020", "10030", "10040"):
        last = None
        for p in sorted(rng.choice(260, 45, replace=False)):
            rows.append({"Code": c, "Date": days[p], "pos": int(p), "base_length": float(p - last) if last is not None else np.nan})
            last = p
    al = pd.DataFrame(rows)
    al["fresh"] = PA.fresh_by_base_length(al["base_length"])
    for j in range(3):
        al[f"f{j}"] = rng.normal(size=len(al))
    al.loc[al.index[::7], "f2"] = np.nan                                     # 欠測どうしは同じとみなす
    al["vol_20d"] = rng.uniform(1.0, 4.0, len(al))
    al["future_rise"] = rng.uniform(0.0, 30.0, len(al))
    al["end_level"] = rng.uniform(0.0, 25.0, len(al))
    al["uptrend_end"] = (rng.random(len(al)) < 0.7).astype(float)
    al["label"] = 0.0
    al["label"] = B.rise_label_for_k(al, B.VOL_NORM_K)
    al["Date"] = al["Date"].astype("datetime64[ns]")
    updates = al[["Code", "Date", "pos", "fresh"]].copy()
    if drop_fresh:
        gone = al.index[al["fresh"].to_numpy()][1:1 + drop_fresh]
        al = al.drop(index=gone)
    frame = al[al["fresh"]].drop(columns=["fresh", "pos"]).reset_index(drop=True)
    frame["ret_o1_20"] = rng.normal(1.0, 8.0, len(frame))
    frame["ret_o1_40"] = rng.normal(1.0, 10.0, len(frame))
    return frame, al.drop(columns=["fresh", "pos"]).reset_index(drop=True), updates


class TestTables(unittest.TestCase):
    def test_assemble(self):
        frame, al, updates = synth_tables()
        with redirect_stdout(io.StringIO()):
            fr, ad, checks = E.assemble(frame, al, updates, TCOLS)
        fresh = PA.fresh_by_base_length(al["base_length"])
        self.assertEqual(len(fr), int(fresh.sum()))
        self.assertEqual(len(ad), int((~fresh).sum()))                        # 足す行は新規ではない行すべて
        self.assertGreater(len(ad), len(fr))
        self.assertTrue((ad["base_length"] <= PA.COOLDOWN).all())
        self.assertEqual(checks["fresh"]["bad_cols"], {})
        self.assertEqual((checks["flag"]["diff"], checks["flag"]["no_update"]), (0, 0))
        self.assertEqual(checks["added"]["orphan"], 0)
        # ラベルは 1.5σ（rise_label_for_k）
        y15 = B.rise_label_for_k(al, E.K)
        self.assertTrue(np.array_equal(fr["label"].to_numpy(), y15[fresh].to_numpy()))
        # 局面は assign_clusters と同じで、同じ局面の行は同じ銘柄
        cl = PA.assign_clusters(updates)
        m = ad.merge(cl[["Code", "Date", "cluster"]], on=["Code", "Date"])
        self.assertTrue((m["run"] == m["cluster"]).all())
        self.assertTrue((ad.groupby("run")["Code"].nunique() == 1).all())
        self.assertEqual(checks["added"]["runs"], int(ad["run"].nunique()))

    def test_orphan_runs(self):
        frame, al, updates = synth_tables(drop_fresh=2)
        with redirect_stdout(io.StringIO()):
            _, ad, checks = E.assemble(frame, al, updates, TCOLS)
        self.assertGreater(checks["added"]["orphan"], 0)                      # 新規の日だけ落ちた局面の足す行
        self.assertEqual(checks["added"]["used"], len(ad))

    def test_stops_when_features_differ(self):
        frame, al, updates = synth_tables()
        bad = frame.copy()
        bad.loc[3, "f1"] = bad.loc[3, "f1"] + 1e-3
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                E.assemble(bad, al, updates, TCOLS)
            with self.assertRaises(SystemExit):                               # いまの学習データにだけある行
                E.assemble(pd.concat([frame, frame.tail(1).assign(Date=pd.Timestamp("2030-01-01"))], ignore_index=True),
                           al, updates, TCOLS)

    def test_stops_when_flag_differs(self):
        frame, al, updates = synth_tables()
        u = updates.copy()
        i = u.index[~u["fresh"].to_numpy()][0]
        u.loc[i, "fresh"] = True                                              # 本番の印だけ変える
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                E.assemble(frame, al, u, TCOLS)

    def test_compare_fresh(self):
        frame, al, _ = synth_tables()
        frame["y15"] = B.rise_label_for_k(frame, E.K)
        fresh = frame.copy()
        c = E.compare_fresh(frame, fresh, TCOLS)
        self.assertEqual((c["only_ds"], c["only_fresh"], c["dup"], c["bad_cols"]), (0, 0, 0, {}))
        self.assertEqual(c["matched"], len(frame))
        f2 = fresh.copy()
        f2.loc[0, "f0"] = np.nan
        f2.loc[[1, 2], "label"] = 1.0 - f2.loc[[1, 2], "label"]
        c = E.compare_fresh(frame, f2.iloc[1:], TCOLS)
        self.assertEqual(c["only_ds"], 1)
        self.assertEqual(c["bad_cols"], {"label": 2})
        self.assertEqual(int(E.same_values([1.0, np.nan, 2.0], [1.0 + 1e-13, np.nan, np.nan]).sum()), 2)


# --------------------------------------------------------------------------- #
# 重み・対照
# --------------------------------------------------------------------------- #
class TestWeights(unittest.TestCase):
    def test_run_weights(self):
        w = E.run_weights([5, 5, 9, 2, 2, 2])
        self.assertTrue(np.allclose(w, [0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3]))
        self.assertAlmostEqual(w.sum(), 3.0)                                  # 合計は局面の数
        self.assertEqual(len(E.run_weights([])), 0)

    def test_shuffle_within_date(self):
        rng = np.random.default_rng(0)
        d = pd.to_datetime(np.repeat(pd.bdate_range("2024-01-01", periods=30), 8).tolist() + ["2024-03-01"])
        y = (rng.random(len(d)) < 0.3).astype(float)
        y[-1] = 1.0
        s1 = E.shuffle_within_date(y, d, 7)
        s2 = E.shuffle_within_date(y, d, 7)
        s3 = E.shuffle_within_date(y, d, 8)
        g = pd.DataFrame({"d": d, "y": y, "s": s1}).groupby("d").sum()
        self.assertTrue((g["y"] == g["s"]).all())                             # 日付ごとの正例の数は同じ
        self.assertEqual(s1[-1], 1.0)                                         # 1行だけの日はそのまま
        self.assertTrue(np.array_equal(s1, s2))
        self.assertFalse(np.array_equal(s1, s3))
        self.assertFalse(np.array_equal(s1, y))

    def test_training_set(self):
        rng = np.random.default_rng(1)
        Xf, yf = rng.normal(size=(6, 2)), np.array([1.0, 0, 0, 1, 0, 0])
        fmask = np.array([True, True, True, True, True, False])
        Xa, ya = rng.normal(size=(6, 2)), np.array([1.0, 1, 0, 0, 1, 0])
        runs = np.array([7, 7, 8, 9, 9, 9])
        amask = np.array([True, True, True, True, True, False])              # 局面 9 の1行は学習の期間の外
        X, y, w, info = E.training_set("A", Xf, yf, fmask, Xa, ya, amask, runs, np.random.default_rng(0))
        self.assertIsNone(w)
        self.assertEqual((X.shape, info["extra"]), ((5, 2), 0))
        X, y, w, info = E.training_set("G", Xf, yf, fmask, Xa, ya, amask, runs, np.random.default_rng(0))
        self.assertEqual(X.shape, (10, 2))
        self.assertTrue(np.array_equal(X[5:], Xa[:5]))
        self.assertTrue(np.array_equal(y, np.r_[yf[:5], ya[:5]]))
        self.assertTrue(np.allclose(w, [1, 1, 1, 1, 1, 0.5, 0.5, 1.0, 0.5, 0.5]))  # 局面 9 は学習に入った2行で合計1
        self.assertEqual((info["rows"], info["extra"]), (5, 5))
        self.assertAlmostEqual(info["extra_weight"], 3.0)
        yp = ya[::-1].copy()
        _, y_p, w_p, _ = E.training_set("P", Xf, yf, fmask, Xa, yp, amask, runs, np.random.default_rng(0))
        self.assertTrue(np.array_equal(y_p[5:], yp[:5]))
        self.assertTrue(np.array_equal(w_p, w))
        Xd, yd, wd, info_d = E.training_set("D", Xf, yf, fmask, None, None, amask, runs, np.random.default_rng(3))
        self.assertEqual(Xd.shape, (10, 2))
        self.assertEqual(info_d["extra"], 5)
        self.assertAlmostEqual(wd[5:].sum(), 3.0)                             # 重みの合計は G の足す行と同じ
        self.assertTrue(np.allclose(wd[5:], 0.6))
        for r, lab_ in zip(Xd[5:], yd[5:]):                                   # 足した行はいまの学習の行の複製
            hit = np.flatnonzero((Xf[:5] == r).all(axis=1))
            self.assertEqual(len(hit), 1)
            self.assertEqual(yf[hit[0]], lab_)
        Xd2, _, _, _ = E.training_set("D", Xf, yf, fmask, None, None, amask, runs, np.random.default_rng(3))
        self.assertTrue(np.array_equal(Xd, Xd2))
        with self.assertRaises(ValueError):
            E.training_set("X", Xf, yf, fmask, Xa, ya, amask, runs, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# out-of-fold と報告
# --------------------------------------------------------------------------- #
PARAMS = {"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1, "n_jobs": 1}
COLS = ["f0", "f1", "f2", "f3"]


def make_fresh(seed=0, per_day=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    n = len(dates) * per_day
    df = pd.DataFrame({"Code": [f"{1000 + i % 50}" for i in range(n)], "Date": np.repeat(dates, per_day)})
    for j in range(4):
        df[f"f{j}"] = rng.normal(size=n)
    df["label"] = (df["f1"] + rng.normal(0, 1.0, n) > 1.0).astype(float)
    df["ret_o1_20"] = rng.normal(0.01, 0.1, n)
    df["ret_o1_40"] = rng.normal(0.01, 0.15, n)
    df["hit10"] = (rng.random(n) < 0.3).astype(float)
    df["tp10c"] = np.where(df["hit10"] > 0, 10.0, rng.normal(-1, 8, n))
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


def make_added(fr, seed=0, share=0.5):
    """いまの行の半分ほどに、1〜4日後の足す行（同じ銘柄・特徴量は少しずらす・ラベルは同じ作り）を付ける。局面はいまの行の番号。"""
    rng = np.random.default_rng(seed)
    pick = np.flatnonzero(rng.random(len(fr)) < share)
    k = rng.integers(1, 5, len(pick))
    idx = np.repeat(pick, k)
    off = np.concatenate([np.arange(1, n + 1) for n in k])
    ad = fr.iloc[idx][["Code", "Date"] + COLS].reset_index(drop=True)
    ad["Date"] = (ad["Date"] + pd.to_timedelta(off, unit="D")).astype("datetime64[ns]")
    ad[COLS] = ad[COLS].to_numpy() + rng.normal(0, 0.3, (len(ad), len(COLS)))
    ad["label"] = (ad["f1"] + rng.normal(0, 1.0, len(ad)) > 1.0).astype(float)
    ad["run"] = idx.astype(np.int64)
    ad["offset"] = off
    return ad


class TestOOF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fr = make_fresh()
        cls.ad = make_added(cls.fr)
        cls.folds = E41.folds_for(cls.fr["Date"], 0)

    def test_a_is_production_oof(self):
        """A（いまの行だけ・重みなし）は、本番と同じ作りの E41.oof_folds と同じスコア。足す行を渡しても使わない。"""
        a, sizes, imp = E.oof_arm(self.fr, self.ad, COLS, PARAMS, 42, self.folds, "A")
        b = E41.oof_folds("lgbm", self.fr, COLS, PARAMS, 42, self.folds)
        self.assertGreaterEqual(a["fold"].nunique(), 4)
        self.assertTrue(np.array_equal(a[["Code", "Date", "fold"]].to_numpy(), b[["Code", "Date", "fold"]].to_numpy()))
        self.assertTrue(np.allclose(a["score"].to_numpy(), b["score"].to_numpy(), rtol=0, atol=1e-12))
        self.assertTrue(all(s["extra"] == 0 for s in sizes))
        self.assertEqual(imp, {})
        for c in ("hit10", "tp10c", "ret_o1_20"):
            self.assertIn(c, a.columns)

    def test_arms_differ_and_repeat(self):
        out = {}
        for arm in E.ARMS:
            out[arm], sizes, imp = E.oof_arm(self.fr, self.ad, COLS, PARAMS, 42, self.folds, arm, with_importance=True)
            self.assertEqual(set(imp), set(COLS))
            self.assertAlmostEqual(sum(imp.values()), 1.0, places=6)
            if arm != "A":
                f = sizes[0]
                inside = self.ad[self.ad["Date"] <= pd.Timestamp(f["train_end"])]
                self.assertEqual(f["extra"], len(inside))                     # 学習の期間の中の足す行だけ
                self.assertAlmostEqual(f["extra_weight"], inside["run"].nunique())
        for arm in E.ARMS:                                                    # 評価の行はどの腕も同じ
            self.assertTrue(np.array_equal(out[arm][["Code", "Date", "fold", "label"]].to_numpy(),
                                           out["A"][["Code", "Date", "fold", "label"]].to_numpy()))
        for x, y in (("A", "G"), ("G", "P"), ("A", "D"), ("G", "D")):
            self.assertGreater(np.abs(out[x]["score"] - out[y]["score"]).max(), 1e-6)
        for arm in ("G", "P", "D"):
            again, _, _ = E.oof_arm(self.fr, self.ad, COLS, PARAMS, 42, self.folds, arm)
            self.assertTrue(np.array_equal(again["score"].to_numpy(), out[arm]["score"].to_numpy()))
        p2, _, _ = E.oof_arm(self.fr, self.ad, COLS, PARAMS, 42, self.folds, "P", shift=2)   # ずらしごとに入れ替えの種が違う
        self.assertGreater(np.abs(p2["score"] - out["P"]["score"]).max(), 1e-6)


class TestRunAndReport(unittest.TestCase):
    def test_end_to_end(self):
        """保存済みの探索（A）から、全部の腕の out-of-fold と報告が最後まで通る（種1つ・ずらし0 と 2）。"""
        fr = make_fresh(seed=1)
        ad = make_added(fr, seed=1)
        fr.attrs["built_utc"] = "test"
        stamp = {"built_utc": "test", "rows": len(fr), "added": len(ad), "date_max": "2022-12-30",
                 "checks": {"fresh": {"matched": len(fr), "only_fresh": 0, "only_ds": 0, "bad_cols": {}},
                            "flag": {"diff": 0},
                            "added": {"rows": len(ad), "no_run": 0, "no_label": 0, "used": len(ad),
                                      "runs": int(ad["run"].nunique()), "orphan": 0}}}
        cutoff = pd.Timestamp("2021-12-31")
        tmp = tempfile.mkdtemp()
        old = (E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE)
        try:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = tmp, (42,), "e106test", None
            rec = {"params": PARAMS, "_cv": {"mean_pr_auc": 0.3, "std": 0.01, "mean_roc_auc": 0.6, "fold_scores": [0.3, 0.31]},
                   "_n_trials": 2, "_cutoff": str(cutoff.date()), "_data": "test"}
            with open(E.path("params_A.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                data_out = E.data_report(fr, ad, stamp)
                res = E.run(list(E.ARMS), fr, ad, COLS, cutoff, 2, stamp, [0, 2])
                summary = E.report(res, [0, 2], stamp, data_out, COLS)
                again = E.run(list(E.ARMS), fr, ad, COLS, cutoff, 2, stamp, [0, 2])   # 2回目は保存済みを読む
            out = buf.getvalue()
            saved = sorted(os.listdir(tmp))
        finally:
            E.OOF_DIR, E.SEEDS, E.RUN, E.DEADLINE = old
            shutil.rmtree(tmp)
        self.assertEqual(E.complete_arms(res, [0, 2]), list(E.ARMS))
        j = summary["judge7"]
        self.assertEqual(j["arm"], "G")
        self.assertIn(j["placebo"], E.CONTROLS)
        self.assertEqual(j["mean_pa"], max(j["controls"].values()))           # 対照は P−A と D−A の大きいほう
        self.assertEqual(set(summary["top"]), set(E.ARMS))
        self.assertIn("pr_G-A", summary["windows"])
        self.assertEqual(set(summary["train_size"]), set(E.ARMS))
        self.assertEqual(summary["train_size"]["A"]["last"]["extra"], 0)
        self.assertGreater(summary["train_size"]["G"]["last"]["extra"], 0)
        self.assertIn("top_G", summary["importance"])
        self.assertEqual(data_out["added"], len(ad))
        self.assertIn("■ 4. §7", out)
        self.assertTrue(any(f.endswith("_info.json") for f in saved))
        for arm in E.ARMS:
            self.assertTrue(np.array_equal(again["prod"][arm]["score"].to_numpy(), res["prod"][arm]["score"].to_numpy()))


class TestConstants(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(E.K, 1.5)
        self.assertEqual(E.ARMS, ("A", "G", "P", "D"))
        self.assertEqual(set(E.CONTROLS), {"P", "D"})
        self.assertEqual(set(E.LABELS), set(E.ARMS))
        self.assertEqual(E.SEEDS, (42, 7, 123))
        self.assertEqual(E.PERM_SEED, 20261018)

    def test_columns(self):
        cols = F.columns(E.BASE_PRESET)                 # 回した時点の既定（249列）
        self.assertEqual(len(cols), 249)
        self.assertTrue(set(E.BASE_COLS) <= set(cols))
        self.assertFalse(set(cols) & set(B.FUTURE_COLS))
        self.assertFalse(set(cols) & {"label", "run", "offset", "hit10", "tp10c"})


if __name__ == "__main__":
    unittest.main()
