#!/usr/bin/env python3
"""
実験134（research/exp/e134_chronos2_fixed.py）の部品。データにも Chronos-2 の重みにも触らない（手で解ける小さな例と、本番の記録）。

- 目的変数: +10% はデータセットの label と全行一致しなければ止まり、+15% は同じ材料から引き直す
- 本番の記録: +10%・+15% の LightGBM の labelSpec が今の定義と一致し、列とパラメータが読める。定義違い・列の不足で止まる
- 点数: 固定の到達 θ の p_end・p_reach は実験109 の点数の必要上昇率を θ にしたものと同じ。p_half は20日目に +θ/2 を超える確率
- A: 最初の窓の点数が実験の共通部品（E41.oof_folds）と同じで、保存済みなら読む
- 集計: 判定の言い方（実験109 と同じ）と、報告が通しで動くこと

  python3 tests/test_e134_chronos2_fixed.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e109_chronos2 as E109  # noqa: E402
import e112_chronos2_cols as E112  # noqa: E402
import e134_chronos2_fixed as E  # noqa: E402
import label_specs as LS  # noqa: E402

LV = [0.1, 0.5, 0.9]


def materials(n=6):
    """ラベルの材料（到達・終盤・トレンド・基準の価格）と、+10% のラベル。"""
    f = pd.DataFrame({"future_rise": [0.12, 0.16, 0.09, 0.20, 0.15, 0.11][:n],
                      "end_level": [0.06, 0.08, 0.07, 0.05, 0.10, 0.04][:n],
                      "uptrend_end": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0][:n],
                      "entry_price": [100.0] * n})
    f["label"] = LS.labels(f, "lgbm")
    return f


class Targets(unittest.TestCase):
    def test_labels_by_hand(self):
        f = materials()
        lb = E.target_labels(f)
        # +10%: 到達 ≥ 10%・終盤 ≥ 5%・トレンド。4行目はトレンドが無い、6行目は終盤が足りない
        self.assertEqual(lb["p10"].tolist(), [1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        # +15%: 到達 ≥ 15%・終盤 ≥ 7.5%
        self.assertEqual(lb["p15"].tolist(), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    def test_stops_when_dataset_label_differs(self):
        f = materials()
        f.loc[2, "label"] = 1.0
        with self.assertRaises(ValueError):
            E.target_labels(f)

    def test_stops_without_materials(self):
        f = materials().drop(columns=["end_level"])
        with self.assertRaises(ValueError):
            E.target_labels(f)


class ProductionRecords(unittest.TestCase):
    def test_reads_repository_records(self):
        for key, (name, thr) in E.TARGETS.items():
            with open(E112.meta_path(name), encoding="utf-8") as fh:
                meta = json.load(fh)
            p, cols, info = E.prod_model(name, meta["features"])
            self.assertEqual(info["model"], name)
            self.assertEqual(cols, meta["features"])
            self.assertEqual(meta["labelSpec"]["threshold"], thr)
            self.assertEqual(p["n_estimators"], E112.LGBM_FIXED["n_estimators"] if "n_estimators" not in meta["params"]
                             else meta["params"]["n_estimators"])
            self.assertEqual(p["objective"], "binary")

    def test_stops_on_wrong_label_or_missing_columns(self):
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "models", "lgbm_k15"))
            meta = {"labelSpec": LS.for_model("lgbm"), "features": ["a", "b"], "params": {"num_leaves": 7}}
            with open(os.path.join(tmp, "models", "lgbm_k15", "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            with self.assertRaises(SystemExit):           # +15% の記録に +10% の定義
                E.prod_model("lgbm_k15", ["a", "b"], root=tmp)
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            p, cols, _ = E.prod_model("lgbm", ["a", "b", "c"], root=tmp)
            self.assertEqual((cols, p["num_leaves"], p["subsample_freq"]), (["a", "b"], 7, 1))
            with self.assertRaises(SystemExit):           # 記録の列が表に無い
                E.prod_model("lgbm", ["a"], root=tmp)
        finally:
            shutil.rmtree(tmp)


class Scores(unittest.TestCase):
    def test_fixed_threshold_by_hand(self):
        # 2行・水準3つ・先2日。基準 log C[t] = 0。20日目（最後の先）の分位点で数える
        q = np.array([[[0.00, 0.05], [0.10, 0.10], [0.20, 0.15]],
                      [[0.00, 0.00], [0.00, 0.05], [0.00, 0.30]]])
        last = np.zeros(2)
        sc = E.fixed_scores(q, LV, last, 0.10)
        x = np.log1p(0.10)                                 # 0.0953
        want_end = [1 - np.interp(x, [0.05, 0.10, 0.15], LV), 1 - np.interp(x, [0.0, 0.05, 0.30], LV)]
        np.testing.assert_allclose(sc["p_end"], want_end)
        # 1行目は1日目の方が届きやすい（分位点が 0〜0.20 に広い）
        want_d1 = 1 - np.interp(x, [0.0, 0.10, 0.20], LV)
        self.assertAlmostEqual(sc["p_reach"].iloc[0], max(want_d1, want_end[0]))
        xh = np.log1p(0.05)
        np.testing.assert_allclose(sc["p_half"], [1 - np.interp(xh, [0.05, 0.10, 0.15], LV),
                                                  1 - np.interp(xh, [0.0, 0.05, 0.30], LV)])
        # 実験109 の点数の必要上昇率を全行 0.10 にしたものと同じ
        ref = E109.scores(q, LV, last, np.full(2, 0.10))
        np.testing.assert_allclose(sc["p_end"], ref["p_end"])
        np.testing.assert_allclose(sc["p_reach"], ref["p_reach"])


def synthetic_frame(n_per_month=45, seed=0):
    """2016-01〜2022-12 の月ごとの行。label は x0 が大きいほど正になりやすい。"""
    rng = np.random.default_rng(seed)
    dates = []
    for m in pd.date_range("2016-01-01", "2022-12-01", freq="MS"):
        days = pd.bdate_range(m, m + pd.offsets.MonthEnd(0))
        dates.extend(rng.choice(days, n_per_month))
    n = len(dates)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.8 * rng.normal(size=n) > 0.9).astype(float)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df.insert(0, "Code", [f"{1000 + i % 300}0" for i in range(n)])
    df.insert(1, "Date", pd.to_datetime(dates).astype("datetime64[ns]"))
    df["label"] = y
    return df.sort_values(["Date", "Code"], kind="mergesort").reset_index(drop=True)


class ArmA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = (E.OOF_DIR, E.SEEDS)
        E.OOF_DIR = self.tmp

    def tearDown(self):
        E.OOF_DIR, E.SEEDS = self.old
        shutil.rmtree(self.tmp)

    def test_out_of_fold_matches_common_part_and_is_saved(self):
        import e41_stop_loss as E41

        df = synthetic_frame()
        cols = ["x0", "x1", "x2", "x3"]
        params = E112.full_lgbm({"learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 20, "n_estimators": 30})
        folds = E41.folds_for(df["Date"], 0)
        E.SEEDS = (42, 7)
        with contextlib.redirect_stdout(io.StringIO()):
            o = E.arm_a("p10", df, cols, params, folds)
        self.assertGreater(o["fold"].nunique(), 1)
        self.assertEqual(set(o.columns), {"Code", "Date", "label", "score", "score42", "fold"})
        # 種42 の点数は共通部品の種42 と同じ、平均は2つの種の平均
        f = next(x for x in folds if E109.fold_rows(df, x) is not None)
        want = E41.oof_folds("lgbm", df, cols, params, 42, [f])["score"].to_numpy()
        np.testing.assert_allclose(o.loc[o["fold"] == f.index, "score42"].to_numpy(), want, rtol=0, atol=1e-12)
        s7 = E109.fold_fit(df, cols, params, 7, f)[1]
        np.testing.assert_allclose(o.loc[o["fold"] == f.index, "score"].to_numpy(), (want + s7) / 2, atol=1e-12)
        # 2回目は保存済みを読む（同じ表・同じパラメータ）
        with contextlib.redirect_stdout(io.StringIO()) as out:
            o2 = E.arm_a("p10", df, cols, params, folds)
        self.assertIn("保存済み", out.getvalue())
        pd.testing.assert_frame_equal(o, o2)


def synthetic_oof(strong_z0: bool, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(6):
        days = pd.bdate_range(f"2021-{f + 1:02d}-01", periods=40)
        for d in days:
            for c in range(8):
                y = float(rng.random() < 0.2)
                sig = rng.normal() + (1.5 * y if strong_z0 else 0.0)
                rows.append({"Code": f"{c}", "Date": d, "fold": f, "label": y,
                             "s_A": rng.normal() + 0.8 * y, "s_Z0": sig})
    o = pd.DataFrame(rows)
    o["Date"] = o["Date"].astype("datetime64[ns]")
    return o


class Judge(unittest.TestCase):
    def setUp(self):
        self.old = E.N_BOOT
        E.N_BOOT = 200

    def tearDown(self):
        E.N_BOOT = self.old

    def test_verdicts(self):
        up = E.judge(synthetic_oof(True))
        self.assertEqual(up["pair"]["verdict"], "上回る")
        self.assertTrue(up["arms"]["Z0"]["skill"])
        down = E.judge(synthetic_oof(False))
        self.assertEqual(down["pair"]["verdict"], "下回る")
        self.assertEqual(down["pair"]["n"], 6)

    def test_report_runs_through(self):
        o = synthetic_oof(False)
        o["score"] = o["s_A"]
        o["score42"] = o["s_A"]
        o["hit10"] = o["label"]
        o["tp10c"] = np.where(o["label"] == 1, 10.0, -1.0)
        n = len(o)
        sc = pd.DataFrame({"p_end": np.linspace(0.01, 0.3, n), "p_reach": np.linspace(0.02, 0.4, n),
                           "p_half": np.linspace(0.05, 0.5, n), "med": np.linspace(-0.1, 0.1, n),
                           "clip_hi": np.zeros(n, dtype=bool), "crossed": np.zeros(n, dtype=bool)})
        info = {"model": "lgbm", "label": LS.name("lgbm"), "preset": "x", "n_features": 4, "trained": "2026-11-27",
                "hash": "h", "oof": {"prAuc": 0.24, "rocAuc": 0.61, "n": 100}}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = E.report_target("p10", o, sc, np.full(n, 0.05), info)
        self.assertIn("判定", out.getvalue())
        self.assertEqual(set(res["top2"]), {"A", "Z0"})
        self.assertEqual(res["post"]["rows"], 0)
        json.dumps(res, default=str)


def synthetic_world(n_codes=40, n_rows=3000, seed=3):
    """作りものの日足（2016-01〜2023-06）と、その上の学習データの行（材料・ラベル・close_position・特徴量4列）。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-01", "2023-06-30")
    parts = []
    for c in range(n_codes):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        hi = close * (1 + rng.uniform(0.001, 0.03, len(days)))
        lo = close * (1 - rng.uniform(0.001, 0.03, len(days)))
        parts.append(pd.DataFrame({"Code": f"{1300 + c}0", "Date": days, "open": close, "high": hi, "low": lo,
                                   "close": close}))
    panel = pd.concat(parts, ignore_index=True)
    panel["Date"] = panel["Date"].astype("datetime64[ns]")
    usable = panel[panel["Date"] <= days[-30]]
    pick = usable.sample(n_rows, random_state=seed).sort_values(["Code", "Date"]).reset_index(drop=True)
    x = rng.normal(size=(n_rows, 4))
    f = pd.DataFrame(x, columns=[f"x{i}" for i in range(4)])
    f.insert(0, "Code", pick["Code"].to_numpy())
    f.insert(1, "Date", pick["Date"].to_numpy())
    f["close_position"] = (pick["close"] - pick["low"]) / (pick["high"] - pick["low"]) * 100.0
    f["vol_20d"] = rng.uniform(1.5, 4.0, n_rows)
    f["ret_o1_20"] = rng.normal(0, 8, n_rows)
    f["future_rise"] = 0.06 + 0.06 * x[:, 0] + rng.normal(0, 0.06, n_rows)
    f["end_level"] = f["future_rise"] * 0.6 + rng.normal(0, 0.02, n_rows)
    f["uptrend_end"] = (rng.random(n_rows) < 0.8).astype(float)
    f["entry_price"] = 100.0
    f["label"] = LS.labels(f, "lgbm")
    return panel, f


class MainEndToEnd(unittest.TestCase):
    """データと Chronos-2 を作りものに替えて、main を最後まで通す（つなぎの確かめ。精度は見ない）。"""

    def setUp(self):
        import e83_tp10_rule as E83
        import e105_technical_ab as E105
        import lab
        import live_track as LT

        self.tmp = tempfile.mkdtemp()
        self.mods = (lab, E105, LT, E83, E112)
        self.saved = (lab.frame, E105.build_panel, LT.load_bars, E83.attach_exits, E112.z0_quantiles, E112.TAG,
                      E.OOF_DIR, E.MODEL_DIR, E.N_BOOT, E.SEEDS)
        panel, frame = synthetic_world()
        model_dir = os.path.join(self.tmp, "model")
        os.makedirs(os.path.join(model_dir, "models", "lgbm_k15"))
        for name, p in (("lgbm", os.path.join(model_dir, "meta.json")),
                        ("lgbm_k15", os.path.join(model_dir, "models", "lgbm_k15", "meta.json"))):
            meta = {"labelSpec": LS.for_model(name), "label": LS.name(name), "features": ["x0", "x1", "x2", "x3"],
                    "preset": "syn", "trainedAt": "2026-11-27T00:00:00", "oofMetrics": {"prAuc": 0.2, "n": 10},
                    "params": {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20, "n_estimators": 20}}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)

        def exits(df, bars, targets=(10,), hold=20):
            k = E83.tp_key(targets[0])
            ex = df[["Code", "Date"]].copy()
            ex[f"hit{k}"] = (np.arange(len(df)) % 3 == 0).astype(float)
            ex[f"tp{k}c"] = np.where(ex[f"hit{k}"] == 1, 10.0, -2.0)
            return ex, {"hold5_match": len(df), "hold5_both": len(df), "order_violations": 0}

        def z0(rows, pnl):
            n = len(rows)
            rng = np.random.default_rng(5)
            mid = rng.normal(0, 0.03, (n, 1, 20))
            q = np.concatenate([mid - 0.08, mid, mid + 0.08], axis=1).astype(np.float32)
            start, end = E109.context_bounds(pnl, rows, E.CONTEXT)
            q += np.log(pnl["close"].to_numpy()[end])[:, None, None]
            return q, [0.1, 0.5, 0.9], {"chunks": 1, "computed": n, "seconds": 0.0}

        lab.frame = lambda rebuild=False: frame.copy()
        E105.build_panel = lambda data_dir, codes: panel.copy()
        LT.load_bars = lambda data_dir, start=None: pd.DataFrame({"Date": pd.to_datetime([])})
        E83.attach_exits = exits
        E112.z0_quantiles = z0
        E.OOF_DIR, E.MODEL_DIR, E.N_BOOT, E.SEEDS = self.tmp, model_dir, 100, (42, 7)

    def tearDown(self):
        lab, E105, LT, E83, E112_ = self.mods
        (lab.frame, E105.build_panel, LT.load_bars, E83.attach_exits, E112_.z0_quantiles, E112_.TAG,
         E.OOF_DIR, E.MODEL_DIR, E.N_BOOT, E.SEEDS) = self.saved
        shutil.rmtree(self.tmp)

    def test_main_runs_both_targets(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(E.main([]), 0)
        text = out.getvalue()
        self.assertIn("目的変数 +10%", text)
        self.assertIn("目的変数 +15%", text)
        self.assertIn("実験109 とのつなぎ", text)
        with open(os.path.join(self.tmp, "e134_summary.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertEqual(set(s["targets"]), {"p10", "p15"})
        for k in ("p10", "p15"):
            self.assertIn(s["targets"][k]["pair"]["verdict"], ("上回る", "下回る", "決められない"))
        self.assertGreater(s["link_e109"]["rows"], 0)
        self.assertEqual(E112.TAG, "e134")                 # 分位点の置き場はこの実験の名前

    def test_dry_stops_before_chronos(self):
        E112.z0_quantiles = None                           # 呼ばれたら落ちる
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(E.main(["--dry", "--targets", "p15"]), 0)
        self.assertIn("[dry]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
