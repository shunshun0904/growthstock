#!/usr/bin/env python3
"""
実験116（research/exp/e116_fins_pit_ab.py）。合成のデータで:

- 2つの版の読み込み: 行の並び・ラベル・列が違えば止める（列の並びは今の版にそろえる）
- 値が変わった印: 数値（欠測どうしは同じ・相対 1e-9）・文字・日付
- 変わった行と列の数え方（出し直しの窓の中 / 外）
- out-of-fold の3つの腕: A0 は今の版で学習・今の版で採点、A は今の版で学習・直す版で採点、B は直す版で学習・直す版で採点。
  学習は窓の訓練の最終日以前のラベルのある行。保存したものは次の回に読む
- 窓ごとの PR-AUC・差・読み方（大きさ 0.002 以上かつ標準誤差の2倍以上で「変わる」）
- 直す版は build_dataset.py --pit-fins を別のプロセスで作る（失敗したら止める）
- 通しで回す（直す版・木の形・決算は差し替え、当てはめは偽物）

  python3 tests/test_e116_fins_pit_ab.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import e116_fins_pit_ab as X  # noqa: E402


def two_versions(seed=0, start="2016-01-04", end="2021-12-31", per_day=12):
    """行・ラベルが同じ2つの版。直す版は、ある銘柄の一部の行だけ決算の列 fin が違う。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    cur = pd.DataFrame({"Date": np.repeat(days, per_day),
                        "Code": np.tile([f"{6000 + j}" for j in range(per_day)], len(days))})
    n = len(cur)
    z = rng.normal(size=n)
    cur["f1"] = z + rng.normal(0, 1, n)
    cur["fin"] = rng.normal(size=n)
    cur["name"] = "x"
    cur["DiscDate"] = cur["Date"] - pd.Timedelta(days=30)
    cur["label"] = (z > 0.9).astype(float)
    pit = cur.copy()
    hit = (cur["Code"] == "6001") & (cur["Date"] >= "2019-03-01") & (cur["Date"] < "2019-04-01")
    pit.loc[hit, "fin"] = pit.loc[hit, "fin"] + 1.0
    pit.loc[hit, "DiscDate"] = pit.loc[hit, "DiscDate"] - pd.Timedelta(days=60)
    win = pd.DataFrame({"Code": ["6001"], "start": [pd.Timestamp("2019-03-01")], "end": [pd.Timestamp("2019-04-01")]})
    return cur, pit, win, hit.to_numpy()


class TestPair(unittest.TestCase):
    def test_load_pair(self):
        cur, pit, _, _ = two_versions(end="2016-03-31")
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.parquet"), os.path.join(tmp, "b.parquet")
            cur.to_parquet(a, index=False)
            pit[list(reversed(pit.columns))].sample(frac=1.0, random_state=0).to_parquet(b, index=False)
            c2, p2 = X.load_pair(a, b)
            self.assertEqual(list(p2.columns), list(c2.columns))      # 列と行の並びをそろえる
            self.assertTrue((c2["Code"].to_numpy() == p2["Code"].to_numpy()).all())
            pit.iloc[:-1].to_parquet(b, index=False)
            with self.assertRaises(SystemExit):
                X.load_pair(a, b)
            bad = pit.copy()
            bad.loc[3, "label"] = 1.0 - bad.loc[3, "label"]
            bad.to_parquet(b, index=False)
            with self.assertRaises(SystemExit):
                X.load_pair(a, b)
            pit.drop(columns=["name"]).to_parquet(b, index=False)
            with self.assertRaises(SystemExit):
                X.load_pair(a, b)

    def test_changed_cells(self):
        a = pd.DataFrame({"x": [1.0, np.nan, 2.0, 3.0], "s": ["a", "b", None, "c"],
                          "d": pd.to_datetime(["2024-01-01", None, "2024-01-03", "2024-01-04"])})
        b = pd.DataFrame({"x": [1.0 + 1e-12, np.nan, np.nan, 3.1], "s": ["a", "c", None, "c"],
                          "d": pd.to_datetime(["2024-01-01", None, "2024-01-02", "2024-01-04"])})
        ch = X.changed_cells(a, b, ["x", "s", "d"])
        self.assertEqual(ch["x"].tolist(), [False, False, True, True])
        self.assertEqual(ch["s"].tolist(), [False, True, False, False])
        self.assertEqual(ch["d"].tolist(), [False, False, True, False])

    def test_change_report(self):
        cur, pit, win, hit = two_versions(end="2019-12-31")
        pit.loc[0, "f1"] += 1.0                                       # 窓の外で動いた列（順位の列の代わり）
        rep = X.change_report(cur, pit, hit, ["f1", "fin"])
        self.assertEqual(rep["window_rows"], int(hit.sum()))
        self.assertEqual(rep["changed_in_windows"], int(hit.sum()))
        self.assertEqual(rep["changed_outside"], 1)
        self.assertEqual(rep["cols_outside"], {"f1": 1})
        self.assertEqual(rep["cols_changed"]["fin"], int(hit.sum()))
        self.assertEqual(rep["cols_changed"]["DiscDate"], int(hit.sum()))
        self.assertEqual(rep["model_changed_rows"], int(hit.sum()) + 1)


class TestOof(unittest.TestCase):
    def test_arms_and_cache(self):
        cur, pit, _, _ = two_versions()
        calls = []

        def fake(algo, params, tr, tes, cols):
            src = "pit" if np.allclose(tr["fin"].sum(), pit.loc[tr.index, "fin"].sum()) and \
                not np.allclose(tr["fin"].sum(), cur.loc[tr.index, "fin"].sum()) else "cur"
            calls.append((src, tr["Date"].max(), [t["Date"].min() for t in tes], [float(t["fin"].sum()) for t in tes],
                          bool(tr["label"].notna().all())))
            return [t["fin"].to_numpy(dtype=float) for t in tes]          # 採点した版の fin をスコアにする

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(X, "fit_predict", side_effect=fake):
            o = X.oof_pair(cur, pit, ["f1", "fin"], "lgbm", {}, 0, "t", out_dir=tmp)
            n = o["fold"].nunique()
            self.assertGreater(n, 1)
            self.assertEqual(len(calls), 2 * n)                        # 窓ごとに今の版・直す版で1回ずつ学習
            folds = {f.index: f for f in X.E41.folds_for(cur["Date"], 0)}
            for k, f in enumerate(sorted(o["fold"].unique())):
                c_cur, c_pit = calls[2 * k], calls[2 * k + 1]
                end = pd.Timestamp(folds[f].train_end)
                self.assertLessEqual(c_cur[1], end)
                self.assertLessEqual(c_pit[1], end)
                self.assertGreater(min(c_cur[2]), end)
                self.assertTrue(c_cur[4] and c_pit[4])
            m = o.merge(cur[["Code", "Date", "fin"]], on=["Code", "Date"]).merge(
                pit[["Code", "Date", "fin"]], on=["Code", "Date"], suffixes=("_cur", "_pit"))
            np.testing.assert_allclose(m["score_A0"], m["fin_cur"])     # 今の版を採点
            np.testing.assert_allclose(m["score_A"], m["fin_pit"])      # 直す版を採点
            np.testing.assert_allclose(m["score_B"], m["fin_pit"])
            k0 = len(calls)
            o2 = X.oof_pair(cur, pit, ["f1", "fin"], "lgbm", {}, 0, "t", out_dir=tmp)
            self.assertEqual(len(calls), k0)                           # 保存したものを読む
            pd.testing.assert_frame_equal(o, o2)


class TestScores(unittest.TestCase):
    def test_window_scores_and_reading(self):
        rng = np.random.default_rng(1)
        rows = []
        for f in (1, 2, 3, 4):
            y = (rng.random(300) < 0.2).astype(float)
            s = y + rng.normal(0, 1, 300)
            rows.append(pd.DataFrame({"fold": f, "label": y, "score_A0": s + 0.3 * y, "score_A": s,
                                      "score_B": s + 0.1 * y}))
        rows.append(pd.DataFrame({"fold": 5, "label": np.zeros(50), "score_A0": 0.0, "score_A": 0.0, "score_B": 0.0}))
        o = pd.concat(rows, ignore_index=True)
        ws = X.window_scores(o)
        self.assertEqual(list(ws["fold"]), [1, 2, 3, 4])               # 正例の無い窓は数えない
        d = X.paired(ws, "B", "A")
        self.assertEqual(d["n"], 4)
        self.assertGreater(d["mean"], 0)
        self.assertEqual(X.paired(ws, "A0", "A")["won"], 4)
        p = X.pooled(o[o["fold"] < 5])
        self.assertGreater(p["A0"]["pr"], p["A"]["pr"])
        self.assertEqual(X.verdict_of({"mean": 0.003, "se": 0.001}), "up")
        self.assertEqual(X.verdict_of({"mean": -0.003, "se": 0.001}), "down")
        self.assertEqual(X.verdict_of({"mean": 0.0015, "se": 0.0001}), "same")    # 小さい
        self.assertEqual(X.verdict_of({"mean": 0.003, "se": 0.002}), "same")      # 標準誤差の2倍に届かない
        self.assertEqual(X.verdict_of({"mean": float("nan"), "se": 0.001}), "same")
        self.assertEqual(X.reading(["same", "same"]), "same")
        self.assertEqual(X.reading(["same", "down"]), "down")
        self.assertEqual(X.reading(["up", "same"]), "up")
        self.assertEqual(X.reading(["up", "down"]), "mixed")


class TestBuildPit(unittest.TestCase):
    def test_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "pit", "dataset.parquet")

            def ok(cmd, **kw):
                self.assertIn("--pit-fins", cmd)
                self.assertEqual(cmd[cmd.index("--out") + 1], out)
                with open(os.path.join(tmp, "pit", "dataset_meta.json"), "w", encoding="utf-8") as fh:
                    json.dump({"pitFins": True}, fh)
                return mock.Mock(returncode=0, stdout="[merge] 決算の出し直し: 1\n[other] x\n")

            buf = io.StringIO()
            with mock.patch("subprocess.run", side_effect=ok), contextlib.redirect_stdout(buf):
                meta = X.build_pit("d", out)
            self.assertEqual(meta, {"pitFins": True})
            self.assertIn("（直す版）[merge]", buf.getvalue())
            self.assertNotIn("[other]", buf.getvalue())
            with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stdout="boom")), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                X.build_pit("d", out)


class TestMain(unittest.TestCase):
    def test_main(self):
        cur, pit, win, _ = two_versions()
        with tempfile.TemporaryDirectory() as tmp:
            cur_path = os.path.join(tmp, "dataset.parquet")
            pit_path = os.path.join(tmp, "pit", "dataset.parquet")
            os.makedirs(os.path.dirname(pit_path))
            cur.to_parquet(cur_path, index=False)
            pit.to_parquet(pit_path, index=False)
            with open(os.path.join(tmp, "dataset_meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"pitFins": False}, fh)
            with open(os.path.join(tmp, "pit", "dataset_meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"pitFins": True}, fh)

            def fake(algo, params, tr, tes, cols):
                rng = np.random.default_rng(len(tr))
                return [t["f1"].to_numpy(dtype=float) + rng.normal(0, 0.01, len(t)) for t in tes]

            out = io.StringIO()
            with mock.patch.object(X, "OOF_DIR", tmp), \
                    mock.patch.object(X, "SUMMARY_PATH", os.path.join(tmp, "e116_summary.json")), \
                    mock.patch.object(X, "PIT_PATH", pit_path), \
                    mock.patch.object(X.lab, "DATASET", cur_path), \
                    mock.patch.object(X.F, "columns", return_value=["f1", "fin"]), \
                    mock.patch.object(X.RB, "prod_params", return_value={"lgbm": {}, "xgb": {}, "cat": {}}), \
                    mock.patch.object(X.BD, "load_parts", return_value=pd.DataFrame()), \
                    mock.patch.object(X.BD, "fin_versions", return_value=pd.DataFrame()), \
                    mock.patch.object(X.BD, "reissue_windows", return_value=win), \
                    mock.patch.object(X, "build_pit") as bp, \
                    mock.patch.object(X, "fit_predict", side_effect=fake), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(X.main(["--shifts", "0", "--algos", "lgbm,cat", "--skip-build"]), 0)
                bp.assert_not_called()
            with open(os.path.join(tmp, "e116_summary.json"), encoding="utf-8") as fh:
                sm = json.load(fh)
        self.assertEqual(sm["report"]["changed_in_windows"], sm["report"]["window_rows"])
        self.assertEqual(sorted(sm["shifts"]), ["cat", "lgbm"])
        self.assertIn(sm["reading"]["overall"], X.READING_JA)
        self.assertIn("読み方", out.getvalue())
        self.assertNotIn("6001", out.getvalue())                    # 公開ログに銘柄コードを出さない


if __name__ == "__main__":
    unittest.main()
