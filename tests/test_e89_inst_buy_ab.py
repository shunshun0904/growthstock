#!/usr/bin/env python3
"""
実験89（research/exp/e89_inst_buy_ab.py）の、重い計算を回さない部分のテスト。

- 比べるモデル・票・種・ラベル・判定の行（回す前に決めたもの）
- ラベルの引き直し（1.2σ はデータセットのまま、1.5σ は B.rise_label_for_k。決まらない行は落とす）
- 評価の行の組（列の分かる行 = inst_buy_60 が欠測でない行）
- 窓ごとの AUC に期間・件数・正例率が付き、少ない窓は落ちる
- 合成の結果で、表（探索の CV・本番と同じ OOF・窓ごと・§7・窓ごとの値の一覧）が最後まで出て、要約が保存される
- 時間の上限を過ぎたら新しい計算を始めない・知らない --k は止める

  python3 tests/test_e89_inst_buy_ab.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "research", "exp"))

import features as F  # noqa: E402
import inst_features as I  # noqa: E402
import e89_inst_buy_ab as E  # noqa: E402


class TestLineup(unittest.TestCase):
    def test_decided_before_running(self):
        self.assertEqual(E.ALGOS, ("lgbm", "xgb", "cat", "logit"))
        self.assertEqual(E.JUDGE_ALGOS, ("lgbm", "xgb", "cat"))
        self.assertEqual(E.SEEDS["logit"], (42,))
        self.assertTrue(all(len(E.SEEDS[a]) == 3 for a in E.JUDGE_ALGOS))
        self.assertEqual(E.KS, (1.5, 1.2))
        self.assertEqual(E.PRIMARY_ROWS, "covered")
        self.assertEqual(E.COVER_COL, "inst_buy_60")
        self.assertEqual(E.PERM_SEED, 20261013)
        self.assertEqual((E.k_tag(1.5), E.k_tag(1.2)), ("k15", "k12"))

    def test_new_columns_are_not_production_columns(self):
        base = F.columns(F.DEFAULT_PRESET)
        self.assertEqual(len(base), 249)            # 2026-10-18 から（実験89 の時点は239列）
        self.assertFalse(set(base) & set(I.COLUMNS))
        self.assertFalse(any(c.startswith("inst_") for c in base))         # 本番の設定には書いていない

    def test_unknown_k_stops(self):
        with self.assertRaises(SystemExit):
            E.main(["--k", "1.8", "--dry"])


def label_frame(n=400, seed=0):
    """ラベルの材料（vol_20d・future_rise・end_level・uptrend_end）を持つ合成の表。"""
    import build_dataset as B
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1.5, 4.0, n)
    need12, end12 = B.rise_thresholds(pd.Series(vol), B.DEFAULT_RISE)
    rise = need12.to_numpy() * rng.uniform(0.5, 1.8, n)
    end = end12.to_numpy() * rng.uniform(0.5, 2.5, n)
    up = rng.integers(0, 2, n).astype(float)
    lab12 = ((rise >= need12.to_numpy()) & (end >= end12.to_numpy()) & (up == 1)).astype(float)
    lab12[:5] = np.nan                                                      # 未確定の行
    return pd.DataFrame({"Code": [f"{1000 + i % 20}0" for i in range(n)],
                         "Date": pd.bdate_range("2023-01-04", periods=n),
                         "vol_20d": vol, "future_rise": rise, "end_level": end, "uptrend_end": up,
                         "label": lab12})


class TestLabels(unittest.TestCase):
    def test_k12_keeps_the_dataset_label(self):
        df = label_frame()
        out = E.with_label(df, 1.2)
        self.assertEqual(len(out), len(df) - 5)
        np.testing.assert_array_equal(out["label"].to_numpy(), df["label"].dropna().to_numpy())

    def test_k15_is_stricter_and_matches_the_rule(self):
        import build_dataset as B
        df = label_frame()
        out = E.with_label(df, 1.5)
        self.assertEqual(len(out), len(df) - 5)
        self.assertLess(out["label"].sum(), df["label"].sum())               # しきい値が上がれば正例は減る
        self.assertTrue(((out["label"] == 1) <= (E.with_label(df, 1.2)["label"] == 1)).all())
        np.testing.assert_array_equal(out["label"].to_numpy(),
                                      B.rise_label_for_k(df, 1.5).dropna().to_numpy())
        self.assertTrue(df["label"].isna().sum() == 5)                       # 元の表は変えない


def synthetic_oof(seed, shift=0, n_folds=3, per_day=10, days=60, strength=0.4):
    """
    fold ごとに日付が分かれた合成の out-of-fold（Code・Date・label・score・fold・ret_o1_20/40）。1窓 600行
    （上位10% の超過は前の窓に 500件以上要る: lab.threshold_edge）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2021-06-01", periods=n_folds * days)
    for f in range(n_folds):
        for d in dates[f * days:(f + 1) * days]:
            for c in range(per_day):
                y = int(rng.random() < 0.25)
                rows.append({"Code": f"{1000 + c}0", "Date": d, "label": y, "fold": f + shift,
                             "score": y * strength + rng.random(), "ret_o1_20": rng.normal(1 + 3 * y, 5),
                             "ret_o1_40": rng.normal(1 + 4 * y, 7)})
    return pd.DataFrame(rows)


class TestWindows(unittest.TestCase):
    def test_windows_have_period_count_and_rate(self):
        o = synthetic_oof(1)
        w = E.windows_on(o, None)
        self.assertEqual(list(w.columns), ["fold", "pr", "roc", "start", "end", "n", "pos"])
        self.assertEqual(w["n"].tolist(), [600, 600, 600])
        self.assertEqual(str(pd.Timestamp(w["start"].iloc[0]).date()), "2021-06-01")
        self.assertTrue(((w["pos"] > 0) & (w["pos"] < 1)).all())

    def test_small_windows_are_dropped_on_a_keyset(self):
        o = synthetic_oof(1)
        # 列の分かる行: fold 0 は 60行だけ（100件未満なので落ちる）、fold 1・2 は全部
        keep = pd.concat([o[o["fold"] == 0].head(60), o[o["fold"] > 0]])
        w = E.windows_on(o, E.keys_of(keep))
        self.assertEqual(w["fold"].tolist(), [1, 2])
        self.assertEqual(len(E.windows_on(o.iloc[:0], None)), 0)

    def test_eval_keysets(self):
        df = pd.DataFrame({"Code": ["10000", "20000", "30000"], "Date": pd.to_datetime(["2021-07-01"] * 3),
                           "inst_buy_20": [np.nan, 1.0, 0.0], "inst_buy_60": [np.nan, np.nan, 0.0]})
        ks = E.eval_keysets(df)
        self.assertEqual(list(ks), ["all", "covered"])
        self.assertIsNone(ks["all"])
        self.assertEqual(len(ks["covered"]), 1)                              # inst_buy_60 が分かる行だけ


class TestReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN)
        E.OOF_DIR, E.RUN = self.dir, "e89test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def results(self, algos=("lgbm", "logit"), shifts=(0, 2)):
        allo = synthetic_oof(0)
        keysets = {"all": None, "covered": E.keys_of(allo[allo["fold"] > 0])}
        out = {}
        for i, a in enumerate(algos):
            cv = {"mean_pr_auc": 0.33, "std": 0.02, "mean_roc_auc": 0.64, "fold_scores": [0.31, 0.32, 0.33, 0.34, 0.35]}
            cvb = {**cv, "mean_pr_auc": 0.335, "fold_scores": [0.32, 0.32, 0.34, 0.33, 0.36]}
            recs = {"A": {"params": {"learning_rate": 0.05, "num_leaves": 31}, "_cv": cv, "_seconds": 10},
                    "B": {"params": {"learning_rate": 0.04, "num_leaves": 15}, "_cv": cvb, "_seconds": 12}}
            prod = {rs: {} for rs in keysets}
            wins = {rs: {} for rs in keysets}
            for j, arm in enumerate(E.ARMS):
                o = synthetic_oof(10 * i + j, strength={"A": 0.4, "B": 0.45, "C": 0.5, "P": 0.38}[arm])
                for rs, keys in keysets.items():
                    prod[rs][arm] = o if keys is None else E.on_rows(o, keys)
                for sh in shifts:
                    o2 = synthetic_oof(100 * i + 10 * j + sh, shift=0,
                                       strength={"A": 0.4, "B": 0.45, "C": 0.5, "P": 0.38}[arm])
                    for rs, keys in keysets.items():
                        wins[rs][(sh, arm)] = E.windows_on(o2, keys)
            out[a] = {"recs": recs, "params": {arm: recs[E.PARAMS_OF[arm]]["params"] for arm in E.ARMS},
                      "prod": prod, "wins": wins}
        return out, keysets

    def test_report_prints_every_section_and_saves_the_summary(self):
        res, keysets = self.results()
        stamp = {"built_utc": "2026-10-13 21:00 UTC", "rows": 540, "covered": 360, "covered_from": "2021-07-13",
                 "rows_by_year": {"2021": 540}, "covered_by_year": {"2021": 360},
                 "ones_by_year": {"inst_buy_20": {"2021": 30}, "inst_buy_60": {"2021": 60}}, "inst": {}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = E.report(res, [0, 2], stamp, keysets, {"label_k": 1.5}, 1.5)
        text = buf.getvalue()
        for head in ("■ 1.", "■ 2a.", "■ 2b.", "■ 3a.", "■ 3b.", "■ 4a.", "■ 4b.", "■ 5a.", "■ 5b."):
            self.assertIn(head, text)
        self.assertIn("判定に使う行（回す前に決めた）", text)                 # covered が判定、all は参考
        self.assertIn("木3モデルがそろってから判定", text)                    # xgb・cat がまだ
        self.assertIn("lgbm・ずらし2か月（3窓）", text)
        self.assertIn("2021-06-01〜", text)                                   # 窓の期間
        self.assertNotIn("nan", text.lower().replace("nan（", ""))
        # 窓ごとの値の一覧は、全行で 2モデル × 2通り × 3窓、列の分かる行で 2窓ずつ
        self.assertEqual(len(s["per_window"]["all"]), 12)
        self.assertEqual(len(s["per_window"]["covered"]), 8)
        self.assertIn("lgbm_covered", s["judge7"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "e89test_k15_summary.json")))
        with open(os.path.join(self.dir, "e89test_k15_summary.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["label_k"], 1.5)
        self.assertEqual(saved["primary_rows"], "covered")


class TestTimeBudget(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT)
        E.OOF_DIR, E.RUN = self.dir, "e89test_k15"

    def tearDown(self):
        E.OOF_DIR, E.RUN, E.DEADLINE, E.TIMED_OUT = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_new_after_the_deadline(self):
        frame = pd.DataFrame({"Code": ["10000"] * 3, "Date": pd.bdate_range("2024-01-04", periods=3),
                              "label": [1, 0, 1], "x": [1.0, 2.0, 3.0]})
        frame.attrs["built_utc"] = "2026-10-13 00:00 UTC"
        E.DEADLINE, E.TIMED_OUT = time.time() - 1, False
        self.assertIsNone(E.oof("lgbm", "new", frame, ["x"], {"n_estimators": 1}, 0, (42,)))
        self.assertIsNone(E.tune("lgbm", "A", frame, ["x"], pd.Timestamp("2024-01-10"), 2,
                                 {"built_utc": "2026-10-13 00:00 UTC"}))
        self.assertTrue(E.TIMED_OUT)
        self.assertEqual(os.listdir(self.dir), [])
        self.assertTrue(E.path("summary.json").endswith("e89test_k15_summary.json"))
        self.assertTrue(E.fpath("frame.parquet").endswith(f"{E.TAG}_frame.parquet"))


if __name__ == "__main__":
    unittest.main()
