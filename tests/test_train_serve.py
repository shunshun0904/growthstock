#!/usr/bin/env python3
"""
学習と予測の食い違いの検出（research/check_train_serve.py）と、予測時の特徴量の
控え（predict_daily.save_live_features）の単体テスト。

固定すること
  - 控えは日付ごとに最初の予測で凍結する（予測の記録と同じ）
  - 後から作り直した値が違えば、その列を割合つきで報告する
  - 今日の行は比べない（同じ入力から作ったばかりなので必ず一致する）
  - 許容を超えた列があれば exit 1

  python3 tests/test_train_serve.py
"""
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import check_train_serve as CTS  # noqa: E402
import predict_daily as PD  # noqa: E402

T = pd.Timestamp


def cand(date, codes, credit):
    return pd.DataFrame({"Code": [f"{c}0" for c in codes], "Date": T(date),
                         "credit_ratio": credit, "vol_20d": [2.0] * len(codes)})


class LiveFeatures(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cols = ["credit_ratio", "vol_20d"]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def saved(self):
        return pd.read_parquet(os.path.join(self.dir, PD.LIVE_FEATURES))

    def test_saves_only_the_latest_date(self):
        c = pd.concat([cand("2026-09-17", ["1111"], [3.0]), cand("2026-09-18", ["2222"], [5.0])])
        PD.save_live_features(c, self.cols, [T("2026-09-17"), T("2026-09-18")], self.dir)
        s = self.saved()
        self.assertEqual(list(s["Code"]), ["22220"])
        self.assertEqual(float(s["credit_ratio"].iloc[0]), 5.0)

    def test_first_prediction_is_frozen(self):
        days = [T("2026-09-18")]
        PD.save_live_features(cand("2026-09-18", ["1111"], [3.0]), self.cols, days, self.dir)
        PD.save_live_features(cand("2026-09-18", ["1111"], [9.0]), self.cols, days, self.dir)
        self.assertEqual(float(self.saved()["credit_ratio"].iloc[0]), 3.0)

    def test_appends_new_dates(self):
        PD.save_live_features(cand("2026-09-17", ["1111"], [3.0]), self.cols,
                              [T("2026-09-17")], self.dir)
        PD.save_live_features(cand("2026-09-18", ["2222"], [5.0]), self.cols,
                              [T("2026-09-18")], self.dir)
        self.assertEqual(self.saved()["Date"].nunique(), 2)


class Compare(unittest.TestCase):
    def test_detects_a_value_that_changed_after_the_fact(self):
        """金曜の行の信用倍率が、翌週の公表後に作り直すと変わる（先読みの典型）。"""
        live = cand("2026-09-11", ["1111", "2222"], [2.0, 4.0])
        rebuilt = pd.concat([cand("2026-09-11", ["1111", "2222"], [9.0, 4.0]),
                             cand("2026-09-15", ["3333"], [1.0])])
        res = CTS.compare(live, rebuilt, ["credit_ratio", "vol_20d"], before=T("2026-09-15"))
        r = res.set_index("column")
        self.assertEqual(r.at["credit_ratio", "diff"], 1)
        self.assertAlmostEqual(r.at["credit_ratio", "rate"], 0.5)
        self.assertEqual(r.at["vol_20d", "diff"], 0)

    def test_nan_on_both_sides_is_a_match(self):
        live = cand("2026-09-11", ["1111"], [np.nan])
        rebuilt = cand("2026-09-11", ["1111"], [np.nan])
        res = CTS.compare(live, rebuilt, ["credit_ratio"])
        self.assertEqual(int(res["diff"].iloc[0]), 0)

    def test_value_that_appears_later_is_counted(self):
        # 予測時は欠測だったが、後から値が入った（遅れて届いたデータ）
        live = cand("2026-09-11", ["1111"], [np.nan])
        rebuilt = cand("2026-09-11", ["1111"], [2.0])
        r = CTS.compare(live, rebuilt, ["credit_ratio"]).iloc[0]
        self.assertEqual(int(r["diff"]), 1)
        self.assertEqual(int(r["live_nan_rebuilt_value"]), 1)

    def test_today_is_not_compared(self):
        live = cand("2026-09-15", ["1111"], [2.0])
        rebuilt = cand("2026-09-15", ["1111"], [9.0])
        res = CTS.compare(live, rebuilt, ["credit_ratio"], before=T("2026-09-15"))
        self.assertEqual(int(res["n"].iloc[0]), 0)


class Main(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_main(self, live, rebuilt, **kw):
        lp, rp = os.path.join(self.dir, "live.parquet"), os.path.join(self.dir, "re.parquet")
        live.to_parquet(lp)
        rebuilt.to_parquet(rp)
        args = ["--live", lp, "--rebuilt", rp, "--min-rows", "1"]
        for k, v in kw.items():
            args += [f"--{k.replace('_', '-')}", str(v)]
        return CTS.main(args)

    def test_fails_when_a_column_exceeds_the_limit(self):
        live = cand("2026-09-11", ["1111", "2222"], [2.0, 4.0])
        rebuilt = pd.concat([cand("2026-09-11", ["1111", "2222"], [9.0, 4.0]),
                             cand("2026-09-15", ["3333"], [1.0])])
        self.assertEqual(self.run_main(live, rebuilt, max_rate=0.1), 1)

    def test_passes_when_everything_matches(self):
        live = cand("2026-09-11", ["1111"], [2.0])
        rebuilt = pd.concat([cand("2026-09-11", ["1111"], [2.0]),
                             cand("2026-09-15", ["3333"], [1.0])])
        self.assertEqual(self.run_main(live, rebuilt), 0)

    def test_no_snapshot_yet_is_not_an_error(self):
        self.assertEqual(CTS.main(["--live", os.path.join(self.dir, "無い.parquet")]), 0)


class ReleaseUploads(unittest.TestCase):
    """
    控え（live_features.parquet）を Release に上げるのは日次予測だけ。

    ほかのワークフローも Release をまるごと落として research/_data/*.parquet を
    上げ直すので、除外を書き忘れると、予測が足した日を古い版で潰す。
    """

    def test_bulk_uploads_skip_live_features(self):
        wf = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(wf)):
            if not name.endswith((".yml", ".yaml")) or name == "predict.yml":
                continue
            with open(os.path.join(wf, name), encoding="utf-8") as f:
                text = f.read()
            if "files=(research/_data/*.parquet" in text:
                self.assertIn(f'= "{PD.LIVE_FEATURES}" ] && continue', text,
                              f"{name} が {PD.LIVE_FEATURES} を上げ直しうる")


if __name__ == "__main__":
    unittest.main(verbosity=2)
