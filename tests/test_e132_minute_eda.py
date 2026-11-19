#!/usr/bin/env python3
"""
実験132の前段の EDA（research/exp/e132_minute_eda.py）。データには触らない（置き場と同じ形の合成データで確かめる）。

- 一通り: 学習データの行（作りもの）に33列を付け、欠測・分布・異常値・整合・時期・相関・極端な値の各節が最後まで通る
- 日足との突き合わせ: VWAP は日足の売買代金 ÷ 出来高と一致する（D に揃っている）。1日前の日足とは一致しない
- 欠測の理由: 分足の前・日足の無い銘柄・期間の営業日が足りない、が数えられる
- 公開ログに銘柄コードを出さない

  python3 -m unittest tests.test_e132_minute_eda
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for extra in ("research", os.path.join("research", "exp"), os.path.join("tests", "fixtures")):
    sys.path.insert(0, os.path.join(ROOT, extra))

import e132_minute_eda as E  # noqa: E402
import features as F  # noqa: E402
import minute_features as MF  # noqa: E402
from volume_profile_inputs import write_inputs  # noqa: E402


def learning_frame():
    """学習データの作りもの: 分足のある2銘柄の毎営業日と、日足の無い銘柄・分足より前の日。"""
    days = pd.bdate_range("2025-09-29", "2025-11-18")
    rows = [(c, d) for d in days for c in ("123A0", "99990")] + [("77770", pd.Timestamp("2025-11-18"))]
    df = pd.DataFrame(rows, columns=["Code", "Date"])
    rng = np.random.default_rng(5)
    df["label"] = (rng.random(len(df)) < 0.3).astype(float)
    for c in F.columns(E.BASE_PRESET)[:20]:
        df[c] = rng.normal(size=len(df))
    df.index = df.index + 1000                                          # 連番でない index でも並びを保つ
    return df


class EdaEndToEndTest(unittest.TestCase):
    def test_all_sections_and_vwap_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_inputs(root)
            out = root / "summary.json"
            buf = io.StringIO()
            with mock.patch.object(E.lab, "frame", return_value=learning_frame()), \
                 mock.patch.object(E.lab, "DATA_DIR", str(root / "raw")), \
                 mock.patch.object(E, "MINUTE_DIR", str(root / "minute")), \
                 mock.patch.object(E, "OUT", str(out)), redirect_stdout(buf):
                self.assertEqual(E.main([]), 0)
            log = buf.getvalue()
            summary = json.loads(out.read_text())
        for title in ("1. 行の数", "2. 欠測", "3. 分布", "4. 異常値", "5. 値の整合", "6. 時期による変化", "7. 相関",
                      "8. 極端な値"):
            self.assertIn(title, log)
        # 祝日（10/13・11/3）は日足が無いので学習データの日にしない（作りものの行は平日すべてなので、その2日は日足の欠測になる）
        status_1d = summary["missing"]["status"]["1d"]
        self.assertIn("daily_missing", status_1d)                       # 77770 と祝日
        self.assertGreater(status_1d["ok"], 0)
        self.assertIn("calendar_history_missing", summary["missing"]["status"]["20d"])   # 分足の始まりの直後
        for w in ("1d", "5d", "20d"):
            v = summary["vwap_check"][w]
            self.assertGreater(v["checked"], 0)
            self.assertLess(v["max_rel_diff"], 1e-9)                    # D の日足と一致
            self.assertEqual(summary["missing"]["torn"][w], 0)
            self.assertEqual(summary["consistency"][w]["poc_outside_value_area"], 0)
            self.assertLess(summary["consistency"][w]["bucket_sum_max_dev"], 1e-9)
        self.assertGreater(summary["vwap_check"]["1d"]["vs_previous_day_median_rel_diff"], 1e-4)   # 前の日とは合わない
        self.assertEqual(summary["rows"]["period"], summary["report"]["target_rows"])
        self.assertEqual(len(summary["distribution"]), 33)
        self.assertNotIn("123A0", log)
        self.assertNotIn("99990", log)


class PartsTest(unittest.TestCase):
    def test_outliers_count_ranges_and_iqr(self):
        df = pd.DataFrame({c: np.linspace(0.1, 0.2, 100) for c in MF.COLUMNS})
        df.loc[0, MF.column("poc_share", 1)] = 1.5                      # 割合が 1 を超える
        df.loc[1, MF.column("vwap_gap", 1)] = -1.0                      # 終値との差が −100%
        df.loc[2, MF.column("vwap_gap", 5)] = 0.9                       # 四分位範囲の外・±30% 超
        df.loc[3, MF.column("close_share", 20)] = np.inf
        out = E.outliers(df)
        self.assertEqual(out[MF.column("poc_share", 1)]["out_of_range"], 1)
        self.assertEqual(out[MF.column("vwap_gap", 1)]["out_of_range"], 1)
        self.assertEqual(out[MF.column("vwap_gap", 5)]["iqr_high"], 1)
        self.assertEqual(out[MF.column("vwap_gap", 5)]["big"], 1)
        self.assertEqual(out[MF.column("close_share", 20)]["inf"], 1)


if __name__ == "__main__":
    unittest.main()
