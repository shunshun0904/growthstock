"""分足から作る価格帯別出来高の特徴量（research/minute_features.py。実験132）のテスト。

守りたいこと:
- 行の並び・件数を変えない（学習データに横に付けるため）
- 当日 D までの分足だけを使う（D より後の日を足しても、D の行の値は変わらない）
- 集計できない行（分足の前の日・置き場に無い銘柄・期間がそろわない）は欠測のまま（値を埋めない）
- 値はタブの集計（window_profiles）と同じ
"""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "research", ROOT / "scripts", ROOT / "tests/fixtures"):
    sys.path.insert(0, str(extra))

import minute_features as MF  # noqa: E402
import volume_profile as VP  # noqa: E402
import volume_profile_data as VD  # noqa: E402
from volume_profile_inputs import write_inputs  # noqa: E402


class MinuteFeaturesTest(unittest.TestCase):
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
        return MF.build(keys, str(minute_dir or self.root / "minute"), str(self.root / "raw"))

    def test_columns_are_eleven_metrics_by_three_windows(self):
        self.assertEqual(len(MF.COLUMNS), 33)
        self.assertEqual(len(set(MF.COLUMNS)), 33)
        self.assertIn("vp_minute_poc_gap_20d", MF.COLUMNS)
        self.assertIn("vp_minute_close_share_1d", MF.COLUMNS)

    def test_rows_keep_order_and_missing_stays_missing(self):
        rows = [("99990", "2025-11-18"), ("123A", "2025-11-18"), ("77770", "2025-11-18"),
                ("123A0", "2025-09-30"), ("123A0", "2025-10-02")]
        out, report = self.build(rows)
        self.assertEqual(list(out["Code"]), [r[0] for r in rows])
        self.assertEqual(list(out["Date"].dt.strftime("%Y-%m-%d")), [r[1] for r in rows])
        self.assertTrue(out.loc[[0, 1], MF.COLUMNS].notna().all().all(), "分足のある2銘柄は33列そろう")
        self.assertTrue(out.loc[2, MF.COLUMNS].isna().all(), "置き場に無い銘柄は欠測")
        self.assertTrue(out.loc[3, MF.COLUMNS].isna().all(), "分足より前の日は欠測")
        # 分足の2日目: 1営業日はそろい、5・20営業日はそろわない
        self.assertTrue(out.loc[4, [MF.column(m, 1) for m in MF.METRICS]].notna().all())
        self.assertTrue(out.loc[4, [MF.column(m, n) for m in MF.METRICS for n in (5, 20)]].isna().all())
        self.assertEqual(report["target_rows"], 4)
        self.assertEqual(report["attached"], 3)

    def test_values_match_the_tab_aggregation(self):
        out, _ = self.build([("123A0", "2025-11-18")])
        quotes = VD.load_quotes([str(self.root / "raw/bars_2025.parquet")], ["123A0"],
                                pd.date_range("2025-10-01", "2025-11-18").strftime("%Y-%m-%d"))["123A0"]
        days = MF.trading_days([str(self.root / "raw/bars_2025.parquet")], "2025-10-01")
        sessions = VD.load_minute_month(str(self.root / "minute/minute_202511.parquet"),
                                        [d for d in days if d >= "2025-11-01"], ["123A0"])["123A0"]
        sessions.update(VD.load_minute_month(str(self.root / "minute/minute_202510.parquet"),
                                             [d for d in days if d < "2025-11-01"], ["123A0"])["123A0"])
        close = quotes["2025-11-18"].close
        for n in VP.WINDOWS:
            want = VP.window_profiles(VP.lookback(days, "2025-11-18", n), n, quotes,
                                      {"tick": {}, "minute": sessions})["minute"]
            self.assertEqual(want["status"], "ok")
            got = out.loc[0, [MF.column(m, n) for m in MF.METRICS]].to_numpy(float)
            np.testing.assert_allclose(got, [MF.values(want, close)[m] for m in MF.METRICS])
            self.assertAlmostEqual(out.loc[0, MF.column("vwap_gap", n)], want["vwap"] / close - 1)
            shares = out.loc[0, [MF.column(f"{b}_share", n) for b in VP.BUCKETS]].sum()
            self.assertAlmostEqual(shares, 1.0)

    def test_later_days_do_not_change_earlier_rows(self):
        """D より後の分足を消しても、D の行の値は同じ（当日 D までしか使っていない）。"""
        full, _ = self.build([("123A0", "2025-11-14"), ("99990", "2025-11-14")])
        cut = Path(self.temp.name) / "cut"
        shutil.copytree(self.root / "minute", cut)
        manifest = json.loads((cut / "minute_manifest.json").read_text())
        manifest["days"] = {d: v for d, v in manifest["days"].items() if d <= "2025-11-14"}
        (cut / "minute_manifest.json").write_text(json.dumps(manifest))
        frame = pd.read_parquet(cut / "minute_202511.parquet")
        frame[pd.to_datetime(frame["Date"]) <= "2025-11-14"].to_parquet(cut / "minute_202511.parquet", index=False)
        early, _ = self.build([("123A0", "2025-11-14"), ("99990", "2025-11-14")], cut)
        pd.testing.assert_frame_equal(full, early)

    def test_status_columns_explain_each_window(self):
        rows = [("123A0", "2025-11-18"), ("77770", "2025-11-18"), ("123A0", "2025-09-30"), ("123A0", "2025-10-02")]
        keys = pd.DataFrame(rows, columns=["Code", "Date"])
        out, _ = MF.build(keys, str(self.root / "minute"), str(self.root / "raw"), with_status=True)
        self.assertEqual(list(out[MF.STATUS_COLUMNS].iloc[0]), ["ok", "ok", "ok"])
        self.assertEqual(out.loc[1, "vp_minute_status_1d"], "daily_missing")       # 日足の無い銘柄
        self.assertEqual(list(out[MF.STATUS_COLUMNS].iloc[2]), [MF.BEFORE_MINUTE] * 3)
        self.assertEqual(list(out[MF.STATUS_COLUMNS].iloc[3]), ["ok", "calendar_history_missing", "calendar_history_missing"])
        plain, _ = MF.build(keys, str(self.root / "minute"), str(self.root / "raw"))
        self.assertFalse(set(MF.STATUS_COLUMNS) & set(plain.columns))              # 既定では付けない
        pd.testing.assert_frame_equal(plain, out.drop(columns=MF.STATUS_COLUMNS))

    def test_store_without_days_leaves_everything_missing(self):
        empty = Path(self.temp.name) / "empty"
        empty.mkdir(exist_ok=True)
        (empty / "minute_manifest.json").write_text(json.dumps({"schema": 1, "days": {}}))
        out, report = self.build([("123A0", "2025-11-18")], empty)
        self.assertTrue(out[MF.COLUMNS].isna().all().all())
        self.assertEqual(report["attached"], 0)


if __name__ == "__main__":
    unittest.main()
