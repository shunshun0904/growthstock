"""Point-in-time inputs and matched samples for intraday feature research."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("research", "research/exp", "tests/fixtures")]
from build_volume_profile_features import feature_columns, file_sha256, main
from volume_profile_ablation import attach
from volume_profile_inputs import write_inputs


class VolumeProfileResearchTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        write_inputs(self.root)
        pd.DataFrame({"Date": ["2025-11-17", "2025-11-18"] * 2, "Code": ["123A", "123A", "9999", "9999"]}).to_csv(
            self.root / "targets.csv", index=False)
        self.options = ["--demo"]
        for flag, file in [
            ("--calendar", "calendar.csv"), ("--daily", "daily.csv"), ("--minute", "minute.csv.gz"),
            ("--ticks", "tick.csv.gz"), ("--targets", "targets.csv"), ("--cache-dir", "cache"), ("--output", "out"),
        ]:
            self.options.extend([flag, str(self.root / file)])

    def build(self, *extra):
        with redirect_stdout(StringIO()):
            self.assertEqual(main([*self.options, *extra]), 0)
        parquet = self.root / "out/volume-features.parquet"
        meta = json.loads(parquet.with_suffix(".meta.json").read_text())
        self.assertEqual(meta["sha256"], file_sha256(parquet))
        self.assertTrue(meta["demo"])
        return pd.read_parquet(parquet), meta

    def test_later_sessions_do_not_change_earlier_features_or_base_rows(self):
        earlier, _ = self.build("--historical", "--to-date", "2025-11-17")
        full, _ = self.build("--historical", "--to-date", "2025-11-18")
        pd.testing.assert_frame_equal(earlier, full.loc[full.Date.le("2025-11-17")].reset_index(drop=True))
        self.assertEqual(len(full.columns), 44)
        base = pd.DataFrame({"Code": ["9999", "123A", "7777"], "Date": ["2025-11-18"] * 3, "label": [0, 1, 0]})
        joined = attach(base, full)
        pd.testing.assert_frame_equal(joined[base.columns], base)
        tick, minute = feature_columns("tick"), feature_columns("minute")
        self.assertTrue(joined.loc[0, tick].isna().all())
        self.assertTrue(joined.loc[0, minute].notna().all())
        self.assertTrue(joined.loc[1, tick + minute].notna().all())
        self.assertTrue(joined.loc[2, tick + minute].isna().all())

    def test_missing_past_observation_and_assumed_release_before_cutoff(self):
        cols = feature_columns("tick") + feature_columns("minute")
        observed, meta = self.build("--to-date", "2025-11-18")
        self.assertEqual(meta["trainingAvailability"], "observed_at")
        self.assertTrue(observed[cols].isna().all().all())
        early, meta = self.build("--historical", "--decision-time", "16:59", "--to-date", "2025-11-18")
        self.assertEqual(meta["trainingAvailability"], "historical_schedule_estimate")
        self.assertTrue(early[cols].isna().all().all())
        at_release, _ = self.build("--historical", "--decision-time", "17:00", "--to-date", "2025-11-18")
        self.assertTrue(at_release.loc[at_release.Code.eq("123A0"), cols].notna().all().all())


if __name__ == "__main__":
    unittest.main()
