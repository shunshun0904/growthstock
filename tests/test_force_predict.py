"""Explicit re-prediction must work on a non-trading day."""
import datetime as dt
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import trading_calendar as TC
import trading_day_gate as GATE


class FixedNow(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 3, 14, tzinfo=dt.timezone.utc).astimezone(tz)


class TestForcePredict(unittest.TestCase):
    def gate_output(self, force):
        days = [dt.date(2026, 11, 2), dt.date(2026, 11, 3)]
        cal = TC.Calendar([days[0]], days)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            predictions = Path(directory) / "predictions.json"
            predictions.write_text('{"asOf":"2026-11-02"}')
            with patch.dict(os.environ, {"FORCE_PREDICT": force,
                                         "GITHUB_OUTPUT": str(output)}), \
                    patch.object(GATE.dt, "datetime", FixedNow), \
                    patch.object(GATE.trading_calendar, "load", return_value=cal), \
                    patch.object(GATE, "last_bar_date", return_value=days[0]):
                self.assertEqual(GATE.main(["--data-dir", directory,
                                            "--predictions", str(predictions)]), 0)
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_force_repredicts_published_bars_on_holiday(self):
        self.assertEqual(self.gate_output("true")["run"], "true")

    def test_automatic_run_still_skips_published_bars_on_holiday(self):
        self.assertEqual(self.gate_output("false")["run"], "false")


if __name__ == "__main__":
    unittest.main()
