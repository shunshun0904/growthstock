"""Contracts that matter for add-on data and dashboard output."""

from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
import base64
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "tests/fixtures")]
from build_volume_profile import main, expected_window, fetch_intraday, prediction_codes
from publish_volume_profile import PrivateStore, checked_snapshot
from volume_profile_data import AddonClient, ProfileCache, bulk_plan
from volume_profile import Intraday, Quote, window_profiles
from volume_profile_inputs import write_inputs


class VolumeProfileTest(unittest.TestCase):
    at = datetime(2025, 11, 18, 10, tzinfo=timezone.utc)
    day = "2025-11-18"

    def test_prediction_targets_are_all_and_only_current_day(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.json"
            path.write_text(json.dumps({"asOf": self.day, "candidates": [
                {"date": self.day, "code": "123A", "name": "A", "agree90": 0},
                {"date": self.day, "code": "9999", "name": "B", "agree90": 3},
                {"date": "2025-11-17", "code": "8888", "name": "Earlier"},
            ]}))
            self.assertEqual(prediction_codes(path), (self.day, {"123A0": "A", "99990": "B"}))

    def test_zero_targets_produce_empty_snapshot_without_api(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "predictions.json"
            path.write_text(json.dumps({"asOf": self.day, "candidates": []}))
            with patch("build_volume_profile.AddonClient") as client, redirect_stdout(StringIO()):
                self.assertEqual(main(["--fetch", "--predictions", str(path), "--output", str(root / "out")]), 0)
            client.assert_not_called()
            profile = json.loads((root / "out/volume-profile.json").read_text())
            self.assertEqual((profile["asOf"], profile["stocks"]), (self.day, []))

    def test_publication_rechecks_date_universe_and_disallows_demo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps({"asOf": self.day, "candidates": [{"date": self.day, "code": "123A"}]}))
            snapshot = root / "volume-profile.json"
            profile = {"kind": "growthstock.volume-profile", "schemaVersion": 1, "asOf": self.day, "demo": False,
                       "stocks": [{"code": "123A0", "date": self.day}]}
            snapshot.write_text(json.dumps(profile))
            self.assertEqual(checked_snapshot(snapshot, predictions)[1:], (self.day, 1))
            for update in ({"asOf": "2025-11-17"}, {"stocks": []}, {"demo": True}):
                snapshot.write_text(json.dumps({**profile, **update}))
                with self.assertRaisesRegex(ValueError, "最新の予測と一致しません"):
                    checked_snapshot(snapshot, predictions)

    def test_destination_must_be_private_and_updates_use_expected_blob(self):
        store = PrivateStore("owner/private-volume", "test-token")
        metadata = {"private": True, "full_name": "owner/private-volume", "default_branch": "main"}
        with patch.object(store, "request", return_value={**metadata, "private": False}) as request:
            with self.assertRaisesRegex(ValueError, "非公開"):
                store.publish(b"{}", self.day)
        request.assert_called_once_with("")
        with patch.object(store, "request", side_effect=[metadata, {"sha": "expected"}, {}]) as request:
            store.publish(b"{}", self.day)
        target, payload = request.call_args.args
        self.assertEqual(target, "/contents/dashboard/volume-profile.json")
        self.assertEqual(payload["branch"], "main")
        self.assertEqual(payload["sha"], "expected")
        self.assertEqual(base64.b64decode(payload["content"]), b"{}")

    def test_tick_and_minute_distributions_keep_the_same_total(self):
        quote = Quote(100, 110, 106, 400, 43_000, 1, None, self.at)
        tick = Intraday(np.array([[100., 100, 10_000], [110., 300, 33_000]]), self.at)
        minute = Intraday(np.array([[107.5, 400, 43_000]]), self.at)
        profile = window_profiles([self.day], 1, {self.day: quote},
                                  {"tick": {self.day: tick}, "minute": {self.day: minute}}, self.at)
        self.assertEqual(profile["tick"]["poc"], [110., 110.5])
        self.assertEqual(profile["minute"]["poc"], [107.5, 108.])
        self.assertEqual(profile["tick"]["aboveCloseShare"], .75)
        self.assertEqual(profile["minute"]["aboveCloseShare"], 1.)
        for p in profile.values():
            self.assertEqual(p["shares"], 400)
            self.assertEqual(p["turnover"], 43_000)
            self.assertEqual(p["vwap"], 107.5)
            self.assertEqual(p["tickMinuteDistance"], 1.)

    def test_split_rescales_old_price_and_shares_only_with_a_confirmed_action(self):
        before = "2025-11-17"
        older = Quote(200, 200, 200, 100, 20_000, 1, None, self.at)
        tape = {before: Intraday(np.array([[200., 100, 20_000]]), self.at),
                self.day: Intraday(np.array([[100., 300, 30_000]]), self.at)}
        for action, status in [("1", "ok"), (None, "share_basis_unknown"), ("3", "share_basis_unknown")]:
            newer = Quote(100, 100, 100, 300, 30_000, .5, action, self.at)
            profile = window_profiles([before, self.day], 2, {before: older, self.day: newer},
                                      {"tick": tape, "minute": {}}, self.at)["tick"]
            self.assertEqual(profile["status"], status)
            if status == "ok":
                self.assertEqual(profile["shares"], 500)
                self.assertEqual(profile["turnover"], 50_000)
                self.assertEqual(profile["vwap"], 100.)
                self.assertTrue(profile["adjustedShares"])

    def test_policy_neither_jumps_missing_calendar_days_nor_uses_late_files(self):
        self.assertEqual(expected_window({"2025-11-16": 1, self.day: 1}, ["2025-11-16", self.day], self.day, 2), [])
        quote = Quote(100, 100, 100, 100, 10_000, 1, None, self.at - timedelta(minutes=5))
        x = Intraday(np.array([[100., 100, 10_000]]), self.at + timedelta(minutes=1))
        y = Intraday(np.array([[100., 50, 5_000]]), self.at)
        args = ([self.day], 1, {self.day: quote}, {"tick": {self.day: x}, "minute": {self.day: y}})
        observed = window_profiles(*args, self.at)
        self.assertEqual(observed["tick"]["status"], "not_available_at_cutoff")
        self.assertEqual(observed["minute"]["status"], "daily_totals_mismatch")
        self.assertEqual(window_profiles(*args, self.at + timedelta(minutes=1))["tick"]["status"], "ok")

    def test_bulk_plan_uses_the_daily_file_once_when_monthly_also_exists(self):
        prefix = "equities/trades/"
        a = prefix + "historical/2025/equities_trades_202511.csv.gz"
        b = prefix + "live/equities_trades_20251118.csv.gz"
        wanted = {("2025-11-17", "123A0"), (self.day, "123A0")}
        plan = bulk_plan([{"Key": a}, {"Key": b}], wanted, "/equities/trades")
        self.assertEqual(plan, {a: {("2025-11-17", "123A0")}, b: {(self.day, "123A0")}})

    def test_pagination_fails_on_repeated_cursor(self):
        class RepeatedPage(AddonClient):
            def __init__(self):
                pass

            def get(self, endpoint, params):
                return {"data": [], "pagination_key": "repeat"}

        with self.assertRaisesRegex(ValueError, "ページング"):
            RepeatedPage().rows("/equities/bars/minute", {"code": "123A0"})

    def test_bulk_fetch_streams_gzip_into_a_verified_session_without_api_headers(self):
        key = "equities/trades/live/equities_trades_20251118.csv.gz"
        signed = "https://example.test/trades.csv.gz?temporary-signature=test"
        payload = gzip.compress(b"Date,Code,Time,SessionDistinction,Price,TradingVolume,TransactionId\n"
                                b"2025-11-18,123A0,10:00:00,1,100,2,1\n")
        client = AddonClient.__new__(AddonClient)
        wanted = {(self.day, "123A0")}
        with tempfile.TemporaryDirectory() as temp, closing(ProfileCache(temp)) as cache:
            cache.add_quotes(pd.DataFrame([dict(Date=self.day, Code="123A0", L=100, H=100, C=100,
                                                 Vo=2, Va=200, AdjFactor=1)]), wanted)
            with patch.object(client, "get", side_effect=[{"data": [{"Key": key}]}, {"url": signed}]) as get, \
                 patch("volume_profile_data.urllib.request.urlopen", return_value=BytesIO(payload)) as open_url, \
                 redirect_stdout(StringIO()):
                fetch_intraday(cache, client, "tick", wanted, False)
                fetch_intraday(cache, client, "tick", wanted, False)
            self.assertEqual(get.call_count, 2)  # The cached session skips the second download.
            get.assert_called_with("/bulk/get", {"key": key})
            open_url.assert_called_once_with(signed, timeout=90)
            _, data = cache.load_code("123A0", self.day, self.day)
            np.testing.assert_array_equal(data["tick"][self.day].points, [[100, 2, 200]])

    def test_incomplete_import_cannot_overwrite_a_verified_window(self):
        wanted = {(self.day, "123A0")}
        with tempfile.TemporaryDirectory() as temp, closing(ProfileCache(temp)) as cache:
            cache.add_quotes(pd.DataFrame([dict(Date=self.day, Code="123A0", L=100, H=100, C=100,
                                                 Vo=2, Va=200, AdjFactor=1)]), wanted)
            trade = pd.DataFrame([dict(Date=self.day, Code="123A0", Price=100, TradingVolume=1, TransactionId="01")])
            other = trade.assign(TransactionId="02")
            self.assertEqual(cache.add_intraday([trade, other], "tick", wanted)["verified"], 1)
            self.assertEqual(cache.add_intraday([trade], "tick", wanted)["notReconciled"], 1)
            _, data = cache.load_code("123A0", self.day, self.day)
            np.testing.assert_array_equal(data["tick"][self.day].points, [[100, 2, 200]])

    def test_vendor_shaped_files_create_only_the_dashboard_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_inputs(root)
            output = root / "out"
            args = ["--calendar", str(root / "calendar.csv"), "--daily", str(root / "daily.csv"),
                    "--minute", str(root / "minute.csv.gz"), "--ticks", str(root / "tick.csv.gz"),
                    "--as-of", self.day, "--codes", "123A", "9999", "--cache-dir", str(root / "cache"),
                    "--output", str(output), "--demo"]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)  # Re-import is replacement, not accumulation.
            profile = json.loads((output / "volume-profile.json").read_text())
            self.assertTrue(profile["demo"])
            first, second = profile["stocks"]
            self.assertEqual(first["windows"]["20"]["tick"]["shares"], 20 * (190+390+680+2050+1770+770+420+270))
            self.assertEqual(first["windows"]["20"]["tick"]["status"], "ok")
            self.assertEqual(second["windows"]["20"]["tick"]["status"], "intraday_missing")
            self.assertEqual(second["windows"]["20"]["minute"]["status"], "ok")
            self.assertEqual([item.name for item in output.iterdir()], ["volume-profile.json"])


if __name__ == "__main__":
    unittest.main()
