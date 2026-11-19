"""価格帯別出来高（research/build_volume_profile.py ほか）のテスト。docs/VOLUME_PROFILE.md。

守りたいこと:
- 対象は直近5つの予測日の候補すべて（銘柄の重複は除く）
- 寄り・前場・後場・引けの分け方と、ザラバだけで数える約定代金の大きさ別の割合
- 日足と照合できない日・置き場に無い日があれば、その期間・種別は欠損（値を作らない）
- 書き出すのは暗号文だけで、J-Quants のキーが無ければ開けない。見出しを書き換えても開けない
- ログ用のまとめに出すのは件数・日付・ファイル名だけ
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date, time
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "research", ROOT / "scripts", ROOT / "tests/fixtures"):
    sys.path.insert(0, str(extra))

import build_volume_profile as B  # noqa: E402
import volume_profile as V  # noqa: E402
import volume_profile_crypto as C  # noqa: E402
import volume_profile_data as D  # noqa: E402
from volume_profile_inputs import write_inputs, write_predictions  # noqa: E402

KEY = "test-jquants-key-not-real-0123456789"
DAY = "2025-11-18"


def us(hh, mm, ss=0, micro=0):
    return ((hh * 60 + mm) * 60 + ss) * 1_000_000 + micro


def tick(rows, day=date(2025, 11, 18)):
    t, s, p, q = zip(*rows)
    return V.tick_session(day, np.array(t), np.array(s), np.array(p, float), np.array(q, float))


class TargetsTest(unittest.TestCase):
    def test_all_candidates_of_the_last_five_prediction_dates(self):
        predictions = {"asOf": DAY, "dates": ["2025-11-12", "2025-11-13", "2025-11-14", "2025-11-17", DAY],
                       "candidates": [
                           {"date": "2025-11-11", "code": "7777", "name": "古い日", "rankInDay": 1},
                           {"date": "2025-11-14", "code": "123A", "name": "A", "rankInDay": 2},
                           {"date": DAY, "code": "123A", "jqCode": "123A0", "name": "A", "rankInDay": 1},
                           {"date": "2025-11-12", "code": "9999", "name": "B"},
                       ]}
        as_of, dates, stocks = B.prediction_targets(predictions)
        self.assertEqual(as_of, DAY)
        self.assertEqual(dates[0], "2025-11-12")
        self.assertEqual(sorted(stocks), ["123A0", "99990"])
        self.assertEqual(stocks["123A0"]["candidates"], [{"date": DAY, "rank": 1}, {"date": "2025-11-14", "rank": 2}])
        self.assertEqual(stocks["99990"]["candidates"], [{"date": "2025-11-12", "rank": None}])

    def test_dates_fall_back_to_candidate_dates(self):
        predictions = {"asOf": DAY, "candidates": [{"date": d, "code": "1301"} for d in
                                                  ("2025-11-10", "2025-11-11", "2025-11-12", "2025-11-13",
                                                   "2025-11-14", DAY)]}
        _, dates, stocks = B.prediction_targets(predictions)
        self.assertEqual(dates, ["2025-11-11", "2025-11-12", "2025-11-13", "2025-11-14", DAY])
        self.assertEqual(len(stocks["13010"]["candidates"]), 5)


class SessionTest(unittest.TestCase):
    def test_tick_session_separates_auctions_buckets_and_sizes(self):
        session = tick([
            (us(9, 0), "01", 100, 300), (us(9, 0), "01", 100, 200),     # 寄り（同じ時刻・価格の2件）
            (us(9, 0, 0, 100), "01", 101, 100),                          # ザラバ・小口
            (us(10, 0), "01", 102, 20_000),                              # ザラバ・中口（204万円）
            (us(11, 30), "01", 103, 400),                                # 前引け
            (us(12, 30), "02", 104, 500),                                # 後場寄り
            (us(13, 0), "02", 105, 100_000),                             # ザラバ・大口（1,050万円）
            (us(15, 30), "02", 106, 700),                                # 引け
        ])
        self.assertIsNone(session.problem)
        np.testing.assert_array_equal(session.buckets, [500, 20_500, 100_500, 700])
        np.testing.assert_array_equal(session.sizes, [10_100, 2_040_000, 10_500_000])
        np.testing.assert_array_equal(session.counts, [1, 1, 1])
        np.testing.assert_array_equal(session.points[:, 0], [100, 101, 102, 103, 104, 105, 106])
        self.assertEqual(session.points[:, 1].sum(), 122_200)

    def test_tick_open_in_the_afternoon_and_open_at_the_close(self):
        session = tick([(us(12, 30), "02", 50, 100), (us(13, 0), "02", 51, 100), (us(15, 30), "02", 52, 300)])
        np.testing.assert_array_equal(session.buckets, [100, 0, 100, 300])
        # 引けの板寄せで初めて寄った日は、その約定を寄りとして数える（引けと二重に数えない）
        only_close = tick([(us(15, 30), "02", 52, 300)])
        np.testing.assert_array_equal(only_close.buckets, [300, 0, 0, 0])
        np.testing.assert_array_equal(only_close.counts, [0, 0, 0])

    def test_closing_auction_time_before_2024_11_05(self):
        old = tick([(us(9, 0), "01", 10, 100), (us(14, 59), "02", 10, 100), (us(15, 0), "02", 10, 200)],
                   day=date(2024, 11, 1))
        np.testing.assert_array_equal(old.buckets, [100, 0, 100, 200])
        new = tick([(us(9, 0), "01", 10, 100), (us(15, 0), "02", 10, 100), (us(15, 30), "02", 10, 200)])
        np.testing.assert_array_equal(new.buckets, [100, 0, 100, 200])

    def test_unknown_session_code_falls_back_to_the_clock(self):
        session = tick([(us(9, 0), "", 10, 100), (us(10, 0), "", 10, 100), (us(13, 0), "", 10, 100)])
        np.testing.assert_array_equal(session.buckets, [100, 100, 100, 0])

    def test_invalid_ticks_are_flagged(self):
        for rows in ([(us(9, 0), "01", 0, 100)], [(us(9, 0), "01", 10, 0.5)], [(us(9, 0), "01", float("nan"), 100)]):
            self.assertEqual(tick(rows).problem, "intraday_invalid")

    def test_minute_session_buckets(self):
        session = V.minute_session(
            date(2025, 11, 18), np.array(["09:00", "09:01", "11:30", "12:30", "15:24", "15:30", "10:00"]),
            np.array([500, 100, 400, 500, 100, 700, 0.]), np.array([50_000, 10_100, 41_200, 52_000, 10_500, 74_200, 0.]),
            np.array([100, 101, 103, 104, 105, 106, 0.]), np.array([100, 101, 103, 104, 105, 106, 0.]))
        self.assertIsNone(session.problem)
        np.testing.assert_array_equal(session.buckets, [500, 500, 600, 700])
        self.assertIsNone(session.sizes)
        bad = V.minute_session(date(2025, 11, 18), np.array(["09:00"]), np.array([100.]), np.array([20_000.]),
                               np.array([100.]), np.array([150.]))
        self.assertEqual(bad.problem, "intraday_invalid")


class WindowTest(unittest.TestCase):
    def quote(self, low, high, close, volume, turnover, factor=1.0):
        return V.Quote(low, high, close, volume, turnover, factor)

    def test_tick_and_minute_keep_the_same_totals_and_new_metrics(self):
        quote = self.quote(100, 110, 106, 400, 43_000)
        # 9:00 は寄り、13:00 は後場寄り（どちらも板寄せ）、13:00:30 はザラバ
        ticks = tick([(us(9, 0), "01", 100, 100), (us(13, 0), "02", 110, 100), (us(13, 0, 30), "02", 110, 200)])
        minute = V.minute_session(date(2025, 11, 18), np.array(["09:00", "13:00"]), np.array([100., 300.]),
                                  np.array([10_000., 33_000.]), np.array([100., 110.]), np.array([100., 110.]))
        profile = V.window_profiles([DAY], 1, {DAY: quote}, {"tick": {DAY: ticks}, "minute": {DAY: minute}})
        for result in profile.values():
            self.assertEqual(result["status"], "ok")
            self.assertEqual((result["shares"], result["turnover"], result["vwap"]), (400, 43_000, 107.5))
            self.assertAlmostEqual(result["closeVsVwap"], 106 / 107.5 - 1)
            self.assertEqual(result["sessions"], {"open": .25, "morning": 0., "afternoon": .75, "close": 0.})
        self.assertEqual(profile["tick"]["tradeSizes"], {"turnover": 22_000., "shares": [1., 0., 0.], "counts": [1, 0, 0]})
        self.assertNotIn("tradeSizes", profile["minute"])
        self.assertEqual(profile["tick"]["tickMinuteDistance"], 0.)

    def test_split_rescales_older_sessions_to_the_latest_share_basis(self):
        before = "2025-11-17"
        older = self.quote(200, 200, 200, 100, 20_000)
        tape = {before: tick([(us(10, 0), "01", 200, 100)]), DAY: tick([(us(10, 0), "01", 100, 300)])}
        for factor, status in [(.5, "ok"), (None, "share_basis_unknown"), (0., "share_basis_unknown")]:
            newer = self.quote(100, 100, 100, 300, 30_000, factor)
            result = V.window_profiles([before, DAY], 2, {before: older, DAY: newer},
                                       {"tick": tape, "minute": {}})["tick"]
            self.assertEqual(result["status"], status)
            if status == "ok":
                self.assertEqual((result["shares"], result["turnover"], result["vwap"]), (500, 50_000, 100.))
                self.assertTrue(result["adjustedShares"])
                self.assertEqual(result["sessions"]["open"], 1.)

    def test_missing_mismatched_invalid_and_no_trade_sessions(self):
        quote = self.quote(100, 100, 100, 200, 20_000)
        good = tick([(us(10, 0), "01", 100, 200)])
        short = tick([(us(10, 0), "01", 100, 100)])
        invalid = tick([(us(10, 0), "01", 0, 100)])
        quotes = {DAY: quote}
        check = lambda session: V.window_profiles([DAY], 1, quotes, {"tick": {DAY: session} if session else {},
                                                                    "minute": {}})["tick"]["status"]
        self.assertEqual(check(good), "ok")
        self.assertEqual(check(None), "intraday_missing")
        self.assertEqual(check(short), "daily_totals_mismatch")
        self.assertEqual(check(invalid), "intraday_invalid")
        quiet = {"2025-11-17": V.Quote(None, None, None, 0, 0, 1.), DAY: quote}
        result = V.window_profiles(["2025-11-17", DAY], 2, quiet, {"tick": {DAY: good}, "minute": {}})["tick"]
        self.assertEqual((result["status"], result["coveredSessions"]), ("ok", 2))
        self.assertEqual(V.window_profiles([DAY], 1, {}, {"tick": {}, "minute": {}})["tick"]["status"], "daily_missing")
        self.assertEqual(V.window_profiles([DAY], 2, quotes, {})["tick"]["status"], "calendar_history_missing")

    def test_reconcile_names_the_reason(self):
        quote = self.quote(100, 110, 105, 300, 31_000)
        ok = np.array([[100., 100, 10_000], [105., 200, 21_000]])
        self.assertEqual(V.reconcile(ok, quote), "ok")
        self.assertEqual(V.reconcile(ok[:1], quote), "volume")
        self.assertEqual(V.reconcile(ok + [0, 0, 5], quote), "turnover")
        self.assertEqual(V.reconcile(np.array([[99., 100, 10_000], [105., 200, 21_000]]),
                                     self.quote(99.5, 110, 105, 300, 31_000)), "range")
        self.assertEqual(V.reconcile(np.empty((0, 3)), V.Quote(None, None, None, 0, 0, 1.)), "ok")
        self.assertEqual(V.reconcile(ok, V.Quote(None, None, None, 0, 0, 1.)), "traded")


class CryptoTest(unittest.TestCase):
    def test_round_trip_binds_the_header_and_needs_the_key(self):
        plain = json.dumps({"kind": "growthstock.volume-profile", "stocks": []}).encode()
        doc = C.encrypt(plain, f"  {KEY}\n", DAY, "2025-11-18T09:00:00+00:00", iterations=100_000)
        self.assertEqual(C.decrypt(doc, KEY), plain)
        with self.assertRaisesRegex(ValueError, "鍵が違う"):
            C.decrypt(doc, KEY + "x")
        with self.assertRaisesRegex(ValueError, "鍵が違う"):
            C.decrypt({**doc, "asOf": "2025-11-19"}, KEY)
        self.assertNotIn(b"growthstock.volume-profile\"", json.dumps(doc).encode())
        self.assertNotEqual(C.encrypt(plain, KEY, DAY, "x", iterations=100_000)["cipher"]["iv"], doc["cipher"]["iv"])

    def test_check_rejects_anything_but_the_envelope(self):
        doc = C.encrypt(b"{}", KEY, DAY, "now", iterations=100_000)
        for bad in ({"kind": "growthstock.volume-profile", "stocks": []}, {**doc, "stocks": []},
                    {**doc, "kind": "x"}, {**doc, "kdf": {**doc["kdf"], "iterations": 10}},
                    {**doc, "cipher": {**doc["cipher"], "iv": "AAAA"}}, {**doc, "data": "not base64!"}):
            with self.assertRaises(ValueError):
                C.check_envelope(bad)
        with tempfile.TemporaryDirectory() as temp:
            good, bad = Path(temp) / "good.json", Path(temp) / "bad.json"
            good.write_text(json.dumps(doc))
            bad.write_text(json.dumps({"kind": "growthstock.volume-profile", "stocks": []}))
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(C.main(["check", str(good)]), 0)
                self.assertEqual(C.main(["check", str(bad)]), 1)


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_inputs(self.root)
        write_predictions(self.root / "predictions.json")
        self.env = {k: os.environ.get(k) for k in ("JQUANTS_API", "JQUANTS_API_KEY")}
        os.environ["JQUANTS_API"] = KEY
        os.environ.pop("JQUANTS_API_KEY", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.temp.cleanup()

    def run_build(self, *extra):
        args = ["--predictions", str(self.root / "predictions.json"), "--work-dir", str(self.root),
                "--calendar", str(self.root / "calendar.csv"), *extra]
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = B.main(args)
        return code, out.getvalue(), err.getvalue()

    def test_store_files_become_an_encrypted_profile_and_a_counts_only_summary(self):
        enc, summary = self.root / "out/enc.json", self.root / "out/summary.json"
        code, out, err = self.run_build("--encrypt-to", str(enc), "--summary-json", str(summary))
        self.assertEqual(code, 0, err)
        doc = json.loads(enc.read_text())
        C.check_envelope(doc)
        data = json.loads(C.decrypt(doc, KEY))
        self.assertEqual((data["kind"], data["schemaVersion"], data["asOf"], data["demo"]),
                         ("growthstock.volume-profile", 2, DAY, False))
        stocks = {s["code"]: s for s in data["stocks"]}
        self.assertEqual(sorted(stocks), ["123A0", "88990", "99990"])
        a = stocks["123A0"]["windows"]
        for n in ("1", "5", "20"):
            self.assertEqual((a[n]["tick"]["status"], a[n]["minute"]["status"]), ("ok", "ok"))
            self.assertAlmostEqual(sum(a[n]["tick"]["sessions"].values()), 1)
            self.assertEqual(a[n]["tick"]["shares"], a[n]["minute"]["shares"])
        self.assertEqual(a["20"]["tick"]["from"], "2025-10-21")
        self.assertEqual(stocks["99990"]["windows"]["5"]["tick"]["status"], "intraday_missing")
        self.assertEqual(stocks["99990"]["windows"]["5"]["minute"]["status"], "ok")
        self.assertEqual(stocks["88990"]["windows"]["1"]["tick"]["status"], "daily_missing")
        report = json.loads(summary.read_text())
        self.assertEqual(report["reconcile"]["tick"], {"missing": 20, "no_daily": 20, "ok": 20})
        self.assertEqual(report["status"]["20"]["minute"], {"daily_missing": 1, "ok": 2})
        # まとめとログには値（価格・数量・銘柄名）を出さない
        close = re.compile(rf"\b{int(stocks['123A0']['close'])}(\.0)?\b")
        for text in (summary.read_text(), out):
            self.assertNotIn("サンプル", text)
            self.assertNotIn("bands", text)
            self.assertIsNone(close.search(text))
            self.assertNotIn(KEY, text)

    def test_plaintext_output_stays_out_of_public_and_dist(self):
        code, _, err = self.run_build("--output", str(ROOT / "public/data/volume-profile.json"))
        self.assertEqual(code, 1)
        self.assertIn("public/・dist/ の外", err)
        self.assertFalse((ROOT / "public/data/volume-profile.json").exists())

    def test_missing_store_day_is_reported_not_invented(self):
        (self.root / "tick" / "tick_20251118.parquet").unlink()
        enc, summary = self.root / "enc.json", self.root / "summary.json"
        code, _, err = self.run_build("--encrypt-to", str(enc), "--summary-json", str(summary))
        self.assertEqual(code, 0, err)
        stocks = {s["code"]: s for s in json.loads(C.decrypt(json.loads(enc.read_text()), KEY))["stocks"]}
        self.assertEqual(stocks["123A0"]["windows"]["1"]["tick"]["status"], "intraday_missing")
        self.assertEqual(stocks["123A0"]["windows"]["1"]["minute"]["status"], "ok")
        report = json.loads(summary.read_text())
        self.assertEqual(report["files"]["tick"], {"planned": 20, "present": 19})
        self.assertTrue(any("tick_20251118.parquet" in w for w in report["warnings"]))

    def test_download_command_receives_tag_dir_and_name(self):
        remote = self.root / "remote"
        os.rename(self.root / "tick", remote)
        log = self.root / "calls.txt"
        script = self.root / "fake_download.py"
        script.write_text(
            "import shutil, sys, pathlib\n"
            f"remote = pathlib.Path({str(remote)!r}); log = pathlib.Path({str(log)!r})\n"
            "tag, folder, name = sys.argv[1:]\n"
            "log.open('a').write(f'{tag} {name}\\n')\n"
            "src = remote / name\n"
            "if src.exists():\n    pathlib.Path(folder).mkdir(parents=True, exist_ok=True); shutil.copy(src, folder)\n")
        enc = self.root / "enc.json"
        code, _, err = self.run_build("--encrypt-to", str(enc), "--download-cmd", f"{sys.executable} {script}",
                                      "--remove-after-read")
        self.assertEqual(code, 0, err)
        calls = log.read_text().splitlines()
        self.assertEqual(calls[0], "data-tick tick_manifest.json")
        self.assertIn("data-tick-2025 tick_20251118.parquet", calls)
        self.assertEqual(len([c for c in calls if c.startswith("data-tick-2025 ")]), 20)
        self.assertFalse(list((self.root / "tick").glob("tick_2025*.parquet")), "読み終えたティックは消す")

    def test_conflicting_daily_rows_drop_only_that_day(self):
        import pandas as pd

        bars = pd.read_parquet(self.root / "raw/bars_2025.parquet")
        clash = bars[(bars.Code == "123A0") & (bars.Date == "2025-11-17")].assign(Vo=1.0)
        pd.concat([bars, clash, bars.iloc[:1]], ignore_index=True).to_parquet(self.root / "raw/bars_2025.parquet")
        enc, summary = self.root / "enc.json", self.root / "summary.json"
        code, _, err = self.run_build("--encrypt-to", str(enc), "--summary-json", str(summary))
        self.assertEqual(code, 0, err)
        windows = {s["code"]: s for s in json.loads(C.decrypt(json.loads(enc.read_text()), KEY))["stocks"]}["123A0"]["windows"]
        self.assertEqual(windows["1"]["tick"]["status"], "ok")
        self.assertEqual(windows["5"]["tick"]["status"], "daily_missing")
        self.assertTrue(any("違う行が 1件" in w for w in json.loads(summary.read_text())["warnings"]))

    def test_plan_uses_manifest_releases_months_and_years(self):
        tick_manifest = {"days": {"2025-12-30": {"release": "data-tick-2025"}, "2026-01-05": {}}}
        minute_manifest = {"days": {"2025-12-30": {}, "2026-01-05": {}}}
        plan = D.plan_assets(["2025-12-30", "2026-01-05", "2026-01-06"], tick_manifest, minute_manifest)
        self.assertEqual(plan["tick"], [("data-tick-2025", "tick_20251230.parquet"),
                                        ("data-tick-2026", "tick_20260105.parquet")])
        self.assertEqual(plan["minute"], [("data-minute", "minute_202512.parquet"), ("data-minute", "minute_202601.parquet")])
        self.assertEqual(plan["bars"], [("data-raw", "bars_2025.parquet"), ("data-raw", "bars_2026.parquet")])

    def test_no_targets_still_writes_an_empty_encrypted_profile(self):
        (self.root / "predictions.json").write_text(json.dumps({"asOf": DAY, "dates": [DAY], "candidates": []}))
        enc = self.root / "enc.json"
        code, _, err = self.run_build("--encrypt-to", str(enc))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(C.decrypt(json.loads(enc.read_text()), KEY))["stocks"], [])


class WorkflowTest(unittest.TestCase):
    def read(self, name):
        return (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")

    def test_update_workflow_uses_only_the_jquants_secret_and_checks_before_upload(self):
        text = self.read("update-volume-profile.yml")
        self.assertEqual(set(re.findall(r"secrets\.([A-Z_]+)", text)), {"JQUANTS_API"})
        self.assertIn("contents: write", text)
        self.assertNotIn("MINUTE_DATA_TOKEN", text)
        self.assertNotIn("PROFILE_REPO", text)
        for workflow in ("Predict Breakouts", "Update Minute Bars", "Update Ticks"):
            self.assertIn(f"'{workflow}'", text)
        check = text.index("volume_profile_crypto.py check")
        self.assertLess(check, text.index("scripts/gh_release_upload.sh"))
        self.assertNotIn("GH_ENSURE_DRAFT", text)  # デプロイの読み取り専用の鍵で読める公開の Release

    def test_deploy_places_only_a_checked_envelope(self):
        text = self.read("deploy-pages.yml")
        self.assertIn("'Update Volume Profiles'", text)
        self.assertLess(text.index("volume_profile_crypto.py check"), text.index("dist/data/volume-profile.enc.json"))
        self.assertNotIn("VITE_VOLUME_PROFILE_REPO", text)
        self.assertNotIn("MINUTE_DATA", text)


if __name__ == "__main__":
    unittest.main()
