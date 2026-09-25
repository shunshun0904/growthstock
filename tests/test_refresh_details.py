#!/usr/bin/env python3
"""
画面の銘柄データの取り直し（research/refresh_details.py・Refresh Stock Details）と、
データを積むワークフローの後のデプロイのテスト（2026-09-26）。

運用者「株価推移を直近78週（サンプル母集団の定義）で表示してほしいです」。株価推移の期間は
取得スクリプトが決めるので、次の日次予測を待たずに stocks.json だけを作り直す。

- 対象の銘柄は日次予測と同じ規則で選ぶ（直近の日次予測と同じ銘柄・同じ順）
- 作り直した stocks.json は、78週になっていない・最後の点が最新の足でない、ならコミットしない
- データを積んだワークフローの後のデプロイは、そのデータのコミットを載せる。以前は起動元が
  走り始めたときのコミット（workflow_run.head_sha）を載せていたので、予測の結果は次に誰かが
  push するまで画面に出なかった（2026-09-23 の run 35902251565 は 876523d を載せ、同じ予測の
  実行が積んだ 1f18614 を載せていなかった）

  python3 tests/test_refresh_details.py
"""
import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import jquants_data_fetcher as JF  # noqa: E402
import predict_daily as P  # noqa: E402
import refresh_details as R  # noqa: E402

WF = os.path.join(ROOT, ".github", "workflows")


def read_wf(name):
    with open(os.path.join(WF, name), encoding="utf-8") as f:
        return f.read()


def wf_name(text):
    return re.search(r"^name:\s*(.+)$", text, re.M).group(1).strip().strip("'\"")


def weekdays(end, n):
    """end で終わる n 本の平日の日付（古い順）。"""
    out, d = [], dt.date.fromisoformat(end)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= dt.timedelta(days=1)
    return out[::-1]


def stock(code, first, last, n=None, as_of=None, close=1000.0):
    """first〜last を n 点（既定は78週ぶんの点の数）で結ぶ株価推移を持つ銘柄。"""
    n = n or R.full_points()
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    step = (b - a) / (n - 1)
    hist = [{"date": (a + step * i).isoformat(), "close": close} for i in range(n)]
    hist[-1]["date"] = last
    return {"code": code, "asOf": as_of or last, "history": hist,
            "milestones": [{"date": last, "type": "breakout", "title": "78週高値を更新"}]}


class TestCodes(unittest.TestCase):
    def _write(self, candidates):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"candidates": candidates}, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_same_codes_and_order_as_the_daily_run(self):
        rows = [{"date": "2026-09-24", "rankInDay": 1, "code": "1111"},
                {"date": "2026-09-25", "rankInDay": 2, "code": "2222"},
                {"date": "2026-09-25", "rankInDay": 1, "code": "3333"},
                {"date": "2026-09-24", "rankInDay": 2, "code": "2222"}]
        got = R.pick_codes(self._write(rows))
        self.assertEqual(got, P.detail_codes(rows, P.TOP_CODES))
        self.assertEqual(got, ["3333", "2222", "1111"])        # 新しい日の上位から・重複なし

    def test_no_candidates_is_an_error(self):
        """空のまま取得スクリプトを回すと、画面の銘柄が消える。"""
        path = self._write([])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(R.main(["codes", "--predictions", path]), 1)
        self.assertEqual(out.getvalue(), "")

    def test_limit_is_the_daily_default(self):
        self.assertEqual(P.TOP_CODES, 120)


class TestCheck(unittest.TestCase):
    def test_full_points_follow_the_fetcher(self):
        """369本の日足を3本ごとに間引き、最初の足も残す: 123点 + 1点。"""
        self.assertEqual(JF.CHART_BARS, JF.HIGH_WINDOW_BARS + 1)
        self.assertEqual(R.full_points(), 124)

    def test_78_weeks_ending_at_the_latest_bar_passes(self):
        r = R.check({"stocks": [stock("1111", "2025-03-28", "2026-09-25")]})
        self.assertEqual(r["problems"], [])
        self.assertEqual(round(r["longest"]["weeks"]), 78)

    def test_one_year_is_rejected(self):
        """以前の取得スクリプト（直近1年）で作られた stocks.json はコミットしない。"""
        r = R.check({"stocks": [stock("1111", "2025-09-25", "2026-09-18", n=72,
                                      as_of="2026-09-18")]})
        self.assertEqual(len(r["problems"]), 1)
        self.assertIn("78週（369営業日）ぶんの株価推移がある銘柄が無い", r["problems"][0])

    def test_last_point_must_be_the_latest_bar(self):
        """9/25 の stocks.json は最後の点が 9/18 だった（間引きを古い側から数えていた）。"""
        r = R.check({"stocks": [stock("1111", "2025-03-21", "2026-09-18", as_of="2026-09-25")]})
        self.assertEqual(len(r["problems"]), 1)
        self.assertIn("最後の点が基準日（最新の足）でない銘柄が 1（1111）", r["problems"][0])

    def test_recently_listed_stocks_may_be_shorter(self):
        r = R.check({"stocks": [stock("1111", "2025-03-28", "2026-09-25"),
                                stock("135A", "2026-01-05", "2026-09-25", n=60)]})
        self.assertEqual(r["problems"], [])
        self.assertIn("78週に満たない銘柄（上場から日が浅いなど）: 135A 38週", R.report(r))

    def test_stock_without_history_is_counted_not_fatal(self):
        r = R.check({"stocks": [stock("1111", "2025-03-28", "2026-09-25"),
                                {"code": "9999", "error": "日次株価データを取得できませんでした",
                                 "history": [], "milestones": []}]})
        self.assertEqual(r["problems"], [])
        self.assertEqual(r["no_history"], ["9999"])

    def test_report_has_no_price_values(self):
        """Actions のログは公開される。J-Quants の値（終値）は出さない。"""
        r = R.check({"stocks": [stock("1111", "2025-03-28", "2026-09-25", close=12345.6)]})
        text = R.report(r)
        self.assertNotIn("12345", text)
        self.assertNotIn("12,345", text)

    def test_the_fetchers_chart_passes(self):
        """
        取得スクリプトの今の作り（CHART_BARS 本・最新の足から間引く）は check を通る。
        祝日の無い暦（平日だけ）なので、369本は暦で約73週になる。判定が暦の週数に
        よらないことの確認でもある（実際の取引所の暦では約78週）。
        """
        dates = weekdays("2026-09-25", JF.PRICE_LOOKBACK_DAYS * 5 // 7)
        quotes = [{"Date": d, "AdjC": 1000.0 + i} for i, d in enumerate(dates)]
        hist = JF.chart_history(quotes)
        r = R.check({"stocks": [{"code": "1111", "asOf": dates[-1], "history": hist,
                                 "milestones": []}]})
        self.assertEqual(r["problems"], [])
        self.assertEqual(len(hist), R.full_points())
        self.assertEqual(hist[-1]["date"], "2026-09-25")
        self.assertEqual(round(r["longest"]["weeks"]), 73)

    def test_main_exit_codes(self):
        for s, want in ((stock("1111", "2025-03-28", "2026-09-25"), 0),
                        (stock("1111", "2025-09-25", "2026-09-25", n=72), 1)):
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump({"stocks": [s]}, f)
            f.close()
            self.addCleanup(os.unlink, f.name)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(R.main(["check", "--stocks", f.name]), want)


class TestWorkflows(unittest.TestCase):
    def test_refresh_only_rewrites_stock_details(self):
        text = read_wf("refresh-details.yml")
        self.assertIn("workflow_dispatch", text)
        self.assertNotIn("schedule:", text)
        adds = re.findall(r"git add[^\n]*", text)
        self.assertEqual(adds, ["git add public/data/stocks.json"])
        self.assertNotIn("predict_daily.py", text)             # 予測はしない
        self.assertNotIn("release upload", text)                # Release に触れない

    def test_refresh_fetches_like_the_daily_run(self):
        """取得スクリプトへの渡し方（TOKYO PRO MARKET の数え始めの一覧）が日次予測と同じ。"""
        want = ("--general-market-start "
                "research/_data/dataset_predict_general_market_start.json")
        for name in ("predict.yml", "refresh-details.yml"):
            text = read_wf(name)
            self.assertIn("scripts/jquants_data_fetcher.py --extra-codes $CODES", text, name)
            self.assertIn(want, text, name)
            self.assertIn("research/build_dataset.py --keep-unlabeled", text, name)
        self.assertIn("refresh_details.py check", read_wf("refresh-details.yml"))

    def test_every_workflow_that_commits_screen_data_is_deployed(self):
        deploy = read_wf("deploy-pages.yml")
        listed = re.search(r"workflows:\s*\[([^\]]*)\]", deploy).group(1)
        names = {n.strip().strip("'\"") for n in listed.split(",")}
        committing = []
        for name in sorted(os.listdir(WF)):
            if not name.endswith((".yml", ".yaml")):
                continue
            text = read_wf(name)
            if re.search(r"git add[^\n]*public/data/", text):
                committing.append(wf_name(text))
        self.assertIn("Refresh Stock Details", committing)
        for n in committing:
            self.assertIn(n, names, f"{n} が積んだデータが画面に出ない")

    def test_deploy_after_data_workflows_takes_the_branch_tip(self):
        deploy = read_wf("deploy-pages.yml")
        ref = re.search(r"^\s*ref:\s*(.+)$", deploy, re.M).group(1)
        self.assertNotIn("workflow_run.head_sha", ref)
        self.assertIn("claude/jquants-browser-app-bwk19a", ref)
        # 予測・取り直しが積む先と同じブランチ
        for name in ("predict.yml", "refresh-details.yml"):
            self.assertIn("ref: claude/jquants-browser-app-bwk19a", read_wf(name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
