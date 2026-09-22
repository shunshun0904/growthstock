#!/usr/bin/env python3
"""
research/edinet_fetch.py の単体テスト。ネットワークには触れない。

肝は3つ。
  - 利用枠を守ること（予算・応答ヘッダの残数・429 で止まる）
  - 取り直さないこと（200 と 404 は記録して二度と叩かない。失敗だけ再試行）
  - 列が会社ごとに違っても壊れずに保存できること
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import edinet_fetch as E  # noqa: E402
import probe_edinetdb as P  # noqa: E402

NOW = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.timezone.utc)


class FakeProbe(P.Probe):
    """paths -> [(status, body, headers), ...] を順に返す。"""

    def __init__(self, table, budget=50):
        super().__init__("TESTONLY-key", budget)
        self.table = {k: list(v) for k, v in table.items()}
        self.calls = []

    def get(self, path, params=None, auth=True):
        if self.used >= self.budget:
            return None, "budget"
        self.used += 1
        self.calls.append((path, dict(params or {})))
        q = self.table.get(path)
        if not q:
            self.last_headers = {}
            return 404, {"error": {"code": "not_found"}}
        st, body, hdrs = q.pop(0) if len(q) > 1 else q[0]
        self.last_headers = dict(hdrs)
        return st, body


def fin_rows(code, n=2, extra=None):
    rows = []
    for i in range(n):
        r = {"fiscal_year": 2025 - i, "doc_id": f"S{code}{i}", "revenue": 100.0 + i,
             "submit_date": f"{2025-i}-06-25 12:00", "is_restated_eps": False}
        r.update(extra or {})
        rows.append(r)
    return rows


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.mapping = pd.DataFrame({"edinet_code": ["E1", "E2", "E3"],
                                     "code": ["72030", "154A0", "99840"]})
        self.targets = ["72030", "154A0", "99840"]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestTargets(Base):
    def test_reads_and_normalizes(self):
        p = os.path.join(self.dir, "t.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# comment\n\n7203\n 154A \n77770\n")
        self.assertEqual(E.read_targets(p), ["72030", "154A0", "77770"])


class TestMapping(Base):
    def test_pages_until_total_pages_and_normalizes_codes(self):
        page = lambda k, rows: (200, {"data": rows, "meta": {"pagination": {
            "page": k, "per_page": 200, "total": 3, "total_pages": 2}}}, {})
        fp = FakeProbe({"/companies": [
            page(1, [{"edinet_code": "E1", "sec_code": "7203"},
                     {"edinet_code": "E2", "sec_code": "154A"}]),
            page(2, [{"edinet_code": "E3", "sec_code": "77770"}]),
        ]})
        m = E.load_manifest(self.dir)
        df = E.refresh_mapping(fp, self.dir, m, NOW, reserve=0)
        self.assertEqual(list(df["code"]), ["72030", "154A0", "77770"])
        self.assertEqual([c[1]["page"] for c in fp.calls], [1, 2])
        self.assertIsNotNone(m["mapping_at"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, E.COMPANIES)))

    def test_stops_if_page_param_is_ignored(self):
        """同じページが返り続けても無限に叩かない。"""
        same = (200, {"data": [{"edinet_code": "E1", "sec_code": "7203"}],
                      "meta": {"pagination": {"total_pages": 99}}}, {})
        fp = FakeProbe({"/companies": [same, same]}, budget=10)
        E.refresh_mapping(fp, self.dir, E.load_manifest(self.dir), NOW, reserve=0)
        self.assertEqual(fp.used, 2)


class TestFetch(Base):
    def test_ok_404_and_error_are_recorded_differently(self):
        fp = FakeProbe({
            "/companies/E1/financials": [(200, {"data": fin_rows("E1")}, {})],
            # E2 は table に無い -> 404
            "/companies/E3/financials": [(500, "boom", {})],
        })
        m = E.load_manifest(self.dir)
        done, rows, unmapped = E.fetch_financials(
            fp, self.dir, m, self.mapping, self.targets, n=10, now=NOW, reserve=0)
        self.assertEqual((done, rows, unmapped), (3, 2, 0))
        self.assertEqual(m["companies"]["E1"]["status"], "ok")
        self.assertEqual(m["companies"]["E2"]["status"], "no_data")
        self.assertEqual(m["companies"]["E3"]["status"], "error")
        fin = pd.read_parquet(os.path.join(self.dir, E.FIN))
        self.assertEqual(len(fin), 2)
        self.assertEqual(set(fin["jq_code"]), {"72030"})

        # 2回目: ok と no_data は叩かない。error だけ再試行する
        fp2 = FakeProbe({"/companies/E3/financials": [(200, {"data": fin_rows("E3")}, {})]})
        done, rows, _ = E.fetch_financials(
            fp2, self.dir, m, self.mapping, self.targets, n=10, now=NOW, reserve=0)
        self.assertEqual([c[0] for c in fp2.calls], ["/companies/E3/financials"])
        self.assertEqual(m["companies"]["E3"]["status"], "ok")
        fin = pd.read_parquet(os.path.join(self.dir, E.FIN))
        self.assertEqual(len(fin), 4)

    def test_refetch_dedupes_on_doc(self):
        body = (200, {"data": fin_rows("E1")}, {})
        fp = FakeProbe({"/companies/E1/financials": [body, body]})
        m = E.load_manifest(self.dir)
        E.fetch_financials(fp, self.dir, m, self.mapping, ["72030"], 10, NOW, 0)
        m["companies"]["E1"]["status"] = "error"      # 取り直させる
        E.fetch_financials(fp, self.dir, m, self.mapping, ["72030"], 10, NOW, 0)
        fin = pd.read_parquet(os.path.join(self.dir, E.FIN))
        self.assertEqual(len(fin), 2)

    def test_unmapped_targets_are_counted_not_fetched(self):
        fp = FakeProbe({})
        m = E.load_manifest(self.dir)
        done, rows, unmapped = E.fetch_financials(
            fp, self.dir, m, self.mapping, ["00000", "11110"], 10, NOW, 0)
        self.assertEqual((done, unmapped, fp.used), (0, 2, 0))

    def test_varying_columns_are_stored_as_union(self):
        """会社ごとに項目が違う（値が無い項目は省かれる）。和集合で壊れず保存。"""
        fp = FakeProbe({
            "/companies/E1/financials": [(200, {"data": fin_rows("E1", 1, {"goodwill": 5.0})}, {})],
            "/companies/E3/financials": [(200, {"data": fin_rows("E3", 1, {"ibd_current": 7.0})}, {})],
        })
        m = E.load_manifest(self.dir)
        E.fetch_financials(fp, self.dir, m, self.mapping, ["72030", "99840"], 10, NOW, 0)
        fin = pd.read_parquet(os.path.join(self.dir, E.FIN))
        self.assertIn("goodwill", fin.columns)
        self.assertIn("ibd_current", fin.columns)
        self.assertEqual(int(fin["goodwill"].notna().sum()), 1)
        self.assertEqual(int(fin["ibd_current"].notna().sum()), 1)


class TestQuota(Base):
    def test_stops_when_probe_budget_is_exhausted(self):
        fp = FakeProbe({f"/companies/E{i}/financials": [(200, {"data": fin_rows(f"E{i}")}, {})]
                        for i in (1, 2, 3)}, budget=2)
        m = E.load_manifest(self.dir)
        done, *_ = E.fetch_financials(fp, self.dir, m, self.mapping, self.targets, 10, NOW, 0)
        self.assertEqual(done, 2)
        self.assertEqual(fp.used, 2)

    def test_stops_when_header_remaining_hits_reserve(self):
        low = {"x-ratelimit-remaining": "5"}
        fp = FakeProbe({f"/companies/E{i}/financials": [(200, {"data": fin_rows(f"E{i}")}, low)]
                        for i in (1, 2, 3)}, budget=50)
        m = E.load_manifest(self.dir)
        done, *_ = E.fetch_financials(fp, self.dir, m, self.mapping, self.targets, 10, NOW,
                                      reserve=10)
        # 1社目の応答で残数 5 <= 予備 10 と分かるので、2社目は送らない
        self.assertEqual(done, 1)
        self.assertEqual(fp.used, 1)

    def test_429_stops_without_recording(self):
        fp = FakeProbe({"/companies/E1/financials": [(429, {"error": "limit"}, {})]})
        m = E.load_manifest(self.dir)
        done, *_ = E.fetch_financials(fp, self.dir, m, self.mapping, self.targets, 10, NOW, 0)
        self.assertEqual(done, 0)
        self.assertNotIn("E1", m["companies"])

    def test_clamp_budget_uses_header(self):
        fp = FakeProbe({}, budget=50)
        fp.used = 3
        fp.last_headers = {"X-RateLimit-Remaining": "20"}
        E.clamp_budget(fp, reserve=5)
        self.assertEqual(fp.budget, 3 + 15)

    def test_manifest_roundtrip_and_month_key(self):
        m = E.load_manifest(self.dir)
        m["requests"][E.month_key(NOW)] = 42
        E.save_manifest(self.dir, m)
        m2 = E.load_manifest(self.dir)
        self.assertEqual(m2["requests"]["2026-09"], 42)
        self.assertIn("updatedAt", m2)


class TestFetchOrder(unittest.TestCase):
    def setUp(self):
        self.map = {"10000": "E1", "20000": "E2", "30000": "E3", "40000": "E4"}

    def test_unfetched_first_then_oldest_refresh(self):
        old = (NOW - dt.timedelta(days=100)).isoformat()
        older = (NOW - dt.timedelta(days=200)).isoformat()
        fresh = (NOW - dt.timedelta(days=10)).isoformat()
        comp = {"E1": {"status": "ok", "at": old},
                "E2": {"status": "no_data", "at": older},
                "E3": {"status": "error", "at": fresh},
                "E4": {"status": "ok", "at": fresh}}
        order = E.fetch_order(["10000", "20000", "30000", "40000", "99990"], self.map, comp, NOW)
        # 失敗は先、次に古い順（E2 200日 → E1 100日）。10日前の E4 と対応表に無い 99990 は含まない
        self.assertEqual(order, ["30000", "20000", "10000"])

    def test_never_fetched_keeps_target_order(self):
        order = E.fetch_order(["40000", "10000", "20000"], self.map, {}, NOW)
        self.assertEqual(order, ["40000", "10000", "20000"])

    def test_missing_at_counts_as_stale(self):
        comp = {"E1": {"status": "ok"}}
        self.assertEqual(E.fetch_order(["10000"], self.map, comp, NOW), ["10000"])

    def test_refresh_appends_new_year_rows(self):
        d = tempfile.mkdtemp()
        try:
            mapping = pd.DataFrame({"edinet_code": ["E1"], "code": ["10000"]})
            m = E.load_manifest(d)
            hdr = {"x-ratelimit-remaining": "50"}
            first = FakeProbe({"/companies/E1/financials": [(200, fin_rows("E1", 2), hdr)]})
            E.fetch_financials(first, d, m, mapping, ["10000"], 5, NOW, 10)
            self.assertEqual(m["companies"]["E1"]["status"], "ok")
            # 61日後、1年分増えた応答で取り直す
            later = NOW + dt.timedelta(days=61)
            rows = fin_rows("E1", 2)
            rows.insert(0, {"fiscal_year": 2026, "doc_id": "SE1new", "revenue": 300.0,
                            "submit_date": "2026-06-25 12:00", "is_restated_eps": False})
            second = FakeProbe({"/companies/E1/financials": [(200, rows, hdr)]})
            done, new_rows, _ = E.fetch_financials(second, d, m, mapping, ["10000"], 5, later, 10)
            self.assertEqual(done, 1)
            fin = pd.read_parquet(os.path.join(d, E.FIN))
            self.assertEqual(len(fin), 3)                       # 重複せず 1年分だけ増える
            self.assertEqual(int(m["companies"]["E1"]["last_fy"]), 2026)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
