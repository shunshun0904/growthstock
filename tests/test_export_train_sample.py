#!/usr/bin/env python3
"""
学習データの見本の書き出し（research/export_train_sample.py）。ネットには出ない（偽のシートで確かめる）。

- 行はラベルが確定している最新の日からさかのぼり、日をまたがずに指定の行数に届くまで（ラベル未確定の行は入れない）
- 列は 日付・コード・銘柄名・ラベル3種・学習に使う特徴量（DEFAULT_PRESET の並びのまま）。1行目に日本語名、2行目に列名
- 値は学習器に渡す値のまま（欠測は空欄、無限大は inf）。1.5σ・1.8σ のラベルは引き直し（1.2σ ⊇ 1.5σ ⊇ 1.8σ）
- タブは中身を入れ替える（無ければ作る）。書き込みは小分け。ログに銘柄コード・銘柄名を出さない
- 書き出し先は学習データの見本の別のスプレッドシート（運用者の依頼 2026-10-20「同じスプレッドシートの中にあると重くて開けない」）。
  いつものスプレッドシートと同じ ID なら止まる。書いて読み戻しが一致したときだけ、いつものスプレッドシートの古い2タブを消す
  （食い違えば消さない）。新しいスプレッドシートの空の「シート1」は消すが、中身のあるタブ・ほかの名前のタブには触らない

  python3 tests/test_export_train_sample.py
"""
import contextlib
import io
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import build_dataset as B  # noqa: E402
import export_train_sample as X  # noqa: E402
import features as F  # noqa: E402

COLS = F.columns(F.DEFAULT_PRESET)


def dataset(per_day=(("2026-09-10", 3), ("2026-09-11", 4), ("2026-09-14", 5)), unlabeled=2, seed=0):
    """
    日ごとの行数を決めた、学習データの形の偽物。特徴量を先に置き、到達・終盤・トレンドの列からラベルを作る
    （vol_20d は特徴量でもあり、ラベルのしきい値にも使う。本物と同じく、ラベルを作った後で値を変えない）。
    """
    rng = np.random.default_rng(seed)
    days = [d for d, k in per_day for _ in range(k)]
    keys = [{"Date": pd.Timestamp(d), "Code": f"{9000 + i}0"} for i, d in enumerate(days)]
    keys += [{"Date": pd.Timestamp("2026-09-15"), "Code": f"{8000 + i}0"} for i in range(unlabeled)]  # ラベル未確定
    m = len(keys)
    ds = pd.concat([pd.DataFrame(keys), pd.DataFrame(rng.normal(0, 1, (m, len(COLS))), columns=COLS)], axis=1)
    ds.loc[0, COLS[0]] = np.nan
    ds.loc[0, COLS[1]] = np.inf
    ds["vol_20d"] = rng.uniform(1.0, 4.0, m)
    ds["future_rise"] = rng.uniform(-0.1, 0.6, m)
    ds["end_level"] = rng.uniform(-0.1, 0.4, m)
    ds["uptrend_end"] = rng.integers(0, 2, m).astype(float)
    ds["label"] = 0.0
    ds["label"] = B.rise_label_for_k(ds, 1.2)
    if unlabeled:
        ds.loc[ds.index[-unlabeled:], "label"] = np.nan
    ds["ref_end"] = 0.0                                           # 学習に使わない列は出さない
    return ds


class FakeWS:
    """書いた値を覚え、読み戻しで返す（Sheets のように、行末の空欄は返さず、数は数・文字は文字のまま）。"""

    def __init__(self, title, rows, cols):
        self.title, self.rows, self.cols = title, rows, cols
        self.calls = []
        self.grid = {}
        self.tamper = None                                       # 読み戻しで1セルだけ変える（食い違いの確かめ）

    def clear(self):
        self.calls.append(("clear",))
        self.grid = {}

    def resize(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.calls.append(("resize", rows, cols))

    def update(self, values, rng, value_input_option=None):
        self.calls.append(("update", rng, len(values), value_input_option))
        top = int(rng[1:]) - 1
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                self.grid[(top + i, j)] = v

    def get_values(self, value_render_option=None):
        self.calls.append(("get_values", value_render_option))
        if not self.grid:
            return []
        out = []
        for i in range(max(r for r, _ in self.grid) + 1):
            row = [self.grid.get((i, j), "") for j in range(max(c for _, c in self.grid) + 1)]
            while row and row[-1] == "":
                row.pop()
            out.append(row)
        if self.tamper:
            (i, j), v = self.tamper
            out[i][j] = v
        return out

    def freeze(self, rows=None, cols=None):
        self.calls.append(("freeze", rows, cols))

    def format(self, rng, fmt):
        self.calls.append(("format", rng))


class FakeBook:
    def __init__(self, existing=()):
        self.sheets = {t: FakeWS(t, 5, 5) for t in existing}
        self.added = []
        self.deleted = []

    def worksheets(self):
        return list(self.sheets.values())

    def del_worksheet(self, ws):
        assert len(self.sheets) > 1, "スプレッドシートのタブを全部は消せない"
        del self.sheets[ws.title]
        self.deleted.append(ws.title)

    def worksheet(self, title):
        import gspread
        if title not in self.sheets:
            raise gspread.WorksheetNotFound(title)
        return self.sheets[title]

    def add_worksheet(self, title, rows, cols):
        self.sheets[title] = FakeWS(title, rows, cols)
        self.added.append((title, rows, cols))
        return self.sheets[title]


class TestPick(unittest.TestCase):
    def test_code4(self):
        self.assertEqual(X.code4("72030"), "7203")
        self.assertEqual(X.code4("285A0"), "285A")
        self.assertEqual(X.code4("25935"), "25935")           # 末尾が0でない（普通株以外）はそのまま
        self.assertEqual(X.code4(None), "")

    def test_recent_whole_days(self):
        ds = dataset()
        s = X.pick_recent(ds, 6)                               # 最新の日 5行 → 足りないので前の日 4行も → 9行
        self.assertEqual(len(s), 9)
        self.assertEqual(sorted(s["Date"].dt.strftime("%m-%d").unique()), ["09-11", "09-14"])
        self.assertTrue(s["label"].notna().all())              # ラベル未確定の行（09-15）は入れない
        self.assertTrue((s[["Date", "Code"]].apply(tuple, axis=1).tolist()
                         == sorted(s[["Date", "Code"]].apply(tuple, axis=1).tolist())))
        self.assertEqual(len(X.pick_recent(ds, 5)), 5)        # ちょうど1日で届けばその日だけ
        self.assertEqual(len(X.pick_recent(ds, 1000)), 12)    # 足りなければある分だけ

    def test_labels_nested(self):
        s = X.attach_labels(X.pick_recent(dataset(per_day=(("2026-09-14", 60),), seed=3), 1000))
        # 学習データの行（1.2σ のラベルが確定）は、どの倍数でも判定できる。しきい値が上がるほど正例は絞られる
        self.assertTrue(s[["label", "label_k15", "label_k18"]].notna().all().all())
        self.assertTrue(((s["label_k15"] <= s["label"]) & (s["label_k18"] <= s["label_k15"])).all())
        self.assertGreater(s["label"].sum(), s["label_k18"].sum())
        # 1.2σ で引き直すと学習データのラベルと一致する
        self.assertTrue((B.rise_label_for_k(s, 1.2).fillna(-1) == s["label"].fillna(-1)).all())


class TestCells(unittest.TestCase):
    def test_cell(self):
        self.assertEqual(X.cell(None), "")
        self.assertEqual(X.cell(float("nan")), "")
        self.assertEqual(X.cell(np.float32("nan")), "")
        self.assertEqual(X.cell(float("inf")), "inf")
        self.assertEqual(X.cell(-np.inf), "-inf")
        self.assertEqual(X.cell(1.0), 1)
        self.assertIsInstance(X.cell(np.int64(3)), int)
        self.assertEqual(X.cell(0.25), 0.25)
        self.assertEqual(X.cell(pd.Timestamp("2026-09-14")), "2026-09-14")
        self.assertEqual(X.cell("abc"), "abc")

    def test_table(self):
        s = X.attach_labels(X.pick_recent(dataset(), 1000))
        names = {str(s.loc[0, "Code"]): "テスト工業"}
        t = X.table(s, names, COLS)
        self.assertEqual(len(t), 2 + len(s))
        self.assertEqual(len(t[0]), 6 + len(COLS))
        self.assertTrue(all(len(r) == len(t[0]) for r in t))
        self.assertEqual(t[0][:6], ["日付", "コード", "銘柄名", "ラベル 1.2σ", "ラベル 1.5σ", "ラベル 1.8σ"])
        self.assertEqual(t[1][:6], ["Date", "Code", "CoName", "label", "label_k15", "label_k18"])
        self.assertEqual(t[1][6:], COLS)                       # 特徴量は学習器に渡す並びのまま
        self.assertNotIn("ref_end", t[1])
        first = t[2]
        self.assertEqual(first[:3], ["2026-09-10", s.loc[0, "Code"][:4], "テスト工業"])
        self.assertEqual(first[6], "")                         # 欠測は空欄
        self.assertEqual(first[7], "inf")                      # 無限大は inf
        self.assertIn(first[3], (0, 1))
        self.assertEqual(t[3][2], "")                          # 名前の分からない銘柄は空欄
        self.assertTrue(all(isinstance(v, (int, float, str)) for r in t for v in r))
        self.assertFalse(any(isinstance(v, float) and math.isnan(v) for r in t for v in r))

    def test_notes(self):
        s = X.attach_labels(X.pick_recent(dataset(), 1000))
        meta = {"trainedAt": "2026-10-11T00:34:57+00:00", "trainFrom": "2018-04-03", "trainTo": "2026-09-08",
                "nTrain": 21937}
        n = X.notes(s, COLS, 1000, meta, "2026-10-17 12:00 UTC")
        text = "\n".join(" ".join(map(str, r)) for r in n)
        for w in ("1ヶ月内+1.2σ", "1ヶ月内+1.5σ", "1ヶ月内+1.8σ", "2018-04-03 〜 2026-09-08 の 21,937行",
                  "2026-09-10 〜 2026-09-14 の 3日・12行", "翌営業日の寄り", "√20"):
            self.assertIn(w, text)
        head = n.index(["列名", "日本語名", "大区分", "グループ", "説明"])
        body = n[head + 1:]
        self.assertEqual([r[0] for r in body], ["Date", "Code", "CoName"] + COLS)
        self.assertTrue(all(r[1] and r[4] for r in body[3:]))   # 特徴量はどれも日本語名と説明がある
        self.assertNotIn("今のモデル", "\n".join(" ".join(map(str, r)) for r in X.notes(s, COLS, 1000, None, "t")))


class TestWrite(unittest.TestCase):
    def test_new_tab_chunked(self):
        book = FakeBook()
        values = [["h"] * 4] + [[i, 1, 2, 3] for i in range(450)]
        ws = X.write_tab(book, "T", values, freeze_rows=2, freeze_cols=3, bold_rows=2, chunk=200)
        self.assertEqual(book.added, [("T", 451, 4)])
        ups = [c for c in ws.calls if c[0] == "update"]
        self.assertEqual([(u[1], u[2]) for u in ups], [("A1", 200), ("A201", 200), ("A401", 51)])
        self.assertTrue(all(u[3] == "RAW" for u in ups))
        self.assertIn(("freeze", 2, 3), ws.calls)
        self.assertIn(("format", "A1:D2"), ws.calls)

    def test_existing_tab_replaced(self):
        book = FakeBook(existing=["T"])
        ws = X.write_tab(book, "T", [["a", "b"], [1, 2]])
        self.assertEqual(book.added, [])
        self.assertEqual(ws.calls[:2], [("clear",), ("resize", 2, 2)])
        self.assertNotIn("freeze", [c[0] for c in ws.calls])


class TestVerify(unittest.TestCase):
    def test_same(self):
        self.assertTrue(X.same(1, 1.0))
        self.assertTrue(X.same(0.1 + 0.2, 0.30000000000000004))
        self.assertFalse(X.same(0.1, 0.1000001))
        self.assertTrue(X.same("", ""))
        self.assertTrue(X.same("inf", "inf"))
        self.assertFalse(X.same("7203", 7203))                  # コードは文字のまま書く（数にならない）
        self.assertFalse(X.same("", 0))

    def test_readback(self):
        book = FakeBook()
        values = [["日付", "コード", "x"], ["Date", "Code", "x"], ["2026-09-14", "7203", 0.5], ["2026-09-14", "285A", ""]]
        ws = X.write_tab(book, "T", values)
        self.assertEqual(X.verify_tab(ws, values), (12, 0, None))
        ws.tamper = ((2, 2), 0.25)
        self.assertEqual(X.verify_tab(ws, values), (12, 1, (3, 3)))


class TestMain(unittest.TestCase):
    def setUp(self):
        # 手元の環境変数で本物のスプレッドシートを開かない
        self._env = {k: os.environ.pop(k) for k in ("GSHEET_ID", "GSHEET_SAMPLE_ID") if k in os.environ}

    def tearDown(self):
        os.environ.update(self._env)

    def run_main(self, argv, book=None, main_book=None, tamper=None):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dataset.parquet")
            dataset().to_parquet(path, index=False)
            pd.DataFrame({"Code": ["90000", "90010"], "CoName": ["テスト工業", "見本商事"]}).to_parquet(
                os.path.join(d, "master.parquet"), index=False)
            if tamper:
                orig = book.add_worksheet

                def add(title, rows, cols):
                    ws = orig(title, rows, cols)
                    if title == X.SAMPLE_TITLE:
                        ws.tamper = tamper
                    return ws
                book.add_worksheet = add
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = X.main(["--dataset", path, "--data-dir", d] + argv, book=book, main_book=main_book)
        return rc, out.getvalue()

    def test_dry_run_logs_no_codes(self):
        rc, log = self.run_main(["--rows", "6", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("[sample] 9行 / 2日", log)
        self.assertIn("[dry-run] 書かない（見本 11行", log)
        for w in ("9000", "9001", "9005", "テスト工業", "見本商事"):
            self.assertNotIn(w, log)

    def test_writes_both_tabs(self):
        book = FakeBook(existing=[X.NOTES_TITLE])
        rc, log = self.run_main(["--rows", "1000"], book=book)
        self.assertEqual(rc, 0)
        self.assertEqual([a[0] for a in book.added], [X.SAMPLE_TITLE])
        self.assertEqual(book.added[0][1:], (14, 6 + len(COLS)))   # 2行の見出し + 12行
        self.assertIn(("freeze", 2, 3), book.sheets[X.SAMPLE_TITLE].calls)
        self.assertIn(("clear",), book.sheets[X.NOTES_TITLE].calls)
        self.assertIn(f"タブ「{X.SAMPLE_TITLE}」に 12行", log)
        self.assertIn("[check] 1.2σ のラベルを引き直すと学習データのラベルと一致: 12 / 12", log)
        self.assertIn("[check] 銘柄名の分からない行（今の銘柄一覧に無い）: 10", log)
        self.assertIn(f"[verify] 「{X.SAMPLE_TITLE}」を読み戻して {14 * (6 + len(COLS)):,}セルを比べた: 食い違い 0", log)
        self.assertIn(f"[verify] 「{X.NOTES_TITLE}」を読み戻して", log)

    def test_label_mismatch_stops(self):
        """引き直したラベルが学習データと合わなければ書かない。"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dataset.parquet")
            ds = dataset()
            ds.loc[0, "label"] = 1.0 - ds.loc[0, "label"]
            ds.to_parquet(path, index=False)
            book = FakeBook()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = X.main(["--dataset", path, "--data-dir", d, "--rows", "1000"], book=book)
        self.assertEqual(rc, 1)
        self.assertEqual(book.added, [])
        self.assertIn("[fatal]", out.getvalue())

    def test_rows_limit(self):
        with self.assertRaises(SystemExit):
            self.run_main(["--rows", "0", "--dry-run"])

    def test_moves_tabs_from_main(self):
        """別のスプレッドシートに書いて確かめた後、いつものスプレッドシートの古い2タブを消す（ほかのタブは残す）。"""
        book = FakeBook(existing=["シート1"])
        main = FakeBook(existing=["予測", X.SAMPLE_TITLE, X.NOTES_TITLE, "保有の推移"])
        rc, log = self.run_main(["--rows", "1000"], book=book, main_book=main)
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(book.sheets), sorted([X.SAMPLE_TITLE, X.NOTES_TITLE]))   # 空のシート1は消した
        self.assertEqual(book.deleted, ["シート1"])
        self.assertEqual(sorted(main.sheets), ["予測", "保有の推移"])
        self.assertEqual(main.deleted, [X.SAMPLE_TITLE, X.NOTES_TITLE])
        self.assertEqual(main.added, [])                                                  # いつものシートには書かない
        self.assertIn(f"[move] いつものスプレッドシートから古いタブを消した: 「{X.SAMPLE_TITLE}」・「{X.NOTES_TITLE}」", log)
        self.assertIn("[sheet] 作ったときからある空のタブを消した: 1枚", log)
        # 2回目: 消すものは無い
        rc, log = self.run_main(["--rows", "1000"], book=book, main_book=main)
        self.assertEqual(rc, 0)
        self.assertIn("[move] いつものスプレッドシートに古いタブは無い（消すものなし）", log)

    def test_keeps_main_tabs_when_readback_differs(self):
        """読み戻しが食い違えば、いつものスプレッドシートのタブは消さない。"""
        book = FakeBook()
        main = FakeBook(existing=["予測", X.SAMPLE_TITLE, X.NOTES_TITLE])
        rc, log = self.run_main(["--rows", "1000"], book=book, main_book=main, tamper=((3, 0), "ちがう"))
        self.assertEqual(rc, 1)
        self.assertEqual(main.deleted, [])
        self.assertIn("[move] 読み戻しが食い違ったので、いつものスプレッドシートのタブは消さない", log)

    def test_default_tab_with_content_is_kept(self):
        """既定の名前でも中身のあるタブ・ほかの名前のタブは消さない。"""
        book = FakeBook(existing=["シート1", "メモ"])
        book.sheets["シート1"].grid[(0, 0)] = "手で書いたメモ"
        rc, _ = self.run_main(["--rows", "1000"], book=book)
        self.assertEqual(rc, 0)
        self.assertEqual(book.deleted, [])

    def test_refuses_main_sheet(self):
        """書き出し先がいつものスプレッドシートと同じ ID なら止まる（重くて開けなくなるので）。"""
        with self.assertRaises(SystemExit) as cm:
            self.run_main(["--rows", "6", "--sheet-id", "abc", "--main-sheet-id", "abc"])
        self.assertIn("いつものスプレッドシート", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
