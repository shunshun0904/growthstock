#!/usr/bin/env python3
"""
保有の推移のタブ（research/holdings_sheet.py）の単体テスト。通信しない。

固定すること
  - 取引の記録タブは見出しの名前で読む。予測日とコードで予測ログの行を見つける
  - 売りは、同じコードで売った日までに買った保有のうち最も新しいものに付く。取消は id の先頭で効く
  - 売りのルール: 買った日を1日目に 20 営業日以内に日中の高値が 買値×1.10 に届いたらその値で売り。
    届かなければ 20 日目の終値で売り、含み損なら持ち越し（その後はルールで売らない）
  - 報告の売りはルールより優先（値段が無ければその日の終値、日足より先の日なら値段で確定）
  - 営業日は取引所のカレンダーで数える。買う日の日足が欠けていたら待つ
  - 日付ごとの表: 銘柄の列は売った日まで、合計には確定した損益が残る
  - グラフは「損益(円)」の列だけ（軸は1本）。合計は淡い青の太い帯。銘柄は保有中と売って
    20 営業日以内のもの。期間の重なる銘柄どうしは違う色で、あとから買っても色が変わらない
  - 予測ログは保有中を赤、売却済みを青（黄色は文字が見えない。2026-10-09）、取り消した行は塗りを外す
  - 公開ログに銘柄名・株価・損益の値を出さない

  python3 tests/test_holdings_sheet.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import holdings_sheet as H  # noqa: E402
import trades_inbox as TI  # noqa: E402
import trading_calendar as TC  # noqa: E402

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    HAVE_CRYPTO = True
except Exception:                         # noqa: BLE001  手元の環境で読み込めないとき
    HAVE_CRYPTO = False

T = pd.Timestamp
#: 9/24（予測日）から 30 営業日。テストでは土日だけを休みにする
DAYS = list(pd.bdate_range("2026-09-24", periods=30))
CAL = TC.Calendar([d.date() for d in DAYS])
ID_A, ID_B, ID_C = "a" * 32, "b" * 32, "c" * 32


def make_bars(code="69120", days=None, o=None, h=None, c=None, adj=None):
    """日足。h を省くと始値と終値の高いほう。adj は調整の比（AdjX = X * adj）。"""
    days = DAYS if days is None else days
    n = len(days)
    o = [1000.0] * n if o is None else list(o)
    c = list(o) if c is None else list(c)
    h = [max(a, b) for a, b in zip(o, c)] if h is None else list(h)
    df = pd.DataFrame({"Date": days, "Code": code, "O": o, "H": h, "C": c})
    k = np.ones(n) if adj is None else np.asarray(adj, dtype=float)
    for col in ("O", "H", "C"):
        df["Adj" + col] = df[col] * k
    return df


def with_value(n, base, changes):
    out = [base] * n
    for i, v in changes.items():
        out[i] = v
    return out


def pos(code="6912", pick="2026-09-24", buy=None, entry=None, shares=100, sell=None,
        sell_price=None, row=2, name="A社"):
    return H.Position(pick_date=pick, code=code, name=name, row=row,
                      buy_date=T(buy) if buy else None, entry_price=entry, shares=shares,
                      sell_date=T(sell) if sell else None, sell_price=sell_price)


# --------------------------------------------------------------------------- #
# 取引の記録
# --------------------------------------------------------------------------- #

class LogTab(unittest.TestCase):
    def test_rows_are_read_by_header_name(self):
        values = [["コード", "id", "種類", "メモ"],
                  ["6912", ID_A, "買い", ""],
                  ["", "", "", ""],                         # 空の行は飛ばす
                  ["7713", ID_B, "売り"]]                    # 右端が欠けた行
        rows = H.log_rows(values)
        self.assertEqual([r["コード"] for r in rows], ["6912", "7713"])
        self.assertEqual(rows[1]["メモ"], "")
        self.assertEqual(H.log_rows([]), [])

    def test_new_events_skip_known_ids_and_follow_the_report_time(self):
        rows = [{"id": ID_A}]
        events = [{"id": ID_C, "at": "2026-10-08T21:00:00+09:00"},
                  {"id": ID_A, "at": "2026-10-08T20:00:00+09:00"},
                  {"id": ID_B, "at": "2026-10-08T20:30:00+09:00"},
                  {"id": "d" * 32, "at": "2026-10-08T21:00:00+09:00"}]     # 同じ時刻は受け皿の順
        self.assertEqual([e["id"] for e in H.new_events(rows, events)], [ID_B, ID_C, "d" * 32])
        self.assertEqual([e["id"] for e in H.new_events([], events[::-1])][2:],
                         ["d" * 32, ID_C])

    def test_event_record(self):
        e = {"id": ID_A, "at": "t", "action": "buy", "pick_date": "2026-09-24", "code": "6912",
             "date": None, "price": None, "shares": 200.0, "note": "", "ref": None}
        rec = H.event_record(e, {"6912": "A社"})
        self.assertEqual(set(rec), set(H.LOG_HEADER))
        self.assertEqual((rec["種類"], rec["銘柄名"], rec["日付"], rec["価格"], rec["株数"]),
                         ("買い", "A社", "", "", 200.0))
        c = H.event_record({"id": ID_B, "action": "cancel", "code": None, "ref": "aaaaaaaa"}, {})
        self.assertEqual((c["種類"], c["取消の対象"], c["コード"]), ("取消", "aaaaaaaa", ""))


class PositionsFromLog(unittest.TestCase):
    PRED = [["予測日", "コード", "銘柄名", "建値"],
            ["2026-09-24", "6912", "A社", ""],
            ["2026/09/28", "7713", "C社", ""],
            ["2026-09-24", "1111", "B社", ""],
            ["2026-10-02", "6912", "A社", ""]]

    @staticmethod
    def rec(rid, kind, code="6912", pick="", day="", price="", shares="", ref=""):
        return {"id": rid, "種類": kind, "コード": code, "予測日": pick, "日付": day,
                "価格": price, "株数": shares, "取消の対象": ref}

    def test_buy_finds_the_prediction_row(self):
        ps, issues, cleared = H.positions_from_log(
            [self.rec(ID_A, "買い", pick="2026-09-24"),
             self.rec(ID_B, "buy", code="7713", pick="2026-09-28", price="1,234", shares="300")],
            self.PRED)
        self.assertEqual((issues, cleared), ([], set()))
        a, c = ps
        self.assertEqual((a.row, a.name, a.jq_code, a.shares, a.entry_price, a.buy_date),
                         (2, "A社", "69120", H.DEFAULT_SHARES, None, None))
        self.assertEqual((c.row, c.name, c.entry_price, c.shares), (3, "C社", 1234.0, 300.0))

    def test_sell_goes_to_the_latest_buy_made_by_that_day(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24"),
                self.rec(ID_B, "買い", pick="2026-10-02"),
                self.rec("d" * 32, "売り", day="2026-09-30", price="1100"),    # 2回目の買いより前
                self.rec("e" * 32, "売り", day="2026-10-20")]
        ps, issues, _ = H.positions_from_log(rows, self.PRED)
        first, second = ps
        self.assertEqual((first.sell_date, first.sell_price), (T("2026-09-30"), 1100.0))
        self.assertEqual((second.sell_date, second.sell_price), (T("2026-10-20"), None))
        self.assertEqual(issues, [])

    def test_sell_without_a_matching_buy_is_reported(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24"),
                self.rec(ID_B, "売り", code="7713", day="2026-10-20"),
                self.rec(ID_C, "売り", day="2026-09-20")]                  # 買う前
        ps, issues, _ = H.positions_from_log(rows, self.PRED)
        self.assertIsNone(ps[0].sell_date)
        self.assertEqual(issues, ["売りに合う買いが無い"] * 2)

    def test_cancel_by_id_prefix_clears_that_row(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24"),
                self.rec(ID_B, "買い", code="1111", pick="2026-09-24"),
                self.rec(ID_C, "取消", code="", ref=ID_B[:8].upper())]
        ps, issues, cleared = H.positions_from_log(rows, self.PRED)
        self.assertEqual([p.code for p in ps], ["6912"])
        self.assertEqual((issues, cleared), ([], {4}))

    def test_cancelled_then_reported_again_keeps_the_row(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24", price="999"),
                self.rec(ID_C, "取消", code="", ref=ID_A[:8]),
                self.rec(ID_B, "買い", pick="2026-09-24", price="1000")]
        ps, issues, cleared = H.positions_from_log(rows, self.PRED)
        self.assertEqual([(p.row, p.entry_price) for p in ps], [(2, 1000.0)])
        self.assertEqual((issues, cleared), ([], set()))

    def test_a_cancelled_sell_is_ignored(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24"),
                self.rec(ID_B, "売り", day="2026-10-20"),
                self.rec(ID_C, "取消", code="", ref=ID_B[:10])]
        ps, _, _ = H.positions_from_log(rows, self.PRED)
        self.assertIsNone(ps[0].sell_date)

    def test_bad_rows_are_reported_without_values(self):
        rows = [self.rec(ID_A, "買い", pick="2026-09-24"),
                self.rec(ID_B, "買い", pick="2026-09-24"),                 # 同じ買い
                self.rec(ID_C, "買い", code="", pick="2026-09-24"),         # コードが無い
                self.rec("d" * 32, "買った"),                               # 種類が違う
                self.rec("e" * 32, "取消", ref="abc"),                       # 短すぎる
                self.rec("f" * 32, "買い", code="9999", pick="2026-09-24")]  # 予測ログに無い
        ps, issues, _ = H.positions_from_log(rows, self.PRED)
        self.assertEqual([p.code for p in ps], ["6912", "9999"])
        self.assertEqual(ps[1].row, None)
        self.assertEqual(sorted(issues), sorted([
            "同じ買いが2回ある", "買いの行に コード・予測日（か日付）が無い",
            "種類が 買い・売り・取消 のどれでもない", "取消の対象が短い（id の先頭 8 文字以上）",
            "買いの予測日とコードに合う予測ログの行が無い"]))
        for text in issues:
            for word in ("6912", "9999", "A社", "2026"):
                self.assertNotIn(word, text)


# --------------------------------------------------------------------------- #
# 売りのルール
# --------------------------------------------------------------------------- #

class Simulate(unittest.TestCase):
    N = len(DAYS)

    def test_buys_at_the_next_open_and_holds_until_the_rule_decides(self):
        b = make_bars(days=DAYS[:10], o=[990] + [1000] * 9, c=[995, 1010, 990] + [1000] * 7)
        p = pos()
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.reason, p.issue), ("保有中", "", ""))
        self.assertEqual((p.entry_day, p.entry), (DAYS[1], 1000.0))
        self.assertEqual(p.day20, DAYS[20])                    # 買った日が1日目
        self.assertEqual(list(path["Date"]), DAYS[1:10])
        self.assertEqual(list(path["pnl_yen"][:2]), [1000.0, -1000.0])
        self.assertIsNone(p.exit_day)

    def test_intraday_high_at_plus_10_percent_sells_at_that_price(self):
        b = make_bars(h=with_value(self.N, 1000, {5: 1100}))
        p = pos()
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.reason), (H.SOLD, "+10%に到達"))
        self.assertEqual(p.exit_day, DAYS[5])
        self.assertAlmostEqual(p.exit_price, 1100.0)
        self.assertEqual(path["Date"].iloc[-1], DAYS[5])        # 売った日で終わる
        self.assertAlmostEqual(p.target, 1100.0)
        self.assertEqual((p.peak, p.peak_day), (1100.0, DAYS[5]))
        self.assertAlmostEqual(path["pnl_yen"].iloc[-1], 10000.0)
        self.assertAlmostEqual(path["pnl_pct"].iloc[-1], 10.0)

    def test_a_high_just_below_the_target_does_not_sell(self):
        b = make_bars(h=with_value(self.N, 1000, {5: 1099.9}), c=with_value(self.N, 1000, {20: 990}))
        p = pos()
        H.simulate(p, b, CAL)
        self.assertEqual(p.state, "持ち越し中")

    def test_the_target_can_be_hit_on_the_buying_day(self):
        b = make_bars(h=with_value(self.N, 1000, {1: 1150}))
        p = pos()
        H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.exit_day), (H.SOLD, DAYS[1]))
        self.assertAlmostEqual(p.exit_price, 1100.0)

    def test_day_20_close_sells_when_not_below_the_entry(self):
        b = make_bars(c=with_value(self.N, 1000, {20: 1050}))
        p = pos()
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.reason, p.exit_day, p.exit_price),
                         (H.SOLD, "20日目の終値", DAYS[20], 1050.0))
        self.assertEqual(len(path), 20)
        self.assertEqual(path["pnl_yen"].iloc[-1], 5000.0)
        # 損益ゼロ（終値 = 買値）も売る
        p0 = pos()
        H.simulate(p0, make_bars(), CAL)
        self.assertEqual((p0.state, p0.exit_day, p0.exit_price), (H.SOLD, DAYS[20], 1000.0))

    def test_a_loss_on_day_20_is_carried_and_the_rule_stops(self):
        b = make_bars(c=with_value(self.N, 1000, {20: 950}), h=with_value(self.N, 1000, {22: 1200}))
        p = pos()
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.reason), ("持ち越し中", "20日目に含み損で持ち越し"))
        self.assertIsNone(p.exit_day)
        self.assertEqual(path["Date"].iloc[-1], DAYS[-1])       # 最新日まで持つ
        self.assertEqual(p.peak, 1000.0)                        # 20日目のあとの高値 1200 は見ない
        self.assertEqual(path["pnl_yen"].iloc[20 - 1], -5000.0)

    def test_day_20_waits_for_its_data(self):
        b = make_bars(days=DAYS[:20])                            # 19日目まで
        p = pos()
        H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.day20), ("保有中", DAYS[20]))
        q = pos()
        H.simulate(q, b, None)                                   # カレンダーが無いと日足で数える
        self.assertEqual((q.state, q.day20), ("保有中", None))

    def test_reported_sell_comes_before_the_rule(self):
        b = make_bars(h=with_value(self.N, 1000, {5: 1100}))
        p = pos(sell=DAYS[3], sell_price=1030)
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.reason, p.exit_day, p.exit_price),
                         (H.SOLD, "報告", DAYS[3], 1030.0))
        self.assertEqual(path["pnl_yen"].iloc[-1], 3000.0)
        self.assertEqual(p.peak, 1000.0)                        # 売った日のあとの高値は見ない

    def test_reported_sell_without_price_uses_that_days_close(self):
        c = with_value(self.N, 1000, {20: 950, 23: 970})
        p = pos(sell=DAYS[23])
        H.simulate(p, make_bars(c=c), CAL)
        self.assertEqual((p.state, p.exit_day, p.exit_price), (H.SOLD, DAYS[23], 970.0))

    def test_reported_sell_after_the_saved_bars(self):
        b = make_bars(days=DAYS[:10])
        p = pos(sell=DAYS[12], sell_price=1020)
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.state, p.exit_day), (H.SOLD, DAYS[12]))
        self.assertEqual(list(path["Date"][-2:]), [DAYS[9], DAYS[12]])
        self.assertEqual(path["pnl_yen"].iloc[-1], 2000.0)
        q = pos(sell=DAYS[12])                                   # 値段が無い → その日の終値を待つ
        path = H.simulate(q, b, CAL)
        self.assertEqual(q.state, "保有中")
        self.assertIn("終値待ち", q.reason)
        self.assertEqual(path["Date"].iloc[-1], DAYS[9])

    def test_reported_sell_dates_that_do_not_fit(self):
        p = pos(sell="2026-09-20")                              # 買う前 → ルールのまま
        H.simulate(p, make_bars(h=with_value(self.N, 1000, {5: 1100})), CAL)
        self.assertEqual((p.state, p.reason, p.issue),
                         (H.SOLD, "+10%に到達", "売りの日付が買った日より前"))
        q = pos(sell="2026-09-26", sell_price=1010)              # 土曜 → その前の営業日
        H.simulate(q, make_bars(), CAL)
        self.assertEqual((q.state, q.exit_day, q.exit_price), (H.SOLD, T("2026-09-25"), 1010.0))
        self.assertIn("営業日ではない", q.issue)

    def test_reported_entry_price_and_buy_date(self):
        b = make_bars(h=with_value(self.N, 1000, {4: 1045}))
        p = pos(entry=950)                                       # 目標は 950 * 1.1 = 1045
        path = H.simulate(p, b, CAL)
        self.assertEqual((p.exit_day, p.entry), (DAYS[4], 950.0))
        self.assertAlmostEqual(path["pnl_yen"].iloc[-1], 9500.0)
        q = pos(buy=DAYS[3])
        H.simulate(q, make_bars(o=[1000 + i for i in range(self.N)]), CAL)
        self.assertEqual((q.entry_day, q.entry, q.day20), (DAYS[3], 1003.0, DAYS[22]))

    def test_split_after_buying(self):
        """買った後に 1:2 の分割。生の値は半分になるが、買った日の株の単位に戻して数える。"""
        n = 10
        adj = [0.5] * 3 + [1.0] * (n - 3)                        # DAYS[3] に分割
        o = [1000, 1000, 1000, 505, 505, 505, 505, 505, 505, 540]
        b = make_bars(days=DAYS[:n], o=o, c=o, adj=adj)
        p = pos()
        path = H.simulate(p, b, CAL)
        self.assertEqual(p.state, "保有中")                       # 買った日の単位で 1080 は +8%
        self.assertEqual(list(path["close"].round(6)), [1000, 1000, 1010, 1010, 1010, 1010, 1010,
                                                        1010, 1080])
        self.assertAlmostEqual(path["pnl_pct"].iloc[-1], 8.0)
        # 分割のあとの生の値 560 は、買った日の単位で 1120（+12%）。高値で +10% に届いて売る
        q = pos()
        H.simulate(q, make_bars(days=DAYS[:n], o=o[:-1] + [560], c=o[:-1] + [560], adj=adj), CAL)
        self.assertEqual((q.state, q.exit_day), (H.SOLD, DAYS[9]))
        self.assertAlmostEqual(q.exit_price, 1100.0)

    def test_waiting_missing_and_no_trade(self):
        p = pos()
        self.assertIsNone(H.simulate(p, make_bars(days=DAYS[:1]), CAL))
        self.assertEqual(p.state, "買い待ち")
        q = pos(code="9999")
        self.assertIsNone(H.simulate(q, make_bars(), CAL))
        self.assertEqual(q.state, "株価なし")
        r = pos()
        nan = float("nan")
        b = make_bars(o=with_value(self.N, 1000, {1: nan}), c=with_value(self.N, 1000, {1: nan}),
                      h=with_value(self.N, 1000, {1: nan}))
        self.assertIsNone(H.simulate(r, b, CAL))
        self.assertEqual(r.state, "寄り付かず")

    def test_a_missing_buying_day_waits_instead_of_buying_later(self):
        b = make_bars(days=[DAYS[0]] + DAYS[2:10])
        p = pos()
        self.assertIsNone(H.simulate(p, b, CAL))
        self.assertEqual((p.state, p.issue), ("買い待ち", "買う日の日足が無い"))

    def test_no_trade_day_keeps_the_previous_close(self):
        nan = float("nan")
        b = make_bars(days=DAYS[:5], o=[1000, 1000, 1000, nan, 1000],
                      c=[1000, 1010, 1020, nan, 1030], h=[1000, 1010, 1020, nan, 1030])
        path = H.simulate(pos(), b, CAL)
        self.assertEqual(list(path["close"]), [1010.0, 1020.0, 1020.0, 1030.0])


# --------------------------------------------------------------------------- #
# 日付ごとの表・グラフ・タブの中身
# --------------------------------------------------------------------------- #

class Daily(unittest.TestCase):
    """A は 9/29 に +10% で売り、C は 9/29 の寄りで買って持っている。"""

    def setUp(self):
        days = DAYS[:10]
        self.bars = pd.concat([
            make_bars("69120", days, h=with_value(10, 1000, {3: 1100})),
            make_bars("77130", days, o=[500] * 10,
                      c=[500, 500, 500, 505, 510, 490, 495, 520, 530, 540])])
        self.a = pos()
        self.c = pos(code="7713", pick="2026-09-28", row=3, name="C社")
        self.ps = [self.a, self.c]
        self.paths = {k: H.simulate(p, self.bars, CAL) for k, p in enumerate(self.ps)}
        self.days = H.trading_days(self.bars, self.paths)
        self.daily = H.daily_table(self.ps, self.paths, self.days)

    def test_columns_and_holding_periods(self):
        d = self.daily.set_index("日付")
        self.assertEqual(list(self.daily.columns), [
            "日付", "合計損益(円)", "合計損益%", "保有数", "A社(6912) 損益(円)", "C社(7713) 損益(円)",
            "A社(6912) 損益%", "C社(7713) 損益%"])
        self.assertEqual(list(d.index), DAYS[1:10])
        self.assertEqual(d.loc[DAYS[3], "A社(6912) 損益(円)"], 10000)
        self.assertTrue(pd.isna(d.loc[DAYS[4], "A社(6912) 損益(円)"]))   # 売ったあとは空
        self.assertTrue(pd.isna(d.loc[DAYS[2], "C社(7713) 損益(円)"]))   # 買う前は空
        self.assertEqual(list(d["保有数"]), [1, 1, 2, 1, 1, 1, 1, 1, 1])

    def test_total_keeps_the_realized_profit(self):
        d = self.daily.set_index("日付")
        self.assertEqual(d.loc[DAYS[3], "合計損益(円)"], 10000 + 500)
        self.assertEqual(d.loc[DAYS[9], "合計損益(円)"], 10000 + 4000)
        self.assertAlmostEqual(d.loc[DAYS[9], "合計損益%"],
                               round(14000 / (1000 * 100 + 500 * 100) * 100, 2))

    def test_summary(self):
        rows = H.summary_rows(self.ps, self.paths)
        head = H.SUMMARY_HEADER
        a, c = (dict(zip(head, r)) for r in rows[:2])
        self.assertEqual((a["状態"], a["売った日"], a["売りの理由"], a["損益(円)"], a["最新終値"]),
                         (H.SOLD, "2026-09-29", "+10%に到達", 10000, ""))
        self.assertAlmostEqual(a["売値"], 1100.0)
        self.assertEqual((c["状態"], c["買った日"], c["建値"], c["20日目"], c["損益(円)"]),
                         ("保有中", "2026-09-29", 500.0, f"{DAYS[22]:%Y-%m-%d}", 4000))
        self.assertEqual((c["+10%の値"], c["20日目までの高値"], c["高値の日"]),
                         (550.0, 540.0, "2026-10-07"))
        self.assertEqual((a["+10%の値"], a["20日目までの高値"], a["高値の日"]),
                         (1100.0, 1100.0, "2026-09-29"))
        total, sold, held = rows[-3:]
        i = head.index("損益(円)")
        self.assertEqual([(r[0], r[i]) for r in (total, sold, held)],
                         [("合計", 14000), ("うち確定（売却済み）", 10000), ("うち含み（保有中）", 4000)])

    def test_same_stock_twice_gets_separate_columns(self):
        b = make_bars("69120", DAYS[:12])
        ps = [pos(), pos(pick="2026-09-30", row=5)]
        paths = {k: H.simulate(p, b, CAL) for k, p in enumerate(ps)}
        d = H.daily_table(ps, paths, H.trading_days(b, paths))
        self.assertIn("A社(6912) 9/25買い 損益(円)", d.columns)
        self.assertIn("A社(6912) 10/1買い 損益(円)", d.columns)

    def test_layout_and_chart(self):
        series = H.chart_series(self.ps, self.paths, self.days)
        self.assertEqual(series, [("合計損益(円)", "#86b6ef", 6),
                                  ("A社(6912) 損益(円)", "#eb6834", 2),
                                  ("C社(7713) 損益(円)", "#1baf7a", 2)])
        lay = H.build_layout("2026-10-07", H.summary_rows(self.ps, self.paths), self.daily,
                             len(self.ps), series)
        self.assertEqual(lay.values[3], [H._cell(h) for h in H.SUMMARY_HEADER])
        self.assertIn("'+10%の値", lay.values[3])
        self.assertEqual(lay.values[lay.daily_header_row][0], "日付")
        self.assertEqual(lay.values[lay.daily_header_row + 1][0], "2026-09-25")
        self.assertEqual([c for c, _, _ in lay.series], [1, 4, 5])
        spec = H.chart_request(123, lay)["addChart"]["chart"]["spec"]["basicChart"]
        self.assertEqual(spec["chartType"], "LINE")
        self.assertEqual({s["targetAxis"] for s in spec["series"]}, {"LEFT_AXIS"})   # 軸は1本
        first, second, third = spec["series"]
        self.assertAlmostEqual(first["colorStyle"]["rgbColor"]["red"], 0x86 / 255)
        self.assertEqual([s["lineStyle"]["width"] for s in spec["series"]], [6, 2, 2])
        self.assertAlmostEqual(second["colorStyle"]["rgbColor"]["red"], 0xeb / 255)
        self.assertAlmostEqual(third["colorStyle"]["rgbColor"]["green"], 0xaf / 255)
        rng = spec["domains"][0]["domain"]["sourceRange"]["sources"][0]
        self.assertEqual((rng["startRowIndex"], rng["endRowIndex"]),
                         (lay.daily_header_row, lay.daily_header_row + 1 + len(self.daily)))
        fmts = H.format_requests(123, lay)
        self.assertTrue(any("numberFormat" in r["repeatCell"]["cell"]["userEnteredFormat"]
                            for r in fmts))

    def test_text_that_looks_like_a_formula_stays_text(self):
        """USER_ENTERED で = + - @ から始まる文字を書くと式になる（「+10%の値」が #ERROR! になった）。"""
        self.assertEqual(H._cell("+10%の値"), "'+10%の値")
        self.assertEqual(H._cell("+10%に到達"), "'+10%に到達")
        self.assertEqual([H._cell(v) for v in ("=SUM(A1)", "-x", "@a")], ["'=SUM(A1)", "'-x", "'@a"])
        self.assertEqual([H._cell(v) for v in (-5.0, 3, "A社", "")], [-5.0, 3, "A社", ""])
        self.ps[0].reason = "+10%に到達"
        lay = H.build_layout("2026-10-07", H.summary_rows(self.ps, self.paths), self.daily, 2, [])
        for row in lay.values:
            for v in row:
                if isinstance(v, str) and v:
                    self.assertFalse(v[0] in "=+-@", v)

    def test_no_positions(self):
        lay = H.build_layout("2026-10-08", [], pd.DataFrame(), 0, [])
        self.assertIn("取引の記録がありません", lay.values[-1][0])
        self.assertIsNone(H.chart_request(1, lay))


class ChartSeries(unittest.TestCase):
    @staticmethod
    def held(code, k, name):
        p = pos(code=code, name=name)
        p.state, p.entry_day = "保有中", DAYS[k]
        return p

    @staticmethod
    def sold(code, k, out, name):
        p = pos(code=code, name=name)
        p.state, p.entry_day, p.exit_day = H.SOLD, DAYS[k], DAYS[out]
        return p

    def test_old_sales_leave_the_chart_and_their_color_is_reused(self):
        a = self.sold("1111", 1, 3, "A")         # 売ってから 26 営業日たった
        b = self.held("2222", 3, "B")            # A と重なる
        c = self.held("3333", 25, "C")           # A がグラフから消えたあとに買った
        ps = [a, b, c]
        paths = {0: None, 1: None, 2: None}
        names = [s[0] for s in H.chart_series(ps, paths, DAYS)]
        self.assertEqual(names, ["合計損益(円)", "B(2222) 損益(円)", "C(3333) 損益(円)"])
        colors = dict((s[0][0], s[1]) for s in H.chart_series(ps, paths, DAYS)[1:])
        self.assertEqual(colors, {"B": "#1baf7a", "C": "#eb6834"})     # B は A と重なった色のまま

    def test_recent_sales_stay_and_colors_do_not_change(self):
        a = self.sold("1111", 1, 3, "A")
        b = self.held("2222", 3, "B")
        days = DAYS[:23]                         # 売ってから 19 営業日 → まだ出す
        before = H.chart_series([a, b], {0: None, 1: None}, days)
        self.assertEqual([s[1] for s in before[1:]], ["#eb6834", "#1baf7a"])
        c = self.held("3333", 22, "C")
        after = H.chart_series([a, b, c], {0: None, 1: None, 2: None}, days)
        self.assertEqual(after[:3], before)      # あとから買っても先の色は変わらない
        self.assertEqual(after[3][1], "#eda100")


class SheetRequests(unittest.TestCase):
    HEADER = ["予測日", "コード", "銘柄名", "建値", "株数", "手仕舞い日", "手仕舞い値", "損益", "メモ"]

    def setUp(self):
        b = make_bars(days=DAYS[:10], h=with_value(10, 1000, {3: 1100}))
        self.a = pos()
        self.b = pos(code="6912", pick="2026-09-30", row=6)
        self.ps = [self.a, self.b]
        self.paths = {k: H.simulate(p, b, CAL) for k, p in enumerate(self.ps)}

    def test_colors(self):
        reqs = H.color_requests(0, self.ps, 9, cleared=[4])
        fmt = [(r["repeatCell"]["range"]["startRowIndex"], r["repeatCell"]["cell"]["userEnteredFormat"],
                r["repeatCell"]["fields"]) for r in reqs]
        self.assertEqual(fmt, [(1, {"backgroundColor": H.BLUE}, "userEnteredFormat.backgroundColor"),
                               (5, {"backgroundColor": H.RED}, "userEnteredFormat.backgroundColor"),
                               (3, {}, "userEnteredFormat.backgroundColor")])
        self.assertEqual(reqs[0]["repeatCell"]["range"]["endColumnIndex"], 9)

    def test_user_cells(self):
        cells = {c["range"]: c["values"][0][0]
                 for c in H.user_cell_updates(self.HEADER, self.ps, self.paths, cleared=[4])}
        self.assertEqual(cells["D2"], 1000.0)
        self.assertEqual(cells["E2"], 100)
        self.assertEqual(cells["F2"], "2026-09-29")
        self.assertAlmostEqual(cells["G2"], 1100.0)
        self.assertEqual(cells["H2"], 10000)
        self.assertEqual(cells["F6"], "")                       # まだ持っている
        for col in "DEFGH":
            self.assertEqual(cells[f"{col}4"], "")              # 取り消した行は空に
        self.assertNotIn("I2", cells)                           # メモには触らない

    def test_report_has_no_names_or_values(self):
        self.a.issue = "売りの日付が買った日より前"
        lines = H.report(3, 1, 2, self.ps, ["売りに合う買いが無い"], 9, "2026-10-07")
        text = "\n".join(lines)
        self.assertIn("受け皿 3件（復号できない 1）・取引の記録に足した 2件", text)
        self.assertIn(f"保有 2件（保有中 1 / {H.SOLD} 1）", text)
        self.assertIn("合わなかった行 2", text)
        for word in ("A社", "6912", "10000", "10,000", "1100", "1000"):
            self.assertNotIn(word, text)


# --------------------------------------------------------------------------- #
# シートとのやりとり（偽のシート）
# --------------------------------------------------------------------------- #

class FakeWorksheet:
    def __init__(self, title, values=None, sid=1, rows=100, cols=30):
        self.title, self.id = title, sid
        self.values = [list(r) for r in (values or [])]
        self.row_count, self.col_count = rows, cols
        self.updates, self.appends, self.cells = [], [], []

    def get_all_values(self):
        return [list(r) for r in self.values]

    def resize(self, rows=None, cols=None):
        self.row_count, self.col_count = rows or self.row_count, cols or self.col_count

    def update(self, values, range_name, value_input_option=None):
        self.updates.append((values, range_name, value_input_option))
        if range_name == "A1" and self.title == H.LOG_TAB:
            self.values[:1] = [list(values[0])]

    def append_rows(self, values, value_input_option=None, table_range=None):
        self.appends.append((values, value_input_option))
        self.values.extend([[str(v) for v in r] for r in values])

    def batch_update(self, cells, value_input_option=None):
        self.cells.extend(cells)


class FakeBook:
    def __init__(self, pred_values):
        self.tabs = {H.ES.SHEET_TITLE: FakeWorksheet(H.ES.SHEET_TITLE, pred_values, sid=0)}
        self.requests = []

    def worksheet(self, title):
        if title not in self.tabs:
            raise KeyError(title)
        return self.tabs[title]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = FakeWorksheet(title, sid=10 + len(self.tabs), rows=rows, cols=cols)
        return self.tabs[title]

    def fetch_sheet_metadata(self, params=None):
        return {"sheets": [{"properties": {"sheetId": ws.id}, "charts": [{"chartId": 7}]}
                           for ws in self.tabs.values()]}

    def batch_update(self, body):
        self.requests.extend(body["requests"])


@unittest.skipUnless(HAVE_CRYPTO, "cryptography を読み込めない")
class EndToEnd(unittest.TestCase):
    PRED = [["予測日", "コード", "銘柄名", "建値", "株数", "手仕舞い日", "手仕舞い値", "損益", "メモ"],
            ["2026-09-24", "6912", "A社", "", "", "", "", "", "x"],
            ["2026-09-24", "1111", "B社", "", "", "", "", "", ""],
            ["2026-09-28", "7713", "C社", "", "", "", "", "", ""]]

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        days = DAYS[:10]
        pd.concat([make_bars("69120", days, h=with_value(10, 1000, {3: 1100})),
                   make_bars("77130", days, o=[500] * 10, c=[500] * 9 + [540])]
                  ).to_parquet(os.path.join(self.dir, "bars_2026.parquet"), index=False)
        pd.DataFrame({"Date": [f"{d:%Y-%m-%d}" for d in DAYS], "HolDiv": "1"}).to_parquet(
            os.path.join(self.dir, "calendar.parquet"), index=False)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode("ascii")
        self.env = json.dumps({"private_key": pem})
        self.inbox = os.path.join(self.dir, "inbox.jsonl")
        pub = TI.public_pem(key)
        events = [TI.make_event("buy", "6912", pick_date="2026-09-24"),
                  TI.make_event("buy", "1111", pick_date="2026-09-24"),
                  TI.make_event("buy", "7713", pick_date="2026-09-28")]
        events.append(TI.make_event("cancel", ref=events[1]["id"][:8]))
        for e in events:
            TI.append(self.inbox, TI.encrypt(e, pub))
        self.old_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = self.env

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        if self.old_env is None:
            os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        else:
            os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = self.old_env

    def run_main(self, book, *extra):
        orig = H.ES.open_sheet
        H.ES.open_sheet = lambda sheet_id: book
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = H.main(["--sheet-id", "x", "--data-dir", self.dir, "--inbox", self.inbox,
                               *extra])
        finally:
            H.ES.open_sheet = orig
        return code, out.getvalue()

    def test_reports_become_rows_tabs_and_colors(self):
        book = FakeBook(self.PRED)
        code, out = self.run_main(book)
        self.assertEqual(code, 0)
        log = book.tabs[H.LOG_TAB]
        self.assertEqual(log.values[0], H.LOG_HEADER)
        self.assertEqual([r[2] for r in log.values[1:]], ["買い", "買い", "買い", "取消"])
        self.assertEqual(log.appends[0][1], "RAW")
        tab = book.tabs[H.TAB].updates[-1][0]
        self.assertEqual([r[0] for r in tab[4:6]], ["A社", "C社"])  # 取り消した B社 は無い
        pred = [r["repeatCell"] for r in book.requests
                if r.get("repeatCell", {}).get("range", {}).get("sheetId") == 0]
        self.assertEqual([(r["range"]["startRowIndex"] + 1, r["cell"]["userEnteredFormat"])
                          for r in pred],
                         [(2, {"backgroundColor": H.BLUE}), (4, {"backgroundColor": H.RED}),
                          (3, {})])
        cells = {c["range"]: c["values"][0][0] for c in book.tabs[H.ES.SHEET_TITLE].cells}
        self.assertEqual((cells["F2"], cells["H2"], cells["F4"], cells["H4"], cells["D3"]),
                         ("2026-09-29", 10000, "", 4000, ""))
        self.assertNotIn("I2", cells)
        for word in ("A社", "C社", "6912", "7713", "10000", "4000"):
            self.assertNotIn(word, out)
        self.assertIn("取引の記録に足した 4件", out)

        # 2回目は足さない（id で見分ける）。タブの行を手で直したら、その値を使う
        log.values[3][7] = "480"                                  # C社 の買値を 480 に
        code, out = self.run_main(book)
        self.assertEqual(len(log.appends), 1)
        self.assertIn("取引の記録に足した 0件", out)
        cells = {c["range"]: c["values"][0][0] for c in book.tabs[H.ES.SHEET_TITLE].cells}
        # 買値 480 の +10% は 528。最後の日の高値 540 で届くので、528 で売って青になる
        self.assertEqual((cells["D4"], cells["G4"], cells["H4"]), (480.0, 528.0, 4800))
        last = [r["repeatCell"] for r in book.requests
                if r.get("repeatCell", {}).get("range", {}).get("sheetId") == 0][-3:]
        self.assertEqual(last[1]["cell"]["userEnteredFormat"], {"backgroundColor": H.BLUE})

    def test_dry_run_writes_nothing(self):
        book = FakeBook(self.PRED)
        code, out = self.run_main(book, "--dry-run")
        self.assertEqual(code, 0)
        self.assertNotIn(H.LOG_TAB, book.tabs)
        self.assertNotIn(H.TAB, book.tabs)
        self.assertEqual(book.requests, [])
        self.assertIn("保有 2件", out)

    def test_without_the_key_nothing_is_read(self):
        os.environ.pop("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        book = FakeBook(self.PRED)
        code, out = self.run_main(book)
        self.assertIn("受け皿 0件", out)
        self.assertIn("取引の記録がありません", book.tabs[H.TAB].updates[-1][0][2][0])


class Workflow(unittest.TestCase):
    def test_runs_after_the_daily_prediction(self):
        path = os.path.join(ROOT, ".github", "workflows", "holdings.yml")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("workflows: ['Predict Breakouts']", text)
        self.assertIn("workflow_dispatch", text)
        self.assertNotIn("schedule:", text)
        self.assertNotIn("git add", text)                  # コミットしない
        self.assertNotIn("gh_release_upload", text)        # Release に書かない
        self.assertIn("research/holdings_sheet.py", text)
        self.assertIn("'bars_*' 'calendar*'", text)        # 日足とカレンダーだけ取り出す
        # export_sheets -> models -> tuning_multi が scikit-learn を読み込むので、予測と同じ一式を入れる
        self.assertIn("pip install -r research/requirements.txt", text)
        # 公開鍵を出すだけの起動（鍵を作り直したとき）。シートには書かない
        self.assertIn("options: [update, pubkey]", text)
        self.assertIn("python3 research/trades_inbox.py pubkey", text)
        self.assertIn("if: github.event_name != 'workflow_dispatch' || inputs.mode != 'pubkey'", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
