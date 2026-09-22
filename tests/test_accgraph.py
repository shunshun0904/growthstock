#!/usr/bin/env python3
"""
research/accgraph/ の単体テスト。

この研究ラインで壊れると致命的なのは次の3つなので、そこを合成データで固定する。

  1. 過去の期に「後から訂正された値」が混ざる（未来情報のリーク）
  2. 決算発表日より前の株価でエントリーしてしまう
  3. 株式分割をまたいでリターンが壊れる

  python3 tests/test_accgraph.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from accgraph import (  # noqa: E402
    backtest, baselines, build, eda, eda_report, edinet, labels as L, leakage,
    panel, schema, splits, synthetic,
)


# --------------------------------------------------------------------------- #
# 合成データの組み立て
# --------------------------------------------------------------------------- #

def fin_row(code, per_end, disc_date, fy_start, per_type, sales, **kw):
    row = {
        "Code": code, "CurPerEn": per_end, "DiscDate": disc_date,
        "DiscTime": "15:00", "CurFYSt": fy_start, "CurPerType": per_type,
        "Sales": float(sales), "OP": float(sales) * 0.1,
        "OdP": float(sales) * 0.11, "NP": float(sales) * 0.07,
        "TA": 1000.0, "Eq": 400.0, "ShEq": 380.0,
        "CashEq": np.nan, "CFO": np.nan, "CFI": np.nan, "CFF": np.nan,
        "FSales": np.nan, "FOP": np.nan, "FOdP": np.nan, "FNP": np.nan,
    }
    row.update(kw)
    return row


def daily_bars(code, start, n, open_px, close_px, split_at=None):
    days = pd.bdate_range(start, periods=n)
    factor = np.ones(n)
    if split_at is not None:
        factor[split_at:] = 2.0
    return pd.DataFrame({
        "Date": days, "Code": code,
        "AdjO": open_px, "AdjC": close_px,
        "AdjH": np.maximum(open_px, close_px), "AdjL": np.minimum(open_px, close_px),
        "AdjVo": np.full(n, 1e6),
        "O": np.asarray(open_px) * factor, "C": np.asarray(close_px) * factor,
        "H": np.asarray(open_px) * factor, "L": np.asarray(close_px) * factor,
        "Vo": 1e6 / factor, "Va": np.asarray(close_px) * factor * (1e6 / factor),
    })


def flat_index(days, code="0000", level=1000.0):
    return pd.DataFrame({"Date": days, "Code": code,
                         "O": level, "C": level, "H": level, "L": level})


# --------------------------------------------------------------------------- #

class TestSchema(unittest.TestCase):

    def test_schema_is_self_consistent(self):
        schema.validate()          # 壊れていれば例外
        self.assertEqual(len(schema.NODES), schema.N_NODES)
        self.assertEqual(len(schema.EDGES), schema.N_EDGES)

    def test_edge_index_shape(self):
        ei = schema.edge_index()
        self.assertEqual(len(ei), 2)
        self.assertEqual(len(ei[0]), schema.N_EDGES)
        self.assertTrue(all(0 <= i < schema.N_NODES for i in ei[0] + ei[1]))

    def test_constants_match_names(self):
        nc = schema.node_constants()
        self.assertEqual(len(nc), schema.N_NODES)
        self.assertTrue(all(len(r) == len(schema.NODE_CONSTANTS) for r in nc))
        ec = schema.edge_constants()
        self.assertEqual(len(ec), schema.N_EDGES)
        self.assertTrue(all(sum(r) == 1.0 for r in ec), "エッジ種別は1つだけ立つ")

    def test_derived_nodes_are_identities_not_estimates(self):
        """
        計算値ノードは必ず開示値の足し引きで決まること。
        推定式が紛れ込むと「取れないものを作った」ことになる。
        """
        for n in schema.NODES:
            if n.source != "derived":
                continue
            plus, minus = n.derive
            self.assertTrue(plus, f"{n.id}: 加算項が空")
            known = (schema.CUMULATIVE_FIELDS + schema.STOCK_FIELDS
                     if n.source_table == "fins"
                     else schema.EDINET_FLOW_FIELDS + schema.EDINET_STOCK_FIELDS)
            for f in list(plus) + list(minus):
                self.assertIn(f, known, f"{n.id} ({n.source_table})")


class TestPointInTime(unittest.TestCase):
    """訂正開示が過去の期に遡って混ざらないこと。"""

    def setUp(self):
        rows = [
            # FY2020（4月開始）
            fin_row("10000", "2020-06-30", "2020-08-10", "2020-04-01", "1Q", 100),
            fin_row("10000", "2020-09-30", "2020-11-10", "2020-04-01", "2Q", 250),
            fin_row("10000", "2020-12-31", "2021-02-10", "2020-04-01", "3Q", 400),
            fin_row("10000", "2021-03-31", "2021-05-10", "2020-04-01", "FY", 600),
            # 1Q を1年後に訂正（100 -> 120）
            fin_row("10000", "2020-06-30", "2021-06-10", "2020-04-01", "1Q", 120),
            # FY2021
            fin_row("10000", "2021-06-30", "2021-08-10", "2021-04-01", "1Q", 110),
        ]
        self.fins = pd.DataFrame(rows)
        self.versions = panel.prepare_versions(self.fins)
        self.periods = panel.period_slots(self.versions)
        self.anchors = panel.anchors(self.versions, self.periods)

    def test_anchor_is_first_disclosure_only(self):
        """訂正開示はアンカーにならない（同じ決算が2サンプルになると二重計上）。"""
        self.assertEqual(len(self.anchors), 5)
        first_q1 = self.anchors[self.anchors["per_end"] == pd.Timestamp("2020-06-30")]
        self.assertEqual(len(first_q1), 1)
        self.assertEqual(first_q1["disc_date"].iloc[0], pd.Timestamp("2020-08-10"))

    def test_history_uses_the_version_known_at_the_time(self):
        mats = panel.build_asof_matrices(self.anchors, self.versions, self.periods)
        cum = mats["cum_Sales"]
        a = self.anchors.reset_index(drop=True)

        # 通期開示（2021-05-10）の時点では訂正前の 100 が見えている
        fy = int(a.index[a["per_end"] == pd.Timestamp("2021-03-31")][0])
        self.assertAlmostEqual(cum[fy, 3], 100.0)

        # 翌年1Q（2021-08-10）の時点では訂正後の 120 が見えている
        nxt = int(a.index[a["per_end"] == pd.Timestamp("2021-06-30")][0])
        self.assertAlmostEqual(cum[nxt, 4], 120.0)

    def test_cumulative_is_expanded_to_single_quarter(self):
        mats = panel.build_asof_matrices(self.anchors, self.versions, self.periods)
        q = mats["q_Sales"]
        a = self.anchors.reset_index(drop=True)
        fy = int(a.index[a["per_end"] == pd.Timestamp("2021-03-31")][0])
        # lag0 = FY(600-400), lag1 = 3Q(400-250), lag2 = 2Q(250-100), lag3 = 1Q(100)
        self.assertAlmostEqual(q[fy, 0], 200.0)
        self.assertAlmostEqual(q[fy, 1], 150.0)
        self.assertAlmostEqual(q[fy, 2], 150.0)
        self.assertAlmostEqual(q[fy, 3], 100.0)

    def test_first_quarter_of_a_new_year_is_not_differenced(self):
        """年度をまたぐところで前年度の累計を引かないこと。"""
        mats = panel.build_asof_matrices(self.anchors, self.versions, self.periods)
        a = self.anchors.reset_index(drop=True)
        nxt = int(a.index[a["per_end"] == pd.Timestamp("2021-06-30")][0])
        self.assertAlmostEqual(mats["q_Sales"][nxt, 0], 110.0)

    def test_missing_history_stays_missing(self):
        """履歴が足りないラグは欠測。0 で埋めると『売上ゼロ』になる。"""
        mats = panel.build_asof_matrices(self.anchors, self.versions, self.periods)
        a = self.anchors.reset_index(drop=True)
        q1 = int(a.index[a["per_end"] == pd.Timestamp("2020-06-30")][0])
        self.assertTrue(np.isnan(mats["cum_Sales"][q1, 1]))


class TestLabels(unittest.TestCase):

    def _fixture(self, split_at=None, n=80):
        days = pd.bdate_range("2020-01-01", periods=n)
        # 始値 100 固定、終値は 1日 +1 の直線。リターンを手計算できる形にする
        open_px = np.full(n, 100.0)
        close_px = 100.0 + np.arange(n, dtype=float)
        bars = daily_bars("10000", "2020-01-01", n, open_px, close_px,
                          split_at=split_at)
        indices = flat_index(days)
        fins = pd.DataFrame([
            fin_row("10000", "2020-01-31", str(days[10].date()),
                    "2019-04-01", "3Q", 100),
        ])
        versions = panel.prepare_versions(fins)
        periods = panel.period_slots(versions)
        anchors = panel.anchors(versions, periods)
        return anchors, bars, indices, days

    def test_entry_is_the_next_trading_day_open(self):
        anchors, bars, indices, days = self._fixture()
        cfg = L.LabelConfig(horizons=(5,), primary_horizon=5)
        out = L.build_labels(anchors, bars, indices, None, None, cfg)
        self.assertEqual(out["entry_date"].iloc[0], days[11])
        self.assertAlmostEqual(out["entry_price"].iloc[0], 100.0)

    def test_return_is_open_to_close(self):
        anchors, bars, indices, days = self._fixture()
        cfg = L.LabelConfig(horizons=(5,), primary_horizon=5)
        out = L.build_labels(anchors, bars, indices, None, None, cfg)
        # エントリー = 11日目の始値 100、エグジット = 16日目の終値 116
        self.assertAlmostEqual(out["ret_5d"].iloc[0], 116.0 / 100.0 - 1.0)

    def test_split_does_not_break_the_return(self):
        """分割日をまたいでも、調整後価格で測るのでリターンは変わらない。"""
        cfg = L.LabelConfig(horizons=(5,), primary_horizon=5)
        a1, b1, i1, _ = self._fixture(split_at=None)
        a2, b2, i2, _ = self._fixture(split_at=13)
        r1 = L.build_labels(a1, b1, i1, None, None, cfg)["ret_5d"].iloc[0]
        r2 = L.build_labels(a2, b2, i2, None, None, cfg)["ret_5d"].iloc[0]
        self.assertAlmostEqual(r1, r2)

    def test_excess_is_benchmark_deducted(self):
        anchors, bars, indices, days = self._fixture()
        # ベンチマークを +10% 一本調子にする
        idx = indices.copy()
        idx["O"] = 1000.0
        idx["C"] = 1000.0 * (1.0 + 0.10 * np.arange(len(idx)) / len(idx))
        cfg = L.LabelConfig(horizons=(5,), primary_horizon=5)
        out = L.build_labels(anchors, bars, idx, None, None, cfg)
        self.assertAlmostEqual(
            out["excess_topix_5d"].iloc[0],
            out["ret_5d"].iloc[0] - out["topix_5d"].iloc[0])
        self.assertFalse(np.isnan(out["topix_5d"].iloc[0]))

    def test_class_boundaries(self):
        cfg = L.LabelConfig()
        got = L.classify(np.array([0.021, 0.02, 0.0, -0.02, -0.021, np.nan]), cfg)
        self.assertEqual(list(got), [2, 1, 1, 1, 0, -1])

    def test_delisted_stock_gets_no_label(self):
        """
        エグジット価格が取れない開示はラベルが付かない。
        黙って消すと生存者バイアスになるので、必ず欠測として残ること。
        """
        anchors, bars, indices, days = self._fixture()
        bars = bars[bars["Date"] <= days[13]]      # エントリー直後に上場廃止
        cfg = L.LabelConfig(horizons=(5,), primary_horizon=5)
        out = L.build_labels(anchors, bars, indices, None, None, cfg)
        self.assertTrue(np.isnan(out["ret_5d"].iloc[0]))
        self.assertEqual(int(out["y_topix_5d"].iloc[0]), -1)


class TestSplits(unittest.TestCase):

    def _meta(self, n=400):
        d = pd.bdate_range("2016-01-01", periods=n, freq="5B")
        return pd.DataFrame({
            "entry_date": d,
            "label_ready_date": d + pd.tseries.offsets.BDay(20),
        })

    def test_purge_removes_samples_whose_label_spills_into_test(self):
        meta = self._meta()
        folds, tr, te = splits.walk_forward(
            meta, min_train_months=24, test_months=12, step_months=12,
            embargo_trading_days=0, min_test_rows=5)
        self.assertGreater(len(folds), 0)
        for f, m in zip(folds, tr):
            ready = pd.to_datetime(meta["label_ready_date"])[m]
            self.assertTrue((ready < pd.Timestamp(f.test_start)).all(),
                            "ラベル確定がテスト期間に食い込む訓練サンプルが残っている")

    def test_embargo_adds_a_buffer(self):
        meta = self._meta()
        _, tr0, _ = splits.walk_forward(meta, min_train_months=24, test_months=12,
                                        step_months=12, embargo_trading_days=0,
                                        min_test_rows=5)
        _, tr1, _ = splits.walk_forward(meta, min_train_months=24, test_months=12,
                                        step_months=12, embargo_trading_days=60,
                                        min_test_rows=5)
        self.assertLessEqual(int(tr1[0].sum()), int(tr0[0].sum()))

    def test_test_windows_do_not_overlap(self):
        meta = self._meta()
        folds, _, te = splits.walk_forward(meta, min_train_months=24,
                                           test_months=12, step_months=12,
                                           embargo_trading_days=20,
                                           min_test_rows=5)
        for a, b in zip(te, te[1:]):
            self.assertEqual(int((a & b).sum()), 0)

    def test_train_is_always_before_test(self):
        meta = self._meta()
        folds, tr, te = splits.walk_forward(meta, min_train_months=24,
                                            test_months=12, step_months=12,
                                            embargo_trading_days=20,
                                            min_test_rows=5)
        d = pd.to_datetime(meta["entry_date"])
        for a, b in zip(tr, te):
            self.assertLess(d[a].max(), d[b].min())


class TestLeakage(unittest.TestCase):

    def _meta(self):
        d = pd.bdate_range("2020-01-01", periods=10)
        return pd.DataFrame({
            "Code": [f"{i}0000" for i in range(10)],
            "per_end": pd.Timestamp("2019-12-31"),
            "disc_date": d,
            "entry_date": d + pd.tseries.offsets.BDay(1),
            "label_ready_date": d + pd.tseries.offsets.BDay(21),
            "excess_topix_20d": np.linspace(-0.1, 0.1, 10),
        })

    def test_detects_entry_before_disclosure(self):
        meta = self._meta()
        meta.loc[0, "entry_date"] = meta.loc[0, "disc_date"]
        res = leakage.check_labels(meta, horizons=(20,))
        self.assertFalse(res[0]["ok"])

    def test_detects_duplicate_periods(self):
        meta = self._meta()
        meta.loc[1, "Code"] = meta.loc[0, "Code"]
        res = leakage.check_labels(meta, horizons=(20,))
        dup = [r for r in res if "会計期間" in r["check"]][0]
        self.assertFalse(dup["ok"])

    def test_detects_values_in_absent_periods(self):
        meta = self._meta()
        nf = np.zeros((10, 8, schema.N_NODES, len(schema.NODE_FEATURES)))
        pm = np.ones((10, 8), dtype=bool)
        pm[0, 7] = False
        nf[0, 7, 0, 0] = 1.23          # 存在しない四半期に値が残っている
        res = leakage.check_features(nf, pm, meta)
        absent = [r for r in res if "存在しない四半期" in r["check"]][0]
        self.assertFalse(absent["ok"])

    def test_passes_on_clean_data(self):
        meta = self._meta()
        nf = np.zeros((10, 8, schema.N_NODES, len(schema.NODE_FEATURES)))
        pm = np.ones((10, 8), dtype=bool)
        res = leakage.run_all(meta, node_feat=nf, period_mask=pm)
        self.assertTrue(all(r["ok"] for r in res))


class TestBacktest(unittest.TestCase):

    def _trades(self, dates, rets):
        meta = pd.DataFrame({
            "entry_date": pd.to_datetime(dates),
            "Code": [f"{i}0000" for i in range(len(dates))],
            "excess_topix_20d": rets,
            "label_ready_date": pd.to_datetime(dates) + pd.tseries.offsets.BDay(20),
        })
        proba = np.tile([0.1, 0.2, 0.7], (len(dates), 1))
        return meta, proba

    def test_cost_is_subtracted_once_per_trade(self):
        meta, proba = self._trades(["2020-01-06"] * 3, [0.05, 0.05, 0.05])
        r = backtest.run(meta, proba, np.array([0, 1, 2]),
                         excess_col="excess_topix_20d", horizon=20,
                         max_horizon=20, rule="predicted_up", cost_bps=30.0)
        self.assertAlmostEqual(r.mean_net_per_trade, 0.05 - 0.003)
        self.assertEqual(r.n_trades, 3)

    def test_overlapping_entries_collapse_into_one_cohort(self):
        """保有中に出たエントリーは同じコホートに数えない（二重計上を防ぐ）。"""
        meta, proba = self._trades(
            ["2020-01-06", "2020-01-07", "2020-01-08"], [0.05, 0.05, 0.05])
        r = backtest.run(meta, proba, np.array([0, 1, 2]),
                         excess_col="excess_topix_20d", horizon=20,
                         max_horizon=20, rule="predicted_up", cost_bps=0.0)
        self.assertEqual(r.n_cohorts, 1)

    def test_max_drawdown(self):
        eq = np.array([1.0, 1.2, 0.9, 1.1])
        self.assertAlmostEqual(backtest._max_drawdown(eq), 0.9 / 1.2 - 1.0)

    def test_all_rule_buys_everything(self):
        meta, proba = self._trades(["2020-01-06", "2020-06-01"], [0.05, -0.05])
        r = backtest.run(meta, proba, np.array([0, 1, 2]),
                         excess_col="excess_topix_20d", horizon=20,
                         max_horizon=20, rule="all", cost_bps=0.0)
        self.assertEqual(r.n_trades, 2)
        self.assertAlmostEqual(r.mean_net_per_trade, 0.0)


class TestEndToEnd(unittest.TestCase):
    """合成データで構築から評価まで一度通す。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="accgraph_test_")
        cls.raw = os.path.join(cls.tmp, "raw")
        cls.out = os.path.join(cls.tmp, "out")
        synthetic.write_all(cls.raw, n_codes=8, start_year=2018, n_years=5)
        cls.meta = build.build(data_dir=cls.raw, out_dir=cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dataset_round_trips(self):
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        self.assertEqual(len(meta), len(self.meta))
        self.assertEqual(nf.shape[1:], (panel.SEQ_LEN, schema.N_NODES,
                                        len(schema.NODE_FEATURES)))
        self.assertEqual(ef.shape[1:], (panel.SEQ_LEN, schema.N_EDGES,
                                        len(schema.EDGE_FEATURES)))
        self.assertEqual(pm.shape, (len(meta), panel.SEQ_LEN))

    def test_leak_checks_pass(self):
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        leakage.run_all(meta, node_feat=nf, period_mask=pm)   # 失敗すれば例外

    def test_cash_flow_is_masked_when_not_disclosed(self):
        """
        1Q / 3Q は CF を開示しない。欠測が 0 として学習されず、
        is_missing が立っていること。
        """
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        j = schema.NODE_INDEX["cfo"]
        f = schema.NODE_FEATURES.index("is_missing")
        v = schema.NODE_FEATURES.index("scaled")
        q = meta["quarter"].to_numpy()
        odd = np.isin(q, [1, 3])
        self.assertTrue((nf[odd, 0, j, f] == 1.0).all(), "1Q/3Q の営業CFが欠測扱いでない")
        self.assertTrue((nf[odd, 0, j, v] == 0.0).all(), "欠測なのに値が入っている")

    def test_half_year_cash_flow_survives_the_expansion(self):
        """
        2Q / 通期の CF が欠測にならないこと。

        前四半期との差だけで累計を展開すると、1Q・3Q の CF が無いせいで
        2Q も通期も差が取れず、CF ノードが全期間まるごと消える。
        実際に一度そうなったので、ここで固定する。
        """
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        j = schema.NODE_INDEX["cfo"]
        f = schema.NODE_FEATURES.index("is_missing")
        sp = schema.NODE_FEATURES.index("span")
        even = np.isin(meta["quarter"].to_numpy(), [2, 4])
        self.assertTrue(even.any(), "2Q/通期のサンプルが無い")
        self.assertTrue((nf[even, 0, j, f] == 0.0).all(), "2Q/通期の営業CFが欠測扱い")
        # 1Q(3Q)が無いぶん、2期ぶんをまとめた値になっている
        self.assertTrue((nf[even, 0, j, sp] == 2.0).all(),
                        "半期ぶんであることが span に出ていない")
        # 売上高は毎期開示されるので1期ぶんのまま
        js = schema.NODE_INDEX["sales"]
        self.assertTrue((nf[even, 0, js, sp] == 1.0).all())

    def test_flatten_dimensions_match_names(self):
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        for kind in ("latest", "nodes", "seq"):
            X, names = baselines.flatten(nf, ef, pm, meta, kind=kind)
            self.assertEqual(X.shape[1], len(names))
            self.assertEqual(len(X), len(meta))
            self.assertFalse(np.isnan(X).any())

    def test_derived_node_equals_the_identity(self):
        """負債 = 総資産 - 純資産 が、実際にその値になっていること。"""
        versions = panel.prepare_versions(build.load_parts("fins", self.raw))
        periods = panel.period_slots(versions)
        anchors = panel.anchors(versions, periods)
        mats = panel.build_asof_matrices(anchors, versions, periods)
        # EDINET を結合していないので、明細ノードは欠測になる。
        # J-Quants だけで決まる恒等式を確かめる
        amount = build.node_amounts(mats)
        got = amount[:, :, schema.NODE_INDEX["liabilities"]]
        want = mats["s_TA"] - mats["s_Eq"]
        ok = ~np.isnan(want)
        np.testing.assert_allclose(got[ok], want[ok])


class TestEdinet(unittest.TestCase):
    """EDINET の明細を四半期グラフに混ぜるところ。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="accgraph_ed_")
        cls.raw = os.path.join(cls.tmp, "raw")
        cls.out = os.path.join(cls.tmp, "out")
        synthetic.write_all(cls.raw, n_codes=8, start_year=2018, n_years=5)
        cls.meta = build.build(data_dir=cls.raw, out_dir=cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_correction_does_not_leak_backwards(self):
        """
        訂正報告書が、訂正前の時点のグラフに混ざらないこと。
        年次パネルは各年度の最初の提出だけを採る。
        """
        fin = edinet.load(self.raw)
        panel_ = edinet.annual_panel(fin)
        dup = panel_.duplicated(["Code", "fiscal_year"]).sum()
        self.assertEqual(int(dup), 0, "同じ年度が2行ある（訂正を落とせていない）")
        # 合成データは2年目を1年後に訂正し、税引前利益を1.4倍にしてある。
        # 採られているのは訂正前の値
        raw = fin.sort_values("submit_date")
        first = raw.drop_duplicates(["jq_code", "fiscal_year"], keep="first")
        merged = panel_.merge(first, left_on=["Code", "fiscal_year"],
                              right_on=["jq_code", "fiscal_year"])
        np.testing.assert_allclose(
            merged["profit_before_tax_x"].to_numpy(),
            merged["profit_before_tax_y"].to_numpy())

    def test_available_the_day_after_submission(self):
        """提出日の翌日から使えること。当日に使うと場中の提出で先読みになる。"""
        fin = edinet.load(self.raw)
        p = edinet.annual_panel(fin)
        gap = (pd.to_datetime(p["avail_date"])
               - pd.to_datetime(fin.sort_values("submit_date")
                                .drop_duplicates(["jq_code", "fiscal_year"],
                                                 keep="first")["submit_date"]
                                ).dt.normalize().reset_index(drop=True))
        self.assertTrue((gap.dt.days == 1).all())

    def test_annual_flow_is_divided_into_quarters(self):
        """
        年次のフローは1四半期あたりに直され、span に4が入ること。
        直さないと、四半期の売上高と年次の減価償却費を同じ土俵で比べてしまう。
        """
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        sp = schema.NODE_FEATURES.index("span")
        mi = schema.NODE_FEATURES.index("is_missing")
        for nid in ("depreciation", "capex", "cost_of_sales"):
            j = schema.NODE_INDEX[nid]
            have = nf[:, 0, j, mi] == 0
            self.assertTrue(have.any(), f"{nid} が1件も引けていない")
            self.assertTrue((nf[have, 0, j, sp] == 4.0).all(), nid)
        # 期末残高は割らない
        j = schema.NODE_INDEX["inventories"]
        have = nf[:, 0, j, mi] == 0
        self.assertTrue((nf[have, 0, j, sp] == 1.0).all())

    def test_age_is_zero_for_quarterly_nodes(self):
        """四半期のノードは当期そのものなので、何年前かは0。"""
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        ag = schema.NODE_FEATURES.index("age_years")
        mi = schema.NODE_FEATURES.index("is_missing")
        j = schema.NODE_INDEX["sales"]
        have = nf[:, 0, j, mi] == 0
        self.assertTrue((nf[have, 0, j, ag] == 0.0).all())
        # EDINET のノードは0〜1年ぶん古い（年1回の開示なので）
        j = schema.NODE_INDEX["pretax_profit"]
        have = nf[:, 0, j, mi] == 0
        age = nf[have, 0, j, ag]
        self.assertTrue(have.any())
        self.assertTrue(((age >= 0) & (age <= edinet.STALE_DAYS / 365.25)).all(),
                        f"age_years が範囲外: {age.min()}〜{age.max()}")

    def test_missing_edinet_leaves_the_coarse_graph_intact(self):
        """
        EDINET が無い会社でも、J-Quants だけの粗いグラフは成立すること。
        明細が取れない会社を丸ごと落とすと、母集団が偏る。
        """
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        mi = schema.NODE_FEATURES.index("is_missing")
        j_ed = schema.NODE_INDEX["pretax_profit"]
        j_jq = schema.NODE_INDEX["sales"]
        no_edinet = nf[:, 0, j_ed, mi] == 1
        self.assertTrue(no_edinet.any(), "EDINET が欠測のサンプルが無い")
        self.assertTrue((nf[no_edinet, 0, j_jq, mi] == 0).all(),
                        "EDINET が無いだけで売上高まで欠測になっている")

    def test_working_capital_is_the_identity(self):
        """運転資本 = 棚卸資産 + 売上債権 − 仕入債務 が実際にその値になること。"""
        meta, nf, ef, pm = build.load(self.out, liquid_only=False)
        sc = schema.NODE_FEATURES.index("to_assets")
        mi = schema.NODE_FEATURES.index("is_missing")
        idx = {k: schema.NODE_INDEX[k] for k in
               ("working_capital", "inventories", "trade_receivables",
                "trade_payables")}
        ok = np.all([nf[:, 0, j, mi] == 0 for j in idx.values()], axis=0)
        self.assertTrue(ok.any())
        want = (nf[ok, 0, idx["inventories"], sc]
                + nf[ok, 0, idx["trade_receivables"], sc]
                - nf[ok, 0, idx["trade_payables"], sc])
        np.testing.assert_allclose(nf[ok, 0, idx["working_capital"], sc], want,
                                   rtol=1e-5, atol=1e-6)


    def test_chunking_does_not_change_the_result(self):
        """
        メモリのためにアンカーを分けて計算しても、結果が変わらないこと。
        14万件を一度に回すと中間配列が7GBを超えてランナーが落ちるので
        分けているが、分け方で値が変わってはいけない。
        """
        fins = build.load_parts("fins", self.raw)
        v = panel.prepare_versions(fins)
        pe = panel.period_slots(v)
        a = panel.anchors(v, pe)
        m = panel.build_asof_matrices(a, v, pe)
        m.update(edinet.asof_matrices(
            a, m["disc_date_days"],
            edinet.annual_panel(edinet.load(self.raw))))
        whole = build.features_in_chunks(m, panel.SEQ_LEN, chunk_rows=10 ** 9)
        split = build.features_in_chunks(m, panel.SEQ_LEN, chunk_rows=37)
        for w, p_ in zip(whole, split):
            np.testing.assert_array_equal(w, p_)


class TestRankWithinDate(unittest.TestCase):

    def test_ranks_are_computed_inside_each_date(self):
        X = np.array([[1.0], [3.0], [2.0], [10.0], [20.0]])
        d = pd.Series(pd.to_datetime(
            ["2020-01-01"] * 3 + ["2020-01-02"] * 2))
        r = baselines.rank_within_date(X, d)
        # 1日目: 1 < 2 < 3 -> 0, 0.5, 1
        np.testing.assert_allclose(r[:3, 0], [0.0, 1.0, 0.5])
        # 2日目: 別の日なので桁の大きさは効かない
        np.testing.assert_allclose(r[3:, 0], [0.0, 1.0])

    def test_single_row_date_is_neutral(self):
        X = np.array([[5.0], [7.0]])
        d = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
        r = baselines.rank_within_date(X, d)
        np.testing.assert_allclose(r[:, 0], [0.5, 0.5])

    def test_rank_sets_have_the_same_width(self):
        n = 40
        nf = np.random.default_rng(0).normal(
            size=(n, panel.SEQ_LEN, schema.N_NODES, len(schema.NODE_FEATURES)))
        ef = np.zeros((n, panel.SEQ_LEN, schema.N_EDGES,
                       len(schema.EDGE_FEATURES)))
        pm = np.ones((n, panel.SEQ_LEN), dtype=bool)
        meta = pd.DataFrame({
            "quarter": np.tile([1, 2, 3, 4], n // 4),
            "turnover_ma20": np.linspace(1, 50, n),
            "entry_date": pd.to_datetime(
                np.repeat(pd.date_range("2020-01-01", periods=n // 4), 4)),
        })
        a, na = baselines.flatten(nf, ef, pm, meta, kind="seq")
        b, nb = baselines.flatten(nf, ef, pm, meta, kind="seq_rank")
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(len(na), len(nb))
        self.assertTrue(all(x.startswith("rank(") for x in nb))
        self.assertTrue(((b >= 0) & (b <= 1)).all())


class TestEda(unittest.TestCase):
    """EDA の集計と組版。合成データで一度通し、結果の形を固定する。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="accgraph_eda_")
        raw = os.path.join(cls.tmp, "raw")
        cls.out = os.path.join(cls.tmp, "out")
        synthetic.write_all(raw, n_codes=8, start_year=2018, n_years=5)
        build.build(data_dir=raw, out_dir=cls.out)
        cls.d = eda.run(cls.out, liquid_only=False, ic_min_n=50)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_has_every_section(self):
        for k in ("availability", "label", "features", "ic", "structure"):
            self.assertIn(k, self.d, f"{k} が集計に無い")

    def test_json_is_serialisable(self):
        """
        NaN や numpy 型が混ざっていると json.dump は通っても読み手が壊れる。
        allow_nan=False で「素の JSON として読めるか」を確かめる。
        """
        json.dumps(self.d, allow_nan=False)

    def test_cash_flow_coverage_shows_the_half_year_pattern(self):
        """
        EDA が CF の半期開示を見えるようにしていること。
        ここが 100% に見えたら、欠測を値と取り違えている。
        """
        nodes = {n["id"]: n for n in self.d["availability"]["nodes"]}
        cfo = nodes["cfo"]
        self.assertLess(cfo["by_quarter"]["1"], 1.0)
        self.assertLess(cfo["by_quarter"]["3"], 1.0)
        self.assertGreater(cfo["by_quarter"]["2"], 99.0)
        self.assertGreater(cfo["by_quarter"]["4"], 99.0)
        self.assertAlmostEqual(cfo["median_span"], 2.0)
        # 売上高は毎期開示されるので、比較対象として全期そろう
        self.assertGreater(nodes["sales"]["overall"], 99.0)

    def test_class_shares_sum_to_100(self):
        for h, blk in self.d["label"]["horizons"].items():
            for bench in ("topix", "sector"):
                if bench not in blk:
                    continue
                total = sum(blk[bench]["share"].values())
                self.assertAlmostEqual(total, 100.0, places=6,
                                       msg=f"{h}日 {bench} の構成比が100%でない")

    def test_report_renders(self):
        html = eda_report.build(self.d)
        self.assertIn("<title>", html)
        # <html>/<body> を付けない（Artifact としてそのまま出せる形）
        self.assertNotIn("<body", html)
        for marker in ("何が取れて、何が取れないか", "目的変数の分布と偏り",
                       "特徴量の診断", "グラフ構造の診断"):
            self.assertIn(marker, html)
        # 色だけに意味を持たせない: 凡例と表が必ず入っている
        self.assertIn('class="key"', html)
        self.assertIn("表で見る", html)

    def test_report_survives_an_empty_aggregate(self):
        """
        集計が空でも組版で落ちないこと。データが足りない時期に
        ワークフローが止まると、原因の切り分けができなくなる。
        """
        thin = {"n_samples": 0, "n_codes": 0, "period": ["-", "-"],
                "availability": {"nodes": [], "edges": [], "seq_len_hist": {},
                                 "full_seq_pct": float("nan"),
                                 "samples_by_year": {}},
                "label": {"horizons": {}}, "features": {"stats": {},
                "constant": [], "redundant": [], "hist": {}, "n_columns": 0},
                "ic": {}, "structure": {}}
        html = eda_report.build(thin)
        self.assertIn("<title>", html)


class TestGeneratedDoc(unittest.TestCase):

    def test_doc_matches_the_schema(self):
        """
        docs/ACCOUNTING_GRAPH.md はスキーマから生成される。
        手で直すと次の生成で消えるうえ、表の定義と実際の計算がずれる。
        """
        from accgraph import docgen
        path = os.path.join(ROOT, "docs", "ACCOUNTING_GRAPH.md")
        self.assertTrue(os.path.exists(path), "docs/ACCOUNTING_GRAPH.md が無い")
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(
                fh.read(), docgen.render(),
                "スキーマと文書がずれています。"
                "python3 research/accgraph/docgen.py で作り直してください")


if __name__ == "__main__":
    unittest.main(verbosity=2)
