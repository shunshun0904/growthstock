#!/usr/bin/env python3
"""
research/build_dataset.py のラベル生成ロジックの単体テスト。

株価予測で最も起きやすい事故は「未来を見てしまう」ことなので、
ブレイクアウト判定とラベル窓の境界を合成データで固定して検証する。

  python3 tests/test_dataset.py
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from build_dataset import (  # noqa: E402
    clip_divergent, mark_new_highs, attach_rise_label,
    HOLD_DAYS, HORIZON_END, HORIZON_START, HIGH_WINDOW, LabelConfig,
    _lag_available, add_cross_sectional_ranks, attach_labels, breakout_flags,
    price_panel, quarterize_panel,
)


def make_bars(closes, vols=None, code="00010", start="2020-01-01"):
    """1銘柄ぶんの日次バーを作る（High=Close, カレンダーは連番の営業日とみなす）。"""
    n = len(closes)
    vols = vols if vols is not None else [100000] * n
    d0 = dt.date.fromisoformat(start)
    return pd.DataFrame({
        "Date": [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)],
        "Code": [code] * n,
        "O": closes, "H": closes, "L": closes, "C": closes,
        "Vo": vols, "Va": [c * v for c, v in zip(closes, vols)],
        "AdjO": closes, "AdjH": closes, "AdjL": closes, "AdjC": closes, "AdjVo": vols,
    })


def flat_then(base_len, base_price, tail_closes, tail_vols=None, base_vol=100000):
    closes = [base_price] * base_len + list(tail_closes)
    vols = [base_vol] * base_len + list(tail_vols if tail_vols else [base_vol] * len(tail_closes))
    return closes, vols


#: 基本挙動を確認するための「定着条件なし」設定。
#: 既定 (DEFAULT_LABEL) は採用定義 E で sustain_days=60 が入っているため、
#: 高値更新・出来高・hold だけを検証したいテストではこちらを使う。
BASE = LabelConfig(sustain_days=0)


class TestBreakoutDetection(unittest.TestCase):
    def _flags(self, closes, vols):
        df = price_panel(make_bars(closes, vols), BASE)
        return breakout_flags(df, BASE)


    def test_detects_valid_breakout(self):
        """高値更新 + 出来高1.5倍以上 + 20日定着 の3条件が揃えば True。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] + [1200] * (HOLD_DAYS + 5),
                                 [300000] + [100000] * (HOLD_DAYS + 5))
        df = self._flags(closes, vols)
        bo = df.loc[df["is_breakout"] == True]  # noqa: E712
        self.assertEqual(len(bo), 1, "ブレイク日はちょうど1日であるべき")
        self.assertEqual(bo.iloc[0]["close"], 1200)

    def test_rejects_breakout_without_volume(self):
        """出来高が伴わない高値更新は除外する（だまし対策）。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] * (HOLD_DAYS + 6),
                                 [100000] * (HOLD_DAYS + 6))
        df = self._flags(closes, vols)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_rejects_breakout_that_collapses(self):
        """ブレイク後に-8%超下落したら正例にしない（定着条件）。"""
        n = HIGH_WINDOW + 5
        tail = [1200] + [1000] * (HOLD_DAYS + 5)   # 翌日に -16%
        tvol = [300000] + [100000] * (HOLD_DAYS + 5)
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_hold_boundary_is_exactly_8_percent(self):
        """-8%ちょうどは許容、-8%超は不可。"""
        n = HIGH_WINDOW + 5
        for drop, expected in [(0.92, 1), (0.919, 0)]:
            tail = [1200] + [1200 * drop] * (HOLD_DAYS + 5)
            tvol = [300000] + [100000] * (HOLD_DAYS + 5)
            closes, vols = flat_then(n, 1000, tail, tvol)
            df = self._flags(closes, vols)
            self.assertEqual(int((df["is_breakout"] == True).sum()), expected,  # noqa: E712
                             f"drop={drop} の判定が期待と違う")

    def test_undetermined_at_series_end_is_nan(self):
        """定着を評価できない末尾は False ではなく NaN。"""
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200], [300000])
        df = self._flags(closes, vols)
        self.assertTrue(pd.isna(df.iloc[-1]["is_breakout"]),
                        "将来データが無い日は判定不能(NaN)であるべき")


class TestSustainCondition(unittest.TestCase):
    """定着条件 (sustain): 一定期間後も水準を保っているか。"""

    def _flags(self, closes, vols, cfg):
        return breakout_flags(price_panel(make_bars(closes, vols), cfg), cfg)

    def test_sustain_rejects_fade(self):
        """ブレイク後にじりじり戻して水準を割ったら正例にしない。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        # ブレイク直後は下げないので hold は通るが、60日後には水準を割る
        tail = [1200] + [1180] * 30 + [1100] * 40
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols, cfg)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 0)  # noqa: E712

    def test_sustain_accepts_holding_level(self):
        """60日後も水準を保っていれば正例。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        tail = [1200] + [1250] * 70
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        df = self._flags(closes, vols, cfg)
        self.assertEqual(int((df["is_breakout"] == True).sum()), 1)  # noqa: E712

    def test_sustain_is_stricter_than_hold_alone(self):
        """同じ系列で、定着条件ありのほうが正例が減る(増えることはない)。"""
        n = HIGH_WINDOW + 5
        tail = [1200] + [1190] * 30 + [1120] * 40
        tvol = [300000] + [100000] * 70
        closes, vols = flat_then(n, 1000, tail, tvol)
        base = self._flags(closes, vols, LabelConfig(sustain_days=0))
        strict = self._flags(closes, vols, LabelConfig(sustain_days=60, sustain_ratio=1.0))
        self.assertLessEqual(int((strict["is_breakout"] == True).sum()),  # noqa: E712
                             int((base["is_breakout"] == True).sum()))

    def test_undetermined_when_sustain_horizon_missing(self):
        """sustain を評価できない末尾は False ではなく NaN。"""
        cfg = LabelConfig(sustain_days=60, sustain_ratio=1.0)
        n = HIGH_WINDOW + 5
        closes, vols = flat_then(n, 1000, [1200] + [1250] * 10, [300000] + [100000] * 10)
        df = self._flags(closes, vols, cfg)
        self.assertTrue(pd.isna(df.iloc[-1]["is_breakout"]))

    def test_forward_needed_accounts_for_sustain(self):
        self.assertEqual(
            LabelConfig(horizon_end=60, hold_days=20, sustain_days=0).forward_needed, 80)
        self.assertEqual(
            LabelConfig(horizon_end=60, hold_days=20, sustain_days=60).forward_needed, 120)


class TestHighWindowParameter(unittest.TestCase):
    def test_78week_window_needs_longer_history(self):
        """78週(368日)窓では、368営業日そろうまで高値が未定義になる。"""
        cfg = LabelConfig(high_window=368, sustain_days=0)
        n = 400
        df = price_panel(make_bars(list(range(1000, 1000 + n)), [100000] * n), cfg)
        self.assertTrue(df["high52w"].iloc[:367].isna().all())
        self.assertTrue(df["high52w"].iloc[367:].notna().all())

    def test_wider_window_is_harder_to_break(self):
        """同じ系列なら、78週高値のほうが52週高値以上になる(超えにくい)。"""
        n = 400
        # 前半に高値、後半は低い水準から回復する形
        closes = list(range(1500, 1500 + 100)) + list(range(1400, 1400 + 300))
        a = price_panel(make_bars(closes, [100000] * n), LabelConfig(high_window=245))
        b = price_panel(make_bars(closes, [100000] * n), LabelConfig(high_window=368))
        both = a["high52w"].notna() & b["high52w"].notna()
        self.assertTrue((b.loc[both, "high52w"] >= a.loc[both, "high52w"]).all())


class TestLabelWindow(unittest.TestCase):
    def _label_at(self, breakout_offset):
        """基準日から breakout_offset 営業日後にブレイクを置き、基準日のラベルを返す。"""
        base = HIGH_WINDOW + 10           # 基準日の位置
        # ラベル確定には基準日から HORIZON_END + HOLD_DAYS 営業日ぶん必要
        n_tail = max(breakout_offset, HORIZON_END) + BASE.forward_needed + 30
        closes = [1000] * base
        vols = [100000] * base
        for i in range(n_tail):
            if i == breakout_offset:
                closes.append(1200); vols.append(300000)
            elif i > breakout_offset:
                closes.append(1200); vols.append(100000)
            else:
                closes.append(1000); vols.append(100000)
        df = price_panel(make_bars(closes, vols), BASE)
        df = breakout_flags(df, BASE)
        df = attach_labels(df, BASE)
        return df.iloc[base - 1]["label"]

    def test_breakout_inside_horizon_is_positive(self):
        """ホライズン内(t+20〜t+120)のブレイクは正例。"""
        self.assertEqual(self._label_at(HORIZON_START + 30), 1.0)

    def test_breakout_too_soon_is_not_counted(self):
        """t+20より手前のブレイクはホライズン外なので数えない。"""
        self.assertEqual(self._label_at(HORIZON_START - 5), 0.0)

    def test_breakout_at_horizon_start_is_positive(self):
        """境界 t+20 ちょうどは含む。"""
        self.assertEqual(self._label_at(HORIZON_START), 1.0)

    def test_breakout_beyond_horizon_is_negative(self):
        """t+120 を超えたブレイクは負例。"""
        self.assertEqual(self._label_at(HORIZON_END + 15), 0.0)

    def test_label_is_nan_when_future_data_insufficient(self):
        """
        将来データが HORIZON_END + HOLD_DAYS 営業日ぶん無い基準日は
        「未確定」として NaN。0（起きなかった）に丸めてはいけない。
        """
        n = HIGH_WINDOW + 60
        df = price_panel(make_bars([1000] * n, [100000] * n), BASE)
        df = attach_labels(breakout_flags(df, BASE), BASE)
        need = BASE.forward_needed
        self.assertTrue(df["label"].iloc[-need:].isna().all(),
                        f"末尾 {need}営業日はラベル未確定であるべき")
        self.assertTrue(df["label"].iloc[:-need].notna().all(),
                        "それ以前はラベルが確定しているべき")


class TestNoLookahead(unittest.TestCase):
    def test_features_do_not_change_when_future_is_appended(self):
        """
        最重要のテスト。
        系列の後ろに未来のバーを足しても、基準日以前の特徴量は1つも変わらないこと。
        変わるならどこかで未来を見ている。
        """
        n = HIGH_WINDOW + 60
        rng = np.random.default_rng(42)
        closes = list(1000 + np.cumsum(rng.normal(0, 10, n)).round(2))
        vols = list(rng.integers(50000, 200000, n))

        short = price_panel(make_bars(closes, vols))
        long_closes = closes + list(1000 + np.cumsum(rng.normal(0, 10, 200)).round(2))
        long_vols = vols + list(rng.integers(50000, 200000, 200))
        long = price_panel(make_bars(long_closes, long_vols))

        cols = ["close", "high52w", "high52w_prior", "r_high",
                "vol_ma20", "volume_trend", "tv_ma20", "r_high_3m", "r_high_6m"]
        a = short[cols].reset_index(drop=True)
        b = long[cols].iloc[:len(short)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False,
                                      obj="未来のバーを足すと過去の特徴量が変わった(先読みの疑い)")

    def test_high52w_prior_excludes_current_day(self):
        """ブレイク判定の基準は『当日を含まない』52週高値であること。"""
        n = HIGH_WINDOW + 3
        closes = [1000] * n + [1500]
        df = price_panel(make_bars(closes, [100000] * (n + 1)))
        last = df.iloc[-1]
        self.assertEqual(last["high52w"], 1500, "当日込みの高値は当日値を含む")
        self.assertEqual(last["high52w_prior"], 1000, "当日を除いた高値は前日までの最大")


class TestCrossSectionalRank(unittest.TestCase):
    """
    横断面正規化: 同じ日付内でのパーセンタイル順位に変換する。

    絶対値のままだと相場局面に依存する（訓練期間の正例率 6.19% に対し
    テスト期間 21.66% と3倍以上ずれていた）。順位に直すと局面依存が消える。
    """

    def _df(self):
        return pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31"] * 4 + ["2024-02-29"] * 4),
            "Code": list("ABCD") * 2,
            "r_high": [70, 80, 90, 95, 50, 60, 70, np.nan],
        })

    def test_rank_is_within_date(self):
        """順位は日付ごとに独立して計算される。"""
        out = add_cross_sectional_ranks(self._df(), ["r_high"])
        jan = out[out["Date"] == "2024-01-31"].set_index("Code")["r_high_r"]
        feb = out[out["Date"] == "2024-02-29"].set_index("Code")["r_high_r"]
        self.assertAlmostEqual(jan["D"], 1.0)      # 1月の最高値
        self.assertAlmostEqual(feb["C"], 1.0)      # 2月の最高値

    def test_same_absolute_value_gets_different_rank(self):
        """
        これが横断面正規化の要点。
        同じ r_high=70 でも、1月は下位25%、2月は最上位になる。
        絶対値では区別できない「その時点での相対位置」を表現できる。
        """
        out = add_cross_sectional_ranks(self._df(), ["r_high"])
        jan_a = out[(out["Date"] == "2024-01-31") & (out["Code"] == "A")]["r_high_r"].iloc[0]
        feb_c = out[(out["Date"] == "2024-02-29") & (out["Code"] == "C")]["r_high_r"].iloc[0]
        self.assertAlmostEqual(jan_a, 0.25)
        self.assertAlmostEqual(feb_c, 1.0)
        self.assertNotAlmostEqual(jan_a, feb_c)

    def test_missing_stays_missing(self):
        """欠測は 0.5 等で埋めない。観測していない情報を与えることになるため。"""
        out = add_cross_sectional_ranks(self._df(), ["r_high"])
        d = out[(out["Date"] == "2024-02-29") & (out["Code"] == "D")]["r_high_r"].iloc[0]
        self.assertTrue(pd.isna(d))

    def test_single_valid_value_gets_no_rank(self):
        """その日に有効値が1件だけなら順位に意味がないので欠測にする。"""
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-03-29"] * 3),
            "Code": list("ABC"),
            "r_high": [80, np.nan, np.nan],
        })
        out = add_cross_sectional_ranks(df, ["r_high"])
        self.assertTrue(out["r_high_r"].isna().all())

    def test_original_column_is_kept(self):
        """絶対値と順位のどちらが効くかを比較するため、元の列は残す。"""
        out = add_cross_sectional_ranks(self._df(), ["r_high"])
        self.assertIn("r_high", out.columns)
        self.assertIn("r_high_r", out.columns)

    def test_rank_is_monotonic_in_value(self):
        """同一日付内では、値が大きいほど順位も大きい。"""
        out = add_cross_sectional_ranks(self._df(), ["r_high"])
        jan = out[out["Date"] == "2024-01-31"].sort_values("r_high")
        self.assertTrue(jan["r_high_r"].is_monotonic_increasing)

    def test_missing_column_is_skipped(self):
        out = add_cross_sectional_ranks(self._df(), ["r_high", "存在しない列"])
        self.assertNotIn("存在しない列_r", out.columns)


class TestFeaturePresets(unittest.TestCase):
    """特徴量セットの定義が壊れていないこと。"""

    def test_all_excludes_rank_columns(self):
        """
        `all` は絶対値のみ。順位版は別プリセットで比較する。

        列数を固定値で縛ると特徴量を足すたびに落ちるので、
        「順位列が混ざっていないこと」と「空でないこと」だけを見る。
        """
        import features as F
        cols = F.columns("all")
        self.assertFalse(any(c.endswith("_r") for c in cols))
        self.assertGreater(len(cols), 10)

    def test_rank_groups_match_rank_targets(self):
        """
        順位グループの対象と RAW_FOR_RANK がずれると、
        存在しない列を含むプリセットが黙って出来る。
        """
        import features as F
        from_groups = {c for g in F.RANKED_GROUPS for c in F.GROUPS[g]}
        self.assertEqual(set(F.RAW_FOR_RANK), from_groups)

    def test_valuation_columns_exist_in_a_preset(self):
        """PER/PBR は API に無く自前で作った列。プリセットから参照できること。"""
        import features as F
        cols = F.columns("valuation_only")
        for c in ("per", "pbr", "earnings_yield", "book_yield"):
            self.assertIn(c, cols)

    def test_rank_all_mirrors_all(self):
        """
        `rank_all` は `all` と同じ構成の順位版。

        手書きにしていたため `all` にグループを足したときに追随せず、
        123列 vs 139列 とずれた。グループ単位で対応を確認する。
        """
        import features as F
        self.assertEqual(len(F.columns("rank_all")), len(F.columns("all")))
        # グループ名の一覧は1箇所（ALL_GROUPS）だけに書く
        self.assertEqual(F.PRESETS["all"], F.ALL_GROUPS)
        # グループ単位でも対応していること
        for g in F.PRESETS["all"]:
            expected = f"{g}_rank" if f"{g}_rank" in F.GROUPS else g
            self.assertIn(expected, F.PRESETS["rank_all"], g)

    def test_every_preset_resolves(self):
        import features as F
        for name in F.PRESETS:
            self.assertGreater(len(F.columns(name)), 0, f"{name} が空")

    def test_rank_targets_cover_non_market_groups(self):
        """市場環境(TOPIX)は全銘柄共通なので順位化しない。"""
        import features as F
        self.assertNotIn("topix_ret_20", F.RAW_FOR_RANK)
        self.assertIn("r_high", F.RAW_FOR_RANK)

    def test_all_columns_is_the_union_over_presets(self):
        """
        `all_columns()` は build_dataset.py がデータセットに残す列を決める。
        ここが `columns("all")` だと順位列が丸ごと落ち、
        rank_* プリセットが「列が無い」ではなく黙って空回りする。
        """
        import features as F
        every = F.all_columns()
        for name in F.PRESETS:
            missing = [c for c in F.columns(name) if c not in every]
            self.assertEqual(missing, [], f"{name} の列が all_columns に無い")

    def test_all_columns_keeps_rank_columns(self):
        import features as F
        every = F.all_columns()
        self.assertEqual(len([c for c in every if c.endswith("_r")]),
                         len(F.RAW_FOR_RANK))


class TestLagOverAvailableValues(unittest.TestCase):
    """
    値が疎な列を開示単位で shift すると、ラグがほぼ全部 NaN になる。
    ROE は通期開示にしか入らないため実際にこれが起きており、
    ROE_chg の 94.4% 欠測 -> 「決算に予測力なし」という誤った結論につながった。
    """

    def _annual_in_quarterly(self):
        """四半期開示が並ぶ中で、年1回だけ値が入る列。ROE と同じ形。"""
        dates = pd.date_range("2020-05-15", periods=12, freq="91D")
        val = [np.nan] * 12
        val[0], val[4], val[8] = 10.0, 12.0, 15.0   # 年1回だけ
        return pd.DataFrame({"Code": "1234", "DiscDate": dates, "x": val})

    def test_picks_previous_available_not_previous_row(self):
        df = self._annual_in_quarterly()
        lag1 = _lag_available(df, "x", 1)
        # 値がある行(4番目)には、その前の値(10.0)が入る
        self.assertAlmostEqual(lag1.iloc[4], 10.0)
        self.assertAlmostEqual(lag1.iloc[8], 12.0)
        # 値が無い行はラグも NaN のまま
        self.assertTrue(np.isnan(lag1.iloc[5]))

    def test_naive_shift_cannot_produce_a_single_change(self):
        """
        従来実装との差を明示する。

        行単位 shift でもラグ列自体には値が入る（3件）。
        しかし入る位置が「値のある行の *次* の行」なので、
        x と x_lag が同じ行に揃わない。
        chg = q0 - q1 は両方揃った行でしか計算できないため、結果は0件になる。
        これが ROE_chg 94.4%欠測 の正体。
        """
        df = self._annual_in_quarterly()
        naive = df.groupby("Code", sort=False)["x"].shift(1)
        self.assertEqual(int(naive.notna().sum()), 3)      # 値自体は入る
        both_naive = df["x"].notna() & naive.notna()
        self.assertEqual(int(both_naive.sum()), 0)         # だが同じ行に揃わない

        fixed = _lag_available(df, "x", 1)
        both_fixed = df["x"].notna() & fixed.notna()
        self.assertEqual(int(both_fixed.sum()), 2)

    def test_second_lag(self):
        df = self._annual_in_quarterly()
        lag2 = _lag_available(df, "x", 2)
        self.assertAlmostEqual(lag2.iloc[8], 10.0)

    def test_rejects_a_stale_previous_value(self):
        """間が空きすぎた開示との比較は無意味なので使わない。"""
        df = pd.DataFrame({
            "Code": "1234",
            "DiscDate": pd.to_datetime(["2015-05-15", "2024-05-15"]),
            "x": [10.0, 20.0]})
        self.assertTrue(np.isnan(_lag_available(df, "x", 1).iloc[1]))

    def test_never_looks_forward(self):
        """ラグは過去方向のみ。未来の値が混ざっていないこと。"""
        df = self._annual_in_quarterly()
        lag1 = _lag_available(df, "x", 1)
        for i in range(len(df)):
            v = lag1.iloc[i]
            if not np.isnan(v):
                past = df["x"].iloc[:i]
                self.assertIn(v, list(past.dropna()))

    def test_does_not_cross_between_codes(self):
        df = pd.DataFrame({
            "Code": ["A", "A", "B", "B"],
            "DiscDate": pd.to_datetime(["2023-05-15", "2023-08-15",
                                        "2023-05-15", "2023-08-15"]),
            "x": [1.0, 2.0, 3.0, 4.0]})
        lag1 = _lag_available(df, "x", 1)
        self.assertTrue(np.isnan(lag1.iloc[2]))   # B の先頭に A の値が来ない
        self.assertAlmostEqual(lag1.iloc[3], 3.0)


class TestBreakoutPopulation(unittest.TestCase):
    """
    母集団を「52週高値の更新日 × 銘柄」にする。
    運用では場中に52週高値を超えた銘柄を、その日の夜に判定する。
    """

    def _panel(self, closes, highs=None, w=5):
        """52週高値の窓を w 日にした小さなパネル。"""
        n = len(closes)
        df = pd.DataFrame({
            "Code": "1234",
            "Date": pd.date_range("2021-01-04", periods=n, freq="B"),
            "close": closes,
            "high": highs if highs is not None else closes,
        })
        df["high52w_prior"] = (df["high"].rolling(w, min_periods=w).max().shift(1))
        return df

    def test_detects_a_new_high(self):
        # 5日窓。6日目に過去最高を超える
        df = self._panel([100, 101, 102, 103, 104, 110, 105])
        out = mark_new_highs(df, cooldown=3)
        self.assertTrue(bool(out["is_new_high"].iloc[5]))
        self.assertFalse(bool(out["is_new_high"].iloc[6]))

    def test_uses_the_prior_high_not_todays(self):
        """当日を含む高値と比べると、自分自身と比べることになり常に成立する。"""
        df = self._panel([100, 101, 102, 103, 104, 110])
        out = mark_new_highs(df, cooldown=3)
        # high52w_prior は当日を含まないので、110 > 104 で成立
        self.assertAlmostEqual(out["high52w_prior"].iloc[5], 104.0)

    def test_consecutive_updates_collapse_to_one(self):
        """
        高値更新が続くと、ほぼ同じ特徴量・重なるラベルのサンプルが量産され、
        件数が水増しされるうえ独立でなくなる。初回だけを残す。
        """
        df = self._panel([100, 101, 102, 103, 104, 110, 111, 112, 113])
        out = mark_new_highs(df, cooldown=3)
        self.assertEqual(int(out["is_new_high"].sum()), 4)      # 110,111,112,113
        self.assertEqual(int(out["is_fresh_break"].sum()), 1)   # 110 のみ
        self.assertTrue(bool(out["is_fresh_break"].iloc[5]))

    def test_a_later_break_after_a_quiet_gap_counts_again(self):
        closes = [100, 101, 102, 103, 104, 110] + [105] * 5 + [120]
        df = self._panel(closes)
        out = mark_new_highs(df, cooldown=3)
        self.assertEqual(int(out["is_fresh_break"].sum()), 2)

    def test_high_based_detection(self):
        """場中に超えれば候補。終値が押し戻されても更新日とみなす。"""
        closes = [100, 101, 102, 103, 104, 103]
        highs = [100, 101, 102, 103, 104, 115]
        df = self._panel(closes, highs)
        on_high = mark_new_highs(df.copy(), cooldown=3, on_high=True)
        on_close = mark_new_highs(df.copy(), cooldown=3, on_high=False)
        self.assertTrue(bool(on_high["is_new_high"].iloc[5]))
        self.assertFalse(bool(on_close["is_new_high"].iloc[5]))


class TestRiseLabel(unittest.TestCase):
    """更新日の終値から、先 horizon 営業日以内に threshold 以上上昇したか。"""

    def _df(self, closes):
        return pd.DataFrame({
            "Code": "1234",
            "Date": pd.date_range("2021-01-04", periods=len(closes), freq="B"),
            "close": closes})

    def test_positive_when_it_rises_enough(self):
        out = attach_rise_label(self._df([100, 105, 125, 110, 108]),
                                horizon=3, threshold=0.20)
        self.assertTrue(bool(out["label"].iloc[0]))    # 125/100-1 = 25%

    def test_negative_when_it_does_not(self):
        out = attach_rise_label(self._df([100, 105, 110, 108, 107]),
                                horizon=3, threshold=0.20)
        self.assertFalse(bool(out["label"].iloc[0]))

    def test_measured_on_close_not_high(self):
        """高値ベースだと「一瞬触れただけ」を正例にしてしまう。"""
        out = attach_rise_label(self._df([100, 119, 119, 119]),
                                horizon=3, threshold=0.20)
        self.assertFalse(bool(out["label"].iloc[0]))

    def test_undetermined_at_the_tail_is_nan_not_false(self):
        """先の営業日が足りない末尾を False にすると負例が水増しされる。"""
        out = attach_rise_label(self._df([100, 105, 110]),
                                horizon=3, threshold=0.20)
        self.assertTrue(pd.isna(out["label"].iloc[1]))
        self.assertTrue(pd.isna(out["label"].iloc[2]))

    def test_never_uses_the_current_day(self):
        """当日の終値は分母。将来の窓は t+1 から始まる。"""
        out = attach_rise_label(self._df([100, 100, 100, 100, 100]),
                                horizon=3, threshold=0.0)
        self.assertAlmostEqual(out["future_rise"].iloc[0], 0.0)


class TestDivergenceGuard(unittest.TestCase):
    """
    比率は分母が小さいと発散する。
    実測で payout_ratio 460,000% / guidance_op_growth 96,285% /
    net_margin -166,800% が出ていた。
    PER/PBR にはガードを入れたのに、後から足した指標には入れ忘れていた。
    """

    def test_out_of_range_becomes_missing(self):
        s = pd.Series([10.0, 500.0, 460000.0, -1e6])
        out = clip_divergent(s, -100.0, 1000.0)
        self.assertAlmostEqual(out.iloc[0], 10.0)
        self.assertAlmostEqual(out.iloc[1], 500.0)
        self.assertTrue(np.isnan(out.iloc[2]))
        self.assertTrue(np.isnan(out.iloc[3]))

    def test_does_not_clamp_to_the_bound(self):
        """
        上限で切り捨てると「上限にへばりついた実在の値」に見えてしまう。
        欠測にするのが正しい。
        """
        out = clip_divergent(pd.Series([5000.0]), 0.0, 1000.0)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertNotEqual(out.iloc[0], 1000.0)

    def test_infinities_are_removed(self):
        out = clip_divergent(pd.Series([np.inf, -np.inf, 1.0]), -10.0, 10.0)
        self.assertEqual(int(out.notna().sum()), 1)

    def test_missing_stays_missing(self):
        out = clip_divergent(pd.Series([np.nan, 5.0]), 0.0, 10.0)
        self.assertTrue(np.isnan(out.iloc[0]))

    def test_keeps_legitimately_large_values(self):
        """ROE が100%を超えるのは実在する。範囲内なら残す。"""
        out = clip_divergent(pd.Series([250.0, -180.0]), -500.0, 500.0)
        self.assertEqual(int(out.notna().sum()), 2)


class TestSectorIsPointInTime(unittest.TestCase):
    """
    最新の銘柄マスタを過去のサンプルに当てると先読みになる。
    とくに市場区分は2022年4月の東証再編で全銘柄が変わっているので、
    2018年のサンプルに現在の区分を付けるのは誤り。
    merge_asof の backward で「その時点で有効だった区分」に合わせる。
    """

    def _master_hist(self):
        """2022-04 に市場区分が変わった銘柄。"""
        return pd.DataFrame({
            "Date": pd.to_datetime(["2021-01-29", "2022-03-31", "2022-04-28"]),
            "Code": "1234",
            "S33": ["3050", "3050", "3050"],
            "Mkt": ["0111", "0111", "0111"],   # 旧区分
        }).assign(Mkt=["0111", "0111", "0112"])   # 2022-04 で新区分へ

    def test_uses_the_segment_in_force_at_the_time(self):
        mh = self._master_hist()
        samples = pd.DataFrame({
            "Date": pd.to_datetime(["2021-06-30", "2022-06-30"]),
            "Code": "1234"})
        merged = pd.merge_asof(samples.sort_values("Date"),
                               mh.sort_values("Date"),
                               on="Date", by="Code", direction="backward")
        # 2021年のサンプルには旧区分、2022年6月には新区分
        self.assertEqual(merged.iloc[0]["Mkt"], "0111")
        self.assertEqual(merged.iloc[1]["Mkt"], "0112")

    def test_never_uses_a_future_snapshot(self):
        """スナップショットより前のサンプルには何も付かない。"""
        mh = self._master_hist()
        samples = pd.DataFrame({"Date": pd.to_datetime(["2020-06-30"]),
                                "Code": "1234"})
        merged = pd.merge_asof(samples, mh.sort_values("Date"),
                               on="Date", by="Code", direction="backward")
        self.assertTrue(pd.isna(merged.iloc[0]["Mkt"]))


class TestValuationUsesUnadjustedPrice(unittest.TestCase):
    """
    EPS / BPS / 株数は開示時点のままで分割調整されていない。
    調整後株価と組み合わせると、分割をまたいだ時点で比率がずれる。
    実測で earnings_yield が最大 +1276%（EPSが株価の12.7倍）まで出ていた。
    """

    def _bars(self, n=300):
        """途中で 1:10 分割した銘柄。AdjC は過去が1/10になる。"""
        d = pd.date_range("2021-01-04", periods=n, freq="B")
        raw = np.full(n, 1000.0)
        raw[n // 2:] = 100.0                  # 分割で株価が1/10に
        adj = np.full(n, 100.0)               # 調整後は一貫して100
        return pd.DataFrame({
            "Code": "1234", "Date": d,
            "O": raw, "H": raw, "L": raw, "C": raw, "Vo": 10000.0,
            "AdjO": adj, "AdjH": adj, "AdjL": adj, "AdjC": adj, "AdjVo": 100000.0,
        })

    def test_panel_keeps_both_prices(self):
        panel = price_panel(self._bars())
        # 分割前: 調整後は100、未調整は1000
        first = panel.iloc[0]
        self.assertAlmostEqual(first["close"], 100.0)
        self.assertAlmostEqual(first["close_raw"], 1000.0)

    def test_adjusted_price_is_still_used_for_breakout_logic(self):
        """ブレイク判定は調整後を使う。未調整だと分割日に偽のブレイクが出る。"""
        panel = price_panel(self._bars())
        self.assertTrue((panel["close"] == 100.0).all())

    def test_using_adjusted_price_would_distort_the_ratio(self):
        """
        この不整合がどれだけ効くかを固定する。
        分割前の時点で、調整後株価を使うと益回りが10倍に化ける。
        """
        panel = price_panel(self._bars())
        first = panel.iloc[0]
        eps = 50.0                                    # 開示時点の1株利益
        correct = eps / first["close_raw"] * 100.0    # 5%
        wrong = eps / first["close"] * 100.0          # 50%
        self.assertAlmostEqual(correct, 5.0)
        self.assertAlmostEqual(wrong, 50.0)
        self.assertAlmostEqual(wrong / correct, 10.0)


class TestQuarterSequenceFeatures(unittest.TestCase):
    """
    52週高値のブレイクは3〜4決算続けて好調な銘柄で起きる。
    「直近1回だけ伸びた」と「3期続けて伸びている」を区別できる必要がある。

    q0=最新 / q1=前回 / q2=2回前 / q3=3回前。
    """

    def _panel(self, values):
        """1銘柄・四半期ごとの開示。values は古い順。"""
        n = len(values)
        return pd.DataFrame({
            "Code": "1234",
            "CurPerType": (["1Q", "2Q", "3Q", "FY"] * (n // 4 + 1))[:n],
            "DiscDate": pd.date_range("2020-05-15", periods=n, freq="91D"),
            "CurFYSt": pd.to_datetime("2020-04-01"),
            "DiscTime": "15:00",
            "Sales": values, "OP": values, "NP": values, "EPS": values,
            "Eq": [1000.0] * n, "TA": [2000.0] * n, "ROE": values,
            "FOP": [100.0] * n, "ShOutFY": [100.0] * n, "TrShFY": [0.0] * n,
        })

    def _axis(self, values, axis="ROE"):
        q = quarterize_panel(self._panel(values))
        return q.iloc[-1]     # 最新の開示行

    def test_each_step_difference_is_present(self):
        """chg1/chg2/chg3 が各段の差になっていること。"""
        r = self._axis([10.0, 20.0, 45.0, 50.0])   # 古い順
        # q0=50, q1=45, q2=20, q3=10
        self.assertAlmostEqual(r["ROE_q0"], 50.0)
        self.assertAlmostEqual(r["ROE_q1"], 45.0)
        self.assertAlmostEqual(r["ROE_q2"], 20.0)
        self.assertAlmostEqual(r["ROE_q3"], 10.0)
        self.assertAlmostEqual(r["ROE_chg1"], 5.0)    # 50-45
        self.assertAlmostEqual(r["ROE_chg2"], 25.0)   # 45-20
        self.assertAlmostEqual(r["ROE_chg3"], 10.0)   # 20-10

    def test_q1_minus_q2_was_the_missing_piece(self):
        """
        以前は chg1 (q0-q1) と chg (q0-q2) しか無く、
        q1-q2 が抜けていた。決定木は q1 と q2 から差を作れないため、
        「前回も伸びていたか」を表現できていなかった。
        """
        r = self._axis([10.0, 20.0, 45.0, 50.0])
        self.assertAlmostEqual(r["ROE_chg2"], r["ROE_q1"] - r["ROE_q2"])

    def test_acceleration(self):
        """加速 = 直近の変化 - その前の変化。"""
        r = self._axis([10.0, 20.0, 45.0, 50.0])
        self.assertAlmostEqual(r["ROE_accel"], 5.0 - 25.0)   # 減速している

    def test_up_streak_counts_consecutive_increases(self):
        r = self._axis([10.0, 20.0, 30.0, 40.0])    # 毎回増加
        self.assertAlmostEqual(r["ROE_up_streak"], 3.0)

    def test_up_streak_stops_at_the_first_decline(self):
        """直近から数える。途中で落ちたらそこで止まる。"""
        r = self._axis([10.0, 40.0, 30.0, 35.0])    # q3=10,q2=40,q1=30,q0=35
        # q0>q1 は真、q1>q2 は偽 -> 1 で止まる
        self.assertAlmostEqual(r["ROE_up_streak"], 1.0)

    def test_up_streak_is_zero_when_latest_declines(self):
        r = self._axis([10.0, 20.0, 30.0, 25.0])
        self.assertAlmostEqual(r["ROE_up_streak"], 0.0)

    def test_pos_ratio_counts_positive_quarters(self):
        r = self._axis([-10.0, -5.0, 30.0, 40.0])
        self.assertAlmostEqual(r["ROE_pos_ratio"], 0.5)   # 4期中2期がプラス

    def test_streak_is_missing_when_too_few_quarters(self):
        """有効な期が2つ未満なら判定しない。"""
        q = quarterize_panel(self._panel([10.0]))
        self.assertTrue(np.isnan(q.iloc[-1]["ROE_up_streak"]))

    def test_three_quarter_change(self):
        r = self._axis([10.0, 20.0, 45.0, 50.0])
        self.assertAlmostEqual(r["ROE_chg_3q"], 40.0)   # 50-10


class TestDefaultLabelIsE(unittest.TestCase):
    """
    既定のラベル定義が、10定義の比較で採用した E であることを固定する。
    ここが黙って変わると、過去の結果と比較できなくなる。

    一時 78週 + 小型株限定に変えたが、E に戻した。
    あの変更は「決算特徴量に予測力が無い」という結論への対応であり、
    その結論自体が決算データの欠損によるもので前提が成り立っていなかった。
    """

    def test_default_matches_definition_e(self):
        from build_dataset import DEFAULT_LABEL as L
        self.assertEqual(L.high_window, 245)         # 52週
        self.assertEqual(L.horizon_start, 20)        # 1ヶ月先から
        self.assertEqual(L.horizon_end, 120)         # 6ヶ月先まで
        self.assertEqual(L.hold_days, 20)
        self.assertAlmostEqual(L.hold_drawdown, 0.92)
        self.assertEqual(L.sustain_days, 60)         # 60営業日後も
        self.assertAlmostEqual(L.sustain_ratio, 1.0) # 水準維持

    def test_market_cap_is_not_filtered(self):
        """
        時価総額では絞らない（全銘柄が対象）。

        絞ると母集団が変わり過去の結果と比較できなくなる。
        絞りたい場合は定数に数値を入れれば有効になるが、
        既定は None であることを固定する。
        """
        import build_dataset as B
        self.assertIsNone(B.MIN_MARKET_CAP)
        self.assertIsNone(B.MAX_MARKET_CAP)

    def test_high_window_is_52_weeks(self):
        """245営業日が52週であること。"""
        from build_dataset import DEFAULT_LABEL as L
        self.assertEqual(round(L.high_window / 245 * 52), 52)
        self.assertIn("52週", L.name)

    def test_forward_needed_is_180(self):
        """
        E はラベル確定に180営業日を要する。
        エンバーゴをこれより短くするとリークする。
        """
        from build_dataset import DEFAULT_LABEL as L
        self.assertEqual(L.forward_needed, 180)


class TestQuarterizePanel(unittest.TestCase):
    def _fins(self):
        rows = []
        # FY2024: 累計 100 -> 220 -> 360 -> 520
        for q, (disc, sales, op, np_, eps) in enumerate(
            [("2024-08-05", 100, 10, 6, 6.0), ("2024-11-05", 220, 24, 15, 15.0),
             ("2025-02-05", 360, 42, 26, 26.0), ("2025-05-12", 520, 64, 40, 40.0)], start=1):
            rows.append({
                "Code": "00010", "DiscDate": disc, "DiscTime": "15:00",
                "CurPerType": ["1Q", "2Q", "3Q", "4Q"][q - 1], "CurFYSt": "2024-04-01",
                "Sales": sales, "OP": op, "NP": np_, "EPS": eps,
                "Eq": 1000, "TA": 2000, "ROE": 4.0, "FOP": 80,
                "FSales": None, "FNP": None, "FEPS": None,
                "ShOutFY": 1_000_000, "TrShFY": 50_000, "DocType": "x", "CurPerEn": disc,
            })
        # FY2025 1Q: 売上 140 (前年同期100 -> +40%)
        rows.append({
            "Code": "00010", "DiscDate": "2025-08-05", "DiscTime": "15:00",
            "CurPerType": "1Q", "CurFYSt": "2025-04-01",
            "Sales": 140, "OP": 16, "NP": 10, "EPS": 10.0,
            "Eq": 1050, "TA": 2100, "ROE": 5.0, "FOP": 90,
            "FSales": None, "FNP": None, "FEPS": None,
            "ShOutFY": 1_000_000, "TrShFY": 50_000, "DocType": "x", "CurPerEn": "2025-08-05",
        })
        return pd.DataFrame(rows)

    def test_cumulative_differencing(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        self.assertAlmostEqual(latest["sales_growth_q0"], 40.0, places=6)
        self.assertAlmostEqual(latest["eps_growth_q0"], (10.0 - 6.0) / 6.0 * 100, places=6)

    def test_lags_are_previous_disclosures(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        # 直近が FY2025-1Q なら q1 は FY2024-4Q、q2 は FY2024-3Q
        self.assertAlmostEqual(latest["ROE_q0"], 5.0)
        self.assertAlmostEqual(latest["ROE_q1"], 4.0)

    def test_progress_vs_base(self):
        q = quarterize_panel(self._fins()).sort_values("DiscDate").reset_index(drop=True)
        latest = q.iloc[-1]
        # 1Q 累計営業利益16 / 通期予想90 = 17.8%、基準 1×25 = 25 -> -7.2
        self.assertAlmostEqual(latest["progress_vs_base"], 16 / 90 * 100 - 25, places=6)

    def test_forecast_only_disclosure_is_dropped(self):
        fins = self._fins()
        noise = fins.iloc[[0]].copy()
        noise["DiscDate"] = "2025-09-01"
        noise[["Sales", "OP", "NP", "EPS"]] = np.nan
        q = quarterize_panel(pd.concat([fins, noise], ignore_index=True))
        self.assertNotIn("2025-09-01", set(q["DiscDate"].astype(str).str[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
