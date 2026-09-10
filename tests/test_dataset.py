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
    clip_divergent, mark_new_highs, attach_rise_label, RiseConfig,
    HOLD_DAYS, HORIZON_END, HORIZON_START, HIGH_WINDOW, LabelConfig,
    _lag_available, add_cross_sectional_ranks, attach_labels, breakout_flags,
    price_panel, quarterize_panel, market_environment, MACRO_ETFS,
    cap_band, fund_complete_flag, CAP_BAND_EDGES, FUND_REQUIREMENT_SETS,
    SECTOR_INDEX, S33_TO_INDEX, attach_sector_index, sector_index_returns,
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

    def test_future_columns_are_never_features(self):
        """
        未来から作った列が特徴量に混ざればリークで結果が無意味になる。

        meta として持ち出しているので、名前の付け替えひとつで
        features 側に流れ込みうる。機械的に止める。
        """
        import features as F
        from build_dataset import FUTURE_COLS
        overlap = set(FUTURE_COLS) & set(F.all_columns())
        self.assertEqual(overlap, set())

    def test_no_redundant_slope_columns(self):
        """
        (q0-q2)/2 は chg の定数倍。EDA で8軸すべて相関 1.000 だった。
        情報が無い列を戻さない。
        """
        import features as F
        self.assertEqual([c for c in F.all_columns() if c.endswith("_slope")], [])

    def test_non_numeric_category_is_encoded_not_nulled(self):
        """
        規模区分が100%欠測だった原因は項目名ではなく、値が
        "TOPIX Small 2" のような文字列で to_numeric が全件 NaN にしていたこと。
        文字列のカテゴリを黙って空列にしない。
        """
        from build_dataset import encode_category
        s = pd.Series(["TOPIX Small 2", "TOPIX Core30", "TOPIX Small 2", None])
        out = encode_category(s, "ScaleCat")
        self.assertEqual(out.notna().sum(), 3)
        self.assertEqual(out.iloc[0], out.iloc[2])
        self.assertNotEqual(out.iloc[0], out.iloc[1])
        # 並びは辞書順で固定。実行ごとに変わるとモデルが再現しない
        self.assertEqual(out.iloc[1], 0.0)   # "TOPIX Core30"
        self.assertEqual(out.iloc[0], 1.0)   # "TOPIX Small 2"

    def test_numeric_category_is_left_alone(self):
        """S33/S17/Mkt は数字コードで来る。符号化し直してはいけない。"""
        from build_dataset import encode_category
        out = encode_category(pd.Series(["3050", "6100", None]), "S33")
        self.assertEqual(out.iloc[0], 3050)
        self.assertEqual(out.iloc[1], 6100)

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
    """到達の軸: 先 horizon 営業日以内に threshold 以上上昇したか。"""

    def _df(self, closes):
        return pd.DataFrame({
            "Code": "1234",
            "Date": pd.date_range("2021-01-04", periods=len(closes), freq="B"),
            "close": closes})

    @staticmethod
    def _reach_only(horizon=3, threshold=0.20):
        """継続の条件を全部切って、到達の軸だけを見る設定。"""
        return RiseConfig(horizon=horizon, threshold=threshold,
                          keep_days=0, end_ratio=None, require_uptrend=False)

    def test_positive_when_it_rises_enough(self):
        out = attach_rise_label(self._df([100, 105, 125, 110, 108]),
                                self._reach_only())
        self.assertTrue(bool(out["label"].iloc[0]))    # 125/100-1 = 25%

    def test_negative_when_it_does_not(self):
        out = attach_rise_label(self._df([100, 105, 110, 108, 107]),
                                self._reach_only())
        self.assertFalse(bool(out["label"].iloc[0]))

    def test_measured_on_close_not_high(self):
        """高値ベースだと「一瞬触れただけ」を正例にしてしまう。"""
        out = attach_rise_label(self._df([100, 119, 119, 119]),
                                self._reach_only())
        self.assertFalse(bool(out["label"].iloc[0]))

    def test_undetermined_at_the_tail_is_nan_not_false(self):
        """先の営業日が足りない末尾を False にすると負例が水増しされる。"""
        out = attach_rise_label(self._df([100, 105, 110]), self._reach_only())
        self.assertTrue(pd.isna(out["label"].iloc[1]))
        self.assertTrue(pd.isna(out["label"].iloc[2]))

    def test_never_uses_the_current_day(self):
        """当日の終値は分母。将来の窓は t+1 から始まる。"""
        out = attach_rise_label(self._df([100, 100, 100, 100, 100]),
                                self._reach_only(threshold=0.0))
        self.assertAlmostEqual(out["future_rise"].iloc[0], 0.0)


class TestMissingBarsDoNotEraseHistory(unittest.TestCase):
    """
    売買が成立しない日は高値が欠測になる（実測で全行の約3%）。

    rolling の min_periods は「窓の中の非欠測の数」で判定されるため、
    min_periods=w にすると欠測1つで後続 w 行の判定が丸ごと消える。
    78週窓では1つの欠測が約1年半を潰し、実際 2021年の高値更新日が
    0件になっていた（銘柄13290 は2021年の prior が 245行中0行）。
    """

    def _bars(self, n=900, nan_at=None):
        dates = pd.bdate_range("2016-10-03", periods=n)
        px = pd.Series(1000.0 + pd.Series(range(n)) * 0.5)
        h = px * 1.01
        if nan_at is not None:
            h = h.copy()
            h.iloc[nan_at] = np.nan
        return pd.DataFrame({
            "Code": "13290", "Date": dates, "C": px, "H": h, "L": px * 0.99,
            "Vo": 100000, "Va": px * 100000,
            "AdjC": px, "AdjH": h, "AdjL": px * 0.99, "AdjVo": 100000})

    def test_one_missing_high_does_not_blank_the_window(self):
        from build_dataset import price_panel, LabelConfig
        w = 368
        cfg = LabelConfig(high_window=w)
        clean = price_panel(self._bars(), cfg)
        holed = price_panel(self._bars(nan_at=500), cfg)
        # 欠測が無いときに基準が作れている行数
        base = int(clean["high52w_prior"].notna().sum())
        self.assertGreater(base, 0)
        got = int(holed["high52w_prior"].notna().sum())
        # 1つの欠測で失われるのはその行の周辺だけで、窓ぶんではない
        self.assertGreaterEqual(got, base - 2,
                                f"欠測1つで {base - got} 行の判定が消えた")

    def test_history_is_still_required(self):
        """欠測を許すようにしても、履歴が足りない先頭では基準を作らない。"""
        from build_dataset import price_panel, LabelConfig
        w = 368
        out = price_panel(self._bars(), LabelConfig(high_window=w))
        self.assertTrue(out["high52w_prior"].iloc[:w].isna().all())
        self.assertTrue(out["high52w_prior"].iloc[w:].notna().all())

    def test_label_survives_a_missing_close(self):
        """ラベル側も同じ。先の窓に売買不成立の日が1つあるだけで未確定にしない。"""
        from build_dataset import attach_rise_label, RiseConfig
        n = 40
        close = pd.Series([100.0] * 5 + [130.0] * (n - 5))
        holed = close.copy()
        holed.iloc[10] = np.nan
        cfg = RiseConfig(horizon=10, threshold=0.20, keep_days=0,
                         end_ratio=None, require_uptrend=False)
        df = pd.DataFrame({"Code": "1", "Date": pd.bdate_range("2021-01-04", periods=n),
                           "close": holed})
        out = attach_rise_label(df, cfg)
        self.assertTrue(bool(out["label"].iloc[0]))
        # 先の行が足りない末尾は従来どおり未確定
        self.assertTrue(pd.isna(out["label"].iloc[n - 1]))


class TestRiseContinuation(unittest.TestCase):
    """
    継続の軸。到達だけを条件にすると、一瞬吹き上げてすぐ崩れた銘柄も正例になる。
    それはモメンタムではないので、続いたことを条件に加える。
    """

    def _df(self, closes):
        return pd.DataFrame({
            "Code": "1234",
            "Date": pd.date_range("2021-01-04", periods=len(closes), freq="B"),
            "close": [float(c) for c in closes]})

    # 100 から +25% まで飛んで、すぐ元へ戻る（吹き上げ）
    SPIKE = [100] + [125] + [100] * 10
    # 100 から +25% まで上がってそのまま張り付く（継続）
    HOLD = [100] + [125] * 11

    def test_spike_and_hold_both_reach_the_threshold(self):
        """前提の確認。到達だけならどちらも正例になってしまう。"""
        cfg = RiseConfig(horizon=10, threshold=0.20, keep_days=0,
                         end_ratio=None, require_uptrend=False)
        for closes in (self.SPIKE, self.HOLD):
            out = attach_rise_label(self._df(closes), cfg)
            self.assertTrue(bool(out["label"].iloc[0]))

    def test_keep_days_rejects_the_spike(self):
        cfg = RiseConfig(horizon=10, threshold=0.20, keep_days=5,
                         end_ratio=None, require_uptrend=False)
        spike = attach_rise_label(self._df(self.SPIKE), cfg)
        hold = attach_rise_label(self._df(self.HOLD), cfg)
        self.assertEqual(spike["keep_days_cnt"].iloc[0], 1)    # 1日だけ
        self.assertEqual(hold["keep_days_cnt"].iloc[0], 10)
        self.assertFalse(bool(spike["label"].iloc[0]))
        self.assertTrue(bool(hold["label"].iloc[0]))

    def test_end_level_rejects_the_spike(self):
        cfg = RiseConfig(horizon=10, threshold=0.20, keep_days=0,
                         end_ratio=0.10, end_window=1, require_uptrend=False)
        spike = attach_rise_label(self._df(self.SPIKE), cfg)
        hold = attach_rise_label(self._df(self.HOLD), cfg)
        self.assertAlmostEqual(spike["end_level"].iloc[0], 0.0)
        self.assertAlmostEqual(hold["end_level"].iloc[0], 0.25)
        self.assertFalse(bool(spike["label"].iloc[0]))
        self.assertTrue(bool(hold["label"].iloc[0]))

    def test_uptrend_rejects_a_rollover(self):
        """到達したあと下落トレンドに入ったものを外す。"""
        # 上げてから、長期平均を割り込むまでじりじり下げる
        closes = ([100] * 12 + [125] * 3
                  + list(range(124, 104, -1)) + [80] * 12)
        cfg = RiseConfig(horizon=20, threshold=0.20, keep_days=0,
                         end_ratio=None, require_uptrend=True,
                         trend_short=3, trend_long=10)
        out = attach_rise_label(self._df(closes), cfg)
        self.assertTrue(out["future_rise"].iloc[11] >= 0.20)  # 到達はしている
        self.assertEqual(out["uptrend_end"].iloc[11], 0.0)    # だが下落トレンド
        self.assertFalse(bool(out["label"].iloc[11]))

    def test_disabled_conditions_do_not_make_labels_undetermined(self):
        """
        条件を切ったなら、その入力が無くてもラベルは決まる。

        終盤条件を切っているのに「終盤の値が無いから未確定」とすると、
        短い系列のラベルが理由なく消える。
        """
        cfg = RiseConfig(horizon=2, threshold=0.20, keep_days=0,
                         end_ratio=None, require_uptrend=False)
        out = attach_rise_label(self._df([100, 130, 130, 130]), cfg)
        self.assertTrue(bool(out["label"].iloc[0]))

    def test_counts_are_per_code(self):
        """銘柄をまたいで窓が漏れると、別の銘柄の値でラベルが決まる。"""
        a = self._df([100] + [125] * 11).assign(Code="1111")
        b = self._df([100] * 12).assign(Code="2222")
        cfg = RiseConfig(horizon=10, threshold=0.20, keep_days=5,
                         end_ratio=None, require_uptrend=False)
        out = attach_rise_label(pd.concat([a, b], ignore_index=True), cfg)
        self.assertEqual(out["keep_days_cnt"].iloc[0], 10)     # 1111
        self.assertEqual(out["keep_days_cnt"].iloc[12], 0)     # 2222
        self.assertTrue(bool(out["label"].iloc[0]))
        self.assertFalse(bool(out["label"].iloc[12]))


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


class TestFundRequirementKeepsTurnarounds(unittest.TestCase):
    """
    決算の完全性フィルタが、赤字->黒字転換の会社を落としてはいけない。

    full4 は eps_growth / sales_growth（前年同期が0以下だと欠測という定義）を
    要求していたため、前年4期すべて黒字だった会社しか残さなかった。
    実測で eps_growth_turn が全10,116行 0、eps_growth_sym_q0 の max が
    99.939（+100 は前期<0<今期のときだけ）と、転換が1件も無い状態になっていた。
    赤字->黒字転換は株価が最も動くイベントなので、これは大きな取りこぼしになる。
    """

    def _panel(self, eps_by_year, sales=100.0):
        """2会計年度ぶんの四半期開示。eps_by_year は [1年目, 2年目] の単一四半期EPS。"""
        rows = []
        for y, eps in enumerate(eps_by_year):
            fy_start = pd.Timestamp("2020-04-01") + pd.DateOffset(years=y)
            for qi, qt in enumerate(["1Q", "2Q", "3Q", "FY"], start=1):
                rows.append({
                    "Code": "1234", "CurPerType": qt,
                    # 累計開示なので、単一四半期 eps の qi 倍を入れる
                    "DiscDate": fy_start + pd.DateOffset(months=3 * qi, days=45),
                    "CurFYSt": fy_start, "DiscTime": "15:00",
                    "Sales": sales * qi, "OP": eps * qi, "NP": eps * qi,
                    "EPS": eps * qi, "Eq": 1000.0, "TA": 2000.0, "ROE": 5.0 * qi,
                    "FOP": 100.0, "ShOutFY": 100.0, "TrShFY": 0.0,
                })
        return pd.DataFrame(rows)

    def _latest(self, eps_by_year):
        from build_dataset import quarterize_panel
        q = quarterize_panel(self._panel(eps_by_year))
        return q.iloc[-1]

    def test_turn_flag_fires_on_a_real_turnaround(self):
        """前年赤字・今期黒字なら転換フラグが立つ。立たないなら定義か経路が壊れている。"""
        r = self._latest([-20.0, 30.0])
        self.assertEqual(r["eps_growth_turn"], 1.0)

    def test_full4_rejects_a_turnaround_but_full4_sym_keeps_it(self):
        """
        これが full4 を full4_sym に変えた理由そのもの。
        非対称の成長率は前年赤字だと欠測になるので、転換した会社は
        full4 の要求を満たせない。対称版なら定義できるので残る。
        """
        from build_dataset import FUND_REQUIREMENT_SETS
        r = self._latest([-20.0, 30.0])
        asym = [c for c in FUND_REQUIREMENT_SETS["full4"] if c.startswith("eps_growth")]
        sym = [c for c in FUND_REQUIREMENT_SETS["full4_sym"] if c.startswith("eps_growth")]
        self.assertTrue(all(pd.isna(r[c]) for c in asym),
                        f"前年赤字でも非対称版が値を持っている: "
                        f"{ {c: r[c] for c in asym} }")
        self.assertTrue(all(pd.notna(r[c]) for c in sym),
                        f"対称版が欠測になっている（転換が落ちる）: "
                        f"{ {c: r[c] for c in sym} }")

    def test_both_sets_keep_a_company_that_stayed_profitable(self):
        """ずっと黒字なら、どちらの要求でも残る（対称版が甘すぎないことの確認）。"""
        from build_dataset import FUND_REQUIREMENT_SETS
        r = self._latest([10.0, 30.0])
        self.assertEqual(r["eps_growth_turn"], 0.0)
        for key in ("full4", "full4_sym"):
            cols = [c for c in FUND_REQUIREMENT_SETS[key] if c.startswith("eps_growth")]
            self.assertTrue(all(pd.notna(r[c]) for c in cols),
                            f"{key} が黒字継続の会社を落としている")

    def test_the_applied_requirement_is_never_the_asymmetric_one(self):
        """
        full4 に黙って戻ると赤字->黒字転換がまた全部消える。
        絞らない（none）か対称版（full4_sym）のどちらかであること。

        いまは none（絞らずに fund_complete フラグで持つ）。
        絞る側に戻すなら full4_sym を使う。
        """
        import build_dataset as B
        self.assertIn(B.FUND_REQUIREMENT, B.FUND_REQUIREMENT_SETS)
        self.assertNotEqual(B.FUND_REQUIREMENT, "full4",
                            "full4 は前年4期すべて黒字の会社しか残さない")
        # 対称版の要求の強さは full4 と同じでなければならない（緩めたわけではない）
        self.assertEqual(len(B.FUND_REQUIREMENT_SETS["full4_sym"]),
                         len(B.FUND_REQUIREMENT_SETS["full4"]))


class TestDefaultLabel(unittest.TestCase):
    """
    既定の定義を固定する。ここが黙って変わると過去の結果と比較できなくなる。

    母集団は breakout（高値更新日）で、目的変数は RiseConfig が決める。
    LabelConfig は month_end 母集団だけで使う旧定義。
    """

    def test_population_is_breakout(self):
        import build_dataset as B
        self.assertEqual(B.POPULATION, "breakout")
        self.assertTrue(B.BREAKOUT_ON_HIGH)          # 高値ベースで判定
        self.assertEqual(B.BREAKOUT_COOLDOWN, 20)

    def test_default_rise_definition(self):
        from build_dataset import DEFAULT_RISE as R
        self.assertEqual(R.horizon, 60)              # 3ヶ月
        self.assertAlmostEqual(R.threshold, 0.20)    # +20%
        # 継続の軸。8定義の比較とチャートの目視で緩い案を採用した。
        # 厳しい案（維持20日 / 終盤+15%）との差は714件。厳しい案は
        # 正例が1,316件（7.49%）まで減り、評価窓ごとの正例が14〜110件に
        # なって推定が揺れたので、緩い案（2,030件・11.55%）に戻した。
        self.assertEqual(R.keep_days, 10)
        self.assertAlmostEqual(R.end_ratio, 0.10)
        self.assertEqual(R.end_window, 5)
        self.assertTrue(R.require_uptrend)
        self.assertEqual((R.trend_short, R.trend_long), (20, 60))

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

    def test_high_window_is_78_weeks(self):
        """
        368営業日が78週であること。

        52週(245日)から広げた。1年前の高値を1円抜いただけの更新を
        母集団から外し、抜けた水準の意味を重くする。
        """
        import build_dataset as B
        self.assertEqual(B.HIGH_WINDOW, 368)
        self.assertEqual(round(B.HIGH_WINDOW / 245 * 52), 78)

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


class TestMarketEnvironment(unittest.TestCase):
    """
    市場環境（地合い）の特徴量。

    ここは「その日は全銘柄同じ値」であることが前提で、
    横断面正規化の対象から外している。前提が崩れると
    順位化の設計そのものが合わなくなるので固定する。
    """

    @staticmethod
    def _topix(n=400, start="2020-01-01"):
        d0 = dt.date.fromisoformat(start)
        return pd.DataFrame({
            "Date": [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)],
            "topix": [1000.0 * (1.001 ** i) for i in range(n)],
        })

    @classmethod
    def _bars(cls, n=400, start="2020-01-01", codes=None):
        codes = codes if codes is not None else list(MACRO_ETFS.values())
        frames = []
        for k, code in enumerate(codes):
            closes = [100.0 * (1.0 + 0.0005 * (k + 1)) ** i for i in range(n)]
            frames.append(make_bars(closes, code=code, start=start))
        return pd.concat(frames, ignore_index=True)

    def test_every_market_feature_is_actually_produced(self):
        """
        features.py の market グループにあって build_dataset が作らない列は、
        データセットから黙って落ちる。列名のずれを検出する。
        """
        import features as F
        env = market_environment(self._bars(), self._topix())
        missing = [c for c in F.GROUPS["market"] if c not in env.columns]
        self.assertEqual(missing, [], f"market グループにあるのに作られない列: {missing}")

    def test_one_value_per_date(self):
        env = market_environment(self._bars(), self._topix())
        self.assertEqual(len(env), env["Date"].nunique())

    def test_levels_are_not_features(self):
        """水準そのものは入れない。TOPIX 2,700 は『2024年』とほぼ同義になる。"""
        env = market_environment(self._bars(), self._topix())
        for label in MACRO_ETFS:
            self.assertNotIn(label, env.columns)
        self.assertNotIn("topix", env.columns)

    def test_returns_use_adjusted_close(self):
        """
        分割をまたぐと素の終値は半値に飛ぶ。調整後（AdjC）を使っていないと
        その日のリターンが -50% になる。
        """
        bars = self._bars()
        code = MACRO_ETFS["nk225"]
        m = bars["Code"] == code
        # 素の終値だけを途中から半分にする（AdjC はそのまま）
        idx = bars.index[m][200:]
        bars.loc[idx, "C"] = bars.loc[idx, "C"] / 2
        env = market_environment(bars, self._topix())
        r = env.loc[env["Date"] == pd.Timestamp("2020-07-19"), "nk225_ret_20"]
        self.assertTrue(r.notna().all())
        self.assertGreater(float(r.iloc[0]), -10.0)

    def test_missing_code_does_not_break_the_build(self):
        """
        データセットは保存済みの生データから作り直す。
        古いスナップショットに ETF が入っていなくても、
        その軸が欠測になるだけで組み立て自体は通ること。
        """
        bars = self._bars(codes=[MACRO_ETFS["nk225"]])
        env = market_environment(bars, self._topix())
        self.assertTrue(env["gold_ret_20"].isna().all())
        self.assertTrue(env["nk225_ret_20"].notna().any())

    def test_risk_off_is_the_difference(self):
        env = market_environment(self._bars(), self._topix())
        row = env.dropna(subset=["risk_off_20"]).iloc[-1]
        self.assertAlmostEqual(
            row["risk_off_20"], row["gold_ret_20"] - row["topix_ret_20"], places=9)

    def test_market_features_are_not_ranked(self):
        """全銘柄共通の値を日付内で順位化しても情報にならない。"""
        import features as F
        for c in F.GROUPS["market"]:
            self.assertNotIn(c, F.RAW_FOR_RANK, c)


class TestIndexIdentification(unittest.TestCase):
    """
    指数コードは名称を返さないので、業種との対応は相関で決めるしかない。
    1位だけを見て決めると、業種どうしがもともと相関するぶんを
    「一致した」と読み違える。
    """

    @staticmethod
    def _frames():
        import numpy as np
        rng = np.random.default_rng(0)
        days = pd.bdate_range("2020-01-01", periods=600)
        a = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        b = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        sec = pd.DataFrame({"A": a, "B": b, "ALL": (a + b) / 2})
        idx = pd.DataFrame({"X": a + rng.normal(0, 0.001, len(days))}, index=days)
        return idx, sec

    def test_picks_the_matching_sector_and_reports_the_runner_up(self):
        import identify_indices as I
        idx, sec = self._frames()
        (r,) = I.match(idx, sec, min_days=100)
        self.assertEqual(r["best"], "A")
        self.assertGreater(r["corr"], 0.9)
        self.assertIn(r["second"], ("ALL", "B"))
        self.assertTrue(r["confident"])

    def test_a_close_second_is_not_confident(self):
        """2位と僅差なら「どちらか分からない」。○にしてはいけない。"""
        import identify_indices as I
        idx, sec = self._frames()
        sec = sec.copy()
        sec["A2"] = sec["A"]          # 同じ系列を2本置く
        (r,) = I.match(idx, sec, min_days=100)
        self.assertAlmostEqual(r["gap"], 0.0, places=6)
        self.assertFalse(r["confident"])

    def test_sector_codes_sort_numerically(self):
        """
        文字列のまま並べると 1, 10, 11, ... 17, 2, 3 になる。
        S33 は全部4桁なので影響しないが、S17 は 1〜17 の1〜2桁で崩れる。
        実際これで S17 の検証が 1/17 しか当たらず、
        「仮説が外れた」と読み違えるところだった。
        """
        import identify_indices as I
        got = sorted(["1", "10", "17", "2", "9"], key=I._sort_key)
        self.assertEqual(got, ["1", "2", "9", "10", "17"])
        # 4桁ゼロ詰めは文字列順でも数値順と一致する（S33 が無事だった理由）
        s33 = ["0050", "1050", "3300", "9050"]
        self.assertEqual(sorted(s33, key=I._sort_key), sorted(s33))

    def test_hex_runs_splits_on_gaps(self):
        """
        指数コードは16進の連番。10進で数えると 0039 の次が 0040 になり、
        003A〜003F を飛ばしたことに気づけない。
        """
        import identify_indices as I
        runs = I._hex_runs(["0040", "0041", "0042", "0048", "0049", "zzzz"])
        self.assertEqual(runs, [["0040", "0041", "0042"], ["0048", "0049"]])

    def test_offset_only_uses_runs_of_the_right_length(self):
        """
        業種の数と長さが違う並びに当てはめると、起点をずらせば
        どこかは当たってしまう。長さが一致する並びだけを対象にする。
        """
        import numpy as np
        import identify_indices as I
        days = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        b = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        sec = pd.DataFrame({"1050": a, "2050": b, "ALL": (a + b) / 2})
        # 業種は2つ。長さ2の連番だけが対象になり、長さ3の並びは無視される
        idx = pd.DataFrame({"0040": a, "0041": b, "0042": a}, index=days)
        out = I.test_offset(idx, sec, "S33")
        self.assertEqual(out["nSectors"], 2)
        self.assertEqual(out["runs"], [])

    def test_offset_reports_rank_not_just_correlation(self):
        """
        相関が低い業種でも、仮説が指す業種が1位なら当たっている。
        相関だけ見ると「弱い業種は同定できない」と誤って切り捨てる。
        """
        import numpy as np
        import identify_indices as I
        days = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        b = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
        sec = pd.DataFrame({"1050": a, "2050": b, "ALL": (a + b) / 2})
        idx = pd.DataFrame({"0040": a, "0041": b}, index=days)
        out = I.test_offset(idx, sec, "S33")
        (run,) = out["runs"]
        self.assertEqual(run["rank1"], 2)
        self.assertEqual([p["rank"] for p in run["pairs"]], [1, 1])

    def test_market_code_is_not_treated_as_a_name(self):
        """
        master_hist の Mkt は数値コード（実測 101〜113）であって名称ではない。
        名称と決めつけて絞り込み、1,000万行が0行になった。
        """
        import identify_indices as I
        self.assertNotIn("Mkt", I.MARKET_NAME_COLS)
        for c in I.MARKET_NAME_COLS:
            self.assertTrue(c.endswith("Nm") or c.endswith("Name"), c)

    def test_too_few_days_is_not_matched(self):
        """日数が足りない指数に業種を割り当てると、偶然の相関を拾う。"""
        import identify_indices as I
        idx, sec = self._frames()
        (r,) = I.match(idx.iloc[:50], sec, min_days=100)
        self.assertNotIn("best", r)
        self.assertEqual(r["note"], "日数不足")


class TestSectorIndexFeatures(unittest.TestCase):
    """
    業種指数は market グループと違い、日付内で銘柄ごとに値が変わる。
    そこが崩れると、この特徴量を足す意味そのものが無くなる。
    """

    @staticmethod
    def _indices(codes, n=200, start="2020-01-01"):
        d0 = dt.date.fromisoformat(start)
        dates = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]
        rows = []
        for k, code in enumerate(codes):
            for i, d in enumerate(dates):
                rows.append({"Date": d, "Code": code,
                             "C": 100.0 * (1.0 + 0.001 * (k + 1)) ** i})
        return pd.DataFrame(rows)

    @staticmethod
    def _samples(s33_list, date="2020-06-01"):
        return pd.DataFrame({
            "Date": [pd.Timestamp(date)] * len(s33_list),
            "Code": [f"{i:05d}" for i in range(len(s33_list))],
            "S33": s33_list,
            "ret_20d": [5.0] * len(s33_list),
            "topix_ret_20": [1.0] * len(s33_list),
        })

    def test_mapping_covers_every_sector_exactly_once(self):
        """業種と指数が1対1でなければ、どこかの業種に別の指数が付く。"""
        self.assertEqual(len(SECTOR_INDEX), 33)
        self.assertEqual(len({ix for ix, _, _ in SECTOR_INDEX}), 33)
        self.assertEqual(len({s33 for _, s33, _ in SECTOR_INDEX}), 33)
        self.assertEqual(len(S33_TO_INDEX), 33)

    def test_index_codes_are_a_consecutive_hex_run(self):
        """
        同定した規則は「16進の連番」。10進で書くと 0039 の次が 0040 になり、
        003A〜003F を飛ばしたことに気づけない。
        """
        vals = sorted(int(ix, 16) for ix, _, _ in SECTOR_INDEX)
        self.assertEqual(vals, list(range(vals[0], vals[0] + 33)))

    def test_values_differ_within_a_date(self):
        """
        同じ日でも業種が違えば違う値になること。
        ここが定数になっていたら market グループと同じで、
        日付内の順位付けには効かない。
        """
        codes = [S33_TO_INDEX["1050"], S33_TO_INDEX["3650"], S33_TO_INDEX["7050"]]
        out = attach_sector_index(self._samples(["1050", "3650", "7050"]),
                                  self._indices(codes))
        self.assertEqual(out["sector_ret_20"].nunique(), 3)
        self.assertEqual(out["rel_sector_20"].nunique(), 3)

    def test_relative_is_the_difference(self):
        codes = [S33_TO_INDEX["1050"], S33_TO_INDEX["3650"]]
        out = attach_sector_index(self._samples(["1050", "3650"]),
                                  self._indices(codes))
        for _, r in out.iterrows():
            self.assertAlmostEqual(r["rel_sector_20"],
                                   r["ret_20d"] - r["sector_ret_20"], places=9)
            self.assertAlmostEqual(r["sector_vs_topix_20"],
                                   r["sector_ret_20"] - r["topix_ret_20"], places=9)

    def test_unknown_sector_becomes_missing_not_zero(self):
        """
        知らない業種に 0 を当てると「業種が横ばい」という意味になる。
        欠測は欠測のままにする。
        """
        codes = [S33_TO_INDEX["1050"]]
        out = attach_sector_index(self._samples(["1050", "9999"]),
                                  self._indices(codes))
        self.assertTrue(pd.notna(out["sector_ret_20"].iloc[0]))
        self.assertTrue(pd.isna(out["sector_ret_20"].iloc[1]))
        self.assertTrue(pd.isna(out["rel_sector_20"].iloc[1]))

    def test_missing_indices_does_not_break_the_build(self):
        """指数を取れていない状態でも、その軸が欠測になるだけで通ること。"""
        out = attach_sector_index(self._samples(["1050", "3650"]),
                                  pd.DataFrame(columns=["Date", "Code", "C"]))
        for c in ("sector_ret_20", "sector_ret_120",
                  "rel_sector_20", "sector_vs_topix_20"):
            self.assertTrue(out[c].isna().all(), c)

    def test_levels_are_not_features(self):
        """水準そのものは入れない。TOPIX と同じ理由（年号とほぼ同義になる）。"""
        import features as F
        for c in F.GROUPS["sector_index"]:
            self.assertTrue(c.endswith(("_20", "_120")), c)

    def test_returns_use_no_forward_fill(self):
        """
        欠測日を前日値で埋めてから変化率を取ると、値の無い日をまたいだ
        ところで 0% のリターンが作られる。
        """
        code = S33_TO_INDEX["1050"]
        ix = self._indices([code], n=60)
        ix = ix.drop(index=ix.index[30:35])       # 途中の5日を落とす
        ret = sector_index_returns(ix)
        self.assertGreater(ret.notna().sum().sum(), 0)


class TestPopulationOrigin(unittest.TestCase):
    """
    母集団の制約を外して増えた行を、チャートで見られるように分類する。
    どの行が新しく入ったのかがずれると、目視検証の対象が変わってしまう。
    """

    @staticmethod
    def _ds(rows):
        import export_label_samples as E
        return pd.DataFrame(rows), E

    def test_classifies_each_row(self):
        df, E = self._ds([
            {"tv_ma20": 1.0, "fund_complete": 1.0},   # 従来からいた
            {"tv_ma20": 0.2, "fund_complete": 1.0},   # 流動性を緩めて入った
            {"tv_ma20": 1.0, "fund_complete": 0.0},   # 決算を緩めて入った
            {"tv_ma20": 0.2, "fund_complete": 0.0},   # 両方だが流動性を優先
        ])
        self.assertEqual(list(E.population_origin(df)),
                         ["base", "added_liq", "added_fund", "added_liq"])

    def test_boundary_belongs_to_the_old_population(self):
        """変更前の下限ちょうどは、変更前も残っていた行。"""
        import export_label_samples as E
        df = pd.DataFrame({"tv_ma20": [E.PREV_MIN_TRADING_VALUE],
                           "fund_complete": [1.0]})
        self.assertEqual(list(E.population_origin(df)), ["base"])

    def test_missing_turnover_is_not_counted_as_added(self):
        """
        売買代金が欠測の行を「流動性を緩めて入った」に入れると、
        増えた行の件数が水増しされる。NaN の比較は False なので base に残る。
        """
        import export_label_samples as E
        df = pd.DataFrame({"tv_ma20": [np.nan], "fund_complete": [1.0]})
        self.assertEqual(list(E.population_origin(df)), ["base"])

    def test_prev_floor_is_above_the_current_one(self):
        """
        比較の基準が現在の下限より下だと、増えた行が0件になって
        「増えていない」ように見える。
        """
        import build_dataset as B
        import export_label_samples as E
        if B.MIN_TRADING_VALUE is not None:
            self.assertGreater(E.PREV_MIN_TRADING_VALUE, B.MIN_TRADING_VALUE)


class TestFeatureDictCoversEveryGroup(unittest.TestCase):
    """
    グループを足して feature_dict.MACRO に登録し忘れると、
    辞書ページの生成が落ちる（実際に落ちた）。
    CI の後段ではなく、ここで気づけるようにする。
    """

    def test_every_group_is_assigned_to_a_macro_section(self):
        import feature_dict as FD
        import features as F
        used = {g for _, gs, _ in FD.MACRO for g in gs}
        known = {g for g in F.GROUPS if not g.endswith("_rank")}
        self.assertEqual(sorted(known - used), [],
                         "大区分に割り当てられていない中区分がある")

    def test_every_feature_has_a_description(self):
        import feature_dict as FD
        import features as F
        undocumented = [c for c in F.columns("all")
                        if FD.describe(c) == "（説明未登録）"]
        self.assertEqual(undocumented, [])


class TestPopulationFlags(unittest.TestCase):
    """
    母集団の制約（流動性の下限・決算の完全性）を外したぶん、
    どちらの群の行なのかをフラグで持たせる。
    """

    def test_cap_band_boundaries(self):
        """境界は「以上」で上の帯に入る。EDA の内訳と同じ切り方にする。"""
        s = pd.Series([1.0, 99.9, 100.0, 299.9, 300.0, 999.9, 1000.0,
                       2999.9, 3000.0, 500000.0])
        self.assertEqual(list(cap_band(s)), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_cap_band_keeps_missing_missing(self):
        """時価総額が無い行を最小の帯に落とすと、規模の分布が歪む。"""
        s = pd.Series([np.nan, 50.0, np.nan])
        out = cap_band(s)
        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertEqual(out.iloc[1], 0)
        self.assertTrue(pd.isna(out.iloc[2]))

    def test_cap_band_has_one_more_band_than_edges(self):
        self.assertEqual(int(cap_band(pd.Series([1e9])).iloc[0]),
                         len(CAP_BAND_EDGES))

    def test_fund_complete_marks_exactly_the_rows_the_filter_would_keep(self):
        """
        fund_complete は「full4_sym で絞ったら残る行」と一致していないと、
        絞る／絞らないの比較ができなくなる。
        """
        cols = FUND_REQUIREMENT_SETS["full4_sym"]
        df = pd.DataFrame({c: [1.0, 1.0, 1.0] for c in cols})
        df.loc[1, cols[0]] = np.nan          # 1列でも欠けたら不成立
        df.loc[2, cols[-1]] = np.nan
        flag = fund_complete_flag(df)
        self.assertEqual(list(flag), [1.0, 0.0, 0.0])
        kept = df[df[cols].notna().all(axis=1)]
        self.assertEqual(len(kept), int(flag.sum()))

    def test_fund_complete_fails_loudly_when_material_is_missing(self):
        """材料が無いのに黙って全0を返すと、母集団の比較が壊れる。"""
        with self.assertRaises(SystemExit):
            fund_complete_flag(pd.DataFrame({"x": [1.0]}))

    def test_no_cap_preset_also_drops_the_band(self):
        """
        all_no_cap は「規模を抜いたら何が残るか」の対照実験。
        log_market_cap だけ抜いても帯が残っていては規模が別口から戻る。
        """
        import features as F
        cols = F.columns("all_no_cap")
        self.assertNotIn("log_market_cap", cols)
        self.assertNotIn("cap_band", cols)
        self.assertIn("cap_band", F.columns("all"))

    def test_scale_flags_are_not_ranked(self):
        """帯の日付内順位は時価総額の順位と同じものになる。二重に持たない。"""
        import features as F
        for c in F.GROUPS["scale"]:
            self.assertNotIn(c, F.RAW_FOR_RANK, c)


class TestSweepOverride(unittest.TestCase):
    """
    掃引が母集団の定義を差し替える口。既定の挙動を変えないことが要件。

    ここが黙って効くと、気づかないまま別の母集団で学習してしまう。
    「効いていないこと」と「効いたら必ず記録が残ること」の両方を固定する。
    """

    def test_no_env_means_no_change(self):
        """環境変数が無ければ既定値のまま、記録も空。"""
        import build_dataset as B
        self.assertEqual(B.HIGH_WINDOW, 368)
        self.assertEqual(B.BREAKOUT_COOLDOWN, 20)
        self.assertEqual(B.MIN_TRADING_VALUE, 0.1)
        self.assertEqual(B.SWEEP_OVERRIDES, {})

    def test_env_overrides_and_is_recorded(self):
        """効いたときは値が変わり、SWEEP_OVERRIDES に残る。"""
        import build_dataset as B
        B.SWEEP_OVERRIDES.clear()
        os.environ["SWEEP_TESTKEY"] = "245"
        try:
            self.assertEqual(B._sweep_override("TESTKEY", 368, int), 245)
            self.assertEqual(B.SWEEP_OVERRIDES, {"TESTKEY": 245})
        finally:
            os.environ.pop("SWEEP_TESTKEY", None)
            B.SWEEP_OVERRIDES.clear()

    def test_blank_env_is_treated_as_absent(self):
        """空文字は「指定なし」。CI で未設定の変数が空で入ることがある。"""
        import build_dataset as B
        B.SWEEP_OVERRIDES.clear()
        os.environ["SWEEP_TESTKEY"] = "  "
        try:
            self.assertEqual(B._sweep_override("TESTKEY", 368, int), 368)
            self.assertEqual(B.SWEEP_OVERRIDES, {})
        finally:
            os.environ.pop("SWEEP_TESTKEY", None)

    def test_label_config_follows_the_override(self):
        """
        DEFAULT_LABEL はクラス定義時に既定値を焼き込むので、
        差し替えが LabelConfig まで届いているかを別に確かめる。
        ここが届いていないと、環境変数を入れても同じ母集団のまま回る。
        """
        import build_dataset as B
        self.assertEqual(B.DEFAULT_LABEL.high_window, B.HIGH_WINDOW)


class TestSweepDesign(unittest.TestCase):
    """
    設計の掃引（research/sweep_design.py）。

    比較の土台が崩れると結果が全部無意味になるので、
    「1因子だけ動く」「評価期間が共通」「エンバーゴが設計に追随する」
    「物差しがラベルに依存しない」の4点を固定する。
    """

    def setUp(self):
        import sweep_design as S
        self.S = S

    def test_each_variant_moves_exactly_one_factor(self):
        """基準から2つ以上動いていたら、どちらが効いたか分からなくなる。"""
        S = self.S
        fields = [f for f in S.BASE.__dataclass_fields__
                  if f not in ("key", "axis", "label")]
        for d in S.DESIGNS:
            if d.key == "base":
                continue
            diff = [f for f in fields if getattr(d, f) != getattr(S.BASE, f)]
            self.assertEqual(len(diff), 1,
                             f"{d.key} が動かした因子: {diff}")

    def test_design_keys_are_unique(self):
        keys = [d.key for d in self.S.DESIGNS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_embargo_follows_the_design_horizon(self):
        """
        ホライズンを伸ばした設計は、エンバーゴも同じだけ伸びること。
        伸びないと訓練末尾のラベルが評価期間の値動きで決まる（リーク）。
        """
        S = self.S
        ts = pd.Timestamp("2025-06-16")
        short = S.train_end_for(S._var("h40", "x", "x", horizon=40), ts)
        base = S.train_end_for(S.BASE, ts)
        long_ = S.train_end_for(S._var("h120", "x", "x", horizon=120), ts)
        self.assertGreater(short, base)
        self.assertLess(long_, base)
        # 60営業日 × 1.45（train_model.TRADING_TO_CALENDAR）≒ 87暦日。
        # 換算を train_model から取ると、そこが変わったとき無言で追随して
        # しまう。学習側と揃っていることを確かめたいので数値で固定する
        self.assertEqual((ts - base).days, 87)

    def test_reference_horizon_is_fixed(self):
        """
        物差しの参照ホライズンは設計から独立。ここが設計に追随すると
        「ラベルを変えても意味が変わらない物差し」でなくなる。
        """
        S = self.S
        self.assertEqual(S.REF_RISE.horizon, S.REF_HORIZON)
        self.assertEqual(S.REF_RISE.keep_days, 0)
        self.assertIsNone(S.REF_RISE.end_ratio)
        self.assertFalse(S.REF_RISE.require_uptrend)
        # ホライズンを動かす設計があっても REF は動かない
        moved = [d for d in S.DESIGNS if d.horizon != S.BASE.horizon]
        self.assertTrue(moved, "ホライズンを動かす設計が無い")
        self.assertEqual(S.REF_RISE.horizon, S.BASE.horizon)

    def test_outcome_stats_ignores_the_label(self):
        """物差しはラベルを見ない。ラベルを反転しても値が変わらないこと。"""
        S = self.S
        df = pd.DataFrame({
            "label": [1.0, 0.0, 1.0, 0.0],
            "ref_end": [0.10, -0.05, 0.20, 0.00],
            "ref_rise": [0.30, 0.02, 0.40, 0.10],
        })
        a = S.outcome_stats(df)
        b = S.outcome_stats(df.assign(label=1.0 - df["label"]))
        self.assertEqual(a, b)
        self.assertEqual(a["n"], 4)
        self.assertAlmostEqual(a["end_median"], 5.0)
        self.assertAlmostEqual(a["win_rate"], 0.5)   # 0.00 は勝ちに数えない

    def test_fresh_break_matches_mark_new_highs(self):
        """
        クールダウンを付け替える関数が、本体と同じ式であること。
        ずれると母集団が静かに変わり、比較の土台が崩れる。
        """
        S = self.S
        closes = [100 + i for i in range(60)]
        bars = make_bars(closes, start="2020-01-01")
        panel = price_panel(bars, LabelConfig(high_window=5))
        panel = mark_new_highs(panel, cooldown=20, on_high=True)
        got = S.fresh_break(panel, 20).to_numpy()
        np.testing.assert_array_equal(got, panel["is_fresh_break"].to_numpy())

    def test_edge_ci_straddles_zero_when_there_is_no_edge(self):
        """
        スコアと実際のリターンが無関係なら、区間は0をまたぐこと。
        またがない実装だと、選べていないのに「選んだ意味があった」と読める。
        """
        S = self.S
        rng = np.random.default_rng(0)
        n = 3000
        dates = (pd.to_datetime("2025-06-16")
                 + pd.to_timedelta(rng.integers(0, 250, n), unit="D"))
        t = pd.DataFrame({"Date": dates, "score": rng.normal(size=n),
                          "ref_end": rng.normal(0.03, 0.25, n)})
        lo, hi = S.edge_ci(t, n_boot=300, seed=0)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_edge_ci_excludes_zero_when_the_edge_is_real(self):
        """スコアが本当にリターンを当てているなら、区間は0を含まないこと。"""
        S = self.S
        rng = np.random.default_rng(1)
        n = 3000
        dates = (pd.to_datetime("2025-06-16")
                 + pd.to_timedelta(rng.integers(0, 250, n), unit="D"))
        sc = rng.normal(size=n)
        t = pd.DataFrame({"Date": dates, "score": sc,
                          "ref_end": 0.03 + 0.15 * sc + rng.normal(0, 0.15, n)})
        lo, hi = S.edge_ci(t, n_boot=300, seed=0)
        self.assertGreater(lo, 0.0)

    def test_longer_cooldown_is_a_subset(self):
        """
        クールダウンを伸ばすと母集団は必ず狭くなる（部分集合）。
        だからデータセットを作り直さずに行を絞るだけでよい。
        """
        S = self.S
        closes = [100 + (i % 7) + i * 0.5 for i in range(120)]
        bars = make_bars(closes, start="2020-01-01")
        panel = price_panel(bars, LabelConfig(high_window=10))
        panel = mark_new_highs(panel, cooldown=5, on_high=True)
        short = S.fresh_break(panel, 5).to_numpy()
        long_ = S.fresh_break(panel, 40).to_numpy()
        self.assertTrue(short[long_].all(),
                        "クールダウン40の新規ブレイクが5の部分集合になっていない")


if __name__ == "__main__":
    unittest.main(verbosity=2)
