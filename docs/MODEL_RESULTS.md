# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 52週 / 1〜6ヶ月 / 定着20日-8% +60日後0%
- データセット: 142,000サンプル / 全体の正例率 **11.19%**
- 期間: 2017-09-29 〜 2025-11-28 / 銘柄数 3,855
- 分割: 訓練 〜2023-01-01 / 検証 2023-01-01〜2024-10-01 / テスト 2024-10-01〜
- **エンバーゴ 180営業日**（ラベル確定に必要な将来日数から自動導出）

  - train: 71,629件 / 2017-09-29 〜 2022-03-31 / 正例率 6.19%
  - val: 18,462件 / 2023-01-31 〜 2023-12-29 / 正例率 19.63%
  - test: 22,893件 / 2024-10-31 〜 2025-11-28 / 正例率 21.66%


## 検証データ (val)

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ベースライン: R_high のみ | 0.3481 | 0.7353 | 37.5% | 38.6% | 1.96x |
| LightGBM [price_only] | 0.3435 | 0.7150 | 45.7% | 43.4% | 2.21x |
| LightGBM [rank_price_only] | 0.3273 | 0.6990 | 42.4% | 38.8% | 1.98x |
| ロジスティック回帰 [rank_price_only] | 0.3194 | 0.7100 | 31.5% | 33.9% | 1.73x |
| ロジスティック回帰 [rank_technical] | 0.3124 | 0.7008 | 29.9% | 32.6% | 1.66x |
| LightGBM [technical] | 0.3071 | 0.6631 | 33.7% | 38.5% | 1.96x |
| ロジスティック回帰 [technical] | 0.2826 | 0.6868 | 15.8% | 23.1% | 1.18x |
| ロジスティック回帰 [price_only] | 0.2785 | 0.6825 | 15.8% | 21.7% | 1.10x |
| ロジスティック回帰 [fund_simple] | 0.2718 | 0.6743 | 14.7% | 21.6% | 1.10x |
| ロジスティック回帰 [raw_and_rank] | 0.2652 | 0.6534 | 18.5% | 23.4% | 1.19x |
| LightGBM [rank_technical] | 0.2642 | 0.6099 | 29.3% | 31.3% | 1.60x |
| LightGBM [rank_all] | 0.2552 | 0.6004 | 29.3% | 29.8% | 1.52x |
| LightGBM [all_no_market] | 0.2536 | 0.6166 | 25.5% | 26.1% | 1.33x |
| LightGBM [fund_simple] | 0.2531 | 0.5929 | 29.3% | 30.1% | 1.53x |
| LightGBM [all] | 0.2522 | 0.6028 | 28.8% | 30.2% | 1.54x |
| LightGBM [raw_and_rank] | 0.2518 | 0.5911 | 32.6% | 29.6% | 1.51x |
| ロジスティック回帰 [rank_all] | 0.2515 | 0.6162 | 20.1% | 22.8% | 1.16x |
| ロジスティック回帰 [all] | 0.2358 | 0.6171 | 12.0% | 16.1% | 0.82x |
| ロジスティック回帰 [all_no_market] | 0.2282 | 0.6047 | 10.3% | 15.9% | 0.81x |
| ベースライン: 出来高モメンタムのみ | 0.2133 | 0.5560 | 15.2% | 17.7% | 0.90x |
| ベースライン: 既存の8軸総合スコア | 0.2130 | 0.5485 | 22.8% | 19.8% | 1.01x |
| ロジスティック回帰 [fundamental] | 0.1914 | 0.5024 | 12.5% | 14.5% | 0.74x |
| LightGBM [rank_fundamental] | 0.1819 | 0.4605 | 13.6% | 17.3% | 0.88x |
| LightGBM [rank_fundamental_v2] | 0.1804 | 0.4529 | 17.4% | 16.7% | 0.85x |
| LightGBM [fundamental] | 0.1795 | 0.4670 | 8.2% | 13.8% | 0.70x |
| ロジスティック回帰 [rank_fundamental] | 0.1774 | 0.4684 | 11.4% | 13.7% | 0.70x |
| LightGBM [fundamental_v3] | 0.1719 | 0.4471 | 19.0% | 11.5% | 0.59x |
| ロジスティック回帰 [fundamental_v3] | 0.1710 | 0.4350 | 5.4% | 20.3% | 1.03x |
| ロジスティック回帰 [fundamental_v2] | 0.1685 | 0.4436 | 5.4% | 9.8% | 0.50x |
| LightGBM [fundamental_v2] | 0.1666 | 0.4353 | 7.6% | 11.4% | 0.58x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1620 | 0.4220 | 4.3% | 10.3% | 0.52x |
| LightGBM [extras_only] | 0.1604 | 0.4123 | 8.2% | 8.3% | 0.42x |
| ロジスティック回帰 [extras_only] | 0.1601 | 0.3833 | 3.8% | 21.3% | 1.09x |
| LightGBM [valuation_only] | 0.1409 | 0.3352 | 8.7% | 7.6% | 0.39x |
| ロジスティック回帰 [valuation_only] | 0.1384 | 0.3293 | 4.3% | 4.3% | 0.22x |

（正例率 = 19.63% / n = 18,462）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [rank_technical] | 0.3442 | 0.6911 | 36.4% | 39.6% | 1.83x |
| LightGBM [price_only] | 0.3432 | 0.6896 | 40.8% | 39.0% | 1.80x |
| ロジスティック回帰 [rank_price_only] | 0.3413 | 0.6962 | 36.4% | 39.3% | 1.82x |
| ベースライン: R_high のみ | 0.3379 | 0.6839 | 44.7% | 39.0% | 1.80x |
| LightGBM [rank_price_only] | 0.3356 | 0.6791 | 39.5% | 40.2% | 1.86x |
| ロジスティック回帰 [price_only] | 0.3258 | 0.6851 | 26.8% | 36.5% | 1.69x |
| ロジスティック回帰 [technical] | 0.3226 | 0.6879 | 23.2% | 33.1% | 1.53x |
| LightGBM [technical] | 0.3208 | 0.6477 | 47.8% | 39.7% | 1.83x |
| ロジスティック回帰 [fund_simple] | 0.3159 | 0.6817 | 19.3% | 31.8% | 1.47x |
| ロジスティック回帰 [raw_and_rank] | 0.3074 | 0.6696 | 22.4% | 28.2% | 1.30x |
| LightGBM [all_no_market] | 0.3032 | 0.6391 | 39.0% | 34.9% | 1.61x |
| LightGBM [raw_and_rank] | 0.2995 | 0.6135 | 44.3% | 36.7% | 1.70x |
| LightGBM [rank_technical] | 0.2984 | 0.6148 | 36.0% | 35.4% | 1.63x |
| ロジスティック回帰 [rank_all] | 0.2910 | 0.6305 | 26.8% | 31.2% | 1.44x |
| LightGBM [fund_simple] | 0.2891 | 0.6060 | 39.5% | 35.3% | 1.63x |
| LightGBM [all] | 0.2886 | 0.6077 | 37.7% | 34.4% | 1.59x |
| ロジスティック回帰 [all] | 0.2831 | 0.6406 | 19.3% | 26.3% | 1.21x |
| LightGBM [rank_all] | 0.2830 | 0.5955 | 32.9% | 35.8% | 1.65x |
| ロジスティック回帰 [all_no_market] | 0.2793 | 0.6333 | 18.9% | 27.6% | 1.28x |
| ベースライン: 出来高モメンタムのみ | 0.2348 | 0.5474 | 16.2% | 22.3% | 1.03x |
| ベースライン: 既存の8軸総合スコア | 0.2333 | 0.5380 | 23.2% | 23.5% | 1.09x |
| LightGBM [fundamental_v3] | 0.2252 | 0.5167 | 23.2% | 21.2% | 0.98x |
| LightGBM [fundamental] | 0.2209 | 0.5110 | 23.7% | 20.7% | 0.96x |
| ロジスティック回帰 [fundamental_v3] | 0.2190 | 0.4882 | 10.5% | 28.8% | 1.33x |
| LightGBM [fundamental_v2] | 0.2131 | 0.5061 | 18.4% | 18.1% | 0.84x |
| ロジスティック回帰 [fundamental] | 0.2122 | 0.5107 | 21.5% | 15.8% | 0.73x |
| ロジスティック回帰 [extras_only] | 0.2110 | 0.4512 | 13.2% | 30.9% | 1.42x |
| LightGBM [rank_fundamental] | 0.2071 | 0.4768 | 19.3% | 18.4% | 0.85x |
| ロジスティック回帰 [rank_fundamental] | 0.2069 | 0.4992 | 16.7% | 15.4% | 0.71x |
| LightGBM [extras_only] | 0.2054 | 0.4693 | 19.3% | 17.8% | 0.82x |
| LightGBM [rank_fundamental_v2] | 0.2009 | 0.4522 | 20.6% | 20.3% | 0.94x |
| ロジスティック回帰 [fundamental_v2] | 0.1987 | 0.4778 | 11.0% | 15.5% | 0.71x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1980 | 0.4732 | 12.7% | 15.7% | 0.73x |
| LightGBM [valuation_only] | 0.1795 | 0.4077 | 21.5% | 14.4% | 0.67x |
| ロジスティック回帰 [valuation_only] | 0.1692 | 0.3879 | 7.0% | 9.5% | 0.44x |

（正例率 = 21.66% / n = 22,893）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **ベースライン: R_high のみ**（テスト PR-AUC 0.3379）。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| ロジスティック回帰 [rank_technical] | 0.3442 | +0.0063 | [-0.0005, +0.0124] | 0.963 | 誤差 |
| LightGBM [price_only] | 0.3432 | +0.0053 | [+0.0002, +0.0100] | 0.979 | 有意 |
| ロジスティック回帰 [rank_price_only] | 0.3413 | +0.0034 | [-0.0032, +0.0097] | 0.832 | 誤差 |
| LightGBM [rank_price_only] | 0.3356 | -0.0023 | [-0.0062, +0.0017] | 0.117 | 誤差 |
| ロジスティック回帰 [price_only] | 0.3258 | -0.0121 | [-0.0204, -0.0042] | 0.002 | 有意に劣る |
| ロジスティック回帰 [technical] | 0.3226 | -0.0153 | [-0.0244, -0.0060] | 0.000 | 有意に劣る |
| LightGBM [technical] | 0.3208 | -0.0171 | [-0.0281, -0.0065] | 0.001 | 有意に劣る |
| LightGBM [rank_technical] | 0.2984 | -0.0395 | [-0.0497, -0.0296] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_all] | 0.2910 | -0.0468 | [-0.0546, -0.0388] | 0.000 | 有意に劣る |
| LightGBM [all] | 0.2886 | -0.0493 | [-0.0597, -0.0386] | 0.000 | 有意に劣る |
| ロジスティック回帰 [all] | 0.2831 | -0.0548 | [-0.0640, -0.0455] | 0.000 | 有意に劣る |
| LightGBM [rank_all] | 0.2830 | -0.0549 | [-0.0660, -0.0444] | 0.000 | 有意に劣る |
| LightGBM [fundamental] | 0.2209 | -0.1170 | [-0.1281, -0.1060] | 0.000 | 有意に劣る |
| ロジスティック回帰 [fundamental] | 0.2122 | -0.1257 | [-0.1366, -0.1156] | 0.000 | 有意に劣る |
| LightGBM [rank_fundamental] | 0.2071 | -0.1308 | [-0.1419, -0.1197] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_fundamental] | 0.2069 | -0.1309 | [-0.1413, -0.1205] | 0.000 | 有意に劣る |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.3442 | 1.83x |
| `price_only` | 3 | price | 0.3432 | 1.80x |
| `rank_price_only` | 3 | price_rank | 0.3413 | 1.82x |
| `technical` | 9 | price + volume + liquidity + supply + market | 0.3226 | 1.53x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.3159 | 1.47x |
| `raw_and_rank` | 146 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.3074 | 1.30x |
| `all_no_market` | 137 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.3032 | 1.61x |
| `rank_all` | 139 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + sector + turnaround + market | 0.2910 | 1.44x |
| `all` | 139 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + market | 0.2886 | 1.59x |
| `fundamental_v3` | 130 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.2252 | 0.98x |
| `fundamental` | 105 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.2209 | 0.96x |
| `fundamental_v2` | 126 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + turnaround | 0.2131 | 0.84x |
| `extras_only` | 23 | valuation + dividend + cashflow + efficiency + guidance + sector | 0.2110 | 1.42x |
| `rank_fundamental` | 105 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.2071 | 0.85x |
| `rank_fundamental_v2` | 126 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + turnaround | 0.2009 | 0.94x |
| `valuation_only` | 7 | valuation | 0.1795 | 0.67x |

## 特徴量の寄与（`rank_technical` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `r_high_r` | +0.926 | ブレイクしやすい |
| `topix_ret_120` | +0.184 | ブレイクしやすい |
| `log_market_cap_r` | -0.179 | しにくい |
| `r_high_6m_r` | -0.176 | しにくい |
| `log_trading_value_r` | +0.156 | ブレイクしやすい |
| `r_high_3m_r` | -0.150 | しにくい |
| `topix_ret_20` | -0.146 | しにくい |
| `credit_ratio_r` | +0.107 | ブレイクしやすい |
| `volume_trend_r` | +0.002 | ブレイクしやすい |

## 読み方

- **PR-AUC** が主指標。下限は正例率で、それを大きく上回るほど良い
- **Lift@5%** は「スコア上位5%の正例率 ÷ 全体の正例率」。1.0 なら無意味
- ベースライン（`R_high` 単独など）を上回らなければ、**追加の特徴量に予測力が無い**という結論になる
- `all` と `all_no_market` の差が、**モデルが相場局面をどれだけ暗記していたか**の目安になる
- 数字が極端に良い場合はまずリークを疑う（`tests/test_dataset.py` の先読み検出テストを参照）

## 特徴量を足して試すには

1. `research/features.py` の `GROUPS` に列を足す
2. `research/build_dataset.py` でその列を作る
3. データセットを再構築（Release から読むので数分・API取得なし）
4. `PRESETS` にセットを1行足して再学習
