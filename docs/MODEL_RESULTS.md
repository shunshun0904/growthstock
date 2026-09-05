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
| LightGBM [price_only] | 0.3446 | 0.7122 | 44.6% | 41.9% | 2.14x |
| ロジスティック回帰 [rank_price_only] | 0.3194 | 0.7100 | 31.5% | 33.9% | 1.73x |
| ロジスティック回帰 [rank_technical] | 0.3124 | 0.7008 | 29.9% | 32.6% | 1.66x |
| LightGBM [rank_technical] | 0.3106 | 0.6333 | 42.4% | 39.3% | 2.00x |
| LightGBM [rank_price_only] | 0.3087 | 0.6765 | 40.2% | 36.9% | 1.88x |
| LightGBM [technical] | 0.2941 | 0.6461 | 33.7% | 35.4% | 1.80x |
| ロジスティック回帰 [technical] | 0.2826 | 0.6868 | 15.8% | 23.1% | 1.18x |
| ロジスティック回帰 [price_only] | 0.2785 | 0.6825 | 15.8% | 21.7% | 1.10x |
| ロジスティック回帰 [fund_simple] | 0.2735 | 0.6755 | 15.8% | 21.3% | 1.09x |
| ロジスティック回帰 [raw_and_rank] | 0.2653 | 0.6545 | 20.1% | 22.8% | 1.16x |
| LightGBM [rank_all] | 0.2648 | 0.6108 | 27.2% | 31.6% | 1.61x |
| ロジスティック回帰 [rank_all] | 0.2563 | 0.6254 | 19.6% | 24.2% | 1.23x |
| LightGBM [fund_simple] | 0.2522 | 0.5930 | 32.1% | 29.4% | 1.50x |
| LightGBM [raw_and_rank] | 0.2480 | 0.5819 | 29.3% | 29.9% | 1.52x |
| ロジスティック回帰 [all] | 0.2408 | 0.6262 | 12.0% | 16.4% | 0.83x |
| LightGBM [all] | 0.2354 | 0.5791 | 23.9% | 25.1% | 1.28x |
| ロジスティック回帰 [all_no_market] | 0.2345 | 0.6178 | 9.8% | 15.3% | 0.78x |
| LightGBM [all_no_market] | 0.2330 | 0.5830 | 25.5% | 22.9% | 1.16x |
| ベースライン: 出来高モメンタムのみ | 0.2133 | 0.5560 | 15.2% | 17.7% | 0.90x |
| ベースライン: 既存の8軸総合スコア | 0.2130 | 0.5486 | 22.8% | 19.8% | 1.01x |
| LightGBM [rank_fundamental] | 0.1979 | 0.4986 | 16.3% | 20.6% | 1.05x |
| ロジスティック回帰 [fundamental] | 0.1930 | 0.5031 | 13.6% | 15.5% | 0.79x |
| LightGBM [rank_fundamental_v2] | 0.1834 | 0.4534 | 13.6% | 23.6% | 1.20x |
| LightGBM [fundamental] | 0.1804 | 0.4728 | 10.9% | 13.0% | 0.66x |
| ロジスティック回帰 [rank_fundamental] | 0.1774 | 0.4680 | 10.9% | 13.7% | 0.70x |
| LightGBM [fundamental_v2] | 0.1619 | 0.4225 | 13.0% | 10.4% | 0.53x |
| ロジスティック回帰 [fundamental_v2] | 0.1615 | 0.4237 | 6.0% | 7.9% | 0.40x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1564 | 0.4077 | 3.8% | 7.0% | 0.36x |
| LightGBM [valuation_only] | 0.1472 | 0.3629 | 8.2% | 10.1% | 0.51x |
| ロジスティック回帰 [valuation_only] | 0.1374 | 0.3243 | 4.9% | 3.8% | 0.19x |

（正例率 = 19.63% / n = 18,462）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [rank_technical] | 0.3442 | 0.6911 | 36.4% | 39.6% | 1.83x |
| LightGBM [price_only] | 0.3414 | 0.6879 | 35.1% | 40.7% | 1.88x |
| ロジスティック回帰 [rank_price_only] | 0.3413 | 0.6962 | 36.4% | 39.3% | 1.82x |
| ベースライン: R_high のみ | 0.3379 | 0.6839 | 44.7% | 39.0% | 1.80x |
| ロジスティック回帰 [price_only] | 0.3258 | 0.6851 | 26.8% | 36.5% | 1.69x |
| ロジスティック回帰 [technical] | 0.3226 | 0.6879 | 23.2% | 33.1% | 1.53x |
| LightGBM [technical] | 0.3220 | 0.6370 | 53.1% | 38.0% | 1.76x |
| LightGBM [rank_price_only] | 0.3186 | 0.6547 | 38.2% | 37.5% | 1.73x |
| LightGBM [rank_technical] | 0.3160 | 0.6260 | 43.4% | 38.9% | 1.80x |
| ロジスティック回帰 [fund_simple] | 0.3159 | 0.6812 | 19.7% | 31.9% | 1.47x |
| ロジスティック回帰 [raw_and_rank] | 0.3087 | 0.6709 | 23.2% | 29.3% | 1.35x |
| LightGBM [raw_and_rank] | 0.2945 | 0.6108 | 39.5% | 37.0% | 1.71x |
| ロジスティック回帰 [rank_all] | 0.2942 | 0.6344 | 25.9% | 32.0% | 1.48x |
| LightGBM [fund_simple] | 0.2931 | 0.6066 | 42.1% | 36.5% | 1.68x |
| ロジスティック回帰 [all] | 0.2858 | 0.6439 | 18.9% | 27.1% | 1.25x |
| LightGBM [all] | 0.2855 | 0.6075 | 39.5% | 33.7% | 1.55x |
| LightGBM [all_no_market] | 0.2850 | 0.6154 | 35.5% | 34.0% | 1.57x |
| ロジスティック回帰 [all_no_market] | 0.2802 | 0.6337 | 21.9% | 28.1% | 1.30x |
| LightGBM [rank_all] | 0.2778 | 0.5928 | 31.1% | 32.7% | 1.51x |
| ベースライン: 出来高モメンタムのみ | 0.2348 | 0.5474 | 16.2% | 22.3% | 1.03x |
| LightGBM [rank_fundamental] | 0.2340 | 0.5254 | 25.0% | 24.0% | 1.11x |
| ベースライン: 既存の8軸総合スコア | 0.2334 | 0.5382 | 23.2% | 23.5% | 1.09x |
| LightGBM [rank_fundamental_v2] | 0.2316 | 0.5106 | 25.4% | 27.3% | 1.26x |
| LightGBM [fundamental] | 0.2284 | 0.5260 | 18.0% | 22.3% | 1.03x |
| ロジスティック回帰 [fundamental] | 0.2118 | 0.5088 | 18.0% | 17.0% | 0.78x |
| LightGBM [fundamental_v2] | 0.2076 | 0.4883 | 20.2% | 17.6% | 0.81x |
| ロジスティック回帰 [rank_fundamental] | 0.2070 | 0.4990 | 16.7% | 15.5% | 0.71x |
| ロジスティック回帰 [fundamental_v2] | 0.1921 | 0.4585 | 11.4% | 13.8% | 0.64x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1915 | 0.4587 | 11.8% | 13.5% | 0.62x |
| LightGBM [valuation_only] | 0.1802 | 0.4186 | 15.4% | 14.5% | 0.67x |
| ロジスティック回帰 [valuation_only] | 0.1670 | 0.3807 | 7.0% | 9.4% | 0.43x |

（正例率 = 21.66% / n = 22,893）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **ベースライン: R_high のみ**（テスト PR-AUC 0.3379）。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| ロジスティック回帰 [rank_technical] | 0.3442 | +0.0063 | [-0.0005, +0.0137] | 0.969 | 誤差 |
| LightGBM [price_only] | 0.3414 | +0.0035 | [-0.0026, +0.0097] | 0.846 | 誤差 |
| ロジスティック回帰 [rank_price_only] | 0.3413 | +0.0034 | [-0.0028, +0.0100] | 0.850 | 誤差 |
| ロジスティック回帰 [price_only] | 0.3258 | -0.0121 | [-0.0208, -0.0030] | 0.003 | 有意に劣る |
| ロジスティック回帰 [technical] | 0.3226 | -0.0153 | [-0.0243, -0.0056] | 0.001 | 有意に劣る |
| LightGBM [technical] | 0.3220 | -0.0159 | [-0.0271, -0.0042] | 0.004 | 有意に劣る |
| LightGBM [rank_price_only] | 0.3186 | -0.0192 | [-0.0257, -0.0126] | 0.000 | 有意に劣る |
| LightGBM [rank_technical] | 0.3160 | -0.0218 | [-0.0303, -0.0132] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_all] | 0.2942 | -0.0437 | [-0.0514, -0.0361] | 0.000 | 有意に劣る |
| ロジスティック回帰 [all] | 0.2858 | -0.0521 | [-0.0612, -0.0428] | 0.000 | 有意に劣る |
| LightGBM [all] | 0.2855 | -0.0524 | [-0.0637, -0.0405] | 0.000 | 有意に劣る |
| LightGBM [rank_all] | 0.2778 | -0.0601 | [-0.0699, -0.0497] | 0.000 | 有意に劣る |
| LightGBM [rank_fundamental] | 0.2340 | -0.1039 | [-0.1164, -0.0927] | 0.000 | 有意に劣る |
| LightGBM [fundamental] | 0.2284 | -0.1095 | [-0.1210, -0.0983] | 0.000 | 有意に劣る |
| ロジスティック回帰 [fundamental] | 0.2118 | -0.1261 | [-0.1374, -0.1156] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_fundamental] | 0.2070 | -0.1309 | [-0.1423, -0.1206] | 0.000 | 有意に劣る |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.3442 | 1.83x |
| `price_only` | 3 | price | 0.3414 | 1.88x |
| `rank_price_only` | 3 | price_rank | 0.3413 | 1.82x |
| `technical` | 9 | price + volume + liquidity + supply + market | 0.3226 | 1.53x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.3159 | 1.47x |
| `raw_and_rank` | 146 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.3087 | 1.35x |
| `rank_all` | 120 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + turnaround + market | 0.2942 | 1.48x |
| `all` | 120 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + turnaround + market | 0.2858 | 1.25x |
| `all_no_market` | 118 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + turnaround | 0.2850 | 1.57x |
| `rank_fundamental` | 105 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.2340 | 1.11x |
| `rank_fundamental_v2` | 111 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + turnaround | 0.2316 | 1.26x |
| `fundamental` | 105 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.2284 | 1.03x |
| `fundamental_v2` | 111 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + turnaround | 0.2076 | 0.81x |
| `valuation_only` | 4 | valuation | 0.1802 | 0.67x |

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
