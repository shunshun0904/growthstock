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
| LightGBM [price_only] | 0.3435 | 0.7150 | 40.8% | 42.6% | 2.17x |
| LightGBM [rank_price_only] | 0.3273 | 0.6990 | 37.0% | 38.8% | 1.98x |
| ロジスティック回帰 [rank_price_only] | 0.3194 | 0.7100 | 31.5% | 33.9% | 1.73x |
| ロジスティック回帰 [rank_technical] | 0.3124 | 0.7008 | 29.9% | 32.6% | 1.66x |
| LightGBM [technical] | 0.3071 | 0.6631 | 33.7% | 38.5% | 1.96x |
| ロジスティック回帰 [technical] | 0.2826 | 0.6868 | 15.8% | 23.1% | 1.18x |
| ロジスティック回帰 [price_only] | 0.2785 | 0.6825 | 15.8% | 21.7% | 1.10x |
| ロジスティック回帰 [fund_simple] | 0.2735 | 0.6755 | 15.8% | 21.3% | 1.09x |
| ロジスティック回帰 [raw_and_rank] | 0.2654 | 0.6544 | 20.1% | 22.3% | 1.14x |
| LightGBM [rank_technical] | 0.2642 | 0.6099 | 29.3% | 31.3% | 1.60x |
| LightGBM [rank_all] | 0.2580 | 0.6032 | 31.0% | 31.9% | 1.62x |
| LightGBM [raw_and_rank] | 0.2569 | 0.5955 | 34.2% | 30.0% | 1.53x |
| LightGBM [all_no_market] | 0.2548 | 0.6157 | 29.9% | 25.6% | 1.30x |
| LightGBM [fund_simple] | 0.2520 | 0.5938 | 29.3% | 30.1% | 1.53x |
| ロジスティック回帰 [rank_all] | 0.2517 | 0.6159 | 20.1% | 23.0% | 1.17x |
| LightGBM [all] | 0.2425 | 0.5925 | 27.2% | 25.7% | 1.31x |
| ロジスティック回帰 [all] | 0.2370 | 0.6183 | 14.1% | 16.7% | 0.85x |
| ロジスティック回帰 [all_no_market] | 0.2299 | 0.6071 | 12.0% | 15.3% | 0.78x |
| ベースライン: 出来高モメンタムのみ | 0.2133 | 0.5560 | 15.2% | 17.7% | 0.90x |
| ベースライン: 既存の8軸総合スコア | 0.2130 | 0.5486 | 22.8% | 19.8% | 1.01x |
| ロジスティック回帰 [fundamental] | 0.1930 | 0.5031 | 14.7% | 15.5% | 0.79x |
| LightGBM [rank_fundamental_v2] | 0.1817 | 0.4548 | 16.8% | 17.1% | 0.87x |
| LightGBM [fundamental] | 0.1797 | 0.4680 | 6.0% | 12.0% | 0.61x |
| LightGBM [rank_fundamental] | 0.1790 | 0.4577 | 13.6% | 15.2% | 0.77x |
| ロジスティック回帰 [rank_fundamental] | 0.1777 | 0.4686 | 10.9% | 13.5% | 0.69x |
| LightGBM [fundamental_v3] | 0.1759 | 0.4601 | 15.2% | 11.5% | 0.59x |
| ロジスティック回帰 [fundamental_v3] | 0.1712 | 0.4340 | 3.8% | 22.0% | 1.12x |
| LightGBM [fundamental_v2] | 0.1697 | 0.4416 | 12.5% | 13.3% | 0.68x |
| ロジスティック回帰 [fundamental_v2] | 0.1675 | 0.4411 | 7.6% | 9.4% | 0.48x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1627 | 0.4234 | 4.3% | 10.8% | 0.55x |
| LightGBM [extras_only] | 0.1603 | 0.4110 | 9.8% | 8.8% | 0.45x |
| ロジスティック回帰 [extras_only] | 0.1592 | 0.3762 | 2.2% | 20.7% | 1.05x |
| LightGBM [valuation_only] | 0.1410 | 0.3366 | 10.3% | 7.3% | 0.37x |
| ロジスティック回帰 [valuation_only] | 0.1385 | 0.3299 | 4.3% | 4.3% | 0.22x |

（正例率 = 19.63% / n = 18,462）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [rank_technical] | 0.3442 | 0.6911 | 36.4% | 39.6% | 1.83x |
| LightGBM [price_only] | 0.3432 | 0.6896 | 40.8% | 38.8% | 1.79x |
| ロジスティック回帰 [rank_price_only] | 0.3413 | 0.6962 | 36.4% | 39.3% | 1.82x |
| ベースライン: R_high のみ | 0.3379 | 0.6839 | 44.7% | 39.0% | 1.80x |
| LightGBM [rank_price_only] | 0.3356 | 0.6791 | 40.4% | 39.5% | 1.82x |
| ロジスティック回帰 [price_only] | 0.3258 | 0.6851 | 26.8% | 36.5% | 1.69x |
| ロジスティック回帰 [technical] | 0.3226 | 0.6879 | 23.2% | 33.1% | 1.53x |
| LightGBM [technical] | 0.3208 | 0.6477 | 47.8% | 39.7% | 1.83x |
| ロジスティック回帰 [fund_simple] | 0.3159 | 0.6812 | 19.7% | 31.9% | 1.47x |
| ロジスティック回帰 [raw_and_rank] | 0.3088 | 0.6708 | 23.2% | 29.5% | 1.36x |
| LightGBM [all_no_market] | 0.3051 | 0.6346 | 38.2% | 37.9% | 1.75x |
| LightGBM [raw_and_rank] | 0.3008 | 0.6135 | 46.9% | 39.9% | 1.84x |
| LightGBM [rank_technical] | 0.2984 | 0.6148 | 36.0% | 35.4% | 1.63x |
| LightGBM [all] | 0.2953 | 0.6112 | 46.9% | 36.9% | 1.70x |
| LightGBM [fund_simple] | 0.2935 | 0.6096 | 40.8% | 35.7% | 1.65x |
| ロジスティック回帰 [rank_all] | 0.2918 | 0.6308 | 25.9% | 32.5% | 1.50x |
| ロジスティック回帰 [all] | 0.2839 | 0.6402 | 19.3% | 27.3% | 1.26x |
| LightGBM [rank_all] | 0.2835 | 0.5939 | 39.9% | 34.6% | 1.60x |
| ロジスティック回帰 [all_no_market] | 0.2802 | 0.6324 | 21.9% | 27.7% | 1.28x |
| ベースライン: 出来高モメンタムのみ | 0.2348 | 0.5474 | 16.2% | 22.3% | 1.03x |
| ベースライン: 既存の8軸総合スコア | 0.2334 | 0.5382 | 23.2% | 23.5% | 1.09x |
| LightGBM [fundamental_v3] | 0.2292 | 0.5225 | 23.7% | 22.2% | 1.03x |
| LightGBM [fundamental] | 0.2207 | 0.5102 | 22.4% | 21.9% | 1.01x |
| ロジスティック回帰 [fundamental_v3] | 0.2190 | 0.4844 | 13.6% | 29.6% | 1.37x |
| ロジスティック回帰 [fundamental] | 0.2118 | 0.5087 | 18.0% | 17.0% | 0.78x |
| LightGBM [fundamental_v2] | 0.2117 | 0.5025 | 18.0% | 16.3% | 0.75x |
| LightGBM [extras_only] | 0.2108 | 0.4776 | 21.1% | 21.9% | 1.01x |
| ロジスティック回帰 [extras_only] | 0.2093 | 0.4421 | 22.4% | 31.7% | 1.47x |
| ロジスティック回帰 [rank_fundamental] | 0.2070 | 0.4991 | 17.1% | 15.6% | 0.72x |
| LightGBM [rank_fundamental] | 0.2063 | 0.4763 | 17.5% | 18.4% | 0.85x |
| LightGBM [rank_fundamental_v2] | 0.1995 | 0.4526 | 18.9% | 19.3% | 0.89x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1990 | 0.4741 | 17.5% | 16.5% | 0.76x |
| ロジスティック回帰 [fundamental_v2] | 0.1973 | 0.4722 | 14.9% | 15.5% | 0.71x |
| LightGBM [valuation_only] | 0.1795 | 0.4080 | 22.4% | 14.1% | 0.65x |
| ロジスティック回帰 [valuation_only] | 0.1694 | 0.3886 | 7.0% | 9.5% | 0.44x |

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
| LightGBM [all] | 0.2953 | -0.0426 | [-0.0537, -0.0314] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_all] | 0.2918 | -0.0461 | [-0.0539, -0.0382] | 0.000 | 有意に劣る |
| ロジスティック回帰 [all] | 0.2839 | -0.0540 | [-0.0628, -0.0443] | 0.000 | 有意に劣る |
| LightGBM [rank_all] | 0.2835 | -0.0544 | [-0.0651, -0.0440] | 0.000 | 有意に劣る |
| LightGBM [fundamental] | 0.2207 | -0.1171 | [-0.1284, -0.1066] | 0.000 | 有意に劣る |
| ロジスティック回帰 [fundamental] | 0.2118 | -0.1261 | [-0.1370, -0.1161] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_fundamental] | 0.2070 | -0.1308 | [-0.1413, -0.1204] | 0.000 | 有意に劣る |
| LightGBM [rank_fundamental] | 0.2063 | -0.1316 | [-0.1429, -0.1206] | 0.000 | 有意に劣る |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.3442 | 1.83x |
| `price_only` | 3 | price | 0.3432 | 1.79x |
| `rank_price_only` | 3 | price_rank | 0.3413 | 1.82x |
| `technical` | 9 | price + volume + liquidity + supply + market | 0.3226 | 1.53x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.3159 | 1.47x |
| `raw_and_rank` | 146 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.3088 | 1.36x |
| `all_no_market` | 137 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.3051 | 1.75x |
| `all` | 139 | fund_level + fund_lag + fund_trend + fund_streak + price + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + market | 0.2953 | 1.70x |
| `rank_all` | 139 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + sector + turnaround + market | 0.2918 | 1.50x |
| `fundamental_v3` | 130 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.2292 | 1.03x |
| `fundamental` | 105 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.2207 | 1.01x |
| `fundamental_v2` | 126 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + turnaround | 0.2117 | 0.75x |
| `extras_only` | 23 | valuation + dividend + cashflow + efficiency + guidance + sector | 0.2108 | 1.01x |
| `rank_fundamental` | 105 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.2070 | 0.72x |
| `rank_fundamental_v2` | 126 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + turnaround | 0.1995 | 0.89x |
| `valuation_only` | 7 | valuation | 0.1795 | 0.65x |

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
