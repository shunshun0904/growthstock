# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 3ヶ月内+20% / 維持20日 / 終盤+15% / MA20>=MA60
- データセット: 10,116サンプル / 全体の正例率 **7.48%**
- 期間: 2018-07-12 〜 2026-06-10 / 銘柄数 2,330
- 分割: 訓練 〜2023-01-01 / 検証 2023-01-01〜2024-10-01 / テスト 2024-10-01〜
- **エンバーゴ 60営業日**（ラベル確定に必要な将来日数から自動導出）

  - train: 3,427件 / 2018-07-12 〜 2022-10-05 / 正例率 7.30%
  - val: 3,039件 / 2023-01-04 〜 2024-07-05 / 正例率 6.71%
  - test: 3,160件 / 2024-10-01 〜 2026-06-10 / 正例率 9.02%


## 検証データ (val)

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM [technical] | 0.1035 | 0.6221 | 20.0% | 11.9% | 1.78x |
| ロジスティック回帰 [delta_core_technical] | 0.1000 | 0.5892 | 10.0% | 13.2% | 1.97x |
| ロジスティック回帰 [rank_technical] | 0.0993 | 0.6137 | 20.0% | 9.9% | 1.48x |
| ロジスティック回帰 [delta_technical] | 0.0982 | 0.5825 | 6.7% | 13.2% | 1.97x |
| ロジスティック回帰 [fundamental_v2] | 0.0979 | 0.5644 | 26.7% | 9.3% | 1.38x |
| ロジスティック回帰 [fundamental_v3] | 0.0975 | 0.5636 | 26.7% | 10.6% | 1.58x |
| ロジスティック回帰 [fund_simple] | 0.0974 | 0.6167 | 6.7% | 12.6% | 1.87x |
| ロジスティック回帰 [technical] | 0.0972 | 0.6048 | 6.7% | 14.6% | 2.17x |
| LightGBM [all] | 0.0963 | 0.6129 | 6.7% | 12.6% | 1.87x |
| ロジスティック回帰 [breakout_only] | 0.0945 | 0.6056 | 6.7% | 10.6% | 1.58x |
| ロジスティック回帰 [all] | 0.0940 | 0.5874 | 16.7% | 9.9% | 1.48x |
| ロジスティック回帰 [all_no_market] | 0.0925 | 0.5711 | 13.3% | 9.3% | 1.38x |
| ロジスティック回帰 [fundamental] | 0.0921 | 0.5457 | 10.0% | 10.6% | 1.58x |
| LightGBM [fundamental_v3] | 0.0912 | 0.5766 | 16.7% | 12.6% | 1.87x |
| LightGBM [all_no_market] | 0.0909 | 0.5863 | 16.7% | 9.3% | 1.38x |
| LightGBM [delta_core_technical] | 0.0899 | 0.6035 | 6.7% | 6.6% | 0.99x |
| ロジスティック回帰 [extras_only] | 0.0894 | 0.5419 | 20.0% | 9.3% | 1.38x |
| ロジスティック回帰 [fund_delta_core] | 0.0889 | 0.5360 | 10.0% | 10.6% | 1.58x |
| ロジスティック回帰 [raw_and_rank] | 0.0886 | 0.5871 | 6.7% | 8.6% | 1.28x |
| LightGBM [delta_technical] | 0.0875 | 0.5812 | 13.3% | 9.3% | 1.38x |
| ロジスティック回帰 [fund_delta] | 0.0865 | 0.5273 | 13.3% | 9.9% | 1.48x |
| LightGBM [breakout_only] | 0.0857 | 0.5388 | 20.0% | 11.9% | 1.78x |
| LightGBM [raw_and_rank] | 0.0851 | 0.5776 | 10.0% | 9.9% | 1.48x |
| ロジスティック回帰 [rank_delta_technical] | 0.0845 | 0.5752 | 10.0% | 11.3% | 1.68x |
| LightGBM [extras_only] | 0.0837 | 0.5644 | 0.0% | 11.3% | 1.68x |
| LightGBM [price_only] | 0.0835 | 0.5568 | 13.3% | 9.3% | 1.38x |
| LightGBM [fund_simple] | 0.0832 | 0.5695 | 3.3% | 10.6% | 1.58x |
| ロジスティック回帰 [price_only] | 0.0829 | 0.5666 | 13.3% | 7.9% | 1.18x |
| LightGBM [rank_technical] | 0.0818 | 0.5465 | 10.0% | 7.9% | 1.18x |
| LightGBM [rank_all] | 0.0812 | 0.5417 | 13.3% | 8.6% | 1.28x |
| ロジスティック回帰 [rank_price_only] | 0.0796 | 0.5344 | 6.7% | 11.9% | 1.78x |
| LightGBM [rank_price_only] | 0.0788 | 0.5560 | 6.7% | 8.6% | 1.28x |
| ロジスティック回帰 [valuation_only] | 0.0787 | 0.5250 | 10.0% | 9.9% | 1.48x |
| ロジスティック回帰 [rank_all] | 0.0782 | 0.5463 | 10.0% | 9.3% | 1.38x |
| LightGBM [fundamental_v2] | 0.0780 | 0.5502 | 3.3% | 6.0% | 0.89x |
| LightGBM [valuation_only] | 0.0776 | 0.5604 | 10.0% | 7.3% | 1.09x |
| LightGBM [rank_fundamental] | 0.0754 | 0.5354 | 3.3% | 6.0% | 0.89x |
| LightGBM [rank_delta_technical] | 0.0748 | 0.5195 | 13.3% | 7.3% | 1.09x |
| LightGBM [fundamental] | 0.0737 | 0.5336 | 6.7% | 9.3% | 1.38x |
| LightGBM [rank_fundamental_v2] | 0.0715 | 0.5152 | 3.3% | 6.0% | 0.89x |
| ロジスティック回帰 [rank_fundamental] | 0.0707 | 0.5037 | 6.7% | 7.9% | 1.18x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.0690 | 0.4946 | 3.3% | 8.6% | 1.28x |
| LightGBM [fund_delta_core] | 0.0666 | 0.4921 | 3.3% | 7.3% | 1.09x |
| LightGBM [fund_delta] | 0.0642 | 0.4823 | 6.7% | 5.3% | 0.79x |

（正例率 = 6.71% / n = 3,039）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [breakout_only] | 0.1337 | 0.6306 | 12.9% | 15.2% | 1.68x |
| LightGBM [rank_technical] | 0.1315 | 0.5371 | 29.0% | 19.6% | 2.18x |
| ロジスティック回帰 [rank_price_only] | 0.1305 | 0.5641 | 32.3% | 16.5% | 1.82x |
| ロジスティック回帰 [price_only] | 0.1276 | 0.5813 | 16.1% | 16.5% | 1.82x |
| LightGBM [technical] | 0.1259 | 0.5948 | 19.4% | 15.2% | 1.68x |
| LightGBM [rank_price_only] | 0.1227 | 0.5947 | 19.4% | 15.2% | 1.68x |
| LightGBM [rank_all] | 0.1226 | 0.5962 | 9.7% | 12.0% | 1.33x |
| ロジスティック回帰 [technical] | 0.1186 | 0.5771 | 6.5% | 15.2% | 1.68x |
| LightGBM [all] | 0.1153 | 0.5805 | 12.9% | 12.0% | 1.33x |
| ロジスティック回帰 [all_no_market] | 0.1152 | 0.5738 | 9.7% | 12.7% | 1.40x |
| ロジスティック回帰 [delta_core_technical] | 0.1152 | 0.5668 | 9.7% | 14.6% | 1.61x |
| ロジスティック回帰 [delta_technical] | 0.1140 | 0.5634 | 9.7% | 14.6% | 1.61x |
| LightGBM [rank_delta_technical] | 0.1127 | 0.5707 | 22.6% | 11.4% | 1.26x |
| ロジスティック回帰 [fund_simple] | 0.1123 | 0.5532 | 12.9% | 15.8% | 1.75x |
| LightGBM [all_no_market] | 0.1098 | 0.5629 | 6.5% | 12.7% | 1.40x |
| ロジスティック回帰 [all] | 0.1095 | 0.5550 | 12.9% | 13.3% | 1.47x |
| LightGBM [extras_only] | 0.1089 | 0.5612 | 3.2% | 12.7% | 1.40x |
| ロジスティック回帰 [rank_technical] | 0.1087 | 0.5442 | 16.1% | 14.6% | 1.61x |
| LightGBM [rank_fundamental_v2] | 0.1063 | 0.5535 | 6.5% | 10.1% | 1.12x |
| ロジスティック回帰 [rank_all] | 0.1056 | 0.5562 | 12.9% | 12.7% | 1.40x |
| ロジスティック回帰 [raw_and_rank] | 0.1054 | 0.5564 | 9.7% | 12.7% | 1.40x |
| ロジスティック回帰 [rank_delta_technical] | 0.1048 | 0.5539 | 6.5% | 11.4% | 1.26x |
| LightGBM [fundamental_v3] | 0.1044 | 0.5372 | 9.7% | 13.9% | 1.54x |
| ロジスティック回帰 [fundamental_v2] | 0.1042 | 0.5438 | 9.7% | 10.8% | 1.19x |
| ロジスティック回帰 [fundamental_v3] | 0.1037 | 0.5426 | 9.7% | 10.1% | 1.12x |
| LightGBM [price_only] | 0.1029 | 0.5305 | 6.5% | 13.3% | 1.47x |
| LightGBM [fund_simple] | 0.1015 | 0.5413 | 16.1% | 7.0% | 0.77x |
| LightGBM [delta_technical] | 0.1011 | 0.5442 | 12.9% | 8.9% | 0.98x |
| ロジスティック回帰 [fund_delta_core] | 0.1009 | 0.5324 | 9.7% | 10.1% | 1.12x |
| LightGBM [raw_and_rank] | 0.1009 | 0.5398 | 9.7% | 11.4% | 1.26x |
| ロジスティック回帰 [extras_only] | 0.1005 | 0.5506 | 0.0% | 7.0% | 0.77x |
| LightGBM [fundamental] | 0.1004 | 0.5121 | 16.1% | 10.1% | 1.12x |
| LightGBM [delta_core_technical] | 0.1001 | 0.5345 | 9.7% | 10.8% | 1.19x |
| LightGBM [fundamental_v2] | 0.0999 | 0.5220 | 9.7% | 10.8% | 1.19x |
| ロジスティック回帰 [fund_delta] | 0.0990 | 0.5198 | 6.5% | 9.5% | 1.05x |
| ロジスティック回帰 [fundamental] | 0.0980 | 0.5250 | 9.7% | 9.5% | 1.05x |
| LightGBM [rank_fundamental] | 0.0961 | 0.5256 | 3.2% | 8.9% | 0.98x |
| LightGBM [breakout_only] | 0.0957 | 0.5219 | 9.7% | 7.6% | 0.84x |
| ロジスティック回帰 [valuation_only] | 0.0955 | 0.5248 | 6.5% | 8.2% | 0.91x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.0950 | 0.5215 | 12.9% | 6.3% | 0.70x |
| ロジスティック回帰 [rank_fundamental] | 0.0949 | 0.5219 | 6.5% | 10.8% | 1.19x |
| LightGBM [fund_delta_core] | 0.0923 | 0.4917 | 16.1% | 11.4% | 1.26x |
| LightGBM [fund_delta] | 0.0921 | 0.4844 | 16.1% | 10.1% | 1.12x |
| LightGBM [valuation_only] | 0.0910 | 0.5150 | 6.5% | 7.6% | 0.84x |

（正例率 = 9.02% / n = 3,160）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **LightGBM [technical]**（テスト PR-AUC 0.1259）。
単変量のベースラインは廃止した。母集団を高値更新日にした時点で
`r_high` は全件ほぼ100の定数になり、勝っても何も言えないため。
決算を使わないモデルを基準にして、決算を足す価値を直接測る。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| ロジスティック回帰 [breakout_only] | 0.1337 | +0.0078 | [-0.0140, +0.0272] | 0.778 | 誤差 |
| LightGBM [rank_technical] | 0.1315 | +0.0057 | [-0.0220, +0.0362] | 0.654 | 誤差 |
| ロジスティック回帰 [rank_price_only] | 0.1305 | +0.0046 | [-0.0244, +0.0273] | 0.595 | 誤差 |
| ロジスティック回帰 [price_only] | 0.1276 | +0.0017 | [-0.0220, +0.0222] | 0.512 | 誤差 |
| LightGBM [rank_price_only] | 0.1227 | -0.0032 | [-0.0275, +0.0144] | 0.328 | 誤差 |
| LightGBM [rank_all] | 0.1226 | -0.0033 | [-0.0269, +0.0181] | 0.371 | 誤差 |
| ロジスティック回帰 [technical] | 0.1186 | -0.0073 | [-0.0278, +0.0094] | 0.206 | 誤差 |
| LightGBM [all] | 0.1153 | -0.0106 | [-0.0331, +0.0056] | 0.105 | 誤差 |
| ロジスティック回帰 [all] | 0.1095 | -0.0164 | [-0.0395, +0.0013] | 0.035 | 誤差 |
| ロジスティック回帰 [rank_technical] | 0.1087 | -0.0171 | [-0.0399, +0.0012] | 0.033 | 誤差 |
| ロジスティック回帰 [rank_all] | 0.1056 | -0.0202 | [-0.0462, +0.0030] | 0.035 | 誤差 |
| LightGBM [price_only] | 0.1029 | -0.0230 | [-0.0450, -0.0061] | 0.005 | 有意に劣る |
| LightGBM [fundamental] | 0.1004 | -0.0254 | [-0.0525, -0.0036] | 0.011 | 有意に劣る |
| ロジスティック回帰 [fundamental] | 0.0980 | -0.0279 | [-0.0545, -0.0074] | 0.005 | 有意に劣る |
| LightGBM [rank_fundamental] | 0.0961 | -0.0298 | [-0.0560, -0.0106] | 0.000 | 有意に劣る |
| ロジスティック回帰 [rank_fundamental] | 0.0949 | -0.0310 | [-0.0584, -0.0100] | 0.004 | 有意に劣る |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `breakout_only` | 5 | breakout | 0.1337 | 1.68x |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.1315 | 2.18x |
| `rank_price_only` | 3 | price_rank | 0.1305 | 1.82x |
| `price_only` | 3 | price | 0.1276 | 1.82x |
| `technical` | 14 | price + breakout + volume + liquidity + supply + market | 0.1259 | 1.68x |
| `rank_all` | 136 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + breakout_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + sector + turnaround + market | 0.1226 | 1.33x |
| `all` | 136 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + market | 0.1153 | 1.33x |
| `all_no_market` | 134 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.1152 | 1.40x |
| `delta_core_technical` | 84 | fund_growth + fund_trend + fund_streak + turnaround + price + breakout + volume + liquidity + supply + market | 0.1152 | 1.61x |
| `delta_technical` | 86 | fund_growth + fund_trend + fund_streak + turnaround + guidance + price + breakout + volume + liquidity + supply + market | 0.1140 | 1.61x |
| `rank_delta_technical` | 86 | fund_growth_rank + fund_trend_rank + fund_streak_rank + turnaround + guidance_rank + price_rank + breakout_rank + volume_rank + liquidity_rank + supply_rank + market | 0.1127 | 1.26x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.1123 | 1.75x |
| `extras_only` | 23 | valuation + dividend + cashflow + efficiency + guidance + sector | 0.1089 | 1.40x |
| `rank_fundamental_v2` | 118 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + turnaround | 0.1063 | 1.12x |
| `raw_and_rank` | 130 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.1054 | 1.40x |
| `fundamental_v3` | 122 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.1044 | 1.54x |
| `fundamental_v2` | 118 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + turnaround | 0.1042 | 1.19x |
| `fund_delta_core` | 70 | fund_growth + fund_trend + fund_streak + turnaround | 0.1009 | 1.12x |
| `fundamental` | 97 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.1004 | 1.12x |
| `fund_delta` | 72 | fund_growth + fund_trend + fund_streak + turnaround + guidance | 0.0990 | 1.05x |
| `rank_fundamental` | 97 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.0961 | 0.98x |
| `valuation_only` | 7 | valuation | 0.0955 | 0.91x |

## 特徴量の寄与（`breakout_only` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `vol_20d` | +0.429 | ブレイクしやすい |
| `break_margin` | -0.100 | しにくい |
| `close_position` | -0.085 | しにくい |
| `ret_20d` | +0.064 | ブレイクしやすい |
| `base_length` | +0.053 | ブレイクしやすい |

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
