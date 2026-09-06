# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 3ヶ月内+20% / 維持20日 / 終盤+15% / MA20>=MA60
- データセット: 17,580サンプル / 全体の正例率 **7.49%**
- 期間: 2018-04-03 〜 2026-06-10 / 銘柄数 3,289
- 分割: 訓練 〜2023-01-01 / 検証 2023-01-01〜2024-10-01 / テスト 2024-10-01〜
- **エンバーゴ 60営業日**（ラベル確定に必要な将来日数から自動導出）

  - train: 7,041件 / 2018-04-03 〜 2022-10-05 / 正例率 6.69%
  - val: 4,764件 / 2023-01-04 〜 2024-07-05 / 正例率 6.91%
  - test: 4,975件 / 2024-10-01 〜 2026-06-10 / 正例率 9.63%


## 検証データ (val)

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [rank_technical] | 0.0999 | 0.6124 | 10.6% | 10.9% | 1.58x |
| LightGBM [all] | 0.0984 | 0.5890 | 17.0% | 12.6% | 1.83x |
| LightGBM [all_no_market] | 0.0968 | 0.6050 | 14.9% | 12.2% | 1.76x |
| LightGBM [technical] | 0.0967 | 0.6116 | 17.0% | 12.6% | 1.83x |
| ロジスティック回帰 [fund_simple] | 0.0957 | 0.6064 | 6.4% | 10.5% | 1.52x |
| ロジスティック回帰 [raw_and_rank] | 0.0951 | 0.5848 | 14.9% | 9.2% | 1.34x |
| LightGBM [extras_only] | 0.0951 | 0.5941 | 10.6% | 11.3% | 1.64x |
| ロジスティック回帰 [technical] | 0.0946 | 0.6060 | 4.3% | 9.7% | 1.40x |
| LightGBM [fundamental_v3] | 0.0919 | 0.5926 | 12.8% | 10.9% | 1.58x |
| ロジスティック回帰 [breakout_only] | 0.0906 | 0.5907 | 10.6% | 9.2% | 1.34x |
| LightGBM [fund_simple] | 0.0905 | 0.5895 | 14.9% | 10.1% | 1.46x |
| LightGBM [raw_and_rank] | 0.0903 | 0.5797 | 12.8% | 11.8% | 1.70x |
| LightGBM [fundamental_v2] | 0.0899 | 0.5871 | 6.4% | 10.9% | 1.58x |
| LightGBM [breakout_only] | 0.0895 | 0.5764 | 12.8% | 10.5% | 1.52x |
| ロジスティック回帰 [price_only] | 0.0892 | 0.5680 | 17.0% | 10.5% | 1.52x |
| ロジスティック回帰 [extras_only] | 0.0888 | 0.5689 | 14.9% | 11.8% | 1.70x |
| ロジスティック回帰 [all] | 0.0885 | 0.5705 | 6.4% | 8.4% | 1.22x |
| ベースライン: 出来高モメンタムのみ | 0.0883 | 0.5728 | 14.9% | 10.1% | 1.46x |
| ロジスティック回帰 [rank_all] | 0.0880 | 0.5751 | 10.6% | 10.1% | 1.46x |
| LightGBM [rank_all] | 0.0871 | 0.5835 | 8.5% | 7.1% | 1.03x |
| LightGBM [rank_price_only] | 0.0851 | 0.5641 | 12.8% | 9.7% | 1.40x |
| ロジスティック回帰 [fundamental_v3] | 0.0850 | 0.5490 | 14.9% | 11.3% | 1.64x |
| LightGBM [valuation_only] | 0.0848 | 0.5669 | 4.3% | 10.1% | 1.46x |
| ロジスティック回帰 [all_no_market] | 0.0846 | 0.5505 | 8.5% | 10.1% | 1.46x |
| ロジスティック回帰 [rank_price_only] | 0.0843 | 0.5459 | 10.6% | 10.5% | 1.52x |
| LightGBM [price_only] | 0.0837 | 0.5581 | 12.8% | 8.4% | 1.22x |
| ロジスティック回帰 [valuation_only] | 0.0833 | 0.5237 | 14.9% | 11.8% | 1.70x |
| ロジスティック回帰 [fundamental_v2] | 0.0833 | 0.5460 | 10.6% | 10.5% | 1.52x |
| LightGBM [rank_technical] | 0.0828 | 0.5417 | 6.4% | 11.3% | 1.64x |
| LightGBM [rank_fundamental_v2] | 0.0826 | 0.5485 | 12.8% | 7.6% | 1.10x |
| ロジスティック回帰 [fundamental] | 0.0815 | 0.5232 | 14.9% | 9.2% | 1.34x |
| LightGBM [fundamental] | 0.0814 | 0.5653 | 6.4% | 7.6% | 1.10x |
| LightGBM [rank_fundamental] | 0.0781 | 0.5546 | 10.6% | 7.1% | 1.03x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.0745 | 0.5169 | 10.6% | 8.4% | 1.22x |
| ロジスティック回帰 [rank_fundamental] | 0.0727 | 0.5170 | 6.4% | 5.5% | 0.79x |
| ベースライン: 既存の8軸総合スコア | 0.0722 | 0.5042 | 4.3% | 8.0% | 1.16x |
| ベースライン: R_high のみ | 0.0642 | 0.4480 | 12.8% | 10.9% | 1.58x |

（正例率 = 6.91% / n = 4,764）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [rank_price_only] | 0.1419 | 0.5971 | 34.7% | 16.1% | 1.68x |
| LightGBM [technical] | 0.1342 | 0.6167 | 16.3% | 15.7% | 1.63x |
| ロジスティック回帰 [price_only] | 0.1289 | 0.5870 | 12.2% | 14.9% | 1.55x |
| LightGBM [rank_all] | 0.1288 | 0.5904 | 24.5% | 14.9% | 1.55x |
| ロジスティック回帰 [breakout_only] | 0.1278 | 0.6146 | 8.2% | 10.5% | 1.09x |
| ロジスティック回帰 [all_no_market] | 0.1261 | 0.5868 | 10.2% | 14.9% | 1.55x |
| ロジスティック回帰 [fund_simple] | 0.1260 | 0.5721 | 20.4% | 14.9% | 1.55x |
| LightGBM [all] | 0.1249 | 0.5957 | 14.3% | 12.9% | 1.34x |
| LightGBM [all_no_market] | 0.1235 | 0.5849 | 2.0% | 15.7% | 1.63x |
| ロジスティック回帰 [raw_and_rank] | 0.1233 | 0.5863 | 14.3% | 10.5% | 1.09x |
| ロジスティック回帰 [technical] | 0.1233 | 0.5845 | 10.2% | 12.5% | 1.30x |
| ロジスティック回帰 [all] | 0.1211 | 0.5705 | 8.2% | 14.5% | 1.51x |
| LightGBM [price_only] | 0.1186 | 0.5723 | 12.2% | 11.7% | 1.21x |
| LightGBM [rank_price_only] | 0.1184 | 0.5681 | 6.1% | 12.9% | 1.34x |
| LightGBM [raw_and_rank] | 0.1170 | 0.5812 | 12.2% | 12.5% | 1.30x |
| ロジスティック回帰 [rank_technical] | 0.1157 | 0.5519 | 14.3% | 13.3% | 1.38x |
| ロジスティック回帰 [rank_all] | 0.1140 | 0.5583 | 12.2% | 13.7% | 1.42x |
| ロジスティック回帰 [fundamental_v3] | 0.1126 | 0.5510 | 12.2% | 11.7% | 1.21x |
| LightGBM [fund_simple] | 0.1122 | 0.5585 | 10.2% | 11.3% | 1.17x |
| ロジスティック回帰 [extras_only] | 0.1112 | 0.5499 | 8.2% | 10.9% | 1.13x |
| LightGBM [extras_only] | 0.1095 | 0.5546 | 4.1% | 8.9% | 0.92x |
| ロジスティック回帰 [fundamental_v2] | 0.1092 | 0.5432 | 12.2% | 12.1% | 1.26x |
| LightGBM [rank_technical] | 0.1082 | 0.5370 | 10.2% | 12.1% | 1.26x |
| LightGBM [breakout_only] | 0.1061 | 0.5475 | 6.1% | 8.1% | 0.84x |
| ロジスティック回帰 [fundamental] | 0.1048 | 0.5377 | 10.2% | 9.3% | 0.96x |
| ベースライン: 既存の8軸総合スコア | 0.1031 | 0.4982 | 10.2% | 14.9% | 1.55x |
| ベースライン: 出来高モメンタムのみ | 0.1030 | 0.5322 | 6.1% | 10.5% | 1.09x |
| LightGBM [rank_fundamental_v2] | 0.1015 | 0.5175 | 14.3% | 10.1% | 1.05x |
| LightGBM [fundamental_v2] | 0.1010 | 0.5189 | 8.2% | 8.5% | 0.88x |
| LightGBM [rank_fundamental] | 0.1006 | 0.5187 | 10.2% | 9.7% | 1.01x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.1000 | 0.5232 | 4.1% | 8.5% | 0.88x |
| LightGBM [fundamental] | 0.0997 | 0.5080 | 6.1% | 12.5% | 1.30x |
| LightGBM [fundamental_v3] | 0.0993 | 0.5239 | 12.2% | 7.7% | 0.80x |
| ロジスティック回帰 [valuation_only] | 0.0985 | 0.4951 | 12.2% | 10.1% | 1.05x |
| ロジスティック回帰 [rank_fundamental] | 0.0975 | 0.5067 | 4.1% | 9.3% | 0.96x |
| LightGBM [valuation_only] | 0.0971 | 0.5051 | 8.2% | 10.9% | 1.13x |
| ベースライン: R_high のみ | 0.0964 | 0.4836 | 10.2% | 10.5% | 1.09x |

（正例率 = 9.63% / n = 4,975）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **ベースライン: R_high のみ**（テスト PR-AUC 0.0964）。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| ロジスティック回帰 [rank_price_only] | 0.1419 | +0.0455 | [+0.0240, +0.0716] | 1.000 | 有意 |
| LightGBM [technical] | 0.1342 | +0.0378 | [+0.0216, +0.0575] | 1.000 | 有意 |
| ロジスティック回帰 [price_only] | 0.1289 | +0.0324 | [+0.0167, +0.0530] | 1.000 | 有意 |
| LightGBM [rank_all] | 0.1288 | +0.0324 | [+0.0164, +0.0527] | 1.000 | 有意 |
| ロジスティック回帰 [breakout_only] | 0.1278 | +0.0314 | [+0.0186, +0.0469] | 1.000 | 有意 |
| ロジスティック回帰 [all_no_market] | 0.1261 | +0.0297 | [+0.0160, +0.0462] | 1.000 | 有意 |
| LightGBM [all] | 0.1249 | +0.0285 | [+0.0136, +0.0440] | 1.000 | 有意 |
| ロジスティック回帰 [technical] | 0.1233 | +0.0269 | [+0.0136, +0.0427] | 1.000 | 有意 |
| ロジスティック回帰 [all] | 0.1211 | +0.0247 | [+0.0107, +0.0411] | 1.000 | 有意 |
| LightGBM [price_only] | 0.1186 | +0.0222 | [+0.0085, +0.0395] | 1.000 | 有意 |
| LightGBM [rank_price_only] | 0.1184 | +0.0220 | [+0.0062, +0.0383] | 0.999 | 有意 |
| ロジスティック回帰 [rank_technical] | 0.1157 | +0.0193 | [+0.0050, +0.0356] | 0.996 | 有意 |
| ロジスティック回帰 [rank_all] | 0.1140 | +0.0176 | [+0.0046, +0.0325] | 0.996 | 有意 |
| LightGBM [rank_technical] | 0.1082 | +0.0118 | [-0.0012, +0.0275] | 0.962 | 誤差 |
| ロジスティック回帰 [fundamental] | 0.1048 | +0.0084 | [-0.0035, +0.0215] | 0.916 | 誤差 |
| LightGBM [rank_fundamental] | 0.1006 | +0.0042 | [-0.0072, +0.0171] | 0.766 | 誤差 |
| LightGBM [fundamental] | 0.0997 | +0.0032 | [-0.0098, +0.0165] | 0.725 | 誤差 |
| ロジスティック回帰 [rank_fundamental] | 0.0975 | +0.0010 | [-0.0101, +0.0131] | 0.614 | 誤差 |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `rank_price_only` | 3 | price_rank | 0.1419 | 1.68x |
| `technical` | 14 | price + breakout + volume + liquidity + supply + market | 0.1342 | 1.63x |
| `price_only` | 3 | price | 0.1289 | 1.55x |
| `rank_all` | 136 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + breakout_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + sector + turnaround + market | 0.1288 | 1.55x |
| `breakout_only` | 5 | breakout | 0.1278 | 1.09x |
| `all_no_market` | 134 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.1261 | 1.55x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.1260 | 1.55x |
| `all` | 136 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + market | 0.1249 | 1.34x |
| `raw_and_rank` | 130 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.1233 | 1.09x |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.1157 | 1.38x |
| `fundamental_v3` | 122 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.1126 | 1.21x |
| `extras_only` | 23 | valuation + dividend + cashflow + efficiency + guidance + sector | 0.1112 | 1.13x |
| `fundamental_v2` | 118 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + turnaround | 0.1092 | 1.26x |
| `fundamental` | 97 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.1048 | 0.96x |
| `rank_fundamental_v2` | 118 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + turnaround | 0.1015 | 1.05x |
| `rank_fundamental` | 97 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.1006 | 1.01x |
| `valuation_only` | 7 | valuation | 0.0985 | 1.05x |

## 特徴量の寄与（`rank_price_only` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `r_high_6m_r` | -0.121 | しにくい |
| `r_high_r` | -0.085 | しにくい |
| `r_high_3m_r` | -0.059 | しにくい |

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
