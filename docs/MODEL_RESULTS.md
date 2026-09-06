# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 52週 / 1〜6ヶ月 / 定着20日-8% +60日後0%
- データセット: 19,542サンプル / 全体の正例率 **24.92%**
- 期間: 2017-09-29 〜 2026-06-10 / 銘柄数 3,317
- 分割: 訓練 〜2023-01-01 / 検証 2023-01-01〜2024-10-01 / テスト 2024-10-01〜
- **エンバーゴ 60営業日**（ラベル確定に必要な将来日数から自動導出）

  - train: 7,638件 / 2017-09-29 〜 2022-10-05 / 正例率 23.53%
  - val: 5,310件 / 2023-01-04 〜 2024-07-05 / 正例率 22.73%
  - test: 5,588件 / 2024-10-01 〜 2026-06-10 / 正例率 30.15%


## 検証データ (val)

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [technical] | 0.3642 | 0.6585 | 47.2% | 50.2% | 2.21x |
| ロジスティック回帰 [breakout_only] | 0.3619 | 0.6661 | 41.5% | 50.2% | 2.21x |
| ロジスティック回帰 [all] | 0.3563 | 0.6443 | 47.2% | 49.1% | 2.16x |
| LightGBM [technical] | 0.3561 | 0.6586 | 50.9% | 46.8% | 2.06x |
| ロジスティック回帰 [all_no_market] | 0.3457 | 0.6329 | 43.4% | 46.8% | 2.06x |
| ロジスティック回帰 [fund_simple] | 0.3446 | 0.6323 | 52.8% | 47.2% | 2.08x |
| ロジスティック回帰 [raw_and_rank] | 0.3412 | 0.6283 | 56.6% | 46.8% | 2.06x |
| LightGBM [all] | 0.3404 | 0.6511 | 41.5% | 43.0% | 1.89x |
| LightGBM [rank_all] | 0.3380 | 0.6350 | 50.9% | 43.0% | 1.89x |
| LightGBM [all_no_market] | 0.3348 | 0.6343 | 41.5% | 44.5% | 1.96x |
| ロジスティック回帰 [rank_all] | 0.3320 | 0.6310 | 37.7% | 44.9% | 1.98x |
| ロジスティック回帰 [rank_technical] | 0.3301 | 0.6236 | 47.2% | 41.9% | 1.84x |
| LightGBM [raw_and_rank] | 0.3278 | 0.6155 | 47.2% | 43.8% | 1.93x |
| LightGBM [fund_simple] | 0.3204 | 0.6241 | 39.6% | 43.0% | 1.89x |
| LightGBM [rank_technical] | 0.3185 | 0.6135 | 45.3% | 43.0% | 1.89x |
| LightGBM [fundamental_v3] | 0.3125 | 0.6041 | 50.9% | 40.4% | 1.78x |
| LightGBM [breakout_only] | 0.3119 | 0.6235 | 43.4% | 38.1% | 1.68x |
| LightGBM [fundamental_v2] | 0.3098 | 0.6013 | 47.2% | 39.2% | 1.73x |
| LightGBM [price_only] | 0.3087 | 0.5953 | 43.4% | 42.6% | 1.88x |
| LightGBM [fundamental] | 0.3044 | 0.5930 | 41.5% | 41.9% | 1.84x |
| ロジスティック回帰 [extras_only] | 0.3004 | 0.5754 | 52.8% | 38.5% | 1.69x |
| LightGBM [rank_fundamental] | 0.3002 | 0.5891 | 45.3% | 40.8% | 1.79x |
| LightGBM [rank_fundamental_v2] | 0.3001 | 0.5866 | 45.3% | 41.9% | 1.84x |
| ロジスティック回帰 [price_only] | 0.2977 | 0.5910 | 41.5% | 39.2% | 1.73x |
| ロジスティック回帰 [fundamental_v3] | 0.2938 | 0.5789 | 37.7% | 40.0% | 1.76x |
| LightGBM [extras_only] | 0.2932 | 0.5861 | 34.0% | 37.4% | 1.64x |
| ロジスティック回帰 [fundamental_v2] | 0.2906 | 0.5714 | 41.5% | 40.8% | 1.79x |
| LightGBM [valuation_only] | 0.2876 | 0.5775 | 39.6% | 34.3% | 1.51x |
| LightGBM [rank_price_only] | 0.2817 | 0.5676 | 45.3% | 35.8% | 1.58x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.2788 | 0.5689 | 34.0% | 33.6% | 1.48x |
| ロジスティック回帰 [rank_fundamental] | 0.2715 | 0.5636 | 39.6% | 32.1% | 1.41x |
| ロジスティック回帰 [rank_price_only] | 0.2709 | 0.5516 | 43.4% | 36.2% | 1.59x |
| ベースライン: 出来高モメンタムのみ | 0.2701 | 0.5630 | 28.3% | 32.8% | 1.44x |
| ロジスティック回帰 [fundamental] | 0.2683 | 0.5494 | 35.8% | 32.1% | 1.41x |
| ロジスティック回帰 [valuation_only] | 0.2650 | 0.5319 | 34.0% | 34.0% | 1.49x |
| ベースライン: 既存の8軸総合スコア | 0.2522 | 0.5250 | 24.5% | 32.5% | 1.43x |
| ベースライン: R_high のみ | 0.2211 | 0.4540 | 32.1% | 26.4% | 1.16x |

（正例率 = 22.73% / n = 5,310）

## テストデータ (test) — 最終評価

| モデル | PR-AUC | ROC-AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ロジスティック回帰 [breakout_only] | 0.4220 | 0.6486 | 47.3% | 50.2% | 1.66x |
| ロジスティック回帰 [technical] | 0.4161 | 0.6290 | 49.1% | 53.0% | 1.76x |
| LightGBM [technical] | 0.4155 | 0.6483 | 52.7% | 48.4% | 1.60x |
| ロジスティック回帰 [all_no_market] | 0.3971 | 0.6087 | 38.2% | 52.7% | 1.75x |
| LightGBM [all_no_market] | 0.3939 | 0.6055 | 54.5% | 51.3% | 1.70x |
| LightGBM [all] | 0.3932 | 0.6094 | 54.5% | 49.1% | 1.63x |
| LightGBM [breakout_only] | 0.3927 | 0.6184 | 54.5% | 45.2% | 1.50x |
| ロジスティック回帰 [fund_simple] | 0.3904 | 0.5913 | 50.9% | 48.7% | 1.62x |
| ロジスティック回帰 [all] | 0.3854 | 0.5916 | 38.2% | 50.9% | 1.69x |
| ロジスティック回帰 [raw_and_rank] | 0.3817 | 0.5893 | 43.6% | 46.2% | 1.53x |
| ロジスティック回帰 [rank_price_only] | 0.3777 | 0.5875 | 52.7% | 43.4% | 1.44x |
| LightGBM [rank_all] | 0.3777 | 0.5942 | 47.3% | 45.9% | 1.52x |
| ロジスティック回帰 [rank_technical] | 0.3730 | 0.5731 | 49.1% | 47.3% | 1.57x |
| LightGBM [rank_price_only] | 0.3725 | 0.5819 | 58.2% | 45.5% | 1.51x |
| LightGBM [extras_only] | 0.3713 | 0.5693 | 50.9% | 48.0% | 1.59x |
| LightGBM [price_only] | 0.3698 | 0.5889 | 32.7% | 43.0% | 1.43x |
| ロジスティック回帰 [price_only] | 0.3676 | 0.5685 | 49.1% | 43.4% | 1.44x |
| ロジスティック回帰 [extras_only] | 0.3638 | 0.5701 | 43.6% | 45.2% | 1.50x |
| LightGBM [fundamental_v3] | 0.3628 | 0.5628 | 41.8% | 44.1% | 1.46x |
| ロジスティック回帰 [rank_all] | 0.3625 | 0.5843 | 41.8% | 44.4% | 1.47x |
| LightGBM [rank_fundamental_v2] | 0.3612 | 0.5542 | 52.7% | 46.2% | 1.53x |
| LightGBM [fundamental_v2] | 0.3609 | 0.5630 | 52.7% | 43.7% | 1.45x |
| LightGBM [rank_technical] | 0.3604 | 0.5626 | 47.3% | 43.0% | 1.43x |
| ロジスティック回帰 [fundamental_v3] | 0.3577 | 0.5613 | 54.5% | 41.9% | 1.39x |
| LightGBM [raw_and_rank] | 0.3545 | 0.5612 | 49.1% | 40.5% | 1.34x |
| LightGBM [fundamental] | 0.3514 | 0.5557 | 50.9% | 40.5% | 1.34x |
| ロジスティック回帰 [fundamental_v2] | 0.3495 | 0.5374 | 49.1% | 45.5% | 1.51x |
| LightGBM [valuation_only] | 0.3480 | 0.5423 | 52.7% | 45.9% | 1.52x |
| LightGBM [fund_simple] | 0.3457 | 0.5517 | 43.6% | 42.3% | 1.40x |
| LightGBM [rank_fundamental] | 0.3422 | 0.5403 | 54.5% | 40.5% | 1.34x |
| ロジスティック回帰 [rank_fundamental_v2] | 0.3414 | 0.5452 | 38.2% | 41.6% | 1.38x |
| ロジスティック回帰 [valuation_only] | 0.3387 | 0.5214 | 49.1% | 44.4% | 1.47x |
| ロジスティック回帰 [fundamental] | 0.3338 | 0.5185 | 49.1% | 40.9% | 1.36x |
| ベースライン: 出来高モメンタムのみ | 0.3330 | 0.5479 | 34.5% | 33.0% | 1.09x |
| ロジスティック回帰 [rank_fundamental] | 0.3276 | 0.5328 | 43.6% | 38.0% | 1.26x |
| ベースライン: 既存の8軸総合スコア | 0.3122 | 0.4972 | 38.2% | 37.3% | 1.24x |
| ベースライン: R_high のみ | 0.2914 | 0.4627 | 45.5% | 32.6% | 1.08x |

（正例率 = 30.15% / n = 5,588）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **ベースライン: R_high のみ**（テスト PR-AUC 0.2914）。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| ロジスティック回帰 [breakout_only] | 0.4220 | +0.1306 | [+0.1115, +0.1522] | 1.000 | 有意 |
| ロジスティック回帰 [technical] | 0.4161 | +0.1247 | [+0.1037, +0.1470] | 1.000 | 有意 |
| LightGBM [technical] | 0.4155 | +0.1240 | [+0.1026, +0.1457] | 1.000 | 有意 |
| ロジスティック回帰 [all_no_market] | 0.3971 | +0.1056 | [+0.0842, +0.1269] | 1.000 | 有意 |
| LightGBM [all_no_market] | 0.3939 | +0.1025 | [+0.0833, +0.1247] | 1.000 | 有意 |
| LightGBM [all] | 0.3932 | +0.1018 | [+0.0809, +0.1239] | 1.000 | 有意 |
| ロジスティック回帰 [all] | 0.3854 | +0.0940 | [+0.0731, +0.1148] | 1.000 | 有意 |
| ロジスティック回帰 [rank_price_only] | 0.3777 | +0.0863 | [+0.0651, +0.1095] | 1.000 | 有意 |
| LightGBM [rank_all] | 0.3777 | +0.0863 | [+0.0662, +0.1070] | 1.000 | 有意 |
| ロジスティック回帰 [rank_technical] | 0.3730 | +0.0816 | [+0.0600, +0.1036] | 1.000 | 有意 |
| LightGBM [rank_price_only] | 0.3725 | +0.0811 | [+0.0605, +0.1030] | 1.000 | 有意 |
| LightGBM [price_only] | 0.3698 | +0.0783 | [+0.0589, +0.1000] | 1.000 | 有意 |
| ロジスティック回帰 [price_only] | 0.3676 | +0.0762 | [+0.0556, +0.1003] | 1.000 | 有意 |
| ロジスティック回帰 [rank_all] | 0.3625 | +0.0711 | [+0.0520, +0.0914] | 1.000 | 有意 |
| LightGBM [rank_technical] | 0.3604 | +0.0689 | [+0.0503, +0.0880] | 1.000 | 有意 |
| LightGBM [fundamental] | 0.3514 | +0.0599 | [+0.0421, +0.0807] | 1.000 | 有意 |
| LightGBM [rank_fundamental] | 0.3422 | +0.0508 | [+0.0323, +0.0709] | 1.000 | 有意 |
| ロジスティック回帰 [fundamental] | 0.3338 | +0.0423 | [+0.0244, +0.0624] | 1.000 | 有意 |
| ロジスティック回帰 [rank_fundamental] | 0.3276 | +0.0362 | [+0.0201, +0.0578] | 1.000 | 有意 |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `breakout_only` | 5 | breakout | 0.4220 | 1.66x |
| `technical` | 14 | price + breakout + volume + liquidity + supply + market | 0.4161 | 1.76x |
| `all_no_market` | 142 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.3971 | 1.75x |
| `all` | 144 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + market | 0.3932 | 1.63x |
| `fund_simple` | 18 | fund_level + price + volume + liquidity + supply + progress + market | 0.3904 | 1.62x |
| `raw_and_rank` | 146 | fund_level + fund_trend + price + volume + liquidity + supply + progress + market + fund_level_rank + fund_trend_rank + price_rank + volume_rank + liquidity_rank + supply_rank + progress_rank | 0.3817 | 1.53x |
| `rank_price_only` | 3 | price_rank | 0.3777 | 1.44x |
| `rank_all` | 144 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + price_rank + breakout_rank + volume_rank + liquidity_rank + supply_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + sector + turnaround + market | 0.3777 | 1.52x |
| `rank_technical` | 9 | price_rank + volume_rank + liquidity_rank + supply_rank + market | 0.3730 | 1.57x |
| `extras_only` | 23 | valuation + dividend + cashflow + efficiency + guidance + sector | 0.3713 | 1.59x |
| `price_only` | 3 | price | 0.3698 | 1.43x |
| `fundamental_v3` | 130 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround | 0.3628 | 1.46x |
| `rank_fundamental_v2` | 126 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank + valuation_rank + dividend_rank + cashflow_rank + efficiency_rank + guidance_rank + turnaround | 0.3612 | 1.53x |
| `fundamental_v2` | 126 | fund_level + fund_lag + fund_trend + fund_streak + progress + valuation + dividend + cashflow + efficiency + guidance + turnaround | 0.3609 | 1.45x |
| `fundamental` | 105 | fund_level + fund_lag + fund_trend + fund_streak + progress | 0.3514 | 1.34x |
| `valuation_only` | 7 | valuation | 0.3480 | 1.52x |
| `rank_fundamental` | 105 | fund_level_rank + fund_lag_rank + fund_trend_rank + fund_streak_rank + progress_rank | 0.3422 | 1.34x |

## 特徴量の寄与（`breakout_only` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `vol_20d` | +0.645 | ブレイクしやすい |
| `base_length` | -0.180 | しにくい |
| `ret_20d` | +0.169 | ブレイクしやすい |
| `break_margin` | -0.103 | しにくい |
| `close_position` | -0.020 | しにくい |

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
