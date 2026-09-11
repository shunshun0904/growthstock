# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 3ヶ月内+1.2σ / 終盤+0.50倍 / MA20>=MA60
- データセット: 22,887サンプル / 全体の正例率 **20.85%**
- 期間: 2018-04-03 〜 2026-06-16 / 銘柄数 4,008
- 分割: **直近12ヶ月をホールドアウト**（2025-06-16 〜 2026-06-16）。それ以前を探索と学習に使う
- **エンバーゴ 60営業日**（ラベル確定に必要な将来日数から自動導出）。2025-03-21 〜 2025-06-16 は捨てる
- ハイパーパラメータの探索もホールドアウトより前だけで行う（`research/run_tuning.py` が同じ境界から打ち切り日を取る）

  - train: 17,449件 / 2018-04-03 〜 2025-03-21 / 正例率 18.71%
  - test: 4,734件 / 2025-06-16 〜 2026-06-16 / 正例率 27.67%


## ホールドアウト（直近12ヶ月）— 唯一の成績

| モデル | PR-AUC | ROC-AUC | 日付内AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LTR [all] | 0.4115 | 0.6345 | 0.6021 | 74.5% | 55.9% | 2.02x |
| LightGBM [all_no_sector_index] | 0.4089 | 0.6261 | 0.6113 | 63.8% | 57.6% | 2.08x |
| LightGBM [all] | 0.4019 | 0.6219 | 0.6054 | 57.4% | 55.1% | 1.99x |
| LTR [all_no_sector_index] | 0.4000 | 0.6279 | 0.5947 | 63.8% | 55.1% | 1.99x |
| ロジスティック回帰 [all] | 0.3900 | 0.5960 | 0.5679 | 72.3% | 55.1% | 1.99x |
| ロジスティック回帰 [all_no_sector_index] | 0.3869 | 0.5936 | 0.5593 | 72.3% | 55.1% | 1.99x |
| LightGBM [all_no_market] | 0.3834 | 0.6176 | 0.5884 | 61.7% | 47.5% | 1.71x |
| LTR [all_no_market] | 0.3703 | 0.6153 | 0.5811 | 53.2% | 46.2% | 1.67x |
| ロジスティック回帰 [all_no_market] | 0.3419 | 0.5934 | 0.5529 | 51.1% | 35.2% | 1.27x |
| 無情報（一定スコア） | 0.2767 | 0.5000 | 0.5000 | 27.7% | 27.7% | 1.00x |

（正例率 = 27.67% / n = 4,734）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **無情報（一定スコア）**（テスト PR-AUC 0.2767）。
単変量のベースラインは廃止した。母集団を高値更新日にした時点で
「高値からの距離」は全銘柄で同じになり、それを基準にしても
何も言えないため。全件同じスコアを与える無情報モデルなら、
PR-AUC はその窓の正例率に一致し、差は「正例率をどれだけ
上回ったか」になる。母集団やラベルの定義を変えても意味が変わらない。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| LTR [all] | 0.4115 | +0.1348 | [+0.1132, +0.1572] | 1.000 | 有意 |
| LightGBM [all_no_sector_index] | 0.4089 | +0.1322 | [+0.1105, +0.1566] | 1.000 | 有意 |
| LightGBM [all] | 0.4019 | +0.1251 | [+0.1042, +0.1477] | 1.000 | 有意 |
| LTR [all_no_sector_index] | 0.4000 | +0.1233 | [+0.1008, +0.1446] | 1.000 | 有意 |
| ロジスティック回帰 [all] | 0.3900 | +0.1133 | [+0.0933, +0.1375] | 1.000 | 有意 |
| ロジスティック回帰 [all_no_sector_index] | 0.3869 | +0.1102 | [+0.0901, +0.1339] | 1.000 | 有意 |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `all` | 151 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + sector_index + market | 0.4115 | 2.02x |
| `all_no_sector_index` | 147 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + market | 0.4089 | 2.08x |
| `all_no_market` | 140 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + sector_index | 0.3834 | 1.71x |

## 特徴量の寄与（`all` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `rel_sector_20` | +0.736 | ブレイクしやすい |
| `vol_20d` | -0.702 | しにくい |
| `ROA_q1` | +0.601 | ブレイクしやすい |
| `ret_20d` | -0.592 | しにくい |
| `ROE_q3` | -0.551 | しにくい |
| `ROA_q0` | -0.534 | しにくい |
| `sales_growth_q0` | -0.452 | しにくい |
| `eps_growth_q1` | -0.441 | しにくい |
| `equity_ratio_chg_3q` | +0.435 | ブレイクしやすい |
| `sales_growth_sym_chg_3q` | -0.418 | しにくい |
| `equity_ratio_q3` | -0.390 | しにくい |
| `ROE_q2` | +0.390 | ブレイクしやすい |
| `ROE_chg3` | -0.384 | しにくい |
| `sales_growth_chg_3q` | +0.383 | ブレイクしやすい |
| `sales_growth_sym_chg1` | +0.352 | ブレイクしやすい |

## 読み方

- **PR-AUC** が主指標。下限は正例率で、それを大きく上回るほど良い
- **Lift@5%** は「スコア上位5%の正例率 ÷ 全体の正例率」。1.0 なら無意味。同点は期待値で分けるので、無情報モデルは必ず 1.00倍 になる
- **無情報（一定スコア）** を上回らなければ、**特徴量に予測力が無い**という結論になる
- この表はテスト期間1本の結果でしかない。局面をまたいで再現するかは[MODEL_WALKFORWARD.md](MODEL_WALKFORWARD.md)、規模の効果を除いても残るかは[MODEL_STRATIFIED.md](MODEL_STRATIFIED.md) で見る
- 数字が極端に良い場合はまずリークを疑う（`tests/test_dataset.py` の先読み検出テストを参照）

## 特徴量を足して試すには

1. `research/features.py` の `GROUPS` に列を足す
2. `research/build_dataset.py` でその列を作る
3. データセットを再構築（Release から読むので数分・API取得なし）
4. `PRESETS` にセットを1行足して再学習
