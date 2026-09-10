# 新高値ブレイクアウト予測モデル 結果

`research/train_model.py` の出力。**実測値のみ**を記載する。
設計と方法論は [MODEL_DESIGN.md](MODEL_DESIGN.md) を参照。

## 条件

- **ラベル定義**: 3ヶ月内+20% / 維持10日 / 終盤+10% / MA20>=MA60
- データセット: 22,936サンプル / 全体の正例率 **11.50%**
- 期間: 2018-04-03 〜 2026-06-16 / 銘柄数 4,017
- 分割: **直近12ヶ月をホールドアウト**（2025-06-16 〜 2026-06-16）。それ以前を探索と学習に使う
- **エンバーゴ 60営業日**（ラベル確定に必要な将来日数から自動導出）。2025-03-21 〜 2025-06-16 は捨てる
- ハイパーパラメータの探索もホールドアウトより前だけで行う（`research/run_tuning.py` が同じ境界から打ち切り日を取る）

  - train: 17,485件 / 2018-04-03 〜 2025-03-21 / 正例率 10.23%
  - test: 4,746件 / 2025-06-16 〜 2026-06-16 / 正例率 14.94%


## ホールドアウト（直近12ヶ月）— 唯一の成績

| モデル | PR-AUC | ROC-AUC | 日付内AUC | P@1% | P@5% | Lift@5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LTR [all_no_sector_index] | 0.1916 | 0.6004 | 0.6186 | 21.3% | 19.4% | 1.30x |
| LTR [all_no_market] | 0.1915 | 0.5962 | 0.6066 | 14.9% | 24.5% | 1.64x |
| LightGBM [all_no_market] | 0.1904 | 0.5944 | 0.6064 | 23.4% | 21.1% | 1.41x |
| LTR [all] | 0.1863 | 0.5909 | 0.6172 | 23.4% | 19.8% | 1.33x |
| LightGBM [all] | 0.1795 | 0.5807 | 0.6288 | 19.1% | 18.1% | 1.21x |
| LightGBM [all_no_sector_index] | 0.1771 | 0.5610 | 0.5923 | 19.1% | 22.4% | 1.50x |
| ロジスティック回帰 [all_no_market] | 0.1728 | 0.5614 | 0.5815 | 21.3% | 15.2% | 1.02x |
| ロジスティック回帰 [all] | 0.1673 | 0.5403 | 0.5874 | 17.0% | 17.3% | 1.16x |
| ロジスティック回帰 [all_no_sector_index] | 0.1666 | 0.5359 | 0.5838 | 17.0% | 17.7% | 1.19x |
| 無情報（一定スコア） | 0.1494 | 0.5000 | 0.5000 | 14.9% | 14.9% | 1.00x |

（正例率 = 14.94% / n = 4,746）

## 差は誤差か（対応のあるブートストラップ B=1000）

基準は **無情報（一定スコア）**（テスト PR-AUC 0.1494）。
単変量のベースラインは廃止した。母集団を高値更新日にした時点で
「高値からの距離」は全銘柄で同じになり、それを基準にしても
何も言えないため。全件同じスコアを与える無情報モデルなら、
PR-AUC はその窓の正例率に一致し、差は「正例率をどれだけ
上回ったか」になる。母集団やラベルの定義を変えても意味が変わらない。
95%CI が 0 をまたぐ場合、その差は誤差と区別できない。

| モデル | PR-AUC | 差 | 95%CI | P(差>0) | 判定 |
| --- | ---: | ---: | :---: | ---: | --- |
| LTR [all_no_sector_index] | 0.1916 | +0.0423 | [+0.0280, +0.0594] | 1.000 | 有意 |
| LTR [all_no_market] | 0.1915 | +0.0421 | [+0.0293, +0.0597] | 1.000 | 有意 |
| LightGBM [all_no_market] | 0.1904 | +0.0410 | [+0.0282, +0.0597] | 1.000 | 有意 |
| LTR [all] | 0.1863 | +0.0369 | [+0.0239, +0.0538] | 1.000 | 有意 |
| LightGBM [all] | 0.1795 | +0.0301 | [+0.0191, +0.0448] | 1.000 | 有意 |
| LightGBM [all_no_sector_index] | 0.1771 | +0.0277 | [+0.0152, +0.0435] | 1.000 | 有意 |
| ロジスティック回帰 [all] | 0.1673 | +0.0179 | [+0.0065, +0.0324] | 0.998 | 有意 |

## 特徴量セット別の比較（テストデータ・2モデルのうち良いほう）

| セット | 列数 | 構成 | PR-AUC | Lift@5% |
| --- | ---: | --- | ---: | ---: |
| `all_no_sector_index` | 147 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + market | 0.1916 | 1.30x |
| `all_no_market` | 140 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + sector_index | 0.1915 | 1.64x |
| `all` | 151 | fund_level + fund_lag + fund_trend + fund_streak + price + breakout + volume + liquidity + supply + progress + valuation + dividend + cashflow + efficiency + guidance + sector + turnaround + scale + sector_index + market | 0.1863 | 1.33x |

## 特徴量の寄与（`all_no_sector_index` のロジスティック回帰・標準化係数 上位15）

| 特徴量 | 係数 | 向き |
| --- | ---: | --- |
| `ROA_q1` | +0.704 | ブレイクしやすい |
| `ROA_q0` | -0.645 | しにくい |
| `ROE_accel` | -0.540 | しにくい |
| `sales_growth_up_streak` | +0.503 | ブレイクしやすい |
| `ROE_chg1` | +0.491 | ブレイクしやすい |
| `sales_growth_sym_chg_3q` | -0.483 | しにくい |
| `sales_growth_sym_up_streak` | -0.471 | しにくい |
| `log_market_cap` | -0.458 | しにくい |
| `equity_ratio_chg` | +0.441 | ブレイクしやすい |
| `equity_ratio_q2` | -0.434 | しにくい |
| `sales_growth_chg_3q` | +0.430 | ブレイクしやすい |
| `sales_growth_chg3` | -0.404 | しにくい |
| `ROA_chg_3q` | -0.387 | しにくい |
| `ROA_chg1` | +0.382 | ブレイクしやすい |
| `ROE_q2` | +0.372 | ブレイクしやすい |

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
