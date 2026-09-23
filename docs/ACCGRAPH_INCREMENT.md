# 会計フローグラフ — EDINET の明細の上積み（2段構え）

`research/accgraph/increment.py` の出力。**実行した結果のみ**を載せる。

## やったこと

明細ありの群だけで学習すると件数が足りず、全体に混ぜると明細は9割以上が
欠測のまま学習される（`docs/ACCGRAPH_DIAGNOSE.md`）。そこで役割を分けた。

1. **1段目**: J-Quants のノードだけ（`latest_jq`）で、全体で学習する
2. **2段目**: 明細ありの群だけで、1段目の予測を固定したまま、明細の特徴量で補正する分だけを学習する

2段目は正則化を強めると補正がゼロに縮み、1段目の予測に戻る。補正の強さは訓練行の中だけの前向き検証で選び、テスト窓は見ない。

- ベンチマーク: TOPIX控除 / 20営業日
- 明細ありの群: 2,923件・106社（うち評価のテスト窓に 1,920件）
- 1段目の walk-forward: 最初の訓練 24か月から（2段目の訓練行にもアウトオブサンプルの予測を付けるため）
- 補正の強さの候補: 10.0, 1.0, 0.3, 0.1, 0.03, 0.01（大きいほど1段目に近い）
- 区間: 発表日単位のブートストラップ 1,000回の95%区間

## 比べるもの

| 名前 | 中身 |
| --- | --- |
| 1段目のまま | 全体で学習した J-Quants のモデルの予測 |
| 明細なしの2段目 | 同じ仕組みで、明細の特徴量を入れず切片だけ学習したもの（群に合わせた較正し直し） |
| 明細ありの2段目 | 同じ仕組みで、明細の特徴量を入れたもの |

明細の効果は **明細ありの2段目 − 明細なしの2段目** で測る。較正し直しの効果と明細の効果を分けるため。

## 判定基準（結果を見る前に決めたもの）

差 = AUC(明細ありの2段目) − AUC(明細なしの2段目)。

| 判定 | 条件 |
| --- | --- |
| 明細が効く | 差の95%区間が 0 を上回り、かつ偽の明細の差をすべて上回る |
| 明細が害になる | 差の95%区間が 0 を下回り、かつ偽の明細の差をすべて下回る |
| 差が見えない | それ以外 |

偽の明細は、明細の特徴量を明細ありの行の間で入れ替えたもの（19回）。区間はテスト行の引き直しだけで、補正の学習そのものの揺れを含まないので、これで補う。

判定に使うのは明細の特徴量を絞った版（compact）だけ。全部入れた版（full）は参考で、偽の明細は回さない。

## 結果

| 1段目 | 明細の特徴量 | 件数 | 1段目のまま | 明細なしの2段目 | 明細ありの2段目 | 明細あり − なし | 偽の明細の差（最小〜最大） | 判定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logit | compact（72列） | 1,920 | 0.5357 [0.516, 0.555] | 0.5331 [0.512, 0.553] | 0.5328 [0.511, 0.553] | -0.0003 [-0.002, +0.002] | -0.0011〜+0.0021 | **差が見えない** |
| logit | full（202列） | 1,920 | 0.5357 [0.516, 0.555] | 0.5331 [0.512, 0.553] | 0.5333 [0.511, 0.554] | +0.0002 [-0.004, +0.004] | — | （参考）差が見えない |
| lgbm | compact（72列） | 1,920 | 0.5215 [0.500, 0.542] | 0.5204 [0.498, 0.542] | 0.5206 [0.499, 0.542] | +0.0002 [-0.001, +0.002] | -0.0007〜+0.0009 | **差が見えない** |
| lgbm | full（202列） | 1,920 | 0.5215 [0.500, 0.542] | 0.5204 [0.498, 0.542] | 0.5220 [0.501, 0.544] | +0.0016 [-0.001, +0.004] | — | （参考）差が見えない |

## 判定の根拠

- `logit/compact`: **差が見えない** — 全フォールドで補正をほぼゼロにする設定が選ばれた。訓練データの中の前向き検証で、明細が予測を良くする設定が見つからなかった。区間が狭いのは補正がほぼゼロだからで、上積みの上限を示すものではない
- `logit/full`: **差が見えない** — 全フォールドで補正をほぼゼロにする設定が選ばれた。訓練データの中の前向き検証で、明細が予測を良くする設定が見つからなかった。区間が狭いのは補正がほぼゼロだからで、上積みの上限を示すものではない
- `lgbm/compact`: **差が見えない** — 全フォールドで補正をほぼゼロにする設定が選ばれた。訓練データの中の前向き検証で、明細が予測を良くする設定が見つからなかった。区間が狭いのは補正がほぼゼロだからで、上積みの上限を示すものではない
- `lgbm/full`: **差が見えない** — 全フォールドで補正をほぼゼロにする設定が選ばれた。訓練データの中の前向き検証で、明細が予測を良くする設定が見つからなかった。区間が狭いのは補正がほぼゼロだからで、上積みの上限を示すものではない

## 較正し直しの効果（参考）

| 1段目 | 明細の特徴量 | 明細なしの2段目 − 1段目のまま |
| --- | --- | ---: |
| logit | compact | -0.0026 [-0.009, +0.003] |
| logit | full | -0.0026 [-0.009, +0.003] |
| lgbm | compact | -0.0011 [-0.005, +0.003] |
| lgbm | full | -0.0011 [-0.005, +0.003] |

## フォールドごとの2段目

補正の強さが毎回最大（10.0）なら、訓練行の中の検証で明細に情報が見つからなかったということ。

| 1段目 | 明細の特徴量 | フォールド | 2段目の訓練件数 | テスト件数 | 選ばれた強さ |
| --- | --- | :-: | ---: | ---: | ---: |
| logit | compact | 0 | 563 | 302 | 10 |
| logit | compact | 1 | 867 | 281 | 10 |
| logit | compact | 2 | 1,147 | 306 | 10 |
| logit | compact | 3 | 1,451 | 339 | 10 |
| logit | compact | 4 | 1,793 | 330 | 10 |
| logit | compact | 5 | 2,121 | 362 | 10 |
| logit | full | 0 | 563 | 302 | 10 |
| logit | full | 1 | 867 | 281 | 10 |
| logit | full | 2 | 1,147 | 306 | 10 |
| logit | full | 3 | 1,451 | 339 | 10 |
| logit | full | 4 | 1,793 | 330 | 10 |
| logit | full | 5 | 2,121 | 362 | 10 |
| lgbm | compact | 0 | 563 | 302 | 10 |
| lgbm | compact | 1 | 867 | 281 | 10 |
| lgbm | compact | 2 | 1,147 | 306 | 10 |
| lgbm | compact | 3 | 1,451 | 339 | 10 |
| lgbm | compact | 4 | 1,793 | 330 | 10 |
| lgbm | compact | 5 | 2,121 | 362 | 10 |
| lgbm | full | 0 | 563 | 302 | 10 |
| lgbm | full | 1 | 867 | 281 | 10 |
| lgbm | full | 2 | 1,147 | 306 | 10 |
| lgbm | full | 3 | 1,451 | 339 | 10 |
| lgbm | full | 4 | 1,793 | 330 | 10 |
| lgbm | full | 5 | 2,121 | 362 | 10 |

## 明細の特徴量（compact）

`cost_of_sales.to_sales`, `cost_of_sales.to_assets`, `cost_of_sales.yoy_sym`, `cost_of_sales.sign`, `cost_of_sales.is_missing`, `sga.to_sales`, `sga.to_assets`, `sga.yoy_sym`, `sga.sign`, `sga.is_missing`, `rnd.to_sales`, `rnd.to_assets`, `rnd.yoy_sym`, `rnd.sign`, `rnd.is_missing`, `pretax_profit.to_sales`, `pretax_profit.to_assets`, `pretax_profit.yoy_sym`, `pretax_profit.sign`, `pretax_profit.is_missing`, `depreciation.to_sales`, `depreciation.to_assets`, `depreciation.yoy_sym`, `depreciation.sign`, `depreciation.is_missing`, `capex.to_sales`, `capex.to_assets`, `capex.yoy_sym`, `capex.sign`, `capex.is_missing`, `inventories.to_sales`, `inventories.to_assets`, `inventories.yoy_sym`, `inventories.sign`, `inventories.is_missing`, `trade_receivables.to_sales`, `trade_receivables.to_assets`, `trade_receivables.yoy_sym`, `trade_receivables.sign`, `trade_receivables.is_missing`, `trade_payables.to_sales`, `trade_payables.to_assets`, `trade_payables.yoy_sym`, `trade_payables.sign`, `trade_payables.is_missing`, `working_capital.to_sales`, `working_capital.to_assets`, `working_capital.yoy_sym`, `working_capital.sign`, `working_capital.is_missing`, `interest_bearing_debt.to_sales`, `interest_bearing_debt.to_assets`, `interest_bearing_debt.yoy_sym`, `interest_bearing_debt.sign`, `interest_bearing_debt.is_missing`, `cost_of_sales->cogs_sga.ratio`, `sga->cogs_sga.ratio`, `rnd->sga.ratio`, `ordinary_profit->pretax_profit.ratio`, `pretax_profit->net_income.ratio`, `pretax_profit->cfo.ratio`, `depreciation->cfo.ratio`, `working_capital->cfo.ratio`, `inventories->working_capital.ratio`, `trade_receivables->working_capital.ratio`, `trade_payables->working_capital.ratio`, `capex->cfi.ratio`, `inventories->total_assets.ratio`, `trade_receivables->total_assets.ratio`, `trade_payables->liabilities.ratio`, `interest_bearing_debt->liabilities.ratio`, `edinet.age_years`

## 読み方

- 「1段目のまま」は切り分け（`docs/ACCGRAPH_DIAGNOSE.md`）の `latest_jq` の「明細あり」列と一致するはず。評価のテスト窓も、その窓の予測を出すモデルの訓練データも同じ。ずれていれば、2段目の件数不足で外したフォールドがある（上のフォールド表で分かる）か、どこかが食い違っている。
- 差が見えないときは、まずフォールド表の「選ばれた強さ」を見る。毎回最大なら補正はほぼゼロで、明細あり版と明細なし版はほぼ同じ予測になる。そのときの区間の狭さは上積みの上限ではなく、訓練データの中で明細が役立つ設定が見つからなかったことを表す。件数が増えれば結果が変わりうる。
- 明細ありの群は本流の母集団の順に取得しているので、大型・人気株に偏っている。ここでの結果は、この群についてのもの。
