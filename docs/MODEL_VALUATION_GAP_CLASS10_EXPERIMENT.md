# 理論株価特徴量とLightGBM固定+10%分類：実験132追補

2026-11-15。研究用。本番モデル・特徴量プリセット・学習パラメータ・予測結果は変更しない。
[ランキングの比較](MODEL_VALUATION_GAP_EXPERIMENT.md)の結果から分類モデルの効果は推定せず、別に実測する。

## 評価前に固定する条件

- 本番と同じ`label_specs.for_model("lgbm")`。翌営業日寄り基準、20営業日以内の終値で+10%到達、20営業日目の直近5営業日平均終値が買値の+5%以上、同日のMA5≧MA20。
- 前回と同じ保存済みマート22,110行・3,748銘柄、同じ価格ラベル・特徴量マートを使う。
- 外側は36か月訓練・6か月評価、最後の2窓。20日ラベルが確定する前の訓練行を除外する。
- 対象はLightGBMの+10%分類だけ。`--tasks class10`で指定する。
- 既存Aを含め全7群を各窓で別々に再調整する。各30候補・内側3窓、同じ候補集合・探索種。
- 内側Average Precisionで設定を選び、全14設定を凍結後に外側を採点する。
- 木200本、最終種42/43/44の予測を平均。訓練ラベルだけから本番と同じ`scale_pos_weight`を計算する。
- 全群共通の事前中立設定による固定比較も行う。理論株価側のERP・成長率・収束期間は既定シナリオを使う。
- 欠損を含む全行と、EVA・RIM両ギャップがある共通行を別に集計する。共通行への絞り込みで再学習しない。
- E120/E121で既参照の期間。独立ホールドアウトとは呼ばず、本番採用の確定判定にも使わない。

## 比較群

| 群 | 内容 |
|---|---|
| A | 既存373特徴量 |
| B | A＋資本効率・簿価WACC・品質 |
| C | B＋現金区分なしEVAのギャップ等 |
| D | C＋ギャップの時系列 |
| R | A＋外生RIMの入力・ギャップ・時系列 |
| P | Dの数値を日付内・欠損パターン内で入れ替える対照。品質は保持 |
| M | B＋価格・PER・PBRの時系列対照 |

## 指標と解釈

主指標はPR-AUCとしてのAverage Precision（AP）。台形積分PR-AUCではない。
ROC-AUC、窓内全レコードのスコア上位10%の適合率、同じ上位10%の20日終値収益も保存する。
上位10%は日次順位ではなく各評価窓をまとめた順位で、運用時の固定スコア閾値の検証とは異なる。
クラス重み付きの生スコアを校正済みの到達確率とは解釈しない。
APの基準となる正例率・行数・日数を各集合ごとに報告する。
全行と共通行では母集団が違うため、各集合内で群Aとの差を見る。
主表は2窓の指標の単純平均。期間別の符号、P/M対照、固定・再調整の両方を確認する。

## 実行

```sh
python research/exp/e132_valuation_gap.py --dataset /private/dataset.parquet \
  --feature-mart /private/valuation-features --prices /private/adjusted-prices.parquet \
  --output /private/e132-class10-retuned-2fold --run --tasks class10 --last-folds 2 \
  --retune --trials 30 --inner-folds 3 --trees 200 --threads 1 --jobs 2

python research/exp/e132_valuation_gap.py --dataset /private/dataset.parquet \
  --feature-mart /private/valuation-features --prices /private/adjusted-prices.parquet \
  --output /private/e132-class10-fixed-2fold --run --tasks class10 --last-folds 2 --threads 1
```

`--tasks`と従来の`--classification`の併用、重複タスクは拒否する。
従来の引数なしはLTRのみ、従来の`--classification`はLTR/+10/+15のまま。
新規分類出力には正解ラベルと共通行フラグを保存し、独立に同一行・評価値を照合する。
過去のランキング出力・コード指紋・レポートは上書きしない。

## 状況

実装とラベル・分類器契約のテストを完了し、全群の再調整を実施する。
分類の性能比較はまだ未確定。結果を見てから特徴量集合や理論価値の仮定を調整しない。
