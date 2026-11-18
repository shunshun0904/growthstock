# 価格帯別出来高の特徴量検証

価格帯別出来高の集計方法は公開画面と共通です。検証用 CLI が学習マートの銘柄・日付に合わせて特徴量を作成します。 [表示と集計の仕様](VOLUME_PROFILE.md)

## 特徴量

分足・ティックの各1・5・20営業日について次の7指標を保存します。列名は `vp_tick_poc_gap_20d` のように、種別・指標・期間を含みます。

| 指標 | 定義 |
| --- | --- |
| `poc_gap` | 最多出来高の帯の中点 ÷ 終値 − 1 |
| `vwap_gap` | 期間VWAP ÷ 終値 − 1 |
| `value_low_gap` / `value_high_gap` | 出来高の70%範囲の下限・上限 ÷ 終値 − 1 |
| `poc_share` | 最多出来高の帯に含まれる株数の割合 |
| `above_close_share` | 終値を上回る価格に配分された株数の割合 |
| `near_close_share` | 終値の±1%以内に配分された株数の割合 |

日足と照合済みの全営業日が揃う場合に、その期間の指標を計算します。評価時点は既定で各営業日の19:00（日本時間）です。端末で取得を確認した時刻を利用します。過去分の取得を使った分析では `--historical` を指定すると、公式の更新予定から各日の17:00に利用可能だったと仮定します。訂正前の版と実際の初回配信時刻には未確認の期間があるため、後者は探索的な比較として記録します。 [公式の更新予定](https://jpx-jquants.com/ja/spec/data-update)

## 実データの検証手順

1. まず銘柄・日付ごとに未調整の日足の出来高・売買代金との照合率を確認します。両方揃う日ではティック分布を基準に、分足の価格帯別シェア、最多の帯と出来高70%の範囲の差を確認します。
2. 学習マートと結合し、種別・期間ごとの算出率と、両種別で20営業日が揃う共通行数を確認します。サンプルの減少による成績の変化を区別するため、4群の評価には同じ共通行を使います。
3. 既存特徴量、＋分足、＋ティック、＋両方の4群を比較します。約12か月の訓練から開始し、3か月ごとにウォークフォワードで評価します。20営業日先の結果が未確定の行は学習から除きます。共通パラメータの比較に加え、各群を同じ30候補・内側3窓で個別に再チューニングします。+10%分類は既存目的変数でAP・上位10%の精度、20営業日ランキングは日付内NDCG@5・上位銘柄リターンを比較します。

```bash
python3 research/build_volume_profile_features.py --fetch --targets /path/to/dataset.parquet \
  --from-date YYYY-MM-DD --to-date YYYY-MM-DD --historical

python3 research/exp/volume_profile_ablation.py \
  --dataset /path/to/dataset.parquet \
  --features research/_data/volume_profile/research/volume-features.parquet \
  --output research/_data/volume_profile/experiment/class10 \
  --prices /path/to/adjusted_daily.parquet --task class10 --run
```

分足・ティックのファイルを取得済みの場合は、`--fetch` の代わりに `--calendar`・`--daily`・`--minute`・`--ticks` で読み込めます。`--task ltr` でランキングを評価します。`volume-features.meta.json` に利用可能時点の扱い、算出状況、ファイルのハッシュを保存し、`feature-coverage.json`、`tuning.json`、`walk-forward.csv` に比較結果を記録します。

## 確認状況

2026-11-18に合成入力で、42列の生成、元マートの行とラベルの維持、後続日を追加したときの過去行の不変性、評価時刻までに利用できるデータだけを使う処理を確認しました。+10%分類と20営業日ランキングは、合成データで4群を同じ行に揃え、2候補・内側3窓で群ごとに再チューニングが完走しました。各内側窓の学習ラベルは検証初日より前に確定していました。

利用可能な環境にアドオンデータとAPI認証が揃った時点で、実データの照合率・算出率・予測力を記録します。本番モデルへの採否は実データの結果を確認した上で判断します。 [本番採用ルール](MODEL_ADOPTION_RULES.md)
