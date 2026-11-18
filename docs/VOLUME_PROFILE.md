# 価格帯別出来高

J-Quantsの分足・ティックアドオンから、銘柄ごとの直近1・5・20営業日の価格帯別出来高を作ります。画面は「価格帯別出来高」タブで表示します。学習用の42列は銘柄・日付単位で別ファイルに保存し、研究用スクリプトで既存マートに結合します。

## データ取得と画面の使い方

1. J-Quantsのダッシュボードでアドオンを有効にし、手元の端末にV2 APIキーを設定します。既存の `JQUANTS_API_KEY` または `JQUANTS_API` を利用できます。 [J-Quantsのデータ提供仕様](https://jpx-jquants.com/ja/spec/data-spec)
2. リポジトリのルートで研究用依存をインストールし、1銘柄・1営業日で疎通を確認します。コード省略時は `public/data/predictions.json` の直近予測対象を使います。日付の既定値は予測の基準日です。利用したい営業日は `--as-of` で指定できます。

   ```bash
   python3 -m pip install -r research/requirements.txt
   python3 research/build_volume_profile.py --fetch --codes 7203 --lookback 1
   ```

3. ローカルで `npm run dev` を起動し、「価格帯別出来高」タブで `research/_data/volume_profile/output/volume-profile.json` を選びます。銘柄・期間・ティック／分足・株数／売買代金を切り替えられます。
4. 通常の更新には `python3 research/build_volume_profile.py --fetch` を使います。予測対象の最大20営業日分から、日足と照合済みのセッションをキャッシュから再利用します。追加契約や取得状況に合わせて `--source minute` または `--source tick` を指定できます。取得済みの訂正を取り込む場合は `--refresh` を付けます。

ティックはBulkのCSV/GZで取得します。日別ファイルと月別ファイルの両方がある日は日別ファイルを使います。ティックは銘柄を指定しても対象日・月の一括ファイルを読むため、初回は1日から確認できます。分足は少数銘柄では銘柄別API、大きい対象ではBulkを使います。公式の提供範囲は直近2年、日次更新は16:30頃です。 [ティック](https://jpx-jquants.com/ja/spec/eq-trades) · [分足](https://jpx-jquants.com/ja/spec/eq-bars-minute) · [Bulk一覧](https://jpx-jquants.com/ja/spec/bulk-list) · [更新時刻](https://jpx-jquants.com/ja/spec/data-update)

ダウンロード済みのデータでも生成できます。`--daily` には `Date,Code,L,H,C,Vo,Va,AdjFactor` を持つ未調整の日足を渡します。分割・併合を識別するため、日足に `ExRT` があれば併せて使います。`--daily`・`--minute`・`--ticks` は複数回指定できます。

```bash
python3 research/build_volume_profile.py --as-of YYYY-MM-DD --codes 7203 \
  --calendar /path/to/calendar.csv --daily /path/to/daily.parquet \
  --minute /path/to/minute.csv.gz --ticks /path/to/trades.csv.gz
```

APIキー・取得データ・集計ファイルは個人の端末に置きます。既定の `research/_data/` はGitの対象外です。画面に渡すデータはファイル選択でブラウザ内に読み込み、ページを再読み込みするまで保持します。アドオンデータの取扱いは契約の利用条件に従います。 [J-Quants利用規約 第8条](https://jpx-jquants.com/ja/termsofservice?lang=ja)

## 計算と品質管理

- **ティック**は、東証立会内で成立した約定を価格ごとに合計します。売買代金は約定価格×株数です。
- **分足**は、各分の株数と売買代金を、その分のVWAP（`Va / Vo`）に割り当てます。この割り当てにより、分の中で価格が動くほどティックと分布が異なります。両方揃う場合、同じ価格帯の出来高シェア差の絶対値を合計して2で割った値を表示します。0%は一致、100%は重なる帯がない状態です。
- 期間内の日足の安値・高値から共通の価格帯を作ります。幅は `1/2/5 × 10の整数乗` 円から選び、帯の上限は含めません。最多の帯は株数で決め、同数なら低い帯を選びます。「出来高の70%範囲」は最多の帯から隣の帯を順に含め、累計が70%以上になった連続範囲です。左右の株数が同じなら低い側を先に含めます。
- 各日に日足の出来高・売買代金・高安範囲と照合します。株数の許容差は0.01株未満、売買代金は1円以内です。必要な営業日が揃わない場合は、その期間・種別を欠損にします。日足で取引なしと確認できる日は出来高0として扱い、基準日の終値がない場合は集計しません。
- 分割・併合を跨ぐ期間は、`AdjFactor` と `ExRT` で確認できたものに限って、過去の価格・株数を基準日の単位に換算します。価格と株数を逆方向に調整し、約定代金は維持します。換算根拠が不十分な期間は欠損です。 [日足仕様](https://jpx-jquants.com/ja/spec/eq-bars-daily) · [調整係数](https://jpx-jquants.com/ja/spec/eq-bars-daily/adj)

この分布が示すのは集計期間に成立した取引です。同じ株の売買も期間中の約定ごとに数えます。以前の「価格帯別需給」が推定していた信用建玉・保有コストとは、計測する対象が異なります。

## 研究用の特徴量と検証

`latest-volume-features.parquet` は表示対象・基準日の特徴量です。過去の特徴量は、学習マートの `Code,Date` を指定して出力します。ティックと分足の各3期間について、以下の7指標を保存します。

| 列名の指標部分 | 意味 |
| --- | --- |
| `poc_gap` | 最多出来高の帯の中点 ÷ 終値 − 1 |
| `vwap_gap` | 期間VWAP ÷ 終値 − 1 |
| `value_low_gap` / `value_high_gap` | 出来高の70%範囲の下限・上限 ÷ 終値 − 1 |
| `poc_share` | 最多出来高の帯に含まれる株数の割合 |
| `above_close_share` | 終値を上回る価格に配分された株数の割合 |
| `near_close_share` | 終値の±1%以内に配分された株数の割合 |

列名は `vp_tick_poc_gap_20d` のように、種別・指標・期間を含みます。履歴の集計は原則として、評価日の19:00（日本時間）までに端末で取得を確認した版を使います。新規取得の過去データを用いる研究では `--historical` を明示し、公式の更新予定を踏まえて各日の17:00に利用可能だったと仮定します。実際の初回配信時刻・過去の訂正前の版は未確認です。`volume-features.meta.json` に集計方法とファイルのハッシュ、`coverage.json` に算出状況を記録します。

```bash
python3 research/build_volume_profile.py --fetch --targets /path/to/dataset.parquet \
  --from-date YYYY-MM-DD --to-date YYYY-MM-DD --historical

python3 research/exp/volume_profile_ablation.py \
  --dataset /path/to/dataset.parquet \
  --features research/_data/volume_profile/output/volume-features.parquet \
  --output research/_data/volume_profile/experiment/class10 \
  --prices /path/to/adjusted_daily.parquet --task class10 --run
```

比較対象は既存特徴量、＋分足、＋ティック、＋両方の4群です。結合後マートには元の全行を保持し、比較では両種別の20営業日集計が揃った共通行を使います。初回は約12か月を訓練に使い、3か月ごとにウォークフォワードで評価します。20営業日先の目的変数が未確定の行を学習から除きます。共通パラメータでの比較に加え、各群を同じ30候補・内側3窓で個別に再チューニングします。LightGBM +10%分類は既存の目的変数でAP等を、`--task ltr` は翌営業日寄りから20営業日目の終値までのリターンで日付内NDCG@5と上位銘柄リターンを出します。結果は `tuning.json`、`walk-forward.csv` と `feature-coverage.json` に保存します。

実データでの疎通、算出率、分足近似の精度、予測力はアドオン利用環境で検証します。学習・予測の日次ワークフローへの組み込みは、比較結果を基に判断します。 [本番採用のルール](MODEL_ADOPTION_RULES.md)
