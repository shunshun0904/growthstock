# 価格帯別出来高

「価格帯別出来高」タブには、最新の予測基準日と日付が一致する予測対象を全件表示します。購入条件に達していない銘柄も含みます。J-Quantsの分足・ティックから直近1・5・20営業日を集計し、未集計の銘柄も一覧に表示します。銘柄名を押すとグラフが開きます。

## 保存先の初期設定

1. `shunshun0904/growthstock-minute` を **Private** で作成し、READMEで初期化します。分足・ティックを蓄積する非公開リポジトリがあれば、同じリポジトリを使えます。 [リポジトリ作成](https://github.com/new)
2. 保存先だけを対象とする fine-grained PAT を **Contents: Read and write** で発行し、公開側 `growthstock` の Actions Secret `MINUTE_DATA_TOKEN` に登録します。別名の保存先を使う場合は Actions Variable `MINUTE_DATA_REPO` に `owner/repository` を設定します。 [PAT発行](https://github.com/settings/personal-access-tokens/new) · [Actions Secrets](https://github.com/shunshun0904/growthstock/settings/secrets/actions) · [Actions Variables](https://github.com/shunshun0904/growthstock/settings/variables/actions)
3. J-Quantsで分足・ティックアドオンを有効にした契約のAPIキーを、既存の Actions Secret `JQUANTS_API` に設定します。初回は **Update Volume Profiles** を手動で起動できます。以降は **Predict Breakouts** の成功後に自動実行します。 [J-Quantsのデータ提供仕様](https://jpx-jquants.com/ja/spec/data-spec) · [実行履歴](https://github.com/shunshun0904/growthstock/actions/workflows/update-volume-profile.yml)

更新ジョブは最新予測と同じ日付・銘柄であることを確認して、非公開リポジトリの `dashboard/volume-profile.json` を更新します。公開側に出るのは画面と既存の予測一覧です。アドオンデータと集計結果へのアクセスは非公開リポジトリで管理します。 [J-Quants利用規約 第8条](https://jpx-jquants.com/ja/termsofservice?lang=ja)

## ダッシュボードで見る

1. 保存先だけに **Contents: Read-only** を持つ閲覧用の fine-grained PAT を別途発行します。 [PAT発行](https://github.com/settings/personal-access-tokens/new)
2. 「価格帯別出来高」タブの「閲覧用GitHubトークン」に入力し、「接続」を押します。ブラウザからGitHubに認証し、そのセッションの間はタブを開くと自動で読み込みます。「接続を解除」でセッションから消去できます。 [ダッシュボード](https://shunshun0904.github.io/growthstock/)
3. 一覧で分足・ティックの照合状況が分かります。銘柄を押し、集計期間と株数・売買代金を切り替えられます。予測と集計の基準日がずれている間は、最新予測の一覧に更新待ちを表示します。「再取得」で保存先を読み直します。

## 集計の確認方法

手元で疎通を確認する場合は、端末で `JQUANTS_API_KEY` または `JQUANTS_API` を設定し、`python3 -m pip install -r research/requirements.txt` を実行します。`python3 research/build_volume_profile.py --fetch --lookback 1` で最新日の全対象銘柄を1営業日分集計します。`--codes` と `--as-of` で銘柄と営業日を指定できます。

ティックはBulkのCSV/GZで取得します。日別ファイルと月別ファイルの両方がある日は日別ファイルを使います。ティックは銘柄を指定しても対象日・月の一括ファイルを読むため、初回は1日から確認できます。分足は少数銘柄では銘柄別API、大きい対象ではBulkを使います。公式の提供範囲は直近2年、日次更新は16:30頃です。 [ティック](https://jpx-jquants.com/ja/spec/eq-trades) · [分足](https://jpx-jquants.com/ja/spec/eq-bars-minute) · [Bulk一覧](https://jpx-jquants.com/ja/spec/bulk-list) · [更新時刻](https://jpx-jquants.com/ja/spec/data-update)

ダウンロード済みのデータでも生成できます。`--daily` には `Date,Code,L,H,C,Vo,Va,AdjFactor` を持つ未調整の日足を渡します。分割・併合を識別するため、日足に `ExRT` があれば併せて使います。`--daily`・`--minute`・`--ticks` は複数回指定できます。

```bash
python3 research/build_volume_profile.py --as-of YYYY-MM-DD --codes 7203 \
  --calendar /path/to/calendar.csv --daily /path/to/daily.parquet \
  --minute /path/to/minute.csv.gz --ticks /path/to/trades.csv.gz
```

端末のAPIキーは環境変数で設定します。キャッシュと集計ファイルの既定の保存先 `research/_data/` はGitの対象外です。

## 計算と品質管理

- **ティック**は、東証立会内で成立した約定を価格ごとに合計します。売買代金は約定価格×株数です。
- **分足**は、各分の株数と売買代金を、その分のVWAP（`Va / Vo`）に割り当てます。この割り当てにより、分の中で価格が動くほどティックと分布が異なります。両方揃う場合、同じ価格帯の出来高シェア差の絶対値を合計して2で割った値を表示します。0%は一致、100%は重なる帯がない状態です。
- 期間内の日足の安値・高値から共通の価格帯を作ります。幅は `1/2/5 × 10の整数乗` 円から選び、帯の上限は含めません。最多の帯は株数で決め、同数なら低い帯を選びます。「出来高の70%範囲」は最多の帯から隣の帯を順に含め、累計が70%以上になった連続範囲です。左右の株数が同じなら低い側を先に含めます。
- 各日に日足の出来高・売買代金・高安範囲と照合します。株数の許容差は0.01株未満、売買代金は1円以内です。必要な営業日が揃わない場合は、その期間・種別を欠損にします。日足で取引なしと確認できる日は出来高0として扱い、基準日の終値がない場合は集計しません。
- 分割・併合を跨ぐ期間は、`AdjFactor` と `ExRT` で確認できたものに限って、過去の価格・株数を基準日の単位に換算します。価格と株数を逆方向に調整し、約定代金は維持します。換算根拠が不十分な期間は欠損です。 [日足仕様](https://jpx-jquants.com/ja/spec/eq-bars-daily) · [調整係数](https://jpx-jquants.com/ja/spec/eq-bars-daily/adj)

この分布が示すのは集計期間に成立した取引です。同じ株の売買も期間中の約定ごとに数えます。
