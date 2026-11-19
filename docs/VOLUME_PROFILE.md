# 価格帯別出来高

「価格帯別出来高」タブには、**直近5営業日の予測対象（候補）すべて**の、予測の基準日までの 1・5・20 営業日の
価格帯別出来高を表示します。あわせて次の指標を出します。

- **終値とVWAPの差**（終値 ÷ VWAP − 1）
- **時間帯別の出来高**（寄り・前場・後場・引け。株数の割合）
- **約定代金の大きさ別の割合**（小口・中口・大口。ティックのザラバの約定から）

集計は J-Quants のアドオン（分足・ティック）から作るため、**暗号化して**載せます。画面で J-Quants の API キーを
入れた人のブラウザの中だけで開きます。キーはどこにも送りません。新しい鍵や Secret の登録は要りません。

## 見かた

1. 「価格帯別出来高」タブを開き、**各端末で最初の1回だけ**、「J-Quants の API キー」に Actions Secret `JQUANTS_API` に
   登録してあるのと同じキーを入れて「開く」を押します。次からは入力なしで開きます。 [ダッシュボード](https://shunshun0904.github.io/growthstock/)
   - キーは、この端末の中で復号の鍵を作るのに使うだけです。どこにも送りません
   - 覚えるのはキーから作った鍵だけで、キーそのものは保存しません。既定でこの端末に覚えます
     （運用者の選択 2026-11-19「最初の1回だけ入力」）。共用の端末では「この端末に記憶する」を外すと、タブを閉じるまでだけ覚えます。
     「鍵を消す」で消えます
   - J-Quants のキーを作り直したら、Secret `JQUANTS_API` も新しいキーに替えてください。次の集計（毎晩）から新しいキーで開けます
2. 銘柄は「銘柄」のプルダウンで選びます（コードを打つ必要はありません）。一覧の表には、直近5つの予測日の候補が銘柄ごとに
   1行で並びます（候補になった日と、その日の順位）。表の行を押しても選べます。
3. 集計期間（1・5・20営業日）、データ種別（ティック・分足）、横軸（株数・売買代金）を切り替えられます。
   一覧の「終値とVWAPの差」「大口」は、選んだ期間の値です。
4. 揃わない期間・種別は、値を作らずに理由を出します（下の「欠損の理由」）。

## 更新の流れ

```
Update Data Store（毎晩の取り込み）
  ├─ Predict Breakouts   予測（public/data/predictions.json）
  ├─ Update Minute Bars  分足を下書きの Release data-minute に足す
  └─ Update Ticks        ティックを下書きの Release data-tick-YYYY に足す
        │ それぞれが終わるたびに
        ▼
Update Volume Profiles   集計して暗号化し、Release data-volume-profile の volume-profile.enc.json に上げる
        ▼
Deploy to GitHub Pages   volume-profile.enc.json を画面の data/ に置く
```

- ワークフローは `.github/workflows/update-volume-profile.yml`、集計は `research/build_volume_profile.py`
- 入力は、分足・ティックの置き場（下書きの Release。[MINUTE_BARS.md](MINUTE_BARS.md)・[TICKS.md](TICKS.md)）、
  照合用の日足（Release `data-raw` の `bars_YYYY.parquet`）、J-Quants の取引カレンダー
- 予測・分足・ティックの3つは並んで走り、終わる順は日によって違います。最後に終わったものの後の集計が、全部そろったものです。
  そろう前の集計では、まだ置き場に無い日の種別が「欠損」と出ます
- 使う Secret は `JQUANTS_API` だけです。置き場の読み書きは Actions の鍵（`github.token`）です
- 手で作り直すときは Actions の **Update Volume Profiles** を Run workflow で起動します。 [実行履歴](https://github.com/shunshun0904/growthstock/actions/workflows/update-volume-profile.yml)

## 暗号化

`research/volume_profile_crypto.py` が暗号化し、画面（`src/lib/volumeProfile.js`）が同じ手順で開きます。

| 項目 | 中身 |
|---|---|
| 鍵 | J-Quants の API キー（前後の空白を除く）から、PBKDF2-HMAC-SHA256（60万回・固定の塩）で 256 ビット |
| 暗号 | AES-256-GCM。IV は毎回ランダムな 12 バイト。見出し（基準日・作った時刻・方式）を追加認証データに入れるので、見出しを書き換えると開けない |
| 中身 | 集計の JSON を gzip で縮めたもの |
| 公開されるもの | 暗号文と見出し（基準日・作った時刻・方式）だけ。銘柄ごとの値は暗号文の中。対象の銘柄の一覧は、予測（`public/data/predictions.json`）として元から公開 |

- Release `data-volume-profile` は公開の Release です。デプロイは読み取り専用の鍵で動くため、下書きにすると読めません。中身は暗号文だけです
- 上げる前（Update Volume Profiles）と載せる前（Deploy to GitHub Pages）に、暗号文の形であること（平文でないこと）を
  `python3 research/volume_profile_crypto.py check FILE` で確かめます。形が違えば上げません・載せません
- 上げる前に、集計が画面の確かめ（`src/lib/volumeProfile.js` の `parseVolumeProfile`）を通ることも
  `node scripts/check_volume_profile.mjs` で見ます。通らなければ上げません（画面でキーを入れても表示できない集計を載せないため）。
  確かめに使う平文はランナーの中だけに置き、確かめたら消します
- 開けるのは J-Quants の API キーを持つ人だけです。キーが漏れていなければ、暗号文から中身は分かりません。
  キーが漏れたときは J-Quants でキーを作り直し、Secret `JQUANTS_API` を替えてください（古い集計は次の更新で置き換わります）
- 集計の中身は J-Quants のデータから作った値なので、平文では公開の場所（`public/`・`dist/`・公開の Release）に置きません。
  平文の書き出し（`--output`）は `public/`・`dist/` の下を拒みます。 [J-Quants 利用規約](https://jpx-jquants.com/ja/termsofservice?lang=ja)

## 計算

- **ティック**は、東証の立会内で成立した約定を価格ごとに合計します。売買代金は約定価格 × 株数です
- **分足**は、各分の株数と売買代金を、その分の VWAP（`Va ÷ Vo`）に割り当てます。分の中で価格が動くほどティックと分布が違います。
  両方揃う期間は、同じ価格帯の出来高シェアの差の絶対値を合計して2で割った値（ティックと分足の分布差）を出します。0% は一致、100% は重なる帯が無い状態です
- 価格帯は、期間内の日足の安値・高値から作る共通の帯です。幅は `1・2・5 × 10の整数乗` 円から選び、帯の上限は含めません。
  最多出来高の帯（POC）は株数で決め、同数なら低い帯を選びます。「出来高の70%範囲」は最多の帯から隣の帯を順に含め、累計が70%以上になった連続範囲です
- **終値とVWAPの差** = 基準日の終値 ÷ 期間の VWAP − 1。VWAP は期間の売買代金 ÷ 株数です（分割・併合は基準日の株数に換算）
- **時間帯別の出来高**（株数の割合）

  | 時間帯 | ティック | 分足 |
  |---|---|---|
  | 寄り | その日の最初の約定（板寄せ）。最初の約定と同じ時刻・同じ価格の約定をまとめて数える | 最初の1分（板寄せのあとのザラバの約定も入る） |
  | 前場 | 前場の残り（11:30 の前引けの板寄せを含む） | 12:00 より前の残りの分 |
  | 後場 | 後場（12:30 の後場寄りの板寄せを含む）から引けを除いたもの | 12:00 以降で引けより前の分 |
  | 引け | 大引けの板寄せ（2024-11-05 からは 15:25 以降の約定。15:25〜15:30 はクロージング・オークションで、15:30 に約定する。それより前は 15:00 以降） | 15:25 以降の分（それより前の日は 15:00 以降） |

  ティックの前場・後場は `SessionDistinction`（01 前場・02 後場）で分けます。それ以外の値の行は時刻で分けます（12:00 より前は前場）。
  引けの板寄せで初めて寄った日は、その約定を寄りとして数えます（引けと二重に数えません）
- **約定代金の大きさ別の割合**（ティックだけ）: ザラバ（板寄せを除く）の約定を、1件の約定（ティック1行）の約定代金で
  小口（100万円未満）・中口（100万円以上 1,000万円未満）・大口（1,000万円以上）に分け、ザラバの売買代金に占める割合と件数を出します。
  板寄せ（寄り・前引け・後場寄り・引け）は除きます。データ種別で分足を選んでいても、この欄はティックの値です
- 各日に、日足の出来高・売買代金・安値〜高値と照合します。株数の許容差は 0.01株未満、売買代金は 1円以内です。
  日足で取引なしと確認できる日は出来高0として扱います。基準日の終値が無い銘柄は集計しません
- 分割・併合を跨ぐ期間は、日足の `AdjFactor`（権利落ち日の行に入る。1:2 の分割なら 0.5）で、それより前の日の価格に係数を掛け、
  株数を係数で割って、基準日の単位に揃えます。約定代金は変わりません。係数が無い・不正な日があれば、その期間は欠損です。
  [日足仕様](https://jpx-jquants.com/ja/spec/eq-bars-daily) · [調整係数](https://jpx-jquants.com/ja/spec/eq-bars-daily/adj)

この分布が示すのは集計期間に成立した取引です。同じ株の売買も期間中の約定ごとに数えます。

## 欠損の理由

| 画面の表示 | 中身（`status`） |
|---|---|
| 選択期間の営業日が足りません | `calendar_history_missing` |
| 選択期間の日足が揃っていません（上場して間もない銘柄など） | `daily_missing` |
| 照合用の日足に欠損または不整合があります | `invalid_daily` |
| 基準日の終値がありません | `close_missing` |
| 選択期間の分割・併合の係数を確かめられません | `share_basis_unknown` |
| 選択期間の分足・ティックが置き場にまだ無いか、足りません | `intraday_missing`（当日分は夜の取得のあとに入る） |
| 分足・ティックに不正な値があります | `intraday_invalid` |
| 分足・ティックの合計が日足の出来高・売買代金と一致しません | `daily_totals_mismatch` |

Update Volume Profiles の実行の画面（公開）には、件数だけを出します: 置き場にある日・無い日、日足との照合の結果
（銘柄×日の件数。`ok`・`volume`・`turnover`・`range`・`missing` など）、期間・種別ごとの状態の件数。価格・数量・銘柄名は出しません。

## 手元で確かめる

合成データ（実データではない）で、集計から画面まで通せます。

```bash
python3 -m pip install -r research/requirements.txt
python3 tests/fixtures/volume_profile_inputs.py /tmp/vp /tmp/vp/predictions.json
JQUANTS_API=test-key python3 research/build_volume_profile.py --predictions /tmp/vp/predictions.json \
  --work-dir /tmp/vp --calendar /tmp/vp/calendar.csv --encrypt-to /tmp/vp/volume-profile.enc.json \
  --output /tmp/vp/plain.json
npm run build && node tests/volume-profile-browser.mjs   # 画面で開く・違うキー・欠損・スマホの幅
```

実データで確かめるには、置き場（下書きの Release）を読める鍵（このリポジトリへの書き込み権限）が要ります。
`GH_TOKEN` を渡して `--download-cmd "bash scripts/gh_release_download.sh"` を付けると、必要なファイルだけを
`--work-dir`（既定 `research/_data/volume_profile`。Git の対象外）に取り出します。平文の `--output` は公開の場所に置かないでください。

## 以前の作り（2026-11-18 まで）

非公開リポジトリ `growthstock-minute` に平文の集計を置き、GitHub の fine-grained PAT（Secret `MINUTE_DATA_TOKEN` と閲覧用トークン）で
読み書きする作りでした（`research/publish_volume_profile.py`）。2026-11-19、運用者の選択「暗号化し J-Quants のキーで開く」
「既存のタブを作り替える」で、いまの作りに替えました。対象も「最新の予測日の候補」から「直近5営業日の候補すべて」に広げました。
作っていた非公開リポジトリ・PAT・Secret `MINUTE_DATA_TOKEN`・Variable `MINUTE_DATA_REPO` は使いません（消してかまいません）。
