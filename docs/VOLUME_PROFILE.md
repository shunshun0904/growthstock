# 価格帯別出来高

## 止めたこと（2026-11-21）

運用者の依頼（2026-11-21）「それと、分足のデータは使わない方針になったので価格帯別出来高の分布タグも消してください。データ取得と
保存は契約が切れるまで引き続きお願いします（分足のデータの保存はそのままにしておいてください）。」→ 選択（AskUserQuestion）:
タブ =「隠して集計も止める」（勧めた）/「タブだけ隠す」/「コードごと消す」、公開中の集計 =「公開をやめる」（勧めた）/「そのまま置く」。

- タブを隠した（`src/App.jsx` の `hidden`）。画面のコード（`src/views/VolumeProfileView.jsx`・`src/lib/volumeProfile.js`）は残す。
  隠している間は集計（`data/volume-profile.json`）を読まない（`tests/volume-profile-browser.mjs`・`tests/smoke.mjs` で確かめる）
- `Update Volume Profiles`（`.github/workflows/update-volume-profile.yml`）の自動の起動（予測・分足・ティックの workflow_run）を外した。
  手では起動できる。2026-11-28 から Release `data-volume-profile` は下書き（非公開）で、起動すると集計は下書きに上がる（公開に
  なっていたら上げずに止まる。`GH_ENSURE_DRAFT`）。集計を作り直すかは、起動する前に運用者に確かめる
- `Deploy to GitHub Pages` から、集計を降ろしてサイトの `data/` に置く段と、`Update Volume Profiles` の完了での起動を外した
- 公開の Release `data-volume-profile` の `volume-profile.json` を消した（Release そのものは空のまま残す）
  - 訂正（2026-11-27）: `volume-profile.json` は消えておらず、公開の Release に残っていた（Release の一覧で確かめた）。運用者の
    選択（2026-11-27）「3つとも一緒に非公開」で、2026-11-28 21:00 JST に Release ごと下書き（非公開）にして残す
    （`docs/OPERATIONS.md`「data-raw を非公開にする」）。2026-11-28 21:01 JST に下書きにした（Make Releases Private の
    run 46142598015。PASS 45 / FAIL 0。アセット 1件は変わらない）
- **分足・ティックの取得と保存は続ける**（`Update Minute Bars`・`Update Ticks`。下書きの Release `data-minute`・`data-tick-YYYY`。
  [MINUTE_BARS.md](MINUTE_BARS.md)・[TICKS.md](TICKS.md)）。集計の台本（`research/build_volume_profile.py`）と単体テストも残す
- 戻すとき: `hidden` を外し、git の履歴（2026-11-21 より前）から `deploy-pages.yml` の集計を置く段と `update-volume-profile.yml` の
  workflow_run を戻し、`Update Volume Profiles` を手で起動して作り直す（公開に戻るので、運用者の判断で）

以下は 2026-11-20 までの説明です。

「価格帯別出来高」タブには、**直近5営業日の予測対象（候補）すべて**の、予測の基準日までの 1・5・20 営業日の
価格帯別出来高を表示します。あわせて次の指標を出します。

- **終値とVWAPの差**（終値 ÷ VWAP − 1）
- **時間帯別の出来高**（寄り・前場・後場・引け。株数の割合）
- **約定代金の大きさ別の割合**（小口・中口・大口。ティックのザラバの約定から）

タブを開くと、キーの入力なしで集計が出ます。集計は**誰でも見られる**形で公開サイトに置いています（下の「公開について」）。

## 見かた

1. 「価格帯別出来高」タブを開きます。キーの入力は要りません。 [ダッシュボード](https://shunshun0904.github.io/growthstock/)
2. 銘柄は「銘柄」のプルダウンで選びます（コードを打つ必要はありません）。一覧の表には、直近5つの予測日の候補が銘柄ごとに
   1行で並びます（候補になった日と、その日の順位）。表の行を押しても選べます。
3. 集計期間（1・5・20営業日）、データ種別（ティック・分足）、横軸（株数・売買代金）を切り替えられます。
   一覧の「終値とVWAPの差」「大口」は、選んだ期間の値です。
4. 揃わない期間・種別は、値を作らずに理由を出します（下の「欠損の理由」）。「再読み込み」で集計を読み直します。

## 更新の流れ

```
Update Data Store（毎晩の取り込み）
  ├─ Predict Breakouts   予測（public/data/predictions.json）
  ├─ Update Minute Bars  分足を下書きの Release data-minute に足す
  └─ Update Ticks        ティックを下書きの Release data-tick-YYYY に足す
        │ それぞれが終わるたびに
        ▼
Update Volume Profiles   集計し、画面の確かめを通してから Release data-volume-profile の volume-profile.json に上げる
        ▼
Deploy to GitHub Pages   画面の確かめを通してから volume-profile.json を画面の data/ に置く
```

- ワークフローは `.github/workflows/update-volume-profile.yml`、集計は `research/build_volume_profile.py`
- 入力は、分足・ティックの置き場（下書きの Release。[MINUTE_BARS.md](MINUTE_BARS.md)・[TICKS.md](TICKS.md)）、
  照合用の日足（Release `data-raw` の `bars_YYYY.parquet`）、J-Quants の取引カレンダー
- 予測・分足・ティックの3つは並んで走り、終わる順は日によって違います。最後に終わったものの後の集計が、全部そろったものです。
  そろう前の集計では、まだ置き場に無い日の種別が「欠損」と出ます
- 使う Secret は `JQUANTS_API` だけで、取引カレンダーを取るためにワークフローの中だけで使います（画面には出しません）。
  置き場の読み書きは Actions の鍵（`github.token`）です
- 上げる前（Update Volume Profiles）と載せる前（Deploy to GitHub Pages）に、集計が画面の確かめ
  （`src/lib/volumeProfile.js` の `parseVolumeProfile`）を通ることを `node scripts/check_volume_profile.mjs` で見ます。
  通らなければ上げません・載せません（画面が「形式を確認してください」と出して何も表示できない集計を載せないため）
- 集計は Git に積みません（毎日変わるため）。`--output` は `public/`・`dist/` の下を拒みます
- 手で作り直すときは Actions の **Update Volume Profiles** を Run workflow で起動します。 [実行履歴](https://github.com/shunshun0904/growthstock/actions/workflows/update-volume-profile.yml)

## 公開について

- 集計（`volume-profile.json`）は、公開サイト（GitHub Pages）と公開の Release `data-volume-profile` に置くので、**誰でも見られます**
- 2026-11-19 の運用者の選択「暗号化をやめて公開する」によります。それまで（#11）は J-Quants の API キーから作った鍵で暗号化し、
  画面でキーを入れて開いていました。リポジトリの Secret は GitHub Actions の中でしか読めず、ブラウザからは使えないため、
  キーの入力をなくすには集計を公開するしかありません（Secret を画面に埋め込むと API キー自体が公開されます）
- 集計の元は J-Quants のアドオン（分足・ティック）です。J-Quants の公式ヘルプでは、個人の利用は私的利用に限られ、データの第三者への
  配信は営利・非営利を問わず禁止とされています（[利用について](https://jpx-jquants.com/ja/help/usage)。2026-11-18 に確認）。
  これを伝えたうえでの選択です。分足・ティックそのもの（置き場）は、これまでどおり下書きの Release に置き、公開しません
- 暗号化していたころのアセット `volume-profile.enc.json` は、Update Volume Profiles が上げるときに消します。
  画面は、そのころ端末に覚えた復号の鍵を開いたときに消します

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
python3 research/build_volume_profile.py --predictions /tmp/vp/predictions.json \
  --work-dir /tmp/vp --calendar /tmp/vp/calendar.csv --output /tmp/vp/volume-profile.json
node scripts/check_volume_profile.mjs /tmp/vp/volume-profile.json   # 画面が読めるか
npm run build && node tests/volume-profile-browser.mjs             # 画面で開く・プルダウン・欠損・スマホの幅
```

実データで確かめるには、置き場（下書きの Release）を読める鍵（このリポジトリへの書き込み権限）が要ります。
`GH_TOKEN` を渡して `--download-cmd "bash scripts/gh_release_download.sh"` を付けると、必要なファイルだけを
`--work-dir`（既定 `research/_data/volume_profile`。Git の対象外）に取り出します。`--output` は `public/`・`dist/` の外に書きます。

## これまでの作り

| 時期 | 作り |
|---|---|
| 2026-11-18 まで（#10） | 非公開リポジトリ `growthstock-minute` に集計を置き、GitHub の fine-grained PAT（Secret `MINUTE_DATA_TOKEN` と閲覧用トークン）で読み書きした（`research/publish_volume_profile.py`）。対象は最新の予測日の候補だけ |
| 2026-11-19（#11） | 運用者の選択「暗号化し J-Quants のキーで開く」「既存のタブを作り替える」。蓄積した分足・ティックから直近5営業日の候補すべてを集計し、J-Quants の API キーから作った鍵で暗号化して載せた。画面でキーを入れて開いた |
| 2026-11-19（#12）から | 運用者の依頼「APIキーの入力自体なくしてください」と選択「暗号化をやめて公開する」。キーの入力なしで開く。銘柄はプルダウンで選ぶ |

作っていた非公開リポジトリ・PAT・Secret `MINUTE_DATA_TOKEN`・Variable `MINUTE_DATA_REPO` は使いません（消してかまいません）。
