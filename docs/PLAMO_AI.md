# PLaMo 3.0 FinによるAI銘柄分析

GitHub PagesのReact画面から、本人認証を通したCloudflare Workerを経由してPLaMo公式APIに質問する個人用MVP。
既存モデルの学習・予測・データ取得には介入しない。GPU・独自ドメイン・有料クラウドは不要。

```text
GitHub Pages（既存Vite/React）
  └ AI銘柄分析：公開データを1銘柄分だけ抽出
       └ POST /chat（利用パスワードによる本人認証）
            Cloudflare Workers Free
              ├ SQLite Durable Object：認証試行・回数・同時実行の制限
              └ PLaMo公式API：plamo-3.0-fin-beta
```

## 公式仕様の確認と実接続の状態

2026-11-09に確認：

- [PFNのFin開発記事](https://www.preferred.jp/ja/blog/tech/plamo-3-fin-development)で、モデルID `plamo-3.0-fin-beta` と利用上限付きの無料β提供を確認。
- [APIリファレンス](https://docs.plamo.preferredai.jp/ja/api)で、`POST https://api.platform.preferredai.jp/v1/chat/completions`、Bearer認証、`messages`・`model`・`max_tokens`、回答本文 `choices[0].message.content` を確認。
- 無認証で同エンドポイントへ最小POSTを行うと、**HTTP 400 / `missing key in request header`** が返ることを確認。これは認証が必要という確認に限られ、Finモデルの利用権や正常回答を確認したものではない。
- APIリファレンスのモデル表にはPrimeのみが記載され、[制限事項](https://docs.plamo.preferredai.jp/ja/limit)にもFin固有の数値上限は見当たらない。Primeの上限をFinの上限として流用しない。
- **APIキーの発行・設定、Cloudflareへのログイン・実デプロイ、認証済みFin APIからの回答取得は未実施**。後述の手順で契約画面の無料β・数値上限を確認し、Secret設定後に実接続する。

公式サイトの更新とコンソールでの提供状況に差がある場合は利用を止める。
無料βの利用不可・終了・モデル不明は設定不足として扱い、有料モデルを試さない。
`PLAMO_FREE_BETA_CONFIRMED` の初期値は `false` で、上流呼び出しを禁止してある。

## 送信する実データと制限

`src/lib/plamo.js` がアプリですでに読み込んだ次の公開ファイルを優先して使う。追加の株価API呼び出しはない。

| 公開ファイル | 利用する情報 |
|---|---|
| `public/data/stocks.json` | コード・名前、取得日時、取得済み指標、指標の算出根拠と財務開示日、必要な場合の3・6か月前スナップショット |
| `public/data/predictions.json` | その銘柄の最新候補行、6分類器の生スコアと過去分位、LTR当日順位、LTR5の直近5営業日プール順位、各行の目標定義 |
| 同 `contrib.top` | 基準LightGBM分類器のTreeSHAP寄与を最大8項目。対数オッズ空間の寄与であり、特徴量の生値ではない |
| `public/data/feature_names.json` | 公開された特徴量名の日本語表示 |

財務質問では財務指標を中心にし、順位・モデルの質問で予測とSHAPを追加する。過去比較の質問でのみ最大2つの過去観測を加える。
指標は新しい実測値を優先するが、**値ごとの基準日・出典**を保持する。古い予測と新しい財務を同日の観測として渡さない。
取得できない値は省略し、ゼロで埋めない。過去予測の目標定義が無ければ最新モデルの定義で補わない。
質問に合う公開データが無い場合は、未取得であることを説明する。

- `src/lib/store.js` のlocalStorage・手入力銘柄・売買履歴は読み取って送信しない。アプリ全体の選択銘柄が公開データにもあるときだけ初期値として使う。
- ランキングスコアを確率に変換しない。分類器の生スコア・分位も実際の将来到達率と同一視しない。
- 分類器のSHAPでランキングモデルの順位理由を説明しない。ランキングモデルの個別SHAP・373列の全特徴量値・秘密の予測ロジックは送らない。
- 過去スナップショットを「類似局面の検索結果」と称さない。類似事例の検索はMVP対象外。
- LTR5の評価日は元の候補日と違う場合があるため、独立した基準日を付ける。

2026-11-09の公開58銘柄×質問例5種類=290コンテキストを検査し、最大4,623バイトだった。今後のデータ量にかかわらず後述の上限で止める。

### 無料プランでのデータの取り扱い

[公式のデータ取扱説明](https://docs.plamo.preferredai.jp/ja/pricing)によれば、Freeプランは入力・出力を保存して学習に利用し、ZDRの対象外。
画面にも明記し、送信内容を開いて確認できる。**質問そのものも外部送信される**ので、個人情報・非公開の売買履歴・秘密情報を入力しない。

APIキーはブラウザへ渡さない。本人認証パスワードは本人が画面に入力し、Worker宛てのAuthorizationヘッダで送る。
ビルド・URL・localStorage・sessionStorage・ログに保存しない。AIタブから離れると入力は破棄される。

## 無料運用の条件と制限

| 項目 | 初期値・上限 |
|---|---|
| 質問 | 1〜1,200文字 |
| 銘柄コンテキスト | 最大16 KiB |
| リクエスト本文 | 最大24 KiB。Content-Lengthだけでなくストリームの実サイズも検査 |
| 回答トークン | `max_tokens=1536`、設定可能な上限2,048 |
| PLaMo待ち時間 | 25秒、設定可能な上限30秒。フロントは40秒で中断 |
| API呼び出し | 個人用全体で3回/分・20回/日・300回/月（UTC） |
| 同時実行 | 個人用全体で1件。複数タブ・Worker拠点をまたいで制御 |
| 認証を含むアクセス | IP単位10回/分・100回/日、全体60回/分・500回/日 |
| 認証試行の保持 | 生IPは保存せずハッシュとカウンタのみ。1日最大256個のIPハッシュ |

日・月の回数は**上流呼び出し開始時に予約し、API失敗でも返却しない**。上流が処理した可能性があるためである。
既定のリースは最大30秒で解除する（25秒のタイムアウト＋5秒）。異常終了しても永続的なロックにしない。
二重送信はフロントでも抑止する。429・5xx・タイムアウト時の自動再試行は行わない。
レート制限用ストレージが使えない場合も、上流へ通さず503で終了する。

`CHAT_PER_MINUTE` / `DAY` / `MONTH` はそれぞれ5/50/1000より大きく設定できない。
これらはアプリ独自の上限であり、**PLaMoの無料β枠の保証ではない**。実際の無料枠が小さければ、その範囲以下に設定する。

Cloudflareは**Workers Freeプラン**を利用する。SQLite-backed Durable ObjectsはFreeで利用可能で、無料枠を超えると操作が失敗する。
Paidへの変更や有料モデル・検索・Deep Researchへの切り替えをコードから行わない。
PLaMo側も無料プラン・無料βへのアクセスを契約画面で確認する。課金設定を変えた場合や提供条件が変わった場合の0円は保証できないため、有料設定に変更しない。

公式確認先：[Workers料金](https://developers.cloudflare.com/workers/platform/pricing/)・[Durable Objects料金](https://developers.cloudflare.com/durable-objects/platform/pricing/)。

## Cloudflareの手動設定

1. CloudflareアカウントでWorkers **Free**を選ぶ。独自ドメインの購入は不要。
2. Node.js 22以降で、この実装が入ったチェックアウトの `worker/` に移動する。

   ```bash
   cd worker
   npm ci
   npx wrangler login
   npm run check
   npm run deploy
   ```

   初回は無料β未確認・キー未設定のためPLaMoを呼べない状態で作成される。SQLite用のbinding/migrationは `wrangler.jsonc` に含まれる。

3. **対話入力**でSecretsを登録する。値をコマンド引数、リポジトリ、Actionsのログに書かない。

   ```bash
   npx wrangler secret put PLAMO_API_KEY
   npx wrangler secret put AI_PASSWORD
   ```

   APIキーは本人がPLaMoコンソールで発行したものを使用する。本実装では発行しない。
   `AI_PASSWORD` は32〜256文字の印字可能ASCII。例えば本人のローカル端末でPythonの `secrets.token_urlsafe(32)` を使って生成し、パスワードマネージャーに保存する。
   例示用の固定パスワードを運用に使用しない。

4. PLaMoコンソールで次を確認する。
   - 無料プランで `plamo-3.0-fin-beta` を利用できること。
   - Fin固有の無料上限。必要ならアプリの回数・トークン上限を下げる。
   - Freeプランのデータ保存・学習利用条件。

5. 確認できた場合にだけ `worker/wrangler.jsonc` の `PLAMO_FREE_BETA_CONFIRMED` を文字列 `"true"` に変更し、`npm run deploy`。
   未確認・利用不可ならfalseのままにする。Primeモデル等を試して代用しない。
6. 表示された `https://growthstock-plamo.<自分のサブドメイン>.workers.dev` を控える。

### 設定値

| 置き場所 | 名前 | 内容 |
|---|---|---|
| Worker Secret | `PLAMO_API_KEY` | PLaMo公式のAPIキー |
| Worker Secret | `AI_PASSWORD` | 本人認証用のランダムなパスワード |
| Worker vars | `ALLOWED_ORIGINS` | `https://shunshun0904.github.io`。パスを含めず完全一致、複数はカンマ区切り。`*` は不可 |
| Worker vars | `PLAMO_ENDPOINT` | 公式Chat Completions URL。HTTPSの公式APIホストとversion付きChat Completionsパスだけ許可 |
| Worker vars | `PLAMO_MODEL` | 既定 `plamo-3.0-fin-beta`。Fin beta形式以外は拒否。応答のmodelも指定値と一致することを検査 |
| Worker vars | `PLAMO_FREE_BETA_CONFIRMED` | 無料β確認後のみtrue |
| Worker vars | `MAX_OUTPUT_TOKENS` / `UPSTREAM_TIMEOUT_MS` | 回答長・待ち時間 |
| Worker vars | `CHAT_PER_MINUTE` / `CHAT_PER_DAY` / `CHAT_PER_MONTH` | 個人用全体の回数上限 |
| Worker binding | `AI_LIMITER` | 自動作成するSQLite版Durable Object。手動のDB ID作成は不要 |
| GitHub Actions **Variable** | `VITE_PLAMO_WORKER_URL` | 公開してよいWorkerのルートURLだけ |

APIキーを `VITE_` の変数に置かない。Worker varsにもAPIキーやパスワードを書かない。
誤って `VITE_PLAMO_API_KEY` 等の機密用途の変数名を渡すと、Viteの設定検査でビルドを停止する。エラーに値は出力しない。
公開Worker URLに認証情報・クエリ・パスを含めた場合も、バンドルへ入る前にビルドを停止する。
Providerへのリダイレクトは追わず、別ホストへのキー転送を防ぐ。
Workerのコードに質問・回答・認証ヘッダをログ出力する処理はない。Wranglerのobservabilityも既定で無効。

### Cloudflare Accessについて

MVPは本人認証パスワードを毎回照合する方式。CORSだけでは認証しない。
Cookieを使わず `credentials: omit` とAuthorizationヘッダで通信するため、GitHub Pagesとworkers.devの異なるoriginで使える。
このクロスオリジン動作は2つのローカルHTTP originを使った実ブラウザテストで検証する。
Cloudflare Accessは今回有効化しておらず、Access経由のOPTIONS・Cookie/CORS・GitHub Pagesとの組み合わせは未検証。設定を追加する場合は別途確認が必要。

## GitHub Pagesの手動設定

1. Repository → Settings → Secrets and variables → Actions → **Variables** に `VITE_PLAMO_WORKER_URL` を作る。
2. 値は上記のWorker URLだけ。`/chat`・クエリ・認証情報を含めない。
3. PRを公開ブランチへマージ後、既存の `Deploy to GitHub Pages` を実行する。PagesのSourceはこれまでどおりGitHub Actions。
4. 「AI銘柄分析」で公開銘柄を選び、本人認証パスワードを入力して質問する。

変更した公開ワークフローはビルド時にこのVariableを渡す3行だけ。既存のデータ取得・学習・予測ワークフローは変更しない。
Variable未設定ならAI画面に設定案内を出し、既存タブを通常どおり利用できる。
アセットと公開データは既存のVite `base: './'` を維持し、`/growthstock/` 以下からの配信をテストする。

## APIとエラー

`POST /chat` は `Authorization: Bearer <本人用パスワード>` と `Origin` が必要。
本文は `{question, context}` のみ。`context` の実スキーマは `worker/src/contract.js` を参照。
未知の項目・クライアント指定モデル・URL・ツール等は受け取らない。
成功は `{answer, model, truncated, usage}`。失敗は `{error:{code,message,retryAfter?}}`。
`GET /usage` も同じ認証が必要で、利用回数と設定の有無だけを返す。PLaMoへの通信は発生しない。

| エラー | 意味・操作 |
|---|---|
| 401 `UNAUTHORIZED` | 本人認証パスワードが不正 |
| 429 `AUTH_RATE_LIMIT` / `RATE_LIMIT` / `DAILY_LIMIT` / `MONTHLY_LIMIT` | Worker独自上限。Retry-Afterを返す |
| 409 `BUSY` | 別の分析が実行中 |
| 429 `UPSTREAM_LIMIT` | PLaMo上限。自動再送・有料移行なし |
| 503 `NOT_CONFIGURED` | Secrets、無料β確認、設定値が未完了 |
| 503 `MODEL_UNAVAILABLE` | β利用不可・不明モデル・プラン制限等 |
| 502 `MODEL_MISMATCH` | 応答が指定したFin betaモデルと異なる |
| 502 `UPSTREAM_UNAVAILABLE` | 上流5xx・通信失敗・リダイレクト・不正応答 |
| 504 `TIMEOUT` | 上流の回答タイムアウト |
| 502 `EMPTY_ANSWER` | 本文が空。内部推論だけの応答は本文として返さない |
| 400/413/415 | 無効な質問・データ不足・サイズ超過・形式不正 |

上流のエラー本文はブラウザへ転送しない。

## テストと実接続の確認

```bash
npm ci
npm test
npm ci --prefix worker
npm test --prefix worker
npm run check --prefix worker
npm run build
# Playwright/Chromiumをインストールした環境で実行
node tests/smoke.mjs
node tests/ai-smoke.mjs
```

`tests/plamo.test.js` は公開データ抽出、欠損・基準日・SHAPの区別、URL未設定、送信内容、認証・429・5xx・タイムアウトを検査。
`worker/test/index.test.js` はMiniflare/workerdとSQLiteの実ストレージを使い、外部PLaMo通信だけをモックにする。
`tests/ai-smoke.mjs` はWorker URLを指定した隔離ビルドを作り、別originのモックAPIへ実ブラウザで通信する。
PC/スマートフォン、銘柄の実測値、Markdown、コピー、二重送信、エラー、パスワード非保存、サブパス配信、ビルドへの秘密混入防止を検査する。
CIにもWorkerテストとAIブラウザテストを追加した。自動テストでは実PLaMo APIに接続しない。

**Secret設定後の手動確認**：公開銘柄について1回質問し、回答本文と `plamo-3.0-fin-beta` 表示を確認する。
正しい認証で受信できた事実・モデルID・無料枠の数値を記録する。回答を財務の元データと突き合わせる。
キー未設定・β利用不可の場合は設定エラーとして停止することを確認し、有料プランへ変更しない。
本番Pages上の実CORS・実APIの応答時間・Finの回答品質・Fin固有の数値上限は、この認証後の確認まで未検証。

停止する場合は `PLAMO_FREE_BETA_CONFIRMED=false` でWorkerを再デプロイするか、PagesのWorker URL Variableを空にして再ビルドする。
どちらも既存の株式データ・予測モデルを変更しない。
