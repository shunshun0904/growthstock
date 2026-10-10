# Jev（TypeSafe AI の判断モデル）の見立てを並べる — 2026-10-10

運用者の依頼（2026-10-10）: 「本番で稼働しているモデルの予測スコア（ダッシュボード）に加え、
売買判断には Jev による予測値を使いたい」。同日の運用者の決定（AskUserQuestion の4問）:

| 問い | 決定 |
|---|---|
| Jev に何を予測させるか | **「翌営業日の寄りで買い、20営業日以内に +10% に届く」の真偽の確率（noul）**。実験70 の物差し（+10% の指値）と同じ |
| Jev に渡す情報 | **候補の実測値 ＋ 4モデルの百分位と寄与**（モデルの読みを踏まえた審査。独立した意見にはしない） |
| 「今日の戦略」での使い方 | **表示＋記録。選定の規則は変えず、買い候補の Jev が 50% 未満なら注意を出す**。実績が溜まってから規則に入れるか決める（`docs/MODEL_ADOPTION_RULES.md` §7 と同じ流儀） |
| 過去分の検証 | **本番で前向きに記録しつつ、直近1年の OOF 候補にも一度当てて測る**（実験73） |

## 1. Jev とは

TypeSafe AI（2024年創業、サンフランシスコ）が **2026-09-15** に公開した「System One」と呼ぶ判断特化のモデル。
LLM と違って文章を生成せず、与えた状態（テキストか JSON）と名前付きの問いに対して、**型付きの値と確率だけ**を返す。
問いは3種類:

| 種類 | 返るもの | 用途 |
|---|---|---|
| `noul` | 真偽の確率（0〜1） | 「この文は正しいか」。ここで使う |
| `choice` | 選択肢ごとの確率と選ばれた1つ、確信度 | 分類・振り分け |
| `score` | 段階の期待値と段階ごとの確率、確信度 | 採点 |

公式: [typesafe.ai](https://typesafe.ai/) / [docs.typesafe.ai](https://docs.typesafe.ai/)（Quick start・
[HTTP API](https://docs.typesafe.ai/api)・[primitives](https://docs.typesafe.ai/primitives)）/
[Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)（`typesafe-sdk`、Python 3.10+）/
[JavaScript SDK](https://github.com/typesafe-ai/typesafe-sdk-js)。OpenRouter・Cloudflare・Vercel の
ゲートウェイ経由でも使える。日本語の解説は Zenn・Qiita・GMO の記事がある（2026-09-18 以降）。

**金融での実績は無い。** 株価の向きを当てる目的で作られたモデルではなく、公開の検証も
「社内の判断を速く安く」の類が中心。コミュニティの検証（[awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev)
に一覧）では、確率は**較正されていない**（「0.52」が 52% の正解率を意味しない）と繰り返し指摘されている。
株に当てた例は [sosopop/jev_stock](https://github.com/sosopop/jev_stock)（香港株の短期の方向。4銘柄×30件で
正解率 45%、作者自身が「検証済みの定量モデルではない」と注記）など、実験の段階のものだけ。
ここでも**「Jev がそう言った」以上の意味を、測るまで与えない**。

## 2. 何をどう問うか（`research/jev_predict.py`）

### 問い（`QUESTIONS`、版 `rise10_v1`）

```
type: noul
instructions: この日本株を翌営業日の寄り付き（始値）で買ったとき、買った日を1日目として
              20営業日以内に、株価の高値が買値の +10% 以上に達する。
criteria.true:  20営業日以内のいずれかの日の高値が、買値の 1.10倍以上になる
criteria.false: 20営業日のあいだ、高値が一度も買値の 1.10倍に届かない
```

「買った日を1日目」「高値で判定」は実験32・70 の `hit10`（`research/exp/e32_takeprofit.py`）と同じ定義。
運用の出口（+20% の指値・20営業日）ではなく +10% にしたのは、+20% の到達が約1割しか無く、確率を問うても
ほぼ「届かない」になるため。実験70 の物差し（+10% の指値の収益）と揃えてある。
文言を変えたら `QUESTION_VERSION` を上げる（控えと予測ファイルに版が残り、違う版の答えを混ぜない）。

### 渡すもの（`build_state`。日本語の鍵の JSON）

| 塊 | 中身 |
|---|---|
| 課題・日付・銘柄 | 何を判定するか。日付。コード・名前・業種・市場・規模区分 |
| 株価 | 終値・78週高値・高値に対する位置・抜けた幅・ベースの長さ・20日リターン・日次ボラ・出来高トレンド |
| 需給・規模 | 時価総額・売買代金20日平均・信用倍率 |
| バリュエーション | PER・PBR・配当利回り |
| 決算 | EPS 成長率・売上成長率・ROE・営業利益率・進捗率と基準・四半期 |
| その日 | 発火数・基準モデルでの順位・地合い（候補全体の地合い寄与の中央値。画面と同じ ±0.05） |
| 機械学習モデルの見立て | LightGBM / XGBoost / CatBoost / ロジスティック回帰 の百分位、必要上昇率。母集団の正例率が約18%という説明 |
| 基準モデルの寄与 | TreeSHAP の区分ごとの寄与と、効いた特徴量の上位8つ（列名・意味・寄与） |

画面に出している値（`predictions.json` の候補）をそのまま使う。**欠測は書かない**（0 で埋めると「実測でゼロ」と混ざる）。
1件あたり 1,500〜2,000文字（公開中の予測ファイルで `python3 research/jev_predict.py --dry-run` が出す）。
較正確率と帯の正例率は入れていない（百分位の単調な写しなので重複。実験73 でも前の窓だけから作り直せないため）。

### 呼び方

HTTP の形は公式 Python SDK（`typesafe-sdk` 0.7.4 の `_schemas/models.py`。`api.typesafe.ai/openapi.json` から
生成されたもの）から写した:

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
{"state": {...}, "model": "jev-latest", "questions": {"rise10": {"type": "noul", ...}}}
→ {"model": "jev-1.13.0", "answers": {"rise10": {"type": "noul", "noul": 0.37}},
   "usage": {"input_tokens": 900, "output_tokens": 1}}
```

SDK は入れず標準ライブラリ（urllib）で呼ぶ（依存を増やさず、通信を差し替えて単体テストするため）。
429 / 5xx / 接続失敗は 2・4・8秒で3回まで再試行（`retry-after` があればそれを優先）。400 系はすぐ止める。
鍵は環境変数 `TYPESAFE_API_KEY` からだけ読み、ログ・例外・repr に出さない。
モデル名は既定 `jev-latest`（別名）。応答に入る実際の版（例 `jev-1.13.0`）を答えごとに控える。
固定したいときは環境変数 `JEV_MODEL`。

### 原則

1. **鍵が無ければ何もしない。** 日次予測はそのまま動き、画面の Jev の列は出ない
2. **答えは (予測日, 銘柄コード, 問いの版) ごとに最初の答えで凍結する。** 控えは `research/_data/jev_answers.parquet`
   （Release `data-raw`。`predict.yml` が毎晩上げ、翌晩の Download raw data で戻る）。同じ日を予測し直しても
   問い直さず、課金もしない。記録（`prediction_history.json` / 台帳）と同じ考え方
3. **1銘柄の失敗で他を止めない。** 失敗は空のまま（0 で埋めない）。3回続けて失敗したら、その実行では残りを問わない
   （障害の日に予測ジョブを何十分も待たせない）
4. 1回の実行で問う数に上限（`JEV_MAX_PER_RUN`、既定 200）。新しい日・上位から順に問う。初回は画面の5営業日ぶん
   （40〜60件）、以後は1日十数件
5. **選定の規則には入れない**（`src/lib/strategy.js` の `strategySignal` は Jev を読まない。`tests/strategy.test.js` が固定）

## 3. 画面と台帳

| 場所 | 出すもの |
|---|---|
| 候補一覧の行 | 「Jev +10%」の列（確率 %。50 未満は黄、以上は緑、無ければ「—」）。鍵が未設定の予測ファイルでは列ごと出ない |
| 行を開いたとき | 「Jev の見立て」: 確率・モデルの版・問うた日時・何を渡したか・規則に入れていないこと |
| 今日の戦略の買い候補 | 「Jev +10%」の値。50% 未満なら「Jev 50%未満」の印と、買い候補の下に注意の帯 |
| 惜しい候補 | 行に「Jev xx%」 |
| 「Jev（判断モデル）」の欄 | 状態（問うているか）・問いの全文・渡すもの・モデルの版・この実行で問うた数と入力トークン |
| 過去に出した銘柄のその後 | 「Jev」の列（記録に値が入り始めてから） |
| 台帳（スプレッドシート） | 列 `Jev+10%確率%`（`docs/OPERATIONS.md`「スプレッドシートの列」）。既存のシートには右端に足される |
| `predictions.json` | `candidates[].jev = {prob, model, question, askedAt, cached}`、`jev = {enabled, model, resolvedModel, question, questionVersion, target, asked, cached, failed, skipped, inputTokens, outputTokens, note}` |
| `prediction_history.json` | `entries[].jevProb` |

注意の線 50（`JEV.line`）は結果を見て引いた線ではなく「確率が五分を切る」という素朴な位置。
実験73 と前向きの記録で引き直す。

## 4. 設定

1. TypeSafe AI の[コンソール](https://console.typesafe.ai/)で API キーを発行する
2. GitHub の `Settings → Secrets and variables → Actions` に **`TYPESAFE_API_KEY`** として登録する
3. 次の夜の `Predict Breakouts` から Jev の列が入る（過去分は遡らない。実験73 で測る）。
   先に確かめるなら、Actions の `Run Experiment` で `exp=e73_jev_oof.py args=--dry-run`（通信しない）か、
   手元で `TYPESAFE_API_KEY=... python3 research/jev_predict.py --check`（GET /v1/models）・`--sample`（1件だけ問う。課金あり）

費用は入力トークン課金（出力は無料と OpenAPI の説明にある）。1件 1,500〜2,000文字なので、毎晩十数件なら
月に数十万トークン。単価は公式の料金表を見る（このセッションからは読めなかった）。

## 5. 実験73（`research/exp/e73_jev_oof.py`）: 直近1年の OOF 候補に当てる

先に決めたこと（結果を見てから分け方を足さない。スクリプトの冒頭に同じ文）:

- **対象**: 本番の OOF（lgbm / xgb / cat / logit）の行のうち、最新日から 365暦日以内で、全モデルの百分位
  （その行より前の窓の分布。実験70 と同じ基準）と +10% の結果が付く行（約3,000件の見込み）
- **state**: 本番と同じ作り。寄与は**その行の窓の** LightGBM（本番のパラメータで窓ごとに学習し直す）の TreeSHAP。
  本番の学習済みモデルの SHAP を使うと、その行を学習に使った in-sample の寄与（ラベルを少し知っている）を
  Jev に渡してしまうため
- **腕**: `prod`（本番と同じ。銘柄名・コード・業種・市場・日付を含む）と `anon`（それらを伏せる）。Jev は 2026-09-15 公開の
  モデルで、過去の日付の銘柄については「その後どうなったか」を学習データから知っている疑いがある。anon との差が
  その大きさの目安。**差が無ければ先読みの心配は小さい。prod だけ良ければ prod の数字は当てにならない**
- **見るもの**: (1) 較正（10pt 刻みの帯ごとの実際の到達率。Brier を「母集団の到達率を常に答える」基準と比べる）
  (2) 分離力（hit10 の ROC-AUC。各モデルの百分位・3モデルの最小と比べる。日付内の AUC も）
  (3) 運用との重なり（線の上・際どい候補・母集団を Jev 50 以上/未満で分ける。差と SE・z。SE は銘柄ごとにまとめる）
  (4) 相関（Spearman。prod と anon も）(5) 費用
- **記録**: 集計だけをログに出す（銘柄ごとの値は出さない。2026-10-10 の決め事）。答えは `research/_data/oof/e73_jev_answers.parquet`
  に控え、Actions のキャッシュに乗る（公開の成果物には上げない）。途中で止まっても続きから

回し方: Actions の `Run Experiment` で ref `claude/jquants-browser-app-bwk19a`（マージ後）、`exp=e73_jev_oof.py`。
`args=--dry-run` で件数・文字数・state の例（anon）だけ。`args="--arms prod --limit 200"` で費用の確かめ。
既定は両腕・直近1年。鍵は `run-experiment.yml` が Secrets から渡す。

**採否の決め方（先に書く）**: Jev を選定の規則に入れるかは、実験73 の「線の上を Jev 50 で分けた差」が
`docs/MODEL_ADOPTION_RULES.md` §7 の精神（対照と区別でき、向きが一貫）で見えること、**かつ** anon でも同じ向きであること、
**かつ** 前向きの記録（2026-11 の集計以降）でも向きが同じこと、の3つがそろってから。1つの実験で決めない
（運用者の方針「1つの OOF で戦略を決めると過適合する」）。

### 結果

（まだ回していない。回したらここに集計を書く。run の ID・対象の件数・期間・腕ごとの較正の表・AUC・線の上の差・
anon との相関・費用）

## 6. 変更したもの

| ファイル | 変更 |
|---|---|
| `research/jev_predict.py` | 新規。state・問い・HTTP・控え・`annotate` |
| `research/predict_daily.py` | byModel の後で `annotate`。`payload.jev`、`candidates[].jev`、`entries[].jevProb` |
| `.github/workflows/predict.yml` | `TYPESAFE_API_KEY` を採点の段に渡す。控えを Release に上げる |
| `.github/workflows/run-experiment.yml` | `TYPESAFE_API_KEY` を渡す（実験73 だけが使う） |
| `research/export_sheets.py` | 列 `Jev+10%確率%` |
| `src/lib/strategy.js` | `JEV` / `jevProb` / `jevNote` / `jevWarnings`。`strategySignal` は変えない |
| `src/views/PredictionView.jsx`・`src/styles.css` | 列・買い候補の印と注意の帯・詳細・素性の欄・追跡表の列 |
| `research/exp/e73_jev_oof.py` | 実験73 |
| `tests/test_jev_predict.py`・`tests/test_e73_jev_oof.py`・`tests/strategy.test.js`・`tests/test_sheets.py`・`tests/test_prediction_record.py`・`tests/smoke.mjs`・`tests/fixtures/synthetic-predictions.json` | テスト |
