# ニュース感情の特徴量: 取得経路の調査（2026-10-16。調査だけ。契約・取り込み・本番は変えていない）

運用者の案（2026-10-16）「直近のニュースの感情分析とかどうでしょうか？ それ（感情スコア）を特徴量にいれるのです。
まずは取得経路や API の調査からですかね」。学習データ（2018-04〜、東証 約3,700銘柄・約22,000行）に足すには、
**銘柄別・日付つき・過去分のある**ニュースが要る。その条件で経路を調べた。運用者の選択は「[No preference]」
（2026-10-16）→ 進めていない。

## 結論

- 無料・個人向けで東証全銘柄の過去ニュースを銘柄別に取れる経路は無かった
- いま現実的なのは **J-Quants API の適時開示（TDnet）アドオン**（2026-05-18 開始。月額 11,000 円税込、Light 以上に追加、
  過去 5 年、API と CSV 一括）。「ニュース」ではなく上場会社の適時開示（決算短信・業績予想の修正・自己株式取得・株式分割・
  増資・業務提携・月次売上 など）の表題と PDF/XBRL。今の Standard プラン・V2 の取り込みに載る
- 小型株は、ニュース配信各社でも「ニュースらしいニュース」がほとんど無く、決算の数字から機械的に作られた記事が大半
  （Bigdata.com で小型株1社を引くと 90 日で 6 件、ほぼ Factset Earnings Comparisons）。「ニュース感情」と呼んでも実体は
  適時開示の内容と調子になる

## 調べた経路（すべて実在を確認。jpx-jquants.com・jpx.co.jp・api.jquants.com はセッションから到達できず、仕様は検索結果と公式 Python クライアントの README で確認）

| 経路 | 中身 | 過去分 | 費用 | 東証小型株 | 評価 |
|---|---|---|---|---|---|
| J-Quants 適時開示アドオン | `/v2/td/list`（日付かコードで開示インデックス: 開示番号 14 桁・日時・表題・添付種別。cursor で差分）、`/v2/td/bulk`（5 年分のインデックス CSV、URL 有効 15 分）、`/v2/td/files`（開示番号 → 全文 PDF / サマリ PDF / XBRL の URL、有効 15 分） | 5 年 | 月 11,000 円（税込）。Light 1,650 円以上。個人利用のみ（法人・学術は不可） | 全上場会社 | 第一候補 |
| Bigdata.com（RavenPack） | 企業別のニュース感情（sentiment tearsheet は直近 90 日） | 長い履歴は法人契約 | 法人向け | 登録あり。中身は決算数値の自動生成記事が大半 | 小型株では決算の数字に戻る |
| Alpha Vantage NEWS_SENTIMENT | 英語ニュースの感情（ティッカー単位） | 2022 年〜 | 無料枠あり | SYMBOL_SEARCH で Toyota を引いても東証上場分が出ない（ADR・ロンドン・OTC のみ） | 使えない |
| FMP（Financial Modeling Prep）news | 銘柄別ニュース | 未確認 | Starter 以上の有料プラン（今の契約では拒否） | 未確認 | 保留 |
| GDELT（GKG / GEG） | 100 言語超のニュースのトーン。BigQuery で無料 | 2015 年〜 | BigQuery の無料枠 | 日本語の企業名との照合と記事量が課題 | 大型株の実験用 |
| NewsAPI.org 無料枠 | 一般ニュース検索 | 過去 1 か月のみ・1 日 100 リクエスト・本文 200 文字 | 無料 | 日本語記事は取れる | 履歴が無く学習に使えない |
| PR TIMES | プレスリリース | 公式 API なし。Apify の有料スクレイパーのみ | 従量課金 | 掲載企業のみ | 使わない |
| 日経・QUICK・ロイター | 日本語の銘柄別ニュース | 法人契約 | 非公開（個人向けのロイター配信は月 9,980 円など） | あり | 問い合わせが要る |

## TDnet 経路で進めるときの形（まだ作っていない）

- 取り込み: 毎晩 `/v2/td/list?date=<当日>` を 1 回（Standard は 1 分 120 リクエスト）。初回は `/v2/td/bulk` で 5 年分。
  PDF は `/v2/td/files` で必要な開示だけ（表題で絞る）。Release data-raw に置き、公開ログには件数だけ
- 時刻: 開示時刻（HH:MM）があるので、予測日の引け後の開示は翌営業日の行に回す（docs/DATA_TIMING.md の決まり）
- 感情スコア: 公開の日本語金融感情モデル。bardsai/finance-sentiment-ja-base（東北大 BERT を Financial PhraseBank の訳で学習、
  3 値）、chABSA データセット（TIS、有価証券報告書の極性。LREC 2022）で学習したモデル。表題だけなら軽い。本文は PDF の
  文字起こしが要る
- 特徴量: 直近 20 日・60 日の開示の件数（種類別）、表題の感情の平均・最小、上方修正・下方修正・自己株式取得の有無 など。
  業績予想の修正は今の 239 列に `guidance_revision` があるので、足して効くのは「それ以外の開示」と「文の調子」
- 履歴 5 年なので 2018〜2021 年の行は欠測。実験 89 と同じく「列の分かる行で判定」（§7 を 1.5σ・1.2σ の両方で）
- 学術的な根拠: 決算短信のテキストの極性と株価を調べた研究はある（白方 2018、松田・津田（同志社）など）。極性は業績との
  結びつきが強く、株式リターンとの関係は業績ほど明確でない、と整理されている。1 か月の到達ラベルで効くかはこの計画の
  データで測るしかない

## 出典

- JPX プレスリリース（TDnet アドオン、2026-05-18）: https://www.jpx.co.jp/corporate/news/news-releases/6020/20260518-01.html
- J-Quants 仕様: https://jpx-jquants.com/ja/spec/td-list / https://jpx-jquants.com/ja/spec/td-bulk /
  https://jpx-jquants.com/ja/spec/td-files / https://jpx-jquants.com/ja/spec/rate-limits
- 公式 Python クライアント（get_td_list / get_td_files / get_td_bulk）: https://github.com/J-Quants/jquants-api-client-python
- GDELT GEG: https://blog.gdeltproject.org/global-entity-graph-geg-neural-sentiment-dataset-now-available-in-bigquery/amp/
- bardsai/finance-sentiment-ja-base: https://huggingface.co/bardsai/finance-sentiment-ja-base
- chABSA（LREC 2022）: https://preview.aclanthology.org/moar-dois/2022.lrec-1.758.pdf
- NewsAPI 無料枠の制限: https://zenn.dev/uinoue/articles/660ee202373f64
- PR TIMES Scraper（Apify）: https://apify.com/nexgendata/prtimes-japan-press-releases
- 松田・津田（同志社。決算短信と株価変動）: https://doshisha.repo.nii.ac.jp/record/2000182/files/023064040005.pdf
- FMP 料金: https://financialmodelingprep.com/developer/docs/pricing
