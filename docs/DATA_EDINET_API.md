# 金融庁 EDINET の公式 API（有価証券報告書・半期報告書の損益）

運用者の指示（2026-10-10）: 候補銘柄の直近の決算・有価証券報告書をサイトから直接取り、売上原価・販管費まで
分けた損益のサンキー図を画面の新しいタブに出す。ここは、その取り込み（`research/edinet_filings.py`、
`Fetch Filings`）が何を根拠に作ってあるかの記録。`docs/DATA_EDINETDB.md`（民間の EDINET DB。研究用の年次財務）とは
別の経路で、こちらは金融庁の公式 API。

## 結論を先に

| 項目 | 内容 |
|---|---|
| 経路 | EDINET API Version 2（`https://api.edinet-fsa.go.jp/api/v2/`）。無料。鍵（Subscription-Key）は運用者が EDINET の API キー発行画面で発行し、Secrets `EDINET_FSA_API_KEY` に入れた（2026-10-10） |
| 規約 | EDINET の情報は公共データ利用規約（PDL1.0）。出典を書けば加工して公開してよい。サイトのスクレイピングは禁止で、API が正しい経路。画面の脚注と決算サンキーのタブに出典を出す |
| 取れる書類 | 有価証券報告書（docTypeCode 120）と半期報告書（160）。**四半期報告書は 2024年4月以降に始まる四半期から廃止**されたので、第1・第3四半期の明細は法定の書類に無い（決算短信の XBRL を自動で取ってよい公式の経路は見つからず、運用者の了承で扱わない） |
| 中身 | 書類取得 API の `type=5`（CSV の ZIP）。損益の要素 ID（日本基準 `jppfs_cor:*`、IFRS `jpigp_cor:*`）、コンテキスト、DEI（会計基準・連結の有無・期間の種類） |
| 画面 | `public/data/filings.json`（1銘柄につき最新の1書類。損益の項目と前期の値）。タブ「決算サンキー」が描く |

## 実測（`research/probe_edinet_api.py`、run 38050003592、2026-10-10）

- **鍵の渡し方**: クエリ `Subscription-Key=…` で開いた（HTTP 200）。印字する URL からは鍵を外す
- **書類一覧 API** `documents.json?date=YYYY-MM-DD&type=2`: 1日ぶんが 0.2〜1.8MB（6月下旬の有報の集中日で 1.8MB）、
  応答は中央値 1.1秒・最大 3.8秒。項目は docID（8文字）・edinetCode・secCode（5桁・末尾 0 の J-Quants と同じ形）・
  docTypeCode・periodStart / periodEnd・submitDateTime・csvFlag・withdrawalStatus など 29項目。
  **証券コードで引く API は無い**ので、日付ごとの一覧を索引（`edinet_filings_index.json`）に貯めて探す。
  10/9 から遡って、3月決算の会社の有報（6/25 提出）を見つけるまで 70 リクエスト、6/10 提出のものまで 89
- **書類取得 API** `documents/{docID}?type=5`: ZIP（100〜210KB）。中は `XBRL_TO_CSV/` に本体（`jpcrp030000-asr-001_….csv`、
  0.65〜1.26MB・1,400〜2,100行）と監査報告書の CSV（`jpaud-*`）。**UTF-16（BOM）・タブ区切り**、見出しは
  `要素ID / 項目名 / コンテキストID / 相対年度 / 連結・個別 / 期間・時点 / ユニットID / 単位 / 値`
- **コンテキスト**: 当期の損益は `CurrentYearDuration`（連結）と `CurrentYearDuration_NonConsolidatedMember`（個別）。
  前期は `Prior1YearDuration`。半期報告書は `InterimDuration`（初回の取り込みで確認。下記）
- **DEI**: `AccountingStandardsDEI`（Japan GAAP / IFRS）、`WhetherConsolidatedFinancialStatementsArePreparedDEI`（true/false）、
  `TypeOfCurrentPeriodDEI`（FY）、`CurrentFiscalYearStartDateDEI` / `CurrentPeriodEndDateDEI`、`AmendmentFlagDEI`
- **日本基準の会社**（任天堂、E02367）: 損益の要素 18本のうち 17本があり（無いのは OperatingRevenue1）、恒等式
  （粗利 = 売上 − 原価、営業利益 = 粗利 − 販管費、経常 = 営業 + 営業外収益 − 営業外費用、税引前 = 経常 + 特別利益 −
  特別損失、純利益 = 税引前 − 法人税等）は**すべて成立**
- **IFRS の会社**（トヨタ、E02144）: `jpigp_cor:RevenueIFRS` も `GrossProfitIFRS` も**無い**（売上収益を独自の要素で開示。
  連結の損益は `CostOfSalesIFRS` / `SellingGeneralAndAdministrativeExpensesIFRS` / `OperatingProfitLossIFRS` /
  `ProfitLossBeforeTaxIFRS` / `IncomeTaxExpenseIFRS` / `ProfitLossIFRS` など 8本）。取り込みは、要素 ID で売上が
  無ければ項目名（売上収益・売上高・営業収益）で拾う。粗利の段は無いので、画面は売上から営業利益までを
  「販管費・その他の費用等」の1本にまとめる

## 取り込み（`research/edinet_filings.py`、`.github/workflows/fetch-filings.yml`）

- 毎晩の予測（Predict Breakouts）の後に workflow_run で走る。手でも起動できる
- **索引**: 日付ごとの一覧から 120 / 130 / 160 / 170 で secCode のある行だけを `edinet_filings_index.json` に貯める
  （Release data-raw）。初回は 400 日遡る（平日 約 290 リクエスト・約 8分）。以後は毎日その日ぶん（最後の日は取り直す）
- **対象**: 画面で選べる銘柄＝予測の候補（`predictions.json` の全日付）と 8軸の銘柄（`stocks.json`）。証券コードの
  最新の 120 / 160（取り下げ無し・CSV あり。訂正報告書 130 / 170 は全文が入っているとは限らないので使わない）
- **控え**: 読んだ書類は `edinet_filings_cache.json`（Release）に置き、同じ docID は二度と取らない
- **出力**: `public/data/filings.json`（git。公開してよい）。1銘柄につき {docID, docType, periodStart / periodEnd,
  periodType, submitDate, standard, consolidated, context, items（要素 ID → 円）, labels, priorItems, checks}
- ログには件数だけ（会社名の一覧・財務の値は出さない）。鍵は出さない
- 予算: 1回の実行で 400 リクエストまで（`--max-requests`）。索引の更新は書類取得のぶん（`--reserve` 60）を残して止まる

## 画面（タブ「決算サンキー」、`src/views/FilingsView.jsx`）

- 銘柄を選ぶと、有報・半期報告書の損益の流れ（`src/lib/filings.js` の `toDetailedFlow`）と、直近の決算短信の流れ
  （既存の `EarningsSankey`。四半期の刻みだが原価・販管費は無い）を並べる
- 段: 日本基準は 売上総利益 → 営業利益 → 経常利益（営業外収益が合流）→ 税引前（特別利益が合流）→ 当期純利益 →
  親会社株主に帰属（非支配株主の取り分があるときだけ）。IFRS は 売上総利益 → 営業利益（その他の収益が合流）→
  税引前利益（金融収益が合流）→ 当期利益 → 親会社の所有者に帰属
- 出ていく額は常に「前の段 + 入ってくる額 − この段」（図が釣り合う）。元データに出ていく項目が無い段は「…等」と書く
- 米国基準・銀行・保険など、売上高から始まらない損益は「対象外」と出す
- 出典（EDINET・PDL1.0）と docID を図の下に出す

## 初回の取り込み（run 38051012338、2026-10-10）

- 索引: 400日ぶん（2025-09-05〜2026-10-09）の平日 286日・9,074件（120 / 130 / 160 / 170 で secCode のあるもの）。
  リクエスト 286、約5分。書類の取得 45件で計 331 リクエスト（予算 400）
- 画面の 45銘柄のうち 44銘柄が読めた（日本基準 40・IFRS 4、有報 33・半期報告書 11）。恒等式が全部成立 42
- **半期報告書の当期のコンテキストは `InterimDuration`**（連結 9件）/ `InterimDuration_NonConsolidatedMember`（個別 2件）。
  DEI の `TypeOfCurrentPeriodDEI` は `HY`
- 読めなかった1件は米国基準の会社（標準の要素 `jppfs_cor` / `jpigp_cor` が1本も無い）。2回目から「対象外」として
  控えに残し、画面は書類の種類・会計基準・提出日と「対象外」の理由を出す
- 銀行（日本基準だが売上高が無く、経常収益から始まる）は項目は読めるが段が作れないので、画面は「対象外」
- 恒等式が合わなかった2件は、百万円に丸めた開示で 3項目を足したときの丸めの差（最大 1.5百万円）。判定を
  「開示の丸めの単位 × 項目数 × 0.5 まで許す」に直した（`probe_edinet_api.check_identities`）

## 売買の判断に使えるか（実験71、2026-10-10）

サンキー図で目に入る3つの比率（粗利率・販管費率・営業外依存度）を、EDINET DB の年次財務から実験70 と同じ過去の候補に付けて
際どい候補を分けたが、買う・見送るを分ける根拠は出なかった（`docs/PLAYBOOK.md`「実験71」）。このタブは「何を買うかを知る」
ためのもので、選定の規則には使わない。

## まだ確かめていないこと

- 要素 ID の無い段（IFRS の売上収益を独自要素で開示する会社）の項目名の揺れ。取り込みが「対象外」と数えた件数で追う
- 訂正報告書（130 / 170）の扱い（いまは使わない。訂正で数字が変わった会社は元の書類のまま出る）
- EDINET API の利用枠（公式の記載を確認できていない。1リクエストごとに 0.3秒空けている。初回の 331 リクエストは
  すべて HTTP 200）
