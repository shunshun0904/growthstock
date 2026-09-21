# EDINET DB (edinetdb.jp) の API 実測

`research/probe_edinetdb.py` を GitHub Actions（`Probe EDINET DB`）で叩いた
結果。**実際に返ってきたものだけ**を書く。開発者ページ
（edinetdb.com/developers）は作業環境から到達できない（egress ブロック）ので、
ドキュメントではなく応答から書き起こしている。

- 実施日: 2026-09-21（run 35584440027、7リクエスト）
- 認証: `X-API-Key` ヘッダ。鍵は Secrets の `EDINET_API_KEY`
- ベース: `https://edinetdb.jp/v1`

## なぜ見たか

J-Quants スタンダードでは BS/PL/CF の明細（販管費・研究開発費・有利子負債・
のれん・棚卸資産・設備投資）が取れない（`/fins/details` はプレミアム限定。
`docs/DATA_FIELDS.md`）。EDINET DB がそこを埋められるか、そして
**特徴量に使ってよい形か**（時点整合性）を確かめる。

## 結論を先に

| 確認事項 | 結果 |
|---|---|
| 各値に提出日が付くか | **付く。** `submit_date`（分単位）と `doc_id` が各行にある。J-Quants の `DiscDate` と同じく `merge_asof` で「その時点で知りえた値」に揃えられる |
| 訂正報告書の扱い | `is_restated_eps` / `is_restated_bps` / `is_restated_diluted_eps` のフラグはある。**明細が訂正で上書きされるかは未確認** |
| どこまで遡れるか | 任天堂で **2012〜2026 の15期**。学習データ（2018年〜）を覆う |
| 証券コード | `sec_code` がある。**表記が揺れる**（一覧 `154A` / ランキング `77770`）。結合時に J-Quants 形式（末尾 0 の5桁）へ正規化する。既存の `jquants_data_fetcher.normalize_code`（4桁→末尾 `0`、5桁はそのまま）がそのまま使える |
| 更新頻度 | **年1回（有価証券報告書）**。`financials` は1行＝1事業年度 |
| 網羅性 | 新規上場は無い（一覧1社目 E39493 GAIA は `404 No financial data`） |
| 利用枠 | **日 100 / 月 900**（応答ヘッダ `x-ratelimit-limit: 100`, `x-ratelimit-monthly-limit: 900`）。月の上限が実質の制約 |

## 叩いたエンドポイントと応答

| エンドポイント | HTTP | 中身 |
|---|---:|---|
| `GET /status`（鍵不要） | 200 | `available_industries`（33業種の英語スラッグ）、`available_metrics`（`roe`, `operating-margin`, `roa`, `roic`, `per`, `pbr`, `psr`, `market-cap`, `ev`, `fcf-yield`, `ev-ebitda`, `dividend-yield`, `payout-ratio`, `doe`, `health-score`, `credit-score`, `revenue-growth`, `eps-growth`, `revenue-cagr-3y` …。3,390文字で途中まで） |
| `GET /companies?per_page=3` | 200 | 11項目: `accounting_standard`, `credit_rating`, `credit_score`, `edinet_code`, `industry`, `is_delisted`, `listing_status`, `name`, `name_en`, `name_ja`, `sec_code` |
| `GET /companies/E02367/financials?years=5` | 200 | **5行 × 128項目**（下記） |
| `GET /companies/E02367/financials?years=30` | 200 | 15行、`fiscal_year` 2012〜2026 |
| `GET /companies/E39493/financials?years=2` | 404 | `{"error":{"code":"not_found","message":"No financial data for E39493"}}` |
| `GET /rankings/roe?limit=3` | 200 | 10項目: `edinet_code`, `fiscal_year`, `industry`（日本語）, `name`, `name_en`, `name_ja`, `rank`, `sec_code`, `unit`, `value` |
| `GET /screener` | 400 | `No screening conditions provided. Use 'conditions' JSON or shorthand params like 'roe_gte=10'` |

## `financials` の128項目（任天堂 E02367、5期）

充足は「5期のうち値が入っていた期数」。**任天堂に無いもの（有利子負債・のれん）が
「項目として存在しない」のか「値が無いので省かれた」のかは、この1社では
分からない。**

### 時点・出典

| 項目 | 充足 | 例 |
|---|---:|---|
| `submit_date` | 5/5 | `2022-06-30 13:56` |
| `doc_id` | 5/5 | `S100O9PP` |
| `fiscal_year` | 5/5 | `2022` |
| `edinet_filing_url` / `edinet_view_url` / `edinet_filing_url_availability` | 5/5 | disclosure2.edinet-fsa.go.jp へのリンク / `available` |
| `accounting_standard` | 5/5 | `JP` |
| `basis` / `basis_source` | 5/5 | `consolidated` / `value_provenance` |
| `is_restated_eps` / `is_restated_bps` / `is_restated_diluted_eps` | 5/5 | `False` |
| `revenue_non_consolidated_source` | 5/5 | `jpcrp_cor:NetSalesSummaryOfBusinessResults` |
| `shares_section_source` | 4/5 | `phase6_zgx_llm` |
| `treasury_shares_source` / `treasury_shares_quality_flag` | 5/5 | `xbrl_struct_jp` / `ok_name_match` |
| `trust_note_type` / `trust_quality_flag` | 4/5 | `none` |

### 損益（PL）

| 項目 | 充足 | 項目 | 充足 |
|---|---:|---|---:|
| `revenue` | 5/5 | `revenue_non_consolidated` | 5/5 |
| **`cost_of_sales`** | 5/5 | **`gross_profit`** | 5/5 |
| **`sga`** | 5/5 | **`rnd_expenses`** | **3/5** |
| `operating_income` | 5/5 | `ordinary_income` | 5/5 |
| `non_operating_income` / `non_operating_expenses` | 5/5 | `extraordinary_income` / `extraordinary_loss` | 5/5 |
| `profit_before_tax` | 5/5 | `income_taxes` | 5/5 |
| `net_income` | 5/5 | `comprehensive_income` | 5/5 |
| `non_controlling_interests` | 5/5 | `effective_tax_rate` | 5/5 |
| `interest_income` / `interest_expenses` | 5/5 | `depreciation` | 5/5 |

### 貸借（BS）

| 項目 | 充足 | 項目 | 充足 |
|---|---:|---|---:|
| `total_assets` | 5/5 | `total_liabilities` | 5/5 |
| `current_assets` / `noncurrent_assets` | 5/5 | `current_liabilities` / `noncurrent_liabilities` | 5/5 |
| `cash` | 5/5 | `short_term_securities` | 5/5 |
| `trade_receivables` | 5/5 | `trade_payables` | 5/5 |
| **`inventories`** | 5/5 | `allowance_for_doubtful_accounts` | 5/5 |
| `ppe` | 5/5 | `land` / `buildings_and_structures` / `machinery_and_equipment` / `construction_in_progress` | 5/5 |
| `intangible_assets` / `software` | 5/5 | `investment_securities` / `investments_and_other_assets` | 5/5 |
| `deferred_tax_assets` | 5/5 | `net_defined_benefit_asset` / `net_defined_benefit_liability` | 5/5 |
| `advances_received` / `contract_liabilities` | 5/5 | `provision_for_bonuses` | 5/5 |
| `net_assets` | 5/5 | `shareholders_equity` | 5/5 |
| `capital_surplus` | 5/5 | `retained_earnings` | 5/5 |
| `treasury_stock` | 5/5 | `equity_ratio_official` | 5/5 |

### キャッシュフロー（CF）

| 項目 | 充足 |
|---|---:|
| `cf_operating` / `cf_investing` / `cf_financing` / `cf_exchange_effect` | 5/5 |
| **`capex`** | 5/5 |
| `cash_dividends_paid` | 5/5 |

### 株式・配当・株主還元

| 項目 | 充足 | 項目 | 充足 |
|---|---:|---|---:|
| `shares_issued` / `shares_issued_fiscal_year_end` | 5/5 | **`float_shares`** | 5/5 |
| `eps` / `bps` / `per` | 5/5 | `adjusted_eps` / `adjusted_bps`（分割調整済み） | 5/5 |
| `split_adjustment_factor` | 5/5 | `roe_official` | 5/5 |
| `dividend_per_share` / `adjusted_dividend_per_share` | 5/5 | `adjusted_interim_dividend_per_share` / `adjusted_yearend_dividend_per_share` | 4/5 |
| `adjusted_dividend_basis` | 5/5 | `payout_ratio` / `payout_ratio_reported` | 4/5 / 5/5 |
| `dividends_total_announced` | 5/5 | `total_shareholder_return` / `total_return_share_price_index` | 5/5 |
| `treasury_shares_count` / `treasury_shares_count_kei` / `treasury_shares_as_of_date` | 5/5 | **`treasury_cancelled_count`** / `treasury_cancelled_amount` | 4/5 |
| `treasury_end_period_holding_count` | 4/5 | `treasury_disposed_solicitation_*` / `treasury_merger_transfer_*` / `treasury_other_disposed_*` | 4/5 |
| **`cross_shareholding_total_book_value`** / `cross_shareholding_total_shares_held` | 5/5 | `directors_shares_held` / **`directors_ownership_ratio`** | 5/5 |

### 人的資本・役員（有報の非財務情報）

| 項目 | 充足 |
|---|---:|
| `num_employees` / `avg_age` / `avg_annual_salary` / `avg_tenure_years` | 5/5 |
| `director_remuneration_total` / `director_remuneration_headcount` | 5/5 |
| `male_directors` / `female_directors` / `female_director_ratio` | 5/5 |
| `female_manager_ratio` / `gender_pay_gap_all` / `gender_pay_gap_regular` / `gender_pay_gap_nonregular` / `male_parental_leave_ratio` | 3/5（開示義務化以降のみ） |

## 単位と調整

- 金額は円（任天堂 FY2022 の `revenue` = 1,695,344,000,000）。
- `eps` は当時の株数ベース（FY2022 = 4,046.69）、`adjusted_eps` は分割調整後
  （404.669、`split_adjustment_factor` = 10。任天堂は2022年10月に1:10分割）。
  株価と突き合わせるなら `adjusted_*` を使う。

## 使うときの制約

**1社＝1リクエスト**（`years=30` で全期が返る）。約3,850社の履歴を揃えるには
月900の枠で **約4.3か月**。検証には全社は要らないので、母集団に多く出る
数百社を先に取れば1か月以内に A/B 実験ができる（`docs/MODEL_TUNING_NOISE.md`
の流儀で、同じ行に特徴量を足す/足さないを比べる）。

年1回・期末から約3か月遅れの更新なので、20営業日のブレイク予測では
「直近の変化」ではなく**動きの遅い水準特徴量**（浮動株比率・自社株買い・
capex/減価償却・棚卸資産回転・販管費率）として効くかどうか、という問い。
決算特徴量はこれまでウォークフォワードで 0勝9敗（`docs/MODEL_FUNDAMENTAL_COVERAGE.md`）
なので、事前確率は高くない。判定は実収益の窓平均で z>2。

## 未確認（次の probe で見る）

- 有利子負債・のれんの項目名（借入とのれんのある大企業で確かめる）
- 訂正報告書で明細が上書きされるか（`doc_id` が訂正報告書のものに変わるか）
- 一括取得の経路（`/screener` の短縮引数、`/companies` のページサイズ）。
  11エンドポイントのうち、ここで叩いたのは5つ
- 半期報告書のデータが別エンドポイントにあるか
