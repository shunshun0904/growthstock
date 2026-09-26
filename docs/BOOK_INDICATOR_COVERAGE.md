# ファンダメンタルズ分析の本の目次 × いまの特徴量・データ源（2026-09-26）

運用者の依頼: 本（目次4枚）の各指標が、いまの特徴量（`all_plus_prog_listing` 206列）でどの程度
網羅されているか、いまのデータ源（J-Quants スタンダード・EDINET DB）で作れるか、追加のデータ源が
要るか、その取得可能性を表にする。

**根拠にしたもの**（推測で書いていない）
- 特徴量: `research/features.py` の `DEFAULT_PRESET`（206列）と `research/feature_dict.py` の説明
- J-Quants スタンダードで **実測して使えている** エンドポイント（`research/jq_bulk.py`）:
  `/fins/summary`（決算短信の要約、111項目）、`/equities/bars/daily`、`/equities/master`、
  `/indices/bars/daily`（79指数、名称なし）、`/markets/margin-interest`、`/markets/margin-alert`、
  `/markets/short-sale-report`、`/markets/short-ratio`、`/equities/valuation`（予想 EPS/PER/ROE）、
  `/edinet/major-shareholders`、`/edinet/large-volume-shareholders`、`/edinet/cross-shareholdings`、
  `/fins/earnings-date`、`/equities/investor-types`、`/markets/calendar`
- J-Quants で **プレミアム限定と実測**したもの（`docs/DATA_FIELDS.md`）: `/fins/details`（BS/PL/CF の
  明細）、`/fins/dividend`、`/markets/breakdown`。「存在しない」と返ったもの: 適時開示・アナリスト予想・
  株主構成の当てずっぽうのパス（正しいパス名は未確認）
- EDINET DB（`docs/DATA_EDINETDB.md`）: 有価証券報告書の**年次**財務 128項目（会社により有無が違う）、
  日 100 / 月 900 の枠。2026-09-26 時点で **395社・5,823期** 取得済み（母集団 3,735社の 10.6%。
  `research/_data/edinet_fin.parquet`）。特徴量の台本 `research/edinet_features.py` はあるが
  **本番の206列には入っていない**（実験28、取得が揃いしだい）
- 外部: このセッションの Alpha Vantage 接続を実際に叩いた（米10年国債利回り 月次 1953年〜が返る、
  USD/JPY 日足が返る。**無料キーは 25 リクエスト/日**で、2本目の CPI は上限で失敗）。FRED / e-Stat /
  内閣府 / 日銀 / 財務省は一般に公開されている統計サイトとして挙げる（この作業環境は egress が
  制限されていて疎通は未確認。GitHub Actions からの取得はキーの登録が要る）

**記号**

| 記号 | 意味 |
|---|---|
| ◎ | 既に特徴量（206列に入っている） |
| ○ | 取得済みのデータから計算できるが、特徴量にはしていない |
| △ | 一部だけ / 代理（年次のみ、ETF の代理、粗い近似） |
| × | いまのデータ源に無い。追加の源が要る |
| — | 数値の指標ではない（概念・読み方・練習問題） |

## 第1章 企業の業績や経済状況から読み取れること

| 項目 | 網羅 | いまの特徴量 / 作り方 | データ源 | 備考 |
|---|:-:|---|---|---|
| ファンダメンタルズ分析は定量と定性の両方を見る | — | 定性（事業内容・経営者）は無い | — | テキストを扱う設計はしていない |
| 経済状況や企業の業績が株価に大きく影響する | — | 地合いは `topix_*` / `nk225_*` / `sector_*`、業績は `fund_*` | J-Quants | |
| バリュー株に投資する（時価総額） | ◎ | `log_market_cap`, `cap_band`, `per`, `pbr`, `earnings_yield`, `book_yield` | J-Quants | |
| グロース株に投資する（成長性） | ◎ | `sales_growth_*`, `eps_growth_*`（水準・変化・加速・連続）, `peg` | J-Quants | |
| テーマ株に投資する（トレンド） | × | テーマの分類が無い | 外部（ニュース・テーマ辞書） | 数値化には銘柄→テーマの対応表が要る。無料の公的な源は無い |
| テーマ株を売買するタイミング | — | | | |

## 第2章 決算書を見て企業の実力を分析する

| 項目 | 網羅 | いまの特徴量 / 作り方 | データ源 | 備考 |
|---|:-:|---|---|---|
| 3カ月ごとの業績の変化を決算短信で確認 | ◎ | `fund_*` 96列（四半期の売上・EPS・ROE・ROA・営業利益率・自己資本比率の水準/変化）, `progress_pct`, `days_since_disc` | J-Quants `/fins/summary` | |
| 有価証券報告書を確認し事業の詳細を把握 | △ | 有報の**数値**だけ（EDINET DB）。事業の記述は無い | EDINET DB | 本文は金融庁 EDINET API（XBRL）を直接読む必要がある（要キー登録、無料。この環境から未確認） |
| 大量保有報告書を見て大株主の動向 | ◎ | `lvs_days`, `lvs_ratio`, `lvs_ratio_chg`, `lvs_n_20/60/250`（大量保有）、`mjr_*` 7列（大株主） | J-Quants `/edinet/large-volume-shareholders`, `/edinet/major-shareholders` | |
| 決算は直近の数字のほか業績予想も確認 | ◎ | `guidance_op_growth`, `guidance_revision`, `rev_*` 8列（修正イベント）, `jq_fwdeps/fwdper/fwdroe` | J-Quants | |
| USGAAP や IFRS など会計基準の違い | ○ | EDINET DB の `accounting_standard`（JP / IFRS / USGAAP）を取得済み。特徴量は未作成（境界の guard に使用） | EDINET DB | 会計基準そのものより「基準をまたぐ年度を比べない」ために使う |
| セグメント情報で伸びている事業 | × | 無い | 金融庁 EDINET API（XBRL のセグメント注記） | EDINET DB の128項目にセグメントは無い。J-Quants にも無い |
| 損益計算書を読み解いて売上と利益の流れ | ◎ | 売上・営業利益・経常利益・純利益（`op_margin`, `ordinary_margin`, `net_margin` と成長率） | J-Quants | |
| 売上高の伸び | ◎ | `sales_growth_q0..q3`, `_chg`, `_accel`, `_up_streak` | J-Quants | |
| 売上総利益の伸び | △ | EDINET DB `gross_profit`（年次。銀行は無し） | EDINET DB（年次）/ J-Quants プレミアム `/fins/details`（四半期） | スタンダードの短信要約に売上総利益は無い |
| 営業利益 | ◎ | `op_margin_*`, `guidance_op_growth` | J-Quants | |
| 経常利益 | ◎ | `ordinary_margin` | J-Quants | IFRS 企業は経常利益が無い（欠測） |
| 当期純利益 | ◎ | `net_margin`, `eps_growth_*`, `eps_growth_turn` | J-Quants | |
| 売上高総利益率 | △ | EDINET DB `gpm`（年次） | 同上 | |
| 売上高営業利益率 | ◎ | `op_margin_q0..q3` と変化 | J-Quants | |
| 売上高経常利益率 | ◎ | `ordinary_margin` | J-Quants | |
| 研究開発費 | △ | EDINET DB `rnd_expenses`, `rnd_r`（年次。2016年度以降の充足 約50%） | EDINET DB | J-Quants スタンダードに無い |
| 設備投資 | △ | EDINET DB `capex`, `capex_dep`（年次。充足 約65%） | EDINET DB | |
| 貸借対照表で資産状況 | ◎/△ | 総資産・自己資本は `equity_ratio_*`, `ROA_*`。内訳は EDINET DB（年次） | J-Quants / EDINET DB | |
| 流動負債 | △ | EDINET DB `current_liabilities`（年次） | EDINET DB / J-Quants プレミアム | |
| 固定負債 | △ | EDINET DB `noncurrent_liabilities` | 同上 | |
| 有利子負債 | △ | EDINET DB `ibd_*`（85% の会社で無く、`short_term_loans` / `long_term_loans` / 社債 / CP の合計で代用）。`debt_r` | EDINET DB | |
| 純資産 | ◎ | `Eq`（自己資本）→ `equity_ratio_*`, `pbr`, `book_yield` | J-Quants | |
| 利益剰余金 | △ | EDINET DB `retained_earnings`, `re_r`, `re_mcap` | EDINET DB | |
| 自己資本比率 | ◎ | `equity_ratio_q0..q3`, `_chg`, `_accel` | J-Quants | |
| 資産の内訳と総資産 | ◎/△ | 総資産は ◎（`ROA`, `asset_turnover`）。内訳は EDINET DB | J-Quants / EDINET DB | |
| 流動資産の内訳で資金繰りリスク | △ | EDINET DB `current_assets`, `cash`, `trade_receivables`, `inventories`（`cash_r`, `inv_r`, `rec_r`） | EDINET DB | |
| 長期保有する資産（固定資産） | △ | EDINET DB `noncurrent_assets`, `ppe`, `intangible_assets`, `investment_securities` | EDINET DB | |
| 減価償却のしくみ | △ | EDINET DB `depreciation`（年次） | EDINET DB | |
| 流動比率 | △ | EDINET DB `current_assets ÷ current_liabilities`（年次。台本に無いが計算できる） | EDINET DB | |
| 固定比率 | △ | EDINET DB `noncurrent_assets ÷ net_assets`（同上） | EDINET DB | |
| 総資産回転率 | ◎ | `asset_turnover` | J-Quants | |
| ギアリング比率 | △ | EDINET DB 有利子負債 ÷ 自己資本（`debt_r` は総資産比。分母を変えるだけ） | EDINET DB | |
| キャッシュフロー計算書 | ○/◎ | J-Quants の `CFO/CFI/CFF` は **2Q と通期だけ**（1Q/3Q は 8〜11%）。◎ `cfo_yield`, `fcf_yield`, `cfo_to_op`, `accruals` | J-Quants（半期）/ EDINET DB（年次） | |
| 現金同等物の残高 | ○ | J-Quants `CashEq`（2Q/通期）、EDINET DB `cash`。特徴量は未作成（`netcash_mcap` は EDINET 台本にある） | 同上 | |
| 営業C/F | ◎ | `cfo_yield`, `cfo_to_op` | J-Quants | |
| 投資C/F | ○ | J-Quants `CFI`（`fcf_yield` の中で使用。単独の列は無い） | J-Quants | |
| 財務C/F | ○ | J-Quants `CFF`。未使用 | J-Quants | |
| フリーキャッシュフロー | ◎ | `fcf_yield`（CFO + CFI ÷ 時価総額） | J-Quants | |
| CFPS | ○ | `CFO ÷ ShOutFY` で計算できる。未作成 | J-Quants | |
| キャッシュフローの増減 | ○ | 半期ごとの差で計算できる。未作成 | J-Quants | |
| EBITDA | △ | EDINET DB `operating_income + depreciation`（年次。台本 `_ebitda`） | EDINET DB | J-Quants に減価償却が無い |
| EV / EBITDA 倍率 | △ | EDINET DB 台本 `ev_ebitda`（年次） | EDINET DB + 時価総額 | |
| DCF 法・時価純資産法 | × | 評価モデル。将来 CF の前提（割引率・成長率）と資産の時価が要る | — | データではなく仮定の問題。特徴量にするなら「簡易 DCF の理論株価 ÷ 株価」だが前提が恣意的 |
| 財務三表の相関関係 | — | | | |
| 夜間取引（PTS）を活用 | × | J-Quants に PTS の価格は無い | PTS 運営（ジャパンネクスト等）のデータ | 取得手段は未確認 |

## 第3章 株価指標を見て売買する銘柄を判断する

| 項目 | 網羅 | いまの特徴量 / 作り方 | データ源 | 備考 |
|---|:-:|---|---|---|
| ROE | ◎ | `ROE_q0..q3`, `_chg`, `_accel`, `_up_streak`, `jq_fwdroe`, `jq_roe_gap` | J-Quants | |
| ROA | ◎ | `ROA_*` 同上 | J-Quants | |
| PER | ◎ | `per`, `earnings_yield`, `jq_fwdper`, `jq_per_gap` | J-Quants | |
| PBR | ◎ | `pbr`, `book_yield` | J-Quants | |
| EPS | ◎ | `eps_growth_*`, `jq_fwdeps`, `jq_fwd_earnings_yield` | J-Quants | |
| BPS | ◎ | `pbr` の分母。BPS の変化そのものは未作成（`Eq` の変化で代用可） | J-Quants | |
| PSR（赤字企業の割安度） | ◎ | `psr`, `sales_yield` | J-Quants | |
| PEG レシオ | ◎ | `peg` | J-Quants | |
| サスティナブル成長率（ROE × (1 − 配当性向)） | ○ | `ROE`（通期）× (1 − `PayoutRatioAnn`) で計算できる（通期のみ、配当性向の充足 61%）。未作成 | J-Quants | |
| 株主資本回転率 | ○ | `Sales ÷ ShEq` で計算できる。未作成 | J-Quants | |
| シラー PER（10年平均の実質 EPS） | △ | EPS は EDINET DB に 15期分（`adjusted_eps`）。物価調整の CPI は外部 | EDINET DB + 総務省 CPI（e-Stat） | 名目のまま10年平均なら EDINET DB だけで作れる |
| 保有株や投資先候補は決算情報で動向を確認 | — | | | |
| 株価下落の一因となる増資の情報 | △ | `ShOutFY`（開示ごとの発行済株式数）の増加で粗く検知できる。適時開示の本文は無い | J-Quants / 金融庁 EDINET API（有価証券届出書） | 「予定」の段階では取れない |
| 減益になっても株価が上がる要因（練習問題①②） | — | | | |
| 配当と優待でインカムゲイン | ◎/× | 配当は `div_yield`, `payout_ratio`, `has_dividend`, `days_since_divrev`。**株主優待は無い** | J-Quants / 優待は外部 | 優待の公的な機械可読データは無い（民間サイトのみ） |
| 増配であれば買いが増す | △ | 配当予想の修正イベントは `days_since_divrev`。増配率（`DivFY` / `FDivFY` の前年比）は計算できるが未作成 | J-Quants | |

## 第4章 経済指標を使って景気動向を見通す

J-Quants にも EDINET DB にも**経済統計は一つも無い**。この章は全項目が外部の源になる。共通の注意:

1. **公表日で結合する**（時点整合）。統計は「対象月」と「公表日」が違い、改定もある。改定前の値を
   公表日で結合しないと先読みになる（FRED は ALFRED で改定前の値が取れる。e-Stat は改定履歴が
   取りにくい）
2. 同じ日はどの銘柄でも同じ値になるので、効き方は**地合い**（`topix_ret_20` など既存の11列と同じ
   種類）。実験50 では高スコアの外れが月に固まっていたので、地合いの列が増えれば効く余地はある
3. 月次・四半期の統計は 20営業日の予測には粗い。まず「直近の公表からの日数」「予想との差」のような
   イベント特徴量として測るのが筋

| 項目 | 網羅 | 作り方 | データ源（候補） | 取得可能性 |
|---|:-:|---|---|---|
| 名目値・実質値と季節調整値 | — | | | |
| 日本の GDP | × | 四半期・公表は期末後 1.5か月 | 内閣府（四半期別 GDP 速報、CSV）/ e-Stat API | 無料。e-Stat は API キー登録が要る |
| 米国の GDP | × | 同上 | FRED API（GDP, GDPC1）/ Alpha Vantage `REAL_GDP` | 無料キー（FRED 無制限に近い、Alpha Vantage 25回/日）。Alpha Vantage はこのセッションで疎通を確認 |
| 貿易収支 | × | 月次 | 財務省 貿易統計 / e-Stat | 無料 |
| 消費者物価指数 CPI（日本） | × | 月次 | 総務省 / e-Stat API | 無料 |
| 企業物価指数 PPI（日本） | × | 月次 | 日本銀行 時系列統計データ検索サイト（CSV） | 無料。API は無く CSV ダウンロード |
| 一般職業紹介状況（有効求人倍率）・労働力調査 | × | 月次 | 厚労省・総務省 / e-Stat API | 無料 |
| 米国の雇用統計 | × | 月次 | FRED（PAYEMS, UNRATE）/ Alpha Vantage `NONFARM_PAYROLL`, `UNEMPLOYMENT` | 無料キー |
| 米国の個人消費支出と個人所得 | × | 月次 | FRED（PCE, PI） | 無料キー |
| 消費者信頼感指数（カンファレンスボード） | × | 月次 | カンファレンスボード（**有料**）。代理: ミシガン大消費者態度指数 FRED（UMCSENT） | 本家は有料。代理は無料 |
| 中古住宅販売件数 | × | 月次 | FRED（EXHOSLUSM495S、全米不動産協会） | 無料キー |
| 新築住宅販売件数 | × | 月次 | FRED（HSN1F） | 無料キー |
| 住宅建築許可件数と住宅着工件数 | × | 月次 | FRED（PERMIT, HOUST） | 無料キー |
| S&P ケース・シラー住宅価格 | × | 月次（2か月遅れ） | FRED（CSUSHPINSA） | 無料キー |
| 米国の小売売上高 | × | 月次 | FRED（RSAFS）/ Alpha Vantage `RETAIL_SALES` | 無料キー |
| 卸売在庫と卸売売上高 | × | 月次 | FRED（WHLSLRIMSA ほか） | 無料キー |
| 日銀短観 | × | 四半期 | 日本銀行（短観のページ、CSV） | 無料 |
| 景気ウォッチャー調査（街角景気） | × | 月次 | 内閣府（CSV） | 無料 |
| ISM 製造業景況指数 | × | 月次 | ISM（**会員向け**。公表値は報道で入手可） | 機械可読の無料の源は未確認 |
| ISM 非製造業景況指数 | × | 同上 | 同上 | 同上 |
| フィラデルフィア連銀製造業景況指数 | × | 月次 | FRED（GACDFSA066MSFRBPHI）/ フィラデルフィア連銀 | 無料キー |
| 機械受注統計 | × | 月次 | 内閣府（CSV） | 無料 |
| 鉱工業指数 | × | 月次 | 経産省 / e-Stat API | 無料 |
| 米国の耐久財受注 | × | 月次 | FRED（DGORDER）/ Alpha Vantage `DURABLES` | 無料キー |
| バルチック海運指数 | × | 日次 | Baltic Exchange（**有料**）。無料の二次配信は要確認 | 未確認 |
| 生産年齢人口の推移 | × | 年次〜月次（人口推計） | 総務省 / e-Stat | 無料。動きが遅く20営業日の予測には効かないと考える |
| 地政学リスク | × | 指数化は難しい。学術指数（GPR index、Caldara & Iacoviello）が公開されている | 研究者の公開 CSV | 存在は知られているが、この環境から未確認 |
| 機械受注統計に影響を受けた銘柄は / 雇用統計からわかる株価の動向は（練習問題） | — | | | |
| 経済指標は相関する | — | | | |

## 第5章 株価への影響が大きい金融指標を読み解く

| 項目 | 網羅 | 作り方 | データ源（候補） | 取得可能性 |
|---|:-:|---|---|---|
| 財政政策と金融政策 | — | | | |
| 日本銀行の B/S | × | 旬次（営業毎旬報告） | 日本銀行 時系列統計 | 無料 CSV |
| FRB の B/S | × | 週次 | FRED（WALCL） | 無料キー |
| ECB の B/S | × | 週次 | ECB Data Portal（API） | 無料 |
| 日本10年国債利回り | × | 日次 | 財務省 国債金利情報（CSV）| 無料。J-Quants で確認した ETF に国内債券の指数連動はまだ見つけていない（`docs/MARKET_DATA.md` は名称検索の結果のみ） |
| 米10年国債利回り | ×/△ | 日次 | FRED（DGS10）/ Alpha Vantage `TREASURY_YIELD`（**このセッションで疎通確認**、月次 1953年〜）。代理: J-Quants の ETF `14820`（米国債7-10年、為替ヘッジあり、2016年〜） | 無料キー / 代理は取得済みの日足 |
| 日本の政策金利 | × | 変更は年数回 | 日本銀行 | 無料。決定会合の日付と併せてイベント化 |
| 無担保コールレート翌日物 | × | 日次 | 日本銀行 時系列統計 | 無料 CSV |
| FF 金利 | × | 日次/月次 | FRED（DFF, FEDFUNDS）/ Alpha Vantage `FEDERAL_FUNDS_RATE` | 無料キー |
| 日銀金融政策決定会合 | × | 日程（イベント） | 日本銀行 | 無料。「会合まで何日」「会合の翌日か」のフラグにできる |
| FRB 議長発言 | × | テキスト/イベント | FOMC 日程（FRB） | 日程は無料。発言の内容は数値化しない |
| ECB の金融政策 | × | イベント | ECB | 無料 |
| 英国の金融政策委員会 | × | イベント | BoE | 無料 |
| 米大統領の発言 | × | テキスト | — | 数値化しない |
| サミットの声明 | × | イベント | — | 日程のみ |
| 有名投資家の見解 | × | テキスト | — | 数値化しない |
| ドルインデックス | × | 日次 | ICE DXY は有料。代理: FRED の貿易加重ドル指数（DTWEXBGS）、USD/JPY は Alpha Vantage `FX_DAILY`（**疎通確認**） | 無料キー |
| 日経平均と TOPIX | ◎ | `topix_ret_20/120`, `topix_vol_20`, `topix_ma200_gap`, `nk225_ret_20/120` | J-Quants（TOPIX は `/indices/bars/daily` の `0000`、日経は ETF `13210`） | |
| 3指数（NYダウ・S&P500・NASDAQ） | △ | 東証 ETF `15460`（ダウ、為替ヘッジなし）、`15470`（S&P500）、`15450`（NASDAQ-100、ヘッジなし）の日足。**円建て＝為替込み** | J-Quants（取得済み）/ 現地指数は Alpha Vantage・FMP | ETF は 2016年〜の日足がある |
| 原油価格 | △ | 東証 ETF `16710`（WTI 連動、2016年〜） | J-Quants / FRED（DCOILWTICO）/ Alpha Vantage `WTI` | ETF は取得済み |
| BEI（ブレークイーブンインフレ率） | × | 日次 | 財務省（物価連動国債の BEI）/ 米は FRED（T10YIE） | 無料 |
| 指標発表日カレンダー | × | 各統計の公表予定 | 各省庁 / FRED の release calendar | J-Quants `/markets/calendar` は**取引日**だけ |

## まとめ

| 章 | 項目数 | ◎ 既に特徴量 | ○ 計算できる | △ 一部/代理 | × 追加の源が要る | — 概念 |
|---|---:|---:|---:|---:|---:|---:|
| 1 企業の業績や経済状況 | 6 | 2 | 0 | 0 | 1 | 3 |
| 2 決算書 | 46 | 15 | 5 | 21 | 3 | 2 |
| 3 株価指標 | 17 | 10 | 2 | 3 | 0 | 2 |
| 4 経済指標 | 31 | 0 | 0 | 0 | 27 | 4 |
| 5 金融指標 | 25 | 1 | 0 | 2 | 21 | 1 |

（◎/△ のように2つ付けた行は前者で数えた）

1. **第2〜3章（企業側）はほぼ網羅**。J-Quants スタンダードの短信要約で取れる範囲（売上・利益・
   EPS・BPS・ROE・自己資本・CF の4本・配当・予想）は特徴量になっている。抜けは **BS/PL/CF の明細**
   （売上総利益・販管費・研究開発費・設備投資・減価償却・有利子負債・流動/固定の内訳・利益剰余金）で、
   これは J-Quants ではプレミアム限定、EDINET DB なら**年次**で取れる（取得 10.6%、枠の都合で全社
   まで約4か月）。実験28 が「取得が揃いしだい」で止まっている
2. 取得済みで未使用のもの（○）は5つあり、追加の API 呼び出しなしで試せる: 現金同等物、投資CF・
   財務CF の単独列、CFPS、CF の増減、サスティナブル成長率、株主資本回転率、増配率
3. **第4〜5章（マクロ）は現状ゼロ**。J-Quants・EDINET DB には無く、全部が外部。無料で機械可読なのは
   FRED（米国。無料キー）、e-Stat（日本。無料キー）、内閣府・日銀・財務省の CSV。有料なのは
   カンファレンスボード・ISM・バルチック海運指数・ICE ドルインデックス。Alpha Vantage は
   このセッションで疎通したが 25回/日
4. マクロを入れるなら、値そのものより**公表日で結合したイベント/サプライズ**として、既存の地合い
   11列と同じ枠で A/B（§7 の手順）にかける。20営業日の予測に月次統計が効くかは事前確率が低い
   （既存の地合い列も §13 で「月ごとの当たり外れ」を説明しきれていない）
