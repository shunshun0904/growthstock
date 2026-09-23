# GrowthStockAnalyzer — Focus

ウィリアム・オニールの CANSLIM 手法とマーク・ミネルヴィニのトレンドテンプレートを定量化した、
**8軸モメンタム・スコアリング成長株分析ダッシュボード**です。

データは [J-Quants API **V2**](https://jpx-jquants.com/)（日本取引所グループ公式）から取得します。

## 🔗 公開ダッシュボード

### **https://shunshun0904.github.io/growthstock/**

GitHub Pages で公開しています。平日 21:00 JST のデータ更新後に自動で再デプロイされます。

![8軸オクタゴン比較](docs/screenshot-compare.png)

> スクリーンショットは J-Quants API V2 から取得した実データで撮影したものです（2026-09-04 時点）。
> なお CI のスモークテストは日々値が変わる実データではなく、決定的な合成データ
> （`tests/fixtures/synthetic-stocks.json` / 銘柄名は「テスト銘柄◯」）を使用しています。

---

## 1. アーキテクチャ

APIキーはブラウザに置けないため、**取得はGitHub Actions、表示は静的サイト**という2段構成にしています。

```
  GitHub Actions (secrets.JQUANTS_API)
        │
        ▼
  scripts/jquants_data_fetcher.py  ──REST──>  J-Quants API
        │  (株価 / 財務 / 信用残)
        ▼
  public/data/stocks.json  ── commit ──> リポジトリ
        │
        ▼
  Vite + React ダッシュボード ── deploy ──> GitHub Pages
        │
        ├── 8軸オクタゴン比較
        ├── タイムマシーン・モード
        └── What-If 感度シミュレーター
```

ブラウザ側は生成済みの JSON を読むだけなので、**APIキーがクライアントに露出しません**。

---

## 2. セットアップ

### 2.1 Repository secret の登録

J-Quants ダッシュボードの **[設定 » API キー]** で APIキーを発行し、
GitHub の `Settings > Secrets and variables > Actions` に登録します。

| Secret | 必須 | 内容 |
| --- | :---: | --- |
| `JQUANTS_API` | ✅ | J-Quants API **V2** の APIキー |

> ### V1 から V2 への移行について
>
> J-Quants API は 2025年12月に **V2** へ移行し、V1 は 2026年6月1日に終了しました。
> 認証方式が変わっています。
>
> | | V1（終了） | V2（現行） |
> | --- | --- | --- |
> | 認証 | メール+パスワード → リフレッシュトークン → IDトークン の3段階 | **APIキー1つ** |
> | ヘッダー | `Authorization: Bearer <IDトークン>` | **`x-api-key: <APIキー>`** |
> | 有効期限 | リフレッシュトークン1週間 / IDトークン24時間 | **期限なし**（再発行するまで有効） |
>
> V2 には有効期限がないため、**定期実行のためのキー更新作業は不要**です。
>
> `JQUANTS_API` に V1 のトークン（`xxx.yyy.zzz` 形式の JWT・800文字超）が残っていると
> 必ず認証エラーになります。その場合ログに次のように表示されます（値そのものは出力されません）。
>
> ```
> [auth] JQUANTS_API を APIキーとして使用します
>        (長さ 902 文字 / JWT形式 (ドット区切り3パート)
>         → これは V1 のリフレッシュトークン/IDトークンの形式です。
>           V2 は APIキー方式に変更されました。)
> ```

### 2.2 GitHub Pages の有効化

`Settings > Pages > Source` を **GitHub Actions** に設定します。

公開URLは以下で確認できます。

| 場所 | 内容 |
| --- | --- |
| `Settings > Pages` | 上部に「Your site is live at ...」として表示 |
| リポジトリ右サイドバーの **Environments > github-pages** | 最新デプロイと URL |
| `Deploy to GitHub Pages` ワークフローの実行結果 | `deploy` ジョブに URL が表示される |

> **`deploy` ジョブがログも出さず即座に失敗する場合**
>
> `build` は成功しているのに `deploy` だけが1秒ほどで失敗し、ログが取得できない
> （ランナーが割り当てられていない）ときは、Pages のリソース自体が
> 未初期化である可能性があります。
>
> `Settings > Pages > Source` を一度 **Deploy from a branch** に切り替えて保存し、
> そのうえで **GitHub Actions** に戻すと初期化され、デプロイが通るようになります。

### 2.3 初回データ取得

`Actions` タブ → `Fetch J-Quants Data` → `Run workflow`。

- `check_auth_only` にチェックを入れると、**認証疎通の確認だけ**を行います（初回の切り分けに便利）。
- `codes` に `7203 6758` のようにスペース区切りで入力すると、その銘柄だけを取得します。

実行後、取得結果のサマリー表がワークフローの Summary に出力されます。

### 2.4 分析対象銘柄の変更

`scripts/watchlist.json` を編集して `Fetch J-Quants Data` を再実行してください。
4桁コードでも5桁コードでも構いません（4桁は仕様書 §3.1 に従い自動で末尾に `0` を付加します）。

---

## 3. ローカル開発

```bash
npm install
npm run dev        # 開発サーバ (http://localhost:5173)
npm run build      # 本番ビルド -> dist/

npm test           # スコアリングエンジンの単体テスト (22件)
npm run test:py    # データパイプラインの単体テスト (20件)
node tests/smoke.mjs   # Chromium での実描画テスト (要 playwright)
```

ローカルで実データを取得する場合:

```bash
export JQUANTS_API='<APIキー>'
python3 scripts/jquants_data_fetcher.py --check-auth   # 疎通確認
python3 scripts/jquants_data_fetcher.py --codes 7203   # 単一銘柄
python3 scripts/jquants_data_fetcher.py                # watchlist 全件
```

---

## 4. スコアリング仕様

### 4.1 8軸

正規化関数 $S(x, min, max) = \mathrm{clamp}(0, 10, \frac{x - min}{max - min} \times 10)$

| # | 軸 | 指標 | ロジック | 満点条件 |
| :-: | --- | --- | --- | --- |
| 1 | EPS成長 | 直近四半期EPS成長率 | `S(x, 0, 50)` | +50% 以上 |
| 2 | 売上成長 | 直近四半期売上高成長率 | `S(x, 0, 40)` | +40% 以上 |
| 3 | 収益質 | ROE | `S(x, 5, 25)` | 25% 以上 |
| 4 | 利益率 | 営業利益率 | `S(x, 0, 20)` | 20% 以上 |
| 5 | テクニカル | 52週高値接近率 $R$ | $R\ge98$→10.0 ／ $R\ge90$→$8.0+\frac{R-90}{8}\times1.5$ ／ $R\ge80$→$6.0+\frac{R-80}{10}\times2.0$ | 98% 以上 |
| 6 | 出来高 | 出来高モメンタム × 機関参入度 | $\mathrm{clamp}(0,10,\ 5.0+\frac{Trend-100}{100}\times5.0)\times decay$ | 売買代金10億円以上かつ増 |
| 7 | 需給 | 信用倍率 $C$ | $C\le1$→10.0 ／ $C\le3$→$10.0-\frac{C-1}{2}\times3.0$ ／ $C\le10$→$7.0-\frac{C-3}{7}\times5.0$ | 1.0倍以下 |
| 8 | 進捗期待 | 決算進捗率 vs 経過基準 | $\mathrm{clamp}(0,10,\ 5.0+\frac{Progress - Quarter\times25}{2})$ | 計画を大幅超過 |

実装は [`src/lib/scoring.js`](src/lib/scoring.js)、検証は [`tests/scoring.test.js`](tests/scoring.test.js) にあります。

### 4.2 機関投資家参入度（売買代金 $V$ 億円 / 時価総額 $Cap$ 億円）

| 判定 | 条件 | 出来高軸の減衰率 |
| --- | --- | :-: |
| 流動性不足 (none) | $V < 1$ | 0.35 |
| 個人・小口主導 (low) | $1 \le V < 5$ | 0.65 |
| 機関参入圏内 (moderate) | $5 \le V < 10$ | 0.85 |
| 機関主導・強 (high) | $10 \le V < 30$ | 1.00 |
| 機関熱狂・Monster (mega) | $V \ge 30$ | 1.00 |
| 時価総額不足 (cap_low) | $Cap < 100$（流動性判定を上書き） | 0.70 |

### 4.3 株価ゾーン

`BREAKOUT` ($R\ge98$) / `HANDLE` ($90\le R<98$) / `BASE` ($80\le R<90$) / `CORRECTION` ($R<80$)

---

## 5. 元仕様書からの変更点・補完箇所

仕様書に記述がなく、実装上の判断で確定させた点を明示します。

| 箇所 | 仕様書の記述 | 本実装での扱い | 理由 |
| --- | --- | --- | --- |
| **欠測値の扱い** | 記述なし | **その軸のスコアを `null` とし、総合スコアの平均から除外**。UI に「n/8軸」を表示 | 取得できなかった指標を 0点 とすると「実測でゼロ」と区別がつかず、存在しない評価を作ってしまうため |
| 総合スコア | $\frac{1}{8}\sum Score_k$ | 上記のため**有効軸のみの平均**を主表示。8軸を0埋めした厳密値も併記 | 同上 |
| 軸5 テクニカル $R<80$ | 未定義 | $R/80 \times 6.0$ で線形外挿（$R=80$ で 6.0 と連続） | 境界での不連続を避けるため |
| 軸6 出来高 | 「出来高増減率を基本とし機関参入レベルに応じて減衰補正」 | 上表 4.1 / 4.2 の式で確定 | 満点条件「10億円以上かつ増」を満たすよう decay を設計 |
| 軸7 需給 $C>10$ | 未定義 | $\max(0,\ 2.0 - \frac{C-10}{10}\times2.0)$（20倍で 0） | 同上 |
| 機関判定 `cap_low` | 他の5段階と並列に列挙 | **流動性ティアを上書きする独立フラグ**として実装（流動性ティアも `liquidity` に保持） | 時価総額と売買代金は別軸の制約であり、UI で両方見えるほうが判断しやすいため |
| 営業利益率 | 「営業利益率」とのみ | **TTM（直近4四半期）**を優先、算出不可なら当期累計。どちらを使ったか UI に表示 | 単一四半期はノイズが大きいため |
| 前年同期比 | 記述なし | 前年同期の値が **0以下なら成長率を `null`** とする | 赤字→黒字転換を「+1000%」等と表示すると誤解を招くため |
| タイムマシーンの過去時点 | 「6ヶ月前 / 3ヶ月前 / 現在」 | **その日までに開示済みのデータのみ**で再計算（先読みなし） | 過去時点で実際に見えていた情報だけで判断を検証するため |
| ストーリータイムライン | 「決算発表、機関買い参入、新高値ブレイク等の定性イベント」 | **株価・出来高・決算開示から機械的に検出できるイベントのみ**を表示 | 定性的なストーリーを創作しないため。検出条件は各イベントに明記 |
| 売買代金 | $P \times Volume / 10^8$ | 仕様どおり算出。API の実績値 `TurnoverValue` も併記 | 仕様に忠実にしつつ、より正確な実績値も確認できるように |

---

## 6. データパイプラインの実装メモ

### J-Quants の決算データは「累計」である

`/fins/statements` は会計年度内の**累計値**を返します（2Q は上期累計）。
そのまま前年同期比を取ると誤った成長率になるため、
[`quarterize()`](scripts/jquants_data_fetcher.py) で同一会計年度内の連続する四半期を差分展開し、
**単一四半期の値**に変換してから前年の同一四半期と比較しています。

### 使用エンドポイント

| V2 エンドポイント | V1（旧） | 用途 |
| --- | --- | --- |
| `/equities/master` | `/listed/info` | 銘柄名・業種・市場区分 |
| `/equities/bars/daily` | `/prices/daily_quotes` | 株価・出来高・売買代金 |
| `/fins/summary` | `/fins/statements` | EPS・売上・ROE・進捗率・発行済株式数 |
| `/markets/margin-interest` | `/markets/weekly_margin_interest` | 信用倍率（需給軸） |

ベースURL は `https://api.jquants.com/v2`、認証は全エンドポイント共通で `x-api-key` ヘッダーです。

**V2 での主な差分**

- レスポンスのデータ配列キーが一律 **`data`** になりました（V1 は `info` / `daily_quotes` 等）。
- 列名が短縮されました。対応は `scripts/jquants_data_fetcher.py` 冒頭の **FIELD MAP** 節に全件記載しています。

  | V1 | V2 | | V1 | V2 |
  | --- | --- | --- | --- | --- |
  | `Close` | `C` | | `NetSales` | `Sales` |
  | `Open` / `High` / `Low` | `O` / `H` / `L` | | `OperatingProfit` | `OP` |
  | `Volume` | `Vo` | | `Profit` | `NP` |
  | `TurnoverValue` | `Va` | | `Equity` | `Eq` |
  | `AdjustmentClose` | `AdjC` | | `DisclosedDate` | `DiscDate` |
  | `CompanyName` | `CoName` | | `TypeOfCurrentPeriod` | `CurPerType` |

- **ROE が API から直接提供される**ようになりました（`/fins/summary` の `ROE` 列）。
  提供値があればそれを使い、無い場合のみ「TTM純利益 ÷ 自己資本」で算出します。
  どちらを使ったかは UI の「基本指標」に表示されます。
- V1 の `TypeOfDocument` による実績決算の判別は、V2 の `DocType` の列挙値が
  公式クライアントのソースから確認できなかったため採用していません。代わりに
  **「`CurPerType` が 1Q〜FY のいずれか」かつ「売上・営業利益・純利益・EPS のいずれかに実績値がある」**
  で判定しています（業績予想の修正のみの開示は実績値を持たないため除外される）。

プラン制約で取得できなかったエンドポイントは `stocks.json` の `unavailableEndpoints` に記録され、
ダッシュボード上部に警告として表示されます。該当する軸は「—」となり、スコアの平均から除外されます。

依存パッケージはなく Python 標準ライブラリのみで動作します。

---

## 7. 研究ライン — 会計フローグラフ（`research/accgraph/`）

決算（PL・BS・CF）を勘定科目のグラフとして表し、決算発表後の
**ベンチマーク控除後リターン**を3クラスに分類する研究ラインです。
ダッシュボードとは独立していて、生データ（`research/_data/*.parquet`）だけを共有します。

現在の到達点は**データ層とベースライン比較まで**で、GNN はまだ入っていません。
まずベースラインの表を埋め、GNN がそれを上回るかどうかで有効性を判断します。

### 取得できるデータの制約（実測）

| エンドポイント | 実測 | 影響 |
| --- | :-: | --- |
| `/fins/summary` | OK | PL・BS の集計値と CF 3区分が取れる（四半期） |
| `/fins/details` | **HTTP 403** | 減価償却費・運転資本・売上原価などの**内訳は取れない** |
| EDINET DB `financials` | OK | 上の内訳が取れる。ただし**年1回**（有価証券報告書） |

`CFO` / `CFI` / `CFF` の開示率は 1Q 9.9% / 2Q 76.0% / 3Q 8.2% / 通期 88.8%
（`docs/DATA_FIELDS.md` の実測）。つまり大半の企業は CF を**半期でしか出しません**。
そのため 1Q・3Q の CF ノードは欠測のままマスクし、2Q・通期は「半期ぶんを
1四半期あたりに直した値」として持ちます。詳細は
[`docs/ACCOUNTING_GRAPH.md`](docs/ACCOUNTING_GRAPH.md)。

### EDINET DB の明細ノード（29ノード / 37エッジ）

J-Quants だけでは組めなかった「税引前利益 → 減価償却費 → 運転資本 → 営業CF」の鎖は、
EDINET DB の有価証券報告書から明細ノードを下位層として足すことで繋がります。

| 足したノード | 元の項目 |
| --- | --- |
| 売上原価 / 販管費 / 研究開発費 | `cost_of_sales` / `sga` / `rnd_expenses` |
| 税引前利益 | `profit_before_tax` |
| 減価償却費 / 設備投資 | `depreciation` / `capex` |
| 棚卸資産 / 売上債権 / 仕入債務 / 運転資本 | `inventories` / `trade_receivables` / `trade_payables` |
| 有利子負債 | `ibd_current` + `ibd_noncurrent` |

**年1回しか出ない**ので、四半期のグラフに混ぜるときは次の約束を置いています。

1. 年次のフローは4で割って1四半期あたりに直す（`span` = 4）
2. その値が何年前の書類かを `age_years` としてノード特徴量に持たせる
3. 基準日から400日より古い書類しか無ければ、明細は全部欠測にする

各四半期の**開示日を基準に** as-of で引くので、過去の期のグラフに
その時点ではまだ出ていない有報が混ざることはありません。
EDINET が取れていない会社では明細ノードが欠測になるだけで、
J-Quants だけの粗いグラフはそのまま成立します。

### EDINET の明細が効くかを、薄めずに測る

EDINET は取得枠（月900リクエスト）の都合で全社ぶんは揃っていません。
全サンプルで測ると9割以上が欠測になり、効果が薄まって判定できません。

そこで **明細が揃ったサンプルだけに絞り、その群の中でノードを出し入れ**します。

| 特徴量セット | 中身 |
| --- | --- |
| `latest_jq` / `seq_jq` | J-Quants のノードのみ（EDINET のノードとエッジを完全に外す） |
| `latest` / `seq` | 全ノード（EDINET の明細を含む） |

母集団を変えずにノードだけ差し替えるので、**明細を足した効果だけ**が取り出せます。
母集団ごと変えて比べると、明細の効果と母集団の違いが混ざって分離できません。

`has_edinet` の判定は「要件の鎖（税引前利益 → 減価償却費 → 運転資本 → 営業CF）を
引けるか」です。研究開発費は業種によって、有利子負債は無借金の会社で
項目ごと省かれるので、それらの欠測では群から落としません。

出力は [`docs/ACCGRAPH_EDINET.md`](docs/ACCGRAPH_EDINET.md)。

```bash
python3 research/accgraph/evaluate.py --edinet-only \
  --feature-sets latest_jq latest seq_jq seq
```

### 明細ありの群で信号が消えた原因を切り分ける

明細ありの群だけで学習・評価すると、J-Quants のノードだけのベースラインでも
AUC が 0.50 前後に落ちました。原因の候補は「訓練件数が足りない」か
「この群がそもそも予測しにくい」かの2つです。

`diagnose.py` は **全体で学習し、群に分けて評価** します。学習データを全体に
戻せば件数の問題は消えるので、それでも明細ありの群だけ落ちるなら群の性質です。
流動性（20日平均売買代金）を明細ありの群に揃えた対照群も並べ、
区間は発表日単位のブートストラップで出します。判定基準は実行前に固定してあります。

| 判定 | 条件 |
| --- | --- |
| 件数の問題 | 明細ありの群の AUC の95%区間が 0.5 を上回り、対照群との差の95%区間が 0 をまたぐ |
| 群の性質 | 明細ありの群の AUC の95%区間が 0.5 をまたぎ、対照群との差の95%区間が負に収まる |
| 判定できない | それ以外 |

出力は [`docs/ACCGRAPH_DIAGNOSE.md`](docs/ACCGRAPH_DIAGNOSE.md)。
ワークフローはこの切り分けと EDINET 比較を本体評価（約140分）より先に回し、
結果を先にコミットします（手動実行で `stage: fast` を選ぶと本体評価を飛ばします）。

```bash
python3 research/accgraph/diagnose.py
```

### 明細の上積みを2段構えで測る

切り分けの結果（2026-09-22）、件数の不足と群の性質の**両方**が効いていました。
群だけで学習すると件数が足りず、全体に混ぜると明細は9割以上が欠測のまま学習されます。
そこで `increment.py` は役割を分けます。

| 段 | 学習に使う行 | 特徴量 | 学ぶもの |
| --- | --- | --- | --- |
| 1段目 | 全体（約5.7万件） | J-Quants のノードだけ（`latest_jq`） | 土台の予測 |
| 2段目 | 明細ありの群だけ | 明細の特徴量 | 1段目の予測を固定したまま足す補正 |

2段目は正則化を強めると補正がゼロに縮み、1段目の予測に戻ります。
補正の強さは訓練行の中だけの前向き検証で選び、テスト窓は見ません。
1段目は最初の2年から walk-forward で回し、2段目の訓練行にも
「その行より前だけで学習したモデルの予測」を付けます。

明細の効果は、同じ2段目で **明細を入れた版 − 入れない版** の AUC の差で測ります。
入れない版（切片だけ）は群に合わせた較正し直しで、その効果と明細の効果を分けるための対照です。

| 判定 | 条件（結果を見る前に固定） |
| --- | --- |
| 明細が効く | 差の95%区間が 0 を上回り、かつ偽の明細19回の差をすべて上回る |
| 明細が害になる | 差の95%区間が 0 を下回り、かつ偽の明細19回の差をすべて下回る |
| 差が見えない | それ以外 |

偽の明細は、明細の特徴量を明細ありの行の間で入れ替えたものです。
ブートストラップの区間はテスト行の引き直しだけで、補正の学習そのものの揺れを含みません。
合成データ（明細に情報が無い）で区間だけだと「効く」と出た例があったので、これで補っています。

出力は [`docs/ACCGRAPH_INCREMENT.md`](docs/ACCGRAPH_INCREMENT.md)。

```bash
python3 research/accgraph/increment.py
```

### Temporal GNN（GraphSAGE + GRU）

`gnn.py` は、四半期ごとの会計フローグラフ（J-Quants の18ノード・21エッジ）を
GraphSAGE でまとめ、過去8四半期ぶんを GRU でつないで3クラスを予測します。
明細の上積みが見つからなかったので、EDINET のノードは使いません。

ベースライン（`latest_jq` / `seq_jq` × logit / lgbm / mlp）と
**同じ母集団・同じ分割・同じ行**で比べ、勝ち負けの理由を分けるために切り離し版も回します。

| 版 | 中身 | 本体との差が表すもの |
| --- | --- | --- |
| `sage_gru` | 本体。エッジ特徴つき GraphSAGE 2層 → GRU | — |
| `mlp_gru` | ノードの間で情報をやりとりしない（エッジを外す） | エッジ（会計フロー）の効果 |
| `sage_latest` | 当該四半期のグラフだけ（時系列を外す） | 時系列の効果 |

各版を乱数3通りで学習し、予測確率を平均して評価します。
判定基準は実行前に固定してあります（区間は発表日単位のブートストラップの95%区間）。

| 比較 | 0 を上回る | 0 を下回る | 0 をまたぐ |
| --- | --- | --- | --- |
| `sage_gru` − 最良のベースライン | GNN が有効 | ベースラインのほうが良い | 差が見えない |
| `sage_gru` − `mlp_gru` | エッジが効いている | エッジが害になっている | エッジの効果は見えない |
| `sage_gru` − `sage_latest` | 時系列が効いている | 時系列が害になっている | 時系列の効果は見えない |

最良のベースラインは6本の中で AUC が最も高いもの（ベースラインに最も有利な選び方）です。
バックテストは参考として並べ、判定には使いません。

GNN は別のワークフロー（`accgraph-gnn.yml`、約2〜3.5時間）で回し、
結果を [`docs/ACCGRAPH_GNN.md`](docs/ACCGRAPH_GNN.md) にコミットします。
torch が要るので、通常の CI では GNN のテストは飛ばされます。

```bash
pip install torch                           # ローカルで回すとき
python3 research/accgraph/gnn.py baselines  # ベースラインの予測を保存
python3 research/accgraph/gnn.py train      # GNN の予測を保存
python3 research/accgraph/gnn.py report     # 比べて docs/ACCGRAPH_GNN.md
```

### 規模効果を抜いて測る

実データで測ると、単変量の情報係数の上位が軒並み `log_size`（企業規模）でした。
そこで **同じ発表日の中で各特徴量を順位に直したセット**（`latest_rank` / `seq_rank`）
を並べて比較します。その日の地合いと規模の絶対水準が消えるので、
会計構造そのものに情報があるかを分離して測れます。

### 使い方

```bash
pip install -r research/requirements.txt

python3 research/accgraph/schema.py      # スキーマの要約を見る
python3 research/accgraph/docgen.py      # docs/ACCOUNTING_GRAPH.md を作り直す
python3 research/accgraph/build.py       # データセットを作る (要 research/_data)
python3 research/accgraph/eda.py         # EDA の集計 -> _data/accgraph/eda.json
python3 research/accgraph/eda_report.py  # 集計を組版 -> docs/accgraph_eda.html
python3 research/accgraph/evaluate.py    # ベースラインを比較して docs に書き出す
python3 research/accgraph/diagnose.py    # 明細ありの群の原因切り分け -> docs
python3 research/accgraph/increment.py   # 明細の上積みを2段構えで測る -> docs
python3 research/accgraph/gnn.py all     # GNN とベースラインを比べる (要 torch) -> docs
python3 tests/test_accgraph.py           # 単体テスト
```

### EDA（探索的データ解析）

`docs/accgraph_eda.html` は単体で開ける HTML で、次の4点を見ます。

| 節 | 見るもの |
| --- | --- |
| 何が取れて、何が取れないか | ノード別の充足率を四半期・年で。CF の半期開示がどこまで効くか |
| 目的変数の分布と偏り | 超過リターンの分布とクラス比が、年・四半期・業種・流動性・決算の混雑度でどう変わるか |
| 特徴量の診断 | 欠損・打ち切り・定数列・冗長な組み合わせ・分布 |
| 単変量の情報量 | 各特徴量と超過リターンの順位相関。材料がそもそもあるか |
| グラフ構造の診断 | CF の符号の型、営業CF÷営業利益、エッジごとの比率と符号一致 |

集計（`eda.py`）と組版（`eda_report.py`）を分けてあります。データセットは
CI 側にしか無いので、図を直すたびに再集計が要る作りにしないためです。

生データが手元に無い場合は、GitHub Actions の `Accounting Graph Baseline`
ワークフローを実行してください（Release のタグ `data-raw` から生データを取ります）。

### 設計上、必ず守っていること

| 事故 | 対策 | 検査 |
| --- | --- | --- |
| 過去の期に後日の訂正値が混ざる | 各決算の発表日を基準に as-of 結合で版を選ぶ | `leakage.check_asof` |
| 発表前の株価でエントリーする | 一律で**翌営業日の始値**を起点にする | `leakage.check_labels` |
| ラベルがテスト期間の価格で決まる | Purge（ラベル確定日）＋ Embargo（緩衝期間） | `splits.walk_forward` |
| 上場廃止銘柄が黙って消える | 除外理由と件数を毎回出す | `labels.report_universe` |
| 株式分割でリターンが壊れる | 調整後価格で測る | `tests/test_accgraph.py` |
| 欠測を 0 と取り違える | `is_missing` を立てたうえで 0 を置く | `leakage.check_features` |

効率的市場仮説の下では、この種の予測が安定して当たるとは想定していません。
Accuracy 55% を大きく超える行が出たら、まずリークを疑ってください。

---

## 8. ディレクトリ構成

```
├── .github/workflows/
│   ├── fetch-data.yml       # J-Quants V2 データ取得 (secrets.JQUANTS_API)
│   ├── deploy-pages.yml     # GitHub Pages へのビルド & デプロイ
│   ├── accgraph.yml         # 会計フローグラフのデータ構築 + ベースライン比較
│   ├── accgraph-gnn.yml     # 会計フローグラフの Temporal GNN
│   └── ci.yml               # テスト + ビルド + ブラウザ描画テスト
├── scripts/
│   ├── jquants_data_fetcher.py   # データ取得・指標算出パイプライン
│   └── watchlist.json            # 分析対象銘柄 (編集して再実行)
├── research/accgraph/            # 会計フローグラフ研究ライン
│   ├── schema.py                 # ノード・エッジ定義（階層的標準化つき）
│   ├── panel.py                  # 発表時点で見えていた値に組み直す (as-of)
│   ├── labels.py                 # 翌営業日始値起点の超過リターン・3クラス
│   ├── build.py                  # グラフ系列データセットの構築
│   ├── splits.py                 # Purged / Embargo つき時系列分割
│   ├── baselines.py              # ロジスティック回帰 / LightGBM / MLP
│   ├── backtest.py               # 取引コスト控除後の損益
│   ├── evaluate.py               # 評価の入口 (CLI)
│   ├── diagnose.py               # 明細ありの群で信号が消えた原因の切り分け
│   ├── increment.py              # 明細の上積みを2段構えで測る
│   ├── gnn.py                    # Temporal GNN（GraphSAGE + GRU）と切り離し版
│   ├── leakage.py                # リーク検査
│   ├── edinet.py                 # EDINET DB の明細を as-of で結合
│   ├── eda.py                    # EDA の集計 (JSON)
│   ├── eda_report.py             # EDA の組版 (単体HTML)
│   ├── synthetic.py              # テスト用の決定的な合成データ
│   └── docgen.py                 # docs/ACCOUNTING_GRAPH.md の生成
├── public/data/stocks.json       # 生成データ (ワークフローが上書き)
├── src/
│   ├── lib/scoring.js            # 8軸スコアリングエンジン
│   ├── lib/store.js              # データロード / localStorage 永続化
│   ├── lib/format.js             # 表示フォーマット
│   ├── components/               # レーダー・軸別表・銘柄カード・追加モーダル
│   └── views/                    # 3つの View
└── tests/
    ├── scoring.test.js           # スコアリング単体テスト
    ├── test_fetcher.py           # パイプライン単体テスト
    ├── test_accgraph.py          # 会計フローグラフ（リーク検査・ラベル定義）
    ├── smoke.mjs                 # Chromium 実描画テスト
    └── fixtures/                 # UI 検証用の合成データ
```

---

## 9. 免責

本ツールは投資判断の**支援**を目的としたものであり、投資勧誘・投資助言を行うものではありません。
スコアはあくまで公開データを機械的に加工した指標であり、将来の価格を予測するものではありません。
最終的な投資判断はご自身の責任で行ってください。

データ提供: [J-Quants API](https://jpx-jquants.com/)（株式会社JPX総研）
