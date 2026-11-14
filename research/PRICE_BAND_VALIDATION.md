# 価格帯別需給：平均建値の補助検証

ユーザーが確定した条件で、出来高重みと回転率による指数減衰を比較する研究用コードです。
通常の取得処理・学習・推論・需給表示・Pages公開を変更しません。

## 確定条件

- 東証上場の普通株。信用買い・信用売りの両方。日付時点の上場銘柄一覧を使い、現在の上場銘柄だけに限定しない。
- 商品区分011・021・024のうち、証券コード5桁目が0の普通株。ETF・REIT・優先株等を除く。商品区分が不明な銘柄日は別途記録する。
- 両手法とも60営業日固定、日次VWAP＝売買代金÷出来高、初期重みは出来高。
- 回転率＝出来高÷（発行済株式数−自己株式数）。各日23:59日本時間までの有効な同一開示の株数一式を使う。
  欠損時は過去の有効な一式を引き継ぎ、なければ計算不可。自己株式を0で補完しない。
- 株数の測定基準日はCurPerEn。測定日から観測日までの分割・併合を調整する。
  価格・出来高も残高基準日の株数単位に揃える。必要な古い分割履歴は既存キャッシュから追加で読む。
- 日iの重み＝`V_i × exp(-k × Σ(r_j, i < j <= T))`。買い・売り、全銘柄で共通のk。
- k＝0〜3を0.1刻みの31候補。主指標RMSPEを最小にする係数を調整期間だけで選ぶ。同点は最小k。
- レコード（銘柄×基準日×買い売り）を均等に集計。金額加重や買い売り50%ずつにはしない。
- 正解平均＝公表約定金額÷公表残高株数。約定金額・正解平均を特徴量や推定の平均合わせに使わない。
- 2026-09-25から取得時点の最新公表済み日。基準日を丸ごと時系列分割し前70%で調整、後30%は一度だけ最終評価。
- 両手法の共通算出可能レコードで比較。残高0、履歴不足、欠損等の理由と算出率を保存。
  公表平均が60日価格範囲外でも、有効なら誤差集計に含める。
- ペアの5営業日移動ブロック・ブートストラップ1000回、seed=20261114、95%参考区間。
  係数をブートストラップ内で再調整しない。短い評価期間での参考値とする。
- 探索的な比較。本番への自動適用、評価後の追加探索、追加の正解分布探索は実施しない。

## 実装

| ファイル | 役割 |
|---|---|
| `price_band_validation_config.json` | 承認済みの設定 |
| `collect_price_band_validation_data.py` | 金額・公表日付き信用残、日付別銘柄情報、取引カレンダーを取得 |
| `price_band_validation.py` | 時点整合した株数、調整、平均推定、係数選択、区間計算 |
| `run_price_band_validation.py` | 入力固定・ハッシュ、除外理由、調整、最終評価の一度限りの実行 |
| `report_price_band_validation.py` | 確定結果だけを読むNotebookを実行し、表とPNGを埋め込んだHTMLに変換 |

`selection.json`は最終評価の誤差計算前に書き出します。最終評価について31候補の誤差を作って選び直す処理はありません。
`run_price_band_validation.py`は出力先が空でなければ停止します。レポートの再生成は評価をやり直しません。

## 取得と実行

認証情報は既存のActions Secret `JQUANTS_API`だけを利用できます。キーの新規発行・追加購入は不要です。
`preview-supply-demand.yml`の研究ブランチ用ジョブは`contents: read`で、Releaseや公開データを書き換えません。
`codex/price-band-demand-validation-*`では通常の表示用プレビューをスキップします。
このワークフロー自身の変更を含む研究ブランチへのpush、またはworkflow_dispatchで取得できます。

```bash
gh workflow run preview-supply-demand.yml --ref codex/price-band-demand-validation-20261114
gh run download RUN_ID --name price-band-validation-source --dir research/_data/validation-source
```

取得物の保存期間は7日です。取得失敗時にHTTP認証本文等をログへ流さず、例外型と一般的なエラーだけを返します。
中間営業日が欠けた場合は停止し、末尾の未公表日だけを記録して除外します。

依存関係は研究用環境へインストールしてください。

```bash
python -m pip install -r research/price_band_validation_requirements.txt
python -m unittest discover -s tests -p 'test_price_band_validation*.py'
```

既存`data-raw` Releaseから取得した株価・財務・カレンダーを使用します。
`CURRENT_RAW`には`bars_2025.parquet`、`bars_2026.parquet`、最新`fins_2026.parquet`を配置します。
`OLDER_RAW`には既存の`fins_2016.parquet`〜`fins_2025.parquet`、`calendar.parquet`、必要な過去年の`bars_*.parquet`を置きます。
古い株数を引き継ぐ場合の分割調整のため、本実行では2017年以降のバーも一部銘柄について参照しました。
`research/_data/`以下を使うと入力・銘柄別監査データを誤ってコミットせずに済みます。

```bash
python research/run_price_band_validation.py \
  --raw-dir CURRENT_RAW \
  --older-raw-dir OLDER_RAW \
  --source-dir research/_data/validation-source \
  --output-dir research/_data/price-band-results-new
python research/report_price_band_validation.py \
  --result-dir research/_data/price-band-results-new
```

`summary.csv`、`coverage.csv`、`scope_coverage.csv`、`training_curve.csv`、`daily_metrics.csv`、
`summary.json`、`source_manifest.json`、実行済みNotebookと静的HTML等を保存します。
銘柄別監査Parquetはローカルに保持し、本リポジトリには集計・設定・Notebook・HTMLを残します。
Notebookを再実行する場合は同じフォルダのCSV・JSONを保持してください。

## 解釈上の制約

日本株市場全体の価格帯別残存信用建玉の正解分布は未入手です。
この検証の正解は平均建値だけで、**分布形状・価格帯残高の精度検証とは異なります**。
本番の平均合わせを使った分布を正解に見せる検証も行っていません。

市場出来高は信用新規・返済のフローではなく、60日より古い建玉や日中の価格分布も表現できません。
公表平均が60日価格範囲外なら、非負の重みを使う両手法では構造的にその平均を再現できません。
株数の過去引継ぎでは、新株発行・自己株取得等の完全な履歴を復元できない点も残ります。
買い売り共通係数のため同一銘柄・同一日の予測平均は同一で、実際の買い売りの違いは表現できません。
現在の過去データには後日訂正が反映される可能性があり、当時のデータ版の厳密な再現ではありません。

実行結果は[2026-11-14の記録](../docs/experiments/price-band-validation-2026-11-14/README.md)を参照してください。
