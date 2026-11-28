# CAPM・理論株価特徴量の初期検証

> この文書はE120の固定パラメータ試験の記録。同条件で再チューニングした [E121の結果](MODEL_THEORETICAL_VALUE_RETUNED.md) を追加した（2026-11-04）。分類への一括追加の優位性は再確認できず、本番は据え置き。

2026-11-03。検証用の明示的な呼び出しだけを実装。本番の特徴量360列、パラメータ、モデル、契約プランは変更していない。

## 算出方法

`research/theoretical_value.py` は、CAPMによる株主資本コストを残余利益モデル（RI）と配当割引モデル（DDM）に適用する。CAPM単独から株価は求まらない。

円建て10年国債の年率実効利回りをrfの代理とし、`ke=rf+β×ERP`。財務省の半年複利利回りを `(1+y/200)^2−1` で変換し、負金利も残す。参照日は候補日より前、最大7暦日の古さまでとする。財務省の翌営業日9:30公表を踏まえた制限である。

βはTOPIXと分割調整株価の過去504市場営業日、最低252有効組でOLS推定する。配当を含まない価格収益率による近似。国債利回りを252乗根で日率化して超過収益を作り、両系列が有効な同じ日だけを使う。欠損日を埋めず、多日株式収益と1日市場収益を混ぜない。今回の事前仮定としてβを `2/3×OLSβ+1/3` に縮約し、負値をゼロに丸めない。

- **RIの30シナリオ**：ERP 3/4/5/6/7%、収束期間H=5/10年、配当性向25/50/75%。初期ROE=TTM純利益/直近自己資本。`ROEt=ke+(ROE0−ke)×(H−t)/H`、`Et=ROEt×B(t−1)`、`Bt=B(t−1)+Et×(1−配当性向)`、`V=B0+Σ[(Et−ke×B(t−1))/(1+ke)^t]`。H年目以降の超過利益はゼロ、H年目の価値は簿価。黒字・正の自己資本・正のkeが必要。
- **DDMの15シナリオ**：同じERP5通り、永久成長率g=0/1/2%。`V=D0×(1+g)/(ke−g)`。D0は最新開示済み通期実績の配当総額/現在株数で、予想配当やTTM配当ではない。通期末から550暦日以内、現在TTM黒字、通期実績配当性向0%超100%以下、`ke−g≥1%` に限定する。無配・欠損を価値ゼロにしない。

各方式で正かつ有限な価格のシナリオが80%以上ある行だけ、最小・最大・等重み平均・中央値・母標準偏差（ddof=0）を出す。この幅は**仮定の感応度であり、信頼区間・確率分布ではない**。β・財務推定の誤差を網羅した値域でもない。両方式の価格は混ぜない。

円価格 `tv_{ri,ddm}_{min,max,mean,median,std}` を保存する。学習候補は価格統計/当日未調整終値−1、標準偏差だけは標準偏差/終値の10列（末尾`_rel`）。元の株価水準や分割単位の影響を避けるためである。

## 財務入力と時点

保存済みJ-Quants開示から、候補日前日以前に公表されたその時点の最新の版を選ぶ。TTM純利益は当期累計＋前通期−前年同期累計、通常年度の通期はそのまま使う。連結・非連結を混ぜず、年度と前年同期の期間一致を確認する。最新開示の欠損を古い完全な開示に戻さない。

自己資本はShEq、株数はShOutFY−TrShFY。不明な自己株数をゼロとしない。期末から候補日への分割倍率の比で株数単位を合わせ、1株EPSの4期和の単位混在を避ける。開示から365日超の財務は使わない。追加特徴量は候補日の同日引け後決算を使わない保守的な定義であり、既存360列の定義とは独立している。

原APIの過去値書き換えまで独立に監査・復元できたわけではない。会計基準変更時の前年数値の遡及組替えも未実施。クリーンサープラス、ROEの収束、一定配当性向、永久成長は企業別予想ではなく固定の研究仮定。OCI・新株発行・自社株買い、特別配当、安定期以前の高成長を十分に表現していない。

## バックテストの設計

`research/exp/e120_theoretical_value.py`。+10%分類、+15%分類、日次LambdaRank、直近5営業日LambdaRankの4タスクを、各条件・種42/43/44、木200本で比較する。

| 条件 | 内容 |
|---|---|
| A | 既存360列、LTR5は値動き4列を追加 |
| C | A＋元財務比率・β・金利・利用可能性13列 |
| R | C＋RIの5統計 |
| D | C＋DDMの5統計 |
| B | C＋両方式10統計 |
| P | C＋日付・利用可能パターン内で10値を一緒に入れ替えた対照 |

主比較はB−AとB−C。CはEPS/BPS/実績配当の株価比、実績配当性向、β、観測数、金利・その古さ、財務の古さ、有効シナリオ率、方式ごとの欠損フラグを含む。入力の組み直しと価値変換を区別するためである。Pは各方式の統計の対応と欠損位置を保つ。これは有意差検定ではない。

既存パラメータを固定し、再チューニングはしない。分類ラベルの境界許容差、訓練負例/正例比の補正、ランキングの5段階ラベル・線形gainは既存処理を再利用する。LTR5の新規指標も候補日の値で凍結する。

各会社の日足で目的変数の確定日を求め、パラメータ探索の訓練行も含めて評価開始より前に結果が確定するよう分離する。Aは同じマートで再学習した既存仕様の参照で、保存済み本番モデルの評価ではない。4タスク×6条件×3種=72 fits。分類はAP、ランキングは日付内NDCG@5。

追加の単純検証は理論価格中央値/株価−1の日付内5分位と20/60/120日先の単純収益。満期未到達を除くため長期ほど対象件数・期間が異なる。収益は配当・費用・同時保有制約・約定制約を含まず、観測窓も重複している。将来の株価が理論レンジに到達する精度を測るものではない。

## 実行

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python research/exp/e120_theoretical_value.py \
  --dataset /path/to/dataset.parquet \
  --prices /path/to/panel_bounded.parquet \
  --data-dir /path/to/raw \
  --yields /path/to/jgbcme_all.csv \
  --output /path/to/output

JUPYTER_RUNTIME_DIR=/tmp/theoretical-value-jupyter IPYTHONDIR=/tmp/theoretical-value-ipython \
  python research/exp/e120_report.py /path/to/output
```

`--prepare-only`で価格・算出率だけを作れる。入力・実装・既存パラメータ・ライブラリ版の指紋が一致したときだけ予測キャッシュを再利用する。Notebook用の追加依存はnbformat、nbclient、nbconvert、beautifulsoup4とPythonカーネル。実行済みNotebookをHTML変換し、JavaScript・外部読込を除いて集計を静的表示する。

結果・入力指紋・検証記録は `docs/experiments/theoretical-value-2026-11-03/`。個別の金融値・行別予測・金融値入りHTMLは公開Gitに含めず、ユーザー向けの別アーティファクトに保存する。

本期間は以前の研究で参照済みの単一ホールドアウト。3乱数種は独立した評価期間ではない。複数条件を試した結果の小幅改善を採用根拠にせず、新たな期間での再検証が必要。XGBoost・CatBoostやWACC/FCFFの企業価値DCFは今回未検証。

## 一次資料

- [財務省：金利の定義と公表時刻](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm)
- [財務省：国債金利CSV](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv)
- [NYU Damodaran：CAPM・割引率](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/dcfinput.html)
- [CFA Institute：残余利益モデル](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/residual-income-valuation)
- [NYU Damodaran：配当割引モデル](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/ddm.html)
