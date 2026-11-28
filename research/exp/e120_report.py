#!/usr/bin/env python3
"""Execute a self-contained notebook report and export static, offline HTML.

Run after e120_theoretical_value.py with OUTPUT_DIRECTORY as the sole argument.
The source dataset and row-level feature export remain private artifacts.
"""
from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

P=Path(sys.argv[1]).resolve()
d=pd.read_parquet(P/'dataset.parquet')
values=[c for c in d if c.startswith(('tv_ri_','tv_ddm_')) and not c.endswith('valid_fraction')]
stats=d[values].describe(percentiles=[.05,.5,.95]).T.reset_index(names='feature')
stats['unique_values']=stats.feature.map({c:d[c].nunique() for c in values})
stats['missing_pct']=stats.feature.map({c:100*d[c].isna().mean() for c in values})
stats.to_csv(P/'feature_summary.csv',index=False)
coverage=[]
for year,g in d.groupby(d.Date.dt.year):
    coverage.append({'year':year,'rows':len(g),'RI_rows':int(g.tv_ri_mean.notna().sum()),
                     'RI_pct':100*g.tv_ri_mean.notna().mean(),'DDM_rows':int(g.tv_ddm_mean.notna().sum()),
                     'DDM_pct':100*g.tv_ddm_mean.notna().mean()})
pd.DataFrame(coverage).to_csv(P/'coverage_by_year.csv',index=False)
qa=[]
for label,mask in [('β未算出',d.tv_beta.isna()),('財務EPS未算出',d.tv_eps.isna()),
                   ('BPS欠損または非正',d.tv_bps.isna()|d.tv_bps.le(0)),('TTM利益非正',d.tv_eps.le(0)),
                   ('実績配当未算出または非正',d.tv_div0.isna()|d.tv_div0.le(0)),
                   ('配当性向欠損・非正・100%超',d.tv_payout.isna()|d.tv_payout.le(0)|d.tv_payout.gt(1)),
                   ('初期ROEが100%超',d.tv_eps.div(d.tv_bps).gt(1)),
                   ('RI有効シナリオ80%未満',d.tv_ri_valid_fraction.lt(.8)),
                   ('DDM有効シナリオ80%未満',d.tv_ddm_valid_fraction.lt(.8))]:
    qa.append({'check':label,'rows':int(mask.sum()),'pct':100*mask.mean()})
pd.DataFrame(qa).to_csv(P/'input_quality.csv',index=False)
examplecols=['Code','Date','close_raw','tv_beta','tv_rf']+[f'tv_{m}_{s}' for m in ['ri','ddm'] for s in ['min','max','mean','median','std']]
example=d[d.Date.eq(d.Date.max()) & d.tv_ri_mean.notna()].sort_values('Code').head(10)[examplecols]
example.to_csv(P/'examples.csv',index=False)
featurecols=['Code','Date','close_raw']+[c for c in d if c.startswith('tv_')]
d[featurecols].to_csv(P/'theoretical_value_features.csv.gz',index=False,compression='gzip')

nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'name':'python3','language':'python','display_name':'Python 3'}
def md(s):return nbformat.v4.new_markdown_cell(s)
def code(s):return nbformat.v4.new_code_cell(s)
nb.cells=[md('''# CAPMを用いた理論株価：算出可否と追加効果の初期バックテスト

2026-11-03。本番モデル・特徴量リスト・パラメータ・契約プランは変更していません。
残余利益（RI）と配当割引（DDM）を別々に算出します。両方式の価格を混ぜた平均は作りません。
**下限・上限・平均・中央値・標準偏差は、仮定を変えたシナリオ集合の統計量です。信頼区間や将来株価の確率分布ではありません。**

## 算出率と検証結果
'''),code('''from pathlib import Path
import json,pandas as pd,numpy as np
from IPython.display import display,HTML
P=Path.cwd()
def table(df):display(HTML(df.to_html(index=False,max_rows=None,float_format=lambda x:f'{x:,.5f}')))
cov=pd.read_csv(P/'coverage.csv');res=pd.read_csv(P/'results.csv')
plan=json.loads((P/'experiment_plan.json').read_text())
table(cov[cov.feature.isin(['tv_beta','tv_ri_mean_rel','tv_ddm_mean_rel'])])
names={'class10':'分類 +10%','class15':'分類 +15%','ltr':'日次ランキング','ltr5':'直近5営業日ランキング'}
allres=res[res.subset.eq('all')]
w=allres.pivot(index='task',columns='arm',values='primary').reindex(names)
summary=w.assign(B_minus_A=w.B-w.A,B_minus_C=w.B-w.C,R_minus_C=w.R-w.C,D_minus_C=w.D-w.C,B_minus_P=w.B-w.P).reset_index()
summary['task']=summary.task.map(names)
table(summary)
summary.to_csv(P/'comparison_summary.csv',index=False)
'''),md('''分類の主指標はAverage Precision（AP）、ランキングは日ごとのNDCG@5です。各条件で3種の予測を平均してから算出しています。
差が正なら主指標が改善しています。異なるタスクの指標の大小は比較できません。
Aは既存特徴量・既存パラメータで再学習した参照モデルで、保存済み本番モデルそのものの評価ではありません。

**B−Cが理論株価への変換を加えた差です。** Cには今回作った財務比率、β・金利、欠損・利用可能性を含めます。
B−Aだけでは、財務入力の組み直しや新たなβの効果と理論株価への変換効果を区別できません。
Pは同じ日・同じ欠損パターンの行間で10値を一緒に入れ替えた対照です。上下限や標準偏差の対応と欠損位置を保ちます。
これは有意差検定ではありません。

## 計算方法と仮定

CAPMは株主資本コストを算出するモデルです。株価を求めるには、利益・配当などの仮定を別途置きます。

- 株主資本コスト `ke = rf + β × ERP`。円建て10年国債利回りをrfの代理とし、ERPを3/4/5/6/7%で変化させます。
- 財務省の利回りは半年複利表示です。年率実効値 `(1 + 利回り%/200)² − 1` に直し、負金利をそのまま残します。
- 財務省は翌営業日9:30公表としているため、評価日の国債利回りを使わず、前日以前の参照日を使います。7暦日超の空白は利用しません。
- βはTOPIXとの過去504市場営業日の超過収益OLS、最低252組の有効観測。株価は分割調整済み、配当込みではありません。
  国債利回りの年率実効値を252乗根で日率化して控除します。長期国債を日次安全資産にも用いる近似です。
  欠損を前値補完せず、複数日株式リターンと1日指数リターンを混在させません。
  βは今回の事前仮定として `2/3 × 推定β + 1/3` と縮約します。負のβをゼロに丸めません。

### 残余利益（RI）：30シナリオ

`V = B0 + Σ[(Et − ke × B(t−1)) / (1+ke)^t]`。
B0は自己資本/現在株数、初期ROEはTTM純利益/直近自己資本（EPS_TTM/BPS）です。平均自己資本ROEとは異なる近似です。
ROEを5年または10年でkeへ直線的に収束させ、各年 `ROEt = ke + (初期ROE−ke) × (H−t)/H`。
配当性向25/50/75%で、`Et=ROEt×B(t−1)`、`Bt=B(t−1)+Et×(1−配当性向)` と進めます。
H年目以降の超過利益をゼロ、H年目の株式価値を簿価と置きます。ERP5通り×期間2通り×配当性向3通りです。
黒字・正の自己資本・正のkeだけを対象とし、赤字企業の回復を勝手に仮定しません。
継続成長・競争優位・クリーンサープラスは仮定です。OCI、自社株買い、新株発行等を個別予測していません。

### 配当割引（DDM）：15シナリオ

`V = D0 × (1+g) / (ke−g)`。ERP5通り×永久成長率0/1/2%です。
D0は最新の開示済み通期実績の年間配当総額/現在株数で、会社予想DPSやTTM配当ではありません。
通期末から550暦日以内、現在TTM黒字、実績配当性向が0%超100%以下に限定します。
`ke−g≥1%` を満たさないシナリオは除外します。無配企業や不適格企業の価値をゼロとはしません。
配当が安定して将来の分配可能額を代表するという強い近似で、特別配当・自社株買い・高成長期は十分に表現できません。

各方式で正かつ有限な価格のシナリオが80%以上ある行だけ、下限・上限・等重み平均・中央値・母標準偏差（ddof=0）を出します。
シナリオの削除がある行の下限・上限は**有効な仮定の中の最小・最大**です。β・財務推定の誤差まで含む完全な値域ではありません。
学習には各価格統計/候補日の未調整終値−1、標準偏差だけは標準偏差/終値を使います。円価格も保存しています。
円価格そのものを学習させることによる株価水準・分割単位の影響を避けるためです。

## 財務データの時点と単位

J-Quantsの保存済み四半期開示を使います。候補日の前日以前に開示された版だけを、その時点ごとに選び直します。
同日引け後の決算を翌日の情報へ誤って持ち込まないための保守的な扱いです。既存360特徴量の定義は変えません。
TTM純利益は当期累計＋前通期−前年同期累計。通期は通常の約1年間の純利益を使います。
連結・非連結を混ぜず、通常年度と前年同期の期間一致を確認します。欠けた最新開示を古い完全な開示へ戻しません。
自己資本はShEq、自己株控除後株数はShOutFY−TrShFY。自己株数不明はゼロとせず欠損です。
期末から候補日までの株数単位を分割倍率の比で調整します。EPS単純4期和の分割単位混在を避けます。
開示から365日超の財務を使わず、欠損を同業平均や未来値で埋めません。
原データ提供元が後日書き換えた過去値まで独立に復元・監査できたことは意味しません。会計基準変更時の前年数値の遡及組替えも未実施です。

## カバレッジと品質

分母は欠損行も含む学習マート全行です。下表の欠損理由は重複します。
'''),code('''table(pd.read_csv(P/'coverage_by_year.csv'))
table(pd.read_csv(P/'input_quality.csv'))
'''),md('''## 追加特徴量の欠損・カーディナリティ・要約統計

`tv_ri_*` / `tv_ddm_*` は円、末尾`_rel`は上記の株価相対値です。極端値を事後クリップしていません。
'''),code('''s=pd.read_csv(P/'feature_summary.csv')
table(s[['feature','count','unique_values','missing_pct','min','5%','50%','95%','max']])
'''),md('''## 理論価格の具体例（推奨銘柄ではなく計算出力）

最新の候補日でRIを算出できたコード順の先頭10行。成績で選んでいません。予測の根拠は上記の固定シナリオです。
'''),code("table(pd.read_csv(P/'examples.csv',dtype={'Code':str}))"),md('''## 比較条件と訓練・評価分離

| 条件 | 内容 |
|---|---|
| A | 既存360列（LTR5は値動き4列を追加） |
| C | A＋元財務比率・β・金利・観測数・古さ・シナリオ有効率・方式ごとの欠損フラグ13列 |
| R | C＋RIの価格相対5統計 |
| D | C＋DDMの価格相対5統計 |
| B | C＋両方式の10統計 |
| P | C＋日付・利用可能パターン内で一緒に入れ替えた10統計 |

各条件で同じ行・既存パラメータ・木200本・乱数種42/43/44。再チューニングはしていません。
分類は本番と同じ訓練負例数/正例数による不均衡補正、ランキングは本番の5段階ラベル・線形gainを使います。
分類目的変数は20日以内+10%かつ終盤+5%以上、または+15%かつ終盤+7.5%以上で、20日目MA5≥MA20も必要です。
LTRは翌営業日寄り→20日目終値の収益を日付内で順位化します。
LTR5の既存・新規特徴量は候補日の値で凍結し、判断日の財務へ更新しません。
各会社の実際の日足で目的変数の確定日を求め、既存パラメータ探索に関係する全訓練行の結果が確定してから評価を開始します。
'''),code('''table(pd.DataFrame(json.loads((P/'split_checks.json').read_text())))
table(pd.DataFrame([{'item':k,'value':plan[k]} for k in ['train_end','latest_parameter_training_outcome','test_start','test_end','seeds','re_tuned','rolling_feature_date']]))
'''),md('''## モデル比較の全指標

収益は分割調整価格の単純収益で、配当・コストは含みません。ランキング上位5件は候補5件以上の日を日付等重みで集計。
分類上位10%は期間全体でスコア上位10%です。同時保有、資金配分、回転、約定可能性を再現した運用損益ではありません。
'''),code('table(allres)'),md('''## 算出できた評価行だけの比較

各条件で同じ部分集合です。全行での主比較を置き換えません。ランキングの部分集合では、その集合内で収益の順位ラベルを作り直します。
'''),code("table(res[~res.subset.eq('all')])"),md('''## 四半期別・乱数種別の安定性

四半期比較は100行以上・20日以上の窓だけ。乱数種は評価期間を増やしたことにはなりません。
'''),code('''for name,keys in [('quarter_results.csv',['task','quarter']),('seed_results.csv',['task','seed'])]:
    z=pd.read_csv(P/name).pivot(index=keys,columns='arm',values='primary')
    table(z.assign(B_minus_A=z.B-z.A,B_minus_C=z.B-z.C,R_minus_C=z.R-z.C,D_minus_C=z.D-z.C).reset_index())
'''),md('''## 理論株価そのものの順位と先の収益

理論価格中央値/当日終値−1を日付内で5分位に分けます。1は低い側、5は高い側。
候補5件以上・異なる評価値2つ以上の日だけを対象にし、同値は平均順位で同じ区分に置きます。
翌日寄りから20/60/120営業日目終値までの単純収益を日付等重みで集計します。
満期に達していない行はゼロ補完せず除外するため、60/120日は対象期間・件数が異なります。
収益の観測窓は重複しており、独立な取引や年率リターンではありません。価格レンジへの将来到達を検証した表でもありません。
'''),code("table(pd.read_csv(P/'valuation_sorts.csv'))\ntable(pd.read_csv(P/'valuation_ic.csv'))"),md('''## 再現性と残る制約

入力・価格・国債・TOPIX・開示・実装・実験・既存パラメータのSHA256と実行ライブラリ版をinputs.jsonに保存。
元マートの既存列が不変であること、財務・金利の日付、LTR5の値動きを実データで検査しています。
'''),code('''for name in ['completed.json','data_checks.json']:
    table(pd.DataFrame([{'item':k,'value':v} for k,v in json.loads((P/name).read_text()).items()]))
'''),md('''この評価期間は過去の研究でも参照した単一ホールドアウトであり、完全な未使用期間ではありません。
複数方式・複数タスクを試しているため、僅かな改善だけで採用しません。XGBoost・CatBoost、再チューニング、独立した将来期間での再検証は未実施です。
FCFF/WACCによる企業価値DCFは、負債・再投資・平準化キャッシュフローを別途整備する必要があるため、今回計算していません。
本番360列の登録や本番モデルの置換は行っていません。

## 一次資料

- [財務省 国債金利CSV](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv)
- [財務省 金利定義・翌営業日公表時刻](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm)
- [NYU Damodaran：CAPMと割引率](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/dcfinput.html)
- [CFA Institute：残余利益モデル](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/residual-income-valuation)
- [NYU Damodaran：配当割引モデル](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/ddm.html)
''')]
if (P/'findings.md').exists():
    explanation,methods=nb.cells[2].source.split('## 計算方法と仮定',1)
    nb.cells[2]=md(explanation)
    nb.cells.insert(3,md('## 計算方法と仮定'+methods))
    nb.cells.insert(3,md((P/'findings.md').read_text(encoding='utf-8')))
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,P/'report.ipynb')
html,_=HTMLExporter(template_name='classic',exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(nb)
soup=BeautifulSoup(html,'html.parser')
for el in soup.find_all('script'):el.decompose()
for el in soup.find_all('link'):
    if str(el.get('href','')).startswith(('http:','https:','//')):el.decompose()
for el in soup.find_all('style'):el.string=re.sub(r'@import[^;]+;','',el.get_text())
(P/'report.html').write_text(str(soup),encoding='utf-8')
print(json.dumps({'executed_cells':sum(c.cell_type=='code' for c in nb.cells),'static_tables':len(soup.find_all('table'))}))
