#!/usr/bin/env python3
"""Execute and export the matched-retuning report; OUTPUT is the sole argument."""
from pathlib import Path
import json
import re
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
import pandas as pd

P=Path(sys.argv[1]).resolve()
plan=json.loads((P/'experiment_plan.json').read_text())
selected=json.loads((P/'selection_frozen.json').read_text())
rows=[];params=[];trials=[]
for kind in plan['tasks']:
    for arm in plan['arms']:
        key=kind+'_'+arm
        rec=json.loads((P/(key+'_trials.json')).read_text())['trials']
        best=selected['selections'][key]
        rows.append({'task':kind,'arm':arm,'candidates':len(rec),'cv_folds':len(best['cv_folds']),
             'chosen_candidate':best['candidate'],'cv_primary':best['cv_mean'],
             'cv_previous_params':rec[0]['cv_mean'],'cv_gain':best['cv_mean']-rec[0]['cv_mean'],
             'fit_seconds':sum(x['seconds'] for x in rec)})
        params.append({'task':kind,'arm':arm,'candidate':best['candidate'],**best['params']})
        trials.extend([{'task':kind,'arm':arm,'candidate':x['candidate'],'cv_primary':x['cv_mean'],
                        **{f'fold_{j+1}':v for j,v in enumerate(x['cv_folds'])}} for x in rec])
pd.DataFrame(rows).to_csv(P/'search_summary.csv',index=False)
pd.DataFrame(params).to_csv(P/'selected_parameters.csv',index=False)
pd.DataFrame(trials).to_csv(P/'all_trials.csv',index=False)
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
def md(s):return nbformat.v4.new_markdown_cell(s)
def code(s):return nbformat.v4.new_code_cell(s)
nb.cells=[md('''# 理論株価特徴量：同条件の再チューニング比較

設計・探索開始：2026-11-03。集計：2026-11-04。以前のE120は既存パラメータを固定した初期試験でした。
今回は**既存仕様・元指標追加・理論株価追加を、それぞれ同じ探索機会で最適化してから比較**します。
本番モデル・パラメータ・特徴量リスト・契約プランは変更していません。

## 再チューニング後の主比較
'''),code('''from pathlib import Path
import json,pandas as pd,numpy as np
from IPython.display import display,HTML
P=Path.cwd()
def table(d):display(HTML(d.to_html(index=False,max_rows=None,float_format=lambda x:f'{x:,.5f}')))
res=pd.read_csv(P/'results.csv');plan=json.loads((P/'experiment_plan.json').read_text())
names={'class10':'分類 +10%（AP）','class15':'分類 +15%（AP）','ltr':'日次ランキング（NDCG@5）','ltr5':'直近5営業日ランキング（NDCG@5）'}
allres=res[res.subset.eq('all')]
w=allres.pivot(index='task',columns='arm',values='primary').reindex(names)
summary=w.assign(B_minus_A=w.B-w.A,B_minus_C=w.B-w.C,C_minus_A=w.C-w.A).reset_index()
summary['task']=summary.task.map(names)
table(summary)
summary.to_csv(P/'comparison_summary.csv',index=False)
''')]
if (P/'findings.md').exists():nb.cells.append(md((P/'findings.md').read_text()))
nb.cells += [md('''A/B/Cの各スコアは、選択した設定で種42/43/44を再学習し、3つの予測を平均してから計算しています。
Aは今回の探索で最適化した既存特徴量の参照モデルで、保存済み本番モデルそのものではありません。
**B−Aは追加一式の効果、B−Cは元指標に加えて理論価格へ変換した効果**です。
CVの最高値を最終評価スコアとして掲載したものではありません。

## そろえた条件

| 条件 | 特徴量 |
|---|---|
| A | 既存360列。LTR5は値動き4列を追加 |
| C | A＋元財務比率・β・金利・観測数・古さ・利用可能性13列 |
| B | C＋RI/DDMそれぞれの最小・最大・平均・中央値・標準偏差の株価相対値10列 |

- 各タスク・各条件で**30候補×3つの前進検証窓、木200本、同じ行**を使います。
- 30候補は既存設定1組＋Optuna RandomSamplerで生成した29組。同じタスクの3条件に完全に同じ候補を渡し、各条件でCV最大の設定を別々に選びます。
  適応型TPEではありません。候補の引きの違いをそろえるための固定されたランダム探索です。
- 学習率0.01〜0.2、葉7〜127、最小葉サンプル10〜300、行割合0.5〜1、列割合0.3〜1、L1/L2正則化1e−8〜10。
  学習率・葉・最小サンプル・正則化は対数探索。行割合の適用頻度は1。早期打ち切りは使いません。
- 同じ探索の種42。探索中の学習種も42。最終当てはめは42/43/44。
- 目的は分類が各検証窓のAPの等重み平均、ランキングが各窓の日次NDCG@5の等重み平均。
  最良の候補番号が同率なら小さい番号を選びます。外側評価のスコアは選択に使いません。
- 検証データでbinを作りません。訓練側のbinを同じ条件内で再利用し、葉サイズを変えられるようfeature_pre_filter=False。
  binningの種42は固定、CV・最終学習の双方で同じ設定。分類不均衡補正とランキングのラベル・線形gainが本番の処理と一致することを別途テストしています。

30候補の有限探索で、全域最適を保証しません。探索候補の種を変えた再探索は未実施です。
以前のE120とは探索・bin設定を含む学習経路が違うため、E120との差をそのまま純粋なチューニング効果とは呼びません。

## 探索期間と最終評価の分離

探索は最終評価より前のデータだけで実施。内側検証は過去18ヶ月を6ヶ月ずつ3窓に区切り、訓練は各窓より前の全履歴です。
各会社の日足による20日目の結果確定日が検証開始日以降になる訓練行を除きます。日付グループを訓練・検証に分割しません。
全12条件の選択をselection_frozen.jsonへ保存した後に、最終評価の予測・スコアを計算します。
LTR5の財務と理論株価は候補日の値を凍結し、判断日に更新しません。
'''),code('''outer=json.loads((P/'outer_split.json').read_text())
table(pd.DataFrame(outer['tasks']))
folds=[{'task':task,**f} for task,fs in plan['inner_fold_checks'].items() for f in fs]
table(pd.DataFrame(folds))
'''),md('''**この外側期間はE120などで既に参照済みです。完全に未使用の将来期間ではありません。**
今回の再探索には使っていませんが、この比較だけで本番採用の独立な証明にはしません。
内側窓や探索予算を外側の新しい成績に応じて変更していません。モデル種3つは独立した3評価期間ではありません。

## 探索の完了数と選択設定

下表のCV値はパラメータ選択の物差しで、外側の主評価とは異なります。
candidate 0は既存設定。各条件が同じ既存設定から出発するので、探索後のCV値はその条件のcandidate 0を下回りません。
'''),code("table(pd.read_csv(P/'search_summary.csv'))\ntable(pd.read_csv(P/'selected_parameters.csv'))"),md('''## 外側評価の全指標

分類上位10%は評価期間全体のスコア上位10%。ランキングは候補5件以上の日で日付等重み。
収益は翌日寄りから20日目終値の配当抜き単純収益で、手数料・同時保有・約定制約を含みません。
'''),code('table(allres)'),md('''## 算出できた部分集合の比較

各条件で同じ行を対象にしています。ランキングはその部分集合内で順位ラベルを作り直しています。
'''),code("table(res[~res.subset.eq('all')])"),md('''## 四半期・学習種別の安定性

100行以上・20日以上ある四半期だけを比較します。種別は同じ選択設定での当てはめのばらつきです。
探索候補の種を変えた場合のばらつきは、この表に含みません。
'''),code('''for name,keys in [('quarter_results.csv',['task','quarter']),('seed_results.csv',['task','seed'])]:
    z=pd.read_csv(P/name).pivot(index=keys,columns='arm',values='primary')
    table(z.assign(B_minus_A=z.B-z.A,B_minus_C=z.B-z.C).reset_index())
'''),md('''## 同じ理論株価特徴量を使っていることの確認

金融入力や価格シナリオをこの評価に合わせて変更していません。元の計算候補率と欠損・統計を以下に再掲します。
学習には下限・上限・平均・中央値を終値で割って1を引き、標準偏差は終値で割って使います。円価格も別途保存済みです。
'''),code('''cov=pd.read_csv(P/'coverage.csv')
table(cov[cov.feature.isin(['tv_ri_mean_rel','tv_ddm_mean_rel','tv_beta'])])
table(pd.read_csv(P/'coverage_by_year.csv'))
s=pd.read_csv(P/'feature_summary.csv')
table(s[['feature','count','unique_values','missing_pct','min','5%','50%','95%','max']])
'''),md('''## 理論株価の仮定

CAPMの株主資本コストke=rf+β×ERPを使います。円建て10年国債の前日以前の利回り、過去504市場日のTOPIX対β（最低252組）、
βの2/3縮約＋1/3、ERP 3/4/5/6/7%です。配当抜きリターンのβ、長期国債を安全利子率とする近似です。

RIは直近自己資本とTTM純利益からROEを作り、5/10年でkeに収束、配当性向25/50/75%の30シナリオ。
V=B0＋各年残余利益の割引現在価値で、期末以降の超過利益はゼロと仮定します。
DDMは通期実績配当総額/現在株数をD0とし、成長率0/1/2%の15シナリオでV=D0×(1+g)/(ke−g)。
配当性向0%超100%以下、現在TTM黒字、ke−g≥1%などの適格条件を付けます。
各方式で80%以上のシナリオが正かつ有限な行だけ統計を出します。
**理論価格の幅は仮定による感応度で、信頼区間・将来価格の確率分布ではありません。**

財務は候補日の前日以前の開示版から選び、連結・非連結を混ぜません。利益は円総額でTTM化し、期末から候補日への分割を調整した株数で割ります。
原APIの過去訂正履歴の完全監査、会計基準移行時の前年値の組替え、OCI・増資・自社株買い・特別配当などの個別予測は未実施です。

## 実行・検証記録
'''),code('''for name in ['completed.json','numeric_verification.json','code_verification.json']:
    table(pd.DataFrame([{'item':k,'value':v} for k,v in json.loads((P/name).read_text()).items()]))
'''),md('''データ、計算実装、探索コード、元パラメータとライブラリ版はinputs.json、候補・分割・予算はexperiment_plan.jsonに保存しています。
全候補のCV成績と選択結果を保存し、後から外側評価で設定を選び直せないよう選択ファイルのSHA256を記録しています。
実行済みNotebookをHTMLへ変換し、JavaScriptと外部読込を除去しています。

XGBoost・CatBoost比較、探索種を変えた反復、独立した将来期間での確認は未実施です。本番採用・再学習・配信はしていません。

## 一次資料

- [財務省：国債金利の定義と公表時刻](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm)
- [NYU Damodaran：CAPM・割引率](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/dcfinput.html)
- [CFA Institute：残余利益モデル](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/residual-income-valuation)
- [NYU Damodaran：配当割引モデル](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/ddm.html)
''')]
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
