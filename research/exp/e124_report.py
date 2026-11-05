#!/usr/bin/env python3
"""Executed E124 notebook and script-free offline HTML. Argument: output directory."""
from pathlib import Path
import json
import os
import re
import sys
import pandas as pd
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from jupyter_client.kernelspec import KernelSpecManager

P=Path(sys.argv[1]).resolve()
metrics=pd.read_csv(P/'probability_metrics.csv')
paired=pd.read_csv(P/'paired_probability_comparison.csv')
exitdiff=pd.read_csv(P/'paired_exit_comparison.csv')
verdict=[]
for target in [10,15]:
    a=metrics.query('cohort=="strict" and target==@target and arm=="static_age"').iloc[0]
    b=metrics.query('cohort=="strict" and target==@target and arm=="daily_state"').iloc[0]
    ci=paired.query('cohort=="strict" and target==@target').iloc[0]
    status='日次更新の誤差が小さく、差の95%区間もゼロ未満' if ci.ci_high<0 else '日次更新の誤差が小さいが、差の95%区間はゼロを含む' if b.brier<a.brier else '日次更新の誤差改善を確認できず'
    verdict.append(f'- **+{target}%**：日数補正のBrier誤差 {a.brier:.4f} → 日次更新 **{b.brier:.4f}**。{status}。対象{int(b.trades)}取引・{int(b.landmarks)}回の未到達時点。')
policy=[]
for target in [10,15]:
    r=exitdiff.query('target==@target and arm=="daily_state"').iloc[0]
    policy.append(f'- +{target}%の利確を基準にした日次売却：固定利確との差は平均 **{r.gain_vs_fixed_pct:+.2f}pt**、95%区間 **{r.ci_low_pct:+.2f}〜{r.ci_high_pct:+.2f}pt**。')
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'name':'growthstock-daily','display_name':'Growthstock daily arrival','language':'python'}
nb.cells=[md('''# 購入後の日次更新：買値から+10%・+15%への到達確率

**2026-11-05集計／購入から最大20営業日／研究のみ・本番変更なし**

購入時の評価を出し続ける方法に対し、その後の値動きで確率を更新すると改善するかを検証しました。
買値と期限を固定し、購入日の寄りから数えた20営業日目までに買値×1.10・1.15へ日中高値が届く確率を、毎日の引け後に推定します。
購入後に高値更新候補から外れても、同じ購入分を追跡します。

## 結果
'''+ '\n'.join(verdict)+'''

Brier誤差は予測確率と実際の0/1到達結果の二乗差で、小さいほど良い指標です。
上の「日数補正」は、購入時の8モデル評価・購入前のテクニカル・経過/残日数を使う、別途チューニングした比較モデルです。
すでに到達した日の確率1は、この精度集計に含めません。

単純な過去の日数別到達率との比較では、+10%は日次モデルが下回り、+15%はほぼ同程度でした。
また、日次モデルの平均推定確率は未到達時点の実際の到達頻度より高く、確率の校正には課題が残っています。

**確率の改善と売却損益の改善は別に評価しています。**
'''+ '\n'.join(policy)+'''

差の区間がゼロを含む場合、優位性を断定できません。この歴史期間は以前の特徴量・出口研究で参照済みで、完全に未使用の将来検証ではありません。
'''),code('''from pathlib import Path
import io,json
import pandas as pd,numpy as np
from IPython.display import display,HTML,Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.cwd()
def read(name):return pd.read_csv(P/name)
def table(d):display(HTML(d.to_html(index=False,max_rows=None,na_rep='—',float_format=lambda v:f'{v:,.4f}')))
audit=json.loads((P/'data_audit.json').read_text())
tracker=read('daily_tracker.csv')
pred=pd.read_parquet(P/'outer_predictions.parquet')
metrics=read('probability_metrics.csv')
labels={'purchase_only':'購入時点の予測を固定','age_prior':'過去の高評価群：日数別到達率','static_age':'購入時情報＋日数補正','daily_state':'購入後の情報で日次更新'}
table(pd.DataFrame([{'評価の購入判断日':'2025-10-08〜2026-10-05','予測の引け日最終':str(pred.asof_date.max().date()),
'高閾値で選択':audit['strict_counts']['outer']['selected_before_quality'],
'価格・順位を再現して評価':audit['strict_counts']['outer']['complete_rank_and_price'],
'学習母集団の取引数':audit['trades'],'日次状態総数':audit['landmarks']}]))
'''),md('''## 日次で何を更新するか

例えば買値1,000円なら目標は1,100円・1,150円です。7営業日目の引けでは、残り13営業日を予測します。
株価が上がっても目標を「現在株価から+10%」に置き直しません。期限を毎日20日先に延ばすこともありません。
すでに達成した目標は「到達済み」、20日目まで未達なら「期限終了・未到達」と記録します。

- **日次の到達確率**：まだ到達していない目標への残り期間内の到達。
- **残り期間の上値分布**：明日以降の最高値の25/50/75%点。現在値比と買値比を保存。
- **損切り先行リスク**：副検証として買値−5%の線を仮定し、利確先着・損切り先着・双方未達を予測。
- **保有継続の価値**：固定利確・20日満了を続ける場合と、次の寄りで売る場合の予測損益差。

日次更新には、含み損益・購入後高値と安値・高値からの下落・高値更新からの日数・最近の騰落・出来高・ATR・TOPIXなどを使います。
将来の高値や20日目の結果は学習用ラベルとして保存し、入力には使いません。
'''),code('''z=metrics[metrics.cohort.eq('strict')].copy();z['arm']=z.arm.map(labels)
table(z[['target','arm','trades','landmarks','brier','trade_equal_brier','logloss','auc','predicted_pct','actual_pct','close_arrival_actual_pct']].rename(columns={
'target':'買値からの目標%','arm':'方法','trades':'取引数','landmarks':'未到達の引け時点数','brier':'Brier誤差',
'trade_equal_brier':'取引等重みの誤差','logloss':'対数損失','auc':'AUC','predicted_pct':'平均推定確率%',
'actual_pct':'実際の日中到達率%','close_arrival_actual_pct':'参考：終値到達率%'}))
table(read('paired_probability_comparison.csv').query('cohort=="strict"').rename(columns={
'target':'目標%','trades':'取引数','landmarks':'時点数','dynamic_minus_static_brier':'日次更新−日数補正のBrier差',
'ci_low':'差95%下限','ci_high':'差95%上限'}))
'''),md('''購入時の確率を固定する方法は、時間の経過や未到達という情報も反映しません。
そのため主比較は、同じ取引・同じ引け時点に対する「購入時情報＋日数補正」と「日次更新」です。
未到達の保有状態を日々予測する目的に合わせ、各リスク時点を等重みで集計しています。
同一取引の日数が増えても独立した取引が増えることにはならないため、区間は購入日の21市場営業日ブロックをまとめて再標本化しました。

61取引のうち、購入当日に+10%へ到達した6取引、+15%へ到達した1取引には、その後の未到達の引け予測がありません。
そのため確率評価はそれぞれ55取引・60取引で、売却損益の比較は61取引すべてです。
表の「実際の日中到達率」は、未到達の各引け時点からの残り期間内の頻度であり、購入時点からの到達率ではありません。

**この日中到達は、元の8モデルの正例条件全体とは異なります。** 終盤の半幅以上の水準やMA条件を要求する正例確率ではありません。
終値到達率は参考実績で、終値到達専用の確率モデルは今回学習していません。

## 購入後の日次推移の例

評価対象のうち、判断日・銘柄コード順に先頭4取引を表示します。成績の良い例を後から選んだものではありません。
確率が上がる・下がるのは、その日までの新情報と残り日数によります。到達後は1、期限終了時の未達は0として表示します。
'''),code('''first=tracker.sort_values(['signal_date','Code']).drop_duplicates('trade').head(4).trade.tolist()
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,trade in zip(axes.ravel(),first):
    z=tracker[tracker.trade.eq(trade)].sort_values('elapsed')
    for target,color in [(10,'#168572'),(15,'#3369b4')]:
        ax.plot(z.elapsed,z[f'p{target}_remaining'],color=color,lw=2,label=f'+{target}% daily')
        ax.plot(z.elapsed,z[f'p{target}_entry_age'],color=color,ls='--',alpha=.6,label=f'+{target}% entry+age')
    ax.set(title=f"{z.Code.iloc[0]} / {z.signal_date.iloc[0]}",xlabel='Sessions since purchase (0=open)',ylabel='Arrival probability',ylim=(0,1.02),xticks=[0,5,10,15,20])
handles,names=axes.ravel()[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=4)
buf=io.BytesIO();fig.savefig(buf,format='png',dpi=145);plt.close(fig);display(Image(data=buf.getvalue()))
z=tracker[tracker.trade.eq(first[0])].copy()
cols=['asof_date','elapsed','remaining','spot_return','p10_remaining','status10','p15_remaining','status15',
'future_high_q0.25_from_entry','future_high_q0.5_from_entry','future_high_q0.75_from_entry','continuation_gain10','continuation_gain15']
for col in cols:
    if col in z and pd.api.types.is_numeric_dtype(z[col]) and col not in ['elapsed','remaining']:z[col]*=100
table(z[cols].rename(columns={'asof_date':'引け日（0日目は寄り）','elapsed':'経過日','remaining':'残日数','spot_return':'含み損益%',
'p10_remaining':'+10%到達確率%','status10':'+10%状態','p15_remaining':'+15%到達確率%','status15':'+15%状態',
'future_high_q0.25_from_entry':'上値25%点：買値比%','future_high_q0.5_from_entry':'上値中央値：買値比%',
'future_high_q0.75_from_entry':'上値75%点：買値比%','continuation_gain10':'+10%継続価値pt','continuation_gain15':'+15%継続価値pt'}))
'''),md('''表の上値分位点は、将来価格の保証された上下限ではありません。残り期間の最高値は現在値より低くなる場合もあります。
また、その最高値で売れるという予測でもありません。20日目は残り期間がないため上値予測を空欄にしています。
「継続価値」は買値比の損益差で、最適な将来売却ルールの価値を求めたものではありません。

## 確率の校正と保有日数別の成績

例えば予測60〜80%の時点で、実際にも近い頻度で到達しているかを確認します。
同じ取引が複数の日・複数の確率帯に現れるので、時点数だけで確実性を判断できません。
'''),code('''cal=read('probability_calibration.csv').query('cohort=="strict"')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,target in zip(axes,[10,15]):
    ax.plot([0,100],[0,100],color='#bbbbbb',ls='--')
    for arm,color in [('static_age','#a0a8b0'),('daily_state','#168572')]:
        z=cal[cal.target.eq(target)&cal.arm.eq(arm)&cal.trades.ge(5)]
        ax.plot(z.predicted_pct,z.actual_pct,'o-',color=color,label=arm)
    ax.set(title=f'+{target}% / at least 5 trades per bin',xlabel='Mean predicted probability (%)',ylabel='Observed arrival (%)',xlim=(0,100),ylim=(0,100));ax.legend()
buf=io.BytesIO();fig.savefig(buf,format='png',dpi=145);plt.close(fig);display(Image(data=buf.getvalue()))
table(cal)
table(read('metrics_by_elapsed.csv').query('cohort=="strict"'))
'''),md('''## 売却損益への接続

固定の+10%／+15%指値・未達は20日目終値をそれぞれ比較基準にします。
日次売却モデルは「固定利確を続ける収益 − 次の寄りで売る収益」を予測します。その値が、前半で固定した閾値を下回った場合に次の寄りで売る方式です。
同じ日の引けを見て、その引け価格で売れたことにはしません。

閾値候補−1／0／+1ptは、チューニング側の時系列予測84取引・62判断日だけで選びました。
後続期間の成績で設定を選び直していません。買った当日に利確した取引も比較母集団に残します。
'''),code('''policy=read('exit_policy_metrics.csv');primary=policy[policy.cost_pct.eq(.2)].copy()
primary['arm']=primary.arm.map({**labels,'fixed_target':'固定指値・20日満了'})
table(primary.rename(columns={'target':'利確目標%','arm':'方法','trades':'取引数','mean_net_pct':'平均純損益%',
'win_pct':'勝率%','worst_net_pct':'最悪取引%','avg_days':'平均保有日数','early_exits':'日次判断で売却'}))
table(read('paired_exit_comparison.csv').rename(columns={'target':'目標%','arm':'方法','trades':'取引数','gain_vs_fixed_pct':'固定指値との差pt','ci_low_pct':'差95%下限','ci_high_pct':'差95%上限','cutoff':'固定した閾値（比）'}))
table(read('paired_state_exit_comparison.csv'))
table(read('continuation_gain_metrics.csv').query('cohort=="strict"'))
table(policy[policy.arm.eq('daily_state')][['target','cost_pct','mean_net_pct','worst_net_pct']])
table(read('gap_fill_sensitivity.csv'))
'''),md('''主集計は往復コスト0.2%。利確以上に寄っても指値価格、日次売却も同じ保守的な利確上限を適用します。
実際の寄り付き価格まで利益を認める感応度も、同じ売却設定で併記しました。
税・配当・実際の板や値幅制限・注文不能・資金枠の競合は未反映。取引あたりの平均で、運用資金の年率収益ではありません。

## 上値分布と損切り先行リスク（副検証）

上値分位点は現在値からの予測で学習し、その日の現在値を通じて買値基準にも換算します。
25〜75%点の区間は約50%の将来最高値を含むことが目安ですが、実際の包含率も確認します。
損切り線は副検証用に買値−5%を仮定。未だ利確にも損切りにも到達していない時点だけで評価します。
日足で双方に触れた順序が不明なら損切り先行、寄り付きで順序が明らかなら寄りを優先しています。
'''),code('''table(read('quantile_metrics.csv').query('cohort=="strict"'))
table(read('competing_risk_metrics.csv').query('cohort=="strict"'))
'''),md('''副検証の3分類モデルには専用の確率校正をしていません。
表の`target_first_above_unconditional_pct`は、独立に学習した損切り先行モデルの利確確率が、主モデルの無条件到達確率を上回る不整合の割合です。
この不整合がある場合、同時に整合した売却用確率としては利用できません。主モデルの+15%確率≤+10%確率は固定の単調化で守っています。
副モデルの点推定だけを、そのまま実運用の損切り判断へ使えるという検証ではありません。

## 同条件のパラメータ調整と時系列分割

- 比較2方式は同じデータ・同じ6候補・同じ3検証窓で、それぞれの最小誤差の設定を選択。木100本、種42、2スレッド。
- 内側検証は2023-10〜2024-03、2024-04〜09、2024-10〜2025-03。各窓より前に20日目まで確定した取引で学習。
- 確率校正は別の2025-04-01〜09-05の母集団で、単調なPlatt sigmoidを学習。校正も後続評価前に固定。
- 元の8モデルの探索ラベル最終確定2025-10-07の翌日以降を、今回の後続評価にしています。
- 後続期間のモデル再学習は月次。20日目の結果が月初より前に確定した取引のみを追加します。途中の早い利確だけを先に学習することもしていません。
- 1取引の行を学習と検証に分けません。未来の到達時間で決まるリスク期間の逆数による重み付けや、確率をゆがめる正例の重み増加はしていません。

6候補の有限探索で、全域最適化を保証しません。元の買いモデル8本は再チューニング・再学習していません。
'''),code('''sel=json.loads((P/'selection_frozen.json').read_text())['selections']
table(pd.DataFrame([{'task':k,'candidate':v['candidate'],'cv_loss':v['cv_loss'],'exit_cutoff':v.get('exit_cutoff'),**v['params']} for k,v in sel.items()]))
table(read('outer_maturity_checks.csv'))
table(read('exit_threshold_development.csv'))
'''),md('''## 入力の同一性と検査

前回E123と同じ現在の8モデルのOOFを使い、8ファイルのSHA256を本番メタデータに再照合しています。
株価は11月5日まで、TOPIXは11月2日まで。今回の評価の最後の引け予測は11月2日なので、必要なTOPIX状態を覆います。
株式分割・併合の調整後価格を使い、将来の一様な分割係数が相殺される相対価格・相対出来高を入力にしています。

購入前の情報と購入後の各日までのOHLCから特徴量を作ります。未来の価格・出来高を変えても以前の特徴量が変わらないテスト、
実際の61取引の買値基準・残り期間・翌日寄りの独立再計算、月次学習のラベル確定日を検査しています。

データ制約として、元マート全体の最新版の完全照合は未実施で、E123の固定マート＋OOFキー再構成に依存します。
OOFランキングに将来R20欠測で除外された候補がある日は、候補全体の順位を再現できず、学習・評価から除いています。
これは履歴を再現できた日に限る条件付き評価で、実運用で将来の欠測を予知して見送れるという意味ではありません。
65購入候補のうち61取引の結果であり、除外4取引を含む全体への性能は未評価です。
'''),code('''table(read('model_identity.csv')[['model','features','rows','trained_at','matches_production_meta']])
verification=json.loads((P/'data_verification.json').read_text())
table(pd.DataFrame([{'価格・状態の独立検査':verification['checks'],'確認した取引数':verification['strict_trades_independently_checked'],
'モデル入力：購入時＋日数':len(json.loads((P/'feature_contract.json').read_text())['static_age']),
'購入後情報を加えた入力':len(json.loads((P/'feature_contract.json').read_text())['daily_state']),
'特徴量の平均欠損率%':audit['technical_missing_pct']}]))
'''),md('''## 保存物と適用範囲

`daily_tracker.csv`は当日時点の情報と予測だけの時系列表です。検証用の未来の実績ラベルは別ファイルへ保存しています。
実行済みNotebook・外部スクリプト不要HTML・予測と実績・全候補の調整結果・選択設定・日次売却の取引別結果・検査記録を保存しました。

この結果は過去の条件付き検証です。実保有銘柄への毎日の自動更新や画面、本番売却判断にはまだ接続していません。
日次の確率推定の改善、確率の校正、売却収益の改善はそれぞれ確認が必要です。今回の後続期間をさらに見て設定を選ぶ場合は、新たな前向き期間での確認が必要になります。
''')]
for cell in nb.cells:
    if cell.cell_type=='markdown':cell.source=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',cell.source,flags=re.S)
kdir=P/'_kernel';kdir.mkdir(exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Growthstock daily arrival','language':'python'}))
KernelSpecManager().install_kernel_spec(str(kdir),kernel_name='growthstock-daily',prefix=str(P/'_jupyter'))
os.environ['JUPYTER_PATH']=str(P/'_jupyter/share/jupyter')+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
nb=NotebookClient(nb,timeout=150,kernel_name='growthstock-daily',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,P/'growthstock_daily_arrival.ipynb')
exporter=HTMLExporter(template_name='lab');exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb);soup=BeautifulSoup(body,'html.parser')
for tag in soup.find_all('script'):tag.decompose()
for tag in soup.find_all('link',href=True):
    if tag['href'].startswith(('http','//')):tag.decompose()
soup.title.string='Growthstock：購入後の日次+10%・+15%到達確率'
style=soup.new_tag('style');style.string='''body{background:#f2f6f9;color:#17354b;font-family:system-ui,"Noto Sans CJK JP",sans-serif}main{max-width:1280px;margin:auto;background:white;padding:24px}h1,h2,h3{color:#174f70}h1{font-size:29px}p,li{line-height:1.85}table{font-size:12px!important;border-collapse:collapse;width:max-content;max-width:100%}th{background:#e8f2f7!important;color:#174f70}td,th{padding:8px!important;border-bottom:1px solid #dbe6ee;text-align:right}td:first-child,th:first-child{text-align:left}.jp-OutputArea-output{overflow-x:auto}.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt{display:none}img{max-width:100%}''';soup.head.append(style)
(P/'growthstock_daily_arrival.html').write_text(str(soup))
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
print({'code_cells':sum(c.cell_type=='code' for c in nb.cells),'tables':len(soup.find_all('table')),'images':len(soup.find_all('img')),'scripts':len(soup.find_all('script'))})
