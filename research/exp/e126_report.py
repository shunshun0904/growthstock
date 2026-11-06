#!/usr/bin/env python3
"""Execute an E126 notebook and export a self-contained, script-free HTML report."""
from pathlib import Path
import json
import os
import sys
import pandas as pd
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from jupyter_client.kernelspec import KernelSpecManager

P=Path(sys.argv[1]).resolve()
policies=pd.read_csv(P/'policies.csv').set_index(['cohort','method'])
def mean(cohort,method):return f"{100*policies.loc[(cohort,method),'mean']:+.2f}%"
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'name':'growthstock-optimal-exit','display_name':'Growthstock optimal exit','language':'python'}
nb.cells=[md(f'''# 購入後の出口戦略：最適停止法の検証

**E126／2026-11-06／買値から+10%・+15%の到達予測／最長20営業日／手数料等0**

初回の実データ実験と過去期間でのパラメータ調整を完了しました。
今回の最適停止法を、既存の出口ルールより優先して本番採用する根拠は得られていません。

スコア分位95以上の69購入分では、平均損益は<strong>最適停止 {mean('base95','tuned_drawdown_boundary')}</strong>、
<strong>20日保有 {mean('base95','hold20')}</strong>、<strong>調整済み3%トレーリング {mean('base95','tuned_fixed_trailing')}</strong>でした。
最適停止は20日保有より損失を抑えましたが、トレーリングを上回っていません。
候補全体3,265件では最適停止 {mean('all','tuned_drawdown_boundary')}、20日保有 {mean('all','hold20')}でした。

既存8モデルは購入候補の選定・スコアの参照に使い、出口用の推定と判断を別に実装しました。
候補全体で推定・調整し、高閾値群を別集計しています。本番モデル・画面・注文処理は変更していません。
'''),code('''from pathlib import Path
import io,json,html
import numpy as np,pandas as pd
from IPython.display import display,HTML,Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.cwd()
def read(name,**kwargs):return pd.read_csv(P/name,**kwargs)
def table(d):display(HTML(d.to_html(index=False,na_rep='—',max_rows=None,float_format=lambda v:f'{v:,.4f}')))
def picture(fig):
    b=io.BytesIO();fig.savefig(b,format='png',dpi=140,bbox_inches='tight');plt.close(fig);display(Image(data=b.getvalue()))
policies=read('policies.csv');paired=read('paired_returns.csv');scores=read('scores.csv')
forecast=read('daily_forecasts.csv',dtype={'Code':str})
chosen=json.loads((P/'selected.json').read_text())['selected']
verification=json.loads((P/'numeric_verification.json').read_text())
numerics=json.loads((P/'numerics.json').read_text())
names={'hold20':'20日保有','first_possible_next_open':'購入翌日の寄りで売却','fixed_tp10':'+10%固定利確',
 'fixed_tp15':'+15%固定利確','tuned_fixed_trailing':'3%トレーリング（調整済）','tuned_drift_gate':'成長率による判断（調整済）',
 'tuned_drawdown_boundary':'最適停止境界（調整済）','boundary_earlier':'停止境界を早める感度分析',
 'boundary_later':'停止境界を遅らせる感度分析','fixed_tp10_limit':'+10%利確：指値固定の感度分析',
 'fixed_tp15_limit':'+15%利確：指値固定の感度分析'}
cohorts={'all':'候補全体','base95':'スコア分位95以上','base97':'スコア分位97以上','base99':'スコア分位99以上'}
methods={'drawdown_boundary':'最適停止用の推定','drift_gate':'成長率判定用の推定','age_prior':'過去の日数別到達率'}
primary=['hold20','fixed_tp10','fixed_tp15','tuned_fixed_trailing','tuned_drift_gate','tuned_drawdown_boundary']
z=policies[policies.method.isin(primary)&policies.cohort.ne('base99')].copy()
z['mean']*=100;z['median']*=100;z['win']*=100;z['tail5']*=100
z['method']=z.method.map(names);z['cohort']=z.cohort.map(cohorts)
table(z[['cohort','method','n','mean','median','win','tail5','mean_holding']].rename(columns={
 'cohort':'対象','method':'出口ルール','n':'仮想購入数','mean':'平均損益%','median':'中央値%','win':'利益割合%',
 'tail5':'下位5%の平均損益%','mean_holding':'平均保有営業日'}))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
labels=['Hold20','TP10','TP15','Trailing3%','Drift gate','Boundary']
for ax,cohort in zip(axes,['all','base95']):
    z=policies[policies.cohort.eq(cohort)].set_index('method').loc[primary]
    ax.barh(labels,100*z['mean'],color=['#8295a5']*3+['#159980','#789050','#ce6a45'])
    ax.axvline(0,color='#243e50',lw=.8);ax.set(title=f'{cohort}: {int(z.n.iloc[0])} episodes',xlabel='Mean return (%)')
picture(fig)
'''),md('''平均損益は、各仮想購入を同じ重みで数えた単純平均です。資金制約・複利を入れた運用利回りではありません。
短期売却は平均保有日数も短くなりますが、空いた資金の再投資効果は含めていません。

## 対象・時刻・目的変数

判断日の**翌市場営業日の寄り**で購入。購入日を1日目とし、1〜19日目の引けで推定・判断します。
裁量売却の約定は**翌市場営業日の寄り**。売らなければ元の20日目の引けで終了し、期限を延長しません。
手数料・固定コスト・モデル上のスリッページ控除はすべて0です。実際の翌朝への価格変動は約定価格に含みます。

目標価格は最後まで**元の買値×1.10・1.15**です。到達は日中高値の接触であり、既存分類器の正例の終盤水準・MA条件とは異なります。
最適停止と成長率判定には、+10%・+15%の強制利確を付けていません。これらは予測対象であり、固定利確は別の比較ルールです。
固定利確では購入当日の到達も含め、翌朝に目標を上回って始まれば実際の寄り値で約定する仮定です。

「候補全体」は、現行8モデルのOOF・完全なランキング母集団・20日分の価格経路を再現できる過去の推論候補です。
全上場銘柄×全営業日ではなく、実際の証券口座の購入履歴でもありません。
購入候補以外にも対象を広げたのは、従来の高閾値・同日上位2件への限定を外した、という意味です。
今回の高閾値群にも同日上位2件の制限はありません。

スコア分位95は、基本3分類器の過去スコア分位の最小値が95以上、という条件です。予測確率95%とは異なります。
6分類器・2ランキングの8モデルすべてを照合しましたが、8モデルの全一致を購入条件にした検証ではありません。
'''),code('''z=read('cohort_counts.csv');z['cohort']=z.cohort.map(cohorts)
z['period']=z.period.map({'history':'使用できる全履歴','initial_fit':'初回評価前に満了した推定用','outer':'共通の後半評価期間'})
table(z.rename(columns={'cohort':'対象','period':'区分','episodes':'仮想購入数','codes':'銘柄数','signal_dates':'判断日数'}))
table(pd.DataFrame([{'価格データ最終日':'2026-11-05','後半評価の判断日':'2025-10-08〜2026-10-05',
 '初回推定用':12002,'候補全体の後半評価':3265,'日次状態数':62035,'高閾値95以上':69,'高閾値97以上':31,'高閾値99以上':3}]))
'''),md('''元の16,492候補から、完全な20日経路が16,438件、元ラベルの満了とfold境界も安全なものが16,262件、
8モデルとランキング母集団の条件を合わせて15,506件です。途中で早く利確できたかどうかで採否を変えていません。
未完了・欠損・停止銘柄等への一般化は、この完全経路への条件付けによって制限されます。
評価期間の追加除外は0件です。

高閾値99以上は3件だけなので、参考値として末尾に保存し、信頼区間や有効性の判断を出しません。
日次62,035時点を独立した購入サンプル数とは数えません。
買値を基準に20日間のどこかで+10%へ到達した取引は、候補全体で1,198/3,265件（36.7%）、
高閾値95以上で34/69件（49.3%）でした。+15%はそれぞれ746/3,265件（22.8%）、21/69件（30.4%）です。
後述する「未到達の各引け時点からの残り到達率」とは分母が異なります。

## 実装した2方式と仮定

**最高値からの下落幅による最適停止（V1）**：du Toit & Peskirの、期間最高値Mと売却価格Pの比 `E[M/P]` を最小化する問題です。
株価が一定の成長率μ・変動率σの幾何ブラウン運動に従う場合、`α=μ/σ²` が0以下なら即売却、1以上なら期限保有、
中間では残日数に応じた最高値からの下落幅で売却します。日中高値を含む、購入後に既に分かっている最高値を使います。
反射ブラウン運動の遷移を積分し、後ろ向きの動的計画法で日次の停止境界を計算しました。

**成長率による判断（V2）**：`E[P/M]` を最大化する別の問題です。
既知・一定の対数成長率δが0以下なら即売却、正なら期限保有、という結果を参照します。
`E[M/P]` と `E[P/M]` は別の目的関数で、平均損益の最大化と同一でもありません。
Shiryaev・Xu・Zhouの研究を参照し、境界条件と2目的の区別はdu Toit・Peskirの定式化に従いました。

現実には成長率が未知なので、満了した過去候補の `log(20日目終値/買値)/20` から成長率の事前値を推定します。
訓練期間の1・99パーセンタイルで外れ値を抑え、各判断日の合計重みを同じにします。
候補が多い決算日に推定全体が支配されるのを抑える措置ですが、未処理の取引平均を直接推定するものではありません。
スコア帯を使う場合は、各帯を全体平均へ100件分縮めます。スコア分位も各推定時点で利用できた生スコアだけで作ります。

購入後の対数成長率は `δ_k=(κ×事前値+log(当日終値/元の買値))/(κ+k)` で更新します。
κは事前値の強さで、∞なら固定です。σは当日引けまでの20日または60日の対数収益率の標準偏差です。
必要な値が欠けた場合はもう一方の窓、その後は訓練期間だけの頑健な変動率で補います。後半評価ではこの補完は発生しませんでした。

これは推定値を当てはめて日々再判断する方式です。未知の成長率の不確実性まで織り込む完全なベイズ最適停止ではありません。
論文の即時売却を翌寄りへ遅らせること、成長率・変動率を再推定することから、実装に元論文の厳密な最適性は引き継がれません。

## 過去期間でのチューニング

共通の3検証期間は2023年10月〜2024年3月、2024年4〜9月、2024年10月〜2025年3月です。
各検証開始前に20日分の結果が確定した取引だけを推定に使い、購入分を途中で分割しません。
成長率判定は8候補、停止境界は16候補、トレーリングは4候補（3・5・8・12%）を比較しました。

選択指標は「候補全体」と「当時の過去データだけで求めた分位95以上群」を半分ずつ重み付けし、
平均損益から判断日平均の標準誤差を引いた値を3期間で平均したものです。
高閾値群が10件または5判断日未満なら、その期間の該当部分は候補全体の値に置き換える規則を事前に定めました。
後半評価結果を使って設定を選び直していません。
'''),code('''table(pd.DataFrame([{'方式':name,**value} for name,value in chosen.items()]))
z=read('tuning_folds.csv').groupby('fold',as_index=False)[['broad_n','tail_n','tail_dates','tail_fallback']].first()
table(z.rename(columns={'fold':'検証期間番号','broad_n':'候補全体','tail_n':'当時の高閾値群','tail_dates':'高閾値の判断日数','tail_fallback':'全体で代用'}))
table(read('tuning_summary.csv').rename(columns={'family':'方式','candidate':'候補ID','score':'3期間の選択指標','folds':'期間数'}))
'''),md('''成長率だけの方式では「全体の事前値を購入後も固定」が選ばれました。後半の全月でこの事前値が負だったため、
**3,265件すべて購入翌日の寄りで売るルールと一致**しました。この方式の損失抑制を、日次情報の変化を読む能力の証拠にはできません。
停止境界方式は「スコア帯別・κ=20・60日σ」が選ばれ、こちらは購入後に更新します。

後半期間では月ごとに、既に満了したデータだけで新規購入用の事前値を推定し直します。
一度購入した取引の事前値とスコア変換は固定し、購入後は観測した株価だけで更新します。
この方式・κ・窓長を後半の途中で選び直すことはしていません。

## 売却損益の差と不確実性

下表は「対象ルール − 比較ルール」の差（パーセントポイント）です。
同じ購入分を組にし、判断日を21市場営業日の循環ブロックで2,000回再標本化した95%区間を付けています。
日次状態ごとには再標本化しません。銘柄の再登場・重複保有による依存を完全に解消する手法ではありません。

高閾値95以上で停止境界と20日保有の差は+5.21pt（95%区間+2.68〜+8.65pt）でした。
一方、3%トレーリングとの差は−1.50pt（−5.15〜+3.78pt）で、最適停止の優位性は確認できません。
候補全体では、停止境界は+10%固定利確にも劣りました（差−1.35pt、区間−2.41〜−0.09pt）。
過去のE121〜E125でも見た評価期間を使っており、多数の比較を含む探索的な結果です。
'''),code('''z=paired[paired.cohort.ne('base99')].copy()
for c in ['delta','ci_low','ci_high']:z[c]*=100
z['cohort']=z.cohort.map(cohorts);z['method']=z.method.map(names);z['control']=z.control.map(names)
table(z.rename(columns={'cohort':'対象','method':'対象ルール','control':'比較ルール','n':'仮想購入数',
 'delta':'平均差pt','ci_low':'95%下限pt','ci_high':'95%上限pt'}))
'''),md('''## +10%・+15%の到達確率

各引け時点から元の期限までの最高値分布を、推定した幾何ブラウン運動から解析的に求めます。
買値から+10%・+15%の価格を超える確率を出し、既に到達済みの状態は精度集計から外します。
比較用の日数別到達率も、購入時点で満了済みの候補全体だけから推定しました。

Brier誤差は小さいほど良い指標です。主集計は未到達の引け時点を等重みにしています。
取引を等重みにする集計も併記します。早く到達する取引は未到達時点が少ないため、両者で結論が変わり得ます。
日数だけの基準は株価を見ないので、個別価格を使う方式に対する簡易な比較対象です。

候補全体では価格を使う2方式が日数別基準より良好でしたが、高閾値95以上の+10%では、
停止境界用推定のBrierが0.1318、日数別基準が0.1173で、主集計では改善しませんでした。
高閾値群の+15%は0.0995対0.1005でほぼ同程度です。
全体での予測精度の改善が、購入対象となる高閾値群の売却益の改善を保証するものではありません。
'''),code('''z=scores.copy();z['cohort']=z.cohort.map(cohorts);z['method']=z.method.map(methods)
table(z.rename(columns={'cohort':'対象','method':'推定方法','target':'目標%','n':'未到達状態を含む購入数','states':'未到達時点数',
 'brier':'時点等重みBrier','trade_equal_brier':'購入等重みBrier','predicted':'平均予測確率','actual':'実際の残り到達率','logloss':'対数損失'}))
for file,title in [('day_buckets.csv','経過日別'),('calibration.csv','確率帯別')]:
    display(HTML('<h3>'+title+'</h3>'))
    z=read(file).query("cohort in ['all','base95']").copy();z['method']=z.method.map(methods);z['cohort']=z.cohort.map(cohorts)
    table(z)
display(HTML('<h3>購入を等重みにしたBrier差：各方法 − 日数別基準</h3>'))
table(read('paired_brier.csv'))
'''),md('''## 残り期間の最高値の平均・中央値・レンジ

その日の終値を基準とする、残期間の最高値の平均と10・50・90%分位点を求め、元の買値比へ戻して記録しています。
10〜90%レンジの名目カバー率は80%ですが、実際の包含率は候補全体で約69%、高閾値95以上の停止境界用推定で約72%でした。
推定レンジは狭めであり、実際の不確実性を十分に捉えていません。

連続な株価を仮定すると、将来の最高値は今の終値以上になります。
実市場では夜間の下落や残期間の短さにより、明日以降の高値が今の終値に届かないことがあります。
その割合は候補全体で5.93%、高閾値95以上で8.39%でした。この構造的なずれは今回のGBMでは表現できません。
決算ジャンプ、裾の厚さ、将来の変動率の変化、成長率推定の不確実性も十分に含まれていません。
最高値が予測範囲に入っても、その価格で売れるという意味ではありません。
'''),code('''z=read('maxima.csv');z['cohort']=z.cohort.map(cohorts);z['method']=z.method.map(methods)
table(z.rename(columns={'cohort':'対象','method':'推定方法','states':'時点数','trades':'仮想購入数','mae':'平均予測のMAE',
 'bias':'平均予測の偏り','rmse':'平均予測のRMSE','median_mae':'中央値予測のMAE','q10_q90_coverage':'10〜90%レンジ包含率',
 'actual_below_current':'未来高値が今の終値未満の割合'}))
'''),md('''## 日次推移の例：判断日・コード順の最初の4件

高閾値95以上群から成績を使わずに選んでいます。売却後も仮想保有を続けた日次記録を残していますが、
その後の予測や実績を前の売却判断に使っていません。到達後の表示用確率は100%です。
'''),code('''high=forecast[forecast.base_min.ge(95)].sort_values(['signal_date','Code','elapsed'])
first=high.drop_duplicates('trade').head(4).trade.tolist()
fig,axes=plt.subplots(4,2,figsize=(12,13),layout='constrained')
for axrow,trade in zip(axes,first):
    z=high[high.trade.eq(trade)];r=z.iloc[0];ax,bx=axrow
    for target,color in [(10,'#147c9a'),(15,'#ce6a45')]:
        ax.plot(z.elapsed,100*z[f'drawdown_boundary_p{target}_ever'],label=f'+{target}%',color=color)
    ax.set(title=f'{r.Code} / {r.signal_date}',ylabel='Arrival probability (%)',ylim=(-3,103),xlabel='Session since purchase');ax.legend()
    bx.plot(z.elapsed,100*z.spot_return,label='Observed close',color='#243e50')
    bx.plot(z.elapsed,z.drawdown_boundary_max_mean_pct_from_buy,label='Expected remaining high',color='#159980')
    bx.fill_between(z.elapsed,z.drawdown_boundary_q10_pct_from_buy,z.drawdown_boundary_q90_pct_from_buy,color='#159980',alpha=.2,label='High q10-q90')
    bx.set(ylabel='Return from original buy (%)',xlabel='Session since purchase');bx.legend(fontsize=8)
picture(fig)
'''),md('''## 高閾値95以上：全69件の日次記録

各行を開くと、19回の引け後の推定を表示します。表の値は当日時点で計算できる情報だけです。
買値比の最高値平均・分位点と、+10%・+15%の未到達時点の確率を示します。
「売却判断」はその引け後に判断した場合、翌寄りで売るという条件です。既に売却した後も比較用に記録しています。
候補全体3,265件・62,035時点の完全な記録は同梱の `daily_forecasts.csv` にあります。
'''),code('''for trade,z in high.groupby('trade',sort=False):
    r=z.iloc[0]
    show=pd.DataFrame({'経過日':z.elapsed,'引け日':z.asof_date,'残日数':z.remaining,'含み損益%':100*z.spot_return})
    for target in [10,15]:
        show[f'+{target}%到達確率%']=[f'{100*p:.1f}' if s=='unarrived' else '到達済' for p,s in zip(z[f'drawdown_boundary_p{target}_remaining'],z[f'status{target}'])]
    for col,label in [('max_mean','最高値平均%'),('q10','最高値q10%'),('q50','最高値中央値%'),('q90','最高値q90%')]:
        show[label]=z[f'drawdown_boundary_{col}_pct_from_buy']
    show['売却判断']=z.drawdown_boundary_decision.map({True:'翌寄り売却',False:'保有'})
    title=f'{r.Code} ／判断日 {r.signal_date} ／購入日 {r.entry_date}'
    display(HTML('<details class="trade"><summary>'+html.escape(title)+'</summary><div class="trade-table">'+show.to_html(index=False,float_format=lambda v:f'{v:.2f}')+'</div></details>'))
'''),md('''## 数値計算・感度分析・検査記録

停止境界は391組の成長率・変動率条件で計算しました。
既知の特殊ケース、反射ブラウン運動の遷移、最高値分布の数値積分、粗い刻みと細かい刻み、広い計算領域を照合しています。
初期の粗いパラメータ表では補間誤差が大きかったため、損益評価前に刻みを細かくしました。
端で株価状態を打ち切る誤差も、遷移の指数モーメントを解析的に積分して修正しました。

確認点での最大の対数停止境界差は0.00535（価格比で約0.54%）でした。これは全状態に対する数学的な誤差保証ではありません。
この幅だけ早める・遅らせる両方の売却結果を保存し、良い方を選んでいません。
既知・一定の係数のGBMを10万経路発生させた確認では、V1の評価指標が即売却・期限保有より改善しました。
これは理想化した計算の確認であり、10万件の実際の取引検証ではありません。
'''),code('''table(pd.DataFrame([verification]))
table(pd.DataFrame([numerics]))
table(read('grid_convergence.csv'));table(read('grid_interpolation.csv'));table(read('synthetic_gbm_check.csv'))
table(read('state_audit.csv'))
display(HTML('<h3>全ルール・感度分析・少数群を含む保存結果</h3>'))
z=policies.copy()
for c in ['mean','median','win','tail5']:z[c]*=100
z['cohort']=z.cohort.map(cohorts);z['method']=z.method.map(names)
table(z.rename(columns={'cohort':'対象','method':'方式','n':'仮想購入数','mean':'平均損益%','median':'中央値%',
 'win':'利益割合%','tail5':'下位5%平均損益%','mean_holding':'平均保有日数','inverse_ratio':'平均M/売却値','capture_ratio':'平均売却値/M'}))
'''),md('''## 保存物・文献・今回の結論

`daily_forecasts.csv` は未来ラベルを含まない日次推定記録です。`p*_remaining` は残期間に目標へ到達する確率、
`p*_ever` は既に到達済みなら1とした表示用の値です。`trade_outcomes.csv` は将来実績を含む検証用の売却記録です。
`policies.csv`・`paired_returns.csv`・`scores.csv` に全比較を保存しました。
Notebookは実行済みで、HTMLは集計表・画像・69件の日次表を内包しています。JavaScriptと外部通信は不要です。
同梱データを同じ場所に置けばNotebookを再実行できます。元データからの再計算手順とプログラムは研究ブランチに保存します。

- Shiryaev, Xu & Zhou (2008), [Thou shalt buy and hold](https://doi.org/10.1080/14697680802563732)。[著者機関の要旨](https://research.polyu.edu.hk/en/publications/thou-shalt-buy-and-hold/)。
- du Toit & Peskir (2009), [Selling a stock at the ultimate maximum](https://arxiv.org/pdf/0908.1014)。DOI: 10.1214/08-AAP566。

今回の単純な推定値を当てはめる最適停止は、研究候補として実装・検証できました。
高閾値群での損失抑制は見られますが、より単純なトレーリングに対する改善は確認できません。
到達確率の全体評価が良くても、高閾値群の精度と売却損益にそのままつながらないことも分かりました。
本番への採用はせず、今回固定した条件で新しい期間の予測を記録することが次の比較になります。
''')]

kdir=P/'_kernel';kdir.mkdir(exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
 'display_name':'Growthstock optimal exit','language':'python'}))
KernelSpecManager().install_kernel_spec(str(kdir),kernel_name='growthstock-optimal-exit',prefix=str(P/'_jupyter'))
os.environ['JUPYTER_PATH']=str(P/'_jupyter/share/jupyter')+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
nb=NotebookClient(nb,timeout=180,kernel_name='growthstock-optimal-exit',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,P/'growthstock_optimal_exit.ipynb')
exporter=HTMLExporter(template_name='lab');exporter.exclude_input=True
exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb);soup=BeautifulSoup(body,'html.parser')
for tag in soup.find_all('script'):tag.decompose()
for tag in soup.find_all('link',href=True):
    if tag['href'].startswith(('http','//')):tag.decompose()
soup.title.string='Growthstock：購入後の最適停止・到達確率の検証'
style=soup.new_tag('style');style.string='''
body{background:#eff3f6;color:#193247;font-family:system-ui,"Noto Sans CJK JP",sans-serif}
main{max-width:1280px;margin:auto;background:white;padding:24px}h1,h2,h3{color:#165777}
h1{font-size:28px}p,li{line-height:1.8}table{font-size:12px!important;border-collapse:collapse}
th{background:#e6f1f6!important;color:#165777}td,th{padding:8px!important;border-bottom:1px solid #d9e5ed;white-space:nowrap;text-align:right}
td:first-child,th:first-child{text-align:left}.jp-OutputArea-output,.trade-table{overflow-x:auto}
.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt{display:none}img{max-width:100%}
details.trade{border:1px solid #d9e5ed;border-radius:7px;margin:8px 0;padding:10px;background:#f8fbfd}
summary{cursor:pointer;font-weight:600;color:#165777}details[open] summary{margin-bottom:12px}
@media(max-width:600px){main{padding:12px}h1{font-size:23px}.jp-Cell{padding:2px}}
''';soup.head.append(style)
(P/'growthstock_optimal_exit.html').write_text(str(soup))
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
assert len(soup.find_all('details'))==69 and not soup.find_all('script')
print({'code_cells':sum(c.cell_type=='code' for c in nb.cells),'tables':len(soup.find_all('table')),
       'images':len(soup.find_all('img')),'details':len(soup.find_all('details')),'scripts':len(soup.find_all('script'))})
