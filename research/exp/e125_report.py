#!/usr/bin/env python3
"""Executed E125 notebook, forecast-only CSV and offline HTML. Argument: results."""
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

P = Path(sys.argv[1]).resolve()
d = pd.read_parquet(P/'daily_tracker.parquet')
q = pd.read_parquet(P/'queries.parquet')
forecast = q[['trade','Code','signal_date','entry_date','asof_date','elapsed','remaining','buy','close','spot_return']].copy()
forecast['status10'] = q.risk10.map({True:'unarrived',False:'arrived'})
forecast['status15'] = q.risk15.map({True:'unarrived',False:'arrived'})
for method, z in d.groupby('method'):
    z = z.set_index('qid').reindex(q.qid)
    for t in (10,15):
        forecast[f'{method}_p{t}_remaining'] = z[f'p{t}'].to_numpy()
        forecast[f'{method}_p{t}_ever'] = z[f'p{t}'].where(q[f'risk{t}'].to_numpy(),1).to_numpy()
    if method=='chronos2':
        ratio = q.close.to_numpy()/q.buy.to_numpy()
        for name in ['max_mean','q10','q25','q50','q75','q90']:
            forecast[f'chronos2_{name}_pct_from_buy'] = 100*(z[name].to_numpy()*ratio-1)
        forecast['chronos2_max_sd_pct_of_buy'] = 100*z.max_sd.to_numpy()*ratio
forecast.to_csv(P/'daily_forecasts.csv',index=False)

md = nbformat.v4.new_markdown_cell; code = nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {'name':'growthstock-tsfm','display_name':'Growthstock TSFM exit','language':'python'}
nb.cells = [md('''# 購入後の上値予測：TSFMと単純モデルの比較

**2026-11-06／買値から+10%・+15%／購入から最大20営業日／研究のみ**

今回のChronos-2を出口判断へ採用する根拠は得られませんでした。
既存8モデルのOOFと購入候補を固定し、購入後の予測だけを検証しています。本番への接続・モデルの差し替えは行っていません。

事前に定めた設定のChronos-2を追加学習なしで使いました。購入後の残り期間の最高値の分布を予測し、その分布から固定した買値×1.10・1.15への到達確率を求めます。
日々の確率変化を売りに使った場合の損益も、同じ購入分で評価しました。

## 結果

- 到達確率のBrier誤差は、+10%でChronos-2 **0.1503**、過去の日数別到達率 **0.1363**、ボラティリティモデル **0.1431**。小さいほど良い指標です。
- +15%ではそれぞれ **0.1193／0.1118／0.1061**。Chronos-2に優位性は見られませんでした。
- 残り期間の最高値の平均予測の絶対誤差は、現在値比でChronos-2 **10.91%**、ボラティリティモデル **7.70%**。
- 「到達確率20%未満で翌朝売却」のChronos-2と固定利確の損益差は、+10% **−0.31pt**、+15% **+1.54pt**。両方とも95%区間がゼロを含みます。

対象は**60購入分・57銘柄・43購入判断日**です。日次1,140時点や乱数で作った4,096経路を独立したサンプル数として扱いません。
この1モデル・1設定の結果で、TSFM全般の有効性を否定することもできません。
'''),code('''from pathlib import Path
import io,json,html
import pandas as pd,numpy as np
from IPython.display import display,HTML,Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.cwd()
def read(name):return pd.read_csv(P/name)
def table(x):display(HTML(x.to_html(index=False,na_rep='—',max_rows=None,float_format=lambda v:f'{v:,.4f}')))
def picture(fig):
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
    display(Image(data=buf.getvalue()))
audit=json.loads((P/'data_audit.json').read_text())
plan=json.loads((P/'protocol.json').read_text())
verification=json.loads((P/'numeric_verification.json').read_text())
scores=read('scores.csv');policies=read('policies.csv')
forecasts=read('daily_forecasts.csv')
tracker=pd.read_parquet(P/'daily_tracker.parquet')
names={'chronos2':'Chronos-2','age_prior':'過去の購入候補：日数別到達率','volatility':'ボラティリティ'}
rules={'fixed':'固定利確・期限','absolute':'残り到達確率20%未満','half':'初日引けの到達確率から半減'}
table(pd.DataFrame([
 {'区分':'過去の購入候補（到達率の参照用）','購入数':215,'日次予測数':215*19,'期間':'2021年〜2025年9月満了'},
 {'区分':'E124の購入選択（価格等の検査前）','購入数':65,'日次予測数':None,'期間':'2025-11-14〜2026-10-02の共通評価期間'},
 {'区分':'価格・8モデル順位を再現できる購入分','購入数':61,'日次予測数':1159,'期間':'同上'},
 {'区分':'追加の文脈長検査後：今回の共通評価','購入数':60,'日次予測数':1140,'期間':'同上'},
 {'区分':'+10%未到達の引け時点','購入数':55,'日次予測数':652,'期間':'購入後1〜19営業日目'},
 {'区分':'+15%未到達の引け時点','購入数':59,'日次予測数':880,'期間':'購入後1〜19営業日目'}]))
'''),md('''## 購入条件・時刻・サンプルの扱い

購入選択は既存研究で固定した「基本3分類器の過去スコア分位がすべて95以上、同日上位2件」です。
8モデルのOOFと現行モデルメタデータのハッシュを照合しています。8モデルの全一致を購入条件にした検証ではありません。
実際の約定記録ではなく、判断日の翌営業日の寄りで買う仮想取引です。資金3枠などの同時保有制約は入れていません。

以前の価格・順位の検査で65件から4件を除外。さらに今回、256営業日分の文脈のうち有効値192点以上という条件に1件が届かず、予測前に除外しました。
この1件は最小157点でした。採用した60件の文脈は最小193点です。未売買日を市場営業日で再索引し、未来から補完していません。
比較方法ごとに都合の良い銘柄を変えず、60件すべてを損益評価に含めています。除外5件の成果は未評価です。

買値をBとして、目標は常にB×1.10・B×1.15。購入日を1日目とし、k日目の引け後に、残り20−k営業日を予測します。
期限は購入から20営業日目。途中で現在値からの+10%へ置き換えたり、保有期限を延ばしたりしません。
到達は**日中高値の接触**です。既存分類モデルの正例に含まれる終盤水準・MA条件とは評価対象が異なります。

すでに目標へ到達した時点は到達確率の精度集計から除外します。購入当日の利確も含め、売却比較は全60件を対象にします。
日次記録は20日満了までの仮想保有を追うため、各売却ルールで売却した後の予測も保存されます。売却後の予測がその取引の損益へ影響することはありません。

## 使ったモデルと仮定

**Chronos-2（ゼロショット）**：公開済みの重みを読み込み、追加学習・購入ラベルによる確率校正・ハイパーパラメータ探索はしていません。
2025-10-30公開の重みをコミットで固定しました。評価の最初の購入判断は2025-11-14です。
入力は分割調整済み日中高値だけから作った256点の時系列。決算、銘柄名、既存8スコア、将来共変量を入力していません。
入力同士のcross learningを無効化し、単独推論と複数件推論の差も検査しました。

**残り期間の最高値を直接予測する工夫**：残りr日なら、過去のr日移動最高値Mを作ります。
Mのr日先は、ちょうど明日から保有期限までの高値の最大値です。

`M[t+r] = max(H[t+1], ..., H[t+r])`

既知の現在終値Cで割って対数化したMの系列を入力し、r日先の分位点を価格比へ戻します。
各日の高値予測の分位点を最大化する方法では、複数日の到達確率を正しく求められません。今回は上の恒等式を使っています。
ただし、加工した移動最高値系列の動きを汎用TSFMが日本株の購入候補で予測できる、という転用上の仮定は残ります。

**日数別到達率**：評価より前に20日分の結果が確定した購入候補215件だけを参照。
同じ経過日・まだ目標未到達の取引から`(到達件数+1)/(件数+2)`を計算し、その表を評価期間中固定しています。
購入しない広い銘柄群のラベルは使っていません。

**ボラティリティモデル**：その日の引けまでの60市場営業日を使い、翌寄りへの対数ギャップの分散と、日中H/Lレンジによる分散を推定します。
ゼロ対数ドリフト、独立な正規ギャップ・日中変動、日中Brownian bridgeの高値を仮定し、残り期間を4,096経路発生。
この経路の最高値から確率・平均・分位点を計算します。ジャンプの裾の厚さ、連続した決算効果、変動率の将来変化を十分表現するものではありません。
対数ドリフトゼロは価格の期待リターンゼロとは一致しません。
'''),code('''z=scores.copy();z['method']=z.method.map(names)
table(z.rename(columns={'method':'方法','target':'目標%','trades':'購入数','states':'未到達時点数','brier':'Brier誤差',
 'logloss':'対数損失','trade_equal_brier':'購入等重みのBrier誤差','predicted':'平均予測確率','actual':'実際の残り到達率','endpoint_fraction':'0または1の予測割合'}))
table(read('paired_brier.csv').rename(columns={'target':'目標%','comparison':'比較','delta_brier':'Brier差','ci_low':'95%下限','ci_high':'95%上限'}))
fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
order=['age_prior','volatility','chronos2']
for ax,target in zip(axes,[10,15]):
    v=scores[scores.target.eq(target)].set_index('method').loc[order]
    ax.bar(['Age prior','Volatility','Chronos-2'],v.brier,color=['#637a92','#159980','#ce6a45'])
    ax.set(title=f'+{target}% arrival: Brier (lower is better)',ylim=(0,.17),ylabel='Brier score')
picture(fig)
'''),md('''Brier差は「Chronos-2 − 比較方法」なので、正ならChronos-2の誤差が大きいことを示します。
いずれの差の95%区間もゼロを含みます。区間は購入判断日の21市場営業日ブロックを循環再標本化して2,000回計算し、同じ購入分の日次記録をまとめて扱いました。
購入日が離れた取引や再購入の依存が完全に消えるわけではありません。

主指標は未到達の引け時点を等重みとした誤差です。購入分を等重みにした列も併記しており、日次記録を長く持つ取引が増える影響を確認できます。
平均予測確率と実際の到達頻度には差が残ります。0/1付近の予測は外れた際に影響が大きく、対数損失では数値上1e−6で端点を制限しています。

### 経過日・確率帯ごとの結果
'''),code('''z=read('day_buckets.csv');z['method']=z.method.map(names)
table(z.rename(columns={'method':'方法','target':'目標%','day_start':'開始日','day_end':'終了日','states':'時点数','trades':'購入数',
 'brier':'Brier誤差','predicted':'平均予測確率','actual':'実際の残り到達率'}))
z=read('calibration.csv');z['method']=z.method.map(names)
table(z.rename(columns={'method':'方法','target':'目標%','bin':'予測確率帯(0から4)','states':'時点数','trades':'購入数',
 'predicted':'平均予測確率','actual':'実際の残り到達率'}))
'''),md('''## 残り期間の最高値のレンジ・平均値

モデルの1〜99%分位点の間は価格に対して線形で補間し、その外側の裾を端点の値で固定した近似分布を主設定にしています。
平均・標準偏差は、この**仮定した分位関数の積分**から計算した値です。真の期待値が分かったという意味ではありません。
上昇分の正値だけを切り取らず、明日以降の高値が今日の終値を下回る可能性も扱います。
最高値がこの範囲になっても、その価格で売れる保証はありません。

下表の誤差・偏りは現在値を1とした比率です。10.91%は平均予測の絶対誤差の平均であり、平均的に10.91%上がるという推定ではありません。
10〜90%区間の名目カバー率は80%。60件の全引け時点ではChronos-2が77.89%、ボラティリティモデルが80.96%でした。
'''),code('''z=read('maxima.csv');z['method']=z.method.map(names)
table(z.rename(columns={'method':'方法','states':'時点数','trades':'購入数','mean_mae':'平均予測のMAE','mean_rmse':'平均予測のRMSE',
 'mean_bias':'平均予測の偏り','median_mae':'中央値予測のMAE','q10_q90_coverage':'10〜90%区間の実績包含率'}))
tail=json.loads((P/'tail_sensitivity.json').read_text())
table(pd.DataFrame([tail]))
'''),md('''裾を線形外挿した感度分析では、平均値の変化の絶対値は平均で現在値比**0.038pt**、最大で**9.43pt**でした。
一部の分位点の大きさが平均値へ強く影響しています。標準偏差の変化は平均0.306pt、到達確率の変化は最大約1ptです。
この仮定を変えて成績の良い設定を選ぶ作業はしていません。分位点の交差15箇所は価格の昇順に並べ直してから使いました。

## 日次推移：判断日・銘柄コード順の先頭4購入分

成績による例示の選択はしていません。確率は未到達なら残り確率、到達後は到達済みの1として描いています。
下の全件表では到達済みを文字で示し、未来の実績との比較は検証欄に分けています。
'''),code('''first=forecasts.sort_values(['signal_date','Code']).drop_duplicates('trade').head(4).trade.tolist()
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,trade in zip(axes.ravel(),first):
    z=forecasts[forecasts.trade.eq(trade)].sort_values('elapsed')
    for target,color in [(10,'#159980'),(15,'#4072b5')]:
        ax.plot(z.elapsed,z[f'chronos2_p{target}_ever'],color=color,label=f'Chronos +{target}%')
        ax.plot(z.elapsed,z[f'volatility_p{target}_ever'],color=color,ls='--',alpha=.6,label=f'Vol +{target}%')
    ax.set(title=f"{z.Code.iloc[0]} / {z.signal_date.iloc[0]}",xlabel='Purchase session',ylabel='Reached / remaining probability',ylim=(0,1.02),xticks=[1,5,10,15,19])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
picture(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,trade in zip(axes.ravel(),first):
    z=forecasts[forecasts.trade.eq(trade)].sort_values('elapsed')
    ax.fill_between(z.elapsed,z.chronos2_q10_pct_from_buy,z.chronos2_q90_pct_from_buy,color='#bfd5ee',alpha=.7,label='Future high q10-q90')
    ax.plot(z.elapsed,z.chronos2_q50_pct_from_buy,color='#4072b5',label='Future high median')
    ax.plot(z.elapsed,z.chronos2_max_mean_pct_from_buy,color='#ce6a45',label='Future high mean (tail assumption)')
    ax.plot(z.elapsed,100*z.spot_return,color='#17354b',ls='--',label='Known close')
    ax.axhline(10,color='#159980',lw=.7);ax.axhline(15,color='#159980',lw=.7,ls=':')
    ax.set(title=f"{z.Code.iloc[0]} / {z.signal_date.iloc[0]}",xlabel='Purchase session',ylabel='% of original purchase price',xticks=[1,5,10,15,19])
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
picture(fig)
'''),md('''## 到達確率を売りに使った場合

すべての方法で、+10%または+15%の固定利確と20日目引けの期限を維持します。
主ルールは「引けの残り到達確率が20%未満なら翌営業日の寄りで売る」。副ルールは「購入初日の引けの確率を基準に、2日目引け以降に半減したら翌朝売る」です。
この2ルールと閾値は今回の予測前に固定しており、60件で最適な閾値を探していません。

主集計は上方の寄り付きギャップを利確幅までとする約定仮定を各方法に同じく適用し、往復費用0.2%を引きます。
平均損益は1購入分に等額投資した単純平均。資金再利用や複利を含むポートフォリオ利回り・年率収益率ではありません。
'''),code('''z=policies[policies.profit_gap.eq('limit')].copy()
for c in ['net','delta_fixed','ci_low','ci_high']:z[c]=100*z[c]
z['method']=z.method.map({**names,'fixed':'固定利確・期限'});z['rule']=z.rule.map(rules)
table(z[['target','method','rule','trades','net','delta_fixed','ci_low','ci_high','win_rate','mean_holding','early_exit_fraction']].rename(columns={
 'target':'目標%','method':'方法','rule':'出口条件','trades':'購入数','net':'費用後平均損益%','delta_fixed':'固定利確との差pt',
 'ci_low':'差95%下限pt','ci_high':'差95%上限pt','win_rate':'費用後勝率','mean_holding':'平均保有営業日','early_exit_fraction':'確率で売却した割合'}))
'''),md('''Chronos-2の20%ルールは、+10%で費用後平均**−0.49%**（固定利確−0.18%）、+15%で**−0.43%**（固定利確−1.97%）でした。
固定利確との差の95%区間は、+10%が**−2.00〜+1.98pt**、+15%が**−1.30〜+4.73pt**です。
確率予測と売却損益のいずれでも、この設定を本番採用する裏付けは不足しています。

日数別到達率の半減ルールでは+15%の差が+1.90pt、区間が+0.52〜+3.80ptでしたが、費用後損益自体は−0.07%です。
複数の方法・2目標・2ルールを見た中の結果で、区間の多重比較補正はしていません。以前の研究で同じ歴史期間も参照済みです。
この部分だけを取り出して優位な売り戦略が確定したとは扱いません。

### 約定・費用の感度

下表では上方ギャップを実際の寄り値で売れたケースも示します。
費用0〜1.0%の表では全方法に同じ往復費用を適用しています。保有日数による借入費用等を入れていないため、固定利確との差はこの固定費用では変わりません。
'''),code('''z=policies.copy();z['method']=z.method.map({**names,'fixed':'固定利確・期限'});z['rule']=z.rule.map(rules)
for c in ['net','delta_fixed','ci_low','ci_high']:z[c]*=100
table(z[['target','profit_gap','method','rule','net','delta_fixed','ci_low','ci_high']].rename(columns={
 'target':'目標%','profit_gap':'ギャップ約定','method':'方法','rule':'出口条件','net':'平均損益%','delta_fixed':'固定利確との差pt','ci_low':'差95%下限pt','ci_high':'差95%上限pt'}))
z=policies[policies.profit_gap.eq('limit')].copy();cols=[c for c in z if c.startswith('net_cost_')]
z[cols]*=100
table(z[['target','method','rule']+cols])
'''),md('''## 全60件の日次予測記録

下の各行を開くと、その購入分の19回の引け予測を読めます。JavaScriptや外部通信は不要です。
含み損益、最高値の予測平均・分位点・標準偏差は、すべて元の買値を基準にした%です。標準偏差は買値に対する価格の散らばりです。
目標到達後は「到達済」と表示します。20日目には新たな残期間の予測を出さず、利確または満了します。
初日からの予測の変化を見るために、個々の売却ルールで売却した後も仮想保有として表示しています。
'''),code('''for trade,z in forecasts.sort_values(['signal_date','Code','elapsed']).groupby('trade',sort=False):
    r=z.iloc[0]
    show=pd.DataFrame({'経過日':z.elapsed,'引け日':z.asof_date,'残日数':z.remaining,'含み損益%':100*z.spot_return})
    for method,short in [('chronos2','Chronos'),('volatility','Vol'),('age_prior','日数別')]:
        for t in [10,15]:
            show[f'{short}+{t}%確率%']=[f'{100*p:.1f}' if status=='unarrived' else '到達済' for p,status in zip(z[f'{method}_p{t}_remaining'],z[f'status{t}'])]
    for field,label in [('max_mean','最高値平均%'),('q10','最高値q10%'),('q50','最高値中央値%'),('q90','最高値q90%')]:
        show[label]=z[f'chronos2_{field}_pct_from_buy']
    show['最高値標準偏差/買値%']=z.chronos2_max_sd_pct_of_buy
    title=f"{r.Code} ／判断日 {r.signal_date} ／購入日 {r.entry_date}"
    display(HTML('<details class="trade"><summary>'+html.escape(title)+'</summary><div class="trade-table">'+show.to_html(index=False,float_format=lambda v:f'{v:.2f}')+'</div></details>'))
'''),md('''## 検査記録と保存物

全件について、過去高値だけからの入力の再計算、残り期間と20日満了日の一致、最高値ラベル、既知時点だけの変動率、確率分布の数値積分、到達確率の入れ子、翌寄りの売却を独立に計算しました。
単独・バッチ推論の最大差は対数値で約1.8e−7でした。売却ケース1,680件の損益も一致しています。
'''),code('''table(pd.DataFrame([{'検査完了':verification['all_passed'],'検査項目数':verification['checks'],'購入分':verification['trades'],
 '引け時点':verification['states'],'個別売却ケース':verification['exit_cases'],'重み公開日':plan['tsfm']['weight_publication'],
 'モデル':plan['tsfm']['model'],'重みコミット':plan['tsfm']['revision']}]))
'''),md('''`daily_forecasts.csv`は当日時点の情報と予測だけの記録です。`p*_remaining`は将来分だけの到達確率、`p*_ever`は既に到達していれば1とした表示用の値です。
`daily_tracker.parquet`には検証用に未来の実績ラベルを結合しています。実運用の入力用ファイルではありません。
実行済みNotebookは結果を開くだけで閲覧できます。HTMLもすべての集計表・画像・全60件の予測表を内包しています。
CSV等を同じ場所へ展開すればNotebookを再実行できます。モデルからの再推論はリポジトリのE125スクリプトと固定した公開重みを使います。

今後は、今回の設定で確率とレンジの記録を固定し、新しい購入期間で検証するのが妥当です。
異なるTSFMや入力・文脈長を試す場合は別実験とし、購入対象に限定した評価と売却損益の比較を維持する必要があります。
本研究の計算は完了しました。実保有の自動追跡・本番画面・注文処理にはまだ接続していません。
''')]

# The explicit kernel points at the Python used to generate this artifact.
kdir=P/'_kernel';kdir.mkdir(exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                          'display_name':'Growthstock TSFM exit','language':'python'}))
KernelSpecManager().install_kernel_spec(str(kdir),kernel_name='growthstock-tsfm',prefix=str(P/'_jupyter'))
os.environ['JUPYTER_PATH']=str(P/'_jupyter/share/jupyter')+(os.pathsep+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
nb=NotebookClient(nb,timeout=180,kernel_name='growthstock-tsfm',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,P/'growthstock_tsfm_exit.ipynb')
exporter=HTMLExporter(template_name='lab');exporter.exclude_input=True
exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb);soup=BeautifulSoup(body,'html.parser')
for tag in soup.find_all('script'):tag.decompose()
for tag in soup.find_all('link',href=True):
    if tag['href'].startswith(('http','//')):tag.decompose()
soup.title.string='Growthstock：購入後TSFM検証 +10%・+15%'
style=soup.new_tag('style');style.string='''
body{background:#eff3f6;color:#193247;font-family:system-ui,"Noto Sans CJK JP",sans-serif}
main{max-width:1280px;margin:auto;background:white;padding:24px}h1,h2,h3{color:#165777}
h1{font-size:28px}p,li{line-height:1.8}strong{font-weight:700}table{font-size:12px!important;border-collapse:collapse}
th{background:#e6f1f6!important;color:#165777}td,th{padding:8px!important;border-bottom:1px solid #d9e5ed;white-space:nowrap;text-align:right}
td:first-child,th:first-child{text-align:left}.jp-OutputArea-output,.trade-table{overflow-x:auto}
.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt{display:none}img{max-width:100%}
details.trade{border:1px solid #d9e5ed;border-radius:7px;margin:8px 0;padding:10px;background:#f8fbfd}
summary{cursor:pointer;font-weight:600;color:#165777}details[open] summary{margin-bottom:12px}
@media(max-width:600px){main{padding:12px}h1{font-size:23px}.jp-Cell{padding:2px}}
''';soup.head.append(style)
(P/'growthstock_tsfm_exit.html').write_text(str(soup))
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
assert len(soup.find_all('details'))==60
print({'code_cells':sum(c.cell_type=='code' for c in nb.cells),'tables':len(soup.find_all('table')),
       'images':len(soup.find_all('img')),'details':len(soup.find_all('details')),'scripts':len(soup.find_all('script'))})
