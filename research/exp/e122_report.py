#!/usr/bin/env python3
"""Executed notebook and offline HTML for the valuation availability audit."""
from pathlib import Path
import sys,json,re
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

P=Path(sys.argv[1]).resolve()
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'name':'python3','language':'python','display_name':'Python 3'}
def md(s):return nbformat.v4.new_markdown_cell(s)
def code(s):return nbformat.v4.new_code_cell(s)
nb.cells=[md('''# 未実装の企業価値評価手法：算出率・仮定・問題点

2026-11-04。分母は学習マートの**22,110レコード（銘柄×候補日）、3,748銘柄**です。
候補日は2018-04-03〜2026-10-02、保存済みデータのスナップショットは2026-11-02です。
算出率は現在の保存データと下記条件による実測値です。方法一般の理論的な上限や将来データ追加後の充足率ではありません。

**算出できたことは、入力の完全性・価格の正確さ・特徴量の有効性を保証しません。**
本番モデル・パラメータ・契約プランは変更していません。これら新手法の学習追加・バックテストも今回実施していません。

## 算出率
'''),code('''from pathlib import Path
import json,pandas as pd,numpy as np
from IPython.display import display,HTML
P=Path.cwd()
def table(d):display(HTML(d.to_html(index=False,max_rows=None,float_format=lambda x:f'{x:,.3f}')))
cov=pd.read_csv(P/'coverage.csv')
names={'per':'同業・実績PER','pbr':'同業PBR','ev_ebitda':'同業EV/EBITDA近似',
'dcf_cash':'DCF：CF明細の設備投資','dcf_flex':'DCF：設備投資額の代用も許容',
'reverse_cash':'リバースDCF：CF明細','reverse_flex':'リバースDCF：設備投資額の代用も許容',
'mc_cash':'仮定型・確率的DCF：CF明細','mc_flex':'仮定型・確率的DCF：代用も許容',
'asset_stress':'簿価の下方ストレス（参考）','adjusted_nav_market':'時価に基づく修正純資産法'}
order=['per','pbr','ev_ebitda','dcf_cash','dcf_flex','reverse_cash','reverse_flex','mc_cash','mc_flex','adjusted_nav_market','asset_stress']
z=cov.set_index('method').reindex(order).reset_index()
summary=z[['method','input_rows','calculated_rows','calculated_pct','positive_value_rows','positive_value_pct','codes']].copy()
summary['method']=summary.method.map(names)
summary.columns=['手法','入力条件を満たす行','算出できた行','全行に対する算出率(%)','中央値が正の行','その割合(%)','算出銘柄数']
table(summary)
summary.to_csv(P/'comparison_summary.csv',index=False)
'''),md('''「算出」は数値が求まった行で、ゼロ評価を含みます。正の中央値の行を別列に示しました。
リバースDCFの出力は成長率なので「正の価格」は適用外です。修正純資産法の0%は**計算に必要な時価情報が不足している**意味で、企業価値0円の意味ではありません。
手法や代用版の対象は重複しており、件数を合算しません。未算出行をゼロや同業平均で埋めていません。

**負債の完全性に留意が必要です。** DCF・EV/EBITDA・逆算・MCの算出行に、前回の負債原本照合済みフラグが付いた行はありません。
今回の有利子負債は観測項目の合計を暫定値としています。未取得項目がゼロと確認できたわけではなく、負債の取り漏れは価値やWACCを歪めます。
従って以下は仮定付きの計算候補率で、本番で使える品質の認定率ではありません。

## 1. DCF法：どのCFをどう予測したか

日本基準・非金融を対象とし、通常の約1年の決算、既知の非支配持分、株主資本の範囲、貸借一致、負債が総負債以下であること等を確認しました。
売上・営業利益が正で、営業利益率が100%以下の会社に限定しています。赤字会社の回復を勝手に仮定していません。
IFRS/USGAAPはリース・償却・投資の整合を別途確認する必要があるため今回は除外しています。

将来5年間のCFを次のドライバーから作ります。

- FCFF＝営業利益×(1−税率)＋減価償却−設備投資−運転資本増加。
- 運転資本は売上債権＋棚卸資産−買入債務の近似。その他の営業資産負債は含めていません。
- 営業利益率、償却/売上、投資/売上、運転資本/売上を5年間固定。売上成長率を−5/0/5/10%で振ります。
  会社の事業計画・会社予想を採用したものではありません。景気循環、M&A、一時的投資、設備能力・規模の変化を表現しきれません。
- **CF明細版**は有形＋無形の取得支出（双方が既知）、または両者合計の取得支出を使用。支出の負号を反転し、欠損をゼロにしません。
- **代用許容版**はCF明細がないときだけ、開示された設備投資額capexを使います。現金支出と投資計上額は、リース・未払・取得時点などで異なります。
  「営業CF＋投資CF」をそのままFCFFとする計算はしていません。

株主資本コストは既存のCAPM入力。ERPを3/5/7%、借入スプレッドを0.5/1.5/3%、税率を25/30/35%、終価成長率を0/1/2%とした**324シナリオ**です。
税率は研究上の仮定で、会社別の法定・限界税率を確定したものではありません。
WACCは親会社時価総額＋非支配持分簿価を株式価値側、観測した有利子負債簿価を負債側とする加重平均です。
非支配持分の時価とリスクを親会社と同じ株主資本コストで近似し、負債簿価を時価の代理としています。
借入金利は円国債利回り＋スプレッドで、会社別の実際の限界調達金利を推定したものではありません。

終価では**安定期ROIC＝WACC、再投資額＝終価成長率/ROIC×NOPAT**とし、成長だけ増やして再投資を無視する計算を避けています。
安定期の超過利益がなくなる仮定なので、競争優位が長く続く会社を過小評価し得ます。5年後に投資構造が安定期へ切り替わる強い仮定でもあります。
企業価値＝5年のFCFF現在価値＋終価現在価値。親会社価値＝企業価値−負債＋現金−非支配持分。
現金を全額非事業資産とみなして加算しており、必要運転現金・拘束現金の切り分けは未実施です。
API時価総額は自己株控除後・百万円表示なので円へ直し、当日株価との比から株価単位を整合させています。

WACC−終価成長率が1%以上の有効シナリオが80%以上ある行だけ統計を算出します。
残余株主価値が負の場合は価格を0にし、**そのゼロを分布に残します**。これは倒産・増資・株式のオプション価値のモデルではありません。
モデル上の0円を実際の株価0円と解釈しません。下限・上限・平均・中央値・母標準偏差は仮定差による感応度です。

**妥当性の判断：感応度分析の試作には使えますが、将来計画・借入条件・負債明細・再投資の照合前に本源的価値の確定値とは呼べません。**

## 2. PER・PBR・EV/EBITDAによる比較

同じ候補日・東証33業種の、他の正の倍率がある銘柄を最低10社使用します。自社を除外し、学習候補に選ばれた銘柄だけを同業とする選択バイアスを避けています。
過去の業種履歴を使い、未来の業種から埋めません。業種情報は最大62暦日前まで、日次倍率は同じ日です。

- PER/PBR参考価格＝当日株価×同業倍率/自社倍率。同じ提供元の実績PER（TTM）・最新四半期BPSのPBRをそろえて使います。
- EV/EBITDAでは**EBITDA＝営業利益＋減価償却の近似**です。償却項目が全ての無形資産・のれん償却を含むとは照合できていません。
  企業価値倍率＝(時価総額＋負債＋非支配持分−現金)/EBITDA。
  同業倍率×自社EBITDAから負債・非支配持分を引き、現金を戻して株価へ換算します。株価をEV倍率の比だけで比例させてはいません。

各方法で同業から得た価格集合の最小・最大・平均・中央値・母標準偏差を保存し、外れ値の影響を見やすくする10/90%点も出します。
外れ値を事後クリップしていないので、特にPERの上限・平均・標準偏差は小さい利益の会社に引っ張られ得ます。
同業数5/10/20社の入力候補率も参考に示しますが、主表の条件は10社のままです。
'''),code("table(pd.read_csv(P/'peer_count_sensitivity.csv'))\ntable(pd.read_csv(P/'peer_universe_checks.csv'))"),md('''**妥当性の判断：相対的な割安・割高の候補指標としては算出しやすいですが、本源的価値とは異なります。**
市場全体・業種全体の割高をそのまま引き継ぎ、成長率・ROE・規模・財務リスク等の差も調整していません。
EV/EBITDAの同業は財務データのそろった会社へ偏り、決算期・年次/TTM・リース・償却・負債範囲の不一致も残ります。
損失会社のPERや負の純資産のPBRを、意味のある正の倍率へ勝手に置換していません。

## 3. 修正純資産法

土地・建物・投資不動産・非上場株などの銘柄別時価、簿外債務、処分税・費用まで一貫してそろっていません。
そのため**時価に基づく修正純資産の算出率は0%**です。現在の簿価だけから含み益を作ってはいません。

別枠の参考として、簿価純資産から有形固定資産・棚卸資産を0/20/40%、無形資産を0/50/100%減額する27通りの下方ストレスを作りました。
欠損項目をゼロとせず、貸借一致と非支配持分を確認した行だけが対象です。
**これは修正純資産の時価推定や、清算価値の下限ではありません。** 調整率に会社別の実証根拠がなく、処分費用・換金性・隠れた債務・税効果も足りません。

## 4. リバースDCF

現在の時価総額＋負債＋非支配持分−現金に一致するよう、上記DCFの**今後5年間の年率売上成長率**だけを逆算しました。
ERP5%、借入スプレッド1.5%、税率30%、終価成長率1%、利益率等を固定し、年率−50〜＋50%の範囲の解を求めています。
このモデルは成長率の5次多項式になるため、数値的に全根を求め、範囲内の実数解が1つだけの行を採用します。
解なし・複数解を便宜的な成長率で埋めず、現在の事業価値が非正の場合等も除外します。
'''),code("table(pd.read_csv(P/'reverse_diagnostics.csv'))"),md('''**妥当性の判断：市場価格を説明する条件付き期待の指標にはなりますが、市場の成長予想を一意に観測したものではありません。**
利益率や資本コストを変えても現在株価を説明できるため、逆算値は固定した条件に依存します。
価格を入力して価格を再現する手法なので、独立した理論価格の予測ではありません。特徴量にするなら逆算成長率やその感応度です。

## 5. 確率的DCF

各レコード1,024経路のMonte Carlo試算を実装しました。CF入力はDCFと同じです。欠損が確率モデルで埋まるわけではありません。

| 入力 | 今回置いた分布 |
|---|---|
| 5年の売上成長率 | 三角分布：−5%、最頻0%、上限10% |
| ERP | 三角分布：3%、最頻5%、上限7% |
| 借入スプレッド | 三角分布：0.5%、最頻1.5%、上限3% |
| 税率 | 一様分布：25〜35% |
| 営業利益率 | 基準利益率の80〜120%の三角分布。上限は100% |
| 終価成長率 | 0〜min(2%, WACC−1%)の条件付き一様分布 |

終価成長率を除く入力は独立、各経路のパラメータは5年間持続すると仮定しました。設備投資・償却・運転資本の対売上比も固定です。
DCFの固定格子とMCでは終価成長率の許容の仕方が違うため、算出率の差を手法の優劣とは解釈しません。
DCF格子で低いWACCのシナリオが不適格になる行も、MCでは終価成長率を条件付きで狭めることで計算されます。
負の残余株主価値は0円として残し、有効経路の80%以上がそろう行で統計を出します。マイナスの経路を捨てて平均を引き上げてはいません。

**妥当性の判断：分布の形・幅・相関を企業別に校正しておらず、価格分布は「仮定した分布からの試算」です。**
実際の将来価格の確率分布、株価上昇確率、95%信頼区間とは呼べません。5/95%点も、この仮定分布の分位点です。
正の基準利益率の80%以上という分布なので、営業赤字へ転落する経路も含んでおらず、下方リスクを過小評価し得ます。
経済変数の相関、景気・レジーム転換、M&A、資金調達・倒産、β推定誤差、モデル自体の誤差を十分に表現していません。
最小・最大は抽出経路数にも依存します。現段階では確率値より感応度の候補として扱います。

## 仮定と過去実績の比較

DCF代用許容版の入力条件を満たした行で、比較可能な過去データを集計しました。
同じ銘柄・会計年度が複数の候補日に現れるため、以下はレコード加重の記述統計で、独立標本数ではありません。
**実績の変動は予測誤差ではなく、この表で将来分布を校正・検証したことにはなりません。**
実効税率と限界税率、既存債務の平均負担と将来の限界調達金利も別物です。
'''),code('''a=pd.read_csv(P/'assumption_evidence.csv')
labels={'historical_sales_growth':'過去の売上成長率','historical_margin_relative_change':'過去の利益率の相対変化',
'historical_effective_tax_rate':'過去の実効税率','historical_interest_over_ending_debt':'支払利息/期末負債小計',
'disclosed_capex_over_cash_capex':'設備投資額/現金取得支出','annual_NWC_ratio_change_pp':'運転資本/売上の変化',
'terminal_PV_ratio_cash':'終価依存度：CF明細版','terminal_PV_ratio_flex':'終価依存度：代用許容版',
'reverse_growth_ERP3_to7_width_pp_cash':'逆算成長率のERP感応度：CF明細版',
'reverse_growth_ERP3_to7_width_pp_flex':'逆算成長率のERP感応度：代用許容版'}
for i,r in a.iterrows():
    percent=r['check'].startswith(('historical_','terminal_'))
    if percent:a.loc[i,['median','p05','p95']]*=100
    a.loc[i,'unit']='%' if percent else ('倍' if 'capex_over' in r['check'] else 'ポイント')
a['check']=a['check'].map(labels)
table(a[['check','rows','median','p05','p95','unit','within_assumed_range_pct']].rename(columns={
'check':'検査','rows':'対象行','median':'中央値','p05':'5%点','p95':'95%点','unit':'単位','within_assumed_range_pct':'比較基準内(%)'}))
'''),md('''売上成長率の範囲に収まる実績は約57%、利益率±20%に収まる実績は約56%でした。
固定された成長・利益率の範囲が十分広いとは判断できず、とりわけ確率モデルの分散を過小評価する懸念があります。
設備投資額と現金支出が±20%以内に一致したのは、比較できた904レコードの約74%です。代用の同一性は確認できていません。
正のDCF中央値が出た行でも、約75%は終価現在価値が企業価値の8割を超え、終価の仮定に強く依存していました。
ERPを3%から7%へ変えたとき、逆算売上成長率の差の中央値は約17パーセントポイントでした。

## 欠損・入力品質

理由は重複しています。「CF明細がない」には有形・無形の片方だけ不明な行も含み、その片方を0とはしていません。
負債小計があっても完全性の確認済みではありません。以下の算出率は財務原本の監査済み率ではありません。
'''),code('''q=pd.read_csv(P/'input_quality.csv')
labels={'no_market_cap':'時価総額なし','no_sector':'過去の業種なし','no_fresh_annual':'新しい年次財務なし・古い',
'unknown_period':'通常年度の確認不可','not_JP_basis':'日本基準・範囲の確認不可（未取得含む）','financial_or_other':'金融・その他',
'no_cash_capex':'取得支出の有形・無形明細不足','no_flex_capex':'明細・代用投資額の両方不足','no_DA':'償却値なし',
'no_debt':'負債小計なし','unreviewed_debt_subtotal':'負債小計の原本照合未了','no_NCI':'非支配持分不明',
'equity_scope_risk':'純資産範囲の疑義','balance_not_reconciled':'貸借整合の確認不可','negative_or_missing_operating_profit':'正の営業利益なし（未取得含む）',
'missing_WC_components':'運転資本の構成項目不足','capex_definition_conflict':'設備投資明細の矛盾'}
q['check']=q['check'].map(labels)
table(q.rename(columns={'check':'検査','rows':'該当行','pct':'全行割合(%)'}))
table(cov[['method','reviewed_debt_rows','capex_note_proxy_rows']])
'''),md('''## 年別算出率

会計項目・提出日・履歴の充足に年差があります。未取得企業・年度を分母から落としていません。
'''),code("table(pd.read_csv(P/'coverage_by_year.csv'))"),md('''## 出力値の要約統計

価格の単位は円、逆算成長率は年率の比率（0.1＝10%）です。平均や上限が極端な相対倍率は、そのままモデル採用すべきではありません。
'''),code("table(pd.read_csv(P/'feature_summary.csv'))"),md('''## 検証の記録

元マートの全キー・順序を維持し、算出率を行別結果から再集計しました。
DCFは実データ80行を独立した多項式の現在価値計算と照合し、逆算価格の残差も確認しています。
MCは実データ10行で保存結果を再現しました。元データと実装のSHA256も保存しています。
'''),code("table(pd.DataFrame([{'item':k,'value':v} for k,v in json.loads((P/'numeric_verification.json').read_text()).items()]))"),md('''株式分割の単位は当日株価とAPI時価総額の比で整合させています。APIは時価総額を自己株控除後と定義する一方、株数等の詳細計算仕様は非公開です。
過去の業績開示は翌営業日から反映される原則がありますが、APIの誤り訂正等によりデータが上書きされる場合があり、取得時刻別の原版を独立に復元できたわけではありません。
EDINET DBについても、全銘柄・全項目の原本照合と、過去訂正履歴・項目の完全性の確認は未完了です。

## 一次資料

- [J-Quants：指標の定義・TTM・株数・時価総額単位](https://jpx-jquants.com/en/spec/eq-valuation/calc)
- [J-Quants：翌営業日反映と訂正方針](https://jpx-jquants.com/en/spec/eq-valuation)
- [NYU Damodaran：FCFFとWACC](https://people.stern.nyu.edu/adamodar/pdfiles/valn2ed/ch15.pdf)
- [NYU Damodaran：終価の整合性](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm)
- [CFA Institute：現在価値・倍率・資産評価モデル](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/equity-valuation-concepts-basic-tools)
- [NYU Damodaran：シミュレーションと変数間の相関](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/distresspaper.htm)
''')]
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(P)}}).execute()
nbformat.write(nb,P/'report.ipynb')
html,_=HTMLExporter(template_name='classic',exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(nb)
s=BeautifulSoup(html,'html.parser')
for el in s.find_all('script'):el.decompose()
for el in s.find_all('link'):
    if str(el.get('href','')).startswith(('http:','https:','//')):el.decompose()
for el in s.find_all('style'):el.string=re.sub(r'@import[^;]+;','',el.get_text())
(P/'report.html').write_text(str(s),encoding='utf-8')
print(json.dumps({'executed_cells':sum(c.cell_type=='code' for c in nb.cells),'static_tables':len(s.find_all('table'))}))
