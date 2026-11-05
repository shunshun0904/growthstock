#!/usr/bin/env python3
"""Executed notebook -> self-contained, script-free HTML report for E123."""
from pathlib import Path
import json
import os
import re
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from jupyter_client.kernelspec import KernelSpecManager

P = Path(sys.argv[1]).resolve()
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {"name": "growthstock-exit", "display_name": "Growthstock exit research", "language": "python"}
nb.cells = [md('''# 8モデルを使った上昇余地・売却戦略の検証

**2026年11月5日集計／最大20営業日／本番変更なし**

推論日の翌営業日の寄りで購入した場合に、どこまで上がるか、いつ売るかを検証しました。
現行の元指標追加版8モデルに付属する学習期間外予測を利用しています。6分類モデルの生スコアを上昇確率とは解釈しません。

## 今回分かったこと

- 基準3モデルがすべて過去分布の95百分位以上だった後続期間の**52件・36日・49銘柄**では、20営業日内の高値が+5%に到達した割合は**76.9%**、+8%は**61.5%**でした。
- 到達した銘柄に限る到達日の中央値は、+5%が**購入から2営業日目**、+8%は**3営業日目**。未到達銘柄はこの日数の中央値に含みません。
- 同じ52件で、往復コスト0.2%を引いた平均損益は、**+5%利確 +1.10%、+8%利確 +0.23%、+10%利確 −0.43%、20日目まで保有 −3.50%**でした。指値に届かなければ20日目終値で売却。ここには損切りを加えていません。
- ただし+5%利確の平均損益の95%区間は**−0.04%〜+2.45%**。+10%利確との差の区間もゼロを含みます。優位性は確定していません。
- 前半の設定選択期間では、同じ強い買い条件を満たすのが**5件**しかありませんでした。事前の最低30件に届かず、売りの最適設定は選べていません。後続期間でよかった設定を採用することはしていません。
- 8モデルから銘柄別の上昇余地を推定する試作も実施。上昇幅の誤差は少し改善しましたが、**売却日の推定は単純な過去平均を上回りませんでした**。現状は研究用です。

**結論：小さめの利確幅を検証する余地はありますが、「高い閾値なら安全に5〜8%取れる」「最適な売却日を予測できた」とは言えません。**
'''), code('''from pathlib import Path
import json,io
import pandas as pd,numpy as np
from IPython.display import display,HTML,Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path.cwd()
pd.set_option('display.max_colwidth',120)
def table(d):
    display(HTML(d.to_html(index=False,max_rows=None,na_rep='—',float_format=lambda x:f'{x:,.2f}')))
def read(name):return pd.read_csv(P/name)
op=read('opportunity_by_threshold.csv')
ex=read('exit_comparison.csv')
selection=json.loads((P/'selection_frozen.json').read_text())
audit=json.loads((P/'data_audit.json').read_text())
signals=pd.read_parquet(P/'signal_paths.parquet')
base=read('holdout_trades_current_tp10_day20.csv')
sample=signals.loc[base.id]
display(HTML('<p><b>評価対象期間：</b>2026-05-11〜2026-10-05の判断日。株価は2026-11-05まで。<br><b>約定：</b>次の市場営業日の寄り。購入日を1日目として最長20営業日。</p>'))
'''), md('''## どのように8モデルを使ったか

1. +10%分類のLightGBM・XGBoost・CatBoost、+15%分類の同3モデルで強い買い条件を比較します。閾値95／97.5／99は、**2025年9月5日以前の学習期間外スコア分布を固定した百分位**です。95は成功確率95%という意味ではありません。
2. 日次ランキングの上半分、直近5営業日ランキングの上位20%／10%を加えた条件も比較します。直近5営業日の順位は、その日の候補だけで並べ直さず、古い候補も含む組全体で計算します。
3. 6分類の百分位＋2ランキングの百分位の計8つを使い、過去の似た予測結果から上昇幅・到達率・売却日を推定する方法を試します。

8モデルすべてが強い評価を出す条件は、後続期間で**0〜1件**しかありません。8つの同意を必須にすれば精度が保証されるとは言えません。
各日2件まで、基準3モデルの最低百分位が高い順、同点はLightGBMスコア・銘柄コード順。将来の価格欠損を確認する前に購入候補を選びます。
ランキングを条件や個別推定に使うのは、その日の候補全体にOOFスコアがある日だけです。分類だけの条件には、この順位の制約を加えません。
'''), code('''labels={
'base95':'基準3本 ≥95','base97.5':'基準3本 ≥97.5','base99':'基準3本 ≥99',
'six95':'分類6本 ≥95','six97.5':'分類6本 ≥97.5','six99':'分類6本 ≥99',
'eight95_80':'基準≥95＋厳格≥90＋日次上半分＋5日上位20%',
'eight95_90':'基準≥95＋厳格≥90＋日次上半分＋5日上位10%',
'eight97.5_80':'基準≥97.5＋厳格≥90＋日次上半分＋5日上位20%',
'eight97.5_90':'基準≥97.5＋厳格≥90＋日次上半分＋5日上位10%'}
show=op[op.period.eq('holdout')].copy();show['entry']=show.entry.map(labels)
cols=['entry','signals','n','n_dates','n_codes','hit5_pct','hit8_pct','mfe_mean_pct','mfe_median_pct','mfe_q25_pct','mfe_q75_pct']
table(show[cols].rename(columns={'entry':'買い条件','signals':'候補件数','n':'集計件数','n_dates':'日数','n_codes':'銘柄数',
'hit5_pct':'+5%到達率%','hit8_pct':'+8%到達率%','mfe_mean_pct':'最大上昇幅平均%','mfe_median_pct':'中央値%',
'mfe_q25_pct':'25%点%','mfe_q75_pct':'75%点%'}))
'''), md('''**期間中の最高値は、その価格で売れることを意味しません。** 最大上昇幅の平均・中央値は上昇余地の記述です。
これは実現損益ではなく、将来価格を見て初めて分かる値です。閾値99の件数は3件に減り、+8%到達率も上がっていません。
'''), code('''fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
vals=np.sort(sample.mfe.to_numpy()*100)
ax[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),color='#2b6b9c',linewidth=2)
for t in [5,8,10]:ax[0].axvline(t,color='#8c9eac',alpha=.5,linestyle='--')
ax[0].set(xlabel='Maximum high / next open - 1 (%)',ylabel='Empirical cumulative proportion',title='20-session upside (52 selected signals)')
for t,color in [(5,'#19846d'),(8,'#df8532'),(10,'#456bb1')]:
    curve=[sample[f'hit{t}_day'].le(day).mean()*100 for day in range(1,21)]
    ax[1].plot(range(1,21),curve,label=f'+{t}%',color=color,linewidth=2)
ax[1].set(xlabel='Session after purchase (purchase day = 1)',ylabel='Ever reached (%)',title='Unconditional first-hit frequency',xticks=[1,5,10,15,20],ylim=(0,100))
ax[1].legend();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);plt.close(fig);display(Image(data=buf.getvalue()))
table(pd.DataFrame([{'対象':'基準3本 ≥95','+5%到達率95%下限':show.iloc[0].hit5_ci_low_pct,
'+5%到達率95%上限':show.iloc[0].hit5_ci_high_pct,'+8%到達率95%下限':show.iloc[0].hit8_ci_low_pct,
'+8%到達率95%上限':show.iloc[0].hit8_ci_high_pct}]))
'''), md('''到達率・損益の区間は21市場営業日の移動ブロック・ブートストラップ2,000回、種123。
同じ日の銘柄をまとめ、重なる20日保有の影響を考慮する近似です。短い評価期間のため独立した相場局面は少なく、区間自体にも不確実性があります。
30件未満、または10日未満の部分集合は区間を表示しません。極端に小さい標本の100%到達を確実性とは解釈できません。

## 実行できる売却ルールの比較

下表はすべて**同じ52購入候補**で比較した、1取引あたりの単純平均です。往復コスト0.2%を差し引いています。
利確・損切りがなければ20営業日目終値で売却。現在の+10%指値を参照にしました。
'''), code('''names={'current_tp10_day20':'+10%指値・20日満了','tp5_day20':'+5%指値・20日満了',
'tp8_day20':'+8%指値・20日満了','close_day20':'20日目終値','development_selected':'選択結果（参照へ据え置き）'}
main=ex[ex.period.eq('holdout')&ex.cost_pct.eq(.2)&ex.order.eq('stop_first')&~ex.exit.eq('development_selected')].copy()
main['exit']=main.exit.map(names)
table(main[['exit','n','mean_net_pct','median_net_pct','win_pct','worst_net_pct','avg_days','ci_low_pct','ci_high_pct']].rename(columns={
'exit':'売却方法','n':'件数','mean_net_pct':'平均純損益%','median_net_pct':'中央値%','win_pct':'勝率%',
'worst_net_pct':'最悪取引%','avg_days':'平均保有日数','ci_low_pct':'平均95%下限','ci_high_pct':'平均95%上限'}))
pair=read('paired_exit_comparison.csv');pair['exit']=pair.exit.map(names)
table(pair.rename(columns={'exit':'売却方法','n':'同じ購入候補数','vs_tp10_mean_diff_pct':'+10%指値との差pt','ci_low_pct':'差95%下限','ci_high_pct':'差95%上限'}))
'''), md('''+5%到達率76.9%は、損切りなしで20日以内の高値を観察した割合です。
先に損切りすれば、その後+5%に到達した銘柄でも利益を取れません。未到達時の損失は利益幅より大きくなるため、到達率だけでは判断できません。

### 損切り・トレーリングの感応度（探索的な比較）

前もって定めた72ルールの後続期間の全数値はCSVに保存しています。下表は20日上限での代表例です。
**この表の後続期間の成績を使って最適な設定を選び直していません。**

同じ日足で利確と損切りの双方に到達した場合、主集計は損切りを先に適用します。寄り付きで既に条件を満たす場合は寄り付きの順序を優先。
損切り価格より低く寄ったら、その寄り付き価格で約定する仮定です。利確以上で寄っても指値価格で約定する保守的な仮定です。
トレーリングは前日までの終値最高値を使い、+5%の終値到達の翌日から有効。同じ日の高値で当日の損切り水準を変えません。
'''), code('''grid=read('holdout_exit_grid_exploratory.csv')
pick=['tp5_sl0_d20','tp5_sl3_d20','tp5_sl5_d20','tp5_sl8_d20','tp8_sl5_d20','trail3_d20','trail5_d20']
z=grid[grid.exit.isin(pick)].copy()
table(z[['exit','n','mean_net_pct','win_pct','worst_net_pct','avg_days','ambiguous_n','target_first_net_pct']].rename(columns={
'exit':'ルール（tp利確/sl損切り/d日数）','n':'件数','mean_net_pct':'主集計の平均純損益%','win_pct':'勝率%',
'worst_net_pct':'最悪取引%','avg_days':'平均日数','ambiguous_n':'日中順序不明件数','target_first_net_pct':'利確先行時の平均%'}))
'''), md('''### コストと同時保有数

コストは1取引あたり買値の0／0.2／0.5／1.0%を一括で引く感応度です。銘柄ごとの実スプレッド・手数料・流動性を推定した値ではありません。
また、候補全体とは別に、既存運用に近い「同時3銘柄・同銘柄の重複保有なし」の診断を載せます。
売却当日の寄りにその枠を再利用しない保守的な扱いです。売却ルールごとに購入できる銘柄が変わるので、こちらは同一銘柄比較ではありません。
'''), code('''costs=ex[ex.period.eq('holdout')&ex.order.eq('stop_first')&~ex.exit.eq('development_selected')].pivot(index='exit',columns='cost_pct',values='mean_net_pct')
costs.index=costs.index.map(names)
table(costs.rename_axis('売却方法').reset_index().rename(columns={0.0:'コスト0%',.2:'0.2%',.5:'0.5%',1.0:'1.0%'}))
slots=ex[ex.period.eq('holdout_3slots')&~ex.exit.eq('development_selected')].copy();slots['exit']=slots.exit.map(names)
table(slots[['exit','n','mean_net_pct','win_pct','worst_net_pct','avg_days']].rename(columns={'exit':'売却方法','n':'実行対象件数','mean_net_pct':'平均純損益%','win_pct':'勝率%','worst_net_pct':'最悪取引%','avg_days':'平均日数'}))
'''), md('''これらは独立した購入シグナルごとの統計です。複利収益・年率リターン・ポートフォリオの最大ドローダウンではありません。
配当・税・板・値幅制限による注文不能・実際のスリッページは未反映です。

## 銘柄別の上昇余地と売却日の推定

6分類モデルの固定百分位と2ランキングの当日内百分位を0〜1にそろえ、8変数の距離が近い過去の高評価サンプルを参照します。
その**平均最大上昇幅、中央値、25%点・75%点、+5%／+8%到達率、5／10／20日以内の到達率**を出す試作です。
参照数K=50／100／200を前半期間だけで比較し、最大上昇幅の二乗誤差でK=50を選択しました。ただし前半の予測評価は4件で、選択の信頼性は低いです。

順位を候補全体で再現できない日の前半1件・後半3件を個別推定から除外し、後続評価は49件です。
上の分類モデルだけによる売却比較52件とは対象が異なります。

月ごとに、月初より前に20営業日分の価格が確定したサンプルだけを参照します。将来の結果を参照先にしません。
売却日は近傍サンプルの終値リターン平均が最大になる1〜20日目を**買う前に固定**し、実際のその日終値で評価しました。
後から分かる最高値の日を、予測した売却日として採用していません。
'''), code('''table(read('estimator_tuning.csv').rename(columns={'k':'参照数K','n':'前半評価件数','mfe_rmse_pct':'最大上昇幅RMSE(pt)','baseline_rmse_pct':'全参照平均RMSE(pt)'}))
f=read('upside_estimator_evaluation.csv');f=f[f.cohort.eq('base95')]
table(f[['n','k','mfe_rmse_pct','baseline_rmse_pct','forecast_mean_mfe_pct','realized_mean_mfe_pct',
'mean_suggested_day','forecast_day_net_pct','baseline_day_net_pct','fixed20_net_pct']].rename(columns={
'n':'後続評価件数','k':'K','mfe_rmse_pct':'上昇幅RMSE(pt)','baseline_rmse_pct':'単純平均RMSE(pt)',
'forecast_mean_mfe_pct':'予測上昇幅平均%','realized_mean_mfe_pct':'実際の最大上昇幅平均%',
'mean_suggested_day':'予測売却日の平均','forecast_day_net_pct':'予測日で売却の純損益%',
'baseline_day_net_pct':'全参照平均の日で売却%','fixed20_net_pct':'20日目売却%'}))
table(read('hit_probability_calibration.csv').rename(columns={'target':'到達目標%','probability_band':'推定確率帯','n':'件数','predicted_pct':'推定到達率%','actual_pct':'実際の到達率%'}))
'''), md('''最大上昇幅の誤差は19.07pt、単純平均の19.82ptより小さいものの、誤差は大きく残っています。
予測した日に売ると平均−2.06%、銘柄ごとに分けず過去全参照の平均から決めた日で売ると+1.16%でした。
上昇余地の推定を、そのまま指値や最適な売り時に変換するのはまだ不十分です。
確率帯別の件数も少なく、表示確率の校正を確立したわけではありません。

### 現在日の試算

2026年11月5日の実際の推論出力にも同じ条件を適用しました。今回は基準3本がすべて固定95百分位以上になる銘柄が**0件**なので、銘柄別の試算は出していません。
評価件数を増やすために現在日の閾値を下げていません。将来表示する場合も、買値は翌日の寄りなので、前日終値を買値と取り違えず比率で提示する設計です。
'''), code('''live=json.loads((P/'live_input_identity.json').read_text())
table(pd.DataFrame([{'推論日':live['asof'],'強い買い条件を通過':live['high_signal_n'],'銘柄別試算件数':live['shown_top2_n']}]))
'''), md('''## 設定選択と評価をどう分けたか

- 分布の基準と8モデルのチューニングは2025-09-05以前。その対象の20会社営業日目が最後に確定するのは2025-10-07で、その翌日以降を評価可能としました。
- 前半：2025-10-08〜2026-03-31。後半：2026-05-11〜2026-10-05。間は前半のラベル確定と学習境界のために空けます。
- 売り側は利確なし／5／8／10%、損切りなし／3／5／8%、期限5／10／15／20日、トレーリングの計72ルール。
  買い条件10通りのうち取引があった7通りと合わせて504組を前半で集計しました。
- 事前の最低30取引・10判断日を満たす条件がなく、選択を見送り、参照ルール+10%・20日のまま固定しました。
  後半の+5%の成績は**新しい仮説を作るための観察**です。最適化に成功した結果とは報告しません。
- 既存の学習境界は20営業日を29暦日で近似しています。会社別の20日目を再計算し、訓練ラベルの確定前・確定当日の予測を除きました。
  例えば2025-11-04の予測24件、2026-05-07の予測43件を除外しています。
- 8つの基礎モデルは依頼どおり固定しました。売り側を追加する今回の検証で、基礎モデルまで別の設定に変更すると、売り方の効果とモデル変更の効果が混ざるためです。

**この歴史期間は、以前の特徴量比較で既に参照されています。** 今回の前半・後半分割は売り側の設定選択を分けたものです。
プロジェクト全体として完全に未使用の将来データでの検証ではありません。本番採用には前向きな記録と追加評価が必要です。
'''), code('''dev=op[op.period.eq('development')][['entry','n','n_dates']].copy();dev['entry']=dev.entry.map(labels)
table(dev.rename(columns={'entry':'前半の買い条件','n':'件数','n_dates':'日数'}))
table(read('fold_maturity_audit.csv').tail(4))
'''), md('''## データ・実装の検査記録

8つのOOFファイルすべてのSHA256が、現在の本番メタデータに記載されたハッシュと一致しています。
現行モデルは11月4日学習の元指標追加版。分類と日次LTRは373列、5日LTRは377列です。
5日LTRの**実際に使ったモデルは5段階**で、11月8日以降向けの10段階設定と混ぜていません。

価格は分割・併合係数からOHLCを全期間で再調整しました。取引カレンダーで購入日と20日目を固定し、寄りの欠測を翌日へずらしません。
54行は20日間のOHLCに欠測があるため診断に残し、完全な経路を要する集計から除外しました。
利益が早く確定した行だけを残して、結果待ちの未到達行を落とす処理はしていません。今回の強い条件の前半5件・後半52件には除外行はありません。

マート全体の最新版を直接取得できなかったため、前回の固定マート22,110行と今回のOOFの追加11キーを合わせた22,121キーで学習境界を監査しました。
件数は現在の学習件数と一致しますが、**最新版の全特徴量値の完全照合まではしていません**。
5日候補を再構成すると学習窓に入る範囲で20日後の会社日足が存在しないキーが7件あります。これらは元の学習でもR20が有限にならず、学習対象外になる行です。
また、学習期間外のランキングにも将来R20が欠けた候補の除外があり、今回の前半・後半の16判断日では候補全体に対する順位を再現できません。
その日はランキングの百分位を不明とし、8モデル条件・近傍の参照学習・個別推定に使いません。分類のみの比較結果は変わりません。
この制約は履歴予測の再現性のためで、実運用で将来の欠測を予知して見送れるという意味ではありません。除外日を含む個別推定の成績は未評価です。
将来の株価調整の訂正履歴やポイントインタイムの完全性も、この検査だけで保証するものではありません。
'''), code('''table(read('model_identity.csv')[['model','features','rows','trained_at','train_to','matches_production_meta']])
table(pd.DataFrame([{'OOF件数':audit['historical_oof_rows'],'完全な価格経路':audit['complete_paths'],
'OHLC経路不完全':audit['path_status'].get('incomplete_ohlc',0),'8出力不足または順位不確定':audit['all8_missing'],
'前半の全候補':audit['period_rows']['development'],'後半の全候補':audit['period_rows']['holdout']}]))
'''), md('''## 保存したものと次に確かめること

実行済みNotebook、外部スクリプト不要のHTML、全閾値の集計、72売却ルールの比較、個別の予測と実績、事前条件・選択記録、入力ハッシュを保存しました。
損切り・同日約定順序・コスト・同時保有の仮定を変えた値もCSVで追えます。本番の予測モデルと売買条件は変更していません。

次の検証候補は、今回の+5%利確・最大20日を前向きに記録し、損切りによる損失抑制と利確前の離脱の両方を評価することです。
上昇幅そのものに加えて「+5%に先に到達するか、損切りに先に到達するか」をモデル化する余地があります。
今回の銘柄別タイミング推定には採用を支える結果が得られていないため、精度を主張して本番に加える段階ではありません。
''')]

# Keep emphasis readable at Japanese word boundaries in both renderers.
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        cell.source = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cell.source, flags=re.S)

# Explicitly use this interpreter, never whichever default kernel is installed.
kernel_dir = P / "_kernel"
kernel_dir.mkdir(exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"Growthstock exit research","language":"python"}))
manager = KernelSpecManager()
manager.install_kernel_spec(str(kernel_dir),kernel_name="growthstock-exit",prefix=str(P / "_jupyter"),replace=True)
old = os.environ.get("JUPYTER_PATH", "")
os.environ["JUPYTER_PATH"] = str(P / "_jupyter/share/jupyter") + (os.pathsep + old if old else "")
nb = NotebookClient(nb,timeout=120,kernel_name="growthstock-exit",resources={"metadata":{"path":str(P)}}).execute()
notebook_path=P / "growthstock_exit_upside.ipynb"
nbformat.write(nb,notebook_path)
exporter=HTMLExporter(template_name="lab")
exporter.exclude_input=True
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb)
soup=BeautifulSoup(body,"html.parser")
for tag in soup.find_all("script"):
    tag.decompose()
for tag in soup.find_all("link",href=True):
    if tag['href'].startswith(('http','//')):tag.decompose()
soup.title.string='Growthstock：上昇余地と売却戦略の検証（2026-11-05）'
style=soup.new_tag('style')
style.string='''body{background:#f2f6f9;color:#172e43;font-family:system-ui,"Noto Sans CJK JP",sans-serif}main{max-width:1280px;margin:auto;background:white;padding:24px}h1,h2,h3{color:#174b70}h1{font-size:30px}p,li{line-height:1.85}table{font-size:12px!important;border-collapse:collapse;width:max-content;max-width:100%}th{background:#e9f2f7!important;color:#174b70}td,th{padding:8px!important;border-bottom:1px solid #dce6ee;text-align:right}td:first-child,th:first-child{text-align:left}div.jp-OutputArea-output{overflow-x:auto}.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt{display:none}img{max-width:100%}a{color:#176c9d}'''
soup.head.append(style)
html_path=P / 'growthstock_exit_upside.html'
html_path.write_text(str(soup))
errors=[o for cell in nb.cells if cell.cell_type=='code' for o in cell.get('outputs',[]) if o.output_type=='error']
if errors:raise RuntimeError(errors)
print(json.dumps({'notebook':str(notebook_path),'html':str(html_path),'bytes':html_path.stat().st_size,
                  'code_cells':sum(c.cell_type=='code' for c in nb.cells),'tables':len(soup.find_all('table')),
                  'scripts':len(soup.find_all('script')),'images':len(soup.find_all('img'))},ensure_ascii=False))
