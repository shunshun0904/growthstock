#!/usr/bin/env python3
"""Execute a result-reading Notebook and export entirely static, embedded HTML.

Reads an already completed evaluation; does not select parameters or rerun
held-out evaluation. No browser-side data loading or calculation is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


def build_report(result_dir):
    import pandas as pd
    import nbformat as nbf
    from jupyter_client import KernelManager
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import ExecutePreprocessor

    root = Path(result_dir).resolve()
    result = json.loads((root / 'summary.json').read_text())
    if result.get('status') != 'completed':
        raise ValueError('final evaluation is incomplete')
    records = pd.read_parquet(root / 'evaluation_records.parquet')
    quantiles, support = [], []
    for (phase, side), d in records.groupby(['phase', 'side']):
        for method in ['baseline', 'candidate']:
            errors = 100 * (d[f'{method}_mean'] - d.actual_mean) / d.actual_mean
            quantiles.append({'phase': phase, 'side': side, 'method': method,
                              'median_absolute_error_percent': float(errors.abs().median()),
                              'p90_absolute_error_percent': float(errors.abs().quantile(.9)),
                              'p99_absolute_error_percent': float(errors.abs().quantile(.99)),
                              'max_absolute_error_percent': float(errors.abs().max())})
        support.append({'phase': phase, 'side': side, 'records': len(d),
                        'outside_support_records': int(d.outside_60_day_price_support.sum()),
                        'outside_support_percent': float(100 * d.outside_60_day_price_support.mean()),
                        'split_window_records': int(d.split_in_window.sum()),
                        'zero_volume_window_records': int((d.zero_volume_days > 0).sum())})
    pd.DataFrame(quantiles).to_csv(root / 'error_quantiles.csv', index=False)
    pd.DataFrame(support).to_csv(root / 'support_diagnostics.csv', index=False)

    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md('''
    # 価格帯別需給の補助検証：平均建値の推定

    **承認済み条件に基づく実データの結果です。** 表・図は実行済みNotebookの出力を埋め込んでいます。
    HTMLを開くだけで確認でき、外部データの読み込みやJavaScriptによる集計は必要ありません。

    検証対象は、東証普通株の信用買い・信用売りの**平均建値**です。
    価格帯別の残存建玉そのものの正解分布は入手できていないため、
    **分布の形状・各価格帯の残高の精度を確認した実験ではありません。**
    本番のモデル・需給表示・公開サイトへの変更はありません。
    ''')
    code('''
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from IPython.display import Image
    from io import BytesIO
    def display_plot(fig):
        image = BytesIO()
        fig.savefig(image, format='png', dpi=130, bbox_inches='tight')
        display(Image(data=image.getvalue()))
        plt.close(fig)
    ROOT = Path.cwd()
    result = json.loads((ROOT / 'summary.json').read_text())
    config = json.loads((ROOT / 'config.json').read_text())
    metrics = pd.read_csv(ROOT / 'summary.csv')
    curve = pd.read_csv(ROOT / 'training_curve.csv')
    daily = pd.read_csv(ROOT / 'daily_metrics.csv')
    coverage = pd.read_csv(ROOT / 'coverage.csv')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.precision', 5)
    s = result['selection']
    b = result['bootstrap']
    final = metrics[(metrics.phase == 'final') & (metrics.side == 'all')].set_index('method')
    baseline = final.loc['baseline', 'rmspe_percent']
    candidate = final.loc['candidate', 'rmspe_percent']
    delta = candidate - baseline
    direction = '小さくなりました' if delta < 0 else ('同じでした' if delta == 0 else '大きくなりました')
    display(Markdown(f"""
    ## 結果

    係数調整期間で選ばれた共通係数は **k = {s['chosen_coefficient']:.1f}** です。
    最終評価のRMSPEは、出来高だけの推定が **{baseline:.4f}%**、回転率モデルが **{candidate:.4f}%**。
    差（回転率−出来高）は **{delta:+.4f}ポイント** で、誤差は{direction}。

    共通の評価対象は **{result['common_records']:,}件、{result['common_codes']:,}銘柄**
    （調整期間と最終評価期間の合計）。最終評価は **{int(final.loc['baseline','records']):,}件** です。
    正解平均が有効なレコードに対する算出可能率は **{result['coverage_among_valid_targets_percent']:.2f}%**。
    """))
    interval = (f"{b['lower']:+.4f}〜{b['upper']:+.4f}ポイント" if b['lower'] is not None else '算出不可')
    display(Markdown(f"""
    ペア・ブートストラップによる差の95%参考区間は **{interval}** です。
    最終評価は{s['final_dates']}営業日だけで、5日ブロックは実質的な独立区間が少なく、
    将来の長期的な優劣を確定する区間ではありません。
    境界の係数が選ばれたか：**{s['at_search_boundary']}**。探索範囲の拡張や評価後の再調整はしていません。
    """))
    if s['chosen_coefficient'] == 0:
        display(Markdown('**k=0では両手法の計算が同一です。差と参考区間が0になるのはそのためで、独立した2モデルの優劣が高い確度で確定したという意味ではありません。**'))
    ''')
    md('''
    ## データと評価期間

    基準日を丸ごと時系列で分け、前70%で係数を決め、後30%は一度だけ最終評価しました。
    現在取得できる過去データを使っており、後日の訂正が反映される可能性があります。
    当時の訂正前データを完全に再現したバックテストではありません。
    ''')
    code('''
    source = json.loads((ROOT / 'source_manifest.json').read_text())
    collection = source['collection']
    display(pd.DataFrame([
        ['取得日時（UTC）', collection['retrieved_at_utc']],
        ['取得範囲の開始', collection['requested_start']],
        ['最新の公表済み残高基準日', collection['observed_end']],
        ['未公表のため含めない基準日', ', '.join(collection['trailing_unpublished_dates'])],
        ['調整期間', f"{s['tuning_start']} 〜 {s['tuning_end']}（{s['tuning_dates']}営業日）"],
        ['最終評価期間', f"{s['final_start']} 〜 {s['final_end']}（{s['final_dates']}営業日）"],
        ['信用残を取得できた普通株の銘柄数', result['ordinary_codes']],
        ['信用残を取得できた普通株の銘柄×日付', result['ordinary_issue_dates']],
        ['信用残の取得済み銘柄日×買い売りレコード', result['potential_side_records']],
        ['公表平均建値を定義できるレコード', result['valid_target_records']],
        ['両手法で計算できる共通レコード', result['common_records']],
        ['出来高推定のみで計算できる有効正解レコード', result['baseline_computable_valid_target_records']],
        ['設定・入力ファイルのSHA256記録', len(source['input_sha256'])],
    ], columns=['項目', '値']))
    universe = pd.read_csv(ROOT / 'universe.csv')
    display(universe)
    scope = json.loads((ROOT / 'scope_summary.json').read_text())
    unknown = universe.loc[universe.universe_status=='unknown_historical_product_type','issue_dates'].sum()
    display(Markdown(f"対象期間に東証上場だった普通株は **{scope['ordinary_listed_codes_period']:,}銘柄** です。下表の `margin_unavailable_issue_dates` は、上場銘柄一覧にはあるものの信用残レコードが返らなかった銘柄日です。残高ゼロとはみなしていません。商品区分が判定不能の **{unknown:,}銘柄日** は普通株と断定できないため評価に含めていません。"))
    display(pd.read_csv(ROOT / 'scope_coverage.csv'))
    listed_side_records = 2 * scope['ordinary_listed_issue_dates']
    display(Markdown(f"東証普通株の全銘柄日×買い売りの枠は **{listed_side_records:,}件**、この全枠に対する共通評価レコードの割合は **{100*result['common_records']/listed_side_records:.2f}%** です。冒頭の算出可能率は、平均建値を定義できる有効正解レコードに対する割合です。残高0・信用残レコード未取得を含む全枠の割合とは分母が異なります。"))
    ''')
    md('''
    ## 推定の仮定

    - 両手法とも60営業日の、分割・併合を調整した日次VWAPと出来高を使います。
    - 出来高のみ：各日の出来高を重みに平均します。
    - 回転率モデル：その後の累積回転率に対して `exp(-k × 累積回転率)` で古い日の重みを減らします。
    - 回転率の分母は、各日23:59日本時間までに開示済みの有効な「発行済−自己株式数」です。
      同じ開示の一式を使い、自己株式数を0で補完しません。過去の一式は古くても引き継ぎます。
    - 株数は測定基準日から当該日までの分割を調整し、株価・出来高も残高基準日の単位に揃えます。
    - 正解は公表約定金額÷公表残高株数。公表約定金額や正解平均を推定への入力・平均合わせに使いません。
    - kは全銘柄・買い売り共通。したがって同じ銘柄・同じ日の予測平均は買い売りで同一です。
    - 0〜3を0.1刻みで調整期間だけ全探索し、RMSPEが最小の係数を固定します。同点なら最小の係数を使います。
    - 各レコードの重みは均等です。買い売り各50%や金額加重にはしていません。

    **RMSPE（%）＝100 × sqrt(mean(((推定平均−公表平均)÷公表平均)^2))**。
    公表平均が低い場合の大きな誤差率が強く効きます。誤差の大きなレコードも削除していません。
    ''')
    code('''
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(curve.coefficient, curve.training_rmspe, marker='o', ms=3, color='#2563eb')
    ax.axvline(s['chosen_coefficient'], color='#dc2626', ls='--', label=f"Selected k={s['chosen_coefficient']:.1f}")
    ax.set(xlabel='Shared decay coefficient k', ylabel='Training RMSPE (%)', title='Training period only: 31 candidate coefficients')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    display_plot(fig)
    display(curve)
    ''')
    md('''
    ## 最終評価と買い・売り別の結果

    主指標はRMSPEです。MAPE、符号付き平均誤差率、円単位のMAEは誤差の性質を理解するために併記します。
    売り残ゼロは平均建値を定義できず除外されるため、買い・売りのレコード数は同じとは限りません。
    ''')
    code('''
    display(metrics[metrics.phase == 'final'])
    display(Markdown('### 調整期間の結果（最終評価とは別）'))
    display(metrics[metrics.phase == 'tuning'])
    d = daily[(daily.phase == 'final') & (daily.side == 'all')]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(d.Date, d.baseline_rmspe, marker='o', label='Volume baseline')
    ax.plot(d.Date, d.candidate_rmspe, marker='o', label='Turnover survival')
    ax.tick_params(axis='x', rotation=40)
    ax.set(ylabel='Daily RMSPE (%)', title='Held-out dates: same records, fixed coefficient')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    display_plot(fig)
    display(daily[daily.phase == 'final'])
    ''')
    md('''
    ## 算出率・除外理由・誤差の分布

    比較は両手法で計算できる共通レコードで行っています。
    残高ゼロ、約定金額等の欠損、60営業日の不足、時点整合の取れた株数の不足などを区別して記録します。
    公表平均建値が60営業日の価格範囲外でも、正解・入力が有効なら評価から除外しません。
    下の `available` は評価に含めたレコード、その他は最初に該当した除外理由です。
    ''')
    code('''
    display(coverage)
    display(pd.read_csv(ROOT / 'support_diagnostics.csv'))
    display(pd.read_csv(ROOT / 'error_quantiles.csv'))
    display(Markdown(f"公表平均建値が60日価格範囲外のレコードは **{result['outside_60_day_price_support_records']:,}件（{100*result['outside_60_day_price_support_records']/result['common_records']:.2f}%）** です。非負の重みでこの60日間の価格を平均する両手法では、その範囲外の建値を構造的に再現できません。これらのレコードも誤差集計に含めています。"))
    ''')
    md('''
    ## 不確実性と解釈上の限界

    - 5営業日の移動ブロックを同じ順序で両手法に適用し、1000回再標本化しました。
      指標は各日の誤差二乗和と件数から再集計し、レコード均等を保ちます。係数の再調整はしません。
    - 最終評価は10営業日と短く、重なる60日窓と同一銘柄の反復もあります。95%区間は参考値です。
    - 全市場の出来高には現物取引や繰り返し売買を含みます。信用の新規建て・返済そのものではありません。
    - 回転率と返済率の関係は推定上の仮定で、係数が信用取引の真の返済率を示すとは限りません。
    - 60日より前の残存建玉を表現できず、日中の価格分布や同日返済も識別できません。
    - 有効な開示株数の引継ぎは、直近で株数の一部が欠損する場合に古い株数を使い続けるという限界があります。
      分割調整は反映しますが、その間の新株発行や自己株取得を完全に復元するものではありません。
    - 買いと売りで実際の建値分布が異なっても、今回の共通係数・共通の価格出来高だけでは分離できません。
    - 平均誤差の改善から、ヒストグラムの形状や損益帯の残高が正しいとは断定できません。
      既存の本番表示は公表平均に合わせる補正を含み、今回の独立した平均検証とは計算が異なります。
    - 合否しきい値は設けず探索的な比較に留めます。本番適用や追加の係数探索は実施していません。
    ''')
    code('''
    display(pd.DataFrame([{k: v for k, v in b.items()}]))
    draws = pd.read_csv(ROOT / 'bootstrap.csv').iloc[:,0]
    if len(draws):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.hist(draws, bins=30, color='#0d9488', alpha=.85)
        ax.axvline(0, color='#334155', ls='--')
        ax.set(xlabel='Candidate minus baseline RMSPE (percentage points)', ylabel='Replicates',
               title='Paired moving-block bootstrap: reference uncertainty')
        fig.tight_layout()
        display_plot(fig)
    ''')
    md('''
    ## 情報源・再現

    - [J-Quants 信用取引残高](https://jpx-jquants.com/ja/spec/mkt-margin-int)
    - [J-Quants 上場銘柄一覧](https://jpx-jquants.com/ja/spec/eq-master)
    - [商品区分コード](https://jpx-jquants.com/ja/spec/eq-master/product-category)
    - [J-Quants データFAQ](https://jpx-jquants.com/ja/help/data)

    設定は `config.json`、入力のハッシュと取得情報は `source_manifest.json`、
    最終評価前に保存した係数は `selection.json` に記録しています。
    このNotebookは確定結果を読むだけで、実行しても係数調整・最終評価をやり直しません。
    Notebookと同じフォルダのCSV・JSONを保持すれば、表・図を再生成できます。
    ''')
    nb.cells = cells
    executor = ExecutePreprocessor(timeout=120, kernel_name='python3')
    # An explicit interpreter keeps the report independent of shell PATH/kernel defaults.
    import sys
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    executor.preprocess(nb, {'metadata': {'path': str(root)}}, km=manager)
    nbf.write(nb, root / 'price-band-validation.ipynb')
    exporter = HTMLExporter(template_name='basic', exclude_input=True)
    body, _ = exporter.from_notebook_node(nb)
    style = '''
      :root { color-scheme: light; font-family: system-ui, "Noto Sans JP", sans-serif; color:#172033; }
      body { max-width:1180px; margin:auto; padding:24px; background:#fff; line-height:1.75; }
      h1,h2,h3 { line-height:1.45; margin-top:1.7em; color:#102a43; }
      h1 { border-bottom:3px solid #2563eb; padding-bottom:14px; }
      h2 { border-bottom:1px solid #d5deeb; padding-bottom:8px; }
      .output_subarea { overflow-x:auto; }
      table { border-collapse:collapse; width:auto; margin:12px 0; font-size:13px; white-space:nowrap; }
      th,td { border:1px solid #d7dfeb; padding:6px 10px; text-align:right; }
      th { background:#eaf1fb; } tr:nth-child(even) { background:#f6f8fc; }
      img { max-width:100%; height:auto; } pre { white-space:pre-wrap; }
      .input,.prompt,.output_prompt { display:none; } a { color:#1d4ed8; }
      code { background:#f1f5f9; padding:2px 4px; border-radius:3px; }
      @media(max-width:600px) { body { padding:12px; } h1{font-size:24px;} }
    '''
    document = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>価格帯別需給：平均建値の検証結果</title><style>' + style + '</style></head><body>'
                + body + '</body></html>')
    (root / 'price-band-validation.html').write_text(document)
    errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    if errors:
        raise ValueError('executed notebook contains errors')
    if document.count('<table') < 10 or document.count('data:image/') < 3:
        raise ValueError('static report is missing rendered tables or embedded plots')
    print(json.dumps({'notebook': str(root / 'price-band-validation.ipynb'),
                      'html': str(root / 'price-band-validation.html'),
                      'executed_code_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells),
                      'html_tables': document.count('<table'), 'embedded_images': document.count('data:image/')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True)
    build_report(parser.parse_args().result_dir)
