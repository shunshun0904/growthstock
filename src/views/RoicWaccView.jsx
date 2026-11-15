import React, { useEffect, useMemo, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { createRoicWaccClient, disclosureMarkers, reasonText, scenarioSeries, seriesCsv } from '../lib/roicWacc.js';

const number = (v, digits = 2) => Number.isFinite(v) ? v.toLocaleString('ja-JP', { maximumFractionDigits: digits }) : '—';
const yen = v => Number.isFinite(v) ? `¥${number(v, 1)}` : '—';
const rate = v => Number.isFinite(v) ? `${number(v * 100, 2)}%` : '—';
const colors = ['var(--blue)', 'var(--green)', 'var(--amber)', 'var(--violet)'];

function Chart({ title, rows, lines, band = false, reference, disclosures = [], format = number }) {
  const available = rows.some(row => lines.some(([key]) => Number.isFinite(row[key])));
  return <section className="card rw-chart">
    <h3>{title}</h3>
    {!available ? <p className="sub rw-empty">この条件で算出できる系列がありません。下の「算出不可の理由」を確認してください。</p> :
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={rows} margin={{ top: 15, right: 12, left: 5, bottom: 5 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="Date" minTickGap={45} tick={{ fontSize: 10, fill: 'var(--text-dim)' }} />
          <YAxis width={65} tick={{ fontSize: 10, fill: 'var(--text-dim)' }} tickFormatter={format} domain={['auto', 'auto']} />
          <Tooltip formatter={(v, name) => [Array.isArray(v) ? v.map(format).join('〜') : format(v), name]}
            contentStyle={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {reference !== undefined && <ReferenceLine y={reference} stroke="var(--text-dim)" strokeDasharray="4 4" />}
          {disclosures.map(event => <ReferenceLine key={event.date} x={event.date} stroke="var(--text-dim)" strokeOpacity={.3} strokeDasharray="2 5" />)}
          {band && <Area dataKey="range" name="有効シナリオの範囲" fill="var(--green)" fillOpacity={.15} stroke="none" connectNulls={false} isAnimationActive={false} />}
          {lines.map(([key, label], i) => <Line key={key} dataKey={key} name={label} stroke={colors[i % colors.length]}
            dot={false} strokeWidth={1.8} connectNulls={false} isAnimationActive={false} />)}
        </ComposedChart>
      </ResponsiveContainer>}
  </section>;
}

export default function RoicWaccView() {
  const client = useMemo(() => createRoicWaccClient(), []);
  const [catalog, setCatalog] = useState(null), [code, setCode] = useState(''), [query, setQuery] = useState('');
  const [data, setData] = useState(null), [error, setError] = useState(''), [loading, setLoading] = useState(false);
  const [retry, setRetry] = useState(0), [period, setPeriod] = useState('252'), [day, setDay] = useState('');
  const [model, setModel] = useState('eva_total_fade5'), [tax, setTax] = useState('tax30'), [erp, setErp] = useState('all');
  const [growth, setGrowth] = useState('all'), [epsBasis, setEpsBasis] = useState('forecast');
  const isRim = model.startsWith('rim_'), isTotal = model.startsWith('eva_total_');
  const returnName = isRim ? '親会社ROE' : isTotal ? '現金控除なしの資本収益率' : '営業投下資本ROIC';
  const discountName = isRim ? '自己資本コストKe' : 'WACC';
  useEffect(() => {
    let active = true;
    const abort = new AbortController();
    setError('');
    client.catalog(abort.signal).then(value => {
      if (!active) return;
      setCatalog(value);
      setCode(current => current || value.stocks.find(s => s.valuedRows > 0)?.code || value.stocks.find(s => s.rows > 0)?.code || '');
    }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; abort.abort(); };
  }, [client, retry]);
  useEffect(() => {
    const stock = catalog?.stocks.find(s => s.code === code);
    if (!stock || !stock.rows) { setData(null); setLoading(false); return; }
    let active = true;
    const abort = new AbortController();
    setLoading(true); setError(''); setData(null);
    client.series(stock, abort.signal).then(value => {
      if (!active) return;
      setData(value); setDay(value.rows.at(-1)?.Date || '');
    }).catch(e => { if (active) setError(e.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; abort.abort(); };
  }, [client, catalog, code]);
  const stocks = (catalog?.stocks || []).filter(s => `${s.code} ${s.name}`.toLowerCase().includes(query.toLowerCase()));
  const selectedStock = catalog?.stocks.find(s => s.code === code);
  const allRows = useMemo(() => data ? scenarioSeries(data, { model, tax, erp, growth, epsBasis }) : [], [data, model, tax, erp, growth, epsBasis]);
  const rows = period === 'all' ? allRows : allRows.slice(-Number(period));
  const current = allRows.filter(r => r.Date <= day).at(-1);
  const disclosed = disclosureMarkers(allRows, rows);
  const valid = allRows.filter(r => Number.isFinite(r.median)).length;
  const download = () => {
    const url = URL.createObjectURL(new Blob([seriesCsv(rows)], { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a'); a.href = url; a.download = `${code}-roic-wacc-${model}-${tax}.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="rw-view">
    <section className="card">
      <h2>資本効率と理論株価（EVA・RIM） <span className="sd-badge">研究用</span></h2>
      <p>現在のモデルの推論対象銘柄を使った日次分析です。主モデルは現金を区分しない総投下資本＋将来EVAの現在価値、比較は外生的な自己資本コストを使うRIMです。対象銘柄の株価・β・時価総額を計算に使いません。予測モデルの特徴量や売買判断には採用していません。</p>
      <p>現金区分なしEVAは、現金を資本から控除せず、評価後にも加算しない近似です。非事業性現金が0と判明したという意味ではなく、理論価値の下限も保証しません。</p>
      <p>表示範囲はERP・成長率などの仮定を変えた感応度で、信頼区間や将来株価の予測区間ではありません。</p>
      <div className="valuation-controls rw-controls">
        <label>銘柄検索 <input aria-label="研究対象の銘柄検索" value={query} onChange={e => setQuery(e.target.value)} placeholder="コード・銘柄名" /></label>
        <label>分析対象 <select aria-label="ROIC・WACCの分析対象" value={code} onChange={e => { setData(null); setCode(e.target.value); }}>
          {selectedStock && !stocks.some(s => s.code === code) && <option value={code}>選択中：{selectedStock.code} {selectedStock.name}</option>}
          {stocks.map(s => <option key={s.code} value={s.code} disabled={!s.rows}>{s.code} {s.name}{!s.rows ? '（履歴なし）' : ''}</option>)}
        </select></label>
        <label>表示期間 <select aria-label="ROIC・WACCの表示期間" value={period} onChange={e => setPeriod(e.target.value)}>
          <option value="126">直近126営業日</option><option value="252">直近252営業日</option><option value="756">直近756営業日</option><option value="all">利用可能な全期間</option>
        </select></label>
        <label>理論価値モデル <select aria-label="理論価値モデル" value={model} onChange={e => { setModel(e.target.value); if (!e.target.value.startsWith('eva_')) setGrowth('all'); }}>
          <option value="eva_total_fade5">近似EVA：現金区分なし・5年収束</option><option value="eva_total_steady">近似EVA：現金区分なし・収益率一定</option>
          <option value="rim_book5">RIM：外生Ke・5年ROE収束</option><option value="rim_book10">RIM：外生Ke・10年ROE収束</option>
          <option value="eva_fade5">営業EVA：現金区分必須・5年収束</option><option value="eva_steady">営業EVA：現金区分必須・ROIC一定</option>
          <option value="steady">比較：市場依存CAPM・定常成長DCF</option><option value="fade5">比較：市場依存CAPM・5年DCF</option>
        </select></label>
        <label>税率{isRim ? '（RIMは実績純利益）' : ''} <select aria-label="税率シナリオ" value={tax} disabled={isRim} onChange={e => setTax(e.target.value)}>
          <option value="tax30">30%固定</option><option value="actual">実績税率</option>
        </select></label>
        <label>株主資本リスクプレミアム <select aria-label="ERPシナリオ" value={erp} onChange={e => setErp(e.target.value)}>
          <option value="all">3・5・7%の全ケース</option>{[.03, .05, .07].map(v => <option key={v} value={v}>{v * 100}%</option>)}
        </select></label>
        <label>長期成長率{isRim ? '（RIMはROE収束）' : ''} <select aria-label="成長率シナリオ" value={growth} disabled={isRim} onChange={e => setGrowth(e.target.value)}>
          <option value="all">0〜3%の全ケース</option>{[0, .01, .02, .03].map(v => <option key={v} value={v}>{v * 100}%</option>)}
          <option value="business" disabled={!model.startsWith('eva_')}>事業実績の単純継続予測</option>
        </select></label>
        <label>PERの利益系列 <select aria-label="PERの利益系列" value={epsBasis} onChange={e => setEpsBasis(e.target.value)}>
          <option value="forecast">会社予想利益から計算したEPS</option><option value="ttm">TTM実績EPS</option>
        </select></label>
      </div>
      {error && <p role="alert">{error} <button className="btn btn-ghost" onClick={() => { setCatalog(null); setRetry(x => x + 1); }}>再読み込み</button></p>}
      {(!catalog || loading) && !error && <p role="status">研究データを読み込んでいます…</p>}
      {catalog && !catalog.stocks.length && <p>研究データはまだ作成されていません。</p>}
      {catalog && !stocks.length && <p>一致する銘柄がありません。</p>}
      {data && <p>{data.code} {data.name} ／ {allRows[0]?.Date}〜{allRows.at(-1)?.Date} ／ 理論株価を算出できる日：{valid.toLocaleString()} / {allRows.length.toLocaleString()}営業日</p>}
      {data && !valid && <p className="banner warn">必要な財務情報や計算条件が揃わないため、理論株価・評価乖離率は算出できません。市場株価と、算出できた資本効率・資本コストを表示します。</p>}
    </section>
    {data && <>
      <div className="rw-grid">
        <Chart title="A. 市場株価・理論株価" rows={rows} lines={[["price", "市場株価"], ["median", "理論株価中央値"]]} band disclosures={disclosed} format={v => number(v, 0)} />
        <Chart title="現金区分なしEVA・RIM（基準シナリオ中央値）" rows={rows} lines={[["baseEvaValue", "近似EVA・5年収束"], ["baseRimValue", "RIM・5年収束"]]} disclosures={disclosed} format={v => number(v, 0)} />
        <Chart title={`B. 市場PER・理論PER（${epsBasis === 'forecast' ? '会社予想' : 'TTM実績'}）`} rows={rows} lines={[["marketPer", "市場PER"], ["theoreticalPer", "理論PER"]]} disclosures={disclosed} />
        <Chart title="C. 評価乖離率（1.0で理論価値と一致）" rows={rows} lines={[["gap", "市場株価÷理論株価"]]} reference={1} disclosures={disclosed} />
        <Chart title="市場価格と理論価値の変化（20営業日・ログ）" rows={rows}
          lines={[["valuation_price_log_change_20d", "市場価格の変化"], ["valuation_value_log_change_20d", "理論価値の変化"], ["valuation_gap_log_change_20d", "乖離率の変化"]]}
          reference={0} disclosures={disclosed} format={rate} />
        <Chart title="過去252営業日のZスコア" rows={rows} lines={[["gapZ", "Zスコア"]]} reference={0} disclosures={disclosed} />
        <Chart title="過去252営業日の百分位順位" rows={rows} lines={[["gapPercentile", "百分位順位"]]} disclosures={disclosed} format={rate} />
        <Chart title="D. 資本効率と資本コスト" rows={rows} lines={[["selectedRoic", returnName], ["selectedWacc", discountName], ["selectedSpread", `${returnName}−${discountName}`], ...(!isTotal ? [["conservativeRoic", "現金控除なしの資本収益率（参考）"]] : [])]} disclosures={disclosed} format={rate} />
      </div>
      <section className="card">
        <div className="rw-detail-head"><h3>任意日付の因子・計算条件</h3><label>確認日 <input type="date" aria-label="因子の確認日" value={day} min={allRows[0]?.Date} max={allRows.at(-1)?.Date} onChange={e => setDay(e.target.value)} /></label>
          <button className="btn btn-ghost" onClick={download}>表示期間のCSVをダウンロード</button></div>
        {current && <>
          <p>指定日以前の直近営業日：{current.Date} ／ 有効シナリオ {current.scenarioCount}/{current.scenarioTotal}</p>
          <dl className="rw-factors">
            {[['市場株価', yen(current.price)], ['理論株価中央値', yen(current.median)], ['下限〜上限', `${yen(current.min)}〜${yen(current.max)}`],
              ['平均／母標準偏差', `${yen(current.mean)}／${yen(current.std)}`], ['評価乖離率', number(current.gap, 3)], [isTotal ? '総資本価値の評価乖離率' : isRim ? 'EV評価乖離率（RIMは対象外）' : 'EV評価乖離率', number(current.selectedEvGap, 3)],
              ['市場価格の20日ログ変化', rate(current.valuation_price_log_change_20d)], ['理論価値の20日ログ変化', rate(current.valuation_value_log_change_20d)],
              [returnName, rate(current.selectedRoic)], ['現金控除なしの資本収益率（参考）', rate(current.conservativeRoic)], [discountName, rate(current.selectedWacc)],
              ['長期成長率の仮定', isRim ? 'ROEがKeへ収束（固定gは使わない）' : growth === 'all' ? '固定0・1・2・3%' : growth === 'business' ? rate(isTotal ? current.business_growth_total_estimate : current.business_growth_estimate) : rate(Number(growth))],
              ['RIMの配当性向', isRim ? rate(current.rim_payout) : '—'],
              ['再投資率', rate(current.selectedReinvestment)], ['市場依存比較モデルのβ', number(current.tv_beta, 3)], ['比較モデルβの有効観測数', number(current.tv_beta_n, 0)],
              ['国債10年金利', rate(current.tv_rf)], ['金利基準日', current.tv_rf_date || '—'], ['負債コスト仮定', `国債金利＋${rate(data.creditSpread)}`],
              ['売上高純利益率M', rate(current.net_margin)], ['総資産回転率T', number(current.asset_turnover)], ['財務レバレッジL', number(current.leverage)],
              ['平均自己資本ベースBPS', yen(current.bps_average_equity)],
              ['財務の公表日', current.financial_source_date || '—'], ['財務の利用開始日', current.financial_available_date || '—'],
              ['EVA年次財務の利用開始日', current.avail_date || '—'], ['RIM利益の根拠公表日', current.rim_source_date || '—'],
              ['過去の訂正履歴', current.revision_unverified ? '原本の全バージョンは未照合' : '系列データの照合状態を確認']]
              .map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
          </dl>
          <h4>算出不可の理由</h4><p>{Number.isFinite(current.median) ? '選択した条件には有効なケースがあります。' : reasonText(current.unavailable_reason) || '選択シナリオが割引率・再投資率・正の株式価値の条件を満たしません。'}</p>
          {!Number.isFinite(current.marketPer) && <p>選択した利益系列に、同じ株数基準の正のEPSがないためPERは算出していません。別系列からの自動補完は行いません。</p>}
        </>}
      </section>
      <section className="card">
        <h3>解釈と制約</h3>
        <p>営業ROICは年次NOPAT÷期首末平均営業投下資本です。近似EVAは現金を含む総資本の収益率を使い、資本経路と同じ定義で利益・再投資を予測します。現金や金融収益を別建て評価する営業EVAとは区別します。</p>
        <p>RIMは親会社自己資本＋将来残余利益の現在価値です。実績TTM利益と年次配当性向を使い、ROEを5年または10年で外生Keへ収束させます。欠損配当を0にせず、配当性向0〜100%を要求します。予測期間のクリーンサープラスと終価＝簿価を仮定しています。</p>
        <p>基準シナリオ比較のEVAは固定g0〜3%・ERP3/5/7%・税率30%、RIMはERP3/5/7%・実績TTM利益・配当性向・5年収束の中央値です。この比較線はフィルタ選択とは別に、同じ日付の基準値を表示します。</p>
        <p>主モデルの割引率は、国債金利＋事前固定の株主資本リスクプレミアムと、簿価の資本構成から計算します。株価由来のβ・時価総額ウェイトを使うCAPMは比較モデルに限定しています。</p>
        <p>主モデルは期首投下資本×予測ROICからNOPATを作り、投下資本の成長と整合する再投資を計算します。ROIC一定と、5年で外生的割引率に収束するケースを比較し、同じ利益・投資の経路でDCFとの一致を検証しています。</p>
        <p>事業データによる成長率は、過去の売上・NOPAT成長率と、取得できた場合の再投資・限界ROICに基づく単純継続予測です。事前に0〜3%へ制限した探索的仮定で、固定成長率ケースとは混ぜません。</p>
        <p>WACC−成長率が1%ポイント未満、再投資率が0〜1の範囲外、必要な残高が欠損、株式価値が0以下などのケースは除外します。</p>
        <p>縦の点線は財務の公表日（休場日は次の営業日）を示します。財務値は公表日の翌営業日から使用します。金利と、比較モデルのベータは評価日までの情報を使います。過去APIの事後訂正は完全には再現できていません。</p>
        <p>価格のグラフは各日の株数基準です。株式分割時は市場株価と理論株価の両方が変わるため、乖離率と分割調整済みのログ変化も併せて確認します。</p>
        <p>20日ログ変化は「乖離率の変化＝市場価格の変化−理論価値の変化」です。企業行動の種類を確認できないイベントを跨ぐ期間は表示しません。</p>
        <p>Zスコアは当日を含む過去252営業日中126以上の有効観測が必要で、標準偏差ゼロなら算出不可です。ERPの全ケースを選んだとき、資本コストの線はERP5%のケースです。</p>
        <p>研究する7要素は、純利益率・総資産回転率・簿価財務レバレッジ・BPS・ROIC・再投資率・外生的割引率です。厳密な乗算恒等式ではありません。市場PERは比較・診断にだけ使います。</p>
        <p>計算上の株価入力を排除しても、財務数値の時価評価、自己株式取得、金利や景気を通じた経済的な市場依存までなくなるとは主張しません。</p>
      </section>
    </>}
  </div>;
}
