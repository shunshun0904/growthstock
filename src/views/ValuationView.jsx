import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ReferenceLine } from 'recharts';
import { loadValuations, rangePosition, trend } from '../lib/valuation.js';
const yen = v => Number.isFinite(v) ? `¥${v.toLocaleString('ja-JP', { maximumFractionDigits: 1 })}` : '—';
const pct = v => Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` : '—';
const num = v => Number.isFinite(v) ? v.toFixed(1) : '—';

export default function ValuationView() {
  const [data, setData] = useState(null), [error, setError] = useState(null);
  const [code, setCode] = useState(''), [method, setMethod] = useState('ri');
  const [overlay, setOverlay] = useState(false), [search, setSearch] = useState('');
  useEffect(() => {
    let active = true;
    loadValuations().then(x => { if (active) setData(x); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  if (error) return <section className="card"><h2>理論株価の比較</h2><p role="alert">{error}</p><p>データ更新後に再読み込みしてください。</p></section>;
  if (!data) return <div className="card" role="status">理論株価データを読み込んでいます…</div>;
  const stocks = data.stocks.filter(s => `${s.code} ${s.name}`.toLowerCase().includes(search.toLowerCase()));
  const stock = stocks.find(s => s.code === code) || stocks[0];
  const selected = stock?.methods.find(m => m.id === method);
  const showRange = overlay && selected?.kind === 'price' && selected.available;
  return <div className="valuation-view">
    <section className="card">
      <h2>理論株価の比較</h2>
      <p>候補銘柄の株価と、各手法の評価額を比較します。株価基準日：{data.asOf} ／ 更新：{data.generatedAt?.slice(0, 16).replace('T', ' ')} UTC</p>
      <div className="banner warn">{data.warning}</div>
      <div className="valuation-controls">
        <label>銘柄検索 <input aria-label="銘柄検索" value={search} onChange={e => setSearch(e.target.value)} placeholder="コード・銘柄名" /></label>
        <label>銘柄 <select aria-label="銘柄" value={stock?.code || ''} onChange={e => setCode(e.target.value)}>{stocks.map(s => <option key={s.code} value={s.code}>{s.code} {s.name}</option>)}</select></label>
      </div>
    </section>
    {!stock ? <div className="card">一致する銘柄がありません。</div> : <>
      <section className="card">
        <h3>{stock.code} {stock.name}　終値 {yen(stock.price)}</h3>
        <p>株価日：{stock.date} ／ 直近{stock.history.length}営業日の最初の終値から {pct(trend(stock.history))}</p>
        {stock.date !== data.asOf && <p className="banner warn">この銘柄の株価は全体の基準日より古いデータです。</p>}
        <p className="sub">株式分割を最新の株数単位に合わせた終値。最新の取引日まで最大20営業日を表示します。</p>
        <label><input type="checkbox" checked={overlay} onChange={e => setOverlay(e.target.checked)} /> 選択した手法のレンジ・中央値をチャートに重ねる</label>
        <div style={{ width: '100%', height: 310 }}>
          <ResponsiveContainer><LineChart data={stock.history} margin={{ top: 15, right: 25, bottom: 8, left: 20 }}>
            <XAxis dataKey="date" tickFormatter={d => d.slice(5)} minTickGap={25} />
            <YAxis domain={['auto', 'auto']} tickFormatter={v => v.toLocaleString()} />
            <Tooltip formatter={v => yen(v)} />
            {showRange && <ReferenceArea y1={selected.min} y2={selected.max} fill="#60a5fa" fillOpacity={0.15} ifOverflow="extendDomain" />}
            {showRange && <ReferenceLine y={selected.median} stroke="#a78bfa" strokeDasharray="4 4" ifOverflow="extendDomain" label="理論中央値" />}
            <ReferenceLine y={stock.price} stroke="#fbbf24" strokeDasharray="3 3" label="現在株価" />
            <Line type="linear" name="終値" dataKey="close" stroke="#38bdf8" strokeWidth={2} dot={false} />
          </LineChart></ResponsiveContainer>
        </div>
        <p>財務開示：{stock.financialDate || '—'} ／ 年次財務利用可能日：{stock.annualDate || '—'} ／ 金利基準日：{stock.yieldDate || '—'}</p>
        <p>β：{num(stock.beta)} ／ リスクフリーレート：{pct(stock.riskFree)}</p>
      </section>
      <section className="card">
        <h3>手法別の評価額</h3>
        <p>乖離率＝理論中央値÷現在株価−1。レンジ内位置は下限0%、上限100%（範囲外は0%未満・100%超）。</p>
        <div className="valuation-table-wrap"><table className="valuation-table"><thead><tr>
          {['手法', '下限', '上限', '平均', '中央値', '標準偏差', '中央値との乖離', '現在株価の位置'].map(x => <th key={x}>{x}</th>)}
        </tr></thead><tbody>{stock.methods.filter(m => m.kind === 'price').map(m => {
          const position = rangePosition(stock.price, m.min, m.max);
          return <tr key={m.id} className={method === m.id ? 'valuation-selected' : ''}>
            <td><button className="btn btn-ghost" onClick={() => { setMethod(m.id); setOverlay(true); }}>{m.label}</button></td>
            {m.available ? <><td>{yen(m.min)}</td><td>{yen(m.max)}</td><td>{yen(m.mean)}</td><td>{yen(m.median)}</td><td>{yen(m.std)}</td><td>{pct(m.deviation)}</td><td>{position === null ? '—' : `${(position * 100).toFixed(1)}%`}</td></> : <td colSpan="7">算出不可：{m.reason}</td>}
          </tr>;
        })}</tbody></table></div>
        <p>下限・上限はシナリオまたは同業比較の最小・最大値です。極端な同業倍率の影響を受けます。</p>
      </section>
      <section className="card"><h3>リバースDCF：株価に織り込まれた成長</h3>
        {stock.methods.filter(m => m.kind === 'growth').map(m => <p key={m.id}>{m.label}：<strong>{m.available ? `年率 ${pct(m.growth)}` : '算出不可'}</strong><br /><span className="sub">{m.note}{!m.available && ` ${m.reason}`}</span></p>)}
      </section>
      <section className="card"><h3>計算の前提と注意点</h3>
        {stock.methods.filter(m => m.kind === 'price').map(m => <p key={m.id}><strong>{m.label}</strong>：{m.note}{Number.isFinite(m.peers) && ` 比較対象：${m.peers}社。`}{Number.isFinite(m.p05) && ` 仮定分布の5〜95%点：${yen(m.p05)}〜${yen(m.p95)}。`}</p>)}
        <p>DCFの資本コストはCAPMと負債コストから計算。負債は取得できた内訳の合計で、欠けた内訳がゼロであることを確認できていません。推定値だけで売買を判断しないでください。</p>
      </section>
    </>}
  </div>;
}
