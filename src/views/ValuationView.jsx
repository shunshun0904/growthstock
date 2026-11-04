import React, { useEffect, useState } from 'react';
import { loadValuations, rankedValuations } from '../lib/valuation.js';
const yen = v => Number.isFinite(v) ? `¥${v.toLocaleString('ja-JP', { maximumFractionDigits: 1 })}` : '—';
const pct = v => Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` : '—';
export default function ValuationView() {
  const [data, setData] = useState(null), [error, setError] = useState(null);
  const [method, setMethod] = useState('ri'), [search, setSearch] = useState('');
  useEffect(() => {
    let active = true;
    loadValuations().then(x => { if (active) setData(x); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, []);
  if (error) return <section className="card"><h2>理論株価ランキング</h2><p role="alert">{error}</p></section>;
  if (!data) return <div className="card" role="status">理論株価データを読み込んでいます…</div>;
  const ranked = rankedValuations(data.stocks, method);
  const rows = ranked.map((s, i) => ({ ...s, rank: Number.isFinite(s.deviation) ? i + 1 : null }))
    .filter(s => `${s.code} ${s.name}`.toLowerCase().includes(search.toLowerCase()));
  const label = method === 'ri' ? 'RIM（残余利益）' : '配当割引法';
  const available = ranked.filter(s => Number.isFinite(s.deviation)).length;
  return <div className="valuation-view">
    <section className="card">
      <h2>理論株価ランキング</h2>
      <p>直近3営業日の推論対象銘柄すべてを、割安度の高い順に表示します。同じ銘柄は最新の推論日で1件にまとめています。</p>
      <p>推論対象日：{data.inferenceDates?.join(' / ') || '—'} ／ 株価基準日：{data.asOf || '—'}</p>
      <div className="valuation-controls">
        <label>算出方法 <select aria-label="算出方法" value={method} onChange={e => setMethod(e.target.value)}>
          <option value="ri">RIM（残余利益）</option><option value="ddm">配当割引法</option>
        </select></label>
        <label>銘柄検索 <input aria-label="銘柄検索" value={search} onChange={e => setSearch(e.target.value)} placeholder="コード・銘柄名" /></label>
      </div>
      <p>{label}：算出可能 {available} / 対象 {ranked.length}銘柄。算出不可の銘柄も末尾に表示します。</p>
      <p>割安度（乖離率）＝理論株価の中央値÷使用終値−1。プラスが大きいほど理論値に対して割安です。</p>
    </section>
    <section className="card">
      <h3>{label}の評価額</h3>
      {!rows.length ? <p>対象銘柄がありません。</p> : <div className="valuation-table-wrap"><table className="valuation-table"><thead><tr>
        {['順位', '銘柄', '最新推論日', '使用終値', '割安度 ↓', '理論中央値', '下限', '上限', '平均', '標準偏差', '株価日'].map(x => <th key={x}>{x}</th>)}
      </tr></thead><tbody>{rows.map(s => <tr key={s.code}>
        <td>{s.rank ?? '—'}</td><td>{s.code} {s.name}</td><td>{s.inferenceDate || '—'}</td><td>{yen(s.price)}</td>
        {s.rank !== null ? <><td><strong>{pct(s.deviation)}</strong></td><td>{yen(s.value.median)}</td><td>{yen(s.value.min)}</td><td>{yen(s.value.max)}</td><td>{yen(s.value.mean)}</td><td>{yen(s.value.std)}</td></> : <td colSpan="6">算出不可：{s.value?.reason || '必要データ・計算条件を満たさない'}</td>}
        <td>{s.date || '—'}{s.priceFallback && <span title={`株価の最新取得日は${s.latestStockPriceDate}。TOPIXとそろう前営業日の終値で評価しています。`}>（前営業日）</span>}</td>
      </tr>)}</tbody></table></div>}
    </section>
    <section className="card">
      <h3>算出の前提</h3>
      <p>{method === 'ri' ? 'RIMは利益と純資産に基づく残余利益法です。ROEを5年・10年で資本コストに収束させ、ERP3〜7%、配当性向25〜75%のシナリオを比較します。' : '配当割引法は実績年間配当を使い、成長率0〜2%、ERP3〜7%のシナリオを比較します。無配・赤字・配当性向や割引率の条件を満たさない銘柄は算出できません。'}</p>
      <p>株価とTOPIXの更新日が異なる場合は、両方がそろう直近の終値を使用します（前営業日まで）。使用した株価日は各行に表示します。</p>
      <p>資本コストはCAPMで算出。各銘柄の使用終値時点より前に利用可能な財務・金利を使います。レンジはシナリオの最小・最大値で、将来株価の予測区間ではありません。</p>
      <p className="sub">更新：{data.generatedAt?.slice(0, 16).replace('T', ' ')} UTC</p>
    </section>
  </div>;
}
