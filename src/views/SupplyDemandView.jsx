import React, { useEffect, useMemo, useRef, useState } from 'react';
import { finite, yen, shares, percent, supplyChoices, profileBands, sideDisplay, PROFILE_STATUS } from '../lib/supplyDemand.js';

function PositionCard({ label, side, profile }) {
  const d = sideDisplay(side, profile);
  return <section className="sd-position">
    <h3>{label}の参考含み損益{d.estimated && <span className="sd-badge">推定</span>}</h3>
    <strong className={`sd-pnl ${finite(d.pnl) ? (d.pnl >= 0 ? 'sd-profit' : 'sd-loss') : ''}`}>{percent(d.pnl)}</strong>
    <p>平均建値{d.estimated ? '（推定）' : '（約定金額÷株数）'} <b>{yen(d.average)}</b></p>
    <p>信用残 <b>{shares(side?.shares)}</b>
      {finite(side?.shares) && finite(side?.reportedShares) && side.shares !== side.reportedShares && <span>（公表時 {shares(side.reportedShares)}）</span>}</p>
    <p>公表約定金額 <b>{finite(side?.contractAmount) ? `${(side.contractAmount / 1e8).toFixed(2)}億円` : '未取得'}</b></p>
  </section>;
}

function PriceProfile({ data, profile, unit }) {
  const ref = useRef(null);
  const [width, setWidth] = useState(700);
  const [active, setActive] = useState(null);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { setActive(null); }, [profile, unit, data]);
  const bands = useMemo(() => profileBands(profile), [profile]);
  const lines = [{ label: '終値', value: data.currentPrice, color: 'var(--text)' }];
  for (const [side, label, color] of [['long', '平均買い建値', 'var(--amber)'], ['short', '平均売り建値', 'var(--violet)']]) {
    const d = sideDisplay(data[side], profile?.[side]);
    if (finite(d.average)) lines.push({ label: `${label}${d.estimated ? '（推定）' : ''}`, value: d.average, color });
  }
  const values = [...bands.flatMap(b => [b.low, b.high]), ...lines.map(l => l.value)].filter(finite);
  const min = Math.min(...values), max = Math.max(...values);
  const padding = Math.max((max - min) * .08, max * .015, 1);
  const low = min - padding, high = max + padding;
  const height = 480, top = 40, bottom = 430, left = width < 480 ? 57 : 76, right = 12;
  const center = (left + width - right) / 2, half = (width - right - left) / 2 - 8;
  const y = price => bottom - (price - low) / (high - low) * (bottom - top);
  const largest = Math.max(...bands.flatMap(b => [b.long, b.short]), 1);
  const value = qty => unit === 'shares' ? qty / 1e4 : qty * data.currentPrice / 1e8;
  const fmt = qty => value(qty).toLocaleString('ja-JP', { maximumFractionDigits: 2 });
  const unitLabel = unit === 'shares' ? '万株' : '億円・現在値換算';
  const activeBand = active === null ? null : bands[active];
  return <div ref={ref} className="sd-chart">
    <div className="sd-reference-lines">{lines.map(l => <span key={l.label} style={{ color: l.color }}>━ {l.label} <b>{yen(l.value)}</b></span>)}</div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="価格帯別の信用売り残・買い残の推定分布。左が売り、右が買い。">
      <text x={center-half/2} y="20" textAnchor="middle" className="sd-svg-heading">◀ 信用売り残</text>
      <text x={center+half/2} y="20" textAnchor="middle" className="sd-svg-heading">信用買い残 ▶</text>
      {Array.from({ length: 6 }, (_, i) => low + (high-low)*i/5).map(p => <g key={p}>
        <line x1={left} x2={width-right} y1={y(p)} y2={y(p)} className="sd-grid" />
        <text x={left-7} y={y(p)+4} textAnchor="end" className="sd-svg-tick">{p.toLocaleString('ja-JP', { maximumFractionDigits: 0 })}</text>
      </g>)}
      <line x1={center} x2={center} y1={top} y2={bottom} className="sd-axis" />
      {bands.map((b, i) => {
        const cy = y(b.high), h = Math.max(2, y(b.low)-cy-1), mid = (b.low+b.high)/2;
        return <g key={`${b.low}`} className="sd-band" tabIndex="0" role="button"
          aria-label={`${yen(b.low)}〜${yen(b.high)}、売り${shares(b.short)}、買い${shares(b.long)}`}
          onMouseEnter={() => setActive(i)} onFocus={() => setActive(i)} onClick={() => setActive(i)}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setActive(i); } }}>
          <rect x={left} y={cy} width={width-right-left} height={h} fill="transparent" />
          {b.short > 0 && <rect x={center-b.short/largest*half} y={cy} width={b.short/largest*half} height={h} rx="2"
            className={mid >= data.currentPrice ? 'sd-bar-profit' : 'sd-bar-loss'} />}
          {b.long > 0 && <rect x={center} y={cy} width={b.long/largest*half} height={h} rx="2"
            className={mid <= data.currentPrice ? 'sd-bar-profit' : 'sd-bar-loss'} />}
          <title>{`${yen(b.low)}〜${yen(b.high)} / 売り ${shares(b.short)} / 買い ${shares(b.long)}`}</title>
        </g>;
      })}
      {lines.map(l => <line key={l.label} x1={left} x2={width-right} y1={y(l.value)} y2={y(l.value)} stroke={l.color}
        strokeWidth={l.label === '終値' ? 1.8 : 1.2} strokeDasharray={l.label === '終値' ? undefined : '5 4'} pointerEvents="none" />)}
      <text x={left} y="452" className="sd-svg-tick">{fmt(largest)}</text>
      <text x={center} y="452" textAnchor="middle" className="sd-svg-tick">0</text>
      <text x={width-right} y="452" textAnchor="end" className="sd-svg-tick">{fmt(largest)}</text>
      <text x={center} y="474" textAnchor="middle" className="sd-svg-tick">横軸：{unitLabel} ／ 縦軸：円</text>
    </svg>
    <div className="sd-chart-caption"><span className="sd-profit">● 含み益側</span><span className="sd-loss">● 含み損側</span><span>棒の長さは推定残高</span></div>
    <div className="sd-band-detail" aria-live="polite">{activeBand
      ? <>{yen(activeBand.low)} 〜 {yen(activeBand.high)}：売り {shares(activeBand.short)} ／ 買い {shares(activeBand.long)}</>
      : '価格帯にカーソルを合わせるか、タップすると推定株数を確認できます。'}</div>
    {!bands.length && <p role="status">この期間では価格帯分布を算出できません。信用残と平均建値は上のカードで確認できます。</p>}
    {['short', 'long'].map(side => PROFILE_STATUS[profile?.[side]?.status] && <p className="sd-status" key={side}>
      {side === 'long' ? '買い' : '売り'}：{PROFILE_STATUS[profile[side].status]}</p>)}
    {!!bands.length && <details className="sd-table"><summary>価格帯ごとの推定株数を表で見る</summary>
      <div className="table-scroll"><table><thead><tr><th>価格帯</th><th>信用売り残（推定）</th><th>信用買い残（推定）</th></tr></thead>
        <tbody>{bands.map(b => <tr key={b.low}><td>{yen(b.low)}〜{yen(b.high)}</td><td>{shares(b.short)}</td><td>{shares(b.long)}</td></tr>)}</tbody></table></div>
    </details>}
  </div>;
}

export default function SupplyDemandView({ stocks, candidates, selectedId, onSelect }) {
  const choices = useMemo(() => supplyChoices(stocks, candidates), [stocks, candidates]);
  const [search, setSearch] = useState('');
  const [window, setWindow] = useState('120');
  const [unit, setUnit] = useState('value');
  const selectedCode = selectedId?.replace(/^(jq-|pred:)/, '');
  const stock = choices.find(s => s.id === selectedId)
    || choices.find(s => String(s.jqCode || s.code) === selectedCode)
    || choices.find(s => s.supplyDemand?.status === 'ok') || choices[0];
  const filtered = choices.filter(s => `${s.code} ${s.name}`.toLowerCase().includes(search.trim().toLowerCase()));
  const options = stock && !filtered.includes(stock) ? [stock, ...filtered] : filtered;
  const data = stock?.supplyDemand, profile = data?.profiles?.[window];
  const complete = choices.filter(s => s.supplyDemand?.status === 'ok').length;
  return <div className="supply-demand-view">
    <section className="card">
      <div className="card-head"><h2>価格帯別需給 <span className="sd-badge">推計</span></h2><span className="sub">データ取得済み {complete}銘柄</span></div>
      <p className="sd-intro">公表された信用残を、過去の価格・出来高から価格帯へ配分した参考図です。実際の建玉明細や板情報ではありません。</p>
      <div className="sd-controls">
        <label>銘柄を検索<input aria-label="需給の銘柄検索" value={search} onChange={e => setSearch(e.target.value)} placeholder="証券コード・銘柄名" /></label>
        <label className="sd-stock-select">分析対象<select aria-label="需給の分析対象銘柄" value={stock?.id || ''} onChange={e => onSelect(e.target.value)}>
          {!options.length && <option value="">該当する銘柄がありません</option>}
          {options.map(s => <option key={s.id} value={s.id}>{s.code} {s.name}{s.supplyDemand?.status === 'ok' ? '' : '（需給データ未取得）'}</option>)}
        </select></label>
      </div>
      {search && filtered.length === 0 && <p role="status">検索条件に一致する銘柄がありません。</p>}
    </section>
    {!data || !['ok', 'missing_adjustment'].includes(data.status) ? <section className="card" role="status">
      <h3>{stock ? `${stock.code} ${stock.name}` : '銘柄データがありません'}</h3>
      <p>{data?.status === 'missing_balance' ? '株価基準日までに利用可能と確認できた信用残がありません。' : data?.status === 'missing_price' ? '株価データを取得できていません。' : 'この銘柄の価格帯別需給データはまだ取得されていません。'}</p>
      <p>日次更新の詳細データ取得対象になった銘柄から表示します。データがない銘柄を残高0としては扱いません。</p>
    </section> : <>
      <section className="card">
        <div className="card-head"><h2>{stock.code} {stock.name}</h2><strong>終値 {yen(data.currentPrice)}</strong></div>
        <p className="sd-dates">株価：{data.asOf} ／ 信用残基準日：{data.balance.date}<br />
          {data.balance.publishedOn ? `公表日：${data.balance.publishedOn}` : data.balance.availabilityBasis === 'public_snapshot'
            ? `公表日未取得（${data.balance.availableOn}の公開データで取得済みと確認）`
            : `公表日：${data.balance.availableOn}（営業日カレンダーによる推定）`}</p>
        {data.status === 'missing_adjustment' && <p role="status">信用残基準日の株式分割調整を確認できないため、現在の株数に換算できません。</p>}
        <div className="sd-positions"><PositionCard label="信用売り" side={data.short} profile={profile?.short} /><PositionCard label="信用買い" side={data.long} profile={profile?.long} /></div>
        <p className="sd-ratio">信用倍率 <b>{finite(data.ratio) ? `${data.ratio.toFixed(2)}倍` : '算出不可'}</b> <span>売り残が0株・欠損の場合は算出しません。</span></p>
      </section>
      <section className="card">
        <div className="sd-controls sd-chart-controls">
          <fieldset><legend>分布推定に使う期間</legend>{['60', '120', '250'].map(w => <button className="btn" key={w} aria-pressed={window === w} onClick={() => setWindow(w)}>{w}営業日</button>)}</fieldset>
          <fieldset><legend>横軸</legend><button className="btn" aria-pressed={unit === 'value'} onClick={() => setUnit('value')}>現在値換算</button><button className="btn" aria-pressed={unit === 'shares'} onClick={() => setUnit('shares')}>株数</button></fieldset>
        </div>
        <p className="sd-dates">使用期間：{profile?.from || '—'} 〜 {profile?.to || '—'}（{profile?.sessions ?? 0}営業日、売買あり {profile?.tradedSessions ?? 0}日）</p>
        {['long', 'short'].some(side => profile?.[side]?.method === 'volume_prior') && <div className="sd-estimate-note">
          約定金額が未取得の側は、出来高による配分のみです。平均建値・含み損益も推定であり、期間を変えると値が変わります。
        </div>}
        <PriceProfile data={data} profile={profile} unit={unit} />
      </section>
      <section className="card sd-method">
        <h3>この図の算出方法と限界</h3>
        <p>信用残基準日以前の各営業日の出来高を、その日のVWAP（売買代金÷出来高）の価格に置き、20価格帯に集約します。価格は株式分割・併合を調整し、株数は株価基準日の株式数に換算します。</p>
        <p>信用残の約定金額が取得できる側は、金額÷株数から平均建値を算出し、その平均と総株数に合うよう分布を補正します。未取得の側は、信用残を過去の出来高比で配分します。信用売り・買いの実際の約定日や返済順序は不明なので、分布の形は仮定に依存します。</p>
        <p>参考含み損益は、公表時の建玉が株価基準日まで残っていると仮定し、買いは「終値÷平均建値−1」、売りは「1−終値÷平均建値」で計算します。金利・貸株料・担保・返済済み建玉は反映しません。</p>
        <p>価格帯の赤・緑は帯の中央価格による含み損益の方向です。帯をまたぐ終値の近辺では概算です。市場全体の売買を信用取引の建玉へ配分するため、特定の保有者の実損益ではありません。</p>
        <p>大口空売りの価格帯、PTS、追証ラインは必要な明細を取得していないため表示していません。大口空売りと信用売り残は重複し得るため、合算もしません。</p>
        <p>{profile?.closeFallbackSessions ? `VWAPに必要な売買代金がない${profile.closeFallbackSessions}日には終値を使用しています。` : '対象期間の価格配分には日次VWAPを使用しています。'}平均建値が対象期間の価格範囲外の場合は分布を表示しません。</p>
        <p>出典：<a href="https://jpx-jquants.com/ja/spec/mkt-margin-int" target="_blank" rel="noreferrer">J-Quants 信用取引残高</a>・日次株価。予測モデルの学習・推論とは独立した表示です。</p>
      </section>
    </>}
  </div>;
}
