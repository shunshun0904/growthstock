import React, { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { MAX_PROFILE_FILE_BYTES, PROFILE_WINDOWS, parseVolumeProfile, volumeCode } from '../lib/volumeProfile.js';

const SOURCE_LABEL = { tick: 'ティック（約定価格）', minute: '分足（VWAP近似）' };
const REASON = {
  calendar_history_missing: '選択期間の取引カレンダーが不足しています。',
  daily_missing: '選択期間の照合用の日足が揃っていません。',
  not_available_at_cutoff: '基準時点までにデータを確認できていません。',
  invalid_daily: '照合用の日足に欠損または不整合があります。',
  close_missing: '基準日の終値がありません。',
  share_basis_unknown: '選択期間の株式数の基準を揃えられません。',
  intraday_missing: '選択期間の日足と照合済みの分足・ティックが不足しています。',
  daily_totals_mismatch: '集計が日足の出来高・売買代金と一致しません。',
};
const number = (n) => n.toLocaleString('ja-JP', { maximumFractionDigits: 2 });
const compact = (n) => n.toLocaleString('ja-JP', { notation: 'compact', maximumFractionDigits: 1 });
const yen = (n) => `${number(n)}円`;
const percent = (n) => `${(n * 100).toFixed(1)}%`;
const bandLabel = ([low, high]) => `${number(low)}–${number(high)}`;
const displayCode = (code) => code.endsWith('0') ? code.slice(0, 4) : code;

function VolumeChart({ series, close, unit }) {
  const chartRows = series.bands.map(([low, high, shares, turnover]) => ({
    label: bandLabel([low, high]), low, high, shares, turnover,
    poc: low === series.poc[0], valueArea: low >= series.valueArea[0] && high <= series.valueArea[1],
  })).reverse();
  const closeBand = chartRows.find((row) => row.low <= close && close < row.high);
  const quantity = unit === 'shares' ? '出来高（株）' : '売買代金（円）';
  return <>
    <div className="volume-legend">
      <span><i className="volume-poc" />最多出来高の帯</span>
      <span><i className="volume-area" />出来高の70%範囲</span>
      <span><i className="volume-close" />終値を含む帯</span>
    </div>
    <div className="volume-chart" aria-label={`価格帯別${quantity}`}>
      <ResponsiveContainer width="100%" height={Math.max(270, Math.min(560, chartRows.length * 17 + 44))}>
        <BarChart data={chartRows} layout="vertical" margin={{ top: 8, right: 15, bottom: 10, left: 0 }}>
          <CartesianGrid stroke="var(--border-soft)" horizontal={false} />
          <YAxis type="category" dataKey="label" width={105} tick={{ fill: 'var(--text-dim)', fontSize: 11 }}
            interval="preserveStartEnd" tickLine={false} minTickGap={6} />
          <XAxis type="number" tickFormatter={compact} tick={{ fill: 'var(--text-dim)', fontSize: 11 }}
            label={{ value: quantity, position: 'insideBottom', offset: -7, fill: 'var(--text-dim)', fontSize: 11 }} />
          <Tooltip cursor={{ fill: 'var(--bg-raised)' }} content={({ active, payload }) => {
            const row = payload?.[0]?.payload;
            if (!active || !row) return null;
            return <div className="volume-tooltip">
              <strong>{yen(row.low)}〜{yen(row.high)}未満</strong>
              <div>{number(row.shares)}株（{percent(row.shares / series.shares)}）</div>
              <div>{yen(row.turnover)}</div>
            </div>;
          }} />
          <Bar dataKey={unit} isAnimationActive={false} maxBarSize={24} radius={[0, 3, 3, 0]}>
            {chartRows.map((row) => <Cell key={row.label}
              fill={row.poc ? 'var(--amber)' : row.valueArea ? 'var(--blue)' : 'var(--volume-muted)'} />)}
          </Bar>
          {closeBand && <ReferenceLine y={closeBand.label} stroke="var(--text)" strokeDasharray="4 3" strokeWidth={1.3} />}
        </BarChart>
      </ResponsiveContainer>
    </div>
    <details className="volume-table">
      <summary>価格帯別の数値</summary>
      <div className="volume-table-scroll">
        <table>
          <thead><tr><th>価格帯（円・上限未満）</th><th>出来高（株）</th><th>売買代金（円）</th></tr></thead>
          <tbody>{chartRows.map((row) => <tr key={row.label}>
            <td>{row.label}{row === closeBand ? ' ・終値' : ''}</td>
            <td>{number(row.shares)}</td><td>{number(row.turnover)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </details>
  </>;
}

export default function VolumeProfileView({ data, onData, stocks, candidates, selectedId }) {
  const [selected, setSelected] = useState('');
  const [query, setQuery] = useState('');
  const [days, setDays] = useState(20);
  const [source, setSource] = useState(null);
  const [unit, setUnit] = useState('shares');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const names = useMemo(() => new Map([...stocks, ...(candidates ?? [])]
    .filter((stock) => stock.code && stock.name)
    .map((stock) => [volumeCode(stock.code), stock.name])), [stocks, candidates]);
  const choices = (data?.stocks ?? []).map((stock) => ({ ...stock, name: stock.name || names.get(stock.code) || '' }));
  const search = query.trim().toUpperCase();
  const matching = choices.filter((stock) => `${stock.code} ${stock.name}`.toUpperCase().includes(search));
  const preferred = selected || volumeCode(stocks.find((stock) => stock.id === selectedId)?.code);
  const stock = matching.find((candidate) => candidate.code === preferred) ?? matching[0];
  const period = stock?.windows[days];
  const activeSource = source ?? (period?.tick.status === 'ok' || period?.minute.status !== 'ok' ? 'tick' : 'minute');
  const series = period?.[activeSource];

  async function readFile(event) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    setError('');
    setBusy(true);
    try {
      if (file.size > MAX_PROFILE_FILE_BYTES) throw new Error('12MB以内の集計ファイルを選んでください。');
      const next = parseVolumeProfile(await file.text());
      onData(next);
      setSelected('');
      setSource(null);
      setQuery('');
      const available = [...PROFILE_WINDOWS].reverse().find((count) => next.stocks.some((item) =>
        item.windows[count].tick.status === 'ok' || item.windows[count].minute.status === 'ok'));
      setDays(available ?? 20);
    } catch (problem) {
      setError(problem.message || 'ファイルを読み込めませんでした。');
    } finally {
      setBusy(false);
    }
  }

  return <section className="card volume-view">
    <div className="card-head"><h2>価格帯別出来高</h2></div>
    <div className="volume-import">
      <label className="btn volume-file">集計ファイルを選択
        <input className="sr-only" type="file" accept=".json,application/json" onChange={readFile} disabled={busy} />
      </label>
      {data && <button className="btn btn-ghost" type="button" onClick={() => { onData(null); setError(''); }}>読み込みを解除</button>}
    </div>
    {error && <p className="banner warn" role="alert">{error}</p>}
    {busy && <p role="status">集計ファイルを読み込み中です…</p>}
    {!data ? <p className="volume-note">
      手元で作成した volume-profile.json を選ぶと、分足・ティックの集計がこのブラウザに表示されます。{' '}
      <a href="https://github.com/shunshun0904/growthstock/blob/claude/jquants-browser-app-bwk19a/docs/VOLUME_PROFILE.md"
        target="_blank" rel="noreferrer">集計ファイルの作り方</a>
    </p> : <>
      <p className="volume-note">基準日 {data.asOf}{data.demo === true ? ' ／ 合成データの表示例' : ''}</p>
      <div className="volume-controls">
        <label>銘柄検索<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="銘柄名・コード" /></label>
        <label>分析対象銘柄<select value={stock?.code ?? ''} onChange={(event) => setSelected(event.target.value)} disabled={!stock}>
          {!stock && <option value="">該当する銘柄なし</option>}
          {matching.map((item) => <option key={item.code} value={item.code}>{displayCode(item.code)} {item.name}</option>)}
        </select></label>
        <label>集計期間<select value={days} onChange={(event) => setDays(Number(event.target.value))}>
          {PROFILE_WINDOWS.map((count) => <option key={count} value={count}>{count}営業日</option>)}
        </select></label>
        <label>データ種別<select value={activeSource} onChange={(event) => setSource(event.target.value)}>
          {Object.entries(SOURCE_LABEL).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
        </select></label>
        <label>横軸<select value={unit} onChange={(event) => setUnit(event.target.value)}>
          <option value="shares">出来高（株）</option><option value="turnover">売買代金（円）</option>
        </select></label>
      </div>
      {stock && series.status !== 'ok' && <div className="volume-empty" role="status">
        <strong>{SOURCE_LABEL[activeSource]}／{days}営業日</strong>
        <p>{REASON[series.status]}</p>
        <span>照合済み {series.coveredSessions}/{days}営業日</span>
      </div>}
      {stock && series.status === 'ok' && <>
        <p className="volume-note">{series.from}〜{series.to} ／ 日足照合 {series.coveredSessions}/{days}営業日
          {series.adjustedShares === true ? ' ／ 分割・併合を基準日の株数に換算' : ''}</p>
        <dl className="volume-metrics">
          <div><dt>終値</dt><dd>{yen(stock.close)}</dd></div>
          <div><dt>最多出来高の価格帯</dt><dd>{bandLabel(series.poc)}円</dd></div>
          <div><dt>VWAP</dt><dd>{yen(series.vwap)}</dd></div>
          <div><dt>出来高の70%範囲</dt><dd>{bandLabel(series.valueArea)}円</dd></div>
          <div><dt>終値±1%の出来高</dt><dd>{percent(series.nearCloseShare)}</dd></div>
          <div><dt>終値より上の出来高</dt><dd>{percent(series.aboveCloseShare)}</dd></div>
        </dl>
        <VolumeChart series={series} close={stock.close} unit={unit} />
        <p className="volume-note">{activeSource === 'tick'
          ? '東証の立会内で成立した約定の数量を、約定価格ごとに集計しています。'
          : '各分の出来高を、その分の売買代金÷出来高に割り当てた近似です。'}
          {series.tickMinuteDistance !== undefined && ` ティックと分足の分布差 ${percent(series.tickMinuteDistance)}。`}
        </p>
      </>}
    </>}
  </section>;
}
