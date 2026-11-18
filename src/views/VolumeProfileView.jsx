import React, { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PROFILE_REPO, PROFILE_WINDOWS, latestVolumeTargets, loadVolumeProfile } from '../lib/volumeProfile.js';

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
const SESSION_KEY = `growthstock.volume-profile.${PROFILE_REPO}`;
const GUIDE = 'https://github.com/shunshun0904/growthstock/blob/claude/jquants-browser-app-bwk19a/docs/VOLUME_PROFILE.md';

function savedToken() {
  try { return sessionStorage.getItem(SESSION_KEY) || ''; }
  catch { return ''; }
}

function saveToken(value) {
  try {
    if (value) sessionStorage.setItem(SESSION_KEY, value);
    else sessionStorage.removeItem(SESSION_KEY);
  } catch { /* The view still works when browser storage is disabled. */ }
}

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

export default function VolumeProfileView({ predictions, predictionError }) {
  const [selected, setSelected] = useState('');
  const [query, setQuery] = useState('');
  const [periodChoice, setPeriodChoice] = useState(null);
  const [source, setSource] = useState(null);
  const [unit, setUnit] = useState('shares');
  const [token, setToken] = useState(savedToken);
  const [entry, setEntry] = useState('');
  const [tokenError, setTokenError] = useState('');
  const [request, setRequest] = useState({ status: 'idle' });
  const [reload, setReload] = useState(0);
  const targets = useMemo(() => latestVolumeTargets(predictions), [predictions]);
  const day = predictions?.asOf;
  const hasTargets = targets.length > 0;

  useEffect(() => {
    if (!token || !day || !hasTargets) return undefined;
    const abort = new AbortController();
    setRequest({ status: 'loading' });
    loadVolumeProfile(token, abort.signal).then((data) => {
      if (abort.signal.aborted) return;
      saveToken(token);
      setRequest({ status: 'loaded', data });
    }).catch((error) => {
      if (!abort.signal.aborted) setRequest({ status: 'error', message: error instanceof TypeError
        ? 'GitHubに接続できませんでした。時間を置いて再取得してください。' : error.message });
    });
    return () => abort.abort();
  }, [token, day, hasTargets, reload]);

  const data = request.data?.asOf === day ? request.data : null;
  const byCode = useMemo(() => new Map((data?.stocks ?? []).map((stock) => [stock.code, stock])), [data]);
  const search = query.trim().toUpperCase();
  const matching = targets.filter((stock) => `${stock.code} ${stock.name}`.toUpperCase().includes(search));
  const target = matching.find((candidate) => candidate.code === selected) ?? matching[0];
  const stock = byCode.get(target?.code);
  const suggestedPeriod = [...PROFILE_WINDOWS].reverse().find((count) => stock?.windows[count].tick.status === 'ok'
    || stock?.windows[count].minute.status === 'ok');
  const days = periodChoice ?? suggestedPeriod ?? 20;
  const period = stock?.windows[days];
  const activeSource = source ?? (period?.minute.status === 'ok' && period.tick.status !== 'ok' ? 'minute' : 'tick');
  const series = period?.[activeSource];
  const available = targets.filter((item) => byCode.get(item.code)?.windows[days][activeSource].status === 'ok').length;

  function connect(event) {
    event.preventDefault();
    const value = entry.trim();
    if (!/^[A-Za-z0-9_-]{20,255}$/.test(value)) {
      setTokenError('GitHubで発行した閲覧用トークンを入力してください。');
      return;
    }
    setTokenError('');
    setEntry('');
    setToken(value);
  }

  function disconnect() {
    saveToken('');
    setToken('');
    setRequest({ status: 'idle' });
  }

  function statusOf(code, kind) {
    const current = byCode.get(code)?.windows[days][kind];
    if (!current) return token ? '集計待ち' : '接続待ち';
    return current.status === 'ok' ? '表示可' : `${current.coveredSessions}/${days}日`;
  }

  return <section className="card volume-view">
    <div className="card-head"><h2>価格帯別出来高</h2></div>
    {predictionError && <p className="banner warn" role="alert">予測対象一覧を取得できませんでした。</p>}
    {!predictions && !predictionError && <p role="status">予測対象一覧を読み込み中です…</p>}
    {day && <p className="volume-note">予測基準日 {day} ／ 対象 {targets.length}銘柄
      {data && ` ／ ${days}営業日の${activeSource === 'tick' ? 'ティック' : '分足'} ${available}/${targets.length}銘柄`}</p>}
    {day && !hasTargets && <p>この日の予測対象銘柄はありません。</p>}
    {hasTargets && <>
      {!token ? <form className="volume-connect" onSubmit={connect}>
        <label>閲覧用GitHubトークン<input type="password" autoComplete="off" required value={entry}
          onChange={(event) => setEntry(event.target.value)} /></label>
        <button className="btn btn-primary" type="submit">接続</button>
        <a href={GUIDE} target="_blank" rel="noreferrer">非公開データの接続手順</a>
      </form> : <div className="volume-connect">
        <button className="btn" type="button" disabled={request.status === 'loading'}
          onClick={() => setReload((value) => value + 1)}>再取得</button>
        <button className="btn btn-ghost" type="button" onClick={disconnect}>接続を解除</button>
      </div>}
      {tokenError && <p className="banner warn" role="alert">{tokenError}</p>}
      {request.status === 'loading' && <p role="status">価格帯別出来高を読み込み中です…</p>}
      {request.status === 'error' && <p className="banner warn" role="alert">{request.message}</p>}
      {request.data && !data && <p className="volume-note" role="status">{day}の出来高は集計待ちです。保存済みの基準日は{request.data.asOf}です。</p>}
      <div className="volume-controls">
        <label>銘柄検索<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="銘柄名・コード" /></label>
        <label>集計期間<select value={days} onChange={(event) => setPeriodChoice(Number(event.target.value))}>
          {PROFILE_WINDOWS.map((count) => <option key={count} value={count}>{count}営業日</option>)}
        </select></label>
        <label>データ種別<select value={activeSource} onChange={(event) => setSource(event.target.value)}>
          {Object.entries(SOURCE_LABEL).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
        </select></label>
        <label>横軸<select value={unit} onChange={(event) => setUnit(event.target.value)}>
          <option value="shares">出来高（株）</option><option value="turnover">売買代金（円）</option>
        </select></label>
      </div>
      <div className="volume-list volume-table-scroll">
        <table aria-label="当日の予測対象銘柄">
          <thead><tr><th scope="col">予測順位</th><th scope="col">銘柄</th><th scope="col">ティック</th><th scope="col">分足</th></tr></thead>
          <tbody>{matching.map((item) => <tr key={item.code}>
            <td>{item.rank ?? '–'}</td>
            <td><button className="btn btn-ghost volume-stock" type="button" aria-pressed={item.code === target?.code}
              onClick={() => { setSelected(item.code); setSource(null); }}>{displayCode(item.code)} {item.name}</button></td>
            <td>{statusOf(item.code, 'tick')}</td><td>{statusOf(item.code, 'minute')}</td>
          </tr>)}</tbody>
        </table>
        {!matching.length && <p>検索条件に一致する銘柄はありません。</p>}
      </div>
      {target && <h3 className="volume-stock-heading">{displayCode(target.code)} {target.name}</h3>}
      {target && data && !stock && <p className="volume-note" role="status">この銘柄の出来高は集計待ちです。</p>}
      {series && series.status !== 'ok' && <div className="volume-empty" role="status">
        <strong>{SOURCE_LABEL[activeSource]}／{days}営業日</strong>
        <p>{REASON[series.status]}</p>
        <span>照合済み {series.coveredSessions}/{days}営業日</span>
      </div>}
      {series?.status === 'ok' && <>
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
