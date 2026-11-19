import React, { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  BUCKETS, PROFILE_WINDOWS, SIZE_LABELS, clearUnlock, decryptEnvelope, deriveKeyBits, loadEnvelope,
  parseVolumeProfile, saveUnlock, savedUnlock, volumeTargets,
} from '../lib/volumeProfile.js';

const SOURCE_LABEL = { tick: 'ティック（約定価格）', minute: '分足（VWAP近似）' };
const REASON = {
  calendar_history_missing: '選択期間の営業日が足りません。',
  daily_missing: '選択期間の日足が揃っていません（上場して間もない銘柄など）。',
  invalid_daily: '照合用の日足に欠損または不整合があります。',
  close_missing: '基準日の終値がありません。',
  share_basis_unknown: '選択期間の分割・併合の係数を確かめられません。',
  intraday_missing: '選択期間の分足・ティックが置き場にまだ無いか、足りません（当日分は夜の取得のあとに入ります）。',
  intraday_invalid: '分足・ティックに不正な値があります。',
  daily_totals_mismatch: '分足・ティックの合計が日足の出来高・売買代金と一致しません。',
};
const BUCKET_LABEL = { open: '寄り', morning: '前場', afternoon: '後場', close: '引け' };
const SIZE_LABEL = { small: '小口（100万円未満）', medium: '中口（100万〜1,000万円）', large: '大口（1,000万円以上）' };
const number = (n) => n.toLocaleString('ja-JP', { maximumFractionDigits: 2 });
const compact = (n) => n.toLocaleString('ja-JP', { notation: 'compact', maximumFractionDigits: 1 });
const yen = (n) => `${number(n)}円`;
const percent = (n) => `${(n * 100).toFixed(1)}%`;
const signed = (n) => `${n >= 0 ? '+' : ''}${(n * 100).toFixed(2)}%`;
const bandLabel = ([low, high]) => `${number(low)}–${number(high)}`;
const displayCode = (code) => (code.length === 5 && code.endsWith('0') ? code.slice(0, 4) : code);
const shortDate = (d) => `${Number(d.slice(5, 7))}/${Number(d.slice(8, 10))}`;
const candidateText = (c) => `${shortDate(c.date)}${c.rank ? `（${c.rank}位）` : ''}`;

/** 1つの全体を部分に分けた細い帯（数字は横の一覧に書く）。 */
function ShareBar({ label, parts }) {
  const shown = parts.filter((p) => p.value > 0);
  return <div className="vp-share">
    <div className="vp-share-bar" role="img"
      aria-label={`${label}: ${parts.map((p) => `${p.label} ${percent(p.value)}`).join('、')}`}>
      {shown.map((p) => <span key={p.key} className={`vp-seg vp-${p.key}`} style={{ flexGrow: p.value }}
        title={`${p.label} ${percent(p.value)}`} />)}
    </div>
    <ul className="vp-share-legend">
      {parts.map((p) => <li key={p.key}><i className={`vp-${p.key}`} />{p.label}<b>{percent(p.value)}</b>
        {p.note && <span className="vp-share-note">{p.note}</span>}</li>)}
    </ul>
  </div>;
}

function VolumeChart({ series, close, unit }) {
  const chartRows = series.bands.map(([low, high, shares, turnover]) => ({
    label: bandLabel([low, high]), low, high, shares, turnover,
    poc: low === series.poc[0], valueArea: low >= series.valueArea[0] && high <= series.valueArea[1],
  })).reverse();
  const closeBand = chartRows.find((row) => row.low <= close && close < row.high) ?? chartRows[0];
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
  const { asOf, dates, targets } = useMemo(() => volumeTargets(predictions), [predictions]);
  const [envelope, setEnvelope] = useState({ status: 'loading' });
  const [profile, setProfile] = useState(null);
  const [unlock, setUnlock] = useState({ status: 'idle' });
  const [entry, setEntry] = useState('');
  // 既定で端末に覚える（運用者の選択 2026-11-19「最初の1回だけ入力」）。共用の端末では外せる
  const [remember, setRemember] = useState(true);
  const [reload, setReload] = useState(0);
  const [selected, setSelected] = useState('');
  const [periodChoice, setPeriodChoice] = useState(null);
  const [source, setSource] = useState(null);
  const [unit, setUnit] = useState('shares');

  useEffect(() => {
    const abort = new AbortController();
    setEnvelope({ status: 'loading' });
    loadEnvelope(abort.signal).then(async (doc) => {
      if (abort.signal.aborted) return;
      if (!doc) { setEnvelope({ status: 'missing' }); setProfile(null); return; }
      setEnvelope({ status: 'loaded', doc });
      const bits = savedUnlock(doc);
      if (!bits) { setProfile(null); return; }
      try {
        const data = parseVolumeProfile(await decryptEnvelope(doc, bits));
        if (!abort.signal.aborted) setProfile(data);
      } catch {
        // 覚えた鍵で開けない（キーを変えた等）。覚えた鍵を消して、キーの入力に戻す
        clearUnlock();
        if (!abort.signal.aborted) setProfile(null);
      }
    }).catch((error) => {
      if (!abort.signal.aborted) setEnvelope({ status: 'error', message: error instanceof TypeError
        ? '集計のファイルに接続できませんでした。時間を置いて再読み込みしてください。' : error.message });
    });
    return () => abort.abort();
  }, [reload]);

  async function open(event) {
    event.preventDefault();
    if (envelope.status !== 'loaded') return;
    setUnlock({ status: 'working' });
    try {
      const bits = await deriveKeyBits(entry, envelope.doc);
      const data = parseVolumeProfile(await decryptEnvelope(envelope.doc, bits));
      saveUnlock(envelope.doc, bits, remember);
      setEntry('');
      setProfile(data);
      setUnlock({ status: 'idle' });
    } catch (error) {
      setUnlock({ status: 'error', message: error.message });
    }
  }

  function lock() {
    clearUnlock();
    setProfile(null);
    setUnlock({ status: 'idle' });
  }

  const byCode = useMemo(() => new Map((profile?.stocks ?? []).map((stock) => [stock.code, stock])), [profile]);
  const list = targets.length ? targets : (profile?.stocks ?? []).map(({ code, name, candidates }) => ({ code, name, candidates }));
  const target = list.find((candidate) => candidate.code === selected) ?? list[0];
  const stock = byCode.get(target?.code);
  const suggestedPeriod = [...PROFILE_WINDOWS].reverse().find((count) => stock?.windows[count].tick.status === 'ok'
    || stock?.windows[count].minute.status === 'ok');
  const days = periodChoice ?? suggestedPeriod ?? 20;
  const period = stock?.windows[days];
  const activeSource = source ?? (period?.minute.status === 'ok' && period.tick.status !== 'ok' ? 'minute' : 'tick');
  const series = period?.[activeSource];
  const tickSeries = period?.tick;
  const available = list.filter((item) => byCode.get(item.code)?.windows[days][activeSource].status === 'ok').length;
  const opened = Boolean(profile);

  function statusOf(code, kind) {
    const current = byCode.get(code)?.windows[days][kind];
    if (!current) return opened ? '集計待ち' : '–';
    return current.status === 'ok' ? '表示可' : `${current.coveredSessions}/${days}日`;
  }

  function rowValue(code) {
    const windows = byCode.get(code)?.windows[days];
    const own = windows?.[activeSource];
    return {
      vwap: own?.status === 'ok' ? signed(own.closeVsVwap) : '–',
      large: windows?.tick.status === 'ok' && windows.tick.tradeSizes.turnover > 0 ? percent(windows.tick.tradeSizes.shares[2]) : '–',
    };
  }

  return <section className="card volume-view">
    <div className="card-head"><h2>価格帯別出来高</h2></div>
    {predictionError && <p className="banner warn" role="alert">予測対象の一覧を取得できませんでした。</p>}
    {!predictions && !predictionError && <p role="status">予測対象の一覧を読み込み中です…</p>}
    {asOf && <p className="volume-note">予測基準日 {asOf} ／ 直近{dates.length}営業日（{dates[0]}〜{dates.at(-1)}）の候補 {targets.length}銘柄
      {opened && ` ／ ${days}営業日の${activeSource === 'tick' ? 'ティック' : '分足'}で表示できる銘柄 ${available}/${list.length}`}</p>}
    {asOf && !targets.length && <p>直近の予測対象の銘柄はありません。</p>}

    {envelope.status === 'loading' && <p role="status">集計を読み込み中です…</p>}
    {envelope.status === 'error' && <p className="banner warn" role="alert">{envelope.message}</p>}
    {envelope.status === 'missing' && <p className="volume-note" role="status">まだ集計がありません。毎晩の分足・ティックの取得と予測のあとに作られます。</p>}
    {envelope.status === 'loaded' && !opened && <>
      <form className="volume-connect" onSubmit={open}>
        <label>J-Quants の API キー<input type="password" autoComplete="off" spellCheck={false} value={entry}
          onChange={(event) => setEntry(event.target.value)} /></label>
        <label className="volume-remember"><input type="checkbox" checked={remember}
          onChange={(event) => setRemember(event.target.checked)} />この端末に記憶する</label>
        <button className="btn btn-primary" type="submit" disabled={unlock.status === 'working'}>
          {unlock.status === 'working' ? '開いています…' : '開く'}</button>
      </form>
      <p className="volume-note">集計は J-Quants の API キーから作った鍵で暗号化してあります。この端末で最初の1回だけキーを入れれば、
        「この端末に記憶する」を選んだままなら次からは入力なしで開きます。キーは鍵を作るのに使うだけで、どこにも送りません。
        覚えるのはキーから作った鍵だけです（キーそのものは保存しません）。集計の基準日 {envelope.doc.asOf}。</p>
      {unlock.status === 'error' && <p className="banner warn" role="alert">{unlock.message}</p>}
    </>}
    {opened && <div className="volume-connect">
      <button className="btn" type="button" disabled={envelope.status === 'loading'}
        onClick={() => setReload((value) => value + 1)}>再読み込み</button>
      <button className="btn btn-ghost" type="button" onClick={lock}>鍵を消す</button>
    </div>}
    {opened && asOf && profile.asOf !== asOf && <p className="volume-note" role="status">
      開いた集計の基準日は {profile.asOf} です（予測の基準日 {asOf} の集計は更新待ち）。</p>}

    {list.length > 0 && <>
      <div className="volume-controls">
        <label>銘柄<select aria-label="銘柄" value={target?.code ?? ''}
          onChange={(event) => { setSelected(event.target.value); setSource(null); }}>
          {list.map((item) => <option key={item.code} value={item.code}>
            {displayCode(item.code)} {item.name}（{item.candidates.map(candidateText).join('・')}）</option>)}
        </select></label>
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
        <table aria-label="直近5営業日の予測対象銘柄">
          <thead><tr><th scope="col">銘柄</th><th scope="col">候補日</th><th scope="col">終値とVWAPの差</th>
            <th scope="col">大口</th><th scope="col">ティック</th><th scope="col">分足</th></tr></thead>
          <tbody>{list.map((item) => {
            const value = rowValue(item.code);
            return <tr key={item.code}>
              <td><button className="btn btn-ghost volume-stock" type="button" aria-pressed={item.code === target?.code}
                onClick={() => { setSelected(item.code); setSource(null); }}>{displayCode(item.code)} {item.name}</button></td>
              <td>{item.candidates.map(candidateText).join('・')}</td>
              <td className="num">{value.vwap}</td><td className="num">{value.large}</td>
              <td>{statusOf(item.code, 'tick')}</td><td>{statusOf(item.code, 'minute')}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
    </>}

    {target && opened && <>
      <h3 className="volume-stock-heading">{displayCode(target.code)} {target.name}
        <span className="volume-candidates">候補日 {target.candidates.map(candidateText).join('・') || '–'}</span></h3>
      {!stock && <p className="volume-note" role="status">この銘柄の出来高は集計待ちです。</p>}
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
          <div><dt>VWAP</dt><dd>{yen(series.vwap)}</dd></div>
          <div><dt>終値とVWAPの差</dt><dd>{signed(series.closeVsVwap)}</dd></div>
          <div><dt>最多出来高の価格帯</dt><dd>{bandLabel(series.poc)}円</dd></div>
          <div><dt>出来高の70%範囲</dt><dd>{bandLabel(series.valueArea)}円</dd></div>
          <div><dt>終値±1%の出来高</dt><dd>{percent(series.nearCloseShare)}</dd></div>
          <div><dt>終値より上の出来高</dt><dd>{percent(series.aboveCloseShare)}</dd></div>
        </dl>
        <div className="vp-blocks">
          <div className="vp-block">
            <h4>時間帯別の出来高（株数）</h4>
            <ShareBar label="時間帯別の出来高" parts={BUCKETS.map((key) => ({
              key, label: BUCKET_LABEL[key], value: series.sessions[key] }))} />
            <p className="volume-note">{activeSource === 'tick'
              ? '寄り・引けは板寄せの約定。前場は前引けを、後場は後場寄りを含みます。'
              : '分足では、寄りは最初の1分（板寄せのあとの約定を含む）、引けは15:30の1分です。'}</p>
          </div>
          <div className="vp-block">
            <h4>約定代金の大きさ別（ティック・ザラバ）</h4>
            {tickSeries?.status === 'ok' && tickSeries.tradeSizes.turnover > 0 ? <>
              <ShareBar label="約定代金の大きさ別の割合" parts={SIZE_LABELS.map((key, i) => ({
                key, label: SIZE_LABEL[key], value: tickSeries.tradeSizes.shares[i],
                note: `${tickSeries.tradeSizes.counts[i].toLocaleString('ja-JP')}件` }))} />
              <p className="volume-note">板寄せを除くザラバの売買代金に占める割合。1件の約定（ティック1行）の約定代金で分けています。</p>
            </> : <p className="volume-note">{tickSeries?.status === 'ok'
              ? 'この期間にザラバの約定がありません。' : 'この期間のティックが揃っていません。'}</p>}
          </div>
        </div>
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
