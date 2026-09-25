import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import RadarPanel from '../components/RadarPanel.jsx';
import AxisTable from '../components/AxisTable.jsx';
import { InstitutionalBadge, ZoneBadge } from '../components/Badges.jsx';
import { computeScores } from '../lib/scoring.js';
import { fmt, fmtOku, fmtDate, scoreColor, DASH } from '../lib/format.js';
import { buildPriceSeries, quarterTicks, fmtTick } from '../lib/pricechart.js';

const SNAPSHOT_ORDER = ['m6', 'm3', 'now'];
const SNAPSHOT_COLORS = { m6: '#94a3b8', m3: '#fbbf24', now: '#4f8dff' };

const EVENT_STYLE = {
  // 決算は株価の線（青）と見分けるため琥珀色にする（タイムラインの印も同じ色）
  earnings:     { color: 'var(--amber)',  icon: '決算',   label: '決算発表' },
  breakout:     { color: 'var(--violet)', icon: '高値',   label: '78週高値の更新' },
  volume_spike: { color: 'var(--green)',  icon: '出来高', label: '出来高急増' },
};
const EVENT_ORDER = ['earnings', 'breakout', 'volume_spike'];

/** 5.2 タイムマシーン・モード View */
export default function TimeMachineView({ rows, selectedId, onSelect }) {
  const row = rows.find((r) => r.stock.id === selectedId) || rows[0];

  const snapshots = useMemo(() => {
    if (!row) return [];
    return SNAPSHOT_ORDER
      .map((key) => {
        const snap = row.stock.snapshots?.[key];
        if (!snap) return null;
        return { key, snap, result: computeScores(snap) };
      })
      .filter(Boolean);
  }, [row]);

  if (!row) return <div className="empty">銘柄がありません。</div>;

  const hasHistory = snapshots.length > 1;
  const series = snapshots.map(({ key, snap, result }) => ({
    name: `${snap.label} (${fmtDate(snap.asOf)})`,
    color: SNAPSHOT_COLORS[key],
    scores: result.scores,
    dashed: key !== 'now',
  }));

  const current = snapshots.find((s) => s.key === 'now');
  const oldest = snapshots[0];

  return (
    <div className="stack" style={{ gap: 'var(--s4)' }}>
      <div className="card">
        <div className="card-head">
          <h2>タイムマシーン・モード</h2>
          <div className="row">
            <label htmlFor="tm-stock" style={{ fontSize: 12, color: 'var(--text-dim)' }}>銘柄</label>
            <select
              id="tm-stock"
              value={row.stock.id}
              onChange={(e) => onSelect(e.target.value)}
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)',
                       borderRadius: 'var(--radius-sm)', padding: '5px 9px', fontSize: 13,
                       flex: '1 1 auto', minWidth: 0, maxWidth: '100%' }}
            >
              {rows.map((r) => (
                <option key={r.stock.id} value={r.stock.id}>{r.stock.code} {r.stock.name}</option>
              ))}
            </select>
          </div>
        </div>

        {!hasHistory && (
          <div className="banner warn" style={{ marginBottom: 'var(--s4)' }}>
            <span>⚠</span>
            <div>
              この銘柄には過去時点のスナップショットがありません（手入力銘柄、または株価履歴が不足）。
              過去比較は J-Quants から取得した銘柄でのみ利用できます。
            </div>
          </div>
        )}

        <div className="split split-fill">
          <div className="chart-slot">
            <RadarPanel series={series} height="100%" />
          </div>
          <div className="stack">
            {snapshots.map(({ key, snap, result }) => (
              <div className="card" key={key} style={{ borderLeft: `3px solid ${SNAPSHOT_COLORS[key]}` }}>
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{snap.label}</div>
                    <div className="stock-code">{fmtDate(snap.asOf)}</div>
                  </div>
                  <div className="score-big" style={{ color: scoreColor(result.totalScore) }}>
                    {result.totalScore === null ? DASH : result.totalScore.toFixed(1)}
                  </div>
                </div>
                <dl style={{ margin: '8px 0 0', display: 'grid', gap: 3 }}>
                  <div className="metric-row"><dt>株価</dt><dd>{snap.price == null ? DASH : `${fmt(snap.price, 0)}円`}</dd></div>
                  <div className="metric-row"><dt>高値接近率</dt><dd>{fmt(snap.highRatio, 1, '%')}</dd></div>
                  <div className="metric-row"><dt>売買代金</dt><dd>{fmtOku(snap.tradingValue)}</dd></div>
                  <div className="metric-row"><dt>決算</dt><dd>{snap.fiscalPeriod || DASH}</dd></div>
                </dl>
                <div className="row" style={{ marginTop: 8, gap: 5 }}>
                  <ZoneBadge zone={result.zone} compact />
                  <InstitutionalBadge level={result.institutional} compact />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <h2>軸別の変化 — {oldest?.snap.label} → 現在</h2>
          </div>
          {current && oldest && oldest.key !== 'now' ? (
            <AxisTable axisScores={current.result.axisScores} compareTo={oldest.result.scores} />
          ) : (
            <div className="empty">比較できる過去時点がありません。</div>
          )}
        </div>

        <PriceCard history={row.stock.history} milestones={row.stock.milestones}
                   bars={row.stock.historyBars} />
      </div>

      <div className="card">
        <div className="card-head">
          <h2>ストーリータイムライン</h2>
          <span className="sub">株価・出来高・決算開示から機械的に検出したイベントのみ（株価推移の破線と同じ）</span>
        </div>
        <Timeline milestones={row.stock.milestones} />
      </div>
    </div>
  );
}

function PriceCard({ history, milestones, bars }) {
  const { data, events, span } = useMemo(
    () => buildPriceSeries(history, milestones, bars), [history, milestones, bars]);
  const counts = EVENT_ORDER
    .map((type) => [type, events.filter((e) => e.type === type).length])
    .filter(([, n]) => n > 0);
  return (
    <div className="card">
      <div className="card-head">
        <h2>株価推移{span ? `（直近${span.weeks}週）` : ''}</h2>
        {span && (
          <span className="sub">{fmtDate(span.from)}〜{fmtDate(span.to)}・終値（3営業日ごと）</span>
        )}
      </div>
      {counts.length > 0 && (
        <div className="event-legend">
          {counts.map(([type, n]) => (
            <span key={type} style={{ color: EVENT_STYLE[type].color }}>
              <i style={{ borderLeftColor: EVENT_STYLE[type].color }} />
              {EVENT_STYLE[type].label} {n}
            </span>
          ))}
        </div>
      )}
      <PriceChart data={data} events={events} span={span} />
    </div>
  );
}

function PriceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="card" style={{ padding: '8px 10px', fontSize: 12, maxWidth: 240 }}>
      <div style={{ color: 'var(--text-dim)' }}>{fmtDate(p.date)}</div>
      <div style={{ fontFamily: 'var(--mono)' }}>終値 {p.close.toLocaleString('ja-JP')}円</div>
      {p.events.map((e, i) => (
        <div key={i} style={{ color: EVENT_STYLE[e.type]?.color, marginTop: 4 }}>
          {fmtDate(e.date)} {e.title}
        </div>
      ))}
    </div>
  );
}

function PriceChart({ data, events, span }) {
  if (!data.length) return <div className="empty">株価履歴がありません。</div>;
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="var(--border-soft)" vertical={false} />
        <XAxis
          dataKey="t" type="number" scale="time" domain={[span.t0, span.t1]}
          ticks={quarterTicks(span.t0, span.t1)} tickFormatter={fmtTick}
          tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border)"
        />
        {/* 目盛りは切りのよい値に（'dataMin'/'dataMax' だと 2,008.5 のような目盛りになった） */}
        <YAxis
          tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border)"
          domain={['auto', 'auto']} width={56} allowDecimals={false}
          tickFormatter={(v) => Math.round(v).toLocaleString('ja-JP')}
        />
        {/* 出来事の日の縦の破線。線より先に描いて、株価の線を上に重ねる */}
        {events.map((e, i) => (
          <ReferenceLine
            key={`${e.date}-${e.type}-${i}`} x={e.t} ifOverflow="hidden"
            stroke={EVENT_STYLE[e.type]?.color || 'var(--slate)'}
            strokeDasharray="3 3" strokeOpacity={0.75}
          />
        ))}
        <Tooltip content={<PriceTooltip />} />
        {/* 点は3営業日ごとなので直線で結ぶ（曲線で補うと実際に無い高値・安値が描かれる） */}
        <Line type="linear" dataKey="close" stroke="var(--accent)" strokeWidth={1.6}
              dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function Timeline({ milestones }) {
  if (!milestones?.length) {
    return (
      <div className="empty">
        検出可能なイベントがありませんでした。<br />
        （決算開示・78週高値更新・出来高急増のいずれも直近78週で条件を満たしていません）
      </div>
    );
  }
  const ordered = [...milestones].sort((a, b) => (a.date < b.date ? 1 : -1));
  return (
    <div className="timeline">
      {ordered.map((e, i) => {
        const style = EVENT_STYLE[e.type] || { color: 'var(--slate)', icon: '—' };
        return (
          <div className="tl-item" key={`${e.date}-${e.type}-${i}`} style={{ color: style.color }}>
            <div className="tl-date">{fmtDate(e.date)}</div>
            <div className="tl-title" style={{ color: 'var(--text)' }}>
              <span className="chip" style={{ color: style.color, marginRight: 6 }}>{style.icon}</span>
              {e.title}
            </div>
            <div className="tl-detail">{e.detail}</div>
          </div>
        );
      })}
    </div>
  );
}
