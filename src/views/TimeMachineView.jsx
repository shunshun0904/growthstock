import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import RadarPanel from '../components/RadarPanel.jsx';
import AxisTable from '../components/AxisTable.jsx';
import { InstitutionalBadge, ZoneBadge } from '../components/Badges.jsx';
import { computeScores } from '../lib/scoring.js';
import { fmt, fmtOku, fmtDate, scoreColor, DASH } from '../lib/format.js';

const SNAPSHOT_ORDER = ['m6', 'm3', 'now'];
const SNAPSHOT_COLORS = { m6: '#94a3b8', m3: '#fbbf24', now: '#4f8dff' };

const EVENT_STYLE = {
  earnings:     { color: 'var(--blue)',   icon: '決算' },
  breakout:     { color: 'var(--violet)', icon: '高値' },
  volume_spike: { color: 'var(--green)',  icon: '出来高' },
};

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
                       borderRadius: 'var(--radius-sm)', padding: '5px 9px', fontSize: 13 }}
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

        <div className="card">
          <div className="card-head">
            <h2>株価推移 (直近1年)</h2>
            <span className="sub">終値・3営業日間引き</span>
          </div>
          <PriceChart history={row.stock.history} />
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h2>ストーリータイムライン</h2>
          <span className="sub">株価・出来高・決算開示から機械的に検出したイベントのみ</span>
        </div>
        <Timeline milestones={row.stock.milestones} />
      </div>
    </div>
  );
}

function PriceChart({ history }) {
  if (!history?.length) return <div className="empty">株価履歴がありません。</div>;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={history} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="var(--border-soft)" vertical={false} />
        <XAxis
          dataKey="date" tick={{ fill: 'var(--text-faint)', fontSize: 10 }}
          stroke="var(--border)" minTickGap={48}
          tickFormatter={(d) => d.slice(5).replace('-', '/')}
        />
        <YAxis
          tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border)"
          domain={['dataMin', 'dataMax']} width={56}
          tickFormatter={(v) => v.toLocaleString('ja-JP')}
        />
        <Tooltip
          contentStyle={{ background: 'var(--bg-raised)', border: '1px solid var(--border)',
                          borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: 'var(--text-dim)' }}
          formatter={(v) => [`${v.toLocaleString('ja-JP')}円`, '終値']}
        />
        <Line type="monotone" dataKey="close" stroke="var(--accent)" strokeWidth={1.6}
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
        （決算開示・52週高値更新・出来高急増のいずれも直近1年で条件を満たしていません）
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
