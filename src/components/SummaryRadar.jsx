import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from 'recharts';
import { SUMMARY_AXES } from '../lib/summary.js';

/**
 * サマリーの六角形（候補1件）。6つの頂点はどれも 0〜1 で外側ほど良い（src/lib/summary.js）。タイムマシーンの8軸の図
 * （RadarPanel）とは別（運用者の依頼 2026-11-21「タイムマシーンのタグのチャートとは別です」）。
 * 値の無い頂点は描かずに隣どうしを結び、頂点の名前に「なし」と添える（0 と見分けがつかなくならないように。RadarPanel と同じ扱い）。
 */
export default function SummaryRadar({ row, height = 230 }) {
  const data = SUMMARY_AXES.map((a) => ({ key: a.key, axis: a.short, label: a.label, value: row.axes[a.key] }));
  const missing = new Set(data.filter((d) => d.value === null).map((d) => d.axis));

  const Tick = ({ x, y, payload, textAnchor }) => {
    const na = missing.has(payload.value);
    const sin = Math.sin((payload.coordinate * Math.PI) / 180);
    const dy = sin > 0.5 ? -4 : sin < -0.5 ? 12 : 4;
    return (
      <text x={x} y={y + dy} textAnchor={textAnchor} fontSize={11.5}
            fill={na ? 'var(--text-faint)' : 'var(--text-dim)'}>
        {na ? `${payload.value}（なし）` : payload.value}
      </text>
    );
  };

  const tip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="summary-tip">
        <div>{d.label}</div>
        <div className="num">{d.value === null ? 'なし' : d.value.toFixed(2)}</div>
      </div>
    );
  };

  return (
    <div className="summary-radar" style={{ height }} role="img"
         aria-label={`${row.c.code} ${row.c.name ?? ''} の六角形: `
           + data.map((d) => `${d.label} ${d.value === null ? 'なし' : d.value.toFixed(2)}`).join('、')}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius="66%" margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="axis" tick={Tick} tickSize={6} />
          <PolarRadiusAxis domain={[0, 1]} tickCount={5} tick={false} axisLine={false} />
          <Radar dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.22} strokeWidth={2}
                 connectNulls isAnimationActive={false} dot={{ r: 2.5, fill: 'var(--accent)', strokeWidth: 0 }} />
          <Tooltip content={tip} cursor={false} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
