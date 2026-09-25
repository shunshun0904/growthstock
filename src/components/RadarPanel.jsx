import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip,
} from 'recharts';
import { AXES } from '../lib/scoring.js';

/**
 * 8軸オクタゴン。series は
 *   [{ name, color, scores: {eps, sales, ...}, dashed?: bool }]
 * の配列。値が null の軸は 0 として描画しつつ、ツールチップでは「データなし」と表示する。
 */
export default function RadarPanel({ series, height = 420 }) {
  // height に '100%' を渡すと親 (.chart-slot) の高さいっぱいに広がる
  const data = AXES.map((axis) => {
    const row = { axis: axis.label, _key: axis.key };
    for (const s of series) {
      const v = s.scores?.[axis.key];
      row[s.name] = Number.isFinite(v) ? Math.round(v * 10) / 10 : 0;
      row[`__na_${s.name}`] = !Number.isFinite(v);
    }
    return row;
  });

  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload;
    return (
      <div className="card" style={{ padding: '8px 10px', fontSize: 12 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
        {payload.map((p) => (
          <div key={p.name} style={{ color: p.color, fontFamily: 'var(--mono)' }}>
            {p.name}: {row[`__na_${p.name}`] ? 'データなし' : p.value.toFixed(1)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      {/* 左右の余白は軸ラベルのぶん。スマホ幅では半径が横幅で決まるので、
          余白が無いと左の「信用倍率」が画面の外に切れる（390px で実測）。
          PC では半径が高さで決まるので、この余白で図は小さくならない */}
      <RadarChart data={data} outerRadius="72%" margin={{ top: 4, right: 44, bottom: 4, left: 44 }}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--text-dim)', fontSize: 11.5 }} />
        <PolarRadiusAxis
          domain={[0, 10]} tickCount={6} angle={90}
          tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border-soft)"
        />
        {series.map((s) => (
          <Radar
            key={s.name}
            name={s.name}
            dataKey={s.name}
            stroke={s.color}
            fill={s.color}
            fillOpacity={series.length > 2 ? 0.1 : 0.18}
            strokeWidth={2}
            strokeDasharray={s.dashed ? '5 4' : undefined}
            isAnimationActive={false}
            dot={series.length <= 3}
          />
        ))}
        <Tooltip content={renderTooltip} />
        {/* 凡例は左右の余白を使わず横幅いっぱいに置く。余白の内側に置くと、
            スマホ幅で銘柄名の折り返しが増えて下の「テクニカル」に重なる（360px で実測） */}
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8, left: 0, width: '100%' }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
