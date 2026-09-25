import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts';
import { AXES } from '../lib/scoring.js';
import {
  NARROW_WIDTH, LABEL_FONT, LINE_H, TICK_GAP, labelLines, radarRadius,
} from '../lib/radar.js';

/** 軸名。頂点の外側に置き、上の軸は上へ・下の軸は下へ・横の軸は縦の中央にそろえる */
function AngleTick({ x, y, payload, textAnchor, narrow }) {
  const lines = labelLines(payload.value, narrow);
  const sin = Math.sin((payload.coordinate * Math.PI) / 180);   // 上が +1
  // 1段目のベースラインの位置。上の軸は最後の行が頂点の上に来るように
  let first;
  if (sin > 0.5) first = -(lines.length - 1) * LINE_H - 3;
  else if (sin < -0.5) first = LABEL_FONT;
  else first = LABEL_FONT * 0.35 - ((lines.length - 1) * LINE_H) / 2;
  return (
    <text x={x} y={y} textAnchor={textAnchor} fill="var(--text-dim)" fontSize={LABEL_FONT}>
      {lines.map((l, i) => (
        <tspan key={l} x={x} dy={i === 0 ? first : LINE_H}>{l}</tspan>
      ))}
    </text>
  );
}

/**
 * 8軸オクタゴン。series は
 *   [{ name, color, scores: {eps, sales, ...}, dashed?: bool }]
 * の配列。
 *
 * 値が無い軸（データなし）は描かない。以前は 0 として描いていたので、線が中心まで落ちて
 * 八角形が崩れて見え、しかも「0点」と区別がつかなかった（9/26 の実データでは 198 時点の
 * うち 126 に値の無い軸がある）。無い軸は隣どうしを結び、ツールチップで「データなし」と出す。
 */
export default function RadarPanel({ series, height = 420 }) {
  // height に '100%' を渡すと親 (.chart-slot) の高さいっぱいに広がる
  const data = AXES.map((axis) => {
    const row = { axis: axis.label, _key: axis.key };
    for (const s of series) {
      const v = s.scores?.[axis.key];
      row[s.name] = Number.isFinite(v) ? Math.round(v * 10) / 10 : null;
      row[`__na_${s.name}`] = !Number.isFinite(v);
    }
    return row;
  });
  const missing = series.filter((s) => AXES.some((a) => !Number.isFinite(s.scores?.[a.key])));

  return (
    <div className="radar-panel" style={height === '100%' ? undefined : { height }}>
      <div className="radar-plot">
        <div className="radar-plot-inner">
          <ResponsiveContainer width="100%" height="100%">
            <RadarBody data={data} series={series} />
          </ResponsiveContainer>
        </div>
      </div>
      {/* 凡例は図の外（HTML）に置く。図の中に置くと高さが読めず、八角形の大きさを決められない */}
      <ul className="radar-legend">
        {series.map((s) => (
          <li key={s.name} style={{ color: s.color }}>
            <i style={{ borderTopColor: s.color, borderTopStyle: s.dashed ? 'dashed' : 'solid' }} />
            {s.name}
          </li>
        ))}
      </ul>
      {missing.length > 0 && (
        <p className="radar-note">
          点の無い軸はデータなし（0点ではありません）
        </p>
      )}
    </div>
  );
}

/** 目盛りの数字。軸に沿って傾けず、まっすぐ置く（0 は中心なので出さない） */
function RadiusTick({ x, y, payload }) {
  if (!payload || payload.value === 0) return null;
  return (
    <text x={x + 4} y={y} dy="0.35em" textAnchor="start" fill="var(--text-faint)" fontSize={10}>
      {payload.value}
    </text>
  );
}

/** ResponsiveContainer から width / height を受け取って描く */
function RadarBody({ width, height, data, series }) {
  if (!width || !height) return null;
  const narrow = width < NARROW_WIDTH;
  const outerRadius = radarRadius(width, height, narrow);

  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload;
    return (
      <div className="card" style={{ padding: '8px 10px', fontSize: 12 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
        {series.map((s) => (
          <div key={s.name} style={{ color: s.color, fontFamily: 'var(--mono)' }}>
            {s.name}: {row[`__na_${s.name}`] ? 'データなし' : row[s.name].toFixed(1)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <RadarChart width={width} height={height} data={data} outerRadius={outerRadius}
                margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
      <PolarGrid stroke="var(--border)" />
      <PolarAngleAxis
        dataKey="axis" tickSize={TICK_GAP}
        tick={(p) => <AngleTick {...p} narrow={narrow} />}
      />
      {/* 目盛りは軸と軸のあいだ（67.5°）に置く。真上（EPS成長の軸）に置くと、
          線と軸名に数字が重なっていた */}
      <PolarRadiusAxis
        domain={[0, 10]} tickCount={narrow ? 3 : 6} angle={67.5} stroke="var(--border-soft)"
        tick={RadiusTick}
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
          connectNulls
          dot={series.length <= 3}
        />
      ))}
      <Tooltip content={renderTooltip} />
    </RadarChart>
  );
}
