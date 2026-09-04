import React from 'react';
import { INSTITUTIONAL_META, ZONE_META } from '../lib/scoring.js';

export function InstitutionalBadge({ level, compact = false }) {
  const meta = INSTITUTIONAL_META[level];
  if (!meta) return <span className="chip" title="売買代金が取得できていません">機関判定 —</span>;
  return (
    <span className={`badge ${meta.tone}`} title={meta.desc}>
      {compact ? meta.short : meta.label}
    </span>
  );
}

export function ZoneBadge({ zone, compact = false }) {
  const meta = ZONE_META[zone];
  if (!meta) return <span className="chip" title="52週高値接近率が取得できていません">ゾーン —</span>;
  return (
    <span className={`badge ${meta.tone}`} title={meta.desc}>
      {compact ? meta.label : `${meta.label} / ${meta.ja}`}
    </span>
  );
}

export function OriginBadge({ origin }) {
  if (origin === 'manual') {
    return <span className="chip" title="この銘柄の指標は手入力値です (J-Quants 由来ではありません)">手入力</span>;
  }
  return <span className="chip" title="J-Quants API から取得した実データ">J-Quants</span>;
}

/** 8軸のうち何軸が実データで埋まっているかを示す */
export function CoverageBadge({ coverage }) {
  const tone = coverage === 8 ? 'green' : coverage >= 5 ? 'amber' : 'red';
  return (
    <span
      className={`badge ${tone}`}
      title={`8軸のうち ${coverage}軸に値があります。欠測軸は総合スコアの平均から除外しています。`}
    >
      {coverage}/8軸
    </span>
  );
}
