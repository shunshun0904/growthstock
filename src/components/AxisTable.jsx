import React from 'react';
import { fmt, DASH } from '../lib/format.js';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function barColor(score) {
  if (score >= 8) return 'var(--violet)';
  if (score >= 6.5) return 'var(--green)';
  if (score >= 5) return 'var(--amber)';
  return 'var(--red)';
}

/** 8軸の「指標の生値 → スコア」対応表。欠測は明示的に「データなし」と出す。 */
export default function AxisTable({ axisScores, compareTo = null }) {
  return (
    <table className="axis-table">
      <thead>
        <tr>
          <th style={{ width: '28%' }}>軸</th>
          <th className="val" style={{ width: '22%' }}>指標値</th>
          <th style={{ width: '30%' }}>スコア</th>
          <th className="val" style={{ width: '20%' }}>{compareTo ? '差分' : '0 – 10'}</th>
        </tr>
      </thead>
      <tbody>
        {axisScores.map((a) => {
          const has = isNum(a.score);
          const delta = compareTo && has && isNum(compareTo[a.key]) ? a.score - compareTo[a.key] : null;
          return (
            <tr key={a.key}>
              <td title={`${a.full} — ${a.rule}`}>
                {a.label}
                <div style={{ fontSize: 10.5, color: 'var(--text-faint)' }}>{a.full}</div>
              </td>
              <td className="val">
                {isNum(a.value)
                  ? `${fmt(a.value, a.unit === '倍' ? 2 : 1)}${a.unit}`
                  : <span className="na" title="このデータは取得できていません">{DASH}</span>}
              </td>
              <td>
                <div className="row" style={{ gap: 8, flexWrap: 'nowrap' }}>
                  <div className="bar" style={{ flex: 1 }}>
                    {has && <i style={{ width: `${a.score * 10}%`, background: barColor(a.score) }} />}
                  </div>
                  <span className="num" style={{ fontSize: 12, minWidth: 30, textAlign: 'right' }}>
                    {has ? a.score.toFixed(1) : <span className="na">{DASH}</span>}
                  </span>
                </div>
              </td>
              <td className="val" style={{ fontSize: 11.5 }}>
                {compareTo
                  ? (delta === null
                      ? <span className="na">{DASH}</span>
                      : <span style={{ color: delta > 0.05 ? 'var(--green)' : delta < -0.05 ? 'var(--red)' : 'var(--text-faint)' }}>
                          {delta > 0 ? '+' : ''}{delta.toFixed(1)}
                        </span>)
                  : <span style={{ color: 'var(--text-faint)' }}>{a.rule}</span>}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
