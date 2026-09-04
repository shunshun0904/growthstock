import React from 'react';
import { InstitutionalBadge, ZoneBadge, OriginBadge, CoverageBadge } from './Badges.jsx';
import { fmt, fmtOku, fmtInt, scoreColor, DASH } from '../lib/format.js';

/** 銘柄コントロールカード (仕様書 §5.1) */
export default function StockCard({ stock, result, color, visible, onToggle, onSelect, selected, onDelete }) {
  const m = stock.metrics || {};
  return (
    <div
      className={`stock-card ${visible ? 'on' : 'dim'}`}
      style={{ borderLeftColor: visible ? color : 'transparent',
               boxShadow: selected ? `inset 0 0 0 1px ${color}` : 'none' }}
    >
      <div className="stock-card-top">
        <input
          type="checkbox"
          className="toggle"
          checked={visible}
          onChange={onToggle}
          style={{ color }}
          aria-label={`${stock.name} をレーダーチャートに表示`}
        />
        <button
          className="btn-ghost"
          onClick={onSelect}
          style={{ flex: 1, textAlign: 'left', padding: 0, background: 'none' }}
        >
          <div className="stock-name">{stock.name}</div>
          <div className="stock-code">
            {stock.code}
            {stock.sector ? ` · ${stock.sector}` : ''}
          </div>
        </button>
        <div style={{ textAlign: 'right' }}>
          <div className="score-big" style={{ color: scoreColor(result.totalScore) }}>
            {result.totalScore === null ? DASH : result.totalScore.toFixed(1)}
            <small> /10</small>
          </div>
        </div>
      </div>

      <div className="row" style={{ marginTop: 8, gap: 5 }}>
        <ZoneBadge zone={result.zone} compact />
        <InstitutionalBadge level={result.institutional} compact />
        <CoverageBadge coverage={result.coverage} />
        <OriginBadge origin={stock.origin} />
      </div>

      <dl style={{ margin: '10px 0 0', display: 'grid', gap: 3 }}>
        <div className="metric-row"><dt>株価</dt><dd>{m.price === null || m.price === undefined ? DASH : `${fmtInt(m.price)}円`}</dd></div>
        <div className="metric-row"><dt>売買代金</dt><dd>{fmtOku(m.tradingValue)}</dd></div>
        <div className="metric-row"><dt>時価総額</dt><dd>{fmtOku(m.marketCap)}</dd></div>
        <div className="metric-row"><dt>52週高値接近率</dt><dd>{fmt(m.highRatio, 1, '%')}</dd></div>
      </dl>

      {stock.note && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--text-dim)',
                      borderTop: '1px solid var(--border-soft)', paddingTop: 6 }}>
          {stock.note}
        </div>
      )}

      {stock.error && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--red)' }}>{stock.error}</div>
      )}

      {onDelete && (
        <button className="btn-ghost" onClick={onDelete} style={{ marginTop: 6, fontSize: 11 }}>
          削除
        </button>
      )}
    </div>
  );
}
