import React from 'react';
import RadarPanel from '../components/RadarPanel.jsx';
import AxisTable from '../components/AxisTable.jsx';
import StockCard from '../components/StockCard.jsx';
import { InstitutionalBadge, ZoneBadge } from '../components/Badges.jsx';
import { INSTITUTIONAL_META, ZONE_META } from '../lib/scoring.js';
import { fmt, fmtOku, fmtInt, fmtDate, scoreColor, DASH } from '../lib/format.js';

/** 5.1 マルチ銘柄8軸オクタゴン比較 View */
export default function CompareView({
  rows, visibleIds, onToggle, selectedId, onSelect, onAdd, onDeleteManual,
}) {
  const visible = rows.filter((r) => visibleIds.has(r.stock.id));
  const series = visible.map((r) => ({
    name: `${r.stock.code} ${r.stock.name}`,
    color: r.color,
    scores: r.result.scores,
  }));
  const selected = rows.find((r) => r.stock.id === selectedId) || visible[0] || rows[0];

  return (
    <div className="stack" style={{ gap: 'var(--s4)' }}>
      <div className="split split-fill">
        {/* --- レーダー --- */}
        <div className="card card-fill">
          <div className="card-head">
            <h2>8軸オクタゴン比較</h2>
            <span className="sub">
              {visible.length} / {rows.length} 銘柄を重ね合わせ表示中
            </span>
          </div>
          <div className="chart-slot">
            {series.length === 0 ? (
              <div className="empty">
                右のリストで銘柄のチェックを入れると、ここに8軸オクタゴンが重ねて描画されます。
              </div>
            ) : (
              <RadarPanel series={series} height="100%" />
            )}
          </div>
        </div>

        {/* --- 銘柄コントロール --- */}
        <div className="card card-fill">
          <div className="card-head">
            <h2>銘柄コントロール</h2>
            <button className="btn btn-primary" onClick={onAdd}>+ 銘柄を追加</button>
          </div>
          <div className="stock-list">
            {rows.map((r) => (
              <StockCard
                key={r.stock.id}
                stock={r.stock}
                result={r.result}
                color={r.color}
                visible={visibleIds.has(r.stock.id)}
                selected={selectedId === r.stock.id}
                onToggle={() => onToggle(r.stock.id)}
                onSelect={() => onSelect(r.stock.id)}
                onDelete={r.stock.origin === 'manual' ? () => onDeleteManual(r.stock.id) : null}
              />
            ))}
          </div>
        </div>
      </div>

      {/* --- 選択銘柄の詳細 --- */}
      {selected && <DetailPanel row={selected} />}

      {/* --- ランキング表 --- */}
      <RankingTable rows={rows} onSelect={onSelect} selectedId={selectedId} />
    </div>
  );
}

function DetailPanel({ row }) {
  const { stock, result } = row;
  const m = stock.metrics || {};
  const inst = INSTITUTIONAL_META[result.institutional];
  const zone = ZONE_META[result.zone];

  return (
    <div className="grid-2">
      <div className="card">
        <div className="card-head">
          <h2>{stock.code} {stock.name} — 軸別内訳</h2>
          <span className="sub">{stock.asOf ? `${fmtDate(stock.asOf)} 時点` : '手入力データ'}</span>
        </div>
        <AxisTable axisScores={result.axisScores} />
      </div>

      <div className="stack" style={{ gap: 'var(--s4)' }}>
        <div className="card">
          <div className="card-head"><h2>判定サマリー</h2></div>
          <div className="stack">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-dim)', fontSize: 13 }}>総合スコア</span>
              <span className="score-big" style={{ color: scoreColor(result.totalScore) }}>
                {result.totalScore === null ? DASH : result.totalScore.toFixed(1)}<small> /10</small>
              </span>
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
              有効な {result.coverage} 軸の平均。8軸すべてを 0 埋めして平均した厳密値は{' '}
              <span className="num">{result.strictTotalScore === null ? DASH : result.strictTotalScore.toFixed(1)}</span>。
            </div>

            <div style={{ borderTop: '1px solid var(--border-soft)', paddingTop: 'var(--s3)' }} className="stack">
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dim)', fontSize: 13 }}>株価ゾーン</span>
                <ZoneBadge zone={result.zone} />
              </div>
              {zone && <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>{zone.desc}</div>}

              <div className="row" style={{ justifyContent: 'space-between', marginTop: 6 }}>
                <span style={{ color: 'var(--text-dim)', fontSize: 13 }}>機関投資家参入度</span>
                <InstitutionalBadge level={result.institutional} />
              </div>
              {inst && <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>{inst.desc}</div>}
              {result.institutional === 'cap_low' && result.liquidity && (
                <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
                  （流動性だけで見れば「{INSTITUTIONAL_META[result.liquidity].label}」水準）
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-head"><h2>基本指標</h2></div>
          <dl style={{ margin: 0, display: 'grid', gap: 6 }}>
            <Row label="直近株価" value={m.price == null ? DASH : `${fmtInt(m.price)}円`} />
            <Row label="52週高値" value={m.high52w == null ? DASH : `${fmtInt(m.high52w)}円`} />
            <Row label="52週高値接近率" value={fmt(m.highRatio, 1, '%')} />
            <Row label="売買代金 (終値×出来高)" value={fmtOku(m.tradingValue)} />
            <Row label="売買代金 (API実績値)" value={fmtOku(m.turnoverValue)} />
            <Row label="時価総額" value={fmtOku(m.marketCap)} />
            <Row label="出来高" value={m.volume == null ? DASH : `${fmtInt(m.volume)}株`} />
            <Row label="20日平均出来高" value={m.ma20Volume == null ? DASH : `${fmtInt(m.ma20Volume)}株`} />
            <Row label="信用倍率" value={m.creditRatio == null ? DASH : `${fmt(m.creditRatio, 2)}倍`} />
            <Row label="直近決算" value={m.fiscalPeriod ? `${m.fiscalPeriod} (${fmtDate(m.disclosedDate)} 開示)` : DASH} />
            <Row label="ROEの算出基準" value={m.roeBasis || DASH} />
            <Row label="営業利益率の算出基準" value={m.opMarginBasis || DASH} />
          </dl>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="metric-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function RankingTable({ rows, onSelect, selectedId }) {
  const sorted = [...rows].sort((a, b) => (b.result.totalScore ?? -1) - (a.result.totalScore ?? -1));
  return (
    <div className="card">
      <div className="card-head">
        <h2>総合スコア ランキング</h2>
        <span className="sub">行をクリックすると上の詳細に反映されます</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="axis-table" style={{ minWidth: 780 }}>
          <thead>
            <tr>
              <th>#</th><th>銘柄</th><th className="val">総合</th>
              <th className="val">EPS</th><th className="val">売上</th><th className="val">ROE</th>
              <th className="val">利益率</th><th className="val">テクニカル</th><th className="val">出来高</th>
              <th className="val">需給</th><th className="val">進捗</th>
              <th>ゾーン</th><th>機関</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr
                key={r.stock.id}
                onClick={() => onSelect(r.stock.id)}
                style={{ cursor: 'pointer', background: selectedId === r.stock.id ? 'var(--bg-raised)' : undefined }}
              >
                <td className="num" style={{ color: 'var(--text-faint)' }}>{i + 1}</td>
                <td>
                  <span style={{ color: r.color }}>●</span>{' '}
                  <span className="num" style={{ fontSize: 11 }}>{r.stock.code}</span>{' '}
                  {r.stock.name}
                </td>
                <td className="val" style={{ color: scoreColor(r.result.totalScore), fontWeight: 700 }}>
                  {r.result.totalScore === null ? DASH : r.result.totalScore.toFixed(1)}
                </td>
                {['eps', 'sales', 'roe', 'margin', 'technical', 'volume', 'supply', 'progress'].map((k) => (
                  <td className="val" key={k}>
                    {Number.isFinite(r.result.scores[k])
                      ? r.result.scores[k].toFixed(1)
                      : <span className="na">{DASH}</span>}
                  </td>
                ))}
                <td><ZoneBadge zone={r.result.zone} compact /></td>
                <td><InstitutionalBadge level={r.result.institutional} compact /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
