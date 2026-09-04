import React, { useMemo, useState, useEffect, useRef } from 'react';
import RadarPanel from '../components/RadarPanel.jsx';
import AxisTable from '../components/AxisTable.jsx';
import { InstitutionalBadge, ZoneBadge } from '../components/Badges.jsx';
import { computeScores, INSTITUTIONAL_META } from '../lib/scoring.js';
import { fmt, fmtOku, scoreColor, DASH } from '../lib/format.js';

/** スライダー定義 (仕様書 §5.3) */
const SLIDERS = [
  { key: 'epsGrowth',    label: '四半期EPS成長率',  min: -50, max: 150, step: 1,    unit: '%',   fallback: 0 },
  { key: 'salesGrowth',  label: '四半期売上成長率',  min: -30, max: 100, step: 1,    unit: '%',   fallback: 0 },
  { key: 'roe',          label: 'ROE',              min: -10, max: 50,  step: 0.5,  unit: '%',   fallback: 5 },
  { key: 'opMargin',     label: '営業利益率',        min: -20, max: 50,  step: 0.5,  unit: '%',   fallback: 0 },
  { key: 'highRatio',    label: '52週高値接近率',    min: 30,  max: 105, step: 0.5,  unit: '%',   fallback: 80 },
  { key: 'volumeTrend',  label: '出来高モメンタム',  min: 0,   max: 400, step: 5,    unit: '%',   fallback: 100 },
  { key: 'creditRatio',  label: '信用倍率',          min: 0.1, max: 20,  step: 0.1,  unit: '倍',  fallback: 3 },
  { key: 'progressRate', label: '決算進捗率',        min: 0,   max: 150, step: 1,    unit: '%',   fallback: 50 },
  { key: 'tradingValue', label: '売買代金',          min: 0,   max: 100, step: 0.5,  unit: '億円', fallback: 5 },
  { key: 'marketCap',    label: '時価総額',          min: 10,  max: 5000, step: 10,  unit: '億円', fallback: 500 },
];

const QUARTERS = [1, 2, 3, 4];

/** 5.3 What-If 感度シミュレーター View */
export default function SimulatorView({ rows, selectedId, onSelect }) {
  const row = rows.find((r) => r.stock.id === selectedId) || rows[0];
  const [params, setParams] = useState({});
  const [frameMs, setFrameMs] = useState(null);
  const pendingRef = useRef(false);

  // 銘柄が切り替わったら現在値でスライダーを初期化する
  useEffect(() => {
    if (!row) return;
    const m = row.stock.metrics || {};
    const init = {};
    for (const s of SLIDERS) {
      const v = m[s.key];
      init[s.key] = Number.isFinite(v) ? clamp(v, s.min, s.max) : s.fallback;
    }
    init.quarter = Number.isFinite(m.quarter) ? m.quarter : 2;
    setParams(init);
  }, [row?.stock.id]);   // eslint-disable-line react-hooks/exhaustive-deps

  const baseline = row?.result ?? null;
  const simulated = useMemo(
    () => (Object.keys(params).length ? computeScores(params) : null),
    [params]
  );

  // 仕様書 §6.1: 再計算 + 再描画が 16ms 以内 (60fps) に収まっているか実測する
  useEffect(() => {
    if (!pendingRef.current) return;
    pendingRef.current = false;
    const start = performance.now();
    const id = requestAnimationFrame(() => setFrameMs(performance.now() - start));
    return () => cancelAnimationFrame(id);
  }, [params]);

  if (!row) return <div className="empty">銘柄がありません。</div>;

  const update = (key) => (e) => {
    pendingRef.current = true;
    setParams((p) => ({ ...p, [key]: Number(e.target.value) }));
  };

  const reset = () => {
    const m = row.stock.metrics || {};
    const init = {};
    for (const s of SLIDERS) {
      const v = m[s.key];
      init[s.key] = Number.isFinite(v) ? clamp(v, s.min, s.max) : s.fallback;
    }
    init.quarter = Number.isFinite(m.quarter) ? m.quarter : 2;
    setParams(init);
  };

  const series = [
    ...(baseline ? [{ name: '現状', color: '#5c6980', scores: baseline.scores, dashed: true }] : []),
    ...(simulated ? [{ name: 'シミュレーション', color: '#4f8dff', scores: simulated.scores }] : []),
  ];

  const delta =
    simulated?.totalScore != null && baseline?.totalScore != null
      ? simulated.totalScore - baseline.totalScore
      : null;

  return (
    <div className="stack" style={{ gap: 'var(--s4)' }}>
      <div className="split split-fill">
        <div className="card card-fill">
          <div className="card-head">
            <h2>What-If 感度シミュレーター</h2>
            <div className="row">
              <select
                value={row.stock.id}
                onChange={(e) => onSelect(e.target.value)}
                aria-label="対象銘柄"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)',
                         borderRadius: 'var(--radius-sm)', padding: '5px 9px', fontSize: 13 }}
              >
                {rows.map((r) => (
                  <option key={r.stock.id} value={r.stock.id}>{r.stock.code} {r.stock.name}</option>
                ))}
              </select>
              <button className="btn" onClick={reset}>現在値に戻す</button>
            </div>
          </div>

          <div className="chart-slot">
            {series.length > 0 && <RadarPanel series={series} height="100%" />}
          </div>

          <div className="row" style={{ justifyContent: 'space-between', marginTop: 'var(--s3)',
                                        borderTop: '1px solid var(--border-soft)', paddingTop: 'var(--s3)' }}>
            <div>
              <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>シミュレーション総合スコア</div>
              <div className="score-big" style={{ color: scoreColor(simulated?.totalScore) }}>
                {simulated?.totalScore == null ? DASH : simulated.totalScore.toFixed(1)}
                <small> /10</small>
                {delta !== null && (
                  <span style={{ fontSize: 13, marginLeft: 10,
                                 color: delta > 0.05 ? 'var(--green)' : delta < -0.05 ? 'var(--red)' : 'var(--text-faint)' }}>
                    {delta > 0 ? '+' : ''}{delta.toFixed(1)} vs 現状
                  </span>
                )}
              </div>
            </div>
            <div className="row" style={{ gap: 5 }}>
              <ZoneBadge zone={simulated?.zone} compact />
              <InstitutionalBadge level={simulated?.institutional} compact />
            </div>
          </div>

          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
            再計算＋再描画: {frameMs === null ? '—' : `${frameMs.toFixed(1)}ms`}
            {frameMs !== null && (
              <span style={{ color: frameMs <= 16 ? 'var(--green)' : 'var(--amber)' }}>
                {' '}({frameMs <= 16 ? '60fps 基準内' : '60fps 基準超過'} / 仕様書 §6.1)
              </span>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h2>パラメータ</h2>
            <span className="sub">操作すると即時に再計算</span>
          </div>

          <div className="field" style={{ marginBottom: 'var(--s4)' }}>
            <label htmlFor="sim-quarter">経過四半期（進捗基準 = Quarter × 25%）</label>
            <div className="row" style={{ gap: 4 }}>
              {QUARTERS.map((q) => (
                <button
                  key={q}
                  className="btn"
                  onClick={() => { pendingRef.current = true; setParams((p) => ({ ...p, quarter: q })); }}
                  style={params.quarter === q
                    ? { background: 'var(--accent-dim)', borderColor: 'var(--accent)' }
                    : undefined}
                >
                  {q}Q
                </button>
              ))}
              <span className="chip" style={{ marginLeft: 4 }}>基準 {(params.quarter ?? 0) * 25}%</span>
            </div>
          </div>

          {SLIDERS.map((s) => (
            <div className="slider-row" key={s.key}>
              <label htmlFor={`sim-${s.key}`}>{s.label}</label>
              <output htmlFor={`sim-${s.key}`}>
                {s.unit === '億円'
                  ? fmtOku(params[s.key])
                  : `${fmt(params[s.key], s.step < 1 ? 1 : 0)}${s.unit}`}
              </output>
              <input
                id={`sim-${s.key}`}
                type="range"
                min={s.min} max={s.max} step={s.step}
                value={params[s.key] ?? s.fallback}
                onChange={update(s.key)}
              />
            </div>
          ))}

          <div className="banner info" style={{ marginTop: 'var(--s3)' }}>
            売買代金と時価総額は機関投資家参入度（
            {INSTITUTIONAL_META[simulated?.institutional]?.label ?? '—'}
            ）を通じて「出来高」軸の減衰率にも効きます。
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <h2>軸別 — 現状との差分</h2>
          </div>
          {simulated && baseline
            ? <AxisTable axisScores={simulated.axisScores} compareTo={baseline.scores} />
            : <div className="empty">比較対象がありません。</div>}
        </div>

        <div className="card">
          <div className="card-head">
            <h2>感度分析</h2>
            <span className="sub">各パラメータを ±10% 動かしたときの総合スコア変化</span>
          </div>
          <Sensitivity params={params} />
        </div>
      </div>
    </div>
  );
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

/** 各パラメータの局所感度 (総合スコアの偏微分の有限差分近似) */
function Sensitivity({ params }) {
  const rowsData = useMemo(() => {
    if (!Object.keys(params).length) return [];
    const base = computeScores(params).totalScore;
    if (base == null) return [];
    return SLIDERS.map((s) => {
      const cur = params[s.key];
      if (!Number.isFinite(cur)) return null;
      const step = Math.max(Math.abs(cur) * 0.1, s.step);
      const up = computeScores({ ...params, [s.key]: cur + step }).totalScore;
      const down = computeScores({ ...params, [s.key]: cur - step }).totalScore;
      return {
        key: s.key, label: s.label, unit: s.unit,
        up: up == null ? 0 : up - base,
        down: down == null ? 0 : down - base,
        span: Math.abs((up ?? base) - (down ?? base)),
        step,
      };
    }).filter(Boolean).sort((a, b) => b.span - a.span);
  }, [params]);

  if (!rowsData.length) return <div className="empty">計算できるパラメータがありません。</div>;
  const maxSpan = Math.max(...rowsData.map((r) => r.span), 0.01);

  return (
    <table className="axis-table">
      <thead>
        <tr>
          <th>パラメータ</th>
          <th className="val">−10%</th>
          <th className="val">+10%</th>
          <th style={{ width: '38%' }}>影響度</th>
        </tr>
      </thead>
      <tbody>
        {rowsData.map((r) => (
          <tr key={r.key}>
            <td>{r.label}</td>
            <td className="val" style={{ color: r.down < -0.01 ? 'var(--red)' : 'var(--text-faint)' }}>
              {r.down.toFixed(2)}
            </td>
            <td className="val" style={{ color: r.up > 0.01 ? 'var(--green)' : 'var(--text-faint)' }}>
              {r.up > 0 ? '+' : ''}{r.up.toFixed(2)}
            </td>
            <td>
              <div className="bar">
                <i style={{ width: `${(r.span / maxSpan) * 100}%`, background: 'var(--accent)' }} />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
