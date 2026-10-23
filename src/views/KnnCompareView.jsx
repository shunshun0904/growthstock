import React, { useMemo, useState } from 'react';
import { fmt, fmtDate, fmtDateTime, fmtInt } from '../lib/format.js';
import { STRATEGY, RULE_JA } from '../lib/strategy.js';
import { dayOrigin } from '../lib/predictions.js';
import {
  KNN_MODES, knnDay, knnRows, knnStatus, lin, niceTicks, scatterFrame, cloudLayout,
  scatterTable, stripTable, cloudTable,
} from '../lib/knn.js';

/**
 * 候補の比較（似た過去の事例）。運用者の指示（2026-10-15）「別タブとして、左から2番目の位置に追加して下さい。3枚で良いです」
 * （実験97 の模型の3枚。docs/MODEL_ADOPTION_RULES.md §57・§58）。
 *
 * データは predictions.json の knn（research/knn_compare.py）。候補ごとに、60日のリターンの形と出来高の比率が似た過去の事例
 * （規則の3本（2026-10-23 から 1.2σ の3本）の最小が85以上で結果が出そろった行から、ワッサーステイン距離の近い50件）がどうなったかを3枚で見せる。
 *   1. 候補の位置（散布図）: 横 = 近傍の収益の下側25%点、縦 = 近傍の −10% 割れの割合。背景は過去の候補（最小95以上）の分布
 *   2. 似た過去の事例の収益の幅: 候補ごとの近傍の 10〜90% 点・25〜75% 点・中央値
 *   3. 近傍50件の「距離 × 収益」: 候補ごとに、近い順の50件がどの距離でどの収益の帯だったかの度数
 * **参考の表示で、選定には使わない。** 買う銘柄は今日の戦略（規則の3本の最小の順）のまま。近傍の25%点で選ぶ後工程（Q）は
 * 同じ期間のデータでは判定を通ったが裏付けが弱く、前向きの確かめ直し（実験94。判定は 2027-10-15 以降）が通るまで使わない。
 *
 * 色は候補の識別だけに使い、印（A・B・C…）と表を必ず添える（色だけに頼らない）。色は dataviz の暗い面向けの段で、
 * 隣り合う色の色覚の差を確かめた並び（6色まで。7件目からは灰色）。数字の文字は本文の色（系列の色で文字を塗らない）。
 */

/** 候補の色（暗い面向け。隣り合う組の色覚の差を確かめた並び）。7件目からは灰色で、印の文字で見分ける */
export const KNN_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9'];
const GRAY = '#8a93a6';
export const colorOf = (i) => KNN_COLORS[i] ?? GRAY;
const QM = (r) => `${r.pickedQ ? '（Q）' : ''}${r.pickedM ? '（M）' : ''}`;

export default function KnnCompareView({ data }) {
  const dates = data?.dates || [];
  const [day, setDay] = useState(() => data?.asOf || dates[dates.length - 1]);
  const [mode, setMode] = useState('pass');
  const [tip, setTip] = useState(null);
  const block = knnDay(data, day);
  const rows = useMemo(
    () => knnRows((data?.candidates || []).filter((c) => c.date === day), block, mode),
    [data, day, block, mode]
  );
  const status = knnStatus(data, block, rows, mode);
  const knn = data?.knn || {};
  const origin = useMemo(() => dayOrigin(data, day), [data, day]);
  const drawn = rows.filter((r) => r.knn);
  const q = rows.find((r) => r.pickedQ);
  const m = rows[0];
  const tips = {
    show: (e, title, lines) => setTip({ x: e.clientX, y: e.clientY, title, lines }),
    hide: () => setTip(null),
  };

  return (
    <div className="knn">
      <section className="card">
        <div className="card-head">
          <h2>候補の比較（似た過去の事例）</h2>
          <span className="sub">
            {'同じ日の候補ごとに、60日のリターンの形と出来高の比率が似た過去の事例（近い50件）がどうなったかを並べます。'
              + '参考の表示で、買う銘柄の選び方は今日の戦略のまま'}
          </span>
        </div>

        <div className="pred-strip">
          <div>
            <span className="lab">対象日</span>
            <div className="pred-days">
              {dates.map((d) => (
                <button key={d} className={`btn btn-ghost${d === day ? ' on' : ''}`} onClick={() => setDay(d)}>
                  {fmtDate(d)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <span className="lab">表示</span>
            <div className="knn-modes" role="group" aria-label="表示する候補の範囲">
              {KNN_MODES.map((o) => (
                <button key={o.id} className={`btn btn-ghost${o.id === mode ? ' on' : ''}`}
                        aria-pressed={o.id === mode} onClick={() => setMode(o.id)}>
                  {o.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <span className="lab">候補</span>
            <strong className="num">{rows.length}</strong> 件
            {rows.length > 0 && <span className="sub">（近傍の集計あり {drawn.length}）</span>}
          </div>
          <div title={`似た事例を探す範囲。${RULE_JA}の最小が85以上で、この日より前に結果（+10% の指値か20営業日目の終値）が出そろった過去の行`}>
            <span className="lab">母集団</span>
            <span className="num">{block?.poolN != null ? fmtInt(block.poolN) : '—'}</span> 件
          </div>
          <div>
            <span className="lab">Q / M</span>
            <span className="num">{q ? `${q.label} ${q.code}` : '—'}</span>
            <span className="sep">/</span>
            <span className="num">{m ? `${m.label} ${m.code}` : '—'}</span>
          </div>
        </div>

        {origin.past && (
          <div className="banner info pred-note">
            <span>ℹ</span>
            <div>{`この日の値は、${origin.generatedAt ? fmtDateTime(origin.generatedAt) : 'その日'}に出したまま固定しています（近傍の集計も同じ）`}</div>
          </div>
        )}

        <div className="banner info pred-note knn-note">
          <span>ℹ</span>
          <div>
            <b>参考の表示です。選定には使いません。</b>
            {` 買う銘柄は今日の戦略（${RULE_JA}のうち一番低い百分位が高い順。M はその先頭）のまま。`}
            Q は近傍50件の収益の下側25%点が最も高い候補（実験93 の選び方）で、同じ期間のデータでは先に決めた判定を通りましたが
            裏付けは弱く（§53・§55）、前向きの確かめ直し（実験94。判定は 2027-10-15 以降）が通るまで選定には使いません。
            <div style={{ marginTop: 4 }}>
              {`似た事例の探し方: ブレイクの日を含む60営業日の、日次リターンを平均0・標準偏差1にそろえた分布と、log(出来高 ÷ 前60営業日の平均) の`
                + `分布を、それぞれ60点の分位関数で持ち、分位関数の差の絶対値の平均（1次元のワッサーステイン距離）を尺度（母集団から無作為に`
                + `取った2件の距離の中央値）で割って足した距離の近い${knn.k ?? 50}件。収益は翌営業日の寄りで買い、+${knn.target ?? 10}% の指値に`
                + `届けば +${knn.target ?? 10}、届かなければ${knn.hold ?? 20}営業日目の終値。候補の色は識別のためだけで、印（A・B・C…）は`
                + `今の並べ方の順。7件目からは灰色（印で見分ける）。過去の日は、その日に出した値のまま。`}
            </div>
          </div>
        </div>

        {rows.length > 0 && (
          <div className="knn-legend" aria-label="候補の印と色">
            {rows.map((r, i) => (
              <span key={r.jqCode} title={r.name || ''}>
                <i className="sw" style={{ background: colorOf(i) }} aria-hidden="true" />
                <span className="lab">{r.label}</span>
                <span className="num">{r.code}</span>
                <span className="name">{r.name || ''}</span>
                <span className="sub">
                  {`最小 ${r.ruleMin === null ? '—' : fmt(r.ruleMin, 1)}${QM(r)}${r.knn ? '' : '・集計なし'}`}
                </span>
              </span>
            ))}
          </div>
        )}
      </section>

      {status.kind !== 'ok' ? (
        <div className="card empty knn-empty">{status.text}</div>
      ) : (
        <div className="knn-grid">
          <Figure title="候補の位置（散布図）"
                  sub={'横軸は近傍50件の収益の下側25%点（右ほど、似た事例の下側が良い）、縦軸は近傍の −10% 割れの割合（下ほど良い）。'
                    + '背景の格子は過去の候補（最小95以上）の分布（件数。濃いほど多い）で、今日の候補が過去のどの辺りにいるかを示す。'
                    + 'Q は二重の輪、M（今の並べ方の先頭）は四角'}
                  headers={['候補', 'コード', '銘柄', '3本の最小', '近傍の25%点', '近傍の中央値', '+10% 到達', '−10% 割れ', '近傍までの距離（形 / 出来高）']}
                  table={scatterTable(rows)}>
            <Scatter rows={rows} context={knn.context} tips={tips} />
          </Figure>
          <Figure title="似た過去の事例の収益の幅"
                  sub={'候補ごとに、近傍50件の収益の 10〜90% 点（薄い帯）と 25〜75% 点（濃い帯）、中央値（明るい線）。並びは今の並べ方（上が先頭）。'
                    + 'Q の行は左に印。右の数字は近傍の +10% 到達と −10% 割れの割合。+10 は指値に届いた事例'}
                  headers={['候補', 'コード', '10%点', '25%点', '中央値', '75%点', '90%点', '+10% 到達', '−10% 割れ']}
                  table={stripTable(rows)}>
            <Strips rows={rows} tips={tips} />
          </Figure>
          <Figure title={`近傍${knn.k ?? 50}件の「距離 × 収益」`}
                  sub={'候補ごとに、近い順に取った50件が、どれくらいの距離（横。1.0 = 母集団から無作為に取った2件の距離の中央値）で、'
                    + 'どの収益の帯（縦）だったかの件数。左の列ほど似ている事例。濃いほど多い'}
                  headers={['候補・帯', ...(knn.dBins || []).slice(0, -1).map((d, i) => `${d}〜${knn.dBins[i + 1]}`)]}
                  table={cloudTable(rows, knn.bands || [])}>
            <Clouds rows={rows} bands={knn.bands || []} dBins={knn.dBins || []} tips={tips} />
          </Figure>
        </div>
      )}

      {tip && (
        <div className="knn-tip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
          <div className="t">{tip.title}</div>
          {tip.lines.map(([k, v]) => (
            <div key={k}><span>{k}</span><b className="num">{v}</b></div>
          ))}
        </div>
      )}
    </div>
  );
}

/** 図の枠。見出し・説明・図・同じ数字の表（色に頼らず読めるように） */
function Figure({ title, sub, headers, table, children }) {
  return (
    <figure className="knn-fig">
      <figcaption>
        <h3>{title}</h3>
        <p className="sub">{sub}</p>
      </figcaption>
      {children}
      <details className="knn-details">
        <summary>表で見る</summary>
        <div className="knn-tbl">
          <table>
            <thead><tr>{headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>
              {table.map((r, i) => (
                <tr key={i}>{r.map((c, j) => <td key={j}>{String(c)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

const tipLines = (r) => [
  ['3本の最小', r.ruleMin === null ? '—' : fmt(r.ruleMin, 1)],
  ['近傍の25%点 / 中央値', `${fmt(r.knn.est.p25, 2)} / ${fmt(r.knn.est.p50, 2)}`],
  ['近傍の +10% 到達 / −10% 割れ', `${r.knn.est.hit}% / ${r.knn.est.bad}%`],
  ['近傍までの距離（形 / 出来高）', `${fmt(r.knn.dNn.R, 2)} / ${fmt(r.knn.dNn.V, 2)}`],
];

/** 1. 候補の位置（散布図） */
function Scatter({ rows, context, tips }) {
  const W = 560; const H = 320; const mg = { l: 48, r: 16, t: 18, b: 48 };
  const { lo, hi, ytop } = scatterFrame(rows, context);
  const x = lin(lo, hi, mg.l, W - mg.r);
  const y = lin(0, ytop, H - mg.b, mg.t);
  const D = context?.density;
  const xe = D?.xEdges || []; const ye = D?.yEdges || [];
  const maxc = Math.max(1, ...(D?.counts?.flat?.() || [0]));
  const dec = context?.p25Deciles;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="knn-svg" role="img" aria-label="候補の位置（近傍の25%点 × −10% 割れ）">
      {D && xe.slice(0, -1).map((_, i) => ye.slice(0, -1).map((__, j) => {
        const c = D.counts[i]?.[j] || 0;
        if (!c) return null;
        const X0 = x(Math.max(lo, xe[i])); const X1 = x(Math.min(hi, xe[i + 1]));
        const Y1 = y(Math.min(ytop, ye[j])); const Y0 = y(Math.min(ytop, ye[j + 1]));
        return (
          <rect key={`${i}-${j}`} className="knn-bg" x={X0} y={Y0} width={Math.max(0, X1 - X0)} height={Math.max(0, Y1 - Y0)}
                fill="var(--accent)" opacity={0.04 + 0.22 * Math.sqrt(c / maxc)}
                onPointerEnter={(e) => tips.show(e, '背景（過去の候補）', [
                  ['件数', String(c)], ['近傍の25%点', `${fmt(xe[i], 1)}〜${fmt(xe[i + 1], 1)}`],
                  ['近傍の −10% 割れ', `${ye[j]}〜${ye[j + 1] >= 100 ? '' : ye[j + 1]}%`]])}
                onPointerLeave={tips.hide} />
        );
      }))}
      {niceTicks(0, ytop, 4).map((t) => (
        <g key={`y${t}`}>
          <line x1={mg.l} x2={W - mg.r} y1={y(t)} y2={y(t)} className="knn-gl" />
          <text x={mg.l - 6} y={y(t) + 3.5} textAnchor="end" className="knn-lbl">{`${t}%`}</text>
        </g>
      ))}
      {niceTicks(lo, hi, 6).map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={mg.t} y2={H - mg.b} className="knn-gl" />
          <text x={x(t)} y={H - mg.b + 14} textAnchor="middle" className="knn-lbl">{fmt(t, 0)}</text>
        </g>
      ))}
      {dec && [['下位10%', 1, 10], ['中央', 5, 22], ['上位10%', 9, 34]].map(([lab, i, dy]) => {
        const v = dec[i];
        if (!(v >= lo && v <= hi)) return null;
        return (
          <g key={lab}>
            <line x1={x(v)} x2={x(v)} y1={mg.t} y2={H - mg.b} className="knn-ax" strokeDasharray="3 3" />
            <text x={x(v) + 3} y={mg.t + dy} className="knn-lbl">{`過去の${lab} ${fmt(v, 1)}`}</text>
          </g>
        );
      })}
      {rows.map((r, i) => {
        if (!r.knn) return null;
        const cx = x(r.knn.est.p25); const cy = y(r.knn.est.bad);
        const left = cx > W - mg.r - 40;
        return (
          <g key={r.jqCode}>
            {r.pickedQ && <circle className="knn-q" data-label={r.label} cx={cx} cy={cy} r={13} fill="none" stroke="var(--text)" strokeWidth={2} />}
            {r.pickedM && <rect className="knn-m" data-label={r.label} x={cx - 9} y={cy - 9} width={18} height={18} fill="none" stroke="var(--text-dim)" strokeWidth={1.5} />}
            <circle className="knn-dot" cx={cx} cy={cy} r={6} fill={colorOf(i)} stroke="var(--bg-card)" strokeWidth={2} />
            <text className="knn-dl knn-dot-label" x={left ? cx - 11 : cx + 11} y={cy - 8} textAnchor={left ? 'end' : 'start'}>{r.label}</text>
            <circle className="knn-hit" cx={cx} cy={cy} r={16}
                    onPointerEnter={(e) => tips.show(e, `候補 ${r.label} ${r.code}${QM(r)}`, tipLines(r))}
                    onPointerLeave={tips.hide} />
          </g>
        );
      })}
      <text x={(mg.l + W - mg.r) / 2} y={H - 6} textAnchor="middle" className="knn-lbl">近傍50件の収益の下側25%点（%）→ 右ほど良い</text>
      <text x={12} y={mg.t - 6} className="knn-lbl">近傍の −10% 割れ（%）↓ 下ほど良い</text>
    </svg>
  );
}

/** 2. 似た過去の事例の収益の幅 */
function Strips({ rows, tips }) {
  const C = rows.filter((r) => r.knn);
  const W = 560; const rowH = 44; const mg = { l: 118, r: 150, t: 26, b: 30 };
  const H = mg.t + rowH * Math.max(1, C.length) + mg.b;
  const lo = Math.min(-20, ...C.map((r) => r.knn.est.p10)) - 1; const hi = 10.5;
  const x = lin(lo, hi, mg.l, W - mg.r);
  const idx = new Map(rows.map((r, i) => [r.jqCode, i]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="knn-svg" role="img" aria-label="似た過去の事例の収益の幅">
      {[-30, -20, -10, 0, 5, 10].filter((t) => t >= lo).map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={mg.t - 6} y2={H - mg.b + 4} className={t === 0 ? 'knn-ax' : 'knn-gl'} />
          <text x={x(t)} y={H - mg.b + 16} textAnchor="middle" className="knn-lbl">{(t > 0 ? '+' : '') + t}</text>
        </g>
      ))}
      {C.map((r, k) => {
        const i = idx.get(r.jqCode); const e = r.knn.est; const col = colorOf(i);
        const cy = mg.t + rowH * k + rowH / 2;
        const p10 = Math.max(lo, e.p10);
        return (
          <g key={r.jqCode} className="knn-strip-row">
            <rect x={x(p10)} y={cy - 7} width={Math.max(1, x(e.p90) - x(p10))} height={14} fill={col} opacity={0.25} rx={4} />
            <rect x={x(e.p25)} y={cy - 9} width={Math.max(2, x(e.p75) - x(e.p25))} height={18} fill={col} rx={4} />
            <line x1={x(e.p50)} x2={x(e.p50)} y1={cy - 11} y2={cy + 11} stroke="var(--bg-card)" strokeWidth={3} />
            <rect x={mg.l - 112} y={cy - 5} width={10} height={10} rx={5} fill={col} />
            <text x={mg.l - 98} y={cy + 4} className="knn-dl">{`${r.label} ${r.code}`}</text>
            {r.pickedQ && <text x={mg.l - 12} y={cy + 4} textAnchor="end" className="knn-dl knn-qmark">Q ▶</text>}
            <text x={W - mg.r + 10} y={cy + 4} className="knn-lbl">{`到達 ${e.hit}%・−10% ${e.bad}%`}</text>
            <rect className="knn-hit" x={mg.l} y={cy - rowH / 2} width={W - mg.l - mg.r} height={rowH}
                  onPointerEnter={(ev) => tips.show(ev, `候補 ${r.label} ${r.code}${QM(r)}`, [
                    ['10 / 25 / 50 / 75 / 90% 点', `${fmt(e.p10, 1)} / ${fmt(e.p25, 1)} / ${fmt(e.p50, 1)} / ${fmt(e.p75, 1)} / ${fmt(e.p90, 1)}`],
                    ['+10% 到達 / −10% 割れ', `${e.hit}% / ${e.bad}%`]])}
                  onPointerLeave={tips.hide} />
          </g>
        );
      })}
      <text x={(mg.l + W - mg.r) / 2} y={H - 4} textAnchor="middle" className="knn-lbl">近傍50件の収益（%。+10 は指値に届いた事例）</text>
    </svg>
  );
}

/** 3. 近傍50件の「距離 × 収益」（候補ごとの格子） */
function Clouds({ rows, bands, dBins, tips }) {
  const C = rows.filter((r) => r.knn);
  const nb = Math.max(0, dBins.length - 1); const ny = bands.length;
  const L = cloudLayout(C.length, { bins: nb });
  const { cell, gap, left, panelW, perRow } = L;
  const W = 560; const H = L.rows * (ny * cell + 56) + 10;
  const idx = new Map(rows.map((r, i) => [r.jqCode, i]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="knn-svg" role="img" aria-label="近傍50件の距離と収益の帯">
      {C.map((r, k) => {
        const i = idx.get(r.jqCode);
        const rr0 = Math.floor(k / perRow); const cc0 = k % perRow;
        const X0 = left + cc0 * (panelW + gap); const Y0 = 24 + rr0 * (ny * cell + 56);
        const counts = r.knn.cloud;
        const maxc = Math.max(1, ...counts.flat());
        return (
          <g key={r.jqCode} className="knn-cloud">
            <rect x={X0 - 14} y={Y0 - 16} width={8} height={8} rx={4} fill={colorOf(i)} />
            <text x={X0} y={Y0 - 8} className="knn-dl">{`候補 ${r.label} ${r.code}${QM(r)}`}</text>
            {counts.map((colv, a) => colv.map((v, b) => {
              const cxp = X0 + a * cell; const cyp = Y0 + (ny - 1 - b) * cell;
              return (
                <g key={`${a}-${b}`}>
                  <rect className="knn-cloud-cell" x={cxp + 1} y={cyp + 1} width={cell - 2} height={cell - 2} rx={2}
                        fill={v ? 'var(--accent)' : 'var(--bg-input)'} opacity={v ? 0.2 + 0.8 * v / maxc : 1}
                        stroke={v ? 'none' : 'var(--border-soft)'}
                        onPointerEnter={(e) => tips.show(e, `候補 ${r.label} の近傍`, [
                          ['距離', `${dBins[a]}〜${dBins[a + 1]}`], ['収益の帯', bands[b]], ['件数', String(v)]])}
                        onPointerLeave={tips.hide} />
                  {v > 0 && (
                    <text x={cxp + cell / 2} y={cyp + cell / 2 + 3.5} textAnchor="middle" className="knn-cell-num"
                          style={{ pointerEvents: 'none' }}>{v}</text>
                  )}
                </g>
              );
            }))}
            {cc0 === 0 && bands.map((bl, b) => (
              <text key={bl} x={X0 - 4} y={Y0 + (ny - 1 - b) * cell + cell / 2 + 3.5} textAnchor="end" className="knn-lbl">{bl}</text>
            ))}
            {dBins.map((d, a) => (a % 2 === 0 ? (
              <text key={d} x={X0 + a * cell} y={Y0 + ny * cell + 12} textAnchor="middle" className="knn-lbl">{String(d)}</text>
            ) : null))}
          </g>
        );
      })}
    </svg>
  );
}
