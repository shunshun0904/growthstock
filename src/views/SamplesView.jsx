import React from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, ReferenceDot,
} from 'recharts';
import { fmt, fmtSigned, fmtDate, fmtInt } from '../lib/format.js';
import {
  splitCases, caseChart, verdictLines, fromEntry, fmtMD, windowText, addedPositiveRate,
} from '../lib/samples.js';

/**
 * 再学習で加わった例（運用者の依頼 2026-10-13「直近追加されたサンプルのチャート（週次のリトレインで新たに追加された
 * サンプルのチャート）、正例、負例で5例ずつ程度で良いので、表示してほしいです。正例は5例もないかもしれませんが、
 * なければないで大丈夫です」）。
 *
 * データは public/data/new_samples.json（New Training Samples が再学習の後に research/new_samples.py で書く）。
 * チャートの線は学習のラベルと同じ式で引いてある（research/export_label_samples.build_case）。
 *   終値（分割調整後）の線1本だけがデータの系列。ほかは注記なので、色ではなく線の形（破線・点線）と文で見分ける
 */
export default function SamplesView({ data }) {
  if (data === undefined) return <div className="empty">読み込んでいます…</div>;
  if (!data) {
    return (
      <div className="card empty">
        まだありません。週次の再学習（日曜）の後に <code>New Training Samples</code> が作ります。
      </div>
    );
  }
  const { pos, neg } = splitCases(data);
  const c = data.counts || {};
  const m = data.model || {};
  const rate = addedPositiveRate(data);
  return (
    <div className="samples">
      <section className="card">
        <div className="card-head">
          <h2>再学習で加わった例</h2>
          <span className="sub">
            週次の再学習で新しく学習に入ったサンプルから、正例・負例を最大5件ずつ（無作為。週ごとに入れ替わる）
          </span>
        </div>
        <div className="pred-strip">
          <div>
            <span className="lab">再学習</span>
            <span className="num">{fmtDate(m.trainedAt)}</span>
          </div>
          <div>
            <span className="lab">加わった更新日</span>
            <span className="num">{windowText(data)}</span>
          </div>
          <div>
            <span className="lab">加わった件数</span>
            <strong className="num">{fmtInt(c.added)}</strong> 件
            <span className="sub">（正例 {fmtInt(c.positive)}・負例 {fmtInt(c.negative)}
              {rate !== null && `・正例率 ${fmt(rate, 1, '%')}`}）</span>
          </div>
          <div>
            <span className="lab">学習の件数</span>
            <span className="num">{fmtInt(m.nTrain)}</span> 件
            {data.prev?.nTrain ? <span className="sub">（前回 {fmtInt(data.prev.nTrain)}）</span> : null}
          </div>
        </div>
        <p className="sub samples-note">
          {`ラベルの定義: ${m.label || '—'}。更新日（78週高値を抜けた日）の翌営業日の寄りを基準に、20営業日のうちに終値が`
            + '「到達に必要な水準」に届き、20日目までの5日平均が「終盤に必要な水準」以上で、20日目に MA5 ≥ MA20 なら正例。'}
        </p>
        <SamplesLegend />
      </section>

      <CaseSection title="正例" cases={pos} total={c.positive}
                   empty="今回加わった正例はありません。" />
      <CaseSection title="負例" cases={neg} total={c.negative}
                   empty="今回加わった負例はありません。" />
    </div>
  );
}

/** 線の見分け方。色だけに頼らず、線の形と言葉で示す。 */
function SamplesLegend() {
  const items = [
    ['solid', 'var(--accent)', '終値（分割調整後）'],
    ['dash', 'var(--amber)', '到達に必要な水準'],
    ['dot', 'var(--amber)', '終盤に必要な水準（20日目までの5日）'],
    ['fine', 'var(--text-dim)', '基準の価格（翌営業日の寄り）'],
    ['area', 'var(--accent)', '判定の20営業日'],
  ];
  return (
    <div className="samples-legend" aria-label="チャートの線の見分け方">
      {items.map(([kind, color, label]) => (
        <span key={label}>
          <i className={`sw ${kind}`} style={{ '--sw': color }} aria-hidden="true" />
          {label}
        </span>
      ))}
    </div>
  );
}

function CaseSection({ title, cases, total, empty }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>{title}</h2>
        <span className="sub">{`${fmtInt(total ?? 0)}件中 ${cases.length}件を表示`}</span>
      </div>
      {cases.length === 0 ? (
        <div className="empty">{empty}</div>
      ) : (
        <div className="samples-grid">
          {cases.map((c) => <SampleCard key={`${c.jqCode}-${c.t}`} c={c} />)}
        </div>
      )}
    </section>
  );
}

function SampleCard({ c }) {
  const lines = verdictLines(c);
  return (
    <article className="sample-card">
      <header className="sample-head">
        <div>
          <strong>{c.name || c.code}</strong>
          <span className="sub num">{c.code}</span>
        </div>
        <span className={`badge ${c.label === 1 ? 'green' : 'slate'}`}>{c.label === 1 ? '正例' : '負例'}</span>
      </header>
      <div className="sub sample-meta">
        {`更新日 ${fmtDate(c.t)}・寄り ${fmt(c.entryPrice, 1)}円・到達に必要 ${fmtSigned(c.verdict?.needPct, 1)}`}
      </div>
      <SampleChart c={c} />
      <ul className="sample-verdict">
        {lines.map((l) => (
          <li key={l.key} className={l.ok ? 'ok' : 'ng'}>
            <span className="mark" aria-hidden="true">{l.ok ? '✓' : '✗'}</span>
            <b>{l.label}</b> {l.text}
            <span className="sr-only">{l.ok ? '（満たす）' : '（満たさない）'}</span>
          </li>
        ))}
      </ul>
    </article>
  );
}

function SampleTooltip({ active, payload, entry }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const r = fromEntry(p.close, entry);
  return (
    <div className="card" style={{ padding: '6px 8px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-dim)' }}>{fmtDate(p.date)}</div>
      <div className="num">終値 {fmt(p.close, 1)}円</div>
      {r !== null && <div className="num" style={{ color: 'var(--text-dim)' }}>寄りから {fmtSigned(r, 1)}</div>}
    </div>
  );
}

function SampleChart({ c }) {
  const ch = caseChart(c);
  if (!ch.points.length) return <div className="empty">日足がありません。</div>;
  return (
    <ResponsiveContainer width="100%" height={180}>
      <ComposedChart data={ch.points} margin={{ top: 8, right: 6, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="var(--border-soft)" vertical={false} />
        <ReferenceArea x1={ch.win[0]} x2={ch.win[1]} fill="var(--accent)" fillOpacity={0.08} stroke="none"
                       ifOverflow="visible" />
        <XAxis dataKey="i" type="number" domain={[0, ch.points.length - 1]} ticks={ch.ticks}
               tickFormatter={(i) => fmtMD(ch.points[i]?.date)}
               tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border)" />
        <YAxis domain={ch.yDomain} ticks={ch.yTicks} width={50} allowDecimals={false}
               tick={{ fill: 'var(--text-faint)', fontSize: 10 }} stroke="var(--border)"
               tickFormatter={(v) => Math.round(v).toLocaleString('ja-JP')} />
        <ReferenceLine x={ch.tPos} stroke="var(--text-faint)" strokeDasharray="2 3" />
        {Number.isFinite(c.entryPrice) && (
          <ReferenceLine y={c.entryPrice} stroke="var(--text-dim)" strokeDasharray="1 3" strokeWidth={1}
                         label={{ value: '寄り', position: 'insideBottomLeft', fill: 'var(--text-dim)', fontSize: 10 }} />
        )}
        {Number.isFinite(c.target) && (
          <ReferenceLine y={c.target} stroke="var(--amber)" strokeDasharray="5 3" strokeWidth={1.5}
                         label={{ value: `到達 ${fmtSigned(c.thresholdPct, 1)}`, position: 'insideTopLeft',
                                  fill: 'var(--text-dim)', fontSize: 10 }} />
        )}
        {Number.isFinite(c.endLineAdopted) && (
          <ReferenceLine segment={[{ x: ch.endWin[0], y: c.endLineAdopted }, { x: ch.endWin[1], y: c.endLineAdopted }]}
                         stroke="var(--amber)" strokeDasharray="1 2" strokeWidth={2.5} ifOverflow="visible" />
        )}
        {Number.isFinite(c.entryPrice) && (
          <ReferenceDot x={ch.entryPos} y={c.entryPrice} r={4} fill="var(--text)" stroke="var(--bg-card)"
                        strokeWidth={2} ifOverflow="visible" />
        )}
        <Tooltip content={<SampleTooltip entry={c.entryPrice} />}
                 cursor={{ stroke: 'var(--text-faint)', strokeWidth: 1 }} />
        {/* 日足の終値を直線で結ぶ（曲線で補うと実際に無い高値・安値が描かれる） */}
        <Line type="linear" dataKey="close" stroke="var(--accent)" strokeWidth={2} dot={false}
              activeDot={{ r: 4, stroke: 'var(--bg-card)', strokeWidth: 2 }} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
