import React, { useEffect, useMemo, useState } from 'react';
import { fmt, fmtDate } from '../lib/format.js';
import { STRATEGY, orderRows, strategySignal } from '../lib/strategy.js';
import { loadValuations } from '../lib/valuation.js';
import { SD_SCALE, SUMMARY_AXES, summaryRow, valueScores } from '../lib/summary.js';
import SummaryRadar from '../components/SummaryRadar.jsx';

/**
 * サマリー（一番左のタブ。運用者の依頼 2026-11-21「パット見で、すべての良し悪しを測りたいので、レーダーチャートで…
 * 六角形で表示してほしいです…サマリーという名のタグで一番左に追加してほしいです」）。
 *
 * 選んだ日の候補ごとに、6つの頂点（6本の分類モデルの平均・最低値・偏差、ランキング学習の日次・5日、割安度）を 0〜1 に
 * そろえた六角形を並べる（外側ほど良い。そろえ方は src/lib/summary.js と docs/MODEL_ADOPTION_RULES.md §100）。
 * 並びはブレイク予測と同じ。印（買う候補・95以上）は今日の戦略の規則のまま。買う銘柄は変えない（並べて見るだけ）。
 */
export default function SummaryView({ pred, predError }) {
  const dates = pred?.dates || [];
  const [day, setDay] = useState(() => pred?.asOf || dates[dates.length - 1]);
  useEffect(() => { if (!day && pred) setDay(pred.asOf || (pred.dates || []).at(-1)); }, [pred, day]);

  // 理論株価（理論株価のタブと同じファイル）。読めなくても六角形は出し、割安度だけ「なし」にする
  const [vals, setVals] = useState(undefined);
  const [valError, setValError] = useState(null);
  useEffect(() => {
    let active = true;
    loadValuations().then((x) => { if (active) setVals(x); })
      .catch((e) => { if (active) { setVals(null); setValError(e.message); } });
    return () => { active = false; };
  }, []);

  const rows = useMemo(() => orderRows((pred?.candidates || []).filter((c) => c.date === day)), [pred, day]);
  const signal = useMemo(() => strategySignal(rows), [rows]);
  const pickIds = useMemo(() => new Set(signal.picks.map((c) => c.jqCode)), [signal]);
  const ruleIds = useMemo(() => new Set(signal.passed.map((c) => c.jqCode)), [signal]);
  const values = useMemo(() => valueScores(pred, vals), [pred, vals]);
  const items = useMemo(() => rows.map((c) => summaryRow(c, values)), [rows, values]);
  const shortOf = useMemo(() => Object.fromEntries((pred?.models || []).map((m) => [m.algo, m.short || m.algo])), [pred]);
  const nValued = values.size;

  if (predError) {
    return <div className="banner warn"><span>⚠</span><div>予測データを読み込めませんでした（{predError}）。</div></div>;
  }
  if (!pred) return <div className="card" role="status">予測データを読み込んでいます…</div>;

  return (
    <div className="summary">
      <section className="card">
        <div className="card-head">
          <h2>サマリー</h2>
          <span className="sub">候補ごとの六角形。6つの頂点はどれも 0〜1 で、外側ほど良い</span>
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
            <span className="lab">候補</span>
            <strong className="num">{rows.length}</strong> 銘柄
          </div>
        </div>
        <ul className="summary-defs">
          <li><b>平均・最低値</b>：6本の分類モデル（+10% と +15% の木3本ずつ）それぞれの「過去の分布での位置」（百分位）の平均・最小 ÷ 100</li>
          <li><b>偏差</b>：6本の百分位のそろい。1 − 標準偏差 ÷ {SD_SCALE}（{SD_SCALE} 以上は 0）。そろっているほど外側</li>
          <li><b>日次・5日</b>：ランキング学習の順位。その日の候補の中（日次）、直近5営業日の候補の組の中（5日）で 1位 = 1・最下位 = 0（1件だけなら 0.5）</li>
          <li><b>割安度</b>：直近5営業日の候補（理論株価を出せる {nValued} 銘柄）の中で、使用終値 ÷ 理論株価の中央値（RIM）が低い順。最も割安 = 1・最も割高 = 0</li>
        </ul>
        {rows.length === 0 && <div className="empty">この日は候補がありません。</div>}
      </section>

      <div className="summary-grid">
        {items.map((r) => (
          <SummaryCard key={r.c.jqCode} r={r} pick={pickIds.has(r.c.jqCode)} rule={ruleIds.has(r.c.jqCode)}
                       shortOf={shortOf} vals={vals} valError={valError} />
        ))}
      </div>
    </div>
  );
}

const two = (v) => (v === null ? 'なし' : v.toFixed(2));
const rankText = (rank, n) => (Number.isFinite(rank) && Number.isFinite(n)
  ? `${Number.isInteger(rank) ? rank : rank.toFixed(1)}位 / ${n}件` : '');

/** 頂点ごとの元の値（表の右の列）。 */
function detail(key, r, shortOf, vals, valError) {
  const { cls, ltr, ltr5, value } = r.raw;
  switch (key) {
    case 'mean': return cls ? `百分位の平均 ${fmt(cls.mean, 1)}` : '6本の百分位がそろわない';
    case 'min': return cls ? `最小 ${fmt(cls.min, 1)}（${shortOf[cls.minModel] || cls.minModel}）` : '6本の百分位がそろわない';
    case 'agree': return cls ? `標準偏差 ${fmt(cls.sd, 1)}` : '6本の百分位がそろわない';
    case 'ltr': return ltr ? rankText(ltr.rank, ltr.n) : '順位が無い';
    case 'ltr5': return ltr5 ? rankText(ltr5.rank, ltr5.n) : '順位が無い';
    case 'value':
      if (value) {
        const rank = Number.isInteger(value.rank) ? value.rank : value.rank.toFixed(1);
        return `${fmt(value.ratio * 100, 0)}%・${value.n}銘柄中 ${rank}位`;
      }
      if (vals === undefined) return '理論株価を読み込んでいます';
      if (valError) return '理論株価を読めない';
      return '理論株価を算出できない';
    default: return '';
  }
}

/** 候補1件のカード: 見出し・六角形・6つの値の表。 */
function SummaryCard({ r, pick, rule, shortOf, vals, valError }) {
  const c = r.c;
  return (
    <article className="summary-card">
      <div className="news-head">
        <div className="news-id">
          <strong><span className="num">{c.code}</span> {c.name}</strong>
          <span className="sub">{[c.market, c.sector].filter(Boolean).join(' / ')}</span>
        </div>
        <div className="news-badges">
          {pick && <span className="badge green" title="今日の戦略で買う候補（上位の枠に入った）">買う候補</span>}
          {!pick && rule && <span className="badge blue" title={`今日の戦略の規則を満たす（上位の枠の外）`}>{`${STRATEGY.agreePct}以上`}</span>}
        </div>
      </div>
      <SummaryRadar row={r} />
      <table className="summary-values">
        <caption className="sr-only">{`${c.code} ${c.name ?? ''} の6つの値`}</caption>
        <tbody>
          {SUMMARY_AXES.map((a) => (
            <tr key={a.key} className={r.axes[a.key] === null ? 'na' : undefined}>
              <th scope="row">{a.label}</th>
              <td className="num">{two(r.axes[a.key])}</td>
              <td className="sub">{detail(a.key, r, shortOf, vals, valError)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}
