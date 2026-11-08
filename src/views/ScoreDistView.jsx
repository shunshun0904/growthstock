import React from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { fmtDate, fmtInt } from '../lib/format.js';
import {
  histRows, toShares, countInto, liveLgbmScores, rangeLabel, decileLabel, pct, candidatePositions, candidateDates,
} from '../lib/scoredist.js';

/**
 * スコアと収益の分布（運用者の依頼 2026-11-08。AskUserQuestion「3つとも載せる」）。
 *   1. +10% の LightGBM（固定 +10% のラベル）の out-of-fold のスコアの分布と、直近の予測の候補のスコア、帯ごとの到達率
 *   2. 直近5営業日の組のランキング学習（ltr5）の out-of-fold のスコアと、収益（20営業日目の終値・20営業日以内の終値の最高値）
 *   3. 2つのモデルの上位10% の重なりと、組み合わせごとの収益
 * データは public/data/score_dist.json（Score Distribution が週次の再学習の後に research/score_dist.py で書く）。集計だけで、
 * 銘柄ごとの値は無い。直近の予測は予測ファイルの候補から数える。
 */
const AX = { fill: 'var(--text-faint)', fontSize: 10 };
const C = { neg: 'var(--slate)', pos: 'var(--amber)', live: 'var(--green)', all: 'var(--slate)', top: 'var(--accent)' };

export default function ScoreDistView({ data, pred }) {
  const dates = React.useMemo(() => candidateDates(pred), [pred]);
  const [date, setDate] = React.useState(null);
  const [selKey, setSelKey] = React.useState(null);
  const day = date && dates.includes(date) ? date : dates[0] || null;
  const cands = React.useMemo(() => (data?.lgbm ? candidatePositions(pred, data, day) : []), [pred, data, day]);
  const sel = cands.find((c) => c.key === selKey) || null;
  if (data === undefined) return <div className="empty">読み込んでいます…</div>;
  if (!data || !data.lgbm || !data.ltr5) {
    return (
      <div className="card empty">
        まだありません。週次の再学習（日曜）の後に <code>Score Distribution</code> が作ります。
      </div>
    );
  }
  return (
    <div className="sdist">
      <CandidatesSection rows={cands} dates={dates} day={day} sel={sel}
                         stale={staleModels(pred, data)}
                         onDate={(d) => { setDate(d); setSelKey(null); }}
                         onSelect={(k) => setSelKey(k === selKey ? null : k)} />
      <LgbmSection d={data.lgbm} pred={pred} sel={sel} />
      <Ltr5Section d={data.ltr5} sel={sel} />
      <OverlapSection d={data.overlap} />
      <p className="sub sdist-foot">
        {`集計の作成 ${fmtDate(data.generatedAt)}。収益の確かめ: 20営業日目の終値の収益を日次の計算と `
          + `${fmtInt(data.checks?.r20?.same)} / ${fmtInt(data.checks?.r20?.rows)} 行で突き合わせて一致。`
          + 'out-of-fold は、その行より前のデータだけで学習したモデルの採点（学習に使っていない行の成績）。'}
      </p>
    </div>
  );
}

function Strip({ items }) {
  return (
    <div className="pred-strip">
      {items.map(([lab, val, sub]) => (
        <div key={lab}>
          <span className="lab">{lab}</span>
          <span className="num">{val}</span>
          {sub ? <span className="sub"> {sub}</span> : null}
        </div>
      ))}
    </div>
  );
}

function CountTip({ active, payload, keys, unit = '件', digits = 0, rangeDigits = 3, rangeUnit = '' }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="card" style={{ padding: '6px 8px', fontSize: 12 }}>
      <div style={{ color: 'var(--text-dim)' }}>{rangeLabel(p.lo, p.hi, rangeDigits, rangeUnit)}</div>
      {keys.map(([k, lab]) => (
        <div key={k} className="num">{`${lab} ${Number(p[k]).toFixed(digits)}${unit}`}</div>
      ))}
    </div>
  );
}

function LgbmSection({ d, pred, sel }) {
  const edges = d.hist?.edges || [];
  const rows = histRows(edges, { neg: d.hist?.neg, pos: d.hist?.pos });
  const live = liveLgbmScores(pred);
  const liveRows = histRows(edges, { live: countInto(live.scores, edges) });
  const q = d.quantiles || {};
  const refs = [['0.5', '中央'], ['0.9', '上位10%'], ['0.99', '上位1%']].filter(([k]) => q[k] != null);
  const bands = (d.bands || []).map((b) => ({ ...b, name: rangeLabel(b.lo, b.hi, 1), rate: (b.posRate ?? 0) * 100 }));
  const nPos = (d.hist?.pos || []).reduce((a, b) => a + b, 0);
  return (
    <section className="card">
      <div className="card-head">
        <h2>+10% の LightGBM のスコアの分布</h2>
        <span className="sub">out-of-fold（学習に使っていない行の採点）と、直近の予測の候補</span>
      </div>
      <Strip items={[
        ['out-of-fold', `${fmtInt(d.n)} 件`, `${d.from} 〜 ${d.to}`],
        ['+10% に届いた', pct(d.positiveRate, 2), `（${fmtInt(nPos)} 件）`],
        ['PR-AUC / ROC-AUC', `${d.prAuc ?? '—'} / ${d.rocAuc ?? '—'}`],
        ['スコア 中央 / 90% / 99% 点', `${q['0.5'] ?? '—'} / ${q['0.9'] ?? '—'} / ${q['0.99'] ?? '—'}`],
        ['モデルの学習', fmtDate(d.trainedAt)],
      ]} />
      <h3 className="sdist-h">out-of-fold のスコア（棒を2色に積む: 灰 = 届かなかった・橙 = +10% に届いた）</h3>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={rows} margin={{ top: 16, right: 8, bottom: 0, left: -8 }} barCategoryGap={1}>
          <CartesianGrid stroke="var(--border-soft)" vertical={false} />
          <XAxis dataKey="lo" tick={AX} stroke="var(--border)" tickFormatter={(v) => v.toFixed(2)} interval={3} />
          <YAxis tick={AX} stroke="var(--border)" width={46} />
          <Tooltip content={<CountTip keys={[['neg', '届かなかった'], ['pos', '届いた']]} />}
                   cursor={{ fill: 'var(--border-soft)' }} />
          <Bar dataKey="neg" stackId="s" fill={C.neg} isAnimationActive={false} name="届かなかった" />
          <Bar dataKey="pos" stackId="s" fill={C.pos} isAnimationActive={false} name="+10% に届いた" />
          {refs.map(([k, lab]) => {
            const i = rows.findIndex((r) => q[k] >= r.lo && q[k] < r.hi);
            return i < 0 ? null : (
              <ReferenceLine key={k} x={rows[i].lo} stroke="var(--text-dim)" strokeDasharray="3 3"
                             label={{ value: `${lab} ${q[k]}`, position: 'top', fill: 'var(--text-dim)', fontSize: 10 }} />
            );
          })}
          {sel && (() => {
            const r = rows.find((x) => sel.score >= x.lo && sel.score < x.hi) || rows[rows.length - 1];
            return r ? (
              <ReferenceLine x={r.lo} stroke="var(--green)" strokeWidth={2}
                             label={{ value: `${sel.name || sel.code} ${sel.score.toFixed(3)}`, position: 'insideTopRight',
                                      fill: 'var(--green)', fontSize: 11 }} />
            ) : null;
          })()}
        </BarChart>
      </ResponsiveContainer>
      <h3 className="sdist-h">
        {live.n
          ? `直近の予測の候補のスコア（${live.n} 件・${live.dates[0]} 〜 ${live.dates[1]}。同じ区間で数えた）`
          : '直近の予測の候補のスコア（固定 +10% で採点した候補はまだありません）'}
      </h3>
      {live.n > 0 && (
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={liveRows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }} barCategoryGap={1}>
            <CartesianGrid stroke="var(--border-soft)" vertical={false} />
            <XAxis dataKey="lo" tick={AX} stroke="var(--border)" tickFormatter={(v) => v.toFixed(2)} interval={3} />
            <YAxis tick={AX} stroke="var(--border)" width={46} allowDecimals={false} />
            <Tooltip content={<CountTip keys={[['live', '候補']]} />} cursor={{ fill: 'var(--border-soft)' }} />
            <Bar dataKey="live" fill={C.live} isAnimationActive={false} />
            {q['0.5'] != null && (
              <ReferenceLine x={rows[Math.max(rows.findIndex((r) => q['0.5'] >= r.lo && q['0.5'] < r.hi), 0)]?.lo}
                             stroke="var(--text-dim)" strokeDasharray="3 3" />
            )}
          </BarChart>
        </ResponsiveContainer>
      )}
      <h3 className="sdist-h">スコア帯ごとの成績（out-of-fold）</h3>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr><th>スコア</th><th>件数</th><th>+10% 到達率</th><th>収益の平均</th><th>収益の中央</th></tr>
          </thead>
          <tbody>
            {bands.map((b) => (
              <tr key={b.name}>
                <td>{b.name}</td><td className="num">{fmtInt(b.n)}</td><td className="num">{pct(b.posRate)}</td>
                <td className="num">{pct(b.retMean, 2, true)}</td><td className="num">{pct(b.retMedian, 2, true)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="sub sdist-note">
        {'収益は、更新日の翌営業日の寄りで買い、20営業日後の5日平均の終値で売った上昇率（予測タブのスコア帯と同じ）。'
          + `点線（直近の予測のグラフ）は out-of-fold の中央。全体の +10% 到達率は ${pct(d.positiveRate)}。`}
      </p>
    </section>
  );
}

function ReturnHist({ h, title }) {
  const sAll = toShares(h?.all);
  const sTop = toShares(h?.top1);
  const rows = histRows(h?.edges, { all: sAll, top1: sTop });
  const e = h?.edges || [];
  return (
    <>
      <h3 className="sdist-h">{`${title}（割合。端は ${e[0]}% / ${e[e.length - 1]}% にまとめた）`}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }} barCategoryGap={1} barGap={0}>
          <CartesianGrid stroke="var(--border-soft)" vertical={false} />
          <XAxis dataKey="lo" tick={AX} stroke="var(--border)" interval={3}
                 tickFormatter={(v) => `${v > 0 ? '+' : ''}${Math.round(v)}%`} />
          <YAxis tick={AX} stroke="var(--border)" width={46} tickFormatter={(v) => `${v}%`} />
          <Tooltip content={<CountTip keys={[['all', '組の行すべて'], ['top1', '組の1位']]} unit="%" digits={1}
                                      rangeDigits={1} rangeUnit="%" />}
                   cursor={{ fill: 'var(--border-soft)' }} />
          <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-dim)' }} />
          <ReferenceLine x={rows.find((r) => r.lo <= 0 && r.hi > 0)?.lo} stroke="var(--text-faint)" />
          <Bar dataKey="all" fill={C.all} isAnimationActive={false} name="組の行すべて" />
          <Bar dataKey="top1" fill={C.top} isAnimationActive={false} name="組の1位" />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

function Ltr5Section({ d, sel }) {
  const s = d.summary || {};
  const scoreRows = histRows(d.scoreHist?.edges, { n: d.scoreHist?.counts });
  const dec = (d.deciles || []).map((x) => ({
    ...x, name: decileLabel(x.dec), r20: (x.r20Mean ?? 0) * 100, rmax: (x.rmaxMean ?? 0) * 100,
  }));
  const rowsT = [['組の行すべて', s.all], ['組の1位', s.top1], ['組の上位10%', s.top10], ['組の下位10%', s.bottom10]];
  return (
    <section className="card">
      <div className="card-head">
        <h2>直近5営業日の組のランキング学習（ltr5）の収益の分布</h2>
        <span className="sub">out-of-fold。判断日ごとに直近5営業日の候補を1つの組にして並べる</span>
      </div>
      <Strip items={[
        ['組の行', `${fmtInt(d.nRows)} 件`, `判断日 ${fmtInt(d.nDays)} 日・${d.from} 〜 ${d.to}`],
        ['段階', `${d.nGrades ?? '—'} 段階`],
        ['NDCG@5', d.ndcg5 ?? '—'],
        ['モデルの学習', fmtDate(d.trainedAt)],
      ]} />
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th></th><th>件数</th><th>20営業日目の終値 平均</th><th>中央</th><th>終値の最高値 平均</th><th>中央</th>
              <th>最高値が +10% 以上</th><th>終値が −10% 未満</th>
            </tr>
          </thead>
          <tbody>
            {rowsT.map(([lab, r]) => (r ? (
              <tr key={lab}>
                <td>{lab}</td><td className="num">{fmtInt(r.n)}</td>
                <td className="num">{pct(r.r20Mean, 2, true)}</td><td className="num">{pct(r.r20Median, 2, true)}</td>
                <td className="num">{pct(r.rmaxMean, 2, true)}</td><td className="num">{pct(r.rmaxMedian, 2, true)}</td>
                <td className="num">{pct(r.rmaxHit10)}</td><td className="num">{pct(r.r20Below10)}</td>
              </tr>
            ) : null))}
          </tbody>
        </table>
      </div>
      <ReturnHist h={d.r20Hist} title="20営業日目の終値で売った収益" />
      <ReturnHist h={d.rmaxHist} title="20営業日以内の終値の最高値で売った収益" />
      <h3 className="sdist-h">組の中の順位の10分位ごとの平均収益（右ほど上位）</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={dec} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
          <CartesianGrid stroke="var(--border-soft)" vertical={false} />
          <XAxis dataKey="name" tick={AX} stroke="var(--border)" interval={0} />
          <YAxis tick={AX} stroke="var(--border)" width={46} tickFormatter={(v) => `${v}%`} />
          <ReferenceLine y={0} stroke="var(--text-faint)" />
          <Tooltip formatter={(v, n) => [`${Number(v).toFixed(2)}%`, n]}
                   contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 11, color: 'var(--text-dim)' }} />
          <Line dataKey="r20" name="20営業日目の終値（平均）" stroke={C.top} strokeWidth={2} dot={{ r: 3 }}
                isAnimationActive={false} />
          <Line dataKey="rmax" name="終値の最高値（平均）" stroke={C.pos} strokeWidth={2} strokeDasharray="5 3"
                dot={{ r: 3 }} isAnimationActive={false} />
          {sel?.ltr5 && (
            <ReferenceLine x={decileLabel(sel.ltr5.dec)} stroke="var(--green)" strokeWidth={2}
                           label={{ value: `${sel.name || sel.code} ${sel.ltr5.rank}位 / ${sel.ltr5.nPool}`,
                                    position: 'insideTopLeft', fill: 'var(--green)', fontSize: 11 }} />
          )}
        </LineChart>
      </ResponsiveContainer>
      <p className="sub sdist-note">
        {'収益は、判断日の翌営業日の寄りで買い、20営業日目の終値で売った収益と、同じ20営業日の終値の最高値で売った収益'
          + '（最高値の日を前もって知っていれば取れた収益で、そのまま取れる収益ではない）。ltr5 は20営業日目の終値の並びを学ぶ。'
          + 'スコアは組の中の並びにだけ意味がある。'}
      </p>
      <h3 className="sdist-h">ltr5 の out-of-fold スコア（参考）</h3>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={scoreRows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }} barCategoryGap={1}>
          <CartesianGrid stroke="var(--border-soft)" vertical={false} />
          <XAxis dataKey="lo" tick={AX} stroke="var(--border)" tickFormatter={(v) => v.toFixed(1)} interval={4} />
          <YAxis tick={AX} stroke="var(--border)" width={46} />
          <Tooltip content={<CountTip keys={[['n', '組の行']]} rangeDigits={1} />} cursor={{ fill: 'var(--border-soft)' }} />
          <Bar dataKey="n" fill={C.all} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

function OverlapSection({ d }) {
  if (!d) return null;
  const g = d.groups || [];
  const t1 = d.ltr5Top1 || {};
  const pk = d.lgbmPick || {};
  return (
    <section className="card">
      <div className="card-head">
        <h2>2つのモデルの上位の重なり</h2>
        <span className="sub">
          {`lgbm の上位10%（out-of-fold のスコア ${d.lgbmThreshold} 以上）と、ltr5 の組の中の上位10%。組の行で数える`}
        </span>
      </div>
      <Strip items={[
        ['両方の上位10%', `${fmtInt(d.both)} 件`, `（でたらめに選んだときの見込み ${fmtInt(Math.round(d.expectedBoth))} 件）`],
        ['ltr5 の上位10% のうち lgbm も上位10%', pct(d.ltr5TopInLgbmTop)],
        ['lgbm の上位10% のうち ltr5 も上位10%', pct(d.lgbmTopInLtr5Top)],
        ['組の中の2つの順位の順位相関', d.rankCorrInPool ?? '—'],
      ]} />
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th></th><th>件数</th><th>20営業日目の終値 平均</th><th>終値の最高値 平均</th>
              <th>最高値が +10% 以上</th><th>終値が −10% 未満</th>
            </tr>
          </thead>
          <tbody>
            {g.map((r) => (
              <tr key={r.key}>
                <td>{r.label}</td><td className="num">{fmtInt(r.n)}</td>
                <td className="num">{pct(r.r20Mean, 2, true)}</td><td className="num">{pct(r.rmaxMean, 2, true)}</td>
                <td className="num">{pct(r.rmaxHit10)}</td><td className="num">{pct(r.r20Below10)}</td>
              </tr>
            ))}
            <tr>
              <td>組の中で lgbm の1位</td><td className="num">{fmtInt(pk.n)}</td>
              <td className="num">{pct(pk.r20Mean, 2, true)}</td><td className="num">{pct(pk.rmaxMean, 2, true)}</td>
              <td className="num">{pct(pk.rmaxHit10)}</td><td className="num">{pct(pk.r20Below10)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="sub sdist-note">
        {`ltr5 の組の1位（${fmtInt(t1.n)} 日）のうち、lgbm の上位10% は ${pct(t1.lgbmTop10)}・上位1% は ${pct(t1.lgbmTop1)}・`
          + `中央より上は ${pct(t1.lgbmAboveMedian)}。組の中で lgbm の1位を選ぶと、ltr5 の1位と同じ銘柄の日は ${pct(pk.sameAsLtr5Top1)}。`
          + '買い方は ltr5 と同じ（判断日の翌営業日の寄り）。組には5営業日前までの候補も入るので、lgbm にとっては本来の買い日'
          + 'より最大4日遅い。「最高値が +10% 以上」は lgbm の正解ラベル（終盤 +5%・20日目の MA5 ≥ MA20 も条件）と同じではない。'
          + '同じ候補が最大5つの組に入るので、行どうしは独立していない。'}
      </p>
    </section>
  );
}

/** 予測ファイルのモデル（予測のときに使った）と、分布の集計のモデルの学習日が違うもの。 */
function staleModels(pred, data) {
  const out = [];
  const at = (v) => (v ? String(v).slice(0, 19) : null);
  if (at(pred?.model?.trainedAt) && at(data?.lgbm?.trainedAt) && at(pred.model.trainedAt) !== at(data.lgbm.trainedAt)) {
    out.push('lgbm');
  }
  if (at(pred?.ltr5?.trainedAt) && at(data?.ltr5?.trainedAt) && at(pred.ltr5.trainedAt) !== at(data.ltr5.trainedAt)) {
    out.push('ltr5');
  }
  return out;
}

/**
 * 予測の候補が分布のどこにいるか（運用者の依頼 2026-11-08「予測を実施したときに、各サンプルが分布のどこに位置するのか、
 * 表示してほしいです」）。予測ファイルの候補を、lgbm の out-of-fold の中の百分位・スコア帯、ltr5 の組の中の10分位、
 * 重なりの組み合わせに当て、それぞれの out-of-fold の成績を並べる。行を押すと、下の図にその候補の位置の線を引く。
 */
function CandidatesSection({ rows, dates, day, sel, stale, onDate, onSelect }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>予測の候補が分布のどこにいるか</h2>
        <span className="sub">行を押すと、下の図（lgbm のスコア・ltr5 の10分位）にその候補の位置を緑の線で引く</span>
      </div>
      {dates.length > 1 && (
        <div className="sdist-dates" role="group" aria-label="予測の日付">
          {dates.map((d) => (
            <button key={d} type="button" className={`btn btn-sm${d === day ? ' btn-primary' : ''}`}
                    aria-pressed={d === day} onClick={() => onDate(d)}>{d}</button>
          ))}
        </div>
      )}
      {stale.length > 0 && rows.length > 0 && (
        <p className="sub sdist-warn">
          {`この日付の候補は、${stale.join('・')} を今の分布より前のモデルで採点した。下の分布は今のモデルの out-of-fold `
            + 'なので、位置は目安（次の予測から同じモデルになる）。'}
        </p>
      )}
      {!rows.length ? (
        <div className="empty">固定 +10% で採点した予測の候補はまだありません。</div>
      ) : (
        <div className="tbl-wrap">
          <table className="tbl sdist-cands">
            <thead>
              <tr>
                <th>銘柄</th><th>lgbm スコア</th><th>out-of-fold の中の位置</th><th>その帯の +10% 到達率</th>
                <th>ltr5 の組の中</th><th>その10分位の 20営業日目の終値</th><th>終値の最高値</th>
                <th>重なり</th><th>その組み合わせの −10% 未満</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key} className={sel?.key === r.key ? 'sel' : ''} tabIndex={0}
                    onClick={() => onSelect(r.key)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(r.key); } }}>
                  <td><strong>{r.name || r.code}</strong> <span className="sub num">{r.code}</span></td>
                  <td className="num">{r.score.toFixed(3)}</td>
                  <td className="num">{r.lgbmPct == null ? '—' : `上位 ${(100 - r.lgbmPct).toFixed(1)}%`}
                    {r.lgbmTop10 ? <span className="badge green sdist-badge">上位10%</span> : null}</td>
                  <td className="num">{r.band ? `${pct(r.band.posRate)}（${rangeLabel(r.band.lo, r.band.hi, 1)}）` : '—'}</td>
                  <td className="num">{r.ltr5 ? `${r.ltr5.rank}位 / ${r.ltr5.nPool}（${decileLabel(r.ltr5.dec)}）` : '—'}</td>
                  <td className="num">{r.ltr5?.stats ? pct(r.ltr5.stats.r20Mean, 2, true) : '—'}</td>
                  <td className="num">{r.ltr5?.stats ? pct(r.ltr5.stats.rmaxMean, 2, true) : '—'}</td>
                  <td>{r.group?.label || '—'}</td>
                  <td className="num">{r.group?.r20Below10 != null ? pct(r.group.r20Below10) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="sub sdist-note">
        {'out-of-fold の中の位置は、候補の lgbm のスコアが学習に使っていない行の採点の中で上から何%か（予測は全期間で学習した'
          + 'モデル、out-of-fold は期間ごとのモデルの採点なので、物差しは近いが同じではない）。ltr5 の組の中の10分位は、予測の日の'
          + '直近5営業日の候補の組の中の順位（右ほど上位）。成績はどれも過去の out-of-fold の平均で、その候補の見込みではない。'}
      </p>
    </section>
  );
}
