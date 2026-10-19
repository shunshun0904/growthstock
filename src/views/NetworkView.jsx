import React, { useEffect, useMemo, useRef, useState } from 'react';
import RecurrenceGraph from '../components/RecurrenceGraph.jsx';
import { fmt, fmtDate, fmtDateTime, fmtInt, fmtSigned } from '../lib/format.js';
import { orderRows, strategySignal, RULE_JA, STRATEGY } from '../lib/strategy.js';
import { labelOf } from '../lib/knn.js';
import { loadRecurrenceIndex, loadRecurrenceFile } from '../lib/predictions.js';
import { FORCE_DEFAULTS } from '../lib/forcegraph.js';
import {
  methodOf, decode, network, neighborsOf, summarize, todayNeighbors, nodeColors, classCounts, outcomeStats, sqDist,
  zOf, classesOf, COLOR_MODES, TIME_STOPS,
} from '../lib/recurrence.js';

/**
 * リカレンス・ネットワーク（購入条件を満たした銘柄ごと）。運用者の依頼（2026-10-19）「Obsidian の球状のプロットのようなことを、
 * ノードを1営業日として、本予測モデルに使えそうな可視化を。市場全体よりも、その日の各候補銘柄単体で。指数リカレントネットワークが
 * 良い。特徴量はいったん、重要度の高いテクニカル系のみ」→ 運用者の選択: ε-リカレンス・ネットワーク / 実験105 の gain 上位9列 /
 * 取れる全期間 / 画面の新しいタブ / データは Release に置く（research/recurrence_net.py）。
 * 対象は同日の指定「候補といったのは、予測サンプルではなく、その中でも購入条件を満たした銘柄です」のとおり、その日の高値更新の
 * 銘柄のうち購入条件（1.5σ の3本がすべて95以上。今日の戦略の strategySignal の passed）を満たした銘柄だけ。
 *
 * 点 = その銘柄の1営業日、線 = テクニカル指標9列の状態が似た日どうし（20営業日より離れた日だけ。辺の密度で ε を決める）。
 * 点の色はその日に候補になっていたら、の結果（+10% の指値か20営業日目の終値・目的変数 1.5σ / 1.2σ。予測日までに分かったものだけ）。
 * 今日（候補の日）の近傍は、その銘柄の過去で今日と似た状態だった日で、どれも結果が出そろっている。
 * 図の力は Obsidian のグラフ表示と同じ4つのつまみ（中心の力・反発の力・リンクの力・リンクの長さ）で変えられる。
 * **参考の表示で、選定には使わない。** 買う銘柄は今日の戦略（1.5σ の3本の最小の順）のまま。
 */

const pct = (v) => (Number.isFinite(v) ? `${fmt(v * 100, 0)}%` : '—');
const TRAIL_DAYS = 20;   // 直近の軌跡の長さ（運用者の指定 2026-10-19「軌跡は直近20営業日で良いです」。前は60営業日）
const bpText = (b) => `${fmt(b / 100, b % 100 ? 1 : 0)}%`;

/** 力のつまみ（Obsidian のグラフ表示の Forces と同じ4つ）。反発は画面では正の数で見せる */
const FORCE_SLIDERS = [
  { key: 'gravity', label: '中心の力', min: 0, max: 0.5, step: 0.01, show: (v) => fmt(v, 2) },
  { key: 'charge', label: '反発の力', min: 0, max: 200, step: 5, sign: -1, show: (v) => fmtInt(-v) },
  { key: 'linkStrength', label: 'リンクの力', min: 0, max: 10, step: 0.5, show: (v) => fmt(v, 1) },
  { key: 'linkDistance', label: 'リンクの長さ', min: 5, max: 100, step: 5, show: (v) => fmtInt(v) },
];
const DEFAULT_FORCES = Object.fromEntries(FORCE_SLIDERS.map((s) => [s.key, FORCE_DEFAULTS[s.key]]));

function labelText(v) {
  if (!Number.isFinite(v)) return '未確定';
  return v >= 0.5 ? '正例' : '負例';
}

function retText(r, target) {
  if (!Number.isFinite(r)) return '未確定';
  if (r >= target - 1e-9) return `+${fmt(target, 0)}% 到達`;
  return fmtSigned(r, 1);
}

/** 買値（翌営業日の寄り）から見た先20営業日の値動き（%）。分からない日は「未確定」 */
function fwdText(v) {
  return Number.isFinite(v) ? fmtSigned(v, 1) : '未確定';
}

function rawText(v) {
  if (!Number.isFinite(v)) return '—';
  return fmt(v, Math.abs(v) >= 10 ? 1 : 2);
}

/**
 * 一覧に出す銘柄: その日の高値更新の銘柄（予測ファイルの候補）のうち、購入条件（1.5σ の3本がすべて95以上）を満たした銘柄だけ。
 * 今日の戦略（strategySignal）の passed と同じ銘柄・同じ並び（3本の最小の順）で、上位の枠に入った銘柄（picks）は pick = true
 * （「買う候補」の印。候補銘柄のニュースのタブと同じ）。目次の項目と作れなかった理由を付ける。
 * 戻り値: { rows, nBreak }。nBreak はその日の高値更新の件数（予測ファイルにその日が無ければ目次の値、それも無ければ null）。
 */
export function candidateRows(data, block, day) {
  const entries = new Map((block?.candidates || []).map((c) => [c.jqCode, c]));
  const skipped = new Map((block?.skipped || []).map((s) => [s.jqCode, s.reason]));
  const all = orderRows((data?.candidates || []).filter((c) => c.date === day));
  const signal = strategySignal(all);
  const picks = new Set(signal.picks.map((c) => c.jqCode));
  const rows = signal.passed.map((c) => ({
    code: c.code, jqCode: c.jqCode, name: c.name, ruleMin: c.minPct, pick: picks.has(c.jqCode),
    entry: entries.get(c.jqCode) || null, reason: skipped.get(c.jqCode) || null,
  }));
  // 予測ファイルに無い銘柄（予測ファイルが先に新しい日へ進んだ・読めなかった）も、目次にあれば後ろに出す（目次の銘柄は、作った
  // ときに購入条件を満たした銘柄）。予測ファイルにあって条件を満たさない銘柄は出さない
  const known = new Set(all.map((c) => c.jqCode));
  for (const [jq, e] of entries) {
    if (!known.has(jq)) rows.push({ code: e.code, jqCode: jq, name: e.name, ruleMin: null, pick: false, entry: e, reason: null });
  }
  const inPred = all.length > 0 || (data?.dates || []).includes(day);
  const nBreak = inPred ? all.length : (Number.isInteger(block?.nBreak) ? block.nBreak : null);
  return { rows: rows.map((r, i) => ({ ...r, label: labelOf(i) })), nBreak };
}

export default function NetworkView({ data }) {
  const [index, setIndex] = useState(undefined);       // undefined = 読み込み中、null = まだ無い
  const [day, setDay] = useState(null);
  const [jq, setJq] = useState(null);
  const [payload, setPayload] = useState(null);         // { file, data } / { file, error } / { file, loading }
  const [rrBp, setRrBp] = useState(null);
  const [mode, setMode] = useState('outcome');
  const [dim, setDim] = useState(3);
  const [spin, setSpin] = useState(true);
  const [trail, setTrail] = useState(true);
  const [selected, setSelected] = useState(null);
  const [zoomCmd, setZoomCmd] = useState(null);
  const [forces, setForces] = useState(DEFAULT_FORCES);
  const [showForces, setShowForces] = useState(false);
  const cache = useRef(new Map());

  useEffect(() => {
    let cancelled = false;
    loadRecurrenceIndex().then((d) => { if (!cancelled) setIndex(d); });
    return () => { cancelled = true; };
  }, []);

  const method = useMemo(() => methodOf(index), [index]);
  const days = index?.days || [];
  useEffect(() => {
    if (!index) return;
    const has = (d) => days.some((x) => x.date === d);
    if (!day || !has(day)) setDay(has(index.asOf) ? index.asOf : days[days.length - 1]?.date ?? null);
    if (rrBp === null) setRrBp(method.rrBp);
  }, [index]); // eslint-disable-line react-hooks/exhaustive-deps

  const block = days.find((d) => d.date === day) || null;
  const { rows, nBreak } = useMemo(() => candidateRows(data, block, day), [data, block, day]);

  useEffect(() => {
    if (!rows.some((r) => r.jqCode === jq && r.entry)) setJq(rows.find((r) => r.entry)?.jqCode ?? null);
  }, [rows]); // eslint-disable-line react-hooks/exhaustive-deps

  const row = rows.find((r) => r.jqCode === jq) || null;
  const file = row?.entry?.file || null;

  useEffect(() => {
    setSelected(null);
    if (!file) { setPayload(null); return undefined; }
    const hit = cache.current.get(file);
    if (hit) { setPayload({ file, data: hit }); return undefined; }
    let cancelled = false;
    setPayload({ file, loading: true });
    loadRecurrenceFile(file)
      .then((d) => {
        cache.current.set(file, d);
        if (!cancelled) setPayload({ file, data: d });
      })
      .catch((e) => { if (!cancelled) setPayload({ file, error: e.message }); });
    return () => { cancelled = true; };
  }, [file]);

  const dec = useMemo(() => {
    if (!payload?.data) return null;
    try {
      return decode(payload.data, method.labels);
    } catch (e) {
      return { error: e.message };
    }
  }, [payload, method]);
  const ok = Boolean(dec && !dec.error);
  const bp = rrBp ?? method.rrBp;
  const net = useMemo(() => (ok ? network(dec, method.theiler, bp) : null), [dec, ok, method, bp]);
  const nbrs = useMemo(() => (net ? neighborsOf(net, dec.n) : null), [net, dec]);
  const summary = useMemo(
    () => (net ? summarize(dec, net, method.theiler, bp, { target: method.target, labels: method.labels }) : null),
    [net, dec, method, bp],
  );
  const colors = useMemo(() => (ok ? nodeColors(dec, mode, method.target) : null), [dec, ok, mode, method]);
  const counts = useMemo(() => (ok && mode !== 'time' ? classCounts(dec, mode, method.target) : null), [dec, ok, mode, method]);
  // 今日に似た日（ε の中の日）の色の区分ごとの件数（凡例の下に文字で出す。色だけに頼らない）
  const nbrCounts = useMemo(
    () => (nbrs && mode !== 'time' ? classCounts(dec, mode, method.target, nbrs.of(dec.n - 1)) : null),
    [nbrs, dec, mode, method],
  );
  const nearList = useMemo(() => (net ? todayNeighbors(dec, net, method.theiler) : []), [net, dec, method]);
  const valueMode = COLOR_MODES.find((m) => m.id === mode && m.value) || null;   // 表に値の列を足す色の付け方

  const tipOf = (i) => {
    if (!ok || !net) return null;
    const today = dec.n - 1;
    return {
      title: `${fmtDate(dec.dates[i])}${i === today ? '（今日）' : ''}${dec.isFresh[i] ? '・78週高値の更新日' : ''}`,
      lines: [
        ['結果', retText(dec.ret[i], method.target)],
        ['目的変数 1.5σ', labelText(dec.labels.l15?.[i])],
        ['目的変数 1.2σ', labelText(dec.labels.l12?.[i])],
        ['20日以内の最高値まで', fwdText(dec.rise[i])],
        ['20日以内の最安値まで', fwdText(dec.fall[i])],
        ['20営業日目の騰落率', fwdText(dec.r20[i])],
        ['似た日（次数）', `${fmtInt(net.deg[i])} 日`],
        ['今日との距離', `${fmt(Math.sqrt(sqDist(dec, i, today)) / dec.q, 2)}σ`],
        ['終値（調整後）', fmt(dec.close[i], dec.close[i] >= 1000 ? 0 : 1)],
      ],
    };
  };

  // 日の単位で出せないとき（目次が無い・その日が無い・作れなかった・候補が無い）は、図の枠ごと出さない
  const dayStatus = (() => {
    if (index === undefined) return { kind: 'loading', text: 'リカレンス・ネットワークを読み込んでいます…' };
    if (index === null) {
      return { kind: 'none', text: 'まだネットワークのデータがありません。日次予測（Predict Breakouts）が Release に上げ、次のデプロイで画面に入ります。' };
    }
    if (!block) return { kind: 'noday', text: 'この日のネットワークはありません。' };
    if (block.status === 'error') return { kind: 'error', text: `この日はネットワークを作れませんでした（${block.reason || '理由は記録されていません'}）。` };
    if (rows.length === 0) {
      if (nBreak === 0) return { kind: 'empty', text: 'この日は新規の高値更新がありません。' };
      return {
        kind: 'empty',
        text: `この日は購入条件（${RULE_JA}がすべて${STRATEGY.agreePct}以上）を満たした銘柄がありません`
          + (Number.isInteger(nBreak) ? `（高値更新 ${nBreak} 銘柄のうち）。` : '。'),
      };
    }
    if (!rows.some((r) => r.entry)) {
      return {
        kind: 'empty',
        text: `この日は購入条件を満たした ${rows.length} 銘柄のネットワークを作れませんでした（`
          + rows.map((r) => `${r.code} ${r.reason || '理由は記録されていません'}`).join('・') + '）。',
      };
    }
    return { kind: 'ok' };
  })();
  // 候補の単位（読み込み中・読めない）は、候補の一覧を出したまま図の枠に書く
  const candStatus = (() => {
    if (!payload || payload.loading) return { kind: 'loading', text: '候補のネットワークを読み込んでいます…' };
    if (payload.error) return { kind: 'error', text: `候補のファイルを読み込めませんでした（${payload.error}）。` };
    if (dec?.error) return { kind: 'error', text: `候補のファイルの形が違います（${dec.error}）。` };
    return { kind: 'ok' };
  })();

  const past = Boolean(index?.asOf) && Boolean(day) && day !== index.asOf;
  const feat = method.features;
  const todayZ = ok ? zOf(dec, dec.n - 1) : [];
  const selZ = ok && Number.isInteger(selected) ? zOf(dec, selected) : null;
  const selStats = ok && Number.isInteger(selected) && nbrs
    ? outcomeStats(dec, Array.from(nbrs.of(selected)), method.target, method.labels) : null;
  const showGraph = candStatus.kind === 'ok' && net && summary;

  return (
    <div className="net">
      <section className="card">
        <div className="card-head">
          <h2>リカレンス・ネットワーク（購入条件を満たした銘柄ごと）</h2>
          <span className="sub">
            {`購入条件（${RULE_JA}がすべて${STRATEGY.agreePct}以上）を満たした銘柄ごとに、点 = その銘柄の1営業日、`
              + '線 = テクニカル指標9列の状態が似た日どうし。今日（候補の日）に似た過去の日がその後どうなったかを見ます'}
          </span>
        </div>

        <div className="pred-strip">
          <div>
            <span className="lab">対象日</span>
            <div className="pred-days">
              {days.map((d) => (
                <button key={d.date} className={`btn btn-ghost${d.date === day ? ' on' : ''}`} onClick={() => setDay(d.date)}>
                  {fmtDate(d.date)}
                </button>
              ))}
            </div>
          </div>
          {block && (
            <div>
              <span className="lab">購入条件を満たした銘柄</span>
              <strong className="num">{rows.length}</strong> 銘柄
              {Number.isInteger(nBreak) && <span className="sub">{`（この日の高値更新 ${nBreak} 銘柄のうち）`}</span>}
            </div>
          )}
          <div>
            <span className="lab">辺の密度</span>
            <div className="net-seg" role="group" aria-label="辺の密度（リカレンス率）">
              {method.rrChoicesBp.map((b) => (
                <button key={b} className={`btn btn-ghost${b === bp ? ' on' : ''}`} aria-pressed={b === bp}
                        onClick={() => setRrBp(b)}
                        title={`結べる組（${method.theiler}営業日より離れた日どうし）のうち、距離の近い ${bpText(b)} を結ぶ`}>
                  {bpText(b)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <span className="lab">色</span>
            <div className="net-seg" role="group" aria-label="点の色">
              {COLOR_MODES.map((m) => (
                <button key={m.id} className={`btn btn-ghost${m.id === mode ? ' on' : ''}`} aria-pressed={m.id === mode}
                        onClick={() => setMode(m.id)} title={m.title}>
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {past && block?.builtAt && (
          <div className="banner info pred-note">
            <span>ℹ</span>
            <div>{`この日の値は、${fmtDateTime(block.builtAt)}に作ったまま固定しています（その日までの日足だけで作っています）`}</div>
          </div>
        )}
        {block?.datasetCheck?.mismatch > 0 && (
          <div className="banner warn pred-note">
            <span>⚠</span>
            <div>{`今日の9列が予測用のデータセットと食い違う候補が ${block.datasetCheck.mismatch} 件あります（${(block.datasetCheck.examples || []).join('・')}）`}</div>
          </div>
        )}

        <div className="banner info pred-note net-note">
          <span>ℹ</span>
          <div>
            <b>参考の表示です。選定には使いません。</b>
            {` 買う銘柄は今日の戦略（${RULE_JA}のうち一番低い百分位が高い順）のまま。`}
            <details className="net-how">
              <summary>作り方と読み方</summary>
              <p>
                {'その銘柄の各営業日（2016-10 からの日足で、9列がそろう日。最初の約250取引日は出来高の250日平均がそろわないので入らない）を点にし、'
                  + 'テクニカル指標9列（実験105 の学習の寄与で上位の列）をその銘柄の中で平均0・標準偏差1にそろえた状態の距離が ε 以下の日どうしを結ぶ'
                  + `（リカレンス・ネットワーク。Marwan ほか 2009・Donner ほか 2010）。${method.theiler}営業日以内の日どうしは指標の窓が重なってほぼ同じ`
                  + '状態になるので結ばない。ε は結べる組のうち辺になる割合（辺の密度）で決める。'}
              </p>
              <p>
                {'点の色は、その日に候補になっていたら、の結果（翌営業日の寄りで買い、'
                  + `+${fmt(method.target, 0)}% の指値か${method.hold}営業日目の終値・目的変数）で、予測日までに分かったものだけ。`
                  + '上の「色」で、翌営業日の寄り（買値）から見た20営業日以内の最高値までの上昇率（高値の最大。10% 以上なら'
                  + ' +10% の指値に届いた）・20営業日目の騰落率（指値なし）・目的変数・時期にも変えられる。'
                  + '今日（白い大きな点）の近傍は、この銘柄の過去で今日と似た状態だった日で、どれも結果が出そろっている。近傍の色が偏っていれば、'
                  + 'この銘柄はこの状態のあとにそうなりやすかった、という読み方になる（件数が少ないので目安）。塊は似た状態が繰り返し現れた時期、'
                  + '外側の殻は似た日の無い珍しい状態の日。黄色の線は直近の日付の順の道筋（今日の状態へどこから入ってきたか）。'
                  + '過去の日は、その日に作った値のまま。'}
              </p>
            </details>
          </div>
        </div>
      </section>

      {dayStatus.kind !== 'ok' ? (
        <div className="card empty">{dayStatus.text}</div>
      ) : (
        <div className="net-main">
          <section className="net-panel net-cpanel">
            <h3>購入条件を満たした銘柄（今日の戦略の順）</h3>
            <div className="net-clist" role="listbox" aria-label="購入条件を満たした銘柄">
              <div className="net-crow head" aria-hidden="true">
                <span />
                <span>銘柄</span>
                <span title={`${RULE_JA}の百分位の最小`}>最小</span>
                <span title="ε の中にある過去の日の数">似た日</span>
                <span title="似た日のうち +10% の指値に届いた割合">+10%</span>
              </div>
              {rows.map((r) => {
                const s = r.entry?.summary;
                return (
                  <button key={r.jqCode} role="option" aria-selected={r.jqCode === jq} disabled={!r.entry}
                          className={`net-crow${r.jqCode === jq ? ' on' : ''}`} onClick={() => setJq(r.jqCode)}
                          title={r.entry ? `${r.name}（${fmtInt(r.entry.n)} 日）` : `作れなかった: ${r.reason || ''}`}>
                    <span className="mark">{r.label}</span>
                    <span className="id">
                      <b className="num">{r.code}</b>
                      {r.pick && <span className="badge green net-pick" title="今日の戦略で買う候補（上位の枠に入った）">買う候補</span>}
                      {' '}<span className="name">{r.name}</span>
                    </span>
                    <span className="num">{r.ruleMin === null ? '—' : fmt(r.ruleMin, 1)}</span>
                    <span className="num">{r.entry ? fmtInt(s?.deg) : '—'}</span>
                    <span className="num">{r.entry ? pct(s?.nbr?.hit) : '作れず'}</span>
                  </button>
                );
              })}
            </div>
            <p className="net-hint">
              {`買う候補 = 今日の戦略で上位${STRATEGY.topK}件の枠に入った銘柄（候補銘柄のニュースのタブと同じ）。`
                + `似た日と +10% は辺の密度 ${bpText(method.rrBp)} のとき。行を選ぶと図が変わります`}
            </p>
          </section>

          <section className="net-stage">
            <div className="net-head">
              <div>
                <strong>{dec && !dec.error ? `${dec.code} ${dec.name}` : (row ? `${row.code} ${row.name}` : '')}</strong>
                {ok && <span className="sub">{`${fmtDate(dec.dates[0])} 〜 ${fmtDate(dec.date)}・${fmtInt(dec.n)} 営業日`}</span>}
              </div>
              <div className="net-controls">
                <div className="net-seg" role="group" aria-label="次元">
                  {[3, 2].map((d) => (
                    <button key={d} className={`btn btn-ghost${d === dim ? ' on' : ''}`} aria-pressed={d === dim}
                            onClick={() => setDim(d)}>{`${d}D`}</button>
                  ))}
                </div>
                <label><input type="checkbox" checked={spin} onChange={(e) => setSpin(e.target.checked)} /> 回転</label>
                <label><input type="checkbox" checked={trail} onChange={(e) => setTrail(e.target.checked)} /> {`直近${TRAIL_DAYS}営業日の軌跡`}</label>
                <div className="net-seg" role="group" aria-label="表示の操作">
                  <button className="btn btn-ghost" onClick={() => setZoomCmd({ kind: 'in', t: Date.now() })} aria-label="拡大">＋</button>
                  <button className="btn btn-ghost" onClick={() => setZoomCmd({ kind: 'out', t: Date.now() })} aria-label="縮小">−</button>
                  <button className="btn btn-ghost" onClick={() => setZoomCmd({ kind: 'reset', t: Date.now() })}>元に戻す</button>
                  <button className="btn btn-ghost" onClick={() => setZoomCmd({ kind: 'reheat', t: Date.now() })}>並べ直す</button>
                  <button className={`btn btn-ghost${showForces ? ' on' : ''}`} aria-pressed={showForces}
                          onClick={() => setShowForces((v) => !v)}>力の調整</button>
                </div>
              </div>
            </div>
            {showForces && (
              <div className="net-forces" role="group" aria-label="力の調整">
                {FORCE_SLIDERS.map((s) => {
                  const v = forces[s.key];
                  const shown = s.sign ? v * s.sign : v;
                  return (
                    <label key={s.key}>
                      <span>{s.label}</span>
                      <input type="range" min={s.min} max={s.max} step={s.step} value={shown}
                             onChange={(e) => setForces((f) => ({ ...f, [s.key]: Number(e.target.value) * (s.sign || 1) }))} />
                      <b className="num">{s.show(v)}</b>
                    </label>
                  );
                })}
                <button className="btn btn-ghost" onClick={() => setForces(DEFAULT_FORCES)}>既定に戻す</button>
                <span className="net-hint">{'中心の力・反発を上げると球に近づき、リンクの力を上げると似た日の塊が締まる（Obsidian のグラフ表示と同じつまみ）'}</span>
              </div>
            )}
            {showGraph ? (
              <>
                <Legend mode={mode} counts={counts} nbrCounts={nbrCounts} nbrN={summary.deg} dec={dec} />
                <RecurrenceGraph
                  dec={dec} net={net} nbrs={nbrs} colors={colors} dim={dim} spin={spin} trail={trail ? TRAIL_DAYS : 0}
                  seedKey={`${dec.jqCode}-${dec.date}-${bp}`} selected={selected} onSelect={setSelected} tipOf={tipOf}
                  zoomCmd={zoomCmd} forces={forces}
                  ariaLabel={`${dec.code} ${dec.name} の ${dec.n} 営業日のリカレンス・ネットワーク。今日に似た過去の日は ${summary.deg} 日。`
                    + '右の「表で見る」に同じ内容があります'}
                />
                <div className="net-foot">
                  {`点 ${fmtInt(dec.n)}・線 ${fmtInt(summary.edges)} 本（結べる組 ${fmtInt(summary.pairs)} の ${bpText(bp)}）・`
                    + `ε = ${fmt(summary.eps, 2)}σ（9列の距離）・線の無い日 ${fmtInt(summary.isolated)}（外側の殻）。`
                    + 'ドラッグで回す（Shift + ドラッグ・2D はずらす）・ホイールで拡大・点に触れると似た日を光らせる・クリックで選ぶ'}
                </div>
              </>
            ) : (
              <div className="net-canvas net-wait">{candStatus.text}</div>
            )}
          </section>

          {showGraph && (
            <aside className="net-side">
              <section className="net-panel">
                <h3>今日の状態</h3>
                <div className="net-tiles">
                  <div className="net-tile">
                    <div className="lab">似た過去の日</div>
                    <div className="v">{`${fmtInt(summary.deg)} 日`}</div>
                    <div className="sub">{`この銘柄の日の ${fmt(summary.degPct, 0)}% 点`}</div>
                  </div>
                  <div className="net-tile">
                    <div className="lab">似た日の +10% 到達</div>
                    <div className="v">{pct(summary.nbr.hit)}</div>
                    <div className="sub">{`全期間 ${pct(summary.base.hit)}`}</div>
                  </div>
                  <div className="net-tile">
                    <div className="lab">似た日の 1.5σ 正例</div>
                    <div className="v">{pct(summary.nbr.l15)}</div>
                    <div className="sub">{`全期間 ${pct(summary.base.l15)}`}</div>
                  </div>
                </div>
                {summary.deg === 0 && (
                  <p className="net-hint">今日の状態に ε の中で似た過去の日はありません（この銘柄では珍しい状態）。辺の密度を上げると近傍が増えます。</p>
                )}
                <div className="net-tbl">
                  <table>
                    <thead>
                      <tr><th>日の集まり</th><th>日数</th><th>+10%</th><th>中央値</th><th>1.5σ</th><th>1.2σ</th></tr>
                    </thead>
                    <tbody>
                      {[
                        ['今日に似た日', summary.nbr], ['うち更新日', summary.nbrFresh],
                        ['全期間', summary.base], ['全期間の更新日', summary.baseFresh],
                      ].map(([k, s]) => (
                        <tr key={k} title={`結果の分かる日 ${fmtInt(s.known)}・目的変数の分かる日 1.5σ ${fmtInt(s.l15N)} / 1.2σ ${fmtInt(s.l12N)}`}>
                          <td>{k}</td><td>{fmtInt(s.n)}</td><td>{pct(s.hit)}</td>
                          <td>{Number.isFinite(s.retMed) ? fmtSigned(s.retMed, 1) : '—'}</td>
                          <td>{s.l15N ? pct(s.l15) : '—'}</td><td>{s.l12N ? pct(s.l12) : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="net-hint">
                  {`似た過去の日 = ε の中の日（次数。中央値 ${fmt(summary.degMed, 0)} 日）。更新日 = 78週高値の更新日（直前20営業日に更新なし。モデルの母集団の日）。`
                    + '+10% は翌営業日の寄りで買って20営業日以内に指値に届いた割合、中央値は収益（届けば +10）。件数が少ないので、割合は目安です'}
                </p>
              </section>

              <section className="net-panel">
                <h3>今日に近い過去の更新日</h3>
                {summary.nearestFresh.length === 0 ? (
                  <p className="net-hint">結果の出そろった過去の更新日がありません。</p>
                ) : (
                  <div className="net-tbl">
                    <table>
                      <thead><tr><th>日付</th><th>距離</th><th>結果</th><th>1.5σ</th><th>1.2σ</th></tr></thead>
                      <tbody>
                        {summary.nearestFresh.map((r) => (
                          <tr key={r.i} className={`pick${r.i === selected ? ' on' : ''}`}
                              onClick={() => setSelected(r.i === selected ? null : r.i)}>
                            <td>{fmtDate(r.date)}{r.inEps ? ' ●' : ''}</td>
                            <td>{`${fmt(r.d, 2)}σ`}</td>
                            <td>{retText(r.ret, method.target)}</td>
                            <td>{labelText(r.l15)}</td>
                            <td>{labelText(r.l12)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <p className="net-hint">{`● = ε（${fmt(summary.eps, 2)}σ）の中。行をクリックすると図の中でその日を選びます`}</p>
              </section>

              <section className="net-panel">
                <h3>{selZ ? `9列の位置（白 = 今日・紫 = ${fmtDate(dec.dates[selected])}）` : '今日の9列'}</h3>
                <div className="net-feats">
                  {feat.map((f, d) => (
                    <div key={f.col} className="net-feat" title={`${f.ja}（${f.col}・${f.group}）`}>
                      <div className="nm">{f.short || f.ja}</div>
                      <div className="val num">{rawText(dec.raw[d])}</div>
                      <div className="bar" aria-hidden="true">
                        <i className="zero" />
                        <i className="m today" style={{ left: `${zPos(todayZ[d])}%` }} />
                        {selZ && <i className="m sel" style={{ left: `${zPos(selZ[d])}%` }} />}
                      </div>
                      <div className="z num">{`${fmtSigned(todayZ[d], 1, '')}σ`}</div>
                    </div>
                  ))}
                </div>
                <p className="net-hint">棒はその銘柄の全期間の中での位置（−3σ〜+3σ。真ん中が平均）。数字は今日の値</p>
              </section>

              {Number.isInteger(selected) && selStats && (
                <section className="net-panel">
                  <h3>{`選んだ日 ${fmtDate(dec.dates[selected])}`}</h3>
                  <dl className="net-dl">
                    <dt>結果</dt><dd>{retText(dec.ret[selected], method.target)}</dd>
                    <dt>目的変数</dt><dd>{`1.5σ ${labelText(dec.labels.l15?.[selected])}・1.2σ ${labelText(dec.labels.l12?.[selected])}`}</dd>
                    <dt>20日以内</dt>
                    <dd>{`最高値まで ${fwdText(dec.rise[selected])}・最安値まで ${fwdText(dec.fall[selected])}・`
                      + `20営業日目 ${fwdText(dec.r20[selected])}`}</dd>
                    <dt>78週高値の更新日</dt><dd>{dec.isFresh[selected] ? 'はい' : 'いいえ'}</dd>
                    <dt>今日との距離</dt><dd>{`${fmt(Math.sqrt(sqDist(dec, selected, dec.n - 1)) / dec.q, 2)}σ`}</dd>
                    <dt>この日に似た日</dt>
                    <dd>{`${fmtInt(selStats.n)} 日・+10% ${pct(selStats.hit)}・1.5σ ${pct(selStats.l15)}（前後の日を含む）`}</dd>
                  </dl>
                  <button className="btn btn-ghost" onClick={() => setSelected(null)}>選ぶのをやめる</button>
                </section>
              )}

              <details className="net-details net-panel">
                <summary>{`表で見る（今日に似た日 ${fmtInt(nearList.length)} 日）`}</summary>
                <div className="net-tbl net-tbl-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>日付</th><th>距離</th><th>結果</th>
                        {valueMode && <th title={valueMode.title}>{valueMode.short}</th>}
                        <th>1.5σ</th><th>1.2σ</th><th>更新日</th>
                      </tr>
                    </thead>
                    <tbody>
                      {nearList.map((r) => (
                        <tr key={r.i}>
                          <td>{fmtDate(dec.dates[r.i])}</td><td>{`${fmt(r.d, 2)}σ`}</td>
                          <td>{retText(dec.ret[r.i], method.target)}</td>
                          {valueMode && <td>{fwdText(dec[valueMode.value][r.i])}</td>}
                          <td>{labelText(dec.labels.l15?.[r.i])}</td><td>{labelText(dec.labels.l12?.[r.i])}</td>
                          <td>{dec.isFresh[r.i] ? '○' : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </aside>
          )}
        </div>
      )}
    </div>
  );
}

/** −3σ〜+3σ を 0〜100% に（外は端に寄せる）。 */
function zPos(z) {
  if (!Number.isFinite(z)) return 50;
  return Math.max(0, Math.min(100, ((z + 3) / 6) * 100));
}

/**
 * 凡例（色の区分と件数・印）。色だけに頼らないよう、図の上に必ず出す。区分のある色の付け方では、今日に似た日の区分ごとの
 * 件数も文字で添える。
 */
function Legend({ mode, counts, nbrCounts, nbrN, dec }) {
  const items = classesOf(mode);
  return (
    <div className="net-legend" aria-label="凡例">
      {items ? items.map((c) => (
        <span key={c.id}>
          <i className="sw" style={{ background: c.color }} aria-hidden="true" />
          {c.label}
          <span className="cnt num">{fmtInt(counts?.[c.id] || 0)}</span>
        </span>
      )) : (
        <span className="net-ramp">
          <span className="num">{fmtDate(dec.dates[0])}</span>
          <i style={{ background: `linear-gradient(90deg, ${TIME_STOPS.join(', ')})` }} aria-hidden="true" />
          <span className="num">{fmtDate(dec.date)}</span>
        </span>
      )}
      <span><i className="mk today" aria-hidden="true" />今日（候補の日）</span>
      <span><i className="mk fresh" aria-hidden="true" />78週高値の更新日</span>
      <span><i className="mk trail" aria-hidden="true" />直近の軌跡</span>
      <span><i className="mk focus" aria-hidden="true" />触れた日の近傍</span>
      {items && nbrCounts && (
        <span className="net-nbrmix">
          {`今日に似た日 ${fmtInt(nbrN)} 日の内訳: `
            + (items.filter((c) => nbrCounts[c.id]).map((c) => `${c.label} ${fmtInt(nbrCounts[c.id])}`).join('・') || 'なし')}
        </span>
      )}
    </div>
  );
}
