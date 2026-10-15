/**
 * 候補の比較（似た過去の事例）の画面（src/views/KnnCompareView.jsx）の、純粋な計算。
 *
 * データは predictions.json の knn（research/knn_compare.py が日次予測で書く）。候補ごとに、60日のリターンの形と
 * 出来高の比率が似た過去の事例（1.5σ の3本の最小が85以上で結果が出そろった行から、ワッサーステイン距離の近い50件）の
 * 収益の分布（10〜90% 点・+10% 到達・−10% 割れ）、近傍までの距離、「距離 × 収益の帯」の度数（雲）が付いている。
 *
 * 候補の並び（A・B・C…）と M（今の並べ方の先頭）は今日の戦略と同じ物差し（strategy.js の orderRows: 1.5σ の3本の最小が
 * 高い順・同点は基準モデルのスコア）。Q は近傍の収益の下側25%点が最も高い候補（実験92 の選び方。同点は今の並べ方）。
 * **参考の表示で、選定には使わない**（docs/MODEL_ADOPTION_RULES.md §52〜§57）。
 */
import { orderRows, STRATEGY } from './strategy.js';

/** 表示する候補の範囲。pass = 規則を満たす候補（今日の戦略の候補）、near = 惜しい候補（85〜95）まで、all = 全候補 */
export const KNN_MODES = [
  { id: 'pass', label: `規則を満たす候補（${STRATEGY.agreePct}以上）` },
  { id: 'near', label: `惜しい候補（${STRATEGY.nearLo}〜${STRATEGY.agreePct}）まで` },
  { id: 'all', label: '全候補' },
];

/** 範囲の下限（1.5σ の3本の最小）。all は下限なし。 */
export function modeMin(mode) {
  if (mode === 'pass') return STRATEGY.agreePct;
  if (mode === 'near') return STRATEGY.nearLo;
  return null;
}

/** その日の近傍の集計（無ければ null。集計を始める前に固定した日・古い予測ファイル）。 */
export function knnDay(data, day) {
  return data?.knn?.byDay?.[day] ?? null;
}

/** 候補の印。今の並べ方の順に A, B, …, Z, A2, B2, … */
export function labelOf(i) {
  const L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  return i < 26 ? L[i] : `${L[i % 26]}${Math.floor(i / 26) + 1}`;
}

/**
 * 表示する候補の行。候補（その日の candidates）を今の並べ方に並べ、範囲で絞り、近傍の集計（knn。無ければ null）と
 * 印（label）、pickedM（今の並べ方の先頭）、pickedQ（近傍の25%点の最上位。同点は先頭に近いほう）を付ける。
 */
export function knnRows(candidates, block, mode = 'pass') {
  const est = block?.candidates || {};
  const min = modeMin(mode);
  const kept = orderRows(candidates).filter((c) => min === null || (c.ruleMin !== null && c.ruleMin >= min));
  const rows = kept.map((c, i) => ({
    ...c, label: labelOf(i), knn: est[c.jqCode] ?? null, pickedM: i === 0, pickedQ: false,
  }));
  let q = -1;
  rows.forEach((r, i) => {
    const p = r.knn?.est?.p25;
    if (Number.isFinite(p) && (q < 0 || p > rows[q].knn.est.p25)) q = i;
  });
  if (q >= 0) rows[q].pickedQ = true;
  return rows;
}

/** 範囲の名前（空のときの文言に使う）。 */
export function modeLabel(mode) {
  return KNN_MODES.find((m) => m.id === mode)?.label ?? mode;
}

/**
 * 図を描けるか。kind: ok / none（knn の無い予測ファイル）/ noday（その日の集計が無い）/ unavailable（作れなかった）/
 * pool_small（母集団が足りない）/ empty（候補が無い・範囲に入る候補が無い・集計の付いた候補が無い）。
 */
export function knnStatus(data, block, rows, mode = 'pass') {
  if (!data?.knn) {
    return { kind: 'none', text: 'この予測ファイルには近傍の集計がありません。次の日次予測から出ます。' };
  }
  if (!block) {
    return { kind: 'noday', text: 'この日の近傍の集計はありません（集計を始める前に出した日は、その日に出した値のまま固定しているため）。' };
  }
  if (block.status === 'unavailable') {
    return { kind: 'unavailable', text: `この日は近傍の集計を作れませんでした（${block.reason || '理由は記録されていません'}）。` };
  }
  if (block.status === 'pool_small') {
    return { kind: 'pool_small',
             text: `似た過去の事例の母集団が ${block.poolN ?? 0} 件しかなく（${data.knn.minPool ?? 100} 件未満）、見積もりを出していません。` };
  }
  if (block.status === 'no_candidates') return { kind: 'empty', text: 'この日は候補がありません。' };
  if (!rows.length) {
    return { kind: 'empty',
             text: mode === 'all' ? 'この日は候補がありません。'
               : `${modeLabel(mode)}はありません。「${modeLabel(mode === 'pass' ? 'near' : 'all')}」に広げると、この日の候補が過去のどの辺りにいるかは見られます。` };
  }
  if (!rows.some((r) => r.knn)) {
    return { kind: 'empty', text: '候補に近傍の集計が付いていません（1.5σ の3本のどれかが欠けているか、60日の日足が足りない）。' };
  }
  return { kind: 'ok', text: '' };
}

/* ------------------------------------------------ 図の座標 */

/** 線形の尺度。値 v（d0〜d1）を r0〜r1 に写す。 */
export function lin(d0, d1, r0, r1) {
  const span = d1 - d0;
  return (v) => (span === 0 ? (r0 + r1) / 2 : r0 + ((v - d0) / span) * (r1 - r0));
}

/** 見やすい刻み（1・2・2.5・5 × 10^k）で lo〜hi を n 本くらいに分ける目盛り。 */
export function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const raw = (hi - lo) / Math.max(1, n);
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || 10 * p;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

/**
 * 散布図の軸の範囲。横 = 近傍の25%点（文脈の格子と今日の候補を収める）、縦 = 近傍の −10% 割れの割合（0〜）。
 * 文脈が無ければ候補だけで決める。
 */
export function scatterFrame(rows, context) {
  const xs = rows.filter((r) => r.knn).map((r) => r.knn.est.p25);
  const ys = rows.filter((r) => r.knn).map((r) => r.knn.est.bad);
  const xe = context?.density?.xEdges || [];
  const lo = Math.min(...(xe.length ? [xe[0]] : []), ...(xs.length ? xs : [-10])) - 0.5;
  const hi = Math.max(...(xe.length ? [xe[xe.length - 1]] : []), ...(xs.length ? xs : [10])) + 0.5;
  const ytop = Math.max(16, ...ys.map((v) => v + 3));
  return { lo, hi, ytop };
}

/** 雲（候補ごとの 距離 × 収益の帯 の格子）の並べ方。1行に何枚入るか。 */
export function cloudLayout(n, { width = 560, left = 70, cell = 20, gap = 26, bins = 7 } = {}) {
  const panelW = bins * cell;
  const perRow = Math.max(1, Math.min(Math.max(1, n), Math.floor((width - left) / (panelW + gap))));
  return { perRow, rows: Math.ceil(Math.max(1, n) / perRow), panelW, cell, gap, left };
}

/* ------------------------------------------------ 表（図と同じ数字。色に頼らず読めるように） */

const num = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');

export function scatterTable(rows) {
  return rows.map((r) => [
    `${r.label}${r.pickedQ ? '（Q）' : ''}${r.pickedM ? '（M）' : ''}`, r.code, r.name || '',
    r.ruleMin === null ? '—' : r.ruleMin.toFixed(1),
    num(r.knn?.est?.p25), num(r.knn?.est?.p50),
    r.knn ? `${r.knn.est.hit}%` : '—', r.knn ? `${r.knn.est.bad}%` : '—',
    r.knn ? `${num(r.knn.dNn.R)} / ${num(r.knn.dNn.V)}` : '—',
  ]);
}

export function stripTable(rows) {
  return rows.map((r) => [
    r.label, r.code, num(r.knn?.est?.p10), num(r.knn?.est?.p25), num(r.knn?.est?.p50), num(r.knn?.est?.p75),
    num(r.knn?.est?.p90), r.knn ? `${r.knn.est.hit}%` : '—', r.knn ? `${r.knn.est.bad}%` : '—',
  ]);
}

/** 候補 × 帯（上の帯から）の行。列は距離の刻み。 */
export function cloudTable(rows, bands) {
  const out = [];
  for (const r of rows) {
    if (!r.knn) continue;
    for (let j = bands.length - 1; j >= 0; j--) {
      out.push([`${r.label}・${bands[j]}`, ...r.knn.cloud.map((col) => col[j])]);
    }
  }
  return out;
}
