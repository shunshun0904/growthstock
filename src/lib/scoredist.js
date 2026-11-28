/**
 * スコアと収益の分布（public/data/score_dist.json。Score Distribution が週次の再学習の後に research/score_dist.py で書く）の
 * 組み立て。描画（JSX）から切り離して単体テストする。
 *
 * 運用者の依頼（2026-11-08）「lgbmの+10%モデルでよいので、スコアの分布がみたいです。oof含めてです」「なんなら、
 * ダッシュボードにほしいくらいです」「ランキング学習による直近5営業日版のランキング予測とその収益（20営業日終値、
 * 20営業日内の終値の最高値）の分布もみたいです」「スコア分布の右側の銘柄と…収益分布の右側の銘柄がどれくらい重なっているのか」。
 *
 * ファイルにあるのは集計（度数・分位点・帯ごとの率）だけ。直近の予測のスコアは予測ファイル（predictions.json）の候補から
 * 画面で数える。
 */

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** ラベルの版。固定 +10% のモデルで採点したスコアだけを数える（1.2σ の時代のスコアは物差しが違う）。 */
export const FIXED_VERSION = 'fixed_return_v1';

/**
 * 度数の区間を棒グラフの行にする。edges は区間の境（n+1 個）、series は {名前: 度数（n 個）}。
 * 行は {i, lo, hi, mid, 名前: 度数…}。
 */
export function histRows(edges, series) {
  const e = Array.isArray(edges) ? edges : [];
  const rows = [];
  for (let i = 0; i + 1 < e.length; i += 1) {
    const row = { i, lo: e[i], hi: e[i + 1], mid: (e[i] + e[i + 1]) / 2 };
    for (const [k, v] of Object.entries(series || {})) row[k] = num(Array.isArray(v) ? v[i] : null) ?? 0;
    rows.push(row);
  }
  return rows;
}

/** 度数を合計に対する割合（%）に。合計が0なら0のまま。 */
export function toShares(counts) {
  const c = Array.isArray(counts) ? counts.map((v) => num(v) ?? 0) : [];
  const s = c.reduce((a, b) => a + b, 0);
  return c.map((v) => (s > 0 ? (v / s) * 100 : 0));
}

/** 値を区間に数える。端の外側は最初・最後の区間にまとめる（research/score_dist.hist と同じ）。 */
export function countInto(values, edges) {
  const e = Array.isArray(edges) ? edges : [];
  const out = new Array(Math.max(e.length - 1, 0)).fill(0);
  if (!out.length) return out;
  for (const v of values || []) {
    const x = num(v);
    if (x === null) continue;
    let i = 0;
    while (i < out.length - 1 && x >= e[i + 1]) i += 1;
    out[i] += 1;
  }
  return out;
}

/**
 * 予測ファイルの候補から、固定 +10% の lgbm のスコアを集める（同じ銘柄・同じ日付は1つ）。
 * 戻り値 {scores, dates: [最初, 最後], n}。
 */
export function liveLgbmScores(pred) {
  const seen = new Set();
  const scores = [];
  const dates = [];
  for (const c of pred?.candidates || []) {
    const m = c?.byModel?.lgbm;
    if (!m || m.labelSpec?.version !== FIXED_VERSION) continue;
    const s = num(m.score);
    if (s === null) continue;
    const key = `${c.code}|${c.date}`;
    if (seen.has(key)) continue;
    seen.add(key);
    scores.push(s);
    if (c.date) dates.push(c.date);
  }
  dates.sort();
  return { scores, n: scores.length, dates: dates.length ? [dates[0], dates[dates.length - 1]] : [] };
}

/** 区間の文字（0.50〜0.60、+10〜+12.5% など）。 */
export function rangeLabel(lo, hi, digits = 2, unit = '') {
  const f = (v) => (unit === '%' && v > 0 ? `+${v.toFixed(digits)}` : v.toFixed(digits));
  return `${f(lo)}〜${f(hi)}${unit}`;
}

/** 組の中の10分位の文字（1 → 下位10%、10 → 上位10%）。 */
export function decileLabel(d) {
  const k = num(d);
  if (k === null) return '';
  return `${(k - 1) * 10}〜${k * 10}%`;
}

/** 比（0.123）を % の文字に。 */
export function pct(v, digits = 1, signed = false) {
  const x = num(v);
  if (x === null) return '—';
  const s = (x * 100).toFixed(digits);
  return `${signed && x > 0 ? '+' : ''}${s}%`;
}

/**
 * スコアを out-of-fold の中の百分位（0〜100）に直す。grid は 0〜100% の1%刻みの分位点（101個。research/score_dist.py の
 * pctGrid）。分位点の間は線形に補間し、外側は 0 / 100。同じ値が続く区間は、その区間の真ん中の百分位。
 */
export function percentileOf(score, grid) {
  const x = num(score);
  const g = Array.isArray(grid) ? grid.map(num) : [];
  if (x === null || g.length < 2 || g.some((v) => v === null)) return null;
  const n = g.length - 1;
  if (x <= g[0]) return 0;
  if (x >= g[n]) return 100;
  let lo = 0;
  while (lo < n && g[lo + 1] <= x) lo += 1;          // g[lo] <= x < g[lo+1]
  let first = lo;
  while (first > 0 && g[first - 1] === g[lo]) first -= 1;
  if (first !== lo && x === g[lo]) return ((first + lo) / 2) * (100 / n);
  const span = g[lo + 1] - g[lo];
  return (lo + (span > 0 ? (x - g[lo]) / span : 0)) * (100 / n);
}

/**
 * 組の中の順位（1 = 最上位）を、上からの百分位と10分位（10 = 上位10%）に直す（research/score_dist.pool_rank と同じ式:
 * pp = 1 − (順位 − 1) / 組の行の数、10分位 = ceil(pp × 10)）。
 */
export function poolDecile(rank, nPool) {
  const r = num(rank);
  const n = num(nPool);
  if (r === null || n === null || n <= 0 || r < 1 || r > n) return null;
  const pp = 1 - (r - 1) / n;
  return { pp, dec: Math.min(Math.ceil(pp * 10 - 1e-12), 10), top10: pp > 0.9 };
}

/** スコア帯（bands の lo < score ≤ hi。最初の帯は lo を含む）。 */
export function bandOf(score, bands) {
  const x = num(score);
  if (x === null) return null;
  const b = Array.isArray(bands) ? bands : [];
  return b.find((r, i) => (i === 0 ? x >= r.lo : x > r.lo) && x <= r.hi) || null;
}

/**
 * 予測の候補が分布のどこにいるか（運用者の依頼 2026-11-08「予測を実施したときに、各サンプルが分布のどこに位置するのか、
 * 表示してほしいです」）。date の候補（固定 +10% の lgbm のスコアがあるもの）を、lgbm のスコアの高い順に並べる。
 *   lgbmPct … out-of-fold の中の百分位、band … そのスコア帯の out-of-fold の +10% 到達率など、lgbmTop10 … 上位10% の線以上
 *   ltr5 … 組の中の順位・10分位と、その10分位の out-of-fold の平均収益（無ければ null）
 *   group … 重なりの4つの組み合わせのどれか（ltr5 が無ければ null）
 */
export function candidatePositions(pred, dist, date) {
  const L = dist?.lgbm || {};
  const R = dist?.ltr5 || {};
  const O = dist?.overlap || {};
  const thr = num(O.lgbmThreshold);
  const decs = new Map((R.deciles || []).map((d) => [d.dec, d]));
  const groups = new Map((O.groups || []).map((g) => [g.key, g]));
  const seen = new Set();
  const rows = [];
  for (const c of pred?.candidates || []) {
    if (date && c?.date !== date) continue;
    const m = c?.byModel?.lgbm;
    if (!m || m.labelSpec?.version !== FIXED_VERSION || num(m.score) === null) continue;
    const key = `${c.code}|${c.date}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const score = m.score;
    const lgbmTop10 = thr !== null && score >= thr;
    const pd = c.ltr5 ? poolDecile(c.ltr5.rank, c.ltr5.nPool) : null;
    const ltr5 = pd ? { rank: c.ltr5.rank, nPool: c.ltr5.nPool, ...pd, stats: decs.get(pd.dec) || null } : null;
    let group = null;
    if (ltr5) {
      const k = lgbmTop10 ? (ltr5.top10 ? 'both' : 'lgbmOnly') : (ltr5.top10 ? 'ltr5Only' : 'neither');
      group = groups.get(k) || { key: k };
    }
    rows.push({
      key, code: c.code, name: c.name, date: c.date, score,
      lgbmPct: percentileOf(score, L.pctGrid), band: bandOf(score, L.bands), lgbmTop10, ltr5, group,
    });
  }
  rows.sort((a, b) => b.score - a.score || String(a.code).localeCompare(String(b.code)));
  return rows;
}

/** 予測の候補がある日付（固定 +10% の lgbm のスコアがあるもの。新しい順）。 */
export function candidateDates(pred) {
  const s = new Set();
  for (const c of pred?.candidates || []) {
    if (c?.byModel?.lgbm?.labelSpec?.version === FIXED_VERSION && c.date) s.add(c.date);
  }
  return [...s].sort().reverse();
}
