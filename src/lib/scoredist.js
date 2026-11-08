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
