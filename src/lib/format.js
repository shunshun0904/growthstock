/** 表示フォーマット用のユーティリティ。値が無い場合は必ず「—」を返す (0 で埋めない)。 */

export const DASH = '—';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function fmt(value, digits = 1, suffix = '') {
  if (!isNum(value)) return DASH;
  return value.toLocaleString('ja-JP', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }) + suffix;
}

/**
 * 実績の値に予想を添える（例「120.0倍（予想 25.0倍）」）。予想が無ければ実績だけ、実績が無ければ「—（予想 …）」。
 * PER・ROE は実績（直近12か月の利益）で、一時的な損益で大きく振れる。証券会社の画面の多くは予想を出すので、
 * 並べて見せる（運用者の選択 2026-11-25「実績と予想を並べる」）。
 */
export function fmtWithForecast(value, forecast, digits = 1, suffix = '') {
  const base = fmt(value, digits, suffix);
  return isNum(forecast) ? `${base}（予想 ${fmt(forecast, digits, suffix)}）` : base;
}

export function fmtInt(value, suffix = '') {
  if (!isNum(value)) return DASH;
  return Math.round(value).toLocaleString('ja-JP') + suffix;
}

export function fmtSigned(value, digits = 1, suffix = '%') {
  if (!isNum(value)) return DASH;
  const sign = value > 0 ? '+' : '';
  return sign + fmt(value, digits, suffix);
}

/** 億円表示。1兆を超える場合は兆単位に切り替える。 */
export function fmtOku(value) {
  if (!isNum(value)) return DASH;
  if (Math.abs(value) >= 10000) return fmt(value / 10000, 2, '兆円');
  if (Math.abs(value) >= 100) return fmtInt(value, '億円');
  return fmt(value, 2, '億円');
}

export function fmtDate(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
}

export function fmtDateTime(iso) {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('ja-JP', { dateStyle: 'medium', timeStyle: 'short' });
}

/** 総合スコアに応じた色 (8.0以上=強気 / 6.0以上=中立 / 未満=弱気) */
export function scoreColor(score) {
  if (!isNum(score)) return 'var(--text-faint)';
  if (score >= 8) return 'var(--violet)';
  if (score >= 6.5) return 'var(--green)';
  if (score >= 5) return 'var(--amber)';
  return 'var(--red)';
}

/** 複数銘柄の重ね描き用パレット (色覚多様性に配慮した順序) */
export const SERIES_COLORS = [
  '#4f8dff', '#34d399', '#fbbf24', '#f472b6',
  '#a78bfa', '#22d3ee', '#fb923c', '#94a3b8',
];
