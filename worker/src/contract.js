/** Public-data-only request contract, shared with the static dashboard. No secrets. */
export const LIMITS = Object.freeze({
  questionChars: 1200, contextBytes: 16 * 1024, requestBytes: 24 * 1024,
  responseBytes: 64 * 1024, answerChars: 20000,
});
export const METRICS = Object.freeze({
  price: { label: '終値', unit: '円', group: 'price' },
  ret20d: { label: '直近20営業日騰落率', unit: '%', group: 'price' },
  vol20d: { label: '20日変動率', unit: '%', group: 'price' },
  tradingValue: { label: '売買代金', unit: '億円', group: 'price' },
  marketCap: { label: '時価総額', unit: '億円', group: 'price' },
  high52w: { label: '52週高値', unit: '円', group: 'price' },
  volumeTrend: { label: '出来高の20日平均比', unit: '%', group: 'price' },
  roe: { label: 'ROE', unit: '%', group: 'financial' },
  per: { label: 'PER', unit: '倍', group: 'financial' },
  pbr: { label: 'PBR', unit: '倍', group: 'financial' },
  divYield: { label: '配当利回り', unit: '%', group: 'financial' },
  epsGrowth: { label: 'EPS成長率', unit: '%', group: 'financial' },
  salesGrowth: { label: '売上成長率', unit: '%', group: 'financial' },
  opMargin: { label: '営業利益率', unit: '%', group: 'financial' },
  progressRate: { label: '会社予想に対する進捗率', unit: '%', group: 'financial' },
  creditRatio: { label: '信用倍率', unit: '倍', group: 'risk' },
  nextEarnDays: { label: '次回決算までの暦日数', unit: '日', group: 'risk' },
});
export const CLASSIFIERS = ['lgbm', 'xgb', 'cat', 'lgbm_k15', 'xgb_k15', 'cat_k15'];
export class InputError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; this.code = 'INVALID_INPUT'; }
}
const bad = () => { throw new InputError('質問または銘柄データの形式が不正です。'); };
const object = (v, keys) => {
  if (!v || typeof v !== 'object' || Array.isArray(v) || Object.keys(v).some(k => !keys.includes(k))) bad();
};
const text = (v, max, min = 1) => {
  if (typeof v !== 'string' || v.trim().length < min || v.length > max || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(v)) bad();
};
const numeric = (v, lo = -1e15, hi = 1e15, integer = false) => {
  if (!Number.isFinite(v) || v < lo || v > hi || (integer && !Number.isInteger(v))) bad();
};
const date = v => {
  if (typeof v !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(v) || !Number.isFinite(Date.parse(v))
      || new Date(v).toISOString().slice(0, 10) !== v) bad();
};
function metrics(items) {
  if (!Array.isArray(items) || items.length > Object.keys(METRICS).length) bad();
  const seen = new Set();
  for (const m of items) {
    object(m, ['key', 'value', 'asOf', 'source', 'basis', 'disclosedDate']);
    if (!Object.hasOwn(METRICS, m.key) || seen.has(m.key)) bad();
    seen.add(m.key); numeric(m.value); date(m.asOf);
    if (!['stocks.json', 'predictions.json'].includes(m.source)) bad();
    if (m.basis !== undefined) text(m.basis, 160);
    if (m.disclosedDate !== undefined) date(m.disclosedDate);
  }
}
function target(t) {
  object(t, ['kind', 'threshold', 'endRatio', 'horizon', 'endWindow', 'entry', 'requireUptrend', 'trendShort', 'trendLong']);
  if (t.kind !== 'fixed_return' || t.entry !== 'next_open') bad();
  numeric(t.threshold, 0.001, 1); numeric(t.horizon, 1, 252, true);
  if (t.endRatio !== undefined) numeric(t.endRatio, 0, 1);
  for (const k of ['endWindow', 'trendShort', 'trendLong']) if (t[k] !== undefined) numeric(t[k], 1, 252, true);
  if (t.requireUptrend !== undefined && typeof t.requireUptrend !== 'boolean') bad();
}
function ranking(r) {
  object(r, ['asOf', 'score', 'rank', 'poolSize', 'percentile']); date(r.asOf); numeric(r.score);
  if (r.rank !== undefined) numeric(r.rank, 1, 100000, true);
  if (r.poolSize !== undefined) numeric(r.poolSize, 1, 100000, true);
  if (r.percentile !== undefined) numeric(r.percentile, 0, 100);
  if (r.rank > r.poolSize) bad();
}
export function validatePayload(payload) {
  object(payload, ['question', 'context']); text(payload.question, LIMITS.questionChars);
  const c = payload.context;
  object(c, ['stockCode', 'stockName', 'asOf', 'generatedAt', 'metrics', 'prediction', 'explanation', 'historicalSnapshots', 'limitations']);
  text(c.stockCode, 4, 4); if (!/^[0-9A-Z]{4}$/.test(c.stockCode)) bad();
  text(c.stockName, 100); date(c.asOf); metrics(c.metrics);
  if (c.generatedAt !== undefined) {
    object(c.generatedAt, ['stocks', 'predictions']);
    for (const value of Object.values(c.generatedAt)) { text(value, 40); if (!Number.isFinite(Date.parse(value))) bad(); }
  }
  if (c.prediction !== undefined) {
    const p = c.prediction; object(p, ['asOf', 'baseline', 'models', 'ltr', 'ltr5']); date(p.asOf);
    if (p.baseline !== undefined) {
      const b = p.baseline; object(b, ['score', 'rank', 'poolSize', 'percentile']); numeric(b.score, 0, 1);
      if (b.rank !== undefined) numeric(b.rank, 1, 100000, true);
      if (b.poolSize !== undefined) numeric(b.poolSize, 1, 100000, true);
      if (b.percentile !== undefined) numeric(b.percentile, 0, 100);
    }
    if (!Array.isArray(p.models) || p.models.length > 6) bad();
    const seen = new Set();
    for (const m of p.models) {
      object(m, ['model', 'score', 'percentile', 'target']);
      if (!CLASSIFIERS.includes(m.model) || seen.has(m.model)) bad();
      seen.add(m.model); numeric(m.score, 0, 1);
      if (m.percentile !== undefined) numeric(m.percentile, 0, 100);
      if (m.target !== undefined) target(m.target);
    }
    for (const k of ['ltr', 'ltr5']) if (p[k] !== undefined) ranking(p[k]);
  }
  if (c.explanation !== undefined) {
    const e = c.explanation; object(e, ['model', 'scale', 'asOf', 'top']); date(e.asOf);
    if (e.model !== 'lgbm' || e.scale !== 'log_odds' || !Array.isArray(e.top) || e.top.length > 8) bad();
    for (const row of e.top) {
      object(row, ['feature', 'name', 'contribution']);
      if (!/^[A-Za-z][A-Za-z0-9_]{0,79}$/.test(row.feature)) bad();
      if (row.name !== undefined) text(row.name, 100);
      numeric(row.contribution);
    }
  }
  if (c.historicalSnapshots !== undefined) {
    if (!Array.isArray(c.historicalSnapshots) || c.historicalSnapshots.length > 2) bad();
    for (const s of c.historicalSnapshots) { object(s, ['asOf', 'metrics']); date(s.asOf); metrics(s.metrics); }
  }
  if (!Array.isArray(c.limitations) || c.limitations.length > 8) bad();
  for (const s of c.limitations) text(s, 200);
  const predicted = c.prediction && (c.prediction.baseline || c.prediction.models.length || c.prediction.ltr || c.prediction.ltr5);
  if (!c.metrics.length && !predicted) throw new InputError('対象銘柄の公開データが不足しています。');
  if (new TextEncoder().encode(JSON.stringify(c)).byteLength > LIMITS.contextBytes)
    throw new InputError('送信する銘柄データが大きすぎます。', 413);
  return { question: payload.question.trim(), context: c };
}
