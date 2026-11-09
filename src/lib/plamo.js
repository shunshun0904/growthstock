import { CLASSIFIERS, LIMITS, METRICS, validatePayload } from '../../worker/src/contract.js';
export { LIMITS, METRICS };

export const PRESET_QUESTIONS = [
  'この銘柄の財務的な特徴を説明してください',
  'ROEとPBRから企業評価を考察してください',
  '予測モデルで高順位になった理由を説明してください',
  'この銘柄の投資リスクを整理してください',
  '過去の類似局面と比較してください',
];
const finite = v => typeof v === 'number' && Number.isFinite(v);
const dateOf = v => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v) && Number.isFinite(Date.parse(v)) ? v : null;
export function aiStockCode(value) {
  const code = String(value || '').replace(/^(jq-|pred:)/, '').toUpperCase();
  return /^[0-9A-Z]{4}0$/.test(code) ? code.slice(0, -1) : /^[0-9A-Z]{4}$/.test(code) ? code : null;
}
function entries(dataset, predictionData) {
  const found = new Map();
  for (const stock of dataset?.stocks || []) {
    const code = aiStockCode(stock.code || stock.jqCode);
    if (!code || stock.origin === 'manual' || !stock.name) continue;
    found.set(code, { stockCode: code, name: String(stock.name).slice(0, 100), stock });
  }
  for (const candidate of [...(predictionData?.candidates || [])].sort((a, b) => String(b.date).localeCompare(String(a.date)))) {
    const code = aiStockCode(candidate.code || candidate.jqCode);
    if (!code || !candidate.name || !dateOf(candidate.date)) continue;
    const item = found.get(code) || { stockCode: code, name: String(candidate.name).slice(0, 100) };
    if (!item.candidate) item.candidate = candidate;
    found.set(code, item);
  }
  return [...found.values()].sort((a, b) => a.stockCode.localeCompare(b.stockCode));
}
export function publicAiStocks(dataset, predictionData) {
  return entries(dataset, predictionData).map(({ stockCode, name }) => ({ stockCode, name }));
}
function scope(question) {
  const financial = /財務|ROE|ROIC|PBR|PER|CCC|利益|売上|配当|企業評価|決算|成長率/i.test(question);
  const prediction = /予測|モデル|順位|ランキング|スコア|SHAP|特徴量|寄与|理由/i.test(question);
  const risk = /リスク|危険|下落|注意/i.test(question);
  const history = /過去|比較|推移|類似|以前/.test(question);
  const general = !financial && !prediction && !risk && !history;
  return { financial: financial || risk || general, prediction: prediction || risk || general, history, risk: risk || general };
}
function selectMetrics(stock, candidate, keys) {
  const fromStock = stock?.metrics || {};
  const stockDate = dateOf(fromStock.date) || dateOf(stock?.asOf);
  const predDate = dateOf(candidate?.date);
  const observations = [];
  for (const key of keys) {
    const stockKey = key;
    const candidateKey = key === 'price' ? 'close' : key;
    const choices = [];
    if (stockDate && finite(fromStock[stockKey])) choices.push({ value: fromStock[stockKey], asOf: stockDate, source: 'stocks.json',
      ...(METRICS[key].group === 'financial' && dateOf(fromStock.disclosedDate) ? { disclosedDate: fromStock.disclosedDate } : {}),
      ...(typeof fromStock[`${key}Basis`] === 'string' ? { basis: fromStock[`${key}Basis`].slice(0, 160) } : {}) });
    if (predDate && finite(candidate?.[candidateKey])) choices.push({ value: candidate[candidateKey], asOf: predDate, source: 'predictions.json' });
    // Different dates/sources are retained per metric; never present a mixed row as one observation date.
    choices.sort((a, b) => b.asOf.localeCompare(a.asOf));
    if (choices.length) observations.push({ key, ...choices[0] });
  }
  return observations;
}
function definedNumbers(source, mapping) {
  return Object.fromEntries(Object.entries(mapping).filter(([, key]) => finite(source?.[key]))
    .map(([to, from]) => [to, source[from]]));
}
function targetOf(spec) {
  if (spec?.kind !== 'fixed_return' || !finite(spec.threshold) || !finite(spec.horizon) || spec.entry !== 'next_open') return undefined;
  return { kind: 'fixed_return', threshold: spec.threshold, horizon: spec.horizon, entry: spec.entry,
    ...definedNumbers(spec, { endRatio: 'end_ratio', endWindow: 'end_window', trendShort: 'trend_short', trendLong: 'trend_long' }),
    ...(typeof spec.require_uptrend === 'boolean' ? { requireUptrend: spec.require_uptrend } : {}) };
}
export function buildAiContext({ dataset, predictionData, stockCode, question = '', featureNames = {} }) {
  const item = entries(dataset, predictionData).find(s => s.stockCode === aiStockCode(stockCode));
  if (!item) throw new Error('対象銘柄の公開データがありません。手入力データは送信対象に含めていません。');
  const { stock, candidate: c } = item, requested = scope(question);
  const keys = Object.keys(METRICS).filter(k => METRICS[k].group === 'price'
    || (METRICS[k].group === 'financial' && requested.financial) || (METRICS[k].group === 'risk' && requested.risk));
  const metrics = selectMetrics(stock, c, keys);
  const asOf = [dateOf(stock?.asOf), dateOf(stock?.metrics?.date), dateOf(c?.date)].filter(Boolean).sort().at(-1);
  if (!asOf) throw new Error('公開データの基準日を確認できません。');
  const context = { stockCode: item.stockCode, stockName: item.name, asOf, metrics, limitations: [
    '取得済みの公開データだけです。リアルタイム株価・最新ニュースにはアクセスしていません。',
  ] };
  const generatedAt = {};
  if (stock && Number.isFinite(Date.parse(dataset?.generatedAt))) generatedAt.stocks = dataset.generatedAt;
  const predGenerated = predictionData?.dayInfo?.[c?.date]?.generatedAt
    || (c?.date === predictionData?.asOf ? predictionData?.generatedAt : undefined);
  if (c && Number.isFinite(Date.parse(predGenerated))) generatedAt.predictions = predGenerated;
  if (Object.keys(generatedAt).length) context.generatedAt = generatedAt;
  if (requested.financial) {
    const missing = ['roe', 'per', 'pbr'].filter(k => !metrics.some(m => m.key === k));
    if (missing.length) context.limitations.push(`取得できていない指標: ${missing.join(', ')}。代わりの数値を補わないでください。`);
  }
  if (requested.prediction && c) {
    const prediction = { asOf: c.date, models: [] };
    if (finite(c.score)) prediction.baseline = definedNumbers(c, { score: 'score', rank: 'rankInDay', poolSize: 'nInDay', percentile: 'pctHistorical' });
    for (const model of CLASSIFIERS) {
      const m = c.byModel?.[model];
      if (!finite(m?.score)) continue;
      const target = targetOf(m.labelSpec);
      prediction.models.push({ model, ...definedNumbers(m, { score: 'score', percentile: 'pctHistorical' }), ...(target ? { target } : {}) });
    }
    if (finite(c.ltr?.score)) prediction.ltr = { asOf: c.date, ...definedNumbers(c.ltr, { score: 'score', rank: 'dayRank', poolSize: 'nDay', percentile: 'dayPct' }) };
    if (finite(c.ltr5?.score) && dateOf(predictionData?.ltr5?.day)) prediction.ltr5 = { asOf: predictionData.ltr5.day,
      ...definedNumbers(c.ltr5, { score: 'score', rank: 'rank', poolSize: 'nPool', percentile: 'pct' }) };
    else if (finite(c.ltr5?.score)) context.limitations.push('LTR5の評価日が未取得のため、そのスコア・順位は送信していません。');
    context.prediction = prediction;
    const top = (c.contrib?.top || []).filter(t => /^[A-Za-z][A-Za-z0-9_]{0,79}$/.test(t.col) && finite(t.contrib)).slice(0, 8)
      .map(t => ({ feature: t.col, contribution: t.contrib,
        name: String(featureNames[t.col]?.name || t.name || t.col).slice(0, 100) }));
    if (top.length) context.explanation = { model: 'lgbm', scale: 'log_odds', asOf: c.date, top };
    context.limitations.push('SHAP寄与は基準LightGBM分類器の一部だけです。ランキング学習モデルの個別SHAPと生の特徴量値は送信データにありません。');
    if (!prediction.models.some(m => m.target)) context.limitations.push('この予測時点のモデル別目標定義は未取得です。現在の目標を当てはめないでください。');
  } else if (requested.prediction) context.limitations.push('この銘柄の予測順位・スコア・SHAPを取得できていません。');
  if (requested.history) {
    const snapshots = Object.entries(stock?.snapshots || {}).filter(([key]) => ['m3', 'm6'].includes(key));
    context.historicalSnapshots = snapshots.filter(([, m]) => dateOf(m.date)).slice(0, 2).map(([, m]) => ({
      asOf: m.date, metrics: selectMetrics({ metrics: m }, null, ['price', 'roe', 'opMargin', 'salesGrowth', 'epsGrowth', 'creditRatio']),
    }));
    context.limitations.push('類似局面を検索した結果は送信していません。過去スナップショットは類似事例の抽出結果ではありません。');
    if (!context.historicalSnapshots.length) context.limitations.push('この銘柄の過去スナップショットは取得できていません。');
  }
  validatePayload({ question: question || PRESET_QUESTIONS[0], context });
  return context;
}

export function workerUrl(value = import.meta.env?.VITE_PLAMO_WORKER_URL || '') {
  if (!value) throw new Error('AI接続先のWorker URLが未設定です。既存の分析タブはそのまま利用できます。');
  try {
    const url = new URL(value.trim());
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if ((url.protocol !== 'https:' && !(local && url.protocol === 'http:')) || url.username || url.password
        || url.search || url.hash || url.pathname !== '/') throw new Error();
    return url.origin;
  } catch { throw new Error('Worker URLにはHTTPSのルートURLを指定してください。'); }
}
const API_ERRORS = {
  UNAUTHORIZED: '利用パスワードが正しくありません。',
  AUTH_RATE_LIMIT: '認証を含むアクセス回数の上限に達しました。時間をおいてください。',
  RATE_LIMIT: '短時間の利用上限に達しました。時間をおいてください。',
  DAILY_LIMIT: '本日のAI利用上限に達しました（UTC基準）。',
  MONTHLY_LIMIT: '今月のAI利用上限に達しました（UTC基準）。',
  BUSY: '別の分析が実行中です。完了後に送信してください。',
  NOT_CONFIGURED: 'WorkerのAPIキーまたは無料βの利用設定が未完了です。',
  UPSTREAM_LIMIT: 'PLaMo APIの利用上限に達しました。有料プランへの切り替えは行いません。',
  UPSTREAM_AUTH: 'Workerに設定したPLaMo APIキーを確認してください。',
  MODEL_UNAVAILABLE: '指定した無料βモデルを利用できません。モデルと無料プランを確認してください。',
  MODEL_MISMATCH: '指定した無料βモデルと異なる応答のため、回答を表示できません。',
  TIMEOUT: 'AIの回答がタイムアウトしました。質問を絞って再度お試しください。',
  EMPTY_ANSWER: 'AIから回答本文を取得できませんでした。質問を絞って再度お試しください。',
  ORIGIN_DENIED: 'このサイトからの接続がWorkerで許可されていません。',
};
export async function requestAnalysis({ workerUrl: configured, password, question, context, signal,
  fetchImpl = globalThis.fetch, timeoutMs = 40000 }) {
  const base = workerUrl(configured); const payload = validatePayload({ question, context });
  if (typeof password !== 'string' || !/^[\x21-\x7e]{32,256}$/.test(password)) throw new Error('32文字以上の利用パスワードを入力してください。');
  const controller = new AbortController(); let timer, timedOut = false;
  const abort = () => controller.abort();
  if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
  timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetchImpl(`${base}/chat`, { method: 'POST', credentials: 'omit', redirect: 'error',
      cache: 'no-store', referrerPolicy: 'no-referrer', signal: controller.signal,
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${password}` }, body: JSON.stringify(payload) });
    let data; try { data = await response.json(); } catch { throw new Error('AIの応答を読み込めませんでした。'); }
    if (!response.ok) {
      const code = data?.error?.code;
      const message = API_ERRORS[code] || (response.status === 429 ? API_ERRORS.UPSTREAM_LIMIT
        : response.status === 504 ? API_ERRORS.TIMEOUT : response.status === 401 ? API_ERRORS.UNAUTHORIZED
          : 'AIサービスと通信できませんでした。時間をおいて手動で再送してください。');
      throw new Error(message);
    }
    if (typeof data?.answer !== 'string' || !data.answer.trim() || data.answer.length > LIMITS.answerChars)
      throw new Error(API_ERRORS.EMPTY_ANSWER);
    if (typeof data.model !== 'string' || !/^plamo-[0-9.]+-fin-beta$/.test(data.model)) throw new Error(API_ERRORS.MODEL_MISMATCH);
    return data;
  } catch (e) {
    if (timedOut) throw new Error(API_ERRORS.TIMEOUT);
    if (controller.signal.aborted) throw new DOMException('分析を中断しました。', 'AbortError');
    if (e instanceof TypeError) throw new Error('Workerに接続できません。接続先URLとCORS設定を確認してください。');
    throw e;
  } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
}
