import { LIMITS, InputError, validatePayload } from './contract.js';
import { integerSetting } from './limiter.js';
import { messagesFor } from './prompt.js';
export { AiLimiter } from './limiter.js';

const DEFAULT_ENDPOINT = 'https://api.platform.preferredai.jp/v1/chat/completions';
const DEFAULT_MODEL = 'plamo-3.0-fin-beta';
const ERRORS = {
  UNAUTHORIZED: '本人認証に失敗しました。利用パスワードを確認してください。',
  AUTH_RATE_LIMIT: '認証を含むアクセス回数が上限に達しました。時間をおいてください。',
  RATE_LIMIT: '短時間の利用上限に達しました。時間をおいてください。',
  DAILY_LIMIT: '本日のAI利用上限に達しました（UTC基準）。',
  MONTHLY_LIMIT: '今月のAI利用上限に達しました（UTC基準）。',
  BUSY: '別の分析が実行中です。完了後に送信してください。',
  NOT_CONFIGURED: 'AIの設定が完了していません。管理者がWorkerのSecretsと無料β設定を確認してください。',
  UPSTREAM_LIMIT: 'PLaMo APIの利用上限に達しました。有料プランへの切り替えは行いません。',
  UPSTREAM_AUTH: 'PLaMo APIの認証設定を確認してください。',
  MODEL_UNAVAILABLE: '指定した無料βモデルを利用できません。モデルとプランの設定を確認してください。',
  MODEL_MISMATCH: '指定した無料βモデルと異なる応答のため、回答を表示できません。',
  UPSTREAM_UNAVAILABLE: 'PLaMo APIに接続できません。時間をおいて手動で再送してください。',
  TIMEOUT: 'AIの回答が制限時間内に完了しませんでした。質問を絞って再度お試しください。',
  EMPTY_ANSWER: 'AIから回答本文を取得できませんでした。質問を絞って再度お試しください。',
  SERVICE_UNAVAILABLE: 'AI分析を現在利用できません。時間をおいてください。',
};
class ServiceError extends Error {
  constructor(code, status = 503, retryAfter) { super(ERRORS[code]); this.code = code; this.status = status; this.retryAfter = retryAfter; }
}
function allowedOrigin(request, env) {
  const origin = request.headers.get('Origin');
  const allowed = String(env.ALLOWED_ORIGINS || '').split(',').map(s => s.trim());
  if (!origin || !allowed.includes(origin)) return null;
  try {
    const url = new URL(origin);
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    return url.origin === origin && (url.protocol === 'https:' || (local && url.protocol === 'http:')) ? origin : null;
  } catch { return null; }
}
function json(data, status, origin, retryAfter) {
  const headers = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', 'Vary': 'Origin' };
  if (origin) headers['Access-Control-Allow-Origin'] = origin;
  if (retryAfter) headers['Retry-After'] = String(retryAfter);
  return new Response(JSON.stringify(data), { status, headers });
}
async function digest(value) {
  return new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)));
}
async function authenticated(request, password) {
  const header = request.headers.get('Authorization') || '';
  const input = header.startsWith('Bearer ') && header.length <= 263 ? header.slice(7) : '';
  const [a, b] = await Promise.all([digest(input), digest(password)]);
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ b[i];
  return difference === 0 && input.length >= 32;
}
async function limit(env, input) {
  if (!env.AI_LIMITER) throw new ServiceError('NOT_CONFIGURED');
  const stub = env.AI_LIMITER.get(env.AI_LIMITER.idFromName('personal-ai-v1'));
  const response = await stub.fetch('https://limiter.internal/', { method: 'POST', body: JSON.stringify(input) });
  if (!response.ok) throw new ServiceError('SERVICE_UNAVAILABLE');
  const result = await response.json();
  if (!result.allowed) throw new ServiceError(ERRORS[result.code] ? result.code : 'SERVICE_UNAVAILABLE', result.status || 503, result.retryAfter);
  return result;
}
export async function readBounded(message, maximum, signal) {
  const length = message.headers.get('Content-Length');
  if (length && (!/^\d+$/.test(length) || Number(length) > maximum)) throw new InputError('データサイズが上限を超えています。', 413);
  if (!message.body) return '';
  const reader = message.body.getReader(); let total = 0; const chunks = [];
  const cancel = () => { reader.cancel().catch(() => {}); };
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    while (true) {
      if (signal?.aborted) throw new ServiceError('TIMEOUT', 504);
      const { value, done } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximum) { cancel(); throw new InputError('データサイズが上限を超えています。', 413); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(total); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } finally { signal?.removeEventListener('abort', cancel); reader.releaseLock(); }
}
async function deadline(fn, milliseconds) {
  const controller = new AbortController(); let timer;
  try {
    return await Promise.race([fn(controller.signal), new Promise((_, reject) => {
      timer = setTimeout(() => { controller.abort(); reject(new ServiceError('TIMEOUT', 504)); }, milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}
function providerConfig(env) {
  try {
    const endpoint = new URL(env.PLAMO_ENDPOINT || DEFAULT_ENDPOINT);
    const model = env.PLAMO_MODEL || DEFAULT_MODEL;
    if (endpoint.protocol !== 'https:' || endpoint.hostname !== 'api.platform.preferredai.jp'
        || endpoint.port || endpoint.username || endpoint.password || endpoint.search || endpoint.hash
        || !/^\/v[1-9][0-9]*\/chat\/completions$/.test(endpoint.pathname) || !/^plamo-[0-9.]+-fin-beta$/.test(model)
        || !env.PLAMO_API_KEY || env.PLAMO_FREE_BETA_CONFIRMED !== 'true') throw new Error();
    return { endpoint: endpoint.href, model, maxTokens: integerSetting(env, 'MAX_OUTPUT_TOKENS', 1536, 2048),
      timeout: integerSetting(env, 'UPSTREAM_TIMEOUT_MS', 25000, 30000) };
  } catch { throw new ServiceError('NOT_CONFIGURED'); }
}
function retrySeconds(value) {
  const seconds = /^\d+$/.test(value || '') ? Number(value) : (Date.parse(value) - Date.now()) / 1000;
  return Number.isFinite(seconds) ? Math.min(86400, Math.max(1, Math.ceil(seconds))) : 60;
}
async function askProvider(payload, env, config) {
  return deadline(async signal => {
    let response;
    try {
      // Workers supports manual/follow, not the browser's redirect:error.
      // Manual keeps the API credential on the configured official endpoint.
      response = await fetch(config.endpoint, { method: 'POST', redirect: 'manual', signal,
        headers: { 'Authorization': `Bearer ${env.PLAMO_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: config.model, messages: messagesFor(payload),
          max_tokens: config.maxTokens, stream: false }) });
    } catch { throw new ServiceError(signal.aborted ? 'TIMEOUT' : 'UPSTREAM_UNAVAILABLE', signal.aborted ? 504 : 502); }
    if (!response.ok) {
      response.body?.cancel().catch(() => {});
      if (response.status === 429) throw new ServiceError('UPSTREAM_LIMIT', 429, retrySeconds(response.headers.get('Retry-After')));
      if (response.status === 401) throw new ServiceError('UPSTREAM_AUTH');
      if ([400, 402, 403, 404].includes(response.status)) throw new ServiceError('MODEL_UNAVAILABLE');
      throw new ServiceError('UPSTREAM_UNAVAILABLE', 502);
    }
    let data;
    try { data = JSON.parse(await readBounded(response, LIMITS.responseBytes, signal)); }
    catch (e) { if (signal.aborted) throw new ServiceError('TIMEOUT', 504); throw new ServiceError('UPSTREAM_UNAVAILABLE', 502); }
    if (data.model !== config.model) throw new ServiceError('MODEL_MISMATCH', 502);
    const answer = data.choices?.[0]?.message?.content;
    if (typeof answer !== 'string' || !answer.trim() || answer.length > LIMITS.answerChars) throw new ServiceError('EMPTY_ANSWER', 502);
    return { answer: answer.trim(), model: config.model, truncated: data.choices[0].finish_reason === 'length' };
  }, config.timeout);
}
export default {
  async fetch(request, env) {
    const origin = allowedOrigin(request, env);
    if (!origin) return json({ error: { code: 'ORIGIN_DENIED', message: 'このアクセス元は許可されていません。' } }, 403, null);
    const path = new URL(request.url).pathname;
    if (!['/chat', '/usage'].includes(path)) return json({ error: { code: 'NOT_FOUND', message: 'APIが見つかりません。' } }, 404, origin);
    if (request.method === 'OPTIONS') {
      const headers = (request.headers.get('Access-Control-Request-Headers') || '').toLowerCase().split(',').map(h => h.trim()).filter(Boolean);
      if (request.headers.get('Access-Control-Request-Method') !== (path === '/chat' ? 'POST' : 'GET')
          || headers.some(h => !['authorization', 'content-type'].includes(h)))
        return json({ error: { code: 'ORIGIN_DENIED', message: '許可されていない通信です。' } }, 403, origin);
      return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': path === '/chat' ? 'POST' : 'GET',
        'Access-Control-Allow-Headers': 'Authorization, Content-Type', 'Access-Control-Max-Age': '600',
        'Vary': 'Origin', 'Cache-Control': 'no-store' } });
    }
    if (request.method !== (path === '/chat' ? 'POST' : 'GET')) return json({ error: { code: 'METHOD_NOT_ALLOWED', message: '未対応のHTTPメソッドです。' } }, 405, origin);
    let reservation;
    try {
      if (!/^[\x21-\x7e]{32,256}$/.test(env.AI_PASSWORD || '')) throw new ServiceError('NOT_CONFIGURED');
      const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
      const fingerprint = [...await digest(`attempt:${ip}`)].map(n => n.toString(16).padStart(2, '0')).join('');
      await limit(env, { action: 'attempt', fingerprint });
      if (!await authenticated(request, env.AI_PASSWORD)) throw new ServiceError('UNAUTHORIZED', 401);
      if (path === '/usage') {
        const state = await limit(env, { action: 'status' });
        let ready = false; try { providerConfig(env); ready = true; } catch { /* setup status only */ }
        return json({ ...state.usage, configured: ready, model: env.PLAMO_MODEL || DEFAULT_MODEL }, 200, origin);
      }
      const config = providerConfig(env);
      if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json') throw new InputError('JSON形式で送信してください。', 415);
      let raw;
      try { raw = JSON.parse(await deadline(signal => readBounded(request, LIMITS.requestBytes, signal), 5000)); }
      catch (e) { if (e instanceof InputError) throw e; throw new InputError('質問データを読み込めませんでした。', e.code === 'TIMEOUT' ? 408 : 400); }
      const payload = validatePayload(raw);
      const id = crypto.randomUUID();
      const state = await limit(env, { action: 'reserve', id }); reservation = id;
      const result = await askProvider(payload, env, config);
      return json({ ...result, usage: state.usage }, 200, origin);
    } catch (e) {
      const known = e instanceof InputError || e instanceof ServiceError;
      const code = known ? e.code : 'SERVICE_UNAVAILABLE';
      return json({ error: { code, message: known ? e.message : ERRORS[code],
        ...(e.retryAfter ? { retryAfter: e.retryAfter } : {}) } }, known ? e.status : 503, origin, e.retryAfter);
    } finally {
      if (reservation) {
        // Errors still count toward the quota. Failed release leaves a bounded lease.
        try { await limit(env, { action: 'release', id: reservation }); } catch { /* lease expires; never bypass quota */ }
      }
    }
  },
};
