import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Miniflare } from 'miniflare';
import { quotaStep } from '../src/limiter.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const ORIGIN = 'https://shunshun0904.github.io';
const PASSWORD = 'test-only-authentication-secret-for-worker-tests';
const API_KEY = 'PROVIDER_TEST_ONLY_NOT_REAL';
const payload = { question: 'この銘柄のROEを説明してください', context: {
  stockCode: '1234', stockName: '合成テスト銘柄', asOf: '2026-11-09',
  metrics: [{ key: 'roe', value: 12.3, asOf: '2026-11-09', source: 'stocks.json' }], limitations: [],
} };
const providerAnswer = overrides => ({ model: 'plamo-3.0-fin-beta',
  choices: [{ message: { content: '基準日は2026-11-09です。ROEは12.3%です。' }, finish_reason: 'stop' }], ...overrides });
function config(overrides = {}) {
  return { ALLOWED_ORIGINS: ORIGIN, AI_PASSWORD: PASSWORD, PLAMO_API_KEY: API_KEY,
    PLAMO_FREE_BETA_CONFIRMED: 'true', CHAT_PER_MINUTE: '5', CHAT_PER_DAY: '20', CHAT_PER_MONTH: '300', ...overrides };
}
function options(bindings, outgoing, persist = false, withLimiter = true) {
  return { name: 'growthstock-plamo-test', modules: true, modulesRoot: path.join(ROOT, 'src'),
    modulesRules: [{ type: 'ESModule', include: ['**/*.js'] }],
    scriptPath: path.join(ROOT, 'src/index.js'), compatibilityDate: '2026-07-01',
    bindings, durableObjects: withLimiter ? { AI_LIMITER: { className: 'AiLimiter', useSQLite: true } } : {},
    durableObjectsPersist: persist, cf: false, outboundService: outgoing };
}
async function app(t, bindings = {}, outgoing = () => Response.json(providerAnswer()), withLimiter = true) {
  const mf = new Miniflare(options(config(bindings), outgoing, false, withLimiter));
  t.after(() => mf.dispose()); await mf.ready; return mf;
}
function request(mf, { body = payload, auth = PASSWORD, origin = ORIGIN, ip = '192.0.2.1', route = '/chat', method = 'POST', headers = {} } = {}) {
  return mf.dispatchFetch(`https://ai.example${route}`, { method,
    headers: { ...(origin ? { Origin: origin } : {}), ...(auth ? { Authorization: `Bearer ${auth}` } : {}),
      'CF-Connecting-IP': ip, 'Content-Type': 'application/json', ...headers },
    ...(method === 'POST' ? { body: typeof body === 'string' ? body : JSON.stringify(body) } : {}) });
}

test('CORSは完全一致。許可されたpreflightだけを返し、認証を代替しない', async t => {
  let calls = 0; const mf = await app(t, {}, () => { calls++; return Response.json(providerAnswer()); });
  const pre = await request(mf, { method: 'OPTIONS', auth: null, headers: { 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'Authorization, Content-Type' } });
  assert.equal(pre.status, 204); assert.equal(pre.headers.get('Access-Control-Allow-Origin'), ORIGIN);
  assert.equal((await request(mf, { origin: 'https://shunshun0904.github.io.attacker.example' })).status, 403);
  assert.equal((await request(mf, { origin: null })).status, 403);
  for (const auth of [null, 'incorrect']) {
    const r = await request(mf, { auth }); assert.equal(r.status, 401);
    assert.equal(r.headers.get('Access-Control-Allow-Origin'), ORIGIN);
  }
  assert.equal(calls, 0);
});
test('認証後に公開値を公式モデルへ1回中継し、秘密を本文・応答へ含めない', async t => {
  const calls = [];
  const mf = await app(t, {}, async req => {
    calls.push({ url: req.url, headers: Object.fromEntries(req.headers), body: await req.json() });
    return Response.json(providerAnswer());
  });
  const r = await request(mf); assert.equal(r.status, 200, JSON.stringify({ response: await r.clone().text(), providerCalls: calls.length }));
  const data = await r.json(); assert.equal(data.model, 'plamo-3.0-fin-beta');
  assert.equal(data.usage.callsToday, 1); assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'https://api.platform.preferredai.jp/v1/chat/completions');
  assert.equal(calls[0].headers.authorization, `Bearer ${API_KEY}`);
  assert.equal(calls[0].body.model, 'plamo-3.0-fin-beta'); assert.equal(calls[0].body.max_tokens, 1536);
  assert.equal(calls[0].body.stream, false); assert.equal(calls[0].body.tools, undefined);
  assert.match(calls[0].body.messages[1].content, /12.3/);
  assert.match(calls[0].body.messages[0].content, /確率へ変換しない/);
  assert.doesNotMatch(JSON.stringify(calls[0].body), new RegExp(`${PASSWORD}|${API_KEY}`));
  assert.doesNotMatch(JSON.stringify(data), new RegExp(`${PASSWORD}|${API_KEY}`));
  assert.equal(r.headers.get('Cache-Control'), 'no-store');
});
test('APIキー未設定・無料確認前・有料モデル・任意URLは呼び出さず停止する', async t => {
  for (const bindings of [{ PLAMO_API_KEY: '' }, { PLAMO_FREE_BETA_CONFIRMED: 'false' },
    { PLAMO_MODEL: 'plamo-3.1-prime' }, { PLAMO_ENDPOINT: 'https://attacker.example/chat' }]) {
    let calls = 0; const mf = await app(t, bindings, () => { calls++; return Response.json(providerAnswer()); });
    assert.equal((await request(mf)).status, 503); assert.equal(calls, 0);
    const status = await (await request(mf, { route: '/usage', method: 'GET' })).json();
    assert.equal(status.configured, false); assert.equal(status.callsToday, 0);
  }
});
test('質問・型・未知フィールド・大きな本文を拒否し、APIへ転送しない', async t => {
  let calls = 0; const mf = await app(t, {}, () => { calls++; return Response.json(providerAnswer()); });
  const inputs = [{ ...payload, question: '' }, { ...payload, question: '字'.repeat(1201) },
    { ...payload, model: 'paid' }, { ...payload, context: { ...payload.context, privateHistory: 'private' } },
    { ...payload, context: { ...payload.context, metrics: [], prediction: { asOf: '2026-11-09', models: [] } } },
    { ...payload, context: { ...payload.context, metrics: [{ ...payload.context.metrics[0], value: '12.3' }] } }];
  for (const body of inputs) assert.equal((await request(mf, { body })).status, 400);
  assert.equal((await request(mf, { body: JSON.stringify({ question: 'x'.repeat(26000) }) })).status, 413);
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode('{"question":"'));
    controller.enqueue(new Uint8Array(26000).fill(120)); controller.close();
  } });
  const chunked = await mf.dispatchFetch('https://ai.example/chat', { method: 'POST', duplex: 'half', body: stream,
    headers: { Origin: ORIGIN, Authorization: `Bearer ${PASSWORD}`, 'Content-Type': 'application/json' } });
  assert.equal(chunked.status, 413);
  assert.equal(calls, 0);
});
for (const [status, code, expected] of [[429, 'UPSTREAM_LIMIT', 429], [500, 'UPSTREAM_UNAVAILABLE', 502],
  [401, 'UPSTREAM_AUTH', 503], [404, 'MODEL_UNAVAILABLE', 503], [302, 'UPSTREAM_UNAVAILABLE', 502]]) {
  test(`上流HTTP${status}を安全なエラーへ変換し、自動再試行しない`, async t => {
    let calls = 0; const mf = await app(t, {}, () => { calls++; return new Response(`private: ${API_KEY}`, { status, headers: { 'Retry-After': '17', 'Location': 'https://attacker.example/' } }); });
    const r = await request(mf); const data = await r.json();
    assert.equal(r.status, expected); assert.equal(data.error.code, code); assert.equal(calls, 1);
    assert.doesNotMatch(JSON.stringify(data), new RegExp(API_KEY));
    if (status === 429) assert.equal(r.headers.get('Retry-After'), '17');
  });
}
test('空回答・別モデルを拒否し、内部推論だけの応答を回答に使わない', async t => {
  for (const answer of [providerAnswer({ choices: [{ message: { content: '', reasoning_content: 'internal reasoning' } }] }),
    providerAnswer({ model: 'plamo-3.1-prime' })]) {
    const mf = await app(t, {}, () => Response.json(answer));
    const r = await request(mf); assert.equal(r.status, 502);
    assert.doesNotMatch(await r.text(), /internal reasoning/);
  }
});
test('上流が完了しない場合は期限で終了する', async t => {
  let calls = 0; const mf = await app(t, { UPSTREAM_TIMEOUT_MS: '50' }, async () => {
    calls++; await new Promise(resolve => setTimeout(resolve, 300)); return Response.json(providerAnswer());
  });
  const r = await request(mf); assert.equal(r.status, 504); assert.equal((await r.json()).error.code, 'TIMEOUT'); assert.equal(calls, 1);
});
test('同時要求は1件だけ上流へ送る', async t => {
  let started, finish; const ready = new Promise(resolve => { started = resolve; });
  const wait = new Promise(resolve => { finish = resolve; }); let calls = 0;
  const mf = await app(t, {}, async () => { calls++; started(); await wait; return Response.json(providerAnswer()); });
  const first = request(mf);
  try {
    await Promise.race([ready, first.then(r => { throw new Error(`First request ended before provider call: HTTP${r.status}`); })]);
    const second = await request(mf); assert.equal(second.status, 409); assert.equal((await second.json()).error.code, 'BUSY');
  } finally { finish(); }
  assert.equal((await first).status, 200); assert.equal(calls, 1);
});
test('失敗した試行も日次上限に含み、SQLite再起動後も上限を保持する', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'plamo-test-')); let mf; let calls = 0;
  const bindings = config({ CHAT_PER_DAY: '1' });
  const outgoing = () => { calls++; return new Response('unavailable', { status: 500 }); };
  try {
    mf = new Miniflare(options(bindings, outgoing, dir)); await mf.ready;
    assert.equal((await request(mf)).status, 502); await mf.dispose(); mf = null;
    mf = new Miniflare(options(bindings, outgoing, dir)); await mf.ready;
    const r = await request(mf); assert.equal(r.status, 429); assert.equal((await r.json()).error.code, 'DAILY_LIMIT'); assert.equal(calls, 1);
  } finally { await mf?.dispose(); await rm(dir, { recursive: true, force: true }); }
});
test('総当たりをIP単位で制限する', async t => {
  let calls = 0; const mf = await app(t, {}, () => { calls++; return Response.json(providerAnswer()); });
  for (let i = 0; i < 10; i++) assert.equal((await request(mf, { auth: 'invalid' })).status, 401);
  const r = await request(mf, { auth: 'invalid' }); assert.equal(r.status, 429); assert.equal((await r.json()).error.code, 'AUTH_RATE_LIMIT');
  assert.equal(calls, 0);
});
test('制限用ストレージを利用できない場合は閉じる', async t => {
  let calls = 0; const mf = await app(t, {}, () => { calls++; return Response.json(providerAnswer()); }, false);
  assert.equal((await request(mf)).status, 503); assert.equal(calls, 0);
});
test('日付・月が変わると対応する上限だけをリセットする。古いlease解放で新しいleaseを消さない', () => {
  const limits = { minute: 5, day: 1, month: 2 };
  let step = quotaStep(undefined, { action: 'reserve', id: 'first' }, Date.parse('2026-11-29T23:59:00Z'), limits);
  step = quotaStep(step.state, { action: 'reserve', id: 'second' }, Date.parse('2026-11-30T00:00:00Z'), limits);
  assert.equal(step.result.allowed, true); assert.equal(step.result.usage.callsThisMonth, 2);
  const released = quotaStep(step.state, { action: 'release', id: 'first' }, Date.parse('2026-11-30T00:00:01Z'), limits);
  assert.equal(released.state.lease.id, 'second');
  step = quotaStep(step.state, { action: 'reserve', id: 'third' }, Date.parse('2026-12-01T00:00:00Z'), limits);
  assert.equal(step.result.usage.callsThisMonth, 1);
});
