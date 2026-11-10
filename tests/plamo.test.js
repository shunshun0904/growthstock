import test from 'node:test';
import assert from 'node:assert/strict';
import { buildAiContext, publicAiStocks, requestAnalysis } from '../src/lib/plamo.js';

const candidate = { code: '1234', jqCode: '12340', name: '公開テスト', date: '2026-11-06',
  close: 1200, roe: 12.3, per: 10, pbr: 1.2, rankInDay: 2, score: .6,
  ltr: { score: -1.2, dayRank: 3, nDay: 10 },
  contrib: { top: [{ col: 'roe', name: 'ROE', contrib: .2 }] },
  byModel: { lgbm: { score: .6, labelSpec: { kind: 'fixed_return', threshold: .1, horizon: 20 } } } };
const pred = { asOf: '2026-11-06', generatedAt: '2026-11-06T10:00:00Z', candidates: [candidate] };
const dataset = { generatedAt: '2026-11-09T10:00:00Z', stocks: [
  { code: '1234', jqCode: '12340', name: '公開テスト', asOf: '2026-11-09', origin: 'jquants',
    metrics: { price: 1300, roe: 13, disclosedDate: '2026-08-01' },
    snapshots: { m3: { date: '2026-08-06', price: 900, roe: 9 } },
    secret: 'NEVER_SEND_PRIVATE_FIELD', holdings: [{ quantity: 100 }] },
  { code: '9999', name: '非公開手入力', origin: 'manual', metrics: { price: 5 } },
] };

test('AI対象は公開データだけで作り、手入力銘柄は含めない', () => {
  assert.equal(publicAiStocks(dataset, pred).length, 1);
  assert.throws(() => buildAiContext({ dataset, predictionData: pred, stockCode: '9999', question: '財務' }), /公開/);
});

test('財務質問は実測値・出典日だけを渡し、予測や私有フィールドを送信しない', () => {
  const context = buildAiContext({ dataset, predictionData: pred, stockCode: '1234', question: 'ROEとPBRを説明' });
  assert.equal(context.metrics.find(m => m.key === 'roe').value, 13);
  assert.equal(context.metrics.find(m => m.key === 'roe').asOf, '2026-11-09');
  assert.equal(context.metrics.find(m => m.key === 'roe').disclosedDate, '2026-08-01');
  assert.equal(context.metrics.find(m => m.key === 'pbr').asOf, '2026-11-06');
  assert.equal(context.prediction, undefined);
  assert.equal(context.explanation, undefined);
  assert.doesNotMatch(JSON.stringify(context), /NEVER_SEND|holdings|9999/);
});

test('順位質問ではLTRスコアを生値で保持し、SHAPの対象を基準分類器に限定する', () => {
  const context = buildAiContext({ dataset, predictionData: pred, stockCode: '12340', question: 'ランキングで高順位の理由' });
  assert.equal(context.prediction.ltr.score, -1.2);
  assert.equal(context.prediction.asOf, '2026-11-06');
  assert.equal(context.explanation.model, 'lgbm');
  assert.equal(context.explanation.scale, 'log_odds');
  assert.equal(context.explanation.top[0].value, undefined);
  assert.ok(context.limitations.some(s => /ランキング.*SHAP/.test(s)));
});

test('過去比較は存在するスナップショットだけ。類似検索結果を捏造しない', () => {
  const context = buildAiContext({ dataset, predictionData: pred, stockCode: '1234', question: '過去の類似局面と比較' });
  assert.equal(context.historicalSnapshots.length, 1);
  assert.equal(context.historicalSnapshots[0].asOf, '2026-08-06');
  assert.ok(context.limitations.some(s => /類似/.test(s)));
  assert.equal(context.similarCases, undefined);
});

test('予測の購入基準やLTR5評価日が未記録なら、仮の値を補わない', () => {
  const predictionData = { ...pred, candidates: [{ ...candidate, ltr5: { score: 1.2, rank: 1, nPool: 5 } }] };
  const c = buildAiContext({ dataset, predictionData, stockCode: '1234', question: '予測モデルの順位' });
  assert.equal(c.prediction.models[0].target, undefined);
  assert.equal(c.prediction.ltr5, undefined);
  assert.ok(c.limitations.some(s => s.includes('LTR5の評価日')));
});

test('Worker URL未設定では外部通信しない', async () => {
  let calls = 0;
  await assert.rejects(requestAnalysis({ workerUrl: '', password: 'test', question: '説明', context: {},
    fetchImpl: () => { calls++; } }), /Worker/);
  assert.equal(calls, 0);
});

const requestContext = () => buildAiContext({ dataset, predictionData: pred, stockCode: '1234', question: '財務' });
const password = 'TEST_ONLY_SHARED_PASSWORD_NOT_A_REAL_SECRET';
test('送信時のパスワードはAuthorizationだけに置く。JSONは公開値と質問だけ', async () => {
  const calls = [];
  const data = await requestAnalysis({ workerUrl: 'https://private-ai.workers.dev', password,
    question: '財務', context: requestContext(), fetchImpl: async (url, options) => {
      calls.push({ url, options }); return Response.json({ model: 'plamo-3.0-fin-beta', answer: '基準日は2026-11-09です。' });
    } });
  assert.equal(calls.length, 1); assert.equal(calls[0].options.credentials, 'omit');
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${password}`);
  assert.deepEqual(Object.keys(JSON.parse(calls[0].options.body)).sort(), ['context', 'question']);
  assert.doesNotMatch(calls[0].options.body, /TEST_ONLY_SHARED/); assert.match(data.answer, /2026-11-09/);
});
for (const [status, code, message] of [[401, 'UNAUTHORIZED', /パスワード/], [429, 'UPSTREAM_LIMIT', /上限/],
  [500, 'UPSTREAM_UNAVAILABLE', /通信/], [504, 'TIMEOUT', /タイムアウト/], [503, 'NOT_CONFIGURED', /未完了/]]) {
  test(`フロントはHTTP${status}を説明し、再試行・上流本文の表示をしない`, async () => {
    let calls = 0;
    await assert.rejects(requestAnalysis({ workerUrl: 'https://private-ai.workers.dev', password, question: '財務', context: requestContext(),
      fetchImpl: async () => { calls++; return Response.json({ error: { code, message: 'DO_NOT_EXPOSE_PROVIDER_DETAILS' } }, { status }); } }), message);
    assert.equal(calls, 1);
  });
}
test('フロントのタイムアウトは通信を中断する', async () => {
  let aborted = false;
  await assert.rejects(requestAnalysis({ workerUrl: 'https://private-ai.workers.dev', password, question: '財務', context: requestContext(), timeoutMs: 5,
    fetchImpl: async (_, { signal }) => new Promise((resolve, reject) => signal.addEventListener('abort', () => {
      aborted = true; reject(new DOMException('aborted', 'AbortError'));
    }, { once: true })) }), /タイムアウト/);
  assert.equal(aborted, true);
});
test('空回答と別モデルの回答を表示しない', async () => {
  for (const data of [{ answer: '', model: 'plamo-3.0-fin-beta' }, { answer: '別モデル', model: 'plamo-3.1-prime' }]) {
    await assert.rejects(requestAnalysis({ workerUrl: 'https://private-ai.workers.dev', password, question: '財務', context: requestContext(),
      fetchImpl: async () => Response.json(data) }), /回答|モデル/);
  }
});
test('ゼロは実測値として残し、欠損をゼロに変換しない', () => {
  const source = { stocks: [{ code: '1234', name: '合成データ', asOf: '2026-11-09', metrics: { roe: 0, per: null } }] };
  const c = buildAiContext({ dataset: source, stockCode: '1234', question: 'ROEとPER' });
  assert.equal(c.metrics[0].value, 0); assert.equal(c.metrics.some(m => m.key === 'per'), false);
});
