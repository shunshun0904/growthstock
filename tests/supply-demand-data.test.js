import test from 'node:test';
import assert from 'node:assert/strict';
import { createSupplyDemandClient, normalizeSupplyCode } from '../src/lib/supplyDemandData.js';

test('canonical codes cover stored four-digit and alphanumeric securities without unsafe paths', () => {
  assert.equal(normalizeSupplyCode('2894'), '28940');
  assert.equal(normalizeSupplyCode('175A'), '175A0');
  assert.equal(normalizeSupplyCode('94346'), '94346');
  assert.equal(normalizeSupplyCode('../x'), null);
});

const response = data => ({ ok: true, json: async () => data });
const index = { schemaVersion: 1, stocks: { '28940': { status: 'ok' } } };

test('a stored stock loads on demand through the Pages prefix and successful reads are cached', async () => {
  const paths = [];
  const client = createSupplyDemandClient({ baseUrl: '/growthstock/', fetchImpl: async url => {
    paths.push(url);
    return response(url.includes('index.json') ? index : { code: '28940', supplyDemand: { schemaVersion: 1, status: 'ok', currentPrice: 373 } });
  } });
  assert.equal((await client.loadIndex()).stocks['28940'].status, 'ok');
  const data = await client.loadStock('2894');
  assert.equal(data.currentPrice, 373);
  assert.equal(await client.loadStock('28940'), data);
  assert.equal(paths.length, 2);
  assert.ok(paths.every(p => p.startsWith('/growthstock/data/supply-demand/')));
});

test('wrong-code responses are rejected and failed requests can be retried', async () => {
  let calls = 0;
  const client = createSupplyDemandClient({ fetchImpl: async () => {
    calls++;
    return response({ code: calls === 1 ? '64670' : '28940', supplyDemand: { schemaVersion: 1, status: 'ok' } });
  } });
  await assert.rejects(client.loadStock('28940'), /形式/);
  assert.equal((await client.loadStock('28940')).status, 'ok');
});

test('404 is reported as a request error instead of synthesizing zero positions', async () => {
  const client = createSupplyDemandClient({ fetchImpl: async () => ({ ok: false, status: 404 }) });
  await assert.rejects(client.loadStock('28940'), /404/);
});

test('a timed-out request produces a retryable error', async () => {
  const client = createSupplyDemandClient({ timeoutMs: 5, fetchImpl: (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  }) });
  await assert.rejects(client.loadIndex(), /時間切れ/);
});
