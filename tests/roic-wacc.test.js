import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { createRoicWaccClient, decodeSeries, disclosureMarkers, normalizeCode, scenarioSeries, seriesCsv } from '../src/lib/roicWacc.js';
import { researchFixture as fixture } from './fixtures/roic-wacc-fixture.mjs';

test('columnar research series decode, with no price imputation', () => {
  const d = decodeSeries(fixture(1));
  assert.equal(d.rows[0].price, 50);
  assert.equal(d.rows[0].eps_forecast, null);
  assert.throws(() => decodeSeries({ ...fixture(1), columns: ['Date', 'Date'] }), /形式/);
  assert.throws(() => decodeSeries({ ...fixture(1), data: [[1]] }), /形式/);
});

test('sensitivity ranges separate business growth and market-dependent comparison', () => {
  const d = decodeSeries(fixture(1));
  const primary = scenarioSeries(d)[0];
  assert.equal(primary.median, 110);
  assert.equal(primary.min, 100); assert.equal(primary.max, 120); assert.equal(primary.std, 10);
  assert.equal(primary.selectedWacc, .08);
  assert.equal(primary.gap, 50 / 110);
  assert.equal(primary.selectedEvGap, 60 / 120);
  assert.equal(scenarioSeries(d, { growth: 'business' })[0].median, 200);
  assert.equal(scenarioSeries(d, { model: 'steady' })[0].median, 300);
  assert.equal(scenarioSeries(d, { model: 'steady' })[0].selectedWacc, .09);
});

test('forecast EPS does not fall back to TTM and CSV retains assumptions', () => {
  const d = decodeSeries(fixture(1));
  assert.equal(scenarioSeries(d)[0].marketPer, null);
  assert.equal(scenarioSeries(d, { epsBasis: 'ttm' })[0].marketPer, 5);
  const csv = seriesCsv(scenarioSeries(d));
  assert.match(csv, /price_independent_model/); assert.match(csv, /eva_fade5/);
  assert.match(csv, /valuation_price_log_change_20d/);
});

test('selected-scenario Z score uses 126 observations and no future data', () => {
  const original = decodeSeries(fixture(150));
  const a = scenarioSeries(original);
  assert.equal(a[124].gapZ, null); assert.ok(Number.isFinite(a[125].gapZ));
  const changed = structuredClone(original);
  changed.rows.slice(130).forEach(row => { row.price = 1e10; row.scenarios['eva-a'] = 1; });
  const b = scenarioSeries(changed);
  assert.deepEqual(a.slice(0, 130), b.slice(0, 130));
  for (const row of a.slice(20)) {
    assert.ok(Math.abs(row.valuation_gap_log_change_20d - row.valuation_price_log_change_20d + row.valuation_value_log_change_20d) < 1e-12);
  }
});

test('missing valuation is missing, never a fabricated zero line', () => {
  const d = decodeSeries(fixture(1)); d.rows[0].scenarios = {};
  const row = scenarioSeries(d)[0];
  assert.equal(row.median, null); assert.equal(row.gap, null); assert.equal(row.range, null);
});

test('intrinsic value changes use share units, and unverified actions stop the log decomposition', () => {
  const input = decodeSeries(fixture(30));
  input.rows.forEach(row => { row.action_uncertainty = 0; });
  const a = scenarioSeries(input);
  const changed = structuredClone(input);
  changed.rows.forEach((row, i) => { row.price *= 1 + i; row.price_adjusted *= 1 + i; });
  const b = scenarioSeries(changed);
  assert.equal(a[25].theoretical_price_change_20d, b[25].theoretical_price_change_20d);
  assert.equal(a[25].valuation_value_log_change_20d, b[25].valuation_value_log_change_20d);
  assert.notEqual(a[25].valuation_price_log_change_20d, b[25].valuation_price_log_change_20d);
  changed.rows.slice(24).forEach(row => { row.action_uncertainty = 1; });
  const unsafe = scenarioSeries(changed)[25];
  assert.equal(unsafe.theoretical_price_change_20d, null);
  assert.equal(unsafe.valuation_price_log_change_20d, null);
  assert.equal(unsafe.valuation_value_log_change_20d, null);
  assert.equal(unsafe.valuation_gap_log_change_20d, null);
});

test('disclosure markers do not fabricate an event at the display boundary', () => {
  const rows = [
    { Date: '2025-07-01', financial_source_date: '2025-06-20', financial_available_date: '2025-06-23' },
    { Date: '2025-07-04', financial_source_date: '2025-06-20', financial_available_date: '2025-06-23' },
    { Date: '2025-07-07', financial_source_date: '2025-07-05', financial_available_date: '2025-07-07' },
  ];
  assert.deepEqual(disclosureMarkers(rows, rows), [{ date: '2025-07-07', sourceDate: '2025-07-05' }]);
});

test('Pages subpath and gzip JSON are supported without calling market APIs', async () => {
  const calls = [];
  const client = createRoicWaccClient({ baseUrl: '/growthstock/', fetchImpl: async url => {
    calls.push(url);
    return new Response(gzipSync(JSON.stringify(fixture(1))));
  } });
  const d = await client.series({ code: '1000', file: '10000.json.gz' });
  assert.equal(d.rows.length, 1);
  assert.deepEqual(calls, ['/growthstock/data/roic-wacc/10000.json.gz']);
});

test('paths, response errors and wrong-security data are rejected', async () => {
  assert.equal(normalizeCode('130a'), '130A0');
  assert.throws(() => normalizeCode('../secret'), /形式/);
  const client = createRoicWaccClient({ fetchImpl: async () => new Response(JSON.stringify(fixture(1))) });
  await assert.rejects(() => client.series({ code: '1000', file: '../data.json' }), /パス/);
  await assert.rejects(() => client.series({ code: '2000' }), /一致/);
  const failed = createRoicWaccClient({ fetchImpl: async () => new Response('', { status: 503 }) });
  await assert.rejects(() => failed.catalog(), /503/);
});
