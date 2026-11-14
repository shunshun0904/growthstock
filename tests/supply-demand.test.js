import test from 'node:test';
import assert from 'node:assert/strict';
import { supplyChoices, profileBands, sideDisplay, yen, shares } from '../src/lib/supplyDemand.js';

test('detailed stock data wins over candidate snapshots; missing stocks stay selectable', () => {
  const choices = supplyChoices([{ id: 'jq-12340', jqCode: '12340', code: '1234', supplyDemand: {} }],
    [{ jqCode: '12340', code: '1234' }, { jqCode: '56780', code: '5678' }]);
  assert.equal(choices.length, 2);
  assert.ok(choices[0].supplyDemand);
  assert.equal(choices[1].id, 'pred:56780');
});

test('table combines both sides on the same price bands without losing shares', () => {
  const bands = profileBands({ long: { bins: [{ low: 10, high: 20, price: 15, shares: 120 }] },
    short: { bins: [{ low: 10, high: 20, price: 16, shares: 60 }, { low: 20, high: 30, price: 25, shares: 10 }] } });
  assert.deepEqual(bands, [{ low: 20, high: 30, long: 0, short: 10 }, { low: 10, high: 20, long: 120, short: 60 }]);
});

test('reported means take priority over estimates and missing values are never zero', () => {
  assert.deepEqual(sideDisplay({ averagePrice: 100, referencePnlRate: .1 }, { estimatedAveragePrice: 90 }),
    { average: 100, pnl: .1, estimated: false });
  assert.equal(sideDisplay({}, { estimatedAveragePrice: 90, estimatedPnlRate: .2 }).estimated, true);
  assert.equal(shares(0), '0株');
  assert.equal(shares(null), '未取得');
  assert.equal(yen(null), '未取得');
  assert.equal(profileBands(null).length, 0);
});
