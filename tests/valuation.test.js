import test from 'node:test';
import assert from 'node:assert/strict';
import { rangePosition, trend } from '../src/lib/valuation.js';
test('current price position preserves out-of-range values and handles collapsed ranges', () => {
  assert.equal(rangePosition(150,100,200),0.5);
  assert.equal(rangePosition(250,100,200),1.5);
  assert.equal(rangePosition(50,100,200),-0.5);
  assert.equal(rangePosition(150,100,100),null);
  assert.equal(rangePosition(null,100,200),null);
});
test('trend uses first/last actual observations and does not turn missing data into zero', () => {
  assert.ok(Math.abs(trend([{close:100},{close:110}])-.1)<1e-12);
  assert.equal(trend([{close:100}]),null);
});
