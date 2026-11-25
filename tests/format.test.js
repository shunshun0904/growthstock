/**
 * 表示フォーマット（src/lib/format.js）。
 *
 * - fmtWithForecast: 実績に予想を添える（PER・ROE。運用者の選択 2026-11-25「実績と予想を並べる」）
 *   予想が無ければ実績だけ、実績が無ければ「—（予想 …）」、どちらも無ければ「—」
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { fmtWithForecast, DASH } from '../src/lib/format.js';

test('実績に予想を添える', () => {
  assert.equal(fmtWithForecast(120.0, 24.96, 1, '倍'), '120.0倍（予想 25.0倍）');
  assert.equal(fmtWithForecast(4.04, 30.06, 1, '%'), '4.0%（予想 30.1%）');
});

test('予想が無ければ実績だけ', () => {
  assert.equal(fmtWithForecast(12.34, null, 1, '倍'), '12.3倍');
  assert.equal(fmtWithForecast(12.34, undefined, 1, '倍'), '12.3倍');
  assert.equal(fmtWithForecast(12.34, Number.NaN, 1, '倍'), '12.3倍');
});

test('実績が無ければ「—」に予想を添える。どちらも無ければ「—」', () => {
  assert.equal(fmtWithForecast(null, 25.0, 1, '倍'), `${DASH}（予想 25.0倍）`);
  assert.equal(fmtWithForecast(null, null, 1, '倍'), DASH);
});
