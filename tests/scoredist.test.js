// スコアと収益の分布（src/lib/scoredist.js）の組み立て
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  histRows, toShares, countInto, liveLgbmScores, rangeLabel, decileLabel, pct, FIXED_VERSION,
} from '../src/lib/scoredist.js';

test('histRows: 区間ごとの行に度数を並べる', () => {
  const r = histRows([0, 0.1, 0.2], { a: [3, 4], b: [1] });
  assert.equal(r.length, 2);
  assert.deepEqual([r[0].lo, r[0].hi, r[0].a, r[0].b], [0, 0.1, 3, 1]);
  assert.equal(r[1].b, 0);
  assert.ok(Math.abs(r[1].mid - 0.15) < 1e-12);
  assert.deepEqual(histRows(null, {}), []);
});

test('toShares: 合計に対する % （合計0なら0）', () => {
  assert.deepEqual(toShares([1, 3]), [25, 75]);
  assert.deepEqual(toShares([0, 0]), [0, 0]);
  assert.deepEqual(toShares(undefined), []);
});

test('countInto: 端の外側は最初・最後の区間にまとめる（research/score_dist.hist と同じ）', () => {
  assert.deepEqual(countInto([-5, 0.05, 0.1, 0.5, 9, null, NaN], [0, 0.1, 0.2, 1]), [2, 1, 2]);
});

test('liveLgbmScores: 固定 +10% で採点した lgbm のスコアだけを、銘柄・日付ごとに1つ', () => {
  const fixed = { version: FIXED_VERSION };
  const pred = {
    candidates: [
      { code: '1', date: '2026-11-02', byModel: { lgbm: { score: 0.5, labelSpec: fixed } } },
      { code: '1', date: '2026-11-02', byModel: { lgbm: { score: 0.5, labelSpec: fixed } } },
      { code: '2', date: '2026-11-04', byModel: { lgbm: { score: 0.3, labelSpec: fixed } } },
      { code: '3', date: '2026-10-30', byModel: { lgbm: { score: 0.9, labelSpec: { version: 'sigma' } } } },
      { code: '4', date: '2026-11-05', byModel: {} },
    ],
  };
  const r = liveLgbmScores(pred);
  assert.deepEqual(r.scores, [0.5, 0.3]);
  assert.deepEqual(r.dates, ['2026-11-02', '2026-11-04']);
  assert.equal(liveLgbmScores(null).n, 0);
});

test('文字: 区間・10分位・%', () => {
  assert.equal(rangeLabel(0.5, 0.6, 1), '0.5〜0.6');
  assert.equal(rangeLabel(-2.5, 0, 1, '%'), '-2.5〜0.0%');
  assert.equal(rangeLabel(10, 12.5, 1, '%'), '+10.0〜+12.5%');
  assert.equal(decileLabel(1), '0〜10%');
  assert.equal(decileLabel(10), '90〜100%');
  assert.equal(pct(0.0242, 2, true), '+2.42%');
  assert.equal(pct(-0.0179, 2, true), '-1.79%');
  assert.equal(pct(null), '—');
});
