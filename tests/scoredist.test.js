// スコアと収益の分布（src/lib/scoredist.js）の組み立て
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  histRows, toShares, countInto, liveLgbmScores, rangeLabel, decileLabel, pct, FIXED_VERSION,
  percentileOf, poolDecile, bandOf, candidatePositions, candidateDates,
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

test('percentileOf: 1%刻みの分位点で線形に補間（外側は 0 / 100）', () => {
  const grid = Array.from({ length: 101 }, (_, i) => i / 100);       // 0.00〜1.00
  assert.equal(percentileOf(0.5, grid), 50);
  assert.ok(Math.abs(percentileOf(0.255, grid) - 25.5) < 1e-9);
  assert.equal(percentileOf(-1, grid), 0);
  assert.equal(percentileOf(2, grid), 100);
  assert.equal(percentileOf(null, grid), null);
  assert.equal(percentileOf(0.5, [0.1]), null);
  const flat = [0, 0.2, 0.2, 0.2, 1];                                 // 同じ値が続く区間は真ん中（4区間 → 25%刻み）
  assert.equal(percentileOf(0.2, flat), 50);
});

test('poolDecile: research/score_dist.pool_rank と同じ式', () => {
  assert.deepEqual(poolDecile(1, 10), { pp: 1, dec: 10, top10: true });
  assert.equal(poolDecile(10, 10).dec, 1);
  assert.equal(poolDecile(2, 10).top10, false);                     // pp = 0.9 は上位10% に入れない（pp > 0.9）
  assert.equal(poolDecile(30, 64).dec, 6);                          // pp = 1 − 29/64 = 0.547
  assert.equal(poolDecile(0, 10), null);
  assert.equal(poolDecile(11, 10), null);
});

test('bandOf: lo < score ≤ hi（最初の帯は lo を含む）', () => {
  const bands = [{ lo: 0, hi: 0.2 }, { lo: 0.2, hi: 0.3 }];
  assert.equal(bandOf(0, bands).hi, 0.2);
  assert.equal(bandOf(0.2, bands).hi, 0.2);
  assert.equal(bandOf(0.25, bands).hi, 0.3);
  assert.equal(bandOf(0.9, bands), null);
});

test('candidatePositions: 日付の候補を lgbm のスコアの順に、位置・帯・10分位・重なりを当てる', () => {
  const fixed = { version: FIXED_VERSION };
  const pred = {
    candidates: [
      { code: 'A', name: 'a', date: 'd2', byModel: { lgbm: { score: 0.3, labelSpec: fixed } }, ltr5: { rank: 1, nPool: 20 } },
      { code: 'B', name: 'b', date: 'd2', byModel: { lgbm: { score: 0.6, labelSpec: fixed } }, ltr5: { rank: 15, nPool: 20 } },
      { code: 'C', name: 'c', date: 'd2', byModel: { lgbm: { score: 0.7, labelSpec: fixed } } },
      { code: 'D', name: 'd', date: 'd1', byModel: { lgbm: { score: 0.9, labelSpec: fixed } } },
      { code: 'E', name: 'e', date: 'd2', byModel: { lgbm: { score: 0.9, labelSpec: { version: 'old' } } } },
    ],
  };
  const dist = {
    lgbm: { pctGrid: Array.from({ length: 101 }, (_, i) => i / 100), bands: [{ lo: 0, hi: 0.5, posRate: 0.1 }, { lo: 0.5, hi: 1, posRate: 0.3 }] },
    ltr5: { deciles: [{ dec: 10, r20Mean: 0.02 }, { dec: 3, r20Mean: 0.004 }] },
    overlap: { lgbmThreshold: 0.55, groups: [{ key: 'both', label: '両方' }, { key: 'lgbmOnly', label: 'lgbm だけ', r20Below10: 0.25 },
                                             { key: 'ltr5Only', label: 'ltr5 だけ' }, { key: 'neither', label: 'どちらも' }] },
  };
  const r = candidatePositions(pred, dist, 'd2');
  assert.deepEqual(r.map((x) => x.code), ['C', 'B', 'A']);
  const [c, b, a] = r;
  assert.equal(c.ltr5, null);
  assert.equal(c.group, null);
  assert.ok(c.lgbmTop10);
  assert.equal(b.group.key, 'lgbmOnly');
  assert.equal(b.group.r20Below10, 0.25);
  assert.equal(b.ltr5.dec, 3);
  assert.equal(b.ltr5.stats.r20Mean, 0.004);
  assert.equal(b.band.posRate, 0.3);
  assert.ok(Math.abs(b.lgbmPct - 60) < 1e-9);
  assert.equal(a.group.key, 'ltr5Only');
  assert.equal(a.ltr5.stats.r20Mean, 0.02);
  assert.equal(a.band.posRate, 0.1);
  assert.deepEqual(candidateDates(pred), ['d2', 'd1']);
  assert.equal(candidatePositions(pred, dist, 'd1').length, 1);
});
