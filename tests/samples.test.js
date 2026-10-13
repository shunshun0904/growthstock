/**
 * 再学習で加わった例（src/lib/samples.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  splitCases, caseChart, verdictLines, fromEntry, fmtMD, windowText, addedPositiveRate, niceAxis,
} from '../src/lib/samples.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'fixtures', 'synthetic-new-samples.json'), 'utf8'));

test('splitCases: 正例と負例に分ける（並びはファイルの順）', () => {
  const { pos, neg } = splitCases(fixture);
  assert.deepEqual(pos.map((c) => c.code), ['1111', '4444']);
  assert.deepEqual(neg.map((c) => c.code), ['2222', '3333']);
  assert.deepEqual(splitCases(null), { pos: [], neg: [] });
});

test('caseChart: 横軸は営業日の並び、窓は更新日の翌日から20営業日、縦軸は線がすべて入る', () => {
  const c = fixture.cases[0];
  const ch = caseChart(c);
  assert.equal(ch.points.length, c.dates.length);
  assert.equal(ch.tPos, 40);
  assert.deepEqual(ch.win, [41, 60]);
  assert.deepEqual(ch.endWin, [56, 60]);                   // 終盤の水準は窓の最後5日
  assert.equal(ch.entryPos, 41);
  const [lo, hi] = ch.yDomain;
  for (const v of [c.entryPrice, c.target, c.endLineAdopted, ...c.close]) {
    assert.ok(v >= lo && v <= hi, `${v} が ${lo}〜${hi} の外`);
  }
  assert.deepEqual(ch.ticks, [0, 40, 60, c.dates.length - 1]);
  // 縦軸の目盛りはきりのよい等間隔で、範囲の両端を含む
  const steps = ch.yTicks.slice(1).map((v, i) => +(v - ch.yTicks[i]).toFixed(6));
  assert.ok(steps.every((d) => d === steps[0]) && [1, 2, 2.5, 5].includes(steps[0] / 10 ** Math.floor(Math.log10(steps[0]))));
  assert.deepEqual([ch.yTicks[0], ch.yTicks.at(-1)], ch.yDomain);
});

test('niceAxis: 1・2・2.5・5 × 10^k の刻みで、範囲を必ず含む', () => {
  assert.deepEqual(niceAxis(953, 1251), { domain: [900, 1300], ticks: [900, 1000, 1100, 1200, 1300] });
  assert.deepEqual(niceAxis(0.95, 1.05).domain, [0.95, 1.05]);
  assert.deepEqual(niceAxis(5, 5).domain[0] <= 5, true);
  assert.deepEqual(niceAxis(NaN, 1), { domain: [0, 1], ticks: [0, 1] });
});

test('caseChart: 日足が窓より短くても壊れない（窓と目盛りを端で切る）', () => {
  const ch = caseChart({ dates: ['2026-09-01', '2026-09-02', '2026-09-03'], close: [1, 2, 3], tPos: 1,
                          horizon: [2, 21], entryPrice: 2, target: 2.4 });
  assert.deepEqual(ch.win, [2, 2]);
  assert.deepEqual(ch.ticks, [0, 1, 2]);
  assert.equal(caseChart({}).points.length, 0);
});

test('verdictLines: 正例は3つとも ✓、負例は外れた条件が ✗', () => {
  const pos = verdictLines(fixture.cases[0]);
  assert.deepEqual(pos.map((l) => l.ok), [true, true, true]);
  assert.equal(pos[0].text, '最大 +15.0%（必要 +9.5%）');
  const neg = verdictLines(fixture.cases.find((c) => c.code === '3333'));
  assert.deepEqual(neg.map((l) => l.ok), [false, false, false]);
  assert.equal(neg[2].text, 'MA5 < MA20');
  assert.equal(verdictLines({})[0].text, '最大 —（必要 —）');
});

test('fromEntry・fmtMD・windowText・addedPositiveRate', () => {
  assert.equal(fromEntry(110, 100).toFixed(1), '10.0');
  assert.equal(fromEntry(null, 100), null);
  assert.equal(fmtMD('2026-09-08'), '9/8');
  assert.equal(windowText({ window: { after: '2026-09-01', to: '2026-09-08' } }), '2026/09/01 より後 〜 2026/09/08');
  assert.equal(windowText({ window: { to: '2026-09-08' } }), '〜 2026/09/08');
  assert.equal(addedPositiveRate({ counts: { added: 20, positive: 5 } }), 25);
  assert.equal(addedPositiveRate({ counts: { added: 0, positive: 0 } }), null);
});
