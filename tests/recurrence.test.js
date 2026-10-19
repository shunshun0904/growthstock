/**
 * リカレンス・ネットワーク（src/lib/recurrence.js）の単体テスト。
 *
 * - 辺と要約は Python（research/recurrence_net.py）と同じ: tests/fixtures/recurrence/ の候補のファイルから、辺の密度ごとに数えた
 *   要約が tests/fixtures/recurrence-expected.json（Python が数えた値。tests/test_recurrence_net.py が同じ値を確かめる）と一致する
 * - ε² は結べる組（タイラー窓より離れた組）の距離の2乗の m 番目（0.5 は切り上げ）。同じ値は全部結ぶ。タイラー窓の中は結ばない
 * - 日付は start に gaps の暦日を足していく（最後の日が予測日）
 * - 色の区分（+10% 到達 / 0〜+10 / −10〜0 / −10 未満 / 未確定、正例 / 負例 / 未確定）と、丸めた z から生の値の目安への戻し方
 *
 *   node --test tests/recurrence.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  decode, network, summarize, neighborsOf, todayNeighbors, outcomeOf, labelClassOf, nodeColors, classCounts,
  rawOf, methodOf, timeColor, OUTCOME_CLASSES,
} from '../src/lib/recurrence.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const FIX = path.join(ROOT, 'tests', 'fixtures', 'recurrence');
const read = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const index = read(path.join(FIX, 'index.json'));
const expected = read(path.join(ROOT, 'tests', 'fixtures', 'recurrence-expected.json'));
const method = methodOf(index);

/** Python の round_summary（小数4桁）と比べる。小数は 1e-4 まで、整数・null・真偽はそのまま */
function near(got, want, where) {
  if (want === null || typeof want !== 'object') {
    if (typeof want === 'number' && !Number.isInteger(want)) {
      assert.ok(Number.isFinite(got) && Math.abs(got - want) <= 1e-4 + 1e-12, `${where}: ${got} ≠ ${want}`);
    } else if (typeof want === 'number' && typeof got === 'number' && !Number.isInteger(got)) {
      assert.ok(Math.abs(got - want) <= 1e-4 + 1e-12, `${where}: ${got} ≠ ${want}`);
    } else {
      assert.equal(got, want, where);
    }
    return;
  }
  if (Array.isArray(want)) {
    assert.equal(got.length, want.length, `${where}: 長さ`);
    want.forEach((w, i) => near(got[i], w, `${where}[${i}]`));
    return;
  }
  assert.deepEqual(Object.keys(got).sort(), Object.keys(want).sort(), `${where}: 項目`);
  for (const k of Object.keys(want)) near(got[k], want[k], `${where}.${k}`);
}

test('フィクスチャの要約が Python と一致する（辺の密度ごと）', () => {
  const files = index.days.flatMap((d) => d.candidates.map((c) => c.file));
  assert.deepEqual(files.sort(), Object.keys(expected).sort());
  for (const f of files) {
    const p = read(path.join(FIX, f));
    const dec = decode(p, method.labels);
    assert.equal(dec.dates[dec.n - 1], p.date, `${f}: 最後のノードが予測日`);
    for (const [bp, want] of Object.entries(expected[f])) {
      const net = network(dec, method.theiler, Number(bp));
      near(summarize(dec, net, method.theiler, Number(bp), { target: method.target, labels: method.labels }), want, `${f} ${bp}bp`);
    }
    near(p.summary, expected[f][String(method.rrBp)], `${f} ファイルの要約`);
  }
});

const tiny = (xs, extra = {}) => {
  const dims = 9;
  const z = [];
  xs.forEach((x) => { z.push(x); for (let d = 1; d < dims; d += 1) z.push(0); });
  const n = xs.length;
  return decode({
    n, features: Array(dims).fill('f'), z, q: 1, start: '2026-01-05', gaps: Array(n - 1).fill(1),
    ret: Array(n).fill(null), l12: Array(n).fill(null), l15: Array(n).fill(null), fresh: [], close: Array(n).fill(100),
    ...extra,
  });
};

test('辺: ε² は m 番目・同じ値は全部・タイラー窓の中は結ばない', () => {
  // test_recurrence_net.TestSummary.test_hand_example と同じ例
  const dec = tiny([0, 10, 1, 30, 2, 31, 9, 0], {
    ret: [10, -3, null, 2.5, 10, -12, 0, null],
    l12: [1, 0, null, 0, 1, 0, 0, null], l15: [1, 0, null, 0, 0, 0, 0, null], fresh: [0, 4, 5, 6],
  });
  const net = network(dec, 1, 3333);
  assert.equal(net.pairs, 21);
  assert.equal(net.m, 7);
  assert.equal(net.t, 4);
  assert.equal(net.nEdges, 8);
  const pairs = [];
  for (let e = 0; e < net.edges.length; e += 2) pairs.push(`${net.edges[e]}-${net.edges[e + 1]}`);
  assert.deepEqual(pairs.sort(), ['0-2', '0-4', '0-7', '1-6', '2-4', '2-7', '3-5', '4-7']);
  assert.deepEqual(Array.from(net.deg), [3, 1, 3, 1, 3, 1, 1, 3]);
  const s = summarize(dec, net, 1, 3333);
  assert.equal(s.deg, 3);
  assert.equal(s.degPct, 75);
  assert.deepEqual(s.nbr, { n: 3, known: 2, hit: 1, retMed: 10, l12: 1, l12N: 2, l15: 0.5, l15N: 2 });
  assert.equal(s.base.known, 6);
  assert.equal(s.base.retMed, 1.25);
  assert.deepEqual(s.nearestFresh.map((r) => [r.i, r.inEps, r.date]),
    [[0, true, '2026-01-05'], [4, true, '2026-01-09'], [5, false, '2026-01-10']]);
  // 隣接の一覧・今日の近傍（近い順）
  const nb = neighborsOf(net, dec.n);
  assert.deepEqual(Array.from(nb.of(7)).sort(), [0, 2, 4]);
  assert.deepEqual(todayNeighbors(dec, net, 1).map((r) => r.i), [0, 2, 4]);
});

test('四捨五入は 0.5 を切り上げる・結べる組が無ければ辺なし', () => {
  const xs = Array.from({ length: 27 }, (_, i) => (i * 7) % 11);
  const net = network(tiny(xs), 2, 50);    // 組 300 × 50bp = 1.5 → 2
  assert.equal(net.pairs, 300);
  assert.equal(net.m, 2);
  const none = network(tiny([0, 1, 2]), 20, 100);
  assert.deepEqual([none.t, none.pairs, none.nEdges], [-1, 0, 0]);
  const s = summarize(tiny([0, 1, 2]), none, 20, 100);
  assert.equal(s.deg, 0);
  assert.equal(s.eps, null);
  assert.equal(s.nbr.hit, null);
});

test('日付は start に gaps の暦日を足す', () => {
  const dec = decode({
    n: 4, features: ['a'], z: [0, 1, 2, 3], q: 20, start: '2026-10-08', gaps: [1, 4, 1],
    ret: [null, null, null, null], l12: [null, null, null, null], l15: [null, null, null, null], fresh: [], close: [1, 2, 3, 4],
  });
  assert.deepEqual(dec.dates, ['2026-10-08', '2026-10-09', '2026-10-13', '2026-10-14']);
  assert.throws(() => decode({ n: 2, features: ['a', 'b'], z: [0, 1, 2], start: '2026-01-01', gaps: [1] }), /状態の数/);
});

test('色の区分', () => {
  assert.equal(outcomeOf(10), 'hit');
  assert.equal(outcomeOf(9.9), 'up');
  assert.equal(outcomeOf(0), 'up');
  assert.equal(outcomeOf(-0.1), 'down');
  assert.equal(outcomeOf(-10), 'down');
  assert.equal(outcomeOf(-10.1), 'bad');
  assert.equal(outcomeOf(NaN), 'na');
  assert.equal(labelClassOf(1), 'pos');
  assert.equal(labelClassOf(0), 'neg');
  assert.equal(labelClassOf(NaN), 'na');
  const dec = tiny([0, 1, 2, 3], { ret: [10, 3, -4, null], l15: [1, 0, null, null] });
  assert.deepEqual(classCounts(dec, 'outcome'), { hit: 1, up: 1, down: 1, na: 1 });
  assert.deepEqual(classCounts(dec, 'l15'), { pos: 1, neg: 1, na: 2 });
  const of = Object.fromEntries(OUTCOME_CLASSES.map((c) => [c.id, c.color]));
  assert.deepEqual(nodeColors(dec, 'outcome'), [of.hit, of.up, of.down, of.na]);
  assert.equal(nodeColors(dec, 'time')[0], timeColor(0));
  assert.equal(timeColor(0), 'rgb(30, 58, 138)');
  assert.equal(timeColor(1), 'rgb(219, 234, 254)');
});

test('丸めた z から生の値の目安へ戻す', () => {
  const pos = { kind: 'pos', lo: 0, hi: 100, s: null, mean: 50, std: 10 };
  assert.equal(rawOf(pos, 20, 20), 60);
  assert.equal(rawOf(pos, 200, 20), 100);              // 切った値より外には出ない
  const heavy = { kind: 'pos_heavy', lo: null, hi: 50, s: null, mean: Math.log1p(4), std: 0.5 };
  assert.ok(Math.abs(rawOf(heavy, 0, 20) - 4) < 1e-12);
  const sh = { kind: 'signed_heavy', lo: -30, hi: 30, s: 3, mean: 0, std: 1 };
  assert.ok(Math.abs(rawOf(sh, 20, 20) - Math.sinh(1) * 3) < 1e-12);
  // フィクスチャの今日の値: 丸めの誤差の中で生の値に戻る
  const f = index.days[0].candidates[0].file;
  const p = read(path.join(FIX, f));
  const dec = decode(p, method.labels);
  dec.scale.forEach((sc, d) => {
    const back = rawOf(sc, dec.z[(dec.n - 1) * dec.dims + d], dec.q);
    const step = Math.abs(rawOf(sc, dec.z[(dec.n - 1) * dec.dims + d] + 1, dec.q) - back) + 1e-9;
    assert.ok(Math.abs(back - p.raw[d]) <= step, `${sc.col}: ${back} vs ${p.raw[d]}`);
  });
});
