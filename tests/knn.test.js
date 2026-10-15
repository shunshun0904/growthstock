/**
 * 候補の比較（src/lib/knn.js）の単体テスト。
 *
 * - 候補の行: 今の並べ方（1.5σ の3本の最小）の順に A, B, C…。M は先頭、Q は近傍の25%点の最上位（同点は先頭に近いほう）
 * - 範囲: pass = 95以上 / near = 85以上 / all = 全部。近傍の集計の無い候補は knn が null
 * - 状態: knn の無いファイル・集計の無い日・作れなかった日・母集団が足りない日・候補が無い・範囲に入らない
 * - 座標と表の補助
 *
 *   node --test tests/knn.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KNN_MODES, modeMin, knnDay, labelOf, knnRows, knnStatus, lin, niceTicks, scatterFrame, cloudLayout,
  scatterTable, stripTable, cloudTable,
} from '../src/lib/knn.js';
import { RULE_MODELS } from '../src/lib/strategy.js';

const cand = (code, [a, b, c], score = 0.5) => ({
  code, jqCode: `${code}0`, name: `銘柄${code}`, date: '2026-10-15', score,
  byModel: Object.fromEntries([[RULE_MODELS[0], a], [RULE_MODELS[1], b], [RULE_MODELS[2], c]]
    .filter(([, v]) => v !== undefined).map(([k, v]) => [k, { score, pctHistorical: v }])),
});
const est = (p25, bad = 4, extra = {}) => ({
  min15: 96, est: { p10: p25 - 3, p25, p50: p25 + 4, p75: 10, p90: 10, hit: 40, bad, n: 50 },
  dNn: { R: 0.9, V: 0.8 }, cloud: Array.from({ length: 7 }, () => [0, 0, 0, 0, 0]), ...extra,
});
const BANDS = ['-10 未満', '-10〜0', '0〜+5', '+5〜+10 未満', '+10 到達'];

const cands = [
  cand('1111', [96, 97, 98], 0.9),    // 最小 96 → B（最小の順）
  cand('2222', [99, 99, 99], 0.5),    // 最小 99 → A（M）
  cand('3333', [88, 95, 99], 0.7),    // 最小 88 → 惜しい
  cand('4444', [30, 80, 90], 0.8),    // 最小 30
  cand('5555', [99, undefined, 99]),  // 欠け → 最小なし（後ろ）
];
const block = {
  status: 'ok', poolN: 1300, n: 4,
  candidates: { 22220: est(-2.7), 11110: est(0.3, 2), 33330: est(-5.0, 8), 44440: est(1.5, 0) },
};
const data = { knn: { minPool: 100, byDay: { '2026-10-15': block } } };

test('KNN_MODES / modeMin: 95以上・85以上・全部', () => {
  assert.deepEqual(KNN_MODES.map((m) => m.id), ['pass', 'near', 'all']);
  assert.equal(modeMin('pass'), 95);
  assert.equal(modeMin('near'), 85);
  assert.equal(modeMin('all'), null);
});

test('knnDay: その日の集計。無ければ null', () => {
  assert.equal(knnDay(data, '2026-10-15'), block);
  assert.equal(knnDay(data, '2026-10-14'), null);
  assert.equal(knnDay({}, '2026-10-15'), null);
  assert.equal(knnDay(null, '2026-10-15'), null);
});

test('labelOf: A〜Z のあとは A2, B2…', () => {
  assert.equal(labelOf(0), 'A');
  assert.equal(labelOf(25), 'Z');
  assert.equal(labelOf(26), 'A2');
  assert.equal(labelOf(27), 'B2');
});

test('knnRows: 今の並べ方の順に印を付け、M は先頭、Q は近傍の25%点の最上位', () => {
  const rows = knnRows(cands, block, 'pass');
  assert.deepEqual(rows.map((r) => [r.label, r.code]), [['A', '2222'], ['B', '1111']]);
  assert.equal(rows[0].pickedM, true);
  assert.equal(rows[1].pickedM, false);
  assert.deepEqual(rows.map((r) => r.pickedQ), [false, true]);     // 1111 の25%点 +0.3 > 2222 の −2.7
  assert.equal(rows[1].knn.est.p25, 0.3);
});

test('knnRows: 範囲を広げると惜しい候補・全候補が入る。集計の無い候補は knn が null', () => {
  const near = knnRows(cands, block, 'near');
  assert.deepEqual(near.map((r) => r.code), ['2222', '1111', '3333']);
  const all = knnRows(cands, block, 'all');
  assert.deepEqual(all.map((r) => [r.label, r.code]),
                   [['A', '2222'], ['B', '1111'], ['C', '3333'], ['D', '4444'], ['E', '5555']]);
  assert.equal(all[4].knn, null);
  assert.deepEqual(all.map((r) => r.pickedQ), [false, false, false, true, false]);   // 4444 の +1.5 が最上位
  assert.equal(all[0].pickedM, true);
});

test('knnRows: 25%点が同点なら今の並べ方で先のほうが Q。集計の無い日は全員 knn が null で Q も無い', () => {
  const tie = { status: 'ok', candidates: { 22220: est(1.0), 11110: est(1.0) } };
  const rows = knnRows(cands, tie, 'pass');
  assert.deepEqual(rows.map((r) => r.pickedQ), [true, false]);
  const none = knnRows(cands, null, 'pass');
  assert.deepEqual(none.map((r) => r.knn), [null, null]);
  assert.deepEqual(none.map((r) => r.pickedQ), [false, false]);
  assert.deepEqual(knnRows([], block, 'all'), []);
});

test('knnStatus: 描けない理由を区別する', () => {
  assert.equal(knnStatus({}, null, [], 'pass').kind, 'none');
  assert.equal(knnStatus(data, null, [], 'pass').kind, 'noday');
  assert.equal(knnStatus(data, { status: 'unavailable', reason: 'FileNotFoundError: x' }, [], 'pass').kind, 'unavailable');
  assert.match(knnStatus(data, { status: 'unavailable', reason: 'FileNotFoundError: x' }, [], 'pass').text, /FileNotFoundError/);
  const small = knnStatus(data, { status: 'pool_small', poolN: 40 }, [], 'pass');
  assert.equal(small.kind, 'pool_small');
  assert.match(small.text, /40 件/);
  assert.equal(knnStatus(data, { status: 'no_candidates' }, [], 'pass').kind, 'empty');
  const empty = knnStatus(data, block, [], 'pass');
  assert.equal(empty.kind, 'empty');
  assert.match(empty.text, /規則を満たす候補（95以上）はありません/);
  assert.match(empty.text, /惜しい候補（85〜95）まで/);
  assert.equal(knnStatus(data, block, [], 'all').text, 'この日は候補がありません。');
  assert.equal(knnStatus(data, block, knnRows(cands, { status: 'ok', candidates: {} }, 'pass'), 'pass').kind, 'empty');
  assert.equal(knnStatus(data, block, knnRows(cands, block, 'pass'), 'pass').kind, 'ok');
});

test('lin / niceTicks: 線形の尺度と見やすい目盛り', () => {
  const x = lin(0, 10, 100, 200);
  assert.equal(x(0), 100);
  assert.equal(x(5), 150);
  assert.equal(x(10), 200);
  assert.equal(lin(3, 3, 0, 10)(3), 5);                 // 幅が無ければ真ん中
  assert.deepEqual(niceTicks(-10.5, 1.5, 6), [-10, -8, -6, -4, -2, 0]);
  assert.deepEqual(niceTicks(0, 16, 4), [0, 5, 10, 15]);
  assert.deepEqual(niceTicks(5, 5, 4), []);
});

test('scatterFrame: 文脈の格子と候補を収める。文脈が無ければ候補だけ', () => {
  const rows = knnRows(cands, block, 'all');
  const cx = { density: { xEdges: [-9.9, -6.2, -5.3, -4.7, -3.8, -3.1, 0.8] } };
  const f = scatterFrame(rows, cx);
  assert.equal(f.lo, -10.4);
  assert.equal(f.hi, 2.0);                               // 候補の +1.5 が格子の右端 0.8 より右
  assert.equal(f.ytop, 16);
  const g = scatterFrame(rows, null);
  assert.equal(g.lo, -5.5);
  assert.equal(g.hi, 2.0);
  const h = scatterFrame([], null);
  assert.deepEqual([h.lo, h.hi, h.ytop], [-10.5, 10.5, 16]);
});

test('cloudLayout: 560px なら1行に2枚', () => {
  assert.deepEqual(cloudLayout(5).perRow, 2);
  assert.equal(cloudLayout(5).rows, 3);
  assert.equal(cloudLayout(1).perRow, 1);
  assert.equal(cloudLayout(0).rows, 1);
});

test('表: 散布図・幅・雲の行', () => {
  const rows = knnRows(cands, block, 'all');
  const st = scatterTable(rows);
  assert.equal(st.length, 5);
  assert.deepEqual(st[0].slice(0, 4), ['A（M）', '2222', '銘柄2222', '99.0']);
  assert.equal(st[3][0], 'D（Q）');
  assert.deepEqual(st[4].slice(3), ['—', '—', '—', '—', '—', '—']);      // 集計の無い候補
  const sp = stripTable(rows);
  assert.deepEqual(sp[0], ['A', '2222', '-5.70', '-2.70', '1.30', '10.00', '10.00', '40%', '4%']);
  const cl = cloudTable(rows, BANDS);
  assert.equal(cl.length, 4 * BANDS.length);                             // 集計の無い E は出ない
  assert.equal(cl[0][0], 'A・+10 到達');
  assert.equal(cl[0].length, 1 + 7);
});
