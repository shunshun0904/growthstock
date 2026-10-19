/**
 * リカレンス・ネットワーク（src/lib/recurrence.js）の単体テスト。
 *
 * - 近傍・線・要約は Python（research/recurrence_net.py）と同じ: tests/fixtures/recurrence/ の候補のファイルから、k ごとに数えた
 *   要約が tests/fixtures/recurrence-expected.json（Python が数えた値。tests/test_recurrence_net.py が同じ値を確かめる）と一致する
 * - 近傍は各ノードの、タイラー窓より離れた日のうち距離の近い k 日（同じ距離は古い日が先。同じ距離がたくさんある例で素朴な並べ方と
 *   一致）。線はどちらかが選べば結ぶ。手で数えた小さな例（Python の test_hand_example_k2・k3 と同じ）
 * - 日付は start に gaps の暦日を足していく（最後の日が予測日）
 * - 色の区分（+10% 到達 / 0〜+10 / −10〜0 / −10 未満 / 未確定、正例 / 負例 / 未確定）と、丸めた z から生の値の目安への戻し方
 * - 先20営業日の値動きの色（最高値までの上昇率の5段、20営業日目の騰落率の4段。区切りは下の端を含む。最安値までの色は運用者の
 *   指定で外した。値はツールチップに出すので読む）。値の無いファイル（版 1）は未確定、終値で測っていた版 2 の上昇率・下落率は
 *   読まない。今日に似た日だけの件数。最高値までと時期の色は青白赤（低い・古い = 青 → 白 → 高い・新しい = 赤。運用者の選択）
 * - 今日に似た日の表（todayNeighbors）は図で光らせる日（neighborsOf の今日）と同じ日・同じ順（運用者の指定「プロット上の近傍の
 *   日付で統一」。要約に「今日に近い過去の更新日」は無い）
 *
 *   node --test tests/recurrence.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  decode, knnLists, network, summarize, neighborsOf, todayNeighbors, periodsOf, outcomeOf, labelClassOf, nodeColors, classCounts,
  rawOf, methodOf, timeColor, riseOf, classesOf, OUTCOME_CLASSES, RISE_CLASSES, R20_CLASSES,
  NA_VALUE, COLOR_MODES, BWR, TIME_STOPS,
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

test('フィクスチャの要約が Python と一致する（k ごと）', () => {
  const files = index.days.flatMap((d) => d.candidates.map((c) => c.file));
  assert.deepEqual(files.sort(), Object.keys(expected).sort());
  assert.deepEqual(method.kChoices, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(method.k, 5);                 // 最初の k（運用者の指定「初期状態のkは5でお願いします」）
  for (const f of files) {
    const p = read(path.join(FIX, f));
    const dec = decode(p, method.labels);
    assert.equal(dec.dates[dec.n - 1], p.date, `${f}: 最後のノードが予測日`);
    const lists = knnLists(dec, method.theiler, Math.max(...method.kChoices));
    assert.deepEqual(Object.keys(expected[f]), method.kChoices.map(String));
    for (const [k, want] of Object.entries(expected[f])) {
      const net = network(dec, method.theiler, Number(k), lists);
      const s = summarize(dec, net, method.theiler, Number(k), { target: method.target, labels: method.labels });
      near(s, want, `${f} k=${k}`);
      // 表に出す今日に似た日は、図で光らせる日（今日の近い k 日）と同じ日・同じ順で、数は要約の deg と同じ
      const table = todayNeighbors(dec, net).map((r) => r.i);
      assert.deepEqual(table, Array.from(neighborsOf(net).of(dec.n - 1)), `${f} k=${k}: 表と図の似た日`);
      assert.equal(table.length, s.deg, `${f} k=${k}: 表の日数`);
      // 一度にまとめて数えた近傍（kMax = 50）から切っても、その k だけで数えても同じ
      const alone = network(dec, method.theiler, Number(k));
      assert.deepEqual(Array.from(alone.edges), Array.from(net.edges), `${f} k=${k}: 線`);
    }
    near(p.byK, expected[f], `${f} ファイルの要約`);
  }
  // 目次の一覧の要約もファイルと同じ
  for (const c of index.days.flatMap((d) => d.candidates)) near(c.byK, expected[c.file], `${c.file} 目次の要約`);
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

const hand = () => tiny([0, 10, 1, 30, 2, 31, 9, 0], {
  ret: [10, -3, null, 2.5, 10, -12, 0, null],
  l12: [1, 0, null, 0, 1, 0, 0, null], l15: [1, 0, null, 0, 0, 0, 0, null], fresh: [0, 4, 5, 6],
});

test('k近傍: 手で数えた例（Python の test_hand_example_k2 と同じ）', () => {
  const dec = hand();
  const net = network(dec, 1, 2);
  // 近い2日: 0→7,2 / 1→6,4 / 2→0,4 / 3→5,1 / 4→2,0 / 5→3,1 / 6→1,4 / 7→0,2。どちらかが選べば1本
  const pairs = [];
  for (let e = 0; e < net.edges.length; e += 2) pairs.push(`${net.edges[e]}-${net.edges[e + 1]}`);
  assert.deepEqual(pairs, ['0-2', '0-4', '0-7', '1-3', '1-4', '1-5', '1-6', '2-4', '2-7', '3-5', '4-6']);
  assert.deepEqual(Array.from(net.deg), [3, 4, 3, 2, 4, 2, 2, 2]);
  const s = summarize(dec, net, 1, 2);
  assert.deepEqual([s.k, s.deg, s.kth2, s.kth, s.periods, s.edges, s.isolated, s.degMed], [2, 2, 1, 1, 2, 11, 0, 2.5]);
  assert.equal(s.kthPct, (100 * 1.5) / 8);                  // 各ノードの2番目の距離 1・64・1・400・4・441・49・1
  assert.deepEqual(s.nbr, { n: 2, known: 1, hit: 1, retMed: 10, l12: 1, l12N: 1, l15: 1, l15N: 1 });
  assert.equal(s.nbrFresh.n, 1);
  assert.equal(s.base.known, 6);
  assert.equal(s.base.retMed, 1.25);
  // 似た日は近い k 日だけ（更新日だけから選んだ近い日の一覧は持たない。Python の要約と同じ項目）
  assert.deepEqual(Object.keys(s).sort(), ['base', 'baseFresh', 'deg', 'degMed', 'edges', 'isolated', 'k', 'kth', 'kth2', 'kthPct',
    'nbr', 'nbrFresh', 'periods', 'theiler']);
  assert.deepEqual(Array.from(neighborsOf(net).of(7)), [0, 2]);           // 今日の似た日（近い順）
  assert.deepEqual(Array.from(neighborsOf(net).of(1)), [6, 4]);           // 線は 1-3・1-5 もあるが、1 が選んだのは 6・4
  assert.deepEqual(todayNeighbors(dec, net).map((r) => [r.i, r.d]), [[0, 0], [2, 1]]);
  // k = 3 は版 3 の ε の例（ε² = 4）と同じ近傍 0・2・4
  const s3 = summarize(dec, network(dec, 1, 3), 1, 3);
  assert.deepEqual([s3.deg, s3.kth2, s3.periods], [3, 4, 3]);
  assert.deepEqual(s3.nbr, { n: 3, known: 2, hit: 1, retMed: 10, l12: 1, l12N: 2, l15: 0.5, l15N: 2 });
  assert.equal(s3.nbrFresh.n, 2);                                         // 近い3日 0・2・4 のうち更新日（0・4）
  assert.deepEqual(todayNeighbors(dec, network(dec, 1, 3)).map((r) => r.i), [0, 2, 4]);
});

/** 決まった種の乱数（-r〜r の整数）。 */
function ints(seed, count, r) {
  let a = seed >>> 0;
  const out = [];
  for (let x = 0; x < count; x += 1) {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    out.push((a % (2 * r + 1)) - r);
  }
  return out;
}

test('k近傍: 素朴な並べ方と一致する（同じ距離がたくさんある例・結べる日が k より少ない例）', () => {
  for (const [seed, n, r, theiler, k] of [[1, 50, 2, 3, 12], [2, 70, 40, 5, 12], [3, 30, 1, 0, 7], [4, 6, 5, 2, 5]]) {
    const z = ints(seed, n * 9, r);
    const dec = decode({
      n, features: Array(9).fill('f'), z, q: 1, start: '2026-01-05', gaps: Array(n - 1).fill(1),
      ret: Array(n).fill(null), l12: Array(n).fill(null), l15: Array(n).fill(null), fresh: [], close: Array(n).fill(1),
    });
    const lists = knnLists(dec, theiler, k);
    const dist = (i, j) => { let acc = 0; for (let d = 0; d < 9; d += 1) acc += (z[i * 9 + d] - z[j * 9 + d]) ** 2; return acc; };
    const want = [];
    for (let i = 0; i < n; i += 1) {
      const cand = [];
      for (let j = 0; j < n; j += 1) if (Math.abs(i - j) > theiler) cand.push(j);
      cand.sort((a, b) => (dist(i, a) - dist(i, b)) || (a - b));
      want.push(cand.slice(0, k));
      assert.equal(lists.cnt[i], Math.min(k, cand.length), `cnt ${seed} ${i}`);
      assert.deepEqual(Array.from(lists.nbr.subarray(i * k, i * k + lists.cnt[i])), want[i], `並び ${seed} ${i}`);
      assert.deepEqual(Array.from(lists.d2.subarray(i * k, i * k + lists.cnt[i])), want[i].map((j) => dist(i, j)));
    }
    for (const kk of [1, Math.min(4, k), k]) {
      const net = network(dec, theiler, kk, lists);
      const has = new Set();
      for (let e = 0; e < net.edges.length; e += 2) {
        assert.ok(net.edges[e] < net.edges[e + 1]);
        has.add(`${net.edges[e]}-${net.edges[e + 1]}`);
      }
      for (let i = 0; i < n; i += 1) {
        for (let j = i + 1; j < n; j += 1) {
          const want1 = want[i].slice(0, kk).includes(j) || want[j].slice(0, kk).includes(i);
          assert.equal(has.has(`${i}-${j}`), want1, `線 ${seed} k=${kk} ${i}-${j}`);
        }
      }
    }
  }
  // 6ノード・タイラー窓 2: 結べる日は 3・2・1・1・2・3 日
  const six = decode({
    n: 6, features: ['f'], z: [0, 1, 2, 3, 4, 5], q: 1, start: '2026-01-05', gaps: [1, 1, 1, 1, 1],
    ret: Array(6).fill(null), l12: Array(6).fill(null), l15: Array(6).fill(null), fresh: [], close: Array(6).fill(1),
  });
  assert.deepEqual(Array.from(knnLists(six, 2, 5).cnt), [3, 2, 1, 1, 2, 3]);
  // 結べる日が無いほど短い: 似た日なし・要約は空
  const few = tiny([0, 1, 2]);
  const none = network(few, 20, 10);
  assert.deepEqual([none.nEdges, none.lists.cnt[2]], [0, 0]);
  const s = summarize(few, none, 20, 10);
  assert.deepEqual([s.deg, s.kth2, s.kth, s.kthPct, s.periods, s.isolated], [0, -1, null, null, 0, 3]);
  assert.equal(s.nbr.hit, null);
  assert.deepEqual(todayNeighbors(few, none), []);
  assert.equal(periodsOf([40, 1, 20, 22, 70], 20), 2);      // 1・20・22・40 は隣との差が20以下でつながり、70 は30離れる
  assert.equal(periodsOf([0, 21, 42], 20), 3);              // 差が 21 ずつ（20 より離れる）なら3つ
  assert.equal(periodsOf([], 20), 0);
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
  // 時期は青白赤（運用者の選択 2026-10-19: 古い = 青・真ん中 = 白・新しい = 赤）
  assert.deepEqual(TIME_STOPS, BWR);
  assert.equal(timeColor(0), 'rgb(58, 123, 240)');          // #3a7bf0
  assert.equal(timeColor(0.5), 'rgb(236, 236, 236)');       // #ececec
  assert.equal(timeColor(1), 'rgb(227, 70, 63)');           // #e3463f
  assert.deepEqual(nodeColors(dec, 'time').at(-1), timeColor(1));
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

test('先20営業日の値動きの色（最高値まで・20営業日目）', () => {
  // 最高値までの上昇率: 0% 未満 / 0〜5 / 5〜10 / 10〜20 / 20% 以上（下の端を含む）
  assert.deepEqual([-0.1, 0, 4.9, 5, 9.9, 10, 19.9, 20, 85, NaN].map(riseOf),
    ['r0', 'r5', 'r5', 'r10', 'r10', 'r20', 'r20', 'r20p', 'r20p', 'na']);
  const dec = tiny([0, 1, 2, 3, 4], {
    v: 3, rise: [25, 12, 3, -1, null], fall: [0.5, -3, -7, -25, null], r20: [15, 4, -2, -12, null],
  });
  assert.deepEqual(classCounts(dec, 'rise'), { r20p: 1, r20: 1, r5: 1, r0: 1, na: 1 });
  assert.deepEqual(classCounts(dec, 'r20'), { hit: 1, up: 1, down: 1, bad: 1, na: 1 });
  assert.deepEqual(classCounts(dec, 'rise', 10, [0, 1]), { r20p: 1, r20: 1 });      // 渡したノードだけ数える
  assert.deepEqual(classCounts(dec, 'time'), {});
  const color = (cl) => Object.fromEntries(cl.map((c) => [c.id, c.color]));
  assert.deepEqual(nodeColors(dec, 'rise'), ['r20p', 'r20', 'r5', 'r0', 'na'].map((k) => color(RISE_CLASSES)[k]));
  assert.deepEqual(nodeColors(dec, 'r20'), ['hit', 'up', 'down', 'bad', 'na'].map((k) => color(R20_CLASSES)[k]));
  // 20営業日目は結果と同じ区切り・同じ色（未確定の色だけ値の色の付け方にそろえる）
  for (const c of R20_CLASSES.filter((x) => x.id !== 'na')) assert.equal(c.color, color(OUTCOME_CLASSES)[c.id]);
  for (const cl of [RISE_CLASSES, R20_CLASSES]) assert.equal(color(cl).na, NA_VALUE);
  assert.equal(classesOf('rise'), RISE_CLASSES);
  // 最高値までは青白赤（低い = 青 → 5〜10% = 白 → +10% の指値に届いた段 = 赤）
  assert.deepEqual(['r0', 'r5', 'r10', 'r20', 'r20p'].map((k) => color(RISE_CLASSES)[k]), BWR);
  assert.deepEqual(BWR, ['#3a7bf0', '#86b0f2', '#ececec', '#f2988e', '#e3463f']);
  assert.equal(classesOf('time'), null);
  assert.deepEqual(COLOR_MODES.filter((m) => m.value).map((m) => m.value), ['rise', 'r20']);
  // 最安値までの色は外した（運用者の指定「安値の色付けは、なしで良いです」）。値は読む（ツールチップ・選んだ日）
  assert.ok(!COLOR_MODES.some((m) => m.id === 'fall'));
  assert.deepEqual(Array.from(dec.fall).slice(0, 4), [0.5, -3, -7, -25]);
  // 値の無いファイル（版 1）はすべて未確定。版 2 の上昇率・下落率（終値で測った別の物差し）は読まず、20営業日目は同じなので読む
  const old = tiny([0, 1, 2]);
  assert.ok([...old.rise, ...old.fall, ...old.r20].every(Number.isNaN));
  const v2 = tiny([0, 1, 2], { v: 2, rise: [5, 6, 7], fall: [-1, -2, -3], r20: [1, 2, 3] });
  assert.ok([...v2.rise, ...v2.fall].every(Number.isNaN));
  assert.deepEqual(Array.from(v2.r20), [1, 2, 3]);
  // フィクスチャ（版 6。値は版 3 と同じ高値・安値の物差し）には値がある。最後の20営業日は未確定、分かる日は
  // 最安値 ≤ 20営業日目の終値 ≤ 最高値（0.1% の丸めの幅）
  const p = read(path.join(FIX, index.days[0].candidates[0].file));
  const fx = decode(p, method.labels);
  assert.equal(p.v, 6);
  assert.ok(Array.from(fx.rise.slice(-20)).every(Number.isNaN));
  // （実データでは20営業日目に売買が無い日は 20営業日目だけが分からない。最大・最小はその日を飛ばして数える）
  let known = 0;
  for (let i = 0; i < fx.n; i += 1) {
    if (Number.isFinite(fx.rise[i])) known += 1;
    if (![fx.rise[i], fx.fall[i], fx.r20[i]].every(Number.isFinite)) continue;
    assert.ok(fx.fall[i] <= fx.r20[i] + 0.1 + 1e-9 && fx.r20[i] <= fx.rise[i] + 0.1 + 1e-9, `${i}: ${fx.fall[i]} ${fx.r20[i]} ${fx.rise[i]}`);
    if (fx.rise[i] >= 10) assert.equal(fx.ret[i], 10, `${i}: 最高値まで ${fx.rise[i]} なら +10% の指値に届いている`);
  }
  assert.equal(known, fx.n - 20);
});
