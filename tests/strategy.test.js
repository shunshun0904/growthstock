/**
 * 運用の戦略（src/lib/strategy.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BOOST, STRATEGY, boostPcts, minPct, passesAgree, dayVerdict, strategySignal, exitPlan,
  freeSlots,
} from '../src/lib/strategy.js';

const cand = (code, pcts, score = 0.5) => ({
  code, jqCode: `${code}0`, name: code, score, close: 1000,
  byModel: Object.fromEntries(
    Object.entries(pcts).map(([a, p]) => [a, { score, pctHistorical: p }])),
});

const day = (n, pcts = [95, 95, 95]) =>
  Array.from({ length: n }, (_, i) =>
    cand(`X${i}`, { lgbm: pcts[0], xgb: pcts[1], cat: pcts[2] }, 0.5 - i * 0.01));

/* ------------------------------------------------ 3モデルの百分位 */

test('minPct: 3モデルの最小を返す', () => {
  assert.equal(minPct(cand('A', { lgbm: 95, xgb: 91, cat: 99 })), 91);
});

test('minPct: 線形・MLP は混ぜない', () => {
  const c = cand('A', { lgbm: 95, xgb: 91, cat: 99 });
  c.byModel.logit = { score: 0.1, pctHistorical: 10 };
  c.byModel.mlp = { score: 0.1, pctHistorical: 5 };
  assert.equal(minPct(c), 91);
  assert.deepEqual(Object.keys(boostPcts(c)), BOOST);
});

test('minPct: 1つでも欠けていれば null（欠測を「満たした」にしない）', () => {
  const c = cand('A', { lgbm: 95, cat: 99 });
  assert.equal(minPct(c), null);
  assert.equal(passesAgree(c), false);
  assert.equal(minPct({}), null);
});

test('passesAgree: 3つすべてが 90 以上のときだけ true', () => {
  assert.equal(passesAgree(cand('A', { lgbm: 90, xgb: 90, cat: 90 })), true);
  assert.equal(passesAgree(cand('B', { lgbm: 99, xgb: 99, cat: 89.9 })), false);
});

/* ------------------------------------------------ その日を取るか */

test('dayVerdict: 発火数で3段階に分かれる', () => {
  assert.equal(dayVerdict(26).kind, 'strong');
  assert.equal(dayVerdict(20).kind, 'strong');
  assert.equal(dayVerdict(19).kind, 'ok');
  assert.equal(dayVerdict(8).kind, 'ok');
  assert.equal(dayVerdict(7).kind, 'skip');
  assert.equal(dayVerdict(0).kind, 'none');
  assert.equal(dayVerdict(null).kind, 'none');
});

/* ------------------------------------------------ シグナル */

test('strategySignal: 発火が多い日は上位2件を買う', () => {
  const rows = day(25);
  rows[3].byModel.lgbm.pctHistorical = 99;   // 最小は 95 のまま
  const s = strategySignal(rows);
  assert.equal(s.nBreak, 25);
  assert.equal(s.verdict.kind, 'strong');
  assert.equal(s.passed.length, 25);
  assert.equal(s.picks.length, 2);
  assert.equal(s.buyable, true);
});

test('strategySignal: 3モデルの最小順位で並ぶ（lgbm 単体の順ではない）', () => {
  const rows = [
    cand('LOW', { lgbm: 99, xgb: 91, cat: 92 }, 0.9),   // 最小 91・スコア最大
    cand('TOP', { lgbm: 93, xgb: 96, cat: 95 }, 0.5),   // 最小 93
    cand('MID', { lgbm: 92, xgb: 92, cat: 99 }, 0.6),   // 最小 92
    ...day(20),
  ];
  const s = strategySignal(rows);
  assert.equal(s.picks[0].code, 'X0');                  // 最小 95 が先頭
  const order = s.passed.map((c) => c.code);
  assert.ok(order.indexOf('TOP') < order.indexOf('MID'));
  assert.ok(order.indexOf('MID') < order.indexOf('LOW'));
});

test('strategySignal: 発火が少ない日は買わない（基準を満たしても）', () => {
  const s = strategySignal(day(5));
  assert.equal(s.verdict.kind, 'skip');
  assert.equal(s.buyable, false);
  assert.equal(s.picks.length, 0);
  assert.equal(s.passed.length, 5);      // 満たした銘柄は見えるようにする
  assert.equal(s.held.length, 5);
});

test('strategySignal: 基準を満たす銘柄が無ければ買わない', () => {
  const s = strategySignal(day(25, [95, 89, 95]));
  assert.equal(s.verdict.kind, 'strong');
  assert.equal(s.passed.length, 0);
  assert.equal(s.picks.length, 0);
});

test('strategySignal: 候補ゼロでも壊れない', () => {
  for (const v of [[], null, undefined]) {
    const s = strategySignal(v);
    assert.equal(s.nBreak, 0);
    assert.equal(s.verdict.kind, 'none');
    assert.deepEqual(s.picks, []);
  }
});

test('strategySignal: byModel の無い古い JSON では誰も通らない', () => {
  const rows = Array.from({ length: 25 }, (_, i) => ({ code: `Y${i}`, score: 0.5 }));
  const s = strategySignal(rows);
  assert.equal(s.nBreak, 25);
  assert.equal(s.passed.length, 0);
});

/* ------------------------------------------------ 出口 */

test('exitPlan: 終値から指値の目安を出す', () => {
  const p = exitPlan(1000);
  assert.equal(p.target, 1200);
  assert.equal(p.takeProfit, STRATEGY.takeProfit);
  assert.equal(p.holdDays, 20);
  assert.equal(exitPlan(0), null);
  assert.equal(exitPlan(null), null);
});

test('STRATEGY: 同時保有の上限は3（実験34で決めた枠）', () => {
  assert.equal(STRATEGY.maxSlots, 3);
});

test('freeSlots: 保有数から空き枠を出す', () => {
  assert.equal(freeSlots(0), 3);
  assert.equal(freeSlots(2), 1);
  assert.equal(freeSlots(3), 0);
  // 上限を超えて持っていても負にはしない
  assert.equal(freeSlots(5), 0);
});

test('freeSlots: 保有数が分からなければ null（枠の話をしない）', () => {
  assert.equal(freeSlots(null), null);
  assert.equal(freeSlots(undefined), null);
  assert.equal(freeSlots(NaN), null);
  assert.equal(freeSlots(-1), null);
  // 玉の本数は整数。小数は入力の誤りなので黙って丸めない
  assert.equal(freeSlots(1.7), null);
});
