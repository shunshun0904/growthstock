/**
 * 運用の戦略（src/lib/strategy.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BOOST, STRATEGY, boostPcts, minPct, passesAgree, dayVerdict, strategySignal, exitPlan,
  freeSlots, pctSpread, laggard, nearMisses, fundContrib, frozenNote,
  BORDER, BORDER_STATS, borderShape, borderNote,
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

test('pctSpread / laggard: 3モデルの幅と最下位モデル', () => {
  // クニミネ工業 2026-09-18 の実際の値
  const k = cand('5388', { lgbm: 86.5, xgb: 91.7, cat: 91.5 });
  assert.equal(minPct(k), 86.5);
  assert.equal(Math.round(pctSpread(k) * 10) / 10, 5.2);
  assert.equal(laggard(k), 'lgbm');
  // 欠けていれば null（欠けたまま判定しない）
  const bad = cand('9999', { lgbm: 90, xgb: 90 });
  assert.equal(pctSpread(bad), null);
  assert.equal(laggard(bad), null);
});

test('nearMisses: 最小85〜90 だけを拾い、幅5以上は弱い形と印を付ける', () => {
  const rows = [
    cand('A', { lgbm: 95, xgb: 94, cat: 96 }),        // 基準通過。対象外
    cand('B', { lgbm: 86.5, xgb: 91.7, cat: 91.5 }),  // 惜しい・幅5.2 → 弱い
    cand('C', { lgbm: 88, xgb: 89, cat: 87 }),        // 惜しい・幅2 → 揃っている
    cand('D', { lgbm: 70, xgb: 95, cat: 95 }),        // 85未満。対象外
  ];
  const n = nearMisses(rows);
  assert.deepEqual(n.map((x) => x.code), ['C', 'B']);  // 最小の降順
  assert.equal(n.find((x) => x.code === 'B').weak, true);
  assert.equal(n.find((x) => x.code === 'C').weak, false);
  assert.equal(n.find((x) => x.code === 'B').laggard, 'lgbm');
});

test('nearMisses: 境界（85ちょうどは入る / 90ちょうどは入らない）', () => {
  assert.equal(nearMisses([cand('E', { lgbm: 85, xgb: 99, cat: 99 })]).length, 1);
  assert.equal(nearMisses([cand('F', { lgbm: 90, xgb: 99, cat: 99 })]).length, 0);
  assert.equal(nearMisses([cand('G', { lgbm: 84.9, xgb: 99, cat: 99 })]).length, 0);
});

test('fundContrib: 決算の寄与（水準＋変化）を足す', () => {
  const c = { contrib: { groups: { '決算（水準）': -0.0075, '決算（変化）': -0.1326,
                                   '株価・ブレイク': 0.4497 } } };
  assert.equal(Math.round(fundContrib(c) * 10000) / 10000, -0.1401);
  // 寄与が無い古い JSON では null（0 と混ぜない）
  assert.equal(fundContrib({}), null);
  assert.equal(fundContrib({ contrib: { groups: {} } }), null);
  // 片方しか無くても足せる
  assert.equal(fundContrib({ contrib: { groups: { '決算（変化）': 0.25 } } }), 0.25);
});

test('frozenNote: 直近20日の日次ボラが frozenVol 未満のときだけ「値動きなし」の注意を返す', () => {
  assert.equal(frozenNote({ vol20d: 0.08 }), '値動きなし（20日ボラ 0.08%。TOB 中の可能性）');
  assert.equal(frozenNote({ vol20d: 0.29 }), '値動きなし（20日ボラ 0.29%。TOB 中の可能性）');
  assert.equal(frozenNote({ vol20d: 0.3 }), null);       // ちょうどは出さない
  assert.equal(frozenNote({ vol20d: 1.2 }), null);
  assert.equal(frozenNote({}), null);                   // 値が無ければ出さない
  assert.equal(frozenNote({ vol20d: null }), null);
  assert.equal(frozenNote(undefined), null);
  assert.equal(frozenNote({ vol20d: 0.5 }, { ...STRATEGY, frozenVol: 0.6 }),
    '値動きなし（20日ボラ 0.50%。TOB 中の可能性）');
  assert.equal(STRATEGY.frozenVol, 0.3);
});


/* ------------------------------------------------ 際どい候補（運用者の線 95 の下） */

test('borderShape: 運用者の例（xgb 96・cat 98・lgbm 85）は「2つ95以上・残り 85〜90」', () => {
  assert.deepEqual(BORDER, { line: 95, lo: 85 });
  assert.equal(borderShape(cand('A', { lgbm: 85, xgb: 96, cat: 98 })), 'two95_mid');
});

test('borderShape: 形の全部と境目', () => {
  const sh = (l, x, c) => borderShape(cand('Z', { lgbm: l, xgb: x, cat: c }));
  assert.equal(sh(96, 97, 98), null);          // 3つとも95以上は線の上（際どくない）
  assert.equal(sh(95, 95, 95), null);          // ちょうど95は上
  assert.equal(sh(91, 96, 97), 'two95_hi');
  assert.equal(sh(96, 89.9, 97), 'two95_mid');
  assert.equal(sh(70, 99, 99), 'two95_lo');    // 2つ95以上なら、残りが低くても際どい
  assert.equal(sh(92, 93, 96), 'min90');       // 95以上は1つ
  assert.equal(sh(90, 91, 92), 'min90');
  assert.equal(sh(86, 88, 99), 'min85');
  assert.equal(sh(85, 88, 94), 'min85');       // 85ちょうどは入る
  assert.equal(sh(84.9, 99, 94), null);        // 95以上は1つで、最小が85未満
  assert.equal(sh(60, 70, 80), null);
  assert.equal(borderShape(cand('Y', { lgbm: 90, xgb: 99 })), null);   // 欠けていれば判定しない
  assert.equal(borderShape({}), null);
});

test('borderNote: LightGBM が一番低いと warn。文言は実験70 の数字から作る', () => {
  const n = borderNote(cand('A', { lgbm: 85, xgb: 96, cat: 98 }));
  assert.equal(n.warn, true);
  assert.equal(n.laggard, 'lgbm');
  assert.equal(n.laggardPct, 85);
  assert.equal(n.badge, '際どい・LightGBM が最下位');
  assert.ok(n.text.includes(`過去ほぼ 0%（0.0%・${BORDER_STATS.two95.lgbm.n}件）`));
  assert.ok(n.text.includes(`（${BORDER_STATS.two95.other.n}件）`));
  assert.ok(n.text.includes(`（${BORDER_STATS.line.n}件）`));
  assert.ok(n.text.includes('LightGBM が一番低い（85.0）'));
  assert.ok(!n.text.includes('−0.0%'));                // 丸めてから符号を付ける
  assert.ok(!/NaN|undefined|null/.test(n.text));
});

test('borderNote: XGBoost / CatBoost が一番低いと warn ではない', () => {
  const n = borderNote(cand('B', { lgbm: 97, xgb: 96, cat: 91 }));
  assert.equal(n.warn, false);
  assert.equal(n.laggard, 'cat');
  assert.equal(n.badge, '際どい・最下位 CatBoost');
  assert.ok(n.text.startsWith('CatBoost が一番低い（91.0）'));
  assert.ok(n.text.includes(`（${BORDER_STATS.two95.other.n}件）`));
  assert.ok(!/NaN|undefined|null/.test(n.text));
});

test('borderNote: 95以上が1つ以下の形は、LightGBM が一番低くても warn にしない（実験70 で差が無い）', () => {
  const n = borderNote(cand('C', { lgbm: 86, xgb: 92, cat: 97 }));
  assert.equal(n.shape, 'min85');
  assert.equal(n.laggard, 'lgbm');
  assert.equal(n.warn, false);
  assert.equal(n.badge, '際どい・最下位 LightGBM');
  assert.ok(n.text.includes('95以上が1つ以下で最小 85〜95 の形は過去 +1.1%（823件）'));
  assert.ok(n.text.includes('どのモデルが一番低いかで差は無い'));
  assert.ok(n.text.includes(`（LightGBM +1.2%・${BORDER_STATS.min.lgbm.n}件 / `
                            + `XGBoost・CatBoost +1.1%・${BORDER_STATS.min.other.n}件）`));
  assert.ok(!/NaN|undefined|null|測っていない/.test(n.text));
  assert.equal(borderNote(cand('D', { lgbm: 96, xgb: 97, cat: 98 })), null);
  assert.equal(borderNote(cand('E', { lgbm: 50, xgb: 60, cat: 70 })), null);
});

test('BORDER_STATS: 実験70 の数字（件数の合計が形の件数と合う）', () => {
  const t = BORDER_STATS;
  assert.equal(t.two95.lgbm.n + t.two95.other.n, 373 + 95 + 97);     // 2つ95以上の3つの形
  assert.equal(t.min.lgbm.n + t.min.other.n, t.min.all.n);
  assert.equal(t.min.all.n, 271 + 552);                                // 最小 90〜95 と 85〜90
  assert.equal(t.line.n, 616);
});

/* ------------------------------------------------ Jev（判断モデル） */

import { JEV, JEV_STATS, jevProb, jevNote, jevWarnings, jevEvidence } from '../src/lib/strategy.js';

const withJev = (c, prob) => ({ ...c, jev: { prob, model: 'jev-1.13.0', question: 'rise10_v1', cached: false } });

test('jevProb: 値が無ければ null（0 にしない）', () => {
  assert.equal(jevProb(cand('A', { lgbm: 95, xgb: 95, cat: 95 })), null);
  assert.equal(jevProb({ jev: null }), null);
  assert.equal(jevProb({ jev: { prob: 'x' } }), null);
  assert.equal(jevProb(withJev(cand('A', { lgbm: 95, xgb: 95, cat: 95 }), 37.5)), 37.5);
  assert.equal(jevProb(withJev({}, 0)), 0);
});

test('jevNote: 線（50）未満なら参考の印、以上なら印なし。規則の値と実験73 の実測が文言に入る', () => {
  assert.equal(JEV.line, 50);
  const low = jevNote(withJev(cand('A', { lgbm: 95, xgb: 95, cat: 95 }), 32));
  assert.equal(low.warn, true);
  assert.equal(low.prob, 32);
  assert.ok(low.badge.includes('50%未満'));
  assert.ok(low.text.includes('32%') && low.text.includes('+10%') && low.text.includes('20営業日'));
  assert.ok(low.text.includes('選定の規則には入れていない'));
  assert.ok(low.text.includes('見送る理由にはしない'));
  assert.ok(low.text.includes('実験73') && low.text.includes('45%'));   // 70〜90% の帯の実際の到達率
  const hi = jevNote(withJev(cand('B', { lgbm: 95, xgb: 95, cat: 95 }), 50));
  assert.equal(hi.warn, false);
  assert.equal(hi.badge, null);
  assert.ok(hi.text.includes('買う理由にはしない'));
  assert.equal(jevNote(cand('C', { lgbm: 95, xgb: 95, cat: 95 })), null);
});

test('JEV_STATS: 実験73 の数字（docs/MODEL_JEV.md と同じ）', () => {
  assert.equal(JEV_STATS.n, 3830);
  assert.equal(JEV_STATS.band.hi.hit, 44.7);
  assert.equal(JEV_STATS.border.d, -2.48);
  assert.ok(JEV_STATS.auc.vol > JEV_STATS.auc.jev && JEV_STATS.auc.ret > JEV_STATS.auc.jev);  // 素朴な基準が上
  assert.ok(jevEvidence().includes('3,830件') && jevEvidence().includes('−2.5pt ± 1.5'.replace('−', '-')));
});

test('jevWarnings: 買い候補のうち線の下のものだけ', () => {
  const picks = [
    withJev(cand('A', { lgbm: 95, xgb: 95, cat: 95 }), 70),
    withJev(cand('B', { lgbm: 95, xgb: 95, cat: 95 }), 20),
    cand('C', { lgbm: 95, xgb: 95, cat: 95 }),
  ];
  const w = jevWarnings(picks);
  assert.deepEqual(w.map((x) => x.c.code), ['B']);
  assert.deepEqual(jevWarnings(null), []);
});

test('strategySignal: Jev の値は選定を変えない（並び・件数・買う銘柄が同じ）', () => {
  const rows = day(25);
  const before = strategySignal(rows);
  const rowsJev = rows.map((c, i) => withJev(c, i === 0 ? 5 : 95));   // 1位の Jev が極端に低くても
  const after = strategySignal(rowsJev);
  assert.deepEqual(after.picks.map((c) => c.code), before.picks.map((c) => c.code));
  assert.equal(after.passed.length, before.passed.length);
  assert.equal(after.verdict.kind, before.verdict.kind);
  assert.equal(jevWarnings(after.picks).length, 1);                 // 注意は出る
});
