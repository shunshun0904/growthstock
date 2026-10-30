/**
 * 運用の戦略（src/lib/strategy.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BOOST, RULE_MODELS, RULE_JA, RULE_SHORT, RULE_SIGMA, RULE_ROW, STRATEGY, boostPcts, pctsOf, minPct, passesAgree, dayVerdict,
  strategySignal, exitPlan,
  freeSlots, pctSpread, laggard, nearMisses, fundContrib, frozenNote,
  BORDER, BORDER_STATS, BORDER_WARN, borderShape, borderNote, borderWarn, orderRows, ruleMinTier,
  ltrDayPct, ltrUpper, passesBoth, bothSummary, ltrRanking, ltr5Rank, ltr5Pool, condLevel,
} from '../src/lib/strategy.js';

const cand = (code, pcts, score = 0.5) => ({
  code, jqCode: `${code}0`, name: code, score, close: 1000,
  byModel: Object.fromEntries(
    Object.entries(pcts).map(([a, p]) => [a, { score, pctHistorical: p }])),
});

/** 今の規則のモデル（2026-10-29 から 1.5σ の3本: lgbm_k15 / xgb_k15 / cat_k15）の百分位で候補を作る。 */
const rule = (code, [a, b, c], score = 0.5) => {
  const pcts = {};
  [a, b, c].forEach((v, i) => { if (v !== undefined) pcts[RULE_MODELS[i]] = v; });
  return cand(code, pcts, score);
};

const day = (n, pcts = [95, 95, 95]) =>
  Array.from({ length: n }, (_, i) => rule(`X${i}`, pcts, 0.5 - i * 0.01));

/* ------------------------------------------------ 規則の3モデルの百分位 */

test('規則のモデル: 2026-10-29 から 1.5σ の3本・線 95（運用者の依頼「候補銘柄の選定モデルに、1.2σではなく、1.5σのモデルにしてほしいです」）', () => {
  assert.deepEqual(RULE_MODELS, ['lgbm_k15', 'xgb_k15', 'cat_k15']);
  assert.equal(STRATEGY.agreePct, 95);
  assert.equal(RULE_JA, '1.5σ の3本');
  assert.equal(RULE_SHORT, 'L15・X15・C15');
  assert.equal(RULE_SIGMA, '1.5σ');
  assert.equal(RULE_ROW, 'k15');
});

test('minPct: 規則の3モデルの最小を返す（1.2σ の値は見ない）', () => {
  const c = rule('A', [95, 91, 99]);
  c.byModel.lgbm = { score: 0.1, pctHistorical: 10 };       // 1.2σ の値が低くても規則には効かない
  assert.equal(minPct(c), 91);
  assert.equal(minPct(c, BOOST), null);                     // 1.2σ の3本はそろっていない
});

test('minPct: 線形・MLP・1.2σ は混ぜない', () => {
  const c = rule('A', [95, 91, 99]);
  c.byModel.logit_k15 = { score: 0.1, pctHistorical: 10 };
  c.byModel.mlp = { score: 0.1, pctHistorical: 5 };
  c.byModel.cat = { score: 0.1, pctHistorical: 3 };
  assert.equal(minPct(c), 91);
  assert.deepEqual(Object.keys(pctsOf(c)), RULE_MODELS);
  assert.deepEqual(Object.keys(boostPcts(c)), BOOST);
});

test('minPct: 1つでも欠けていれば null（欠測を「満たした」にしない）', () => {
  const c = rule('A', [95, undefined, 99]);
  assert.equal(minPct(c), null);
  assert.equal(passesAgree(c), false);
  assert.equal(minPct({}), null);
  // 1.2σ の3本だけがそろった候補（1.5σ の値が無い）は通らない
  assert.equal(passesAgree(cand('B', { lgbm: 99, xgb: 99, cat: 99 })), false);
});

test('passesAgree: 3つすべてが 95 以上のときだけ true', () => {
  assert.equal(passesAgree(rule('A', [95, 95, 95])), true);
  assert.equal(passesAgree(rule('B', [99, 99, 94.9])), false);
  assert.equal(passesAgree(rule('C', [90, 90, 90])), false);    // 以前の線 90 では通らない
});

/* ------------------------------------------------ その日を取るか */

test('dayVerdict: 発火数で帯に分かれる。2026-10-12 から発火の数では見送らない（7件以下は「発火が少ない日」）', () => {
  assert.equal(STRATEGY.skipBreaks, 1);
  assert.equal(STRATEGY.quietBreaks, 8);
  assert.equal(dayVerdict(26).kind, 'strong');
  assert.equal(dayVerdict(20).kind, 'strong');
  assert.equal(dayVerdict(19).kind, 'ok');
  assert.equal(dayVerdict(8).kind, 'ok');
  assert.equal(dayVerdict(7).kind, 'quiet');
  assert.equal(dayVerdict(1).kind, 'quiet');
  assert.equal(dayVerdict(0).kind, 'none');
  assert.equal(dayVerdict(null).kind, 'none');
  // 線を戻せば（以前の 8）7件以下は見送り
  assert.equal(dayVerdict(7, { ...STRATEGY, skipBreaks: 8 }).kind, 'skip');
});

/* ------------------------------------------------ シグナル */

test('strategySignal: 発火が多い日は上位2件を買う', () => {
  const rows = day(25);
  rows[3].byModel.lgbm_k15.pctHistorical = 99;   // 最小は 95 のまま
  const s = strategySignal(rows);
  assert.equal(s.nBreak, 25);
  assert.equal(s.verdict.kind, 'strong');
  assert.equal(s.passed.length, 25);
  assert.equal(s.picks.length, 2);
  assert.equal(s.buyable, true);
});

test('strategySignal: 3モデルの最小順位で並ぶ（lgbm 単体の順ではない）', () => {
  const rows = [
    rule('LOW', [99, 95.1, 95.2], 0.9),   // 最小 95.1・スコア最大
    rule('TOP', [96.5, 99, 98], 0.5),     // 最小 96.5
    rule('MID', [95.5, 95.5, 99], 0.6),   // 最小 95.5
    ...day(20, [97, 97, 97]),
  ];
  const s = strategySignal(rows);
  assert.equal(s.picks[0].code, 'X0');                  // 最小 97 が先頭
  const order = s.passed.map((c) => c.code);
  assert.ok(order.indexOf('TOP') < order.indexOf('MID'));
  assert.ok(order.indexOf('MID') < order.indexOf('LOW'));
});

/* ------------------------------------------------ 一覧の並び順（運用者の決定 2026-10-13） */

test('orderRows: 規則の3本（1.5σ）の最小が高い順（基準モデルの LightGBM のスコア順ではない）', () => {
  const rows = [
    { ...rule('BASE1', [83.6, 97.1, 97.4], 0.9), rankInDay: 1 },   // 基準モデル（1.2σ の LightGBM）では1位・最小 83.6
    { ...rule('PASS', [99.2, 95.2, 98.4], 0.4), rankInDay: 4 },    // 最小 95.2（規則を満たす）
    { ...rule('NEAR', [94.4, 93.5, 95.3], 0.8), rankInDay: 2 },    // 最小 93.5
  ];
  rows[0].byModel.lgbm = { score: 0.9, pctHistorical: 99.9 };      // 1.2σ の値は並び順に使わない
  const out = orderRows(rows);
  assert.deepEqual(out.map((c) => c.code), ['PASS', 'NEAR', 'BASE1']);
  assert.deepEqual(out.map((c) => c.order), [1, 2, 3]);
  assert.deepEqual(out.map((c) => c.ruleMin), [95.2, 93.5, 83.6]);
  assert.deepEqual(out.map((c) => c.rankInDay), [4, 2, 1]);        // 基準モデルの順位はそのまま残す
  assert.equal(rows[0].order, undefined);                          // 元の行は変えない
});

test('orderRows: 最小が同じなら基準モデルのスコアが高い順（strategySignal の並びと同じ）', () => {
  const rows = [
    ...day(3, [96, 96, 96]),                       // スコア 0.50 / 0.49 / 0.48
    rule('HI', [96, 99, 99], 0.7),
    rule('TOP', [97, 97, 97], 0.1),
    ...day(20, [50, 50, 50]).map((c, i) => ({ ...c, code: `L${i}`, jqCode: `L${i}0` })),
  ];
  const out = orderRows(rows);
  assert.deepEqual(out.slice(0, 5).map((c) => c.code), ['TOP', 'HI', 'X0', 'X1', 'X2']);
  const s = strategySignal(rows);
  assert.deepEqual(s.passed.map((c) => c.code), out.filter((c) => c.ruleMin >= 95).map((c) => c.code));
});

test('orderRows: 規則の3本の百分位が欠けた行は後ろに、基準モデルのスコア順で並ぶ', () => {
  const rows = [
    { ...rule('MISS_HI', [99, 99, undefined], 0.9), rankInDay: 1 },
    { ...rule('LOW', [10, 20, 30], 0.2), rankInDay: 5 },
    { ...cand('OLD', { lgbm_k15: 99, lgbm: 99, xgb: 99, cat: 99 }, 0.5), rankInDay: 3 },   // 規則の3本のうち1本だけ
    { ...rule('HIGH', [90, 91, 92], 0.3), rankInDay: 4 },
  ];
  const out = orderRows(rows);
  assert.deepEqual(out.map((c) => c.code), ['HIGH', 'LOW', 'MISS_HI', 'OLD']);
  assert.deepEqual(out.map((c) => c.ruleMin), [90, 10, null, null]);
});

test('orderRows: 規則の3本がそろわない日（固定した過去の日など）は基準モデルの順のまま', () => {
  const rows = [3, 1, 2].map((r) => ({ ...cand(`R${r}`, { lgbm: 90 - r }, 0.6 - r * 0.1), rankInDay: r }));
  const out = orderRows(rows);
  assert.deepEqual(out.map((c) => c.code), ['R1', 'R2', 'R3']);
  assert.ok(out.every((c) => c.ruleMin === null));
  assert.deepEqual(orderRows([]), []);
  assert.deepEqual(orderRows(null), []);
});

test('ruleMinTier: 規則の線（95）以上・惜しい範囲（85〜95）・それ未満', () => {
  assert.equal(ruleMinTier(95), 'pass');
  assert.equal(ruleMinTier(100), 'pass');
  assert.equal(ruleMinTier(94.9), 'near');
  assert.equal(ruleMinTier(85), 'near');
  assert.equal(ruleMinTier(84.9), 'below');
  assert.equal(ruleMinTier(0), 'below');
  assert.equal(ruleMinTier(null), null);
  assert.equal(ruleMinTier(NaN), null);
});

test('strategySignal: 発火が少ない日（7件以下）も基準を満たせば上位2件を買う（2026-10-12 から）', () => {
  const s = strategySignal(day(5));
  assert.equal(s.verdict.kind, 'quiet');
  assert.equal(s.buyable, true);
  assert.equal(s.picks.length, 2);
  assert.equal(s.passed.length, 5);
  assert.equal(s.held.length, 3);
});

test('strategySignal: 規則の3本の百分位が欠けた候補が1件でもあれば判定できない（買わない。live_track と同じ）', () => {
  const s = strategySignal([rule('A', [99, 99, 99]), rule('B', [99, undefined, 99]), ...day(8, [50, 50, 50])]);
  assert.equal(s.missing, 1);
  assert.equal(s.buyable, false);
  assert.equal(s.picks.length, 0);
  // 規則の3本の値が無い日（1.2σ の値だけ。1.5σ を予測に入れる前に出し、その日の値のまま固定した日）も判定できない
  const old = strategySignal([cand('C', { lgbm: 99, xgb: 99, cat: 99 }),
    cand('D', { lgbm: 98, xgb: 97, cat: 99 })]);
  assert.equal(old.missing, 2);
  assert.equal(old.picks.length, 0);
  assert.equal(strategySignal(day(10)).missing, 0);
  assert.equal(strategySignal(day(10)).picks.length, 2);
});

test('strategySignal: 見送りの線を戻すと（以前の 8）、発火が少ない日は買わない', () => {
  const s = strategySignal(day(5), { ...STRATEGY, skipBreaks: 8 });
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

test('exitPlan: 終値から指値の目安を出す（2026-10-12 から +10%）', () => {
  const p = exitPlan(1000);
  assert.equal(STRATEGY.takeProfit, 10);
  assert.ok(Math.abs(p.target - 1100) < 1e-9);          // 1000 × 1.1（浮動小数点の端数は画面で丸める）
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
  // クニミネ工業 2026-09-18 の 1.2σ の実際の値を、規則のモデル（1.5σ）にも当てる
  const k = rule('5388', [86.5, 91.7, 91.5]);
  assert.equal(minPct(k), 86.5);
  assert.equal(Math.round(pctSpread(k) * 10) / 10, 5.2);
  assert.equal(laggard(k), 'lgbm_k15');
  const k12 = cand('5388', { lgbm: 86.5, xgb: 91.7, cat: 91.5 });
  assert.equal(laggard(k12, BOOST), 'lgbm');                  // 1.2σ の3本で見るとき
  // 欠けていれば null（欠けたまま判定しない）
  const bad = rule('9999', [90, 90]);
  assert.equal(pctSpread(bad), null);
  assert.equal(laggard(bad), null);
});

test('nearMisses: 最小85〜95 だけを拾い、幅5以上の形に印を付ける', () => {
  const rows = [
    rule('A', [96, 95, 97]),          // 基準通過。対象外
    rule('B', [86.5, 91.7, 91.5]),    // 惜しい・幅5.2
    rule('C', [88, 89, 87]),          // 惜しい・幅2 → 揃っている
    rule('D', [70, 95, 95]),          // 85未満。対象外
    rule('E', [94, 96, 97]),          // 惜しい（以前の線 90 なら通っていた）
  ];
  const n = nearMisses(rows);
  assert.deepEqual(n.map((x) => x.code), ['E', 'C', 'B']);  // 最小の降順
  assert.equal(n.find((x) => x.code === 'B').weak, true);
  assert.equal(n.find((x) => x.code === 'C').weak, false);
  assert.equal(n.find((x) => x.code === 'B').laggard, 'lgbm_k15');
});

test('nearMisses: 境界（85ちょうどは入る / 95ちょうどは入らない）', () => {
  assert.equal(nearMisses([rule('E', [85, 99, 99])]).length, 1);
  assert.equal(nearMisses([rule('F', [95, 99, 99])]).length, 0);
  assert.equal(nearMisses([rule('G', [84.9, 99, 99])]).length, 0);
  assert.equal(nearMisses([rule('H', [94.9, 99, 99])]).length, 1);
});

test('dayVerdict: 文言は今の規則で測った数字（7件以下は件数が少なく当たり外れが大きいと添える）', () => {
  assert.ok(dayVerdict(25).note.includes('1.5σ の3本が95以上で218件 +4.25%・勝率74%'));
  assert.ok(dayVerdict(10).note.includes('116件 +2.22%・勝率62%'));
  const quiet = dayVerdict(5).note;
  assert.ok(quiet.includes('枠が空いていれば買う') && quiet.includes('20件 +4.22%'));
  assert.ok(quiet.includes('32件 +2.16%') && quiet.includes('−10%割れ 12.5%'));
  assert.equal(dayVerdict(5).label, '発火が少ない日');
  for (const n of [25, 10, 5, 0]) assert.ok(!/NaN|undefined|null/.test(dayVerdict(n).note));
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


/* ------------------------------------------------ 際どい候補（規則の3本 = 1.5σ の3本で見た線 95 の下） */

test('borderShape: 運用者の例（xgb 96・cat 98・lgbm 85）を 1.5σ の3本で見ると「2本95以上・残り 85〜90」', () => {
  assert.deepEqual(BORDER, { line: 95, lo: 85 });
  assert.equal(borderShape(rule('A', [85, 96, 98])), 'two95_mid');
  // 1.2σ の値は見ない（1.2σ の値だけでは判定しない）
  assert.equal(borderShape(cand('B', { lgbm: 85, xgb: 96, cat: 98 })), null);
});

test('borderShape: 形の全部と境目', () => {
  const sh = (l, x, c) => borderShape(rule('Z', [l, x, c]));
  assert.equal(sh(96, 97, 98), null);          // 3本とも95以上は線の上（際どくない）
  assert.equal(sh(95, 95, 95), null);          // ちょうど95は上
  assert.equal(sh(91, 96, 97), 'two95_hi');
  assert.equal(sh(96, 89.9, 97), 'two95_mid');
  assert.equal(sh(70, 99, 99), 'two95_lo');    // 2本95以上なら、残りが低くても際どい
  assert.equal(sh(92, 93, 96), 'min90');       // 95以上は1本
  assert.equal(sh(90, 91, 92), 'min90');
  assert.equal(sh(86, 88, 99), 'min85');
  assert.equal(sh(85, 88, 94), 'min85');       // 85ちょうどは入る
  assert.equal(sh(84.9, 99, 94), null);        // 95以上は1本で、最小が85未満
  assert.equal(sh(60, 70, 80), null);
  assert.equal(borderShape(rule('Y', [90, 99])), null);   // 欠けていれば判定しない
  assert.equal(borderShape({}), null);
});

test('borderWarn: 先に決めた線（20件以上で平均 0% 以下）', () => {
  assert.deepEqual(BORDER_WARN, { minN: 20, ret: 0 });
  assert.equal(borderWarn({ n: 20, ret: 0 }), true);
  assert.equal(borderWarn({ n: 300, ret: -1.2 }), true);
  assert.equal(borderWarn({ n: 19, ret: -3 }), false);      // 件数が少ない
  assert.equal(borderWarn({ n: 50, ret: 0.01 }), false);
  assert.equal(borderWarn({ n: 0, ret: null }), false);
  assert.equal(borderWarn(undefined), false);
});

/** 文言の組み立てを確かめるための作りものの数字（実測は BORDER_STATS） */
const ST = {
  line: { n: 500, ret: 4.0 },
  two95: { all: { n: 130, ret: 1.0 }, lgbm_k15: { n: 40, ret: -0.4 }, xgb_k15: { n: 60, ret: 2.0 },
           cat_k15: { n: 30, ret: 1.5 } },
  min: { all: { n: 110, ret: 1.0 }, lgbm_k15: { n: 100, ret: 1.04 }, xgb_k15: { n: 10, ret: -2.0 },
         cat_k15: { n: 0, ret: null } },
};

test('borderNote: その組の平均が 0% 以下（20件以上）なら赤。ほかのモデルと線の上の数字を添える', () => {
  const n = borderNote(rule('A', [85, 96, 98]), BORDER, ST);
  assert.equal(n.shape, 'two95_mid');
  assert.equal(n.laggard, 'lgbm_k15');
  assert.equal(n.laggardPct, 85);
  assert.equal(n.warn, true);
  assert.equal(n.few, false);
  assert.equal(n.badge, '1.5σで際どい・LightGBM が最下位');
  assert.ok(n.text.startsWith('LightGBM が一番低い（85.0）。'));
  assert.ok(n.text.includes('2本が95以上の形で LightGBM が一番低いときは、過去 −0.4%（40件）。空き枠（0%）と同じか悪い'));
  assert.ok(n.text.includes('ほかが一番低いときは XGBoost +2.0%（60件）・CatBoost +1.5%（30件）'));
  assert.ok(n.text.includes('3本とも95以上は +4.0%（500件）'));
  assert.ok(!/NaN|undefined|null/.test(n.text));
});

test('borderNote: 平均がプラスの組は黄', () => {
  const n = borderNote(rule('B', [97, 96, 91]), BORDER, ST);
  assert.equal(n.shape, 'two95_hi');
  assert.equal(n.laggard, 'cat_k15');
  assert.equal(n.warn, false);
  assert.equal(n.badge, '1.5σで際どい・最下位 CatBoost');
  assert.ok(n.text.startsWith('CatBoost が一番低い（91.0）。'));
  assert.ok(!n.text.includes('空き枠'));
});

test('borderNote: 件数の少ない組は赤にしない（平均がマイナスでも）。0件の組は「0件」', () => {
  const x = borderNote(rule('C', [92, 86, 97]), BORDER, ST);
  assert.equal(x.shape, 'min85');
  assert.equal(x.laggard, 'xgb_k15');
  assert.equal(x.warn, false);
  assert.equal(x.few, true);
  assert.ok(x.text.includes('95以上が1本以下・最小 85〜95 の形で XGBoost が一番低いときは、過去 −2.0%（10件）。件数が少なく目安にしにくい'));
  assert.ok(x.text.includes('LightGBM +1.0%（100件）・CatBoost 0件'));
  const c = borderNote(rule('D', [92, 93, 88]), BORDER, ST);
  assert.equal(c.laggard, 'cat_k15');
  assert.equal(c.few, true);
  assert.ok(c.text.includes('CatBoost が一番低いときは、過去 0件'));
  assert.ok(!/NaN|undefined|null/.test(x.text + c.text));
  assert.equal(borderNote(rule('E', [96, 97, 98]), BORDER, ST), null);
  assert.equal(borderNote(rule('F', [50, 60, 70]), BORDER, ST), null);
});

test('BORDER_STATS: probe_k15_panel の数字（件数の合計が形の件数と合う・今は赤の組が無い）', () => {
  const t = BORDER_STATS;
  for (const fam of ['two95', 'min']) {
    assert.equal(RULE_MODELS.reduce((a, m) => a + t[fam][m].n, 0), t[fam].all.n);
  }
  assert.equal(t.two95.all.n, 347 + 113 + 42);      // 2本95以上の3つの形（probe_k15_panel の ■4a）
  assert.equal(t.min.all.n, 314 + 580);              // 最小 90〜95 と 85〜90
  assert.equal(t.line.n, 561);
  const reds = ['two95', 'min'].flatMap((f) => RULE_MODELS.filter((m) => borderWarn(t[f][m])));
  assert.deepEqual(reds, []);
  // 実際の数字で作った文言（運用者の例の形）
  const n = borderNote(rule('A', [85, 96, 98]));
  assert.equal(n.warn, false);
  assert.equal(n.badge, '1.5σで際どい・最下位 LightGBM');
  assert.ok(n.text.includes('2本が95以上の形で LightGBM が一番低いときは、過去 +1.5%（154件）'));
  assert.ok(n.text.includes('ほかが一番低いときは XGBoost +2.1%（208件）・CatBoost +2.1%（140件）'));
  assert.ok(n.text.includes('3本とも95以上は +2.7%（561件）'));
  assert.ok(n.text.includes('（1.5σ の3本・前向きの百分位。'));
  assert.ok(!/NaN|undefined|null|測っていない/.test(n.text));
});

/* ------------------------------------------------ ランキング学習（参考。両方使う形） */

const withLtr = (c, dayPct, upper) => ({ ...c, ltr: { score: 0.1, dayPct, dayRank: 1, nDay: 4, upper } });

test('ltrDayPct / ltrUpper: 予測の ltr を読む。無い・壊れた値は null', () => {
  const c = withLtr(rule('A', [96, 97, 98]), 66.7, true);
  assert.equal(ltrDayPct(c), 66.7);
  assert.equal(ltrUpper(c), true);
  assert.equal(ltrDayPct(rule('B', [96, 97, 98])), null);
  assert.equal(ltrUpper(rule('B', [96, 97, 98])), null);
  assert.equal(ltrUpper({ ltr: { upper: 'yes' } }), null);
  assert.equal(ltrDayPct({ ltr: { dayPct: null } }), null);
});

test('passesBoth: 今の規則を満たし、かつランキング学習が上半分のときだけ（予測の upper で決める）', () => {
  assert.equal(passesBoth(withLtr(rule('A', [96, 97, 98]), 66.7, true)), true);
  assert.equal(passesBoth(withLtr(rule('B', [96, 97, 98]), 33.3, false)), false);
  assert.equal(passesBoth(withLtr(rule('C', [94, 97, 98]), 100, true)), false);
  assert.equal(passesBoth(rule('D', [96, 97, 98])), false);
});

test('bothSummary: 規則を満たした候補のうち上半分を、passed の並びのまま。ランキング学習の無い日は数えない', () => {
  const rows = [
    withLtr(rule('A', [99, 99, 99], 0.9), 100, true),
    withLtr(rule('B', [97, 98, 99], 0.8), 33.3, false),
    withLtr(rule('C', [96, 96, 96], 0.7), 66.7, true),
    withLtr(rule('D', [50, 60, 70], 0.6), 0, false),
  ];
  const sig = strategySignal(rows);
  const b = bothSummary(rows, sig);
  assert.equal(b.available, true);
  assert.equal(b.nPassed, 3);
  assert.deepEqual(b.both.map((c) => c.code), ['A', 'C']);
  // 買う銘柄（今の規則の上位2件）は変えない
  assert.deepEqual(sig.picks.map((c) => c.code), ['A', 'B']);
  const old = rows.map(({ ltr, ...c }) => c);
  const ob = bothSummary(old, strategySignal(old));
  assert.equal(ob.available, false);
  assert.deepEqual(ob.both, []);
  assert.equal(bothSummary([], strategySignal([])).available, false);
});

test('ltrRanking: ランキング学習の順位の順に並べ、上半分・今の規則・両方使う形の印を付ける。順位が欠けた日は出さない', () => {
  const r = (code, pcts, rank, dayPct, upper) => ({ ...rule(code, pcts), ltr: { score: 0.1, dayPct, dayRank: rank, nDay: 4, upper } });
  const rows = [
    r('A', [99, 99, 99], 3, 33.3, false),
    r('B', [50, 60, 70], 1, 100, true),
    r('C', [96, 97, 98], 2, 66.7, true),
    r('D', [96, 97, 98], 4, 0, false),
  ];
  const k = ltrRanking(rows);
  assert.equal(k.available, true);
  assert.deepEqual(k.items.map((x) => x.c.code), ['B', 'C', 'A', 'D']);
  assert.deepEqual(k.items.map((x) => x.rank), [1, 2, 3, 4]);
  assert.equal(k.nUpper, 2);
  assert.deepEqual(k.items.map((x) => x.upper), [true, true, false, false]);
  assert.deepEqual(k.items.map((x) => x.rule), [false, true, true, true]);
  assert.deepEqual(k.items.map((x) => x.both), [false, true, false, false]);   // passesBoth と同じ
  assert.deepEqual(k.items.map((x) => x.both), k.items.map((x) => passesBoth(x.c)));
  assert.deepEqual(k.items.map((x) => x.dayPct), [100, 66.7, 33.3, 0]);
  assert.equal(k.items[0].nDay, 4);
  assert.equal(rows[0].rank, undefined);                                       // 元の行は変えない
  // 同じ順位は銘柄コードの順
  const tie = [r('Z', [1, 1, 1], 1, 75, true), r('Y', [1, 1, 1], 1, 75, true)];
  assert.deepEqual(ltrRanking(tie).items.map((x) => x.c.code), ['Y', 'Z']);
  // 1件でも順位が無ければ出さない（bothSummary と同じ扱い）。候補ゼロも
  assert.equal(ltrRanking([rows[0], rule('E', [96, 97, 98])]).available, false);
  assert.equal(ltrRanking([{ ...rows[0], ltr: { ...rows[0].ltr, dayRank: 1.5 } }]).available, false);
  assert.deepEqual(ltrRanking([]), { available: false, items: [], nUpper: 0 });
  assert.equal(ltrRanking(null).available, false);
});

test('ltr5Rank / ltr5Pool / condLevel: 直近5営業日の組の中の順位と3条件・4条件の印（予測の ltr5・conds。2026-11-02 から）', () => {
  const c = { ltr5: { score: 0.3, rank: 2, nPool: 41, pct: 97.5 }, conds: { d1: true, w1: false, r15: true, r0: true, c3: false, c4: false } };
  assert.equal(ltr5Rank(c), 2);
  assert.equal(ltr5Pool(c), 41);
  assert.equal(condLevel(c), null);                                       // 3条件・4条件とも満たさない
  assert.equal(condLevel({ conds: { c3: true, c4: false } }), 'c3');
  assert.equal(condLevel({ conds: { c3: true, c4: true } }), 'c4');
  assert.equal(condLevel({ conds: { c3: null, c4: null } }), null);       // 決まらない（日次の順位が無いなど）
  // 値が無い日（2026-11-02 より前）・形が違う値は null
  assert.equal(ltr5Rank({}), null);
  assert.equal(ltr5Pool({ ltr5: { nPool: '41' } }), null);
  assert.equal(ltr5Rank({ ltr5: { rank: 1.5 } }), null);
  assert.equal(condLevel(undefined), null);
});

test('ltrRanking: 順位表の行に、直近5営業日の組の中の順位と3条件・4条件の印も付ける（無ければ null）', () => {
  const r = (code, rank, extra = {}) => ({ ...rule(code, [96, 97, 98]), ltr: { score: 0.1, dayPct: 50, dayRank: rank, nDay: 2, upper: rank === 1 },
                                           ...extra });
  const k = ltrRanking([
    r('B', 2),
    r('A', 1, { ltr5: { rank: 1, nPool: 30 }, conds: { c3: true, c4: true } }),
  ]);
  assert.deepEqual(k.items.map((x) => [x.c.code, x.rank5, x.pool5, x.cond]), [['A', 1, 30, 'c4'], ['B', null, null, null]]);
});
