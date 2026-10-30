/**
 * モデル別の並び（src/lib/predictions.js modelRows）。
 *
 * - 並び順は payload の models に従い、無いモデルは出さない
 * - ラベル違い（1.5σ のラベルで学習した lgbm / xgb / cat の木3本）は塊 k15 で、記号は payload の short
 *   （無ければ L15 / X15 / C15）。1.8σ の4本（L18 など）は 2026-10-24 に、ロジスティック回帰（LR / LR15）は 2026-11-02 の
 *   予測から外したので、予測ファイルにあっても出さない
 *   （RETIRED_VARIANTS・activeModels。表・スコアの物差しにも入れない）
 * - 表の形（modelGrid）: 行 = ラベル（1.2σ / 1.5σ）、列 = 学習器。無いモデルのマスは null。古い予測ファイルは1行
 * - マスの字の色（pctInk）: 明るい区分（60 以上）は暗い字、暗い区分は明るい字
 * - 選んだ日の値の素性（dayOrigin）: 過去の日は dayInfo のモデルの学習日と出した時刻。dayInfo が無ければ今のモデル
 * - 効いた特徴量の名前（featureLabel）: feature_names.json の日本語名 → 行の name → 元の列名。説明に元の列名を添える
 * - 次の決算発表（earningsNote / earningsSummary）: 持つ間（20営業日）に来るなら黄。未公表・項目の無いファイル
 *
 *   node --test tests/predictions.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { modelRows, modelGrid, pctInk, scoreRange, scoreT, scoreColor, scoreInk, SCORE_COLORS, MODEL_SHORT, MODEL_FAMILY, FAMILY_JA, LABEL_JA, dayOrigin, featureLabel,
  earningsNote, earningsSummary, RETIRED_VARIANTS, RETIRED_MODELS, activeModels } from '../src/lib/predictions.js';

const models = [
  { algo: 'lgbm', name: 'LightGBM', short: 'LGB' },
  { algo: 'xgb', name: 'XGBoost', short: 'XGB' },
  { algo: 'lgbm_k15', name: 'LightGBM（1.5σ）', short: 'L15', variant: true },
  { algo: 'logit_k15', name: 'ロジスティック回帰（1.5σ）', short: 'LR15', variant: true },
  { algo: 'lgbm_k18', name: 'LightGBM（1.8σ）', short: 'L18', variant: true },
  { algo: 'xgb_k18', name: 'XGBoost（1.8σ）', short: 'X18', variant: true },
];

const cand = {
  byModel: {
    lgbm: { score: 0.5, pctHistorical: 96 },
    xgb: { score: 0.4, pctHistorical: 91 },
    lgbm_k15: { score: 0.3, pctHistorical: 94 },
    logit_k15: { score: 0.3, pctHistorical: 88 },
    lgbm_k18: { score: 0.2, pctHistorical: 70 },
    xgb_k18: { score: 0.2, pctHistorical: 65 },
  },
};

test('modelRows: ラベル違いは k ごとの塊で、記号は元の記号の頭 + k。外した 1.8σ・ロジスティック回帰は予測ファイルにあっても出さない', () => {
  const rows = modelRows(cand, models);
  assert.deepEqual(rows.map((r) => r.algo), ['lgbm', 'xgb', 'lgbm_k15']);
  assert.deepEqual(rows.map((r) => r.family), ['tree', 'tree', 'k15']);
  assert.deepEqual(rows.map((r) => r.short), ['LGB', 'XGB', 'L15']);
  // models が無い（古い）ファイルでも、byModel の 1.8σ・ロジスティック回帰は出さない
  assert.deepEqual(modelRows(cand, []).map((r) => r.algo), ['lgbm', 'xgb', 'lgbm_k15']);
  assert.deepEqual([...RETIRED_VARIANTS].sort(), ['cat_k18', 'lgbm_k18', 'logit_k18', 'xgb_k18']);
  assert.deepEqual([...RETIRED_MODELS].sort(), ['cat_k18', 'lgbm_k18', 'logit', 'logit_k15', 'logit_k18', 'xgb_k18']);
  assert.deepEqual(activeModels(models).map((m) => m.algo), ['lgbm', 'xgb', 'lgbm_k15']);
  assert.deepEqual(activeModels(null), []);
  assert.equal(rows[2].pct, 94);
  assert.equal(FAMILY_JA.k15, 'ラベル1.5σ');
  assert.equal(FAMILY_JA.k18, 'ラベル1.8σ');
  assert.equal(MODEL_FAMILY.cat_k15, 'k15');
  assert.equal(MODEL_FAMILY.logit_k18, 'k18');
  assert.equal(MODEL_SHORT.lgbm_k18, 'L18');
  assert.equal(MODEL_SHORT.cat_k18, 'C18');
  // 記号は重ならない（シートの列名にもなる）
  const shorts = Object.values(MODEL_SHORT);
  assert.equal(new Set(shorts).size, shorts.length);
});

test('modelRows: payload に無いモデルは並べない（古い予測ファイル）', () => {
  const rows = modelRows(cand, models.slice(0, 2));
  assert.deepEqual(rows.map((r) => r.algo), ['lgbm', 'xgb']);
  // short が無い models でも控えの記号で出る
  const rows2 = modelRows(cand, [{ algo: 'lgbm_k15', name: 'x' }, { algo: 'xgb_k18', name: 'y' }]);
  assert.deepEqual(rows2.map((r) => r.short), ['L15']);
});

test('modelGrid: ラベル × 学習器の 2行 × 3列（1.8σ の行・ロジスティック回帰の列は出さない）。並びは payload の models の順', () => {
  const learners = ['lgbm', 'xgb', 'cat', 'logit'];
  const all = [];
  const byModel = {};
  ['', '_k15', '_k18'].forEach((sfx, r) => learners.forEach((l, i) => {
    const algo = `${l}${sfx}`;
    all.push({ algo, name: algo, short: MODEL_SHORT[algo] });
    byModel[algo] = { score: 0.1, pctHistorical: 90 + r * 3 + i };       // 行と列が分かる値
  }));
  const g = modelGrid({ byModel }, all);
  assert.deepEqual(g.cols.map((c) => c.short), ['LGB', 'XGB', 'CAT']);
  assert.deepEqual(g.rows.map((r) => r.labelJa), ['1.2σ', '1.5σ']);
  assert.deepEqual(g.rows.map((r) => r.cells.map((x) => x.pct)),
    [[90, 91, 92], [93, 94, 95]]);
  assert.equal(g.rows[1].cells[2].algo, 'cat_k15');
  assert.equal(LABEL_JA.base, '1.2σ');
});

test('modelGrid: 無いモデルは null、古い予測ファイルは 1.2σ の1行（MLP があれば NN の列）', () => {
  // ラベル違いが一部だけの日（cand は lgbm / xgb / lgbm_k15 / logit_k15 / lgbm_k18 / xgb_k18。logit_k15 は外したので出さない）
  const g = modelGrid(cand, models);
  assert.deepEqual(g.cols.map((c) => c.learner), ['lgbm', 'xgb']);
  assert.deepEqual(g.rows.map((r) => r.label), ['base', 'k15']);
  assert.deepEqual(g.rows[0].cells.map((x) => x && x.pct), [96, 91]);
  assert.deepEqual(g.rows[1].cells.map((x) => x && x.pct), [94, null]);
  const old = modelGrid(
    { byModel: { lgbm: { pctHistorical: 50 }, xgb: { pctHistorical: 60 }, mlp: { pctHistorical: 70 } } },
    [{ algo: 'lgbm', name: 'LightGBM' }, { algo: 'xgb', name: 'XGBoost' }, { algo: 'mlp', name: 'ニューラルネット' }]);
  assert.deepEqual(old.cols.map((c) => c.short), ['LGB', 'XGB', 'NN']);
  assert.equal(old.rows.length, 1);
  assert.deepEqual(modelGrid({}, models), { cols: [], rows: [] });          // byModel が無い予測ファイル
});

test('pctInk: 明るい地（60 以上）は暗い字、暗い地は明るい字、値が無ければ薄い字', () => {
  for (const v of [60, 80, 90, 95, 100]) assert.equal(pctInk(v), 'var(--bg)');
  for (const v of [0, 29.9, 30, 59.9]) assert.equal(pctInk(v), 'var(--text)');
  assert.equal(pctInk(null), 'var(--text-faint)');
  assert.equal(pctInk(NaN), 'var(--text-faint)');
});


test('dayOrigin: 過去の日は、その日に出した値のモデル（dayInfo）。最新の日と dayInfo が無いファイルは今のモデル', () => {
  const data = {
    asOf: '2026-10-13',
    model: { trainedAt: '2026-10-11T01:00:00+00:00' },
    dayInfo: {
      '2026-10-09': { trainedAt: '2026-10-04T06:23:15+00:00', generatedAt: '2026-10-09T11:53:25+00:00' },
      '2026-10-13': { trainedAt: '2026-10-11T01:00:00+00:00', generatedAt: '2026-10-13T09:50:00+00:00' },
    },
  };
  const past = dayOrigin(data, '2026-10-09');
  assert.equal(past.past, true);
  assert.equal(past.trainedAt, '2026-10-04T06:23:15+00:00');
  assert.equal(past.generatedAt, '2026-10-09T11:53:25+00:00');
  assert.equal(past.sameModel, false);
  const now = dayOrigin(data, '2026-10-13');
  assert.equal(now.past, false);
  assert.equal(now.sameModel, true);
  // dayInfo が無い（固定を始める前の形の）ファイル: 今のモデルの学習日、出した時刻は無し
  const old = dayOrigin({ asOf: '2026-09-11', model: { trainedAt: 't' } }, '2026-09-10');
  assert.deepEqual(old, { past: true, trainedAt: 't', generatedAt: null, sameModel: true });
  assert.equal(dayOrigin(null, '2026-09-10').past, false);
});

test('featureLabel: 日本語名は feature_names.json → 行の name → 元の列名。説明に元の列名を添える', () => {
  const names = { vol_20d: { name: '20日ボラティリティ', desc: '直近20営業日の日次リターン標準偏差（%）' } };
  const a = featureLabel({ col: 'vol_20d', ja: '古い説明', contrib: 0.1 }, names);
  assert.equal(a.name, '20日ボラティリティ');
  assert.equal(a.title, '直近20営業日の日次リターン標準偏差（%）（vol_20d）');
  assert.equal(a.translated, true);
  // 名前の辞書が無い・読めない（{}）: 行の name、それも無ければ元の列名（説明は行の ja。** は落とす）
  const b = featureLabel({ col: 'eps_growth_q1', name: 'EPS成長率（1期前）', ja: '**説明**' }, {});
  assert.deepEqual(b, { name: 'EPS成長率（1期前）', title: '説明（eps_growth_q1）', translated: true });
  const c = featureLabel({ col: 'x_new' }, null);
  assert.deepEqual(c, { name: 'x_new', title: 'x_new', translated: false });
});

test('earningsNote: 次の決算が持つ間（20営業日）に来るなら黄。より先・未公表は灰。項目の無い行は出さない', () => {
  const soon = earningsNote({ nextEarnDate: '2026-10-30', nextEarnQuarter: '2Q', nextEarnDays: 17,
                              nextEarnTradingDays: 13, daysSinceDisc: 75 }, 20);
  assert.equal(soon.label, '決算 13営業日後');
  assert.equal(soon.tone, 'amber');
  assert.equal(soon.inWindow, true);
  assert.equal(soon.text, '次の決算発表 2026/10/30（第2四半期）: 13営業日後・17日後。買ってから20営業日の中に入る（決算の結果で動きうる）');
  const far = earningsNote({ nextEarnDate: '2026-12-10', nextEarnQuarter: 'FY', nextEarnDays: 58, nextEarnTradingDays: 40 }, 20);
  assert.equal(far.tone, 'slate');
  assert.ok(far.text.endsWith('買ってから20営業日より先'));
  assert.equal(earningsNote({ nextEarnDate: '2026-10-13', nextEarnTradingDays: 0, nextEarnDays: 0 }).label, '決算 当日');
  // ちょうど20営業日目は入る、21 は入らない
  assert.equal(earningsNote({ nextEarnDate: 'x', nextEarnTradingDays: 20 }, 20).inWindow, true);
  assert.equal(earningsNote({ nextEarnDate: 'x', nextEarnTradingDays: 21 }, 20).inWindow, false);
  const none = earningsNote({ nextEarnDate: null, nextEarnTradingDays: null, daysSinceDisc: 120 });
  assert.equal(none.known, false);
  assert.equal(none.label, '決算予定 未公表');
  assert.ok(none.text.includes('前回の決算開示から 120日'));
  assert.equal(earningsNote({ code: 'old' }), null);          // 項目の無い行（その日に出した値のまま固定した日など）
  assert.ok(!/NaN|undefined|null/.test(soon.text + far.text + none.text));
});

test('earningsSummary: 持つ間に決算が来る候補の数。項目の無い日は null', () => {
  const rows = [
    { nextEarnDate: '2026-10-30', nextEarnTradingDays: 13 },
    { nextEarnDate: '2026-12-10', nextEarnTradingDays: 40 },
    { nextEarnDate: null, nextEarnTradingDays: null },
  ];
  assert.deepEqual(earningsSummary(rows, 20), { total: 3, known: 2, soon: 1 });
  assert.equal(earningsSummary([{ code: 'old' }], 20), null);
  assert.equal(earningsSummary(null, 20), null);
});

// スコアそのものの表（運用者の依頼 2026-10-16）: 色の物差しは渡した候補の全モデル（外した 1.8σ は入れない）
test('scoreRange: 全候補・全モデルのスコアの最小〜最大。外した 1.8σ・ロジスティック回帰は入れない。スコアが無ければ null', () => {
  const cands = [
    { byModel: { lgbm: { score: 0.2, pctHistorical: 50 }, xgb_k18: { score: 0.05 }, logit: { score: 0.9 } } },
    { byModel: { cat_k15: { score: 0.61 }, logit_k15: { score: 0.01 }, mlp: {} } },
    { date: '2026-10-16' },
  ];
  assert.deepEqual(scoreRange(cands), { lo: 0.2, hi: 0.61, n: 2 });       // xgb_k18・logit・logit_k15 は物差しに入らない
  assert.equal(scoreRange([]), null);
  assert.equal(scoreRange([{ byModel: { lgbm: { pctHistorical: 90 } } }]), null);
  assert.equal(scoreRange(null), null);
});

test('scoreColor / scoreInk: 最小は暗い青、最大は白に近い青、範囲の外は端に揃え、値が無ければ薄い地', () => {
  const r = { lo: 0.1, hi: 0.5, n: 2 };
  const lo = `rgb(${SCORE_COLORS.lo.join(', ')})`;
  const hi = `rgb(${SCORE_COLORS.hi.join(', ')})`;
  assert.equal(scoreColor(0.1, r), lo);
  assert.equal(scoreColor(0.5, r), hi);
  assert.equal(scoreColor(0.0, r), lo);                                   // 範囲の外は端
  assert.equal(scoreColor(0.9, r), hi);
  const mid = { lo: 0, hi: 0.4, n: 2 };
  assert.equal(scoreT(0.2, mid), 0.5);
  assert.equal(scoreColor(0.2, mid), 'rgb(125, 146, 196)');               // 中間（丸め）
  assert.equal(scoreT(0.3, { lo: 0.3, hi: 0.3, n: 1 }), 1);               // 幅が 0 なら最大と同じ
  assert.equal(scoreColor(NaN, r), 'var(--bg-input)');
  assert.equal(scoreColor(0.3, null), 'var(--bg-input)');
  assert.equal(scoreInk(0.2, mid), 'var(--bg)');                          // 上半分は暗い字
  assert.equal(scoreInk(0.19, mid), 'var(--text)');
  assert.equal(scoreInk(0.5, r), 'var(--bg)');
  assert.equal(scoreInk(undefined, r), 'var(--text-faint)');
  // 同じ範囲なら、モデルやラベルが違っても同じスコアは同じ色（横断の物差し）
  assert.equal(scoreColor(0.42, r), scoreColor(0.42, { ...r }));
});
