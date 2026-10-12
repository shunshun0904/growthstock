/**
 * モデル別の並び（src/lib/predictions.js modelRows）。
 *
 * - 並び順は payload の models に従い、無いモデルは出さない
 * - ラベル違い（1.5σ / 1.8σ のラベルで学習した lgbm / xgb / cat / logit の8本）は k ごとの塊（k15 / k18）で、
 *   記号は payload の short（無ければ L15 / X15 / C15 / LR15 / L18 / X18 / C18 / LR18）
 * - 表の形（modelGrid）: 行 = ラベル（1.2σ / 1.5σ / 1.8σ）、列 = 学習器。無いモデルのマスは null。古い予測ファイルは1行
 * - マスの字の色（pctInk）: 明るい区分（60 以上）は暗い字、暗い区分は明るい字
 * - 選んだ日の値の素性（dayOrigin）: 過去の日は dayInfo のモデルの学習日と出した時刻。dayInfo が無ければ今のモデル
 *
 *   node --test tests/predictions.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { modelRows, modelGrid, pctInk, MODEL_SHORT, MODEL_FAMILY, FAMILY_JA, LABEL_JA, dayOrigin }
  from '../src/lib/predictions.js';

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

test('modelRows: ラベル違いは k ごとの塊で、記号は元の記号の頭 + k', () => {
  const rows = modelRows(cand, models);
  assert.deepEqual(rows.map((r) => r.algo), ['lgbm', 'xgb', 'lgbm_k15', 'logit_k15', 'lgbm_k18', 'xgb_k18']);
  assert.deepEqual(rows.map((r) => r.family), ['tree', 'tree', 'k15', 'k15', 'k18', 'k18']);
  assert.deepEqual(rows.map((r) => r.short), ['LGB', 'XGB', 'L15', 'LR15', 'L18', 'X18']);
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
  assert.deepEqual(rows2.map((r) => r.short), ['L15', 'X18']);
});

test('modelGrid: 12本はラベル × 学習器の 3行 × 4列。並びは payload の models の順', () => {
  const learners = ['lgbm', 'xgb', 'cat', 'logit'];
  const all = [];
  const byModel = {};
  ['', '_k15', '_k18'].forEach((sfx, r) => learners.forEach((l, i) => {
    const algo = `${l}${sfx}`;
    all.push({ algo, name: algo, short: MODEL_SHORT[algo] });
    byModel[algo] = { score: 0.1, pctHistorical: 90 + r * 3 + i };       // 行と列が分かる値
  }));
  const g = modelGrid({ byModel }, all);
  assert.deepEqual(g.cols.map((c) => c.short), ['LGB', 'XGB', 'CAT', 'LR']);
  assert.deepEqual(g.rows.map((r) => r.labelJa), ['1.2σ', '1.5σ', '1.8σ']);
  assert.deepEqual(g.rows.map((r) => r.cells.map((x) => x.pct)),
    [[90, 91, 92, 93], [93, 94, 95, 96], [96, 97, 98, 99]]);
  assert.equal(g.rows[1].cells[2].algo, 'cat_k15');
  assert.equal(LABEL_JA.base, '1.2σ');
});

test('modelGrid: 無いモデルは null、古い予測ファイルは 1.2σ の1行（MLP があれば NN の列）', () => {
  // ラベル違いが一部だけの日（cand は lgbm / xgb / lgbm_k15 / logit_k15 / lgbm_k18 / xgb_k18）
  const g = modelGrid(cand, models);
  assert.deepEqual(g.cols.map((c) => c.learner), ['lgbm', 'xgb', 'logit']);
  assert.deepEqual(g.rows.map((r) => r.label), ['base', 'k15', 'k18']);
  assert.deepEqual(g.rows[0].cells.map((x) => x && x.pct), [96, 91, null]);
  assert.deepEqual(g.rows[1].cells.map((x) => x && x.pct), [94, null, 88]);
  assert.deepEqual(g.rows[2].cells.map((x) => x && x.pct), [70, 65, null]);
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
