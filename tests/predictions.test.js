/**
 * モデル別の並び（src/lib/predictions.js modelRows）。
 *
 * - 並び順は payload の models に従い、無いモデルは出さない
 * - ラベル違い（1.5σ / 1.8σ のラベルで学習した lgbm / xgb / cat / logit の8本）は k ごとの塊（k15 / k18）で、
 *   記号は payload の short（無ければ L15 / X15 / C15 / LR15 / L18 / X18 / C18 / LR18）
 *
 *   node --test tests/predictions.test.js
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { modelRows, MODEL_SHORT, MODEL_FAMILY, FAMILY_JA } from '../src/lib/predictions.js';

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
