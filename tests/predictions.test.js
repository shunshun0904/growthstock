/**
 * モデル別の並び（src/lib/predictions.js modelRows）。
 *
 * - 並び順は payload の models に従い、無いモデルは出さない
 * - ラベル違い（lgbm_k15 / lgbm_k18。1.5σ / 1.8σ で学習した LightGBM）は「ラベル違い」の塊で、
 *   記号は payload の short（無ければ L15 / L18）
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
  { algo: 'lgbm_k18', name: 'LightGBM（1.8σ）', short: 'L18', variant: true },
];

const cand = {
  byModel: {
    lgbm: { score: 0.5, pctHistorical: 96 },
    xgb: { score: 0.4, pctHistorical: 91 },
    lgbm_k15: { score: 0.3, pctHistorical: 94 },
    lgbm_k18: { score: 0.2, pctHistorical: 70 },
  },
};

test('modelRows: ラベル違いは variant の塊で、記号は L15 / L18', () => {
  const rows = modelRows(cand, models);
  assert.deepEqual(rows.map((r) => r.algo), ['lgbm', 'xgb', 'lgbm_k15', 'lgbm_k18']);
  assert.deepEqual(rows.map((r) => r.family), ['tree', 'tree', 'variant', 'variant']);
  assert.deepEqual(rows.map((r) => r.short), ['LGB', 'XGB', 'L15', 'L18']);
  assert.equal(rows[2].pct, 94);
  assert.equal(FAMILY_JA.variant, 'ラベル違い');
  assert.equal(MODEL_FAMILY.lgbm_k15, 'variant');
  assert.equal(MODEL_SHORT.lgbm_k18, 'L18');
});

test('modelRows: payload に無いモデルは並べない（古い予測ファイル）', () => {
  const rows = modelRows(cand, models.slice(0, 2));
  assert.deepEqual(rows.map((r) => r.algo), ['lgbm', 'xgb']);
  // short が無い models でも控えの記号で出る
  const rows2 = modelRows(cand, [{ algo: 'lgbm_k15', name: 'x' }]);
  assert.equal(rows2[0].short, 'L15');
});
