import test from 'node:test';
import assert from 'node:assert/strict';
import {modelGrid, modelRows} from '../src/lib/predictions.js';
test('frozen sigma predictions retain sigma rows with a new model registry', () => {
  const models=[{algo:'lgbm',labelSpec:{kind:'fixed_return',threshold:.10}},{algo:'cat_k15',name:'CatBoost（+15%）',labelSpec:{kind:'fixed_return',threshold:.15}}];
  const c={byModel:{lgbm:{score:.2,pctHistorical:80},cat_k15:{score:.1,pctHistorical:90}}};
  assert.deepEqual(modelGrid(c,models).rows.map(r=>r.labelJa),['1.2σ','1.5σ']);
  assert.equal(modelRows(c,models)[1].name,'CatBoost（1.5σ）');
});
test('new predictions expose their actual fixed-return targets', () => {
  const c={byModel:{lgbm:{score:.2,labelSpec:{kind:'fixed_return',threshold:.10}},cat_k15:{score:.1,labelSpec:{kind:'fixed_return',threshold:.15}}}};
  assert.deepEqual(modelGrid(c,[]).rows.map(r=>r.labelJa),['+10%','+15%']);
});
