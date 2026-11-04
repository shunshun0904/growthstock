import test from 'node:test';
import assert from 'node:assert/strict';
import { rankedValuations } from '../src/lib/valuation.js';
test('rank all companies by method-specific median/price, keeping unavailable last', () => {
  const make = (code, price, ri, ddm) => ({code,price,methods:[{id:'ri',available:ri!==null,median:ri},{id:'ddm',available:ddm!==null,median:ddm}]});
  const stocks=[make('B',100,90,200),make('C',100,null,null),make('A',100,150,110),make('D',0,200,200)];
  assert.deepEqual(rankedValuations(stocks,'ri').map(x=>x.code),['A','B','C','D']);
  assert.deepEqual(rankedValuations(stocks,'ddm').map(x=>x.code),['B','A','C','D']);
  assert.equal(rankedValuations(stocks,'ri')[0].deviation,.5);
  assert.equal(rankedValuations(stocks,'ri')[3].deviation,null);
  assert.deepEqual(stocks.map(x=>x.code),['B','C','A','D']);
});
