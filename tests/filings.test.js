/**
 * 有報・半期報告書の損益の流れ（src/lib/filings.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { toDetailedFlow, standardOf, periodLabel } from '../src/lib/filings.js';

const jgaap = (over = {}) => ({
  standard: 'Japan GAAP', periodType: 'FY', periodStart: '2025-04-01', periodEnd: '2026-03-31',
  items: {
    NetSales: 1000, CostOfSales: 600, GrossProfit: 400,
    SellingGeneralAndAdministrativeExpenses: 250, OperatingIncome: 150,
    NonOperatingIncome: 20, NonOperatingExpenses: 10, OrdinaryIncome: 160,
    ExtraordinaryIncome: 5, ExtraordinaryLoss: 15, IncomeBeforeIncomeTaxes: 150,
    IncomeTaxes: 45, ProfitLoss: 105,
    ProfitLossAttributableToNonControllingInterests: 5, ProfitLossAttributableToOwnersOfParent: 100,
    ...over,
  },
});

test('standardOf: 会計基準の文字列、無ければ項目から', () => {
  assert.equal(standardOf({ standard: 'Japan GAAP' }), 'jgaap');
  assert.equal(standardOf({ standard: 'IFRS' }), 'ifrs');
  assert.equal(standardOf({ standard: 'US GAAP' }), null);
  assert.equal(standardOf({ items: { RevenueIFRS: 1 } }), 'ifrs');
  assert.equal(standardOf({ items: { NetSales: 1 } }), 'jgaap');
  assert.equal(standardOf({}), null);
});

test('toDetailedFlow: 日本基準の6段。出ていく額は引き算で釣り合う', () => {
  const f = toDetailedFlow(jgaap());
  assert.equal(f.standard, 'jgaap');
  assert.equal(f.sales, 1000);
  assert.equal(f.salesLabel, '売上高');
  assert.deepEqual(f.steps.map((s) => s.key), ['gp', 'op', 'odp', 'pbt', 'np', 'own']);
  const by = Object.fromEntries(f.steps.map((s) => [s.key, s]));
  assert.deepEqual(by.gp.out, { label: '売上原価', value: 600, derived: false });
  assert.equal(by.gp.in, null);
  assert.deepEqual(by.op.out, { label: '販売費及び一般管理費', value: 250, derived: false });
  assert.deepEqual(by.odp.in, { label: '営業外収益', value: 20 });
  assert.deepEqual(by.odp.out, { label: '営業外費用', value: 10, derived: false });   // 150 + 20 − 160
  assert.deepEqual(by.pbt.in, { label: '特別利益', value: 5 });
  assert.equal(by.pbt.out.value, 15);
  assert.equal(by.np.out.value, 45);
  assert.equal(by.own.out.value, 5);
  assert.equal(f.truncated, null);
  assert.deepEqual(f.missing, []);
  assert.deepEqual(f.derived, []);
  // 各段で 前 + in − out = value
  let prev = f.sales;
  for (const s of f.steps) {
    assert.equal(prev + (s.in?.value || 0) - (s.out?.value || 0), s.value, s.key);
    prev = s.value;
  }
});

test('toDetailedFlow: 出ていく項目が無い段は「…等」と derived に印', () => {
  const f = toDetailedFlow(jgaap({ NonOperatingExpenses: undefined }));
  const odp = f.steps.find((s) => s.key === 'odp');
  assert.equal(odp.out.label, '営業外費用等');
  assert.equal(odp.out.value, 10);
  assert.equal(odp.out.derived, true);
  assert.deepEqual(f.derived, ['odp']);
});

test('toDetailedFlow: 非支配株主の取り分が無ければ親会社株主の段は描かない。段が無ければ missing', () => {
  const f = toDetailedFlow(jgaap({ ProfitLossAttributableToOwnersOfParent: 105,
                                   ProfitLossAttributableToNonControllingInterests: 0 }));
  assert.deepEqual(f.steps.map((s) => s.key), ['gp', 'op', 'odp', 'pbt', 'np']);
  const g = toDetailedFlow(jgaap({ GrossProfit: undefined, OrdinaryIncome: undefined }));
  assert.deepEqual(g.missing, ['gp', 'odp']);
  assert.deepEqual(g.steps.map((s) => s.key), ['op', 'pbt', 'np', 'own']);
  assert.equal(g.steps[0].out.value, 850);                                // 1000 − 150（原価＋販管費）
});

test('toDetailedFlow: 赤字の段で幹を止める', () => {
  const f = toDetailedFlow(jgaap({ OperatingIncome: -50, OrdinaryIncome: -40 }));
  assert.deepEqual(f.steps.map((s) => s.key), ['gp', 'op']);
  const op = f.steps[1];
  assert.equal(op.loss, true);
  assert.equal(op.value, -50);
  assert.equal(op.out.value, 400);                                         // 前の段がまるごと出ていく
  assert.equal(f.truncated, 'op');
});

test('toDetailedFlow: IFRS の5段（経常・特別損益は無く、金融収益・費用がある）', () => {
  const f = toDetailedFlow({
    standard: 'IFRS',
    items: { RevenueIFRS: 1000, CostOfSalesIFRS: 600, GrossProfitIFRS: 400,
             SellingGeneralAndAdministrativeExpensesIFRS: 230, OtherIncomeIFRS: 10, OtherExpensesIFRS: 30,
             OperatingProfitLossIFRS: 150, FinanceIncomeIFRS: 8, FinanceCostsIFRS: 18,
             ProfitLossBeforeTaxIFRS: 140, IncomeTaxExpenseIFRS: 40, ProfitLossIFRS: 100,
             ProfitLossAttributableToOwnersOfParentIFRS: 100 },
  });
  assert.equal(f.standard, 'ifrs');
  assert.equal(f.salesLabel, '売上収益');
  assert.deepEqual(f.steps.map((s) => s.key), ['gp', 'op', 'pbt', 'np']);
  const op = f.steps[1];
  assert.deepEqual(op.in, { label: 'その他の収益', value: 10 });
  assert.equal(op.out.value, 260);                                         // 400 + 10 − 150（販管費 230 + その他 30）
  assert.equal(op.out.label, '販管費・その他の費用');
  const pbt = f.steps[2];
  assert.deepEqual(pbt.in, { label: '金融収益', value: 8 });
  assert.equal(pbt.out.value, 18);
});

test('toDetailedFlow: 描けないとき null', () => {
  assert.equal(toDetailedFlow(null), null);
  assert.equal(toDetailedFlow({ standard: 'US GAAP', items: { NetSales: 1 } }), null);
  assert.equal(toDetailedFlow({ standard: 'Japan GAAP', items: { NetSales: 0 } }), null);
  assert.equal(toDetailedFlow({ standard: 'Japan GAAP', items: { NetSales: 100 } }), null);   // 段が1つも無い
});

test('periodLabel', () => {
  assert.equal(periodLabel({ periodType: 'FY', periodStart: '2025-04-01', periodEnd: '2026-03-31' }),
               '2025-04-01〜2026-03-31（通期）');
  assert.equal(periodLabel({ periodType: 'HY', periodStart: '2026-04-01', periodEnd: '2026-09-30' }),
               '2026-04-01〜2026-09-30（上期）');
  assert.equal(periodLabel(null), '');
});
