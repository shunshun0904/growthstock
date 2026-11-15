/* Entirely synthetic UI data, never financial observations. */
export function researchFixture(n = 150) {
  const scenarios = [
    { id: 'eva-a', model: 'eva_fade5', tax: 'tax30', erp: .03, growth: 0, priceIndependent: true },
    { id: 'eva-b', model: 'eva_fade5', tax: 'tax30', erp: .07, growth: .02, priceIndependent: true },
    { id: 'business', model: 'eva_fade5', tax: 'tax30', erp: .05, growth: 'business', priceIndependent: true },
    { id: 'capm', model: 'steady', tax: 'tax30', erp: .05, growth: 0 },
  ];
  const columns = ['Date', 'price', 'price_adjusted', 'eps_forecast', 'eps_ttm', 'scenarios', 'waccScenarios',
    'comparisonWaccScenarios', 'roic_operating_tax30', 'ev_bridge_per_share', 'comparison_ev_bridge_per_share', 'share_unit_index'];
  const data = Array.from({ length: n }, (_, i) => [new Date(Date.UTC(2025, 0, i + 1)).toISOString().slice(0, 10),
    50 + i / 2, 50 + i / 2, null, 10, { 'eva-a': 100, 'eva-b': 120, business: 200, capm: 300 },
    { tax30_05: .08 }, { tax30_05: .09 }, .1, 10, 10, 1.]);
  return { schemaVersion: 1, researchOnly: true, code: '1000', name: '合成テスト銘柄', columns, data, scenarios, creditSpread: .015 };
}
