/**
 * 有価証券報告書・半期報告書（EDINET）の損益を「流れ」に変換する。描画からは独立した純粋関数。
 *
 * 元データは `public/data/filings.json`（Fetch Filings が EDINET の公式 API から取って作る）。
 * 1銘柄につき最新の書類1つ。損益の項目は EDINET の要素 ID をそのまま鍵にしてある
 * （日本基準 NetSales / CostOfSales / …、IFRS RevenueIFRS / CostOfSalesIFRS / …）。
 *
 * 決算短信のサマリー（src/lib/earnings.js）と違い、売上原価と販管費が分かるので、
 *
 *   売上高 ─┬→ 売上原価
 *           └→ 売上総利益 ─┬→ 販管費
 *                          └→ 営業利益 ─┬→ 営業外費用       （営業外収益は下から合流）
 *                                       └→ 経常利益 ─┬→ 特別損失   （特別利益は下から合流）
 *                                                    └→ 税引前利益 ─┬→ 法人税等
 *                                                                   └→ 当期純利益 ─┬→ 非支配株主持分
 *                                                                                  └→ 親会社株主に帰属
 *
 * の形が描ける。IFRS は段の名前と並びが違う（経常利益・特別損益が無く、金融収益・費用がある）。
 *
 * 出ていく額（out）は常に「前の段 + 入ってくる額 − この段」で出す。元データの項目（売上原価など）と
 * 一致するかは取り込み側（恒等式）で確かめてあり、画面は常に釣り合う図を描く。元データに出ていく項目が
 * 無い段は、名前を「…等」として引き算で出したことを note で断る。
 *
 * 損失の扱いは earnings.js と同じ。段階利益が 0 以下になったらそこで幹を止め、以降は描かない。
 */

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** 段の定義。value の鍵、出ていく額の鍵（無ければ引き算）、入ってくる額の鍵。 */
const STAGES = {
  jgaap: {
    sales: ['NetSales', 'OperatingRevenue1', 'Revenue'],
    steps: [
      { key: 'gp', label: '売上総利益', value: 'GrossProfit', out: 'CostOfSales', outLabel: '売上原価' },
      { key: 'op', label: '営業利益', value: 'OperatingIncome',
        out: 'SellingGeneralAndAdministrativeExpenses', outLabel: '販売費及び一般管理費' },
      { key: 'odp', label: '経常利益', value: 'OrdinaryIncome',
        out: 'NonOperatingExpenses', outLabel: '営業外費用', in: 'NonOperatingIncome', inLabel: '営業外収益' },
      { key: 'pbt', label: '税引前当期純利益', value: 'IncomeBeforeIncomeTaxes',
        out: 'ExtraordinaryLoss', outLabel: '特別損失', in: 'ExtraordinaryIncome', inLabel: '特別利益' },
      { key: 'np', label: '当期純利益', value: 'ProfitLoss', out: 'IncomeTaxes', outLabel: '法人税等' },
      { key: 'own', label: '親会社株主に帰属する当期純利益', value: 'ProfitLossAttributableToOwnersOfParent',
        out: 'ProfitLossAttributableToNonControllingInterests', outLabel: '非支配株主に帰属する当期純利益',
        optional: true },
    ],
  },
  ifrs: {
    sales: ['RevenueIFRS'],
    steps: [
      { key: 'gp', label: '売上総利益', value: 'GrossProfitIFRS', out: 'CostOfSalesIFRS', outLabel: '売上原価' },
      { key: 'op', label: '営業利益', value: 'OperatingProfitLossIFRS',
        out: 'SellingGeneralAndAdministrativeExpensesIFRS', outLabel: '販管費・その他の費用',
        in: 'OtherIncomeIFRS', inLabel: 'その他の収益' },
      { key: 'pbt', label: '税引前利益', value: 'ProfitLossBeforeTaxIFRS',
        out: 'FinanceCostsIFRS', outLabel: '金融費用等', in: 'FinanceIncomeIFRS', inLabel: '金融収益' },
      { key: 'np', label: '当期利益', value: 'ProfitLossIFRS', out: 'IncomeTaxExpenseIFRS', outLabel: '法人所得税費用' },
      { key: 'own', label: '親会社の所有者に帰属する当期利益', value: 'ProfitLossAttributableToOwnersOfParentIFRS',
        out: 'ProfitLossAttributableToNonControllingInterestsIFRS', outLabel: '非支配持分に帰属する当期利益',
        optional: true },
    ],
  },
};

/** 会計基準の文字列（EDINET の AccountingStandardsDEI）から段の定義を選ぶ。分からなければ null。 */
export function standardOf(stmt) {
  const s = String(stmt?.standard || '').toLowerCase();
  if (s.includes('ifrs')) return 'ifrs';
  if (s.includes('japan')) return 'jgaap';
  if (s.includes('us')) return null;                 // 米国基準は段の形が違う。対象外
  const items = stmt?.items || {};
  if (isNum(items.RevenueIFRS)) return 'ifrs';
  if (STAGES.jgaap.sales.some((k) => isNum(items[k]))) return 'jgaap';
  return null;
}

/**
 * 1つの書類の損益を段階の流れにする。描けなければ null。
 *
 * 返すもの
 *   sales      売上高（幹の太さの基準）
 *   salesLabel 売上高の名前（売上収益など）
 *   steps      {key, label, value, loss, out:{label, value, derived}, in:{label, value}} の並び
 *   truncated  損失で幹が途切れた段の key（無ければ null）
 *   missing    元データに無くて飛ばした段の key
 *   derived    出ていく額を引き算で出した段の key（元データに出ていく項目が無い）
 *   standard   'jgaap' | 'ifrs'
 */
export function toDetailedFlow(stmt) {
  const std = standardOf(stmt);
  if (!std) return null;
  const def = STAGES[std];
  const items = stmt.items || {};
  const salesKey = def.sales.find((k) => isNum(items[k]) && items[k] > 0);
  if (!salesKey) return null;
  const sales = items[salesKey];

  const steps = [];
  const missing = [];
  const derived = [];
  let prev = sales;
  let truncated = null;
  for (const d of def.steps) {
    if (truncated) break;
    const v = items[d.value];
    if (!isNum(v)) {
      if (!d.optional) missing.push(d.key);
      continue;
    }
    // 親会社株主の段は、非支配株主の取り分が無ければ描かない（同じ値が2つ並ぶだけ）
    if (d.optional && Math.abs(prev - v) <= Math.max(1, Math.abs(prev) * 1e-6)) continue;
    const inVal = d.in && isNum(items[d.in]) && items[d.in] > 0 ? items[d.in] : 0;
    const outVal = prev + inVal - v;
    const hasOut = d.out && isNum(items[d.out]);
    if (v <= 0) {
      steps.push({
        key: d.key, label: d.label, value: v, loss: true,
        out: { label: hasOut ? d.outLabel : `${d.outLabel}等`, value: prev + inVal, derived: !hasOut },
        in: inVal > 0 ? { label: d.inLabel, value: inVal } : null,
      });
      truncated = d.key;
      break;
    }
    if (!hasOut) derived.push(d.key);
    steps.push({
      key: d.key, label: d.label, value: v, loss: false,
      out: outVal > 0 ? { label: hasOut ? d.outLabel : `${d.outLabel}等`, value: outVal, derived: !hasOut } : null,
      in: inVal > 0 ? { label: d.inLabel, value: inVal } : null,
    });
    prev = v;
  }
  if (!steps.length) return null;
  return {
    sales, salesLabel: std === 'ifrs' ? '売上収益' : '売上高',
    steps, truncated, missing, derived, standard: std,
  };
}

/** 書類の種類（EDINET の docTypeCode）の表示名。 */
export const DOC_TYPE_JA = { 120: '有価証券報告書', 160: '半期報告書' };

/** 期間の表示。「2025-04-01〜2026-03-31（通期）」の形。 */
export function periodLabel(stmt) {
  if (!stmt) return '';
  const kind = stmt.periodType === 'FY' ? '通期' : stmt.periodType === 'HY' ? '上期' : (stmt.periodType || '');
  const a = stmt.periodStart || '';
  const b = stmt.periodEnd || '';
  return `${a}〜${b}${kind ? `（${kind}）` : ''}`;
}
