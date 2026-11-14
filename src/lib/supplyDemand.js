/** Display helpers only. Raw margin balances are never inferred from model scores. */
export const finite = v => typeof v === 'number' && Number.isFinite(v);
export const yen = v => finite(v) ? `${v.toLocaleString('ja-JP', { maximumFractionDigits: 1 })}円` : '未取得';
export const shares = v => finite(v) ? `${v.toLocaleString('ja-JP', { maximumFractionDigits: 0 })}株` : '未取得';
export const percent = v => finite(v) ? `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%` : '算出不可';

export function supplyChoices(stocks = [], candidates = []) {
  const byCode = new Map();
  for (const stock of stocks) byCode.set(String(stock.jqCode || stock.code), stock);
  for (const c of candidates) {
    const code = String(c.jqCode || c.code);
    if (!byCode.has(code)) byCode.set(code, { ...c, id: `pred:${code}` });
  }
  return [...byCode.values()].sort((a, b) => String(a.code).localeCompare(String(b.code), 'ja'));
}

export function profileBands(profile) {
  const bands = new Map();
  for (const side of ['long', 'short']) {
    for (const bin of profile?.[side]?.bins || []) {
      if (![bin.low, bin.high, bin.price, bin.shares].every(finite) || bin.shares < 0) continue;
      const key = `${bin.low}:${bin.high}`;
      if (!bands.has(key)) bands.set(key, { low: bin.low, high: bin.high, long: 0, short: 0 });
      const band = bands.get(key);
      band[side] += bin.shares;
    }
  }
  return [...bands.values()].sort((a, b) => b.low - a.low);
}

export function sideDisplay(side, profile) {
  const observed = finite(side?.averagePrice);
  return {
    average: observed ? side.averagePrice : profile?.estimatedAveragePrice,
    pnl: observed ? side.referencePnlRate : profile?.estimatedPnlRate,
    estimated: !observed,
  };
}

export const PROFILE_STATUS = {
  missing_quantity: '信用残の株数が未取得のため分布を算出できません。',
  missing_volume: '対象期間の出来高を取得できず、価格帯への配分ができません。',
  mean_outside_window: '平均建値が対象期間の価格範囲外のため分布を推定できません。より長い期間を選んでください。',
  no_positions: '公表された信用残は0株です。',
};
