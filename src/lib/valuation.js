export function rangePosition(price, low, high) {
  if (![price, low, high].every(Number.isFinite) || high <= low) return null;
  return (price - low) / (high - low);
}
export function trend(history) {
  const good = (history || []).filter(x => Number.isFinite(x.close) && x.close > 0);
  return good.length >= 2 ? good.at(-1).close / good[0].close - 1 : null;
}
export async function loadValuations() {
  const base = import.meta.env?.BASE_URL ?? '/';
  const response = await fetch(`${base}data/valuations.json?v=${typeof __BUILD_ID__ === 'string' ? __BUILD_ID__ : 'dev'}`);
  if (!response.ok) throw new Error(`理論株価データを取得できません（HTTP ${response.status}）`);
  const data = await response.json();
  if (data.schemaVersion !== 1 || !Array.isArray(data.stocks)) throw new Error('理論株価データの形式が不正です');
  return data;
}
