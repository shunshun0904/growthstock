export function rankedValuations(stocks, method) {
  return (stocks || []).map(stock => {
    const value = stock.methods?.find(m => m.id === method);
    const deviation = value?.available && Number.isFinite(value.median)
      && Number.isFinite(stock.price) && stock.price > 0 ? value.median / stock.price - 1 : null;
    return { ...stock, value, deviation };
  }).sort((a, b) => {
    const aa = Number.isFinite(a.deviation), bb = Number.isFinite(b.deviation);
    if (aa !== bb) return aa ? -1 : 1;
    return (aa ? b.deviation - a.deviation : 0) || a.code.localeCompare(b.code);
  });
}
export async function loadValuations() {
  const base = import.meta.env?.BASE_URL ?? '/';
  const response = await fetch(`${base}data/valuations.json?v=${typeof __BUILD_ID__ === 'string' ? __BUILD_ID__ : 'dev'}`);
  if (!response.ok) throw new Error(`理論株価データを取得できません（HTTP ${response.status}）`);
  const data = await response.json();
  if (data.schemaVersion !== 1 || !Array.isArray(data.stocks)) throw new Error('理論株価データの形式が不正です');
  return data;
}
