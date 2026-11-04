export function rankedValuations(stocks, method) {
  return (stocks || []).map(stock => {
    const value = stock.methods?.find(m => m.id === method);
    const priceRatio = value?.available && Number.isFinite(value.median) && value.median > 0
      && Number.isFinite(stock.price) && stock.price > 0 ? stock.price / value.median : null;
    return { ...stock, value, priceRatio };
  }).sort((a, b) => {
    const aa = Number.isFinite(a.priceRatio), bb = Number.isFinite(b.priceRatio);
    if (aa !== bb) return aa ? -1 : 1;
    return (aa ? a.priceRatio - b.priceRatio : 0) || a.code.localeCompare(b.code);
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
