/** Public static data only. No J-Quants credentials or browser-side inference. */
const BUILD = typeof __BUILD_ID__ === 'string' ? __BUILD_ID__ : 'dev';

export function normalizeSupplyCode(value) {
  const code = String(value ?? '').trim().toUpperCase();
  if (/^[0-9A-Z]{4}$/.test(code)) return `${code}0`;
  return /^[0-9A-Z]{5}$/.test(code) ? code : null;
}

export function createSupplyDemandClient({ baseUrl = import.meta.env?.BASE_URL ?? '/', fetchImpl = globalThis.fetch, timeoutMs = 20000 } = {}) {
  const cache = new Map();
  const base = `${baseUrl.replace(/\/?$/, '/')}data/supply-demand/`;
  async function read(file, validate) {
    if (cache.has(file)) return cache.get(file);
    const task = (async () => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetchImpl(`${base}${file}?v=${BUILD}`, { signal: controller.signal, cache: 'no-cache' });
        if (!response.ok) throw new Error(`需給データを読み込めませんでした（HTTP ${response.status}）。`);
        const data = await response.json();
        if (!validate(data)) throw new Error('需給データの形式を確認できませんでした。');
        return data;
      } catch (e) {
        if (e.name === 'AbortError') throw new Error('需給データの読み込みが時間切れになりました。再試行してください。');
        throw e;
      } finally { clearTimeout(timer); }
    })();
    cache.set(file, task);
    try { return await task; }
    catch (e) { cache.delete(file); throw e; }
  }
  return {
    loadIndex: () => read('index.json', d => d?.schemaVersion === 1 && d.stocks && typeof d.stocks === 'object'),
    async loadStock(value) {
      const code = normalizeSupplyCode(value);
      if (!code) throw new Error('証券コードを確認できませんでした。');
      const d = await read(`${code}.json`, x => x?.code === code && x.supplyDemand?.schemaVersion === 1
        && ['ok', 'missing_balance', 'missing_price', 'missing_adjustment'].includes(x.supplyDemand.status));
      return d.supplyDemand;
    },
  };
}
