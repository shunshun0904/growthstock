/**
 * データロードと状態の永続化。
 *
 * - J-Quants 由来のデータ  : public/data/stocks.json (GitHub Actions が生成)
 * - 利用者が手入力した銘柄 : localStorage (仕様書 §5.4)
 *
 * 両者は origin フィールド ('jquants' / 'manual') で必ず区別し、
 * UI 上でも見分けられるようにする。
 */

const LS_MANUAL = 'focus.manualStocks.v1';
const LS_VISIBLE = 'focus.visible.v1';

const DATA_URL = () => `${import.meta.env?.BASE_URL ?? '/'}data/stocks.json`;

export async function loadDataset() {
  const res = await fetch(DATA_URL(), { cache: 'no-cache' });
  if (!res.ok) {
    throw new Error(
      `データファイルを読み込めませんでした (HTTP ${res.status})。` +
      'GitHub Actions の「Fetch J-Quants Data」ワークフローを実行してください。'
    );
  }
  const json = await res.json();
  if (!json || !Array.isArray(json.stocks)) {
    throw new Error('stocks.json の形式が不正です');
  }
  return json;
}

/* ------------------------------------------------------------------ */

function safeParse(raw, fallback) {
  if (!raw) return fallback;
  try {
    const v = JSON.parse(raw);
    return v ?? fallback;
  } catch {
    return fallback;
  }
}

export function loadManualStocks() {
  try {
    return safeParse(localStorage.getItem(LS_MANUAL), []);
  } catch {
    return [];
  }
}

export function saveManualStocks(stocks) {
  try {
    localStorage.setItem(LS_MANUAL, JSON.stringify(stocks));
  } catch {
    /* プライベートウィンドウ等で書けなくても致命的ではない */
  }
}

/**
 * 手入力銘柄と表示状態を全部消す。
 *
 * ウォッチリストを空にしたので、画面に残るのは利用者が足した銘柄だけになる。
 * 古い銘柄が localStorage に残って消せないと、画面を初期状態に戻す手段が
 * 「ブラウザのサイトデータを消す」しかなくなる。
 */
export function resetSavedState() {
  for (const k of [LS_MANUAL, LS_VISIBLE]) {
    try { localStorage.removeItem(k); } catch { /* 消せなくても致命的ではない */ }
  }
}

export function loadVisibility() {
  try {
    return safeParse(localStorage.getItem(LS_VISIBLE), null);
  } catch {
    return null;
  }
}

export function saveVisibility(map) {
  try {
    localStorage.setItem(LS_VISIBLE, JSON.stringify(map));
  } catch {
    /* noop */
  }
}

/** 手入力フォームの値から、J-Quants 由来データと同じ形状の銘柄オブジェクトを作る。 */
export function manualStockFromForm(form) {
  const num = (v) => {
    if (v === '' || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  const metrics = {
    price: num(form.price),
    highRatio: num(form.highRatio),
    high52w: num(form.high52w),
    tradingValue: num(form.tradingValue),
    marketCap: num(form.marketCap),
    volumeTrend: num(form.volumeTrend),
    epsGrowth: num(form.epsGrowth),
    salesGrowth: num(form.salesGrowth),
    roe: num(form.roe),
    opMargin: num(form.opMargin),
    creditRatio: num(form.creditRatio),
    progressRate: num(form.progressRate),
    quarter: num(form.quarter),
    date: null,
  };

  const sources = Object.fromEntries(
    Object.entries(metrics).map(([k, v]) => [k, v === null ? 'unavailable' : 'manual'])
  );

  return {
    id: form.id || `manual-${form.code}-${Date.now()}`,
    code: String(form.code || '').trim(),
    name: String(form.name || '').trim() || String(form.code || '').trim(),
    note: form.note || '',
    sector: form.sector || null,
    origin: 'manual',
    asOf: null,
    metrics,
    snapshots: { now: { ...metrics, label: '現在', asOf: null } },
    milestones: [],
    history: [],
    sources,
  };
}

/** J-Quants 由来 / 手入力 の銘柄を1つのリストに束ねる (id を必ず付与)。 */
/**
 * J-Quants 由来の銘柄と、利用者が足した銘柄をまとめる。
 *
 * 同じ銘柄が両側にあるときは J-Quants 側を残す。
 * 予測タブから送った銘柄は「その日の値」しか持たないが、日次の取得が
 * 同じ銘柄を拾うと過去スナップショットと株価履歴の付いた版が入る。
 * 両方を並べると、利用者が薄いほうを選んでタイムマシーンが
 * 「過去時点がありません」になる。厚いほうを残す。
 */
export function mergeStocks(dataset, manual) {
  const fromApi = (dataset?.stocks ?? []).map((s) => ({ ...s, id: `jq-${s.jqCode || s.code}` }));
  const apiCodes = new Set(fromApi.map((s) => String(s.jqCode || s.code)));
  const rest = (manual ?? []).filter(
    (s) => !apiCodes.has(String(s.jqCode || s.code))
  );
  return [...fromApi, ...rest];
}
