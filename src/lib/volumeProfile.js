export const PROFILE_WINDOWS = [1, 5, 20];
export const MAX_PROFILE_FILE_BYTES = 12 * 1024 * 1024;

const SOURCES = ['tick', 'minute'];
const STATUSES = new Set([
  'ok', 'calendar_history_missing', 'daily_missing', 'not_available_at_cutoff',
  'invalid_daily', 'close_missing', 'share_basis_unknown', 'intraday_missing', 'daily_totals_mismatch',
]);
const finite = (n) => typeof n === 'number' && Number.isFinite(n);
const positive = (n) => finite(n) && n > 0;
const nonnegative = (n) => finite(n) && n >= 0;
const proportion = (n) => nonnegative(n) && n <= 1 + 1e-8;
const date = (value) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
const range = (value) => Array.isArray(value) && value.length === 2
  && nonnegative(value[0]) && positive(value[1]) && value[1] > value[0];
const closeTo = (a, b) => Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * 1e-8);

export function volumeCode(value) {
  const code = String(value ?? '').trim().toUpperCase();
  return /^[0-9A-Z]{4}$/.test(code) ? `${code}0` : code;
}

function validBand(band) {
  return Array.isArray(band) && band.length === 4 && range(band.slice(0, 2))
    && nonnegative(band[2]) && nonnegative(band[3]) && (band[2] > 0) === (band[3] > 0);
}

function validSeries(series, days, asOf, close) {
  if (!series || !STATUSES.has(series.status) || !Number.isInteger(series.coveredSessions)
      || series.coveredSessions < 0 || series.coveredSessions > days) return false;
  if (series.status !== 'ok') return true;
  const { bands } = series;
  if (series.coveredSessions !== days || !date(series.from) || series.from > asOf || series.to !== asOf
      || !positive(close) || !positive(series.vwap) || !positive(series.shares) || !positive(series.turnover)
      || !range(series.poc) || !range(series.valueArea) || !proportion(series.pocShare)
      || !proportion(series.aboveCloseShare) || !proportion(series.nearCloseShare)
      || (series.tickMinuteDistance !== undefined && !proportion(series.tickMinuteDistance))
      || !Array.isArray(bands) || bands.length < 1 || bands.length > 60 || !bands.every(validBand)) return false;
  const shares = bands.reduce((total, b) => total + b[2], 0);
  const turnover = bands.reduce((total, b) => total + b[3], 0);
  return closeTo(shares, series.shares) && closeTo(turnover, series.turnover)
    && bands.every((b, i) => i === 0 || b[0] === bands[i - 1][1])
    && close >= bands[0][0] && close <= bands.at(-1)[1];
}

export function parseVolumeProfile(text) {
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error('集計ファイルを読み込めません。JSON形式を確認してください。'); }
  if (data?.kind !== 'growthstock.volume-profile' || data.schemaVersion !== 1 || !date(data.asOf)
      || !Array.isArray(data.stocks) || data.stocks.length === 0 || data.stocks.length > 250) {
    throw new Error('価格帯別出来高の集計ファイルを選んでください。');
  }
  const codes = new Set();
  for (const stock of data.stocks) {
    if (!stock || typeof stock.code !== 'string' || !/^[0-9A-Z]{5}$/.test(stock.code)
        || codes.has(stock.code) || stock.date !== data.asOf || !(stock.close === null || positive(stock.close))
        || typeof stock.name !== 'string' || stock.name.length > 200
        || !PROFILE_WINDOWS.every((days) => SOURCES.every((source) =>
          validSeries(stock.windows?.[days]?.[source], days, data.asOf, stock.close)))) {
      throw new Error('銘柄・営業日・価格帯の形式を確認してください。');
    }
    codes.add(stock.code);
  }
  return data;
}
