export const PROFILE_WINDOWS = [1, 5, 20];
const MAX_PROFILE_BYTES = 75 * 1024 * 1024;
export const PROFILE_REPO = import.meta.env?.VITE_VOLUME_PROFILE_REPO || 'shunshun0904/growthstock-minute';

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

export function latestVolumeTargets(predictions) {
  if (!date(predictions?.asOf) || !Array.isArray(predictions.candidates)) return [];
  const targets = new Map();
  for (const row of predictions.candidates) {
    if (row?.date !== predictions.asOf) continue;
    const code = volumeCode(row.jqCode || row.code);
    if (!/^[0-9A-Z]{5}$/.test(code) || targets.has(code)) continue;
    targets.set(code, { code, name: row.name || code, rank: row.rankInDay });
  }
  return [...targets.values()].sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));
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
  catch { throw new Error('価格帯別出来高の応答を読み込めませんでした。'); }
  if (data?.kind !== 'growthstock.volume-profile' || data.schemaVersion !== 1 || !date(data.asOf)
      || !Array.isArray(data.stocks) || data.demo === true) {
    throw new Error('価格帯別出来高のデータ形式が一致しません。');
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

export async function loadVolumeProfile(token, signal) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(PROFILE_REPO)) throw new Error('非公開リポジトリの設定が無効です。');
  const endpoint = `https://api.github.com/repos/${PROFILE_REPO}/contents/dashboard/volume-profile.json`;
  const response = await fetch(endpoint, {
    signal, credentials: 'omit', redirect: 'error', cache: 'no-store', referrerPolicy: 'no-referrer',
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github.raw+json' },
  });
  if (response.status === 404) throw new Error('当日の集計がまだ保存されていないか、非公開リポジトリの閲覧権限がありません。');
  if (response.status === 401) throw new Error('閲覧用トークンが無効です。再接続してください。');
  if (response.status === 403 || response.status === 429) throw new Error('GitHubの閲覧権限または利用上限を確認してください。');
  if (!response.ok) throw new Error('集計データの取得に失敗しました。時間を置いて再取得してください。');
  if (Number(response.headers.get('content-length')) > MAX_PROFILE_BYTES) throw new Error('集計データが大きすぎます。');
  const text = await response.text();
  if (text.length > MAX_PROFILE_BYTES) throw new Error('集計データが大きすぎます。');
  return parseVolumeProfile(text);
}
