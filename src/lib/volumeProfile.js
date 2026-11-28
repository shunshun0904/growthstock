/**
 * 「価格帯別出来高」タブのデータ（docs/VOLUME_PROFILE.md）。
 *
 * 集計（data/volume-profile.json）は Update Volume Profiles が蓄積した分足・ティックから作り、デプロイで画面に載せる。
 * 運用者の選択（2026-11-19「暗号化をやめて公開する」）で、暗号化せずに載せ、キーの入力なしで開く。
 */
export const PROFILE_WINDOWS = [1, 5, 20];
export const SOURCES = ['tick', 'minute'];
export const BUCKETS = ['open', 'morning', 'afternoon', 'close'];
export const SIZE_LABELS = ['small', 'medium', 'large'];
export const TARGET_DAYS = 5;

const PROFILE_KIND = 'growthstock.volume-profile';
const MAX_BYTES = 20 * 1024 * 1024;
/** 暗号化していたころ（2026-11-19 の #11・#12 の途中）に端末へ覚えた復号の鍵。もう使わないので消す */
const OLD_UNLOCK_KEY = 'growthstock.volume-profile.unlock.v1';
const STATUSES = new Set([
  'ok', 'calendar_history_missing', 'daily_missing', 'invalid_daily', 'close_missing',
  'share_basis_unknown', 'intraday_missing', 'intraday_invalid', 'daily_totals_mismatch',
]);

const finite = (n) => typeof n === 'number' && Number.isFinite(n);
const positive = (n) => finite(n) && n > 0;
const nonnegative = (n) => finite(n) && n >= 0;
const proportion = (n) => nonnegative(n) && n <= 1 + 1e-8;
const isDate = (value) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
const range = (value) => Array.isArray(value) && value.length === 2
  && nonnegative(value[0]) && positive(value[1]) && value[1] > value[0];
const closeTo = (a, b, tolerance = 1e-8) => Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * tolerance);
const sumsToOne = (values) => Math.abs(values.reduce((t, v) => t + v, 0) - 1) <= 1e-6;

export const profileUrl = () => `${import.meta.env?.BASE_URL ?? '/'}data/volume-profile.json`;

export function volumeCode(value) {
  const code = String(value ?? '').trim().toUpperCase();
  return /^[0-9A-Z]{4}$/.test(code) ? `${code}0` : code;
}

/** 直近 days 個の予測日の候補すべて（銘柄で重複を除く）。並びは最新の候補日の新しい順・その日の順位の順。 */
export function volumeTargets(predictions, days = TARGET_DAYS) {
  if (!isDate(predictions?.asOf) || !Array.isArray(predictions.candidates)) return { asOf: null, dates: [], targets: [] };
  const { asOf } = predictions;
  const listed = new Set([...(Array.isArray(predictions.dates) ? predictions.dates : []),
    ...predictions.candidates.map((row) => row?.date)].filter((d) => isDate(d) && d <= asOf));
  const dates = [...listed].sort().slice(-days);
  const keep = new Set(dates);
  const byCode = new Map();
  for (const row of predictions.candidates) {
    if (!keep.has(row?.date)) continue;
    const code = volumeCode(row.jqCode || row.code);
    if (!/^[0-9A-Z]{5}$/.test(code)) continue;
    const entry = byCode.get(code) ?? { code, name: String(row.name || code), candidates: [] };
    entry.candidates.push({ date: row.date, rank: Number.isInteger(row.rankInDay) && row.rankInDay > 0 ? row.rankInDay : null });
    byCode.set(code, entry);
  }
  const targets = [...byCode.values()].map((t) => ({
    ...t, candidates: t.candidates.sort((a, b) => b.date.localeCompare(a.date) || (a.rank ?? Infinity) - (b.rank ?? Infinity)),
  }));
  targets.sort((a, b) => b.candidates[0].date.localeCompare(a.candidates[0].date)
    || (a.candidates[0].rank ?? Infinity) - (b.candidates[0].rank ?? Infinity) || a.code.localeCompare(b.code));
  return { asOf, dates, targets };
}

function validSessions(sessions) {
  return sessions && typeof sessions === 'object' && Object.keys(sessions).sort().join() === [...BUCKETS].sort().join()
    && BUCKETS.every((b) => proportion(sessions[b])) && sumsToOne(BUCKETS.map((b) => sessions[b]));
}

function validTradeSizes(sizes) {
  if (!sizes || !nonnegative(sizes.turnover) || !Array.isArray(sizes.shares) || !Array.isArray(sizes.counts)
      || sizes.shares.length !== 3 || sizes.counts.length !== 3 || !sizes.shares.every(proportion)
      || !sizes.counts.every((n) => Number.isInteger(n) && n >= 0)) return false;
  return sizes.turnover > 0 ? sumsToOne(sizes.shares) : sizes.shares.every((s) => s === 0);
}

function validBand(band) {
  return Array.isArray(band) && band.length === 4 && range(band.slice(0, 2))
    && nonnegative(band[2]) && nonnegative(band[3]) && (band[2] > 0) === (band[3] > 0);
}

function validSeries(series, days, asOf, close, source) {
  if (!series || !STATUSES.has(series.status) || !Number.isInteger(series.coveredSessions)
      || series.coveredSessions < 0 || series.coveredSessions > days) return false;
  if (series.status !== 'ok') return series.reasons === undefined
    || (Array.isArray(series.reasons) && series.reasons.every((r) => STATUSES.has(r)));
  const { bands } = series;
  if (series.coveredSessions !== days || !isDate(series.from) || series.from > asOf || series.to !== asOf
      || !positive(close) || !positive(series.vwap) || !positive(series.shares) || !positive(series.turnover)
      || !finite(series.closeVsVwap) || !closeTo(series.closeVsVwap + 1, close / series.vwap, 1e-9)
      || !range(series.poc) || !range(series.valueArea) || !proportion(series.pocShare)
      || !proportion(series.aboveCloseShare) || !proportion(series.nearCloseShare)
      || !validSessions(series.sessions)
      || (source === 'tick' ? !validTradeSizes(series.tradeSizes) : series.tradeSizes !== undefined)
      || (series.tickMinuteDistance !== undefined && !proportion(series.tickMinuteDistance))
      || !Array.isArray(bands) || bands.length < 1 || bands.length > 60 || !bands.every(validBand)) return false;
  const shares = bands.reduce((total, b) => total + b[2], 0);
  const turnover = bands.reduce((total, b) => total + b[3], 0);
  return closeTo(shares, series.shares) && closeTo(turnover, series.turnover)
    && bands.every((b, i) => i === 0 || b[0] === bands[i - 1][1])
    && close >= bands[0][0] && close <= bands.at(-1)[1];
}

const validCandidate = (c) => c && isDate(c.date) && (c.rank === null || (Number.isInteger(c.rank) && c.rank > 0));

export function parseVolumeProfile(text) {
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error('集計を読み込めませんでした。'); }
  if (data?.kind !== PROFILE_KIND || data.schemaVersion !== 2 || !isDate(data.asOf) || !Array.isArray(data.stocks)
      || !Array.isArray(data.targetDates) || !data.targetDates.every(isDate)) {
    throw new Error('集計の形式が一致しません。');
  }
  if (data.demo !== false) throw new Error('合成データの集計は表示しません。');
  const codes = new Set();
  for (const stock of data.stocks) {
    if (!stock || typeof stock.code !== 'string' || !/^[0-9A-Z]{5}$/.test(stock.code) || codes.has(stock.code)
        || typeof stock.name !== 'string' || stock.name.length > 200 || !(stock.close === null || positive(stock.close))
        || !Array.isArray(stock.candidates) || !stock.candidates.every(validCandidate)
        || !PROFILE_WINDOWS.every((days) => SOURCES.every((source) =>
          validSeries(stock.windows?.[days]?.[source], days, data.asOf, stock.close, source)))) {
      throw new Error('銘柄・営業日・価格帯の形式を確認してください。');
    }
    codes.add(stock.code);
  }
  return data;
}

/** 集計を取る。まだ無ければ null。 */
export async function loadVolumeProfile(signal) {
  const response = await fetch(profileUrl(), { signal, cache: 'no-store', credentials: 'same-origin' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error('集計を取得できませんでした。時間を置いて再読み込みしてください。');
  if (Number(response.headers.get('content-length')) > MAX_BYTES) throw new Error('集計が大きすぎます。');
  const text = await response.text();
  if (text.length > MAX_BYTES) throw new Error('集計が大きすぎます。');
  return parseVolumeProfile(text);
}

/** 暗号化していたころに端末へ覚えた復号の鍵を消す（もう使わない）。 */
export function forgetOldUnlock() {
  for (const name of ['sessionStorage', 'localStorage']) {
    try { globalThis[name]?.removeItem(OLD_UNLOCK_KEY); }
    catch { /* 保存が使えなくても画面は動く */ }
  }
}
