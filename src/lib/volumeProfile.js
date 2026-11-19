/**
 * 「価格帯別出来高」タブのデータ（docs/VOLUME_PROFILE.md）。
 *
 * 集計は J-Quants のアドオン（分足・ティック）から作るので、Pages には暗号文
 * （data/volume-profile.enc.json。research/volume_profile_crypto.py が書く）だけを載せる。利用者が J-Quants の
 * API キーを入れると、ブラウザの中で同じ鍵（PBKDF2-HMAC-SHA256）を作り、AES-GCM で開く。キーはどこにも送らない。
 * 端末に覚えるのはキーから作った鍵だけ（キーそのものは保存しない）。
 */
export const PROFILE_WINDOWS = [1, 5, 20];
export const SOURCES = ['tick', 'minute'];
export const BUCKETS = ['open', 'morning', 'afternoon', 'close'];
export const SIZE_LABELS = ['small', 'medium', 'large'];
export const TARGET_DAYS = 5;

const ENVELOPE_KIND = 'growthstock.volume-profile.encrypted';
const ENVELOPE_KEYS = ['asOf', 'cipher', 'compression', 'data', 'generatedAt', 'kdf', 'kind', 'version'];
const PROFILE_KIND = 'growthstock.volume-profile';
const MAX_BYTES = 20 * 1024 * 1024;
const UNLOCK_KEY = 'growthstock.volume-profile.unlock.v1';
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

export class KeyMismatchError extends Error {
  constructor() {
    super('キーが一致しません。J-Quants の API キーを確かめてください（キーを変えた場合は、次の集計（毎晩）から新しいキーで開けます）。');
    this.name = 'KeyMismatchError';
  }
}

export const profileUrl = () => `${import.meta.env?.BASE_URL ?? '/'}data/volume-profile.enc.json`;

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

function fromBase64(text) {
  if (typeof text !== 'string' || !/^[A-Za-z0-9+/]*={0,2}$/.test(text) || text.length % 4) throw new Error('base64');
  const raw = atob(text);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
  return out;
}

function toBase64(bytes) {
  let raw = '';
  for (const b of bytes) raw += String.fromCharCode(b);
  return btoa(raw);
}

/** 暗号文の見出しを確かめる（research/volume_profile_crypto.py の check_envelope と同じ条件）。 */
export function parseEnvelope(text) {
  let doc;
  try { doc = JSON.parse(text); }
  catch { throw new Error('集計のファイルを読み込めませんでした。'); }
  const bad = () => new Error('集計のファイルの形式が一致しません。');
  if (!doc || typeof doc !== 'object' || Object.keys(doc).sort().join() !== ENVELOPE_KEYS.join()
      || doc.kind !== ENVELOPE_KIND || doc.version !== 1 || doc.compression !== 'gzip' || !isDate(doc.asOf)
      || typeof doc.generatedAt !== 'string' || !doc.generatedAt
      || Object.keys(doc.kdf ?? {}).sort().join() !== 'hash,iterations,name,salt' || doc.kdf.name !== 'PBKDF2'
      || doc.kdf.hash !== 'SHA-256' || !Number.isInteger(doc.kdf.iterations)
      || doc.kdf.iterations < 100_000 || doc.kdf.iterations > 10_000_000
      || Object.keys(doc.cipher ?? {}).sort().join() !== 'iv,name,tagLength' || doc.cipher.name !== 'AES-GCM'
      || doc.cipher.tagLength !== 128) throw bad();
  let bytes;
  try { bytes = { salt: fromBase64(doc.kdf.salt), iv: fromBase64(doc.cipher.iv), data: fromBase64(doc.data) }; }
  catch { throw bad(); }
  if (bytes.salt.length !== 16 || bytes.iv.length !== 12 || bytes.data.length < 36) throw bad();
  return { ...doc, bytes };
}

/** 追加認証データ（research/volume_profile_crypto.py の aad と同じ並び）。 */
export function envelopeAad(doc) {
  const { kdf, cipher } = doc;
  return new TextEncoder().encode([doc.kind, doc.version, doc.asOf, doc.generatedAt, kdf.name, kdf.hash,
    kdf.iterations, kdf.salt, cipher.name, cipher.iv, cipher.tagLength, doc.compression].join('\n'));
}

function subtleCrypto() {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('このブラウザでは集計を開けません（Web Crypto が使えません）。');
  return subtle;
}

/** J-Quants の API キーから復号の鍵（256 ビット）を作る。キーはこの関数の外に出さない。 */
export async function deriveKeyBits(secret, envelope) {
  const value = String(secret ?? '').trim();
  if (!value) throw new Error('J-Quants の API キーを入力してください。');
  const subtle = subtleCrypto();
  const material = await subtle.importKey('raw', new TextEncoder().encode(value), 'PBKDF2', false, ['deriveBits']);
  const bits = await subtle.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt: envelope.bytes.salt,
    iterations: envelope.kdf.iterations }, material, 256);
  return new Uint8Array(bits);
}

export async function decryptEnvelope(envelope, keyBits) {
  const subtle = subtleCrypto();
  const key = await subtle.importKey('raw', keyBits, { name: 'AES-GCM' }, false, ['decrypt']);
  let plain;
  try {
    plain = await subtle.decrypt({ name: 'AES-GCM', iv: envelope.bytes.iv, additionalData: envelopeAad(envelope),
      tagLength: 128 }, key, envelope.bytes.data);
  } catch {
    throw new KeyMismatchError();
  }
  const stream = new Blob([plain]).stream().pipeThrough(new DecompressionStream('gzip'));
  const text = await new Response(stream).text();
  if (text.length > MAX_BYTES * 4) throw new Error('集計が大きすぎます。');
  return text;
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

/** 暗号文を取る。まだ無ければ null。 */
export async function loadEnvelope(signal) {
  const response = await fetch(profileUrl(), { signal, cache: 'no-store', credentials: 'same-origin' });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error('集計のファイルを取得できませんでした。時間を置いて再読み込みしてください。');
  if (Number(response.headers.get('content-length')) > MAX_BYTES) throw new Error('集計のファイルが大きすぎます。');
  const text = await response.text();
  if (text.length > MAX_BYTES) throw new Error('集計のファイルが大きすぎます。');
  return parseEnvelope(text);
}

function storages() {
  const out = [];
  for (const name of ['sessionStorage', 'localStorage']) {
    try { if (globalThis[name]) out.push(globalThis[name]); }
    catch { /* 保存が使えなくても画面は動く */ }
  }
  return out;
}

/** 覚えた復号の鍵（同じ塩・回数で作ったものだけ）。 */
export function savedUnlock(envelope) {
  for (const store of storages()) {
    try {
      const saved = JSON.parse(store.getItem(UNLOCK_KEY) || 'null');
      if (saved?.salt === envelope.kdf.salt && saved.iterations === envelope.kdf.iterations) {
        const bits = fromBase64(saved.key);
        if (bits.length === 32) return bits;
      }
    } catch { /* 壊れた保存は無視する */ }
  }
  return null;
}

/** 復号の鍵を覚える。既定はこのタブを閉じるまで（sessionStorage）、remember なら端末に（localStorage）。 */
export function saveUnlock(envelope, bits, remember) {
  const value = JSON.stringify({ salt: envelope.kdf.salt, iterations: envelope.kdf.iterations, key: toBase64(bits) });
  clearUnlock();
  for (const store of storages()) {
    const isLocal = store === globalThis.localStorage;
    if (isLocal && !remember) continue;
    try { store.setItem(UNLOCK_KEY, value); }
    catch { /* 保存できなくても開いた集計は表示する */ }
  }
}

export function clearUnlock() {
  for (const store of storages()) {
    try { store.removeItem(UNLOCK_KEY); }
    catch { /* 何もしない */ }
  }
}
