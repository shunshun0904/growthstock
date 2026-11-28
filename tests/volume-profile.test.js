/**
 * 「価格帯別出来高」タブのデータ（src/lib/volumeProfile.js）の単体テスト。
 *   npm test
 *
 * tests/fixtures/synthetic-volume-profile.json は、合成データ（tests/fixtures/volume_profile_inputs.py。実データではない）を
 * research/build_volume_profile.py に通して作った集計。Python が書く集計を画面がそのまま読めることを守る。
 *
 * 作り直し方:
 *   python3 tests/fixtures/volume_profile_inputs.py /tmp/vp /tmp/vp/predictions.json
 *   python3 research/build_volume_profile.py --predictions /tmp/vp/predictions.json --work-dir /tmp/vp \
 *     --calendar /tmp/vp/calendar.csv --output tests/fixtures/synthetic-volume-profile.json
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { forgetOldUnlock, parseVolumeProfile, volumeTargets } from '../src/lib/volumeProfile.js';

const text = fs.readFileSync(new URL('./fixtures/synthetic-volume-profile.json', import.meta.url), 'utf8');
const predictions = JSON.parse(fs.readFileSync(new URL('./fixtures/synthetic-volume-profile-predictions.json', import.meta.url), 'utf8'));

test('Python が書いた集計をそのまま読める', () => {
  const data = parseVolumeProfile(text);
  assert.equal(data.asOf, '2025-11-18');
  assert.deepEqual(data.stocks.map((s) => s.code), ['123A0', '88990', '99990']);
  const a = data.stocks[0].windows[20];
  assert.equal(a.tick.status, 'ok');
  assert.equal(a.minute.status, 'ok');
  assert.ok(Math.abs(Object.values(a.tick.sessions).reduce((t, v) => t + v, 0) - 1) < 1e-9);
  assert.equal(a.tick.tradeSizes.counts.length, 3);
  assert.equal(a.minute.tradeSizes, undefined);
  assert.equal(data.stocks[2].windows[5].tick.status, 'intraday_missing');
});

test('集計の中身の形を確かめる', () => {
  const data = JSON.parse(text);
  const broken = (change) => {
    const copy = structuredClone(data);
    change(copy);
    return JSON.stringify(copy);
  };
  assert.throws(() => parseVolumeProfile('{'), /読み込めませんでした/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.demo = true; })), /合成データ/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.schemaVersion = 1; })), /形式が一致しません/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.stocks[0].windows[5].tick.sessions.open += 0.1; })), /形式/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.stocks[0].windows[5].minute.tradeSizes = d.stocks[0].windows[5].tick.tradeSizes; })), /形式/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.stocks[0].windows[5].tick.closeVsVwap += 0.01; })), /形式/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.stocks[0].windows[1].tick.status = 'not_available_at_cutoff'; })), /形式/);
  assert.throws(() => parseVolumeProfile(broken((d) => { d.stocks[1].code = d.stocks[0].code; })), /形式/);
});

test('対象は直近5つの予測日の候補すべて（銘柄の重複を除き、新しい候補日・順位の順）', () => {
  const { asOf, dates, targets } = volumeTargets(predictions);
  assert.equal(asOf, '2025-11-18');
  assert.deepEqual(dates, ['2025-11-12', '2025-11-13', '2025-11-14', '2025-11-17', '2025-11-18']);
  assert.deepEqual(targets.map((t) => t.code), ['123A0', '99990', '88990']);
  assert.deepEqual(targets[0].candidates, [{ date: '2025-11-18', rank: 1 }, { date: '2025-11-14', rank: 2 }]);
  assert.ok(!targets.some((t) => t.code === '77770'), '5つより前の予測日の候補は入れない');
  const noDates = volumeTargets({ asOf: '2025-11-18', candidates: predictions.candidates });
  assert.deepEqual(noDates.dates, ['2025-11-11', '2025-11-12', '2025-11-14', '2025-11-17', '2025-11-18']);
  assert.deepEqual(volumeTargets(null), { asOf: null, dates: [], targets: [] });
});

test('暗号化していたころに端末へ覚えた復号の鍵を消す', () => {
  const store = () => {
    const items = new Map([['growthstock.volume-profile.unlock.v1', 'x'], ['other', 'y']]);
    return { items, removeItem: (k) => items.delete(k) };
  };
  const saved = { sessionStorage: globalThis.sessionStorage, localStorage: globalThis.localStorage };
  globalThis.sessionStorage = store();
  globalThis.localStorage = store();
  try {
    forgetOldUnlock();
    for (const name of ['sessionStorage', 'localStorage']) {
      assert.deepEqual([...globalThis[name].items.keys()], ['other']);
    }
  } finally {
    globalThis.sessionStorage = saved.sessionStorage;
    globalThis.localStorage = saved.localStorage;
  }
});
