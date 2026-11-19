/**
 * 「価格帯別出来高」タブのブラウザ・スモークテスト。
 *
 * 置き場と同じ形の合成データ（tests/fixtures/volume_profile_inputs.py）から research/build_volume_profile.py で
 * 暗号化した集計を作り、ビルド済みの画面に data/volume-profile.enc.json として配る。J-Quants のキー（テスト用の値）を
 * 入れてブラウザの中で開けること、違うキー・集計待ち・欠損・古い集計の表示、キーがどこにも送られないことを確かめる。
 *
 *   npm run build && node tests/volume-profile-browser.mjs
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = fileURLToPath(new URL('../', import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'growthstock-volume-profile-'));
const shots = process.env.SMOKE_ARTIFACT_DIR || path.join(root, 'docs');
const python = process.env.PYTHON || 'python3';
const KEY = 'test-jquants-key-not-real-0123456789';
const UNLOCK = 'growthstock.volume-profile.unlock.v1';
const at = (name) => path.join(tmp, name);
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const rel = new URL(req.url, 'http://localhost').pathname.replace(/^\/growthstock\//, '');
  const file = path.join(root, 'dist', rel || 'index.html');
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return res.writeHead(404).end();
  res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});

let browser;
try {
  execFileSync(python, [path.join(root, 'tests/fixtures/volume_profile_inputs.py'), tmp, at('predictions.json')]);
  const build = (out, extra = []) => execFileSync(python, [path.join(root, 'research/build_volume_profile.py'),
    '--predictions', at('predictions.json'), '--work-dir', tmp, '--calendar', at('calendar.csv'),
    '--encrypt-to', out, ...extra], { env: { ...process.env, JQUANTS_API: KEY, JQUANTS_API_KEY: '' } });
  build(at('current.enc.json'));
  build(at('stale.enc.json'), ['--as-of', '2025-11-17']);
  const envelopes = { current: fs.readFileSync(at('current.enc.json'), 'utf8'), stale: fs.readFileSync(at('stale.enc.json'), 'utf8') };
  assert.doesNotMatch(envelopes.current, /123A0|サンプル|bands|vwap/, '配る集計は暗号文だけ');

  const fixture = JSON.parse(fs.readFileSync(at('predictions.json'), 'utf8'));
  const published = JSON.parse(fs.readFileSync(path.join(root, 'dist/data/predictions.json'), 'utf8'));
  const template = published.candidates[0];
  const candidates = fixture.candidates.map((row) => ({ ...template, ...row, nInDay: 3, agree90: 0, agreeK90: 0 }));
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}/growthstock/`;
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/usr/bin/chromium';
  browser = await chromium.launch(fs.existsSync(executablePath) ? { executablePath } : {});
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 1050 } })).newPage();
  const errors = [], requests = [];
  let serve = 'current';
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (req) => requests.push({ method: req.method(), url: req.url(), body: req.postData() || '' }));
  await page.route('**/data/predictions.json*', (route) => route.fulfill({ json: { ...published, asOf: fixture.asOf,
    dates: fixture.dates, candidates } }));
  await page.route('**/data/volume-profile.enc.json*', (route) => (serve === 'missing'
    ? route.fulfill({ status: 404, body: 'not found' })
    : route.fulfill({ contentType: 'application/json', body: envelopes[serve] })));

  await page.goto(base, { waitUntil: 'networkidle' });
  const tab = page.getByRole('tab', { name: '価格帯別出来高', exact: true });
  await tab.click();
  const panel = page.locator('.volume-view');
  const list = page.getByRole('table', { name: '直近5営業日の予測対象銘柄' });
  await list.waitFor();
  // 直近5つの予測日の候補（123A は2日ぶんを1行に）。それより前の日の 7777 は出ない
  assert.equal(await list.locator('tbody tr').count(), 3);
  assert.match(await panel.innerText(), /候補 3銘柄/);
  assert.doesNotMatch(await list.innerText(), /7777/);
  assert.match(await list.locator('tbody tr').first().innerText(), /123A サンプル甲[\s\S]*11\/18（1位）・11\/14（2位）/);
  // 銘柄はプルダウンで選ぶ（コード・名前を打つ検索欄は無い）。キーを入れる前から一覧と同じ並びで選べる
  const stockSelect = panel.getByLabel('銘柄', { exact: true });
  assert.deepEqual(await stockSelect.locator('option').evaluateAll((options) => options.map((o) => o.value)),
    ['123A0', '99990', '88990']);
  assert.match(await stockSelect.locator('option').first().innerText(), /123A サンプル甲（11\/18（1位）・11\/14（2位））/);
  assert.equal(await panel.getByLabel('銘柄検索').count(), 0);
  assert.equal(await panel.getByPlaceholder('銘柄名・コード').count(), 0);
  const keyInput = page.getByLabel('J-Quants の API キー');
  await keyInput.waitFor();
  assert.equal(await keyInput.getAttribute('type'), 'password');
  assert.ok(await panel.getByLabel('この端末に記憶する').isChecked(), '既定で端末に覚える（最初の1回だけ入力）');

  await keyInput.fill(`${KEY}-wrong`);
  await panel.getByRole('button', { name: '開く', exact: true }).click();
  await panel.getByRole('alert').waitFor();
  assert.match(await panel.getByRole('alert').innerText(), /キーが一致しません/);

  await keyInput.fill(`  ${KEY}  `);
  await panel.getByRole('button', { name: '開く', exact: true }).click();
  const volumeBar = panel.locator('.recharts-bar-rectangle');
  await volumeBar.first().waitFor();
  const time = page.getByLabel('集計期間');
  const source = page.getByLabel('データ種別');
  assert.equal(await panel.getByRole('button', { name: '123A サンプル甲' }).getAttribute('aria-pressed'), 'true');
  assert.equal(await time.inputValue(), '20');
  assert.equal(await source.inputValue(), 'tick');
  let text = await panel.innerText();
  assert.match(text, /日足照合 20\/20営業日/);
  assert.match(text, /終値とVWAPの差\s*[+-]\d+\.\d{2}%/);
  assert.match(text, /時間帯別の出来高/);
  for (const label of ['寄り', '前場', '後場', '引け', '大口（1,000万円以上）']) assert.match(text, new RegExp(label));
  assert.match(text, /寄り・引けは板寄せの約定/);
  assert.equal(await panel.locator('.vp-share-bar').count(), 2);
  assert.match(await list.locator('tbody tr').first().innerText(), /[+-]\d+\.\d{2}%\s+\d+\.\d%\s+表示可\s+表示可/);
  fs.mkdirSync(shots, { recursive: true });
  await page.screenshot({ path: path.join(shots, 'screenshot-volume-profile-desktop.png') });

  await source.selectOption('minute');
  text = await panel.innerText();
  assert.match(text, /分足では、寄りは最初の1分/);
  assert.match(text, /各分の出来高/);
  assert.match(text, /大口（1,000万円以上）/, '大きさ別はティックで出す');
  await time.selectOption('5');
  assert.match(await panel.innerText(), /日足照合 5\/5営業日/);
  await page.getByLabel('横軸').selectOption('turnover');
  await page.locator('.volume-table summary').click();
  assert.match(await panel.innerText(), /売買代金（円）/);

  await stockSelect.selectOption('99990');
  assert.equal(await panel.getByRole('button', { name: '9999 サンプル乙' }).getAttribute('aria-pressed'), 'true');
  assert.match(await panel.locator('.volume-stock-heading').innerText(), /9999 サンプル乙/);
  assert.equal(await source.inputValue(), 'minute', 'ティックが無ければ分足を選ぶ');
  assert.match(await panel.innerText(), /この期間のティックが揃っていません/);
  await source.selectOption('tick');
  text = await panel.innerText();
  assert.match(text, /置き場にまだ無いか、足りません/);
  assert.match(text, /照合済み 0\/5営業日/);
  // 表の行を押しても選べ、プルダウンも合わせて変わる
  await page.getByRole('button', { name: '8899 集計できない銘柄' }).click();
  assert.equal(await stockSelect.inputValue(), '88990');
  assert.match(await panel.innerText(), /日足が揃っていません/);
  assert.equal(await list.locator('tbody tr').count(), 3);

  // 既定で端末に覚えるので、新しいタブで開き直しても入力なしで開く（キーそのものは保存しない）
  const stored = await page.evaluate(() => JSON.stringify({ ...sessionStorage, ...localStorage }));
  assert.ok(!stored.includes(KEY), 'キーそのものは保存しない');
  assert.notEqual(await page.evaluate((k) => localStorage.getItem(k), UNLOCK), null, '既定でこの端末に覚える');
  const later = await page.context().newPage();
  await later.route('**/data/predictions.json*', (route) => route.fulfill({ json: { ...published, asOf: fixture.asOf,
    dates: fixture.dates, candidates } }));
  await later.route('**/data/volume-profile.enc.json*', (route) => route.fulfill({ contentType: 'application/json',
    body: envelopes.current }));
  later.on('request', (req) => requests.push({ method: req.method(), url: req.url(), body: req.postData() || '' }));
  await later.goto(base, { waitUntil: 'networkidle' });
  await later.getByRole('tab', { name: '価格帯別出来高', exact: true }).click();
  await later.locator('.volume-view .recharts-bar-rectangle').first().waitFor();
  assert.equal(await later.getByLabel('J-Quants の API キー').count(), 0, '2回目からは入力なし');
  await later.close();
  await page.reload({ waitUntil: 'networkidle' });
  await tab.click();
  await volumeBar.first().waitFor();
  assert.equal(await keyInput.count(), 0);

  // 古い集計（予測の基準日より前）
  serve = 'stale';
  await panel.getByRole('button', { name: '再読み込み' }).click();
  await panel.getByText(/開いた集計の基準日は 2025-11-17/).waitFor();

  // 鍵を消すとキーの入力に戻る
  await panel.getByRole('button', { name: '鍵を消す' }).click();
  await keyInput.waitFor();
  assert.deepEqual(await page.evaluate((k) => [sessionStorage.getItem(k), localStorage.getItem(k)], UNLOCK), [null, null]);

  // まだ集計が無い
  serve = 'missing';
  await page.reload({ waitUntil: 'networkidle' });
  await tab.click();
  await panel.getByText(/まだ集計がありません/).waitFor();
  assert.equal(await keyInput.count(), 0);

  // スマホの幅
  serve = 'current';
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload({ waitUntil: 'networkidle' });
  await tab.click();
  await keyInput.fill(KEY);
  await panel.getByLabel('この端末に記憶する').uncheck();
  await panel.getByRole('button', { name: '開く', exact: true }).click();
  await volumeBar.first().waitFor();
  assert.equal(await page.evaluate((k) => localStorage.getItem(k), UNLOCK), null, '外したときは端末に覚えない');
  assert.notEqual(await page.evaluate((k) => sessionStorage.getItem(k), UNLOCK), null, 'タブを閉じるまでは覚える');
  await page.screenshot({ path: path.join(shots, 'screenshot-volume-profile-mobile.png'), fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'スマホの幅で横にはみ出さない');

  assert.deepEqual(errors, []);
  assert.ok(requests.every((req) => !req.url.includes(KEY) && !req.body.includes(KEY)), 'キーはどこにも送らない');
  assert.ok(requests.every((req) => req.url.startsWith(base) || !/volume-profile|predictions/.test(req.url)));
  assert.ok(requests.filter((req) => /volume-profile/.test(req.url)).every((req) => req.method === 'GET'));
  console.log('PASS: 直近5営業日の候補・プルダウンで選ぶ・ブラウザの中での復号・違うキー・欠損・古い集計・集計待ち・2回目から入力なし・スマホの幅');
} finally {
  await browser?.close();
  if (server.listening) await new Promise((resolve) => server.close(resolve));
  fs.rmSync(tmp, { recursive: true, force: true });
}
