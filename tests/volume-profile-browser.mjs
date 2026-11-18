/** Build the fixture with the real collector, then exercise the Pages UI. */
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
  execFileSync(python, [path.join(root, 'tests/fixtures/volume_profile_inputs.py'), tmp]);
  execFileSync(python, [path.join(root, 'research/build_volume_profile.py'), '--demo', '--as-of', '2025-11-18',
    '--codes', '123A', '9999', '--calendar', at('calendar.csv'), '--daily', at('daily.csv'),
    '--minute', at('minute.csv.gz'), '--ticks', at('tick.csv.gz'), '--cache-dir', at('cache'), '--output', at('output')]);
  const data = JSON.parse(fs.readFileSync(at('output/volume-profile.json'), 'utf8'));
  assert.equal(data.stocks.length, 2);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}/growthstock/`;
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/usr/bin/chromium';
  browser = await chromium.launch(fs.existsSync(executablePath) ? { executablePath } : {});
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [], requests = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (req) => requests.push({ method: req.method(), url: req.url() }));
  await page.goto(base, { waitUntil: 'networkidle' });
  const tab = page.getByRole('tab', { name: '価格帯別出来高', exact: true });
  await tab.click();
  assert.equal(await page.getByRole('tab', { name: '価格帯別需給', exact: true }).count(), 0);
  const input = page.getByLabel('集計ファイルを選択');
  await input.waitFor();
  const beforeImport = requests.length;
  await input.setInputFiles(at('output/volume-profile.json'));
  const stock = page.getByLabel('分析対象銘柄');
  const time = page.getByLabel('集計期間');
  const source = page.getByLabel('データ種別');
  const panel = page.locator('.volume-view');
  const volumeBar = panel.locator('.recharts-bar-rectangle');
  await volumeBar.first().waitFor();
  assert.equal(await stock.inputValue(), '123A0');
  assert.equal(await time.inputValue(), '20');
  assert.equal(await source.inputValue(), 'tick');
  assert.match(await panel.innerText(), /合成データの表示例/);
  assert.match(await panel.innerText(), /日足照合 20\/20営業日/);
  fs.mkdirSync(shots, { recursive: true });
  await page.screenshot({ path: path.join(shots, 'screenshot-volume-profile-desktop.png') });

  await source.selectOption('minute');
  assert.match(await panel.innerText(), /各分の出来高/);
  await time.selectOption('5');
  assert.match(await panel.innerText(), /日足照合 5\/5営業日/);
  await page.getByLabel('横軸').selectOption('turnover');
  await page.locator('.volume-table summary').click();
  assert.ok(await panel.locator('tbody tr').count() > 1);
  assert.match(await panel.innerText(), /売買代金（円）/);
  await page.getByLabel('銘柄検索').fill('9999');
  assert.equal(await stock.inputValue(), '99990');
  await source.selectOption('tick');
  assert.match(await panel.innerText(), /照合済み 0\/5営業日/);
  await page.getByLabel('銘柄検索').fill('');

  await input.setInputFiles({ name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('{"stocks":[]}') });
  assert.match(await page.getByRole('alert').innerText(), /集計ファイル/);
  assert.deepEqual(requests.slice(beforeImport), [], 'Selecting and exploring the local file makes no HTTP request');
  await page.getByRole('tab', { name: 'ブレイク予測', exact: true }).click();
  await page.waitForLoadState('networkidle');
  const beforeReturn = requests.length;
  await tab.click();
  assert.equal(await stock.inputValue(), '123A0');
  assert.equal(await source.inputValue(), 'tick');
  await page.setViewportSize({ width: 390, height: 844 });
  await volumeBar.first().waitFor();
  await page.screenshot({ path: path.join(shots, 'screenshot-volume-profile-mobile.png'), fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile has no page overflow');
  assert.deepEqual(errors, []);
  assert.deepEqual(requests.slice(beforeReturn), [], 'Returning to the imported profile makes no HTTP request');
  assert.ok(requests.filter((req) => /\/(assets|data)\//.test(req.url)).every((req) => req.url.startsWith(base)));
  console.log('PASS: private CSV/GZ→JSON→Pages UI, period/source/units/filter, unavailable/error, desktop and mobile');
} finally {
  await browser?.close();
  if (server.listening) await new Promise((resolve) => server.close(resolve));
  fs.rmSync(tmp, { recursive: true, force: true });
}
