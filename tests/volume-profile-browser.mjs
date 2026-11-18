/** Exercise the live prediction universe and authenticated delivery on Pages. */
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
  data.demo = false;
  const published = JSON.parse(fs.readFileSync(path.join(root, 'dist/data/predictions.json'), 'utf8'));
  const template = published.candidates[0];
  const candidates = [
    ['123A', '123A0', 'サンプル甲', data.asOf],
    ['9999', '99990', 'サンプル乙', data.asOf],
    ['8899', '88990', '集計待ち銘柄', data.asOf],
    ['8888', '88880', '前日の銘柄', '2025-11-17'],
  ].map(([code, jqCode, name, date], i) => ({ ...template, code, jqCode, name, date, rankInDay: i + 1,
    nInDay: 3, agree90: 0, agreeK90: 0 }));
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}/growthstock/`;
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/usr/bin/chromium';
  browser = await chromium.launch(fs.existsSync(executablePath) ? { executablePath } : {});
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [], requests = [], profileRequests = [];
  const endpoint = 'https://api.github.com/repos/shunshun0904/growthstock-minute/contents/dashboard/volume-profile.json';
  const token = 'github_pat_mock_viewer_token_123456';
  let stale = false;
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (req) => requests.push({ method: req.method(), url: req.url() }));
  await page.route('**/data/predictions.json*', (route) => route.fulfill({ json: { ...published, asOf: data.asOf,
    dates: ['2025-11-17', data.asOf], candidates } }));
  await page.route(endpoint, async (route) => {
    const headers = { 'access-control-allow-origin': '*', 'access-control-allow-headers': 'authorization,accept' };
    if (route.request().method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
    profileRequests.push(route.request().method());
    const authenticated = (await route.request().headerValue('authorization')) === `Bearer ${token}`;
    if (!authenticated) return route.fulfill({ status: 401, headers, json: {} });
    assert.equal(await route.request().headerValue('accept'), 'application/vnd.github.raw+json');
    return route.fulfill({ headers, json: stale ? { ...data, asOf: '2025-11-17', stocks: [] } : data });
  });
  await page.goto(base, { waitUntil: 'networkidle' });
  const tab = page.getByRole('tab', { name: '価格帯別出来高', exact: true });
  await tab.click();
  assert.equal(await page.getByRole('tab', { name: '価格帯別需給', exact: true }).count(), 0);
  const panel = page.locator('.volume-view');
  const list = page.getByRole('table', { name: '当日の予測対象銘柄' });
  await list.waitFor();
  assert.equal(await list.locator('tbody tr').count(), 3);
  assert.match(await panel.innerText(), /対象 3銘柄/);
  assert.doesNotMatch(await list.innerText(), /8888|前日の銘柄/);
  assert.equal(await panel.locator('input[type=file]').count(), 0);
  assert.deepEqual(profileRequests, []);

  await page.getByLabel('閲覧用GitHubトークン').fill('github_pat_mock_invalid_token_123456');
  await panel.getByRole('button', { name: '接続', exact: true }).click();
  await panel.getByRole('alert').waitFor();
  assert.match(await panel.getByRole('alert').innerText(), /無効/);
  await panel.getByRole('button', { name: '接続を解除' }).click();
  await page.getByLabel('閲覧用GitHubトークン').fill(token);
  await panel.getByRole('button', { name: '接続', exact: true }).click();
  const time = page.getByLabel('集計期間');
  const source = page.getByLabel('データ種別');
  const volumeBar = panel.locator('.recharts-bar-rectangle');
  await volumeBar.first().waitFor();
  assert.equal(await panel.getByRole('button', { name: '123A サンプル甲' }).getAttribute('aria-pressed'), 'true');
  assert.equal(await time.inputValue(), '20');
  assert.equal(await source.inputValue(), 'tick');
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
  await page.getByRole('button', { name: '9999 サンプル乙' }).click();
  await source.selectOption('tick');
  assert.match(await panel.innerText(), /照合済み 0\/5営業日/);
  await page.getByRole('button', { name: '8899 集計待ち銘柄' }).click();
  assert.match(await panel.innerText(), /この銘柄の出来高は集計待ち/);
  await page.getByLabel('銘柄検索').fill('123A');
  assert.equal(await list.locator('tbody tr').count(), 1);
  await page.getByLabel('銘柄検索').fill('');

  stale = true;
  await panel.getByRole('button', { name: '再取得' }).click();
  await panel.getByText(/保存済みの基準日は2025-11-17/).waitFor();
  assert.equal(await list.locator('tbody tr').count(), 3);
  assert.equal(await volumeBar.count(), 0);
  stale = false;
  await page.getByRole('tab', { name: 'ブレイク予測', exact: true }).click();
  await page.waitForLoadState('networkidle');
  await tab.click();
  await page.setViewportSize({ width: 390, height: 844 });
  await volumeBar.first().waitFor();
  assert.equal(await page.getByLabel('閲覧用GitHubトークン').count(), 0);
  await page.screenshot({ path: path.join(shots, 'screenshot-volume-profile-mobile.png'), fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile has no page overflow');
  assert.deepEqual(errors, []);
  assert.deepEqual(profileRequests, ['GET', 'GET', 'GET', 'GET']);
  assert.ok(requests.every((req) => !req.url.includes(token)), 'Tokens stay out of URLs');
  assert.ok(requests.filter((req) => /\/(assets|data)\//.test(req.url)).every((req) => req.url.startsWith(base)));
  console.log('PASS: all current predictions, automatic private read, auth/stale/missing states, desktop and mobile');
} finally {
  await browser?.close();
  if (server.listening) await new Promise((resolve) => server.close(resolve));
  fs.rmSync(tmp, { recursive: true, force: true });
}
