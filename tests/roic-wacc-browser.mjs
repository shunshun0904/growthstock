/* Real built React flows; synthetic financial inputs are explicitly labeled. */
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { researchFixture } from './fixtures/roic-wacc-fixture.mjs';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const SHOTS = process.env.SMOKE_ARTIFACT_DIR || path.join(ROOT, 'docs');
fs.mkdirSync(SHOTS, { recursive: true });
const realDir = process.env.ROIC_WACC_REAL_DIR;
const realCode = process.env.ROIC_WACC_REAL_CODE;
const fixture = researchFixture();
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.gz': 'application/gzip' };
const index = { schemaVersion: 1, researchOnly: true, stocks: [
  { code: '1000', name: '合成テスト銘柄', rows: 150, file: '10000.json.gz' },
  { code: '2000', name: '合成通信失敗', rows: 150, file: '20000.json.gz' },
] };
const server = http.createServer((req, res) => {
  const rel = new URL(req.url, 'http://localhost').pathname.replace(/^\/growthstock\//, '');
  if (rel.startsWith('data/roic-wacc/')) {
    if (realDir) {
      const file = path.join(realDir, path.basename(rel));
      if (!fs.existsSync(file)) return res.writeHead(404).end();
      res.writeHead(200, { 'Content-Type': mime[path.extname(file)] }); return fs.createReadStream(file).pipe(res);
    }
    if (rel.endsWith('index.json')) return res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(index));
    if (rel.endsWith('20000.json.gz')) return res.writeHead(503).end();
    return res.writeHead(200, { 'Content-Type': 'application/gzip' }).end(gzipSync(JSON.stringify(fixture)));
  }
  const fixtures = { 'data/stocks.json': 'synthetic-stocks.json', 'data/predictions.json': 'synthetic-predictions.json',
    'data/valuations.json': 'synthetic-valuations.json', 'data/news_links.json': 'synthetic-news-links.json' };
  const file = fixtures[rel] ? path.join(ROOT, 'tests/fixtures', fixtures[rel]) : path.join(ROOT, 'dist', rel || 'index.html');
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const executable = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/usr/bin/chromium';
const browser = await chromium.launch(fs.existsSync(executable) ? { executablePath: executable } : {});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
const errors = [], requests = [];
page.on('pageerror', e => errors.push(e.message));
page.on('request', r => requests.push(r.url()));
try {
  await page.goto(`http://127.0.0.1:${server.address().port}/growthstock/`, { waitUntil: 'networkidle' });
  await page.getByRole('tab', { name: '理論株価', exact: true }).click();
  await page.locator('.valuation-table tbody tr').first().waitFor();
  assert.equal(await page.getByRole('tab').count(), 4);
  assert.ok(!requests.some(u => u.includes('/data/roic-wacc/')), 'Research files are lazy-loaded');
  await page.getByRole('button', { name: 'ROIC・WACC（研究）', exact: true }).click();
  await page.getByRole('button', { name: '表示期間のCSVをダウンロード' }).waitFor();
  if (realDir && realCode) {
    await page.getByLabel('ROIC・WACCの分析対象', { exact:true }).selectOption(realCode);
    await page.waitForFunction(code=>[...document.querySelectorAll('.rw-view > .card p')]
      .some(p=>p.textContent.startsWith(`${code} `)),realCode);
  }
  await page.getByRole('heading', { name: '市場価格と理論価値の変化（20営業日・ログ）', exact: true }).waitFor();
  assert.match(await page.locator('.rw-view').innerText(), /株価・β・時価総額を計算に使いません/);
  assert.equal(await page.getByLabel('理論価値モデル', { exact: true }).inputValue(), 'eva_total_fade5');
  if (!realDir) {
    assert.match(await page.locator('.rw-factors').innerText(), /¥110/);
    await page.getByLabel('成長率シナリオ', { exact: true }).selectOption('business');
    assert.match(await page.locator('.rw-factors').innerText(), /¥200/);
    await page.getByLabel('成長率シナリオ', { exact: true }).selectOption('all');
    await page.getByLabel('理論価値モデル', { exact: true }).selectOption('steady');
    assert.match(await page.locator('.rw-factors').innerText(), /¥300/);
    await page.getByLabel('理論価値モデル', { exact: true }).selectOption('rim_book5');
    assert.match(await page.locator('.rw-factors').innerText(), /¥100/);
    assert.equal(await page.getByLabel('税率シナリオ', { exact: true }).isDisabled(),true);
    assert.equal(await page.getByLabel('成長率シナリオ', { exact: true }).isDisabled(),true);
    await page.getByLabel('理論価値モデル', { exact: true }).selectOption('eva_total_fade5');
    await page.getByLabel('PERの利益系列', { exact: true }).selectOption('ttm');
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '表示期間のCSVをダウンロード' }).click()]);
    assert.match(fs.readFileSync(await download.path(), 'utf8'), /price_independent_model/);
  } else {
    assert.match(await page.locator('.rw-view').innerText(), /現金を資本から控除せず/);
    assert.ok(!/理論株価を算出できる日：0\s*\//.test(await page.locator('.rw-view').innerText()));
    await page.getByLabel('理論価値モデル', { exact: true }).selectOption('rim_book5');
    assert.ok(!/理論株価を算出できる日：0\s*\//.test(await page.locator('.rw-view').innerText()));
  }
  await page.screenshot({ path: path.join(SHOTS, `screenshot-eva-${realDir ? 'real' : 'synthetic'}-desktop.png`), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Research UI fits mobile');
  await page.screenshot({ path: path.join(SHOTS, `screenshot-eva-${realDir ? 'real' : 'synthetic'}-mobile.png`), fullPage: true });
  if (!realDir) {
    await page.getByLabel('研究対象の銘柄検索', { exact: true }).fill('通信');
    assert.equal(await page.getByLabel('ROIC・WACCの分析対象', { exact: true }).inputValue(), '1000');
    await page.getByLabel('ROIC・WACCの分析対象', { exact: true }).selectOption('2000');
    await page.getByRole('alert').waitFor();
    assert.match(await page.getByRole('alert').innerText(), /503/);
    assert.equal(await page.locator('.rw-factors').count(), 0, 'Old-security data are cleared on failure');
  }
  await page.getByRole('button', { name: '銘柄ランキング', exact: true }).click();
  await page.locator('.valuation-table tbody tr').first().waitFor();
  assert.equal(await page.getByLabel('算出方法', { exact: true }).locator('option').count(), 2, 'Existing RIM/DDM retained');
  assert.deepEqual(errors, []);
  assert.ok(requests.filter(u => /\/(assets|data)\//.test(u)).every(u => new URL(u).pathname.startsWith('/growthstock/')));
  console.log(`PASS: EVA research flow (${realDir ? 'real derived histories' : 'synthetic scenarios/errors/CSV'}; desktop/mobile; Pages prefix; RIM/DDM preserved)`);
} finally {
  await browser.close(); await new Promise(resolve => server.close(resolve));
}
