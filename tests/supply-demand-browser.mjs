/** Built-site regression checks, including GitHub Pages' /growthstock/ prefix.
 * node tests/supply-demand-browser.mjs
 * While the tab is hidden, verify its absence and that no demand assets are fetched.
 * When enabled, PRICE_DEMAND_REAL=1 also captures the checked-in public dataset.
 */
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const SHOTS = process.env.SMOKE_ARTIFACT_DIR || path.join(ROOT, 'docs');
fs.mkdirSync(SHOTS, { recursive: true });
const real = process.env.PRICE_DEMAND_REAL === '1';
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/fixtures/synthetic-supply-demand.json')));
let retryRequests = 0;
const catalog = { schemaVersion: 1, stocks: { '28940': { status: 'ok' }, '88880': { status: 'ok' } } };
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const rel = pathname.replace(/^\/growthstock\//, '');
  const file = path.join(ROOT, 'dist', rel || 'index.html');
  if (!real && rel === 'data/stocks.json') return res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(fixture));
  if (!real && rel === 'data/supply-demand/index.json') return res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(catalog));
  if (!real && /^data\/supply-demand\/(28940|88880)\.json$/.test(rel)) {
    const code = rel.match(/([0-9]+)\.json$/)[1];
    if (code === '88880' && ++retryRequests === 1) return res.writeHead(503).end('temporarily unavailable');
    return setTimeout(() => res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ code, supplyDemand: fixture.stocks[0].supplyDemand })), 80);
  }
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end('not found');
  res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const chromiumPath = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/usr/bin/chromium';
const browser = await chromium.launch(fs.existsSync(chromiumPath) ? { executablePath: chromiumPath } : {});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, hasTouch: true });
const errors = [], requests = [];
page.on('pageerror', e => errors.push(e.message));
page.on('request', r => requests.push(r.url()));
async function checkSupplyDemandView() {
  if (!real) await page.addInitScript(() => localStorage.setItem('focus.manualStocks.v1', JSON.stringify([
    { id: 'saved-2894', code: '2894', name: '合成・保存済み銘柄', origin: 'prediction', metrics: {} },
    { id: 'saved-8888', code: '8888', name: '合成・再試行', origin: 'prediction', metrics: {} },
  ])));
  if (real) await page.addInitScript(() => localStorage.setItem('focus.manualStocks.v1', JSON.stringify([
    { id: 'saved-2894', code: '2894', name: '石井食品', origin: 'prediction', metrics: {} },
    { id: 'saved-6467', code: '6467', name: 'ニチダイ', origin: 'prediction', metrics: {} },
  ])));
  await page.goto(`http://127.0.0.1:${server.address().port}/growthstock/`, { waitUntil: 'networkidle' });
  // 運用者の指示 2026-11-15。非表示中もPages配下・PC/スマホでタブと画面が出ないことを確かめる。
  if (await page.getByRole('tab', { name: '価格帯別需給', exact: true }).count() === 0) {
    assert.deepEqual(await page.getByRole('tab').allTextContents(), [
      'ブレイク予測', '候補銘柄のニュース', '理論株価', 'タイムマシーン',   // 価格帯別出来高も 2026-11-21 から隠している
    ]);
    assert.equal(await page.locator('.supply-demand-view').count(), 0);
    await page.screenshot({ path: path.join(SHOTS, 'screenshot-price-demand-hidden-desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.getByRole('tab', { name: '価格帯別需給', exact: true }).count(), 0);
    assert.equal(await page.locator('.supply-demand-view').count(), 0);
    await page.screenshot({ path: path.join(SHOTS, 'screenshot-price-demand-hidden-mobile.png') });
    assert.ok(!requests.some(u => new URL(u).pathname.includes('/data/supply-demand/')), 'Hidden view does not fetch demand assets');
    assert.deepEqual(errors, []);
    assert.ok(requests.filter(u => /\/(assets|data)\//.test(u)).every(u => new URL(u).pathname.startsWith('/growthstock/')), 'All assets resolve under the Pages prefix');
    console.log('PASS: price-demand tab is hidden (desktop/mobile; no demand asset requests; Pages subpath)');
    return;
  }
  await page.getByRole('tab', { name: '価格帯別需給', exact: true }).click();
  await page.waitForSelector('.sd-positions');
  // スコアと収益の分布・再学習で加わったサンプルは 2026-11-15 から、価格帯別出来高は 2026-11-21 から隠している（src/App.jsx の hidden）
  assert.equal(await page.getByRole('tab').count(), 5);
  assert.equal(await page.getByRole('tab').nth(1).textContent(), '候補銘柄のニュース');
  assert.equal(await page.getByRole('tab').last().textContent(), 'タイムマシーン');
  if (!real) {
    await page.getByLabel('需給の分析対象銘柄').selectOption('jq-11110');
    assert.match(await page.locator('.sd-positions').innerText(), /9\.1%/);
    assert.match(await page.locator('.sd-positions').innerText(), /-33\.3%/);
    assert.match(await page.locator('.sd-positions').innerText(), /約定金額÷株数/);
    assert.equal(await page.locator('.sd-band').count(), 3);
    await page.locator('.sd-band').first().focus();
    assert.match(await page.locator('.sd-band-detail').innerText(), /株/);
    await page.getByRole('button', { name: '60営業日', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: '60営業日', exact: true }).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: '株数', exact: true }).click();
    assert.match(await page.locator('.sd-chart svg').textContent(), /万株/);
    await page.getByLabel('需給の分析対象銘柄').selectOption('jq-22220');
    assert.match(await page.locator('.sd-estimate-note').innerText(), /約定金額が未取得/);
    assert.match(await page.locator('.sd-positions').innerText(), /推定/);
    assert.match(await page.locator('.sd-positions').innerText(), /0株/);
    assert.match(await page.locator('.sd-status').innerText(), /0株/);
    await page.getByLabel('需給の分析対象銘柄').selectOption('jq-33330');
    assert.match(await page.locator('.supply-demand-view').innerText(), /まだ取得されていません/);
    await page.getByLabel('需給の銘柄検索').fill('存在しないテスト検索');
    assert.match(await page.locator('.supply-demand-view').innerText(), /一致する銘柄がありません/);
    await page.getByLabel('需給の銘柄検索').fill('');
    // A stock outside stocks.json is delivered through its own static asset.
    await page.getByLabel('需給の分析対象銘柄').selectOption('saved-2894');
    await page.waitForFunction(() => document.querySelector('.sd-positions')?.textContent.includes('9.1%'));
    assert.match(await page.locator('.supply-demand-view .card-head').allTextContents().then(x => x.join(' ')), /合成・保存済み銘柄/);
    assert.ok(requests.some(u => new URL(u).pathname === '/growthstock/data/supply-demand/28940.json'));
    // The request fails once; retry is explicit and never fabricates zero data.
    await page.getByLabel('需給の分析対象銘柄').selectOption('saved-8888');
    await page.getByRole('button', { name: '需給データを再試行' }).waitFor();
    assert.match(await page.locator('.supply-demand-view').innerText(), /503/);
    assert.equal(retryRequests, 1);
    await page.getByRole('button', { name: '需給データを再試行' }).click();
    await page.waitForFunction(() => document.querySelector('.sd-positions')?.textContent.includes('9.1%'));
    assert.equal(retryRequests, 2);
    await page.getByLabel('需給の分析対象銘柄').selectOption('jq-11110');
  } else {
    const options = await page.getByLabel('需給の分析対象銘柄').locator('option').allTextContents();
    const example = options.findIndex(text => /^6145 /.test(text));
    if (example >= 0) await page.getByLabel('需給の分析対象銘柄').selectOption({ index: example });
    assert.match(await page.locator('.card-head').allTextContents().then(x => x.join(' ')), /価格帯別需給/);
    assert.ok(await page.locator('.sd-band').count() > 0);
    for (const [id, name, code] of [['saved-2894', '石井食品', '28940'], ['saved-6467', 'ニチダイ', '64670']]) {
      await page.getByLabel('需給の分析対象銘柄').selectOption(id);
      await page.locator('.sd-positions').waitFor();
      assert.match(await page.locator('.supply-demand-view .card-head').allTextContents().then(x => x.join(' ')), new RegExp(name));
      assert.ok(await page.locator('.sd-band').count() > 0);
      assert.ok(requests.some(u => new URL(u).pathname === `/growthstock/data/supply-demand/${code}.json`));
    }
  }
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(SHOTS, `screenshot-price-demand-${real ? 'real' : 'synthetic'}-desktop.png`), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(150);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No horizontal page overflow on mobile');
  assert.ok(await page.locator('.sd-chart svg').boundingBox().then(b => b.width <= 390));
  await page.locator('.sd-band').first().tap();
  assert.match(await page.locator('.sd-band-detail').innerText(), /株/);
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(SHOTS, `screenshot-price-demand-${real ? 'real' : 'synthetic'}-mobile.png`), fullPage: true });
  assert.deepEqual(errors, []);
  assert.ok(requests.filter(u => /\/(assets|data)\//.test(u)).every(u => new URL(u).pathname.startsWith('/growthstock/')), 'All assets resolve under the Pages prefix');
  console.log(`PASS: price-demand browser checks (${real ? 'real snapshot' : 'synthetic edge cases'}; desktop/mobile; Pages subpath)`);
}
try {
  await checkSupplyDemandView();
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
