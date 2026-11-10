/** Real browser + two local HTTP origins; external API is mocked, never billed. */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'growthstock-ai-'));
const PASSWORD = 'TEST_ONLY_BROWSER_PASSWORD_NOT_A_REAL_SECRET';
const BUILD_SECRET = 'TEST_ONLY_PROVIDER_SECRET_MUST_NOT_ENTER_BUILD';
const answer = '## 財務の特徴\n\n基準日は2026-09-11です。\n\n- ROEは公開データに基づきます。\n- スコアと実際の確率は区別します。\n\n<script>window.__aiXss=true</script>\n![画像](https://attacker.example/pixel)\n[危険リンク](javascript:alert(1))';
const calls = []; let mode = 'success';
const listen = server => new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const close = server => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });
const backend = http.createServer(async (req, res) => {
  const origin = req.headers.origin;
  res.setHeader('Access-Control-Allow-Origin', origin || 'http://127.0.0.1');
  res.setHeader('Access-Control-Allow-Methods', 'POST');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');
  res.setHeader('Content-Type', 'application/json');
  if (req.method === 'OPTIONS') { res.writeHead(204).end(); return; }
  let body = ''; for await (const chunk of req) body += chunk;
  calls.push({ authorization: req.headers.authorization, body: JSON.parse(body) });
  await new Promise(resolve => setTimeout(resolve, 150));
  if (req.headers.authorization !== `Bearer ${PASSWORD}`) { res.writeHead(401).end(JSON.stringify({ error: { code: 'UNAUTHORIZED' } })); return; }
  if (mode === 'limit') { res.writeHead(429).end(JSON.stringify({ error: { code: 'UPSTREAM_LIMIT' } })); return; }
  if (mode === 'failure') { res.writeHead(502).end(JSON.stringify({ error: { code: 'UPSTREAM_UNAVAILABLE' } })); return; }
  if (mode === 'timeout') { res.writeHead(504).end(JSON.stringify({ error: { code: 'TIMEOUT' } })); return; }
  res.end(JSON.stringify({ answer, model: 'plamo-3.0-fin-beta', usage: { callsToday: calls.length, dailyLimit: 20, callsThisMonth: calls.length, monthlyLimit: 300 } }));
});
let browser, server;
try {
  await listen(backend);
  const endpoint = `http://127.0.0.1:${backend.address().port}`;
  const dist = path.join(temp, 'dist');
  let blocked = false;
  try {
    execFileSync(process.execPath, [path.join(ROOT, 'node_modules/vite/bin/vite.js'), 'build', '--outDir', path.join(temp, 'unsafe')], {
      cwd: ROOT, env: { ...process.env, VITE_PLAMO_API_KEY: BUILD_SECRET }, stdio: 'pipe',
    });
  } catch (e) {
    const output = String(e.stderr); blocked = /秘密情報をVITE_変数/.test(output);
    assert.ok(!output.includes(BUILD_SECRET), 'Build errors must not print secret values');
  }
  assert.ok(blocked, 'A mistakenly public API-key variable must fail the build');
  let unsafeUrlBlocked = false;
  try {
    execFileSync(process.execPath, [path.join(ROOT, 'node_modules/vite/bin/vite.js'), 'build', '--outDir', path.join(temp, 'unsafe-url')], {
      cwd: ROOT, env: { ...process.env, VITE_PLAMO_WORKER_URL: `https://user:${BUILD_SECRET}@private-ai.workers.dev/` }, stdio: 'pipe',
    });
  } catch (e) {
    const output = String(e.stderr); unsafeUrlBlocked = /認証情報・クエリを含まないHTTPS/.test(output);
    assert.ok(!output.includes(BUILD_SECRET), 'Invalid public URLs must not print credential values');
  }
  assert.ok(unsafeUrlBlocked, 'Credentials embedded in a public URL must fail the build');
  execFileSync(process.execPath, [path.join(ROOT, 'node_modules/vite/bin/vite.js'), 'build', '--outDir', dist], {
    cwd: ROOT, env: { ...process.env, VITE_PLAMO_WORKER_URL: endpoint, PLAMO_API_KEY: BUILD_SECRET, AI_PASSWORD: PASSWORD }, stdio: 'pipe',
  });
  const assets = fs.readdirSync(path.join(dist, 'assets')).filter(n => n.endsWith('.js'));
  for (const name of assets) {
    const text = fs.readFileSync(path.join(dist, 'assets', name), 'utf8');
    assert.ok(!text.includes(BUILD_SECRET) && !text.includes(PASSWORD), 'Secrets must not enter the JS bundle');
  }
  const fixtures = { 'stocks.json': 'synthetic-stocks.json', 'predictions.json': 'synthetic-predictions.json',
    'valuations.json': 'synthetic-valuations.json', 'new_samples.json': 'synthetic-new-samples.json', 'news_links.json': 'synthetic-news-links.json' };
  const mime = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css' };
  const notFound = [];
  server = http.createServer((req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    const off = pathname.startsWith('/without-ai/growthstock/');
    const prefix = off ? '/without-ai/growthstock/' : '/growthstock/';
    if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
    let relative = pathname.slice(prefix.length) || 'index.html';
    const fixture = relative.startsWith('data/') ? fixtures[relative.slice(5)] : null;
    const file = fixture ? path.join(ROOT, 'tests/fixtures', fixture) : path.join(off ? path.join(ROOT, 'dist') : dist, relative);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { notFound.push(relative); res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await listen(server);
  const base = `http://127.0.0.1:${server.address().port}/growthstock/`;
  const executable = process.env.PLAYWRIGHT_CHROMIUM_PATH || (fs.existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
  browser = await chromium.launch({ ...(executable ? { executablePath: executable } : {}), args: ['--no-sandbox'] });
  const results = [];
  for (const width of [1440, 390]) {
    mode = 'success'; const context = await browser.newContext({ viewport: { width, height: 1000 }, permissions: ['clipboard-read', 'clipboard-write'] });
    const page = await context.newPage(); const errors = [], external = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (r.url().includes('attacker.example')) external.push(r.url()); });
    await page.goto(base); await page.getByRole('tab', { name: 'AI銘柄分析', exact: true }).click();
    await page.locator('#ai-stock option').first().waitFor({ state: 'attached' });
    await page.waitForFunction(() => document.querySelector('#ai-stock')?.value === '0001');
    assert.equal(await page.locator('#ai-stock').inputValue(), '0001', 'Current global public selection initializes the AI view');
    await page.locator('#ai-stock').selectOption('9324');
    await page.getByRole('button', { name: '予測モデルで高順位になった理由を説明してください', exact: true }).click();
    await page.locator('#ai-password').fill(PASSWORD);
    const before = calls.length;
    await page.locator('.ai-form').evaluate(form => { form.requestSubmit(); form.requestSubmit(); });
    await page.getByRole('button', { name: '分析しています…', exact: true }).waitFor();
    await page.locator('.ai-markdown h2').waitFor();
    assert.equal(calls.length, before + 1, 'Double submission makes one request');
    const sent = calls.at(-1).body;
    assert.equal(sent.context.stockCode, '9324');
    assert.equal(sent.context.metrics.find(m => m.key === 'price').value, 2655);
    assert.equal(sent.context.prediction.baseline.score, .514135);
    assert.equal(sent.context.explanation.model, 'lgbm');
    assert.equal(JSON.stringify(sent).includes(PASSWORD), false);
    assert.equal(await page.locator('.ai-markdown ul li').count(), 2);
    assert.equal(await page.evaluate(() => window.__aiXss), undefined);
    assert.equal(await page.locator('.ai-markdown script,.ai-markdown img').count(), 0);
    assert.equal(await page.locator('.ai-markdown a[href^="javascript:"]').count(), 0);
    assert.equal(external.length, 0);
    await page.getByRole('button', { name: '回答をコピー', exact: true }).click();
    await page.getByText('コピーしました', { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), answer);
    await page.locator('.ai-context summary').click();
    assert.equal(await page.locator('.ai-context pre').isVisible(), true);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2), false);
    const shots = process.env.AI_ARTIFACT_DIR || path.join(ROOT, 'docs'); fs.mkdirSync(shots, { recursive: true });
    await page.screenshot({ path: path.join(shots, `screenshot-ai-smoke-${width}.png`), fullPage: true });
    for (const [next, text] of [['limit', 'PLaMo APIの利用上限'], ['failure', 'AIサービスと通信'], ['timeout', 'タイムアウト']]) {
      mode = next; await page.getByRole('button', { name: 'PLaMoに質問する', exact: true }).click();
      await page.locator('[role="alert"]').filter({ hasText: text }).waitFor();
      assert.equal(await page.getByRole('button', { name: 'PLaMoに質問する', exact: true }).isEnabled(), true);
    }
    await page.locator('#ai-password').fill('WRONG_PASSWORD_WITH_32_OR_MORE_CHARACTERS');
    await page.getByRole('button', { name: 'PLaMoに質問する', exact: true }).click();
    await page.getByRole('alert').filter({ hasText: 'パスワード' }).waitFor();
    assert.equal(await page.evaluate(() => Object.values(localStorage).some(v => v.includes('TEST_ONLY_BROWSER_PASSWORD'))), false);
    await page.getByRole('tab', { name: 'ブレイク予測', exact: true }).click();
    await page.getByRole('tab', { name: 'AI銘柄分析', exact: true }).click();
    await page.locator('#ai-password').waitFor(); assert.equal(await page.locator('#ai-password').inputValue(), '');
    assert.deepEqual(errors, []);
    results.push({ width, passed: true, relativeAssetPaths: true, doubleSubmissionBlocked: true, markdownSafe: true });
    await context.close();
  }
  assert.equal(notFound.some(p => p.startsWith('assets/') || ['data/stocks.json', 'data/predictions.json'].includes(p)), false);
  const disabledPage = await browser.newPage();
  await disabledPage.goto(`http://127.0.0.1:${server.address().port}/without-ai/growthstock/`);
  await disabledPage.getByRole('tab', { name: 'AI銘柄分析', exact: true }).click();
  await disabledPage.getByText('AI接続先が未設定です', { exact: true }).waitFor();
  assert.equal(await disabledPage.getByRole('button', { name: 'PLaMoに質問する', exact: true }).isDisabled(), true);
  await disabledPage.getByRole('tab', { name: 'ブレイク予測', exact: true }).click();
  assert.equal(await disabledPage.getByRole('tab', { name: 'ブレイク予測', exact: true }).getAttribute('aria-selected'), 'true');
  await disabledPage.close();
  console.log(JSON.stringify({ allPassed: true, actualProviderCalled: false, publicSecretVariableBlocked: blocked, unsafePublicUrlBlocked: unsafeUrlBlocked,
    unconfiguredSiteUsable: true, checks: results }, null, 2));
} finally {
  await browser?.close(); if (server) await close(server); await close(backend);
  fs.rmSync(temp, { recursive: true, force: true });
}
