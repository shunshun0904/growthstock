/**
 * ブラウザ・スモークテスト。
 * ビルド済み dist を静的配信し、Chromium で3つの View を実際に描画して検証する。
 * データは tests/fixtures/synthetic-stocks.json (合成データ) を使用する。
 *
 *   node tests/smoke.mjs
 */
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DIST = path.join(ROOT, 'dist');
const FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-stocks.json');
// 予測タブも合成データで描く。実データを使うと、その日の候補数や
// モデルの有無で結果が変わり、失敗したときに再現できない
const PRED_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-predictions.json');
const SHOTS = path.join(ROOT, 'docs');
fs.mkdirSync(SHOTS, { recursive: true });

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml',
};

const server = http.createServer((req, res) => {
  let rel = decodeURIComponent(req.url.split('?')[0]);
  if (rel === '/') rel = '/index.html';
  // データファイルだけは合成フィクスチャに差し替える
  const fixtures = {
    '/data/stocks.json': FIXTURE,
    '/data/predictions.json': PRED_FIXTURE,
  };
  const file = fixtures[rel] || path.join(DIST, rel);
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});

const failures = [];
const check = (cond, label) => {
  console.log(`${cond ? '  ok  ' : ' FAIL '} ${label}`);
  if (!cond) failures.push(label);
};

await new Promise((r) => server.listen(0, r));
const base = `http://127.0.0.1:${server.address().port}/`;

// 環境ごとの Chromium の場所を解決する。
// - このリポジトリの CI: playwright が ~/.cache/ms-playwright に自前で用意する
// - 一部のサンドボックス: /opt/pw-browsers/chromium にプリインストールされている
// 存在するときだけ executablePath を渡し、無ければ playwright の解決に任せる。
const PREINSTALLED = process.env.PLAYWRIGHT_CHROMIUM_PATH || '/opt/pw-browsers/chromium';
const launchOptions = fs.existsSync(PREINSTALLED) ? { executablePath: PREINSTALLED } : {};
console.log(launchOptions.executablePath
  ? `Chromium: ${PREINSTALLED} (プリインストール)`
  : 'Chromium: playwright 管理のブラウザを使用');
const browser = await chromium.launch(launchOptions);
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

console.log('\n== ブレイク予測 View ==');
await page.goto(base, { waitUntil: 'networkidle' });
await page.waitForSelector('.pred-row', { timeout: 15000 });

const predBody = await page.textContent('body');
const nRows = await page.locator('.pred-row').count();
check(nRows === 5, `候補が5件描画される (実際: ${nRows})`);

// モデル別の縦棒。5モデル × 候補数 だけ出る
const strips = await page.locator('.pred-ms-i').count();
check(strips === nRows * 5, `モデル別の棒が ${nRows}行 × 5モデル (実際: ${strips})`);
const shorts = await page.locator('.pred-row').first().locator('.pred-ms-i .k')
  .allTextContents();
check(shorts.join(',') === 'LGB,XGB,CAT,LR,NN',
  `モデルの並び順が行をまたいで固定 (実際: ${shorts.join(',')})`);
check(predBody.includes('が上位10%'), '一致度（上位10%と見たモデル数）が表示される');
// 塊の区切り（木3本 | 木以外2本）が各行に1本入る
const seps = await page.locator('.pred-ms-sep').count();
check(seps === nRows, `塊の区切りが各行に1本ある (実際: ${seps})`);
check(predBody.includes('決定木系') && predBody.includes('木以外'),
  '塊の名前が一覧パネルに出る');
check(predBody.includes('並べているモデル'), 'モデル一覧パネルが表示される');
check(predBody.includes('ニューラルネット') && predBody.includes('ロジスティック回帰'),
  '5モデルの日本語名が出る');
check(!predBody.includes('NaN'), '予測タブに NaN が出ていない');
check(!predBody.includes('undefined'), '予測タブに undefined が出ていない');

// 1件目（5モデルとも高い候補）を開いて、モデル別の見立てを確かめる
await page.locator('.pred-row').first().locator('.pred-main').click();
await page.waitForSelector('.pred-detail', { timeout: 5000 });
const detail = await page.textContent('.pred-detail');
check(detail.includes('モデル別の見立て'), '詳細にモデル別の見立てが出る');
check(detail.includes('混ぜていません'), 'アンサンブルではないと明示される');
const detailBars = await page.locator('.pred-detail .pred-b').count();
check(detailBars >= 5, `詳細にモデル別の棒が5本以上ある (実際: ${detailBars})`);
const fams = await page.locator('.pred-detail .pred-fam').allTextContents();
check(fams.join(',') === '決定木系,木以外',
  `詳細のモデル別が塊で分かれる (実際: ${fams.join(',')})`);
await page.screenshot({ path: path.join(SHOTS, 'screenshot-prediction.png') });
await page.locator('.pred-row').first().locator('.pred-main').click();

console.log('\n== 8軸オクタゴン比較 View ==');
await page.getByRole('tab', { name: '8軸オクタゴン比較' }).click();
await page.waitForSelector('.recharts-surface', { timeout: 15000 });

const body = await page.textContent('body');
check(body.includes('テスト銘柄A'), '銘柄名が描画される');
check(await page.locator('.stock-card').count() === 3, '3銘柄ぶんのコントロールカード');
check(await page.locator('.recharts-radar').count() >= 1, 'レーダーチャートが描画される');
check(body.includes('BREAKOUT'), 'BREAKOUT ゾーン判定が表示される');
check(body.includes('機関熱狂') || body.includes('MEGA'), '機関投資家参入度バッジが表示される');
const covBadges = await page.locator('.stock-card .badge').allTextContents();
check(covBadges.includes('8/8軸'), '全軸そろった銘柄は 8/8軸');
check(covBadges.includes('5/8軸'), '欠測のある銘柄は 5/8軸 (EPS成長・需給・進捗が欠測)');
check(covBadges.includes('CAP LOW'), '時価総額100億円未満は CAP LOW と判定される');
check(!body.includes('NaN'), 'NaN が画面に出ていない');
check(!body.includes('undefined'), 'undefined が画面に出ていない');

// テスト銘柄A の総合スコアを scoring.js の期待値と突き合わせる
const scoreA = await page.locator('.stock-card').first().locator('.score-big').first().textContent();
check(/^\d\.\d/.test(scoreA.trim()), `総合スコアが数値で表示される (取得値: ${scoreA.trim()})`);

// 欠測軸が「—」になっていること
await page.locator('.stock-card').nth(1).locator('button').first().click();
await page.waitForTimeout(250);
const naCount = await page.locator('.axis-table .na').count();
check(naCount >= 3, `欠測軸が「—」で表示される (${naCount}箇所)`);

await page.screenshot({ path: path.join(SHOTS, 'screenshot-compare.png'), fullPage: false });

console.log('\n== タイムマシーン View ==');
await page.getByRole('tab', { name: 'タイムマシーン' }).click();
await page.waitForSelector('#tm-stock', { timeout: 10000 });
// 過去スナップショットを持つ銘柄を明示的に選択する
await page.selectOption('#tm-stock', { label: '0001 テスト銘柄A' });
await page.waitForTimeout(300);
const tmBody = await page.textContent('body');
check(tmBody.includes('6ヶ月前') && tmBody.includes('3ヶ月前'), '過去スナップショットが表示される');
check(tmBody.includes('52週高値を更新'), 'ストーリータイムラインのイベントが表示される');
check(await page.locator('.recharts-line').count() >= 1, '株価推移チャートが描画される');
await page.screenshot({ path: path.join(SHOTS, 'screenshot-timemachine.png') });

console.log('\n== What-If シミュレーター View ==');
await page.getByRole('tab', { name: 'What-If シミュレーター' }).click();
await page.waitForSelector('input[type=range]', { timeout: 10000 });
const sliders = await page.locator('input[type=range]').count();
check(sliders === 10, `スライダーが10本ある (実際: ${sliders})`);

// 対象を テスト銘柄A に固定してから操作する
await page.locator('select[aria-label="対象銘柄"]').selectOption({ label: '0001 テスト銘柄A' });
await page.waitForTimeout(300);
const readScore = async () =>
  parseFloat((await page.locator('.score-big').first().textContent()).trim());

const before = await readScore();
// EPS成長率を 0% まで下げる -> 軸1が満点(10.0)から0点になり総合が下がるはず
await page.locator('#sim-epsGrowth').fill('0');
await page.waitForTimeout(300);
const lowered = await readScore();
check(lowered < before, `スライダーを下げるとスコアが下がる (${before} -> ${lowered})`);

// 現在値に戻すと元のスコアに復帰する
await page.getByRole('button', { name: '現在値に戻す' }).click();
await page.waitForTimeout(300);
check(Math.abs((await readScore()) - before) < 0.05, '「現在値に戻す」で元のスコアに復帰する');

// 信用倍率を上げると需給軸が下がる
await page.locator('#sim-creditRatio').fill('15');
await page.waitForTimeout(300);
check((await readScore()) < before, '信用倍率を上げると総合スコアが下がる');

const perfText = await page.textContent('body');
const perfMatch = perfText.match(/再計算＋再描画:\s*([\d.]+)ms/);
check(!!perfMatch, '再描画時間が実測表示される');
if (perfMatch) {
  const ms = parseFloat(perfMatch[1]);
  check(ms <= 16, `§6.1 再計算+再描画が16ms以内 (実測 ${ms}ms)`);
}
check((await page.textContent('body')).includes('感度分析'), '感度分析テーブルが表示される');
await page.screenshot({ path: path.join(SHOTS, 'screenshot-simulator.png') });

console.log('\n== 銘柄追加モーダル ==');
await page.getByRole('tab', { name: '8軸オクタゴン比較' }).click();
await page.getByRole('button', { name: '+ 銘柄を追加' }).click();
await page.waitForSelector('.modal');
await page.locator('#f-code').fill('9999');
await page.locator('#f-name').fill('手入力テスト');
await page.locator('#f-epsGrowth').fill('45');
await page.locator('#f-highRatio').fill('99');
await page.getByRole('button', { name: '追加する' }).click();
await page.waitForTimeout(400);
const afterAdd = await page.textContent('body');
check(afterAdd.includes('手入力テスト'), '手入力銘柄が追加される');
check(await page.locator('.stock-card').count() === 4, '銘柄数が4に増える');
check(afterAdd.includes('手入力'), '手入力バッジで J-Quants 由来と区別される');

console.log('\n== コンソールエラー ==');
check(consoleErrors.length === 0, `JS エラーなし (${consoleErrors.length}件)`);
consoleErrors.slice(0, 5).forEach((e) => console.log('   ', e.slice(0, 200)));

await browser.close();
server.close();

console.log(`\n${failures.length === 0 ? '✅ すべて成功' : `❌ ${failures.length}件失敗`}`);
failures.forEach((f) => console.log('  -', f));
process.exit(failures.length === 0 ? 0 : 1);
