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
const FILINGS_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-filings.json');
const SAMPLES_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-new-samples.json');
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
    '/data/filings.json': FILINGS_FIXTURE,
    '/data/new_samples.json': SAMPLES_FIXTURE,
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

// モデル別の百分位の表（運用者の指示 2026-10-12「12個全てを載せてほしい。棒グラフ形式でなくてよい。大小が分かればよい」）。
// 行 = ラベル（1.2σ / 1.5σ / 1.8σ）、列 = 学習器。合成データは12本で、5件目だけ logit_k18 が欠ける
const gridCells = page.locator('.pred-list .pred-row .pred-main .pred-mg .c');
check((await gridCells.count()) === nRows * 12, `行ごとに12マス (実際: ${await gridCells.count()})`);
const firstGrid = page.locator('.pred-row').first().locator('.pred-main .pred-mg');
const heads = await firstGrid.locator('.h').allTextContents();
check(heads.join(',') === ',LGB,XGB,CAT,LR', `列は LGB / XGB / CAT / LR (実際: ${heads.join(',')})`);
const labels = await firstGrid.locator('.r').allTextContents();
check(labels.join(',') === '1.2σ,1.5σ,1.8σ', `行は 1.2σ / 1.5σ / 1.8σ (実際: ${labels.join(',')})`);
const vals = await firstGrid.locator('.c').allTextContents();
check(vals.join(',') === '85,96,98,56,84,97,97,54,81,94,95,51', `1件目の12本の値 (実際: ${vals.join(',')})`);
check(((await firstGrid.locator('.c').first().getAttribute('title')) || '').includes('過去スコア分布の下から 85.0%'),
      'マスの title に小数1桁の百分位が出る');
const look = (loc) => loc.evaluate((el) => {
  const st = getComputedStyle(el);
  return `${st.backgroundColor}|${st.color}`;
});
check((await look(firstGrid.locator('.c').nth(2))) === 'rgb(219, 234, 254)|rgb(10, 14, 23)',
      '98 のマスは最上位（95以上）の色に暗い字');
// 行は並び順で探さず、銘柄名で探す（並び順は下で別に確かめる）
const rowOf = (name) => page.locator('.pred-row', { hasText: name });
const lowCell = rowOf('野村アセットマネジメント').locator('.pred-main .pred-mg .c').first();   // 37.3
check((await look(lowCell)) === 'rgb(29, 78, 216)|rgb(232, 237, 246)', '37 のマスは 30 の区分の色に明るい字');
const naCells = rowOf('大和自動車交通').locator('.pred-main .pred-mg .c.na');
check((await naCells.count()) === 1 && ((await naCells.textContent()) || '').includes('—'),
      '欠けたモデルのマスは「—」');
check((await page.locator('.pred-ms-i').count()) === 0, 'モデル別の縦棒は出さない');
check(!predBody.includes('12本の合意'), '12本の合意（平均）は出さない');
check(predBody.includes('決定木系') && predBody.includes('ラベル1.8σ'), '塊の名前が一覧パネルに出る');

// 同じ表でスコアそのものに切り替えられる（運用者の依頼 2026-10-16「目的変数横断での比較ができないので、モデルのスコア自体も
// 表示してほしい。同じ形式で、青白の濃淡でモデル×目的変数横断で色付け」）。色の物差しは全候補・12本で1つ（最小〜最大の帯）
await page.locator('.pred-metric button', { hasText: 'スコア' }).click();
const scoreVals = await firstGrid.locator('.c').allTextContents();
check(scoreVals.length === 12 && scoreVals.every((v) => /^\d\.\d{2}$/.test(v)),
      `スコアに切り替えると1件目の12マスが小数2桁のスコアになる (実際: ${scoreVals.slice(0, 3).join(',')}…)`);
check((await page.locator('.pred-row').first().locator('.pred-ms .lab').textContent()).includes('モデル別のスコア'),
      '見出しが「モデル別のスコア」になる');
check((await page.locator('.pred-row').first().locator('.pred-ms-grad i').count()) === 1, 'スコアの色の物差し（最小〜最大の帯）が見出しに出る');
await page.locator('.pred-metric button', { hasText: '百分位' }).click();
check((await firstGrid.locator('.c').allTextContents()).join(',') === '85,96,98,56,84,97,97,54,81,94,95,51',
      '百分位に戻すと元の値');
check(predBody.includes('並べているモデル'), 'モデル一覧パネルが表示される');
check(predBody.includes('LightGBM（1.5σ）') && predBody.includes('ロジスティック回帰（1.8σ）'),
  '12本の日本語名が出る');
// 今日の戦略パネル（docs/PLAYBOOK.md の手順）。
// 合成データは候補5件。2026-10-12 から発火の数では見送らないので「発火が少ない日」。
// 1.5σ の3本がすべて95以上の候補は無いので、買う銘柄は出ない
const strat = await page.textContent('.strat');
check(strat.includes('今日の戦略'), '戦略パネルが表示される');
check(strat.includes('発火が少ない日'), '発火5件の日は「発火が少ない日」と出る（見送りではない）');
check(strat.includes('発火の数では見送らない'), 'パネルの見出しに「発火の数では見送らない」と出る');
check(strat.includes('1.5σ の3本がすべて95以上の銘柄はありません'), '買わない理由（基準を満たす銘柄なし）が出る');
check((await page.locator('.strat-pick').count()) === 0,
      '基準を満たす銘柄が無い日は買う銘柄を出さない');
check(strat.includes('発火数'), '発火数が表示される');
check(!strat.includes('NaN') && !strat.includes('undefined'),
      '戦略パネルに NaN / undefined が出ていない');

check(!predBody.includes('NaN'), '予測タブに NaN が出ていない');
check(!predBody.includes('undefined'), '予測タブに undefined が出ていない');

// 際どい候補の印（今の規則の線「1.5σ の3本とも95以上」の下）。合成データの1件目は 1.5σ の3本で
// lgbm_k15 83.6・xgb_k15 97.1・cat_k15 97.4（2本95以上で LightGBM が最下位。今の数字では赤の組が無いので黄）。
// 5件目は 1.2σ の3本なら際どい形（96・100・78.1）だが、1.5σ の3本（92.9・100・76.8）では際どくない → 印なし
const borders = page.locator('.pred-list .pred-border');
check((await borders.count()) === 1, `際どい候補の印が1件に出る (実際: ${await borders.count()})`);
const firstBorder = page.locator('.pred-row').first().locator('.pred-border');
check(((await firstBorder.textContent()) || '').includes('1.5σで際どい・最下位 LightGBM'),
      '1件目は「1.5σで際どい・最下位 LightGBM」と出る');
check(((await firstBorder.getAttribute('class')) || '').includes('amber'), '1件目の印は黄（赤の組は無い）');
check(((await firstBorder.getAttribute('title')) || '').includes('LightGBM が一番低いときは、過去 +1.6%（141件）'),
      '印の title にその形の過去の数字が出る');
check((await rowOf('大和自動車交通').locator('.pred-border').count()) === 0,
      '大和自動車交通（1.2σ の3本なら際どい形）には印が出ない');

// 一覧の並び順（運用者の決定 2026-10-13「1.5σ の3本のうち一番低い百分位でいきましょう」）。
// 合成データの 1.5σ の3本の最小: 安田倉庫 83.6・大和自動車交通 76.8・ソフトバンク 70.5・シンプレクス 59.9・
// 野村 35.5（1.2σ の LightGBM のスコア順なら 安田倉庫・野村・シンプレクス・ソフトバンク・大和自動車交通）
const names = await page.locator('.pred-list .pred-row .pred-id strong').allTextContents();
const want = ['安田倉庫', '大和自動車交通', 'ソフトバンク', 'シンプレクス', '野村アセットマネジメント'];
check(names.length === 5 && want.every((w, i) => (names[i] || '').includes(w)),
      `一覧は 1.5σ の3本の最小が高い順 (実際: ${names.map((t) => t.slice(0, 6)).join(' / ')})`);
const ranks = await page.locator('.pred-list .pred-rank').allTextContents();
check(ranks.join(',') === '183.6,276.8,370.5,459.9,535.5', `順位の数字は並び順で、下に最小を添える (実際: ${ranks.join(',')})`);
const rank2 = page.locator('.pred-list .pred-rank').nth(1);
check(((await rank2.getAttribute('title')) || '').includes('基準モデル（1.2σ の LightGBM）の順位 5 / 5'),
      '順位の title に基準モデルの順位を添える');
check((await rank2.evaluate((el) => getComputedStyle(el).color)) === 'rgb(151, 163, 184)',
      '最小が 85 未満の順位は控えめの字');
check(((await page.textContent('.card-head:has(h2:text-is("ブレイク予測")) .sub')) || '')
        .includes('1.5σ の3本のうち一番低い百分位が高い順'),
      '見出しに並び順の物差しが出る');
check((await page.locator('.pred-order-note').count()) === 0, '1.5σ の値がそろった日は並び順の注記を出さない');

// 1件目を開いて、モデル別の見立てを確かめる
await page.locator('.pred-row').first().locator('.pred-main').click();
await page.waitForSelector('.pred-detail', { timeout: 5000 });
const detail = await page.textContent('.pred-detail');
check(detail.includes('モデル別の見立て'), '詳細にモデル別の見立てが出る');
check(detail.includes('タイムマシーンで見る') && !detail.includes('8軸オクタゴン'),
      '詳細の送るボタンはタイムマシーン行き（8軸オクタゴン比較のタブは隠した）');
check(detail.includes('混ぜていません'), 'アンサンブルではないと明示される');
const bigGrids = page.locator('.pred-detail .pred-mg.lg');
check((await bigGrids.count()) === 2, `詳細には百分位とスコアの2つの表が出る (実際: ${await bigGrids.count()})`);
const bigCells = await bigGrids.nth(0).locator('.c').allTextContents();
check(bigCells.length === 12 && bigCells[0] === '85.0', `詳細に12本の表（小数1桁）が出る (実際: ${bigCells.slice(0, 3).join(',')}…)`);
const bigScores = await bigGrids.nth(1).locator('.c').allTextContents();
check(bigScores.length === 12 && bigScores.every((v) => /^\d\.\d{3}$/.test(v)),
      `詳細のスコアの表は12本・小数3桁 (実際: ${bigScores.slice(0, 3).join(',')}…)`);
check(detail.includes('スコアそのもの') && detail.includes('この表の12本の中の最小〜最大'),
      '詳細にスコアの表の読み方（色はこの表の12本の中の最小〜最大）が出る');
// 百分位の表とスコアの表は上下に並べたとき列がそろう（列の最小幅を数字の幅に依らない値にした。2026-10-16）
const gridWidths = await bigGrids.evaluateAll((els) => els.map((e) => Math.round(e.getBoundingClientRect().width)));
check(gridWidths.length === 2 && gridWidths[0] === gridWidths[1], `詳細の2つの表は同じ幅 (実際: ${gridWidths.join(' / ')})`);
// 12本とも本番で使うので、1.2σ を「本番」と呼ばない（運用者の指示 2026-10-17）。規則の 1.5σ の行に「規則」と添える
check(detail.includes('1.5σ（規則）') && !detail.includes('本番'), '詳細の表は 1.5σ の行に「規則」と出て、どこにも「本番」と出ない');
check(/並び順\s*1\.5σ の3本の最小\s*（83\.6）/.test(detail) && /帯の基準\s*LightGBM（1\.2σ）\s*（1 \/ 5 位）/.test(detail),
      '詳細に並び順の物差し（1.5σ の3本の最小）と帯の基準（1.2σ の LightGBM の順位）が分けて出る');
check(detail.includes('基準モデルの見立ての内訳') && detail.includes('基準モデル（1.2σ の LightGBM）の出力を')
        && detail.includes('一覧の並び順（1.5σ の3本の最小）の内訳ではありません'),
      '寄与の欄は 1.2σ の LightGBM のものと明示される');
check(detail.includes('上位10%と見たモデル') && /1\.2σ\s*2\/4/.test(detail) && detail.includes('1.5σ・1.8σ 4/8'),
      '詳細に上位10%と見たモデルの数（1.2σ の4本と、1.5σ・1.8σ の8本）が出る');
check(detail.includes('実験82'), '詳細に「1.5σ・1.8σ も95以上」を足しても安定しない（実験82）と出る');
check(!detail.includes('12本の合意'), '詳細にも12本の合意（平均）は出さない');
// 次の決算発表（予測日までに公表済みの予定）。合成データの1件目は7営業日後（持つ20営業日の中 → 黄）、
// 4件目は36営業日後（灰）、ほかは未公表（灰）。帯に「20営業日以内に決算 1 / 5」
const earnBadges = page.locator('.pred-list .pred-earn');
check((await earnBadges.count()) === 5, `次の決算の印が全件に出る (実際: ${await earnBadges.count()})`);
const firstEarn = page.locator('.pred-row').first().locator('.pred-earn');
check(((await firstEarn.textContent()) || '').includes('決算 7営業日後')
      && ((await firstEarn.getAttribute('class')) || '').includes('amber'), '1件目は「決算 7営業日後」で黄');
const strip = (await page.textContent('.pred-strip')) || '';
check(strip.includes('20営業日以内に決算') && /1\s*\/\s*5/.test(strip), '帯に「20営業日以内に決算 1 / 5」が出る');
check(((await page.textContent('.pred-detail')) || '').includes('次の決算発表 2026/09/25（第2四半期）'),
      '詳細に次の決算発表の日付と四半期が出る');

// 効いた特徴量は日本語名で出す（public/data/feature_names.json。元の列名はホバーの説明に添える）
const featNames = await page.locator('.pred-detail .pred-feats .pred-b .k').allTextContents();
check(featNames.length > 0 && featNames[0] === '20日ボラティリティ' && !featNames.some((t) => /_/.test(t)),
      `効いた特徴量が日本語名で出る (実際: ${featNames.slice(0, 3).join(' / ')})`);
const featTitle = await page.locator('.pred-detail .pred-feats .pred-b').first().getAttribute('title');
check((featTitle || '').includes('（vol_20d）'), 'ホバーの説明に元の列名が添えてある');
const borderNote = await page.textContent('.pred-detail .pred-border-note');
check((borderNote || '').includes('1.5σ の3本で見た際どい候補') && (borderNote || '').includes('+1.6%（141件）'),
      '詳細の先頭に際どい候補の注意（1.5σ の3本）が出る');
await page.screenshot({ path: path.join(SHOTS, 'screenshot-prediction.png') });
await page.locator('.pred-row').first().locator('.pred-main').click();

console.log('\n== タイムマシーン View ==');
await page.getByRole('tab', { name: 'タイムマシーン' }).click();
await page.waitForSelector('#tm-stock', { timeout: 10000 });
// 過去スナップショットを持つ銘柄を明示的に選択する
await page.selectOption('#tm-stock', { label: '0001 テスト銘柄A' });
await page.waitForTimeout(300);
const tmBody = await page.textContent('body');
check(tmBody.includes('6ヶ月前') && tmBody.includes('3ヶ月前'), '過去スナップショットが表示される');
check(tmBody.includes('78週高値を更新'), 'ストーリータイムラインのイベントが表示される');
check(await page.locator('.recharts-line').count() >= 1, '株価推移チャートが描画される');
await page.screenshot({ path: path.join(SHOTS, 'screenshot-timemachine.png') });

console.log('\n== 隠したタブ ==');
// 運用者の指示（2026-10-13）「決算サンキーとWhat-Ifシミュレーターのタブは消してください。表示から消してくれるだけで
// 良いです」、同日夜「8軸オクタゴン比較のタブも消してください」。画面のコードは残し、タブの一覧から外しただけ
// （src/App.jsx の TABS の hidden）。8軸オクタゴン比較の画面（と、そこにあった銘柄追加のボタン・直近決算のサンキー図）の
// 確かめはタブと一緒に外した
const tabNames = (await page.getByRole('tab').allTextContents()).map((t) => t.trim());
// 候補の比較（2026-10-15 に左から2番目に足した）は、運用者の指示 2026-10-16「候補の比較タブはいったん、なくしてください」で
// 隠している（src/App.jsx の hidden。画面のコードと日次予測の knn の計算は残す）。
// 再学習で加わったサンプルは一番右（運用者の指示 2026-10-14「一番右側に持ってきてください」）
// 候補銘柄のニュースは左から2番目（運用者の依頼 2026-10-17「候補銘柄のニュース として新たにタブを追加してほしいです。
// 左から2番目で」）
check(tabNames.join(',') === 'ブレイク予測,候補銘柄のニュース,タイムマシーン,再学習で加わったサンプル',
      `タブは ブレイク予測 / 候補銘柄のニュース / タイムマシーン / 再学習で加わったサンプル の4つ (実際: ${tabNames.join(',')})`);
check(!tabNames.includes('決算サンキー') && !tabNames.includes('What-If シミュレーター')
        && !tabNames.includes('8軸オクタゴン比較') && !tabNames.includes('候補の比較'),
      '決算サンキー・What-If シミュレーター・8軸オクタゴン比較・候補の比較のタブは出ない');
check((await page.locator('.knn').count()) === 0, '候補の比較の画面はどのタブにも出ない（隠している間）');
check((await page.locator('.sankey').count()) === 0 && (await page.locator('.stock-card').count()) === 0,
      '8軸オクタゴン比較の画面（銘柄カード・サンキー図）はどのタブにも出ない');
check(!((await page.textContent('.footer')) || '').includes('EDINET'),
      '決算サンキーを隠している間は EDINET の出典を出さない');

// 候補の比較の画面の確かめ（3枚の図・Q / M・母集団の件数・ホバー）は、タブを隠している間は外してある。純粋な計算の
// テストは tests/knn.test.js に残っている。タブを戻すときは git の履歴（2026-10-16 より前の smoke.mjs）から戻す

console.log('\n== 候補銘柄のニュース ==');
// 候補ごとに外部サイトのページへのリンク（src/lib/news.js。URL の形は実在を確かめたものだけ）。合成データの候補は5件
await page.getByRole('tab', { name: '候補銘柄のニュース' }).click();
await page.waitForSelector('.news-card', { timeout: 10000 });
const newsText = (await page.textContent('.news')) || '';
check((await page.locator('.news-card').count()) === 5, `合成データの候補5件のカード (実際: ${await page.locator('.news-card').count()})`);
check((await page.locator('.news-card').first().locator('.news-link').count()) === 10,
      '1件のカードにリンクが10本（ニュース3・適時開示1・アナリスト予想と業績4・個人投資家の声2）');
check((await page.locator('.news-market .news-link').count()) === 6, '市場全体のリンクが6本');
check(await page.locator('.news-card a[href="https://kabutan.jp/stock/news?code=9324"]').count() === 1,
      '株探のニュースのリンクが候補のコードで作られる');
check(await page.locator('.news-card a[href="https://finance.yahoo.co.jp/quote/9324.T/forum"]').count() === 1,
      'Yahoo!ファイナンスの掲示板のリンクが「<コード>.T」で作られる');
const newsLinkAttrs = await page.locator('.news a.news-link').evaluateAll((as) => as.map((a) => [a.target, a.rel, a.href]));
check(newsLinkAttrs.length > 0 && newsLinkAttrs.every(([t, r, h]) => t === '_blank' && r.includes('noopener')
        && r.includes('noreferrer') && h.startsWith('https://')),
      'リンクはすべて新しいタブで開き（noopener noreferrer）、https');
check(newsText.includes('ニュース') && newsText.includes('適時開示') && newsText.includes('アナリスト予想・業績')
        && newsText.includes('個人投資家の声'), 'リンクの塊の見出しが4つ出る');
check(newsText.includes('モデルの特徴量に入っていない材料') && newsText.includes('中身を取り込んでいません'),
      'リンク集の目的と、中身を取り込んでいないことが書いてある');
check(newsText.includes('決算 7営業日後'), '決算までの営業日の印が出る（合成データの1件目は7営業日後）');
check(!newsText.includes('NaN') && !newsText.includes('undefined'), '候補銘柄のニュースに NaN / undefined が出ていない');
// 規則を通った候補だけに絞る（合成データは1.5σ の百分位が無いので0件になり、その旨を出す）
await page.locator('.news .pred-metric .btn').nth(1).click();
check((await page.locator('.news-card').count()) === 0 && ((await page.textContent('.news')) || '').includes('の候補がありません'),
      '「95以上」に絞ると、通った候補が無い日はその旨を出す');
await page.locator('.news .pred-metric .btn').nth(0).click();
check((await page.locator('.news-card').count()) === 5, '「すべての候補」に戻すと5件に戻る');
// スマホの幅でもカードが枠に収まる（横にはみ出さない）
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(200);
const newsOverflow = await page.evaluate(() => {
  const card = document.querySelector('.news-card');
  const r = card.getBoundingClientRect();
  return { right: r.right, vw: document.documentElement.clientWidth,
           scroll: document.documentElement.scrollWidth > document.documentElement.clientWidth };
});
check(newsOverflow.right <= newsOverflow.vw + 0.5 && !newsOverflow.scroll,
      `スマホの幅（390px）でカードが画面に収まり、横に崩れない (右端 ${newsOverflow.right.toFixed(1)} / 幅 ${newsOverflow.vw})`);
await page.setViewportSize({ width: 1440, height: 1000 });

console.log('\n== 再学習で加わったサンプル ==');
// 運用者の依頼（2026-10-13）「直近追加されたサンプルのチャート（週次のリトレインで新たに追加されたサンプルのチャート）、
// 正例、負例で5例ずつ程度」、同日夜「チャートの表示期間は78週にしてください。タブの名前も「例」ではなく「サンプル」」。
// 合成データは加わった4件（正例2・負例2）を全部出す。日足は更新日（2025-07-28）の78週前 2024-02-28 から 2025-09-01 まで
await page.getByRole('tab', { name: '再学習で加わったサンプル' }).click();
await page.waitForSelector('.sample-card', { timeout: 10000 });
await page.waitForSelector('.sample-card .recharts-line', { timeout: 10000 });
const smp = (await page.textContent('.samples')) || '';
check((await page.locator('.sample-card').count()) === 4, `正例2・負例2の4枚のチャート (実際: ${await page.locator('.sample-card').count()})`);
check((await page.locator('.sample-card .recharts-line').count()) === 4, '各カードに終値の線が1本ずつ');
check(smp.includes('2026/10/11') && smp.includes('2025/07/25 より後 〜 2025/07/28'),
      '再学習の日と、加わった更新日の範囲が出る');
check(/加わった件数\s*4\s*件/.test(smp) && smp.includes('正例 2・負例 2・正例率 50.0%'), '加わった件数と正例率が出る');
check(smp.includes('2件中 2件を表示'), '正例・負例の見出しに件数が出る');
check(smp.includes('最大 +15.0%（必要 +9.5%）'), '正例の成否の内訳（到達）が数字で出る');
check((await page.locator('.sample-verdict li.ng').count()) === 6, '負例2件の3条件はどれも ✗（6行）');
check((await page.locator('.sample-verdict li.ok').count()) === 6, '正例2件の3条件はどれも ✓（6行）');
check(smp.includes('到達に必要な水準') && smp.includes('判定の20営業日'), '線の見分け方（凡例）が言葉で出る');
check(!smp.includes('NaN') && !smp.includes('undefined'), '再学習で加わったサンプルに NaN / undefined が出ていない');
check(smp.includes('再学習で加わったサンプル') && smp.includes('更新日の78週前から'), '見出しが「サンプル」で、78週の表示と書いてある');
const xTicks = (await page.locator('.sample-card').first().locator('.recharts-xAxis .recharts-cartesian-axis-tick-value')
  .allTextContents()).map((t) => t.trim());
check(xTicks.includes('24/4') && xTicks.includes('25/7') && xTicks.every((t) => /^\d{2}\/\d{1,2}$/.test(t)),
      `横軸は78週を四半期の初め（年/月）で刻む (実際: ${xTicks.join(' ')})`);
await page.screenshot({ path: path.join(SHOTS, 'screenshot-samples.png'), fullPage: true });

console.log('\n== コンソールエラー ==');
check(consoleErrors.length === 0, `JS エラーなし (${consoleErrors.length}件)`);
consoleErrors.slice(0, 5).forEach((e) => console.log('   ', e.slice(0, 200)));

await browser.close();
server.close();

console.log(`\n${failures.length === 0 ? '✅ すべて成功' : `❌ ${failures.length}件失敗`}`);
failures.forEach((f) => console.log('  -', f));
process.exit(failures.length === 0 ? 0 : 1);
