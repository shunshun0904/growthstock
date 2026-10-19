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
import { decode, knnLists, network, todayNeighbors, methodOf, BWR, NA_VALUE } from '../src/lib/recurrence.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DIST = path.join(ROOT, 'dist');
const FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-stocks.json');
// 予測タブも合成データで描く。実データを使うと、その日の候補数や
// モデルの有無で結果が変わり、失敗したときに再現できない
const PRED_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-predictions.json');
const FILINGS_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-filings.json');
const SAMPLES_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-new-samples.json');
// 候補銘柄のニュースのリンクを確かめた結果。research/news_links.py の書き出しで作った（偽の取得。9324 の掲示板は開けない）
const NEWS_FIXTURE = path.join(ROOT, 'tests', 'fixtures', 'synthetic-news-links.json');
// リカレンス・ネットワーク（目次と候補ごとのファイル）。tests/test_recurrence_net.py --write-fixtures が作りものの日足から作る
// （日付・銘柄コードは synthetic-predictions.json と同じ 2026-09-11 の 9324・9434）
const RECURRENCE_FIXTURES = path.join(ROOT, 'tests', 'fixtures', 'recurrence');
const RECURRENCE_EXPECTED = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'fixtures', 'recurrence-expected.json'), 'utf8'));
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
    '/data/news_links.json': NEWS_FIXTURE,
  };
  const rec = rel.startsWith('/data/recurrence/') ? path.join(RECURRENCE_FIXTURES, path.basename(rel)) : null;
  const file = fixtures[rel] || rec || path.join(DIST, rel);
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
// リカレンス・ネットワークは候補銘柄のニュースの右（運用者の依頼 2026-10-19。場所の指定は無かったので、候補ごとに見る
// タブどうしを並べた。再学習で加わったサンプルは一番右のまま）
check(tabNames.join(',') === 'ブレイク予測,候補銘柄のニュース,リカレンス・ネットワーク,タイムマシーン,再学習で加わったサンプル',
      `タブは ブレイク予測 / 候補銘柄のニュース / リカレンス・ネットワーク / タイムマシーン / 再学習で加わったサンプル の5つ (実際: ${tabNames.join(',')})`);
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

console.log('\n== 候補銘柄のニュース ==');
// 出すのは今日の戦略の候補（1.5σ の3本すべて95以上）だけ（運用者の指示 2026-10-17「取得対象は、候補銘柄のみで良いです」）。
// 合成の予測データには規則を通る候補が無いので、9324・9434・9082 の 1.5σ の3本の百分位を上書きして配り直す
// （tests/fixtures/synthetic-news-links.json は同じ値から research/news_links.py の書き出しで作った。9082 はリンクの記録が無い）
const PASS = { 9324: [98.1, 97.1, 97.4], 9434: [97.1, 96.5, 95.2], 9082: [96.0, 100.0, 95.0] };
const predPass = JSON.parse(fs.readFileSync(PRED_FIXTURE, 'utf8'));
for (const c of predPass.candidates) {
  const p = PASS[c.code];
  if (p) ['lgbm_k15', 'xgb_k15', 'cat_k15'].forEach((a, i) => { c.byModel[a].pctHistorical = p[i]; });
}
await page.route('**/data/predictions.json*',
  (route) => route.fulfill({ contentType: 'application/json', body: JSON.stringify(predPass) }));
await page.reload();
await page.getByRole('tab', { name: '候補銘柄のニュース' }).click();
await page.waitForSelector('.news-card', { timeout: 10000 });
const newsText = (await page.textContent('.news')) || '';
const cards = page.locator('.news-card');
check((await cards.count()) === 3, `規則を通った3件だけのカード（高値更新の5件のうち）(実際: ${await cards.count()})`);
const c1 = (await cards.nth(0).textContent()) || '';
const c3 = (await cards.nth(2).textContent()) || '';
check(c1.includes('9324') && c1.includes('買う候補') && c3.includes('9082') && c3.includes('95以上'),
      '並びは最小の百分位が高い順で、上位2件に「買う候補」、枠の外に「95以上」');
// 確かめは裏でやり、画面には印や状態を出さない（運用者の指示 2026-10-17「確かめた、未確認などの表示はいらないです」）。
// 9324 は掲示板が確かめを通らなかったので、Yahoo!ファイナンスの3本と自動では開かないサイトの6本で9本。9434 は10本
check((await cards.nth(0).locator('a.news-link').count()) === 9 && (await cards.nth(1).locator('a.news-link').count()) === 10,
      '確かめを通らなかったリンクだけを出さない（9324 は9本、9434 は10本）');
check((await cards.nth(0).locator('a[href="https://finance.yahoo.co.jp/quote/9324.T/forum"]').count()) === 0
        && (await cards.nth(0).locator('a[href="https://finance.yahoo.co.jp/quote/9324.T/news"]').count()) === 1
        && (await cards.nth(0).locator('a[href="https://kabutan.jp/stock/news?code=9324"]').count()) === 1
        && (await cards.nth(0).locator('a[href="https://www.nikkei.com/nkd/company/news/?scode=9324"]').count()) === 1,
      'リンクの URL は確かめた結果のファイルのまま（通らなかった掲示板は出ず、株探・日経は同じ塊に並ぶ）');
const groups1 = await cards.nth(0).locator('.news-group').evaluateAll((gs) => gs.map((g) => [
  g.querySelector('.lab').textContent, g.querySelectorAll('a').length]));
check(JSON.stringify(groups1) === JSON.stringify([['ニュース', 3], ['適時開示', 1], ['アナリスト予想・業績', 4], ['個人投資家の声', 1]]),
      `リンクは見出しの塊ごとに、自動では開かないサイトのものも同じ塊に並ぶ (実際: ${JSON.stringify(groups1)})`);
check(c3.includes('まだありません') && (await cards.nth(2).locator('a').count()) === 0,
      'リンクの記録の無い銘柄はリンクを1本も出さない');
check(['確かめた', '未確認', '出していないリンク', '判定し直し', '市場全体'].every((w) => !newsText.includes(w))
        && (await page.locator('.news .mark, .news .is-checked, .news .is-unchecked').count()) === 0,
      '確かめた・未確認の印、確かめの状態、市場全体のカードは出さない');
check((await page.locator('.news a.news-link').count()) === 19, '画面のリンクは候補の2銘柄ぶんの19本だけ');
const newsAttrs = await page.locator('.news a.news-link').evaluateAll((as) => as.map((a) => [a.target, a.rel, a.href]));
check(newsAttrs.length > 0 && newsAttrs.every(([t, r, h]) => t === '_blank' && r.includes('noopener') && r.includes('noreferrer')
        && h.startsWith('https://')), 'リンクはすべて新しいタブで開き（noopener noreferrer）、https');
check(newsText.includes('有料の記事やログインが要るページがあります'), '下に有料の記事の注意が出る');
check(!newsText.includes('NaN') && !newsText.includes('undefined'), '候補銘柄のニュースに NaN / undefined が出ていない');
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(200);
const newsFit = await page.evaluate(() => {
  const r = document.querySelector('.news-card').getBoundingClientRect();
  return { right: r.right, vw: document.documentElement.clientWidth,
           scroll: document.documentElement.scrollWidth > document.documentElement.clientWidth };
});
check(newsFit.right <= newsFit.vw + 0.5 && !newsFit.scroll,
      `スマホの幅（390px）でカードが画面に収まり、横に崩れない (右端 ${newsFit.right.toFixed(1)} / 幅 ${newsFit.vw})`);
await page.setViewportSize({ width: 1440, height: 1000 });
await page.unroute('**/data/predictions.json*');

console.log('\n== リカレンス・ネットワーク ==');
// 運用者の依頼（2026-10-19）「Obsidian の球状のプロットのようなことを、ノードを1営業日として…その日の各候補銘柄単体で」。
// 対象は購入条件（1.5σ の3本がすべて95以上）を満たした銘柄だけ（同日の指定「候補といったのは、予測サンプルではなく、その中でも
// 購入条件を満たした銘柄です」）。元の合成データ（2026-09-11 の5件。どれも満たさない）では図を出さずに理由を書く
await page.reload({ waitUntil: 'networkidle' });
await page.getByRole('tab', { name: 'リカレンス・ネットワーク' }).click();
// 読み込み中の文言も同じ枠（.card.empty）に出るので、理由の文言が出るまで待つ（枠だけを待つと読み込み中を読んでしまう）
await page.locator('.net .card.empty', { hasText: '購入条件' }).waitFor({ timeout: 10000 }).catch(() => {});
const netEmpty = (await page.locator('.net .card.empty').textContent()) || '';
check(netEmpty.includes('購入条件') && netEmpty.includes('高値更新 5 銘柄のうち'),
      `購入条件を満たした銘柄が無い日は、図を出さずに理由を書く (実際: ${netEmpty.slice(0, 70)})`);
check((await page.locator('.net-canvas canvas').count()) === 0 && (await page.locator('.net-crow').count()) === 0,
      '満たさない銘柄（高値更新の5件）は一覧にも図にも出さない');
// 上の候補銘柄のニュースと同じ predPass（9324・9434・9082 が満たす）で読み込み直す。ネットワークのフィクスチャは購入条件を
// 満たした3件のうち 9324・9434 の2件（9082 は日足が無いので作れない。tests/test_recurrence_net.py の fixture_build）
await page.route('**/data/predictions.json*',
  (route) => route.fulfill({ contentType: 'application/json', body: JSON.stringify(predPass) }));
await page.reload({ waitUntil: 'networkidle' });
await page.getByRole('tab', { name: 'リカレンス・ネットワーク' }).click();
await page.waitForSelector('.net-canvas canvas', { timeout: 10000 });
const exp = (code, k) => RECURRENCE_EXPECTED[`2026-09-11_${code}.json`][String(k)];
const netFocus = async () => (await page.locator('.net-canvas').getAttribute('data-focus')) || '';
const netN = (code) => JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'fixtures', 'recurrence', `2026-09-11_${code}.json`), 'utf8')).n;
const crows = page.locator('.net-crow:not(.head)');
check((await crows.count()) === 3, `一覧は購入条件を満たした3件だけ（高値更新の5件ではない） (実際: ${await crows.count()})`);
const crowCodes = await crows.locator('.id b').allTextContents();
check(crowCodes.join(',') === '9324,9434,9082', `今日の戦略の順（3本の最小 97.1・95.2・95.0） (実際: ${crowCodes.join(',')})`);
const pickMarks = await crows.evaluateAll((els) => els.map((e) => Boolean(e.querySelector('.net-pick'))));
check(pickMarks.join(',') === 'true,true,false', `上位2件（9324・9434）に「買う候補」の印 (実際: ${pickMarks.join(',')})`);
check((await page.locator('.net-crow:not(.head):not([disabled])').count()) === 2, 'ネットワークのある2件だけ選べる');
check(((await crows.nth(2).getAttribute('title')) || '').includes('日足が無い'), '作れなかった 9082 は理由（日足が無い）を出す');
const netStrip = (await page.locator('.net .pred-strip').innerText()) || '';
check(netStrip.includes('購入条件を満たした銘柄') && netStrip.includes('高値更新 5 銘柄のうち'),
      `件数の欄に「購入条件を満たした銘柄 3（高値更新 5 銘柄のうち）」 (実際: ${netStrip.replace(/\s+/g, ' ').slice(0, 90)})`);
check((await crows.first().getAttribute('aria-selected')) === 'true', '最初はネットワークのある先頭の銘柄（9324）を出す');
const netSide = async () => (await page.locator('.net-side').innerText()) || '';
let side = await netSide();
// k近傍（運用者の選択 2026-10-19「k近傍だけにする」。k は指定「kは1〜10までで良いです」で 1〜10、既定10）
const kBtn = (v) => page.locator(`.net-seg[aria-label^="似た日の数"] button[aria-label="${v} 日"]`);
const kBtns = await page.locator('.net-seg[aria-label^="似た日の数"] button').allTextContents();
check(kBtns.join(',') === '1,2,3,4,5,6,7,8,9,10', `k は 1〜10 日から選ぶ (実際: ${kBtns.join(',')})`);
check((await page.locator('.net-seg[aria-label^="似た日の数"] button[aria-pressed="true"]').getAttribute('aria-label')) === '10 日', '既定の k は 10');
check((await page.locator('[aria-label="辺の密度（リカレンス率）"]').count()) === 0
        && !(await page.locator('.net .pred-strip').innerText()).includes('辺の密度'), '辺の密度のボタンは無い（ε はやめた）');
check(side.includes(`${exp('93240', 10).deg} 日`) && side.includes(`${exp('93240', 10).periods} つの時期`) && side.includes('珍しい状態'),
      `9324 の今日に似た日は ${exp('93240', 10).deg} 日・${exp('93240', 10).periods} つの時期（Python の要約と同じ）で、珍しさの説明が出る`);
const crowCells = await crows.first().locator(':scope > span.num').allTextContents();   // 最小・似た日・全期間
const pctTxt = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—');
check(crowCells[1] === pctTxt(exp('93240', 10).nbr.hit) && crowCells[2] === pctTxt(exp('93240', 10).base.hit),
      `一覧の「似た日」「全期間」は目次の k = 10 の要約（+10% 到達）と同じ (実際: ${crowCells.join(' / ')})`);
// 今日に似た日を光らせる（運用者の依頼「対象日と似ている（紐づく）サンプルもハイライト」）。強調している日:似た日の数
await page.waitForFunction(() => document.querySelector('.net-canvas')?.dataset.focus);
check(await netFocus() === `${netN('93240') - 1}:${exp('93240', 10).deg}`,
      `開いたときから今日（最後の点）と似た ${exp('93240', 10).deg} 日を光らせている (実際: ${await netFocus()})`);
await page.locator('label', { hasText: '今日に似た日を光らせる' }).click();
await page.waitForFunction(() => document.querySelector('.net-canvas')?.dataset.focus === '');
check(await netFocus() === '', '「今日に似た日を光らせる」を外すと強調しない');
await page.locator('label', { hasText: '今日に似た日を光らせる' }).click();
const inked = await page.evaluate(() => {
  const c = document.querySelector('.net-canvas canvas');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4 * 7) if (d[i] > 40 || d[i + 1] > 40 || d[i + 2] > 60) n += 1;
  return n;
});
check(inked > 200, `図に点と線が描かれている（明るい画素 ${inked}）`);
await page.locator('.net-crow', { hasText: '9434' }).click();
await page.waitForFunction(() => document.querySelector('.net-head strong')?.textContent.includes('9434'));
await kBtn(5).click();
side = await netSide();
check(side.includes(`${exp('94340', 5).deg} 日`) && side.includes(`${exp('94340', 5).periods} つの時期`),
      `9434 を k = 5 で数え直すと今日に似た日 ${exp('94340', 5).deg} 日・${exp('94340', 5).periods} つの時期（Python と同じ）`);
const foot = (await page.locator('.net-foot').textContent()) || '';
check(foot.includes(`線 ${exp('94340', 5).edges.toLocaleString('ja-JP')} 本`), `線の本数も Python と同じ（${exp('94340', 5).edges}）`);
await page.waitForFunction((want) => document.querySelector('.net-canvas')?.dataset.focus === want,
  `${netN('94340') - 1}:${exp('94340', 5).deg}`, { timeout: 5000 }).catch(() => {});
check(await netFocus() === `${netN('94340') - 1}:${exp('94340', 5).deg}`, `k = 5 なら光らせる似た日も 5 日 (実際: ${await netFocus()})`);
// 右の一覧は図で光らせる日と同じ近い k 日（運用者の指定 2026-10-19「プロット上の近傍の日付で統一してほしい」）。日付を
// フィクスチャから同じ式で数えた今日の近い5日（近い順）と比べる。更新日だけから選んだ「今日に近い過去の更新日」の欄は無い
const nearWant = (() => {
  const idx = JSON.parse(fs.readFileSync(path.join(RECURRENCE_FIXTURES, 'index.json'), 'utf8'));
  const m = methodOf(idx);
  const dec = decode(JSON.parse(fs.readFileSync(path.join(RECURRENCE_FIXTURES, '2026-09-11_94340.json'), 'utf8')), m.labels);
  const net = network(dec, m.theiler, 5, knnLists(dec, m.theiler, 5));
  return todayNeighbors(dec, net).map((r) => dec.dates[r.i].replace(/-/g, '/') + (dec.isFresh[r.i] ? ' ○' : ''));   // ○ = 更新日
})();
const nearDates = await page.locator('.net-near tbody tr td:first-child').allTextContents();
check(nearDates.join(',') === nearWant.join(',') && nearDates.length === exp('94340', 5).deg,
      `「今日に似た日」の表は図で光らせる近い5日と同じ日・同じ順 (実際: ${nearDates.join(',')} / 期待: ${nearWant.join(',')})`);
check((await page.locator('.net-near h3').textContent()).includes(`図で今日と結んだ ${exp('94340', 5).deg} 日`)
        && !(await netSide()).includes('今日に近い過去の更新日') && (await page.locator('.net-details').count()) === 0,
      '「今日に近い過去の更新日」の欄と、同じ中身だった「表で見る」は無い（一覧は1つ）');
await page.locator('label', { hasText: '回転' }).click();
await page.waitForSelector('.net-canvas[data-settled="1"]', { timeout: 20000 });
const [todayX, todayY] = ((await page.locator('.net-canvas').getAttribute('data-today')) || '0,0').split(',').map(Number);
await page.locator('.net-canvas canvas').scrollIntoViewIfNeeded();    // 図の下が画面の外だと、そこにある点に触れられない
const cbox = await page.locator('.net-canvas canvas').boundingBox();
await page.mouse.move(cbox.x + todayX, cbox.y + todayY);
await page.waitForSelector('.net-tip', { timeout: 3000 }).catch(() => {});
const netTip = (await page.locator('.net-tip').textContent().catch(() => '')) || '';
check(netTip.includes('（今日）') && netTip.includes('2026/09/11'), `今日の点に触れると日付と（今日）が出る (実際: ${netTip.slice(0, 40)})`);
check(netTip.includes('20日以内の最高値まで未確定') && netTip.includes('20営業日目の騰落率未確定'),
      '今日の点の先20営業日の値動きは未確定（予測日より後の日足は読まない）');
await page.mouse.move(cbox.x + 2, cbox.y + 2);
await page.locator('.net-seg button', { hasText: '時期' }).click();
check((await page.locator('.net-legend .net-ramp').count()) === 1, '時期の色にすると凡例が古い → 新しいの帯になる');
// 時期・最高値までの色は青白赤（運用者の指定「青白ではなく、青白赤で」→ 選択「高い・新しい方を赤」）
const rgbOf = (h) => `rgb(${parseInt(h.slice(1, 3), 16)}, ${parseInt(h.slice(3, 5), 16)}, ${parseInt(h.slice(5, 7), 16)})`;
const rampBg = await page.locator('.net-legend .net-ramp i').evaluate((e) => getComputedStyle(e).backgroundImage);
const rampStops = rampBg.match(/rgb\([^)]*\)/g) || [];
check(rampStops.join(',') === BWR.map(rgbOf).join(','),
      `時期の帯は古い = 青 → 白 → 新しい = 赤 (実際: ${rampStops.join(' ')})`);
// ノードの色付け（運用者の依頼 2026-10-19「先20日間の最大上昇率は最低限」→「その日から最高到達価格（20日以内最高高値）
// までの上昇率」）: 凡例の5段と未確定の件数が、フィクスチャの値をここで数え直した件数と同じ。今日に似た日の内訳・表の列も出る
await page.locator('.net-seg button', { hasText: '最高値までの上昇率（20日以内）' }).click();
const fx94 = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests', 'fixtures', 'recurrence', '2026-09-11_94340.json'), 'utf8'));
const riseWant = { '20% 以上': 0, '10〜20%': 0, '5〜10%': 0, '0〜5%': 0, '0% 未満': 0, '未確定': 0 };
for (const v of fx94.rise) {
  riseWant[v === null ? '未確定' : v >= 20 ? '20% 以上' : v >= 10 ? '10〜20%' : v >= 5 ? '5〜10%' : v >= 0 ? '0〜5%' : '0% 未満'] += 1;
}
const legendOf = async () => Object.fromEntries((await page.locator('.net-legend > span:has(.sw)').allInnerTexts()).map((t) => {
  const m = t.replace(/\s+/g, ' ').trim().match(/^(.*) ([\d,]+)$/);
  return m ? [m[1], Number(m[2].replace(/,/g, ''))] : [t, NaN];
}));
const riseGot = await legendOf();
check(JSON.stringify(riseGot) === JSON.stringify(riseWant),
      `最高値までの上昇率の凡例（高い順の5段と未確定）の件数がファイルの値と同じ (実際: ${JSON.stringify(riseGot)})`);
const riseSw = await page.locator('.net-legend .sw').evaluateAll((els) => els.map((e) => getComputedStyle(e).backgroundColor));
check(riseSw.join(',') === [...BWR].reverse().concat(NA_VALUE).map(rgbOf).join(','),
      `最高値までの段は高い順に 赤・薄い赤・白（5〜10%）・薄い青・青 と未確定の灰 (実際: ${riseSw.join(' ')})`);
const nbrMix = (await page.locator('.net-nbrmix').textContent()) || '';
check(nbrMix.includes(`今日に似た日 ${exp('94340', 5).deg} 日の内訳`), `凡例の下に今日に似た日の内訳 (実際: ${nbrMix.slice(0, 60)})`);
const riseHead = await page.locator('.net-near thead th').allTextContents();
const riseCells = await page.locator('.net-near tbody tr td:nth-child(4)').allTextContents();
check(riseHead.includes('最高値まで') && riseCells.length === exp('94340', 5).deg && riseCells.every((t) => /^[+-]?\d+\.\d%$/.test(t)),
      `今日に似た日の表に最高値までの列（どれも結果が出そろっているので値がある） (実際: ${riseCells.slice(0, 3).join(' ')})`);
// 最安値までの色は外した（運用者の指定「安値の色付けは、なしで良いです」）。軌跡は直近20営業日（「軌跡は直近20営業日で良いです」）
const colorBtns = await page.locator('.net-seg[aria-label="点の色"] button').allTextContents();
check(colorBtns.length === 6 && !colorBtns.some((t) => t.includes('最安値')),
      `色の付け方に最安値までは無い（6通り） (実際: ${colorBtns.join(' / ')})`);
check((await page.locator('label', { hasText: '直近20営業日の軌跡' }).count()) === 1, '軌跡は直近20営業日');
await page.locator('.net-seg button', { hasText: '20営業日目の騰落率' }).click();
const r20Got = await legendOf();
check(Object.keys(r20Got).join(',') === '+10% 以上,0〜+10%,−10〜0%,−10% 未満,未確定'
        && r20Got['未確定'] === fx94.r20.filter((v) => v === null).length,
      `20営業日目の騰落率は結果と同じ区切りの4段と未確定 (実際: ${JSON.stringify(r20Got)})`);
// 並べ方（運用者の依頼「無理に球状にしなくても3次元ネットワーク構造でかっこよく」。最初は力学: 同日の選択「力学（Obsidian 風）」）
const layoutBtns = await page.locator('.net-seg[aria-label="並べ方"] button').allTextContents();
check(layoutBtns.join(',') === '力学,状態の地図,時間のらせん'
        && (await page.locator('.net-seg[aria-label="並べ方"] button[aria-pressed="true"]').textContent()) === '力学',
      `並べ方は 力学（最初）・状態の地図・時間のらせん (実際: ${layoutBtns.join(',')})`);
const inkedNow = () => page.evaluate(() => {
  const c = document.querySelector('.net-canvas canvas');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let m = 0;
  for (let i = 0; i < d.length; i += 4 * 7) if (d[i] > 60 || d[i + 1] > 60 || d[i + 2] > 90) m += 1;
  return m;
});
for (const name of ['状態の地図', '時間のらせん']) {
  await page.locator('.net-seg[aria-label="並べ方"] button', { hasText: name }).click();
  await page.waitForSelector('.net-canvas[data-settled="1"]', { timeout: 5000 });
  await page.waitForTimeout(150);
  check((await inkedNow()) > 200 && (await page.locator('.net-seg button', { hasText: '力の調整' }).count()) === 0,
        `${name}でも描ける（動かさない並べ方なので力の調整は出さない）`);
}
await page.locator('.net-seg[aria-label="並べ方"] button', { hasText: '力学' }).click();
await page.locator('.net-seg button', { hasText: '2D' }).click();
await page.locator('.net-seg button', { hasText: '力の調整' }).click();
check((await page.locator('.net-forces input[type=range]').count()) === 4, '力の調整は Obsidian と同じ4つのつまみ');
side = await netSide();
check(!side.includes('NaN') && !side.includes('undefined'), 'ネットワークの欄に NaN / undefined が出ていない');
// README の画像（docs/screenshot-network.png）は実データで撮るので、ここでは別の名前に書く（.gitignore。CI の画像には入る）
await page.screenshot({ path: path.join(SHOTS, 'screenshot-network-smoke.png') });
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(200);
const netFit = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
check(netFit, 'スマホの幅（390px）で横にはみ出さない');
await page.setViewportSize({ width: 1440, height: 1000 });
await page.unroute('**/data/predictions.json*');
// 指で拡大縮小（運用者の依頼 2026-10-19「指で拡大縮小できるようにしたい」）: 触れる画面で、2本指を 60px → 220px に広げると
// 3.67 倍になる（指の間を中心に）。CDP の触る操作を Chromium が pointer イベントにする
{
  const tctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  const tpage = await tctx.newPage();
  tpage.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
  await tpage.route('**/data/predictions.json*',
    (route) => route.fulfill({ contentType: 'application/json', body: JSON.stringify(predPass) }));
  await tpage.goto(base, { waitUntil: 'networkidle' });
  await tpage.getByRole('tab', { name: 'リカレンス・ネットワーク' }).click();
  await tpage.waitForSelector('.net-canvas canvas', { timeout: 10000 });
  await tpage.locator('.net-canvas canvas').scrollIntoViewIfNeeded();
  const tb = await tpage.locator('.net-canvas canvas').boundingBox();
  const cx = tb.x + tb.width / 2;
  const cy = tb.y + tb.height / 2;
  const cdp = await tctx.newCDPSession(tpage);
  const pts = (dd) => [{ x: cx - dd, y: cy, id: 1 }, { x: cx + dd, y: cy, id: 2 }];
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: pts(30) });
  for (let dd = 40; dd <= 110; dd += 10) await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: pts(dd) });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await tpage.waitForTimeout(200);
  const zoomed = Number(await tpage.locator('.net-canvas').getAttribute('data-zoom'));
  check(Math.abs(zoomed - 220 / 60) < 0.01, `2本指でつまんで広げると拡大する（60px → 220px で ${(220 / 60).toFixed(2)} 倍。実際: ${zoomed}）`);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: pts(110) });
  for (let dd = 100; dd >= 50; dd -= 10) await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: pts(dd) });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await tpage.waitForTimeout(200);
  const back = Number(await tpage.locator('.net-canvas').getAttribute('data-zoom'));
  check(Math.abs(back - zoomed * (100 / 220)) < 0.01, `つまんで縮めると縮小する（実際: ${back}）`);
  await tctx.close();
}

console.log('\n== コンソールエラー ==');
check(consoleErrors.length === 0, `JS エラーなし (${consoleErrors.length}件)`);
consoleErrors.slice(0, 5).forEach((e) => console.log('   ', e.slice(0, 200)));

await browser.close();
server.close();

console.log(`\n${failures.length === 0 ? '✅ すべて成功' : `❌ ${failures.length}件失敗`}`);
failures.forEach((f) => console.log('  -', f));
process.exit(failures.length === 0 ? 0 : 1);
