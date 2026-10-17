/**
 * 候補銘柄のニュースのリンクの並べ方（src/lib/news.js）の単体テスト。URL と確かめた結果は
 * public/data/news_links.json（research/news_links.py）にあり、ここはそれを画面の形に並べるだけ。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_GROUPS, stockCode, stockLinks, marketLinks, linkSummary, noAutoReasons } from '../src/lib/news.js';

const link = (id, group, state, extra = {}) => ({
  id, group, state, site: 'S', label: id, href: `https://finance.yahoo.co.jp/quote/7203.T/${id}`, note: 'n',
  reason: state === 'ok' ? '' : 'r', ...extra,
});

const DATA = {
  checkedAt: '2026-10-17T10:05:00Z',
  groups: DEFAULT_GROUPS,
  noAuto: [
    { host: 'www.nikkei.com', reason: '日経は、自動化された手段での取得を規約で禁じている' },
    { host: 'indexes.nikkei.co.jp', reason: '日経は、自動化された手段での取得を規約で禁じている' },
    { host: 'kabuyoho.jp', reason: '株予報Pro は、robots.txt で 20時間の間隔を求めている' },
  ],
  stocks: {
    7203: {
      links: [
        link('news', 'news', 'ok'),
        link('nikkei', 'news', 'unchecked', { href: 'https://www.nikkei.com/nkd/company/news/?scode=7203' }),
        link('disclosure', 'disclosure', 'ng'),
        link('performance', 'forecast', 'ok'),
        link('forum', 'voice', 'ok'),
        link('bad', 'news', 'ok', { href: 'javascript:alert(1)' }),       // https でないものは出さない
        link('odd', 'news', 'weird'),                                     // 状態が分からないものは出さない
      ],
    },
    '285A': { links: [link('news', 'news', 'ok', { href: 'https://finance.yahoo.co.jp/quote/285A.T/news' })] },
  },
  market: [
    { id: 'jpx_short', site: 'JPX', label: '空売り集計', href: 'https://www.jpx.co.jp/x.html', state: 'ok' },
    { id: 'nk225vi', site: '日経', label: 'VI', href: 'https://indexes.nikkei.co.jp/x', state: 'unchecked', reason: 'r' },
    { id: 'jpx_investor', site: 'JPX', label: '投資部門別', href: 'https://www.jpx.co.jp/y.html', state: 'ng', reason: 'HTTP 404' },
  ],
  summary: { stocks: 2, links: 8, ok: 4, ng: 1, unchecked: 1 },
};

test('stockCode: 4桁（英字入りも）だけを大文字で返す', () => {
  assert.equal(stockCode({ code: '7203' }), '7203');
  assert.equal(stockCode({ code: '285a' }), '285A');
  assert.equal(stockCode({ code: '72030' }), null);
  assert.equal(stockCode({}), null);
  assert.equal(stockCode(null), null);
});

test('stockLinks: 確かめて開けたものを塊ごとに、開けなかったもの・未確認は分けて返す', () => {
  const g = stockLinks(DATA, { code: '7203' });
  assert.equal(g.known, true);
  assert.deepEqual(g.checked.map((x) => [x.id, x.links.map((l) => l.id)]),
    [['news', ['news']], ['forecast', ['performance']], ['voice', ['forum']]]);   // 開けたものが無い塊（適時開示）は出さない
  assert.deepEqual(g.failed.map((l) => l.id), ['disclosure']);
  assert.deepEqual(g.unchecked.map((l) => l.id), ['nikkei']);
  // https でないリンク・状態が分からないリンクはどこにも出さない
  const all = [...g.checked.flatMap((x) => x.links), ...g.failed, ...g.unchecked].map((l) => l.id);
  assert.ok(!all.includes('bad') && !all.includes('odd'));
});

test('stockLinks: 英字入りのコードは大文字で引く。記録が無い銘柄はリンクを1本も出さない', () => {
  assert.equal(stockLinks(DATA, { code: '285a' }).checked[0].links[0].href, 'https://finance.yahoo.co.jp/quote/285A.T/news');
  for (const g of [stockLinks(DATA, { code: '9999' }), stockLinks(null, { code: '7203' }), stockLinks(DATA, { code: 'x' })]) {
    assert.deepEqual(g, { known: false, checked: [], failed: [], unchecked: [] });
  }
});

test('stockLinks: 塊の定義がファイルに無ければ既定の並び', () => {
  const g = stockLinks({ stocks: DATA.stocks }, { code: '7203' });
  assert.deepEqual(g.checked.map((x) => x.label), ['ニュース', 'アナリスト予想・業績', '個人投資家の声']);
});

test('marketLinks / linkSummary / noAutoReasons', () => {
  const m = marketLinks(DATA);
  assert.deepEqual([m.checked.map((l) => l.id), m.unchecked.map((l) => l.id), m.failed.map((l) => l.id)],
    [['jpx_short'], ['nk225vi'], ['jpx_investor']]);
  assert.deepEqual(marketLinks(undefined), { checked: [], failed: [], unchecked: [] });
  assert.deepEqual(linkSummary(DATA), { checkedAt: '2026-10-17T10:05:00Z', stocks: 2, ok: 4, ng: 1, unchecked: 1 });
  assert.equal(linkSummary(null), null);
  assert.deepEqual(linkSummary({}), { checkedAt: null, stocks: 0, ok: 0, ng: 0, unchecked: 0 });
  // 同じ理由は1つに
  assert.deepEqual(noAutoReasons(DATA), ['日経は、自動化された手段での取得を規約で禁じている',
    '株予報Pro は、robots.txt で 20時間の間隔を求めている']);
  assert.deepEqual(noAutoReasons(undefined), []);
});
