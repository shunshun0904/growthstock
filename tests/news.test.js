/**
 * 候補銘柄のニュースのリンク（src/lib/news.js）の単体テスト。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { NEWS_GROUPS, NEWS_SOURCES, MARKET_LINKS, stockCode, newsLinks } from '../src/lib/news.js';

test('stockCode: 4桁（英字入りも）だけを大文字で返す', () => {
  assert.equal(stockCode({ code: '7203' }), '7203');
  assert.equal(stockCode({ code: '285a' }), '285A');
  assert.equal(stockCode({ code: ' 130A ' }), '130A');
  assert.equal(stockCode({ code: '72030' }), null);        // J-Quants の5桁は使わない
  assert.equal(stockCode({ code: '72-3' }), null);
  assert.equal(stockCode({ code: '' }), null);
  assert.equal(stockCode({}), null);
  assert.equal(stockCode(null), null);
});

test('newsLinks: 確かめた形の URL を塊ごとに返す（数字のコード）', () => {
  const g = newsLinks({ code: '7203', name: 'トヨタ自動車' });
  assert.deepEqual(g.map((x) => x.id), ['news', 'disclosure', 'forecast', 'voice']);
  assert.deepEqual(g.map((x) => x.label), NEWS_GROUPS.map((x) => x.label));
  const href = Object.fromEntries(g.flatMap((x) => x.links).map((l) => [l.id, l.href]));
  assert.deepEqual(href, {
    kabutan_news: 'https://kabutan.jp/stock/news?code=7203',
    yahoo_news: 'https://finance.yahoo.co.jp/quote/7203.T/news',
    nikkei_news: 'https://www.nikkei.com/nkd/company/news/?scode=7203',
    yahoo_disclosure: 'https://finance.yahoo.co.jp/quote/7203.T/disclosure',
    kabuyoho_analyst: 'https://kabuyoho.jp/reportAnalyst?bcode=7203',
    minkabu_analyst: 'https://minkabu.jp/stock/7203/analyst_consensus',
    yahoo_perf: 'https://finance.yahoo.co.jp/quote/7203.T/performance',
    nikkei_kessan: 'https://www.nikkei.com/nkd/company/kessan/?scode=7203',
    yahoo_forum: 'https://finance.yahoo.co.jp/quote/7203.T/forum',
    minkabu_pick: 'https://minkabu.jp/stock/7203/pick',
  });
  // どのリンクにもサイト名・見出し・注意がある
  g.flatMap((x) => x.links).forEach((l) => {
    assert.ok(l.site && l.label && l.note, l.id);
  });
});

test('newsLinks: 英字入りのコードも同じ形（大文字）', () => {
  const links = newsLinks({ code: '285a', name: 'X' }).flatMap((x) => x.links);
  assert.equal(links.length, NEWS_SOURCES.length);
  assert.ok(links.find((l) => l.id === 'kabutan_news').href.endsWith('code=285A'));
  assert.ok(links.find((l) => l.id === 'yahoo_news').href.includes('/quote/285A.T/'));
  assert.ok(links.every((l) => l.href.includes('285A')));
});

test('newsLinks: コードの形が違えばリンクを作らない', () => {
  assert.deepEqual(newsLinks({ code: '72030', name: 'X' }), []);
  assert.deepEqual(newsLinks({ name: 'X' }), []);
  assert.deepEqual(newsLinks(undefined), []);
});

test('リンクはすべて https で、置き換え忘れの {} が無く、id が重ならない', () => {
  const all = [...newsLinks({ code: '7203' }).flatMap((x) => x.links).map((l) => l.href),
    ...MARKET_LINKS.map((l) => l.url)];
  all.forEach((u) => {
    assert.ok(u.startsWith('https://'), u);
    assert.ok(!/[{}]/.test(u), u);
    assert.doesNotThrow(() => new URL(u), u);
  });
  const ids = [...NEWS_SOURCES.map((s) => s.id), ...MARKET_LINKS.map((l) => l.id)];
  assert.equal(new Set(ids).size, ids.length);
  // 塊に属さないリンクは無い
  NEWS_SOURCES.forEach((s) => assert.ok(NEWS_GROUPS.some((g) => g.id === s.group), s.id));
});

test('確かめられなかった出どころ（Google ニュース検索・X・四季報・IR BANK）は置かない', () => {
  const all = [...NEWS_SOURCES.map((s) => s.url('7203')), ...MARKET_LINKS.map((l) => l.url)].join(' ');
  ['news.google.com', 'x.com', 'twitter.com', 'shikiho.toyokeizai.net', 'irbank.net'].forEach((h) => {
    assert.ok(!all.includes(h), h);
  });
});
