/**
 * 候補銘柄のニュースのリンクの並べ方（src/lib/news.js）の単体テスト。URL と確かめた結果は
 * public/data/news_links.json（research/news_links.py）にあり、ここはそれを画面の形に並べるだけ。
 * 画面には確かめの印を出さず、確かめを通らなかったリンク（ng）だけを出さない（運用者の指示 2026-10-17）。
 *   npm test
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_GROUPS, stockCode, stockLinks } from '../src/lib/news.js';

const link = (id, group, state, extra = {}) => ({
  id, group, state, site: 'S', label: id, href: `https://finance.yahoo.co.jp/quote/7203.T/${id}`, note: 'n',
  reason: state === 'ok' ? '' : 'r', ...extra,
});

const DATA = {
  groups: DEFAULT_GROUPS,
  stocks: {
    7203: {
      links: [
        link('news', 'news', 'ok'),
        link('nikkei', 'news', 'unchecked', { href: 'https://www.nikkei.com/nkd/company/news/?scode=7203' }),
        link('disclosure', 'disclosure', 'ng'),                          // 確かめを通らなかった → 出さない
        link('performance', 'forecast', 'ok'),
        link('kabuyoho', 'forecast', 'unchecked', { href: 'https://kabuyoho.jp/reportAnalyst?bcode=7203' }),
        link('forum', 'voice', 'ok'),
        link('bad', 'news', 'ok', { href: 'javascript:alert(1)' }),       // https でないものは出さない
        link('odd', 'news', 'weird'),                                     // 状態が分からないものは出さない
      ],
    },
    '285A': { links: [link('news', 'news', 'ok', { href: 'https://finance.yahoo.co.jp/quote/285A.T/news' })] },
  },
};

test('stockCode: 4桁（英字入りも）だけを大文字で返す', () => {
  assert.equal(stockCode({ code: '7203' }), '7203');
  assert.equal(stockCode({ code: '285a' }), '285A');
  assert.equal(stockCode({ code: '72030' }), null);
  assert.equal(stockCode({}), null);
  assert.equal(stockCode(null), null);
});

test('stockLinks: 確かめたものと自動では開かないサイトのものを、印なしで見出しの塊ごとに（ファイルの順）', () => {
  const g = stockLinks(DATA, { code: '7203' });
  assert.equal(g.known, true);
  assert.deepEqual(g.groups.map((x) => [x.id, x.links.map((l) => l.id)]),
    [['news', ['news', 'nikkei']], ['forecast', ['performance', 'kabuyoho']], ['voice', ['forum']]]);
  // 確かめを通らなかった（ng）・https でない・状態が分からないリンクはどこにも出さない。出すリンクが無い塊（適時開示）も出さない
  const all = g.groups.flatMap((x) => x.links).map((l) => l.id);
  assert.ok(!all.includes('disclosure') && !all.includes('bad') && !all.includes('odd'));
});

test('stockLinks: 英字入りのコードは大文字で引く。記録が無い銘柄はリンクを1本も出さない', () => {
  assert.equal(stockLinks(DATA, { code: '285a' }).groups[0].links[0].href, 'https://finance.yahoo.co.jp/quote/285A.T/news');
  for (const g of [stockLinks(DATA, { code: '9999' }), stockLinks(null, { code: '7203' }), stockLinks(DATA, { code: 'x' })]) {
    assert.deepEqual(g, { known: false, groups: [] });
  }
});

test('stockLinks: 塊の定義がファイルに無ければ既定の並び', () => {
  const g = stockLinks({ stocks: DATA.stocks }, { code: '7203' });
  assert.deepEqual(g.groups.map((x) => x.label), ['ニュース', 'アナリスト予想・業績', '個人投資家の声']);
});
