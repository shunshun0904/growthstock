/**
 * 候補銘柄のニュース（運用者の依頼 2026-10-17「候補銘柄に関してのニュースや市場のセンチメント情報を収集して、
 * ダッシュボードに表示することは可能ですか？リンクでも良いです」「候補銘柄のニュース として新たにタブを追加して
 * ほしいです。左から2番目で」）。候補ごとに、外部サイトの銘柄のページへのリンクを並べる。
 *
 * URL の形は、実在を確かめたものだけを置く（推測で足さない）:
 *   - Yahoo!ファイナンス・日経・株予報Pro … Actions で見本の銘柄（7203）と英字入りのコードのページを開き、
 *     銘柄のページ（題名と本文にコード）が返ることを確かめた（research/exp/probe_news_links.py、run 39017769927）
 *   - 市場全体の日経・Yahoo!ファイナンス・日経平均VI・JPX … 同じ probe で開いて題名を確かめた
 *   - 株探・みんかぶ … Actions からの接続を断られた（405）ので、検索結果に実在するページで形を確かめた
 *     （株探 stock/news?code=5185・6594・285A、みんかぶ stock/3562・285A/analyst_consensus・285A/pick・
 *     financial_item_ranking/buy_picks_rise。2026-10-17）
 * 確かめられなかったもの（Google ニュースの検索・X・四季報オンライン・IR BANK）は置かない。
 *
 * 見出しを自動で集めて画面に載せることはしない。Google ニュースと TDnet は robots.txt で全体を禁止し、
 * Yahoo!ファイナンス（LINEヤフー共通利用規約 8.3）・株予報Pro（サイトポリシー）・JPX（ご利用上の注意）は
 * 転載・機械的な取得を禁じている（同じ probe の §2・§3。docs/OPERATIONS.md「候補銘柄のニュース」）。
 */

/** 東証の銘柄コード（4桁。2024年からは英字入りもある）。それ以外ではリンクを作らない。 */
const CODE_RE = /^[0-9A-Z]{4}$/;

/** リンクの塊。並びは画面の並び。 */
export const NEWS_GROUPS = [
  { id: 'news', label: 'ニュース' },
  { id: 'disclosure', label: '適時開示' },
  { id: 'forecast', label: 'アナリスト予想・業績' },
  { id: 'voice', label: '個人投資家の声' },
];

/**
 * 銘柄ごとのリンク。url は4桁のコードを受け取る。note は画面の title（有料・ログインなどの注意）。
 * Yahoo!ファイナンスのコードは「<コード>.T」（東証）。
 */
export const NEWS_SOURCES = [
  { id: 'kabutan_news', group: 'news', site: '株探', label: 'ニュース',
    url: (c) => `https://kabutan.jp/stock/news?code=${c}`,
    note: '材料ニュース・決算速報・適時開示の一覧。一部の記事は有料（株探プレミアム）' },
  { id: 'yahoo_news', group: 'news', site: 'Yahoo!ファイナンス', label: 'ニュース',
    url: (c) => `https://finance.yahoo.co.jp/quote/${c}.T/news`,
    note: '複数の配信元のニュースの一覧' },
  { id: 'nikkei_news', group: 'news', site: '日経', label: 'ニュース',
    url: (c) => `https://www.nikkei.com/nkd/company/news/?scode=${c}`,
    note: '日本経済新聞の記事の一覧。有料会員限定の記事がある' },
  { id: 'yahoo_disclosure', group: 'disclosure', site: 'Yahoo!ファイナンス', label: '適時開示',
    url: (c) => `https://finance.yahoo.co.jp/quote/${c}.T/disclosure`,
    note: '会社が出した適時開示（決算短信・業績予想の修正など）の一覧' },
  { id: 'kabuyoho_analyst', group: 'forecast', site: '株予報Pro', label: 'コンセンサス',
    url: (c) => `https://kabuyoho.jp/reportAnalyst?bcode=${c}`,
    note: 'アナリスト予想の平均（コンセンサス）と、その変化' },
  { id: 'minkabu_analyst', group: 'forecast', site: 'みんかぶ', label: 'アナリスト予想',
    url: (c) => `https://minkabu.jp/stock/${c}/analyst_consensus`,
    note: 'アナリストの目標株価とレーティングの内訳' },
  { id: 'yahoo_perf', group: 'forecast', site: 'Yahoo!ファイナンス', label: '業績・予想',
    url: (c) => `https://finance.yahoo.co.jp/quote/${c}.T/performance`,
    note: '業績の推移と会社予想・アナリスト予想' },
  { id: 'nikkei_kessan', group: 'forecast', site: '日経', label: '決算・財務',
    url: (c) => `https://www.nikkei.com/nkd/company/kessan/?scode=${c}`,
    note: '決算発表と業績・財務の情報' },
  { id: 'yahoo_forum', group: 'voice', site: 'Yahoo!ファイナンス', label: '掲示板',
    url: (c) => `https://finance.yahoo.co.jp/quote/${c}.T/forum`,
    note: '個人投資家の書き込み。真偽の確かめられない書き込みも多い' },
  { id: 'minkabu_pick', group: 'voice', site: 'みんかぶ', label: '個人の売買予想',
    url: (c) => `https://minkabu.jp/stock/${c}/pick`,
    note: '個人投資家の買い・売りの予想と予想株価' },
];

/** 市場全体（銘柄によらない）。地合いと需給の目安。 */
export const MARKET_LINKS = [
  { id: 'nikkei_markets', site: '日経', label: 'マーケットのニュース', url: 'https://www.nikkei.com/markets/' },
  { id: 'yahoo_newslist', site: 'Yahoo!ファイナンス', label: '経済ニュース', url: 'https://finance.yahoo.co.jp/news' },
  { id: 'nk225vi', site: '日経', label: '日経平均VI（変動の大きさの予想）',
    url: 'https://indexes.nikkei.co.jp/nkave/index/profile?idx=nk225vi' },
  { id: 'jpx_short', site: 'JPX', label: '空売り集計',
    url: 'https://www.jpx.co.jp/markets/statistics-equities/short-selling/index.html' },
  { id: 'jpx_investor', site: 'JPX', label: '投資部門別売買状況',
    url: 'https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html' },
  { id: 'minkabu_rank', site: 'みんかぶ', label: '買い予想数の上昇ランキング',
    url: 'https://minkabu.jp/financial_item_ranking/buy_picks_rise' },
];

/** 候補の4桁のコード（大文字）。形が違えば null。 */
export function stockCode(candidate) {
  const code = String(candidate?.code ?? '').trim().toUpperCase();
  return CODE_RE.test(code) ? code : null;
}

/**
 * 候補1件のリンクを塊ごとに返す: [{ id, label, links: [{ id, site, label, href, note }] }]。
 * コードの形が違う候補は空の配列（リンクを作らない）。
 */
export function newsLinks(candidate) {
  const code = stockCode(candidate);
  if (!code) return [];
  const enc = encodeURIComponent(code);
  return NEWS_GROUPS.map((g) => ({
    id: g.id,
    label: g.label,
    links: NEWS_SOURCES.filter((s) => s.group === g.id)
      .map((s) => ({ id: s.id, site: s.site, label: s.label, href: s.url(enc), note: s.note })),
  })).filter((g) => g.links.length > 0);
}
