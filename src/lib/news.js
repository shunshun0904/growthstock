/**
 * 候補銘柄のニュース（運用者の依頼 2026-10-17「候補銘柄のニュース として新たにタブを追加してほしいです。左から2番目で」、
 * 同日「取得対象は、候補銘柄のみで良いです…ウソのリンクや無効なリンクがないことのバリデーションに時間をかけてください」）。
 *
 * リンクの URL と確かめた結果は public/data/news_links.json にある（research/news_links.py が毎晩の予測の後に、候補ごとに
 * ページを開いて書く。.github/workflows/news-links.yml）。ここはそれを画面の形に並べるだけで、URL は書き写さない。
 *
 * リンクの状態（state）:
 *   ok         候補ごとに開いて、その銘柄の、見出しどおりのページだと確かめた（「確かめた」の印を付けて出す）
 *   ng         開いたが条件を満たさなかった（無効なリンク。出さない。理由だけ小さく出す）
 *   unchecked  自動では開かないサイト（日経・株予報Pro・株探・みんかぶ。規約・robots.txt・接続の拒否）。URL の形だけ
 *              確かめたもの。「未確認」の印を付けて分けて出す（運用者の決定 2026-10-17「確かめられないものも印つきで出す」）
 */

/** 東証の銘柄コード（4桁。2024年からは英字入りもある）。 */
const CODE_RE = /^[0-9A-Z]{4}$/;
const STATES = ['ok', 'ng', 'unchecked'];

/** news_links.json に塊の定義が無いときの並び（research/news_links.py の GROUPS と同じ）。 */
export const DEFAULT_GROUPS = [
  { id: 'news', label: 'ニュース' },
  { id: 'disclosure', label: '適時開示' },
  { id: 'forecast', label: 'アナリスト予想・業績' },
  { id: 'voice', label: '個人投資家の声' },
];

/** 候補の4桁のコード（大文字）。形が違えば null。 */
export function stockCode(candidate) {
  const code = String(candidate?.code ?? '').trim().toUpperCase();
  return CODE_RE.test(code) ? code : null;
}

/** 画面に出してよいリンクの形か（https・状態が分かるもの）。ファイルが壊れていても変なリンクを出さない。 */
function isLink(l) {
  return !!l && typeof l.href === 'string' && /^https:\/\/[^\s"'<>]+$/.test(l.href) && STATES.includes(l.state);
}

function split(rows) {
  const list = (Array.isArray(rows) ? rows : []).filter(isLink);
  return {
    ok: list.filter((l) => l.state === 'ok'),
    failed: list.filter((l) => l.state === 'ng'),
    unchecked: list.filter((l) => l.state === 'unchecked'),
  };
}

/**
 * 候補1件のリンク: { known, checked: [{ id, label, links }]（確かめて開けたものを塊ごと）, failed, unchecked }。
 * known = この銘柄を確かめた記録があるか（無ければリンクを1本も出さない）。
 */
export function stockLinks(data, candidate) {
  const code = stockCode(candidate);
  const rec = code ? data?.stocks?.[code] : null;
  if (!rec || !Array.isArray(rec.links)) return { known: false, checked: [], failed: [], unchecked: [] };
  const groups = Array.isArray(data?.groups) && data.groups.length ? data.groups : DEFAULT_GROUPS;
  const { ok, failed, unchecked } = split(rec.links);
  return {
    known: true,
    checked: groups
      .map((g) => ({ id: g.id, label: g.label, links: ok.filter((l) => l.group === g.id) }))
      .filter((g) => g.links.length > 0),
    failed,
    unchecked,
  };
}

/** 市場全体のリンク: { checked, failed, unchecked }。 */
export function marketLinks(data) {
  const { ok, failed, unchecked } = split(data?.market);
  return { checked: ok, failed, unchecked };
}

/**
 * 確かめた日時と本数（画面の上の帯）と、確かめ方そのものの確かめ（開けたページを、わざと間違えた見出し・ほかの銘柄の
 * コードで判定し直した数 decoyTried と、通ってしまった数 decoyPassed）。ファイルが無ければ null。
 */
export function linkSummary(data) {
  if (!data || typeof data !== 'object') return null;
  const s = data.summary || {};
  const num = (v) => (Number.isFinite(v) ? v : 0);
  return {
    checkedAt: data.checkedAt || null,
    stocks: num(s.stocks),
    ok: num(s.ok),
    ng: num(s.ng),
    unchecked: num(s.unchecked),
    decoyTried: num(s.decoyTried),
    decoyPassed: num(s.decoyPassed),
  };
}

/** 自動では開かないサイトの理由（同じ文は1つに）。 */
export function noAutoReasons(data) {
  const seen = new Set();
  return (Array.isArray(data?.noAuto) ? data.noAuto : [])
    .map((x) => x?.reason)
    .filter((r) => typeof r === 'string' && r && !seen.has(r) && seen.add(r));
}
