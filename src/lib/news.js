/**
 * 候補銘柄のニュース（運用者の依頼 2026-10-17「候補銘柄のニュース として新たにタブを追加してほしいです。左から2番目で」、
 * 同日「取得対象は、候補銘柄のみで良いです…ウソのリンクや無効なリンクがないことのバリデーションに時間をかけてください」）。
 *
 * リンクの URL と確かめた結果は public/data/news_links.json にある（research/news_links.py が毎晩の予測の後に、候補ごとに
 * ページを開いて書く。.github/workflows/news-links.yml）。ここはそれを画面の形に並べるだけで、URL は書き写さない。
 *
 * 画面には確かめの印や状態を出さない（運用者の指示 2026-10-17「確かめた、未確認などの表示はいらないです。バックでやって
 * くれれば良いので」）。リンクの状態（state）による出し分けだけをする:
 *   ok         候補ごとに開いて、その銘柄の、見出しどおりのページだと確かめた → 出す
 *   ng         開いたが確かめを通らなかった（無効なリンク・ウソのリンク）→ 出さない
 *   unchecked  自動では開かないサイト（日経・株予報Pro・株探・みんかぶ。規約・robots.txt・接続の拒否）。URL の形だけ
 *              確かめたもの → ok と同じ塊に並べて出す（同日の決定「全部を印なしで出す」）
 */

/** 東証の銘柄コード（4桁。2024年からは英字入りもある）。 */
const CODE_RE = /^[0-9A-Z]{4}$/;
/** 画面に出す状態（ng は出さない）。 */
const SHOWN = ['ok', 'unchecked'];

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

/** 画面に出すリンクか（https・出す状態のもの）。ファイルが壊れていても変なリンクを出さない。 */
function isShown(l) {
  return !!l && typeof l.href === 'string' && /^https:\/\/[^\s"'<>]+$/.test(l.href) && SHOWN.includes(l.state);
}

/**
 * 候補1件のリンク: { known, groups: [{ id, label, links }] }（出すリンクを見出しの塊ごとに、ファイルの順で。
 * リンクの無い塊は除く）。known = この銘柄のリンクの記録があるか（無ければリンクを1本も出さない）。
 */
export function stockLinks(data, candidate) {
  const code = stockCode(candidate);
  const rec = code ? data?.stocks?.[code] : null;
  if (!rec || !Array.isArray(rec.links)) return { known: false, groups: [] };
  const groups = Array.isArray(data?.groups) && data.groups.length ? data.groups : DEFAULT_GROUPS;
  const shown = rec.links.filter(isShown);
  return {
    known: true,
    groups: groups
      .map((g) => ({ id: g.id, label: g.label, links: shown.filter((l) => l.group === g.id) }))
      .filter((g) => g.links.length > 0),
  };
}
