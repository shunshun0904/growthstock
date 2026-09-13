/**
 * 決算の損益を「流れ」に変換する。描画からは独立した純粋関数。
 *
 * 何が作れて、何が作れないか
 * ------------------------
 * J-Quants の決算短信サマリーにあるのは 売上高 / 営業利益 / 経常利益 /
 * 当期純利益 だけで、**売上原価と販管費は入っていない**。
 * よくある「売上 → 原価 ＋ 粗利 → 販管費 → 営業利益」の形は作れない。
 *
 * 作れるのは段階利益の流れ。
 *
 *   売上高 ─┬→ 営業費用（原価＋販管費）   = 売上高 − 営業利益
 *           └→ 営業利益 ─┬→ 営業外費用    = 営業利益 − 経常利益
 *                        └→ 経常利益 ─┬→ 法人税等・特別損失
 *                                      └→ 当期純利益
 *
 * 「営業費用」は原価と販管費を足したものを1本にまとめた値で、引き算で
 * 出しているだけ。内訳は元データに無いので割らない（割ると捏造になる）。
 *
 * 段階が増える向きに動くこともある（営業外収益・特別利益）。そのときは
 * 下から合流する流入として扱う。
 *
 * 損失の扱い
 * ---------
 * サンキーは負の流れを描けない。段階利益が 0 以下になったら、そこで幹を
 * 止めて損失として示し、それ以降の段階は描かない。無理に描くと、
 * 実際には無い流れを見せることになる。
 */

/** 表示モード。元データにある値だけを使う。 */
export const MODES = [
  { id: 'q', label: '単四半期', note: 'その四半期だけの実績' },
  { id: 'cum', label: '累計', note: '期首からの累計実績' },
  { id: 'forecast', label: '会社予想', note: '会社が出した通期の予想' },
];

const FIELDS = {
  q: ['qNetSales', 'qOperatingProfit', 'qOrdinaryProfit', 'qProfit'],
  cum: ['cumNetSales', 'cumOperatingProfit', 'cumOrdinaryProfit', 'cumProfit'],
  forecast: ['forecastNetSales', 'forecastOperatingProfit',
             'forecastOrdinaryProfit', 'forecastProfit'],
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** 直近の決算（開示日が最も新しいもの）。無ければ null。 */
export function latestQuarter(quarters) {
  const rows = (quarters || []).filter((q) => q && q.disclosedDate);
  if (!rows.length) return null;
  return rows.reduce((a, b) => (a.disclosedDate >= b.disclosedDate ? a : b));
}

/**
 * 1つの決算を段階の流れにする。
 *
 * 返すもの
 *   sales     売上高（幹の太さの基準）
 *   steps     段階ごとの {key, label, value, out, in} の並び
 *   truncated 損失で幹が途切れた段階の key（途切れていなければ null）
 *   missing   元データに無くて飛ばした段階の key の配列
 */
export function toFlow(quarter, mode = 'q') {
  if (!quarter) return null;
  const [fSales, fOp, fOdp, fNp] = FIELDS[mode] || FIELDS.q;
  const sales = quarter[fSales];
  if (!isNum(sales) || sales <= 0) return null;

  // 経常利益は元データで欠けることがある（開示形式による）。
  // 欠けていたら段階を飛ばし、営業利益から直接 当期純利益へ繋ぐ。
  const raw = [
    { key: 'op', label: '営業利益', value: quarter[fOp],
      costLabel: '営業費用（原価＋販管費）' },
    { key: 'odp', label: '経常利益', value: quarter[fOdp],
      costLabel: '営業外費用', gainLabel: '営業外収益' },
    { key: 'np', label: '当期純利益', value: quarter[fNp],
      costLabel: '法人税等・特別損失', gainLabel: '特別利益' },
  ];
  const missing = raw.filter((s) => !isNum(s.value)).map((s) => s.key);
  const kept = raw.filter((s) => isNum(s.value));
  if (!kept.length) return null;
  // 経常を飛ばしたときは、当期純利益までの差を1本にまとめて名前も変える
  if (missing.includes('odp')) {
    const np = kept.find((s) => s.key === 'np');
    if (np) np.costLabel = '営業外・特別損益・法人税等';
  }

  const steps = [];
  let prev = sales;
  let truncated = null;
  for (const s of kept) {
    if (truncated) break;
    if (s.value <= 0) {
      // ここで幹が尽きる。前段の値がまるごと消える形で示し、以降は描かない
      steps.push({
        key: s.key, label: s.label, value: s.value, loss: true,
        out: { label: s.costLabel, value: prev }, in: null,
      });
      truncated = s.key;
      break;
    }
    const d = prev - s.value;
    steps.push({
      key: s.key, label: s.label, value: s.value, loss: false,
      out: d > 0 ? { label: s.costLabel, value: d } : null,
      in: d < 0 ? { label: s.gainLabel || '収益', value: -d } : null,
    });
    prev = s.value;
  }
  return { sales, steps, truncated, missing, mode };
}

/**
 * 金額の表示。決算は桁が大きいので億・兆に丸める。
 * 損失は会計の慣行どおり △ を付ける。色だけに頼らないための副次符号でもある
 * （緑と赤は2型色覚で ΔE 6.5 しか離れない。実測値）。
 */
export function fmtYen(v) {
  if (!isNum(v)) return '—';
  const a = Math.abs(v);
  const sign = v < 0 ? '△' : '';
  if (a >= 1e12) return `${sign}${(a / 1e12).toFixed(2)}兆円`;
  if (a >= 1e8) return `${sign}${(a / 1e8).toFixed(a >= 1e10 ? 0 : 1)}億円`;
  if (a >= 1e4) return `${sign}${Math.round(a / 1e4).toLocaleString('ja-JP')}万円`;
  return `${sign}${Math.round(a).toLocaleString('ja-JP')}円`;
}

/** 売上高に対する構成比。 */
export function share(value, sales) {
  if (!isNum(value) || !isNum(sales) || sales <= 0) return null;
  return (value / sales) * 100;
}
