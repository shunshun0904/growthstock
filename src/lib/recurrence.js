/**
 * リカレンス・ネットワーク（画面の「リカレンス・ネットワーク」タブ。src/views/NetworkView.jsx）の、純粋な計算。
 *
 * データは research/recurrence_net.py が日次予測で作り、Release の recurrence.tar.gz をデプロイのときに dist/data/recurrence/ に
 * 展開する。候補ごとのファイルには、その銘柄の各営業日（ノード）の状態（テクニカル指標9列を銘柄の中で標準化し、1/q σ 刻みの
 * 整数にした値）・結果（+10% の指値か20営業日目の終値・目的変数 1.2σ / 1.5σ）・78週高値の更新日が入っている。
 *
 * 網は k近傍（運用者の選択 2026-10-19「k近傍だけにする」。k は続けて指定「正直なところ、kは1〜10までで良いです」で 1〜10、
 * 既定は指定「初期状態のkは5でお願いします」で 5。版 3 までは ε で結んでいた）。
 * 各日に、タイラー窓（20営業日）より離れた日のうち状態の近い k 日（同じ距離は古い日が先）を選び、どちらかが選べば線で結ぶ。
 * 距離は版 7 から、9列の重なりを差し引くマハラノビス距離（Python が白色化して 1/q 刻みに丸めた座標 zw のユークリッド距離。
 * docs/MODEL_ADOPTION_RULES.md §74）。版 6 までのファイル（zw が無い）は、これまでどおり状態 z のユークリッド距離で数える。
 * 近傍・線・要約は、ここで Python と同じ式で数え直す（画面で k を変えられるように）。距離は整数の2乗で比べるので、浮動小数点の
 * 差は出ない。tests/recurrence.test.js が Python の数えた値（tests/fixtures/recurrence-expected.json）と一致することを確かめる。
 * **参考の表示で、選定には使わない。**
 */

/** 目的変数の鍵と到達しきい値の倍数（research/recurrence_net.py の LABEL_KS と同じ。index.json の method.labels が正本） */
export const DEFAULT_LABELS = [{ key: 'l12', k: 1.2 }, { key: 'l15', k: 1.5 }];

/** 既定値（index.json の method が無いときだけ使う。正本は research/recurrence_net.py） */
export const DEFAULTS = { theiler: 20, k: 5, kChoices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], q: 20, zCap: 6, target: 10, hold: 20 };

const posInt = (v) => Number.isInteger(v) && v > 0;

/** index.json の method から、計算に使う値をそろえる（欠けた値は既定値。版 3 の目次には k が無いので既定値になる）。 */
export function methodOf(index) {
  const m = index?.method || {};
  const kChoices = Array.isArray(m.kChoices) && m.kChoices.length && m.kChoices.every(posInt) ? m.kChoices : DEFAULTS.kChoices;
  return {
    theiler: Number.isInteger(m.theiler) ? m.theiler : DEFAULTS.theiler,
    k: posInt(m.k) && kChoices.includes(m.k) ? m.k : (kChoices.includes(DEFAULTS.k) ? DEFAULTS.k : kChoices[0]),
    kChoices,
    q: Number.isInteger(m.q) ? m.q : DEFAULTS.q,
    zCap: Number.isFinite(m.zCap) ? m.zCap : DEFAULTS.zCap,
    target: Number.isFinite(m.target) ? m.target : DEFAULTS.target,
    hold: Number.isInteger(m.hold) ? m.hold : DEFAULTS.hold,
    labels: Array.isArray(m.labels) && m.labels.length ? m.labels : DEFAULT_LABELS,
    features: Array.isArray(m.features) ? m.features : [],
    // 距離の種類（版 7 から mahalanobis。版 6 までの目次には無いので euclid）。画面の説明の文言を選ぶのに使う
    metric: m.metric === 'mahalanobis' ? 'mahalanobis' : 'euclid',
  };
}

const nanOf = (v) => (v === null || v === undefined ? NaN : Number(v));

/**
 * 候補のファイル → 計算に使う形。dates はノードの日付（start から gaps の暦日を足していく。UTC で数えるので時差に左右されない）。
 */
export function decode(p, labels = DEFAULT_LABELS) {
  const n = p.n;
  const dims = p.features.length;
  const z = Int16Array.from(p.z);
  if (z.length !== n * dims) throw new Error(`状態の数が合いません（${z.length} / ${n} × ${dims}）`);
  // 距離に使う座標（版 7: 白色化した zw = マハラノビス距離。版 6 まで: 状態 z そのまま = ユークリッド距離）
  const zw = Array.isArray(p.zw) ? Int16Array.from(p.zw) : null;
  if (zw && zw.length !== n * dims) throw new Error(`距離の座標の数が合いません（${zw.length} / ${n} × ${dims}）`);
  if (p.gaps.length !== n - 1) throw new Error('日付の数が合いません');
  const ret = Float64Array.from(p.ret, nanOf);
  // 先20営業日の値動き（色付け用）。版 2 の rise / fall は終値で測った別の物差しなので読まない（版 3 から高値・安値）。
  // 無いファイル（版 1）はすべて未確定
  const ver = Number(p.v) || 1;
  const fwd = (k, since) => (ver >= since && Array.isArray(p[k]) && p[k].length === n
    ? Float64Array.from(p[k], nanOf) : new Float64Array(n).fill(NaN));
  const lab = {};
  for (const { key } of labels) lab[key] = Float64Array.from(p[key] || [], nanOf);
  const dates = new Array(n);
  let t = Date.parse(`${p.start}T00:00:00Z`);
  dates[0] = p.start;
  for (let i = 1; i < n; i += 1) {
    t += p.gaps[i - 1] * 86400000;
    dates[i] = new Date(t).toISOString().slice(0, 10);
  }
  const fresh = Int32Array.from(p.fresh || []);
  const isFresh = new Uint8Array(n);
  for (const i of fresh) isFresh[i] = 1;
  return {
    n, dims, q: p.q, z, zd: zw || z, metric: zw ? 'mahalanobis' : 'euclid',
    ret, rise: fwd('rise', 3), fall: fwd('fall', 3), r20: fwd('r20', 2), labels: lab, dates, fresh, isFresh,
    close: Float64Array.from(p.close || [], nanOf),
    scale: p.scale || [], features: p.features, raw: p.raw || [],
    code: p.code, jqCode: p.jqCode, name: p.name, date: p.date,
  };
}

/** 2ノードの距離の2乗（整数。距離に使う座標 zd から）。 */
export function sqDist(dec, i, j) {
  const { zd: z, dims } = dec;
  const a = i * dims;
  const b = j * dims;
  let acc = 0;
  for (let d = 0; d < dims; d += 1) {
    const v = z[a + d] - z[b + d];
    acc += v * v;
  }
  return acc;
}

/** ノードの番号の上限（距離の2乗 × KEY + 番号 を浮動小数点で正確に持つ。距離の2乗は高々 9 × 240² で、積は 2^53 よりずっと小さい） */
const KEY = 8192;

/**
 * ノードごとの近い順の近傍（theiler より離れた日だけ）を kMax 日まで（Python の knn_order と同じ）。並びは（距離の2乗, 番号）の
 * 小さい順で、同じ距離なら古い日（番号の小さい日）が先。各ノードで、いちばん遠い近傍を先頭に置くヒープで kMax 日を残す。
 * 戻り値: { kMax, nbr（n × kMax。足りないところは −1）, d2（同じ形の距離の2乗）, cnt（ノードごとの近傍の数）}
 */
export function knnLists(dec, theiler, kMax) {
  const { n, dims, zd: z } = dec;
  if (n > KEY) throw new Error(`ノードが多すぎます（${n} > ${KEY}）`);
  const nbr = new Int32Array(n * kMax).fill(-1);
  const d2 = new Int32Array(n * kMax).fill(-1);
  const cnt = new Int32Array(n);
  const heap = new Float64Array(Math.max(1, kMax));
  for (let i = 0; i < n; i += 1) {
    let size = 0;
    // いま残している中でいちばん遠い近傍の距離の2乗（kMax 日たまるまでは打ち切らない）。足していく途中でこれを超えたら、
    // その日は入らない（同じ距離なら番号で比べるので、超えたときだけ打ち切る）
    let worst = kMax > 0 ? Infinity : -1;
    const a = i * dims;
    for (let j = 0; j < n; j += 1) {
      if (j >= i - theiler && j <= i + theiler) { j = i + theiler; continue; }   // タイラー窓の中は選ばない
      const b = j * dims;
      let acc = 0;
      let d = 0;
      for (; d < dims; d += 1) {
        const v = z[a + d] - z[b + d];
        acc += v * v;
        if (acc > worst) break;
      }
      if (d < dims) continue;
      const key = acc * KEY + j;
      if (size < kMax) {
        let c = size;
        heap[c] = key;
        size += 1;
        while (c > 0) {
          const p = (c - 1) >> 1;
          if (heap[p] >= heap[c]) break;
          const t = heap[p]; heap[p] = heap[c]; heap[c] = t;
          c = p;
        }
        if (size === kMax) worst = Math.floor(heap[0] / KEY);
      } else if (key < heap[0]) {
        heap[0] = key;
        let c = 0;
        for (;;) {
          const l = 2 * c + 1;
          const r = l + 1;
          let m = c;
          if (l < size && heap[l] > heap[m]) m = l;
          if (r < size && heap[r] > heap[m]) m = r;
          if (m === c) break;
          const t = heap[m]; heap[m] = heap[c]; heap[c] = t;
          c = m;
        }
        worst = Math.floor(heap[0] / KEY);
      }
    }
    const got = heap.slice(0, size).sort();
    cnt[i] = size;
    for (let r = 0; r < size; r += 1) {
      const j = got[r] % KEY;
      nbr[i * kMax + r] = j;
      d2[i * kMax + r] = (got[r] - j) / KEY;
    }
  }
  return { kMax, nbr, d2, cnt };
}

/**
 * k近傍の網（Python の knn_adjacency と同じ）。どちらか一方が相手を近い k 日に選んでいれば結ぶ（i < j の組で1本）。lists は
 * knnLists の戻り値（k 以上の kMax で数えたもの。無ければここで数える）。次数は k 以上になりうる（多くの日に選ばれる日ほど大きい）。
 * 戻り値: { k, theiler, lists, nEdges, edges（[i0, j0, i1, j1, …]。i < j で小さい順）, deg }
 */
export function network(dec, theiler, k, lists = null) {
  const { n } = dec;
  const L = lists && lists.kMax >= k ? lists : knnLists(dec, theiler, k);
  const take = (i) => Math.min(k, L.cnt[i]);
  let total = 0;
  for (let i = 0; i < n; i += 1) total += take(i);
  const pairs = new Float64Array(total);
  let t = 0;
  for (let i = 0; i < n; i += 1) {
    for (let r = 0, c = take(i); r < c; r += 1) {
      const j = L.nbr[i * L.kMax + r];
      pairs[t] = i < j ? i * n + j : j * n + i;
      t += 1;
    }
  }
  pairs.sort();
  let m = 0;
  for (let x = 0; x < total; x += 1) {
    if (x === 0 || pairs[x] !== pairs[x - 1]) { pairs[m] = pairs[x]; m += 1; }
  }
  const edges = new Uint32Array(2 * m);
  const deg = new Int32Array(n);
  for (let e = 0; e < m; e += 1) {
    const a = Math.floor(pairs[e] / n);
    const b = pairs[e] - a * n;
    edges[2 * e] = a;
    edges[2 * e + 1] = b;
    deg[a] += 1;
    deg[b] += 1;
  }
  return { k, theiler, lists: L, nEdges: m, edges, deg };
}

/**
 * ノードごとの似た日（その日に近い k 日。近い順）。触れた日・選んだ日・今日の似た日を光らせるのに使う。線（どちらかが選べば
 * 結ぶ）とは違い、その日が選んだ k 日だけ。
 */
export function neighborsOf(net) {
  const L = net.lists;
  return { of: (i) => L.nbr.subarray(i * L.kMax, i * L.kMax + Math.min(net.k, L.cnt[i])) };
}

const median = (v) => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};

/**
 * ノードの集まり idx の結果（Python の outcome_stats と同じ）。hit は +10% の指値に届いた割合（ret ≥ target）。
 * 割合・中央値は分かっている日だけで数える（無ければ null）。
 */
export function outcomeStats(dec, idx, target = DEFAULTS.target, labels = DEFAULT_LABELS) {
  const r = [];
  for (const i of idx) if (Number.isFinite(dec.ret[i])) r.push(dec.ret[i]);
  const out = {
    n: idx.length,
    known: r.length,
    hit: r.length ? r.filter((x) => x >= target - 1e-9).length / r.length : null,
    retMed: median(r),
  };
  for (const { key } of labels) {
    const lab = dec.labels[key];
    let cnt = 0;
    let pos = 0;
    for (const i of idx) {
      const v = lab ? lab[i] : NaN;
      if (Number.isFinite(v)) {
        cnt += 1;
        pos += v;
      }
    }
    out[key] = cnt ? pos / cnt : null;
    out[`${key}N`] = cnt;
  }
  return out;
}

/** ノードの集まりが、theiler より離れたところで区切るといくつの時期（塊）になるか（Python の periods_of と同じ。空なら 0） */
export function periodsOf(idx, theiler) {
  const v = [...idx].sort((a, b) => a - b);
  if (!v.length) return 0;
  let p = 1;
  for (let x = 1; x < v.length; x += 1) if (v[x] - v[x - 1] > theiler) p += 1;
  return p;
}

/**
 * 要約（Python の summarize と同じ項目・同じ式）。今日 = 最後のノード。似た日は今日の近い k 日（近い順）。
 *   kth / kthPct … k 番目に近い日までの距離（σ）と、どのノードも自分の k 番目までの距離で比べたときの今日の位置（中位の順位の
 *                  百分位。高いほど似た日の少ない珍しい状態）
 *   periods      … 似た日がいくつの時期に固まっているか（20営業日より離れたところで区切る）
 * 似た日の一覧は todayNeighbors（図で光らせる日と同じ）。版 4 までの「今日に近い過去の更新日」（更新日だけから近い順に5日。
 * 近い k 日の外の日も入った）は、運用者の指定 2026-10-19「プロット上の近傍の日付で統一してほしい」でやめた
 */
export function summarize(dec, net, theiler, k, { target = DEFAULTS.target, labels = DEFAULT_LABELS } = {}) {
  const { n, q } = dec;
  const today = n - 1;
  const L = net.lists;
  const take = (i) => Math.min(k, L.cnt[i]);
  const nb = n ? Array.from(L.nbr.subarray(today * L.kMax, today * L.kMax + take(today))) : [];
  const kth2 = nb.length ? L.d2[today * L.kMax + nb.length - 1] : -1;
  let below = 0;
  let same = 0;
  let valid = 0;
  for (let i = 0; i < n; i += 1) {
    const c = take(i);
    if (!c) continue;
    valid += 1;
    const v = L.d2[i * L.kMax + c - 1];
    if (v < kth2) below += 1;
    else if (v === kth2) same += 1;
  }
  const nbFresh = nb.filter((i) => dec.isFresh[i]);
  const all = Array.from({ length: n }, (_, i) => i);
  let isolated = 0;
  for (let i = 0; i < n; i += 1) if (net.deg[i] === 0) isolated += 1;
  return {
    k, theiler, edges: net.nEdges, isolated, degMed: median(Array.from(net.deg)),
    deg: nb.length, kth2, kth: kth2 >= 0 ? Math.sqrt(kth2) / q : null,
    kthPct: nb.length && valid ? (100 * (below + 0.5 * same)) / valid : null,
    periods: periodsOf(nb, theiler),
    nbr: outcomeStats(dec, nb, target, labels),
    base: outcomeStats(dec, all, target, labels),
    nbrFresh: outcomeStats(dec, nbFresh, target, labels),
    baseFresh: outcomeStats(dec, [...dec.fresh], target, labels),
  };
}

/** 今日の似た日（近い k 日）の一覧（画面の「今日に似た日」の表。図で光らせる日と同じ）。近い順（同じ距離は古い日が先）。 */
export function todayNeighbors(dec, net) {
  const today = dec.n - 1;
  const L = net.lists;
  const out = [];
  for (let r = 0, c = Math.min(net.k, L.cnt[today]); r < c; r += 1) {
    out.push({ i: L.nbr[today * L.kMax + r], d: Math.sqrt(L.d2[today * L.kMax + r]) / dec.q });
  }
  return out;
}

// --------------------------------------------------------------------------- //
// 色。どの色の付け方も、低い・古い・届かない = 青、高い・新しい・届いた = 赤（運用者の指定 2026-10-19「青白ではなく、青白赤で」
// → 選択「高い・新しい方を赤」、2026-10-20「揃えたいです」→ 選択「目的変数も（正例 = 赤）」）。暗い面（#0a0e17）で dataviz の
// validate_palette.js の式で確かめた値をそれぞれに書く
// --------------------------------------------------------------------------- //

/**
 * 値で色分けするときの未確定（先20営業日がまだ無い日）。下の段（結果・最高値まで・20営業日目）のどの色とも、通常の見え方で
 * ΔE 31.8 以上・色覚の3型で 18.2 以上離れる（いちばん近いのは青 #3a7bf0 と赤 #e3463f）。面との差 1.70:1 と暗いので、凡例の件数を添える
 */
export const NA_VALUE = '#3a3a3a';

/**
 * 青白赤の5色（低い・古い = 青 → 白 → 高い・新しい = 赤）。運用者の指定（2026-10-19）「リカレンスプロット内の色のコンターは
 * 青白ではなく、青白赤で」→ 選択（AskUserQuestion）: 対象「最高値までの上昇率」「時期」・向き「高い・新しい方を赤」。
 * 暗い面 #0a0e17 で dataviz の validate_palette.js の式で確かめた: 青の腕・赤の腕はそれぞれ順序の段の検査を通る（明度が単調・
 * 隣の段の明度差 0.06 以上・1色相）。白からの明度差は両腕でほぼ同じ（0.19 / 0.17 と 0.34 / 0.33）。隣どうしは通常の見え方で
 * ΔE 17.0 以上・色覚の1型2型で 15.0 以上。どの色も面との差 4.79:1 以上
 */
export const BWR = ['#3a7bf0', '#86b0f2', '#ececec', '#f2988e', '#e3463f'];

/**
 * 結果（+10% の指値か20営業日目の終値）の4区分。青白赤の両端と白の隣の4色で、0% のところが白の境目（−10% 未満 = 青・
 * −10〜0% = 薄い青・0〜+10% = 薄い赤・+10% 到達 = 赤。2026-10-20 に向きをそろえた。前は +10% 到達が明るい青・下げが赤）。
 * 未確定を含めて全組で通常の見え方 ΔE 17.0 以上・色覚の3型で 12.8 以上（いちばん近いのは青と薄い青）
 */
export const OUTCOME_CLASSES = [
  { id: 'hit', label: '+10% 到達', color: BWR[4] },
  { id: 'up', label: '0〜+10%', color: BWR[3] },
  { id: 'down', label: '−10〜0%', color: BWR[1] },
  { id: 'bad', label: '−10% 未満', color: BWR[0] },
  { id: 'na', label: '未確定', color: NA_VALUE },
];

/** 収益（%）の区分。+10 以上 = 指値に届いた。 */
export function outcomeOf(ret, target = DEFAULTS.target) {
  if (!Number.isFinite(ret)) return 'na';
  if (ret >= target - 1e-9) return 'hit';
  if (ret >= 0) return 'up';
  if (ret >= -10) return 'down';
  return 'bad';
}

/**
 * 目的変数の区分。正例 = 赤（2026-10-20 の選択「目的変数も（正例 = 赤）」。前は明るい青）・負例 = 灰。全組で通常の見え方
 * ΔE 18.3 以上・色覚の3型で 11.0 以上
 */
export const LABEL_CLASSES = [
  { id: 'pos', label: '正例', color: BWR[4] },
  { id: 'neg', label: '負例', color: '#64748b' },
  { id: 'na', label: '未確定', color: '#334155' },
];

export function labelClassOf(v) {
  if (!Number.isFinite(v)) return 'na';
  return v >= 0.5 ? 'pos' : 'neg';
}

/**
 * 最高値までの上昇率（%）の5段（運用者の選択 2026-10-19「5段＋未確定」）。最高到達価格（t+1〜t+20 の高値の最大）÷ 翌営業日の寄り
 * − 1（運用者の指定「その日から最高到達価格（20日以内最高高値）までの上昇率」・選択「基準は翌営業日の始値」。10% 以上なら +10% の
 * 指値に届いた）。色は青白赤（BWR）: 5〜10% が白、それより低いほど青、+10% の指値に届いた段（10〜20%・20% 以上）が赤。
 * 0% 未満は高値で測るのでほぼ出ない（買値の日の高値は寄り以上）。並びは高い順（凡例も同じ）
 */
export const RISE_CLASSES = [
  { id: 'r20p', label: '20% 以上', color: BWR[4] },
  { id: 'r20', label: '10〜20%', color: BWR[3] },
  { id: 'r10', label: '5〜10%', color: BWR[2] },
  { id: 'r5', label: '0〜5%', color: BWR[1] },
  { id: 'r0', label: '0% 未満', color: BWR[0] },
  { id: 'na', label: '未確定', color: NA_VALUE },
];

/** 最高値までの上昇率（%）の段。下の端を含む（5.0 は「5〜10%」。値は 0.1% 刻みに切り下げてあるので 10 以上 = 指値に届いた） */
export function riseOf(v) {
  if (!Number.isFinite(v)) return 'na';
  if (v >= 20) return 'r20p';
  if (v >= 10) return 'r20';
  if (v >= 5) return 'r10';
  if (v >= 0) return 'r5';
  return 'r0';
}

/**
 * 20営業日目の騰落率（%）の4段。20営業日目の終値 ÷ 翌営業日の寄り − 1（指値なしで持ち切り）。区切り（+10・0・−10）と色は
 * 結果（OUTCOME_CLASSES）と同じで、違うのは +10% の指値で売らないことだけ
 */
export const R20_CLASSES = [
  { id: 'hit', label: '+10% 以上', color: BWR[4] },
  { id: 'up', label: '0〜+10%', color: BWR[3] },
  { id: 'down', label: '−10〜0%', color: BWR[1] },
  { id: 'bad', label: '−10% 未満', color: BWR[0] },
  { id: 'na', label: '未確定', color: NA_VALUE },
];

/** 時期の色（古い → 新しい。青白赤 BWR: 古い = 青・真ん中の時期 = 白・新しい = 赤。運用者の選択 2026-10-19。前は青 → 白） */
export const TIME_STOPS = BWR;

const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));

/** 0〜1 の位置の時期の色（隣り合う段の間を線形に混ぜる）。 */
export function timeColor(f) {
  const x = Math.min(1, Math.max(0, f)) * (TIME_STOPS.length - 1);
  const k = Math.min(TIME_STOPS.length - 2, Math.floor(x));
  const a = hex(TIME_STOPS[k]);
  const b = hex(TIME_STOPS[k + 1]);
  const u = x - k;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * u));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/**
 * 色の付け方。outcome = 結果、rise / r20 = 翌営業日の寄り（買値）から見た先20営業日の値動き（運用者の依頼 2026-10-19
 * 「ノードの色付け機能がほしい。先20日間の最大上昇率は最低限」→「最大上昇率というより、その日から最高到達価格（20日以内最高
 * 高値）までの上昇率」。選択「5段＋未確定」「基準は翌営業日の始値」「最安値までの下落率・20営業日目も同じ基準にそろえる」。
 * そのあと「安値の色付けは、なしで良いです」で最安値までの色を外した（値はツールチップと選んだ日に出す））、
 * l15 / l12 = 目的変数、time = 時期。value があるものは、ノードの値を表やツールチップに出す（short は表の見出し）
 */
export const COLOR_MODES = [
  { id: 'outcome', label: '結果（+10% / 20営業日目）',
    title: '翌営業日の寄りで買い、+10% の指値か20営業日目の終値で売ったときの収益' },
  { id: 'rise', label: '最高値までの上昇率（20日以内）', short: '最高値まで', value: 'rise',
    title: '20営業日以内の最高値（高値の最大）÷ 翌営業日の寄り − 1（10% 以上なら +10% の指値に届いた）' },
  { id: 'r20', label: '20営業日目の騰落率', short: '20日目', value: 'r20',
    title: '20営業日目の終値 ÷ 翌営業日の寄り − 1（指値なしで持ち切ったとき。20営業日目に売買が無かった日は未確定）' },
  { id: 'l15', label: '目的変数 1.5σ' },
  { id: 'l12', label: '目的変数 1.2σ' },
  { id: 'time', label: '時期' },
];

/** 区分で塗る色の付け方: 区分の並び（凡例）と、ノード i の区分。time は区分なし（連続の色） */
const BINS = {
  outcome: { classes: OUTCOME_CLASSES, of: (dec, i, target) => outcomeOf(dec.ret[i], target) },
  rise: { classes: RISE_CLASSES, of: (dec, i) => riseOf(dec.rise[i]) },
  r20: { classes: R20_CLASSES, of: (dec, i, target) => outcomeOf(dec.r20[i], target) },
};
const binOf = (mode) => BINS[mode]
  || (mode === 'time' ? null : { classes: LABEL_CLASSES, of: (dec, i) => labelClassOf(dec.labels[mode]?.[i]) });

/** 色の付け方の区分の並び（凡例に出す順）。区分の無い付け方（時期）は null */
export function classesOf(mode) {
  return binOf(mode)?.classes ?? null;
}

/** ノードごとの色（文字列の配列）。 */
export function nodeColors(dec, mode, target = DEFAULTS.target) {
  const out = new Array(dec.n);
  const bin = binOf(mode);
  const color = bin ? Object.fromEntries(bin.classes.map((c) => [c.id, c.color])) : null;
  for (let i = 0; i < dec.n; i += 1) {
    out[i] = bin ? color[bin.of(dec, i, target)] : timeColor(dec.n > 1 ? i / (dec.n - 1) : 1);
  }
  return out;
}

/** 色の区分ごとの件数（凡例に出す）。idx を渡すとそのノードだけ数える（今日に似た日の内訳）。区分の無い付け方は {} */
export function classCounts(dec, mode, target = DEFAULTS.target, idx = null) {
  const counts = {};
  const bin = binOf(mode);
  if (!bin) return counts;
  const add = (i) => { const c = bin.of(dec, i, target); counts[c] = (counts[c] || 0) + 1; };
  if (idx) for (const i of idx) add(i);
  else for (let i = 0; i < dec.n; i += 1) add(i);
  return counts;
}

// --------------------------------------------------------------------------- //
// 状態の値（丸めた z → 生の値の目安）
// --------------------------------------------------------------------------- //

/**
 * 丸めた z（整数）から、生の値の目安へ戻す（標準化を戻し、log1p は expm1・asinh は sinh × s で戻す）。切った値（上下の分位点）より
 * 外は分からないので、戻した値は切った値の中に収まる。丸め（1/q σ）のぶんの誤差がある。
 */
export function rawOf(param, zq, q = DEFAULTS.q) {
  if (!param) return NaN;
  const v = (zq / q) * param.std + param.mean;
  let x = v;
  if (param.kind === 'pos_heavy') x = Math.expm1(v);
  else if (param.kind === 'signed_heavy') x = Math.sinh(v) * (param.s || 1);
  if (param.lo !== null && param.lo !== undefined) x = Math.max(param.lo, x);
  if (param.hi !== null && param.hi !== undefined) x = Math.min(param.hi, x);
  return x;
}

/** ノード i の9列の z（σ 単位）。 */
export function zOf(dec, i) {
  const out = [];
  for (let d = 0; d < dec.dims; d += 1) out.push(dec.z[i * dec.dims + d] / dec.q);
  return out;
}
