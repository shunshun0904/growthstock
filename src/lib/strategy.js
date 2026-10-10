/**
 * 運用の戦略（docs/PLAYBOOK.md）を画面で再現する。
 *
 * 実測（out-of-fold 2021-11〜2026-08、14,966件）から決めた手順:
 *
 *   1. その日の発火数（78週高値を更新した銘柄数）で、その日を取るか決める
 *        20件以上 … 本命。持ち切り +3.27%、勝率63%
 *         8〜19件 … 枠が空いていれば取る（+1.0〜2.2%、勝率54〜55%）
 *         7件以下 … 見送る（142件で −0.11%、−10%割れが 14.1%）
 *      7件以下と8件以上の差は 2.44pt / z≈2.6 で、ここだけは線が実在する。
 *      **20件未満を全部見送るのは誤り。** 空き枠の収益は 0% なので、
 *      +2% の帯を捨てて枠を遊ばせると損になる（実験35）
 *   2. ブースティング3モデル（LightGBM / XGBoost / CatBoost）**すべて**が
 *      過去分布の上位10%（90パーセンタイル以上）
 *   3. 通った銘柄を「3モデルの最小順位」で並べて上位1〜2件
 *        lgbm 単体の順位で並べると2番手が +0.87% まで落ちる（最小順位なら +2.99%）
 *   4. 翌営業日の寄りで買う
 *   5. +20% の指値、届かなければ 20営業日で手仕舞い
 *   6. 同時に持つのは 3銘柄まで。枠が埋まっていたら**見送る**
 *        保有中の銘柄は、より良い候補が出ても切らない。実測（実験34）で
 *        乗り換え37回のうち24回（65%）は切らないほうが良く、切った銘柄が
 *        そのまま持っていれば得ていた分は平均 +5.69%、実現は +3.20%。
 *        「含み損でなければ」という条件は、**勝っている玉しか切れない**
 *        ことを意味するため、構造的に勝ち玉を刈ることになる
 *
 * ここは純粋な変換だけを置く（単体テストできるように）。表示は
 * views/PredictionView.jsx。
 */

/** ブースティング3モデル。合議の対象はこの3つだけ（線形・MLP は入れない）。 */
export const BOOST = ['lgbm', 'xgb', 'cat'];

/** 画面に出す短い呼び名。正本は research/models.py。 */
export const MODEL_JA = { lgbm: 'LightGBM', xgb: 'XGBoost', cat: 'CatBoost' };

/**
 * 決算の寄与（水準＋変化）。正本は research/feature_dict.py の MACRO。
 *
 * 決算は「モデルの外の材料」ではない。本番153列のうち118列（77%）が
 * 決算由来なので、候補を見て別途決算を確かめるのは、モデルが既に読んだ
 * ものを読み直すことになる。どちらに効いているかを画面に出す。
 */
export const FUND_GROUPS = ['決算（水準）', '決算（変化）'];

export function fundContrib(candidate) {
  const g = candidate?.contrib?.groups;
  if (!g) return null;
  const v = FUND_GROUPS.map((k) => g[k]).filter((x) => Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) : null;
}

export const STRATEGY = {
  agreePct: 90,      // 3モデルすべてがこの百分位以上
  strongBreaks: 20,  // この件数以上の発火なら本命
  skipBreaks: 8,     // これ未満の発火なら見送る
  topK: 2,           // 買うのは上位何件まで
  takeProfit: 20,    // 利確の指値（%）
  holdDays: 20,      // 届かなければ何営業日で手仕舞いするか
  maxSlots: 3,       // 同時に持てる銘柄数。埋まっていたら見送る
  nearLo: 85,        // 「惜しい候補」として画面に出す下限
  frozenVol: 0.3,    // 直近20日の日次ボラ（%）がこれ未満なら「値動きが止まっている」注意を出す（TOB 中の可能性。実験63）
};

/**
 * 「値動きが止まっている」注意。直近20日の日次ボラ（candidate.vol20d、%）が s.frozenVol 未満なら文言を返す。
 *
 * TOB の公表後は株価が買付価格に張り付き、日々の値動きがほぼ 0 になる。そのまま「78週高値の更新」として
 * 候補に入り、5モデルは「静かな高値更新」を最高評価するので最上位に来る。実験63（docs/MODEL_SELECTION_EDA.md §8）で
 * 全5モデル 95以上を満たした 97件のうち 20日ボラ 0.3% 未満は 2件あり、どちらも TOB 中で外れた（上がりようがない）。
 * 値が無ければ null（注意を出さない）。選定の規則（strategySignal）は変えない。注意だけ出す。
 */
export function frozenNote(candidate, s = STRATEGY) {
  const raw = candidate?.vol20d;
  if (raw === null || raw === undefined || raw === '') return null;   // 値が無い（Number(null) は 0 になるので先に弾く）
  const v = Number(raw);
  if (!Number.isFinite(v) || v >= s.frozenVol) return null;
  return `値動きなし（20日ボラ ${v.toFixed(2)}%。TOB 中の可能性）`;
}

/**
 * 空いている枠の数。手で入れた保有数から出す。
 *
 * 保有数は玉の本数なので整数のはず。小数や負の値は入力の誤りなので
 * 黙って丸めず null を返す（画面は枠の話をしない）。
 */
export function freeSlots(held, s = STRATEGY) {
  if (!Number.isInteger(held) || held < 0) return null;
  return Math.max(0, s.maxSlots - held);
}

/**
 * 3モデルの百分位。1つでも欠けていれば null を混ぜて返す
 * （欠けたまま「満たした」と判定しないため）。
 */
export function boostPcts(candidate) {
  const per = candidate?.byModel;
  if (!per) return null;
  const out = {};
  for (const a of BOOST) {
    const v = per[a]?.pctHistorical;
    out[a] = Number.isFinite(v) ? v : null;
  }
  return out;
}

/** 3モデルの最小の百分位。1つでも欠けていれば null。 */
export function minPct(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  return vals.some((v) => v === null) ? null : Math.min(...vals);
}

/** 3モデルの百分位の幅（最大−最小）。1つでも欠けていれば null。 */
export function pctSpread(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return Math.max(...vals) - Math.min(...vals);
}

/** 3モデルのうち、いちばん低い評価を出しているモデル。 */
export function laggard(candidate) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  return BOOST[vals.indexOf(Math.min(...vals))];
}

/**
 * 運用者の線と「際どい」範囲（運用者の決定 2026-10-10）。
 *
 * 運用者は「3モデル（lgbm・xgb・cat）がすべて 95以上」を買いの目安にしている。上の戦略パネルの線
 * （STRATEGY.agreePct = 90）とは別。その線の下で「際どい」のは次の全部:
 *   - 2モデルが 95以上で、残り1つが 95未満
 *   - 3モデルの最小が 85〜95
 * 例: xgb 96・cat 98・lgbm 85。
 */
export const BORDER = { line: 95, lo: 85 };

/**
 * 際どい候補の過去の成績（実験70。本番の239列の OOF 2021-11〜2026-09、docs/PLAYBOOK.md「実験70」）。
 * 物差しは +10% の指値（20営業日以内に届けば +10%、届かなければ持ち切り）の平均。
 * 形は「2つ95以上（two95）」と「95以上は1つ以下・最小 85〜95（min）」、それぞれ一番低いモデルが
 * lgbm か、xgb / cat か。
 *   two95: lgbm が一番低い 145件は約 0.0%、xgb / cat が一番低い 420件は +1.83%（差 +1.8pt ± 0.9、z≈2.0）
 *   min:   lgbm 269件 +1.19%、xgb / cat 554件 +1.06%（差 +0.13pt ± 0.60）→ この形では差が無い
 * 「lgbm が一番低いと弱い」は 2つ95以上の形でだけ出た（実験36 は 153列のころの 85〜90 の帯で同じ向き、
 * いまのモデルの 85〜90 の帯では出ない）。だから注意（warn）は two95 の形でだけ出す。
 */
export const BORDER_STATS = {
  line: { n: 616, ret: 2.66 },                     // 3つとも95以上（線の上）
  two95: { lgbm: { n: 145, ret: -0.01 }, other: { n: 420, ret: 1.83 } },
  min: { all: { n: 823, ret: 1.10 }, lgbm: { n: 269, ret: 1.19 }, other: { n: 554, ret: 1.06 } },
};

/**
 * 際どい候補の形。際どくなければ null。3モデルの百分位が1つでも欠けていれば null。
 *   two95_hi / two95_mid / two95_lo  2つが95以上、残り1つが 90〜95 / 85〜90 / 85未満
 *   min90 / min85                    95以上は1つ以下で、最小が 90〜95 / 85〜90
 */
export function borderShape(candidate, b = BORDER) {
  const p = boostPcts(candidate);
  if (!p) return null;
  const vals = BOOST.map((a) => p[a]);
  if (vals.some((v) => v === null)) return null;
  const n95 = vals.filter((v) => v >= b.line).length;
  const mn = Math.min(...vals);
  if (n95 === 3) return null;
  if (n95 === 2) return mn >= 90 ? 'two95_hi' : mn >= b.lo ? 'two95_mid' : 'two95_lo';
  if (mn >= 90) return 'min90';
  if (mn >= b.lo) return 'min85';
  return null;
}

/** 収益の表示。小数1桁に丸めてから符号を付ける（−0.01 を「−0.0%」と出さない）。 */
const fmtRet = (v) => {
  const r = Math.round(v * 10) / 10;
  return r === 0 ? '0.0%' : `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}%`;
};

/** 「+1.8%（420件）」。数字が無ければ「（測っていない）」。 */
const past = (x) => (Number.isFinite(x?.ret) && x?.n ? `${fmtRet(x.ret)}（${x.n}件）` : '（測っていない）');

/**
 * 際どい候補への注意。一番低いモデルと、その形の過去の成績を添える。際どくなければ null。
 *   warn   2つが95以上で、一番低いのが LightGBM（過去ほぼ 0%）。ほかの形では出さない（差が無い）
 *   badge  行に出す短い文言
 *   text   詳しい文言（行の title と、開いたときの本文）
 * 選定の規則（strategySignal）は変えない。注意だけ出す。
 */
export function borderNote(candidate, b = BORDER, st = BORDER_STATS) {
  const shape = borderShape(candidate, b);
  if (!shape) return null;
  const lag = laggard(candidate);
  const p = boostPcts(candidate);
  const name = MODEL_JA[lag] || lag;
  const head = `${name} が一番低い（${p[lag].toFixed(1)}）。`;
  const lineTxt = `3つとも${b.line}以上は ${past(st.line)}`;
  const src = '（実験70。+10% の指値で売り、届かなければ20営業日で手仕舞いしたときの平均）';
  let warn = false;
  let text;
  if (shape.startsWith('two95')) {
    warn = lag === 'lgbm';
    text = warn
      ? `${head}2つが${b.line}以上でも、LightGBM が一番低い形は過去ほぼ 0%`
        + `（${fmtRet(st.two95.lgbm.ret)}・${st.two95.lgbm.n}件）。`
        + `XGBoost / CatBoost が一番低いなら ${past(st.two95.other)}、${lineTxt}。`
        + `logit が90以上かは手がかりにならない${src}`
      : `${head}2つが${b.line}以上で XGBoost / CatBoost が一番低い形は過去 ${past(st.two95.other)}。`
        + `LightGBM が一番低い形は ${past(st.two95.lgbm)}、${lineTxt}${src}`;
  } else {
    text = `${head}${b.line}以上が1つ以下で最小 ${b.lo}〜${b.line} の形は過去 ${past(st.min.all)}。`
      + 'この形では、どのモデルが一番低いかで差は無い'
      + `（LightGBM ${fmtRet(st.min.lgbm.ret)}・${st.min.lgbm.n}件 / `
      + `XGBoost・CatBoost ${fmtRet(st.min.other.ret)}・${st.min.other.n}件）。${lineTxt}${src}`;
  }
  return {
    shape, laggard: lag, laggardPct: p[lag], warn,
    badge: warn ? `際どい・${name} が最下位` : `際どい・最下位 ${name}`,
    text,
  };
}

/**
 * 基準に届かなかったが惜しい候補（3モデルの最小が 85〜90）。
 *
 * 画面に出すのは「なぜ買わないか」を毎日考え直さないため。実測（実験36、
 * 発火8件以上の日・枠の制約なし）:
 *
 *   最小 85〜90 で 3モデルの幅 < 5   118件 +2.70% 勝率58% −10%割れ 5.1%
 *   最小 85〜90 で 3モデルの幅 5〜10  270件 +0.30% 勝率49% −10%割れ 12.6%
 *   （参考）最小 90以上               943件 +2.84% 勝率62% −10%割れ 5.1%
 *
 * つまり「1つのモデルだけが5〜10pt下」の形は、空き枠（0%）と変わらない。
 * とくに最下位が LightGBM のときが弱い（104件 +0.39%）。
 * ※ 上は 153列のころの実験36。いまの239列のモデル（実験70、2026-10-10）では、85〜90 の帯で最下位が
 *   LightGBM の形は弱くない（LightGBM が最下位 202件 +1.3%、XGBoost / CatBoost が最下位 445件 +0.7%。
 *   形ごとの表の値から合成）。弱いのは「2つが95以上で LightGBM が最下位」の形（borderNote）。
 *   幅で分けた数字（+0.30% / +2.70%）は、いまのモデルでは測り直していない。
 */
export function nearMisses(rows, s = STRATEGY) {
  const list = Array.isArray(rows) ? rows : [];
  return list
    .map((c) => ({ c, m: minPct(c), sp: pctSpread(c) }))
    .filter((x) => x.m !== null && x.m >= s.nearLo && x.m < s.agreePct)
    .sort((a, b) => b.m - a.m)
    .map(({ c, m, sp }) => ({
      ...c,
      minPct: m,
      spread: sp,
      laggard: laggard(c),
      // 幅が5以上なら「1つだけ下」の弱い形
      weak: sp !== null && sp >= 5,
    }));
}

/** 3モデルすべてが上位10%か。 */
export function passesAgree(candidate, pct = STRATEGY.agreePct) {
  const m = minPct(candidate);
  return m !== null && m >= pct;
}

/**
 * その日を取るかの判定。発火数だけで決める。
 * 地合いの条件は足さない（発火15件以上の日は100%が TOPIX 200日線より上で、
 * 発火数が地合いを内包している。docs/MODEL_DAY_REGIME.md）。
 */
export function dayVerdict(nBreak, s = STRATEGY) {
  if (!Number.isFinite(nBreak) || nBreak <= 0) {
    return { kind: 'none', tone: 'slate', label: '発火なし',
             note: 'この日は新規の高値更新がありません' };
  }
  if (nBreak >= s.strongBreaks) {
    return { kind: 'strong', tone: 'green', label: '本命の日',
             note: `発火 ${nBreak} 件。実測でいちばん成績が良い帯（20件以上）` };
  }
  if (nBreak >= s.skipBreaks) {
    return { kind: 'ok', tone: 'amber', label: '枠が空いていれば',
             note: `発火 ${nBreak} 件。悪くはない帯（8〜19件、+1.0〜2.2%）。`
                   + '枠が遊ぶより買うほうが良い' };
  }
  return { kind: 'skip', tone: 'red', label: '見送り',
           note: `発火 ${nBreak} 件。7件以下は142件で −0.11%、`
                 + '−10%割れが 14.1%。枠が空いていても買わない' };
}

/**
 * その日の戦略シグナル。
 *
 * rows はその日の候補（predictions.json の candidates を日で絞ったもの）。
 * 戻り値の picks が「買う銘柄」、passed が「基準を満たした全銘柄」。
 */
export function strategySignal(rows, s = STRATEGY) {
  const list = Array.isArray(rows) ? rows : [];
  const nBreak = list.length;
  const verdict = dayVerdict(nBreak, s);
  const passed = list
    .filter((c) => passesAgree(c, s.agreePct))
    .map((c) => ({ ...c, minPct: minPct(c) }))
    // 最小順位の降順。同点なら基準モデルのスコアで決める（表示順を安定させる）
    .sort((a, b) => (b.minPct - a.minPct) || ((b.score ?? 0) - (a.score ?? 0)));
  const buyable = verdict.kind === 'strong' || verdict.kind === 'ok';
  return {
    nBreak,
    verdict,
    passed,
    picks: buyable ? passed.slice(0, s.topK) : [],
    // 基準は満たしたが、その日を取らない判断で見送るもの
    held: buyable ? passed.slice(s.topK) : passed,
    buyable,
  };
}

/**
 * 想定の出口。買値は翌営業日の寄りなので当日は分からない。
 * 終値を仮の買値として指値の目安を出す（あくまで目安と画面に書く）。
 */
export function exitPlan(close, s = STRATEGY) {
  if (!Number.isFinite(close) || close <= 0) return null;
  return {
    target: close * (1 + s.takeProfit / 100),
    takeProfit: s.takeProfit,
    holdDays: s.holdDays,
  };
}

/* ------------------------------------------------------------------ Jev（判断モデル） */

/**
 * Jev（TypeSafe AI の判断モデル）の見立て。predict_daily が各候補に付ける
 * （research/jev_predict.py。candidates[].jev = { prob, model, question, askedAt, cached }）。
 * 問いは「翌営業日の寄りで買い、20営業日以内に高値が買値の +10% に達する」の真偽の確率（%）。
 *
 * 選定の規則（strategySignal）には入れない（運用者の決定 2026-10-10）。4モデルと同じく
 * 並べて見るだけ。確率は較正されていないので、「50%」が過去に半分当たったという意味ではない
 * （過去の候補で測った結果は docs/MODEL_JEV.md の実験73）。買い候補の Jev が line 未満なら
 * 注意を出す。line は結果を見て引いた線ではなく「確率が五分を切る」という素朴な位置で、
 * 実績が溜まったら引き直す。
 */
export const JEV = { line: 50, target: 10, hold: 20 };

/** Jev の確率（%）。無ければ null（鍵が未設定・失敗・問うていない）。 */
export function jevProb(candidate) {
  const v = candidate?.jev?.prob;
  return Number.isFinite(v) ? v : null;
}

/**
 * Jev の値への注意。値が無ければ null。
 *   warn   line 未満
 *   badge  行に出す短い文言（line 以上なら null）
 *   text   詳しい文言（title と、開いたときの本文）
 */
export function jevNote(candidate, j = JEV) {
  const p = jevProb(candidate);
  if (p === null) return null;
  const warn = p < j.line;
  const head = `Jev は「翌営業日の寄りで買い、${j.hold}営業日以内に +${j.target}%」を ${p.toFixed(0)}% と見ている`;
  return {
    prob: p,
    warn,
    badge: warn ? `Jev ${j.line}%未満` : null,
    text: warn
      ? `${head}（${j.line}% 未満）。選定の規則には入れていない。確率は較正されていないので、`
        + '過去の候補で測った実績（docs/MODEL_JEV.md 実験73）と照らして読む'
      : `${head}（${j.line}% 以上）。選定の規則には入れていない`,
  };
}

/** 買い候補のうち Jev が line 未満のもの（戦略パネルの注意の帯に使う）。 */
export function jevWarnings(picks, j = JEV) {
  return (Array.isArray(picks) ? picks : [])
    .map((c) => ({ c, n: jevNote(c, j) }))
    .filter((x) => x.n?.warn);
}
