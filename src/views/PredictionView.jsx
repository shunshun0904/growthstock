import React, { useMemo, useState } from 'react';
import { fmt, fmtInt, fmtSigned, fmtOku, fmtDate, fmtDateTime, DASH } from '../lib/format.js';
import { bandColor, bandLabel, pctColor, pctInk, PCT_LEGEND, modelRows, modelGrid, MODEL_SHORT, MODEL_FAMILY,
  FAMILY_JA, marketTone, candidateToStock } from '../lib/predictions.js';
import { STRATEGY, RULE_MODELS, RULE_JA, BORDER, strategySignal, exitPlan, nearMisses,
         MODEL_JA, fundContrib, frozenNote, borderNote,
} from '../lib/strategy.js';

/**
 * ブレイク予測タブ。
 *
 * その日に78週高値を更新した銘柄を並べ替えたもの。
 * 見つけるのがこの画面、入るかどうかを決めるのは「8軸オクタゴン比較」なので、
 * 候補をオクタゴン側へ送る導線を必ず置く。
 *
 * スコアはモデルごとに並べる。混ぜて1つにはしない（アンサンブルにしない）。
 * 学習器が違えばスコアのスケールも意味も違うので、各モデル自身の過去スコア
 * 分布での位置に揃えて並べ、買うかどうかは人間が統合して決める。
 * 12本（本番の4モデル + ラベル違いの8本）は、ラベル × 学習器の表に数字と色で出す
 * （運用者の指示 2026-10-12「12本の平均ではなく、12個全てを載せてほしい。棒グラフ形式でなくてよい。
 * 大小が分かればよい」。同じ日に出した「12本の合意の百分位」の欄はこの指示で外した）。
 * 順位・帯・追跡は基準モデル（LightGBM）のまま。
 * byModel が無い予測ファイル（5モデルを学習する前のもの）でも壊れないよう、
 * 基準モデル1本の表示に落ちる作りにしてある。
 */
export default function PredictionView({ data, history, onSendToOctagon, sentIds }) {
  const dates = data?.dates || [];
  const [day, setDay] = useState(() => data?.asOf || dates[dates.length - 1]);
  const [openId, setOpenId] = useState(null);

  const rows = useMemo(
    () => (data?.candidates || [])
      .filter((c) => c.date === day)
      .sort((a, b) => a.rankInDay - b.rankInDay),
    [data, day]
  );
  const tone = useMemo(() => marketTone(rows), [rows]);
  const signal = useMemo(() => strategySignal(rows), [rows]);
  const near = useMemo(() => nearMisses(rows), [rows]);
  const m = data?.model || {};

  return (
    <div className="pred">
      <StrategyPanel signal={signal} near={near} day={day} models={data?.models}
                     onSend={(c) => onSendToOctagon(candidateToStock(c))}
                     sentIds={sentIds} />

      <section className="card">
        <div className="card-head">
          <h2>ブレイク予測</h2>
          <span className="sub">
            78週高値を更新した日の銘柄を、上がって続く見込みの順に並べます
          </span>
        </div>

        <div className="pred-strip">
          <div>
            <span className="lab">対象日</span>
            <div className="pred-days">
              {dates.map((d) => (
                <button key={d} className={`btn btn-ghost${d === day ? ' on' : ''}`}
                        onClick={() => { setDay(d); setOpenId(null); }}>
                  {fmtDate(d)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <span className="lab">候補</span>
            <strong className="num">{rows.length}</strong> 銘柄
          </div>
          <div>
            <span className="lab">その日の地合い</span>
            {tone ? (
              <span className={`badge ${tone.tone}`}>{tone.label}</span>
            ) : DASH}
          </div>
          <div>
            <span className="lab">モデル</span>
            <span className="num">{fmtDate(m.trainedAt)}</span> 学習
            <span className="sep">/</span>
            訓練 <span className="num">{fmtInt(m.nTrain)}</span>件
          </div>
        </div>

        {tone?.tone === 'red' && (
          <div className="banner warn pred-note">
            <span>⚠</span>
            <div>
              この日は候補全体で地合いの寄与がマイナスです。順位は「候補の中での
              相対」なので、1位でも水準そのものが低いことがあります。
              下の<b>スコア帯の実績</b>で絶対水準を確かめてください。
            </div>
          </div>
        )}

        {rows.length === 0 && (
          <div className="empty">この日は新規の高値更新がありません。</div>
        )}

        <div className="pred-list">
          {rows.map((c) => (
            <Row key={c.jqCode} c={c} models={data?.models} open={openId === c.jqCode}
                 onToggle={() => setOpenId(openId === c.jqCode ? null : c.jqCode)}
                 onSend={() => onSendToOctagon(candidateToStock(c))}
                 sent={sentIds?.has(`pred:${c.jqCode}`)} />
          ))}
        </div>
      </section>

      <BandTable bands={data?.scoreBands} />
      <ModelLineup models={data?.models} />
      <HistoryPanel history={history} horizon={m.riseHorizon} />
      <ModelCard model={m} notes={data?.notes} generatedAt={data?.generatedAt} />
    </div>
  );
}

/* ------------------------------------------------------ 今日の戦略シグナル */

/**
 * docs/PLAYBOOK.md の手順を、その日のデータでそのまま実行して見せる。
 *
 * 判断の材料（発火数・規則の3モデル（2026-10-12 から 1.5σ の3本）の百分位）は下の一覧にも出ているが、
 * 「今日は買うのか、買うなら何を、いくらで手仕舞うのか」を1か所で
 * 言い切る場所が要る。迷いどころを毎日つくらないための画面。
 */
function StrategyPanel({ signal, near, day, models, onSend, sentIds }) {
  const { nBreak, verdict, picks, passed, buyable } = signal;
  return (
    <section className="card strat">
      <div className="card-head">
        <h2>今日の戦略</h2>
        <span className="sub">
          {RULE_JA}（L15・X15・C15）がすべて{STRATEGY.agreePct}以上の銘柄を{STRATEGY.topK}件まで。
          翌営業日の寄りで買い、+{STRATEGY.takeProfit}% の指値か
          {STRATEGY.holdDays}営業日目の終値で降りる（同時保有は{STRATEGY.maxSlots}銘柄まで、
          発火{STRATEGY.skipBreaks}件未満の日は見送り）
        </span>
      </div>

      <div className="strat-head">
        <span className={`badge ${verdict.tone} strat-verdict`}>{verdict.label}</span>
        <div className="strat-facts">
          <div>
            <span className="lab">発火数</span>
            <strong className="num">{nBreak}</strong>
            <span className="sub">
              件（{STRATEGY.strongBreaks}件以上が本命 /
              {STRATEGY.skipBreaks - 1}件以下は見送り）
            </span>
          </div>
          <div>
            <span className="lab">{RULE_JA}がすべて{STRATEGY.agreePct}以上</span>
            <strong className="num">{passed.length}</strong>
            <span className="sub">件 / {nBreak}件</span>
          </div>
        </div>
        <p className="sub strat-note">{verdict.note}</p>
      </div>

      {picks.length > 0 ? (
        <>
          <div className="strat-picks">
            {picks.map((c, i) => (
              <PickCard key={c.jqCode} c={c} n={i + 1} models={models}
                        onSend={() => onSend(c)}
                        sent={sentIds?.has(`pred:${c.jqCode}`)} />
            ))}
          </div>
          <ol className="strat-steps">
            <li><b>翌営業日の寄り</b>で成行。終値では買えない（候補が分かるのが終値後）</li>
            <li>買えたら <b>+{STRATEGY.takeProfit}% の指値</b>を置く
                （過去の予測でこの手順のとおりに買った取引のうち、届いたのは約40%、届くときの中央値は7営業日）</li>
            <li>届かなければ <b>{STRATEGY.holdDays}営業日目の終値</b>で売る</li>
            <li><b>枠は{STRATEGY.maxSlots}つまで</b>。埋まっていたら見送る。
                保有中の銘柄は、ここに良い候補が出ても<b>切らない</b>
                （実測で乗り換え37回のうち24回は切らないほうが良かった）</li>
          </ol>
        </>
      ) : (
        <div className="empty strat-empty">
          {nBreak === 0 ? 'この日は新規の高値更新がありません。'
            : passed.length === 0
              ? `${RULE_JA}がすべて${STRATEGY.agreePct}以上の銘柄はありません。買いません。`
              : `発火が ${nBreak} 件しかありません。基準を満たした銘柄はありますが、`
                + 'この日は見送ります。'}
        </div>
      )}

      {!buyable && passed.length > 0 && (
        <div className="strat-held">
          <span className="lab">参考（基準は満たしたが見送る）</span>
          {passed.slice(0, 3).map((c) => (
            <span key={c.jqCode} className="strat-chip">
              {c.name || c.code}
              <em className="num">{fmt(c.minPct, 0)}</em>
            </span>
          ))}
        </div>
      )}

      {near.length > 0 && <NearMiss near={near} />}

      <p className="strat-src sub">
        実測（画面と同じ百分位、2021-05〜2026-09 の out-of-fold、{RULE_JA}がすべて95以上の354件。
        +10% の指値・届かなければ20営業日目の終値）: 発火20件以上は192件で +4.9%・勝率78%、
        8〜19件は140件で +1.5%・勝率59%、7件以下は22件と少なく +5.4%。この手順のとおり3枠で回すと
        107取引・1取引 +1.5%（その時点の百分位で測ると118取引・+1.8%）。これまでの目安（1.2σ の3本が
        95以上）と比べて、どちらが良いかは過去のデータでは見分けられない
        （docs/MODEL_ADOPTION_RULES.md §46）。
      </p>
    </section>
  );
}

/**
 * 惜しい候補（規則の3モデル（1.5σ の3本）の最小が 85〜95）。**買わない**ことの根拠を添えて出す。
 *
 * 毎日「あと少しなのに」と迷い直さないための欄。数字は今の規則で測ったもの（research/exp/probe_k15_baseline.py。
 * 画面と同じ百分位、発火8件以上の日）。以前の規則（1.2σ の3本、最小 85〜90）で「1つだけ5pt以上低い形は空き枠と
 * 変わらない」とした実験36 は、1.5σ の3本では成り立たない（幅5以上のほうが少し上）。
 */
function NearMiss({ near }) {
  return (
    <div className="strat-near">
      <span className="lab">
        惜しい候補（最小 {STRATEGY.nearLo}〜{STRATEGY.agreePct}）— 買わない
      </span>
      <div className="strat-near-list">
        {near.map((c) => (
          <div key={c.jqCode} className="strat-near-row">
            <span className="strat-near-name">
              {c.name || c.code}
              <BorderBadge c={c} />
            </span>
            <span className="num strat-near-pct">最小 {fmt(c.minPct, 1)}</span>
            <span className="sub">
              {c.weak
                ? `${MODEL_JA[c.laggard] || c.laggard} だけ `
                  + `${fmt(c.spread, 1)}pt 低い形。最小 85〜95・幅5以上の形は過去 584件で +2.1%・勝率63%`
                  + `（3本とも95以上は354件で +3.6%）`
                : `3本がほぼ揃っている（幅 ${fmt(c.spread, 1)}pt）。最小 85〜95・幅5未満の形は過去 369件で `
                  + '+1.1%・勝率58%（3本とも95以上は354件で +3.6%）'}
              <FundNote c={c} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 決算がスコアを押しているか引いているか。
 *
 * 「決算は悪くないのに」と思ったときに見る欄。決算は153列中118列（77%）を
 * 占めるので、スコアは既に決算を読んだ結果になっている。
 */
function FundNote({ c }) {
  const f = fundContrib(c);
  if (f === null) return null;
  return (
    <>
      {' '}
      <span className={f < 0 ? 'strat-fund-down' : 'strat-fund-up'}>
        {/* 寄与は対数オッズ。% ではないので単位は付けない */}
        決算の寄与 {fmtSigned(f, 2, '')}
        {f < 0 ? '（決算が引いている）' : '（決算が押している）'}
      </span>
    </>
  );
}

function PickCard({ c, n, models, onSend, sent }) {
  const plan = exitPlan(c.close);
  return (
    <div className="strat-pick">
      <div className="strat-pick-head">
        <span className="strat-pick-n num">{n}</span>
        <span className="pred-id">
          <strong>{c.name || c.code}</strong>
          <span className="sub num">{c.code}</span>
          {c.sector && <span className="sub">{c.sector}</span>}
          <BorderBadge c={c} />
        </span>
        <button className="btn btn-primary" onClick={onSend} disabled={sent}>
          {sent ? '送信済み' : '8軸で見る'}
        </button>
      </div>
      <div className="strat-pick-body">
        <div className="strat-metric">
          <span className="lab">{RULE_JA}の最小</span>
          <strong className="num">
            <i className="pct-sw" style={{ background: pctColor(c.minPct) }} aria-hidden="true" />
            {fmt(c.minPct, 0)}
          </strong>
          <span className="sub">百分位</span>
        </div>
        {modelRows(c, models).length > 0 ? (
          <div className="strat-metric strat-mg">
            <span className="lab">モデル別の百分位</span>
            <ModelGrid c={c} models={models} />
          </div>
        ) : RULE_MODELS.map((a) => (
          <div key={a} className="strat-metric">
            <span className="lab">{MODEL_SHORT[a]}</span>
            <span className="num">{fmt(c.byModel?.[a]?.pctHistorical, 0)}</span>
          </div>
        ))}
        <div className="strat-metric">
          <span className="lab">終値</span>
          <span className="num">{fmtInt(c.close)}</span>
        </div>
        {plan && (
          <div className="strat-metric">
            <span className="lab">+{plan.takeProfit}% の目安</span>
            <span className="num">{fmtInt(plan.target)}</span>
            <span className="sub">終値基準。実際は寄り値から</span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * 際どい候補の印（1.5σ の3本で見た今の規則の線「3本とも95以上」の下。strategy.js の borderNote）。
 * その形でそのモデルが一番低い組の過去の平均が 0% 以下（20件以上）なら赤、それ以外は黄（先に決めた線。BORDER_WARN）。
 * 詳しい文言は title と、行を開いたときの本文に出す。選定の規則は変えない。
 */
function BorderBadge({ c }) {
  const n = borderNote(c);
  if (!n) return null;
  return (
    <span className={`badge ${n.warn ? 'red' : 'amber'} pred-warn pred-border`} title={n.text}>
      {n.badge}
    </span>
  );
}

/* ------------------------------------------------------------------ 候補1件 */

function Row({ c, models, open, onToggle, onSend, sent }) {
  const color = bandColor(c.band);
  const mr = modelRows(c, models);
  return (
    <div className={`pred-row${open ? ' open' : ''}`}>
      <button className="pred-main" onClick={onToggle} aria-expanded={open}>
        <span className="pred-rank num" style={{ color }}>{c.rankInDay}</span>
        <span className="pred-id">
          <strong>{c.name || c.code}</strong>
          <span className="sub num">{c.code}</span>
          {c.sector && <span className="sub">{c.sector}</span>}
          {frozenNote(c) && (
            <span className="badge red pred-warn"
                  title={`${frozenNote(c)}。直近20日の日々の値動きがほぼ無い銘柄は、TOB 等で価格が固定されている可能性があります。`
                         + '実験63では該当した2件がどちらも外れました（上がりようがない）'}>
              値動きなし
            </span>
          )}
          <BorderBadge c={c} />
        </span>
        {mr.length > 0 ? (
          <span className="pred-ms">
            <span className="lab">モデル別の百分位<PctLegend /></span>
            <ModelGrid c={c} models={models} />
          </span>
        ) : (
          <span className="pred-bar" title={`スコア ${fmt(c.score, 4)}`}>
            <i style={{ width: `${Math.max(2, (c.pctHistorical ?? 0))}%`, background: color }} />
            <em className="num">{fmt(c.pctHistorical, 0)}</em>
          </span>
        )}
        <span className="pred-cell">
          <span className="lab">帯の正例率</span>
          <span className="num">{fmt(c.bandPositiveRate, 1, '%')}</span>
        </span>
        <span className="pred-cell">
          <span className="lab">帯の実収益</span>
          <span className="num" style={{ color: (c.bandOutcome ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {fmtSigned(c.bandOutcome, 2)}
          </span>
        </span>
        <span className="pred-cell">
          <span className="lab">必要上昇率</span>
          <span className="num">{fmt(c.needPct, 1, '%')}</span>
        </span>
        <span className={`badge ${c.band >= 9 ? 'violet' : c.band >= 7 ? 'green' : c.band >= 4 ? 'amber' : 'red'}`}
              title={`基準モデル（LightGBM）のスコア帯 ${c.band}/10。`
                     + '左のモデル別の百分位とは別の物差しです'}>
          {bandLabel(c.band)}
        </span>
      </button>

      {open && <Detail c={c} models={models} onSend={onSend} sent={sent} />}
    </div>
  );
}

/** 百分位の色の区分（明るいほど上）。表の見出しに添える。 */
function PctLegend() {
  return (
    <span className="pred-ms-legend"
          title="色の区分（明るいほど上）: 95以上（上位5%。運用の選定の線）/ 90以上 / 80以上 / 60以上 / 30以上 / それ未満">
      {PCT_LEGEND.map((t) => (
        <i key={t} style={{ background: pctColor(t) }} aria-hidden="true" />
      ))}
      95+
    </span>
  );
}

/**
 * モデル別の百分位の表（運用者の指示 2026-10-12「12個全てを載せてほしい。棒グラフ形式でなくてよい。
 * 大小が分かればよい」）。
 *
 * 行 = 学習に使ったラベル（1.2σ = 本番 / 1.5σ / 1.8σ）、列 = 学習器（LGB / XGB / CAT / LR）。マスの数字は
 * **そのモデル自身の**過去スコア分布での位置（百分位）、地の色は区分（明るいほど上。pctColor）。
 * モデルをまたいで生スコアを比べられないので、位置に揃えてある。並び順は行をまたいで固定
 * （payload の models 順）。予測ファイルに無いモデルのマスは「—」。
 * large は行を開いた詳しい欄（小数1桁・行の見出しに「本番」）。
 */
function ModelGrid({ c, models, large = false }) {
  const g = modelGrid(c, models);
  if (!g.rows.length) return null;
  return (
    <span className={`pred-mg${large ? ' lg' : ''}`} style={{ '--mg-cols': g.cols.length }}
          aria-label="モデル別の百分位（行はラベル、列は学習器）">
      <span className="h" aria-hidden="true" />
      {g.cols.map((k) => <span key={k.learner} className="h">{k.short}</span>)}
      {g.rows.map((r) => (
        <React.Fragment key={r.label}>
          <span className="r"
                title={r.label === 'base' ? '本番のラベル（到達しきい値 1.2σ）で学習したモデル'
                  : `ラベル違い（到達しきい値 ${r.labelJa}）で学習したモデル。本番の判定には使わない`}>
            {r.labelJa}{large && r.label === 'base' ? '（本番）' : ''}
          </span>
          {r.cells.map((x, i) => (x && Number.isFinite(x.pct) ? (
            <span key={g.cols[i].learner} className="c num"
                  style={{ background: pctColor(x.pct), color: pctInk(x.pct) }}
                  title={`${x.name}: 過去スコア分布の下から ${fmt(x.pct, 1, '%')}／生スコア ${fmt(x.score, 4)}`}>
              {fmt(x.pct, large ? 1 : 0)}
            </span>
          ) : (
            <span key={g.cols[i].learner} className="c na">{DASH}</span>
          )))}
        </React.Fragment>
      ))}
    </span>
  );
}

const NUMS = [
  ['株価', (c) => fmtInt(c.close, '円')],
  ['時価総額', (c) => fmtOku(c.marketCap)],
  ['売買代金(20日平均)', (c) => fmt(c.tradingValue, 2, '億円/日')],
  ['日次ボラ', (c) => fmt(c.vol20d, 2, '%')],
  ['必要上昇率', (c) => fmt(c.needPct, 1, '%')],
  ['20日リターン', (c) => fmtSigned(c.ret20d, 1)],
  ['高値からの位置', (c) => fmt(c.rHigh, 1, '%')],
  ['抜けた幅', (c) => fmtSigned(c.breakMargin, 2)],
  ['ベース日数', (c) => fmtInt(c.baseLength, '日')],
  ['信用倍率', (c) => fmt(c.creditRatio, 2, '倍')],
  ['PER', (c) => fmt(c.per, 1, '倍')],
  ['PBR', (c) => fmt(c.pbr, 2, '倍')],
  ['配当利回り', (c) => fmt(c.divYield, 2, '%')],
  ['EPS成長', (c) => fmtSigned(c.epsGrowth, 1)],
  ['売上成長', (c) => fmtSigned(c.salesGrowth, 1)],
];

function Detail({ c, models, onSend, sent }) {
  const g = c.contrib?.groups || {};
  const maxAbs = Math.max(...Object.values(g).map(Math.abs), 0.001);
  const top = (c.contrib?.top || []).slice(0, 10);
  const maxTop = Math.max(...top.map((t) => Math.abs(t.contrib)), 0.001);

  const mr = modelRows(c, models);
  const bn = borderNote(c);

  return (
    <div className="pred-detail">
      {bn && (
        <div className={`banner ${bn.warn ? 'warn' : 'info'} pred-border-note`}>
          <span>{bn.warn ? '⚠' : 'ℹ'}</span>
          <div>
            <b>{RULE_JA}で見た際どい候補（3本とも{BORDER.line}以上の線の下）。</b>{bn.text}
          </div>
        </div>
      )}
      <div className="pred-detail-grid">
        {mr.length > 0 && (
          <div>
            <h4>モデル別の見立て</h4>
            <p className="sub">
              {mr.length}本のモデルが、それぞれ独立に採点したもの。
              <b>混ぜていません</b>（アンサンブルではない）。生スコアはモデル間で
              比べられないので、<b>各モデル自身の過去スコア分布での位置</b>（百分位）に
              揃えてあります。地の色が明るいほど上です<PctLegend />
            </p>
            <ModelGrid c={c} models={models} large />
            <div className="pred-split" style={{ marginTop: 10 }}>
              <span>
                <span className="lab">上位10%と見たモデル</span>
                <b className="num">{c.agree90}/{c.nModels}</b>
                {Number.isFinite(c.agreeK90) && c.nK > 0 && (
                  <span className="sub">（ラベル違い {c.agreeK90}/{c.nK}）</span>
                )}
              </span>
              <span>
                <span className="lab">順位・帯の基準</span>
                <b>LightGBM</b>
              </span>
            </div>
            <p className="sub" style={{ margin: '8px 0 0' }}>
              行は学習に使ったラベルで、1.2σ が本番（順位・帯の基準）です。1.5σ・1.8σ は、より大きく上がった
              銘柄だけを正例にして学習したもの。今日の戦略の規則は、2026-10-12 から 1.5σ の木の3本
              （L15・X15・C15）がすべて95以上です（運用者の決定）。
              同じ行・同じ列のモデルは似た見方をするので、そろって高くても独立した票ではありません。
              1.2σ の3本が95以上の候補に「1.5σ・1.8σ の3本も95以上」を足しても、過去の前向きの
              検証では成績は安定して上がりませんでした（実験82）。
            </p>
          </div>
        )}
        <div>
          <h4>この順位の根拠</h4>
          <p className="sub">
            モデルの出力を特徴量ごとに分解したもの（TreeSHAP）。
            対数オッズ空間の寄与なので、確率の差ではなく
            <b>押し上げ／押し下げの相対の大きさ</b>として読みます。
          </p>
          <div className="pred-split">
            <span>
              <span className="lab">地合い</span>
              <b className="num" style={{ color: c.contrib.marketContrib >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {fmtSigned(c.contrib.marketContrib, 3, '')}
              </b>
              <span className="sub">（寄与の {fmt(c.contrib.marketShare, 0, '%')}）</span>
            </span>
            <span>
              <span className="lab">銘柄固有</span>
              <b className="num" style={{ color: c.contrib.stockContrib >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {fmtSigned(c.contrib.stockContrib, 3, '')}
              </b>
            </span>
          </div>
          <div className="pred-bars">
            {Object.entries(g).map(([k, v]) => (
              <div key={k} className="pred-b">
                <span className="k">{k}</span>
                <span className="t">
                  <i className={v >= 0 ? 'pos' : 'neg'}
                     style={{ width: `${(Math.abs(v) / maxAbs) * 100}%` }} />
                </span>
                <span className="v num">{fmtSigned(v, 3, '')}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4>効いた特徴量</h4>
          <div className="pred-bars">
            {top.map((t) => (
              <div key={t.col} className="pred-b" title={t.ja || t.col}>
                <span className="k mono">{t.col}</span>
                <span className="t">
                  <i className={t.contrib >= 0 ? 'pos' : 'neg'}
                     style={{ width: `${(Math.abs(t.contrib) / maxTop) * 100}%` }} />
                </span>
                <span className="v num">{fmtSigned(t.contrib, 3, '')}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4>エントリー判断に使う値</h4>
          <dl className="pred-dl">
            {NUMS.map(([k, f]) => (
              <React.Fragment key={k}>
                <dt>{k}</dt><dd className="num">{f(c)}</dd>
              </React.Fragment>
            ))}
          </dl>
          <button className={`btn ${sent ? '' : 'btn-primary'}`} onClick={onSend} disabled={sent}>
            {sent ? '8軸オクタゴンに追加済み' : '8軸オクタゴンで見る →'}
          </button>
          <p className="sub" style={{ marginTop: 8 }}>
            予測パイプラインが持っている実測値をそのまま渡します。
            取れていない指標は「—」のままで、0 では埋めません。
          </p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ 補助パネル */

// 物差しを変える前に学習したモデルは end_median のまま。週次の再学習が
// 一巡するまで両方を見る（一巡したら outcome_median だけでよい）。
const bandOutcome = (r) => r?.outcome_median ?? r?.end_median;

function BandTable({ bands }) {
  if (!bands?.bands?.length) return null;
  return (
    <section className="card">
      <div className="card-head">
        <h2>スコア帯の実績（基準モデル）</h2>
        <span className="sub">
          out-of-fold 予測（その行より前のデータだけで学習したモデルの採点）で数えた実測値。
          帯は基準モデル（LightGBM）のスコアで切っています
        </span>
      </div>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr><th>帯</th><th>件数</th><th>スコア下限</th><th>正例率</th>
              <th>実収益の中央値</th><th>勝率</th></tr>
          </thead>
          <tbody>
            {bands.bands.map((r) => (
              <tr key={r.band}>
                <td><span className="badge" style={{ color: bandColor(r.band) }}>{r.band}</span></td>
                <td className="num">{fmtInt(r.n)}</td>
                <td className="num">{fmt(r.score_lo, 4)}</td>
                <td className="num">{fmt(r.positive_rate * 100, 1, '%')}</td>
                <td className="num" style={{ color: bandOutcome(r) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {fmtSigned(bandOutcome(r), 2)}
                </td>
                <td className="num">{fmt(r.win_rate * 100, 1, '%')}</td>
              </tr>
            ))}
            <tr className="tot">
              <td>全体</td><td className="num">{fmtInt(bands.n)}</td><td>{DASH}</td>
              <td className="num">{fmt(bands.base_positive_rate * 100, 1, '%')}</td>
              <td className="num">
                {fmtSigned(bands.base_outcome_median ?? bands.base_end_median, 2)}
              </td>
              <td className="num">{fmt(bands.base_win_rate * 100, 1, '%')}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="sub">
        {bands.outcome?.label ? (
          <>
            「実収益」は{bands.outcome.label}。
            ブレイク当日の終値では買えない（候補が判明するのは終値が出た後）ので、
            買いは翌営業日の寄りで測っています。
          </>
        ) : (
          <>
            「実収益」は基準日の終値で買い、60営業日後の5日平均終値で売ったときの
            上昇率。<b>これは物差しを変える前のモデルの数字</b>で、次の週次再学習から
            「翌営業日の寄り買い・20営業日」に変わります。
          </>
        )}
        いずれも正例・負例の判定条件とは無関係に測った実測値です。
      </p>
    </section>
  );
}

/**
 * 画面に並んでいるモデルの素性。
 *
 * 週次の学習は5モデルを別々に回し、1つ失敗しても他を止めない作りなので、
 * どのモデルがいつ学習されたのかを画面から追えるようにしておく。
 * 探索時の PR-AUC も出すが、これは内側検証の値で運用成績ではない。
 */
function ModelLineup({ models }) {
  if (!models?.length) return null;
  return (
    <section className="card">
      <div className="card-head">
        <h2>並べているモデル</h2>
        <span className="sub">
          {models.length}モデルがそれぞれ独立に採点しています（混ぜていません）
        </span>
      </div>
      <div className="tbl-wrap">
        <table className="tbl tbl-lineup">
          <thead>
            <tr><th>記号</th><th>塊</th><th>モデル</th><th>何を見ているか</th>
              <th>学習日</th><th>OOF件数</th><th>探索PR-AUC</th></tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.algo}>
                <td className="num">{m.short || MODEL_SHORT[m.algo] || m.algo}</td>
                <td className="sub">{FAMILY_JA[MODEL_FAMILY[m.algo]] || DASH}</td>
                <td>{m.name}{m.variant && Number.isFinite(m.positiveRate)
                  ? <span className="sub">（正例率 {fmt(m.positiveRate * 100, 1, '%')}）</span> : null}</td>
                <td className="sub">{m.note || DASH}</td>
                <td className="num">{fmtDate(m.trainedAt)}</td>
                <td className="num">{fmtInt(m.nOof)}</td>
                <td className="num">{fmt(m.cv?.mean_pr_auc, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="sub">
        探索PR-AUC は探索時の内側検証（5分割・50試行・木200本固定）の値です。
        ラベル違い（1.5σ / 1.8σ。記号の末尾が 15 / 18）はラベルが違うので、この値を本番のモデルと比べられません。
        運用は上位だけを買うので、この順位がそのまま運用の優劣にはなりません。
        どのモデルを重く見るかは、上の「スコア帯の実績」と
        「過去に出した銘柄のその後」で確かめてください。
      </p>
    </section>
  );
}

function HistoryPanel({ history, horizon }) {
  const entries = history?.entries || [];
  return (
    <section className="card">
      <div className="card-head">
        <h2>過去に出した銘柄のその後</h2>
        <span className="sub">
          毎営業日の上位5銘柄を記録し、現在値と比べ直しています
        </span>
      </div>
      {entries.length === 0 ? (
        <div className="empty">
          まだ記録がありません。日次の予測を回すごとに1日ぶんずつ増えます。
        </div>
      ) : (
        <div className="tbl-wrap">
          <table className="tbl">
            <thead>
              <tr><th>予測日</th><th>順位</th><th>銘柄</th><th>帯</th>
                <th>予測時</th><th>現在</th><th>騰落</th><th>経過</th></tr>
            </thead>
            <tbody>
              {entries.slice(0, 40).map((e) => (
                <tr key={`${e.date}-${e.jqCode}`}>
                  <td className="num">{fmtDate(e.date)}</td>
                  <td className="num">{e.rank}/{e.nInDay}</td>
                  <td>{e.name || e.code} <span className="sub num">{e.code}</span></td>
                  <td><span className="badge" style={{ color: bandColor(e.band) }}>{e.band}</span></td>
                  <td className="num">{fmtInt(e.closeAtPick)}</td>
                  <td className="num">{fmtInt(e.closeNow)}</td>
                  <td className="num" style={{ color: (e.returnPct ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {fmtSigned(e.returnPct, 2)}
                  </td>
                  <td className="num sub">
                    {fmtInt(e.daysElapsed)}/{fmtInt(horizon)}営業日
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="sub">
        経過が参照ホライズンに届いていないものは途中経過です。
        判断材料にするのは十分な件数がたまってからにしてください。
      </p>
    </section>
  );
}

function ModelCard({ model, notes, generatedAt }) {
  const cv = model?.cv || {};
  return (
    <section className="card">
      <div className="card-head">
        <h2>モデルの素性</h2>
        <span className="sub">数字の出どころ</span>
      </div>
      <dl className="pred-dl wide">
        <dt>正例の定義</dt><dd>{model?.label || DASH}</dd>
        <dt>到達しきい値</dt>
        <dd>
          銘柄自身の{fmtInt(model?.riseHorizon)}営業日σの{fmt(model?.volNormK, 1)}倍
          <span className="sub">（固定%ではないので、銘柄ごとに必要上昇率が違います）</span>
        </dd>
        <dt>母集団</dt>
        <dd>
          {fmtInt(Math.round((model?.highWindow ?? 0) / 245 * 52))}週高値の更新日
          <span className="sep">/</span>
          20日平均売買代金 {fmt(model?.minTradingValue, 1)}億円以上
        </dd>
        <dt>学習</dt>
        <dd className="num">
          {fmtDateTime(model?.trainedAt)}<span className="sep">/</span>
          {fmtInt(model?.nTrain)}件（{fmtDate(model?.trainFrom)}〜{fmtDate(model?.trainTo)}）
          <span className="sep">/</span>正例率 {fmt((model?.positiveRate ?? 0) * 100, 2, '%')}
        </dd>
        <dt>特徴量</dt><dd className="num">{fmtInt(model?.nFeatures)}列（{model?.preset}）</dd>
        {cv.mean_pr_auc != null && (
          <>
            <dt>探索時のCV</dt>
            <dd className="num">
              PR-AUC {fmt(cv.mean_pr_auc, 4)}<span className="sep">/</span>
              ROC-AUC {fmt(cv.mean_roc_auc, 4)}
              <span className="sub">（{cv.n_splits}分割・{cv.n_trials}試行）</span>
            </dd>
          </>
        )}
        <dt>データ生成</dt><dd className="num">{fmtDateTime(generatedAt)}</dd>
      </dl>
      {notes?.length > 0 && (
        <ul className="pred-notes">{notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      )}
    </section>
  );
}
