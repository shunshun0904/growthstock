import React, { useMemo, useState } from 'react';
import { fmt, fmtInt, fmtSigned, fmtOku, fmtDate, fmtDateTime, DASH } from '../lib/format.js';
import { bandColor, bandLabel, marketTone, candidateToStock } from '../lib/predictions.js';

/**
 * ブレイク予測タブ。
 *
 * その日に78週高値を更新した銘柄を LightGBM が並べ替えたもの。
 * 見つけるのがこの画面、入るかどうかを決めるのは「8軸オクタゴン比較」なので、
 * 候補をオクタゴン側へ送る導線を必ず置く。
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
  const m = data?.model || {};

  return (
    <div className="pred">
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
            <Row key={c.jqCode} c={c} open={openId === c.jqCode}
                 onToggle={() => setOpenId(openId === c.jqCode ? null : c.jqCode)}
                 onSend={() => onSendToOctagon(candidateToStock(c))}
                 sent={sentIds?.has(`pred:${c.jqCode}`)} />
          ))}
        </div>
      </section>

      <BandTable bands={data?.scoreBands} />
      <HistoryPanel history={history} horizon={m.riseHorizon} />
      <ModelCard model={m} notes={data?.notes} generatedAt={data?.generatedAt} />
    </div>
  );
}

/* ------------------------------------------------------------------ 候補1件 */

function Row({ c, open, onToggle, onSend, sent }) {
  const color = bandColor(c.band);
  return (
    <div className={`pred-row${open ? ' open' : ''}`}>
      <button className="pred-main" onClick={onToggle} aria-expanded={open}>
        <span className="pred-rank num" style={{ color }}>{c.rankInDay}</span>
        <span className="pred-id">
          <strong>{c.name || c.code}</strong>
          <span className="sub num">{c.code}</span>
          {c.sector && <span className="sub">{c.sector}</span>}
        </span>
        <span className="pred-bar" title={`スコア ${fmt(c.score, 4)}`}>
          <i style={{ width: `${Math.max(2, (c.pctHistorical ?? 0))}%`, background: color }} />
          <em className="num">{fmt(c.pctHistorical, 0)}</em>
        </span>
        <span className="pred-cell">
          <span className="lab">帯の正例率</span>
          <span className="num">{fmt(c.bandPositiveRate, 1, '%')}</span>
        </span>
        <span className="pred-cell">
          <span className="lab">帯の実収益</span>
          <span className="num" style={{ color: (c.bandEndMedian ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {fmtSigned(c.bandEndMedian, 2)}
          </span>
        </span>
        <span className="pred-cell">
          <span className="lab">必要上昇率</span>
          <span className="num">{fmt(c.needPct, 1, '%')}</span>
        </span>
        <span className={`badge ${c.band >= 9 ? 'violet' : c.band >= 7 ? 'green' : c.band >= 4 ? 'amber' : 'red'}`}>
          {bandLabel(c.band)}
        </span>
      </button>

      {open && <Detail c={c} onSend={onSend} sent={sent} />}
    </div>
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

function Detail({ c, onSend, sent }) {
  const g = c.contrib?.groups || {};
  const maxAbs = Math.max(...Object.values(g).map(Math.abs), 0.001);
  const top = (c.contrib?.top || []).slice(0, 10);
  const maxTop = Math.max(...top.map((t) => Math.abs(t.contrib)), 0.001);

  return (
    <div className="pred-detail">
      <div className="pred-detail-grid">
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

function BandTable({ bands }) {
  if (!bands?.bands?.length) return null;
  return (
    <section className="card">
      <div className="card-head">
        <h2>スコア帯の実績</h2>
        <span className="sub">
          out-of-fold 予測（その行より前のデータだけで学習したモデルの採点）で数えた実測値
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
                <td className="num" style={{ color: r.end_median >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {fmtSigned(r.end_median, 2)}
                </td>
                <td className="num">{fmt(r.win_rate * 100, 1, '%')}</td>
              </tr>
            ))}
            <tr className="tot">
              <td>全体</td><td className="num">{fmtInt(bands.n)}</td><td>{DASH}</td>
              <td className="num">{fmt(bands.base_positive_rate * 100, 1, '%')}</td>
              <td className="num">{fmtSigned(bands.base_end_median, 2)}</td>
              <td className="num">{fmt(bands.base_win_rate * 100, 1, '%')}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="sub">
        「実収益」は参照ホライズン（60営業日後の5日平均終値）の上昇率。
        ラベル定義に依存しないので、ラベルを変えても意味が変わりません。
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
