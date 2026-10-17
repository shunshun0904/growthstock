import React, { useMemo, useState } from 'react';
import { fmt, fmtDate } from '../lib/format.js';
import { earningsNote } from '../lib/predictions.js';
import { STRATEGY, RULE_JA, orderRows, strategySignal } from '../lib/strategy.js';
import { newsLinks, MARKET_LINKS } from '../lib/news.js';

/**
 * 候補銘柄のニュース（運用者の依頼 2026-10-17。左から2番目のタブ）。
 *
 * モデルの特徴量に入っていない材料（ニュース・適時開示・アナリスト予想・個人投資家の声）を、買う前に自分の目で
 * 確かめるためのリンク集。並びはブレイク予測のタブと同じ（1.5σ の3本の最小の百分位が高い順）。
 * 見出しはこの画面に取り込まない（各サイトの robots.txt・規約。src/lib/news.js の冒頭）。
 */
export default function NewsView({ data }) {
  const dates = data?.dates || [];
  const [day, setDay] = useState(() => data?.asOf || dates[dates.length - 1]);
  // すべての候補か、今日の戦略の規則（1.5σ の3本すべて95以上）を通った候補だけか
  const [only, setOnly] = useState('all');

  const rows = useMemo(
    () => orderRows((data?.candidates || []).filter((c) => c.date === day)),
    [data, day]
  );
  const signal = useMemo(() => strategySignal(rows), [rows]);
  const passIds = useMemo(() => new Set(signal.passed.map((c) => c.jqCode)), [signal]);
  const pickIds = useMemo(() => new Set(signal.picks.map((c) => c.jqCode)), [signal]);
  const shown = only === 'pass' ? rows.filter((c) => passIds.has(c.jqCode)) : rows;

  return (
    <div className="news">
      <section className="card">
        <div className="card-head">
          <h2>候補銘柄のニュース</h2>
          <span className="sub">
            候補ごとに、ニュース・適時開示・アナリスト予想・個人投資家の声のページを開けます（外部サイト）
          </span>
        </div>

        <div className="pred-strip">
          <div>
            <span className="lab">対象日</span>
            <div className="pred-days">
              {dates.map((d) => (
                <button key={d} className={`btn btn-ghost${d === day ? ' on' : ''}`} onClick={() => setDay(d)}>
                  {fmtDate(d)}
                </button>
              ))}
            </div>
          </div>
          <div>
            <span className="lab">表示</span>
            <div className="pred-days pred-metric">
              <button className={`btn btn-ghost${only === 'all' ? ' on' : ''}`} onClick={() => setOnly('all')}>
                すべての候補
              </button>
              <button className={`btn btn-ghost${only === 'pass' ? ' on' : ''}`} onClick={() => setOnly('pass')}>
                {`${RULE_JA}が${STRATEGY.agreePct}以上`}
              </button>
            </div>
          </div>
          <div>
            <span className="lab">候補</span>
            <strong className="num">{shown.length}</strong> / {rows.length} 銘柄
          </div>
        </div>

        <div className="banner info pred-note news-why">
          <span>ℹ</span>
          {/* 和文の途中で JSX の改行が空白にならないよう、1つの文字列にする */}
          <div>
            {'モデルの特徴量に入っていない材料（直近のニュース・会社とアナリストの業績予想・個人投資家の声）を、'
              + '買う前に確かめるためのリンクです。今の規則を通る候補は、次の決算発表が近い銘柄に偏っています'
              + `（過去の検証で、規則を通った行の約8割は、予測日の翌営業日から${STRATEGY.holdDays}営業日の中に決算短信が出ていた）。`
              + '決算の前後は上にも下にも大きく動くので、直近の開示と、会社予想とアナリスト予想の差を見てから判断してください。'
              + 'リンク先は外部のサイトで、この画面は中身を取り込んでいません。'}
          </div>
        </div>

        {rows.length === 0 && <div className="empty">この日は新規の高値更新がありません。</div>}
        {rows.length > 0 && shown.length === 0 && (
          <div className="empty">{`この日は${RULE_JA}が${STRATEGY.agreePct}以上の候補がありません。`}</div>
        )}

        <div className="news-list">
          {shown.map((c) => (
            <NewsCard key={c.jqCode} c={c} pass={passIds.has(c.jqCode)} pick={pickIds.has(c.jqCode)} />
          ))}
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>市場全体</h2>
          <span className="sub">地合いと需給の目安（銘柄によらない）</span>
        </div>
        <div className="news-links news-market">
          {MARKET_LINKS.map((l) => (
            <ExtLink key={l.id} href={l.url} site={l.site} label={l.label} />
          ))}
        </div>
      </section>

      <p className="sub news-foot">
        {'リンクの形は 2026-10-17 に実在を確かめたものだけです（Yahoo!ファイナンス・日経・株予報Pro・JPX はページを'
          + '実際に開いて確認、株探・みんかぶは検索結果にあるページで確認）。サイトの作りが変わって開けなくなったときは知らせてください。'
          + '有料の記事やログインが要るページがあります。'}
      </p>
    </div>
  );
}

/** 候補1件。見出し（コード・名前・市場・業種と印）と、塊ごとのリンク。 */
function NewsCard({ c, pass, pick }) {
  const groups = newsLinks(c);
  const earn = earningsNote(c, STRATEGY.holdDays);
  return (
    <article className="news-card">
      <div className="news-head">
        <div className="news-id">
          <strong><span className="num">{c.code}</span> {c.name}</strong>
          <span className="sub">{[c.market, c.sector].filter(Boolean).join(' / ')}</span>
        </div>
        <div className="news-badges">
          {pick && <span className="badge green" title="今日の戦略で買う候補（上位の枠に入った）">買う候補</span>}
          {pass && !pick && (
            <span className="badge blue" title={`${RULE_JA}がすべて${STRATEGY.agreePct}以上`}>
              {`${STRATEGY.agreePct}以上`}
            </span>
          )}
          {Number.isFinite(c.ruleMin) && (
            <span className="badge slate" title={`${RULE_JA}の百分位の最小（ブレイク予測の並び順の物差し）`}>
              最小 {fmt(c.ruleMin, 1)}
            </span>
          )}
          {earn && <span className={`badge ${earn.tone}`} title={earn.text}>{earn.label}</span>}
        </div>
      </div>
      {groups.length === 0 ? (
        <div className="sub">コードの形が違うので、リンクを作れません。</div>
      ) : (
        <div className="news-groups">
          {groups.map((g) => (
            <div key={g.id} className="news-group">
              <span className="lab">{g.label}</span>
              <div className="news-links">
                {g.links.map((l) => (
                  <ExtLink key={l.id} href={l.href} site={l.site} label={l.label} note={l.note} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

/** 外部サイトへのリンク。新しいタブで開き、参照元を渡さない。 */
function ExtLink({ href, site, label, note }) {
  return (
    <a className="news-link" href={href} target="_blank" rel="noopener noreferrer"
       title={note ? `${site} ${label}: ${note}` : `${site} ${label}`}>
      <span className="site">{site}</span>
      <span className="what">{label}</span>
      <span className="ext" aria-hidden="true">↗</span>
    </a>
  );
}
