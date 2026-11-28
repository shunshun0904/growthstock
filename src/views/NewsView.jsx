import React, { useMemo, useState } from 'react';
import { fmt, fmtDate } from '../lib/format.js';
import { earningsNote } from '../lib/predictions.js';
import { STRATEGY, RULE_JA, NEWS_LTR_TOP, newsRows, orderRows, strategySignal } from '../lib/strategy.js';
import { stockLinks } from '../lib/news.js';

/**
 * 候補銘柄のニュース（運用者の依頼 2026-10-17。左から2番目のタブ）。
 *
 * 出すのは今日の戦略の候補（規則の3本すべて95以上。2026-10-30 から 1.2σ の3本。運用者の指示「取得対象は、候補銘柄のみで良いです」）と、
 * 2026-11-21 からは規則を通らなくてもランキング学習（日次）がその日の3位以内の候補（strategy.js の newsRows。運用者の依頼「候補銘柄の
 * ニュースの探索条件を緩くしてほしいです…ランキング学習モデルが良ければ買うことはあり得るので」→ 選択「日次ランキングの3位以内」。
 * 足した候補に理由の印は付けない（選択「付けない」）。規則を通らない候補には「95以上」「買う候補」の印も出さない）。
 * リンクは public/data/news_links.json（毎晩の予測の後に research/news_links.py が候補ごとにページを開いて確かめた結果）
 * から並べる。確かめは裏でやり、画面には印や状態を出さない（運用者の指示「確かめた、未確認などの表示はいらないです。
 * バックでやってくれれば良いので」）。確かめを通らなかったリンクは出さない。自動では開かないサイトのリンクも、同じ塊に
 * 印なしで並べる（決定「全部を印なしで出す」）。市場全体のカードは外した（指示「市場全体のカードも不要です」）。
 * 見出しや本文は取り込まない。
 */
export default function NewsView({ data, links }) {
  const dates = data?.dates || [];
  const [day, setDay] = useState(() => data?.asOf || dates[dates.length - 1]);

  const rows = useMemo(
    () => orderRows((data?.candidates || []).filter((c) => c.date === day)),
    [data, day]
  );
  const signal = useMemo(() => strategySignal(rows), [rows]);
  const pickIds = useMemo(() => new Set(signal.picks.map((c) => c.jqCode)), [signal]);
  const ruleIds = useMemo(() => new Set(signal.passed.map((c) => c.jqCode)), [signal]);
  const shown = useMemo(() => newsRows(rows, signal), [rows, signal]);

  return (
    <div className="news">
      <section className="card">
        <div className="card-head">
          <h2>候補銘柄のニュース</h2>
          <span className="sub">
            {`今日の戦略の候補（${RULE_JA}がすべて${STRATEGY.agreePct}以上）と、ランキング学習（日次）がその日の${NEWS_LTR_TOP}位以内の`
              + '候補ごとに、ニュース・適時開示・アナリスト予想・'
              + '個人投資家の声のページを開けます（外部サイト）'}
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
            <span className="lab">候補</span>
            <strong className="num">{shown.length}</strong> 銘柄
            <span className="sub">（この日の高値更新 {rows.length} 銘柄のうち）</span>
          </div>
        </div>

        <div className="banner info pred-note news-why">
          <span>ℹ</span>
          {/* 和文の途中で JSX の改行が空白にならないよう、1つの文字列にする */}
          <div>
            {'モデルの特徴量に入っていない材料（直近のニュース・会社とアナリストの業績予想・個人投資家の声）を、'
              + '買う前に確かめるためのリンクです。今の規則を通る候補は、次の決算発表が近い銘柄に偏っています'
              + `（過去の検証で、規則を通った行の約8割は、予測日の翌営業日から${STRATEGY.holdDays}営業日の中に決算短信が出ていた）。`
              + '決算の前後は上にも下にも大きく動くので、直近の開示と、会社予想とアナリスト予想の差を見てから判断してください。'}
          </div>
        </div>

        {rows.length === 0 && <div className="empty">この日は新規の高値更新がありません。</div>}
        {rows.length > 0 && shown.length === 0 && (
          <div className="empty">
            {`この日は、今日の戦略の候補（${RULE_JA}がすべて${STRATEGY.agreePct}以上）も、ランキング学習（日次）の${NEWS_LTR_TOP}位以内の`
              + '候補もありません。'}
          </div>
        )}

        <div className="news-list">
          {shown.map((c) => (
            <NewsCard key={c.jqCode} c={c} pick={pickIds.has(c.jqCode)} rule={ruleIds.has(c.jqCode)} links={links} />
          ))}
        </div>
      </section>

      <div className="sub news-foot">
        <p>有料の記事やログインが要るページがあります。リンクが開けなかったときは知らせてください。</p>
      </div>
    </div>
  );
}

/**
 * 候補1件。見出し（コード・名前・市場・業種と印）と、見出しの塊ごとのリンク。印（買う候補・95以上・最小）は今の規則を満たす候補
 * （rule）だけに出す。ランキング学習（日次）の順位で足した候補には印を出さない（運用者の選択 2026-11-21「付けない」）。
 */
function NewsCard({ c, pick, rule, links }) {
  const g = stockLinks(links, c);
  const earn = earningsNote(c, STRATEGY.holdDays);
  return (
    <article className="news-card">
      <div className="news-head">
        <div className="news-id">
          <strong><span className="num">{c.code}</span> {c.name}</strong>
          <span className="sub">{[c.market, c.sector].filter(Boolean).join(' / ')}</span>
        </div>
        <div className="news-badges">
          {pick ? <span className="badge green" title="今日の戦略で買う候補（上位の枠に入った）">買う候補</span>
            : rule && <span className="badge blue" title={`${RULE_JA}がすべて${STRATEGY.agreePct}以上（上位の枠の外）`}>
              {`${STRATEGY.agreePct}以上`}
            </span>}
          {rule && Number.isFinite(c.minPct) && (
            <span className="badge slate" title={`${RULE_JA}の百分位の最小（ブレイク予測の並び順の物差し）`}>
              最小 {fmt(c.minPct, 1)}
            </span>
          )}
          {earn && <span className={`badge ${earn.tone}`} title={earn.text}>{earn.label}</span>}
        </div>
      </div>

      {links === undefined ? <div className="sub">リンクを読み込んでいます…</div>
        : !g.known ? <div className="sub">この銘柄のリンクはまだありません（毎晩の予測の後に用意します）。</div>
          : g.groups.length === 0 ? <div className="sub">出せるリンクがありません。</div>
            : (
              <div className="news-groups">
                {g.groups.map((grp) => (
                  <div key={grp.id} className="news-group">
                    <span className="lab">{grp.label}</span>
                    <div className="news-links">
                      {grp.links.map((l) => <ExtLink key={l.id} l={l} />)}
                    </div>
                  </div>
                ))}
              </div>
            )}
    </article>
  );
}

/** 外部サイトへのリンク。新しいタブで開き、参照元を渡さない。 */
function ExtLink({ l }) {
  return (
    <a className="news-link" href={l.href} target="_blank" rel="noopener noreferrer"
       title={[`${l.site} ${l.label}`, l.note].filter(Boolean).join(' / ')}>
      <span className="site">{l.site}</span>
      <span className="what">{l.label}</span>
    </a>
  );
}
