import React, { useMemo, useState } from 'react';
import FlowSankey, { SankeyLegend } from '../components/FlowSankey.jsx';
import EarningsSankey from '../components/EarningsSankey.jsx';
import { toDetailedFlow, periodLabel, DOC_TYPE_JA } from '../lib/filings.js';
import { fmtYen, share } from '../lib/earnings.js';
import { fmtDate, fmtDateTime, DASH } from '../lib/format.js';

/**
 * 決算サンキーのタブ。
 *
 * 候補銘柄（ブレイク予測の候補と、8軸へ送った銘柄）の損益を2つの流れで見る。
 *   1. 有価証券報告書・半期報告書（EDINET）。売上原価・販管費・営業外・特別損益・法人税等まで分かれる
 *   2. 直近の決算短信（J-Quants）。四半期の刻みだが、原価・販管費は無い（営業費用1本）
 * 第1・第3四半期の明細は扱わない（四半期報告書は 2024年4月以降の四半期から廃止。決算短信の明細を
 * 自動で取ってよい公式の経路が無い）。
 *
 * データは public/data/filings.json（Fetch Filings が毎晩、予測の後に EDINET から作る）。
 * 出典は EDINET（金融庁）。公共データ利用規約（PDL1.0）に従い、出典を表示する。
 */
export default function FilingsView({ pred, rows, filings }) {
  const choices = useMemo(() => candidates(pred, rows), [pred, rows]);
  const [picked, setPicked] = useState(null);
  const current = choices.find((c) => c.jqCode === picked) || choices[0] || null;
  const doc = current ? filings?.docs?.[current.jqCode] : null;
  const stock = current ? rows.find((r) => r.jqCode === current.jqCode || r.code === current.code) : null;

  return (
    <div className="filings">
      <section className="card">
        <div className="card-head">
          <h2>決算サンキー</h2>
          <span className="sub">
            有価証券報告書・半期報告書（EDINET）の損益を、売上原価・販管費まで分けて流れで見ます
          </span>
        </div>

        {choices.length === 0 ? (
          <div className="empty">
            候補の銘柄がありません。ブレイク予測の候補か、8軸へ送った銘柄がここに並びます。
          </div>
        ) : (
          <div className="filings-pick">
            <label className="lab" htmlFor="filings-stock">銘柄</label>
            <select id="filings-stock" className="filings-select"
                    value={current?.jqCode || ''} onChange={(e) => setPicked(e.target.value)}>
              {choices.map((c) => (
                <option key={c.jqCode} value={c.jqCode}>
                  {c.name || c.code}（{c.code}）{c.date ? ` ${fmtDate(c.date)} の候補` : ' 8軸の銘柄'}
                </option>
              ))}
            </select>
            <span className="sub">
              {filings?.generatedAt
                ? <>取り込み {fmtDateTime(filings.generatedAt)}</>
                : '有報・半期報告書はまだ取り込まれていません'}
            </span>
          </div>
        )}
      </section>

      {current && (
        <section className="card">
          <div className="card-head">
            <h2>{current.name || current.code} <span className="sub num">{current.code}</span></h2>
            <span className="sub">有価証券報告書・半期報告書の損益（EDINET）</span>
          </div>
          <FilingSankey doc={doc} filings={filings} />
        </section>
      )}

      {current && (
        <section className="card">
          <div className="card-head">
            <h2>直近の決算短信</h2>
            <span className="sub">四半期の刻み（J-Quants）。原価・販管費は元データに無い</span>
          </div>
          {stock ? (
            <EarningsSankey quarters={stock.quarters} />
          ) : (
            <div className="empty">
              四半期の数字は、ブレイク予測の候補を「8軸で見る」で送ると取れます。
            </div>
          )}
        </section>
      )}
    </div>
  );
}

/** 選べる銘柄。予測の候補（新しい日・上位が先）と、8軸の銘柄。同じ銘柄は1つに。 */
export function candidates(pred, rows) {
  const out = [];
  const seen = new Set();
  const cands = [...(pred?.candidates || [])]
    .sort((a, b) => (b.date > a.date ? 1 : b.date < a.date ? -1 : (a.rankInDay ?? 0) - (b.rankInDay ?? 0)));
  for (const c of cands) {
    if (!c.jqCode || seen.has(c.jqCode)) continue;
    seen.add(c.jqCode);
    out.push({ jqCode: c.jqCode, code: c.code, name: c.name, date: c.date });
  }
  for (const r of rows || []) {
    const jq = r.jqCode || (r.code ? `${r.code}0` : null);
    if (!jq || seen.has(jq)) continue;
    seen.add(jq);
    out.push({ jqCode: jq, code: r.code, name: r.name, date: null });
  }
  return out;
}

function FilingSankey({ doc, filings }) {
  const flow = useMemo(() => toDetailedFlow(doc), [doc]);
  if (!filings) {
    return (
      <div className="empty">
        有報・半期報告書はまだ取り込まれていません。
        <div className="sub" style={{ marginTop: 6 }}>
          毎晩の予測の後に <code>Fetch Filings</code> が EDINET から取ります。
        </div>
      </div>
    );
  }
  if (!doc) {
    return (
      <div className="empty">
        この銘柄の有報・半期報告書はまだありません。
        <div className="sub" style={{ marginTop: 6 }}>
          次の取り込み（毎晩）で探します。上場して間もない銘柄・ETF・REIT は見つからないことがあります。
        </div>
      </div>
    );
  }
  if (!flow) {
    return (
      <div className="empty">
        この書類の損益は図にできません（会計基準 {doc.standard || DASH}）。
        <div className="sub" style={{ marginTop: 6 }}>
          米国基準・銀行・保険など、売上高から始まる形でない損益計算書は対象外です。
        </div>
      </div>
    );
  }
  const own = flow.steps[flow.steps.length - 1];
  const prior = doc.priorItems || {};
  const salesKey = Object.keys(doc.items || {}).find((k) => doc.items[k] === flow.sales);
  const yoy = salesKey && Number.isFinite(prior[salesKey]) && prior[salesKey] > 0
    ? (flow.sales / prior[salesKey] - 1) * 100 : null;
  return (
    <div className="sankey">
      <div className="sankey-modes filings-meta">
        <span className="badge blue">{DOC_TYPE_JA[doc.docType] || `書類 ${doc.docType}`}</span>
        <span className="sub">
          {periodLabel(doc)}
          <span className="sep">/</span>
          {fmtDate(doc.submitDate)} 提出
          <span className="sep">/</span>
          {doc.standard || DASH}・{doc.consolidated ? '連結' : '個別'}
          {yoy != null && (
            <> <span className="sep">/</span> {flow.salesLabel} 前期比 <b className="num">{yoy >= 0 ? '+' : ''}{yoy.toFixed(1)}%</b></>
          )}
        </span>
      </div>
      <FlowSankey flow={flow} ariaLabel="有価証券報告書の損益の流れ" shortLabel={shortLabel}>
        <SankeyLegend flow={flow} truncated={flow.truncated} />
        <p className="sub sankey-note">
          帯の太さは{flow.salesLabel}に対する割合です。
          {flow.derived.length > 0 && (
            <> 「…等」の段は、元データに出ていく項目が無いので引き算で出しています。</>
          )}
          {flow.missing.length > 0 && (
            <> 開示に無い段（{flow.missing.length}つ）は飛ばしています。</>
          )}
          {flow.truncated && (
            <> <b>{flow.steps.find((s) => s.key === flow.truncated)?.label}が赤字</b>のため、
            そこで流れを止めています。</>
          )}
          {own && !flow.truncated && (
            <> {flow.salesLabel}100円あたり <b className="num">{share(own.value, flow.sales)?.toFixed(1)}円</b> が
            {own.label}として残りました。</>
          )}
          {doc.checks && Object.values(doc.checks).some((v) => v === false) && (
            <> <b>注意:</b> 段の足し算が元データと一致しない箇所があります（開示の丸め・表示科目の違い）。</>
          )}
        </p>
        <p className="sub sankey-note">
          出典: <a href={filings.source?.url || 'https://disclosure2dl.edinet-fsa.go.jp/'} target="_blank" rel="noreferrer">
            {filings.source?.name || 'EDINET（金融庁）'}
          </a>
          の書類（{doc.docID}）をもとに作成（{filings.source?.license || '公共データ利用規約（PDL1.0）'}）。
          金額は開示の値（{fmtYen(flow.sales)} など）で、百万円単位に丸めてある会社があります。
        </p>
      </FlowSankey>
    </div>
  );
}

/** 図の中では短く（列の間隔は約 150px）。正式名は title 属性で読める。 */
const SHORT = [
  ['販売費及び一般管理費', '販管費'], ['税引前当期純利益', '税引前利益'],
  ['親会社株主に帰属する当期純利益', '親会社株主帰属'], ['親会社の所有者に帰属する当期利益', '親会社帰属'],
  ['非支配株主に帰属する当期純利益', '非支配株主'], ['非支配持分に帰属する当期利益', '非支配持分'],
  ['販管費・その他の費用', '販管費等'],
];
function shortLabel(s) {
  for (const [a, b] of SHORT) s = s.replace(a, b);
  return s;
}
