import React, { useMemo, useState } from 'react';
import { MODES, toFlow, latestQuarter, share } from '../lib/earnings.js';
import { fmtDate, DASH } from '../lib/format.js';
import FlowSankey, { SankeyLegend } from './FlowSankey.jsx';

/**
 * 直近決算（決算短信のサマリー）の損益をサンキー図で描く。
 * 描画は FlowSankey。ここはモード（単四半期・累計・会社予想）の切り替えと、
 * 「原価・販管費は元データに無い」という断り書きを持つ。
 */
export default function EarningsSankey({ quarters }) {
  const [mode, setMode] = useState('q');
  const q = useMemo(() => latestQuarter(quarters), [quarters]);

  // その決算にどのモードの値が入っているかは銘柄ごとに違う。
  // 選べないモードはボタンごと落とす（押せるのに何も出ない、を避ける）
  const usable = useMemo(
    () => MODES.filter((m) => toFlow(q, m.id)), [q]);
  const active = usable.some((m) => m.id === mode) ? mode : usable[0]?.id;
  const flow = useMemo(() => toFlow(q, active), [q, active]);

  if (!q) {
    return (
      <div className="empty">
        この銘柄には決算データがありません。
        <div className="sub" style={{ marginTop: 6 }}>
          ETF・REIT など、決算短信を出さない銘柄では空になります。
        </div>
      </div>
    );
  }
  if (!flow) {
    return <div className="empty">直近決算に損益の数字が入っていません。</div>;
  }

  const { truncated, missing } = flow;
  return (
    <div className="sankey">
      <div className="sankey-modes">
        {usable.map((m) => (
          <button key={m.id} title={m.note}
                  className={`btn btn-ghost${m.id === active ? ' on' : ''}`}
                  onClick={() => setMode(m.id)}>
            {m.label}
          </button>
        ))}
        <span className="sub">
          {q.period || DASH}
          <span className="sep">/</span>
          {fmtDate(q.disclosedDate)} 開示
          {q.periodEnd && <> <span className="sep">/</span> {fmtDate(q.periodEnd)} 締め</>}
        </span>
      </div>

      <FlowSankey flow={flow} ariaLabel="直近決算の損益の流れ" shortLabel={shortLabel}>
        <SankeyLegend flow={flow} truncated={truncated} />
        <Note flow={flow} truncated={truncated} missing={missing} />
      </FlowSankey>
    </div>
  );
}

/** 長い名前は図の中では短くする。正式名は title 属性で読める。 */
function shortLabel(s) {
  return s.replace('（原価＋販管費）', '').replace('法人税等・特別損失', '税金・特別損失');
}

function Note({ flow, truncated, missing }) {
  const np = flow.steps.find((s) => s.key === 'np');
  return (
    <p className="sub sankey-note">
      帯の太さは売上高に対する割合です。<b>売上原価と販管費は決算短信の
      サマリーに無い</b>ため、その2つは「営業費用」として1本にまとめてあります
      （売上高 − 営業利益。引き算で出した値で、内訳は元データにありません）。
      {missing.includes('odp') && (
        <> この銘柄は経常利益の開示が無いため、営業利益から当期純利益まで
        1段でまとめています。</>
      )}
      {truncated && (
        <> <b>{flow.steps.find((s) => s.key === truncated)?.label}が赤字</b>の
        ため、そこで流れを止めています（サンキー図は負の流れを描けません）。</>
      )}
      {np && !truncated && (
        <> 売上高100円あたり <b className="num">{share(np.value, flow.sales)?.toFixed(1)}円</b> が
        最終的に残りました。</>
      )}
    </p>
  );
}
