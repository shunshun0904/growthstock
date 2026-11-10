import React, { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { aiStockCode, buildAiContext, LIMITS, METRICS, PRESET_QUESTIONS, publicAiStocks,
  requestAnalysis, workerUrl } from '../lib/plamo.js';

export default function AiView({ dataset, predictionData, selectedId, featureNames }) {
  const options = useMemo(() => publicAiStocks(dataset, predictionData), [dataset, predictionData]);
  const [code, setCode] = useState('');
  const [question, setQuestion] = useState(PRESET_QUESTIONS[0]);
  const [password, setPassword] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');
  const [copyState, setCopyState] = useState('');
  const [loading, setLoading] = useState(false);
  const busy = useRef(false), abort = useRef(null);
  useEffect(() => {
    const selected = aiStockCode(selectedId);
    setCode(current => options.some(s => s.stockCode === selected) ? selected
      : options.some(s => s.stockCode === current) ? current : options[0]?.stockCode || '');
  }, [selectedId, options]);
  useEffect(() => () => abort.current?.abort(), []);
  const connection = useMemo(() => {
    try { return { url: workerUrl() }; } catch (e) { return { error: e.message }; }
  }, []);
  const context = useMemo(() => {
    if (!code) return { error: '分析対象の公開銘柄データを読み込むと選択できます。' };
    try { return { value: buildAiContext({ dataset, predictionData, stockCode: code, question, featureNames }) }; }
    catch (e) { return { error: e.message }; }
  }, [code, dataset, predictionData, question, featureNames]);
  const bytes = context.value ? new TextEncoder().encode(JSON.stringify(context.value)).byteLength : 0;
  const choose = value => { setCode(value); setResponse(null); setError(''); setCopyState(''); };
  async function submit(event) {
    event.preventDefault();
    if (busy.current) return;
    setError(''); setCopyState('');
    if (connection.error || context.error) { setError(connection.error || context.error); return; }
    busy.current = true; setLoading(true); setResponse(null);
    abort.current = new AbortController();
    const sentContext = context.value, sentQuestion = question.trim();
    try {
      const answer = await requestAnalysis({ workerUrl: connection.url, password, question: sentQuestion,
        context: sentContext, signal: abort.current.signal });
      setResponse({ ...answer, stockName: sentContext.stockName, stockCode: sentContext.stockCode,
        asOf: sentContext.asOf, question: sentQuestion });
    } catch (e) { if (e.name !== 'AbortError') setError(e.message); }
    finally { busy.current = false; setLoading(false); }
  }
  async function copy() {
    try { await navigator.clipboard.writeText(response.answer); setCopyState('コピーしました'); }
    catch { setCopyState('コピーできませんでした。回答を選択してコピーしてください。'); }
  }
  return <div className="ai-page">
    <section className="card">
      <div className="card-head"><h2>AI銘柄分析</h2><span className="badge blue">PLaMo 3.0 Fin</span></div>
      <p className="ai-intro">公開済みの財務指標や予測結果をもとに、日本語で分析・解説します。</p>
      <p className="ai-disclosure">質問と下記の銘柄データをPLaMoへ送信します。無料プランでは入力・出力が保存され、学習に利用されます。
        非公開の売買履歴・個人情報・認証情報は質問に入力しないでください。手入力銘柄のデータは送信しません。
        {' '}<a href="https://docs.plamo.preferredai.jp/ja/pricing" target="_blank" rel="noopener noreferrer">データの取り扱い</a>
      </p>
      {connection.error && <div className="banner info" role="status"><div>
        <strong>AI接続先が未設定です</strong><div>{connection.error}</div>
        <a href="https://github.com/shunshun0904/growthstock/blob/HEAD/docs/PLAMO_AI.md" target="_blank" rel="noopener noreferrer">接続の設定手順</a>
      </div></div>}
    </section>
    <div className="ai-layout">
      <section className="card">
        <form onSubmit={submit} className="ai-form">
          <label className="ai-label" htmlFor="ai-stock">分析対象銘柄 <span className="sub">公開データ {options.length}銘柄</span></label>
          <select id="ai-stock" value={code} onChange={e => choose(e.target.value)} disabled={loading || !options.length}>
            {!options.length && <option value="">公開データの読み込み待ち</option>}
            {options.map(s => <option key={s.stockCode} value={s.stockCode}>{s.stockCode}　{s.name}</option>)}
          </select>
          {context.value && <div className="ai-facts">
            <strong>{context.value.stockName} <span className="num">{context.value.stockCode}</span></strong>
            <span>データ基準日 {context.value.asOf}</span>
            {context.value.metrics.filter(m => ['price', 'roe', 'pbr'].includes(m.key)).map(m => <span key={m.key}>
              {METRICS[m.key].label} <b className="num">{m.value.toLocaleString('ja-JP', { maximumFractionDigits: 2 })}</b>
              {METRICS[m.key].unit} <small>({m.asOf})</small>
            </span>)}
          </div>}
          <label className="ai-label" htmlFor="ai-question">質問 <span className="sub">{question.length}/{LIMITS.questionChars}文字</span></label>
          <textarea id="ai-question" rows={5} maxLength={LIMITS.questionChars} value={question} disabled={loading}
            onChange={e => { setQuestion(e.target.value); setError(''); }}
            placeholder="ROEやPBR、予測順位など、確認したい内容を入力してください。" required />
          <div className="ai-presets" aria-label="質問例">
            {PRESET_QUESTIONS.map(q => <button key={q} type="button" className="btn btn-ghost" disabled={loading}
              onClick={() => { setQuestion(q); setError(''); }}>{q}</button>)}
          </div>
          <label className="ai-label" htmlFor="ai-password">利用パスワード</label>
          <input id="ai-password" type="password" value={password} minLength={32} maxLength={256}
            autoComplete="off" spellCheck={false} autoCapitalize="none" disabled={loading}
            placeholder="管理者が設定した本人認証用パスワード" onChange={e => setPassword(e.target.value)} required />
          <div className="ai-password-note">このタブを開いている間だけ保持します。ブラウザの保存領域には記録しません。</div>
          <div className="ai-actions">
            <button type="submit" className="btn btn-primary" disabled={loading || !!connection.error || !!context.error || !question.trim() || !password}>
              {loading ? '分析しています…' : 'PLaMoに質問する'}
            </button>
            <button type="button" className="btn" disabled={loading || !password} onClick={() => setPassword('')}>パスワードを消去</button>
          </div>
          {context.error && <p className="ai-muted" role="status">{context.error}</p>}
          {error && <div className="banner warn" role="alert">{error}</div>}
        </form>
      </section>
      <aside className="card ai-context">
        <div className="card-head"><h2>送信するデータ</h2><span className="sub">{(bytes / 1024).toFixed(1)} KB</span></div>
        <p className="ai-muted">質問に合わせて、選んだ1銘柄の指標・予測・寄与の一部だけを抽出します。取得できない数値は補いません。</p>
        {context.value && <>
          <div className="ai-tags">
            <span className="chip">指標 {context.value.metrics.length}件</span>
            {context.value.prediction && <span className="chip">モデル別スコア・順位</span>}
            {context.value.explanation && <span className="chip">基準LightGBMのSHAP</span>}
            {!!context.value.historicalSnapshots?.length && <span className="chip">過去観測 {context.value.historicalSnapshots.length}件</span>}
          </div>
          <ul className="ai-limitations">{context.value.limitations.map(note => <li key={note}>{note}</li>)}</ul>
          <details><summary>送信内容を確認</summary><pre>{JSON.stringify(context.value, null, 2)}</pre></details>
        </>}
      </aside>
    </div>
    <section className="card ai-answer" aria-busy={loading} aria-live="polite">
      <div className="card-head"><h2>AIの回答</h2>{response && <button type="button" className="btn" onClick={copy}>回答をコピー</button>}</div>
      {loading ? <p role="status">PLaMoの回答を待っています。再送信せず、このままお待ちください。</p>
        : !response ? <p className="ai-muted">銘柄と質問を選んで送信すると、ここに回答を表示します。</p>
          : <>
            <div className="ai-answer-meta">{response.stockCode} {response.stockName} ／ データ基準日 {response.asOf} ／ {response.model}</div>
            <p className="ai-asked">質問：{response.question}</p>
            <div className="ai-markdown"><ReactMarkdown skipHtml disallowedElements={['img']}
              components={{ a: props => <a href={props.href} target="_blank" rel="noopener noreferrer">{props.children}</a> }}>
              {response.answer}
            </ReactMarkdown></div>
            {response.truncated && <p className="ai-muted">回答が長さの上限に達しました。必要な項目に絞って再度質問してください。</p>}
            {response.usage && <p className="ai-muted">利用回数（UTC）：本日 {response.usage.callsToday}/{response.usage.dailyLimit}回 ／
              今月 {response.usage.callsThisMonth}/{response.usage.monthlyLimit}回。通信失敗した試行も回数に含みます。</p>}
            <p className="ai-muted">AIの説明には誤りが含まれる場合があります。数値と基準日を確認してご利用ください。</p>
          </>}
      {copyState && <p role="status">{copyState}</p>}
    </section>
  </div>;
}
