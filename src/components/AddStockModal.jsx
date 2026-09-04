import React, { useState } from 'react';

const FIELDS = [
  { key: 'code', label: '証券コード', type: 'text', required: true, placeholder: '6920' },
  { key: 'name', label: '銘柄名', type: 'text', required: true, placeholder: 'レーザーテック' },
  { key: 'sector', label: '業種', type: 'text', placeholder: '電気機器' },
  { key: 'price', label: '株価', unit: '円' },
  { key: 'highRatio', label: '52週高値接近率', unit: '%' },
  { key: 'tradingValue', label: '売買代金', unit: '億円' },
  { key: 'marketCap', label: '時価総額', unit: '億円' },
  { key: 'volumeTrend', label: '出来高モメンタム', unit: '% (20日平均比)' },
  { key: 'epsGrowth', label: '四半期EPS成長率', unit: '%' },
  { key: 'salesGrowth', label: '四半期売上成長率', unit: '%' },
  { key: 'roe', label: 'ROE', unit: '%' },
  { key: 'opMargin', label: '営業利益率', unit: '%' },
  { key: 'creditRatio', label: '信用倍率', unit: '倍' },
  { key: 'progressRate', label: '決算進捗率', unit: '%' },
  { key: 'quarter', label: '経過四半期', unit: '1〜4' },
];

/** 動的銘柄追加モーダル (仕様書 §5.4) */
export default function AddStockModal({ onClose, onSubmit, initial = {} }) {
  const [form, setForm] = useState(() =>
    Object.fromEntries(FIELDS.map((f) => [f.key, initial[f.key] ?? '']))
  );
  const [note, setNote] = useState(initial.note ?? '');
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const valid = form.code.trim() !== '' && form.name.trim() !== '';

  const submit = (e) => {
    e.preventDefault();
    if (!valid) return;
    onSubmit({ ...form, note, id: initial.id });
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form className="modal" onSubmit={submit}>
        <h2>銘柄を追加</h2>
        <p className="hint">
          空欄にした指標は「データなし」として扱われ、その軸はスコアの平均から除外されます。
          推定値で埋めるより、分からない項目は空欄のままにしてください。
        </p>

        <div className="field-grid">
          {FIELDS.map((f) => (
            <div className="field" key={f.key}>
              <label htmlFor={`f-${f.key}`}>
                {f.label}{f.required && <span style={{ color: 'var(--red)' }}> *</span>}
                {f.unit && <span className="unit"> ({f.unit})</span>}
              </label>
              <input
                id={`f-${f.key}`}
                type={f.type === 'text' ? 'text' : 'number'}
                step="any"
                inputMode={f.type === 'text' ? 'text' : 'decimal'}
                placeholder={f.placeholder || ''}
                value={form[f.key]}
                onChange={set(f.key)}
              />
            </div>
          ))}
        </div>

        <div className="field" style={{ marginTop: 'var(--s4)' }}>
          <label htmlFor="f-note">定性メモ</label>
          <textarea
            id="f-note" rows={3} value={note} onChange={(e) => setNote(e.target.value)}
            placeholder="例: 主力製品の値上げ効果が3Qから顕在化。空売り比率が上昇中。"
          />
        </div>

        <div className="row" style={{ justifyContent: 'flex-end', marginTop: 'var(--s5)' }}>
          <button type="button" className="btn" onClick={onClose}>キャンセル</button>
          <button type="submit" className="btn btn-primary" disabled={!valid}>追加する</button>
        </div>
      </form>
    </div>
  );
}
