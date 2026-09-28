import { useEffect, useState } from 'react';
import { api, getToken } from '../api';
import { pct } from '../format';

interface ReviewItem {
  id: string;
  created: number;
  state: unknown;
  question_id: string;
  question: { type: 'choice' | 'noul' | 'score'; instructions: unknown; criteria: unknown };
  answer: { probabilities?: Record<string, number>; noul?: number; confidence?: number; choice?: string; score?: number };
  options: string[];
}

function asText(v: unknown): string {
  return typeof v === 'string' ? v : JSON.stringify(v, null, 2);
}

/** What the model thought each option's probability was. */
function modelProbs(item: ReviewItem): Record<string, number> {
  if (item.question.type === 'noul') return { false: 1 - (item.answer.noul ?? 0.5), true: item.answer.noul ?? 0.5 };
  return item.answer.probabilities ?? {};
}

function optionLabel(item: ReviewItem, key: string): string {
  const c = item.question.criteria;
  if (item.question.type === 'score' && Array.isArray(c)) return `${key} · ${asText(c[Number(key)])}`;
  if (item.question.type === 'noul') return key === 'true' ? 'yes' : 'no';
  return key;
}

export function Review() {
  const [data, setData] = useState<{ pending: number; reviewed: number; items: ReviewItem[] } | null>(null);
  const [error, setError] = useState('');

  const load = () =>
    api<{ pending: number; reviewed: number; items: ReviewItem[] }>('/review?limit=20')
      .then(setData)
      .catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  async function decide(item: ReviewItem, label: string | null) {
    setError('');
    try {
      await api(`/review/${item.id}${label === null ? '/skip' : ''}`, {
        method: 'POST',
        body: label === null ? undefined : JSON.stringify({ label }),
      });
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function exportJsonl() {
    const res = await fetch('/api/review/export', { headers: { authorization: `Bearer ${getToken()}` } });
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dragonfly-reviewed.jsonl';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <div className="stats">
        <div className="card stat">
          <span className="muted small">Waiting for review</span>
          <strong>{data?.pending ?? '–'}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Reviewed (training data)</span>
          <strong>{data?.reviewed ?? '–'}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Learning loop</span>
          <button onClick={exportJsonl} disabled={!data?.reviewed}>
            Export training JSONL
          </button>
        </div>
      </div>
      <p className="muted small">
        Decisions the model was unsure about land here when the <code>human-review</code> plugin is enabled. Your answers
        become training data: export them and train a specialist, or add them to the next training mix.
      </p>
      {error && <p className="error">{error}</p>}
      {data && data.items.length === 0 && <p className="muted">Nothing to review right now.</p>}
      {data?.items.map((item) => {
        const probs = modelProbs(item);
        return (
          <section key={item.id} className="card review">
            <div className="row">
              <strong>{item.question_id}</strong>
              <span className="pill">{item.question.type}</span>
              <span className="grow" />
              <span className="muted small">{new Date(item.created * 1000).toLocaleString()}</span>
            </div>
            {item.question.instructions != null && <p className="headline">{asText(item.question.instructions)}</p>}
            <pre className="ocr">{asText(item.state)}</pre>
            <p className="muted small">The model's guess (it was unsure). Pick the right answer:</p>
            {item.options.map((key) => (
              <div key={key} className="bar-row review-option">
                <button onClick={() => decide(item, key)}>{optionLabel(item, key)}</button>
                <span className="bar">
                  <span style={{ width: `${Math.max((probs[key] ?? 0) * 100, 0.5)}%` }} />
                </span>
                <span className="bar-value">{pct(probs[key] ?? 0)}</span>
              </div>
            ))}
            <div className="row">
              <span className="grow" />
              <button onClick={() => decide(item, null)}>Skip</button>
            </div>
          </section>
        );
      })}
    </>
  );
}
