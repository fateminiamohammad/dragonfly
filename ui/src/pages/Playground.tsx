import { useState } from 'react';
import { api, type DecideResponse } from '../api';
import { headline, pct, topOptions } from '../format';
import { fileToBase64, mediaType } from './Media';

const EXAMPLE = {
  state: {
    message: "I ordered a new card two weeks ago and it still hasn't arrived. This is really frustrating.",
    customer: { plan: 'premium', account_age_days: 812 },
  },
  questions: {
    intent: {
      type: 'choice',
      instructions: 'Which banking intent best describes this customer message?',
      criteria: {
        card_arrival: 'Customer asks where their card is',
        card_not_working: "Customer's card is declined or broken",
        refund_not_showing_up: 'Customer is waiting for a refund',
        change_pin: null,
      },
    },
    escalate: { type: 'noul', instructions: 'Should this be escalated to a human agent right away?' },
    sentiment: {
      type: 'score',
      instructions: 'How does the customer feel?',
      criteria: ['very negative', 'negative', 'neutral', 'positive', 'very positive'],
    },
  },
};

export function Playground() {
  const [text, setText] = useState(JSON.stringify(EXAMPLE, null, 2));
  const [result, setResult] = useState<DecideResponse | null>(null);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  /** Put an image/audio file into the request's state as a media object; the server converts it to text. */
  async function attach(file: File | undefined) {
    if (!file) return;
    const kind = mediaType(file);
    if (!kind) {
      setError('Attach an image or an audio file.');
      return;
    }
    try {
      const body = JSON.parse(text);
      const state = typeof body.state === 'object' && body.state !== null && !Array.isArray(body.state) ? body.state : { text: body.state };
      let n = 1;
      while (`attachment_${n}` in state) n++;
      state[`attachment_${n}`] = { type: kind, name: file.name, data: await fileToBase64(file) };
      setText(JSON.stringify({ ...body, state }, null, 2));
      setError('');
    } catch (e) {
      setError(`Fix the request JSON first: ${(e as Error).message}`);
    }
  }

  async function run() {
    setError('');
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch (e) {
      setError(`Invalid JSON: ${(e as Error).message}`);
      return;
    }
    setBusy(true);
    const t = performance.now();
    try {
      setResult(await api<DecideResponse>('/model/playground', { method: 'POST', body: JSON.stringify(body) }));
      setRoundTrip(performance.now() - t);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="split">
      <section className="card">
        <div className="row">
          <h2>Request</h2>
          <span className="muted small">POST /v1/systemone</span>
        </div>
        <textarea spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} aria-label="request JSON" />
        <div className="row">
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? 'Deciding…' : 'Decide'}
          </button>
          <button onClick={() => setText(JSON.stringify(EXAMPLE, null, 2))}>Reset example</button>
          <label className="button">
            Attach image/audio
            <input type="file" accept="image/*,audio/*" hidden onChange={(e) => attach(e.target.files?.[0])} />
          </label>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <div className="row">
          <h2>Answers</h2>
          {result && (
            <span className="muted small">
              model {result.latency_ms.toFixed(1)} ms{result.cached ? ' (cached)' : ''}
              {result.usage.media_ms ? ` · media ${result.usage.media_ms.toFixed(0)} ms` : ''} · round trip{' '}
              {roundTrip?.toFixed(0)} ms · {result.usage.input_tokens} tokens
            </span>
          )}
        </div>
        {!result && <p className="muted">Run a request to see every answer's probabilities.</p>}
        {result &&
          Object.entries(result.answers).map(([id, a]) => (
            <div key={id} className="answer">
              <div className="row">
                <strong>{id}</strong>
                <span className="pill">{a.type}</span>
                {a.tier && <span className="pill">tier {a.tier}</span>}
                <span className="grow" />
                <span className="muted small">confidence {pct(a.confidence)}</span>
              </div>
              <div className="headline">{headline(a)}</div>
              {topOptions(a).map((o) => (
                <div key={o.label} className="bar-row" title={`${o.label}: ${pct(o.p)}`}>
                  <span className="bar-label">{o.label}</span>
                  <span className="bar">
                    <span style={{ width: `${Math.max(o.p * 100, 0.5)}%` }} />
                  </span>
                  <span className="bar-value">{pct(o.p)}</span>
                </div>
              ))}
            </div>
          ))}
      </section>
    </div>
  );
}
