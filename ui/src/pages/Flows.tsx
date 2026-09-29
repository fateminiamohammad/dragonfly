import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { headline, pct } from '../format';
import type { Answer } from '../api';

interface Step {
  model?: string;
  state?: string;
  questions: Record<string, unknown>;
  next?: { if?: string; to: string }[];
}
interface Flow {
  name?: string;
  start?: string;
  steps: Record<string, Step>;
}
interface Trace {
  path: string[];
  latency_ms: number;
  mermaid: string;
  steps: { step: string; model: string; specialist?: string; answers: Record<string, Answer>; latency_ms: number; next: string | null }[];
}

const EXAMPLE: Flow = {
  start: 'classify',
  steps: {
    classify: {
      questions: {
        intent: {
          type: 'choice',
          instructions: 'What does the customer want?',
          criteria: { refund: 'wants money back', delivery: 'asks where the parcel is', other: null },
        },
      },
      next: [
        { if: 'intent.choice == refund and intent.confidence > 0.5', to: 'refund_risk' },
        { if: 'intent.choice == delivery', to: 'delivery' },
      ],
    },
    refund_risk: {
      state: '{{state}}\n\nThe customer asks for a refund.',
      questions: {
        escalate: { type: 'noul', instructions: 'Should a human approve this refund?' },
        mood: { type: 'score', criteria: ['angry', 'neutral', 'happy'] },
      },
    },
    delivery: { questions: { late: { type: 'noul', instructions: 'Is the parcel late?' } } },
  },
};
const EXAMPLE_STATE = 'I was charged twice for order 5512 and want my money back. Third time I write!';

/** Template: check an LLM's output before it reaches a user (grounded? on policy? right format? no private data?). */
const LLM_CHECK: Flow = {
  start: 'check',
  steps: {
    check: {
      questions: {
        grounded: { type: 'noul', instructions: 'Is every claim in the answer supported by the source text?' },
        on_policy: { type: 'noul', instructions: 'Does the answer follow the policy (no promises of refunds, no legal or medical advice)?' },
        format_ok: { type: 'noul', instructions: 'Is the answer a short, polite reply to the customer, without internal notes?' },
        leaks_data: { type: 'noul', instructions: "Does the answer reveal another customer's personal data?" },
      },
      next: [
        { if: 'leaks_data.noul > 0.5 or grounded.noul < 0.3 or on_policy.noul < 0.3', to: 'block' },
        { if: 'grounded.confidence < 0.5 or on_policy.confidence < 0.5', to: 'review' },
      ],
    },
    block: { questions: { severity: { type: 'score', instructions: 'How harmful would sending this answer be?', criteria: ['low', 'medium', 'high'] } } },
    review: { questions: { fixable: { type: 'noul', instructions: 'Could a small edit make the answer acceptable?' } } },
  },
};
const LLM_CHECK_STATE = JSON.stringify(
  {
    source: 'Order 5512 was shipped on May 3 and delivered on May 6. Refunds require a return within 30 days.',
    llm_answer: 'Your order 5512 was delivered on May 6. I have issued a full refund to your card.',
  },
  null,
  2,
);

/** Same drawing as the model-service's flows.mermaid(), for the editor preview before a run. */
function toMermaid(flow: Flow, path: string[] = []): string {
  const lines = ['flowchart LR'];
  const label = (s: string) => s.replace(/"/g, "'");
  for (const [name, step] of Object.entries(flow.steps)) {
    lines.push(`    ${name}["${name}<br/><small>${label(step.model ?? 'general')}: ${Object.keys(step.questions ?? {}).join(', ')}</small>"]`);
  }
  for (const [name, step] of Object.entries(flow.steps)) {
    for (const edge of step.next ?? []) lines.push(`    ${name} -- "${label(edge.if ?? 'otherwise')}" --> ${edge.to}`);
  }
  if (path.length) {
    lines.push('    classDef taken fill:#1d6b5a,stroke:#3fd0a8,color:#fff');
    lines.push(`    class ${[...new Set(path)].join(',')} taken`);
  }
  return lines.join('\n');
}

let renderCount = 0;

function Diagram({ source }: { source: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let cancelled = false;
    import('mermaid')
      .then(async ({ default: mermaid }) => {
        const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'default', securityLevel: 'strict' });
        const { svg } = await mermaid.render(`flow-${++renderCount}`, source);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = svg;
          setError('');
        }
      })
      .catch((e) => !cancelled && setError(String(e?.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, [source]);
  return (
    <>
      <div ref={ref} className="diagram" />
      {error && <p className="error small">{error}</p>}
    </>
  );
}

export function Flows() {
  const [saved, setSaved] = useState<{ name: string; flow: Flow }[]>([]);
  const [name, setName] = useState('support-triage');
  const [text, setText] = useState(JSON.stringify(EXAMPLE, null, 2));
  const [state, setState] = useState(EXAMPLE_STATE);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);

  const load = () =>
    api<{ flows: { name: string; flow: Flow }[] }>('/flows')
      .then((r) => setSaved(r.flows))
      .catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  let flow: Flow | null = null;
  let parseError = '';
  try {
    flow = JSON.parse(text);
    if (!flow?.steps || typeof flow.steps !== 'object') throw new Error('needs a "steps" object');
  } catch (e) {
    flow = null;
    parseError = (e as Error).message;
  }

  async function run() {
    if (!flow) return;
    setBusy(true);
    setError('');
    setTrace(null);
    try {
      setTrace(await api<Trace>('/flows/try', { method: 'POST', body: JSON.stringify({ flow, state }) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!flow) return;
    setError('');
    try {
      await api(`/flows/${name}`, { method: 'PUT', body: JSON.stringify({ flow }) });
      setNotice(`Saved. Run it from your app: POST /v1/flows/${name}/run {"state": ...}`);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(n: string) {
    if (!confirm(`Delete the flow ${n}?`)) return;
    await api(`/flows/${n}`, { method: 'DELETE' }).catch((e) => setError(e.message));
    load();
  }

  return (
    <>
      <p className="muted small">
        A flow chains dragonflies: each step asks one model (general, a specialist or <code>auto</code>) a few questions,
        and the answers pick the next step. The whole chain runs inside the model-service, one batched decision per step.
      </p>
      {saved.length > 0 && (
        <div className="row wrap">
          <span className="muted small">Saved:</span>
          {saved.map((f) => (
            <span key={f.name} className="pill-button">
              <button
                onClick={() => {
                  const { name: _n, ...rest } = f.flow;
                  setName(f.name);
                  setText(JSON.stringify(rest, null, 2));
                  setTrace(null);
                }}
              >
                {f.name}
              </button>
              <button aria-label={`delete ${f.name}`} onClick={() => remove(f.name)}>
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="split">
        <section className="card">
          <div className="row">
            <h2>Flow</h2>
            <span className="grow" />
            <input className="flow-name" value={name} onChange={(e) => setName(e.target.value.toLowerCase())} aria-label="flow name" />
            <button onClick={save} disabled={!flow}>
              Save
            </button>
          </div>
          <textarea spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} aria-label="flow JSON" />
          {parseError && <p className="error small">JSON: {parseError}</p>}
          <label>
            State (the document the flow decides about)
            <textarea className="short" value={state} onChange={(e) => setState(e.target.value)} aria-label="flow state" />
          </label>
          <div className="row">
            <button className="primary" onClick={run} disabled={busy || !flow}>
              {busy ? 'Running…' : 'Run flow'}
            </button>
            <button onClick={() => setText(JSON.stringify(EXAMPLE, null, 2))}>Reset example</button>
            <button
              onClick={() => {
                setName('llm-output-check');
                setText(JSON.stringify(LLM_CHECK, null, 2));
                setState(LLM_CHECK_STATE);
                setTrace(null);
              }}
            >
              Template: check an LLM answer
            </button>
          </div>
          {error && <p className="error">{error}</p>}
          {notice && <p className="notice">{notice}</p>}
        </section>
        <section className="card">
          <div className="row">
            <h2>{trace ? 'Path taken' : 'Graph'}</h2>
            <span className="grow" />
            {trace && (
              <span className="muted small">
                {trace.path.length} steps · {trace.latency_ms.toFixed(1)} ms in the model-service
              </span>
            )}
          </div>
          {flow && <Diagram source={trace ? trace.mermaid : toMermaid(flow)} />}
          {trace?.steps.map((s) => (
            <div key={s.step} className="answer">
              <div className="row">
                <strong>{s.step}</strong>
                <span className="pill">{s.specialist ?? s.model}</span>
                <span className="grow" />
                <span className="muted small">
                  {s.latency_ms.toFixed(1)} ms{s.next ? ` → ${s.next}` : ' · end'}
                </span>
              </div>
              {Object.entries(s.answers).map(([qid, a]) => (
                <div key={qid} className="small">
                  <strong>{qid}</strong>: {headline(a)} <span className="muted">(confidence {pct(a.confidence)}, tier {a.tier})</span>
                </div>
              ))}
            </div>
          ))}
        </section>
      </div>
    </>
  );
}
