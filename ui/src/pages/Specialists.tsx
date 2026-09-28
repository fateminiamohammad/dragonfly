import { useEffect, useState } from 'react';
import { api } from '../api';
import { pct } from '../format';

interface Specialist {
  name: string;
  description: string;
  tier: 'S' | 'M';
  metrics: { accuracy?: number; ece?: number; test_questions?: number; train_questions?: number; seconds?: number };
  created?: string;
  loaded: boolean;
  served: number;
}

interface Job {
  id: string;
  name: string;
  tier: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  examples?: number;
  step?: number;
  total?: number;
  loss?: number;
  metrics?: Specialist['metrics'];
  error?: string;
  created?: number;
}

type QType = 'choice' | 'noul' | 'score';

const CSV_HELP: Record<QType, string> = {
  noul: 'text,label\n"My card never arrived",yes\n"Thanks, all good",no',
  choice: 'text,label\n"Where is my card?",card_arrival\n"I want my money back",refund',
  score: 'text,label\n"Terrible service",0\n"Great, thanks!",2',
};

export function Specialists() {
  const [specialists, setSpecialists] = useState<Specialist[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    name: '',
    description: '',
    tier: 'S' as 'S' | 'M',
    format: 'csv' as 'csv' | 'jsonl',
    type: 'noul' as QType,
    instructions: '',
    options: '',
    epochs: 3,
  });
  const [data, setData] = useState<{ name: string; text: string } | null>(null);

  const load = () =>
    api<{ specialists: Specialist[]; jobs: Job[] }>('/specialists')
      .then((r) => {
        setSpecialists(r.specialists);
        setJobs(r.jobs);
      })
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  // poll while something is training
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [active]);

  const set = (patch: Partial<typeof form>) => setForm((f) => ({ ...f, ...patch }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setNotice('');
    if (!data) {
      setError('Choose a file with labelled examples.');
      return;
    }
    const body: Record<string, unknown> = {
      name: form.name,
      description: form.description,
      tier: form.tier,
      format: form.format,
      data: data.text,
      epochs: form.epochs,
    };
    if (form.format === 'csv') {
      body.question = {
        type: form.type,
        instructions: form.instructions,
        options: form.type === 'noul' ? undefined : form.options.split(',').map((o) => o.trim()).filter(Boolean),
      };
    }
    setBusy(true);
    try {
      const r = await api<{ examples: number; held_out: number }>('/specialists', { method: 'POST', body: JSON.stringify(body) });
      setNotice(`Training ${form.name} on ${r.examples} examples (${r.held_out} held out to measure accuracy).`);
      load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    if (!confirm(`Delete the specialist ${name}? This removes its files.`)) return;
    try {
      await api(`/specialists/${name}`, { method: 'DELETE' });
      setNotice(`Deleting ${name}…`);
      setTimeout(load, 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <>
      <p className="muted small">
        A swarm of dragonflies: each specialist is a small model trained for one job, sharing the GPU and the endpoint.
        Call one with <code>"model": "&lt;name&gt;"</code>, or <code>"model": "auto"</code> to let Dragonfly pick.
      </p>
      {error && <p className="error">{error}</p>}
      {notice && <p className="notice">{notice}</p>}

      <div className="specialists">
        {specialists.length === 0 && <p className="muted">No specialists yet. Train one below.</p>}
        {specialists.map((s) => (
          <section key={s.name} className="card specialist">
            <div className="row">
              <strong>{s.name}</strong>
              <span className="pill">tier {s.tier}</span>
              {s.loaded && <span className="pill">in memory</span>}
            </div>
            <p className="small">{s.description}</p>
            <div className="specialist-stats">
              <div>
                <span className="muted small">held-out accuracy</span>
                <strong>{s.metrics.accuracy != null ? pct(s.metrics.accuracy) : '–'}</strong>
              </div>
              <div>
                <span className="muted small">test questions</span>
                <strong>{s.metrics.test_questions ?? '–'}</strong>
              </div>
              <div>
                <span className="muted small">requests served</span>
                <strong>{s.served}</strong>
              </div>
            </div>
            <div className="row">
              <span className="muted small">{s.created}</span>
              <span className="grow" />
              <button onClick={() => remove(s.name)}>Delete</button>
            </div>
          </section>
        ))}
      </div>

      {jobs.length > 0 && (
        <section className="card">
          <h2>Training jobs</h2>
          {jobs.map((j) => (
            <div key={j.id} className="bar-row job">
              <span className="bar-label">
                {j.name} <span className="pill">{j.status}</span>
              </span>
              <span className="bar">
                <span style={{ width: `${j.status === 'done' ? 100 : j.total ? ((j.step ?? 0) / j.total) * 100 : 0}%` }} />
              </span>
              <span className="bar-value small">
                {j.status === 'failed'
                  ? j.error
                  : j.status === 'done'
                    ? `accuracy ${j.metrics?.accuracy != null ? pct(j.metrics.accuracy) : '–'}`
                    : j.total
                      ? `step ${j.step}/${j.total} · loss ${j.loss?.toFixed(3)}`
                      : `${j.examples ?? ''} examples`}
              </span>
            </div>
          ))}
        </section>
      )}

      <section className="card">
        <h2>Train a specialist</h2>
        <form className="train-form" onSubmit={submit}>
          <label>
            Name
            <input value={form.name} onChange={(e) => set({ name: e.target.value.toLowerCase() })} placeholder="support-escalation" required />
          </label>
          <label>
            Description (used by "auto" routing)
            <input value={form.description} onChange={(e) => set({ description: e.target.value })} placeholder="customer support tickets: does this need a human?" required />
          </label>
          <label>
            Tier
            <select value={form.tier} onChange={(e) => set({ tier: e.target.value as 'S' | 'M' })}>
              <option value="S">S: fast, trains in minutes beside serving</option>
              <option value="M">M: adapter on the 4B model (needs a free GPU)</option>
            </select>
          </label>
          <label>
            Data format
            <select value={form.format} onChange={(e) => set({ format: e.target.value as 'csv' | 'jsonl' })}>
              <option value="csv">CSV: one question, columns text,label</option>
              <option value="jsonl">JSONL: full requests with labels (training format)</option>
            </select>
          </label>
          {form.format === 'csv' && (
            <>
              <label>
                Question type
                <select value={form.type} onChange={(e) => set({ type: e.target.value as QType })}>
                  <option value="noul">yes / no</option>
                  <option value="choice">choice</option>
                  <option value="score">score (ordered levels)</option>
                </select>
              </label>
              <label>
                Question
                <input value={form.instructions} onChange={(e) => set({ instructions: e.target.value })} placeholder="Does this ticket need a human agent?" required />
              </label>
              {form.type !== 'noul' && (
                <label>
                  Options, comma separated {form.type === 'score' ? '(lowest first)' : ''}
                  <input value={form.options} onChange={(e) => set({ options: e.target.value })} placeholder={form.type === 'score' ? 'negative, neutral, positive' : 'card_arrival, refund, other'} required />
                </label>
              )}
            </>
          )}
          <label>
            Epochs
            <input type="number" min={1} max={10} value={form.epochs} onChange={(e) => set({ epochs: Number(e.target.value) })} />
          </label>
          <label>
            Labelled examples (at least 50; 20% are held out to measure accuracy)
            <input
              type="file"
              accept={form.format === 'csv' ? '.csv,text/csv' : '.jsonl,.json,application/json'}
              onChange={async (e) => {
                const f = e.target.files?.[0];
                setData(f ? { name: f.name, text: await f.text() } : null);
              }}
            />
          </label>
          {form.format === 'csv' && <pre className="ocr small">{CSV_HELP[form.type]}</pre>}
          <div className="row">
            <button className="primary" disabled={busy}>
              {busy ? 'Uploading…' : 'Train'}
            </button>
            {data && <span className="muted small">{data.name}</span>}
          </div>
        </form>
      </section>
    </>
  );
}
