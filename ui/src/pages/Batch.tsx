import { useEffect, useState } from 'react';
import { api, type Answer } from '../api';
import { headline, pct } from '../format';

type QType = 'choice' | 'noul' | 'score';

interface Job {
  id: string;
  status: 'running' | 'done' | 'failed';
  total: number;
  done: number;
  error?: string;
  results?: ({ answers: Record<string, Answer> } | { error: string })[];
  stats?: { requests: number; errors: number; questions: number; seconds: number; decisions_per_s: number };
}

/** RFC 4180 CSV: quoted fields, doubled quotes, commas and newlines inside quotes. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      if (row.some((f) => f.trim())) rows.push(row);
      row = [];
      field = '';
    } else field += c;
  }
  row.push(field);
  if (row.some((f) => f.trim())) rows.push(row);
  return rows;
}

const csvCell = (v: string) => (/[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

/** The question the batch asks every row, in /v1/systemone form. */
export function buildQuestion(type: QType, instructions: string, options: string) {
  const opts = options.split(',').map((o) => o.trim()).filter(Boolean);
  if (type === 'noul') return { type, instructions };
  if (type === 'choice') return { type, instructions, criteria: Object.fromEntries(opts.map((o) => [o, null])) };
  return { type, instructions, criteria: opts };
}

export function Batch() {
  const [table, setTable] = useState<{ name: string; header: string[]; rows: string[][] } | null>(null);
  const [type, setType] = useState<QType>('noul');
  const [instructions, setInstructions] = useState('');
  const [options, setOptions] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!job || job.status !== 'running') return;
    const t = setInterval(async () => {
      try {
        // progress only while running (a 10,000-row result set is not re-sent every second); results once at the end
        const state = await api<Job>(`/batch/jobs/${job.id}?results=false`);
        setJob(state.status === 'running' ? state : await api<Job>(`/batch/jobs/${job.id}`));
      } catch (e) {
        setError((e as Error).message);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [job]);

  async function start() {
    setError('');
    if (!table) return setError('Choose a CSV file first.');
    if (!instructions.trim()) return setError('Write the question to ask every row.');
    const question = buildQuestion(type, instructions, options);
    const requests = table.rows.map((r) => ({
      state: Object.fromEntries(table.header.map((h, i) => [h, r[i] ?? ''])),
      questions: { answer: question },
    }));
    try {
      const started = await api<{ id: string; total: number }>('/batch/jobs', { method: 'POST', body: JSON.stringify({ requests }) });
      setJob({ id: started.id, status: 'running', total: started.total, done: 0 });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function download() {
    if (!table || !job?.results) return;
    const lines = [[...table.header, 'answer', 'confidence', 'tier'].map(csvCell).join(',')];
    table.rows.forEach((r, i) => {
      const res = job.results![i];
      const a = res && 'answers' in res ? res.answers.answer : null;
      lines.push([...r, a ? headline(a) : 'error' in (res ?? {}) ? `error: ${(res as { error: string }).error}` : '', a ? a.confidence.toFixed(4) : '', a?.tier ?? ''].map((v) => csvCell(String(v))).join(','));
    });
    const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = (table.name.replace(/\.csv$/i, '') || 'batch') + '-decided.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <p className="muted small">
        Map a decision over a whole table: every row becomes the state of one request, and the same question is asked of
        each. Batches run at low priority, so live traffic is never delayed. From code, call <code>POST /v1/batch</code>.
      </p>
      <div className="split">
        <section className="card">
          <h2>Data and question</h2>
          <div className="train-form">
            <label>
              CSV file (header row; up to 10,000 rows)
              <input
                type="file"
                accept=".csv,text/csv"
                aria-label="batch csv"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  const rows = parseCsv(await f.text());
                  if (rows.length < 2) return setError('The CSV needs a header row and at least one data row.');
                  setTable({ name: f.name, header: rows[0], rows: rows.slice(1, 10001) });
                  setJob(null);
                }}
              />
            </label>
            {table && (
              <span className="muted small">
                {table.rows.length} rows · columns: {table.header.join(', ')}
              </span>
            )}
            <label>
              Question type
              <select value={type} onChange={(e) => setType(e.target.value as QType)}>
                <option value="noul">yes / no</option>
                <option value="choice">choice</option>
                <option value="score">score (ordered levels)</option>
              </select>
            </label>
            <label>
              Question
              <input value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="Does this ticket need a human agent?" aria-label="batch question" />
            </label>
            {type !== 'noul' && (
              <label>
                Options, comma separated {type === 'score' ? '(lowest first)' : ''}
                <input value={options} onChange={(e) => setOptions(e.target.value)} placeholder="refund, delivery, other" aria-label="batch options" />
              </label>
            )}
            <div className="row">
              <button className="primary" onClick={start} disabled={job?.status === 'running'}>
                {job?.status === 'running' ? 'Running…' : 'Run batch'}
              </button>
            </div>
            {error && <p className="error">{error}</p>}
          </div>
        </section>
        <section className="card">
          <div className="row">
            <h2>Results</h2>
            <span className="grow" />
            {job?.status === 'done' && <button onClick={download}>Download CSV</button>}
          </div>
          {!job && <p className="muted">Run a batch to see progress and answers.</p>}
          {job && (
            <div className="bar-row job">
              <span className="bar-label">
                <span className="pill">{job.status}</span>
              </span>
              <span className="bar">
                <span style={{ width: `${job.total ? (job.done / job.total) * 100 : 0}%` }} />
              </span>
              <span className="bar-value small">
                {job.done} / {job.total}
              </span>
            </div>
          )}
          {job?.stats && (
            <p className="muted small">
              {job.stats.questions} decisions in {job.stats.seconds.toFixed(2)} s · {job.stats.decisions_per_s} decisions/s · {job.stats.errors} errors
            </p>
          )}
          {job?.error && <p className="error">{job.error}</p>}
          {table && job?.results && (
            <table className="results">
              <thead>
                <tr>
                  <th>#</th>
                  <th>{table.header[0]}</th>
                  <th>answer</th>
                  <th>confidence</th>
                </tr>
              </thead>
              <tbody>
                {job.results.slice(0, 50).map((res, i) => {
                  const a = 'answers' in res ? res.answers.answer : null;
                  return (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="cell-text">{table.rows[i]?.[0]}</td>
                      <td>{a ? headline(a) : 'error' in res ? res.error : ''}</td>
                      <td>{a ? pct(a.confidence) : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </>
  );
}
