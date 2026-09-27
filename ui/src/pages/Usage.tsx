import { useEffect, useState } from 'react';
import { api, type UsageRow } from '../api';
import { totalsByDay } from '../format';

export function Usage() {
  const [data, setData] = useState<{ days: string[]; rows: UsageRow[] } | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api<{ days: string[]; rows: UsageRow[] }>('/usage?days=30').then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  const totals = totalsByDay(data.days, data.rows);
  const max = Math.max(1, ...totals.map((t) => t.questions));
  const sum = totals.reduce((a, t) => ({ r: a.r + t.requests, q: a.q + t.questions, t: a.t + t.tokens }), { r: 0, q: 0, t: 0 });

  return (
    <>
      <div className="stats">
        <div className="card stat">
          <span className="muted small">Requests, 30 days</span>
          <strong>{sum.r.toLocaleString()}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Questions answered</span>
          <strong>{sum.q.toLocaleString()}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Input tokens</span>
          <strong>{sum.t.toLocaleString()}</strong>
        </div>
      </div>
      <section className="card">
        <h2>Questions per day</h2>
        <div className="columns" role="img" aria-label="questions answered per day, last 30 days">
          {totals.map((t) => (
            <div key={t.day} className="column" title={`${t.day}: ${t.questions} questions, ${t.requests} requests`}>
              <span style={{ height: `${(t.questions / max) * 100}%` }} />
            </div>
          ))}
        </div>
        <div className="row muted small">
          <span>{data.days[0]}</span>
          <span className="grow" />
          <span>{data.days[data.days.length - 1]}</span>
        </div>
      </section>
      <section className="card">
        <h2>By key and day</h2>
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>Key</th>
              <th className="num">Requests</th>
              <th className="num">Questions</th>
              <th className="num">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.length === 0 && (
              <tr>
                <td colSpan={5} className="muted">
                  No usage yet. Usage is recorded for requests made with an API key.
                </td>
              </tr>
            )}
            {[...data.rows].reverse().map((r) => (
              <tr key={r.day + r.keyId}>
                <td>{r.day}</td>
                <td>{r.keyName}</td>
                <td className="num">{r.requests.toLocaleString()}</td>
                <td className="num">{r.questions.toLocaleString()}</td>
                <td className="num">{r.tokens.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
