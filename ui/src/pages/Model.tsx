import { useEffect, useState } from 'react';
import { api } from '../api';

interface Tier {
  tier: string;
  backbone: string;
  trained: boolean;
  temperature: number;
  device: string;
}

interface ModelCard extends Partial<Tier> {
  name: string;
  trained: boolean;
  device: string;
  threshold?: number;
  escalation_rate?: number | null;
  small?: Tier;
  large?: Tier;
  plugins: { name: string; version: string; api_version: string }[];
  cache: { size: number; entries: number; hits: number; misses: number };
  batches: { count: number; requests: number; queued: number };
}

function TierRow({ label, t }: { label: string; t: Tier }) {
  return (
    <tr>
      <td>{label}</td>
      <td>
        <code>{t.backbone}</code>
      </td>
      <td>{t.trained ? 'trained' : <span className="error">untrained</span>}</td>
      <td className="num">{t.temperature.toFixed(2)}</td>
      <td>{t.device}</td>
    </tr>
  );
}

export function Model() {
  const [card, setCard] = useState<ModelCard | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api<{ models: ModelCard[] }>('/model')
      .then((r) => setCard(r.models[0]))
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="error">Model service: {error}</p>;
  if (!card) return <p className="muted">Loading…</p>;
  const tiers: [string, Tier][] = card.small && card.large ? [['S (fast)', card.small], ['M (quality)', card.large]] : [[card.tier ?? '', card as Tier]];
  const hitRate = card.cache.hits + card.cache.misses ? card.cache.hits / (card.cache.hits + card.cache.misses) : 0;

  return (
    <>
      {!card.trained && (
        <div className="notice">
          <strong>This server runs an untrained head.</strong> Answers are meaningless until you train a checkpoint and
          point <code>DRAGONFLY_CHECKPOINT</code> at it.
        </div>
      )}
      <section className="card">
        <h2>Model</h2>
        <table>
          <thead>
            <tr>
              <th>Tier</th>
              <th>Backbone</th>
              <th>Status</th>
              <th className="num">Temperature</th>
              <th>Device</th>
            </tr>
          </thead>
          <tbody>
            {tiers.map(([label, t]) => (
              <TierRow key={label} label={label} t={t} />
            ))}
          </tbody>
        </table>
        {card.threshold !== undefined && (
          <p className="muted">
            Cascade: questions below {Math.round(card.threshold * 100)}% confidence go to tier M. Escalated so far:{' '}
            {card.escalation_rate == null ? 'n/a' : `${(card.escalation_rate * 100).toFixed(1)}%`}.
          </p>
        )}
      </section>
      <div className="stats">
        <div className="card stat">
          <span className="muted small">Requests served</span>
          <strong>{card.batches.requests.toLocaleString()}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Avg batch size</span>
          <strong>{card.batches.count ? (card.batches.requests / card.batches.count).toFixed(1) : '-'}</strong>
        </div>
        <div className="card stat">
          <span className="muted small">Cache hit rate</span>
          <strong>{card.cache.size ? `${(hitRate * 100).toFixed(1)}%` : 'off'}</strong>
        </div>
      </div>
      <section className="card">
        <h2>Plugins</h2>
        {card.plugins.length === 0 ? (
          <p className="muted">
            No plugins loaded. Install a plugin package and list it in <code>DRAGONFLY_PLUGINS</code>.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Plugin API</th>
              </tr>
            </thead>
            <tbody>
              {card.plugins.map((p) => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td>{p.version}</td>
                  <td>{p.api_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </>
  );
}
