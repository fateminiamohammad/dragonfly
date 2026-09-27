import { useEffect, useState } from 'react';
import { api, type ApiKey } from '../api';

export function Keys() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [name, setName] = useState('');
  const [created, setCreated] = useState<ApiKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');

  const load = () => api<ApiKey[]>('/keys').then(setKeys).catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    try {
      setCreated(await api<ApiKey>('/keys', { method: 'POST', body: JSON.stringify({ name }) }));
      setCopied(false);
      setName('');
      load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function revoke(k: ApiKey) {
    if (!confirm(`Revoke "${k.name}"? Requests using it will fail within 30 seconds.`)) return;
    await api(`/keys/${k.id}`, { method: 'DELETE' }).catch((e) => setError(e.message));
    load();
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(created!.key!);
      setCopied(true);
    } catch {
      setError('Copy failed: select the key and copy it manually.');
    }
  }

  return (
    <section className="card">
      <h2>API keys</h2>
      <p className="muted">
        Send a key as <code>Authorization: Bearer &lt;key&gt;</code> to <code>/v1/systemone</code>. TypeSafe and Kev
        SDKs work by pointing their base URL at this server.
      </p>
      <form className="row" onSubmit={create}>
        <input placeholder="Key name, e.g. production" value={name} onChange={(e) => setName(e.target.value)} required maxLength={80} />
        <button className="primary">Create key</button>
      </form>
      {created?.key && (
        <div className="notice">
          <strong>Copy this key now. It won't be shown again.</strong>
          <div className="row">
            <code className="secret">{created.key}</code>
            <button onClick={copy}>{copied ? 'Copied' : 'Copy'}</button>
          </div>
        </div>
      )}
      {error && <p className="error">{error}</p>}
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Key</th>
            <th>Created</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {keys.length === 0 && (
            <tr>
              <td colSpan={5} className="muted">
                No keys yet.
              </td>
            </tr>
          )}
          {keys.map((k) => (
            <tr key={k.id} className={k.revokedAt ? 'revoked' : ''}>
              <td>{k.name}</td>
              <td>
                <code>{k.prefix}…</code>
              </td>
              <td>{new Date(k.createdAt).toLocaleDateString()}</td>
              <td>{k.revokedAt ? `revoked ${new Date(k.revokedAt).toLocaleDateString()}` : 'active'}</td>
              <td>{!k.revokedAt && <button onClick={() => revoke(k)}>Revoke</button>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
